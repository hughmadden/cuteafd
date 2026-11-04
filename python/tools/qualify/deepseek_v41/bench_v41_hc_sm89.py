#!/usr/bin/env python3
"""SM89 V4.1 mHC boundary probe: today's native begin and post, SparkInfer's
lagged begin exported for sm_89, and native/tests/v41_hc_sm89_probe.cu.

Run in the coordinator dev image on one SM89 GPU, under its hardware lock.
Inputs:
  --native  the SM89 coordinator libcuteafd_native.so (V4.1 FP8 bundle with hc_project)
  --lagged  a lagged probe library: on the SM89 GPU,
              B12X_COMPILE_DISK_CACHE=0 python3 -c "import sys; sys.path[:0] = ['python/tools/lib']; \\
                import _pinned_sparkinfer; from b12x.norm.mhc._v41_lagged_aot import compile_v41_lagged_aot as c; \\
                c().export_to_c('OUT', 'v41_hc_lagged', 'cuteafd_v41_hc_lagged')"
            then g++ -O2 -shared -fPIC -IOUT native/tests/v41_hc_lagged_probe.cc OUT/v41_hc_lagged.o \\
                -L$(python3 -m cutlass.cute.export.aot_config --libdir) -lcute_dsl_runtime -lcudart
  --probe   native/tests/v41_hc_sm89_probe.cu built as its header says.

Numerics on identical inputs: today's decode arithmetic for every row (the
native begin over chunks of <= 16 rows, then today's post) is the bit
reference; the lagged begin is also checked against SparkInfer's lagged
oracle (tests/norm/test_mhc_lagged.py tolerances) and an FP64 reference.
Timing follows the decode-profile method: one CUDA graph holds R calls over R
rotated fn tensors (cold weights, warm residual), warmed, median of three
samples of back-to-back replays; time per call = replay / R. --emulate-sms N
also times with (SMs - N) SMs held by a resident kernel that fills their
register files (a 114-SM RTX 4090 D on a 128-SM RTX 4090).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[2] / _d) for _d in ("lib",)]  # sibling tool dirs
import argparse
import ctypes as C
import json
import math
import statistics
import time
from pathlib import Path

import torch
import _pinned_sparkinfer  # noqa: F401  # verifies and prepends the pinned fork (oracle imports b12x)

P, I32, U64, F32 = C.c_void_p, C.c_int32, C.c_uint64, C.c_float
dev = torch.device("cuda")
bf16 = torch.bfloat16
LAYERS = 40
# Best measured tiling per row count (W*100 + PW*10, rows per CTA) on an RTX 4090.
PLAN = {1: (410, 1), 6: (420, 2), 8: (420, 2), 16: (420, 4), 24: (420, 6), 48: (420, 12), 64: (420, 8), 96: (420, 8)}


def plan(rows):
    return PLAN.get(rows, (420, min(12, max(2, -(-rows // 4)))))


def ok(status, what):
    if status != 0:
        raise RuntimeError(f"{what}: CUDA status {status}")


def bind(lib, name, *args):
    f = getattr(lib, name); f.argtypes = list(args); f.restype = I32
    return f


class Libs:
    def __init__(self, native, lagged, probe):
        n, lg, pr = C.CDLL(str(native)), C.CDLL(str(lagged)), C.CDLL(str(probe))
        self.project_init = bind(n, "cuteafd_v41_hc_project_initialize")
        self.mixes = bind(n, "cuteafd_v41_hc_mixes_workspace", P, P, P, P, P, P, P, P, U64, I32, P)
        self.pre = bind(n, "cuteafd_v41_hc_pre", P, P, P, I32, P)
        self.post = bind(n, "cuteafd_v41_hc_post", P, P, P, P, P, I32, P)
        self.rmsnorm = bind(n, "cuteafd_cuda_ds4_rmsnorm_bf16_rne_async", P, P, P, I32, I32, F32, P)
        self.lagged_init = bind(lg, "initialize")
        self.lagged = bind(lg, "launch", P, P, P, P, P, P, P, P, P, P, P, I32, P)
        self.begin = bind(pr, "hc_begin_sm89", P, P, P, P, P, P, P, P, P, P, P, U64, I32, I32, I32, I32, P, P)
        self.post_finish = bind(pr, "hc_post_finish_sm89", P, P, P, P, P, P, P, P, P, I32, P)
        self.occupy_start = bind(pr, "occupy_start", I32)
        self.occupy_stop = bind(pr, "occupy_stop")


class Case:
    """One boundary's planes (the daemon's HcStorage) with R rotated fn tensors."""

    def __init__(self, L, rows, rotations):
        self.L, self.rows = L, rows
        g = torch.Generator(device=dev); g.manual_seed(731 + rows)
        self.residual = torch.randn(rows, 4, 5120, device=dev, generator=g).to(bf16)
        self.fns = [torch.randn(24 * 20480, device=dev, generator=g) * 0.01 for _ in range(rotations)]
        self.scale = torch.tensor([0.1, 0.1, 0.1], device=dev)
        self.base = torch.randn(24, device=dev, generator=g)
        self.incoming = torch.rand(rows, 4, device=dev, generator=g)
        self.norm_w = torch.linspace(0.5, 1.5, 5120, device=dev).to(bf16)
        self.sub = torch.randn(rows, 5120, device=dev, generator=g).to(bf16)
        self.scratch = torch.zeros(rows * 10240 // 4, device=dev)  # the collapsed plane
        self.counters = torch.zeros(rows, dtype=torch.int32, device=dev)
        self.collapsed = torch.empty(rows, 5120, dtype=bf16, device=dev)
        self.outs = {}

    def planes(self, name):
        if name not in self.outs:
            r = self.rows
            nan = float("nan")
            self.outs[name] = dict(pre=torch.full((r, 4), nan, device=dev), post=torch.full((r, 4), nan, device=dev),
                                   comb=torch.full((r, 4, 4), nan, device=dev),
                                   y=torch.full((r, 5120), nan, dtype=bf16, device=dev),
                                   out=torch.full((r, 4, 5120), nan, dtype=bf16, device=dev))
        return self.outs[name]

    def native_begin(self, rep, name="native", lo=0, hi=None):
        L, fn, o = self.L, self.fns[rep % len(self.fns)], self.planes(name)
        hi = self.rows if hi is None else hi

        def f(s):
            res = self.residual[lo:hi]
            ok(L.mixes(res.data_ptr(), fn.data_ptr(), self.scale.data_ptr(), self.base.data_ptr(), o["pre"][lo:hi].data_ptr(),
                       o["post"][lo:hi].data_ptr(), o["comb"][lo:hi].data_ptr(), self.scratch.data_ptr(),
                       self.scratch.numel() * 4, hi - lo, s), "mixes")
            ok(L.pre(res.data_ptr(), self.incoming[lo:hi].data_ptr(), self.collapsed[lo:hi].data_ptr(), hi - lo, s), "pre")
            ok(L.rmsnorm(self.collapsed[lo:hi].data_ptr(), self.norm_w.data_ptr(), o["y"][lo:hi].data_ptr(), hi - lo,
                         5120, 1e-20, s), "rmsnorm")
        return f

    def native_post(self, name="native"):
        L, o = self.L, self.planes(name)
        return lambda s: ok(L.post(self.sub.data_ptr(), self.residual.data_ptr(), o["post"].data_ptr(),
                                   o["comb"].data_ptr(), o["out"].data_ptr(), self.rows, s), "post")

    def lagged(self, rep, name="lagged"):
        L, fn, o = self.L, self.fns[rep % len(self.fns)], self.planes(name)
        return lambda s: ok(L.lagged(self.residual.data_ptr(), fn.data_ptr(), self.scale.data_ptr(), self.base.data_ptr(),
                                     self.incoming.data_ptr(), self.norm_w.data_ptr(), o["pre"].data_ptr(),
                                     o["post"].data_ptr(), o["comb"].data_ptr(), o["y"].data_ptr(),
                                     self.scratch.data_ptr(), self.rows, s), "lagged")

    def fused(self, rep, cfg, rpc, mode, name="fused"):
        L, fn, o = self.L, self.fns[rep % len(self.fns)], self.planes(name)
        return lambda s: ok(L.begin(self.residual.data_ptr(), fn.data_ptr(), self.scale.data_ptr(), self.base.data_ptr(),
                                    self.incoming.data_ptr(), self.norm_w.data_ptr(), o["pre"].data_ptr(), o["post"].data_ptr(),
                                    o["comb"].data_ptr(), o["y"].data_ptr(), self.scratch.data_ptr(), self.scratch.numel() * 4,
                                    self.rows, cfg, rpc, mode, self.counters.data_ptr(), s), f"begin {cfg} {rpc} {mode}")

    def post_finish(self, name="folded"):
        L, o = self.L, self.planes(name)
        return lambda s: ok(L.post_finish(self.sub.data_ptr(), self.residual.data_ptr(), self.scratch.data_ptr(),
                                          self.scale.data_ptr(), self.base.data_ptr(), o["pre"].data_ptr(),
                                          o["post"].data_ptr(), o["comb"].data_ptr(), o["out"].data_ptr(),
                                          self.rows, s), "post finish")


def run_once(*calls):
    s = torch.cuda.current_stream().cuda_stream
    for f in calls:
        f(s)
    torch.cuda.synchronize()


def capture(instances):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for f in instances:
            f(s.cuda_stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s, capture_error_mode="thread_local"):
        for f in instances:
            f(s.cuda_stream)
    g.replay(); torch.cuda.synchronize()
    return g, len(instances)


def time_graph(graph, min_ms=40.0, samples=3):
    """Event-synchronized only, so it is safe beside the SM-holding kernel."""
    g, count = graph
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        g.replay()
        e0.record(); g.replay(); e1.record(); e1.synchronize()
        n = max(3, min(2000, math.ceil(min_ms / max(e0.elapsed_time(e1), 1e-3))))
        times = []
        for _ in range(samples):
            e0.record()
            for _ in range(n):
                g.replay()
            e1.record(); e1.synchronize()
            times.append(e0.elapsed_time(e1) * 1000.0 / (n * count))
    return round(statistics.median(times), 3)


def compare(a, b, keys=("pre", "post", "comb", "y")):
    out = {}
    for key in keys:
        x, y = a[key], b[key]
        bits = torch.int16 if x.dtype == bf16 else torch.int32
        same = x.view(bits) == y.view(bits)
        out[key] = dict(equal=int(same.sum()), n=same.numel(), max_abs=float((x.double() - y.double()).abs().max()))
    out["bit_identical"] = all(v["equal"] == v["n"] for v in out.values())
    return out


def reference64(case):
    x = case.residual.double().flatten(1); fn = case.fns[0].double().view(24, 20480)
    mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20)
    sc, b = case.scale.double(), case.base.double()
    pre = torch.sigmoid(mixes[:, :4] * sc[0] + b[:4]) + 1e-6
    post = 2 * torch.sigmoid(mixes[:, 4:8] * sc[1] + b[4:8])
    comb = torch.softmax(mixes[:, 8:].view(-1, 4, 4) * sc[2] + b[8:].view(4, 4), dim=-1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    y = (case.incoming.double().unsqueeze(-1) * case.residual.double()).sum(1)
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-20) * case.norm_w.double()
    return dict(pre=pre, post=post, comb=comb, y=y)


def oracle_pass(case, a):
    """SparkInfer's _lagged_reference with its tolerances (rtol 2e-5; atol 4e-5 FP32, 0.008 BF16)."""
    from b12x.testing.mhc import pre_reference
    import torch.nn.functional as F
    flat = case.residual.flatten(1).float(); fn = case.fns[0].view(24, 20480)
    projection = F.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-20)
    want = dict(pre=torch.sigmoid(projection[:, :4] * case.scale[0] + case.base[:4]) + 1e-6)
    _, want["post"], want["comb"] = pre_reference(case.residual, fn, case.scale, case.base, rms_eps=1e-20, hc_eps=1e-6,
                                                  sinkhorn_iters=20)
    y = (case.incoming.unsqueeze(-1) * case.residual.float()).sum(1).bfloat16().float()
    want["y"] = (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-20) * case.norm_w.float()).bfloat16()
    result = {}
    for key in want:
        try:
            torch.testing.assert_close(a[key], want[key], rtol=2e-5, atol=0.008 if a[key].dtype == bf16 else 4e-5)
            result[key] = True
        except AssertionError:
            result[key] = False
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--lagged", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--rows", default="1,6,8,16,24,48,64,96")
    ap.add_argument("--rotations", type=int, default=64)
    ap.add_argument("--emulate-sms", default="", help="comma list of smaller SM counts to time as well, e.g. 114")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    L = Libs(args.native, args.lagged, args.probe)
    ok(L.project_init(), "hc project initialize"); ok(L.lagged_init(), "lagged initialize")
    props = torch.cuda.get_device_properties(0)
    full = props.multi_processor_count
    counts = [full] + [int(v) for v in args.emulate_sms.split(",") if v]
    report = {"device": props.name, "sms": full, "layers": LAYERS, "cases": []}
    for rows in [int(r) for r in args.rows.split(",")]:
        cfg, rpc = plan(rows)
        case = Case(L, rows, args.rotations); R = args.rotations
        o = case.outs
        # Bit reference: today's decode arithmetic per row, then today's post.
        for lo in range(0, rows, 16):
            run_once(case.native_begin(0, "decode", lo, min(rows, lo + 16)))
        run_once(case.native_post("decode"))
        run_once(case.native_begin(0), case.lagged(0), case.fused(0, cfg, rpc, 0, "fused"), case.fused(0, cfg, rpc, 6, "fused2"))
        run_once(case.fused(0, cfg, rpc, 1, "folded"), case.post_finish())
        row = {"rows": rows, "tiling": [cfg, rpc], "numerics": {
            "today_vs_decode": compare(o["native"], o["decode"]),
            "lagged_vs_decode": compare(o["lagged"], o["decode"]),
            "fused_vs_decode": compare(o["fused"], o["decode"]),
            "fused2_vs_decode": compare(o["fused2"], o["decode"]),
            "folded_vs_decode": compare(o["folded"], o["decode"], ("pre", "post", "comb", "out")),
            "lagged_oracle": oracle_pass(case, o["lagged"]), "decode_oracle": oracle_pass(case, o["decode"])}}
        ref = reference64(case)
        row["numerics"]["fp64_max_abs"] = {name: {k: float((o[name][k].double() - ref[k]).abs().max()) for k in ref}
                                           for name in ("decode", "lagged")}
        graphs = {
            "native_begin": capture([case.native_begin(i) for i in range(R)]),
            "post": capture([case.native_post() for _ in range(R)]),
            "lagged_begin": capture([case.lagged(i) for i in range(R)]),
            "fused_begin": capture([case.fused(i, cfg, rpc, 6) for i in range(R)]),
            "native_block": capture([f for i in range(R) for f in (case.native_begin(i), case.native_post())]),
            "fused_block": capture([f for i in range(R) for f in (case.fused(i, cfg, rpc, 6), case.native_post())]),
            "folded_block": capture([f for i in range(R) for f in (case.fused(i, cfg, rpc, 1, "folded"), case.post_finish())]),
        }
        row["us"] = {}
        for sms in counts:
            if sms < full:
                ok(L.occupy_start(full - sms), "occupy"); time.sleep(0.05)
            try:
                t = {k: time_graph(g) for k, g in graphs.items()}
            finally:
                if sms < full:
                    ok(L.occupy_stop(), "release")
            for k in ("native_block", "fused_block", "folded_block"):
                t[k] = round(t[k] * 2, 3)  # the graphs hold 2R calls
            row["us"][str(sms)] = t
        del graphs
        report["cases"].append(row)
        print(json.dumps({"rows": rows, "us": row["us"],
                          "bit_identical": {k: v["bit_identical"] for k, v in row["numerics"].items() if "vs" in k}}), flush=True)
        args.output.write_text(json.dumps(report, indent=1) + "\n")
        del case
        torch.cuda.empty_cache()
    for sms in counts:
        print(f"\n{sms} SMs, us per block (begin + post); per layer = 2 blocks; per step = lanes x {LAYERS} layers")
        for row in report["cases"]:
            t = row["us"][str(sms)]
            print(f"  rows {row['rows']:3d}: today {t['native_block']:6.2f}  lagged {t['lagged_begin'] + t['post']:6.2f}  "
                  f"fused {t['fused_block']:6.2f}  folded {t['folded_block']:6.2f}")


if __name__ == "__main__":
    main()
