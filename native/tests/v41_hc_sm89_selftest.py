"""Check the SM89 V4.1 mHC boundary (cuteafd_v41_hc_begin + cuteafd_v41_hc_post_finish)
against today's SM89 decode arithmetic, bit for bit.

Loads an sm_89 native library and, for every row count 1-96 and the 128-row
cap, runs the boundary on two input sets (unit-scale residual with random
incoming coefficients; a small residual with the one-hot first-stage
coefficients). The reference is today's chain at decode widths: mixes over
chunks of at most 16 rows (the hc_project AOT and its finish), pre and the
5120-wide RMSNorm, then post on those coefficients. Every output must be
bit-identical: the normalized collapse, next pre, post, comb and the post
output. Above 16 rows each row must also equal the boundary run on that row
alone (verify rows equal decode rows). A CUDA graph replay must reproduce the
eager outputs, and malformed calls must be refused.

Usage: v41_hc_sm89_selftest.py NATIVE_LIB [--rows 1-96,128] [--output JSON]
Run on an SM89 GPU in the coordinator dev image.
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import torch

P, I32, U64, F32 = C.c_void_p, C.c_int32, C.c_uint64, C.c_float
bf16 = torch.bfloat16
dev = torch.device("cuda")


def bind(lib, name, *args):
    f = getattr(lib, name)
    f.argtypes = list(args)
    f.restype = I32
    return f


def parse_rows(text):
    rows = []
    for item in text.split(","):
        lo, _, hi = item.partition("-")
        rows += list(range(int(lo), int(hi or lo) + 1))
    return rows


class Boundary:
    """One boundary's planes for `rows`, as the daemon's HcStorage lays them out."""

    def __init__(self, lib, rows, seed, small):
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        self.lib, self.rows = lib, rows
        self.residual = (torch.randn(rows, 4, 5120, device=dev, generator=g) * (0.01 if small else 1.0)).to(bf16)
        self.fn = torch.randn(24 * 20480, device=dev, generator=g) * 0.01
        self.scale = torch.rand(3, device=dev, generator=g) * 0.2
        self.base = torch.randn(24, device=dev, generator=g)
        if small:
            self.incoming = torch.zeros(rows, 4, device=dev)
            self.incoming[:, 0] = 1
        else:
            self.incoming = torch.rand(rows, 4, device=dev, generator=g)
        self.norm = (torch.rand(5120, device=dev, generator=g) + 0.5).to(bf16)
        self.sublayer = torch.randn(rows, 5120, device=dev, generator=g).to(bf16)
        self.scratch = torch.zeros(rows * 10240 // 4, device=dev)  # the collapsed plane
        self.collapsed = torch.empty(rows, 5120, dtype=bf16, device=dev)

    def outputs(self):
        r = self.rows
        nan = float("nan")
        return dict(pre=torch.full((r, 4), nan, device=dev), post=torch.full((r, 4), nan, device=dev),
                    comb=torch.full((r, 4, 4), nan, device=dev),
                    y=torch.full((r, 5120), nan, dtype=bf16, device=dev),
                    out=torch.full((r, 4, 5120), nan, dtype=bf16, device=dev))

    def chain(self, o, stream):
        """Today's decode arithmetic: <= 16-row mixes chunks, pre, RMSNorm, post."""
        L = self.lib
        for lo in range(0, self.rows, 16):
            hi = min(self.rows, lo + 16)
            check(L.mixes(self.residual[lo:hi].data_ptr(), self.fn.data_ptr(), self.scale.data_ptr(),
                          self.base.data_ptr(), o["pre"][lo:hi].data_ptr(), o["post"][lo:hi].data_ptr(),
                          o["comb"][lo:hi].data_ptr(), self.scratch.data_ptr(), self.scratch.numel() * 4,
                          hi - lo, stream), "mixes")
        check(L.pre(self.residual.data_ptr(), self.incoming.data_ptr(), self.collapsed.data_ptr(), self.rows, stream), "pre")
        check(L.rmsnorm(self.collapsed.data_ptr(), self.norm.data_ptr(), o["y"].data_ptr(), self.rows, 5120, 1e-20,
                        stream), "rmsnorm")
        check(L.post(self.sublayer.data_ptr(), self.residual.data_ptr(), o["post"].data_ptr(), o["comb"].data_ptr(),
                     o["out"].data_ptr(), self.rows, stream), "post")

    def boundary(self, o, stream, lo=0, hi=None):
        L, hi = self.lib, self.rows if hi is None else hi
        rows = hi - lo
        scratch = self.scratch[lo * 2560:]
        check(L.begin(self.residual[lo:hi].data_ptr(), self.fn.data_ptr(), self.scale.data_ptr(), self.base.data_ptr(),
                      self.incoming[lo:hi].data_ptr(), self.norm.data_ptr(), o["pre"][lo:hi].data_ptr(),
                      o["post"][lo:hi].data_ptr(), o["comb"][lo:hi].data_ptr(), o["y"][lo:hi].data_ptr(),
                      scratch.data_ptr(), scratch.numel() * 4, rows, stream), "begin")
        check(L.post_finish(self.sublayer[lo:hi].data_ptr(), self.residual[lo:hi].data_ptr(), scratch.data_ptr(),
                            scratch.numel() * 4, self.scale.data_ptr(), self.base.data_ptr(), o["pre"][lo:hi].data_ptr(),
                            o["post"][lo:hi].data_ptr(), o["comb"][lo:hi].data_ptr(), o["out"][lo:hi].data_ptr(),
                            rows, stream), "post finish")


def check(status, what):
    if status != 0:
        raise RuntimeError(f"{what}: CUDA status {status}")


def differences(a, b):
    """Per output, the count of elements whose bits differ."""
    result = {}
    for key in a:
        bits = torch.int16 if a[key].dtype == bf16 else torch.int32
        result[key] = int((a[key].view(bits) != b[key].view(bits)).sum())
    return result


def refusals(L, case):
    """Malformed begin and post-finish calls return an error and launch nothing."""
    o = case.outputs()
    stream = torch.cuda.current_stream().cuda_stream
    s = case.scratch
    good = [case.residual.data_ptr(), case.fn.data_ptr(), case.scale.data_ptr(), case.base.data_ptr(),
            case.incoming.data_ptr(), case.norm.data_ptr(), o["pre"].data_ptr(), o["post"].data_ptr(),
            o["comb"].data_ptr(), o["y"].data_ptr(), s.data_ptr()]
    begin_cases = {"rows 0": (good, s.numel() * 4, 0), "rows 129": (good, 129 * 1536, 129),
                   "short scratch": (good, case.rows * 1536 - 1, case.rows),
                   "misaligned residual": ([good[0] + 2] + good[1:], s.numel() * 4, case.rows),
                   "scratch over normalized": (good[:10] + [good[9]], s.numel() * 4, case.rows),
                   "normalized over residual": (good[:9] + [good[0]] + good[10:], s.numel() * 4, case.rows)}
    refused = {}
    for name, (pointers, size, rows) in begin_cases.items():
        refused["begin " + name] = L.begin(*pointers, size, rows, stream) != 0
    finish = [case.sublayer.data_ptr(), case.residual.data_ptr(), s.data_ptr(), s.numel() * 4, case.scale.data_ptr(),
              case.base.data_ptr(), o["pre"].data_ptr(), o["post"].data_ptr(), o["comb"].data_ptr(), o["out"].data_ptr()]
    finish_cases = {"rows 0": (finish, 0), "rows 129": (finish, 129),
                    "output over residual": (finish[:9] + [finish[1]], case.rows),
                    "comb over post": (finish[:8] + [finish[7]] + finish[9:], case.rows),
                    "short scratch": (finish[:3] + [case.rows * 1536 - 1] + finish[4:], case.rows)}
    for name, (args, rows) in finish_cases.items():
        refused["post finish " + name] = L.post_finish(*args, rows, stream) != 0
    torch.cuda.synchronize()
    return refused


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("native_lib", type=Path)
    ap.add_argument("--rows", default="1-96,128")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    lib = C.CDLL(str(args.native_lib))

    class L:
        project_init = bind(lib, "cuteafd_v41_hc_project_initialize")
        sm89_init = bind(lib, "cuteafd_v41_hc_sm89_initialize")
        mixes = bind(lib, "cuteafd_v41_hc_mixes_workspace", P, P, P, P, P, P, P, P, U64, I32, P)
        pre = bind(lib, "cuteafd_v41_hc_pre", P, P, P, I32, P)
        post = bind(lib, "cuteafd_v41_hc_post", P, P, P, P, P, I32, P)
        rmsnorm = bind(lib, "cuteafd_cuda_ds4_rmsnorm_bf16_rne_async", P, P, P, I32, I32, F32, P)
        begin = bind(lib, "cuteafd_v41_hc_begin", P, P, P, P, P, P, P, P, P, P, P, U64, I32, P)
        post_finish = bind(lib, "cuteafd_v41_hc_post_finish", P, P, P, U64, P, P, P, P, P, P, I32, P)

    props = torch.cuda.get_device_properties(0)
    assert (props.major, props.minor) == (8, 9), f"needs an SM89 GPU, found sm_{props.major}{props.minor}"
    check(L.project_init(), "hc project initialize")
    check(L.sm89_init(), "SM89 mHC initialize")
    report = {"device": props.name, "sms": props.multi_processor_count, "cases": [], "failures": []}
    stream = torch.cuda.current_stream().cuda_stream
    for rows in parse_rows(args.rows):
        for small in (False, True):
            case = Boundary(L, rows, 7 + rows * 2 + small, small)
            ref, got = case.outputs(), case.outputs()
            case.chain(ref, stream)
            case.boundary(got, stream)
            torch.cuda.synchronize()
            diff = differences(got, ref)
            row_diff = {}
            if rows > 16:  # each verify row against the boundary on that row alone
                alone = case.outputs()
                for row in range(rows):
                    case.boundary(alone, stream, row, row + 1)
                torch.cuda.synchronize()
                row_diff = differences(got, alone)
            entry = {"rows": rows, "small": small, "differing": diff, "differing_vs_one_row": row_diff}
            report["cases"].append(entry)
            if any(diff.values()) or any(row_diff.values()):
                report["failures"].append(entry)
                print("FAIL", json.dumps(entry), flush=True)
    # Graph replay reproduces the eager boundary (initialization precedes capture).
    case = Boundary(L, 48, 991, False)
    eager, replay = case.outputs(), case.outputs()
    case.boundary(eager, stream)
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        case.boundary(replay, capture_stream.cuda_stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture_stream):
            case.boundary(replay, capture_stream.cuda_stream)
    for value in replay.values():
        value.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    report["graph_replay_differing"] = differences(replay, eager)
    report["refused"] = refusals(L, Boundary(L, 6, 5, False))
    ok = (not report["failures"] and not any(report["graph_replay_differing"].values())
          and all(report["refused"].values()))
    report["passed"] = ok
    summary = {"passed": ok, "cases": len(report["cases"]), "failures": len(report["failures"]),
               "graph_replay_differing": report["graph_replay_differing"],
               "not_refused": [k for k, v in report["refused"].items() if not v]}
    print(json.dumps(summary), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
