"""Check exported V4.1 routed-expert roles through the native expert ABI.

Loads a native library linked with the role shims (``v41_experts.cc`` and its
role wrappers), the expert packers and every exported object, then for each
role and capacity of the export manifests:

* checks the immutable metadata Rust validates (info, group rows, output kind,
  reduction attestation for the backbone roles);
* packs random official MXFP4 experts with the native loaders
  (``cuteafd_pack_expert_weights_async`` and ``cuteafd_v41_pack_scale_codes_async``)
  into a few spread slots of full-size resident planes;
* binds and initializes caller scratch, fills the caller pointer slots and
  launches through ``cuteafd_*_expert_launch`` (BF16 rows for the coordinator,
  rows from ``cuteafd_v41_expert_input_quantize_async`` for the FP8 K32 roles);
* compares the ordered FP32 route planes with an FP32 reference of the V4.1
  W4A8 recipe, requires zero planes for invalid routes, bit-identical repeats,
  and bit-identical first-row routes at every capacity of a role.

Usage: v41_experts_abi_check.py NATIVE_LIB EXPORT_ROOT [--roles a,b] [--max-rows N]
EXPORT_ROOT holds one directory per role export (v41_experts, v41_tp2_experts,
v41_local_experts, v41_dspark_tp2_experts), each with its v41_experts.json.
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import torch

P, I32, U32, U64 = C.c_void_p, C.c_int32, C.c_uint32, C.c_uint64
H = 5120
WIRE = H + H // 32
ROLES = {  # export directory, symbol prefix, native role id
    "coordinator": ("v41_experts", "cuteafd_v41_expert", 0),
    "rtx_backbone": ("v41_local_experts", "cuteafd_v41_local_expert", 2),
    "rtx_tp2": ("v41_tp2_experts", "cuteafd_v41_tp2_expert", 3),
    "dspark_tp2": ("v41_dspark_tp2_experts", "cuteafd_v41_dspark_tp2_expert", 4),
}
REDUCTION = {"rtx_backbone": "cuteafd_v41_local_expert_info_reduction",
             "rtx_tp2": "cuteafd_v41_tp2_expert_info_reduction"}


class Info(C.Structure):
    _fields_ = [("abi_version", U32), ("role", U32), ("experts", U32), ("hidden", U32),
                ("logical_intermediate", U32), ("kernel_intermediate", U32), ("topk", U32),
                ("capacity_rows", U32), ("scratch_bytes", U64), ("max_rows", I32),
                ("rows_padded", I32), ("max_tasks", I32), ("max_phys_tiles", I32),
                ("max_active_clusters", I32), ("input_dtype", U32)]


class Launch(C.Structure):
    _fields_ = [("tensors", P * 44), ("num_tokens", I32), ("max_rows", I32), ("scatter_rows", I32),
                ("rows_padded", I32), ("max_tasks", I32), ("max_phys_tiles", I32),
                ("max_active_clusters", I32), ("stream", P)]


class Reduction(C.Structure):
    _fields_ = [(n, U32) for n in ("version", "slice_width", "fc2_k_steps", "plane_count", "plane_fold",
                                   "implementation", "mma_k", "block_threads", "fc1_k_stage")]


class Job(C.Structure):
    _fields_ = [("source", P), ("destination", P), ("source_bytes", U64), ("destination_bytes", U64),
                ("pitch", U64), ("rows", U32), ("columns", U32), ("stored_rows", U32), ("start_row", U32)]


assert C.sizeof(Info) == 64 and C.sizeof(Launch) == 392 and C.sizeof(Job) == 56


class Native:
    def __init__(self, path):
        self.lib = C.CDLL(str(path))

    def fn(self, name, types):
        fn = getattr(self.lib, name)
        fn.argtypes, fn.restype = types, I32

        def checked(*values):
            status = fn(*values)
            if status:
                raise RuntimeError(f"{name} returned CUDA status {status}")
        return checked


def ptr(t):
    return P(t.data_ptr())


def stream():
    return P(torch.cuda.current_stream().cuda_stream)


def mx_quant(x):
    shp = x.shape
    b = x.float().reshape(-1, shp[-1] // 32, 32)
    amax = b.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    bits = (amax * (1.0 / 448.0)).contiguous().view(torch.int32)
    bits = torch.where(bits & 0x7FFFFF != 0, (bits + 0x800000) & 0x7F800000, bits)
    q = (b * ((254 - ((bits >> 23) & 0xFF)) << 23).view(torch.float32)).clamp(-448, 448)
    return (q.to(torch.float8_e4m3fn).float() * bits.view(torch.float32)).reshape(shp)


class Experts:
    """Random official experts packed by the native loaders into chosen slots."""

    LUT = [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]

    def __init__(self, native, info, manifest, slots, seed):
        self.inter = info.logical_intermediate
        self.experts = info.experts
        sizes = manifest["expert_plane_bytes"]
        self.sizes = sizes
        self.planes = [torch.empty(self.experts * size, dtype=torch.uint8, device="cuda") for size in sizes]
        self.slots = slots
        self.logical = {}
        g = torch.Generator(device="cuda").manual_seed(seed)
        pack_weights = native.fn("cuteafd_pack_expert_weights_async", [P, P, U32, P])
        pack_codes = native.fn("cuteafd_v41_pack_scale_codes_async", [P, U32, P, P])
        n = self.inter
        for slot in slots:
            w = {name: torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=g)
                 for name, shape in (("w1", (n, H // 2)), ("w3", (n, H // 2)), ("w2", (H, n // 2)))}
            s = {name: torch.randint(118, 125, (*w[name].shape[:-1], w[name].shape[-1] // 16),
                                     dtype=torch.uint8, device="cuda", generator=g) for name in w}
            sources = (P * 3)(*(ptr(w[name]) for name in ("w1", "w3", "w2")))
            destinations = (P * 2)(P(self.planes[0].data_ptr() + slot * sizes[0]),
                                   P(self.planes[2].data_ptr() + slot * sizes[2]))
            pack_weights(sources, destinations, U32(n), stream())
            jobs = (Job * 3)()
            s13 = self.planes[1].data_ptr() + slot * sizes[1]
            for i, (name, start) in enumerate((("w3", 0), ("w1", n))):
                jobs[i] = Job(ptr(s[name]), P(s13), n * (H // 32), sizes[1], H // 32, n, H // 32, 2 * n, start)
            jobs[2] = Job(ptr(s["w2"]), P(self.planes[3].data_ptr() + slot * sizes[3]), H * (n // 32),
                          sizes[3], n // 32, H, n // 32, H, 0)
            # The job table is mapped pinned host memory: read by the host and the device.
            mapped_jobs = torch.frombuffer(bytearray(jobs), dtype=torch.uint8).pin_memory()
            error = torch.full((1,), -1, dtype=torch.int64, device="cuda")
            pack_codes(ptr(mapped_jobs), U32(3), ptr(error), stream())
            torch.cuda.synchronize()
            if int(error.item()) != -1:
                raise RuntimeError(f"scale code span error {int(error.item()):#x}")
            self.logical[slot] = (w, s)

    def dequant(self, slot):
        lut = torch.tensor(self.LUT, device="cuda")
        w, s = self.logical[slot]
        out = []
        for name in ("w1", "w3", "w2"):
            codes = w[name]
            values = torch.stack((lut[(codes & 15).long()], lut[(codes >> 4).long()]), -1).flatten(-2)
            out.append(values * torch.exp2(s[name].float() - 127).repeat_interleave(32, -1))
        return out


def reference(experts, xdq, ids, weights):
    rows, topk = ids.shape
    out = torch.zeros(rows * topk, H, device="cuda")
    for slot in sorted(set(ids[ids >= 0].tolist())):
        tok, k = torch.where(ids == slot)
        w1, w3, w2 = experts.dequant(slot)
        a = xdq[tok]
        gate = (a @ w1.T).bfloat16().float().clamp(max=10.0)
        up = (a @ w3.T).bfloat16().float().clamp(-10.0, 10.0)
        h = (gate * torch.sigmoid(gate) * up * weights[tok, k, None]).bfloat16().float()
        out[tok * topk + k] = mx_quant(h) @ w2.T
    return out


def check_role(native, role, root, max_rows, seed):
    directory, prefix, role_id = ROLES[role]
    manifest = json.loads((root / directory / "v41_experts.json").read_text())
    info_fn = native.fn(prefix + "_info", [I32, C.POINTER(Info)])
    rows_fn = native.fn(prefix + "_group_rows", [I32, I32, C.POINTER(U32)])
    kind_fn = native.fn(prefix + "_output_kind", [I32, C.POINTER(U32)])
    init_fn = native.fn(prefix + "_initialize", [I32, C.POINTER(P)])
    bind_fn = native.fn(prefix + "_bind_scratch", [P, P, U64, P])
    scratch_fn = native.fn(prefix + "_initialize_scratch_async", [P, P, U64, P])
    launch_fn = native.fn(prefix + "_launch", [P, C.POINTER(Launch)])
    quant_init = native.fn("cuteafd_v41_expert_input_quant_initialize", [C.POINTER(P)])
    quantize = native.fn("cuteafd_v41_expert_input_quantize_async", [P, P, P, U32, P])
    quant = P()
    quant_init(C.byref(quant))
    capacities = [v["capacity_rows"] for v in manifest["variants"]]
    first = Info()
    info_fn(I32(capacities[0]), C.byref(first))
    topk, experts_n = first.topk, first.experts
    # Spread slots exercise large expert strides; routes use only these.
    slots = sorted({0, 1, 5, experts_n // 3, experts_n // 2 + 3, experts_n - 2, experts_n - 1})
    experts = Experts(native, first, manifest, slots, seed)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    first_rows = {}
    for capacity in capacities:
        info = Info()
        info_fn(I32(capacity), C.byref(info))
        expected = (2, role_id, experts_n, H, experts.inter, experts.inter, topk, capacity)
        actual = (info.abi_version, info.role, info.experts, info.hidden, info.logical_intermediate,
                  info.kernel_intermediate, info.topk, info.capacity_rows)
        assert actual == expected, (role, capacity, actual, expected)
        assert info.scratch_bytes > 0 and info.input_dtype == (1 if role == "coordinator" else 7)
        assert info.max_active_clusters == manifest["physical_sms"] or manifest["capability"] != [8, 9]
        group = U32()
        rows_fn(I32(capacity), I32(0), C.byref(group))
        kind = U32()
        kind_fn(I32(capacity), C.byref(kind))
        assert group.value == 16 and kind.value == 0, (group.value, kind.value)
        if role in REDUCTION:
            red = Reduction()
            native.fn(REDUCTION[role], [I32, C.POINTER(Reduction)])(I32(capacity), C.byref(red))
            assert (red.version == 1 and red.slice_width in (32, 64, 128, 192) and red.mma_k == 32
                    and red.block_threads == 128 and red.fc1_k_stage == 128
                    and red.fc2_k_steps == red.slice_width // 32
                    and red.plane_count == -(-experts.inter // red.slice_width)
                    and red.plane_fold == 1 and red.implementation == 1), (role, capacity)
        kernel = P()
        init_fn(I32(capacity), C.byref(kernel))
        scratch = torch.empty(info.scratch_bytes, dtype=torch.uint8, device="cuda")
        tensors = (P * 44)()
        bind_fn(kernel, ptr(scratch), U64(info.scratch_bytes), tensors)
        scratch_fn(kernel, ptr(scratch), U64(info.scratch_bytes), stream())
        rows = min(capacity, max_rows)
        live = rows if rows < 3 or rows < capacity else rows - 1  # odd live counts
        x = (torch.randn(capacity, H, device="cuda", generator=g) * 0.6).bfloat16()
        x[0] = first_rows.get("x", x[0])
        first_rows.setdefault("x", x[0].clone())
        ids = torch.stack([torch.tensor(slots, device="cuda")[torch.randperm(len(slots), generator=g,
                                                                           device="cuda")[:topk]]
                           for _ in range(capacity)]).int()
        ids[0] = first_rows.get("ids", ids[0])
        first_rows.setdefault("ids", ids[0].clone())
        if live > 1:
            ids[live - 1, topk - 1] = -1  # an invalid route; row 0 stays comparable
        weights = torch.rand(capacity, topk, device="cuda", generator=g) * 0.3 + 0.05
        weights[0] = first_rows.get("weights", weights[0])
        first_rows.setdefault("weights", weights[0].clone())
        flat_ids = ids.reshape(-1).contiguous()
        flat_w = weights.reshape(-1).contiguous()
        wire = torch.zeros(capacity, WIRE, dtype=torch.uint8, device="cuda")
        dummy = torch.zeros(16, dtype=torch.uint8, device="cuda")
        for slot in range(44):
            if not tensors[slot]:
                tensors[slot] = ptr(dummy)
        if role == "coordinator":
            tensors[0] = ptr(x)
        else:
            quantize(quant, ptr(x), ptr(wire), U32(live), stream())
            tensors[0] = ptr(wire)
        tensors[1], tensors[2] = ptr(flat_ids), ptr(flat_w)
        for slot, plane in zip((30, 31, 32, 33), experts.planes):
            tensors[slot] = ptr(plane)
        out_slot = tensors[41]
        args = Launch(tensors, live, info.max_rows, live * topk, info.rows_padded, info.max_tasks,
                      info.max_phys_tiles, info.max_active_clusters, stream())
        results = []
        for _ in range(2):
            launch_fn(kernel, C.byref(args))
            torch.cuda.synchronize()
            planes = _device_view(out_slot, capacity * topk * H).view(capacity * topk, H)
            results.append(planes[:live * topk].clone())
        assert torch.equal(results[0], results[1]), f"{role} m{capacity}: repeat differs"
        output = results[0]
        if role == "coordinator":
            wire_rows = _device_view(tensors[11], capacity * WIRE, torch.uint8).view(capacity, WIRE)[:live]
        else:
            wire_rows = wire[:live]
        codes = wire_rows[:, :H].contiguous().view(torch.float8_e4m3fn).float().view(live, H // 32, 32)
        xdq = (codes * torch.exp2(wire_rows[:, H:].float() - 127)[..., None]).view(live, H)
        assert torch.equal(xdq, mx_quant(x[:live].float())), f"{role} m{capacity}: input quantizer differs"
        expected_routes = reference(experts, xdq, ids[:live], weights[:live])
        valid = (flat_ids[:live * topk] >= 0)
        assert bool((output[~valid] == 0).all()), f"{role} m{capacity}: invalid routes not zero"
        diff = (output[valid] - expected_routes[valid]).norm() / expected_routes[valid].norm()
        assert torch.isfinite(output).all() and diff.item() < 5e-3, (role, capacity, diff.item())
        if "routes0" in first_rows:
            assert torch.equal(output[:topk], first_rows["routes0"]), f"{role} m{capacity}: row 0 changed"
        else:
            first_rows["routes0"] = output[:topk].clone()
        print(f"{role:12s} m{capacity:<5d} live={live:<5d} routes={int(valid.sum()):<6d} relL2={diff.item():.2e} "
              f"scratch={info.scratch_bytes} ok", flush=True)
        del scratch


def _device_view(address, count, dtype=torch.float32):
    """Borrow ``count`` elements of device memory at ``address`` as a tensor."""
    class Holder:
        __cuda_array_interface__ = {"shape": (count,), "typestr": {torch.float32: "<f4", torch.uint8: "|u1"}[dtype],
                                    "data": (address, False), "version": 3}
    return torch.as_tensor(Holder(), device="cuda")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_lib", type=Path)
    parser.add_argument("export_root", type=Path)
    parser.add_argument("--roles", default=",".join(ROLES))
    parser.add_argument("--max-rows", type=int, default=4096)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    native = Native(args.native_lib)
    for index, role in enumerate(args.roles.split(",")):
        check_role(native, role, args.export_root, args.max_rows, 8900 + index)
    print("V41_EXPERTS_ABI PASS")


if __name__ == "__main__":
    main()
