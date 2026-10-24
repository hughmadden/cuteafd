#!/usr/bin/env python3
"""MiniMax M3.1 tower parity on real weights (G2) and repeat stability/timing (G3).

Run inside the coordinator CUDA container with the native library of a WIP slot built
with the vision attention AOT. Only the 523 tower/projector tensors are read.

The reference is python/reference/minimax_m3_vision.py (qualified against Transformers'
MiniMaxM3VL tower in scripts/tests/test_minimax_m3_vision_reference.py); inputs are
prepared by python/reference/minimax_m3_media.py, which processor-parity.py checks
against the release processors. The reference runs twice on the GPU: with the release
numerics (BF16, F32 convolution) and in FP32 as the yardstick. A stage passes when the
native tower is no farther from FP32 (relative L2, mean row cosine) than the BF16
reference, with a small margin. Segments above --reference-max-patches are timed only:
the reference materializes full attention scores.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python/reference"))
import minimax_m3_media as media
import minimax_m3_vision as reference

NO_OFFSET = (1 << 64) - 1
PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")
TOWER = reference.VISION_PREFIX
FIELDS = "qkv qkv_bias proj proj_bias gate_up gate_up_bias down down_bias norm1 norm2 key0_bias".split()


class Block(C.Structure):
    _fields_ = [(k, C.c_uint64) for k in FIELDS] + [("window", C.c_int32), ("column_order", C.c_int32)]


class Spec(C.Structure):
    _fields_ = [(k, C.c_uint32) for k in ("abi_version", "max_tokens", "output_width", "reserved")]
    _fields_ += [(k, C.c_uint64) for k in ("weight_bytes", "patch", "merger_norm", "merger_fc1", "merger_fc2", "inv_freq")]
    _fields_ += [("blocks", Block * 28)]
    _fields_ += [(k, C.c_uint32) for k in ("hidden", "depth", "heads", "kv_heads", "head_dim", "intermediate",
                                           "patch_size", "merger_width")]
    _fields_ += [("norm_eps", C.c_float), ("geometry_reserved", C.c_uint32)]
    _fields_ += [(k, C.c_uint64) for k in ("patch_bias", "pos_embed", "merger_norm_bias", "merger_fc1_bias",
                                           "merger_fc2_bias")]
    _fields_ += [("merger_extra", C.c_uint64 * 8)]
    _fields_ += [(k, C.c_uint64 * 28) for k in ("norm1_bias", "norm2_bias", "q_norm", "k_norm")]
    _fields_ += [("blocks_extra", Block * 4), ("norm1_bias_extra", C.c_uint64 * 4), ("norm2_bias_extra", C.c_uint64 * 4)]


class Ledger(C.Structure):
    _fields_ = [(k, C.c_uint64) for k in ("weights", "scratch", "blas_workspace", "device_allocations", "encodes")]

    def as_dict(self):
        return {k: getattr(self, k) for k, _ in self._fields_}


OBSERVER = C.CFUNCTYPE(C.c_int32, C.c_void_p, C.c_int32, C.c_void_p, C.c_int32, C.c_int32, C.c_int32)
assert C.sizeof(Spec) == 4240 and C.sizeof(Block) == 96


def read_tower(snapshot):
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for shard in sorted({v for k, v in index.items() if k.startswith(PREFIXES)}):
        if Path(shard).is_absolute() or any(p in ("", ".", "..") for p in shard.split("/")):
            raise ValueError("tower shard escapes snapshot")
        with (snapshot / shard).open("rb") as source:
            size = int.from_bytes(source.read(8), "little")
            header = json.loads(source.read(size))
            for name, entry in header.items():
                if not name.startswith(PREFIXES):
                    continue
                dtype = {"BF16": "<u2", "F32": "<f4"}[entry["dtype"]]
                start, end = entry["data_offsets"]
                source.seek(8 + size + start)
                raw = source.read(end - start)
                if len(raw) != end - start or name in tensors:
                    raise ValueError(f"truncated or duplicate tower tensor {name}")
                tensors[name] = np.frombuffer(raw, dtype).reshape(entry["shape"]).copy()
    if len(tensors) != 523:
        raise ValueError(f"expected the release's 523 tower tensors, found {len(tensors)}")
    return tensors


def pack_spec(tensors, max_tokens=4096):
    """TowerSpec::minimax's arena, byte for byte (BF16 vectors widen to FP32)."""
    spec = Spec(abi_version=3, reserved=4, max_tokens=max_tokens, output_width=6144, hidden=1280, depth=32,
                heads=16, kv_heads=16, head_dim=80, intermediate=5120, patch_size=14, merger_width=24576,
                norm_eps=1e-5, patch_bias=NO_OFFSET, pos_embed=NO_OFFSET)
    for name, count in (("q_norm", 28), ("k_norm", 28), ("merger_extra", 8)):
        for i in range(count):
            getattr(spec, name)[i] = NO_OFFSET
    chunks, used, cursor = [], set(), 0

    def put(name, shape, vector, align, dtype):
        nonlocal cursor
        value = tensors[name]
        if list(value.shape) != list(shape) or value.dtype != np.dtype(dtype):
            raise ValueError(f"{name}: {value.dtype} {value.shape}, need {dtype} {shape}")
        used.add(name)
        if align:
            gap = -cursor % 256
            chunks.append(bytes(gap))
            cursor += gap
        offset = cursor
        data = ((value.astype(np.uint32) << 16).view(np.float32) if vector else value).tobytes()
        chunks.append(data)
        cursor += len(data)
        return offset

    spec.patch = put(TOWER + "embeddings.patch_embedding.weight", [1280, 3, 2, 14, 14], False, True, "<f4")
    spec.merger_norm = put(TOWER + "pre_layrnorm.weight", [1280], True, True, "<u2")
    spec.merger_norm_bias = put(TOWER + "pre_layrnorm.bias", [1280], True, True, "<u2")
    for i in range(32):
        def field(suffix, shape, vector, align):
            return put(f"{TOWER}encoder.layers.{i}.{suffix}", shape, vector, align, "<u2")
        block = Block(key0_bias=NO_OFFSET)
        block.qkv = field("self_attn.q_proj.weight", [1280, 1280], False, True)
        field("self_attn.k_proj.weight", [1280, 1280], False, False)
        field("self_attn.v_proj.weight", [1280, 1280], False, False)
        block.qkv_bias = field("self_attn.q_proj.bias", [1280], True, True)
        field("self_attn.k_proj.bias", [1280], True, False)
        field("self_attn.v_proj.bias", [1280], True, False)
        block.proj = field("self_attn.out_proj.weight", [1280, 1280], False, True)
        block.proj_bias = field("self_attn.out_proj.bias", [1280], True, True)
        block.gate_up = field("mlp.fc1.weight", [5120, 1280], False, True)
        block.gate_up_bias = field("mlp.fc1.bias", [5120], True, True)
        block.down = field("mlp.fc2.weight", [1280, 5120], False, True)
        block.down_bias = field("mlp.fc2.bias", [1280], True, True)
        block.norm1 = field("layer_norm1.weight", [1280], True, True)
        block.norm2 = field("layer_norm2.weight", [1280], True, True)
        n1, n2 = field("layer_norm1.bias", [1280], True, True), field("layer_norm2.bias", [1280], True, True)
        if i < 28:
            spec.blocks[i], spec.norm1_bias[i], spec.norm2_bias[i] = block, n1, n2
        else:
            spec.blocks_extra[i - 28], spec.norm1_bias_extra[i - 28], spec.norm2_bias_extra[i - 28] = block, n1, n2
    for i, stem, shape in ((0, "multi_modal_projector.linear_1", [6144, 1280]),
                           (2, "multi_modal_projector.linear_2", [6144, 6144])):
        spec.merger_extra[i] = put(stem + ".weight", shape, False, True, "<u2")
        spec.merger_extra[i + 1] = put(stem + ".bias", [6144], True, True, "<u2")
    spec.merger_fc1 = put("patch_merge_mlp.linear_1.weight", [6144, 24576], False, True, "<u2")
    spec.merger_fc1_bias = put("patch_merge_mlp.linear_1.bias", [6144], True, True, "<u2")
    spec.merger_fc2 = put("patch_merge_mlp.linear_2.weight", [6144, 6144], False, True, "<u2")
    spec.merger_fc2_bias = put("patch_merge_mlp.linear_2.bias", [6144], True, True, "<u2")
    if used != set(tensors):
        raise ValueError(f"unmapped tower tensors: {sorted(set(tensors) - used)}")
    gap = -cursor % 256
    chunks.append(bytes(gap))
    cursor += gap
    spec.inv_freq = cursor
    inv = (np.float32(1) / np.power(np.float32(10000), np.arange(13, dtype=np.float32) / np.float32(13))).astype("<f4")
    chunks.append(inv.tobytes())
    spec.weight_bytes = cursor + 13 * 4
    return spec, b"".join(chunks)


def reference_weights(tensors, dtype, device):
    """Checkpoint tensors as the reference takes them: the F32 convolution stays F32."""
    import torch
    weights = {}
    for name, value in tensors.items():
        tensor = torch.from_numpy(value)
        if value.dtype == np.uint16:
            tensor = tensor.view(torch.bfloat16)
        weights[name] = tensor.to(device, torch.float32 if value.dtype == np.float32 else dtype)
    return weights


def metrics(native, ref):
    a, b = native.astype(np.float64), ref.astype(np.float64)
    an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cos = np.sum(a * b, axis=-1) / np.maximum(an * bn, 1e-30)
    diff = np.abs(a - b)
    return dict(relative_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)), mean_cosine=float(cos.mean()),
                worst_cosine=float(cos.min()), max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                ref_mean_abs=float(np.abs(b).mean()))


def verdict(native, floor):
    """No farther from FP32 than the BF16 reference, with a small margin. Worst-row cosine
    is one row's minimum and is reported, not gated."""
    return (native["relative_l2"] <= floor["relative_l2"] * 1.25 + 0.002
            and native["mean_cosine"] >= floor["mean_cosine"] - 0.0005)


def inputs(args):
    """(name, frames [2t, H, W, 3] uint8 processor-sized, grid, video?)."""
    from PIL import Image
    cases = []
    for path in args.images:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        frame = media.resize_rgb(rgb, *media.smart_resize(*rgb.shape[:2]))
        cases.append((Path(path).stem, np.stack([frame, frame]), (1, frame.shape[0] // 14, frame.shape[1] // 14), False))
    clip = [np.asarray(Image.open(p).convert("RGB")) for p in args.clip_frames]
    for size in args.clip_sizes:
        frames = [media.resize_rgb(frame, *size) for frame in clip]
        frames += frames[-1:] * (len(frames) % 2)  # an odd clip repeats its last frame
        t = len(frames) // 2
        if not 1 <= t <= 4:
            raise ValueError("a clip segment holds one to eight frames")
        cases.append((f"clip-{t}x{size[0]}x{size[1]}", np.stack(frames), (t, size[0] // 14, size[1] // 14), True))
    return cases


def run(args):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tensors = read_tower(args.snapshot)
    spec, blob = pack_spec(tensors)
    library, cudart = C.CDLL(str(args.library)), C.CDLL("libcudart.so")
    library.cuteafd_vision_required.argtypes = [C.POINTER(Spec), C.POINTER(Ledger)]
    library.cuteafd_vision_create.argtypes = [C.POINTER(Spec), C.c_int32, C.c_uint64, C.POINTER(C.c_void_p)]
    library.cuteafd_vision_upload.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64]
    library.cuteafd_vision_encode.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p, C.c_int32, C.c_int32,
                                              C.c_void_p, C.c_uint64, OBSERVER, C.c_void_p]
    library.cuteafd_vision_encode_frames.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p, C.c_int32,
                                                     C.c_int32, C.c_int32, C.c_void_p, C.c_uint64]
    library.cuteafd_vision_get_ledger.argtypes = [C.c_void_p, C.POINTER(Ledger)]
    library.cuteafd_vision_destroy.argtypes = [C.c_void_p]
    cudart.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]

    def check(status):
        if status:
            raise RuntimeError(f"native MiniMax vision status {status}")

    table = media.normalization_lut()
    owner, required = C.c_void_p(), Ledger()
    check(library.cuteafd_vision_required(C.byref(spec), C.byref(required)))
    check(library.cuteafd_vision_create(C.byref(spec), 0, required.weights + required.scratch + required.blas_workspace,
                                        C.byref(owner)))
    result = dict(checkpoint=str(args.snapshot), device=torch.cuda.get_device_name(0),
                  sm=list(torch.cuda.get_device_capability(0)), ledger=required.as_dict(),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  g2={}, g3={}, encode_ms={})
    try:
        check(library.cuteafd_vision_upload(owner, 0, blob, len(blob)))
        del blob

        def encode(frames, grid, video, observe=False):
            t, gh, gw = grid
            rgb = np.ascontiguousarray(frames if video else frames[0])
            out = np.empty((t * gh * gw // 4, 6144), np.uint16)
            stages, errors = {}, []

            def callback(_ctx, stage, pointer, rows, width, column):
                try:
                    if stage == 33:
                        return 0
                    value = np.empty((rows, width), np.float32)
                    check(cudart.cudaMemcpy(value.ctypes.data, pointer, value.nbytes, 2))
                    stages[stage] = value
                    return 0
                except BaseException as error:
                    errors.append(repr(error))
                    return 1

            started = time.perf_counter()
            if video:
                check(library.cuteafd_vision_encode_frames(owner, rgb.ctypes.data, rgb.nbytes, table.ctypes.data,
                                                           t, gh, gw, out.ctypes.data, out.nbytes))
            else:
                observer = OBSERVER(callback) if observe else OBSERVER()
                check(library.cuteafd_vision_encode(owner, rgb.ctypes.data, rgb.nbytes, table.ctypes.data,
                                                    gh, gw, out.ctypes.data, out.nbytes, observer, None))
            elapsed = (time.perf_counter() - started) * 1000
            if errors:
                raise RuntimeError(errors)
            return (out.astype(np.uint32) << 16).view(np.float32), stages, elapsed

        cases = inputs(args)
        saved, natives = {}, {}
        for name, frames, grid, video in cases:
            encode(frames, grid, video)  # first use of this shape
            natives[name] = encode(frames, grid, video, observe=not args.native_only and not video)
            saved[name] = natives[name][0]
        if not args.native_only:
            arms = {}
            for arm, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
                weights = reference_weights(tensors, dtype, "cuda")
                for name, frames, grid, video in cases:
                    if grid[0] * grid[1] * grid[2] > args.reference_max_patches:
                        continue
                    captured = {}
                    observe = lambda stage, hidden: captured.__setitem__(stage, hidden.float().cpu().numpy()) \
                        if stage % 4 == 0 else None
                    pixels = torch.from_numpy(media.patches(frames, table)).cuda()
                    out = reference.forward(pixels, [grid], weights, dtype=dtype, observe=observe)
                    captured["output"] = out["features"].float().cpu().numpy()
                    arms.setdefault(name, {})[arm] = captured
                    del out
                del weights
                torch.cuda.empty_cache()
            for name, (native, stages, _) in natives.items():
                if name not in arms:
                    continue
                fp32, bf16 = arms[name]["fp32"], arms[name]["bf16"]
                observed = {str(s): v for s, v in stages.items()}
                observed["output"] = native
                floor = {str(s): metrics(v, fp32[s]) for s, v in bf16.items()}
                measured = {}
                for stage, value in observed.items():
                    key = stage if stage == "output" else int(stage)
                    m = metrics(value, fp32[key])
                    measured[stage] = dict(m, versus_bf16_reference=metrics(value, bf16[key])["relative_l2"],
                                           **{"pass": verdict(m, floor[stage])})
                grid = next(g for n, _, g, _ in cases if n == name)
                result["g2"][name] = dict(grid=list(grid), tokens=grid[0] * grid[1] * grid[2] // 4,
                                          native=measured, bf16_reference=floor)
                print(json.dumps(dict(event="G2", case=name, grid=grid, native=measured["output"],
                                      bf16_reference=floor["output"])), flush=True)
        del tensors
        initial, final = Ledger(), Ledger()
        check(library.cuteafd_vision_get_ledger(owner, C.byref(initial)))
        for repeat in range(args.repeats):
            for name, frames, grid, video in (reversed(cases) if repeat % 2 else cases):
                value, _, elapsed = encode(frames, grid, video)
                result["encode_ms"].setdefault(name, []).append(round(elapsed, 3))
                result["g3"][name] = result["g3"].get(name, True) and bool(np.array_equal(value, saved[name]))
        check(library.cuteafd_vision_get_ledger(owner, C.byref(final)))
        result.update(initial_ledger=initial.as_dict(), final_ledger=final.as_dict(),
                      no_encode_device_allocation=initial.device_allocations == final.device_allocations == 2,
                      g2_pass=bool(result["g2"]) and all(m["pass"] for case in result["g2"].values()
                                                         for m in case["native"].values()),
                      g3_pass=bool(result["g3"]) and all(result["g3"].values()))
        result["encode_ms_median"] = {k: float(np.median(v)) for k, v in result["encode_ms"].items()}
        args.output.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: result[k] for k in ("g2_pass", "g3_pass", "no_encode_device_allocation",
                                                  "encode_ms_median")}), flush=True)
        return 0 if result["g3_pass"] and result["no_encode_device_allocation"] and (result["g2_pass"] or args.native_only) else 1
    finally:
        check(library.cuteafd_vision_destroy(owner))


def size(value):
    h, w = (int(v) for v in value.split("x"))
    if h % 28 or w % 28:
        raise argparse.ArgumentTypeError("clip sizes are HxW multiples of 28")
    return h, w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="*", default=[])
    parser.add_argument("--clip-frames", type=Path, nargs="*", default=[],
                        help="one to eight sampled frames of a clip (one attention segment)")
    parser.add_argument("--clip-sizes", type=size, nargs="*", default=[], help="HxW sizes to encode the clip at")
    parser.add_argument("--reference-max-patches", type=int, default=7000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--native-only", action="store_true")
    raise SystemExit(run(parser.parse_args()))
