#!/usr/bin/env python3
"""MiniMax M3.1 tower parity on real weights (G2) and repeat stability/timing (G3).

Run inside the coordinator CUDA container with the native library of a WIP slot built
with the vision attention AOT. Only the 523 tower/projector tensors are read.

The reference is an independent PyTorch transcription of the release's tower (the
newer Transformers modeling_minimax_m3_vl.py and MiniMax's sglang minimax_vl_common.py
agree on the graph): Conv3d 2x14x14 -> pre LayerNorm -> 32 CLIP layers (LayerNorm, MHA
with 3D axial RoPE on 78 of 80 channels, LayerNorm, GELU MLP) -> per-patch GELU
projector -> 2x2 merge GELU MLP. Video segments attend within at most four temporal
patches (vision_segment_max_frames). It runs in FP32 as the yardstick target, and in
BF16 with the numerics each vendor path uses:

  sglang       BF16 pixels and conv, FP32 rotary (one BF16 rounding of q and k)
  hf-5.19      BF16 pixels and conv, BF16 cos/sin and products
  native-fp32  FP32 conv on FP32 pixels (the checkpoint's F32 weight), FP32 rotary
  native-bf16  FP32 conv, Transformers' BF16 rotary

The native tower passes when its distance to the FP32 reference is within the sglang
BF16 arm's distance plus a small margin, per stage.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

NO_OFFSET = (1 << 64) - 1
PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")
TOWER = "vision_tower.vision_model."
FIELDS = "qkv qkv_bias proj proj_bias gate_up gate_up_bias down down_bias norm1 norm2 key0_bias".split()
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


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


def lut():
    # The fast processor fuses rescale into FP32 mean/std: (v - mean*255) / (std*255).
    values = np.arange(256, dtype=np.float32)
    return np.ascontiguousarray(np.stack([(values - np.float32(m) * np.float32(255)) / (np.float32(s) * np.float32(255))
                                          for m, s in zip(MEAN, STD)]).astype(np.float32))


def smart_resize(height, width, factor=28, min_pixels=3136, max_pixels=451584):
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar, w_bar = math.floor(height / beta / factor) * factor, math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resized(rgb, height, width):
    """The release processor's torch uint8 antialiased bicubic resize (HWC in and out)."""
    import torch
    import torch.nn.functional as F
    if rgb.shape[:2] == (height, width):
        return np.ascontiguousarray(rgb)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
    out = F.interpolate(tensor, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    return np.ascontiguousarray(out[0].permute(1, 2, 0).numpy())


def patches(frames, table):
    """[t, bh, bw, mh, mw, C, temporal, ph, pw] rows from 2*t RGB8 frames (HWC)."""
    frames = np.asarray(frames)
    count, h, w, _ = frames.shape
    t, gh, gw = count // 2, h // 14, w // 14
    values = np.stack([table[c][frames[..., c]] for c in range(3)], axis=1)  # [2t, C, H, W]
    values = values.reshape(t, 2, 3, gh // 2, 2, 14, gw // 2, 2, 14).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(values.reshape(t * gh * gw, 1176))


def positions(t, gh, gw):
    """(t, h, w) per patch in merge-window order, as the processor lays patches out."""
    p = np.arange(t * gh * gw)
    spatial, unit, r = p % (gh * gw), (p % (gh * gw)) // 4, p % 4
    return np.stack([p // (gh * gw), (unit // (gw // 2)) * 2 + r // 2, (unit % (gw // 2)) * 2 + r % 2], axis=-1)


class Reference:
    def __init__(self, tensors, device):
        import torch
        self.torch, self.device = torch, device
        self.raw = {}
        for name, value in tensors.items():
            if value.dtype == np.uint16:
                value = (value.astype(np.uint32) << 16).view(np.float32)
            self.raw[name] = torch.from_numpy(np.ascontiguousarray(value)).to(device)
        self.inv = 1.0 / (10000.0 ** (torch.arange(0, 26, 2, dtype=torch.float32, device=device) / 26))
        self.layers = sum(name.endswith(".layer_norm1.weight") for name in tensors)

    def __call__(self, pixel_values, grid, dtype, conv_fp32, rotary_fp32, capture=None):
        torch = self.torch
        import torch.nn.functional as F
        W = {k: v.to(dtype) for k, v in self.raw.items()}
        t, gh, gw = grid
        n = t * gh * gw
        x = torch.from_numpy(pixel_values).to(self.device).view(n, 3, 2, 14, 14)
        conv_dtype = torch.float32 if conv_fp32 else dtype
        h = F.conv3d(x.to(conv_dtype), self.raw[TOWER + "embeddings.patch_embedding.weight"].to(conv_dtype),
                     stride=(2, 14, 14)).view(n, 1280).to(dtype)
        h = F.layer_norm(h, (1280,), W[TOWER + "pre_layrnorm.weight"], W[TOWER + "pre_layrnorm.bias"], 1e-5)
        if capture is not None:
            capture[0] = h.float().cpu().numpy()
        pos = torch.from_numpy(positions(t, gh, gw)).to(self.device).float()
        freqs = (pos[:, :, None] * self.inv).flatten(1)
        freqs = torch.cat([freqs, freqs], dim=-1)  # [n, 78]
        cos, sin = freqs.cos(), freqs.sin()
        if not rotary_fp32:
            cos, sin = cos.to(dtype), sin.to(dtype)

        def rotate(v):  # v: [n, 16, 80]
            rot, rest = v[..., :78], v[..., 78:]
            if rotary_fp32:
                rot = rot.float()
            half = torch.cat([-rot[..., 39:], rot[..., :39]], dim=-1)
            rot = rot * cos[:, None, :] + half * sin[:, None, :]
            return torch.cat([rot.to(v.dtype), rest], dim=-1)

        segment = 4 * gh * gw  # vision_segment_max_frames temporal patches per attention segment
        for i in range(self.layers):
            p = f"{TOWER}encoder.layers.{i}."
            y = F.layer_norm(h, (1280,), W[p + "layer_norm1.weight"], W[p + "layer_norm1.bias"], 1e-5)
            q, k, v = (F.linear(y, W[p + f"self_attn.{s}_proj.weight"], W[p + f"self_attn.{s}_proj.bias"]).view(n, 16, 80)
                       for s in ("q", "k", "v"))
            q, k = rotate(q), rotate(k)
            outputs = []
            for start in range(0, n, segment):
                ks, vs = (z[start:start + segment].transpose(0, 1)[None] for z in (k, v))
                end = min(n, start + segment)
                # Query chunks bound the MATH backend's score matrix; keys stay the whole segment.
                for row in range(start, end, 2048):
                    qs = q[row:min(end, row + 2048)].transpose(0, 1)[None]
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        outputs.append(F.scaled_dot_product_attention(qs, ks, vs, scale=80 ** -0.5)[0].transpose(0, 1))
            attn = torch.cat(outputs).reshape(n, 1280)
            h = h + F.linear(attn, W[p + "self_attn.out_proj.weight"], W[p + "self_attn.out_proj.bias"])
            y = F.layer_norm(h, (1280,), W[p + "layer_norm2.weight"], W[p + "layer_norm2.bias"], 1e-5)
            y = F.gelu(F.linear(y, W[p + "mlp.fc1.weight"], W[p + "mlp.fc1.bias"]))
            h = h + F.linear(y, W[p + "mlp.fc2.weight"], W[p + "mlp.fc2.bias"])
            if capture is not None and (i + 1) % 4 == 0:
                capture[i + 1] = h.float().cpu().numpy()
        y = F.gelu(F.linear(h, W["multi_modal_projector.linear_1.weight"], W["multi_modal_projector.linear_1.bias"]))
        y = F.linear(y, W["multi_modal_projector.linear_2.weight"], W["multi_modal_projector.linear_2.bias"])
        y = F.gelu(F.linear(y.reshape(n // 4, 24576), W["patch_merge_mlp.linear_1.weight"], W["patch_merge_mlp.linear_1.bias"]))
        y = F.linear(y, W["patch_merge_mlp.linear_2.weight"], W["patch_merge_mlp.linear_2.bias"])
        return y.float().cpu().numpy()


def metrics(native, ref):
    a, b = native.astype(np.float64), ref.astype(np.float64)
    an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cos = np.sum(a * b, axis=-1) / np.maximum(an * bn, 1e-30)
    diff = np.abs(a - b)
    return dict(relative_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)), mean_cosine=float(cos.mean()),
                worst_cosine=float(cos.min()), max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                ref_mean_abs=float(np.abs(b).mean()))


def verdict(native, floor):
    """Within the sglang BF16 arm's distance to FP32, with a small margin."""
    return (native["relative_l2"] <= floor["relative_l2"] * 1.25 + 0.002
            and native["mean_cosine"] >= floor["mean_cosine"] - 0.0005
            and native["worst_cosine"] >= floor["worst_cosine"] - 0.005)


def inputs(args):
    """(name, frames [2t, H, W, 3] uint8 already processor-sized, grid, video?)."""
    from PIL import Image
    cases = []
    for path in args.images:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        h, w = smart_resize(*rgb.shape[:2])
        frame = resized(rgb, h, w)
        cases.append((Path(path).stem, np.stack([frame, frame]), (1, h // 14, w // 14), False))
    rng = np.random.default_rng(7)
    for t, h, w in args.videos:
        base = rng.integers(0, 256, size=(h // 8 + 1, w // 8 + 1, 3)).astype(np.uint8)
        frames = []
        for k in range(2 * t):  # a panning texture: temporal positions matter
            shifted = np.roll(np.kron(base, np.ones((8, 8, 1), np.uint8)), 3 * k, axis=1)[:h, :w]
            frames.append(np.ascontiguousarray(shifted))
        cases.append((f"video-{t}x{h}x{w}", np.stack(frames), (t, h // 14, w // 14), True))
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

    table = lut()
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
        reference = None if args.native_only else Reference(tensors, "cuda")
        del tensors

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

        saved = {}
        arms = {"sglang": (torch.bfloat16, False, True), "hf-5.19": (torch.bfloat16, False, False),
                "native-fp32": (torch.bfloat16, True, True), "native-bf16": (torch.bfloat16, True, False)}
        for name, frames, grid, video in inputs(args):
            encode(frames, grid, video)  # first use of this shape
            native, stages, _ = encode(frames, grid, video, observe=reference is not None and not video)
            saved[name] = native
            if reference is None:
                continue
            pixels = patches(frames, table)
            with torch.inference_mode():
                fp32_stages = {}
                fp32 = reference(pixels, grid, torch.float32, True, True, fp32_stages)
                measured = {str(s): metrics(v, fp32_stages[s]) for s, v in stages.items()}
                measured["output"] = metrics(native, fp32)
                yardsticks = {}
                for arm, (dtype, conv_fp32, rotary_fp32) in arms.items():
                    arm_stages = {}
                    value = reference(pixels, grid, dtype, conv_fp32, rotary_fp32, arm_stages)
                    yardsticks[arm] = {str(s): metrics(v, fp32_stages[s]) for s, v in arm_stages.items()}
                    yardsticks[arm]["output"] = metrics(value, fp32)
                    yardsticks[arm]["native_vs_arm_output"] = metrics(native, value)
            floor = yardsticks["sglang"]
            measured = {stage: dict(value, **{"pass": verdict(value, floor[stage])}) for stage, value in measured.items()}
            result["g2"][name] = dict(grid=list(grid), tokens=grid[0] * grid[1] * grid[2] // 4, native=measured,
                                      yardsticks=yardsticks)
            print(json.dumps(dict(event="G2", case=name, grid=grid, output=measured["output"],
                                  sglang=floor["output"], native_fp32_rotary=yardsticks["native-fp32"]["output"],
                                  native_bf16_rotary=yardsticks["native-bf16"]["output"])), flush=True)
        reference = None
        torch.cuda.empty_cache()
        initial, final = Ledger(), Ledger()
        check(library.cuteafd_vision_get_ledger(owner, C.byref(initial)))
        cases = inputs(args)
        for repeat in range(args.repeats):
            for name, frames, grid, video in (reversed(cases) if repeat % 2 else cases):
                value, _, elapsed = encode(frames, grid, video)
                result["encode_ms"].setdefault(name, []).append(round(elapsed, 3))
                result["g3"][name] = result["g3"].get(name, True) and bool(np.array_equal(value, saved[name]))
        check(library.cuteafd_vision_get_ledger(owner, C.byref(final)))
        result.update(initial_ledger=initial.as_dict(), final_ledger=final.as_dict(),
                      no_encode_device_allocation=initial.device_allocations == final.device_allocations == 2,
                      g2_pass=all(m["pass"] for case in result["g2"].values() for m in case["native"].values()),
                      g3_pass=bool(result["g3"]) and all(result["g3"].values()))
        result["encode_ms_median"] = {k: float(np.median(v)) for k, v in result["encode_ms"].items()}
        args.output.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: result[k] for k in ("g2_pass", "g3_pass", "no_encode_device_allocation",
                                                  "encode_ms_median")}), flush=True)
        return 0 if result["g3_pass"] and result["no_encode_device_allocation"] and (result["g2_pass"] or args.native_only) else 1
    finally:
        check(library.cuteafd_vision_destroy(owner))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="*", default=[])
    parser.add_argument("--videos", type=lambda s: tuple(int(v) for v in s.split("x")), nargs="*", default=[],
                        help="TxHxW video segments (T temporal patches, 1..4; H and W multiples of 28)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--native-only", action="store_true")
    raise SystemExit(run(parser.parse_args()))
