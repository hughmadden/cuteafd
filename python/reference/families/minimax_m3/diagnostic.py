#!/usr/bin/env python3
"""Replay one native layer dump from its exact inputs, including causal history.

Native capture uses CUTEAFD_MINIMAX_DUMP_DIR and CUTEAFD_MINIMAX_DUMP_LAYER.
The manifest has schema, layer and steps; each step contains positions and a
tensors map of {path,dtype,shape}. Payloads are little endian, row-major;
q2k is [rows,heads,topk]. Capture must start at position zero for one request.
This diagnostic tolerance is separate from the full-logit fidelity gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
from families.minimax_m3 import model

SCHEMA = "cuteafd.minimax.layer-diagnostic/1"
DTYPES = {"bf16": "<u2", "f32": "<f4", "i32": "<i4", "i64": "<i8", "u8": "u1"}
ORDER = ("input", "norm", "q", "k", "v", "index_q", "index_k", "k_fp8", "v_fp8",
         "index_q_fp8", "index_q_scales", "index_k_fp8", "index_k_scales", "q2k",
         "attention", "attention_projected", "post_attention", "ffn_norm",
         "router_ids", "router_weights", "shared", "residual")
TENSOR_DTYPES = {name: ("f32" if name in ("input", "post_attention", "residual", "router_weights")
                       or name.endswith("_scales") else "u8" if name.endswith("_fp8")
                       else "i32" if name in ("q2k", "router_ids") else "bf16") for name in ORDER}


def read_tensor(root, entry):
    path = root / entry["path"]
    shape, dtype = entry["shape"], entry["dtype"]
    if (not path.resolve().is_relative_to(root.resolve()) or dtype not in DTYPES
            or not shape or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError("invalid diagnostic tensor descriptor")
    storage = np.dtype(DTYPES[dtype])
    if path.stat().st_size != math.prod(shape) * storage.itemsize:
        raise ValueError("diagnostic tensor byte extent differs: " + entry["path"])
    if "sha256" in entry and hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("diagnostic tensor hash differs: " + entry["path"])
    value = torch.from_numpy(np.fromfile(path, dtype=storage).reshape(shape))
    return value.view(torch.bfloat16) if dtype == "bf16" else value


def read_dump(root, cfg):
    manifest = json.loads((root / "manifest.json").read_bytes())
    layer, steps = manifest.get("layer"), manifest.get("steps")
    if (manifest.get("schema") != SCHEMA or manifest.get("activation") != "served-cache" or type(layer) is not int
            or not 0 <= layer < cfg.num_hidden_layers or not steps):
        raise ValueError("invalid MiniMax layer diagnostic manifest")
    required = {"input", "norm", "q", "k", "v", "q2k", "attention", "attention_projected",
                "post_attention", "ffn_norm", "residual"}
    if cfg.msa["sparse_attention_freq"][layer]:
        required.update(("index_q", "index_k", "index_q_fp8", "index_q_scales",
                         "index_k_fp8", "index_k_scales", "k_fp8", "v_fp8"))
    if cfg.moe_layer_freq[layer]:
        required.update(("router_ids", "router_weights", "shared"))
    chunks, cursor, inventory = [], 0, None
    for step in steps:
        positions = step["positions"]
        if not positions or positions != list(range(cursor, cursor + len(positions))):
            raise ValueError("diagnostic requires contiguous causal history from position zero")
        names = set(step["tensors"])
        if not required <= names or not names <= set(ORDER) or (inventory is not None and names != inventory):
            raise ValueError("incomplete or inconsistent diagnostic tensor inventory")
        if any(entry["dtype"] != TENSOR_DTYPES[name] for name, entry in step["tensors"].items()):
            raise ValueError("diagnostic tensor dtype differs")
        values = {name: read_tensor(root, entry) for name, entry in step["tensors"].items()}
        if any(t.shape[0] != len(positions) for t in values.values()):
            raise ValueError("diagnostic tensors must have a leading query-row axis")
        if values["input"].shape != (len(positions), cfg.hidden_size) or values["input"].dtype != torch.float32:
            raise ValueError("diagnostic input must be FP32 residual rows")
        chunks.append(values)
        cursor += len(positions)
        inventory = names
    return manifest, {name: torch.cat([chunk[name] for chunk in chunks]) for name in inventory}


def write_dump(root, layer, tensors):
    root.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name, value in tensors.items():
        value = value.detach().cpu().contiguous()
        dtype = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int32: "i32",
                 torch.int64: "i64", torch.uint8: "u8"}[value.dtype]
        data = value.view(torch.uint16).numpy() if dtype == "bf16" else value.numpy()
        path = name + ".bin"
        data.astype(DTYPES[dtype], copy=False).tofile(root / path)
        entries[name] = dict(path=path, dtype=dtype, shape=list(value.shape))
    manifest = dict(schema=SCHEMA, activation="served-cache", layer=layer,
                    steps=[dict(positions=list(range(len(tensors["input"]))), tensors=entries)])
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def difference(name, actual, expected, *, rtol, atol, exact=False):
    if actual.shape != expected.shape:
        return dict(tensor=name, passed=False, reason="shape mismatch",
                    actual_shape=list(actual.shape), expected_shape=list(expected.shape))
    if exact:
        match = actual == expected
        finite = bool(torch.isfinite(actual.float()).all() and torch.isfinite(expected.float()).all())
    else:
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        match = torch.isclose(actual.float(), expected.float(), rtol=rtol, atol=atol)
    delta = (actual.double() - expected.double()).abs()
    return dict(tensor=name, passed=finite and bool(match.all()), exact=exact,
                finite=finite, mismatches=int((~match).sum()), elements=actual.numel(),
                max_abs=float(delta.max()) if finite else None)


def selection_from_cache(native, cfg):
    query = native["index_q_fp8"].view(torch.float8_e4m3fn).float().transpose(0, 1)
    key = native["index_k_fp8"].view(torch.float8_e4m3fn).float().transpose(0, 1)
    qs, ks = native["index_q_scales"].T, native["index_k_scales"].T
    selected = []
    for start in range(0, query.shape[1], model.ROWS):
        end = min(start + model.ROWS, query.shape[1])
        scores = model.index_scores(query[:, start:end], qs[:, start:end], key[:, :end], ks[:, :end])
        positions = torch.arange(start, end)
        selected.append(model.msa_select(scores, positions, cfg.msa["sparse_block_size"],
                                        cfg.msa["sparse_topk_blocks"], cfg.msa["sparse_local_block"]).transpose(0, 1))
    return torch.cat(selected)


def compare(snapshot, dump, *, device="cpu", rtol=.02, atol=.02, reference_out=None):
    if not all(math.isfinite(v) and v >= 0 for v in (rtol, atol)):
        raise ValueError("diagnostic tolerances must be finite and nonnegative")
    cfg = model.Config.from_dict(json.loads((snapshot / "config.json").read_bytes()))
    manifest, native = read_dump(dump, cfg)
    layer = manifest["layer"]
    weights = model.Weights(snapshot, device=device)
    weights.validate(cfg)
    trace = {}
    with torch.inference_mode():
        model.layer_forward(native["input"].to(device), weights.load_layer(layer, cfg),
                            cfg, layer, activation="served-cache", trace=trace)
    weights.close()
    trace = {name: value.cpu() for name, value in trace.items()}
    if reference_out is not None:
        write_dump(reference_out, layer, trace)

    # Test pack programs on the dumped operands. Upstream norm differences
    # must not be misreported as conversion/packing defects.
    packed = {"k_fp8": model.cache_e4m3(native["k"]).view(torch.uint8),
              "v_fp8": model.cache_e4m3(native["v"]).view(torch.uint8)}
    for prefix in ("index_q", "index_k"):
        if prefix in native:
            value, scale = model.index_e4m3(native[prefix])
            packed[prefix + "_fp8"], packed[prefix + "_scales"] = value.view(torch.uint8), scale
    checks = []
    for name in ORDER:
        if name in native:
            expected = packed[name] if name in packed else trace[name]
            checks.append(difference(name, native[name], expected, rtol=rtol, atol=atol,
                                     exact=name in packed or name in ("input", "q2k", "router_ids")))
            checks[-1]["comparison_inputs"] = "native_program_operands" if name in packed else "layer_input"
            if name == "q2k":
                checks.append(difference("q2k_native_inputs", native[name], selection_from_cache(native, cfg),
                                         rtol=0., atol=0., exact=True))
                checks[-1]["comparison_inputs"] = "native_program_operands"
    failed = next((entry["tensor"] for entry in checks if not entry["passed"]), None)
    return dict(schema=SCHEMA, layer=layer, rows=len(native["input"]), activation="served-cache",
                passed=failed is None, first_failure=failed, rtol=rtol, atol=atol, tensors=checks,
                source_sha256=hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
                limitations="FP32 eager reductions; approximate native norm/exp and tensor-core accumulation require hardware qualification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reference-out", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rtol", type=float, default=.02)
    parser.add_argument("--atol", type=float, default=.02)
    args = parser.parse_args()
    report = compare(args.snapshot, args.dump, device=args.device, rtol=args.rtol,
                     atol=args.atol, reference_out=args.reference_out)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
