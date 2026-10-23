#!/usr/bin/env python3
"""Layer-major MiniMax M3 text golden: native served QDQ or BF16-activation diagnostic.

Only one packed layer and one window are resident. Resume reads hash-sealed
BF16 layer checkpoints and requires identical source, snapshot, set and arm.
The CLI always qualifies byte-identical common-prefix logits before scoring.
Vision and the bundled drafter are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_windows import (LayerCheckpoints, admit_reference_memory, canonical,
                              finish_golden, load_set, verify_snapshot, write_scored_logits)
from families.minimax_m3 import model

VLLM_REVISION = "2d40c0609fd672e845ddfef0324ab6d8dd616af4"
# Immutable source paths and line ranges describe the arithmetic, never imported.
VLLM_SOURCE_PINS = {
    "vllm/model_executor/layers/quantization/utils/mxfp8_utils.py":
        ("03a16184c905b3a5f0f4aa52736cefeb0528d97fa3d79a85da08d10b68b86c09", "38-69"),
    "vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py":
        ("b9208fdc99be5182c96fefeac24c65eb9b95544ff85ebed97c4646948ba50966", "695-745"),
    "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py":
        ("9ea7e2d2a1030177685daabfa4409e8082a7bc68ff5952cb49af51d122dfc2c7", "111-137"),
    "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py":
        ("8d2f16193042ba2274c9259396107b5956a07158e98791c1780aba7d8ea62fdd", "223-253"),
    "vllm/model_executor/layers/quantization/utils/b12x_moe.py":
        ("40021553fc321f0ebe2f2c92c5f7005ac319723b800805cd151d0b6f29e6091d", "52-63,89-90"),
}
ARITHMETIC = {
    "weights": "Exact E4M3 * 2^(UE8M0-127) per row/32; low-nibble-first E2M1 * E4M3 per16 / weight_global_scale",
    "served_activations": "Dynamic group32 MXFP8; group16 NVFP4 using calibrated input_global_scale, E4M3 scales and E2M1 ties-to-even",
    "nvfp4_scale_policy": "vLLM B12X/CUTLASS per expert; gate/up use min checkpoint input globals; equal gate/up weight globals required; down uses its own globals",
    "boundaries": "FP32 GEMM accumulation and sparse attention; BF16 projection, norm, RoPE, SwiGLU, residual and head outputs; FP32 sigmoid router and expert combine",
    "selection": "BF16 index Q/K; FP32 dot products; causal block max; forced local blocks; stable lower-id ties; ascending selected block ids",
    "limitations": "Format-level QDQ oracle, not bitwise CUDA reduction/rcp.approx or fused norm/RoPE/activation emulation; no FP8 KV/index cache; no FlashInfer global-across-experts scale reduction",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    return {"transformers": {"revision": model.TRANSFORMERS_REVISION, "sha256": model.SOURCE_PINS},
            "vllm": {"revision": VLLM_REVISION,
                     "files": {p: {"sha256": pin, "lines": lines} for p, (pin, lines) in VLLM_SOURCE_PINS.items()}},
            "model_sha256": digest(Path(model.__file__)), "golden_sha256": digest(Path(__file__)),
            "shared_sha256": {name: digest(ROOT / "python/reference" / name)
                              for name in ("shape_invariant.py", "fidelity_windows.py")}}


def admission(weights, cfg, length, layer=None):
    """Reserve packed layer, decoded matrices, query tiles and one live window."""
    if weights.device.type != "cuda":
        return
    import torch
    resident = (weights.layer_bytes(cfg, layer) if layer is not None else
                cfg.vocab_size * cfg.hidden_size * 2 + cfg.hidden_size * 2)
    largest = max(cfg.dense_intermediate_size * cfg.hidden_size,
                  cfg.num_attention_heads * cfg.head_dim * cfg.hidden_size)
    workspace = 12 * largest + length * cfg.hidden_size * 32 + 8 * 2**30
    admit_reference_memory(torch.cuda, resident + workspace)


def run_layers(snapshot, manifest, out, *, device="cpu", activation="served", resume=None,
               stop_after_layer=None, layers=None):
    import torch
    import torch.nn.functional as F
    cfg = model.Config.from_dict(json.loads((snapshot / "config.json").read_bytes()))
    if stop_after_layer is not None and not 0 <= stop_after_layer < cfg.num_hidden_layers:
        raise ValueError("stop layer outside decoder")
    weights = model.Weights(snapshot, device=device)
    weights.validate(cfg)
    windows = manifest["windows"]
    if any(w.get("media") for w in windows):
        raise ValueError("MiniMax reference is text-only")
    shapes = {w["id"]: [len(w["tokens"]), cfg.hidden_size] for w in windows}
    binding = {"set_sha256": manifest["set_sha256"], "source": source_identity(),
               "config_sha256": digest(snapshot / "config.json"),
               "index_sha256": digest(snapshot / "model.safetensors.index.json"),
               "snapshot_revision": snapshot.name, "activation": activation,
               "arithmetic": ARITHMETIC, "torch": torch.__version__, "device": str(weights.device),
               "image_sha256": os.environ.get("CUTEAFD_REFERENCE_IMAGE_SHA256", ""),
               "tokens": {w["id"]: hashlib.sha256(canonical(w["tokens"])).hexdigest() for w in windows},
               "shards": {name: [path.stat().st_size, path.stat().st_mtime_ns]
                          for name in sorted(set(weights.index.values()))
                          for path in [snapshot / name]}}
    if weights.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        binding["capability"] = list(torch.cuda.get_device_capability(weights.device))
    torch.set_float32_matmul_precision("highest")
    out.mkdir(parents=True, exist_ok=True)
    state = LayerCheckpoints(out / "state", binding, shapes, resume=resume, streaming=True)
    times = []
    with torch.inference_mode():
        if state.resumed is not None:
            done, states, times = state.resumed
        else:
            admission(weights, cfg, max(len(w["tokens"]) for w in windows))
            embedding = weights.tensor(model.TEXT + "embed_tokens.weight")
            weights.close()
            folder = state.begin(-1)
            states = []
            for window in windows:
                ids = torch.tensor(window["tokens"], device=weights.device)
                hidden = F.embedding(ids, embedding)
                states.append(state.write_window(folder, window["id"], hidden.cpu().view(torch.uint16).numpy()))
            del embedding, hidden, ids
            weights.close()
            done = -1
        if stop_after_layer is not None and done > stop_after_layer:
            raise ValueError("resume is beyond requested stop layer")
        end = cfg.num_hidden_layers if stop_after_layer is None else stop_after_layer + 1
        for index in range(done + 1, end):
            started, read_before = time.monotonic(), weights.read_bytes
            admission(weights, cfg, max(len(w["tokens"]) for w in windows), index)
            layer = weights.load_layer(index, cfg)
            load_seconds = time.monotonic() - started
            folder = state.begin(index)
            next_states = []
            for window, previous in zip(windows, states, strict=True):
                hidden = torch.from_numpy(previous.load()).view(torch.bfloat16).to(weights.device)
                hidden = model.layer_forward(hidden, layer, cfg, index, activation)
                if not torch.isfinite(hidden).all():
                    raise ValueError(f"non-finite MiniMax hidden at layer {index}")
                bits = hidden.cpu().view(torch.uint16).numpy()
                next_states.append(state.write_window(folder, window["id"], bits))
                if layers and index in layers:
                    target = out / "windows" / window["id"]
                    target.mkdir(parents=True, exist_ok=True)
                    hidden.float().cpu().numpy().astype("<f4").tofile(target / f"layer{index:02d}.bin")
                del hidden, bits
            del layer
            weights.close()
            times.append({"layer": index, "seconds": time.monotonic() - started,
                          "load_seconds": load_seconds, "read_bytes": weights.read_bytes - read_before})
            state.commit(index, next_states, times)
            states = next_states
            if index == 0 and (state.root / "embedding").exists():
                shutil.rmtree(state.root / "embedding")
            print(f"layer {index}: committed; {times[-1]['seconds']:.3f}s", flush=True)
    return states, times


def execute(args):
    import torch
    from shape_invariant import linear, qualify
    started = time.monotonic()
    manifest = load_set(args.windows, "minimax")
    identity = verify_snapshot(manifest, args.snapshot)
    if manifest["checkpoint"] != model.CHECKPOINT or args.snapshot.name != model.REVISION:
        raise ValueError("MiniMax golden requires the pinned release checkpoint")
    cfg = model.Config.from_dict(json.loads((args.snapshot / "config.json").read_bytes()))
    weights = model.Weights(args.snapshot, args.device)
    weights.validate(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        print(json.dumps({"snapshot_identity": identity, "source_pin": source_identity(),
                          "activation": args.activation, "arithmetic": ARITHMETIC}))
        return
    probing = getattr(args, "_prefix_probe", False)
    # A deliberate early stop writes only resumable state, never golden metadata.
    proof = qualify(args, manifest, execute) if args.stop_after_layer is None else None
    if args.prefix_only and not probing:
        return
    states, times = run_layers(args.snapshot, manifest, args.out, device=args.device,
                              activation=args.activation, resume=None if probing else args.resume,
                              stop_after_layer=args.stop_after_layer, layers=args.layers)
    if args.stop_after_layer is not None:
        return
    admission(weights, cfg, max(len(w["tokens"]) for w in manifest["windows"]))
    with torch.inference_mode():
        norm = weights.tensor(model.TEXT + "norm.weight")
        head = weights.tensor(model.HEAD)
        weights.close()
        entries = []
        for window, state in zip(manifest["windows"], states, strict=True):
            hidden = torch.from_numpy(state.load()).view(torch.bfloat16).to(weights.device)
            selected = hidden[window["score_from"] - 1:len(window["tokens"]) - 1]
            logits = linear(model.gemma_norm(selected, norm, cfg.rms_norm_eps), head).float()
            entry = write_scored_logits(args.out, window, logits.cpu().numpy())
            (args.out / entry["path"] / "meta.json").write_bytes(canonical(entry) + b"\n")
            entries.append(entry)
            del hidden, selected, logits
        del head, norm
        weights.close()
    total = sum(e["scored_tokens"] for e in entries)
    finish_golden(args.out, manifest, entries, snapshot_identity=identity,
                  source_pin=source_identity(), prefix_qualification=proof,
                  reference="independent MiniMax M3 text; native weight QDQ; fixed-M128",
                  activation=args.activation, diagnostic_only=args.activation == "bf16",
                  arithmetic=ARITHMETIC, dtype="bf16", attention="eager_fp32",
                  mtp=False, use_cache=False, fixed_rows=128, torch=torch.__version__,
                  nll=sum(e["nll"] * e["scored_tokens"] for e in entries) / total,
                  scored_tokens=total, seconds=time.monotonic() - started, seconds_per_layer=times,
                  platform=({"capability": list(torch.cuda.get_device_capability(weights.device)),
                             "gpu": torch.cuda.get_device_name(weights.device)}
                            if weights.device.type == "cuda" else {"device": "cpu"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "windows", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--activation", choices=("served", "bf16"), default="served")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path, help="previous run's state directory; use a fresh output")
    parser.add_argument("--stop-after-layer", type=int)
    parser.add_argument("--layers", type=int, nargs="*", help="also export selected full layer states")
    parser.add_argument("--prefix-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.prefix_only and args.stop_after_layer is not None:
        parser.error("prefix qualification requires all layers")
    execute(args)


if __name__ == "__main__":
    main()
