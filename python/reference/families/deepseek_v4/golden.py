#!/usr/bin/env python3
"""Golden activations for DeepSeek V4 from the official inference/model.py.

Runs the unmodified reference one Block at a time on one GPU (the full model
does not fit), with kernel.py replaced by kernel_torch. Saves the embedding,
every requested layer's output stream [1, tokens, hc, dim] and the logits.

  golden.py --snapshot SNAP --text "..." --out DIR [--layers 0 1 2 ...] [--device 0]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from types import SimpleNamespace
from pathlib import Path

# Purge freed CPU staging pages immediately on the ARM Torch allocator.
import os
os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shape_invariant import install, qualify
from fidelity_windows import (CheckpointStorage, release_checkpoint, load_set,
                              verify_snapshot, write_scored_logits, finish_golden)

HERE = Path(__file__).resolve().parent


def import_reference(snapshot: Path):
    spec = importlib.util.spec_from_file_location("kernel", HERE / "kernel_torch.py")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    sys.modules["kernel"] = kernel
    fht = types.ModuleType("fast_hadamard_transform")
    fht.hadamard_transform = kernel.hadamard_transform
    sys.modules["fast_hadamard_transform"] = fht
    spec = importlib.util.spec_from_file_location("v4_reference", snapshot / "inference" / "model.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


class Weights:
    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self.files: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        return self.files[shard].get_tensor(name).clone()

    def names(self, prefix: str) -> list[str]:
        return [n for n in self.index if n.startswith(prefix)]


def reference_name(name: str) -> str:
    """The key the official convert.py would give a checkpoint tensor."""
    if name.startswith("model."):
        name = name[len("model."):]
    return (name.replace("self_attn", "attn").replace("mlp", "ffn")
            .replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias"))


def load_module(module: torch.nn.Module, weights: Weights, prefix: str) -> None:
    """Load `prefix`* tensors into `module` with convert.py's transformations:
    wo_a is dequantized to BF16 with its 128x128 scales (and the scale dropped),
    packed FP4 is reinterpreted, everything else is copied (with dtype casts)."""
    params = dict(module.named_parameters())
    missing = set(params)
    by_reference = {reference_name(n): n for n in weights.index}
    for ref, name in by_reference.items():
        if not ref.startswith(prefix):
            continue
        key = ref[len(prefix):]
        if key.endswith("wo_a.scale"):
            continue
        value = weights.get(name)
        if key.endswith("wo_a.weight"):
            scale = weights.get(name[: -len("weight")] + "scale").float()
            value = (value.float().unflatten(0, (-1, 128)).unflatten(-1, (-1, 128))
                     * scale[:, None, :, None]).flatten(2, 3).flatten(0, 1).bfloat16()
        if key not in params:
            raise KeyError(f"checkpoint tensor {name} has no parameter {key}")
        param = params[key]
        if value.dtype != param.dtype and value.element_size() == param.element_size() and value.shape == param.shape:
            value = value.view(param.dtype)  # packed fp4 (I8) and e8m0 bit patterns
        elif value.dtype in (torch.int8, torch.uint8) and param.dtype == torch.float4_e2m1fn_x2:
            value = value.view(torch.float4_e2m1fn_x2)
        with torch.no_grad():
            param.data.copy_(value.to(param.device) if value.dtype == param.dtype else value.to(param.device, param.dtype))
        missing.discard(key)
    if missing:
        raise KeyError(f"{prefix}: parameters without checkpoint tensors: {sorted(missing)[:8]}")
    release_checkpoint(torch.cuda, weights)


def initial_runtime_buffers(block):
    return [(module, name, value.detach().cpu().clone())
            for module in block.modules()
            for name, value in module._buffers.items()
            if name in module._non_persistent_buffers_set and name != "freqs_cis" and value is not None]


def reset_runtime_buffers(initial):
    # Compressor aliases point into these buffers: reset in place, never replace them.
    for module, name, value in initial:
        module._buffers[name].copy_(value)


def run_windows(a, ref, args, weights):
    manifest = load_set(a.windows, "deepseek_v4")
    identity = verify_snapshot(manifest, a.snapshot)
    proof = qualify(a, manifest, lambda probe: run_windows(probe, ref, args, weights))
    if getattr(a, "prefix_only", False) and not getattr(a, "_prefix_probe", False):
        return
    args.max_seq_len = max(256, max(len(w["tokens"]) for w in manifest["windows"]))
    started, rows, times, states = time.time(), [], [], []
    with torch.inference_mode():
        embed = ref.ParallelEmbedding(args.vocab_size, args.dim)
        load_module(embed, weights, "embed.")
        for w in manifest["windows"]:
            ids = torch.tensor([w["tokens"]], dtype=torch.long)
            h = embed(ids).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
            states.append(h.cpu())
        del embed, ids, h
        memory = CheckpointStorage(torch.cuda, weights)
        for layer_id in range(args.n_layers):
            start = time.time()
            block = ref.Block(layer_id, args)
            load_module(block, weights, f"layers.{layer_id}.")
            block.eval()
            initial = initial_runtime_buffers(block)
            for i, w in enumerate(manifest["windows"]):
                reset_runtime_buffers(initial)
                ids = torch.tensor([w["tokens"]], dtype=torch.long)
                h = block(states[i].cuda(), 0, ids)
                if a.layers is not None and layer_id in a.layers:
                    folder = a.out / "windows" / w["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    torch.save(h.cpu(), folder / f"layer{layer_id:02d}.pt")
                states[i] = h.cpu()
                del h, ids
            # Top-k helper lru caches also own CUDA tensors from the last visit.
            ref.get_window_topk_idxs.cache_clear()
            ref.get_compress_topk_idxs.cache_clear()
            del initial, block
            memory.release()
            memory.check(f"layer {layer_id}")
            times.append(time.time() - start)
            print(f"layer {layer_id} {times[-1]:.1f}s ({len(states)} windows)", flush=True)
        hc_fn = weights.get("hc_head_fn").cuda().float()
        hc_scale = weights.get("hc_head_scale").cuda().float()
        hc_base = weights.get("hc_head_base").cuda().float()
        norm = ref.RMSNorm(args.dim, args.norm_eps)
        load_module(norm, weights, "norm.")
        head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        load_module(head, weights, "head.")
        # hc_head needs only scalar config, not an unused full expert layer.
        tail = SimpleNamespace(norm_eps=args.norm_eps, hc_eps=args.hc_eps)
        for i, w in enumerate(manifest["windows"]):
            h = states[i][:, w["score_from"] - 1:len(w["tokens"]) - 1].cuda()
            h = ref.Block.hc_head(tail, h, hc_fn, hc_scale, hc_base)
            logits = head(norm(h), full_logits=True)[0].float()
            rows.append(write_scored_logits(a.out, w, logits.cpu().numpy()))
            states[i] = None
            del h, logits
    finish_golden(a.out, manifest, rows, snapshot=str(a.snapshot),
        reference="official inference/model.py (kernel_torch, fixed-M128 linears)",
        seconds=time.time() - started, seconds_per_layer=times, snapshot_identity=identity,
        prefix_qualification=proof)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--text", help="prompt text (tokenized with the snapshot tokenizer)")
    p.add_argument("--tokens", type=int, nargs="+", help="explicit token ids")
    p.add_argument("--layers", type=int, nargs="*", help="layers whose outputs to save (default all)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--windows", type=Path, help="pinned fidelity set; layer-major scored logits")
    p.add_argument("--prefix-only", action="store_true")
    a = p.parse_args()
    if a.prefix_only and not a.windows:
        p.error("--prefix-only requires --windows")
    if a.windows and (a.text is not None or a.tokens is not None):
        p.error("--windows cannot be combined with legacy text/tokens")

    torch.cuda.set_device(a.device)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    install()
    ref = import_reference(a.snapshot)
    from shape_invariant import bounded_sparse
    ref.sparse_attn = bounded_sparse(ref.sparse_attn)
    config = json.loads((a.snapshot / "inference" / "config.json").read_text())
    if a.windows:
        manifest = load_set(a.windows, "deepseek_v4")
        tokens = max(manifest["windows"], key=lambda w: len(w["tokens"]))["tokens"]
    elif a.tokens:
        tokens = a.tokens
    else:
        from tokenizers import Tokenizer
        tokens = Tokenizer.from_file(str(a.snapshot / "tokenizer.json")).encode(a.text, add_special_tokens=False).ids
    args = ref.ModelArgs(**config, max_batch_size=1, max_seq_len=max(len(tokens), 256))
    # What Transformer.__init__ sets for a single rank.
    ref.world_size, ref.rank = 1, 0
    ref.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    ref.scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
    ref.scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32

    weights = Weights(a.snapshot)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.windows:
        run_windows(a, ref, args, weights)
        return
    ids = torch.tensor([tokens], dtype=torch.long)
    save = set(range(args.n_layers)) if a.layers is None or not a.layers else set(a.layers)
    with torch.inference_mode():
        embed = ref.ParallelEmbedding(args.vocab_size, args.dim)
        load_module(embed, weights, "embed.")
        h = embed(ids)
        torch.save(h.cpu(), a.out / "embed.pt")
        h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        del embed
        memory = CheckpointStorage(torch.cuda, weights)
        for layer in range(args.n_layers):
            start = time.time()
            block = ref.Block(layer, args)
            load_module(block, weights, f"layers.{layer}.")
            block.eval()
            h = block(h, 0, ids)
            if layer in save:
                torch.save(h.cpu(), a.out / f"layer{layer:02d}.pt")
            del block
            memory.release()
            memory.check(f"layer {layer}")
            print(f"layer {layer} {time.time() - start:.1f}s", flush=True)
        hc_fn = weights.get("hc_head_fn").cuda().float()
        hc_scale = weights.get("hc_head_scale").cuda().float()
        hc_base = weights.get("hc_head_base").cuda().float()
        tail = SimpleNamespace(norm_eps=args.norm_eps, hc_eps=args.hc_eps)
        h = ref.Block.hc_head(tail, h, hc_fn, hc_scale, hc_base)
        norm = ref.RMSNorm(args.dim, args.norm_eps)
        load_module(norm, weights, "norm.")
        head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        load_module(head, weights, "head.")
        logits = head(norm(h), full_logits=True)
        torch.save(logits.float().cpu(), a.out / "logits.pt")
    (a.out / "meta.json").write_text(json.dumps({"tokens": tokens, "snapshot": str(a.snapshot),
                                                  "argmax_last": int(logits[0, -1].argmax())}, indent=1))
    print("argmax of last position:", int(logits[0, -1].argmax()))


if __name__ == "__main__":
    main()
