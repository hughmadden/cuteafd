#!/usr/bin/env python3
"""Golden logits for DeepSeek V4.1 from the official inference/model.py.

Runs the unmodified reference one Block at a time on one GPU (the full model
does not fit), with kernel.py replaced by kernel_torch and the two engram
tables (~98 GB each) read row by row from the safetensors shards through a
memory map. Writes tokens.bin (i32 [T]), logits.bin (f32 [T, vocab]) and
meta.json with the mean NLL: the format the other families' goldens use
(scripts/bench/make-fidelity-reference.py reads it).

  golden.py --snapshot SNAP --text-file prompt.txt --max-tokens 640 --out DIR [--device 1]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fidelity_windows import load_set, write_scored_logits, finish_golden, verify_snapshot


def initial_runtime_buffers(layer):
    # Non-persistent buffers are sequence state, except immutable RoPE frequencies.
    return [(module, name, value.detach().cpu().clone())
            for module in layer.modules()
            for name, value in module._buffers.items()
            if name in module._non_persistent_buffers_set and name != "freqs_cis" and value is not None]


def reset_runtime_buffers(buffers, layer, ref):
    for module, name, initial in buffers:
        module._buffers[name].copy_(initial)
    for module in layer.modules():
        if isinstance(module, ref.Indexer):
            module.freqs_cis = None


def stage_shared_attention(shared):
    return {name: value.detach().cpu().clone() if value is not None else None
            for name, value in vars(shared).items()}


def restore_shared_attention(ref, state, device):
    # Each window owns the source KV/index/candidate rows handed between layers.
    shared = ref.SharedAttentionRuntime()
    for name, value in state.items():
        setattr(shared, name, value.to(device) if value is not None else None)
    ref.shared_attn = shared


def run_windows(a, ref, args, backend, weights):
    manifest = load_set(a.windows, "deepseek_v41")
    identity = verify_snapshot(manifest, a.snapshot)
    args.max_seq_len = max(len(w["tokens"]) for w in manifest["windows"])
    started, rows = time.time(), []
    layout = ref.EngramLayout.from_args(args)

    class Wrapped:
        backend_tokenizer = backend

        def __len__(self):
            return backend.get_vocab_size(with_added_tokens=True)

    # Stage streams on CPU between window visits, not all attention masks on GPU.
    # Load each layer once, but isolate both module buffers and cross-layer state.
    states = []
    shared_states = [{} for _ in manifest["windows"]]
    with torch.inference_mode():
        embed = ref.ParallelEmbedding(args.vocab_size, args.dim)
        load_module(embed, weights, "embed.")
        for w in manifest["windows"]:
            ids = torch.tensor([w["tokens"]], dtype=torch.long)
            hashes = ref.NgramHashState(args, layout, Wrapped())(ids, 0) if layout is not None else None
            h = embed(ids).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
            states.append((h.cpu(), ref.make_identity_pre_mix(h, args.hc_mult).cpu(),
                           hashes.cpu() if hashes is not None else None))
            del h, ids, hashes
        del embed
        times = []
        for layer_id in range(args.n_layers):
            start = time.time()
            layer = ref.Block(layer_id, args, layout)
            load_module(layer, weights, f"layers.{layer_id}.")
            initial = initial_runtime_buffers(layer)
            for i, w in enumerate(manifest["windows"]):
                reset_runtime_buffers(initial, layer, ref)
                host_h, host_mix, hashes = states[i]
                h, pre_mix = host_h.cuda(), host_mix.cuda()
                restore_shared_attention(ref, shared_states[i], h.device)
                if layer.engram is not None:
                    h = layer.engram(h, hashes.cuda()[:, :, layer.engram.layer_hash_index, :], None)
                h, pre_mix = layer(h, 0, pre_mix, None)
                shared_states[i] = stage_shared_attention(ref.shared_attn)
                if a.layers is not None and layer_id in a.layers:
                    folder = a.out / "windows" / w["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    h[0].contiguous().view(torch.int16).cpu().numpy().tofile(folder / f"layer{layer_id:02d}.bin")
                if layer_id == args.n_layers - 1:
                    # Collapse only the hidden rows predicting scored token positions.
                    first, last = w["score_from"] - 1, len(w["tokens"]) - 1
                    h = layer.hc_pre(h[:, first:last], pre_mix[:, first:last])
                states[i] = (h.cpu(), pre_mix.cpu(), hashes)
                del h, pre_mix
            ref.shared_attn = ref.SharedAttentionRuntime()
            del initial, layer
            torch.cuda.empty_cache()
            times.append(time.time() - start)
            print(f"layer {layer_id}: {times[-1]:.1f}s ({len(states)} windows)", flush=True)
        norm = ref.RMSNorm(args.dim, args.norm_eps)
        load_module(norm, weights, "norm.")
        head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        load_module(head, weights, "head.")
        for i, w in enumerate(manifest["windows"]):
            h = states[i][0].cuda()
            logits = head(norm(h), full_logits=True)[0].float()
            rows.append(write_scored_logits(a.out, w, logits.cpu().numpy()))
            states[i] = None
            del h, logits
    finish_golden(a.out, manifest, rows, snapshot=str(a.snapshot),
        reference="official inference/model.py (kernel_torch, mapped engram tables)",
        seconds=time.time() - started, seconds_per_layer=times, snapshot_identity=identity)


def import_reference(snapshot: Path):
    spec = importlib.util.spec_from_file_location("kernel", HERE / "kernel_torch.py")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    sys.modules["kernel"] = kernel
    sys.path.insert(0, str(snapshot / "inference"))
    spec = importlib.util.spec_from_file_location("v41_reference", snapshot / "inference" / "model.py")
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
        return self.files[shard].get_tensor(name)

    def mapped(self, name: str) -> np.memmap:
        """A uint8 memory map of tensor `name` shaped [rows, bytes per row]."""
        path = self.snapshot / self.index[name]
        with open(path, "rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(size))
        entry = header[name]
        begin, end = entry["data_offsets"]
        rows = entry["shape"][0]
        return np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size + begin, shape=(rows, (end - begin) // rows))


def mapped_engram_embedding(base):
    """ParallelEngramEmbedding without the table on the GPU: rows are gathered
    from the memory-mapped shard on lookup, then dequantized as the reference does."""

    class MappedEngramEmbedding(torch.nn.Module):
        def __init__(self, num_embeddings: int, dim: int):
            super().__init__()
            self.num_embeddings, self.dim, self.block_size = num_embeddings, dim, base.fp8_block_size
            self.weight_map = self.scale_map = None

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            flat = indices.reshape(-1).cpu().numpy()
            unique, inverse = np.unique(flat, return_inverse=True)
            weight = torch.from_numpy(np.ascontiguousarray(self.weight_map[unique])).view(torch.float8_e4m3fn)
            scale = torch.from_numpy(np.ascontiguousarray(self.scale_map[unique])).view(base.scale_dtype)
            values = weight.cuda().float().unflatten(-1, (-1, self.block_size)) * scale.cuda().float().unsqueeze(-1)
            values = values.flatten(-2).to(torch.bfloat16)
            return values[torch.from_numpy(inverse).cuda()].reshape(*indices.shape, self.dim)

    return MappedEngramEmbedding


def load_module(module: torch.nn.Module, weights: Weights, prefix: str) -> None:
    """Load `prefix`* tensors into `module` with convert.py's transformations:
    wo_a dequantized to BF16 with its block scales, packed FP4 reinterpreted,
    engram tables memory-mapped, everything else copied (with dtype casts)."""
    params = dict(module.named_parameters())
    missing = set(params)
    for name in weights.index:
        if not name.startswith(prefix):
            continue
        key = name[len(prefix):]
        if key.endswith("wo_a.scale"):
            continue
        if ".engram.embed." in "." + key:
            owner = module.engram.embed
            if key.endswith("weight"):
                owner.weight_map = weights.mapped(name)
            else:
                owner.scale_map = weights.mapped(name)
            continue
        value = weights.get(name)
        if key.endswith("wo_a.weight"):
            scale = weights.get(name[: -len("weight")] + "scale").float()
            ob, ib = value.shape[0] // scale.shape[0], value.shape[1] // scale.shape[1]
            value = (value.float().unflatten(0, (-1, ob)).unflatten(-1, (-1, ib))
                     * scale[:, None, :, None]).flatten(2, 3).flatten(0, 1).bfloat16()
        if key not in params:
            raise KeyError(f"checkpoint tensor {name} has no parameter {key}")
        param = params[key]
        if value.dtype != param.dtype and value.element_size() == param.element_size() and value.shape == param.shape:
            value = value.view(param.dtype)  # e8m0 bit patterns
        elif value.dtype in (torch.int8, torch.uint8) and param.dtype == torch.float4_e2m1fn_x2:
            value = value.view(torch.float4_e2m1fn_x2)
        with torch.no_grad():
            param.data.copy_(value.to(param.device) if value.dtype == param.dtype else value.to(param.device, param.dtype))
        missing.discard(key)
    if missing:
        raise KeyError(f"{prefix}: parameters without checkpoint tensors: {sorted(missing)[:8]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--text-file", type=Path, help="prompt text (tokenized with the snapshot tokenizer)")
    p.add_argument("--tokens", type=int, nargs="+", help="explicit token ids")
    p.add_argument("--windows", type=Path, help="pinned fidelity set; scored-row logits only")
    p.add_argument("--layers", type=int, nargs="*", help="window-mode streams to save; default none")
    p.add_argument("--max-tokens", type=int, default=0, help="truncate the prompt to this many tokens")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()

    torch.cuda.set_device(a.device)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = import_reference(a.snapshot)
    config = json.loads((a.snapshot / "inference" / "config.json").read_text())
    from tokenizers import Tokenizer
    backend = Tokenizer.from_file(str(a.snapshot / "tokenizer.json"))
    if a.windows and (a.tokens or a.text_file or a.max_tokens):
        p.error("--windows cannot be combined with legacy token/text/truncation options")
    manifest = load_set(a.windows, "deepseek_v41") if a.windows else None
    tokens = manifest["windows"][0]["tokens"] if manifest else a.tokens or backend.encode(a.text_file.read_text(), add_special_tokens=False).ids
    if a.max_tokens:
        tokens = tokens[: a.max_tokens]
    args = ref.ModelArgs(**config)
    args.max_batch_size, args.max_seq_len = 1, max(len(tokens), 256)
    # What Transformer.__init__ sets for a single rank.
    ref.world_size, ref.rank = 1, 0
    ref.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    ref.ParallelEngramEmbedding = mapped_engram_embedding(ref)

    class Wrapped:
        backend_tokenizer = backend

        def __len__(self):
            return backend.get_vocab_size(with_added_tokens=True)

    weights = Weights(a.snapshot)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.windows:
        run_windows(a, ref, args, backend, weights)
        return
    ids = torch.tensor([tokens], dtype=torch.long)
    started = time.time()
    with torch.inference_mode():
        layout = ref.EngramLayout.from_args(args)
        engram_hashes = ref.NgramHashState(args, layout, Wrapped())(ids, 0) if layout is not None else None
        embed = ref.ParallelEmbedding(args.vocab_size, args.dim)
        load_module(embed, weights, "embed.")
        h = embed(ids).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        del embed
        pre_mix = ref.make_identity_pre_mix(h, args.hc_mult)
        layer = None
        for layer_id in range(args.n_layers):
            start = time.time()
            layer = ref.Block(layer_id, args, layout)
            load_module(layer, weights, f"layers.{layer_id}.")
            if layer.engram is not None:
                h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], None)
            h, pre_mix = layer(h, 0, pre_mix, None)
            if layer_id < args.n_layers - 1:
                del layer
            torch.cuda.empty_cache()
            print(f"layer {layer_id} {time.time() - start:.1f}s", flush=True)
        h = layer.hc_pre(h, pre_mix)
        norm = ref.RMSNorm(args.dim, args.norm_eps)
        load_module(norm, weights, "norm.")
        head = ref.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        load_module(head, weights, "head.")
        logits = head(norm(h), full_logits=True)[0].float()
    np.asarray(tokens, dtype=np.int32).tofile(a.out / "tokens.bin")
    logits.cpu().numpy().astype(np.float32).tofile(a.out / "logits.bin")
    logp = torch.log_softmax(logits.double(), -1)
    target = torch.tensor(tokens[1:], device=logp.device)
    nll = -logp[:-1].gather(1, target[:, None]).mean().item()
    accuracy = (logits[:-1].argmax(-1) == target).float().mean().item()
    meta = {"tokens": tokens, "snapshot": str(a.snapshot),
            "reference": "official inference/model.py (kernel_torch, mapped engram tables)",
            "argmax_last": int(logits[-1].argmax()), "next_token_accuracy": accuracy, "mean_nll": nll,
            "seconds": time.time() - started}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(tokens)} tokens, mean NLL {nll:.4f}, next-token accuracy {accuracy:.3f}, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
