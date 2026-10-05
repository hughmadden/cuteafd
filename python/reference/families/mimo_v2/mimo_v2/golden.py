#!/usr/bin/env python3
"""Golden activations for MiMo V2 Flash (mimo_v2_flash) from transformers' reference.

Runs one MiMoV2FlashDecoderLayer at a time on one GPU (the model does not
fit), eager attention with explicit additive masks (causal for the 9 full
layers, causal and 128-token sliding window for the 39 SWA layers, as
``create_sliding_window_causal_mask``: key k is visible to query q when
``q - 128 < k <= q``). FP8 weights are dequantized to BF16 with their FP32
block scales: 128x128 blocks, except the 9 full-attention ``k_proj`` tensors
[768, 4096] whose [8, 32] grid is per KV head (each 192-row head is a 128-row
block then a 64-row block). The router weight and ``e_score_correction_bias`` stay FP32
as stored. Routed experts are summed in FP32 and rounded once, as the
checkpoint's own modeling code does (transformers' DeepseekV3Experts sums in
BF16: +0.015 nats mean NLL on the prompt, and top-1 agreement between the two
is only 88%: this model's sigmoid top-8 routes flip on rounding-level
changes, so compare engines by NLL as well as by agreement). Writes the raw
files the golden commands read:

  tokens.bin      i32  [T]
  layerNN.bin     bf16 [T, hidden]   output of layer NN
  logits.bin      f32  [T, vocab]
  meta.json

  golden.py --snapshot SNAP --text-file prompt.txt --out DIR [--layers 0 1 ...]
            [--requant-ue8m0 {dense,experts,all}]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Purge freed CPU staging pages immediately on the ARM Torch allocator.
import os
os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")

import torch
from safetensors import safe_open

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shape_invariant import install
from fidelity_windows import CheckpointStorage, release_checkpoint


HEAD_DIM = 192


def scale_row_of(rows: int, scale_rows: int, device="cpu") -> torch.Tensor:
    """Scale-grid row of every weight row: 128-row blocks, or per 192-row head
    (blocks of 128 then 64 rows) when the grid has two rows per head."""
    r = torch.arange(rows, device=device)
    if -(-rows // 128) == scale_rows:
        return r // 128
    per_head = -(-HEAD_DIM // 128)
    if rows % HEAD_DIM == 0 and scale_rows == rows // HEAD_DIM * per_head:
        return r // HEAD_DIM * per_head + r % HEAD_DIM // 128
    raise ValueError(f"no block layout maps {rows} rows onto {scale_rows} scale rows")


class Weights:
    # Re-quantize FP8 blocks to power-of-two (UE8M0) scales, the format b12x's
    # FP8 linears and MXFP8 expert kernels take: "dense" = everything but the
    # routed experts, "experts" = routed experts only, "all" = both.
    requant_ue8m0: str | None = None

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self.files: dict[str, object] = {}

    def raw(self, name: str) -> torch.Tensor:
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        return self.files[shard].get_tensor(name).clone()

    def get(self, name: str, device: str = "cuda") -> torch.Tensor:
        """BF16/FP32 tensors as stored; FP8 weights times their block scales."""
        value = self.raw(name).to(device)
        if value.dtype != torch.float8_e4m3fn:
            return value
        scale = self.raw(name.removesuffix("weight") + "weight_scale_inv").to(device).float()
        rows, cols = value.shape
        row_of = scale_row_of(rows, scale.shape[0], device)
        grow = lambda t: t[row_of].repeat_interleave(128, 1)[:, :cols]  # noqa: E731
        values = value.float()
        expert = ".mlp.experts." in name
        mode = self.requant_ue8m0
        if mode == "all" or (mode == "dense" and not expert) or (mode == "experts" and expert):
            # w * s = w' * 2^ceil(log2 s): w' = fp8(w * s / 2^ceil(log2 s)), never larger than w.
            power = torch.exp2(torch.ceil(torch.log2(scale)))
            values = (values * grow(scale / power)).to(torch.float8_e4m3fn).float()
            scale = power
        return (values * grow(scale)).bfloat16()


def checkpoint_name(key: str) -> str:
    """transformers parameter name -> checkpoint tensor name (within a layer)."""
    return {"self_attn.sinks": "self_attn.attention_sink_bias"}.get(key, key)


def load_layer(layer: torch.nn.Module, weights: Weights, prefix: str) -> None:
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    loaded = set()
    if experts is not None:
        count = experts.gate_up_proj.shape[0]
        for e in range(count):
            base = f"{prefix}mlp.experts.{e}."
            with torch.no_grad():
                experts.gate_up_proj[e].copy_(torch.cat([weights.get(base + "gate_proj.weight"),
                                                         weights.get(base + "up_proj.weight")], 0))
                experts.down_proj[e].copy_(weights.get(base + "down_proj.weight"))
        loaded |= {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    for key, param in layer.named_parameters():
        if key in loaded:
            continue
        name = prefix + checkpoint_name(key)
        if name not in weights.index:
            raise KeyError(f"{name}: no checkpoint tensor")
        with torch.no_grad():
            param.copy_(weights.get(name).to(param.dtype))
    for key, buffer in layer.named_buffers():
        name = prefix + key
        if name in weights.index:
            with torch.no_grad():
                buffer.copy_(weights.get(name).to(buffer.dtype))
    release_checkpoint(torch.cuda, weights)


def experts_fp32(self, hidden_states, top_k_index, top_k_weights):
    """DeepseekV3Experts.forward with the routed sum in FP32 (one BF16 rounding)."""
    final = torch.zeros_like(hidden_states, dtype=torch.float32)
    mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
    for expert in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        expert = expert[0]
        slot, token = torch.where(mask[expert])
        gate, up = torch.nn.functional.linear(hidden_states[token], self.gate_up_proj[expert]).chunk(2, dim=-1)
        out = torch.nn.functional.linear(self.act_fn(gate) * up, self.down_proj[expert])
        final.index_add_(0, token, out * top_k_weights[token, slot, None])
    return final.to(hidden_states.dtype)


def masks(t: int, window: int, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Additive BF16 masks [1, 1, T, T] as transformers' eager path adds them."""
    q = torch.arange(t, device=device)[:, None]
    k = torch.arange(t, device=device)[None, :]
    low = torch.finfo(torch.bfloat16).min
    causal = k <= q
    sliding = causal & (k > q - window)
    make = lambda keep: torch.where(keep, 0.0, low).to(torch.bfloat16)[None, None]  # noqa: E731
    return {"full_attention": make(causal), "sliding_attention": make(sliding)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--text", help="prompt text (tokenized with the snapshot tokenizer)")
    p.add_argument("--text-file", type=Path, help="prompt text from a file")
    p.add_argument("--layers", type=int, nargs="*", help="layers whose outputs to save (default all)")
    p.add_argument("--stop-after", type=int, help="run only layers 0..N (no logits)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--requant-ue8m0", choices=("dense", "experts", "all"),
                   help="re-quantize FP8 blocks to power-of-two scales (b12x FP8 formats)")
    a = p.parse_args()

    from tokenizers import Tokenizer
    from transformers import AutoConfig
    from transformers.models.mimo_v2_flash import modeling_mimo_v2_flash as ref

    ref.MiMoV2FlashExperts.forward = experts_fp32
    torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    install()
    config = AutoConfig.from_pretrained(a.snapshot)
    config._attn_implementation = "eager"
    text = a.text_file.read_text() if a.text_file else a.text
    tokens = Tokenizer.from_file(str(a.snapshot / "tokenizer.json")).encode(text, add_special_tokens=False).ids
    weights = Weights(a.snapshot)
    weights.requant_ue8m0 = a.requant_ue8m0
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tokens.bin").write_bytes(torch.tensor(tokens, dtype=torch.int32).numpy().tobytes())
    layers = config.num_hidden_layers if a.stop_after is None else min(a.stop_after + 1, config.num_hidden_layers)
    save = set(range(layers)) if not a.layers else set(a.layers)
    ids = torch.tensor([tokens], device="cuda")
    positions = torch.arange(len(tokens), device="cuda")[None]
    mask = masks(len(tokens), config.sliding_window)
    with torch.inference_mode():
        h = torch.nn.functional.embedding(ids, weights.get("model.embed_tokens.weight"))
        rotary = ref.MiMoV2FlashRotaryEmbedding(config=config).cuda()
        cos_sin = {kind: rotary(h, positions, layer_type=kind) for kind in set(config.layer_types)}
        memory = CheckpointStorage(torch.cuda, weights)
        for layer_id in range(layers):
            start = time.time()
            kind = config.layer_types[layer_id]
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                layer = ref.MiMoV2FlashDecoderLayer(config, layer_id)
            torch.set_default_dtype(torch.float32)
            layer = layer.to_empty(device="cuda")
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None:
                # The checkpoint stores the router weight and bias in FP32.
                gate.float()
            load_layer(layer, weights, f"model.layers.{layer_id}.")
            h = layer(h, attention_mask=mask[kind], position_ids=positions, position_embeddings=cos_sin[kind])
            if layer_id in save:
                (a.out / f"layer{layer_id:02d}.bin").write_bytes(
                    h[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
            del layer
            memory.release()
            memory.check(f"layer {layer_id}")
            print(f"layer {layer_id} ({kind}) {time.time() - start:.1f}s", flush=True)
        if layers < config.num_hidden_layers:
            return
        norm = ref.MiMoV2FlashRMSNorm(config.hidden_size, config.rms_norm_eps).cuda().to(torch.bfloat16)
        norm.weight.copy_(weights.get("model.norm.weight"))
        logits = torch.nn.functional.linear(norm(h).float(), weights.get("lm_head.weight").float())
        (a.out / "logits.bin").write_bytes(logits[0].contiguous().cpu().numpy().tobytes())
    argmax = logits[0].argmax(-1)
    next_ok = (argmax[:-1] == ids[0, 1:]).float().mean().item()
    nll = -torch.log_softmax(logits[0, :-1].double(), -1).gather(1, ids[0, 1:, None]).mean().item()
    (a.out / "meta.json").write_text(json.dumps({
        "tokens": len(tokens), "snapshot": str(a.snapshot), "reference": "transformers mimo_v2_flash (eager)",
        "requant_ue8m0": a.requant_ue8m0, "argmax_last": int(argmax[-1]), "next_token_accuracy": next_ok,
        "mean_nll": nll,
    }, indent=1))
    print(f"argmax of last position: {int(argmax[-1])}; next-token accuracy {next_ok:.3f}; mean NLL {nll:.4f}")


if __name__ == "__main__":
    main()
