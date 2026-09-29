#!/usr/bin/env python3
"""Golden activations for GLM 5.x (glm_moe_dsa) from transformers' reference.

Runs one GlmMoeDsaDecoderLayer at a time on one GPU (the model does not fit),
eager attention with the DSA top-k mask, FP8 weights dequantized to BF16 with
their 128x128 FP32 block scales. Writes the raw files dsv4-golden style
harnesses read:

  tokens.bin      u32 [T]
  layerNN.bin     bf16 [T, hidden]   output of layer NN
  logits.bin      f32  [T, vocab]
  meta.json

  golden.py --snapshot SNAP --text "..." --out DIR [--layers 0 1 2 ...] [--device 0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open


class Weights:
    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self.files: dict[str, object] = {}

    def raw(self, name: str) -> torch.Tensor:
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        return self.files[shard].get_tensor(name)

    def get(self, name: str, device: str = "cuda") -> torch.Tensor:
        """BF16/FP32 tensors as stored; FP8 weights times their block scales."""
        value = self.raw(name).to(device)
        if value.dtype != torch.float8_e4m3fn:
            return value
        scale = self.raw(name.removesuffix("weight") + "weight_scale_inv").to(device).float()
        rows, cols = value.shape
        expanded = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
        return (value.float() * expanded).bfloat16()


def load_layer(layer: torch.nn.Module, weights: Weights, prefix: str) -> None:
    params = dict(layer.named_parameters())
    loaded = set()
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    if experts is not None:
        count = experts.gate_up_proj.shape[0]
        for e in range(count):
            base = f"{prefix}mlp.experts.{e}."
            with torch.no_grad():
                experts.gate_up_proj[e].copy_(torch.cat([weights.get(base + "gate_proj.weight"),
                                                         weights.get(base + "up_proj.weight")], 0))
                experts.down_proj[e].copy_(weights.get(base + "down_proj.weight"))
        loaded |= {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    for key, param in params.items():
        if key in loaded:
            continue
        name = prefix + key
        if name not in weights.index:
            raise KeyError(f"{name}: no checkpoint tensor")
        with torch.no_grad():
            param.copy_(weights.get(name).to(param.dtype))
        loaded.add(key)
    for key, buffer in layer.named_buffers():
        name = prefix + key
        if name in weights.index:
            with torch.no_grad():
                buffer.copy_(weights.get(name).to(buffer.dtype))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--text", help="prompt text (tokenized with the snapshot tokenizer)")
    p.add_argument("--text-file", type=Path, help="prompt text from a file")
    p.add_argument("--layers", type=int, nargs="*", help="layers whose outputs to save (default all)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()

    from tokenizers import Tokenizer
    from transformers import AutoConfig
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as ref

    torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = AutoConfig.from_pretrained(a.snapshot)
    config._attn_implementation = "eager"
    text = a.text_file.read_text() if a.text_file else a.text
    tokens = Tokenizer.from_file(str(a.snapshot / "tokenizer.json")).encode(text, add_special_tokens=False).ids
    weights = Weights(a.snapshot)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tokens.bin").write_bytes(torch.tensor(tokens, dtype=torch.int32).numpy().tobytes())
    layers = config.num_hidden_layers
    save = set(range(layers)) if not a.layers else set(a.layers)
    ids = torch.tensor([tokens], device="cuda")
    positions = torch.arange(len(tokens), device="cuda")[None]
    with torch.inference_mode():
        h = torch.nn.functional.embedding(ids, weights.get("model.embed_tokens.weight"))
        rotary = ref.GlmMoeDsaRotaryEmbedding(config=config).cuda()
        cos_sin = rotary(h, position_ids=positions)
        topk = None
        for layer_id in range(layers):
            start = time.time()
            # BF16 parameters; the router keeps its explicit FP32 bias buffer.
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                layer = ref.GlmMoeDsaDecoderLayer(config, layer_id)
            torch.set_default_dtype(torch.float32)
            layer = layer.to_empty(device="cuda")
            load_layer(layer, weights, f"model.layers.{layer_id}.")
            h, topk = layer(h, attention_mask=None, position_ids=positions, position_embeddings=cos_sin,
                            prev_topk_indices=topk)
            if layer_id in save:
                (a.out / f"layer{layer_id:02d}.bin").write_bytes(h[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
            del layer
            torch.cuda.empty_cache()
            print(f"layer {layer_id} {time.time() - start:.1f}s", flush=True)
        norm = ref.GlmMoeDsaRMSNorm(config.hidden_size, config.rms_norm_eps).cuda()
        norm.weight.copy_(weights.get("model.norm.weight"))
        logits = norm(h).float() @ weights.get("lm_head.weight").float().T
        (a.out / "logits.bin").write_bytes(logits[0].contiguous().cpu().numpy().tobytes())
    argmax = logits[0].argmax(-1)
    next_ok = (argmax[:-1] == ids[0, 1:]).float().mean().item()
    (a.out / "meta.json").write_text(json.dumps({"tokens": len(tokens), "snapshot": str(a.snapshot),
        "argmax_last": int(argmax[-1]), "next_token_accuracy": next_ok}, indent=1))
    print(f"argmax of last position: {int(argmax[-1])}; next-token accuracy {next_ok:.3f}")


if __name__ == "__main__":
    main()
