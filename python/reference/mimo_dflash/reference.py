#!/usr/bin/env python3
"""MiMo V2.6 Pro's DFlash block drafter (the snapshot's dflash/ directory) in
plain PyTorch: the oracle the engine's drafter is compared with.

The snapshot's dflash.py is upstream's generic Qwen3 DFlash; it predates this
checkpoint's attention sinks, value scale and partial RoPE. The semantics here
are SGLang's DFlash path for MiMo (models/dflash.py DFlashAttention,
speculative/dflash_worker_v2.py, models/mimo_v2.py aux-hidden capture), in
HF BF16 statement order:

  taps     target layer outputs (pre-norm residual) of target_layer_ids
           [0, 15, 31, 47, 69], concatenated: the golden's layerNN.bin
  context  fused = RMSNorm(fc(taps)); per draft layer K = rope(k_norm(
           k_proj(fused))), V = v_proj(fused) * attention_value_scale
  block    [anchor, mask x 7] at positions p..p+7; the mask token's row is
           dflash/mask_embedding.pt's trained vector (SGLang merges it into
           the target table); per layer: n = RMSNorm(h); q/k per-head RMSNorm
           then NeoX RoPE on the first 64 of 128 dims (partial_rotary_factor
           0.5, theta 1e4); v * value scale; non-causal attention over the
           block and the context positions c with p + r - c < 1024 (sliding
           window 1024), a learned per-head sink logit in the softmax, FP32
           scores, BF16 probabilities; o_proj; h += o; the SwiGLU MLP likewise
  head     drafts = argmax(lm_head(norm(h)))[rows 1..7] with the target head

Qwen3RMSNorm rounds the unit-RMS value to BF16 before the weight; RoPE
rounds cos/sin, each product and the sum to BF16 (transformers).

Writes, for every anchor position p in --positions (context = golden taps of
rows 0..p-1, anchor = tokens[p]):
  drafts.bin   u32 [N, 7]
  unary.bin    f32 [N, 7, 16]  top-16 logits of each drafted row (largest first)
  hidden.bin   bf16 [N, 8, hidden] final-norm output
  meta.json

  reference.py --snapshot SNAP --golden DIR --positions 64:1500:16 --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float, dim: int) -> torch.Tensor:
    """NeoX RoPE on the first `dim` dims of [n, heads, head_dim], BF16 like transformers."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.int64).float() / dim))
    freqs = positions.float()[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos().to(x.dtype)[:, None], emb.sin().to(x.dtype)[:, None]
    rot, keep = x[..., :dim], x[..., dim:]
    half = dim // 2
    rotated = torch.cat([-rot[..., half:], rot[..., :half]], -1)
    return torch.cat([rot * cos + rotated * sin, keep], -1)


class Drafter:
    def __init__(self, snapshot: Path, device: str):
        draft = snapshot / "dflash"
        self.cfg = json.loads((draft / "config.json").read_text())
        dc = self.cfg["dflash_config"]
        self.block, self.taps = dc["block_size"], dc["target_layer_ids"]
        self.v_scale = float(dc.get("attention_value_scale", 1.0))
        self.sinks = bool(dc.get("attention_sink_bias", False))
        self.heads, self.kv_heads = self.cfg["num_attention_heads"], self.cfg["num_key_value_heads"]
        self.head_dim = self.cfg["head_dim"]
        self.rope_dim = int(self.head_dim * self.cfg.get("partial_rotary_factor", 1.0))
        self.theta = float(self.cfg["rope_theta"])
        self.window, self.layers = self.cfg["sliding_window"], self.cfg["num_hidden_layers"]
        self.eps = float(self.cfg["rms_norm_eps"])
        self.device = device
        f = safe_open(str(draft / "dflash_draft_model.safetensors"), framework="pt", device=device)
        self.w = {k: f.get_tensor(k) for k in f.keys()}
        mask = torch.load(draft / "mask_embedding.pt", map_location="cpu", weights_only=True)
        self.mask_token = int(mask["mask_token_id"])
        assert self.mask_token == dc["mask_token_id"]
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        files: dict[str, object] = {}

        def target(name: str) -> torch.Tensor:
            shard = index[name]
            if shard not in files:
                files[shard] = safe_open(str(snapshot / shard), framework="pt", device="cpu")
            return files[shard].get_tensor(name)

        self.embed = target("model.embed_tokens.weight")
        self.embed[self.mask_token] = mask["embedding"].to(self.embed.dtype)
        self.lm_head = target("lm_head.weight").to(device)

    def context(self, taps: torch.Tensor, positions: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        fused = rms(F.linear(taps, self.w["fc.weight"]), self.w["hidden_norm.weight"], self.eps)
        out = []
        for l in range(self.layers):
            p = f"layers.{l}.self_attn."
            k = F.linear(fused, self.w[p + "k_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
            v = F.linear(fused, self.w[p + "v_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
            k = rope(rms(k, self.w[p + "k_norm.weight"], self.eps), positions, self.theta, self.rope_dim)
            out.append((k, v * self.v_scale))
        return out

    def body(self, anchor: int, position: int, ctx: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Final-norm output [block, hidden] (the context holds positions
        position - len(ctx) .. position - 1)."""
        n_ctx = ctx[0][0].shape[0]
        q_pos = torch.arange(position, position + self.block, device=self.device)[:, None]
        k_pos = torch.cat([torch.arange(position - n_ctx, position, device=self.device),
                           torch.arange(position, position + self.block, device=self.device)])[None]
        visible = (q_pos - k_pos) < self.window  # non-causal inside the block
        tokens = torch.tensor([anchor] + [self.mask_token] * (self.block - 1))
        h = self.embed[tokens].to(self.device)
        positions = torch.arange(position, position + self.block, device=self.device)
        group = self.heads // self.kv_heads
        for l in range(self.layers):
            p = f"layers.{l}."
            a = p + "self_attn."
            n = rms(h, self.w[p + "input_layernorm.weight"], self.eps)
            q = F.linear(n, self.w[a + "q_proj.weight"]).view(self.block, self.heads, self.head_dim)
            k = F.linear(n, self.w[a + "k_proj.weight"]).view(self.block, self.kv_heads, self.head_dim)
            v = F.linear(n, self.w[a + "v_proj.weight"]).view(self.block, self.kv_heads, self.head_dim) * self.v_scale
            q = rope(rms(q, self.w[a + "q_norm.weight"], self.eps), positions, self.theta, self.rope_dim)
            k = rope(rms(k, self.w[a + "k_norm.weight"], self.eps), positions, self.theta, self.rope_dim)
            kc, vc = ctx[l]
            keys = torch.cat([kc, k]).repeat_interleave(group, 1)      # [keys, heads, d]
            values = torch.cat([vc, v]).repeat_interleave(group, 1)
            scores = torch.einsum("qhd,khd->hqk", q.float(), keys.float()) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~visible[None], float("-inf"))
            if self.sinks:
                sink = self.w[a + "attention_sink_bias"].float()[:, None, None].expand(-1, self.block, 1)
                scores = torch.cat([scores, sink], -1)
            probs = torch.softmax(scores - scores.amax(-1, keepdim=True), -1).to(torch.bfloat16)
            if self.sinks:
                probs = probs[..., :-1]
            attn = torch.einsum("hqk,khd->qhd", probs.float(), values.float()).to(torch.bfloat16)
            o = F.linear(attn.reshape(self.block, -1), self.w[a + "o_proj.weight"])
            h = h + o
            n = rms(h, self.w[p + "post_attention_layernorm.weight"], self.eps)
            m = p + "mlp."
            act = F.silu(F.linear(n, self.w[m + "gate_proj.weight"])) * F.linear(n, self.w[m + "up_proj.weight"])
            h = h + F.linear(act, self.w[m + "down_proj.weight"])
        return rms(h, self.w["norm.weight"], self.eps)


def parse_positions(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if ":" in part:
            a, b, *s = (int(x) for x in part.split(":"))
            out.extend(range(a, b, s[0] if s else 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True, help="MiMo V2.6 Pro snapshot (with dflash/)")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--positions", required=True, help="anchor positions, e.g. 64:1500:16,1510")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    drafter = Drafter(args.snapshot, args.device)
    tokens = torch.frombuffer(bytearray((args.golden / "tokens.bin").read_bytes()), dtype=torch.int32)
    hidden = drafter.cfg["hidden_size"]
    taps = torch.cat([torch.frombuffer(bytearray((args.golden / f"layer{l:02d}.bin").read_bytes()),
                                       dtype=torch.bfloat16).view(-1, hidden) for l in drafter.taps], -1).to(args.device)
    positions = parse_positions(args.positions)
    with torch.no_grad():
        ctx = drafter.context(taps, torch.arange(taps.shape[0], device=args.device))
        drafts, unary, finals = [], [], []
        for p in positions:
            first = max(0, p - drafter.window)
            h = drafter.body(int(tokens[p]), p, [(k[first:p], v[first:p]) for k, v in ctx])
            logits = F.linear(h[1:], drafter.lm_head).float()
            top = torch.topk(logits, 16, -1)
            drafts.append(top.indices[:, 0].tolist())
            unary.append(top.values.cpu())
            finals.append(h.cpu())
    args.out.mkdir(parents=True, exist_ok=True)
    torch.tensor(drafts, dtype=torch.int32).numpy().tofile(args.out / "drafts.bin")
    torch.stack(unary).numpy().tofile(args.out / "unary.bin")
    torch.stack(finals).view(torch.int16).numpy().tofile(args.out / "hidden.bin")
    truth = [[int(tokens[p + 1 + i]) if p + 1 + i < len(tokens) else -1 for i in range(drafter.block - 1)]
             for p in positions]
    accepted = [next((i for i, (d, t) in enumerate(zip(ds, ts)) if d != t), len(ds)) for ds, ts in zip(drafts, truth)]
    histogram = [accepted.count(i) for i in range(drafter.block)]
    (args.out / "meta.json").write_text(json.dumps({"positions": positions, "accepted_vs_tokens": accepted,
        "mean_accepted": sum(accepted) / len(accepted), "histogram": histogram}, indent=1))
    print(f"{len(positions)} anchors, mean accepted prefix vs golden tokens {sum(accepted) / len(accepted):.2f}, "
          f"histogram {histogram}")


if __name__ == "__main__":
    main()
