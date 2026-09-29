#!/usr/bin/env python3
"""DFlash2 drafter for GLM 5.3 (incoai/GLM-5.3-DFlash2) in plain PyTorch.

The oracle the engine's drafter is compared with. Semantics are those of the
upstream DFlash2DraftModel as glmrt reproduced them bit for bit
(python/reference/glmrt_reference/{dspark_update,dspark_body,dflash_head}_capture.py
in ../glmrt-release), written as BF16 torch statements:

  context  fused = RMSNorm(fc(concat(target layer outputs 5,19,33,47,61,75)))
           per draft layer: K = rope(k_norm(k_proj(fused))), V = v_proj(fused)
  block    [anchor, mask x 7] embeddings at positions p..p+7; per layer:
           n = RMSNorm(h); dyn = attention_conv.kernel_projection(n)
           qkv(conv0(n)) -> q/k norm, rope; non-causal attention over the
           last <= 2048 context entries and the whole block; o_proj;
           h += conv1(o); n = RMSNorm(h); the MLP likewise with mlp_conv.
           conv_s(x)[r] = base[s,0]*x[r] + dyn[r,s,0]*x[r] + base[s,1]*x[r-1]
           + dyn[r,s,1]*x[r-1] (per 16-channel group, x[-1] = 0 in the block),
           each term accumulated in BF16 as upstream does.
  head     logits = lm_head(norm(h))[1:8]; top-16 per row; the selector walks
           the rows greedily: score_k = unary_k + ((pred_cb[prev] * proj(h)) .
           succ_cb[cand_k]) in BF16, prev = anchor, then each chosen token.

Writes, for every anchor position p in --positions (context = golden taps of
rows 0..p-1, anchor = tokens[p]):
  drafts.bin    u32 [N, 7]
  features.bin  f32 [N, 7, 4]  margin, best probability, entropy, rank among the 16
  hidden.bin    bf16 [N, 8, 6144] final-norm output
  meta.json

  reference.py --draft SNAP --target SNAP --golden DIR --positions 64:1500:16 --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

EPS = 1e-5


def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Qwen3RMSNorm: the unit-RMS value is rounded to BF16 before the weight."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return w * xf.to(x.dtype)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """rotate_half RoPE on [n, heads, dim] with BF16 cos/sin, as transformers."""
    dim = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.int64).float() / dim))
    freqs = positions.float()[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos().to(x.dtype)[:, None], emb.sin().to(x.dtype)[:, None]
    half = dim // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos + rotated * sin


def conv(x: torch.Tensor, dyn: torch.Tensor, base: torch.Tensor, side: int, group: int) -> torch.Tensor:
    """Two-tap grouped dynamic convolution over block rows ([B, H], row 0 has no predecessor)."""
    groups = x.shape[-1] // group
    d0 = dyn[:, side * 2 * groups:side * 2 * groups + groups].repeat_interleave(group, -1)
    d1 = dyn[:, side * 2 * groups + groups:(side + 1) * 2 * groups].repeat_interleave(group, -1)
    prev = torch.zeros_like(x)
    prev[1:] = x[:-1]
    out = base[side, 0] * x
    out = torch.addcmul(out, d0, x)
    out = out + base[side, 1] * prev
    return torch.addcmul(out, d1, prev)


class Drafter:
    def __init__(self, draft: Path, target: Path, device: str):
        self.cfg = json.loads((draft / "config.json").read_text())
        dc = self.cfg["dflash_config"]
        self.block, self.mask, self.taps = dc["block_size"], dc["mask_token_id"], dc["target_layer_ids"]
        self.group, self.topk = dc["conv_group_size"], dc["selector_top_k"]
        self.heads, self.kv_heads = self.cfg["num_attention_heads"], self.cfg["num_key_value_heads"]
        self.head_dim, self.theta = self.cfg["head_dim"], float(self.cfg["rope_parameters"]["rope_theta"])
        self.window, self.layers = self.cfg["sliding_window"], self.cfg["num_hidden_layers"]
        self.device = device
        f = safe_open(str(draft / "model.safetensors"), framework="pt", device=device)
        self.w = {k: f.get_tensor(k) for k in f.keys()}
        index = json.loads((target / "model.safetensors.index.json").read_text())["weight_map"]
        self.target_files = {}

        def target_tensor(name: str) -> torch.Tensor:
            shard = index[name]
            if shard not in self.target_files:
                self.target_files[shard] = safe_open(str(target / shard), framework="pt", device="cpu")
            return self.target_files[shard].get_tensor(name)

        self.embed = target_tensor("model.embed_tokens.weight")
        self.lm_head = target_tensor("lm_head.weight").to(device)

    def context(self, taps: torch.Tensor, positions: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer context K (roped) and V for tap rows [n, 6 * hidden]."""
        fused = rms(F.linear(taps, self.w["fc.weight"]), self.w["hidden_norm.weight"])
        out = []
        for l in range(self.layers):
            p = f"layers.{l}.self_attn."
            k = F.linear(fused, self.w[p + "k_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
            v = F.linear(fused, self.w[p + "v_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
            out.append((rope(rms(k, self.w[p + "k_norm.weight"]), positions, self.theta), v))
        return out

    def body(self, anchor: int, position: int, ctx: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Final-norm output [block, hidden] of one block at `position`."""
        tokens = torch.tensor([anchor] + [self.mask] * (self.block - 1))
        h = self.embed[tokens].to(self.device)
        positions = torch.arange(position, position + self.block, device=self.device)
        n = rms(h, self.w["layers.0.input_layernorm.weight"])
        for l in range(self.layers):
            p = f"layers.{l}."
            dyn = F.linear(n, self.w[p + "attention_conv.kernel_projection.weight"])
            base = self.w[p + "attention_conv.base_kernel"]
            x = conv(n, dyn, base, 0, self.group)
            a = p + "self_attn."
            q = F.linear(x, self.w[a + "q_proj.weight"]).view(self.block, self.heads, self.head_dim)
            k = F.linear(x, self.w[a + "k_proj.weight"]).view(self.block, self.kv_heads, self.head_dim)
            v = F.linear(x, self.w[a + "v_proj.weight"]).view(self.block, self.kv_heads, self.head_dim)
            q = rope(rms(q, self.w[a + "q_norm.weight"]), positions, self.theta)
            k = rope(rms(k, self.w[a + "k_norm.weight"]), positions, self.theta)
            kc, vc = ctx[l]
            keys, values = torch.cat([kc, k]), torch.cat([vc, v])
            attn = F.scaled_dot_product_attention(q.transpose(0, 1)[None], keys.transpose(0, 1)[None],
                values.transpose(0, 1)[None], is_causal=False, enable_gqa=True, scale=1.0 / math.sqrt(self.head_dim))
            attn = attn[0].transpose(0, 1).reshape(self.block, -1)
            o = F.linear(attn, self.w[a + "o_proj.weight"])
            h = h + conv(o, dyn, base, 1, self.group)
            n = rms(h, self.w[p + "post_attention_layernorm.weight"])
            dyn = F.linear(n, self.w[p + "mlp_conv.kernel_projection.weight"])
            base = self.w[p + "mlp_conv.base_kernel"]
            x = conv(n, dyn, base, 0, self.group)
            m = p + "mlp."
            act = F.silu(F.linear(x, self.w[m + "gate_proj.weight"])) * F.linear(x, self.w[m + "up_proj.weight"])
            d = F.linear(act, self.w[m + "down_proj.weight"])
            h = h + conv(d, dyn, base, 1, self.group)
            last = l + 1 == self.layers
            n = rms(h, self.w["norm.weight" if last else f"layers.{l + 1}.input_layernorm.weight"])
        return n

    def select(self, anchor: int, hidden: torch.Tensor) -> tuple[list[int], list[list[float]]]:
        """Greedy candidate path over rows 1.. of the block."""
        rows = hidden[1:]
        logits = F.linear(rows, self.lm_head)
        unary, candidates = torch.topk(logits, self.topk, dim=-1)
        projected = F.linear(rows, self.w["candidate_selector.hidden_projection.weight"])
        pred_cb = self.w["candidate_selector.predecessor_codebook"]
        succ_cb = self.w["candidate_selector.successor_codebook"]
        previous, tokens, features = anchor, [], []
        for i in range(rows.shape[0]):
            conditioned = pred_cb[previous] * projected[i]
            transition = torch.einsum("r,kr->k", conditioned, succ_cb[candidates[i]])
            scores = (unary[i] + transition).float()
            best = scores.max()
            index = int((scores == best).nonzero()[0])
            runner = torch.where(torch.arange(self.topk, device=scores.device) == index,
                torch.tensor(-math.inf, device=scores.device), scores).max()
            shifted = scores - best
            mass = shifted.exp()
            total = mass.sum()
            entropy = total.log() - (mass * shifted).sum() / total
            features.append([float(best - runner), float(1.0 / total), float(entropy), float(index)])
            previous = int(candidates[i, index])
            tokens.append(previous)
        return tokens, features


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
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--positions", required=True, help="anchor positions, e.g. 64:1500:16,1510")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    drafter = Drafter(args.draft, args.target, args.device)
    tokens = torch.frombuffer(bytearray((args.golden / "tokens.bin").read_bytes()), dtype=torch.int32)
    hidden_size = drafter.cfg["hidden_size"]
    taps = []
    for layer in drafter.taps:
        raw = torch.frombuffer(bytearray((args.golden / f"layer{layer:02d}.bin").read_bytes()), dtype=torch.bfloat16)
        taps.append(raw.view(-1, hidden_size))
    taps = torch.cat(taps, -1).to(args.device)
    positions = parse_positions(args.positions)
    rows = torch.arange(taps.shape[0], device=args.device)
    with torch.no_grad():
        ctx = drafter.context(taps, rows)
        drafts, features, hidden = [], [], []
        for p in positions:
            first = max(0, p - drafter.window)
            window = [(k[first:p], v[first:p]) for k, v in ctx]
            h = drafter.body(int(tokens[p]), p, window)
            t, f = drafter.select(int(tokens[p]), h)
            drafts.append(t)
            features.append(f)
            hidden.append(h.cpu())
    args.out.mkdir(parents=True, exist_ok=True)
    torch.tensor(drafts, dtype=torch.int32).numpy().tofile(args.out / "drafts.bin")
    torch.tensor(features, dtype=torch.float32).numpy().tofile(args.out / "features.bin")
    torch.stack(hidden).view(torch.int16).numpy().tofile(args.out / "hidden.bin")
    truth = [[int(tokens[p + 1 + i]) if p + 1 + i < len(tokens) else -1 for i in range(drafter.block - 1)]
             for p in positions]
    accepted = [next((i for i, (d, t) in enumerate(zip(ds, ts)) if d != t), len(ds)) for ds, ts in zip(drafts, truth)]
    (args.out / "meta.json").write_text(json.dumps({"positions": positions, "accepted_vs_tokens": accepted,
        "mean_accepted": sum(accepted) / len(accepted)}, indent=1))
    print(f"{len(positions)} anchors, mean accepted prefix vs golden tokens {sum(accepted) / len(accepted):.2f}")


if __name__ == "__main__":
    main()
