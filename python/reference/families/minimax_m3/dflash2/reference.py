#!/usr/bin/env python3
"""DFlash2 drafter for MiniMax M3.1 Flash (RadixArk/Minimax-M3.1-Flash-Dflash2) in plain PyTorch.

The oracle the engine's shared drafter (cuteafd-daemon shared::speculation::dflash, the
`_gemma` kernels of native/families/glm5/cuda/glm_dflash.cu) is compared with. Semantics
are SGLang's MiniMaxDFlash2DraftModel (python/sglang/srt/models/dflash.py, branch
minimax-m3.1-flash) with the release config: GemmaRMSNorm everywhere, NeoX RoPE on the
first 64 of 128 channels (theta 1e7), SwiGLU-OAI (alpha 1.702, limit 7), 64 query heads over
4 KV heads, a 4096-position sliding window and, as SGLang serves sliding-window draft layers
whose config sets no is_causal, causal block rows. The Gemma norms, RoPE and the activation
are the MiniMax target reference's (families/minimax_m3/model.py: FP32, rounded once); the
convolution, selector and BF16 GEMM boundaries are GLM's DFlash2 reference's:

  context  fused = gemma(fc(concat(target layer outputs)), hidden_norm), each output the
           golden's FP32 layerNN.bin rounded to BF16 (as the engine taps the FP32 residual)
           per draft layer: K = rope(gemma_heads(k_proj(fused))), V = v_proj(fused)
  block    [anchor, mask x 7] embeddings at positions p..p+7; per layer:
           n = gemma(h); dyn = attention_conv.kernel_projection(n)
           qkv(conv0(n)) -> per-head gemma q/k norms, partial RoPE; attention over the
           context and the block, block row r seeing context positions q with p + r - q <
           4096 and block rows 0..r (FP32 scores and softmax); o_proj;
           s = h + conv1(o) in FP32, h = bf16(s), n = gemma(s) (a fused add-norm sees the
           unrounded sum); the MLP likewise with mlp_conv, SwiGLU-OAI on [gate | up].
  head     logits = lm_head(norm(h))[1:8]; top-16 per row; the selector walks the rows
           greedily (GLM's reference), scoring each candidate as its unary logit plus the BF16
           edge in FP32 as SGLang's DFlash2 selector does (--selector-score bf16: GLM's rounding).

Writes, for every anchor position p in --positions (context = golden taps of rows
max(0, p - 4096)..p-1, anchor = tokens[p]):
  drafts.bin    i32 [N, 7]
  features.bin  f32 [N, 7, 4]  margin, best probability, entropy, rank among the 16
  hidden.bin    bf16 [N, 8, hidden] final-norm output
  meta.json

  reference.py --draft SNAP --target SNAP --golden WINDOW_DIR --positions 64:1500:16 --out DIR

WINDOW_DIR is one window of a MiniMax golden run with golden.py --layers 3 17 31 45 59
(its tokens.bin and layerNN.bin); minimax-golden --draft SNAP --draft-oracle DIR compares the
engine with the output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from families.glm5.dflash2.reference import Drafter as GlmDrafter, conv, parse_positions  # noqa: E402
from families.minimax_m3.model import gemma_norm, rope_partial, swiglu_oai  # noqa: E402

EMBED = "language_model.model.embed_tokens.weight"
HEAD = "language_model.lm_head.weight"


def block_mask(position: int, context: int, block: int, window: int, causal: bool, device) -> torch.Tensor:
    """Visible keys [block, context + block] of block rows at position..position+block-1 over the
    context positions position-context..position-1, then the block's own rows."""
    rows = torch.arange(block, device=device)[:, None]
    keys = torch.arange(position - context, position, device=device)[None]
    visible = torch.cat([(position + rows - keys) < window, torch.ones(block, block, dtype=torch.bool, device=device)], 1)
    if causal:
        visible[:, context:] &= torch.arange(block, device=device)[None] <= rows
    return visible


class Drafter(GlmDrafter):
    """The GLM reference's selector over MiniMax's Gemma-norm, partial-RoPE, SwiGLU-OAI block."""

    def __init__(self, draft: Path, target: Path, device: str, causal: bool | None = None,
                 score_fp32: bool = True):
        self.cfg = json.loads((draft / "config.json").read_text())
        c, dc = self.cfg, self.cfg["dflash_config"]
        if (not c.get("use_gemma_norm") or c.get("hidden_act") != "swigluoai" or c.get("attention_output_gate")
                or c.get("qk_norm_type", "per_head") != "per_head"
                or any(kind != "sliding_attention" for kind in c.get("layer_types", ["sliding_attention"]))):
            raise ValueError("not a Gemma-norm SwiGLU-OAI sliding-window DFlash2 draft")
        self.block, self.mask, self.taps = dc["block_size"], dc["mask_token_id"], dc["target_layer_ids"]
        self.group, self.topk = dc["conv_group_size"], dc["selector_top_k"]
        self.heads, self.kv_heads, self.head_dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.theta, self.eps = float(c["rope_parameters"]["rope_theta"]), float(c["rms_norm_eps"])
        factor = c.get("partial_rotary_factor", c["rope_parameters"].get("partial_rotary_factor", 1.0))
        self.rotary = round(self.head_dim * factor)
        self.alpha, self.limit = float(c["swiglu_alpha"]), float(c["swiglu_limit"])
        self.window, self.layers = c["sliding_window"], c["num_hidden_layers"]
        self.causal = c.get("is_causal", True) if causal is None else causal
        # SGLang's DFlash2 selector scores candidates in FP32 (unary logit plus BF16 edge).
        self.score_fp32 = score_fp32
        self.device = device
        f = safe_open(str(draft / "model.safetensors"), framework="pt", device=device)
        self.w = {k: f.get_tensor(k) for k in f.keys()}
        index = json.loads((target / "model.safetensors.index.json").read_text())["weight_map"]
        files = {}

        def target_tensor(name: str) -> torch.Tensor:
            if index[name] not in files:
                files[index[name]] = safe_open(str(target / index[name]), framework="pt", device="cpu")
            return files[index[name]].get_tensor(name)

        self.embed = target_tensor(EMBED)
        self.lm_head = target_tensor(HEAD).to(device)
        self.mhc = False

    def norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return gemma_norm(x, self.w[name], self.eps)

    def heads_norm(self, x: torch.Tensor, heads: int, name: str) -> torch.Tensor:
        return gemma_norm(x.view(-1, heads, self.head_dim), self.w[name], self.eps)

    def rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return rope_partial(x, positions, self.rotary, self.theta)

    def context(self, taps: torch.Tensor, positions: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer context K (normed, roped) and V for BF16 tap rows [n, 5 * hidden]."""
        fused = self.norm(F.linear(taps, self.w["fc.weight"]), "hidden_norm.weight")
        out = []
        for l in range(self.layers):
            p = f"layers.{l}.self_attn."
            k = self.heads_norm(F.linear(fused, self.w[p + "k_proj.weight"]), self.kv_heads, p + "k_norm.weight")
            v = F.linear(fused, self.w[p + "v_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
            out.append((self.rope(k, positions), v))
        return out

    def attend(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """FP32 scores and softmax over visible keys; [block, heads * dim] BF16."""
        group = self.heads // self.kv_heads
        keys = keys.float().repeat_interleave(group, 1).transpose(0, 1)
        values = values.float().repeat_interleave(group, 1).transpose(0, 1)
        scores = q.float().transpose(0, 1) @ keys.transpose(1, 2) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~visible[None], -math.inf).softmax(-1)
        return (scores @ values).transpose(0, 1).reshape(q.shape[0], -1).to(q.dtype)

    def body(self, anchor: int, position: int, ctx: list[tuple[torch.Tensor, torch.Tensor]],
             row_window: bool = True) -> torch.Tensor:
        """Final-norm output [block, hidden] of one block at `position` (the context holds
        positions position - len(ctx) .. position - 1)."""
        visible = block_mask(position, ctx[0][0].shape[0], self.block, self.window if row_window else 1 << 62,
                             self.causal, self.device)
        h = self.embed[torch.tensor([anchor] + [self.mask] * (self.block - 1))].to(self.device)
        positions = torch.arange(position, position + self.block, device=self.device)
        n = self.norm(h, "layers.0.input_layernorm.weight")
        for l in range(self.layers):
            p = f"layers.{l}."
            dyn = F.linear(n, self.w[p + "attention_conv.kernel_projection.weight"])
            base = self.w[p + "attention_conv.base_kernel"]
            x = conv(n, dyn, base, 0, self.group)
            a = p + "self_attn."
            q = self.rope(self.heads_norm(F.linear(x, self.w[a + "q_proj.weight"]), self.heads, a + "q_norm.weight"),
                          positions)
            k = self.rope(self.heads_norm(F.linear(x, self.w[a + "k_proj.weight"]), self.kv_heads, a + "k_norm.weight"),
                          positions)
            v = F.linear(x, self.w[a + "v_proj.weight"]).view(self.block, self.kv_heads, self.head_dim)
            kc, vc = ctx[l]
            attn = self.attend(q, torch.cat([kc, k]), torch.cat([vc, v]), visible)
            o = F.linear(attn, self.w[a + "o_proj.weight"])
            total = h.float() + conv(o, dyn, base, 1, self.group).float()
            h, n = total.to(h.dtype), gemma_norm(total, self.w[p + "post_attention_layernorm.weight"], self.eps).to(h.dtype)
            dyn = F.linear(n, self.w[p + "mlp_conv.kernel_projection.weight"])
            base = self.w[p + "mlp_conv.base_kernel"]
            x = conv(n, dyn, base, 0, self.group)
            m = p + "mlp."
            act = swiglu_oai(F.linear(x, self.w[m + "gate_proj.weight"]), F.linear(x, self.w[m + "up_proj.weight"]),
                             self.alpha, self.limit)
            d = F.linear(act, self.w[m + "down_proj.weight"])
            total = h.float() + conv(d, dyn, base, 1, self.group).float()
            following = "norm.weight" if l + 1 == self.layers else f"layers.{l + 1}.input_layernorm.weight"
            h, n = total.to(h.dtype), gemma_norm(total, self.w[following], self.eps).to(h.dtype)
        return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True, help="one golden window directory")
    parser.add_argument("--positions", required=True, help="anchor positions, e.g. 64:1500:16,1510")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--causal", type=int, choices=(0, 1), help="block rows attend causally (default: the "
                        "config's is_causal, else 1 as SGLang serves sliding-window layers)")
    parser.add_argument("--selector-score", choices=("fp32", "bf16"), default="fp32",
                        help="selector scores: FP32 as SGLang's DFlash2 selector (serving's default) or BF16-rounded "
                        "as the GLM ports (minimax-golden --draft-selector-score)")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    drafter = Drafter(args.draft, args.target, args.device, None if args.causal is None else bool(args.causal),
                      score_fp32=args.selector_score == "fp32")
    tokens = torch.frombuffer(bytearray((args.golden / "tokens.bin").read_bytes()), dtype=torch.int32)
    hidden_size = drafter.cfg["hidden_size"]
    taps = torch.cat([torch.frombuffer(bytearray((args.golden / f"layer{layer:02d}.bin").read_bytes()),
                                       dtype=torch.float32).view(-1, hidden_size).bfloat16() for layer in drafter.taps], -1)
    taps = taps.to(args.device)
    positions = parse_positions(args.positions)
    if any(p < 1 or p >= len(tokens) for p in positions):
        raise ValueError("anchor positions must lie inside the window after its first token")
    rows = torch.arange(taps.shape[0], device=args.device)
    with torch.no_grad():
        ctx = drafter.context(taps, rows)
        drafts, features, hidden = [], [], []
        for p in positions:
            first = max(0, p - drafter.window)
            h = drafter.body(int(tokens[p]), p, [(k[first:p], v[first:p]) for k, v in ctx])
            t, f = drafter.select(int(tokens[p]), h, fp32_score=drafter.score_fp32)
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
    (args.out / "meta.json").write_text(json.dumps({"positions": positions, "causal": drafter.causal,
        "selector_score": args.selector_score,
        "accepted_vs_tokens": accepted, "mean_accepted": sum(accepted) / len(accepted)}, indent=1))
    print(f"{len(positions)} anchors, mean accepted prefix vs golden tokens {sum(accepted) / len(accepted):.2f}")


if __name__ == "__main__":
    main()
