#!/usr/bin/env python3
"""dSpark drafter for GLM 5.3 Flash (RedHatAI/GLM-5.3-Flash-speculator.dspark-preview)
in plain PyTorch: the oracle the engine's drafter (glmf-golden --draft-oracle) is
compared with.

Semantics are those of the Speculators training model (DSparkDraftModel over
DFlashDraftModel, speculators 0.7) and vLLM's DSparkSpeculator, written as BF16
torch statements:

  taps     the target's aux_hidden_state_layer_ids are vLLM aux ids: id L is the
           input of layer L, i.e. the output of layer L - 1 (20, 28, 32, 36,
           40, 44 -> golden layer19 .. layer43), each the BF16 mean of the
           four mHC streams (FP32 sum in stream order, then / 4)
  context  fused = RMSNorm(fc(concat(taps)), hidden_norm); per draft layer
           K = rope(k_norm(k_proj(fused))), V = v_proj(fused); the block at
           anchor position p sees context positions p - 2048 .. p - 1 (the
           training mask's window, the same for every block row)
  block    [anchor, mask x 7] embeddings at positions p .. p + 7; Qwen3 layers
           (input norm, q/k/v, per-head q/k norm, rotate_half RoPE theta 1e4,
           attention over the context and the block, causal inside the block
           for sliding layers unless sliding_window_non_causal; o_proj;
           residual; post norm; SiLU MLP; residual); final norm
  heads    sample_from_anchor: block row k predicts token p + k + 1.
           base = lm_head(h) in FP32; Markov (vanilla): row k adds
           W2 . W1[prev] with prev the anchor for k = 0, else draft k - 1;
           greedy argmax (lowest index on ties). Confidence: sigmoid(w .
           [h_k, W1[prev_k]] + b) in FP32, the predicted acceptance of row k.

Writes, for every anchor position p in --positions (context = golden taps of
rows 0..p-1, anchor = tokens[p]):
  drafts.bin      u32  [N, 8]
  confidence.bin  f32  [N, 8]
  hidden.bin      bf16 [N, 8, hidden] final-norm output
  meta.json

--score A:B also replays every anchor in A..B teacher-forced and prints the
drafts accepted against the golden greedy picks (logits.bin argmax), as the
engine's --draft-replay does.

  reference.py --draft SNAP --golden DIR --positions 64:1500:16 --out DIR
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
    """Qwen3RMSNorm: the unit-RMS value is rounded to BF16 before the weight."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
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


class Drafter:
    def __init__(self, draft: Path, device: str, tap_shift: int = 1, causal: bool | None = None):
        self.cfg = json.loads((draft / "config.json").read_text())
        assert self.cfg.get("speculators_model_type") == "dspark", "not a dSpark checkpoint"
        t = self.cfg["transformer_layer_config"]
        self.block, self.mask = self.cfg["block_size"], self.cfg["mask_token_id"]
        assert self.cfg["sample_from_anchor"] and self.cfg["markov_head_type"] == "vanilla"
        self.taps = [i - tap_shift for i in self.cfg["aux_hidden_state_layer_ids"]]
        self.heads, self.kv_heads, self.head_dim = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        self.theta, self.eps = float(t["rope_parameters"]["rope_theta"]), float(t["rms_norm_eps"])
        self.window, self.layers, self.hidden = t["sliding_window"], t["num_hidden_layers"], t["hidden_size"]
        sliding = all(kind == "sliding_attention" for kind in t.get("layer_types") or [])
        self.causal = (sliding and not self.cfg.get("sliding_window_non_causal", False)) if causal is None else causal
        self.device = device
        f = safe_open(str(draft / "model.safetensors"), framework="pt", device=device)
        self.w = {k: f.get_tensor(k) for k in f.keys() if k != "embed_tokens.weight"}
        self.embed = f.get_tensor("embed_tokens.weight").cpu() if device != "cpu" else f.get_tensor("embed_tokens.weight")
        self.head = self.w["lm_head.weight"].float()
        self.w2 = self.w["markov_head.markov_w2.weight"].float()
        self.w1 = self.w["markov_head.markov_w1.weight"]
        self.conf_w = self.w["confidence_head.proj.weight"].float()[0]
        self.conf_b = self.w["confidence_head.proj.bias"].float()[0]

    def context(self, taps: torch.Tensor, positions: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer context K (normed, roped) and V for tap rows [n, taps * hidden]."""
        fused = rms(F.linear(taps, self.w["fc.weight"]), self.w["hidden_norm.weight"], self.eps)
        out = []
        for l in range(self.layers):
            p = f"layers.{l}.self_attn."
            k = F.linear(fused, self.w[p + "k_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
            v = F.linear(fused, self.w[p + "v_proj.weight"]).view(-1, self.kv_heads, self.head_dim)
            out.append((rope(rms(k, self.w[p + "k_norm.weight"], self.eps), positions, self.theta), v))
        return out

    def body(self, anchor: int, position: int, ctx: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Final-norm output [block, hidden] of one block at `position`."""
        b = self.block
        n_ctx = ctx[0][0].shape[0]
        mask = torch.ones(b, n_ctx + b, dtype=torch.bool, device=self.device)
        if self.causal:
            mask[:, n_ctx:] = torch.ones(b, b, dtype=torch.bool, device=self.device).tril()
        tokens = torch.tensor([anchor] + [self.mask] * (b - 1))
        h = self.embed[tokens].to(self.device)
        positions = torch.arange(position, position + b, device=self.device)
        for l in range(self.layers):
            p = f"layers.{l}."
            a = p + "self_attn."
            n = rms(h, self.w[p + "input_layernorm.weight"], self.eps)
            q = F.linear(n, self.w[a + "q_proj.weight"]).view(b, self.heads, self.head_dim)
            k = F.linear(n, self.w[a + "k_proj.weight"]).view(b, self.kv_heads, self.head_dim)
            v = F.linear(n, self.w[a + "v_proj.weight"]).view(b, self.kv_heads, self.head_dim)
            q = rope(rms(q, self.w[a + "q_norm.weight"], self.eps), positions, self.theta)
            k = rope(rms(k, self.w[a + "k_norm.weight"], self.eps), positions, self.theta)
            kc, vc = ctx[l]
            keys, values = torch.cat([kc, k]), torch.cat([vc, v])
            # FP32 softmax over BF16 scores, as the engine's kernel.
            scores = torch.einsum("qhd,khd->hqk", q.float(), keys.float()) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask[None], -math.inf)
            probs = scores.softmax(-1)
            attn = torch.einsum("hqk,khd->qhd", probs, values.float()).to(h.dtype).reshape(b, -1)
            h = h + F.linear(attn, self.w[a + "o_proj.weight"])
            n = rms(h, self.w[p + "post_attention_layernorm.weight"], self.eps)
            m = p + "mlp."
            act = F.silu(F.linear(n, self.w[m + "gate_proj.weight"])) * F.linear(n, self.w[m + "up_proj.weight"])
            h = h + F.linear(act, self.w[m + "down_proj.weight"])
        return rms(h, self.w["norm.weight"], self.eps)

    def heads_(self, anchor: int, hidden: torch.Tensor) -> tuple[list[int], list[float]]:
        """Markov-biased greedy chain over the block rows and the confidence per row."""
        logits = F.linear(hidden.float(), self.head)
        previous, tokens, confidence = anchor, [], []
        for k in range(self.block):
            e = self.w1[previous]
            score = logits[k] + F.linear(e.float(), self.w2)
            best = int(torch.argmax(score))  # first index of the maximum
            c = torch.dot(self.conf_w[:self.hidden], hidden[k].float()) + \
                torch.dot(self.conf_w[self.hidden:], e.float()) + self.conf_b
            confidence.append(float(torch.sigmoid(c)))
            tokens.append(best)
            previous = best
        return tokens, confidence


def parse_positions(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if ":" in part:
            a, b, *s = (int(x) for x in part.split(":"))
            out.extend(range(a, b, s[0] if s else 1))
        else:
            out.append(int(part))
    return out


def golden_taps(golden: Path, layers: list[int], hidden: int, device: str) -> torch.Tensor:
    taps = []
    for layer in layers:
        raw = torch.frombuffer(bytearray((golden / f"layer{layer:02d}.bin").read_bytes()), dtype=torch.bfloat16)
        # FP32 sum of the four streams in order, then the division, rounded to BF16.
        streams = raw.view(-1, 4, hidden).float()
        taps.append(((((streams[:, 0] + streams[:, 1]) + streams[:, 2]) + streams[:, 3]) / 4).to(torch.bfloat16))
    return torch.cat(taps, -1).to(device)


def greedy_picks(golden: Path, tokens: int, vocab: int) -> list[int]:
    out = []
    with open(golden / "logits.bin", "rb") as f:
        for _ in range(tokens):
            row = torch.frombuffer(bytearray(f.read(vocab * 4)), dtype=torch.float32)
            out.append(int(torch.argmax(row)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--positions", help="anchor positions, e.g. 64:1500:16,1510")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--score", help="A:B: teacher-forced acceptance vs the golden greedy picks")
    parser.add_argument("--tap-shift", type=int, default=1, help="target layer = aux id - shift (vLLM: 1)")
    parser.add_argument("--causal", type=int, choices=(0, 1), help="override in-block causality")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    drafter = Drafter(args.draft, args.device, args.tap_shift, None if args.causal is None else bool(args.causal))
    tokens = torch.frombuffer(bytearray((args.golden / "tokens.bin").read_bytes()), dtype=torch.int32)
    taps = golden_taps(args.golden, drafter.taps, drafter.hidden, args.device)
    rows = torch.arange(taps.shape[0], device=args.device)
    with torch.no_grad():
        ctx = drafter.context(taps, rows)

        def draft(p: int) -> tuple[list[int], list[float], torch.Tensor]:
            first = max(0, p - drafter.window)
            h = drafter.body(int(tokens[p]), p, [(k[first:p], v[first:p]) for k, v in ctx])
            t, c = drafter.heads_(int(tokens[p]), h)
            return t, c, h

        if args.score:
            a, b = (int(x) for x in args.score.split(":"))
            b = min(b, len(tokens) - drafter.block - 1)
            greedy = greedy_picks(args.golden, len(tokens), drafter.head.shape[0])
            kept_total, first_ok, conf_sum, hist = 0, 0, [0.0] * drafter.block, [0] * (drafter.block + 1)
            for p in range(a, b):
                t, c, _ = draft(p)
                kept = 0
                while kept < drafter.block and t[kept] == greedy[p + kept] and \
                        (kept == 0 or t[kept - 1] == int(tokens[p + kept])):
                    kept += 1
                kept_total += kept
                hist[kept] += 1
                first_ok += int(t[0] == greedy[p])
                conf_sum = [s + x for s, x in zip(conf_sum, c)]
            n = b - a
            print(f"score taps {drafter.taps} causal {drafter.causal}: {n} anchors, accepted vs greedy "
                  f"{kept_total / n:.3f} of {drafter.block}, first draft {100 * first_ok / n:.1f}%, histogram {hist}, "
                  f"mean confidence {[round(s / n, 3) for s in conf_sum]}")
        if args.positions:
            positions = parse_positions(args.positions)
            drafts, confidence, hidden = [], [], []
            for p in positions:
                t, c, h = draft(p)
                drafts.append(t)
                confidence.append(c)
                hidden.append(h.cpu())
            args.out.mkdir(parents=True, exist_ok=True)
            torch.tensor(drafts, dtype=torch.int32).numpy().tofile(args.out / "drafts.bin")
            torch.tensor(confidence, dtype=torch.float32).numpy().tofile(args.out / "confidence.bin")
            torch.stack(hidden).view(torch.int16).numpy().tofile(args.out / "hidden.bin")
            truth = [[int(tokens[p + 1 + i]) if p + 1 + i < len(tokens) else -1 for i in range(drafter.block)]
                     for p in positions]
            accepted = [next((i for i, (d, t) in enumerate(zip(ds, ts)) if d != t), len(ds))
                        for ds, ts in zip(drafts, truth)]
            (args.out / "meta.json").write_text(json.dumps({"positions": positions, "taps": drafter.taps,
                "causal": drafter.causal, "accepted_vs_tokens": accepted,
                "mean_accepted": sum(accepted) / len(accepted)}, indent=1))
            print(f"{len(positions)} anchors, mean accepted prefix vs golden tokens {sum(accepted) / len(accepted):.2f}")


if __name__ == "__main__":
    main()
