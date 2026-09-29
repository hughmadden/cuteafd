#!/usr/bin/env python3
"""MiMo V2 Flash / V2.6 Pro native MTP drafter (model.mtp.layers.0..2) in
plain PyTorch, teacher-forced on a golden run: acceptance of its drafts.

The layer is SGLang's (models/mimo_v2_nextn.py; EAGLE multi-layer: draft
step k runs MTP layer k, model_loader._filter_mtp_weights): layer k at row j
takes x = eh_proj(cat(enorm(embed(t)), hnorm(h))) with t = t_{j+k+1}; one
SWA decoder layer (sinks, 128-token window, value scale, partial NeoX RoPE
with the SWA theta) with a dense SwiGLU MLP, then final_layernorm and the
target LM head predict t_{j+k+2}. h (--chain): the target's last layer output
at j (the golden's layerNN.bin) for every stage ("target", the default), or
the previous stage's pre-norm ("prev") / final-norm ("normed") output at j.
Measured (Flash golden, per-stage accuracy vs text): target 0.43/0.37/0.33,
prev 0.43/0.14/0.10, normed 0.43/0.17/0.10: the stages read the target's
hidden state. Positions: row j (--positions hidden) or j + k + 1 (token);
the two agree within 0.01.

Teacher-forced along the golden text, the draft chain after anchor p (whose
next token t_{p+1} the target already produced) is accepted while
argmax(head(H^k_p)) == t_{p+k+2}; the chain is exact under that forcing
because every accepted draft equals the forced token.

  reference.py --snapshot SNAP --golden DIR [--pro] [--positions hidden|token]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def rms(x, w, eps):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rope(x, positions, theta, dim):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.int64).float() / dim))
    emb = torch.cat([positions.float()[:, None] * inv[None]] * 2, -1)
    cos, sin = emb.cos().to(x.dtype)[:, None], emb.sin().to(x.dtype)[:, None]
    rot, keep = x[..., :dim], x[..., dim:]
    half = dim // 2
    return torch.cat([rot * cos + torch.cat([-rot[..., half:], rot[..., :half]], -1) * sin, keep], -1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--pro", action="store_true", help="V2.6 Pro (fused qkv) instead of V2 Flash")
    p.add_argument("--positions", choices=("hidden", "token"), default="hidden")
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--chain", choices=("prev", "normed", "target", "target_first", "layer0"), default="target",
                   help="stage k's hidden input: the previous stage's pre-norm output (prev), its final-norm "
                        "output (normed), the target's (target); layer0: every stage runs MTP layer 0")
    p.add_argument("--anchors", default="16:1560:8", help="report acceptance at these anchors (DFlash's)")
    p.add_argument("--out", type=Path, help="write predictions.bin (i32 [stages, T]: stage k's argmax at row j) "
                   "and meta.json")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((a.snapshot / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    if a.pro:
        from mimo_v26.golden import Weights
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(a.snapshot, trust_remote_code=True)
        w = Weights(a.snapshot, hf)
        get = lambda name: w.get(name, 1)  # noqa: E731  (an SWA layer's qkv geometry)
    else:
        from mimo_v2.golden import Weights

        w = Weights(a.snapshot)
        get = w.get
    hidden, eps = cfg["hidden_size"], cfg.get("layernorm_epsilon", 1e-5)
    heads, kv = cfg["swa_num_attention_heads"], cfg["swa_num_key_value_heads"]
    hd, vd = cfg["swa_head_dim"], cfg["swa_v_head_dim"]
    rope_dim = int(hd * cfg["partial_rotary_factor"])
    theta, window = float(cfg["swa_rope_theta"]), cfg["sliding_window"]
    v_scale = float(cfg.get("attention_value_scale", 1.0))
    tokens = torch.frombuffer(bytearray((a.golden / "tokens.bin").read_bytes()), dtype=torch.int32).long()
    T = len(tokens)
    last = cfg["num_hidden_layers"] - 1
    h_prev = torch.frombuffer(bytearray((a.golden / f"layer{last:02d}.bin").read_bytes()),
                              dtype=torch.bfloat16).view(T, hidden).cuda()
    h_target = h_prev
    embed = get("model.embed_tokens.weight")
    head = get("lm_head.weight")
    vocab = head.shape[0]
    glog = torch.frombuffer(bytearray((a.golden / "logits.bin").read_bytes()), dtype=torch.float32).view(T, vocab)
    greedy = glog.argmax(-1)
    q_pos = torch.arange(T, device="cuda")[:, None]
    predictions = []
    with torch.no_grad():
        for k in range(a.stages):
            pre = f"model.mtp.layers.{0 if a.chain == 'layer0' else k}."
            shift = 1 if a.chain == "target_first" else k + 1
            nxt = torch.cat([tokens[shift:], torch.zeros(shift, dtype=torch.long)])  # t_{j+k+1} (t_{j+1})
            e = embed[nxt.cuda()]
            x = F.linear(torch.cat([rms(e, get(pre + "enorm.weight"), eps), rms(h_prev, get(pre + "hnorm.weight"), eps)],
                                   -1), get(pre + "eh_proj.weight"))
            positions = torch.arange(T, device="cuda") + (k + 1 if a.positions == "token" else 0)
            n = rms(x, get(pre + "input_layernorm.weight"), eps)
            at = pre + "self_attn."
            if a.pro:
                qkv = F.linear(n, get(at + "qkv_proj.weight"))
                q, kk, v = qkv.split([heads * hd, kv * hd, kv * vd], -1)
            else:
                q, kk, v = (F.linear(n, get(at + f"{s}_proj.weight")) for s in "qkv")
            q = rope(q.view(T, heads, hd), positions, theta, rope_dim)
            kk = rope(kk.view(T, kv, hd), positions, theta, rope_dim)
            v = (v.view(T, kv, vd) * v_scale)
            group = heads // kv
            mask = (q_pos[None] >= q_pos.T[None]) & (q_pos.T[None] > q_pos[None] - window)  # [1, q, k]
            sink = get(at + "attention_sink_bias").float()
            out = torch.empty(T, heads, vd, dtype=torch.bfloat16, device="cuda")
            for h0 in range(0, heads, 16):
                qs = q[:, h0:h0 + 16].float()
                ks = kk[:, h0 // group:(h0 + 16) // group].repeat_interleave(group, 1)[:, :16].float() \
                    if group <= 16 else kk[:, h0 // group][:, None].expand(-1, 16, -1).float()
                vs = v[:, h0 // group:(h0 + 16) // group].repeat_interleave(group, 1)[:, :16].float() \
                    if group <= 16 else v[:, h0 // group][:, None].expand(-1, 16, -1).float()
                s = torch.einsum("qhd,khd->hqk", qs, ks) / math.sqrt(hd)
                s = s.masked_fill(~mask, float("-inf"))
                s = torch.cat([s, sink[h0:h0 + 16, None, None].expand(-1, T, 1)], -1)
                pr = torch.softmax(s - s.amax(-1, keepdim=True), -1).to(torch.bfloat16)[..., :-1]
                out[:, h0:h0 + 16] = torch.einsum("hqk,khd->qhd", pr.float(), vs).to(torch.bfloat16)
            x = x + F.linear(out.view(T, -1), get(at + "o_proj.weight"))
            n = rms(x, get(pre + "pre_mlp_layernorm.weight"), eps)
            m = pre + "mlp."
            x = x + F.linear(F.silu(F.linear(n, get(m + "gate_proj.weight"))) * F.linear(n, get(m + "up_proj.weight")),
                             get(m + "down_proj.weight"))
            normed = rms(x, get(pre + "final_layernorm.weight"), eps)
            h_prev = {"prev": x, "layer0": x, "normed": normed, "target": h_target,
                      "target_first": h_target}[a.chain]
            logits = F.linear(normed, head)
            predictions.append(logits.float().argmax(-1).cpu())  # predicts t_{j+k+2}
    a0, a1, *step = (int(v) for v in a.anchors.split(":"))
    anchors = range(a0, a1, step[0] if step else 1)
    text, cons = [], []
    for p in anchors:  # DFlash anchor p: context 0..p-1, next token t_p -> MTP row p - 1
        j = p - 1
        acc = 0
        while acc < a.stages and j + acc + 2 < T and int(predictions[acc][j]) == int(tokens[j + acc + 2]):
            acc += 1
        text.append(acc)
        acc = 0
        while acc < a.stages and j + acc + 1 < T and int(predictions[acc][j]) == int(greedy[j + acc + 1]) \
                and (acc == 0 or int(tokens[j + acc + 1]) == int(predictions[acc - 1][j])):
            acc += 1
        cons.append(acc)
    first = [float((predictions[k][:-k - 2] == tokens[k + 2:]).float().mean()) for k in range(a.stages)]
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        torch.stack(predictions).int().numpy().tofile(a.out / "predictions.bin")
        (a.out / "meta.json").write_text(json.dumps({"stages": a.stages, "chain": a.chain, "positions": a.positions,
            "tokens": T, "anchors": list(anchors), "accepted_vs_text": text, "accepted_vs_greedy": cons}))
    print(f"MTP ({'pro' if a.pro else 'flash'}, positions {a.positions}, chain {a.chain}) at {len(text)} anchors: accepted prefix vs "
          f"text {sum(text) / len(text):.2f}, vs target greedy {sum(cons) / len(cons):.2f} of {a.stages}; "
          f"per-stage accuracy vs text {[round(x, 3) for x in first]}; histogram "
          f"{[text.count(i) for i in range(a.stages + 1)]}")


if __name__ == "__main__":
    main()
