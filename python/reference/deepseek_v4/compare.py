#!/usr/bin/env python3
"""Compare the b12x DeepSeek V4 prefill (b12x_model.py) with golden.py output.

Reads GOLDEN/meta.json (tokens), GOLDEN/embed.pt, GOLDEN/layerNN.pt ([1,T,4,dim])
and GOLDEN/logits.pt ([1,T,V]); runs b12x_model one layer at a time and reports,
per layer, max-abs / relative-L2 error and cosine similarity of the output stream,
then top-1 agreement and KL of the final logits.

Two feeds (both by default, sharing one weight load per layer):
  golden  layer L consumes golden layer L-1 (embed for L=0): per-layer op error.
  chain   layer L consumes b12x layer L-1: accumulated end-to-end error.
Logits are compared for the chain feed when all layers ran, and for the golden
feed (head applied to the golden last-layer stream) when that file exists.

  compare.py --golden DIR [--snapshot SNAP] [--device 0] [--layers 0-5 | 0 2 3]
             [--feeds golden,chain] [--wo-mode b12x|reference]
             [--moe-activation silu_v41|silu] [--json OUT] [--ledger]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b12x_model import DEFAULT_SNAPSHOT, DeepseekV4B12x  # noqa: E402


def stream_metrics(ours: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    a = ours.float()
    b = ref.to(a.device).float().reshape(a.shape)
    diff = a - b
    token_cos = F.cosine_similarity(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1), dim=1)
    return {
        "max_abs": float(diff.abs().max()),
        "rel_l2": float(diff.norm() / b.norm().clamp_min(1e-30)),
        "cos": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
        "min_token_cos": float(token_cos.min()),
        "ref_max_abs": float(b.abs().max()),
        "nonfinite": int((~torch.isfinite(a)).sum()),
    }


def logits_metrics(ours: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    a = ours.float()
    b = ref.to(a.device).float().reshape(a.shape)
    top_a, top_b = a.argmax(-1), b.argmax(-1)
    logp_a, logp_b = a.log_softmax(-1), b.log_softmax(-1)
    kl = (logp_b.exp() * (logp_b - logp_a)).sum(-1)  # KL(golden || b12x) per position
    return {
        "top1_agree": float((top_a == top_b).float().mean()),
        "last_top1_match": bool(top_a[-1] == top_b[-1]),
        "last_top1_b12x": int(top_a[-1]),
        "last_top1_golden": int(top_b[-1]),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "cos": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
    }


def parse_layers(values: list[str] | None, n_layers: int) -> list[int]:
    if not values:
        return list(range(n_layers))
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                out.extend(range(int(lo), int(hi) + 1))
            elif part:
                out.append(int(part))
    return sorted(set(out))


def fmt(m: dict) -> str:
    return (f"max_abs={m['max_abs']:.4g} rel={m['rel_l2']:.3e} cos={m['cos']:.6f} "
            f"min_tok_cos={m['min_token_cos']:.5f}" + (f" NONFINITE={m['nonfinite']}" if m["nonfinite"] else ""))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, default=None, help="default: meta.json snapshot, else Flash-0731")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--layers", nargs="*", help="layers to run/compare, e.g. 0-5 or 0 2 3 (default all)")
    p.add_argument("--feeds", default="golden,chain")
    p.add_argument("--wo-mode", choices=("b12x", "reference"), default="b12x")
    p.add_argument("--moe-activation", choices=("silu_v41", "silu"), default="silu_v41")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--ledger", action="store_true", help="print exact per-op scratch/buffer bytes (last T seen)")
    a = p.parse_args()

    meta = json.loads((a.golden / "meta.json").read_text())
    tokens = [int(t) for t in meta["tokens"]]
    snapshot = a.snapshot or Path(meta.get("snapshot") or DEFAULT_SNAPSHOT)
    feeds = [f.strip() for f in a.feeds.split(",") if f.strip()]
    torch.backends.cuda.matmul.allow_tf32 = False

    model = DeepseekV4B12x(snapshot, a.device, wo_mode=a.wo_mode, moe_activation=a.moe_activation)
    cfg, dev, T = model.cfg, model.device, len(tokens)
    layers = parse_layers(a.layers, cfg.n_layers)
    ids = torch.tensor(tokens, dtype=torch.int64, device=dev)
    print(f"T={T} layers={layers[0]}..{layers[-1]} ({len(layers)}) feeds={feeds} wo={a.wo_mode} "
          f"moe={a.moe_activation} snapshot={snapshot}", flush=True)

    def golden_stream(layer: int) -> torch.Tensor | None:
        path = a.golden / f"layer{layer:02d}.pt"
        return torch.load(path, map_location=dev)[0].contiguous() if path.exists() else None

    results: dict = {"tokens": T, "layers": {}, "config": vars(a) | {"golden": str(a.golden)}}
    embed_ours = model.embed(tokens)
    embed_path = a.golden / "embed.pt"
    if embed_path.exists():
        e = torch.load(embed_path, map_location=dev)[0]
        results["embed_max_abs"] = float((embed_ours.float() - e.float()).abs().max())
        print(f"embed max_abs={results['embed_max_abs']:.3g}")
    chain = model.expand(embed_ours) if "chain" in feeds else None
    if "chain" in feeds and layers[0] != 0:
        print("note: chain feed runs every layer from 0 up to the last requested layer")
    chain_layers = range(0, layers[-1] + 1) if "chain" in feeds else []

    for layer in sorted(set(layers) | set(chain_layers)):
        start = time.time()
        block = model.load_layer(layer)
        row: dict = {"ratio": cfg.ratio(layer), "hash": cfg.is_hash_layer(layer)}
        ref = golden_stream(layer) if layer in layers else None
        if "golden" in feeds and layer in layers:
            if layer == 0:
                e = torch.load(embed_path, map_location=dev)[0] if embed_path.exists() else embed_ours
                prev = model.expand(e.to(torch.bfloat16))
            else:
                prev = golden_stream(layer - 1)
            if prev is None:
                row["golden"] = "missing input"
            elif ref is not None:
                out = model.layer_forward(layer, prev.to(torch.bfloat16).contiguous(), ids, block=block)
                row["golden"] = stream_metrics(out, ref)
                del out
        if chain is not None and layer in chain_layers:
            chain = model.layer_forward(layer, chain, ids, block=block)
            if ref is not None:
                row["chain"] = stream_metrics(chain, ref)
        row["seconds"] = round(time.time() - start, 2)
        results["layers"][layer] = row
        text = " | ".join(f"{k}: {fmt(v)}" for k, v in row.items() if isinstance(v, dict))
        print(f"layer {layer:2d} C{row['ratio']:<3d}{'H' if row['hash'] else ' '} {row['seconds']:6.1f}s {text}",
              flush=True)
        del block
        torch.cuda.empty_cache()

    logits_path = a.golden / "logits.pt"
    if logits_path.exists():
        ref_logits = torch.load(logits_path, map_location="cpu")[0]
        if chain is not None and layers[-1] == cfg.n_layers - 1:
            results["logits_chain"] = logits_metrics(model.head(chain), ref_logits.to(dev))
            print("logits chain :", results["logits_chain"])
        last = golden_stream(cfg.n_layers - 1)
        if "golden" in feeds and last is not None:
            results["logits_golden_feed"] = logits_metrics(model.head(last.to(torch.bfloat16)), ref_logits.to(dev))
            print("logits golden:", results["logits_golden_feed"])
    if a.ledger:
        print(model.ops.ledger.report(), flush=True)
    results["ledger"] = {"scratch": model.ops.ledger.entries, "buffers": model.ops.ledger.buffers}
    if a.json:
        a.json.write_text(json.dumps(results, indent=1, default=str))
        print("wrote", a.json)


if __name__ == "__main__":
    main()
