#!/usr/bin/env python3
"""Compact top-k fidelity reference from a family golden run.

The golden harnesses (python/reference/families/<id>/golden.py) write the
reference model's full logits for a prompt (tokens.bin i32 [T], logits.bin
f32 [T, vocab], meta.json). The benchmark's quick-quality check scores the
served model on the same tokens and compares against a compact summary of
those rows: per position the reference's top-k ids and log-probabilities,
the log of the remaining tail mass and the next token's log-probability.

    make-fidelity-reference.py --golden runs/qwen4-golden --model 'Qwen/Qwen3.8-Flash-Next*' \
        --out rust/crates/cuteafd-bench/references/qwen3.8-flash-next.json
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--model", action="append", required=True,
                        help="checkpoint id pattern this reference applies to (fnmatch, repeatable)")
    parser.add_argument("--name", help="display name (default: the first pattern)")
    parser.add_argument("--positions", type=int, default=512, help="scored positions from --from")
    parser.add_argument("--from", dest="start", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--kl-max", type=float, default=0.15)
    parser.add_argument("--top1-min", type=float, default=0.80)
    args = parser.parse_args()

    meta = json.loads((args.golden / "meta.json").read_text())
    tokens = np.fromfile(args.golden / "tokens.bin", dtype=np.int32)
    t = len(tokens)
    logits = np.memmap(args.golden / "logits.bin", dtype=np.float32, mode="r")
    vocab = logits.size // t
    assert vocab * t == logits.size, "logits.bin is not [T, vocab]"
    logits = logits.reshape(t, vocab)
    end = min(t, args.start + args.positions)
    snapshot = pathlib.Path(meta.get("snapshot", ""))
    # Goldens run in containers that mount the hub at /root/.cache/huggingface/hub.
    container_hub = pathlib.Path("/root/.cache/huggingface/hub")
    local = pathlib.Path.home() / ".cache/huggingface/hub" / snapshot.relative_to(container_hub) \
        if snapshot.is_relative_to(container_hub) else snapshot
    tokenizer = local / "tokenizer.json"
    ids, lps, tail, nxt = [], [], [], []
    for p in range(args.start, end):
        # Row p - 1 predicts token p.
        row = logits[p - 1].astype(np.float64)
        top = row.max()
        lse = top + np.log(np.exp(row - top).sum())
        logp = row - lse
        order = np.argpartition(-row, args.top_k)[: args.top_k]
        order = order[np.lexsort((order, -row[order]))]
        mass = np.exp(logp[order]).sum()
        ids.append([int(i) for i in order])
        lps.append([round(float(v), 5) for v in logp[order]])
        tail.append(round(float(np.log(max(1.0 - mass, 1e-30))), 5))
        nxt.append(round(float(logp[tokens[p]]), 5))
    reference = {
        "schema": "cuteafd.bench.reference/1",
        "name": args.name or args.model[0],
        "models": args.model,
        "source": {
            "golden": args.golden.name,
            "snapshot": str(snapshot),
            "reference": meta.get("reference", "family golden.py"),
            "experts_snapshot": meta.get("experts_snapshot"),
        },
        "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest() if tokenizer.exists() else None,
        "vocab": vocab,
        "tokens": [int(x) for x in tokens[:end]],
        "score_from": args.start,
        "top_k": args.top_k,
        "ids": ids,
        "lps": lps,
        "tail_lp": tail,
        "next_lp": nxt,
        "nll": round(-float(np.mean(nxt)), 6),
        "expect": {"kl_max": args.kl_max, "top1_min": args.top1_min},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reference, separators=(",", ":")) + "\n")
    print(f"{args.out}: {end - args.start} positions, top-{args.top_k}, ref NLL {reference['nll']:.4f}, "
          f"{args.out.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
