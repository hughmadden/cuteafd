#!/usr/bin/env python3
"""Full-vocabulary model quality from same-input A16/A8 golden exports."""
import argparse
import json
from pathlib import Path
import numpy as np


def load(directory, expected_tokens=None):
    tokens = np.fromfile(directory / "tokens.bin", dtype="<u4")
    if expected_tokens is not None and not np.array_equal(tokens, expected_tokens):
        raise ValueError(f"{directory}: token ids differ")
    if len(tokens) <= 256:
        raise ValueError(f"{directory}: use >256 tokens to exercise A8 prefill")
    size = (directory / "logits.bin").stat().st_size
    if size % (4 * len(tokens)):
        raise ValueError(f"{directory}: logits length is not [tokens, vocab] f32")
    vocab = size // (4 * len(tokens))
    if not vocab or int(tokens.max()) >= vocab:
        raise ValueError(f"{directory}: invalid vocabulary/tokens")
    return tokens, np.memmap(directory / "logits.bin", dtype="<f4", mode="r", shape=(len(tokens), vocab))


def logp(row):
    row = row.astype(np.float64)
    if not np.isfinite(row).all():
        raise ValueError("nonfinite logits")
    row -= row.max()
    row -= np.log(np.exp(row).sum())
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a16", required=True, type=Path)
    p.add_argument("--a8", required=True, type=Path)
    p.add_argument("--golden", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-nat", type=float, default=0.005)
    a = p.parse_args()
    if not np.isfinite(a.max_nat) or a.max_nat < 0:
        p.error("--max-nat must be finite and nonnegative")
    protected = {directory / name for directory in (a.a16, a.a8, a.golden) for name in ("tokens.bin", "logits.bin")}
    if a.out.resolve() in {path.resolve() for path in protected}:
        p.error("--out cannot overwrite an input artifact")
    a.out.unlink(missing_ok=True)
    tokens, base = load(a.a16)
    _, candidate = load(a.a8, tokens)
    _, golden = load(a.golden, tokens)
    if base.shape != candidate.shape or base.shape != golden.shape:
        raise ValueError("logits shapes differ")
    sums = dict(a16_nll=0., a8_nll=0., golden_nll=0., golden_kl_a16=0., golden_kl_a8=0., a16_kl_a8=0.)
    agree = dict(golden_top1_a16=0, golden_top1_a8=0, a16_top1_a8=0)
    for r in range(len(tokens)):
        b, c, g = map(logp, (base[r], candidate[r], golden[r]))
        sums["golden_kl_a16"] += float(np.dot(np.exp(g), g-b))
        sums["golden_kl_a8"] += float(np.dot(np.exp(g), g-c))
        sums["a16_kl_a8"] += float(np.dot(np.exp(b), b-c))
        agree["golden_top1_a16"] += int(g.argmax() == b.argmax())
        agree["golden_top1_a8"] += int(g.argmax() == c.argmax())
        agree["a16_top1_a8"] += int(b.argmax() == c.argmax())
        if r + 1 < len(tokens):
            for key, row in (("a16_nll", b), ("a8_nll", c), ("golden_nll", g)):
                sums[key] -= float(row[tokens[r+1]])
    metrics = {key: value / (len(tokens)-1 if key.endswith("nll") else len(tokens)) for key, value in sums.items()}
    metrics.update({key: value/len(tokens) for key, value in agree.items()})
    metrics["nll_delta"] = metrics["a8_nll"]-metrics["a16_nll"]
    metrics["golden_kl_delta"] = metrics["golden_kl_a8"]-metrics["golden_kl_a16"]
    gate = metrics["nll_delta"] <= a.max_nat and metrics["golden_kl_delta"] <= a.max_nat and metrics["a16_kl_a8"] <= a.max_nat
    result = dict(schema="cuteafd.exl3-a8.model-quality/1", positions=len(tokens), vocab=base.shape[1], full_vocabulary=True,
                  sources={"a16":str(a.a16),"a8":str(a.a8),"golden":str(a.golden)}, metrics=metrics,
                  max_nat=a.max_nat, numerics_gate=gate, note="Numerics only; performance/tool gates remain separate.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
