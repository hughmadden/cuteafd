#!/usr/bin/env python3
"""Gate FP8 target-head promotion using BF16/all Release smoke report.json files.

Both reports must contain the same cold teacher-forced fidelity probe. KL is
the golden top-k plus tail-bucket KL, a lower bound on full-vocabulary KL.
Top-1 must hold at every BF16 position, and golden agreement must not fall.
Run: compare-fp8-head.py BF16/report.json ALL/report.json [--out gate.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def fidelity(report: dict) -> dict:
    if report["status"] != "done":
        raise ValueError("benchmark did not finish")
    check = next(c for c in report["baseline"]["quality"]["checks"] if c["id"] == "fidelity")
    metrics = check["metrics"]
    probe = metrics["probe"]
    if not probe["engine"] or probe["error"] or not probe["cold"] or probe["cached_tokens"]:
        raise ValueError("fidelity probe was not a successful cold scoring pass")
    rows = probe["rows"]
    if not rows or probe["scored"] != len(rows) or metrics["missing"]:
        raise ValueError("fidelity scoring rows are missing")
    positions = [r["position"] for r in rows]
    if len(set(positions)) != len(rows) or positions != list(range(positions[0], positions[0] + len(rows))):
        raise ValueError("fidelity positions are duplicated or noncontiguous")
    if any(not r["finite"] for r in rows):
        raise ValueError("fidelity contains nonfinite logits")
    if metrics["positions"] != len(rows) or not all(math.isfinite(metrics[k]) for k in ("kl", "nll", "top1")):
        raise ValueError("invalid fidelity metrics")
    return metrics


def compare(before: dict, after: dict) -> dict:
    for field in ("model", "revision", "hardware", "build"):
        if before["server"][field] != after["server"][field]:
            raise ValueError(f"comparison changes server {field}")
    a, b = fidelity(before), fidelity(after)
    if a["reference"] != b["reference"] or a["probe"]["prompt_ids"] != b["probe"]["prompt_ids"]:
        raise ValueError("comparison changes reference or prompt ids")
    ar, br = a["probe"]["rows"], b["probe"]["rows"]
    if [r["position"] for r in ar] != [r["position"] for r in br]:
        raise ValueError("comparison changes scored positions")
    flips = [x["position"] for x, y in zip(ar, br) if x["argmax"] != y["argmax"]]
    rise = b["kl"] - a["kl"]
    return {
        "passed": rise <= 0.005 and b["top1"] >= a["top1"] and not flips,
        "positions": len(ar),
        "bf16": {k: a[k] for k in ("kl", "nll", "top1", "ref_nll")},
        "all": {k: b[k] for k in ("kl", "nll", "top1", "ref_nll")},
        "kl_rise_nats": rise,
        "kl_limit_nats": 0.005,
        "kl_kind": "golden top-k plus tail bucket (full-vocabulary lower bound)",
        "bf16_top1_agreement": 1 - len(flips) / len(ar),
        "top1_flip_positions": flips,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bf16", type=Path)
    parser.add_argument("all", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = compare(json.loads(args.bf16.read_text()), json.loads(args.all.read_text()))
    except (KeyError, ValueError, StopIteration) as error:
        parser.error(str(error))
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
