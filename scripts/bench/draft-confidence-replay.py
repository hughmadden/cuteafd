#!/usr/bin/env python3
"""Score draft-length priors offline on a fixed-K7 `draft_confidence` trace.

Every record of a fixed-K7 run is a fully verified block: its accepted count `a` (0..7) and
the selector features and history rates it was priced at. At temperature 0 the target's
output does not depend on how many drafts were verified, so one trace prices any length rule:
verifying `d` drafts yields `min(a, d) + 1` tokens in one cycle of `T(d + 1)` ms, where
`T(n) = --row-ms base,per_row` (`base + per_row * (n - 1)`).

Rows:
- ceiling: the oracle verifies exactly the accepted drafts, `a + 1` rows.
- fixed-7: every block verified (the run itself).
- each prior with the `simple` rule (the largest `d` whose joint acceptance `prod p_1..p_d`
  exceeds 0.5), with per-position Platt scaling fitted on the other folds (as the policy's
  online calibration would) and raw:
  - history: the request's own conditional rates (`DraftHistory::conditional`);
  - generic: the shared GLM-5.3 `SelectorFit` (the engine's fallback);
  - fitted: `calibrate-draft-confidence.py`'s fit, cross-validated by request.

Per position, each prior's log-loss and AUC over the observed positions (Platt, cross-validated).
Requests map to `--requests` lines by admission order; `--classes NAME:FIRST-LAST,...` names
their classes (1-based lines), and a line's `thinking.type` splits each class by thinking.

  draft-confidence-replay.py --trace k7.jsonl --requests requests.jsonl \\
      --classes code:1-16,prose:17-28,json:29-40 --json out.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parent / "calibrate-draft-confidence.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_draft_confidence", TOOL)
calibrate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibrate)

POSITIONS = 7
# The engine's fallback prior (shared/draft_confidence.rs SelectorFit::default).
GENERIC = dict(
    mean=[1.0337757155740168, 1.5042334365117003, 4.105818737585014, 0.5404882231666059,
          0.18709320564567553, 0.36878205711857714],
    scale=[0.6869594087005682, 0.8183292270430992, 3.7913657744791154, 0.6412433617632998,
           0.4768998480274627, 0.23888156204918426],
    intercept=1.3954686667753908,
    coefficients=[0.05344836366609415, -0.3418558408836222, 1.7687177070514315,
                  -0.5006880248156605, -0.21189320574726964, -0.09773211967803144])


def load(trace: Path, key: str | None):
    """Fixed-K7 blocks: request id, accepted count, censored flag, history [7], features [7, 4]."""
    blocks = []
    for line in trace.read_text().splitlines():
        row = json.loads(line)
        if row.get("type") != "draft_confidence":
            continue
        if key is not None and row["key"] != key:
            raise ValueError(f"trace key {row['key']} differs from {key}")
        if row["proposed"] != POSITIONS:
            continue
        if len(row["history"]) < POSITIONS or len(row["features"]) < POSITIONS:
            raise ValueError("a fixed-K7 record misses history or features")
        blocks.append(dict(request=int(row["request"]), accepted=int(row["accepted"]),
                           observed=int(row["observed"]), history=row["history"][:POSITIONS],
                           features=row["features"][:POSITIONS]))
    if not blocks:
        raise ValueError("no fixed-K7 draft_confidence records")
    return blocks


def request_lines(blocks, expected: int, first_id: int | None):
    """Admission id -> 0-based requests-file line: consecutive ids from the first replayed one."""
    ids = sorted({b["request"] for b in blocks})
    if first_id is None:
        first_id = ids[0] if ids[-1] - ids[0] < expected else ids[-1] - expected + 1
    return {i: i - first_id for i in ids if 0 <= i - first_id < expected}


def parse_classes(spec: str, count: int):
    names = ["all"] * count
    for part in filter(None, spec.split(",")):
        name, span = part.split(":")
        first, last = (int(v) for v in span.split("-"))
        for line in range(first, last + 1):
            names[line - 1] = name
    return names


def features(block):
    """The calibration tool's six features per position, at the block's own history rates."""
    return np.asarray([calibrate.transform(block["history"][i], i + 1, block["features"][i])
                       for i in range(POSITIONS)])


def logistic(x, model):
    z = model["intercept"] + ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(
        model["coefficients"])
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def observations(blocks, positions):
    """(block index, position) pairs a block observed, and their labels."""
    pairs, labels = [], []
    for b, block in enumerate(blocks):
        for i in range(min(block["observed"], POSITIONS)):
            if positions is None or i in positions:
                pairs.append((b, i))
                labels.append(float(i < block["accepted"]))
    return pairs, np.asarray(labels)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def platt_fit(x, y, regularization=1.0):
    """sigmoid(a x + b) by damped Newton steps, the slope held lightly toward one."""
    def objective(a, b):
        z = a * x + b
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * regularization * (a - 1) ** 2)

    a, b = 1.0, 0.0
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(a * x + b, -40, 40)))
        w = p * (1 - p)
        gradient = np.asarray([((p - y) * x).sum() + regularization * (a - 1), (p - y).sum()])
        hessian = np.asarray([[(w * x * x).sum() + regularization, (w * x).sum()], [(w * x).sum(), w.sum() + 1e-9]])
        step = np.linalg.solve(hessian, gradient)
        fraction, before = 1.0, objective(a, b)
        while objective(a - fraction * step[0], b - fraction * step[1]) > before and fraction > 1e-6:
            fraction *= 0.5
        a, b = a - fraction * step[0], b - fraction * step[1]
        if np.max(np.abs(fraction * step)) < 1e-9:
            break
    return a, b


def platt(raw, blocks, train):
    """Per-position Platt scaling of `raw` [blocks, 7] fitted on the `train` blocks' observations."""
    out = raw.copy()
    for i in range(POSITIONS):
        pairs, y = observations([blocks[b] for b in train], {i})
        if len(y) < 8 or len(set(y.tolist())) < 2:
            continue
        a, b = platt_fit(logit(np.asarray([raw[train[k], i] for k, _ in pairs])), y)
        out[:, i] = 1 / (1 + np.exp(-np.clip(a * logit(raw[:, i]) + b, -40, 40)))
    return out


def depth(p):
    """The simple rule: the most drafts whose joint acceptance exceeds one half."""
    joint, d = 1.0, 0
    for value in p:
        joint *= value
        if joint <= 0.5:
            break
        d += 1
    return d


def auc(scores, labels):
    """Rank AUC with tied scores averaged."""
    positive, negative = labels.sum(), len(labels) - labels.sum()
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def replay(blocks, lines, names, thinking, row_ms, folds, regularization):
    base, per_row = row_ms
    cost = lambda rows: base + per_row * (rows - 1)
    x = np.stack([features(b) for b in blocks])
    raw = {"history": np.asarray([b["history"] for b in blocks]),
           "generic": logistic(x, GENERIC),
           "fitted": np.full((len(blocks), POSITIONS), np.nan)}
    calibrated = {name: np.full_like(raw["history"], np.nan) for name in raw}
    line_of = np.asarray([lines[b["request"]] for b in blocks])
    fold_of = line_of % folds
    for fold in range(folds):
        train = np.flatnonzero(fold_of != fold)
        test = np.flatnonzero(fold_of == fold)
        pairs, y = observations([blocks[b] for b in train], None)
        design = np.asarray([x[train[b], i] for b, i in pairs])
        model, _, _ = calibrate.fit(design, y, regularization)
        # The fold's fit everywhere: in sample for its Platt scaling, out of sample where scored.
        fitted = logistic(x, model)
        raw["fitted"][test] = fitted[test]
        for name, values in (("history", raw["history"]), ("generic", raw["generic"]), ("fitted", fitted)):
            calibrated[name][test] = platt(values, blocks, train)[test]
    accepted = np.asarray([b["accepted"] for b in blocks])
    groups = {}
    for b in range(len(blocks)):
        line = line_of[b]
        for group in ("all", names[line], f"{names[line]}/{thinking[line]}"):
            groups.setdefault(group, []).append(b)
    rules = {"ceiling": lambda b: accepted[b], "fixed-7": lambda b: POSITIONS}
    for name in raw:
        rules[f"{name}"] = lambda b, name=name: depth(calibrated[name][b])
        rules[f"{name} raw"] = lambda b, name=name: depth(raw[name][b])
    table = {}
    for group, members in sorted(groups.items()):
        table[group] = {}
        for rule, choose in rules.items():
            drafts = np.asarray([choose(b) for b in members])
            tokens = np.minimum(accepted[members], drafts) + 1
            ms = cost(drafts + 1)
            table[group][rule] = dict(cycles=len(members), tokens_per_cycle=float(tokens.mean()),
                                      drafts_per_cycle=float(drafts.mean()), ms_per_cycle=float(ms.mean()),
                                      tok_s=float(1e3 * tokens.sum() / ms.sum()))
    quality = {}
    for name, p in calibrated.items():
        quality[name] = {}
        for i in range(POSITIONS):
            pairs, y = observations(blocks, {i})
            scores = np.asarray([p[b, i] for b, _ in pairs])
            if len(y) == 0:
                continue
            quality[name][i + 1] = dict(observed=len(y), accepted=float(y.mean()),
                                        logloss=calibrate.logloss(scores, y), auc=auc(scores, y))
    return table, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--requests", type=Path, required=True, help="the replayed requests file (JSON lines)")
    parser.add_argument("--classes", default="", help="NAME:FIRST-LAST,... over 1-based request lines")
    parser.add_argument("--key", help="refuse records of another drafter key")
    parser.add_argument("--first-id", type=int, help="the admission id of the first request line")
    parser.add_argument("--row-ms", default="33,11.3", help="cycle cost: base ms at one row, ms per extra row")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--regularization", type=float, default=10.0)
    parser.add_argument("--json", type=Path, help="write the tables here")
    args = parser.parse_args()
    requests = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    blocks = [b for trace in args.trace for b in load(trace, args.key)]
    lines = request_lines(blocks, len(requests), args.first_id)
    blocks = [b for b in blocks if b["request"] in lines]
    names = parse_classes(args.classes, len(requests))
    thinking = [(r.get("thinking") or {}).get("type", "default") for r in requests]
    row_ms = tuple(float(v) for v in args.row_ms.split(","))
    table, quality = replay(blocks, lines, names, thinking, row_ms, args.folds, args.regularization)
    print(f"{len(blocks)} fixed-K7 blocks from {len(set(lines[b['request']] for b in blocks))} of {len(requests)} "
          f"requests; T(n) = {row_ms[0]} + {row_ms[1]} (n - 1) ms")
    for group, rules in table.items():
        print(f"\n{group} ({rules['ceiling']['cycles']} blocks)")
        print(f"  {'rule':<14}{'tok/cycle':>10}{'drafts':>8}{'ms/cycle':>10}{'tok/s':>8}")
        for rule, row in rules.items():
            print(f"  {rule:<14}{row['tokens_per_cycle']:>10.2f}{row['drafts_per_cycle']:>8.2f}"
                  f"{row['ms_per_cycle']:>10.1f}{row['tok_s']:>8.1f}")
    print("\nper position (Platt, cross-validated): log-loss / AUC")
    for name, positions in quality.items():
        print(f"  {name:<8}" + "".join(f" {i}: {q['logloss']:.3f}/{q['auc']:.3f}" for i, q in positions.items()))
    if args.json:
        args.json.write_text(json.dumps(dict(row_ms=row_ms, table=table, quality=quality), indent=1) + "\n")


if __name__ == "__main__":
    main()
