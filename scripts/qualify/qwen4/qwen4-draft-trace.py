#!/usr/bin/env python3
"""Check the shared draft policy's models against a serve-qwen4 --mtp trace.

serve-qwen4 writes CUTEAFD_SPECULATION_TRACE (JSON lines): one record per verify
cycle with the request ids, rows, each sequence's drafts ("depths") and kept
rows, the verify / MTP / whole-cycle ms, the cost model's verify prediction
before it folded the step in, its fit, the conditional acceptance rates the
plan started from (before the online calibration) and the calibration's
fit. This prints, per number of sequences in the step:

- verify ms observed vs predicted by rows (the online fit, draft_policy.rs),
- MTP ms by chained steps and host ms by kept rows,
- acceptance: traced (uncalibrated) rate vs observed frequency, binned
  (positions count when drafted and every earlier draft was kept),
- emitted tokens per ms and the depth mix.
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict


def median(values):
    return statistics.median(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", nargs="+")
    parser.add_argument("--min-count", type=int, default=10, help="hide cells with fewer cycles")
    args = parser.parse_args()
    for path in args.trace:
        cycles = [json.loads(line) for line in open(path)]
        print(f"{path}: {len(cycles)} cycles")
        for seqs in sorted({c["seqs"] for c in cycles}):
            group = [c for c in cycles if c["seqs"] == seqs]
            kept = sum(sum(c["kept"]) for c in group)
            wall = sum(c["cycle_ms"] for c in group)
            depths = Counter(d for c in group for d in c["depths"])
            total = sum(depths.values())
            print(f"  {seqs} sequences: {len(group)} cycles, {kept / wall:.3f} tokens/ms, depths "
                  + " ".join(f"{d}:{100 * n / total:.0f}%" for d, n in sorted(depths.items())))
            by_rows = defaultdict(list)
            for c in group:
                by_rows[c["rows"]].append(c)
            cells = [f"{rows}:{median([c['verify_ms'] for c in v]):.1f}/{median([c['predicted_ms'] for c in v]):.1f}"
                     for rows, v in sorted(by_rows.items()) if len(v) >= args.min_count]
            print("    verify ms by rows, observed/predicted: " + " ".join(cells))
            by_steps = defaultdict(list)
            for c in group:
                if c["draft_steps"]:
                    by_steps[c["draft_steps"]].append(c["draft_ms"])
            print("    MTP ms by steps: " + " ".join(f"{s}:{median(v):.2f}" for s, v in sorted(by_steps.items())
                                                    if len(v) >= args.min_count))
            by_kept = defaultdict(list)
            for c in group:
                by_kept[sum(c["kept"])].append(c["cycle_ms"] - c["verify_ms"] - c["draft_ms"])
            print("    host ms by kept rows: " + " ".join(f"{k}:{median(v):.2f}" for k, v in sorted(by_kept.items())
                                                        if len(v) >= args.min_count))
            bins = defaultdict(lambda: [0, 0.0, 0])
            for c in group:
                for depth, kept_rows, rates in zip(c["depths"], c["kept"], c.get("rates") or []):
                    if kept_rows == 0:
                        continue  # the request finished in this cycle
                    for position in range(1, depth + 1):
                        if kept_rows < position:
                            break
                        cell = bins[min(int(rates[position - 1] * 10), 9)]
                        cell[0] += 1
                        cell[1] += rates[position - 1]
                        cell[2] += kept_rows - 1 >= position
            if bins:
                print("    acceptance predicted/observed (n): " + " ".join(
                    f"{p / n:.2f}/{ok / n:.2f}({n})" for _, (n, p, ok) in sorted(bins.items()) if n >= args.min_count))
        if cycles and "fit" in cycles[-1]:
            (a, b, c), (d, e) = cycles[-1]["fit"]
            print(f"  last fit: verify {a:.2f} ms + {b:.3f} x table + {c:.2f} ms per extra sequence; "
                  f"MTP {d:.2f} ms + {e:.2f} ms per step")
        if cycles and "calibration" in cycles[-1]:
            a, b = cycles[-1]["calibration"]
            print(f"  last acceptance calibration: {a:.2f} + {b:.2f} x rate")


if __name__ == "__main__":
    main()
