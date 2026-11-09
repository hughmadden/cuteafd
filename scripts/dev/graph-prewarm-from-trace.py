#!/usr/bin/env python3
"""Turn device graph traces into a GLM shape prewarm list and reserve suggestion.

Example:
  python3 scripts/dev/graph-prewarm-from-trace.py --family glm5 \
    --trace trace.device-0.json --output prewarm.json

Input is the versioned GRAPH_TRACE JSON snapshot, not arbitrary log text.
Frequencies are aggregated by geometry, not bank id or layer. Pin candidates
must have both dominant frequency and the lowest row/table parameter rank.
Reserve suggestions use the maximum per-device arena, never a sum of ranks.
"""
import argparse
import json
import re
from pathlib import Path

CHUNK = 2 << 20
GEOMETRY = re.compile(r"rows: (\d+), table_width: (\d+), table_stride: (\d+)")


def packed(value):
    return (value + CHUNK - 1) // CHUNK * CHUNK


def generate(traces, family, coverage=0.999, pin_frequency=0.9):
    if family != "glm5":
        raise ValueError("only glm5 has a checked-in shape-list consumer")
    if not 0 < coverage <= 1 or not 0 < pin_frequency <= 1:
        raise ValueError("coverage and pin frequency must be in (0, 1]")
    counts = {}
    arenas = []
    largest = 0
    for trace in traces:
        if trace.get("schema_version") != 1:
            raise ValueError("expected GRAPH_TRACE schema_version=1")
        cache = trace["cache"]
        arenas.append(int(cache["charged"]))
        for site, values in trace.get("sites", {}).items():
            if site.startswith(family + "."):
                largest = max(largest, int(values["largest_bytes"]))
        # Snapshots for the same device may overlap; max, not sum, prevents
        # double-counting repeated snapshots from one trace session.
        device_counts = {}
        for graph in trace["graphs"]:
            if not graph["site"].startswith(family + "."):
                continue
            match = GEOMETRY.search(graph["shape"])
            if match is None:
                raise ValueError(f"cannot parse {family} geometry: {graph['shape']!r}")
            rows, width, stride = map(int, match.groups())
            if rows <= 0 or width <= 0 or stride != width:
                raise ValueError("unsupported prewarm geometry")
            key = (rows, width)
            cycles = graph.get("cycles")
            # Startup prewarm is attributed to Other, not request frequency.
            count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify")) if cycles is not None else int(graph["count"])
            if count > 0:
                device_counts[key] = device_counts.get(key, 0) + count
        for key, count in device_counts.items():
            counts[key] = max(counts.get(key, 0), count)
    if not counts:
        raise ValueError("trace contains no matching graph identities")
    ranked = sorted(counts, key=lambda key: (-counts[key], key[1], key[0]))
    total = sum(counts.values())
    covered = 0
    selected = []
    for key in ranked:
        selected.append(key)
        covered += counts[key]
        if covered >= total * coverage:
            break
    peak = max(counts.values())
    minimum = min(counts, key=lambda key: (key[0] * key[1], key))
    entries = [{"rows": rows, "table_width": width,
                "pin": (rows, width) == minimum and counts[(rows, width)] >= peak * pin_frequency}
               for rows, width in sorted(selected, key=lambda key: (key[1], key[0]))]
    base = packed(max(arenas))
    # One largest measured capture is the explicit capture-before-reuse
    # headroom; keep it separate from the observed arena base.
    extra = packed(largest)
    return {"schema_version": 1, "family": family,
            "reserve_base_bytes": base, "reserve_extra_bytes": extra,
            "entries": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["glm5"])
    parser.add_argument("--trace", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coverage", type=float, default=0.999)
    parser.add_argument("--pin-frequency", type=float, default=0.9)
    args = parser.parse_args()
    try:
        result = generate([json.loads(path.read_text()) for path in args.trace],
                          args.family, args.coverage, args.pin_frequency)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(result['entries'])} shapes; base={result['reserve_base_bytes']} "
          f"extra={result['reserve_extra_bytes']} bytes; "
          f"pinned={sum(entry['pin'] for entry in result['entries'])}")


if __name__ == "__main__":
    main()
