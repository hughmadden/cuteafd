#!/usr/bin/env python3
"""Propose a family prewarm list from GRAPH_TRACE snapshots.

Request frequency selects coverage. Schema 2 pins sustained rolling-window
heat or repeated eviction/recapture, and reports permanent pin cost separately
from measured step-boundary allocator growth. Review and qualify before use.
"""
import argparse
import json
import re
from pathlib import Path

CHUNK = 2 << 20
GEOMETRY = re.compile(r"rows: (\d+), table_width: (\d+), table_stride: (\d+)")
# Families sharing one engine share its site names.
SITE_PREFIX = {"glm5": "glm5.", "mistral_large4": "glm5."}
GLMF_GEOMETRY = re.compile(
    r"GraphKey \{ segment: \d+, rows: (\d+), spec: (true|false), long: (true|false), "
    r"pool_width: (\d+), page_stride: (\d+), pool_stride: (\d+) \}")


def packed(value):
    return (value + CHUNK - 1) // CHUNK * CHUNK


def sustained(graph):
    histogram = graph.get("hit_histogram", {})
    counts = histogram.get("counts", [])
    interval_ms = int(histogram.get("interval_ms", 10_000))
    start_ms = int(histogram.get("start_ms", 0))
    first = max(0, (int(graph.get("first_seen_ms", 0)) - start_ms + interval_ms - 1) // interval_ms)
    # The current (partial) interval cannot prove sustained heat.
    complete = counts[first:-1]
    return len(complete) >= 3 and sum(int(count) > 0 for count in complete) >= len(complete) * 0.6


def generate(traces, family, coverage=0.999, pin_frequency=0.9, pin_report=None, layout=None):
    if family == "glm5_flash":
        if layout is not None:
            raise ValueError("--layout applies to mistral_large4 only")
        return generate_glm5_flash(traces, coverage, pin_frequency)
    prefix = SITE_PREFIX.get(family)
    if prefix is None:
        raise ValueError(f"no checked-in shape-list consumer for {family}")
    if family == "mistral_large4":
        if layout not in (1, 2):
            raise ValueError("mistral_large4 keys its list by layout: pass --layout 1 or 2")
    elif layout is not None:
        raise ValueError("--layout applies to mistral_large4 only")
    # A two-rank trace is recorded on the context placement, whose per-rank key
    # carries the half table; a checked-in entry holds the width a step is
    # built with, which is what the engine warms.
    scale = 2 if layout == 2 else 1
    if not 0 < coverage <= 1 or not 0 < pin_frequency <= 1:
        raise ValueError("coverage and pin frequency must be in (0, 1]")
    counts = {}
    arenas = []
    reserves = {}
    reasons = {}
    # Costs/heat use only the latest complete inventory per device.
    identities = {}
    legacy = set()
    latest = {}
    for trace in traces:
        version = trace.get("schema_version")
        if version not in (1, 2):
            raise ValueError("expected GRAPH_TRACE schema_version=1 or 2")
        cache = trace["cache"]
        device = int(cache["device"])
        arenas.append(int(cache["charged"]))
        reserves[device] = max(reserves.get(device, 0), int(cache.get("reserve", 0)))
        # Snapshots enumerate the entire bounded cache. Missing identities in a
        # newer snapshot have been forgotten; old heat/cost must not survive.
        inferred_end = max((int(graph.get("hit_histogram", {}).get("start_ms", 0)) +
                            len(graph.get("hit_histogram", {}).get("counts", [])) *
                            int(graph.get("hit_histogram", {}).get("interval_ms", 0))
                            for graph in trace["graphs"]), default=0)
        snapshot_time = int(cache.get("sampled_at_ms", inferred_end))
        if device not in latest or snapshot_time >= latest[device][0]:
            latest[device] = (snapshot_time, trace)

        device_counts = {}
        for graph in trace["graphs"]:
            if not graph["site"].startswith(prefix):
                continue
            match = GEOMETRY.search(graph["shape"])
            if match is None:
                raise ValueError(f"cannot parse {family} geometry: {graph['shape']!r}")
            rows, width, stride = map(int, match.groups())
            if rows <= 0 or width <= 0 or stride != width:
                raise ValueError("unsupported prewarm geometry")
            if scale != 1 and width % 2:
                raise ValueError(f"{family} layout {layout} key width {width} is not an even half table")
            key = (rows, width * scale)
            cycles = graph.get("cycles")
            count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify")) if cycles is not None else int(graph["count"])
            if count > 0:
                device_counts[key] = device_counts.get(key, 0) + count
                if version == 1:
                    legacy.add(key)
        for key, count in device_counts.items():
            counts[key] = max(counts.get(key, 0), count)
    if not counts:
        raise ValueError("trace contains no matching graph identities")
    for device, (_, trace) in latest.items():
        for graph in trace["graphs"]:
            if graph["site"].startswith(prefix):
                rows, width, _ = map(int, GEOMETRY.search(graph["shape"]).groups())
                identity = (device, graph["site"], graph.get("bank", 0), graph.get("key_id", graph["shape"]))
                identities[identity] = ((rows, width * scale), graph)
    for identity, (key, graph) in identities.items():
        cycles = graph.get("cycles", {})
        if not any(int(cycles.get(cycle, 0)) > 0 for cycle in ("decode", "verify")):
            continue
        why = reasons.setdefault(key, set())
        histogram = graph.get("hit_histogram", {})
        histogram_end = int(histogram.get("start_ms", 0)) + len(histogram.get("counts", [])) * int(histogram.get("interval_ms", 0))
        if histogram_end >= latest[identity[0]][0] and sustained(graph):
            why.add("sustained_heat")
        if int(graph.get("evictions", 0)) >= 2 and int(graph.get("recaptures", 0)) >= 2:
            why.add("repeated_recapture")
    ranked = sorted(counts, key=lambda key: (-counts[key], key[1], key[0]))
    total = sum(counts.values())
    covered = 0
    selected = set()
    for key in ranked:
        selected.add(key)
        covered += counts[key]
        if covered >= total * coverage:
            break
    # A thrashing low-frequency key can fall outside frequency coverage.
    selected.update(key for key, why in reasons.items() if why)
    peak = max(counts.values())
    minimum = min(counts, key=lambda key: (key[0] * key[1], key))
    if minimum in legacy and counts[minimum] >= peak * pin_frequency:
        reasons.setdefault(minimum, set()).add("legacy_frequency")
    entries = [{"rows": rows, "table_width": width, "pin": bool(reasons.get((rows, width)))}
               for rows, width in sorted(selected, key=lambda key: (key[1], key[0]))]
    if pin_report is not None:
        for key in sorted(selected):
            if not reasons.get(key):
                continue
            for device in sorted(reserves):
                graphs = [graph for identity, (geometry, graph) in identities.items()
                          if identity[0] == device and geometry == key]
                if not graphs:
                    continue
                known = all("permanent_bytes" in graph for graph in graphs)
                cost = sum(int(graph["permanent_bytes"]) for graph in graphs) if known else None
                pin_report.append({"device": device, "rows": key[0], "table_width": key[1],
                                   "reasons": sorted(reasons[key]), "permanent_bytes": None,
                                   "observed_permanent_lower_bound_bytes": cost,
                                   "reserve_bytes": reserves[device], "reserve_fraction": None,
                                   "observed_reserve_fraction_lower_bound": cost / reserves[device] if cost is not None and reserves[device] else None,
                                   "capture_bytes": sum(int(graph.get("capture_bytes") or 0) for graph in graphs),
                                   "nodes": sum(int(graph.get("nodes") or 0) for graph in graphs),
                                   "executables": sum(int(graph.get("executables", 1)) for graph in graphs),
                                   "basis": graphs[0].get("basis"),
                                   "cost_basis": "observed_inventory_lower_bound_unknown_total" if known else "unknown_legacy_trace"})
    base = packed(max(arenas))
    # Runtime headroom is resolved by GPU_GRAPH_RESERVE_EXTRA, not the trace.
    extra = CHUNK
    result = {"schema_version": 1, "family": family}
    if layout is not None:
        result["layout"] = layout
    result.update({"reserve_base_bytes": base, "reserve_extra_bytes": extra, "entries": entries})
    return result


def glmf_identity(shape):
    """A GLM Flash graph identity as its stable key tuple.

    Short steps key neither the pool top-k's width nor the pool table stride
    (`GraphGeometry::keyed`), so those identities collapse to one shape.
    """
    match = GLMF_GEOMETRY.fullmatch(shape.strip())
    if match is None:
        raise ValueError(f"cannot parse glm5_flash graph identity: {shape!r}")
    rows, spec, long, width, page_stride, pool_stride = match.groups()
    rows, width, page_stride, pool_stride = (int(value) for value in (rows, width, page_stride, pool_stride))
    if rows <= 0 or page_stride <= 0:
        raise ValueError("unsupported prewarm geometry")
    if long == "false":
        width = pool_stride = 0
    elif pool_stride <= 0:
        raise ValueError("unsupported prewarm geometry")
    return (rows, spec == "true", long == "true", width, page_stride, pool_stride)


def generate_glm5_flash(traces, coverage=0.999, pin_frequency=0.9):
    """A GLM Flash startup list: shapes by request frequency, the measured
    arena of one device with the executables it held, and capture headroom
    sized from the same trace's request demand. Startup prewarm itself runs
    as Other, never as request demand."""
    if not 0 < coverage <= 1 or not 0 < pin_frequency <= 1:
        raise ValueError("coverage and pin frequency must be in (0, 1]")
    counts = {}
    arenas = []
    nodes = {}
    for trace in traces:
        if trace.get("schema_version") != 1:
            raise ValueError("expected GRAPH_TRACE schema_version=1")
        matching = [graph for graph in trace["graphs"] if graph["site"].startswith("glm5_flash.")]
        # The arena held `live_execs` executables; trace rows also include keys
        # whose capture was refused (they ran eagerly), which the arena never held.
        held = int(trace["cache"].get("pools", {}).get("live_execs") or 0)
        arenas.append((int(trace["cache"]["charged"]), held or len(matching)))
        device_counts = {}
        device_nodes = {}
        for graph in matching:
            key = glmf_identity(graph["shape"])
            cycles = graph.get("cycles")
            count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify")) \
                if cycles is not None else int(graph["count"])
            if count > 0:
                device_counts[key] = device_counts.get(key, 0) + count
                device_nodes[graph["shape"]] = key
        for key, count in device_counts.items():
            counts[key] = max(counts.get(key, 0), count)
        for shape, key in device_nodes.items():
            nodes[shape] = key
    if not counts:
        raise ValueError("trace contains no matching graph identities")
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    total = sum(counts.values())
    covered = 0
    selected = []
    for key in ranked:
        selected.append(key)
        covered += counts[key]
        if covered >= total * coverage:
            break
    peak = max(counts.values())
    pinned = {key for key in selected if counts[key] >= peak * pin_frequency}
    entries = [{"rows": key[0], "spec": key[1], "long": key[2], "pool_width": key[3],
                "page_stride": key[4], "pool_stride": key[5], "pin": key in pinned}
               for key in sorted(selected)]
    charged, executables = max(arenas)
    if charged <= 0 or executables <= 0:
        raise ValueError("trace has no measured GLM Flash arena")
    demanded = len(nodes)
    held_pinned = sum(1 for key in nodes.values() if key in pinned)
    # The launch's request demand is the arena its lazy captures need; the
    # pinned shapes already hold their share of it in the reserve base.
    demand = -(-charged * demanded // executables)
    base = -(-charged * held_pinned // executables)
    return {"schema_version": 1, "family": "glm5_flash",
            "measured_bytes": charged, "measured_executables": executables,
            "reserve_extra_bytes": packed(demand) - packed(base), "entries": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted((*SITE_PREFIX, "glm5_flash")))
    parser.add_argument("--layout", type=int, choices=(1, 2),
                        help="coordinator layout the trace was recorded on (mistral_large4)")
    parser.add_argument("--trace", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coverage", type=float, default=0.999)
    parser.add_argument("--pin-frequency", type=float, default=0.9, help="schema 1 legacy frequency threshold")
    args = parser.parse_args()
    report = []
    try:
        result = generate([json.loads(path.read_text()) for path in args.trace],
                          args.family, args.coverage, args.pin_frequency, report, args.layout)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    pinned = sum(entry["pin"] for entry in result["entries"])
    if "measured_bytes" in result:
        print(f"{len(result['entries'])} shapes; measured={result['measured_bytes']} bytes "
              f"for {result['measured_executables']} executables; "
              f"extra={result['reserve_extra_bytes']} bytes; pinned={pinned}")
    else:
        layout = f" layout={result['layout']};" if "layout" in result else ""
        print(f"{len(result['entries'])} shapes;{layout} base={result['reserve_base_bytes']} "
              f"extra={result['reserve_extra_bytes']} bytes; pinned={pinned}")
    for pin in report:
        print("pin " + json.dumps(pin, sort_keys=True))


if __name__ == "__main__":
    main()
