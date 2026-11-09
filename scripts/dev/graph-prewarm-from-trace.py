#!/usr/bin/env python3
"""Propose a family prewarm list from GRAPH_TRACE snapshots.

Request frequency selects the traced working set. Schema 2 pins sustained
rolling-window heat or repeated eviction/recapture. Qwen4's reserve base is
the sum of the selected identities' measured `capture_bytes` per device,
rounded once, with a fixed 2 MiB capture headroom; the trace's charged
high-water is never used. Review and qualify before use.

Examples:
  python3 scripts/dev/graph-prewarm-from-trace.py --family glm5 \
    --trace trace.device-0.json --output prewarm.json
  python3 scripts/dev/graph-prewarm-from-trace.py --family glm5_flash \
    --trace 1-sparks=trace-1-sparks.json --trace 2=trace-2.json \
    --trace 2-sparks=trace-2-sparks.json --output prewarm.json

Input is the versioned GRAPH_TRACE JSON snapshot, not arbitrary log text.
Frequencies are aggregated by geometry, not bank id or layer. Pin candidates
must have both dominant frequency and the lowest row/table parameter rank.
Reserve suggestions use the maximum per-device arena, never a sum of ranks.
GLM Flash prices the base from the selected entries' measured capture bytes per
device (the maximum across devices and quants), not the trace's whole-session
arena: shapes outside the coverage set never prewarm and never charge.
GLM Flash keys its list by resolved card shape, `<devices>[-sparks]`: onboarding
and TP2 each add captured segments, so each shape's own capture demand prices
its own reserve and a launch reads only its own entry. A shape's entries are
the same for every quant of the checkpoint (a shape is a pure function of the
launch geometry); when a shape is traced on more than one quant, the costlier
quant's measured arena and largest capture price it. A shape with no trace
warms the engine's bootstrap set, never another shape's list.

Qwen4 keys its list by resolved capture shape (RTX count, Sparks attached):
pass `--shape` with `--merge qwen_prewarm.json` to install the generated object
into that shape's entry, replacing its `trace: null`; every other shape keeps
its own reserve and list. `--output` writes the generated object alone for
review. A traced shape's base is the per-device sum of its selected prewarm
identities' measured capture bytes (startup-only siblings included,
deduplicated by identity, rounded once to 2 MiB); selection ranks by replayed
hits only, so a zero-hit startup shape is never selected and a
captured-but-never-replayed identity stays lazy.
"""
import argparse
import json
import re
from pathlib import Path

CHUNK = 2 << 20
GLM5_FLASH_LAYOUTS = {"1", "1-sparks", "2", "2-sparks"}
EXTRA_BYTES = 2 << 20
GEOMETRY = re.compile(r"rows: (\d+), table_width: (\d+), table_stride: (\d+)")
# Families sharing one engine share its site names.
SITE_PREFIX = {"glm5": "glm5.", "mistral_large4": "glm5."}
GLMF_GEOMETRY = re.compile(
    r"GraphKey \{ segment: \d+, rows: (\d+), spec: (true|false), long: (true|false), "
    r"pool_width: (\d+), page_stride: (\d+), pool_stride: (\d+) \}")
EXTRA_BYTES = 2 << 20
QWEN_GEOMETRY = re.compile(
    r"GraphKey \{ segment: (\d+), rows: (\d+), spec: (true|false), long: (true|false), "
    r"pool_width: (\d+), page_stride: (\d+), pool_stride: (\d+) \}")
QWEN_SHAPES = ("single_0spark", "single_sparks", "dual_0spark", "dual_sparks")


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
    """A GLM Flash startup list: every selected shape prewarms at startup, the
    always-hot ones are pinned, and the reserve base is what those selected
    shapes measured, device by device, plus the fixed 2 MiB allocation tail.
    Startup prewarm itself runs as Other, never as request demand. The
    whole-session arena in `cache.charged` also holds captures outside the
    coverage set (startup warmups and one-off shapes this launch never
    prewarms), so it is not the base. Runtime capture headroom comes from the
    shared extra keys, not the trace."""
    if not 0 < coverage <= 1 or not 0 < pin_frequency <= 1:
        raise ValueError("coverage and pin frequency must be in (0, 1]")
    counts = {}
    for trace in traces:
        if trace.get("schema_version") != 2:
            raise ValueError("expected GRAPH_TRACE schema_version=2")
        matching = [graph for graph in trace["graphs"] if graph["site"].startswith("glm5_flash.")]
        device_counts = {}
        for graph in matching:
            key = glmf_identity(graph["shape"])
            cycles = graph.get("cycles")
            count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify")) \
                if cycles is not None else int(graph["count"])
            if count > 0:
                device_counts[key] = device_counts.get(key, 0) + count
        for key, count in device_counts.items():
            counts[key] = max(counts.get(key, 0), count)
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
    # Price exactly the entries this list prewarms: sum each selected shape's
    # measured capture bytes on one device, take the costliest quant traced for
    # that device, and keep one base per coordinator device. A selected shape
    # with no measured share means the trace cannot price this list; it is
    # missing data, never a zero.
    wanted = set(selected)
    shares = {}
    for trace in traces:
        device = int(trace["cache"].get("device", 0))
        seen = set()
        captured = 0
        executables = 0
        for graph in trace["graphs"]:
            if not graph["site"].startswith("glm5_flash."):
                continue
            key = glmf_identity(graph["shape"])
            if key not in wanted:
                continue
            capture = graph.get("capture_bytes")
            if capture is None:
                raise ValueError(f"GLM Flash selected shape {key} has no measured capture bytes "
                                 f"on device {device}")
            seen.add(key)
            captured += int(capture)
            executables += int(graph.get("executables", 1) or 0)
        if seen != wanted:
            raise ValueError(f"GLM Flash trace for device {device} lacks selected shapes "
                             f"{sorted(wanted - seen)}; its measured base is unknown")
        if captured > 0 and executables > 0:
            previous = shares.get(device)
            if previous is None or (packed(captured), executables) > previous:
                shares[device] = (packed(captured), executables)
    if not shares:
        raise ValueError("trace has no measured GLM Flash arena")
    devices = sorted(shares)
    if devices != list(range(len(devices))):
        raise ValueError(f"GLM Flash traces must cover devices 0..n-1; saw {devices}")
    measured_bytes = [shares[device][0] for device in devices]
    measured_executables = [shares[device][1] for device in devices]
    # Startup prewarms every selected shape, so the reserve base already holds
    # the whole request demand. The fixed allocation tail is the arena's own
    # headroom for its single physical allocation; runtime capture headroom is
    # the shared extra keys' (never a trace-derived largest capture).
    extra = CHUNK
    return {"schema_version": 1, "family": "glm5_flash",
            "measured_bytes": measured_bytes, "measured_executables": measured_executables,
            "reserve_extra_bytes": extra, "entries": entries}


def qwen_geometry(graph):
    match = QWEN_GEOMETRY.search(graph["shape"])
    if match is None:
        raise ValueError(f"cannot parse qwen4 identity: {graph['shape']!r}")
    segment, rows, spec, long, width, page_stride, pool_stride = match.groups()
    if int(rows) <= 0 or int(page_stride) <= 0 or int(pool_stride) <= 0:
        raise ValueError("unsupported prewarm geometry")
    return (int(rows), spec == "true", long == "true", int(width),
            int(page_stride), int(pool_stride))


def generate_qwen(traces, shape, coverage=0.999, pin_report=None):
    """One `QwenShapePrewarm` object for a resolved capture shape.

    The reserve base is the traced working set's arena: each selected
    identity's measured `capture_bytes`, summed per device, deduplicated by
    identity, rounded once to the 2 MiB chunk. Startup-only siblings (captured
    but never replayed) contribute too, and a selected identity without a
    measured share fails. The trace's charged high-water is never the base;
    `reserve_extra_bytes` is the fixed 2 MiB capture headroom.
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    counts = {}
    reserves = {}
    reasons = {}
    identities = {}
    latest = {}
    devices = set()
    for trace in traces:
        version = trace.get("schema_version")
        if version not in (1, 2):
            raise ValueError("expected GRAPH_TRACE schema_version=1 or 2")
        cache = trace["cache"]
        device = int(cache["device"])
        devices.add(device)
        reserves[device] = max(reserves.get(device, 0), int(cache.get("reserve", 0)))
        inferred_end = max((int(graph.get("hit_histogram", {}).get("start_ms", 0)) +
                            len(graph.get("hit_histogram", {}).get("counts", [])) *
                            int(graph.get("hit_histogram", {}).get("interval_ms", 0))
                            for graph in trace["graphs"]), default=0)
        snapshot_time = int(cache.get("sampled_at_ms", inferred_end))
        if device not in latest or snapshot_time >= latest[device][0]:
            latest[device] = (snapshot_time, trace)
    owners = 1 if shape.startswith("single") else 2
    devices = sorted(devices)
    if len(devices) != owners:
        raise ValueError(f"{shape} needs {owners} device traces, found {devices}")
    # Frequency and heat use only each device's latest complete inventory:
    # identities missing from it have been forgotten and must not survive.
    for device, (_, trace) in latest.items():
        for graph in trace["graphs"]:
            if not graph["site"].startswith("qwen4."):
                continue
            geometry = qwen_geometry(graph)
            identity = (device, graph["site"], graph.get("bank", 0),
                        graph.get("key_id", graph["shape"]))
            previous = identities.get(identity)
            if previous is None or (previous[1].get("capture_bytes") is None
                                    and graph.get("capture_bytes") is not None):
                identities[identity] = (geometry, graph)
            cycles = graph.get("cycles")
            count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify")) \
                if cycles is not None else int(graph["count"])
            if count <= 0:
                continue
            # Every layer segment runs once per decode step: rank a shape by its
            # busiest segment rather than summing all of them.
            counts[geometry] = max(counts.get(geometry, 0), count)
            identity = (device, graph["site"], graph.get("bank", 0),
                        graph.get("key_id", graph["shape"]))
            previous = identities.get(identity)
            if previous is None or (previous[1].get("capture_bytes") is None
                                    and graph.get("capture_bytes") is not None):
                identities[identity] = (geometry, graph)
    if not counts:
        raise ValueError("trace contains no matching graph identities")
    for identity, (key, graph) in identities.items():
        cycles = graph.get("cycles", {})
        if not any(int(cycles.get(cycle, 0)) > 0 for cycle in ("decode", "verify")):
            continue
        histogram = graph.get("hit_histogram", {})
        histogram_end = int(histogram.get("start_ms", 0)) + \
            len(histogram.get("counts", [])) * int(histogram.get("interval_ms", 0))
        if histogram_end >= latest[identity[0]][0] and sustained(graph):
            reasons.setdefault(key, set()).add("sustained_heat")
        if int(graph.get("evictions", 0)) >= 2 and int(graph.get("recaptures", 0)) >= 2:
            reasons.setdefault(key, set()).add("repeated_recapture")
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    total = sum(counts.values())
    covered = 0
    selected = set()
    for key in ranked:
        selected.add(key)
        covered += counts[key]
        if covered >= total * coverage:
            break
    # A thrashing low-frequency key can fall outside frequency coverage, but a
    # zero-hit startup shape is never demand and must not re-enter this way.
    selected.update(key for key, why in reasons.items() if why and key in counts)
    entries = [{"rows": rows, "spec": spec, "long": long, "pool_width": width,
                "page_stride": page_stride, "pool_stride": pool_stride,
                "pin": bool(reasons.get(key))}
               for key in sorted(selected)
               for rows, spec, long, width, page_stride, pool_stride in [key]]
    if pin_report is not None:
        for key in sorted(selected):
            if not reasons.get(key):
                continue
            for device in devices:
                graphs = [graph for identity, (geometry, graph) in identities.items()
                          if identity[0] == device and geometry == key]
                if not graphs:
                    continue
                cost = sum(int(graph["permanent_bytes"]) for graph in graphs) \
                    if all("permanent_bytes" in graph for graph in graphs) else None
                pin_report.append({"device": device, "shape": qwen_shape_text(key),
                                   "reasons": sorted(reasons[key]), "permanent_bytes": None,
                                   "observed_permanent_lower_bound_bytes": cost,
                                   "reserve_bytes": reserves[device], "reserve_fraction": None,
                                   "observed_reserve_fraction_lower_bound": cost / reserves[device] if cost is not None and reserves[device] else None,
                                   "capture_bytes": sum(int(graph["capture_bytes"]) for graph in graphs
                                                        if graph.get("capture_bytes") is not None),
                                   "null_capture_bytes": sum(1 for graph in graphs
                                                             if graph.get("capture_bytes") is None),
                                   "nodes": sum(int(graph.get("nodes") or 0) for graph in graphs),
                                   "executables": sum(int(graph.get("executables", 1)) for graph in graphs),
                                   "basis": graphs[0].get("basis"),
                                   "cost_basis": "observed_inventory_lower_bound_unknown_total" if cost is not None else "unknown_legacy_trace"})
    base = []
    for device in devices:
        total = 0
        for selected_key in sorted(selected):
            for identity, (geometry, graph) in identities.items():
                if identity[0] != device or geometry != selected_key:
                    continue
                share = graph.get("capture_bytes")
                if share is None:
                    raise ValueError(
                        f"{shape} device {device} selects {qwen_shape_text(selected_key)} "
                        f"without a measured capture share: {graph['shape']!r}")
                total += int(share)
        base.append(packed(total))
    extra = [EXTRA_BYTES for _ in devices]
    provenance = (f"GRAPH_TRACE {len(traces)} snapshots at {shape}; devices {devices}; "
                  f"{len(selected)} selected shapes; capture base {base} + {EXTRA_BYTES} extra bytes")
    return {"shape": shape, "reserve_base_bytes": base,
            "reserve_extra_bytes": extra,
            "trace": {"provenance": provenance, "entries": entries}}


def qwen_shape_text(key):
    rows, spec, long, width, page_stride, pool_stride = key
    return (f"rows={rows} spec={spec} long={long} pool_width={width} "
            f"page_stride={page_stride} pool_stride={pool_stride}")


def qwen_provenance(shapes):
    """The list document's provenance sentence for its current traced set."""
    traced = [prewarm["shape"] for prewarm in shapes if prewarm.get("trace") is not None]
    untraced = [prewarm["shape"] for prewarm in shapes if prewarm.get("trace") is None]
    text = ("Traced shapes regenerate from their own GRAPH_TRACE snapshots ("
            + ", ".join(traced) + ") with scripts/dev/graph-prewarm-from-trace.py; "
            "a traced shape's base is the per-device sum of its selected prewarm "
            "identities' measured capture bytes, including startup-only siblings, "
            "deduplicated by identity and rounded once to 2 MiB; selection ranks by "
            "replayed hits only, so a zero-hit startup shape is never selected and a "
            "captured-but-never-replayed identity stays lazy. Extra "
            "is the fixed 2 MiB capture headroom; the shared graph reserve extra adds "
            "room for runtime captures.")
    if untraced:
        text += (" Untraced (" + ", ".join(untraced) + ") keeps the bootstrap reserve "
                 "above and prewarms the 682-shape bootstrap startup set at launch with "
                 "a warning; an untraced shape never borrows another shape's list.")
    return text


def merge_qwen(document, generated):
    """Install one generated shape object into a checked-in prewarm list.

    Every other shape keeps its reserve and trace untouched, so a shape is only
    ever replaced by its own trace. The provenance sentence is derived from the
    installed set, which keeps the document reproducible from the generator.
    """
    merged = json.loads(json.dumps(document))
    shape = generated["shape"]
    if not any(prewarm["shape"] == shape for prewarm in merged["shapes"]):
        raise ValueError(f"prewarm list has no {shape} shape")
    merged["shapes"] = [generated if prewarm["shape"] == shape else prewarm
                        for prewarm in merged["shapes"]]
    merged["provenance"] = qwen_provenance(merged["shapes"])
    return merged


def layout_groups(specs, family):
    """Group `--trace` arguments by layout: `LAYOUT=PATH` for GLM Flash, a bare
    path (the one layout) for GLM. A GLM Flash key names a resolved card shape,
    `<devices>[-sparks]`; a shape left out warms the engine's bootstrap set, so
    a trace is required only for the shapes this run measured."""
    groups = {}
    for spec in specs:
        layout, path = spec.split("=", 1) if "=" in spec else ("min", spec)
        if family != "glm5_flash" and layout != "min":
            raise ValueError(f"{family} has one layout, not {layout!r}")
        if family == "glm5_flash" and layout not in GLM5_FLASH_LAYOUTS:
            raise ValueError(f"unknown GLM Flash layout {layout!r}; expected one of {sorted(GLM5_FLASH_LAYOUTS)}")
        groups.setdefault(layout, []).append(Path(path))
    return groups


def generate_layouts(groups, family, coverage=0.999, pin_frequency=0.9, pin_report=None, layout=None):
    """The checked-in document: one entry per traced shape, each priced from its
    own traces. A shape's pins and arena describe that shape's traffic; absent
    keys fall back to the engine's bootstrap set."""
    layouts = {name: generate([json.loads(path.read_text()) for path in paths],
                              family, coverage, pin_frequency, pin_report, layout)
               for name, paths in sorted(groups.items())}
    if family != "glm5_flash":
        return layouts.pop("min")
    for values in layouts.values():
        for key in ("schema_version", "family"):
            values.pop(key)
    return {"schema_version": 3, "family": family, "layouts": layouts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True,
                        choices=sorted((*SITE_PREFIX, "glm5_flash", "qwen4")))
    parser.add_argument("--layout", type=int, choices=(1, 2),
                        help="coordinator layout the trace was recorded on (mistral_large4)")
    parser.add_argument("--shape", choices=QWEN_SHAPES,
                        help="qwen4 resolved capture shape (owner count, Sparks attached)")
    parser.add_argument("--trace", action="append", required=True,
                        help="PATH, or LAYOUT=PATH for GLM Flash (repeatable)")
    parser.add_argument("--output", type=Path,
                        help="write the generated object here (one shape for qwen4, the list otherwise)")
    parser.add_argument("--merge", type=Path,
                        help="qwen4 only: install the generated shape into this prewarm list in place")
    parser.add_argument("--coverage", type=float, default=0.999)
    parser.add_argument("--pin-frequency", type=float, default=0.9, help="schema 1 legacy frequency threshold")
    args = parser.parse_args()
    report = []
    summary = {}
    if args.family == "qwen4" and args.shape is None:
        parser.error("--shape is required for qwen4")
    if args.family != "qwen4" and args.shape is not None:
        parser.error("--shape applies only to qwen4")
    if args.family != "qwen4" and args.merge is not None:
        parser.error("--merge applies only to qwen4")
    if (args.output is None) == (args.merge is None):
        parser.error("exactly one of --output or --merge is required")
    try:
        if args.family == "qwen4":
            traces = [json.loads(Path(spec).read_text()) for spec in args.trace]
            result = generate_qwen(traces, args.shape, args.coverage, report, summary)
        else:
            result = generate_layouts(layout_groups(args.trace, args.family),
                                      args.family, args.coverage, args.pin_frequency, report, args.layout)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        parser.error(str(error))
    if args.merge is not None:
        document = merge_qwen(json.loads(args.merge.read_text()), result)
        args.merge.write_text(json.dumps(document, indent=2) + "\n")
    else:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.family == "qwen4":
        print(f"{result['shape']}: {len(result['trace']['entries'])} shapes; "
              f"base={result['reserve_base_bytes']} extra={result['reserve_extra_bytes']} bytes; "
              f"pinned={sum(entry['pin'] for entry in result['trace']['entries'])}; "
              f"zero_replay={summary.get('zero_replay_identities', 0)} "
              f"({summary.get('zero_replay_bytes', 0)} bytes); "
              f"zero_share={summary.get('zero_share_identities', 0)} "
              f"({summary.get('zero_share_bytes', 0)} bytes filled)")
        for pin in report:
            print("pin " + json.dumps(pin, sort_keys=True))
        return
    values = result["layouts"].values() if "layouts" in result else [result]
    for values in values:
        pinned = sum(entry["pin"] for entry in values["entries"])
        if "measured_bytes" in values:
            print(f"{len(values['entries'])} shapes; measured={values['measured_bytes']} bytes "
                  f"for {values['measured_executables']} executables; "
                  f"extra={values['reserve_extra_bytes']} bytes; pinned={pinned}")
        else:
            layout = f" layout={values['layout']};" if "layout" in values else ""
            print(f"{len(values['entries'])} shapes;{layout} base={values['reserve_base_bytes']} "
                  f"extra={values['reserve_extra_bytes']} bytes; pinned={pinned}")
    for pin in report:
        print("pin " + json.dumps(pin, sort_keys=True))


if __name__ == "__main__":
    main()
