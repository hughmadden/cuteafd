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

DeepSeek V4 keys a shape by the resolved coordinator GPU count and Spark
attachment (`rtx1-spark`, `rtx1-local`, `rtx2-spark`, `rtx2-local`), and
captures different graphs per rank, so `--layout` plus `--list` replaces that
one shape's per-rank arrays in the checked-in `v4_prewarm.json` and marks it
traced: devices map to ranks in ascending device order, `reserve_base_bytes` is
the emitted prewarm set's own measured captures (the trace's whole charged
arena also holds lazy captures, which the shared reserve-extra policy covers),
`reserve_extra_bytes` is the arena's 2 MiB allocation tail (the shared capture
headroom is separate), and `bootstrap_bytes` is the largest measured capture
unless the trace has none.
Every verify geometry the trace observed becomes an entry unless it is rare -
used once, or only inside one sampling interval, such as one panel or the 32K
context pass - which stays lazy under the shared capture headroom. A width-0
startup row follows a stricter test: it is warmed only when the trace replayed
it, so a row captured once and never looked up again stays lazy. Pins follow
sustained heat, plus two keys that cost every step: plain C1 (rows 1, width 0)
and the layout's dominant verify geometry pin whatever their heat class. A row
hot in one sampling interval, or merely recurring, is otherwise prewarmed but
stays evictable for ARC. Entries follow the fan-out order - pinned first, then
lifetime replay hits descending - so a reduced reserve warms the hottest rows
first. A sustained geometry is pinned, a regular one prewarmed unpinned, and
the heat report prints each class's keys and arena bytes. The engine warms the
entries its pool can express and skips the rest.
"""
import argparse
import ast
import json
import re
from pathlib import Path

CHUNK = 2 << 20
GLM5_FLASH_LAYOUTS = {"1", "1-sparks", "2", "2-sparks"}
EXTRA_BYTES = 2 << 20
GEOMETRY = re.compile(r"rows: (\d+), table_width: (\d+), table_stride: (\d+)")
# The prewarm arena's allocation tail; capture headroom is a shared policy key.
V4_TAIL = 2 << 20
# GraphKey's display shape; width 0 is the local-expert startup set or the tail.
V4_GEOMETRY = re.compile(r'rows: (\d+), chunked: \w+, attention: "([^"]+)", '
                         r"table_width: (\d+), table_stride: (\d+)")
# Families sharing one engine share its site names.
SITE_PREFIX = {"glm5": "glm5.", "mistral_large4": "glm5.",
               "deepseek_v4": "deepseek_v4."}
GLMF_GEOMETRY = re.compile(
    r"GraphKey \{ segment: \d+, rows: (\d+), spec: (true|false), long: (true|false), "
    r"pool_width: (\d+), page_stride: (\d+), pool_stride: (\d+) \}")
EXTRA_BYTES = 2 << 20
QWEN_GEOMETRY = re.compile(
    r"GraphKey \{ segment: (\d+), rows: (\d+), spec: (true|false), long: (true|false), "
    r"pool_width: (\d+), page_stride: (\d+), pool_stride: (\d+) \}")
QWEN_SHAPES = ("single_0spark", "single_sparks", "dual_0spark", "dual_sparks")


V41_DECODE_ROWS = 64
V41_REPLAY_ROWS = 128
V41_PRODUCERS = {"window", "compressor"}
V41_HEADS = {"target_head", "target_norm", "vocabulary_logits", "vocabulary_greedy"}
V41_LAYER_ROWS = {
    "lane_row", "lane_prefill", "attention.query", "attention.query.tp2_prefix",
    "attention.output", "backbone.shared", "router.remote", "router.local", "index.query",
    "projection.query.rank", "projection.query.gather",
    "projection.output.rank", "projection.output.gather", "tp2_ffn",
}


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


# Hottest class wins when ranks disagree, because one prewarm set is shared.
HEAT_ORDER = {"rare": 0, "regular": 1, "sustained": 2, "startup": 3}


def heat(graph):
    """The startup class of one traced geometry.

    `rare` is a key the trace used once - a capture with no later hit - or only
    inside a single sampling interval, which is one point or panel such as the
    32K context pass; it stays lazy under the shared capture headroom.
    `sustained` is the rolling-window heat a pin is earned by, and `regular` is
    everything between: worth prewarming, not worth permanent residency.
    """
    histogram = graph.get("hit_histogram", {})
    counts = histogram.get("counts", [])
    intervals = sum(int(count) > 0 for count in counts) if counts else int(int(graph.get("count", 0)) > 0)
    if int(graph.get("hits", 0)) == 0 or intervals <= 1:
        return "rare"
    return "sustained" if sustained(graph) else "regular"


def geometry(family, shape):
    """`(rows, table_width)` of a captured segment's display shape."""
    if family != "deepseek_v4":
        match = GEOMETRY.search(shape)
        if match is None:
            raise ValueError(f"cannot parse {family} geometry: {shape!r}")
        rows, width, stride = map(int, match.groups())
        if rows <= 0 or width <= 0 or stride != width:
            raise ValueError("unsupported prewarm geometry")
        return rows, width
    match = V4_GEOMETRY.search(shape)
    if match is None:
        raise ValueError(f"cannot parse deepseek_v4 geometry: {shape!r}")
    rows, width, stride = int(match[1]), int(match[3]), int(match[4])
    if rows <= 0 or width < 0 or stride != width:
        raise ValueError("unsupported prewarm geometry")
    return rows, width


def prewarmable(family, shape):
    """Whether a captured key can stand as a prewarm entry.

    V4 keeps tails eager (#1638), so a width-0 tail capture must not become a
    startup entry; every other width-0 key is the local-expert/TP2 startup set
    (a 1-RTX shape captures `local-experts`, a 2-RTX one `tp2-broadcast` and
    `tp2-replicated`), and their geometry is the same `(rows, 0)` key.
    """
    if family != "deepseek_v4":
        return True
    match = V4_GEOMETRY.search(shape)
    return match is None or int(match[3]) > 0 or match[2] != "tail"


def generate_deepseek_v4(traces, layout, existing, coverage=0.999, pin_frequency=0.9,
                         pin_report=None, heat_report=None):
    """Replace one deepseek_v4 layout in the checked-in per-rank prewarm list."""
    if layout is None or existing is None:
        raise ValueError("deepseek_v4 needs --layout and --list")
    if existing.get("family") != "deepseek_v4":
        raise ValueError(f"list is for {existing.get('family')}, not deepseek_v4")
    prior = existing.get("layouts", {}).get(layout)
    if prior is None:
        raise ValueError(f"list has no layout {layout!r}")
    ranks = len(prior["reserve_base_bytes"])
    if len(prior["bootstrap_bytes"]) != ranks:
        raise ValueError(f"layout {layout} has inconsistent rank arrays")
    if not 0 < coverage <= 1 or not 0 < pin_frequency <= 1:
        raise ValueError("coverage and pin frequency must be in (0, 1]")
    prefix = SITE_PREFIX["deepseek_v4"]
    counts = {}
    reserves = {}
    captures = {}
    # Per device and geometry, the largest capture total any one snapshot saw.
    spreads = {}
    matched = set()
    reasons = {}
    # The startup class of every observed geometry, hottest rank wins.
    classes = {}
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
        device_bytes = {}
        for graph in trace["graphs"]:
            if not graph["site"].startswith(prefix):
                continue
            matched.add(device)
            if not prewarmable("deepseek_v4", graph["shape"]):
                continue
            captures[device] = max(captures.get(device, 0), int(graph.get("capture_bytes") or 0))
            key = geometry("deepseek_v4", graph["shape"])
            device_bytes[key] = device_bytes.get(key, 0) + int(graph.get("capture_bytes") or 0)
            cycles = graph.get("cycles")
            count = sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify")) if cycles is not None else int(graph["count"])
            if count > 0:
                device_counts[key] = device_counts.get(key, 0) + count
                if version == 1:
                    legacy.add(key)
        for key, value in device_bytes.items():
            spreads[(device, key)] = max(spreads.get((device, key), 0), value)
        for key, count in device_counts.items():
            counts[key] = max(counts.get(key, 0), count)
    if not counts:
        raise ValueError("trace contains no matching graph identities")
    for device, (_, trace) in latest.items():
        for graph in trace["graphs"]:
            if graph["site"].startswith(prefix) and prewarmable("deepseek_v4", graph["shape"]):
                key = geometry("deepseek_v4", graph["shape"])
                identity = (device, graph["site"], graph.get("bank", 0), graph.get("key_id", graph["shape"]))
                identities[identity] = (key, graph)
    for identity, (key, graph) in identities.items():
        cycles = graph.get("cycles", {})
        if not any(int(cycles.get(cycle, 0)) > 0 for cycle in ("decode", "verify")):
            continue
        # A width-0 row is a startup row once the trace proves a replay; a
        # capture with no later hit stays an ordinary lazy graph. Interval
        # concentration is a verify-geometry test: a short panel still proves a
        # decode row is reachable at that layout. The hottest rank decides a
        # shared entry's class.
        observed = heat(graph)
        if key[1] == 0:
            observed = "startup" if int(graph.get("hits", 0)) > 0 else "rare"
        if key not in classes or HEAT_ORDER[observed] > HEAT_ORDER[classes[key]]:
            classes[key] = observed
        why = reasons.setdefault(key, set())
        histogram = graph.get("hit_histogram", {})
        histogram_end = int(histogram.get("start_ms", 0)) + len(histogram.get("counts", [])) * int(histogram.get("interval_ms", 0))
        if histogram_end >= latest[identity[0]][0] and sustained(graph):
            why.add("sustained_heat")
        if int(graph.get("evictions", 0)) >= 2 and int(graph.get("recaptures", 0)) >= 2:
            why.add("repeated_recapture")
    # Startup residency is for heat the trace proves. A sustained verify
    # geometry is pinned, a regular one is prewarmed unpinned, and a rare
    # one - captured once, or used only inside one sampling interval such
    # as the 32K context pass - stays lazy under the shared capture
    # headroom. A width-0 startup row is kept when the trace replayed it,
    # but it earns a pin only on sustained heat: a row hot in one interval,
    # or merely recurring, lives in the prewarmed set and stays evictable
    # so ARC can reclaim it if serving never revisits it. One captured once
    # and never replayed is neither warmed nor pinned. A thrashing key is
    # kept even when rare, because the trace shows the churn rather than a
    # single use. Two keys are the cost of every step, so they pin whatever
    # their heat class: plain C1 (rows 1, width 0) and the layout's
    # dominant verify geometry, the width>0 key with the most lifetime hits
    # in the latest device snapshots. Entries keep the fan-out order -
    # pinned first, then lifetime replay hits descending - so a reduced
    # reserve warms the hottest rows before the list ends. Order never
    # changes selection, base or pin flags.
    selected = {key for key in counts
                if classes.get(key, "rare") != "rare" or reasons.get(key)}
    if (1, 0) in counts:
        selected.add((1, 0))
        reasons.setdefault((1, 0), set()).add("c1_decode")
    lifetime_hits = {}
    for key, graph in identities.values():
        lifetime_hits[key] = lifetime_hits.get(key, 0) + int(graph.get("hits", 0))
    verify_hits = {key: hits for key, hits in lifetime_hits.items() if key[1] > 0}
    if verify_hits:
        dominant = max(verify_hits, key=lambda key: (verify_hits[key], key))
        selected.add(dominant)
        reasons.setdefault(dominant, set()).add("dominant_verify")
    peak = max(counts.values())
    minimum = min(counts, key=lambda key: (key[0] * key[1], key))
    if minimum in legacy and counts[minimum] >= peak * pin_frequency:
        reasons.setdefault(minimum, set()).add("legacy_frequency")
    # Fan-out order: the pinned set first, then lifetime replay hits
    # descending, so a shrunken reserve keeps what serving replays most.
    order = sorted(selected, key=lambda key: (not reasons.get(key),
                                              -lifetime_hits.get(key, 0), key[1], key[0]))
    entries = [{"rows": rows, "table_width": width, "pin": bool(reasons.get((rows, width)))}
               for rows, width in order]
    if pin_report is not None:
        for key in sorted(selected):
            if not reasons.get(key):
                continue
            for device in sorted(reserves):
                graphs = [graph for identity, (shape, graph) in identities.items()
                          if identity[0] == device and shape == key]
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
    devices = sorted(matched)
    if len(devices) != ranks:
        raise ValueError(f"layout {layout} traces {len(devices)} matching devices, needs {ranks}")
    if heat_report is not None:
        for device in devices:
            totals = {}
            for key, observed in classes.items():
                bucket = totals.setdefault(observed, [0, 0, False])
                bucket[0] += 1
                bucket[1] += spreads.get((device, key), 0)
                bucket[2] = bucket[2] or key in selected
            for observed in ("startup", "sustained", "regular", "rare"):
                if observed in totals:
                    observed_keys, arena, kept = totals[observed]
                    heat_report.append({"device": device, "heat": observed,
                                        "keys": observed_keys, "arena_bytes": arena,
                                        "packed_bytes": packed(arena), "selected": kept})
    base = [packed(sum(spreads.get((device, key), 0) for key in selected))
            for device in devices]
    # The prewarm arena's tail only; runtime capture headroom is charged by
    # the shared GPU_GRAPH_RESERVE_EXTRA/_CAP policy, never hand-picked here.
    extra = [V4_TAIL] * ranks
    # One measured capture estimates the next; a trace without capture
    # shares keeps the checked-in number.
    bootstrap = [min(packed(captures.get(device, 0)) or prior["bootstrap_bytes"][rank], base[rank] + extra[rank])
                 for rank, device in enumerate(devices)]
    layouts = dict(existing["layouts"])
    layouts[layout] = {"traced": True,
                       "reserve_base_bytes": base, "reserve_extra_bytes": extra,
                       "bootstrap_bytes": bootstrap, "entries": entries}
    return {"schema_version": existing.get("schema_version", 1), "family": "deepseek_v4",
            "layouts": layouts}


def v41_geometry(shape, allowed, site):
    """Decode native Debug keys by site, never by guessing an integer axis.

    Pointer/fingerprint axes identify executables, but startup consumes row
    preferences and (request count, draft width) preferences, not addresses.
    """
    def integers(value, length=None):
        return isinstance(value, (tuple, list)) and (length is None or len(value) == length) and all(type(v) is int and v >= 0 for v in value)

    try:
        value = ast.literal_eval(shape)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"cannot parse V4.1 geometry at site {site}: {shape!r}") from error
    name = site.removeprefix("deepseek_v41.")
    if name == "dspark_cache" and value == ():
        return None
    if name in {"dspark_chain", "dspark_distributed_terminal"}:
        if name == "dspark_chain" and integers(value, 2):
            count, width = value
        elif name == "dspark_distributed_terminal" and integers(value, 3) and value[1] in (0, 1):
            width, _, count = value
        else:
            raise ValueError(f"invalid V4.1 draft geometry at {site}: {shape!r}")
        if not 1 <= count <= V41_DECODE_ROWS or width not in (5, 7):
            raise ValueError(f"invalid V4.1 draft axes at {site}: {shape!r}")
        return ("draft", count, width)
    if name in V41_PRODUCERS and integers(value, 2):
        rows = value[0]
    elif name in V41_LAYER_ROWS and integers(value, 2):
        rows = value[1]
    elif name in V41_HEADS | {"target_embedding", "engram.gate"} and type(value) is int:
        rows = value
    elif name == "index_selection" and integers(value) and len(value) >= 9:
        rows = value[1]
    elif name in {"sparse_attention", "dual_attention_tail"} and isinstance(value, tuple) and len(value) == 2 and type(value[0]) is int and integers(value[1]):
        fingerprint = value[1]
        if name == "dual_attention_tail" and len(fingerprint) >= 2:
            rows = fingerprint[-2]
        elif name == "sparse_attention" and len(fingerprint) >= 5:
            rows = fingerprint[2] if fingerprint[0] in (2**32 - 1, 2**64 - 1) else fingerprint[1]
        else:
            raise ValueError(f"invalid V4.1 fingerprint at {site}: {shape!r}")
    else:
        raise ValueError(f"cannot parse V4.1 geometry at site {site}: {shape!r}")
    # TP2 draft heads flatten count * width into a row axis.
    head_rows = {count * width for count in range(1, V41_DECODE_ROWS + 1) for width in (5, 7)}
    if rows not in allowed and not (name in V41_HEADS and rows in head_rows):
        raise ValueError(f"V4.1 row width {rows} at site {site} is outside the prewarm domain")
    return ("row", rows, 0)



def generate_glm5(traces, family, coverage=0.999, pin_frequency=0.9, pin_report=None, layout=None):
    prefix = SITE_PREFIX.get(family)
    if prefix is None:
        raise ValueError(f"no checked-in shape-list consumer for {family}; qwen4 is keyed by --shape")
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


def generate(traces, family, coverage=0.999, pin_frequency=0.9, prefill_rows=None, pin_report=None, layout=None, existing=None, heat_report=None):
    if family == "deepseek_v4":
        return generate_deepseek_v4(traces, layout, existing, coverage, pin_frequency, pin_report, heat_report)
    if family == "glm5_flash":
        if layout is not None:
            raise ValueError("--layout applies to mistral_large4 or deepseek_v41 only")
        return generate_glm5_flash(traces, coverage, pin_frequency)
    if family in SITE_PREFIX:
        return generate_glm5(traces, family, coverage, pin_frequency, pin_report=pin_report, layout=layout)
    if family != "deepseek_v41":
        raise ValueError("only glm5 and deepseek_v41 have a checked-in shape-list consumer")
    if not 0 < coverage <= 1 or not 0 < pin_frequency <= 1:
        raise ValueError("coverage and pin frequency must be in (0, 1]")
    if prefill_rows is not None and not 1 <= prefill_rows <= 4096:
        raise ValueError("prefill rows must be in 1..4096")
    prefill = {V41_REPLAY_ROWS}
    if prefill_rows is not None:
        prefill.add(prefill_rows)
    allowed = set(range(1, V41_DECODE_ROWS + 1)) | prefill
    counts, reasons, latest = {}, {}, {}
    observed = False
    heads, drafts = set(), set()
    arenas, largest = [], 0
    legacy = set()
    for trace in traces:
        version = trace.get("schema_version")
        if version not in (1, 2):
            raise ValueError("expected GRAPH_TRACE schema_version=1 or 2")
        cache = trace["cache"]
        device = int(cache["device"])
        arenas.append(int(cache["charged"]))
        inferred_end = max((int(g.get("hit_histogram", {}).get("start_ms", 0)) +
                            len(g.get("hit_histogram", {}).get("counts", [])) *
                            int(g.get("hit_histogram", {}).get("interval_ms", 0))
                            for g in trace["graphs"]), default=0)
        sampled = int(cache.get("sampled_at_ms", inferred_end))
        if device not in latest or sampled >= latest[device][0]:
            latest[device] = (sampled, trace)
        for site, values in trace.get("sites", {}).items():
            if site.startswith(family + "."):
                largest = max(largest, int(values["largest_bytes"]))
        device_counts = {}
        for graph in trace["graphs"]:
            if not graph["site"].startswith(family + "."):
                continue
            geometry = v41_geometry(graph["shape"], allowed, graph["site"])
            if geometry is None:
                continue
            observed = True
            kind, rows, width = geometry
            cycles = graph.get("cycles")
            count = (sum(int(cycles.get(c, 0)) for c in ("decode", "verify", "prefill"))
                     if cycles is not None else int(graph["count"]))
            if not count:
                continue
            if kind == "draft":
                key = (kind, rows, width)
                drafts.add(key)
            elif rows > V41_DECODE_ROWS:
                name = graph["site"].removeprefix(family + ".")
                # A flattened draft head is not a decode startup row. Its
                # actual observed request/width owner must survive coverage.
                if name in V41_HEADS and (cycles is None or not cycles.get("prefill", 0)):
                    heads.add(rows)
                    continue
                if rows not in prefill:
                    raise ValueError(f"unreachable prefill width {rows}")
                key = ("prefill", rows, 0)
            elif rows in prefill and cycles is not None and cycles.get("prefill", 0):
                key = ("prefill", rows, 0)
            else:
                key = ("row", rows, 0)
            device_counts[key] = device_counts.get(key, 0) + count
            if version == 1:
                legacy.add(key)
        for key, count in device_counts.items():
            counts[key] = max(counts.get(key, 0), count)
    if not observed:
        raise ValueError("trace contains no matching graph identities")
    for sampled, trace in latest.values():
        for graph in trace["graphs"]:
            if not graph["site"].startswith(family + "."):
                continue
            geometry = v41_geometry(graph["shape"], allowed, graph["site"])
            if geometry is None:
                continue
            kind, rows, width = geometry
            key = ("prefill", rows, 0) if kind == "row" and rows > V41_DECODE_ROWS else geometry
            if key not in counts:
                continue
            histogram = graph.get("hit_histogram", {})
            end = int(histogram.get("start_ms", 0)) + len(histogram.get("counts", [])) * int(histogram.get("interval_ms", 0))
            why = reasons.setdefault(key, set())
            if end >= sampled and sustained(graph):
                why.add("sustained_heat")
            if int(graph.get("evictions", 0)) >= 2 and int(graph.get("recaptures", 0)) >= 2:
                why.add("repeated_recapture")
    selected, covered = set(), 0
    total = sum(counts.values())
    for key in sorted(counts, key=lambda key: (-counts[key], key)):
        selected.add(key)
        covered += counts[key]
        if covered >= total * coverage:
            break
    selected.update(key for key, why in reasons.items() if why)
    for rows in heads:
        matching = {key for key in drafts if key[1] * key[2] == rows}
        if not matching:
            raise ValueError(f"flattened draft-head rows {rows} lack an observed draft_entries owner")
        selected.update(matching)
    for kind in ("row", "draft"):
        keys = [key for key in counts if key[0] == kind]
        if keys:
            minimum = min(keys, key=lambda key: (key[1] * max(1, key[2]), key))
            if minimum in legacy and counts[minimum] >= max(counts[k] for k in keys) * pin_frequency:
                reasons.setdefault(minimum, set()).add("legacy_frequency")
    entries = [{"rows": rows, "pin": bool(reasons.get((kind, rows, width)))}
               for kind, rows, width in sorted(selected) if kind == "row"]
    draft_entries = [{"requests": rows, "width": width, "pin": bool(reasons.get((kind, rows, width)))}
                     for kind, rows, width in sorted(selected) if kind == "draft"]
    # These are existing full-chunk/replay captures, not new graph shapes.
    # Their measured bytes are already inside the whole-device arena base.
    prefill_entries = [{"rows": rows, "pin": True} for rows in sorted(prefill)]
    if pin_report is not None:
        for device, (_, trace) in sorted(latest.items()):
            graphs = [g for g in trace["graphs"] if g["site"].startswith(family + ".")
                      and int(g.get("cycles", {}).get("prefill", 0)) > 0]
            pin_report.append({"device": device, "prefill_capture_bytes": sum(int(g.get("capture_bytes") or 0) for g in graphs),
                               "cost_basis": "measured step growth, split by estimate within shared steps"})
    return {"schema_version": 1, "family": family,
            "reserve_base_bytes": packed(max(arenas)), "reserve_extra_bytes": packed(largest),
            "entries": entries, "prefill_entries": prefill_entries, "draft_entries": draft_entries}


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


def qwen_replay(graph):
    """Replayed launches for one qwen4 graph identity (0 = captured, never used)."""
    replay = graph.get("hits")
    if replay is None:
        cycles = graph.get("cycles")
        replay = (sum(int(cycles.get(cycle, 0)) for cycle in ("decode", "verify"))
                  if cycles is not None else int(graph["count"])) - int(graph.get("misses", 0))
    return int(replay)


def generate_qwen(traces, shape, coverage=0.999, pin_report=None, summary=None):
    """One `QwenShapePrewarm` object for a resolved capture shape.

    Only identities with at least one replay can be selected: a capture is an
    event, not demand, so a shape captured once and never replayed stays lazy
    and its count and bytes are reported. The reserve base is the traced
    working set's arena: each identity of a selected shape's measured
    `capture_bytes` (every segment row, replayed or a startup-only sibling),
    summed per device, deduplicated by identity, rounded once to the 2 MiB
    chunk, and a selected identity without a measured share fails. The trace's
    charged high-water is never the base; `reserve_extra_bytes` is the fixed
    2 MiB capture headroom.
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
    # Only replayed graphs rank the working set: a startup capture is an event
    # (and a miss), never demand, so a zero-hit bootstrap shape can never be
    # selected (#1652).
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
            replay = qwen_replay(graph)
            # Every layer segment runs once per decode step: rank a shape by its
            # busiest segment rather than summing all of them.
            if replay > 0:
                counts[geometry] = max(counts.get(geometry, 0), replay)
    if not counts:
        raise ValueError("trace records no replayed graph identities")
    # A captured-but-never-replayed identity is first touch, not demand: it is
    # never selected, and its count and bytes are reported rather than hidden.
    zero_replay = [graph for (_, graph) in identities.values() if qwen_replay(graph) == 0]
    zero_replay_bytes = sum(int(graph.get("capture_bytes") or 0) for graph in zero_replay)
    if summary is not None:
        summary["zero_replay_identities"] = len(zero_replay)
        summary["zero_replay_bytes"] = zero_replay_bytes
    # A completed capture boundary with no new physical growth is allocator
    # reuse, not a free graph: the shared cache keeps such a record's estimate
    # for exactly this case. Charge a selected identity whose measured share is
    # zero the launch's measured mean capture on its device, so the traced
    # working set's base is never undercounted by the proportional split.
    measured = {}
    for (device, _, _, _), (_, graph) in identities.items():
        share = graph.get("capture_bytes")
        if share is None:
            continue
        captures, total = measured.get(device, (0, 0))
        measured[device] = (captures + max(int(graph.get("captures") or 0), 1),
                            total + int(share))
    mean = {device: total / captures for device, (captures, total) in measured.items()}
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
    zero_share = []
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
                share = int(share)
                if share == 0:
                    share = round(mean.get(device, 0.0))
                    zero_share.append(identity)
                total += share
        base.append(packed(total))
    zero_share_bytes = sum(round(mean.get(identity[0], 0.0)) for identity in zero_share)
    if summary is not None:
        summary["zero_share_identities"] = len(zero_share)
        summary["zero_share_bytes"] = zero_share_bytes
    extra = [EXTRA_BYTES for _ in devices]
    provenance = (f"GRAPH_TRACE {len(traces)} snapshots at {shape}; devices {devices}; "
                  f"{len(selected)} selected shapes; capture base {base} + {EXTRA_BYTES} extra bytes; "
                  f"{len(zero_replay)} zero-replay identities ({zero_replay_bytes} bytes) stay lazy; "
                  f"{len(zero_share)} zero-share selected identities ({zero_share_bytes} bytes) "
                  f"charged at the device's measured mean capture")
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
                              family, coverage, pin_frequency, pin_report=pin_report, layout=layout)
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
                        choices=sorted((*SITE_PREFIX, "glm5_flash", "qwen4", "deepseek_v41")))
    parser.add_argument("--layout", choices=("1", "2", "rtx1-spark", "rtx1-local", "rtx2-spark", "rtx2-local"),
                        help="coordinator layout the trace was recorded on (mistral_large4), "
                             "or the deepseek_v4 layout key to replace in --list")
    parser.add_argument("--list", type=Path,
                        help="deepseek_v4 only: checked-in prewarm list the layout is replaced in")
    parser.add_argument("--shape", choices=QWEN_SHAPES,
                        help="qwen4 resolved capture shape (owner count, Sparks attached)")
    parser.add_argument("--prefill-rows", type=int, help="V4.1 exact current full prefill chunk width")
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
    heat_rows = []
    summary = {}
    if args.family == "qwen4" and args.shape is None:
        parser.error("--shape is required for qwen4")
    if args.family != "qwen4" and args.shape is not None:
        parser.error("--shape applies only to qwen4")
    if args.family != "qwen4" and args.merge is not None:
        parser.error("--merge applies only to qwen4")
    if (args.family == "deepseek_v4") != (args.list is not None):
        parser.error("--list applies only to deepseek_v4")
    if args.family == "deepseek_v4" and args.layout is None:
        parser.error("--layout is required for deepseek_v4")
    if args.family != "deepseek_v4" and args.list is not None:
        parser.error("--list applies only to deepseek_v4")
    layout = args.layout
    if args.family == "mistral_large4":
        if layout not in ("1", "2"):
            parser.error("mistral_large4 keys its list by layout: pass --layout 1 or 2")
        layout = int(layout)
    elif layout is not None and args.family != "deepseek_v4":
        parser.error("--layout applies to mistral_large4 and deepseek_v4 only")
    if (args.output is None) == (args.merge is None):
        parser.error("exactly one of --output or --merge is required")
    try:
        if args.family == "qwen4":
            traces = [json.loads(Path(spec).read_text()) for spec in args.trace]
            result = generate_qwen(traces, args.shape, args.coverage, report, summary)
        elif args.family == "deepseek_v4":
            result = generate_deepseek_v4([json.loads(Path(spec).read_text()) for spec in args.trace],
                                          layout, json.loads(args.list.read_text()),
                                          args.coverage, args.pin_frequency, report, heat_rows)
        elif args.family == "deepseek_v41":
            result = generate([json.loads(Path(path).read_text()) for path in args.trace],
                              args.family, args.coverage, args.pin_frequency, args.prefill_rows, report, layout)
        else:
            result = generate_layouts(layout_groups(args.trace, args.family),
                                      args.family, args.coverage, args.pin_frequency, report, layout)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        parser.error(str(error))
    if args.merge is not None:
        document = merge_qwen(json.loads(args.merge.read_text()), result)
        args.merge.write_text(json.dumps(document, indent=2) + "\n")
    else:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.family == "deepseek_v4":
        entry = result["layouts"][layout]
        verify = sum(1 for row in entry["entries"] if row["table_width"] > 0)
        print(f"{len(entry['entries'])} shapes ({verify} verify geometries, "
              f"{len(entry['entries']) - verify} width-0 startup rows); "
              f"base={entry['reserve_base_bytes']} "
              f"extra={entry['reserve_extra_bytes']} bootstrap={entry['bootstrap_bytes']} bytes; "
              f"pinned={sum(entry['pin'] for entry in entry['entries'])}")
        for row in heat_rows:
            print("heat " + json.dumps(row, sort_keys=True))
        for pin in report:
            print("pin " + json.dumps(pin, sort_keys=True))
        return
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
