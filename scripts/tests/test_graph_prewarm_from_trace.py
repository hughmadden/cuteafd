"""Trace-derived startup lists preserve hot shapes and per-device ownership."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "dev" / "graph-prewarm-from-trace.py"
SPEC = importlib.util.spec_from_file_location("graph_prewarm", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def trace(device=0, charged=8 << 20):
    def graph(rows, width, count):
        return {"device": device, "site": "glm5.segment", "bank": 1,
                "shape": f"GlmGraphKey {{ layer: 0, rows: {rows}, table_width: {width}, table_stride: {width}, previous: First, head: false }}",
                "count": count}
    return {"schema_version": 1, "cache": {"device": device, "charged": charged},
            "sites": {"glm5.segment": {"largest_bytes": 3 << 20}},
            "graphs": [graph(1, 16, 100), graph(16, 32, 99), graph(64, 128, 1)]}


def test_frequency_and_low_parameter_pin_candidate():
    result = MODULE.generate([trace()], "glm5", coverage=0.99)
    assert result["entries"] == [{"rows": 1, "table_width": 16, "pin": True},
                                 {"rows": 16, "table_width": 32, "pin": False}]
    assert result["reserve_base_bytes"] == 8 << 20
    assert result["reserve_extra_bytes"] == 2 << 20


def test_devices_are_not_added_and_repeated_snapshots_do_not_duplicate_shapes():
    result = MODULE.generate([trace(), trace(), trace(1, 12 << 20)], "glm5")
    assert len(result["entries"]) == 3
    assert result["reserve_base_bytes"] == 12 << 20


def test_startup_prewarm_counts_do_not_select_cold_shapes():
    data = trace()
    for graph in data["graphs"]:
        graph["cycles"] = {"other": 100, "verify": graph["count"]}
    data["graphs"][2]["cycles"] = {"other": 1000}
    result = MODULE.generate([data], "glm5")
    assert len(result["entries"]) == 2
    assert all(entry["rows"] != 64 for entry in result["entries"])


def test_cold_low_rank_shape_is_not_pinned():
    data = trace()
    data["graphs"][0]["count"] = 1
    result = MODULE.generate([data], "glm5")
    assert not any(entry["pin"] for entry in result["entries"])


@pytest.mark.parametrize("change", [lambda t: t.update(schema_version=3),
                                    lambda t: t.update(graphs=[]),
                                    lambda t: t["graphs"][0].update(shape="bad")])
def test_invalid_trace_fails(change):
    data = trace()
    change(data)
    with pytest.raises(ValueError):
        MODULE.generate([data], "glm5")








@pytest.mark.parametrize("site,shape,expected", [
    ("window", "(16, 987654321)", ("row", 16, 0)),
    ("compressor", "(1, 998)", ("row", 1, 0)),
    ("index_selection", "[7, 16, 100, 200, 300, 1, 500, 512, 0, 1]", ("row", 16, 0)),
    ("sparse_attention", "(7, [7, 16, 100, 200, 300])", ("row", 16, 0)),
    ("sparse_attention", "(7, [18446744073709551615, 7, 16, 100, 200, 300])", ("row", 16, 0)),
    ("dual_attention_tail", "(7, [100, 200, 300, 16, 500])", ("row", 16, 0)),
    ("target_head", "112", ("row", 112, 0)),
    ("dspark_chain", "(16, 7)", ("draft", 16, 7)),
    ("dspark_distributed_terminal", "(5, 1, 16)", ("draft", 16, 5)),
    ("dspark_cache", "()", None),
])
def test_v41_native_site_axes(site, shape, expected):
    assert MODULE.v41_geometry(shape, set(range(1, 65)) | {128}, "deepseek_v41." + site) == expected


@pytest.mark.parametrize("site,shape", [
    ("dspark_chain", "(5, 16)"),
    ("dspark_distributed_terminal", "(7, 2, 16)"),
    ("window", "(987654321, 16)"),
    ("index_selection", "[7, 16]"),
    ("sparse_attention", "(7, [1, 2, 3])"),
    ("unknown", "16"),
])
def test_v41_unknown_or_invalid_axes_fail(site, shape):
    with pytest.raises(ValueError):
        MODULE.v41_geometry(shape, set(range(1, 65)), "deepseek_v41." + site)






def flash_graph(rows, spec, long, width, stride, pool_stride, count, segment=0,
                site="glm5_flash.decode", capture_bytes=1 << 20):
    return {"device": 0, "site": site, "bank": 1,
            "shape": f"GraphKey {{ segment: {segment}, rows: {rows}, spec: {spec}, long: {long}, "
                     f"pool_width: {width}, page_stride: {stride}, pool_stride: {pool_stride} }}",
            "count": count, "capture_bytes": capture_bytes, "executables": 1}


def flash_trace(graphs=None, charged=16 << 20, capture_bytes=1 << 20, device=0):
    return {"schema_version": 2, "cache": {"device": device, "charged": charged},
            "sites": {"glm5_flash.decode": {"largest_bytes": 3 << 20}},
            "graphs": graphs if graphs is not None else
            [flash_graph(4, "false", "false", 0, 64, 0, 100, segment=0,
                         capture_bytes=capture_bytes),
             flash_graph(4, "false", "false", 0, 64, 0, 90, segment=3,
                         capture_bytes=capture_bytes),
             flash_graph(16, "true", "false", 0, 128, 0, 30, segment=0,
                         capture_bytes=capture_bytes),
             flash_graph(1, "true", "true", 2048, 64, 16, 1, segment=2,
                         capture_bytes=capture_bytes)]}


def test_flash_shapes_aggregate_segments_and_pin_the_hot_ones():
    result = MODULE.generate([flash_trace()], "glm5_flash", coverage=0.99)
    assert result["family"] == "glm5_flash"
    # The base is the selected entries' measured capture bytes (three rows of
    # 1 MiB on device 0), not the 16 MiB whole-session arena the trace held.
    assert result["measured_bytes"] == [4 << 20]
    assert result["measured_executables"] == [3]
    # Startup prewarms every traced shape, so the reserve base holds the whole
    # request demand; the tail is the fixed 2 MiB allocation headroom.
    assert result["reserve_extra_bytes"] == 2 << 20
    # 99% coverage keeps the two hot shapes (190 and 30 hits); only the peak is pinned.
    assert result["entries"] == [
        {"rows": 4, "spec": False, "long": False, "pool_width": 0,
         "page_stride": 64, "pool_stride": 0, "pin": True},
        {"rows": 16, "spec": True, "long": False, "pool_width": 0,
         "page_stride": 128, "pool_stride": 0, "pin": False},
    ]


def test_flash_short_steps_collapse_the_unkeyed_pool_geometry():
    graphs = [flash_graph(4, "false", "false", 1, 64, 16, 10),
              flash_graph(4, "false", "false", 4, 64, 32, 5)]
    result = MODULE.generate([flash_trace(graphs)], "glm5_flash")
    assert result["entries"] == [{"rows": 4, "spec": False, "long": False, "pool_width": 0,
                                  "page_stride": 64, "pool_stride": 0, "pin": True}]


def test_flash_startup_prewarm_is_not_request_demand():
    graphs = flash_trace()["graphs"]
    for graph in graphs:
        graph["cycles"] = {"other": 1000}
    graphs[0]["cycles"] = {"other": 1000, "verify": 5}
    result = MODULE.generate([flash_trace(graphs)], "glm5_flash")
    assert [(entry["rows"], entry["pin"]) for entry in result["entries"]] == [(4, True)]


def test_flash_each_device_keeps_its_own_measured_base():
    first = flash_trace([flash_graph(4, "false", "false", 0, 64, 0, 10)],
                        charged=8 << 20)
    second = flash_trace([flash_graph(4, "false", "false", 0, 64, 0, 10,
                                      capture_bytes=3 << 20)],
                         charged=24 << 20, device=1)
    result = MODULE.generate([first, second], "glm5_flash")
    # One selected shape per device, 1 MiB and 3 MiB; each packs once to its
    # own 2 MiB / 4 MiB base, never scaled from the other rank's price.
    assert (result["measured_bytes"], result["measured_executables"]) == ([2 << 20, 4 << 20], [1, 1])


def test_flash_a_selected_shape_without_a_measured_share_fails():
    missing = flash_trace()
    missing["graphs"][0]["capture_bytes"] = None
    with pytest.raises(ValueError, match="no measured capture bytes"):
        MODULE.generate([missing], "glm5_flash")
    # A device trace that never captured one of the layout's selected shapes
    # cannot price that shape; the base would be short by an unknown share.
    incomplete = flash_trace([flash_graph(4, "false", "false", 0, 64, 0, 100)])
    with pytest.raises(ValueError, match="lacks selected shapes"):
        MODULE.generate([flash_trace(), incomplete], "glm5_flash")


@pytest.mark.parametrize("change", [
    lambda t: t["graphs"][0].update(shape="GlmGraphKey { rows: 1 }"),
    lambda t: t["graphs"][0].update(capture_bytes=None),
    lambda t: t["cache"].update(device=1),
    lambda t: t.update(graphs=[flash_graph(0, "false", "false", 0, 64, 0, 5)]),
    lambda t: t.update(graphs=[flash_graph(4, "false", "true", 1, 64, 0, 5)]),
])
def test_flash_invalid_trace_fails(change):
    data = flash_trace()
    change(data)
    with pytest.raises(ValueError):
        MODULE.generate([data], "glm5_flash")


def flash_trace_path(tmp_path, name, capture_bytes):
    path = tmp_path / name
    path.write_text(json.dumps(flash_trace(capture_bytes=capture_bytes)))
    return path


def test_flash_shapes_are_priced_separately_by_resolved_shape(tmp_path):
    one = flash_trace_path(tmp_path, "one.json", 1 << 20)
    two = flash_trace_path(tmp_path, "two.json", 3 << 20)
    groups = MODULE.layout_groups([f"1-sparks={one}", f"2={two}"], "glm5_flash")
    result = MODULE.generate_layouts(groups, "glm5_flash")
    assert result["schema_version"] == 3
    assert result["family"] == "glm5_flash"
    assert set(result["layouts"]) == {"1-sparks", "2"}
    # Each layout prices its own selected entries: the same four rows of
    # captures, 4 MiB at the 1 MiB row price and 12 MiB at the 3 MiB row price.
    assert result["layouts"]["1-sparks"]["measured_bytes"] == [4 << 20]
    assert result["layouts"]["2"]["measured_bytes"] == [12 << 20]
    # A shape entry carries only its own trace's prices, never the document's
    # schema or another shape's arena.
    assert "schema_version" not in result["layouts"]["1-sparks"]
    assert result["layouts"]["1-sparks"]["entries"] == result["layouts"]["2"]["entries"]


def test_flash_prices_a_layout_from_the_costlier_quant(tmp_path):
    # One layout traced twice, as an EXL3 and an FP8 checkpoint would be. The
    # entries are the same request shapes either way; the price is the costlier
    # quant's selected-entry sum, taken as that trace's own (bytes, executables)
    # pair, never a mix of the two.
    cheap = flash_trace([flash_graph(4, "false", "false", 0, 64, 0, 100, segment=0),
                         flash_graph(4, "false", "false", 0, 64, 0, 90, segment=1)], charged=16 << 20)
    dear = flash_trace([flash_graph(4, "false", "false", 0, 64, 0, 100, segment=0,
                                    capture_bytes=3 << 20),
                        flash_graph(4, "false", "false", 0, 64, 0, 90, segment=1,
                                    capture_bytes=3 << 20)], charged=20 << 20)
    dear["sites"]["glm5_flash.decode"]["largest_bytes"] = 5 << 20
    paths = []
    for name, data in (("exl3.json", cheap), ("fp8.json", dear)):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        paths.append(path)
    groups = MODULE.layout_groups([f"1-sparks={path}" for path in paths], "glm5_flash")
    layout = MODULE.generate_layouts(groups, "glm5_flash")["layouts"]["1-sparks"]
    assert (layout["measured_bytes"], layout["measured_executables"]) == ([6 << 20], [2])
    assert layout["reserve_extra_bytes"] == 2 << 20
    assert [(entry["rows"], entry["spec"]) for entry in layout["entries"]] == [(4, False)]


def test_flash_accepts_a_subset_of_traced_shapes_and_no_unknown_keys(tmp_path):
    path = flash_trace_path(tmp_path, "trace.json", 16 << 20)
    # An untraced shape falls back to the engine's bootstrap set, so a partial
    # document is valid; an unknown key is still refused.
    groups = MODULE.layout_groups([f"2-sparks={path}"], "glm5_flash")
    assert set(MODULE.generate_layouts(groups, "glm5_flash")["layouts"]) == {"2-sparks"}
    with pytest.raises(ValueError, match="unknown GLM Flash layout"):
        MODULE.layout_groups([f"min={path}"], "glm5_flash")
    with pytest.raises(ValueError, match="unknown GLM Flash layout"):
        MODULE.layout_groups([f"1-sparks={path}", f"mid={path}"], "glm5_flash")


def test_single_layout_families_keep_the_flat_document(tmp_path):
    data = trace()
    path = tmp_path / "glm5.json"
    path.write_text(json.dumps(data))
    result = MODULE.generate_layouts(MODULE.layout_groups([str(path)], "glm5"), "glm5")
    assert "layouts" not in result
    assert result["family"] == "glm5"

def modern_trace():
    data = trace()
    data["schema_version"] = 2
    data["cache"]["reserve"] = 16 << 20
    for graph in data["graphs"]:
        graph.update(hits=graph["count"], cycles={"verify": graph["count"]},
                     evictions=0, recaptures=0, ghost_hits=0, first_seen_ms=0,
                     hit_histogram={"rolling": True, "interval_ms": 10_000,
                                    "start_ms": 0, "counts": [5, 0, 0, 0, 0]},
                     permanent_bytes=1000, capture_bytes=0, nodes=10,
                     executables=1, basis={"kind": "nodes", "node_rate": 10, "graph_rate": 900})
    return data


def test_sustained_heat_pins_nonminimum_geometry_and_reports_cost():
    data = modern_trace()
    data["graphs"][1]["hit_histogram"]["counts"] = [1, 1, 1, 1, 0]
    report = []
    result = MODULE.generate([data, data], "glm5", pin_report=report)
    assert [entry["rows"] for entry in result["entries"] if entry["pin"]] == [16]
    assert len(report) == 1
    assert report[0]["permanent_bytes"] is None
    assert report[0]["observed_permanent_lower_bound_bytes"] == 1000
    assert report[0]["capture_bytes"] == 0
    assert report[0]["reserve_fraction"] is None
    assert report[0]["observed_reserve_fraction_lower_bound"] == 1000 / (16 << 20)
    assert report[0]["reasons"] == ["sustained_heat"]


def test_repeated_recapture_survives_frequency_coverage():
    data = modern_trace()
    data["graphs"][2].update(evictions=3, recaptures=2, ghost_hits=4)
    result = MODULE.generate([data], "glm5", coverage=0.99)
    assert {"rows": 64, "table_width": 128, "pin": True} in result["entries"]
    assert not result["entries"][0]["pin"], "early-only frequency is not sustained heat"


def test_partial_and_preidentity_intervals_do_not_prove_sustained_heat():
    graph = modern_trace()["graphs"][0]
    graph["hit_histogram"]["counts"] = [1, 1, 1]
    assert not MODULE.sustained(graph)
    graph["hit_histogram"]["counts"] = [0, 0, 1, 1, 1, 0]
    graph["first_seen_ms"] = 20_000
    assert MODULE.sustained(graph)


def test_rolling_window_not_lifetime_heat():
    graph = modern_trace()["graphs"][0]
    graph["hit_histogram"].update(start_ms=600_000, counts=[1] * 60)
    assert MODULE.sustained(graph)
    graph["hit_histogram"]["counts"] = [0] * 60
    assert not MODULE.sustained(graph)


def test_latest_rolling_snapshot_overrides_old_heat_even_in_reverse_order():
    import copy
    hot = modern_trace()
    hot["graphs"][0]["hit_histogram"]["counts"] = [1] * 5
    cold = copy.deepcopy(hot)
    for graph in cold["graphs"]:
        graph["hit_histogram"].update(start_ms=600_000, counts=[0] * 60)
    for traces in ([hot, cold], [cold, hot]):
        assert not any(entry["pin"] for entry in MODULE.generate(traces, "glm5")["entries"])


def test_pin_cost_includes_startup_only_siblings_of_same_geometry():
    import copy
    data = modern_trace()
    graph = data["graphs"][0]
    graph["hit_histogram"]["counts"] = [1] * 5
    sibling = copy.deepcopy(graph)
    sibling.update(bank=2, cycles={"other": 1}, count=1)
    data["graphs"].append(sibling)
    report = []
    MODULE.generate([data, data], "glm5", pin_report=report)
    assert report[0]["observed_permanent_lower_bound_bytes"] == 2000
    assert report[0]["executables"] == 2


def test_cost_keeps_distinct_key_ids_with_identical_display_shapes():
    import copy
    data = modern_trace()
    graph = data["graphs"][0]
    graph.update(key_id=1)
    graph["hit_histogram"]["counts"] = [1] * 5
    sibling = copy.deepcopy(graph)
    sibling["key_id"] = 2
    data["graphs"].append(sibling)
    report = []
    MODULE.generate([data], "glm5", pin_report=report)
    assert report[0]["observed_permanent_lower_bound_bytes"] == 2000
    assert report[0]["executables"] == 2


def test_new_snapshot_forgets_omitted_identity_heat_and_cost():
    import copy
    old = modern_trace()
    old["cache"]["sampled_at_ms"] = 40_000
    old["graphs"][0]["hit_histogram"]["counts"] = [1, 1, 1, 1, 0]
    new = copy.deepcopy(old)
    new["cache"]["sampled_at_ms"] = 700_000
    new["graphs"] = new["graphs"][1:]
    for graph in new["graphs"]:
        graph["hit_histogram"].update(start_ms=110_000, counts=[0] * 60)
    for inputs in ([old, new], [new, old]):
        report = []
        result = MODULE.generate(inputs, "glm5", pin_report=report)
        assert not any(entry["pin"] for entry in result["entries"])
        assert report == []


def test_partial_geometry_inventory_is_only_a_cost_lower_bound():
    data = modern_trace()
    data["graphs"] = data["graphs"][:1]
    data["graphs"][0]["hit_histogram"]["counts"] = [1] * 5
    report = []
    MODULE.generate([data], "glm5", pin_report=report)
    assert report[0]["observed_permanent_lower_bound_bytes"] == 1000
    assert report[0]["permanent_bytes"] is None
    assert report[0]["reserve_fraction"] is None
    assert report[0]["cost_basis"] == "observed_inventory_lower_bound_unknown_total"


def test_latest_snapshot_row_wins_within_the_same_histogram_interval():
    import copy
    old = modern_trace()
    old["cache"]["sampled_at_ms"] = 40_001
    old["graphs"][0]["hit_histogram"]["counts"] = [1] * 5
    new = copy.deepcopy(old)
    new["cache"]["sampled_at_ms"] = 40_002
    new["graphs"][0]["hit_histogram"]["counts"] = [0] * 5
    for inputs in ([old, new], [new, old]):
        assert not any(entry["pin"] for entry in MODULE.generate(inputs, "glm5")["entries"])










def context_trace(device=0):
    """A two-rank trace: the context placement records each rank's half table."""
    data = trace(device=device)
    for graph in data["graphs"]:
        rows, width, _ = map(int, MODULE.GEOMETRY.search(graph["shape"]).groups())
        graph["shape"] = graph["shape"].replace(
            f"table_width: {width}, table_stride: {width}",
            f"table_width: {width // 2}, table_stride: {width // 2}")
    return data


def test_mistral_large4_selects_its_site_prefix_and_layout():
    one = MODULE.generate([trace()], "mistral_large4", coverage=0.99, layout=1)
    assert one["family"] == "mistral_large4" and one["layout"] == 1
    assert [(entry["rows"], entry["table_width"]) for entry in one["entries"]] == [(1, 16), (16, 32)]
    # Checked-in entries hold the width a step is built with, not the rank's half.
    two = MODULE.generate([context_trace()], "mistral_large4", coverage=0.99, layout=2)
    assert two["layout"] == 2
    assert [(entry["rows"], entry["table_width"]) for entry in two["entries"]] == [(1, 16), (16, 32)]
    assert two["reserve_base_bytes"] == one["reserve_base_bytes"]


def test_layout_is_required_for_mistral_large4_and_reserved_for_it():
    with pytest.raises(ValueError):
        MODULE.generate([trace()], "mistral_large4")
    with pytest.raises(ValueError):
        MODULE.generate([trace()], "glm5", layout=1)


def test_odd_half_table_width_is_refused():
    data = context_trace()
    data["graphs"][0]["shape"] = data["graphs"][0]["shape"].replace(
        "table_width: 8, table_stride: 8", "table_width: 9, table_stride: 9")
    with pytest.raises(ValueError):
        MODULE.generate([data], "mistral_large4", layout=2)


@pytest.mark.parametrize("family", ["glm5", "mistral_large4"])
def test_generator_prefix_matches_shared_engine_graph_site(family):
    import re
    engine = SCRIPT.parents[2] / "rust/crates/cuteafd-daemon/src/families/glm5/engine.rs"
    sites = re.findall(r'name:\s*"([^"]+)"', engine.read_text())
    assert "glm5.segment" in sites
    assert all(site.startswith(MODULE.SITE_PREFIX[family]) for site in sites)


def test_v41_uses_its_layout_schema_not_a_glm_prefix_consumer():
    assert "deepseek_v41" not in MODULE.SITE_PREFIX
    with pytest.raises(ValueError, match="deepseek_v41 requires --layout 1 or 2"):
        MODULE.generate([trace()], "deepseek_v41")


def test_flash_is_not_advertised_as_a_glm_prefix_consumer():
    assert "glm5_flash" not in MODULE.SITE_PREFIX


@pytest.mark.parametrize("family", ["glm5", "mistral_large4"])
def test_every_consumer_uses_fixed_allocation_tail_not_largest_graph(family):
    data = trace()
    for graph in data["graphs"]:
        graph["site"] = MODULE.SITE_PREFIX[family] + "segment"
    data["sites"] = {MODULE.SITE_PREFIX[family] + "segment": {"largest_bytes": 80 << 20}}
    result = MODULE.generate([data], family, **({"layout": 1} if family == "mistral_large4" else {}))
    assert result["reserve_base_bytes"] == 8 << 20
    assert result["reserve_extra_bytes"] == 2 << 20


@pytest.mark.parametrize("filename", ["glm_prewarm.json", "ml4_prewarm_1.json", "ml4_prewarm_2.json"])
def test_regeneration_reproduces_checked_in_base_and_tail(filename):
    import json
    root = SCRIPT.parents[2] / "rust/crates/cuteafd-loader/src/serving_capacity"
    checked_in = json.loads((root / filename).read_text())
    data = trace(charged=checked_in["reserve_base_bytes"])
    # The measured arena, not a largest capture or the old reservation, is the base.
    data["sites"]["glm5.segment"]["largest_bytes"] = 80 << 20
    result = MODULE.generate([data], checked_in["family"],
                             **({"layout": checked_in["layout"]} if checked_in.get("layout") else {}))
    for key in ("reserve_base_bytes", "reserve_extra_bytes"):
        assert result[key] == checked_in[key]


def qwen_identity(segment, rows=64, spec=False, long=False, width=64, page_stride=64, pool_stride=128):
    return (f"GraphKey {{ segment: {segment}, rows: {rows}, spec: {str(spec).lower()}, "
            f"long: {str(long).lower()}, pool_width: {width}, page_stride: {page_stride}, "
            f"pool_stride: {pool_stride} }}")


def qwen_trace(device=0, charged=8 << 20, capture=(1 << 20, 2 << 20, 4 << 20)):
    def graph(segment, rows, count, share, hits=None):
        return {"device": device, "site": "qwen4.segment", "bank": 0,
                "key_id": segment * 1000 + rows,
                "shape": qwen_identity(segment, rows=rows),
                "count": count, "cycles": {"decode": count},
                "hits": count if hits is None else hits,
                "hit_histogram": {"rolling": True, "interval_ms": 10_000,
                                  "start_ms": 0, "counts": [5, 0, 0, 0, 0]},
                "evictions": 0, "recaptures": 0,
                "capture_bytes": share, "permanent_bytes": share * 2,
                "nodes": 10, "executables": 1, "basis": {"kind": "bytes"}}
    return {"schema_version": 2,
            "cache": {"device": device, "charged": charged, "sampled_at_ms": 10_000},
            "sites": {"qwen4.segment": {"largest_bytes": 3 << 20}},
            "graphs": [graph(0, 64, 100, capture[0]), graph(1, 64, 100, capture[1]),
                       graph(1, 128, 1, capture[2])]}


def test_qwen_shape_list_is_keyed_by_resolved_shape_and_device():
    result = MODULE.generate_qwen([qwen_trace(0, 8 << 20), qwen_trace(1, 12 << 20)],
                                  "dual_0spark", coverage=0.99)
    assert result["shape"] == "dual_0spark"
    # One 2 MiB chunk each: the selected (rows=64) identities' measured shares
    # 1 MiB + 2 MiB on rank 0, 1 MiB + 2 MiB on rank 1, rounded once. The
    # trace's charged high-water is never the base.
    assert result["reserve_base_bytes"] == [4 << 20, 4 << 20]
    assert result["reserve_extra_bytes"] == [2 << 20, 2 << 20]
    # Layer segments share one capture shape: one entry, not one per segment.
    assert result["trace"]["entries"] == [
        {"rows": 64, "spec": False, "long": False, "pool_width": 64,
         "page_stride": 64, "pool_stride": 128, "pin": False}]
    assert "dual_0spark" in result["trace"]["provenance"]


def test_qwen_base_ignores_charged_and_counts_startup_only_siblings():
    quiet = qwen_trace(0, 64 << 20, capture=(1 << 20, 2 << 20, 4 << 20))
    # A startup-only sibling is captured but never replayed; it still costs its
    # share in the arena. Moving the charged counter must not move the base.
    quiet["graphs"][1]["hits"] = 0
    result = MODULE.generate_qwen([quiet], "single_0spark", coverage=0.99)
    assert result["reserve_base_bytes"] == [4 << 20]
    loud = copy.deepcopy(quiet)
    loud["cache"]["charged"] = 5 << 30
    assert MODULE.generate_qwen([loud], "single_0spark", coverage=0.99)["reserve_base_bytes"] == [4 << 20]


def test_qwen_zero_hit_bootstrap_shapes_are_never_selected():
    # The provisional startup set captures many shapes the traffic never
    # replays. A startup capture is a miss and an event, never demand: ranking
    # by replay hits keeps every zero-hit shape out of the list (#1652), even
    # though their captured bytes are real.
    data = qwen_trace()
    hot = data["graphs"][0]
    hot["hits"] = 100
    cold = []
    for segment in range(1, 12):
        graph = dict(data["graphs"][1])
        graph.update(key_id=segment, hits=0, count=1, misses=1, cycles={"decode": 1},
                     capture_bytes=1 << 20, shape=qwen_identity(segment, rows=4))
        cold.append(graph)
    data["graphs"] = [hot] + cold
    summary = {}
    result = MODULE.generate_qwen([data], "single_0spark", summary=summary)
    assert result["trace"]["entries"] == [
        {"rows": 64, "spec": False, "long": False, "pool_width": 64,
         "page_stride": 64, "pool_stride": 128, "pin": False}]
    assert result["reserve_base_bytes"] == [2 << 20]
    # Captured once, never replayed: never selected, and reported rather than
    # hidden (11 cold identities, 1 MiB each).
    assert summary == {"zero_replay_identities": len(cold), "zero_replay_bytes": len(cold) << 20,
                       "zero_share_identities": 0, "zero_share_bytes": 0}
    assert f"{len(cold)} zero-replay identities" in result["trace"]["provenance"]


def test_qwen_zero_share_identity_is_charged_the_device_mean():
    # A capture boundary with no new physical growth is allocator reuse, not a
    # free graph. The shared cache keeps such a record's estimate; the base
    # must not treat the zero as a measured zero, or a selected shape's share
    # vanishes from the reserve.
    data = qwen_trace()
    data["graphs"][0]["capture_bytes"] = 0
    data["graphs"][2]["capture_bytes"] = 1 << 20
    summary = {}
    result = MODULE.generate_qwen([data], "single_0spark", coverage=0.99, summary=summary)
    # Device mean over three captures: (0 + 2 + 1) MiB / 3 = 1 MiB; the zero
    # selected identity is charged it, so 1 + 2 = 3 MiB -> one 2 MiB chunk.
    assert result["reserve_base_bytes"] == [4 << 20]
    assert summary["zero_share_identities"] == 1
    assert summary["zero_share_bytes"] == 1 << 20
    assert "1 zero-share selected identities" in result["trace"]["provenance"]
    # A measured zero outside the selected set is not filled.
    data = qwen_trace()
    data["graphs"][2]["capture_bytes"] = 0
    assert MODULE.generate_qwen([data], "single_0spark", coverage=0.99)["reserve_base_bytes"] == [4 << 20]


def test_qwen_selected_identity_without_a_measured_share_fails():
    data = qwen_trace()
    data["graphs"][1]["capture_bytes"] = None
    with pytest.raises(ValueError, match="without a measured capture share"):
        MODULE.generate_qwen([data], "single_0spark", coverage=0.99)
    # A shape outside the selected set may stay unmeasured.
    data = qwen_trace()
    data["graphs"][2]["capture_bytes"] = None
    assert MODULE.generate_qwen([data], "single_0spark", coverage=0.99)["trace"]["entries"]


def test_qwen_regeneration_reproduces_a_checked_in_generation_exactly():
    # A committed fixture trace reproduces byte-identical base and extra: the
    # checked-in lists must never be hand-edited.
    fixture = Path(__file__).resolve().parent / "fixtures" / "qwen-prewarm-trace"
    traces = [json.loads(path.read_text()) for path in sorted(fixture.glob("trace.device-*.json"))]
    generated = MODULE.generate_qwen(traces, "single_0spark", coverage=0.99)
    expected = json.loads((fixture / "expected-single_0spark.json").read_text())
    assert generated == expected


def qwen_list():
    def shape(name, owners):
        return {"shape": name, "reserve_base_bytes": [5 << 30] * owners,
                "reserve_extra_bytes": [2 << 20] * owners, "trace": None}
    return {"schema_version": 1, "family": "qwen4", "provenance": "bootstrap",
            "bootstrap": {"entries": []},
            "shapes": [shape("single_0spark", 1), shape("single_sparks", 1),
                       shape("dual_0spark", 2), shape("dual_sparks", 2)]}


def fixture_traces():
    fixture = Path(__file__).resolve().parent / "fixtures" / "qwen-prewarm-trace"
    return [json.loads(path.read_text()) for path in sorted(fixture.glob("trace.device-*.json"))]


def test_qwen_merge_replaces_only_the_traced_shape():
    generated = MODULE.generate_qwen(fixture_traces(), "single_0spark", coverage=0.99)
    merged = MODULE.merge_qwen(qwen_list(), generated)
    installed = {prewarm["shape"]: prewarm for prewarm in merged["shapes"]}
    assert installed["single_0spark"] == generated
    assert installed["dual_0spark"]["trace"] is None
    assert installed["dual_0spark"]["reserve_base_bytes"] == [5 << 30, 5 << 30]
    assert "single_0spark" in merged["provenance"]
    assert "dual_sparks" in merged["provenance"], "untraced shapes are named"
    assert MODULE.merge_qwen(merged, generated) == merged, "merging is idempotent"


def test_qwen_merge_installs_the_generated_base_and_extra_exactly():
    fixture = Path(__file__).resolve().parent / "fixtures" / "qwen-prewarm-trace"
    generated = MODULE.generate_qwen(fixture_traces(), "single_0spark", coverage=0.99)
    merged = MODULE.merge_qwen(qwen_list(), generated)
    installed = next(prewarm for prewarm in merged["shapes"] if prewarm["shape"] == "single_0spark")
    assert installed == json.loads((fixture / "expected-single_0spark.json").read_text())


def test_qwen_merge_rejects_a_list_without_the_shape():
    generated = MODULE.generate_qwen(fixture_traces(), "single_0spark", coverage=0.99)
    document = qwen_list()
    document["shapes"] = [prewarm for prewarm in document["shapes"]
                          if prewarm["shape"] != "single_0spark"]
    with pytest.raises(ValueError, match="no single_0spark"):
        MODULE.merge_qwen(document, generated)


def test_qwen_single_shape_needs_exactly_one_device_trace():
    assert len(MODULE.generate_qwen([qwen_trace()], "single_0spark")["reserve_base_bytes"]) == 1
    with pytest.raises(ValueError):
        MODULE.generate_qwen([qwen_trace()], "dual_sparks")


def test_qwen_unparseable_identity_fails():
    data = qwen_trace()
    data["graphs"][0]["shape"] = "bad"
    with pytest.raises(ValueError):
        MODULE.generate_qwen([data], "single_0spark")


def test_qwen_family_requires_shape_argument():
    with pytest.raises(ValueError):
        MODULE.generate([qwen_trace()], "qwen4")


def v4_shape(rows, width, attention="c4"):
    return (f'GraphKey {{ layer: 3, rows: {rows}, chunked: false, attention: "{attention}", '
            f'table_width: {width}, table_stride: {width}, previous: 0, '
            'tp2_broadcast: false, exchange_f32: false }')


def v4_list():
    shapes = {"rtx1-spark": 1, "rtx1-local": 1, "rtx2-spark": 2, "rtx2-local": 2}
    return {"schema_version": 1, "family": "deepseek_v4", "layouts": {
        key: {"traced": False, "reserve_base_bytes": [0] * ranks,
              "reserve_extra_bytes": [0] * ranks,
              "bootstrap_bytes": [1 << 20] * ranks, "entries": []}
        for key, ranks in shapes.items()}}


def v4_trace(device, charged, capture=2 << 20):
    """One rank's snapshot: a regular verify geometry, a startup row and a rare
    verify geometry used once inside one sampling interval."""
    def graph(rows, width, count, attention="c4", heat="regular"):
        counts = {"rare": [count, 0, 0, 0, 0, 0],
                  "regular": [1, 0, 0, count - 2, 0, 0],
                  "sustained": [count // 6] * 6}[heat]
        hits = 0 if heat == "rare" else count - 1
        return {"device": device, "site": "deepseek_v4.segment", "bank": 1,
                "shape": v4_shape(rows, width, attention), "count": count, "key_id": (rows, width),
                "hits": hits, "cycles": {"decode": count}, "capture_bytes": capture,
                "hit_histogram": {"interval_ms": 10_000, "start_ms": 0, "counts": counts}}
    return {"schema_version": 2, "cache": {"device": device, "charged": charged},
            "sites": {"deepseek_v4.segment": {"largest_bytes": 3 << 20}},
            "graphs": [graph(16, 512, 100), graph(4, 0, 90, "local-experts"),
                       graph(1, 16, 1, heat="rare")]}


def test_v4_geometry_allows_the_width_zero_startup_set():
    assert MODULE.geometry("deepseek_v4", v4_shape(4, 0, "local-experts")) == (4, 0)
    assert MODULE.geometry("deepseek_v4", v4_shape(16, 512)) == (16, 512)
    with pytest.raises(ValueError):
        MODULE.geometry("deepseek_v4", "GraphKey { rows: 1 }")


def test_v4_eager_tails_never_become_prewarm_entries():
    assert MODULE.prewarmable("deepseek_v4", v4_shape(4, 0, "local-experts"))
    # A 2-RTX shape's startup set is the TP2 router pair at width 0.
    assert MODULE.prewarmable("deepseek_v4", v4_shape(4, 0, "tp2-broadcast"))
    assert MODULE.prewarmable("deepseek_v4", v4_shape(4, 0, "tp2-replicated"))
    assert MODULE.prewarmable("deepseek_v4", v4_shape(16, 512))
    assert not MODULE.prewarmable("deepseek_v4", v4_shape(5, 0, "tail"))
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].append({"device": 0, "site": "deepseek_v4.segment", "bank": 1,
                            "shape": v4_shape(5, 0, "tail"), "count": 4, "key_id": (5, 0),
                            "cycles": {"decode": 4}, "capture_bytes": 2 << 20})
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list())
    assert (5, 0) not in {(row["rows"], row["table_width"])
                          for row in entry["layouts"]["rtx1-local"]["entries"]}


def test_v4_replaces_one_shape_with_per_rank_arrays():
    result = MODULE.generate([v4_trace(0, 5 << 20, 6 << 20), v4_trace(1, 7 << 20, 4 << 20)],
                             "deepseek_v4", layout="rtx2-spark", existing=v4_list())
    assert result["layouts"]["rtx1-spark"] == v4_list()["layouts"]["rtx1-spark"]
    assert result["layouts"]["rtx2-local"] == v4_list()["layouts"]["rtx2-local"]
    entry = result["layouts"]["rtx2-spark"]
    assert entry["traced"] is True
    # The one-use geometry is rare and stays lazy; the base is the warmed set's
    # own captures: two geometries of 6 MiB on device 0, 4 MiB on 1.
    assert {(row["rows"], row["table_width"]) for row in entry["entries"]} == {(4, 0), (16, 512)}
    assert entry["reserve_base_bytes"] == [12 << 20, 8 << 20]
    # The arena tail only: capture headroom is a shared launcher policy.
    assert entry["reserve_extra_bytes"] == [2 << 20, 2 << 20]
    assert entry["bootstrap_bytes"] == [6 << 20, 4 << 20]


def test_v4_heat_split_pins_sustained_prewarms_regular_and_leaves_rare_lazy():
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].append({
        "device": 0, "site": "deepseek_v4.segment", "bank": 1,
        "shape": v4_shape(8, 64), "count": 60, "key_id": (8, 64), "hits": 59,
        "cycles": {"verify": 60}, "capture_bytes": 3 << 20,
        "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                          "counts": [10, 10, 10, 10, 10, 10]}})
    heat = []
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list(),
                            heat_report=heat)["layouts"]["rtx1-local"]
    entries = {(row["rows"], row["table_width"]): row["pin"] for row in entry["entries"]}
    assert entries[(8, 64)] is True, "sustained heat is pinned"
    assert entries[(16, 512)] is True, "the dominant verify geometry always pins"
    assert entries[(4, 0)] is False, "a recurring startup row is prewarmed, not pinned"
    assert (1, 16) not in entries, "a one-use geometry stays lazy"
    classes = {(row["device"], row["heat"]): (row["keys"], row["arena_bytes"], row["selected"])
               for row in heat}
    assert classes[(0, "startup")] == (1, 2 << 20, True)
    assert classes[(0, "sustained")] == (1, 3 << 20, True)
    assert classes[(0, "regular")] == (1, 2 << 20, True)
    assert classes[(0, "rare")] == (1, 2 << 20, False)
    # The rare geometry's bytes are not charged in the base: 2 + 2 + 3 MiB of
    # warmed captures, rounded up to whole 2 MiB chunks.
    assert entry["reserve_base_bytes"] == [8 << 20]


def test_v4_startup_rows_prewarm_and_only_sustained_heat_pins():
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].append({"device": 0, "site": "deepseek_v4.segment", "bank": 1,
                            "shape": v4_shape(64, 0, "local-experts"), "count": 90,
                            "key_id": (64, 0), "hits": 89, "cycles": {"decode": 90},
                            "capture_bytes": 2 << 20,
                            "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                                              "counts": [18, 18, 18, 18, 18, 0]}})
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list())
    entries = {(row["rows"], row["table_width"]): row["pin"]
               for row in entry["layouts"]["rtx1-local"]["entries"]}
    # A width-0 row the trace replayed is a startup row at every row count;
    # only the one whose own heat is sustained earns permanent residency.
    assert entries[(4, 0)] is False
    assert entries[(64, 0)] is True
    # The dominant verify geometry always pins; the other rare one stays lazy.
    assert entries[(16, 512)] is True
    assert (1, 16) not in entries


def test_v4_single_interval_startup_row_is_prewarmed_unpinned():
    # A width-0 row replayed inside one sampling interval is reachable, so it
    # is warmed; a pin otherwise needs sustained heat, and ARC must be able to
    # reclaim a row serving never revisits (TJ, 2026-11-09).
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].append({"device": 0, "site": "deepseek_v4.segment", "bank": 1,
                            "shape": v4_shape(32, 0, "local-experts"), "count": 90,
                            "key_id": (32, 0), "hits": 90, "cycles": {"decode": 90},
                            "capture_bytes": 2 << 20,
                            "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                                              "counts": [0, 0, 0, 0, 90, 0]}})
    heat = []
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list(),
                            heat_report=heat)["layouts"]["rtx1-local"]
    entries = {(row["rows"], row["table_width"]): row["pin"] for row in entry["entries"]}
    assert entries[(32, 0)] is False
    assert ("startup", True) in {(row["heat"], row["selected"]) for row in heat}
    # The rare one-use verify geometry stays lazy; the three warmed captures
    # are the base.
    assert entry["reserve_base_bytes"] == [6 << 20]


def test_v4_c1_and_dominant_verify_pin_whatever_their_heat():
    # Plain C1 (rows 1, width 0) and the layout's highest-lifetime-hit verify
    # geometry cost every step, so they pin even when their heat is regular;
    # the other regular verify geometry stays prewarmed and evictable.
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].extend([
        {"device": 0, "site": "deepseek_v4.segment", "bank": 1,
         "shape": v4_shape(1, 0, "local-experts"), "count": 60, "key_id": (1, 0),
         "hits": 59, "cycles": {"decode": 60}, "capture_bytes": 2 << 20,
         "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                           "counts": [1, 0, 0, 57, 0, 0]}},
        {"device": 0, "site": "deepseek_v4.segment", "bank": 1,
         "shape": v4_shape(12, 64), "count": 40, "key_id": (12, 64),
         "hits": 39, "cycles": {"verify": 40}, "capture_bytes": 2 << 20,
         "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                           "counts": [1, 0, 0, 38, 0, 0]}},
    ])
    report = []
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list(),
                            pin_report=report)["layouts"]["rtx1-local"]
    entries = {(row["rows"], row["table_width"]): row["pin"] for row in entry["entries"]}
    assert entries[(1, 0)] is True, "plain C1 pins whatever its heat"
    # (16, 512) has 99 lifetime hits against (12, 64)'s 39.
    assert entries[(16, 512)] is True
    assert entries[(12, 64)] is False
    assert entries[(4, 0)] is False
    reasons = {(row["rows"], row["table_width"]): row["reasons"] for row in report}
    assert reasons[(1, 0)] == ["c1_decode"]
    assert reasons[(16, 512)] == ["dominant_verify"]
    assert (12, 64) not in reasons
    assert entry["reserve_base_bytes"] == [8 << 20]


def test_v4_entries_order_pins_first_then_lifetime_hits():
    # A shrunken reserve warms entries in list order, so the fan-out order puts
    # the pinned set first and then the geometries serving replays most; order
    # never changes the selected set, base or pin flags.
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].extend([
        {"device": 0, "site": "deepseek_v4.segment", "bank": 1,
         "shape": v4_shape(1, 0, "local-experts"), "count": 60, "key_id": (1, 0),
         "hits": 59, "cycles": {"decode": 60}, "capture_bytes": 2 << 20,
         "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                           "counts": [1, 0, 0, 57, 0, 0]}},
        {"device": 0, "site": "deepseek_v4.segment", "bank": 1,
         "shape": v4_shape(32, 0, "local-experts"), "count": 8, "key_id": (32, 0),
         "hits": 7, "cycles": {"decode": 8}, "capture_bytes": 2 << 20,
         "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                           "counts": [0, 0, 0, 7, 0, 0]}},
    ])
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list())["layouts"]["rtx1-local"]
    order = [(row["rows"], row["table_width"]) for row in entry["entries"]]
    # Pins: (16, 512) has 99 lifetime hits, plain C1 59. Unpinned rows follow
    # in hit order: (4, 0) 89, then (32, 0) 7; the one-use (1, 16) stays lazy.
    assert order == [(16, 512), (1, 0), (4, 0), (32, 0)]
    assert [row["pin"] for row in entry["entries"]] == [True, True, False, False]
    assert entry["reserve_base_bytes"] == [8 << 20]


def test_v4_startup_row_captured_once_is_not_warmed_or_pinned():
    # One miss and zero hits is a capture with no replay: warming it would
    # charge the startup arena for a graph the workload never reuses, and
    # pinning it would make that one-off permanent (TJ, 2026-11-09).
    trace = v4_trace(0, 5 << 20)
    trace["graphs"].append({"device": 0, "site": "deepseek_v4.segment", "bank": 1,
                            "shape": v4_shape(50, 0, "local-experts"), "count": 18,
                            "key_id": (50, 0), "hits": 0, "misses": 18,
                            "cycles": {"verify": 18}, "capture_bytes": 3 << 20,
                            "hit_histogram": {"interval_ms": 10_000, "start_ms": 0,
                                              "counts": [0, 0, 0, 0, 0, 0]}})
    heat = []
    entry = MODULE.generate([trace], "deepseek_v4", layout="rtx1-local", existing=v4_list(),
                            heat_report=heat)["layouts"]["rtx1-local"]
    entries = {(row["rows"], row["table_width"]): row["pin"] for row in entry["entries"]}
    assert (50, 0) not in entries
    # It is a rare leftover, not startup residency, and stays lazy.
    classes = {row["heat"]: row for row in heat}
    assert classes["rare"]["keys"] == 2  # the one-use verify geometry and this row
    assert classes["rare"]["selected"] is False
    assert entry["reserve_base_bytes"] == [4 << 20]


def test_v4_shape_needs_one_trace_per_rank():
    with pytest.raises(ValueError):
        MODULE.generate([v4_trace(0, 5 << 20)], "deepseek_v4", layout="rtx2-spark", existing=v4_list())
    single = MODULE.generate([v4_trace(0, 5 << 20)], "deepseek_v4", layout="rtx1-local", existing=v4_list())
    # The 5 MiB charge is mostly lazy captures; only the warmed set is the base.
    assert single["layouts"]["rtx1-local"]["reserve_base_bytes"] == [4 << 20]


def test_v4_unknown_layout_fails():
    with pytest.raises(ValueError):
        MODULE.generate([v4_trace(0, 5 << 20)], "deepseek_v4", layout="rtx4", existing=v4_list())


def v41_graph(rows=1, *, hits=10, cost=1234, bank=1, key_id=1, site="lane_row", cycles=None):
    return {"site": "deepseek_v41." + site, "shape": str((7, rows)), "hits": hits,
            "capture_bytes": cost, "bank": bank, "key_id": key_id,
            "cycles": {"verify": hits + 1} if cycles is None else cycles}


def v41_trace(device=0, graphs=None, sampled=1):
    return {"schema_version": 2, "cache": {"device": device, "charged": 999999999,
            "sampled_at_ms": sampled}, "graphs": [v41_graph()] if graphs is None else graphs}


def test_v41_replays_select_whole_geometry_but_not_capture_only_geometry():
    hot = v41_graph()
    sibling = v41_graph(hits=0, cost=4000, bank=2)
    cold = v41_graph(rows=16, hits=0, cost=999999, key_id=2)
    report = []
    result = MODULE.generate([v41_trace(graphs=[hot, hot, sibling, cold])], "deepseek_v41", layout=1, pin_report=report)
    rank = result["ranks"][0]
    assert rank["entries"] == [{"rows": 1, "pin": False}]
    assert rank["reserve_base_bytes"] == 2 << 20
    assert rank["reserve_extra_bytes"] == 2 << 20
    assert report[0]["capture_bytes"] == 5234
    assert report[0]["zero_replay_sibling_bytes"] == 4000
    assert report[0]["zero_replay_sibling_identities"] == 1


def test_v41_latest_inventory_and_exact_ids_win_independently_per_rank():
    old = v41_trace(graphs=[v41_graph(cost=99999999)], sampled=1)
    new = v41_trace(graphs=[v41_graph(cost=0), v41_graph(cost=2345, key_id=2)], sampled=2)
    second = v41_trace(device=1, graphs=[v41_graph(cost=5 << 20)])
    for inputs in [[old, new, second], [second, new, old]]:
        result = MODULE.generate(inputs, "deepseek_v41", layout=2)
        assert [rank["reserve_base_bytes"] for rank in result["ranks"]] == [2 << 20, 6 << 20]


@pytest.mark.parametrize("value", [None, True, False, -1, 1.5, "1"])
def test_v41_capture_shares_are_strict_nonnegative_integers(value):
    with pytest.raises(ValueError, match="capture_bytes"):
        MODULE.generate([v41_trace(graphs=[v41_graph(cost=value)])], "deepseek_v41", layout=1)


def test_v41_missing_share_and_missing_replay_count_fail():
    for field in ["capture_bytes", "hits"]:
        graph = v41_graph()
        del graph[field]
        with pytest.raises(ValueError, match=field):
            MODULE.generate([v41_trace(graphs=[graph])], "deepseek_v41", layout=1)


def test_v41_unreachable_full_chunk_is_not_a_pin():
    replay = v41_graph(128, hits=1, cycles={"prefill": 2})
    chunk = v41_graph(2752, hits=0, key_id=2, cycles={"other": 1})
    rank = MODULE.generate([v41_trace(graphs=[replay, chunk])], "deepseek_v41", layout=1, prefill_rows=2752)["ranks"][0]
    assert rank["prefill_entries"] == [{"rows": 128, "pin": True}]
    chunk["cycles"] = {"prefill": 1}
    rank = MODULE.generate([v41_trace(graphs=[replay, chunk])], "deepseek_v41", layout=1, prefill_rows=2752)["ranks"][0]
    assert rank["prefill_entries"] == [{"rows": 128, "pin": True}, {"rows": 2752, "pin": True}]


def test_v41_unknown_layout_or_missing_rank_has_no_fallback():
    for layout in [None, 0, 2, 3]:
        with pytest.raises(ValueError):
            MODULE.generate([v41_trace()], "deepseek_v41", layout=layout)


@pytest.mark.parametrize("width", [5, 7])
def test_v41_flattened_draft_heads_keep_their_owner_and_cost(width):
    graphs = []
    for count in range(1, 17):
        draft = v41_graph(hits=1, key_id=count, site="dspark_chain")
        draft["shape"] = str((count, width))
        head = v41_graph(hits=1000, key_id=count, site="target_head")
        head["shape"] = str(count * width)
        graphs.extend([draft, head])
    rank = MODULE.generate([v41_trace(graphs=graphs)], "deepseek_v41", layout=1, coverage=0.01)["ranks"][0]
    owners = {entry["requests"] * entry["width"] for entry in rank["draft_entries"]}
    assert {count * width for count in range(1, 17) if count * width > 64} <= owners
    head = v41_graph(site="target_head")
    head["shape"] = "112"
    with pytest.raises(ValueError, match="lack an observed"):
        MODULE.generate([v41_trace(graphs=[head])], "deepseek_v41", layout=1)


def test_v41_stale_histogram_cannot_pin_and_current_histogram_can():
    graph = v41_graph()
    graph["first_seen_ms"] = 0
    graph["hit_histogram"] = {"start_ms": 0, "interval_ms": 10_000, "counts": [1] * 5}
    for sampled, pinned in [(40_000, True), (500_000, False)]:
        rank = MODULE.generate([v41_trace(graphs=[graph], sampled=sampled)],
                               "deepseek_v41", layout=1)["ranks"][0]
        assert rank["entries"][0]["pin"] is pinned
