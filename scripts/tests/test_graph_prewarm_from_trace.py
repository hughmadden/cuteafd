"""Trace-derived startup lists preserve hot shapes and per-device ownership."""
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


@pytest.mark.parametrize("family", ["deepseek_v41"])
def test_native_family_schemas_are_not_advertised_as_glm_consumers(family):
    assert family not in MODULE.SITE_PREFIX
    with pytest.raises(ValueError, match="no checked-in shape-list consumer"):
        MODULE.generate([trace()], family)


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
