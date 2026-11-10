"""One SM89 coordinator export serves every supported SM89 card (CPU-only).

v41_fp8.cmake lists the FP8 shapes that get one exact plan per card; the
exporter derives the same set from SparkInfer's SM89 plan rule and refuses a
stale list. Both run here without CUDA: the pure plan functions are lifted out
of their modules.
"""
import ast
import os
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "python/tools/aot/export_b12x_v41_fp8_aot.py"
CMAKE = ROOT / "native/cmake/families/deepseek_v41/v41_fp8.cmake"
PLAN = ROOT / "third_party/sparkinfer/b12x/_lib/sm89_fp8.py"
sys.path.insert(0, str(ROOT / "python/tools/lib"))
import sm89_cards  # noqa: E402
from v41_capacities import PROJECTION_CAPACITIES  # noqa: E402


def lift(path, names, namespace):
    """Execute only the named top-level functions and assignments of ``path``."""
    body = [node for node in ast.parse(path.read_text()).body
            if (isinstance(node, ast.FunctionDef) and node.name in names)
            or (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) in names)]
    found = {node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id for node in body}
    assert found == set(names), set(names) - found
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.delenv("CUTEAFD_EXPORT_NARROW_AOT", raising=False)
    monkeypatch.delenv("CUTEAFD_V41_FP8_CARD_PLANS", raising=False)
    monkeypatch.delenv("CUTEAFD_V41_FP8_CARD_SHAPES", raising=False)
    plan = lift(PLAN, ("GEMV_MAX_ROWS", "_wave_efficiency", "_dense_part", "_gemv_part",
                       "_verify_dense_part", "_prefill_part", "sm89_block32_plan"), {})
    namespace = lift(EXPORTER, ("PROJECTIONS", "SM120_CARD_SMS", "CARD_PLAN_SHAPES", "expected_rows",
                                "sm89_card_shapes", "check_declared_card_plans"),
                     {"os": os, "SM89_CARD_SMS": sm89_cards.CARD_SMS})
    namespace["plan"] = plan["sm89_block32_plan"]
    return namespace


def cmake_card_plans():
    text = CMAKE.read_text()
    common = re.search(r"set\(CUTEAFD_V41_FP8_CARD_SHAPES ([^)]*)\)", text)[1].split()
    sm89 = re.search(r"if\(CUTEAFD_V41_FP8_SM89\)\s*set\(CUTEAFD_V41_FP8_CARD_PLANS ([^)]*)\)\s*"
                     r"list\(APPEND CUTEAFD_V41_FP8_CARD_SHAPES ([^)]*)\)\s*else\(\)\s*"
                     r"set\(CUTEAFD_V41_FP8_CARD_PLANS ([^)]*)\)", text)
    shapes = lambda items: {(name, int(rows)) for name, rows in (item.split(":") for item in items)}
    return {
        "sm89": (tuple(map(int, sm89[1].split())), shapes(common + sm89[2].split())),
        "sm120": (tuple(map(int, sm89[3].split())), shapes(common)),
    }


def test_supported_sm89_cards_are_the_4090_family():
    # RTX 4090 D (114 SMs) and RTX 4090 (128 SMs); capacity for the larger.
    assert sm89_cards.CARD_SMS == (114, 128)
    assert sm89_cards.WORKSPACE_SMS == 128


def test_cmake_lists_exactly_the_card_plans_the_sm89_rule_needs(exporter):
    listed = cmake_card_plans()
    derived = exporter["sm89_card_shapes"](exporter["PROJECTIONS"], PROJECTION_CAPACITIES, exporter["plan"])
    assert listed["sm89"] == (sm89_cards.CARD_SMS, derived)
    # The q_a/kv decode shapes stay card-planned on both architectures, so an
    # unsupported card fails closed at every row count.
    assert exporter["CARD_PLAN_SHAPES"] <= derived
    assert listed["sm120"] == (exporter["SM120_CARD_SMS"], set(exporter["CARD_PLAN_SHAPES"]))
    # Every other card-planned shape is one whose SM89 kernels differ by SM count.
    for name, n, k in exporter["PROJECTIONS"]:
        groups = 8 if name == "o_a" else 1
        for capacity in PROJECTION_CAPACITIES:
            plans = {repr(exporter["plan"](size_m=capacity, size_n=n // groups, size_k=k // groups,
                                           num_groups=groups, sm_count=sms,
                                           fixed_rows=exporter["expected_rows"](
                                               name, capacity, k, n, groups, sms) < capacity))
                     for sms in sm89_cards.CARD_SMS}
            card_planned = (name, capacity) in derived
            assert card_planned == ((len(plans) > 1) or (name, capacity) in exporter["CARD_PLAN_SHAPES"])


def test_exporter_refuses_a_stale_cmake_list(exporter, monkeypatch):
    check = exporter["check_declared_card_plans"]
    projections = exporter["PROJECTIONS"]
    derived = exporter["sm89_card_shapes"](projections, PROJECTION_CAPACITIES, exporter["plan"])
    check(sm89_cards.CARD_SMS, derived, projections, PROJECTION_CAPACITIES)  # nothing declared
    listed = ",".join(f"{name}:{rows}" for name, rows in sorted(derived))
    monkeypatch.setenv("CUTEAFD_V41_FP8_CARD_PLANS", "114,128")
    monkeypatch.setenv("CUTEAFD_V41_FP8_CARD_SHAPES", listed)
    check(sm89_cards.CARD_SMS, derived, projections, PROJECTION_CAPACITIES)
    # A subset export compares only the shapes it exports.
    subset = [p for p in projections if p[0] in ("q_a", "engram")]
    rows = (1, 64)
    check(sm89_cards.CARD_SMS, exporter["sm89_card_shapes"](subset, rows, exporter["plan"]), subset, rows)
    stale = ",".join(item for item in listed.split(",") if item != "engram:64")
    monkeypatch.setenv("CUTEAFD_V41_FP8_CARD_SHAPES", stale)
    with pytest.raises(ValueError, match="v41_fp8.cmake lists card plans"):
        check(sm89_cards.CARD_SMS, derived, projections, PROJECTION_CAPACITIES)
    monkeypatch.setenv("CUTEAFD_V41_FP8_CARD_SHAPES", listed)
    monkeypatch.setenv("CUTEAFD_V41_FP8_CARD_PLANS", "114")
    with pytest.raises(ValueError, match="v41_fp8.cmake lists card plans"):
        check(sm89_cards.CARD_SMS, derived, projections, PROJECTION_CAPACITIES)


def test_expert_exports_size_sm89_capacity_for_the_largest_card():
    slices = (ROOT / "python/tools/aot/export_b12x_slices_aot.py").read_text()
    experts = (ROOT / "python/tools/aot/export_b12x_v41_experts_aot.py").read_text()
    assert "workspace_sms = SM89_WORKSPACE_SMS if sm89 else 48 if props.minor == 1 else 188" in slices
    assert "sms = (SM89_WORKSPACE_SMS if manifest['capability'] == [8, 9]" in experts
    for source in (slices, experts, EXPORTER.read_text()):
        assert "physical_sms'] if" not in source and "multi_processor_count if sm89" not in source
