"""MiniMax launch metadata, opt-in programs and native expert scopes (CPU only)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))
import wip_aot_scopes as aot
from test_exlr8_aot_scope import build_key_lines


def release_config():
    cases = json.loads((ROOT / "rust/crates/cuteafd-loader/tests/fixtures/launch-families.json").read_text())
    return next(case["config"] for case in cases if case["name"] == "MiniMax M3 release routed extent")


def test_launch_extent_and_actual_routed_count():
    config = release_config()
    assert aot.checkpoint_family.describe(config) == ("minimax_m3", "minimax_m3_vl", 3, 59)
    assert aot.checkpoint_family.routed_layers(config) == 57
    assert aot.checkpoint_family.onboard("all", config) == "57"


@pytest.mark.parametrize("tp", [2, 3, 4, 6])
def test_native_minimax_scopes_use_explicit_tp(tp):
    family, scopes = aot.config_scopes(dict(FAMILY="minimax_m3", SPARK_COUNT="6", SPARK_TP=str(tp)))
    assert family == "minimax_m3"
    assert scopes["CUTEAFD_WIP_MINIMAX_AOT"] == "ON"
    assert scopes["CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES"] == "minimax:nvfp4"
    assert scopes["CUTEAFD_WIP_EXPERT_FAMILIES"] == "minimax:nvfp4"
    assert scopes["CUTEAFD_WIP_GENERIC_SPARK_COUNTS"] == f"minimax={tp}"


def test_minimax_switch_does_not_change_disabled_slot_keys():
    base = {"PATH": "/usr/bin:/bin", "CUTEAFD_WIP_GLM_AOT": "ON"}
    unset = build_key_lines(base)
    assert "MINIMAX" not in unset
    assert build_key_lines({**base, "CUTEAFD_WIP_MINIMAX_AOT": "OFF"}) == unset
    assert "CUTEAFD_WIP_MINIMAX_AOT=ON\n" in build_key_lines({**base, "CUTEAFD_WIP_MINIMAX_AOT": "ON"})
    build = (ROOT / "scripts/build/build-wip-artifacts.sh").read_text()
    block = build[build.index("    env | grep -E"):build.index("    printf", build.index("    env | grep -E"))]
    def keyed(extra):
        return subprocess.check_output(["bash", "-c", block], env={**base, **extra}, text=True)
    assert keyed({"CUTEAFD_WIP_MINIMAX_AOT": "OFF"}) == keyed({})
    assert keyed({"CUTEAFD_WIP_MINIMAX_AOT": "ON"}) != keyed({})


def test_minimax_switch_reaches_exporter_and_native_dispatch():
    assert '-e "CUTEAFD_WIP_MINIMAX_AOT=${CUTEAFD_WIP_MINIMAX_AOT:-OFF}"' in (ROOT / "wip.sh").read_text()
    for mode in ("wip", "release"):
        source = (ROOT / f"scripts/build/build-{mode}-artifacts.sh").read_text()
        assert ('-DCUTEAFD_ENABLE_MINIMAX_AOT="$( [[ "$role" == coordinator ]] && echo '
                f'"${{CUTEAFD_{mode.upper()}_MINIMAX_AOT:-OFF}}" || echo OFF)"') in source
    cmake = (ROOT / "native/cmake/shared/dsv4_programs.cmake").read_text()
    assert 'set(CUTEAFD_PROGRAM_GEOMETRY "minimax")' in cmake
    assert 'set(CUTEAFD_PROGRAM_GEOMETRY "${CUTEAFD_PROGRAM_GEOMETRY},minimax")' in cmake
    assert "export_b12x_minimax_aot.py" in cmake
    top = (ROOT / "native/CMakeLists.txt").read_text()
    assert top.count("CUTEAFD_ENABLE_MINIMAX_AOT OR") == 3


def test_model_context_and_comparison_jobs_include_minimax():
    import importlib.util
    path = ROOT / "scripts/build/compare-sm-exports.py"
    spec = importlib.util.spec_from_file_location("minimax_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jobs = {name: (exporter, args) for name, exporter, args in module.jobs("all")}
    assert {"minimax", "routed-minimax"} <= jobs.keys()
    assert jobs["minimax"] == ("export_b12x_dsv4_aot.py", ["--geometry", "minimax"])
    # The coordinator exporter is the only path to the MiniMax programs.
    assert not any(exporter == "export_b12x_minimax_aot.py" for exporter, _ in jobs.values())


def test_minimax_expert_scope_is_a_cmake_entry():
    cmake = (ROOT / "native/cmake/shared/fp8_moe.cmake").read_text()
    family, scopes = aot.config_scopes(dict(FAMILY="minimax_m3", SPARK_COUNT="2", SPARK_TP="2"))
    for entry in scopes["CUTEAFD_WIP_EXPERT_FAMILIES"].split(","):
        assert f'entry STREQUAL "{entry}"' in cmake, entry
