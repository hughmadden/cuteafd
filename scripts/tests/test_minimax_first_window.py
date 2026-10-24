"""The default first-window runbook checks files and prints, without touching hardware."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/qualify/minimax_m3/first-window.sh"
SPEC = importlib.util.spec_from_file_location("minimax_first_checks", SCRIPT.with_name("first-window-check.py"))
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


@pytest.fixture
def ready(tmp_path):
    snapshot = tmp_path / checks.REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"model_type": "minimax_m3_vl"}))
    payload = bytes(range(1, 255)) * 20
    header = json.dumps({"tensor": {"dtype": "U8", "shape": [len(payload)],
                                   "data_offsets": [0, len(payload)]}}).encode()
    (snapshot / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"tensor": "model.safetensors"}}))
    golden = tmp_path / "reference"
    window = golden / "windows" / "sample"
    window.mkdir(parents=True)
    (window / "tokens.bin").write_bytes(bytes(2177 * 4))
    for name in ("logits.bin", "layer03.bin", "layer05.bin"):
        (window / name).write_bytes(bytes(16))
    (golden / "windows.json").write_text(json.dumps({"set_sha256": "a" * 64}))
    (golden / "meta.json").write_text(json.dumps({"schema": "cuteafd.fidelity.golden/2",
        "family": "minimax_m3", "checkpoint": checks.MODEL, "snapshot_identity": {"snapshot_revision": checks.REVISION},
        "activation": "served-cache", "set_sha256": "a" * 64, "prefix_qualification": {
            "schema": "cuteafd.fidelity.prefix/1", "passed": True, "finite": True, "different_rows": [],
            "set_sha256": "a" * 64, "snapshot_identity": {"snapshot_revision": checks.REVISION}},
        "windows": [{"id": "sample", "path": "windows/sample"}]}))
    configs = tmp_path / "configs"
    configs.mkdir()
    for card in ("pro6000", "5090"):
        for tp in (2, 3, 4):
            text = (ROOT / "examples/configs" / f"minimax-m31-flash-nvfp4-{card}-tp{tp}.config").read_text()
            # Documentation-only synthetic addresses; no process may use them.
            text += "\nEXPERT_PORT=19471\n" + "\n".join(
                f"SPARK_{rank}_HOST=rank{rank}\nSPARK_{rank}_LANE_A=192.0.2.{rank + 1}"
                for rank in range(tp))
            (configs / f"minimax-m31-flash-nvfp4-{card}-tp{tp}.config").write_text(text)
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = tmp_path / "forbidden-calls"
    for name in ("docker", "ssh", "nvidia-smi", "cuteafd", "curl", "nest", "hwlock.py"):
        path = tools / name
        path.write_text('#!/bin/sh\nprintf "%s\\n" "$0" >>"$FORBIDDEN_CALLS"\nexit 99\n')
        path.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("MINIMAX_") and key != "CUTEAFD_MINIMAX_HARDWARE_WINDOW"}
    env.update(MINIMAX_SNAPSHOT=str(snapshot), MINIMAX_GOLDEN=str(golden), MINIMAX_CONFIG_DIR=str(configs),
               MINIMAX_PLAN_BIN=str(tools / "cuteafd"), MINIMAX_RUN_ID="synthetic",
               MINIMAX_GPU_PRO6000="0", MINIMAX_GPU_5090="0", MINIMAX_PYTHON=sys.executable,
               PATH=f"{tools}:{os.environ['PATH']}", FORBIDDEN_CALLS=str(calls))
    return env, calls


def invoke(ready, *args):
    env, calls = ready
    result = subprocess.run([str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=30)
    assert not calls.exists(), calls.read_text() if calls.exists() else ""
    return result


def test_default_prints_ordered_schedule_and_checks_only_local_files(ready):
    result = invoke(ready)
    assert result.returncode == 0, result.stderr
    stages = ["1. Plan", "2. Build", "3. Single-layer", "4. Golden", "Spark TP4", "TP4 serve", "Spark TP2", "Spark TP3"]
    assert [result.stdout.index(stage) for stage in stages] == sorted(result.stdout.index(stage) for stage in stages)
    assert result.stdout.index("5090: 1.") > result.stdout.index("pro6000: TP3 serve")
    for text in ("top-1 >=94%", "KL <=0.03 nat", "byte-exact prefix restore", "MSA selection", "remain unverified"):
        assert text in result.stdout


def test_card_segment_does_not_claim_other_card(ready):
    result = invoke(ready, "--card", "5090")
    assert result.returncode == 0, result.stderr
    assert "5090: 1." in result.stdout
    assert "pro6000:" not in result.stdout
    assert "Selected configurations passed" not in result.stdout


@pytest.mark.parametrize("ack", [None, "1", "yes", "authorized "])
def test_run_requires_exact_explicit_authorization(ready, ack):
    if ack is not None:
        ready[0]["CUTEAFD_MINIMAX_HARDWARE_WINDOW"] = ack
    result = invoke(ready, "--run")
    assert result.returncode == 2
    assert "CUTEAFD_MINIMAX_HARDWARE_WINDOW=authorized" in result.stderr


def test_authorized_environment_does_not_make_default_execute(ready):
    ready[0]["CUTEAFD_MINIMAX_HARDWARE_WINDOW"] = "authorized"
    assert invoke(ready).returncode == 0


@pytest.mark.parametrize("damage", ["missing", "header", "zero", "sparse"])
def test_missing_or_fake_weight_payloads_fail_closed(ready, damage):
    path = Path(ready[0]["MINIMAX_SNAPSHOT"]) / "model.safetensors"
    data = path.read_bytes()
    boundary = 8 + struct.unpack("<Q", data[:8])[0]
    if damage == "missing":
        path.unlink()
    elif damage == "sparse":
        path.write_bytes(data[:boundary])
        with path.open("r+b") as handle:
            handle.truncate(len(data))
    else:
        path.write_bytes(data[:boundary] + (bytes(len(data) - boundary) if damage == "zero" else b""))
    result = invoke(ready)
    assert result.returncode == 2
    assert "first-window:" in result.stderr


@pytest.mark.parametrize("field,value", [("activation", "served"), ("family", "other"),
    ("diagnostic_only", True), ("prefix_qualification", {"passed": False}), ("windows", [])])
def test_incomplete_or_wrong_reference_fails_closed(ready, field, value):
    path = Path(ready[0]["MINIMAX_GOLDEN"]) / "meta.json"
    meta = json.loads(path.read_text())
    meta[field] = value
    path.write_text(json.dumps(meta))
    assert invoke(ready).returncode == 2


def test_missing_partial_layer_reference_fails_closed(ready):
    (Path(ready[0]["MINIMAX_GOLDEN"]) / "windows/sample/layer03.bin").unlink()
    result = invoke(ready)
    assert result.returncode == 2
    assert "--layers 3 5" in result.stderr


def test_sparse_selection_requires_a_pruned_context(ready):
    (Path(ready[0]["MINIMAX_GOLDEN"]) / "windows/sample/tokens.bin").write_bytes(bytes(2048 * 4))
    result = invoke(ready)
    assert result.returncode == 2
    assert ">=2177" in result.stderr


def test_config_is_data_never_sourced(ready, tmp_path):
    config = Path(ready[0]["MINIMAX_CONFIG_DIR"]) / "minimax-m31-flash-nvfp4-pro6000-tp4.config"
    marker = tmp_path / "should-not-exist"
    with config.open("a") as handle:
        handle.write(f"\nINSTANCE=$(touch {marker})\n")
    invoke(ready)
    assert not marker.exists()


def test_output_root_cannot_escape_task_or_replace_its_root(tmp_path):
    golden = checks.load_harness()
    task = tmp_path / "task"
    task.mkdir()
    assert golden.output_root(task) == task / "golden"
    assert golden.output_root(task, task / "window/run") == task / "window/run"
    for path in (task, task.parent, task / "../outside"):
        with pytest.raises(ValueError, match="child of the task"):
            golden.output_root(task, path)
    (task / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="child of the task"):
        golden.output_root(task, task / "escape/out")



def test_worker_budget_is_the_plans_spark_budget(tmp_path):
    config = tmp_path / "tp2.config"
    config.write_text("COORDINATOR_GPU_BUDGET_GIB=31.8\n")
    plan = tmp_path / "tp2-plan.json"
    plan.write_text(json.dumps({"report": {"spark_budget_bytes": 108 << 30}}))
    command = [sys.executable, str(SCRIPT.with_name("first-window-check.py")), "budgets", str(config), "--plan", str(plan)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["31.8", str(108 << 30)]
    plan.write_text(json.dumps({"report": {}}))
    assert subprocess.run(command, capture_output=True, timeout=10).returncode == 2
