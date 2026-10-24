"""MiniMax launch arguments and CPU-only header admission."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from test_example_configs import PRIVATE
from test_run_family_dispatch import _family_launch_result
from test_serve_launch_new_families import _flag, _hosts, _line

ROOT = Path(__file__).resolve().parents[2]
MODEL = "MiniMaxAI/MiniMax-M3.1-Flash"
REVISION = "54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e"
CONFIG = {"model_type": "minimax_m3_vl", "text_config": {
    "model_type": "minimax_m3", "num_hidden_layers": 60,
    "moe_layer_freq": [0] * 3 + [1] * 57,
}}
PLANNED = {"placement_supported": True, "fits": True, "spark_budget_bytes": 108 << 30}


def _launch(tmp_path, keys="", *, ranks=4, **kwargs):
    return _family_launch_result(tmp_path, CONFIG, MODEL,
        f"SPARK_COUNT={ranks}\nRTX_GPUS=1\nVISION=off\nAUDIO=off\nHOST_CACHE_BYTES=0\n"
        + _hosts(ranks) + keys, encoder_plan=kwargs.pop("encoder_plan", PLANNED), **kwargs)


@pytest.mark.parametrize("ranks", [2, 3, 4])
@pytest.mark.parametrize("kv", ["fp8", "bf16"])
def test_minimax_launch_matches_planner(tmp_path, ranks, kv):
    result = _launch(tmp_path, f"KV_CACHE={kv}\nRTX_EXPERT_LAYERS=auto\n", ranks=ranks)
    serve = _line(result, "serve-minimax --snapshot")
    plan = _line(result, "cuteafd plan")
    for flag, value in (("--kv-cache", kv), ("--speculator", "off")):
        assert _flag(serve, flag) == _flag(plan, flag) == value
    assert _flag(serve, "--rtx-expert-layers") == "auto"
    assert _flag(serve, "--spark-budget-bytes") == str(108 << 30)
    assert _flag(serve, "--max-sequences") == _flag(plan, "--concurrency") == "64"
    assert _flag(serve, "--prefill-batch-tokens") == _flag(plan, "--prefill-rows") == "4096"
    assert _flag(serve, "--prefill-lanes") == _flag(plan, "--prefill-lanes") == "2"
    assert _flag(plan, "--spark-ranks") == str(ranks)
    assert "--mtp" not in serve and "--split-device" not in serve
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert len(workers) == ranks
    assert all(f"--world {ranks}" in line for line in workers)


@pytest.mark.parametrize("keys,diagnostic", [
    ("RTX_GPUS=2\n", "two-GPU head split is not implemented"),
    ("RTX_GPUS=invalid\n", "RTX_GPUS must be auto, 1 or 2"),
    ("COORDINATOR_SPLIT=invalid\n", "COORDINATOR_SPLIT must be auto, heads or off"),
    ("COORDINATOR_GPUS=0,1\n", "two-GPU head split is not implemented"),
    ("COORDINATOR_GPU=0,1\n", "two-GPU head split is not implemented"),
    ("COORDINATOR_SPLIT=heads\n", "two-GPU head split is not implemented"),
    ("SPECULATOR=mtp\n", "supports SPECULATOR=dspark or off"),
    ("SPECULATOR=dflash2\n", "supports SPECULATOR=dspark or off"),
    ("RTX_EXPERT_LAYERS=max\n", "must be auto or a fixed routed-layer count"),
    ("RTX_EXPERT_LAYERS=25%\n", "must be auto or a fixed routed-layer count"),
    ("RTX_EXPERT_LAYERS=58\n", "checkpoint has 57 routed layers"),
    ("KV_CACHE=int8\n", "KV_CACHE must be bf16 or fp8"),
    ("SPARK_COUNT=6\n", "requires Spark TP2, TP3 or TP4"),
    ("EXPERT_BACKEND=local\nSPARK_COUNT=0\n", "requires Spark TP2, TP3 or TP4"),
    ("CONCURRENCY=65\n", "CONCURRENCY must be 1..64"),
    ("PREFILL_BATCH_TOKENS=4097\n", "PREFILL_BATCH_TOKENS must be 1..4096"),
    ("CONCURRENCY=16\nPREFILL_BATCH_TOKENS=8\n", "CONCURRENCY must not exceed PREFILL_BATCH_TOKENS"),
    ("PREFILL_LANES=5\n", "PREFILL_LANES must be 1 to 4"),
    ("MAX_CONTEXT_TOKENS=1048577\n", "MAX_CONTEXT_TOKENS must be 0 (automatic) or 1..1048576"),
    ("ATTENTION_PLACEMENT=heads\n", "supports only ATTENTION_PLACEMENT=auto"),
    ("EMBEDDING=host\n", "requires EMBEDDING=gpu"),
    ("SPECULATOR_DEPTH=3\n", "unused configuration key: SPECULATOR_DEPTH"),
])
def test_minimax_refuses_unsupported_launch(tmp_path, keys, diagnostic):
    result = _launch(tmp_path, keys)
    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert "docker run -d" not in result.stderr


def test_minimax_refuses_push_loading(tmp_path):
    result = _launch(tmp_path, extra_env={"CUTEAFD_PUSH_SOCKET": "enabled"})
    assert result.returncode != 0
    assert "mount-loaded Spark experts" in result.stderr
    assert "docker run -d" not in result.stderr


def _cpu_dry_run(tmp_path: Path, config_text: str):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps(CONFIG))
    config = tmp_path / "launch.config"
    config.write_text(config_text + f"\nMODEL_ID={snapshot}\n")
    tools = tmp_path / "bin"
    tools.mkdir()
    calls = tmp_path / "forbidden.calls"
    for name in ("docker", "ssh", "nvidia-smi", "nest"):
        tool = tools / name
        tool.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$0 $*" >> "$FORBIDDEN_CALLS"\nexit 99\n')
        tool.chmod(0o755)
    planner = tools / "cuteafd-plan"
    planner.write_text('''#!/usr/bin/env python3
import json, sys
print(json.dumps({"args": sys.argv[1:], "executable": False}))
''')
    planner.chmod(0o755)
    env = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}",
           "CUTEAFD_PLAN_BIN": str(planner), "FORBIDDEN_CALLS": str(calls)}
    result = subprocess.run([str(ROOT / "run.sh"), "--config", str(config), "--dry-run"],
                            capture_output=True, text=True, env=env, timeout=30)
    assert not calls.exists(), calls.read_text() if calls.exists() else ""
    return result


@pytest.mark.parametrize("card,budget", [("5090", "31.8"), ("pro6000", "95.5")])
@pytest.mark.parametrize("tp", [2, 3, 4])
def test_minimax_examples_have_cpu_only_dry_run(tmp_path, card, budget, tp):
    example = ROOT / "examples/configs" / f"minimax-m31-flash-nvfp4-{card}-tp{tp}.config"
    text = example.read_text()
    assert all(needle not in text.lower() for needle in PRIVATE)
    assert f"MODEL_ID={MODEL}\n" in text and f"MODEL_REVISION={REVISION}\n" in text
    assert f"COORDINATOR_GPU_BUDGET_GIB={budget}\n" in text
    assert f"RTX_EXPERT_LAYERS={'auto' if tp == 2 else '0'}\n" in text
    result = _cpu_dry_run(tmp_path, text)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert not output["executable"]
    args = output["args"]
    prefill_rows = "1024" if (card, tp) == ("5090", 2) else "4096"
    for flag, value in (("--spark-ranks", str(tp)), ("--rtx", "1"),
                        ("--coordinator-gpu-budget-gib", budget), ("--speculator", "off"),
                        ("--kv-cache", "fp8"), ("--prefill-rows", prefill_rows)):
        assert args[args.index(flag) + 1] == value
    assert "--require-ready" not in args
    assert "header admission only" in result.stderr


def test_minimax_cpu_dry_run_checks_family_keys(tmp_path):
    result = _cpu_dry_run(tmp_path, "SPARK_COUNT=4\nSPECULATOR_DEPTH=3\n")
    assert result.returncode != 0
    assert "unused configuration key: SPECULATOR_DEPTH" in result.stderr


@pytest.mark.parametrize("kind", ["spark", "rtx"])
def test_minimax_routes_bundled_vision_from_the_plan(tmp_path, kind):
    config = {**CONFIG, "vision_config": {"depth": 32}}
    placement = {"kind": kind, "rank" if kind == "spark" else "gpu": 0}
    plan = {**PLANNED, "encoder_plan_hash": "ab" * 32,
            "encoder": {"kind": placement, "replicas": []}, "max_image_tokens": 4096}
    result = _family_launch_result(tmp_path, config, MODEL,
        "SPARK_COUNT=4\nRTX_GPUS=1\nVISION=auto\nAUDIO=off\n" + _hosts(4),
        encoder_plan=plan)
    serve = _line(result, "serve-minimax --snapshot")
    assert _flag(serve, "--vision") == f"{kind}:0"
    assert _flag(serve, "--host-cache-bytes") == "0"
    if kind == "spark":
        assert _flag(serve, "--encoder-plan-hash") == "ab" * 32
        workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
        assert sum("--encoder-listen" in line for line in workers) == 1
        assert any("--encoder-max-tokens 4096" in line for line in workers)


def test_minimax_forwards_explicit_spark_budget_to_plan_and_runtime(tmp_path):
    budget = 106 << 30
    result = _launch(tmp_path, f"SPARK_DEVICE_BUDGET_BYTES={budget}\n")
    serve = _line(result, "serve-minimax --snapshot")
    plan = _line(result, "cuteafd plan")
    assert _flag(serve, "--spark-budget-bytes") == str(budget)
    assert float(_flag(plan, "--spark-budget-gib")) == 106
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert workers and all(f"--device-budget-bytes {budget}" in line for line in workers)


def test_minimax_bundled_dspark_is_explicit(tmp_path):
    result = _launch(tmp_path, "SPECULATOR=dspark\n")
    assert result.returncode == 0, result.stderr
    serve = _line(result, "serve-minimax --snapshot")
    plan = _line(result, "cuteafd plan")
    assert _flag(serve, "--speculator") == _flag(plan, "--speculator") == "dspark"
    assert "--draft " not in serve
