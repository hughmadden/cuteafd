"""MiniMax launch arguments, the shared dry run and the example configs."""
from __future__ import annotations

import json
import re
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
}, "quantization_config": {"quant_method": "mxfp8", "moe_quant_algo": "NVFP4",
                           "moe_quant_format": "nvfp4-pack-quantized", "group_size": 16}}
PLANNED = {"placement_supported": True, "fits": True, "spark_budget_bytes": 108 << 30}
# The qualified DFlash2 draft's revision, the launcher's default.
RADIXARK_REVISION = "923bf1d4b567d5ccb71b0488c6e61b5e4794e92a"


def _launch(tmp_path, keys="", *, ranks=4, **kwargs):
    return _family_launch_result(tmp_path, CONFIG, MODEL,
        f"SPARK_COUNT={ranks}\nRTX_GPUS=1\nVISION=off\nAUDIO=off\nHOST_CACHE_BYTES=0\n"
        + _hosts(ranks) + keys, encoder_plan=kwargs.pop("encoder_plan", PLANNED), **kwargs)


@pytest.mark.parametrize("ranks", [2, 3, 4, 6])
@pytest.mark.parametrize("kv", ["fp8", "bf16"])
def test_minimax_launch_matches_planner(tmp_path, ranks, kv):
    result = _launch(tmp_path, f"KV_CACHE={kv}\nRTX_EXPERT_LAYERS=auto\n", ranks=ranks)
    serve = _line(result, "serve-minimax --snapshot")
    plan = _line(result, "cuteafd plan")
    # DFlash2 is MiniMax's default speculator.
    for flag, value in (("--kv-cache", kv), ("--speculator", "dflash2")):
        assert _flag(serve, flag) == _flag(plan, flag) == value
    assert _flag(serve, "--draft") == _flag(plan, "--drafter-snapshot")
    assert _flag(serve, "--rtx-expert-layers") == "auto"
    assert _flag(serve, "--spark-budget-bytes") == str(108 << 30)
    assert _flag(serve, "--max-sequences") == _flag(plan, "--concurrency") == "64"
    # Prefill units: the plan resolves #762's shape (no key pins it) and serve runs what it resolved.
    unit, waves = ("2752", "3") if ranks == 6 else ("4096", "2")
    assert (_flag(serve, "--prefill-rows"), _flag(serve, "--prefill-waves")) == (unit, waves)
    assert _flag(plan, "--prefill-rows") is None and _flag(plan, "--prefill-waves") is None
    assert not any(flag in serve + plan for flag in ("--prefill-lanes", "--prefill-batch-tokens", "--prefill-span-rows"))
    assert f"Resolved prefill: PREFILL_ROWS={unit} PREFILL_WAVES={waves}" in result.stdout
    assert _flag(plan, "--spark-ranks") == str(ranks)
    assert "--mtp" not in serve and "--split-device" not in serve
    workers = [line for line in result.stderr.splitlines() if "cuteafd expertd-native" in line]
    assert len(workers) == ranks
    assert all(f"--world {ranks}" in line for line in workers)


@pytest.mark.parametrize("keys,quota,store", [
    ("", "auto", "on-evict"),
    ("HOST_CACHE_STORE=on-retain\n", "auto", "on-retain"),
    ("HOST_CACHE_BYTES=16GiB\n", "16GiB", "on-evict"),
    ("HOST_CACHE_BYTES=0\nHOST_CACHE_STORE=on-retain\n", None, None),
])
def test_minimax_host_prefix_tier_defaults_on_and_stores_on_eviction(tmp_path, keys, quota, store):
    result = _family_launch_result(tmp_path, CONFIG, MODEL,
        "SPARK_COUNT=4\nRTX_GPUS=1\nVISION=off\nAUDIO=off\n" + _hosts(4) + keys, encoder_plan=PLANNED)
    serve = _line(result, "serve-minimax --snapshot")
    assert _flag(serve, "--host-cache-bytes") == quota
    assert _flag(serve, "--host-cache-store") == store


@pytest.mark.parametrize("keys,diagnostic", [
    ("RTX_GPUS=2\n", "two-GPU head split is not implemented"),
    ("RTX_GPUS=invalid\n", "RTX_GPUS must be auto, 1 or 2"),
    ("COORDINATOR_SPLIT=invalid\n", "COORDINATOR_SPLIT must be auto, heads or off"),
    ("COORDINATOR_GPUS=0,1\n", "two-GPU head split is not implemented"),
    ("COORDINATOR_GPU=0,1\n", "two-GPU head split is not implemented"),
    ("COORDINATOR_SPLIT=heads\n", "two-GPU head split is not implemented"),
    ("SPECULATOR=mtp\n", "supports SPECULATOR=dspark, dflash2 or off"),
    ("SPECULATOR=dflash2\nSPECULATOR_FP8=maybe\n", "SPECULATOR_FP8 must be auto, on or off"),
    ("SPECULATOR=dflash2\nDRAFT_KV_CACHE=int8\n", "DRAFT_KV_CACHE must be bf16 or fp8"),
    ("SPECULATOR=dspark\nDRAFT_KV_CACHE=bf16\n", "DRAFT_KV_CACHE apply to SPECULATOR=dflash2"),
    ("SPECULATOR=off\nSPECULATOR_MODEL_ID=other/draft\n", "apply to SPECULATOR=dflash2"),
    ("SPECULATOR=dspark\nSPECULATOR_MODEL_REVISION=abc\n", "apply to SPECULATOR=dflash2"),
    ("RTX_EXPERT_LAYERS=max\n", "must be auto or a fixed routed-layer count"),
    ("RTX_EXPERT_LAYERS=25%\n", "must be auto or a fixed routed-layer count"),
    ("RTX_EXPERT_LAYERS=58\n", "checkpoint has 57 routed layers"),
    ("KV_CACHE=int8\n", "KV_CACHE must be bf16 or fp8"),
    ("HOST_CACHE_STORE=always\n", "HOST_CACHE_STORE must be on-evict or on-retain"),
    ("EXPERT_BACKEND=local\nSPARK_COUNT=0\n", "MiniMax requires Spark experts"),
    ("CONCURRENCY=65\n", "CONCURRENCY must be 1..64"),
    # The lane grammar is gone: prefill units are PREFILL_ROWS and PREFILL_WAVES.
    ("PREFILL_LANES=2\n", "unknown configuration key: PREFILL_LANES"),
    ("PREFILL_BATCH_TOKENS=4096\n", "unused configuration key: PREFILL_BATCH_TOKENS"),
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


@pytest.mark.parametrize("keys,pinned,resolved", [
    ("PREFILL_ROWS=2048\nPREFILL_WAVES=4\n", ("2048", "4"), (2048, 4)),
    ("PREFILL_WAVES=3\n", (None, "3"), (2752, 3)),
    ("PREFILL_ROWS=1024\n", ("1024", None), (1024, 2)),
])
def test_minimax_pins_prefill_units_through_the_plan(tmp_path, keys, pinned, resolved):
    # Set keys reach the plan, as for Kimi K3; serve takes the shape the plan resolved.
    result = _launch(tmp_path, keys, deployment_overrides={"prefill_unit_rows": resolved[0], "prefill_waves": resolved[1]})
    serve, plan = _line(result, "serve-minimax --snapshot"), _line(result, "cuteafd plan")
    assert (_flag(plan, "--prefill-rows"), _flag(plan, "--prefill-waves")) == pinned
    assert (_flag(serve, "--prefill-rows"), _flag(serve, "--prefill-waves")) == tuple(map(str, resolved))
    assert "--prefill-lanes" not in plan + serve


def test_minimax_spark_counts_come_from_the_plan(tmp_path):
    # The launcher sets no list of its own: the plan refuses a count the expert package does not
    # ship (plan::experts::spark_worlds), and the launch stops on its deployment guard.
    refusal = "native placement refused: MiniMax admission does not support 5 Spark ranks: its expert package ships Spark worlds [2, 3, 4, 6]"
    result = _launch(tmp_path, ranks=5, encoder_plan={
        **PLANNED, "placement_supported": False, "hints": [{"what": refusal, "how": "Choose a packaged world."}]})
    assert result.returncode != 0
    assert refusal in result.stderr and "invalid deployment specification" in result.stderr
    assert "docker run -d" not in result.stderr


def test_minimax_on_an_rtx_5090_stops_on_the_plans_refusal(tmp_path):
    # The plan refuses an RTX 5090-class coordinator budget by name; the launch passes the
    # budget through and stops on the deployment guard.
    refusal = "native placement refused: MiniMax on an RTX 5090 is not supported on this engine yet"
    result = _launch(tmp_path, "COORDINATOR_GPU_BUDGET_GIB=31.8\n", gpu_total_mib=32607, gpu_free_mib=32107,
                     encoder_plan={**PLANNED, "placement_supported": False, "hints": [{"what": refusal, "how": ""}]})
    assert result.returncode != 0 and refusal in result.stderr
    plan = next(line for line in result.stderr.splitlines() if "cuteafd plan" in line)
    assert _flag(plan, "--coordinator-gpu-budget-gib") == "31.8"
    assert "docker run -d" not in result.stderr


def test_minimax_refuses_push_loading(tmp_path):
    result = _launch(tmp_path, extra_env={"CUTEAFD_PUSH_SOCKET": "enabled"})
    assert result.returncode != 0
    assert "mount-loaded Spark experts" in result.stderr
    assert "docker run -d" not in result.stderr


def test_minimax_dry_run_takes_the_shared_deployment_plan(tmp_path):
    result = _launch(tmp_path, extra_args=("--wip", "qualification", "--dry-run"))
    assert result.returncode == 0, result.stderr
    # The deployment the launch would take is stdout's last line, after the resolved prefill units.
    deployment = json.loads(result.stdout.splitlines()[-1])
    assert deployment["family"] == "minimax_m3" and deployment["wip_launchable"]
    assert (deployment["prefill_unit_rows"], deployment["prefill_waves"]) == (4096, 2)
    assert "Resolved prefill: PREFILL_ROWS=4096 PREFILL_WAVES=2" in result.stdout
    plan = _line(result, "cuteafd plan")
    assert _flag(plan, "--spark-ranks") == "4" and "--deployment" in plan
    assert "docker run -d" not in result.stderr and "serve-minimax --snapshot" not in result.stderr


# The examples' card, as CUDA reports it: an RTX PRO 6000 (PRO_TOTAL_BYTES).
CARD_MIB = {"pro6000": 101_973_491_712 >> 20}


def _example_launch(tmp_path, pairs, card, tp):
    """The coordinator, plan and worker commands a WIP launch of these keys would start."""
    keys = "".join(f"{key}={value}\n" for key, value in pairs if key not in ("MODEL_ID", "MODEL_REVISION"))
    result = _family_launch_result(tmp_path, CONFIG, MODEL, keys + _hosts(max(tp, 4)), encoder_plan=PLANNED,
        gpu_total_mib=CARD_MIB[card], gpu_free_mib=CARD_MIB[card] - 500, extra_args=("--wip", "example"))
    commands = [re.sub(r"/tmp/\S*?/(hf|home|repo)/", r"<\1>/", line.strip()) for line in result.stderr.splitlines()
                if any(name in line for name in ("serve-minimax --snapshot", "cuteafd plan", "cuteafd expertd-native"))]
    return result.returncode, sorted(commands)


@pytest.mark.parametrize("card,tp", [("pro6000", tp) for tp in (2, 3, 4, 6)])
def test_minimax_examples_state_only_what_differs_from_the_launcher(tmp_path, card, tp):
    example = ROOT / "examples/configs" / f"minimax-m31-flash-nvfp4-{card}-tp{tp}.config"
    text = example.read_text()
    assert all(needle not in text.lower() for needle in PRIVATE)
    assert f"./run.sh --config examples/configs/{example.name} --wip SLOT --dry-run" in text
    pairs = re.findall(r"^([A-Z_0-9]+)=(.*)$", text, re.M)
    assert dict(pairs)["MODEL_ID"] == MODEL and dict(pairs)["MODEL_REVISION"] == REVISION
    assert dict(pairs)["SPARK_COUNT"] == str(tp)
    code, commands = _example_launch(tmp_path / "example", pairs, card, tp)
    assert code == 0 and len(commands) == tp + 2, commands
    # Every key changes what the launcher starts: none restates a default. The release pair is
    # the promoted one every example names; a WIP launch runs the slot instead.
    for index, (key, value) in enumerate(pairs):
        if key in ("MODEL_ID", "MODEL_REVISION", "COORDINATOR_DOCKER_INFERENCE", "SPARK_EXPERT_DOCKER_INFERENCE"):
            continue
        assert _example_launch(tmp_path / key, pairs[:index] + pairs[index + 1:], card, tp) != (code, commands), \
            f"{key}={value} is the launcher's default"


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
    assert _flag(serve, "--host-cache-bytes") == "auto"
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


def test_minimax_dspark_takes_the_shared_draft_length_keys(tmp_path):
    result = _launch(tmp_path, "SPECULATOR=dspark\nDRAFT_POLICY=simple-rc\nDRAFT_ROW_COST=power\nSPECULATOR_DRAFTS=3\n")
    assert result.returncode == 0, result.stderr
    serve = _line(result, "serve-minimax --snapshot")
    assert "CUTEAFD_DRAFT_POLICY=simple-rc" in serve and "CUTEAFD_DRAFT_ROW_COST=power" in serve
    assert _flag(serve, "--draft-fixed") == "3"


def test_minimax_bundled_dspark_is_explicit(tmp_path):
    result = _launch(tmp_path, "SPECULATOR=dspark\n")
    assert result.returncode == 0, result.stderr
    serve = _line(result, "serve-minimax --snapshot")
    plan = _line(result, "cuteafd plan")
    assert _flag(serve, "--speculator") == _flag(plan, "--speculator") == "dspark"
    assert "--draft " not in serve


def test_minimax_defaults_to_the_users_own_dflash2_download_and_keeps_dspark_and_off(tmp_path):
    # No SPECULATOR: DFlash2 at the qualified revision, from the user's Hugging Face cache.
    result = _launch(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "note: MiniMax M3 drafts with DFlash2" in result.stderr
    serve = _line(result, "serve-minimax --snapshot")
    assert _flag(serve, "--speculator") == "dflash2"
    assert _flag(serve, "--draft").endswith(f"models--RadixArk--Minimax-M3.1-Flash-Dflash2/snapshots/{RADIXARK_REVISION}")
    # Explicit choices stay: the release's bundled DSpark, or no drafter.
    for value in ("dspark", "off"):
        tmp = tmp_path / value
        tmp.mkdir()
        result = _launch(tmp, f"SPECULATOR={value}\n")
        assert result.returncode == 0, result.stderr
        serve = _line(result, "serve-minimax --snapshot")
        assert _flag(serve, "--speculator") == value and "--draft " not in serve
    # A draft that is not downloaded refuses before launch, pointing at its page and the alternatives.
    missing = tmp_path / "missing"
    missing.mkdir()
    result = _launch(missing, "SPECULATOR_MODEL_REVISION=0123456789abcdef0123456789abcdef01234567\n")
    assert result.returncode != 0 and "docker run -d" not in result.stderr
    for needle in ("which cuteafd does not distribute", "https://huggingface.co/RadixArk/Minimax-M3.1-Flash-Dflash2",
                   "hf download RadixArk/Minimax-M3.1-Flash-Dflash2 --revision 0123456789abcdef", "SPECULATOR=dspark",
                   "SPECULATOR=off"):
        assert needle in result.stderr, (needle, result.stderr)


def test_minimax_speculation_trace_reaches_the_server_for_selector_calibration(tmp_path):
    trace = tmp_path / "traces" / "dflash2.jsonl"
    result = _launch(tmp_path, f"SPECULATOR=dflash2\nSPECULATOR_DRAFTS=7\nSPECULATION_TRACE={trace}\n")
    assert result.returncode == 0, result.stderr
    assert f"CUTEAFD_SPECULATION_TRACE={trace}" in result.stderr
    assert _flag(_line(result, "serve-minimax --snapshot"), "--draft-fixed") == "7"
    relative = tmp_path / "relative"
    relative.mkdir()
    result = _launch(relative, "SPECULATOR=dflash2\nSPECULATION_TRACE=relative.jsonl\n")
    assert result.returncode != 0 and "SPECULATION_TRACE must be an absolute path" in result.stderr


@pytest.mark.parametrize("keys,fp8,kv", [("", None, "fp8"), ("SPECULATOR_FP8=off\nDRAFT_KV_CACHE=bf16\n", "false", "bf16"),
                                         ("SPECULATOR_FP8=on\n", "true", "fp8")])
def test_minimax_dflash2_serves_and_plans_the_radixark_drafter_on_the_shared_drafter(tmp_path, keys, fp8, kv):
    result = _launch(tmp_path, "SPECULATOR=dflash2\nSPECULATOR_DRAFTS=5\n" + keys)
    assert result.returncode == 0, result.stderr
    serve = _line(result, "serve-minimax --snapshot")
    plan = _line(result, "cuteafd plan")
    assert _flag(serve, "--speculator") == _flag(plan, "--speculator") == "dflash2"
    draft = _flag(serve, "--draft")
    assert draft.endswith(f"models--RadixArk--Minimax-M3.1-Flash-Dflash2/snapshots/{RADIXARK_REVISION}")
    assert _flag(plan, "--drafter-snapshot") == draft
    assert _flag(serve, "--draft-fp8") == fp8
    assert ("--minimax-draft-bf16" in plan) == (fp8 == "false")
    assert _flag(serve, "--draft-kv-cache") == _flag(plan, "--draft-kv-cache") == kv
    assert _flag(serve, "--draft-fixed") == "5"
