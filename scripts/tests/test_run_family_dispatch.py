"""run.sh hands every family but DeepSeek V4.1 to scripts/launch/run-family.sh."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "lib").mkdir(parents=True)
    (repo / "scripts" / "launch").mkdir(parents=True)
    shutil.copy(ROOT / "run.sh", repo / "run.sh")
    for name in ("release-common.sh", "checkpoint-family.py"):
        shutil.copy(ROOT / "scripts" / "lib" / name, repo / "scripts" / "lib" / name)
    fake = repo / "scripts" / "launch" / "run-family.sh"
    fake.write_text('#!/usr/bin/env bash\nprintf "run-family %s\\n" "$*"\n')
    fake.chmod(0o755)
    return repo


def _snapshot(hf: Path, model: str, config: dict) -> None:
    root = hf / "hub" / f"models--{model.replace('/', '--')}"
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text("abc")
    (root / "snapshots" / "abc").mkdir(parents=True)
    (root / "snapshots" / "abc" / "config.json").write_text(json.dumps(config))


def _run(repo: Path, hf: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HF_HOME": str(hf)}
    return subprocess.run(["bash", str(repo / "run.sh"), *args], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=60)


def test_glm_flash_config_goes_to_run_family(tmp_path: Path) -> None:
    repo, hf = _repo(tmp_path), tmp_path / "hf"
    _snapshot(hf, "zai-org/GLM-5.3-Flash", {"model_type": "glm5_next", "num_hidden_layers": 45,
                                           "layer_types": ["linear_attention"] * 45,
                                           "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42})
    (repo / "glmf.config").write_text("MODEL_ID=zai-org/GLM-5.3-Flash\nDRAFT_MODEL_ID=incoai/x\n")
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--restart")
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"run-family --config {repo / 'glmf.config'} --family glm5_flash --restart\n"
    # DeepSeek V4.1 options do not apply to other families.
    result = _run(repo, hf, "--config", str(repo / "glmf.config"), "--concurrency", "4")
    assert result.returncode != 0 and "take only --config and --restart" in result.stderr


def test_family_table_names_every_launchable_family() -> None:
    table = ROOT / "scripts" / "lib" / "checkpoint-family.py"
    cases = {
        "deepseek_v41": {"model_type": "deepseek_v41"},
        "deepseek_v4": {"model_type": "deepseek_v4"},
        "glm5": {"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3},
        "glm5_flash": {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
                       "layer_types": ["linear_attention", "deepseek_sparse_attention"]},
        "mimo_v2": {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
        "qwen4": {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 2,
                                                             "layer_types": ["linear_attention", "full_attention"]}},
    }
    for family, config in cases.items():
        path = Path(os.environ.get("TMPDIR", "/tmp")) / f"family-{os.getpid()}-{family}.json"
        path.write_text(json.dumps(config))
        try:
            out = subprocess.run(["python3", str(table), str(path)], capture_output=True, text=True, check=True)
        finally:
            path.unlink()
        assert out.stdout.split()[0] == family


def test_family_table_matches_the_rust_launch_fixtures(tmp_path: Path) -> None:
    """checkpoint-family.py and cuteafd-loader plan::launch read the same cases the
    same way (both spellings, exact pattern lengths, agreement)."""
    table = ROOT / "scripts" / "lib" / "checkpoint-family.py"
    fixtures = ROOT / "rust" / "crates" / "cuteafd-loader" / "tests" / "fixtures" / "launch-families.json"
    for case in json.loads(fixtures.read_text()):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(case["config"]))
        out = subprocess.run(["python3", str(table), str(path)], capture_output=True, text=True)
        got = out.stdout.strip() if out.returncode == 0 else None
        assert got == case["line"], (case["name"], out.stderr)


def _family_launch_result(tmp_path: Path, family_config: dict, model: str, keys: str,
                          physical_gpus: tuple[int, ...] = (0, 1)) -> subprocess.CompletedProcess[str]:
    """Run the real run-family.sh up to its docker calls (docker/ssh/nest/curl are stubs
    that print their argv) and return what it would launch."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "lib").mkdir(parents=True)
    (repo / "scripts" / "launch").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "launch" / "run-family.sh", repo / "scripts" / "launch")
    shutil.copy(ROOT / "scripts" / "lib" / "checkpoint-family.py", repo / "scripts" / "lib")
    hf = tmp_path / "hf"
    _snapshot(hf, model, family_config)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("docker", "ssh", "nest"):
        (bin_dir / tool).write_text('#!/usr/bin/env bash\nprintf "%s " "$(basename "$0")" "$@" >&2; echo >&2\n'
                                    'case "$*" in *"docker logs"*) echo "worker ready" ;; esac\n')
        (bin_dir / tool).chmod(0o755)
    (bin_dir / "curl").write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' \'{"data":[{"id":"test/model"}]}\'\n')
    (bin_dir / "curl").chmod(0o755)
    (bin_dir / "nvidia-smi").write_text("#!/usr/bin/env bash\nprintf '%s\\n' " +
                                         " ".join(map(str, physical_gpus)) + "\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    config = repo / "f.config"
    config.write_text(f"MODEL_ID={model}\nSPARK_COUNT=1\nSPARK_0_HOST=h0\nSPARK_0_LANE_A=10.0.0.1\n{keys}")
    env = {**os.environ, "HF_HOME": str(hf), "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(repo / "scripts" / "launch" / "run-family.sh"), "--config", str(config)],
                          env=env, capture_output=True, text=True, timeout=30)


def _family_launch_lines(tmp_path: Path, family_config: dict, model: str, keys: str) -> str:
    return _family_launch_result(tmp_path, family_config, model, keys).stderr


SPLIT_CONFIGS = {
    "qwen4": {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 2,
               "layer_types": ["linear_attention", "full_attention"]}},
    "glm5_flash": {"model_type": "glm5_next", "num_hidden_layers": 2,
                   "mlp_layer_types": ["sparse"] * 2,
                   "layer_types": ["linear_attention", "deepseek_sparse_attention"]},
    "mimo_flash": {"model_type": "mimo_v2_flash", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
    "mimo_pro": {"model_type": "mimo_v2", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
}


@pytest.mark.parametrize("checkpoint", ["qwen4", "glm5_flash"])
@pytest.mark.parametrize("keys", ["RTX_GPUS=2\n", "COORDINATOR_GPUS=0,1\n", "COORDINATOR_SPLIT=heads\n"])
def test_explicit_split_rejects_missing_checkpoint_kernels_before_launch(
        tmp_path: Path, checkpoint: str, keys: str) -> None:
    model = "zai-org/GLM-5.3-Flash" if checkpoint == "glm5_flash" else "test/model"
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS[checkpoint], model, keys)
    assert result.returncode == 2, result.stderr
    assert "two-GPU head split is unsupported" in result.stderr
    assert "kernels" in result.stderr and "COORDINATOR_SPLIT=off" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


@pytest.mark.parametrize("checkpoint", ["qwen4", "glm5_flash"])
def test_auto_keeps_unsupported_checkpoint_on_one_gpu(tmp_path: Path, checkpoint: str) -> None:
    model = "zai-org/GLM-5.3-Flash" if checkpoint == "glm5_flash" else "test/model"
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS[checkpoint], model, "")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-" in line)
    assert "device=0" in launch and "--split-device" not in launch
    assert "auto selected GPU 0 alone" in result.stderr


def test_split_off_explicitly_uses_the_first_gpu(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  "RTX_GPUS=2\nCOORDINATOR_GPUS=1,0\nCOORDINATOR_SPLIT=off\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "device=1" in launch and "--split-device" not in launch


@pytest.mark.parametrize("checkpoint", ["mimo_flash", "mimo_pro"])
def test_both_mimo_model_types_keep_their_supported_split(tmp_path: Path, checkpoint: str) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS[checkpoint], "test/model",
                                  "COORDINATOR_GPUS=1,0\n")
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    assert "--device 1 --split-device 0" in launch and "device=0,1" in launch


def test_auto_uses_one_gpu_when_only_one_exists(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["mimo_pro"], "test/model", "", physical_gpus=(0,))
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-mimo" in line)
    assert "device=0" in launch and "--split-device" not in launch


def test_split_off_needs_only_the_first_physical_gpu(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["qwen4"], "test/model",
                                  "RTX_GPUS=2\nCOORDINATOR_SPLIT=off\n", physical_gpus=(0,))
    assert result.returncode == 0, result.stderr
    launch = next(line for line in result.stderr.splitlines() if "cuteafd serve-qwen4" in line)
    assert "device=0" in launch and "--split-device" not in launch


@pytest.mark.parametrize("keys", ["RTX_GPUS=2\n", "COORDINATOR_GPUS=0,1\n",
                                 "COORDINATOR_SPLIT=heads\n"])
def test_explicit_split_requires_two_physical_gpus(tmp_path: Path, keys: str) -> None:
    result = _family_launch_result(tmp_path, SPLIT_CONFIGS["mimo_pro"], "test/model", keys,
                                  physical_gpus=(0,))
    assert result.returncode == 2, result.stderr
    assert "physical" in result.stderr
    assert not any(line.startswith(("docker ", "ssh ", "nest ")) for line in result.stderr.splitlines())


def test_speculator_and_its_pre_rename_keys_launch_the_same(tmp_path: Path) -> None:
    config = {"model_type": "mimo_v2_flash", "num_hidden_layers": 3, "moe_layer_freq": [0, 1, 1]}
    new = _family_launch_lines(tmp_path / "a", config, "XiaomiMiMo/MiMo-V2-Flash", "SPECULATOR=mtp\nSPECULATOR_DEPTH=2\n")
    old = _family_launch_lines(tmp_path / "b", config, "XiaomiMiMo/MiMo-V2-Flash", "MTP=2\n")
    launch = lambda text: [l for l in text.splitlines() if "cuteafd serve-mimo" in l]
    assert launch(new) and "--mtp 2" in launch(new)[0]
    assert [l.replace(str(tmp_path / "b"), "X") for l in launch(old)] == \
        [l.replace(str(tmp_path / "a"), "X") for l in launch(new)]
    assert "deprecated" in old and "deprecated" not in new
    bad = _family_launch_lines(tmp_path / "c", config, "XiaomiMiMo/MiMo-V2-Flash", "SPECULATOR=dspark\n")
    assert "does not apply to mimo_v2" in bad


def test_qwen_launches_with_the_prefix_cache_keys(tmp_path: Path) -> None:
    config = {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 2,
                                                         "layer_types": ["linear_attention", "full_attention"]}}
    keys = "PREFIX_CACHE_ENTRIES=8\nHOST_CACHE_BYTES=16GiB\nPOOL_TOKENS=65536\n"
    text = _family_launch_lines(tmp_path / "a", config, "Qwen/Qwen3.8-Flash-Next", keys)
    launch = [l for l in text.splitlines() if "cuteafd serve-qwen4" in l]
    assert launch, text
    for flag in ("--prefix-cache-entries 8", "--host-cache-bytes 16GiB", "--pool-tokens 65536"):
        assert flag in launch[0], flag
    default = _family_launch_lines(tmp_path / "b", config, "Qwen/Qwen3.8-Flash-Next", "")
    launch = [l for l in default.splitlines() if "cuteafd serve-qwen4" in l]
    assert "--prefix-cache-entries 20" in launch[0]
    assert "--host-cache-bytes" not in launch[0] and "--pool-tokens" not in launch[0]


def test_deepseek_v4_launches_with_the_prefix_cache_keys(tmp_path: Path) -> None:
    config = {"model_type": "deepseek_v4"}
    keys = "PREFIX_CACHE_ENTRIES=12\nHOST_CACHE_BYTES=32GiB\nPOOL_TOKENS=524288\nSPECULATOR=dspark\n"
    text = _family_launch_lines(tmp_path / "a", config, "deepseek-ai/DeepSeek-V4-Flash-0731", keys)
    launch = [l for l in text.splitlines() if "cuteafd serve-dsv4" in l]
    assert launch, text
    for flag in ("--prefix-cache-entries 12", "--host-cache-bytes 32GiB", "--pool-tokens 524288", "--dspark"):
        assert flag in launch[0], flag
    default = _family_launch_lines(tmp_path / "b", config, "deepseek-ai/DeepSeek-V4-Flash-0731", "")
    launch = [l for l in default.splitlines() if "cuteafd serve-dsv4" in l]
    assert "--prefix-cache-entries 20" in launch[0]
    assert "--host-cache-bytes" not in launch[0] and "--pool-tokens" not in launch[0]
