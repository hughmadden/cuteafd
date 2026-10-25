"""Spark legs read the model from the coordinator's namespace hf-home.

The Spark worker container mounts only the namespace root, so both Spark-side
blocks of run-family.sh (the preflight and the worker launch) must resolve the
snapshot under the forwarded namespace hf-home. The Spark host's own HF_HOME or
~/.cache/huggingface must never decide it, for the standard /mnt/sparknest root
as for a custom one, and a path the container cannot see fails before Docker.
"""

import os
import subprocess
from pathlib import Path

import pytest

from test_run_family_dispatch import GLMF_TWO_LAYER, _snapshot

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "scripts/launch/run-family.sh").read_text()
MODEL = "zai-org/GLM-5.3-Flash"
SNAPSHOT_REL = "hub/models--zai-org--GLM-5.3-Flash/snapshots/abc"
STANDARD = "/mnt/sparknest"


def _preflight_block() -> tuple[str, str]:
    invocation, remote = SOURCE.split('spark_manifest="$(release_ssh', 1)[1].split("<<'REMOTE'\n", 1)
    return invocation, remote.split("\nREMOTE", 1)[0]


def _launch_block() -> tuple[str, str]:
    invocation, remote = SOURCE.split('echo "== starting native Spark experts =="', 1)[1].split("<<'REMOTE' &", 1)
    return invocation, remote.split("\nREMOTE", 1)[0]


def _stale_host(tmp_path: Path) -> dict[str, str]:
    """A Spark host whose own HF_HOME and ~/.cache/huggingface hold a valid,
    different copy: the old fallback would have used it."""
    home = tmp_path / "spark-home"
    stale = tmp_path / "stale-hf"
    if not stale.exists():
        _snapshot(home / ".cache/huggingface", MODEL, GLMF_TWO_LAYER)
        _snapshot(stale, MODEL, GLMF_TWO_LAYER)
    return {"HOME": str(home), "HF_HOME": str(stale)}


def _run_preflight(tmp_path: Path, namespace_root: str, namespace_hf_home: str | None,
                   host_env: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    _, remote = _preflight_block()
    calls = tmp_path / "docker.calls"
    stubs = tmp_path / "preflight-bin"
    stubs.mkdir(exist_ok=True)
    # Docker is the first thing after the namespace checks: record it and stop.
    (stubs / "docker").write_text(f'#!/usr/bin/env bash\necho "$*" >> {calls}\nexit 1\n')
    (stubs / "docker").chmod(0o755)
    positional = ["image", "engine", "fork", SNAPSHOT_REL, "spark1", "false", "__none__", "__none__",
                  "__none__", "__none__", "__default__", "__none__", namespace_root]
    if namespace_hf_home is not None:
        positional.append(namespace_hf_home)
    env = {**os.environ, **host_env, "PATH": f"{stubs}:{os.environ['PATH']}"}
    result = subprocess.run(["bash", "-c", remote, "remote", *positional], cwd=ROOT, text=True,
                            capture_output=True, env=env)
    return result, (calls.read_text().splitlines() if calls.exists() else [])


def _run_launch(tmp_path: Path, env_prefix: dict[str, str], host_env: dict[str, str]) -> list[str]:
    _, remote = _launch_block()
    positional = ["image", "worker", "1", "107374182400", "19441", SNAPSHOT_REL, "fingerprint", "6",
                  "0", "", "", "", "", "", "info", "__none__"]
    # readlink -f is stubbed to echo its operand so the chosen path is visible
    # even where the standard root does not exist on the test host.
    harness = ('source scripts/lib/release-common.sh; source scripts/lib/cdi-gpus.sh\n'
               'docker() { printf "%s\\n" "$@"; }\n'
               'readlink() { printf "%s\\n" "${@: -1}"; }\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUTEAFD_NAMESPACE")}
    env.update(host_env)
    env.update(env_prefix)
    result = subprocess.run(["bash", "-euc", harness + remote.replace(" >/dev/null", ""), "remote", *positional],
                            cwd=ROOT, text=True, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_invocations_forward_the_coordinator_namespace_hf_home():
    preflight, _ = _preflight_block()
    launch, _ = _launch_block()
    assert preflight.rstrip().endswith('"$namespace_root" "$namespace_hf_home"')
    assert 'CUTEAFD_NAMESPACE_HF_HOME=$(printf %q "$namespace_hf_home")' in launch
    for _, remote in (_preflight_block(), _launch_block()):
        assert "/mnt/sparknest ]]" not in remote, "no special case for the standard root"
        assert "$HOME/.cache/huggingface" not in remote and "${HF_HOME" not in remote


def test_standard_root_ignores_the_spark_hosts_hf_home(tmp_path):
    # The host's own HF_HOME and ~/.cache/huggingface hold valid snapshots; the
    # launcher must still look under /mnt/sparknest/hf-home and fail there
    # (absent on the test host), before any Docker call.
    for forwarded in (f"{STANDARD}/hf-home", None):
        result, calls = _run_preflight(tmp_path, STANDARD, forwarded, _stale_host(tmp_path))
        assert result.returncode != 0
        assert f"model snapshot is missing: {STANDARD}/hf-home/{SNAPSHOT_REL}" in result.stderr
        assert calls == []


@pytest.mark.parametrize("host_hf_home", ["unset", "stale"])
def test_standard_root_launch_reads_the_namespace_snapshot(tmp_path, host_hf_home):
    host = _stale_host(tmp_path)
    if host_hf_home == "unset":
        del host["HF_HOME"]
    for prefix in ({"CUTEAFD_NAMESPACE_ROOT": STANDARD, "CUTEAFD_NAMESPACE_HF_HOME": f"{STANDARD}/hf-home"},
                   {"CUTEAFD_NAMESPACE_ROOT": STANDARD}):
        args = _run_launch(tmp_path, prefix, host)
        assert args[args.index("--snapshot") + 1] == f"{STANDARD}/hf-home/{SNAPSHOT_REL}"
        assert [args[i + 1] for i, a in enumerate(args) if a == "-v"] == [f"{STANDARD}:{STANDARD}:ro"]


def test_custom_root_preflight_and_launch_use_the_forwarded_hf_home(tmp_path):
    root = tmp_path / "srv-namespace"
    hf = root / "hf-home"
    _snapshot(hf, MODEL, GLMF_TWO_LAYER)
    root, hf = root.resolve(), hf.resolve()
    result, calls = _run_preflight(tmp_path, str(root), str(hf), _stale_host(tmp_path))
    # Every namespace check passed: the block went on to Docker.
    assert "the Docker daemon is unavailable" in result.stderr, result.stderr
    assert calls == ["info"]
    args = _run_launch(tmp_path, {"CUTEAFD_NAMESPACE_ROOT": str(root), "CUTEAFD_NAMESPACE_HF_HOME": str(hf)},
                       _stale_host(tmp_path))
    assert args[args.index("--snapshot") + 1] == f"{hf}/{SNAPSHOT_REL}"
    assert [args[i + 1] for i, a in enumerate(args) if a == "-v"] == [f"{root}:{root}:ro"]


def test_paths_the_container_cannot_see_fail_before_docker(tmp_path):
    root = tmp_path / "ns"
    (root / "hf-home").mkdir(parents=True)
    outside = tmp_path / "outside-hf"
    _snapshot(outside, MODEL, GLMF_TWO_LAYER)
    root, outside = root.resolve(), outside.resolve()
    # A forwarded hf-home outside the mounted root.
    result, calls = _run_preflight(tmp_path, str(root), str(outside), {})
    assert result.returncode != 0 and "outside the mounted namespace" in result.stderr
    assert calls == []
    # A snapshot under the root that resolves outside it (a host-side link).
    (root / "hf-home" / "hub").symlink_to(outside / "hub")
    result, calls = _run_preflight(tmp_path, str(root), str(root / "hf-home"), {})
    assert result.returncode != 0 and "model snapshot resolves outside the mounted namespace" in result.stderr
    assert calls == []
