"""INSTANCE namespaces the coordinator container name.

Two agents' cleanups could kill each other's server because run.sh and stop.sh
always used the single name `cuteafd-coordinator`. The name is now derived in
`release_load_config` from INSTANCE, with the same acceptance rule and shape
scripts/launch/run-family.sh uses, and the unset default stays exactly
`cuteafd-coordinator`.

CPU-only: the derivation is executed in bash against the real library, and
stop.sh runs against logging docker/ssh/ss/ps stubs on PATH with a docker daemon
that reports the coordinator container as present and running. No container,
host or process is contacted.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "scripts" / "lib" / "release-common.sh"
STOP = ROOT / "stop.sh"
CANDIDATE = ROOT / "scripts" / "launch" / "run-tp-ep-native-candidate.sh"

DOCKER_STUB = """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${MOCK_DOCKER_LOG:?}"
case "${1:-}" in
  info) exit 0 ;;
  container) exit 0 ;;                 # the coordinator container exists
  inspect) printf 'true\\n'; exit 0 ;; # ...and is running
  *) exit 0 ;;
esac
"""

SSH_STUB = """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${MOCK_SSH_LOG:?}"
exit 0
"""

NOOP_STUB = "#!/usr/bin/env bash\nexit 0\n"


def write_config(path: Path, instance: str | None) -> Path:
    text = (ROOT / "cuteafd.config").read_text(encoding="utf-8")
    if instance is not None:
        text += f"\nINSTANCE={instance}\n"
    path.write_text(text, encoding="utf-8")
    return path


def load_name(config: Path, mode: str = "launch") -> subprocess.CompletedProcess:
    script = (
        "set -euo pipefail\n"
        f'source "{COMMON}"\n'
        f'release_load_config "{config}" {mode}\n'
        'printf "NAME=%s\\n" "$RELEASE_COORDINATOR_CONTAINER_NAME"\n'
        "release_load_config "
        f'"{config}" {mode}\n'
        'printf "AGAIN=%s\\n" "$RELEASE_COORDINATOR_CONTAINER_NAME"\n'
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          timeout=60, check=False)


@pytest.mark.parametrize(
    "instance,expected",
    [(None, "cuteafd-coordinator"),
     ("", "cuteafd-coordinator"),
     ("own", "cuteafd-coordinator-own"),
     ("agent.2_ab-cd", "cuteafd-coordinator-agent.2_ab-cd")],
)
def test_instance_suffixes_the_coordinator_and_repeats_stably(tmp_path, instance, expected):
    result = load_name(write_config(tmp_path / "cuteafd.config", instance))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f"NAME={expected}", f"AGAIN={expected}"]


@pytest.mark.parametrize(
    "instance,message",
    [("-leading", "INSTANCE must be [A-Za-z0-9_.-]"),
     (".dot", "INSTANCE must be [A-Za-z0-9_.-]"),
     ("a/b", "INSTANCE must be [A-Za-z0-9_.-]"),
     # Whitespace fails the parser's quoting rule first, which is strictly safer.
     ("a b", "unquoted whitespace is not allowed for INSTANCE"),
     ("own$", "INSTANCE must be [A-Za-z0-9_.-]"),
     ("x" * 42, "INSTANCE must be [A-Za-z0-9_.-]")],
)
def test_invalid_instance_is_refused_before_any_side_effect(tmp_path, instance, message):
    result = load_name(write_config(tmp_path / "cuteafd.config", instance))
    assert result.returncode == 2
    assert message in result.stderr
    assert "NAME=" not in result.stdout


class StopHarness:
    def __init__(self, tmp_path: Path):
        binary = tmp_path / "bin"
        binary.mkdir()
        for name, body in (("docker", DOCKER_STUB), ("ssh", SSH_STUB),
                           ("ss", NOOP_STUB), ("ps", NOOP_STUB)):
            stub = binary / name
            stub.write_text(body, encoding="utf-8")
            stub.chmod(0o755)
        self.docker_log = tmp_path / "docker.log"
        self.ssh_log = tmp_path / "ssh.log"
        self.env = dict(
            os.environ,
            PATH=f"{binary}{os.pathsep}{os.environ['PATH']}",
            MOCK_DOCKER_LOG=str(self.docker_log),
            MOCK_SSH_LOG=str(self.ssh_log),
        )

    def run_stop(self, config: Path) -> list[str]:
        self.docker_log.write_text("")
        result = subprocess.run(["bash", str(STOP), "--config", str(config)],
                                cwd=ROOT, env=self.env, capture_output=True,
                                text=True, timeout=120, check=False)
        assert result.returncode == 0, result.stderr
        return self.docker_log.read_text().splitlines()


@pytest.mark.parametrize("instance,expected", [(None, "cuteafd-coordinator"),
                                               ("own", "cuteafd-coordinator-own")])
def test_stop_targets_only_the_instances_coordinator(tmp_path, instance, expected):
    harness = StopHarness(tmp_path)
    lines = harness.run_stop(write_config(tmp_path / "cuteafd.config", instance))
    assert f"stop -t 30 {expected}" in lines
    assert f"rm -f {expected}" in lines
    other = "cuteafd-coordinator" if expected != "cuteafd-coordinator" else "cuteafd-coordinator-own"
    # An exact line match: `cuteafd-coordinator-own` must not satisfy, or be
    # satisfied by, the shared name.
    assert f"stop -t 30 {other}" not in lines
    assert f"rm -f {other}" not in lines


def test_candidate_launcher_refuses_every_namespaced_production_name():
    """The candidate launcher may not start or stop any production name, and a
    namespaced launch is a production name: `cuteafd-coordinator-*` joins the
    list, so an INSTANCE launcher cannot be hijacked either."""
    text = CANDIDATE.read_text(encoding="utf-8")
    line = next(
        line for line in text.splitlines()
        if line.strip().startswith("cuteafd-coordinator|") and "spark-expert" in line
    )
    pattern = line.strip().rstrip(")")
    assert "cuteafd-coordinator-*" in pattern
    harness = (
        "set -euo pipefail\n"
        "for name in \"$@\"; do\n"
        "  case \"$name\" in\n"
        f"    {pattern}) printf 'refused %s\\n' \"$name\" ;;\n"
        "    *) printf 'allowed %s\\n' \"$name\" ;;\n"
        "  esac\n"
        "done\n"
    )
    result = subprocess.run(
        ["bash", "-c", harness, "_", "cuteafd-coordinator", "cuteafd-coordinator-wip",
         "cuteafd-coordinator-own", "cuteafd-spark-expert",
         "cuteafd-spark-expert-ostrich-19441", "cuteafd-tpep-dev"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "refused cuteafd-coordinator",
        "refused cuteafd-coordinator-wip",
        "refused cuteafd-coordinator-own",
        "refused cuteafd-spark-expert",
        "refused cuteafd-spark-expert-ostrich-19441",
        "allowed cuteafd-tpep-dev",
    ]
