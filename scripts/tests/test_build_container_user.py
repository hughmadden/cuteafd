"""Non-root build containers: the `--user`/env group and its call sites.

A build container that compiles into a bind-mounted host directory used to run
as root, leaving root-owned Cargo target dirs and staging behind. It now runs as
the invoking user with the env a non-passwd UID needs. Everything here is
CPU-only: the helper block is extracted from build.sh by its markers and run in
bash, and the dev-shell entry point is executed against a logging docker stub on
PATH. No Docker, image or GPU is contacted.
"""
from __future__ import annotations

import os
from pathlib import Path
import pwd
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build.sh"
DEV_SHELL = REPO / "scripts" / "build" / "cuteafd-dev.sh"
NVFP4_VERIFIER = REPO / "scripts" / "build" / "verify-nvfp4-modelopt-container.sh"
ARTIFACTS = REPO / "scripts" / "build" / "build-release-artifacts.sh"
START = "# release-build-container-user:start"
END = "# release-build-container-user:end"
SENTINEL = "__ARGS__"

DOCKER_STUB = """#!/usr/bin/env bash
set -euo pipefail
for token in "$@"; do printf 'A\\t%s\\n' "$token" >> "${MOCK_DOCKER_LOG:?}"; done
printf 'E\\n' >> "$MOCK_DOCKER_LOG"
exit 0
"""


def block() -> str:
    text = BUILD.read_text(encoding="utf-8")
    for marker in (START, END):
        assert marker in text, f"build.sh lost its {marker!r} marker"
    return text.split(START, 1)[1].split(END, 1)[0]


def run_helper(build_root: str) -> tuple[list[str], str]:
    body = (
        "set -euo pipefail\n"
        + block()
        + "\nargs=()\n"
        + f"mapfile -t args < <(release_build_container_user_args_render {build_root!r})\n"
        + 'printf \'%s\\n\' "${args[@]}"\n'
        + f"printf '{SENTINEL}\\n'\n"
        + f"printf 'home=%s\\n' \"$(release_build_container_home {build_root!r})\"\n"
    )
    result = subprocess.run(
        ["bash", "-c", body], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    args_text, rest = result.stdout.split(SENTINEL + "\n", 1)
    return args_text.splitlines(), rest.strip()


@pytest.mark.parametrize(
    "build_root,expected_home",
    [
        ("", "/tmp/cuteafd-home"),
        ("/scratch/cuteafd-build.task-1", "/scratch/cuteafd-build.task-1/container-home"),
    ],
)
def test_user_group_matches_the_invoking_user(build_root, expected_home):
    """UID/GID come from the docker host, and every name env is explicit: a
    --user UID has no passwd entry for getpass or Torch Dynamo to resolve."""
    args, rest = run_helper(build_root)
    user = pwd.getpwuid(os.getuid()).pw_name
    assert args == [
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"HOME={expected_home}",
        "-e", f"USER={user}",
        "-e", f"LOGNAME={user}",
        "-e", f"TORCHINDUCTOR_CACHE_DIR={expected_home}/torchinductor",
        "-e", f"CARGO_HOME={expected_home}/cargo",
    ]
    assert rest == f"home={expected_home}"


def test_the_relocated_cargo_home_matches_the_unwritable_image_default():
    """The image's CARGO_HOME is root-owned, so a --user container must not
    inherit it: the relocation is the reason, not a preference."""
    assert "CARGO_HOME=/opt/cargo" in (REPO / "docker" / "Dockerfile.dev").read_text(encoding="utf-8")


def test_both_release_legs_and_the_artifact_compiler_carry_the_group():
    text = BUILD.read_text(encoding="utf-8")
    # The coordinator leg uses the rendered array exactly once...
    assert text.count('"${release_build_user_args[@]}" \\') == 1
    assert 'release_build_container_user_args_render "$release_build_root"' in text
    # ...and the Spark leg computes the group inside the remote heredoc, where
    # `id` reports the Spark's own user (raptor is 1000, the Sparks 1001).
    remote = text.split(
        'echo "== building Spark development and inference images natively on $seed_host =="', 1
    )[1].split("<<'REMOTE'", 1)[1].split("\nREMOTE", 1)[0]
    for token in ('--user "$(id -u):$(id -g)"', '-e "HOME=$container_home"',
                  '-e "USER=$(id -un)"', '-e "LOGNAME=$(id -un)"',
                  '-e "TORCHINDUCTOR_CACHE_DIR=$container_home/torchinductor"',
                  '-e "CARGO_HOME=$container_home/cargo"'):
        assert token in remote, token
    assert remote.count('--user "$(id -u):$(id -g)"') == 1
    assert 'container_home="$release_build_root/container-home"' in remote
    # The container, not the caller, creates the writable roots: the bind-mounted
    # home does not exist on the host before the container first writes it.
    artifacts = ARTIFACTS.read_text(encoding="utf-8")
    for name in ("${HOME:-}", "${CARGO_HOME:-}", "${TORCHINDUCTOR_CACHE_DIR:-}"):
        assert name in artifacts, name
    assert 'mkdir -p "$writable_root"' in artifacts


def test_dev_shell_execs_docker_with_the_group(tmp_path):
    """Execute the real entry point: the argv is what docker actually receives."""
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "docker.log"
    stub = binary / "docker"
    stub.write_text(DOCKER_STUB)
    stub.chmod(0o755)
    # The optional compiler cache rewraps argv; this test pins the plain launch.
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUTEAFD_KACHE")}
    env.update(
        PATH=f"{binary}{os.pathsep}{os.environ['PATH']}",
        HOME=str(tmp_path / "home"),
        MOCK_DOCKER_LOG=str(log),
        HF_HOME=str(tmp_path / "hf"),
    )
    result = subprocess.run(
        ["bash", str(DEV_SHELL), "expert", "--", "true"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    tokens: list[str] = []
    for line in log.read_text().splitlines():
        if line.startswith("A\t"):
            tokens.append(line[2:])
    user = pwd.getpwuid(os.getuid()).pw_name
    for pair in (
        ["--user", f"{os.getuid()}:{os.getgid()}"],
        ["-e", f"USER={user}"],
        ["-e", f"LOGNAME={user}"],
        ["-e", "HOME=/tmp/cuteafd-dev-home"],
        ["-e", "TORCHINDUCTOR_CACHE_DIR=/tmp/cuteafd-dev-home/torchinductor"],
        ["-e", "CARGO_HOME=/tmp/cuteafd-dev-home/cargo"],
    ):
        assert any(tokens[i:i + 2] == pair for i in range(len(tokens) - 1)), pair
    assert tokens[0] == "run"
    assert tokens[-1] == "true"
    assert tokens[-2] == "cuteafd-spark-expert-dev"


def test_nvfp4_verifier_writes_the_reference_as_the_calling_user():
    text = NVFP4_VERIFIER.read_text(encoding="utf-8")
    assert '--user "$(id -u):$(id -g)"' in text
    for key in ("HOME=", "USER=", "LOGNAME=", "TORCHINDUCTOR_CACHE_DIR=", "CARGO_HOME="):
        assert f'-e "{key}$container_home' in text or f'-e "{key}$(id -un)"' in text, key
