#!/usr/bin/env python3
"""CPU-only regressions for build.sh's Spark image distribution modes.

`rdma` (default) needs rdmapipe on every build host. `ssh-hub` saves the seed's
image once into a task-owned archive on the build host and loads it on the other
hosts over plain ssh, for hub-and-spoke fabrics with no rdmapipe and no direct
spoke-to-spoke route. The region between the release-image-distribution markers is
run in bash against stub `ssh` and `docker`; no Docker, SSH or network is touched.
The stub docker store keeps one image per host: `save` prints its inspect record,
`load` stores whatever arrives and reports a new image ID, as a containerd store
does, so the identity gate has to fall back to portable content.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build.sh"
MARKERS = ("# release-image-distribution:start", "# release-image-distribution:end")
DIE = 2
IMAGE = "ghcr.io/example/cuteafd-spark-expert:v9-pending-1"

SSH_STUB = r"""#!/usr/bin/env bash
# Drop ssh options, then run the remote command as HOST against the stub docker.
while [[ $# -gt 0 ]]; do
  case "$1" in -o|-F|-c) shift 2 ;; -*) shift ;; *) break ;; esac
done
host="$1"; shift
echo "$host $*" >>"$STATE/ssh.log"
HOST="$host" exec bash -c "$*"
"""

DOCKER_STUB = r"""#!/usr/bin/env bash
h="$STATE/hosts/$HOST"
case "$1 $2" in
  "version --format") [[ ! -e "$h/nodocker" ]] ;;
  "image inspect")
    [[ -f "$h/inspect.json" ]] || exit 1
    if [[ "$3" == -f ]]; then cat "$h/id"; else cat "$h/inspect.json"; fi ;;
  "image save")
    [[ ! -e "$h/fail-save" ]] || exit 1
    echo "save $HOST" >>"$STATE/save.log"
    cat "$h/inspect.json" ;;
  "image load")
    mkdir "$STATE/active/$HOST"
    n="$(ls "$STATE/active" | wc -l)"
    (( n <= $(cat "$STATE/max" 2>/dev/null || echo 0) )) || echo "$n" >"$STATE/max"
    sleep 0.3
    rmdir "$STATE/active/$HOST"
    [[ ! -e "$h/fail-load" ]] || { cat >/dev/null; exit 7; }
    mkdir -p "$h"; cat >"$h/inspect.json"
    [[ ! -e "$h/corrupt" ]] || echo '[{"Architecture":"arm64","Os":"linux","RootFS":{"Layers":["sha256:other"]}}]' >"$h/inspect.json"
    echo "sha256:loaded-$HOST" >"$h/id"
    echo "Loaded image: x" ;;
  *) echo "unexpected docker $*" >&2; exit 99 ;;
esac
"""


def region() -> str:
    text = BUILD.read_text(encoding="utf-8")
    start, end = MARKERS
    assert start in text and end in text, "build.sh lost its image distribution markers"
    return text.split(start, 1)[1].split(end, 1)[0]


def seed_record() -> str:
    return json.dumps([{
        "Id": "sha256:seed", "Architecture": "arm64", "Os": "linux",
        "Config": {"Labels": {"org.opencontainers.image.revision": "abc"}},
        "RootFS": {"Type": "layers", "Layers": ["sha256:l1", "sha256:l2"]},
    }])


@pytest.fixture
def env(tmp_path: Path) -> dict:
    state = tmp_path / "state"
    (state / "active").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("ssh", SSH_STUB), ("docker", DOCKER_STUB)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    seed = state / "hosts" / "spark2"
    seed.mkdir(parents=True)
    (seed / "inspect.json").write_text(seed_record())
    (seed / "id").write_text("sha256:seed\n")
    hub = tmp_path / "hub"
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}", "STATE": str(state), "HOME": str(tmp_path),
            "CUTEAFD_RELEASE_IMAGE_TRANSPORT": "ssh-hub", "CUTEAFD_RELEASE_IMAGE_HUB_DIR": str(hub),
            "CUTEAFD_RELEASE_IMAGE_HUB_PARALLEL": "2", "CUTEAFD_RELEASE_IMAGE_HUB_MIN_FREE_GIB": "1"}


HOSTS = ["spark2", "spark1", "spark3", "spark4", "spark5", "spark6"]


def run(env: dict, body: str, hosts=HOSTS) -> subprocess.CompletedProcess:
    script = (
        "set -euo pipefail\nsource scripts/lib/release-common.sh\n"
        f"repo_root={REPO}\nRELEASE_BUILD_HOSTS=({' '.join(hosts)})\nseed_host={hosts[0]}\n"
        f"SPARK_EXPERT_DOCKER_INFERENCE='{IMAGE}'\n" + region() + "\n" + body
    )
    return subprocess.run(["bash", "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)


def host_file(env: dict, host: str, name: str) -> Path:
    return Path(env["STATE"]) / "hosts" / host / name


def test_ssh_hub_saves_once_and_loads_every_peer_bounded(env: dict) -> None:
    result = run(env, "release_image_distribution_preflight\nrelease_distribute_spark_image\necho DONE")
    assert result.returncode == 0, result.stderr
    assert "DONE" in result.stdout
    assert (Path(env["STATE"]) / "save.log").read_text().splitlines() == ["save spark2"]
    for host in HOSTS[1:]:
        assert json.loads(host_file(env, host, "inspect.json").read_text()) == json.loads(seed_record())
    assert int((Path(env["STATE"]) / "max").read_text()) == 2
    assert list(Path(env["CUTEAFD_RELEASE_IMAGE_HUB_DIR"]).iterdir()) == []
    # Every transfer is seed -> hub or hub -> spoke: no command runs a spoke-to-spoke copy.
    assert "rdmapipe" not in (Path(env["STATE"]) / "ssh.log").read_text()


def test_one_failed_load_fails_the_build_and_names_the_host(env: dict) -> None:
    host_file(env, "spark4", "inspect.json").parent.mkdir(parents=True)
    host_file(env, "spark4", "fail-load").touch()
    result = run(env, "release_image_distribution_preflight\nrelease_distribute_spark_image\necho DONE")
    assert result.returncode == DIE
    assert "DONE" not in result.stdout
    assert "load on spark4 failed (exit 7)" in result.stderr
    assert list(Path(env["CUTEAFD_RELEASE_IMAGE_HUB_DIR"]).iterdir()) == []


def test_failed_seed_save_leaves_no_partial_archive(env: dict) -> None:
    host_file(env, "spark2", "fail-save").touch()
    Path(env["CUTEAFD_RELEASE_IMAGE_HUB_DIR"]).mkdir()
    result = run(env, "release_distribute_spark_image\necho DONE")
    assert result.returncode == DIE
    assert "docker image save on spark2" in result.stderr
    assert list(Path(env["CUTEAFD_RELEASE_IMAGE_HUB_DIR"]).iterdir()) == []


def test_loaded_image_with_other_content_is_rejected(env: dict) -> None:
    host_file(env, "spark6", "inspect.json").parent.mkdir(parents=True)
    host_file(env, "spark6", "corrupt").touch()
    result = run(env, "release_image_distribution_preflight\nrelease_distribute_spark_image\necho DONE")
    assert result.returncode == DIE
    assert "spark6 Spark image sha256:loaded-spark6 differs from spark2's sha256:seed" in result.stderr


def test_preflight_fails_before_compile_when_a_host_cannot_run_docker(env: dict) -> None:
    host_file(env, "spark5", "inspect.json").parent.mkdir(parents=True)
    host_file(env, "spark5", "nodocker").touch()
    result = run(env, "release_image_distribution_preflight\necho DONE")
    assert result.returncode == DIE
    assert "spark5 cannot run docker over ssh" in result.stderr


def test_rdma_default_preflight_requires_rdmapipe_and_names_the_alternative(env: dict) -> None:
    env = {k: v for k, v in env.items() if not k.startswith("CUTEAFD_RELEASE_IMAGE")}
    result = run(env, "release_image_distribution_preflight\necho DONE")
    assert result.returncode == DIE
    assert "spark2 lacks it" in result.stderr and "ssh-hub" in result.stderr


@pytest.mark.parametrize(("settings", "message"), [
    ({"CUTEAFD_RELEASE_IMAGE_TRANSPORT": "scp"}, "must be rdma or ssh-hub"),
    ({"CUTEAFD_RELEASE_IMAGE_HUB_DIR": ""}, "needs CUTEAFD_RELEASE_IMAGE_HUB_DIR"),
    ({"CUTEAFD_RELEASE_IMAGE_HUB_DIR": str(REPO / "hub")}, "must not be"),
    ({"CUTEAFD_RELEASE_IMAGE_HUB_DIR": "/tmp/a b"}, "CUTEAFD_RELEASE_IMAGE_HUB_DIR"),
    ({"CUTEAFD_RELEASE_IMAGE_HUB_PARALLEL": "0"}, "HUB_PARALLEL"),
])
def test_bad_settings_are_refused_up_front(env: dict, settings: dict, message: str) -> None:
    result = run({**env, **settings}, "echo DONE")
    assert result.returncode == DIE
    assert message in result.stderr


def test_build_preflights_before_compiling_and_distributes_after_inventory() -> None:
    text = BUILD.read_text(encoding="utf-8")
    preflight = text.index("\nrelease_image_distribution_preflight\n")
    assert preflight < text.index("\nbuild_coordinator_release() (")
    distribute = text.index("\nrelease_distribute_spark_image\n")
    assert text.index('echo "== distributing fresh Spark inference image =="') < distribute
    assert distribute < text.index("docker tag \"$COORDINATOR_DOCKER_INFERENCE\" \"$complete_coordinator_image\"")
    assert text.count("rdmapipe --send") == 1
    assert 'echo "  Spark image distribution: $release_image_transport_plan"' in text
