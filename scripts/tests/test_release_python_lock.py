#!/usr/bin/env python3
"""CPU-only checks for the release images' frozen Python dependencies.

docker/release-python-lock.json is the canonical input; scripts/build/release-python-lock.py
verifies a wheelhouse against it and writes the hash-pinned requirement lines the
release Dockerfile installs offline. Everything here uses synthetic wheels in a
temporary directory or reads committed files: no network, Docker or package install.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "build" / "release-python-lock.py"
LOCK = REPO / "docker" / "release-python-lock.json"
RELEASE = REPO / "docker" / "Dockerfile.release"
BUILD = REPO / "build.sh"
DIE = 2

spec = importlib.util.spec_from_file_location("release_python_lock", TOOL)
lockmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lockmod)


def run_tool(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(TOOL), *map(str, args)],
                          capture_output=True, text=True, timeout=60)


def make_lock(tmp_path: Path, payloads: dict[str, bytes]) -> tuple[Path, Path]:
    """A lock whose packages are pure wheels with these payloads, and a wheelhouse holding them."""
    house = tmp_path / "wheels"
    house.mkdir()
    packages = []
    for name, payload in payloads.items():
        filename = f"{name}-1.0-py3-none-any.whl"
        (house / filename).write_bytes(payload)
        wheel = {"filename": filename, "sha256": hashlib.sha256(payload).hexdigest(),
                 "url": f"https://files.pythonhosted.org/packages/aa/bb/{filename}",
                 "upload_time": "2026-01-01T00:00:00Z"}
        packages.append({"name": name, "version": "1.0", "requires": [],
                         "wheels": {"amd64": wheel, "arm64": wheel}})
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"schema": lockmod.SCHEMA, "base_image": "base@sha256:" + "0" * 64,
                                "python": "3.12", "arches": ["amd64", "arm64"], "min_age_days": 7,
                                "cutoff": "2026-02-01T00:00:00Z", "packages": packages}))
    return lock, house


def test_requirements_are_hash_pinned_local_files(tmp_path: Path) -> None:
    lock, house = make_lock(tmp_path, {"uv": b"u", "alpha": b"a"})
    result = run_tool("requirements", "--lock", lock, "--wheelhouse", house, "--arch", "arm64", "--exclude", "uv")
    assert result.returncode == 0, result.stderr
    digest = hashlib.sha256(b"a").hexdigest()
    assert result.stdout.splitlines() == [
        f"alpha @ file://{(house / 'alpha-1.0-py3-none-any.whl').resolve()} --hash=sha256:{digest}"]


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_missing_or_tampered_wheels_fail_verification(tmp_path: Path, damage: str) -> None:
    lock, house = make_lock(tmp_path, {"alpha": b"a", "beta": b"b"})
    wheel = house / "beta-1.0-py3-none-any.whl"
    wheel.unlink() if damage == "missing" else wheel.write_bytes(b"evil")
    for args in (("verify",), ("requirements", "--arch", "amd64")):
        result = run_tool(*args, "--lock", lock, "--wheelhouse", house)
        assert result.returncode == 1
        assert "beta-1.0-py3-none-any.whl" in result.stderr


def test_fetch_never_replaces_a_present_file_with_another_hash(tmp_path: Path) -> None:
    lock, house = make_lock(tmp_path, {"alpha": b"a"})
    (house / "alpha-1.0-py3-none-any.whl").write_bytes(b"evil")
    result = run_tool("fetch", "--lock", lock, "--wheelhouse", house)
    assert result.returncode == 1 and "the lock says" in result.stderr
    assert (house / "alpha-1.0-py3-none-any.whl").read_bytes() == b"evil"


@pytest.mark.parametrize(("field", "value", "message"), [
    ("sha256", "abc", "no sha256"),
    ("url", "https://example.com/alpha-1.0-py3-none-any.whl", "files.pythonhosted.org"),
    ("upload_time", "2026-03-01T00:00:00Z", "after the lock cutoff"),
    ("filename", "alpha-2.0-py3-none-any.whl", "does not fit"),
    ("filename", "alpha-1.0-cp312-cp312-musllinux_1_2_x86_64.whl", "does not fit"),
])
def test_malformed_locks_are_refused(tmp_path: Path, field: str, value: str, message: str) -> None:
    lock, house = make_lock(tmp_path, {"alpha": b"a"})
    data = json.loads(lock.read_text())
    data["packages"][0]["wheels"]["amd64"][field] = value
    lock.write_text(json.dumps(data))
    result = run_tool("verify", "--lock", lock, "--wheelhouse", house)
    assert result.returncode == 1 and message in result.stderr


@pytest.mark.parametrize(("filename", "arch", "usable"), [
    ("x-1-py3-none-any.whl", "amd64", True),
    ("x-1-cp38-abi3-manylinux_2_28_aarch64.whl", "arm64", True),
    ("x-1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", "amd64", True),
    ("x-1-cp312-cp312-manylinux_2_17_x86_64.whl", "arm64", False),
    ("x-1-cp311-cp311-manylinux_2_17_x86_64.whl", "amd64", False),
    ("x-1-cp312-cp312-manylinux_2_41_x86_64.whl", "amd64", False),
    ("x-1-cp312-cp312-musllinux_1_2_aarch64.whl", "arm64", False),
])
def test_wheel_tags_must_fit_the_cpython312_linux_base(filename: str, arch: str, usable: bool) -> None:
    assert (lockmod.wheel_rank(filename, arch) is not None) == usable


def test_closure_check_reports_missing_and_conflicting_requirements(tmp_path: Path) -> None:
    pytest.importorskip("packaging")
    lock, _ = make_lock(tmp_path, {"alpha": b"a"})
    data = json.loads(lock.read_text())
    data["packages"][0]["requires"] = ["beta>=2", "gamma; sys_platform == 'win32'", "delta; extra == 'x'"]
    inventory = {"beta": {"version": "1.5", "installer": "pip", "requires": []},
                 "omega": {"version": "1", "installer": "pip", "requires": ["alpha<1"]}}
    problems, replaced = lockmod.closure_problems(data, inventory, "amd64")
    assert problems == ["alpha 1.0 needs beta>=2: have 1.5", "omega 1 needs alpha<1: have 1.0"]
    assert replaced == {}


# The committed lock.

def committed() -> dict:
    return lockmod.load_lock(LOCK)


def test_committed_lock_is_valid_for_both_architectures() -> None:
    lock = committed()
    assert lock["arches"] == ["amd64", "arm64"]
    assert lock["min_age_days"] == 7
    names = {lockmod.normalize(p["name"]) for p in lock["packages"]}
    # Every package the release image used to install from an index is locked.
    for name in ("uv", "safetensors", "huggingface-hub", "nvidia-cutlass-dsl", "apache-tvm-ffi", "cuda-tile",
                 "nvidia-cudnn-frontend", "nvidia-ml-py", "transformers", "flashinfer-python", "pytest",
                 "python-dotenv"):
        assert name in names, name


def test_committed_lock_keeps_the_huggingface_hub_floor() -> None:
    floor = re.search(r'"huggingface-hub>=([0-9.]+)"', (REPO / "python" / "pyproject.toml").read_text())[1]
    locked = next(p["version"] for p in committed()["packages"] if p["name"] == "huggingface-hub")
    as_tuple = lambda v: tuple(int(x) for x in v.split("."))
    assert as_tuple(locked) >= as_tuple(floor)


def test_release_and_dev_images_share_the_locked_base_digest() -> None:
    base = committed()["base_image"]
    assert "@sha256:" in base
    for dockerfile in (RELEASE, REPO / "docker" / "Dockerfile.dev"):
        assert f"ARG BASE_IMAGE={base}\n" in dockerfile.read_text(), dockerfile


def test_release_dockerfile_installs_only_offline_from_the_lock() -> None:
    text = RELEASE.read_text()
    assert "type=cache" not in text
    installs = re.findall(r"(?:uv pip install|pip install)[^;&]*", text)
    assert installs, "the release Dockerfile lost its installs"
    for install in installs:
        assert "--no-index" in install and "--no-deps" in install, install
    runs = [run for run in text.split("\nRUN ")[1:] if "pip install" in run]
    assert runs and all(run.startswith("--network=none") for run in runs)
    assert "--require-hashes" in text and "uv pip check" in text
    assert "--mount=type=bind,from=wheelhouse,target=/wheelhouse" in text
    # tool-eval-bench runs from its verified source; nothing builds it with setuptools-scm.
    assert "SETUPTOOLS_SCM_PRETEND_VERSION" not in text


def test_tool_eval_wrapper_runs_the_entry_point_wip_slots_use() -> None:
    wrapper = (REPO / "docker" / "tool-eval-bench-wrapper.sh").read_text()
    common = (REPO / "scripts" / "lib" / "release-common.sh").read_text()
    entry = "from tool_eval_bench.cli.dispatch import main; sys.exit(main())"
    assert entry in wrapper and entry in common
    assert os.access(REPO / "docker" / "tool-eval-bench-wrapper.sh", os.X_OK)


def test_build_verifies_the_wheelhouse_before_compiling_and_passes_it_to_both_images() -> None:
    text = BUILD.read_text()
    preflight = text.index("\nrelease_verify_wheelhouse\n")
    assert preflight < text.index("\nbuild_coordinator_release() (")
    assert text.count('--build-context "wheelhouse=$release_wheelhouse"') == 1
    assert text.count("--build-context wheelhouse=.cuteafd-release/wheelhouse") == 1
    assert "--wheelhouse $release_remote_wheelhouse --arch arm64" in text


def region() -> str:
    text = BUILD.read_text()
    return text.split("# release-python-lock:start", 1)[1].split("# release-python-lock:end", 1)[0]


@pytest.mark.parametrize("state", ["unset", "empty", "complete"])
def test_build_wheelhouse_gate(tmp_path: Path, state: str) -> None:
    house = tmp_path / "house"
    house.mkdir()
    env = {**os.environ}
    env.pop("CUTEAFD_RELEASE_WHEELHOUSE", None)
    lock = LOCK
    if state != "unset":
        env["CUTEAFD_RELEASE_WHEELHOUSE"] = str(house)
    if state == "complete":
        lock, house_full = make_lock(tmp_path, {"alpha": b"a"})
        env["CUTEAFD_RELEASE_WHEELHOUSE"] = str(house_full)
    script = (f"set -euo pipefail\nsource scripts/lib/release-common.sh\nrepo_root={REPO}\n" + region()
              + f"\nrelease_python_lock={lock}\nrelease_verify_wheelhouse\necho PASSED\n")
    result = subprocess.run(["bash", "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    if state == "complete":
        assert result.returncode == 0 and "PASSED" in result.stdout, result.stderr
    else:
        assert result.returncode == DIE and "PASSED" not in result.stdout
        assert ("CUTEAFD_RELEASE_WHEELHOUSE must name" if state == "unset" else "does not hold") in result.stderr
