"""Opt-in cache setup, Docker argument transport, and plain compiler fallback."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/build/compiler-cache.sh"


def run_setup(tmp_path, extra=None, expression='env | sort'):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CUTEAFD_KACHE", "KACHE_")) and key not in
           ("RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CC", "CXX",
            "CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER", "CMAKE_CUDA_COMPILER_LAUNCHER")}
    env.update(extra or {})
    return subprocess.run(["bash", "-c", f'set -eu; source "{HELPER}"; '
                           'cuteafd_compiler_cache_setup "$1"; ' + expression,
                           "test", str(tmp_path / "build")], env=env,
                          text=True, capture_output=True, check=True)


def fake_kache(tmp_path):
    binary = tmp_path / "kache"
    binary.write_text('#!/bin/sh\nif [ "$1" = --version ]; then printf "kache test\\n"; exit 0; fi\nexit 42\n')
    binary.chmod(0o755)
    return str(binary)


def test_unset_is_noop(tmp_path):
    result = run_setup(tmp_path)
    assert result.stderr == ""
    assert "KACHE_" not in result.stdout
    assert "RUSTC_WRAPPER=" not in result.stdout
    assert not (tmp_path / "build").exists()


def test_missing_executable_warns_and_stays_plain(tmp_path):
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": "/nonexistent/kache"})
    assert result.stderr.count("warning:") == 1
    assert "RUSTC_WRAPPER=" not in result.stdout
    assert "CUTEAFD_KACHE_MODE=disabled" in result.stdout


def test_remote_failure_stays_plain(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file")
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache"),
                                "CUTEAFD_KACHE_REMOTE": str(blocker)})
    assert result.stderr.count("warning:") == 1
    assert "RUSTC_WRAPPER=" not in result.stdout
    assert "CUTEAFD_KACHE_MODE=disabled" in result.stdout


def test_existing_wrapper_is_preserved(tmp_path):
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path), "RUSTC_WRAPPER": "/prior/wrapper"})
    assert "RUSTC_WRAPPER=/prior/wrapper" in result.stdout
    assert result.stderr.count("warning:") == 1


def test_native_launchers_and_wrapper_fallback(tmp_path):
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache")},
                       '"$RUSTC_WRAPPER" /usr/bin/printf "%s\\n" native-success; '
                       '"$RUSTC_WRAPPER" /usr/bin/printf "%s\\n" rust-success; env | sort')
    assert "native-success" in result.stdout and "rust-success" in result.stdout
    assert result.stderr.count("warning:") == 1
    assert f"CC={HELPER} cc" in result.stdout
    assert f"CXX={HELPER} c++" in result.stdout
    for language in ("C", "CXX", "CUDA"):
        assert f"CMAKE_{language}_COMPILER_LAUNCHER={HELPER}" in result.stdout
    assert "CUTEAFD_KACHE_MODE=enabled" in result.stdout
    assert not (tmp_path / "cache/target").exists()


def test_local_index_rejects_remote_filesystem(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    findmnt = binary / "findmnt"
    findmnt.write_text('#!/bin/sh\nprintf \'{"filesystems":[{"fstype":"nfs4","options":"rw"}]}\\n\'\n')
    findmnt.chmod(0o755)
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache"),
                                "PATH": str(binary) + os.pathsep + os.environ["PATH"]})
    assert result.stderr.count("warning:") == 1
    assert "RUSTC_WRAPPER=" not in result.stdout
    assert not (tmp_path / "cache").exists()


def test_successful_compile_is_not_followed_by_daemon_shutdown(tmp_path):
    binary = tmp_path / "kache"
    binary.write_text('#!/bin/sh\nif [ "$1" = --version ]; then exit 0; fi\n'
                      'if [ "$1" = daemon ]; then exit 124; fi\nexec "$@"\n')
    binary.chmod(0o755)
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": str(binary),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache")},
                       '"$RUSTC_WRAPPER" /bin/true')
    assert result.stderr == ""
    for name in ("compiler-cache.sh", "build-wip-artifacts.sh", "build-release-artifacts.sh"):
        assert "daemon stop" not in (ROOT / "scripts/build" / name).read_text()
        assert "sync --push" not in (ROOT / "scripts/build" / name).read_text()


def test_compiler_errors_are_not_hidden(tmp_path):
    result = run_setup(tmp_path, {"CUTEAFD_KACHE": fake_kache(tmp_path),
                                "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache")},
                       'if "$RUSTC_WRAPPER" /bin/false; then exit 99; fi')
    assert result.stderr.count("warning:") == 1


def test_cmake_opt_out_clears_persisted_cache_launcher(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    (native / "CMakeCache.txt").write_text(f'CMAKE_C_COMPILER_LAUNCHER:STRING={HELPER}\n')
    result = subprocess.run(["bash", "-c", f'source "{HELPER}"; unset CUTEAFD_KACHE_MODE; '
                             'cuteafd_compiler_cache_cmake_args "$1"', "test", str(native)],
                            text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [f"-DCMAKE_{language}_COMPILER_LAUNCHER="
                                         for language in ("C", "CXX", "CUDA")]


def test_docker_missing_remote_emits_no_partial_mounts(tmp_path):
    env = {**os.environ, "CUTEAFD_KACHE": fake_kache(tmp_path),
           "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache"),
           "CUTEAFD_KACHE_REMOTE": str(tmp_path / "missing-remote")}
    result = subprocess.run(["bash", "-c", f'set -eu; source "{HELPER}"; cuteafd_compiler_cache_docker_args'],
                            env=env, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ["-e", "CUTEAFD_KACHE_REQUESTED=1"]
    assert result.stderr.count("warning:") == 1


def test_cpu_dev_shell_has_no_gpu_or_cache_requirement(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    docker.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUTEAFD_KACHE")}
    env["PATH"] = str(binary) + os.pathsep + env["PATH"]
    result = subprocess.run(["bash", str(ROOT / "scripts/build/cuteafd-dev.sh"), "cpu", "--", "true"],
                            env=env, text=True, capture_output=True, check=True)
    assert "--gpus" not in result.stdout
    assert "/opt/cuteafd-kache" not in result.stdout
    assert result.stdout.splitlines()[-2:] == ["cuteafd-coordinator-dev", "true"]


def test_cpu_dev_shell_mounts_opt_in_target_and_cache(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    docker.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
           "CUTEAFD_KACHE": fake_kache(tmp_path), "CUTEAFD_KACHE_CACHE_DIR": str(tmp_path / "cache"),
           "CUTEAFD_DEV_TARGET_DIR": str(tmp_path / "target")}
    env.pop("CUTEAFD_KACHE_REMOTE", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/build/cuteafd-dev.sh"), "cpu", "--", "true"],
                            env=env, text=True, capture_output=True, check=True)
    assert "--gpus" not in result.stdout
    assert "dst=/opt/cuteafd-target" in result.stdout
    assert "CARGO_TARGET_DIR=/opt/cuteafd-target" in result.stdout
    assert "CUTEAFD_KACHE=/opt/cuteafd-kache" in result.stdout
    assert "cuteafd_compiler_cache_setup" in result.stdout


def test_provenance_records_fallback_and_hash(tmp_path):
    import hashlib
    import json
    (tmp_path / "BUILD_IDENTITY.json").write_text('{"commit":"abc123","dirty":true}')
    binary = tmp_path / "cuteafd"
    binary.write_bytes(b"test binary")
    warning = tmp_path / "warned"
    warning.mkdir()
    destination = tmp_path / "seal.json"
    subprocess.run(["python3", str(ROOT / "scripts/build/write-compiler-provenance.py"),
                    str(tmp_path), str(binary), str(destination)],
                   env={**os.environ, "CUTEAFD_KACHE_MODE": "enabled", "CUTEAFD_KACHE_WARNING_DIR": str(warning)},
                   check=True)
    seal = json.loads(destination.read_text())
    assert seal["source"] == {"commit": "abc123", "dirty": True}
    assert seal["kache_mode"] == "fallback"
    assert seal["binary_sha256"] == hashlib.sha256(b"test binary").hexdigest()
    assert seal["rustc"] and seal["cargo"]


def test_docker_unset_is_noop(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUTEAFD_KACHE")}
    result = subprocess.run(["bash", "-c", f'source "{HELPER}"; cuteafd_compiler_cache_docker_args'],
                            env=env, text=True, capture_output=True, check=True)
    assert result.stdout == result.stderr == ""
