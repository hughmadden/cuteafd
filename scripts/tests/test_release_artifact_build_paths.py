#!/usr/bin/env python3
"""CPU-only regressions for the release artifact build path handling.

The release artifact compiler runs inside a container that mounts the source
tree read-only (`build.sh`: `-v "$repo_root:/source:ro"`). Two failures are
guarded here; neither needs a GPU, CMake or Cargo:

1. `assert-build-filesystem.py` fails closed on read-only paths, so probing the
   read-only SOURCE_DIR aborted every release build before it started:
   `unsafe build filesystem: build path /source is on a read-only filesystem`.
2. The daemon was built into one cargo target directory and installed from a
   different one, so relocating the target broke the install.

Mount options cannot be faked with chmod, so the read-only cases drive the real
guard with a stubbed `findmnt` (the same technique as
`test_build_filesystem.py`). The last test is behavioural: a stub `cargo` writes
only into the selected target directory and the daemon must still reach the
output directory.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "build" / "build-release-artifacts.sh"
GUARD = REPO / "scripts" / "build" / "assert-build-filesystem.py"


def script_text() -> str:
    return SCRIPT.read_text()


def guard_arguments() -> str:
    """The `assert-build-filesystem.py` argument text, whitespace-normalised."""
    text = script_text()
    match = re.search(
        r'python3 "\$\(dirname "\$0"\)/assert-build-filesystem\.py"'
        r"(?P<args>(?:[^\n]*\\\n)*[^\n]*)",
        text,
    )
    assert match, "guard invocation not found in build-release-artifacts.sh"
    return " ".join(match.group("args").replace("\\\n", " ").split())


def load_guard():
    spec = importlib.util.spec_from_file_location("release_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def findmnt_stub(read_only: set[str]):
    """Report `ro` for the given targets and `rw` for everything else."""

    def run(command, **_kwargs):
        target = command[command.index("--target") + 1]
        options = "ro,relatime" if str(target) in read_only else "rw,relatime"
        payload = json.dumps(
            {"filesystems": [{"fstype": "ext4", "options": options}]}
        )
        return subprocess.CompletedProcess(command, 0, payload, "")

    return run


# --------------------------------------------------------------------------
# Static wiring: the guard must probe the written paths, never the input.
# --------------------------------------------------------------------------


def test_guard_does_not_probe_the_read_only_source():
    assert "$source_dir" not in guard_arguments(), (
        "the guard must not probe SOURCE_DIR: the release container mounts it "
        "read-only and the guard fails closed on read-only paths"
    )


def test_guard_still_probes_the_writable_output_and_target():
    arguments = guard_arguments()
    assert "$output_dir" in arguments
    assert "CARGO_TARGET_DIR" in arguments


def test_cargo_build_and_daemon_install_share_one_target_dir():
    text = script_text()
    assert 'export CARGO_TARGET_DIR="$cargo_target_dir"' in text, (
        "cargo must build into the selected target directory"
    )
    assert 'cuteafd_host_isa_cargo_build "$cargo_target_dir"' in text, (
        "every host-ISA daemon must build beneath the selected target directory"
    )
    assert 'cuteafd_host_isa_install "$output_dir" "$build_root/native" "$cargo_target_dir"' in text, (
        "the daemon install must read from the same selected target directory"
    )
    assert "$build_root/source/rust/target/release/cuteafd" not in text, (
        "no install may hardcode the default target path"
    )


# --------------------------------------------------------------------------
# Guard behaviour under a read-only SOURCE mount (stubbed findmnt).
# --------------------------------------------------------------------------


def test_fixed_guard_arguments_accept_a_read_only_source(tmp_path, monkeypatch):
    guard = load_guard()
    source = tmp_path / "source"
    output = tmp_path / "output"
    target = output / "cargo-target"
    source.mkdir()
    output.mkdir()
    monkeypatch.setattr(guard.subprocess, "run", findmnt_stub({str(source)}))

    # The pre-fix argument set aborted the real build.
    with pytest.raises(ValueError, match="read-only filesystem"):
        for value in (str(source), str(output), str(source / "rust" / "target")):
            guard.check_path(value)

    # The fixed argument set never probes the read-only input.
    for value in (str(output), str(target), str(tmp_path / "cargo-home"), str(tmp_path)):
        guard.check_path(value)


def test_guard_still_rejects_a_read_only_writable_target(tmp_path, monkeypatch):
    """Fail-closed is preserved for the paths that really are written."""
    guard = load_guard()
    target = tmp_path / "ro-target"
    target.mkdir()
    monkeypatch.setattr(guard.subprocess, "run", findmnt_stub({str(target)}))
    with pytest.raises(ValueError, match="read-only filesystem"):
        guard.check_path(str(target))


# --------------------------------------------------------------------------
# Behavioural: a relocated cargo target still yields an installed daemon.
# --------------------------------------------------------------------------


def _write_stub(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body))
    path.chmod(0o755)


def _fake_source(root: Path) -> Path:
    """Minimal tree satisfying the script's structural checks."""
    src = root / "source"
    for relative in (
        "rust/Cargo.toml",
        "native/CMakeLists.txt",
        "THIRD_PARTY_NOTICES.md",
        "third_party/sparkinfer/LICENSE",
        "third_party/sparkinfer.lock.json",
        "third_party/xgrammar/LICENSE",
        "third_party/xgrammar.lock.json",
    ):
        target = src / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stub\n")
    (src / "scripts" / "build").mkdir(parents=True, exist_ok=True)
    (src / "scripts" / "build" / "assert-build-filesystem.py").write_bytes(GUARD.read_bytes())
    return src


@pytest.fixture
def scratch():
    """A probeable working directory for a real build-filesystem guard run.

    The guard resolves each path's mount through findmnt, and a sandboxed or
    private /tmp has no resolvable filesystem entry, so pytest's default temporary
    directory makes the real `assert-build-filesystem.py` call fail closed before
    the behaviour under test is reached. This checkout's cache directory is on the
    root filesystem and is ignored by the release source inventory.
    """
    root = REPO / ".cuteafd-cache" / "test-release-artifact-build-paths"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _stub_build(tmp_path: Path):
    """Fake source, stub cargo/cmake/python3 (each logging its call) and env."""
    src = _fake_source(tmp_path)
    output = tmp_path / "output"
    relocated = tmp_path / "relocated-target"
    shims = tmp_path / "shims"
    shims.mkdir()

    _write_stub(
        shims / "cargo",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        echo cargo >>"$STUB_CALL_LOG"
        [[ "${STUB_CARGO_FAIL:-0}" == 0 ]] || exit 9
        triple=""
        while [[ $# -gt 0 ]]; do [[ "$1" == --target ]] && triple="$2/"; shift; done
        mkdir -p "$CARGO_TARGET_DIR/${triple}release"
        printf 'daemon\\n' >"$CARGO_TARGET_DIR/${triple}release/cuteafd"
        chmod +x "$CARGO_TARGET_DIR/${triple}release/cuteafd"
        """,
    )
    _write_stub(
        shims / "cmake",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        echo cmake >>"$STUB_CALL_LOG"
        build_dir=""
        while [[ $# -gt 0 ]]; do
          case "$1" in
            -B) build_dir="$2"; shift 2 ;;
            *) shift ;;
          esac
        done
        [[ -n "$build_dir" ]] || exit 0
        for expert_dir in v41_local_experts v41_tp2_experts v41_nvfp4_rtx_backbone v41_nvfp4_rtx_tp2; do
          mkdir -p "$build_dir/$expert_dir"
          printf '%s\\n' '{}' >"$build_dir/$expert_dir/v41_experts.json"
          cp "$build_dir/$expert_dir/v41_experts.json" "$build_dir/$expert_dir/v41_nvfp4_experts.json"
        done
        mkdir -p "$build_dir/v41_experts" "$build_dir/v41_fp8" "$build_dir/x86-64-v4"
        printf 'native\\n' >"$build_dir/libcuteafd_native.so"
        printf 'native v4\\n' >"$build_dir/x86-64-v4/libcuteafd_native.so"
        printf '{}\\n' >"$build_dir/v41_experts/v41_experts.json"
        printf '{}\\n' >"$build_dir/v41_fp8/v41_fp8.json"
        exit 0
        """,
    )
    _write_stub(
        shims / "python3",
        f"""
        #!/usr/bin/env bash
        set -euo pipefail
        for arg in "$@"; do
          if [[ "$arg" == *assert-build-filesystem.py ]]; then
            exec {sys.executable} "$@"
          fi
        done
        previous=""
        for arg in "$@"; do
          case "$previous" in
            --output|--write)
              mkdir -p "$(dirname "$arg")"
              printf '{{}}\\n' >"$arg"
              ;;
          esac
          previous="$arg"
        done
        exit 0
        """,
    )

    env = dict(os.environ)
    env["PATH"] = f"{shims}:{env['PATH']}"
    env["CARGO_TARGET_DIR"] = str(relocated)
    # This stub proves relocation, not compiler-cache target suffixing.
    env["CUTEAFD_BUILD_CACHES"] = "off"
    env.pop("CUTEAFD_KACHE", None)
    env.pop("CUTEAFD_SCCACHE_CUDA", None)
    env.pop("CUTEAFD_RELEASE_SPARK_TP_ROLES", None)
    # The build root parent is relocated to the same probeable filesystem: with the
    # hook unset the script guards its literal /tmp default, which a private or
    # sandboxed /tmp cannot resolve, and that would mask the target-directory
    # behaviour this test exists to prove.
    build_root = tmp_path / "build-root"
    build_root.mkdir(parents=True, exist_ok=True)
    env["CUTEAFD_RELEASE_BUILD_ROOT"] = str(build_root)
    # The script guards CARGO_HOME too, and a user's default may sit on a
    # read-only mount in a sandboxed runner.
    env["CARGO_HOME"] = str(tmp_path / "cargo-home")
    env["STUB_CALL_LOG"] = str(tmp_path / "calls.log")
    for name in ("CUTEAFD_RELEASE_ARTIFACT_PHASE", "CUTEAFD_RELEASE_ARTIFACT_STATE"):
        env.pop(name, None)
    return src, output, relocated, env


def test_relocated_cargo_target_is_where_the_daemon_is_installed(scratch):
    """Behavioural proof that build and install follow one selected target."""
    src, output, relocated, env = _stub_build(scratch)
    result = subprocess.run(
        ["bash", str(SCRIPT), str(src), "coordinator", "120", str(output)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # The stub only ever wrote into the relocated target, one directory per host ISA...
    triple = "x86_64-unknown-linux-gnu"
    assert (relocated.parent / f"{relocated.name}-x86-64-v2" / triple / "release" / "cuteafd").exists()
    assert (relocated.parent / f"{relocated.name}-x86-64-v4" / triple / "release" / "cuteafd").exists()
    # ...and the daemons still reached the output, because build and install
    # read the same selected target rather than a hardcoded default.
    assert (output / "cuteafd").read_text() == "daemon\n"
    assert (output / "bin-variants" / "x86-64-v4" / "cuteafd").read_text() == "daemon\n"
    assert (output / "lib-variants" / "x86-64-v4" / "libcuteafd_native.so").read_text() == "native v4\n"
    assert not (output / "cargo-target").exists()


# --------------------------------------------------------------------------
# Phases: Rust without a GPU lease, then native in the same state.
# --------------------------------------------------------------------------


def _phase(src, output, env, phase, state, **extra):
    run_env = {**env, "CUTEAFD_RELEASE_ARTIFACT_PHASE": phase,
               "CUTEAFD_RELEASE_ARTIFACT_STATE": str(state), **extra}
    return subprocess.run(["bash", str(SCRIPT), str(src), "coordinator", "120", str(output)],
                          capture_output=True, text=True, env=run_env)


def test_rust_then_native_phase_builds_once_and_consumes_the_state(scratch):
    src, output, _, env = _stub_build(scratch)
    calls = Path(env["STUB_CALL_LOG"])
    state = scratch / "state"
    rust = _phase(src, output, env, "rust", state)
    assert rust.returncode == 0, rust.stdout + rust.stderr
    assert calls.read_text().split() == ["cargo", "cargo"]  # both host ISAs, no CMake
    assert (state / "rust.done").is_file() and not output.exists()
    # The GPU assignment is the one input allowed to change between phases.
    native = _phase(src, output, env, "native", state, CUDA_VISIBLE_DEVICES="GPU-x", NVIDIA_VISIBLE_DEVICES="GPU-x")
    assert native.returncode == 0, native.stdout + native.stderr
    assert calls.read_text().split() == ["cargo", "cargo", "cmake", "cmake"]  # no repeated Cargo
    assert (output / "cuteafd").read_text() == "daemon\n"
    assert not state.exists()


def test_native_phase_refuses_missing_rust_state(scratch):
    src, output, _, env = _stub_build(scratch)
    result = _phase(src, output, env, "native", scratch / "state")
    assert result.returncode == 2
    assert "no completed rust phase" in result.stderr
    assert not Path(env["STUB_CALL_LOG"]).exists()


@pytest.mark.parametrize("change", ["env", "source", "daemon"])
def test_native_phase_refuses_stale_rust_state(scratch, change):
    src, output, relocated, env = _stub_build(scratch)
    state = scratch / "state"
    assert _phase(src, output, env, "rust", state).returncode == 0
    extra = {}
    if change == "env":
        extra["CUTEAFD_RELEASE_AUDIO_AOT"] = "OFF"
    elif change == "source":
        (src / "THIRD_PARTY_NOTICES.md").write_text("changed\n")
    else:
        (relocated.parent / f"{relocated.name}-x86-64-v4" / "x86_64-unknown-linux-gnu" / "release" / "cuteafd").write_text("other\n")
    result = _phase(src, output, env, "native", state, **extra)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "changed" in result.stderr
    assert "cmake" not in Path(env["STUB_CALL_LOG"]).read_text()
    assert not state.exists()


def test_failed_rust_phase_removes_its_state(scratch):
    src, output, _, env = _stub_build(scratch)
    state = scratch / "state"
    result = _phase(src, output, env, "rust", state, STUB_CARGO_FAIL="1")
    assert result.returncode != 0
    assert not state.exists()
    assert _phase(src, output, env, "rust", state).returncode == 0
    assert _phase(src, output, env, "rust", state).returncode == 2  # a state is never reused


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
