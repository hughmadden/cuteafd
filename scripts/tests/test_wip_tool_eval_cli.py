"""The WIP coordinator layout exposes the slot's lock-verified tool-eval-bench CLI.

The tool_eval panel runs `tool-eval-bench --help` and then the benchmark from PATH.
Release images install the pinned submodule; WIP layouts stage it from the sealed
slot's source, verified against its lock, behind bin/tool-eval-bench. CPU-only: a
fake slot under a temporary WIP root, a docker stub that reports no container, and
a stand-in python-dotenv wheel whose hash the test pins.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "scripts" / "lib" / "release-common.sh"
TOOL_EVAL = ROOT / "third_party" / "tool-eval-bench"


def _tree(path: Path) -> dict[str, bytes]:
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


@pytest.fixture
def env(tmp_path):
    if not (TOOL_EVAL / "pyproject.toml").is_file():
        pytest.fail("third_party/tool-eval-bench is not initialized; run scripts/dev/submodule-init-local")
    wip_root = tmp_path / "wip"
    ws = wip_root / "slots/sealed/coordinator/workspace"
    (ws / ".cuteafd-wip").mkdir(parents=True)
    (ws / ".cuteafd-wip/cuteafd").write_text("bin")
    (ws / ".cuteafd-wip/libcuteafd_native.so").write_text("lib")
    (ws / "docker").mkdir()
    (ws / "docker/release-entrypoint.sh").write_text("entry")
    (ws / "third_party/sparkinfer").mkdir(parents=True)
    (ws / "third_party/sparkinfer.lock.json").write_text("{}")
    (ws / "scripts/build").mkdir(parents=True)
    (ws / "scripts/build/verify-sparkinfer-source.py").write_text("raise SystemExit(0)\n")
    shutil.copy(ROOT / "scripts/build/verify-tool-eval-source.py", ws / "scripts/build/")
    shutil.copy(ROOT / "third_party/tool-eval-bench.lock.json", ws / "third_party/")
    source = ws / "third_party/tool-eval-bench"
    shutil.copytree(TOOL_EVAL, source, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    # A submodule gitfile whose module store is not reachable from the staged copy.
    (source / ".git").write_text("gitdir: ../../../../.git/modules/third_party/tool-eval-bench\n")
    shutil.copytree(ws, tmp_path / "workspace-twin")  # spark-expert role, same contents
    shutil.move(str(tmp_path / "workspace-twin"), wip_root / "slots/sealed/spark-expert/workspace")

    wheel_dir = tmp_path / "wheel-src"
    wheel = wheel_dir / "python_dotenv-test-py3-none-any.whl"
    wheel_dir.mkdir()
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("dotenv/__init__.py", "def load_dotenv(*args, **kwargs):\n    return False\n")
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    cache = tmp_path / "wheels"
    (cache / sha).mkdir(parents=True)
    shutil.copy(wheel, cache / sha / wheel.name)

    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "docker").write_text('#!/usr/bin/env bash\n[[ "$1 $2" == "container inspect" ]] && exit 1\necho "unexpected docker $*" >&2; exit 9\n')
    (binary / "docker").chmod(0o755)
    (binary / "curl").write_text('#!/usr/bin/env bash\necho "unexpected curl $*" >&2; exit 7\n')
    (binary / "curl").chmod(0o755)

    def stage(role="coordinator", wheel_name=wheel.name, wheel_sha=sha):
        layout = tmp_path / f"layout-{role}"
        script = (f'source "{COMMON}"; WIP_ROOT="{wip_root}"; RELEASE_TOOL_EVAL_DOTENV_WHEEL={wheel_name}; '
                  f'RELEASE_TOOL_EVAL_DOTENV_SHA256={wheel_sha}; release_stage_wip_layout c sealed {role} "{layout}"')
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                                env=dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", HOME=str(tmp_path),
                                         CUTEAFD_WHEEL_CACHE=str(cache)))
        return result, layout

    return type("Env", (), {"stage": staticmethod(stage), "source": source, "ws": ws, "cache": cache, "sha": sha})


def _run_cli(layout: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(layout / "bin/tool-eval-bench"), *args], capture_output=True, text=True, cwd=cwd,
                          env=dict(os.environ, CUTEAFD_WIP_SOURCE_ROOT=str(layout / "source"),
                                   PYTHONPATH="/existing/sparkinfer"))


def test_coordinator_layout_stages_verified_source_and_wrapper(env, tmp_path):
    before = _tree(env.ws)
    result, layout = env.stage()
    assert result.returncode == 0, result.stderr
    assert "verified tool-eval-bench revision=" in result.stderr
    staged = layout / "source/third_party/tool-eval-bench"
    assert not (staged / ".git").exists()
    assert (layout / "source/tool-eval-deps/dotenv/__init__.py").is_file()
    wrapper = (layout / "bin/tool-eval-bench").read_text()
    assert os.access(layout / "bin/tool-eval-bench", os.X_OK)
    assert 'export PYTHONPATH="$root/third_party/tool-eval-bench/src:$root/tool-eval-deps${PYTHONPATH:+:$PYTHONPATH}"' in wrapper
    assert '"${CUTEAFD_WIP_SOURCE_ROOT:-/source}"' in wrapper
    assert (layout / "bin/cuteafd").read_text() == "bin"  # existing layout unchanged
    assert _tree(env.ws) == before  # the sealed slot is only read


def test_wrapper_runs_help_from_the_staged_source(env, tmp_path):
    missing = [m for m in ("httpx", "yaml", "rich") if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"tool-eval-bench runtime deps not importable here: {missing} (run in the coordinator dev image)")
    result, layout = env.stage()
    assert result.returncode == 0, result.stderr
    cwd = tmp_path / "run"
    cwd.mkdir()
    help_ = _run_cli(layout, "--help", cwd=cwd)
    assert help_.returncode == 0, help_.stderr
    assert "--base-url" in help_.stdout and "--api-key" in help_.stdout and "--hardmode" in help_.stdout
    # The import resolves inside the staged source, the existing PYTHONPATH is kept,
    # and no bytecode lands in the read-only source.
    probe = subprocess.run(["sh", "-c", 'sed "s|^exec python3 -P -c .*|exec python3 -P -c \\"$1\\"|" "$0" > "$2" && sh "$2"',
                            str(layout / "bin/tool-eval-bench"),
                            "import sys, tool_eval_bench, dotenv; print(tool_eval_bench.__file__); print(dotenv.__file__); print(sys.path)",
                            str(tmp_path / "probe.sh")],
                           capture_output=True, text=True, cwd=cwd,
                           env=dict(os.environ, CUTEAFD_WIP_SOURCE_ROOT=str(layout / "source"), PYTHONPATH="/existing/sparkinfer"))
    assert probe.returncode == 0, probe.stderr
    module, dotenv_file, path = probe.stdout.splitlines()
    assert Path(module).is_relative_to(layout / "source/third_party/tool-eval-bench/src")
    assert Path(dotenv_file).is_relative_to(layout / "source/tool-eval-deps")
    assert "/existing/sparkinfer" in path and str(cwd) not in path
    assert not list((layout / "source").rglob("__pycache__"))


def test_stale_slot_source_fails_staging(env):
    target = env.source / "src/tool_eval_bench/__init__.py"
    target.write_text(target.read_text() + "\n# drift\n")
    result, layout = env.stage()
    assert result.returncode == 2
    assert "tool-eval-bench source content does not match the lock" in result.stderr
    assert "rebuild the slot with ./wip.sh" in result.stderr
    assert not layout.exists()


def test_slot_bytecode_is_not_staged(env):
    cache = env.source / "src/tool_eval_bench/__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"not verified")
    result, layout = env.stage()
    assert result.returncode == 0, result.stderr
    assert not list((layout / "source/third_party/tool-eval-bench").rglob("*.pyc"))
    assert (cache / "__init__.cpython-312.pyc").exists()  # slot untouched


def test_slot_without_tool_eval_source_stages_no_cli(env):
    shutil.rmtree(env.source)
    env.source.mkdir()  # an uninitialized submodule directory
    result, layout = env.stage()
    assert result.returncode == 0, result.stderr
    assert "no tool-eval-bench source" in result.stderr
    assert not (layout / "bin/tool-eval-bench").exists()
    assert not (layout / "source/third_party/tool-eval-bench").exists()
    assert (layout / "bin/cuteafd").read_text() == "bin"


def test_wheel_hash_mismatch_fails_staging(env):
    (env.cache / env.sha / "python_dotenv-test-py3-none-any.whl").write_bytes(b"tampered")
    result, layout = env.stage()
    assert result.returncode == 2 and "does not match its pinned sha256" in result.stderr
    assert not layout.exists()


def test_missing_wheel_without_network_is_a_named_error(env):
    shutil.rmtree(env.cache / env.sha)
    result, layout = env.stage()
    assert result.returncode == 2 and "cannot download python_dotenv-test-py3-none-any.whl" in result.stderr
    assert not layout.exists()


def test_expert_layout_is_unchanged(env):
    result, layout = env.stage("spark-expert")
    assert result.returncode == 0, result.stderr
    assert not (layout / "bin/tool-eval-bench").exists()
    assert sorted(p.name for p in (layout / "source/third_party").iterdir()) == ["sparkinfer", "sparkinfer.lock.json"]
    assert sorted(p.name for p in (layout / "source/scripts/build").iterdir()) == ["verify-sparkinfer-source.py"]


def test_pinned_dotenv_wheel_matches_pypi_release():
    text = COMMON.read_text()
    assert "RELEASE_TOOL_EVAL_DOTENV_WHEEL=python_dotenv-1.2.4-py3-none-any.whl" in text
    assert "RELEASE_TOOL_EVAL_DOTENV_SHA256=42269a8a5b3fd54ffa6f3d84b18abed50064717576b4ecf03dc4a55d8aa04fdc" in text
    assert "/python_dotenv-1.2.4-py3-none-any.whl\n" in text
