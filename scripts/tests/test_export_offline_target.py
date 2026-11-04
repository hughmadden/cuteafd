"""sm_89 exports compile on b12x's offline SM89 target with CUDA hidden.

CUTEAFD_EXPORT_OFFLINE_TARGET=sm89 makes _pinned_sparkinfer describe the target
from sm89_cards before any exporter touches CUDA; the native CMake sets it for
sm_89 builds only. The bootstrap tests import the real module in a subprocess
against a verified fake SparkInfer tree whose offline initializer records its
call, so they need neither CUDA nor torch.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "python/tools/lib"
sys.path.insert(0, str(LIB))
import sm89_cards  # noqa: E402

_spec = importlib.util.spec_from_file_location("verify_sparkinfer_offline", ROOT / "scripts/build/verify-sparkinfer-source.py")
VERIFIER = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(VERIFIER)

RECORDING_INITIALIZER = '''
import json, os

def _initialize_worker(ordinal, capability, uuid, name, sms, smem_block, smem_sm, activity):
    with open(os.environ["OFFLINE_RECORD"], "a") as record:
        record.write(json.dumps([ordinal, list(capability), uuid, name, sms, smem_block, smem_sm,
                                 list(activity), callable(activity.get_lock)]) + "\\n")
'''
FAKE_TORCH = '''
import os

class cuda:
    @staticmethod
    def is_initialized():
        return os.environ.get("FAKE_CUDA_INITIALIZED") == "1"

    @staticmethod
    def _lazy_init():
        pass
'''
PROBE = '''
import json, os
if os.environ.get("FAKE_CUDA_INITIALIZED"):
    import torch
import _pinned_sparkinfer as pinned
import torch
print(json.dumps({"target": pinned.OFFLINE_TARGET,
                  "admission": getattr(torch.cuda._lazy_init, "_cuteafd_context_admission", False)}))
'''


@pytest.fixture
def bootstrap(tmp_path):
    """Import _pinned_sparkinfer in a subprocess; return (process, recorded initializer calls)."""
    source = tmp_path / "sparkinfer"
    (source / "b12x/_lib").mkdir(parents=True)
    (source / "LICENSE").write_text("Apache-2.0 fixture\n")
    (source / "pyproject.toml").write_text('[project]\nname = "b12x"\nversion = "1.1.0"\n')
    (source / "b12x/__init__.py").write_text("")
    (source / "b12x/_lib/__init__.py").write_text("")
    (source / "b12x/_lib/compile_pool.py").write_text(RECORDING_INITIALIZER)
    lock = tmp_path / "sparkinfer.lock.json"
    lock.write_text(json.dumps({"schema": 1, "repository": "https://github.com/tpurtell/sparkinfer-glmrt.git",
                                "revision": "1" * 40, "source_tree_sha256": VERIFIER.source_tree_sha256(source)}))
    (tmp_path / "fake/torch").mkdir(parents=True)
    (tmp_path / "fake/torch/__init__.py").write_text(FAKE_TORCH)
    record = tmp_path / "record.jsonl"

    def run(**env):
        environment = {key: value for key, value in os.environ.items()
                       if key not in ("CUTEAFD_EXPORT_OFFLINE_TARGET", "FAKE_CUDA_INITIALIZED")}
        environment.update(
            CUTEAFD_SPARKINFER_SOURCE_DIR=str(source), CUTEAFD_SPARKINFER_LOCK_FILE=str(lock),
            # A host target of "native" keeps the real CuTe DSL (where installed) out of it.
            CUTEAFD_CUTE_HOST_TARGET="native", CUTEAFD_EXPORT_HOST_SLOT_DIR=str(tmp_path / "slots"),
            PYTHONPATH=os.pathsep.join((str(tmp_path / "fake"), str(LIB))), PYTHONDONTWRITEBYTECODE="1",
            OFFLINE_RECORD=str(record), **env)
        result = subprocess.run([sys.executable, "-c", PROBE], env=environment, cwd=tmp_path,
                                capture_output=True, text=True, timeout=60)
        calls = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
        return result, calls
    return run


def test_unset_keeps_the_visible_gpu(bootstrap):
    result, calls = bootstrap()
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"target": None, "admission": True}
    assert calls == []


def test_sm89_initializes_the_offline_target_from_sm89_cards(bootstrap):
    result, calls = bootstrap(CUTEAFD_EXPORT_OFFLINE_TARGET="sm89")
    assert result.returncode == 0, result.stderr
    # No CUDA context can exist, so no context admission either.
    assert json.loads(result.stdout) == {"target": "sm89", "admission": False}
    assert calls == [[0, [8, 9], "sm89-offline", "sm89-offline", 128, 101376, 102400, [0, 0], True]]
    assert (sm89_cards.CAPABILITY, sm89_cards.WORKSPACE_SMS) == ((8, 9), 128)
    assert "AOT export target: offline sm89, 128 SMs, CUDA hidden" in result.stderr


def test_unknown_target_fails_before_any_export(bootstrap):
    result, calls = bootstrap(CUTEAFD_EXPORT_OFFLINE_TARGET="sm120")
    assert result.returncode != 0 and calls == []
    assert "CUTEAFD_EXPORT_OFFLINE_TARGET='sm120': the offline export target is sm89" in result.stderr


def test_cuda_initialized_before_the_bootstrap_fails_loudly(bootstrap):
    result, calls = bootstrap(CUTEAFD_EXPORT_OFFLINE_TARGET="sm89", FAKE_CUDA_INITIALIZED="1")
    assert result.returncode != 0 and calls == []
    assert "CUDA was initialized before _pinned_sparkinfer" in result.stderr


def test_cmake_sets_the_target_for_sm89_builds_only():
    text = (ROOT / "native/CMakeLists.txt").read_text()
    block = re.search(r'\n  if\(CUTEAFD_CUDA_ARCHITECTURES STREQUAL "89"\)\n'
                      r'    list\(APPEND CUTEAFD_SPARKINFER_PYTHON_ENV "CUTEAFD_EXPORT_OFFLINE_TARGET=sm89"\)\n'
                      r'  endif\(\)\n', text)
    assert block and text.count("CUTEAFD_EXPORT_OFFLINE_TARGET") == 1
    # Appended to the exporter environment after it is defined and before the
    # 120 -> 120f rewrite or any exporter command reads it.
    assert text.index("set(CUTEAFD_SPARKINFER_PYTHON_ENV") < block.start()
    assert block.end() < text.index('set(CUTEAFD_CUDA_ARCHITECTURES "120f")')
    for path in (ROOT / "native").rglob("*.cmake"):
        assert "CUTEAFD_EXPORT_OFFLINE_TARGET" not in path.read_text(), path


def test_comparison_gate_exports_on_the_builds_offline_target():
    path = ROOT / "scripts/build/compare-sm-exports.py"
    spec = importlib.util.spec_from_file_location("compare_sm_offline_target", path)
    compare_sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare_sm)
    target = compare_sm.ARCHES["sm89"]
    assert tuple(target["capability"]) == sm89_cards.CAPABILITY
    assert max(target["sms"]) == sm89_cards.WORKSPACE_SMS
    worker = next(node for node in ast.parse(path.read_text()).body
                  if isinstance(node, ast.FunctionDef) and node.name == "worker")
    call, = [node for node in ast.walk(worker) if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "_initialize_worker"]
    assert [ast.literal_eval(arg) for arg in call.args[5:7]] == [
        sm89_cards.MAX_SHARED_MEMORY_PER_BLOCK, sm89_cards.MAX_SHARED_MEMORY_PER_SM]


def test_real_offline_target_hides_cuda():
    # In the coordinator compiler image this runs b12x's real offline target.
    pytest.importorskip("cutlass")
    probe = ("import os, torch, _pinned_sparkinfer as pinned\n"
             "props = torch.cuda.get_device_properties(0)\n"
             "assert (props.major, props.minor, props.multi_processor_count) == (8, 9, 128), props\n"
             "assert pinned.export_target(props) == 'sm89'\n"
             "assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()\n"
             "try:\n    torch.cuda.synchronize()\nexcept RuntimeError as error:\n    print(error)\n")
    env = dict(os.environ, CUTEAFD_EXPORT_OFFLINE_TARGET="sm89", PYTHONPATH=str(LIB), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "compiler workers cannot initialize CUDA" in result.stdout
