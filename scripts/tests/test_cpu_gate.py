"""scripts/dev/cpu-gate: verdicts from parsed test summaries, never exit codes alone."""
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOADER = importlib.machinery.SourceFileLoader('cpu_gate', str(ROOT / 'scripts/dev/cpu-gate'))
SPEC = importlib.util.spec_from_loader('cpu_gate', LOADER)
gate = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(gate)

ARGS = SimpleNamespace(cargo_build_timeout=60, cargo_timeout=60, scripts_timeout=60, skip_live_planners=False)


def _fake(monkeypatch, outputs):
    """outputs: list of (exit, text) in call order for run_logged."""
    calls = []
    def run_logged(cmd, log_path, env, cwd, timeout):
        code, text = outputs[len(calls)]
        calls.append(cmd)
        Path(log_path).write_text(text)
        return code, 1.0
    monkeypatch.setattr(gate, 'run_logged', run_logged)
    monkeypatch.setattr(gate, 'checked', lambda cmd, **k: '')
    return calls


def test_cargo_passes_only_with_parsed_tests_and_no_failures(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    ok = 'test result: ok. 12 passed; 0 failed; 1 ignored; 0 measured\ntest result: ok. 3 passed; 0 failed; 0 ignored\n'
    _fake(monkeypatch, [(0, 'built'), (0, ok)])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'pass' and result['passed'] == 15 and result['binaries'] == 2
    # Exit 0 but nothing ran (e.g. a wrapper that masked a missing lock): failed.
    _fake(monkeypatch, [(0, 'built'), (0, 'fd: unbound variable\n')])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and 'no test summaries' in result['error']
    bad = ok + 'test a::b ... FAILED\ntest result: FAILED. 4 passed; 1 failed; 0 ignored\n'
    _fake(monkeypatch, [(0, 'built'), (101, bad)])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and result['failing'] == ['a::b']
    _fake(monkeypatch, [(1, 'error[E0425]: cannot find value\nerror: could not compile\n')])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and 'E0425' in result['error']


def test_live_planner_skip_is_explicit_and_recorded(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    calls = _fake(monkeypatch, [(0, ''), (0, 'test result: ok. 1 passed; 0 failed; 0 ignored\n')])
    result = gate.cargo(ROOT, tmp_path, SimpleNamespace(**{**vars(ARGS), 'skip_live_planners': True}), {})
    test_cmd = calls[1]
    assert test_cmd[-1] == '--exact' and test_cmd.count('--skip') == 2
    assert result['skipped_live_planners'] == list(gate.LIVE_PLANNERS)
    record = {'task': 't', 'verdict': 'pass', 'head': 'a' * 40, 'dirty': False, 'suites': [result]}
    assert 'live V4 planner tests skipped' in gate.render(record)


def test_scripts_parse_failing_ids_and_empty_runs(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    monkeypatch.setattr(gate, 'python_for', lambda source: Path('/usr/bin/python3'))
    _fake(monkeypatch, [(1, 'FAILED scripts/tests/test_x.py::test_y - assert 0\n=== 1 failed, 20 passed, 2 skipped in 3.10s ===\n')])
    result = gate.scripts(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and result['failing'] == ['scripts/tests/test_x.py::test_y']
    assert (result['passed'], result['failed'], result['skipped']) == (20, 1, 2)
    _fake(monkeypatch, [(0, '808 passed in 100.00s\n')])
    assert gate.scripts(ROOT, tmp_path, ARGS, {})['status'] == 'pass'
    _fake(monkeypatch, [(5, 'no tests ran in 0.01s\n')])
    assert gate.scripts(ROOT, tmp_path, ARGS, {})['status'] == 'failed'


def test_build_slot_refuses_a_missing_lock_directory(tmp_path):
    result = subprocess.run(['bash', str(ROOT / 'scripts/build/build-slot.sh'), 'true'], capture_output=True, text=True,
                            env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path / 'absent')})
    assert result.returncode == 2 and 'lock directory' in result.stderr and 'missing' in result.stderr
    (tmp_path / 'locks').mkdir()
    result = subprocess.run(['bash', str(ROOT / 'scripts/build/build-slot.sh'), 'true'], capture_output=True, text=True,
                            env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path / 'locks')})
    assert result.returncode == 0 and (tmp_path / 'locks/build.lock').exists()
