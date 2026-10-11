import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'hw' / 'hwlock.py'
spec = importlib.util.spec_from_file_location('hwlock', SCRIPT)
hwlock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hwlock)


@pytest.fixture
def lockdir(tmp_path, monkeypatch):
    monkeypatch.setattr(hwlock, 'LOCK_DIR', tmp_path)
    monkeypatch.setattr(hwlock, 'POLL_S', 0.05)
    return tmp_path


def spawn(lockdir, locks, seconds, marker):
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    cmd = [sys.executable, str(SCRIPT), '--locks', locks, '--wait', '30', '--',
           sys.executable, '-c', f'import time,pathlib; pathlib.Path({str(marker)!r}).write_text(str(time.time())); time.sleep({seconds})']
    return subprocess.Popen(cmd, env=env, stderr=subprocess.DEVNULL)


def wait_for(path, timeout=10):
    end = time.time() + timeout
    while not path.exists():
        assert time.time() < end, f'{path} never appeared'
        time.sleep(0.02)
    return float(path.read_text())


def test_canonical_order_and_unknown_locks():
    assert hwlock.canonical(['gpu1', 'sparks.lock', 'gpu0']) == ['sparks', 'gpu0', 'gpu1']
    with pytest.raises(ValueError):
        hwlock.canonical(['gpu2'])


def test_older_dual_gpu_waiter_is_not_overtaken_by_later_single_gpu_job(lockdir):
    # gpu1 is busy; a dual-GPU job queues; a later gpu1-only job must wait behind it.
    busy = spawn(lockdir, 'gpu1', 1.0, lockdir / 'busy')
    wait_for(lockdir / 'busy')
    dual = spawn(lockdir, 'gpu0,gpu1', 0.3, lockdir / 'dual')
    time.sleep(0.3)
    single = spawn(lockdir, 'gpu1', 0.1, lockdir / 'single')
    for proc in (busy, dual, single):
        assert proc.wait(timeout=30) == 0
    assert wait_for(lockdir / 'dual') < wait_for(lockdir / 'single')


def test_disjoint_locks_run_in_parallel_past_a_waiter(lockdir):
    busy = spawn(lockdir, 'gpu1', 1.0, lockdir / 'busy')
    wait_for(lockdir / 'busy')
    dual = spawn(lockdir, 'gpu0,gpu1', 0.1, lockdir / 'dual')
    time.sleep(0.3)
    sparks = spawn(lockdir, 'sparks', 0.1, lockdir / 'sparks')
    started = wait_for(lockdir / 'sparks')
    assert not (lockdir / 'dual').exists() or started < wait_for(lockdir / 'dual')
    for proc in (busy, dual, sparks):
        assert proc.wait(timeout=30) == 0


def test_stale_ticket_from_dead_process_is_ignored_and_removed(lockdir):
    queue = lockdir / 'hwq'
    queue.mkdir()
    stale = queue / '000000000007.99999.ticket'
    stale.write_text('{"locks": ["gpu1"]}')  # nobody holds its flock
    with hwlock.hold(['gpu1'], wait=5):
        pass
    assert not stale.exists()


def test_timeout_exits_75_without_holding_anything(lockdir):
    busy = spawn(lockdir, 'gpu1', 6.0, lockdir / 'busy')
    wait_for(lockdir / 'busy')
    env = {**os.environ, 'CUTEAFD_LOCK_DIR': str(lockdir)}
    result = subprocess.run([sys.executable, str(SCRIPT), '--locks', 'gpu1', '--wait', '0.3', '--', 'true'],
                            env=env, capture_output=True)
    assert result.returncode == 75
    assert busy.wait(timeout=30) == 0
    assert not list((lockdir / 'hwq').glob('*.ticket'))
