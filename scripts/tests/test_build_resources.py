"""Build-only resources count leases; serving hosts remain exclusive."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python/tools/lib'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resource = load('build_resource', ROOT / 'scripts/build/build-resource.py')
export = load('export_jobs', ROOT / 'python/tools/lib/_export_jobs.py')


def test_seed_capacity_uses_cores_and_memory():
    assert resource.capacity(20, 124608) == 3
    assert resource.capacity(64, 65536) == 2
    assert resource.capacity(2, 4096) == 1


def test_counting_slots_preserve_legacy_exclusion_and_release(tmp_path):
    leases = [resource.acquire(tmp_path, 'kakapo', 3) for _ in range(3)]
    try:
        assert all(leases)
        assert resource.acquire(tmp_path, 'kakapo', 3) is None
        assert resource.try_lock(tmp_path / 'kakapo.lock') is None
        for handle in leases.pop():
            handle.close()
        extra = resource.acquire(tmp_path, 'kakapo', 3)
        assert extra
        leases.append(extra)
    finally:
        for lease in leases:
            for handle in lease:
                handle.close()
    exclusive = resource.try_lock(tmp_path / 'kakapo.lock')
    assert exclusive
    try:
        assert resource.acquire(tmp_path, 'kakapo', 3) is None
    finally:
        exclusive.close()


def test_busy_kakapo_falls_back_only_under_serving_host_lock(tmp_path):
    held = resource.acquire(tmp_path, 'kakapo', 1)
    rhea = resource.try_lock(tmp_path / 'rhea.lock')
    lease = tmp_path / 'lease'
    lease.mkdir()
    os.mkfifo(lease / 'release')
    process = subprocess.Popen(['python3', str(ROOT / 'scripts/build/build-resource.py'),
                               '--slots', '1', '--fallback', 'rhea,moa', '--wait', '0',
                               '--lease-dir', str(lease)],
                              env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path)})
    try:
        deadline = time.monotonic() + 5
        while not (lease / 'granted').exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert json.loads((lease / 'granted').read_text())['host'] == 'moa'
        assert resource.try_lock(tmp_path / 'moa.lock') is None
        with (lease / 'release').open('w') as handle:
            handle.write('x')
        assert process.wait(timeout=5) == 0
        released = resource.try_lock(tmp_path / 'moa.lock')
        assert released
        released.close()
    finally:
        process.terminate()
        process.wait(timeout=5)
        rhea.close()
        for handle in held:
            handle.close()


@pytest.mark.parametrize('kind,cores,expected', [('raptor', 64, 16), ('remote', 20, 8), ('remote', 4, 4)])
def test_export_budget(kind, cores, expected, monkeypatch):
    monkeypatch.setattr(os, 'cpu_count', lambda: cores)
    monkeypatch.delenv('CUTEAFD_EXPORT_JOBS', raising=False)
    monkeypatch.delenv('CARGO_BUILD_JOBS', raising=False)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_CLASS', kind)
    assert export.export_jobs() == expected
    monkeypatch.setenv('CARGO_BUILD_JOBS', '7')
    assert export.export_jobs() == expected
    monkeypatch.setenv('CUTEAFD_EXPORT_JOBS', '3')
    assert export.export_jobs() == 3
    monkeypatch.setenv('CUTEAFD_EXPORT_JOBS', '0')
    with pytest.raises(ValueError):
        export.export_jobs()


def test_export_slot_released_after_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('CUTEAFD_EXPORT_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_JOBS', '1')
    with pytest.raises(RuntimeError):
        with export.export_slot():
            assert resource.try_lock(tmp_path / '0.lock') is None
            raise RuntimeError('compiler failure')
    with export.export_slot():
        assert resource.try_lock(tmp_path / '0.lock') is None


def test_build_slot_default_admits_five_and_keeps_nice(tmp_path):
    held = []
    try:
        for name in ['build.lock', 'build.lock.1', 'build.lock.2', 'build.lock.3']:
            held.append(resource.try_lock(tmp_path / name))
        result = subprocess.run(['bash', str(ROOT / 'scripts/build/build-slot.sh'), 'bash', '-c',
                                 'ps -o ni= -p $$'], capture_output=True, text=True, timeout=5,
                                env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path), 'CUTEAFD_BUILD_SLOT_WAIT': '0'})
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == '19'
        assert 'build slot 4:' in result.stderr
    finally:
        for handle in held:
            handle.close()


def test_failed_probe_is_unavailable(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(255, 'ssh')
    monkeypatch.setattr(resource.subprocess, 'run', fail)
    assert resource.probe('kakapo') is None


def test_owner_death_before_grant_does_not_acquire(tmp_path):
    held = resource.acquire(tmp_path, 'kakapo', 1)
    lease = tmp_path / 'lease'
    lease.mkdir()
    os.mkfifo(lease / 'release')
    owner = subprocess.Popen(['sleep', '30'])
    process = subprocess.Popen(['python3', str(ROOT / 'scripts/build/build-resource.py'), '--slots', '1',
                                '--owner', str(owner.pid), '--wait', '30', '--lease-dir', str(lease)],
                               env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path)})
    try:
        time.sleep(0.1)
        owner.terminate()
        owner.wait(timeout=5)
        assert process.wait(timeout=5) == 0
        assert not (lease / 'granted').exists()
    finally:
        process.terminate()
        process.wait(timeout=5)
        for handle in held:
            handle.close()


def test_lease_has_no_elapsed_unlock_deadline_and_stops_before_unlock():
    text = (ROOT / 'scripts/build/build-resource.py').read_text()
    granted = text[text.index("(args.lease_dir / 'granted')"):]
    assert 'deadline' not in granted and '7260' not in granted
    assert granted.index('docker stop -t 10') < granted.index('handle.close()')


def test_host_worker_admission_is_shared_across_builds(tmp_path, monkeypatch):
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path / 'host'))
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_WORKERS', '1')
    monkeypatch.setenv('CUTEAFD_EXPORT_JOBS', '8')
    monkeypatch.setenv('CUTEAFD_EXPORT_SLOT_DIR', str(tmp_path / 'build-a'))
    with export.export_slot():
        assert resource.try_lock(tmp_path / 'host/0.lock') is None
        assert resource.try_lock(tmp_path / 'build-a/0.lock') is None
    with export.export_slot():
        assert resource.try_lock(tmp_path / 'host/0.lock') is None


def test_context_ram_reservation_precedes_cuda_and_includes_reclaimable_cache(tmp_path, monkeypatch):
    import platform
    monkeypatch.setattr(export, '_context_lease', None)
    monkeypatch.setattr(platform, 'machine', lambda: 'aarch64')
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_RAM_MIB', '1536')
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_WORKERS', '20')
    original = Path.read_text
    def text(path, *args, **kwargs):
        if str(path) == '/proc/self/cgroup':
            return ''
        if str(path) == '/proc/meminfo':
            return 'MemFree: 6291456 kB\nMemAvailable: 125829120 kB\n'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', text)
    export.context_admission()
    assert export._context_lease
    assert json.loads((tmp_path / 'context-baseline.json').read_text())['available'] == 122880
    assert (tmp_path / 'context.0.lock').read_text() == '1536'
    export._context_lease.close()
    export._context_lease = None


@pytest.mark.parametrize('available,reservation,expected', [(120000, 1536, True), (7000, 4096, False)])
def test_context_admission_accounts_live_reservations_across_worker_limits(tmp_path, monkeypatch, available, reservation, expected):
    monkeypatch.setattr(export, '_context_lease', None)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_RAM_MIB', str(reservation))
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_WORKERS', '2')
    (tmp_path / 'context-baseline.json').write_text(json.dumps({'available': available}))
    held = (tmp_path / 'context.9.lock').open('w+')
    held.write('4096')
    held.flush()
    fcntl.flock(held, fcntl.LOCK_EX)
    original = Path.read_text
    def text(path, *args, **kwargs):
        if str(path) == '/proc/self/cgroup':
            return ''
        if str(path) == '/proc/meminfo':
            return f'MemFree: 512000 kB\nMemAvailable: {available * 1024} kB\n'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', text)
    def stop_wait(seconds):
        assert seconds == 1
        raise TimeoutError('test stops the bounded admission wait')
    monkeypatch.setattr(export.time, 'sleep', stop_wait)
    try:
        if expected:
            export.context_admission()
            assert export._context_lease
            assert (tmp_path / 'context.0.lock').read_text() == str(reservation)
        else:
            with pytest.raises(TimeoutError):
                export.context_admission()
            assert export._context_lease is None
    finally:
        held.close()
        if export._context_lease:
            export._context_lease.close()
        export._context_lease = None


_ADMIT_CHILD = """
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
import _export_jobs
_export_jobs.context_admission()
admitted = time.monotonic()
time.sleep(float(sys.argv[2]))
print(json.dumps([admitted, time.monotonic()]), flush=True)
"""


def _admit_processes(tmp_path, count, hold, extra_env):
    env = {**os.environ, 'CUTEAFD_EXPORT_HOST_SLOT_DIR': str(tmp_path),
           'CUTEAFD_EXPORT_WORKER_RAM_MIB': '1', **extra_env}
    processes = [subprocess.Popen([sys.executable, '-c', _ADMIT_CHILD, str(ROOT / 'python/tools/lib'), str(hold)],
                                  env=env, stdout=subprocess.PIPE, text=True) for _ in range(count)]
    spans = []
    for process in processes:
        out, _ = process.communicate(timeout=60)
        assert process.returncode == 0
        spans.append(json.loads(out.strip().splitlines()[-1]))
    return sorted(spans)


def test_context_cap_and_spacing_hold_across_processes(tmp_path):
    spans = _admit_processes(tmp_path, 5, 0.6, {'CUTEAFD_EXPORT_HOST_CONTEXTS': '2',
                                                 'CUTEAFD_EXPORT_CONTEXT_SPACING_S': '0.25'})
    starts = [start for start, _ in spans]
    assert all(b - a >= 0.24 for a, b in zip(starts, starts[1:]))
    for start, _ in spans:
        live = sum(1 for other_start, other_end in spans if other_start <= start < other_end)
        assert live <= 2
    # Every lease was released on exit: all are lockable again.
    leases = sorted(tmp_path.glob('context.*.lock'))
    assert [path.name for path in leases] == ['context.0.lock', 'context.1.lock']
    for path in leases:
        assert resource.try_lock(path) is not None


@pytest.mark.parametrize('cls,machine,workers,expected', [
    ('remote', 'x86_64', '20', 8), ('raptor', 'aarch64', '20', 8),
    ('raptor', 'x86_64', '64', 64), ('remote', 'x86_64', '4', 4)])
def test_host_context_cap_defaults(monkeypatch, cls, machine, workers, expected):
    import platform
    monkeypatch.delenv('CUTEAFD_EXPORT_HOST_CONTEXTS', raising=False)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_CLASS', cls)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_WORKERS', workers)
    monkeypatch.setattr(platform, 'machine', lambda: machine)
    assert export.host_context_cap() == expected
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_CONTEXTS', '3')
    assert export.host_context_cap() == 3


def test_parked_planner_frees_its_lease_but_keeps_its_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(export, '_context_lease', None)
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_CONTEXTS', '1')
    monkeypatch.setenv('CUTEAFD_EXPORT_CONTEXT_SPACING_S', '0')
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_RAM_MIB', '1024')
    export.context_admission()
    try:
        export.park_context_lease()
        assert resource.try_lock(tmp_path / 'context.0.lock') is not None
        parked = list(tmp_path.glob('context.parked-*.lock'))
        assert len(parked) == 1 and parked[0].read_text() == '1024'
        # A worker still gets the single lease while the planner is parked.
        _admit_processes(tmp_path, 1, 0, {'CUTEAFD_EXPORT_HOST_CONTEXTS': '1',
                                          'CUTEAFD_EXPORT_CONTEXT_SPACING_S': '0'})
    finally:
        export._context_lease.close()
        export._context_lease = None
    # The next admission reaps the exited planner's parked file.
    _admit_processes(tmp_path, 1, 0, {'CUTEAFD_EXPORT_HOST_CONTEXTS': '1',
                                      'CUTEAFD_EXPORT_CONTEXT_SPACING_S': '0'})
    assert not list(tmp_path.glob('context.parked-*.lock'))


def test_weighted_gpu_budget_shared_across_package_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(export, '_gpu_lease', None)
    monkeypatch.setenv('CUTEAFD_EXPORT_GPU_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_GPU_BUDGET_MIB', '12288')
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_GPU_MIB', '2688')
    held = (tmp_path / 'context.999.lock').open('w+')
    held.write('10752')
    held.flush()
    fcntl.flock(held, fcntl.LOCK_EX)
    def stop_wait(seconds):
        raise TimeoutError('budget full')
    monkeypatch.setattr(export.time, 'sleep', stop_wait)
    try:
        with pytest.raises(TimeoutError):
            export.gpu_admission()
        assert export._gpu_lease is None
    finally:
        held.close()
    export.gpu_admission()
    assert export._gpu_lease
    assert (tmp_path / 'context.999.lock').read_text() == '2688'
    export._gpu_lease.close()
    export._gpu_lease = None


@pytest.mark.parametrize('signum', [1, 3])
def test_seed_hup_quit_stops_container_before_unlock(tmp_path, signum):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    marker = tmp_path / 'stopped'
    ssh = bin_dir / 'ssh'
    ssh.write_text(f'#!/bin/sh\nsleep 0.3\ntouch {marker}\n')
    ssh.chmod(0o755)
    lease = tmp_path / 'lease'
    lease.mkdir()
    os.mkfifo(lease / 'release')
    (lease / 'active').touch()
    process = subprocess.Popen(['python3', str(ROOT / 'scripts/build/build-resource.py'), '--host', 'rhea',
                                '--slots', '1', '--container', 'owned', '--lease-dir', str(lease)],
                               env={**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}',
                                    'CUTEAFD_LOCK_DIR': str(tmp_path)})
    try:
        deadline = time.monotonic() + 5
        while not (lease / 'granted').exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (lease / 'granted').exists()
        process.send_signal(signum)
        assert resource.try_lock(tmp_path / 'rhea.lock') is None
        assert process.wait(timeout=5) == 0
        assert marker.exists()
        handle = resource.try_lock(tmp_path / 'rhea.lock')
        assert handle
        handle.close()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_gpu_lease_pid_collision_never_truncates_live_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(export, '_gpu_lease', None)
    monkeypatch.setenv('CUTEAFD_EXPORT_GPU_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_GPU_BUDGET_MIB', '12288')
    monkeypatch.setenv('CUTEAFD_EXPORT_WORKER_GPU_MIB', '640')
    live = tmp_path / f'context.{os.getpid()}.lock'
    held = live.open('w+')
    held.write('2688')
    held.flush()
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        export.gpu_admission()
        assert live.read_text() == '2688'
        assert Path(export._gpu_lease.name) != live
        assert export._gpu_lease.readable()
    finally:
        held.close()
        if export._gpu_lease:
            export._gpu_lease.close()
        export._gpu_lease = None
