"""Shared process budget for independent AOT variants and nested Ninja jobs."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import atexit
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import uuid

from allocatable_memory import host_allocatable_memory


GB10_EXPORT_JOBS = 8  # each worker holds a CUDA context plus ~4 GiB; GB10 has 121 GiB unified
# GB10 context creation fails in the driver (NV_ERR_NO_MEMORY) when ~16
# contexts start within seconds, even with ~100 GiB MemAvailable (mostly page
# cache that the burst cannot reclaim). Cap live export contexts host-wide and
# space their creation.
GB10_EXPORT_CONTEXTS = 8
CONTEXT_SPACING_S = 1.5  # GB10 only; discrete GPUs never showed the burst failure
# Long-lived EXL3 variant workers per package on GB10: each keeps one CUDA
# context for all its variants instead of creating one per variant.
GB10_GPU_WORKERS = 4
# GB10 driver allocations (NV_ERR_NO_MEMORY) fail under the build's own memory
# pressure while MemAvailable still looks ample: no new context is admitted
# while memory PSI "some avg10" is above this percentage.
MEMORY_PSI_LIMIT = 10.0
PRESSURE_PATH = Path('/proc/pressure/memory')


def _gb10_host() -> bool:
    # Release builds do not set the host class; every aarch64 builder is a GB10.
    return (os.environ.get('CUTEAFD_EXPORT_HOST_CLASS', 'raptor') != 'raptor'
            or platform.machine() == 'aarch64')


def host_context_cap() -> int:
    """Concurrent CUDA export contexts on this host, across steps and builds."""
    value = os.environ.get('CUTEAFD_EXPORT_HOST_CONTEXTS')
    if value:
        return int(value)
    workers = int(os.environ.get('CUTEAFD_EXPORT_HOST_WORKERS', os.cpu_count() or 1))
    return min(workers, GB10_EXPORT_CONTEXTS) if _gb10_host() else workers


def export_jobs() -> int:
    # CARGO_BUILD_JOBS is a CPU-compile knob, not a GPU-context budget.
    value = os.environ.get('CUTEAFD_EXPORT_JOBS')
    cores = os.cpu_count() or 1
    if os.environ.get('CUTEAFD_EXPORT_HOST_CLASS', 'raptor') == 'raptor':
        default = cores // 4
    else:  # Spark (GB10) expert-role builds
        default = min(cores, GB10_EXPORT_JOBS)
    jobs = int(value) if value else max(1, default)
    if jobs < 1:
        raise ValueError('CUTEAFD_EXPORT_JOBS must be positive')
    return jobs


def gpu_workers(jobs: int) -> int:
    """Persistent context-owning workers one package may run at once."""
    return min(jobs, GB10_GPU_WORKERS) if _gb10_host() else jobs


def memory_pressure() -> float:
    """Memory PSI "some avg10" (percent); 0 where the kernel has no PSI."""
    try:
        for line in PRESSURE_PATH.read_text().splitlines():
            if line.startswith('some '):
                return float(dict(field.split('=', 1) for field in line.split()[1:])['avg10'])
    except (OSError, ValueError, KeyError):
        pass
    return 0.0


def live_cuda_processes() -> str:
    """Host-wide compute processes as nvidia-smi sees them (pid:MiB), for logs."""
    smi = shutil.which('nvidia-smi')
    if smi is None:
        return 'nvidia-smi unavailable'
    try:
        out = subprocess.run([smi, '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5, check=True).stdout
    except (subprocess.SubprocessError, OSError) as error:
        return f'nvidia-smi failed ({type(error).__name__})'
    rows = [line.replace(' ', '').replace(',', ':') for line in out.splitlines() if line.strip()]
    return f'{len(rows)} compute processes' + (f' ({" ".join(rows)})' if rows else '')


@contextmanager
def _slot(root, count):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 7200
    while True:
        for slot in range(count):
            handle = (root / f'{slot}.lock').open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield
            finally:
                handle.close()
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('AOT variant waited 7200s for an export slot')
        time.sleep(0.2)


@contextmanager
def export_slot():
    local = os.environ.get('CUTEAFD_EXPORT_SLOT_DIR',
                           str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'export-slots'))
    host = os.environ.get('CUTEAFD_EXPORT_HOST_SLOT_DIR', local)
    # Both bounds matter: a shared host pool must not enlarge one export's
    # admitted GPU/context budget when sibling Ninja packages multiply workers.
    with _slot(local, export_jobs()):
        if host == local:
            yield
        else:
            with _slot(host, int(os.environ.get('CUTEAFD_EXPORT_HOST_WORKERS', os.cpu_count() or 1))):
                yield


_context_lease = None
_context_admitted_at = None
_gpu_lease = None


def gpu_admission():
    """Weighted device budget shared by all packages on the selected GPU."""
    global _gpu_lease
    if _gpu_lease is not None or not os.environ.get('CUTEAFD_EXPORT_GPU_SLOT_DIR'):
        return
    root = Path(os.environ['CUTEAFD_EXPORT_GPU_SLOT_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    budget = int(os.environ['CUTEAFD_EXPORT_GPU_BUDGET_MIB'])
    reservation = int(os.environ.get('CUTEAFD_EXPORT_WORKER_GPU_MIB', '640'))
    if reservation < 1 or reservation > budget:
        raise ValueError('export GPU reservation exceeds budget')
    deadline = time.monotonic() + 7200
    while True:
        with (root / 'admission.lock').open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            reserved = 0
            vacant = None
            for path in root.glob('context.*.lock'):
                with path.open('a+') as existing:
                    try:
                        fcntl.flock(existing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        vacant = path
                    except BlockingIOError:
                        existing.seek(0)
                        reserved += int(existing.read())
            if reserved + reservation <= budget:
                path = vacant or root / f'context.{uuid.uuid4().hex}.lock'
                handle = path.open('r+' if vacant else 'x+')
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle.seek(0)
                handle.truncate()
                handle.write(str(reservation))
                handle.flush()
                _gpu_lease = handle
                atexit.register(handle.close)
                print(f'AOT GPU admitted: {reservation} MiB, {reserved} MiB reserved, {budget} MiB budget', flush=True)
                return
        if time.monotonic() >= deadline:
            raise TimeoutError('CUDA context waited 7200s for GPU budget')
        time.sleep(1)


def context_admission():
    """Reserve RAM before CUDA creates a context; hold until process exit.

    One host-wide lease set (``context.<index>.lock``, at most
    host_context_cap()) covers every Ninja step and every pool worker inside a
    step, and consecutive admissions are spaced so contexts are created one
    at a time rather than in a burst.
    """
    global _context_lease
    if _context_lease is not None or not os.environ.get('CUTEAFD_EXPORT_HOST_SLOT_DIR'):
        return
    root = Path(os.environ['CUTEAFD_EXPORT_HOST_SLOT_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    per_worker = int(os.environ.get('CUTEAFD_EXPORT_WORKER_RAM_MIB', '1536'))
    count = host_context_cap()
    spacing = float(os.environ.get('CUTEAFD_EXPORT_CONTEXT_SPACING_S',
                                   CONTEXT_SPACING_S if _gb10_host() else 0))
    if per_worker < 1 or count < 1 or spacing < 0:
        raise ValueError('export worker RAM and count must be positive')
    deadline = time.monotonic() + 7200
    pressured = False
    while True:
        admitted = None
        with (root / 'context-admission.lock').open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            pressure = memory_pressure()
            if pressure > MEMORY_PSI_LIMIT:
                if not pressured:
                    print(f'AOT context waits: memory pressure some avg10={pressure:.2f} > {MEMORY_PSI_LIMIT:g}',
                          flush=True)
                pressured = True
            else:
                admitted = _admit_locked(root, per_worker, count, spacing)
        if admitted is not None:
            print(admitted, flush=True)
            return _context_admitted_at
        if time.monotonic() >= deadline:
            raise TimeoutError('CUDA context waited 7200s for host RAM admission')
        time.sleep(1)


def _admit_locked(root, per_worker, count, spacing):
    """One admission attempt under context-admission.lock; a log line or None."""
    global _context_lease
    reserved = 0
    leased = parked = 0
    for path in root.glob("context.*.lock"):
        with path.open('a+') as existing:
            try:
                fcntl.flock(existing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if path.name.startswith('context.parked-'):
                    path.unlink()  # its process exited
            except BlockingIOError:
                existing.seek(0)
                reserved += int(existing.read() or per_worker)
                if path.name.startswith('context.parked-'):
                    parked += 1
                else:
                    leased += 1
    # Reserve before CUDA initialization, using cgroup-bounded OS headroom.
    available = host_allocatable_memory() // (1024 * 1024)
    baseline_path = root / 'context-baseline.json'
    if reserved == 0:
        baseline_path.write_text(json.dumps({'available': available}))
    baseline = json.loads(baseline_path.read_text())['available']
    # Existing live contexts already reduce available RAM: subtract
    # reservations from the common baseline, not twice from the sample.
    budget = min(available, baseline - reserved)
    if budget < per_worker + 1024:
        return None
    for index in range(count):
        handle = (root / f'context.{index}.lock').open('a+')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        # Every admission names the live contexts (parked planners included),
        # so a later driver failure carries the host's context census. Taken
        # before the spacing wait: CUDA init must follow the spaced admission.
        census = live_cuda_processes()
        # Spacing holds the admission lock, so waiters queue behind it.
        last_path = root / 'context-last-admit'
        try:
            last = float(last_path.read_text())
        except (FileNotFoundError, ValueError):
            last = None
        now = time.monotonic()
        if last is not None and 0 <= now - last < spacing:
            time.sleep(spacing - (now - last))
        global _context_admitted_at
        _context_admitted_at = time.monotonic()
        last_path.write_text(repr(_context_admitted_at))
        handle.seek(0)
        handle.truncate()
        handle.write(str(per_worker))
        handle.flush()
        _context_lease = handle
        atexit.register(handle.close)
        return (f'AOT context admitted: lease {index}/{count}, {per_worker} MiB reservation, '
                f'{available} MiB free, {reserved} MiB already reserved; live contexts '
                f'{leased} leased + {parked} parked before this one; memory PSI some avg10={memory_pressure():.2f}; '
                f'host compute: {census}')
    return None


def park_context_lease():
    """Free this process's counted lease while keeping its RAM reservation.

    A planner whose own context idles while its pool workers run must not
    hold one of the host's context leases: concurrent planners would starve
    (or, at the cap, deadlock) the workers they wait on.
    """
    global _context_lease
    root = os.environ.get('CUTEAFD_EXPORT_HOST_SLOT_DIR')
    if _context_lease is None or not root:
        return
    root = Path(root)
    with (root / 'context-admission.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        _context_lease.seek(0)
        reservation = _context_lease.read()
        parked = (root / f'context.parked-{uuid.uuid4().hex}.lock').open('x+')
        fcntl.flock(parked, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parked.write(reservation)
        parked.flush()
        _context_lease.close()
        _context_lease = parked
        atexit.register(parked.close)


def install_context_admission():
    if not os.environ.get('CUTEAFD_EXPORT_HOST_SLOT_DIR'):
        return
    import torch
    original = torch.cuda._lazy_init
    if getattr(original, '_cuteafd_context_admission', False):
        return
    def admitted_init(*args, **kwargs):
        context_admission()
        gpu_admission()
        return original(*args, **kwargs)
    admitted_init._cuteafd_context_admission = True
    torch.cuda._lazy_init = admitted_init
