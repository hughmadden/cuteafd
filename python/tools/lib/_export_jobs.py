"""Shared process budget for independent AOT variants and nested Ninja jobs."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import atexit
import json
import os
from pathlib import Path
import platform
import resource
import threading
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
ADMISSION_TIMEOUT_S = 600
ADMISSION_WARN_S = 30


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


def _wait_for_admission(started, next_warn, status, kind):
    now = time.monotonic()
    if now >= next_warn:
        print(f'WARN AOT {kind} admission waiting {now - started:.0f}s: {status}', flush=True)
        next_warn = now + ADMISSION_WARN_S
    if now - started >= ADMISSION_TIMEOUT_S:
        raise TimeoutError(f'CUDA {kind} admission timed out after {ADMISSION_TIMEOUT_S}s: {status}')
    time.sleep(1)
    return next_warn


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
    started = time.monotonic()
    next_warn = started
    while True:
        with (root / 'admission.lock').open('a') as guard:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                next_warn = _wait_for_admission(started, next_warn,
                    f'free=unknown/{budget} MiB, needed={reservation} MiB, leases=unknown; admission lock busy', 'GPU')
                continue
            reserved = leases = 0
            vacant = None
            for path in root.glob('context.*.lock'):
                with path.open('a+') as existing:
                    try:
                        fcntl.flock(existing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        vacant = path
                    except BlockingIOError:
                        existing.seek(0)
                        reserved += int(existing.read())
                        leases += 1
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
        next_warn = _wait_for_admission(started, next_warn,
            f'free={max(0, budget - reserved)} MiB, needed={reservation} MiB, leases={leases}, '
            f'reserved={reserved} MiB, budget={budget} MiB', 'GPU')


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
    per_worker = int(os.environ.get('CUTEAFD_EXPORT_WORKER_RAM_MIB') or '1536')
    if _gb10_host():
        # Cold builds start conservatively; subsequent admissions use the
        # measured process + allocator high-water footprint on this host.
        try:
            measured = int((root / 'context-footprint-mib').read_text())
        except (FileNotFoundError, ValueError):
            measured = 0
        per_worker = max(per_worker, 4096, measured)
    count = host_context_cap()
    spacing = float(os.environ.get('CUTEAFD_EXPORT_CONTEXT_SPACING_S',
                                   CONTEXT_SPACING_S if _gb10_host() else 0))
    if per_worker < 1 or count < 1 or spacing < 0:
        raise ValueError('export worker RAM and count must be positive')
    started = time.monotonic()
    next_warn = started
    status = 'waiting for admission lock'
    while True:
        admitted = None
        details = {}
        with (root / 'context-admission.lock').open('a') as guard:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                status = (f'available={host_allocatable_memory() // (1024 * 1024)} MiB, '
                          f'needed={per_worker + 1024} MiB, leases=unknown/{count}; admission lock busy')
            else:
                admitted = _admit_locked(root, per_worker, count, spacing, details)
                status = details['status']
        if admitted is not None:
            print(admitted, flush=True)
            return _context_admitted_at
        next_warn = _wait_for_admission(started, next_warn, status, 'context')


def _admit_locked(root, per_worker, count, spacing, details):
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
    # GB10 can reclaim file cache. Burst protection comes from the seed export
    # turn, host context cap, spaced creation and PSI brake, not MemFree.
    available = host_allocatable_memory() // (1024 * 1024)
    baseline_path = root / 'context-baseline.json'
    if reserved == 0:
        baseline_path.write_text(json.dumps({'available': available}))
    baseline = json.loads(baseline_path.read_text())['available']
    # Existing live contexts already reduce available RAM: subtract
    # reservations from the common baseline, not twice from the sample.
    budget = min(available, baseline - reserved)
    pressure = memory_pressure()
    details['status'] = (f'available={available} MiB, budget={budget} MiB, needed={per_worker + 1024} MiB, '
                         f'leases={leased}/{count}, parked={parked}, reserved={reserved} MiB; '
                         f'memory pressure some avg10={pressure:.2f} (limit {MEMORY_PSI_LIMIT:g})')
    if pressure > MEMORY_PSI_LIMIT or budget < per_worker + 1024:
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
                f'{available} MiB available, {reserved} MiB already reserved; live contexts '
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


def _record_footprint(torch, root, initial_available):
    """Keep a host-wide measured high-water charge, including allocator cache.

    RSS covers Python, compiler IR/workspace and mapped driver pages; device
    allocator reservations can be outside RSS on GB10, so charge them as well.
    This intentionally overestimates shared library pages instead of trusting
    nvidia-smi's context-only figure as the entire export footprint.
    """
    peak_rss = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + 1023) // 1024
    allocated = torch.cuda.max_memory_allocated() // 1024**2
    cached = torch.cuda.max_memory_reserved() // 1024**2
    footprint = peak_rss + cached
    available = host_allocatable_memory() // 1024**2
    with (root / 'context-admission.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        path = root / 'context-footprint-mib'
        try:
            previous = int(path.read_text())
        except (FileNotFoundError, ValueError):
            previous = 0
        if footprint > previous:
            path.write_text(str(footprint))
        if _context_lease is not None and not _context_lease.closed:
            _context_lease.seek(0)
            charge = max(int(_context_lease.read() or 0), footprint)
            _context_lease.seek(0)
            _context_lease.truncate()
            _context_lease.write(str(charge))
            _context_lease.flush()
    return (f'pid={os.getpid()} rss_peak={peak_rss} MiB, '
            f'torch_allocated_peak={allocated} MiB, torch_reserved_peak={cached} MiB, '
            f'charge={footprint} MiB, available={available} MiB, '
            f'init_dip={initial_available - available} MiB')


def install_context_admission():
    if not os.environ.get('CUTEAFD_EXPORT_HOST_SLOT_DIR'):
        if not _gb10_host():
            return
        # Standalone GB10 exports share the same host pool as build launchers.
        os.environ['CUTEAFD_EXPORT_HOST_SLOT_DIR'] = str(
            Path.home() / '.cache/cuteafd/export-workers')
    import torch
    original = torch.cuda._lazy_init
    if getattr(original, '_cuteafd_context_admission', False):
        return
    initializing = threading.local()
    init_lock = threading.RLock()
    ready = False

    def admitted_init(*args, **kwargs):
        nonlocal ready
        # current_stream() below calls _lazy_init again. Let that recursive
        # call reach torch, without taking another host lock or another lease.
        if getattr(initializing, 'active', False):
            return original(*args, **kwargs)
        with init_lock:
            if ready or not _gb10_host():
                context_admission()
                gpu_admission()
                return original(*args, **kwargs)
            root = Path(os.environ['CUTEAFD_EXPORT_HOST_SLOT_DIR'])
            root.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            next_warn = started
            with (root / 'context-init.lock').open('a') as guard:
                while True:
                    try:
                        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        next_warn = _wait_for_admission(started, next_warn,
                            'waiting for actual CUDA context initialization', 'initialization')
                context_admission()
                gpu_admission()
                available = host_allocatable_memory() // 1024**2
                initializing.active = True
                try:
                    result = original(*args, **kwargs)
                    # Torch lazy init alone need not allocate the graphics
                    # context. This is the exact driver allocation that failed
                    # in current_cuda_stream AFTER the old admission spacing.
                    torch.cuda.current_stream().synchronize()
                    print('AOT context initialized: ' + _record_footprint(torch, root, available), flush=True)
                    ready = True
                finally:
                    initializing.active = False
            stopped = threading.Event()
            minimum_available = [available]
            def sample_memory():
                while not stopped.wait(1):
                    _record_footprint(torch, root, available)
                    minimum_available[0] = min(minimum_available[0],
                        host_allocatable_memory() // 1024**2)
            sampler = threading.Thread(target=sample_memory, daemon=True)
            sampler.start()
            def finished():
                stopped.set()
                sampler.join(timeout=2)
                print('AOT context finished: ' + _record_footprint(torch, root, available)
                      + f', sampled_min_available={minimum_available[0]} MiB', flush=True)
            atexit.register(finished)
            return result
    admitted_init._cuteafd_context_admission = True
    torch.cuda._lazy_init = admitted_init
