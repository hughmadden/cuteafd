"""Shared process budget for independent AOT variants and nested Ninja jobs."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import atexit
import json
import os
from pathlib import Path
import time


def export_jobs() -> int:
    value = os.environ.get('CUTEAFD_EXPORT_JOBS') or os.environ.get('CARGO_BUILD_JOBS')
    cores = os.cpu_count() or 1
    default = cores // 4 if os.environ.get('CUTEAFD_EXPORT_HOST_CLASS', 'raptor') == 'raptor' else cores
    jobs = int(value) if value else max(1, default)
    if jobs < 1:
        raise ValueError('CUTEAFD_EXPORT_JOBS must be positive')
    return jobs


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


def context_admission():
    """Reserve RAM before CUDA creates a context; hold until process exit."""
    global _context_lease
    if _context_lease is not None or not os.environ.get('CUTEAFD_EXPORT_HOST_SLOT_DIR'):
        return
    root = Path(os.environ['CUTEAFD_EXPORT_HOST_SLOT_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    per_worker = int(os.environ.get('CUTEAFD_EXPORT_WORKER_RAM_MIB', '1536'))
    count = int(os.environ.get('CUTEAFD_EXPORT_HOST_WORKERS', os.cpu_count() or 1))
    if per_worker < 1 or count < 1:
        raise ValueError('export worker RAM and count must be positive')
    deadline = time.monotonic() + 7200
    while True:
        with (root / 'context-admission.lock').open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            reserved = 0
            for path in root.glob("context.*.lock"):
                with path.open('a+') as existing:
                    try:
                        fcntl.flock(existing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        existing.seek(0)
                        reserved += int(existing.read() or per_worker)
            memory = {line.split(':')[0]: int(line.split()[1]) // 1024
                      for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemFree:', 'MemAvailable:'))}
            # GB10 allocations reclaim clean file cache; cudaMemGetInfo and
            # MemFree under-report the usable unified-memory budget.
            available = memory['MemAvailable']
            baseline_path = root / 'context-baseline.json'
            if reserved == 0:
                baseline_path.write_text(json.dumps({'available': available}))
            baseline = json.loads(baseline_path.read_text())['available']
            # Existing live contexts already reduce available RAM: subtract
            # reservations from the common baseline, not twice from the sample.
            budget = min(available, baseline - reserved)
            if budget >= per_worker + 1024:
                for index in range(count):
                    handle = (root / f'context.{index}.lock').open('a+')
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close()
                        continue
                    handle.seek(0)
                    handle.truncate()
                    handle.write(str(per_worker))
                    handle.flush()
                    _context_lease = handle
                    atexit.register(handle.close)
                    print(f'AOT context admitted: {per_worker} MiB reservation, {available} MiB free, '
                          f'{reserved} MiB already reserved', flush=True)
                    return
        if time.monotonic() >= deadline:
            raise TimeoutError('CUDA context waited 7200s for host RAM admission')
        time.sleep(60)


def install_context_admission():
    if not os.environ.get('CUTEAFD_EXPORT_HOST_SLOT_DIR'):
        return
    import torch
    original = torch.cuda._lazy_init
    if getattr(original, '_cuteafd_context_admission', False):
        return
    def admitted_init(*args, **kwargs):
        context_admission()
        return original(*args, **kwargs)
    admitted_init._cuteafd_context_admission = True
    torch.cuda._lazy_init = admitted_init
