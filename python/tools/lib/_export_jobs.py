"""Shared process budget for independent AOT variants and nested Ninja jobs."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
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
def export_slot():
    root = Path(os.environ.get('CUTEAFD_EXPORT_SLOT_DIR',
                               str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'export-slots')))
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 7200
    while True:
        for slot in range(export_jobs()):
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
