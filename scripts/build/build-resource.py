#!/usr/bin/env python3
"""Bounded build-only host leases; serving seeds retain their hardware mutex."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def capacity(cores: int, memory_mib: int) -> int:
    return max(1, min(cores // 6, memory_mib // 32768))


def try_lock(path: Path, shared: bool = False):
    handle = path.open('a')
    try:
        fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        return handle
    except BlockingIOError:
        handle.close()
        return None


def acquire(directory: Path, host: str, slots: int):
    # Shared umbrella excludes legacy exclusive kakapo.lock users.
    umbrella = try_lock(directory / f'{host}.lock', shared=True)
    if umbrella is None:
        return None
    for slot in range(slots):
        handle = try_lock(directory / f'{host}.build.{slot}.lock')
        if handle is not None:
            return [umbrella, handle]
    umbrella.close()
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='kakapo')
    parser.add_argument('--slots', type=int)
    parser.add_argument('--fallback', default='')
    parser.add_argument('--lease-dir', type=Path, required=True)
    parser.add_argument('--wait', type=int, default=1800)
    args = parser.parse_args()
    parent = os.getppid()
    slots = args.slots
    if slots is None:
        output = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', args.host,
                                 'nproc; awk \'/MemTotal/ {print int($2/1024)}\' /proc/meminfo'],
                                capture_output=True, text=True, timeout=15, check=True).stdout.split()
        slots = capacity(*map(int, output))
    if slots < 1 or args.wait < 0:
        parser.error('slots must be positive and wait nonnegative')
    directory = Path(os.environ.get('CUTEAFD_LOCK_DIR', str(Path.home() / '.cache/cuteafd')))
    if not directory.is_dir():
        parser.error(f'lock directory does not exist: {directory}')
    deadline = time.monotonic() + args.wait
    handles = None
    while handles is None:
        host = args.host
        handles = acquire(directory, host, slots)
        if handles is None:
            for candidate in filter(None, args.fallback.split(',')):
                if candidate not in ('rhea', 'moa'):
                    parser.error('fallback seeds must be rhea or moa')
                lock = try_lock(directory / f'{candidate}.lock')
                if lock is not None:
                    handles, host = [lock], candidate
                    break
        if handles is not None:
            break
        if time.monotonic() >= deadline:
            raise SystemExit(75)
        time.sleep(15)
    (args.lease_dir / 'granted').write_text(json.dumps({'host': host, 'slots': slots}))
    # Nonblocking FIFO permits parent-death detection and a hard lease deadline.
    fd = os.open(args.lease_dir / 'release', os.O_RDONLY | os.O_NONBLOCK)
    try:
        while os.getppid() == parent and time.monotonic() < deadline + 7260:
            if os.read(fd, 1):
                break
            time.sleep(0.2)
    finally:
        os.close(fd)
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
