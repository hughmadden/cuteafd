#!/usr/bin/env python3
"""Bounded build-only host leases; serving seeds retain their hardware mutex."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def capacity(cores: int, memory_mib: int) -> int:
    return max(1, min(cores // 6, memory_mib // 32768))


def probe(host: str) -> int | None:
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host,
                                 'nproc; awk \'/MemTotal/ {print int($2/1024)}\' /proc/meminfo'],
                                capture_output=True, text=True, timeout=15, check=True)
        return capacity(*map(int, result.stdout.split()))
    except (subprocess.SubprocessError, OSError, ValueError, TypeError):
        return None


def owner_alive(pid: int, start: str) -> bool:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == start
    except (OSError, IndexError):
        return False


def try_lock(path: Path, shared: bool = False):
    handle = path.open('a')
    try:
        fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        return handle
    except BlockingIOError:
        handle.close()
        return None


def acquire(directory: Path, host: str, slots: int):
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
    parser.add_argument('--owner', type=int, default=os.getppid())
    parser.add_argument('--container')
    parser.add_argument('--lock-host')
    args = parser.parse_args()
    if args.slots is not None and args.slots < 1 or args.wait < 0:
        parser.error('slots must be positive and wait nonnegative')
    candidates = list(filter(None, args.fallback.split(',')))
    if any(host not in ('rhea', 'moa') for host in candidates):
        parser.error('fallback seeds must be rhea or moa')
    start = Path(f'/proc/{args.owner}/stat').read_text().rsplit(')', 1)[1].split()[19]
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    directory = Path(os.environ.get('CUTEAFD_LOCK_DIR', str(Path.home() / '.cache/cuteafd')))
    if not directory.is_dir():
        parser.error(f'lock directory does not exist: {directory}')
    slots = args.slots if args.slots is not None else probe(args.host)
    deadline = time.monotonic() + args.wait
    handles = None
    while not stopping and owner_alive(args.owner, start):
        host = args.host
        if slots is None:
            handles = None
        elif host == 'kakapo':
            handles = acquire(directory, host, slots)
        else:
            lock = try_lock(directory / f'{args.lock_host or host}.lock')
            handles = [lock] if lock is not None else None
        if handles is None:
            for candidate in candidates:
                lock = try_lock(directory / f'{candidate}.lock')
                if lock is not None:
                    handles, host = [lock], candidate
                    break
        if handles is not None:
            break
        if time.monotonic() >= deadline:
            raise SystemExit(75)
        time.sleep(1)
    if handles is None:
        return
    fd = None
    try:
        if stopping or not owner_alive(args.owner, start):
            return
        (args.lease_dir / 'granted').write_text(json.dumps({'host': host, 'slots': slots or 1}))
        fd = os.open(args.lease_dir / 'release', os.O_RDONLY | os.O_NONBLOCK)
        # No lease expiry: each phase has its own terminating timeout. Stop
        # the instance's build container before *any* mutex release.
        while not stopping and owner_alive(args.owner, start):
            if os.read(fd, 1):
                break
            time.sleep(0.2)
        if args.container and (args.lease_dir / 'active').exists():
            while True:
                try:
                    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host,
                                         f'names=$(docker ps -a --filter name=^/{args.container}$ --format \'{{{{.Names}}}}\') && '
                                         f'{{ test -z "$names" || {{ docker stop -t 10 {args.container} >/dev/null; '
                                         f'test "$(docker inspect -f \'{{{{.State.Running}}}}\' {args.container})" = false; }}; }}'],
                                            timeout=30, check=False)
                except (subprocess.SubprocessError, OSError):
                    time.sleep(60)
                    continue
                if result.returncode == 0:
                    break
                # Fail closed: an unreachable host is not proof that CUDA stopped.
                time.sleep(60)
    finally:
        if fd is not None:
            os.close(fd)
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
