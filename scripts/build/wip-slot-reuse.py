#!/usr/bin/env python3
"""Find and adopt a sealed WIP slot built from the same inputs (wip.sh build key).

Runs on the host that holds the slot (raptor for coordinator slots, the seed
Spark for expert slots; wip.sh pipes it over ssh). A candidate qualifies only
when its META records exactly the requested build key and its seal verifies:
FINGERPRINT == sha256(META.json), the source and artifact manifests match the
hashes META records, and every artifact matches its manifest line. Adoption
hardlinks the candidate into this instance's root and rewrites only the slot
identity in META (instance, root, slot), then reseals it; the source and
artifact manifests, and so every built byte, are the candidate's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROLES = ('coordinator', 'spark-expert')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(slot: Path, key: str) -> dict:
    """The candidate's META if its seal and every artifact verify, else raise."""
    meta_bytes = (slot / 'META.json').read_bytes()
    if hashlib.sha256(meta_bytes).hexdigest() != (slot / 'FINGERPRINT').read_text().strip():
        raise ValueError('META seal mismatch')
    meta = json.loads(meta_bytes)
    if meta.get('build_key') != key:
        raise ValueError('build key differs')
    artifacts = slot / 'workspace/.cuteafd-wip'
    for relative, field in (('SOURCE_SHA256SUMS', 'source_manifest_sha256'),
                            ('workspace/.cuteafd-wip/ARTIFACT_SHA256SUMS', 'artifact_manifest_sha256')):
        if sha256(slot / relative) != meta[field]:
            raise ValueError(relative + ' seal mismatch')
    for line in (artifacts / 'ARTIFACT_SHA256SUMS').read_text().splitlines():
        expected, name = line.split(None, 1)
        if sha256(artifacts / name.lstrip('*')) != expected:
            raise ValueError('artifact mismatch: ' + name)
    # Packages are not in ARTIFACT_SHA256SUMS; their own manifests were verified
    # at build time, and the source manifest covers the workspace tree. Check the
    # source manifest against the files on disk as the final gate.
    checked = subprocess.run(['sha256sum', '--quiet', '--strict', '-c', str(slot / 'SOURCE_SHA256SUMS')],
                             cwd=slot / 'workspace', capture_output=True, text=True)
    if checked.returncode:
        raise ValueError('source tree mismatch: ' + (checked.stdout or checked.stderr).strip()[-200:])
    return meta


def find(args) -> int:
    builds = Path.home() / '.cache/cuteafd/builds'
    candidates = []
    # This instance's own slot first: a repeat build of the same key is a no-op.
    own = Path(args.exclude_root) / 'slots' / args.slot / args.role if args.exclude_root else None
    paths = ([own / 'META.json'] if own and (own / 'META.json').is_file() else [])
    paths += [p for p in builds.glob(f'wip-*/slots/*/{args.role}/META.json') if not own or p.parent.resolve() != own.resolve()]
    for meta_path in paths:
        try:
            if json.loads(meta_path.read_text()).get('build_key') != args.key:
                continue
        except (OSError, ValueError):
            continue
        candidates.append(meta_path.parent)
    # Own slot, then newest first; the first one that verifies wins.
    ordered = sorted(candidates, key=lambda p: (own is not None and p == own, p.joinpath('META.json').stat().st_mtime), reverse=True)
    for slot in ordered:
        try:
            verify(slot, args.key)
        except (OSError, ValueError, KeyError) as error:
            print(f'wip slot reuse: skipping {slot}: {error}', file=sys.stderr)
            continue
        print(slot)
        return 0
    return 1


def adopt(args) -> int:
    source = Path(args.source).resolve()
    meta = verify(source, args.key)
    root = Path(args.root)
    target = root / 'slots' / args.slot / args.role
    if target.resolve() == source:
        print(f'wip slot reuse: {target} already holds the build')
        return 0
    incoming = root / 'incoming' / f'{args.slot}.{args.role}.reuse.{os.getpid()}'
    shutil.rmtree(incoming, ignore_errors=True)
    incoming.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(['cp', '-al', str(source), str(incoming)], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        shutil.rmtree(incoming, ignore_errors=True)
        subprocess.run(['cp', '-a', str(source), str(incoming)], check=True)
    # META and FINGERPRINT are rewritten, never edited through a shared link.
    for name in ('META.json', 'FINGERPRINT'):
        (incoming / name).unlink()
    meta.update(wip_instance=args.instance, wip_root=args.wip_root, slot=args.slot,
                reused_from={'path': str(source), 'seal_sha256': (source / 'FINGERPRINT').read_text().strip(),
                             'wip_instance': meta.get('wip_instance'), 'slot': meta.get('slot')})
    meta_bytes = (json.dumps(meta, indent=2, sort_keys=True) + '\n').encode()
    (incoming / 'META.json').write_bytes(meta_bytes)
    (incoming / 'FINGERPRINT').write_text(hashlib.sha256(meta_bytes).hexdigest() + '\n')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(target, ignore_errors=True)
    os.replace(incoming, target)
    print(f'WIP slot reused: slot={args.slot} role={args.role} from={source} '
          f'fingerprint={hashlib.sha256(meta_bytes).hexdigest()}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    finder = sub.add_parser('find')
    finder.add_argument('--key', required=True)
    finder.add_argument('--role', required=True, choices=ROLES)
    finder.add_argument('--slot', required=True)
    finder.add_argument('--exclude-root', default='')
    adopter = sub.add_parser('adopt')
    adopter.add_argument('--key', required=True)
    adopter.add_argument('--role', required=True, choices=ROLES)
    adopter.add_argument('--source', required=True)
    adopter.add_argument('--root', required=True)
    adopter.add_argument('--slot', required=True)
    adopter.add_argument('--instance', default='')
    adopter.add_argument('--wip-root', default='')
    args = parser.parse_args()
    if len(args.key) != 64 or any(c not in '0123456789abcdef' for c in args.key):
        parser.error('--key must be a sha256 hex digest')
    return find(args) if args.command == 'find' else adopt(args)


if __name__ == '__main__':
    raise SystemExit(main())
