#!/usr/bin/env python3
"""CPU-only proof of the deployment's exact Spark packages, before startup."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def verify(root, requirements, identity):
    seal = root.parent.parent / "META.json"
    if seal.is_file():
        identity += " seal " + hashlib.sha256(seal.read_bytes()).hexdigest()
    for package, layouts in requirements.items():
        for layout in layouts:
            directory = root / package / layout
            library = 'libcuteafd_exl3.so' if package.startswith('exl3/') else 'libcuteafd_fp8moe.so'
            if not (directory / library).is_file():
                raise ValueError(f'{identity}: missing Spark package {package}/{layout}')


def main():
    try:
        requirements, identity, *roots = sys.argv[1:]
        verify(Path(roots[0] if roots else '/opt/cuteafd/lib').expanduser(), json.loads(requirements), identity)
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
