#!/usr/bin/env python3
"""Print `sha256  path` for every file that is an input of the WIP native build.

The tree is native/, python/ and scripts/build/, minus the patterns that
scripts/build/kernel-input-rules.json declares are never kernel inputs (tests,
benchmarks, docs, and the records/comparison/placement tooling the native build
never executes), plus the Rust sources CMake and the exporters read for shared
constants (EXTERNAL_INPUTS). Matching uses fnmatch, as kernel-records.py does.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
from pathlib import Path
import sys

TREES = ('native', 'python', 'scripts/build')
# Read by native/CMakeLists.txt (V4.1 concurrency and draft capacities, MXFP4
# policy inputs) and by python/tools/lib (v41_capacities.py, mxfp4_layouts.py).
# scripts/tests/test_wip_scope_exports.py requires every such reference here.
EXTERNAL_INPUTS = (
    'rust/crates/cuteafd-core/src/concurrency.rs',
    'rust/crates/cuteafd-core/src/dspark_verify.rs',
    'rust/crates/cuteafd-core/src/serving_capacity.rs',
)


def keyed_files(root: Path):
    ignore = json.loads((root / 'scripts/build/kernel-input-rules.json').read_text())['ignore']
    for tree in TREES:
        for path in (root / tree).rglob('*'):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and not any(fnmatch.fnmatch(relative, pattern) for pattern in ignore):
                yield relative
    for relative in EXTERNAL_INPUTS:
        if not (root / relative).is_file():
            raise SystemExit(f'native key input is missing: {relative}')
        yield relative


def main():
    root = Path(sys.argv[1]).resolve()
    for relative in sorted(keyed_files(root), key=lambda name: name.encode()):
        print(hashlib.sha256((root / relative).read_bytes()).hexdigest() + '  ' + relative)


if __name__ == '__main__':
    main()
