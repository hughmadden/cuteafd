#!/usr/bin/env python3
"""Run inside a dev container with /source:ro and a writable HOME/cache mount."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, default=Path('/source'))
p.add_argument('--jit', action='store_true', help='compile native loader, GDS, RoCE and allocator helpers (no GPU needed)')
a = p.parse_args()
root = a.source.resolve()
subprocess.run([sys.executable, str(root / 'scripts/build/verify-sparkinfer-source.py'),
                '--source', str(root / 'third_party/sparkinfer'),
                '--lock', str(root / 'third_party/sparkinfer.lock.json')], check=True)
package = root / 'third_party/sparkinfer'
assert os.statvfs(package).f_flag & os.ST_RDONLY, 'smoke requires a read-only mount'
home = Path(os.environ['HOME'])
for key, leaf in [('TORCH_EXTENSIONS_DIR', 'torch-extensions'), ('XDG_CACHE_HOME', '.cache'),
                  ('B12X_ROCE_CACHE_DIR', 'roce'), ('TRITON_CACHE_DIR', 'triton'),
                  ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor')]:
    cache = Path(os.environ.setdefault(key, str(home / leaf))).resolve()
    assert not cache.is_relative_to(root), (key, cache)
    cache.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(package))
import b12x
assert Path(b12x.__file__).resolve().is_relative_to(package)
import b12x.loader
import b12x.preparation
import b12x.comm.roce
if a.jit:
    from b12x.loader import _native, _gds_native
    from b12x.comm.roce import _proxy
    from b12x.preparation import _memory
    for output in (_native._build(), _proxy._build()):
        assert Path(output).is_relative_to(home), output
    # NGC's default cufile.json contains comments; use strict JSON for this smoke.
    cufile = home / 'cufile-smoke.json'
    cufile.write_text('{}\n')
    os.environ['CUFILE_ENV_PATH_JSON'] = str(cufile)
    module = _gds_native.load()
    assert Path(module.__file__).is_relative_to(home), module.__file__
    module = _memory._counter()
    assert Path(module.__file__).is_relative_to(home), module.__file__
print('b12x read-only source smoke passed' + (' (all native JIT helpers)' if a.jit else ''))
