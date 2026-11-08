"""Install the export-cache read recorder in every Python process an export starts.

scripts/build/export-cache.py puts this directory first on PYTHONPATH for one
export command; site imports sitecustomize before any user code runs, in the
command's process and in every multiprocessing/subprocess Python child (venvs
included, unlike usercustomize). The interpreter's own sitecustomize, if any,
still runs: it is loaded from the rest of sys.path afterwards.
"""
import importlib.machinery
import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("CUTEAFD_EXPORT_CACHE_READS"):
    sys.path.insert(0, os.path.dirname(_here))
    try:
        import _export_cache
    finally:
        sys.path.pop(0)
    _export_cache.child_bootstrap()
    _export_cache.install_bookkeeping_pauses()

_rest = [entry for entry in sys.path if os.path.abspath(entry or ".") != _here]
_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _rest)
if _spec is not None and _spec.origin and os.path.abspath(_spec.origin) != os.path.abspath(__file__):
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
