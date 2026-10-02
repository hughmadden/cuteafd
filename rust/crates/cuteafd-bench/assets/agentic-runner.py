# Test runner of the agentic benchmark (from scripts/bench/bench-agentic-session.py).
import importlib.util, sys, traceback
from pathlib import Path
root = Path(sys.argv[1]).resolve()
selection = sys.argv[2] if len(sys.argv) > 2 else ""
sys.path.insert(0, str(root))
sys.dont_write_bytecode = True
target, _, only = selection.partition("::")
files = sorted(p for p in (root / "tests").glob("test_*.py"))
if target:
    files = [p for p in files if p.relative_to(root).as_posix() == target.strip().lstrip("./")]
    if not files:
        print(f"no test file matches {selection!r}")
        sys.exit(4)
passed = failed = 0
def where(tb):
    frames = [f for f in traceback.extract_tb(tb) if f.filename.startswith(str(root))]
    return "; ".join(f"{Path(f.filename).relative_to(root).as_posix()}:{f.lineno}: {f.line}" for f in frames[-3:])
for path in files:
    rel = path.relative_to(root).as_posix()
    spec = importlib.util.spec_from_file_location("tests." + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException as error:
        failed += 1
        print(f"ERROR {rel} - {type(error).__name__}: {error} [{where(error.__traceback__)}]")
        continue
    for name in sorted(n for n in vars(module) if n.startswith("test_") and callable(vars(module)[n])):
        if only and name != only:
            continue
        try:
            vars(module)[name]()
        except BaseException as error:
            failed += 1
            message = str(error).splitlines()[0] if str(error) else ""
            print(f"FAILED {rel}::{name} - {type(error).__name__}: {message} [{where(error.__traceback__)}]")
        else:
            passed += 1
            print(f"PASSED {rel}::{name}")
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
