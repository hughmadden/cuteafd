"""Session-owned, immutable launcher inputs; writable files never share inodes."""
from pathlib import Path
import shutil

_ROOT = None
_TEMPLATES = {}


def configure(root):
    global _ROOT
    _ROOT = root
    _TEMPLATES.clear()


def copy_input(source, destination):
    """Link a read-only input from a private session copy, not the checkout."""
    source, destination = Path(source), Path(destination)
    if destination.is_dir():
        destination /= source.name
    if _ROOT is None:
        return shutil.copy(source, destination)
    key = source.resolve()
    if key not in _TEMPLATES:
        cached = _ROOT / str(len(_TEMPLATES))
        shutil.copy(source, cached)
        _TEMPLATES[key] = cached
    destination.hardlink_to(_TEMPLATES[key])
    return str(destination)


def stub_cuda_capacity(common, total_bytes):
    """Launcher captures fake the host; driver/UUID behavior has its own unit tests."""
    common = Path(common)
    source = common.read_text()
    common.unlink()  # Cached inputs may be hardlinked into other captures.
    common.write_text(source + '\nrelease_cuda_total_bytes() { printf "%s\\n" '
                      + str(int(total_bytes)) + '; }\n')


def stub_cuda_python(bin_dir, total_bytes):
    """Intercept only the driver query in captures that launch the live checkout."""
    import sys
    wrapper = Path(bin_dir) / 'python3'
    wrapper.write_text('#!' + sys.executable + '\n'
        'import subprocess, sys\n'
        'args = sys.argv[1:]\n'
        'source = sys.stdin.read() if args and args[0] == "-" else None\n'
        'if source is not None and "cuDeviceTotalMem_v2" in source:\n'
        f'    print({int(total_bytes)})\n'
        'else:\n'
        f'    raise SystemExit(subprocess.run([{sys.executable!r}, *args], input=source, text=True).returncode)\n')
    wrapper.chmod(0o755)


def copy_tree(source, destination, **kwargs):
    """Build the tree once per worker; give each capture its own directory entries.

    Source registries, launch scripts and verifier are edited by capture tests,
    so those use real copies. All other files are read-only inputs.
    """
    if _ROOT is None:
        return shutil.copytree(source, destination, **kwargs)
    key = ('tree', Path(source).resolve())
    if key not in _TEMPLATES:
        cached = _ROOT / str(len(_TEMPLATES))
        shutil.copytree(source, cached, **kwargs)
        _TEMPLATES[key] = cached

    def clone(src, dst):
        path = Path(src)
        if path.suffix == '.sh' or path.name == 'verify-sparkinfer-source.py':
            return shutil.copy2(src, dst)
        Path(dst).hardlink_to(src)
        return dst

    return shutil.copytree(_TEMPLATES[key], destination, symlinks=True, copy_function=clone)
