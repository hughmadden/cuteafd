"""Script tests start their shells without the runner's shell startup files.

A test runner inside an NGC-based image inherits BASH_ENV (and ENV), whose startup code
runs `nvidia-smi -q -d COMPUTE | grep "^CUDA Version" | sed ...`. With a bash-script
`nvidia-smi` stub first on PATH, as the launcher tests use, every stub call re-entered
itself through that startup code without end. conftest.py removes both variables for every
test. These tests run a stubbed launch in a child pytest that starts with such a BASH_ENV,
once with that fixture and once without it.

Both runs are safe on any host: the stub counts its own nesting and refuses to go deeper
than 2, whatever the fixture does, and every child process has a short timeout.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Stands in for the image's /etc/bash.bashrc -> /etc/shinit_v2: it asks for the CUDA version.
STARTUP = """_RUNNING_CUDA_VERSION="$(nvidia-smi -q -d COMPUTE 2>/dev/null | grep "^CUDA Version" | sed 's/^.*: //')"
"""

# The stub. Its guard is POSIX sh, for which no shell runs a startup file when it is not
# interactive, so the guard always runs first: it logs the depth and refuses past 2. Then
# it answers from bash, which runs $BASH_ENV first, like the launcher tests' stubs.
STUB = """#!/bin/sh
depth=$(( ${STUB_DEPTH:-0} + 1 ))
echo "$depth" >> "$STUB_LOG"
if [ "$depth" -gt 2 ]; then
    echo "nvidia-smi stub: re-entered $depth deep, refusing" >&2
    exit 97
fi
export STUB_DEPTH="$depth"
exec bash -c 'printf "%s\\n" 0'
"""

# The stubbed launch: a bash command that lists the GPUs through the stub.
CHILD = '''
import os
import subprocess


def test_stubbed_launch():
    env = {**os.environ, "PATH": f"{os.environ['STUB_BIN']}:{os.environ['PATH']}"}
    result = subprocess.run(["bash", "-c", "nvidia-smi --query-gpu=index --format=csv,noheader"],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "0\\n"
'''


def _launch_under_startup_file(tmp_path: Path, *pytest_args: str) -> tuple[subprocess.CompletedProcess[str], list[int]]:
    """Run CHILD in a pytest whose environment names STARTUP in BASH_ENV and ENV; return
    the result and the stub depths that were reached."""
    startup = tmp_path / "bash.bashrc"
    startup.write_text(STARTUP)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text(STUB)
    (bin_dir / "nvidia-smi").chmod(0o755)
    (tmp_path / "test_child.py").write_text(CHILD)
    log = tmp_path / "depths.log"
    log.touch()
    env = {**os.environ, "BASH_ENV": str(startup), "ENV": str(startup), "STUB_BIN": str(bin_dir),
           "STUB_LOG": str(log), "PYTHONPATH": str(HERE), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *pytest_args,
                             "test_child.py"], cwd=tmp_path, env=env, capture_output=True, text=True,
                            timeout=60)
    return result, sorted({int(line) for line in log.read_text().split()})


def test_stubbed_launch_runs_no_shell_startup_file(tmp_path: Path) -> None:
    result, depths = _launch_under_startup_file(tmp_path, "-p", "conftest")
    assert result.returncode == 0, result.stdout + result.stderr
    assert depths == [1], f"the stub re-entered itself through BASH_ENV: depths {depths}"


def test_stub_stops_its_own_recursion_without_the_fixture(tmp_path: Path) -> None:
    # The control: without conftest.py the startup file runs and re-enters the stub, which
    # refuses at depth 3. The launch still answers, so only the depth log shows it.
    result, depths = _launch_under_startup_file(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert depths == [1, 2, 3]
