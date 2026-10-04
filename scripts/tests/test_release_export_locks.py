"""CPU-only export locking, contention and daemon-owned container cleanup."""
import fcntl
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run_export(tmp_path, command='true', host=''):
    lock_dir = tmp_path / '.cache/cuteafd'
    lock_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / 'docker'
    docker.write_text('''#!/usr/bin/env bash
set -eu
# Cleanup must run before either lock is released.
flock -n "$HOME/.cache/cuteafd/sparks.lock" true && exit 80
flock -n "$HOME/.cache/cuteafd/gpu1.lock" true && exit 81
printf '%s\\n' "$*" >>"$HOME/cleanup.log"
if [[ ${MOCK_ALREADY_REMOVED:-0} == 1 ]]; then
  echo "Error response from daemon: No such container: housekeeping-export" >&2
  exit 1
fi
''')
    docker.chmod(0o755)
    ssh = bin_dir / 'ssh'
    ssh.write_text('#!/usr/bin/env bash\n[[ $1 == ostrich ]] || exit 90\nshift\nexec "$@"\n')
    ssh.chmod(0o755)
    script = f'''source "{ROOT}/scripts/lib/release-common.sh"
source "{ROOT}/scripts/lib/release-export-locks.sh"
release_ssh_opts=()
release_with_export_locks '{host}' housekeeping-export {command}
'''
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=10,
                          env={**os.environ, 'HOME': str(tmp_path), 'PATH': f'{bin_dir}:' + os.environ['PATH'],
                               'CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS': '1'})


@pytest.mark.parametrize('host', ['', 'ostrich'])
@pytest.mark.parametrize('command,status', [('true', 0), ('false', 1), ('timeout 0.1 sleep 5', 124)])
def test_export_always_cleans_container_under_both_locks(tmp_path, command, status, host):
    result = run_export(tmp_path, command, host)
    assert result.returncode == status, result.stderr
    assert (tmp_path / 'cleanup.log').read_text().strip() == 'rm -f housekeeping-export'
    for name in ('sparks', 'gpu1'):
        with (tmp_path / f'.cache/cuteafd/{name}.lock').open('w') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize('held', ['sparks', 'gpu1'])
def test_lock_contention_times_out_without_export_or_cleanup(tmp_path, held):
    lock_dir = tmp_path / '.cache/cuteafd'
    lock_dir.mkdir(parents=True)
    with (lock_dir / f'{held}.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        result = run_export(tmp_path)
    assert result.returncode != 0
    assert f'timed out waiting for {held}.lock' in result.stderr
    assert not (tmp_path / 'cleanup.log').exists()


def test_cleanup_accepts_auto_removed_container(tmp_path, monkeypatch):
    monkeypatch.setenv('MOCK_ALREADY_REMOVED', '1')
    result = run_export(tmp_path)
    assert result.returncode == 0, result.stderr


def test_build_wraps_both_gpu_legs_and_uses_unique_names():
    build = (ROOT / 'build.sh').read_text()
    assert 'release_with_export_locks "" "$export_container-coordinator"' in build
    assert 'timeout "$export_timeout" docker run --rm --name "$export_container-coordinator"' in build
    assert 'release_with_export_locks "$seed_host" "$export_container-expert" build_spark_release_leg export' in build
    assert 'timeout "$export_timeout" ssh "${release_ssh_opts[@]}" "$seed_host"' in build
    helper = (ROOT / 'scripts/lib/release-export-locks.sh').read_text()
    assert helper.index('flock -w "$wait_seconds" 9') < helper.index('flock -w "$wait_seconds" 8')


def test_term_stops_the_export_before_releasing_locks(tmp_path):
    result = run_export(tmp_path, "bash -c 'kill -TERM \"$PPID\"'")
    assert result.returncode == 143, result.stderr
    assert (tmp_path / 'cleanup.log').read_text().strip() == 'rm -f housekeeping-export'
