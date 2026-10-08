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


def test_build_takes_a_cpu_lock_and_only_guards_its_export_gpus():
    """`build.sh` must not hold a hardware lock while it compiles, downloads,
    assembles images or exports: the build's serialization is its own CPU lock,
    and each AOT export borrows one idle GPU behind a guard instead."""
    build = (ROOT / 'build.sh').read_text()
    assert 'exec 9>"$release_build_lock_dir/build.lock"' in build
    assert 'release_build_lock_dir="$HOME/.cache/cuteafd"' in build
    # The only flock in the build is the build lock: no sparks.lock/gpu1.lock,
    # no hardware-lock helper, and therefore no wait behind or for a measurement.
    assert [line.strip() for line in build.splitlines() if 'flock' in line and not line.strip().startswith('release_need')] == [
        'flock -w "$release_build_lock_timeout" 9 ||'
    ]
    assert 'release_with_export_locks' not in build
    assert 'release-export-locks.sh' not in build
    assert build.index('if ((dry_run)); then') < build.index('exec 9>"$release_build_lock_dir/build.lock"')
    # The coordinator export picks an idle device (<512 MiB used), pins it by
    # UUID, watches it, and cleans its container up inline (the old cleanup rode
    # on the lock helper's traps).
    assert 'export_gpu_pick="$(release_select_idle_export_gpu)" || exit 2' in build
    assert '--gpus "device=$export_gpu_uuid"' in build
    assert '-e "CUDA_VISIBLE_DEVICES=$export_gpu_uuid"' in build
    assert 'release_watch_export_gpu "$export_gpu_uuid" "$coordinator_export_container" &' in build
    assert 'timeout "$export_timeout" --foreground docker run --rm --name "$coordinator_export_container"' in build
    assert "trap 'exit 143' TERM" in build
    assert "trap 'release_stop_export_watchdog; release_export_cleanup; rm -rf \"$release_source_dir\"' EXIT" in build
    # The Spark export is guarded on the host: no serving worker, >=100 GiB
    # CUDA-free, an ssh-side EXIT/HUP cleanup and a contention watchdog.
    assert 'build_spark_release_leg export' in build
    assert 'timeout "$export_timeout" --foreground ssh "${release_ssh_opts[@]}" "$seed_host"' in build
    assert 'trap cleanup_spark_phase EXIT' in build
    assert "trap 'exit 129' HUP" in build
    assert "name=^cuteafd-spark-expert-[a-z0-9_.-]+-[0-9]+$" in build
    assert 'CUTEAFD_RELEASE_SPARK_MIN_FREE_GIB' in build
    helper = (ROOT / 'scripts/lib/release-export-locks.sh').read_text()
    assert helper.index('flock -w "$wait_seconds" 9') < helper.index('flock -w "$wait_seconds" 8')


def test_idle_export_selector_keeps_wait_status_off_stdout(tmp_path):
    build = (ROOT / 'build.sh').read_text()
    selector = build.split('release_select_idle_export_gpu() {', 1)[1].split('\n}\n', 1)[0]
    state = tmp_path / 'idle'
    script = f'''set -euo pipefail
release_idle_wait_seconds=30
release_idle_gpu_limit_mib=512
release_die() {{ printf '%s\\n' "$*" >&2; exit 2; }}
nvidia-smi() {{
  if [[ $1 == --query-gpu=index,uuid,memory.used ]]; then
    if [[ -e '{state}' ]]; then
      printf '0, GPU-busy, 39222\\n1, GPU-idle, 12\\n'
    else
      printf '0, GPU-busy, 39222\\n1, GPU-idle, 45194\\n'
    fi
  else
    printf '0, 39222\\n1, 45194\\n'
  fi
}}
sleep() {{ [[ $1 == 15 ]]; touch '{state}'; }}
release_select_idle_export_gpu() {{{selector}
}}
pick="$(release_select_idle_export_gpu)"
read -r index uuid used <<<"$pick"
[[ $index == 1 && $uuid == GPU-idle && $used == 12 ]]
printf '%s\\n' "$pick"
'''
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert state.exists(), 'fixture must exercise the waiting path'
    assert result.stdout == '1 GPU-idle 12\n'
    assert 'waiting for an idle RTX' in result.stderr


def test_term_stops_the_export_before_releasing_locks(tmp_path):
    result = run_export(tmp_path, "bash -c 'kill -TERM \"$PPID\"'")
    assert result.returncode == 143, result.stderr
    assert (tmp_path / 'cleanup.log').read_text().strip() == 'rm -f housekeeping-export'
