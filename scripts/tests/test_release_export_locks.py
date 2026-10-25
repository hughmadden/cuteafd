"""CPU-only export locking, contention and daemon-owned container cleanup."""
import fcntl
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run_export(tmp_path, command='true', host='', locks='sparks,gpu1', preflight='', hook=''):
    lock_dir = tmp_path / '.cache/cuteafd'
    lock_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / 'docker'
    docker.write_text('''#!/usr/bin/env bash
set -eu
# Cleanup must run before any requested lock is released.
for name in ${STUB_LOCKS//,/ }; do
  flock -n "$HOME/.cache/cuteafd/$name.lock" true && exit 80
done
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
{hook}
release_with_export_locks '{host}' housekeeping-export '{locks}' "$PREFLIGHT" {command}
'''
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=10,
                          env={**os.environ, 'HOME': str(tmp_path), 'PATH': f'{bin_dir}:' + os.environ['PATH'],
                               'CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS': '1', 'PREFLIGHT': preflight,
                               'STUB_LOCKS': locks})


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
    # The fair queue (scripts/hw/hwlock.py) waits for every lock at once.
    assert 'timed out waiting for a fair turn on sparks.lock and gpu1.lock' in result.stderr
    assert not (tmp_path / 'cleanup.log').exists()


def test_cleanup_accepts_auto_removed_container(tmp_path, monkeypatch):
    monkeypatch.setenv('MOCK_ALREADY_REMOVED', '1')
    result = run_export(tmp_path)
    assert result.returncode == 0, result.stderr


def test_build_leases_hardware_only_for_the_native_export_phase():
    """`build.sh` holds no hardware lock while it compiles Rust, downloads,
    assembles images or distributes them: only each leg's native/AOT phase
    takes a fair hwlock lease (with preflight), between a lock-free Rust phase
    and image assembly, and its container is removed before release."""
    build = (ROOT / 'build.sh').read_text()
    assert 'exec 9>"$release_build_lock_dir/build.lock"' in build
    # The only flock in build.sh is the build lock; hardware leases go through the helper.
    assert [line.strip() for line in build.splitlines() if 'flock' in line and not line.strip().startswith('release_need')] == [
        'flock -w "$release_build_lock_timeout" 9 ||'
    ]
    assert 'source "$repo_root/scripts/lib/release-export-locks.sh"' in build
    assert build.index('if ((dry_run)); then') < build.index('exec 9>"$release_build_lock_dir/build.lock"')
    coord = build.split('build_coordinator_release() (', 1)[1].split('\nbuild_spark_release() (', 1)[0]
    rust = coord.index('-e CUTEAFD_RELEASE_ARTIFACT_PHASE=rust')
    pick = coord.index('export_gpu_pick="$(release_select_idle_export_gpu)" || exit 2')
    lease = coord.index('release_with_export_locks "" "$coordinator_export_container" "$export_gpu_lock" "$export_gpu_preflight"')
    image = coord.index('docker build')
    assert rust < pick < lease < image
    # The Rust phase gets no GPU; the native phase is the only GPU run and runs leased.
    rust_run = coord[coord.rindex('timeout --foreground', 0, rust):rust]
    assert 'CDI' not in rust_run and 'CUDA_VISIBLE_DEVICES' not in rust_run
    native = coord.split('release_coordinator_native_export() {', 1)[1].split('\n}\n', 1)[0]
    assert '"${CUTEAFD_CDI_ARGS[@]}"' in native and 'CUTEAFD_RELEASE_ARTIFACT_PHASE=native' in native
    assert 'release_watch_export_gpu "$export_gpu_uuid" "$coordinator_export_container" &' in native
    assert coord.count('CUTEAFD_CDI_ARGS') == 1
    spark = build.split('build_spark_release_leg dev\n', 1)[1].split('\nverify_remote_source_manifest', 1)[0]
    assert spark.index('build_spark_release_leg rust') < spark.index('release_with_export_locks "$seed_host" "$export_container-expert" "$spark_export_lock"')
    assert spark.index('  build_spark_release_leg export') < spark.index('build_spark_release_leg image')
    remote = build.split("<<'REMOTE'\n", 1)[1]
    remote_rust = remote.split('if [[ "$phase" == rust ]]; then', 1)[1].split('\nfi\n', 1)[0]
    assert 'CDI' not in remote_rust and 'CUTEAFD_RELEASE_ARTIFACT_PHASE=rust' in remote_rust
    assert 'trap cleanup_spark_phase EXIT' in build and "trap 'exit 129' HUP" in build
    assert "name=^cuteafd-spark-expert-[a-z0-9_.-]+-[0-9]+$" in build


@pytest.mark.parametrize('seed,lock', [('spark-9b14', 'sparks'), ('ostrich', 'sparks'),
                                       ('kakapo', 'kakapo-export'), ('cassowary', 'cassowary-export')])
def test_spark_export_lease_names_the_seed_lock(seed, lock):
    build = (ROOT / 'build.sh').read_text()
    fn = 'release_spark_export_lock() {' + build.split('release_spark_export_lock() {', 1)[1].split('\n}\n', 1)[0] + '\n}\n'
    result = subprocess.run(['bash', '-c', fn + f'release_spark_export_lock {seed}'], capture_output=True, text=True)
    assert result.stdout.strip() == lock


def test_preflight_runs_before_enqueue_and_at_grant_under_the_lock(tmp_path):
    log = tmp_path / 'preflight.log'
    lock = tmp_path / '.cache/cuteafd/gpu0.lock'
    preflight = f'if flock -n {lock} true; then echo free >>{log}; else echo held >>{log}; fi'
    result = run_export(tmp_path, 'true', locks='gpu0', preflight=preflight)
    assert result.returncode == 0, result.stderr
    assert log.read_text().split() == ['free', 'held']
    # Only the requested lease: no sparks/gpu1 lock file was even opened.
    assert not (tmp_path / '.cache/cuteafd/sparks.lock').exists()


def test_failed_preflight_runs_nothing_and_cleans_nothing(tmp_path):
    marker = tmp_path / 'ran'
    result = run_export(tmp_path, f'touch {marker}', locks='gpu0', preflight='false')
    assert result.returncode != 0
    assert 'export preflight failed for gpu0.lock' in result.stderr
    assert not marker.exists() and not (tmp_path / 'cleanup.log').exists()


@pytest.mark.parametrize('command,status', [('true', 0), ('false', 1), ("bash -c 'kill -TERM \"$PPID\"'", 143)])
def test_cleanup_hook_runs_before_container_removal_under_the_lock(tmp_path, command, status):
    hook = ('release_export_locked_cleanup() { flock -n "$HOME/.cache/cuteafd/gpu0.lock" true && exit 82; '
            'printf hook >>"$HOME/cleanup.log"; }')
    result = run_export(tmp_path, command, locks='gpu0', hook=hook)
    assert result.returncode == status, result.stderr
    assert (tmp_path / 'cleanup.log').read_text() == 'hookrm -f housekeeping-export\n'


def test_unknown_lock_is_refused_before_any_wait(tmp_path):
    result = run_export(tmp_path, 'true', locks='gpu2')
    assert result.returncode != 0 and 'unknown hardware locks: gpu2' in result.stderr


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


@pytest.mark.parametrize(
    'apps,cgroups,top,total,killed,warning,inspect_ok,query_ok',
    [
        # Parallel export contexts may exceed the limit without foreign work.
        ('101, 4500, GPU-export\n102, 4392, GPU-export',
         {101: 'own', 102: 'own'}, None, 8892, False, False, True, True),
        ('101, 8892, GPU-export\n201, 8200, GPU-export',
         {101: 'own', 201: 'foreign'}, None, 17092, True, False, True, True),
        # The limit applies to the sum of foreign processes, not each one.
        ('201, 4200, GPU-export\n202, 4200, GPU-export',
         {201: 'foreign', 202: 'foreign'}, None, 8400, True, False, True, True),
        ('101, 8892, GPU-export\n201, 8192, GPU-export',
         {101: 'own', 201: 'foreign'}, None, 17084, False, False, True, True),
        # Filter the UUID even if a driver returns rows from other devices.
        ('101, 8892, GPU-export\n201, 20000, GPU-other',
         {101: 'own', 201: 'foreign'}, None, 8892, False, False, True, True),
        # Unreadable/disappeared cgroups cannot prove foreign ownership.
        ('101, 8892, GPU-export', {}, None, 8892, False, True, True, True),
        ('101, N/A, GPU-export', {101: 'own'}, None,
         8892, False, True, True, True),
        ('101, 8892, GPU-export', {101: 'own'}, None,
         8892, False, True, True, False),
        ('', {}, None, 8892, False, False, True, True),
        ('101, 8892, GPU-export', {101: 'own-systemd'}, None,
         8892, False, False, True, True),
        # docker top provides host PIDs when inspect cannot identify cgroups.
        ('101, 8892, GPU-export', {}, 'PID\n101',
         8892, False, False, False, True),
        ('101, 8892, GPU-export\n201, 8200, GPU-export',
         {101: 'own', 201: 'foreign'}, 'PID\n101',
         17092, True, False, False, True),
        # A departed exporter absent from top must not be labeled foreign.
        ('101, 8892, GPU-export', {}, 'PID',
         8892, False, True, False, True),
        # Partial process-memory support: total minus known export use.
        ('101, 8892, GPU-export\n201, N/A, GPU-export',
         {101: 'own', 201: 'foreign'}, None, 17092, True, False, True, True),
        ('101, 8892, GPU-export\n201, N/A, GPU-export',
         {101: 'own', 201: 'foreign'}, None, 9000, False, False, True, True),
        # Never subtract incomplete export usage from a device-wide total.
        ('101, N/A, GPU-export\n201, N/A, GPU-export',
         {101: 'own', 201: 'foreign'}, None, 17092, False, True, True, True),
        ('101, 8892, GPU-export\n201, N/A, GPU-export\n301, 10, GPU-export',
         {101: 'own', 201: 'foreign'}, None, 17102, False, True, True, True),
    ],
    ids=[
        'export-only-above-limit', 'foreign-above-limit', 'foreign-sum',
        'foreign-at-limit', 'other-device-ignored', 'unattributable',
        'export-memory-unavailable', 'process-query-unavailable', 'no-process-rows',
        'systemd-scope-own',
        'docker-top-own', 'docker-top-foreign', 'departed-process',
        'fallback-kills', 'fallback-excludes-own', 'fallback-unknown-own',
        'fallback-unknown-owner',
    ],
)
def test_export_watchdog_attributes_foreign_memory(
    tmp_path, apps, cgroups, top, total, killed, warning, inspect_ok, query_ok,
):
    build = (ROOT / 'build.sh').read_text()
    guard = build.split('# release-aot-export-guard:start', 1)[1].split(
        '# release-aot-export-guard:end', 1,
    )[0]
    container_id = 'a' * 64
    proc = tmp_path / 'proc'
    proc.mkdir()
    for pid, owner in cgroups.items():
        entry = proc / str(pid)
        entry.mkdir()
        cgroup = {
            'own': f'0::/docker/{container_id}',
            'own-systemd': f'0::/system.slice/docker-{container_id}.scope',
            'foreign': '0::/other-job',
        }[owner]
        (entry / 'cgroup').write_text(cgroup + '\n')
    (tmp_path / 'apps').write_text(apps)
    (tmp_path / 'top').write_text(top or '')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    smi = bin_dir / 'nvidia-smi'
    smi.write_text('''#!/usr/bin/env bash
set -eu
[[ $2 == --format=csv,noheader,nounits && $3 == -i && $4 == GPU-export ]] || exit 90
case $1 in
  --query-compute-apps=pid,used_memory,gpu_uuid)
    [[ $QUERY_OK == 1 ]] || exit 1
    cat "$FIXTURE/apps" ;;
  --query-gpu=memory.used) printf '%s\\n' "$TOTAL" ;;
  *) exit 91 ;;
esac
''')
    smi.chmod(0o755)
    docker = bin_dir / 'docker'
    docker.write_text('''#!/usr/bin/env bash
set -eu
case $1 in
  inspect)
    [[ $INSPECT_OK == 1 ]] || exit 1
    printf '%s\\n' "$CONTAINER_ID" ;;
  top)
    [[ $TOP_OK == 1 ]] || exit 1
    cat "$FIXTURE/top" ;;
  rm) printf '%s\\n' "$*" >>"$FIXTURE/killed" ;;
  *) exit 92 ;;
esac
''')
    docker.chmod(0o755)
    script = f'''set -euo pipefail
{guard}
release_export_gpu_poll_seconds=15
export_gpu_limit_mib=8192
polls=0
sleep() {{ polls=$((polls + 1)); ((polls <= 3)); }}
release_watch_export_gpu GPU-export export-fixture "$FIXTURE/proc"
'''
    result = subprocess.run(
        ['bash', '-c', script], capture_output=True, text=True, timeout=5,
        env={
            **os.environ, 'PATH': f'{bin_dir}:' + os.environ['PATH'],
            'FIXTURE': str(tmp_path), 'CONTAINER_ID': container_id,
            'TOTAL': str(total), 'INSPECT_OK': str(int(inspect_ok)),
            'TOP_OK': str(int(top is not None)), 'QUERY_OK': str(int(query_ok)),
        },
    )
    assert result.returncode == int(killed), result.stderr
    assert (tmp_path / 'killed').exists() == killed
    if killed:
        assert (tmp_path / 'killed').read_text() == 'rm -f export-fixture\n'
        assert 'foreign process use' in result.stderr
    assert result.stderr.count('WARNING:') == int(warning), result.stderr
    incomplete_count = 3 if warning else 0
    assert result.stderr.count('AOT export GPU watchdog (GPU-export):') == 1, result.stderr
    assert f'attribution incomplete {incomplete_count} times' in result.stderr


@pytest.mark.parametrize('stop', ['return', 'term', 'background'])
def test_export_watchdog_summary_reaches_console_outside_leg_log(tmp_path, stop):
    build = (ROOT / 'build.sh').read_text()
    guard = build.split('# release-aot-export-guard:start', 1)[1].split(
        '# release-aot-export-guard:end', 1,
    )[0]
    script = f'''set -euo pipefail
{guard}
release_export_foreign_gpu_mib() {{ printf '0 1\\n'; }}
release_export_gpu_poll_seconds=15
export_gpu_limit_mib=8192
polls=0
sleep() {{
  polls=$((polls + 1))
  if ((polls > 3)); then
    if [[ {stop} == background ]]; then
      : >"$LEG_LOG.ready"
      command sleep 0.1
    fi
    [[ {stop} != term ]] || kill -TERM "$BASHPID"
    return 1
  fi
}}
exec {{release_export_summary_fd}}>&2
if [[ {stop} == background ]]; then
  release_watch_export_gpu GPU-export export-fixture >"$LEG_LOG" 2>&1 &
  export_watchdog_pid=$!
  until [[ -e "$LEG_LOG.ready" ]]; do command sleep 0.01; done
  release_stop_export_watchdog
else
  release_watch_export_gpu GPU-export export-fixture >"$LEG_LOG" 2>&1
fi
'''
    log = tmp_path / 'coordinator.log'
    result = subprocess.run(
        ['bash', '-c', script], capture_output=True, text=True, timeout=5,
        env={**os.environ, 'LEG_LOG': str(log)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == (
        'AOT export GPU watchdog (GPU-export): attribution incomplete 3 times\n'
    )
    assert log.read_text().count('WARNING:') == 1
    assert 'times' not in log.read_text()
    supervisor = build.split('release_build_legs() {', 1)[1].split('\n}', 1)[0]
    assert supervisor.index('exec {release_export_summary_fd}>&2') < supervisor.index(
        'build_supervise',
    )


def test_term_stops_the_export_before_releasing_locks(tmp_path):
    result = run_export(tmp_path, "bash -c 'kill -TERM \"$PPID\"'")
    assert result.returncode == 143, result.stderr
    assert (tmp_path / 'cleanup.log').read_text().strip() == 'rm -f housekeeping-export'
