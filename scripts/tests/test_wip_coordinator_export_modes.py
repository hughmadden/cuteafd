"""wip.sh's coordinator export: off-queue modes first (remote export host,
capped beside serving), the fair GPU turn last, falling through on exit 75."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WIP = (ROOT / 'wip.sh').read_text()


def _function(name):
    start = WIP.index(name + '() {')
    return WIP[start:WIP.index('\n}\n', start) + 3]


def _dispatch():
    start = WIP.index('    local mode exported=0')
    return WIP[start:WIP.index('    ((exported)) || release_die', start)]


@pytest.mark.parametrize('setting,host,expected', [
    ('auto', '', 'capped turn'),
    ('auto', 'oliver', 'remote capped turn'),
    ('remote', '', 'remote capped turn'),
    ('capped', 'oliver', 'capped turn'),
    ('turn', 'oliver', 'turn'),
])
def test_mode_order(setting, host, expected):
    script = 'release_die() { echo "DIE $*"; exit 9; }\n' + _function('wip_coordinator_export_modes') + \
        f'CUTEAFD_WIP_COORDINATOR_EXPORT={setting} CUTEAFD_EXPORT_HOST={host} wip_coordinator_export_modes\n'
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split() == expected.split()


def test_unknown_mode_is_refused():
    script = 'release_die() { echo "DIE $*"; exit 9; }\n' + _function('wip_coordinator_export_modes') + \
        'CUTEAFD_WIP_COORDINATOR_EXPORT=bogus wip_coordinator_export_modes\n'
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 9 and 'must be auto, remote, capped or turn' in result.stdout


@pytest.mark.parametrize('remote,capped,expected_done,expected_calls', [
    (0, 0, 'remote', ['remote']),
    (75, 0, 'capped', ['remote', 'capped']),
    (75, 75, 'turn', ['remote', 'capped', 'turn']),
])
def test_dispatch_falls_through_to_the_fair_turn(tmp_path, remote, capped, expected_done, expected_calls):
    trace = tmp_path / 'trace'
    script = f'''
release_die() {{ echo "DIE $*"; exit 9; }}
wip_coordinator_export_modes() {{ printf 'remote capped turn\\n'; }}
wip_remote_export() {{ echo remote >> {trace}; return {remote}; }}
wip_capped_export() {{ echo capped >> {trace}; return {capped}; }}
wip_hwlock() {{ echo turn >> {trace}; }}
wip_export_gpu_lock=gpu1 coordinator_build=(docker exec c)
run() {{
{_dispatch()}
}}
run
'''
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'done by {expected_done}' in result.stdout
    assert trace.read_text().split() == expected_calls


def test_capped_export_takes_no_hardware_lock_and_is_watched():
    capped = _function('wip_capped_export')
    assert 'wip_hwlock' not in capped and 'flock' not in capped
    assert 'export-placement.py' in WIP and '"$placement" admit' in capped
    assert 'watch --gpu "$gpu" --cap-mib "$cap" --cgroup "$id"' in capped
    # The GPU with the most free memory at that moment, read from the host smi.
    assert "sort -t, -k2 -nr | head -1" in capped
    remote = _function('wip_remote_export')
    assert 'wip_hwlock' not in remote and 'CUTEAFD_WIP_NATIVE_REQUIRE_HIT=1' in remote


def test_remote_export_admits_from_the_hosts_wsl_smi_never_the_container():
    remote = (ROOT / 'scripts/build/export-remote.sh').read_text()
    assert 'smi=/usr/lib/wsl/lib/nvidia-smi' in remote
    admit = remote.index('"$placement" admit --smi "$smi"')
    container = remote.index('docker run --rm --gpus all --user')
    assert admit < container  # admission precedes the export container
    assert 'mem_get_info' not in remote
    # Bulk data only through sparknest, never one ssh pipe.
    assert 'nest replicate --hosts "$host"' in remote and 'wip-native/x86_64/coordinator-120' in remote
    # Interim #122 guard, marked for removal.
    assert 'remove once #122 lands' in remote and 'sm-check' in remote
