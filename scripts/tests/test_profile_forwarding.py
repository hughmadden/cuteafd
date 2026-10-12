"""PROFILE=... forwards one set of profiling switches to the coordinator and every
Spark worker in every family (run-family.sh) and on the V4.1 path (run.sh)."""
import socket
import subprocess
from pathlib import Path

import pytest

from test_run_family_dispatch import GLMF_TWO_LAYER, _family_launch_result

ROOT = Path(__file__).resolve().parents[2]
HOST = socket.gethostname().split('.')[0]


def _helper(*args):
    script = 'source scripts/lib/release-common.sh; release_profile_docker_args "$@"; printf "%s\\n" ${RELEASE_PROFILE_ARGS[@]+"${RELEASE_PROFILE_ARGS[@]}"}'
    return subprocess.run(['bash', '-c', script, 'test', *args], cwd=ROOT, capture_output=True, text=True)


def test_helper_maps_switches_host_and_directory():
    out = _helper('census,transport-timing,expert-timing', '/data/prof', 'kiwi')
    assert out.returncode == 0, out.stderr
    args = out.stdout.split('\n')
    for flag in ('CUTEAFD_GRAPH_CENSUS=1', 'CUTEAFD_PROTOCOL_V2_TCP_TIMING=1', 'CUTEAFD_EXPERT_TIMING=1',
                 'CUTEAFD_PROFILE_HOST=kiwi', 'CUTEAFD_PROFILE_DIR=/data/prof', '/data/prof:/data/prof'):
        assert flag in args
    assert _helper('', '', 'kiwi').stdout == '\n'


@pytest.mark.parametrize('profile,directory,message', [
    ('census,flamegraph', '', 'unknown PROFILE switch flamegraph'),
    ('census;rm', '', 'PROFILE must be a comma list'),
    ('', '/data/prof', 'PROFILE_DIR needs PROFILE'),
    ('census', 'relative', 'PROFILE_DIR must be an absolute directory'),
])
def test_helper_refuses_bad_values(profile, directory, message):
    out = _helper(profile, directory, 'kiwi')
    assert out.returncode != 0 and message in out.stderr


def test_family_launch_forwards_profile_to_coordinator_and_each_worker(tmp_path):
    prof = tmp_path / 'prof'
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, 'test/glmf',
                                   f'GLM5_FLASH_FP8_MODEL_ID=off\nPROFILE=transport-timing,expert-timing\nPROFILE_DIR={prof}\n')
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    coordinator = next(line for line in lines if 'cuteafd serve-glmf' in line)
    workers = [line for line in lines if 'cuteafd expertd-native' in line]
    assert workers
    for line, host in [(coordinator, HOST), *[(w, 'h0') for w in workers]]:
        for flag in ('CUTEAFD_PROTOCOL_V2_TCP_TIMING=1', 'CUTEAFD_EXPERT_TIMING=1', f'CUTEAFD_PROFILE_HOST={host}',
                     f'CUTEAFD_PROFILE_DIR={prof}', f'{prof}:{prof}'):
            assert flag in line, (flag, line)
    assert all(f'mkdir -p {prof} && release_require_writable_state {prof} && docker run' in w for w in workers)
    assert prof.is_dir()


def test_family_launch_without_profile_adds_nothing(tmp_path):
    result = _family_launch_result(tmp_path, GLMF_TWO_LAYER, 'test/glmf', 'GLM5_FLASH_FP8_MODEL_ID=off\n')
    assert result.returncode == 0, result.stderr
    assert 'CUTEAFD_PROFILE' not in result.stderr and 'CUTEAFD_EXPERT_TIMING' not in result.stderr


def test_v41_worker_takes_profile_args_after_every_positional():
    source = (ROOT / 'run.sh').read_text()
    invocation, remote = source.split('echo "== starting native Spark experts =="', 1)[1].split("<<'REMOTE' &", 1)
    remote = remote.split('\nREMOTE', 1)[0]
    assert '--profile-args' in invocation
    assert '"${rdma_args[@]}" "${wip_args[@]}" ${profile_args[@]+"${profile_args[@]}"}' in remote
    positional = ['image', 'name', '1', '4096', '107374182400', '19441', 'snapshot', 'fingerprint', '3', '4',
                  '0', '', '', '', '', '', 'info', '__none__']
    for tail, expected in [([], []), (['--profile-args', '-e', 'CUTEAFD_EXPERT_TIMING=1', '-e', 'CUTEAFD_PROFILE_HOST=emu'],
                                      ['CUTEAFD_EXPERT_TIMING=1', 'CUTEAFD_PROFILE_HOST=emu'])]:
        result = subprocess.run(['bash', '-euc', 'source scripts/lib/release-common.sh; docker() { printf "%s\\n" "$@"; }\n' + remote.replace(' >/dev/null', ''),
                                 'test', *positional, *tail], cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        arguments = result.stdout.splitlines()
        assert arguments[arguments.index('--world') + 1] == '4'
        for flag in expected:
            assert flag in arguments
        assert ('CUTEAFD_PROFILE_HOST=emu' in arguments) == bool(expected)
    coordinator = source.split('start_coordinator() {', 1)[1].split('\n}\n', 1)[0] if 'start_coordinator() {' in source else source
    assert '${profile_args[@]+"${profile_args[@]}"}' in coordinator
