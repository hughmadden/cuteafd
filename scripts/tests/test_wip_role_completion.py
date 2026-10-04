"""Fresh single-role builds succeed; paired and cloned slots stay strict."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def check_readiness(tmp_path, role, present, from_slot=''):
    text = (ROOT / 'wip.sh').read_text()
    block = text.split('# wip-slot-readiness:start', 1)[1].split('# wip-slot-readiness:end', 1)[0]
    slots = tmp_path / 'slots'
    for part in present:
        workspace = slots / 'test' / part / 'workspace'
        workspace.mkdir(parents=True)
        (workspace / 'cuteafd.config').write_text('config')
        (workspace.parent / 'FINGERPRINT').write_text('fingerprint')
    bins = tmp_path / 'bin'
    bins.mkdir()
    docker = bins / 'docker'
    docker.write_text('''#!/usr/bin/env python3
import os, subprocess, sys
script = sys.stdin.read().replace('/wip/slots', os.environ['MOCK_SLOTS'])
sys.exit(subprocess.run(['bash', '-s', '--', 'test'], input=script, text=True).returncode)
''')
    docker.chmod(0o755)
    ssh = bins / 'ssh'
    ssh.write_text('#!/usr/bin/env bash\nshift 3\nexec "$@"\n')
    ssh.chmod(0o755)
    return subprocess.run(['bash', '-c', 'set -euo pipefail\nrole=$1; from_slot=$2; slot=test; '
                           'coordinator_container=fixture; spark_container=fixture; seed_host=fixture\n' + block,
                           'test', role, from_slot], capture_output=True, text=True, timeout=10,
                          env={**os.environ, 'MOCK_SLOTS': str(slots), 'PATH': str(bins)+':'+os.environ['PATH']})


@pytest.mark.parametrize('role', ['coordinator', 'expert'])
def test_fresh_role_only_build_needs_only_selected_artifacts(tmp_path, role):
    present = [role if role == 'coordinator' else 'spark-expert']
    result = check_readiness(tmp_path, role, present)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('role', ['coordinator', 'expert', 'both'])
@pytest.mark.parametrize('present', [[], ['coordinator'], ['spark-expert']])
def test_cloned_and_both_role_builds_require_both_artifacts(tmp_path, role, present):
    result = check_readiness(tmp_path, role, present, 'baseline' if role != 'both' else '')
    assert result.returncode != 0


@pytest.mark.parametrize('role', ['coordinator', 'expert', 'both'])
def test_complete_pair_passes(tmp_path, role):
    result = check_readiness(tmp_path, role, ['coordinator', 'spark-expert'], 'baseline')
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('role', ['coordinator', 'expert'])
def test_selected_role_still_requires_its_artifacts(tmp_path, role):
    result = check_readiness(tmp_path, role, [])
    assert result.returncode != 0


def test_wip_state_symlink_is_excluded_from_source_staging(tmp_path):
    source = tmp_path / 'repo'
    source.mkdir()
    cache = tmp_path / 'cache'
    cache.mkdir()
    (source / '.cuteafd-wip').symlink_to(cache, target_is_directory=True)
    text = (ROOT / 'wip.sh').read_text()
    block = text.split('snapshot_args=(\n', 1)[1].split('\n)', 1)[0]
    stage = tmp_path / 'stage'
    result = subprocess.run(['bash', '-c', 'snapshot_args=(\n'+block+'\n)\n'
                             'rsync "${snapshot_args[@]}" "$1/" "$2/"', 'test', str(source), str(stage)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not (stage / '.cuteafd-wip').is_symlink()
    release_stage = tmp_path / 'release-stage'
    result = subprocess.run([str(ROOT / 'scripts/build/stage-release-source.sh'), str(source), str(release_stage)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not (release_stage / '.cuteafd-wip').is_symlink()
