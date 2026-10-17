"""CPU-only matrix derivation and fail-closed image coverage contracts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('release_scope', REPO / 'scripts/build/release-scope.py')
SCOPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCOPE)
P = 'CUTEAFD_RELEASE_'


@pytest.fixture
def matrix(tmp_path):
    (tmp_path / 'glm.config').write_text('FAMILY=glm5\nMODEL_ID=fixture/glm\nSPARK_COUNT=4\nRTX_GPUS=1\n')
    (tmp_path / 'flash.config').write_text('FAMILY=glm5_flash\nMODEL_ID=fixture/flash-nvfp4\nSPARK_COUNT=4\n')
    path = tmp_path / 'matrix.json'
    path.write_text(json.dumps(dict(defaults=dict(set=dict(AUDIO='off')), entries=[
        dict(name='glm-fp8', family='glm5', config='glm.config', sparks=['rank-a', 'rank-b'], gpus=[0, 1],
             set=dict(SPARK_TP='2', RTX_GPUS='1')),
        dict(name='flash-nvfp4', family='glm5_flash', config='flash.config', sparks=[], gpus=[0]),
    ])))
    # The helper imports WIP derivation from the source repo, not a fixture repo.
    return tmp_path, path


@pytest.fixture
def fixture_repo(matrix, monkeypatch):
    repo, path = matrix
    real = SCOPE.scopes_module(REPO)
    monkeypatch.setattr(SCOPE, 'scopes_module', lambda _: real)
    return repo, path


def test_union_equals_per_entry_derivation(fixture_repo):
    repo, path = fixture_repo
    union, entries = SCOPE.derive(repo, path, environ={})
    per_entry = [SCOPE.derive(repo, path, entry['name'], {})[0] for entry in entries]
    for key, value in union.items():
        if key.endswith('_AOT'):
            assert value == ('ON' if any(e[key] == 'ON' for e in per_entry) or key.removeprefix(P) in SCOPE.PREVIOUS_RELEASE_FLOOR else 'OFF')
        else:
            expected = set().union(*(SCOPE.tokens(e[key]) for e in per_entry))
            expected |= SCOPE.PREVIOUS_RELEASE_FLOOR.get(key.removeprefix(P), set())
            assert SCOPE.tokens(value) == expected
    assert SCOPE.PREVIOUS_RELEASE_FLOOR['GENERIC_SPARK_COUNTS'] <= SCOPE.tokens(union[P + 'GENERIC_SPARK_COUNTS'])
    assert union[P + 'EXPERT_FAMILIES'] == 'glm:fp8;glmf:nvfp4;glmfdense:nvfp4;mimo:fp8'
    assert SCOPE.requirements(repo, path, 'glm-fp8')[0]['spark_count'] == 4
    assert not SCOPE.requirements(repo, path, 'glm-fp8')[0]['dual']  # set wins over gpus


def test_raw_environment_adds_but_does_not_shrink(fixture_repo):
    repo, path = fixture_repo
    added, _ = SCOPE.derive(repo, path, environ={P + 'EXPERT_FAMILIES': 'qwen4:nvfp4', P + 'GLM_AOT': 'OFF',
                                               P + 'MIMO_GEOMETRIES': 'mimof,mimof2'})
    assert added[P + 'GLM_AOT'] == 'ON'
    assert added[P + 'EXPERT_FAMILIES'] == 'glm:fp8;glmf:nvfp4;glmfdense:nvfp4;mimo:fp8;qwen4:nvfp4'
    assert added[P + 'MIMO_GEOMETRIES'] == 'mimo;mimo2;mimof;mimof2'
    empty, _ = SCOPE.derive(repo, path, environ={P + 'EXPERT_FAMILIES': ''})
    assert empty[P + 'EXPERT_FAMILIES'] == 'glm:fp8;glmf:nvfp4;glmfdense:nvfp4;mimo:fp8'


def test_selectors_and_explicit(fixture_repo):
    repo, path = fixture_repo
    family, entries = SCOPE.derive(repo, path, 'glm5_flash', {})
    assert [e['name'] for e in entries] == ['flash-nvfp4']
    assert family[P + 'EXPERT_FAMILIES'] == 'glmf:nvfp4;glmfdense:nvfp4'
    assert family[P + 'GLM_AOT'] == 'OFF'
    assert family[P + 'MIMO_AOT'] == 'OFF'
    assert family[P + 'SPARK_TP_ROLES'] == ''
    both, _ = SCOPE.derive(repo, path, 'glm-fp8,flash-nvfp4', {})
    all_scope = SCOPE.derive(repo, path, 'all', {})[0]
    assert all_scope[P + 'SPARK_TP_ROLES'] == 'tp2;tp3;tp6'
    for key in SCOPE.PREVIOUS_RELEASE_FLOOR:
        all_scope[P + key] = both[P + key]
    assert both == all_scope
    explicit, entries = SCOPE.derive(repo, path, 'explicit', {P + 'EXPERT_FAMILIES': 'glm:fp8', P + 'GLM_AOT': 'OFF'})
    assert not entries
    assert explicit[P + 'EXPERT_FAMILIES'] == 'glm:fp8'
    assert explicit[P + 'GLM_AOT'] == 'OFF'
    assert SCOPE.requirements(repo, path, 'explicit') == []
    with pytest.raises(ValueError, match='unknown release scope'):
        SCOPE.derive(repo, path, 'missing', {})


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fake_inventory(root, family='glm'):
    import hashlib
    write_json(root / 'share/PROGRAMS.json', dict(programs=[dict(family=family, name='norm')]))
    for role in ('coordinator', 'expert'):
        target = root / role
        write_json(target / 'share/PROGRAMS.json', dict(programs=[dict(family=family, name='norm')]))
        package = target / 'lib/fp8/fp8-glm'
        layouts = ['tp1', 'tp2']
        files = {}
        for layout in layouts:
            name = f'{layout}/libcuteafd_fp8_moe.so'
            file = package / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'fake compiled package')
            files[name] = dict(bytes=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        write_json(package / 'manifest.json', dict(role='spark' if role == 'expert' else role,
                                                  layouts={layout: {} for layout in layouts}, files=files))
    return root


def test_inventory_requires_actual_program_family_and_package(fixture_repo, tmp_path):
    repo, matrix = fixture_repo
    wanted = SCOPE.requirements(repo, matrix, 'glm-fp8')
    root = fake_inventory(tmp_path / 'images')
    SCOPE.check_inventory(root / 'coordinator', 'coordinator', wanted, symbols=set())
    SCOPE.check_inventory(root / 'expert', 'expert', wanted, symbols=set())
    write_json(root / 'coordinator/share/PROGRAMS.json', dict(programs=[dict(family='dsv4f')]))
    with pytest.raises(ValueError, match='entry glm-fp8.*missing program families.*glm'):
        SCOPE.check_inventory(root / 'coordinator', 'coordinator', wanted, symbols=set())
    write_json(root / 'coordinator/share/PROGRAMS.json', dict(programs=[dict(family='glm')]))
    (root / 'coordinator/lib/fp8/fp8-glm/manifest.json').unlink()
    with pytest.raises(ValueError, match='entry glm-fp8.*manifest.json'):
        SCOPE.check_inventory(root / 'coordinator', 'coordinator', wanted, symbols=set())


def test_inventory_rejects_missing_spark_layout_and_corruption(fixture_repo, tmp_path):
    repo, matrix = fixture_repo
    wanted = SCOPE.requirements(repo, matrix, 'glm-fp8')
    root = fake_inventory(tmp_path / 'images') / 'expert'
    manifest = root / 'lib/fp8/fp8-glm/manifest.json'
    data = json.loads(manifest.read_text())
    del data['layouts']['tp2']
    write_json(manifest, data)
    with pytest.raises(ValueError, match='entry glm-fp8.*missing layouts.*tp2'):
        SCOPE.check_inventory(root, 'expert', wanted, symbols=set())
    data['layouts']['tp2'] = {}
    write_json(manifest, data)
    (root / 'lib/fp8/fp8-glm/tp2/libcuteafd_fp8_moe.so').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        SCOPE.check_inventory(root, 'expert', wanted, symbols=set())


def test_exact_spark_slices_are_not_satisfied_by_padded_package(tmp_path):
    root = fake_inventory(tmp_path / 'images') / 'expert'
    path = root / 'lib/fp8/fp8-glm/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['layouts']['tp6'] = {'intermediate': 2048}
    write_json(path, manifest)
    with pytest.raises(ValueError, match='missing layouts.*tp6-w256.*tp6-w384'):
        SCOPE.check_package(root, 'fp8/fp8-glm', ['tp6'], 'expert')


def test_resources_do_not_override_config_topology(fixture_repo):
    repo, matrix = fixture_repo
    entry = SCOPE.requirements(repo, matrix, 'flash-nvfp4')[0]
    assert entry['spark_count'] == 4  # resource list is empty; config still wins
    assert not entry['dual']


def test_coordinator_requires_qwen_exact_width_siblings(tmp_path):
    root = fake_inventory(tmp_path / 'images') / 'coordinator'
    path = root / 'lib/fp8/fp8-glm/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['layouts']['tp2'] = {'intermediate': 640}
    write_json(path, manifest)
    with pytest.raises(ValueError, match='missing layouts.*tp2-w256.*tp2-w384'):
        SCOPE.check_package(root, 'fp8/fp8-glm', ['tp1', 'tp2'], 'coordinator')


def test_stdin_checker_has_no_repository_or_working_directory_dependency(tmp_path):
    import subprocess
    helper = REPO / 'scripts/build/release-scope.py'
    result = subprocess.run(['python3', '-', 'check-files', '--role', 'coordinator', '--root', str(tmp_path),
                             '--requirements', '[]'], input=helper.read_text(), cwd='/', text=True,
                            capture_output=True)
    # It gets as far as the native inventory read, not __file__ resolution.
    assert 'libcuteafd_native.so' in result.stderr
    assert 'IndexError' not in result.stderr


def test_qwen_dual_coordinator_requires_exact_layouts_without_geometry_metadata(tmp_path):
    wanted = [dict(name='qwen-dual', family='qwen4', programs=['qwen4'], experts=['qwen4:nvfp4'],
                   spark_experts=[], spark_count=0, dual=True, audio=False, context_split=False)]
    root = fake_inventory(tmp_path / 'images', family='qwen4') / 'coordinator'
    import shutil
    base = root / 'lib/fp8/fp8-glm'
    shutil.copytree(base, root / 'lib/fp8/fp8-qwen4-nvfp4')
    with pytest.raises(ValueError, match='qwen-dual.*missing layouts.*tp2-w256.*tp2-w384'):
        SCOPE.check_inventory(root, 'coordinator', wanted, symbols=set())


def test_pending_tags_cleanup_runs_after_success_and_failure(tmp_path):
    import subprocess
    build = (REPO / 'build.sh').read_text()
    cleanup = 'release_pending_cleanup() {' + build.split('release_pending_cleanup() {', 1)[1].split("trap 'exit 130' INT", 1)[0]
    # The extraction retains the parent EXIT trap; image promotion never clears it.
    assert 'trap - EXIT' not in build.split('coordinator_container="$(docker create', 1)[1].split('release_ssh "$seed_host" bash', 1)[0]
    for status in (0, 2):
        log = tmp_path / f'cleanup-{status}.log'
        script = tmp_path / 'ssh'
        script.write_text('#!/bin/sh\nprintf "%s\n" "$*" >> "$CLEANUP_LOG"\n')
        script.chmod(0o755)
        result = subprocess.run(['bash', '-c', f"""set -euo pipefail
export PATH={tmp_path}:/usr/bin:/bin CLEANUP_LOG={log}
COORDINATOR_DOCKER_INFERENCE=coordinator-pending-123
SPARK_EXPERT_DOCKER_INFERENCE=spark-pending-123
RELEASE_BUILD_HOSTS=(fixture-a fixture-b)
release_ssh_opts=(-o BatchMode=yes)
docker() {{ printf '%s\n' "$*" >> "$CLEANUP_LOG"; }}
{cleanup}
exit {status}
"""], capture_output=True, text=True)
        assert result.returncode == status, result.stderr
        lines = log.read_text().splitlines()
        assert lines == ['image rm coordinator-pending-123',
                         '-o BatchMode=yes fixture-a docker image rm spark-pending-123',
                         '-o BatchMode=yes fixture-b docker image rm spark-pending-123']


def test_release_requirements_use_matrix_and_configs_at_build_revision(tmp_path):
    import shutil
    import subprocess
    repo = tmp_path / 'revision-repo'
    shutil.copytree(REPO / 'scripts/lib', repo / 'scripts/lib', ignore=shutil.ignore_patterns('__pycache__'))
    config = repo / 'examples/configs/fixture.config'
    config.parent.mkdir(parents=True)
    config.write_text('FAMILY=glm5\nMODEL_ID=fixture/glm\nSPARK_COUNT=4\nRTX_GPUS=1\n')
    matrix = repo / SCOPE.DEFAULT_MATRIX
    write_json(matrix, dict(entries=[dict(name='glm-fixture', family='glm5',
                                         config='examples/configs/fixture.config')]))
    (repo / 'cuteafd.config').write_text('FAMILY=glm5\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', 'commit', '-qm', 'Fixture'], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    config.write_text('FAMILY=glm5\nMODEL_ID=fixture/glm\nSPARK_COUNT=0\nRTX_GPUS=2\n')
    matrix.write_text('{"entries": []}')
    wanted = SCOPE.revision_requirements(repo, revision)
    assert wanted[0]['spark_count'] == 4
    assert not wanted[0]['dual']
