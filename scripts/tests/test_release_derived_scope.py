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
            assert value == ('ON' if any(e[key] == 'ON' for e in per_entry) else 'OFF')
        else:
            expected = set().union(*(SCOPE.tokens(e[key]) for e in per_entry))
            if key.endswith('SPARK_TP_ROLES'):
                expected |= {'tp2', 'tp3', 'tp6'}
            assert SCOPE.tokens(value) == expected
    assert union[P + 'GENERIC_SPARK_COUNTS'] == 'glm=2'
    assert union[P + 'EXPERT_FAMILIES'] == 'glm:fp8;glmf:nvfp4;glmfdense:nvfp4'
    assert SCOPE.requirements(repo, path, 'glm-fp8')[0]['spark_count'] == 2
    assert not SCOPE.requirements(repo, path, 'glm-fp8')[0]['dual']  # set wins over gpus


def test_raw_environment_adds_but_does_not_shrink(fixture_repo):
    repo, path = fixture_repo
    added, _ = SCOPE.derive(repo, path, environ={P + 'EXPERT_FAMILIES': 'qwen4:nvfp4', P + 'GLM_AOT': 'OFF',
                                               P + 'MIMO_GEOMETRIES': 'mimof,mimof2'})
    assert added[P + 'GLM_AOT'] == 'ON'
    assert added[P + 'EXPERT_FAMILIES'] == 'glm:fp8;glmf:nvfp4;glmfdense:nvfp4;qwen4:nvfp4'
    assert added[P + 'MIMO_GEOMETRIES'] == 'mimof;mimof2'
    empty, _ = SCOPE.derive(repo, path, environ={P + 'EXPERT_FAMILIES': ''})
    assert empty[P + 'EXPERT_FAMILIES'] == 'glm:fp8;glmf:nvfp4;glmfdense:nvfp4'


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
    all_scope[P + 'SPARK_TP_ROLES'] = both[P + 'SPARK_TP_ROLES']
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
