"""Direct WIP builds derive the same family scopes as single gate cards."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import wip_aot_scopes as aot

SPEC = importlib.util.spec_from_file_location('wip_family_cards', ROOT / 'scripts/bench/wip-cards.py')
cards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cards)

CASES = [
    ('deepseek_v41', 'DeepSeek-V4.1-Flash', 4, None, ''),
    ('deepseek_v4', 'DeepSeek-V4-Flash', 2, 'DSV4', 'dsv4f:rtx_backbone'),
    ('deepseek_v4', 'DeepSeek-V4-Pro-EXL3-K2', 4, 'DSV4', 'dsv4p:exl3-k23'),
    ('glm5', 'GLM-5.3-EXL3-K4', 4, 'GLM', 'glm:exl3-k45'),
    ('glm5_flash', 'GLM-5.3-Flash-tr3', 2, 'GLMF', 'glmf:exl3-k34'),
    ('glm5_flash', 'GLM-5.3-Flash-NVFP4', 4, 'GLMF', 'glmfdense:nvfp4'),
    ('mimo_v2', 'MiMo-V2.6-Flash', 4, 'MIMO', 'mimof:fp8'),
    ('mimo_v2', 'MiMo-V2.6-Pro', 6, 'MIMO', 'mimop:fp8'),
    ('qwen4', 'Qwen3.8-Flash-Next-EXL3-K4.25', 0, 'QWEN4', 'qwen4:exl3-k45'),
]


def config_file(tmp_path, family, model, count):
    path = tmp_path / 'model.config'
    hosts = '\n'.join(f'SPARK_{i}_HOST=spark{i}\nSPARK_{i}_LANE_A=192.0.2.{i + 1}\n'
                      f'SPARK_{i}_LANE_B=198.51.100.{i + 1}' for i in range(count))
    path.write_text(f'FAMILY={family}\nMODEL_ID=test/{model}\nSPARK_COUNT={count}\n'
                    f'RTX_GPUS=2\nCOORDINATOR_GPU=1\n{hosts}\n')
    return path


@pytest.mark.parametrize('family,model,count,switch,scope', CASES)
def test_config_scopes_match_equivalent_single_card(tmp_path, family, model, count, switch, scope):
    path = config_file(tmp_path, family, model, count)
    detected, derived = aot.config_scopes(aot.read_config(path))
    gate_card = dict(family=family, name=model.lower(), config=str(path))
    expected = cards.build_scopes([gate_card])
    assert detected == family
    assert {key: derived[key] for key in expected if key in derived} == {
        key: value for key, value in expected.items() if key in derived}
    if switch:
        assert derived[f'CUTEAFD_WIP_{switch}_AOT'] == 'ON'
    if scope:
        assert scope in derived['CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES'].split(';')
    if family in ('glm5', 'glm5_flash'):
        assert derived['CUTEAFD_WIP_CONTEXT_SPLIT_AOT'] == 'ON'
    if count == 0:
        assert derived['CUTEAFD_WIP_EXPERT_FAMILIES'] == ''


def dry_run(tmp_path, path, overrides=None):
    stub = tmp_path / 'bin'
    stub.mkdir(exist_ok=True)
    marker = tmp_path / 'hardware-called'
    for tool in ('docker', 'ssh', 'rsync', 'nvidia-smi'):
        script = stub / tool
        script.write_text(f'#!/bin/sh\necho {tool} >> "{marker}"\nexit 1\n')
        script.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('CUTEAFD_WIP_') and key not in ('SPARK_COUNT', 'SPARK_HOSTS', 'WIP_ROOT', 'WIP_INSTANCE')}
    env.update(PATH=str(stub) + ':' + os.environ['PATH'], **(overrides or {}))
    result = subprocess.run([str(ROOT / 'wip.sh'), '--config', str(path), '--role', 'coordinator', '--dry-run'],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()
    return result.stdout


@pytest.mark.parametrize('family,model,count,switch,scope', CASES)
def test_direct_wip_derives_defaults_and_env_override_wins(tmp_path, family, model, count, switch, scope):
    path = config_file(tmp_path, family, model, count)
    output = dry_run(tmp_path, path)
    assert f'WIP AOT scopes ({family})' in output
    if scope:
        assert scope in output
    if switch:
        key = f'CUTEAFD_WIP_{switch}_AOT'
        assert key + '=ON' in output
        output = dry_run(tmp_path, path, {key: 'OFF', 'CUTEAFD_WIP_EXPERT_FAMILIES': '',
                                          'CUTEAFD_WIP_CONTEXT_SPLIT_AOT': 'OFF',
                                          'CUTEAFD_WIP_AUDIO_AOT': 'OFF'})
        assert key + '=OFF' in output and key + '=ON' not in output
        assert 'CUTEAFD_WIP_EXPERT_FAMILIES= ' in output
        assert 'Audio AOT: OFF' in output
        if family in ('glm5', 'glm5_flash'):
            assert 'CUTEAFD_WIP_CONTEXT_SPLIT_AOT=OFF' in output


@pytest.mark.parametrize('family,model,count,switch,scope', CASES)
def test_seal_refuses_empty_programs_before_touching_slot(tmp_path, family, model, count, switch, scope):
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'cuteafd').write_text('#!/bin/sh\n')
    (output / 'cuteafd').chmod(0o755)
    (output / 'libcuteafd_native.so').write_text('stub native output\n')
    (output / 'PROGRAMS.json').write_text('{"schema":1,"programs":[]}\n')
    if switch is None:
        aot.check_programs(family, output / 'PROGRAMS.json')
        return
    result = subprocess.run(['bash', str(ROOT / 'scripts/build/finalize-wip-slot.sh'), str(ROOT),
                             'coordinator', 'test', str(output), 'image', 'image-id'],
                            env={**os.environ, 'CUTEAFD_WIP_PROGRAM_FAMILY': family},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert 'cannot seal coordinator slot for ' + family in result.stderr
    assert f'CUTEAFD_WIP_{switch}_AOT=ON' in result.stderr
    assert 'ARTIFACT_SHA256SUMS' not in result.stderr  # refused before checksums/copy/seal
    (output / 'PROGRAMS.json').write_text('{"schema":1,"programs":[{"name":"stub"}]}\n')
    aot.check_programs(family, output / 'PROGRAMS.json')


@pytest.mark.parametrize('family', ['deepseek_v41', 'deepseek_v4', 'glm5', 'glm5_flash', 'mimo_v2', 'qwen4'])
def test_resolves_model_metadata_with_existing_launcher_table(tmp_path, monkeypatch, family):
    configs = [case['config'] for case in json.loads((ROOT / 'rust/crates/cuteafd-loader/tests/fixtures/launch-families.json').read_text())
               if case.get('expected', {}).get('family') == family]
    if not configs:
        model_type = next(kind for kind, value in aot.checkpoint_family.FAMILIES.items() if value == family)
        config = dict(model_type=model_type, num_hidden_layers=2, first_k_dense_replace=0,
                      layer_types=['linear_attention', 'full_attention'] if family == 'qwen4'
                      else ['linear_attention', 'deepseek_sparse_attention'])
    else:
        config = configs[0]
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    snapshot = tmp_path / 'hub/models--test--unlabelled/snapshots/revision'
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text(json.dumps(config))
    values = dict(MODEL_ID='test/unlabelled', MODEL_REVISION='revision')
    assert aot.config_family(values) == family
    values['FAMILY'] = 'qwen4' if family != 'qwen4' else 'glm5'
    with pytest.raises(ValueError, match='does not match checkpoint'):
        aot.config_family(values)


def test_missing_metadata_fails_closed_unless_family_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    with pytest.raises(ValueError, match='set FAMILY'):
        aot.config_family(dict(MODEL_ID='test/unavailable'))
    assert aot.config_family(dict(MODEL_ID='test/unavailable', FAMILY='glm5')) == 'glm5'
    with pytest.raises(ValueError, match='unsupported FAMILY'):
        aot.config_family(dict(FAMILY='unknown'))


def test_seal_guard_is_forwarded_after_native_build():
    wip = (ROOT / 'wip.sh').read_text()
    coordinator = wip.split('build_coordinator() {', 1)[1].split('build_expert() (', 1)[0]
    assert coordinator.index('"${coordinator_build[@]}" native') < coordinator.index('CUTEAFD_WIP_PROGRAM_FAMILY=')
    assert coordinator.index('CUTEAFD_WIP_PROGRAM_FAMILY=') < coordinator.index('finalize-wip-slot.sh')


@pytest.mark.parametrize('contents', ['{}', '{"programs":null}', '{"programs":{}}', 'broken'])
def test_missing_or_malformed_programs_are_refused(tmp_path, contents):
    path = tmp_path / 'PROGRAMS.json'
    path.write_text(contents)
    with pytest.raises(ValueError, match='cannot seal'):
        aot.check_programs('qwen4', path)
    with pytest.raises(ValueError, match='cannot seal'):
        aot.check_programs('qwen4', tmp_path / 'missing.json')


@pytest.mark.parametrize('count,tp,role', [('6', '2', 'tp2'), ('6', '6', 'tp6'), ('3', '3', 'tp3'), ('6', '', 'tp6')])
def test_v41_direct_config_uses_explicit_spark_tp_role(count, tp, role):
    values = {'FAMILY': 'deepseek_v41', 'SPARK_COUNT': count, 'SPARK_TP': tp}
    _, scopes = aot.config_scopes(values)
    assert scopes['CUTEAFD_WIP_SPARK_TP_ROLES'] == role
