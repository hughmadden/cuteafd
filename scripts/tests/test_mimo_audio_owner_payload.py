"""CPU contracts for retained serving-library audio payload diagnostics."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('audio_owner_payload', ROOT / 'scripts/qualify/mimo_v2/audio-owner-payload.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_official_scoring_reference_is_cuda_and_cpu_is_informational():
    assert MODULE.reference_device('official') == 'cuda'
    assert MODULE.reference_device('cpu-info') == 'cpu'
    with pytest.raises(ValueError, match='unknown reference role'):
        MODULE.reference_device('auto')
    MODULE.validate_reference_sm(121, 121)
    with pytest.raises(ValueError, match='differs from serving'):
        MODULE.validate_reference_sm(120, 121)


def test_reference_pcm_rejects_identity_and_geometry_changes(tmp_path):
    path = tmp_path / 'pcm.f32'
    np.zeros(481, dtype='<f4').tofile(path)
    span = {'samples': 481, 'len': 1, 'pcm_sha256': MODULE.sha256(path)}
    clip = {'pcm': path, 'span': span}
    assert MODULE.reference_pcm(clip).shape == (481,)
    for field, bad in [('samples', 482), ('len', 2), ('pcm_sha256', '0' * 64)]:
        changed = {**clip, 'span': {**span, field: bad}}
        with pytest.raises(ValueError, match='canonical reference PCM identity'):
            MODULE.reference_pcm(changed)
    np.full(481, np.nan, dtype='<f4').tofile(path)
    clip['span']['pcm_sha256'] = MODULE.sha256(path)
    with pytest.raises(ValueError, match='canonical reference PCM identity'):
        MODULE.reference_pcm(clip)


def test_bf16_round_matches_serving_nearest_even():
    values = np.array([0x3f800000, 0x80000000, 0x3f808000, 0x3f818000], dtype=np.uint32).view(np.float32)
    assert MODULE.bf16_round(values).tolist() == [0x3f80, 0x8000, 0x3f80, 0x3f82]


def test_difference_locations_and_signed_ulp_distribution():
    expected = np.array([[0x3f80, 0xbf80, 0], [0x4000, 0xc000, 0x8000]], dtype=np.uint16)
    actual = np.array([[0x3f81, 0xbf81, 0], [0x4002, 0xc000, 0x8000]], dtype=np.uint16)
    result = MODULE.differences(actual, expected)
    assert result['different_elements'] == 3
    assert result['positions_row_dim'] == [[0, 0], [0, 1], [1, 0]]
    assert result['ulp_histogram'] == {'1': 2, '2': 1}
    assert result['max_bf16_ulp'] == 2
    assert result['max_abs'] == .03125
    assert result['different_elements_per_row'] == [2, 1]


def test_identical_payload_and_invalid_inputs():
    bits = np.array([[0x3f80, 0xbf80]], dtype=np.uint16)
    result = MODULE.differences(bits, bits)
    assert result['different_elements'] == result['max_abs'] == result['max_bf16_ulp'] == 0
    assert result['positions_row_dim'] == []
    with pytest.raises(ValueError, match='shape/dtype'):
        MODULE.differences(bits, bits.astype(np.uint32))
    with pytest.raises(ValueError, match='nonfinite'):
        MODULE.differences(np.array([[0x7f80, 0]], dtype=np.uint16), bits)


@pytest.mark.parametrize('hidden', [4096, 6144])
def test_compare_uses_checkpoint_width_and_checks_hashes(tmp_path, hidden):
    from mimo_media import snapshot_identity
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'config.json').write_text(json.dumps({'hidden_size': hidden}))
    for name in ('tokenizer.json', 'modeling_mimo_v2.py', 'preprocessor_config.json'):
        (snapshot / name).write_text('{}')
    features = tmp_path / 'features'
    capture = tmp_path / 'capture'
    capture.mkdir()
    span = {'start': 4, 'len': 1, 'samples': 481, 'key': 'a' * 64, 'pcm_sha256': 'b' * 64}
    geometry = MODULE.oracle.token_geometry(span['samples'])
    bits = np.full((1, hidden), 0x3f80, dtype='<u2')
    codes = np.zeros((geometry['codes'], 20), dtype='<i8')
    MODULE.oracle.write_probe_features(features, span, bits, codes, snapshot_identity(snapshot))
    bits.tofile(capture / '0.bf16')
    codes.tofile(capture / '0.codes.i64')
    clip = {'id': 'test', 'span': span, 'payload_prefix': '0',
            'bf16_sha256': MODULE.sha256(capture / '0.bf16'),
            'codes_sha256': MODULE.sha256(capture / '0.codes.i64')}
    clip['e2e_bf16_sha256'] = clip['bf16_sha256']
    (capture / 'capture.json').write_text(json.dumps({'library_sha256': 'c' * 64,
        'backend': 'test', 'clips': [clip]}))
    args = SimpleNamespace(capture=capture, features=features, snapshot=snapshot, output=tmp_path / 'result.json')
    MODULE.compare(args)
    result = json.loads(args.output.read_text())['results'][0]
    assert result['rvq_identical'] and result['matches_e2e_owner_hash']
    assert result['bf16']['elements'] == hidden
    (capture / '0.bf16').write_bytes(b'bad')
    with pytest.raises(ValueError, match='hash differs'):
        MODULE.compare(args)
