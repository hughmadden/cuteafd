"""Reference-only publication storage and privacy contracts, without hardware."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('reference_packaging', ROOT / 'scripts/bench/package-reference-only-fidelity.py')
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def test_top1024_storage_uses_stable_f16_ties_and_normalized_target():
    rows = np.full((2, 1100), -8., dtype='<f2')
    rows[:, 1099] = -1.
    window = dict(tokens=[0, 1099, 1098], roles=['ctx', 'gen', 'ctx'], score_from=1)
    result = packager.compact_tensors(rows, window)
    assert result['top_ids'][0].tolist() == [1099] + list(range(1023))
    assert result['roles'].tolist() == [1, 0]
    assert result['positions'].tolist() == [1, 2]
    assert result['next_token_ids'].tolist() == [1099, 1098]
    for i, target in enumerate([1099, 1098]):
        raw = rows[i].astype(np.float64)
        expected = raw[target] - np.log(np.exp(raw).sum())
        assert result['next_token_log_prob'][i] == np.float32(expected)
        tail = np.delete(raw, result['top_ids'][i])
        assert result['tail_log_mass'][i] == np.float32(np.log(np.exp(tail).sum()))
        assert np.array_equal(result['top_log_probs'][i], rows[i, result['top_ids'][i]])
    rows[0, 0] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        packager.compact_tensors(rows, window)


def test_public_panel_preserves_tokens_and_generation_identity_not_deployment():
    source = dict(set_sha256='sealed', windows=[dict(tokens=[1, 2], roles=['ctx', 'gen'], score_from=1)],
                  generation_server=dict(build=dict(commit='a' * 40), model='Qwen/model', family='qwen4',
                                         settings=[dict(name='snapshot', value='/home/tj/private')]))
    public = packager.public_panel(source)
    assert public['windows'] == source['windows']
    assert public['generation_server'] == dict(build=dict(commit='a' * 40), model='Qwen/model', family='qwen4')
    assert 'settings' in source['generation_server']
    assert public['publication_provenance']['source_set_sha256'] == 'sealed'
    source['windows'][0]['provenance'] = 'ghp_' + 'a' * 30
    with pytest.raises(ValueError, match='credential'):
        packager.public_panel(source)


def test_seal_paths_cannot_escape(tmp_path):
    for value in ('../secret', '/secret', '.'):
        with pytest.raises(ValueError):
            packager.contained(tmp_path, value)
    (tmp_path / 'link').symlink_to('/etc/passwd')
    with pytest.raises(ValueError):
        packager.contained(tmp_path, 'link')
