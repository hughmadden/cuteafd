"""CPU checks for audio score identity and KV allocation evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('audio_score_report', ROOT / 'scripts/qualify/mimo_v2/audio-score-report.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_kv_startup_record(tmp_path):
    path = tmp_path / 'startup.log'
    allocation = {'devices': [{'requested_pool_bytes': 128, 'fixed_bytes': 256}]}
    ledger = {'devices': [{'scopes': {'kv': 512}, 'used': 1024}]}
    path.write_text('MiMo steady allocation contract reservations=' + json.dumps(allocation)
                    + '\nMiMo KV records pool_tokens=16 bytes_per_token=8\n'
                    + 'memory ledger sequence=0 report=' + json.dumps(ledger) + '\n')
    result = MODULE.kv_records(path)
    assert result['logical_pool_bytes'] == result['requested_pool_bytes'] == 128
    assert result['fixed_bytes'] == 256
    path.write_text(path.read_text() + 'MiMo KV records pool_tokens=16 bytes_per_token=8\n')
    with pytest.raises(ValueError, match='one KV'):
        MODULE.kv_records(path)


def test_response_rejects_prefix_or_override(tmp_path):
    span = {'start': 1, 'len': 1, 'key': 'a' * 64, 'samples': 481, 'pcm_sha256': 'b' * 64}
    record = {'engine': 'mimo_v2', 'cold': True, 'no_speculation': True, 'cached_tokens': 0,
              'score_path': 'decode', 'prompt_ids': [1, 2], 'audio': [span], 'scored': 1,
              'rows': [{'position': 1, 'finite': True}]}
    MODULE.check_response({'probe': record}, [1, 2], span, 1, 'native', tmp_path)
    record['cached_tokens'] = 1
    with pytest.raises(ValueError, match='identity/cold'):
        MODULE.check_response({'probe': record}, [1, 2], span, 1, 'native', tmp_path)
    record['cached_tokens'] = 0
    record['provenance'] = {'mode': 'audio_reference_features'}
    with pytest.raises(ValueError, match='override'):
        MODULE.check_response({'probe': record}, [1, 2], span, 1, 'native', tmp_path)
