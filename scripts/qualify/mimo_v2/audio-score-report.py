#!/usr/bin/env python3
"""Audit retained audio scoring arms and full-vocabulary KL, without hardware."""
import argparse
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('media_paired_probe', ROOT / 'scripts/bench/media-paired-probe.py')
MEDIA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEDIA)


def kv_records(path):
    text = re.sub(r'\x1b\[[0-9;]*m', '', path.read_text())
    matches = re.findall(r'MiMo KV records.*?pool_tokens=(\d+).*?bytes_per_token=(\d+)', text)
    if len(matches) != 1:
        raise ValueError('require one KV allocation record')
    tokens, width = map(int, matches[0])
    allocations = [json.loads(line.split('reservations=', 1)[1]) for line in text.splitlines()
                   if 'MiMo steady allocation contract' in line]
    if len(allocations) != 1 or len(allocations[0]['devices']) != 1:
        raise ValueError('require one single-device startup allocation')
    device = allocations[0]['devices'][0]
    if device['requested_pool_bytes'] < tokens * width:
        raise ValueError('startup KV allocation below logical pool')
    ledgers = [json.loads(line.split('report=', 1)[1]) for line in text.splitlines()
               if 'memory ledger sequence=0 ' in line]
    if len(ledgers) != 1 or len(ledgers[0]['devices']) != 1:
        raise ValueError('require one startup memory ledger')
    ledger = ledgers[0]['devices'][0]
    return {'pool_tokens': tokens, 'bytes_per_token': width, 'logical_pool_bytes': tokens * width,
            'fixed_bytes': device['fixed_bytes'], 'requested_pool_bytes': device['requested_pool_bytes'],
            'ledger_kv_bytes': ledger['scopes']['kv'], 'ledger_used_bytes': ledger['used'],
            'source_log': str(path)}


def check_response(response, tokens, span, score_from, mode, features):
    record = response.get('probe') or {}
    if (record.get('engine') != 'mimo_v2' or record.get('error') or not record.get('cold')
            or not record.get('no_speculation') or record.get('cached_tokens') != 0
            or record.get('score_path') != 'decode' or record.get('prompt_ids') != tokens
            or record.get('audio') != [span] or record.get('scored') != len(tokens) - score_from):
        raise ValueError('audio arm identity/cold/scoring contract differs')
    positions = list(range(score_from, len(tokens)))
    if [r['position'] for r in record.get('rows', [])] != positions or not all(r['finite'] for r in record['rows']):
        raise ValueError('incomplete/nonfinite audio rows')
    provenance = record.get('provenance')
    if mode == 'native':
        if provenance is not None:
            raise ValueError('native scoring unexpectedly used an override')
    elif mode == 'official':
        metadata = json.loads((features / (span['key'] + '.json')).read_text())
        expected = {'mode': 'audio_reference_features', 'probe_only': True,
                    'encoder_bypassed': True, 'features': [metadata]}
        if provenance != expected:
            raise ValueError('official feature override provenance differs')
    else:
        raise ValueError('unknown scoring mode')


def compare(root, name):
    native = json.loads((root / 'scores' / (name + '-native.json')).read_text())
    official = json.loads((root / 'scores' / (name + '-official.json')).read_text())
    if native['server'] != official['server']:
        raise ValueError('scoring server identity/settings differ')
    tokens = native['probe']['prompt_ids']
    span = native['probe']['audio'][0]
    first = native['probe']['rows'][0]['position']
    for mode, response in [('native', native), ('official', official)]:
        check_response(response, tokens, span, first, mode, root / 'features')
    window = {'tokens': tokens, 'score_from': first}
    paths = [MEDIA.dump_rows(root / 'scores' / (name + '-' + mode), window, 152576)
             for mode in ('native', 'official')]
    kl, agreement = [], []
    for pos in paths[0]:
        q, p = [MEDIA.log_probs(path[pos], 152576) for path in paths]
        kl.append(max(0.0, float(np.sum(np.exp(p) * (p - q)))))
        agreement.append(int(p.argmax() == q.argmax()))
    return {'id': name, 'rows': len(kl), 'vocab': 152576, 'kl_mean_nat': float(np.mean(kl)),
            'kl_max_nat': max(kl), 'top1_agreement': float(np.mean(agreement)), 'kl_per_row_nat': kl,
            'direction': 'KL(official CPU features || native Spark features)',
            'cold_both': True, 'cached_tokens_both': 0, 'prompt_ids_and_audio_identical': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    existing = json.loads((args.root / 'scoring-report.json').read_text())
    rows = [compare(args.root, x['id']) for x in existing['native_vs_official']]
    kv = {mode: kv_records(args.root / ('cuteafd-mm-audio-flash-e2e-' + name + '.log'))
          for mode, name in [('off', 'off'), ('on', 'spark-0')]}
    args.output.write_text(json.dumps({'kind': 'cpu_audio_score_artifact_audit', 'native_vs_official': rows,
        'kv': kv, 'kv_pool_identical': all(kv['off'][k] == kv['on'][k]
            for k in ('pool_tokens', 'bytes_per_token', 'logical_pool_bytes', 'requested_pool_bytes',
                      'fixed_bytes', 'ledger_kv_bytes', 'ledger_used_bytes')),
        'qualification_complete': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
