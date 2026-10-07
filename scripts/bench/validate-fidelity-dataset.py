#!/usr/bin/env python3
"""Qualify top1024 files against two immutable full-vocabulary baseline repeats."""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
from safetensors.numpy import load_file

# Paths and daemon identity are explicit; no serving binary is rebuilt or replaced.
REF = CONFIG = OUT = DAEMON = BINARY = None
ARMS = []
IDENTITY = FEATURE = SOURCE_DIFF = None
MASK = (1 << 64) - 1


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(1 << 20):
            h.update(data)
    return h.hexdigest()


def lse(x):
    peak = np.max(x)
    return peak + np.log(np.exp(x - peak).sum(dtype=np.float64))


def indices(x):
    threshold = np.partition(x, len(x) - 1024)[-1024]
    above = np.flatnonzero(x > threshold)
    tied = np.flatnonzero(x == threshold)[:1024 - len(above)]
    ids = np.concatenate([above, tied])
    return ids[np.lexsort((ids, -x[ids]))]


def engine(path, tensor, vocab):
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        assert size <= 1 << 20
        t = json.loads(f.read(size))[tensor]
        assert t['dtype'] == 'F32' and t['shape'] == [vocab]
        start, end = t['data_offsets']
        assert end - start == vocab * 4
        assert 8 + size + end <= path.stat().st_size
        f.seek(8 + size + start)
        data = f.read(end - start)
        assert len(data) == vocab * 4
        x = np.frombuffer(data, dtype='<f4').astype(np.float64)
    assert np.isfinite(x).all()
    return x - lse(x)


def calibration(top1, kl):
    return dict(top1_min=max(.90, (math.floor(top1 * 100) - 2) / 100),
                kl_max=min(.06, (math.ceil(kl * 100) + 2) / 100))


def bootstrap(sums, counts, seed, repeats, samples_only=False):
    state = seed
    n = len(sums)
    threshold = ((-n) & MASK) % n
    samples = []
    for _ in range(repeats):
        total = 0.
        count = 0
        for _ in range(n):
            while True:
                state = (state + 0x9e3779b97f4a7c15) & MASK
                z = state
                z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & MASK
                z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & MASK
                z ^= z >> 31
                if z >= threshold:
                    break
            idx = z % n
            total += sums[idx]
            count += counts[idx]
        samples.append(total / count)
    return samples if samples_only else float(np.quantile(samples, .95))


def metrics(run):
    rows = [r for r in run['score']['records'] if r['role'] == 'gen']
    assert rows and all(r["role"] == "gen" for r in rows)
    return dict(top1=sum(r['agree'] for r in rows) / len(rows),
                kl=sum(r['kl'] for r in rows) / len(rows),
                confident_top1=(sum(r['agree'] for r in rows if r['confident']) /
                                sum(r['confident'] for r in rows)) if any(r['confident'] for r in rows) else None,
                top3=sum(r['top3_contained'] for r in rows) / len(rows))


def compact_absolute(run, expect):
    rows = [r for r in run['score']['records'] if r['role'] == 'gen']
    by_window = {}
    for r in rows:
        by_window.setdefault(r['window'], []).append(r)
    m = metrics(run)
    return (m['top1'] + 1e-12 >= max(.90, expect['top1_min'])
            and m['kl'] <= min(.06, expect['kl_max'])
            and all(sum(r['agree'] for r in rs) / len(rs) + 1e-12 >= .80 for rs in by_window.values()))


def tripwire_calibration(values):
    assert values and all(math.isfinite(v) and 0 <= v <= 1 for v in values)
    return (math.floor(min(values) * 100) - 2) / 100


def compact_tripwires(runs, expect, seed, repeats):
    messages = []
    for name, run in zip(('baseline', 'candidate'), runs):
        m = metrics(run)
        if m['confident_top1'] is not None and m['confident_top1'] + 1e-12 < expect['confident_top1_min']:
            messages.append(name + ': confident top1 below family calibrated minimum')
        if m['top3'] + 1e-12 < expect['top3_min']:
            messages.append(name + ': top3 containment below family calibrated minimum')
    maps = [{(r['window'], r['position']): r for r in run['score']['records']} for run in runs]
    reported = {}
    for label, confident_only, field, margin in (
            ('confident_top1', True, 'agree', expect['confident_drop_margin']),
            ('top3', False, 'top3_contained', expect['top3_drop_margin'])):
        windows = {}
        baseline = candidate = 0
        for key in sorted(maps[0]):
            b, a = maps[0][key], maps[1][key]
            if b['role'] != 'gen' or (confident_only and not b['confident']):
                continue
            baseline += b[field]
            candidate += a[field]
            entry = windows.setdefault(key[0], [0., 0])
            entry[0] += int(b[field]) - int(a[field])
            entry[1] += 1
        if not windows:
            reported[label] = None
            continue
        sums, counts = zip(*(windows[k] for k in sorted(windows)))
        samples = bootstrap(sums, counts, seed, repeats, samples_only=True)
        lower, upper = map(float, np.quantile(samples, [.05, .95]))
        reported[label] = dict(baseline=baseline / sum(counts), candidate=candidate / sum(counts),
            loss=(baseline - candidate) / sum(counts), lower95=lower, upper95=upper, gross_margin=margin)
        if lower > margin:
            messages.append('paired confident top1 drop exceeds 1 point beyond noise' if confident_only else
                            'paired top3 containment drop exceeds 0.5 point beyond noise')
    code = [[r['nll'] for r in run['score']['records'] if r['block'] == 'C'] for run in runs]
    if code[0] and np.mean(code[1]) - np.mean(code[0]) > .01:
        messages.append('human code NLL increases >0.01 nat')
    return messages, reported


def main():
    # Admit all four immutable reports before creating the output directory.
    manifest = json.loads((REF / 'rows.json').read_text())
    dataset = json.loads((CONFIG / 'manifest.json').read_text())
    panel = json.loads((CONFIG / 'windows.json').read_text())
    assert manifest['set_sha256'] == dataset['set_sha256'] == panel['set_sha256']
    assert digest(CONFIG / 'windows.json') == dataset['windows_sha256']
    raw_runs = {shape: [json.loads((a / f'full-{shape}.json').read_text()) for a in ARMS]
                for shape in ('decode', 'prefill')}
    expect_by_path = {}
    baseline_metrics = {}
    for shape, runs in raw_runs.items():
        baseline_metrics[shape] = [metrics(r) for r in runs]
        assert all(m['top1'] >= .92 for m in baseline_metrics[shape]), 'STOP BAR'
        expect_by_path[shape] = calibration(min(m['top1'] for m in baseline_metrics[shape]),
                                          max(m['kl'] for m in baseline_metrics[shape]))
        expect_by_path[shape]['tripwires'] = dict(
            confident_top1_min=tripwire_calibration([m['confident_top1'] for m in baseline_metrics[shape]]),
            top3_min=tripwire_calibration([m['top3'] for m in baseline_metrics[shape]]),
            confident_drop_margin=.01, top3_drop_margin=.005)
    expect = dict(top1_min=min(v['top1_min'] for v in expect_by_path.values()),
                  kl_max=max(v['kl_max'] for v in expect_by_path.values()),
                  tripwires=dict(confident_top1_min=min(v['tripwires']['confident_top1_min'] for v in expect_by_path.values()),
                                 top3_min=min(v['tripwires']['top3_min'] for v in expect_by_path.values()),
                                 confident_drop_margin=.01, top3_drop_margin=.005))
    assert not OUT.exists(), 'Immutable validation attempt exists'
    OUT.mkdir()
    projected = copy.deepcopy(raw_runs)
    for shape, runs in projected.items():
        for i, r in enumerate(runs):
            r['floor_top1'] = expect['top1_min']
            r['floor_kl'] = expect['kl_max']
            r['tripwire_expect'] = expect['tripwires']
            (OUT / f'baseline-{i}-full-{shape}.json').write_text(json.dumps(r) + '\n')
    command = [str(BINARY), 'bench', 'fidelity', 'compare-full']
    for arm, i in (('a', 1), ('b', 0)):
        for shape in ('decode', 'prefill'):
            command += [f'--{arm}-{shape}', str(OUT / f'baseline-{i}-full-{shape}.json')]
    command += ['--out', str(OUT / 'paired-full.json')]
    result = subprocess.run(command, timeout=180, capture_output=True, text=True)
    (OUT / 'paired-full.log').write_text(result.stdout + result.stderr)
    # A measured FAIL is valid evidence too; only a missing report is an error.
    assert (OUT / 'paired-full.json').exists(), result.stderr
    paired = json.loads((OUT / 'paired-full.json').read_text())
    report = dict(schema='cuteafd.top1024.validation/1', set_sha256=manifest['set_sha256'],
                  qualification_scope='reference repeatability, not a precision-default verdict',
                  common_floor=dict(top1_min=.90, kl_max=.06),
                  family_expect=expect, expect_by_path=expect_by_path,
                  expect_provenance="calibrated from this config's baselines",
                  paired_bar=dict(top1_margin=.005, kl_margin=.005),
                  baseline_metrics=baseline_metrics,
                  storage=dict(ids='u32', log_probs='f16', tail_log_mass='f32'),
                  source_report_sha256={f'{i}-{shape}': digest(a / f'full-{shape}.json')
                                        for i, a in enumerate(ARMS) for shape in raw_runs},
                  coordinator_sha256=digest(DAEMON),
                  comparison_sha256=digest(BINARY),
                  legacy_v41_tripwire_context=dict(confident_top1_min=.98, top3_min=.99),
                  daemon_identity=IDENTITY,
                  coordinator_feature_commit=FEATURE,
                  coordinator_source_diff_sha256=digest(SOURCE_DIFF),
                  build_report_note='Daemon identity and feature revision are separately sealed from comparison binary',
                  shapes={})
    windows = {w['id']: w for w in panel['windows']}
    files = {f['window']: f for f in dataset['files']}
    assert len(windows) == len(files) == len(manifest['windows']) == len(panel['windows'])
    scored_positions = sum(len(w['tokens']) - w['score_from'] for w in panel['windows'])
    assert scored_positions == sum(len(w['positions']) for w in manifest['windows'])

    for shape, runs in projected.items():
        compact_runs = copy.deepcopy(runs)
        maps = [{(r['window'], r['position']): r for r in run['score']['records']} for run in runs]
        compact_maps = [{(r['window'], r['position']): r for r in run['score']['records']}
                        for run in compact_runs]
        assert all(len(m) == scored_positions for m in maps)

        def window_work(item):
            number, meta = item
            wid = meta['id']
            window = windows[wid]
            full_path = REF / meta['path']
            assert digest(full_path) == meta['sha256'] == files[wid]['source_full_sha256']
            assert full_path.stat().st_size == len(meta['positions']) * manifest['vocab'] * 2
            full_rows = np.memmap(full_path, dtype='<f2', mode='r',
                                  shape=(len(meta['positions']), manifest['vocab']))
            assert digest(CONFIG / files[wid]['path']) == files[wid]['sha256']
            tensors = load_file(str(CONFIG / files[wid]['path']))
            dumps = []
            for arm in ARMS:
                folder = arm / f'dump-{shape}/window-{number:03d}'
                records = [json.loads(s) for s in (folder / 'manifest.jsonl').read_text().splitlines()]
                mapping = {r['position']: r for r in records}
                assert len(mapping) == len(records) and sorted(mapping) == meta['positions']
                dumps.append((folder, mapping))
            rows = []
            max_reproduction = 0.
            for i, pos in enumerate(meta['positions']):
                raw = full_rows[i].astype(np.float64)
                assert np.isfinite(raw).all()
                ids = indices(raw)
                mask = np.ones(manifest['vocab'], dtype=bool)
                mask[ids] = False
                assert np.array_equal(tensors['top_ids'][i], ids)
                assert np.array_equal(tensors['top_log_probs'][i], full_rows[i, ids])
                assert tensors['tail_log_mass'][i] == np.float32(lse(raw[mask]))
                assert tensors['positions'][i] == pos and tensors['next_token_ids'][i] == window['tokens'][pos]
                assert bool(tensors['roles'][i]) == (window['roles'][pos] == 'gen')
                full = raw - lse(raw)
                assert tensors['next_token_log_prob'][i] == np.float32(full[window['tokens'][pos]])
                lp = np.r_[tensors['top_log_probs'][i].astype(np.float64),
                           float(tensors['tail_log_mass'][i])]
                lp -= lse(lp)
                kls = []
                for arm, (folder, mapping) in enumerate(dumps):
                    entry = mapping[pos]
                    q = engine(folder / entry['file'], entry['tensor'], manifest['vocab'])
                    compact = float(np.dot(np.exp(lp), lp - np.r_[q[ids], lse(q[mask])]))
                    full_kl = float(np.dot(np.exp(full), full - q))
                    assert compact >= -1e-10 and full_kl >= -1e-10
                    record = maps[arm][wid, pos]
                    assert int(np.argmax(q)) == record['argmax']
                    max_reproduction = max(max_reproduction, abs(full_kl - record['kl']))
                    revised = compact_maps[arm][wid, pos]
                    revised.update(kl=max(compact, 0.), ref_nll=-float(tensors['next_token_log_prob'][i]),
                                   reference_argmax=int(ids[0]), agree=record['argmax'] == int(ids[0]),
                                   confident=math.exp(lp[0]) >= .5,
                                   top3_contained=record['argmax'] in ids[:3],
                                   nll=-float(q[window['tokens'][pos]]))
                    kls.append(max(compact, 0.))
                rows.append((pos, kls))
            return wid, rows, max_reproduction

        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for wid, rows, error in pool.map(window_work, enumerate(manifest['windows'])):
                errors.append(error)
                print(shape, wid, 'complete', flush=True)
        assert max(errors) <= 1e-8, ('Full-score reproduction differs', max(errors))
        differences = []
        for wid in sorted(windows):
            keys = [(wid, pos) for pos in range(windows[wid]['score_from'], len(windows[wid]['tokens']))
                    if windows[wid]['roles'][pos] == 'gen']
            if keys:
                differences.append((sum(compact_maps[1][k]['kl'] - compact_maps[0][k]['kl'] for k in keys), len(keys)))
        sums, counts = zip(*differences)
        delta = sum(sums) / sum(counts)
        old = paired[shape]
        upper = bootstrap(sums, counts, old['seed'], old['bootstrap'])
        tripwires, tripwire_metrics = compact_tripwires(compact_runs, expect['tripwires'], old['seed'], old['bootstrap'])
        absolute = all(compact_absolute(r, expect) for r in compact_runs)
        # Recompute top-1 upper bound if F16 top support changes metadata.
        top_sums = []
        for wid in sorted(windows):
            keys = [k for k, r in compact_maps[0].items() if k[0] == wid and r['role'] == 'gen']
            if keys:
                top_sums.append(sum(int(compact_maps[0][k]['agree']) - int(compact_maps[1][k]['agree']) for k in keys))
        metadata_exact = all(compact_maps[i][k]['agree'] == maps[i][k]['agree']
                             and compact_maps[i][k]['confident'] == maps[i][k]['confident']
                             and compact_maps[i][k]['top3_contained'] == maps[i][k]['top3_contained']
                             for i in (0, 1) for k in maps[i])
        top_samples = bootstrap(top_sums, counts, old['seed'], old['bootstrap'], samples_only=True)
        top_loss = sum(top_sums) / sum(counts)
        top_upper = top_loss + 1.6448536269514722 * float(np.std(top_samples, ddof=1))
        if metadata_exact:
            assert abs(top_upper - old['top1_upper95']) < 1e-12
            assert sorted(tripwires) == sorted(old['tripwires'])
            for name, values in tripwire_metrics.items():
                if values is None:
                    assert old['calibrated_tripwires'][name] is None
                else:
                    for field, value in values.items():
                        assert abs(value - old['calibrated_tripwires'][name][field]) < 1e-12
        tripwire_difference = 0.
        for name, values in tripwire_metrics.items():
            full_values = old['calibrated_tripwires'][name]
            assert (values is None) == (full_values is None)
            if values is not None:
                tripwire_difference = max(tripwire_difference, *(abs(values[k] - full_values[k])
                    for k in ('baseline', 'candidate', 'loss', 'lower95', 'upper95')))
        verdict = absolute and not tripwires and top_upper < old['top1_margin'] and upper < old['kl_margin']
        for i, r in enumerate(compact_runs):
            r['kl_kind'] = 'top1024-plus-tail-validation-only'
            (OUT / f'baseline-{i}-compact-{shape}.json').write_text(json.dumps(r) + '\n')
        report['shapes'][shape] = dict(positions=sum(counts), windows=len(counts),
            kl_delta=delta, kl_upper95=upper, full_kl_delta=old['kl_delta'], full_kl_upper95=old['kl_upper95'],
            delta_difference=delta - old['kl_delta'], bound_difference=upper - old['kl_upper95'],
            full_score_max_reproduction_error=max(errors), metadata_exact=metadata_exact,
            top1_loss=top_loss, top1_upper95=top_upper, full_top1_upper95=old['top1_upper95'],
            absolute_pass=absolute, tripwires=tripwires, calibrated_tripwires=tripwire_metrics,
            full_calibrated_tripwires=old['calibrated_tripwires'], pass_verdict=verdict, original_pass=old['pass'],
            tripwire_max_difference=tripwire_difference,
            qualifies=abs(delta - old['kl_delta']) <= 1e-4 and abs(upper - old['kl_upper95']) <= 1e-4
                and abs(top_upper - old['top1_upper95']) <= 1e-4 and tripwire_difference <= 1e-4 and verdict == old['pass'])
        (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    report['qualifies'] = all(v['qualifies'] for v in report['shapes'].values())
    report['repeatability_pass'] = paired['pass']
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        assert calibration(.9396454820579334, .03866359253316875) == dict(top1_min=.91, kl_max=.06)
        assert calibration(.9695287721, .0085467330) == dict(top1_min=.94, kl_max=.03)
        assert calibration(.8999, .1) == dict(top1_min=.90, kl_max=.06)
        x = np.r_[np.ones(1023), np.zeros(9)]
        assert indices(x).tolist() == list(range(1024))
        assert bootstrap([0., 0.], [1, 2], 20260829, 100) == 0.
        assert tripwire_calibration([.9716271541464343, .972]) == .95
        assert tripwire_calibration([.9973194984868137, .998]) == .97
        expect = dict(confident_top1_min=.95, top3_min=.97, confident_drop_margin=.01, top3_drop_margin=.005)
        records = [dict(window=str(i // 3855), position=i % 3855, role='gen', confident=True,
                        agree=True, top3_contained=True, kl=.01, nll=.5, block='A') for i in range(11565)]
        b = dict(score=dict(records=records))
        for flips, gross in ((15, False), (50, True)):
            a = copy.deepcopy(b)
            for r in a['score']['records']:
                r['agree'] = r['top3_contained'] = r['position'] >= flips
            messages, bounds = compact_tripwires([b, a], expect, 1, 100)
            assert bool(messages) == gross
            assert abs(bounds['confident_top1']['lower95'] - flips / 3855) < 1e-12
        print('Calibration, deterministic support, bootstrap, and calibrated gross tripwire self-tests PASS')
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ("reference-rows", "config", "arms", "out", "daemon", "comparison", "source-diff"):
            parser.add_argument("--" + name, required=True, type=Path)
        parser.add_argument("--daemon-identity", required=True)
        parser.add_argument("--feature-revision", required=True)
        args = parser.parse_args()
        REF, CONFIG, OUT = args.reference_rows, args.config, args.out
        ARMS = [args.arms / f"baseline-{i}" for i in (0, 1)]
        DAEMON, BINARY, SOURCE_DIFF = args.daemon, args.comparison, args.source_diff
        IDENTITY, FEATURE = args.daemon_identity, args.feature_revision
        assert len(FEATURE) == 40 and all(c in "0123456789abcdef" for c in FEATURE)
        main()
