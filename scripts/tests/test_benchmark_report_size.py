"""Committed benchmark reports stay summaries: `cuteafd bench publish` slims them."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 2_000_000


def test_committed_reports_are_slim():
    fat = []
    for path in sorted(ROOT.glob('benchmarks/*/*/report.json')):
        report = json.loads(path.read_text())
        baseline = report.get('baseline') or {}
        rows = sum(len(((check.get('metrics') or {}).get('run') or {}).get('score', {}).get('records') or [])
                   for check in (baseline.get('quality') or {}).get('checks', []))
        if path.stat().st_size > LIMIT or rows or baseline.get('fabric'):
            fat.append(f'{path.relative_to(ROOT)}: {path.stat().st_size} B, {rows} per-row records, '
                       f'baseline fabric {bool(baseline.get("fabric"))}')
    assert not fat, 'run `cuteafd bench publish` to slim these:\n' + '\n'.join(fat)
