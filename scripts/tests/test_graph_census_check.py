"""Family-neutral, fail-closed timed CUDA graph census assertion."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'bench/check-graph-census.py'


def run_check(tmp_path, capture_s=None, anchor=True, evidence=True, final_marker=False, **changes):
    lines = ['2026-11-04T12:00:00.000000Z INFO starting']
    if anchor:
        lines.append('2026-11-04T12:00:01.000000Z INFO cuteafd::thermal: current={"queried_s":1.0}')
    if evidence:
        lines.append('2026-11-04T12:00:02.000000Z INFO graph census executable site exec=0x1 site=startup.rs:10:1')
    if capture_s is not None:
        lines.append(f'2026-11-04T12:00:{capture_s:09.6f}Z INFO graph census executable site exec=0x2 site=verify.rs:20:1')
    lines.append('2026-11-04T12:00:30.000000Z INFO stopped')
    if final_marker:
        lines.append('2026-11-04T12:00:32.000000Z INFO cuteafd::bench_census: final server thermal sample '
                     'server_now_s=32.0 source=census-final-thermal.json')
    log = tmp_path / 'coordinator.log'
    log.write_text('\n'.join(lines) + '\n')
    point = {'c': 16, 'phase': 'levels', 'thermal': {'from_s': 10.0, 'to_s': 20.0}}
    point.update(changes)
    sweep = tmp_path / 'sweep.json'
    sweep.write_text(json.dumps({'points': [point]}))
    result = subprocess.run([sys.executable, '-I', str(SCRIPT), '--log', str(log), '--sweep', str(sweep)],
                            capture_output=True, text=True)
    return result.returncode, json.loads(result.stdout)


def test_clean_run_excludes_startup_and_warmup(tmp_path):
    code, report = run_check(tmp_path, capture_s=9.0)
    assert code == 0
    assert report['status'] == 'pass'
    assert report['total_logged_captures'] == 2
    assert report['windows'][0]['captures'] == 0


def test_capture_in_timed_window_fails(tmp_path):
    code, report = run_check(tmp_path, capture_s=15.0)
    assert code == 1
    assert report['status'] == 'fail'
    assert report['windows'][0]['sites'] == {'verify.rs:20:1': 1}


def test_missing_clock_anchor_fails_closed(tmp_path):
    code, report = run_check(tmp_path, anchor=False)
    assert code == 2
    assert 'clock anchors' in report['error']


def test_missing_census_evidence_fails_closed(tmp_path):
    code, report = run_check(tmp_path, evidence=False)
    assert code == 2
    assert 'PROFILE=census evidence' in report['error']


@pytest.mark.parametrize('stamp', [9.999, 10.0, 20.0, 20.001])
def test_rounded_window_edges_fail_conservatively(tmp_path, stamp):
    code, report = run_check(tmp_path, capture_s=stamp)
    assert code == 1
    assert report['windows'][0]['captures'] == 1


@pytest.mark.parametrize('thermal', [None, {}, {'from_s': 20, 'to_s': 10},
                                     {'from_s': float('nan'), 'to_s': 20}])
def test_missing_or_invalid_point_window_fails_closed(tmp_path, thermal):
    code, report = run_check(tmp_path, thermal=thermal)
    assert code == 2
    assert report['status'] == 'error'


def test_final_server_marker_closes_last_window(tmp_path):
    code, report = run_check(tmp_path, final_marker=True, thermal={'from_s': 10, 'to_s': 31})
    assert code == 0
    assert report['windows'][0]['captures'] == 0


def test_log_must_bracket_whole_window(tmp_path):
    code, report = run_check(tmp_path, thermal={'from_s': 10, 'to_s': 31})
    assert code == 2
    assert 'bracket' in report['error']


@pytest.mark.parametrize('capture_s,expected', [(2, 'pass'), (15, 'fail')])
def test_retained_topology_capture_path(tmp_path, capture_s, expected):
    # Reduced from #1837's retained GLM Flash log (2026-11-12).
    run_check(tmp_path, evidence=False)
    log = tmp_path / 'coordinator.log'
    lines = log.read_text().splitlines()
    lines[-1:-1] = [
        'CUTEAFD_GRAPH_CENSUS {"event":"capture_ref","id":1,"exec":"0x1234"}',
        f'2026-11-04T12:00:{capture_s:02}.000000Z INFO cuteafd::graph_capture: graph census capture key '
        'family="glm5_flash" bank="decode" instance=131777698886488 rank=0 lane=Some(0) '
        'key=GraphKey { segment: 0, rows: 1, spec: false, long: false, pool_width: 0, '
        'page_stride: 64, pool_stride: 0 } exec=0x1234',
    ]
    log.write_text('\n'.join(lines) + '\n')
    topology = tmp_path / 'graph-census-captures-raptor.jsonl'
    topology.write_text('CUTEAFD_GRAPH_CENSUS ' + json.dumps({
        'event': 'capture', 'id': 1, 'exec': '0x1234', 'device': 0,
        'nodes': 1, 'node_status': 0, 'details': [{'kind': 0, 'kernel': 'retained_kernel'}],
    }) + '\n')
    result = subprocess.run([sys.executable, '-I', str(SCRIPT), '--log', str(log),
                             '--sweep', str(tmp_path / 'sweep.json')], capture_output=True, text=True)
    assert json.loads(result.stdout)['status'] == expected
    assert result.returncode == (0 if expected == 'pass' else 1)


@pytest.mark.parametrize('missing', ['topology', 'key', 'ref', 'node_status', 'lost'])
def test_retained_path_missing_evidence_fails_closed(tmp_path, missing):
    test_retained_topology_capture_path(tmp_path, 2, 'pass')
    log = tmp_path / 'coordinator.log'
    topology = tmp_path / 'graph-census-captures-raptor.jsonl'
    if missing == 'topology':
        topology.unlink()
    elif missing == 'node_status':
        topology.write_text(topology.read_text().replace('"node_status": 0', '"node_status": 1'))
    elif missing == 'lost':
        log.write_text(log.read_text() + 'CUTEAFD_GRAPH_CENSUS {"event":"capture_write_failed","id":2}\n')
    else:
        log.write_text('\n'.join(line for line in log.read_text().splitlines()
                                 if ('capture key' if missing == 'key' else 'capture_ref') not in line))
    result = subprocess.run([sys.executable, '-I', str(SCRIPT), '--log', str(log),
                             '--sweep', str(tmp_path / 'sweep.json')], capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)['status'] == 'error'


def test_retained_exec_pointer_reuse_is_joined_by_capture_id(tmp_path):
    test_retained_topology_capture_path(tmp_path, 2, 'pass')
    log = tmp_path / 'coordinator.log'
    lines = log.read_text().splitlines()
    lines[-1:-1] = [
        'CUTEAFD_GRAPH_CENSUS {"event":"capture_ref","id":2,"exec":"0x1234"}',
        '2026-11-04T12:00:15.000000Z INFO cuteafd::graph_capture: graph census capture key '
        'family="glm5_flash" bank="decode" rank=0 key=256 exec=0x1234',
    ]
    log.write_text('\n'.join(lines) + '\n')
    topology = tmp_path / 'graph-census-captures-raptor.jsonl'
    topology.write_text(topology.read_text() + topology.read_text().replace('"id": 1', '"id": 2'))
    result = subprocess.run([sys.executable, '-I', str(SCRIPT), '--log', str(log),
                             '--sweep', str(tmp_path / 'sweep.json'), '--captures', str(topology)],
                            capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report['total_logged_captures'] == 2
    assert report['windows'][0]['captures'] == 1


def test_wip_profile_topology_is_discovered(tmp_path):
    test_retained_topology_capture_path(tmp_path, 2, 'pass')
    logs, profile = tmp_path / 'smoke/logs', tmp_path / 'profile'
    logs.mkdir(parents=True)
    profile.mkdir()
    (tmp_path / 'coordinator.log').rename(logs / 'coordinator.log')
    (tmp_path / 'graph-census-captures-raptor.jsonl').rename(profile / 'graph-census-captures-raptor.jsonl')
    # Spark identities are process-local; never join their ids to coordinator captures.
    (profile / 'graph-census-captures-dodo.jsonl').write_text('not coordinator topology')
    result = subprocess.run([sys.executable, '-I', str(SCRIPT), '--log', str(logs / 'coordinator.log'),
                             '--sweep', str(tmp_path / 'sweep.json')], capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)['total_logged_captures'] == 1
