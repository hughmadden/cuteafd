"""Offline Spark load-balance scorer for PROFILE=expert-routes logs (#408)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('expert_route_balance', ROOT / 'scripts/bench/expert-route-balance.py')
erb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(erb)


def _uniform(tp, rule, unique, steps=8000):
    result = erb.score(erb.synthetic_uniform([unique], steps, 408), [tp], [rule], 'unique', False, None, 0)
    return result['rules'][rule][str(tp)]


def test_rules_place_every_unit_once():
    for rule in erb.RULES:
        for tp in (4, 6):
            for expert in (0, 1, 2, 255):
                assert sum(erb.rank_units(rule, expert, 40, tp)) == erb.UNITS


def test_exlr8_rule_matches_the_design_comment():
    # rank = (16e + u + L) mod N; TP6 rank 1 at L040 owns shared (e = 256) units {5, 11}.
    owned = [u for u in range(16) if erb.unit_rank('exlr8-unit', 256, u, 40, 6) == 1]
    assert owned == [5, 11]
    # 16e mod 4 = 0: every expert puts exactly 4 units on each TP4 rank.
    assert all(erb.rank_units('exlr8-unit', e, L, 4) == [4] * 4 for e in range(256) for L in (3, 40, 77))
    # 16e mod 6 cycles 0, 4, 2: a 3/3/3/3/2/2 split starting at a different rank.
    assert sorted(erb.rank_units('exlr8-unit', 7, 40, 6)) == [2, 2, 3, 3, 3, 3]


def test_uniform_routes_reproduce_408():
    assert _uniform(6, 'exl3-k4', 8)['mean'] == pytest.approx(0.1875)
    assert _uniform(6, 'exl3-k4', 43)['p99'] == pytest.approx(0.1875)
    assert _uniform(4, 'exl3-k4', 23)['mean'] == pytest.approx(0.25)
    assert _uniform(4, 'exlr8-unit', 23)['mean'] == pytest.approx(0.25)
    # #408: 0.177 / 0.173 / 0.171 at U = 8 / 23 / 43 (TP6, any manifest).
    for unique, expected in ((8, 0.177), (23, 0.173), (43, 0.171)):
        mean = _uniform(6, 'exlr8-unit', unique)['mean']
        assert 0.170 <= mean <= 0.1775 and mean == pytest.approx(expected, abs=0.0015), (unique, mean)


def test_exlr8_is_never_worse_than_exl3_at_tp6():
    # At most 3 units of one expert on one rank under either rule.
    for unique in (1, 2, 8):
        s = _uniform(6, 'exlr8-unit', unique, steps=2000)
        assert s['max'] <= 0.1875 + 1e-12
        assert set(s['busiest_rank_units_per_expert']) <= {'2', '3'}


def test_log_records_score_from_topk_and_expert_rows(tmp_path):
    log = tmp_path / 'expert-routes-raptor.jsonl'
    records = [
        {'event': 'expert_routes', 'kind': 'mtp_verify', 'layer': 3, 'rows': 2, 'topk': [[0, 1], [1, 2]],
         'weights': [[0.5, 0.5], [0.5, 0.5]]},
        {'event': 'expert_routes', 'kind': 'prefill', 'layer': 4, 'rows': 100, 'expert_rows': [[5, 60], [9, 140]]},
        {'event': 'expert_timing'},
    ]
    log.write_text('\n'.join(json.dumps(r) for r in records) + '\n{"event": "expert_rou')
    out = tmp_path / 'score.json'
    assert erb.main([str(log), '--tp', '6', '--rule', 'exl3-k4', '--json', str(out)]) == 0
    result = json.loads(out.read_text())
    assert result['kinds_seen'] == {'mtp_verify': 1, 'prefill': 1}
    assert result['rules']['exl3-k4']['6']['layer_steps'] == 2
    assert erb.record_experts(records[0]) == {0: 1, 1: 2, 2: 1}
    # Kind filter and row weighting.
    assert erb.main([str(log), '--tp', '4', '--kind', 'decode']) == 2
    rows = erb.score(erb.read_log([log]), [6], ['exlr8-unit'], 'rows', False, {'prefill'}, 0)
    loads, _ = erb.step_loads({5: 60, 9: 140}, 4, 'exlr8-unit', 6, 'rows', False)
    assert rows['rules']['exlr8-unit']['6']['mean'] == pytest.approx(max(loads) / sum(loads))


def test_shared_expert_joins_every_layer_step():
    loads, placed = erb.step_loads({1: 1}, 40, 'exlr8-unit', 6, 'unique', True)
    assert len(placed) == 2 and sum(loads) == 32


def test_cli_refuses_bad_arguments(capsys):
    for argv in (['--synthetic-uniform', '8', '--rule', 'nope'], ['--synthetic-uniform', '8', '--tp', '0'], []):
        with pytest.raises(SystemExit):
            erb.main(argv)
