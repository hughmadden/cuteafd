"""CPU gates for the offline draft-length replay of fixed-K7 draft_confidence traces."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/bench/draft-confidence-replay.py"
SPEC = importlib.util.spec_from_file_location("draft_confidence_replay", TOOL)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)
KEY = "minimax_m3/dflash2/test-r1"


def synthetic(tmp_path, requests=16, blocks=60, first_id=5):
    """Blocks whose selector probability carries their acceptance; requests alternate thinking."""
    rng = np.random.default_rng(7)
    rows = []
    for request in range(requests):
        easy = request < requests // 2
        for _ in range(blocks):
            confidence = rng.normal(1.2 if easy else -0.2, 1.0, 7)
            accepted_each = rng.random(7) < 1 / (1 + np.exp(-2 * confidence))
            accepted = int(np.argmin(accepted_each)) if not accepted_each.all() else 7
            probability = np.clip(1 / (1 + np.exp(-1.5 * confidence + rng.normal(0, 0.3, 7))), 0.02, 1.0)
            features = [[float(abs(c) + 0.1), float(p), float(np.clip(1.5 - c, 0, 2.7)), float(rng.integers(0, 3))]
                        for c, p in zip(confidence, probability)]
            x = np.asarray([replay.calibrate.transform(0.6, i + 1, f) for i, f in enumerate(features)])
            rows.append(dict(type="draft_confidence", key=KEY, request=first_id + request, history=[0.6] * 7,
                             features=features, prior=replay.logistic(x, replay.GENERIC).tolist(), proposed=7,
                             accepted=accepted, observed=min(7, accepted + 1)))
    # A non-fixed record and a per-cycle row are ignored.
    rows.append(dict(rows=8, seqs=1))
    rows.append(dict(rows[0], proposed=3, accepted=1, observed=2))
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    lines = [json.dumps({"messages": [{"role": "user", "content": str(i)}],
                         "thinking": {"type": "enabled" if i % 2 else "disabled"}}) for i in range(requests)]
    request_file = tmp_path / "requests.jsonl"
    request_file.write_text("\n".join(lines) + "\n")
    return trace, request_file


def test_simple_rule_takes_the_most_drafts_above_one_half_joint():
    assert replay.depth([0.9, 0.9, 0.5, 0.99]) == 2
    assert replay.depth([0.4, 0.99]) == 0
    assert replay.depth([0.99] * 7) == 7
    assert replay.auc(np.asarray([0.1, 0.4, 0.35, 0.8]), np.asarray([0, 0, 1, 1])) == 0.75
    assert replay.auc(np.asarray([0.5, 0.5]), np.asarray([0, 1])) == 0.5


def test_admission_ids_map_to_request_lines_from_the_first_replayed_one():
    blocks = [dict(request=i) for i in (5, 6, 9)]
    assert replay.request_lines(blocks, 64, None) == {5: 0, 6: 1, 9: 4}
    # Earlier requests (a warm-up) fall before the replayed window.
    blocks = [dict(request=i) for i in [0, 1] + list(range(10, 74))]
    lines = replay.request_lines(blocks, 64, None)
    assert lines[10] == 0 and lines[73] == 63 and 0 not in lines
    assert replay.request_lines(blocks, 64, 1) == {i: i - 1 for i in [1] + list(range(10, 65))}


def test_replay_prices_rules_and_cross_validated_priors_per_class(tmp_path):
    trace, request_file = synthetic(tmp_path)
    out = tmp_path / "replay.json"
    result = subprocess.run([sys.executable, str(TOOL), "--trace", str(trace), "--requests", str(request_file),
                             "--classes", "easy:1-8,hard:9-16", "--key", KEY, "--json", str(out)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "960 fixed-K7 blocks from 16 of 16 requests" in result.stdout
    data = json.loads(out.read_text())
    table, quality = data["table"], data["quality"]
    assert data["recorded_prior_difference"] < 1e-12
    assert set(table) >= {"all", "easy", "hard", "easy/enabled", "hard/disabled"}
    for rules in table.values():
        assert rules["ceiling"]["tok_s"] >= max(row["tok_s"] for row in rules.values()) - 1e-9
        assert rules["fixed-7"]["drafts_per_cycle"] == 7
        assert rules["ceiling"]["tokens_per_cycle"] == rules["fixed-7"]["tokens_per_cycle"]
    # The selector carries acceptance here; a constant history cannot rank positions.
    assert quality["fitted"]["1"]["auc"] > 0.75 > quality["history"]["1"]["auc"]
    assert quality["fitted"]["1"]["logloss"] < quality["history"]["1"]["logloss"]
    assert table["all"]["fitted"]["tok_s"] > table["all"]["history"]["tok_s"]
    # Easy requests draft deeper than hard ones under a selector prior.
    assert table["easy"]["fitted"]["drafts_per_cycle"] > table["hard"]["fitted"]["drafts_per_cycle"]
