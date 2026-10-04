"""Acceptance failures must never promote a target vocabulary conversion."""
import copy
import runpy
from pathlib import Path

import pytest

compare = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualify/deepseek_v41/compare-fp8-head.py"))["compare"]


def report():
    return {
        "status": "done",
        "server": {"model": "v41", "revision": "r", "hardware": {}, "build": {}},
        "baseline": {"quality": {"checks": [{"id": "fidelity", "metrics": {
            "reference": "golden", "kl": 0.002, "nll": 2.0, "ref_nll": 2.0,
            "top1": 0.5, "positions": 2, "missing": 0,
            "probe": {"engine": "v41", "error": None, "cold": True, "cached_tokens": 0,
                "prompt_ids": [0, 1, 2], "scored": 2,
                "rows": [{"position": 1, "argmax": 1, "finite": True},
                         {"position": 2, "argmax": 2, "finite": True}]},
        }}]}},
    }


def metrics(r):
    return r["baseline"]["quality"]["checks"][0]["metrics"]


def test_kl_rise_is_relative_to_bf16():
    a = report()
    b = copy.deepcopy(a)
    metrics(b)["kl"] = 0.006
    assert compare(a, b)["passed"], "absolute golden KL > .005 is not the gate"
    metrics(b)["kl"] = 0.008
    assert not compare(a, b)["passed"]


def test_top1_flip_fails_even_when_aggregate_golden_score_is_unchanged():
    a, b = report(), report()
    metrics(b)["probe"]["rows"][1]["argmax"] = 9
    result = compare(a, b)
    assert not result["passed"]
    assert result["top1_flip_positions"] == [2]
    assert result["bf16_top1_agreement"] == 0.5


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nonfinite", "warm", "prompt", "build"])
def test_invalid_comparisons_fail_closed(fault):
    a, b = report(), report()
    m = metrics(b)
    if fault == "missing":
        m["probe"]["rows"].pop()
    elif fault == "duplicate":
        m["probe"]["rows"][1]["position"] = 1
    elif fault == "nonfinite":
        m["probe"]["rows"][1]["finite"] = False
    elif fault == "warm":
        m["probe"]["cached_tokens"] = 1
    elif fault == "prompt":
        m["probe"]["prompt_ids"][0] = 9
    elif fault == "build":
        b["server"]["build"] = {"commit": "other"}
    with pytest.raises(ValueError):
        compare(a, b)
