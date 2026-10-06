"""The CPU half of python/tools/qualify/glm5_flash/qualify_glmf_index_topk.py: its integer-score
cases and the reference selection it holds the GLM 5.3 Flash index top-k to (no GPU, no torch)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "qualify" / "glm5_flash" / "qualify_glmf_index_topk.py"
SPEC = importlib.util.spec_from_file_location("qualify_glmf_index_topk", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def brute(scores, group, lengths, topk):
    """Every visible pool sorted by (score descending, index ascending): the first `topk`, ascending."""
    out = np.full((scores.shape[0], topk), -1, dtype=np.int64)
    for row in range(scores.shape[0]):
        n = int(lengths[row])
        order = sorted(range(n), key=lambda j: (-int(scores[row, group[j]]), j))[:topk]
        out[row, : len(order)] = sorted(order)
    return out


@pytest.mark.parametrize("seed", range(40))
def test_the_reference_is_the_score_then_index_top_k(seed):
    rng = np.random.default_rng(seed)
    pools, groups = int(rng.integers(1, 3000)), int(rng.integers(1, 9))
    topk = int(rng.choice([1, 7, 16, 512]))
    rows = 5
    group = rng.integers(0, groups, pools)
    positions = [np.flatnonzero(group == g) for g in range(groups)]
    # Few distinct values: ties between groups as well as within them.
    scores = rng.integers(0, 4, (rows, groups))
    lengths = rng.integers(0, pools + 1, rows)
    lengths[0], lengths[-1] = pools, min(pools, topk // 2)
    assert np.array_equal(MODULE.reference_topk(scores, positions, lengths, topk), brute(scores, group, lengths, topk))


def test_cases_score_exactly_what_the_keys_queries_and_weights_give():
    case = MODULE.Case(pools=64 * 40, rows=6, seed=3)
    keys = np.zeros((case.pools, MODULE.DIM), dtype=np.int64)
    keys[np.arange(case.pools), case.dim[case.group]] = case.amplitude[case.group]
    for row in range(case.rows):
        direct = (np.maximum(np.einsum("hd,pd->hp", case.q[row], keys), 0) * case.w[row][:, None]).sum(axis=0)
        assert np.array_equal(direct, case.scores[row, case.group])
    # Integers small enough for E4M3 (|q| <= 4, amplitude 1..4) and FP32 sums (< 2^24).
    assert np.abs(case.q).max() <= 4 and set(np.unique(case.amplitude)) <= {1, 2, 3, 4}
    assert case.scores.max() < 2**24
    # Some groups share a key: equal scores across groups on every row.
    assert any(np.array_equal(case.scores[:, a], case.scores[:, b]) for a in range(16) for b in range(a))


def test_group_sizes_cover_the_pools_and_put_the_boundary_inside_a_group():
    for pools in (64 * 129, 64 * 512, 262_144):
        sizes = MODULE.group_sizes(pools, 16)
        assert sum(sizes) == pools and min(sizes) >= 0 and len(sizes) == 16
    case = MODULE.Case(pools=262_144, rows=64, seed=0)
    lengths = np.full(64, 262_144)
    picks = MODULE.reference_topk(case.scores, case.positions, lengths)
    # Every row fills 512, and nearly every row ends inside a tied group (some pool of its lowest
    # selected score is left out), where the index order decides.
    inside = 0
    for row in range(64):
        chosen = picks[row]
        assert (chosen >= 0).all()
        boundary = case.scores[row, case.group[chosen]].min()
        inside += (case.scores[row, case.group] == boundary).sum() > (case.scores[row, case.group[chosen]] == boundary).sum()
    assert inside >= 48, inside


def test_physical_slots_follow_the_page_table():
    table = np.array([7, 3, 9], dtype=np.int32)
    logical = np.array([[0, 63, 64, 130, -1]])
    assert MODULE.physical(logical, table).tolist() == [[7 * 64, 7 * 64 + 63, 3 * 64, 9 * 64 + 2, -1]]
    assert (MODULE.pool_pages(1_048_576), MODULE.pool_pages(131_072), MODULE.pool_pages(1_000_000)) == (4096, 512, 3907)
