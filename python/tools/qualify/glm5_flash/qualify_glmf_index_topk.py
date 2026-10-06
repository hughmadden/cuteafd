#!/usr/bin/env python3
"""Qualify GLM 5.3 Flash's DSA index top-k at its long context extent (the
``glmf_index_topk_*_ctx{N}`` programs of CUTEAFD_GLMF_MAX_CONTEXT, 1,048,576 tokens by default:
262,144 pools in 4,096 pool pages), compiled in process on this GPU:

1. exact against a CPU reference on integer scores with ties at every pool, for 1, 64 and 4,096
   rows (the decode ``m64`` and ``m128`` and prefill ``m4096`` capacities), twice per case (the
   same bits);
2. at widths the plain (131,072-token, 512-page) programs cover, the long program's selection is
   the plain one's bit for bit, over one zeroed scratch both programs share, as in the engine;
3. the prepared plan of each program (route, supertile, chunks, scratch) and its launch time.

Integer scores: every pool key is one of a few basis keys, an E4M3 integer at one of the 128 dims
with an FP32 scale of 1, every query an E4M3 integer per head and dim, every head weight an integer,
so ``sum_h relu(q_h . k) * w_h`` is exact in FP32 in any order and whole groups of pools tie. The
reference keeps the top 512 by (score descending, pool ascending) in ascending pool order, then -1
padding (b12x ``tiled_topk``'s deterministic select), as physical slots ``page * 64 + row``.

  python3 python/tools/qualify/glm5_flash/qualify_glmf_index_topk.py [--context 1048576] \\
      [--json out.json] [--quick]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[2] / _d) for _d in ("lib",)]  # sibling tool dirs

import argparse
import json
import math
import time

import numpy as np

TOPK = 512  # index_topk 2048 / kpool 4
HEADS = 32
DIM = 128
PAGE = 64
PAGE_BYTES = 8448
BASE_CONTEXT = 131_072
# First physical page whose byte offset exceeds 2^31 (AGENTS.md big-pid rule).
HIGH_PAGE = (2**31) // PAGE_BYTES + 5


def pool_pages(context: int) -> int:
    """Pool-cache pages of ``context`` tokens: 64 pools of 4 tokens a page."""
    return -(-context // (4 * PAGE))


def reference_topk(scores: np.ndarray, positions: list[np.ndarray], lengths: np.ndarray, topk: int = TOPK):
    """The selection per row: ``scores`` int64 [rows, groups] (every pool of a group scores the
    same), ``positions[g]`` the ascending pool indices of group ``g``, ``lengths`` the pools each row
    sees (indices below it). Returns int64 [rows, topk]: the top ``topk`` pools by (score
    descending, index ascending), in ascending index order, then -1."""
    rows = scores.shape[0]
    out = np.full((rows, topk), -1, dtype=np.int64)
    for row in range(rows):
        n = int(lengths[row])
        visible = [p[: np.searchsorted(p, n)] for p in positions]
        chosen, left = [], topk
        for level in np.unique(scores[row])[::-1]:
            tied = [visible[g] for g in np.flatnonzero(scores[row] == level)]
            count = sum(p.size for p in tied)
            if count <= left:
                chosen.extend(tied)
                left -= count
            else:
                # The boundary: the lowest indices among the tied pools (one group's are sorted already).
                lowest = tied[0][:left] if len(tied) == 1 else np.sort(np.concatenate([p[:left] for p in tied]))[:left]
                chosen.append(lowest)
                left = 0
            if left == 0:
                break
        picks = np.sort(np.concatenate(chosen)) if chosen else np.empty(0, np.int64)
        out[row, : picks.size] = picks
    return out


def physical(logical: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Logical pool indices (-1 kept) as physical slots ``table[page] * 64 + row``."""
    slots = table[np.maximum(logical, 0) // PAGE].astype(np.int64) * PAGE + np.maximum(logical, 0) % PAGE
    return np.where(logical >= 0, slots, -1)


def group_sizes(pools: int, groups: int) -> list[int]:
    """Geometric group sizes (20, 30, 45, ...) and the rest in the last group, so each row's top 512
    ends inside one group (a tie at the boundary) whichever groups score highest."""
    sizes, size = [], 20.0
    for _ in range(groups - 1):
        sizes.append(min(int(size), max(0, pools - sum(sizes) - 1)))
        size *= 1.5
    sizes.append(pools - sum(sizes))
    return sizes


class Case:
    """Integer keys, queries and weights over ``pools`` pools of ``groups`` basis keys."""

    def __init__(self, pools: int, rows: int, seed: int, groups: int = 16):
        rng = np.random.default_rng(seed)
        self.pools, self.rows = pools, rows
        # Groups share dims and amplitudes in pairs now and then: equal keys, so ties across groups too.
        self.dim = rng.integers(0, DIM, groups)
        self.amplitude = rng.integers(1, 5, groups)
        for g in range(1, groups, 5):
            self.dim[g], self.amplitude[g] = self.dim[g - 1], self.amplitude[g - 1]
        self.group = np.repeat(np.arange(groups), group_sizes(pools, groups))
        rng.shuffle(self.group)
        self.positions = [np.flatnonzero(self.group == g) for g in range(groups)]
        self.q = rng.integers(-4, 5, (rows, HEADS, DIM))
        self.w = rng.integers(1, 4, (rows, HEADS))
        # score[row, g] = amplitude[g] * sum_h relu(q[row, h, dim[g]]) * w[row, h]
        self.scores = (np.maximum(self.q[:, :, self.dim], 0) * self.w[:, :, None]).sum(axis=1) * self.amplitude[None, :]

    def tensors(self, device, table: np.ndarray, cache_pages: int):
        """(q, w, cache): the queries (E4M3), the head weights (FP32) and the paged index cache with
        pool ``j`` at physical page ``table[j // 64]``."""
        import torch

        keys = torch.zeros((self.pools, DIM), dtype=torch.float32)
        keys[torch.arange(self.pools), torch.from_numpy(self.dim[self.group])] = \
            torch.from_numpy(self.amplitude[self.group]).float()
        data = keys.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1, PAGE * DIM)
        pages = data.shape[0]
        scales = torch.ones((pages, PAGE), dtype=torch.float32).view(torch.uint8)
        cache = torch.zeros((cache_pages, PAGE_BYTES), dtype=torch.uint8, device=device)
        index = torch.from_numpy(table[:pages].astype(np.int64)).to(device)
        cache[index] = torch.cat([data, scales], dim=1).to(device)
        q = torch.from_numpy(self.q).float().to(torch.float8_e4m3fn).to(device).contiguous()
        w = torch.from_numpy(self.w).float().to(device).contiguous()
        return q, w, cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context", type=int, default=1_048_576, help="the long extent (CUTEAFD_GLMF_MAX_CONTEXT)")
    parser.add_argument("--json", type=_Path, help="write the report here")
    parser.add_argument("--quick", action="store_true", help="fewer rows in the 4,096-row case (a smoke)")
    parser.add_argument("--repeat", type=int, default=5, help="timed launches per case (median)")
    args = parser.parse_args()

    import torch

    import _pinned_sparkinfer  # noqa: F401  (verifies and prepends the pinned b12x tree)
    from b12x.integration.cuteafd import GLM53_FLASH as g
    from b12x.integration.cuteafd import exportable_compilation, glmf
    from b12x.integration.cuteafd.dsv4_indexer import _prepared_layout

    device = torch.device("cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    long_pages, base_pages = pool_pages(args.context), pool_pages(BASE_CONTEXT)
    pools = long_pages * PAGE
    report = {"context": args.context, "pools": pools, "pool_pages": long_pages, "sms": sms,
              "device": torch.cuda.get_device_name(0), "programs": {}, "cases": [], "ab": []}
    geometry = glmf._PoolTopK(index_topk=g.index_topk // g.index_kpool, index_heads=g.index_heads)

    programs = {}
    with exportable_compilation():
        for mode, rows in (("decode", 64), ("decode", 128), ("prefill", 4096)):
            for pages in (base_pages, long_pages):
                started = time.time()
                program = glmf.compile_glmf_index_topk_aot(g, max_rows=rows, max_pages=pages, mode=mode)
                layout = _prepared_layout(geometry, rows, pages, mode, g.index_heads)
                programs[(mode, rows, pages)] = program
                report["programs"][f"{mode}_m{rows}_p{pages}"] = {
                    "route": layout.route, "supertile": int(layout.supertile_tokens),
                    "max_chunks": int(layout.max_chunks), "stream_scorer_ctas": int(layout.stream_scorer_ctas),
                    "scratch": int(program.scratch_bytes(rows)["scratch"]),
                    "compile_s": round(time.time() - started, 1)}
                print(f"compiled {mode} m{rows} at {pages} pages: {report['programs'][f'{mode}_m{rows}_p{pages}']}",
                      flush=True)

    def launch(program, q, w, cache, table, lengths, width, stride, scratch, out=None):
        if out is None:
            out = torch.full((q.shape[0], TOPK), -7, dtype=torch.int32, device=device)
        program.launch(q, w, cache, table, lengths, out, scratch, scalars=(q.shape[0], width, stride))
        return out

    def timed(fn):
        times = []
        for _ in range(args.repeat):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return round(float(np.median(times)), 3)

    failures = 0
    # One zeroed scratch per capacity, as large as either program's, never zeroed again (the engine's).
    scratch = {key[:2]: torch.zeros((max(programs[(key[0], key[1], p)].scratch_bytes(key[1])["scratch"]
                                        for p in (base_pages, long_pages)),), dtype=torch.uint8, device=device)
               for key in programs}
    cases = [("decode", 64, 1, False), ("decode", 64, 64, True), ("decode", 128, 128, False),
             ("prefill", 4096, 1, False), ("prefill", 4096, 512 if args.quick else 4096, True)]
    for seed, (mode, cap, rows, high) in enumerate(cases):
        case = Case(pools, rows, seed)
        rng = np.random.default_rng(100 + seed)
        base = HIGH_PAGE if high else 0
        table = (rng.permutation(2 * long_pages)[:long_pages] + base).astype(np.int32)
        q, w, cache = case.tensors(device, table, base + 2 * long_pages + 1)
        if mode == "prefill":
            # The tail of a 1M prompt: row t sees (position + 1) // 4 pools.
            positions = np.arange(rows) + args.context - rows
            lengths = np.minimum((positions + 1) // 4, pools)
            table_arg, stride = torch.from_numpy(table).to(device), 0
        else:
            lengths = rng.integers(pools // 2, pools + 1, rows)
            lengths[0] = pools
            if rows > 2:
                lengths[1] = TOPK // 2  # fewer visible pools than 512: -1 padding
                lengths[2] = TOPK      # exactly 512
            table_arg, stride = torch.from_numpy(np.tile(table, (rows, 1))).to(device), long_pages
        lengths_t = torch.from_numpy(lengths.astype(np.int32)).to(device)
        program = programs[(mode, cap, long_pages)]
        buffer = scratch[(mode, cap)]
        first = launch(program, q, w, cache, table_arg, lengths_t, long_pages, stride, buffer).cpu().numpy()
        second = launch(program, q, w, cache, table_arg, lengths_t, long_pages, stride, buffer).cpu().numpy()
        expected = physical(reference_topk(case.scores, case.positions, lengths), table)
        exact = bool(np.array_equal(first, expected))
        same = bool(np.array_equal(first, second))
        bad = np.flatnonzero((first != expected).any(axis=1))
        sink = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
        ms = timed(lambda: launch(program, q, w, cache, table_arg, lengths_t, long_pages, stride, buffer, sink))
        failures += not (exact and same)
        entry = {"mode": mode, "capacity": cap, "rows": rows, "pools": pools, "high_pages": high,
                 "exact": exact, "repeatable": same, "rows_differing": int(bad.size),
                 "first_differing_row": int(bad[0]) if bad.size else None, "ms": ms}
        report["cases"].append(entry)
        print(f"{'PASS' if exact and same else 'FAIL'} {entry}", flush=True)

    # Widths the plain programs cover: the long program selects the same bits (shared scratch).
    for seed, (mode, cap, rows, width) in enumerate([("decode", 64, 64, 512), ("decode", 64, 7, 300),
                                                     ("decode", 128, 128, 512), ("prefill", 4096, 4096 // 8, 512),
                                                     ("prefill", 4096, 64, 129)]):
        short = width * PAGE
        case = Case(short, rows, 50 + seed)
        rng = np.random.default_rng(200 + seed)
        table = rng.permutation(2 * width)[:width].astype(np.int32)
        q, w, cache = case.tensors(device, table, 2 * width + 1)
        if mode == "prefill":
            lengths = np.minimum((np.arange(rows) + width * 256 - rows + 1) // 4, short)
            table_arg, stride = torch.from_numpy(table).to(device), 0
        else:
            lengths = rng.integers(1, short + 1, rows)
            table_arg, stride = torch.from_numpy(np.tile(table, (rows, 1))).to(device), width
        lengths_t = torch.from_numpy(lengths.astype(np.int32)).to(device)
        buffer = scratch[(mode, cap)]
        outs = {}
        for pages in (long_pages, base_pages, long_pages):
            outs.setdefault(pages, []).append(launch(programs[(mode, cap, pages)], q, w, cache, table_arg,
                                                     lengths_t, width, stride, buffer).cpu().numpy())
        expected = physical(reference_topk(case.scores, case.positions, lengths), table)
        identical = all(np.array_equal(o, outs[base_pages][0]) for o in outs[long_pages])
        exact = bool(np.array_equal(outs[base_pages][0], expected))
        failures += not (identical and exact)
        sink = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
        entry = {"mode": mode, "capacity": cap, "rows": rows, "width": width, "long_equals_plain": identical,
                 "exact": exact,
                 "plain_ms": timed(lambda: launch(programs[(mode, cap, base_pages)], q, w, cache, table_arg, lengths_t,
                                                  width, stride, buffer, sink)),
                 "long_ms": timed(lambda: launch(programs[(mode, cap, long_pages)], q, w, cache, table_arg, lengths_t,
                                                 width, stride, buffer, sink))}
        report["ab"].append(entry)
        print(f"{'PASS' if identical and exact else 'FAIL'} {entry}", flush=True)

    # Whether the exported routes depend on the card: the stream scorer's CTAs per row are the
    # one SM-derived value the exported (non-fused) routes take.
    report["stream_scorer_ctas_by_sms"] = {str(n): {str(r): math.ceil(n / r) for r in (64, 128, 4096)}
                                           for n in (170, 188, sms)}
    report["passed"] = failures == 0
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    print(f"{'PASSED' if failures == 0 else f'FAILED ({failures})'}: {len(report['cases'])} exactness cases, "
          f"{len(report['ab'])} plain-vs-long cases", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
