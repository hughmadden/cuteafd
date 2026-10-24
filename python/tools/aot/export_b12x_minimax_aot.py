"""MiniMax M3 native coordinator programs for the coordinator exporter.

`export_b12x_dsv4_aot.py --geometry minimax` exports them (the `minimax`
family of the shared program archive); this module only lists them.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer  # noqa: F401  (verified SparkInfer source, pinned host target)


# Prefill scorer/extend tile: the scorer's block scores scale with rows x
# max_pages (539 MB at 4096 rows and 1M tokens), so prefill tiles a chunk.
EXTEND_ROWS = 1024


def capacity(name, max_rows=4096, decode_rows=64):
    if name.endswith("_decode"):
        return decode_rows
    return min(max_rows, EXTEND_ROWS) if name in ("index_prefill", "attention_extend") else max_rows


def builders(group, max_rows=4096, max_context=1048576, decode_rows=64):
    if min(max_rows, max_context, decode_rows) <= 0:
        raise ValueError("MiniMax capacities must be positive")
    if group == "indexer":
        from b12x.integration.cuteafd.minimax_indexer import compile_minimax_index_aot
        return {f"index_{mode}": (lambda mode=mode: compile_minimax_index_aot(
                    max_rows=capacity(f"index_{mode}", max_rows, decode_rows), max_pages=(max_context + 63) // 64,
                    mode=mode)) for mode in ("decode", "prefill")}
    if group == "attention":
        from b12x.integration.cuteafd.minimax_attention import compile_minimax_attention_aot
        return {f"attention_{mode}": (lambda mode=mode: compile_minimax_attention_aot(
                    max_rows=capacity(f"attention_{mode}", max_rows, decode_rows), max_batch=decode_rows, mode=mode))
                for mode in ("decode", "extend")}
    from b12x.integration.cuteafd import minimax as m
    if group == "projections":
        # Decode rows fill one M tile, so their programs split K instead.
        return {**{f"mxfp8_{name}": (lambda name=name: m.compile_minimax_linear_aot(name, max_rows=max_rows))
                   for name in m.LINEAR_SHAPES},
                **{f"mxfp8_{name}_decode": (lambda name=name: m.compile_minimax_linear_aot(
                       name, max_rows=decode_rows, splits=m.DECODE_SPLITS[name])) for name in m.DECODE_SPLITS}}
    if group != "pointwise":
        raise ValueError(f"unknown MiniMax export group: {group}")
    result = {
        "norm": m.compile_minimax_norm_aot,
        "scale_prep": m.compile_minimax_scale_prep_aot,
        "router_scores": m.compile_minimax_router_scores_aot,
        "route": m.compile_minimax_route_aot,
        "residual": m.compile_minimax_residual_aot,
        "rope": m.compile_minimax_rope_tables_aot,
        "index_q_quant": lambda: m.compile_minimax_index_quant_aot(heads=4),
        "index_k_pack": lambda: m.compile_minimax_index_quant_aot(heads=1, paged=True),
        "kv_pack": m.compile_minimax_kv_pack_aot,
        "kv_split": m.compile_minimax_kv_split_aot,
    }
    for name, heads in (("q", 64), ("k", 4), ("index_q", 4), ("index_k", 1)):
        result[name + "_norm_rope"] = lambda heads=heads: m.compile_minimax_qk_norm_rope_aot(heads=heads)
    for width in (3072, 12288):
        result[f"swiglu_i{width}"] = lambda width=width: m.compile_minimax_swiglu_aot(width=width)
    return result


def programs(max_rows, max_context, decode_rows=64):
    """Entries for the existing Programs dispatcher; capacities are planned bounds."""
    return [(name, name, {"max_rows": capacity(name, max_rows, decode_rows)}, build)
            for group in ("pointwise", "projections", "indexer", "attention")
            for name, build in builders(group, max_rows, max_context, decode_rows).items()]
