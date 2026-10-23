#!/usr/bin/env python3
"""Export MiniMax M3 native coordinator programs and their named operand ABIs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer


def builders(group, max_rows=4096, max_context=1048576, decode_rows=64):
    if min(max_rows, max_context, decode_rows) <= 0:
        raise ValueError("MiniMax capacities must be positive")
    if group == "indexer":
        from b12x.integration.cuteafd.minimax_indexer import compile_minimax_index_aot
        return {f"index_{mode}": (lambda mode=mode, rows=rows: compile_minimax_index_aot(
                    max_rows=rows, max_pages=(max_context + 63) // 64, mode=mode))
                for mode, rows in (("decode", decode_rows), ("prefill", max_rows))}
    if group == "attention":
        from b12x.integration.cuteafd.minimax_attention import compile_minimax_attention_aot
        return {f"attention_{mode}": (lambda mode=mode, rows=rows: compile_minimax_attention_aot(
                    max_rows=rows, max_batch=decode_rows, mode=mode))
                for mode, rows in (("decode", decode_rows), ("extend", max_rows))}
    from b12x.integration.cuteafd import minimax as m
    if group == "projections":
        return {f"mxfp8_{name}": (lambda name=name: m.compile_minimax_linear_aot(name, max_rows=max_rows))
                for name in m.LINEAR_SHAPES}
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
        "index_k_quant": lambda: m.compile_minimax_index_quant_aot(heads=1),
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
    return [(name, name, {"max_rows": decode_rows if name.endswith("_decode") else max_rows}, build)
            for group in ("pointwise", "projections", "indexer", "attention")
            for name, build in builders(group, max_rows, max_context, decode_rows).items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", choices=("pointwise", "projections", "indexer", "attention"), default="pointwise")
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--decode-rows", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=1048576)
    args = parser.parse_args()
    if min(args.max_rows, args.decode_rows, args.max_context) <= 0:
        parser.error("capacities must be positive")
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header

    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / f"minimax_{args.group}.json"
    manifest.unlink(missing_ok=True)
    records = []
    for name, build in builders(args.group, args.max_rows, args.max_context, args.decode_rows).items():
        label = "minimax_" + name
        with exportable_compilation():
            program = build()
        program.export_to_c(str(destination), label, "cuteafd_" + label)
        validate_exported_header(program, destination / (label + ".h"), "cuteafd_" + label)
        capacity = args.decode_rows if name.endswith("_decode") else args.max_rows
        artifacts = {label + ext: hashlib.sha256((destination / (label + ext)).read_bytes()).hexdigest()
                     for ext in (".o", ".h")}
        records.append(dict(label=label, geometry=dict(program.geometry), abi=program.abi,
                            capacity_rows=capacity, scratch_bytes_at_capacity=program.scratch_bytes(capacity),
                            artifacts=artifacts))
        print(f"exported {label}", flush=True)
    manifest.write_text(json.dumps(dict(schema=1, sparkinfer=_pinned_sparkinfer.LOCK_DATA,
                                        programs=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
