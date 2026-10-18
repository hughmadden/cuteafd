#!/usr/bin/env python3
"""Export native Step-5 pointwise and staged attention ABIs from the verified fork."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heads", type=int, choices=(32, 64), default=64)
    parser.add_argument("--group", choices=("pointwise", "attention", "metadata"), default="pointwise")
    parser.add_argument("--summary-capacity", type=int, default=131072)
    parser.add_argument("--max-staged-regions", type=int, default=513)
    args = parser.parse_args()
    if min(args.summary_capacity, args.max_staged_regions) <= 0:
        parser.error("summary and staged-region capacities must be positive")
    import _pinned_sparkinfer
    import torch
    from b12x.integration.cuteafd import exportable_compilation, validate_exported_header
    from b12x.integration.cuteafd import pointwise as p

    heads = args.heads
    groups = 4 if heads == 64 else 2
    builders = {
        "residual_bf16": lambda: p.compile_residual_add_aot(width=4096),
        "residual_f32": lambda: p.compile_residual_add_aot(width=4096, delta_dtype=torch.float32),
        "hidden_norm": lambda: p.compile_affine_norm_aot(width=4096, zero_centered=True),
        "index_q_norm": lambda: p.compile_affine_norm_aot(width=256, heads=groups * 4,
            eps=1e-6, zero_centered=True, rotary=32,
            input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16),
        "index_k_norm": lambda: p.compile_affine_norm_aot(width=256, eps=1e-6,
            subtract_mean=True, bias=True, rotary=32,
            input_dtype=torch.bfloat16, parameter_dtype=torch.bfloat16),
        "index_weight": lambda: p.compile_bf16_scaled_projection_aot(
            width=4096, outputs=groups * 4, prescale=0.5),
        "head_gate": lambda: p.compile_head_gate_aot(heads=heads, dim=192),
        "route_weights": lambda: p.compile_route_weights_aot(experts=352, topk=8),
    }
    for kind, rotary in (("swa", 192), ("csa", 64)):
        for name, count in (("q", heads), ("k", groups)):
            builders[f"{kind}_{name}_norm_rope"] = lambda count=count, rotary=rotary: p.compile_affine_norm_aot(
                width=192, heads=count, rotary=rotary, zero_centered=True,
                input_dtype=torch.bfloat16)
    # Shared/routed FFNs have per-layer limits; dense and MTP widths differ.
    for width in (1536, 13824):
        for limit in (7.0, 10.0):
            builders[f"post_silu_w{width}_c{int(limit)}"] = lambda width=width, limit=limit: p.compile_post_silu_aot(
                width=width, limit=limit)
    if args.group == "attention":
        from b12x.integration.cuteafd import region_attention as a
        builders = {
            "region_compress": a.compile_region_compress_aot,
            "region_scores": lambda: a.compile_region_scores_aot(
                groups=groups, capacity=args.summary_capacity),
            "swa_attention": lambda: a.compile_selected_gqa_aot(
                query_heads=heads, kv_heads=groups, selection_width=512, selection_groups=1),
            "csa_attention": lambda: a.compile_selected_gqa_aot(query_heads=heads, kv_heads=groups),
        }
    elif args.group == "metadata":
        from b12x.integration.cuteafd import cache_transactions as c, region_selection as s
        builders = {
            "sliding_positions": s.compile_sliding_positions_aot,
            "region_selection": lambda: s.compile_region_selection_aot(
                groups=groups, capacity=args.summary_capacity),
            "sliding_commit": lambda: c.compile_sliding_commit_aot(kv_heads=groups),
            "region_commit": lambda: c.compile_region_commit_aot(
                max_staged_regions=args.max_staged_regions, capacity=args.summary_capacity),
        }
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / f"step5_h{heads}_{args.group}.json"
    manifest.unlink(missing_ok=True)
    programs = []
    for name, build in builders.items():
        label = f"step5_h{heads}_{name}"
        with exportable_compilation():
            program = build()
        program.export_to_c(str(destination), label, "cuteafd_" + label)
        validate_exported_header(program, (destination / (label + ".h")).read_text(), "cuteafd_" + label)
        artifacts = {label + ext: hashlib.sha256((destination / (label + ext)).read_bytes()).hexdigest()
                     for ext in (".o", ".h")}
        programs.append(dict(label=label, geometry=dict(program.geometry), abi=program.abi,
                             scratch_bytes_per_row=program.scratch_bytes(1), artifacts=artifacts))
        print(f"exported {label}", flush=True)
    manifest.write_text(json.dumps(dict(schema=1, sparkinfer=_pinned_sparkinfer.LOCK_DATA,
                                        programs=programs), indent=2) + "\n")


if __name__ == "__main__":
    main()
