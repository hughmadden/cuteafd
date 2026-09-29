#!/usr/bin/env python3
"""Export the coordinator programs (b12x.integration.cuteafd): DeepSeek V4
(``--geometry flash,pro``, families ``dsv4f``/``dsv4p``), GLM 5.x
(``--geometry glm``, family ``glm``), MiMo V2 Flash (``--geometry mimo``,
family ``mimo``) and GLM 5.3 Flash (``--geometry glmf``, family ``glmf``), in
any combination, into one table.

One object and header per program, a manifest with every program's pointer
ABI and scratch sizes at its capacity, and ``dsv4_programs.h``: the table the
generic native shim (native/src/dsv4_programs.cc) launches from. Capacities are
compile-time: decode programs cover ``--decode-rows``, prefill programs
``--prefill-rows``, and cache extents follow ``--max-context``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
import _pinned_sparkinfer

SCALAR_KINDS = {"int32": "i", "int64": "l", "float32": "f"}
PAGE_ROWS = 64


def programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """(stem suffix, op, params, compile thunk) for every exported program."""
    from b12x.integration.cuteafd import dsv4_compressor as comp
    from b12x.integration.cuteafd import dsv4_ffn as ffn
    from b12x.integration.cuteafd import weights
    from b12x.integration.cuteafd import dsv4_indexer as idx
    from b12x.integration.cuteafd import dsv4_mhc as mhc
    from b12x.integration.cuteafd import dsv4_producer as prod
    from b12x.integration.cuteafd import dsv4_sparse_mla as mla
    from b12x.integration.cuteafd import dsv4_wo as wo

    # Index cache: one row per four tokens, 64 rows per page.
    index_pages = -(-max_context // 4 // PAGE_ROWS)
    # C128 layers attend every completed compressed entry.
    c128_width = -(-max_context // 128 // 64) * 64
    out = [
        ("mhc_pre", "mhc_pre", {}, lambda: mhc.compile_dsv4_mhc_pre_aot(g)),
        ("mhc_post", "mhc_post", {}, lambda: mhc.compile_dsv4_mhc_post_aot(g)),
        ("mhc_head", "mhc_head", {}, lambda: mhc.compile_dsv4_mhc_head_aot(g)),
        ("router_scores", "router_scores", {}, lambda: ffn.compile_dsv4_router_scores_aot(g)),
        ("expert_input_quant", "expert_input_quant", {},
         lambda: ffn.compile_dsv4_expert_input_quant_aot(g)),
        # Load-time weight preparation (the rest of every weight is raw bytes).
        ("block_fp8_scale_prep", "block_fp8_scale_prep", {},
         lambda: weights.compile_dsv4_block_fp8_scale_prep_aot()),
        ("i64_to_i32", "i64_to_i32", {}, lambda: weights.compile_dsv4_i64_to_i32_aot()),
    ]
    for rows in (decode_rows, prefill_rows):
        out += [
            (f"mhc_post_pre_m{rows}", "mhc_post_pre", {"max_rows": rows},
             lambda r=rows: mhc.compile_dsv4_mhc_post_pre_aot(g, max_rows=r)),
            (f"producer_m{rows}", "producer", {"max_rows": rows},
             lambda r=rows: prod.compile_dsv4_producer_aot(g, max_rows=r)),
            (f"index_producer_m{rows}", "index_producer", {"max_rows": rows},
             lambda r=rows: prod.compile_dsv4_index_producer_aot(g, max_rows=r)),
            (f"wo_m{rows}", "wo", {"max_rows": rows},
             lambda r=rows: wo.compile_dsv4_wo_projection_aot(g, max_rows=r)),
            (f"shared_ffn_m{rows}", "shared_ffn", {"max_rows": rows},
             lambda r=rows: ffn.compile_dsv4_shared_ffn_aot(g, max_rows=r)),
        ]
    for ratio in (4, 128):
        for mode in ("decode", "prefill", "continuation"):
            fn = getattr(comp, f"compile_dsv4_compressor_{mode}_aot")
            out.append((f"compressor_{mode}_c{ratio}", f"compressor_{mode}", {"ratio": ratio},
                        lambda fn=fn, ratio=ratio: fn(g, ratio=ratio)))
    for mode, rows in (("decode", decode_rows), ("prefill", prefill_rows)):
        out.append((f"index_topk_{mode}_m{rows}", "index_topk",
                    {"mode": mode, "max_rows": rows, "max_pages": index_pages},
                    lambda mode=mode, rows=rows: idx.compile_dsv4_index_topk_aot(
                        g, max_rows=rows, max_pages=index_pages, mode=mode)))
    attention = (("win", 0, 64), ("c4", g.index_topk, 64), ("c128", c128_width, 2))
    for route, rows in (("decode", decode_rows), ("prefill", prefill_rows)):
        for kind, width, page_rows in attention:
            out.append((f"sparse_mla_{route}_{kind}_m{rows}", "sparse_mla",
                        {"route": route, "max_rows": rows, "indexed_width": width,
                         "indexed_page_rows": page_rows},
                        lambda route=route, rows=rows, width=width, page_rows=page_rows:
                            mla.compile_dsv4_sparse_mla_aot(
                                g, route=route, max_rows=rows, indexed_width=width,
                                indexed_page_rows=page_rows)))
    return out


def glm_programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """GLM 5.x programs, same (stem suffix, op, params, thunk) shape as ``programs``."""
    from b12x.integration.cuteafd import glm_attention as attn
    from b12x.integration.cuteafd import glm_ffn as ffn
    from b12x.integration.cuteafd import glm_indexer as idx
    from b12x.integration.cuteafd import glm_sparse_mla as mla

    # Index and latent caches: one row per token, 64 rows per page.
    index_pages = -(-max_context // PAGE_ROWS)
    out = [
        ("norm", "norm", {}, lambda: ffn.compile_glm_norm_aot(g)),
        ("router_scores", "router_scores", {}, lambda: ffn.compile_glm_router_scores_aot(g)),
        ("expert_input_quant", "expert_input_quant", {}, lambda: ffn.compile_glm_expert_input_quant_aot(g)),
    ]
    # Decode programs also take the FP8 checkpoint weights (E4M3 + FP32 block
    # scales) for their few-row GEMVs; prefill programs take BF16 only.
    for rows in (decode_rows, prefill_rows):
        f8 = rows == decode_rows
        out += [
            (f"producer_m{rows}", "producer", {"max_rows": rows, "fp8": f8},
             lambda r=rows, f=f8: attn.compile_glm_producer_aot(g, max_rows=r, fp8=f)),
            (f"index_producer_m{rows}", "index_producer", {"max_rows": rows, "fp8": f8},
             lambda r=rows, f=f8: attn.compile_glm_index_producer_aot(g, max_rows=r, fp8=f)),
            (f"o_m{rows}", "o", {"max_rows": rows, "fp8": f8},
             lambda r=rows, f=f8: attn.compile_glm_o_aot(g, max_rows=r, fp8=f)),
        ]
        for inter in (g.moe_inter, g.dense_inter):
            out.append((f"ffn_i{inter}_m{rows}", "ffn", {"max_rows": rows, "inter": inter, "fp8": f8},
                        lambda r=rows, i=inter, f=f8: ffn.compile_glm_ffn_aot(g, inter=i, max_rows=r, fp8=f)))
    for mode, rows in (("decode", decode_rows), ("prefill", prefill_rows)):
        out.append((f"index_topk_{mode}_m{rows}", "index_topk",
                    {"mode": mode, "max_rows": rows, "max_pages": index_pages},
                    lambda mode=mode, rows=rows: idx.compile_glm_index_topk_aot(
                        g, max_rows=rows, max_pages=index_pages, mode=mode)))
        out.append((f"sparse_mla_{mode}_m{rows}", "sparse_mla", {"route": mode, "max_rows": rows},
                    lambda mode=mode, rows=rows: mla.compile_glm_sparse_mla_aot(g, route=mode, max_rows=rows)))
    return out


def mimo_programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """MiMo V2 programs, same (stem suffix, op, params, thunk) shape as ``programs``.
    BF16 weights throughout (FP8 decode weights are a later step)."""
    from b12x.integration.cuteafd import mimo_attention as attn
    from b12x.integration.cuteafd import mimo_ffn as ffn

    out = [
        ("norm", "norm", {}, lambda: ffn.compile_mimo_norm_aot(g)),
        ("router_scores", "router_scores", {}, lambda: ffn.compile_mimo_router_scores_aot(g)),
        ("expert_input_quant", "expert_input_quant", {}, lambda: ffn.compile_mimo_expert_input_quant_aot(g)),
    ]
    for mode, rows in (("decode", decode_rows), ("prefill", prefill_rows)):
        out += [
            (f"o_m{rows}", "o", {"max_rows": rows}, lambda r=rows: attn.compile_mimo_o_aot(g, max_rows=r)),
            (f"ffn_m{rows}", "ffn", {"max_rows": rows, "inter": g.dense_inter},
             lambda r=rows: ffn.compile_mimo_ffn_aot(g, max_rows=r)),
        ]
        for kind in ("full", "swa"):
            out += [
                (f"{kind}_producer_m{rows}", "producer", {"kind": kind, "max_rows": rows},
                 lambda k=kind, r=rows: attn.compile_mimo_producer_aot(g, kind=k, max_rows=r)),
                (f"{kind}_attention_{mode}_m{rows}", "attention", {"kind": kind, "route": mode, "max_rows": rows},
                 lambda k=kind, m=mode, r=rows: attn.compile_mimo_attention_aot(g, kind=k, route=m, max_rows=r)),
            ]
    return out


def glmf_programs(g, decode_rows: int, prefill_rows: int, max_context: int):
    """GLM 5.3 Flash programs, same (stem suffix, op, params, thunk) shape as ``programs``.
    mHC is the DeepSeek V4 program set at this model's width and epsilons."""
    from b12x.integration.cuteafd import dsv4_mhc as mhc
    from b12x.integration.cuteafd import glm_attention as attn
    from b12x.integration.cuteafd import glm_sparse_mla as mla
    from b12x.integration.cuteafd import glmf

    mg = glmf.mhc_geometry(g)
    # Pool index cache: 64 pools (256 tokens) per page.
    pool_pages = -(-max_context // (g.index_kpool * PAGE_ROWS))
    out = [
        ("mhc_pre", "mhc_pre", {}, lambda: mhc.compile_dsv4_mhc_pre_aot(mg)),
        ("mhc_post", "mhc_post", {}, lambda: mhc.compile_dsv4_mhc_post_aot(mg)),
        ("head", "head", {}, lambda: glmf.compile_glmf_head_aot(g)),
        ("add", "add", {}, lambda: glmf.compile_glmf_add_aot(g)),
        ("router_scores", "router_scores", {}, lambda: glmf.compile_glmf_router_scores_aot(g)),
        ("expert_input_quant", "expert_input_quant", {}, lambda: glmf.compile_glmf_expert_input_quant_aot(g)),
        ("index_expand", "index_expand", {}, lambda: glmf.compile_glmf_index_expand_aot(g)),
    ]
    for mode, rows in (("decode", decode_rows), ("prefill", prefill_rows)):
        out += [
            (f"index_producer_m{rows}", "index_producer", {"max_rows": rows},
             lambda r=rows: glmf.compile_glmf_index_producer_aot(g, max_rows=r)),
            (f"index_topk_{mode}_m{rows}", "index_topk", {"mode": mode, "max_rows": rows, "max_pages": pool_pages},
             lambda m=mode, r=rows: glmf.compile_glmf_index_topk_aot(g, max_rows=r, max_pages=pool_pages, mode=m)),
            (f"mhc_post_pre_m{rows}", "mhc_post_pre", {"max_rows": rows},
             lambda r=rows: mhc.compile_dsv4_mhc_post_pre_aot(mg, max_rows=r)),
            (f"kda_m{rows}", "kda", {"max_rows": rows}, lambda r=rows: glmf.compile_glmf_kda_aot(g, max_rows=r)),
            (f"mla_producer_m{rows}", "mla_producer", {"max_rows": rows},
             lambda r=rows: glmf.compile_glmf_mla_producer_aot(g, max_rows=r)),
            (f"o_m{rows}", "o", {"max_rows": rows}, lambda r=rows: attn.compile_glm_o_aot(g, max_rows=r)),
            (f"sparse_mla_{mode}_m{rows}", "sparse_mla", {"route": mode, "max_rows": rows},
             lambda m=mode, r=rows: mla.compile_glm_sparse_mla_aot(g, route=m, max_rows=r, name="glmf_sparse_mla")),
        ]
        for inter in (g.moe_inter, g.dense_inter):
            out.append((f"ffn_i{inter}_m{rows}", "ffn", {"max_rows": rows, "inter": inter},
                        lambda r=rows, i=inter: glmf.compile_glmf_ffn_aot(g, inter=i, max_rows=r)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry", default="flash",
                        help="comma-separated geometries in one table: flash, pro (DeepSeek V4), glm (GLM 5.x), "
                             "mimo (MiMo V2 Flash), glmf (GLM 5.3 Flash)")
    parser.add_argument("--decode-rows", type=int, default=64)
    parser.add_argument("--prefill-rows", type=int, default=4096)
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument("--only", help="comma-separated stem suffixes (diagnostics)")
    args = parser.parse_args()

    import torch
    from b12x.integration.cuteafd import (
        FLASH, GLM53, GLM53_FLASH, MIMO_V2_FLASH, PRO, exportable_compilation, validate_exported_header,
    )

    geometries = [name.strip() for name in args.geometry.split(",") if name.strip()]
    if not geometries or any(name not in ("flash", "pro", "glm", "mimo", "glmf") for name in geometries):
        raise SystemExit("--geometry takes flash, pro, glm, mimo and/or glmf")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 0):
        raise SystemExit("coordinator programs export on SM120")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    selected = set(args.only.split(",")) if args.only else None
    manifest = {
        "schema": 1,
        "families": {},
        "capacities": {"decode_rows": args.decode_rows, "prefill_rows": args.prefill_rows,
                       "max_context": args.max_context},
        "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "capability": [props.major, props.minor],
        "programs": [],
    }
    entries, includes = [], []
    work = []
    for name in geometries:
        g = {"flash": FLASH, "pro": PRO, "glm": GLM53, "mimo": MIMO_V2_FLASH, "glmf": GLM53_FLASH}[name]
        family = {"flash": "dsv4f", "pro": "dsv4p", "glm": "glm", "mimo": "mimo", "glmf": "glmf"}[name]
        manifest["families"][family] = {k: v for k, v in vars(g).items()}
        make = {"glm": glm_programs, "mimo": mimo_programs, "glmf": glmf_programs}.get(name, programs)
        work += [(family, *item) for item in make(g, args.decode_rows, args.prefill_rows, args.max_context)]
    for family, suffix, op, params, thunk in work:
        if selected is not None and suffix not in selected:
            continue
        stem = f"{family}_{suffix}"
        with exportable_compilation():
            program = thunk()
        program.export_to_c(str(output), stem, "cuteafd_" + stem)
        header = output / f"{stem}.h"
        checked = validate_exported_header(program, header, "cuteafd_" + stem)
        abi = program.abi
        kinds = "".join(SCALAR_KINDS[kind] for _, kind in abi["scalars"])
        if checked["argument_count"] != len(abi["pointers"]) + len(kinds) + 1:
            raise ValueError(f"{stem}: header argument count disagrees with the ABI")
        capacity = int(params.get("max_rows", args.prefill_rows))
        manifest["programs"].append({
            "name": stem,
            "family": family,
            "op": op,
            "params": params,
            "pointers": [dict(zip(("name", "dtype", "shape", "role"), p)) for p in abi["pointers"]],
            "scalars": [dict(zip(("name", "type"), s)) for s in abi["scalars"]],
            "scratch_bytes_at_capacity": program.scratch_bytes(capacity),
            "capacity_rows": capacity,
            "entry": checked["symbol"],
            "object_sha256": hashlib.sha256((output / f"{stem}.o").read_bytes()).hexdigest(),
        })
        includes.append(f'#include "{stem}.h"')
        entries.append(
            f'{{"{stem}", _mlir_cuteafd_{stem}_cuda_init, _mlir_cuteafd_{stem}_cuda_load_to_device, '
            f'{checked["symbol"]}, {len(abi["pointers"])}, "{kinds}"}}'
        )
        print(f"exported {stem}: {len(abi['pointers'])} pointers, scalars '{kinds}'", flush=True)
    if not entries:
        raise SystemExit("no programs selected")
    (output / "dsv4_programs.h").write_text("\n".join([
        "#pragma once",
        *includes,
        f"#define CUTEAFD_DSV4_CC_MINOR {props.minor}",
        "#define CUTEAFD_DSV4_PROGRAMS " + ", ".join(entries),
        "",
    ]))
    (output / "dsv4_programs.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
