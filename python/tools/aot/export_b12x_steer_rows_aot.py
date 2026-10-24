#!/usr/bin/env python3
"""Export the weight-variant row programs, offline.

From ``b12x.integration.cuteafd.steer_rows``: ``steer_rows_one`` (k_max == 1)
and ``steer_rows_general`` (k_max <= 32) for ``y -= A (B^T y)``, and
``add_rows_f32`` for ``y += d`` (stored-delta sites). Widths, strides, row
counts and k_max are launch scalars and the grid is one CTA per row, so the
export holds no model geometry and no SM count.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys

STEER_POINTERS = ("x", "vidx", "a", "b", "k")
STEER_SCALARS = (("ld", "int64_t"), ("rows", "int32_t"), ("h", "int32_t"), ("k_max", "int32_t"))
ADD_POINTERS = ("y", "d", "vidx", "on")
ADD_SCALARS = (("ld_y", "int64_t"), ("ld_d", "int64_t"), ("rows", "int32_t"), ("h", "int32_t"))
# (stem, pointers, scalars); the BF16-delta add_rows form is compiled on request only.
PROGRAMS = (("steer_rows_one", STEER_POINTERS, STEER_SCALARS),
            ("steer_rows_general", STEER_POINTERS, STEER_SCALARS),
            ("add_rows_f32", ADD_POINTERS, ADD_SCALARS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capability", type=int, choices=(120, 121), required=True)
    args = parser.parse_args()
    # No driver/runtime access or initialized device context is permitted here.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
    os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
    import _pinned_sparkinfer
    import multiprocessing
    from b12x._lib.compile_pool import _initialize_worker
    capability = (args.capability // 10, args.capability % 10)
    _initialize_worker(0, capability, "steer-rows-offline", "steer-rows-offline", 1,
                       101376, 101376, multiprocessing.Array("i", [0, 0]))
    from cutlass.cutlass_dsl import CuTeDSL
    CuTeDSL._get_dsl().envar.arch = f"sm_{args.capability}a"
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from cuda.bindings import driver as cuda
    from b12x._lib.compiler import compile as b12x_compile
    from b12x.integration.cuteafd import steer_rows

    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    for stem, pointer_names, scalar_types in PROGRAMS:
        symbol = "cuteafd_" + stem
        manifest_path = destination / (stem + ".json")
        manifest_path.unlink(missing_ok=True)
        kind, form = stem.rsplit("_", 1)
        if kind == "steer_rows":
            operands, launch = steer_rows.OPERANDS, steer_rows.steer_rows_launch(form)
            scalars = (cutlass.Int64(1), cutlass.Int32(1), cutlass.Int32(128), cutlass.Int32(1))
            geometry = dict(form=form, threads=steer_rows.THREADS, max_hidden=steer_rows.MAX_HIDDEN,
                            max_k=1 if form == "one" else steer_rows.MAX_K, hidden_multiple=128)
        else:
            operands, launch = steer_rows.add_rows_operands(form), steer_rows.add_rows_launch(form)
            scalars = (cutlass.Int64(8), cutlass.Int64(8), cutlass.Int32(1), cutlass.Int32(8))
            geometry = dict(form=form, threads=steer_rows.ADD_THREADS, hidden_multiple=8)
        pointers = [make_ptr(o.cute_dtype, 16, cute.AddressSpace.gmem, assumed_align=o.align)
                    for o in operands]
        program = b12x_compile(launch, *pointers, *scalars, cuda.CUstream(0))
        program.export_to_c(str(destination), stem, symbol)
        header = (destination / (stem + ".h")).read_text()
        signature = re.search(r"static inline int32_t cute_dsl_" + symbol + r"_wrapper\(([^)]*)\)", header)
        if signature is None:
            raise ValueError(f"missing generated {stem} wrapper")
        expected = [symbol + "_Kernel_Module_t *module"]
        expected += ["void *" + name for name in pointer_names]
        expected += [f"{ctype} {name}" for name, ctype in scalar_types]
        expected += ["cudaStream_t stream"]
        if re.sub(r"\s+", "", signature[1]) != re.sub(r"\s+", "", ",".join(expected)):
            raise ValueError(f"unexpected generated {stem} ABI: " + signature[1])
        artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.glob(stem + ".*")) if p.is_file() and p != manifest_path}
        manifest = dict(schema=1, sparkinfer_revision=_pinned_sparkinfer.REVISION,
            source_tree_sha256=_pinned_sparkinfer.LOCK_DATA["source_tree_sha256"],
            capability=list(capability), geometry=geometry,
            abi=dict(pointers=list(pointer_names), scalars=[n for n, _ in scalar_types], stream="stream"),
            artifacts=artifacts)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(manifest_path, flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("offline steer_rows export initialized CUDA")


if __name__ == "__main__":
    main()
