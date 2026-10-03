#!/usr/bin/env python3
"""Check a Spark image's BF16 expert-input capability without allocating CUDA."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def verify_sibling(root: Path, family: str, layout: str, capacity: int,
                   revision: str, verify) -> None:
    if not revision or revision == "unknown":
        raise ValueError("image does not identify its SparkInfer source pin")
    primary = verify(root / f"fp8-{family}")
    sibling = verify(root / f"fp8-{family}-bf16")
    infos = []
    for manifest, name, input_kind in ((primary, "primary", "wire"), (sibling, "BF16 sibling", "bf16")):
        if (manifest.get("role"), manifest.get("geometry"), manifest.get("compute"),
                manifest.get("sparkinfer_revision")) != ("spark", family, [12, 1], revision):
            raise ValueError(f"{name} does not match Spark SM121 geometry {family} and source pin {revision}")
        info = manifest.get("layouts", {}).get(layout)
        if info is None or info.get("input") != input_kind:
            raise ValueError(f"{name} has no {input_kind} {layout} layout")
        if capacity < 1 or not any(c["capacity"] >= capacity for c in info.get("capacities", [])):
            raise ValueError(f"{name} has no capacity for {capacity} rows")
        infos.append(info)
    # Match the Rust preallocation admission contract before starting workers.
    for key in ("weights", "hidden", "experts", "top_k", "intermediate", "tp", "slice", "swiglu_limit"):
        if key not in infos[0] or key not in infos[1] or infos[0][key] != infos[1][key]:
            raise ValueError(f"BF16 sibling differs from primary in {key}")


def main() -> int:
    try:
        family, layout, rows = sys.argv[1:]
        tool = Path("/opt/cuteafd/python/tools/aot/package_fp8_moe_aot.py")
        spec = importlib.util.spec_from_file_location("fp8_package", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        verify_sibling(Path("/opt/cuteafd/lib/fp8"), family, layout, int(rows),
                       os.environ.get("CUTEAFD_SPARKINFER_COMMIT", ""), module.verify)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f"BF16 expert-input package unavailable: {error}", file=sys.stderr)
        return 2
    print(f"BF16 expert-input package ready: {family}/{layout}, {rows} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
