#!/usr/bin/env python3
"""Shared native dense operands used by EXLR8 checkpoints (opt-in table)."""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer

# (name, K, N, sharded side): the GLM 5.3 checkpoint's unique trellis shapes.
TRELLIS_SHAPES = (
    ("q_b", 2048, 16384, 1),
    ("o", 16384, 6144, 2),
    ("gate_up", 6144, 12288, 1),
    ("down", 12288, 6144, 2),
)


def programs():
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    result = []
    for name, k, n, side in TRELLIS_SHAPES:
        params = dict(input_features=k, output_features=n, bits=4,
                      codebook="e4m3", sharded_side=side, rotation_count=4,
                      max_rows=4096)
        result.append((f"trellis_{name}", "trellis_linear", params,
                       partial(compile_trellis_linear_aot, **params)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from export_b12x_dsv4_aot import main as export
    sys.argv = [sys.argv[0], "--geometry", "exlr8_dense",
                "--output-dir", str(args.output_dir)]
    export()


if __name__ == "__main__":
    main()
