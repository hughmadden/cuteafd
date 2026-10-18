#!/usr/bin/env python3
"""Export EXLR8 standalone/serving K1 and rotated-domain glue programs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import _pinned_sparkinfer


def programs(hidden=6144, units=16, top_k=8, max_groups=512, role="all"):
    from b12x.integration.cuteafd.exlr8_glue import exlr8_export_programs
    return exlr8_export_programs(role, hidden=hidden, units=units,
                                  top_k=top_k, max_groups=max_groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from export_b12x_dsv4_aot import main as export
    sys.argv = [sys.argv[0], "--geometry", "exlr8",
                "--output-dir", str(args.output_dir)]
    export()


if __name__ == "__main__":
    main()
