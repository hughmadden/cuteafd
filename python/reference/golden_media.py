#!/usr/bin/env python3
"""Export official MiMo BF16/FP32 media features for G4/G2 probes, without the LM."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from fidelity_media import require_media_flag
from fidelity_windows import canonical, load_set, verify_snapshot
from mimo_media import window_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--windows", required=True, type=Path)
    p.add_argument("--media-root", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tower-dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()
    manifest = load_set(a.windows, "mimo_v2")
    require_media_flag(manifest, True, "mimo_v2")
    identity = verify_snapshot(manifest, a.snapshot)
    import torch
    torch.cuda.set_device(a.device)
    a.media_features_out = a.out
    started = time.monotonic()
    features, media_identity = window_features(a, manifest)
    result = {"schema": "cuteafd.media.export/1", "set_sha256": manifest["set_sha256"],
        "checkpoint": manifest["checkpoint"], "snapshot_identity": {**identity, **media_identity},
        "tower_dtype": a.tower_dtype, "feature_shapes": {key: list(value.shape) for key, value in features.items()},
        "seconds": time.monotonic() - started, "scope": "tower only; not an LM prefix qualification"}
    (a.out / "export.json").write_bytes(canonical(result) + b"\n")
    print(f"{a.out}: {len(features)} feature arrays ({a.tower_dtype}), {result['seconds']:.3f}s")


if __name__ == "__main__":
    main()
