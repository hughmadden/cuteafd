#!/usr/bin/env python3
"""Export official BF16/FP32 media features for G4/G2 probes, without the LM."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from fidelity_media import require_media_flag
from fidelity_windows import canonical, load_set, verify_snapshot


def family_features(family):
    """Only implemented official tower adapters can export reference features."""
    if family == "mimo_v2":
        from mimo_media import window_features
    elif family == "glm5_flash":
        from glm_flash_media import window_features
    else:
        raise ValueError(f"{family} official media exporter is not implemented")
    return window_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("mimo_v2", "glm5_flash"), default="mimo_v2")
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--windows", required=True, type=Path)
    p.add_argument("--media-root", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tower-dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()
    manifest = load_set(a.windows, a.family)
    require_media_flag(manifest, True, a.family, implemented_families=("glm5_flash",))
    identity = verify_snapshot(manifest, a.snapshot)
    window_features = family_features(a.family)
    import torch
    torch.cuda.set_device(a.device)
    a.media_features_out = a.out
    started = time.monotonic()
    features, media_identity = window_features(a, manifest)
    result = {"schema": "cuteafd.media.export/1", "set_sha256": manifest["set_sha256"],
        **({"family": a.family} if a.family != "mimo_v2" else {}),
        "checkpoint": manifest["checkpoint"], "snapshot_identity": {**identity, **media_identity},
        "tower_dtype": a.tower_dtype, "feature_shapes": {key: list(value.shape) for key, value in features.items()},
        "seconds": time.monotonic() - started, "scope": "tower only; not an LM prefix qualification"}
    (a.out / "export.json").write_bytes(canonical(result) + b"\n")
    print(f"{a.out}: {len(features)} feature arrays ({a.tower_dtype}), {result['seconds']:.3f}s")


if __name__ == "__main__":
    main()
