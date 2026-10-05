"""Host-only media identity, fixture and probe-feature contracts."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HEX = re.compile(r"[0-9a-f]{64}\Z")


def validate_media(media, tokens, roles, score_from):
    if not isinstance(media, list):
        raise ValueError("media must be an ordered list")
    previous = 0
    for span in media:
        if not isinstance(span, dict) or span.get("kind") != "image":
            raise ValueError("unsupported fidelity media kind")
        start, length = span.get("start"), span.get("len")
        if (type(start) is not int or type(length) is not int or length < 1
                or start < previous or start + length > score_from):
            raise ValueError("media spans must be sorted, disjoint and before scoring")
        grid = span.get("grid")
        if (not isinstance(grid, list) or len(grid) != 3
                or any(type(n) is not int or n <= 0 for n in grid)
                or grid[0] != 1 or grid[1] % 2 or grid[2] % 2
                or grid[0] * grid[1] * grid[2] // 4 != length):
            raise ValueError("image grid does not match merged token extent")
        if not isinstance(span.get("key"), str) or not HEX.fullmatch(span["key"]):
            raise ValueError("media key must be lowercase SHA256 hex")
        fixture = span.get("fixture")
        if not isinstance(fixture, dict):
            raise ValueError("fixture requires a relative path and source hash")
        path = fixture.get("path")
        if (not isinstance(path, str) or not path or "\\" in path
                or Path(path).is_absolute() or any(p in (".", "..") for p in path.split("/"))
                or any(not p for p in path.split("/"))):
            raise ValueError("fixture path escapes fixture root")
        if not isinstance(fixture.get("sha256"), str) or not HEX.fullmatch(fixture["sha256"]):
            raise ValueError("fixture lacks a SHA256")
        if any(r != "ctx" for r in roles[start:start + length]):
            raise ValueError("image placeholders cannot be generated positions")
        if len(set(tokens[start:start + length])) != 1:
            raise ValueError("image span must contain one repeated placeholder id")
        previous = start + length
    return media


def read_fixture(root: Path, span: dict) -> bytes:
    path = (root / span["fixture"]["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("fixture path escapes fixture root")
    cap = 32 * 1024 * 1024
    with path.open("rb") as source:
        if path.stat().st_size > cap:
            raise ValueError("fixture exceeds image byte cap")
        data = source.read(cap + 1)
    if len(data) > cap:
        raise ValueError("fixture exceeds image byte cap")
    if hashlib.sha256(data).hexdigest() != span["fixture"]["sha256"]:
        raise ValueError("fixture bytes differ from pinned media identity")
    return data


def require_media_flag(manifest, enabled, family):
    present = any(w.get("media") for w in manifest["windows"])
    if present and not enabled:
        raise ValueError("media windows require explicit --media; refusing text-only scoring")
    if enabled and not present:
        raise ValueError("--media requires pinned media windows")
    if present and family != "mimo_v2":
        raise ValueError(f"{family} official media golden is not implemented")
    return present


def write_features(root: Path, span: dict, values, *, tower_dtype: str, identity: dict):
    """Write immutable row-major BF16 probe rows; values are CPU uint16 bits."""
    import numpy as np
    bits = np.asarray(values)
    if bits.dtype != np.uint16 or bits.ndim != 2 or bits.shape[0] != span["len"]:
        raise ValueError("feature rows must be uint16 BF16 [media.len, hidden]")
    if not np.isfinite((bits.astype(np.uint32) << 16).view(np.float32)).all():
        raise ValueError("non-finite reference features")
    if tower_dtype not in ("bf16", "fp32"):
        raise ValueError("unsupported tower precision")
    if not HEX.fullmatch(span["key"]):
        raise ValueError("unsafe feature key")
    data = bits.astype("<u2").tobytes()
    meta = {"schema": "cuteafd.media.features/1", "key": span["key"],
            "grid": span["grid"], "shape": list(bits.shape), "dtype": "bf16-le",
            "sha256": hashlib.sha256(data).hexdigest(), "tower_dtype": tower_dtype,
            "fixture_sha256": span["fixture"]["sha256"], "snapshot_identity": identity}
    encoded = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    root.mkdir(parents=True, exist_ok=True)
    for suffix, content in ((".bf16", data), (".json", encoded)):
        path = root / (span["key"] + suffix)
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError("feature identity reused with different bytes or metadata")
        else:
            with path.open("xb") as output:
                output.write(content)
    return meta
