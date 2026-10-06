"""Host-only media identity, fixture and probe-feature contracts."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import secrets

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


def require_media_flag(manifest, enabled, family, *, implemented_families=()):
    """Family runners opt in only after implementing their official tower hook."""
    present = any(w.get("media") for w in manifest["windows"])
    if present and not enabled:
        raise ValueError("media windows require explicit --media; refusing text-only scoring")
    if enabled and not present:
        raise ValueError("--media requires pinned media windows")
    if present and family not in {"mimo_v2", "qwen4", *implemented_families}:
        raise ValueError(f"{family} official media golden is not implemented")
    return present


@contextmanager
def _feature_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    # Directory-authorized writers need only read access to the persistent lock.
    fd = os.open(root / ".features.lock", os.O_RDONLY | os.O_CREAT, 0o666)
    with os.fdopen(fd, "rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _publish_locked(contents):
    """Validate every existing member before publishing any new member."""
    missing = []
    for path, data in contents:
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("feature identity reused with different bytes or metadata; feature index is immutable")
        else:
            missing.append((path, data))
    for path, data in missing:
        temporary = None
        try:
            for _ in range(100):
                candidate = path.parent / (".feature-" + secrets.token_hex(16))
                try:
                    fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                except FileExistsError:
                    continue
                temporary = candidate
                break
            else:
                raise FileExistsError("cannot allocate an exclusive feature staging file")
            # Ordinary creation permissions preserve the caller's umask, unlike mkstemp.
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            # Readers see complete bytes; the held lock prevents writer replacement.
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if missing:
        directory = os.open(missing[0][0].parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def publish_immutable(path: Path, content: bytes):
    """Publish complete immutable bytes under the shared feature-set writer lock."""
    with _feature_lock(path.parent):
        _publish_locked([(path, content)])


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
    with _feature_lock(root):
        _publish_locked([(root / (span["key"] + ".bf16"), data),
                         (root / (span["key"] + ".json"), encoded)])
    return meta
