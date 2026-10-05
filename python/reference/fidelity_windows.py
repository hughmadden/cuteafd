"""Host-only contracts shared by the set builder, goldens and converter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SET_SCHEMA = "cuteafd.fidelity.set/1"
MAX_TOKENS = 16384


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def set_hash(manifest: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != "set_sha256"})).hexdigest()


def validate_set(manifest: dict) -> dict:
    if manifest.get("schema") != SET_SCHEMA:
        raise ValueError("unsupported fidelity set schema")
    if manifest.get("set_sha256") != set_hash(manifest):
        raise ValueError("fidelity set hash mismatch")
    seen = set()
    for w in manifest["windows"]:
        name = w["id"]
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("unsafe window id")
        if name in seen:
            raise ValueError("duplicate window id")
        seen.add(name)
        tokens, roles, start = w["tokens"], w["roles"], w["score_from"]
        if not 1 <= start < len(tokens) <= MAX_TOKENS:
            raise ValueError("window length/score_from exceeds the 16K contract")
        if any(type(t) is not int or not 0 <= t <= 2147483647 for t in tokens):
            raise ValueError("token ids must be nonnegative i32")
        if len(roles) != len(tokens) or any(r not in ("ctx", "gen") for r in roles):
            raise ValueError("one ctx/gen role is required per token")
        if w["bucket"] != bucket(start):
            raise ValueError("context bucket does not match first scored position")
    if not seen or len(set(manifest["quick_windows"])) != len(manifest["quick_windows"]) or not set(manifest["quick_windows"]) <= seen:
        raise ValueError("invalid quick subset")
    return manifest


def load_set(path: Path, family: str | None = None) -> dict:
    manifest = validate_set(json.loads(path.read_text()))
    if family is not None and manifest["family"] != family:
        raise ValueError(f"set family {manifest['family']} does not match {family}")
    return manifest


def bucket(position: int) -> str:
    return "0-2K" if position <= 2048 else "2-8K" if position <= 8192 else "8-16K" if position < MAX_TOKENS else "16K+"


def verify_snapshot(manifest: dict, snapshot: Path) -> dict:
    """Check cheap, stable identities without hashing hundreds of GB of weights."""
    tokenizer_sha = hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest()
    config_sha = hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest()
    if manifest.get("tokenizer_sha256") not in (None, tokenizer_sha):
        raise ValueError("snapshot tokenizer differs from the pinned fidelity set")
    arm = manifest.get("generation_arm", {})
    revision = arm.get("snapshot_revision")
    if revision is not None and snapshot.name != revision:
        raise ValueError("snapshot revision differs from the generation arm")
    if arm.get("config_sha256") not in (None, config_sha):
        raise ValueError("snapshot config differs from the generation arm")
    return {"snapshot_revision": snapshot.name, "tokenizer_sha256": tokenizer_sha,
            "config_sha256": config_sha}


def write_scored_logits(out: Path, window: dict, logits: np.ndarray) -> dict:
    """Rows are already selected: row i predicts token score_from + i."""
    positions = list(range(window["score_from"], len(window["tokens"])))
    if logits.ndim != 2 or logits.shape[0] != len(positions):
        raise ValueError("scored logits shape does not match window positions")
    folder = out / "windows" / window["id"]
    folder.mkdir(parents=True, exist_ok=True)
    np.asarray(window["tokens"], dtype="<i4").tofile(folder / "tokens.bin")
    np.asarray(logits, dtype="<f4").tofile(folder / "logits.bin")
    return {"id": window["id"], "path": str(folder.relative_to(out)),
            "positions": positions, "vocab": logits.shape[1]}


def finish_golden(out: Path, manifest: dict, rows: list[dict], **meta) -> None:
    (out / "windows.json").write_bytes(canonical(manifest) + b"\n")
    (out / "meta.json").write_bytes(canonical({"schema": "cuteafd.fidelity.golden/2",
        "set_sha256": manifest["set_sha256"], "family": manifest["family"],
        "checkpoint": manifest["checkpoint"], "windows": rows, **meta}) + b"\n")
