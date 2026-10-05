"""Host-only contracts shared by the set builder, goldens and converter."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
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


def prefix_comparison(short: np.ndarray, extended: np.ndarray) -> dict:
    """Require finite, bit-identical f32 common-prefix rows, not just argmax."""
    if short.ndim != 2 or extended.ndim != 2 or short.shape[1] != extended.shape[1] or extended.shape[0] < short.shape[0]:
        raise ValueError("prefix qualification logits have incompatible shapes")
    left = np.asarray(short, dtype="<f4")
    right = np.asarray(extended[:len(short)], dtype="<f4")
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    changed = np.any(left.view("<u4") != right.view("<u4"), axis=1)
    return {"passed": finite and not bool(changed.any()), "finite": finite,
            "rows": len(left), "vocab": left.shape[1],
            "different_rows": np.flatnonzero(changed).tolist(),
            "argmax_disagreements": int(np.count_nonzero(left.argmax(1) != right.argmax(1)))}


def qualify_prefix(a, manifest: dict, execute) -> dict:
    """Fail closed before the full panel; execute the family's actual golden loop."""
    source = next((w for w in manifest["windows"] if len(w["tokens"]) >= 640), None)
    if source is None:
        raise ValueError("prefix qualification requires a pinned window of at least 640 tokens")
    root = Path(tempfile.mkdtemp(prefix="prefix-gate-", dir=a.out))
    panel = copy.deepcopy(manifest)
    panel["windows"] = []
    for size, name in ((576, "prefix_short"), (640, "prefix_extended")):
        window = copy.deepcopy(source)
        window.update(id=name, tokens=source["tokens"][:size], roles=["ctx"] * size,
                      score_from=64, bucket=bucket(64))
        panel["windows"].append(window)
    panel["quick_windows"] = ["prefix_short", "prefix_extended"]
    panel["set_sha256"] = set_hash(panel)
    validate_set(panel)
    windows = root / "input.json"
    windows.write_bytes(canonical(panel) + b"\n")
    probe = copy.copy(a)
    probe.windows, probe.out, probe.layers, probe._prefix_probe = windows, root, None, True
    execute(probe)
    meta = json.loads((root / "meta.json").read_text())
    if meta.get("set_sha256") != panel["set_sha256"] or meta.get("family") != manifest["family"]:
        raise ValueError("prefix golden provenance mismatch")
    entries = {w["id"]: w for w in meta["windows"]}
    arrays, hashes = [], []
    for window in panel["windows"]:
        entry = entries[window["id"]]
        positions = list(range(window["score_from"], len(window["tokens"])))
        if entry["positions"] != positions:
            raise ValueError("prefix golden positions mismatch")
        folder = (root / entry["path"]).resolve()
        if not folder.is_relative_to(root.resolve()):
            raise ValueError("prefix golden path escapes output")
        if not np.array_equal(np.fromfile(folder / "tokens.bin", dtype="<i4"), window["tokens"]):
            raise ValueError("prefix golden tokens mismatch")
        path = folder / "logits.bin"
        if path.stat().st_size != len(positions) * entry["vocab"] * 4:
            raise ValueError("prefix golden logits extent mismatch")
        arrays.append(np.memmap(path, dtype="<f4", mode="r", shape=(len(positions), entry["vocab"])))
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    result = prefix_comparison(*arrays)
    proof = {"schema": "cuteafd.fidelity.prefix/1", "family": manifest["family"],
             "set_sha256": manifest["set_sha256"], "source_window": source["id"],
             "lengths": [576, 640], "score_from": 64, "fixed_rows": 128,
             "snapshot_identity": meta["snapshot_identity"], "logits_sha256": hashes,
             "seconds": meta["seconds"], **result}
    (root / "qualification.json").write_bytes(canonical(proof) + b"\n")
    print(f"prefix qualification: {len(result['different_rows'])}/{result['rows']} different rows; {root}", flush=True)
    if not result["passed"]:
        raise ValueError(f"reference prefix invariance failed; see {root / 'qualification.json'}")
    return proof


def validate_qualification(proof: dict | None, manifest: dict, identity: dict) -> None:
    if not isinstance(identity, dict) or not identity.get("snapshot_revision"):
        raise ValueError("reference prefix qualification lacks snapshot identity")
    if not isinstance(proof, dict) or proof.get("schema") != "cuteafd.fidelity.prefix/1":
        raise ValueError("reference lacks prefix-invariance qualification")
    required = {"passed": True, "finite": True, "rows": 512, "different_rows": [],
                "argmax_disagreements": 0, "lengths": [576, 640], "score_from": 64,
                "fixed_rows": 128, "family": manifest["family"],
                "set_sha256": manifest["set_sha256"], "snapshot_identity": identity}
    if any(proof.get(k) != v for k, v in required.items()):
        raise ValueError("reference prefix-invariance qualification failed or mismatched provenance")


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
