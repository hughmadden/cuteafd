"""Media fidelity contracts: CPU-only, no checkpoint or GPU needed."""
import copy
import hashlib
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_media import read_fixture, require_media_flag, validate_media, write_features


def span():
    return {"start": 1, "len": 4, "kind": "image", "key": "a" * 64,
            "grid": [1, 4, 4], "fixture": {"path": "code.png", "sha256": "b" * 64}}


def test_media_contract():
    s = span()
    validate_media([s], [1, 9, 9, 9, 9, 2], ["ctx"] * 5 + ["gen"], 5)
    for field, value in [("len", 0), ("start", -1), ("grid", [1, 3, 3]),
                         ("key", "A" * 64), ("fixture", {"path": "../x", "sha256": "b" * 64})]:
        bad = copy.deepcopy(s)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_media([bad], [1, 9, 9, 9, 9, 2], ["ctx"] * 5 + ["gen"], 5)
    with pytest.raises(ValueError):
        validate_media([s, s], [1, 9, 9, 9, 9, 2], ["ctx"] * 5 + ["gen"], 5)
    with pytest.raises(ValueError):
        validate_media([s], [1, 9, 8, 9, 9, 2], ["ctx"] * 5 + ["gen"], 5)


def test_feature_writer_binds_immutable_bytes(tmp_path):
    s = span()
    values = np.full((4, 8), 0x3f80, dtype=np.uint16)
    metadata = write_features(tmp_path, s, values, tower_dtype="bf16", identity={"snapshot_revision": "pinned"})
    assert metadata["shape"] == [4, 8]
    assert metadata["sha256"] == hashlib.sha256(values.astype("<u2").tobytes()).hexdigest()
    assert write_features(tmp_path, s, values, tower_dtype="bf16", identity={"snapshot_revision": "pinned"}) == metadata
    with pytest.raises(ValueError, match="identity reused"):
        write_features(tmp_path, s, values + 1, tower_dtype="bf16", identity={"snapshot_revision": "pinned"})
    with pytest.raises(ValueError, match="non-finite"):
        write_features(tmp_path, s, np.full((4, 8), 0x7f80, dtype=np.uint16), tower_dtype="bf16", identity={})


def test_fixture_hash_and_family_fail_closed(tmp_path):
    s = span()
    (tmp_path / "code.png").write_bytes(b"fixture")
    s["fixture"]["sha256"] = hashlib.sha256(b"fixture").hexdigest()
    assert read_fixture(tmp_path, s) == b"fixture"
    (tmp_path / "code.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="pinned"):
        read_fixture(tmp_path, s)
    manifest = {"windows": [{"media": [s]}]}
    assert require_media_flag(manifest, True, "mimo_v2")
    for family, flag in [("mimo_v2", False), ("qwen4", True), ("glm5_flash", True)]:
        with pytest.raises(ValueError):
            require_media_flag(manifest, flag, family)
