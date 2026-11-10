"""Recovery barriers and bounded retirement without a CUDA context."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3.core_k6.storage import StageStore
from kimi_k3.uniform_k2.storage import file_identity, sha256_file


def payload(store, stage="a", name="batch.bin"):
    directory = store.directory(stage)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"activation")
    return dict(filename=name, sha256=sha256_file(path), file_identity=file_identity(path))


def test_partial_phase_is_never_trusted(tmp_path):
    store = StageStore(tmp_path, {"run": 1})
    receipt = payload(store)
    assert store.load("a") is None
    result = store.commit("a", [receipt], {"rows": 12})
    assert store.load("a", checksum=True) == result
    assert store.commit("a", [receipt], {"rows": 12}) == result
    with pytest.raises(ValueError, match="replace"):
        store.commit("a", [receipt], {"rows": 13})


def test_resume_rejects_changed_identity_or_payload(tmp_path):
    store = StageStore(tmp_path, {"run": 1})
    receipt = payload(store)
    store.commit("a", [receipt])
    with pytest.raises(ValueError, match="identity"):
        StageStore(tmp_path, {"run": 2}).load("a")
    (store.directory("a") / "batch.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="changed"):
        store.load("a")


def test_retire_preserves_successor(tmp_path):
    store = StageStore(tmp_path, {})
    store.commit("a", [payload(store)])
    store.commit("d", [payload(store, "d")])
    store.retire("a")
    assert store.load("a") is None
    assert store.load("d", checksum=True)
    store.clear()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["../bad", "/bad", "a/b", "", ".", ".."])
def test_reject_unsafe_stage_names(tmp_path, name):
    with pytest.raises(ValueError):
        StageStore(tmp_path, {}).directory(name)


def test_reject_payload_paths_and_symlinks(tmp_path):
    store = StageStore(tmp_path, {})
    receipt = payload(store)
    with pytest.raises(ValueError):
        store.commit("a", [dict(receipt, filename="../batch.bin")])
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "b").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.retire("b")
