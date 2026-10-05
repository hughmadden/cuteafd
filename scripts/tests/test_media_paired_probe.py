"""Paired encoder gates decide against golden rows, never direct arm KL."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("media_paired_probe", ROOT / "scripts/bench/media-paired-probe.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_window_bootstrap_is_paired_and_deterministic():
    stats, counts = [[.02, .01], [.04, .02]], [10, 20]
    result = module.paired_bounds(stats, counts, 500, 7)
    assert result == module.paired_bounds(stats, counts, 500, 7)
    assert result["kl_increase"] == pytest.approx(.002)
    assert result["top1_loss"] == pytest.approx(.001)
    assert result["kl_increase_upper95"] == pytest.approx(.002)
    assert result["top1_loss_upper95"] == pytest.approx(.001)
    with pytest.raises(ValueError):
        module.paired_bounds([[0, 0]], [1])


def test_dump_positions_and_safe_files(tmp_path):
    window = {"score_from": 2, "tokens": [1, 2, 3, 4]}
    rows = [{"position": pos, "vocab_size": 3, "file": f"row-{pos}.safetensors",
             "tensor": "log_probs", "dtype": "F32", "byte_order": "little"} for pos in [2, 3]]
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert list(module.dump_rows(tmp_path, window, 3)) == [2, 3]
    rows[1]["position"] = 2
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="duplicate"):
        module.dump_rows(tmp_path, window, 3)
    rows[1]["position"], rows[0]["file"] = 3, "../escape"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        module.dump_rows(tmp_path, window, 3)


def test_record_requires_native_identity_and_override_metadata(tmp_path):
    span = {"start": 0, "len": 1, "kind": "image", "key": "a" * 64, "grid": [1, 2, 2]}
    window = {"tokens": [9, 2], "score_from": 1, "media": [span]}
    record = {"engine": "mimo", "cold": True, "no_speculation": True, "cached_tokens": 0,
              "score_path": "decode", "prompt_ids": [9, 2], "media": [span],
              "rows": [{"position": 1, "finite": True}], "scored": 1}
    response = {"probe": record, "server": {"model": "model"}}
    module.check_record(window, response, "model", "native")
    record["media"] = []
    with pytest.raises(ValueError, match="echo"):
        module.check_record(window, response, "model", "native")
    record["media"] = [span]
    metadata = {"key": span["key"], "sha256": "b" * 64}
    (tmp_path / (span["key"] + ".json")).write_text(json.dumps(metadata))
    record["provenance"] = {"mode": "reference_features", "probe_only": True,
                            "encoder_bypassed": True, "features": [metadata]}
    module.check_record(window, response, "model", "reference", tmp_path)
    with pytest.raises(ValueError, match="native arm"):
        module.check_record(window, response, "model", "native")
    record["provenance"]["features"][0] = {"sha256": "c" * 64}
    with pytest.raises(ValueError, match="provenance"):
        module.check_record(window, response, "model", "reference", tmp_path)


def test_log_probs_validate_shape_and_normalize(tmp_path):
    from safetensors.numpy import save_file
    path = tmp_path / "row.safetensors"
    save_file({"log_probs": np.array([-1., -2., -3.], dtype=np.float32)}, str(path))
    values = module.log_probs(path, 3)
    assert np.exp(values).sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="shape"):
        module.log_probs(path, 4)
    with pytest.raises(ValueError, match="nonfinite"):
        module.normalize([0, float("nan")])


def test_compare_uses_golden_difference_not_direct_kl(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from safetensors.numpy import save_file
    from fidelity_windows import canonical, set_hash
    panel = {"schema": "cuteafd.fidelity.set/1", "family": "mimo_v2", "checkpoint": "model",
             "quick_windows": ["w0", "w1"], "windows": []}
    span = {"start": 0, "len": 1, "kind": "image", "key": "a" * 64, "grid": [1, 2, 2],
            "fixture": {"path": "image.png", "sha256": "b" * 64}}
    for name in ["w0", "w1"]:
        panel["windows"].append({"id": name, "block": "vision", "bucket": "0-2K",
                                "tokens": [9] * 577, "roles": ["ctx"] * 65 + ["gen"] * 512,
                                "score_from": 65, "media": [span]})
    panel["set_sha256"] = set_hash(panel)
    windows = tmp_path / "windows.json"
    windows.write_bytes(canonical(panel))
    identity = {"snapshot_revision": "pinned"}
    proof = {"schema": "cuteafd.fidelity.prefix/1", "passed": True, "finite": True,
             "rows": 512, "different_rows": [], "argmax_disagreements": 0,
             "lengths": [576, 640], "score_from": 64, "fixed_rows": 128,
             "family": "mimo_v2", "set_sha256": panel["set_sha256"], "snapshot_identity": identity,
             "source_window": "w0", "media": [span], "vocab": 2}
    golden = tmp_path / "golden"
    golden.mkdir()
    meta = {"set_sha256": panel["set_sha256"], "checkpoint": "model", "snapshot_identity": identity,
            "prefix_qualification": proof, "windows": []}
    # Golden is uniform. Symmetric arms are equally distant from it, but very far
    # from each other: the deciding paired KL increase is zero, direct KL is large.
    for window in panel["windows"]:
        folder = golden / window["id"]
        folder.mkdir()
        np.zeros((512, 2), dtype="<f4").tofile(folder / "logits.bin")
        np.asarray(window["tokens"], dtype="<i4").tofile(folder / "tokens.bin")
        meta["windows"].append({"id": window["id"], "path": window["id"], "positions": list(range(65, 577)),
                                "vocab": 2, "media": [span]})
    (golden / "meta.json").write_bytes(canonical(meta))
    features = tmp_path / "features"
    features.mkdir()
    metadata = {"key": span["key"], "sha256": "c" * 64}
    (features / (span["key"] + ".json")).write_text(json.dumps(metadata))
    for mode, row in [("native", [0., -5.]), ("reference", [-5., 0.])]:
        root = tmp_path / mode
        root.mkdir()
        capture = {"schema": "cuteafd.media.paired.capture/1", "mode": mode, "set_sha256": panel["set_sha256"],
                   "checkpoint": "model", "quick": False,
                   "server": {"model": "model", "build": {"commit": "same", "image": mode}}, "windows": []}
        for window in panel["windows"]:
            dump = root / window["id"]
            dump.mkdir()
            manifest, files = [], []
            for pos in range(65, 577):
                path = dump / f"row-{pos}.safetensors"
                save_file({"log_probs": np.array(row, dtype=np.float32)}, str(path))
                manifest.append({"position": pos, "vocab_size": 2, "file": path.name,
                                 "tensor": "log_probs", "dtype": "F32", "byte_order": "little"})
                files.append({"position": pos, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            (dump / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in manifest))
            response = {"server": capture["server"], "probe": {"engine": "mimo", "cold": True,
                "no_speculation": True, "cached_tokens": 0, "score_path": "decode",
                "prompt_ids": window["tokens"], "media": [span], "scored": 512,
                "rows": [{"position": pos, "finite": True} for pos in range(65, 577)]}}
            if mode == "reference":
                response["probe"]["provenance"] = {"mode": "reference_features", "probe_only": True,
                    "encoder_bypassed": True, "features": [metadata]}
            response_path = root / (window["id"] + ".json")
            response_path.write_bytes(canonical(response))
            capture["windows"].append({"id": window["id"], "path": str(dump), "files": files,
                "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256((dump / "manifest.jsonl").read_bytes()).hexdigest()})
        (root / "capture.json").write_bytes(canonical(capture))
    out = tmp_path / "g4.json"
    args = SimpleNamespace(windows=windows, golden=golden, native=tmp_path / "native",
                           reference=tmp_path / "reference", features=features, out=out, bootstrap=100, seed=7)
    assert module.compare(args)
    result = json.loads(out.read_text())
    assert result["kl_increase_upper95"] == pytest.approx(0)
    assert result["top1_loss_upper95"] == pytest.approx(-1)
    assert result["windows"][0]["direct_reference_native_kl"] > 4
    assert result["criterion"] == module.CRITERION
    capture_path = tmp_path / "reference/capture.json"
    sealed = json.loads(capture_path.read_text())
    sealed["windows"].pop()
    capture_path.write_bytes(canonical(sealed))
    args.out = tmp_path / "incomplete.json"
    with pytest.raises(ValueError, match="incomplete windows"):
        module.compare(args)
