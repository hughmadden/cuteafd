"""Schema 2 row selection, compact serialization, and legacy equality gates."""
import ast
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_windows import (canonical, finish_golden, load_set, set_hash, validate_set,
                              write_scored_logits, verify_snapshot, prefix_comparison,
                              qualify_prefix, validate_qualification)

spec = importlib.util.spec_from_file_location("make_fidelity_reference", ROOT / "scripts/bench/make-fidelity-reference.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def tiny_set():
    windows = []
    for name in ("legacy", "a00"):
        windows.append({"id": name, "block": "E" if name == "legacy" else "A", "bucket": "0-2K",
                        "tokens": [1, 2, 3, 4, 5], "roles": ["ctx", "ctx", "gen", "gen", "ctx"], "score_from": 2})
    manifest = {"schema": "cuteafd.fidelity.set/1", "family": "deepseek_v41", "model": "test-model",
                "checkpoint": "test-model", "generation_arm": {"head": "bf16"},
                "quick_windows": ["legacy"], "windows": windows}
    manifest["set_sha256"] = set_hash(manifest)
    return manifest


def fixture_proof(manifest, identity=None, vocab=16):
    # Synthetic qualification metadata for converter tests, not hardware evidence.
    return {"schema": "cuteafd.fidelity.prefix/1", "passed": True, "finite": True,
            "rows": 512, "vocab": vocab, "different_rows": [], "argmax_disagreements": 0,
            "lengths": [576, 640], "score_from": 64, "fixed_rows": 128,
            "family": manifest["family"], "set_sha256": manifest["set_sha256"],
            "snapshot_identity": identity or {"snapshot_revision": "fixture"}}


def fixture_golden(tmp_path):
    manifest = tiny_set()
    golden = tmp_path / "golden"
    golden.mkdir()
    logits = np.arange(48, dtype=np.float32).reshape(3, 16) / 16
    logits[0, 0] = logits[0, 15]  # Deterministic top-k boundary tie handling.
    rows = [write_scored_logits(golden, w, logits) for w in manifest["windows"]]
    finish_golden(golden, manifest, rows, snapshot="fixture", reference="fake",
                  snapshot_identity={"snapshot_revision": "fixture"}, prefix_qualification=fixture_proof(manifest))
    return manifest, golden, logits


def options(tmp_path, golden):
    return SimpleNamespace(windows=golden / "windows.json", golden=golden,
        rows_dir=tmp_path / "rows", out=tmp_path / "reference.json", quick_out=tmp_path / "quick.json",
        top_k=12, name=None, model=["test-model"], kl_max=None, top1_min=None)


def test_schema2_scored_rows_and_full_manifest(tmp_path):
    manifest, golden, logits = fixture_golden(tmp_path)
    args = options(tmp_path, golden)
    converter.convert_windows(args)
    reference = json.loads(args.out.read_text())
    assert reference["schema"] == "cuteafd.fidelity.reference/2"
    assert reference["set_sha256"] == manifest["set_sha256"]
    assert reference["checkpoint"] == "test-model"
    assert reference["expect"] == {"kl_max": 0.06, "top1_min": 0.90}
    assert [w["id"] for w in json.loads(args.quick_out.read_text())["windows"]] == ["legacy"]
    rows = json.loads((args.rows_dir / "rows.json").read_text())
    assert rows["dtype"] == "<f2" and rows["kind"] == "log_softmax"
    for entry, window in zip(rows["windows"], reference["windows"]):
        path = args.rows_dir / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert entry["shape"] == [3, 16] and entry["positions"] == [2, 3, 4]
        full = np.fromfile(path, dtype="<f2").reshape(3, 16)
        exact = logits.astype(np.float64)
        exact -= np.log(np.exp(exact).sum(axis=1, keepdims=True))
        np.testing.assert_array_equal(full, exact.astype("<f2"))
        for pos, compact in enumerate(window["positions"]):
            assert compact["pos"] == pos + 2 and compact["next"] == manifest["windows"][0]["tokens"][pos + 2]
            assert compact["next_lp"] == round(float(exact[pos, compact["next"]]), 5)
            assert compact["top"][0]["id"] == (0 if pos == 0 else 15)


@pytest.mark.parametrize("change", ["hash", "checkpoint", "tokens", "positions", "nonfinite", "duplicate", "missing", "extra"])
def test_converter_refuses_mismatched_or_corrupt_goldens(tmp_path, change):
    manifest, golden, logits = fixture_golden(tmp_path)
    meta = json.loads((golden / "meta.json").read_text())
    if change in ("hash", "checkpoint"):
        meta["set_sha256" if change == "hash" else "checkpoint"] = "wrong"
    elif change == "duplicate":
        meta["windows"].append(dict(meta["windows"][0]))
    elif change == "missing":
        meta["windows"].pop()
    elif change == "extra":
        meta["windows"].append({**meta["windows"][0], "id": "extra"})
    elif change == "positions":
        meta["windows"][0]["positions"] = [1, 2, 3]
    elif change == "tokens":
        np.array([1, 2, 7, 4, 5], dtype="<i4").tofile(golden / "windows/legacy/tokens.bin")
    else:
        logits[0, 0] = np.nan
        logits.tofile(golden / "windows/legacy/logits.bin")
    (golden / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        converter.convert_windows(options(tmp_path, golden))


def test_legacy_output_is_byte_identical_to_base_script(tmp_path, monkeypatch):
    golden = tmp_path / "legacy-golden"
    golden.mkdir()
    rng = np.random.default_rng(72)
    tokens = rng.integers(0, 64, 514, dtype=np.int32)
    tokens.tofile(golden / "tokens.bin")
    logits = rng.normal(size=(514, 64)).astype(np.float32)
    logits.tofile(golden / "logits.bin")
    (golden / "meta.json").write_text(json.dumps({"snapshot": str(tmp_path / "snapshot"), "reference": "synthetic"}))
    original = subprocess.check_output(["git", "-C", str(ROOT), "show",
        "7e5f769:scripts/bench/make-fidelity-reference.py"], text=True)
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    monkeypatch.setattr(sys, "argv", ["make", "--golden", str(golden), "--model", "test", "--out", str(before)])
    exec(compile(original, "original_make_reference.py", "exec"), {"__name__": "__main__"})
    monkeypatch.setattr(sys, "argv", ["make", "--golden", str(golden), "--model", "test", "--out", str(after)])
    converter.main()
    assert before.read_bytes() == after.read_bytes()
    # Schema2 --top-k 12 reproduces all legacy compact fields to five decimals.
    legacy = json.loads(after.read_text())
    manifest = tiny_set()
    manifest["windows"] = [{"id": "legacy", "block": "E", "bucket": "0-2K", "tokens": legacy["tokens"],
        "roles": ["ctx"] * len(legacy["tokens"]), "score_from": 1}]
    manifest["set_sha256"] = set_hash(manifest)
    rows = [write_scored_logits(golden, manifest["windows"][0], logits[:512])]
    finish_golden(golden, manifest, rows, snapshot=str(tmp_path / "snapshot"),
                  snapshot_identity={"snapshot_revision": "fixture"}, prefix_qualification=fixture_proof(manifest, vocab=64))
    args = options(tmp_path, golden)
    converter.convert_windows(args)
    window = json.loads(args.out.read_text())["windows"][0]
    assert [[t["id"] for t in p["top"]] for p in window["positions"]] == legacy["ids"]
    assert [[t["lp"] for t in p["top"]] for p in window["positions"]] == legacy["lps"]
    assert [p["tail_lp"] for p in window["positions"]] == legacy["tail_lp"]
    assert [p["next_lp"] for p in window["positions"]] == legacy["next_lp"]


@pytest.mark.parametrize("mutation", ["length", "mask", "id", "bucket", "duplicate", "quick"])
def test_window_contract_failures(mutation):
    m = tiny_set()
    w = m["windows"][0]
    if mutation == "length":
        w["tokens"] = [0] * 16385
        w["roles"] = ["ctx"] * 16385
    elif mutation == "mask":
        w["roles"][0] = "assistant"
    elif mutation == "id":
        w["id"] = "../escape"
    elif mutation == "bucket":
        w["bucket"] = "8-16K"
    elif mutation == "duplicate":
        m["windows"].append(w)
    else:
        m["quick_windows"] = ["absent"]
    m["set_sha256"] = set_hash(m)
    with pytest.raises(ValueError):
        validate_set(m)


@pytest.mark.parametrize("change", ["missing", "failed", "nonfinite", "family", "hash", "snapshot", "signed_zero"])
def test_prefix_qualification_fails_closed(tmp_path, change):
    manifest, golden, _ = fixture_golden(tmp_path)
    meta = json.loads((golden / "meta.json").read_text())
    proof = meta["prefix_qualification"]
    if change == "missing":
        del meta["prefix_qualification"]
    elif change == "failed":
        proof["passed"] = False
    elif change == "nonfinite":
        proof["finite"] = False
    elif change == "family":
        proof["family"] = "other"
    elif change == "hash":
        proof["set_sha256"] = "other"
    elif change == "snapshot":
        proof["snapshot_identity"] = {"snapshot_revision": "other"}
    else:
        left = np.zeros((512, 16), dtype=np.float32)
        right = left.copy()
        right[10, 3] = -0.0
        proof.update(prefix_comparison(left, right))
    (golden / "meta.json").write_bytes(canonical(meta))
    args = options(tmp_path, golden)
    with pytest.raises(ValueError, match="prefix"):
        converter.convert_windows(args)
    assert not args.out.exists() and not args.rows_dir.exists()


@pytest.mark.parametrize("bad", [False, True])
def test_actual_prefix_runner_compares_full_rows_and_binds_set(tmp_path, bad):
    manifest = tiny_set()
    window = manifest["windows"][0]
    window["tokens"], window["roles"] = [i % 16 for i in range(640)], ["ctx"] * 640
    manifest["set_sha256"] = set_hash(manifest)
    identity = {"snapshot_revision": "fixture"}
    a = SimpleNamespace(out=tmp_path, layers=[0])

    def execute(probe):
        assert probe._prefix_probe and probe.layers is None
        panel = load_set(probe.windows)
        entries = []
        for w in panel["windows"]:
            count = len(w["tokens"]) - w["score_from"]
            logits = np.arange(count * 16, dtype=np.float32).reshape(count, 16)
            if bad and w["id"] == "prefix_extended":
                logits[3, 2] += .125  # Same argmax, different non-top probability.
            entries.append(write_scored_logits(probe.out, w, logits))
        finish_golden(probe.out, panel, entries, snapshot_identity=identity, seconds=1.0)

    if bad:
        with pytest.raises(ValueError, match="prefix invariance"):
            qualify_prefix(a, manifest, execute)
    else:
        proof = qualify_prefix(a, manifest, execute)
        validate_qualification(proof, manifest, identity)
        assert proof["vocab"] == 16 and proof["source_window"] == window["id"]
    assert a.layers == [0] and not hasattr(a, "_prefix_probe")
    saved = json.loads(next(tmp_path.glob("prefix-gate-*/qualification.json")).read_text())
    assert saved["passed"] is not bad
    assert saved["different_rows"] == ([3] if bad else [])


def test_prefix_comparison_rejects_nonfinite_logits():
    rows = np.ones((512, 4), dtype=np.float32)
    for value in (np.nan, np.inf, -np.inf):
        rows[3, 2] = value
        proof = prefix_comparison(rows, rows.copy())
        assert not proof["passed"] and not proof["finite"]


def test_snapshot_provenance_must_match_pinned_set(tmp_path):
    snapshot = tmp_path / "revision"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "config.json").write_text('{"layers":70}')
    manifest = tiny_set()
    identity = verify_snapshot(manifest, snapshot)
    manifest["tokenizer_sha256"] = identity["tokenizer_sha256"]
    manifest["generation_arm"]["snapshot_revision"] = "revision"
    assert verify_snapshot(manifest, snapshot) == identity
    manifest["tokenizer_sha256"] = "wrong"
    with pytest.raises(ValueError, match="tokenizer"):
        verify_snapshot(manifest, snapshot)
    manifest["tokenizer_sha256"] = identity["tokenizer_sha256"]
    manifest["generation_arm"]["snapshot_revision"] = "other"
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(manifest, snapshot)


def test_v41_two_lengths_isolate_module_and_cross_layer_state():
    tree = ast.parse((ROOT / "python/reference/families/deepseek_v41/golden.py").read_text())
    names = {"initial_runtime_buffers", "reset_runtime_buffers", "stage_shared_attention", "restore_shared_attention"}
    helpers = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    scope = {}
    exec(compile(helpers, "golden_runtime_helpers", "exec"), scope)

    class Tensor:
        def __init__(self, array): self.array = np.asarray(array)
        def detach(self): return self
        def cpu(self): return self
        def clone(self): return Tensor(self.array.copy())
        def to(self, _device): return self
        def copy_(self, other): np.copyto(self.array, other.array)

    class Indexer:
        def __init__(self):
            self._buffers = {"k_cache": Tensor(np.zeros(5))}
            self._non_persistent_buffers_set = {"k_cache"}
            self.freqs_cis = "previous window"

    class Shared:
        def __init__(self):
            self.compress_kv = self.index_k = self.topk_idxs = self.candidates = None

    indexer = Indexer()
    compressor = SimpleNamespace(_buffers={"kv_state": Tensor(np.zeros(5)),
        "score_state": Tensor(np.full(5, -np.inf))}, _non_persistent_buffers_set={"kv_state", "score_state"})
    layer = SimpleNamespace(_buffers={"window_kv_cache": Tensor(np.zeros(5)),
        "freqs_cis": Tensor(np.arange(5))}, _non_persistent_buffers_set={"window_kv_cache", "freqs_cis"})
    layer.modules = lambda: [layer, indexer, compressor]
    ref = SimpleNamespace(Indexer=Indexer, SharedAttentionRuntime=Shared, shared_attn=Shared())
    initial = scope["initial_runtime_buffers"](layer)
    snapshots = []
    for length in [2, 5]:
        scope["reset_runtime_buffers"](initial, layer, ref)
        assert indexer.freqs_cis is None
        assert not indexer._buffers["k_cache"].array.any()
        assert not layer._buffers["window_kv_cache"].array.any()
        assert np.isneginf(compressor._buffers["score_state"].array).all()
        assert np.array_equal(layer._buffers["freqs_cis"].array, np.arange(5))
        indexer._buffers["k_cache"].array[:length] = length
        for name in vars(ref.shared_attn):
            setattr(ref.shared_attn, name, Tensor(indexer._buffers["k_cache"].array[:length]))
        snapshots.append(scope["stage_shared_attention"](ref.shared_attn))
        indexer._buffers["k_cache"].array[:] = -99
        compressor._buffers["score_state"].array[:] = 99
    for length, state in zip([2, 5], snapshots):
        scope["restore_shared_attention"](ref, state, "cpu")
        for value in vars(ref.shared_attn).values():
            assert value.array.shape == (length,)
            assert np.array_equal(value.array, np.full(length, length))


def test_glm_flash_window_dsa_handoff_is_per_window(tmp_path):
    from contextlib import nullcontext
    manifest = tiny_set()
    manifest["family"] = "glm5_flash"
    w = manifest["windows"][1]
    w["tokens"], w["roles"] = [1, 2, 3, 4, 5, 6, 7], ["ctx"] * 7
    manifest["set_sha256"] = set_hash(manifest)
    path = tmp_path / "input.json"
    path.write_bytes(canonical(manifest))
    tree = ast.parse((ROOT / "python/reference/families/glm5_flash/golden.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_windows")
    calls = []

    class Tensor:
        def __init__(self, data): self.data = np.asarray(data)
        def cpu(self): return Tensor(self.data.copy())
        def cuda(self): return Tensor(self.data.copy())
        def float(self): return self
        def numpy(self): return self.data.astype(np.float32)
        def __getitem__(self, key): return Tensor(self.data[key])
        def unsqueeze(self, axis): return Tensor(np.expand_dims(self.data, axis))
        def expand(self, *shape):
            shape = tuple(old if new == -1 else new for old, new in zip(self.data.shape, shape))
            return Tensor(np.broadcast_to(self.data, shape))
        def contiguous(self): return Tensor(self.data.copy())
        def mean(self, dim): return Tensor(self.data.mean(axis=dim))
        def to(self, _dtype): return self
        def copy_(self, other): self.data = other.data.copy()

    class Layer:
        def __init__(self, _config, layer_id): self.layer_id = layer_id
        def to_empty(self, **_kwargs): return self
        def eval(self): return self
        def named_parameters(self): return []
        def named_buffers(self): return []
        def __call__(self, h, *, prev_topk_indices, **_kwargs):
            length = h.data.shape[1]
            incoming = None if prev_topk_indices is None else int(prev_topk_indices.data[0, 0, 0])
            calls.append((self.layer_id, length, incoming))
            expected = length if self.layer_id == 1 else None
            assert incoming == expected
            topk = Tensor(np.full((1, length, 1), length)) if self.layer_id == 0 else None
            return h, topk

    class Norm:
        weight = Tensor([1.0])
        def cuda(self): return self
        def to(self, _dtype): return self
        def __call__(self, h): return h

    weights = SimpleNamespace(get=lambda name: Tensor(np.ones((16, 1)) if "lm_head" in name else [1.0]))
    torch = SimpleNamespace(inference_mode=nullcontext, device=lambda _name: nullcontext(),
        bfloat16="bf16", float32="f32", bool="bool", set_default_dtype=lambda _dtype: None,
        tensor=lambda data, **_kwargs: Tensor(data), arange=lambda n, **_kwargs: Tensor(np.arange(n)),
        ones=lambda *shape, **_kwargs: Tensor(np.ones(shape)), cuda=SimpleNamespace(empty_cache=lambda: None),
        nn=SimpleNamespace(functional=SimpleNamespace(
            embedding=lambda ids, _weights: Tensor(ids.data[..., None]),
            linear=lambda x, w: Tensor(x.data @ w.data.T))))
    import time
    scope = dict(torch=torch, time=time, load_set=load_set, verify_snapshot=lambda *_args: {},
        qualify=lambda *_args: None, write_scored_logits=write_scored_logits, finish_golden=finish_golden,
        PREFIX="model.language_model.", FP32_KEYS=(), load_layer=lambda *_args: None)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "glm_window_runner", "exec"), scope)
    config = SimpleNamespace(hc_mult=4, num_hidden_layers=3, hidden_size=1, rms_norm_eps=1e-6,
                             layer_types=["dsa", "dsa", "kda"])
    ref = SimpleNamespace(Glm5NextTextDecoderLayer=Layer, Glm5NextTextRMSNorm=lambda *_args: Norm())
    scope["run_windows"](SimpleNamespace(windows=path, snapshot=tmp_path, out=tmp_path, layers=None, experts_snapshot=None),
                         config, ref, weights, weights)
    assert calls == [(0, 5, None), (0, 7, None), (1, 5, 5), (1, 7, 7), (2, 5, None), (2, 7, None)]
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert [w["positions"] for w in meta["windows"]] == [list(range(2, 5)), list(range(2, 7))]


@pytest.mark.parametrize("family", ["deepseek_v41", "mimo_v2/mimo_v26", "qwen4", "glm5_flash"])
def test_goldens_have_layer_major_window_loops_and_scored_head_selection(family):
    path = ROOT / "python/reference/families" / family / "golden.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_windows")
    layer_loop = next(n for n in ast.walk(function) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "layer_id")
    assert any(isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) for n in ast.walk(layer_loop))
    source = ast.unparse(function)
    assert "score_from" in source and "write_scored_logits" in source and "finish_golden" in source
    assert "a.layers is not None" in source
    assert "load_set" in source
