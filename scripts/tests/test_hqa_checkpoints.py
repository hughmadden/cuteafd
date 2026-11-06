"""HQA checkpoint assembly on a tiny fixture: both views, failures, resume, links."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quantization"))
import hqa  # noqa: E402

PARENT_REV = "1" * 40
SOURCE_REV = "2" * 40
PREFIX = "model.layers"


def safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]):
    header, cursor, payload = {}, 0, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[cursor, cursor + len(data)])
        cursor += len(data)
        payload += data
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def exl3(stem, out, inp, bits=6):
    trellis = (bytes([bits]) * (out * inp // 8))
    tensors = {f"{stem}.mcg": ("I32", [], b"\x01\x00\x00\x00"),
               f"{stem}.suh": ("F16", [inp], b"\x02" * (2 * inp)),
               f"{stem}.svh": ("F16", [out], b"\x03" * (2 * out)),
               f"{stem}.trellis": ("I16", [out * inp // 16], trellis)}
    storage = dict(bits_per_weight=bits, in_features=inp, out_features=out, quant_format="exl3",
                   stored_out_features=out, mcg_multiplier=3417055213,
                   stored_tensors={n: dict(shape=s, torch_dtype={"I32": "int32", "F16": "float16",
                                                                   "I16": "int16"}[d])
                                   for n, (d, s, _) in tensors.items()})
    return tensors, storage


@pytest.fixture
def fixture(tmp_path):
    """A two-layer parent with one EXL3 o_proj per layer (layer 1 is 'MTP')."""
    hub = tmp_path / "hub"
    parent = hub / "models--wrldsuksgo2mars--Tiny-EXL3" / "snapshots" / PARENT_REV
    source = hub / "models--org--Tiny-BF16" / "snapshots" / SOURCE_REV
    blobs = hub / "models--org--Tiny-BF16" / "blobs"
    for d in (parent, source, blobs):
        d.mkdir(parents=True)
    storage, weight_map, shard_tensors = {}, {}, {}
    for layer, bits in ((0, 6), (1, 4)):
        stem = f"{PREFIX}.{layer}.self_attn.o_proj"
        tensors, storage[stem] = exl3(stem, 32, 64, bits)
        tensors[f"{PREFIX}.{layer}.norm.weight"] = ("BF16", [32], bytes(range(64)))
        if layer == 0:  # an unchanged EXL3 attention module sharing the selection rule
            q, storage[f"{PREFIX}.0.self_attn.q_proj"] = exl3(f"{PREFIX}.0.self_attn.q_proj", 64, 32)
            tensors.update(q)
        shard = f"model-{layer + 1:05d}-of-00002.safetensors"
        safetensors(parent / shard, tensors)
        shard_tensors[shard] = tensors
        weight_map.update({n: shard for n in tensors})
    other, storage["lm_head"] = exl3("lm_head", 16, 32)
    safetensors(parent / "model-00003-of-00003.safetensors", other)
    weight_map.update({n: "model-00003-of-00003.safetensors" for n in other})
    total = sum(len(t[2]) for ts in (*shard_tensors.values(), other) for t in ts.values())
    index = dict(metadata=dict(total_size=total), weight_map=dict(sorted(weight_map.items())))
    quantize = dict(bits=4.0, checkpoint_format="exl3", format="exl3", quant_method="exl3",
                    per_module_metadata="quantize_config.json:tensor_storage",
                    module_include=[r"^model\.layers\.[01]\.self_attn\.(?:q_proj|o_proj)$", r"^lm_head$"],
                    tensor_storage=storage)
    config = dict(architectures=["Tiny"], model_type="tiny", num_hidden_layers=1,
                  quantization_config=dict(bits=4.0, quant_method="exl3",
                                           per_module_metadata="quantize_config.json:tensor_storage"))
    for name, value in (("model.safetensors.index.json", index), ("quantize_config.json", quantize),
                        ("config.json", config)):
        (parent / name).write_text(json.dumps(value, separators=(",", ":")) + "\n")
    (parent / "README.md").write_text("---\nlicense: mit\ntags:\n- exl3\n---\n# Tiny\n")
    (parent / "LICENSE").write_text("MIT License\n")
    (parent / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
                                           "model.safetensors.index.json filter=lfs diff=lfs merge=lfs -text\n")
    (parent / "quantization-errors").mkdir()
    (parent / "quantization-errors" / "index.json").write_text("{}\n")
    # Source: BF16 weights, cached as HF blobs.
    source_map, source_tree = {}, {}
    for layer in (0, 1):
        stem = f"{PREFIX}.{layer}.self_attn.o_proj"
        data = bytes((layer * 7 + i) % 251 for i in range(32 * 64 * 2))
        tensors = {f"{stem}.weight": ("BF16", [32, 64], data),
                   f"{PREFIX}.{layer}.norm.weight": ("BF16", [32], b"\x00" * 64)}
        name = f"model-{layer + 1:05d}-of-00002.safetensors"
        tmp = tmp_path / name
        safetensors(tmp, tensors)
        sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
        shutil.move(tmp, blobs / sha)
        (source / name).symlink_to(f"../../blobs/{sha}")
        source_tree[name] = dict(bytes=(blobs / sha).stat().st_size, blob=sha)
        source_map.update({n: name for n in tensors})
    source_index = json.dumps(dict(metadata={}, weight_map=source_map)).encode()
    (source / "model.safetensors.index.json").write_bytes(source_index)
    source_tree["model.safetensors.index.json"] = dict(bytes=len(source_index), blob=blob(source_index))
    parent_tree = {}
    for path in parent.rglob("*"):
        if path.is_file():
            name, data = path.relative_to(parent).as_posix(), path.read_bytes()
            parent_tree[name] = dict(bytes=len(data), blob=(hashlib.sha256(data).hexdigest()
                                     if name.endswith(".safetensors") else blob(data)))
    for path in parent.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    family = hqa.Family("tiny", "Tiny-EXL3", PARENT_REV, "org/Tiny-BF16", SOURCE_REV, SOURCE_REV,
                        r"^model\.layers\.(\d+)\.self_attn\.(o_proj)$", {"o_proj": 2}, "BF16")
    remote = {("wrldsuksgo2mars/Tiny-EXL3", PARENT_REV): parent_tree, ("org/Tiny-BF16", SOURCE_REV): source_tree}
    make = lambda **kw: hqa.Run(dataclasses.replace(family, **kw), hub=hub, state=tmp_path / "state",
                                remote=remote)
    return dict(hub=hub, parent=parent, source=source, run=make, tmp=tmp_path)


def parent_snapshot(parent):
    return {p.relative_to(parent).as_posix(): (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes())
            for p in parent.rglob("*") if p.is_file()}


def assemble(fixture, **kw):
    run = fixture["run"]()
    inv, plan, state = hqa.build(run, **kw)
    return run, inv, plan, state, Path(state["staging"])


def test_both_selections_and_validation(fixture):
    before = parent_snapshot(fixture["parent"])
    run, inv, plan, state, staging = assemble(fixture)
    assert plan["count"] == 2 and plan["spread"] == 0
    hqa_view, compact = hqa.open_view(staging, "hqa"), hqa.open_view(staging, "compact")
    restored = {m["module"] for m in inv["modules"]}
    assert {hqa_view.dispatch[s] for s in restored} == {"bf16"}
    assert {compact.dispatch[s] for s in restored} == {"exl3-K6", "exl3-K4"}
    assert hqa_view.dispatch["lm_head"] == compact.dispatch["lm_head"] == "exl3-K6"
    native = {s["file"] for s in plan["shards"]}
    assert native <= hqa_view.opened and not compact.opened & native
    assert hqa_view.identity != compact.identity
    hqa_map = hqa_view.index["weight_map"]
    assert not any(n.endswith((".trellis", ".suh", ".svh", ".mcg")) and n.rsplit(".", 1)[0] in restored
                   for n in hqa_map)
    assert (staging / hqa.VIEWS["compact"]["index"]).read_bytes() == \
        (fixture["parent"] / "model.safetensors.index.json").read_bytes()
    compact_quant = json.loads((staging / hqa.VIEWS["compact"]["quantize"]).read_bytes())
    assert compact_quant["per_module_metadata"] == "quantize_config.compact-attention.json:tensor_storage"
    assert json.loads((staging / hqa.VIEWS["compact"]["config"]).read_bytes())["quantization_config"][
        "per_module_metadata"] == "quantize_config.compact-attention.json:tensor_storage"
    hqa_quant = json.loads((staging / "quantize_config.json").read_bytes())
    assert restored.isdisjoint(hqa_quant["tensor_storage"])
    assert hqa_quant["module_include"] == [r"^model\.layers\.[01]\.self_attn\.(?:q_proj)$", r"^lm_head$"]
    results = hqa.validate(run, inv, plan, state)
    assert results["status"] == "passed"
    assert results["sizes"]["compact_active"] == inv["parent_total_size"]
    assert results["sizes"]["hqa_active"] == (inv["parent_total_size"] + plan["total_payload_bytes"]
                                               - sum(m["compact_bytes"] for m in inv["modules"]))
    for dest, name in state["inherited"].items():
        assert os.path.samefile(staging / dest, fixture["parent"] / name)
    assert parent_snapshot(fixture["parent"]) == before


def test_compact_loads_without_native_shards(fixture):
    run, inv, plan, state, staging = assemble(fixture)
    for shard in plan["shards"]:
        (staging / shard["file"]).unlink()
    view = hqa.open_view(staging, "compact")
    assert view.active_bytes == inv["parent_total_size"]
    with pytest.raises(hqa.SelectionError, match="unavailable"):
        hqa.open_view(staging, "hqa")


def test_header_scan_ignores_unselected_tensors(fixture):
    """Inherited shards still hold the compact EXL3 buffers; HQA must ignore them."""
    run, inv, plan, state, staging = assemble(fixture)
    view = hqa.open_view(staging, "hqa")
    header, _ = hqa.read_header(staging / "model-00001-of-00002.safetensors")
    stray = set(header) - {"__metadata__"} - set(view.index["weight_map"])
    assert stray == {f"{PREFIX}.0.self_attn.o_proj.{s}" for s in ("mcg", "suh", "svh", "trellis")}


@pytest.mark.parametrize("damage", ["missing-compact-index", "missing-compact-quantize",
                                    "mixed-config-pointer", "compact-has-native", "hqa-has-exl3"])
def test_missing_or_mixed_bundles_fail(fixture, damage):
    run, inv, plan, state, staging = assemble(fixture)
    view = fixture["tmp"] / "view"
    shutil.copytree(staging, view, symlinks=False)
    mode = "hqa" if damage == "hqa-has-exl3" else "compact"
    if damage == "missing-compact-index":
        (view / hqa.VIEWS["compact"]["index"]).unlink()
    elif damage == "missing-compact-quantize":
        (view / hqa.VIEWS["compact"]["quantize"]).unlink()
    elif damage == "mixed-config-pointer":
        path = view / hqa.VIEWS["compact"]["config"]
        config = json.loads(path.read_bytes())
        config["quantization_config"]["per_module_metadata"] = "quantize_config.json:tensor_storage"
        path.write_text(json.dumps(config))
    elif damage == "compact-has-native":
        path = view / hqa.VIEWS["compact"]["quantize"]
        quant = json.loads(path.read_bytes())
        quant["native_attention"] = {}
        path.write_text(json.dumps(quant))
    else:
        path = view / "model.safetensors.index.json"
        index = json.loads(path.read_bytes())
        index["weight_map"][f"{PREFIX}.0.self_attn.o_proj.trellis"] = "model-00001-of-00002.safetensors"
        path.write_text(json.dumps(index))
    with pytest.raises(hqa.SelectionError):
        hqa.open_view(view, mode)


def test_view_identities_separate_caches(fixture):
    run, inv, plan, state, staging = assemble(fixture)
    a, b = hqa.open_view(staging, "hqa"), hqa.open_view(staging, "compact")
    assert a.identity != b.identity
    assert hqa.open_view(staging, "hqa").identity == a.identity


def test_interrupted_write_resumes(fixture):
    run = fixture["run"]()
    with pytest.raises(KeyboardInterrupt):
        hqa.build(run, fault_after=1)
    staging = next((fixture["hub"] / "models--wrldsuksgo2mars--Tiny-EXL3-HQA" / ".hqa-staging").iterdir())
    assert list(staging.glob("*.partial")) and not list(staging.glob("hqa-attention-*.safetensors"))
    inv, plan, state = hqa.build(fixture["run"]())
    assert not list(staging.glob("*.partial"))
    for name, record in state["shards"].items():
        assert hashlib.sha256((staging / name).read_bytes()).hexdigest() == record["sha256"]
    assert hqa.validate(fixture["run"](), inv, plan, state)["status"] == "passed"


def test_idempotent_rebuild(fixture):
    run, inv, plan, state, staging = assemble(fixture)
    inodes = {p.name: p.stat().st_ino for p in staging.iterdir()}
    inv2, plan2, state2 = hqa.build(fixture["run"]())
    assert state2["run_id"] == state["run_id"] and state2["shards"] == state["shards"]
    assert {p.name: p.stat().st_ino for p in staging.iterdir()} == inodes


def test_unexplained_staging_entry_rejected(fixture):
    run, inv, plan, state, staging = assemble(fixture)
    (staging / "stray.bin").write_bytes(b"x")
    with pytest.raises(hqa.HqaError, match="unexplained"):
        hqa.build(fixture["run"]())


def test_changed_shard_is_diagnosed(fixture):
    run, inv, plan, state, staging = assemble(fixture)
    name = plan["shards"][0]["file"]
    data = bytearray((staging / name).read_bytes())
    (staging / name).unlink()
    data[-1] ^= 1
    (staging / name).write_bytes(bytes(data))
    with pytest.raises(hqa.HqaError, match="differs from its record"):
        hqa.build(fixture["run"]())


def test_source_and_count_mismatch_fail(fixture):
    with pytest.raises(hqa.HqaError, match="counts"):
        hqa.inventory(fixture["run"](expected={"o_proj": 3}))
    with pytest.raises(hqa.HqaError, match="source package"):
        hqa.inventory(fixture["run"](scales=("qscale_weight",), block=128))


def test_new_metadata_never_writes_through_links(fixture):
    run, inv, plan, state, staging = assemble(fixture)
    for name in ("config.json", "quantize_config.json", "model.safetensors.index.json", "README.md"):
        assert not os.path.samefile(staging / name, fixture["parent"] / name) if (staging / name).exists() else True
    with pytest.raises(hqa.HqaError, match="differs"):
        hqa.write_new(staging / "LICENSE", b"changed\n")
    assert (fixture["parent"] / "LICENSE").read_text() == "MIT License\n"


def test_shard_balance_is_deterministic():
    modules = [dict(module=f"m{i}", native_bytes=b) for i, b in enumerate([9, 7, 7, 5, 3, 3, 1])]
    hqa.SHARD_TARGET = 10
    try:
        plan = hqa.shard_plan(modules)
    finally:
        hqa.SHARD_TARGET = 2_000_000_000
    assert plan["count"] == 4
    assert [s["payload_bytes"] for s in plan["shards"]] == [9, 10, 8, 8]
    assert [s["modules"] for s in plan["shards"]] == [["m0"], ["m1", "m5"], ["m2", "m6"], ["m3", "m4"]]


def test_redirect_changes_only_references():
    raw = b'{"a":"quantize_config.json:tensor_storage","b":"quantize_config.json is here","c":["config.json"]}\n'
    mapping = {"quantize_config.json": "quantize_config.compact-attention.json",
               "config.json": "config.compact-attention.json"}
    out, found = hqa.redirect(raw, mapping)
    assert json.loads(out) == {"a": "quantize_config.compact-attention.json:tensor_storage",
                               "b": "quantize_config.json is here", "c": ["config.compact-attention.json"]}
    assert hqa.redirect(out, {v: k for k, v in mapping.items()})[0] == raw
