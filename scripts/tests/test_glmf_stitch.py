"""Offline stitching preserves source payloads and never needs loader side files."""
import json
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quantization.glm5_flash import stitch as tool


def shard(path, tensors):
    header, payload = {}, bytearray()
    for name, (dtype, shape, data) in sorted(tensors.items()):
        start = len(payload)
        payload.extend(data)
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(payload)])
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def fixture(tmp_path):
    base, fp8 = tmp_path / "base", tmp_path / "fp8"
    base.mkdir()
    fp8.mkdir()
    cfg = dict(model_type="glm5_next", text_config=dict(num_hidden_layers=2, first_k_dense_replace=1,
        layer_types=["linear_attention", "deepseek_sparse_attention"], mlp_layer_types=["dense", "sparse"]),
        quantization_config=dict(quant_method="exl3", bits=3))
    (base / "config.json").write_text(json.dumps(cfg))
    for name in ("quantize_config.json", "quantization_config.json"):
        (base / name).write_text(json.dumps(dict(quant_method="exl3", tensor_storage={})))
    (base / "tokenizer_config.json").write_text('{"chat_template":"unchanged"}')
    contract = json.loads(tool.CONTRACT.read_text())
    names = []
    for layer in range(2):
        prefix = f"model.language_model.layers.{layer}"
        names += [f"{prefix}.mlp{'.shared_experts' if layer else ''}.{name}" for name in contract["mlp"]]
        if layer:
            names += [f"{prefix}.self_attn.{name}" for name in contract["mla"]]
    original = {name: ("BF16", [128, 128], b"\x01\x02" * 128**2) for name in names}
    original["model.language_model.layers.1.mlp.experts.0.gate_proj.trellis"] = ("I16", [4], b"\x03\x04" * 4)
    original["model.language_model.layers.0.self_attn.q_proj.weight"] = ("BF16", [128, 128], b"\x05\x06" * 128**2)
    native = {name: ("F8_E4M3", [128, 128], bytes([index + 1]) * 128**2) for index, name in enumerate(names)}
    native.update({name + "_scale_inv": ("F32", [1, 1], struct.pack("<f", 0.5)) for name in names})
    native["ignored.official.tensor"] = ("BF16", [1], b"\x07\x08")
    for root, tensors in ((base, original), (fp8, native)):
        shard(root / "model-00001-of-00001.safetensors", tensors)
        (root / "model.safetensors.index.json").write_text(json.dumps(dict(
            metadata={}, weight_map={name: "model-00001-of-00001.safetensors" for name in tensors})))
    return tool.Snapshot(base), tool.Snapshot(fp8)


def test_stitch_exact_union_and_native_formats(tmp_path):
    base, fp8 = fixture(tmp_path)
    output = tmp_path / "stitched"
    result = tool.stitch(base, fp8, output)
    assert result == dict(tensors=22, fp8_tensors=20, byte_identical=True,
                          tensor_bytes=10 * (128**2 + 4) + 128**2 * 2 + 8)
    stitched = tool.Snapshot(output)
    assert "ignored.official.tensor" not in stitched.tensors
    assert stitched.tensors["model.language_model.layers.0.self_attn.q_proj.weight"]["dtype"] == "BF16"
    cfg = json.loads((output / "config.json").read_text())
    assert cfg["_name_or_path"] == tool.TARGET_REPO
    assert cfg["quantization_config"]["quant_method"] == "exl3"
    assert cfg["quantization_config"]["native_core_quantization"]["weight_block_size"] == [128, 128]
    assert json.loads((output / "quantize_config.json").read_text())["tensor_storage"] == {}
    assert (output / "tokenizer_config.json").read_bytes() == (base.root / "tokenizer_config.json").read_bytes()
    assert tool.TARGET_REPO in (output / "README.md").read_text()
    provenance = json.loads((output / "stitch-provenance.json").read_text())
    assert all("/" not in row["source"] for row in provenance["tensors"].values())
    # The checkpoint tensor catalog works independently of audit provenance.
    (output / "stitch-provenance.json").unlink()
    assert set(tool.Snapshot(output).tensors) == set(stitched.tensors)
    with pytest.raises(FileExistsError):
        tool.stitch(base, fp8, output)


def test_verify_detects_changed_bytes_and_provenance(tmp_path):
    base, fp8 = fixture(tmp_path)
    output = tmp_path / "stitched"
    tool.stitch(base, fp8, output)
    provenance = json.loads((output / "stitch-provenance.json").read_text())
    provenance["tensors"][next(iter(provenance["tensors"]))]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes/provenance differ"):
        tool.verify(base, fp8, tool.Snapshot(output), provenance)
    provenance = json.loads((output / "stitch-provenance.json").read_text())
    path = output / "model-00001-of-00001.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="bytes/provenance differ"):
        tool.verify(base, fp8, tool.Snapshot(output), provenance)


def test_stitch_rejects_invalid_scale_before_writing(tmp_path):
    base, fp8 = fixture(tmp_path)
    selected = next(name for name in tool.selection(base, fp8) if name.endswith("_scale_inv"))
    fp8.tensors[selected]["shape"] = [2, 1]
    with pytest.raises(ValueError, match="invalid native FP8"):
        tool.stitch(base, fp8, tmp_path / "stitched")
    assert not (tmp_path / "stitched").exists()


def test_stitch_refuses_index_header_disagreement(tmp_path):
    base, _ = fixture(tmp_path)
    index = json.loads((base.root / "model.safetensors.index.json").read_text())
    index["weight_map"]["missing.weight"] = "model-00001-of-00001.safetensors"
    (base.root / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="missing from shard headers"):
        tool.Snapshot(base.root)
