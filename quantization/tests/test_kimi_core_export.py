"""CPU checks for the incremental Kimi core inventory and byte-preserving export."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3.core_k6 import export, recipe
from kimi_k3.uniform_k2.storage import read_header


def config_fixture():
    return dict(num_hidden_layers=93, hidden_size=7168, vocab_size=163840,
                q_lora_rank=1536, kv_lora_rank=512, num_attention_heads=96,
                qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
                num_experts=896, num_experts_per_token=16, num_shared_experts=2,
                moe_intermediate_size=3072, routed_expert_hidden_size=3584,
                first_k_dense_replace=1,
                linear_attn_config=dict(full_attn_layers=list(recipe.FULL_ATTENTION_LAYERS),
                    kda_layers=[x for x in range(1, 94) if x not in recipe.FULL_ATTENTION_LAYERS],
                    head_dim=128, num_heads=96))


@pytest.fixture
def inventory_source(monkeypatch, tmp_path):
    """926 logical matrices, represented by headers without allocating their data."""
    selection = recipe.reviewed_modules()
    headers, weight_map = {}, {}
    for module, geometry in selection.items():
        name = module + ".weight"
        parts = module.split(".")
        shard = f"layer-{parts[3]}.safetensors" if ".layers." in module else "extras.safetensors"
        header = headers.setdefault(shard, {})
        start = max([value["data_offsets"][1] for value in header.values()] or [0])
        header[name] = dict(dtype="BF16", shape=[geometry["out_features"], geometry["in_features"]],
                            data_offsets=[start, start + geometry["out_features"] * geometry["in_features"] * 2])
        weight_map[name] = shard
    monkeypatch.setattr(recipe, "read_header", lambda path: (headers[path.name], 8))
    return SimpleNamespace(snapshot=tmp_path, config=dict(text_config=config_fixture()),
                           text_config=config_fixture(), weight_map=weight_map, headers=headers)


def test_exact_recipe_scope_and_padded_payload(inventory_source):
    selected = recipe.selected_modules(inventory_source)
    assert len(selected) == 926
    assert sum(".self_attn." in name for name in selected) == 465
    assert sum(".block_sparse_moe." in name for name in selected) == 460
    assert sum(name.endswith("kv_a_proj_with_mqa") for name in selected) == 24
    assert "language_model.model.layers.0.mlp.gate_proj" not in selected
    assert "language_model.model.layers.3.self_attn.kv_b_proj" not in selected
    assert not any(".experts." in name for name in selected)
    assert selected["language_model.model.layers.3.self_attn.kv_a_proj_with_mqa"] == dict(
        in_features=7168, out_features=576, stored_out_features=640)
    total = 0
    for module, geometry in selected.items():
        for dtype, shape in recipe.packed_specs(module, geometry).values():
            count = 1
            for dimension in shape:
                count *= dimension
            total += count * dict(I16=2, F16=2, I32=4)[dtype]
    assert total == recipe.SELECTED_K6_BYTES
    assert recipe.BASE_PAYLOAD_BYTES - recipe.SELECTED_BF16_BYTES + total == recipe.TARGET_PAYLOAD_BYTES
    assert recipe.BASE_TENSOR_COUNT + len(selected) * 3 == recipe.TARGET_TENSOR_COUNT


@pytest.mark.parametrize("damage", ["dtype", "shape", "missing", "extra", "config", "layers", "already"])
def test_selector_fails_closed(inventory_source, damage):
    source = inventory_source
    name = "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight"
    value = source.headers[source.weight_map[name]][name]
    if damage == "dtype":
        value["dtype"] = "F16"
    elif damage == "shape":
        value["shape"][0] = 640
    elif damage == "missing":
        del source.weight_map[name]
    elif damage == "extra":
        source.weight_map["language_model.model.layers.3.self_attn.q_proj.weight"] = "unexpected"
    elif damage == "config":
        source.text_config["hidden_size"] = 8192
    elif damage == "layers":
        source.text_config["linear_attn_config"]["full_attn_layers"] = [1]
    else:
        source.weight_map[name.removesuffix(".weight") + ".trellis"] = "unexpected"
    with pytest.raises(ValueError):
        recipe.selected_modules(source)


def packed_fixture(path, module, geometry):
    tensors = {}
    for name, (dtype, shape) in recipe.packed_specs(module, geometry).items():
        tensors[name] = torch.ones(shape, dtype={"I16": torch.int16, "F16": torch.float16,
                                               "I32": torch.int32}[dtype])
    tensors[module + ".mcg"] = torch.tensor(recipe.MCG_MULTIPLIER - 2**32, dtype=torch.int32)
    save_file(tensors, path)
    return path


@pytest.fixture
def tiny_source(monkeypatch, tmp_path):
    snapshot = tmp_path / "source"
    snapshot.mkdir()
    module = "language_model.model.layers.1.self_attn.q_proj"
    geometry = dict(in_features=128, out_features=128, stored_out_features=128)
    original = {module + ".weight": torch.arange(128 * 128).reshape(128, 128).to(torch.bfloat16),
                "language_model.model.layers.1.block_sparse_moe.experts.0.w1.trellis": torch.arange(32, dtype=torch.int16),
                "language_model.model.layers.1.input_layernorm.weight": torch.ones(128, dtype=torch.bfloat16)}
    shard = "model-00002-of-00094.safetensors"
    save_file(original, snapshot / shard)
    source = SimpleNamespace(snapshot=snapshot, weight_map={name: shard for name in original}, config={})
    monkeypatch.setattr(export, "reviewed_modules", lambda: {module: geometry})
    candidate = packed_fixture(tmp_path / "packed.safetensors", module, geometry)
    return source, module, geometry, candidate, tmp_path / "output"


def test_layer_assembly_preserves_routed_and_native_bytes(tiny_source):
    source, module, geometry, candidate, output = tiny_source
    receipt = export.assemble_layer(source, 1, {module: candidate}, output, dict(run="test"))
    header, _ = read_header(output / receipt["filename"])
    assert module + ".weight" not in header
    assert set(recipe.packed_specs(module, geometry)) <= set(header)
    assert receipt["preserved_audit"]["tensors"] == 2
    assert receipt["preserved_audit"]["payload_bytes"] == 64 + 256
    assert receipt["payload_bytes"] == 128 * 128 * 6 // 8 + 128 * 2 * 2 + 4 + 320
    assert export.assemble_layer(source, 1, {module: candidate}, output, dict(run="test")) == receipt


def test_orphan_assembly_is_replaced(tiny_source):
    source, module, _, candidate, output = tiny_source
    output.mkdir()
    (output / "model-00002-of-00094.safetensors").write_bytes(b"incomplete")
    receipt = export.assemble_layer(source, 1, {module: candidate}, output, "identity")
    assert (output / receipt["filename"]).stat().st_size > 10


def test_resume_rejects_identity_and_payload_tampering(tiny_source):
    source, module, _, candidate, output = tiny_source
    receipt = export.assemble_layer(source, 1, {module: candidate}, output, "identity")
    with pytest.raises(ValueError, match="identity"):
        export.assemble_layer(source, 1, {module: candidate}, output, "other")
    path = output / receipt["filename"]
    with path.open("r+b") as file:
        file.seek(-1, 2)
        file.write(b"\xff")
    # Even preserving stat data cannot defeat the resume checksum check.
    import os
    os.utime(path, ns=(path.stat().st_atime_ns, receipt["file_identity"]["mtime_ns"]))
    with pytest.raises(ValueError, match="checksum"):
        export.assemble_layer(source, 1, {module: candidate}, output, "identity")


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "mcg"])
def test_replacement_validation(tiny_source, damage):
    source, module, geometry, candidate, output = tiny_source
    if damage == "missing":
        replacements = {}
    elif damage == "extra":
        replacements = {module: candidate, "wrong": candidate}
    else:
        from safetensors.torch import load_file
        tensors = load_file(candidate)
        if damage == "shape":
            tensors[module + ".svh"] = torch.ones(64, dtype=torch.float16)
        else:
            tensors[module + ".mcg"] = torch.tensor(0, dtype=torch.int32)
        save_file(tensors, candidate)
        replacements = {module: candidate}
    with pytest.raises(ValueError):
        export.assemble_layer(source, 1, replacements, output, "identity")


def test_independent_audit_catches_preserved_byte_corruption(tiny_source, monkeypatch):
    source, module, _, candidate, output = tiny_source
    original_assemble = export.assemble_shard

    def corrupt(entries, path):
        receipt = original_assemble(entries, path)
        header, base = read_header(path)
        name = "language_model.model.layers.1.input_layernorm.weight"
        with path.open("r+b") as file:
            file.seek(base + header[name]["data_offsets"][0])
            file.write(b"\xff\xff")
        return receipt

    monkeypatch.setattr(export, "assemble_shard", corrupt)
    with pytest.raises(ValueError, match="preserved tensor bytes"):
        export.assemble_layer(source, 1, {module: candidate}, output, "identity")
    assert not (output / "layer-001-receipt.json").exists()


def test_asset_copy_retains_base_provenance_without_overwriting_core(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for name in ("tokenizer.json", "LICENSE", "modeling_kimi_linear.py", "config.json",
                 "quantize_config.json", "README.md", "quantization-provenance.json", "layer-001-receipt.json"):
        (source / name).write_text("original " + name)
    (source / "quantization-errors").mkdir()
    (source / "quantization-errors/layer-001.json").write_text("base errors")
    (output / "quantize_config.json").write_text("core metadata")
    export._copy_assets(SimpleNamespace(snapshot=source), output)
    assert (output / "quantize_config.json").read_text() == "core metadata"
    assert (output / "base-k2-quantization-provenance.json").read_text().endswith("quantization-provenance.json")
    assert (output / "quantization-errors/base-k2/layer-001.json").read_text() == "base errors"
    assert (output / "modeling_kimi_linear.py").exists()
    assert not (output / "layer-001-receipt.json").exists()


def test_mla_logical_padding_requires_full_stored_tensors(tmp_path):
    module = "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa"
    geometry = dict(in_features=128, out_features=576, stored_out_features=640)
    candidate = packed_fixture(tmp_path / "padded.safetensors", module, geometry)
    expected = export._check_packed(module, geometry, candidate)
    assert expected[module + ".trellis"] == ("I16", [8, 40, 96])
    assert expected[module + ".svh"] == ("F16", [640])


def test_durability_pass_flushes_target_files_then_directories(tmp_path, monkeypatch):
    import os
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "original.json").write_text("source is never flushed")
    nested = target / "quantization-errors/core-k6"
    nested.mkdir(parents=True)
    (nested / "evidence.json").write_text("new evidence")
    (target / "README.md").write_text("new card")
    seen = []
    fsync = os.fsync

    def record(descriptor):
        seen.append(Path(os.readlink(f"/proc/self/fd/{descriptor}")))
        fsync(descriptor)

    monkeypatch.setattr(os, "fsync", record)
    export._fsync_tree(target)
    assert set(seen) == {nested / "evidence.json", nested, nested.parent,
                         target / "README.md", target}
    assert seen.index(nested / "evidence.json") < seen.index(nested) < seen.index(nested.parent) < seen.index(target)
    assert seen.index(target / "README.md") < seen.index(target)
    assert seen[-1] == target
    (target / "external-link").symlink_to(source / "original.json")
    with pytest.raises(ValueError, match="symlink"):
        export._fsync_tree(target)


def test_final_inventory_and_mixed_metadata(tmp_path, monkeypatch):
    source_dir, output = tmp_path / "source", tmp_path / "output"
    source_dir.mkdir()
    output.mkdir()
    module = "language_model.model.layers.0.self_attn.q_proj"
    head = "language_model.lm_head"
    geometry = dict(in_features=128, out_features=128, stored_out_features=128)
    selection = {module: geometry, head: geometry}
    source = SimpleNamespace(snapshot=source_dir, weight_map={}, config=dict(
        architectures=["KimiK3ForConditionalGeneration"], text_config=dict(
            architectures=["KimiLinearForCausalLM"])))
    source_bytes = 0
    for index in range(1, 95):
        shard = f"model-{index:05d}-of-00094.safetensors"
        if index == 94:
            tensors = {head + ".weight": torch.ones(128, 128, dtype=torch.bfloat16),
                       "language_model.model.embed_tokens.weight": torch.ones(5, 128, dtype=torch.bfloat16)}
        else:
            tensors = {f"language_model.model.layers.{index - 1}.input_layernorm.weight": torch.ones(128, dtype=torch.bfloat16)}
            if index == 1:
                tensors[module + ".weight"] = torch.ones(128, 128, dtype=torch.bfloat16)
        save_file(tensors, source_dir / shard)
        source.weight_map.update({name: shard for name in tensors})
        source_bytes += sum(value.numel() * value.element_size() for value in tensors.values())
    base_storage = {"base-k2-module": dict(quant_format="exl3", bits_per_weight=2,
                                          mcg_multiplier=recipe.MCG_MULTIPLIER, stored_tensors={})}
    (source_dir / "quantize_config.json").write_text(json.dumps(dict(
        bits=2, tensor_storage=base_storage, codebook="mcg", module_include=["base-pattern"],
        meta=dict(quantization_provenance=dict(base="unchanged")))))
    (source_dir / "quantization-provenance.json").write_text('{"base": "unchanged"}')
    monkeypatch.setattr(export, "reviewed_modules", lambda: selection)
    monkeypatch.setattr(export, "selected_modules", lambda _: selection)
    monkeypatch.setattr(export, "BASE_TENSOR_COUNT", len(source.weight_map))
    monkeypatch.setattr(export, "TARGET_TENSOR_COUNT", len(source.weight_map) + 6)
    monkeypatch.setattr(export, "BASE_PAYLOAD_BYTES", source_bytes)
    packed_bytes = (128 * 128 * 6 // 8 + 128 * 2 * 2 + 4) * 2
    target_bytes = source_bytes - 128 * 128 * 2 * 2 + packed_bytes
    monkeypatch.setattr(export, "TARGET_PAYLOAD_BYTES", target_bytes)
    candidates = {name: packed_fixture(tmp_path / f"candidate-{index}.safetensors", name, geometry)
                  for index, name in enumerate(selection)}
    for layer in range(93):
        replacements = {module: candidates[module]} if layer == 0 else {}
        export.assemble_layer(source, layer, replacements, output, "identity")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "projection.json").write_text('{"metric_status": "ok"}')
    flushed = []
    fsync_tree = export._fsync_tree
    def note_durable(directory):
        fsync_tree(directory)
        flushed.append(directory)
    monkeypatch.setattr(export, "_fsync_tree", note_durable)
    summary = export.finalize_export(source, selection, {head: candidates[head]}, output, "identity", evidence)
    assert flushed[-1] == output
    assert summary["payload_bytes"] == target_bytes
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == target_bytes
    assert head + ".weight" not in index["weight_map"]
    assert head + ".trellis" in index["weight_map"]
    actual = json.loads((output / "quantize_config.json").read_text())
    assert actual["tensor_storage"]["base-k2-module"] == base_storage["base-k2-module"]
    assert actual["tensor_storage"][head]["bits_per_weight"] == 6
    assert actual["tensor_storage"][head]["stored_out_features"] == 128
    assert actual["meta"]["quantization_provenance"] == dict(base="unchanged")
    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == source.config["architectures"]
    assert config["text_config"]["architectures"] == source.config["text_config"]["architectures"]
    assert config["quantization_config"]["bits"] == 2
    assert config["quantization_config"]["core_bits"] == 6
    assert (output / "quantization-errors/core-k6/projection.json").exists()
    (output / "orphan.safetensors").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="shard inventory"):
        export.finalize_export(source, selection, {head: candidates[head]}, output, "identity", evidence)
