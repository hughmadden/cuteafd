"""Bounded mixed K2/K6 assembly with an independent preserved-byte audit."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import shutil

from ..uniform_k2.storage import (assemble_shard, atomic_json, canonical,
                                  read_header, verify_receipt)
from .recipe import (BASE_REPO, BASE_REVISION, BASE_PAYLOAD_BYTES, BASE_TENSOR_COUNT,
                     MCG_MULTIPLIER, MODULE_COUNT, SELECTED_BF16_BYTES,
                     SELECTED_K6_BYTES, SUFFIXES, TARGET_PAYLOAD_BYTES,
                     TARGET_TENSOR_COUNT, packed_specs,
                     reviewed_modules, selected_modules)

_WIDTHS = dict(I16=2, F16=2, I32=4)
_DTYPES = dict(I16="int16", F16="float16", I32="int32")
_CHUNK = 16 * 1024 * 1024


def _spec(header, name):
    value = header[name]
    return value["dtype"], value["shape"]


def _layer_prefix(layer):
    return f"language_model.model.layers.{layer}."


def _preserved_audit(source_entries, target, *, chunk_bytes=_CHUNK):
    """Read both sides independently after assembly, with bounded open handles.

    The digest includes each sorted tensor name, length and original payload;
    every target byte is compared against that original. Header caches and
    descriptors are scoped to a single layer and released before its successor.
    """
    if not 1 <= chunk_bytes <= 64 * 1024 * 1024:
        raise ValueError("invalid preserved audit chunk bound")
    target_header, target_base = read_header(target)
    digest = hashlib.sha256()
    total = 0
    with ExitStack() as stack:
        output = stack.enter_context(Path(target).open("rb"))
        sources = {}
        for name, (path, original_name) in sorted(source_entries.items()):
            path = Path(path)
            if path not in sources:
                header, base = read_header(path)
                sources[path] = (stack.enter_context(path.open("rb")), header, base)
            original, header, base = sources[path]
            if name not in target_header or _spec(header, original_name) != _spec(target_header, name):
                raise ValueError(f"preserved tensor geometry differs: {name}")
            start, end = header[original_name]["data_offsets"]
            target_start, target_end = target_header[name]["data_offsets"]
            if end - start != target_end - target_start:
                raise ValueError(f"preserved tensor length differs: {name}")
            length = end - start
            digest.update(canonical([name, length]).encode() + b"\n")
            original.seek(base + start)
            output.seek(target_base + target_start)
            while length:
                count = min(length, chunk_bytes)
                left, right = original.read(count), output.read(count)
                if len(left) != count or left != right:
                    raise ValueError(f"preserved tensor bytes differ: {name}")
                digest.update(left)
                length -= count
            total += end - start
    return dict(tensors=len(source_entries), payload_bytes=total,
                names_and_payload_sha256=digest.hexdigest(),
                method="sorted JSON [tensor name, byte length] plus newline and payload; independent byte comparison")


def _check_packed(module, geometry, path):
    expected = packed_specs(module, geometry)
    header, base = read_header(path)
    if set(header) - {"__metadata__"} != set(expected):
        raise ValueError(f"replacement tensor inventory differs: {module}")
    for name, spec in expected.items():
        if _spec(header, name) != spec:
            raise ValueError(f"replacement EXL3 geometry differs: {name}")
    # A structurally valid MCG scalar with the wrong multiplier is unusable.
    import struct
    with Path(path).open("rb") as file:
        file.seek(base + header[module + ".mcg"]["data_offsets"][0])
        if struct.unpack("<I", file.read(4))[0] != MCG_MULTIPLIER:
            raise ValueError(f"replacement EXL3 MCG multiplier differs: {module}")
    return expected


def _assemble(source, source_entries, selection, replacements, shard, receipt_path, identity, **labels):
    if set(replacements) != set(selection):
        raise ValueError("replacement modules do not exactly cover this shard's reviewed selection")
    entries = dict(source_entries)
    expected_specs = {}
    source_payload = 0
    headers = {}
    for name, (path, original_name) in source_entries.items():
        if path not in headers:
            headers[path] = read_header(path)[0]
        value = headers[path][original_name]
        source_payload += value["data_offsets"][1] - value["data_offsets"][0]
        expected_specs[name] = _spec(headers[path], original_name)
    removed_bytes, added_bytes = 0, 0
    for module, geometry in selection.items():
        name = module + ".weight"
        if name not in entries or expected_specs[name] != ("BF16", [geometry["out_features"], geometry["in_features"]]):
            raise ValueError(f"source selected tensor geometry differs: {module}")
        del entries[name]
        del expected_specs[name]
        removed_bytes += geometry["in_features"] * geometry["out_features"] * 2
        for packed_name, (dtype, shape) in packed_specs(module, geometry).items():
            if packed_name in entries:
                raise ValueError(f"replacement collides with original tensor: {packed_name}")
            entries[packed_name] = (Path(replacements[module]), packed_name)
            expected_specs[packed_name] = (dtype, shape)
            count = 1
            for dimension in shape:
                count *= dimension
            added_bytes += count * _WIDTHS[dtype]
    preserved = {name: entry for name, entry in source_entries.items() if name in entries}
    expected_payload = source_payload - removed_bytes + added_bytes
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity or receipt["filename"] != shard.name or any(receipt.get(k) != v for k, v in labels.items()):
            raise ValueError("export shard run identity differs")
        if receipt.get("payload_bytes") != expected_payload or receipt.get("tensors") != len(entries):
            raise ValueError("export shard receipt inventory differs")
        audit = receipt.get("preserved_audit", {})
        if audit.get("tensors") != len(preserved) or audit.get("payload_bytes") != source_payload - removed_bytes:
            raise ValueError("export shard missing preserved-byte audit")
        verify_receipt(shard, receipt, checksum=True)
        header, _ = read_header(shard)
        if {name: _spec(header, name) for name in header if name != "__metadata__"} != expected_specs:
            raise ValueError("committed export geometry differs")
        return receipt
    for module, geometry in selection.items():
        _check_packed(module, geometry, replacements[module])
    if shard.exists():
        shard.unlink()  # A shard without its durable receipt is uncommitted.
    receipt = assemble_shard(entries, shard)
    if receipt["payload_bytes"] != expected_payload:
        raise ValueError("assembled payload size differs")
    audit = _preserved_audit(preserved, shard)
    receipt.update(identity=identity, filename=shard.name, tensors=len(entries),
                   preserved_audit=audit, **labels)
    atomic_json(receipt_path, receipt)
    return receipt


def assemble_layer(source, layer, replacements, export, identity):
    """Replace only the reviewed core tensors, retaining every routed K2 byte."""
    if not 0 <= layer < 93:
        raise ValueError("invalid Kimi layer")
    export = Path(export)
    export.mkdir(parents=True, exist_ok=True)
    prefix = _layer_prefix(layer)
    selection = {name: geometry for name, geometry in reviewed_modules().items() if name.startswith(prefix)}
    originals = {name: (source.snapshot / shard, name) for name, shard in source.weight_map.items() if name.startswith(prefix)}
    return _assemble(source, originals, selection, replacements,
                     export / f"model-{layer + 1:05d}-of-00094.safetensors",
                     export / f"layer-{layer:03d}-receipt.json", identity, layer=layer)


def _copy_assets(source, export):
    excluded = {"config.json", "quantize_config.json", "model.safetensors.index.json",
                "README.md", ".gitattributes", "quantization-provenance.json"}
    for path in source.snapshot.iterdir():
        if not path.is_file() or path.name in excluded or path.name.endswith((".safetensors", "-receipt.json")):
            continue
        shutil.copyfile(path, export / path.name)
    provenance = source.snapshot / "quantization-provenance.json"
    if provenance.exists():
        shutil.copyfile(provenance, export / "base-k2-quantization-provenance.json")
    old_errors = source.snapshot / "quantization-errors"
    if old_errors.exists():
        shutil.copytree(old_errors, export / "quantization-errors/base-k2", dirs_exist_ok=True)


def finalize_export(source, selection, head_replacements, export, identity, evidence_root):
    """Audit the entire standard checkpoint inventory and publish mixed metadata."""
    export = Path(export)
    if selection != selected_modules(source):
        raise ValueError("final core selection differs from reviewed source")
    extras = {name: (source.snapshot / shard, name) for name, shard in source.weight_map.items()
              if not name.startswith("language_model.model.layers.")}
    head_selection = {name: value for name, value in selection.items() if name == "language_model.lm_head"}
    _assemble(source, extras, head_selection, head_replacements,
              export / "model-00094-of-00094.safetensors", export / "extras-receipt.json", identity)
    expected_shards = {f"model-{index:05d}-of-00094.safetensors" for index in range(1, 95)}
    if {path.name for path in export.glob("*.safetensors")} != expected_shards:
        raise ValueError("final export shard inventory differs")
    expected_map = {name: shard for name, shard in source.weight_map.items()
                    if name.removesuffix(".weight") not in selection or not name.endswith(".weight")}
    for module, geometry in selection.items():
        layer = re.fullmatch(r"language_model\.model\.layers\.([0-9]+)\..*", module)
        shard = f"model-{int(layer[1]) + 1:05d}-of-00094.safetensors" if layer else "model-00094-of-00094.safetensors"
        expected_map.update({name: shard for name in packed_specs(module, geometry)})
    weight_map, total = {}, 0
    source_total = 0
    for shard in sorted(expected_shards):
        header, _ = read_header(export / shard)
        source_header, _ = read_header(source.snapshot / shard)
        source_total += sum(v["data_offsets"][1] - v["data_offsets"][0] for n, v in source_header.items() if n != "__metadata__")
        receipt_path = export / ("extras-receipt.json" if shard.startswith("model-00094") else
                                f"layer-{int(shard.split('-')[1]) - 1:03d}-receipt.json")
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity:
            raise ValueError("final export shard identity differs")
        # Assembly performed the independent byte audit; the durable receipt
        # pins that exact complete file. Resume has already rehashed its bytes.
        verify_receipt(export / shard, receipt)
        if receipt.get("preserved_audit", {}).get("tensors") is None:
            raise ValueError("final shard has no preserved-byte audit")
        shard_modules = {module: g for module, g in selection.items()
                         if expected_map[module + ".trellis"] == shard}
        replacements = {name: spec for module, g in shard_modules.items() for name, spec in packed_specs(module, g).items()}
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if name in weight_map or expected_map.get(name) != shard:
                raise ValueError(f"unexpected/duplicate exported tensor: {name}")
            if name in replacements:
                if _spec(header, name) != replacements[name]:
                    raise ValueError(f"final K6 tensor geometry differs: {name}")
            elif name not in source_header or _spec(header, name) != _spec(source_header, name):
                raise ValueError(f"preserved exported tensor geometry differs: {name}")
            weight_map[name] = shard
            total += value["data_offsets"][1] - value["data_offsets"][0]
    if weight_map != expected_map or len(weight_map) != TARGET_TENSOR_COUNT or total != TARGET_PAYLOAD_BYTES:
        raise ValueError("final export tensor/payload inventory differs")
    if len(source.weight_map) != BASE_TENSOR_COUNT or source_total != BASE_PAYLOAD_BYTES:
        raise ValueError("base K2 checkpoint inventory differs")
    quant_config = json.loads((source.snapshot / "quantize_config.json").read_text())
    tensor_storage = quant_config["tensor_storage"]
    original_storage = canonical(tensor_storage)
    if any(module in tensor_storage for module in selection):
        raise ValueError("core tensor already has quantization metadata")
    for module, geometry in selection.items():
        tensor_storage[module] = dict(quant_format="exl3", bits_per_weight=6,
            mcg_multiplier=MCG_MULTIPLIER, **geometry,
            stored_tensors={name: dict(shape=shape, torch_dtype=_DTYPES[dtype])
                            for name, (dtype, shape) in packed_specs(module, geometry).items()})
    if canonical({name: value for name, value in tensor_storage.items() if name not in selection}) != original_storage:
        raise ValueError("base K2 tensor metadata changed")
    declaration = dict(quant_method="exl3", format="exl3", checkpoint_format="exl3",
                       bits=2., core_bits=6., per_module_metadata="quantize_config.json:tensor_storage")
    config = json.loads(canonical(source.config))
    config["quantization_config"] = dict(config.get("quantization_config", {}), **declaration)
    config["text_config"]["quantization_config"] = dict(config["text_config"].get("quantization_config", {}), **declaration)
    quant_config.update(bits=2., core_bits=6., tensor_storage=tensor_storage,
                        meta=dict(quant_config.get("meta", {}), core_quantization_provenance=identity))
    quant_config["module_include"] = list(quant_config.get("module_include", [])) + ["^" + re.escape(module) + "$" for module in selection]
    atomic_json(export / "model.safetensors.index.json", dict(metadata=dict(total_size=total), weight_map=weight_map))
    atomic_json(export / "config.json", config)
    atomic_json(export / "quantize_config.json", quant_config)
    _copy_assets(source, export)
    evidence_root = Path(evidence_root)
    for path in evidence_root.rglob("*.json"):
        target = export / "quantization-errors/core-k6" / path.relative_to(evidence_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    atomic_json(export / "quantization-provenance.json", identity)
    (export / "README.md").write_text(_model_card(identity))
    return dict(tensors=len(weight_map), payload_bytes=total, shards=94,
                core_projections=MODULE_COUNT, routed_projections=247296,
                replaced_bf16_bytes=SELECTED_BF16_BYTES, core_k6_bytes=SELECTED_K6_BYTES)


def _model_card(identity):
    return f'''---
license: other
license_name: kimi-k3
license_link: https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE
base_model:
- {BASE_REPO}
- moonshotai/Kimi-K3
datasets:
- wrldsuksgo2mars/agentic-quant-panel-v1
tags:
- exl3
- quantized
- kimi_k3
---
# Kimi K3 EXL3 K2 + Core K6

Incremental core quantization of [{BASE_REPO}](https://huggingface.co/{BASE_REPO}),
pinned at `{BASE_REVISION}`. All 247,296 routed expert projections retain their
original K2 EXL3 bytes. The 926 new K6 EXL3 MCG projections comprise 465 attention
projections, 460 shared-expert and latent MoE projections, and the language head.
All other tensors retain their original bytes, including the embedding, dense
layer-0 FFN, routers, norms, recurrent KDA pieces, MLA kv_b_proj, AttnRes mixers,
vision tower and multimodal projector. The source architecture and native assets
are preserved. Tensor-specific `quantize_config.json` metadata is authoritative:
the checkpoint combines 2-bit routed experts, 6-bit selected core projections,
and native unquantized tensors. MLA kv_a logical output width 576 is stored at
width 640, with explicit logical and padded geometry in that metadata.

Calibration uses all 1,426 prompts (1,201,086 Kimi tokens) from agentic panel
revision `875ff1d3786bf4346ee7ff52ea5e3f1f698f5902`, in original order, batch one,
without packing, padding or truncation. The native segmented Kimi chat encoder
uses thinking enabled and thinking effort max. Calibration propagates the
already quantized K2 routed experts and each newly serialized core projection,
so successor projections see the corresponding incremental model activations.
Dense Hessians use the actual calibration stream; no expert-coverage identity
rows are added. Per-projection numerical evidence is in `quantization-errors/core-k6`;
the original routed-expert evidence is in `quantization-errors/base-k2`.

Quantization ran on one RTX PRO 6000 Blackwell Max-Q with a disk-backed rolling
AttnRes frontier. Every preserved tensor was independently compared byte-for-byte
against the base checkpoint, and all K6 dtype/shape and complete checkpoint
inventories were checked before publication. Full-model generation and serving
benchmark evaluation have not been performed. Source license and tokenizer files
are included. Run provenance is in `quantization-provenance.json`.
'''
