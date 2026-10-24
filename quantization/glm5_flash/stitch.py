#!/usr/bin/env python3
"""Offline, byte-preserving EXL3 + native core-FP8 checkpoint construction.

Run from the repository with python3 -m quantization.glm5_flash.stitch.
No upload, conversion, serving hook, or dependency on torch is involved.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import shutil

from quantization.stream_shard import read_header, repack

TARGET_REPO = "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-Core-FP8"
BASE_REPO = "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1"
BASE_REVISION = "0490d2f708b12145f6516555ab066aaeb401cd21"
FP8_REPO = "zai-org/GLM-5.3-Flash"
FP8_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
CONTRACT = (Path(__file__).resolve().parents[2] / "rust/crates/cuteafd-loader/src/"
            "families/glm5_flash/core-fp8-projections.json")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Snapshot:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.index = read_json(self.root / "model.safetensors.index.json")
        self.weight_map = self.index["weight_map"]
        self.headers = {}
        self.tensors = {}
        for shard in sorted(set(self.weight_map.values())):
            if Path(shard).name != shard or not shard.endswith(".safetensors"):
                raise ValueError(f"invalid shard name {shard!r}")
            header, offset = read_header(self.root / shard)
            self.headers[shard] = (header, offset)
            for name, item in header.items():
                if name == "__metadata__":
                    continue
                if name in self.tensors or self.weight_map.get(name) != shard:
                    raise ValueError(f"index/header disagreement for {name}")
                self.tensors[name] = item
        if set(self.tensors) != set(self.weight_map):
            raise ValueError("index names tensors missing from shard headers")

    def entry(self, name):
        return self.root / self.weight_map[name], name

    def span(self, name):
        shard = self.weight_map[name]
        item = self.tensors[name]
        start, end = item["data_offsets"]
        return self.root / shard, self.headers[shard][1] + start, end - start


class Hashes:
    """Bound open file handles while hashing tensor spans without decoding."""
    def __init__(self):
        self.handles = OrderedDict()

    def digest(self, source, name):
        path, start, size = source.span(name)
        stream = self.handles.pop(path, None)
        if stream is None:
            stream = path.open("rb")
        self.handles[path] = stream
        while len(self.handles) > 32:
            self.handles.popitem(last=False)[1].close()
        stream.seek(start)
        digest = hashlib.sha256()
        while size:
            chunk = stream.read(min(size, 8 * 1024 * 1024))
            if not chunk:
                raise ValueError(f"truncated tensor {name}")
            digest.update(chunk)
            size -= len(chunk)
        return digest.hexdigest()

    def close(self):
        for stream in self.handles.values():
            stream.close()
        self.handles.clear()


def selection(base, fp8):
    """Expand the runtime loader's projection contract, not a second tensor list."""
    cfg = read_json(base.root / "config.json")
    text = cfg.get("text_config", cfg)
    contract = read_json(CONTRACT)
    types = text["layer_types"]
    dense = text.get("mlp_layer_types", ["dense" if layer < text["first_k_dense_replace"] else "sparse"
                                        for layer in range(text["num_hidden_layers"])])
    if len(types) != text["num_hidden_layers"] or len(dense) != len(types):
        raise ValueError("inconsistent GLM Flash layer geometry")
    selected = set()
    for layer, (attention, mlp) in enumerate(zip(types, dense, strict=True)):
        prefix = f"model.language_model.layers.{layer}"
        if attention not in ("linear_attention", "deepseek_sparse_attention") or mlp not in ("dense", "sparse"):
            raise ValueError(f"unsupported layer kind at {layer}")
        names = [f"{prefix}.mlp{'.shared_experts' if mlp == 'sparse' else ''}.{name}"
                 for name in contract["mlp"]]
        if attention == "deepseek_sparse_attention":
            names += [f"{prefix}.self_attn.{name}" for name in contract["mla"]]
        for name in names:
            original, native = base.tensors[name], fp8.tensors[name]
            scale_name = name + "_scale_inv"
            scale = fp8.tensors[scale_name]
            shape = native["shape"]
            if (original["dtype"] != "BF16" or native["dtype"] != "F8_E4M3"
                    or len(shape) != 2 or shape != original["shape"]
                    or scale["dtype"] != "F32"
                    or scale["shape"] != [(dim + 127) // 128 for dim in shape]):
                raise ValueError(f"invalid native FP8 projection/scale {name}")
            if scale_name in base.tensors:
                raise ValueError(f"base already has FP8 scales for {name}")
            selected.update((name, scale_name))
    return selected


def source_map(base, fp8, selected):
    return {name: "fp8" if name in selected else "base" for name in sorted(set(base.tensors) | selected)}


def verify(base, fp8, output, provenance):
    selected = selection(base, fp8)
    expected = source_map(base, fp8, selected)
    if set(output.tensors) != set(expected) or set(provenance["tensors"]) != set(expected):
        raise ValueError("stitched tensor inventory differs from expected source union")
    hashes = Hashes()
    try:
        for number, (name, key) in enumerate(expected.items(), 1):
            source = {"base": base, "fp8": fp8}[key]
            actual, original = output.tensors[name], source.tensors[name]
            recorded = provenance["tensors"][name]
            if any(actual[field] != original[field] for field in ("dtype", "shape")):
                raise ValueError(f"stitched dtype/shape differs: {name}")
            digest = hashes.digest(source, name)
            if (digest != hashes.digest(output, name) or recorded != dict(source=key, sha256=digest)):
                raise ValueError(f"stitched bytes/provenance differ: {name}")
            if number % 10000 == 0:
                print(f"verified {number}/{len(expected)} tensors", flush=True)
    finally:
        hashes.close()
    return dict(tensors=len(expected), fp8_tensors=len(selected), byte_identical=True,
                tensor_bytes=sum(output.span(name)[2] for name in expected))


def model_card(provenance, total_bytes, shards):
    return f"""---
base_model:
- {BASE_REPO}
- {FP8_REPO}
library_name: cuteafd
license: mit
tags:
- glm5
- exl3
- fp8
- quantized
- mixture-of-experts
---

# GLM-5.3-Flash EXL3 K3.25 Core FP8

`{TARGET_REPO}` is a byte-preserving stitch of two existing checkpoints,
not a new quantization or calibration run. It replaces `{BASE_REPO}` for
CuteAFD's GLM Flash EXL3 serving recipe without a second runtime checkpoint.

- Routed experts, including the checkpoint's MTP experts, are unchanged EXL3
  K3/K4 bytes from `{BASE_REPO}` at `{BASE_REVISION}`.
- The MLA q_a, kv_a, q_b and output projections, dense MLPs, and shared
  experts use the official E4M3 weights and FP32 128x128 block scales from
  `{FP8_REPO}` at `{FP8_REVISION}`.
- Every other tensor is unchanged from the EXL3 source, including BF16 KDA,
  vocabulary head, embeddings, routers, norms and the bundled vision tower.
  KDA/head load-time representation choices remain the engine's responsibility.
- Payload: {total_bytes:,} tensor bytes ({total_bytes / 2**30:.3f} GiB) in
  {shards} safetensors shards. Tensor formats are discoverable from the
  standard config, safetensors index and headers; no overlay is required.

`stitch-provenance.json` records both immutable source revisions, a source and
SHA256 for every tensor, and the byte-verification result. It is audit evidence,
not a loader input. The base quantization metadata is retained unchanged apart
from the explicit native-core FP8 declaration.

No new quality or speed claims are made here. Local engine qualification is
reported separately. Original model and quantization authors retain credit;
this stitch changes storage provenance, not the routed expert quantization.
"""


def write_quantization_metadata(base, selected, destination):
    native = dict(quant_method="fp8", fmt="e4m3", activation_scheme="dynamic", weight_block_size=[128, 128],
                  modules=sorted(name.removesuffix(".weight") for name in selected if name.endswith(".weight")))
    cfg = read_json(base.root / "config.json")
    cfg["_name_or_path"] = TARGET_REPO
    # EXL3 manifests reserve top-level keys for their storage contract.
    cfg["quantization_config"].setdefault("meta", {})["native_core_quantization"] = native
    write_json(destination / "config.json", cfg)
    for filename in ("quantize_config.json", "quantization_config.json"):
        quant = read_json(base.root / filename)
        quant.setdefault("meta", {})["native_core_quantization"] = native
        write_json(destination / filename, quant)


def stitch(base, fp8, destination):
    selected = selection(base, fp8)
    expected = source_map(base, fp8, selected)
    destination.mkdir(parents=True, exist_ok=False)
    provenance = dict(schema="cuteafd.glmf-core-fp8-stitch.v1", model_id=TARGET_REPO,
                      sources={"base": dict(repo=BASE_REPO, revision=BASE_REVISION),
                               "fp8": dict(repo=FP8_REPO, revision=FP8_REVISION)}, tensors={})
    hashes = Hashes()
    weight_map = {}
    try:
        # Preserve the source shard partition; appended scales follow their weight.
        shards = sorted(set(base.weight_map.values()))
        groups = {shard: {} for shard in shards}
        for name, key in expected.items():
            source = {"base": base, "fp8": fp8}[key]
            shard = base.weight_map[name.removesuffix("_scale_inv") if name not in base.weight_map else name]
            groups[shard][name] = source.entry(name)
            weight_map[name] = shard
            provenance["tensors"][name] = dict(source=key, sha256=hashes.digest(source, name))
        for number, (shard, entries) in enumerate(groups.items(), 1):
            print(f"writing shard {number}/{len(groups)}: {shard}", flush=True)
            repack(entries, destination / shard)
    finally:
        hashes.close()
    total_bytes = sum((fp8 if key == "fp8" else base).span(name)[2] for name, key in expected.items())
    write_json(destination / "model.safetensors.index.json", dict(metadata=dict(total_size=total_bytes), weight_map=weight_map))
    # Copy serving assets, not old validation reports or claims about another artifact.
    for asset in base.root.iterdir():
        if (asset.is_file() and asset.name not in ("README.md", "config.json", "model.safetensors.index.json",
                                                  "quantize_config.json", "quantization_config.json")
                and not asset.name.endswith(".safetensors") and asset.suffix in (".json", ".jinja", ".model", ".txt")):
            shutil.copyfile(asset, destination / asset.name)
    write_quantization_metadata(base, selected, destination)
    write_json(destination / "stitch-provenance.json", provenance)
    (destination / "README.md").write_text(model_card(provenance, total_bytes, len(shards)))
    result = verify(base, fp8, Snapshot(destination), provenance)
    provenance["verification"] = result
    write_json(destination / "stitch-provenance.json", provenance)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fp8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    # Pinned provenance must describe the snapshots actually read, not arbitrary paths.
    for path, revision in ((args.base, BASE_REVISION), (args.fp8, FP8_REVISION)):
        if path.name != revision:
            parser.error(f"source snapshot must be the pinned revision {revision}")
    base, fp8 = Snapshot(args.base), Snapshot(args.fp8)
    if args.verify_only:
        result = verify(base, fp8, Snapshot(args.output), read_json(args.output / "stitch-provenance.json"))
        print(json.dumps(result, sort_keys=True))
    else:
        stitch(base, fp8, args.output)


if __name__ == "__main__":
    main()
