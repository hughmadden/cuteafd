"""Official snapshot MiMo tower, shared by media window goldens and G2 exports."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import types

from fidelity_media import read_fixture, write_features


def snapshot_identity(snapshot):
    return {"snapshot_revision": snapshot.name, **{key + "_sha256": hashlib.sha256(
        (snapshot / filename).read_bytes()).hexdigest() for key, filename in (
        ("config", "config.json"), ("tokenizer", "tokenizer.json"),
        ("modeling", "modeling_mimo_v2.py"), ("preprocessor", "preprocessor_config.json"))}}


def load_tower(snapshot, dtype="bf16"):
    """Execute only the official vision class definitions, never construct the LM."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors import safe_open

    config = json.loads((snapshot / "config.json").read_text())["vision_config"]
    if not config or dtype not in ("bf16", "fp32"):
        raise ValueError("snapshot has no supported official MiMo tower")
    source = (snapshot / "modeling_mimo_v2.py").read_text()
    start, end = source.index("def _rotate_half_vision"), source.index("# Audio encoder")
    namespace = dict(torch=torch, nn=nn, F=F, math=math, ACT2FN={"silu": nn.SiLU()})
    exec(compile(source[start:end], str(snapshot / "modeling_mimo_v2.py") + " [official vision]", "exec"), namespace)
    # Keep constructor-created rotary buffers; to_empty would leave them uninitialized.
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        model = namespace["MiMoVisionTransformer"](types.SimpleNamespace(**config))
    finally:
        torch.set_default_dtype(default)
    model = model.to(torch.float32 if dtype == "fp32" else torch.bfloat16)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    expected = dict(model.named_parameters())
    missing = set(expected)
    for shard in sorted({v for k, v in index.items() if k.startswith("visual.")}):
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as reader:
            for name in sorted(k for k, v in index.items() if v == shard and k.startswith("visual.")):
                key = name.removeprefix("visual.")
                if key not in expected:
                    raise ValueError(f"unexpected official tower tensor {name}")
                tensor = reader.get_tensor(name)
                if tensor.dtype != torch.bfloat16 or tensor.shape != expected[key].shape:
                    raise ValueError(f"tower checkpoint dtype/shape differs: {name}")
                with torch.no_grad():
                    expected[key].copy_(tensor)
                missing.remove(key)
    zero_biases = {"merger.ln_q.bias", "merger.mlp.0.bias", "merger.mlp.2.bias"}
    if missing != zero_biases:
        raise ValueError(f"unsupported missing tower keys: {sorted(missing)}")
    with torch.no_grad():
        for key in missing:
            expected[key].zero_()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # .to(dtype) also casts official rotary buffers, matching the full BF16 module.
    return model.cuda().eval(), config


def processor(snapshot):
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    config = json.loads((snapshot / "preprocessor_config.json").read_text())
    config.pop("image_processor_type", None)
    config.pop("processor_class", None)
    config["max_pixels"] = min(config["max_pixels"], 4096 * config["patch_size"] ** 2 * config["merge_size"] ** 2)
    return Qwen2VLImageProcessorPil(**config)


def encode_span(model, image_processor, root, span, *, dtype="bf16"):
    import torch
    from PIL import Image
    data = read_fixture(root, span)
    with Image.open(io.BytesIO(data)) as image:
        # Host preprocessing deliberately drops alpha, as the WP1 qualified path does.
        if image.mode == "RGBA":
            image = image.convert("RGB")
        prepared = image_processor(images=image, return_tensors="pt")
    grid = prepared["image_grid_thw"].tolist()
    if grid != [span["grid"]]:
        raise ValueError("official processor grid differs from prepared engine media span")
    pixels = prepared["pixel_values"].cuda().to(torch.float32 if dtype == "fp32" else torch.bfloat16)
    # LM prefix qualification pads GEMM rows; the tower must retain official arithmetic.
    import sys
    import torch.nn.functional as F
    invariant = sys.modules.get("shape_invariant")
    linear, einsum = F.linear, torch.einsum
    if invariant is not None:
        F.linear, torch.einsum = invariant._linear, invariant._einsum
    try:
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            features = model(pixels, prepared["image_grid_thw"].cuda())
    finally:
        F.linear, torch.einsum = linear, einsum
    if features.shape[0] != span["len"] or not torch.isfinite(features).all():
        raise ValueError("invalid official media feature shape or nonfinite rows")
    return features.cpu()


def window_features(args, manifest):
    import torch
    root = args.media_root or args.windows.parent
    dtype = getattr(args, "tower_dtype", "bf16")
    model, config = load_tower(args.snapshot, dtype)
    image_processor = processor(args.snapshot)
    identity = snapshot_identity(args.snapshot)
    features, entries = {}, []
    try:
        for window in manifest["windows"]:
            for span in window.get("media", []):
                binding = (span["grid"], span["fixture"])
                if span["key"] in features:
                    if features[span["key"]][0] != binding:
                        raise ValueError("one media key binds different fixtures/grids")
                    continue
                output = encode_span(model, image_processor, root, span, dtype=dtype)
                if output.shape[1] != config["out_hidden_size"]:
                    raise ValueError("tower output width differs from checkpoint")
                features[span["key"]] = (binding, output)
                if args.media_features_out and not getattr(args, "_prefix_probe", False):
                    entries.append(write_features(args.media_features_out, span,
                        output.bfloat16().contiguous().view(torch.uint16).numpy(), tower_dtype=dtype, identity=identity))
    finally:
        del model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if args.media_features_out and not getattr(args, "_prefix_probe", False):
        from fidelity_windows import canonical
        index = {"schema": "cuteafd.media.features.index/1", "family": manifest["family"],
            "checkpoint": manifest["checkpoint"], "tokenizer_sha256": identity["tokenizer_sha256"],
            "set_sha256": manifest["set_sha256"], "features": [entry["key"] for entry in entries]}
        path = args.media_features_out / "features.json"
        encoded = canonical(index) + b"\n"
        if path.exists() and path.read_bytes() != encoded:
            raise ValueError("feature index is immutable")
        path.write_bytes(encoded)
    return {key: value[1] for key, value in features.items()}, identity
