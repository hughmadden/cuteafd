"""Pinned official GLM Flash tower and media feature provenance, never the LM."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import importlib
import json
from pathlib import Path
import subprocess
import sys
import struct

from fidelity_media import publish_immutable, read_fixture, write_features

PREFIX = "model.visual."
TRANSFORMERS_REVISION = "62d7ebd7de4938e072b7aaeb881593b79dc56835"
SOURCE_MODULES = (
    "models.glm5_next.modeling_glm5_next", "models.glm5_next.image_processing_pil_glm5_next",
    "models.glm5_next.configuration_glm5_next", "vision_utils", "modeling_utils",
    "integrations.sdpa_attention", "image_utils", "image_processing_backends", "image_transforms",
    "image_processing_utils", "image_processing_base", "processing_utils", "configuration_utils",
    "activations", "initialization", "utils.generic", "utils.output_capturing",
)


def verify_sources(modeling, processing):
    """Admit only actual imports from the complete clean pinned source checkout."""
    root = Path(modeling.__file__).resolve().parents[2]
    checkout = root.parent.parent
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, timeout=10).strip()
        changes = subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
            text=True, timeout=10)
        if revision != TRANSFORMERS_REVISION or changes:
            raise ValueError("GLM media requires the clean pinned Transformers source checkout")
        modules = {"models.glm5_next.modeling_glm5_next": modeling,
                   "models.glm5_next.image_processing_pil_glm5_next": processing}
        for name in SOURCE_MODULES:
            module = modules.get(name) or importlib.import_module("transformers." + name)
            actual = Path(module.__file__).resolve()
            relative = name.replace(".", "/") + ".py"
            if actual != root / relative:
                raise ValueError(f"GLM media requires the pinned imported {name} source")
            expected = subprocess.check_output(
                ["git", "-C", str(checkout), "show", f"{TRANSFORMERS_REVISION}:src/transformers/{relative}"],
                timeout=10)
            if hashlib.sha256(actual.read_bytes()).digest() != hashlib.sha256(expected).digest():
                raise ValueError(f"GLM media requires the pinned imported {name} source")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        raise ValueError("GLM media requires the pinned Transformers source checkout") from error


def snapshot_identity(snapshot):
    from transformers.models.glm5_next import modeling_glm5_next, image_processing_pil_glm5_next
    verify_sources(modeling_glm5_next, image_processing_pil_glm5_next)
    return {"snapshot_revision": snapshot.name, **{key + "_sha256": hashlib.sha256(
        path.read_bytes()).hexdigest() for key, path in (
        ("config", snapshot / "config.json"), ("tokenizer", snapshot / "tokenizer.json"),
        ("preprocessor", snapshot / "processor_config.json"), ("modeling", Path(modeling_glm5_next.__file__)),
        ("image_processing", Path(image_processing_pil_glm5_next.__file__)),
        ("index", snapshot / "model.safetensors.index.json"))}}


def shard_path(snapshot, shard):
    path = Path(shard)
    if path.is_absolute() or any(p in ("", ".", "..") for p in shard.split("/")):
        raise ValueError("tensor shard escapes snapshot")
    return snapshot / path


def expert_snapshot_identity(snapshot):
    """Validate official-format experts by headers, not tensor-byte equivalence."""
    document = json.loads((snapshot / "config.json").read_text())
    quant = document.get("quantization_config") or {}
    if document.get("model_type") != "glm5_next" or quant.get("quant_method") not in (None, "fp8"):
        raise ValueError("GLM media expert source requires official BF16/FP8; EXL3 is unsupported")
    config = document["text_config"]
    index_bytes = (snapshot / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_bytes)["weight_map"]
    headers, records = {}, {}

    def tensor_header(name):
        if name not in index:
            raise ValueError(f"missing official expert tensor {name}")
        shard = index[name]
        if shard not in headers:
            with shard_path(snapshot, shard).open("rb") as source:
                size_bytes = source.read(8)
                if len(size_bytes) != 8:
                    raise ValueError("truncated expert safetensors header")
                size = struct.unpack("<Q", size_bytes)[0]
                if not 2 <= size <= 64 * 1024 * 1024:
                    raise ValueError("invalid expert safetensors header extent")
                data = source.read(size)
                if len(data) != size:
                    raise ValueError("truncated expert safetensors header")
                headers[shard] = json.loads(data)
        header = headers[shard].get(name)
        if not isinstance(header, dict):
            raise ValueError(f"missing official expert header {name}")
        records[name] = {"shard": shard, **header}
        return header

    dtypes = set()
    for layer in range(config["first_k_dense_replace"], config["num_hidden_layers"]):
        for expert in range(config["n_routed_experts"]):
            for projection in ("gate", "up", "down"):
                name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
                shape = [config["moe_intermediate_size"], config["hidden_size"]]
                if projection == "down":
                    shape.reverse()
                header = tensor_header(name)
                dtype = header.get("dtype")
                if dtype not in ("BF16", "F8_E4M3") or header.get("shape") != shape:
                    raise ValueError(f"unsupported official expert dtype/shape: {name}")
                dtypes.add(dtype)
                if dtype == "F8_E4M3":
                    scale = tensor_header(name.removesuffix("weight") + "weight_scale_inv")
                    if scale.get("dtype") != "F32" or scale.get("shape") != [(n + 127) // 128 for n in shape]:
                        raise ValueError(f"unsupported official FP8 expert block scale: {name}")
    if not records:
        raise ValueError("GLM reference snapshot has no routed experts")
    from fidelity_windows import canonical
    return {"snapshot_revision": snapshot.name,
            "config_sha256": hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "storage_headers_sha256": hashlib.sha256(canonical(records)).hexdigest(),
            "storage_dtypes": sorted(dtypes), "scope": "snapshot and storage-header identity; tensor bytes not hashed"}


def validate_spans(manifest, config):
    """Use checkpoint markers; a repeated arbitrary id is not an image span."""
    if config.get("model_type") != "glm5_next":
        raise ValueError("GLM media requires a glm5_next checkpoint")
    ids = [config.get(name) for name in ("image_token_id", "image_start_token_id", "image_end_token_id")]
    if any(type(n) is not int or not 0 <= n < config["text_config"]["vocab_size"] for n in ids) or len(set(ids)) != 3:
        raise ValueError("checkpoint image token ids are absent or invalid")
    placeholder, begin, end = ids
    for window in manifest["windows"]:
        tokens, covered, starts, ends = window["tokens"], set(), [], []
        for span in window.get("media", []):
            start, stop = span["start"], span["start"] + span["len"]
            if (start < 1 or stop >= len(tokens) or tokens[start - 1] != begin
                    or tokens[stop] != end or any(n != placeholder for n in tokens[start:stop])):
                raise ValueError("GLM media span differs from checkpoint image markers")
            if span["len"] > 4096:
                raise ValueError("GLM image exceeds qualified tower capacity")
            covered.update(range(start, stop))
            starts.append(start - 1)
            ends.append(stop)
        if ({i for i, token in enumerate(tokens) if token == placeholder} != covered
                or [i for i, token in enumerate(tokens) if token == begin] != starts
                or [i for i, token in enumerate(tokens) if token == end] != ends):
            raise ValueError("GLM native image tokens differ from the pinned span list")


@contextmanager
def official_tower_arithmetic():
    """The LM's fixed-M linears must not change official vision arithmetic."""
    import torch
    import torch.nn.functional as F
    invariant = sys.modules.get("shape_invariant")
    linear, einsum = F.linear, torch.einsum
    if invariant is not None:
        F.linear, torch.einsum = invariant._linear, invariant._einsum
    try:
        yield
    finally:
        F.linear, torch.einsum = linear, einsum


def load_tower(snapshot, dtype="bf16"):
    import torch
    from safetensors import safe_open
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    if dtype not in ("bf16", "fp32"):
        raise ValueError("unsupported official tower precision")
    document = json.loads((snapshot / "config.json").read_text())
    config = document["vision_config"]
    if document.get("model_type") != "glm5_next" or config["out_hidden_size"] != document["text_config"]["hidden_size"]:
        raise ValueError("GLM official tower and LM widths differ")
    vision = Glm5NextVisionConfig(**config)
    vision._attn_implementation = "sdpa"
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        model = Glm5NextVisionModel(vision)
    finally:
        torch.set_default_dtype(default)
    # HF loading casts checkpoint parameters, not canonical FP32 nonpersistent RoPE buffers.
    expected = dict(model.named_parameters())
    for parameter in expected.values():
        parameter.data = parameter.data.to(torch.float32 if dtype == "fp32" else torch.bfloat16)
    missing = set(expected)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in sorted({v for k, v in index.items() if k.startswith(PREFIX)}):
        with safe_open(str(shard_path(snapshot, shard)), framework="pt", device="cpu") as reader:
            for name in sorted(k for k, v in index.items() if v == shard and k.startswith(PREFIX)):
                key = name.removeprefix(PREFIX)
                if key not in missing:
                    raise ValueError(f"unexpected or duplicate GLM tower tensor {name}")
                tensor = reader.get_tensor(name)
                if tensor.dtype != torch.bfloat16 or tensor.shape != expected[key].shape:
                    raise ValueError(f"official BF16 tower dtype/shape differs: {name}")
                with torch.no_grad():
                    expected[key].copy_(tensor)
                missing.remove(key)
    if missing:
        raise ValueError(f"missing GLM tower tensors: {sorted(missing)}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return model.cuda().eval(), config


def processor(snapshot):
    from transformers.models.glm5_next.image_processing_pil_glm5_next import Glm5NextImageProcessorPil
    config = json.loads((snapshot / "processor_config.json").read_text())["image_processor"]
    config = dict(config)
    config.pop("image_processor_type", None)
    config["max_image_tokens"] = min(config.get("max_image_tokens", 8000), 4096)
    return Glm5NextImageProcessorPil(**config)


def encode_span(model, image_processor, root, span, *, dtype="bf16"):
    import torch
    from PIL import Image
    from transformers.image_utils import load_image
    with Image.open(io.BytesIO(read_fixture(root, span))) as image:
        # Official loading normalizes EXIF orientation before dropping alpha to RGB.
        prepared = image_processor(images=load_image(image), return_tensors="pt")
    if prepared["image_grid_thw"].tolist() != [span["grid"]]:
        raise ValueError("official GLM processor grid differs from engine media span")
    pixels = prepared["pixel_values"].cuda().to(torch.float32 if dtype == "fp32" else torch.bfloat16)
    with official_tower_arithmetic(), torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        features = model(pixels, prepared["image_grid_thw"].cuda()).pooler_output
    if features.shape != (span["len"], model.config.out_hidden_size) or not torch.isfinite(features).all():
        raise ValueError("invalid official GLM feature extent or nonfinite rows")
    return features.cpu()


def inject_embeddings(embeddings, spans, features):
    """Inject one stream before mHC replication; token ids/positions stay native."""
    for span in spans:
        value = features[span["key"]]
        if tuple(value.shape) != (span["len"], embeddings.shape[-1]):
            raise ValueError("official GLM tower and LM feature widths differ")
        embeddings[0, span["start"]:span["start"] + span["len"]].copy_(value)
    return embeddings


def window_features(args, manifest):
    import torch
    from fidelity_windows import canonical
    config = json.loads((args.snapshot / "config.json").read_text())
    validate_spans(manifest, config)
    root = args.media_root or args.windows.parent
    dtype = getattr(args, "tower_dtype", "bf16")
    identity = snapshot_identity(args.snapshot)
    model, _ = load_tower(args.snapshot, dtype)
    features, bindings, entries = {}, {}, []
    try:
        image_processor = processor(args.snapshot)
        for window in manifest["windows"]:
            for span in window.get("media", []):
                binding = (span["grid"], span["fixture"])
                if span["key"] in features:
                    if bindings[span["key"]] != binding:
                        raise ValueError("one GLM media key binds different fixtures/grids")
                    continue
                value = encode_span(model, image_processor, root, span, dtype=dtype)
                features[span["key"]], bindings[span["key"]] = value, binding
                if args.media_features_out and not getattr(args, "_prefix_probe", False):
                    entries.append(write_features(args.media_features_out, span,
                        value.bfloat16().contiguous().view(torch.uint16).numpy(), tower_dtype=dtype, identity=identity))
    finally:
        torch.cuda.synchronize()
        del model
        torch.cuda.empty_cache()
    if args.media_features_out and not getattr(args, "_prefix_probe", False):
        index = {"schema": "cuteafd.media.features.index/1", "family": manifest["family"],
            "checkpoint": manifest["checkpoint"], "tokenizer_sha256": identity["tokenizer_sha256"],
            "set_sha256": manifest["set_sha256"], "features": [entry["key"] for entry in entries]}
        path = args.media_features_out / "features.json"
        encoded = canonical(index) + b"\n"
        publish_immutable(path, encoded)
    return features, identity
