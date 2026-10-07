#!/usr/bin/env python3
"""Pinned CPU oracle for MiMo V2.6 MOPD audio, not a serving implementation.

Load only tokenizer encoder, speech embeddings and patch encoder extents. Run
snapshot definitions verbatim; no LM, tokenizer decoder or vocoder is loaded.
The official Qwen2 patch module requires SDPA for its is_causal=False override.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import urllib.request

SOURCES = {
    "tokenizer": {
        "repo": "XiaomiMiMo/MiMo-Audio-Tokenizer",
        "commit": "b62b59922979bf9f389b373169298a251587653f",
        "path": "mimo_audio_tokenizer/utils.py",
        "sha256": "39b9a93362ec990301dd3bb65e587d91f1e93729f45ed5dd235b8c94c26c62ae",
        "license": "Apache-2.0",
    },
    "mimo_audio": {
        "repo": "XiaomiMiMo/MiMo-Audio",
        "commit": "691ce54144a6844cc641fd96046a6ba20776c8b0",
        "path": "src/mimo_audio/mimo_audio.py",
        "sha256": "f9b17435db19d0e124acbc283c10c3c9b362de539fab3dc4276d62616dc15f1f",
        "license": "Apache-2.0",
    },
    "sglang": {
        "repo": "sgl-project/sglang",
        "commit": "1c42ad3679fcad7fa4609189763b01ee9f5bd28b",
        "path": "python/sglang/srt/multimodal/processors/mimo_audio.py",
        "sha256": "24589332f570e69b9c04085026d3ec9df199d742db8e361fe7d70f92a270a8a1",
        "license": "Apache-2.0",
    },
    "vllm": {
        "repo": "vllm-project/vllm",
        "commit": "e6fc81bc7892f2f58c0e347a701fc060ceef44bb",
        "path": "vllm/transformers_utils/processors/mimo_v2_omni.py",
        "sha256": "73b9e11e4d82ec1b82336cb0c60e705fedd42cdef6376fbb79a0758eb5f21c6a",
        "license": "Apache-2.0",
    },
}
SNAPSHOTS = {
    "flash": "2479e2d0029eca9a34cc7e7f55a121925f81908e",
    "pro": "adea8e2c5373181e5a973fa1ecb343cb31af214b",
}
MODEL_SHA256 = "a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad"
TRANSFORMERS_COMMIT = "62d7ebd7de4938e072b7aaeb881593b79dc56835"
TORCH_VERSION = "2.9.1+cpu"
TORCHAUDIO_VERSION = "2.9.1+cpu"
TORCH_COMMIT = "5811a8d7da873dd699ff6687092c225caffcf1bb"
TORCHAUDIO_COMMIT = "a224ab24a7f4797f6707051257265e223e12576f"
MEL_PARAMETERS = {
    "sample_rate": 24000, "n_fft": 960, "win_length": 960, "hop_length": 240,
    "f_min": 0.0, "f_max": 12000.0, "n_mels": 128, "power": 1.0,
    "center": True, "pad": 0, "pad_mode": "reflect", "normalized": False,
    "norm": None, "mel_scale": "htk", "window": "hann_periodic", "log_min": 1e-7,
    "log": "natural", "segment_frames": 6000,
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_source(path: Path, expected: str) -> str:
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"source pin mismatch: {path}")
    return data.decode()


def fetch_sources(root: Path) -> dict[str, Path]:
    paths = {}
    for name, pin in SOURCES.items():
        path = root / name / pin["path"]
        if not path.exists():
            url = f"https://raw.githubusercontent.com/{pin['repo']}/{pin['commit']}/{pin['path']}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            if digest(data) != pin["sha256"]:
                raise ValueError(f"downloaded source pin mismatch: {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        checked_source(path, pin["sha256"])
        paths[name] = path
    return paths


def definitions(source: str, names: list[str], namespace: dict, filename: str) -> dict:
    tree = ast.parse(source, filename)
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError(f"missing official definitions in {filename}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def token_geometry(samples: int) -> dict[str, int | list[int]]:
    if samples <= 480:
        raise ValueError(f"audio clip too short: {samples} samples, need > 480")
    if samples > 300 * 24000:
        raise ValueError("audio clip exceeds 300 seconds")
    frames = samples // 240 + 1
    segments = [6000] * (frames // 6000)
    if frames % 6000:
        segments.append(frames % 6000)
    codes = sum(((n + 1) // 2 + 1) // 2 for n in segments)
    return {"samples": samples, "mel_frames": frames, "segments": segments, "codes": codes,
            "tokens": (codes + 3) // 4}


def log_mel(pcm):
    import torch
    from torchaudio.transforms import MelSpectrogram
    if pcm.device.type != "cpu" or pcm.dtype != torch.float32 or pcm.ndim != 1:
        raise ValueError("reference input must be CPU mono float32 PCM at 24000 Hz")
    geometry = token_geometry(pcm.numel())
    if not torch.isfinite(pcm).all():
        raise ValueError("audio PCM contains nonfinite samples")
    transform = MelSpectrogram(
        sample_rate=24000, n_fft=960, win_length=960, hop_length=240,
        f_min=0.0, f_max=12000.0, n_mels=128, power=1.0, center=True,
        pad=0, pad_mode="reflect", normalized=False, norm=None, mel_scale="htk",
        window_fn=torch.hann_window, wkwargs={"periodic": True},
    )
    mel = torch.log(torch.clamp_min(transform(pcm[None, :]), 1e-7)).squeeze(0).transpose(0, 1)
    assert mel.shape == (geometry["mel_frames"], 128)
    return mel.contiguous()


def official_log_mel(pcm, source: Path):
    from torchaudio.transforms import MelSpectrogram
    import torch
    namespace = {"torch": torch, "MelSpectrogram": MelSpectrogram,
                 "MiMoAudioTokenizerConfig": SimpleNamespace, "MEL_TRANSFORM": None}
    text = checked_source(source, SOURCES["tokenizer"]["sha256"])
    definitions(text, ["mel_spectrogram"], namespace, str(source))
    cfg = SimpleNamespace(sampling_rate=24000, nfft=960, hop_length=240,
                          window_size=960, fmin=0, fmax=None, n_mels=128)
    return namespace["mel_spectrogram"](pcm, cfg).transpose(0, 1).contiguous()


def verify_runtime(root: Path) -> dict:
    import torch
    import torchaudio
    import transformers
    import subprocess
    if (torch.__version__, torchaudio.__version__) != (TORCH_VERSION, TORCHAUDIO_VERSION):
        raise ValueError("CPU oracle requires torch==2.9.1+cpu and torchaudio==2.9.1+cpu")
    if (torch.version.git_version, torchaudio.version.git_version) != (TORCH_COMMIT, TORCHAUDIO_COMMIT):
        raise ValueError("CPU oracle torch/torchaudio source commit mismatch")
    lock = json.loads((root / "third_party/transformers.lock.json").read_text())
    if lock["revision"] != TRANSFORMERS_COMMIT:
        raise ValueError("CPU oracle transformers source commit mismatch")
    subprocess.run([sys.executable, str(root / "scripts/build/verify-transformers-source.py"),
                    "--source", str(root / "third_party/transformers"),
                    "--lock", str(root / "third_party/transformers.lock.json")], check=True)
    expected = (root / "third_party/transformers/src/transformers").resolve()
    if Path(transformers.__file__).resolve().parent != expected:
        raise ValueError(f"import transformers from locked checkout: {expected}")
    torch.use_deterministic_algorithms(True)
    source_commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    source_dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True))
    return {"source_commit": source_commit, "source_dirty": source_dirty,
            "reference_sha256": digest(Path(__file__).read_bytes()),
            "torch": torch.__version__, "torchaudio": torchaudio.__version__,
            "torch_commit": TORCH_COMMIT, "torchaudio_commit": TORCHAUDIO_COMMIT,
            "transformers": transformers.__version__, "transformers_commit": TRANSFORMERS_COMMIT,
            "cpu_threads": torch.get_num_threads(), "torch_config": torch.__config__.show()}


def selected_tensors(path: Path, prefixes: tuple[str, ...]):
    import torch
    with path.open("rb") as file:
        size = int.from_bytes(file.read(8), "little")
        header = json.loads(file.read(size))
        for name, entry in header.items():
            if not name.startswith(prefixes):
                continue
            dtype = {"BF16": torch.bfloat16, "F32": torch.float32}[entry["dtype"]]
            start, end = entry["data_offsets"]
            file.seek(8 + size + start)
            data = bytearray(file.read(end - start))
            if len(data) != end - start:
                raise ValueError(f"truncated tensor: {path}:{name}")
            tensor = torch.frombuffer(data, dtype=dtype).reshape(entry["shape"]).float()
            yield name, tensor, {"file": str(path), "header": entry, "sha256": digest(data)}


def load_modules(snapshot: Path):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers.activations import ACT2FN
    from transformers.configuration_utils import PretrainedConfig
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    source = checked_source(snapshot / "modeling_mimo_v2.py", MODEL_SHA256)
    names = ["_parse_maybe_list", "_build_speech_embeddings", "_pad_and_group_audio_codes",
             "AudioProjection", "MiMoAudioEncoder", "MiMoAudioTokenizerConfig", "EuclideanCodebook",
             "VectorQuantization", "ResidualVectorQuantization", "ResidualVectorQuantizer",
             "AudioTokenizerRotaryEmbedding", "_at_get_position_ids", "_at_get_sequence_mask",
             "_at_unpack_hidden_states", "_at_rotate_half", "_at_apply_rotary_pos_emb",
             "AudioTokenizerAttention", "AudioTokenizerTransformerLayer", "AudioTokenizerEncoder",
             "_at_group_by_length", "tokenize_audio_batch"]
    ns = {"torch": torch, "nn": nn, "F": F, "math": math, "Optional": Optional,
          "ACT2FN": ACT2FN, "PretrainedConfig": PretrainedConfig, "Qwen2Config": Qwen2Config,
          "Qwen2Model": Qwen2Model, "_AT_LAYER_NORM": {"LayerNorm": nn.LayerNorm}}
    definitions(source, names, ns, str(snapshot / "modeling_mimo_v2.py"))
    cfg = json.loads((snapshot / "config.json").read_text())
    codec_cfg = json.loads((snapshot / "audio_tokenizer/config.json").read_text())
    with torch.device("meta"):
        codec = ns["AudioTokenizerEncoder"](ns["MiMoAudioTokenizerConfig"](**codec_cfg))
        patch = ns["MiMoAudioEncoder"](SimpleNamespace(**cfg["audio_config"]))
        speech = ns["_build_speech_embeddings"](SimpleNamespace(**cfg["audio_config"]))
    # Official checkpoints omit the unused Qwen2 vocabulary matrix. inputs_embeds
    # bypasses it; do not allocate an uninitialized 156 MiB parameter as a weight.
    patch.input_local_transformer.embed_tokens = nn.Identity()
    patch.input_local_transformer.config._attn_implementation = "sdpa"
    provenance = {}
    codec_state = {}
    for name, tensor, record in selected_tensors(snapshot / "audio_tokenizer/model.safetensors", ("encoder.",)):
        codec_state[name.removeprefix("encoder.")] = tensor
        provenance["audio_tokenizer/" + name] = record
    codec.load_state_dict(codec_state, strict=True, assign=True)
    state = {"audio_encoder": {}, "speech_embeddings": {}}
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({f for name, f in index.items() if name.startswith(("audio_encoder.", "speech_embeddings."))})
    for filename in files:
        for name, tensor, record in selected_tensors(snapshot / filename, ("audio_encoder.", "speech_embeddings.")):
            prefix, key = name.split(".", 1)
            state[prefix][key] = tensor
            provenance[name] = record
    patch.load_state_dict(state["audio_encoder"], strict=True, assign=True)
    speech.load_state_dict(state["speech_embeddings"], strict=True, assign=True)
    # Nonpersistent rotary buffers constructed on meta are not state-dict weights.
    codec.position_embedding = ns["AudioTokenizerRotaryEmbedding"](
        codec_cfg["rope_theta"], codec_cfg["d_model"] // codec_cfg["encoder_attention_heads"],
        codec.max_source_positions, codec_cfg["rope_type"])
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    patch.input_local_transformer.rotary_emb = Qwen2RotaryEmbedding(patch.input_local_transformer.config)
    return codec.eval(), patch.eval(), speech.eval(), ns, provenance


def tensor_bytes(tensor) -> bytes:
    import torch
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def pipeline(pcm, codec, patch, speech, ns, *, mel_fn=None) -> dict:
    import torch
    import torch.nn.functional as F
    stages = {"pcm": pcm, "mel": (mel_fn or log_mel)(pcm)}
    handles = []
    for name, module in [("conv1", codec.conv1), ("conv2", codec.conv2),
                         ("tokenizer_norm", codec.layer_norm), ("pre_rvq", codec.down_sample_norm)]:
        def capture(_module, _inputs, output, key=name):
            stages[key] = output.detach().clone()
        handles.append(module.register_forward_hook(capture))
    original = F.scaled_dot_product_attention
    local_calls = []
    checking_patch = False
    def checked_sdpa(q, k, v, attn_mask=None, **kwargs):
        if checking_patch and q.ndim == 4 and q.shape[-2] == 4 and q.shape[1] == 16:
            local_calls.append({"causal": bool(kwargs.get("is_causal", False)), "mask": attn_mask is not None})
            if kwargs.get("is_causal", False) or attn_mask is not None:
                raise ValueError("official patch encoder unexpectedly used causal attention")
        return original(q, k, v, attn_mask=attn_mask, **kwargs)
    F.scaled_dot_product_attention = checked_sdpa
    try:
        with torch.inference_mode():
            stages["codes"] = ns["tokenize_audio_batch"]([stages["mel"]], codec)[0]
            grouped = ns["_pad_and_group_audio_codes"](stages["codes"], 20, 4)
            stages["speech_sum"] = patch._apply_speech_embeddings(grouped, speech)
            checking_patch = True
            stages["local_transformer"] = patch._apply_input_local_transformer(stages["speech_sum"])
            stages["projection"] = patch.projection(stages["local_transformer"].reshape(grouped.shape[0], -1))
            stages["bf16_rows"] = stages["projection"].bfloat16()
        if len(local_calls) != 6:
            raise ValueError(f"expected six full-attention patch layers, got {local_calls}")
        geometry = token_geometry(pcm.numel())
        assert stages["codes"].shape == (geometry["codes"], 20)
        assert stages["projection"].shape == (geometry["tokens"], patch.out_hidden_size)
        return stages
    finally:
        F.scaled_dot_product_attention = original
        for handle in handles:
            handle.remove()


def main():
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pcm", type=Path, help="raw 24kHz mono float32 little-endian PCM")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[5]
    runtime = verify_runtime(root)
    paths = fetch_sources(args.source_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.pcm:
        fixtures = {"input": torch.from_numpy(np.fromfile(args.pcm, dtype="<f4"))}
    else:
        t = torch.arange(24000, dtype=torch.float32) / 24000
        fixtures = {"silence": torch.zeros(481), "impulse": torch.cat((torch.ones(1), torch.zeros(960))),
                    "tone": (0.2 * torch.sin(2 * math.pi * 440 * t)),
                    "chirp": 0.2 * torch.sin(2 * math.pi * (100 * t + 3000 * t.square()))}
    report = {"sources": SOURCES, "snapshot": str(args.snapshot), "model_sha256": MODEL_SHA256,
              "runtime": runtime, "mel_parameters": MEL_PARAMETERS, "fixtures": {}}
    if not args.preprocess_only:
        codec, patch, speech, ns, provenance = load_modules(args.snapshot)
        report["tensors"] = provenance
    for name, pcm in fixtures.items():
        mel = log_mel(pcm)
        official = official_log_mel(pcm, paths["tokenizer"])
        if tensor_bytes(mel) != tensor_bytes(official):
            raise ValueError(f"{name}: preprocessing not byte-exact to official extractor")
        fixture = {"geometry": token_geometry(pcm.numel()), "mel_byte_exact": True, "stages": {}}
        if args.preprocess_only:
            stages = {"pcm": pcm, "mel": mel}
        else:
            stages = pipeline(pcm, codec, patch, speech, ns)
            with torch.inference_mode():
                official_rows = patch.get_audio_feature([mel], speech, codec)
            fixture["official_module_byte_exact"] = tensor_bytes(stages["projection"]) == tensor_bytes(official_rows)
            if not fixture["official_module_byte_exact"]:
                raise ValueError(f"{name}: staged reference differs from official get_audio_feature")
            repeated = pipeline(pcm, codec, patch, speech, ns)
            fixture["byte_deterministic"] = all(tensor_bytes(t) == tensor_bytes(repeated[k]) for k, t in stages.items())
            if not fixture["byte_deterministic"]:
                raise ValueError(f"{name}: official CPU pipeline is not byte-deterministic")
        for key, tensor in stages.items():
            data = tensor_bytes(tensor)
            filename = f"{name}.{key}.bin"
            (args.output / filename).write_bytes(data)
            fixture["stages"][key] = {"file": filename, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                      "bytes": len(data), "sha256": digest(data)}
        report["fixtures"][name] = fixture
        print(json.dumps({"fixture": name, "geometry": fixture["geometry"], "mel_byte_exact": True,
                          "byte_deterministic": fixture.get("byte_deterministic")}), flush=True)
    (args.output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
