#!/usr/bin/env python3
"""Pinned-source CUDA audio diagnostic, not native encoder qualification.

NGC lacks torchaudio: execute only the SHA-checked official processor classes
and functions against torch. The transform's CPU-created tables are transferred
to CUDA without recomputing them. Compare explicit frame/rFFT geometry with the
official STFT, then run both through the same official FP32 audio tower.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Union
import urllib.request
import warnings

TORCHAUDIO_SOURCES = {
    "functional/functional.py": "e65fb4ec12d502cd4f354f0d33608f57f1939eadd5f276d5f6d23bb74bfbf58e",
    "transforms/_transforms.py": "8bec44011a827e768393d6f4cb399890fa3d1a0d4dd622af6622d231d5f5e3e4",
}
TORCH_VERSION = "2.12.0a0+5aff3928d8.nv26.05"
CUDA_VERSION = "13.2"


def reference_module():
    path = Path(__file__).with_name("audio_reference.py")
    spec = importlib.util.spec_from_file_location("mimo_audio_cpu_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_processor(source_dir: Path, device):
    import torch
    ref = reference_module()
    ns = {"torch": torch, "Tensor": torch.Tensor, "math": math, "warnings": warnings,
          "Optional": Optional, "Union": Union, "Callable": Callable}
    for name, sha in TORCHAUDIO_SOURCES.items():
        path = source_dir / "torchaudio" / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/pytorch/audio/{ref.TORCHAUDIO_COMMIT}/src/torchaudio/{name}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            if ref.digest(data) != sha:
                raise ValueError(f"torchaudio source pin mismatch: {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        text = ref.checked_source(path, sha)
        if name.startswith("functional"):
            names = ["spectrogram", "_get_spec_norms", "_hz_to_mel", "_mel_to_hz",
                     "_create_triangular_filterbank", "melscale_fbanks"]
        else:
            ns["F"] = SimpleNamespace(**{key: value for key, value in ns.items() if callable(value)})
            names = ["Spectrogram", "MelScale", "MelSpectrogram"]
        ref.definitions(text, names, ns, str(path))
    # Official constructors create Hann/filterbank buffers on CPU. Moving the
    # finished module preserves those bytes while selecting CUDA STFT/matmul.
    class DeviceMelSpectrogram(ns["MelSpectrogram"]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.to(device)
    source = ref.fetch_sources(source_dir)["tokenizer"]
    processor_ns = {"torch": torch, "MelSpectrogram": DeviceMelSpectrogram,
                    "MiMoAudioTokenizerConfig": SimpleNamespace, "MEL_TRANSFORM": None}
    ref.definitions(ref.checked_source(source, ref.SOURCES["tokenizer"]["sha256"]),
                    ["mel_spectrogram"], processor_ns, str(source))
    config = SimpleNamespace(sampling_rate=24000, nfft=960, hop_length=240,
                             window_size=960, fmin=0, fmax=None, n_mels=128)
    def extract(pcm):
        validate_pcm(pcm)
        return processor_ns["mel_spectrogram"](pcm, config).transpose(0, 1).contiguous()
    return extract, processor_ns


def validate_pcm(pcm):
    import torch
    if pcm.dtype != torch.float32 or pcm.ndim != 1:
        raise ValueError("reference input must be mono float32 PCM at 24000 Hz")
    reference_module().token_geometry(pcm.numel())
    if not torch.isfinite(pcm).all():
        raise ValueError("audio PCM contains nonfinite samples")


def framed_log_mel(pcm, window, filterbank):
    import torch
    validate_pcm(pcm)
    frames = torch.nn.functional.pad(pcm[None], (480, 480), mode="reflect").unfold(-1, 960, 240)
    spectrum = torch.fft.rfft(frames * window).abs()
    return (spectrum @ filterbank).clamp_min(1e-7).log().squeeze(0).contiguous()


def passes_gate(row):
    return (row["mel_byte_exact"] and row["byte_deterministic"]
            and row["official_module_byte_exact"] and row["rvq_agreement"] == 1.0
            and row["rel_l2"] <= 0.03 and row["mean_cos"] >= 0.9995
            and row["worst_cos"] >= 0.99)


def decoded_pcm_acceptance(rows):
    """Lossless equality and licensed-speech MP3 decoder-floor diagnostic.

    Synthetic MP3 remains reported but does not decide speech acceptance. This
    cannot qualify native compute or replace the MP3 LM KL/QA gates.
    """
    by_name = {Path(row["fixture"]).parent.name: row for row in rows}
    clips = []
    lossless = []
    for name, row in by_name.items():
        manifest = row.get("fixture_manifest", {})
        if manifest.get("kind") == "decoder_noise_floor":
            continue
        if manifest.get("format") in {"wav", "flac"}:
            lossless.append(row["rvq_mismatches"] == 0)
        elif manifest.get("format") == "mp3" and manifest.get("provenance", {}).get("license"):
            floor = by_name.get(name + "-ffmpeg-floor")
            if floor is None or floor.get("fixture_manifest", {}).get("kind") != "decoder_noise_floor":
                raise ValueError(f"{name}: missing mainstream MP3 decoder floor")
            frames = row["geometry"]["codes"]
            cap = max(floor["rvq_mismatches"], 0.02 * frames)
            clips.append({"fixture": name, "frames": frames,
                "candidate_flips": row["rvq_mismatches"], "floor_flips": floor["rvq_mismatches"],
                "clip_cap": cap, "within_clip_cap": row["rvq_mismatches"] <= cap,
                "candidate_positions_frame_codebook": row["rvq_positions_frame_codebook"],
                "floor_positions_frame_codebook": floor["rvq_positions_frame_codebook"],
                "projection_row_rel_l2": row["projection_row_rel_l2"],
                "projection_row_cosine": row["projection_row_cosine"]})
        elif manifest.get("format") != "mp3":
            raise ValueError(f"{name}: missing or unsupported fixture manifest")
    candidate = sum(clip["candidate_flips"] for clip in clips)
    floor = sum(clip["floor_flips"] for clip in clips)
    stable = all(row["byte_deterministic"] and row["official_module_byte_exact"] for row in rows)
    return {"kind": "decoded_pcm_diagnostic_not_native_or_end_to_end_qualification",
        "lossless_clips": len(lossless), "lossless_rvq_exact": bool(lossless) and all(lossless),
        "speech_mp3_clips": clips, "speech_mp3_candidate_flips": candidate,
        "speech_mp3_floor_flips": floor, "speech_mp3_aggregate_within_floor": candidate <= floor,
        "end_to_end_pending": ["MP3-conditioned LM KL <= mainstream-decoder KL floor", "audio-QA >= 10/12"],
        "passed": bool(lossless) and all(lossless) and stable and bool(clips)
            and candidate <= floor and all(clip["within_clip_cap"] for clip in clips)}


def main():
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pcm", type=Path, action="append", default=[], help="raw f32 LE fixture; repeatable")
    parser.add_argument("--pcm-swap", action="store_true", help="compare each PCM with adjacent official_pcm.f32")
    args = parser.parse_args()
    source_bytes = Path(__file__).read_bytes()
    if torch.__version__ != TORCH_VERSION or torch.version.cuda != CUDA_VERSION:
        raise ValueError("CUDA diagnostic requires NGC PyTorch 26.05 with CUDA 13.2")
    root = Path(__file__).resolve().parents[5]
    import transformers
    if Path(transformers.__file__).resolve().parent != (root / "third_party/transformers/src/transformers").resolve():
        raise ValueError("import transformers from the locked checkout")
    lock = json.loads((root / "third_party/transformers.lock.json").read_text())
    ref = reference_module()
    if lock["revision"] != ref.TRANSFORMERS_COMMIT:
        raise ValueError("CUDA diagnostic transformers source commit mismatch")
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    official, ns = official_processor(args.source_dir, device)
    codec, patch, speech, definitions, tensors = ref.load_modules(args.snapshot)
    codec.to(device); patch.to(device); speech.to(device)
    if args.pcm:
        fixtures = {str(path): torch.from_numpy(np.fromfile(path, dtype="<f4")).to(device) for path in args.pcm}
    else:
        t = torch.arange(24000, dtype=torch.float32) / 24000
        fixtures = {"silence": torch.zeros(481), "impulse": torch.cat((torch.ones(1), torch.zeros(960))),
                    "tone": 0.2 * torch.sin(2 * math.pi * 440 * t),
                    "chirp": 0.2 * torch.sin(2 * math.pi * (100 * t + 3000 * t.square())),
                    "noise": torch.rand(24000, generator=torch.Generator().manual_seed(42)) - 0.5}
        fixtures = {name: pcm.to(device) for name, pcm in fixtures.items()}
    rows = []
    for name, pcm in fixtures.items():
        reference_pcm = (torch.from_numpy(np.fromfile(Path(name).with_name("official_pcm.f32"), dtype="<f4")).to(device)
                         if args.pcm_swap else pcm)
        if reference_pcm.shape != pcm.shape:
            raise ValueError(f"{name}: decoded/resampled PCM length differs from oracle")
        expected = ref.pipeline(reference_pcm, codec, patch, speech, definitions, mel_fn=official)
        # Instantiate the same checked transform; only its frame/FFT path changes.
        official(pcm)
        transform = ns["MEL_TRANSFORM"]
        def candidate(pcm):
            return framed_log_mel(pcm, transform.spectrogram.window, transform.mel_scale.fb)
        actual = ref.pipeline(pcm, codec, patch, speech, definitions, mel_fn=candidate)
        repeated = ref.pipeline(pcm, codec, patch, speech, definitions, mel_fn=candidate)
        with torch.inference_mode():
            official_rows = patch.get_audio_feature([expected["mel"]], speech, codec)
        a, b = actual["projection"], expected["projection"]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        error = (actual["mel"] - expected["mel"]).abs()
        magnitude = expected["mel"].exp()
        signal = error[magnitude >= 1e-4]
        row = {"fixture": name, "geometry": ref.token_geometry(pcm.numel()),
               "pcm_sha256": ref.digest(ref.tensor_bytes(pcm)),
               "rvq_agreement": float((actual["codes"] == expected["codes"]).float().mean()),
               "rel_l2": float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b)),
               "mean_cos": float(cos.mean()), "worst_cos": float(cos.min()),
               "mel_byte_exact": ref.tensor_bytes(actual["mel"]) == ref.tensor_bytes(expected["mel"]),
               "mel_max_abs": float(error.max()),
               "mel_signal_max_abs": float(signal.max()) if signal.numel() else 0.0,
               "mel_energy_weighted_abs": float((error * magnitude).sum() / magnitude.sum()),
               "official_module_byte_exact": ref.tensor_bytes(b) == ref.tensor_bytes(official_rows),
               "byte_deterministic": all(ref.tensor_bytes(t) == ref.tensor_bytes(repeated[k]) for k, t in actual.items())}
        differences = torch.nonzero(actual["codes"] != expected["codes"]).cpu().tolist()
        row["rvq_mismatches"] = len(differences)
        row["rvq_positions_frame_codebook"] = differences
        row["rvq_frames_with_mismatches"] = sorted({frame for frame, _ in differences})
        row["projection_row_rel_l2"] = (torch.linalg.vector_norm(a-b, dim=-1)
            / torch.linalg.vector_norm(b, dim=-1)).cpu().tolist()
        row["projection_row_cosine"] = cos.cpu().tolist()
        row["reference_pcm_sha256"] = ref.digest(ref.tensor_bytes(reference_pcm))
        row["pcm_max_abs"] = float((pcm - reference_pcm).abs().max())
        metadata = Path(name).with_name("fixture.json")
        if metadata.is_file():
            row["fixture_manifest"] = json.loads(metadata.read_text())
        row["passed"] = passes_gate(row) if not args.pcm_swap else (
            row["byte_deterministic"] and row["official_module_byte_exact"]
            and not differences and row["rel_l2"] <= 0.03
            and row["mean_cos"] >= 0.9995 and row["worst_cos"] >= 0.99)
        print(json.dumps(row), flush=True)
        rows.append(row)
    report = {"kind": ("decoded_pcm_swap_diagnostic_not_native_qualification" if args.pcm_swap
                       else "torch_cufft_geometry_diagnostic_not_native_qualification"),
              "torch": torch.__version__, "torch_commit": torch.version.git_version,
              "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
              "reference_sha256": ref.digest(source_bytes),
              "transformers_commit": ref.TRANSFORMERS_COMMIT,
              "torchaudio_commit": ref.TORCHAUDIO_COMMIT, "torchaudio_sources": TORCHAUDIO_SOURCES,
              "processor_sources": ref.SOURCES, "snapshot": str(args.snapshot), "tensors": tensors,
              "fixtures": rows, "passed": all(row["passed"] for row in rows)}
    if args.pcm_swap and all("fixture_manifest" in row for row in rows):
        report["decoded_pcm_acceptance"] = decoded_pcm_acceptance(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cuda-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("CUDA audio geometry diagnostic failed")


if __name__ == "__main__":
    main()
