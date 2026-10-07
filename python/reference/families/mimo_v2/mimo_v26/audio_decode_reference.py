#!/usr/bin/env python3
"""Qualify bounded Rust audio decoding on synthetic and opt-in licensed speech.

Pinned CPU torch/torchaudio resampling and libsndfile independently decode the
fixtures. --public-speech fetches a hash-pinned CC-BY-4.0 LibriSpeech subset;
no private recordings are accepted. --decoder-floor records independent ffmpeg
MP3 PCM for the CUDA tower swap. Additional fixture-only dependencies are
pyarrow==23.0.1 and imageio-ffmpeg==0.6.0. No tokenizer or LM is loaded here.
"""
from __future__ import annotations
import argparse
import importlib.util
import io
import json
import urllib.request
from pathlib import Path
import subprocess


PUBLIC_SPEECH = {
    "url": "https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/5be91486e11a2d616f4ec5db8d3fd248585ac07a/clean/validation-00000-of-00001.parquet",
    "sha256": "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603",
    "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "upstream": "https://www.openslr.org/12/",
    "attribution": "LibriSpeech, Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur (2015)",
}


def public_speech(directory, ref):
    import pyarrow.parquet as pq
    import soundfile as sf
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "validation.parquet"
    if not path.exists():
        with urllib.request.urlopen(PUBLIC_SPEECH["url"], timeout=120) as response:
            data = response.read(10 * 1024 * 1024)
        if ref.digest(data) != PUBLIC_SPEECH["sha256"]:
            raise ValueError("public LibriSpeech fixture digest mismatch")
        path.write_bytes(data)
    if ref.digest(path.read_bytes()) != PUBLIC_SPEECH["sha256"]:
        raise ValueError("public LibriSpeech fixture digest mismatch")
    rows = pq.read_table(path).to_pylist()
    clips = []
    for index in [0, 10, 20]:
        row = rows[index]
        encoded = row["audio"]["bytes"]
        pcm, rate = sf.read(io.BytesIO(encoded), dtype="float32")
        if rate != 16000 or pcm.ndim != 1:
            raise ValueError("pinned LibriSpeech fixture geometry changed")
        clips.append((pcm[:rate * 3], {**PUBLIC_SPEECH, "id": row["id"],
            "clip_source_sha256": ref.digest(encoded), "samples": min(len(pcm),rate * 3),
            "source_rate": rate, "excerpt_seconds": 3, "transcript_full_recording": row["text"]}))
    return clips


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-speech", action="store_true")
    parser.add_argument("--speech-source-dir", type=Path)
    parser.add_argument("--decoder-floor", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name("audio_reference.py")
    spec = importlib.util.spec_from_file_location("mimo_audio_cpu_reference", path)
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    root = Path(__file__).resolve().parents[5]
    torch.set_num_threads(8)
    runtime = ref.verify_runtime(root)
    if sf.__version__ != "0.13.1":
        raise ValueError("fixture generation requires soundfile==0.13.1")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    ffmpeg = None
    if args.decoder_floor:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_version = (subprocess.run([ffmpeg, "-version"], check=True, capture_output=True,
        timeout=30).stdout.decode().splitlines()[0] if ffmpeg else None)

    def decode(name, fmt, source, rate, subtype, provenance=None):
        encoded = args.output / f"{name}.{fmt}"
        sf.write(encoded, source, rate, format=fmt.upper(), subtype=subtype)
        destination = args.output / name
        command = [str(args.binary.resolve()), fmt, str(encoded), str(destination)]
        subprocess.run(command, check=True, timeout=60)
        actual = np.fromfile(destination / "pcm.f32", dtype="<f4")
        # Whole preparation must be byte deterministic, including nanomp3 gapless trimming.
        repeated = subprocess.run(command, capture_output=True, check=True, timeout=60)
        if repeated.returncode != 0 or destination.joinpath("pcm.f32").read_bytes() != actual.tobytes():
            raise ValueError(f"{name}: repeated decoding changed PCM bytes")
        decoded, decoded_rate = sf.read(encoded, dtype="float32", always_2d=True)
        official = torch.from_numpy(decoded.T.copy())
        if decoded_rate != 24000:
            official = torchaudio.transforms.Resample(decoded_rate, 24000)(official)
        expected = official.mean(0).numpy()
        (destination / "official_pcm.f32").write_bytes(expected.astype("<f4").tobytes())
        geometry = json.loads(destination.joinpath("geometry.json").read_text())
        geometry_ok = geometry == {k: v for k, v in ref.token_geometry(actual.size).items() if k != "samples"}
        max_abs = float(np.max(np.abs(actual-expected))) if actual.size == expected.size else None
        limit = 2e-6 if fmt == "mp3" else 2e-7
        row = {"fixture": name, "format": fmt, "rate": rate, "channels": decoded.shape[1],
               "samples": actual.size, "official_samples": expected.size,
               "pcm_sha256": ref.digest(actual.tobytes()), "geometry": geometry,
               "pcm_byte_exact": actual.size == expected.size and actual.tobytes() == expected.tobytes(),
               "pcm_max_abs": max_abs, "limit": limit, "byte_deterministic": True,
               "passed": geometry_ok and max_abs is not None and max_abs <= limit}
        row["provenance"] = provenance or {"kind": "synthetic_public_code"}
        if fmt == "mp3" and ffmpeg:
            ffmpeg_data = subprocess.run([ffmpeg, "-v", "error", "-i", str(encoded),
                "-f", "f32le", "-acodec", "pcm_f32le", "-"], capture_output=True,
                check=True, timeout=60).stdout
            other = np.frombuffer(ffmpeg_data, dtype="<f4").reshape(-1, decoded.shape[1])
            other = torch.from_numpy(other.T.copy())
            if decoded_rate != 24000:
                other = torchaudio.transforms.Resample(decoded_rate, 24000)(other)
            other = other.mean(0).numpy()
            if other.shape != expected.shape:
                raise ValueError(f"{name}: ffmpeg gapless geometry differs")
            floor = args.output / f"{name}-ffmpeg-floor"
            floor.mkdir(exist_ok=True)
            (floor / "pcm.f32").write_bytes(other.astype("<f4").tobytes())
            (floor / "official_pcm.f32").write_bytes(expected.astype("<f4").tobytes())
            (floor / "fixture.json").write_text(json.dumps({"kind": "decoder_noise_floor",
                "reference_decoder": f"libsndfile {sf.__libsndfile_version__}",
                "candidate_decoder": ffmpeg_version, "source_fixture": name,
                "provenance": row["provenance"]}, indent=2) + "\n")
            row["ffmpeg_floor_pcm_max_abs"] = float(np.max(np.abs(other-expected)))
        (destination / "fixture.json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row), flush=True); rows.append(row)

    for rate in [8000, 16000, 22050, 24000, 44100, 48000, 96000, 192000]:
        n = rate // 10 + 1
        t = np.arange(n) / rate
        signals = {"silence": np.zeros(n, np.float32),
                   "impulse": np.eye(1, n, n//2, dtype=np.float32)[0],
                   "tone": (0.5*np.sin(2*np.pi*440*t)).astype(np.float32),
                   "chirp": (0.5*np.sin(2*np.pi*(100*t+5000*t*t))).astype(np.float32),
                   "noise": np.random.default_rng(42).uniform(-0.5, 0.5, n).astype(np.float32)}
        for name, pcm in signals.items():
            for channels in [1, 2]:
                source = pcm if channels == 1 else np.stack([pcm, pcm*0.3], axis=1)
                decode(f"{rate}-{name}-{channels}", "wav", source, rate, "FLOAT")
    t = np.arange(24000)/24000
    tone = (0.4*np.sin(2*np.pi*440*t)).astype(np.float32)
    decode("decode-flac", "flac", tone, 24000, "PCM_16")
    decode("decode-mp3", "mp3", tone, 24000, "MPEG_LAYER_III")
    if args.public_speech:
        for pcm, provenance in public_speech(args.speech_source_dir or args.output / "public-source", ref):
            for rate in [16000, 44100, 48000]:
                source = torch.from_numpy(pcm.copy())
                if rate != 16000:
                    source = torchaudio.transforms.Resample(16000,rate)(source)
                source = source.numpy()
                for fmt, subtype in [("wav","FLOAT"),("flac","PCM_16"),("mp3","MPEG_LAYER_III")]:
                    decode(f"speech-{provenance['id']}-{rate}-{fmt}",fmt,source,rate,subtype,provenance)
    report = {"runtime": runtime, "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
              "binary_sha256": ref.digest(args.binary.read_bytes()), "ffmpeg": ffmpeg_version,
              "public_speech_source": PUBLIC_SPEECH if args.public_speech else None, "fixtures": rows,
              "passed": all(row["passed"] for row in rows)}
    (args.output / "decode-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("audio decode/resample qualification failed")


if __name__ == "__main__":
    main()
