#!/usr/bin/env python3
"""MiniMax M3.1 processor parity oracle (CPU).

Runs the checkpoint's own bundled processors (image_processor.py, video_processor.py,
processing_minimax.py) and MiniMax's serving sampler and frame sizing (sglang
minimax_m3_vl.py, copied below verbatim as pure functions) on deterministic inputs, and
writes what they produce for the Rust parity test:

  CUTEAFD_MINIMAX_PROCESSOR_PARITY=OUT cargo test -p cuteafd-loader minimax_vendor_parity -- --ignored

Images and frames are PNG files (lossless, so both sides see the same RGB8). Patches are
float32 little-endian files; the test compares them bit for bit. Needs torch, torchvision,
transformers 5.14.0 and pillow; a header-only snapshot is enough (no weights are read). The
release processors run on Transformers 5.4.0 through 5.14.0 only (earlier releases do not pass
`resample` to `_preprocess`, 5.19 no longer passes `do_convert_rgb` to the video one); their
uniform `sample_frames` is 4.57.1's, which the loader pins as its sampling reference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

SOURCES = {  # MiniMaxAI/MiniMax-M3.1-Flash 54e4e69 (blob-identical at 9e7ebac)
    "image_processor.py": "8218f98634c365608a98ea887bfebac37131ec2acdb8445188350490de7cf994",
    "video_processor.py": "ced0f8d85d73f8722fecb26033d2d91ecf4a4e5112c91ce1b1522ca7edd0943d",
    "processing_minimax.py": "0706ae8b93b3489a3b3bc7f48a5f29fb34b1449ebebb37586250e4d5fa8f5e35",
}


# sglang python/sglang/srt/multimodal/processors/minimax_m3_vl.py (sha256 8e1f9b30...),
# the two functions the serving path applies before the bundled processor.
def round_up(x, multiple):
    return (x + multiple - 1) // multiple * multiple


def get_hw_multiple_of(image_size, multiple, max_size=None):
    w, h = image_size
    if isinstance(max_size, int):
        ratio = 1.0
        max_dim = max(w, h)
        if max_dim > max_size:
            ratio = max_size / max_dim
        new_w = round_up(round(w * ratio), multiple)
        new_h = round_up(round(h * ratio), multiple)
        return new_w, new_h
    raise NotImplementedError("only the integer frame_max_size path serves decoded videos")


def _compute_sampled_frame_indices(total_frames, video_fps, fps, max_frames=None):
    if total_frames <= 0 or video_fps <= 0 or fps <= 0:
        return [0] if total_frames > 0 else []
    read_time_interval = 1.0 / fps
    eps = 1e-4
    indices = []
    prev_kept_ts = -float("inf")
    while True:
        if not indices:
            target_frame = 0
        else:
            target_ts = prev_kept_ts + read_time_interval - eps
            target_frame = math.ceil(target_ts * video_fps)
            target_frame = max(target_frame, indices[-1] + 1)
        if target_frame >= total_frames:
            break
        indices.append(target_frame)
        prev_kept_ts = target_frame / video_fps
    last_frame_idx = total_frames - 1
    last_ts = last_frame_idx / video_fps
    if indices and indices[-1] != last_frame_idx and last_ts - prev_kept_ts > eps:
        indices.append(last_frame_idx)
    if not indices:
        indices = [0]
    return indices


def load(snapshot: Path, name: str):
    path = snapshot / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCES[name]:
        raise ValueError(f"{name}: sha256 {digest} is not the pinned release source")
    spec = importlib.util.spec_from_file_location(f"minimax_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(height, width, seed):
    """Smooth gradients, hard edges and per-pixel noise: exercises every resampler tap."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((height, width), dtype=np.float64)
    base = np.stack([
        128 + 100 * np.sin(x / max(width, 1) * 6.3 + seed),
        128 + 100 * np.cos(y / max(height, 1) * 4.1 - seed),
        ((x // 9 + y // 7) % 2) * 180 + 30,
    ], axis=-1)
    noise = rng.integers(-40, 41, size=(height, width, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def formula(height, width, salt=0):
    """Integer pattern the Rust unit test regenerates to pin vendor hashes without fixtures."""
    y, x = np.indices((height, width), dtype=np.int64)
    c = np.arange(3, dtype=np.int64)
    value = x[..., None] * 37 + y[..., None] * 101 + c * 59 + (x * y % 97)[..., None] + salt * 13
    return (value % 256).astype(np.uint8)


def save_png(image, path):
    from PIL import Image
    Image.fromarray(image, "RGB").save(path, format="PNG", compress_level=6)


def write_patches(values, path):
    data = np.ascontiguousarray(values.numpy() if hasattr(values, "numpy") else values, dtype="<f4").tobytes()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers.video_utils import VideoMetadata
    if transformers.__version__ != "5.14.0":
        raise ValueError("pin transformers 5.14.0, the release processors' compatible framework")
    torch.set_num_threads(8)
    image_module = load(args.snapshot, "image_processor.py")
    video_module = load(args.snapshot, "video_processor.py")
    processing = load(args.snapshot, "processing_minimax.py")
    config = json.loads((args.snapshot / "preprocessor_config.json").read_text())
    kwargs = {k: v for k, v in config.items() if k not in ("processor_class", "auto_map")}
    images = image_module.MiniMaxM3VLImageProcessor(**kwargs)
    videos = video_module.MiniMaxM3VLVideoProcessor()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    # The release processor names neither its attributes nor their classes (newer
    # Transformers infer them from __init__); name the same three explicitly.
    cls = processing.MiniMaxVLProcessor
    cls.attributes = ["image_processor", "tokenizer", "video_processor"]
    cls.image_processor_class, cls.video_processor_class, cls.tokenizer_class = (
        "AutoImageProcessor", "AutoVideoProcessor", "AutoTokenizer")
    processor = cls(image_processor=images, tokenizer=tokenizer, video_processor=videos)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preprocessor_config.json").write_bytes((args.snapshot / "preprocessor_config.json").read_bytes())
    cases = {"transformers": transformers.__version__, "torch": torch.__version__,
             "sources": SOURCES, "images": [], "videos": [], "serving_sampler": [],
             "uniform_sampler": [], "serving_sizing": [], "smart_sizing": []}

    # Still images: shapes around the 672^2 budget, both aspect extremes, upscales,
    # 28-multiples that need no resize and multi-tile-sized inputs the processor shrinks.
    shapes = [(672, 672), (512, 512), (480, 640), (720, 1280), (1080, 1920), (1344, 1344),
              (672, 2016), (2016, 672), (100, 37), (28, 28), (1, 1), (5, 999), (333, 777),
              (448, 924), (1008, 1008), (3000, 17)]
    for i, (h, w) in enumerate(shapes):
        rgb = synthetic(h, w, i)
        name = f"image-{h}x{w}"
        save_png(rgb, args.out / f"{name}.png")
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        result = images(images=[tensor], return_tensors="pt")
        grid = [int(v) for v in result["image_grid_thw"][0]]
        sha = write_patches(result["pixel_values"], args.out / f"{name}.f32")
        cases["images"].append({"name": name, "height": h, "width": w, "grid": grid,
                                "tokens": grid[0] * grid[1] * grid[2] // 4, "patches_sha256": sha})

    # Formula inputs: their vendor patch hashes are pinned in the Rust unit tests.
    for h, w in [(1080, 1920), (333, 777), (100, 37)]:
        rgb = formula(h, w)
        result = images(images=[torch.from_numpy(rgb).permute(2, 0, 1).contiguous()], return_tensors="pt")
        data = np.ascontiguousarray(result["pixel_values"].numpy(), dtype="<f4").tobytes()
        cases.setdefault("formula_images", []).append({"height": h, "width": w,
            "grid": [int(v) for v in result["image_grid_thw"][0]], "patches_sha256": hashlib.sha256(data).hexdigest()})
    for count, h, w, resize in [(9, 56, 112, False), (5, 360, 640, True)]:
        frames = np.stack([formula(h, w, k) for k in range(count)])
        result = videos(videos=[torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()], do_resize=resize,
                        return_tensors="pt")
        data = np.ascontiguousarray(result["pixel_values_videos"].numpy(), dtype="<f4").tobytes()
        cases.setdefault("formula_videos", []).append({"frames": count, "height": h, "width": w,
            "do_resize": resize, "grid": [int(v) for v in result["video_grid_thw"][0]],
            "patches_sha256": hashlib.sha256(data).hexdigest()})

    # Frame lists through the bundled video processor: odd and even counts across the
    # eight-frame segment boundary, no resize (28-aligned) and the bundled smart resize.
    for count, (h, w), resize in [(1, (56, 84), False), (2, (56, 84), False), (3, (112, 56), False),
                                  (8, (84, 84), False), (9, (56, 112), False), (11, (28, 56), False),
                                  (5, (360, 640), True), (8, (720, 1280), True), (3, (90, 50), True)]:
        frames = [synthetic(h, w, 100 + count * 10 + k) for k in range(count)]
        name = f"video-{count}x{h}x{w}-{'smart' if resize else 'off'}"
        for k, frame in enumerate(frames):
            save_png(frame, args.out / f"{name}-{k}.png")
        tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
        # The real processor call (processing_minimax.py), with source timestamps at 4 fps;
        # the expanded prompt is read back from the token ids it produced.
        metadata = VideoMetadata(total_num_frames=count, fps=4.0, frames_indices=list(range(count)))
        result = processor(text=["]<]video[>["], videos=[tensor], videos_kwargs={"do_resize": resize,
                           "video_metadata": [metadata], "return_metadata": True}, return_tensors="pt")
        grid = [int(v) for v in result["video_grid_thw"][0]]
        sha = write_patches(result["pixel_values_videos"], args.out / f"{name}.f32")
        text = tokenizer.decode(result["input_ids"][0].tolist(), skip_special_tokens=False)
        cases["videos"].append({"name": name, "frames": count, "height": h, "width": w,
                                "do_resize": resize, "grid": grid, "patches_sha256": sha,
                                "source_fps": 4.0, "placeholder": text})

    for total, video_fps, fps in [(300, 30.0, 1.0), (300, 30000 / 1001, 1.0), (18, 5.0, 1.0), (16, 5.0, 1.0),
                                  (90, 30.0, 2.0), (10, 4.0, 8.0), (1, 30.0, 1.0), (1000, 23.976, 1.0),
                                  (751, 25.0, 0.5), (61, 60.0, 1.0), (240, 24.0, 3.0), (7, 2.0, 1.0),
                                  (3601, 29.97, 1.0), (45, 15.0, 1.5)]:
        cases["serving_sampler"].append({"total": total, "video_fps": video_fps, "fps": fps,
            "indices": _compute_sampled_frame_indices(total, video_fps, fps)})
    for total, video_fps, num_frames, fps in [(10, None, 4, None), (90, 30.0, None, None), (11, 2.5, None, 1.0),
                                             (100, 30.0, None, 2.0), (17, None, 16, None), (29, 7.0, None, 3.0),
                                             (1000, 29.97, None, 1.0), (13, None, 13, None)]:
        metadata = VideoMetadata(total_num_frames=total, fps=video_fps)
        indices = videos.sample_frames(metadata, num_frames=num_frames, fps=fps)
        cases["uniform_sampler"].append({"total": total, "video_fps": video_fps, "num_frames": num_frames,
                                         "fps": fps, "indices": [int(v) for v in indices]})
    for h, w in [(720, 1280), (1080, 1920), (480, 640), (1280, 720), (2160, 3840), (760, 1000), (28, 28),
                 (1, 1), (757, 3), (1001, 999), (756, 756), (300, 4000), (99, 100), (1081, 1919)]:
        new_w, new_h = get_hw_multiple_of((w, h), 28, 756)
        cases["serving_sizing"].append({"height": h, "width": w, "resized": [new_h, new_w]})
        try:
            resized = list(video_module.smart_resize(h, w, factor=28, min_pixels=videos.min_pixels,
                                                     max_pixels=videos.max_pixels))
        except ValueError:  # aspect ratio above 200
            resized = None
        cases["smart_sizing"].append({"height": h, "width": w, "resized": resized})
    (args.out / "cases.json").write_text(json.dumps(cases, indent=1) + "\n")
    print(json.dumps({"images": len(cases["images"]), "videos": len(cases["videos"]),
                      "out": str(args.out)}))


if __name__ == "__main__":
    sys.exit(main())
