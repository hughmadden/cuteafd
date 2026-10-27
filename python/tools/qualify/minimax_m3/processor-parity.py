#!/usr/bin/env python3
"""MiniMax M3.1 processor parity oracle (CPU).

Runs the checkpoint's own bundled image processor (image_processor.py) on deterministic
inputs, checks that the Python preprocessing in python/reference/minimax_m3_media.py
reproduces it bit for bit, and writes what it produces for the Rust parity test:

  CUTEAFD_MINIMAX_PROCESSOR_PARITY=OUT cargo test -p cuteafd-loader minimax_vendor_parity -- --ignored

Images are PNG files (lossless, so both sides see the same RGB8). Patches are float32
little-endian files; the test compares them bit for bit. Needs torch, torchvision,
transformers 5.14.0 and pillow; a header-only snapshot is enough (no weights are read). The
release processor runs on Transformers 5.4.0 through 5.14.0 only (earlier releases do not pass
`resample` to `_preprocess`).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python/reference"))
import minimax_m3_media as media

SOURCES = {  # MiniMaxAI/MiniMax-M3.1-Flash 54e4e69 (blob-identical at 9e7ebac)
    "image_processor.py": "8218f98634c365608a98ea887bfebac37131ec2acdb8445188350490de7cf994",
    "video_processor.py": "ced0f8d85d73f8722fecb26033d2d91ecf4a4e5112c91ce1b1522ca7edd0943d",
    "processing_minimax.py": "0706ae8b93b3489a3b3bc7f48a5f29fb34b1449ebebb37586250e4d5fa8f5e35",
}


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
    if transformers.__version__ != "5.14.0":
        raise ValueError("pin transformers 5.14.0, the release processors' compatible framework")
    torch.set_num_threads(8)
    image_module = load(args.snapshot, "image_processor.py")
    config = json.loads((args.snapshot / "preprocessor_config.json").read_text())
    kwargs = {k: v for k, v in config.items() if k not in ("processor_class", "auto_map")}
    images = image_module.MiniMaxM3VLImageProcessor(**kwargs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preprocessor_config.json").write_bytes((args.snapshot / "preprocessor_config.json").read_bytes())
    cases = {"transformers": transformers.__version__, "torch": torch.__version__,
             "sources": SOURCES, "images": []}

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
        size = media.smart_resize(h, w)
        if size != image_module.smart_resize(h, w) or [1, size[0] // 14, size[1] // 14] != grid:
            raise ValueError(f"{name}: minimax_m3_media.smart_resize differs from the release")
        frame = media.resize_rgb(rgb, *size)
        if not np.array_equal(media.patches(np.stack([frame, frame])), result["pixel_values"].numpy()):
            raise ValueError(f"{name}: minimax_m3_media patches differ from the release processor")
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
    (args.out / "cases.json").write_text(json.dumps(cases, indent=1) + "\n")
    print(json.dumps({"images": len(cases["images"]), "out": str(args.out)}))


if __name__ == "__main__":
    sys.exit(main())
