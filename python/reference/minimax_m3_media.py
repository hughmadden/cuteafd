"""MiniMax M3.1 image preprocessing, the Python definitions (the Rust twin is
rust/crates/cuteafd-loader/src/media/minimax.rs).

python/tools/qualify/minimax_m3/processor-parity.py checks these against the
release's own image_processor.py and feeds the Rust parity test.
"""
from __future__ import annotations

import math

PATCH, MERGE, TEMPORAL = 14, 2, 2
FACTOR = PATCH * MERGE
MIN_PIXELS = 4 * 28 * 28
IMAGE_MAX_PIXELS = 451_584        # image_processor.py
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=IMAGE_MAX_PIXELS):
    """image_processor.py smart_resize (Python round is half to even)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def normalization_lut():
    """[3, 256] FP32: the fast processor fuses rescale into mean and std, (v - mean*255) / (std*255)."""
    import numpy as np
    values = np.arange(256, dtype=np.float32)
    return np.stack([(values - np.float32(m) * np.float32(255)) / (np.float32(s) * np.float32(255))
                     for m, s in zip(MEAN, STD)]).astype(np.float32)


def resize_rgb(rgb, height, width):
    """The processors' resize of HWC uint8: torch's antialiased bicubic on uint8."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    if rgb.shape[:2] == (height, width):
        return np.ascontiguousarray(rgb)
    tensor = torch.from_numpy(np.array(rgb, dtype=np.uint8, order="C")).permute(2, 0, 1)[None]
    out = F.interpolate(tensor, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    return np.ascontiguousarray(out[0].permute(1, 2, 0).numpy())


def patches(frames, lut=None):
    """Patch rows [t, bh, bw, mh, mw | C, temporal, ph, pw] from 2*t processor-sized HWC
    uint8 frames (an image repeats its frame for both temporal samples)."""
    import numpy as np
    frames = np.asarray(frames)
    lut = normalization_lut() if lut is None else lut
    count, h, w, _ = frames.shape
    if count % TEMPORAL or h % FACTOR or w % FACTOR:
        raise ValueError("frames must pair temporally and align to merged patches")
    t, gh, gw = count // TEMPORAL, h // PATCH, w // PATCH
    values = np.stack([lut[c][frames[..., c]] for c in range(3)], axis=1)  # [2t, C, H, W]
    values = values.reshape(t, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    values = values.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(values.reshape(t * gh * gw, 3 * TEMPORAL * PATCH * PATCH))
