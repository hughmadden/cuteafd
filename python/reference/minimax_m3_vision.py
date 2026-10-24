"""Reference for the bundled MiniMax M3 vision tower, using checkpoint names.

Inputs are normalized, merge-window-ordered patches [N, 3 * 2 * 14 * 14].
Each grid is one image or video segment; attention covers that entire grid.
The convolution accumulates in F32; all subsequent activations and weights are
BF16 (the release numerics), or F32 throughout for a qualification yardstick.
All tensors share one device, which the caller chooses; no checkpoint loader is
implicit here.
"""
from __future__ import annotations

import math

VISION_PREFIX = "vision_tower.vision_model."
MODEL_REVISION = "54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e"
MODELING_SHA256 = "1c755515c8aefdbc6f9145ae8f051e43907e21022d9644e81afe47cd2e37b58d"
POSITION_SOURCE_SHA256 = "bcecd5a92b3266b9926272a549d2b1a0f1fe7646c698c0fa19bd96f976085356"


def position_ids(grid_thw, merge_size=2):
    """Return T/H/W triples in block-major spatial merge order."""
    positions = []
    if merge_size < 1:
        raise ValueError("merge size must be positive")
    for t, h, w in grid_thw:
        if min(t, h, w) < 1 or h % merge_size or w % merge_size:
            raise ValueError("grid must contain complete spatial merge windows")
        positions.extend((frame, row + dy, col + dx)
                         for frame in range(t)
                         for row in range(0, h, merge_size)
                         for col in range(0, w, merge_size)
                         for dy in range(merge_size)
                         for dx in range(merge_size))
    return positions


def _one_device(*tensors):
    if len({t.device for t in tensors}) > 1:
        raise ValueError("MiniMax vision reference tensors must share one device")


def rotary_factors(positions, dtype, head_dim=80, theta=10000.0, device="cpu"):
    import torch
    axis_dim = 2 * ((head_dim // 3) // 2)
    positions = torch.as_tensor(positions, dtype=torch.float32, device=device)
    if positions.ndim != 2 or positions.shape[1] != 3 or axis_dim == 0:
        raise ValueError("RoPE requires T/H/W triples and at least six head channels")
    inv_freq = theta ** (-torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim)
    phase = (positions[:, :, None] * inv_freq).flatten(1)
    phase = torch.cat((phase, phase), dim=-1)
    return phase.cos().to(dtype), phase.sin().to(dtype)


def apply_rope(x, cos, sin):
    """Rotate all 78 leading channels as two halves; preserve the two-channel tail.

    MiniMax's serving applier (sglang ``_minimax_rope_applier``) rotates in FP32 with
    FP32 cos/sin and rounds the result once to the activation dtype.
    """
    import torch
    _one_device(x, cos, sin)
    width = cos.shape[-1]
    rot = x[..., :width].float()
    first, second = rot[..., :width // 2], rot[..., width // 2:]
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat(((rot * cos.float()[:, None, :] + rotated * sin.float()[:, None, :]).to(x.dtype),
                      x[..., width:]), dim=-1)


def merge_patches(features, merge_size=2):
    if features.ndim != 2 or features.shape[0] % (merge_size ** 2):
        raise ValueError("patch features must contain complete merge windows")
    return features.reshape(features.shape[0] // merge_size ** 2, -1)


def _linear(x, weights, name):
    import torch.nn.functional as F
    return F.linear(x, weights[name + ".weight"], weights[name + ".bias"])


def _norm(x, weights, name, eps):
    import torch.nn.functional as F
    return F.layer_norm(x, (x.shape[-1],), weights[name + ".weight"], weights[name + ".bias"], eps)


def encoder_layer(hidden, weights, positions, *, heads=16, eps=1e-5):
    """One noncausal CLIP layer; keys are relative to encoder.layers.<i>."""
    import torch
    import torch.nn.functional as F
    _one_device(hidden, *weights.values())
    tokens, width = hidden.shape
    norm = _norm(hidden, weights, "layer_norm1", eps)
    q, k, v = (_linear(norm, weights, "self_attn." + name + "_proj")
               .reshape(tokens, heads, width // heads) for name in ("q", "k", "v"))
    cos, sin = rotary_factors(positions, torch.float32, width // heads, device=hidden.device)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scores = (q.transpose(0, 1) @ k.transpose(0, 1).transpose(-1, -2)) / math.sqrt(width // heads)
    probabilities = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
    attention = (probabilities @ v.transpose(0, 1)).transpose(0, 1).reshape(tokens, width)
    hidden = hidden + _linear(attention, weights, "self_attn.out_proj")
    norm = _norm(hidden, weights, "layer_norm2", eps)
    return hidden + _linear(F.gelu(_linear(norm, weights, "mlp.fc1"), approximate="none"), weights, "mlp.fc2")


def project_and_merge(hidden, weights, merge_size=2):
    import torch.nn.functional as F
    _one_device(hidden, *weights.values())
    projected = _linear(F.gelu(_linear(hidden, weights, "multi_modal_projector.linear_1"),
                               approximate="none"), weights, "multi_modal_projector.linear_2")
    merged = merge_patches(projected, merge_size)
    return _linear(F.gelu(_linear(merged, weights, "patch_merge_mlp.linear_1"),
                          approximate="none"), weights, "patch_merge_mlp.linear_2")


def forward(patches, grid_thw, weights, *, layers=32, project=True, dtype=None, observe=None):
    """Run the tower using release tensor names; return useful comparison stages.

    ``layers=1, project=False`` exercises the real layer geometry economically.
    ``project=True`` also consumes the release's two projector MLPs. Multiple
    grids execute independently so attention never crosses image boundaries.
    ``dtype`` is the activation and weight type: BF16 (the release) by default,
    F32 for a yardstick. ``observe(stage, hidden)`` sees the pre-LayerNorm output
    as stage 0 and each layer's output as stage ``layer + 1``.
    """
    import torch
    import torch.nn.functional as F
    dtype = dtype or torch.bfloat16
    _one_device(patches, *weights.values())
    positions = position_ids(grid_thw)
    if patches.shape != (len(positions), 3 * 2 * 14 * 14):
        raise ValueError("patch tensor shape differs from the image grids")
    for name, value in weights.items():
        expected = torch.float32 if name == VISION_PREFIX + "embeddings.patch_embedding.weight" else dtype
        if value.dtype != expected:
            raise ValueError(f"unexpected checkpoint dtype: {name}")
    with torch.no_grad():
        embedded = F.conv3d(patches.float().reshape(-1, 3, 2, 14, 14),
                            weights[VISION_PREFIX + "embeddings.patch_embedding.weight"],
                            stride=(2, 14, 14)).flatten(1).to(dtype)
        hidden = _norm(embedded, weights, VISION_PREFIX + "pre_layrnorm", 1e-5)
        if observe:
            observe(0, hidden)
        sizes = [t * h * w for t, h, w in grid_thw]
        for layer in range(layers):
            prefix = VISION_PREFIX + f"encoder.layers.{layer}."
            layer_weights = {key.removeprefix(prefix): value for key, value in weights.items() if key.startswith(prefix)}
            outputs, start = [], 0
            for size in sizes:
                outputs.append(encoder_layer(hidden[start:start + size], layer_weights, positions[start:start + size]))
                start += size
            hidden = torch.cat(outputs)
            if observe:
                observe(layer + 1, hidden)
        output = project_and_merge(hidden, weights) if project else hidden
    return {"embeddings": embedded, "hidden_states": hidden, "features": output}
