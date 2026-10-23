"""CPU reference for the bundled MiniMax M3 vision tower, using checkpoint names.

Inputs are normalized, merge-window-ordered patches [N, 3 * 2 * 14 * 14].
Each grid is one image or video segment; attention covers that entire grid.
The convolution accumulates in F32; all subsequent activations and weights
are BF16. No checkpoint loader or device placement is implicit here.
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


def _cpu(*tensors):
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("MiniMax vision reference accepts CPU tensors only")


def rotary_factors(positions, dtype, head_dim=80, theta=10000.0):
    import torch
    axis_dim = 2 * ((head_dim // 3) // 2)
    positions = torch.as_tensor(positions, dtype=torch.float32, device="cpu")
    if positions.ndim != 2 or positions.shape[1] != 3 or axis_dim == 0:
        raise ValueError("RoPE requires T/H/W triples and at least six head channels")
    inv_freq = theta ** (-torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
    phase = (positions[:, :, None] * inv_freq).flatten(1)
    phase = torch.cat((phase, phase), dim=-1)
    return phase.cos().to(dtype), phase.sin().to(dtype)


def apply_rope(x, cos, sin):
    """Rotate all 78 leading channels as two halves; preserve the two-channel tail."""
    import torch
    _cpu(x, cos, sin)
    width = cos.shape[-1]
    first, second = x[..., :width // 2], x[..., width // 2:width]
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((x[..., :width] * cos[:, None, :] + rotated * sin[:, None, :],
                      x[..., width:]), dim=-1)


def merge_patches(features, merge_size=2):
    _cpu(features)
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
    _cpu(hidden, *weights.values())
    tokens, width = hidden.shape
    norm = _norm(hidden, weights, "layer_norm1", eps)
    q, k, v = (_linear(norm, weights, "self_attn." + name + "_proj")
               .reshape(tokens, heads, width // heads) for name in ("q", "k", "v"))
    cos, sin = rotary_factors(positions, hidden.dtype, width // heads)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scores = (q.transpose(0, 1) @ k.transpose(0, 1).transpose(-1, -2)) / math.sqrt(width // heads)
    probabilities = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
    attention = (probabilities @ v.transpose(0, 1)).transpose(0, 1).reshape(tokens, width)
    hidden = hidden + _linear(attention, weights, "self_attn.out_proj")
    norm = _norm(hidden, weights, "layer_norm2", eps)
    return hidden + _linear(F.gelu(_linear(norm, weights, "mlp.fc1"), approximate="none"), weights, "mlp.fc2")


def project_and_merge(hidden, weights, merge_size=2):
    import torch.nn.functional as F
    _cpu(hidden, *weights.values())
    projected = _linear(F.gelu(_linear(hidden, weights, "multi_modal_projector.linear_1"),
                               approximate="none"), weights, "multi_modal_projector.linear_2")
    merged = merge_patches(projected, merge_size)
    return _linear(F.gelu(_linear(merged, weights, "patch_merge_mlp.linear_1"),
                          approximate="none"), weights, "patch_merge_mlp.linear_2")


def forward(patches, grid_thw, weights, *, layers=32, project=True):
    """Run the tower using release tensor names; return useful comparison stages.

    ``layers=1, project=False`` exercises the real layer geometry economically.
    ``project=True`` also consumes the release's two projector MLPs. Multiple
    grids execute independently so attention never crosses image boundaries.
    """
    import torch
    import torch.nn.functional as F
    _cpu(patches, *weights.values())
    positions = position_ids(grid_thw)
    if patches.shape != (len(positions), 3 * 2 * 14 * 14):
        raise ValueError("patch tensor shape differs from the image grids")
    for name, value in weights.items():
        expected = torch.float32 if name == VISION_PREFIX + "embeddings.patch_embedding.weight" else torch.bfloat16
        if value.dtype != expected:
            raise ValueError(f"unexpected checkpoint dtype: {name}")
    with torch.no_grad():
        embedded = F.conv3d(patches.float().reshape(-1, 3, 2, 14, 14),
                            weights[VISION_PREFIX + "embeddings.patch_embedding.weight"],
                            stride=(2, 14, 14)).flatten(1).to(torch.bfloat16)
        hidden = _norm(embedded, weights, VISION_PREFIX + "pre_layrnorm", 1e-5)
        sizes = [t * h * w for t, h, w in grid_thw]
        for layer in range(layers):
            prefix = VISION_PREFIX + f"encoder.layers.{layer}."
            layer_weights = {key.removeprefix(prefix): value for key, value in weights.items() if key.startswith(prefix)}
            outputs, start = [], 0
            for size in sizes:
                outputs.append(encoder_layer(hidden[start:start + size], layer_weights, positions[start:start + size]))
                start += size
            hidden = torch.cat(outputs)
        output = project_and_merge(hidden, weights) if project else hidden
    return {"embeddings": embedded, "hidden_states": hidden, "features": output}
