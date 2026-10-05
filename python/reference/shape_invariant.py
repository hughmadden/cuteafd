"""Fixed row geometry for reference arithmetic, never for the serving engine."""
from __future__ import annotations

import torch
import torch.nn.functional as F

ROWS = 128
_linear = F.linear
_einsum = torch.einsum


def linear(x, weight, bias=None):
    """Keep GEMM M fixed even when a window or an expert's token count changes."""
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] == 0:
        return _linear(x, weight, bias)
    result = []
    for start in range(0, flat.shape[0], ROWS):
        chunk = flat[start:start + ROWS]
        count = chunk.shape[0]
        if count != ROWS:
            padded = chunk.new_zeros((ROWS, chunk.shape[-1]))
            padded[:count].copy_(chunk)
            chunk = padded
        result.append(_linear(chunk, weight, bias)[:count])
    return torch.cat(result, dim=0).reshape(*x.shape[:-1], weight.shape[0])


def einsum(equation, *operands):
    """Fix query-row batch geometry for V4.1 grouped projection/attention."""
    axes = {"bsgd,grd->bsgr": (0,), "bmhd,bmtd->bmht": (0, 1),
            "bmht,bmtd->bmhd": (0, 1)}.get(equation)
    if axes is None:
        return _einsum(equation, *operands)
    count = operands[0].shape[1]
    if count == 0:
        return _einsum(equation, *operands)
    result = []
    for start in range(0, count, ROWS):
        width = min(ROWS, count - start)
        args = list(operands)
        for axis in axes:
            chunk = args[axis][:, start:start + ROWS]
            if width != ROWS:
                shape = list(chunk.shape)
                shape[1] = ROWS
                padded = chunk.new_zeros(shape)
                padded[:, :width].copy_(chunk)
                chunk = padded
            args[axis] = chunk
        result.append(_einsum(equation, *args)[:, :width])
    return torch.cat(result, dim=1)


def qualify(a, manifest, execute):
    """Nested layer-major probes must not change the outer family's dtype."""
    if getattr(a, "_prefix_probe", False):
        return None
    from fidelity_windows import qualify_prefix
    dtype = torch.get_default_dtype()
    try:
        proof = qualify_prefix(a, manifest, execute)
        from fidelity_windows import canonical
        (a.out / "prefix-qualification.json").write_bytes(canonical(proof) + b"\n")
        return proof
    finally:
        torch.set_default_dtype(dtype)


def bounded_eager(function, rows=1024):
    """Call unchanged eager arithmetic per query block, retaining all keys.

    Golden runners discard attention weights; returning them would reconstruct
    the quadratic allocation that this wrapper is intended to avoid.
    """
    def forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if module.training or dropout != 0:
            raise ValueError("bounded reference attention requires inference without dropout")
        outputs = []
        for start in range(0, query.shape[-2], rows):
            end = min(start + rows, query.shape[-2])
            mask = attention_mask
            if mask is not None and mask.shape[-2] != 1:
                if mask.shape[-2] != query.shape[-2]:
                    raise ValueError("attention mask query coverage differs")
                mask = mask[..., start:end, :]
            sliced = dict(kwargs)
            for name in ("selected_kv", "selected_valid"):
                if sliced.get(name) is not None:
                    sliced[name] = sliced[name][:, start:end]
            output, weights = function(module, query[..., start:end, :], key, value, mask,
                                       scaling=scaling, dropout=dropout, **sliced)
            outputs.append(output)
            del weights
        return torch.cat(outputs, dim=1), None
    return forward


def bounded_sparse(function, rows=128):
    """Bound official-checkpoint sparse hooks before their KV gather allocation."""
    def forward(q, kv, attn_sink, topk_idxs, softmax_scale):
        outputs = []
        for start in range(0, q.shape[1], rows):
            outputs.append(function(q[:, start:start + rows], kv, attn_sink,
                                    topk_idxs[:, start:start + rows], softmax_scale))
        return torch.cat(outputs, dim=1)
    return forward


def install_eager(module):
    """Bound the family's eager fallback without replacing its math or mask."""
    if hasattr(module, "eager_attention_forward"):
        module.eager_attention_forward = bounded_eager(module.eager_attention_forward)


def install():
    """Also cover checkpoint modules and HC code that call F.linear directly."""
    F.linear = linear
    torch.einsum = einsum
