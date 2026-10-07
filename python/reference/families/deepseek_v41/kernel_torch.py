"""Torch emulation of the official DeepSeek V4.1 inference/kernel.py (tilelang).

Installed as the `kernel` module before importing the official model.py, so the
reference runs on any CUDA torch without tilelang. Quantization is exact (the
same power-of-two or E4M3 scale rules and clamps, FP8/FP4 rounding by torch
casts); GEMMs dequantize and multiply in FP32, which differs from the tiled
kernels only in summation order. Signatures follow V4.1's kernel.py (32- or
128-element blocks), which differ from V4's.
"""
from __future__ import annotations

from typing import Optional

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _pow2_ceil(value: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(value)) as the kernel's IEEE bit trick computes it."""
    bits = value.float().contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    exponent = exponent + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
    return torch.ldexp(torch.ones_like(value, dtype=torch.float32), exponent)


def _groups(x: torch.Tensor, block: int) -> torch.Tensor:
    return x.float().reshape(*x.shape[:-1], x.shape[-1] // block, block)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
              scale_dtype: torch.dtype = torch.float32, inplace: bool = False):
    g = _groups(x, block_size)
    amax = g.abs().amax(-1).clamp_min(1e-4)
    scale = _pow2_ceil(amax / FP8_MAX) if scale_fmt is not None else amax / FP8_MAX
    q = (g / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * scale.unsqueeze(-1)).reshape(x.shape).to(x.dtype))
        return x
    return q.reshape(x.shape), scale.to(scale_dtype)


def _e2m1_round(v: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the e2m1 grid (the float -> fp4 cast)."""
    grid = _E2M1.to(v.device)
    mag = v.abs().clamp(max=FP4_MAX)
    mids = (grid[1:] + grid[:-1]) / 2
    idx = torch.bucketize(mag, mids)
    tie = (idx > 0) & (mag == mids[(idx - 1).clamp_min(0)]) & (idx % 2 == 1)
    idx = idx - tie.to(idx.dtype)
    return torch.copysign(grid[idx], v)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False,
                  scale_dtype: torch.dtype = torch.float8_e8m0fnu):
    g = _groups(x, block_size)
    amax = g.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        # Compressed KV: E4M3 scales, an all-zero group keeps a nonzero one.
        scale = (amax.clamp_min(6 * 2.0 ** -9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        scale = _pow2_ceil(amax.clamp_min(6 * 2.0 ** -126) / FP4_MAX)
    q = _e2m1_round((g / scale.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((q * scale.unsqueeze(-1)).reshape(x.shape).to(x.dtype))
        return x
    return q.reshape(x.shape), scale.to(scale_dtype)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """[N, K/2] packed e2m1 (low nibble first) -> [N, K] float32."""
    raw = packed.contiguous().view(torch.uint8)
    lo, hi = raw & 0x0F, raw >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(*raw.shape[:-1], raw.shape[-1] * 2).long()
    grid = _E2M1.to(raw.device)
    return torch.where(codes & 8 != 0, -grid[codes & 7], grid[codes & 7])


def _dequant_act(a: torch.Tensor, a_s: torch.Tensor) -> torch.Tensor:
    k = a.shape[-1]
    groups = a_s.shape[-1]
    return (a.float().reshape(*a.shape[:-1], groups, k // groups) * a_s.float().unsqueeze(-1)).reshape(a.shape)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size: int = 128):
    n, k = b.shape
    bs = b_s.float().repeat_interleave(block_size, 0)[:n].repeat_interleave(block_size, 1)[:, :k]
    from shape_invariant import linear
    out = linear(_dequant_act(a, a_s), b.float() * bs)
    return out.to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size: int = 128):
    w = unpack_fp4(b) * b_s.float().repeat_interleave(32, 1)
    from shape_invariant import linear
    out = linear(_dequant_act(a, a_s), w[:, : a.shape[-1]])
    return out.to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    gathered = torch.gather(kv.unsqueeze(1).expand(b, m, *kv.shape[1:]), 2,
                            idx.clamp_min(0).unsqueeze(-1).expand(b, m, idx.shape[-1], d))
    gathered = gathered * valid.unsqueeze(-1)
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    # The kernel's finite lower bound: a row without a valid index gives zeros.
    peak = scores.amax(-1, keepdim=True).clamp_min(-1e30)
    probs = torch.exp(scores - peak)
    denom = probs.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - peak.squeeze(-1))
    out = torch.einsum("bmht,bmtd->bmhd", probs.to(torch.bfloat16).float(), gathered.float())
    return (out / denom.unsqueeze(-1)).to(torch.bfloat16)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).reshape(*m.shape[:-1], hc, hc)
    comb = torch.softmax(comb, -1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
