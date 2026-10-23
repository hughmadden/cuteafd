"""Shared block dSpark rotary/attention numerical gate (CUDA PyTorch).

Usage: python native/tests/block_dspark_selftest.py LIBRARY
The library can contain only glmf_dspark.cu and shared draft_attention.cu.
"""
import argparse
import ctypes as C

import torch

P = C.c_void_p
I = C.c_int32
F = C.c_float


def bind(path):
    lib = C.CDLL(path)
    signatures = {
        "cuteafd_glmf_dspark_qk_rope": ([P] * 8 + [I] * 3 + [F] * 2 + [P], I),
        "cuteafd_block_dspark_qk_rope": ([P] * 9 + [I] * 4 + [F] * 3 + [P], I),
        "cuteafd_block_dspark_attention_workspace": ([I] * 6, C.c_uint64),
        "cuteafd_block_dspark_attention": ([P] * 10 + [I] * 8 + [F, P], I),
        "cuteafd_block_dspark_attention_paged": ([P] * 11 + [I] * 9 + [F, P], I),
        "cuteafd_glmf_dspark_attention_workspace": ([I] * 3, C.c_uint64),
        "cuteafd_glmf_dspark_attention": ([P] * 10 + [I] * 7 + [F, P], I),
    }
    for name, (arguments, result) in signatures.items():
        getattr(lib, name).argtypes = arguments
        getattr(lib, name).restype = result
    return lib


def rotary_check(lib):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    stream = torch.cuda.current_stream().cuda_stream
    positions = torch.tensor([0, 8191, 8192, 262143], device="cuda", dtype=torch.int64)
    slots = torch.tensor([2, -1, 0, 3], device="cuda", dtype=torch.int32)
    for dim, heads, kvh in ((64, 64, 64), (128, 40, 8), (128, 0, 8)):
        rows = positions.numel()
        source = torch.randn(rows, heads + 2 * kvh, dim, device="cuda").bfloat16()
        qnorm = (torch.randn(dim, device="cuda") * 0.1 + 1).bfloat16()
        knorm = (torch.randn(dim, device="cuda") * 0.1 + 1).bfloat16()
        q = torch.empty(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.full((rows, kvh, dim), -123, device="cuda", dtype=torch.bfloat16)
        v = torch.full_like(k, -123)
        if dim == 128:
            config = Qwen3Config(hidden_size=5120, num_attention_heads=40, num_key_value_heads=8,
                head_dim=128, max_position_embeddings=262144,
                rope_parameters=dict(rope_type="yarn", rope_theta=10000000., factor=32.,
                    original_max_position_embeddings=8192, beta_fast=32., beta_slow=1.))
            rotary = Qwen3RotaryEmbedding(config, device="cuda")
            inv_freq = rotary.inv_freq.contiguous()
            factor = rotary.attention_scaling
        else:
            inv_freq, factor = None, 1.
        pointers = [source.data_ptr(), qnorm.data_ptr(), knorm.data_ptr(), positions.data_ptr(), slots.data_ptr()]
        outputs = [q.data_ptr(), k.data_ptr(), v.data_ptr()]
        assert lib.cuteafd_block_dspark_qk_rope(*pointers,
            inv_freq.data_ptr() if inv_freq is not None else None, *outputs,
            rows, heads, kvh, dim, 10000., factor, 1e-6, stream) == 0
        if dim == 64:
            legacy = [torch.empty_like(q), torch.full_like(k, -123), torch.full_like(v, -123)]
            assert lib.cuteafd_glmf_dspark_qk_rope(*pointers, *[x.data_ptr() for x in legacy],
                rows, heads, kvh, 10000., 1e-6, stream) == 0
            torch.cuda.synchronize()
            assert all(torch.equal(a, b) for a, b in zip((q, k, v), legacy)), "legacy rotary changed"
        else:
            cos, sin = rotary(source[:, :1], positions[None])

            def expected(x, weight):
                unit = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
                normed = unit * weight
                half = torch.cat((-normed[..., dim // 2:], normed[..., :dim // 2]), dim=-1)
                return normed * cos[0, :, None] + half * sin[0, :, None]

            expected_q = expected(source[:, :heads], qnorm)
            expected_k = expected(source[:, heads:heads + kvh], knorm)
            torch.cuda.synchronize()
            # FP32 norm reductions and libdevice sin/cos can straddle a BF16
            # midpoint; no blanket FP32 equality is asserted for those ties.
            if heads:
                torch.testing.assert_close(q, expected_q, rtol=0.016, atol=0.016)
            for row, slot in enumerate(slots.tolist()):
                if slot >= 0:
                    torch.testing.assert_close(k[slot], expected_k[row], rtol=0.016, atol=0.016)
            print(f"rotary dim={dim} heads={heads}/{kvh} max_q_error="
                  f"{(q.float() - expected_q.float()).abs().max().item() if heads else 0:.6g}")
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0:
                assert torch.equal(v[slot], source[row, heads + kvh:]), "V changed"
        assert torch.all(k[1] == -123) and torch.all(v[1] == -123), "negative slot wrote cache"
    print("rotary legacy64 dispatch is byte-identical; YaRN128 boundary positions passed")


def attention_check(lib):
    stream = torch.cuda.current_stream().cuda_stream
    for dim, heads, kvh, block in ((64, 64, 64, 8), (128, 40, 8, 7)):
        for sequences in (1, 8):
            for context in (0, 33, 1024):
                for causal in (False, True):
                    capacity = max(context + 17, 64)
                    q = torch.randn(sequences, block, heads, dim, device="cuda").bfloat16()
                    k = torch.randn(sequences, block, kvh, dim, device="cuda").bfloat16()
                    v = torch.randn_like(k)
                    kc = torch.randn(sequences, capacity, kvh, dim, device="cuda").bfloat16()
                    vc = torch.randn_like(kc)
                    slots = torch.arange(sequences, device="cuda", dtype=torch.int32).flip(0).contiguous()
                    lengths = torch.tensor([max(0, context - i % 3) for i in range(sequences)],
                                           device="cuda", dtype=torch.int32)
                    ends = lengths.clone()
                    max_keys = context + block
                    size = lib.cuteafd_block_dspark_attention_workspace(16, heads, kvh, block, max_keys, dim)
                    assert size > 0
                    scratch = torch.full((size + 256,), 173, device="cuda", dtype=torch.uint8)
                    out = torch.full_like(q, float("nan"))
                    pointers = [x.data_ptr() for x in (q, k, v, kc, vc, slots, lengths, ends)]
                    geometry = [sequences, block, heads, kvh, capacity, max_keys, int(causal)]
                    assert lib.cuteafd_block_dspark_attention(*pointers, out.data_ptr(), scratch.data_ptr(),
                        *geometry, dim, dim ** -0.5, stream) == 0
                    # Fragmented pages must preserve the exact arithmetic of the
                    # slot-local route, including page boundaries and peaked scores.
                    width = (capacity + 63) // 64
                    page_ids = torch.randperm(sequences * width, device="cuda", dtype=torch.int32)
                    page_tables = page_ids.reshape(sequences, width)[slots.long()].contiguous()
                    kp = torch.zeros(sequences * width, 64, kvh, dim, device="cuda", dtype=torch.bfloat16)
                    vp = torch.zeros_like(kp)
                    for slot in range(sequences):
                        padded_k = torch.nn.functional.pad(kc[slot], (0, 0, 0, 0, 0, width * 64 - capacity))
                        padded_v = torch.nn.functional.pad(vc[slot], (0, 0, 0, 0, 0, width * 64 - capacity))
                        kp[page_ids.reshape(sequences, width)[slot].long()] = padded_k.reshape(width, 64, kvh, dim)
                        vp[page_ids.reshape(sequences, width)[slot].long()] = padded_v.reshape(width, 64, kvh, dim)
                    for amplitude in (1., 8.):
                        query = (q.float() * amplitude).bfloat16()
                        contiguous = torch.empty_like(q)
                        paged = torch.empty_like(q)
                        assert lib.cuteafd_block_dspark_attention(query.data_ptr(), *pointers[1:],
                            contiguous.data_ptr(), scratch.data_ptr(), *geometry, dim, dim ** -0.5, stream) == 0
                        paged_pointers = [query.data_ptr(), k.data_ptr(), v.data_ptr(), kp.data_ptr(), vp.data_ptr(),
                                          slots.data_ptr(), lengths.data_ptr(), ends.data_ptr(), page_tables.data_ptr()]
                        assert lib.cuteafd_block_dspark_attention_paged(*paged_pointers, paged.data_ptr(),
                            scratch.data_ptr(), *geometry, dim, width, dim ** -0.5, stream) == 0
                        torch.cuda.synchronize()
                        assert torch.equal(contiguous, paged), "fragmented pages changed attention bytes"
                    if dim == 64:
                        legacy_size = lib.cuteafd_glmf_dspark_attention_workspace(16, kvh, max_keys)
                        assert size == legacy_size
                        legacy = torch.empty_like(q)
                        assert lib.cuteafd_glmf_dspark_attention(*pointers, legacy.data_ptr(), scratch.data_ptr(),
                            *geometry, dim ** -0.5, stream) == 0
                        torch.cuda.synchronize()
                        assert torch.equal(out, legacy), "legacy attention changed"
                    reference = torch.empty_like(q, dtype=torch.float64)
                    for s in range(sequences):
                        count, slot = lengths[s].item(), slots[s].item()
                        keys = torch.cat((kc[slot, :count], k[s])).double()
                        values = torch.cat((vc[slot, :count], v[s])).double()
                        for h in range(heads):
                            kvhead = h // (heads // kvh)
                            scores = q[s, :, h].double() @ keys[:, kvhead].T * dim ** -0.5
                            if causal:
                                mask = torch.arange(block, device="cuda")[None, :] > torch.arange(block, device="cuda")[:, None]
                                scores[:, count:].masked_fill_(mask, -torch.inf)
                            reference[s, :, h] = scores.softmax(-1) @ values[:, kvhead]
                    delta = out.double() - reference
                    maximum = delta.abs().max().item()
                    rms = (delta.square().mean() / reference.square().mean().clamp_min(1e-20)).sqrt().item()
                    assert torch.all(scratch[size:] == 173), "workspace overrun"
                    assert torch.isfinite(out).all() and maximum < 0.025 and rms < 0.01, (maximum, rms)
                    print(f"attention dim={dim} heads={heads}/{kvh} block={block} sequences={sequences} "
                          f"context={context} causal={causal} max_abs={maximum:.4g} rel_rms={rms:.4g}")
    print("attention legacy64 bytes and 35-query/two-slice FP64 gate passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    args = parser.parse_args()
    torch.manual_seed(276)
    lib = bind(args.library)
    rotary_check(lib)
    attention_check(lib)
    print("block dSpark kernels: all checks passed")


if __name__ == "__main__":
    main()
