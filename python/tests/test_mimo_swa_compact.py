"""Opt-in native compact SWA records, direct readers and byte-preserving commits."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import os

import pytest

if os.environ.get("CUTEAFD_MIMO_SWA_GPU") != "1":
    pytest.skip("requires explicit MiMo SWA CUDA gate", allow_module_level=True)

torch = pytest.importorskip("torch")


def geometry(model, ranks):
    from b12x.integration.cuteafd._common import MIMO_V2_FLASH, MIMO_V26_PRO
    g = MIMO_V2_FLASH if model == "flash" else MIMO_V26_PRO
    return replace(g, heads=g.heads // ranks, full_kv_heads=g.full_kv_heads // ranks,
                   swa_kv_heads=g.swa_kv_heads // ranks)


def pack(g, records, fmt):
    from b12x.integration.cuteafd.mimo_attention import KV_GROUPS
    group, heads, rows = KV_GROUPS[fmt], g.swa_kv_heads, len(records)
    codes, scales = [], []
    for data in (records[:, :heads * 192], records[:, heads * 192:]):
        values = data.float().reshape(rows, heads, -1, group)
        amax = values.abs().amax(-1, keepdim=True)
        scale = torch.where(amax > 0, amax / torch.full_like(amax, 448 if fmt == "fp8" else 127),
                            torch.ones_like(amax))
        normalized = values / scale
        code = normalized.to(torch.float8_e4m3fn) if fmt == "fp8" else normalized.round().clamp(-127, 127).to(torch.int8)
        codes.append(code.reshape(rows, -1).view(torch.uint8))
        scales.append(scale.reshape(rows, -1))
    return torch.cat((*codes, torch.cat(scales, 1).contiguous().view(torch.uint8)), 1)


def unpack(g, records, fmt):
    from b12x.integration.cuteafd.mimo_attention import KV_GROUPS
    heads, group, rows = g.swa_kv_heads, KV_GROUPS[fmt], len(records)
    dtype = torch.float8_e4m3fn if fmt == "fp8" else torch.int8
    codes = records[:, :heads * 320].contiguous().view(dtype).float()
    scales = records[:, heads * 320:].contiguous().view(torch.float32)
    keys = (codes[:, :heads * 192].reshape(rows, heads, 192 // group, group) *
            scales[:, :heads * 192 // group].reshape(rows, heads, 192 // group, 1)).bfloat16()
    values = (codes[:, heads * 192:].reshape(rows, heads, 128 // group, group) *
              scales[:, heads * 192 // group:].reshape(rows, heads, 128 // group, 1)).bfloat16()
    return torch.cat((keys.flatten(1), values.flatten(1)), 1)


@lru_cache(None)
def post(model, ranks, fmt):
    from b12x.integration.cuteafd._common import Operand, Scalar, compile_program
    from b12x.integration.cuteafd._mimo_kernels import MimoQkvRope
    from b12x.integration.cuteafd.mimo_attention import KV_GROUPS
    g = geometry(model, ranks)
    launch = MimoQkvRope(heads=g.heads, kv_heads=g.swa_kv_heads, v_scale=g.v_scale,
                         k_stride=g.qkv_k_stride, kv8={"bf16": None, "fp8": "e4m3", "int8": "s8"}[fmt],
                         kv_group=KV_GROUPS.get(fmt, 64))
    return compile_program(launch, name="mimo_swa_post_probe", operands=(
        Operand("qkv", torch.bfloat16, "[rows,W]"), Operand("positions", torch.int64, "[rows]", align=8),
        Operand("cos_sin", torch.float32, "[P,64]", align=4),
        Operand("slots", torch.int64, "[rows]", align=8), Operand("query", torch.bfloat16, "[rows,N,D]", "out"),
        Operand("cache", torch.bfloat16 if fmt == "bf16" else torch.uint8, "[records,R]", "inout")),
        scalars=(Scalar("rows"),), key=(model, ranks, fmt))


@pytest.mark.parametrize("model,ranks", [("flash", 1), ("flash", 2), ("pro", 1), ("pro", 2)])
@pytest.mark.parametrize("fmt", ["int8", "fp8"])
def test_swa_producer_exact_bytes_and_inactive_append(model, ranks, fmt):
    from b12x.integration.cuteafd.mimo_attention import record_bytes
    g, rows = geometry(model, ranks), 5
    torch.manual_seed(71)
    qkv = torch.randn(rows, g.qkv_width("swa"), device="cuda").bfloat16()
    qkv[-1].zero_()
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)
    angles = torch.randn(rows, 32, device="cuda")
    rope = torch.cat((angles.cos(), angles.sin()), 1)
    slots = torch.tensor([63, 64, -1, 255, 256], device="cuda", dtype=torch.int64)
    bf = torch.zeros(258, g.record_elems("swa"), device="cuda", dtype=torch.bfloat16)
    compact = torch.zeros(258, record_bytes(g, "swa", fmt), device="cuda", dtype=torch.uint8)
    queries = [torch.empty(rows, g.heads, 192, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    post(model, ranks, "bf16").launch(qkv, positions, rope, slots, queries[0], bf, scalars=[rows])
    post(model, ranks, fmt).launch(qkv, positions, rope, slots, queries[1], compact, scalars=[rows])
    torch.cuda.synchronize()
    live = slots[slots >= 0]
    assert torch.equal(queries[0], queries[1])
    assert torch.equal(compact[live], pack(g, bf[live], fmt))
    untouched = torch.ones(258, device="cuda", dtype=torch.bool)
    untouched[live] = False
    assert torch.count_nonzero(compact[untouched]) == 0
    before = compact.clone()
    slots.fill_(-1)
    slots[0] = 257
    post(model, ranks, fmt).launch(qkv, positions, rope, slots, queries[1], compact, scalars=[rows])
    torch.cuda.synchronize()
    assert torch.equal(before[:257], compact[:257])


@lru_cache(None)
def attention(model, ranks, fmt, route):
    from b12x.integration.cuteafd.mimo_attention import compile_mimo_attention_aot
    return compile_mimo_attention_aot(geometry(model, ranks), kind="swa", route=route,
                                       max_rows=64 if route == "decode" else 4096, kv=fmt)


@pytest.mark.parametrize("model,ranks", [("flash", 1), ("flash", 2), ("pro", 1), ("pro", 2)])
@pytest.mark.parametrize("fmt", ["int8", "fp8"])
@pytest.mark.parametrize("route", ["decode", "prefill"])
@pytest.mark.parametrize("peaked", [False, True])
def test_swa_dequantized_reader_wrap_masks_and_graph(model, ranks, fmt, route, peaked):
    g = geometry(model, ranks)
    torch.manual_seed(199)
    # Decode mixes sequence starts and ring wraps; prefill crosses a ring boundary
    # with a partial final query tile. Padding queries must not become live rows.
    starts = [(0, 3), (127, 2), (255, 3), (513, 2)] if route == "decode" else [(255, 19)]
    rows = sum(n for _, n in starts)
    positions, slots, firsts, histories, step_rows = [], [], [], [], []
    row = 0
    for seq, (start, n) in enumerate(starts):
        history = torch.randn(256, g.record_elems("swa"), device="cuda").bfloat16()
        history *= torch.logspace(-2, 1, 256, device="cuda")[:, None]
        histories.append(history)
        step_rows.append(torch.randn(n, g.record_elems("swa"), device="cuda").bfloat16())
        positions.extend(range(start, start + n))
        slots.extend(seq * 256 + p % 256 for p in range(start, start + n))
        firsts.extend([row] * n)
        row += n
    packed_ring, packed_step = pack(g, torch.cat(histories), fmt), pack(g, torch.cat(step_rows), fmt)
    decoded_ring, decoded_step = unpack(g, packed_ring, fmt), unpack(g, packed_step, fmt)
    query = (torch.randn(rows + 3, g.heads, 192, device="cuda") * (9 if peaked else .3)).bfloat16()
    pos = torch.tensor(positions, device="cuda", dtype=torch.int64)
    ring_slots = torch.tensor(slots, device="cuda", dtype=torch.int64)
    seq_first = torch.tensor(firsts, device="cuda", dtype=torch.int32)
    sinks = torch.randn(g.heads, device="cuda").bfloat16()
    compact_ring, bf_ring = packed_ring.clone(), decoded_ring.clone()
    outputs = [torch.full((rows + 3, g.heads, 128), 17, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    scratch = torch.full((2048,), 0xA5, device="cuda", dtype=torch.uint8)

    def run(index, fmt_name, step, ring):
        attention(model, ranks, fmt_name, route).launch(query, step, ring, pos, ring_slots, seq_first,
                                                       sinks, outputs[index], scratch[:1024], scalars=[rows])

    run(0, fmt, packed_step, compact_ring)
    run(1, "bf16", decoded_step, bf_ring)
    torch.cuda.synchronize()
    assert torch.equal(outputs[0][:rows], outputs[1][:rows]), "direct reader differs from identical-geometry dequantized BF16"
    assert torch.all(outputs[0][rows:] == 17) and torch.all(outputs[1][rows:] == 17)
    expected_ring = packed_ring.clone()
    expected_ring[ring_slots] = packed_step
    assert torch.equal(compact_ring, expected_ring), "ring commit must copy codes AND scales unchanged"
    assert torch.equal(bf_ring, unpack(g, expected_ring, fmt))

    # Independent FP64 softmax also checks the mask, learned sink and online
    # rescaling; peaked scores span several nats rather than hiding under a flat softmax.
    for r, p in enumerate(positions):
        first = firsts[r]
        start = positions[first]
        seq = slots[r] // 256
        indices = range(max(0, p - g.window + 1), p + 1)
        records = torch.stack([decoded_step[first + k - start] if k >= start else decoded_ring[seq * 256 + k % 256]
                               for k in indices]).double()
        keys = records[:, :g.swa_kv_heads * 192].reshape(-1, g.swa_kv_heads, 192).repeat_interleave(g.heads // g.swa_kv_heads, 1)
        values = records[:, g.swa_kv_heads * 192:].reshape(-1, g.swa_kv_heads, 128).repeat_interleave(g.heads // g.swa_kv_heads, 1)
        scores = torch.einsum("hd,thd->ht", query[r].double(), keys) * g.softmax_scale
        if peaked and p >= 127:
            assert (scores.max(-1).values - scores.min(-1).values).max() > 5
        probs = torch.cat((scores, sinks.double()[:, None]), 1).softmax(-1)[:, :-1]
        oracle = torch.einsum("ht,thd->hd", probs, values)
        error = (outputs[0][r].double() - oracle).norm() / oracle.norm()
        assert error < .01, (r, error.item())
    saved = outputs[0].clone()
    compact_ring.copy_(packed_ring)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(0, fmt, packed_step, compact_ring)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(outputs[0], saved)
    assert torch.equal(compact_ring, expected_ring)
    assert torch.all(scratch[1024:] == 0xA5), "SWA scratch overrun"
