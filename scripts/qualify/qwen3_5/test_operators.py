"""GPU operator gates against the pinned, unmodified Transformers reference.

These synthetic layer checks do not establish full-model golden or prefix parity.
Run in the matching coordinator container with the candidate SparkInfer pin.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_SHA256 = "7788020ca357c6c8fb6e750fd847dd80ac77625902213b47f4949c8085344a58"


@pytest.fixture(scope="module")
def reference():
    source = ROOT / "third_party/transformers/src"
    path = source / "transformers/models/qwen3_5/modeling_qwen3_5.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256
    sys.path.insert(0, str(source))
    from transformers.models.qwen3_5 import modeling_qwen3_5
    assert Path(modeling_qwen3_5.__file__).resolve() == path.resolve()
    assert torch.cuda.is_available(), "operator qualification requires CUDA"
    return modeling_qwen3_5


def assert_numeric(actual, expected, *, relative_rmse=0.015):
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).square().mean().sqrt()
    magnitude = expected.float().square().mean().sqrt()
    assert magnitude > 0, "zero reference cannot qualify this operator"
    assert error / magnitude < relative_rmse
    torch.testing.assert_close(actual, expected, atol=float(magnitude) * 0.12, rtol=0.04)


@pytest.mark.parametrize("geometry_name,norm_dtype", [
    ("QWEN35_9B", torch.float32), ("QWEN38_27B", torch.bfloat16),
])
@pytest.mark.parametrize("rows", [3, 65])
def test_native_gdn_against_pinned_reference(reference, geometry_name, norm_dtype, rows):
    from b12x.integration.cuteafd import _common
    from b12x.integration.cuteafd.qwen_dense import compile_qwen_dense_gdn_aot

    torch.manual_seed(42)
    g = getattr(_common, geometry_name)
    program = compile_qwen_dense_gdn_aot(g, max_rows=max(64, rows), norm_dtype=norm_dtype)
    x = torch.randn(rows, g.hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    w_in = torch.randn(g.gdn_in_width, g.hidden, device="cuda", dtype=torch.bfloat16) * 0.01
    w_out = torch.randn(g.hidden, g.gdn_value_width, device="cuda", dtype=torch.bfloat16) * 0.01
    conv_weight = (torch.randn(g.gdn_conv_width, 4, device="cuda") * 0.1).bfloat16()
    a_log = torch.randn(g.gdn_value_heads, device="cuda") * 0.1
    dt_bias = (torch.randn(g.gdn_value_heads, device="cuda") * 0.1).bfloat16()
    norm_weight = (1 + torch.randn(128, device="cuda") * 0.1).to(norm_dtype)
    conv = torch.zeros(1, 3, g.gdn_conv_width, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(1, g.gdn_value_heads, 128, 128, device="cuda")
    slots = torch.zeros(rows, device="cuda", dtype=torch.int32)
    first = torch.zeros_like(slots)
    out = torch.empty_like(x)
    scratch = torch.empty(program.scratch_bytes(rows)["scratch"], device="cuda", dtype=torch.uint8)
    args = (x, w_in, conv_weight.float(), a_log, dt_bias.float(), norm_weight, w_out,
            conv, state, slots, first, out)
    if rows <= 64:
        program.launch(*args, None, scratch, scalars=[rows, 0])
    else:
        program.launch(*args, scratch, scalars=[rows])

    c, v, h = g.gdn_conv_width, g.gdn_value_width, g.gdn_value_heads
    # Separate checkpoint projections, as executed by the reference module.
    qkv, z, b, a = [torch.nn.functional.linear(x, weight) for weight in
                    w_in.split((c, v, h, h), dim=0)]
    filtered = reference.causal_conv1d_fn(qkv.T[None], conv_weight, activation="silu")[0].T
    q, k, value = filtered.split((g.gdn_key_width, g.gdn_key_width, v), dim=-1)
    ratio = h // g.gdn_key_heads
    q = q.reshape(1, rows, g.gdn_key_heads, 128).repeat_interleave(ratio, dim=2)
    k = k.reshape(1, rows, g.gdn_key_heads, 128).repeat_interleave(ratio, dim=2)
    value = value.reshape(1, rows, h, 128)
    decay = -a_log.exp() * torch.nn.functional.softplus(a.float() + dt_bias)
    recurrent, expected_state = reference.torch_recurrent_gated_delta_rule(
        q, k, value, decay[None], b.sigmoid()[None], output_final_state=True,
        use_qk_l2norm_in_kernel=True)
    norm = reference.Qwen3_5RMSNormGated(128).to(device="cuda", dtype=norm_dtype)
    norm.weight = torch.nn.Parameter(norm_weight, requires_grad=False)
    gated = norm(recurrent.reshape(rows, h, 128), z.reshape(rows, h, 128))
    expected = torch.nn.functional.linear(gated.reshape(rows, v), w_out)
    assert_numeric(out, expected)
    # The reference stores [key,value], the engine stores [value,key].
    assert_numeric(state, expected_state.transpose(-1, -2))
    projected = scratch[:rows * g.gdn_in_width * 2].view(torch.bfloat16).reshape(rows, g.gdn_in_width)
    assert torch.equal(conv[0], projected[-3:, :c])


@pytest.mark.parametrize("geometry_name", ["QWEN35_9B", "QWEN38_27B"])
def test_full_attention_against_pinned_reference(reference, geometry_name):
    from b12x.integration.cuteafd import _common
    from b12x.integration.cuteafd.qwen_dense import (
        compile_qwen_dense_attn_o_aot, compile_qwen_dense_full_gqa_aot,
        compile_qwen_dense_producer_aot,
    )

    torch.manual_seed(42)
    g = getattr(_common, geometry_name)
    config = reference.Qwen3_5TextConfig(
        hidden_size=g.hidden, num_attention_heads=g.heads, num_key_value_heads=g.kv_heads,
        head_dim=g.head_dim, rope_parameters={"rope_type": "default", "rope_theta": g.rope_theta,
                                             "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]})
    config._attn_implementation = "eager"
    attention = reference.Qwen3_5Attention(config, layer_idx=0).to(device="cuda", dtype=torch.bfloat16).eval()
    for weight in attention.parameters():
        weight.data.normal_(std=0.01)
    rows = 3
    x = torch.randn(rows, g.hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.cat((attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight), dim=0)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)
    rope_positions = positions[:, None].expand(-1, 3).to(torch.int32).contiguous()
    cache = torch.empty(64, 2, g.kv_heads, 256, device="cuda", dtype=torch.bfloat16)
    query = torch.empty(rows, g.heads, 256, device="cuda", dtype=torch.bfloat16)
    gate, attn = torch.empty_like(query), torch.empty_like(query)
    out = torch.empty_like(x)
    table = torch.zeros(1, device="cuda", dtype=torch.int32)
    producer = compile_qwen_dense_producer_aot(g, max_rows=64)
    full = compile_qwen_dense_full_gqa_aot(g, max_rows=64, max_splits=4)
    output = compile_qwen_dense_attn_o_aot(g, max_rows=64)
    scratch = torch.empty(max(p.scratch_bytes(64)["scratch"] for p in (producer, full, output)),
                          device="cuda", dtype=torch.uint8)
    producer.launch(x, weight, attention.q_norm.weight, attention.k_norm.weight, rope_positions,
                    positions, cache, query, gate, scratch, scalars=[rows])
    full.launch(query, cache, positions, table, attn, scratch, scalars=[rows, 0, 4])
    output.launch(attn, gate, attention.o_proj.weight, out, scratch, scalars=[rows])
    rotary = reference.Qwen3_5TextRotaryEmbedding(config).to(device="cuda")
    embedding = rotary(x[None], rope_positions.T[:, None].to(torch.int64))
    mask = torch.full((rows, rows), -torch.inf, device="cuda").triu(1)[None, None]
    expected, _ = attention(x[None], embedding, mask)
    assert_numeric(out, expected[0])


@pytest.mark.parametrize("geometry_name", ["QWEN35_9B", "QWEN38_27B"])
def test_norm_and_ffn_against_pinned_reference(reference, geometry_name):
    from b12x.integration.cuteafd import _common
    from b12x.integration.cuteafd.qwen_dense import compile_qwen_dense_ffn_aot, compile_qwen_dense_norm_aot

    torch.manual_seed(42)
    g = getattr(_common, geometry_name)
    x = torch.randn(3, g.hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    norm = reference.Qwen3_5RMSNorm(g.hidden).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.data.normal_(std=0.1)
    out = torch.empty_like(x)
    compile_qwen_dense_norm_aot(g).launch(x, norm.weight, out, scalars=[3])
    assert_numeric(out, norm(x), relative_rmse=0.005)
    config = reference.Qwen3_5TextConfig(hidden_size=g.hidden, hidden_act="silu")
    ffn = reference.Qwen3_5MLP(config, g.intermediate).to(device="cuda", dtype=torch.bfloat16)
    for weight in ffn.parameters():
        weight.data.normal_(std=0.01)
    packed = torch.cat((ffn.gate_proj.weight, ffn.up_proj.weight), dim=0)
    program = compile_qwen_dense_ffn_aot(g, max_rows=64)
    scratch = torch.empty(program.scratch_bytes(64)["scratch"], device="cuda", dtype=torch.uint8)
    program.launch(x, packed, ffn.down_proj.weight, out, scratch, scalars=[3])
    assert_numeric(out, ffn(x))
