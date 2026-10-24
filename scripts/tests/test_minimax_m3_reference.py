"""MiniMax M3 text reference contracts, using CPU-only synthetic release tensors."""
import importlib
import json
import math
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
model = importlib.import_module("families.minimax_m3.model")


def tiny_config():
    return {
        "model_type": "minimax_m3_vl",
        "text_config": {
            "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "vocab_size": 64, "max_position_embeddings": 256, "rms_norm_eps": 1e-6,
            "use_gemma_norm": True, "attention_output_gate": False,
            "rope_theta": 10000000, "rotary_dim": 16, "partial_rotary_factor": .5,
            "hidden_act": "swigluoai", "use_qk_norm": True, "tie_word_embeddings": False,
            "dense_intermediate_size": 32, "shared_intermediate_size": 32,
            "num_local_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
            "scoring_func": "sigmoid", "use_routing_bias": True, "moe_layer_freq": [0, 1],
            "qk_norm_type": "per_head", "swiglu_alpha": 1.702, "swiglu_limit": 7.,
            "routed_scaling_factor": 2.,
            "sparse_attention_config": {
                "use_sparse_attention": True, "sparse_index_dim": 32,
                "sparse_num_index_heads": 1, "sparse_topk_blocks": 2,
                "sparse_block_size": 2, "sparse_score_type": "max",
                "sparse_init_block": 0, "sparse_local_block": 1,
                "sparse_attention_freq": [1, 1], "sparse_disable_index_value": [1, 1],
            },
        },
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "mxfp8", "activation_scheme": "dynamic",
            "weight_block_size": [1, 32], "group_size": 16,
            "moe_quant_algo": "NVFP4", "moe_quant_format": "nvfp4-pack-quantized",
        },
    }


@pytest.fixture
def snapshot(tmp_path):
    folder = tmp_path / "snapshot"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(tiny_config()))
    generator = torch.Generator().manual_seed(71)
    tensors = {}

    def bf16(name, shape, scale=.1):
        tensors[name] = (torch.randn(shape, generator=generator) * scale).bfloat16()

    def mxfp8(name, rows, cols):
        tensors[name + ".weight"] = torch.randn(rows, cols, generator=generator).to(torch.float8_e4m3fn)
        tensors[name + ".weight_scale_inv"] = torch.full((rows, cols // 32), 123, dtype=torch.uint8)

    def nvfp4(name):
        tensors[name + ".weight_packed"] = torch.randint(0, 256, (32, 16), generator=generator, dtype=torch.uint8)
        tensors[name + ".weight_scale"] = torch.full((32, 2), .125).to(torch.float8_e4m3fn)
        tensors[name + ".weight_global_scale"] = torch.tensor(4.)
        tensors[name + ".input_global_scale"] = torch.tensor(32.)

    bf16("language_model.model.embed_tokens.weight", (64, 32), .5)
    bf16("language_model.model.norm.weight", (32,))
    bf16("language_model.lm_head.weight", (64, 32))
    for layer in range(2):
        prefix = f"language_model.model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            bf16(prefix + name + ".weight", (32,))
        for name in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
            bf16(prefix + "self_attn." + name + ".weight", (32,))
        for name, rows, cols in (("q_proj", 64, 32), ("k_proj", 32, 32),
                                 ("v_proj", 32, 32), ("o_proj", 32, 64),
                                 ("index_q_proj", 32, 32), ("index_k_proj", 32, 32)):
            mxfp8(prefix + "self_attn." + name, rows, cols)
        if layer == 0:
            for name in ("gate_proj", "up_proj", "down_proj"):
                mxfp8(prefix + "mlp." + name, 32, 32)
        else:
            bf16(prefix + "block_sparse_moe.gate.weight", (4, 32))
            tensors[prefix + "block_sparse_moe.gate.weight"] = tensors[prefix + "block_sparse_moe.gate.weight"].float()
            tensors[prefix + "block_sparse_moe.e_score_correction_bias"] = torch.tensor([.2, -.1, .0, .1])
            for expert in range(4):
                for name in ("w1", "w2", "w3"):
                    nvfp4(prefix + f"block_sparse_moe.experts.{expert}." + name)
            for name in ("gate_proj", "up_proj", "down_proj"):
                mxfp8(prefix + "block_sparse_moe.shared_experts." + name, 32, 32)
    save_file(tensors, folder / "model.safetensors")
    (folder / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }))
    return folder


def test_mxfp8_dequantization_row_and_32_column_blocks():
    weights = torch.tensor([[1., -2., .5, 3.] * 16, [-1., 2., -.5, -3.] * 16]).to(torch.float8_e4m3fn)
    scales = torch.tensor([[127, 128], [126, 125]], dtype=torch.uint8)
    expected = weights.float() * torch.tensor([[1.] * 32 + [2.] * 32, [.5] * 32 + [.25] * 32])
    assert torch.equal(model.dequant_mxfp8(weights, scales), expected)


def test_nvfp4_dequantization_nibble_order_signed_values_and_scales():
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8)
    scales = torch.tensor([[2., 4.]]).to(torch.float8_e4m3fn)
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    expected = torch.cat((values, values * 2))[None]
    assert torch.equal(model.dequant_nvfp4(packed, scales, torch.tensor(2.)), expected)


def test_activation_quantizers_preserve_representable_values_and_zero():
    mx = torch.tensor([[0., .5, -1., 448.] * 8 + [0., 1., -2., 896.] * 8])
    assert torch.equal(model.quantize_mxfp8(mx), mx)
    fp4 = torch.tensor([[0., .5, -1., 1.5, -2., 3., -4., 6.] * 2])
    assert torch.equal(model.quantize_nvfp4(fp4, torch.tensor(2.)), fp4)
    zero = torch.zeros(2, 32)
    assert torch.equal(model.quantize_mxfp8(zero), zero)
    assert torch.equal(model.quantize_nvfp4(zero, torch.tensor(2.)), zero)


def test_nvfp4_activation_rounding_ties_to_even():
    values = torch.tensor([[.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.] * 2])
    expected = torch.tensor([[0., 1., 1., 2., 2., 4., 4., 6.] * 2])
    assert torch.equal(model.quantize_nvfp4(values, torch.tensor(1.)), expected)


def test_router_rust_vector_bias_is_selection_only():
    ids, weights = model.route(torch.tensor([[0., 2., 1., -1.]]), torch.tensor([5., 0., 0., 0.]), 2, 2.)
    assert ids.tolist() == [[0, 1]]
    s2 = 1 / (1 + math.exp(-2))
    torch.testing.assert_close(weights, torch.tensor([[.5 / (.5 + s2) * 2, s2 / (.5 + s2) * 2]]))
    assert weights.dtype == torch.float32
    assert torch.allclose(weights.sum(-1), torch.tensor([2.]))


def test_router_rust_ties_prefer_lower_id():
    ids, weights = model.route(torch.zeros(1, 4), torch.zeros(4), 2, 2.)
    assert ids.tolist() == [[0, 1]]
    assert torch.equal(weights, torch.ones(1, 2))


def test_swiglu_oai_rust_vectors():
    gate = torch.tensor([0., 9., 7., 2., 2., -9., -7., 2.])
    up = torch.tensor([3., 0., 0., -20., -7., 0., 0., .5])
    out = model.swiglu_oai(gate, up)
    assert out[0] == 0 and out[1] == out[2] and out[3] == out[4]
    assert out[5] != out[6]
    assert out[7].item() == pytest.approx(2 / (1 + math.exp(-1.702 * 2)) * 1.5, abs=1e-6)


def test_gemma_norm_rust_vector_and_bf16_boundary():
    out = model.gemma_norm(torch.tensor([3., 4.]), torch.tensor([0., 1.]), 0.)
    torch.testing.assert_close(out, torch.tensor([3 / math.sqrt(12.5), 8 / math.sqrt(12.5)]))
    x = torch.tensor([[3., 4.]], dtype=torch.bfloat16)
    assert model.gemma_norm(x, torch.tensor([0., 1.]), 0.).dtype == torch.bfloat16


def test_partial_rope_rust_vector_neox_pairs_and_tail():
    x = torch.arange(128, dtype=torch.float32).reshape(1, 1, 128)
    assert torch.equal(model.rope_partial(x, torch.tensor([0]), 64, 1e7), x)
    out = model.rope_partial(x, torch.tensor([5]), 64, 1e7)
    assert torch.equal(out[..., 64:], x[..., 64:])
    assert out[0, 0, 0].item() == pytest.approx(-32 * math.sin(5), abs=1e-4)
    assert out[0, 0, 32].item() == pytest.approx(32 * math.cos(5), abs=1e-4)
    assert torch.equal(x.flatten(), torch.arange(128, dtype=torch.float32))


def test_msa_rust_short_context_vectors_and_padding():
    scores = torch.zeros(1, 2, 2048)
    selected = model.msa_select(scores, torch.tensor([0, 2047]))
    assert selected[0, 0].tolist() == [0] + [-1] * 15
    assert selected[0, 1].tolist() == list(range(16))


def test_msa_rust_long_context_vector_local_forced_and_future_masked():
    position = 128 * 40 + 5
    scores = torch.full((1, 1, position + 128), -1.)
    hot = [3, 7, 11, 13, 17, 19, 23, 29, 31, 33, 35, 36, 37, 38, 39]
    for block in hot:
        scores[0, 0, block * 128 + 77] = 10 + block
    scores[..., position + 1:] = 1e6
    selected = model.msa_select(scores, torch.tensor([position]))
    assert selected[0, 0].tolist() == hot + [40]
    assert torch.equal(selected, model.msa_select(scores[..., :position + 1], torch.tensor([position])))


def test_msa_selections_are_per_kv_head_and_ties_use_lower_id():
    scores = torch.zeros(2, 1, 8)
    scores[0, 0, 2] = 20
    scores[1, 0, 4] = 30
    selected = model.msa_select(scores, torch.tensor([7]), block=2, topk=2, local=1)
    assert selected[:, 0].tolist() == [[1, 3], [2, 3]]
    tied = model.msa_select(torch.zeros_like(scores), torch.tensor([7]), block=2, topk=2, local=1)
    assert tied[:, 0].tolist() == [[0, 3], [0, 3]]


def forward(snapshot, tokens, activation):
    cfg = model.Config.from_dict(json.loads((snapshot / "config.json").read_text()))
    weights = model.Weights(snapshot, device="cpu")
    try:
        weights.validate(cfg)
        hidden = weights.tensor("language_model.model.embed_tokens.weight")[tokens]
        states = [hidden.clone()]
        for index in range(2):
            hidden = model.layer_forward(hidden, weights.load_layer(index, cfg), cfg, index, activation=activation)
            states.append(hidden.clone())
        hidden = model.gemma_norm(hidden, weights.tensor("language_model.model.norm.weight"), 1e-6).bfloat16()
        head = weights.tensor("language_model.lm_head.weight")
        logits = torch.nn.functional.linear(hidden.float(), head.float()) if activation == "served-cache" else torch.nn.functional.linear(hidden, head)
        return states, logits
    finally:
        weights.close()


@pytest.mark.parametrize("activation", ["served", "served-cache", "bf16"])
def test_two_layer_cpu_forward_dense_sparse_moe_and_prefix(snapshot, activation):
    tokens = torch.tensor([1, 7, 3, 9, 11, 5, 2])
    states, logits = forward(snapshot, tokens, activation)
    dtype = torch.float32 if activation == "served-cache" else torch.bfloat16
    assert logits.shape == (7, 64) and logits.dtype == dtype
    assert torch.isfinite(logits).all()
    for before, after in zip(states, states[1:]):
        dtype = torch.float32 if activation == "served-cache" else torch.bfloat16
        assert after.dtype == dtype and torch.isfinite(after).all()
        assert not torch.equal(before, after)
    prefix_states, prefix_logits = forward(snapshot, tokens[:5], activation)
    for whole, prefix in zip(states, prefix_states):
        torch.testing.assert_close(whole[:5], prefix, rtol=0, atol=0)
    torch.testing.assert_close(logits[:5], prefix_logits, rtol=0, atol=0)


def test_served_and_diagnostic_activation_arms_differ(snapshot):
    tokens = torch.tensor([1, 7, 3, 9, 11, 5, 2])
    _, served = forward(snapshot, tokens, "served")
    _, diagnostic = forward(snapshot, tokens, "bf16")
    assert not torch.equal(served, diagnostic)


def panel(tokens):
    from fidelity_windows import set_hash
    result = dict(schema="cuteafd.fidelity.set/1", family="minimax_m3", checkpoint=model.CHECKPOINT,
                  windows=[dict(id="sample", tokens=tokens, roles=["ctx"] * 3 + ["gen"] * (len(tokens) - 3),
                                score_from=3, bucket="0-2K")], quick_windows=["sample"])
    result["set_sha256"] = set_hash(result)
    return result


@pytest.mark.parametrize("activation", ["served", "served-cache", "bf16"])
def test_layer_resume_is_byte_exact_and_loads_only_remaining_layer(snapshot, tmp_path, monkeypatch, activation):
    from families.minimax_m3.golden import run_layers
    manifest = panel([1, 7, 3, 9, 11, 5, 2])
    complete, complete_times = run_layers(snapshot, manifest, tmp_path / "complete", activation=activation)
    stopped, times = run_layers(snapshot, manifest, tmp_path / "partial", activation=activation, stop_after_layer=0)
    assert [t["layer"] for t in times] == [0]
    assert not (tmp_path / "partial/meta.json").exists()
    assert not torch.equal(torch.from_numpy(stopped[0].load()), torch.from_numpy(complete[0].load()))
    loaded = []
    original = model.Weights.load_layer
    def tracked(self, index, cfg):
        loaded.append(index)
        return original(self, index, cfg)
    monkeypatch.setattr(model.Weights, "load_layer", tracked)
    resumed, times = run_layers(snapshot, manifest, tmp_path / "resumed", activation=activation,
                               resume=tmp_path / "partial/state")
    assert loaded == [1]
    assert [t["layer"] for t in times] == [0, 1]
    assert len(complete_times) == 2
    assert complete[0].path.read_bytes() == resumed[0].path.read_bytes()


def test_resume_rejects_arithmetic_token_and_blob_changes(snapshot, tmp_path):
    from families.minimax_m3.golden import run_layers
    manifest = panel([1, 7, 3, 9, 11, 5, 2])
    states, _ = run_layers(snapshot, manifest, tmp_path / "partial", stop_after_layer=0)
    for name, changed, arm in (("arm", manifest, "bf16"),
                               ("tokens", panel([2, 7, 3, 9, 11, 5, 2]), "served")):
        with pytest.raises(ValueError, match="identity differs"):
            run_layers(snapshot, changed, tmp_path / name, activation=arm, resume=tmp_path / "partial/state")
    payload = bytearray(states[0].path.read_bytes())
    payload[0] ^= 1
    states[0].path.write_bytes(payload)
    with pytest.raises(ValueError, match="bytes changed"):
        run_layers(snapshot, manifest, tmp_path / "corrupt", resume=tmp_path / "partial/state")


def test_interrupted_successor_cannot_advance_resume(snapshot, tmp_path, monkeypatch):
    from families.minimax_m3.golden import run_layers
    from fidelity_windows import LayerCheckpoints
    manifest = panel([1, 7, 3, 9, 11, 5, 2])
    complete, _ = run_layers(snapshot, manifest, tmp_path / "complete")
    original = LayerCheckpoints.commit
    def interrupt(self, layer_id, states, times):
        if layer_id == 1:
            raise RuntimeError("synthetic interruption before durable commit")
        return original(self, layer_id, states, times)
    monkeypatch.setattr(LayerCheckpoints, "commit", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_layers(snapshot, manifest, tmp_path / "interrupted")
    assert json.loads((tmp_path / "interrupted/state/latest.json").read_text())["layer"] == "layer00"
    monkeypatch.setattr(LayerCheckpoints, "commit", original)
    resumed, _ = run_layers(snapshot, manifest, tmp_path / "resumed", resume=tmp_path / "interrupted/state")
    assert complete[0].path.read_bytes() == resumed[0].path.read_bytes()


def test_checkpoint_inventory_and_formats_fail_closed(snapshot):
    cfg = model.Config.from_dict(tiny_config())
    weights = model.Weights(snapshot)
    weights.validate(cfg)
    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    weights.descriptor(name)["shape"] = [32, 32]
    with pytest.raises(ValueError, match="shape/dtype"):
        weights.validate(cfg)
    del weights.index[name]
    with pytest.raises(ValueError, match="inventory"):
        weights.validate(cfg)


@pytest.mark.parametrize("bad", [0., -1., float("inf"), float("nan")])
def test_nvfp4_rejects_invalid_global_scales(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        model.quantize_nvfp4(torch.ones(1, 32), torch.tensor(bad))


def test_projection_gate_up_uses_shared_min_input_scale(snapshot):
    cfg = model.Config.from_dict(tiny_config())
    weights = model.Weights(snapshot).load_layer(1, cfg)
    stem = "block_sparse_moe.experts.0"
    weights[stem + ".w1.input_global_scale"] = torch.tensor(8.)
    weights[stem + ".w3.input_global_scale"] = torch.tensor(64.)
    hidden = torch.linspace(-4, 4, 64).reshape(2, 32).bfloat16()
    gate = model.project(hidden, weights, stem + ".w1", "served", torch.tensor(8.))
    up = model.project(hidden, weights, stem + ".w3", "served", torch.tensor(8.))
    assert torch.isfinite(model.swiglu_oai(gate, up)).all()
    weights[stem + ".w3.weight_global_scale"] = torch.tensor(8.)
    # Route every row through expert 0, so a mismatched fused scale cannot hide.
    weights["block_sparse_moe.e_score_correction_bias"][0] = 100
    with pytest.raises(ValueError, match="equal weight global scales"):
        model.feed_forward(hidden, weights, cfg, 1, "served")


def test_family_registrations_are_present():
    import ast
    for script, symbol in (("golden-run.py", "SUBCOMMANDS"), ("fidelity-set.py", "FAMILIES")):
        tree = ast.parse((ROOT / "scripts/bench" / script).read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == symbol for t in node.targets))
        value = ast.literal_eval(assignment.value)
        assert ("minimax" if symbol == "SUBCOMMANDS" else "minimax_m3") in value
        if symbol == "SUBCOMMANDS":
            assert value["minimax"] == "minimax-golden"


@pytest.mark.parametrize("activation", ["served", "served-cache", "bf16"])
def test_cpu_golden_scores_and_qualifies_real_fidelity_contract(snapshot, tmp_path, monkeypatch, activation):
    from types import SimpleNamespace
    import numpy as np
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from fidelity_windows import validate_qualification
    from families.minimax_m3.golden import execute
    Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="0")).save(str(snapshot / "tokenizer.json"))
    # Only the revision gate is synthetic; all snapshot, set, prefix and output
    # contracts execute normally. No checkpoint Python is imported.
    monkeypatch.setattr(model, "REVISION", snapshot.name)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU reference must not inspect or access CUDA")
    for method in ("is_available", "synchronize", "empty_cache", "get_device_capability",
                   "get_device_name", "mem_get_info", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, method, forbidden)
    manifest = panel(([1, 7, 3, 9, 11, 5, 2] * 92)[:640])
    windows = tmp_path / "windows.json"
    windows.write_text(json.dumps(manifest))
    out = tmp_path / "golden"
    args = SimpleNamespace(snapshot=snapshot, windows=windows, out=out, device="cpu", activation=activation,
                           resume=None, stop_after_layer=None, layers=None, prefix_only=False, check_only=False)
    execute(args)
    meta = json.loads((out / "meta.json").read_bytes())
    validate_qualification(meta["prefix_qualification"], manifest, meta["snapshot_identity"])
    assert meta["diagnostic_only"] == (activation == "bf16")
    assert meta["activation"] == activation and not meta["mtp"] and not meta["use_cache"]
    entry = meta["windows"][0]
    assert entry["positions"] == list(range(3, 640))
    assert meta["scored_tokens"] == 637 and math.isfinite(meta["nll"])
    logits = np.fromfile(out / entry["path"] / "logits.bin", dtype="<f4").reshape(637, 64)
    assert np.isfinite(logits).all()
    np.testing.assert_array_equal(np.fromfile(out / entry["path"] / "tokens.bin", dtype="<i4"), manifest["windows"][0]["tokens"])
    assert len(meta["seconds_per_layer"]) == 2


def test_cache_e4m3_rounds_ties_even_saturates_and_preserves_sign():
    values = torch.tensor([0., -0., 1.0625, 1.1875, -1.0625, -1.1875,
                           2. ** -10, 3. * 2. ** -10, 480., -480., float("inf"), -float("inf")])
    expected = [0x00, 0x80, 0x38, 0x3a, 0xb8, 0xba, 0x00, 0x02, 0x7e, 0xfe, 0x7e, 0xfe]
    assert model.cache_e4m3(values).view(torch.uint8).tolist() == expected


def test_index_e4m3_uses_fp32_scale_and_reciprocal_then_multiply():
    values = torch.tensor([[[0., -0., 1.0625, 448.], [0., 3., -1., .5]],
                           [[0., 0., 0., 0.], [-224., 1.1875, 3., 7.]]], dtype=torch.bfloat16)
    packed, scale = model.index_e4m3(values)
    expected_scale = torch.tensor([[1., 3. / 448.], [1., .5]], dtype=torch.float32)
    assert scale.dtype == torch.float32 and torch.equal(scale, expected_scale)
    inverse = torch.ones_like(expected_scale) / expected_scale
    expected = (values.float() * inverse[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(packed.view(torch.uint8), expected.view(torch.uint8))
    assert packed.view(torch.uint8)[0, 0].tolist() == [0, 128, 56, 126]
    assert torch.count_nonzero(packed[1, 0].float()) == 0


def test_index_reciprocal_rounding_differs_from_direct_division():
    values = torch.tensor([[49., 203. / 1024.]], dtype=torch.bfloat16)
    packed, scale = model.index_e4m3(values)
    assert scale.item() == 7. / 64.
    # RN(64/7) * (203/1024) lands above the 1.8125 E4M3 tie.
    assert packed.view(torch.uint8).tolist() == [[0x7e, 0x3f]]
    direct = (values.float() / scale[:, None]).to(torch.float8_e4m3fn)
    assert direct.view(torch.uint8).tolist() == [[0x7e, 0x3e]]


def test_index_scores_apply_scales_after_accumulation():
    q = torch.tensor([[[1., 2., -1., 3.], [-2., 1., 4., 0.]]]).to(torch.float8_e4m3fn)
    k = torch.tensor([[[2., 1., -1., 0.], [1., -3., 2., 4.]]]).to(torch.float8_e4m3fn)
    qs, ks = torch.tensor([[.3, .7]]), torch.tensor([[.2, .9]])
    expected = (torch.tensor([[[5., 5.], [-7., 3.]]]) * (qs * .5)[..., None]) * ks[..., None, :]
    assert torch.equal(model.index_scores(q, qs, k, ks), expected)


def test_served_cache_arm_changes_values_and_retains_fp32_residual(snapshot):
    cfg = model.Config.from_dict(tiny_config())
    weights = model.Weights(snapshot)
    hidden = weights.tensor("language_model.model.embed_tokens.weight")[:7]
    layer = weights.load_layer(0, cfg)
    trace = {}
    cached = model.layer_forward(hidden, layer, cfg, 0, "served-cache", trace)
    original = model.layer_forward(hidden, layer, cfg, 0, "served")
    assert cached.dtype == torch.float32 and trace["norm"].dtype == torch.bfloat16
    assert not torch.equal(cached, original.float())
    assert not torch.equal(cached, cached.bfloat16().float())
    assert trace["q2k"].shape == (7, 1, 2)
    assert trace["k_fp8"].dtype == torch.uint8
    assert torch.equal(trace["k_fp8"], model.cache_e4m3(trace["k"]).view(torch.uint8))


@pytest.mark.parametrize("layer", [0, 1])
def test_layer_diagnostic_roundtrip_and_exact_selection_gate(snapshot, tmp_path, layer):
    from families.minimax_m3 import diagnostic
    cfg = model.Config.from_dict(tiny_config())
    weights = model.Weights(snapshot)
    hidden = weights.tensor("language_model.model.embed_tokens.weight")[:7].float()
    hidden[0, 0] += 2. ** -18
    trace = {}
    model.layer_forward(hidden, weights.load_layer(layer, cfg), cfg, layer, "served-cache", trace)
    dump = tmp_path / "dump"
    diagnostic.write_dump(dump, layer, trace)
    result = diagnostic.compare(snapshot, dump, reference_out=tmp_path / "reference")
    assert result["passed"] and result["first_failure"] is None
    assert all(entry["mismatches"] == 0 for entry in result["tensors"])
    import numpy as np
    selected = np.fromfile(dump / "q2k.bin", dtype="<i4")
    selected[0] += 1
    selected.tofile(dump / "q2k.bin")
    result = diagnostic.compare(snapshot, dump, rtol=100., atol=100.)
    assert not result["passed"] and result["first_failure"] == "q2k"


def test_layer_diagnostic_combines_causal_history_and_rejects_holes(snapshot, tmp_path):
    from families.minimax_m3 import diagnostic
    cfg = model.Config.from_dict(tiny_config())
    weights = model.Weights(snapshot)
    hidden = weights.tensor("language_model.model.embed_tokens.weight")[:7].float()
    trace = {}
    model.layer_forward(hidden, weights.load_layer(0, cfg), cfg, 0, "served-cache", trace)
    dump = tmp_path / "dump"
    dump.mkdir()
    manifests = []
    for index, (start, end) in enumerate(((0, 3), (3, 7))):
        folder = dump / str(index)
        diagnostic.write_dump(folder, 0, {name: tensor[start:end] for name, tensor in trace.items()})
        step = json.loads((folder / "manifest.json").read_text())["steps"][0]
        step["positions"] = list(range(start, end))
        for descriptor in step["tensors"].values():
            descriptor["path"] = f"{index}/" + descriptor["path"]
        manifests.append(step)
    manifest = dict(schema=diagnostic.SCHEMA, activation="served-cache", layer=0, steps=manifests)
    (dump / "manifest.json").write_text(json.dumps(manifest))
    assert diagnostic.compare(snapshot, dump)["passed"]
    manifest["steps"][1]["positions"][0] = 4
    (dump / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contiguous causal history"):
        diagnostic.compare(snapshot, dump)


def test_layer_diagnostic_rejects_bad_cache_payload_and_nonfinite(snapshot, tmp_path):
    from families.minimax_m3 import diagnostic
    cfg = model.Config.from_dict(tiny_config())
    weights = model.Weights(snapshot)
    hidden = weights.tensor("language_model.model.embed_tokens.weight")[:7].float()
    trace = {}
    model.layer_forward(hidden, weights.load_layer(0, cfg), cfg, 0, "served-cache", trace)
    trace["index_k_fp8"][0, 0, 0] ^= 1
    dump = tmp_path / "dump"
    diagnostic.write_dump(dump, 0, trace)
    result = diagnostic.compare(snapshot, dump)
    assert not result["passed"] and result["first_failure"] == "index_k_fp8"
    entry = diagnostic.difference("x", torch.tensor([float("nan")]), torch.ones(1), rtol=1., atol=1.)
    assert not entry["passed"] and not entry["finite"] and entry["max_abs"] is None


@pytest.mark.parametrize("field", ["hash", "dtype", "activation"])
def test_layer_diagnostic_validates_tensor_identity(snapshot, tmp_path, field):
    from families.minimax_m3 import diagnostic
    cfg = model.Config.from_dict(tiny_config())
    weights = model.Weights(snapshot)
    hidden = weights.tensor("language_model.model.embed_tokens.weight")[:7].float()
    trace = {}
    model.layer_forward(hidden, weights.load_layer(0, cfg), cfg, 0, "served-cache", trace)
    dump = tmp_path / "dump"
    diagnostic.write_dump(dump, 0, trace)
    manifest = json.loads((dump / "manifest.json").read_bytes())
    if field == "hash":
        manifest["steps"][0]["tensors"]["q2k"]["sha256"] = "0" * 64
    elif field == "dtype":
        manifest["steps"][0]["tensors"]["q2k"]["dtype"] = "f32"
    else:
        manifest["activation"] = "served"
    (dump / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash differs|dtype differs|manifest"):
        diagnostic.compare(snapshot, dump)


