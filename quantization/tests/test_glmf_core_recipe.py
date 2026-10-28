"""Fail-closed native Flash inventory and exact mixed-format storage accounting."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.core_k6 import recipe


def config_fixture():
    return {'architectures': ['Glm5NextForConditionalGeneration'],
     'image_end_token_id': 154831,
     'image_start_token_id': 154830,
     'image_token_id': 154854,
     'model_type': 'glm5_next',
     'text_config': {'attention_bias': False,
                     'attention_dropout': 0.0,
                     'dtype': 'bfloat16',
                     'eos_token_id': [154820, 154827, 154829],
                     'first_k_dense_replace': 3,
                     'hc_eps': 1e-06,
                     'hc_mult': 4,
                     'hc_sinkhorn_iters': 20,
                     'head_dim': 0,
                     'hidden_act': 'silu',
                     'hidden_size': 4096,
                     'index_head_dim': 128,
                     'index_kpool': 4,
                     'index_kpool_always_select_tail': True,
                     'index_kpool_compress': True,
                     'index_n_heads': 32,
                     'index_share_for_mtp_iteration': True,
                     'index_topk': 2048,
                     'indexer_rope_interleave': True,
                     'indexer_types': ['full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full',
                                       'full'],
                     'initializer_range': 0.02,
                     'intermediate_size': 12288,
                     'kv_lora_rank': 512,
                     'layer_types': ['linear_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'deepseek_sparse_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'deepseek_sparse_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'deepseek_sparse_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'deepseek_sparse_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'deepseek_sparse_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'deepseek_sparse_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'deepseek_sparse_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'deepseek_sparse_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'deepseek_sparse_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'deepseek_sparse_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'deepseek_sparse_attention',
                                     'linear_attention'],
                     'linear_attn_config': {'full_attn_layers': [3,
                                                                 7,
                                                                 11,
                                                                 15,
                                                                 19,
                                                                 23,
                                                                 27,
                                                                 31,
                                                                 35,
                                                                 39,
                                                                 43],
                                            'gate_lower_bound': -5.0,
                                            'head_dim': 128,
                                            'kda_layers': [0,
                                                           1,
                                                           2,
                                                           4,
                                                           5,
                                                           6,
                                                           8,
                                                           9,
                                                           10,
                                                           12,
                                                           13,
                                                           14,
                                                           16,
                                                           17,
                                                           18,
                                                           20,
                                                           21,
                                                           22,
                                                           24,
                                                           25,
                                                           26,
                                                           28,
                                                           29,
                                                           30,
                                                           32,
                                                           33,
                                                           34,
                                                           36,
                                                           37,
                                                           38,
                                                           40,
                                                           41,
                                                           42,
                                                           44],
                                            'num_heads': 64,
                                            'short_conv_kernel_size': 4},
                     'max_position_embeddings': 1048576,
                     'mhc': True,
                     'mla_use_nope': True,
                     'mlp_layer_types': ['dense',
                                         'dense',
                                         'dense',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse',
                                         'sparse'],
                     'model_type': 'glm5_next_text',
                     'moe_intermediate_size': 2048,
                     'moe_router_dtype': 'float32',
                     'n_group': 1,
                     'n_routed_experts': 288,
                     'n_shared_experts': 1,
                     'norm_topk_prob': True,
                     'num_attention_heads': 64,
                     'num_experts_per_tok': 8,
                     'num_hidden_layers': 45,
                     'num_key_value_heads': 64,
                     'num_nextn_predict_layers': 1,
                     'output_router_logits': False,
                     'pad_token_id': 154820,
                     'q_lora_rank': 1536,
                     'qk_head_dim': 256,
                     'qk_nope_head_dim': 256,
                     'qk_rope_head_dim': 0,
                     'rms_norm_eps': 1e-05,
                     'routed_scaling_factor': 2.5,
                     'router_aux_loss_coef': 0.001,
                     'scoring_func': 'sigmoid',
                     'swiglu_limit': 10.0,
                     'tie_word_embeddings': False,
                     'topk_group': 1,
                     'topk_method': 'noaux_tc',
                     'use_cache': True,
                     'v_head_dim': 256,
                     'vocab_size': 154880},
     'tie_word_embeddings': False,
     'transformers_version': '5.16.0',
     'video_end_token_id': 154833,
     'video_start_token_id': 154832,
     'video_token_id': 154855,
     'vision_config': {'attention_bias': True,
                       'attention_dropout': 0.0,
                       'depth': 24,
                       'hidden_act': 'silu',
                       'hidden_size': 1024,
                       'image_size': 448,
                       'in_channels': 3,
                       'initializer_range': 0.02,
                       'intermediate_size': 4096,
                       'model_type': 'glm5_next_vision',
                       'num_heads': 16,
                       'out_hidden_size': 4096,
                       'patch_size': 14,
                       'projection_intermediate_size': 10240,
                       'rms_norm_eps': 1e-05,
                       'spatial_merge_size': 2,
                       'swiglu_limit': 10.0,
                       'temporal_patch_size': 2}}


@pytest.fixture
def source(monkeypatch, tmp_path):
    headers, weight_map, offsets = {}, {}, {}
    for name, (dtype, shape) in recipe.reviewed_source_specs().items():
        number = name.split(".")[3] if name.startswith("model.language_model.layers.") else "extras"
        shard = f"source-{number}.safetensors"
        start = offsets.get(shard, 0)
        size = math.prod(shape) * (4 if dtype == "F32" else 2)
        headers.setdefault(shard, {})[name] = dict(dtype=dtype, shape=shape, data_offsets=[start,start+size])
        offsets[shard] = start+size
        weight_map[name] = shard
    monkeypatch.setattr(recipe,"read_header",lambda path:(headers[path.name],8))
    document = config_fixture()
    return SimpleNamespace(config=document["text_config"], document=document,
        snapshot=tmp_path, weight_map=weight_map, headers=headers)


def test_flash_reviewed_scope_and_independent_storage_totals(source):
    modules = recipe.reviewed_modules()
    assert len(recipe.reviewed_core_modules(include_mtp=False)) == 316
    assert len(recipe.reviewed_core_modules()) == 324
    assert len(recipe.reviewed_expert_modules()) == 37152
    assert len(modules) == 37476
    assert modules["model.language_model.layers.4.self_attn.q_proj"]["out_features"] == 8192
    kv = modules["model.language_model.layers.7.self_attn.kv_a_proj_with_mqa"]
    assert kv["out_features"] == kv["stored_out_features"] == 512
    assert recipe.packed_specs("kv_a",kv)["kv_a.trellis"] == ("I16",[256,32,96])
    assert "model.language_model.layers.4.self_attn.f_a_proj" not in modules
    assert "model.language_model.layers.7.self_attn.kv_b_proj" not in modules
    assert not any(name.startswith("model.visual.") for name in modules)
    assert all(g["bits"] == 4 for n,g in modules.items() if n.startswith("model.language_model.layers.45."))
    result = recipe.inventory(source)
    assert result["source_tensors"] == 38770
    assert result["source_payload_bytes"] == 642646653816
    assert result["selected_modules"] == 37476
    assert result["target_tensors"] == 151198
    assert result["target_payload_bytes"] == 165621841672
    assert result["preserved_tensors"] == 1294
    assert result["preserved_bytes"] == 3390684024
    assert sum(p["packed_bytes"] for p in result["parts"].values()) + result["preserved_bytes"] == result["target_payload_bytes"]
    assert sum(name.startswith("model.visual.") for name in source.weight_map)==347


@pytest.mark.parametrize("damage",["missing","extra","dtype","shape","extent","orphan_header","text_config","root_config","mtp","vision","recurrent","shard_path"])
def test_flash_inventory_fails_closed(source,damage):
    name="model.language_model.layers.7.self_attn.kv_a_proj_with_mqa.weight"
    spec=source.headers[source.weight_map[name]][name]
    if damage=="missing":del source.weight_map[name]
    elif damage=="extra":source.weight_map["unexpected.weight"]="source-extras.safetensors"
    elif damage=="dtype":spec["dtype"]="F16"
    elif damage=="shape":spec["shape"]=[576,4096]
    elif damage=="extent":spec["data_offsets"][1]-=2
    elif damage=="orphan_header":source.headers[source.weight_map[name]]["unexpected.weight"]=deepcopy(spec)
    elif damage=="text_config":source.config["swiglu_limit"]=0
    elif damage=="root_config":source.document["vision_config"]["depth"]=23
    elif damage=="mtp":del source.weight_map["model.language_model.layers.45.shared_head.norm.weight"]
    elif damage=="vision":del source.weight_map["model.visual.blocks.0.attn.qkv.weight"]
    elif damage=="recurrent":
        n="model.language_model.layers.0.self_attn.A_log"
        source.headers[source.weight_map[n]][n]["dtype"]="BF16"
    else:source.weight_map[name]="../escaped.safetensors"
    with pytest.raises(ValueError):recipe.inventory(source)


@pytest.mark.parametrize("bits",[0,2,3,7,16])
def test_flash_unreviewed_bits_rejected(bits):
    with pytest.raises(ValueError):recipe.packed_specs("head",recipe.reviewed_core_modules()["lm_head"],bits=bits)
