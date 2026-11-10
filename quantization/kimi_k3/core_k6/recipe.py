"""Reviewed, fail-closed tensor inventory for the Kimi K3 K2 + core K6 recipe."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..uniform_k2.storage import read_header

BASE_REPO = "wrldsuksgo2mars/Kimi-K3-EXL3-K2"
BASE_REVISION = "acba35b61a672561652b16a96fca1be85c1bd875"
TARGET_REPO = "wrldsuksgo2mars/Kimi-K3-EXL3-K2-Core-K6"
MODULE_COUNT = 926
SELECTED_BF16_BYTES = 107_429_756_928
SELECTED_K6_BYTES = 40_323_110_520
BASE_PAYLOAD_BYTES = 798_382_459_904
TARGET_PAYLOAD_BYTES = 731_275_813_496
BASE_TENSOR_COUNT = 991_812
TARGET_TENSOR_COUNT = 994_590
FULL_ATTENTION_LAYERS = tuple(range(4, 93, 4)) + (93,)
MCG_MULTIPLIER = 0xCBAC1FED
SUFFIXES = ("trellis", "suh", "svh", "mcg")
_CANDIDATE = re.compile(
    r"language_model\.model\.layers\.[0-9]+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|"
    r"g_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa)|block_sparse_moe\."
    r"(?:shared_experts\.(?:gate_proj|up_proj|down_proj)|routed_expert_(?:down|up)_proj))\.weight$"
)


def reviewed_modules():
    """The only allowed selection; names use zero-based checkpoint layer indices."""
    selected = {}

    def add(name, inputs, outputs):
        selected[name] = dict(in_features=inputs, out_features=outputs,
                              stored_out_features=(outputs + 127) // 128 * 128)

    for layer in range(93):
        prefix = f"language_model.model.layers.{layer}."
        attention = prefix + "self_attn."
        if layer + 1 in FULL_ATTENTION_LAYERS:
            add(attention + "q_a_proj", 7168, 1536)
            add(attention + "q_b_proj", 1536, 18432)
            add(attention + "kv_a_proj_with_mqa", 7168, 576)
        else:
            for projection in ("q_proj", "k_proj", "v_proj"):
                add(attention + projection, 7168, 12288)
        add(attention + "g_proj", 7168, 12288)
        add(attention + "o_proj", 12288, 7168)
        if layer:
            moe = prefix + "block_sparse_moe."
            add(moe + "shared_experts.gate_proj", 7168, 6144)
            add(moe + "shared_experts.up_proj", 7168, 6144)
            add(moe + "shared_experts.down_proj", 6144, 7168)
            add(moe + "routed_expert_down_proj", 7168, 3584)
            add(moe + "routed_expert_up_proj", 3584, 7168)
    add("language_model.lm_head", 7168, 163840)
    return selected


def packed_specs(module, geometry, *, bits=6):
    """Safetensors dtype/shape expectations, including output padding."""
    inputs = geometry["in_features"]
    outputs = geometry["stored_out_features"]
    if inputs % 128 or outputs % 128 or bits not in (2, 6):
        raise ValueError(f"unreviewed EXL3 alignment: {module}")
    return {module + ".trellis": ("I16", [inputs // 16, outputs // 16, bits * 16]),
            module + ".suh": ("F16", [inputs]),
            module + ".svh": ("F16", [outputs]),
            module + ".mcg": ("I32", [])}


def selected_modules(source):
    """Validate the checkpoint's reviewed dense matrices without loading weights.

    Iterate one source header at a time. In particular, do not retain all 94
    headers: the base checkpoint contains almost a million routed buffers.
    """
    config = source.text_config
    expected_config = dict(num_hidden_layers=93, hidden_size=7168, vocab_size=163840,
                           q_lora_rank=1536, kv_lora_rank=512, num_attention_heads=96,
                           qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
                           num_experts=896, num_experts_per_token=16,
                           num_shared_experts=2, moe_intermediate_size=3072,
                           routed_expert_hidden_size=3584, first_k_dense_replace=1)
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"unreviewed Kimi config {key}: {config.get(key)!r}")
    linear = config["linear_attn_config"]
    if tuple(linear["full_attn_layers"]) != FULL_ATTENTION_LAYERS:
        raise ValueError("unreviewed MLA layer inventory")
    if tuple(linear["kda_layers"]) != tuple(x for x in range(1, 94) if x not in FULL_ATTENTION_LAYERS):
        raise ValueError("unreviewed KDA layer inventory")
    for key, expected in dict(head_dim=128, num_heads=96).items():
        if linear.get(key) != expected:
            raise ValueError(f"unreviewed KDA {key}")
    selected = reviewed_modules()
    candidates = {name.removesuffix(".weight") for name in source.weight_map
                  if _CANDIDATE.fullmatch(name) or name == "language_model.lm_head.weight"}
    if candidates != set(selected):
        raise ValueError(f"core selection inventory differs: missing={sorted(set(selected) - candidates)} "
                         f"extra={sorted(candidates - set(selected))}")
    by_shard = defaultdict(list)
    for module in selected:
        by_shard[source.weight_map[module + ".weight"]].append(module)
    total = 0
    for shard, modules in by_shard.items():
        header, _ = read_header(source.snapshot / shard)
        for module in modules:
            spec = header.get(module + ".weight")
            geometry = selected[module]
            if spec is None or spec["dtype"] != "BF16" or spec["shape"] != [geometry["out_features"], geometry["in_features"]]:
                raise ValueError(f"unreviewed dense tensor dtype/geometry: {module}")
            total += spec["data_offsets"][1] - spec["data_offsets"][0]
            if any(module + "." + suffix in source.weight_map for suffix in SUFFIXES):
                raise ValueError(f"already quantized core tensor: {module}")
    groups = Counter("head" if module == "language_model.lm_head" else
                     "attention" if ".self_attn." in module else "moe" for module in selected)
    if groups != dict(attention=465, moe=460, head=1) or len(selected) != MODULE_COUNT or total != SELECTED_BF16_BYTES:
        raise ValueError("reviewed core totals differ")
    return selected
