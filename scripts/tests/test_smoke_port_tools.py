"""The smoke-port skill's phase-0 tools (scripts/agents/skills/smoke-port/scripts/).

A tiny Mistral-native snapshot (params.json, consolidated shards with real safetensors
headers) goes through the census, the header-only stub and the HF-style view, CPU only and
without network. The image extractor's reference parsing is checked offline.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts" / "agents" / "skills" / "smoke-port" / "scripts"


def write_shard(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> int:
    """A valid safetensors file whose data is zeros; returns its size."""
    width = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = width[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n]}
        offset += n
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))
    return path.stat().st_size


PARAMS = {
    "dim": 64, "n_layers": 4, "head_dim": 16, "hidden_dim": 128, "n_heads": 4, "n_kv_heads": 4,
    "rope_theta": 10000.0, "norm_eps": 1e-05, "vocab_size": 32, "tied_embeddings": False,
    "max_position_embeddings": 4096, "nope": "rope,nope", "softmax_tanh_gating": 30.0,
    "q_lora_rank": None, "qk_rope_head_dim": 8, "qk_nope_head_dim": 8, "kv_lora_rank": 16,
    "v_head_dim": 16, "ssmax_scaling": True, "lerope": True,
    "dsa": {"topk": 8, "n_heads": 2, "head_dim": 16, "rope_head_dim": 8, "match_nope": True,
            "ssmax_scaling": True, "norm_eps": 1e-06, "sharing": "full,full,shared,full",
            "fp8_indexer": True, "fp8_kv_cache": True},
    "yarn": {"original_max_position_embeddings": 512, "factor": 8, "apply_scale": False, "beta": 32, "alpha": 1},
    "kv_sharing": {"source_layers_str": "0,1,2,1"},
    "moe": {"expert_parallel": 1, "expert_model_parallel": 1, "route_every_n": 1, "first_k_dense_replace": 1,
            "num_experts": 2, "num_experts_per_tok": 1, "num_expert_groups": 1, "num_expert_groups_per_tok": 1,
            "routed_scale": 1.0, "expert_hidden_dim": 256, "num_shared_experts": 1},
    "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "group_size": 16,
                            "ignore": [], "quantized_layers": {"layers.1.experts": {"quant_algo": "NVFP4", "group_size": 16}}},
    "mtp": {"n_layers": 1, "depth": 3, "target_activation_layers": "1,3"},
}


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "params.json").write_text(json.dumps(PARAMS))
    (snap / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>", "eos_token": "</s>"}))
    (snap / "tokenizer.json").write_text(json.dumps({"added_tokens": [
        {"id": 0, "content": "<unk>"}, {"id": 1, "content": "<s>"}, {"id": 2, "content": "</s>"}]}))
    first = {"tok_embeddings.weight": ("BF16", [32, 64]), "lerope.alpha": ("BF16", [4])}
    for layer in range(4):
        rope = layer in (0, 2)
        p = f"layers.{layer}.attention."
        first[p + "wq.weight"] = ("F8_E4M3", [64, 64])
        first[p + "wkv_b.weight"] = ("BF16", [4 * ((8 if rope else 16) + 16), 16])
        if layer != 3:  # layer 3 reads layer 1's cache
            first[p + "wkv_a_with_mqa.weight"] = ("BF16", [24 if rope else 16, 64])
    second = {f"layers.{layer}.experts.{e}.w{w}.weight": ("U8", [256, 32])
              for layer in (1, 2, 3) for e in range(2) for w in (1, 2, 3)}
    second["output.weight"] = ("BF16", [32, 64])
    second["mtp_dspark.backbone.mask_embedding"] = ("BF16", [2, 64])
    weight_map = {}
    for i, tensors in enumerate((first, second), 1):
        name = f"consolidated-0000{i}-of-00002.safetensors"
        write_shard(snap / name, tensors)
        weight_map.update({t: name for t in tensors})
    (snap / "consolidated.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return snap


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", *args], capture_output=True, text=True, check=False)


def test_census_reports_components_per_layer_shapes_and_spark_fit(snapshot: Path, tmp_path: Path) -> None:
    out = tmp_path / "census"
    result = run(str(TOOLS / "ckpt_census.py"), "--local", str(snapshot), "--out", str(out),
                 "--spark-budget-gib", "0.000001")
    assert result.returncode == 0, result.stderr
    census = json.loads(out.with_suffix(".json").read_text())
    assert census["config_file"] == "params.json"
    assert census["index_file"] == "consolidated.safetensors.index.json"
    cats = census["bytes_by_category"]
    assert cats["routed_experts"] == 3 * 2 * 3 * 256 * 32
    assert cats["embed_head"] == 2 * 32 * 64 * 2
    assert cats["drafter"] == 2 * 64 * 2
    assert sum(cats.values()) == census["tensor_bytes"]
    # Per-layer variation is kept: RoPE layers carry a wider kv_a than NoPE layers.
    assert census["groups"]["layers.N.attention.wkv_a_with_mqa.weight"]["shapes"] == {"[24, 64]": "0,2", "[16, 64]": "1"}
    fit = census["spark_fit"]
    assert fit["expert_intermediate"] == 256
    assert [w["tp"] for w in fit["worlds"]] == [1, 2]  # 256 rows = two 128-row blocks
    assert fit["worlds"][1]["share"] == 0.5
    assert "| `layers.N.attention.wq.weight` | coordinator |" in out.with_suffix(".md").read_text()


def test_stub_keeps_true_sizes_and_headers_only(snapshot: Path, tmp_path: Path) -> None:
    stub = tmp_path / "stub"
    result = run(str(TOOLS / "ckpt_census.py"), "--local", str(snapshot), "--out", str(tmp_path / "c"),
                 "--stub-dir", str(stub))
    assert result.returncode == 0, result.stderr
    for shard in snapshot.glob("consolidated-*.safetensors"):
        real, copy = shard.read_bytes(), (stub / shard.name).read_bytes()
        assert len(copy) == len(real)
        n = struct.unpack("<Q", real[:8])[0]
        assert copy[:8 + n] == real[:8 + n] and not any(copy[8 + n:])
    assert (stub / "params.json").read_text() == (snapshot / "params.json").read_text()


def test_mistral_view_translates_params_and_links_every_file(snapshot: Path, tmp_path: Path) -> None:
    view = tmp_path / "view"
    result = run(str(TOOLS / "checkpoint_view.py"), "mistral", str(snapshot), str(view))
    assert result.returncode == 0, result.stderr
    config = json.loads((view / "config.json").read_text())
    assert config["model_type"] == "mistral_large_4"
    assert config["architectures"] == ["MistralLarge4ForCausalLM"]
    assert config["nope_layer_flags"] == [False, True, False, True]
    assert config["kv_source_layers"] == [0, 1, 2, 1]
    assert config["index_sharing"] == ["full", "full", "shared", "full"]
    assert config["indexer_nope_layer_flags"] == [False, True, False, True]
    assert config["scoring_func"] == "softmax" and config["norm_topk_prob"] is True
    assert config["n_routed_experts"] == 2 and config["moe_intermediate_size"] == 256
    rope = config["rope_parameters"]
    assert rope["rope_type"] == "yarn" and rope["attention_factor"] == 1.0 and rope["factor"] == 8.0
    assert config["quantization_config"]["quant_algo"] == "MIXED_PRECISION"
    assert config["mtp_config"]["target_activation_layers"] == "1,3"
    assert (config["bos_token_id"], config["eos_token_id"]) == (1, 2)
    index = view / "model.safetensors.index.json"
    assert index.stat().st_ino == (snapshot / "consolidated.safetensors.index.json").stat().st_ino
    for shard in snapshot.glob("consolidated-*.safetensors"):
        assert (view / shard.name).stat().st_ino == shard.stat().st_ino


@pytest.mark.parametrize("change", [
    {"unknown_key": 1},
    {"kv_sharing": {"source_layers_str": "0,1,2,2"}},  # layer 3 (NoPE) cannot share a RoPE layer's cache
])
def test_mistral_view_refuses_what_it_does_not_understand(snapshot: Path, tmp_path: Path, change: dict) -> None:
    (snapshot / "params.json").write_text(json.dumps({**PARAMS, **change}))
    result = run(str(TOOLS / "checkpoint_view.py"), "mistral", str(snapshot), str(tmp_path / "view"))
    assert result.returncode != 0


def test_image_reference_parsing() -> None:
    spec = importlib.util.spec_from_file_location("image_ref_extract", TOOLS / "image_ref_extract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parse = module.parse_ref
    assert parse("vllm/vllm-openai:mistral-large-4") == ("registry-1.docker.io", "vllm/vllm-openai", "mistral-large-4")
    assert parse("ubuntu") == ("registry-1.docker.io", "library/ubuntu", "latest")
    assert parse("docker.io/lmsysorg/sglang:v1") == ("registry-1.docker.io", "lmsysorg/sglang", "v1")
    assert parse("ghcr.io/org/img@sha256:abc") == ("ghcr.io", "org/img", "sha256:abc")
    assert parse("localhost:5000/img:tag") == ("localhost:5000", "img", "tag")
