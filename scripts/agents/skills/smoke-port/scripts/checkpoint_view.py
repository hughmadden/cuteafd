#!/usr/bin/env python3
"""Present a non-HF checkpoint to cuteafd as an HF-style snapshot view.

cuteafd reads `config.json` and `model.safetensors.index.json` in many places.
A checkpoint shipped in another layout (Mistral-native: `params.json` +
`consolidated.safetensors.index.json` + `consolidated-*.safetensors`) can be
planned and served on day 0 through a *view*: a directory of hard links (or
symlinks) to every shipped file, plus

* `config.json` translated from the native config (no information dropped:
  the native keys the family needs are carried under their HF-style names,
  and the original file is linked beside it), and
* the shard index under its HF name (same content, same shard names).

No weight byte is copied, renamed or rewritten; tensor names stay as shipped
and the family maps them. This is the smoke-port seam. The release seam is a
native reader in `cuteafd-loader`, after which the view is deleted.

    checkpoint_view.py mistral SNAPSHOT VIEW_DIR [--symlink]

Adapters: `mistral` (params.json with MLA / DSA / MoE / NoPE / KV sharing /
YaRN / DSpark / vision blocks; mirrors vLLM's
`transformers_utils/configs/mistral.py` `adapt_config_dict` for those blocks).
Standard library only. Run it isolated: `python3 -I checkpoint_view.py ...`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


def expand(pattern: list, n: int) -> list:
    assert pattern, "per-layer patterns must not be empty"
    return (pattern * math.ceil(n / len(pattern)))[:n]


def mistral_config(params: dict, tokenizer_config: dict | None) -> dict:
    p = dict(params)
    c: dict = {}
    rename = {"dim": "hidden_size", "n_layers": "num_hidden_layers", "n_heads": "num_attention_heads",
              "n_kv_heads": "num_key_value_heads", "hidden_dim": "intermediate_size", "norm_eps": "rms_norm_eps",
              "head_dim": "head_dim", "vocab_size": "vocab_size", "max_position_embeddings": "max_position_embeddings",
              "q_lora_rank": "q_lora_rank", "kv_lora_rank": "kv_lora_rank", "qk_rope_head_dim": "qk_rope_head_dim",
              "qk_nope_head_dim": "qk_nope_head_dim", "v_head_dim": "v_head_dim"}
    for k, v in rename.items():
        if k in p:
            c[v] = p.pop(k)
    c["tie_word_embeddings"] = bool(p.pop("tied_embeddings", False))
    c["hidden_act"] = p.pop("activation", "silu")
    n = c["num_hidden_layers"]

    if (yarn := p.pop("yarn", None)) is not None:
        yarn = dict(yarn)
        rp = {"rope_type": "yarn", "rope_theta": p.pop("rope_theta", 10000.0), "mscale_all_dim": 1,
              "factor": float(yarn.pop("factor")),
              "original_max_position_embeddings": int(yarn.pop("original_max_position_embeddings")),
              "beta_fast": float(yarn.pop("beta", 32)), "beta_slow": float(yarn.pop("alpha", 1))}
        if not yarn.pop("apply_scale", True):
            rp["attention_factor"] = 1.0
        assert not yarn, f"unparsed yarn keys: {yarn}"
        c["rope_parameters"] = rp
    else:
        c["rope_parameters"] = {"rope_type": "default", "rope_theta": p.pop("rope_theta", 10000.0)}

    if (moe := p.pop("moe", None)) is not None:
        moe = dict(moe)
        for src, dst in {"route_every_n": "moe_layer_freq", "first_k_dense_replace": "first_k_dense_replace",
                         "num_experts_per_tok": "num_experts_per_tok", "num_experts": "n_routed_experts",
                         "expert_hidden_dim": "moe_intermediate_size", "routed_scale": "routed_scaling_factor",
                         "num_shared_experts": "n_shared_experts", "num_expert_groups": "n_group",
                         "num_expert_groups_per_tok": "topk_group"}.items():
            if src in moe:
                c[dst] = moe.pop(src)
        for k in ("expert_parallel", "expert_model_parallel"):  # training-time layout, no serving meaning
            moe.pop(k, None)
        assert not moe, f"unparsed moe keys: {moe}"
        c.update(scoring_func="softmax", norm_topk_prob=True, topk_method=None)

    if (nope := p.pop("nope", None)):
        c["nope_layer_flags"] = expand([v == "nope" for v in nope.split(",")], n)
    else:
        c["nope_layer_flags"] = [False] * n
    sources = list(range(n))
    if (kvs := p.pop("kv_sharing", None) or {}).get("source_layers_str"):
        sources = [int(v) for v in kvs["source_layers_str"].split(",")]
    assert len(sources) == n
    for layer, src in enumerate(sources):
        assert src <= layer and sources[src] == src, f"layer {layer} must share KV with an earlier owner"
        assert c["nope_layer_flags"][layer] == c["nope_layer_flags"][src]
    c["kv_source_layers"] = sources

    if (dsa := p.pop("dsa", None)) is not None:
        aliases = {"topk": "index_topk", "n_heads": "index_n_heads", "head_dim": "index_head_dim",
                   "rope_head_dim": "index_rope_head_dim", "nope": "index_nope", "match_nope": "indexer_match_nope",
                   "ssmax_scaling": "indexer_ssmax_scaling", "norm_eps": "indexer_norm_eps", "sharing": "index_sharing"}
        dsa = {aliases.get(k, k): v for k, v in dsa.items()}
        for k in ("index_topk", "index_n_heads", "index_head_dim", "indexer_norm_eps", "indexer_ssmax_scaling"):
            c[k] = dsa.pop(k)
        assert dsa.pop("index_rope_head_dim", None) in (None, -1, c.get("qk_rope_head_dim"))
        assert not dsa.pop("index_nope", False), "index_nope is not supported"
        c["indexer_nope_layer_flags"] = list(c["nope_layer_flags"]) if dsa.pop("indexer_match_nope", False) else [False] * n
        c["indexer_fp8"] = bool(dsa.pop("fp8_indexer", True))
        c["fp8_kv_cache"] = bool(dsa.pop("fp8_kv_cache", False))
        sharing = dsa.pop("index_sharing", None)
        c["index_sharing"] = expand(sharing.split(","), n) if sharing else ["full"] * n
        assert set(c["index_sharing"]) <= {"full", "shared"} and c["index_sharing"][0] == "full"
        assert not dsa, f"unparsed dsa keys: {dsa}"

    for k in ("ssmax_scaling", "lerope", "softmax_tanh_gating", "llama_4_scaling", "ragged_attention"):
        if k in p:
            c[k] = p.pop(k)
    if (mtp := p.pop("mtp", None)) is not None:
        c["mtp_config"] = mtp
    if (vision := p.pop("vision_encoder", None)) is not None:
        c["vision_config"] = vision
    if (q := p.pop("quantization_config", None)) is not None:
        c["quantization_config"] = q
    assert not p, f"unparsed params.json keys: {sorted(p)}"

    is_dsa = "index_topk" in c
    is_mla = bool(c.get("qk_nope_head_dim"))
    if is_dsa:
        assert is_mla and c.get("n_shared_experts"), "Mistral DSA requires MLA + MoE"
        c["model_type"], c["architectures"] = "mistral_large_4", ["MistralLarge4ForCausalLM"]
    elif is_mla and c.get("n_shared_experts"):
        c["model_type"], c["architectures"] = "deepseek_v3", ["MistralLarge3ForCausalLM"]
    else:
        c["model_type"] = "mistral"
    if tokenizer_config:
        c["bos_token_id"], c["eos_token_id"] = 1, 2  # tekken: <s>=1, </s>=2 (checked below)
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter", choices=["mistral"])
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("view", type=Path)
    ap.add_argument("--symlink", action="store_true", help="symlink instead of hard-link (crossing filesystems)")
    a = ap.parse_args()

    src, view = a.snapshot.resolve(), a.view
    assert not (src / "config.json").exists(), "snapshot already has config.json; use it directly"
    params_path = src / "params.json"
    params = json.loads(params_path.read_text())
    tok_cfg = json.loads((src / "tokenizer_config.json").read_text()) if (src / "tokenizer_config.json").is_file() else None
    if tok_cfg and (src / "tokenizer.json").is_file():
        added = {t["content"]: t["id"] for t in json.loads((src / "tokenizer.json").read_text())["added_tokens"]}
        assert added.get(tok_cfg.get("bos_token")) == 1 and added.get(tok_cfg.get("eos_token")) == 2, \
            "bos/eos ids are not tekken's 1/2; extend the adapter"
    config = mistral_config(params, tok_cfg)
    config["checkpoint_view"] = {"adapter": a.adapter, "source": str(src),
                                 "params_sha256": hashlib.sha256(params_path.read_bytes()).hexdigest()}

    view.mkdir(parents=True, exist_ok=False)
    for path in sorted(src.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        dst = view / path.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        (os.symlink if a.symlink else os.link)(path, dst)
    index = view / "consolidated.safetensors.index.json"
    if index.is_file():
        (os.symlink if a.symlink else os.link)(src / index.name, view / "model.safetensors.index.json")
    (view / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    print(f"{view}: {config['model_type']} view of {src} ({sum(1 for _ in view.iterdir())} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
