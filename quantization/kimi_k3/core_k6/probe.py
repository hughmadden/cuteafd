#!/usr/bin/env python3
"""Qualify native KDA/MLA staged replay against complete checkpoint blocks."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("KIMI_GPTQMODEL_SOURCE", str(REPO / "third_party/gptqmodel")))
sys.path.insert(0, str(REPO / "quantization"))
os.environ.setdefault("FLA_DISABLE_TENSOR_CACHE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import psutil
import torch
from torch import nn
from torch.nn import functional as F

from gptqmodel.utils.kimi_k3_core_format import (KimiK3EXL3Source,
    quantize_core_projection, reconstruct_core_projection)
from gptqmodel.utils.kimi_k3_core_stream import (DenseRawHessian, attention_inputs,
    attention_finish, capture_linear_input, shared_down_input, routed_up_input,
    complete_moe, final_head_input)
from kimi_k3.prepare_corpus import load_token_records
from kimi_k3.uniform_k2.storage import atomic_json

GIB = 1024**3


def memory():
    return dict(rss_gib=psutil.Process().memory_info().rss/GIB,
                open_files=psutil.Process().num_fds(),
                cuda_allocated_gib=torch.cuda.memory_allocated()/GIB,
                cuda_peak_gib=torch.cuda.max_memory_allocated()/GIB)


def event(name, **fields):
    record = dict(event=name, **fields)
    record.update(memory())
    print(json.dumps(record), flush=True)


def clear():
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(86*GIB / torch.cuda.get_device_properties(0).total_memory)
    source = KimiK3EXL3Source(args.source_snapshot)
    record = load_token_records(args.tokens)[0]
    ids = torch.tensor(record["input_ids"][:128], device="cuda:0")
    embedding = source.tensor("language_model.model.embed_tokens.weight", "cuda:0")
    hidden = F.embedding(ids, embedding).unsqueeze(0)
    del embedding
    clear()
    report = dict(layers=[], device=torch.cuda.get_device_name(), tokens=ids.numel())
    # Reset, KDA+K2 MoE, and MLA+K2 MoE. Separate native blocks use the same
    # input stream and compare exact arithmetic within each block.
    for layer in (0, 1, 3):
        started = time.monotonic()
        bank_count = 0 if layer == 0 else 1
        bank = hidden.reshape(-1, 7168).unsqueeze(1).repeat(1, bank_count, 1)
        block = source.load_block(layer)
        event("block_loaded", layer=layer, seconds=time.monotonic()-started)
        normalized, residual, next_bank = attention_inputs(block, hidden, bank)
        projection = block.self_attn.q_proj if block.is_linear_attn else block.self_attn.kv_a_proj_with_mqa
        hessian = DenseRawHessian(7168)
        hessian.add(normalized)
        raw = hessian.raw_dict()
        oracle = normalized.reshape(-1, 7168).double().T @ normalized.reshape(-1, 7168).double()
        error = float((raw["H"].double()-oracle).norm()/oracle.norm())
        if not error < 1e-6:
            raise ValueError(f"FP64 Hessian oracle differs: {error}")
        del oracle
        # One ordinary full-size K6 matrix and one padded MLA matrix. Layer0
        # shares the KDA path already exercised by layer1.
        if layer:
            packed, metrics, evidence = quantize_core_projection(projection.weight, raw)
            projection.weight = nn.Parameter(reconstruct_core_projection(packed, tuple(projection.weight.shape)), requires_grad=False)
            event("projection_quantized", layer=layer, evidence=evidence, metrics=metrics)
            del packed, metrics, evidence
        del hessian, raw
        clear()
        o_input = capture_linear_input(block.self_attn.o_proj, block.self_attn,
                                       hidden_states=normalized, attention_mask=None, use_cache=False)
        attention = block.self_attn.o_proj(o_input)
        prefix, next_bank, ffn_input = attention_finish(block, residual, next_bank, attention)
        if layer:
            try:
                source.materialize_routed_experts(block)
            except BaseException:
                event("expert_load_failed", layer=layer)
                raise
            event("experts_loaded", layer=layer)
            shared = shared_down_input(block.block_sparse_moe.shared_experts, ffn_input)
            routed = routed_up_input(block.block_sparse_moe, ffn_input)
            actual, actual_bank = complete_moe(prefix, next_bank, shared, routed, block.block_sparse_moe)
            del shared, routed
        else:
            actual, actual_bank = prefix + block.mlp(ffn_input), next_bank
        expected, expected_bank = block(hidden, block_residual=bank, use_cache=False)
        if not torch.equal(actual, expected) or not torch.equal(actual_bank, expected_bank):
            raise ValueError(f"layer {layer} staged decoder differs from complete native block")
        if not bool(torch.isfinite(actual).all()):
            raise ValueError("nonfinite native stage replay")
        report["layers"].append(dict(layer=layer, byte_exact=True, hessian_fp64_relative_error=error,
                                     seconds=time.monotonic()-started, **memory()))
        event("native_replay_passed", **report["layers"][-1])
        hidden = actual.clone()
        del block, expected, expected_bank, actual, actual_bank, prefix, bank, next_bank, normalized, residual, attention, o_input, ffn_input, projection
        clear()
    # Check final model head input without allocating vocabulary logits.
    ref = source.reference
    cfg = source.text_config
    model_norm = ref.KimiRMSNorm(7168, eps=cfg["rms_norm_eps"]).to(device="cuda:0", dtype=torch.bfloat16)
    mixer_norm = ref.KimiRMSNorm(7168, eps=cfg["rms_norm_eps"]).to(device="cuda:0", dtype=torch.bfloat16)
    mixer = nn.Linear(7168, 1, bias=False, device="cuda:0", dtype=torch.bfloat16)
    model_norm.weight = nn.Parameter(source.tensor("language_model.model.norm.weight", "cuda:0"), requires_grad=False)
    mixer_norm.weight = nn.Parameter(source.tensor("language_model.model.output_attn_res_norm.weight", "cuda:0"), requires_grad=False)
    mixer.weight = nn.Parameter(source.tensor("language_model.model.output_attn_res_proj.weight", "cuda:0"), requires_grad=False)
    bank = hidden.reshape(-1, 7168).unsqueeze(1).repeat(1,8,1)
    expected = model_norm(ref._apply_attn_res(hidden.view(-1,7168), bank, mixer, mixer_norm).view_as(hidden))
    actual = final_head_input(hidden, bank, output_attn_res_proj=mixer, output_attn_res_norm=mixer_norm,
                             model_norm=model_norm, apply_attn_res=ref._apply_attn_res)
    if not torch.equal(expected, actual):
        raise ValueError("final AttnRes/norm input differs")
    report.update(status="passed", final_head_byte_exact=True, memory=memory())
    source.close()
    atomic_json(args.output, report)
    event("probe_passed", output=str(args.output))


if __name__ == "__main__":
    main()
