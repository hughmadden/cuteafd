#!/usr/bin/env python3
"""Qualify real K3 decoder boundaries and one full-size K2 projection."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "third_party/gptqmodel"))
sys.path.insert(0, str(REPO / "quantization"))
os.environ.setdefault("FLA_DISABLE_TENSOR_CACHE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import psutil
import torch
from torch.nn import functional as F
from gptqmodel.utils.kimi_k3 import KimiK3Source, prepare_moe, finish_moe
from kimi_k3.prepare_corpus import load_token_records
from kimi_k3.uniform_k2.run import quantize_projection
from kimi_k3.uniform_k2.storage import atomic_json


def memory():
    return dict(rss_gib=psutil.Process().memory_info().rss / 1024**3,
                cuda_allocated_gib=torch.cuda.memory_allocated() / 1024**3,
                cuda_peak_gib=torch.cuda.max_memory_allocated() / 1024**3)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = KimiK3Source(args.source_snapshot)
    record = load_token_records(args.tokens)[0]
    ids = torch.tensor(record["input_ids"][:128], dtype=torch.long, device="cuda:0")
    embedding = source.tensor("language_model.model.embed_tokens.weight", "cuda:0")
    hidden = F.embedding(ids, embedding).unsqueeze(0)
    del embedding
    residual = hidden.new_empty((ids.numel(), 0, 7168))
    report = dict(source_revision=source.snapshot.name, start=memory(), layers=[])
    for layer in (0, 1):
        start = time.monotonic()
        block = source.load_block(layer)
        print(json.dumps(dict(event="layer_loaded", layer=layer, seconds=time.monotonic()-start, **memory())), flush=True)
        expected, expected_bank = block(hidden, block_residual=residual, use_cache=False)
        prefix, bank, mlp_input = prepare_moe(block, hidden, residual)
        output = block.mlp(mlp_input) if layer == 0 else block.block_sparse_moe(mlp_input)
        actual, actual_bank = finish_moe(prefix, bank, output)
        if not torch.equal(actual, expected) or not torch.equal(actual_bank, expected_bank):
            raise ValueError("split decoder differs from native checkpoint")
        if not bool(torch.isfinite(actual).all()):
            raise ValueError("nonfinite decoder output")
        report["layers"].append(dict(layer=layer, byte_exact=True, **memory()))
        if layer == 1:
            selected, _ = block.block_sparse_moe.gate(mlp_input)
            counts = torch.bincount(selected.flatten(), minlength=896)
            expert = int(counts.argmax())
            rows = block.block_sparse_moe.routed_expert_down_proj(mlp_input.reshape(-1, 7168))
            selected_rows = rows[(selected == expert).any(dim=-1)].float()
            hessian = selected_rows.T @ selected_rows
            missing = max(0, 1024 - selected_rows.shape[0])
            hessian.diagonal().add_(float(missing))
            start = time.monotonic()
            packed, metrics = quantize_projection(block.block_sparse_moe.experts[expert].w1.weight,
                                                  dict(H=hessian, count=selected_rows.shape[0]+missing, finalized=False))
            elapsed = time.monotonic() - start
            if tuple(packed["trellis"].shape) != (224, 192, 32):
                raise ValueError("K2 packed projection geometry differs")
            raw = hessian.clone()
            up, up_metrics = quantize_projection(block.block_sparse_moe.experts[expert].w3.weight,
                                                 dict(H=hessian, count=selected_rows.shape[0]+missing, finalized=False))
            repeated, _ = quantize_projection(block.block_sparse_moe.experts[expert].w3.weight,
                                              dict(H=hessian, count=selected_rows.shape[0]+missing, finalized=False))
            if not torch.equal(raw, hessian) or any(not torch.equal(value, repeated[key]) for key, value in up.items()):
                raise ValueError("raw Hessian/restarted projection reproducibility differs")
            report["projection"] = dict(expert=expert, seconds=elapsed, metrics=metrics,
                                        up_metrics=up_metrics, resume_byte_exact=True,
                                        indicative_projection_days=elapsed * 247296 / 86400, **memory())
            print(json.dumps(dict(event="projection_probe", **report["projection"])), flush=True)
        hidden, residual = actual, actual_bank
        del block, expected, expected_bank, prefix, bank, mlp_input, output, actual, actual_bank
        gc.collect()
        torch.cuda.empty_cache()
    report["status"] = "passed"
    atomic_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
