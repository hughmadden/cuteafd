"""Pinned real-weight layer-3 CSA attention branch; no experts or cache publication."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "step5_csa_golden", ROOT / "python/reference/families/step5/golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)
ROWS = (1, 7, 8, 9, 128, 513, 4105)


def test_official_layer3_csa_fixtures():
    destination = os.environ.get("CUTEAFD_STEP5_CSA_FIXTURE_DIR")
    if not destination:
        pytest.skip("set CUTEAFD_STEP5_CSA_FIXTURE_DIR for real-weight CUDA fixtures")
    torch = pytest.importorskip("torch")
    root = Path(destination).resolve()
    assert root.is_relative_to(Path(os.environ["CARGO_TARGET_DIR"]).resolve())
    root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(os.environ["STEP5_REFERENCE_SNAPSHOT"])
    ref, kernel = golden.load_release(snapshot)
    values = json.loads((snapshot / "config.json").read_text())
    config_values = dict(values.get("text_config", values))
    config_values["tp_size"] = 1
    config = ref.Step5Config(**config_values)
    with torch.device("meta"):
        attention = ref.Step5Attention(config, 3, dtype=torch.bfloat16)
        norm = ref.Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
    attention.to_empty(device="cuda")
    norm.to_empty(device="cuda")
    reader = golden.Weights(snapshot)
    sources = {}
    with torch.no_grad():
        for prefix, module in [("self_attn.", attention), ("input_layernorm.", norm)]:
            for name, parameter in module.named_parameters():
                key = "model.layers.3." + prefix + name
                value = reader.get(key)
                assert value.dtype == parameter.dtype and value.shape == parameter.shape
                sources[key] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
                parameter.copy_(value)
    reader.files.clear()
    attention.build_rope(max_position=max(ROWS), device=torch.device("cuda"))
    generator = torch.Generator(device="cpu").manual_seed(274)
    residual = torch.randn((max(ROWS), config.hidden_size), generator=generator).cuda()
    manifest = {"schema": 1, "checkpoint": golden.CHECKPOINT, "revision": golden.REVISION,
                "source_pins": golden.SOURCE_PINS, "seed": 274, "layer": 3,
                "rows": list(ROWS), "epsilon": config.rms_norm_eps, "query_tile": 64,
                "weights_sha256": sources, "torch": torch.__version__, "cuda": torch.version.cuda,
                "files": {}, "cases": {}, "scope": "attention branch only; layer3 FFN is routed",
                "attention_error_ratio_bound": 1.01, "gemm_error_ratio_bound": 1.01,
                "norm_error_ratio_bound": 1.01, "norm_row_max_ulp_slack": 1,
                "norm_oracle": "unrounded FP64 RMS/LayerNorm; native input/affine dtypes; BF16 normalization before RoPE",
                "score_relative_l2_bound": 1e-5, "score_max_abs_bound": 1e-4,
                "attention_oracle": "same BF16 QKV, FP32 scale; online tiles32 BF16 P, unrounded denominator"}

    def save(name, tensor):
        value = tensor.detach().contiguous().cpu()
        data = value.view(torch.uint8).numpy().tobytes()
        (root / (name + ".bin")).write_bytes(data)
        manifest["files"][name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                                   "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    with torch.inference_mode():
        for rows in ROWS:
            start = rows - 1 if rows == 4105 else 0
            count = rows - start
            regions = rows // 8
            positions = torch.arange(rows, device="cuda", dtype=torch.int32)
            qpositions = positions[start:]
            x = residual[:rows].contiguous()
            hidden = norm(x).to(torch.bfloat16)
            a = attention
            packed_qkv = a._project_packed_qkv(hidden)
            qraw, kraw, value = a._split_packed_qkv(packed_qkv)
            query, key, value = a._project(hidden, positions)
            iqraw = a.sparse_indexer_q(hidden)
            ikraw = a.sparse_indexer_k(hidden)
            iz = a.sparse_indexer_z(hidden)
            iq, ik, iz, weights = a.indexer.project(hidden, positions)
            save(f"r{rows}_norm_fp64", golden.norm_accuracy_fp64(x, norm.weight, config.rms_norm_eps))
            for name, operand, weight, bias, centered, mean, cos, sin in [
                ("query", qraw, a.q_norm.weight, None, True, False, a.rope_cos, a.rope_sin),
                ("key", kraw, a.k_norm.weight, None, True, False, a.rope_cos, a.rope_sin),
                ("iq", iqraw.view(rows, 16, 256), a.sparse_indexer_q_norm.weight.to(torch.bfloat16),
                 None, True, False, a.sparse_indexer_rope_cos, a.sparse_indexer_rope_sin),
                ("ik", ikraw.view(rows, 1, 256), a.sparse_indexer_k_norm.weight.to(torch.bfloat16),
                 a.sparse_indexer_k_norm.bias.to(torch.bfloat16), False, True,
                 a.sparse_indexer_rope_cos, a.sparse_indexer_rope_sin),
            ]:
                save(f"r{rows}_{name}_fp64", golden.norm_accuracy_fp64(operand, weight,
                    config.rms_norm_eps, zero_centered=centered, bias=bias, subtract_mean=mean,
                    cos=cos[positions], sin=sin[positions]))
            logits = a.g_proj(hidden[start:])
            cache = kernel.DSALayerCache(batch=1, max_tokens=rows, num_kv_groups=4,
                head_dim=192, geometry=a.indexer.geometry, device=x.device, dtype=torch.bfloat16)
            cache.write_kv(0, 0, key, value)
            kernel.update_summaries_prefill(cache, ik, iz, torch.tensor([rows], device=x.device, dtype=torch.int32))
            summaries = cache.summary[:rows // 8].view(-1, 1, 256)
            # A dummy invisible column makes zero-complete-region cases legal GEMMs.
            score_keys = cache.summary[:max(regions, 1)].to(torch.bfloat16)
            if regions == 0:
                score_keys = torch.zeros_like(score_keys)
            scores = kernel.indexer_logits(iq[start:], weights[start:], score_keys)
            visible = qpositions[:, None] // 8
            scores = scores.view(4, count, -1).transpose(0, 1).contiguous()
            scores.masked_fill_(torch.arange(scores.shape[-1], device=x.device)[None, None, :] >= visible[:, None], float("-inf"))
            group_major = scores.transpose(0, 1).contiguous().view(4 * count, -1)
            packed, counts = kernel.prefill_sparse_meta(group_major, qpositions.repeat(4), cache.block_table,
                torch.zeros(4 * count, device=x.device, dtype=torch.int32), topk=512,
                region_size=8, regions_per_page=a.indexer.geometry.regions_per_page)
            selected = torch.full((count, 4, 4104), -1, device=x.device, dtype=torch.int32)
            metadata = packed.view(4, count, 513).transpose(0, 1)
            # Official packs current immediately after live history; the native
            # ABI reserves slot512 for current, leaving history padding in place.
            current_slot = (qpositions // 8).clamp_max(512)
            for slot in range(513):
                word = metadata[:, :, slot]
                offsets = torch.arange(8, device=x.device)
                tokens = (word & 0xFFFFFF).unsqueeze(-1) * 8 + offsets
                keep = (word >= 0).unsqueeze(-1) & (offsets < (word >> 24).unsqueeze(-1))
                history = (slot < current_slot)[:, None, None]
                if slot < 512:
                    selected[:, :, slot * 8:(slot + 1) * 8] = torch.where(keep & history, tokens, -1)
                current = (slot == current_slot)[:, None, None]
                selected[:, :, 4096:] = torch.where(keep & current, tokens, selected[:, :, 4096:])
            offsets = torch.arange(8, device=x.device)
            original_tokens = (metadata & 0xFFFFFF).unsqueeze(-1) * 8 + offsets
            original_keep = (metadata >= 0).unsqueeze(-1) & (offsets < (metadata >> 24).unsqueeze(-1))
            original_tokens = torch.where(original_keep, original_tokens, -1).reshape(count, 4, 4104)
            assert torch.equal(selected.sort(-1).values, original_tokens.sort(-1).values), "selection set changed during ABI normalization"
            assert int((selected >= rows).sum()) == 0
            if rows == 4105:
                history = selected[0, :, :4096].reshape(4, 512, 8)[:, :, 0] // 8
                assert (history >= 0).all() and (history < 513).all()
                assert all(len(set(v.tolist())) == 512 for v in history)
                assert torch.equal(selected[0, :, 4096], torch.full((4,), 4104, device=x.device))
            output, _ = kernel.sparse_attention_prefill(query[start:], cache.key, cache.value,
                packed, counts, num_kv_groups=4, region_size=8, softmax_scale=a.scaling)
            gated = output * logits.sigmoid().unsqueeze(-1)
            delta = a.o_proj(gated.reshape(count, -1))
            post = x[start:] + delta.float()
            manifest["cases"][str(rows)] = {"query_start": start, "queries": count, "regions": regions,
                                            "score_columns": scores.shape[-1]}
            for name, tensor in {"residual": x, "norm": hidden, "qraw": qraw, "kraw": kraw,
                "value": value, "query": query, "key": key, "iqraw": iqraw, "ikraw": ikraw,
                "iz": iz, "iq": iq, "ik": ik, "iw": weights, "summary": summaries,
                "scores": scores, "selected": selected, "logits": logits,
                "attention": output, "gated": gated, "delta": delta, "post": post}.items():
                save(f"r{rows}_{name}", tensor)
            # Bound projection oracles to live queries; history-only producer rows
            # are still recorded and checked for deterministic eager/replay execution.
            for name, operand, weight in [("qraw", hidden[start:], a.q_proj.weight),
                ("kraw", hidden[start:], a.k_proj.weight), ("value", hidden[start:], a.v_proj.weight),
                ("logits", hidden[start:], a.g_proj.weight), ("iqraw", hidden[start:], a.sparse_indexer_q.weight),
                ("ikraw", hidden[start:], a.sparse_indexer_k.weight), ("iz", hidden[start:], a.sparse_indexer_z.weight),
                ("delta", gated.reshape(count, -1), a.o_proj.weight)]:
                save(f"r{rows}_{name}_fp64", torch.nn.functional.linear(operand.double(), weight.double()))
            save(f"r{rows}_iw_fp64", torch.nn.functional.linear(hidden[start:].double(),
                a.sparse_indexer_w.weight.to(torch.bfloat16).double()) * 0.5)
            full, rounded = [], []
            scale = float(torch.tensor(a.scaling, dtype=torch.float32))
            # The selected order is load-bearing for online probability rounding.
            for first in range(0, count, 64):
                last = min(first + 64, count)
                local = selected[first:last]
                result64, result_p = [], []
                for group in range(4):
                    q64 = query[start + first:start + last, group * 16:(group + 1) * 16].double().transpose(0, 1)
                    k64, v64 = key[:, group].double(), value[:, group].double()
                    all_scores = (q64 @ k64.T) * scale
                    keep = torch.zeros((last - first, rows), device=x.device, dtype=torch.bool)
                    for row in range(last - first):
                        keep[row, local[row, group][local[row, group] >= 0].long()] = True
                    masked = all_scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
                    result64.append((masked.softmax(-1) @ v64).transpose(0, 1))
                    maximum = torch.full((16, last - first, 1), float("-inf"), device=x.device, dtype=torch.float64)
                    denominator = torch.zeros_like(maximum)
                    numerator = torch.zeros((16, last - first, 192), device=x.device, dtype=torch.float64)
                    for column in range(0, 4104, 32):
                        ids = local[:, group, column:column + 32].long()
                        if not bool((ids >= 0).any()):
                            continue
                        valid, ids = ids >= 0, ids.clamp_min(0)
                        tile = all_scores.gather(2, ids.unsqueeze(0).expand(16, -1, -1))
                        tile.masked_fill_(~valid.unsqueeze(0), float("-inf"))
                        next_max = torch.maximum(maximum, tile.amax(-1, keepdim=True))
                        safe_max = next_max.masked_fill(torch.isneginf(next_max), 0.)
                        rescale = torch.exp(maximum - safe_max)
                        probability = torch.exp(tile - safe_max)
                        denominator = denominator * rescale + probability.sum(-1, keepdim=True)
                        numerator = numerator * rescale + torch.einsum("hqt,qtd->hqd",
                            probability.to(torch.bfloat16).double(), v64[ids])
                        maximum = next_max
                    result_p.append((numerator / denominator).transpose(0, 1).to(torch.bfloat16))
                full.append(torch.cat(result64, dim=1))
                rounded.append(torch.cat(result_p, dim=1))
            save(f"r{rows}_attention_fp64", torch.cat(full))
            save(f"r{rows}_attention_bf16_p", torch.cat(rounded))
            print(f"CSA fixture rows={rows} queries={count} complete_regions={regions}", flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
