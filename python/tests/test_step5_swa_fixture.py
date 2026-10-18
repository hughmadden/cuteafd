"""Optional pinned official layer-0 prefill oracle and same-input FP64 attention."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "step5_swa_golden", ROOT / "python/reference/families/step5/golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)
ROWS = (1, 7, 128, 513)


def test_official_layer0_swa_fixtures():
    destination = os.environ.get("CUTEAFD_STEP5_SWA_FIXTURE_DIR")
    if not destination:
        pytest.skip("set CUTEAFD_STEP5_SWA_FIXTURE_DIR for real-weight CUDA fixtures")
    torch = pytest.importorskip("torch")
    root = Path(destination).resolve()
    assert root.is_relative_to(Path(os.environ["CARGO_TARGET_DIR"]).resolve())
    root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(os.environ["STEP5_REFERENCE_SNAPSHOT"])
    ref, _ = golden.load_release(snapshot)
    values = json.loads((snapshot / "config.json").read_text())
    config_values = dict(values.get("text_config", values))
    config_values["tp_size"] = 1
    config = ref.Step5Config(**config_values)
    with torch.device("meta"):
        layer = ref.Step5DecoderLayer(config, 0, dtype=torch.bfloat16)
    layer.to_empty(device="cuda")
    reader = golden.Weights(snapshot)
    sources = {}
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            value = reader.get("model.layers.0." + name)
            assert value.dtype == parameter.dtype and value.shape == parameter.shape
            sources[name] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
            parameter.copy_(value)
    reader.files.clear()
    # to_empty preserves the shape-zero rotary buffers; build the official recipe.
    layer.self_attn.build_rope(max_position=max(ROWS), device=torch.device("cuda"))
    generator = torch.Generator(device="cpu").manual_seed(274)
    residual = torch.randn((max(ROWS), config.hidden_size), generator=generator).cuda()
    manifest = {"schema": 1, "checkpoint": golden.CHECKPOINT, "revision": golden.REVISION,
                "source_pins": golden.SOURCE_PINS, "seed": 274, "layer": 0,
                "rows": list(ROWS), "epsilon": config.rms_norm_eps,
                "weights_sha256": sources, "torch": torch.__version__,
                "cuda": torch.version.cuda, "files": {},
                "attention_official_sdpa_ratio": "reported_only_bf16_p_precision_differs",
                "attention_bf16_p_error_ratio_bound": 1.01,
                "attention_bf16_p": "FP64 online tiles32; unnormalized P BF16; denominator unrounded; output BF16",
                "gemm_error_ratio_bound": 1.01,
                "norm_error_ratio_bound": 1.01, "norm_row_max_ulp_slack": 1}

    def save(name, tensor):
        value = tensor.detach().contiguous().cpu()
        data = value.view(torch.uint8).numpy().tobytes()
        (root / (name + ".bin")).write_bytes(data)
        manifest["files"][name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                                    "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    with torch.inference_mode():
        for rows in ROWS:
            x = residual[:rows].contiguous()
            positions = torch.arange(rows, device="cuda", dtype=torch.int32)
            seq_lens = torch.tensor([rows], device="cuda", dtype=torch.int32)
            past_lens = torch.zeros_like(seq_lens)
            a = layer.self_attn
            hidden = layer.input_layernorm(x).to(torch.bfloat16)
            packed = a._project_packed_qkv(hidden)
            qraw, kraw, v = a._split_packed_qkv(packed)
            query, key, value = a._project(hidden, positions)
            save(f"r{rows}_norm_fp64", golden.norm_accuracy_fp64(
                x, layer.input_layernorm.weight, config.rms_norm_eps))
            for name, operand, weight in [("query", qraw, a.q_norm.weight), ("key", kraw, a.k_norm.weight)]:
                save(f"r{rows}_{name}_fp64", golden.norm_accuracy_fp64(operand, weight,
                    config.rms_norm_eps, cos=a.rope_cos[positions], sin=a.rope_sin[positions]))
            attention = a._sliding_prefill_chunked(query, key, value, 512)
            logits = a.g_proj(hidden)
            gated = attention * logits.sigmoid().unsqueeze(-1)
            delta = a.o_proj(gated.reshape(rows, -1))
            post = x + delta.float()
            output = post + layer._feed_forward(
                layer.post_attention_layernorm(post).to(torch.bfloat16)).float()
            cache = ref.Step5Cache(config, batch=1, max_tokens=rows,
                                  device=torch.device("cuda"), dtype=torch.bfloat16)
            direct = layer(x, positions, cache, seq_lens, past_lens, is_decode=False)
            assert torch.equal(output, direct)
            for name, tensor in {"residual": x, "norm": hidden, "qraw": qraw,
                                 "kraw": kraw, "value": value, "query": query, "key": key,
                                 "logits": logits, "attention": attention, "gated": gated,
                                 "delta": delta, "post": post, "output": output}.items():
                save(f"r{rows}_{name}", tensor)
            mask = positions[:, None] - positions[None, :]
            mask = (mask >= 0) & (mask < 512)
            q64 = query.double().transpose(0, 1)
            k64 = key.double().repeat_interleave(16, dim=1).transpose(0, 1)
            v64 = value.double().repeat_interleave(16, dim=1).transpose(0, 1)
            # Use the exact FP32 launch scale widened to FP64, not a new scale.
            scale = float(torch.tensor(a.scaling, dtype=torch.float32))
            scores = (q64 @ k64.transpose(-1, -2)) * scale
            scores.masked_fill_(~mask, float("-inf"))
            attention64 = (scores.softmax(-1) @ v64).transpose(0, 1)
            save(f"r{rows}_attention_fp64", attention64)
            # Match selected-forward's 32-column online tiles, including leading
            # masked selections. Only P and final output are BF16 materializations.
            maximum = torch.full((64, rows, 1), float("-inf"), device="cuda", dtype=torch.float64)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros((64, rows, 192), device="cuda", dtype=torch.float64)
            selected = positions.long()[:, None] - 511 + torch.arange(512, device="cuda")[None, :]
            for start in range(0, 512, 32):
                ids = selected[:, start:start + 32]
                valid = ids >= 0
                ids = ids.clamp_min(0)
                tile = scores.gather(2, ids.unsqueeze(0).expand(64, -1, -1))
                tile = tile.masked_fill(~valid.unsqueeze(0), float("-inf"))
                next_max = torch.maximum(maximum, tile.amax(-1, keepdim=True))
                safe_max = next_max.masked_fill(torch.isneginf(next_max), 0.)
                rescale = torch.exp(maximum - safe_max)
                probability = torch.exp(tile - safe_max)
                denominator = denominator * rescale + probability.sum(-1, keepdim=True)
                numerator = numerator * rescale
                rounded = probability.to(torch.bfloat16).double()
                for head in range(64):
                    numerator[head] += (rounded[head, :, :, None] * v64[head, ids]).sum(1)
                maximum = next_max
            bf16_p = (numerator / denominator).transpose(0, 1).to(torch.bfloat16)
            save(f"r{rows}_attention_bf16_p", bf16_p)
            for name, operand, weight in [("qraw", hidden, a.q_proj.weight),
                                           ("kraw", hidden, a.k_proj.weight),
                                           ("value", hidden, a.v_proj.weight),
                                           ("logits", hidden, a.g_proj.weight),
                                           ("delta", gated.reshape(rows, -1), a.o_proj.weight)]:
                save(f"r{rows}_{name}_fp64",
                     torch.nn.functional.linear(operand.double(), weight.double()))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
