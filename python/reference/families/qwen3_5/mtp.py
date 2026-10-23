#!/usr/bin/env python3
"""Independent BF16 dense Qwen native-MTP oracle on canonical target goldens.

The pinned Transformers skips mtp.*. Contract: vLLM commit
0510bd66795683d41f0dcdbcb374bfbd6e91595f:
- model_executor/models/qwen3_5.py:213 inherits Qwen3NextModel.forward;
  model_executor/models/qwen3_next.py:739 returns post-final-norm target hidden.
- model_executor/models/qwen3_5_mtp.py:157-164 normalizes the next embedding
  and target hidden separately (zero-centered RMSNorm), concatenates
  [embedding, hidden], then applies the checkpoint's single 2H -> H fc.
- qwen3_5_mtp.py:170-189 runs one full-attention decoder and mtp.norm;
  its output is both the next-step feedback and the shared head input
  (worker/gpu/spec_decode/target_dependent_ar/speculator.py:430-438).
- target_dependent_ar/speculator.py:452-456 keeps pair (h_t, x_{t+1}) at
  position t, predicting x_{t+2}. No position shift or calibration is used.

Block math is the locked Transformers Qwen3_5DecoderLayer. Teacher-forced
pairs run as one causal prefill. Attention uses FP32 probabilities/PV, as in
our Qwen4 MTP oracle; --bf16-probabilities selects Transformers eager math.

Reads tokens.bin and layer63.bin (pre-final-norm) from the BF16 target
reference. Writes the Qwen4 oracle layout: mtp_streams.bin BF16 [T-1,H]
(post-mtp.norm feedback), mtp_argmax.bin i32 [T-1], mtp_logits.bin FP32
[K,V] at meta.json logit_rows. Head accumulation is FP32.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from golden import qualification_reference, verify_reference_source

VLLM_REVISION = "0510bd66795683d41f0dcdbcb374bfbd6e91595f"
VLLM_MTP_BLOB = "2acac6b5b7c5979fbca477dabcd0a0aa5bfb0ff7"


def logit_rows(rows: int, every: int) -> list[int]:
    if rows < 1 or every < 1:
        raise ValueError("positive row count and logit stride required")
    return sorted(set(range(0, rows, every)) | {rows - 1})


class Weights:
    def __init__(self, snapshot: Path, index: dict):
        self.snapshot = snapshot
        self.names = index["weight_map"]

    def get(self, name):
        from safetensors import safe_open
        with safe_open(self.snapshot / self.names[name], framework="pt", device="cpu") as shard:
            value = shard.get_tensor(name)
        return value.cuda()


def fp32_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    scores = query.float() @ key.float().transpose(2, 3) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask.float()
    probs = torch.softmax(scores, dim=-1)
    out = (probs @ value.float()).to(query.dtype).transpose(1, 2).contiguous()
    return out, probs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--logit-every", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bf16-probabilities", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.logit_every < 1:
        parser.error("--logit-every must be positive")

    import numpy as np
    import torch
    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_5 import modeling_qwen3_5 as ref

    qualification = qualification_reference()
    _, text, index, source = qualification.reference_identity(args.snapshot)
    source_pin = verify_reference_source(source)
    if text.get("mtp_num_hidden_layers") != 1 or text.get("mtp_use_dedicated_embeddings", False):
        raise ValueError("one MTP layer sharing target embeddings required")
    golden_meta = json.loads((args.golden / "meta.json").read_text())
    if (golden_meta.get("family") != "qwen3_5" or golden_meta.get("mtp") is not False
            or golden_meta.get("config_sha256") != hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest()
            or golden_meta.get("reference_sha256") != hashlib.sha256(source).hexdigest()):
        raise ValueError("canonical BF16 target golden/source identity mismatch")
    tokens = np.fromfile(args.golden / "tokens.bin", dtype="<i4")
    count, hidden = len(tokens), text["hidden_size"]
    last = text["num_hidden_layers"] - 1
    target_path = args.golden / f"layer{last:02d}.bin"
    if count < 3 or target_path.stat().st_size != count * hidden * 2:
        raise ValueError("at least three canonical BF16 target rows required")
    weights = Weights(args.snapshot, index)
    required = {"mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_hidden.weight",
                "mtp.pre_fc_norm_embedding.weight", "model.language_model.norm.weight",
                "model.language_model.embed_tokens.weight", "lm_head.weight"}
    if not required.issubset(weights.names):
        raise ValueError(f"missing MTP/shared tensors: {sorted(required - weights.names.keys())}")
    if args.check_only:
        print(json.dumps({"preflight": "ok", "rows": count - 1, "source_pin": source_pin,
                          "vllm_revision": VLLM_REVISION}))
        return

    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not args.bf16_probabilities:
        ref.eager_attention_forward = fp32_attention
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config = copy.deepcopy(config)
    config.num_hidden_layers = 1
    config.layer_types = ["full_attention"]
    config._attn_implementation = "eager"
    eps, rows = config.rms_norm_eps, count - 1
    ids = torch.tensor(tokens.astype(np.int64), device="cuda")
    canonical = torch.from_numpy(np.fromfile(target_path, dtype="<i2").copy()).view(torch.bfloat16).view(count, hidden).cuda()

    def norm(x, name):
        v = x.float()
        v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
        return (v * (1.0 + weights.get(name).float())).to(x.dtype)

    with torch.inference_mode():
        target = norm(canonical[:-1], "model.language_model.norm.weight")
        target = norm(target, "mtp.pre_fc_norm_hidden.weight")
        embedding = torch.nn.functional.embedding(ids[1:], weights.get("model.language_model.embed_tokens.weight"))
        embedding = norm(embedding, "mtp.pre_fc_norm_embedding.weight")
        h = torch.nn.functional.linear(torch.cat([embedding, target], dim=-1), weights.get("mtp.fc.weight"))[None]
        del embedding, target, canonical
        with torch.device("meta"):
            layer = ref.Qwen3_5DecoderLayer(config, 0).to(dtype=torch.bfloat16)
        layer = layer.to_empty(device="cuda").eval()
        for name, param in layer.named_parameters():
            value = weights.get("mtp.layers.0." + name)
            if value.shape != param.shape or value.dtype != torch.bfloat16:
                raise ValueError(f"unexpected MTP tensor: {name}, {value.shape}, {value.dtype}")
            param.copy_(value)
        positions = torch.arange(rows, device="cuda")[None]
        rotary = ref.Qwen3_5TextRotaryEmbedding(config=config).cuda()
        position_embeddings = rotary(h, positions[None].expand(3, -1, -1))
        causal = create_causal_mask(config=config, inputs_embeds=h, attention_mask=None,
                                    past_key_values=None, position_ids=positions, allow_is_causal_skip=False)
        output = layer(h, position_embeddings=position_embeddings, attention_mask=causal,
                       position_ids=positions, past_key_values=None)[0]
        feedback = norm(output, "mtp.norm.weight")
        logits = feedback.float() @ weights.get("lm_head.weight").float().T
        if not torch.isfinite(logits).all() or not torch.isfinite(feedback).all():
            raise ValueError("non-finite MTP output")
        argmax = logits.argmax(-1)
        scored = rows - 1
        accuracy = (argmax[:scored] == ids[2:]).float().mean().item()
        nll = torch.nn.functional.cross_entropy(logits[:scored].double(), ids[2:]).item()
        selected = logit_rows(rows, args.logit_every)
        args.out.mkdir(parents=True, exist_ok=True)
        feedback.contiguous().view(torch.int16).cpu().numpy().astype("<i2").tofile(args.out / "mtp_streams.bin")
        argmax.to(torch.int32).cpu().numpy().astype("<i4").tofile(args.out / "mtp_argmax.bin")
        logits[selected].contiguous().cpu().numpy().astype("<f4").tofile(args.out / "mtp_logits.bin")
    meta = {"family": "qwen3_5", "rows": rows, "hidden": hidden, "vocab": config.vocab_size,
            "logit_rows": selected, "draft_accuracy": accuracy, "mean_nll": nll,
            "scored_tokens": scored, "feedback": "post-mtp.norm", "target_hidden": "post-final-norm",
            "concat": ["embedding", "hidden"], "position": "target", "head": "fp32",
            "reference": "vLLM Qwen3_5MultiTokenPredictor contract, locked Transformers block math",
            "vllm_revision": VLLM_REVISION, "vllm_mtp_blob": VLLM_MTP_BLOB,
            "source_pin": source_pin, "torch": torch.__version__,
            "attention": "bf16 probabilities (eager)" if args.bf16_probabilities else "fp32 probabilities",
            "tokens_sha256": hashlib.sha256((args.golden / "tokens.bin").read_bytes()).hexdigest(),
            "target_hidden_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            "oracle_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"MTP greedy accuracy on x_(t+2): {accuracy:.4f}, mean NLL {nll:.4f} over {scored} rows", flush=True)


if __name__ == "__main__":
    main()
