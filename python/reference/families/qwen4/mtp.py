#!/usr/bin/env python3
"""Torch reference of Qwen 3.8 Flash Next's native MTP layer on golden inputs.

The pinned transformers skips the ``mtp.*`` weights, so the MTP contract is
vLLM's ``Qwen4ExpMultiTokenPredictor`` (vllm 039ea826,
``vllm/models/qwen4_exp/nvidia/mtp.py``) with transformers' block math:

* input (``residual_linear_shared``): the target's pre-mixer streams ``h_t``
  (``layer47.bin``: the four hyper-connection streams after the last layer)
  are RMS-normalized over all ``4 * hidden`` values (``pre_fc_norm_hidden``,
  ``1 + w``), projected per stream by the shared ``fc_hidden``; the next
  token's embedding (the shared ``embed_tokens``) is normalized
  (``pre_fc_norm_embedding``) and projected by ``fc_embedding``, then added to
  every stream (vLLM's combine with no injection logits: unit weight);
* one full-attention (QSA) decoder layer ``mtp.layers.0`` without PLE, at the
  target positions (vLLM's proposer keeps them: pair ``(h_t, x_{t+1})`` sits
  at position ``t``); its MoE experts are ``mtp.layers.0.mlp.experts``;
* ``mtp.hyper_connection_mixer`` then the shared ``lm_head`` (FP32): row ``t``
  predicts ``x_{t+2}``.

The pair rows ``t = 0 .. T-2`` run as one causal prefill (teacher forcing: the
MTP's cache holds the canonical pairs, as a serving engine commits them).

Attention keeps FP32 probabilities and an FP32 PV product (as FlashAttention-
style kernels and the engine do): the MTP layer's attention logits are large
(row max |logit| ~300), and transformers' eager path, which rounds the
probabilities to BF16 before PV, lands ~0.998 row cosine away from it
(``--bf16-probabilities`` selects the eager path).

Writes into ``--out``:

  mtp_streams.bin  bf16 [T-1, 4, hidden]  the MTP layer's output streams (the next draft step's input)
  mtp_argmax.bin   i32  [T-1]             greedy draft of each row
  mtp_logits.bin   f32  [K, vocab]        logits of rows meta["logit_rows"]
  meta.json        accuracy / NLL against the true x_{t+2}

  PYTHONPATH=<cuteafd>/third_party/transformers/src USE_HUB_KERNELS=0 \\
    mtp.py --snapshot FP8_SNAP --golden runs/qwen4-golden --out runs/qwen4-golden/mtp
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from golden import PREFIX, Weights, experts_fp32, load_experts, load_module


def gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    v = x.float()
    v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
    return (v * (1.0 + w.float())).to(x.dtype)


def fp32_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """transformers' eager attention with FP32 probabilities and PV (output rounded to BF16 once)."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import repeat_kv

    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query.float(), key.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask.float()
    weights = torch.softmax(weights, dim=-1)
    out = torch.matmul(weights, value.float()).to(query.dtype).transpose(1, 2).contiguous()
    return out, weights


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True, help="dense weights (and MTP experts unless --experts-snapshot)")
    p.add_argument("--experts-snapshot", type=Path)
    p.add_argument("--golden", type=Path, required=True, help="golden.py output (tokens.bin, layer47.bin)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--logit-every", type=int, default=32)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--bf16-probabilities", action="store_true",
                   help="transformers' eager attention (BF16 probabilities before PV)")
    a = p.parse_args()

    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen4_exp import modeling_qwen4_exp as ref

    ref.Qwen4ExpTextExperts.forward = experts_fp32
    if not a.bf16_probabilities:
        ref.eager_attention_forward = fp32_attention
    torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = AutoConfig.from_pretrained(a.snapshot).text_config
    config._attn_implementation = "eager"
    if int(config.mtp_num_hidden_layers) != 1:
        raise ValueError("one MTP layer expected")
    tokens = np.fromfile(a.golden / "tokens.bin", dtype=np.int32)
    t = len(tokens)
    hidden = config.hidden_size
    last = config.num_hidden_layers - 1
    streams = torch.from_numpy(np.fromfile(a.golden / f"layer{last:02d}.bin", dtype=np.int16).copy()) \
        .view(torch.bfloat16).view(t, config.hc_count * hidden).cuda()
    dense = Weights(a.snapshot)
    experts_src = Weights(a.experts_snapshot) if a.experts_snapshot else dense
    eps = config.rms_norm_eps
    a.out.mkdir(parents=True, exist_ok=True)
    n = t - 1
    ids = torch.tensor(tokens, device="cuda", dtype=torch.long)[None]
    with torch.inference_mode():
        embed_w = dense.get(PREFIX + "embed_tokens.weight")
        token = torch.nn.functional.embedding(ids[:, 1:], embed_w)[0]
        token = torch.nn.functional.linear(gemma_rmsnorm(token, dense.get("mtp.pre_fc_norm_embedding.weight"), eps),
                                           dense.get("mtp.fc_embedding.weight"))
        previous = gemma_rmsnorm(streams[:-1], dense.get("mtp.pre_fc_norm_hidden.weight"), eps)
        previous = torch.nn.functional.linear(previous.view(n, config.hc_count, hidden), dense.get("mtp.fc_hidden.weight"))
        h = (previous + token[:, None]).flatten(-2)[None]

        mcfg = copy.deepcopy(config)
        mcfg.num_hidden_layers = 1
        mcfg.layer_types = ["full_attention"]
        mcfg.ple_layer_ids = []
        mcfg._attn_implementation = "eager"
        position_ids = torch.arange(n, device="cuda").view(1, 1, -1).expand(4, 1, -1)
        text_positions, mrope_positions = position_ids[0], position_ids[1:]
        causal = create_causal_mask(config=mcfg, inputs_embeds=h[..., :hidden], attention_mask=None,
                                    past_key_values=None, position_ids=text_positions, allow_is_causal_skip=False)
        rotary = ref.Qwen4ExpTextRotaryEmbedding(config=mcfg).cuda()
        position_embeddings = rotary(h[..., :hidden], mrope_positions)
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            layer = ref.Qwen4ExpTextDecoderLayer(mcfg, 0)
            mixer = ref.Qwen4ExpTextGatedResidual(mcfg, use_combine=False)
        torch.set_default_dtype(torch.float32)
        if layer.ple is not None or layer.layer_type != "full_attention":
            raise ValueError("the MTP layer is one full-attention layer without PLE")
        layer, mixer = layer.to_empty(device="cuda"), mixer.to_empty(device="cuda")
        load_module(layer, dense, "mtp.layers.0.", {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"})
        load_experts(layer.mlp.experts, experts_src, "mtp.layers.0.")
        load_module(mixer, dense, "mtp.hyper_connection_mixer.", set())
        out = layer(h, position_embeddings=position_embeddings, attention_mask=causal, conv_mask=None,
                    past_key_values=None, ple_input_ids=ids[:, :n])
        (a.out / "mtp_streams.bin").write_bytes(out[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
        final = mixer(out)[0]
        head = dense.get("lm_head.weight").float()
        logits = final.float() @ head.T
        del head
        argmax = logits.argmax(-1)
        (a.out / "mtp_argmax.bin").write_bytes(argmax.to(torch.int32).cpu().numpy().tobytes())
        rows = list(range(0, n, a.logit_every)) + ([n - 1] if (n - 1) % a.logit_every else [])
        (a.out / "mtp_logits.bin").write_bytes(logits[rows].contiguous().cpu().numpy().tobytes())
        target = ids[0, 2:]
        scored = n - 1
        accuracy = (argmax[:scored] == target).float().mean().item()
        nll = 0.0
        for first in range(0, scored, 512):
            end = min(first + 512, scored)
            lp = torch.log_softmax(logits[first:end].double(), -1)
            nll -= lp.gather(1, target[first:end, None]).sum().item()
    meta = {"rows": n, "logit_rows": rows, "draft_accuracy": accuracy, "mean_nll": nll / scored,
            "reference": "vLLM 039ea826 Qwen4ExpMultiTokenPredictor contract, transformers 62d7ebd7 block math, "
                         "target positions, FP32 routed sum",
            "attention": "bf16 probabilities (eager)" if a.bf16_probabilities else "fp32 probabilities"}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"MTP greedy accuracy on x_(t+2): {accuracy:.4f}, mean NLL {nll / scored:.4f} over {scored} rows")


if __name__ == "__main__":
    main()
