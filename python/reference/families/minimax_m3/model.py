"""Independent MiniMax M3 text math; native-format QDQ, no engine imports.

Served activations model vLLM's B12X/CUTLASS per-expert NVFP4 scale policy.
GEMMs accumulate decoded operands in FP32 and narrow outputs to BF16. This is
an arithmetic oracle, not an emulator of tensor-core reductions or approximate
CUDA reciprocals. BF16 activations are an explicitly diagnostic second arm.
The served-cache arm adds the native E4M3 cache and FP32 residual boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct

import torch
import torch.nn.functional as F

from shape_invariant import ROWS, linear

CHECKPOINT = "MiniMaxAI/MiniMax-M3.1-Flash"
REVISION = "9e7ebacac350497ad4a1272a2e817639bf11561c"
TRANSFORMERS_REVISION = "6b6792ff3b1a434913dc59659d4461acc9c3a51a"
SOURCE_PINS = {
    "src/transformers/models/minimax_m3_vl/modular_minimax_m3_vl.py":
        "1c755515c8aefdbc6f9145ae8f051e43907e21022d9644e81afe47cd2e37b58d",
    "src/transformers/models/minimax_m2/modeling_minimax_m2.py":
        "f4a5f69d3907af2873e38c5ec63e743a34c1900c66cb852981aa3168a9da831a",
}
TEXT = "language_model.model."
HEAD = "language_model.lm_head.weight"


@dataclass(frozen=True)
class Config:
    text: dict
    msa: dict

    @classmethod
    def from_dict(cls, root):
        c = root["text_config"]
        m = c["sparse_attention_config"]
        if (root.get("model_type") != "minimax_m3_vl" or not c["use_gemma_norm"]
                or not c["use_qk_norm"] or c["qk_norm_type"] != "per_head"
                or c["hidden_act"] != "swigluoai" or c["scoring_func"] != "sigmoid"
                or not c["use_routing_bias"] or c["n_shared_experts"] != 1
                or c.get("attention_output_gate", False) or c.get("tie_word_embeddings", False)
                or m["sparse_score_type"] != "max" or m.get("sparse_init_block", 0) != 0
                or not m["use_sparse_attention"]):
            raise ValueError("unsupported MiniMax text semantics")
        for key in ("hidden_size", "vocab_size", "head_dim", "rotary_dim", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "num_local_experts",
                    "num_experts_per_tok", "dense_intermediate_size", "intermediate_size",
                    "shared_intermediate_size"):
            if type(c[key]) is not int or c[key] <= 0:
                raise ValueError("invalid MiniMax dimension: " + key)
        layers = c["num_hidden_layers"]
        flags = [c["moe_layer_freq"], m["sparse_attention_freq"], m["sparse_disable_index_value"]]
        if any(len(v) != layers or any(x not in (0, 1) for x in v) for v in flags):
            raise ValueError("invalid MiniMax layer flags")
        if (not all(m["sparse_disable_index_value"])
                or c["num_attention_heads"] % c["num_key_value_heads"]
                or c["num_experts_per_tok"] > c["num_local_experts"]
                or not 0 < c["rotary_dim"] <= min(c["head_dim"], m["sparse_index_dim"])
                or c["rotary_dim"] % 2
                or c.get("partial_rotary_factor", c["rotary_dim"] / c["head_dim"]) * c["head_dim"] != c["rotary_dim"]
                or m["sparse_num_index_heads"] != c["num_key_value_heads"]
                or not 0 < m["sparse_local_block"] <= m["sparse_topk_blocks"]
                or m["sparse_block_size"] <= 0):
            raise ValueError("unsupported MiniMax attention or routing geometry")
        q = root["quantization_config"]
        if (q["quant_method"] != "mxfp8" or q["activation_scheme"] != "dynamic"
                or q["weight_block_size"] != [1, 32] or q["group_size"] != 16
                or q["moe_quant_algo"] != "NVFP4" or q["moe_quant_format"] != "nvfp4-pack-quantized"):
            raise ValueError("unsupported MiniMax quantization")
        return cls(c, m)

    def __getattr__(self, name):
        return self.text[name]


def gemma_norm(x, weight, eps):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
            * (1.0 + weight.float())).to(x.dtype)


def rope_partial(x, positions, rotary, theta):
    """NeoX pairing on leading channels, FP32 trigonometry and rotation."""
    frequency = 1.0 / (theta ** (torch.arange(0, rotary, 2, device=x.device).float() / rotary))
    phase = positions.float()[:, None] * frequency
    cos, sin = phase.cos()[:, None], phase.sin()[:, None]
    a, b = x[..., :rotary // 2].float(), x[..., rotary // 2:rotary].float()
    return torch.cat(((a * cos - b * sin).to(x.dtype),
                      (b * cos + a * sin).to(x.dtype), x[..., rotary:]), -1)


def swiglu_oai(gate, up, alpha=1.702, limit=7.0):
    g, u = gate.float().clamp(max=limit), up.float().clamp(-limit, limit)
    return (g * torch.sigmoid(alpha * g) * (u + 1.0)).to(gate.dtype)


def route(logits, bias, topk, scale):
    scores = logits.float().sigmoid()
    ids = (scores + bias.float()).argsort(dim=-1, descending=True, stable=True)[..., :topk]
    weights = scores.gather(-1, ids)
    return ids, weights / weights.sum(-1, keepdim=True) * scale


def dequant_mxfp8(weight, scale):
    if (weight.dtype != torch.float8_e4m3fn or weight.ndim != 2 or weight.shape[1] % 32
            or scale.dtype != torch.uint8 or scale.shape != (weight.shape[0], weight.shape[1] // 32)
            or torch.any(scale == 255)):
        raise ValueError("invalid MXFP8 E4M3/UE8M0 layout")
    return torch.ldexp(weight.float().reshape(*scale.shape, 32),
                       scale.int()[..., None] - 127).reshape(weight.shape)


def positive_scale(scale):
    if scale.numel() != 1 or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("global scale must be finite and positive")
    return scale.float()


def dequant_nvfp4(packed, scale, global_scale):
    if (packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] % 8
            or scale.dtype != torch.float8_e4m3fn or scale.shape != (packed.shape[0], packed.shape[1] // 8)):
        raise ValueError("invalid NVFP4 E2M1/E4M3 layout")
    global_scale = positive_scale(global_scale)
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         device=packed.device, dtype=torch.float32)
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1)
    return (table[codes.long()].reshape(*scale.shape, 16) * scale.float()[..., None]
            / global_scale).reshape(packed.shape[0], -1)


def quantize_mxfp8(x):
    """vLLM mxfp8_utils.py:38-69: ceil-log2 UE8M0 group-32 scale."""
    if x.shape[-1] % 32:
        raise ValueError("MXFP8 activation width must divide 32")
    blocks = x.float().reshape(*x.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = (torch.ceil(torch.log2(maximum / 448.)) + 127).clamp(0, 254).int() - 127
    scale = torch.ldexp(torch.ones_like(maximum), exponent)
    return ((blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)


def quantize_nvfp4(x, global_scale):
    """vLLM nvfp4_emulation_utils.py:695-745: E4M3 scales and E2M1 RN-even."""
    if x.shape[-1] % 16:
        raise ValueError("NVFP4 activation width must divide 16")
    global_scale = positive_scale(global_scale)
    blocks = x.float().reshape(*x.shape[:-1], -1, 16)
    scale = (blocks.abs().amax(-1, keepdim=True) * global_scale / 6.).clamp(max=448)
    scale = scale.to(torch.float8_e4m3fn).float()
    value = torch.where(scale > 0, blocks * global_scale / scale, 0.).clamp(-6, 6)
    boundaries = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=x.device, dtype=torch.float32)
    absolute = value.abs()
    codes = torch.bucketize(absolute.contiguous(), boundaries)
    # Even E2M1 encodings win exactly halfway between adjacent values.
    codes += ((codes % 2 == 1) & (absolute == boundaries[codes.clamp(max=6)])).long()
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=x.device, dtype=torch.float32)
    rounded = torch.copysign(table[codes], value)
    return (rounded * scale / global_scale).reshape(x.shape)


def cache_e4m3(x):
    """kv_pack: FP32 multiply by reciprocal descale=1, RN-even/satfinite E4M3."""
    return x.float().clamp(-448., 448.).to(torch.float8_e4m3fn)


def index_e4m3(x):
    """index_{k_pack,q_quant}: row amax/448, RN reciprocal, multiply, E4M3.

    Both divisions and the product materialize FP32. Dividing x by scale
    directly changes rounding. An all-zero row stores scale=1, not zero.
    """
    value = x.float()
    maximum = value.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum > 0., maximum / 448., torch.ones_like(maximum))
    inverse = torch.ones_like(scale) / scale
    return cache_e4m3(value * inverse), scale.squeeze(-1)


def index_scores(query, query_scale, key, key_scale):
    """Native indexer scales the FP32 dot product, not its E4M3 operands."""
    dot = torch.matmul(query.float(), key.float().transpose(-1, -2))
    weight = query_scale.float() * query.shape[-1] ** -.5
    return (dot * weight[..., None]) * key_scale.float()[..., None, :]


def record(trace, name, value):
    if trace is not None:
        trace[name] = value.detach().clone()


def msa_select(scores, positions, block=128, topk=16, local=1):
    """Rust reference.rs contract: causal maxima, forced locals, stable low-id ties."""
    keys = torch.arange(scores.shape[-1], device=scores.device)
    scores = scores.masked_fill(keys[None, None] > positions[None, :, None], -float("inf"))
    scores = F.pad(scores, (0, -scores.shape[-1] % block), value=-float("inf"))
    pooled = scores.reshape(*scores.shape[:-1], -1, block).amax(-1)
    for offset in range(local):
        ids = (positions // block - offset).clamp_min(0)
        pooled.scatter_(-1, ids[None, :, None].expand(pooled.shape[0], -1, 1), float("inf"))
    selected = pooled.argsort(dim=-1, descending=True, stable=True)[..., :topk]
    valid = pooled.gather(-1, selected) != -float("inf")
    selected = selected.masked_fill(~valid, pooled.shape[-1]).sort(-1).values
    selected = selected.masked_fill(selected == pooled.shape[-1], -1)
    return F.pad(selected, (0, topk - selected.shape[-1]), value=-1).to(torch.int32)


def tensor_contract(cfg, index=None):
    """Checkpoint spelling/layout matches the loader's MiniMax tensor contract."""
    result = {}
    def plain(name, shape, dtype="BF16"):
        result[name] = (shape, dtype)
    def matrix(name, n, k, nvfp4=False):
        if nvfp4:
            plain(name + ".weight_packed", [n, k // 2], "U8")
            plain(name + ".weight_scale", [n, k // 16], "F8_E4M3")
            for suffix in ("weight_global_scale", "input_global_scale"):
                plain(name + "." + suffix, [], "F32")
        else:
            plain(name + ".weight", [n, k], "F8_E4M3")
            plain(name + ".weight_scale_inv", [n, k // 32], "U8")
    h = cfg.hidden_size
    if index is None:
        plain(TEXT + "embed_tokens.weight", [cfg.vocab_size, h])
        plain(TEXT + "norm.weight", [h])
        plain(HEAD, [cfg.vocab_size, h])
        return result
    p = f"{TEXT}layers.{index}."
    for norm in ("input_layernorm", "post_attention_layernorm"):
        plain(p + norm + ".weight", [h])
    for proj, n, k in (("q", cfg.num_attention_heads * cfg.head_dim, h),
                       ("k", cfg.num_key_value_heads * cfg.head_dim, h),
                       ("v", cfg.num_key_value_heads * cfg.head_dim, h),
                       ("o", h, cfg.num_attention_heads * cfg.head_dim)):
        matrix(p + "self_attn." + proj + "_proj", n, k)
    for norm in ("q_norm", "k_norm"):
        plain(p + "self_attn." + norm + ".weight", [cfg.head_dim])
    if cfg.msa["sparse_attention_freq"][index]:
        dim = cfg.msa["sparse_index_dim"]
        matrix(p + "self_attn.index_q_proj", cfg.num_key_value_heads * dim, h)
        matrix(p + "self_attn.index_k_proj", dim, h)
        for norm in ("index_q_norm", "index_k_norm"):
            plain(p + "self_attn." + norm + ".weight", [dim])
    def mlp(stem, inter, expert=False):
        for proj, n, k in (("w1" if expert else "gate_proj", inter, h),
                           ("w3" if expert else "up_proj", inter, h),
                           ("w2" if expert else "down_proj", h, inter)):
            matrix(p + stem + "." + proj, n, k, expert)
    if cfg.moe_layer_freq[index]:
        stem = "block_sparse_moe"
        plain(p + stem + ".gate.weight", [cfg.num_local_experts, h], "F32")
        plain(p + stem + ".e_score_correction_bias", [cfg.num_local_experts], "F32")
        mlp(stem + ".shared_experts", cfg.shared_intermediate_size)
        for expert in range(cfg.num_local_experts):
            mlp(f"{stem}.experts.{expert}", cfg.intermediate_size, True)
    else:
        mlp("mlp", cfg.dense_intermediate_size)
    return result


class Weights:
    """Clone only the active layer, closing all shard mappings before execution."""
    def __init__(self, snapshot, device="cpu"):
        self.snapshot, self.device = Path(snapshot), torch.device(device)
        self.index = json.loads((self.snapshot / "model.safetensors.index.json").read_bytes())["weight_map"]
        self.headers, self.files = {}, {}
        self.read_bytes = 0

    def descriptor(self, name):
        shard = self.index[name]
        path = self.snapshot / shard
        if not path.resolve().is_relative_to(self.snapshot.resolve()) and not path.is_symlink():
            raise ValueError("unsafe checkpoint shard path")
        if shard not in self.headers:
            with path.open("rb") as stream:
                self.headers[shard] = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
        return self.headers[shard][name]

    def validate(self, cfg):
        expected = tensor_contract(cfg)
        for index in range(cfg.num_hidden_layers):
            expected.update(tensor_contract(cfg, index))
        actual = {n for n in self.index if n.startswith(TEXT) or n == HEAD}
        if actual != set(expected):
            raise ValueError("MiniMax text tensor inventory differs")
        for name, (shape, dtype) in expected.items():
            d = self.descriptor(name)
            if d["shape"] != shape or d["dtype"] != dtype:
                raise ValueError("MiniMax tensor shape/dtype differs: " + name)

    def tensor(self, name):
        from safetensors import safe_open
        shard = self.index[name]
        d = self.descriptor(name)
        if shard not in self.files:
            self.files[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        value = self.files[shard].get_tensor(name).to(device=self.device, copy=True)
        self.read_bytes += d["data_offsets"][1] - d["data_offsets"][0]
        return value

    def load_layer(self, index, cfg):
        prefix = f"{TEXT}layers.{index}."
        names = tensor_contract(cfg, index)
        ordered = sorted(names, key=lambda n: (self.index[n], self.descriptor(n)["data_offsets"][0]))
        result = {name.removeprefix(prefix): self.tensor(name) for name in ordered}
        self.close()
        return result

    def close(self):
        from fidelity_windows import release_checkpoint
        release_checkpoint(torch.cuda if self.device.type == "cuda" else None, self)

    def layer_bytes(self, cfg, index):
        return sum(self.descriptor(n)["data_offsets"][1] - self.descriptor(n)["data_offsets"][0]
                   for n in tensor_contract(cfg, index))


def project(x, weights, stem, activation, input_scale=None):
    if activation not in ("served", "served-cache", "bf16"):
        raise ValueError("unknown activation arm")
    if stem + ".weight_packed" in weights:
        weight = dequant_nvfp4(weights[stem + ".weight_packed"], weights[stem + ".weight_scale"],
                               weights[stem + ".weight_global_scale"])
        scale = weights[stem + ".input_global_scale"] if input_scale is None else input_scale
        operand = quantize_nvfp4(x, scale) if activation != "bf16" else x.float()
    else:
        weight = dequant_mxfp8(weights[stem + ".weight"], weights[stem + ".weight_scale_inv"])
        operand = quantize_mxfp8(x) if activation != "bf16" else x.float()
    return linear(operand, weight).to(torch.bfloat16)


def attention(hidden, weights, cfg, index, activation, trace=None):
    p = "self_attn."
    length, dim = len(hidden), cfg.head_dim
    positions = torch.arange(length, device=hidden.device)
    def normalized(proj, norm, heads, width):
        x = project(hidden, weights, p + proj, activation).reshape(length, heads, width)
        return rope_partial(gemma_norm(x, weights[p + norm + ".weight"], cfg.rms_norm_eps),
                            positions, cfg.rotary_dim, cfg.rope_theta)
    q = normalized("q_proj", "q_norm", cfg.num_attention_heads, dim)
    k = normalized("k_proj", "k_norm", cfg.num_key_value_heads, dim)
    v = project(hidden, weights, p + "v_proj", activation).reshape(length, cfg.num_key_value_heads, dim)
    for name, value in (("q", q), ("k", k), ("v", v)):
        record(trace, name, value)
    if activation == "served-cache":
        k, v = cache_e4m3(k), cache_e4m3(v)
        record(trace, "k_fp8", k.view(torch.uint8))
        record(trace, "v_fp8", v.view(torch.uint8))
        k, v = k.float(), v.float()
    sparse = cfg.msa["sparse_attention_freq"][index]
    if sparse:
        iq = normalized("index_q_proj", "index_q_norm", cfg.num_key_value_heads, cfg.msa["sparse_index_dim"])
        ik = normalized("index_k_proj", "index_k_norm", 1, cfg.msa["sparse_index_dim"])
        record(trace, "index_q", iq)
        record(trace, "index_k", ik)
        if activation == "served-cache":
            iq, iqs = index_e4m3(iq)
            ik, iks = index_e4m3(ik)
            record(trace, "index_q_fp8", iq.view(torch.uint8))
            record(trace, "index_k_fp8", ik.view(torch.uint8))
            record(trace, "index_q_scales", iqs)
            record(trace, "index_k_scales", iks)
            iq, ik = iq.float(), ik.float()
    outputs = []
    selections = []
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    for start in range(0, length, ROWS):
        count, extent = min(ROWS, length - start), start + ROWS
        def query(x):
            return F.pad(x[start:start + ROWS], (0, 0, 0, 0, 0, ROWS - count)).transpose(0, 1).float()
        def key(x):
            return F.pad(x[:extent], (0, 0, 0, 0, 0, max(0, extent - length))).transpose(0, 1).float()
        pos = torch.arange(start, extent, device=hidden.device)
        keys = torch.arange(extent, device=hidden.device)
        causal = (keys[None] <= pos[:, None]) & (keys[None] < length)
        if sparse:
            if activation == "served-cache":
                qs = F.pad(iqs[start:start + ROWS], (0, 0, 0, ROWS - count), value=1.).T
                ks = F.pad(iks[:extent], (0, 0, 0, max(0, extent - length)), value=1.).T
                scores = index_scores(query(iq), qs, key(ik), ks)
            else:
                scores = torch.matmul(query(iq), key(ik).transpose(-1, -2))
            selected = msa_select(scores, pos, cfg.msa["sparse_block_size"],
                                  cfg.msa["sparse_topk_blocks"], cfg.msa["sparse_local_block"])
            if trace is not None:
                selections.append(selected[:, :count].transpose(0, 1))
            keep = (keys[None, None, :, None] // cfg.msa["sparse_block_size"]
                    == selected[:, :, None, :]).any(-1) & causal[None]
        groups = []
        for head in range(cfg.num_key_value_heads):
            qs = query(q[:, head * group:(head + 1) * group])
            ks, vs = key(k[:, head:head + 1]), key(v[:, head:head + 1])
            mask = keep[head] if sparse else causal
            score = torch.matmul(qs, ks.transpose(-1, -2)) * dim ** -.5
            score.masked_fill_(~mask[None], -float("inf"))
            groups.append(torch.matmul(score.softmax(-1), vs).transpose(0, 1)[:count])
        outputs.append(torch.cat(groups, 1).reshape(count, -1).to(torch.bfloat16))
    joined = torch.cat(outputs)
    record(trace, "attention", joined.reshape(length, cfg.num_attention_heads, dim))
    if selections:
        record(trace, "q2k", torch.cat(selections))
    projected = project(joined, weights, p + "o_proj", activation)
    record(trace, "attention_projected", projected)
    return projected


def feed_forward(hidden, weights, cfg, index, activation, trace=None):
    def mlp(x, stem, expert=False):
        gate, up, down = ("w1", "w3", "w2") if expert else ("gate_proj", "up_proj", "down_proj")
        scale = None
        if expert:
            if not torch.equal(weights[stem + ".w1.weight_global_scale"], weights[stem + ".w3.weight_global_scale"]):
                raise ValueError("fused NVFP4 gate/up requires equal weight global scales")
            scale = torch.minimum(weights[stem + ".w1.input_global_scale"], weights[stem + ".w3.input_global_scale"])
        g = project(x, weights, stem + "." + gate, activation, scale)
        u = project(x, weights, stem + "." + up, activation, scale)
        return project(swiglu_oai(g, u, cfg.swiglu_alpha, cfg.swiglu_limit),
                       weights, stem + "." + down, activation)
    if not cfg.moe_layer_freq[index]:
        return mlp(hidden, "mlp")
    stem = "block_sparse_moe"
    logits = linear(hidden.float(), weights[stem + ".gate.weight"].float())
    ids, scales = route(logits, weights[stem + ".e_score_correction_bias"],
                        cfg.num_experts_per_tok, cfg.routed_scaling_factor)
    record(trace, "router_ids", ids.to(torch.int32))
    record(trace, "router_weights", scales)
    # Ascending expert order fixes accumulation independently of suffix routing.
    output = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(cfg.num_local_experts):
        rows, slots = torch.where(ids == expert)
        if len(rows):
            value = mlp(hidden[rows], f"{stem}.experts.{expert}", True)
            output[rows] += value.float() * scales[rows, slots, None]
    shared = mlp(hidden, stem + ".shared_experts")
    record(trace, "shared", shared)
    if activation == "served-cache":
        output = output.to(torch.bfloat16).float()
    return (output + shared.float()).to(hidden.dtype)


def layer_forward(hidden, weights, cfg, index, activation="served", trace=None):
    if activation == "served-cache":
        hidden = hidden.float()
    record(trace, "input", hidden)
    normalized = gemma_norm(hidden, weights["input_layernorm.weight"], cfg.rms_norm_eps)
    if activation == "served-cache":
        normalized = normalized.bfloat16()
    record(trace, "norm", normalized)
    hidden = hidden + attention(normalized, weights, cfg, index, activation, trace)
    record(trace, "post_attention", hidden)
    normalized = gemma_norm(hidden, weights["post_attention_layernorm.weight"], cfg.rms_norm_eps)
    if activation == "served-cache":
        normalized = normalized.bfloat16()
    record(trace, "ffn_norm", normalized)
    hidden = hidden + feed_forward(normalized, weights, cfg, index, activation, trace)
    record(trace, "residual", hidden)
    return hidden
