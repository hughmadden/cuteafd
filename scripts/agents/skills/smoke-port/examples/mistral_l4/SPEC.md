# Mistral Large 4 (1T-A52B) NVFP4: reference semantics

Smoke-port phase 0 output for `mistralai/Mistral-Large-4-1T-A52B-NVFP4` at
revision `bb6d00fe249b0e876f0cffc6b85014be02de3e68`.

**Sources.** Geometry and bytes come from the revision's `params.json` and the
32 shard headers (`census.md`, produced by `scripts/ckpt_census.py`). Semantics
come from the reference implementation in the serving image
`vllm/vllm-openai:mistral-large-4` (linux/amd64 manifest
`sha256:13bc2adaca593fddfa8f88e41f7cea53582d46d169d63bd86e1477157575fa52`,
vLLM `0.1.dev23730+g64053a10b`), extracted with `scripts/image_ref_extract.py`.
vLLM `main` (`7495e9e5`, committed 30 Oct 2026 -0500) does **not** carry this model; the image
does. File references below are relative to `site-packages/vllm/`:

| Topic | File |
| --- | --- |
| params.json → config | `transformers_utils/configs/mistral.py` (`adapt_config_dict`, `_remap_mistral_large_4_args`, `get_mistral_large_4_dspark_config`) |
| Per-layer spec | `models/mistral/common/layer_spec.py` |
| RoPE | `models/mistral/common/rope.py` (+ `model_executor/layers/rotary_embedding/{common,deepseek_scaling_rope}.py`) |
| Decoder, loader, name map | `models/mistral_large4/nvidia/model.py` |
| MLA attention | `models/mistral/nvidia/attention.py` |
| SSMax | `models/mistral/nvidia/ssmax.py`, `ops/ssmax.py` |
| DSA indexer | `models/mistral/nvidia/indexer.py`, `ops/indexer.py` |
| MoE / MLP | `models/mistral/nvidia/{moe,mlp}.py` |
| DSpark drafter | `models/mistral_large4/nvidia/dspark.py`, `v1/worker/gpu/spec_decode/dspark/speculator.py`, `model_executor/models/qwen3_dspark.py` (`DSparkMarkovHead`), `v1/worker/gpu/spec_decode/eagle/eagle3_utils.py` (tap ids) |
| Vision | `model_executor/models/pixtral.py` (hosts the text model) |

## 1. Geometry

| Item | Value |
| --- | --- |
| Layers | 72; layer 0 dense FFN (24,576), layers 1–71 MoE |
| Hidden / vocab | 6,144 / 204,800; untied BF16 `tok_embeddings`, `output` |
| Norm | RMSNorm, eps 1e-5 (`attention_norm`, `ffn_norm`, `kv_a_norm`, `norm`) |
| Attention | MLA, 64 heads, **no q-LoRA**; `kv_lora_rank` 512; RoPE layers: qk = nope 64 + rope 64; NoPE layers: qk = nope 128; v 128 |
| Positions | 1,048,576 (YaRN 2,048 × 512) |
| Routed experts | 128, top-4, intermediate 6,144, SiLU-gated (w1 gate, w3 up, w2 down) |
| Shared expert | 1, intermediate 6,144 |
| Indexer | 32 heads × 128, top-k 2,048, FP8 (UE8M0 scales, 128 blocks) |
| Drafter | DSpark: 3 dense MLA layers, block 3, 5 target taps, RNN Markov head rank 512 |

### Per-layer pattern (exact, from headers)

- **NoPE layers:** 3, 7, 11, …, 71 (`nope = rope,rope,rope,nope` repeated).
  Their `wkv_a_with_mqa` is `[512, 6144]` (no rope key), `wkv_b` `[16384, 512]`
  (k_nope 128 + v 128 per head). RoPE layers: `wkv_a` `[576, 6144]`, `wkv_b`
  `[12288, 512]` (k_nope 64 + v 128).
- **KV-sharing layers:** 39, 43, 47, 51, 55, 59, 63, 67, 71 read **layer 35's**
  latent cache (all NoPE; the source must be NoPE too). They have no
  `wkv_a_with_mqa` / `kv_a_norm` and write no cache; they keep their own `wq`,
  `wkv_b`, `wo` and indexer. 63 layers own a latent cache.
- **Shared-indexer layers:** 17, 18, 21, 22, 25, 26, …, 69, 70 (28 layers)
  reuse the top-k of the **closest preceding indexer layer**; 44 layers run an
  indexer. Layer 0 must have one.
- **Sparse attention is on for every layer** (`use_sparse = index_topk is not
  None`). With ≤ 2,048 visible keys the top-k selects all of them, so **sparse
  attention equals dense attention for contexts ≤ 2,048 tokens** — the smoke
  port can bring up dense MLA first and add the indexer second.
- **Indexer RoPE follows the layer** (`match_nope`): NoPE layers' indexers use no
  RoPE.

## 2. Decoder layer

```text
x0 = tok_embeddings[ids]                          # BF16, no scaling
for L in 0..71:
    h = RMSNorm(x, attention_norm[L])
    x = x + wo[L] · MLA_L(h, pos)                    # §3
    h = RMSNorm(x, ffn_norm[L])
    x = x + (L == 0 ? MLP(h) : MoE_L(h))             # §5
logits = 30 · tanh( output · RMSNorm(x, norm) / 30 )   # softmax_tanh_gating = soft-cap 30
```

vLLM fuses residual-add + norm; the stream entering layer L is the residual
after folding layer L-1's output (this matters for drafter taps, §7).

## 3. MLA attention (per layer L)

```text
q = wq · h                       # FP8 block weight-only, [64, 128]
if owns_kv(L):
    kv = wkv_a · h               # BF16, 576 (RoPE) or 512 (NoPE) rows
    c  = RMSNorm(kv[:512], kv_a_norm)            # latent, cached
    k_pe = kv[512:576] if rope(L) else 0         # cached (rope applied)
else: read layer 35's cache (no write)
if rope(L):
    q[:, 64:128], k_pe = RoPE(q[:, 64:128], k_pe, pos)   # §4; MLA layout [nope | rope]
if ssmax: q *= s[h] · log(pos + 1) + b[h]        # per head, whole q, after RoPE (§6)
# NoPE: q is 128 nope dims, padded with 64 zero rope dims to the shared 576-wide layout
k_nope, v = split(wkv_b · c)     # BF16; 64/128 (RoPE) or 128/128 (NoPE) per head
attn = softmax( (q_nope·k_nope + q_pe·k_pe) · 128^-0.5 ) over top-k keys (§6)
out  = attn · v                  # [64, 128] -> wo (FP8 block weight-only)
```

- Softmax scale is `qk_head_dim^-0.5 = 128^-0.5` on **every** layer (NoPE
  included); YaRN magnitude scaling is off (mscale 1, asserted).
- The latent record is the GLM/DeepSeek MLA shape (512 latent + 64 rope). vLLM
  stores NoPE layers in the same 576-wide layout with a zero rope key; an
  engine may store 512 for them.

## 4. RoPE: interleaved YaRN with LeRoPE

- **Interleaved (GPT-J) pairs** `(2i, 2i+1)`, `is_neox_style=False`;
  `rotary_dim` 64; θ = 10,000.
- `pos_freqs[i] = θ^(2i/64)`, i = 0..31;
  `extrap = exp(alpha) / pos_freqs`, `interp = exp(alpha) / (2048 · pos_freqs)`,
  where `alpha = lerope.alpha` (`[32]`, shared by all layers).
- YaRN ramp on the **unscaled** frequencies:
  `low = floor(64·ln(512/(32·2π)) / (2 ln θ)) = 3`,
  `high = ceil(64·ln(512/(1·2π)) / (2 ln θ)) = 16`,
  `ramp[i] = clamp((i − 3)/(16 − 3), 0, 1)`,
  `inv_freq = interp · ramp + extrap · (1 − ramp)`.
- cos/sin unscaled (attention_factor 1.0). Table length 512 × 2048 = 1,048,576.
- The **drafter** uses the same YaRN **without** LeRoPE (alpha unset).

## 5. MoE and MLPs

```text
router  = gate · h               # BF16 weight, FP32 logits, [128]
p       = softmax(router)        # scoring "softmax", no bias, no groups
top4    = topk(p, 4); w = p[top4] / sum(p[top4])      # norm_topk_prob = True
routed  = 1.0 · Σ w_e · Expert_e(h)                   # routed_scale 1.0
shared  = w2_s · (SiLU(w1_s · h) ⊙ (w3_s · h))       # FP8 block weight-only
MoE(h)  = routed + shared
MLP(h)  = w2 · (SiLU(w1 · h) ⊙ (w3 · h))             # layer 0, FP8 block weight-only
```

Routed experts are ModelOpt NVFP4 g16: `weight` U8 `[6144, 3072]` (two E2M1
per byte), `weight_scale` E4M3 `[6144, 384]` (per 16), `weight_scale_2` FP32
scalar (global), `input_scale` FP32 scalar (activation, W4A4). Stored **per
expert and unfused** (`layers.L.experts.E.w{1,2,3}`). cuteafd's fused gate/up
activation quantization needs each expert's w1/w3 `input_scale` to agree —
check on the local snapshot before packaging.

## 6. SSMax and the DSA indexer

**SSMax** (`ssmax_scaling: true`): `q ← q · (s_h · log(pos + 1) + b_h)`, per
head, computed in BF16 (each op rounded), applied to the full query after
RoPE. `layers.L.attention.ssmax_scaling.{s,b}` `[64]` for MLA and
`layers.L.attention.indexer.ssmax_scaling.{s,b}` `[32]` for the indexer.

**Indexer** (44 layers):

```text
q_in = LayerNorm(h; q_input_norm.{weight,bias}, eps 1e-6)
q    = wq_b · q_in                       # FP8 block weight-only, [32, 128]
kw   = [wk ; weights_proj] · h           # BF16, 128 + 32
k    = LayerNorm(kw[:128]; k_norm.{weight,bias}, eps 1e-6, FP32)
w    = kw[128:]                          # per-head weights [32]
q   *= s_h · log(pos + 1) + b_h          # indexer SSMax, before RoPE
if rope(L): RoPE on dims [0:64] of q and k   # indexer layout [rope | nope] (MLA is [nope | rope])
q8, qs = fp8_ue8m0_quant(q, block 128);  k8 = fp8_ue8m0_quant(k) -> indexer cache (per layer)
score[t, s] = Σ_h (w[t,h] · 128^-0.5 · 32^-0.5 · qs[t,h]) · ReLU(q8[t,h] · k8[s])   # DeepSeek V3.2 indexer
topk[t] = top-2048 causal keys by score
```

The indexer op is vLLM's `SparseAttnIndexer` (the DeepSeek V3.2 lightning
indexer) with Mistral's norms, SSMax and RoPE layout around it; cuteafd's GLM 5
DSA path implements the same op.

## 7. DSpark drafter (`mtp_dspark.*`, BF16)

Config (`get_mistral_large_4_dspark_config`): 3 layers, block (depth) 3,
`head_type` rnn, `draft_input` position_specific, `markov_rank` 512, sliding
window 2,048, `sample_from_anchor` true, mask token id = vocab size (204,800).

- **Taps:** `target_activation_layers 1,18,36,54,71` → aux layers `i+1`
  (`eagle3_utils.py`): the residual stream **after** layers 1, 18, 36, 54 and
  71 (72 = the final pre-norm stream). Concatenate (5 × 6,144),
  `target_projection` (→ 6,144), `target_norm` RMSNorm.
- **Context KV:** for each draft layer, `kv_a` of the projected taps →
  `kv_a_norm` latent + RoPE'd `k_pe`, written for every context token,
  bypassing the layer's input norm.
- **Draft block:** `[anchor, mask_1, mask_2]`; anchor embedding from the
  target's `tok_embeddings`, masks from `mask_embedding[0..1]` (position
  specific). Three dense MLA layers (target geometry, RoPE on all, no SSMax, no
  indexer, dense FFN 24,576), **non-causal within the block**, window = 2,048
  context keys before the anchor + the block. `final_norm`, then the target's
  LM head with the 30 soft-cap → base logits for each position.
- **RNN Markov head**, sequential over the 3 positions, `state₀ = 0`,
  `prev₀ = anchor`:
  ```text
  e        = embedding[prev]                              # [512]
  g, c, o  = split3(joint_proj · [state ; e ; h_i] + bias) # 7168 -> 1536
  state    = σ(g) ⊙ state + (1 − σ(g)) ⊙ tanh(c)
  logits_i = base_logits_i + output · tanh(o)             # bias NOT soft-capped
  prev     = sample(logits_i)
  ```
- vLLM runs `num_speculative_tokens = 3` (= depth; position-specific masks
  require it).

## 8. Vision (Pixtral tower; deferred for a text smoke)

Patch 14, max 1,540 px, `patch_conv` (no bias) → `ln_pre` RMSNorm → 30 blocks
(RMSNorm eps 1e-5, 16-head attention with 2-D RoPE θ 10,000, SiLU-gated FFN
6,144) → `pre_mm_projector_norm` → 2 × 2 `patch_merger` (8,192 → 2,048) →
`vision_language_adapter` `w_in` (2,048 → 6,144), GELU, `w_out` (6,144 → 6,144).
Tokens: `[IMG]` per merged patch, `[IMG_BREAK]` per row, `[IMG_END]`.
Processor: Pixtral, mean/std CLIP, `longest_edge` 1,540.

## 9. Tokenizer, template, API

- `tokenizer.json` (HF tokenizers backend) ships beside `tekken.json`; bos `<s>`
  (1), eos `</s>` (2). Control tokens: `[INST]` 3, `[/INST]` 4,
  `[AVAILABLE_TOOLS]` 5/6, `[TOOL_RESULTS]` 7/8, `[TOOL_CALLS]` 9, `[IMG]` 10,
  `[IMG_BREAK]` 12, `[IMG_END]` 13, `[SYSTEM_PROMPT]` 17/18, `[ARGS]` 32,
  `[CALL_ID]` 33, `[THINK]` 34, `[/THINK]` 35, `[MODEL_SETTINGS]` 36/37.
- `chat_template.jinja` renders `reasoning_effort` (low/medium/high, default
  high) and an optional JSON schema into `[MODEL_SETTINGS]{…}[/MODEL_SETTINGS]`.
- Reasoning `[THINK]…[/THINK]`; tool calls `[TOOL_CALLS]name[ARGS]{json}`.
- Recommended sampling: temperature 1.0, top_p 0.95.

## 10. Checkpoint name map (Mistral → HF-style)

| Mistral | HF-style |
| --- | --- |
| `tok_embeddings`, `norm`, `output` | `model.embed_tokens`, `model.norm`, `lm_head` |
| `layers.L.attention_norm` / `ffn_norm` | `input_layernorm` / `post_attention_layernorm` |
| `attention.wq` / `wkv_a_with_mqa` / `kv_a_norm` / `wkv_b` / `wo` | `self_attn.q_proj` / `kv_a_proj_with_mqa` / `kv_a_layernorm` / `kv_b_proj` / `o_proj` |
| `attention.ssmax_scaling`, `attention.indexer.ssmax_scaling` | `self_attn.ssmax`, `self_attn.indexer.ssmax` |
| `feed_forward.w1/w3/w2` | `mlp.gate_proj/up_proj/down_proj` |
| `shared_experts.w*`, `gate`, `experts.E.w*` | `mlp.shared_experts.*`, `mlp.gate`, `mlp.experts.E.*` |
| `lerope.alpha` | RoPE table input (not a module weight) |
| FP8 `weight_scale` `[R/128, 1, C/128, 1]` | 128 × 128 block scales (squeeze dims 1 and 3) |

Quantization lives in `params.json` `quantization_config` (`quant_method
modelopt`, `MIXED_PRECISION`) with per-module `quantized_layers`
(`FP8_PB_WO`: FP8 128 × 128 block, weight-only; `NVFP4` g16 for `experts`).
