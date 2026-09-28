# DeepSeek V4 port map (delete when deepseek_v4 serves)

I compared the two reference implementations line by line and checked the actual checkpoint tensor names, dtypes and shapes in all three snapshots. The short answer: **V4 shares nothing across layers.** Every V4 layer owns its window cache, compressor and indexer. Nothing in the V4.1 KV-source, index-source or candidate-block machinery carries over.

Refs: `v4:` = V4-Flash-0731 `inference/model.py`, `v41:` = V4.1-Flash `inference/model.py`, `k4:`/`k41:` = the matching `kernel.py`.

The four biggest problems:
1. **mHC pre-mix timing differs.** V4 uses the pre-weights computed in the same sublayer; V4.1 uses the previous sublayer's. V4 also has a separate `hc_head`.
2. **V4 has per-head Q RMS normalisation.** V4.1 does not.
3. **V4's compressor is different.** Ratio 4 uses overlapping windows plus a learned `ape` score bias, and ratio 128 has no indexer and attends to every compressed entry.
4. **The KV and quantisation formats are all different.** V4 keeps FP8 in 64-blocks on 448 dims with the RoPE tail in BF16; V4.1 uses FP8/32 over all 512 dims and FP4/16 (E4M3 scales) for compressed KV. Weight and activation blocks are 128 in V4, 32 in V4.1.

---

## 1. Config / ModelArgs

| field | V4 Flash | V4 Pro | V4.1 Flash | engine consequence |
|---|---|---|---|---|
| dim | 4096 | 7168 | 5120 | parameterize; `v41_hc.cu`, `v41_router.cu`, `v41_attention_ops.cu` (tap), `v41_compressor.cu:150` and others hardcode 5120/20480/15360 |
| n_layers (+MTP) | 43 (+3) | 61 (+3) | 40 (+3) | param |
| n_heads / o_groups | 64 / 8 | 128 / 16 | 64 / 8 | Flash same; Pro needs param |
| q_lora_rank | 1024 | 1536 | 1280 | param |
| o_lora, head_dim, rope_dim | 1024, 512, 64 | same | same | reuse |
| experts / top-k / inter | 256/6/2048 | 384/6/3072 | 384/6/2304 | param |
| MTP experts | 256/6 (same size as backbone) | 384/6 | 128/3 | param (`v41:142-149` get_moe_config) |
| index heads × dim / topk | 64×128 / 512 | 64×128 / 1024 | 32×128 / 512 | param (`v41_index_scores.cu` hardcodes 32 heads) |
| n_hash_layers | 3 | 3 | none | new logic (§3) |
| route_scale | 1.5 | **2.5** | 1.5 | param |
| norm_eps | **1e-6** (default `v4:62`; HF `rms_norm_eps`) | 1e-6 | **1e-20** (`v41:75`, inference cfg:23) | param everywhere, including the HC rsqrt (`v41_hc.cu:84,139,160` hardcode 1e-20) |
| hc_mult / sinkhorn / hc_eps | 4 / 20 / 1e-6 | same | same | reuse |
| rope | θ 1e4 (window-only layers, no YaRN); compressed layers θ 1.6e5 + YaRN f=16, orig 65536, β 32/1 | same | same | reuse; `v4:481-488` ≡ `v41:680-687` |
| swiglu_limit, score_func | 10, sqrtsoftplus | same | same | reuse |
| dspark block / noise / rank | 5 / 128799 / 256 | 5 / 128799 / **512** | 5 / 128799 / 256 | param |
| dspark targets | [40,41,42] = layer **outputs** | [58,59,60] | [37,38,39] = layer **inputs** | new tap point (§7) |
| weight block | 128×128 FP8, e8m0 | 128 (experts are EXL3) | 32×32 | §9 |
| V4.1-only fields | — | — | kv/index_source, candidate_*, engram_*, vision_*, gate_temp, bias_vl | disable |

**What compress_ratios values mean:**
- **V4** (`v4:66`, `v4:296`, `v4:472-477`):
  - `0`: window only.
  - `4`: overlapping gated pooling plus a per-layer indexer (top-512).
  - `128`: non-overlapping gated pooling, no indexer, attends to *all* compressed entries.
  - Flash layout: layers 0–1 → 0; 2..42 alternate 4 (even) / 128 (odd); MTP → 0.
  - Pro layout: layers 0–1 → **128**; 2..60 alternate 4/128.
- **V4.1** (`v41:80-84`, `v41:446-462`):
  - `0`: window only.
  - `2`: gated pooling of 2, no overlap, no ape.
  - `1`: a plain `norm(wkv(x))` projection with no gate (one entry per token).
  - A non-zero ratio does *not* mean the layer compresses; only `kv_source_layers` do.

**Config traps:**
- HF `num_nextn_predict_layers=1` in the V4 Flash and Pro configs is wrong. The checkpoints contain `mtp.0..2`, and `compress_ratios` has 3 trailing zeros. Use n_mtp_layers=3.
- The Pro snapshot has only an HF-style config, so key names need mapping (hidden_size→dim, num_hash_layers→n_hash_layers, etc.).

## 2. Attention

**Q path.**
- V4 applies an unweighted per-head RMS norm after `wq_b`: `q *= rsqrt(mean(q²)+eps)` over 512 dims (`v4:502-505`; DSpark `v4:776`).
- V4.1 goes straight from `wq_b` to RoPE (`v41:770-772`).
- Engine: new logic (a small kernel, or fused into `rope_kernel` in `v41_attention_ops.cu`), used in `v41_attention_query.rs`.

**Window KV.**
- V4: `kv_norm(wkv(x))`, RoPE on the last 64 dims, then FP8 quantise-dequantise of **only the 448 non-RoPE dims, block 64**. The RoPE tail stays BF16 (`v4:508-512`).
- V4.1: FP8 over **all 512 dims including RoPE, block 32** (`v41:705-707`).
- Engine: the FP8 values can be stored in the engine's 32-block format *losslessly*. V4 64-block e8m0 values rescale to 32-block pow2 scales exactly, because it is only an exponent shift. The **RoPE tail cannot**: FP8-quantising it deviates from the reference.
- So `v41_kv.cu` pack/store and `v41_sparse_attention.cu` need a variant that keeps 64 BF16 RoPE dims (or stores all 512 as BF16). This is new.

**Compressor** (`v4:285-383` vs `v41:429-485`):
- **ape.** V4 has `ape [ratio, coff·d]`, a learned F32 bias added to the gate score by position-in-group (`v4:300,338,341,344,351`). V4.1 has none.
- **Overlap (ratio 4).**
  - `wkv`/`wgate` output 2×d (1024; checkpoint `[1024,4096]`).
  - Entry *j* is a per-channel softmax over 8 slots: slots 0–3 are tokens of group *j−1* using channel half A (`[:d]`); slots 4–7 are tokens of group *j* using half B (`[d:]`).
  - Group 0's slots 0–3 are kv=0, score=−inf (`overlap_transform` `v4:313-320`).
  - Decode state is `[B, 2r, 2d]`. Slots r..2r hold the current group; after a group completes they shift down to 0..r (`v4:352-360`).
  - Prefill seeds slots 0..r from the last complete group (`v4:336-338`).
- **Ratio 128.** Plain 128-slot per-channel softmax pooling, `wkv [512,4096]`, `ape [128,512]`, state `[B,128,512]`.
- **Precision.** fp32 projections as in V4.1 ratio>1 (`v4:328`).
- **Order.** norm → RoPE at position `j·ratio` (`v4:369-373`, same rule as `v41:753-758`) → quantise.
  - Main compressor: FP8 block 64 on the non-RoPE dims, BF16 RoPE (`v4:378`), i.e. the same format as window KV.
  - V4.1: FP4 e2m1 with **E4M3 scale per 16** (`v41:760`).
- Engine: `v41_compressor.cu` `pool_kernel`/`project` accept only ratio∈{1,2} and a hardcoded 512×5120 (`:130,150`). This needs **new pooling logic**: overlap + ape + ratio 128, plus larger decode state in `v41_compressor/{commit,prefix,source_cache}.rs`.
- Chunked prefill in the engine must carry the previous full group for overlap. The reference only does prefill at start_pos 0.

**Indexer** (`v4:386-439` vs `v41:488-580`):
- **Which layers.** V4: every ratio-4 layer (21 in Flash, 30 in Pro). V4.1: index_source layers, with shared `topk_idxs` for layers in between.
- **Index keys.** V4 uses the indexer's **own Compressor** (head_dim 128, overlap, own ape `[4,256]`/wkv/wgate/norm, `rotate=True`) run on x (`v4:404,423`). V4.1 derives K from the main latent via `wk [128,512]` + `k_norm` (`v41:517-548`).
- **Hadamard.** V4 applies a Walsh-Hadamard transform (scale 128^-½, no random signs) to both q (`v4:420`) and index-K (`v4:374-375`) before FP4 quantisation. V4.1 has none. The engine needs a new 128-point FWHT in `index_query_prepare_kernel`/`index_store_kernel` (`v41_compressor.cu`).
- **Same in both:** FP4 block 32 e8m0 for q and K (`v4:422,376` ≡ `v41:546,552`); `weights_proj(x)·(128^-½·H^-½)`; relu-weighted head sum; causal mask; `topk=min(k, end//ratio)`; `+offset`.
- **Ordering.** V4.1 sorts the indices (`v41:579`); V4 does not. It doesn't matter to sparse_attn.
- **Candidate blocks.** V4 has none. Disable `v41_candidate_blocks.cu` and the candidate path in `v41_index_selection.rs`.
- Engine: reuse `v41_index_scores.cu`/`v41_index_topk.cu` with heads parameterised (32→64). Pro topk is 1024.

**Compressed topk for ratio-128 layers** (`get_compress_topk_idxs`, `v4:274-282`, `v4:518-519`): a dense, causally masked list of all completed groups. Width is seq/128 (512 at 64K, 8192 at 1M). The engine's selection/attention must accept a variable compressed width with no indexer. This is new.

**Concatenation and offset.** Identical: `[window | compressed]`, offset = prefill seqlen or `win` (`v4:515` ≡ `v41:776`).

**KV cache layout.**
- V4 per layer: `kv_cache[B, win + max_seq/ratio, 512]` (`v4:479-480`), plus `indexer.kv_cache[B, max_seq/4, 128]` (`v4:405`).
- V4.1: a window cache per layer; compressed and index-K caches only on kv_sources (`v41:663-679`), handed down via `SharedAttentionRuntime` (`v41:1166-1180`).
- Rough per-token compressed footprint, assuming 583 B per entry (448 FP8 + 7 scales + 128 B RoPE):

  | layer group (Flash) | count | per token |
  |---|---|---|
  | ratio-4 main compressed | 21 | ≈3.06 KB |
  | ratio-128 main compressed | 20 | ≈0.09 KB |
  | index K | 21 | ≈0.36 KB |
  | **V4 Flash total** | | **≈3.5 KB** |
  | V4.1 total | | ≈0.9 KB |

- Engine: `v41_backbone_cache.rs`/`v41_memory.rs` need a per-layer (not per-source) compressed plus index cache: about 4× the V4.1 footprint, so re-budget capacity.

**Attention sink, sparse_attn, output projection.**
- These are identical: `attn_sink[H]` F32; the inverse RoPE on o; the grouped `wo_a` (convert dequantises it to BF16) einsum then `wo_b` (`v4:539-547` ≡ `v41:781-788`).
- Engine: `v41_attention_output.rs` and `v41_projection_tp2.rs` can be reused. Pro needs o_groups=16 and n_heads=128 as parameters.
- `sparse_attn` initialises the running max at −inf (`k4:318`) vs −1e30 (`k41:355`). V4 index rows always start with a valid slot, so the engine's −1e30 is a safe superset.

## 3. Gate / router

- **Hash layers** (`v4:561-566,581-588`), for layer < 3:
  - The expert *indices* come from `tid2eid[token_id]` (checkpoint I64 `[129280,6]`; store as i32).
  - There is **no bias**, but the gate GEMM still runs: `weights = sqrtsoftplus(x·Wᵀ)` gathered at those 6 experts, sum-normalised, × route_scale.
  - Pro layers 0–2 are also hash layers.
- **Normal layers:** same as V4.1 (bias used for selection only).
- **Small differences:**
  - V4 normalises with no epsilon and no `topk>1` guard (`v4:586-587`); V4.1 uses `+1e-20` (`v41:824-825`).
  - V4 has no `gate_temp` (V4.1 default is 1) and no `bias_vl` (`v41:807,819-820`).
  - V4 layers 0–2 have no `gate.bias` tensor at all.
- Engine: `v41_router.cu` `select_fast_kernel` needs a hash mode (a token-id table lookup instead of topk), and hardcoded 5120/384 must become parameters. Pass `bias_vl=null`. This goes in `v41_backbone_router.rs`.

## 4. MoE / Expert

- **Expert maths:** identical, including the swiglu clamp (up ±10, gate ≤10), routing weight applied before `w2`, and one shared expert (`v4:592-649` ≡ `v41:830-904`).
- **Routed-expert FP4:** identical: `[out, in/2]` e2m1 with an e8m0 scale per 32 along K.
- **Differences are all in quantisation blocks:**
  - Activation quant for the FP4 GEMM is 128-block in V4 (`v4:120`, `k4:458`) vs 32-block in V4.1 (`v41:187-195`).
  - The shared expert and all other FP8 linears use 128×128 weight scales in V4 vs 32×32 in V4.1.
- Engine: `v41_expert_pack.cu`, `v41_experts.rs`, `v41_route_reduce.cu` and `v41_fp8.cu` (shared_swiglu) are reusable with dimensions parameterised (4096/2048). §9 covers the scale handling.
- **Pro:** routed experts are EXL3 (`model.layers.N.mlp.experts.E.{gate,up,down}_proj.{trellis,suh,svh,mcg}`, including MTP). `v41_exl3_wire.cu` may be reusable. Naming: gate=w1, up=w3, down=w2.

## 5. Block and mHC

- **Pre-mix timing, the biggest structural difference.**
  - V4 (`v4:680-707`): `hc_pre` computes `pre, post, comb` from the stream at this sublayer's input and collapses with **this** `pre`.
  - V4.1 (`v41:948-994`) lags: attention collapses with the previous FFN's `pre`, and FFN with this attention's `pre`. It seeds with a one-hot `[1,0,0,0]` (`v41:1159-1163,1260`).
  - Engine: `v41_hc.rs` `incoming_pre`/`next_pre` must be rewired so the fresh pre from `mixes_kernel`/`finish_mixes` feeds `pre_kernel` in the same sublayer. That serialises mixes before collapse, so the V4.1 overlap disappears.
- **Unchanged:** the Sinkhorn kernel (`k4:371-438` ≡ `k41:406-475`) and `hc_post`.
- **rsqrt epsilon:** 1e-6 in V4 vs 1e-20 in V4.1.
- **hc_head, V4 only** (`v4:709-716`): `pre = sigmoid(mixes·scale + base) + hc_eps`, with a single-scalar scale, no Sinkhorn and no post.
  - Tensors: `hc_head_fn [4, 4·dim]`, `hc_head_base [4]`, `hc_head_scale [1]`, both at top level and in `mtp.2`.
  - V4.1 has no such tensors; the final collapse reuses the last lagged pre (`v41:1268`).
  - Engine: new small kernel. `project_kernel` in `v41_hc.cu` can compute the 4 projections, then apply sigmoid+eps.

## 6. Head, embedding, final norm

- **Embedding:** the same `ParallelEmbedding`.
- **Final path:** V4 is hc_head → `norm` (eps 1e-6) → `head` in F32 (`v4:922-923`). V4.1 is hc_pre → norm → head.
- `head.weight` is BF16 `[V, dim]` in both, not tied.
- Engine: `v41_target_head.rs` is reusable apart from the hc_head pre-step and eps; `v41_target_embedding.rs` as-is, with dim as a parameter.

## 7. DSpark / MTP

- **Stages and block:** 3 stages, block 5, noise id 128799 in both. Stages are window-only (ratio 0, θ 1e4, no YaRN).
- **Shared vs own weights:** embed and head are tied to the backbone (convert skips `mtp.*emb*`/`head`). Each stage owns its attention, MoE and HC weights; nothing is shared with target experts. Expert counts are V4 256/6 vs V4.1 128/3.
- **Tap point.**
  - V4 taps the **output** of layers 40–42: `h.mean(dim=2)` after the layer (`v4:919-921`).
  - V4.1 taps the **input** of 37–39, after engram (`v41:1264-1266`).
  - `main_proj` takes `3·dim` (12288→4096 in V4 Flash).
  - Engine: `dspark_tap_kernel` (mean of the 4 streams) is reusable; move the tap to after `hc_post` of the FFN.
- **Attention:** as in V4.1 (`v4:750-792` ≡ `v41:1032-1074`) except the per-head Q RMS norm (`v4:776`) and the KV quantisation format (§2).
  - Prefill only seeds the window cache from `main_x`.
  - Decode: queries at `start_pos+1..+5` see the window plus all 5 block tokens (bidirectional).
- **Markov head naming:** V4 `markov_head.markov_w1` (embedding) / `markov_w2` (head) vs V4.1 `markov_head.embed` / `.head`. Both are `[V, rank]`. Rename at load.
- **Head collapse:** V4 uses `mtp.2.hc_head_*` via hc_head (`v4:862`); V4.1 uses the lagged pre (`v41:1144`).
- **Confidence head input:** `[collapsed pre-norm hidden ‖ markov_embed]`. Width is dim+rank: 4352 (V4 Flash), 7680 (Pro), 5376 (V4.1).
- **Stage MoE gate:** normal routing with bias (not hash).
- Engine: `v41_dspark.cu` (confidence, vocabulary_row), `v41_dspark_attention.cu`, `v41_dspark_cache.*` and `v41_spark_topology.rs` are reusable once dims, Q-norm, KV format and hc_head are handled.

## 8. Engram and vision (V4.1 only)

- **Engram:** n-gram hash embeddings added into the stream at layers 1 and 14 (`v41:296-365`, `engram.py`).
- **Vision:** ViT, aligner, image span embeddings and `bias_vl` (`v41:1215-1256`, `vision.py`).
- V4 has none of these. Compile out `v41_engram.rs`, `v41_vision.*` and the `image_mask` plumbing.
- Tokenizer: vocab and merges are identical; 9 added-token names differ (e.g. 128799 is a placeholder in V4 but `<｜System｜>` in V4.1). The chat encoding module differs (`encoding_dsv4.py` vs `encoding.py`; default reasoning effort "low" vs "high").

## 9. kernel.py

- **FP8 GEMM:**
  - V4: fixed 128-group activation and weight scales, one weight scale per 128-row N block (`k4:203-273`).
  - V4.1: block 32 or 128, per-row weight-scale indexing (`k41:207-305`).
  - **Reuse trick:** replicate each V4 128×128 weight scale into 16 32×32 entries at load. This is exact, so the engine's 32-block GEMM runs V4 weights unchanged. Activations then quantise at 32 instead of 128, which is finer than the reference: a slight, benign numeric deviation. Match exactly only if bit parity is needed.
- **FP4 GEMM:** the activation group is fixed at 128 in V4 (`k4:458`) and 32/128 in V4.1 (`k41:502`); the weight format is identical.
- **fp4_act_quant:** V4 supports e8m0 only; V4.1 adds an E4M3-scale branch (`k41:160-166`) used only for V4.1 compressed KV. V4 has no use for it.
- **act_quant:** identical kernel. V4 calls it with block 64 for KV (`v4:378,512`).
- **sparse_attn:** −inf vs −1e30 init (see §2).
- **Sinkhorn:** identical.
- **New for V4:** a Hadamard transform (the `fast_hadamard_transform` external library, `v4:253-257`). V4 has no other new kernels.

## 10. Other numerics

- **RMSNorm:** the V4 weight is F32 (`v4:195`) vs the default dtype in V4.1, and eps differs everywhere. The compressor norm, `kv_norm`, `q_norm`, the HC rsqrt and `main_norm` all use norm_eps.
- **Both:** softmax/pooling in fp32, gate in fp32, logits in fp32.
- **Reference limits (both):** prefill only at start_pos 0, then one token per step. Engine chunked-prefill and multi-token-verify paths must reproduce the compressor state and hash-routing semantics themselves.

---

## Porting checklist, in order (riskiest first)

1. **Config loader and dimension parameterisation.** Map the HF keys and force n_mtp=3. Remove the hardcoded 5120/20480/15360/384/32-head/512×5120 constants from the `v41_*.cu` kernels. Required before anything else runs.
2. **Weight loader.**
   - Expand 128→32 FP8 scales.
   - Load `tid2eid` as i32.
   - Rename `markov_w1/w2`.
   - Load `hc_head_*` and the per-layer compressor/indexer tensors (including `ape` and the 2×d overlap projections).
   - Skip engram/vision.
   - (Pro only) EXL3 experts.
3. **mHC rewire (high risk).** Use the fresh pre in the same sublayer, change eps to 1e-6, and add the `hc_head` kernel for the final and DSpark heads. Validate the layer-0 output against the reference first.
4. **Q per-head RMS norm (easy to miss).** Backbone and DSpark.
5. **KV format (high risk).** FP8 non-RoPE dims plus a BF16 RoPE tail, used for both window and compressed KV. This is a new pack/store variant and a new sparse-attention dequant path.
6. **Compressor (high risk).** Ratio-4 overlap with ape, ratio-128 pooling, bigger decode/chunk state, and a per-layer compressed cache (re-budget memory for about 3.5 KB/token).
7. **Indexer.** Own compressor (dim 128), Hadamard on q and K, 64 heads, per-layer (no sharing, no candidates). Ratio-128 layers get dense compressed indices with variable width.
8. **Router.** Hash mode for layers 0–2 (and Pro's 0–2), no-eps normalisation, `bias_vl` off.
9. **MoE and shared expert.** Dims as parameters; FP4 format unchanged.
10. **Head.** hc_head → norm (1e-6) → head.
11. **DSpark.** Tap after the layer; stage MoE 256/6; confidence width dim+rank; hc_head; Q-norm; KV format.
12. **Validation.** Compare layer by layer against `v4 model.py` with fixed prompts: prefill ≥ 256 tokens so ratio-4 overlap and at least two ratio-128 groups are exercised, then a decode crossing a 128 boundary, then DSpark acceptance rate.
