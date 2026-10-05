# Multimodal input (images, then audio): design (2026-10-06)

Decides how CuteAFD runs the officially bundled vision and audio towers, where they live, how their outputs enter the language model, how embeddings and the prefix cache work together, and how all of it is qualified. It is written for Codex agents who will implement one work package (§10) per brief. Sources: PLAN.md "Release v2 scope" item 4, the V4.1 vision path, the checkpoints' own `config.json` and modeling code, pinned `third_party/transformers` (`qwen4_exp`, `glm5_next`), Hugh Madden's `mimo26f-afd` v1.3.0 (MIT: `crates/mimo26-coordinator/src/vision.rs`, `kernels/vision.cu`, `crates/mimo26-image`) and `docs/fidelity-design.md` (origin/work/fidelity-v2).

## 0. Decisions in one box

- **Scope:** every family whose checkpoint ships a tower: DeepSeek V4.1 Flash (done), MiMo V2.6 Pro MOPD and Flash MOPD (vision + audio), Qwen 3.8 Flash Next (vision), GLM 5.3 Flash (vision). GLM 5.3, DeepSeek V4 Flash/Pro and MiMo V2 Flash ship none and stay text-only. Every quant of these checkpoints (official FP8, EXL3, NVFP4) keeps the tower in **BF16**, so no quantized encoder kernels are needed. Video is out of v2 (400).
- **One shared encoder service.** A data-driven ViT runner (a `TowerSpec` per family) on a dedicated owner thread with its own low-priority CUDA stream. It encodes one image at a time, deterministically (no batch effects), uses scratch admitted at load, and never runs on the scheduler thread.
- **Placement is a planner decision.** `VISION=auto|off|rtx[:gpu]|spark[:rank]`, default `auto` = **Spark** whenever Sparks exist. On a Spark the tower runs inside the `expertd` process (same CUDA context) on the rank with the most free memory and the least expert work, or as an encoder-only `expertd` on an idle Spark. If no Spark has room, the planner falls back to the RTX (GPU1 on 2-RTX hosts) only if the KV target (2M PRO 6000 / 1M 5090) still holds; otherwise `off` with a shortfall report. RTX-only layouts use the RTX. V4.1 keeps its RTX tower in v2. `off` reads zero tower bytes.
- **Interface to the LM is per family, the mechanism is shared.** A tokenized placeholder is expanded to N placeholder rows; a shared kernel scatters feature rows over the gathered token embeddings, replicated across HC lanes. Native token ids stay authoritative for every side input (Qwen PLE n-grams, V4.1 Engram, drafters). Positions: V4.1/MiMo 1-D, GLM Flash none (no LM RoPE), Qwen interleaved M-RoPE (a new per-row 3-D rope table; the row index keeps paging and causality).
- **Image identity:** `ImageKey = SHA-256(tag ‖ EncoderId ‖ PreprocessId ‖ grid ‖ resized RGB8)`. Prefix-cache keys for image rows are a deterministic 31-bit function of `(ImageKey, row)` with bit 31 set (Hugh's scheme), so there is no interning, no recycling, and the host tier is safe. Every snapshot stores its `(start, len, ImageKey)` list and restore verifies all 256 bits. No snapshot frontier lies strictly inside an image span.
- **Embedding cache:** coordinator host RAM, LRU by bytes, default 8 GiB, keyed by `ImageKey`, pinned while a request uses it, touched on every reference including prefix hits. Plus an API-side source memo (raw-bytes SHA-256 → prepared image) so a long session that resends 40 images per turn hashes them instead of decoding them. No GPU-resident and no Spark-side cache tier in v2.
- **Gates:**
  - preprocessing byte-exact against the reference PIL path;
  - tower against the official FP32 module: relative L2 ≤ 0.03, mean cosine ≥ 0.9995, worst-token cosine ≥ 0.99, byte-deterministic;
  - LM effect: paired encoder-swap KL ≤ 0.005 nat and Δtop-1 ≤ 0.5 point on fidelity vision windows;
  - end to end: image-QA smoke ≥ 22/24;
  - prefix: five exact-restore scenarios with images;
  - text paths: unchanged (Qwen text-only byte-identical under M-RoPE; V4.1 untouched).

## 1. What V4.1 already does, and what generalizes

| Piece (V4.1 file) | V4.1-specific | Generalizes as |
|---|---|---|
| `cuteafd-loader/.../v41_image.rs` | 14 px patches, 3×3 unfold, ≥544² pad (no resample), mean/std 0.5, `convert("RGB")` (drop alpha, no EXIF), span `h·(w+1)+2` tokens with start/newline/end rows, 1024 tokens / 16 images | `ImageProcessor` trait; decode policy flags; `ImageKey`; span expansion |
| `v41_vision.rs` + `v41_vision.cu` | 32×1024 blocks, widths hard-coded (1024/2816/5120/9216), cuBLAS batched-GEMM attention (non-flash), aligner + span inside the tower | Owner pattern (preallocated slots, observer for stage dumps, drain on error) → `VitRuntime` with a `TowerSpec`; kernels generalized by width |
| `v41_requests/images.rs` (`RequestImages`) | `ROW_BYTES = 10240` | `RequestMedia`: lazy per-span features, `needed(resume, end)`, per-chunk rows and mask, hidden width from the family |
| `v41_native_serve/prefix/images.rs` (`ImageKeySpace`) | Interned u32 keys ≥ 2^31 recycled through `Weak` | Concept kept (vocabulary-disjoint keys, native tokens authoritative); mechanism replaced by deterministic keys plus full-key verification, because recycled keys are unsafe for a host tier that outlives device pins |
| `scheduler.rs` admission | Encodes synchronously in admission, blocking the serve loop for every request; D2H then H2D per chunk | Asynchronous: peek → encode → admit; host staging per chunk kept (≤ 32 MiB per 4K-row chunk ≈ 1.3 ms H2D) |
| `cuteafd-api/src/openai/images.rs` | `Vec<V41Image>`, DeepSeek renderer only | `MediaPreparer` with the family's processor; templated families must pass image parts through (today they are dropped, see §7.1) |
| `v41_target_embedding.rs` + `cuteafd_v41_vision_embed` | 4 mHC lanes, width 5120 | `cuteafd_embed_inject(features, indices, out, rows, width, copies)` in `native/shared/cuda/embedding.cu` |
| `scripts/qualify/deepseek_v41/qualify-*.py` | V4.1 reference loader | Template for per-family component gates (relative L2 0.03, cosine 0.9995) |

V4.1 is the regression anchor: v2 changes none of its serving path. WP‑11 (optional, after v2's gates) moves it onto the shared keys, cache and service.

## 2. Shared architecture: the encoder service

### 2.1 Components

```
cuteafd-api      MediaPreparer: fetch/data-URL → decode → family preprocessing → PreparedImage {key, grid, rgb8, tokens}
                 SourceMemo: sha256(raw bytes) → Arc<PreparedImage>  (LRU, 512 entries)
cuteafd-loader   media/: decode policy, Pillow-exact bicubic, smart_resize variants, ImageKey/EncoderId/PreprocessId,
                 span expansion and validation per family (placeholder ids from config.json)
cuteafd-engine   media/: MediaKeys (radix keys, snapshot image lists, verify), EmbeddingCache, RequestMedia,
                 EncoderClient trait {submit(job)->ticket, poll, cancel}, in-flight dedupe by ImageKey
cuteafd-daemon   shared/vision/: TowerSpec, VitRuntime, EncoderService (owner thread), LocalEncoder / RemoteEncoder
                 expertd: --encoder (thread inside the expert process) and --encoder-only (idle Spark)
native/shared    cuda/vision.cu + include/cuteafd_vision.h (generalized from v41_vision.cu and Hugh's vision.cu);
                 embedding.cu: cuteafd_embed_inject; b12x varlen attention AOT (non-causal, windowed, GQA)
cuteafd-ffi      shared/vision.rs
```

### 2.2 Where a tower can run

| Option | Verdict | Reason |
|---|---|---|
| Spark, inside `expertd` (own thread, own lowest-priority stream, same context) | **Default** | Keeps RTX memory for KV and local experts. One CUDA context means block-level scheduling by stream priority, not context time-slicing. Shares the loaded native library, memory ledger and sparknest reader. Lives and dies with the rank. |
| Spark, encoder-only `expertd --encoder-only` | Preferred when the layout leaves a Spark idle (e.g. GLM Flash TP2 on a 6-Spark pool) | No interference at all |
| Spark, separate process beside `expertd` | Rejected | Second context: GPU time-slicing against expert waves with no priority control; duplicate runtime |
| RTX resident (GPU1 on 2 RTX, else GPU0) | Fallback; default for RTX-only layouts; V4.1 in v2 | About 4× faster encode; costs 1.4–2.0 GiB |
| RTX transient (Hugh: weights in pinned host RAM, uploaded per request) | Rejected for v2 | Admission-before-allocation still has to reserve weights plus scratch, so peak memory is unchanged; adds a 1.4 GB H2D (~30–60 ms) per encode |
| Host CPU (Threadripper 9970X, AVX-512 BF16) | Rejected | Estimated 3–6 TF/s effective → 1–2 s per 1024-token image; would need a second implementation and its own gates |

### 2.3 Spark placement mechanics

- **Rank choice.** The planner picks, in order: an idle Spark in the pool; else the expert rank with the lightest slice (e.g. ranks 4–5 under uneven TP6 slices); else the most free memory after experts, rings and the OS. It requires `tower + scratch + 1 GiB` free. Ledger reference: MiMo Pro TP6 Spark has 4.0 GiB free before the page-cache drop and ~13.5 after; GLM Flash TP4 78 GiB; V4.1 TP4 at 2 RTX 71 GiB.
- **Loading.** The rank streams only the tower's extents from sparknest (no replication), in parallel with its expert load. About 1.4 GB at ~5 GB/s ≈ 0.3 s. Readiness waits for every enabled component.
- **Wire.** Control and data run over a dedicated TCP connection to the rank in WP‑4. The request carries the resized RGB8 (`patches × 768 B`, or 588 B for GLM's 14 px patches) plus grid and `ImageKey`; device-side normalization uses a 256×3 LUT that is byte-exact with the CPU reference. The reply is BF16 rows into the coordinator's host embedding cache. Payload per 1024-token image: 3 MiB up; 8 MiB back (MiMo Flash; 12 MiB Pro, 5 MiB Qwen, 8 MiB GLM). That is ~0.4 ms at 200 Gb/s and ~3–4 ms over TCP, against ≥ 130 ms of GB10 encode.
  - Switch to a verbs QP writing into a registered host slab only if TCP exceeds 5% of encode wall time.
  - This is a deliberate exception to the "device-driven exchange" rule: it is a cold path whose destination is host RAM (the cache), and the hot prefill path then takes the same H2D V4.1 already takes.
- **Handshake.** The rank reports `EncoderId` (family, checkpoint revision, tower tensor-header digest, engine encoder numerics version, SM arch) and its capacity in patches. The coordinator refuses a mismatch.
- **Replicas.** `--vision-replicas N` (default 1) places towers on N Spark ranks. The service spreads one request's images across them (16 images on GB10: ~2.6 s serial vs ~0.65 s on 4 replicas). Default stays 1 until a measured need exists.
- **Interference control.** Lowest stream priority, and every kernel launch bounded (GEMM tiles by row chunks of 4096, flash attention CTAs). WP‑4 measures a concurrent stream's C1 decode on the same Spark while a 4096-token image encodes.
  - **Stop bar:** more than 30% slowdown over the encode window, or any slowdown when no image is in flight. Either one moves the default to an idle Spark or the RTX for that layout. The "no image" case must read 0%: an idle encoder thread launches nothing.
- **Failure.** If the encoder rank fails or is unreachable, image requests get 503 "vision encoder unavailable", text continues, and `/health` reports `vision: failed`. If the encoder rank is also an expert rank, the existing expert-failure path governs.

### 2.4 Default placement per hardware class

| Hardware | Default `VISION=auto` | Fallback |
|---|---|---|
| 1× PRO 6000 + Sparks | Spark (rule above) | GPU0 if the 2M-token KV target holds with tower + scratch charged; else off + shortfall |
| 2× PRO 6000 + Sparks | Spark | GPU1 (typically 60–80 GiB free for the generic families per the 2026-10-03 ledger); else off |
| 5090 (32 GB) + Sparks | Spark | GPU0 only if the 1M target holds (rarely); else off + shortfall |
| RTX only (e.g. Qwen maximum, one PRO 6000) | GPU1 / GPU0 if the KV target holds | off + shortfall |
| V4.1 (any) | RTX, unchanged | — (WP‑11 adds Spark) |

D1 for TJ: on 2× PRO 6000 the GPU1 tower would cost nothing material against the 2M target and saves ~100 ms TTFT per new image. This design keeps Spark as the default per TJ's stated preference; flipping 2× PRO 6000 to `rtx:1` is a one-line policy change.

### 2.5 Latency budget against time-to-first-token

Estimated encode time per image. Effective BF16 rates are assumed: PRO 6000 150 TF/s, 5090 100 TF/s, GB10 40 TF/s; the GB10 F16 MMA peak measured 124.8 TF/s per PLAN. Hugh measured 30–100 ms for upload plus encode of ~300-token images on a 5090, which is consistent. WP‑3 and WP‑4 replace these with measurements.

| Tower | LM tokens | TFLOP | PRO 6000 | 5090 | GB10 | Embeddings back |
|---|---:|---:|---:|---:|---:|---:|
| MiMo (28×1280, 24 windowed) | 256 / 1024 / 4096 | 1.5 / 6.3 / 31.9 | 10 / 42 / 213 ms | 15 / 63 / 319 ms | 37 / 158 / 798 ms | 2 / 8 / 32 MiB (Pro ×1.5) |
| Qwen (27×1152, all full) | 256 / 1024 / 4096 | 1.0 / 5.5 / 47.2 | 7 / 37 / 315 ms | 10 / 55 / 472 ms | 25 / 138 / 1180 ms | 1.25 / 5 / 20 MiB |
| GLM Flash (24×1024, all full) | 256 / 1024 / 4096 | 1.0 / 5.3 / 40.9 | 7 / 35 / 273 ms | 10 / 53 / 409 ms | 25 / 132 / 1023 ms | 2 / 8 / 32 MiB |
| V4.1 (32×1024, 3×3 merge) | 256 / 1024 | 2.6 / 18.9 | 18 / 126 ms | 26 / 189 ms | 66 / 472 ms | 2.5 / 10 MiB |

Typical inputs: 640×480 → 300 tokens; 1280×720 → 880 tokens (MiMo/Qwen) or 1196 (GLM); 1920×1080 → 2040 or 2691; a 4K screenshot → 8160 tokens before our cap.

Budget rule: on the default placement, encoding a ≤ 1024-token image must add ≤ 250 ms to TTFT and ≤ 1.0× the prefill time of its own tokens. The Spark default meets this by estimate (~160 ms vs ~200 ms of prefill at ~5k tok/s). If WP‑4 measures above it, the planner prefers RTX for that layout.

Overlap: once WP‑5 lands, text rows before the first uncached image prefill while the Spark encodes. This helps fresh prompts; agentic turns, whose prefix restores instantly, gain nothing.

### 2.6 Memory per tower (weights from safetensors headers; scratch at the 4096-token cap)

| Tower | Weights | Scratch (16,384 patches, 4096-row GEMM chunks) | Total |
|---|---:|---:|---:|
| V4.1 | 925.6 MiB | as today (9216 patches) | unchanged |
| MiMo Flash vision | 1389.7 MiB | ≈ 0.6 GiB | ≈ 2.0 GiB |
| MiMo Pro vision | ≈ 1410 MiB (merger out 6144) | ≈ 0.6 GiB | ≈ 2.0 GiB |
| Qwen vision | 856.3 MiB | ≈ 0.5 GiB | ≈ 1.4 GiB |
| GLM Flash vision | 1075.0 MiB | ≈ 0.5 GiB | ≈ 1.6 GiB |
| MiMo audio (patch encoder 448 + speech embeddings 50 + tokenizer encoder ≈ 622; the 1.15 GiB tokenizer decoder is not loaded) | ≈ 1120 MiB (Pro ≈ 1185) | ≈ 0.2 GiB | ≈ 1.3 GiB |

## 3. Interface between encoder and LM

### 3.1 Common mechanism

1. The template renders one placeholder per image: MiMo/Qwen `<|vision_start|><|image_pad|><|vision_end|>`, GLM `<|begin_of_image|><|image|><|end_of_image|>`, V4.1 its own.
2. After tokenization, the family's `SpanExpander` checks that the placeholder count equals the number of images, that each placeholder sits between its start/end tokens, and that every token is below the vocabulary size; then it expands each placeholder to N rows. Literal special tokens typed into user text therefore fail as a count mismatch (400).
3. The expanded prompt length counts against context and KV admission. `prompt_tokens` includes image rows; `prompt_cache_hit_tokens` includes restored image rows.
4. Prefill: for chunk `[a, b)`, `RequestMedia` gathers the feature rows of every overlapping span into the pinned staging buffer (one H2D). The family's embedding stage runs its normal token gather, then `cuteafd_embed_inject` overwrites the image rows by index and replicates them across HC lanes (`copies` = 4 for Qwen, GLM Flash and V4.1; 1 for MiMo).
5. Decode never sees image rows. Images live only in prompts, so decode graphs are unaffected (zero steady-state captures preserved).
6. Drafters: wherever a drafter consumes a token embedding at an image row (Qwen MTP's `fc_embedding` input, MiMo MTP/DFlash prefill), it gets the injected feature row, as vLLM/SGLang do. Drafts are verified, so this affects acceptance only, never output.

### 3.2 Per family

| Family | Placeholder id (start/end) | Rows per image | LM positions | Family-specific work |
|---|---|---|---|---|
| V4.1 | 129264 (tower emits start/newline/end rows) | `h(w+1)+2` | 1-D | none (done) |
| MiMo Flash/Pro | 151655 (151652/151653) | `(H/32)(W/32)` | 1-D (`position_ids = cache_position`) | injection in the MiMo embedding stage; width 4096 / 6144 |
| Qwen 3.8 | 248056 (248053/248054) | `(H/32)(W/32)` | **interleaved M-RoPE**: `mrope_section [11,11,10]` over the 32 rotary pairs (`partial_rotary_factor 0.25` × head 256); text rows `t=h=w=p`; an image at `s` with LLM grid `h'×w'` gets `t=s, h=s+r, w=s+c`; the next text starts at `s+max(h',w')` | new rope table, AOT re-export, PLE on native ids (§3.3) |
| GLM 5.3 Flash | 154854 (154830/154831) | `≈(H/28)(W/28)`, GLM's own `smart_resize` | none (MLA `qk_rope_head_dim 0`, KDA) | injection into 4 mHC streams only |

### 3.3 Qwen M-RoPE in our engine

- Qwen programs take `positions` (i64 per row) and use it for paging-adjacent indexing, causality (`qwen4_sparse_gqa`, `qwen4_index_topk`, `qwen4_index_expand`) and RoPE. Keep `positions` = row index. Add `rope_positions: [rows][3] i32`, read only by the RoPE sites: the attention producer's q/k, the QSA indexer's q/k (block keys are rotated at their group-start row's M-RoPE position), and the MTP layer.
- Angle per rotary pair `i`: `pos[sel(i)] * inv_freq[i]`, where `sel(i)` = H for `i ≡ 1 (mod 3), i < 33`, W for `i ≡ 2 (mod 3), i < 30`, else T (reference `recomposition_frequencies`).
- With `t=h=w` this is the same arithmetic as today: the text-only gate is **byte-identical** logits.
- The positions are a pure function of native tokens and span grids. The host computes them per chunk; decode rows use `row + δ`, where `δ = Σ_images (max(h',w') − h'·w')`. Nothing is added to the prefix mark: a restored request recomputes them from its own span list, exactly as `history_of` recomputes the PLE context.
- PLE n-gram ids come from native ids (the placeholder 248056 per row), as the reference does with `ple_input_ids = input_ids`. Never from radix keys.

## 4. Image identity, embedding cache and prefix-cache integration

### 4.1 Keys

- `EncoderId = SHA-256("cuteafd-encoder-v1" ‖ family ‖ snapshot revision ‖ digest of the tower tensors' safetensors header entries ‖ ENGINE_ENCODER_NUMERICS (u32, bumped on any kernel rounding change) ‖ SM arch)`. The SM arch is included because SM120 and SM121 builds need not agree bitwise.
- `PreprocessId = SHA-256` of the processor parameters: patch, merge, temporal, mean/std, resample, min/max pixels, the effective token cap (including `detail`), and the decode policy (EXIF, alpha).
- `ImageKey = SHA-256("cuteafd-image-v1" ‖ EncoderId ‖ PreprocessId ‖ grid t,h,w ‖ resized RGB8)`. RGB8 is hashed before normalization; normalization is a function of `PreprocessId`. The V4.1 path keeps its current identity until WP‑11.
- `AudioKey` (WP‑10): the same structure over 24 kHz mono PCM (f32 LE) plus mel and tokenizer parameters.

### 4.2 Tiers and eviction

| Tier | Where | Holds | Size / policy |
|---|---|---|---|
| Source memo | API process RAM | raw-bytes SHA-256 → `Arc<PreparedImage>` (grid, key, RGB8) | 512 entries LRU; skips decode and resize for resent history |
| Request media | host, per request | `Arc<[u8]>` feature rows per span, installed lazily | lives with the request |
| Embedding cache | coordinator host RAM (not pinned; staged through the per-chunk pinned buffer) | `ImageKey` → BF16 rows | default `min(8 GiB, 5% RAM)`, carved from the host-cache headroom accounting; LRU by bytes; pinned while referenced by an admitted or encoding request; touched on every reference, including references covered by a prefix hit, so a long session's images stay warm |
| GPU | none persistent | per-chunk staging only | — |
| Spark | none in v2 | — | the coordinator is the only consumer; revisit only with measured misses |

8 GiB holds about 1000 MiMo Flash images of 1024 tokens (≈ 650 for Pro, ≈ 1600 for Qwen).

### 4.3 Prefix-cache integration (generic engines, `cuteafd-engine::prefix`)

1. **Keyed tokens.**
   - Cache lookup, capture and the host page chain (`page_chain`) see keyed tokens: image row `i` of image `K` becomes `0x8000_0000 | (mix64(fold(K), i) & 0x7fff_ffff)` (Hugh's `image_token_id`).
   - Load asserts the vocabulary is below 2^31 (all families: ≤ 248,320).
   - The model, PLE, Engram and drafters see native tokens. `Entry.tokens` stores keyed tokens.
2. **Verification.**
   - Every device `Entry` and host payload carries `media: Vec<(start, len, ImageKey)>` for the spans inside its tokens.
   - A restore of length `L` compares the full 256-bit keys of every span below `L` against the request's own spans. A mismatch is a miss, counted as `media_key_collisions`. This makes the 31-bit row keys a performance hint, not a correctness assumption.
3. **Frontiers.**
   - Prompt-end and turn-end captures are never inside a span (the end/close token follows).
   - Chunk-end points (`--prefix-point-gap`), message-boundary points and parked cancelled prefills round down to the span start.
   - Partial-rule resumes (MiMo `partial`, V4.1 replay) also round down to the span start.
4. **Peek before encoding.**
   - `PrefixCache::peek(keyed) -> resume` is a non-mutating lookup across device and host tiers. Needed spans are those with `end > resume`. For each: embedding-cache hit → attach; miss → submit to the encoder (deduped in flight).
   - The request waits in a `media_pending` queue, holding no KV or state slot (mirrors `DeferredAdmission`), while the serve loop keeps decoding others.
   - When every needed feature is ready, normal `admit` runs. If its resume is earlier than the peek's (the snapshot was evicted meanwhile), the newly needed spans are looked up and the request re-enters `media_pending` (bounded at 2 retries, then cold prefill of the whole prompt).
5. **Byte-exact contract.**
   - Restoring a prefix with images gives state byte-identical to prefilling that prefix from scratch with the same embedding bytes. Restore never needs embeddings: the snapshot already holds the image rows' effect on KV, recurrent state, positions and the PLE context.
   - Re-encoding after an embedding-cache eviction must reproduce identical bytes, which requires a byte-deterministic encoder: fixed cuBLAS algorithm, no split-K atomics, one image per launch so results do not depend on batch composition.
6. **Stats** (`/v1/stats`): `media: {encodes, encode_ms_p50/p99, cache_hits, cache_bytes, memo_hits, prefix_skipped_images, media_key_collisions, pending}`.

## 5. Per-family towers

All towers are BF16 with FP32 accumulation and one BF16 rounding per op, as V4.1 does.

| | MiMo V2.6 (Pro/Flash) | Qwen 3.8 Flash Next | GLM 5.3 Flash |
|---|---|---|---|
| Patch | Conv3d 3×2×16×16 → 1280, no bias (K=1536, still image duplicated in time) | Conv3d 3×2×16×16 → 1152, bias | Conv3d 3×2×14×14 → 1024, bias (K=1176) |
| Position | 2-D RoPE (16 + 16 freqs, θ 1e4, split-half, merge-block order) | learned 48×48 `pos_embed`, bilinear `align_corners=True` (host-computed taps and weights, device 4-tap sum) + axial RoPE (18 + 18 freqs, θ 1e4) | axial RoPE (16 + 16, θ 1e4) |
| Blocks | 28 × [RMSNorm 1e-6; qkv + bias, GQA 32q/8kv × 64; key-0 sink bias per head; o + bias; RMSNorm; SwiGLU 4608 with biases] | 27 × [LayerNorm + bias 1e-6; qkv + bias, MHA 16 × **72**; proj + bias; LayerNorm; fc1 4304 + bias, GELU-tanh, fc2 + bias] | 24 × [RMSNorm 1e-5; qkv + bias, MHA 16 × 64, **per-head q/k RMSNorm**; proj + bias; RMSNorm; SwiGLU 4096 with biases, **clamp** gate ≤ 10, up ∈ [−10, 10]] |
| Attention pattern | full at blocks 0/9/18/27; window \|i−j\| ≤ 64 elsewhere; blocks 1–4, 10–13, 19–22 row order, 5–8, 14–17, 23–26 column-major merge units (gather/scatter at type changes) | full | full |
| Merger | LayerNorm(1280) → view 5120 → Linear 5120 → GELU(erf) → Linear 4096/6144 | LayerNorm(1152) → view 4608 → fc1 4608 + bias → GELU(erf) → fc2 2560 + bias | RMSNorm(1024) post → 2×2 downsample Conv2d (GEMM, K = 4×1024 gathered as `c,kh,kw`) + bias → proj 4096 → LayerNorm + bias → GELU(erf) → SwiGLU 10240 (clamped) → 4096 |
| Preprocess | Qwen2-VL `smart_resize` factor 32, CLIP mean/std, bicubic; EXIF transpose, alpha over white (HF `convert_to_rgb`) | same, mean/std 0.5, min 65,536 px (64 tokens), max 16,777,216 (16,384 tokens) | GLM `smart_resize` (ceil-align 28, binary search into budget), CLIP mean/std, bicubic; min 16, max 8000 tokens |

Hazards each tower's golden must pin:

- **MiMo checkpoint lacks biases.** `merger.mlp.{0,2}.bias` and `merger.ln_q.bias` are missing from the checkpoint, so the reference loads them as initialized (zero for transformers 5 defaults). The golden asserts `missing_keys` is exactly these and that they are zero after load.
- **MiMo "sinks" are not b12x sinks.** They are an additive per-head bias on key index 0 of the current (possibly reordered) sequence, combined with the window mask, so they are inert for rows farther than 64 from index 0. They are **not** b12x's learnable sink (an extra logit in the softmax denominator).
- **MiMo min/max pixels conflict.** `preprocessor_config.json` (3,136 / 12,845,056) disagrees with `config.json processor_config` (8,192 / 8,388,608).
  - Under our 4096-token cap only the minimum matters (images under ~8 K pixels).
  - D3: default to `preprocessor_config.json`, which HF `AutoProcessor` loads and Hugh's crate matches. WP‑1 checks SGLang's MiMo processor and reports.
- **Qwen head_dim 72.** b12x varlen's tile table covers ≤ 64, ≤ 128 and 256, so 72 takes the ≤ 128 tile; WP‑7 adds a test.
- **Default token cap.** `--max-image-tokens 4096` per image for MiMo/Qwen/GLM (applied as a smaller `max_pixels` to the official resize, which is the official processor with a parameter). `detail: "low"` caps at 256 tokens.
  - Without the cap, Qwen's official maximum (16,384 tokens = 65,536 patches, all full attention) costs ~590 TF: ≈ 4 s on a PRO 6000 and ≈ 15 s on GB10.

Kernels: what exists, what to write (WP‑3 unless noted).

| Op | Source |
|---|---|
| BF16 GEMM + bias epilogue | cuBLAS(Lt), as `v41_vision.cu`, with `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` and a pinned algorithm for determinism. The tower is a cold path; b12x dense AOT exports per shape are not worth it unless measured. |
| Attention | b12x varlen AOT: non-causal, `window_size (64,64)`, GQA, head 64/72 (already used by `b12x.norm.vision` tests). MiMo key-0 bias: add a `key0_bias` score-mod mode to b12x varlen (sparkinfer fork); if not done within a day, port Hugh's `m26v_attn` (MIT). |
| RMSNorm / LayerNorm (any width, ± bias), GELU erf/tanh, SwiGLU (± bias, ± clamp), residual add | hand CUDA, generalized from `v41_vision.cu` (widths currently fixed at 1024) and Hugh's `vision.cu` |
| 2-D / axial RoPE (merge-block order, ± fused per-head q/k RMSNorm), unit gather/scatter (window order), merge gather (2×2, 3×3, conv order), bilinear pos-embed add, RGB8 → normalized patch | hand CUDA (small) |
| Injection | `cuteafd_embed_inject` in `native/shared/cuda/embedding.cu` |

## 6. Audio (MiMo only, after vision)

- **Input:** OpenAI `input_audio` `{data: base64, format: wav|mp3|flac}`; ≤ 300 s per clip (`max_audio_seconds`), ≤ 4 clips and ≤ 600 s per request. Video stays 400.
- **Pipeline** (`modeling_mimo_v2.py`, `audio_tokenizer/config.json`):
  1. decode → 24 kHz mono → log-mel (128 bins, n_fft 960, hop 240, window 960);
  2. tokenizer encoder: conv1 k3, conv2 k3 s2 (50 Hz); 24 layers d=1024 with 16 heads, causal, SWA window (128, 0) on even layers and full causal on odd, RoPE θ 1e4; skip connection from layer 3; LayerNorm; Conv1d k2 s2 + GELU + LayerNorm (25 Hz); segments of 6000 mel frames;
  3. RVQ encode with 20 codebooks (F32);
  4. sum of 20 `speech_embeddings` (1280×1024 each);
  5. groups of 4 frames → a 6-layer Qwen2 transformer bidirectional **within each 4-frame group** (θ 640,000);
  6. projection 4096 → 16384 → 4096/6144 (GELU, no bias);
  7. **6.25 LM tokens per second** (5 min = 1875 tokens).
  
  Placeholder `<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>` (151673 / 151669 / 151674), 1-D positions, injected like images.
- **Unknown to resolve first (WP‑10a):** the exact mel extractor (window function, power, log clamp, padding) is not in the snapshot. Locate the official MiMo-Audio-Tokenizer / SGLang processor and pin it before any kernel work.
- **Kernels:** conv1d via im2col GEMM; b12x varlen causal with left window 128; RVQ as a per-codebook distance GEMM (F32) plus argmin with first-index tie-break; the rest reuses the ViT toolkit. Cost ≈ 20 TF for 300 s (≈ 0.13 s PRO 6000, ≈ 0.5 s GB10).
- **Placement:** the same service and rank as vision. `AUDIO=off` by default until its gates pass; then `auto` follows `VISION` (D4).
- **Gates:**
  - log-mel max abs error ≤ 1e-4 against the pinned extractor;
  - RVQ code agreement ≥ 99.5% (a near-tie argmin flip is a discrete change);
  - projected embeddings within the vision component bounds;
  - audio-QA smoke ≥ 10/12 (spoken digits/words rendered with a permissively licensed TTS or recorded in-house);
  - exact prefix restore with audio.

## 7. API, limits and the no-multimodal mode

### 7.1 Existing gap

`cuteafd-api` decodes images only on the DeepSeek renderer path (`rendered.image_sources`). For templated families (GLM, Qwen and MiMo, which uses `ModelEncoding::Qwen`), `template_message` copies the `content` array into the Jinja template, which emits image placeholders, while `image_sources` is `Vec::new()`. An image request today therefore reaches the model as placeholder embeddings with no image. WP‑0 rejects image and audio parts with 400 ("this deployment has no image input; launch with VISION=auto") until the family's encoder is loaded, and adds a test per family.

### 7.2 Inputs and limits

- Inputs: `image_url` (`data:` URLs; `http(s)` URLs as V4.1 does today, with a new `--image-url-fetch off|public|any` defaulting to `public`, i.e. rejecting private and link-local addresses), `detail` (`low` ≤ 256 tokens; `high`/`auto` = cap), `input_audio`.
- Images are extracted in template order (messages in order, content items in order); the placeholder-count check (§3.1) catches any template that drops a message.
- Long agentic sessions resend all history, so limits separate history from new work:

| Limit | Default |
|---|---|
| Images per request (all, including history) | 128 |
| Images needing decode (source-memo misses) per request | 16 |
| Images needing encode (cache and prefix misses) per request | 16, and ≤ 32,768 new image tokens |
| Encoded bytes per image / decoded total per request / body | 32 MiB / 64 MiB / 256 MiB |
| Tokens per image | 4096 (`--max-image-tokens`), `detail=low` 256 |
| V4.1 | unchanged: 16 images, 1024 tokens, 96 MiB body |

- `/v1/models` adds `"capabilities": {"vision": bool, "audio": bool}`; usage adds `prompt_tokens_details.image_tokens`.

### 7.3 No-multimodal mode

`VISION=off` and `AUDIO=off` (CLI `--vision`, `--audio`):

- the planner marks `Component::Vision` (and a new `Component::Audio`) `disabled`;
- no tower tensor is read (ledger shows 0), no service thread starts, no Spark rank is told to load;
- the API returns 400 with the launch hint;
- `cuteafd plan` prints the bytes saved.

The plan hash includes the placement, so coordinator and Sparks agree.

## 8. Quality gates and the fidelity framework

| Gate | Measure | Bar |
|---|---|---|
| G1 preprocessing | 24 generated fixtures (JPEG 4:2:0/4:4:4/progressive/CMYK, PNG RGBA/16-bit/gray/palette, EXIF rotations, 1×200, 8×8, 6000×4000) × each processor, against pinned transformers' **PIL** path; resized RGB8 and patches | byte-exact; report max abs difference against the torchvision path for information |
| G2 tower | per stage (patch, every 4th block, final norm, merger) against the official module in FP32 (MiMo: snapshot `modeling_mimo_v2.py`; Qwen/GLM: pinned transformers), fixtures at 256/1024/4096 tokens | relative L2 ≤ 0.03, mean cosine ≥ 0.9995, worst-token cosine ≥ 0.99 (Hugh's MiMo: 1.5–1.7e-2, worst ≥ 0.99); missing-key assertion (MiMo) |
| G3 determinism | same image 3×, interleaved with other sizes; RTX and Spark separately | byte-identical per arch |
| G4 encoder swap (LM effect) | fidelity vision windows: paired run, engine with native features vs engine with reference-tower features injected through a probe-only hook (`CUTEAFD_MEDIA_FEATURES_DIR`) | KL ≤ 0.005 nat, Δtop-1 ≤ 0.5 point (one-sided 95%, the fidelity statistics) |
| G5 LM correctness | 2 windows per family against the official full model (tower + LM) via `golden.py --media` | inside the family's text floor (top-1 ≥ 90%, KL ≤ 0.06) |
| G6 image-QA smoke | 24 generated questions with exact answers: OCR of rendered code/terminal lines, chart values, shape counts and colours, UI element labels; greedy, thinking Low | ≥ 22/24 per family |
| G7 exact prefix with images | (a) repeat prompt: total hit, 0 encodes, identical first-token row; (b) next turn adds image B: restores turn 1, encodes only B, state at restore byte-equal to a cold prefill; (c) same text, different image with the same grid: hit stops at the span start; (d) host-tier round trip; (e) embedding-cache eviction → re-encode byte-identical, restore still exact | all pass; `media_key_collisions` = 0 |
| G8 perf | added TTFT per image (256/1024/4096 tokens) on the default placement; C1 decode with the tower loaded and idle; interference (§2.3) | idle C1 unchanged (quick parity ≥ 0.98, escalation per AGENTS); TTFT within §2.5 budget or the planner switches placement |
| G9 plan | `cuteafd plan` matrix: PRO 6000 ×1/×2, 5090, RTX-only, `VISION=off` | expected placements and bytes; `off` = 0 tower bytes |

Fidelity hookup (with `work/fidelity-v2`):

- Schema 2 windows gain `media: [{start, len, kind, key, grid, fixture}]`.
- `fidelity-set.py` adds a **vision bucket** built only from our own generated images (repository code rendered with a bundled DejaVu Sans Mono, terminal logs, matplotlib charts, synthetic diagrams; no third-party licences): 8 windows per family in the full tier, 2 in the quick tier, assistant-generated positions scored.
- `golden.py` per family gets `--media`: the official tower in BF16 for windows, FP32 for G2.
- Any later change to tower numerics (e.g. FP8 vision GEMMs) or to injection is judged by the same paired 0.5-point / 0.005-nat bars on this bucket.

## 9. Planner integration

- `ModelSpec` already flags `vision`. Add `audio` (from `audio_config` plus an `audio_tokenizer/` directory) and `Component::Audio`; `glm.rs`/`qwen.rs`/`mimo.rs` stop returning "text-only" when enabled.
- New `EncoderPlacement { kind: Off | Rtx(gpu) | Spark(rank) | SparkIdle(host), weights, scratch, replicas }`, resolved after expert placement and before KV sizing. Its bytes are charged to the chosen device's admitted budget, before pool resolution (the MiMo admission bug in PLAN item 5 is the cautionary case).
- The default policy is one pure function `default_encoder_placement(hardware, budgets, layout, kv_target)` with unit tests for each row of §2.4.
- `cuteafd plan` output: tensors, bytes, kernel requirements (`vision.sm120` / `vision.sm121`, b12x varlen exports), the chosen device and why, the KV-target effect, or an `unsupported`/`disabled` reason naming what is missing (e.g. "audio: audio_tokenizer/model.safetensors absent").

## 10. Work packages

Sizes are Codex Sol agent-days. Each package has its own branch and worktree; AGENTS rules apply (locks, builds under `~/.cache/cuteafd/builds/<task>`, measurements in commit messages, credit Hugh's files by repo/tag/file).

| WP | Content | Model / size / hardware | Depends | Gate |
|---|---|---|---|---|
| **0 Guard** | Reject image/audio parts on families without a loaded encoder (§7.1); `VISION`/`AUDIO` flags parsed into the plan; `/v1/models` capabilities | DeepSeek Flash or Sol high, 0.5 d, none | — | API tests per family; cargo test |
| **1 Host media** | `cuteafd-loader::media`: decode policy (reuse V4.1 turbojpeg decoder + flags), Pillow-exact bicubic (port `mimo26-image/resize.rs`), smart_resize variants (Qwen2-VL factor 32 with MiMo/Qwen params; GLM's), keys, `PreparedImage`, `SpanExpander`; API `MediaPreparer`, source memo, limits, template-order extraction; `NativeRequest.images` generalized with V4.1 untouched | Sol high, 2–3 d, CPU | 0 | G1 byte-exact; V4.1 image identity tests unchanged; cargo + pytest |
| **2 Engine media** | `cuteafd-engine::media`: deterministic keys, `Entry`/host payload media lists + verify, frontier rounding, `peek`, `media_pending` admission, `EmbeddingCache`, `RequestMedia`, `EncoderClient` + fake encoder, stats | Sol xhigh, 2–3 d, CPU | 0 | Unit tests (forced 31-bit collision → miss; eviction/pin; host round trip in hostcache sims); `qualify-prefix-cache.py` text-only unchanged |
| **3 ViT runtime + MiMo tower** | `native/shared/cuda/vision.cu`, `cuteafd_vision.h`, ffi, `TowerSpec`/`VitRuntime`/`EncoderService` (local), b12x varlen key-0 bias (or Hugh's kernel), injection kernel; MiMo Flash then Pro tower specs | Sol xhigh, 4–5 d, 1 RTX + 1 Spark (SM121 build) | — (CPU parts of 1/2 not required) | G2, G3 on SM120 and SM121; measured encode-time table replaces §2.5 estimates; no allocation during encode (ledger). Stop bar: key-0 bias not in b12x within 1 d → Hugh's kernel |
| **4 Spark placement + planner** | `expertd --encoder` / `--encoder-only`, TCP channel (verbs only if > 5% of encode time), handshake, replicas, readiness and health; `EncoderPlacement`, default policy, admission, plan hash, `cuteafd plan` output | Sol high, 3 d, Sparks + RTX | 3, 2 | G9; Spark encode byte-identical to local SM121; interference stop bar (§2.3); `VISION=off` 0 bytes and C1 unchanged |
| **5 MiMo serving** | Media states in `mimo_v2/serve.rs`, injection in MiMo's embedding stage, drafter rows, stats; Flash then Pro; ships on RTX placement and switches to `auto` once 4 lands | Sol xhigh, 3–4 d, cluster | 1, 2, 3 (4 for the default) | G4–G8 for Flash and Pro; 64-image history request; C4 with images; V4.1 quick parity not required (no shared hot path touched); text C1/C16 quick check on MiMo |
| **6 Qwen M-RoPE LM** | `rope_positions` table, sparkinfer AOT re-export (attention producer, indexer, MTP), host M-RoPE position builder; PLE on native ids | Sol xhigh, 3–4 d, 1 RTX | — (start day 0) | Text-only Qwen golden **byte-identical**; MTP acceptance unchanged; positions equal to reference `get_rope_index` on 50 synthetic layouts (host test) |
| **7 Qwen tower + serving** | Tower spec (LayerNorm + bias, GELU-tanh, bilinear pos-embed, head 72), injection with 4 HC copies, serving hookup | Sol xhigh, 3 d, 1 RTX → cluster | 3, 6, (1, 2) | G2–G8 |
| **8 Fidelity media** | Schema `media`, vision bucket, fixture generator, `golden.py --media` (MiMo, Qwen, GLM), probe feature-injection hook | Sol high, 2–3 d, 1 RTX for references | 1 | References reproduce; quick tier stays within the Release smoke 5-minute budget |
| **9 GLM Flash tower + serving** | Tower spec (q/k norm, clamped SwiGLU, downsample-conv merger), mHC injection in `glm5_flash`; the PR touches Hugh's GLM Flash serve loop, so coordinate with him | Sol xhigh, 3 d, 1 RTX → cluster | 3, 1, 2 | G2–G8 |
| **10 MiMo audio** | a) mel/processor research and pin (1 d); b) host audio decode/resample/mel + `AudioKey` (2 d); c) tokenizer encoder + RVQ + local transformer + projection on the toolkit (3–4 d); d) serving + gates (2 d) | Sol xhigh, ~8 d | 5 | §6 gates |
| **11 (optional) V4.1 migration** | V4.1 onto deterministic keys, embedding cache and the async service; Spark placement option | Sol xhigh, 3 d | 2, 4 | Full V4.1 parity (3 sessions per arm) |

Parallel lanes:

- **CPU:** WP‑0 → WP‑1 ∥ WP‑2.
- **GPU (one RTX):** WP‑3 → WP‑7 tower ∥ WP‑9 tower (small, serialized on `gpu1.lock`).
- **Independent:** WP‑6 from day 0; WP‑8 after WP‑1.
- **Cluster:** WP‑4 after WP‑3, then WP‑5 → WP‑7/WP‑9 serving → WP‑10.

Critical path to MiMo vision served end to end: max(WP‑1/2 ≈ 3 d, WP‑3 ≈ 5 d) + WP‑5 ≈ 4 d ≈ **9–10 agent-days**. Qwen and GLM each follow in about 3–4 days in parallel; audio ≈ 8 days after MiMo vision.

## 11. Decisions for TJ

1. **D1 placement default.** Spark everywhere Sparks exist, including 2× PRO 6000 (recommended per your stated preference). The alternative is GPU1 on 2× PRO 6000, about 100 ms faster per new image at no material KV cost.
2. **D2 per-image cap** of 4096 LM tokens (`detail=low` 256). Raise only with a measured TTFT budget.
3. **D3 MiMo preprocessing.** `preprocessor_config.json` (3,136 / 12,845,056 px; Hugh's choice) vs `config.json processor_config` (8,192 / 8,388,608). Only images under ~8 K pixels differ under D2.
4. **D4 audio default.** `off` until its gates pass, then follows `VISION`.
5. **D5 video.** Out of v2 (all three new towers support it; 400 for now).
