# MiniMax M3.1 Flash

A native multimodal MoE (428B total, 23B active) with MiniMax Sparse
Attention on every layer, quantization-aware-trained NVFP4 experts and
MXFP8 dense projections, a bundled vision tower and a bundled DSpark drafter.

Status: **under qualification**. `serve-minimax` serves the release from a
`./wip.sh` slot on one coordinator GPU with Spark TP2, TP3 or TP4 (examples in
`examples/configs/`); `cuteafd plan --layout` admits it from the slot's
`PROGRAMS.json` with the server's own allocation recipe.

## Supported checkpoints / quants

- `MiniMaxAI/MiniMax-M3.1-Flash` (revision `54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e`),
  the official release, served in its own formats: routed experts
  compressed-tensors NVFP4 W4A4 (`weight_packed`, E4M3 per-16 scales, divisor
  `weight_global_scale` / `input_global_scale`), attention, indexer, dense and
  shared projections MXFP8 (E4M3, UE8M0 per 1x32), BF16 embedding and head,
  FP32 router. 233 GiB, plus the 2.1 GiB `dspark/` drafter.

## Engineering summary

- 60 layers, hidden 6144; layers 0-2 dense (SwiGLU-OAI, 12288), 3-59 MoE:
  128 experts, sigmoid top-4 with a selection-only bias, renormalized, times
  2.0, plus one 3072-wide shared expert. SwiGLU-OAI everywhere:
  `g * sigmoid(1.702 g) * (u + 1)` with `g = min(gate, 7)`, `u = clamp(up, ±7)`
  (b12x `swigluoai_uninterleave`, gate rows first).
- Attention: GQA 64 query / 4 KV heads of 128, per-head Gemma RMSNorm on q
  and k (`x * (1 + w)`), NeoX RoPE on the first 64 channels (theta 1e7).
- MSA: a lightning indexer (4 query heads, one per KV head, over one shared
  128-wide key, both Gemma-normed and RoPE'd) scores every earlier token; each
  128-token block scores its max; the query's own block is forced; each query
  row and KV head attends its top 16 blocks (2,048 keys). Per-row selection by
  construction. Contract: b12x `attention/dsa_indexer/MSA.md`; CPU oracle
  `python/reference/families/minimax_m3/model.py` (`msa_select`).
- Cache: per layer, 64-row K and V pages for the rank's KV heads (E4M3 by
  default) and the shared index key in the paged indexer layout (E4M3 row +
  FP32 scale); 69,360 B/token. Purely paged: exact prefix restores share
  pages.
- Experts: package `minimax:nvfp4` on Spark worlds 2, 3, 4 and 6 at exact
  1536/1024/768/512 widths; bottom routed layers onboard to the RTX (TP1)
  when a Spark rank would exceed its budget.
- Vision: Conv3d (2x14x14, FP32) patch embed, 32 CLIP layers (1280, 16 heads
  of 80) with 3D axial RoPE on 78 channels, a GELU projector to 6144 and a 2x2
  patch-merge GELU MLP. Images smart-resize to at most 451,584 px (576 tokens:
  the `dynamic_res` pinpoints are inactive, so large images shrink and never
  tile); video samples 1 fps and attends in segments of 4 temporal patches
  (8 frames). See [Vision](#vision).
- Speculator: the bundled DSpark drafter (5 dense sliding-4096 layers, blocks
  of 7, Markov rank 256, confidence head; taps 3/17/31/45/59), or
  `RadixArk/Minimax-M3.1-Flash-Dflash2` (revision `923bf1d4b567d5ccb71b0488c6e61b5e4794e92a`,
  `--speculator dflash2 --draft SNAPSHOT`) on the shared DFlash2 drafter: 5 Gemma-norm
  sliding-4096 layers, 64 query / 4 KV heads, NeoX RoPE on 64 channels, SwiGLU-OAI, blocks of
  an anchor and 7 drafts, the same taps; it borrows the target's embedding and BF16 head.

## Serving behaviour

- Thinking follows the vendor's documented controls on chat, Responses and
  Anthropic requests. `thinking.type: enabled` opens `<mm:think>` in the prompt;
  `disabled`, `enable_thinking: false` or effort `none`/`minimal` turn it off
  (the prompt ends in the close marker); absent is adaptive (the model decides).
  An effort (`low`..`max`, or an Anthropic budget: <=1024 low, <=4096 medium,
  <=16384 high, <=32768 xhigh, else max) is rendered only while thinking is on.
- Tool arguments: a `strict: true` tool's call must match its schema or it is
  returned as text; other tools are read best effort, as vLLM's `minimax_m3`
  parser does (values coerce to the declared type when they parse, a repeated
  parameter becomes a list).
- DFlash2 (`SPECULATOR=dflash2`, `SPECULATOR_MODEL_ID` defaults to the RadixArk draft) taps
  the FP32 residual's last 4096 rows of every step into the shared drafter's ring, on the
  target's stream (the decode recordings carry the taps). Its block attends as SGLang serves the
  release: a per-row 4096 window and causal block rows (no `is_causal` in the config). The
  GEMMs load as one E4M3 copy by default (`SPECULATOR_FP8=off` keeps the checkpoint's BF16),
  the rings as FP8 K/V (`DRAFT_KV_CACHE=bf16`). `minimax-golden --draft SNAPSHOT
  --draft-replay P` replays it on a golden window's taps; `--draft-oracle DIR` compares it with
  `python/reference/families/minimax_m3/dflash2/reference.py`. The selector scores each top-16
  candidate as its unary logit plus the BF16 edge in FP32, as SGLang's DFlash2 selector does (the
  GLM ports round the score to BF16; hidden `--draft-selector-score bf16` keeps that A/B). Every
  request drafts its whole room; the shared scheduler then prices each draft's selector features (margin, candidate
  probability, entropy, rank) through the keyed `SelectorFit`
  (`draft-confidence.<numerics>.json` beside the drafter's config; the generic GLM-5.3 fit until
  one is fitted). With `SPECULATION_TRACE` the scheduler writes the `draft_confidence` records
  `scripts/bench/calibrate-draft-confidence.py` fits from a fixed-7 run (`SPECULATOR_DRAFTS=7`).
- DSpark draft length uses the shared `simple` policy (default;
  `DRAFT_POLICY`/`DRAFT_ROW_COST` select others, `SPECULATOR_DRAFTS=N` fixes the
  depth). TP2 on a 5090, aggregate tok/s at C1/C4/C16: target only
  26.3/48.6/76.9, fixed depth 7 37.2/61.2/120.9, `simple` 41.2/69.7/114.5. The
  C16 shortfall against fixed 7 is open in the shared policy.
- Fidelity: `minimax-golden` against `golden.py --activation served-cache` on
  the standard 64-window set (`fidelity-set.py` text recipe; 13,534
  generated-role rows) scores top-1 0.957, KL 0.0186 nat and NLL 0.2520 against
  the reference's 0.2536 at TP2, decode- and prefill-shaped alike. The gate is
  top-1 >= 0.94, KL <= 0.03 and NLL at most 0.01 above the reference. The
  reference's own BF16-K/V arm differs from its FP8-K/V arm by the same amount
  (top-1 0.955, KL 0.0186, NLL 0.2537): FP8 K/V costs no NLL. Rows outside the
  chat template (raw text with no turn marker, text after an end of turn) are
  near-degenerate and swing between any two arithmetics (KL 0.6-4.7 nat between
  the two reference arms), so the gate scores generated rows only. The bench
  fidelity panel needs the set published as a dataset with a `PUBLICATIONS` row.

## Planned cards

`cuteafd plan --layout` on the release headers with the slot's `PROGRAMS.json`
and the example configs' shapes: 64 sequences, two prefill lanes of 4096 rows,
1M context, FP8 K/V, vision off, the default 114 GiB per Spark rank. Pools are offline forecasts; serving re-admits measured free memory.

| Coordinator | Sparks | Routed layers on the RTX | KV pool, speculator off | KV pool, DSpark | Spark GiB/rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5090, 31.8 GiB | 2 | 0 | 179,712 | 132,288 | 112.9 |
| RTX 5090, 31.8 GiB | 3 | 0 | 179,712 | 132,288 | 76.8 |
| RTX 5090, 31.8 GiB | 4 | 0 | 179,712 | 132,288 | 58.8 |
| RTX PRO 6000 | 2 | 0 | 1,162,880 | 1,115,392 | 112.9 |
| RTX PRO 6000 | 3 | 0 | 1,161,472 | 1,113,984 | 76.8 |
| RTX PRO 6000 | 4 | 0 | 1,160,000 | 1,112,512 | 58.8 |

## Changelog

| Date | Change | Basic eval |
| --- | --- | --- |
| 2026-10-23 | Planner family, tensor contract, cache layout | — |
| 2026-10-23 | Re-pinned to `54e4e69`: the release rewrote its history (README/docs only; every weight, config, tokenizer and template blob is identical to `9e7ebac`) | — |

## Vision

The bundled tower runs natively: BF16 cuBLAS GEMMs with FP32 accumulation,
the b12x D80 varlen attention export (build with
`CUTEAFD_WIP_VISION_ATTENTION_AOT=ON`) and q/k rotated in FP32 as MiniMax's
serving applier does. `--vision auto` (the default) keeps it on the
coordinator GPU, as the other families do, and the KV pool takes what is left
(startup logs the cost in pool tokens); `--vision-peers` with `expertd-native
--encoder` on the ranks hosts it on the Sparks instead.
Preprocessing is the release's: smart resize (factor 28); video samples 1 fps
keeping the last frame, scales the longer side to at most 756 px and gives
every temporal pair its own image delimiters and timestamp.

- Preprocessing parity (CPU): `python/tools/qualify/minimax_m3/processor-parity.py`
  runs the release's own processors (Transformers 5.14.0): patches, grids and
  prompt expansion are bit-identical for 16 images and 9 frame lists, and the
  1 fps sampler matches sglang's. MP4 frames are scaled by ffmpeg's bicubic
  rather than torch's, so encoded video differs in about 1.5% of bytes (±1).
- Tower parity (real weights, RTX 5090, `qualify-vision.py`): the native
  output stays within the BF16 reference's own distance from an FP32
  reference (per-row p90 and mean cosine); repeats are bit-stable with no
  device allocation per encode.

| Input | Tokens | Encode ms | Output rel. L2 vs FP32: native / BF16 ref |
| --- | ---: | ---: | --- |
| chart 512x512 | 324 | 16.7 | 0.015 / 0.015 |
| one 672x672 tile | 576 | 26.6 | 0.060 / 0.036 (row p90 0.092 / 0.093) |
| 1344x1344 (shrinks to 672x672) | 576 | 26.4 | 0.084 / 0.073 |
| banner 2016x672 | 533 | 25.0 | 0.037 / 0.043 |
| 8-frame segment 504x644 | 1,656 | 94.9 | 0.090 / 0.074 |
| 8-frame segment 896x896 | 4,096 | 377.5 | — |

Served at TP2 (two Sparks; RTX 5090 coordinator, FP8 KV, 4096 prefill
rows, 16 sequences), the resident RTX tower takes 2.57 GB (its 2.50 GB ledger
and module): at 1M context the largest admitted pool drops from 189,568 to
152,512 tokens (-37,056), which is what the default serves, and the
reference's 65,536-token pool fits either way.

A Spark-hosted tower does not fit at TP2. It needs 3.57 GB per rank
(weights, scratch, 1 GiB guard) beside 117.44 GB of experts, rings and
workspace, against a 120.3-122.4 GB budget (MemAvailable - 3 GiB), and
admission refuses it by name. At TP3 (three Sparks, a tower on
every rank, `--vision-peers` on the coordinator) all the live checks below
pass, and the coordinator keeps the 2.57 GB. TTFT there, with text of the
same length (two runs, no throttling):

| TP3, Spark-hosted tower | Prompt tokens | s | text only, s |
| --- | ---: | ---: | ---: |
| one 672x672 image | 774 | 0.84-0.86 | 0.69-0.71 |
| 6 s 640x480 MP4 | 1,892 | 1.82-1.85 | 1.18-1.19 |
| the same 7 frames as PNGs | 1,892 | 1.71-1.72 | 1.18-1.19 |

A remote image costs about 0.15 s over text there, against 0.06 s for the
tower on the 5090: the GB10 encode and the round trip. Text TTFT is lower at
TP3 because three ranks share the experts.

The live checks (`scripts/qualify/minimax_m3/qualify-native-vision.py` on the
`generate-media-fixtures.py --set semantics` fixtures) count and colour shapes,
read a four-quadrant image, a banner, a chart and a terminal, order two images,
and read a clip's numbers and motion from MP4 and from frames. Exact
repeats of an image and a video request restore the whole prompt; leading
media shared with an earlier, different prompt is not a hit, since MiniMax
restores prompt-end and turn-end snapshots only.

| TTFT (streamed, fresh media, median of 3) | Prompt tokens | s |
| --- | ---: | ---: |
| one 672x672 image | 774 | 0.875 |
| 1344x1344 image | 774 | 0.910 |
| text only | 796-804 | 0.810 |
| 6 s 640x480 MP4 (7 sampled frames, one segment) | 1,892 | 1.735 |
| the same 7 frames as PNGs | 1,892 | 1.555 |
| text only | 1,908-1,932 | 1.305 |

The scheduler starts a prefill once its features are ready. The prefill
takes 0.84 s for the image prompt and 1.34 s for the clip against 0.80 s and
1.30 s for text of the same length. Before it, the image spends about 35 ms
(decode, resize, 27 ms encode), the PNG frames 0.22 s and the MP4 0.40 s: the
95 ms encode plus preparation, where the MP4's ffprobe and ffmpeg take 72 ms
and 155 ms in the coordinator container (a process start alone is about 70 ms
on the coordinator host). No thermal throttling.

Text throughput with the tower resident, same slot and day, vision off -> on:
prefill 4.4K/16K 1509/1650 -> 1499/1654 tok/s, decode C1/C4/C16
27.1/49.4/79.5 -> 27.0/50.3/80.6 tok/s (no thermal throttling).

## Allocation ledger

`serve-minimax` and `cuteafd plan --layout` admit one coordinator through the
same recipe (`plan::families::minimax::admission::Recipe`): the server sizes it
against measured free memory and the rings its Spark links register, the
planner against the card's budget with its estimated CUDA context and code.
Without the image's `PROGRAMS.json` the planner refuses, as Step-5's does.

- Logits are `logit_rows x 200,064 x 4` bytes: the 64-row decode tile, or seven
  rows per drafting ring if more; every prefill row only with
  `--full-prefill-logits`.
- The per-row buffers hold one prefill chunk: two lanes of 4096 rows by
  default, each lane with its own Spark exchange in flight.
- Program scratch is one arena at the largest program's need (the prefill
  scorer's 134.8 MB). Prefill tiles each chunk through 1024-row index-prefill
  and attention-extend launches over index keys staged for the whole context
  (132 B/token: 138 MB at 1M).
- Decode runs 64-row tiles; startup records one decode graph per verify row
  count (8 MiB each, estimated until the census replaces it).
- RoPE tables are 64 MiB each for cos and sin at 1M context; `--max-context`
  shrinks them along with the supported context.
- DSpark: 2.30 GB of weights, 40 MiB of BF16 K/V rings per drafting ring
  (default: concurrency, at most eight; a request admitted while every ring is
  busy decodes target-only) and taps holding a chunk's five target layers.
- DFlash2: the shared drafter's own admission (`families::minimax_m3::dflash::DflashMemory`,
  planner and server alike): 2.22 GB of E4M3 GEMMs and BF16 norms, kernels and selector
  codebooks (4.11 GB in BF16), 21 MiB of FP8 K/V rings per drafting ring (40 MiB in BF16),
  361 MB of taps and context-update buffers for a 4096-row tail, and its step buffers (51 MB of
  logits at eight sequences). FP8 packing holds one 377 MB BF16 source matrix (`fc`) at load. On the
  RTX 5090 at TP2 (planned cards' shapes) the pool is 138,112 tokens with FP8 GEMMs and FP8
  rings, 111,104 with BF16 GEMMs.
- Spark intake: per lane and rank, a GPU plane of the lane's rows where the NIC
  lands in GPU memory; on a GeForce coordinator one GPU row, with the partials in
  pinned host planes.
- Each Spark rank holds its expert slice, transport, the package's scratch (its
  exported value when the package carries the world's layout beside the
  manifest, else 512 MiB until worker admission measures it) and the runtime
  residual. Bottom routed layers onboard to the RTX (TP1) only when a rank
  would exceed its budget; a Spark-hosted tower that alone would make
  onboarding necessary is refused by name instead.
- `--embedding-placement host` maps the 2.46 GB BF16 input table from host
  memory; the untied 2.46 GB output head stays on the GPU.

## Host prefix tier

Retained prompt and turn snapshots share the device page pool with live
requests; the pool evicts its least recently used snapshot when an admission
needs pages or a bank (`--prefix-cache-entries`, 20) is full. The shared host
tier keeps evicted snapshots in pinned RAM (`HOST_CACHE_BYTES`, launcher
default `auto`: full-context KV for concurrency + 4, at most min(32 GiB, 25%
of RAM)), and a returning conversation restores them into fresh device pages
byte for byte. MiniMax runs it `on-evict` (`HOST_CACHE_STORE`): nothing is
copied until the device evicts a snapshot, and that eviction waits for its
copy; `on-retain` copies each snapshot when it is kept. A page is 180
segments (K, V and index planes of 60 layers, 4.44 MB per 64 tokens);
snapshots under 512 tokens stay on the device.

TP2 (RTX 5090 on PCIe Gen4 + 2 Sparks), speculator off, 131,072-token pool,
16 sequences, 4096 prefill rows, one launch per arm, 24-25 Oct:

| Host tier | Prefill 4.4K / 16K / 32K tok/s | Decode C1 / C4 / C16 tok/s | Ready after launch |
| --- | --- | --- | --- |
| off | 1492 / 1650 / 1621 | 26.6 / 50.2 / 78.4 | 59.5 s |
| 16 GiB, on-evict | — | — | 61.9 s |
| 32 GiB, on-evict | 1485 / 1652 / 1622 | 26.6 / 49.2 / 79.5 | 65.9 s |
| 32 GiB, on-retain | 1488 / 1649 / 1623 | 26.6 / 48.7 / 78.7 | — |

The pool pins its quota before serving, from transparent hugepages
(`cudaHostRegister` of madvised anonymous memory; `cudaHostAlloc` where THP
is off): 16 GiB in 1.7 s, 32 GiB in 7.1 s on this host, against 15 and 32-46 s
through `cudaHostAlloc`. Without page pressure both modes copied only the six
snapshots the full banks evicted (3.5 GB); on-evict's evictions waited 25 ms
in all. A churn of one 59,460-token and six 15,000-token conversations (more
than the pool holds), then each again, restored every return from RAM:
identical greedy text, and bit-identical logits on all 192 decode steps after
the restores.

| Returning prompt | Prefill TTFT | Restore TTFT, on-evict |
| --- | ---: | ---: |
| 59,460 tokens | 37.6 s | 0.26 s |
| 15,000 tokens | 9.1 s | 0.05-0.10 s |

Restores run at 25 GB/s (one contiguous copy of the same bytes: 27): the copy
engine submits each plan in 1440-entry pieces, so the driver's ~0.85 us per
entry overlaps the copies. On-retain returns faster under churn (0.31 against
0.41 s at 59K before those copy changes) because its evictions find the copy
done; on-evict copies nothing that stays on the device and keeps no device
duplicates in the host quota.
