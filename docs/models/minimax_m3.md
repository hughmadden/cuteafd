# MiniMax M3.1 Flash

A native multimodal MoE (428B total, 23B active) with MiniMax Sparse
Attention on every layer, quantization-aware-trained NVFP4 experts and
MXFP8 dense projections, a bundled vision tower and a bundled DSpark drafter.

Status: **planned**. `cuteafd plan` recognizes the release, validates every
tensor header and plans memory; the engine is being brought up.

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
  `cuteafd-loader/src/families/minimax_m3/reference.rs`.
- Cache: per layer, 64-row K and V pages for the rank's KV heads (E4M3 by
  default) and the shared index key in the paged indexer layout (E4M3 row +
  FP32 scale); 69,360 B/token on one GPU, 38,640 per GPU with two splitting
  the KV heads. Purely paged: exact prefix restores share pages.
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
  of 7, Markov rank 256, confidence head; taps 3/17/31/45/59).

## Planned cards (header lower bound)

`cuteafd plan --layout` on the release headers. Without a workspace manifest, workspaces and graphs remain estimates; drafter on the
RTX, vision on a Spark, FP8 KV, Spark budget 108 GiB per rank.

| card | Sparks | routed layers on RTX | KV pool (tokens) | Spark GiB/rank |
| --- | --- | --- | --- | --- |
| 1x RTX PRO 6000 | 4 | 0 | 1,120,640 | 54.1 |
| 1x RTX PRO 6000 | 3 | 0 | 1,120,640 | 72.1 |
| 1x RTX PRO 6000 | 2 | 3 | 944,256 | 102.5 |
| 1x RTX 5090 | 4 or 3 | 0 | 142,720 (178,176 with a host embedding) | 54.1 / 72.1 |
| 1x RTX 5090 | 2 | — | does not fit | — |

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

Served at TP2 (spark7, spark8; RTX 5090 coordinator, FP8 KV, 4096 prefill
rows, 16 sequences), the resident RTX tower takes 2.57 GB (its 2.50 GB ledger
and module): at 1M context the largest admitted pool drops from 189,568 to
152,512 tokens (-37,056), which is what the default serves, and the
reference's 65,536-token pool fits either way. A Spark-hosted tower does not fit at TP2: it needs 3.57 GB per rank
(weights, scratch, 1 GiB guard) beside 117.44 GB of experts, rings and
workspace, against a 120.3-122.4 GB budget (MemAvailable - 3 GiB), and
admission refuses it by name; the SM121 tower itself starts on the ranks.
TP3 (spark7, spark8, spark10) has room and is not yet measured.

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
on juliette). No thermal throttling.

Text throughput with the tower resident, same slot and day, vision off -> on:
prefill 4.4K/16K 1509/1650 -> 1499/1654 tok/s, decode C1/C4/C16
27.1/49.4/79.5 -> 27.0/50.3/80.6 tok/s (no thermal throttling).

## Single 32 GiB coordinator: KV-saving design

These are **CPU plans, not throughput or fidelity results**. The bundled DSpark
stays on. Use the release revision above, a 31.8 GiB coordinator budget, the
170-SM MiniMax program manifest and `--speculator dspark`. Native NVFP4 W4A4
experts and MXFP8 projections do not change. The input embedding and output
head are untied. Vision auto places the bundled tower on a Spark when it fits;
its allocation is included in the remote budget below.

A reproducible base command (paths supplied by the caller):

```sh
cuteafd plan "$SNAPSHOT" --layout --rtx 1 --spark-ranks 4 \
  --coordinator-gpu-budget-gib 31.8 --workspace-manifest "$PROGRAMS" \
  --speculator dspark --host-cache-bytes 0
```

`--host-cache-bytes 0` disables the separate host prefix snapshot tier for the
capacity comparison; it does not disable the proposed host K/V backing.
Pool tokens are aggregate capacity, not a larger model context: each sequence
still obeys the checkpoint's 1,048,576-token limit.

### Mapped embedding and workspace sizing

`--embedding-placement host` now removes the 2,458,386,432-byte BF16 input
table from native GPU admission and loads it through the shared mapped-host
embedding path. The 2.29 GiB untied output head stays on GPU, including for
DSpark. Host admission charges the input table. Runtime token lookup uses the
existing pinned mapping; device qualification must still check lookup parity,
prefill bandwidth and the seven-row drafting path.

The baseline allocation ledger is 14.890 GiB target weights, 5.103 GiB DSpark,
4.304 GiB target workspace, 0.648 GiB context/code, 2 GiB headroom and a
4.854 GiB FP8/index pool (75,136 tokens). The allocations explain the costs:

- Target logits alone are `4096 * 200064 * 4` = 3,277,848,576 bytes. Other
  row-sized activations and Spark intake scale with prefill rows. The executor
  currently reserves all-row logits even without `--full-prefill-logits`.
- Program scratch is **one shared maximum**, 53,478,400 bytes at the exported
  4096 capacity, not the sum of all programs. The 539,197,952-byte contiguous
  index-prefill scratch and attention-extend scratch are not allocated:
  current prefill uses causal paged decode tiles. Smaller runtime rows do
  not shrink exported scratch. A smaller `--workspace-manifest` must come
  from a real export; never scale its byte inventory arithmetically.
- Decode tile capacity remains the manifest's 64, independent of sequence
  concurrency. Page tables and `q2k` follow that extent. Shrinking decode
  exports requires matching runtime stride/shape changes and enough rows for
  DSpark verification; a C1 request is not necessarily one verification row.
  RoPE tables are 128 MiB at 1M context. `--context-tokens` can reduce them,
  but also reduces the supported context and is not a free saving.
- DSpark weights are 2,301,852,928 bytes. Its five BF16 K/V sliding-4096 rings
  cost **40 MiB per sequence slot**, 2.5 GiB at 64 slots. Tap buffers cost
  288 MiB at 4096 rows; the remaining buffers include the draft attention
  workspace (178.75 MiB at the baseline shape). None are target K/V storage.

`--minimax-profile rtx5090` is an **opt-in planning profile**, equivalent to
`--prefill-rows 256 --concurrency 1`; explicit nonzero values win. It retains
full context, unchanged native formats, the exported 64-row decode tile and
the full drafter. For the runtime use `--prefill-rows 256 --max-sequences 1`
with `--speculator dspark`; there is no automatic card-specific runtime
switch. Smaller prefill chunks may reduce prefill throughput. C1 decode has
no intended semantic loss, but latency must be measured before changing a
default. The profile deliberately gives up C>1 admission.

`--draft-context-slots N` may state the MiniMax ring slot count, but must equal
concurrency: rings are indexed by scheduler slot ID. Shrinking that count
independently would alias live sequence state. The 4096-token sliding window
is part of the drafter semantics and cannot be shortened as a memory tweak.
A future independent active-draft batch can reduce scratch, but not these
persistent per-sequence rings without exact spill/restore support.

The isolated TP4 knobs produce these CPU forecasts (same 4096-capacity
exports and unchanged DSpark weights):

| Shape | Target workspace GiB | DSpark GiB | Pool tokens |
| --- | ---: | ---: | ---: |
| 4096 prefill rows, 64 sequences | 4.304 | 5.103 | 75,136 |
| 256 prefill rows, 64 sequences | 0.444 | 4.839 | 139,008 |
| 4096 prefill rows, 1 sequence | 4.304 | 2.470 | 115,904 |
| 256 prefill rows, 1 sequence | 0.444 | 2.207 | 179,712 |


### NVFP4 K/V (accounting only)

`--kv-cache nvfp4` reports a proposed pool and refuses executable admission.
For each 128-element K or V head row: 64 bytes packed E2M1 plus 8 E4M3 scales
(one per 16 elements). Across 60 layers and four KV heads this is
**34,560 B/token K/V + 7,920 B/token FP8 index = 42,480 B/token**, versus
69,360 for FP8. The proposed per-head K/V FP32 globals add **1,920 fixed
bytes**, not bytes/token. Payload math is exact; allocations, padding and
execution remain proposed, so the ledger marks the result estimated.

The reference vLLM packed layout is `[pages, 2*kv_heads, 128, 72]`, with
alternating K/V head slots. Its current insert kernel actually uses **one K
and one V scalar global per layer** (480 bytes total), rather than per-head
globals; our 1,920-byte reservation allows the requested per-head contract.
Its sparse decode backend targets the SM100 family, not an existing SM120
implementation. See the [reference layout](https://github.com/vllm-project/vllm/blob/ccd4e4030f4b86f959a3cb3435437c0f592c88b1/vllm/models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py)
and [insert kernel](https://github.com/vllm-project/vllm/blob/ccd4e4030f4b86f959a3cb3435437c0f592c88b1/csrc/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu).

b12x MSA accepts BF16/FP8 today. Required work: native NVFP4 insertion and
calibrated scale policy, packed data/scale descriptors, sparse decode and
causal prefill on SM120, and prefix copies that retain packed bytes and all
scales exactly. Keep sorted, duplicate-free `q2k_indices`, the forced local
block, causal tails and stable pointer lifetimes. Never dequantize the full
cache to BF16 as its default representation. Validate NLL/logits and accepted
DSpark continuations against the FP8 path before execution admission.

### Selected-block paging (accounting only)

`--kv-placement host|spark` keeps index keys on GPU and plans full K/V backing
elsewhere. `--kv-hot-cache-gib` defaults to 1 GiB, including resident slot
metadata. Host backing uses `--kv-host-budget-gib` (default 192 GiB); this is
an explicit backing budget, not a claim that all physical RAM is free. On a
251 GiB host, embedding, prefix snapshots, OS reserve and other allocations
must also fit; the host ledger charges the backing. Spark backing is equally
sharded and bounded by the **least free rank** after experts, runtime,
transport, estimated scratch and vision. Remote capacity is not free memory
that can be spent twice.

The GPU proposal charges 7,920 B/token index plus **30 B/token** for a proposed
16-byte entry per layer/head/128-token block. That directory format is an
estimate. The hot-cache budget also covers 32 bytes per physical head-block
slot and must hold at least every layer's selected set plus insertion blocks.
This accounting reserves slots rather than asserting any cache hit rate.

Decode data flow: GPU index query -> max-pool/top-16 -> stable `q2k` -> GPU
page-directory lookup and missing-block deduplication -> reserve/pin hot slots
-> gather misses -> publish completed mappings -> sparse attention. Logical
selection order stays unchanged; physical cache placement is a separate map.
Keep the writable local block resident. Key the GPU LRU by sequence/page
generation, layer, KV head and logical block; never evict in-flight pages.
GPU writes maintain canonical backing before eviction or snapshot publication.
Prefill processes bounded tiles, gathering the union of selections across
the tile. That union, and DSpark's multi-row verification union at C1, may
exceed 16 blocks; split tiles rather than overrun the hot-cache admission.

At FP8 a single decode row selects at most
`16 * 128 * 4 * 128 * 2` = **2 MiB/layer**, or **120 MiB over 60 layers**.
Per-head blocks are 32 KiB. NVFP4 reduces those figures to 1.125 MiB/layer
and 67.5 MiB per target row, with globals already resident. Writes add
61,440 B/committed token at FP8 (34,560 at NVFP4), plus synchronization.

| Path | Bandwidth assumption | Cold FP8 transfer floor per target row |
| --- | --- | --- |
| PCIe Gen4 x16 | 24 GB/s payload | 5.24 ms |
| PCIe Gen5 x16 | 48 GB/s payload | 2.62 ms |
| RoCE 100 Gb/s | 12.5 GB/s wire ceiling | 10.07 ms |
| RoCE 200 Gb/s | 25 GB/s wire ceiling | 5.03 ms |
| Ethernet 10 Gb/s | 1.25 GB/s wire ceiling | 100.66 ms |

PCIe payload figures are assumptions below the approximately 31.5/63 GB/s
encoded x16 ceilings ([PCI-SIG](https://pcisig.com/what-bit-rates-does-pcie-50-specification-support-and-how-does-it-compare-prior-pcie-generations));
no host link width, negotiated speed or bandwidth was probed. With 50%/90%
hits multiply the transfer term by 0.5/0.1; local-block inclusion alone does
not establish those hit rates. Add lookup, roundtrip, completion, writes,
index scanning and expert-traffic contention. Next-layer selection depends
on the preceding attention/FFN, so C1 cannot hide all transfers behind the
layer pipeline. Overlap resident-block attention with miss gathers and
independent requests where available. C1 cost is an incremental transfer
bound per verified target row, not a measured emitted-token penalty;
drafting acceptance and row-union reuse determine its amortization.

Mapped pinned host storage permits GPU-initiated coalesced gather loads and
writes without CPU byte copying or polling; see the [CUDA mapped-memory
contract](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html#mapped-memory).
It still consumes PCIe bandwidth. Native device-initiated gather is the host
proposal; CPU-driven per-layer memcpy staging would violate the hot-path goal.
**Do not assume Spark-to-5090 GPUDirect:** NVIDIA excludes GeForce from
[supported GPUDirect RDMA](https://docs.nvidia.com/holoscan/archive/2.9.0/set_up_gpudirect_rdma.html),
and [DGX Spark](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)
does not support GPUDirect RDMA to `cudaMalloc` memory; its documented verbs
alternative registers `cudaHostAlloc` memory. NIC-to-host staging followed
by GPU gather introduces a host hop. Thus `spark` remains a capacity-only,
non-executable design until a supported device-initiated transport satisfying
the no-staging constraint is demonstrated. Nominal RoCE bandwidth alone
cannot establish feasibility.

Prefix restores must retain canonical K/V and index page bytes, scales,
logical identities, valid lengths and generation metadata. Share immutable
pages and copy-on-write tails; publish prompt/turn snapshots only after all
writes complete. Restore mappings before attention, invalidate uncommitted
speculative generations and preserve the existing drafter cold/warm contract.
LRU eviction changes residency only, never model state. Gate implementation
on byte-exact restore versus scratch prefill, causal block-boundary tests,
miss/eviction/rollback races and DSpark continuation parity.


### Capacity summary (DSpark on in every cell)

**Default TP2 at 108 GiB/Spark remains refused for every variant:** onboarding
requires the missing native TP1 expert manifest. No guessed local scratch is
presented as exact admission. The TP2* column below is a separate conditional
forecast using `--spark-budget-gib 116`, keeping all routed layers remote;
baseline remote usage reaches 114.43 GiB on the vision rank. That budget is
not a measured free-memory guarantee and must pass worker admission. TP3/TP4
retain the default 108 GiB budget. No hardware configuration was changed.

L = exact checkpoint/native allocation formulas, with calibrated context and
estimated code/remote scratch; consequently **pool capacity is an offline
forecast**, not an exact live-memory promise. P = additionally proposed cache
allocation/metadata; execution explicitly unimplemented. All rows use full
1M context, automatic pool, 1 GiB hot cache where paged, and 192 GiB host
backing. E is host embedding, S the 256-row/one-sequence profile, N NVFP4,
H host paging and R Spark paging; the final column is accounting status.

| Option | TP2* (116 GiB/Spark) | TP3 | TP4 | Accounting |
| --- | ---: | ---: | ---: | --- |
| Baseline | 76,608 | 75,840 | 75,136 | L |
| Host embedding (E) | 112,064 | 111,296 | 110,592 | L |
| 5090 C1 sizing (S) | 179,840 | 179,776 | 179,712 | L |
| NVFP4 K/V (N) | 125,056 | 123,904 | 122,624 | P |
| Host paging (H) | 533,376 | 526,976 | 520,704 | P |
| Spark paging (R) | 54,784 | 526,976 | 520,704 | P |
| E + S | 215,296 | 215,232 | 215,168 | L |
| E + S + N | 351,488 | 351,360 | 351,360 | P |
| E + S + H | 1,743,232 | 1,742,848 | 1,742,464 | P |
| E + S + R | 81,280 | 1,593,728 | 1,742,464 | P |
| E + S + N + H | 1,743,232 | 1,742,848 | 1,742,464 | P |
| E + S + N + R | 144,640 | 1,742,848 | 1,742,464 | P |

These figures use the supplied release headers and 170-SM program manifest;
the table evaluates the ledger equations, with ten cells cross-checked
against `cuteafd plan` JSON. Native pools round to 64-token pages, proposals
to 128-token blocks. TP differences in coordinator capacity include Spark
intake buffers. Spark paging is also capped by remote memory after vision.
At TP3/TP4 the full host-paging combination is index-limited: NVFP4 reduces
backing bytes/traffic rather than increasing that pool. A one-sequence profile
cannot consume more than 1M tokens in one request; surplus is aggregate page
capacity for retained prefixes or a subsequent explicit concurrency increase.

Proceed first with mapped embedding plus the explicit 5090 sizing profile:
**215,168 tokens at TP4**, keeping native formats and DSpark. Next obtain the
real TP1/remote package manifests to resolve default-budget TP2, then qualify
mapped lookup, chunked prefill, draft continuation and exact prefix restore
on an authorized device. Measure warmed C1 emitted throughput and acceptance
before changing defaults. NVFP4 needs the native SM120 kernels and numerical
gates; host paging needs a device-gather prototype and miss traces; Spark
paging remains blocked on the transport feasibility requirement above.
