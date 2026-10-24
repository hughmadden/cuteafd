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
- Vision: Conv3d (2x14x14) patch embed, 32 CLIP layers (1280, 16 heads of
  80) with 3D axial RoPE, a GELU projector to 6144 and a 2x2 patch-merge GELU
  MLP; 576 tokens per 672x672 tile, `dynamic_res` tiling; images and video.
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

`--host-cache-bytes 0` leaves the host prefix snapshot tier out of the
capacity comparison.
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

### Host prefix tier

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
16 sequences, 4096 prefill rows, one launch per arm, 24 Oct:

| Host tier | Prefill 4.4K / 16K / 32K tok/s | Decode C1 / C4 / C16 tok/s | Ready after launch |
| --- | --- | --- | --- |
| off | 1492 / 1650 / 1621 | 26.6 / 50.2 / 78.4 | 62 s |
| 32 GiB, on-evict | 1485 / 1652 / 1622 | 26.6 / 49.2 / 79.5 | 113 s |
| 32 GiB, on-retain | 1488 / 1649 / 1623 | 26.6 / 48.7 / 78.7 | 93 s |
| 16 GiB, on-evict | — | — | 82 s |

Without page pressure both modes copied only the six snapshots the full banks
evicted (3.5 GB); on-evict's evictions waited 25 ms in all. A churn of one
59,460-token and six 15,000-token conversations (more than the pool holds),
then each again, restored every return from RAM: identical greedy text, and
bit-identical logits on all 192 decode steps after the restores.

| Returning prompt | Prefill TTFT | Restore TTFT, on-evict | on-retain |
| --- | ---: | ---: | ---: |
| 59,460 tokens | 37.5 s | 0.41 s | 0.31 s |
| 15,000 tokens | 9.1 s | 0.06-0.13 s | 0.06-0.08 s |

On-retain returns faster under churn because its evictions find the copy
done; on-evict copies nothing that stays on the device and keeps no device
duplicates in the host quota. Restores average 16 GB/s; the copy engine
alone moves a 929-page restore's 167,220 segments at 25.8 GB/s (one
contiguous copy: 27.2). Readiness pays for pinning the quota up front, at
about 1 GB/s on this host.

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
1M context and an automatic pool. E is host embedding, S the
256-row/one-sequence profile and N NVFP4; the final column is accounting status.

| Option | TP2* (116 GiB/Spark) | TP3 | TP4 | Accounting |
| --- | ---: | ---: | ---: | --- |
| Baseline | 76,608 | 75,840 | 75,136 | L |
| Host embedding (E) | 112,064 | 111,296 | 110,592 | L |
| 5090 C1 sizing (S) | 179,840 | 179,776 | 179,712 | L |
| NVFP4 K/V (N) | 125,056 | 123,904 | 122,624 | P |
| E + S | 215,296 | 215,232 | 215,168 | L |
| E + S + N | 351,488 | 351,360 | 351,360 | P |

These figures use the supplied release headers and 170-SM program manifest;
the table evaluates the ledger equations, with ten cells cross-checked
against `cuteafd plan` JSON. Native pools round to 64-token pages, proposals
to 128-token blocks. TP differences in coordinator capacity include Spark
intake buffers. A one-sequence profile cannot consume more than 1M tokens in
one request; surplus is aggregate page capacity for retained prefixes or a
subsequent explicit concurrency increase.

Proceed first with mapped embedding plus the explicit 5090 sizing profile:
**215,168 tokens at TP4**, keeping native formats and DSpark. Next obtain the
real TP1/remote package manifests to resolve default-budget TP2, then qualify
mapped lookup, chunked prefill, draft continuation and exact prefix restore
on an authorized device. Measure warmed C1 emitted throughput and acceptance
before changing defaults. NVFP4 needs the native SM120 kernels and numerical
gates.
