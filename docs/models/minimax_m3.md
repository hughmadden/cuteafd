# MiniMax M3.1 Flash

A native multimodal MoE (428B total, 23B active) with MiniMax Sparse
Attention on every layer, quantization-aware-trained NVFP4 experts and
MXFP8 dense projections, a bundled vision tower and a bundled DSpark drafter.

Status: **planned**. `cuteafd plan` recognizes the release, validates every
tensor header and plans memory; the engine is being brought up.

## Supported checkpoints / quants

- `MiniMaxAI/MiniMax-M3.1-Flash` (revision `9e7ebacac350497ad4a1272a2e817639bf11561c`),
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

`cuteafd plan --layout` on the release headers. Workspaces and graphs are an
estimate until the engine's allocation ledger replaces them; drafter on the
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
