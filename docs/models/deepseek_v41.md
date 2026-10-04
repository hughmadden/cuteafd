# DeepSeek V4.1 Flash

The regression anchor for CuteAFD: every phase and every shared-hot-path
change is checked against its parity numbers before merging.

## Supported checkpoints / quants

- `deepseek-ai/DeepSeek-V4.1-Flash` — official release: FP8 32x32-block
  coordinator weights (E4M3 + UE8M0 scales), native MXFP4 routed experts
  (E2M1 + UE8M0 per 32).
- EXL3 K2–K4 routed-expert quants of the same checkpoint.
- NVIDIA ModelOpt NVFP4 release: lossless downcast onto the existing W4A8
  expert path (the official MXFP4 weights already use power-of-two scales).

## Engineering summary

- Attention: compressed MLA with a per-layer compression ratio schedule and
  CED (compressed encoder/decoder) KV sharing across index source layers
  `[2, 8, 14, 20]`.
- Engram memory-mapped embedding tables gate layers 1 and 14, read through
  the shared `MappedTable` path (page cache, bounded prefetch, pinned
  upload ring).
- Routed experts run on 2, 3, 4 or 6 Spark ranks over `expertd-native`
  (MXFP4, EXL3 K2–K4, or ModelOpt NVFP4); mHC hyper-connections mix the
  coordinator residual stream.
- Speculator: three-stage dSpark drafter (`markov_head`, `confidence_head`,
  `main_proj`) with adaptive width from a calibrated confidence and cost
  model.
- KV format: compressed MLA latent in FP8 32x32 blocks; the only family with
  a full exact prefix cache today — radix banks keyed by token ids, shared
  FP4 pages, a copied SWA "front", pinned-host tier in `cuteafd-hostcache`.
- RTX/Spark layouts: natural minimum is 1 RTX + 4 Sparks; maximum is 2 RTX +
  4 Sparks. V4.1 keeps the coordinator layer-range split by default — the
  measured head-split hop cost on this fabric does not clear the bar its
  attention weights would need to win (see `PLAN.md` Phase 6).
- Vocabulary head: shared BF16 by default. `CUTEAFD_V41_FP8_HEAD=all`
  retains one E4M3 weight plus FP32 scales per row and 128-wide K block,
  shared by target and dSpark; dual RTX partitions this weight by vocabulary
  rows. BF16 is released after packing drains. `off` keeps BF16; `draft`
  retains both formats and remains experimental. Single-copy `all` matches
  the accepted quality result but missed the required C1 speedup on dual RTX
  and short-response parity on both layouts. It remains opt-in and BF16
  stays the default.
- Optional vision tower (MoonViT-style) when the checkpoint carries
  `vision_config`.

## Changelog

| Version | Date | Change | Basic eval |
| --- | --- | --- | --- |
| v0 | 2026-10-02 | First release | <a href="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (min)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-2rtx-4spark/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-2rtx-4spark/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (max)"></a> |

## Additional benchmarks

| Engine version | Date | Profile | Hardware | Report |
| --- | --- | --- | --- | --- |

_No additional benchmark reports yet._
