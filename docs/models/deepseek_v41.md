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
- KV format: compressed MLA latent in FP8 32x32 blocks; a specialized
  exact prefix cache — radix banks keyed by token ids, shared
  FP4 pages, a copied SWA "front", pinned-host tier in `cuteafd-hostcache`.
- RTX/Spark layouts: natural minimum is 1 RTX + 4 Sparks; maximum is 2 RTX +
  4 Sparks. V4.1 keeps the coordinator layer-range split by default — the
  measured head-split hop cost on this fabric does not clear the bar its
  attention weights would need to win (see `PLAN.md` Phase 6).
- Vocabulary head: BF16 stays the default after the single-copy FP8 gate
  missed the dual-RTX C1 promotion bar. `CUTEAFD_V41_FP8_HEAD=all` uses one
  shared E4M3 copy for target and dSpark (FP32 scales per row and 128-wide K
  block), partitioned by vocabulary rows on dual RTX, and releases BF16
  after packing. `off` keeps BF16; `draft` retains both formats. Target-head
  quality was accepted (golden KL +0.000534 nat, practically unchanged NLL,
  top-1 472 → 465 of 512); the opt-in modes do not qualify a default change.
- Optional vision tower (MoonViT-style) when the checkpoint carries
  `vision_config`.

## Known limits

- The opt-in FP8 vocabulary head regressed the 32-token "hello" reply to
  0.93× / 0.92× on 1 / 2 RTX in its earlier comparison. BF16 remains the
  default; FP8 needs a new performance gate before promotion.
- Prompt and turn-end prefix restores are byte-exact against their own
  snapshots. Cold prefill can differ because Spark FP32 atomic expert
  reductions are arrival-ordered at 256+ rows; batch-invariant prefill and
  verify remain deferred.
- Large index-selection and attention-query graph entries can be evicted
  between encoder and replay shapes, so warmed requests can recapture graphs.
- Device-driven exchange remains opt-in (`CUTEAFD_V41_DEVICE=1`); write mode
  (`CUTEAFD_SPARK_WRITE=1`) can stall on written flags and is unqualified.
- Native NVFP4 W4A4 prefill and W4A4 decode/verify remain follow-ups; the
  supported NVFP4 release uses the existing W4A8 expert path.
- One-RTX startup still waits on slow Spark layer reads after the
  coordinator-first placement handoff; further load-speed work is open.

## Changelog

| Version | Date | Change | Basic eval |
| --- | --- | --- | --- |
| v1 | 2026-10-04 | Coordinator-first loading and smaller one-RTX workspaces; opt-in device exchange and shared single-copy FP8 head; exact turn-end restore check. BF16 head remains default. | Pending v1.0.0-rc1 Release smoke; cards populated by `cuteafd bench publish`. |
| v0 | 2026-10-02 | First release | <a href="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (min)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-2rtx-4spark/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-2rtx-4spark/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (max)"></a> |

## Additional benchmarks

| Engine version | Date | Profile | Hardware | Report |
| --- | --- | --- | --- | --- |

_No additional benchmark reports yet._
