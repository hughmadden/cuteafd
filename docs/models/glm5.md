# GLM 5.3

Ported from `../glmrt`: MLA with a DeepSeek Sparse Attention (DSA) indexer,
a top-8 sigmoid router, and a DFlash2 draft speculator.

## Supported checkpoints / quants

- `zai-org/GLM-5.3` official FP8 — supported, but only fits a small KV pool
  at full size; EXL3 and NVFP4 quants are the recommended way to run the
  full model (see `PLAN.md` quant scope).
- `wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1` and similar EXL3 K4/K5 publications —
  the benchmarked path on both reference configs.
- NVIDIA ModelOpt NVFP4 (`nvidia/GLM-5.3-NVFP4`) — routed experts in NVFP4
  group-16; dense parts run through the FP8-block/W8A8 paths described
  below.

## Engineering summary

- Attention: MLA with a DSA indexer selecting top-k tokens; pool-gated
  indexer variant shares a previous full layer's selection on shared-indexer
  layers. Dense first layers, then MoE from `first_moe_layer` on.
- Router: sigmoid noaux_tc with `e_score_correction_bias`, routed scale 2.5,
  top-8 of the checkpoint's expert count.
- Routed experts run on Spark TP x EP via `expertd-native`: FP8 128x128
  blocks, EXL3 K4/K5, or ModelOpt NVFP4 (group 16); GLM 5.3 has no local
  (RTX-resident) expert path.
- Speculator: the native MTP layer is not run — DFlash2 external drafters
  are the measured-best speculator for this family, with an adaptive policy
  priced by distinct Spark expert reads.
- KV format: FP8 MLA latent record (E4M3 + per-128 FP32 group scales, plus
  BF16 RoPE where the layer carries rotary dims); mHC hyper-connections with
  Sinkhorn iterations and an `hc_head` final collapse.
- RTX/Spark layouts: natural minimum is 1 RTX + 4 Sparks with a DFlash2
  drafter; maximum is 2 RTX + 6 Sparks with head split as the default
  (replicated latent, split `q_b`/`kv_b`/`o_proj`/shared-expert across both
  GPUs with one hidden all-reduce per layer).
- Prefix cache: merged — page-only state (no host-tier compaction yet; a
  stand-in tail covers that path).

## Known limits

- The official FP8 checkpoint's KV pool is small at full context; run EXL3
  or NVFP4 for serious context lengths.
- No local (RTX-only) expert path — GLM 5.3 always needs at least one Spark
  rank.

## Changelog

| Version | Date | Change | Basic eval |
| --- | --- | --- | --- |
| v0 | 2026-10-02 | First release | — |

## Additional benchmarks

| Engine version | Date | Profile | Hardware | Report |
| --- | --- | --- | --- | --- |

_No additional benchmark reports yet._
