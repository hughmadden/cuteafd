# Qwen 3.8 Flash Next

`qwen4_exp`: Gated DeltaNet (GDN) linear attention with a full-attention
layer every fourth, a PLE n-gram memory table, and fused expert tensors.

## Supported checkpoints / quants

- `Qwen/Qwen3.8-Flash-Next` official release — FP8 128x128-block routed
  experts (`qwen4:fp8`).
- EXL3 K4.25 PLE publications of the same checkpoint (`qwen4:exl3-k45`).
- NVIDIA ModelOpt NVFP4 — routed experts run W4A16 (`qwen4:nvfp4`).

## Engineering summary

- Attention: Gated DeltaNet linear recurrence on most layers, full GQA with
  an indexer every fourth layer.
- Shared expert with a sigmoid gate; hyper-connections (low-rank) mix the
  residual stream alongside the router.
- PLE n-gram memory table: a mapped table gathering 16 rows of 160 per token
  from pinned host RAM (or GPU), with bounded prefetch.
- Routed experts: 512 experts, top-10, hidden 2560 / intermediate 640, SiLU
  unclamped, stored as one fused `[experts, ...]` tensor per projection per
  layer; EXL3 K4/K5, FP8 128x128 blocks, or ModelOpt NVFP4 group-16; a local
  (RTX-resident, TP1) expert path is supported.
- Speculator: a native MTP layer (full attention, 512 experts, and a
  hyper-connection feedback path) exists in the checkpoint but is not yet
  wired into the plain serve path.
- RTX/Spark layouts: fits comfortably on one RTX with local experts (~73 GB
  for the EXL3 or FP8 package); Spark EXL3 runs at TP3 today since the
  640-wide intermediate does not split evenly across TP4.
- Prefix cache: merged — 256-row units over the full-attention layers, a
  combined GDN-state + PLE mark, with n-gram history recomputed from token
  ids rather than cached.

## Default precision (single residency)

Every weight has one resident format. Precision is chosen by measurement:
FP8 converts at load into the only copy where it is faster and the golden
stays within ~0.005 nat KL/NLL; drafters run FP8 whenever emitted tok/s is
higher (they cannot change the output). Measured 2026-10-03, natural minimum,
one warm launch per arm, `CONCURRENCY=4`, code tok/s (C4 aggregate), golden
512 tokens.

| Arm | C1 | C4 | 8K prefill | KL · top-1 · NLL |
| --- | ---: | ---: | ---: | --- |
| checkpoint (BF16 projections, head) | 196 | 439 | 6,146 | 0.034 · 88.5% · 3.297 |
| **FP8 head (default)** | 222 | 441 | 6,100 | 0.036 · 88.5% · 3.300 |
| FP8 head + FP8 GDN/attention projections | 247 | 489 | 6,177 | 0.046 · 86.7% · 3.360 |

Qwen 3.8 Flash Next EXL3 K4.25, 1 RTX, MTP 3. FP8 projections fail the KL gate (+0.012) and stay opt-in (`QWEN_FP8_DECODE=on`); `QWEN_FP8_HEAD=off` keeps BF16.

## Known limits

- FP8 experts have no Spark TP layout yet: 640 is not evenly divisible the
  way the FP8 MoE kernel currently tiles larger TP degrees, so the Spark
  path today is EXL3-only.
- The native MTP drafter's weights are present in the checkpoint but not
  yet served; CuteAFD verifies copy-window drafts only.

## Changelog

| Version | Date | Change | Basic eval |
| --- | --- | --- | --- |
| v0 | 2026-10-02 | First release | <a href="../../benchmarks/qwen4/2026-10-02-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx/report.svg"><img src="../../benchmarks/qwen4/2026-10-02-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx/card.svg" width="360" alt="Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) (min)"></a> <a href="../../benchmarks/qwen4/2026-10-02-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-4spark/report.svg"><img src="../../benchmarks/qwen4/2026-10-02-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-4spark/card.svg" width="360" alt="Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) (max)"></a> <a href="../../benchmarks/qwen4/2026-10-02-smoke-qwen3-8-flash-next-nvfp4-1rtx-4spark/report.svg"><img src="../../benchmarks/qwen4/2026-10-02-smoke-qwen3-8-flash-next-nvfp4-1rtx-4spark/card.svg" width="360" alt="Qwen3.8-Flash-Next-NVFP4 (nvfp4-g16) (min)"></a> |

## Additional benchmarks

| Engine version | Date | Profile | Hardware | Report |
| --- | --- | --- | --- | --- |

_No additional benchmark reports yet._
