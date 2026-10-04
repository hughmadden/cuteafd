# Benchmarks

Reports from `cuteafd bench` (profiles other than the basic one run when asked). Each directory holds `report.svg`, `report.json` and the share card; `cuteafd bench publish` rebuilds this index and the root README's table.

<!-- reports:begin -->

## DeepSeek V4

- 2026-10-02 · Release smoke · wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 · 2× RTX PRO 6000 @ 325 W + 6× DGX Spark · build v0.1.0 · [report](deepseek_v4/2026-10-02-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark/report.svg) · ⚠ quality gate failed
- 2026-10-02 · Release smoke · deepseek-ai/DeepSeek-V4-Flash-0731 · 2× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](deepseek_v4/2026-10-02-smoke-deepseek-v4-flash-0731-2rtx-4spark/report.svg)
- 2026-10-02 · Release smoke · wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 · 1× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](deepseek_v4/2026-10-02-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark/report.svg) · ⚠ quality gate failed
- 2026-10-02 · Release smoke · deepseek-ai/DeepSeek-V4-Flash-0731 · 1× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](deepseek_v4/2026-10-02-smoke-deepseek-v4-flash-0731-1rtx-4spark/report.svg)

## DeepSeek V4.1

- 2026-10-02 · Release smoke · deepseek-ai/DeepSeek-V4.1-Flash · 1× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/report.svg)
- 2026-10-02 · Release smoke · deepseek-ai/DeepSeek-V4.1-Flash · 2× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-2rtx-4spark/report.svg)

## GLM 5.3

- 2026-10-02 · Release smoke · nvidia/GLM-5.3-NVFP4 · 2× RTX PRO 6000 @ 325 W + 6× DGX Spark · build v0.1.0 · [report](glm5/2026-10-02-smoke-glm-5-3-nvfp4-2rtx-6spark/report.svg)
- 2026-10-02 · Release smoke · wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 · 2× RTX PRO 6000 @ 325 W + 6× DGX Spark · build v0.1.0 · [report](glm5/2026-10-02-smoke-glm-5-3-exl3-k4-v1-2rtx-6spark/report.svg) · ⚠ quality gate failed
- 2026-10-02 · Release smoke · nvidia/GLM-5.3-NVFP4 · 1× RTX PRO 6000 @ 325 W + 6× DGX Spark · build v0.1.0 · [report](glm5/2026-10-02-smoke-glm-5-3-nvfp4-1rtx-6spark/report.svg)
- 2026-10-02 · Release smoke · wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 · 1× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](glm5/2026-10-02-smoke-glm-5-3-exl3-k4-v1-1rtx-4spark/report.svg) · ⚠ quality gate failed

## GLM 5.3 Flash

- 2026-10-04 · FP8 defaults recheck · EXL3 K3.25 · 1× RTX + 2× Spark / 2× RTX + 4× Spark (head split) · build 45955d8 + benchmark hook 0d6809d · [comparison and recommendation](glm5_flash/2026-10-04-fp8-recheck/comparison.json)
- 2026-10-04 · FP8 recheck arm D (Release smoke quality) · wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 · 2× RTX PRO 6000 @ 325 W + 4× DGX Spark (head split) · build 45955d8 + benchmark hook 0d6809d · [report](glm5_flash/2026-10-04-fp8-recheck-d-2rtx-4spark/report.svg)
- 2026-10-04 · FP8 recheck arm F (Release smoke quality) · wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 · 2× RTX PRO 6000 @ 325 W + 4× DGX Spark (head split) · build 45955d8 + benchmark hook 0d6809d · [report](glm5_flash/2026-10-04-fp8-recheck-f-2rtx-4spark/report.svg)
- 2026-10-04 · FP8 recheck arm D (Release smoke quality) · wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 · 1× RTX PRO 6000 @ 325 W + 2× DGX Spark · build 45955d8 + benchmark hook 0d6809d · [report](glm5_flash/2026-10-04-fp8-recheck-d-1rtx-2spark/report.svg)
- 2026-10-04 · FP8 recheck arm F (Release smoke quality) · wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 · 1× RTX PRO 6000 @ 325 W + 2× DGX Spark · build 45955d8 + benchmark hook 0d6809d · [report](glm5_flash/2026-10-04-fp8-recheck-f-1rtx-2spark/report.svg)
- 2026-10-02 · Release smoke · brandonmusic/GLM-5.3-Flash-tr3-4bpw · 1× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](glm5_flash/2026-10-02-smoke-glm-5-3-flash-tr3-4bpw-1rtx-4spark/report.svg) · ⚠ quality gate failed
- 2026-10-02 · Release smoke · brandonmusic/GLM-5.3-Flash-tr3-4bpw · 1× RTX PRO 6000 @ 325 W + 2× DGX Spark · build v0.1.0 · [report](glm5_flash/2026-10-02-smoke-glm-5-3-flash-tr3-4bpw-1rtx-2spark/report.svg) · ⚠ quality gate failed
- 2026-10-02 · Release smoke · nvidia/GLM-5.3-Flash-NVFP4 · 1× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](glm5_flash/2026-10-02-smoke-glm-5-3-flash-nvfp4-1rtx-4spark/report.svg)
- 2026-10-02 · Release smoke · wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 · 1× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](glm5_flash/2026-10-02-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-4spark/report.svg) · ⚠ quality gate failed
- 2026-10-02 · Release smoke · wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1 · 1× RTX PRO 6000 @ 325 W + 2× DGX Spark · build v0.1.0 · [report](glm5_flash/2026-10-02-smoke-glm-5-3-flash-exl3-k3-25-v1-1rtx-2spark/report.svg)

## MiMo V2

- 2026-10-04 · Release smoke · XiaomiMiMo/MiMo-V2.6-Pro-MOPD · 2× RTX PRO 6000 @ 325 W + 6× DGX Spark · build mopd-319ccdb · [report](mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-2rtx-6spark/report.svg)
- 2026-10-04 · Release smoke · XiaomiMiMo/MiMo-V2.6-Pro-MOPD · 1× RTX PRO 6000 @ 325 W + 6× DGX Spark · build mopd-319ccdb · [report](mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-1rtx-6spark/report.svg)
- 2026-10-02 · Release smoke · XiaomiMiMo/MiMo-V2-Flash · 2× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](mimo_v2/2026-10-02-smoke-mimo-v2-flash-2rtx-4spark/report.svg)
- 2026-10-02 · Release smoke · XiaomiMiMo/MiMo-V2-Flash · 1× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](mimo_v2/2026-10-02-smoke-mimo-v2-flash-1rtx-4spark/report.svg)

## Qwen 3.8

- 2026-10-02 · Release smoke · nvidia/Qwen3.8-Flash-Next-NVFP4 · 1× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](qwen4/2026-10-02-smoke-qwen3-8-flash-next-nvfp4-1rtx-4spark/report.svg)
- 2026-10-02 · Release smoke · wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 · 1× RTX PRO 6000 @ 325 W + 4× DGX Spark · build v0.1.0 · [report](qwen4/2026-10-02-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-4spark/report.svg)
- 2026-10-02 · Release smoke · wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 · 1× RTX PRO 6000 @ 325 W · build v0.1.0 · [report](qwen4/2026-10-02-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx/report.svg)

<!-- reports:end -->
