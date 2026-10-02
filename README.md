<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/cuteafd-logo-color-dark.svg">
    <img src="assets/brand/cuteafd-logo-color.svg" alt="cuteafd" width="480">
  </picture>
</p>

An attention/FFN-disaggregated LLM engine for consumer Blackwell: attention on
one or two RTX cards (SM120), routed experts on a pool of DGX Sparks (SM121)
over RoCE. It loads standard Hugging Face checkpoints directly and serves an
OpenAI-compatible API with a live console.

Models: DeepSeek V4.1 Flash, DeepSeek V4 Flash / Pro, GLM 5.3, GLM 5.3 Flash,
MiMo V2 Flash, MiMo V2.6 Pro, Qwen 3.8 Flash Next.

## Results

Basic benchmark profile per family on its natural-minimum (1× RTX + fewest
Sparks) and maximum (2× RTX + 4 or 6 Sparks) hardware. Other reports:
[`benchmarks/`](benchmarks/README.md).

<!-- results:begin -->

#### Qwen 3.8

| Checkpoint | Hardware | C1 code | prose | JSON | 8K prefill | TTFT | Quality | Run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) | 1× RTX (min) | 115 | 112 | 122 | 6,826 | 1.17 s | ✓ KL 0.034 · top-1 88.5% ✓ lossless spec | [2026-10-02 · f09913484126](benchmarks/qwen4/2026-10-02-smoke-1rtx/report.svg) |

tok/s; C1 decode with thinking off, 8K prefill cold. Quality: logit fidelity against the family golden reference, prefix-cache restore exactness, lossless speculation.

<!-- results:end -->

## Working on it

[`AGENTS.md`](AGENTS.md) holds the standing rules, [`PLAN.md`](PLAN.md) the plan.
