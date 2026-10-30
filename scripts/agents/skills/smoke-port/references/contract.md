# Smoke contract: gates, bars, probes, deferrals

## Gates

| Gate | Command | Smoke | Later (qualification) |
| --- | --- | --- | --- |
| Compile | `cargo check --workspace --tests --locked` (from `rust/`, `CARGO_TARGET_DIR` per AGENTS.md) | **required** | |
| Targeted unit tests | `cargo test -p cuteafd-loader <fam>`, `cargo test -p cuteafd-daemon <fam>` | **required** | |
| CPU gate, default tier | `scripts/dev/cpu-gate --task NAME --base SHA`: no new failing ids against the base | **required** | `--full` before a dev merge |
| Executable deployment plan | `cuteafd plan SNAPSHOT --deployment` for the cell (the launcher refuses a plan without prefill units, even for WIP) | **required** | |
| Source gates | the family file in the lane, readiness and exchange source tests (shared-engine.md §1-2, §6) | **required** | |
| Export comparison | `scripts/build/compare-sm-exports.py --sms 188,170` (CPU, in `scripts/build/cuteafd-dev.sh cpu`) | **required when exports change** | required |
| GPU oracle per new or recompiled program | SparkInfer tests in the dev container, against dense FP32, on SM120 and SM121 | **required before the first serve** | |
| Slot seal | `cuteafd programs-check` | **required** | |
| Layer replay | the family golden's per-layer dump on real weights against the reference golden (P4 builds the mode; flags differ per family): one layer of each kind | **required** | |
| Teacher-forced windows | 2-8 windows through the family golden or the probe API; `scripts/bench/compare-logits.py` | **required, smoke bars** | full set, release bars |
| Probe set | below | **required** | |
| `cache_exact` | in the smoke card's baseline | required if the prefix cache is on; `PREFIX_CACHE_ENTRIES=0` shows Skipped | required |
| Smoke card | `cuteafd bench --url URL --profile smoke` (≤ 5 min per config) | **the deliverable** | |
| Golden N=1 vs N | family golden: prefill units N=1 vs N byte-identical (logits and KV), shared-engine.md §6.2 | recommended, cheap once the golden exists | required |
| Lossless speculation | `spec_lossless_set` panel | deferred (speculator off: Skipped) | required with a speculator |
| wip-cards pair against dev | `scripts/bench/wip-cards.py` | only if a shared hot path changed | required |
| Row curves | 2048/2752/4096 band (AGENTS.md "Kernel row curves") | deferred | required for new kernels |
| Published fidelity set | `fidelity-set.py`, publication pins | deferred (quick fidelity shows Unsupported) | per release |

## Bars

Release bars come from the families' own docs: MiniMax top-1 ≥ 0.94, KL ≤ 0.03 nat and NLL ≤
reference + 0.01 on generated rows; GLM 5.3 dense-layer cosine ≥ 0.9999 at bring-up; DeepSeek V4
per-layer cosine about 0.999 and top-1 about 94 %. Smoke bars are deliberately looser, because two
to eight windows are noisy and calibrated W4A4 moves logits. They are proposed, not yet ratified:

| Measure | Smoke bar |
| --- | --- |
| Layer replay, dense or attention layers | cosine ≥ 0.999 per layer |
| Layer replay, MoE layers (real Spark experts) | cosine ≥ 0.995 per layer |
| Teacher-forced windows, generated rows | top-1 ≥ 0.90, KL ≤ 0.05 nat, NLL ≤ reference + 0.05 |

Read logit comparisons on generated rows only: sparse-attention top-k near-ties put prompt rows
nats apart without any defect (MiniMax).

## Probe set

Greedy, through the OpenAI API of the WIP instance:

1. "What is the capital of France?" → mentions Paris.
2. "What is 17 * 23?" → 391.
3. "Write a Python function is_prime." → syntactically valid code.
4. A needle at about 4K tokens; for a sparse-attention model also one **past the indexer's top-k**
   (dense and sparse attention agree below it, so a short-context smoke never exercises the indexer).
5. One tool call through the family's parser.
6. Thinking on, and off where the template supports it: the reasoning is split out.
7. C1, C4 and C16 completions without errors; one prompt at the planned context limit.

## Deferral list (paste into `docs/models/<family>.md`)

```markdown
**Smoke release (unqualified).** Served text-only, speculator off, one decode lane, on <cell>.
Deferred: published fidelity set; `cpu-gate --full`; lossless speculation; the <drafter> drafter;
<vision/audio> tower; exact prefix cache <if off>; two-RTX placements; RTX 5090; RTX expert
onboarding; CUDA graphs and packing; kernel row curves; performance tuning; card A/Bs.
```
