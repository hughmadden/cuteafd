# Mistral Large 4 NVFP4 smoke port: plan

**Inputs.** MODEL `mistralai/Mistral-Large-4-1T-A52B-NVFP4@bb6d00fe249b0e876f0cffc6b85014be02de3e68` ·
CELL 1 × RTX PRO 6000 + 6 DGX Spark · REFERENCE `vllm/vllm-openai:mistral-large-4` (`SPEC.md`) ·
SCOPE text-only, speculator off, one lane.

**Resume here (30 Oct 2026).** Phase 0 is done: `SPEC.md` (reference semantics, every open question
settled from the image), `census.md`, and a `cuteafd plan` run on a header-only stub through
`scripts/checkpoint_view.py mistral`. That run found two shared gaps, both fixed with tests on this
branch (gaps 2 and 3). The planner now classifies every module correctly and stops at "family not
recognized", which is the expected start of phase 1. Next: brief packets P1-P6; land gap 1's native
seam before P2's name rules.

## Fit (from `census.md`; KV is an estimate until the descriptor plans it)

| Component | GiB | Lives on |
| --- | ---: | --- |
| routed experts (71 × 128, NVFP4 g16 + scales) | 539.16 | Sparks, TP6: 89.9 GiB/rank against the 100 GiB default (TP5 112.3, TP4 134.8: refused) |
| backbone (attention, indexers, shared experts, layer-0 FFN, router, norms) | 17.21 | RTX |
| embedding + LM head (BF16) | 4.69 | RTX |
| DSpark drafter (BF16) | 3.91 | deferred |
| Pixtral tower | 3.17 | deferred |
| KV per token | ≈ 44.9 KiB | RTX: 54 RoPE owners × 656 B + 9 NoPE owners × 528 B + 44 indexer keys × 132 B = 45,984 B; 1,048,576 tokens ≈ 44.9 GiB |

Decision: one PRO 6000 (about 22 GiB of weights leaves room for a pool near 1M tokens) and six
Sparks at TP6, the only Spark world that fits. The RTX 5090 does not fit this checkpoint and is out
of scope.

## Templates chosen

| Subsystem | Source | Tier |
| --- | --- | --- |
| skeleton: config → descriptor → `Ops` chain with CPU oracle → shared `Backend` → `PrefillHooks` | Kimi K3 | T1 |
| MLA + DSA attention, index top-k, sparse MLA, hand E4M3 prefill (656/528 B records) | GLM 5.3 programs at a Mistral geometry | T1 |
| query producer without q-LoRA | GLM `glm_stage_*` composed around FP8 projections (the K6 decomposition) | T2 |
| routed experts NVFP4 W4A4 | Step-5's `step5:nvfp4` path (data rows) | T0 |
| RMSNorm; LayerNorm with bias | `glm_norm`; Step-5 `affine_norm` | T1 |
| router | shared `softmax_select_kernel` (softmax, top-4, renormalized) | T0 |
| tokenizer | the shipped `tokenizer.json` (bos 1, eos 2 checked against `tokenizer_config.json`) | T0 |
| chat template | the shipped `chat_template.jinja` through the API's minijinja path (as Step-5, GLM and Qwen4 do) | T5 (small) |

## Gaps

| # | Gap | Tier | Packet | Shared? | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | Mistral-native layout (`params.json`, `consolidated*`): `CheckpointFormat::MistralNative` with an in-memory HF-style config and alias names, after the EXLR8 inventory; route the readers that bypass `Checkpoint` | T5 | P1 | yes | planning bridge done (`checkpoint_view.py`); native seam open |
| 2 | ModelOpt `FP8_PB_WO` / `FP8_BLOCK_SCALES` declared numerics were refused | T0 | — | yes | **fixed**: `W8CheckpointActivations`, test `mixed_precision_block_fp8_declarations_are_weight_only` |
| 3 | FP8 block entries inherited the checkpoint-wide NVFP4 `group_size` 16 as their block | T0 | — | yes | **fixed**: own `group_size` or 128, test `block_fp8_does_not_inherit_the_nvfp4_group` |
| 4 | A snapshot without `config.json` silently launches on the DeepSeek V4.1 path (`release_config_family`) | T0 | P5 | yes | open: fail loudly; teach `checkpoint-family.py` `params.json` |
| 5 | Config reader and descriptor with typed refusals (q-LoRA present, `index_nope`, grouped routing, unknown `nope`/sharing patterns, `apply_scale: true`) | T0/T4 | P1 | no | open |
| 6 | `ExpertGeometry` 6144 × 128 × top-4 × 6144, SiLU, **no clamp**; `GEOMETRIES` row; TP6 package; Spark worlds; census rows | T0 | P2 | SparkInfer row | open; first check every expert's w1/w3 `input_scale` are bit-identical on the local snapshot |
| 7 | Expert names `layers.L.experts.E.w{1,2,3}` with `weight`, `weight_scale`, `weight_scale_2`, `input_scale` | T0 | P2 | via gap 1's aliases | open |
| 8 | Query producer without q-LoRA (`wq` FP8 6144 → 64 × 128; `wkv_a` BF16 576 or 512) | T2 | P3 | `L/formats/mla.rs` accepts `q_rank == 0` | open |
| 9 | NoPE layers: 128-dim nope query, zero rope, `wkv_b` 16384 rows, 528 B record | T1/T4 | P3, P4 | no | open |
| 10 | KV sharing: layers 39, 43, …, 71 read layer 35's pages and never write | T4 | P4, P1 (cache geometry) | spec field | open |
| 11 | SSMax: `q *= s_h · log(pos + 1) + b_h`, 64 attention heads and 32 indexer heads | T3 | P3 | no | open; fold into the query-rope stage only if the KL holds |
| 12 | YaRN × 2048 with LeRoPE `alpha` (interleaved; host cos/sin table; the drafter has no alpha) | T2 | P3 | `L/plan/rope.rs` recipe | open |
| 13 | Indexer query from `LayerNorm(h)` (bias) rather than the q-LoRA residual; `k_norm` LayerNorm with bias; SSMax; `[rope | nope]` layout | T2 | P3 | no | open; **smoke shortcut:** ≤ 2,048 visible keys equals dense MLA, so bring up dense first and the indexer second |
| 14 | Shared-indexer map: 28 layers reuse the closest preceding indexer layer | T4 | P4 | no | open |
| 15 | Final logits soft-cap `30 · tanh(x / 30)` (none in cuteafd today) | T3 | P3 | the shared head | open |
| 16 | Dense layer 0 (FFN 24576) and the shared expert (6144), FP8 block weight-only | T1 | P3 | no | open |
| 17 | Mistral API: stop ids, `[THINK]…[/THINK]` reasoning split, `[TOOL_CALLS]name[ARGS]{json}` parser, `reasoning_effort` passed to the template | T5 | P5 | `ModelEncoding` arm | open |
| 18 | Launcher arms, WIP scope, example config, gate card | T0 | P5 | yes | open |
| 19 | DSpark drafter (MLA backbone, 5 taps, RNN Markov head rank 512) | T6 | — | — | deferred: DeepSeek V4's drafter plus an RNN step is the base |
| 20 | Pixtral tower (30 layers > ABI depth 28; `[IMG_BREAK]` row layout) | T6 | — | — | deferred |
| 21 | Reference golden: a layer-major streamed torch reference with the image's formulas (the image needs 8 × B200 to run the model itself) | T1 | P6 | no | open |

## Packets

| Packet | Owner model (`USING_AGENTS.md`) | Gate | Status |
| --- | --- | --- | --- |
| P1 checkpoint + descriptor (gaps 1, 5, cache geometry for 9-10) | Sol | `cuteafd plan` on the stub names only package and program gaps | not started |
| P2 experts (6, 7) | Sol | unit tests; one MoE layer golden on a Spark | not started |
| P3 programs (8, 9, 11, 12, 13, 15, 16) | Sol; Opus for the SSMax rounding decision | CPU contracts; GPU oracle per program | not started |
| P4 layer chain + golden (9, 10, 14) | Opus subagent (per-layer semantics) | CPU `Ops` oracle at tiny geometry with every layer kind | not started |
| P5 serving + API + launcher (4, 17, 18) | Sol | `plan --deployment`; `run.sh --dry-run` | not started |
| P6 reference golden (21) | Sol | per-layer dumps for 4 windows covering dense, MoE, NoPE, KV-sharing and shared-indexer layers | not started |

## Phase 2 log

| When (with zone) | Step | Result | Next |
| --- | --- | --- | --- |
