# Reuse catalog: what exists, where, and how far it stretches

Distilled from reviews of every family on upstream `main` `1adc5bfb3` (30 Oct 2026) and the MiniMax
M3 port on the fork, with SparkInfer at `7d13eb37`. Paths are stable; **find symbols by grep**:
line numbers drift, and `docs/architecture/shared-engine.md`'s pointers already have. Prefixes:
`L/` = `rust/crates/cuteafd-loader/src/`, `D/` = `rust/crates/cuteafd-daemon/src/`,
`C/` = `rust/crates/cuteafd-core/src/`, `A/` = `rust/crates/cuteafd-api/src/`,
`SI/` = `third_party/sparkinfer/b12x/integration/cuteafd/`.

## 1. Pick a template per subsystem

A port rarely forks one family. Take each subsystem from the family that already has it, at the
smallest reuse tier that works (tiers: `taxonomy.md`).

| Feature in the new model | Best existing source | Tier | Stretch notes |
| --- | --- | --- | --- |
| MLA with q-LoRA | GLM 5.3 `glm_*` programs (`SI/glm_attention.py`); Kimi `kimi_mla.py` | T1 | GLM geometry guards: kv_lora 512, rope 64, index head 128; `softmax_scale` is derived, so a different scale needs a field |
| MLA **without** q-LoRA | GLM's `glm_stage_{rank_pack,absorb,query_rope,index_post}` composed around plain FP8 projections (the K6 path in `D/families/glm5/engine/native_dense.rs`) | T2 | the shared MLA packer `L/formats/mla.rs` refuses `q_rank == 0`: extend it |
| FP8 latent record 576 / 656 B (528 B without RoPE) | GLM sparse MLA; hand E4M3 prefill `native/families/glm5/cuda/glm_mla_prefill.cu`; Kimi `dense_mla` | T1 | record bytes are runtime; derive them from geometry instead of the 656/132 literals |
| DSA indexer, top-k selection | GLM `glm_index_producer`, `glm_index_topk_*` (DeepSeek's tiled top-k with a heads parameter) | T1 | the index query comes from the q-LoRA residual in GLM; without q-LoRA it needs a new input (T2) |
| Shared-indexer layers | GLM 5.3 (`L/families/glm5/config.rs`) | T1 | GLM Flash refuses them; check "closest preceding full layer" semantics |
| Cross-layer KV sharing | DeepSeek V4.1 `kv_source_layer_ids` (private, hard-coded) | T4 | make it a per-layer `kv_source` in the spec, cache geometry and prefix pages |
| Per-layer NoPE | `ModelSpec` per-layer kinds (`L/plan/spec.rs`); Kimi's NoPE MLA | T4 | encode as data and assert it in the golden |
| GQA, sliding window, sinks, paged INT8/FP8 records | MiMo `MimoGqaAttention` (`SI/`), used by dense Qwen | T1 | MiMo V2.6 needed its own program set because eps, theta, value scale and router dtype differed |
| Linear attention (GDN, KDA) | Qwen4 (GDN), GLM Flash and Kimi (KDA) | T1 | |
| CSA / SWA mixes | Step-5 | T1 | |
| RMSNorm | `glm_norm`; Step-5's generic `affine_norm` | T1 | match the reference's rounding point (GLM rounds before the weight) |
| LayerNorm with bias | Step-5 `affine_norm` (`python/tools/aot/export_b12x_step5_aot.py`) | T1 | |
| RoPE, YaRN | YaRN math in `L/formats/block_dspark.rs`; recipes in `L/plan/rope.rs`; Kimi drafter `_DraftRotary` with runtime `inv_freq` | T2 | host-side cos/sin tables cover any per-frequency variant (scaled, learned) |
| Router: softmax, sigmoid, sqrtsoftplus | `native/shared/cuda/expert_router.cu` (`softmax_select_kernel` renormalizes over top-k) | T0 | quirks in `C/dense_attention.rs` |
| Dense FFN, shared expert, router scores | `glm_ffn_i*`, `glm_router_scores`, `glm_expert_input_quant` | T1 | |
| FP8 128×128 block weights | `D/shared/fp8_linear.rs` (W8A16), `native/shared/cuda/fp8_block_dequant.cu`; GLM W8A8 prefill | T0-T1 | ModelOpt `FP8_PB_WO` = these blocks, scales `[R/128, 1, C/128, 1]` |
| Routed experts NVFP4 W4A4 / FP8 / EXL3 on Sparks | shared FP8-MoE worker; `GEOMETRIES` in `SI/fp8_moe.py`; Step-5's `step5:nvfp4` path is data only | T0 | gate/up `input_scale` must be bit-identical; compressed-tensors NVFP4 global scales divide (MiniMax) |
| Embedding, vocab head | `TokenEmbedding::load`; the shared BF16 head (V4.1-named but generic) | T0 | |
| Scheduler, decode lanes | shared Qwen scheduler `Backend` (`D/shared/qwen/scheduler.rs`); `LaneSet::run_one` + `lanes::commit` | T0 | 26 required methods, about 14 of them draft, probe or trace plumbing |
| Prefill | `PrefillHooks` + `run_prefill_units` (`D/shared/prefill_pipeline.rs`) | T0 | pin waves to 1 while hooks are serial |
| Spark exchange, owners, admission | `SparkLink::new_waves`, `admit_family`, the shared placement solver (`L/placement/`) | T0 | unknown families get silent `=> 0` arms in `L/plan/layout.rs`: add explicit ones |
| Prefix cache and host tier | the shared prefix engine through `Backend` | T0 | |
| Draft policy | `C/draft_policy.rs` | T0 | |
| DFlash2 drafters | `D/shared/speculation/dflash.rs` | T1 | |
| DSpark drafters | DeepSeek V4 (about 600 lines, runtime dims, `cuteafd_dsv4_markov_drafts`, `draft_binding.rs`); `D/shared/speculation/block_dspark.rs` (GQA backbone, "vanilla" Markov only) | T2-T6 | an RNN Markov head or an MLA backbone is new work: defer for a smoke |
| Vision towers | shared service `D/shared/vision.rs`; one hand-coded encode path per tower (`native/shared/cuda/vision_*.cuh`) | T6 | ABI depth fixed at 28 (`CUTEAFD_VISION_DEPTH`); defer for a smoke |
| Chat encodings and parsers | `A/openai/chat/<fam>.rs`; Kimi's native encoder; `ModelEncoding` enum | T5 | stop on the template's turn markers, not only EOS |
| Tokenizers | `tokenizer.json`; tiktoken shim `L/tokenizer/tiktoken.rs` | T0-T5 | |
| Checkpoint formats | `Checkpoint` (`L/plan/checkpoint.rs`), ModelOpt (`L/formats/modelopt.rs`), EXL3, EXLR8 inventory (`L/plan/exlr8.rs`: the precedent for a non-HF format) | T0-T5 | about 12 readers bypass `Checkpoint` and assume HF names |
| Golden and reference | `python/reference/families/<fam>/golden.py` (dense Qwen's is 158 lines); daemon golden subcommands; `scripts/bench/golden-run.py`; `scripts/bench/compare-logits.py` | T1 | `golden-run.py` maps one site's hosts: other cells need a shim |
| Layer chain with a CPU oracle | Kimi `D/families/kimi_k3/block.rs` (`Ops` trait) + `reference.rs` | T1 | the newest and cleanest pattern |

## 2. Families as templates

| Family | Take | Avoid |
| --- | --- | --- |
| `glm5` (GLM 5.3) | MLA + DSA programs, the stage programs, hand MLA prefill, shared-indexer handling, FP8 W8A8/W8A16 dense paths | the two-GPU executors, EXLR8, context-split programs (not needed for a smoke) |
| `glm5_flash` | the dSpark binding pattern; local experts; vision plumbing | KDA, mHC, pooled indexer, no-RoPE MLA and 52 `EngineArgs` fields |
| `kimi_k3` | the skeleton: config → descriptor → `Ops` chain + CPU oracle → device lanes → shared `Backend` → `PrefillHooks` units; MLA page banks | about 260 literal geometry constants; the fixed drafter geometry |
| `step5` | the NVFP4 W4A4 expert path (data only); `affine_norm`; per-layer-kind discipline (its layer 91 bug) | its two whitelisted layouts |
| `qwen3_5` (dense Qwen) | the smallest shared-built family (4,675 daemon lines, 1.8 % duplicated): `Backend` with no drafts, serve boot, a 158-line golden | |
| `mimo_v2` | the GQA/SWA/sinks attention class; quick bring-up history | its private prefill loop (the one remaining) |
| `qwen4` | GDN; the vision service wiring | 30K-graph startup sets |
| `deepseek_v4` | the 3-hour port order; the DSpark drafter (runtime dims); the program-table pattern | geometry checks that accept only its hidden sizes |
| `deepseek_v41` | nothing as a template: it is an imported engine (private scheduler, launcher, vision, NVFP4 path, fixed-geometry drafter) | everything private; shared pieces still carry its name (catalog, vocab head, expert ABI) |
| `minimax_m3` (fork) | the port record: the 12-step minimal path and the six shared-gap branches | the pre-v4 shapes it was re-hosted from |

## 3. Shared traps every port hits

- **Silent defaults.** `ExpertGeometry::swiglu_limit()` clamps any unlisted shape at 10; planner
  `spark_endpoints` and `spark_slot_bytes` return 0 for an unknown family, which under-admits
  rather than errors. Add explicit arms.
- **Family names in shared code.** Despite the guide's rule, `CacheOptions`, `DraftOwnership::selected`,
  35 of 75 `LayoutOptions` fields, the expert reader's name rules and the launcher's `run-family.sh`
  arms are keyed by family. A port must grep for a sibling family's name across the whole tree.
- **HF layout assumed.** `config.json`, `model.safetensors.index.json`, `tokenizer.json` and HF tensor
  names in about 12 readers. A snapshot without `config.json` silently launches on the DeepSeek V4.1
  path (`release_config_family` in `scripts/lib/release-common.sh`).
- **Catalog dispatch order.** `text_config` routes to the V4.1 reader before the `model_type` match.
- **Shape-baked programs fail closed** on any constant (eps, θ, value scale, router dtype): budget one
  geometry per checkpoint variant.
- **Rounding points** must match the reference (RMSNorm before the weight, SiLU of the gate); FP32
  re-association destroyed a drafter's acceptance once.
- **Masked sparse slots read record 0**: keep it finite (`zero_masked_v`).
