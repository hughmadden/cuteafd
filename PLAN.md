# CuteAFD plan

One attention/FFN-disaggregated engine for the open-weights MoE models that
soar on consumer Blackwell: strong RTX PRO 6000 head(s) for attention, dense
projections, routing, shared experts, sampling and speculation; a pool of DGX
Sparks for routed experts over RoCE. Rust host code, CuTe-DSL/Triton AOT
kernels from our b12x fork, hand CUDA where it pays. Load any checkpoint of a
supported family directly from the HF snapshot. Say precisely what is missing
when something is not supported. Never slower than the engine it replaces.

Companion notes for agents: `AGENTS.md`. Code is king;
measurements are short tables in commit messages and `docs/` stays tiny.

## What we start from (survey 2026-09-28)

- `../ds41rt` is the base. Live path is the `v41_*` modules (serve, experts,
  target pass, attention, index, engram, vision, dspark) plus 7 crates:
  core, ffi, loader, transport, hostcache, api, daemon. ~2.2k Rust tests,
  ~60 qualification scripts, 16-lane parallel expert loader, RoCE verbs
  transport with TP2/3/4/6 × EP, dual-RTX TP2 local experts, GPU sampling,
  xgrammar constraints, live console, v15 images published.
- `ds41rt-daemon/src/commands/real_full/` (~167k LOC) is the legacy DS4
  path. Nothing live references it. It is not ported; DS4 Flash/Pro are
  re-hosted on the new engine instead (their kernels survive: `ds4_pro_aot`,
  `ds4_flash_aot`, `mla_indexing`, `packed_fp8_mla_exact`).
- `../ds4rt` has no engine code ds41rt lacks. It contributes the GPTQModel
  distributed quantization pipeline, Pro K2 evidence, and three API defaults
  (thinking on, 32K output budget, hidden internal model names).
- `../glmrt` is a port, not a merge (~150 divergent Rust files): GLM model
  code (MLA + DSA indexer, top-8 sigmoid router, dense first layers, shared
  expert path, native MTP), `dsa_indexer.cu`, mixed EXL3 K3/K4 top-8 kernels
  and loader layouts, DFlash2 block draft engine, GLM XML tool grammar,
  MoonViT vision, FP8/NVFP4/BF16 KV profiles.
- Kernel library: `../sparkinfer-glmrt` (b12x fork, master `7fcc094`,
  224 ahead of upstream) is the only fork; ds41rt already pins its head and
  it contains everything ds4rt/glmrt pinned. Publish there.
- `../GPTQModel` fork `main`; ds41rt imports `gptqmodel.utils.v41_*` which
  must be located (submodule copy or unpushed branch) before Phase 4.
- xgrammar v0.2.3 is pristine upstream; all customization is engine-side
  adapter code. Vendor as submodule with lock, same as today.
- Storage: sparknest FUSE at `/mnt/sparknest/hf-home` on every host. Local
  sealed copies read at NVMe passthrough speed; missing copies stream over
  the fabric at ~5 GB/s. `nest where` / `nest replicate` place copies.
  No Rust data-path client exists; the engine reads through the mount.

## Target models

Core (parity tier, must match or beat the parent engines):

| Model | Family | Format | Notes |
| --- | --- | --- | --- |
| deepseek-ai/DeepSeek-V4.1-Flash | deepseek_v41 | FP8 block + MXFP4 experts | engram ×2, dSpark 3-stage, vision. The regression anchor. |
| wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 | glm_dsa | EXL3 K4 experts | 78 layers, MLA+DSA, DFlash2 draft (`incoai/GLM-5.3-DFlash2`) |
| deepseek-ai/DeepSeek-V4-Flash-0731 | deepseek_v4 | FP8 block + FP4 experts | compress 4/128 alternating, nextn 1 |
| wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 | deepseek_v4 | EXL3 K2 experts | 61 layers, 7168 wide, ~96 GiB per Spark rank at TP4 |

Extend (new families; kernels largely exist in b12x already):

| Model | Family | New pieces |
| --- | --- | --- |
| zai-org/GLM-5.3-Flash, brandonmusic/GLM-5.3-Flash-tr3-4bpw | glm_next | hybrid KDA linear attention (34) + DSA MLA (11), mHC, EXL3 tr3 |
| XiaomiMiMo/MiMo-V2-Flash | mimo_v2 | GQA full + SWA with sink, no shared expert, FP8 |
| XiaomiMiMo/MiMo-V2.6-Pro-RL | mimo_v2 | 70 layers, 128 heads, mxfp4 store dtype, dflash dir |
| Qwen/Qwen3.8-Flash-Next (+ EXL3 K4.25 PLE variants) | qwen4_exp | GDN linear attention, n-gram memory tables, PLE, MTP; `../qflashrt` has a single-device port |

Speculation: one best speculator per family (native MTP/nextn, dSpark,
DFlash2). Adaptive width with calibrated confidence and cost model, as in
ds41rt `dspark_policy` and glmrt `dflash2_confidence`.

## Architecture

```
rust/crates/
  afd-core       ids, geometry, placement math, admission/lanes, KV allocator, sampling params
  afd-ffi        libloading C ABI; one module per kernel family, family-namespaced symbols
  afd-loader     checkpoint catalog, family readers (HF config -> ModelSpec), LoadPlan,
                 capability check, fast sliced readers, mapped tables, sparknest placement
  afd-transport  ExpertProtocolV2, verbs RoCE, TP x EP topology (unchanged from ds41rt)
  afd-hostcache  pinned host RAM prefix snapshots (unchanged)
  afd-engine     model-agnostic serve runtime: scheduler, lanes, prefix cache, memory,
                 speculative transaction framework, console state
  afd-api        OpenAI chat + completions, constraints, tools, images, console
  afd-families/  deepseek_v41, deepseek_v4, glm_dsa, glm_next, mimo_v2, qwen4_exp
                 each: spec reader, block execution, attention variants, speculator wiring,
                 chat template + tool parser + grammar generator
  afd-spec/      speculators: nextn_mtp, dspark, dflash2
  afd-daemon     CLI: serve, expertd, plan, inspect, doctor, bench-*
native/
  shared/        norm, sampling_gpu, embedding, router/route_reduce, expert_pack, peer_copy,
                 kv, engram/mapped tables, verbs, xgrammar adapter
  families/      deepseek_v41/ (hc, compressor, index*, sparse_attention, dspark*, vision),
                 deepseek_v4/ (mla_indexing, ds4_*_aot), glm/ (dsa_indexer, mixed exl3),
                 mimo/, qwen/
  aot/           b12x export manifests per family x format x SM x TP role
python/          reference impls, exporters, qualification tools
quantization/    unified distributed EXL3 pipeline (ds41rt + ds4rt + glmrt), gptqmodel fork
third_party/     sparkinfer, xgrammar, gptqmodel, transformers (submodules + tree locks)
build.sh wip.sh run.sh stop.sh   kept; config gains MODEL + family auto-detect
```

Traits the engine programs against (keep them few and concrete):

- `Family`: reads `config.json` + `quantization_config` + safetensors index
  into a `ModelSpec` (layer kinds, attention kind per layer, MoE geometry,
  speculator, mapped tables, vision) and builds the per-layer `Block`s.
- `Attention` per layer kind: CED compressed MLA (V4.1), alternating
  compressed MLA (V4), MLA + DSA indexer (GLM), GQA full / SWA + sink (MiMo),
  KDA / GDN linear (GLM Flash, Qwen). Each owns its KV layout and formats.
- `ExpertBackend`: Spark TP×EP over RoCE, local RTX TP1/TP2, dSpark-style
  drafter experts. Format-specific pack/unpack lives with the AOT shim.
- `Speculator`: propose, verify, commit/rollback inside the existing
  speculative transaction protocol.
- `MappedTable`: engram / n-gram memory / PLE tables in host RAM with bounded
  prefetch and GPU staging; later optionally Spark-resident.

Load plan and capability check (`cuteafd plan`):

- Runs without GPUs. Enumerates every tensor to owner (RTX0/RTX1/host-mapped/
  Spark rank slice), source format, kernel requirement (family, format,
  shape class, SM, TP role) and read plan.
- Checks against the kernel capability registry embedded in the built
  image (what AOT families and roles it carries) and against
  `nest where` for copy placement.
- Unsupported is a first-class result, never a crash: prints a hint block
  for a code agent naming the tensors, formats, shapes, the exporter or
  kernel to add, and the fast-read path missing (fallback: generic pread).
- Plan is the contract between coordinator and Sparks; every rank verifies
  the same plan hash at startup, as ds41rt does with `plan.json`.

Loading:

- Generalize the six-region TP read plan into a sliced-extent reader: given
  tensor shape, dtype/packing, slice axis and rank, produce coalesced extents;
  16-lane pinned staging, fadvise/O_DIRECT, per-rank local reads. No speed
  regression on V4.1 (readiness 31.7 s dual / 58.6 s single today).
- Before loading, the coordinator asks sparknest where copies live. Default
  reads through the mount wherever the file is. Optional `--place` runs
  `nest replicate` of the shards a rank needs onto that rank so repeated
  starts hit NVMe passthrough.

API and dashboard: keep ds41rt `native_v41` router; add `/v1/completions`;
per-family chat templates (deepseek-recipe crates for DeepSeek; minijinja
over `tokenizer_config.chat_template` as the generic path) with per-family
tool-call parsers and xgrammar tag grammars; console made model-agnostic.

## Phases

Each phase ends with: V4.1 Flash parity table (C1 code decode 1/2 RTX,
8K prefill, tool eval) unchanged or better, plus the phase's own table.
Commit small, push often. Opus executes all phases; Fable wrote the specs.

**Phase 0 — skeleton and anchor.**
Import, purge, restructure, and prove V4.1 parity on the new tree before
any new model work. Concrete recipe:

1. Import: `git -C ../ds41rt archive 3067d06 | tar -x -C .` then remove
   `.gitmodules` and re-add the four submodules at the same commits:
   sparkinfer `7fcc094e` (tpurtell/sparkinfer-glmrt, master),
   xgrammar `557becfb` (mlc-ai, v0.2.3), gptqmodel `5340775d`
   (tpurtell/GPTQModel, main), transformers `62d7ebd7` (malaiwah fork).
   Keep the `*.lock.json` tree locks and `verify-*-source.py`.
2. Purge in the import commit: `docs/` (940 evidence files), `runs/`,
   `ds41rt.build-v*.config`, `scripts/render-*`, `scripts/summarize-*`,
   `scripts/bench/` campaign files, `assemble-release-v2.py`,
   `update-ds41-v6-release-docs.py`, `release_semantic_quality.py`,
   `release_throughput_checks.py`, `migrate-layer-boundary-*`,
   `scripts/fixtures/release-*`, the matching `scripts/tests/test_v*` and
   `test_*release*` tests, and the legacy TCP scripts
   (`real-full-*`, `real-slice-*`, `start-spark-experts-tcp.sh`,
   `phase0-*`). Drop `README.md`, `DEVELOPER.md`, `AGENT_DEV_HINTS.md`,
   `architecture.md` (replaced by `AGENTS.md` here). Keep `LICENSE`,
   `THIRD_PARTY_NOTICES.md`, `docker/`, `examples/configs/`,
   `build.sh`/`wip.sh`/`run.sh`/`stop.sh`, `justfile`, `quantization/`,
   `python/`, `native/`, `rust/`.
3. Delete `rust/crates/ds41rt-daemon/src/commands/real_full/` and the
   legacy CLI commands (`Coordinator`, `Expertd`, real-full benches), then
   the Python tools and fixtures that only they used
   (`validate_ds4_*`, `validate_native_flash_*`, `tune_w8a16_*`,
   `tune_mtp_*`, legacy sparse-lm-head tuners). Native `ds4_*_aot.cu`,
   `mla_indexing.cu`, `packed_fp8_mla_exact.cu` stay for Phase 1.
   Build must pass after this step.
4. Rename: crates `ds41rt-*` → `afd-*`, binary `cuteafd`, native lib
   `libcuteafd_native`, symbol prefix `cuteafd_`, env/config prefix
   `CUTEAFD_`, image names `cuteafd-{coordinator,spark-expert}`, build
   cache `~/.cache/cuteafd/builds`. Mechanical, one commit.
5. Restructure: `native/{shared,families/deepseek_v41,families/deepseek_v4}`;
   `v41_*` daemon modules → `afd-families/deepseek_v41`; carve
   `afd-engine` (scheduler, lanes, prefix, memory, speculative transaction,
   console state) out of `v41_native_serve` behind the traits in
   Architecture. Do this incrementally with V4.1 serving between steps.
6. Generalize: `cuteafd.config` takes MODEL (hf id or path), REVISION,
   TOPOLOGY; the family reader replaces the embedded official config and
   model-id check with schema validation. Add `cuteafd plan` with the kernel
   capability registry and the unsupported-hint block. Add sparknest
   placement awareness (`nest where`, optional `--place`).
7. Serve V4.1 Flash on 2×RTX + 4 Sparks and on 1×RTX; record the parity
   table against ds41rt v15 in the commit message. Tag `p0`.

**Phase 1 — DeepSeek V4 family.**
`deepseek_v4` family on the new engine: V4 Flash 0731 (native FP8/FP4) and
V4 Pro EXL3 K2 (TP4, ~96 GiB/rank; consider TP6 across all six Sparks).
Reuse `ds4_*_aot`, `mla_indexing` kernels; nextn MTP speculator. Port the
three ds4rt API defaults. Targets: Pro ≥ 50 decode / 2,500 prefill tok/s
(ds4rt floor 33 / 1,650).

**Phase 2 — GLM 5.3 (glm_dsa) + DFlash2.**
Port from glmrt: DSA indexer kernel, top-8 router, dense/shared paths,
mixed EXL3 K3/K4 routes and loader layout, DFlash2 speculator, GLM tool
grammar, KV profiles. Target: ≥ glmrt's 25.96 weighted tok/s on 1 RTX,
better on 2 RTX with local expert layers.

**Phase 3 — new families.**
`glm_next` (GLM 5.3 Flash: KDA + DSA, mHC; b12x `kda_prefill`/`gdn_decode`),
`mimo_v2` (GQA + SWA sink; b12x paged FP8 KV attention), `qwen4_exp`
(from `../qflashrt`: GDN, n-gram tables via `MappedTable`, PLE, MTP).
Each lands with a `plan` that says what is missing before any kernel work.
Finish downloads/placement first (MiMo checkpoints are incomplete today).

**Phase 4 — quantization and release.**
Unify `quantization/` (ds41rt K3.25/FP4PLE, ds4rt Pro K2, glmrt K3.25
mixed) on the gptqmodel fork; extend to new families; publish quants under
wrldsuksgo2mars. Official images `ghcr.io/tpurtell/cuteafd-{coordinator,
spark-expert}`, concise README with one headline table.

Ongoing, any phase: engram/n-gram tables in host RAM now; Spark-RAM
replicas and fabric-fed tables are explorations, kept behind options.

## Decisions (2026-09-28)

- Fresh copy of ds41rt at `3067d06` into this repo; no history import. The
  import is also a purge: release evidence, perf traces, per-release configs,
  render/summarize scripts and archived patches stay behind in ds41rt.
  Keep qualification and bench tools that exercise live code.
- Delete `real_full` and the legacy commands outright; DS4 Flash/Pro are
  rebuilt as the `deepseek_v4` family.
- All seven hosts and all storage are ours to manage. Replicate a model to
  every rank while working on it, then shrink to one copy or 1/N when done.
  `/mnt/scratch` archive is slow (150 MB/s write, 500 MB/s read).
- Work on `main`, small commits, push often, tag phase boundaries.

## Cluster and rules

- raptor: 2× RTX PRO 6000 (SM120), capped at 325 W while TJ is away.
  Sparks: ostrich, dodo, emu, kiwi, rhea, moa (GB10, SM121, 121 GiB each),
  fabric 10.55.0.1–6. Six Sparks are the pool now; TP4 stays the qualified
  V4.1 default until a six-rank layout beats it.
- Never build on `/mnt/scratch` (NTFS kernel bug). Build under
  `~/.cache/cuteafd/builds/<task>` and keep ds41rt's filesystem assertion.
- Serialize WIP builds and performance runs; one model served at a time.
- Root via `agent-sudo`. Storage via `nest`; whole-file placement is
  explicit, reads never replicate silently.
- Fork changes go to `sparkinfer-glmrt` master and `GPTQModel` main first,
  then bump pin + tree lock here.
