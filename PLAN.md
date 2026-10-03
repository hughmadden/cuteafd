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
  re-hosted on the new engine instead (the dsv4 programs replaced the
  legacy `ds4_*_aot` and `packed_fp8_mla_exact` kernels, removed 2026-09-30).
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
| wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 | glm5 | EXL3 K4 experts | 78 layers, MLA+DSA, DFlash2 draft (`incoai/GLM-5.3-DFlash2`) |
| deepseek-ai/DeepSeek-V4-Flash-0731 | deepseek_v4 | FP8 block + FP4 experts | compress 4/128 alternating, nextn 1 |
| wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 | deepseek_v4 | EXL3 K2 experts | 61 layers, 7168 wide, ~96 GiB per Spark rank at TP4 |

Extend (new families; kernels largely exist in b12x already):

| Model | Family | New pieces |
| --- | --- | --- |
| zai-org/GLM-5.3-Flash, brandonmusic/GLM-5.3-Flash-tr3-4bpw | glm5_flash | hybrid KDA linear attention (34) + DSA MLA (11), mHC, EXL3 tr3 |
| XiaomiMiMo/MiMo-V2-Flash | mimo_v2 | GQA full + SWA with sink, no shared expert, FP8 |
| XiaomiMiMo/MiMo-V2.6-Pro-RL | mimo_v2 | 70 layers, 128 heads, mxfp4 store dtype, dflash dir |
| Qwen/Qwen3.8-Flash-Next (+ EXL3 K4.25 PLE variants) | qwen4 | GDN linear attention, n-gram memory tables, PLE, MTP; `../qflashrt` has a single-device port |

Speculation: one best speculator per family (native MTP/nextn, dSpark,
DFlash2). Adaptive width with calibrated confidence and cost model, as in
ds41rt `dspark_policy` and glmrt `dflash2_confidence`.

## Architecture

```
rust/crates/
  cuteafd-core       ids, geometry, placement math, admission/lanes, KV allocator, sampling params
  cuteafd-ffi        libloading C ABI; one module per kernel family, family-namespaced symbols
  cuteafd-loader     checkpoint catalog, family readers (HF config -> ModelSpec), LoadPlan,
                 capability check, fast sliced readers, mapped tables, sparknest placement
  cuteafd-transport  ExpertProtocolV2, verbs RoCE, TP x EP topology (unchanged from ds41rt)
  cuteafd-hostcache  pinned host RAM prefix snapshots (unchanged)
  cuteafd-engine     model-agnostic serve runtime: scheduler, lanes, prefix cache, memory,
                 speculative transaction framework, console state
  cuteafd-api        OpenAI chat + completions, constraints, tools, images, console
  cuteafd-families/  deepseek_v41, deepseek_v4, glm5, glm5_flash, mimo_v2, qwen4
                 each: spec reader, block execution, attention variants, speculator wiring,
                 chat template + tool parser + grammar generator
  cuteafd-spec/      speculators: nextn_mtp, dspark, dflash2
  cuteafd-daemon     CLI: serve, expertd, plan, inspect, doctor, bench-*
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

Fabric discovery (transport, at every startup, coordinator and ranks):

- Probe and log per host: each RDMA device and port (`/sys/class/infiniband/
  */ports/*/{rate,state}`), its netdev, negotiated speed and MTU, the rail
  addresses/GIDs and whether rails sit on isolated subnets, and the PCIe
  link generation and width of the NIC and of the GPU. Publish this in the
  startup plan so every rank sees the whole fabric picture.
- Choose the queue-pair strategy from that data, not from a constant:
  rail count, which rail carries request/response vs. reduction traffic,
  in-flight depth and chunk sizes. Lesson learned: dual rail at 100G links
  with 200+ Gb/s of inbound PCIe traffic caused head-of-line blocking, so a
  second rail is only used when link rate and PCIe ingress justify it.
  Only one physical configuration exists at a time, but the engine records
  the inputs and the chosen strategy so the choice can be revisited.
- Today: Sparks negotiate 200 Gb/s on two ports (rail A 10.55.0.x, rail B
  10.55.1.x, separate subnets); raptor has one 400 Gb/s port carrying both
  rail subnets. The switch is moving from 100G to 200G; verify the
  effective rate rather than trusting the port. NCCL is not required, so
  its isolated-subnet rule is informational only.

API and dashboard: keep ds41rt `native_v41` router; add `/v1/completions`;
per-family chat templates (deepseek-recipe crates for DeepSeek; minijinja
over `tokenizer_config.chat_template` as the generic path) with per-family
tool-call parsers and xgrammar tag grammars; console made model-agnostic.
Status (2026-10-02): the live console (`/`) is family-neutral: `shared/console` (producer
`Ticket`/`Step`/`layer_mark`, console thread, wire schema) fed by every serve loop; each family
declares its header, speculator, micro-step stages and layer classes in a `console::Layout`.
Shared page shell, palette and SVG chart primitives at `/assets/cuteafd-ui.{css,js}` (also used
by `/bench`); logo and favicon at `/assets/cuteafd-{logo,mark}.svg`.

## Execution engines (decided 2026-09-29)

V4.1's execution stack is specialized to its causal encoder/decoder: KV and
index sources [2,8,14,20], engram gates at layers 1 and 14, dSpark taps from
layer 37, encoder/replay stages. Other families share none of that, so there
are two engines under one scheduler, API, transport, host cache and sampler:

- `deepseek_v41` keeps its specialized path; nothing regresses it.
- A generic per-layer engine runs every other family: each layer is a block
  (norm/HC in, family attention with its own window/compressed/index/state
  cache, router, shared expert, routed experts on Sparks, HC/residual out),
  plus head and native speculator. DeepSeek V4 is its first tenant, then GLM,
  MiMo, Qwen. It adopts V4.1's proven optimizations (chained stages, lanes,
  dual RTX, route packing) as it matures.

Kernels are shared by parameterizing dimensions; `cuteafd plan` feeds the
AOT exporters the geometry each image must carry.

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
   `mla_indexing.cu`, `packed_fp8_mla_exact.cu` stayed for Phase 1 (the
   unbound ones went in the layout purge).
   Build must pass after this step.
4. Rename: every `ds41rt` token → `cuteafd` (crates `cuteafd-*`, binary `cuteafd`, native lib
   `libcuteafd_native`, symbol prefix `cuteafd_`, env/config prefix
   `CUTEAFD_`, image names `cuteafd-{coordinator,spark-expert}`, build
   cache `~/.cache/cuteafd/builds`). Mechanical, one commit.
5. Restructure: `native/{shared,families/deepseek_v41,families/deepseek_v4}`;
   `v41_*` daemon modules → `cuteafd-families/deepseek_v41`; carve
   `cuteafd-engine` (scheduler, lanes, prefix, memory, speculative transaction,
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

**Phase 2 — GLM 5.3 (glm5) + DFlash2.**
Port from glmrt: DSA indexer kernel, top-8 router, dense/shared paths,
mixed EXL3 K3/K4 routes and loader layout, DFlash2 speculator, GLM tool
grammar, KV profiles. Target: ≥ glmrt's 25.96 weighted tok/s on 1 RTX,
better on 2 RTX with local expert layers.

**Phase 3 — new families.**
`glm5_flash` (GLM 5.3 Flash: KDA + DSA, mHC; b12x `kda_prefill`/`gdn_decode`),
`mimo_v2` (GQA + SWA sink; b12x paged FP8 KV attention), `qwen4`
(from `../qflashrt`: GDN, n-gram tables via `MappedTable`, PLE, MTP).
Each lands with a `plan` that says what is missing before any kernel work.
Finish downloads/placement first (MiMo checkpoints are incomplete today).

**Phase 4 — quantization and release.**
Unify `quantization/` (ds41rt K3.25/FP4PLE, ds4rt Pro K2, glmrt K3.25
mixed) on the gptqmodel fork; extend to new families; publish quants under
wrldsuksgo2mars. Official images `ghcr.io/tpurtell/cuteafd-{coordinator,
spark-expert}`, concise README with one headline table.

**Prefix cache for every family (top priority, design 2026-09-30).** Only
V4.1 has one (radix banks Prompt/Turn keyed by token ids, shared FP4 pages +
a 2.72 MB copied "front" of 128 SWA rows per layer, 128-token approximate
replay for partial hits, pinned-host tier in `cuteafd-hostcache`). Generic
version in the engine crate: `PrefixFamily` trait (layout, capture/restore of
the positional "mark", shareable page rows, commit point), one refcounted
`RefPagePool` with CoW tails replacing the five per-family free lists, a
device mark arena sized by lanes (recurrent marks are 110–141 MiB) with the
host tier holding the rest, Hugh-style victim order and 64-token hash-chained
host page identity, `After{greedy, logits}` for exact-length hits, drafters
restored cold with a `context_valid_from` mask. Prompt snapshot is the
guarantee; Turn snapshot opportunistic, later made exact by a template-aware
canonical turn snapshot. Order: S0 engine crate + agentic reasoning-on
benchmark (record/replay of multi-turn tool sessions; turn TTFT, hit rate,
decode tok/s), S1 MiMo V2.6 Pro (+V2 Flash), S2 GLM 5.3 Flash, S3 GLM 5.3,
S4 Qwen, S5 V4 Flash/Pro, S6 canonical turn snapshot, S7 embedding cache.
Gate: resume-at-P restores are byte-identical to straight prefill; greedy
text identical to cache-off; V4.1 parity unchanged.
Status: S0-S5 merged (S2 GLM 5.3 Flash: 256-row units = 4 MLA pages + the
pool page, 140.8 MiB KDA mark, kda_len commit point; S3 GLM 5.3: pages only,
host tier via a stand-in tail; S4 Qwen 3.8: the same 256-row units over 12
full-attention layers, 110.3 MiB GDN+PLE mark, state_len commit point, n-gram
history recomputed from the ids; S5 V4 Flash/Pro: 256-token units = the C4, index and C128 pages
of one index, 27.9 / 42.2 MB mark = every layer's last 128 window rows, dSpark rings included
(drafts warm), plus the compressors' FP32 rolling state; commit point = placement length), GPU-gated
with experts skipped; S5's live agentic gate ran on Sparks (V4 Flash TP4 + dSpark, one
session: 8/8 turns reuse the whole previous turn, later-turn hit ratio 0.93, TTFT 0.83 -> 0.20 s,
task solved); the others next (Qwen: one live session with local EXL3 experts on one RTX). The DSA index top-k (b12x `tiled_topk`: GLM 5.x, GLM 5.3
Flash, V4 Flash/Pro) is deterministic: ties go to the lower index and picks
come out in ascending order, so `--resume-at` is byte-exact past 2048 tokens
too. Still arrival-ordered: the fused decode route (`persistent_topk`, used
only by plans of 16 rows or fewer, not by the exported m64 programs).

**Repo layout and naming (study 2026-09-30; run at a quiet point after a merge
round, before prefix-cache S1 and planner S0 create new modules).** Family ids:
long ids `deepseek_v41`, `deepseek_v4`, `glm5`, `glm5_flash`, `mimo_v2`,
`qwen4` for directories, modules, plan ids, goldens; short tags `v41`,
`dsv4`(f/p), `glm`, `glmf`, `mimo`(p), `qwen4` only for C symbols, AOT
program prefixes and package dirs (baked into manifests; unchanged). Target:
`shared/` + `families/<id>/` in the daemon, ffi and loader; transport
`v41_expert` → `expert`; api `native_v41` → `openai` + `chat/<id>`;
`native/{shared,families/<id>}/{cuda,src,include}` + `cmake/{shared,families}`;
`python/reference/families/<id>`, `python/tools/{aot,bench,hf,qualify/<id>}`;
`scripts/{lib,build,launch,bench/<id>,qualify/<id>}`; multimodal input path in
the engine crate, per-family encoders beside their family. Steps: P1 purge
dead legacy (≈6.4k lines of unbound ds4/b12x/w8a16 kernels, python runtime,
stale justfile recipes, unused core modules), P2 baselines; M1–M9 pure
`git mv` commits with only forced build-file edits and root re-exports
(`crate::v41_*` paths keep compiling), verified per group, plus
`path-map.tsv` and a rebase-across-move script; then a naming pass (generic
types off V4.1 names, one `serve`/`expertd`/`golden` CLI with family
detection, `run.sh` absorbs `run-dsv4.sh`, cmake/config/env aliases for one
release; shared C symbols and headers drop `v41` with no aliases, since the
library and binary ship together; V4.1-specific names stay). ≈3 days move pass,
≈3–4 days naming pass; V4.1 parity before tagging.
Status: purge, move pass (M1–M8) and naming pass (N1–N9) done on
`work/restructure`, N10 (shared C symbols) on `work/p0`;
`scripts/build/path-map.tsv`, `rename-map.tsv` and `rebase-across-move.sh`
carry older branches across. Full native AOT build,
V4.1 parity vs 9015bff, GLM 5.3 golden NLL and a MiMo launch checked on
hardware before the merge into `work/p0`; M9 (fork layout) separately.

**Phase 5 — NVIDIA ModelOpt NVFP4 checkpoints (queued; after the families
above reach their performance targets).** nvidia/{DeepSeek-V4.1-Flash,
GLM-5.3, GLM-5.3-Flash, Qwen3.8-Flash-Next}-NVFP4. Common contract: U8
`[N,K/2]` E2M1 low nibble first, E4M3 per-16 scales in linear layout,
F32 `weight_scale_2`, static `input_scale` (W4A4 spec); routed experts are
FP4, most other weights BF16/FP8 as released. Design (study 2026-09-30):
- One `QuantOperand` descriptor in the loader, read from
  `hf_quant_config.json`/`config.json` by one ModelOpt reader; model code
  and engines never see the format.
- Routed experts: `ExpertFormat::Nvfp4` in the existing `fp8-<geom>`
  package (ABI word 3), W4A16 by default (E2M1×E4M3 is exact in BF16, so
  it reproduces NVIDIA's weights with unquantized activations); scale
  swizzle, when a kernel wants it, runs on the GPU at load
  (`nvfp4_scale.cu`, parameterized); no offline repack.
- Dense NVFP4/per-tensor-FP8 parts should retain their compact checkpoint
  representation. Existing load-time BF16 fallbacks are compatibility debt
  to close under the v1 resident-weight work below.
- V4.1: its NVFP4 release has exactly the official MXFP4 weights
  (power-of-two scales) → lossless downcast at load onto the existing W4A8
  path; the W4A4 44-slot family stays opt-in (ds41rt measured it slower).
- W4A4 prefill (MmaMXF4NVF4Op, in-kernel per-16 quantization) only if it
  measures faster and stays within 0.005 nats KL of W4A16.
Stages: S0 loader + `plan`; S1 W4A16 experts (GLM 5.3 Flash first, then
Qwen on one RTX: 68 GB); S2 dense + MTP dispositions; S3 V4.1
convergence; S4 W4A4 prefill experiment. Gates per stage: oracle cosine,
KL vs golden within 0.005 of the FP8-expert path, tok/s ≥ it, readiness
not worse, V4.1 parity. GLM 5.3 NVFP4 experts (~407 GB) need TP6.
Status (2026-10-02): S0, S1 and the S4 kernels landed (`work/nvfp4`, fork
`cuteafd/nvfp4-w4a16`). The ModelOpt reader (`formats/modelopt.rs`) checks
every weight against hf_quant_config.json / config.json. `glm|glmf|qwen4:nvfp4`
build `fp8-<family>-nvfp4` packages (W4A16: GEMV, stream above 2048 rows);
`:nvfp4a4` builds W4A4 large-row steps (static input_scale, mxf4nvf4 MMAs,
above 512 rows); every `:nvfp4` entry also builds it and W4A4 is the default
(`CUTEAFD_NVFP4_ACTIVATIONS=a16` keeps W4A16). Native SM121
packages pass the CPU oracle on GB10 at tp2/3/4/6. GLM 5.3 Flash TP4 Sparks
(EXL3 K3.25 coordinator weights): NVFP4 W4A16 NLL 2.3880 / KL 0.0587 (EXL3
K3.25 2.4082 / 0.0616), C1 step 16.3 ms vs 15.3 (4.5 vs 3.25 bits read), 8K
prefill equal; W4A4 KL 0.0799, prefill 1.33x. Qwen on one RTX: NVFP4 8K
prefill 1.08x EXL3 (W4A4 1.36x), 1.6K 0.96x (W4A4 1.33x). W4A4 costs
+0.02 KL (over this plan's 0.005 bound) but is the checkpoint's calibrated
numerics: W4A4 is the default (TJ, 2026-10-02). S2: GLM 5.3 Flash NVFP4 dense MLPs run natively (one-expert
`fp8-glmfdense-nvfp4`): nvidia/GLM-5.3-Flash-NVFP4 serves alone (NLL 2.3896);
nvidia/GLM-5.3-NVFP4 serves on six Sparks (TP6 W4A16: NLL 2.4814 / KL 0.0571
vs golden 2.4677; 8K prefill 4.22 s, W4A4 3.20 s). Its per-tensor FP8 dense
MLPs prefill as static W8A8 on their own input_scale / weight_scale (plain
E4M3 MMAs; decode GEMVs read the same bytes under a uniform grid): NLL 2.4827 /
KL 0.0580, MLP 1.4x faster than block W8A8 at 4096 rows. Its BF16 attention,
indexer and shared experts quantize to FP8 blocks at load by default; with
`CUTEAFD_GLM_BF16=native` they run as-is on the BF16 programs (one or two
RTX): KL 0.0491, but C1 step 41.7 vs 34.3 ms with the Sparks (coordinator
alone 28.6 vs 19.2 ms one RTX, 19.8 vs 16.1 two), coordinator 8K prefill
3.35 vs 2.75 s (2.36 vs 2.07), weights 32.3 vs 17.6 GiB. FP8 blocks stay the
default (TJ, 2026-10-02): the official zai-org GLM 5.3 ships these tensors as
128x128 block FP8, so the load-time conversion matches the official format. W4A4 gate/up + SwiGLU + FP4 quant run fused (bit-exact;
layer 7-17% faster, GB10 13-17%). Open: W4A16 stream efficiency (GB10 4096
rows 14.3 ms/layer TP4 vs EXL3 9.1), SM121 route thresholds.

**Phase 6 — placement planner (design 2026-09-30).** One planner for every
family: (model, inventory of 1–2 coordinator GPUs — real or simulated by a
memory budget — and 1–8 Sparks, KV target, objective) → a hashed
`placement.json` that the loader, workers, engines and launchers all
consume. It generalizes V4.1's startup handoff (`v41_native_serve/placement.rs`),
live memory planner and 20/20 backbone split. Per MoE layer the experts are
resident on GPU0/GPU1 (full width, or TP2 where measured) or on a Spark group
with TP×EP and uneven whole-block slices; EP means expert subsets per group
(new: today groups are replicated). Cost model counts the busiest rank
(E[max] routed experts per group), fabric intake as a shared prefill
resource, and per-family coordinator step tables; search is exhaustive.
Cold components get dispositions: official vision/audio towers run whole on
a Spark when they fit (`ENCODERS=rtx0` pins them) with a hash-keyed
embedding cache refcounted by the prefix cache; MTP layers are unused when a
DFlash2 drafter drafts. Dual-GPU coordinator default is the layer-range split
(memory, ~1.9× GPU-bound prefill); TP2 of dense layers only if a P2P probe
shows it pays (GPU0/GPU1 cross the host bridge; ds41rt measured it a loss).
Evaluate per family, not globally: GQA models with many KV heads (MiMo V2.6
Pro: 128 q × 192, 8 KV heads, 16K-wide o_proj, ~18 GB streamed per token)
can head-split attention across two GPUs with KV partitioned (4+4 KV heads,
no replication) and one hidden all-reduce per layer — a possible C1 win the
DeepSeek MLA models never showed (ds41rt's V4.1 head split with replicated KV
measured −5…−8%; its attention weights are small next to hop and launch
costs). GLM 5.3 is the other candidate: MLA, but ~205 MiB of coordinator
weights per layer (o_proj 96, q_b 32, kv_b 14, shared expert 36), ~16 GB
streamed per token, and a small replicated latent (656 B/token/layer). DCP2
(KV split by sequence) is a capacity-only option and is not needed for V4.1
(compressed KV, 14M-token default pool). Order: P2P probe, then head-split vs
layer-range A/B for MiMo Pro and GLM 5.3.
Status (2026-10-01): `cuteafd fabric --p2p` measured GPU0<->GPU1 (NODE) hops of
3.3 us for 12 KiB (SM push + release flag, graph), a two-way exchange of 3.4 /
4.8 / 25 us for 12 KiB / 96 KiB / 1 MiB and 1.1 ms for a 48 MiB prefill chunk
(copy engine 0.9 ms); saturating host->GPU0 ingress roughly quadruples small
hops. So the head split pays and is the default with two RTX
(`--split-device`, run-family `RTX_GPUS`/`COORDINATOR_GPUS`, `COORDINATOR_SPLIT=off`
opts out): MiMo V2.6 Pro (coordinator-only decode -41%, 8K prefill -43%; with
6 Sparks C1 decode 30.9 -> 26.0 ms, prefill Spark-bound) and GLM 5.3
(coordinator-only 8K prefill -28%; with 6 Sparks decode -9.5%, prefill
Spark-bound) and DeepSeek V4 (4 Sparks, all experts remote: Flash decode -10%,
Pro decode -12%, prefill neutral). Shared plumbing in `shared/peer_split.rs`: per-slot release flags,
partials exchanged and summed in the same operand order on both GPUs (identical
residual streams), GPU1 queued a layer ahead of GPU0's Spark exchange, decode
graphs captured per GPU. The layer-range split is not needed for these three. V4.1 Flash re-measured on p8 images (2 RTX + Sparks, code,
dSpark on): its TP2 modes still lose (C1 187 -> 170 tok/s for TP2_ATTENTION and for
TP2 q+o projections; C4 501 -> 466 / 482). Each projection there ends in a
cross-device event and a host wait, so a gain needs V4.1's per-layer flow rebuilt
around device-side flags. The attention-only ceiling is about the DeepSeek V4 Flash
split (-10%, all experts remote), and less with RTX-resident expert layers, so
V4.1 keeps its layer split. Measured 2026-10-02 (C1 code, 6-row verify rounds of
~27 ms): each RTX is busy ~33% (C4 ~47%), the head-splittable work (q_b, sparse
core, wo_a, wo_b) is ~115 us of a ~170 us attention block per layer, and a 32-head
core saves only 9 of 24 us. With KV replicas, peer inputs and the exchange, a
V4-style split projects +2-4% C1, ~0-2% C4, less at C16, and the replicated 14M-token
pool (+5-8 GB per card) costs two RTX expert layers at the release config: not built.
V4.1's lever is its host-driven layer. Per layer at C1 (5-row verify, nsys): remote
(20 layers) 789 us = GPU to route ids 118, route D2H to host 12, Spark round trip and
collect 544, combine to next attention 99 (55 of kernels, the rest host launch
gaps); local TP2 (20) 450 us = 120, route to expert launch 35, experts 164, host-run
TP2 reduce 31, next attention 96. C4 lanes: remote 1086 (round trip 782), local 673.
A device-driven exchange (GPU-written requests + proxy post, GPU-landed replies with
a NIC-written flag, device wait, device-flag TP2 reduce) projects C1 +5-7%, C4
+3-6%; also capturing the whole verify step (no launch gaps) C1 +10-15%, C4 +8-12%.
Unknown: the host share inside the Spark round trip (worker-side timing needed).
Outside the layer loop: ~0.8 ms host gap per round after argmax, and two BF16
vocab heads (target, draft) of 450 us each.
The drafter follows the GPU that owns the last backbone layers (taps and head
live there); TP2 drafters are ≤1% on DFlash2 and not built unless the P2P
probe shows ≤15 µs hops; the win is lane B drafting on GPU1 while lane A
verifies on GPU0 at C≥2. Benchmark only the natural minimum and maximum
configs (AGENTS.md); the planner's estimates cover the rest.
One `ExpertRouter` replaces the six per-family stage/send/land/reduce copies
and is the device-driven Spark exchange below (V4.1 moves onto it too). Stages: S0 planner + `plan` (must
reproduce today's layouts), S1 manifest handoff + workers, S2 router in the
generic engines + GPU1 as expert host, S3 EP subsets (only if a quantized
model needs them; GLM 5.3 official FP8 is out of scope — EXL3 and NVFP4
quants cover it), S4 encoder service + multimodal input, S5 coordinator
range split, S6 eight Sparks.

**Device-driven Spark exchange (decided 2026-10-02, `work/v41-device`).** No engine is
device-routed toward the Sparks today: every family (V4, GLM, GLM Flash, MiMo, Qwen) downloads
route ids, weights and wire rows per MoE layer, synchronizes the stream, builds the request on
the host, posts the RDMA sends, polls the CQ and only then queues the reduce; only RTX-local
expert layers read device routes (V4 `local.rs`). V4.1 has its own exchange (`NativeTp4Wave`,
`receive_owned` with per-frame H2D uploads, host ownership planner) plus host waits inside its
TP2 RTX expert layers (`chain::settle`, peer-copy `wait()`s). Measured on V4.1 (p8, 2 RTX + 4
Sparks, C1 code, nsys): a ~29 ms verify round keeps some GPU busy only 14.7 ms; the RTX-expert
layers 0-18 leave ~2.5 ms with both GPUs idle (host hops between attention, routes, TP2 slices,
peer reduce), the Spark layers ~10.9 ms (21 x ~470 us Spark round trips plus ~1 ms of host
hops). Decision: retrofit, not a port. V4.1 is not served through the V4 engine (no shared
attention, cache, encoder, engram or dSpark code; the V4 engine's Spark path is just as
host-driven); instead one shared exchange is built and both engines move onto it:
- GPU side: the router's ids, weights and wire rows are copied into a pinned, device-mapped
  mailbox and a kernel publishes a sequence number (release, system scope); the combine waits on
  the proxy's completion sequence with an acquire spin (`peer_exchange.cu`'s graph-safe pattern:
  sequences live in device memory, so replays advance them). Replies land GPU-direct in the
  intake planes (dma-buf), so a decode/verify step can be queued, and later captured, whole.
- Host side: one proxy thread per transport (the prefill lane pattern) spins on the mailbox,
  builds the request from it, posts the sends, polls the CQ, validates, and publishes
  completion; no inference-thread sync, D2H parse or launch on the critical path. GPU-initiated
  doorbells (IBGDA via mlx5dv) only if measured to pay over the proxy.
- Stages: D0 shared component (`shared/spark_intake` + `cuteafd-transport` device lane, native
  signal/wait kernels); D1 DeepSeek V4 decode/verify on it (opt-in `CUTEAFD_SPARK_DEVICE=1`),
  measured against the host path; D2 V4.1 Spark layers adopt SparkIntake/GPU landing and the
  device exchange, ownership choice on the device or precomputed; D3 V4.1 TP2 RTX expert layers
  without host waits (device-ordered peer copies and reduce); D4 whole-step graph capture
  (decode/verify), head split re-checked; D5 prefill; D6 other families adopt it. The Spark
  worker (`shared/experts/service.rs`) is host-driven too (CQ, launch, sync, send): its per-wave
  host overhead is recorded and a device-driven worker proposed if material. Gates per stage:
  golden NLL / byte-exact greedy vs the host path, prefix-cache restore exactness, quick C1/C4
  A/B on 1 RTX + 4 Sparks and 2 RTX + 4 Sparks, full V4.1 parity before proposing a default;
  the default stays byte-identical while off.
  MiMo (2026-10-02): decode/verify already run as captured per-layer segments between exchanges
  (`--decode-graphs`, opt-in): byte-exact, but neutral against the host exchange (the GPU bounds
  each segment), so D6 for MiMo is capturing those segments back to back.

**Spark-side reduction (measured and parked 2026-10-01, `work/spark-reduce`).**
TP ranks reduce-scatter their routed partials by rows over an RC mesh between
the Sparks (`expertd --reduce-rail`, SEND_WITH_IMM tagged per wave, FP32 sum in
rank order) and each returns only its rows, so the coordinator lands one plane
instead of N. Correct (sums bit-identical to the coordinator reduce, oracle
cosine 0.999995, MiMo V2.6 Pro golden NLL unchanged at 2.4123), but MiMo V2.6
Pro TP6 8K prefill gained only ~3% on one 200 Gb rail and ~9-12% on two: every
rank waits for the slowest peer's slice, the exchange is bandwidth-bound
(NCCL's ceiling on rhea+moa: ~20 GB/s per direction on two rails, ~11 on one),
and it gets worse at 100 Gb. Coordinator-side intake and pipelining come first.

Ongoing, any phase: engram/n-gram tables are memory-mapped from the
checkpoint (`formats::mapped_table` + daemon `shared::mapped_table`: page
cache, bounded prefetch, gather pool/worker, pinned upload ring, stats; V4.1
engram and Qwen PLE use it); Spark-RAM replicas and fabric-fed tables are
explorations, kept behind options.

## Release v0 (2026-10-02)

Cut early from `work/p0` (p9 + benchmark dashboard + live console for every
family + README card grid) with the V4.1 8K prefill staging regression fixed.
The full Release smoke matrix (family × quant × natural-minimum and maximum
hardware) published as the README card grid is the v0 artifact. Images stay
local until TJ says to push them. No model license notes: we bundle no weights.

## Release v1 — priority plan (2026-10-02)

Everything after v0 lands as v1. Helpers: read AGENTS.md, then pick the top
open item; each names its branch (pushed WIP) and the next step. Merge green
steps into `work/p0`; tag `v1.0.0` when the list's top half is done.

Current agent integration: `codex/v1`, isolated from the orchestration checkout.
The user chooses release cuts; development fixes below do not cut a release.

Joint serving capacity is in progress: the pure resolver budgets each physical
GPU at 97% of total minus existing usage and explicit reservations, preserves
small pool overrides, and reports the common 2,097,152-token target and its
shortfall. `cuteafd plan` now describes canonical target-only cache
storage for GLM, GLM Flash, Qwen and MiMo, including replicated versus
head-partitioned KV, C16 / 20 state slots and exact prefix mark bytes. It does
not infer a compiled index limit or claim that weight placement alone admits
the serving configuration. Native MTP costs are available when explicitly
requested; target-only costs do not include an external drafter.
MiMo workspace allocation now consumes the same pure size description that
startup admission will use. Split peer workspaces and fully split target
prefill use their assigned attention heads, including INT8 prefill BF16
shadows; lead decode retains global geometry for unsplit MTP. The change
passed exact full-model restore, pipeline and continuation checks.
Serving also admits only the final prefill logits row; diagnostic engines
retain all rows, and decode/verification keep their full output extent.
The smaller allocation passes both reference layouts' exact logits/KV checks,
split-rank cache restores, continuation and serving with speculation on/off.
Unsupported explicit MiMo head splits now fail before loading, and stream
cleanup preserves the primary error. Failed split-rank submissions now abort
queued peer waits; teardown releases owners only after proven retirement and
retains them when completion is unproved.
MiMo expert scratch, Spark intake and startup negotiation now admit at least
the full decode/verify extent even when prefill chunks are smaller. The
narrow-prefill full-model gate matches physical KV/marks, every layer and
continuation exactly; the old path fails its native capacity guard.

Next: startup must consume the same resolved pool/context/state values before
loading weights, with actual weight conversions, all lane/workspace shapes,
native scratch, prefix marks, transport and optional draft allocations in the
profile. Keep checkpoint maximum and effective compiled context separate.
The common default target is 2,097,152 logical GPU KV tokens for every family,
including DeepSeek, with C16 and 20 front-state slots. Reserve that pool and
all steady runtime storage first, then onboard the maximum expert layers from
the remaining per-device budget. Admit weight-loading temporaries separately
so released staging does not reduce steady capacity. Larger pools and smaller
benchmark overrides remain explicit launch options. Report hardware shortfall
and maximum-context feasibility; do not silently change precision or context.
MiMo Pro's checkpoint-preserving one-RTX 2M configuration does not fit after
mandatory workspaces/state and fails admission before loading. The private
two-RTX candidate passes physical allocation accounting through all decode
graph shapes, full draft/masked sampling and bounded host restoration, with
one shared prefill KV shadow per rank. Grammar ownership and the complete
serving footprint still need qualification before changing defaults. Bounded
host storage retains inactive exact prefixes for every supported family;
active admission deferral and active KV paging are separate remaining work.

1. **Device-driven Spark exchange, shared by every family** — branch
   [`work/v41-device`](https://github.com/tpurtell/cuteafd/tree/work/v41-device). Today every family does 2–3
   blocking host round trips per MoE layer (router ids D2H + sync, host-built
   request and RDMA post, host CQ poll before the reduce); no device-initiated
   networking exists. Design: GPU kernels write requests and set a ready flag;
   a host proxy thread posts pre-built WQEs (spins only while a step is in
   flight, parks on a futex when idle); replies land GPU-direct with a
   NIC-written completion flag and the reduce waits on the device; whole step
   in one CUDA graph. Order: DeepSeek V4 → V4.1 (adopt SparkIntake/GPU landing,
   device-side replica ownership) → GLM 5.3, GLM Flash, MiMo, Qwen. Then the
   Spark worker loop, then IBGDA (GPU rings the NIC doorbell) to drop the
   proxy. Projection (V4.1, `ae91c6a`): exchange alone C1 +5–7% / C4 +3–6%;
   with whole-step graphs C1 +10–15% / C4 +8–12%.
   State at 057be61 (WIP, all behind env switches, default byte-identical):
   decision B (retrofit V4.1; both engines share one device exchange,
   `SparkDeviceLane`, idle 0.6% of a core, ~5 µs wake). V4 Flash
   `CUTEAFD_SPARK_DEVICE=1`: decode 13.2 → 13.1 ms, verify 4.7 → 4.5 ms.
   V4.1 `CUTEAFD_V41_DEVICE=1` (one lane device-ordered): 2 RTX C1 185.5 →
   192.9 (+4%), C4 flat; 1 RTX +1%; greedy byte-identical with the fixed draft
   policy. Spark worker host cost ~20 of ~544 µs (GB10 expert kernel ~510 µs:
   TP6 is the bigger lever); worker now parks after 5 ms idle. Next: test the
   built-but-unrun GPU-direct receive (`CUTEAFD_SPARK_WRITE=1`, `v41-ab2.sh
   v41-2rtx-w.config 2rtx-w 0 1 1+write`); fix corruption with both lanes
   device-ordered (`CUTEAFD_V41_DEVICE_LANES=1`, 0/10 consistent C4); whole-step
   graphs (D4); recheck head split; full parity. IBGDA is possible
   (ConnectX-7 fw 28.43/28.45) but needs `PeerMappingOverride=1` on raptor.
   Gotchas: build with `build-coord.sh`/`build-spark.sh` in
   `~/.cache/cuteafd/builds/v41-device` (Spark build on moa as root; artifacts
   relay via raptor); use the fixed draft policy for byte-exact A/B;
   `chain::settle` must stay a host wait.
2. **Whole-step graphs** — MiMo's per-layer segments are merged and opt-in
   (`DECODE_GRAPHS=on`, [`work/mimo-graphs`](https://github.com/tpurtell/cuteafd/tree/work/mimo-graphs)); flat today,
   they pay once item 1 removes the host hops. Same for every family.
3. **V4.1 step wins** (from the critical-path note): device-side draft
   acceptance (~0.8 ms host gap per round, up to +3%); FP8 target and draft
   heads (2 × ~450 µs per round); one host thread serves both lanes (26–43% of
   wall time in CUDA calls) — item 1 removes most of it.
4. **Model-specific issues found** (fix in v1, not essential for v0):
   - GLM 5.3 Flash (likely GLM 5.3): a JSON-schema request whose grammar
     accepts the stop token keeps decoding; xgrammar `fill_bitmask` then fails
     and the whole batch fails. Fixed in `f3c7505`: every compiler stop token
     terminates a speculative grammar proposal before another matcher call.
   - MiMo V2.6 Pro: two-lane prefill runs only without the head split, so 8K
     prefill is slower on 2 RTX (4.79 s) than on 1 RTX (3.18 s). Fixed in
     `bf1a061`: both head-split GPUs pipeline the lanes; both GPUs' KV bytes,
     final logits and greedy continuation match the serial reference exactly.
   - V4.1 on 1 RTX: startup is serial (Sparks load all 40 layers at ~0.4 GB/s
     each, ~205 s, including 5 the RTX holds; then the coordinator). Start the
     coordinator first, skip RTX-held layers on the Sparks, speed up the Spark
     layer load. 2 RTX: 108 s. `be7049f` qualifies coordinator-first auto
     placement and the worker layer boundary; Spark read throughput remains open.
   - DeepSeek V4 Flash: the native expert format refuses 2 Sparks (min config
     needs 4); V4.1 TP3 fits per `cuteafd plan` but is unqualified.
   - Benchmarks: reasoning-effort panel re-run after the pool back-off fix;
     turn-end cache check is informational (greedy non-repeat); code sandbox
     network isolation is fixed in `fd74aaf` (required network/PID namespaces,
     unavailable rather than unisolated execution); tool-eval-bench reaches images with the next
     `./build.sh`.
   - From the v0 Release smoke matrix (10 of 22 cards fail the gate;
     logs in `~/.cache/cuteafd/builds/v0/kit/smoke-state/`):
     a. Forced tool calls: GLM 5.3 EXL3 (min, max) crashes the coordinator
        ("matcher terminated after accepting the stop token"); GLM 5.3 Flash
        EXL3 max, tr3 4bpw min/max and MiMo V2.6 Pro min/max abort the stream
        mid-response. Same grammar/matcher path: stop when the grammar accepts
        the stop token, never fail the batch. Stop-token handling is fixed in
        `f3c7505`; model smoke cards still need their own reruns.
        GLM Flash tr3 forced-tool serving passes under the required sandbox;
        `0a441b9` also fixes a narrow-prefill workspace admission failure.
     b. A worker failure mid-stream drops the SSE connection with no error
        event (all families). Fixed in `f3c7505`: one structured error event,
        preserving the backend cause, with no successful terminal event.
        A private MiMo scheduler follow-up now reports a fatal cause to other
        accepted active, prefill, KV-deferred and already-queued requests too,
        while excluding completed responses before fallible context updates.
        Request-local tokenization/first-token/grammar failures retain their
        cause. Actual two-RTX/TP6 scheduler faults now pass through raw channels
        and loopback HTTP: active, partially prefilled, KV-deferred and queued
        requests receive the same primary error exactly once, without successful
        completion. Already-finished requests remain complete after a real
        drafter context-update failure. The old-code negative control reproduces
        the lost errors with the same actual request frontiers. Reserved
        HTTP image-preparation permits and startup errors are outside this
        native-request follow-up.
        Private MiMo terminal ownership fixes also cover failures after engine
        creation, including a partially installed peer stream. Focused startup
        tests and real connected-RDMA endpoint faults pass: successful teardown
        retires registrations before buffers; failed teardown retains native
        module and storage ownership. Full MiMo two-RTX/TP6 owner faults now
        pass, including actual pending expert waves, complete SparkLink Drop,
        failed QP/publication/intake drains and post-target sampler failure.
        Successful retirement releases owners only after draining; unproved
        completion retains their storage and native module. Active grammar and
        host-copy fault coverage and shared V4.1 parity remain open.
        A private common FP8 packing fix retains the native module and pinned
        staging after an unprovable drain. Directed CPU module/packing fixtures
        pass for failure retention and normal cleanup. Private MiMo loader fixes drain
        row dequantization and pitched projection copies even after a launch
        or later tensor error, retaining storage and the native module when
        completion cannot be proved. CPU native fault injection detects all
        five original failure paths and passes with the fix. Nine real queued-CUDA
        cases now pass, including GPU1 peer work still pending across native-owner
        Drop and a later missing scale-shard read. The peer gate exposed a blocking
        free of an earlier GPU0 output; completed split outputs and earlier device
        and pinned owners now survive an unproved drain. Healthy retirement releases
        those owners and closes the actual native handle once; quarantine does neither.
     c. Speculation not lossless: V4 Pro EXL3 K2 dSpark diverges at token 4
        (1.95 nat), C4 ≠ C1 at token 15; GLM 5.3 Flash tr3 DFlash2 0.84 nat;
        GLM 5.3 EXL3 0.57 nat. Suspect multi-row verify numerics/state.
        `3010e1c` adds strict GLM Flash rejected-suffix causality, committed
        state and continuation checks. Those checks pass. The first numerical
        difference is split-dependent rounding of normalized BF16 MLA partials;
        EXL3's narrow K128 accumulation adds drift relative to K64 decode.
        GLM Flash decode retains MLA partials in FP32 and selects the K64 EXL3
        specialization for 2–16 live rows, admitting its workspace before weights.
        The full nine-row quality gate passes, with exact rejected-suffix,
        committed-state and continuation checks. This reduces numerical drift;
        serial/wide byte equality and C1/C4 batch invariance remain open.
        The serving lossless panel passes its near-tie criterion, not byte equality.
        `16cf99d` separately makes no-speculation requests skip actual GLM
        Flash neural drafter forwards; this does not change verify numerics.
     d. Batch invariance: C4 ≠ C1 greedy on V4 Pro, GLM 5.3, GLM 5.3 Flash.
     e. NVFP4 local experts on one RTX (GLM 5.3 Flash, Qwen 3.8): the severe
        slowdown came from implicit bounded paging, not a slow SM120 kernel.
        Fixed in `4c02f2f`: local experts are fully resident by default,
        admitted with scratch before allocation; paging requires an explicit
        window. Qwen NVFP4 A/B logits are byte-exact. GLM Flash's local expert
        set does not fit one RTX and now reports the admission failure.
     f. GLM 5.3 Flash and Qwen ignore `RTX_GPUS=2` (no head split), so their
        max layout is 1 RTX + 4 Sparks. `7d54499` rejects unsupported explicit
        two-GPU layouts before launch; real head splits remain open.
     g. Qwen 3.8 EXL3: 84 tok/s with 4 Sparks vs 261 on one RTX alone.
     h. Prefill gets worse with more hardware: V4 Pro min 879 tok/s (9.2 s
        TTFT) vs 2,438 max; MiMo Flash max 2,899 vs min 5,877; MiMo Pro max
        1,754 vs min 2,741 (two-lane prefill off under the head split).
        Private MiMo paired prefill now passes full-model checkpoint-mode
        correctness on one and two RTX with TP6: raw KV, taps, draft context,
        logits, proposals and fixed continuations match serial execution.
        Real scheduler/cache cases and cancellation after Spark dispatch also
        pass without late captures or tracked allocation growth. The same
        full-model matrix also passes with the single-copy FP8 bundle on both
        layouts. Clean-binary throughput/latency gates now pass three
        interleaved serial/paired comparisons on each reference layout,
        preserving C1 outputs and steady graph captures. The measured cohort
        includes cold request prefills; it is not sustained C16 decode.
        HTTP transport and MTP pairing are outside this gate.
     i. V4 / V4.1 turn-end prefix-cache restore not byte-exact (reported,
        not gated). V4 Flash also differs across repeated uncached solo
        prefills: captured inputs and coordinator reduction are exact, while
        Spark expert outputs vary with FP32 atomic arrival order. A private
        ordered reducer removes that component's repeat drift but is too slow
        to promote. Its lower-traffic serial-slice successor now passes the
        captured-input component gate with exact repeat outputs and wins its
        interleaved component timing. Its compiled exports now pass mixed-size
        graph replay and scratch poisoning. The private full-model candidate
        repeats logits exactly and passes the existing official-reference
        golden with slightly lower KL/NLL. Prefix restores across the first
        physical-page boundary also match logits, paged KV, window/compressor
        state and continuation exactly on the one-RTX reference layout.
        Serving performance and parity still need qualification before any
        native pin or default change; admission changes must not hide this
        baseline defect.
     j. MiMo V2 Flash fidelity is the weakest that passes (KL 0.10, top-1 82%).
        `73dfbbe` packages opt-in same-pin BF16 expert-input siblings;
        `bbd9a6b` preflights their capacity and arithmetic contract before launch.
        BF16 decode improves the diagnostic fidelity probe; serving defaults
        stay FP8 pending tool/agentic, batch/state and reference-layout gates.
     k. Qwen 3.8 FP8 has no Spark expert package (173 GB, no one-RTX fit).
     l. Generic-family startup did not enforce the compiled index context
        extent, and `HOST_CACHE_BYTES=auto` failed byte parsing. GLM, GLM
        Flash and Qwen now retain the manifest extent and reject unsupported
        requested contexts before engine allocation, naming the exporter
        setting. Automatic retained-prefix host budgets account for each
        family's page/mark slabs and are capped by live host/cgroup memory
        after headroom; disabled retention or unavailable rank-copy support
        allocates no host tier. Rank-aware MiMo host restores now pass the
        full-model two-RTX gate: evict device snapshots, overwrite both ranks'
        KV/rings/marks, promote from host, then compare restored storage,
        retained logits, every suffix layer and greedy continuation exactly.
        Copy engines retain the actual device allocations until all owning
        streams drain. Host retention is available with an explicit bounded
        quota; adopting the common default remains part of planner integration.
     m. Cancelled scoped stage chains could release borrowed staging while
        queued CUDA work still used it. The scope now drains pending work
        before dropping its future, preserving the caller GPU and thread-local
        scope. Actual CUDA cancellation/unwind and immediate reuse checks pass,
        with the unchanged implementation failing the negative control.
        Combined host-copy/cancellation changes pass repeated V4.1 parity on
        one and two RTX GPUs; completed and unpolled scopes retain their
        existing explicit-drain behavior. Closeout also exposed a separate
        inherited V4.1 cold-prefill defect: large index-selection and
        attention-query graph entries are evicted between encoder and replay
        shapes, causing repeated request-time captures despite warmup. Those
        sources are unchanged in the current batch. Preserve the failed
        zero-capture evidence, require zero warmed decode captures and bound
        candidate prefill captures by the identical-config baseline. Fixing
        these cache lifetimes remains a follow-up, outside this closeout.
     n. Generic families rejected temporarily exhausted KV pools even when
        an active request would soon release enough pages. A bounded FIFO
        waiter now returns borrowed state slots, retries after page/reference
        release, and drops cancelled requests. Impossible admissions still
        fail. MiMo Pro passes the deliberately tiny-pool full-model gate on
        one RTX and on two RTX with host-prefix promotion: overlapping
        completions and complete sampled vocabulary rows match solo runs
        exactly. Disabling the waiter produces the expected rejection.
5. **Spark expert kernels**: MiMo V2.6 Pro TP6 prefill is Spark-bound (~35 of
   ~42 ms per layer); GLM 5.3 verify is bound by distinct expert reads; NVFP4
   W4A16 GB10 prefill (14.3 vs EXL3 9.1 ms/layer TP4). Pro's installed SM121
   `fp8-mimop/tp6` package at `5b9c135` already streams MXFP8 × MXFP4 gate/up
   above 640 live rows: the host dispatch and embedded CUDA binary contain
   the 640-row branch and `QMMA.SF.16832.F32.E4M3.E2M1.E8`. Its down projection
   still consumes BF16 SwiGLU output and widens MXFP4 weights for BF16 MMAs.
   The first private A8-down candidate fails the full-model added-error gate
   despite its component speedup; it is not deployed. Improve its accuracy
   before further performance promotion; retain the existing gate/up.
   MiMo's short-request C16 serving campaign is dominated by serialized cold
   prefills; decode already batches active sequences. Evaluate independent
   requests on the two existing prefill lanes before changing weight precision
   further. Preserve separate placements, rings, complete drafter taps and
   both first-token outputs; keep shared KV-widening scratch consumers ordered
   and drain both expert waves on failure. Gate against serial KV/context and
   continuation exactness, including mixed cached/uncached prefixes, then
   measure emitted throughput and per-step active rows on both RTX layouts.
   A private candidate pairs adjacent 1–2047-row requests without changing
   their kernel row partitions. Larger chunks and MTP keep the existing
   path. Queue, placement and memory-admission checks pass in the composed
   workspace. Full-model exactness passes in checkpoint and single-copy FP8
   modes on both reference layouts; clean-binary emitted throughput and
   per-request latency also pass three interleaved comparisons per layout.
6. **RTX 5090 audit and claim**: hard-coded `4*188` grid clamps and the
   per-tensor FP8 GEMM grid sized for 188 SMs; one SM120 build must serve both.
   `6d4ea7a` derives expert quantizer grids from each engine's GPU and removes
   the CLI's fixed SM default. Card-specific AOT/ABI guards remain to qualify.
7. **Phase 6 placement planner** (incl. cold components such as the vision
   encoder on a Spark) and **multimodal input** (official encoders only).
   **Joint serving capacity policy** (TJ): default C16, with 20 active
   SWA/front-layer state slots to tolerate a small burst. Target a common
   2,097,152-token logical GPU KV pool, with bounded host-prefix overflow for
   fast session resume. Larger pools are requested from the planner at launch.
   Report the target and feasible capacity separately when it cannot fit.
   Reserve KV and runtime storage before maximizing expert onboarding. Keep checkpoint
   context, compiled index extent and effective serving context distinct.
   Budget each physical GPU to 97% of total minus pre-existing non-engine
   usage, reserving resident weights, all workspace/replay/graph/transport
   storage, optional drafters, active state and exact-prefix marks before
   allocating the aligned shared pool. GLM KV is replicated under a head
   split; MiMo KV heads are partitioned, so aggregate GPU bytes cannot be
   used as interchangeable capacity. One resolved plan must drive both
   `cuteafd plan` and runtime startup; preserve explicit benchmark overrides.
   Retained host-prefix storage has its own bounded budget and exact restore
   gate; it does not extend active GPU KV capacity. Generic admission deferral
   now passes the MiMo pressure/host-restore gate; active KV paging/offload
   remains separate work.
   MiMo's private runtime admission candidate rejects an impossible one-RTX
   request before module/weight loading and preserves explicit small-pool
   outputs. Its two-RTX startup check found that capturing all decode shapes
   exceeds the provisional runtime reservation. Account for the measured
   module, head-initialization and graph costs before promoting this planner.
8. **NVFP4 follow-ups**: native per-tensor FP8 decode with static scales.
   **Revisit W4A4 for `nvidia/DeepSeek-V4.1-Flash-NVFP4`** (TJ): V4.1's own
   NVFP4 path keeps the ds41rt 44-slot W4A4 family opt-in because ds41rt
   measured it slower, which is implausible for FP4 MMAs on Blackwell and
   contradicts the checkpoint's declared numerics (W4A4, static input_scale).
   Honor the checkpoint: profile why ds41rt's W4A4 lost (likely activation
   quant overhead, tile shapes, or decode rows taking the W4A4 path), port the
   shared fp8_moe W4A4 route (fused gate/up + SwiGLU + FP4 quant, large-row
   threshold, W4A16 decode rows) to V4.1 if it wins, and make W4A4 the default
   for that checkpoint. Gate: V4.1 golden/KL vs the official FP8 reference,
   8K prefill and C1/C4 vs the current NVFP4 default.
   **W4A4 decode/verify rows** (TJ): decode rows (1–16, incl. speculative
   verify) run W4A16 even on W4A4 checkpoints. Measure W4A4 decode with the
   activation quant fused into the GEMV/MMA prologue on one NVFP4 model (C1
   step, KL); bandwidth-bound either way, so expect parity — if so, make W4A4
   decode the default for checkpoints that declare it (one numerics path from
   prefill through verify).
9. **Activation precision policy** (TJ, 2026-10-02): converge on **A8
   wherever quality holds** (FP8/MXFP8 activations on tensor cores, the speed
   lever for prefill and wide verify) and **A4 only where the checkpoint
   declares it** (NVIDIA ModelOpt NVFP4). Every A8 switch is gated on golden
   NLL/KL (≤0.005 nat) plus a tool-eval/agentic check, per model.
   - EXL3 × A8: EXL3 trellis experts (V4 Pro, GLM 5.3, GLM Flash, Qwen) run
     A16 today. brandonmusic had unmerged MXFP8 EXL3 WIP; check upstream b12x
     first, else build an EXL3 decode-to-FP8 tile path with MXFP8/FP8
     activations in the fork.
   - MXFP4 experts: V4.1 already W4A8; MiMo V2.6 Pro's SM121 large-row
     gate/up already uses MXFP8 × MXFP4 above 640 live rows. Extend A8 to Pro's
     BF16-input down projection, retaining the existing small-row route and
     checkpoint weights. Gate added activation error separately from the
     established model/reference floor, then qualify tool/agentic behavior.
   - FP8 experts: extend W8A8 (MiMo GB10 gate/up) to the down projection and
     to RTX-local experts where KL allows (Qwen FP8 local was +0.024: needs
     finer activation scales).
10. **Resident weight representations** (TJ, 2026-10-02): close loader
    shortcuts that permanently widen compact checkpoint tensors to BF16.
    Keep exactly one resident BF16 or FP8 representation per weight set.
    A duplicate is allowed only when genuinely tiny or justified by an
    exceptionally large measured performance benefit; name its bytes and
    measured justification explicitly. Audit every target and drafter
    family; report source dtype, resident dtype/layout, bytes and the consumer
    that requires each copy. Temporary loading buffers and in-kernel
    dequantization are separate from persistent weight storage.
    MiMo Pro's target QKV and dense FFN already retain checkpoint FP8; its
    target o_proj, embedding and head are BF16 in the checkpoint. The target
    keeps additional FP8 o_proj/head copies, and DFlash retains BF16 weights
    plus FP8 copies with a BF16 fallback above its skinny-row limit. MiMo's
    generic BF16 operand loader can also widen FP8 o_proj/head sources.
    Cover prefill, decode, batched verify, context updates and graph/replay
    paths before releasing a required representation. Prefer native compact
    kernels or bounded staging, selecting the representation at startup.
    Partially split matrices must not retain overlapping BF16/FP8 rows.
    Do not silently change target checkpoint precision to save
    memory: added target quantization needs its own golden NLL/KL (<=0.005 nat)
    and tool/agentic gates. Preserve each checkpoint tensor's precision by
    default (TJ, 2026-10-03), including BF16 drafter O projections and other
    BF16 weights; keep native FP8 compact. Additional weight quantization
    should arrive in a checkpoint. A reasonable calibration-free conversion
    may be an explicit convenience option. Model-specific defaults require
    measured quality/performance evidence and explicit approval; MiMo V2.6
    Pro's approved exception is recorded below.
    Finish evaluating the existing FP8 feature as that optional path. Its
    drafter precision is judged by net emitted tokens/s after target
    verification and actual memory use, including context updates, drafting
    cost and proposal acceptance. Focus optimization on kernels and plumbing;
    use A8 activations where the quality/performance gates support them,
    independently of the checkpoint's weight precision. Drafter proposal KL
    is diagnostic, not a target-quality
    threshold or a standalone rejection criterion. Target verification,
    final-output correctness and cache-state contracts remain mandatory.
    Eliminate wasted duplicate representations whichever precision wins.
    Compare separately loaded BF16-only and FP8-only candidates first
    (TJ, 2026-10-03), with no dual-resident control or runtime precision
    switching. Target and drafter share the selected vocabulary head;
    checkpoint-native FP8 QKV/FFN stay compact in both candidates. Cover
    wide rows without a second weight representation. The historical
    legacy/no-speculation comparison found divergent concurrent output
    (C1 matched); preserve that evidence, but qualify target quality and
    verifier/cache-state correctness directly on the two single-copy paths.
    The private FP8 feature now passes original-reference quality, exact
    prefix restoration, fixed-history repeat/graph/causal-anchor checks and
    tool serving on both MiMo Pro reference layouts. Three interleaved
    checkpoint/FP8 serving pairs now pass on each layout, including uncached
    8K prompt latency, readiness, resident memory and C1/C16 emitted throughput.
    Transport selection is identical and decode completion logs contain no
    late graph captures. TJ approved the measured single-copy FP8 bundle as
    the default specifically for MiMo V2.6 Pro (2026-10-03): target O
    projections, the shared vocabulary head and DFlash weights. The scoped
    selection is implemented with explicit checkpoint/BF16 overrides, preserving
    checkpoint precision for other models and other tensors. On both reference
    layouts, the automatic default matches the saved qualified FP8 target logits
    and explicit-setting DFlash serving output exactly. Existing
    measurements do not automatically qualify a larger KV pool. Repeated
    concurrent serving still changes some responses in both representations;
    the fixed-history checks do not prove all serving histories correct.
    The private checkpoint-driven default and shared per-rank prefill KV
    scratch now match explicit BF16 target quality, logits, prefix restoration
    and fixed-history state on both reference layouts. Actual two-lane prefill
    matches serial prefill and continuation exactly on both layouts; the
    earlier smaller-chunk gate exercised only the serial path. Native
    promotion still needs V4.1 parity. The bounded real scheduler and terminal
    ownership gates pass; complete concurrent-history correctness remains open.
    The source audit also finds implicit BF16 quantization and duplicate
    matrices in Qwen attention/MTP, shared GLM/GLM Flash DFlash, and the
    GLM Flash launcher's default KDA path. Correct the checkpoint-preserving
    defaults, then replace optional dual-format paths with immutable compact
    consumers across every row shape. GLM target projection selection also
    quantizes BF16 sources implicitly; mixed source groups need per-projection
    dispatch and BF16 dense-FFN exports. GLM target native-FP8 head/index
    operands are widened persistently today: add compact consumers or report
    the missing format before allocation. GLM Flash's loaders have similar
    unsupported-format gaps, but the inspected qualified EXL3 and official
    FP8 checkpoints store their head, indexer, routers and KDA weights in BF16;
    those default inputs are not widened. Private GLM Flash KDA checkpoint
    defaults/header guards and shared DFlash single-copy loaders are composed
    with the MiMo changes; workspace and script checks pass. Shared GLM/GLM
    Flash drafter gates pass exact same-shape replay, ring-wrap/tail checks,
    batch-state isolation and physical weight ownership. Both families now
    have original-target-conditioned drafter comparisons. GLM Flash's exact
    four-stream fold and original 64/1097-token anchors pass; the 2305-token
    wrap case remains an explicitly cyclic state fixture. These component
    checks do not qualify full-target quality or emitted throughput. Optional
    FP8 proposal drift is smaller with real conditioning than in the earlier
    synthetic long-context case; checkpoint BF16 remains the default.
    GLM target precision admission also passes loader/planner/exporter checks;
    its compact index-key and BF16 dense consumers still need native gates.
    Private GLM Flash direct-CLI guards now reject BF16 block inputs before
    native loading instead of silently quantizing them. Actual checkpoint
    headers validate the qualified EXL3-primary/official-FP8-side route;
    standalone planner diagnostics now match this policy, with named missing
    BF16 consumers and supported native block-FP8 inputs. The planner does
    not yet model a secondary FP8 snapshot.
    Private Qwen defaults preserve checkpoint BF16 projections and share the
    target head with MTP. Legacy duplicate-storage options reject before
    native loading until compact all-row/shared-head consumers exist. These
    changes pass composed workspace/script checks. The one-RTX local EXL3
    smoke now matches explicit BF16 settings exactly on a saved 64-token
    prefill/one-token original-reference probe and actual fixed-width MTP
    serving. The GLM Flash tr3/official-FP8-side smoke also matches explicit
    checkpoint settings exactly across the full 1524-token target reference,
    actual DFlash code serving and a forced tool call on one RTX plus two
    Sparks. Readiness is recorded; these bounded gates do not qualify all
    layouts, formats, concurrent histories or emitted throughput. Qwen NVFP4
    with checkpoint-FP8 MTP experts remains explicitly unsupported without
    its separate expert package. GLM Flash direct-CLI and
    launcher guards reject duplicate-storage options before native loading
    or worker launch; the composed loader, planner and option changes pass
    workspace/script checks. Compact single-copy KDA/head consumers and other
    persistent FP8 widening remain open.
    GLM head admission now reads indexed checkpoint headers independently of
    the routed-expert catalog. The
    audited DeepSeek V4 target/dSpark paths preserve checkpoint weight values;
    their expanded scale metadata is not a second weight representation.
    Check exactness when arithmetic is preserved,
    readiness, C1/C16 decode and 8K prefill on both reference layouts; include
    every surviving copy and expanded scale layout in admission.
11. **Parked**: Spark-side reduce-scatter ([`work/spark-reduce`](https://github.com/tpurtell/cuteafd/tree/work/spark-reduce),
   +3% one rail, +9–12% two rails at 200G); split intake
   ([`work/split-intake`](https://github.com/tpurtell/cuteafd/tree/work/split-intake), slower). Revisit only on new evidence.
12. **Housekeeping**: prune agent test images on raptor; delete
    `~/.cache/cuteafd/builds/{n10-rel,bisect-rel}` on ostrich (root); refresh
    the inherited script-test failure ids in AGENTS.md. The stale fixture and
    sibling-checkout failures are fixed in `37adfc4`; current failing ids are empty.

## Backlog (lowest priority: only when nothing planned is left)

- Qwen 3.8 Flash Next NVFP4 without Sparks, competitive with vLLM on one
  RTX PRO 6000 (localmaxxing card, 2026-10: batch 1, MTP, 2,821 in / 2,048
  out: 373 tok/s output, 12,078 tok/s prefill, 234 ms TTFT; ours today ~100-130
  C1 with EXL3 local experts, NVFP4 W4A4 8K prefill ~9.5K tok/s). Outside the
  usual scope; reference configs for it in spirit: min = simulated RTX 5090
  (32 GB) + 1 Spark, max = 2x RTX with no Sparks.

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
