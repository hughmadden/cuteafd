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
- Dense NVFP4/per-tensor-FP8 parts dequantize to BF16 at load today;
  item 10 of the v1 plan replaces that with compact consumers.
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

**Memory audit and planner core (2026-10-03, `work/v1-memory`).** Every
device, pinned and RDMA allocation now goes through a process-wide ledger
(`cuteafd_ffi::memory_ledger`: thread-local category scopes, the checkpoint
tensor and resident format being uploaded; `cuteafd::memory` log reports;
`scripts/bench/memory-audit.py` tabulates and `--compare`s against the
planner). One launch per config (codex/v1 tree + ledger, after an 8K prefill,
a C4 and a C1 request; default pools). GiB per device:

| config | device | weights | emb | drafter | KV | marks | workspace | exchange | experts | runtime | used | free |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiMo V2.6 Pro 2 RTX + 6 | GPU0 | 10.72 | 1.75 | 3.21 | 2.36 | 0.77 | 2.49 | 0.38 | | 1.37 | 23.0 | 71.9 |
| | GPU1 | 9.52 | | | 2.34 | 0.77 | 1.20 | 0.38 | | 1.00 | 15.2 | 79.7 |
| | Spark (TP6) | | | | | | 0.52 | 1.15 rings | 92.79 | 7.5 OS + 11.0 cache | 117.6 | 4.0 |
| MiMo V2.6 Pro 1 RTX + 6 | GPU0 | 20.24 | 1.75 | 3.21 | 4.70 | 1.54 | 2.90 | | | 1.23 | 35.6 | 59.4 |
| GLM 5.3 K4 2 RTX + 6 | GPU0 | 10.48 | 1.77 | 5.88 | 13.18 | | 5.59 | 0.56 | | 1.72 | 39.2 | 55.8 |
| | GPU1 | 8.49 | | | 13.18 | | 4.22 | 0.56 | | 1.26 | 27.7 | 67.3 |
| | Spark (TP6) | | | | | | 1.9 | 2.29 rings | 63.45 | 7.7 OS + 7.1 cache | 85.4 | 36.2 |
| GLM 5.3 K4 1 RTX + 4 | GPU0 | 17.56 | 1.77 | 5.88 | 13.18 | | 6.51 | | | 1.55 | 46.5 | 48.5 |
| | Spark (TP4) | | | | | | 1.9 | 2.29 rings | 84.59 | 7.7 OS + 3.4 cache | 103.3 | 18.3 |
| GLM 5.3 Flash 1 RTX + 2 / + 4 | GPU0 | 13.07 | 1.18 | 3.24 | 2.12 | 2.47 | 4.72 | | | 1.24 | 28.0 | 66.9 |
| | Spark (TP2 / TP4) | | | | | | 0.56 | 0.77 rings | 58.3 / 29.4 | 7.3 OS | 78.1 / 43.6 | 43.6 / 78.0 |
| V4.1 Flash 2 RTX + 4 | GPU0 / GPU1 | 5.16 / 3.91 | in weights | 0 / 8.12 | 7.34 / 4.90 | 0.1 | 9.48 / 5.70 | | ~67 each (20 layers TP2) | 2.0 / 1.9 | 93.1 / 93.5 | 1.8 / 1.5 |
| | Spark (TP4) | | | | | | 0.21 | 0.25 rings | 37.35 (20 layers) | 7.3 OS | 50.6 | 71.1 |
| V4.1 Flash 1 RTX + 4 | GPU0 | 9.29 | in weights | 8.21 | 15.60 | 0.1 | 21.08 | | 33.94 (5 layers) | 2.83 | 93.1 | 1.9 |
| | Spark (TP4) | | | | | | 0.21 | 0.48 rings | 65.37 (35 layers) | 7.4 OS | 79.2 | 42.4 |
| V4 Flash 2 RTX + 4 | GPU0 / GPU1 | 5.38 / 3.74 | 0.99 | | 2.08 / 1.96 | 1.0 / 1.0 | 4.14 / 3.55 | 0.25 / 0.31 | 73.47 / 0 | 1.0 / 0.9 | 88.3 / 11.4 | 6.7 / 83.6 |
| Qwen 3.8 EXL3 1 RTX | GPU0 | 8.21 | 1.18 | | 1.93 | 1.94 | 1.20 | | 62.61 | 1.41 | 78.5 | 16.5 |

Findings, against the suspects: no tensor is resident in two formats in any
family after codex/v1 (the ledger checks every upload by full tensor name);
no load-time conversion uploads at source size (GLM NVFP4 was the one case);
V4.1 Sparks hold only the remote layers (runtime placement handoff: 35 of 40
at one RTX, 20 at two). The waste is elsewhere:
1. Idle coordinator memory: the generic families' fixed pools (MiMo 131072,
   GLM 262144, GLM Flash 65536 tokens) leave 48–80 GiB of every GPU unused,
   while V4.1 fills its GPUs with expert layers and KV. Fixed for MiMo
   (`POOL_TOKENS=auto`, now the run-family default, through the codex capacity
   contract: 131072 -> 2,097,152 tokens, GPU0 23.0 -> 49.4 GiB, C1/C4/8K prefill
   unchanged) and GLM 5.3 Flash (default auto via `planned_pool_tokens`: 65536 ->
   2,097,152 tokens, 44 GiB still free, unchanged speed). GLM 5.3 takes
   `POOL_TOKENS=auto` (262144 -> 1,292,672 tokens, 65 GiB KV per GPU) but keeps
   its fixed default until item 5 is bounded (2.9 GiB left on GPU0 after a bench).
2. Spark page cache: ~10 GiB of the checkpoint stays cached per Spark after
   loading (CUDA free 4.0 of 121.6 GiB on MiMo Pro TP6); the worker's own
   fadvise does not reach sparknest's passthrough pages. Fixed: launchers
   drop Spark caches once every rank is resident (+9.5 GiB CUDA free).
3. Retained load staging: the copy_h2d pinned buffer stayed at the largest
   upload (0.45 GiB per Spark on MiMo, 1.77 GiB pinned on raptor for GLM,
   1.0 for V4 Flash). Fixed: released after loading.
4. Head-split workspaces sized for all heads: GLM 5.3 6.39/5.02 -> 5.59/4.22
   GiB, V4 Flash rank 1 3.80 -> 3.55; V4 Flash prefill logits for 4096 rows
   (2.1 GB) -> 64 rows with chunked golden downloads: GPU0 6.09 -> 4.14 GiB.
   DeepSeek V4 golden output identical; GLM 5.3 golden NLL 2.4686 in both (TP6 + head split; KL 0.03444, top-1 91.54%).
   GLM 5.3 2 RTX + 6 Sparks, 4 interleaved launches per arm: C1 54.3 vs 55.6
   (-2.4%, t=-1.6), C4 87.3 vs 88.2 (-1.1%, t=-1.1), 8K prefill TTFT medians
   3.454-3.58 vs 3.45-3.52 s: within run-to-run noise (C1 spans 50-58 in both).
5. Graph executables (untracked): GLM 5.3 grows from 0.89 to 2.62 GiB on
   GPU0 over one C1/C4 + prefill bench and keeps rising (graphs per layer x
   exact row count x table width); an auto pool sized with a 0.6 GiB graph
   reserve hit cudaGraphInstantiate OOM and poisoned the context. The planner
   reserves 3 GiB per GPU for GLM until the cache is bounded (bucket rows or
   cap entries; MiMo's codex plan bounds its own at 616/566 MiB).
Inventory for the planner (not fixed; bytes per device):
- Spark slices padded to the widest 128-row block: MiMo V2.6 Pro TP6 stores
  384 of 352/320 rows (7.7 GiB on ranks 0-3, 15.5 on 4-5, 62 GiB cluster);
  GLM 5.3 EXL3 TP6 384 of 341 (~7 GiB/rank); V4.1 TP4 640 of 576 (~3.7 GiB
  at 20 layers, 6.5 at 35). Uneven whole-block slices (384,384,384,384,256,
  256) free ranks 4-5 at no speed cost; kernels that run 32-row tails also
  cut the critical rank's rows 8-10% (MiMo Pro prefill is Spark-bound).
- sparknestd holds 7.2 GiB RSS on every Spark (host OS total ~13 GiB idle).
- RDMA rings: 1.15-2.31 GiB per Spark (depth 8 x 8 MiB slots per endpoint)
  and 6.9 (MiMo) / 13.7 (GLM) GiB pinned on raptor.
- GLM 5.3 head split replicates the MLA latent KV (13.18 GiB per GPU at 262K
  tokens) and 1.2 GiB of attention operands plus the indexer on GPU1.
- V4 Flash/Pro keep RTX-local expert layers on GPU0 only: GPU1 83.6 GiB free.
- V4.1 one RTX: decode and prefill target passes own complete workspaces
  (prefill pass 10.5 + backbone lanes 9.0 GiB) ~ three RTX expert layers.
- MiMo FP8 scales expanded per row twice (row and K-major): ~1 GiB.
- GLM DFlash2 drafter resident BF16 (5.88 GiB incl. 1.3 GiB buffers; the
  explicit FP8 representation is ~2.3 GiB smaller).
Planner core (S0): `cuteafd plan MODEL --layout [--rtx 1|2] [--pool-tokens N]`
lays every device out (weights by group and resident format, embedding,
drafter, KV records and state, prefix marks, workspaces, peer exchange,
runtime and graph allowance, Spark experts with padding, workspace and
rings) and sizes the pool from the tightest KV-owning GPU; MiMo weights come
from the codex resident layout, GLM Flash from its FP8-snapshot conversion.
Per-family costs are calibrated from the ledger (`plan::layout::family_costs`);
device totals at ready match the ledger within 2% (GLM 5.3 39.10/27.63 vs
39.18/27.70, MiMo Pro 22.81/14.99 vs 23.03/15.23, GLM Flash 27.55 vs 28.05), and
on the held-out auto-pool launches (MiMo Pro 49.42/41.86 vs 49.4/41.6, GLM Flash
50.93 vs 50.7, GLM 5.3 93.04/81.96 vs 92.1/80.6 with its full graph allowance).
Engines take admission from it: MiMo (capacity contract, pool 0 = auto),
GLM 5.3 and GLM 5.3 Flash (`--pool-tokens 0`). Next: V4/V4.1 and Qwen
geometry in the planner, graph-cache bounds, `placement.json` handoff (S1).
Follow-ups (2026-10-03, measurements pending in `~/.cache/cuteafd/builds/v1-memory/kit/out`):
- GLM 5.3 decode graphs bounded: steps pad to row buckets (exact to 16, then
  20..64) over a scratch page past the pool, page tables to power-of-two widths
  >= 16 pages; every shape captured at startup (2 RTX + 6 Sparks: 22,608
  graphs, 1.96 / 1.80 GiB per GPU, 5.8 s); real rows' logits byte-identical to
  unpadded steps; a failed capture runs its segment uncaptured (keeps the peer
  exchange in step). Measured vs base (2 launches each, 4 batches): C1 54.5 vs
  51.8 tok/s (no per-request captures), C4 85.8 vs 84.2, 8K prefill unchanged;
  untracked memory now flat after startup. GLM 5.3 POOL_TOKENS defaults to auto:
  262144 -> 1,292,672 tokens (65 GiB KV per GPU), 2.7 GiB left on GPU0 after the
  bench, C1 53.3 / C4 85.1 / prefill 2853 tok/s.
- V4.1 quick parity for the ledger (1 RTX + 4 Sparks, code, warm): C1 155.7 ->
  154.7, C16 1049.6 -> 1054.8, 8K prefill 6160 -> 6331 tok/s: no cost; tagging
  stays per allocation.
- Exact Spark slices: FP8/MXFP4/NVFP4 Spark packages also build tp<n>-w<width>
  layouts (ranks own whole 128-row blocks, no zero padding; MiMo V2.6 Pro TP6
  ranks 4-5 61.9 instead of 92.8 GiB); EXL3 already had them. Measured MiMo V2.6
  Pro 2 RTX + 6 Sparks, exact vs padded: rank 5 free 11.8 -> 43.2 GiB (rank 0
  unchanged), golden NLL 2.4150 -> 2.4088 (KL 0.0457 -> 0.0445; rank partials
  partition the rows differently), engine 8K prefill 3095 -> 3002 tok/s, served
  8K 2684 -> 2725, C1 63.9-69.1 -> 68.5-69.5, C4 102-108 both: neutral. Shortening the
  busiest rank (352/320 rows) needs 32-row tails in three MXFP4 kernels (fork
  master now has `work/mimo-perf`'s A8 down): the decode GEMV (`GroupedMxfp4Gemv`
  needs K % (128 x warps) for down), the BF16 stream down (`I % 128`, 128-K
  weight blocks) and the A8 stream down (128-K blocks via cp.async, u32 scale
  loads). Gate/up already tiles I in 32 rows (11 vs 12 CTA columns: -8%); down
  only gains if its last K block is predicated at 32 (TMA zero-fill covers the
  BF16 route's loads; the A8 route needs predicated cp.async), and scale rows
  of I/32 = 11 bytes need padding to 12 in the package layout. Expected: busiest
  rank -5..-8% expert time (MiMo Pro prefill is Spark-bound) and -7.7 GiB on
  ranks 0-3. V4.1 TP4 (576 -> 640)
  goes through the V4.1 packer: not done.
- V4.1 one RTX: row buffers at the live 2048-row chunk instead of the 4096 AOT
  capacity (as on two RTX), reindex selection shares the source's scratch:
  workspaces 21.08 -> 11.01 GiB, RTX expert layers 5 -> 6 (6.0 GiB still free),
  C1 154.7 -> 159.1, C16 1054.8 -> 1086.3, 8K prefill 6331 -> 6326 tok/s. Left:
  window-wave temporaries (3.5 GiB, per-layer streams; would make 7 layers) and
  engram gate sharing (0.6 GiB).

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

Status (2026-10-02, `work/v41-device`, all opt-in): D0 done (`SparkDeviceLane` proxy +
`cuteafd_host_signal`/`cuteafd_peer_wait`; proxy spins only while announced waves are
outstanding, parks otherwise: idle 0.6% of a core, wake 5 us). D1 V4 Flash
(`CUTEAFD_SPARK_DEVICE=1`): top-1 100%, decode 13.2 -> 13.1 ms/tok, 6-row verify 4.7 -> 4.5
ms/tok. D2+D3 V4.1 (`CUTEAFD_V41_DEVICE=1`: device-ordered verify passes, TP2 layers without host
waits, staging fences, remote waves on the device exchange; only while the other lane is idle):
2 RTX + 4 Sparks code C1 185.5 -> 192.9 tok/s, C4 unchanged (old path), greedy byte-identical; 1 RTX
C1 +1%. Spark worker per 5-6-row wave: kernel ~510 of the ~544 us round trip (GB10 bandwidth,
~24 experts x 5.9 MB at TP4), host ~20 us: TP6 is the bigger remote lever. Write mode
(`CUTEAFD_SPARK_WRITE=1`, NIC-written rows + flags, GPU waits; needs this branch's Spark build) is
built but not yet run. Open: both lanes device-ordered corrupt C4 (`CUTEAFD_V41_DEVICE_LANES=1`,
cause not found); whole-step graphs (D4); nsys traces of the overlay image came out empty.

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

## Release v1 scope (decided 2026-10-04)

v1 ships when these are done; everything else below moves to v1.x/v2.
- **In v1:** device-driven exchange for V4.1 and MiMo V2.6 Pro (default only
  if it beats the current default and is hang-free; otherwise opt-in);
  byte-exact prefix-cache restores at turn end for V4 / V4.1; planner core for
  every family (per-device memory layout + admission; V4, V4.1, Qwen still
  missing); all Release smoke cards green with MOPD as the MiMo Pro default
  and a refreshed README; MXFP4 32-row tails; the Spark kernel wins already
  landed; known-issue notes (NVFP4 local experts on one RTX, Qwen with Sparks).
- **Status 2026-10-04:** the device exchange is merged opt-in and hang-free
  (`CUTEAFD_V41_DEVICE=1`; +1.5% C1 on 2 RTX, flat elsewhere; MiMo/GLM
  adoption on `work/device-mimo-glm` gains nothing — their segments are
  GPU-bound). It ships opt-in in v1. Turn-end prefix restores proved exact
  (the check was wrong; fixed). V4.1 FP8 vocabulary head: single-copy gate
  in progress.
- **Cut to v1.x/v2:** whole-step graphs (D4: context-length-dependent index
  graphs, per-request pointers in graph keys, host-built per-layer metadata,
  warm re-captures) and device-side draft acceptance; deterministic
  (batch-invariant) prefill and verify;
  multimodal input (v2); `placement.json` handoff and cold-component placement;
  V4.1 NVFP4 W4A4 revisit and W4A4 decode; EXL3 × A8 (an independent SM120
  implementation is the interesting part — not a port of b12x PR #342, whose
  ShapleyMcg licence covers re-implementations made with reference to it);
  parked Spark-side reduce / split intake.
- **RTX 5090 support: Hugh** (external collaborator). Brief: one SM120 build
  serves RTX PRO 6000 (188 SMs, 96 GB) and RTX 5090 (170 SMs, 32 GB) with no
  regression on the 6000; remove SM-count assumptions (hard-coded `4*188` grid
  clamps; the per-tensor FP8 GEMM grid sized for 188 SMs; any L2-size
  assumptions); simulate a 5090 on a 6000 via the planner's device inventory
  (`cuteafd plan MODEL --layout`, 32 GB budget) and validate on real 5090s;
  start from `work/p0`, branch `work/rtx5090`, follow AGENTS.md.

## Release v1 — priority plan (2026-10-02)

Everything after v0 lands as v1. Helpers: read AGENTS.md, then pick the top
open item; each names its branch (pushed WIP) and the next step. Merge green
steps into `work/p0`; tag `v1.0.0` when the list's top half is done.
The codex/v1 line (merged 2026-10-03, `work/codex-merge`) closed several
item-4 bugs and started items 7 and 10; commit messages carry its evidence.

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
   State (2026-10-04, merged into work/p0 at db9ed89; all opt-in, default unchanged
   except the single-RTX head keeping one graph per verify width): `SparkDeviceLane`
   proxy, V4 `CUTEAFD_SPARK_DEVICE=1`, V4.1 `CUTEAFD_V41_DEVICE=1` (device-ordered
   verify passes: chained SM handoff between RTX, chained split head, engram uploads
   in the chain, per-layer staging, 60 s watchdog that logs lane sequences and exits).
   Both lanes device-ordered are consistent now (codex's attention-staging fence).
   Spark worker idle loop that waited on one connection (aa507bd) stalled device
   waves: reverted. Code case, warm, 325 W, one launch per arm: 2 RTX + 4 Sparks
   C1 190.4/192.4 → 194.6/194.1 (+1.5%), C4 flat; 1 RTX C1 −1%, C4 +0.5%: the
   exchange alone does not pay enough to be default. MiMo V2.6 Pro and GLM 5.3 on
   the exchange (+ whole-step decode graphs, `work/device-mimo-glm`): no gain
   (MiMo host C1 80.4/81.5 vs device step 77.0/79.8; GLM ~65 both) — their
   segments are GPU-bound. Write mode (`CUTEAFD_SPARK_WRITE=1`) launches but the
   pass sticks on the written flags (open). The remaining V4.1 lever is whole-step
   capture; blockers in order: index-selection shape vs context length, per-request
   pointer fingerprints (sparse/index), host-built per-layer metadata (one per-pass
   arena would remove it), 1-RTX GPU landing at 3.6 GB/s after weights load.
   FP8 draft head (`CUTEAFD_V41_FP8_HEAD=draft`, E4M3 copy, W8A16 GEMV 882 → 416
   µs per 129280-row head): parity, 3 interleaved sessions per arm, base → fp8d:
   2 RTX C1 183.6 → 186.4 (1.016), C4 489.0 → 512.0 (1.047), C16 1413.7 → 1442.2
   (1.020), weighted decode 0.997; 1 RTX C1 1.014, C4 1.039, C16 1.087, decode
   1.012. Proposed as the default (outputs unchanged; drafts only). Kit:
   `~/.cache/cuteafd/builds/v41-device` (STATUS.md, build-coord.sh/build-spark.sh,
   v41-ab3.sh, v41-c4.sh, run-parity.sh).
2. **Whole-step graphs** — MiMo's per-layer segments are merged and opt-in
   (`DECODE_GRAPHS=on`, [`work/mimo-graphs`](https://github.com/tpurtell/cuteafd/tree/work/mimo-graphs)); flat today,
   they pay once item 1 removes the host hops. Same for every family.
3. **V4.1 step wins** (from the critical-path note): device-side draft
   acceptance (~0.8 ms host gap per round, up to +3%); FP8 target head
   (draft head done, see item 1; the target head changes outputs and needs a
   quality gate); one host thread serves both lanes (26–43% of
   wall time in CUDA calls) — item 1 removes most of it.
4. **Model-specific issues found** (fix in v1, not essential for v0):
   - GLM 5.3 Flash (likely GLM 5.3): a JSON-schema request whose grammar
     accepts the stop token keeps decoding; xgrammar `fill_bitmask` then fails
     and the whole batch fails. Fixed (`f3c7505`): every stop token ends a
     speculative grammar proposal.
   - MiMo V2.6 Pro: two-lane prefill runs only without the head split, so 8K
     prefill is slower on 2 RTX (4.79 s) than on 1 RTX (3.18 s). Fixed
     (`bf1a061`): both head-split GPUs pipeline the lanes; short independent
     requests also pair on the two lanes (`5b191f2`).
   - V4.1 on 1 RTX: startup is serial (Sparks load all 40 layers at ~0.4 GB/s
     each, ~205 s, including 5 the RTX holds; then the coordinator). Start the
     coordinator first, skip RTX-held layers on the Sparks, speed up the Spark
     layer load. 2 RTX: 108 s. Coordinator-first auto placement on 1 RTX
     landed (`be7049f`); Spark layer read speed is still open.
   - DeepSeek V4 Flash: the native expert format refuses 2 Sparks (min config
     needs 4); V4.1 TP3 fits per `cuteafd plan` but is unqualified.
   - Benchmarks: reasoning-effort panel re-run after the pool back-off fix;
     turn-end cache check gates restores against their snapshot (4i); the code
     sandbox requires user/net/PID namespaces (`fd74aaf`; coordinators run
     with `docker/seccomp-code-bench.json`); tool-eval-bench reaches images
     with the next `./build.sh`.
   - From the v0 Release smoke matrix (10 of 22 cards fail the gate;
     logs in `~/.cache/cuteafd/builds/v0/kit/smoke-state/`):
     a. Forced tool calls: GLM 5.3 EXL3 (min, max) crashes the coordinator
        ("matcher terminated after accepting the stop token"); GLM 5.3 Flash
        EXL3 max, tr3 4bpw min/max and MiMo V2.6 Pro min/max abort the stream
        mid-response. Same grammar/matcher path: stop when the grammar accepts
        the stop token, never fail the batch. Fixed in `f3c7505`; rerun the
        smoke cards.
     b. A worker failure mid-stream drops the SSE connection with no error
        event (all families). Fixed in `f3c7505` (one structured error event);
        MiMo reports a fatal cause to every accepted request (`50b2608`) and
        retains native owners until queued work provably drains.
     c. Speculation not lossless: V4 Pro EXL3 K2 dSpark diverges at token 4
        (1.95 nat), C4 ≠ C1 at token 15; GLM 5.3 Flash tr3 DFlash2 0.84 nat;
        GLM 5.3 EXL3 0.57 nat. Suspect multi-row verify numerics/state.
        GLM Flash: FP32 MLA decode partials + K64 EXL3 for 2–16 rows
        (`262cf29`) cut serial-vs-verify KL 0.0060 → 0.0041; rejected-suffix
        causality checks pass. Byte equality and C1/C4 invariance open.
     d. Batch invariance: C4 ≠ C1 greedy on V4 Pro, GLM 5.3, GLM 5.3 Flash.
     e. NVFP4 local experts on one RTX: was implicit expert paging, not a
        slow kernel. Fixed (`4c02f2f`): local experts are resident by
        default (Qwen NVFP4 decode 21 s → 8.7 ms/step); paging needs an
        explicit `--expert-window`.
     f. GLM 5.3 Flash has a two-GPU head split (`work/glmf-split`, `glmf2`
        programs: half the KDA/MLA heads and their state, half the dense /
        shared-expert intermediate; default with RTX_GPUS=auto/2). 2 vs 1 RTX +
        4 Sparks: EXL3+DFlash2 C1 code 159 -> 168 tok/s, NVFP4 C1 76 -> 83,
        8K prefill equal; golden NLL 2.4073 -> 2.4054. Qwen still has none
        (two-GPU requests serve from the first GPU).
     g. Qwen 3.8 EXL3: 84 tok/s with 4 Sparks vs 261 on one RTX alone.
     h. Prefill gets worse with more hardware: V4 Pro min 879 tok/s (9.2 s
        TTFT) vs 2,438 max; MiMo Flash max 2,899 vs min 5,877; MiMo Pro max
        1,754 vs min 2,741 (two-lane prefill off under the head split).
     i. V4 / V4.1 turn-end prefix-cache restores are byte-exact (fixed in the
        check, `0f65c9b`): the old check compared a restored turn with a cold
        recompute, and V4 Flash / V4.1 prefill does not repeat bit for bit
        (Spark FP32 atomic expert reduction at 256+ rows); its turns also
        ended at EOS with one row to compare. The check now judges each
        restore against its own snapshot (turn rows, prompt-snapshot
        reference, decode step after the turn restore) and reports the cold
        recompute only. Smoke V4 Flash min and V4.1 min: prompt and turn end
        2 rows byte-identical, cold recompute differs. Deterministic prefill
        stays open: an ordered serial-slice reducer passed component gates
        on a private codex branch (Flash TP4 only); not merged.
     j. MiMo V2 Flash fidelity is the weakest that passes (KL 0.10, top-1 82%).
        Opt-in BF16 expert-input Spark packages (`EXPERT_INPUT=bf16`,
        `CUTEAFD_*_FP8_MOE_BF16_FAMILIES=mimo`) improve it; default stays FP8.
     k. Qwen 3.8 FP8 has no Spark expert package (173 GB, no one-RTX fit).
     l. Fixed on codex/v1: generic families reject contexts beyond the
        compiled index extent before allocation; `HOST_CACHE_BYTES=auto`
        is bounded by live host memory; cancelled stage chains drain before
        staging is reused; generic KV admission waits (FIFO) under
        transient pool pressure instead of rejecting.
     m. V4.1 cold prefill: large index-selection / attention-query graph
        entries are evicted between encoder and replay shapes, so warm
        requests still capture graphs. Open.
5. **Spark expert kernels**: MiMo V2.6 Pro TP6 prefill is Spark-bound (~35 of
   ~42 ms per layer); GLM 5.3 verify is bound by distinct expert reads; NVFP4
   W4A16 GB10 prefill (14.3 vs EXL3 9.1 ms/layer TP4).
   GLM 5.3 (2026-10-03, `work/glm-perf`): 8K prefill is ~3.0 s on min and max
   alike because both are Spark-bound — worker kernel time per 2752-row wave
   (3 lanes) is 11.7 ms at TP4 width 512, 11.0 at TP6 width 384, 7.6 at width
   256, so six Sparks save only ~5% Spark time (the width-384 package ran
   128-wide tiles). Fork f6bb38bc (dynamic tile claims, FP8 wire input,
   192-wide TP6 tiles; bit-identical) cuts live waves to 11.24 / 9.87 / 7.05
   ms and Spark busy per 8K to 2.62 s (TP4) / 2.30 s (TP6); still Spark-bound.
   Served 8K TTFT with E4M3 MLA + these packages vs v0.1.0 (2026-10-04, one
   launch per arm): max 2.96-2.99 -> 2.49-2.64 s, min 3.08-3.11 -> 2.82-2.91 s;
   C1 code flat (max ~68, min 58-61; text changes with the MLA numerics); C4 is
   dominated by within-batch greedy divergence (item 4d) in every arm. The
   head split only moves the wait from the GPU to the Sparks. GB10's
   wave is bound by bytes (FC1 BF16 input gathers per N tile, FC2 partial
   round trip ~2 ms, top-k sum 1.6 ms) and the FC1 rotation, not MMA rate.
   Coordinator GPU-only 8K prefill is 2.7 s, half of it the sparse MLA prefill
   kernel. E4M3 MLA prefill (`work/glm-mla-fp8`, merged, default e4m3-p2;
   real-expert TP4 gate: KL vs golden 0.0364 -> 0.0379, NLL 2.4680 -> 2.4717,
   deterministic) cuts the GPU-only 8K prefill 2.89 -> 2.57 s, but 8K with
   Sparks stays 2.95 s (GPU wait 2.11 -> 1.45 s, Spark wait up): the GB10
   wave is the bound. Lanes 2 or 4 lose to 3.
   Verify layouts (busiest-rank expert reads, uniform routes): TP6 beats
   TP2xEP3 up to 16 rows (1.50 vs 2.02 expert-equivalents at 1 row, 10.8 vs
   11.2 at 8) and loses by 1-8% only at 32-64 rows; keep TP6. DFlash2 at max
   (TP6 + split, code C1): adaptive 66-68 tok/s vs fixed 7/5/3 at 65/62/61;
   offline trace scoring with the measured TP6 table: adaptive 72.9 vs best
   fixed 62.7 (oracle 83.3) — the policy is not the limit.
6. **RTX 5090 audit and claim**: hard-coded `4*188` grid clamps and the
   per-tensor FP8 GEMM grid sized for 188 SMs; one SM120 build must serve both.
   Expert quantizer grids now come from each engine's GPU (`6d4ea7a`).
7. **Phase 6 placement planner** (incl. cold components such as the vision
   encoder on a Spark) and **multimodal input** (official encoders only).
   **Joint serving capacity** (TJ): default C16 with 20 front-state slots and
   a common 2,097,152-token GPU KV pool; reserve KV, workspaces, graphs,
   transport and drafters per physical GPU (97% of total minus existing use)
   before onboarding expert layers; report shortfall instead of silently
   shrinking. Pure resolver: `cuteafd-core`/`cuteafd-loader`
   `serving_capacity`; `cuteafd plan` describes cache storage. MiMo admits
   its runtime reservations before loading (`mimo_v2/admission.rs`). Next:
   startup consumes the same resolved plan for every family.
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
     A16 today. Checked 2026-10-03: nothing upstream runs standard (MCG) EXL3
     with A8 — master's W4A8 trellis decodes only QSRT; brandonmusic's PR #342
     (MCG->E4M3, SM120 TP4, source-available licence, expert-shared suh) does
     not fit our checkpoints. GB10 measured full-rate F16 MMA (124.8 TFLOPS,
     E4M3/INT8 248), and an INT8 A8 prototype (fork `cuteafd/exl3-a8`, local;
     INT8 weights 0.9-1.6% rel error vs 3.7% for E4M3) saves only ~7% of a
     GB10 wave: not built. The wave is byte- and rotation-bound (item 5).
     SM120 measurement branch `work/exl3-a8-rtx`: pinned real-weight local
     layer probes and an independent-chain MMA ceiling benchmark are ready.
     The 15% gain gate remains unqualified: Nsight Compute approvals expired,
     and repeated 8K GPU kernel durations varied substantially. No A8 port
     or default change; collect counters and resolve timing variance first.
   - MXFP4 experts (V4.1 already W4A8; MiMo V2.6 Pro W4A16): A8 prefill for
     MiMo Pro (Spark-bound prefill), and MXFP4 × MXFP8 MMAs for both.
   - FP8 experts: extend W8A8 (MiMo GB10 gate/up) to the down projection and
     to RTX-local experts where KL allows (Qwen FP8 local was +0.024: needs
     finer activation scales).
10. **Resident weight representations** (TJ, 2026-10-03): one resident
    BF16 or FP8 representation per weight set; preserve each checkpoint
    tensor's precision by default; calibration-free conversion only as an
    explicit option; drafter precision is chosen by emitted tok/s and memory.
    Landed: MiMo resolves head/O/drafter formats from headers; MiMo V2.6
    Pro defaults to single-copy FP8 head/O/DFlash (TJ-approved exception;
    `MIMO_WEIGHT_POLICY=checkpoint` opts out). Dual-copy options now fail
    before loading until single-copy consumers exist: Qwen
    `--fp8-decode`/`--mtp-fp8-head`, GLM Flash KDA `row128`/`channel` and
    FP8 head. GLM/GLM Flash DFlash default to checkpoint BF16; single-copy
    FP8 is `SPECULATOR_FP8=on`. Open: compact FP8 consumers for Qwen
    projections and GLM Flash KDA (recover the dual-copy decode speed),
    GLM target head/index operands, and a measured drafter-precision default.
11. **Parked**: Spark-side reduce-scatter ([`work/spark-reduce`](https://github.com/tpurtell/cuteafd/tree/work/spark-reduce),
   +3% one rail, +9–12% two rails at 200G); split intake
   ([`work/split-intake`](https://github.com/tpurtell/cuteafd/tree/work/split-intake), slower). Revisit only on new evidence.
12. **Housekeeping**: prune agent test images on raptor; delete
    `~/.cache/cuteafd/builds/{n10-rel,bisect-rel}` on ostrich (root).

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
