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
  cuteafd-core       ids, geometry, placement math, admission/lanes, KV allocator, sampling params
  cuteafd-ffi        libloading C ABI; one module per kernel family, family-namespaced symbols
  cuteafd-loader     checkpoint catalog, family readers (HF config -> ModelSpec), LoadPlan,
                 capability check, fast sliced readers, mapped tables, sparknest placement
  cuteafd-transport  ExpertProtocolV2, verbs RoCE, TP x EP topology (unchanged from ds41rt)
  cuteafd-hostcache  pinned host RAM prefix snapshots (unchanged)
  cuteafd-engine     model-agnostic serve runtime: scheduler, lanes, prefix cache, memory,
                 speculative transaction framework, console state
  cuteafd-api        OpenAI chat + completions, constraints, tools, images, console
  cuteafd-families/  deepseek_v41, deepseek_v4, glm_dsa, glm_next, mimo_v2, qwen4_exp
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
   `mla_indexing.cu`, `packed_fp8_mla_exact.cu` stay for Phase 1.
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
- Dense NVFP4/per-tensor-FP8 parts dequantize to BF16 at load.
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
The drafter follows the GPU that owns the last backbone layers (taps and head
live there); TP2 drafters are ≤1% on DFlash2 and not built unless the P2P
probe shows ≤15 µs hops; the win is lane B drafting on GPU1 while lane A
verifies on GPU0 at C≥2. Benchmark only the natural minimum and maximum
configs (AGENTS.md); the planner's estimates cover the rest.
One `ExpertRouter` replaces the six per-family stage/send/land/reduce copies;
V4.1's `receive_owned` stays untouched. Stages: S0 planner + `plan` (must
reproduce today's layouts), S1 manifest handoff + workers, S2 router in the
generic engines + GPU1 as expert host, S3 EP subsets (only if a quantized
model needs them; GLM 5.3 official FP8 is out of scope — EXL3 and NVFP4
quants cover it), S4 encoder service + multimodal input, S5 coordinator
range split, S6 eight Sparks.

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
