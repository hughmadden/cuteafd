# SM89 DeepSeek-V4.1-Flash performance review — 2× RTX 4090D-48GB + 4 Sparks

- **Date:** 2026-11-11 (Australia/Sydney)
- **Reviewed commit:** [`379b26792`](https://github.com/hughmadden/cuteafd/commit/379b26792) — "Pin SparkInfer to the exact SM89 expert MMA"
- **Kernel pin:** `third_party/sparkinfer` @ [`ce9aec54`](https://github.com/hughmadden/sparkinfer-glmrt/commit/ce9aec549307cf7736ea296b0e71d190923fc9c1) (read at the pin via `git show <pin>:…`; the working-tree submodule is parked on an unrelated commit)
- **Review branch:** `dsh/dsf41-4090d-perf-review`
- **Target:** DeepSeek-V4.1-Flash (`deepseek_v41`), official MXFP4 release, on a 2× RTX 4090D-48GB (SM89/Ada) coordinator pair + 4 DGX Spark (GB10/SM121) expert ranks at TP4.

## Scope and method

Static code review plus online research. **No engine, GPU, or build was run.** Read
`PLAN.md`, `docs/architecture/shared-engine.md`, `docs/models/deepseek_v41.md`, the
SM89 attention / expert / mHC / dSpark code, and the fork's SM89 commit chain. Other
agents' work, worktrees, and prior review notes on this machine were not consulted.

Estimates below are grounded in the fork's own measured constants (cited) and labeled
by confidence; they are not new measurements. Per the repo's measure-first rule, each
lever lists a ceiling and a cheap next measurement.

## Bottom line

The fork is already heavily optimized:

- SM89 W4A8 expert kernel streams at **94–97% of the 1008 GB/s roofline**.
- SM89 dense prefill GEMM at **91–96% of plain FP8** (commit `565cbf10`).
- Decode/verify dense projections use a measured per-row-band GEMV / row-split plan
  (`sm89_block32_plan`, `b12x/_lib/sm89_fp8.py`) — already ahead of the vLLM SM89
  port's "M-aware GEMM" fix.
- KV is compact (~890 B/token source + 2.72 MB/sequence window) and is **not** the
  memory constraint on 48 GB.

Remaining gains are incremental and specific. The two highest-value levers are the
**Spark expert wave (77% of the GB10 roofline)** and **profile-guided expert-tier
residency** (plumbing exists; operational change). One **accuracy-parity risk** in the
dense SM89 FP8 path deserves a check. **KV-format changes are not recommended.**

## Where the decode step time goes

Basis: measured `TierStepModel::C1_VERIFY_48GB_PAIR_4_SPARKS`
(`rust/crates/cuteafd-loader/src/placement/expert_tier.rs:226`) and its physics test
(`expert_tier.rs:637`, measured 4 Nov 2026):

- 40 routed layers × 384 experts, top-6, ~24 distinct experts per C1 verify step (6 rows).
- **Coordinator GPU tier:** 9.35 µs/expert + 6.4 µs fixed + 1.6–4.7 µs route plan;
  TP2 half = 9,131,264 B → **~977 GB/s (97% of 1008 GB/s)**.
- **Spark TP4 wave:** 21.25 µs/expert + 34 µs transport/host; slice = 4,573,312 B →
  **215 GB/s = 77% of the GB10's 278 GB/s roofline**.
- All-Spark routed time ≈ 40 × 544 µs ≈ **21.8 ms/step**; the auto tier (60 experts/layer,
  ~16% of bytes) cuts ~3.0–3.4 ms. Attention/dense/mHC/router/index/engram/drafter/sampling
  make up the rest.

Routed experts dominate decode; the Spark side is both the slower pool and the one
leaving roofline headroom. That ordering drives the recommendations.

## Ranked improvements

### 1. Close the Spark expert-wave roofline gap (77% → ~90%) — biggest single perf lever

The coordinator's SM89 W4A8 kernel proves ~95% of DRAM roofline is achievable on a
memory-bound expert wave; the Spark SM121 block-scaled MXFP4 wave sits at 77%
(`expert_tier.rs:633`). Wave merging already landed (#844); the residual gap is likely
gather/scatter of merged units, prefetch depth, or the reply reduce on GB10 unified memory.

- **Estimate:** lifting the non-resident share (~84% of bytes at the auto tier) from
  21.25 → ~18.3 µs/expert (≈88% roofline) saves ~2–2.5 ms/step → **~5–10% decode tok/s**.
  Half-gap closure is the conservative case; GB10 unified memory may cap practical gains
  below the coordinator's 97%.
- **Accuracy:** none (same numerics, faster streaming).
- **Effort:** Spark-side kernel profiling + tuning (SM121), gated by the fork's row-curve method.
- **Confidence:** medium (77% is measured; the target is inferred from the coordinator).

### 2. Profile-guided static expert tier (operational, already supported) — best effort/reward

The auto tier ranks by *uniform* popularity; real DeepSeek routing is skewed (the fork's
concentrated-profile test has 16 experts taking ~98% of routes, `expert_tier.rs:591`).
The census → profile → re-plan path exists (`CUTEAFD_V41_ROUTE_CENSUS`,
`--gpu-expert-tier-profile`, `expert_tier.rs:384`). At the *same* memory budget, a profiled
tier on skewed traffic raises byte-hit rate from ~16% toward 40–90%: the physics test shows
40% hit saves **7.8–8.4 ms** vs uniform 81/layer's 4.1–4.45 ms (`expert_tier.rs:713`).

- **Estimate:** up to ~2× the tier's decode saving on skewed traffic → **~5–15% decode tok/s**,
  workload-dependent. Near-zero code risk (config/re-plan only); residency stays fixed for
  the server lifetime, so exact prefix restores are preserved.
- **Action:** collect a census on representative traffic, re-plan with the profile, A/B once.
- **Confidence:** high that it helps on skewed traffic; magnitude depends on the census.

### 3. Verify the engram host-read path on the 4090D host — high value if RAM-constrained

Engram is 203 GB of host-mapped tables read at layers 1 & 14 every token. The Engram paper's
efficiency claim rests on *deterministic addressing → runtime prefetch from host memory,
negligible overhead* ([arXiv:2601.07372](https://arxiv.org/abs/2601.07372)) — which only holds
if the host path is fast. The fork already has look-ahead + a bounded row cache (#916/#917:
8K prefill 10.3k→11.2k, +9% on SM120), `ENGRAM_WARM`, and `MAPPED_SCALE_PRELOAD_GIB`
(`docs/models/deepseek_v41.md:68`).

- **Estimate:** if the hot n-gram working set misses page cache and hits NVMe/SparkNest random
  I/O, this is a per-token stall the SM120 cards don't pay; sizing the row cache to the hot set
  and enabling warm/preload when host RAM allows could recover **several % of decode and more
  of prefill**. If RAM already covers the hot set, gain ≈ 0.
- **Action:** `PROFILE` the engram read latency on the actual host first; size
  `rust/crates/cuteafd-loader/src/formats/mapped_table/row_cache.rs` shares to the measured hot set.
- **Confidence:** host-dependent; measure before building.

### 4. Make attention split-K depth adaptive (parts is fixed at 10) — modest kernel lever

SM89 runs the native FP32 ten-part kernel (the AOT path is SM120-only,
`rust/crates/cuteafd-daemon/src/families/deepseek_v41/v41_native_serve.rs:199`). Each call
writes/reads `rows×10×64×514×4 B` of FP32 partials plus a merge launch; at 6 rows that's
~15.8 MB/layer × 40 ≈ 630 MB/step of pure partial traffic
(`native/families/deepseek_v41/cuda/v41_sparse_attention.cu:439`). With 640 keys = 10 tiles,
parts=10 means one tile per part — maximal split, merge-dominated.

- **Estimate:** fewer parts (4–5) with more tiles/part roughly halves partial traffic →
  **~0.3–0.6 ms/step (~1–2%)**. The shared-tile path already amortizes KV decode across head
  groups, so the win is the partial/merge overhead, not KV reads.
- **Accuracy:** changing parts reorders FP32 merge accumulation → low-bit logit diffs only
  (acceptable under the repo's rules; prefix-cache state unchanged).
- **Effort:** small kernel + row-curve check. **Confidence:** medium.

### 5. Dynamic expert re-residency at quiet points — measure-first

The replay simulator, quiet-point gating and prefix-invalidation pricing are built
(`rust/crates/cuteafd-loader/src/placement/tier_cache.rs`, `commands/tier_cache.rs`), but
**serving is static-only**. Windowed-LFU/LRU would track routing drift; the oracle bound is
already computable.

- **Estimate:** workload-dependent; run the simulator on a real census trace *before*
  implementing. The exactness cost (GPU vs Spark low-bit differences invalidate snapshots
  touched by a residency change) is real and already priced by `invalidations`.
- **Recommendation:** only build if the simulator shows a clear win over the profiled static
  tier (item 2) on non-stationary traffic. **Confidence:** low until simulated.

### 6. dSpark drafter L2 residency + SM89 cost-model re-calibration — small, profiling-gated

The two-row Markov-head GEMM trick (commit `0b1354601`) cut the 66 MB head from 381→126 µs by
keeping it in the 4090's 72 MB L2 — precedent that L2-residency tricks pay here. The whole
draft graph is ~2.93 ms at C1. The adaptive-width confidence→width cost model was calibrated
on SM120 timings; the SM89 draft/verify ratio differs.

- **Estimate:** draft is ~10% of the round; a 20% draft improvement ≈ **~2% overall**, plus a
  possible few % from re-calibrating width to SM89 timings (judge by emitted tok/s, not acceptance).
- **Confidence:** low-medium; profile first.

### Smaller items

- Fuse the `slot_of` route-plan remap into the router/metadata upload (1.6–4.7 µs/layer,
  ~0.6% of step, `expert_tier.rs:222`).
- Pin/size the decode graph bank to stop the known encoder↔replay graph eviction/recapture
  leak (`docs/models/deepseek_v41.md:88`) — a stability/efficiency fix, not a sized speedup.

## KV cache: leave the format alone

KV is already compact and **is not the memory constraint** on 48 GB: ~890 B/token source pages
(FP4) + 2.72 MB/sequence FP8 window marks (`PLAN.md:2354`); a 2M-token pool is only ~2 GB, and
the pool-first policy (`rust/crates/cuteafd-loader/src/placement/pool.rs`) leaves the bulk of
memory to the expert tier. Quantizing the FP8 window to FP4 would save tens of MB while
degrading the most-attended recent tokens — a bad trade under "no accuracy regression." The
productive KV-adjacent levers are prefix-cache hit rate (already exact radix banks + pinned-host
tier) and item 4's partial traffic, not the storage format. **Recommendation: no KV format change.**

## Accuracy flag (verify, don't assume)

The dense SM89 block-32 GEMM **and** GEMV still use `mma.m16n8k32` E4M3 per K32 block
(`b12x/_lib/sm89_fp8.py` at pin `ce9aec54`, lines 310/814/823) — the *same* FP8 MMA whose
limited in-block K32 accumulation the expert pin `3cd2737e` just proved flips ~25% of BF16 route
outputs (75% recipe equality, relL2 2.6e-3) versus 99.97% / 2.5e-5 for the exact F16 MMA. The
dense module's docstring claims results "differ from the SM120 block-scaled MMA only by FP32
accumulation order," which the expert evidence contradicts for the in-block sum. The dense
outputs feed norms/attention (more tolerant than the experts' tight MXFP8 requant), and the fork
presumably passes its KL/top-1 gates — so this is **not a confirmed regression**, but the parity
claim is unverified.

- **Action:** run the experts' recipe-parity oracle on the dense projections. If it shows
  expert-scale error, apply the same exact-F16-MMA fix; expect the same **~1.5–5% memory-bound
  cost** the expert fix paid (a small speed trade for accuracy parity, consistent with the repo's
  native-numerics principle). This is the one place accuracy may already be silently below the recipe.

## Suggested next measurements (cheap; none run here)

1. Census on representative traffic → re-plan the tier with the profile → one A/B (item 2).
   Highest reward, lowest risk.
2. `PROFILE=expert-timing,transport-timing` on a Spark wave to attribute the 77% gap
   (kernel vs gather vs reply) before tuning (item 1).
3. Engram read-latency profile on the 4090D host (item 3).
4. Dense recipe-parity oracle (accuracy flag).

Each has a clear ceiling (the roofline / physics-test numbers above) and stop bar. Avoid
cross-engine comparison tables — the vLLM SM89 port is useful only as technique prior art (it
confirms the M-aware decode GEMM lever the fork already implements, and its FlashInfer
`0.6.14+sm89` "one scale for four accumulators" quality decay independently corroborates the
fork's exact-MMA stance).

## Sources

- [vLLM DeepSeek-V4 SM89 port](https://github.com/cwysm/vllm-deepseek-v4-sm89-0.27.2/blob/sm89/README.SM89.md)
- [sglang SM89 FP8 indexer PR #28620](https://github.com/sgl-project/sglang/pull/28620)
- [Engram paper (arXiv:2601.07372)](https://arxiv.org/abs/2601.07372)
- [MoE inference survey — expert caching](https://ar5iv.labs.arxiv.org/html/2412.14219)
- [RTX 4090 72 MB L2](https://www.notebookcheck.com/GeForce-RTX-4090-vs-GeForce-RTX-3050-vs-GeForce-RTX-3050-Ti-Laptop-GPU_11436_11144_10667.247532.0.html)

Hardware (114 SMs, 1008 GB/s, 48 GB) taken from the fork's own
`python/tools/lib/sm89_cards.py` and measured commit tables.