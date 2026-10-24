# MiniMax M3.1-Flash on cuteafd: implementation review

Branch `hugh/minimax-m3` at `51f79cf68` (46 commits over upstream `work/p0` `f7bb2da19`,
185 files, +17,184 / −258), kernel fork `hughmadden/sparkinfer-glmrt` at `012972f7`
(7 commits over the base pin `56ec22d0`). Reviewed 2026-10-24 by the Claude agent within
Turq, from the code on the branch plus the raw run logs under
`~/.cache/cuteafd/builds/minimax-juliette/resume-logs/`. No plan, hand-off or report
written by another agent was used.

## Verdict

The bring-up is functionally real: native NVFP4 W4A4 experts, native MXFP8 W8A8
projections, exact paged MSA cache with a byte-exact prefix-restore gate, a working bundled
DSpark drafter, TP2/3/4 all booting on real weights. What it is not is fast, and the
reasons are structural rather than tuning:

1. On the RTX 5090 the expert exchange is host-synchronous per layer (a full stream drain,
   host route parsing and a payload memcpy on every one of 57 routed layers).
2. Prefill runs through the decode kernels one 64-row tile at a time, while the prefill
   kernels that were exported for exactly this are never launched.
3. There are no CUDA graphs: roughly 2,300 kernel launches per decode step, each wrapped in
   a device get/set pair.
4. On a 32 GiB card the allocation ledger spends ~5.2 GiB on an all-row FP32 logits buffer
   and on DSpark rings sized for 64 sequences, then proposes NVFP4 KV and host paging to
   recover capacity that these two fixes already return.
5. The MXFP8 linears have no split-K and the four attention/indexer projections are
   launched separately, so coordinator weight streaming at decode runs well under HBM speed.

Measured on juliette (RTX 5090, Sparks 7–10 at TP4, build `51f79cf68`): C1 decode ≈20 tok/s
without the drafter and ≈45 tok/s with it; four concurrent streams give an *aggregate* of
15–21 tok/s, i.e. less than one stream; prefill of a 4.4K prompt takes ≈6 s (≈700 tok/s).
My estimate of what this cell should do with the same kernels and formats is ≈50–55 tok/s
C1 target-only, 100+ tok/s with DSpark, and ≥2,000 tok/s prefill (network-bound on the 100G
spokes). The gap is 2.5–3× at decode and 3–4× at prefill, and every item below is a
contributor whose size can be reasoned about from the code.

The fixes are ordinary engine work in this codebase's own idioms (the device-driven
exchange, the graph-captured step, split-K linears and prefill kernels all exist for other
families or in the fork already). None needs a new checkpoint format, a new transport, or
the capacity schemes the branch documents.

Two side surfaces have their own serious items (appendices A and B): the tool-call parser
is quadratic in the size of a tool-call block and drops every call in a block on one bad
invoke, and video is segmented at 4 raw frames where the model segments at 4 temporal
patches (8 frames), with defaults that reject ordinary 720p/1080p input. Still images and
the chat-template rendering are faithful.

## Evidence used

| Run (`resume-logs/`) | Config | Result |
| --- | --- | --- |
| `boot-real-off` | C1, TP2, no drafter | `code`: 208 tokens in 10.7 s incl. a 203-token prefill → ≈20 tok/s decode (≈50 ms/step) |
| `boot-real-dspark` | C1, TP2, DSpark | `code`: 208 tokens in 5.04 s → ≈45 tok/s; 3.0–5.1 accepted tokens per cycle |
| `boot-real-c4-off` | C4, TP2 | round 0: 284 tokens / 18.9 s → 15.0 tok/s aggregate |
| `boot-real-c4-dspark` / `-tp3-` / `-tp4-` | C4, DSpark | round 0: 21.6 / 20.7 / 18.3 tok/s aggregate (TP2 / TP3 / TP4) |
| `boot-real-c4-tp4-dspark/coordinator.log` | 480-token request | 157 verify cycles in 16.75 s → 107 ms per verify step; `experts_s` ≈ `verify_s` |
| `needle` probes (4,379-token prompt) | C1 | 8.1 s (off) and 7.1 s (DSpark) total for 42 tokens → ≈6 s prefill → ≈700 tok/s |
| all runs | | readiness ≈61 s; needles to 6.6K tokens answered correctly; prefix cache hit on repeat |

Observations that frame the findings: step time *grows* with rows (≈50 ms at C1 → ≈107 ms
for a C1–C2 7-row DSpark verify) although every GEMM in the step is weight-bandwidth bound
at these row counts; TP4 is slower than TP3, which is impossible if the Sparks were the
bottleneck (4 × 273 GB/s against 3 ×); and the scheduler attributes the entire verify time to
`experts_s` because the backend's profile bucket is the whole execution.

Hardware floor used below (C1, one token): coordinator weights ≈13.2 GB per token (attention
+ indexer 6.65 GB, shared experts 3.23 GB, dense layers 0.68 GB, routers 0.18 GB, head
2.46 GB) → ≈7.4 ms at 1.79 TB/s; routed experts 4 × 28.3 MB × 57 layers = 6.45 GB per token,
1.6 GB per rank at TP4 → ≈6–8 ms at 200–273 GB/s, overlapping only the shared expert; two
RDMA hops per layer ≈0.05–0.1 ms. That is ≈0.3 ms per layer, ≈18–20 ms per token.

## Findings, ranked

### 1. The 5090 expert exchange drains the stream and hops through the host on every routed layer — CONFIRMED

`rust/crates/cuteafd-daemon/src/families/minimax_m3/routed.rs` `PinnedSpark::dispatch`
(lines 80–101) and `collect` (103–115), introduced by `50ab9619a`.

Per routed layer, per step: (a) `RouteStaging::enqueue` copies ids, weights and the BF16
rows device→pinned; (b) `commands.drain()` — a full `cudaStreamSynchronize` of the compute
stream — so the host can read the routes; (c) the host parses routes and copies the whole
wire payload into a fresh `bytes::Bytes` (`Bytes::copy_from_slice(layout.wire(staged)?)`,
line 96); (d) the host posts the RDMA wave; (e) `collect` blocks the scheduler thread in
`tokio::Runtime::block_on` with a 120 s timeout until all rank planes have landed and been
uploaded, then queues the reduce. Fifty-seven times per token.

Why it matters:
- The drain empties the launch queue at every layer boundary, so every one of the ≈38
  kernel launches per layer (see 3) is exposed as latency instead of overlapping with
  device work. The ≈0.85 ms per layer measured at C1 is dominated by this: my estimate of
  the device work in a layer at C1 is ≈0.25–0.3 ms.
- The payload memcpy is 12 KiB × rows: negligible at C1, 50 MiB per layer at the 4,096-row
  prefill chunk (≈2.9 GB of single-threaded memcpy per chunk). The zero-copy egress path
  (`SparkLink::egress` / `egress_payload`, `EGRESS_MIN_BYTES = 1 MiB`) exists upstream and is
  not used.
- Host work per layer grows with ranks (one receive, one upload order per rank) and with
  rows, which is why TP4 measures slower than TP3 and why four streams aggregate below one.
- It violates the engine's own rule (AGENTS.md, Device-driven exchange): "no host hop and no
  idle CPU burn on a request's hot path. The host launches work and waits on it; it does not
  shuttle bytes between devices or poll in a spin loop."

The GeForce constraint is real (no dma-buf / GPUDirect RDMA, so replies cannot land in GPU
memory), but it only forces *host-landed replies*; it does not force a stream drain, a
host memcpy, or a blocking wait on the scheduler thread.

Fix (device-paced pinned exchange, mirroring `SparkDeviceLink`):
- Route mailbox in *mapped* pinned memory: `minimax_route` writes ids/weights straight into
  a host-mapped arena (or a tiny kernel copies them there) and a `cuEventRecord` follows; no
  D2H staging kernel, no drain.
- A proxy thread (the `SparkDeviceLane` pattern already spawns one) waits on that event,
  builds the request header from the mapped routes, and posts RDMA writes *from* the mapped
  rows (zero-copy egress). The scheduler thread never blocks.
- Replies land in pinned planes as today; the completion is a NIC write to a mapped flag.
  Order the reduce with `cuStreamWaitValue32` on that flag (or let the proxy thread record
  an event the compute stream waits on). Either way the compute stream never drains and the
  host never copies planes by hand.
- Keep the shared-expert GEMMs between dispatch and collect so they overlap the Spark round
  trip (already the case).

Expected effect: per-layer fixed cost ≈0.85 → ≈0.3 ms at C1 (≈50 → ≈20 ms per token), TP4 ≥
TP3, and concurrency scaling restored. Verify with the C1 + C16 code-decode A/B that AGENTS.md
prescribes for shared hot paths.

### 2. Prefill runs through the decode kernels; the exported prefill kernels are never launched — CONFIRMED

`execution.rs` `attention()` lines 241–267: for every layer, the chunk is cut into
`decode_rows` (64) tiles; for each tile the host builds a per-row page table
(`vec![-1i32; count*stride]`), uploads `page_table`, `cache_lengths` and `cu_seqlens_q`
(three `copy_h2d_ordered` per tile per layer), then launches `minimax_index_decode` and
`minimax_attention_decode` with each row as an independent one-token request.

Meanwhile `minimax_index_prefill` and `minimax_attention_extend` are exported
(`python/tools/aot/export_b12x_minimax_aot.py` lines 18–27), required by admission
(`plan/families/minimax/admission.rs` `PROGRAM_NAMES`), and compiled into every image, but
the ledger explicitly refuses to allocate their scratch: "Prefill uses causal paged decode
tiles; contiguous index/extend arenas have no owner in this executor" (admission.rs
107–111). The fork's `minimax_attention.py` `extend` mode tiles 8 queries over the union of
their selections (≤128 blocks) so K/V loads are shared across queries; the decode mode
re-streams each row's 2,048 selected keys for that row alone.

Costs, for one 4,096-token chunk:
- 64 tiles × 60 layers = 3,840 host-built page tables and 11,520 small H2D uploads; the page
  tables are layer-invariant but rebuilt and re-uploaded 60 times.
- The decode indexer scans the whole causal prefix per row (`DSAScheduledMultiRowLogitsKernel`
  in `PAGE_HEAD_MAX` mode), ≈1.1 GB of index-page reads per layer at 4K; the contiguous
  prefill scorer (`DSAContiguousLogitsPrefillKernelBlockScores`) is a tiled GEMM.
- `clear_pages` memsets `rows × 4 × max_pages × 4` bytes per tile per layer — 16 MiB each at
  the exported 1M-token capacity, ≈60 GB of memset per chunk (minimax_indexer.py 168–170).
- Decode attention at 64 rows × 4 KV heads = 256 CTAs per tile, each re-reading its row's
  2 MiB of selected K/V with no cross-row reuse.

Measured consequence: ≈700 tok/s prefill where ≈2,000+ is the network floor on this cell.

Fix:
- Build positions, slots and the per-request page tables once per step, outside the layer
  loop; upload once. (At 1M context a per-row table is 64 KiB; the extend kernel takes
  per-*request* tables plus `cu_seqlens`, so the table is `requests × stride`, not
  `rows × stride`.)
- Stage the chunk's index-K rows contiguously (rows × 132 B, trivial) and launch
  `minimax_index_prefill` with `k_start`/`k_end`; feed its `q2k` to
  `minimax_attention_extend`. Size the prefill scorer arena by the admitted context
  (`--max-context`), not the 1M export (`max_pages` is a compile-time capacity; export a
  second geometry for the 5090's ≈180K-token pool, or chunk the scorer by key range).
- Keep the decode-tile path only as the fallback for the last partial tile if the extend
  kernel's 128-aligned request-start contract is inconvenient.

Expected effect: 3–5× on TTFT for long prompts; the per-layer host work in prefill goes to
zero. Gate: byte-identical cache pages against the decode-tile path on the same prompt
(the golden's `check_prefix` already compares cache bytes), then NLL on the window set.

### 3. No CUDA graphs; ≈2,300 kernel launches per decode step, each wrapped in a device switch — CONFIRMED (no graphs), estimate (count)

`cuteafd_ffi::minimax::Commands::launch` (ffi `minimax.rs` 228–288) runs every program
through `on_device`, i.e. `cudaGetDevice` + `cudaSetDevice` + launch + `cudaSetDevice`, after
re-validating operand extents and 16-byte alignment on the host. Per layer the executor
issues ≈31 stream operations (norm, 4 linears, kv_split, 4 norm/rope, 3 pack/quant, 3
uploads, index_decode (6 internal launches), attention_decode (2), o-proj, residual, norm,
router_scores, route, dispatch, 3 shared-expert ops, collect/reduce, add, residual, draft
taps). The planner's own note records it: "causal paged MSA tiles; zero graphs".

At ≈3–4 µs of host issue per launch that is ≈7–9 ms of host-bound time per token at C1 (of a
≈50 ms step), and with the per-layer drain of finding 1 none of it overlaps device work.
AGENTS.md's rule "zero steady-state graph captures per request" presumes a graphed step
captured at start-up; this executor has no capture at all.

Fix: capture one graph per (row bucket, kind) for the attention → router → shared-expert
segment of a layer and for the post-collect segment, at start-up, with the scratch views
and page-table buffers as fixed graph inputs (the DeepSeek V4.1 `decode_graph` module shows
the shape). On the device-driven exchange the dispatch/collect are mailbox writes and a
reduce and can be graph nodes; on the pinned path they bound the segments. Drop the
per-launch `cudaGetDevice`/`cudaSetDevice` pair for the common case where the stream's
device is already current (set once per step).

Expected effect: ≈5–8 ms per token at C1, more at higher concurrency; necessary for the
C16 code-decode number to be competitive at all.

### 4. The allocation ledger wastes ≈5.2 GiB on a 32 GiB coordinator — CONFIRMED

`plan/families/minimax/admission.rs` `RuntimeLedger::new` (lines 128–165) and
`families/minimax_m3/serve.rs`:

- `logits` is `rows × vocab × 4` = 4,096 × 200,064 × 4 = 3.05 GiB (line 147), because
  `TargetHeadOwner::logits` projects *every* row of every prefill chunk
  (`execution.rs` 199, `head.rs` 11–21) and `prefill_device` then slices the last
  `logit_rows` (`backend.rs` 149–153). That is a 10 TFLOP head GEMM per 4,096-token chunk
  whose output is discarded except for one row per request, and `full_prefill_logits` is
  hard-coded `true` in `geometry()` (backend.rs 112) while the `--full-prefill-logits` flag
  (serve.rs 51) is parsed and never read.
- DSpark context rings are sized by `max_sequences` (64): `ring_plane_bytes = slots × 4096 ×
  512 × 2` × 10 planes = 40 MiB per slot, 2.5 GiB (`loader/.../dspark.rs` 93–94,
  `serve.rs` 129). The branch's own capacity doc notes rings "are indexed by scheduler slot
  ID" and so cannot shrink — that is a one-line indirection (slot → ring index) away from
  being sized by *active* sequences.
- `headroom: 2 << 30` is hard-coded in `serve.rs` 147/166 on top of the planner's headroom.

The doc `docs/models/minimax_m3.md` tabulates the result (4.304 GiB target workspace,
5.103 GiB DSpark, 75,136-token pool at TP4) and then proposes NVFP4 KV, host paging and
Spark paging — all "accounting only, not implemented" — plus a `--minimax-profile rtx5090`
that buys capacity by giving up concurrency (C1) and prefill chunk size (256 rows).

Fix: `logits` → `[decode_rows, vocab]` with the head run only on the last `logit_rows` rows
of each request (gather the needed `head_input` rows first; decode already fits in 64
rows); ring slots → admitted concurrency with a slot map; make `--full-prefill-logits`
mean what it says. Savings ≈3.0 GiB + ≈2.2 GiB (at 8 ring slots) → ≈+80K tokens, a ≈155K
pool at C64/4,096 rows, about what the "E + S" profile reaches by sacrificing both. Then
decide NVFP4 KV on fidelity grounds with the golden, not as the capacity plan.

### 5. MXFP8 linears: no split-K, four separate QKV/indexer launches — CONFIRMED (kernel), estimate (magnitude)

Fork `b12x/integration/cuteafd/_mxfp8_linear.py` lines 17–32: `Mxfp8LinearStage` sets
`self.slices = 1` and builds `_DenseGemmLaunch` directly with `mma_tiler_mn=(64,128)`,
`cluster (1,1)`, one K slice, bypassing `Fp8LinearStage`'s split-K lowering
(`_linear.py` 117–127, which supports `slices > 1` with FP32 partial planes and one
reduce). The persistent tile scheduler therefore has `ceil(rows/64) × N/128` tiles: at
decode that is 64 (q), 8 (kv), 48 (o), 4 (index_q), **1** (index_k), 48/48 (shared), 192/48
(dense) CTAs on a 170-SM RTX 5090 (188 on the PRO 6000). `execution.rs` 245–247 launches
q, kv, index_q and index_k as four GEMMs over the same normalized input.

A weight-streaming GEMM with 1–64 CTAs and one TMA pipeline per CTA does not saturate HBM;
I estimate ≈2× over the ideal ≈95 µs per layer for the ≈167 MB of attention, indexer and
shared-expert weights at C1, i.e. ≈6 ms per token, and far worse per byte for the 1–8-CTA
projections (`index_k` streams 786 KiB through a single SM).

Fix: fuse `q_proj`, `k_proj`, `v_proj`, `index_q_proj`, `index_k_proj` into one
`[9856, 6144]` operand at load (the scale `scale_prep` is per row, so fusion is a
concatenation; `kv_split` becomes a view); route the MXFP8 stage through the split-K
lowering for `rows ≤ 64` (slices ≈ 4–8 so every projection reaches ≥170 CTAs), with the
existing FP32-partial reduce. Gate: bitwise-identical output for the fused/unfused pair is
not expected (reduction order changes), so use the golden's KL/NLL window and the existing
kernel contract tests.

### 6. Speculation depth ignores MoE expert fan-in, so concurrency does not pay — CONFIRMED (mechanism), measured (effect)

`serve.rs` 212: `Drafts::Neural { depth: 7, fixed: false }` for every request;
`backend.rs` `draft_geometry` (177–186) returns one untimed layer with `slice_bytes: 0`, so
the shared draft policy has no cost model to trade depth against.

With top-4 routing over 128 experts, a C1 target-only step touches ≈4 experts per layer; a
7-row DSpark verify touches up to 28; four such sequences touch up to 112 of 128, at which
point each Spark streams nearly all of its ≈51 GB expert slice (TP4) per step, ≈190 ms at
273 GB/s. The measured 107 ms per 7–14-row verify step is consistent with 25–45 distinct
experts per layer (10–18 GB per rank) plus the per-layer overhead of finding 1, and the flat
≈18–21 tok/s aggregate at C4 is the same effect: expert bytes per step grow with rows until
every step reads most of the model. At C1 the drafter is worth 2.2× (≈20 → ≈45 tok/s); at
C4 it is roughly break-even; at C8+ it will lose.

Fix: give the policy a real resource line for the routed layers (bytes per distinct expert
per rank, with the distinct-expert count as the row-dependent quantity), so depth falls as
active rows rise (7 at C1, ≈3 at C4, off above ≈C8), and report acceptance and emitted
tok/s per concurrency in the bench (judge speculation by emitted tok/s, per AGENTS.md).
Independently of policy, make the verify batch cheaper per expert: the Spark package's
decode route groups (row, route) pairs by expert; check that the grouped GEMV reaches the
rank's bandwidth at 100+ active experts rather than per-expert fixed cost (the TP4 < TP3
measurement says fixed costs dominate today; see 1).

### 7. Decode attention and indexer launch geometry — estimate

`minimax_attention.py` decode traits: `cta_tile_q=16, cta_tile_kv=64, num_warps_kv=4`; grid
is rows × 4 KV-head groups, so a C1 decode launches **4 CTAs** that each stream 2,048
selected keys (512 KiB of K+V) serially: ≈10 µs per layer, 0.6 ms per token on 4 SMs. The
kernel already emits base-2 LSE per head, so a split-KV launch over the 16 selected blocks
(grid rows × 4 × 16) plus an LSE combine is the natural fix and would make this ≈1 µs. The
indexer's `clear_pages` (see 2) and the `finish_scores` kernel iterate `max_pages/2 = 8,192`
blocks per (row, head) regardless of live length. Low priority at C1 (≈1 ms per token in
total) but it compounds with 3 once graphs exist.

### 8. Qualification is thinner than the gates claim — CONFIRMED

- `families/minimax_m3/golden.rs` `verdict` (lines 201–206): full-model pass is `top1 >= 0.90
  && mean_kl <= 0.06` nats, partial-layer pass is `cosine >= 0.99 && relative_l2 <= 0.05`.
  For the *same quantized weights* served against a Python oracle of the checkpoint's own
  arithmetic, FP8-KV-level effects are ~1e-3–1e-2 nats; a gate at 0.06 would pass a wrong
  RoPE pairing or a mis-scaled head. Tighten to what the served-cache oracle arm actually
  measures, with a margin, once it has been run.
- No real-weight full golden result exists under `~/.cache/cuteafd/builds/minimax-*`; the
  evidence is CPU pytest fixtures, a layer-3 dump (`diag-layer3-off/result.json`:
  `"status": "dumped"`, no verdict) and API probes (Paris / 391 / is_prime / needles).
  `first-window.sh` lists the goldens but they have not run.
- BF16 KV cannot be measured on device: `serve.rs` 78 and `cache.rs` `for_runtime` refuse
  anything but FP8 because only FP8 programs were exported. The FP8 KV default was chosen
  without the arm that would quantify it.
- `kv_pack` takes per-request/per-head `k_descale`/`v_descale` (`_minimax_kernels.py`
  156–184) and the executor passes all ones (`execution.rs` 104–106): E4M3 at scale 1 gives
  V (typically |v| ≪ 1) three mantissa bits around 2^-9. A per-layer/head static scale from
  a calibration pass costs nothing at runtime and the plumbing already exists.
- `routed.rs` test `minimax_spark_falls_back_to_pinned_intake_with_terminal_ownership`
  (228–242) asserts on the *source text* of `routed.rs`, `serve.rs` and `golden.rs` with
  `include_str!` and `str::find` ordering. It tests spelling, not behaviour, and will break
  on any refactor including the one in finding 1. Replace with a fixture-library test of
  the fallback decision (the `native_library_lifetime_fixture` exists for exactly this).

### 9. Merge readiness — CONFIRMED

- The kernel fork pin `012972f7` sits on `56ec22d0`, the pin at the branch base; upstream
  `work/p0` has since moved its pin 66 commits (`1b921992`: EXLR8 decode/prefill work). The
  MiniMax changes themselves are additive (seven commits, +1,467/−46 against the base), so a
  rebase of the fork onto `1b921992` and a re-pin is mechanical, but it must precede any merge
  and the compare-sm-exports gate must run on the result.
- Two of the seven fork commits fix shared upstream kernels, not MiniMax code: `41a9e309`
  (FP8 multi-warp `PagedForwardKernel` reset discarded earlier tiles' state) and `012972f7`
  (LSE layout). The first affects any family using FP8 paged KV with `num_warps_kv > 1`.
  Both belong upstream as their own PR (the pin is a private fork today).
- `scripts/build/compare-sm-exports.py --sms 188,170` is the documented kernel pre-merge
  gate; the MiniMax programs were exported at the 170-SM geometry with runtime SM queries, so
  this should be clean, but it has to be shown.

### 10. Vision and API surfaces

Reviewed independently; findings and verification are in the appendix at the end of this
document (added after the core review).

### 11. Smaller items

- `execution.rs` 169–170: `widen` zeroes the entire 100 MiB FP32 `residual` buffer every
  step (`Commands::widen` = `zero` + accumulate) even for one row; zero `rows × 6144 × 4`.
- `backend.rs` 77–79: `embed` → `drain` → inject features → execute → `drain`; plus two
  `download` drains per draft step (`ffi/minimax/dspark.rs` 203–209) and one in
  `update_draft`. Five or more full synchronizations per decode step before the exchange
  adds its 57. Events, not drains.
- `weights.rs` 76–91: one synchronous `DenseLoad::mount` per operand and a `commands.drain()`
  per scale tensor during load (hundreds of syncs). Readiness is ≈61 s so it is tolerable;
  batch the scale preps and drain once.
- `routed.rs` 27–29: the warm wave sizes every rank's rings for `rows = 4096` BF16 rows
  (50 MiB frames) even for decode-only deployments; fine for prefill, but means the Spark
  side reserves 4,096-row intake per lane whether or not `PREFILL_BATCH_TOKENS` is reduced.
- `serve.rs` 150–151: vision auto-placement sizes its KV target as `pool_tokens.unwrap_or(262_144)
  × 69,360 B` = 17 GiB, which on a 32 GiB card means "auto" always exiles the tower to a
  Spark. Intentional, but should say so in the log line.
- `admission.rs` `RuntimeSizing::validate` caps `sequences ≤ decode_rows ≤ 64`; with DSpark's
  seven rows per sequence a 64-sequence verify is 448 rows across 7 decode tiles per layer.
  Fine today; becomes the next scaling wall after 1–3.
- `plan/families/minimax/mod.rs` `runtime()` returns `Planned` and `docs/models/minimax_m3.md`
  says "Status: planned" while `serve-minimax` serves real weights; the launcher hides the
  command. Update once 8 is addressed.

## What is done well

- Native formats are honoured end to end: compressed-tensors NVFP4 W4A4 with the
  checkpoint's calibrated `input_global_scale` divisors on the Sparks, MXFP8 E4M3/UE8M0
  W8A8 with hardware MMA on the coordinator, FP32 router, SwiGLU-OAI with one BF16 rounding.
  The loader reads all of it from `config.json` and the safetensors headers; the planner
  refuses anything else with a named reason.
- The cache is purely paged, the prefix cache shares pages and copies only the partial tail
  (`cache.rs` `copy_spans` covers the index-scale suffix), and the golden has a byte-exact
  restore-versus-cold comparison at 63/64/65-token frontiers. That is the hard part of exact
  prefix restore and it is right.
- Admission is a real ledger (`RuntimeLedger`, `Reservations`, `admit`) shared by the planner
  and the server, with tests that the plan and the live allocation agree byte for byte.
- The two kernel bugs found on hardware (multi-warp FP8 tile state, LSE layout) were fixed
  at the root with GPU tests against dense FP32 attention, not papered over in the engine.
- The CPU reference (`python/reference/families/minimax_m3/model.py`) is independent of
  the engine and pins the Transformers sources it was derived from.

## Recommended order of work

Each step has a stop bar so hardware time is spent only where a decision needs it
(AGENTS.md, measure-first). Juliette and Sparks 7–10 are the dev cell and can be stopped
and restarted freely.

| # | Work | Expected | Stop bar / gate |
| --- | --- | --- | --- |
| A1 | Ledger: decode-row logits + last-row head, ring slots = admitted concurrency, honour `--full-prefill-logits` (finding 4) | +≈80K pool tokens on the 5090; ≈50 ms less head work per prefill chunk | `cuteafd plan` and live admission agree; pool tokens ≥150K at TP4/C64/4096 rows |
| A2 | Prefill on the exported kernels; page tables once per step (finding 2) | TTFT 3–5× better on ≥4K prompts | cache pages byte-identical to the decode-tile path; golden NLL unchanged; ≥2,000 tok/s on the 4.4K needle |
| B1 | Device-paced pinned exchange: mapped mailbox, proxy thread, zero-copy egress, flag-ordered reduce (finding 1) | C1 off ≈20 → ≈45–55 tok/s; TP4 ≥ TP3; aggregate rises with C | C1 + C16 code decode A/B vs today's build, same day, per AGENTS.md; no `cudaStreamSynchronize` in the routed-layer path |
| B2 | Push `41a9e309` and `012972f7` upstream; rebase fork onto `1b921992`; re-pin; run `compare-sm-exports.py` (finding 9) | mergeable branch | gate clean |
| C1 | Fused QKV/indexer operand + split-K for rows ≤ 64 (finding 5) | ≈5–6 ms per token at C1 | golden window KL within gate; kernel contract tests |
| C2 | Graph-captured layer segments; drop per-launch device switching (finding 3) | ≈5–8 ms per token at C1, more at C16 | zero captures after warm-up; C16 decode A/B |
| D1 | Draft depth policy aware of distinct-expert fan-in (finding 6) | DSpark never below target-only at any C | emitted tok/s at C1/C4/C16 with depth 7 / policy / off |
| D2 | Golden: run the full real-weight window set, tighten the gate to measured headroom, export BF16 KV for the A/B arm, calibrate FP8 KV descales (finding 8) | a defensible fidelity claim | top-1 / KL on the pinned window set, both KV formats |

A1 and A2 are a day or two each and need no new design; B1 is the one piece of real
engineering and the one that unlocks everything else on the 5090. Do not start the NVFP4-KV
or host-paging designs until A1 and B1 have been measured: the capacity they target is
mostly already in the ledger, and the latency they would add sits exactly on the path B1 is
removing.

## Appendix A: API layer (chat template, tool parser, grammar, gateway)

Reviewed by a separate agent against the branch and the checkpoint's own
`chat_template.jinja`; I re-read the cited lines before including each item. The template
fixture is byte-identical to the checkpoint's; the goldens were regenerated from the real
template with jinja2 and matched byte for byte, so they are not circular; the crate's 27
MiniMax tests pass against the real snapshot headers. Findings, ranked:

1. **Tool-call block parsing is O(n²) — CONFIRMED, measured.** `openai/chat/minimax_m3/parser.rs`
   36–55: while inside a `<tool_call>` block every pushed token re-runs
   `self.pending.find(CALL_CLOSE)` over the whole held block, which is never shortened until
   the close tag; lines 56–62 also rebuild the marker `Vec` and `to_owned()` the match per
   pass. Measured with 4-byte pushes: 64 KiB of arguments 3.3 s, 128 KiB 12.8 s, 256 KiB 56 s.
   An agent writing a 60 KB file burns seconds of CPU on its stream task. Fix: search only
   from `pending.len() − (CALL_CLOSE.len() − 1)` before each append, or consume per `<invoke>`
   as the GLM parser does.
2. **Argument typing ignores JSON-Schema `type` arrays, `enum`/`const` and top-level
   `additionalProperties` — CONFIRMED.** `parser.rs` 173 `schema["type"].as_str()` is `None` for
   `["integer","null"]` (the OpenAI strict-mode idiom for optional fields), so the value falls
   to the string branch at 201; line 118 lacks the `additionalProperties` fallback that 186
   has. Every strict-mode tool call then fails `tools.validate` with "native arguments do not
   satisfy the schema"; non-strict tools get wrong-typed arguments. Fix: resolve `type` as a
   set, parse with `serde_json::from_str` and accept when the JSON type is in the set, match
   `enum`/`const` by JSON spelling, apply `additionalProperties` at the top level.
3. **One malformed invoke silently drops every call in the block — CONFIRMED.** `parse_calls`
   (107–125) returns `None` on any `?`, and line 51 then emits the entire block as `Raw`
   content with no log line. Three parallel calls with one mistyped scalar → zero tool
   calls, raw namespaced XML in `content`, `finish_reason: stop`. Fix: parse per invoke, emit
   the well-formed ones, return only the broken one as text, and `tracing::warn!` with the
   completion id as the GLM/Qwen/Step-5 parsers do.
4. **`developer`/`root` roles are unreachable from `/v1/chat/completions` — CONFIRMED (the
   adapter's role enum has no `developer` variant).** `prompt.rs` 30–59 and three goldens
   render them, but the request is first deserialized through deepseek-recipe's
   `ChatCompletionRequestMessage` (System/User/Assistant/Tool/LatestReminder), so
   `role: "developer"` is a 400 and the template's high-priority slot always carries the
   default identity text. The tests miss it because they call `encoding.render()` directly.
   Fix: map `developer` → `system` for the adapter while leaving `raw` for the template, or
   remove the dead `root`/`current_*` paths; add a router-level test.
5. **No whitespace or post-call text policy — CONFIRMED code, PLAUSIBLE effect.** `parser.rs`
   29–32 emits whatever remains as `Raw` with no leading trim, no held trailing whitespace
   and no "after the first call" guard (compare `glm5/parser.rs` 222–238). `…</tool_call>\n`
   before EOS yields `content: "\n"` beside `tool_calls`; through the Messages gateway that
   opens a trailing `text` block after `tool_use`. Fix: reuse the GLM `emit_content` policy.
6. **`thinking.type: "enabled"` is treated as adaptive — PLAUSIBLE (design).** `minimax_m3.rs`
   82–87 only appends `</mm:think>` for *off*; `grammar.rs` 58–61 makes the think block
   `optional`; nothing prefills `<mm:think>`. An Anthropic client that asks for enabled
   thinking with a budget can get a reply with no thinking block. Fix: for an explicit
   `enabled`, end the generation prompt with `<mm:think>` and make the grammar prefix
   mandatory; keep adaptive as the default.
7. **The structural-tag EBNF is never compiled by XGrammar in any test — PLAUSIBLE.**
   `grammar.rs` 9–37 generate 13 nullable right-recursive rules; `tests.rs` 405–497 only
   re-interpret the text. The DSML family has native compile tests. A syntax slip surfaces as
   a 400 on the first `tool_choice: required` request. Also `[ \n\t]*` is unbounded where
   DSML bounds `{0,16}`. Fix: a native compile-and-match test over the rendered
   `tool_round_trip` block.
8. **Engine stop set is EOS only — CONFIRMED.** `openai.rs` 221 returns `eos_token_ids`
   (`[200020]`) for MiniMax; the turn markers `]~b]` (200019) and `]~!b[` (200034) stop
   generation only when the text parser sees them, so a model that starts a fake next turn
   spends tokens first and the finish is reported as plain `stop`. Fix: add the turn markers
   to `MiniMaxTokenIds` and return them from `stop_token_ids` as GLM/Qwen/Step-5 do.
9. **`thinking.budget_tokens` sets `<effort>` even when thinking is off — CONFIRMED, minor.**
   `prompt.rs` 73–76 reads the budget independently of the resolved switch. Gate it.

Done well: prompt fidelity (the checkpoint's own Jinja through a Transformers-faithful
minijinja environment, goldens from real jinja2), hold-until-close parsing with
chunk-invariance tests, a KMP-derived "no namespace inside values" EBNF checked against
200k random strings, and thin, coherent gateway mapping (effort buckets, sampling defaults
from `generation_config`, full added-token shielding, media markers expanded after literal
shielding exactly as `processing_minimax.py` does).

## Appendix B: vision and media path

Reviewed by a separate agent against the checkpoint's bundled `image_processor.py`,
`video_processor.py` and `processing_minimax.py`, and against MiniMax's own serving
implementation (sglang `minimax_vl_common.py` / `processors/minimax_m3_vl.py`, which the
newer Transformers `modular_minimax_m3_vl.py` agrees with; the pinned `third_party/
transformers` revision carries a broken two-axis vision RoPE and is not a usable oracle). I
re-read the cited branch lines before including each item. Still images are in good shape;
video is not.

1. **Video segments are 4 raw frames (≤2 temporal patches); the model segments by 4
   temporal patches (8 frames) — CONFIRMED against the vendor's serving code.**
   `loader/media/minimax.rs` 9 `MAX_SEGMENT_FRAMES = 4` and 128–130 (`grid.t =
   frames.len().div_ceil(2)`), `api/openai/media/video.rs` 89, `vision_minimax.cuh` 108
   (`gt > 2` rejected), and the `(1..=2)` gates in `ffi/shared/vision.rs` 287,
   `shared/vision/local.rs` 49, `remote.rs` 119. The checkpoint's bundled processor does not
   segment at all; the only executable definition of `vision_segment_max_frames = 4` is
   sglang's `_apply_max_frames_limit`, which splits `grid_t` (temporal patches, `frames/2`)
   into runs of ≤4. An 8-frame clip is therefore one 4-patch attention segment with
   temporal positions 0..3 in the reference, and two 2-patch segments with positions 0,1
   twice on the branch: half the temporal attention context and wrong positions for every
   frame after the fourth. Fix: segment at 8 frames (`gt ∈ 1..=4`) at every gate above; the
   existing 4,096-token / 16,384-patch scratch already matches that worst case (4 × 768
   tokens). If `gt ≤ 2` is kept, shrink the scratch: `scratch_minimax` (cuh 36–50) reserves
   ≈758 MB on the coordinator (and the same in `spark_reserve_bytes`) for 2.7× more patches
   than can be admitted.
2. **Default video sizing and sampling do not follow the vendor path; the default rejects
   normal videos — CONFIRMED (rejection), PLAUSIBLE (vendor sizing).** `video.rs` 38–40 default
   `do_sample_frames = false` and `do_resize = false`; `minimax.rs` 122 then errors unless
   frame dimensions are multiples of 28, so 1280×720, 1920×1080 and 640×480 all fail. With
   `do_resize = true` the branch applies the image smart-resize with a 602,112-pixel budget
   (720p → 560×1008, 180 tokens per frame); MiniMax's processor rounds up to 28, caps at
   756×756 by aspect and never upscales (720p → 756×448, 108 tokens per frame). With
   sampling off every decoded frame is kept (30 fps → 15 temporal patches per second);
   MiniMax samples at 1 fps with constant-interval keep-last indices. Fix: default to 1 fps
   constant-mode sampling with keep-last, implement the vendor sizing as the default, and
   derive the `]<]x.x seconds[>[` timestamps from those indices.
3. **Encoded videos are fully decoded at source resolution and frame rate before sampling,
   under a 64 MiB budget — CONFIRMED.** `loader/media/video.rs` 25–58 runs ffmpeg with
   `-fps_mode passthrough`, no scale filter, and `limit = min(budget, frame_bytes × 768)`;
   `openai/media.rs` sets `decoded_bytes = 64 MiB`. A 720p clip is 2.76 MB per frame, so the
   decoder hits the limit at 24 frames (0.8 s at 30 fps) and the request fails with
   "decoder output exceeds byte or frame limit"; 1080p fails at 10 frames. Fix: sample and
   scale inside ffmpeg (`-vf fps=1,scale=W:H:flags=bicubic`) so the byte budget applies to
   sampled, resized frames.
4. **Per-image encode is launch-bound — PLAUSIBLE (structure confirmed, magnitude
   estimated).** `vision_minimax.cuh` 3 `MM_CHUNK = 1024`, 130–155, 173–174: a 672×672 image
   (2,304 patches) runs 3 chunks per GEMM site × ≈7 sites per layer plus pointwise kernels
   (≈45 launches × 32 layers), every GEMM writing FP32 and a separate bias/RoPE/GELU/
   residual kernel re-reading it, then a D2H copy and `cudaStreamSynchronize` per image.
   Roughly 5–10 ms of launch and latency overhead against 10–15 ms of compute on a
   5090-class card, on the GPU the language model is using. Fix: size the chunk to the
   segment or use cuBLASLt epilogues with BF16 output; batch images through the varlen
   `cu_seqlens` the attention kernel already supports.
5. **Rotary numerics emulate the Transformers BF16 path, not the vendor's FP32 path —
   PLAUSIBLE.** `vision_minimax.cuh` 84–93 rounds cos, sin and each product to BF16; the
   sglang applier rotates in FP32 and rounds once. A systematic ≈2⁻⁸ relative error on q/k
   in all 32 layers against the vendor's numerics. Fix: FP32 rotation, one BF16 rounding;
   simpler and faster as well.
6. **Video segments are gated by the image token policy, not tower capacity — CONFIRMED,
   low.** `video.rs` 94 compares each segment with `config.max_image_tokens` (the
   `--max-image-tokens` detail cap) although the tower's `max_tokens` is the real limit;
   lowering the image cap below 1,536 silently makes 720p segments unencodable.
7. **Media features make a device → host → device trip per request — CONFIRMED, low,
   not MiniMax-specific.** `vision_minimax.cuh` 173 copies to host, the embedding cache
   holds host bytes, and `RequestMedia::write_chunk` re-uploads per prefill chunk; ≈7 MB
   per image each way even on cache hits. A device-resident feature cache for the local
   placement is the fix.

No defects were found in the span and offset plumbing (placeholder shape per temporal
pair, optional timestamps, row offsets into one encoded segment, prefix-cache row
identities, `write_chunk` slicing), and ffmpeg runs off the scheduler thread in
`spawn_blocking` behind a 4-slot semaphore with protocol and format whitelists, single
threads, byte and time limits and kill-on-deadline.

Done well: the 3D RoPE (13 frequencies × T/H/W over 78 dims, 2 pass-through, merge-window
positions) and the projector → 2×2 merge → `patch_merge_mlp` order with erf-GELU match the
vendor implementation exactly; the 523-tensor contract matches the checkpoint (F32 conv
weight honoured, no post-LN); still-image preprocessing is faithful to the bundled fast
processor (banker's-rounding smart resize at 451,584 px, PyTorch uint8 antialiased bicubic
with adaptive int16 precision, hash-tested against torch 2.8, fused FP32 rescale/normalize,
the `[t,bh,bw,mh,mw,C,T,ph,pw]` patch order); validation is bounded before allocation
everywhere.
