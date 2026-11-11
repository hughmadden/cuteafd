# DSF 4.1 on 2× RTX 4090 D: six-review comparison scorecard (Phase 1, static)

11 November 2026 (Sydney). Six agents reviewed `hughmadden/cuteafd@379b26792` with SparkInfer `ce9aec54` for inference-engine, kernel and KV-cache improvements that keep accuracy. This scorecard rates those reviews against the two pinned commits. Nothing was built or run; runtime validation is Phase 2. It was scored independently of the earlier comparison report. The full report, with every rated finding and the session evidence, is the LAN page of the same date (`cuteafd-dsf41-six-review-comparison-20261111.html`).

## Scorecard

| # | Reviewer | Inference-engine optimisation | Kernel optimisation | KV cache with accuracy preserved | Estimates and evidence | Applying research | Overall |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Claude Fable 5.2 | 8.0 | 7.5 | 8.0 | 7.0 | 6.0 | **7.3** |
| 2 | Claude Opus 5.5 | 6.5 | 8.0 | 3.5 | 7.5 | 9.0 | **6.9** |
| 3 | Astra 6 | 4.0 | 6.5 | 8.5 | 7.5 | 7.0 | **6.7** |
| 4 | Kimi 3.1 Pro | 3.5 | 5.5 | 4.5 | 5.0 | 6.5 | **5.0** |
| 5 | Qwen 4 Preview Max | 3.5 | 5.0 | 3.0 | 4.5 | 4.5 | **4.1** |
| 6 | Carbon (Gemini) | 3.0 | 2.0 | 2.5 | 2.0 | 3.0 | **2.5** |

Each category is scored 0–10 from the grades of that review's findings: **A** confirmed in the pinned source, not already active, material and accuracy-safe; **B** confirmed and new but small or hedged; **C** partly right, generic or already in PLAN.md; **D** not checkable; **E** refuted, already active, or would break exactness. Research is scored from what each session actually retrieved, and estimates from traceability, arithmetic, stated confidence and stop bars. Overall is the plain average. Fable stays first under every weighting tried; Opus and Astra swap second and third depending on the weights.

## Conditions each review ran under

| Reviewer | Model and effort in the log | Team | Output tokens | Minutes to report | Pinned report |
|---|---|---|---:|---:|---|
| Claude Fable 5.2 | `claude-fable-5-2`, max | 9 agents | 1,327,949 | 58 | LAN HTML snapshot (not in git) |
| Claude Opus 5.5 | `claude-opus-5-5`, max | 3 agents | 441,220 | 42 | [21182f438](https://github.com/hughmadden/cuteafd/blob/21182f438/docs/reviews/2026-11-11-sm89-v41-flash.md) |
| Astra 6 | `gpt-6-astra`, xhigh | 1 agent | 31,435 | 16 | [5bac6cfe2](https://github.com/hughmadden/cuteafd/blob/5bac6cfe2/PLAN.md#L6570) |
| Kimi 3.1 Pro | `k3.1-pro (kimi-coding)`, max | 1 agent | 54,511 | 27 | [e57614442](https://github.com/hughmadden/cuteafd/blob/e57614442/PLAN.md#L6570) |
| Qwen 4 Preview Max | `qwen4-max-preview (qwen-token-plan)`, max | 1 agent | 50,459 | 32 | [ce1ee7404](https://github.com/hughmadden/cuteafd/blob/ce1ee7404/REPORT-sm89-dsf41-4090d.md) |
| Carbon (Gemini) | `Gemini 4 Carbon (display name only)`, High | 1 agent | 103,208 | 21 | [97bacf2a0](https://github.com/hughmadden/cuteafd/blob/97bacf2a0/docs/reviews/deepseek-v41-sm89-4090d-review.md) |

Conditions were not equal. Fable ran nine agents and Opus three; the rest ran alone. Carbon searched local agent-session history once in the minute before it was told not to, and Qwen saw the subject line of Kimi's commit before branching. Nobody started the engine. The scorer is Claude Fable 5.2, the same model as the top-ranked reviewer: verdicts are tied to quoted source lines, and the gap between first and second is within judgement error.

## Best ideas across the six reviews

Ranked by what the source confirms, how much it could matter, and how cheap the first measurement is. Nothing here was run; Phase 2 decides.

| # | Idea | Found by | Why | First measurement |
|---|---|---|---|---|
| 1 | Make partial prefix reuse exact: an exact-only switch, then an exact snapshot at the last message boundary | Astra, Fable | The one confirmed breach of the repository's exact-restore rule. The switch is a few lines because the exact branch already exists and is unit-tested. Upstream already defaults boundary snapshots for dense Qwen for this reason. | Count partial hits on real traffic. Run `golden --resume-at` for the KL of an approximate replay. Time shared-preamble first turns with exact-only. |
| 2 | Check the dense FP8 projections against the recipe, starting with the shared expert | Opus, Qwen, Fable | All 14 dense projections use the multiply the experts dropped after it changed a quarter of outputs. The shared expert is the same cascade as a routed expert and runs for every token. | Apply the experts' oracle test to `q_b`, `o_b` and the shared FFN on real weights. Any fix must cover all three dense kernels and every row band, or bits will depend on concurrency. |
| 3 | Stop the host prefix cache dropping a mark a live session needs | Fable | A confirmed chain to a full cold prefill (about 20 s at 100K tokens) once the 17.75 GiB host tier fills. The event is counted only as `prefix.fork_no_room`. | On a long-running instance read `host_cache.bytes_used`, `quota_bytes`, `host_evictions` and `prefix.fork_no_room`. A larger explicit `HOST_CACHE_BYTES` is an immediate mitigation. |
| 4 | Turn the GPU expert tier on, rank it from a route census, and give it more memory | Opus, Qwen, Fable, Carbon | The launcher leaves the tier off and `auto` ranks uniformly. Memory for it exists: 3.64 GiB per card in automatic-pool reserves and a 1.23 GiB embedding table on the tighter GPU. | C1 decode with the tier off, with `auto`, and with a census profile. The planner's model predicts about 1.4–1.7 ms per step over the real default, more if routing is skewed. Soak for memory growth before cutting reserves. |
| 5 | Move the history copy search off the critical path | Fable | Up to five scans of the whole context per round on the engine thread, for greedy requests, while both GPUs wait. Running it behind the draft is exact. | Time the search per round at 32K and 128K context. |
| 6 | Index selection at long context: one fused select, query staged once, wider loads | Opus, Fable | Selection grows from 0.37 ms per step at 32K to 1.03 ms at 128K and keeps growing. Both proposals keep the same bits. | Selection time per step at 128K before and after. The commit table gives the baseline. |
| 7 | Engram: try preload, and hoist the projection | Fable, Opus | Preload is one config key and the host has the RAM; its gain is uncertain. The projection hoist is exact and worth about 1.2% of a C1 step. | One 32K prefill with `MAPPED_SCALE_PRELOAD_GIB=190`, reading `stall_prefill_ns`. Drop it if prefill moves less than 2%. |
| 8 | L2 weight prefetch: enable it, then issue it after the shared and tier weights | Fable | Off by default; as ordered on two GPUs, the weights that follow evict it. Exact. | One C1 A/B with `L2_PREFETCH=auto`, then with the reorder. |
| 9 | TP6 over six Sparks for decode | Opus | Measured +14% at C1 and +7.5% at C16 on the PRO 6000 rig. Prefill fell 7.8%, so it needs a decode and prefill split or faster links. | One decode card and one prefill card at TP4 and at TP6 on this pair. |
| 10 | Per-layer copies and snapshot copies | Opus, Fable, Astra | Eight device copies and seven uploads per layer; 200 copies per snapshot in the normal case. All removable without changing bits. | One trace of a warmed C1 verify step: count the copy rows and the gaps between them. |
| 11 | Small exact kernel changes | Astra, Kimi, Opus | Hoist the merge's shared scalars; quantise the `q_a` and `kv` input once; stop scheduling empty expert groups; derive the 344 MiB position plane; reuse decoded keys across reuse layers. | Time each component alone and keep only those that move it by 5% or more. |

Not worth runtime time:

- Device-side reply descriptors (Kimi): already fixed in commit 0d556132e.
- Sparse attention overhaul, tensor-core index scoring, `o_b` rerouting and “enable the host cache” (Carbon): already in the code, or measured and rejected in the fork.
- Adaptive attention split count (Qwen, Carbon): changes softmax rounding and breaks the tested rule that a row's bits do not depend on batch size.
- Whole verify-step graph capture (Kimi): cut in PLAN.md; the first half of that projection measured +1.5%.
- Dynamic tier eviction: the tree's simulator prices it at about −67% net under exact restores.
- Draft cost-model recalibration (Qwen): the default policy has no cost model.

Found during verification, in none of the six reviews (static reading, unmeasured):

- The tier's gain should be measured against the real default, which already keeps about six whole expert layers on the GPUs: about 1.4–1.7 ms per step, not 3.2–3.5.
- A third dense kernel (the 17–96-row tiles) also uses the FP8 multiply, so a GEMV-only fix would make a row's bits depend on concurrency.
- The radix top-k never uses its merge scratch, yet 163,840 B per row is allocated: 430 MiB per GPU at a 2,752-row prefill unit.
- State restored after an approximate replay is saved as an ordinary snapshot and later counted as an exact hit.
- A V4.1 prompt runs as drained calls of three units. A per-call cost fits the unexplained prefill remainder as well as a per-row cost does.
- One 32 GiB threshold, written in three places, makes a 48 GB card take the 96 GB card's pool target, reserves and graph policy.
- With the tier on, lanes send a Spark wave for every routed layer, including fully resident ones, while the planner charges nothing for those layers.
- Whole-table Engram preload was merged upstream as #891 from this fork's author. If the rig already sets it, the Engram item's main lever is already in use.

## Each review: score, pros, cons

### 1. Claude Fable 5.2: 7.3

The broadest review, and the only one to cover prefill, decode scheduling, transport, memory and the prefix cache together. Twenty rated findings, four graded A and none refuted. Its weak points are at the top: the Engram preload estimate rests on an inference the code does not support, and one piece of evidence for the prefix-cache change was misread.

Scores: engine 8.0, kernel 7.5, KV cache 8.0, estimates 7.0, research 6.0. Rated findings: 4 A, 13 B, 3 C, 0 D, 0 E.

Best ideas:

- Move the per-round history copy search behind the draft (exact; up to 1.5 ms per step at 200K context).
- Fix the host prefix-cache eviction path that can force a full cold prefill.
- Fuse index selection across tiles at long context.
- Issue the L2 weight prefetch after the shared and tier weights; reclaim 3.64 GiB per card of automatic-pool reserve for the tier.

Pros:

- Found the per-round history copy search: up to five scans of the whole context on the engine thread between draft and verify. Exact to hoist, and it grows with context.
- Found the host prefix-cache failure chain: once the 17.75 GiB host tier fills, a live session can lose the mark its device entry depends on and fall to a full cold prefill.
- Got the defaults right where others did not: host cache on at 17.75 GiB, draft policy `simple` with no cost model, L2 prefetch off and ordered before the weights that evict it.
- Separated measured from modelled, defined its confidence levels, and gave a measurement order with stop bars.

Cons:

- Top-ranked item (Engram preload, “roughly +30% at 32K”): the mechanism is real, the 1.5 s attribution is not. Row supply runs on its own thread a call ahead and, at the inferred cost, would supply three times what prefill consumes.
- Misread the release card: “restores 332, a mid-snapshot hit … most tool-call turns take the approximate path”. 332 is the exact prompt-end snapshot; a partial hit would report 204. All 39 published cards took the exact path.
- Quoted the documented L2 prefetch gain on a 5090 (+2.4%) and left out the rest of the line: “no clear gain on the RTX PRO 6000”.
- Argued the tier simulator's exactness rule is stricter than the engine's contract; `AGENTS.md` at this commit supports the simulator.
- Dismissed Spark kernel headroom (“nothing to gain”) on a community tool's reading that the engine already exceeds.

### 2. Claude Opus 5.5: 6.9

The best-researched review, with a sourced time model for a decode step and clear confidence and effort labels. Two A findings (the dense FP8 accuracy risk and index scoring at long context) and several clean B findings. It missed the engine's approximate cache path while asserting the opposite, and its largest item quotes PLAN.md selectively.

Scores: engine 6.5, kernel 8.0, KV cache 3.5, estimates 7.5, research 9.0. Rated findings: 2 A, 6 B, 4 C, 2 D, 0 E.

Best ideas:

- Check and fix the dense FP8 projections with the exact F16 multiply the experts use.
- Rework the index scorer for long context (rows as extra columns, query staged once).
- Hoist the Engram projection off the critical path (exact, about 1.2% of a C1 step).
- TP6 over six Sparks for decode (+14% C1 measured on the PRO 6000 rig).

Pros:

- Fullest treatment of the dense FP8 risk: showed that the test bar is one BF16 unit wide, so it cannot see the effect, and backed it with two outside studies of the Ada accumulator.
- Index scoring at long context: byte-wise loads and a query re-decoded by every warp, with gains tabulated from 32K to 1M tokens.
- Engram projection hoist: exact, high confidence, line references correct.
- The only review to notice that the launcher leaves the GPU expert tier off, and to bring measured TP6 evidence with its prefill cost.

Cons:

- “The engine's exact-restore rule correctly does not copy that approach”: false for V4.1 mid-snapshot hits, which replay approximately and are hard-wired on.
- Head-split attention (+6.5–8% C1) is two to three times the same PLAN paragraph's own projection (+2–4%), and leaves out that the predecessor's head split measured −5 to −8%.
- “Under [the default] the Sparks take ~21.8 ms”: the default still holds about six whole expert layers on the GPUs (about 19.9 ms modelled).
- “The draft-width cost model is fitted online”: not for the default policy, which has no cost model.
- Host prefix cache called “ample”; the eviction path that forces a cold prefill was not found.

### 3. Astra 6: 6.7

The most precise review and by far the cheapest: one agent, 16 minutes, 31 thousand output tokens. Eight findings, none refuted; every size, count and line reference I checked was right. It led with the one confirmed breach of the repository's exact-restore rule. Its scope is narrow: kernels and the prefix cache.

Scores: engine 4.0, kernel 6.5, KV cache 8.5, estimates 7.5, research 7.0. Rated findings: 1 A, 6 B, 1 C, 0 D, 0 E.

Best ideas:

- Add an exact-only policy for partial prefix hits; the engine forces the approximate path on.
- Exact encoder-only snapshots at message boundaries, so exact reuse stays fast.
- Hoist the attention merge's shared scalars (180,224 exponentials for 704 values per row).
- Stop scheduling expert-kernel blocks for empty groups.

Pros:

- Led with the accuracy finding: V4.1 forces approximate partial-prefix reuse on, with no switch. Framed correctly as confirmed control flow with unmeasured task impact.
- Designed the exact fix properly: encoder-only marks at message boundaries, with the tag, the 128-token suffix rule and the sizes all correct.
- Exact counts that held up: 180,224 exponentials for 704 distinct values in the attention merge; 4,176 expert blocks scheduled where 928 work; 120 or 200 copies per snapshot.
- Best inventory of what already exists. It explicitly declined to recommend features the fork already has.

Cons:

- Replay cost of an approximate hit put at 23–26 ms by scaling bulk prefill throughput. A 128-row pass sweeps nearly every expert in each layer, so 0.13–0.17 s is more likely. The error understates the value of its own fix.
- Did not notice that the dense projections still use the FP8 multiply the experts had just dropped, although it warns against undoing the expert fix.
- No time model of a step, so no whole-system estimate and no ranking by impact.
- Nothing on the expert tier default, L2 prefetch, Engram, the host cache or decode scheduling.

### 4. Kimi 3.1 Pro: 5.0

A compact, well-organised review with an accuracy contract for every item and five sound, mostly bit-exact findings. Two of its top three priorities rest on stale premises: one targets a gap a fork commit had already closed, and its headline ceiling is a projection PLAN.md later cut.

Scores: engine 3.5, kernel 5.5, KV cache 4.5, estimates 5.0, research 6.5. Rated findings: 0 A, 5 B, 3 C, 0 D, 1 E.

Best ideas:

- Derive candidate positions instead of storing them (344 MiB per index-owning GPU).
- Overlap the FP4 key gather with QK in the attention kernel.
- Quantise the shared `q_a` and `kv` input once.
- Measure graph-shape memory before sizing the tier.

Pros:

- Candidate-position plane: 344 MiB per index-owning GPU that can be derived instead of stored.
- Async key gather overlapped with QK in the attention kernel, sized to fit the SM89 shared-memory budget and bit-exact by construction.
- `q_a` and `kv` quantise the same input twice: confirmed for the backbone as well as the drafter.
- Graph-shape census before giving more memory to the tier: V4.1 graph memory is unmeasured and unbounded above 32 GiB.

Cons:

- Priority 3 (device-side reply descriptors) is already done: commit 0d556132e made each block read only its rank's descriptor, and tried and dropped the table variant. Kimi's figures are the earlier commit's “before” numbers.
- Priority 1 (whole verify-step capture, +10–15% C1) is a 2 October projection for the PRO 6000 that includes the exchange work. That half measured +1.5%, whole-step graphs measured no gain on two other families, and PLAN.md cut the item.
- The ordered K64 router split is bit-identical only if all 80 partials per output are stored.
- Missed the approximate partial-prefix path and the dense FP8 risk.
- No time model; whole-step percentages are asserted, not derived.

### 5. Qwen 4 Preview Max: 4.1

The thinnest code reading (12 file reads) in a session slowed by rate limits. It still caught the dense FP8 accuracy risk and saw that the expert tier ranks by uniform popularity. Two items are wrong, and its headline rests on a roofline that exists in the tree only as a comment.

Scores: engine 3.5, kernel 5.0, KV cache 3.0, estimates 4.5, research 4.5. Rated findings: 1 A, 1 B, 3 C, 2 D, 2 E.

Best ideas:

- Run the experts' recipe-parity test on the dense FP8 projections.
- Rank the expert tier from a route census instead of uniform popularity.

Pros:

- Flagged the dense FP8 path as “the one place accuracy may already be silently below the recipe”, with the right test.
- Saw that `auto` ranks the expert tier by uniform popularity and that a census-profile path exists: a real configuration lever.
- Used the tier step-model constants correctly and gave a confidence label and a first measurement for each item.
- Found an outside SM89 port whose notes corroborate the fork's exact-multiply stance.

Cons:

- Adaptive attention split depth would change softmax rounding and break a tested rule that a row's bits do not depend on batch size. Its claim that the change cannot reach cached state is also wrong.
- “The adaptive-width cost model was calibrated on SM120 timings”: the default policy has no cost model and nothing is calibrated offline.
- Headline (close the Spark gap from 77% to 90%, worth 5–10%) names no mechanism; the 278 GB/s roofline is a code comment with no measurement behind it.
- No KV-cache improvement proposed; calls the prefix cache “already exact”.

### 6. Carbon (Gemini): 2.5

The most detailed-looking report: eleven items with line ranges, byte counts and microsecond savings, and no stated uncertainty. On inspection it is the least reliable. Four items are refuted, none is confirmed as stated, and many of its numbers have no source in either repository.

Scores: engine 3.0, kernel 2.0, KV cache 2.5, estimates 2.0, research 3.0. Rated findings: 0 A, 0 B, 7 C, 1 D, 4 E.

Best ideas:

- Host the embedding table to free 1.23 GiB on the tighter GPU (`EMBEDDING=host`).
- A/B the exact packed head against the FP8 head.

Pros:

- Host embedding is a real memory lever on the tighter GPU (`EMBEDDING=host` frees 1.23 GiB on GPU0), though not for the reason or by the edit Carbon gives.
- The exact packed head is a legitimate accuracy A/B under the repository's own rule, already documented as pending.
- Correctly saw that tiered lanes send a Spark wave even when nothing remote is needed.
- Quoted the SM89 probe ceilings and several commit measurements correctly.

Cons:

- Sparse attention overhaul: the vectorised loads, in-register rescale and bit-built exponents it proposes are already in the kernel; the table lookups, `ldexpf` calls and 32 KB spill it describes are not. The 5.2 and 10.3 ms per step it attributes to partial traffic exceed the measured time of the whole attention call.
- Index scorer “uses scalar FMAs instead of tensor cores”: it already uses the BF16 tensor-core multiply.
- “Enable the host cache”: it is on by default. The 32K prefill time behind “~500× faster” is about 15 times too low.
- Rerouting `o_b` to the GEMM is not bit-identical and was measured and rejected in the fork; the commit it cites (`b0ce1576`) exists in neither repository.
- Invented or mis-derived figures: snapshot mark sizes found nowhere in the tree, a per-token drafter KV that does not exist, expert timings back-computed from swapped ratios, MB written as MiB, a tenfold arithmetic slip.

## What the pinned source settled

- **Are prefix-cache restores exact?** Only at a snapshot end. A hit inside a longer snapshot is hard-wired to an approximate path: the 20 encoder windows are rebuilt by replaying 128 tokens from empty. `AGENTS.md` asks for no approximate reconstruction; `PLAN.md` documents the replay as deliberate; the README tells users the cache is exact. No measurement of the approximation is recorded. (`v41_native_serve/scheduler.rs:205,248` pass `true`; `core/prefix.rs:247-248`; `v41_backbone_cache/prefix.rs:84-105`; `AGENTS.md:92-99`)
- **Do the dense projections still use the FP8 multiply the experts dropped?** Yes: all 14 dense projections, in three kernels, take each 32-wide partial from Ada's FP8 tensor-core multiply. The expert kernel left it after it changed a quarter of outputs against the recipe. Nobody has checked the dense path the same way. (`b12x/_lib/sm89_fp8.py:6-9,310,814`; `dense_gemm.py:317-352`; kernel commit 3cd2737e)
- **Is the GPU expert tier on, and how is it ranked?** Off by default in the launcher. `auto` ranks experts uniformly by id; a route-census profile is supported but not the default. With the tier off the pair still holds about six whole expert layers, so the tier's modelled gain over the real default is about 1.4–1.7 ms per step, not the 3.2–3.5 ms the reviews quote. (`scripts/launch/run-family.sh:1070-1077`; `placement/families/deepseek_v41.rs:575-577`; commit 7f3760587)
- **Is there a measured decode rate for this rig?** No. The fork's history records prefill only (4,853–4,924 tok/s at 8K, 5,588 at 32K). Every decode percentage in every review is modelled, and several quoted kernel timings were taken on an RTX 4090, not the 4090 D. (commit 643b6aa09; no decode table in the 116 fork commits)
- **Is the host prefix cache on, and can it lose state a session needs?** On by default: `HOST_CACHE_BYTES=auto` resolves to 17.75 GiB here. When it fills it evicts least-recently-used leaves first; a device entry keeps only a host key for its mark; a missing mark means a cold prefill with no retry on an older snapshot. (`cuteafd.config:75`; `hostcache/snapshot.rs:168,388`; `engine/prefix/entry.rs:69`; `engine/prefix/cache.rs:1204`)
- **What does the SM89 attention kernel already do?** Loads are vectorised, rescaling is in registers and exponents are built from bits. Compressed layers always use ten parts, and a test requires each row's output to match its single-row launch bit for bit. The merge recomputes 180,224 exponentials for 704 distinct values per row. (`native/.../v41_sparse_attention.cu:52,207-208,439-463`; commit 8f9453a81)
- **Does the index scorer use tensor cores?** Yes, the BF16 multiply. But it loads keys, scales and queries a byte at a time and every warp re-decodes the query. Selection takes 11 calls per step at 32K context and 32 at 128K (0.37 and 1.03 ms). (`b12x/attention/dsa_indexer/_v41_overlay.py:37,107`; commit 2e7dee4e8)
- **What is the default draft policy?** `simple`: verify while calibrated acceptance stays above a fixed 0.5, with no cost model. Priced policies are opt-in. (`cuteafd-core/src/simple_draft.rs:168`; `shared/draft.rs:32-63`)
- **Is L2 weight prefetch used?** Off for V4.1. On two GPUs it is issued before the shared and tier weights, which then pass through the same cache; nothing is prefetched across the GPU hand-off. The docs record +2.4% on a 5090 and no clear gain on the PRO 6000. (`shared/l2_prefetch.rs:43-44`; `v41_backbone_lane.rs:105-107,510-513`; `docs/models/deepseek_v41.md:68`)
- **How are Engram rows read?** Copied out of a read-only file mapping unless the tables (189 GiB) are preloaded into process memory. Preload and warm are both off by default. Nothing in the tree measures the cost of first touches. (`formats/mapped_table.rs:216-217,916-918`; `run-family.sh:1056-1064`)
- **Is KV memory the constraint?** No. KV is 890 B per token (534 on GPU0, 356 on GPU1), the checkpoint's own format. The automatic pool is capped at 2M tokens, about 1.74 GiB across both cards. (`serving_capacity/deepseek.rs:146,166`; `placement/pool.rs:1-2`)
- **Was the reply-descriptor cost still open?** No. A fork commit already made each pull block read only its own rank's descriptor, and tried and dropped a device-side table. (commit 0d556132e)

## Method and limits

- **Reports.** The six pinned reports were read in full. Kimi's and Astra's are the sections appended to PLAN.md on their branches. Fable's is the HTML on the LAN, byte-identical to the archived snapshot (sha256 `4c2463e8…`).
- **Source check.** Both repositories were extracted at the pinned commits; the engine's submodule pin equals the kernel baseline. About 100 distinct claims were grouped by code area and checked by seven read-only verification passes, each told to find the code itself and quote it. I re-read the lines behind the pivotal verdicts. Commit measurements were checked against the full messages of the 116 fork commits and the kernel fork commits.
- **Sessions.** Model, effort, tool calls, web calls, sub-agents and token counts were read from the raw logs. Every session was also checked for reads outside its own workspace and for any engine start.
- **Independence.** I did not read the existing comparison report, its scorecard or its provenance files on the LAN, nor the sessions and worktrees of other comparison runs. Fable's snapshot in that folder was identified by checksum only.
- **Same-model caveat.** I am Claude Fable 5.2, the model that wrote the top-ranked review, and the verification passes ran on the same model. Guards: verdicts are tied to quoted source lines; the same grading rules were applied to every review; borderline calls on Fable's own items went against it (its Engram estimate counts against it, its half-wave idea is graded C, and its kernel, KV and estimate scores sit below the best of the others); and the overall score is an unweighted average. Treat the 0.4-point gap between first and second as within judgement error.
- **Limits.** Static only. Estimates were judged for traceability and arithmetic, not truth; Phase 2 measures them. Timings quoted from commits were often taken on an RTX 4090, not the 4090 D. Whether the compiled kernels already remove duplicated conversions cannot be told from source.
