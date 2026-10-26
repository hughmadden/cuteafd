# The shared engine

There is one engine. A model family supplies facts and hooks: its geometry,
attention, router, experts, head, speculator taps, a lane entry with its
reason, and a prefill estimate. The shared code owns the serve loop, the decode
lanes, the prefill schedule, the Spark exchange, the Spark compute owners and
admission. This page maps those shared parts to the code. Read it before you
port a family (MiniMax Flash is next) or change shared code. Why the engine is
built this way: [PLAN "One engine, shared structure"](../../PLAN.md#one-engine-shared-structure-tj-2026-10-25).
Working rules: [AGENTS.md](../../AGENTS.md).

Pointers are `file:line` at work/p0 `6bb2d5e61`. Lines drift, so each one also
names a symbol you can grep for. Short prefixes used below:

| Prefix | Path |
|---|---|
| `daemon/` | `rust/crates/cuteafd-daemon/src/` |
| `loader/` | `rust/crates/cuteafd-loader/src/` |
| `core/` | `rust/crates/cuteafd-core/src/` |
| `transport/` | `rust/crates/cuteafd-transport/src/` |

## 1. One engine, families supply facts

### The family descriptor

`Family` ([loader/plan/family.rs:113](../../rust/crates/cuteafd-loader/src/plan/family.rs#L113))
is a static, CPU-only descriptor. It is registered in `REGISTRY` (`:162`).
`cuteafd plan`, serve and golden all read the same descriptor. It states:

- **`decode_lanes() -> LaneSupport`** ([loader/plan/decode_lanes.rs:19](../../rust/crates/cuteafd-loader/src/plan/decode_lanes.rs#L19)).
  The three variants:
  - `One { reason }`: the reason is a measurement or a predicted cost, with
    its issue number. The planner prints it when `--decode-lanes 2` is
    refused (`resolve`, `:40`).
  - `OneOrTwo { default }`: resolves to one lane at concurrency 1 (`:58`).
  - `FixedTwo`: V4.1 only.

  The trait default, `DRAFTER` (`:15`), is for drafter checkpoints only. Two
  tests enforce this: `one_lane_families_record_their_reason` (`:103`) and
  `only_drafters_keep_the_default_reason` (`:116`).
- **`spark_decode() -> Option<SparkDecode { rows }>`** (`family.rs:128`): the
  rows of one decode/verify request to a Spark. `None` means the family has
  no Spark experts.
- **`prefill_limits() -> PrefillLimits`**
  ([loader/serving_capacity/prefill_waves.rs:58](../../rust/crates/cuteafd-loader/src/serving_capacity/prefill_waves.rs#L58)).
  Three parts:
  - `estimate: PrefillEstimate` (`:22`), which is one of:
    - `Timing { spark_ms, turnaround_ms, reference_ranks, reference_unit_rows, source }`:
      measured S and T.
    - `MeasuredWaves { waves, reference_ranks, larger_topology_waves, source }`:
      a measured best k, used when there is no independent S/T.
    - `Unmeasured`: two waves.
  - `initial_unit_rows`: a measured starting unit (Step-5: 2048).
  - Hard executor caps: `max_waves` (Qwen4: 2, because its mapped-PLE ring
    has two slots), `max_unit_rows`, `capacities`.
- **`spark_prefill() -> SparkPrefill`** (`family.rs:130`): `Ladder` for every
  shared-executor family. `Streams(2)` exists only for V4.1's legacy CED
  executor ([loader/plan/spark_executions.rs:36](../../rust/crates/cuteafd-loader/src/plan/spark_executions.rs#L36)).
- **`open() -> FamilyModel`** (`family.rs:55`), which carries the tensor
  contract (`accepts`, `experts`) and `placement()` (`:70`). `placement()` is
  the family's adapter over the shared solver (section 5).

Worked examples:

- GLM 5.3 / GLM Flash: [loader/plan/families/glm.rs:28](../../rust/crates/cuteafd-loader/src/plan/families/glm.rs#L28)
  (`GLM5`: `Timing`; `GLM5_FLASH`: `MeasuredWaves`).
- DeepSeek: [deepseek.rs:25](../../rust/crates/cuteafd-loader/src/plan/families/deepseek.rs#L25).
- Step-5: [step5.rs:339](../../rust/crates/cuteafd-loader/src/plan/families/step5.rs#L339).
- Kimi: [kimi/mod.rs:254](../../rust/crates/cuteafd-loader/src/plan/families/kimi/mod.rs#L254).

Every `source` string links the measurement it came from. Never put a family
name, Spark-count threshold or per-family row cap in shared code: add a
descriptor field instead.

Current entries:

| Family | LaneSupport | Prefill estimate |
|---|---|---|
| V4.1 | FixedTwo (C8 1.17×, C16 1.19×, #723) | legacy CED, `Streams(2)` |
| V4 | One (V4 Flash two-lane A/B pending) | MeasuredWaves k3 @ 6 |
| GLM 5.3 | One (second graph set ≈3 GiB/GPU) | Timing S 13 ms, T 19.5 ms @ 6, 2752 rows |
| GLM Flash | One (min-card two-lane A/B pending) | MeasuredWaves k2 @ 4, k3 above |
| MiMo | One (measured 0.66–0.93×) | legacy (private prefill, below) |
| Kimi K3 | One (predicted loss) | MeasuredWaves k3 @ 6 |
| Step-5 | One (needs its own decode link) | MeasuredWaves k4 @ 6, initial unit 2048 |
| Qwen4, dense Qwen | One (no Spark round trip) | Unmeasured, `max_waves: 2` |

### What a family must not do

- **Run a private serve loop.** Decode rounds go through `LaneSet`
  (section 2), and prefill goes through `run_prefill_units` (section 3). Each
  port adds a source-gate test that pins this; copy the pattern in
  `decode_runs_as_lane_zero_of_a_shared_lane_set`
  ([daemon/families/deepseek_v4/serve.rs:1036](../../rust/crates/cuteafd-daemon/src/families/deepseek_v4/serve.rs#L1036)).
  MiMo's prefill (`prefill_lane_count`, [daemon/families/mimo_v2/engine.rs:136](../../rust/crates/cuteafd-daemon/src/families/mimo_v2/engine.rs#L136))
  is the one remaining private loop, and its port is open (#744, #671). Do not
  copy it.
- **Own a placement policy.** A family builds a `PlacementRequest` from its
  geometry and calls the shared solver. It does not pick pool sizes, onboard
  counts or wave depths itself.
- **Leave compat shims.** A port deletes the family's old loop, workspaces and
  launch keys in the same change. It leaves no dormant path, removed-key
  diagnostic or alias. If a feature truly cannot fit the shared structure, name
  it and the reason on #744 instead of forking the path.

Qwen-shaped families (Kimi, Step-5, Qwen4, dense Qwen) go further. They
implement `Backend` ([daemon/shared/qwen/scheduler.rs:93](../../rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs#L93))
and use the shared scheduler `schedule()` (`:618`) unchanged: admission,
prefix snapshots, drafting, sampling and the console. A new family should start
there unless its request lifecycle truly differs.

## 2. Decode: lanes, commit, pacing, readiness

All of this is in [daemon/shared/serve/lanes.rs](../../rust/crates/cuteafd-daemon/src/shared/serve/lanes.rs).

- **`LaneSet`** (`:19`) holds one stop flag (`Cell<bool>`) for its lanes. All
  lanes are polled on the engine thread. Lanes share request banks through
  `RefCell`s, borrowed only for synchronous work and never across an await
  (`:3-8`).
- **`LaneSet::run_one(lane)`** (`:53`) runs the only lane of a
  `LaneSupport::One` family as lane 0. It uses the same lease, failure stop and
  pacing as a pair. Every family except V4.1 calls it once per round from its
  scheduler, for example:
  - [qwen/scheduler.rs:1039](../../rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs#L1039)
  - [glm5/serve.rs:772](../../rust/crates/cuteafd-daemon/src/families/glm5/serve.rs#L772)
  - [mimo_v2/serve.rs:934](../../rust/crates/cuteafd-daemon/src/families/mimo_v2/serve.rs#L934)
- **`LaneSet::run(first, second)`** (`:42`) runs two lane loops through
  `join` (`:62`), which uses `join!`, not `try_join!`. A failing lane sets the
  stop flag (`LaneLease::run`, `:113`), and its peer returns at its next
  completed boundary. A peer is never dropped mid-round, because it may own
  queued device or transport work. The first lane's error wins. V4.1 is the
  only two-lane user ([deepseek_v41/v41_native_serve/scheduler/independent.rs:17](../../rust/crates/cuteafd-daemon/src/families/deepseek_v41/v41_native_serve/scheduler/independent.rs#L17)).
  The scheduler also uses `stop()` to pause both lanes for admission. After
  that, `rebalance` (`:90`) and `decode_lanes::assign` (`decode_lanes.rs:66`)
  keep the lanes within one request of each other.
- **`try_map(setup)`** (`:36`) constructs each lane's owners with the lease
  that later runs the lane, so lane index = owner identity end to end. For
  example, V4.1 opens its Spark endpoint per lane with `lane.index()`
  ([v41_native_serve.rs:453](../../rust/crates/cuteafd-daemon/src/families/deepseek_v41/v41_native_serve.rs#L453)).
  The Spark worker then routes decode rows of endpoint lane *n* to decode owner
  *n* (`execution_owner`, [transport/worker_selection.rs:84](../../rust/crates/cuteafd-transport/src/worker_selection.rs#L84)).
  A second lane's bytes are a planner charge made before allocation and
  labelled per lane: see the `lane {lane} …` demands in
  [loader/placement/families/deepseek_v41/owners.rs:179](../../rust/crates/cuteafd-loader/src/placement/families/deepseek_v41/owners.rs#L179).
- **`LaneCommit` + `lanes::commit`** (`:122`, `:136`) end every round. A
  round's commit runs in four steps:
  1. `begin` enqueues both halves: the target's cache or state commit, and the
     drafter's context update.
  2. `ready` is polled, yielding to peer lanes, until both halves have drained.
  3. `publish` makes the commit visible.
  4. On any failure, `abort_and_drain` runs before storage is revoked, and the
     original error is returned.

  The four implementations show the range: V4 commits by position inside the
  round, so its commit enqueues nothing; GLM Flash replays KDA kept rows; Qwen
  accepts; V4.1 is the two-lane case. Look them up with
  `grep -n 'impl.*LaneCommit for' -r daemon/`.
- **Pacing.** `paced` (`:78`) wraps every lane future in a `Backoff`
  ([transport/wait_policy.rs:115](../../rust/crates/cuteafd-transport/src/wait_policy.rs#L115)).
  It works as follows:
  - Under `--host-wait block` (the default), a pending stretch spins for
    `BLOCK_SPIN` = 10 ms (`:92`), then sleeps `BLOCK_SLEEP` = 20 µs per poll
    (`:94`). A completed poll restarts the spin.
  - Why 10 ms: every per-layer decode wait ends inside that window, so decode
    never pays a wake-up. The earlier 250 µs window cost Kimi BF16 max C8
    0.91× (#731).
  - `spin` never sleeps.
  - Both modes park after 30 s idle (`:1-21`).

  Do not add a family-private spin or sleep: use `Backoff` or `spin_window`
  (`:103`).
- **Readiness.** Signal ready only after every fallible init step, including
  prefix-cache construction and capacity publication. MiMo once signalled ready
  before its prefix cache existed, and its first card recorded pool = 0 (#791).
  Where the rule lives:
  - Shared scheduler: `initialize(init, ready)` ([qwen/scheduler.rs:476](../../rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs#L476)).
  - Own schedulers: a single `ready();` after the last `?` and before
    `loop {` ([glm5/serve.rs:491](../../rust/crates/cuteafd-daemon/src/families/glm5/serve.rs#L491)).
  - The tests are in `readiness_tests` ([daemon/shared/serve.rs:6](../../rust/crates/cuteafd-daemon/src/shared/serve.rs#L6)).
    Add your family to them.

## 3. Prefill: continuous row units

### The executor

[daemon/shared/prefill_pipeline.rs](../../rust/crates/cuteafd-daemon/src/shared/prefill_pipeline.rs)
owns the schedule. A family implements `PrefillHooks` (`:170`). Each hook
enqueues work and returns:

- `route(layer)`: `Local` (dense or RTX-resident experts, finished inside one
  hook) or `Spark` (a wave sent in one hook and landed in a later one).
- `begin_unit`, `attention`, `local_ffn`.
- `send_experts -> Ticket`, then `land_experts(ticket)`.
- `end_unit`, then `commit_unit`, called in unit order.

The family then calls `run_prefill_units(hooks, units, waves)` (`:339`). The
`prefill_schedule` (`:264`) is pure, and the CPU tests check it. It enforces:

- unit k+1 attends layer L only after unit k attended L;
- waves land in dispatch order, and units commit in order;
- a newly admitted unit is served before older units land again, so the Sparks
  never drain between units.

Each in-flight unit owns its slot's slice (`unit.slot`) of every buffer that
lives across a hook. Hook-local scratch is shared. `prefill_units` (`:216`)
cuts the span into equal, 64-row-aligned units in multiples of `waves`. A span
that fits one unit splits only if each wave keeps `MIN_SPLIT_ROWS` = 1024
(`:206`).

Reference ports:

- Kimi (the first): `UnitHooks` at [kimi_k3/layers.rs:708](../../rust/crates/cuteafd-daemon/src/families/kimi_k3/layers.rs#L708).
- GLM 5.3: [glm5/engine.rs:2402](../../rust/crates/cuteafd-daemon/src/families/glm5/engine.rs#L2402).
- V4: [deepseek_v4/engine/prefill.rs:128](../../rust/crates/cuteafd-daemon/src/families/deepseek_v4/engine/prefill.rs#L128).
- Step-5, Qwen4 and GLM Flash also implement it.

The ordering, row-ownership and head rules are in
[PLAN "Prefill contract"](../../PLAN.md#prefill-contract-tj-2026-10-23).

### Waves and the 8192-row reserve

This is all in [loader/serving_capacity/prefill_waves.rs](../../rust/crates/cuteafd-loader/src/serving_capacity/prefill_waves.rs).

- **The reserve and units:**
  - One fixed `RESERVE_ROWS` = 8192 (`:6`) for every family. Units are
    reserve / k, on exact export capacities: k = 2/3/4 gives 4096 / 2752 /
    2048 rows (`unit_rows`, `:11`).
  - Three 2752-row units round up to an 8256-row span, which is why the cap
    is `RESERVE_ROWS + 64`.
  - `MAX_WAVES` = 4 (`:7`).
- **Choosing k:**
  - `PrefillEstimate::waves` (`:43`) scales the descriptor's S to the deployed
    Spark count (S = S_ref × N_ref / N) and calls `waves_for` (`:218`), which
    computes **k = ⌈T/S⌉ + 1, clamped to 2..4**.
  - `MeasuredWaves` returns its measured k, or `larger_topology_waves` above
    its reference count.
  - Zero Sparks means one wave.
- **Fitting:** `prefill_shape` (`:136`) lowers an automatic k until the shape
  fits the admitted arena and the executor caps. An explicit pin
  (`PREFILL_WAVES`/`PREFILL_ROWS`) never moves; it either fits or fails with a
  typed `PrefillError` (`:81`).
- **Short prompts:** `waves_used` (`:226`) uses fewer waves when a prompt cannot
  keep each unit at least `MIN_FILL` = 50% full. Storage is not reallocated and
  graphs are not recaptured.

**What waves assume of kernels.** A unit's Spark time must scale roughly
linearly with its rows across the whole 2048–4096 band. If a kernel has a
per-row cliff, larger units become silently slower, and that shows up as a wave
regression rather than a kernel problem.

The example that exposed this was Step-5's BF16-input FP8 kernel. It switched
GEMV→GEMM above 2048 rows, costing 33–54% more per row on GB10, so 3 × 2752
measured slower than 3 × 2048 ([#762 audit](https://github.com/tpurtell/cuteafd/issues/762#issuecomment-6413296191)).
The guard is the kernel row-curve rule in
[AGENTS.md "Build and run"](../../AGENTS.md#build-and-run). If a kernel keeps a
cliff, add a descriptor preference such as `initial_unit_rows` that avoids the
bad unit.

### Deployment contract: units, span, slots

`cuteafd plan --deployment` publishes:

- `prefill_unit_rows`, `prefill_waves`, `prefill_slots`, `prefill_span_rows`
  and `decode_lanes` (`DeploymentSpec`, [loader/plan/deployment.rs:67](../../rust/crates/cuteafd-loader/src/plan/deployment.rs#L67));
- typed topology, roles and admission (the same struct).

The launcher reads those fields back in `release_prefill_shape`
([scripts/launch/run-family.sh:402](../../scripts/launch/run-family.sh#L402)).
It refuses a plan that names no positive unit, wave or slot count. The family's mode
comes from `family_prefill_mode` (`:395`):

| Mode | Families | Serve flags |
|---|---|---|
| `units` | step5, kimi_k3, qwen4, deepseek_v4 | `--prefill-rows` (admitted unit), `--prefill-waves` (starting depth), `--prefill-slots` |
| `units-span` | glm5, glm5_flash | the above plus `--prefill-span-rows`: the rows every per-slot plane spans. Static: unit × waves; adaptive: `reserve_rows(slots, unit)` (`IntakeSpan`, `prefill_waves.rs:89`). |
| `legacy` | mimo_v2 (V4.1 launches through `run.sh`) | not on the shared executor |

A new family opts in here once its serve CLI accepts the shape.

**`--prefill-slots` and runtime depth (#762).** The deployment publishes
`prefill_slots`, and the launcher passes `--prefill-slots` to every `units`
and `units-span` family; every units serve takes it as `PrefillOverrides.slots`:

- **slots == waves** pins the depth (static families, and any user
  `PREFILL_WAVES`, which makes the plan admit only its own waves).
- **slots > waves** is an adaptive family (GLM 5 and Kimi today; Step,
  GLM Flash and V4 to follow). It admits four slots of the widest unit the
  reserve allows, `ADAPTIVE_UNIT_ROWS` = reserve / 2 = 4096 rows
  (`prefill_waves.rs:134`), bounded by the compiled capacity. Its plan names
  that unit as `prefill_unit_rows`, starts at the estimate's depth cut from it
  (`wave_control::unit_for`; 3 × 2752 at six Sparks), and spans
  `reserve_rows(4, 4096)` = 8256 rows
  ([loader/serving_capacity/wave_control.rs:49](../../rust/crates/cuteafd-loader/src/serving_capacity/wave_control.rs#L49)).
  The controller can then run any depth from 2 to 4.
- **At each decode→prefill transition** the engine reports the previous call's
  Spark pace to its `PrefillDepth`
  ([daemon/shared/prefill_depth.rs:19](../../rust/crates/cuteafd-daemon/src/shared/prefill_depth.rs#L19)).
  The pace is busy and idle time per unit, measured on the Spark prefill owner
  and returned in the reply header, and it is read without a host wait. The
  engine then takes the next depth, re-cuts the prefill link's return slots at
  rest, and brackets the call. GLM 5 does this in `start_prefill`
  ([daemon/families/glm5/engine.rs:1368](../../rust/crates/cuteafd-daemon/src/families/glm5/engine.rs#L1368)),
  Kimi in `RoutedLane::start_prefill`
  ([daemon/families/kimi_k3/spark.rs:140](../../rust/crates/cuteafd-daemon/src/families/kimi_k3/spark.rs#L140)).
- **One call** is `call_rows` (`wave_control.rs:57`): the smallest
  k × unit over the candidate depths, 8192 rows at U = 4096. Every depth
  therefore cuts whole units.
- **Storage never changes.** Every depth's slots are views at
  `slot * unit_for(k)` inside per-field arenas spanning the reserve. Hook-local
  scratch is shared by the slots on a GPU (GLM 5: one step's program and top-k
  scratch, LM head, logits rows, EXLR8 execution and native dense scratch).
  Per-slot owners (peer exchange, TP2 route slots, commit blocks, KV tables,
  Spark owners) cover the admitted slots, and the planner charges exactly that.
- **Goldens** pin the starting depth at its own cut.

## 4. Spark exchange and typed owners

### Coordinator side: one exchange per link

[daemon/shared/spark_waves.rs](../../rust/crates/cuteafd-daemon/src/shared/spark_waves.rs)
(#731) gives each link one persistent transport worker. The worker owns the
link's QPs and carries every wave, decode and prefill alike. The GPU-issuing
thread never waits on it. The two meet through monotone u32 counters
(`staged`, `landed`, `consumed`) in pinned, device-mapped memory:

- the stream writes `staged` after its copies;
- `Waves::land` (`:816`) queues a stream wait on `landed`;
- the reduce writes `consumed`.

Wave k uses slot k mod waves, and a slot's next wave waits for its previous
one to land. All staging copies around these waits are SM copies through the
mapped alias, never `cudaMemcpyAsync` (the module doc, `:1-64`, explains why).

On a transport error, the worker records a terminal health failure and
releases every queued wait. Every consumer checks `healthy()` (`:125`) after
its sampling sync.

Links are `SparkLink::new_waves(..., waves, wave_rows)`
([daemon/shared/spark_intake.rs:1474](../../rust/crates/cuteafd-daemon/src/shared/spark_intake.rs#L1474)).
Every request and reply is one registered RC SEND, with no chunking and no
credit beyond ring depth ([PLAN, "Frames"](../../PLAN.md#prefill-executor-api-shared-2026-10-23)).

### Spark side: typed compute owners

`SparkExecutions` ([core/spark_executions.rs:44](../../rust/crates/cuteafd-core/src/spark_executions.rs#L44))
holds one small **decode owner per lane** (`DecodeExecution`, `:16`) and
exactly **one prefill owner per Spark** (`PrefillExecution`, `:25`):

- **Decode owners** are sized from `spark_decode().rows` at the covering export.
- **The prefill owner** has scratch at the export covering the admitted unit:
  - `buffered_units` = k retains each unit's own I/O, but scratch and the
    stream are not multiplied;
  - `MAX_BUFFERED_PREFILL_UNITS` = 4 (`:10`).

Plan, serve and golden all build owners through `admit_family`
([loader/plan/spark_executions.rs:113](../../rust/crates/cuteafd-loader/src/plan/spark_executions.rs#L113)).
The worker's backend export ladder (`SparkBackend::exports`, `:81`) fixes the
covering export. An uncovered capacity is a typed `ExecutionAdmissionError`,
not a fallback.

**FIFO units and credit.** On the worker
([daemon/shared/experts/service/local.rs](../../rust/crates/cuteafd-daemon/src/shared/experts/service/local.rs)):

- `execution_owner` (`:719`) maps every request to one owner. Prefill and
  benchmark rows go to the prefill owner; decode and MTP-verify rows go to
  `Decode(lane)`. A request that mixes owners is rejected.
- `OwnerQueues` (`:739`) publishes GPU completion in owner FIFO order.
- A completed unit keeps its I/O and its admission credit until its reply is
  released (`release`, `:783`).
- Decode lanes never borrow prefill credit, or each other's (tests at `:975`,
  `:998`, `:1036`).

**Completion: decode stream vs prefill ordering events.** `PendingUnits::for_owner`
([execution/timing.rs:49](../../rust/crates/cuteafd-daemon/src/shared/experts/execution/timing.rs#L49))
does the following:

- A decode owner holds one unit (`SingleFlight`), so a stream query is its
  completion fence.
- The prefill owner records one ordering-only event per queued unit
  (`Event::ordering`, `:18`, created without timing). An earlier unit's reply
  therefore never waits for later work on the same stream.
- Timing-enabled events exist only where profiling publishes elapsed times.

**Packed route staging.**

- Each queued unit packs its route ids and weights into its own pinned staging
  slot, which is uploaded with one contiguous H2D
  ([service/routes.rs](../../rust/crates/cuteafd-daemon/src/shared/experts/service/routes.rs),
  `pack` `:34`, `upload` `:51`).
- The slot stays owned until that unit's fence.
- Do not copy routes as strided D2D copies out of mapped request records: on
  GB10 that cost 1062 µs per 2752 × top-8 unit, against 7 µs packed
  (`f2f05dfa0`).

**Ingress dtype.** Each endpoint has a typed `SparkIngress`
([core/expert_geometry.rs:57](../../rust/crates/cuteafd-core/src/expert_geometry.rs#L57)):
`Fp8K32`, `Bf16`, or `Bf16Decode { rows }` (BF16 decode rows and FP8 K32
prefill on one QP). Two checks keep the wire and the kernels in agreement:

- The worker checks the installed packages against the endpoint before serving
  (`validate_ingress`, [service/backend.rs:132](../../rust/crates/cuteafd-daemon/src/shared/experts/service/backend.rs#L132);
  `supports_ingress`, `:498`).
- Each request's header dtype must match the bound kernel's native input
  (`require_input_dtype`, [transport/expert.rs:322](../../rust/crates/cuteafd-transport/src/expert.rs#L322)).

So per-layer native input already works on the wire. **Landing (#802):**
`SparkIngress::Native` lets each layer use its package's native input. Step
uses FP8 K32 on its FP8 layers and BF16 on layers 88–90, with fail-closed
package validation. Send the checkpoint's native activation format; don't
convert on the coordinator to fit an endpoint
([AGENTS.md "Engineering principles"](../../AGENTS.md#engineering-principles)).

## 5. Admission and the planner

- **Plan == runtime.** One CPU-only solver serves every family
  ([loader/placement/mod.rs:1](../../rust/crates/cuteafd-loader/src/placement/mod.rs#L1),
  `solve` in `solve.rs:85`). The planner (`FamilyModel::placement`,
  `family.rs:70`) and the runtime admission build the same `PlacementRequest`
  from the same family inputs. They differ only in the GPU `Baseline`
  (`mod.rs:145`): `Planned` charges context and loaded weights, and `Measured`
  is one CUDA sample. Spark owners use the same `admit_family`. A planner that
  disagrees with the runtime is a bug in one of them; don't add a margin.
- **Typed charges.**
  - Every demand has a `Category` ([core/memory_layout.rs:15](../../rust/crates/cuteafd-core/src/memory_layout.rs#L15)):
    `Weights`, `Kv`, `Workspace`, `Transport` and so on, plus a group label
    (for example `lane 1 backbone`).
  - The runtime allocation ledger uses the same labels, so a prediction can be
    compared with what was allocated (`compare`, `:327`). The ready report
    logs it (`memory_report::log("ready")`).
  - Weight and workspace admission happen before any allocation.
- **Pool first.** `PoolPolicy` ([loader/placement/pool.rs:11](../../rust/crates/cuteafd-loader/src/placement/pool.rs#L11))
  targets a 2M-token KV pool on cards above 32 GiB and 1M at 32 GiB or less.
  The solver's order (`mod.rs:7-10`) is:
  1. fixed demands (graphs, workspaces) and mandatory movables;
  2. the pool target;
  3. RTX expert layers in what is left.
- **Onboarding.** `Onboard::Auto` (`mod.rs:425`) is the default:
  - onboarding is on in every shipped config;
  - explicit `N`, `N%` and `all` are pins;
  - a zero-onboard config is only a labelled measurement arm, never a shipped
    default;
  - `max` (experts-first) is legacy, being removed with the shared planner
    (#776): don't use it for a new family.
- **The 30 s budget.** Every full-plan test runs through `BudgetedPlanInput`
  ([loader/plan/testing.rs:12](../../rust/crates/cuteafd-loader/src/plan/testing.rs#L12)).
  It allows 30 s per plan (`:31`) and prints phase timings, so a planner that
  grows slow fails a CPU test, not a launch.
- **`cuteafd plan --deployment`** (cli flags at [daemon/cli.rs:244](../../rust/crates/cuteafd-daemon/src/cli.rs#L244))
  emits the typed launch decisions the launcher consumes: shapes, lanes,
  topology, roles and the memory layout. `cuteafd plan MODEL` without it tells
  you which tensors, formats and kernels a new checkpoint still needs.

## 6. Gates for a new family

These are the gates; the tiers and bars are in
[AGENTS.md "Engineering rules"](../../AGENTS.md#engineering-rules).

1. **CPU gate.** Run `scripts/dev/cpu-gate --task NAME` while iterating, and
   `--full` before the merge into work/p0.
   - `--full` adds the installed deployment sweep,
     `installed_example_deployments_publish_executable_prefill_shapes`
     ([daemon/tests/budget_cli.rs:220](../../rust/crates/cuteafd-daemon/tests/budget_cli.rs#L220)).
     It plans every `examples/configs/*.config` (40 today) and runs it through
     `release_prefill_shape`.
   - Add your family's min and max configs there. Link them from the family
     doc and from `examples/configs/README.md`;
     `scripts/tests/test_example_configs.py` checks both.
   - Add your family to the lane and readiness source gates (sections 1 and 2).
2. **Golden byte-exactness.** Run your family's golden subcommand through
   `scripts/bench/golden-run.py`. Unit N=1 vs N must be byte-identical (logits
   and KV). Families whose Spark exports use FP32 atomics gate on KL instead
   ([PLAN "Prefill contract", gates](../../PLAN.md#prefill-contract-tj-2026-10-23)).
3. **Exact prefix cache.**
   - The bench's `cache_exact` check ([rust/crates/cuteafd-bench/src/baseline.rs:319](../../rust/crates/cuteafd-bench/src/baseline.rs#L319))
     covers prompt-end and turn-end restores, which must be byte-identical.
   - `scripts/qualify/qualify-prefix-cache.py` covers multi-turn traffic, the
     torture phase and leak accounting.
4. **Lossless speculation.** The `spec_lossless_set` panel
   ([cuteafd-bench/src/panels/lossless.rs](../../rust/crates/cuteafd-bench/src/panels/lossless.rs))
   must pass 10/10: identical ids and row hashes with drafts on and off, with
   no tie tolerance.
5. **One wip-cards pair.** One launch per arm, candidate against work/p0,
   through `scripts/bench/wip-cards.py` ([docs/wip-cards.md](../wip-cards.md)):
   - panels `baseline` and `spec_lossless_set`;
   - C1/C8/C16;
   - 8K and 64K prefill.

   Speed ratios must be ≥ 0.95, and the port must also match the family's
   absolute prefill reference
   ([PLAN "Prefill contract"](../../PLAN.md#prefill-contract-tj-2026-10-23)).
6. **New kernels.** These need three checks:
   - `scripts/build/compare-sm-exports.py --sms 188,170` (CPU-only; review
     object and cubin differences);
   - the row-curve check across the 2048/2752/4096 band (section 3);
   - for kernel code, landing on `sparkinfer-glmrt` master first, then a pin
     bump.
7. **One SM120 build for PRO 6000 and 5090.** Don't bake in an SM count or L2
   size. Query them at runtime with `NativeLibrary::sm_count` and
   `l2_cache_bytes` ([cuteafd-ffi/src/shared/l2_prefetch.rs:8](../../rust/crates/cuteafd-ffi/src/shared/l2_prefetch.rs#L8))
   and size grids from them, as `Fp8QuantizeGrid` does
   ([daemon/shared/launch_grid.rs:18](../../rust/crates/cuteafd-daemon/src/shared/launch_grid.rs#L18)).
   The SM188/170 export comparison is the CPU-side check that a build doesn't
   depend on one card.

## 7. Porting a new family: checklist

1. Run `cuteafd plan MODEL` on the checkpoint. Read what it says is missing
   (tensors, formats, kernels) before you write kernel code.
2. Add the descriptor (`loader/plan/families/<family>.rs`) and register it in
   `REGISTRY`. It needs:
   - `detect` and `open` from the checkpoint's own `config.json` and headers;
   - `decode_lanes`: `One { reason }` until a two-lane A/B wins, and the reason
     must cite the issue;
   - `spark_decode` rows;
   - `prefill_limits`: `MeasuredWaves`, `Timing` or `Unmeasured`, with
     `source`;
   - `spark_prefill() = Ladder`.
3. Write the placement adapter: a `PlacementRequest` builder over the shared
   solver (`loader/placement/families/`). Charge every owner with a category
   and group before allocation, and don't run a private pool or onboard policy.
4. Write the coordinator programs and expert packages. Use the AOT exports at
   the exact capacities (1/16/80/256/1024/2048/2752/4096 for EXL3), and the
   checkpoint's native numerics and ingress.
5. Prefill: implement `PrefillHooks` and call `run_prefill_units`. Each unit
   owns its slot slices. Opt the family into `family_prefill_mode` in
   `run-family.sh`.
6. Decode:
   - implement `Backend` for the shared Qwen scheduler, or call
     `LaneSet::run_one(decode_round)` from your scheduler;
   - end every round with `lanes::commit` on a `LaneCommit`;
   - use `Backoff` for all waits;
   - signal readiness after all fallible init.
7. Spark experts: use the shared exchange (`SparkLink::new_waves`). Owners come
   from `admit_family`, and route staging uses the shared packed path.
8. Add source gates (only `run_one` and `lanes::commit` in the round; readiness
   ordering), min/max example configs, and family-doc links.
9. Run the gates: `cpu-gate --full`, golden N=1 vs N, exact cache, lossless set,
   one wip-cards pair, and, for kernels, SM188/170 plus row curves.
10. Delete what the shared path replaced. If something cannot fit the shared
    structure, write the reason on #744 rather than forking.
