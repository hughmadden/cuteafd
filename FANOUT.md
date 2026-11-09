# Bounded graph cache family rollout

The shared controller is `shared/decode_graph/device.rs`; typed executable
ownership stays in `shared/decode_graph/bank.rs`. GLM 5.3 is the reference
integration in `families/glm5/engine.rs`. Do not copy its model geometry into
shared policy.

## Family inputs

- Supply a `Site` name, default `Cycle`, `Preference`, bootstrap byte estimate,
  device-wide `Basis`, and explicit `spans_devices`. CUDA memcpy inspection
  cannot discover peer pointers embedded in arbitrary kernel arguments.
- Attach every bank on one physical device to the same reserve and basis.
  Node rates must match across banks. Use calibrated architecture/family node
  and executable coefficients only for their measured route; otherwise select
  bytes for the entire device and warn. An unknown node count runs eagerly;
  never mix byte estimates into a node pool.
- Before each decode/verify/prefill/wave boundary, call `cache_cycle` with the
  actual cycle. Pinned hits accumulate locally without controller locking or
  ARC-list mutation; changing cycles flushes the previous cycle's hits.
- Supply an exact typed key and retained executable/resource owner. On a miss,
  use `cache_miss`; Eager means execute normally, never reject the request.
  Cancel a ticket on every failed capture path. After successful instantiate,
  report native nodes/width/span and insert the owner before launching it.
- Supply a checked-in versioned prewarm list with stable geometry and explicit
  pin flags, plus `reserve_base_bytes` and `reserve_extra_bytes`. Prewarm is
  best effort, not a seal. First-touch eagerly before sampling capture growth.
  Attribute startup activity to Other rather than decode/verify frequency.
- Admission owns the full configured graph reserve once, before KV sizing.
  Measured warmup does not replace or shrink that budget. GPU_RESERVE cannot
  remove it. A scalar GPU_GRAPH_RESERVE applies independently to each GPU;
  gpuN selectors override individual coordinator ranks. Zero runs eagerly.

## Measurement and lifetime rules

Queue the entire multi-rank step before synchronizing any rank. Peer segments
may wait for work not yet queued: never synchronize around an individual
capture, instantiate or segment. Sample each physical device separately at the
completed-step boundary, then report only that device's observed growth.
Drain every consumer before destroying a selected executable or its resources.
Error paths use the same submission-drain lifetime proof.

Historical node/executable peaks advance only for successful surviving owners
at a completed measurement boundary, never when reserving a ticket. Cancelled
captures retain pending growth until sampling because a failed instantiate can
still grow driver storage. Destruction usually frees zero visible bytes; the
measured graph charge is permanent high-water, while ARC frees live capacity.
Chain reuse discounts apply only to qualified single-device chain captures.
Non-chain and explicitly peer-spanning sites reserve each capture separately.

Estimates select capture or eager and victims; measured growth decides charge.
Underestimation logs an overshoot and prevents additional growth, not request
refusal. Do not claim an arbitrary native allocation is hard-bounded by its
estimate. Check both warmed residency and reserve plus explicit capture
headroom on the family's qualification run.

## Trace and prewarm refresh

Launch the normal card plus concurrency-variance sweep with GRAPH_TRACE set
to an absolute host path whose parent exists. The launcher writes one
`<stem>.device-N.json` snapshot per physical device. Preserve the trace beside
the card results. It contains schema_version, cache charge/reserve/basis/pools,
site measurement classes, and identities with count, per-cycle counts,
first-seen time, native maximum ready-frontier width and device-span flag.

Generate a proposal with:

```sh
python3 scripts/dev/graph-prewarm-from-trace.py --family glm5 \
  --trace /absolute/trace.device-0.json --output /absolute/proposal.json
```

The generator aggregates segment identities into geometry frequency, excludes
startup Other counts when per-cycle counts exist, takes the maximum across
overlapping snapshots/devices, selects requested frequency coverage, and pins
a hot minimum-parameter geometry. The base is the maximum measured physical
graph high-water, rounded to 2 MiB; extra is one largest measured capture,
separately rounded. Review the proposal, update the family's checked-in list,
and relaunch before accepting it. Extend the parser/consumer for each new
family rather than pretending its keys are GLM keys.

The GLM reference list comes from the trace6 ordinary EXL3 K4 1-RTX/4-Spark
baseline plus C1/2/4/8/16 sweep: 28 geometries cover every one of 151,522
request-cycle graph events (startup Other excluded), with 1x16 pinned. The
measured base is 1,189,085,184 bytes and one-capture headroom is 2,097,152 bytes.
All 7,742 inspected executables were width-one and single-device. A final
fixed-DRAFT_LENGTH=4 dev/candidate C1/C16 relaunch must qualify this list and
reserve before acceptance; the trace is a sizing receipt, not that final gate.

## Delete the old family policy

Remove startup sealing, declared-key request refusals, per-bank lazy budgets,
and old census replacement of admission graph owners from the integrated
family. Retain any declared-key census used solely for provenance/bootstrap,
clearly separate from the fixed reserve. Shared legacy APIs remain only for
families not yet migrated; do not remove another family's working path in a
mechanical rollout.

## Reconciliation and Spark status

GLM publishes memory_reconciliation schema_version=1 with Ready and
first_request stage objects. Each device row carries measured_bytes,
charged_bytes excluding release padding, padding_bytes, owners, provenance,
measurement_method and unavailable_reason, plus three signed byte fields:

- `delta_vs_admitted`: observer-excluded CUDA used minus all admitted serving
  Growth owners (padding excluded), including lazy Workspace/Staging,
  allowances and derived tails. Negative values can be unused admitted capacity.
- `graph_reserve_slack`: admitted graph reserve minus measured graph arena
  high-water; negative means the arena exceeded its reserve. The component
  values are `graph_reserve_bytes` and `graph_arena_high_water_bytes`.
- `delta_vs_measured`: delta_vs_admitted plus graph_reserve_slack, replacing the
  admitted graph budget with actual measured high-water. This is the
  accounting-error signal, not the admitted-owner delta.

`beyond_half_padding` is true only when abs(delta_vs_measured) exceeds half
padding_bytes. A negative admitted delta exactly equal to unused graph reserve
is not drift. Missing physical samples yield null deltas/drift. Missing arena
telemetry for a nonzero graph owner yields null measured delta/slack/drift and
an explicit graph_measurement_unavailable_reason, never an assumed zero.
Devices with no graph owner and no controller have zero arena/slack.
RTX rows retain raw_used_bytes for audit and report observer_bytes plus
observer_provenance (the measured sampler context). The same observer exclusion
as cuda_allocatable_memory_info applies exactly once before either delta; absent
observers contribute zero. Device-wide CUDA use is not process attribution.
Ready still includes unused admitted Growth owners, so a negative Ready delta
need not be an accounting defect. GB10 admission MemAvailable/allocatable samples
remain separately labelled.

Spark Ready aggregation uses the reconciled worker resident/workspace and UMA
host-owner ledger, not the coordinator's empty placeholder plan. Device/pinned
detail aliases are not charged twice. Its physical sample comes from the existing
admission detail's optional memory_reconciliation object. New coordinators accept missing and unknown
optional detail fields; coordinator and Spark release images are a compatible
pair, not arbitrary old/new binary combinations. Post-request Spark samples
at the coordinator are null with the #1631 reason until control-plane telemetry
exists. Worker first-request logs include their own raw CUDA sample and a
separately labelled ledger-only residual, not a fabricated planner delta.
Recorder integration for cards/sweep.json is owned by a separate task.

The live native Spark worker routes through service/backend.rs Execution
launch/flush paths, not the dormant capture/replay methods in experts/execution.rs.
There are no live worker graph sites to attach today. SPARK_GRAPH_RESERVE is
forwarded but has no active worker graph owner to enforce; a family enabling
worker capture must add admission ownership and this controller together.

## Qualification checklist

Run controller, bank ownership, capture cancellation, basis mismatch, cycle
attribution, optional Spark detail and signed-delta tests; workspace all-target
check; CUDA-ON all-target native build; default CPU gate. Measure a real card's
no-refusal/no-OOM behavior, eager/eviction counters, measured graph high-water
versus reserve/headroom, Ready/first-request deltas, and same-day warmed C1/C16
speed with thermal conditions. Report missing telemetry explicitly. Preserve
logs/results and stop every task-owned container before releasing its ticket.
