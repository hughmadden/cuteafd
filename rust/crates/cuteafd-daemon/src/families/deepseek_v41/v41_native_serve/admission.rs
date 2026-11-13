//! V4.1 pre-weight facts for the shared planner; execution consumes its one admission.
pub(crate) use crate::shared::memory::budget::LocalLayers;
use anyhow::{Context, Result, ensure};
use cuteafd_loader::placement::*;

fn spark_model(catalog: &cuteafd_loader::OfficialV41Catalog, capacity: u32, waves: u8, bf16: bool)
    -> Result<cuteafd_transport::worker_selection::ModelAdmission> {
    let decode = super::DECODE_LANE_ROWS;
    use cuteafd_core::{SparkEndpoint, SparkIngress};
    use cuteafd_loader::plan::spark_executions::{admit_family, SparkBackend};
    let ingress = if bf16 { SparkIngress::Bf16 } else { SparkIngress::Fp8K32 };
    // The shared ladder: the unit's covering export, `waves` units buffered.
    let executions = admit_family("deepseek_v41", &SparkBackend::for_catalog(catalog)?.exports(), None,
        u64::from(capacity), u64::from(waves))?;
    // The two decode lanes' links at the decode width and the one prefill
    // link at the unit (its `waves` units are the prefill owner's buffered units).
    let model = cuteafd_transport::worker_selection::ModelAdmission::new(
        vec![SparkEndpoint { ring_depth: None, capacity: decode.min(capacity), ingress, partial_reply_bytes: None },
             SparkEndpoint { ring_depth: None, capacity: decode.min(capacity), ingress, partial_reply_bytes: None },
             SparkEndpoint { ring_depth: None, capacity, ingress, partial_reply_bytes: None }],
        crate::shared::placement::spark_rings(64 << 20)?, executions)?;
    model.validate(cuteafd_core::ExpertGeometry::DEEPSEEK_V41)?;
    Ok(model)
}

/// The Spark admission (headers only), before any worker is contacted.
pub(super) fn plan_sparks(args: &crate::cli::NativeServeArgs,
    catalog: &cuteafd_loader::OfficialV41Catalog, local_layers: usize,
    capacity: u32)
    -> Result<Option<crate::shared::placement::SparkAdmission>> {
    let geometry = catalog.routed_experts().geometry()?;
    ensure!(local_layers <= geometry.layers as usize, "V4.1 local boundary exceeds checkpoint");
    if local_layers == geometry.layers as usize { return Ok(None); }
    if args.peers.is_empty() {
        ensure!(local_layers == geometry.layers as usize, "V4.1 remote layers require Spark peers");
        return Ok(None);
    }
    let slots = super::AdmittedPrefill::resolve(args)?.waves;
    spark_admission(&cuteafd_loader::plan::checkpoint::CheckpointView::select(
        &args.snapshot, cuteafd_loader::plan::checkpoint::CoreView::Original)?, catalog, local_layers, args.peers.len(), capacity, slots).map(Some)
}

/// The Spark admission for `ranks` workers serving layers `local_layers..`,
/// the decode lanes' and prefill link's endpoints at `capacity` (the unit),
/// `slots` prefill units buffered: what serve (and the GPU fixtures) start
/// before any transport connects.
pub(crate) fn spark_admission(view: &cuteafd_loader::plan::checkpoint::CheckpointView, catalog: &cuteafd_loader::OfficialV41Catalog,
    local_layers: usize, ranks: usize, capacity: u32, slots: usize) -> Result<crate::shared::placement::SparkAdmission> {
    let geometry = catalog.routed_experts().geometry()?;
    let working = SparkWorkingSet { spark_layers: (local_layers..geometry.layers as usize).collect(),
        rtx_layers: (0..local_layers).collect(), spark_ranks: ranks, prefill_lanes: slots as u64 };
    crate::shared::placement::SparkAdmission::plan_view(view, geometry, Some(catalog), &working, ranks,
        spark_model(catalog, capacity, u8::try_from(slots)?, catalog.nvfp4().is_some())?, None)
}

#[cfg(test)]
mod spark_tests {
    use super::*;

    #[test]
    fn planner_and_serve_admit_identical_spark_owners() -> Result<()> {
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_v41_snapshot(root.path());
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let backend = cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(&catalog)?;
        for live in [256u32, 2048, 4096] {
            for waves in [1u8, 2] {
                // The planner's resolved unit and waves, as serve admits them.
                let options = cuteafd_loader::plan::layout::LayoutOptions { prefill_unit_rows: Some(u64::from(live)),
                    prefill_waves: u64::from(waves), ..Default::default() };
                assert_eq!(spark_model(&catalog, live, waves, false)?.executions,
                    cuteafd_loader::plan::layout::planned_spark_executions("deepseek_v41", backend, None, &options)?);
            }
        }
        Ok(())
    }

    /// The GPU fixture's admission (endpoints at the 64-row verify width, two
    /// 64-row prefill units) passes the same validation the workers run.
    #[test]
    fn the_fixture_admission_shape_validates() -> Result<()> {
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_v41_snapshot(root.path());
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let model = spark_model(&catalog, cuteafd_core::V41_VERIFY_ROWS as u32, 2, false)?;
        model.validate(cuteafd_core::ExpertGeometry::DEEPSEEK_V41)?;
        assert!(spark_model(&catalog, 16, 2, false).is_err(),
            "16-row endpoints cannot admit the 64-row decode owners");
        Ok(())
    }

    #[test]
    fn endpoint_contract_covers_the_decode_and_prefill_links() -> Result<()> {
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_v41_snapshot(root.path());
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        for (live, compiled) in [(256, 256), (2048, 4096), (4096, 4096)] {
            for bf16 in [false, true] {
                let model = spark_model(&catalog, live, 2, bf16)?;
                assert_eq!((model.capacity(), model.compiled_capacity()), (live, compiled));
                // Two decode lanes, one prefill owner holding the two units in flight.
                assert_eq!(model.executions.decode, [cuteafd_core::DecodeExecution {
                    capacity: cuteafd_core::V41_VERIFY_ROWS as u32, compiled_capacity: cuteafd_core::V41_VERIFY_ROWS as u32 }; 2]);
                assert_eq!(model.executions.prefill, cuteafd_core::PrefillExecution { capacity: live,
                    compiled_capacity: compiled, unit_rows: live, buffered_units: 2 });
                // The decode lanes' links at the decode width (P3), the prefill link at the unit.
                let ingress = if bf16 { cuteafd_core::SparkIngress::Bf16 } else { cuteafd_core::SparkIngress::Fp8K32 };
                assert_eq!(model.endpoints.iter().map(|e| (e.capacity, e.ingress)).collect::<Vec<_>>(),
                    [((cuteafd_core::V41_VERIFY_ROWS as u32).min(live), ingress), ((cuteafd_core::V41_VERIFY_ROWS as u32).min(live), ingress), (live, ingress)]);
            }
        }
        Ok(())
    }
}

pub(super) fn embedding_placement(
    explicit: Option<crate::shared::memory::EmbedPlacement>,
    total: usize,
) -> crate::shared::memory::EmbedPlacement {
    let placement = explicit.unwrap_or(if total <= 32usize << 30 {
        crate::shared::memory::EmbedPlacement::Host
    } else {
        crate::shared::memory::EmbedPlacement::Gpu
    });
    cuteafd_bench::context::set_resolved("embedding-placement", match placement {
        crate::shared::memory::EmbedPlacement::Host => "host",
        crate::shared::memory::EmbedPlacement::Gpu => "gpu",
    });
    placement
}

pub(super) fn measured_pool_memory(lib: &cuteafd_ffi::NativeLibrary) -> Result<(usize, usize)> {
    // The inventory is the raw CUDA observation. Graph/runtime growth and
    // allocator margins are explicit solver demands, not fabricated occupancy.
    Ok(lib.cuda_memory_info()?)
}

pub(super) fn startup_phase(
    lib: &cuteafd_ffi::NativeLibrary,
    phase: &str,
    device_total: usize,
) -> Result<()> {
    if std::env::var("CUTEAFD_STARTUP_MEMORY_PROBE").as_deref() == Ok("1") {
        crate::shared::inventory::startup_probe(lib, &[lib.cuda_get_device()?], phase, Default::default())?;
    }
    if device_total <= 32usize << 30 {
        let (free, total) = lib.cuda_memory_info()?;
        let device = lib.cuda_get_device()?;
        let ledger = cuteafd_ffi::memory_ledger::snapshot();
        let space = cuteafd_ffi::memory_ledger::Space::Device;
        tracing::info!(
            phase,
            occupied_bytes = total - free,
            tracked_live_bytes = ledger.total(space, device),
            tracked_peak_bytes = ledger.peak.get(&(space, device)).copied().unwrap_or(0),
            "V4.1 small-card startup phase high-water ledger"
        );
    }
    Ok(())
}

/// Runs `build` with its device allocations owned by `"{scope}/{component}"`
/// (ledger owner tag and default category). Category scopes nested inside
/// (`sampler`, `workspace/fp8-linear`) keep the owner, so the plan == runtime
/// check sums every byte the component allocates and reports each component.
pub(super) fn scoped<T>(scope: &str, component: &str, build: impl FnOnce() -> Result<T>) -> Result<T> {
    let label = format!("{scope}/{component}");
    let _owner = cuteafd_ffi::memory_ledger::owner(&label);
    let _scope = cuteafd_ffi::memory_ledger::scope_owned(&label);
    build()
}

/// Plan == runtime (WP6): the decode lanes' and prefill slots' device
/// allocations on each of `gpus` (their ledger owners) are exactly what the
/// planner charges (`NativePrograms::target`). A retained temporary or an
/// unmodelled owner fails startup instead of silently shrinking the pool.
pub(super) fn verify_target_charges(lib: &cuteafd_ffi::NativeLibrary, native_lib: &std::path::Path, gpus: &[i32],
    decode_rows: u32, reserve: u64, unit: u32, slots: usize, concurrency: u32) -> Result<()> {
    use cuteafd_ffi::memory_ledger::{snapshot, Space};
    let manifest = families::deepseek_v41::owners::native_manifest(native_lib, "V41_FP8_AOT.json")?;
    let programs = families::deepseek_v41::owners::NativePrograms::new(&manifest, u32::try_from(lib.sm_count()?)?)?
        .with_compressed_attention(crate::shared::inventory::device_arch(lib, gpus[0])?.v41_compressed_attention());
    let head_mode = crate::families::deepseek_v41::v41_tensors::fp8_head();
    let window_batch = gpus.len() == 1 && super::BackboneExecution::window_batch_enabled();
    let ranks = gpus.len();
    let lanes = programs.decode_lanes(u64::from(decode_rows), ranks, head_mode, window_batch)?;
    let prefill = programs.prefill_slots(reserve, u64::from(unit), ranks, slots, u64::from(concurrency),
        head_mode, window_batch)?;
    let ledger = snapshot();
    for (rank, &gpu) in gpus.iter().enumerate() {
        let owners = ledger.by_owner(Space::Device, gpu);
        for (scope, demands, prefix) in [(super::DECODE_LANES_SCOPE, &lanes, "lane "), (super::PREFILL_SLOTS_SCOPE, &prefill, "prefill slots ")] {
            // Live bytes per component (`scoped`), and the plan's per group
            // (decode lanes' groups summed over both lanes).
            let mut live = std::collections::BTreeMap::<String, u64>::new();
            // Engram rows and gates are charged apart but one owner allocates both on two RTX.
            let component = |name: &str| if name.starts_with("engram") { "engram".to_string() } else { name.to_string() };
            for (label, &bytes) in owners.iter().filter(|(label, _)| label.starts_with(scope)) {
                let name = label.strip_prefix(scope).unwrap_or(label).trim_start_matches('/');
                *live.entry(if name.is_empty() { "(unscoped)".into() } else { component(name) }).or_default() += bytes as u64;
            }
            let mut plan = std::collections::BTreeMap::<String, u64>::new();
            for demand in demands.iter().filter(|d| usize::from(d.gpu) == rank) {
                let group = demand.group.strip_prefix(prefix).unwrap_or(&demand.group);
                let group = if prefix == "lane " { group.split_once(' ').map_or(group, |(_, g)| g) } else { group };
                // The decode lanes' window batch table belongs to their producers.
                let group = if group == "window batch table" { "cache producers" } else { group };
                *plan.entry(component(group)).or_default() += demand.bytes;
            }
            let (live_total, plan_total) = (live.values().sum::<u64>(), plan.values().sum::<u64>());
            let differing: Vec<_> = live.keys().chain(plan.keys()).collect::<std::collections::BTreeSet<_>>().into_iter()
                .filter_map(|group| { let (l, p) = (live.get(group).copied().unwrap_or(0), plan.get(group).copied().unwrap_or(0));
                    (l != p).then(|| format!("{group}: live {l} plan {p} ({:+})", l as i64 - p as i64)) }).collect();
            tracing::info!(gpu, scope, live = live_total, plan = plan_total, ?differing, "V4.1 target owners: plan == runtime");
            ensure!(live_total == plan_total && differing.is_empty(),
                "V4.1 {scope} on GPU {gpu}: allocated {live_total} B, planned {plan_total} B ({:+} B); {}",
                live_total as i64 - plan_total as i64, differing.join("; "));
        }
    }
    Ok(())
}

/// Plan == runtime for the coordinator expert workspaces (WP6): the planner's
/// charge (`expert_inventory`: slot arenas at the unit, decode lanes at their
/// width) equals what serve admits and allocates (`exact`, from the native
/// library's own plans and the TP2 owner policy).
pub(super) fn verify_expert_charges(catalog: &cuteafd_loader::OfficialV41Catalog, native_lib: &std::path::Path,
    sms: u32, ranks: usize, unit: u32, slots: usize, reserve: u64, exact: u64) -> Result<()> {
    let directory = native_lib.parent().context("native library has no directory")?;
    let programs = [directory.join("PROGRAMS.json"), directory.join("../share/PROGRAMS.json"),
        directory.join("../../share/PROGRAMS.json")].into_iter().find(|path| path.is_file())
        .context("missing PROGRAMS.json beside the native library or in its share directory")?;
    let (_, planned) = families::deepseek_v41::expert_inventory(catalog, &programs, u64::from(unit), ranks, sms,
        slots, reserve, u64::from(super::DECODE_LANE_ROWS))?;
    tracing::info!(planned, exact, ranks, unit, slots, reserve, "V4.1 expert workspaces: plan == runtime");
    ensure!(planned == exact, "V4.1 expert workspaces: planned {planned} B per GPU, serve admits {exact} B ({:+} B)",
        exact as i64 - planned as i64);
    Ok(())
}

/// Startup admission runs before any weight payload or device owner.
/// Loading-only owners are charged at their sequential phase, not retained in the KV ledger.
fn validate_owner_modes(args: &crate::cli::NativeServeArgs) -> Result<()> {
    ensure!(!args.tp2_attention,
        "unsupported V4.1 --tp2-attention: pre-weight contract is missing owner facts for replicated attention weights and per-lane workspaces");
    ensure!(!(args.tp2_query_projection || args.tp2_output_projection) || args.rtx_gpus == 2,
        "V4.1 TP2 query/output projections require two RTX GPUs");
    Ok(())
}

/// `preweight_context` is each rank's sampled context plus selected modules,
/// excluding ledger-tracked allocations. Module bytes are a subset, not additive.
pub(super) fn admit_startup(
    lib: &cuteafd_ffi::NativeLibrary,
    catalog: &cuteafd_loader::OfficialV41Catalog,
    args: &crate::cli::NativeServeArgs,
    preweight_context: &[u64],
) -> Result<Admitted> {
    use super::*;
    validate_owner_modes(args)?;
    let (_, total) = lib.cuda_memory_info()?;
    let coordinator_arch = crate::shared::inventory::device_arch(lib, 0)?;
    let ranks = args.rtx_gpus as usize;
    let capacity = cuteafd_core::coordinator_programs::v41_live_rows(args.prefill_batch_tokens)
        .context("invalid V4.1 prefill batch")?;
    use cuteafd_core::memory_layout::{Basis, Category};
    use families::deepseek_v41::{owners, resident};
    let config = serde_json::from_slice(&std::fs::read(catalog.snapshot().join("config.json"))?)?;
    // The checkpoint must still be a supported native backbone; its retained
    // device weights are charged as the exact arenas serving allocates.
    resident::resident_headers(&config, catalog.tensors().iter().map(|tensor| &tensor.metadata), ranks, false)
        .context("unsupported V4.1 native backbone residency")?;
    let gpu_embedding = embedding_placement(args.embedding_placement, total) == crate::shared::memory::EmbedPlacement::Gpu;
    let target_layout = crate::families::deepseek_v41::weight_layout::target_layout_for_arch(catalog,
        (ranks == 2).then(crate::families::deepseek_v41::v41_backbone_cache::CachePlacement::encoder_decoder),
        gpu_embedding, crate::families::deepseek_v41::v41_tensors::fp8_head(),
        ranks == 2 && args.tp2_query_projection, ranks == 2 && args.tp2_output_projection, coordinator_arch)?;
    let weights = owners::target_arena_demands(&target_layout);
    let manifest = owners::native_manifest(&args.native_lib, "V41_FP8_AOT.json")?;
    let programs = owners::NativePrograms::new(&manifest, u32::try_from(lib.sm_count()?)?)?
        .with_compressed_attention(coordinator_arch.v41_compressed_attention())
        .with_projection_tp2(args.tp2_query_projection, args.tp2_output_projection);
    // Every owner serve allocates: two decode lanes, the prefill slots' storage.
    let shape = super::AdmittedPrefill::resolve(args)?;
    let passes = shape.waves;
    let mut target = programs.target(u64::from(super::DECODE_LANE_ROWS), shape.reserve_rows() as u64,
        u64::from(capacity), ranks, passes, u64::from(args.concurrency),
        crate::families::deepseek_v41::v41_tensors::fp8_head(),
        ranks == 1 && BackboneExecution::window_batch_enabled())?;
    if ranks == 1 { validate_engram_rows(&target, shape.reserve_rows(), super::DECODE_LANE_ROWS as usize)?; }
    // Test builds retain the materialized query reference buffer; production
    // intentionally has no such owner.
    #[cfg(test)]
    for lane in 0..2 {
        target.push(Demand::new(0, Category::Workspace,
            format!("lane {lane} test query reference"), u64::from(capacity) * 65536, Basis::Formula));
    }
    let mut startup = if ranks == 1 { vec![owners::vocabulary_load_phase(&target_layout.ranks[0],
        crate::families::deepseek_v41::v41_tensors::fp8_head())?] } else { Vec::new() };
    ensure!(args.resolved_prefix_entries() <= 128, "invalid retained-turn limit");
    let slots = super::prefix::Arenas::slots_for(args) as u64;
    let mut target_owners = owners::TargetOwners::new(weights, target, ranks,
        owners::transports_bytes(u64::from(super::DECODE_LANE_ROWS), u64::from(capacity), args.peers.len(), passes, cuteafd_core::SparkReplyFormat::resolve(cuteafd_core::spark_reply::requested().prefill, false))?, slots)?;
    let draft = if args.dspark {
        let mut draft = if ranks == 1 {
            crate::families::deepseek_v41::v41_experts::dspark::DsparkWeights::single_inventory(
                lib, catalog, capacity, args.concurrency, if args.draft_length.draft_length.unwrap_or(5) > 5 { 7 } else { 5 })?
        } else {
            crate::shared::memory::device::Device { library: lib, id: 1 }.run(||
                crate::families::deepseek_v41::v41_experts::dspark::DsparkWeights::distributed_inventory(
                    lib, catalog, capacity, args.concurrency, if args.draft_length.draft_length.unwrap_or(5) > 5 { 7 } else { 5 },
                    args.tp2_dspark_experts, 64640, (slots * 3 * 128 * 528) as usize))?
        };
        draft.marks = slots * 3 * 128 * 528;
        startup.extend(target_owners.draft_load_phases(&draft)?);
        Some(draft)
    } else { None };
    let mut weight_arenas = target_layout.ranks.clone();
    if local_vision_owner(cuteafd_api::openai::vision_input_enabled(), args.vision_peers.is_some()) {
        // The tower's arena (root, logical claim, tail) and its execution
        // buffers are separate owners allocated together on its thread.
        let arena = cuteafd_loader::placement::families::deepseek_v41::weight_layout::vision_layout_from_headers(
            catalog.tensors().iter().map(|tensor| &tensor.metadata))?;
        let device = crate::families::deepseek_v41::v41_vision::VisionRuntime::device_bytes(catalog, 9216)? as u64;
        let scratch = device.checked_sub(arena.bytes() as u64).context("vision scratch underflow")?;
        target_owners.add_vision(owners::arena_demand(0, &arena))?;
        target_owners.add_vision(Demand::new(0, Category::Workspace, "vision scratch", scratch, Basis::Formula))?;
        weight_arenas.push(arena);
    }
    let projection_owners = programs.projection_tp2_owners(u64::from(super::DECODE_LANE_ROWS),
        u64::from(capacity), ranks, args.tp2_query_projection, args.tp2_output_projection)?;
    let fixed_demands = target_owners.fixed;
    let fixed = fixed_demands.iter().try_fold(0u64, |n, d|
        n.checked_add(d.bytes).context("startup fixed-owner overflow"))?;
    let expert_rows = cuteafd_core::coordinator_programs::v41_aot_rows(capacity)
        .context("invalid local expert capacity")?;
    let experts = expert_demands(lib, catalog, expert_rows, ranks)?;
    let workspace = if ranks == 1 { local_workspace(lib, catalog, args, expert_rows)? } else {
        let lane = |rows, unit| crate::families::deepseek_v41::v41_experts::tp2_ffn::Storage::storage_bytes(lib, rows, unit, catalog.nvfp4().is_some());
        (lane(shape.reserve_rows() as u32, capacity)? + 2 * lane(super::DECODE_LANE_ROWS, super::DECODE_LANE_ROWS)?) as u64
    };
    let tier = TierAsk::of(args, catalog)?;
    if tier.is_some() {
        ensure!(ranks == 2, "the GPU expert tier requires two RTX GPUs");
        // The shared tier recipe adds its private remapping tables once.
    }
    // Resolve both GPUs' half-projection native plans before observing the
    // pre-weight context, just as expert geometry queries do below.
    for (enabled, kind) in [(args.tp2_query_projection,
        crate::families::deepseek_v41::v41_projection_tp2::Kind::QueryB),
        (args.tp2_output_projection, crate::families::deepseek_v41::v41_projection_tp2::Kind::OutputB)] {
        if !enabled { continue; }
        for gpu in 0..ranks {
            crate::shared::memory::device::Device { library: lib, id: gpu as i32 }.run(|| {
                for rows in [super::DECODE_LANE_ROWS, capacity] {
                    crate::families::deepseek_v41::v41_projection_tp2::Storage::storage_bytes(lib, kind, rows, 0)?;
                }
                Ok(())
            })?;
        }
    }
    // Native geometry queries may initialize modules. Observe that CUDA cost
    // only after every query, still before payloads or retained device owners.
    let mut memory = Vec::new();
    for gpu in 0..ranks {
        lib.cuda_set_device(gpu as i32)?;
        memory.push(measured_pool_memory(lib)?);
    }
    lib.cuda_set_device(0)?;
    let pool = resolve_with_owners(args, &memory, experts, workspace, &vec![0; ranks], usize::from(ranks == 2 && tier.is_none()),
        &fixed_demands, &weight_arenas, &startup, draft.as_ref(), preweight_context, crate::shared::inventory::device_arch(lib, 0)?, &projection_owners)
        .with_context(|| format!("V4.1 startup refused before weights: fixed owners {fixed}"))?;
    tracing::info!(fixed_owner_bytes=fixed, admitted_cache_bytes=?pool.cache_bytes,
        host_ledger=?pool.host_ledger, "V4.1 single allocation-owner startup admission before weight loads");
    Ok(pool)
}

/// The Engram upload planes are formula owners, separate from the gates:
/// the prefill slots' one shared row plane over the `reserve` rows (each
/// in-flight unit's rows a window of it, never unit scratch) and each decode
/// lane's private planes at `decode_rows`, all on the gate GPU (GPU0).
fn validate_engram_rows(target: &[Demand], reserve: usize, decode_rows: usize) -> Result<()> {
    use cuteafd_core::memory_layout::Basis;
    let row_bytes = cuteafd_loader::ENGRAM_DEVICE_ROW_BYTES as u64;
    let rows = target.iter().filter(|d| d.group.ends_with(" engram rows")).collect::<Vec<_>>();
    let slots = rows.iter().filter(|d| d.group == "prefill slots engram rows").collect::<Vec<_>>();
    let lanes = rows.iter().filter(|d| d.group.starts_with("lane ")).collect::<Vec<_>>();
    ensure!(rows.iter().all(|d| d.gpu == 0 && d.basis == Basis::Formula)
        && rows.len() == 3 && slots.len() == 1 && lanes.len() == 2
        && slots[0].bytes == reserve as u64 * row_bytes
        && lanes.iter().all(|d| d.bytes == decode_rows as u64 * row_bytes),
        "V4.1 Engram upload admission differs from the reserve row plane and the decode lanes' rows: {:?}",
        rows.iter().map(|d| (d.group.as_str(), d.gpu, d.bytes)).collect::<Vec<_>>());
    Ok(())
}

// Remote vision reserves its tower on the Spark, never in the RTX startup bound.
fn local_vision_owner(enabled: bool, remote: bool) -> bool {
    enabled && !remote
}

#[cfg(test)]
mod startup_tests {
    use super::families::deepseek_v41::owners;
    /// The planner's own target list (decode lanes + prefill slot set) passes
    /// the Engram check at every pinned depth, and the check still refuses a
    /// missing, mis-sized or misplaced owner.
    #[test]
    fn engram_row_admission_matches_the_slot_set_and_decode_lanes() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let manifest = cuteafd_loader::plan::testing::write_v41_workspaces(dir.path());
        let fp8: serde_json::Value = serde_json::from_slice(&std::fs::read(
            manifest.parent().unwrap().join("V41_FP8_AOT.json"))?)?;
        let programs = super::families::deepseek_v41::owners::NativePrograms::new(&fp8, 188)?;
        for k in 2..=4u64 {
            let unit = cuteafd_loader::serving_capacity::prefill_waves::unit_rows(k);
            let reserve = (k * unit) as usize;
            let target = programs.target(owners::DECODE_LANE_ROWS, reserve as u64, unit, 1, k as usize, 16,
                super::families::deepseek_v41::resident::Fp8Head::Off, true)?;
            super::validate_engram_rows(&target, reserve, owners::DECODE_LANE_ROWS as usize)?;
            assert!(super::validate_engram_rows(&target, reserve - 64, owners::DECODE_LANE_ROWS as usize).is_err(), "reserve mismatch");
            let missing: Vec<_> = target.iter().filter(|d| d.group != "lane 1 engram rows").cloned().collect();
            assert!(super::validate_engram_rows(&missing, reserve, owners::DECODE_LANE_ROWS as usize).is_err(), "missing decode lane");
            let mut moved = target.clone();
            moved.iter_mut().find(|d| d.group == "prefill slots engram rows").unwrap().gpu = 1;
            assert!(super::validate_engram_rows(&moved, reserve, owners::DECODE_LANE_ROWS as usize).is_err(), "off the gate GPU");
        }
        Ok(())
    }

    #[test]
    fn verifier_extent_matches_heads_sparse_admission_and_planner() -> anyhow::Result<()> {
        use crate::families::deepseek_v41::graph_policy;
        use crate::families::deepseek_v41::v41_target_head::{
            TargetHeadWave, distributed_target::DistributedTargetHead,
        };
        let rows = cuteafd_core::V41_VERIFY_ROWS;
        assert_eq!(rows, 64);
        assert_eq!(graph_policy::MAX_DECODE_ROWS as usize, rows);
        assert_eq!(cuteafd_core::MAX_DSPARK_PROPOSALS, 7);
        let sparse_growth = cuteafd_ffi::V41SparseAttention::split_scratch_bytes(rows, 10)?
            + cuteafd_ffi::V41SparseAttention::batch_descriptor_bytes(rows)?
            - cuteafd_ffi::V41SparseAttention::split_scratch_bytes(48, 10)?
            - cuteafd_ffi::V41SparseAttention::batch_descriptor_bytes(48)?;
        let single = 2
            * (TargetHeadWave::device_bytes(rows)? - TargetHeadWave::device_bytes(48)?
                + sparse_growth);
        assert_eq!(
            single as u64,
            cuteafd_loader::placement::families::deepseek_v41::verify_workspace_growth(1, 0)
        );
        let wide = DistributedTargetHead::device_bytes(rows, 64640)?;
        let narrow = DistributedTargetHead::device_bytes(48, 64640)?;
        for rank in 0..2 {
            assert_eq!(
                (2 * (wide[rank] - narrow[rank] + sparse_growth)) as u64,
                cuteafd_loader::placement::families::deepseek_v41::verify_workspace_growth(2, rank)
            );
        }
        Ok(())
    }

    #[test]
    fn startup_vision_bound_only_charges_enabled_rtx_fallback() {
        assert!(super::local_vision_owner(true, false));
        assert!(!super::local_vision_owner(true, true));
        assert!(!super::local_vision_owner(false, false));
        assert!(!super::local_vision_owner(false, true));
    }
    #[test]
    fn embedding_profile_preserves_explicit_overrides_and_pro_default() {
        use crate::shared::memory::EmbedPlacement::{Gpu, Host};
        for total in [31usize << 30, 32usize << 30, 96usize << 30] {
            assert_eq!(super::embedding_placement(Some(Gpu), total), Gpu);
            assert_eq!(super::embedding_placement(Some(Host), total), Host);
            assert_eq!(
                super::embedding_placement(None, total),
                if total <= 32usize << 30 { Host } else { Gpu }
            );
        }
    }
}

/// One shared decision before cache or conditional routed weights are allocated.
/// `memory` is the context/module inventory less subsequent live owners.
pub(super) struct Admitted {
    pub served_context: u32,
    pub expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    pub pages: [usize; 4],
    pub cache_bytes: Vec<usize>,
    pub global_bytes: usize,
    pub reservation_bytes: Vec<usize>,
    pub placement: Placement,
    pub host: families::deepseek_v41::admission::HostFacts,
    pub host_ledger: cuteafd_core::memory_layout::HostLayout,
    /// The admitted plan, kept so the Ready boundary can verify its predicted
    /// runtime-base owners against the device without re-solving.
    pub solved: cuteafd_loader::placement::planner::ResolvedPlan,
    /// The checkpoint header the plan was admitted from, kept for the pre-pool
    /// measurement's pool re-fit (never re-read from disk).
    pub config: serde_json::Value,
    /// The pool policy's ceiling, exactly the value `prepare` bounded the
    /// solve with.
    pub maximum_pool_tokens: u64,
}

pub(super) fn local_workspace(
    lib: &cuteafd_ffi::NativeLibrary,
    catalog: &cuteafd_loader::OfficialV41Catalog,
    args: &crate::cli::NativeServeArgs,
    rows: u32,
) -> Result<u64> {
    if args.rtx_expert_layers == LocalLayers::Count(0) {
        return Ok(0);
    }
    use crate::families::deepseek_v41::v41_experts::local::LocalExpertWave;
    let lane = |rows: u32| LocalExpertWave::device_bytes_for(lib, rows, catalog.nvfp4().is_some());
    // One local expert wave per prefill slot at the unit, one per decode lane
    // at decode width.
    let slots = super::AdmittedPrefill::resolve(args)?.waves;
    Ok((lane(rows)?.checked_mul(slots).context("local serving lanes overflow")?
        + 2 * lane(super::DECODE_LANE_ROWS)?) as u64)
}

/// With `draft`, the sample must precede all draft weights, runtime buffers and
/// the draft-mark arena. `None` is for samples that already contain those owners.
#[cfg(test)]
pub(super) fn resolve(
    args: &crate::cli::NativeServeArgs,
    memory: &[(usize, usize)],
    experts: Vec<ExpertCost>,
    workspace: u64,
    deferred: &[u64],
    minimum: usize,
    draft: Option<&families::deepseek_v41::DraftInventory>,
) -> Result<Admitted> {
    resolve_with_owners(args, memory, experts, workspace, deferred, minimum, &[], &[], &[], draft, &[], cuteafd_core::GpuArch::Sm120, &[])
}

/// Fixture view of serving's logical resident-weight cap. Physical arena
/// padding and driver tails remain separate owners in the memory plan.
#[cfg(test)]
pub(crate) struct WeightCap {
    limit: u64,
    fixed: Vec<u64>,
}

#[cfg(test)]
fn fixture_target(args: &crate::cli::NativeServeArgs, catalog: &cuteafd_loader::OfficialV41Catalog,
    gpus: usize, total: usize) -> Result<families::deepseek_v41::weight_layout::TargetLayout> {
    crate::families::deepseek_v41::weight_layout::target_layout_for_arch(catalog,
        (gpus == 2).then(crate::families::deepseek_v41::v41_backbone_cache::CachePlacement::encoder_decoder),
        embedding_placement(args.embedding_placement, total) == crate::shared::memory::EmbedPlacement::Gpu,
        crate::families::deepseek_v41::v41_tensors::fp8_head(),
        gpus == 2 && args.tp2_query_projection, gpus == 2 && args.tp2_output_projection, cuteafd_core::GpuArch::Sm89)
}

#[cfg(test)]
impl WeightCap {
    pub(super) fn loaded(args: &crate::cli::NativeServeArgs, catalog: &cuteafd_loader::OfficialV41Catalog,
        gpus: usize, total: usize, draft: Option<&families::deepseek_v41::DraftInventory>) -> Result<Self> {
        let layout = fixture_target(args, catalog, gpus, total)?;
        let mut fixed = layout.ranks.iter().map(|arena| arena.payload_bytes() as u64).collect::<Vec<_>>();
        if let Some(draft) = draft {
            let home = gpus - 1;
            let payload = draft.resident.checked_sub(draft.dense.as_ref().map_or(0, |arena| arena.padding_bytes() as u64))
                .context("draft arena padding exceeds resident weights")?;
            fixed[home] = fixed[home].checked_add(payload).context("draft resident weights overflow")?;
            if gpus == 2 { fixed[0] = fixed[0].checked_add(draft.peer_half).context("draft peer weights overflow")?; }
        }
        Ok(Self { limit: crate::shared::memory::budget::resident_weight_limit().0, fixed })
    }
}

/// `--gpu-expert-tier` as serving asks the solver for it: the planner's own
/// [`families::deepseek_v41::TierChoice`] over the checkpoint's routed shape,
/// so serving admits the resident set `cuteafd plan --layout` shows.
pub(crate) struct TierAsk {
    choice: families::deepseek_v41::TierChoice,
}

impl TierAsk {
    /// `None` when the tier is off (a profile without one is refused).
    pub(crate) fn of(args: &crate::cli::NativeServeArgs, _catalog: &cuteafd_loader::OfficialV41Catalog)
        -> Result<Option<Self>> {
        let mut choice = families::deepseek_v41::TierChoice { expert_reply_reduce_tier: false, setting: args.gpu_expert_tier,
            profile: args.gpu_expert_tier_profile.clone(), model: TierStepModel::default() };
        if !choice.setting.enabled() {
            ensure!(choice.profile.is_none(), "--gpu-expert-tier-profile needs --gpu-expert-tier auto or N");
            return Ok(None);
        }
        // Shape/profile validation belongs to the shared recipe, once.
        let manifest = families::deepseek_v41::owners::native_manifest(&args.native_lib, "PROGRAMS.json")?;
        ensure!(manifest["programs"].is_array(), "PROGRAMS.json has no program list");
        choice.expert_reply_reduce_tier = families::deepseek_v41::TierChoice::expert_reply_reduce_tier_available(&manifest);
        Ok(Some(Self { choice }))
    }
}

/// `startup` contains full sequential peaks above `memory`, never staging-only
/// deltas. Draft owners supplied separately must not also appear in `owners`.
/// `preweight_context` holds the measured pre-weight occupancy per rank.
/// Empty uses the supplied availability sample, also as one named owner.
///
/// One typed request over the sampled owners: the same component contract the
/// offline planner admits decides the pool and the local expert prefix, and
/// execution consumes that placement without sampling memory again.
pub(super) fn resolve_with_owners(
    args: &crate::cli::NativeServeArgs,
    memory: &[(usize, usize)],
    experts: Vec<ExpertCost>,
    workspace: u64,
    deferred: &[u64],
    minimum: usize,
    owners: &[Demand],
    weight_arenas: &[cuteafd_loader::serving_capacity::WeightArenaLayout],
    startup: &[Demand],
    draft: Option<&families::deepseek_v41::DraftInventory>,
    preweight_context: &[u64],
    coordinator_arch: cuteafd_core::GpuArch,
    additional_owners: &[families::deepseek_v41::admission::Owner],
) -> Result<Admitted> {
    let catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    let tier = TierAsk::of(args, &catalog)?;
    let limit = crate::shared::memory::budget::resident_weight_limit().0;
    resolve_tiered_owners(args, memory, experts, workspace, deferred, minimum, owners,
        weight_arenas, startup, draft, preweight_context, tier.as_ref(), limit, coordinator_arch, additional_owners)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn resolve_tiered(args: &crate::cli::NativeServeArgs, memory: &[(usize, usize)],
    experts: Vec<ExpertCost>, workspace: u64, deferred: &[u64], minimum: usize,
    owners: &[Demand], startup: &[Demand], draft: Option<&families::deepseek_v41::DraftInventory>,
    tier: Option<&TierAsk>, cap: Option<&WeightCap>) -> Result<Admitted> {
    let catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    let layout = fixture_target(args, &catalog, memory.len(), memory[0].1)?;
    let mut fixed = families::deepseek_v41::owners::target_arena_demands(&layout);
    fixed.extend_from_slice(owners);
    resolve_tiered_owners(args, memory, experts, workspace, deferred, minimum, &fixed, &layout.ranks, startup,
        draft, &[], tier, cap.map_or(u64::MAX, |cap| cap.limit), cuteafd_core::GpuArch::Sm89, &[])
}

#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_tiered_owners(
    args: &crate::cli::NativeServeArgs,
    memory: &[(usize, usize)],
    experts: Vec<ExpertCost>,
    workspace: u64,
    deferred: &[u64],
    minimum: usize,
    owners: &[Demand],
    weight_arenas: &[cuteafd_loader::serving_capacity::WeightArenaLayout],
    startup: &[Demand],
    draft: Option<&families::deepseek_v41::DraftInventory>,
    preweight_context: &[u64],
    tier: Option<&TierAsk>,
    weight_limit: u64,
    coordinator_arch: cuteafd_core::GpuArch,
    additional_owners: &[families::deepseek_v41::admission::Owner],
) -> Result<Admitted> {
    use families::deepseek_v41::admission::{
        self, live_budgets, production_request, Owner, ProductionInput,
    };
    use cuteafd_core::memory_layout::Category;
    use cuteafd_loader::placement::planner::{Operation, PrefillShape};
    cuteafd_core::concurrency::validate("deepseek_v41", u64::from(args.concurrency))?;
    let ranks = memory.len();
    ensure!(
        matches!(ranks, 1 | 2) && deferred.len() == ranks && minimum <= experts.len()
            && (preweight_context.is_empty() || preweight_context.len() == ranks),
        "invalid V4.1 admission inventory"
    );
    ensure!(
        (0..=1_048_576).contains(&args.max_context_tokens),
        "invalid V4.1 context limit"
    );
    let onboard = args.rtx_expert_layers.onboard(experts.len());
    if let Onboard::Layers(count) = onboard {
        ensure!(
            count <= experts.len(),
            "explicit V4.1 prefix exceeds available expert inventory"
        );
    }
    // The sampled total is already the device cap (`GPU_MEMORY_CAP`, applied
    // by `cuda_memory_info`); the named reserve is kept free inside it.
    let mut ceilings = Vec::with_capacity(ranks);
    let mut cards = Vec::with_capacity(ranks);
    for (gpu, &(free, total)) in memory.iter().enumerate() {
        ensure!(total > 0 && free <= total, "invalid GPU {gpu} memory sample");
        ceilings.push(total);
        let free = if let Some(&context) = preweight_context.get(gpu) {
            (total as u64).checked_sub(context).context("V4.1 context sample exceeds physical capacity")?
        } else { free as u64 };
        cards.push((total as u64, total as u64, free));
    }
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(args.snapshot.join("config.json"))?)?;
    let maximum = cuteafd_loader::serving_capacity::checkpoint_context_limit(&config)?
        .context("V4.1 checkpoint lacks context capability")?;
    ensure!(u64::from(args.max_context_tokens) <= maximum, "V4.1 context exceeds checkpoint capability");
    let serving_context = cuteafd_loader::placement::planner::Context::resolve(
        if args.max_context_tokens == 0 { maximum } else { u64::from(args.max_context_tokens) },
        Some(u64::from(args.max_context_tokens)));
    let context = serving_context.maximum;
    let totals = memory.iter().map(|&(_, total)| total as u64).collect::<Vec<_>>();
    let tails = args.concurrency as u64 + 2 * args.resolved_prefix_entries() as u64;
    let mut pool = PoolPolicy::resolve(
        &totals,
        context,
        args.pool_tokens,
        512);
    if let Some(bytes) = args.kv_pool_size.filter(|_| args.pool_tokens.is_none()) {
        let groups = bytes.0 as u64 / (5 * 256 * 356);
        ensure!(
            groups > tails && groups <= 131072,
            "explicit KV byte pool is outside physical/admission limits"
        );
        pool = PoolPolicy::resolve(
            &totals,
            context,
            Some((groups - tails) * 512),
            512);
    }
    pool = pool.with_context(serving_context);
    // An explicit pool below the pinned context is the first typed refusal;
    // zero and an omitted pool both selected automatic admission above.
    pool.validate_explicit()?;
    // The family executor's shape: the live capacity the chunk rounds to and
    // the configured units in flight.
    let prefill = PrefillShape {
        rows: u64::from(
            cuteafd_core::coordinator_programs::v41_live_rows(args.prefill_batch_tokens)
                .context("invalid V4.1 prefill batch")?,
        ),
        waves: u64::from(args.prefill_waves),
    };
    // One graph arena per coordinator GPU, charged before KV sizing. The
    // daemon's bounded device cache spends it; a measured warmup never shrinks
    // or replaces the configured reserve.
    let mut allowances = Vec::new();
    for gpu in 0..memory.len() {
        allowances.push(admission::graph_allowance(memory.len(), gpu)?);
    }
    let mut retained = owners.iter().map(Owner::from_demand).collect::<Vec<_>>();
    retained.extend_from_slice(additional_owners);
    for (gpu, &bytes) in deferred.iter().enumerate().filter(|(_, &bytes)| bytes > 0) {
        retained.push(Owner::charged(
            gpu as u8,
            Category::Transport,
            "deferred serving lanes and setup",
            bytes as u64,
        ));
    }
    let config = serde_json::from_slice(&std::fs::read(args.snapshot.join("config.json"))?)?;
    let limit = weight_limit;
    let formats = cuteafd_core::spark_reply::requested();
    let sparks =
        vec![cuteafd_loader::plan::layout::DEFAULT_SPARK_DEVICE_BYTES; args.peers.len()];
    #[cfg(not(test))]
    let host_memory = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
    #[cfg(test)]
    let host_memory = (1u64 << 40, 1u64 << 40);
    let catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    let mut host = admission::HostFacts::prepare(&config,
        &catalog.tensors().iter().map(|tensor| &tensor.metadata).collect::<Vec<_>>(),
        prefill, ranks, args.peers.len(), context, args.concurrency as u64,
        args.dspark, match args.prefix.host_cache_bytes { crate::shared::prefix::HostBudget::Auto => None,
            crate::shared::prefix::HostBudget::Bytes(bytes) => Some(bytes) },
        args.prefix.host_cache_bytes.enabled() && args.resolved_prefix_entries() > 0,
        crate::shared::prefix::PrefixArgs::host_chunk_bytes(super::prefix::layout(1, args.dspark)),
        host_memory, cuteafd_loader::plan::mapped_scale_preload_bytes(args.mapped_scale_preload_gib)?, None, false)?;
    if embedding_placement(args.embedding_placement, memory[0].1) == crate::shared::memory::EmbedPlacement::Host {
        let bytes = catalog.tensors().iter().filter(|tensor| tensor.metadata.name == "embed.weight")
            .map(|tensor| tensor.metadata.byte_length).sum();
        retained.push(Owner::host(Category::Embedding, "embedding", bytes)
            .at(cuteafd_loader::placement::planner::Location::PINNED));
    }
    #[cfg(not(test))]
    if catalog.exl3().is_none() && catalog.nvfp4().is_none() && catalog.fp8().is_none() {
        use crate::shared::experts::{layer::ExpertLayer, rtx::push::{CoordinatorExperts, WholeOwner}};
        let owners = (0..ranks).map(|rank| WholeOwner { rank, budget: usize::MAX,
            selections: (0..40).map(|layer| if ranks == 1 { ExpertLayer::BackboneFull { layer } }
                else { ExpertLayer::BackboneTp2 { layer, rank } })
                .chain((0..if args.dspark && ranks == 2 { 3 } else { 0 })
                    .map(|stage| ExpertLayer::DsparkTp2 { stage, rank })).collect() }).collect();
        let actual = CoordinatorExperts::Native { owners }.host_envelope(&catalog, ranks, &[])?;
        host.push.validate_usage(&actual)?;
        host.push.expert_bytes = actual.expert_bytes;
        host.push.retained_bytes = actual.retained_bytes;
        host.owners.retain(|owner| !owner.group.starts_with("coordinator push "));
        for owner in host.push.allocations("v41.push")?.into_iter().filter(|owner| !owner.id.ends_with(".dense")) {
            host.owners.push(Owner::host(owner.category, owner.group, owner.bytes)
                .at(owner.location).with_lifetime(owner.lifetime));
        }
    }
    retained.extend(host.owners.clone());
    allowances.extend(host.estimates.clone());
    let mut budgets = live_budgets(&cards, &sparks)?;
    budgets.extend(admission::production_budgets(&cards.iter().map(|&(total, ceiling, _)| (total, ceiling)).collect::<Vec<_>>(),
        0, Some(host_memory), 0, &[])?.into_iter().filter(|budget| budget.locations.contains(&cuteafd_loader::placement::planner::Location::HOST)));
    let mut blas = families::deepseek_v41::owners::blas_owners(ranks,
        crate::shared::target_head::TargetHead::configured().blas_head(
            !crate::families::deepseek_v41::v41_tensors::fp8_head().target()), draft.is_some())?;
    if local_vision_owner(cuteafd_api::openai::vision_input_enabled(), args.vision_peers.is_some()) {
        use cuteafd_loader::placement::planner::{Activation, Lifetime, Location, serving::BlasOwners};
        blas.push(BlasOwners { name: "V4.1 local vision".into(), process: "coordinator".into(),
            thread: "v41/blas/media".into(), location: Location::coordinator(0), gemm: true, lt: false,
            lifetime: Lifetime::RESIDENT, activation: Activation::always() });
    }
    let input = ProductionInput {
        coordinator_arch,
        blas,
        expert_usage: args.expert_usage_enabled()?,
        config: &config,
        gpus: ranks,
        context,
        maximum_pool_tokens: pool.maximum_tokens(),
        prefill,
        concurrency: args.concurrency as u64,
        retained_turns: args.resolved_prefix_entries() as u64,
        replicated: args.tp2_attention,
        spark_ranks: args.peers.len(),
        onboard_minimum: minimum,
        onboard_counts: None,
        experts,
        workspace,
        owners: retained,
        weight_arenas: weight_arenas.to_vec(),
        startup: startup.iter().map(Owner::from_demand).collect(),
        allowances,
        draft: draft.cloned(),
        encoders: Vec::new(),
        media_cache_bytes: 0,
        resident_weight_limit: limit,
        spark_reply_formats: formats,
        operation: Operation::Serving,
        budgets,
    };
    let request = production_request(
        serving_context,
        pool.requested,
        onboard,
        prefill,
        Operation::Serving,
        limit,
        formats,
    );
    let prepared = input.prepare()?;
    let admitted = if let Some(tier) = tier {
        ensure!(ranks == 2 && minimum == 0,
            "the GPU expert tier runs TP2 halves on two RTX GPUs, with no mandatory whole-layer prefix");
        prepared.admit_with_tier(request, &tier.choice)?
    } else { prepared.admit(request)? };
    cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "deepseek_v41");
    for note in &admitted.tier_notes { tracing::info!(note, "GPU expert tier admission"); }
    let host_ledger = admitted.host_ledger(host.prefix_bytes)?;
    let placement = admitted.placement;
    cuteafd_bench::context::ResolvedSettings::placement(&placement).publish();
    if tier.is_some() {
        ensure!(placement.expert_tier.as_ref().is_some_and(|placed| placed.experts > 0
                && placed.shard == ExpertTierShard::Tp2 && placement.onboard_layers == 0),
            "the GPU expert tier admitted no experts beside the pool: lower the pool or use --gpu-expert-tier off");
    }
    let groups = usize::try_from(placement.pool_tokens.div_ceil(512) + tails)?;
    let exact = cuteafd_loader::serving_capacity::deepseek_v41_cache_bytes(
        &config,
        ranks,
        args.concurrency as u64,
        groups as u64,
        args.tp2_attention,
    )?;
    for (gpu, &bytes) in exact.iter().enumerate() {
        ensure!(
            placement.items[gpu]
                .iter()
                .filter(|i| i.category == Category::Kv)
                .map(|i| i.bytes)
                .sum::<u64>()
                == bytes,
            "shared V4.1 cache demand differs from allocator"
        );
    }
    cuteafd_bench::context::set_resolved("kv-pool-size", &(groups * (5 * 256 * 356)).to_string());
    Ok(Admitted {
        served_context: u32::try_from(admitted.solved.served_context())?,
        expert_usage: admitted.expert_usage,
        host, host_ledger,
        pages: [groups, groups, groups, 2 * groups],
        cache_bytes: exact
            .into_iter()
            .map(usize::try_from)
            .collect::<std::result::Result<_, _>>()?,
        global_bytes: groups * (5 * 256 * 356),
        reservation_bytes: ceilings,
        placement,
        solved: admitted.solved,
        maximum_pool_tokens: pool.maximum_tokens(),
        config,
    })
}

/// Publish identity only after measured-pool adoption has finalized owners.
pub(super) fn publish_provenance(args: &crate::cli::NativeServeArgs,
    inventory: &crate::shared::inventory::RuntimeInventory, admitted: &Admitted) -> Result<()> {
    use sha2::{Digest, Sha256};
    let hash = |path: &std::path::Path| -> Result<String> {
        let mut file = std::fs::File::open(path)?;
        let mut digest = Sha256::new();
        std::io::copy(&mut file, &mut digest)?;
        Ok(format!("{:x}", digest.finalize()))
    };
    let manifest = families::deepseek_v41::owners::native_manifest(&args.native_lib, "PROGRAMS.json")?;
    let programs_hash = format!("{:x}", Sha256::digest(serde_json::to_vec(&manifest)?));
    cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "deepseek_v41");
    cuteafd_ffi::memory_ledger::fault::set_inputs(serde_json::json!({
        "memory_reconciliation_provenance": {
            "schema_version": 1, "family": "deepseek_v41",
            "source_commit": env!("CUTEAFD_BUILD_COMMIT"), "source_dirty": env!("CUTEAFD_BUILD_DIRTY"),
            "toolchain": env!("CUTEAFD_BUILD_TOOLCHAIN"), "native_sha256": hash(&args.native_lib)?,
            "programs_sha256": programs_hash, "programs_hash_method": "sha256 canonical serde_json manifest",
            "layout": {"snapshot": args.snapshot, "ranks": inventory.gpus.len(), "concurrency": args.concurrency,
                "context": args.max_context_tokens, "prefill_rows": args.prefill_batch_tokens,
                "attention": format!("{:?}", args.tp2_attention), "dspark": args.dspark,
                "spark_world": args.peers.len()},
            "devices": inventory.gpus, "onboard_layers": admitted.solved.decisions.onboard_layers,
            "prefill": admitted.solved.decisions.prefill,
            "graph_basis": "layout/rank measured step growth, split by estimate within shared steps"
        }
    }));
    Ok(())
}

/// Native info, not an independently rounded padding formula, determines every
/// resident pack. The checkpoint header supplies staging and EXL3 arena layout.
pub(super) fn expert_demands(
    lib: &cuteafd_ffi::NativeLibrary,
    catalog: &cuteafd_loader::OfficialV41Catalog,
    rows: u32,
    ranks: usize,
) -> Result<Vec<ExpertCost>> {
    let stored = {
        let info = match (ranks, catalog.nvfp4().is_some()) {
            (1, false) => lib.v41_local_expert_info(rows)?,
            (1, true) => lib.v41_nvfp4_local_expert_info(rows)?,
            (2, false) => lib.v41_tp2_expert_info(1)?,
            (2, true) => lib.v41_nvfp4_tp2_expert_info(1)?,
            _ => anyhow::bail!("invalid V4.1 expert owners"),
        };
        info.kernel_intermediate as u64
    };
    (0..40)
        .map(|layer| {
            let mut cost = ExpertCost {
                whole: Bytes2::default(),
                half: [Bytes2::default(); 2],
                tp2: ranks == 2,
                whole_supported: true, spark_ok: true,
            };
            if ranks == 1 {
                cost.whole = families::deepseek_v41::expert_cost(catalog, layer, 1, 0, stored)?;
            } else {
                for rank in 0..2 {
                    cost.half[rank] =
                        families::deepseek_v41::expert_cost(catalog, layer, 2, rank, stored)?;
                }
            }
            Ok(cost)
        })
        .collect()
}

/// The residual bar the memory campaign holds a plan to: a named-owner ledger
/// within this of the device at Ready needs no further attribution.
const RUNTIME_BASE_RESIDUAL_BAR: u64 = 50 << 20;

/// Compare the admitted pre-weight sample (including selected modules) with
/// the device at Ready. Verification only: the pool already exists, so a
/// residual reports later runtime growth without changing placement.
pub(super) fn log_runtime_base(lib: &cuteafd_ffi::NativeLibrary, admitted: &Admitted) -> Result<()> {
    use families::deepseek_v41::admission::{runtime_base_check, runtime_base_components};
    let ranks = admitted.placement.items.len();
    let current = lib.cuda_get_device()?;
    for gpu in 0..ranks {
        lib.cuda_set_device(gpu as i32)?;
        let (free, total) = lib.cuda_memory_info()?;
        let used = (total as u64).saturating_sub(free as u64);
        let components = runtime_base_components(&admitted.solved.facts, gpu as u8);
        let graph_arena = crate::shared::decode_graph::device::snapshots().into_iter()
            .find(|sample| sample.device == gpu as i32).map_or(0, |sample| sample.charged);
        let (predicted, measured) = runtime_base_check(&admitted.solved.facts, gpu as u8, used.saturating_sub(graph_arena))
            .with_context(|| format!("V4.1 runtime base verification on GPU {gpu}"))?;
        ensure!(components.total() == predicted,
            "V4.1 runtime base on GPU {gpu} reports {} charged bytes against a prediction of {predicted}",
            components.total());
        let residual = i128::from(measured) - i128::from(predicted);
        if residual.unsigned_abs() > u128::from(RUNTIME_BASE_RESIDUAL_BAR) {
            tracing::warn!(gpu, context_bytes = components.context_bytes,
                module_bytes = components.module_bytes, predicted_bytes = predicted,
                measured_bytes = measured, residual_bytes = %residual,
                bar_bytes = RUNTIME_BASE_RESIDUAL_BAR,
                "V4.1 predicted runtime base leaves a residual (device state beyond the CUDA context and module images)");
        } else {
            tracing::info!(gpu, context_bytes = components.context_bytes,
                module_bytes = components.module_bytes, predicted_bytes = predicted,
                measured_bytes = measured, residual_bytes = %residual, "V4.1 predicted runtime base");
        }
    }
    lib.cuda_set_device(current)?;
    Ok(())
}

/// Adopt the pre-pool device measurement into the admitted plan (#1634).
///
/// Called once, after the last loading owner and before the KV pool exists:
/// the device's occupancy at that point is exactly the plan's owners plus
/// what the plan cannot name — the driver's 2 MiB backing beyond each live
/// request, and any remaining device state. The measurement is charged as
/// named occupancy and the pool is re-fitted to what the budgets still hold,
/// so a budget-bound boot loses KV instead of discovering the bytes at Ready.
pub(super) fn adopt_measured_pool(
    lib: &cuteafd_ffi::NativeLibrary,
    args: &crate::cli::NativeServeArgs,
    admitted: &mut Admitted,
) -> Result<u32> {
    use cuteafd_ffi::memory_ledger::{Space, snapshot};
    use families::deepseek_v41::admission::{MeasuredOccupancy, adopt_measured, untracked_ready_owners, whole_gpu_reserve};
    // Explicit reserve overrides retain the sampled base but replace the
    // optional runtime-growth owners, so do not reintroduce them here.
    if whole_gpu_reserve() { return Ok(admitted.served_context); }
    let gpus = admitted.placement.items.len();
    let current = lib.cuda_get_device()?;
    let mut measured = Vec::with_capacity(gpus);
    for gpu in 0..gpus {
        lib.cuda_set_device(gpu as i32)?;
        let (free, total) = lib.cuda_memory_info()?;
        let used = (total as u64).saturating_sub(free as u64);
        let device = lib.cuda_get_device()?;
        let ledger = snapshot();
        let tracked = ledger.total(Space::Device, device) as u64;
        let rounding = ledger.granularity.get(&(Space::Device, device)).copied().unwrap_or_default();
        let backing = rounding.tail as u64;
        let named = untracked_ready_owners(&admitted.solved.facts, gpu as u8)?;
        let unattributed = used
            .saturating_sub(tracked)
            .saturating_sub(named)
            .saturating_sub(backing);
        tracing::info!(gpu, used_bytes = used, tracked_bytes = tracked,
            named_bytes = named, backing_bytes = backing,
            large_requests = rounding.large_requests, small_bytes = rounding.small_bytes,
            unattributed_bytes = unattributed,
            "V4.1 pre-pool device measurement");
        measured.push(MeasuredOccupancy { gpu: gpu as u8, backing, unattributed });
    }
    lib.cuda_set_device(current)?;
    let ranks = gpus;
    let slots = args.concurrency as u64;
    let retained = args.resolved_prefix_entries() as u64;
    let tier_weights = admitted.solved.facts.resident_weights.iter()
        .filter(|owner| owner.id.starts_with("v41.tier.")).cloned().collect::<Vec<_>>();
    let tier_owners = admitted.solved.facts.allocations.iter()
        .filter(|owner| owner.id.starts_with("v41.tier."))
        .map(|owner| (owner.id.clone(), owner.bytes, owner.location, owner.lifetime)).collect::<Vec<_>>();
    let adopted = adopt_measured(&admitted.config, ranks, slots, retained,
        args.tp2_attention, admitted.maximum_pool_tokens, &mut admitted.solved, &measured)
        .context("V4.1 pre-pool measurement adoption")?;
    let adopted_tier = admitted.solved.facts.allocations.iter()
        .filter(|owner| owner.id.starts_with("v41.tier."))
        .map(|owner| (owner.id.clone(), owner.bytes, owner.location, owner.lifetime)).collect::<Vec<_>>();
    let adopted_weights = admitted.solved.facts.resident_weights.iter()
        .filter(|owner| owner.id.starts_with("v41.tier.")).cloned().collect::<Vec<_>>();
    ensure!(tier_owners == adopted_tier && tier_weights == adopted_weights,
        "measured KV adoption changed solved tier owners or resident-weight claims");
    let exact = cuteafd_loader::serving_capacity::deepseek_v41_cache_bytes(
        &admitted.config, ranks, slots, adopted.groups, args.tp2_attention)?;
    ensure!(exact == adopted.bytes, "adopted V4.1 cache owners differ from allocator");
    let groups = usize::try_from(adopted.groups)?;
    // The runtime geometry follows the adopted plan exactly: the same page
    // array, per-GPU cache bytes and global extent the solve would have built.
    admitted.pages = [groups, groups, groups, 2 * groups];
    admitted.cache_bytes = exact.into_iter()
        .map(usize::try_from)
        .collect::<std::result::Result<_, _>>()?;
    admitted.global_bytes = groups * (5 * 256 * 356);
    admitted.placement.pool_tokens = admitted.solved.pool_tokens;
    admitted.placement.items = (0..ranks)
        .map(|rank| admitted.solved.items(
            cuteafd_loader::placement::planner::Location::coordinator(rank as u8)))
        .collect::<std::result::Result<_, _>>()?;
    admitted.served_context = u32::try_from(admitted.solved.served_context())?;
    cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "deepseek_v41");
    tracing::info!(pages = adopted.pages, groups = adopted.groups,
        pool_tokens = admitted.placement.pool_tokens, cache_bytes = ?admitted.cache_bytes,
        served_context = admitted.served_context,
        "V4.1 pool re-fitted to the measured pre-pool device state");
    Ok(admitted.served_context)
}

/// Charge all occupancy arriving after the context/module inventory, including
/// lazy module growth, not just the tracked weight allocations.
/// Bind target modules before the inventory sample and first immutable weight upload.
pub(super) fn preload(
    lib: &cuteafd_ffi::NativeLibrary,
    rows: u32,
    ranks: usize,
    nvfp4: bool,
) -> Result<()> {
    use super::*;
    if !nvfp4 {
        let capacity = if ranks == 1 {
            cuteafd_core::coordinator_programs::v41_aot_rows(rows).context("expert capacity")?
        } else {
            rows
        };
        let infos = families::deepseek_v41::expert_capacities(capacity as u64, ranks == 1)?
            .into_iter()
            .map(|capacity| {
                match ranks {
                    1 => lib.v41_local_expert_info(capacity as u32),
                    2 => lib.v41_tp2_expert_info(capacity as u32),
                    _ => anyhow::bail!("invalid V4.1 expert owners"),
                }
                .with_context(|| format!("V4.1 MXFP4 {ranks}-GPU m{capacity} export"))
            })
            .collect::<Result<Vec<_>>>()?;
        crate::shared::experts::ExpertReductionPlan::native(lib, infos)?;
    }
    if ranks == 1 {
        BackboneLane::workspace_bytes(lib, rows)?;
    } else {
        BackboneLane::placed_workspace_bytes(lib, rows)?;
    }
    IndexLane::workspace_bytes(lib, rows)?;
    for &(k, n) in &[(5120, 1152), (1152, 5120)] {
        if ranks == 2 {
            lib.v41_fp8_matrix_plan(rows, k, n)?;
        }
    }
    {
        let capacity = if ranks == 1 {
            cuteafd_core::coordinator_programs::v41_aot_rows(rows).context("expert capacity")?
        } else {
            rows
        };
        for c in families::deepseek_v41::expert_capacities(capacity as u64, ranks == 1)? {
            match (ranks, nvfp4) {
                (1, false) => {
                    lib.v41_local_expert_kernel(c as u32)?;
                }
                (1, true) => {
                    lib.v41_nvfp4_local_expert_kernel(c as u32)?;
                }
                (2, false) => {
                    lib.v41_tp2_expert_kernel(c as u32)?;
                }
                (2, true) => {
                    lib.v41_nvfp4_tp2_expert_kernel(c as u32)?;
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod admission_tests {

    /// The planner's transport charge is exactly what serve allocates: two
    /// decode lanes' linked waves, every prefill slot's reduce output (slot 0
    /// holds the link) and the one prefill link's intake over its slots.
    #[test]
    fn transport_charge_equals_serve_allocation() -> anyhow::Result<()> {
        use crate::families::deepseek_v41::v41_experts::coordinator::{NativeTp4Wave, V41Link};
        for (rows, ranks) in [(1024u32, 3usize), (1408, 3), (2752, 4), (2752, 6), (4096, 4)] {
            for slots in 1..=4usize {
                for format in [cuteafd_core::SparkReplyFormat::Bf16, cuteafd_core::SparkReplyFormat::Fp8,
                    cuteafd_core::SparkReplyFormat::Nvfp4] {
                    let reserve = slots as u64 * u64::from(rows);
                    let prefill = V41Link::prefill_device_bytes(reserve as usize, ranks, format)?;
                    assert_eq!(prefill as u64, ranks as u64
                        * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(reserve * 5120 * 2)?);
                    let runtime = 2 * NativeTp4Wave::device_bytes_for(crate::families::deepseek_v41::v41_native_serve::DECODE_LANE_ROWS, ranks)?
                        + slots * rows as usize * 5120 * 2 + prefill;
                    assert_eq!(cuteafd_loader::placement::families::deepseek_v41::owners::transports_bytes(u64::from(crate::families::deepseek_v41::v41_native_serve::DECODE_LANE_ROWS), u64::from(rows), ranks, slots, format)?, runtime as u64,
                        "{format}: {slots} x {rows} rows on {ranks} ranks");
                }
            }
        }
        Ok(())
    }
    use super::*;
    use clap::Parser;

    fn args(directory: &std::path::Path) -> crate::cli::NativeServeArgs {
        if !directory.join("model.safetensors.index.json").exists() {
            cuteafd_loader::plan::testing::write_v41_snapshot(directory);
        }
        let cli = crate::cli::Cli::try_parse_from([
            "cuteafd",
            "serve-native",
            "--snapshot",
            directory.to_str().unwrap(),
            "--native-lib",
            directory.join("lib/unused.so").to_str().unwrap(),
            "--peers",
            "127.0.0.1:19441",
        ])
        .unwrap();
        let crate::cli::Commands::ServeNative(args) = cli.command else {
            unreachable!()
        };
        args
    }

    /// Routed inventory for pool, owner and drafter tests: forty zero-cost
    /// layers, so only the pool, the retained owners and the cache are charged.
    fn no_experts() -> Vec<ExpertCost> {
        vec![ExpertCost {
            whole: Bytes2 { resident: 0, staging: 0 },
            half: [Bytes2 { resident: 0, staging: 0 }; 2],
            tp2: false,
            whole_supported: true,
            spark_ok: true,
        }; 40]
    }

    #[test]
    fn v41_admission_contract_synthetic_inventory() -> Result<()> {
        use cuteafd_core::memory_layout::{Basis, Category};
        use families::deepseek_v41::{owners, resident};
        for nvfp4 in [false, true] {
            let dir = tempfile::tempdir()?;
            if nvfp4 {
                cuteafd_loader::plan::testing::write_v41_nvfp4_snapshot(dir.path());
            } else {
                cuteafd_loader::plan::testing::write_v41_snapshot(dir.path());
            }
            let manifest = cuteafd_loader::plan::testing::write_v41_workspaces(dir.path());
            let fp8 = serde_json::from_slice(&std::fs::read(
                manifest.parent().unwrap().join("V41_FP8_AOT.json"))?)?;
            let programs = owners::NativePrograms::new(&fp8, 188)?;
            let catalog = cuteafd_loader::read_expert_catalog(dir.path())?;
            let config = serde_json::from_slice(&std::fs::read(dir.path().join("config.json"))?)?;
            for ranks in [1, 2] {
                let weights = resident::resident_headers(&config,
                    catalog.tensors().iter().map(|tensor| &tensor.metadata), ranks, false)
                    .context("synthetic native backbone residency")?;
                let weights = weights.into_iter().enumerate().flat_map(|(gpu, items)|
                    items.into_iter().map(move |item| Demand::new(gpu as u8, item.category,
                        item.group, item.bytes, item.basis).with_resident_weights(ResidentWeights(item.bytes))))
                    .collect::<Vec<_>>();
                let target = programs.target(owners::DECODE_LANE_ROWS, 512, 256, ranks, 2, 16, resident::Fp8Head::All, ranks == 1)?;
                // The startup validators accept the planner's own target list.
                if ranks == 1 { validate_engram_rows(&target, 512, owners::DECODE_LANE_ROWS as usize)?; }
                let fixed = owners::TargetOwners::new(weights, target, ranks,
                    2 * owners::transport_bytes(256, 4)?, 42)?.fixed;
                let (experts, workspace) = families::deepseek_v41::expert_inventory(
                    &catalog, &manifest, 256, ranks, 188, 2, 512, owners::DECODE_LANE_ROWS)?;
                // Deliberately synthetic pre-weight inventory, independent of
                // planner output. This is not a CUDA sample or ready-parity gate.
                let memory = vec![(80usize << 30, 96usize << 30); ranks];
                for (onboard, pool_tokens, concurrency, turns) in [
                    (LocalLayers::Auto, None, 16, 20),
                    (LocalLayers::Count(3), Some(0), 16, 20),
                    (LocalLayers::Count(3), Some(1 << 20), 1, 0),
                ] {
                    let mut args = args(dir.path());
                    args.peers = vec![args.peers[0].clone(); 4];
                    args.rtx_gpus = ranks as u32;
                    // Match the synthetic target owners' 256-row, two-wave prefill arena.
                    args.prefill_batch_tokens = 256;
                    args.prefill_waves = 2;
                    args.prefill_reserve_rows = 512;
                    args.max_context_tokens = 262_144;
                    args.pool_tokens = pool_tokens;
                    args.concurrency = concurrency;
                    args.prefix.prefix_cache_entries = turns as usize;
                    args.rtx_expert_layers = onboard;
                    let admitted = crate::commands::bench::assert_resolved_family_seam("deepseek_v41", &[], || {
                        super::super::normalize_serving(&mut args)?;
                        super::super::AdmittedPrefill::resolve(&args)?;
                        super::super::attention_plan_for_arch(&args, cuteafd_core::GpuArch::Sm120)?;
                        embedding_placement(args.embedding_placement, memory[0].1);
                        resolve_with_owners(&args, &memory, experts.clone(), workspace,
                            &vec![0; ranks], usize::from(ranks == 2), &fixed, &[], &[], None, &[], cuteafd_core::GpuArch::Sm120, &[])
                    })?;
                    assert!(admitted.placement.pool_tokens >= 262_144);
                    if let Some(pool) = pool_tokens.filter(|&n| n > 0) {
                        assert_eq!(admitted.placement.pool_tokens, pool);
                    }
                    for owner in fixed.iter()
                        .filter(|d| d.lifetime == cuteafd_core::memory_layout::ItemLifetime::ReadyResident)
                    {
                        let expected = fixed.iter().filter(|d| d.gpu == owner.gpu
                            && d.category == owner.category && d.group == owner.group)
                            .map(|d| d.bytes).sum::<u64>();
                        let actual = admitted.placement.items[usize::from(owner.gpu)].iter()
                            .filter(|item| item.category == owner.category && item.group == owner.group)
                            .map(|item| item.bytes).sum::<u64>();
                        assert_eq!(actual, expected, "{owner:?}");
                        assert!(admitted.placement.items[usize::from(owner.gpu)].iter()
                            .filter(|item| item.category == owner.category && item.group == owner.group)
                            .all(|item| item.lifetime == owner.lifetime));
                    }
                    assert_eq!(admitted.placement.attention_placement,
                        if ranks == 2 { AttentionPlacement::Layers } else { AttentionPlacement::Heads });
                    assert_eq!(admitted.placement.layers.len(), 40);
                    for (layer, assigned) in admitted.placement.layers.iter().enumerate() {
                        assert_eq!(assigned.mode, LayerMode::Whole {
                            gpu: u8::from(ranks == 2 && layer >= 20), ffn: FfnMode::Owner });
                    }
                    let over = [Demand::new(0, Category::Staging, "synthetic load overflow",
                        (80 << 30) + 1, Basis::Exact)];
                    assert!(resolve_with_owners(&args, &memory, experts.clone(), workspace,
                        &vec![0; ranks], usize::from(ranks == 2), &fixed, &[], &over, None, &[], cuteafd_core::GpuArch::Sm120, &[]).is_err());
                    let mut missing_owner = fixed.clone();
                    missing_owner.push(Demand::new(0, Category::Workspace,
                        "synthetic owner overflow", 80 << 30, Basis::Exact));
                    assert!(resolve_with_owners(&args, &memory, experts.clone(), workspace,
                        &vec![0; ranks], usize::from(ranks == 2), &missing_owner, &[], &[], None, &[], cuteafd_core::GpuArch::Sm120, &[]).is_err());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn tier_capability_requires_selected_program_inventory() -> Result<()> {
        use crate::shared::experts::layer::tp2_half_tests::expert_fixture;
        let (_dir, catalog) = expert_fixture()?;
        let mut args = args(catalog.snapshot());
        args.gpu_expert_tier = ExpertTierSetting::PerLayer(6);
        let manifest = catalog.snapshot().join("share/PROGRAMS.json");
        assert!(!TierAsk::of(&args, &catalog)?.context("tier is on")?.choice.expert_reply_reduce_tier);
        std::fs::write(&manifest, br#"{"programs":[{"name":"expert_reply_reduce_tier"}]}"#)?;
        assert!(TierAsk::of(&args, &catalog)?.context("tier is on")?.choice.expert_reply_reduce_tier);
        for invalid in [br#"{}"#.as_slice(), br#"{"programs":{}}"#.as_slice()] {
            std::fs::write(&manifest, invalid)?;
            let error = TierAsk::of(&args, &catalog).err().context("missing inventory must be refused")?;
            assert!(error.to_string().contains("PROGRAMS.json has no program list"), "{error:#}");
        }
        Ok(())
    }

    /// `--gpu-expert-tier` admits what `cuteafd plan --layout
    /// --gpu-expert-tier` plans (one shared request): at an exact size the
    /// same resident experts per layer at the same bytes per GPU, in place of
    /// whole layers, its arena holding the TP2 FFN workspace, both decode
    /// lanes' slot tables and one expert's staging; automatic tiers are
    /// prefixes of one order; `--rtx-expert-layers N` pins N whole layers in
    /// it. It refuses one RTX, a mandatory prefix and a tier that fits nothing.
    #[test]
    fn gpu_expert_tier_admits_the_plans_resident_set() -> Result<()> {
        use crate::shared::experts::layer::tp2_half_tests::{expert_fixture, planned_bytes};
        use cuteafd_core::memory_layout::Category;
        use families::deepseek_v41::{expert_inventory, owners::DECODE_LANE_ROWS, tier_lane_bytes};
        let (_dir, catalog) = expert_fixture()?;
        let snapshot = catalog.snapshot().to_path_buf();
        let (experts, ffn) = expert_inventory(&catalog, &snapshot.join("share/PROGRAMS.json"), 256, 2, 128, 2, 512,
            DECODE_LANE_ROWS)?;
        let workspace = ffn;
        let tier_workspace = workspace + 2 * tier_lane_bytes(40, 384, DECODE_LANE_ROWS) + tier_lane_bytes(40, 384, 256);
        let card = ArchContext::reference_total_bytes(cuteafd_core::GpuArch::Sm89) as usize;
        let memory = vec![(card - (4 << 30), card); 2];
        let serve = |tier: &str, layers: &str, memory: &[(usize, usize)], experts: Vec<ExpertCost>, minimum: usize|
            -> Result<Admitted> {
            let mut args = args(&snapshot);
            args.peers = vec![args.peers[0].clone(); 4];
            args.rtx_gpus = memory.len() as u32;
            args.max_context_tokens = 262_144;
            args.prefill_batch_tokens = 256;
            args.prefill_waves = 2;
            args.prefill_reserve_rows = 512;
            args.gpu_expert_tier = tier.parse().map_err(anyhow::Error::msg)?;
            args.rtx_expert_layers = layers.parse()?;
            let ask = TierAsk::of(&args, &catalog)?.context("the tier is on")?;
            resolve_tiered(&args, memory, experts, workspace, &vec![0; memory.len()], minimum, &[], &[], None, Some(&ask),
                None)
        };
        let planned = |setting| -> Result<ExpertTierPlacement> {
            planned_bytes(&snapshot, false, setting)?.solver_placement.and_then(|placement| placement.expert_tier)
                .context("the plan admitted no tier")
        };
        let exact = serve("6", "auto", &memory, experts.clone(), 0)?;
        let tier = exact.placement.expert_tier.as_ref().context("serve admitted no tier")?;
        let plan = planned(ExpertTierSetting::PerLayer(6))?;
        assert_eq!((tier.experts, tier.shard, exact.placement.onboard_layers), (240, ExpertTierShard::Tp2, 0));
        assert!(exact.placement.tp2.is_none());
        assert_eq!(tier.resident, plan.resident, "serve and plan admit different experts");
        assert_eq!(tier.resident_bytes, plan.resident_bytes);
        for gpu in 0..2 {
            let staging = experts.iter().zip(&tier.resident).filter(|(_, ids)| !ids.is_empty())
                .map(|(cost, _)| cost.half[gpu].staging).max().unwrap_or(0);
            assert_eq!(tier.peak_bytes[gpu], tier_workspace + tier.resident_bytes[gpu] + staging, "GPU {gpu} tier arena");
            let owner = |suffix| exact.solved.facts.allocations.iter()
                .find(|owner| owner.id == format!("v41.tier.{gpu}.{suffix}")).expect("solved tier owner");
            assert_eq!((owner(0).category, owner(0).bytes), (Category::Experts, tier.resident_bytes[gpu]));
            assert_eq!((owner(1).category, owner(1).bytes), (Category::Workspace, tier_workspace));
            assert_eq!((owner(2).category, owner(2).bytes), (Category::Staging, staging));
            assert_eq!(owner(2).lifetime, cuteafd_loader::placement::planner::Lifetime {
                first: cuteafd_loader::placement::planner::Phase::Load,
                last: cuteafd_loader::placement::planner::Phase::Load });
            let weights = exact.solved.facts.resident_weights.iter()
                .find(|owner| owner.id == format!("v41.tier.{gpu}.0")).context("tier weight claim")?;
            assert_eq!(weights.weights.0, tier.resident_bytes[gpu]);
            assert_eq!(exact.placement.items[gpu].iter().filter(|item| item.category == Category::Experts)
                .map(|item| item.bytes).sum::<u64>(), tier.resident_bytes[gpu], "GPU {gpu} charges resident experts once");
        }
        // Automatic tiers are prefixes of one ranked order.
        let auto = serve("auto", "auto", &memory, experts.clone(), 0)?.placement.expert_tier.context("auto tier")?;
        let plan = planned(ExpertTierSetting::Auto)?;
        let (small, large) = if auto.experts <= plan.experts { (&auto, &plan) } else { (&plan, &auto) };
        assert!(small.experts > 0 && small.resident.iter().zip(&large.resident)
            .all(|(a, b)| a.iter().all(|id| b.contains(id))), "automatic tiers are not prefixes of one order");
        // Pinned whole layers lead the tier.
        let pinned = serve("40", "2", &memory, experts.clone(), 0)?.placement;
        let pinned = pinned.expert_tier.context("pinned tier")?;
        assert_eq!((pinned.resident[0].len(), pinned.resident[1].len(), pinned.whole_layers), (384, 384, 2));
        // Refused: one RTX, a mandatory prefix, a tier that fits no expert.
        let one = serve("6", "auto", &memory[..1], experts.clone(), 0).map(|_| ()).unwrap_err().to_string();
        assert!(one.contains("two RTX"), "{one}");
        assert!(serve("6", "auto", &memory, experts.clone(), 1).is_err());
        let huge = experts.iter().map(|cost| ExpertCost { half: cost.half.map(|half|
            Bytes2 { resident: 384 * (64 << 30), ..half }), ..*cost }).collect();
        let empty = serve("auto", "auto", &memory, huge, 0).map(|_| ()).unwrap_err().to_string();
        assert!(empty.contains("admitted no experts"), "{empty}");
        Ok(())
    }

    /// Serving's coordinator weight cap counts each GPU's weights outside the
    /// routed experts as `cuteafd plan` charges them (its weight, embedding and
    /// table owners), and a cap below the admitted tier sheds tier experts
    /// until every GPU's retained weights fit; a cap below the fixed weights
    /// refuses.
    #[test]
    fn serving_weight_cap_counts_the_plans_weights_and_sheds_tier_experts() -> Result<()> {
        use crate::shared::experts::layer::tp2_half_tests::{expert_fixture, planned_bytes};
        use cuteafd_core::memory_layout::Category;
        use families::deepseek_v41::{expert_inventory, owners::DECODE_LANE_ROWS};
        let (_dir, catalog) = expert_fixture()?;
        let snapshot = catalog.snapshot().to_path_buf();
        let report = planned_bytes(&snapshot, false, ExpertTierSetting::PerLayer(6))?;
        let layout = report.memory_layout.as_ref().context("the plan has a memory layout")?;
        let card = ArchContext::reference_total_bytes(cuteafd_core::GpuArch::Sm89) as usize;
        let mut args = args(&snapshot);
        args.peers = vec![args.peers[0].clone(); 4];
        args.rtx_gpus = 2;
        args.max_context_tokens = 262_144;
        args.prefill_batch_tokens = 256;
        args.prefill_waves = 2;
        args.prefill_reserve_rows = 512;
        let cap = WeightCap::loaded(&args, &catalog, 2, card, None)?;
        let target = fixture_target(&args, &catalog, 2, card)?;
        for gpu in 0..2 {
            let arena = &target.ranks[gpu];
            let extent = arena.bytes() as u64;
            let padding = arena.padding_bytes() as u64;
            let tail = cuteafd_loader::placement::planner::allocation_granularity_forecast(&[extent])?;
            let planned = layout.devices[gpu].items.iter()
                .filter(|item| matches!(item.category, Category::Weights | Category::Embedding | Category::Tables))
                .map(|item| item.bytes).sum::<u64>();
            assert!(planned > 0);
            // Physical plan items include the root's tail; the cap counts
            // logical payload P, not extent B or driver backing pages.
            assert_eq!(cap.fixed[gpu] + padding, extent, "GPU {gpu}: serving arena payload/extent mismatch");
            assert_eq!(cap.fixed[gpu] + padding + tail, planned,
                "GPU {gpu}: serving counts other weights than the plan");
            assert_eq!(layout.devices[gpu].items.iter().filter(|item| item.category == Category::Weights
                && item.group == "weight arena").map(|item| item.bytes).sum::<u64>(), extent);
            eprintln!("GPU {gpu}: fixed payload {} B, arena {extent} B, padding {padding} B, tail {tail} B, plan {planned} B",
                cap.fixed[gpu]);
        }
        // Under a cap 20 tier experts' halves short, the tier sheds experts.
        let (experts, workspace) = expert_inventory(&catalog, &snapshot.join("share/PROGRAMS.json"), 256, 2, 128, 2, 512,
            DECODE_LANE_ROWS)?;
        args.gpu_expert_tier = "6".parse().map_err(anyhow::Error::msg)?;
        let ask = TierAsk::of(&args, &catalog)?.context("the tier is on")?;
        let memory = vec![(card - (4 << 30), card); 2];
        let admit = |setting, cap: &WeightCap| {
            let ask = TierAsk { choice: families::deepseek_v41::TierChoice { setting, ..ask.choice.clone() } };
            resolve_tiered(&args, &memory, experts.clone(), workspace, &[0; 2], 0, &[], &[],
                None, Some(&ask), Some(cap))
        };
        let roomy = admit(ExpertTierSetting::PerLayer(6), &WeightCap { limit: u64::MAX, fixed: cap.fixed.clone() })?;
        let tier = roomy.placement.expert_tier.as_ref().context("tier")?;
        assert_eq!(tier.experts, 240);
        for gpu in 0..2 {
            let planned = roomy.solved.facts.resident_weights.iter()
                .filter(|owner| owner.location == cuteafd_loader::placement::planner::Location::coordinator(gpu as u8)
                    && roomy.solved.facts.allocations.iter().any(|allocation| allocation.id == owner.id
                        && matches!(allocation.category, Category::Weights | Category::Embedding | Category::Tables)))
                .map(|owner| owner.weights.0).sum::<u64>();
            assert_eq!(cap.fixed[gpu], planned, "GPU {gpu}: serving counts other fixed weight claims than the plan");
        }
        let half = tier.resident_bytes[0] / 240;
        let limit = cap.fixed.iter().zip(&tier.resident_bytes).map(|(fixed, tier)| fixed + tier).max().unwrap() - 20 * half;
        // Exact counts are mandatory in the shared recipe; automatic tiers
        // preserve the fork's weight-cap shedding policy.
        assert!(admit(ExpertTierSetting::PerLayer(6), &WeightCap { limit, fixed: cap.fixed.clone() }).is_err());
        let tight = admit(ExpertTierSetting::Auto, &WeightCap { limit, fixed: cap.fixed.clone() })?;
        let shed = tight.placement.expert_tier.as_ref().context("tier under the cap")?;
        assert!((200..240).contains(&shed.experts), "{} experts under the cap", shed.experts);
        for gpu in 0..2 {
            assert!(cap.fixed[gpu] + shed.resident_bytes[gpu] <= limit, "GPU {gpu} retains more than the cap");
        }
        // Below the fixed weights alone nothing optional is left to shed.
        assert!(admit(ExpertTierSetting::Auto, &WeightCap { limit: cap.fixed[0].min(cap.fixed[1]) - 1, fixed: cap.fixed.clone() }).is_err());
        Ok(())
    }

    #[test]
    fn all_pins_every_routed_expert_layer_and_refuses_when_short() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("config.json"), include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
        )))?;
        for ranks in [1, 2] {
            let mut args = args(dir.path());
            args.rtx_gpus = ranks as u32;
            args.max_context_tokens = 262144;
            args.rtx_expert_layers = "all".parse()?;
            let experts = vec![ExpertCost {
                whole: Bytes2 { resident: 1 << 30, staging: 64 << 20 },
                half: [Bytes2 { resident: 1 << 30, staging: 64 << 20 }; 2],
                tp2: ranks == 2,
                whole_supported: true, spark_ok: true,
            }; 40];
            let roomy = vec![(80usize << 30, 96usize << 30); ranks];
            let all = resolve(&args, &roomy, experts.clone(), 64 << 20,
                &vec![0; ranks], usize::from(ranks == 2), None)?;
            assert_eq!(all.placement.onboard_layers, 40);
            let short = vec![(8usize << 30, 96usize << 30); ranks];
            let refused = resolve(&args, &short, experts, 64 << 20,
                &vec![0; ranks], usize::from(ranks == 2), None).map(|_| ()).unwrap_err();
            assert!(matches!(
                refused.downcast_ref::<cuteafd_loader::placement::planner::Refusal>(),
                Some(cuteafd_loader::placement::planner::Refusal::FullContextShortfall {
                    context: 262_144, .. })), "{refused}");
        }
        Ok(())
    }

    #[test]
    fn pre_draft_admission_charges_weights_runtime_and_marks_once() -> Result<()> {
        use cuteafd_core::memory_layout::{Basis, Category};
        use families::deepseek_v41::DraftInventory;
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("config.json"), include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
        )))?;
        for ranks in [1, 2] {
            let mut args = args(dir.path());
            args.rtx_gpus = ranks as u32;
            args.max_context_tokens = 512;
            args.pool_tokens = Some(512);
            args.rtx_expert_layers = LocalLayers::Count(0);
            let home = (ranks - 1) as u8;
            let draft = DraftInventory {
                resident: 8 << 30,
                runtime: (0..ranks).map(|gpu| Demand::new(gpu as u8,
                    Category::Workspace, "draft runtime", 64 << 20, Basis::Exact)).collect(),
                marks: 32 << 20,
                peer_half: if ranks == 2 { 512 << 20 } else { 0 },
                // Staging is not a retained owner or a complete startup phase.
                staging: [100 << 30; 2],
                loading_experts: [0; 2],
                loader_peak: [0; 2],
                // No dense arena: every retained draft byte is routed experts.
                dense: None,
            };
            let memory = vec![(80usize << 30, 96usize << 30); ranks];
            let moved = resolve(&args, &memory, no_experts(), 0, &vec![0; ranks], 0, Some(&draft))?;
            assert_eq!(draft::home(&moved.placement), Some(home));
            assert_eq!(moved.placement.items[home as usize].iter()
                .filter(|item| item.group == "dSpark experts").map(|item| item.bytes).sum::<u64>(), draft.resident);
            assert_eq!(moved.placement.items[home as usize].iter()
                .filter(|item| item.group == "dSpark marks").map(|item| item.bytes).sum::<u64>(), draft.marks);
            for gpu in 0..ranks {
                assert_eq!(moved.placement.items[gpu].iter().filter(|item| item.group == "draft runtime")
                    .map(|item| item.bytes).sum::<u64>(), 64 << 20);
            }
            // The same owners as an already-loaded inventory: retained draft
            // weights and marks on the home GPU, its peer half on GPU0, and the
            // request runtime lanes.
            let mut fixed = vec![
                Demand::new(home, Category::Drafter, "dSpark weights", draft.resident, Basis::Exact)
                    .with_resident_weights(ResidentWeights(draft.resident)),
                Demand::new(home, Category::Prefix, "dSpark marks", draft.marks, Basis::Exact),
            ];
            if draft.peer_half > 0 {
                fixed.push(Demand::new(0, Category::Drafter, "dSpark TP2 expert half",
                    draft.peer_half, Basis::Exact).with_resident_weights(
                    ResidentWeights(draft.peer_half)));
            }
            fixed.extend(draft.runtime.iter().cloned());
            // Its retained storage is in `fixed`, but the loaded drafter still
            // borrows the serving pool's Markov GEMM backend.
            let loaded_draft = DraftInventory::default();
            let original = resolve_with_owners(&args, &memory, no_experts(), 0, &vec![0; ranks], 0,
                &fixed, &[], &[], Some(&loaded_draft), &[], cuteafd_core::GpuArch::Sm120, &[])?;
            assert_eq!(moved.cache_bytes, original.cache_bytes);
            for gpu in 0..ranks {
                let total = |items: &[cuteafd_core::memory_layout::Item]| items.iter().map(|i| i.bytes).sum::<u64>();
                assert_eq!(total(&moved.placement.items[gpu]), total(&original.placement.items[gpu]));
            }
            // A pre-draft sample does not yet contain these bytes. Omitting
            // the movable would incorrectly admit this constrained inventory.
            let constrained = vec![(8usize << 30, 96usize << 30); ranks];
            assert!(resolve(&args, &constrained, no_experts(), 0, &vec![0; ranks], 0, None).is_ok());
            assert!(resolve(&args, &constrained, no_experts(), 0, &vec![0; ranks], 0, Some(&draft)).is_err());
            // Only the caller's full coexistence peak gates the load phase.
            let peak = [Demand::new(home, Category::Staging, "full draft load phase",
                81 << 30, Basis::Exact)];
            assert!(resolve_with_owners(&args, &memory, no_experts(), 0, &vec![0; ranks], 0,
                &[], &[], &peak, Some(&draft), &[], cuteafd_core::GpuArch::Sm120, &[]).is_err());
        }
        Ok(())
    }

    #[test]
    fn sampled_preweight_base_replaces_context_and_module_predictions() -> Result<()> {
        use cuteafd_core::memory_policy::MemoryPolicy;
        use families::deepseek_v41::admission::{CONTEXT_OWNER, MODULE_OWNER};
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("config.json"), include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
        )))?;
        let mut args = args(dir.path());
        args.rtx_gpus = 2;
        args.rtx_expert_layers = LocalLayers::Count(0);
        let occupied = [682_885_120, 691_273_728];
        let memory = occupied.map(|bytes| (((96u64 << 30) - bytes) as usize, 96usize << 30));
        for reserve in [None, Some(0), Some(200 << 20)] {
            MemoryPolicy { gpu_reserve: reserve, ..Default::default() }.scoped(|| -> Result<()> {
                let sampled = resolve_with_owners(&args, &memory, no_experts(), 0, &[0; 2], 0,
                    &[], &[], &[], None, &[], cuteafd_core::GpuArch::Sm120, &[])?;
                let named = resolve_with_owners(&args, &memory, no_experts(), 0, &[0; 2], 0,
                    &[], &[], &[], None, &occupied, cuteafd_core::GpuArch::Sm120, &[])?;
                assert_eq!(sampled.solved.pool_tokens, named.solved.pool_tokens);
                assert_eq!(sampled.solved.decisions, named.solved.decisions);
                for gpu in 0..2usize {
                    let facts = &named.solved.facts;
                    assert!(facts.allocations.iter().all(|owner| owner.group != CONTEXT_OWNER && owner.group != MODULE_OWNER));
                    let owners = facts.allowances.iter().filter(|owner|
                        owner.location == cuteafd_loader::placement::planner::Location::coordinator(gpu as u8)
                            && cuteafd_loader::placement::sampled_baseline::is_source(&owner.source)).collect::<Vec<_>>();
                    assert_eq!(owners.len(), 1);
                    assert_eq!(owners[0].bytes, occupied[gpu]);
                    assert_eq!(families::deepseek_v41::admission::runtime_base_components(facts, gpu as u8).total(), occupied[gpu]);
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    #[test]
    fn explicit_byte_pool_must_cover_fixed_context_after_snapshot_tails() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("config.json"), include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
        )))?;
        let mut args = args(dir.path());
        args.max_context_tokens = 131_072;
        args.rtx_expert_layers = LocalLayers::Count(0);
        let tails = args.concurrency as usize + 2 * args.resolved_prefix_entries() as usize;
        let memory = [(80usize << 30, 96usize << 30)];
        for tokens in [32_768, 131_072] {
            args.kv_pool_size = Some(crate::shared::memory::budget::ByteSize(
                (tokens / 512 + tails) * 5 * 256 * 356));
            let result = resolve(&args, &memory, no_experts(), 0, &[0], 0, None);
            if tokens < args.max_context_tokens as usize {
                let error = result.err().context("small explicit byte pool was admitted")?;
                assert_eq!(error.downcast_ref::<PlacementError>(), Some(&PlacementError::RequestedBelowContext {
                    requested: tokens as u64, floor: args.max_context_tokens as u64,
                }));
                assert!(error.to_string().contains("KV_POOL_SIZE"));
            } else {
                assert_eq!(result?.placement.pool_tokens, tokens as u64);
            }
        }
        Ok(())
    }

    #[test]
    fn logical_pool_excludes_snapshot_tails_and_charges_each_cache_owner() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("config.json"),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
            )),
        )?;
        for (ranks, replicas) in [(1, false), (2, false), (2, true)] {
            let mut args = args(dir.path());
            args.rtx_gpus = ranks as u32;
            args.tp2_attention = replicas;
            args.rtx_expert_layers = LocalLayers::Count(0);
            let memory = vec![(80usize << 30, 96usize << 30); ranks];
            let admitted = resolve(&args, &memory, no_experts(), 0, &vec![0; ranks], 0, None)?;
            assert_eq!(admitted.placement.pool_tokens, 2 << 20);
            assert_eq!(admitted.pages, [4152, 4152, 4152, 8304]);
            args.kv_pool_size = Some(crate::shared::memory::budget::ByteSize(
                (4152 * 5 * 256 * 356) + 1,
            ));
            let exact = resolve(&args, &memory, no_experts(), 0, &vec![0; ranks], 0, None)?;
            assert_eq!(exact.pages, admitted.pages);
            assert_eq!(exact.cache_bytes, admitted.cache_bytes);
            args.kv_pool_size = None;
            args.pool_tokens = Some(67_108_864);
            assert!(resolve(&args, &memory, no_experts(), 0, &vec![0; ranks], 0, None).is_err());
            args.pool_tokens = Some(2 << 20);
            assert!(
                resolve(
                    &args,
                    &vec![(1usize << 30, 96usize << 30); ranks],
                    vec![],
                    0,
                    &vec![0; ranks],
                    0, None
                )
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn strict_pool_respects_each_owner_deferred_bytes_and_the_capped_sample() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("config.json"),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
            )),
        )?;
        // The sample is already capped (`GPU_MEMORY_CAP` via cuda_memory_info);
        // an odd byte cap checks that no rounding admits past it.
        let total = "80.000001GiB".parse::<crate::shared::memory::budget::ByteSize>()?.0;
        for (ranks, replicas, expert_usage) in [
            (1, false, true), (1, false, false),
            (2, false, true), (2, false, false),
            (2, true, true), (2, true, false),
        ] {
            let mut args = args(dir.path());
            args.rtx_gpus = ranks as u32;
            args.tp2_attention = replicas;
            args.expert_usage = expert_usage;
            args.rtx_expert_layers = LocalLayers::Count(0);
            args.pool_tokens = Some(1 << 20);
            let deferred = vec![64u64 << 20; ranks];
            let roomy = vec![(64usize << 30, total); ranks];
            let admitted = resolve(&args, &roomy, no_experts(), 0, &deferred, 0, None)?;
            assert_eq!(admitted.expert_usage.is_some(), expert_usage);
            // Counters and the device snapshot are future GPU owners. Pinned
            // aggregates/history use host budgets, not the sampled GPU bytes.
            let usage_bytes = admitted.expert_usage.as_ref()
                .map_or(0, |geometry| 2 * geometry.table_bytes().unwrap());
            for items in &admitted.placement.items {
                assert_eq!(items.iter().filter(|item| item.group == "expert usage")
                    .map(|item| item.bytes).sum::<u64>(), usage_bytes as u64);
            }
            assert_eq!(admitted.placement.pool_tokens, args.pool_tokens.unwrap());
            assert_eq!(admitted.reservation_bytes, vec![total; ranks]);
            let graph_reserves = (0..ranks).map(|rank| graph_reserve_bytes(ranks, rank)).collect::<Result<Vec<_>>>()?;
            let exact = admitted
                .cache_bytes
                .iter()
                .enumerate()
                .map(|(gpu, &bytes)| {
                    let reserve = cuteafd_loader::placement::reserve::gpu("deepseek_v41") as usize
                        + graph_reserves[gpu];
                    let blas = admitted.placement.items[gpu].iter().filter(|item|
                        item.group.starts_with("cuBLAS handles:")
                            || item.group.starts_with("BLAS device initialization"))
                        .map(|item| item.bytes).sum::<u64>() as usize;
                    let mappings = admitted.placement.items[gpu].iter()
                        .filter(|item| item.group == cuteafd_core::memory_layout::DRIVER_PAGE_TABLES_GROUP)
                        .map(|item| item.bytes as usize).sum::<usize>();
                    (bytes + reserve + deferred[gpu] as usize + usage_bytes + blas + mappings, total)
                })
                .collect::<Vec<_>>();
            assert_eq!(
                resolve(&args, &exact, no_experts(), 0, &deferred, 0, None)?.pages,
                admitted.pages
            );
            for gpu in 0..ranks {
                let mut short = exact.clone();
                short[gpu].0 -= 1;
                assert!(
                    resolve(&args, &short, no_experts(), 0, &deferred, 0, None).is_err(),
                    "rank {gpu} cannot borrow the other owner's free bytes"
                );
            }
            // Snapshot arenas are already live in the measured sample; they
            // reduce fit, never the logical-to-physical tail conversion.
            let occupied =
                50 * crate::families::deepseek_v41::v41_backbone_cache::BackboneMark::BYTES;
            let with_snapshots = roomy
                .iter()
                .map(|&(free, total)| (free - occupied, total))
                .collect::<Vec<_>>();
            assert_eq!(
                resolve(&args, &with_snapshots, no_experts(), 0, &deferred, 0, None)?.pages,
                admitted.pages
            );
            args.kv_pool_size = Some(crate::shared::memory::budget::ByteSize(1 << 20));
            args.pool_tokens = None;
            assert!(resolve(&args, &roomy, no_experts(), 0, &deferred, 0, None).is_err());
        }
        Ok(())
    }

    #[test]
    fn invalid_admission_shapes_and_explicit_prefixes_refuse() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("config.json"),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
            )),
        )?;
        let mut args = args(dir.path());
        args.rtx_expert_layers = LocalLayers::Count(3);
        let expert = ExpertCost {
            whole: Bytes2 {
                resident: 1 << 30,
                staging: 20 << 20,
            },
            half: [Bytes2::default(); 2],
            tp2: false,
            whole_supported: true, spark_ok: true,
        };
        let memory = [(80usize << 30, 96usize << 30)];
        let admitted = resolve(&args, &memory, vec![expert; 40], 600 << 20, &[0], 0, None)?;
        assert_eq!(admitted.placement.onboard_layers, 3);
        assert!(resolve(&args, &memory, vec![expert; 2], 600 << 20, &[0], 0, None).is_err());
        args.rtx_expert_layers = LocalLayers::Count(0);
        for context in [1_048_577, u32::MAX] {
            args.max_context_tokens = context;
            assert!(resolve(&args, &memory, no_experts(), 0, &[0], 0, None).is_err());
        }
        args.max_context_tokens = 262_144;
        for concurrency in [0, 65, u32::MAX] {
            args.concurrency = concurrency;
            assert!(resolve(&args, &memory, no_experts(), 0, &[0], 0, None).is_err());
        }
        for concurrency in [17, 32, 33, 64] {
            args.concurrency = concurrency;
            let admitted = resolve(&args, &memory, no_experts(), 0, &[0], 0, None)?;
            assert!(admitted.placement.pool_tokens >= args.max_context_tokens as u64);
        }
        args.concurrency = 16;
        args.prefix.prefix_cache_entries = 129;
        assert!(resolve(&args, &memory, no_experts(), 0, &[0], 0, None).is_err());
        args.prefix.prefix_cache_entries = 20;
        assert!(resolve(&args, &[(97usize << 30, 96usize << 30)], no_experts(), 0, &[0], 0, None).is_err());
        assert!(resolve(&args, &[(0, 0)], no_experts(), 0, &[0], 0, None).is_err());
        Ok(())
    }

    #[test]
    fn unsupported_tp2_modes_name_the_missing_preweight_owners() {
        let dir = tempfile::tempdir().unwrap();
        let mut args = args(dir.path());
        validate_owner_modes(&args).unwrap();
        args.tp2_attention = true;
        let error = validate_owner_modes(&args).unwrap_err().to_string();
        assert!(error.contains("--tp2-attention") && error.contains("missing owner facts"), "{error}");
        args.tp2_attention = false;
        for (query, output) in [(true, false), (false, true), (true, true)] {
            args.tp2_query_projection = query;
            args.tp2_output_projection = output;
            args.rtx_gpus = 1;
            assert!(validate_owner_modes(&args).unwrap_err().to_string().contains("two RTX"));
            args.rtx_gpus = 2;
            validate_owner_modes(&args).unwrap();
        }
    }

    #[test]
    fn omitted_and_zero_pool_consume_identical_live_admission() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut args = args(dir.path());
        args.rtx_expert_layers = LocalLayers::Count(0);
        args.max_context_tokens = 262_144;
        for ranks in [1, 2] {
            args.rtx_gpus = ranks as u32;
            let memory = vec![(80usize << 30, 96usize << 30); ranks];
            args.pool_tokens = None;
            let omitted = resolve(&args, &memory, no_experts(), 0, &vec![0; ranks], 0, None)?;
            args.pool_tokens = Some(0);
            let zero = resolve(&args, &memory, no_experts(), 0, &vec![0; ranks], 0, None)?;
            assert_eq!(omitted.placement, zero.placement);
            assert_eq!(omitted.reservation_bytes, zero.reservation_bytes);
            assert_eq!(zero.reservation_bytes, vec![96usize << 30; ranks]);
            assert_eq!(omitted.host_ledger, zero.host_ledger);
        }
        Ok(())
    }

    /// The same configured reserve admission charged, resolved through the
    /// family list and `GPU_GRAPH_RESERVE`, so a test cannot invent a third
    /// graph budget.
    fn graph_reserve_bytes(ranks: usize, rank: usize) -> Result<usize> {
        let (base, extra) = cuteafd_loader::placement::graph_reserve::v41_default(ranks, rank);
        Ok(cuteafd_loader::placement::graph_reserve::current(rank, false, base, extra)
            .map_err(anyhow::Error::msg)?.bytes as usize)
    }

    #[test]
    fn graph_reserves_and_small_card_logical_target_remain_distinct() -> Result<()> {
        // Each physical rank owns its measured inventory and the shared
        // extra; card size and logical KV target do not change that owner.
        for ranks in [1, 2] {
            for rank in 0..ranks {
                let (base, tail) = cuteafd_loader::placement::graph_reserve::v41_default(ranks, rank);
                let expected = cuteafd_loader::placement::graph_reserve::resolve(None, rank, false, base, tail)?;
                assert_eq!(graph_reserve_bytes(ranks, rank)?, expected.bytes as usize);
            }
        }
        assert_ne!(graph_reserve_bytes(2, 0)?, graph_reserve_bytes(2, 1)?);
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("config.json"),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
            )),
        )?;
        let mut args = args(dir.path());
        args.rtx_expert_layers = LocalLayers::Count(0);
        args.max_context_tokens = 262_144;
        let admitted = resolve(&args, &[(30usize << 30, 32usize << 30)], no_experts(), 0, &[0], 0, None)?;
        assert_eq!(admitted.placement.pool_tokens, 1 << 20);
        assert_eq!(admitted.pages[0], 2048 + 56);
        assert_eq!(admitted.reservation_bytes, vec![32usize << 30]);
        // A GPU_RESERVE that leaves no room for the pool refuses admission.
        let reserved = cuteafd_core::memory_policy::MemoryPolicy { gpu_reserve: Some(31 << 30), ..Default::default() }
            .scoped(|| resolve(&args, &[(30usize << 30, 32usize << 30)], no_experts(), 0, &[0], 0, None));
        assert!(reserved.is_err());
        Ok(())
    }
}

#[cfg(test)]
mod plan_runtime_breakdown {
    /// Prints the decode lane's planner groups beside the runtime owners'
    /// own byte functions at decode width (CUTEAFD_NATIVE_LIB: a WIP or image
    /// library; no device work).
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB"]
    fn decode_lane_groups_against_runtime_byte_functions() -> anyhow::Result<()> {
        use crate::families::deepseek_v41::*;
        let path = std::path::PathBuf::from(std::env::var("CUTEAFD_NATIVE_LIB")?);
        // SAFETY: trusted native library; only plan metadata is read.
        let lib = unsafe { cuteafd_ffi::NativeLibrary::load(&path)? };
        let manifest = super::families::deepseek_v41::owners::native_manifest(&path, "V41_FP8_AOT.json")?;
        let programs = super::families::deepseek_v41::owners::NativePrograms::new(&manifest, 188)?;
        let rows = cuteafd_core::V41_VERIFY_ROWS as u32;
        let plan = programs.decode_lanes(rows as u64, 1, super::families::deepseek_v41::resident::Fp8Head::Off, true)?;
        for d in plan.iter().filter(|d| d.group.starts_with("lane 0 ")) { eprintln!("PLAN {} {}", d.group, d.bytes); }
        let r = rows as usize;
        let lane: usize = v41_backbone_lane::BackboneLane::workspace_bytes(&lib, rows)?.into_iter().sum();
        let index: usize = v41_index_lane::IndexLane::workspace_bytes(&lib, rows)?.into_iter().sum();
        let producers = v41_backbone_execution::BackboneExecution::workspace_bytes(&lib, rows)?;
        let gates = 2 * v41_engram::layer::EngramGate::device_bytes(&lib, r)?;
        let head = v41_target_head::TargetHeadWave::device_bytes(cuteafd_core::V41_VERIFY_ROWS)?;
        for (name, bytes) in [("backbone", lane), ("index", index), ("cache producers+batch", producers),
            ("embedding", v41_target_embedding::TargetEmbeddingWave::device_bytes(r)?),
            ("engram rows", v41_engram::EngramDeviceRows::device_bytes(r)?), ("engram gates", gates),
            ("target head", head), ("target taps", v41_target_pass::TargetTapWave::device_bytes(r)?)] {
            eprintln!("RUNTIME {name} {bytes}");
        }
        Ok(())
    }
}
