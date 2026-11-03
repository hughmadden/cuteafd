//! Startup handle sweep for the two-RTX V4.1 coordinator (`distributed::worker`).
//!
//! Plans the official V4.1 fixture for the measured coordinator GPUs exactly as
//! `cuteafd plan --layout --rtx 2 --coordinator-arch <arch> --spark-ranks 4`
//! does against the library's own manifests, derives serve's arguments and
//! shapes from that plan, then creates every native handle startup creates,
//! on the GPU startup creates it on and at the same sizes, through the Rust
//! FFI. Startup's own seams run where they need no weights (native preload,
//! sizing, plan == runtime expert charges, the dSpark inventory, every
//! weight-free storage owner, attention plan installation, the expert
//! executions over synthetic resident planes); weight-bound owners are
//! replayed call by call. Every refusal is collected, and the test fails once
//! with all of them.
use super::admission;
use crate::families::deepseek_v41::v41_backbone_cache::CachePlacement;
use crate::shared::memory::device::Device;
use crate::shared::memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use std::collections::BTreeMap;

/// Created handles per kind and every refusal, in creation order.
#[derive(Default)]
struct Sweep {
    created: BTreeMap<String, usize>,
    refused: Vec<String>,
}

impl Sweep {
    fn record<T>(&mut self, kind: &str, what: impl std::fmt::Display, result: Result<T>) -> Option<T> {
        match result {
            Ok(value) => {
                *self.created.entry(kind.to_owned()).or_default() += 1;
                Some(value)
            }
            Err(error) => {
                let line = format!("{kind} [{what}]: {error:#}");
                eprintln!("REFUSED {line}");
                self.refused.push(line);
                None
            }
        }
    }
}

/// An FP8 capacity plan as the owners build it: the plan, its full scratch
/// arena and alpha, initialized once on a load stream and drained.
fn fp8_plan(lib: &NativeLibrary, rows: u32, k: u32, n: u32) -> Result<u64> {
    let plan = lib.v41_fp8_matrix_plan(rows, k, n)?;
    let scratch = DeviceAllocation::new(lib, plan.info().scratch_bytes as usize)?;
    let alpha = DeviceAllocation::new(lib, 4)?;
    let stream = LoadStream { library: lib, raw: lib.cuda_stream_create()? };
    // SAFETY: scratch and alpha are distinct live allocations on the current
    // device; the stream drains before either is released.
    let initialized = unsafe { plan.initialize_scratch(scratch.buffer, alpha.buffer, stream.raw) };
    let drained = unsafe { lib.cuda_stream_synchronize(stream.raw) };
    initialized.and(drained)?;
    Ok(plan.info().scratch_bytes)
}

/// A 4 MiB native workspace, as the compressor and vocabulary owners allocate.
fn workspace(lib: &NativeLibrary) -> Result<DeviceAllocation<'_>> {
    DeviceAllocation::new(lib, cuteafd_ffi::V41Compressor::WORKSPACE_BYTES)
}

/// One resident expert layer over zeroed planes in the native packed layout,
/// as a coordinator push publishes it; the execution path never reads them.
fn synthetic_experts<'a>(device: Device<'a>, layer: crate::families::deepseek_v41::v41_experts::ExpertLayer,
    intermediate: usize, experts: usize) -> Result<crate::families::deepseek_v41::v41_experts::ExpertWeights<'a>> {
    use crate::shared::memory::weight_arena::{ArenaLayout, ExpectedCoverage, LoadingArena};
    let planes = cuteafd_loader::push_load::native::NativePlanes::with_scale_codes(5120, intermediate, intermediate,
        experts, false, true)?;
    let arena = LoadingArena::new(device, ArenaLayout::new([planes.bytes], planes.bytes)?, ExpectedCoverage::default())?
        .done()?.initialize(|_| Ok(()))?;
    let budget = crate::families::deepseek_v41::v41_experts::ExpertLoadBudget { resident_bytes: planes.bytes,
        device_staging_bytes: 0, pinned_host_bytes: 0, read_scratch_bytes: 0 };
    device.run(|| crate::families::deepseek_v41::v41_experts::ExpertWeights::from_push(device.library,
        std::rc::Rc::new(arena), 0, layer, &planes, budget))
}

/// Serve's arguments as the launcher derives them from the deployment plan.
fn serve_args(snapshot: &std::path::Path, native_lib: &std::path::Path, unit: u64, waves: u64, reserve: u64)
    -> Result<crate::cli::NativeServeArgs> {
    use clap::{Args, FromArgMatches};
    let (snapshot, native_lib) = (snapshot.display().to_string(), native_lib.display().to_string());
    let (unit, waves, reserve) = (unit.to_string(), waves.to_string(), reserve.to_string());
    let matches = crate::cli::NativeServeArgs::augment_args(clap::Command::new("serve-native"))
        .try_get_matches_from(["serve-native", "--snapshot", &snapshot, "--native-lib", &native_lib,
            "--peers", "127.0.0.1:19441,127.0.0.1:19442,127.0.0.1:19443,127.0.0.1:19444",
            "--rtx-gpus", "2", "--embedding-placement", "host", "--dspark", "--concurrency", "8",
            "--prefix-cache-entries", "20", "--max-context-tokens", "1048576", "--max-output-tokens", "262144",
            "--pool-tokens", "0", "--prefill-batch-tokens", &unit, "--prefill-waves", &waves,
            "--prefill-reserve-rows", &reserve])?;
    Ok(crate::cli::NativeServeArgs::from_arg_matches(&matches)?)
}

/// The deployment plan `cuteafd plan --layout --rtx 2 --coordinator-arch ARCH
/// --spark-ranks 4 --workspace-manifest DIR/PROGRAMS.json` makes of the fixture
/// with the bring-up's settings, and the serve arguments the launcher derives.
fn deployment(arch: cuteafd_core::GpuArch, directory: &std::path::Path, snapshot: &std::path::Path,
    native_lib: &std::path::Path) -> Result<(cuteafd_loader::plan::PlanReport, crate::cli::NativeServeArgs)> {
    use clap::Parser;
    let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan",
        &snapshot.display().to_string(), "--layout", "--rtx", "2", "--coordinator-arch", arch.as_str(),
        "--spark-ranks", "4", "--available-sparks", "4", "--workspace-manifest",
        &directory.join("PROGRAMS.json").display().to_string(), "--speculator", "dspark", "--concurrency", "8",
        "--context-tokens", "1048576", "--prefix-cache-entries", "20", "--pool-tokens", "0",
        "--embedding-placement", "host"])?;
    let crate::cli::Commands::Plan(mut plan_args) = cli.command else { anyhow::bail!("expected the plan command") };
    // The launcher's global VISION=off and AUDIO=off, as main hands them to the plan.
    plan_args.vision = cuteafd_loader::plan::MediaMode::Off;
    plan_args.audio = cuteafd_loader::plan::MediaMode::Off;
    let report = crate::commands::plan::v41_report_for_test(plan_args)?;
    ensure!(report.executable(), "the {arch} plan is not executable:\n{}", cuteafd_loader::plan::render(&report));
    let (unit, waves, reserve) = (report.prefill_unit_rows.context("plan unit")?,
        report.prefill_waves.context("plan waves")?, report.prefill_reserve_rows.context("plan reserve")?);
    let notes = &report.memory_layout.as_ref().context("plan layout")?.notes;
    eprintln!("PLAN arch={arch} decode_lanes={} unit={unit} waves={waves} reserve={reserve} rtx_expert_layers={:?} \
        notes={:?}", report.decode_lanes, report.solver_placement.as_ref().map(|p| p.onboard_layers),
        notes.iter().filter(|n| n.contains("attention")).collect::<Vec<_>>());
    let args = serve_args(snapshot, native_lib, unit, waves, reserve)?;
    Ok((report, args))
}

#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB with its manifests beside it and two coordinator GPUs"]
fn cuda_v41_two_rtx_startup_handles_create_at_planned_sizes() -> Result<()> {
    use crate::families::deepseek_v41::*;
    let native_lib = std::path::PathBuf::from(std::env::var("CUTEAFD_NATIVE_LIB")?);
    let directory = native_lib.parent().context("native library has no directory")?.to_path_buf();
    // SAFETY: the trusted library under test stays loaded for the whole sweep.
    let lib = unsafe { NativeLibrary::load(&native_lib)? };
    let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
    let arch = crate::shared::inventory::device_arch(&lib, 0)?;
    ensure!(crate::shared::inventory::device_arch(&lib, 1)? == arch, "the two coordinator GPUs differ");
    lib.cuda_set_device(0)?;
    for device in devices { device.run(|| lib.cuda_enable_peer(1 - device.id))?; }
    let sms = u32::try_from(lib.sm_count()?)?;
    let mut sweep = Sweep::default();

    // The deployment plan for this architecture's reference cards and the
    // library's manifests, on the official header fixture.
    let fixture = tempfile::tempdir()?;
    cuteafd_loader::plan::testing::write_v41_snapshot(fixture.path());
    let catalog = cuteafd_loader::read_official_v41_catalog(fixture.path())?;
    let (report, args) = deployment(arch, &directory, fixture.path(), &native_lib)?;
    let onboard = report.solver_placement.as_ref().context("plan placement")?.onboard_layers;

    // Serve's shapes, through its own resolution of those arguments.
    let capacity = cuteafd_core::coordinator_programs::v41_live_rows(args.prefill_batch_tokens).context("live rows")?;
    let aot_capacity = super::prefill_capacity(args.prefill_batch_tokens)?;
    let shape = super::AdmittedPrefill::resolve(&args)?;
    ensure!(shape.unit_cap == capacity as usize, "prefill unit {} differs from live rows {capacity}", shape.unit_cap);
    let (slots, reserve) = (shape.waves, shape.reserve_rows());
    let decode = super::DECODE_LANE_ROWS;
    let width = if args.dspark_draft_limit > 5 { 7 } else { 5 };
    let requests = args.concurrency;
    let map = CachePlacement::encoder_decoder();
    let topology = crate::shared::spark_topology::resolve(args.spark_tp, args.spark_ep, args.peers.len(), "serve-native")?;
    let spark_world = crate::shared::spark_topology::world(topology, args.peers.len())?;
    let ranks = spark_world.topology().world_size();
    let snapshot_slots = cuteafd_core::prefix::mark_slots_with_host(0, u64::from(args.resolved_prefix_entries()), 1,
        u64::MAX, args.host_cache_bytes.enabled()) as usize;
    eprintln!("SERVE capacity={capacity} aot={aot_capacity} slots={slots} reserve={reserve} decode={decode} \
        width={width} requests={requests} spark_ranks={ranks} snapshot_slots={snapshot_slots}");

    // 1. Startup's pre-weight seams, as distributed::worker runs them.
    sweep.record("compact reducer", format!("{ranks} ranks"),
        lib.v41_compact_reducer().and_then(|r| r.require_rank_count(ranks as u32)));
    sweep.record("native preload", format!("rows {capacity}, both GPUs"),
        crate::shared::inventory::RuntimeInventory::measure(&lib, &[0, 1],
            crate::shared::inventory::InventoryPhase::PreWeight, |_| admission::preload(&lib, capacity, 2, false)));
    sweep.record("expert demands", capacity, admission::expert_demands(&lib, &catalog, capacity, 2));
    sweep.record("weight bytes", "backbone lane", v41_backbone_lane::BackboneLaneWeights::distributed_device_bytes_with_split(
        &lib, &catalog, map, args.tp2_query_projection, args.tp2_output_projection));
    sweep.record("weight bytes", "cache producers",
        v41_backbone_execution::CacheProducerWeights::distributed_device_bytes(&lib, &catalog, map));
    sweep.record("weight bytes", "index lane", v41_index_lane::IndexLaneWeights::distributed_device_bytes(&lib, &catalog, map));
    sweep.record("weight bytes", "Engram", v41_engram::placement::PlacedEngramWeights::device_bytes(&lib, &catalog, map));
    sweep.record("weight bytes", "target head", v41_target_head::TargetHeadWeights::device_bytes(&catalog));
    for rows in [capacity, decode] {
        sweep.record("workspace bytes", format!("backbone lane {rows}"),
            v41_backbone_lane::BackboneLane::placed_workspace_bytes_with_split(&lib, rows, false, false));
        sweep.record("workspace bytes", format!("cache producers {rows}"),
            v41_backbone_execution::PlacedProducerWaves::device_bytes(&lib, map, rows));
        for gpu in 0..2 {
            sweep.record("workspace bytes", format!("index lane {rows} GPU{gpu}"),
                v41_index_lane::IndexLane::placed_workspace_bytes(&lib, map, rows, gpu));
        }
        sweep.record("workspace bytes", format!("Engram {rows}"),
            v41_engram::placement::PlacedEngram::device_bytes(&lib, map, rows as usize));
    }
    sweep.record("workspace bytes", "lane storage", v41_backbone_lane::LaneStorage::storage_bytes(&lib, reserve as u32,
        capacity, false, false, true));
    sweep.record("workspace bytes", "target head", v41_target_head::distributed_target::DistributedTargetHead::device_bytes(
        cuteafd_core::V41_VERIFY_ROWS, 64640));
    // Plan == runtime for the expert arenas (serve's own check, from the
    // native library's plans and the TP2 owner policy).
    let ffn = |rows: u32, unit: u32| v41_experts::tp2_ffn::Storage::storage_bytes(&lib, rows, unit, false);
    if let Some(arenas) = sweep.record("TP2 FFN arena bytes", format!("{reserve}/{capacity} and 2 x {decode}"),
        (|| Ok(ffn(reserve as u32, capacity)? + 2 * ffn(decode, decode)?))()) {
        sweep.record("plan == runtime", "expert workspaces", admission::verify_expert_charges(&catalog, &native_lib,
            sms, 2, capacity, slots, reserve as u64, arenas as u64));
    }
    sweep.record("dSpark inventory", format!("context {capacity}, {requests} requests, width {width}"),
        devices[1].run(|| v41_experts::dspark::DsparkWeights::distributed_inventory(&lib, &catalog, capacity, requests,
            width, args.tp2_dspark_experts, 64640, super::prefix::Arenas::draft_mark_bytes(snapshot_slots))));
    sweep.record("transport bytes", format!("{decode} rows, {ranks} ranks"),
        v41_experts::coordinator::NativeTp4Wave::device_bytes_for(decode, ranks));

    // Every expert role the plan can use, at each capacity its package
    // exports, on the GPU its handles live on: the dSpark draft (coordinator,
    // GPU1), the single-RTX fallback's full-width backbone (GPU0), the TP2
    // halves and the dSpark TP2 halves (both GPUs). Routed roles also check
    // their compiled reduction attestation (`V41ExpertReduction::validate`).
    for (role, name, gpus) in [(0u32, "coordinator", vec![1usize]), (2, "rtx_backbone", vec![0]),
        (3, "rtx_tp2", vec![0, 1]), (4, "dspark_tp2", vec![0, 1])] {
        let Some(exports) = sweep.record("expert package", name, lib.native_expert_exports(role)) else { continue };
        eprintln!("EXPERTS {name} capacities={:?} decode={:?}", exports.capacities, exports.decode);
        for &gpu in &gpus {
            let mut infos = Vec::new();
            for &capacity in &exports.capacities {
                let at = format!("{name} GPU{gpu} m{capacity}");
                let info = sweep.record("expert kernel", at.clone(), devices[gpu].run(|| {
                    let kernel = match role {
                        0 => lib.v41_expert_kernel(capacity)?,
                        2 => lib.v41_local_expert_kernel(capacity)?,
                        3 => lib.v41_tp2_expert_kernel(capacity)?,
                        _ => lib.v41_dspark_tp2_expert_kernel(capacity)?,
                    };
                    kernel.group_rows_for(0)?;
                    Ok(*kernel.info())
                }));
                if let Some(info) = info.filter(|_| matches!(role, 2 | 3)) {
                    sweep.record("expert reduction", at, lib.native_expert_reduction(&info).map(|recipe| {
                        eprintln!("REDUCTION {name} m{capacity}: {recipe:?}");
                    }));
                    infos.push(info);
                }
            }
            if !infos.is_empty() {
                sweep.record("expert reduction plan", format!("{name} GPU{gpu}"),
                    crate::shared::experts::ExpertReductionPlan::native(&lib, infos).map(drop));
            }
        }
    }
    // The single-RTX fallback's preload (peer access refused at startup).
    sweep.record("native preload", format!("single RTX rows {capacity}"),
        devices[0].run(|| admission::preload(&lib, capacity, 1, false)));

    // 2. The attention plan on every lane's and slot's sparse wave (no weights).
    let plan = sweep.record("attention plan", arch, super::attention_plan(&lib, 0, &args));
    if let Some(plan) = plan {
        eprintln!("ATTENTION rows={} capacity={} compressed={:?} window={:?}", plan.rows(), plan.capacity_rows(),
            plan.sparse_recipe(true), plan.sparse_recipe(false));
        sweep.record("attention plan", "loaded drafter", plan.validate_loaded(Some(width)).map_err(anyhow::Error::from));
        for device in devices {
            for lane in 0..2 {
                sweep.record("sparse attention install", format!("GPU{} decode lane {lane}", device.id), device.run(|| {
                    let mut wave = v41_sparse_attention::SparseAttentionWave::new(&lib, decode as usize,
                        v41_sparse_attention::SparseAttentionWave::device_bytes(decode as usize)?)?;
                    wave.install_attention_plan(plan)
                }));
            }
            let storage = sweep.record("sparse attention storage", format!("GPU{} unit {capacity}", device.id),
                device.run(|| v41_sparse_attention::SparseAttentionStorage::<'_, 64>::new(&lib, capacity as usize, &[0, 1])));
            if let Some(storage) = storage {
                for slot in 0..slots {
                    sweep.record("sparse attention install", format!("GPU{} prefill slot {slot}", device.id), device.run(|| {
                        let mut wave = v41_sparse_attention::SparseAttentionWave::on_storage(storage.clone())?;
                        wave.install_attention_plan(plan)
                    }));
                }
            }
        }
        // The guard stays: SM89 exports no BF16 AOT attention, so a BF16 AOT
        // plan must refuse there (and install where the architecture has it).
        sweep.record("attention guard", "BF16 AOT plan", devices[0].run(|| {
            let bf16_aot = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(plan.capacity_rows(),
                Some(usize::from(args.dspark_draft_limit)), args.attention_plan_rows)?;
            let mut wave = v41_sparse_attention::SparseAttentionWave::new(&lib, decode as usize,
                v41_sparse_attention::SparseAttentionWave::device_bytes(decode as usize)?)?;
            let installed = wave.install_attention_plan(bf16_aot);
            eprintln!("GUARD BF16 AOT plan on {arch}: {}", installed.as_ref().err()
                .map_or_else(|| "installed".to_owned(), |error| format!("{error:#}")));
            ensure!(installed.is_err() == (arch == cuteafd_core::GpuArch::Sm89),
                "a BF16 AOT attention plan {} on {arch}", if installed.is_ok() { "installed" } else { "refused" });
            Ok(())
        }));
    }

    // Plan == runtime per FP8 plan scratch and head scratch of the target
    // owners: the native sizes verify_target_charges compares per owner.
    let fp8_manifest = families_manifest(&native_lib)?;
    let programs = cuteafd_loader::placement::families::deepseek_v41::owners::NativePrograms::new(&fp8_manifest, sms)?
        .with_compressed_attention(arch.v41_compressed_attention());
    let planned = |sweep: &mut Sweep, gpu: Device<'_>, rows: u32, k: u32, n: u32, owner: &str| {
        let runtime = sweep.record("FP8 plan", format!("GPU{} {owner} m{rows} {k}x{n}", gpu.id),
            gpu.run(|| fp8_plan(&lib, rows, k, n)));
        // dSpark's owners are admitted from the runtime's own inventory.
        if let Some(runtime) = runtime.filter(|_| !owner.starts_with("dSpark")) {
            sweep.record("plan == runtime", format!("FP8 scratch {owner} m{rows} {k}x{n}"), (|| {
                let plan = programs.scratch(u64::from(rows), u64::from(k), u64::from(n))?;
                ensure!(plan == runtime, "planner charges {plan} B, the library's plan allocates {runtime} B");
                Ok(())
            })());
        }
    };

    // 3. Backbone lanes: four decode lanes (two per GPU) and one storage set
    // per GPU with three slot lanes on it.
    for device in devices {
        let g = device.id;
        for rows in [decode, capacity] {
            for (k, n) in [(5120, 1280), (1280, 32768), (32768, 8192), (8192, 5120)] {
                let owners = if rows == decode { 2 } else { 1 };
                for _ in 0..owners { planned(&mut sweep, device, rows, k, n, "backbone lane"); }
            }
        }
        for lane in 0..2 {
            for boundary in 0..2 {
                sweep.record("mHC", format!("GPU{g} lane {lane} boundary {boundary}"), device.run(|| {
                    v41_hc::HcStorage::new(&lib, decode as usize)?;
                    lib.v41_hc().map(drop)
                }));
            }
            for owner in ["query", "output"] {
                sweep.record("attention ops", format!("GPU{g} lane {lane} {owner}"), device.run(|| lib.v41_attention_ops().map(drop)));
            }
            sweep.record("router", format!("GPU{g} lane {lane}"), device.run(|| {
                v41_backbone_router::BackboneRouterStorage::new(&lib, decode, decode)?;
                lib.v41_router()?;
                lib.v41_expert_input_quantizer().map(drop)
            }));
            sweep.record("layer clock", format!("GPU{g} lane {lane}"),
                device.run(|| crate::shared::draft::clock::LayerClock::new(&lib, 40).map(drop)));
        }
        sweep.record("lane storage", format!("GPU{g} rows {reserve} unit {capacity}"), device.run(|| {
            v41_hc::HcStorage::new(&lib, reserve)?;
            v41_hc::HcStorage::new(&lib, reserve)?;
            v41_backbone_router::BackboneRouterStorage::new(&lib, reserve as u32, capacity).map(drop)
        }));
        for slot in 0..slots {
            sweep.record("slot lane", format!("GPU{g} slot {slot}"), device.run(|| {
                lib.v41_hc()?;
                lib.v41_hc()?;
                lib.v41_attention_ops()?;
                lib.v41_attention_ops()?;
                lib.v41_router()?;
                lib.v41_expert_input_quantizer()?;
                crate::shared::draft::clock::LayerClock::new(&lib, 40).map(drop)
            }));
        }
        // Scale packing while the layers load (one kernel per matrix and layer).
        for (k, n) in [(5120, 1280), (1280, 32768), (32768, 8192), (8192, 5120), (1280, 4096), (5120, 512)] {
            sweep.record("FP8 load kernel", format!("GPU{g} m1 {k}x{n}"),
                device.run(|| lib.v41_fp8_matrix_kernel(1, k, n).map(drop)));
        }
    }
    for pass in 0..2 + slots {
        for (source, destination) in [(devices[1], devices[0]), (devices[0], devices[1])] {
            sweep.record("residual hop", format!("pass {pass} GPU{}->GPU{}", source.id, destination.id),
                crate::shared::peer_split::hop::DirectInputHop::new(source, destination,
                    cuteafd_loader::placement::families::deepseek_v41::execution_hops(2)).map(drop));
        }
    }

    // 4. Index lanes: selection (source on both GPUs, reindex follower on
    // GPU1), the query projection with its compressor.
    for device in devices {
        let g = device.id;
        for (rows, unit, owners, readers) in [(decode as usize, decode, 2, vec![g]), (reserve, capacity, 1, vec![0, 1])] {
            for owner in 0..owners {
                let storage = sweep.record("index selection storage", format!("GPU{g} rows {rows} unit {unit} #{owner}"),
                    device.run(|| v41_index_selection::IndexSelectionWave::storage(&lib, rows, unit as usize, &readers)));
                if let Some(storage) = storage {
                    if g == 1 {
                        sweep.record("index selection follower", format!("GPU{g} rows {rows}"), device.run(||
                            v41_index_selection::IndexSelectionWave::follower_storage(&storage)
                                .and_then(v41_index_selection::IndexSelectionWave::on_storage).map(drop)));
                    }
                    let waves = if rows == decode as usize { 1 } else { slots };
                    for wave in 0..waves {
                        sweep.record("index selection", format!("GPU{g} rows {rows} wave {wave}"), device.run(||
                            v41_index_selection::IndexSelectionWave::on_storage(storage.clone()).map(drop)));
                    }
                }
                planned(&mut sweep, device, unit, 1280, 4096, "index query");
                sweep.record("compressor", format!("GPU{g} index query unit {unit}"), device.run(|| {
                    let workspace = workspace(&lib)?;
                    // SAFETY: the workspace outlives the handle, dropped first.
                    unsafe { lib.v41_compressor(workspace.buffer) }.map(drop)
                }));
                sweep.record("attention ops", format!("GPU{g} index query"), device.run(|| lib.v41_attention_ops().map(drop)));
            }
        }
    }

    // 5. Cache producers: one window producer per pass and GPU, the source
    // compressors (layers 2, 8, 14 on GPU0; 20 on GPU1), and the slots' set.
    for device in devices {
        let g = device.id as usize;
        let layers: Vec<usize> = (0..40).filter(|&l| usize::from(l >= 20) == g).collect();
        let sources: Vec<usize> = [2, 8, 14, 20].into_iter().filter(|&l| usize::from(l >= 20) == g).collect();
        for lane in 0..2 {
            sweep.record("window storage", format!("GPU{g} lane {lane}"), device.run(||
                v41_window::WindowStorage::new(&lib, decode as usize, decode as usize, &layers, false).map(drop)));
            planned(&mut sweep, device, decode, 5120, 512, "window producer");
            for &layer in &sources {
                sweep.record("compressor storage", format!("GPU{g} lane {lane} layer {layer}"), device.run(||
                    v41_compressor::CompressorStorage::new(&lib, decode as usize, decode as usize, &[layer]).map(drop)));
            }
        }
        for _ in 0..2 + slots {
            sweep.record("window and compressed KV", format!("GPU{g} {} layers", layers.len()), device.run(|| {
                for _ in &layers { lib.v41_attention_ops()?; lib.v41_kv()?; }
                for _ in &sources { lib.v41_compressed_kv()?; lib.v41_attention_ops()?; }
                Ok(())
            }));
        }
    }
    sweep.record("producer storage", format!("rows {reserve} unit {capacity}"),
        v41_backbone_execution::PlacedProducerStorage::new(&lib, map, reserve, capacity as usize).map(drop));
    for device in devices { planned(&mut sweep, device, capacity, 5120, 512, "window producer"); }

    // 6. Engram gates and row uploads (layers 1 and 14, both on GPU0).
    sweep.record("workspace bytes", "Engram gate", devices[0].run(|| v41_engram::layer::EngramGate::device_bytes(&lib, decode as usize)));
    for (rows, unit, owners) in [(decode as usize, decode as usize, 2), (reserve, capacity as usize, 1)] {
        for owner in 0..owners {
            sweep.record("Engram rows", format!("rows {rows} #{owner}"), devices[0].run(|| if rows == unit {
                v41_engram::EngramDeviceRows::new(&lib, rows, v41_engram::EngramDeviceRows::device_bytes(rows)?).map(drop)
            } else { v41_engram::EngramDeviceRows::storage(&lib, rows, unit).map(drop) }));
            for layer in [1, 14] {
                sweep.record("Engram gate storage", format!("rows {rows} unit {unit} layer {layer} #{owner}"),
                    devices[0].run(|| v41_engram::layer::EngramGate::storage(&lib, rows, unit).map(drop)));
            }
            planned(&mut sweep, devices[0], unit as u32, 6144, 25600, "Engram gate");
        }
    }
    sweep.record("FP8 load kernel", "GPU0 m16 6144x25600", devices[0].run(|| lib.v41_fp8_kernel(16).map(drop)));

    // 7. Embedding (GPU0) and target taps (GPU1).
    for (rows, unit, owners) in [(decode as usize, decode as usize, 2), (reserve, capacity as usize, 1)] {
        for owner in 0..owners {
            sweep.record("embedding storage", format!("rows {rows} #{owner}"), devices[0].run(||
                v41_target_embedding::TargetEmbeddingWave::storage(&lib, rows, unit).map(drop)));
            if rows == unit {
                sweep.record("target taps", format!("rows {rows} #{owner}"), devices[1].run(||
                    v41_target_pass::TargetTapWave::new(&lib, rows, v41_target_pass::TargetTapWave::device_bytes(rows)?).map(drop)));
            } else if let Some(storage) = sweep.record("target tap storage", format!("rows {rows} unit {unit}"),
                devices[1].run(|| v41_target_pass::TargetTapWave::storage(&lib, rows, unit))) {
                for slot in 0..slots {
                    sweep.record("target taps", format!("slot {slot}"), devices[1].run(||
                        v41_target_pass::TargetTapWave::on_storage(storage.clone()).map(drop)));
                }
            }
        }
    }
    for _ in 0..2 + slots {
        sweep.record("attention ops", "embedding", devices[0].run(|| lib.v41_attention_ops().map(drop)));
    }

    // 8. Target heads (two decode lanes, every slot): the vocabulary shard on
    // each GPU, its peer copy and FP8 head scratch; normalization on GPU1.
    let head_rows = cuteafd_core::V41_VERIFY_ROWS;
    for head in 0..2 + slots {
        for device in devices {
            sweep.record("vocabulary shard", format!("head {head} GPU{}", device.id), device.run(|| {
                let workspace = workspace(&lib)?;
                // SAFETY: the workspace outlives the projection, dropped first.
                unsafe { lib.v41_vocabulary_shard(workspace.buffer, 64640) }?;
                lib.v41_peer_copy().map(drop)
            }));
            if let Some(scratch) = sweep.record("FP8 head scratch", format!("head {head} GPU{} rows {head_rows}", device.id),
                device.run(|| crate::shared::fp8_linear::scratch(&lib, head_rows, &[(5120, 64640)]).map(|s| s.buffer.bytes))) {
                sweep.record("plan == runtime", format!("FP8 head scratch rows {head_rows}"), (|| {
                    let plan = programs.head_scratch(super::super::v41_tensors::fp8_head(), head_rows as u64, 64640)?;
                    ensure!(plan == scratch as u64, "planner charges {plan} B, the head allocates {scratch} B");
                    Ok(())
                })());
            }
        }
        sweep.record("head normalization", format!("head {head}"), devices[1].run(|| {
            lib.v41_hc()?;
            crate::shared::sampler::TargetSamplingWave::new(&lib, head_rows, 129280).map(drop)
        }));
    }

    // 9. dSpark on GPU1: load kernels, main contexts, three stages per lane,
    // the vocabulary ranks on both GPUs, terminals and committed windows.
    let draft = 40u32;
    sweep.record("dSpark expert info", "m16", devices[1].run(|| lib.v41_expert_info(16).map(drop)));
    sweep.record("expert packer", "2304", devices[1].run(|| lib.v41_expert_packer(2304).map(drop)));
    for stage in 0..3 {
        sweep.record("dSpark expert kernel", format!("load stage {stage} m16"), devices[1].run(|| lib.v41_expert_kernel(16).map(drop)));
    }
    for (rows, k, n) in [(16, 15360, 5120), (16, 5120, 1280), (16, 1280, 32768), (16, 5120, 512), (16, 8192, 5120),
        (16, 5120, 2304), (16, 2304, 5120), (1, 32768, 8192)] {
        sweep.record("FP8 load kernel", format!("GPU1 dSpark m{rows} {k}x{n}"),
            devices[1].run(|| lib.v41_fp8_matrix_kernel(rows, k, n).map(drop)));
    }
    for rows in [capacity, decode] {
        planned(&mut sweep, devices[1], rows, 15360, 5120, "dSpark main context");
        for _ in 0..3 { planned(&mut sweep, devices[1], rows, 5120, 512, "dSpark main context"); }
        sweep.record("attention ops", format!("dSpark main context {rows}"), devices[1].run(|| lib.v41_attention_ops().map(drop)));
    }
    for stage in 0..3 {
        let weights = sweep.record("dSpark synthetic experts", format!("stage {stage}"),
            synthetic_experts(devices[1], v41_experts::ExpertLayer::Dspark { stage }, 2304, 128));
        for lane in 0..2 {
            let at = format!("stage {stage} lane {lane}");
            if let Some(weights) = &weights {
                sweep.record("dSpark expert execution", format!("{at} m{draft}"),
                    devices[1].run(|| weights.execution(draft, draft, usize::MAX).map(drop)));
            }
            sweep.record("mHC", format!("dSpark {at}"), devices[1].run(|| { lib.v41_hc()?; lib.v41_hc().map(drop) }));
            sweep.record("router", format!("dSpark {at}"), devices[1].run(|| lib.v41_router().map(drop)));
            sweep.record("dSpark shared FFN", at.clone(), devices[1].run(|| v41_shared_ffn::SharedFfnStorage::new(&lib, draft).map(drop)));
            for (k, n) in [(5120, 1280), (1280, 32768), (5120, 512), (32768, 8192), (8192, 5120)] {
                planned(&mut sweep, devices[1], draft, k, n, "dSpark stage");
            }
            sweep.record("dSpark attention", format!("{at} width {width}"), devices[1].run(|| {
                lib.v41_attention_ops_width(width)?;
                lib.v41_dspark_attention_width(width).map(drop)
            }));
        }
    }
    for lane in 0..2 {
        sweep.record("attention ops", format!("dSpark chain {lane} width {width}"),
            devices[1].run(|| lib.v41_attention_ops_width(width).map(drop)));
        for device in devices {
            sweep.record("vocabulary shard", format!("dSpark lane {lane} GPU{}", device.id), device.run(|| {
                let workspace = workspace(&lib)?;
                // SAFETY: the workspace outlives the projection, dropped first.
                unsafe { lib.v41_vocabulary_shard(workspace.buffer, 64640) }?;
                lib.v41_peer_copy()?;
                crate::shared::fp8_linear::scratch(&lib, requests.div_ceil(2) as usize * width, &[(5120, 64640)]).map(drop)
            }));
        }
        sweep.record("dSpark terminal", format!("lane {lane}"), devices[1].run(|| {
            let workspace = workspace(&lib)?;
            // SAFETY: the workspace outlives the projection, dropped first.
            unsafe { lib.v41_dspark_markov(workspace.buffer) }?;
            lib.v41_dspark_confidence()?;
            lib.v41_draft_step()?;
            lib.v41_hc().map(drop)
        }));
    }
    for window in 0..3 {
        sweep.record("dSpark window", format!("#{window} {requests} slots, {capacity} rows"), devices[1].run(||
            v41_dspark_cache::DsparkWindow::new(&lib, requests as usize, capacity,
                v41_dspark_cache::DsparkWindow::device_bytes(requests as usize, capacity)?).map(drop)));
    }

    // 10. TP2 FFN on both GPUs (prefill unit and decode width): shared halves,
    // the owner reduce and the routed executions over the onboarded layers.
    if onboard > 0 {
        use crate::shared::experts::rtx::{native::NativeTp2, pair::OwnerPolicy};
        let policy = OwnerPolicy::native(&lib, false)?;
        let OwnerPolicy::Native { selection, output, .. } = policy else { anyhow::bail!("native TP2 owner policy") };
        for device in devices {
            let rank = device.id as usize;
            sweep.record("expert packer", format!("GPU{rank} 1152"), device.run(|| lib.v41_expert_packer(1152).map(drop)));
            sweep.record("expert load kernels", format!("GPU{rank}"),
                device.run(|| crate::shared::fp8_load_quant::LoadKernels::new(&lib).map(drop)));
            for (k, n) in [(5120, 1152), (1152, 5120)] {
                sweep.record("FP8 load kernel", format!("GPU{rank} m1 {k}x{n}"), device.run(|| lib.v41_fp8_matrix_kernel(1, k, n).map(drop)));
            }
            let weights = sweep.record("TP2 synthetic experts", format!("GPU{rank} layer 0"),
                synthetic_experts(device, v41_experts::ExpertLayer::BackboneTp2 { layer: 0, rank }, 1152, 384));
            let weights = weights.and_then(|w| sweep.record("TP2 weight owner", format!("GPU{rank}"),
                device.own(|| Ok(vec![w])).map(std::rc::Rc::new)));
            for (rows, owners) in [(capacity, 1), (decode, 2)] {
                for owner in 0..owners {
                    let at = format!("GPU{rank} rows {rows} #{owner}");
                    for (k, n) in [(5120, 1152), (1152, 5120)] {
                        sweep.record("FP8 plan", format!("{at} shared TP2 {k}x{n}"), device.run(|| fp8_plan(&lib, rows, k, n)));
                    }
                    sweep.record("shared TP2", at.clone(), device.run(|| { lib.v41_shared_tp2_swiglu()?; lib.bf16_add().map(drop) }));
                    for _ in 0..2 {
                        sweep.record("peer transfer", at.clone(), crate::shared::memory::device::PeerTransfer::new(
                            devices[1 - rank], device).map(drop));
                    }
                    sweep.record("owner reduce", at.clone(), crate::shared::experts::rtx::combine::OwnerReduce::new(devices,
                        crate::shared::experts::rtx::combine::Combine::OwnerReduce { owner: rank as u8 }, rows,
                        cuteafd_core::ExpertGeometry::DEEPSEEK_V41).map(drop));
                    if let Some(weights) = &weights {
                        sweep.record("TP2 expert execution", at.clone(), (|| {
                            let workspace = NativeTp2::workspace_bytes_for_format(&lib, rows as usize, selection, output, false)?;
                            NativeTp2::execution(device, rank as u8, vec![0], rows as usize, workspace, weights.clone(),
                                selection, output).map(drop)
                        })());
                    }
                    sweep.record("peer copy", at.clone(), device.run(|| { lib.v41_peer_copy()?; lib.bf16_add().map(drop) }));
                }
            }
        }
    }

    // 11. Transports on GPU1 (unlinked slots; links need live Sparks) and the
    // exchange's native setup, prefix arenas on GPU0.
    for slot in 1..slots {
        sweep.record("transport", format!("unlinked slot {slot} rows {capacity}"), devices[1].run(||
            v41_experts::coordinator::NativeTp4Wave::unlinked(&lib, capacity).map(drop)));
    }
    sweep.record("exchange setup", format!("{ranks} ranks"), devices[1].run(|| {
        lib.v41_compact_reducer()?.require_plane_reduction(ranks as u32)?;
        lib.tp2_native_initialize()?;
        lib.peer_exchange_initialize()
    }));
    sweep.record("prefix arenas", format!("{snapshot_slots} slots"),
        super::prefix::Arenas::new(devices[0], None, snapshot_slots).map(drop));

    for device in devices {
        sweep.record("device drained", format!("GPU{}", device.id), device.run(|| lib.cuda_device_synchronize()));
    }
    eprintln!("CREATED {} creation steps in {} kinds:", sweep.created.values().sum::<usize>(), sweep.created.len());
    for (kind, count) in &sweep.created { eprintln!("  {count:6}  {kind}"); }
    eprintln!("REFUSED {}", sweep.refused.len());
    for line in &sweep.refused { eprintln!("  {line}"); }
    ensure!(sweep.refused.is_empty(), "{} startup handles refused on {arch}:\n{}", sweep.refused.len(),
        sweep.refused.join("\n"));
    Ok(())
}

/// The V4.1 FP8 allocation manifest beside the library (serve's lookup).
fn families_manifest(native_lib: &std::path::Path) -> Result<serde_json::Value> {
    cuteafd_loader::placement::families::deepseek_v41::owners::native_manifest(native_lib, "V41_FP8_AOT.json")
}

/// One coordinator GPU: every V4.1 native handle kind at every capacity its
/// package exports, without the two-GPU planned layout (no peers, no TP2
/// pair, no planned-size storage). Expert roles check their reduction
/// attestation; the attention plan installs with the device's compressed
/// backend, and a BF16 AOT plan must refuse on SM89.
#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB with its manifests beside it and a coordinator GPU"]
fn cuda_v41_single_gpu_handles_create_at_every_exported_capacity() -> Result<()> {
    use crate::families::deepseek_v41::*;
    let native_lib = std::path::PathBuf::from(std::env::var("CUTEAFD_NATIVE_LIB")?);
    // SAFETY: the trusted library under test stays loaded for the whole sweep.
    let lib = unsafe { NativeLibrary::load(&native_lib)? };
    lib.cuda_set_device(0)?;
    let device = Device { library: &lib, id: 0 };
    let arch = crate::shared::inventory::device_arch(&lib, 0)?;
    let sms = u32::try_from(lib.sm_count()?)?;
    let (_, total) = lib.cuda_memory_info()?;
    eprintln!("DEVICE arch={arch} sms={sms} total={total}");
    let mut sweep = Sweep::default();

    // Every FP8 geometry as a capacity plan (each exported kernel capacity,
    // drafter covers included) with its scratch initialized, and the
    // planner's charge from the library's manifest beside it.
    let fp8_manifest = families_manifest(&native_lib)?;
    let programs = cuteafd_loader::placement::families::deepseek_v41::owners::NativePrograms::new(&fp8_manifest, sms)?
        .with_compressed_attention(arch.v41_compressed_attention());
    let geometries = [(6144, 25600), (5120, 2304), (2304, 5120), (5120, 1152), (1152, 5120), (15360, 5120),
        (5120, 1280), (1280, 32768), (1280, 16384), (1280, 4096), (5120, 512), (8192, 5120), (8192, 2560), (32768, 8192)];
    for (k, n) in geometries {
        for rows in [1, 16, 40, cuteafd_core::V41_VERIFY_ROWS as u32, 256, 1024, 4096] {
            let runtime = sweep.record("FP8 plan", format!("m{rows} {k}x{n}"), fp8_plan(&lib, rows, k, n));
            if let Some(runtime) = runtime.filter(|_| rows != 40 && n != 16384 && n != 2560) {
                sweep.record("plan == runtime", format!("FP8 scratch m{rows} {k}x{n}"), (|| {
                    let plan = programs.scratch(u64::from(rows), u64::from(k), u64::from(n))?;
                    ensure!(plan == runtime, "planner charges {plan} B, the library's plan allocates {runtime} B");
                    Ok(())
                })());
            }
        }
        sweep.record("FP8 info", format!("{k}x{n}"), lib.v41_fp8_matrix_info(1, k, n).map(drop));
    }
    sweep.record("FP8 kernel", "Engram m16", lib.v41_fp8_kernel(16).map(drop));
    for rows in [16, cuteafd_core::V41_VERIFY_ROWS] {
        for vocabulary in [64640, 129280] {
            if let Some(scratch) = sweep.record("FP8 head scratch", format!("rows {rows} vocab {vocabulary}"),
                crate::shared::fp8_linear::scratch(&lib, rows, &[(5120, vocabulary)]).map(|s| s.buffer.bytes)) {
                sweep.record("plan == runtime", format!("FP8 head scratch rows {rows} vocab {vocabulary}"), (|| {
                    let plan = programs.vocabulary_scratch(rows as u64, vocabulary as u64)?;
                    ensure!(plan == scratch.max(256) as u64, "planner charges {plan} B, the head allocates {scratch} B");
                    Ok(())
                })());
            }
            sweep.record("packed head workspace", format!("rows {rows} vocab {vocabulary}"),
                lib.packed_head_workspace(rows, 5120, vocabulary).map(drop));
        }
    }

    // Every expert role at every exported capacity, with the routed roles'
    // reduction attestation, then the single-RTX preload.
    for (role, name) in [(0u32, "coordinator"), (2, "rtx_backbone"), (3, "rtx_tp2"), (4, "dspark_tp2")] {
        let Some(exports) = sweep.record("expert package", name, lib.native_expert_exports(role)) else { continue };
        eprintln!("EXPERTS {name} capacities={:?} decode={:?}", exports.capacities, exports.decode);
        let mut infos = Vec::new();
        for &capacity in &exports.capacities {
            let at = format!("{name} m{capacity}");
            let info = sweep.record("expert kernel", at.clone(), (|| {
                let kernel = match role {
                    0 => lib.v41_expert_kernel(capacity)?,
                    2 => lib.v41_local_expert_kernel(capacity)?,
                    3 => lib.v41_tp2_expert_kernel(capacity)?,
                    _ => lib.v41_dspark_tp2_expert_kernel(capacity)?,
                };
                kernel.group_rows_for(0)?;
                Ok(*kernel.info())
            })());
            if let Some(info) = info {
                eprintln!("EXPERT {name} m{capacity}: scratch {} B, max_active_clusters {}, input_dtype {}",
                    info.scratch_bytes, info.max_active_clusters, info.input_dtype);
                if matches!(role, 2 | 3) {
                    sweep.record("expert reduction", at, lib.native_expert_reduction(&info).map(|recipe| {
                        eprintln!("REDUCTION {name} m{capacity}: {recipe:?}");
                    }));
                    infos.push(info);
                }
            }
        }
        if !infos.is_empty() {
            sweep.record("expert reduction plan", name,
                crate::shared::experts::ExpertReductionPlan::native(&lib, infos).map(drop));
        }
    }
    sweep.record("native preload", "single RTX rows 2752", admission::preload(&lib, 2752, 1, false));
    for intermediate in [2304, 1152] {
        sweep.record("expert packer", intermediate, lib.v41_expert_packer(intermediate).map(drop));
    }
    sweep.record("expert load kernels", "", crate::shared::fp8_load_quant::LoadKernels::new(&lib).map(drop));
    sweep.record("expert input quantizer", "", lib.v41_expert_input_quantizer().map(drop));
    // The draft experts' execution as dSpark binds it (m40) and the TP2 rank's
    // executions over every capacity (max rows 4096) and the decode width.
    if let Some(weights) = sweep.record("synthetic experts", "dSpark stage 0",
        synthetic_experts(device, v41_experts::ExpertLayer::Dspark { stage: 0 }, 2304, 128)) {
        sweep.record("dSpark expert execution", "m40", weights.execution(40, 40, usize::MAX).map(drop));
    }
    {
        use crate::shared::experts::rtx::{native::NativeTp2, pair::OwnerPolicy};
        let OwnerPolicy::Native { selection, output, .. } = OwnerPolicy::native(&lib, false)?
            else { anyhow::bail!("native TP2 owner policy") };
        let weights = sweep.record("synthetic experts", "TP2 rank 0",
            synthetic_experts(device, v41_experts::ExpertLayer::BackboneTp2 { layer: 0, rank: 0 }, 1152, 384))
            .and_then(|w| sweep.record("TP2 weight owner", "rank 0", device.own(|| Ok(vec![w])).map(std::rc::Rc::new)));
        if let Some(weights) = weights {
            for rows in [cuteafd_core::V41_VERIFY_ROWS, 4096] {
                sweep.record("TP2 expert execution", format!("max rows {rows}"), (|| {
                    let workspace = NativeTp2::workspace_bytes_for_format(&lib, rows, selection, output, false)?;
                    NativeTp2::execution(device, 0, vec![0], rows, workspace, weights.clone(), selection, output).map(drop)
                })());
            }
        }
    }
    for ranks in 1..=8u32 {
        sweep.record("compact reducer", format!("{ranks} ranks"), lib.v41_compact_reducer().and_then(|r| {
            r.require_rank_count(ranks)?;
            r.require_plane_reduction(ranks)
        }));
    }
    sweep.record("route reducer", "", lib.v41_route_reducer().map(drop));
    sweep.record("TP2 expert reducer", "route sums", lib.v41_tp2_expert_reducer().and_then(|r| {
        ensure!(r.supports_route_sums(), "no ordered FP32 route sums");
        Ok(())
    }));
    sweep.record("local expert reducer", "", lib.v41_local_expert_reducer().map(drop));

    // The attention plan with the device's compressed backend, on decode and
    // storage-backed sparse waves, and the guard on a BF16 AOT plan.
    let args = serve_args(std::path::Path::new("/nonexistent"), &native_lib, 256, 3, 8192)?;
    if let Some(plan) = sweep.record("attention plan", arch, super::attention_plan(&lib, 0, &args)) {
        eprintln!("ATTENTION rows={} capacity={} compressed={:?} window={:?}", plan.rows(), plan.capacity_rows(),
            plan.sparse_recipe(true), plan.sparse_recipe(false));
        sweep.record("sparse attention install", "decode wave", (|| {
            let rows = cuteafd_core::V41_VERIFY_ROWS;
            let mut wave = v41_sparse_attention::SparseAttentionWave::new(&lib, rows,
                v41_sparse_attention::SparseAttentionWave::device_bytes(rows)?)?;
            wave.install_attention_plan(plan)
        })());
        if let Some(storage) = sweep.record("sparse attention storage", "unit 256",
            v41_sparse_attention::SparseAttentionStorage::<'_, 64>::new(&lib, 256, &[0])) {
            sweep.record("sparse attention install", "storage wave", (|| {
                let mut wave = v41_sparse_attention::SparseAttentionWave::on_storage(storage.clone())?;
                wave.install_attention_plan(plan)
            })());
        }
        sweep.record("attention guard", "BF16 AOT plan", (|| {
            let bf16_aot = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(plan.capacity_rows(),
                Some(usize::from(args.dspark_draft_limit)), args.attention_plan_rows)?;
            let rows = cuteafd_core::V41_VERIFY_ROWS;
            let mut wave = v41_sparse_attention::SparseAttentionWave::new(&lib, rows,
                v41_sparse_attention::SparseAttentionWave::device_bytes(rows)?)?;
            let installed = wave.install_attention_plan(bf16_aot);
            eprintln!("GUARD BF16 AOT plan on {arch}: {}", installed.as_ref().err()
                .map_or_else(|| "installed".to_owned(), |error| format!("{error:#}")));
            ensure!(installed.is_err() == (arch == cuteafd_core::GpuArch::Sm89),
                "a BF16 AOT attention plan {} on {arch}", if installed.is_ok() { "installed" } else { "refused" });
            Ok(())
        })());
    }
    sweep.record("sparse attention", "32 heads", lib.v41_sparse_attention_heads32().map(drop));

    // Every remaining handle kind, through the weight-free owners at decode
    // width and an exported 256-row unit.
    let rows = cuteafd_core::V41_VERIFY_ROWS;
    sweep.record("mHC", "storage and projection", v41_hc::HcStorage::new(&lib, rows).and_then(|_| lib.v41_hc().map(drop)));
    sweep.record("router", "storage, router and quantizer", (|| {
        v41_backbone_router::BackboneRouterStorage::new(&lib, rows as u32, rows as u32)?;
        lib.v41_router()?;
        lib.v41_expert_input_quantizer().map(drop)
    })());
    for width in [5, 7] {
        sweep.record("attention ops", format!("width {width}"), lib.v41_attention_ops_width(width).map(drop));
        sweep.record("dSpark attention", format!("width {width}"), lib.v41_dspark_attention_width(width).map(drop));
    }
    sweep.record("layer clock", "40 layers", crate::shared::draft::clock::LayerClock::new(&lib, 40).map(drop));
    for (rows, unit) in [(rows, rows), (256, 256)] {
        if let Some(storage) = sweep.record("index selection storage", format!("rows {rows} unit {unit}"),
            v41_index_selection::IndexSelectionWave::storage(&lib, rows, unit, &[0])) {
            sweep.record("index selection follower", format!("rows {rows}"),
                v41_index_selection::IndexSelectionWave::follower_storage(&storage)
                    .and_then(v41_index_selection::IndexSelectionWave::on_storage).map(drop));
            sweep.record("index selection", format!("rows {rows}"),
                v41_index_selection::IndexSelectionWave::on_storage(storage).map(drop));
        }
        sweep.record("producer storage", format!("rows {rows} unit {unit}"),
            v41_backbone_execution::ProducerStorage::new(&lib, rows, unit).map(drop));
        sweep.record("Engram gate storage", format!("rows {rows} unit {unit}"),
            v41_engram::layer::EngramGate::storage(&lib, rows, unit).map(drop));
        sweep.record("Engram rows", format!("rows {rows} unit {unit}"),
            v41_engram::EngramDeviceRows::storage(&lib, rows, unit).map(drop));
        sweep.record("embedding storage", format!("rows {rows} unit {unit}"),
            v41_target_embedding::TargetEmbeddingWave::storage(&lib, rows, unit).map(drop));
        if let Some(storage) = sweep.record("target tap storage", format!("rows {rows} unit {unit}"),
            v41_target_pass::TargetTapWave::storage(&lib, rows, unit)) {
            sweep.record("target taps", format!("rows {rows}"), v41_target_pass::TargetTapWave::on_storage(storage).map(drop));
        }
    }
    for layers in [vec![2usize], vec![20]] {
        sweep.record("compressor storage", format!("layers {layers:?}"),
            v41_compressor::CompressorStorage::new(&lib, rows, rows, &layers).map(drop));
    }
    sweep.record("window and KV", "", (|| {
        lib.v41_kv()?;
        lib.v41_compressed_kv()?;
        lib.v41_kv_store_layers().map(drop)
    })());
    sweep.record("index scoring", "scores, top-k, candidate blocks", (|| {
        lib.v41_index_scores()?;
        lib.v41_index_topk()?;
        lib.v41_candidate_blocks().map(drop)
    })());
    sweep.record("compressor", "4 MiB workspace", (|| {
        let workspace = workspace(&lib)?;
        // SAFETY: the workspace outlives the handle, dropped first.
        unsafe { lib.v41_compressor(workspace.buffer) }.map(drop)
    })());
    sweep.record("grouped output", "4 MiB workspace", (|| {
        let workspace = workspace(&lib)?;
        // SAFETY: the workspace outlives the handle, dropped first.
        unsafe { lib.v41_grouped_output(workspace.buffer) }.map(drop)
    })());
    for (kind, vocabulary) in [("vocabulary head", 0usize), ("vocabulary shard", 64640), ("Markov", 1)] {
        sweep.record(kind, vocabulary, (|| {
            let workspace = workspace(&lib)?;
            // SAFETY: the workspace outlives the projection, dropped first.
            unsafe { match vocabulary {
                0 => lib.v41_vocabulary_head(workspace.buffer),
                1 => lib.v41_dspark_markov(workspace.buffer),
                rows => lib.v41_vocabulary_shard(workspace.buffer, rows),
            } }.map(drop)
        })());
    }
    sweep.record("target sampling", "rows 64", crate::shared::sampler::TargetSamplingWave::new(&lib, rows, 129280).map(drop));
    sweep.record("dSpark heads", "confidence, draft step, cache", (|| {
        lib.v41_dspark_confidence()?;
        lib.v41_draft_step()?;
        lib.v41_dspark_cache().map(drop)
    })());
    sweep.record("dSpark window", "8 slots, 256 rows", v41_dspark_cache::DsparkWindow::new(&lib, 8, 256,
        v41_dspark_cache::DsparkWindow::device_bytes(8, 256)?).map(drop));
    for unit in [40, rows as u32] {
        sweep.record("shared FFN storage", unit, v41_shared_ffn::SharedFfnStorage::new(&lib, unit).map(drop));
    }
    sweep.record("shared SwiGLU", "full and TP2", (|| {
        lib.v41_shared_swiglu()?;
        lib.v41_shared_tp2_swiglu().map(drop)
    })());
    sweep.record("device ops", "peer copy and BF16 add", (|| { lib.v41_peer_copy()?; lib.bf16_add().map(drop) })());
    sweep.record("transport", "unlinked rows 64", v41_experts::coordinator::NativeTp4Wave::unlinked(&lib, rows as u32).map(drop));
    sweep.record("exchange setup", "", (|| { lib.tp2_native_initialize()?; lib.peer_exchange_initialize() })());
    sweep.record("prefix arenas", "4 slots", super::prefix::Arenas::new(device, None, 4).map(drop));

    sweep.record("device drained", "GPU0", lib.cuda_device_synchronize());
    eprintln!("CREATED {} creation steps in {} kinds:", sweep.created.values().sum::<usize>(), sweep.created.len());
    for (kind, count) in &sweep.created { eprintln!("  {count:6}  {kind}"); }
    eprintln!("REFUSED {}", sweep.refused.len());
    for line in &sweep.refused { eprintln!("  {line}"); }
    ensure!(sweep.refused.is_empty(), "{} handles refused on {arch} ({sms} SMs):\n{}", sweep.refused.len(),
        sweep.refused.join("\n"));
    Ok(())
}
