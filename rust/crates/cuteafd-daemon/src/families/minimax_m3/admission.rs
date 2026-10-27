//! The server's edge of the shared coordinator allocation recipe: measured free memory, and the
//! ring geometry its Spark links register.
use anyhow::{ensure, Result};
use cuteafd_loader::{
    families::{minimax_m3::config::MiniMaxM3Config, qwen4::Qwen4KvCache}, formats::fp8_experts::ExpertFormat,
    plan::{checkpoint::Checkpoint, families::minimax::admission::{self, Reservations}},
};
use std::path::{Path, PathBuf};
pub(super) use admission::{spark_worlds, LocalLayers, Request, RuntimeLedger, RuntimeSizing};

pub(super) struct Admitted {
    pub cache: admission::Admission,
    pub ledger: RuntimeLedger,
    pub local_layers: Vec<usize>,
    pub spark_layers: Vec<usize>,
    /// Decode steps the warm-up records before ready; their bytes are reserved.
    pub graphs: cuteafd_loader::placement::inventory::StartupGraphs<usize>,
    reservations: Reservations,
}

impl Admitted {
    pub fn readmit(&mut self, cfg: &MiniMaxM3Config, sizing: RuntimeSizing, available: u64,
        requested_pool: Option<u64>, kv_format: Qwen4KvCache) -> Result<()> {
        self.cache = admission::admit(cfg, sizing, &self.ledger, self.reservations, available, requested_pool, kv_format)?;
        Ok(())
    }
}

/// The coordinator's native TP1 expert package in `native_lib`'s image and the manifest at its
/// package root (`tp1/` holds only the module); `None` when the image ships no package, which
/// refuses onboarding.
pub(super) fn local_package(native_lib: &Path) -> Result<(PathBuf, Option<serde_json::Value>)> {
    let directory = crate::shared::experts::fp8::package_directory_for(native_lib, 1, ExpertFormat::Nvfp4,
        cuteafd_core::ExpertGeometry::MINIMAX_M3.family());
    let manifest = cuteafd_loader::placement::inventory::package_manifest(&directory, None)?;
    Ok((directory, manifest))
}

/// The ring geometry the server's Spark link registers.
pub(super) fn spark_rings() -> Result<cuteafd_core::SparkRings> {
    crate::shared::placement::spark_rings(cuteafd_transport::TcpTransportConfig::default().max_frame_bytes)
}

/// The worker admission serve seals to every Spark rank: the one endpoint decode and every prefill
/// wave share (a unit's rows, BF16 in), and the typed owners the shared family admission gives
/// MiniMax's NVFP4 package ladder: one decode owner, and one prefill owner buffering a unit per wave.
pub(super) fn spark_model(sizing: RuntimeSizing) -> Result<cuteafd_transport::worker_selection::ModelAdmission> {
    let endpoint = sizing.spark_endpoint()?;
    let exports = cuteafd_loader::plan::spark_executions::SparkBackend::Fp8.exports();
    let executions = cuteafd_loader::plan::spark_executions::admit_family("minimax_m3", &exports, None,
        sizing.unit_rows as u64, sizing.waves as u64)?;
    cuteafd_transport::worker_selection::ModelAdmission::for_clients(cuteafd_core::ExpertGeometry::MINIMAX_M3,
        endpoint, spark_rings()?, 1, executions)
}

/// Admit `request` into `available` measured free bytes: already capped by the logical device
/// budget, after code/context/media allocation and before target weight allocation.
pub(super) fn solve(checkpoint: &Checkpoint, programs: &serde_json::Value, local_package: Option<&serde_json::Value>,
    request: Request, available: u64, requested_pool: Option<u64>) -> Result<Admitted> {
    let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
    let catalog = cuteafd_loader::read_expert_catalog(&checkpoint.snapshot)?;
    let recipe = admission::Recipe::new(&cfg, checkpoint, &catalog, programs, local_package, request)?;
    let (cache, reservations) = recipe.admit(&cfg, available, requested_pool)?;
    ensure!(cache.total_bytes <= available, "MiniMax admission exceeded its measured free bytes");
    Ok(Admitted { cache, ledger: recipe.ledger, local_layers: recipe.onboarding.local_layers,
        spark_layers: recipe.onboarding.spark_layers, graphs: recipe.graphs, reservations })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::{self, layout::LayoutOptions, testing, ExpertPlacement, PlanOptions};

    fn manifest() -> serde_json::Value {
        serde_json::json!({"programs":admission::PROGRAM_NAMES.iter().map(|suffix| {
            let rows = if suffix.ends_with("_decode") {64} else if matches!(*suffix, "index_prefill" | "attention_extend") {1024} else {4096};
            serde_json::json!({"name":format!("minimax_{suffix}"),"capacity_rows":rows,
                "scratch_bytes_at_capacity":{"scratch":if *suffix == "index_prefill" {134799872} else {53478400}},
                "geometry":{"max_pages":16384,"max_batch":64}})
        }).collect::<Vec<_>>()})
    }

    /// The native expert package beside PROGRAMS.json: TP1 for onboarded layers and, optionally,
    /// the Spark world's layout with its exported scratch.
    fn package(remote_scratch: Option<u64>) -> serde_json::Value {
        let layout = |tp: usize, scratch: u64| serde_json::json!({"hidden":6144,"intermediate":3072,"experts":128,
            "top_k":4,"tp":tp,"slice":3072 / tp,"input":"bf16","weights":"nvfp4","activation":"swigluoai_uninterleave",
            "projection":"bf16","swiglu_limit":7.0,"abi_version":5,"capacities":[{"capacity":256,"scratch_bytes":scratch}]});
        let mut layouts = serde_json::json!({"tp1": layout(1, 102_837_248)});
        if let Some(scratch) = remote_scratch { layouts["tp3"] = layout(3, scratch); }
        serde_json::json!({"layouts": layouts})
    }

    /// Serve and golden read the TP1 package's manifest at the package root, as the image ships it.
    #[test]
    fn minimax_local_package_reads_the_manifest_at_the_package_root() -> Result<()> {
        let image = tempfile::tempdir()?;
        let root = image.path().join("fp8/fp8-minimax-nvfp4a4");
        std::fs::create_dir_all(root.join("tp1"))?;
        std::fs::write(root.join("tp1/libcuteafd_fp8moe.so"), b"module")?;
        let native = image.path().join("libcuteafd_native.so");
        assert_eq!(local_package(&native)?, (root.join("tp1"), None));
        std::fs::write(root.join("manifest.json"), package(None).to_string())?;
        assert_eq!(local_package(&native)?, (root.join("tp1"), Some(package(None))));
        std::fs::write(root.join("manifest.json"), b"not JSON")?;
        assert!(local_package(&native).is_err());
        Ok(())
    }

    /// Serve seals one endpoint (decode and every wave) and the typed owners to every rank: one
    /// decode owner at the package's covering m80 export, and one prefill owner at the unit's
    /// covering export (m4096 for 4096, 2752 and 2048 rows) buffering a unit per wave.
    #[test]
    fn minimax_spark_model_seals_one_endpoint_and_typed_owners() -> Result<()> {
        use cuteafd_core::{DecodeExecution, PrefillExecution};
        for (rows, lanes) in [(4096, 1), (4096, 2), (2752, 3), (2048, 4)] {
            let sizing = RuntimeSizing { unit_rows: rows, decode_rows: 64, sequences: 64, context: 1 << 20, spark_ranks: 6,
                logit_rows: 64, waves: lanes };
            let model = spark_model(sizing)?;
            assert_eq!(model.endpoints, [sizing.spark_endpoint()?]);
            assert_eq!(model.executions.decode, [DecodeExecution { capacity: 64, compiled_capacity: 80 }]);
            assert_eq!(model.executions.prefill, PrefillExecution { capacity: rows as u32, compiled_capacity: 4096,
                unit_rows: rows as u32, buffered_units: lanes as u8 });
            assert_eq!((model.capacity(), model.compiled_capacity()), (rows as u32, 4096));
            model.require_clients(sizing.spark_endpoint()?, 1)?;
        }
        Ok(())
    }

    /// A plan of the test snapshot on `ranks` Sparks at #762's shape (2 x 4096 below six Sparks,
    /// 3 x 2752 at six), and the sizing serve runs it with.
    fn planned(ranks: usize) -> Result<(tempfile::TempDir, cuteafd_loader::plan::PlanReport, RuntimeSizing)> {
        let dir = tempfile::tempdir()?;
        testing::write_minimax_m3_snapshot(dir.path(), false, false);
        let path = dir.path().join("PROGRAMS.json");
        std::fs::write(&path, manifest().to_string())?;
        let report = plan::plan(dir.path(), &PlanOptions { placement: ExpertPlacement::Sparks { ranks },
            layout: Some(LayoutOptions { workspace_manifest: Some(path), concurrency: 16, context_tokens: 4096,
                pool_tokens: Some(4096), headroom_bytes: 0, native_mtp_layers: 0, ..Default::default() }),
            ..Default::default() })?;
        ensure!(report.placement_supported, "TP{ranks}: {:?}", report.hints);
        let shape = admission::prefill_shape(0, 0, ranks)?;
        let sizing = RuntimeSizing { unit_rows: shape.unit_rows as usize, decode_rows: 64, sequences: 16, context: 4096,
            spark_ranks: ranks, logit_rows: 64, waves: shape.waves as usize };
        Ok((dir, report, sizing))
    }

    /// The planned coordinator item for the Spark intake.
    fn planned_intake(report: &cuteafd_loader::plan::PlanReport) -> Option<u64> {
        report.memory_layout.as_ref()?.devices.iter().find(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx)?
            .items.iter().find(|i| i.group == "MiniMax spark_intake").map(|i| i.bytes)
    }

    /// The coordinator's Spark intake is charged as the one link's return slots, a BF16 plane of a
    /// unit's rows per wave and rank, whatever the intake mode: the ledger has no landing input, and
    /// the intake allocates exactly those device planes in GPU and pinned modes alike
    /// (`minimax_spark_intake_allocates_the_planned_return_slots_in_both_modes`, CPU fixture).
    #[test]
    fn minimax_planner_charges_the_link_return_slots_at_the_adaptive_shapes() -> Result<()> {
        for (ranks, unit, waves) in [(2, 4096, 2), (3, 4096, 2), (4, 4096, 2), (6, 2752, 3)] {
            let (_dir, report, sizing) = planned(ranks)?;
            assert_eq!((sizing.unit_rows, sizing.waves, report.prefill_unit_rows, report.prefill_waves),
                (unit, waves, Some(unit as u64), Some(waves as u64)), "TP{ranks}");
            let slots = (waves * ranks * unit * 6144 * 2) as u64;
            assert_eq!(planned_intake(&report), Some(slots), "TP{ranks}");
            assert_eq!(slots, cuteafd_loader::serving_capacity::prefill_return_bytes(waves as u64, unit as u64, ranks as u64, 6144));
        }
        Ok(())
    }

    /// Every Spark rank is charged exactly the one transport serve opens to it: the endpoint serve
    /// seals (a unit's rows, BF16 in), with the rings serve's link registers, and one execution slot
    /// per wave in flight.
    #[test]
    fn minimax_planner_charges_the_one_spark_endpoint_serve_seals() -> Result<()> {
        use cuteafd_core::memory_layout::DeviceKind;
        for ranks in [2, 3, 4, 6] {
            let (_dir, report, sizing) = planned(ranks)?;
            let model = spark_model(sizing)?;
            assert_eq!(model.endpoints, [cuteafd_core::SparkEndpoint { ring_depth: None, capacity: sizing.unit_rows as u32,
                ingress: cuteafd_core::SparkIngress::Bf16 }], "TP{ranks}");
            assert_eq!((model.executions.prefill.buffered_units as usize, report.prefill_transports), (sizing.waves, Some(1)),
                "TP{ranks}");
            assert_eq!(cuteafd_loader::plan::layout::spark_endpoints("minimax_m3", sizing.waves as u64), 1);
            let ring = cuteafd_loader::plan::layout::spark_ring_geometry(1);
            assert_eq!(model.rings, cuteafd_core::SparkRings { max_frame_bytes: ring.max_frame_bytes, depth: ring.depth,
                slot_bytes: ring.slot_bytes, alignment: ring.alignment }, "the planner registers serve's rings");
            // One endpoint's transport on every rank (every routed layer is on the Sparks here).
            let working = cuteafd_loader::placement::SparkWorkingSet { rtx_layers: vec![], spark_layers: (3..60).collect(),
                spark_ranks: ranks, prefill_lanes: 1 };
            let io = working.rank_endpoints_io(0, cuteafd_core::ExpertGeometry::MINIMAX_M3, &model.endpoints, model.rings)?;
            let sparks = report.memory_layout.as_ref().unwrap().devices.iter().filter(|d| d.kind == DeviceKind::Spark)
                .collect::<Vec<_>>();
            assert_eq!(sparks.len(), ranks);
            for spark in sparks {
                let item = |group: &str| spark.items.iter().find(|i| i.group == group).map(|i| i.bytes);
                assert_eq!((item("expert intake"), item("endpoint response prefixes"), item("rdma rings")),
                    (Some(io.intake), Some(io.endpoint_staging), Some(io.mapped_rings)), "TP{ranks} rank {}", spark.index);
                assert_eq!(item("execution slots").is_some(), sizing.waves > 1, "TP{ranks} rank {}", spark.index);
            }
        }
        Ok(())
    }

    /// The Spark link's intake allocates exactly the planned intake on the coordinator GPU, in the
    /// default GPU-landing mode and with `CUTEAFD_SPARK_INTAKE=pinned` alike: a BF16 plane of a unit's
    /// rows per wave and rank. Pinned landing adds host staging (the link's landing staging and a
    /// small-wave buffer), never device memory.
    #[test]
    #[ignore = "CPU-only native fixture; requires an explicit build slot and fixture directory"]
    fn minimax_spark_intake_allocates_the_planned_return_slots_in_both_modes() -> Result<()> {
        use crate::shared::spark_intake::{IntakeMode, SparkIntake};
        use anyhow::Context;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 40)?;
        library.cuda_set_device(0)?;
        for ranks in [2, 3, 4, 6] {
            let (_dir, report, sizing) = planned(ranks)?;
            let planned = planned_intake(&report).context("planned Spark intake")?;
            for mode in [IntakeMode::Gpu, IntakeMode::Pinned] {
                // The fixture's own device accounting (`cuda_memory_info` takes host memory for its
                // integrated-looking device).
                let (before, _) = library.cuda_physical_memory_info()?;
                let intake = SparkIntake::with_waves(&library, mode, ranks, sizing.unit_rows, 6144 * 2, sizing.waves)?;
                let (after, _) = library.cuda_physical_memory_info()?;
                assert_eq!((before - after) as u64, planned, "TP{ranks} {mode:?}: device bytes the intake allocated");
                assert_eq!((0..sizing.waves).flat_map(|slot| intake.slot_buffers(slot)).map(|plane| plane.bytes as u64)
                    .sum::<u64>(), planned, "TP{ranks} {mode:?}: every return slot's rank planes");
                drop(intake);
                assert_eq!(library.cuda_physical_memory_info()?.0, before, "TP{ranks} {mode:?}: the intake released its planes");
            }
        }
        Ok(())
    }

    /// The ledger's extend-launch tables cover the shared executor's cuts: a prefill call cut into
    /// units (`prefill_units`), each unit tiled into extend launches of at most the 1024 rows the
    /// index-prefill and attention-extend programs take.
    #[test]
    fn minimax_extend_launches_cover_every_unit_cut() -> Result<()> {
        use crate::shared::prefill_pipeline::prefill_units;
        for (rows, lanes) in [(4096, 1), (4096, 2), (2752, 3), (2048, 4), (1024, 2), (256, 2)] {
            let sizing = RuntimeSizing { unit_rows: rows, decode_rows: 64, sequences: 1, context: 1 << 20, spark_ranks: 4,
                logit_rows: 64, waves: lanes };
            let bound = admission::extend_launches(sizing, 1024);
            for span in (1..=sizing.slot_rows()).step_by(37).chain([sizing.slot_rows()]) {
                let units = prefill_units(span, rows, lanes, 256)?;
                assert!(units.len() <= lanes && units.iter().all(|unit| unit.len() <= rows), "{lanes} x {rows}: span {span}");
                let tiles: u64 = units.iter().map(|unit| (unit.len() as u64).div_ceil(1024)).sum();
                assert!(tiles <= bound, "{lanes} x {rows}: span {span} cuts into {tiles} tiles, bound {bound}");
            }
        }
        Ok(())
    }

    #[test]
    fn minimax_runtime_and_manifest_plan_share_allocation_bytes() -> Result<()> {
        runtime_and_manifest_plan_share_allocation_bytes(false, None)
    }

    #[test]
    fn minimax_host_embedding_runtime_and_manifest_plan_share_allocation_bytes() -> Result<()> {
        runtime_and_manifest_plan_share_allocation_bytes(true, None)
    }

    /// An exported Spark scratch large enough to onboard layers (each rank runs two execution slots
    /// of it, one per wave) moves both ledgers alike.
    #[test]
    fn minimax_exported_remote_scratch_onboards_alike_in_plan_and_server() -> Result<()> {
        runtime_and_manifest_plan_share_allocation_bytes(false, Some(20 << 30))
    }

    fn runtime_and_manifest_plan_share_allocation_bytes(host_embedding: bool, remote_scratch: Option<u64>) -> Result<()> {
        let dir = tempfile::tempdir()?;
        testing::write_minimax_m3_snapshot(dir.path(), false, false);
        let checkpoint = Checkpoint::open(dir.path())?;
        let programs = manifest();
        let path = dir.path().join("PROGRAMS.json");
        std::fs::write(&path, programs.to_string())?;
        let package = package(remote_scratch);
        std::fs::create_dir_all(dir.path().join("fp8/fp8-minimax-nvfp4a4"))?;
        std::fs::write(dir.path().join("fp8/fp8-minimax-nvfp4a4/manifest.json"), package.to_string())?;
        let options = LayoutOptions { workspace_manifest: Some(path), prefill_rows: 256, concurrency: 16,
            context_tokens: 4096, pool_tokens: Some(4096), headroom_bytes: 0, native_mtp_layers: 0, host_embedding, ..Default::default() };
        let report = plan::plan(dir.path(), &PlanOptions { placement: ExpertPlacement::Sparks { ranks: 3 },
            layout: Some(options.clone()), ..Default::default() })?;
        ensure!(report.placement_supported, "{:?}", report.hints);
        let memory = report.memory_layout.as_ref().unwrap();
        let device = memory.devices.iter().find(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx).unwrap();
        // The planner's runtime items the live sample already holds (context,
        // code); the startup graphs are admitted on top of it, as the server does.
        let graph_groups = ["graphs", cuteafd_loader::placement::inventory::GRAPH_GROWTH];
        let measured_baseline: u64 = device.items.iter().filter(|i| i.category == cuteafd_core::memory_layout::Category::Runtime
            && !graph_groups.contains(&i.group.as_str())).map(|i| i.bytes).sum();
        // The plan's #762 waves at TP3, with its 256-row units.
        let lanes = admission::prefill_shape(256, 0, 3)?.waves as usize;
        assert_eq!(lanes, 2);
        let request = Request {
            sizing: RuntimeSizing { unit_rows: 256, decode_rows: 64, sequences: 16, context: 4096, spark_ranks: 3, logit_rows: 64,
                waves: lanes },
            kv_format: Qwen4KvCache::Fp8, local_layers: LocalLayers::Fewest,
            spark_budget: cuteafd_loader::plan::families::minimax::admission::SPARK_BUDGET_BYTES, spark_vision_bytes: 0,
            headroom: 0, draft_bytes: 0, draft_rings: 0, draft_staging: 0, host_embedding, startup_graphs: false, rings: spark_rings()?,
        };
        let available = options.rtx_bytes[0] - measured_baseline;
        let mut live = solve(&checkpoint, &programs, Some(&package), request.clone(), available, Some(4096))?;
        assert_eq!(live.cache.total_bytes + measured_baseline, device.used_bytes());
        assert_eq!(live.cache.pool_tokens, memory.pool_tokens);
        let onboarded = memory.notes.iter().find_map(|n| n.split_once(" routed layers on the RTX")).unwrap().0;
        assert_eq!(onboarded, live.local_layers.len().to_string());
        assert_eq!(live.local_layers.is_empty(), remote_scratch.is_none());
        assert_eq!(live.spark_layers, (3 + live.local_layers.len()..60).collect::<Vec<_>>());
        if remote_scratch.is_some() { return Ok(()); }
        // Decode runs eagerly: neither ledger reserves startup decode graphs.
        assert!(live.graphs.keys.is_empty() && live.graphs.inventory.bytes(0) == 0);
        assert!(device.items.iter().all(|i| !graph_groups.contains(&i.group.as_str())));
        assert_eq!(live.ledger.bytes("head_workspace"), Some(cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE as u64));
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        assert!(live.readmit(&cfg, request.sizing, live.cache.total_bytes - 1, Some(4096), Qwen4KvCache::Fp8).is_err());
        // A Spark-hosted tower that leaves no room for the experts is refused by name.
        let refused = solve(&checkpoint, &programs, None, Request { spark_vision_bytes: (108 << 30) - (1 << 20), ..request },
            available, Some(4096)).err().map(|error| format!("{error:#}")).unwrap_or_default();
        assert!(refused.starts_with("Spark-hosted vision needs 115.96 GB on every Spark rank, but each TP3 rank has"), "{refused}");
        Ok(())
    }

    /// Automatic vision at the shipped TP2 shapes (64 sequences, the plan's two 4096-row waves, the
    /// checkpoint's context, FP8 K/V, an automatic pool, the default Spark budget and 2 GiB headroom),
    /// speculator off and DFlash2 with its default rings, on an RTX PRO 6000. The planner keeps the
    /// bundled tower on the coordinator GPU, as serve-minimax does, and plans the server's pool: the
    /// text admission, readmitted into what the loaded tower leaves.
    #[test]
    fn minimax_pro6000_auto_vision_plans_the_tower_and_pool_serve_admits() -> Result<()> {
        use cuteafd_loader::{families::{glm5::draft_representation::GlmDraftRepresentation, minimax_m3::dflash::DflashMemory},
            placement::inventory::ArchContext, plan::{encoder::EncoderKind, MediaMode}};
        let rtx = cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES;
        use cuteafd_core::memory_layout::{Category, DeviceKind};
        let dir = tempfile::tempdir()?;
        testing::write_minimax_m3_snapshot(dir.path(), true, false);
        let checkpoint = Checkpoint::open(dir.path())?;
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        let programs = manifest();
        let path = dir.path().join("PROGRAMS.json");
        std::fs::write(&path, programs.to_string())?;
        std::fs::create_dir_all(dir.path().join("fp8/fp8-minimax-nvfp4a4"))?;
        std::fs::write(dir.path().join("fp8/fp8-minimax-nvfp4a4/manifest.json"), package(None).to_string())?;
        let drafter = tempfile::tempdir()?;
        std::fs::write(drafter.path().join("config.json"), testing::minimax_dflash2_config().to_string())?;
        let shape = admission::prefill_shape(0, 0, 2)?;
        let (rows, sequences, lanes, headroom) = (shape.unit_rows as usize, 64, shape.waves as usize, 2u64 << 30);
        let rings = admission::default_draft_rings(sequences, admission::DECODE_ROWS);
        let graph_groups = ["graphs", cuteafd_loader::placement::inventory::GRAPH_GROWTH];
        let arch = ArchContext::coordinator(rtx, None);
        for dflash2 in [false, true] {
            let plan_with = |vision| plan::plan(dir.path(), &PlanOptions { placement: ExpertPlacement::Sparks { ranks: 2 },
                vision, layout: Some(LayoutOptions { rtx_bytes: vec![rtx], workspace_manifest: Some(path.clone()),
                    concurrency: sequences as u64, headroom_bytes: headroom, native_mtp_layers: 0,
                    minimax_drafter_snapshot: dflash2.then(|| drafter.path().to_path_buf()), ..Default::default() }),
                ..Default::default() });
            let (report, text) = (plan_with(MediaMode::Auto)?, plan_with(MediaMode::Off)?);
            ensure!(report.placement_supported, "{:?}", report.hints);
            let encoder = report.encoder.as_ref().unwrap();
            assert_eq!(encoder.kind, EncoderKind::Rtx { gpu: 0 }, "DFlash2 {dflash2}: {}", encoder.reason);
            let memory = report.memory_layout.as_ref().unwrap();
            let device = memory.devices.iter().find(|d| d.kind == DeviceKind::Rtx).unwrap();
            // The live sample already holds the planner's context and code (as above).
            let measured_baseline: u64 = device.items.iter().filter(|i| i.category == Category::Runtime
                && !graph_groups.contains(&i.group.as_str())).map(|i| i.bytes).sum();
            // serve-minimax sizes DFlash2 with the planner's own function.
            let draft = dflash2.then(|| DflashMemory::new(&testing::minimax_dflash2_config(), rings, arch.sms.into(),
                GlmDraftRepresentation::Fp8Only, Qwen4KvCache::Fp8)).transpose()?;
            let sizing = RuntimeSizing { unit_rows: rows, decode_rows: rows.min(admission::DECODE_ROWS), sequences,
                context: cfg.max_position as u64, spark_ranks: 2,
                logit_rows: RuntimeSizing::head_rows(shape.slot_rows() as usize, false, 0), waves: lanes };
            let request = Request { sizing, kv_format: Qwen4KvCache::Fp8, local_layers: LocalLayers::Fewest,
                spark_budget: admission::SPARK_BUDGET_BYTES, spark_vision_bytes: 0, headroom,
                draft_bytes: draft.map_or(0, |d| d.bytes), draft_rings: draft.map_or(0, |d| d.slots),
                draft_staging: draft.map_or(0, |d| d.staging), host_embedding: false, startup_graphs: false,
                rings: spark_rings()? };
            let available = rtx - measured_baseline;
            let mut served = solve(&checkpoint, &programs, Some(&package(None)), request, available, None)?;
            assert_eq!(served.cache.pool_tokens, text.memory_layout.as_ref().unwrap().pool_tokens);
            served.readmit(&cfg, sizing, available - encoder.admitted_bytes(), None, Qwen4KvCache::Fp8)?;
            assert_eq!(served.cache.pool_tokens, memory.pool_tokens, "DFlash2 {dflash2}");
            assert!(served.local_layers.is_empty());
        }
        Ok(())
    }
}
