//! The server's edge of the shared coordinator allocation recipe: measured free memory, and the
//! ring geometry its Spark links register.
use anyhow::{ensure, Result};
use cuteafd_loader::{
    families::{minimax_m3::config::MiniMaxM3Config, qwen4::Qwen4KvCache}, formats::fp8_experts::ExpertFormat,
    plan::{checkpoint::Checkpoint, families::minimax::admission::{self, Reservations}},
};
use std::path::{Path, PathBuf};
pub(super) use admission::{LocalLayers, Request, RuntimeLedger, RuntimeSizing};

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

/// The ring geometry the server's Spark links register.
pub(super) fn spark_rings() -> Result<cuteafd_core::SparkRings> {
    crate::shared::placement::spark_rings(cuteafd_transport::TcpTransportConfig::default().max_frame_bytes)
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

    #[test]
    fn minimax_runtime_and_manifest_plan_share_allocation_bytes() -> Result<()> {
        runtime_and_manifest_plan_share_allocation_bytes(false, None)
    }

    #[test]
    fn minimax_host_embedding_runtime_and_manifest_plan_share_allocation_bytes() -> Result<()> {
        runtime_and_manifest_plan_share_allocation_bytes(true, None)
    }

    /// An exported Spark scratch large enough to onboard layers moves both ledgers alike.
    #[test]
    fn minimax_exported_remote_scratch_onboards_alike_in_plan_and_server() -> Result<()> {
        runtime_and_manifest_plan_share_allocation_bytes(false, Some(40 << 30))
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
        let request = Request {
            sizing: RuntimeSizing { rows: 256, decode_rows: 64, sequences: 16, context: 4096, spark_ranks: 3, logit_rows: 64,
                lanes: cuteafd_loader::serving_capacity::prefill_lanes::DEFAULT_PREFILL_LANES, host_landing: false },
            kv_format: Qwen4KvCache::Fp8, local_layers: LocalLayers::Fewest,
            spark_budget: cuteafd_loader::plan::families::minimax::admission::SPARK_BUDGET_BYTES, spark_vision_bytes: 0,
            headroom: 0, draft_bytes: 0, draft_rings: 0, host_embedding, startup_graphs: true, rings: spark_rings()?,
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
        // The same decode graph keys are recorded by the server and charged by both ledgers.
        assert_eq!(live.graphs.keys, (1..=16).collect::<Vec<_>>());
        let charged: u64 = device.items.iter().filter(|i| graph_groups.contains(&i.group.as_str())).map(|i| i.bytes).sum();
        assert_eq!(charged, live.graphs.inventory.bytes(0));
        assert_eq!(live.ledger.bytes("head_workspace"), Some(cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE as u64));
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        assert!(live.readmit(&cfg, request.sizing, live.cache.total_bytes - 1, Some(4096), Qwen4KvCache::Fp8).is_err());
        // A Spark-hosted tower that leaves no room for the experts is refused by name.
        let refused = solve(&checkpoint, &programs, None, Request { spark_vision_bytes: (108 << 30) - (1 << 20), ..request },
            available, Some(4096)).err().map(|error| format!("{error:#}")).unwrap_or_default();
        assert!(refused.starts_with("Spark-hosted vision needs 115.96 GB on every Spark rank, but each TP3 rank has"), "{refused}");
        Ok(())
    }
}
