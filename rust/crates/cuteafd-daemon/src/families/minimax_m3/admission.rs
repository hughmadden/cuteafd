//! Measured pre-weight edge of the shared coordinator allocation recipe.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{
    families::{minimax_m3::config::MiniMaxM3Config, qwen4::Qwen4KvCache},
    placement::SparkWorkingSet,
    plan::{checkpoint::Checkpoint, families::minimax::{admission::{self, Reservations}, coordinator::RankWeightLayout}},
};
pub(super) use admission::{ProgramSizes, RuntimeLedger, RuntimeSizing};

pub(super) struct Request {
    pub sizing: RuntimeSizing,
    /// Measured free bytes, already capped by the logical device budget, after
    /// code/context/media allocation and before target weight allocation.
    pub available: u64,
    pub requested_pool: Option<u64>,
    pub kv_format: Qwen4KvCache,
    pub onboard: Option<usize>,
    pub spark_budget: u64,
    pub headroom: u64,
    pub draft_bytes: u64,
    pub host_embedding: bool,
    /// Prefill lanes: each holds its own endpoint (connections and rings) on every rank.
    pub lanes: usize,
}

pub(super) struct Admitted {
    pub cache: admission::Admission,
    pub ledger: RuntimeLedger,
    pub programs: ProgramSizes,
    pub local_layers: Vec<usize>,
    pub spark_layers: Vec<usize>,
    pub weights: RankWeightLayout,
    pub local_capacity: Option<usize>,
    pub spark_capacity: usize,
    reservations: Reservations,
}

impl Admitted {
    pub fn readmit(&mut self, cfg: &MiniMaxM3Config, sizing: RuntimeSizing, available: u64,
        requested_pool: Option<u64>, kv_format: Qwen4KvCache) -> Result<()> {
        self.cache = admission::admit(cfg, sizing, &self.ledger, self.reservations, available, requested_pool, kv_format)?;
        Ok(())
    }
}

pub(super) fn solve(checkpoint: &Checkpoint, programs: &serde_json::Value,
    local_package: Option<&serde_json::Value>, request: Request) -> Result<Admitted> {
    let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
    request.sizing.validate(&cfg)?;
    cuteafd_loader::families::minimax_m3::cache::MiniMaxCacheLayout::for_runtime(
        &cfg, 1, request.sizing.context, request.kv_format)?;
    let programs = ProgramSizes::from_manifest(programs, request.sizing)?;
    let ledger = RuntimeLedger::new(&cfg, request.sizing, &programs)?;
    let weights = RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?;
    let weights = if request.host_embedding { weights.with_host_embedding(checkpoint)? } else { weights };
    let catalog = cuteafd_loader::read_expert_catalog(&checkpoint.snapshot)?;
    let layers = cfg.moe.iter().enumerate().filter_map(|(layer, &moe)| moe.then_some(layer)).collect();
    let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: layers,
        spark_ranks: request.sizing.spark_ranks, prefill_lanes: request.lanes as u64 };
    let endpoint = cuteafd_core::SparkEndpoint { capacity: request.sizing.rows.try_into()?, ingress: cuteafd_core::SparkIngress::Bf16 };
    let rings = cuteafd_core::SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 };
    let io = working.rank_endpoints_io(0, catalog.routed_experts().geometry()?, &vec![endpoint; request.lanes], rings)?;
    // Remote code/scratch is architecture-specific. This selects owners only;
    // measured worker admission remains mandatory before any transport launch.
    let remote_reserved = [io.intake, io.host_exchange, io.row_indices, io.endpoint_staging, io.mapped_rings,
        512 << 20, cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_BYTES]
        .into_iter().try_fold(0u64, |sum, b| sum.checked_add(b)).context("MiniMax remote reservation overflow")?;
    let onboard = admission::onboard_layers(&catalog, &cfg, request.sizing, request.spark_budget, remote_reserved, request.onboard)?;
    let (expert_workspace, local_capacity) = if onboard.local_layers.is_empty() { (0, None) } else {
        let package = local_package.context("MiniMax onboarded layers require the native TP1 expert package manifest")?;
        let bytes = admission::expert_workspace(package, 1, request.sizing.rows)?;
        let capacity = package["layouts"]["tp1"]["capacities"].as_array().context("MiniMax TP1 package capacities")?
            .iter().filter_map(|c| c["capacity"].as_u64()).filter(|c| *c >= request.sizing.rows as u64)
            .min().context("MiniMax TP1 package row capacity")?;
        (bytes, Some(usize::try_from(capacity)?))
    };
    let mut reservations = Reservations { weights: weights.resident_bytes.checked_add(request.draft_bytes).context("MiniMax draft admission overflow")?, local_experts: onboard.local_bytes,
        expert_workspace, headroom: request.headroom, ..Default::default() };
    reservations.startup = reservations.fixed_bytes(&ledger)?.checked_add(admission::scale_prep_staging_bytes(&weights))
        .context("MiniMax load peak overflow")?;
    let cache = admission::admit(&cfg, request.sizing, &ledger, reservations, request.available,
        request.requested_pool, request.kv_format)?;
    ensure!(cache.total_bytes <= request.available, "MiniMax admission exceeded its measured free bytes");
    Ok(Admitted { cache, ledger, programs, local_layers: onboard.local_layers, spark_layers: onboard.spark_layers,
        weights, local_capacity, spark_capacity: request.sizing.rows, reservations })
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

    #[test]
    fn minimax_runtime_and_manifest_plan_share_allocation_bytes() -> Result<()> {
        runtime_and_manifest_plan_share_allocation_bytes(false)
    }

    #[test]
    fn minimax_host_embedding_runtime_and_manifest_plan_share_allocation_bytes() -> Result<()> {
        runtime_and_manifest_plan_share_allocation_bytes(true)
    }

    fn runtime_and_manifest_plan_share_allocation_bytes(host_embedding: bool) -> Result<()> {
        let dir = tempfile::tempdir()?;
        testing::write_minimax_m3_snapshot(dir.path(), false, false);
        let checkpoint = Checkpoint::open(dir.path())?;
        let programs = manifest();
        let path = dir.path().join("PROGRAMS.json");
        std::fs::write(&path, programs.to_string())?;
        let options = LayoutOptions { workspace_manifest: Some(path), prefill_rows: 256, concurrency: 16,
            context_tokens: 4096, pool_tokens: Some(4096), headroom_bytes: 0, native_mtp_layers: 0, host_embedding, ..Default::default() };
        let report = plan::plan(dir.path(), &PlanOptions { placement: ExpertPlacement::Sparks { ranks: 3 },
            layout: Some(options.clone()), ..Default::default() })?;
        ensure!(report.placement_supported, "{:?}", report.hints);
        let memory = report.memory_layout.as_ref().unwrap();
        let device = memory.devices.iter().find(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx).unwrap();
        let measured_baseline: u64 = device.items.iter().filter(|i| i.category == cuteafd_core::memory_layout::Category::Runtime)
            .map(|i| i.bytes).sum();
        let mut live = solve(&checkpoint, &programs, None, Request {
            sizing: RuntimeSizing { rows: 256, decode_rows: 64, sequences: 16, context: 4096, spark_ranks: 3, logit_rows: 64 },
            available: options.rtx_bytes[0] - measured_baseline, requested_pool: Some(4096),
            kv_format: Qwen4KvCache::Fp8, onboard: None, spark_budget: 108 << 30, headroom: 0, draft_bytes:0, host_embedding,
            lanes: 1,
        })?;
        assert_eq!(live.cache.total_bytes + measured_baseline, device.used_bytes());
        assert_eq!(live.cache.pool_tokens, memory.pool_tokens);
        assert_eq!(live.weights.operands.iter().any(|o| o.name.ends_with("embed_tokens.weight")), !host_embedding);
        assert!(live.weights.operands.iter().any(|o| o.name == "language_model.lm_head.weight"));
        assert!(live.local_layers.is_empty());
        assert_eq!(live.spark_layers, (3..60).collect::<Vec<_>>());
        assert_eq!(live.ledger.bytes("head_workspace"), Some(cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE as u64));
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        assert!(live.readmit(&cfg, RuntimeSizing { rows: 256, decode_rows: 64, sequences: 16, logit_rows: 64,
            context: 4096, spark_ranks: 3 }, live.cache.total_bytes - 1, Some(4096), Qwen4KvCache::Fp8).is_err());
        Ok(())
    }
}
