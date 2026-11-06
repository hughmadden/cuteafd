//! serve-dsv4 admission's CPU half: the `placement` request from the
//! engine's arguments, its loaded config and one sample per GPU. The CUDA
//! half (samples, native workspace) is the caller's, so the equality test
//! drives this exact path with fake samples.
use super::EngineArgs;
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::deepseek_v4::DeepseekV4Config;
use cuteafd_loader::placement::{families::deepseek_v4 as v4, planner::Pin, Baseline, Onboard};

pub(crate) struct Inputs<'a> {
    pub cfg: &'a DeepseekV4Config,
    pub catalog: &'a cuteafd_loader::OfficialV41Catalog,
    pub manifest: &'a serde_json::Value,
    /// `dsv4f` / `dsv4p`.
    pub family: &'a str,
    /// (total bytes, baseline) per GPU: `--device` then `--split-device`.
    pub gpus: Vec<(u64, Baseline)>,
    /// None before uploads; exact frozen modes after uploads. Rebase measured
    /// free bytes by their non-embedding weights, which remain explicit demands.
    pub loaded_modes: Option<&'a [cuteafd_loader::placement::LayerMode]>,
    /// dSpark stages the loaded drafter owns caches for.
    pub cache_stages: usize,
    pub prefill_rows: usize,
    pub decode_rows: usize,
    pub max_context: usize,
    pub prefix: Option<&'a crate::shared::prefix::PrefixArgs>,
    /// The local expert arena's workspace; `None` when this build lacks the
    /// coordinator expert kernels.
    pub expert_workspace: Option<u64>,
    /// Actual per-rank TP2 expert workspace probe; None means no TP2 package.
    pub tp2_workspace: Option<[u64; 2]>,
    pub exchange_f32: bool,
    /// Each GPU's physical capacity beside its logical ceiling in `gpus`.
    pub physical: Vec<u64>,
    /// Host RAM (total, available) sampled before any allocation.
    pub host_memory: (u64, u64),
}

#[cfg(test)]
mod spark_tests {

    fn planned(family: &str, backend: cuteafd_loader::plan::spark_executions::SparkBackend, decode: Option<u32>,
        unit: u64, waves: u64) -> cuteafd_core::SparkExecutions {
        let options = cuteafd_loader::plan::layout::LayoutOptions { prefill_unit_rows: Some(unit),
            prefill_waves: waves, ..Default::default() };
        cuteafd_loader::plan::layout::planned_spark_executions(family, backend, decode, &options).unwrap()
    }

    #[test]
    fn planner_and_serve_admit_identical_spark_owners() -> anyhow::Result<()> {
        use cuteafd_loader::plan::spark_executions::SparkBackend;
        for backend in [SparkBackend::Fp8, SparkBackend::Native, SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest)] {
            for (waves, unit) in [(1usize, 4096usize), (2, 4096), (3, 2752), (4, 2048)] {
                assert_eq!(super::spark_model(backend, 64, unit, waves)?.executions,
                    planned("deepseek_v4", backend, Some(64), unit as u64, waves as u64));
            }
        }
        Ok(())
    }

    #[test]
    fn v4_spark_endpoints_match_the_decode_and_prefill_links() -> anyhow::Result<()> {
        use cuteafd_loader::plan::spark_executions::SparkBackend;
        for (decode_rows, compiled) in [(16, 16), (64, 80), (80, 80)] {
            let model = super::spark_model(SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest), decode_rows, 2752, 3)?;
            model.validate(cuteafd_core::ExpertGeometry::DEEPSEEK_V4_FLASH)?;
            // The static prefill owner holds the selected unit; EXL3 exports m2752.
            assert_eq!((model.capacity(), model.compiled_capacity()), (2752, 2752));
            assert_eq!(model.executions.decode, [cuteafd_core::DecodeExecution {
                capacity: decode_rows as u32, compiled_capacity: compiled }]);
            assert_eq!((model.executions.prefill.unit_rows, model.executions.prefill.buffered_units), (2752, 3));
            assert_eq!(model.endpoints.len(), 2);
            assert_eq!(model.endpoints[0].capacity, decode_rows as u32);
            assert_eq!(model.endpoints[1].capacity, 2752);
            assert!(model.endpoints.iter().all(|endpoint| endpoint.ingress == cuteafd_core::SparkIngress::Fp8K32));
        }
        Ok(())
    }
}

pub(crate) fn spark_model(backend: cuteafd_loader::plan::spark_executions::SparkBackend,
    decode_rows: usize, unit_rows: usize, waves: usize) -> Result<cuteafd_transport::worker_selection::ModelAdmission> {
    use cuteafd_core::{SparkEndpoint, SparkIngress};
    // The installed coordinator manifest fixes this build's decode rows.
    let executions = cuteafd_loader::plan::spark_executions::admit_family("deepseek_v4", &backend.exports(),
        Some(u32::try_from(decode_rows)?), unit_rows as u64, waves as u64)?;
    let endpoints = vec![
        SparkEndpoint { ring_depth: None, capacity: u32::try_from(decode_rows)?, ingress: SparkIngress::Fp8K32, partial_reply_bytes: None },
        SparkEndpoint { ring_depth: None, capacity: u32::try_from(unit_rows.max(decode_rows))?, ingress: SparkIngress::Fp8K32, partial_reply_bytes: None },
    ];
    cuteafd_transport::worker_selection::ModelAdmission::new(endpoints,
        crate::shared::placement::spark_rings(64 << 20)?, executions)
}

/// The retained prefix config serve-dsv4 builds for exactly the admitted quota.
/// Admission already charged it with every pinned table and host owner.
pub(crate) fn host_config(prefix: Option<&crate::shared::prefix::PrefixArgs>,
    layout: cuteafd_engine::prefix::FamilyLayout, max_context: usize, sequences: usize,
    admitted: cuteafd_loader::placement::planner::serving::HostPrefixOwners)
    -> Result<Option<cuteafd_hostcache::config::Config>> {
    let (Some(prefix), true) = (prefix, admitted.quota_bytes > 0) else { return Ok(None) };
    let mut exact = prefix.clone();
    exact.host_cache_bytes = crate::shared::prefix::budget::HostBudget::Bytes(admitted.quota_bytes);
    let config = exact.resolve_host_config(layout, max_context, sequences)?;
    ensure!(config.as_ref().is_some_and(|c| c.bytes == admitted.quota_bytes
        && c.chunk_bytes == admitted.chunk_bytes),
        "V4 host prefix config differs from the admitted quota {admitted:?}");
    Ok(config)
}

/// Admit one placement from these inputs: one solve over every device and
/// host owner, before any allocation.
pub(crate) fn admit(inputs: &v4::V4Inputs)
    -> Result<(cuteafd_loader::placement::Placement, cuteafd_loader::placement::planner::serving::HostPrefixOwners)> {
    let prepared = v4::prepare(inputs).map_err(anyhow::Error::msg)?;
    let prefix = prepared.prefix().owners;
    for note in prepared.prefix().notes(inputs.max_context, inputs.sequences) { tracing::info!("{note}"); }
    let request = v4::production_request(inputs);
    prepared.admit(request).map(|admitted| (admitted.placement, prefix)).map_err(anyhow::Error::msg)
}

/// The family facts serve-dsv4 admits: the same inputs `plan --layout` builds.
pub(crate) fn inputs(args: &EngineArgs, inputs: &Inputs<'_>) -> Result<v4::V4Inputs> {
    let gpus = inputs.gpus.len();
    let weights = cuteafd_loader::families::deepseek_v4::resident::WeightInventory::from_headers(
        inputs.cfg, gpus, args.dspark, inputs.catalog.tensors().iter().map(|t| &t.metadata))?;
    let mut baselines = inputs.gpus.clone();
    if let Some(modes) = inputs.loaded_modes {
        let loaded = weights.loaded_bytes(modes, gpus)?;
        for ((total, baseline), bytes) in baselines.iter_mut().zip(loaded) {
            *baseline = match *baseline {
                Baseline::Measured { free_bytes } => {
                    let free_bytes = free_bytes.checked_add(bytes).context("V4 measured weight rebase overflow")?;
                    ensure!(free_bytes <= *total, "V4 measured sample and resident weights exceed total memory");
                    Baseline::Measured { free_bytes }
                }
                Baseline::Planned { .. } => anyhow::bail!("V4 loaded modes require a measured sample"),
            };
        }
    }
    // The loader's legacy config requests one nextn stage, but a dSpark
    // checkpoint loads all three concrete stages. Describe those actual
    // cache owners rather than the caller's nextn configuration.
    let mut cache_cfg = inputs.cfg.clone();
    cache_cfg.n_mtp_layers = inputs.cache_stages;
    let shape = args.prefill_shape(inputs.manifest["capacities"]["prefill_rows"]
        .as_u64().context("V4 manifest prefill_rows")?, None)?;
    let (prefill, decode) = (shape.unit_rows, inputs.decode_rows as u64);
    ensure!(inputs.prefill_rows as u64 == prefill,
        "V4 runtime unit rows {} disagree with admitted shape {prefill}", inputs.prefill_rows);
    let context = v4::manifest_context(inputs.manifest, inputs.family, inputs.cfg.n_heads, decode);
    let attention = cuteafd_loader::placement::attention::parse(&args.attention_placement).map_err(anyhow::Error::msg)?;
    let resolved = attention.unwrap_or(if gpus == 2 && context.is_some() {
        cuteafd_loader::placement::AttentionPlacement::Context
    } else { cuteafd_loader::placement::AttentionPlacement::Heads });
    let host_marks = v4::host_marks_enabled(gpus, resolved != cuteafd_loader::placement::AttentionPlacement::Heads,
        inputs.prefix.is_some_and(|p| p.host_cache_bytes.enabled()));
    let mark_slots = v4::mark_slots(&cache_cfg, gpus, prefill, inputs.cache_stages, args.max_sequences as u64,
        inputs.prefix.map_or(0, |p| p.prefix_cache_entries) as u64,
        (inputs.prefix.map_or(0, |p| p.prefix_cache_mark_mib) as u64) << 20, host_marks)?;
    let scratch = cuteafd_loader::serving_capacity::deepseek_v4_workspace_scratch(
        inputs.manifest, inputs.family, gpus == 2, prefill, decode)?;
    let workspace = cuteafd_loader::serving_capacity::deepseek_v4_workspace_geometry(&cache_cfg, prefill, decode,
        cuteafd_loader::serving_capacity::compiled_c128_width(inputs.manifest, inputs.family)? * 128, gpus, shape.waves, scratch)?;
    let onboard = args.onboard(inputs.catalog.routed_experts().routed_layers());
    let stages = if args.dspark && !args.skip_routed_experts { inputs.cache_stages } else { 0 };
    let routed = *inputs.catalog.routed_experts();
    let (mut experts, draft) = if args.skip_routed_experts { (Vec::new(), Vec::new()) }
        else { v4::expert_costs(inputs.catalog, stages)? };
    let expert_workspace = match inputs.expert_workspace {
        Some(bytes) => bytes,
        // TP2 backbone executors do not need the TP1/dSpark package.
        None if gpus == 2 && (stages == 0 || args.tp2_dspark_experts) => 0,
        None => {
            ensure!(stages == 0 && !args.fixed_onboard(),
                "V4 local expert kernels are missing: export the matching rtx_backbone package for explicit local layers/dSpark");
            tracing::warn!("V4 local expert kernels unavailable; routing all backbone layers to Sparks");
            experts.clear();
            0
        }
    };
    let (experts_half, tp2_workspace) = if gpus == 2 && !args.skip_routed_experts {
        match inputs.tp2_workspace {
            Some(workspace) => (v4::expert_half_costs(inputs.catalog)?, workspace),
            None => {
                ensure!(!args.tp2_dspark_experts && (onboard == Onboard::Auto || onboard.layers(routed.layers - routed.first_layer) == Some(0)), "V4 TP2 expert kernels are missing: export the matching rtx_tp2 package for explicit local layers");
                tracing::warn!("V4 TP2 expert kernels unavailable; routing all backbone layers to Sparks");
                (Vec::new(), [0; 2])
            }
        }
    } else { (Vec::new(), [0; 2]) };
    let spark_ranks = if args.skip_routed_experts { 0 } else { args.peers.split(',').filter(|p| !p.is_empty()).count() };
    let embedding_weights = if args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Gpu {
        inputs.catalog.tensor("embed.weight")?.metadata.byte_length
    } else { 0 };
    let mut weights = weights;
    weights.fixed.push(cuteafd_loader::placement::Demand::new(0,
        cuteafd_core::memory_layout::Category::Embedding, "loaded embedding weights", 0,
        cuteafd_core::memory_layout::Basis::Exact)
        .with_resident_weights(cuteafd_loader::placement::ResidentWeights(embedding_weights)));
    Ok(v4::V4Inputs {
        resident_weight_limit: Some(crate::shared::memory::budget::resident_weight_limit()), spark_reply_format: cuteafd_core::SparkReplyFormat::resolve(cuteafd_core::spark_reply::requested().prefill, false), automatic_context: args.max_context == 0, cfg: cache_cfg, weights, cache_stages: inputs.cache_stages,
        gpus: baselines, headroom_floor: 0, spark_ranks,
        sequences: args.max_sequences as u64, prefill_rows: prefill, prefill_waves: shape.waves, decode_rows: decode,
        max_context: inputs.max_context as u64, reserve_bytes: (args.reserve_gib as u64) << 30, mark_slots,
        workspace: Some(workspace), draft_half: if args.tp2_dspark_experts { v4::draft_half_costs(inputs.catalog, stages)? } else { Vec::new() },
        experts, draft, expert_workspace, first_routed: routed.first_layer,
        experts_half, tp2_workspace, exchange_f32: inputs.exchange_f32,
        requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
        onboard: if args.skip_routed_experts { Onboard::Auto } else { onboard },
        full_prefill_logits: 0, code_bytes: v4::code_bytes(inputs.cfg.dim, v4::code_experts(inputs.catalog), gpus),
        context, attention, extra: Vec::new(), physical_bytes: inputs.physical.clone(),
        host: Some(v4::V4Host { total: inputs.host_memory.0, available: inputs.host_memory.1,
            headroom: inputs.prefix.map_or(8 << 30, |p| p.host_cache_headroom_bytes).max(inputs.host_memory.0 / 10),
            cache_bytes: match inputs.prefix.map(|p| p.host_cache_bytes) {
                Some(crate::shared::prefix::budget::HostBudget::Auto) => None,
                Some(crate::shared::prefix::budget::HostBudget::Bytes(bytes)) => Some(bytes),
                None => Some(0),
            },
            prefix_entries: inputs.prefix.map_or(0, |p| p.prefix_cache_entries as u64),
            span_rows: shape.span_rows, full_prefill_logits: args.full_prefill_logits,
            embedding_bytes: if args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Host {
                inputs.catalog.tensor("embed.weight")?.metadata.byte_length } else { 0 },
            c128_width: cuteafd_loader::serving_capacity::compiled_c128_width(inputs.manifest, inputs.family)? }) })
}
