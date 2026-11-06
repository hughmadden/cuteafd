//! MiMo physical facts. Both planner and runtime declare the same owners and
//! retained representations; only the GPU baseline differs.
pub mod admission;

pub use admission::route_exchange_bytes;

use crate::families::mimo_v2::MimoV2Config;
use crate::placement::{PlacementError, ResidentWeights};
use cuteafd_core::memory_layout::Category;

/// Item category for one profile reservation or pool owner.
pub fn reservation_category(name: &str) -> Category {
    if name.contains("embed_tokens") { Category::Embedding }
    else if name == "draft.fp8_scratch" { Category::Workspace }
    else if name.starts_with("draft.") { Category::Drafter }
    else if name.starts_with("workspace.") || name.starts_with("decode.") || name.starts_with("prefill") || name.starts_with("sampling.")
        || name == "transport.spark_intake_planes" || name == "state.prefill_kv_wide" { Category::Workspace }
    else if name.starts_with("transport.") { Category::Transport }
    else if name.starts_with("prefix.") { Category::Prefix }
    else if name.starts_with("state.") || name.starts_with("context.") { Category::Kv }
    else if name == "experts.local_scratch" { Category::Workspace }
    else if name.starts_with("experts.") { Category::Experts }
    else if name.starts_with("runtime.") || name.starts_with("graphs.") { Category::Runtime }
    else { Category::Weights }
}

/// Item group: physical families that keep their own ledger row.
pub fn reservation_group(name: &str) -> &str {
    match name {
        "state.active_rings" => "state",
        "prefix.exact_mark_arena" => "marks",
        "prefix.dflash_context_marks" => "DFlash context marks",
        "prefix.dflash_valid_floor_transfer" => "DFlash valid-floor transfer",
        _ => name,
    }
}

/// Already-loaded tower owners follow physical CUDA ordinals, not rank numbers.
/// Their physical storage is in the baseline; only checkpoint weights are added.
pub fn add_local_media_weights(weights: &mut [ResidentWeights], devices: &[i32],
    media: Option<(i32, ResidentWeights)>, limit: ResidentWeights) -> Result<(), PlacementError> {
    if weights.len() != devices.len() {
        return Err(PlacementError::Inventory("MiMo media owner/rank geometry"));
    }
    if let Some((device, media)) = media {
        if let Some(rank) = devices.iter().position(|&ordinal| ordinal == device) {
            weights[rank] = weights[rank].checked_add(media)?;
        } else {
            let gpu = u8::try_from(device)
                .map_err(|_| PlacementError::Inventory("MiMo media CUDA ordinal"))?;
            media.check(gpu, limit)?;
        }
    }
    Ok(())
}

/// Exact native weight/scale bytes, shared with the allocation path.
pub fn expert_layer_bytes(tensors: &crate::formats::fp8_experts::Fp8ExpertTensors,
    tp: usize, rank: usize, slicing: crate::formats::fp8_experts::Slicing) -> anyhow::Result<u64> {
    use crate::formats::fp8_experts::Fp8Projection;
    Fp8Projection::ALL.iter().try_fold(0u64, |total, &projection| {
        let (weights, _) = tensors.slice_bytes_with(projection, tp, rank, slicing)?;
        let scales = tensors.scale_region_bytes_with(projection, tp, rank, slicing)?;
        total.checked_add((tensors.shape().experts as u64).checked_mul(weights as u64)
            .ok_or_else(|| anyhow::anyhow!("MiMo expert weight size overflow"))?)
            .and_then(|bytes| bytes.checked_add(scales as u64))
            .ok_or_else(|| anyhow::anyhow!("MiMo expert layer size overflow"))
    })
}

/// Select the same BF16-input package and whole-block slices as the runtime,
/// using only package manifests (no CUDA or native-library initialization).
pub fn package_directory(lib: &std::path::Path, cfg: &MimoV2Config,
    format: crate::formats::fp8_experts::ExpertFormat, tp: usize) -> Result<std::path::PathBuf, anyhow::Error> {
    let wire = spark_package_directory(lib, cfg, format, tp)?;
    let package = wire.parent().ok_or_else(|| anyhow::anyhow!("MiMo package parent"))?;
    let sibling = package.with_file_name(format!("{}-bf16", package.file_name()
        .and_then(|name| name.to_str()).ok_or_else(|| anyhow::anyhow!("MiMo package name"))?));
    Ok(if sibling.join(format!("tp{tp}")).is_dir() { sibling.join(format!("tp{tp}")) } else { wire })
}

/// Spark workers select their wire package before the optional BF16 sibling.
pub fn spark_package_directory(lib: &std::path::Path, cfg: &MimoV2Config,
    format: crate::formats::fp8_experts::ExpertFormat, tp: usize) -> Result<std::path::PathBuf, anyhow::Error> {
    use crate::formats::fp8_experts::ExpertFormat;
    let family = match (cfg.program_family()?, format) {
        ("mimo", ExpertFormat::Mxfp4) => "mimof",
        (family, _) => family,
    };
    let package = lib.join("fp8").join(format.package_name(family));
    Ok(package.join(format!("tp{tp}")))
}

/// A missing implicit manifest makes local onboarding ineligible. A malformed
/// or explicitly selected manifest is an unsupported package, never a fallback.
pub fn package_manifest(directory: &std::path::Path, explicit: Option<&std::path::Path>)
    -> anyhow::Result<Option<serde_json::Value>> {
    let path = explicit.map(std::path::Path::to_path_buf).unwrap_or_else(||
        directory.parent().unwrap_or(directory).join("manifest.json"));
    match std::fs::read(&path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).map_err(|error|
            anyhow::anyhow!("MiMo expert manifest {}: {error}", path.display()))?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && explicit.is_none() => Ok(None),
        Err(error) => Err(anyhow::anyhow!("MiMo expert manifest {}: {error}", path.display())),
    }
}

pub fn exact_package(directory: &std::path::Path,
    tensors: &crate::formats::fp8_experts::Fp8ExpertTensors, tp: usize, rank: usize)
    -> (std::path::PathBuf, crate::formats::fp8_experts::Slicing) {
    crate::formats::fp8_experts::exact_layout_directory(directory, tensors, tp, rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::mimo_v2::{MimoAttention, MimoKvFormats};
    use crate::placement::planner::Pin;
    use crate::placement::{Bytes2, ExpertCost, Onboard};
    use crate::serving_capacity::mimo_cache_geometry;
    use cuteafd_core::serving_capacity::{
        CapacityProfile, ContextLimits, DeviceCosts, MemoryReservation,
    };

    #[test]
    fn media_resources_follow_physical_ordinals_and_check_independent_owners() {
        for devices in [[7, 2], [2, 7]] {
            for rank in 0..2 {
                let mut weights = [ResidentWeights(11), ResidentWeights(22)];
                add_local_media_weights(&mut weights, &devices,
                    Some((devices[rank], ResidentWeights(123))), ResidentWeights(256)).unwrap();
                assert_eq!(weights[rank].0, [11, 22][rank] + 123);
                assert_eq!(weights[1 - rank].0, [11, 22][1 - rank]);
            }
            let mut weights = [ResidentWeights(11), ResidentWeights(22)];
            add_local_media_weights(&mut weights, &devices, None, ResidentWeights(256)).unwrap();
            add_local_media_weights(&mut weights, &devices, Some((9, ResidentWeights(256))), ResidentWeights(256)).unwrap();
            assert_eq!(weights, [ResidentWeights(11), ResidentWeights(22)]);
            assert_eq!(add_local_media_weights(&mut weights, &devices,
                Some((9, ResidentWeights(257))), ResidentWeights(256)),
                Err(PlacementError::ResidentWeightLimit { gpu: 9, required: 257, limit: 256 }));
        }
        assert!(add_local_media_weights(&mut [ResidentWeights(u64::MAX)], &[7],
            Some((7, ResidentWeights(1))), ResidentWeights(u64::MAX)).is_err());
        assert!(add_local_media_weights(&mut [], &[7], None, ResidentWeights(1)).is_err());
    }

    #[test]
    fn route_exchange_matches_native_slot_alignment_and_rejects_overflow() {
        for rows in [1, 3, 64, 4096] {
            for lanes in [1, 2, 3] {
                let payload = (rows * 8 * 4u64).div_ceil(16) * 32;
                assert_eq!(route_exchange_bytes(rows, 8, lanes).unwrap(),
                    lanes * 2 * payload.max(256) + 2 * 64 * 8 * 8 + ((lanes * 2 + 3) * 16).max(256) + 256);
            }
        }
        for (rows, topk, lanes) in [(u64::MAX, 8, 3), (64, u64::MAX, 3), (64, 8, u64::MAX)] {
            assert!(route_exchange_bytes(rows, topk, lanes).is_err());
        }
        assert_eq!(cuteafd_core::memory_layout::Category::of_scope("peer-split"), Category::Transport);
    }

    #[test]
    fn implicit_missing_explicit_and_malformed_package_contract() {
        let dir = tempfile::tempdir().unwrap();
        let layout = dir.path().join("tp2");
        assert!(package_manifest(&layout, None).unwrap().is_none());
        assert!(package_manifest(&layout, Some(&dir.path().join("missing.json"))).is_err());
        let manifest = dir.path().join("manifest.json");
        std::fs::write(&manifest, "not json").unwrap();
        assert!(package_manifest(&layout, None).is_err());
        std::fs::write(&manifest, "{}").unwrap();
        assert!(package_manifest(&layout, None).unwrap().is_some());
    }

    /// Four backbone layers, one routed slot: enough to exercise the contract
    /// without a checkpoint, at MiMo's real pool quantum.
    fn fixture(ranks: usize, context: u64) -> (MimoV2Config, CapacityProfile, CapacityProfile) {
        let mut cfg = MimoV2Config::from_hf(&crate::plan::testing::mimo_pro_config()).unwrap();
        cfg.layers = 4;
        cfg.dense = vec![true, false, false, false];
        cfg.attention = vec![MimoAttention::Full; 4];
        let cache = mimo_cache_geometry(&cfg, 4, ranks, MimoKvFormats::default(), 0).unwrap();
        let profile = CapacityProfile {
            context: ContextLimits { checkpoint_max_tokens: 1 << 20, compiled_index_max_tokens: None },
            pool_unit_rows: cache.logical_unit_rows,
            host_prefix_bytes: 0,
            devices: cache.ranks.iter().enumerate().map(|(rank, storage)| DeviceCosts {
                device: rank as u32,
                pool_unit_bytes: storage.persistent_unit_bytes + storage.pool_metadata_unit_bytes + 256,
                reservations: vec![MemoryReservation { name: format!("state.rank{rank}"),
                    bytes: 64 << 20 }],
            }).collect(),
        };
        let _ = context;
        (cfg, profile.clone(), profile)
    }

    fn input<'a>(cfg: &'a MimoV2Config, spark: &'a CapacityProfile, no_spark: &'a CapacityProfile,
        ranks: usize, context: u64) -> admission::ProductionInput<'a> {
        let cost = ExpertCost { whole: Bytes2 { resident: 1 << 30, staging: 0 },
            half: [Bytes2 { resident: 512 << 20, staging: 0 }; 2], tp2: ranks == 2,
            whole_supported: true, spark_ok: true };
        admission::ProductionInput {
            cfg, layers: 4, kv: MimoKvFormats::default(), mtp_stages: 0, context,
            maximum_pool_tokens: admission::maximum_pool_tokens(&[96 << 30; 2][..ranks], context,
                None, spark.pool_unit_rows, false),
            prefill: crate::placement::planner::PrefillShape { rows: 4096, waves: 3 },
            lanes: 3, embedding: crate::placement::planner::EmbeddingHome::Device(0),
            gpus: ranks, spark_ranks: 6,
            experts: vec![None, Some(cost), Some(cost), Some(cost)],
            expert_workspace: if ranks == 1 { 1 << 20 } else { 0 },
            tp2_workspace: if ranks == 2 { [1 << 20; 2] } else { [0; 2] },
            spark: spark.clone(), no_spark: no_spark.clone(),
            resident_weights: vec![ResidentWeights(11 << 30); ranks],
            operation: crate::placement::planner::Operation::Serving,
            budgets: admission::production_budgets(&vec![(96 << 30, 96 << 30); ranks],
                &[0; 2][..ranks], vec![crate::placement::planner::Baseline::Forecast {
                    occupied: 0, source: "unit test".into(), assumed: false }; ranks],
                (128 << 30, 128 << 30), 108 << 30, 6).unwrap(),
        }
    }

    #[test]
    fn zero_and_unset_pool_admit_identically() {
        let (cfg, spark, no_spark) = fixture(1, 256 << 10);
        let context = 256 << 10;
        let prepared = input(&cfg, &spark, &no_spark, 1, context).prepare().unwrap();
        let request = |pool| admission::production_request(context, pool, Onboard::Auto,
            prepared.recipe.prefill, crate::placement::planner::Operation::Serving,
            ResidentWeights(u64::MAX), Default::default());
        let unset = prepared.admit(request(None)).unwrap();
        let zero = prepared.admit(request(Some(0))).unwrap();
        assert!(matches!(unset.solved.request.pool, Pin::Auto));
        assert!(matches!(zero.solved.request.pool, Pin::Auto));
        assert_eq!(unset.placement.pool_tokens, zero.placement.pool_tokens);
        assert_eq!(unset.spark.rtx_layers, zero.spark.rtx_layers);
        assert_eq!(unset.placement.items, zero.placement.items);
    }

    #[test]
    fn unaffordable_card_returns_a_typed_refusal() {
        let (cfg, spark, no_spark) = fixture(1, 256 << 10);
        let context = 256 << 10;
        let mut prepared = input(&cfg, &spark, &no_spark, 1, context);
        prepared.budgets = admission::production_budgets(&[(1 << 30, 1 << 30)], &[0],
            vec![crate::placement::planner::Baseline::Forecast { occupied: 0,
                source: "unit test".into(), assumed: false }], (128 << 30, 128 << 30),
            108 << 30, 6).unwrap();
        let prepared = prepared.prepare().unwrap();
        let request = admission::production_request(context, None, Onboard::Auto,
            prepared.recipe.prefill, crate::placement::planner::Operation::Serving,
            ResidentWeights(u64::MAX), Default::default());
        let refused = prepared.admit(request).err().expect("a 1 GiB card cannot hold the context floor");
        assert!(matches!(refused, crate::placement::planner::Refusal::Capacity { .. }
            | crate::placement::planner::Refusal::FullContextShortfall { .. }
            | crate::placement::planner::Refusal::Unsupported), "unexpected refusal {refused}");
    }
}
