//! Routed-expert package layouts this build's packagers produce, and the
//! slice arithmetic the expert staging uses. The planner's placement check
//! reads these, so it accepts exactly the layouts an image can carry.
//!
//! - FP8 (`python/tools/aot/package_fp8_moe_aot.py`, b12x fp8_moe): Spark
//!   layouts tp4, tp2 and tp6 where they split the intermediate into whole
//!   128-row blocks (TP6 of any intermediate of at least six blocks, uneven
//!   blocks padded to the widest); a tp1 coordinator package.
//! - MXFP4 through the same programs (MiMo V2.6 Pro `mimop:fp8`): Spark tp6
//!   and tp2, a tp1 coordinator package.
//! - NVFP4 (ModelOpt) through the same programs (`glmf:nvfp4`, `qwen4:nvfp4`,
//!   `glm:nvfp4`): Spark tp4, tp2, tp3 and tp6 in whole 16-value blocks, a tp1
//!   coordinator package.
//! - EXL3 (`python/tools/aot/package_exl3_aot.py`): Spark worlds 4, 2, 3, and
//!   6 when the intermediate has at least six 128-row blocks.
//! - DeepSeek native experts (expertd-native MXFP4 / EXL3): 2, 3, 4 and 6.
//!
//! The expert transport (RoCE verbs, TCP) runs 2, 3, 4 or 6 Spark ranks.

/// Spark worlds the expert transport runs.
pub const TRANSPORT_WORLDS: [usize; 4] = [2, 3, 4, 6];

/// Default Spark layouts of an FP8 (E4M3, 128x128 scales) expert package.
pub fn fp8_spark_worlds(intermediate: usize) -> Vec<usize> {
    let blocks = intermediate / 128;
    [2usize, 4, 6]
        .into_iter()
        .filter(|&tp| intermediate % 128 == 0 && (intermediate % (128 * tp) == 0 || (tp == 6 && blocks >= 6)))
        .collect()
}

/// Default Spark layouts of an MXFP4 package on the fp8_moe programs.
pub fn mxfp4_spark_worlds() -> Vec<usize> {
    vec![2, 6]
}

/// Default Spark layouts of an NVFP4 package on the fp8_moe programs: the
/// transport worlds whose ranks each own a 16-value block.
pub fn nvfp4_spark_worlds(intermediate: usize) -> Vec<usize> {
    TRANSPORT_WORLDS.into_iter().filter(|&tp| intermediate % 16 == 0 && intermediate / 16 >= tp).collect()
}

/// Spark worlds of an EXL3 package (`package_exl3_aot.py` profiles).
pub fn exl3_spark_worlds(intermediate: usize) -> Vec<usize> {
    let blocks = intermediate / 128;
    if intermediate % 128 != 0 || blocks < 2 {
        return Vec::new();
    }
    let mut worlds = vec![2, 3, 4];
    if blocks >= 6 {
        worlds.push(6);
    }
    worlds.retain(|&tp| blocks >= tp);
    worlds
}

/// The stored intermediate slice of the widest rank of `tp`: whole `block`-row
/// blocks, as evenly as the blocks allow, padded to 128 rows. `None` when the
/// intermediate does not split into whole blocks over `tp` ranks.
/// `formats::fp8_experts::Fp8ExpertTensors::slice` stages exactly this width.
pub fn stored_slice(intermediate: usize, block: usize, tp: usize) -> Option<usize> {
    (tp > 0 && block > 0 && intermediate % block == 0 && intermediate / block >= tp)
        .then(|| ((intermediate / block).div_ceil(tp) * block).div_ceil(128) * 128)
}

/// The qualified 128-row layout stays the default. Mixed package contracts
/// cannot claim the MXFP4-only layout, and the padded-layout override wins.
pub(super) fn mxfp4_tails_enabled(package: &str) -> bool {
    package == "mimop:fp8 (MXFP4)"
        && std::env::var("CUTEAFD_MXFP4_TAILS").is_ok_and(|v| v == "1")
        && !std::env::var("CUTEAFD_FP8_EXACT_SLICES").is_ok_and(|v| v == "0")
}

/// Count stored down-scale rows, including only experts actually present in
/// the checkpoint. Sparse checkpoint fixtures must not inherit the model's
/// nominal layer and expert counts.
pub(super) fn mxfp4_down_scale_rows(checkpoint: &super::Checkpoint) -> u64 {
    checkpoint.tensors.iter().filter(|t| t.meta.name.contains(".mlp.experts.")
        && t.meta.name.ends_with(".down_proj.weight_scale") && t.meta.shape.len() == 2)
        .map(|t| t.meta.shape[0] as u64).sum()
}

/// Exact resident bytes of a 32-row MXFP4 slice. Weight and source-scale
/// bytes scale with the true width; only the down-scale row stride rounds
/// up to a u32. Integer arithmetic preserves byte-exact admission at limits.
pub(super) fn mxfp4_tail_bytes(routed: u64, intermediate: usize, tp: usize, rank: usize,
    down_scale_rows: u64) -> Option<(usize, u64)> {
    if tp == 0 || rank >= tp || intermediate % 32 != 0 || intermediate / 32 < tp {
        return None;
    }
    let blocks = intermediate / 32;
    let width = (blocks / tp + usize::from(rank < blocks % tp)) * 32;
    let padding = (width / 32).div_ceil(4) * 4 - width / 32;
    let bytes = routed as u128 * width as u128 / intermediate as u128
        + down_scale_rows as u128 * padding as u128;
    Some((width, u64::try_from(bytes).ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_layouts_follow_the_packagers() {
        // 2048 = 16 blocks: every FP8 layout; Qwen's 640 = 5 blocks: none.
        assert_eq!(fp8_spark_worlds(2048), [2, 4, 6]);
        assert!(fp8_spark_worlds(640).is_empty());
        assert_eq!(exl3_spark_worlds(2048), [2, 3, 4, 6]);
        assert_eq!(exl3_spark_worlds(640), [2, 3, 4]);
        assert_eq!(stored_slice(2048, 128, 6), Some(384));
        assert_eq!(stored_slice(2048, 32, 6), Some(384));
        assert_eq!(stored_slice(2304, 128, 4), Some(640));
        assert_eq!(stored_slice(640, 128, 6), None);
        assert_eq!(stored_slice(2048, 128, 0), None);
        assert_eq!(nvfp4_spark_worlds(640), [2, 3, 4, 6]);
        assert_eq!(stored_slice(640, 16, 6), Some(128));
        assert_eq!(stored_slice(2048, 16, 6), Some(384));
    }

    #[test]
    fn mxfp4_tails_count_exact_weight_and_padded_scale_bytes() {
        let (hidden, intermediate, experts, layers) = (6144u64, 2048usize, 384u64, 69u64);
        let down_rows = hidden * experts * layers;
        let routed = down_rows * intermediate as u64 * 3 * 17 / 32;
        let widths = [352, 352, 352, 352, 320, 320];
        let mut cluster = 0;
        for (rank, width) in widths.into_iter().enumerate() {
            let (actual_width, bytes) = mxfp4_tail_bytes(routed, intermediate, 6, rank, down_rows).unwrap();
            let projection_weights = down_rows * width as u64 / 2;
            let gate_scales = down_rows * width as u64 / 32;
            let down_scales = down_rows * 12;
            assert_eq!(actual_width, width);
            assert_eq!(bytes, 3 * projection_weights + 2 * gate_scales + down_scales);
            cluster += bytes;
        }
        assert_eq!(cluster, routed + down_rows * 8);
        assert_eq!(mxfp4_tail_bytes(routed, intermediate, 2, 0, down_rows).unwrap().1, routed / 2);
        for (i, tp, rank) in [(2047, 6, 0), (128, 6, 0), (2048, 0, 0), (2048, 6, 6)] {
            assert!(mxfp4_tail_bytes(routed, i, tp, rank, down_rows).is_none());
        }
    }
}
