//! A DFlash2 drafter's coordinator memory on the shared GQA drafter
//! (`cuteafd-daemon` `shared::speculation::dflash`), as MiniMax's admission recipe charges it:
//! the planner and serve-minimax both size it here, from the drafter's own `config.json`.
use crate::families::glm5::draft_representation::{draft_geometry, draft_resident_bytes_with_mode, draft_window,
    GlmDraftCapacity, GlmDraftRepresentation, GlmDraftRuntimeLayout};
use crate::families::qwen4::Qwen4KvCache;

/// The shared drafter's FP8 GEMM mode MiniMax loads (W8A16, mode 0).
const FP8_ROWS_W8A16: u8 = 0;
/// Draft steps take at most this many sequences (the shared drafter's step arenas).
pub const MAX_DRAFT_SEQUENCES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DflashMemory {
    /// Context rings (one per drafting request) and sequences one draft step takes.
    pub slots: usize,
    pub sequences: usize,
    /// The drafter's own weights (the target's embedding and head are borrowed).
    pub weights_bytes: u64,
    /// K/V rings of every layer and slot.
    pub ring_bytes: u64,
    /// Everything resident: weights, rings, taps, context-update and step buffers, scratch.
    pub bytes: u64,
    /// The largest BF16 source matrix held while packing one FP8 weight (0 for BF16).
    pub staging: u64,
}

impl DflashMemory {
    /// The drafter of `config` with `slots` rings in `representation` and `kv_cache` rings, on a
    /// GPU of `sms` SMs.
    pub fn new(config: &serde_json::Value, slots: usize, sms: u64, representation: GlmDraftRepresentation,
        kv_cache: Qwen4KvCache) -> anyhow::Result<Self> {
        use anyhow::Context;
        let sequences = slots.min(MAX_DRAFT_SEQUENCES);
        let (geometry, block) = draft_geometry(config)?;
        let ring = draft_window(config)?;
        let capacity = GlmDraftCapacity::with_ring(slots, sequences, usize::try_from(block)?, ring)?;
        let layout = GlmDraftRuntimeLayout::new(geometry, representation, capacity, usize::try_from(ring)?)?;
        let (resident, scratch) = draft_resident_bytes_with_mode(config, slots, sequences, sms, representation,
            FP8_ROWS_W8A16, kv_cache)?;
        let record = kv_cache.plane_record_bytes(usize::try_from(geometry.kv_heads)?, usize::try_from(geometry.head_dim)?);
        let ring_bytes = [2, geometry.layers, slots as u64, ring, record as u64].into_iter()
            .try_fold(1u64, |a, b| a.checked_mul(b)).context("DFlash2 ring bytes overflow")?;
        Ok(Self { slots, sequences, weights_bytes: layout.weights.resident_bytes()?, ring_bytes,
            bytes: resident.checked_add(scratch).context("DFlash2 drafter bytes overflow")?,
            staging: layout.weights.max_load_staging })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RadixArk/Minimax-M3.1-Flash-Dflash2's geometry (config.json fields the admission reads).
    fn config() -> serde_json::Value {
        serde_json::json!({"head_dim": 128, "hidden_size": 6144, "intermediate_size": 12288,
            "num_attention_heads": 64, "num_hidden_layers": 5, "num_key_value_heads": 4, "sliding_window": 4096,
            "vocab_size": 200064, "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
                "mask_token_id": 200058, "selector_rank": 256, "selector_top_k": 16,
                "target_layer_ids": [3, 17, 31, 45, 59]}})
    }

    #[test]
    fn minimax_dflash_weights_match_the_checkpoint_and_fp8_halves_the_gemms() -> anyhow::Result<()> {
        let bf16 = DflashMemory::new(&config(), 8, 170, GlmDraftRepresentation::Bf16Only, Qwen4KvCache::Bf16)?;
        // 2,054,677,760 BF16 parameters: the checkpoint's own 4.11 GB, no head or embedding copy.
        assert_eq!(bf16.weights_bytes, 2_054_677_760 * 2);
        assert_eq!(bf16.staging, 0);
        let fp8 = DflashMemory::new(&config(), 8, 170, GlmDraftRepresentation::Fp8Only, Qwen4KvCache::Bf16)?;
        // E4M3 GEMM values, FP32 scales per row and 128-wide block, BF16 norms, kernels and codebooks.
        let gemm = 1_951_924_224u64;
        assert_eq!(fp8.weights_bytes, gemm + gemm / 128 * 4 + 641_536 + 204_865_536);
        // Packing holds one BF16 source matrix, the largest: fc [6144, 5 x 6144].
        assert_eq!(fp8.staging, 6144 * 30720 * 2);
        // Four KV heads of 128 over a 4096-position ring for each of five layers' K and V.
        assert_eq!(bf16.ring_bytes, 2 * 5 * 8 * 4096 * 4 * 128 * 2);
        let fp8_rings = DflashMemory::new(&config(), 8, 170, GlmDraftRepresentation::Bf16Only, Qwen4KvCache::Fp8)?;
        assert_eq!(fp8_rings.ring_bytes, 2 * 5 * 8 * 4096 * 528);
        assert_eq!(bf16.bytes - fp8_rings.bytes, bf16.ring_bytes - fp8_rings.ring_bytes);
        assert!(bf16.bytes > bf16.weights_bytes + bf16.ring_bytes);
        assert_eq!((bf16.slots, bf16.sequences), (8, 8));
        assert_eq!(DflashMemory::new(&config(), 64, 170, GlmDraftRepresentation::Bf16Only, Qwen4KvCache::Fp8)?.sequences, 32);
        Ok(())
    }

    /// The 4096-position ring and tap rows scale the owned arenas; GLM's 2048 keeps its own.
    #[test]
    fn the_ring_and_tap_rows_follow_the_checkpoint_window() -> anyhow::Result<()> {
        let at = |window: u64| -> anyhow::Result<DflashMemory> {
            let mut config = config();
            config["sliding_window"] = window.into();
            DflashMemory::new(&config, 4, 188, GlmDraftRepresentation::Bf16Only, Qwen4KvCache::Bf16)
        };
        let (wide, narrow) = (at(4096)?, at(2048)?);
        assert_eq!(wide.ring_bytes, 2 * narrow.ring_bytes);
        // Taps [rows, 5 x 6144] BF16, two [rows, 6144] context buffers, K|V rows and their tables.
        let per_tap_row = 5 * 6144 * 2 + 6144 * 4 + 512 * 4 + 12;
        assert!(wide.bytes - narrow.bytes >= (wide.ring_bytes - narrow.ring_bytes) + 2048 * per_tap_row);
        // A config without a window keeps the GLM releases' 2048; a zero window has no ring.
        let mut config = config();
        config.as_object_mut().unwrap().remove("sliding_window");
        assert_eq!(DflashMemory::new(&config, 4, 188, GlmDraftRepresentation::Bf16Only, Qwen4KvCache::Bf16)?, narrow);
        config["sliding_window"] = 0.into();
        assert!(DflashMemory::new(&config, 4, 188, GlmDraftRepresentation::Bf16Only, Qwen4KvCache::Bf16).is_err());
        Ok(())
    }
}
