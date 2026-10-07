//! Immutable GLM/GLM Flash DFlash storage and its device-loading contract.
//! The target owns its one vocabulary head (BF16, or GLM 5.3 Flash's FP8-only
//! head with --fp8-head); the drafter borrows it and never copies it.
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmDraftRepresentation {
    Bf16Only,
    Fp8Only,
}

impl GlmDraftRepresentation {
    /// Current DFlash readers support BF16 checkpoint weights. Quantization
    /// is an explicit convenience choice, independent of activation formats.
    pub fn from_fp8_option(fp8: Option<bool>) -> Self {
        // Drafter precision cannot change committed tokens; FP8 drafts measured faster
        // (GLM 5.3, 1 RTX + 4 Sparks: C1 code 41.5 -> 43.9, C4 63.9 -> 71.1 tok/s).
        if fp8 == Some(false) { Self::Bf16Only } else { Self::Fp8Only }
    }

    pub fn name(self) -> &'static str {
        match self { Self::Bf16Only => "BF16", Self::Fp8Only => "FP8" }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlmDraftCapacity {
    pub context_slots: usize,
    pub max_batch_sequences: usize,
    pub block_rows: usize,
}

impl GlmDraftCapacity {
    pub fn new(context_slots: usize, max_batch_sequences: usize, block: usize)
        -> Result<Self, GlmDraftStorageError> {
        if context_slots == 0 || max_batch_sequences == 0 || max_batch_sequences > context_slots || block < 2 {
            return Err(GlmDraftStorageError::Unsupported("draft batch must fit nonzero context slots and block >= 2"));
        }
        // Native ring destinations use signed32 row indices (-1 means skip).
        if mul(context_slots as u64, 2048)? > i32::MAX as u64 + 1 {
            return Err(GlmDraftStorageError::Unsupported("context slots exceed native ring row indices"));
        }
        let block_rows = max_batch_sequences.checked_mul(block).ok_or(GlmDraftStorageError::Overflow)?;
        Ok(Self { context_slots, max_batch_sequences, block_rows })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GlmDraftGeometry {
    pub hidden: u64,
    pub intermediate: u64,
    pub layers: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub taps: u64,
    pub vocab: u64,
    pub conv_group: u64,
    pub selector_rank: u64,
}

/// One owned GEMM `[n,k]`; the target vocabulary head is absent by design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlmDraftLinearShape {
    pub k: u64,
    pub n: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlmDraftWeightLayout {
    pub bf16_values: u64,
    pub fp8_values: u64,
    pub fp8_scales: u64,
    /// Norms and two-tap convolution base kernels, preserving BF16 values.
    pub bf16_auxiliary: u64,
    /// Distinct learned selector codebooks; they are not GEMM copies.
    pub bf16_codebooks: u64,
    /// One BF16 source matrix while packing. Drain before releasing it.
    pub max_load_staging: u64,
}

impl GlmDraftWeightLayout {
    pub fn resident_bytes(&self) -> Result<u64, GlmDraftStorageError> {
        sum([self.bf16_values, self.fp8_values, self.fp8_scales, self.bf16_auxiliary, self.bf16_codebooks])
    }

    pub fn loading_peak_bytes(&self) -> Result<u64, GlmDraftStorageError> {
        add(self.resident_bytes()?, self.max_load_staging)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmDraftFp8ScratchLayout {
    pub rows: usize,
    pub shapes: Vec<GlmDraftLinearShape>,
}

/// Admission and loading consume this same checked weight/scratch contract.
/// Ring, attention and activation arenas remain separately sized from capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmDraftRuntimeLayout {
    pub capacity: GlmDraftCapacity,
    pub weights: GlmDraftWeightLayout,
    pub fp8_scratch: Option<GlmDraftFp8ScratchLayout>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GlmDraftStorageError {
    #[error("unsupported GLM DFlash storage: {0}")]
    Unsupported(&'static str),
    #[error("GLM DFlash storage byte count overflow")]
    Overflow,
}

fn mul(a: u64, b: u64) -> Result<u64, GlmDraftStorageError> {
    a.checked_mul(b).ok_or(GlmDraftStorageError::Overflow)
}
fn add(a: u64, b: u64) -> Result<u64, GlmDraftStorageError> {
    a.checked_add(b).ok_or(GlmDraftStorageError::Overflow)
}
fn sum(values: impl IntoIterator<Item = u64>) -> Result<u64, GlmDraftStorageError> {
    values.into_iter().try_fold(0, add)
}

impl GlmDraftRuntimeLayout {
    pub fn new(g: GlmDraftGeometry, mode: GlmDraftRepresentation, capacity: GlmDraftCapacity,
        tap_rows: usize) -> Result<Self, GlmDraftStorageError> {
        if [g.hidden, g.intermediate, g.layers, g.heads, g.kv_heads, g.head_dim, g.taps, g.vocab,
            g.conv_group, g.selector_rank].contains(&0) || g.heads % g.kv_heads != 0
            || g.hidden % g.conv_group != 0 || tap_rows == 0 {
            return Err(GlmDraftStorageError::Unsupported("invalid drafter geometry"));
        }
        if g.head_dim != 128 || g.selector_rank != 256 {
            return Err(GlmDraftStorageError::Unsupported("kernels require head_dim 128 and selector rank 256"));
        }
        let h = g.hidden;
        let attention = mul(g.heads, g.head_dim)?;
        let kv = mul(g.kv_heads, g.head_dim)?;
        let conv = mul(4, h)? / g.conv_group;
        let fc = GlmDraftLinearShape { k: mul(g.taps, h)?, n: h };
        let projection = GlmDraftLinearShape { k: h, n: g.selector_rank };
        let layer_shapes = [
            GlmDraftLinearShape { k: h, n: conv },
            GlmDraftLinearShape { k: h, n: add(attention, mul(2, kv)?)? },
            GlmDraftLinearShape { k: attention, n: h },
            GlmDraftLinearShape { k: h, n: conv },
            GlmDraftLinearShape { k: h, n: mul(2, g.intermediate)? },
            GlmDraftLinearShape { k: g.intermediate, n: h },
        ];
        let values = sum([mul(fc.k, fc.n)?, mul(projection.k, projection.n)?,
            mul(g.layers, sum(layer_shapes.iter().map(|s| mul(s.k, s.n)).collect::<Result<Vec<_>, _>>()?)?)?])?;
        // Two layer norms, two [2,2,hidden] base kernels and Q/K head norms.
        let bf16_auxiliary = mul(2, sum([mul(2, h)?,
            mul(g.layers, add(mul(10, h)?, mul(2, g.head_dim)?)?)?])?)?;
        let bf16_codebooks = mul(4, mul(g.vocab, g.selector_rank)?)?;
        let uses_fp8 = mode == GlmDraftRepresentation::Fp8Only;
        let mut shapes = vec![fc, projection];
        shapes.extend(layer_shapes);
        // Context updates consume only the K|V row slice of packed QKV.
        shapes.push(GlmDraftLinearShape { k: h, n: mul(2, kv)? });
        if uses_fp8 && shapes.iter().any(|s| s.n % 16 != 0 || s.k % 128 != 0) {
            return Err(GlmDraftStorageError::Unsupported("FP8 GEMMs require N multiples of 16 and K multiples of 128"));
        }
        let max_load_staging = if uses_fp8 {
            mul(2, shapes.iter().map(|s| mul(s.k, s.n)).collect::<Result<Vec<_>, _>>()?
                .into_iter().max().unwrap_or(0))?
        } else { 0 };
        let weights = GlmDraftWeightLayout {
            bf16_values: if uses_fp8 { 0 } else { mul(2, values)? },
            fp8_values: if uses_fp8 { values } else { 0 },
            fp8_scales: if uses_fp8 { mul(values / 128, 4)? } else { 0 },
            bf16_auxiliary, bf16_codebooks, max_load_staging,
        };
        Ok(Self { capacity, weights, fp8_scratch: uses_fp8.then_some(GlmDraftFp8ScratchLayout {
            rows: tap_rows.max(capacity.block_rows), shapes,
        }) })
    }
}

/// Rows of one DFlash2 context ring (the drafter's sliding window, `dflash::RING`).
pub const GLM_DRAFT_RING: u64 = 2048;
/// Rows of the drafter's tap buffers (`dflash::TAP_ROWS`).
pub const GLM_DRAFT_TAP_ROWS: u64 = 2048;
/// The vocabulary head's cuBLAS workspace of a draft step (`VOCABULARY_HEAD_WORKSPACE`).
const GLM_DRAFT_HEAD_WORKSPACE: u64 = 4 << 20;
/// Smallest device allocation the drafter makes.
const FLOOR: u64 = 256;

/// A DFlash2 checkpoint's drafter geometry and block size from its `config.json` (the fields
/// `glm5::dflash::DflashConfig::read` takes); None for any other drafter (dSpark).
pub fn glm_dflash_geometry(config: &serde_json::Value) -> Option<(GlmDraftGeometry, u64)> {
    let int = |value: &serde_json::Value, key: &str| value[key].as_u64();
    let d = config.get("dflash_config")?;
    Some((GlmDraftGeometry {
        hidden: int(config, "hidden_size")?, intermediate: int(config, "intermediate_size")?,
        layers: int(config, "num_hidden_layers")?, heads: int(config, "num_attention_heads")?,
        kv_heads: int(config, "num_key_value_heads")?, head_dim: int(config, "head_dim")?,
        taps: d["target_layer_ids"].as_array()?.len() as u64, vocab: int(config, "vocab_size")?,
        conv_group: int(d, "conv_group_size")?, selector_rank: int(d, "selector_rank")?,
    }, int(d, "block_size")?))
}

/// How the FP8 drafter's GEMMs run (GLM 5.3 Flash's `--draft-linear`, the native
/// `cuteafd_fp8_linear` modes): what their scratch holds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GlmDraftLinear {
    /// W8A16 in passes of 64 rows.
    #[default]
    W8a16,
    /// The same bits in passes of 128 rows.
    Wide,
    /// E4M3 activations per row and 128-wide K block past one draft block, in passes of 128 rows.
    W8a8,
}

/// SMs the FP8 GEMMs' split-K plans for when the planner sizes their scratch without a device:
/// the RTX PRO 6000's 188, the most of the SM120 cards one build serves. More SMs split K
/// further, so the scratch is at most this (an RTX 5090's 170 SMs need 2 MiB less for DFlash2).
pub const GLM_DRAFT_SCRATCH_SMS: u64 = 188;

/// `cuteafd_fp8_linear_workspace` (native `fp8_gemv.cu` `layout_bytes`): one call's row scales,
/// B fragments, W8A8 activation scales and split-K partials, for a device of `sms` SMs.
pub fn fp8_linear_workspace_bytes(rows: u64, k: u64, n: u64, linear: GlmDraftLinear, sms: u64) -> u64 {
    if n < 16 || k < 128 {
        return 0;
    }
    let align = |bytes: u64| bytes.div_ceil(256) * 256;
    let chunk = if linear == GlmDraftLinear::W8a16 { 64 } else { 128 };
    let m = rows.min(chunk);
    let (tiles, kbs) = (n / 16, k / 128);
    let wanted = (sms * 16).div_ceil(tiles);
    let s = wanted.clamp(1, kbs.max(1));
    let kb_per_split = kbs.div_ceil(s);
    let splits = kbs.div_ceil(kb_per_split);
    align(2 * chunk * 4) + align(k * chunk * 2)
        + if linear == GlmDraftLinear::W8a8 { align(k / 128 * chunk * 4) } else { 0 }
        + if splits > 1 { align(splits * m * n * 4) } else { 0 }
}

/// Device bytes of a loaded DFlash2 drafter (`glm5::dflash::GlmDrafter::load`, its draft-step
/// workspace for the largest batch and its FP8 GEMM scratch), every allocation floored as the
/// drafter allocates it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmDraftDeviceBytes {
    /// Own weights in their one resident representation.
    pub weights: u64,
    /// Every layer's K and V rings for the context slots.
    pub rings: u64,
    /// The tap rows, their fused and normed rows, the context K | V rows, positions and ring slots.
    pub taps: u64,
    /// One draft step's buffers for the largest batch, with the native attention and top-k
    /// scratch and the vocabulary head's workspace (the ledger's `drafter/workspace`).
    pub workspace: u64,
    /// The FP8 GEMMs' scratch (the ledger's `workspace/fp8-linear`); 0 for a BF16 drafter.
    pub fp8_scratch: u64,
}

impl GlmDraftDeviceBytes {
    /// The ledger's `drafter` scope: weights, rings and tap buffers.
    pub fn resident(&self) -> u64 {
        self.weights + self.rings + self.taps
    }

    pub fn total(&self) -> u64 {
        self.resident() + self.workspace + self.fp8_scratch
    }
}

impl GlmDraftRuntimeLayout {
    /// Device bytes of this drafter with draft blocks of `block` rows (`block - 1` drafts), its
    /// FP8 GEMMs in `linear` mode on a device of `sms` SMs.
    pub fn device_bytes(&self, g: &GlmDraftGeometry, block: u64, linear: GlmDraftLinear, sms: u64)
        -> Result<GlmDraftDeviceBytes, GlmDraftStorageError> {
        let floor = |bytes: u64| bytes.max(FLOOR);
        let (h, kv) = (g.hidden, mul(g.kv_heads, g.head_dim)?);
        let slots = self.capacity.context_slots as u64;
        let rings = mul(mul(g.layers, 2)?, mul(mul(slots, GLM_DRAFT_RING)?, mul(kv, 2)?)?)?;
        let tap = GLM_DRAFT_TAP_ROWS;
        let taps = sum([mul(tap, mul(mul(g.taps, h)?, 2)?)?, mul(tap, h * 2)?, mul(tap, h * 2)?,
            mul(tap, mul(2, kv * 2)?)?, tap * 8, tap * 4].map(floor))?;
        let sequences = self.capacity.max_batch_sequences as u64;
        let rows = mul(sequences, block)?;
        let drafted = mul(sequences, block.saturating_sub(1))?;
        let attention = mul(g.heads, g.head_dim)?;
        // `cuteafd_glm_dflash_attention_workspace` over the ring and a block (128-key chunks of
        // 64 queries, 128 values and the max | sum pair, FP32), `cuteafd_glm_dflash_topk_workspace`
        // (64 chunks of 16 value | index pairs per drafted row).
        let chunks = (GLM_DRAFT_RING + block).div_ceil(128);
        let attention_workspace = mul(mul(mul(sequences, g.kv_heads)?, chunks)?, 64 * (128 + 2) * 4)?;
        let topk_workspace = mul(drafted, 64 * 16 * 8)?;
        let workspace = sum([rows * h * 2, rows * h * 2, rows * h * 2, rows * (4 * h / g.conv_group) * 2,
            rows * (attention + 2 * kv) * 2, rows * attention * 2, rows * kv * 2, rows * kv * 2,
            rows * attention * 2, rows * h * 2, rows * 2 * g.intermediate * 2, rows * g.intermediate * 2,
            mul(rows, g.vocab * 4)?, drafted * 16 * 4, drafted * 16 * 4, rows * g.selector_rank * 2,
            sequences * 4, rows * 4, drafted * 4, drafted * 16, rows * 8, 3 * sequences * 4,
            attention_workspace, topk_workspace, GLM_DRAFT_HEAD_WORKSPACE].map(floor))?;
        let fp8_scratch = self.fp8_scratch.as_ref().map_or(0, |scratch| {
            scratch.shapes.iter().map(|s| fp8_linear_workspace_bytes(scratch.rows as u64, s.k, s.n, linear, sms))
                .max().unwrap_or(0).max(FLOOR)
        });
        Ok(GlmDraftDeviceBytes { weights: self.weights.resident_bytes()?, rings, taps, workspace, fp8_scratch })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(flash: bool) -> GlmDraftGeometry {
        GlmDraftGeometry {
            hidden: if flash { 4096 } else { 6144 }, intermediate: 12288,
            layers: if flash { 5 } else { 6 }, heads: if flash { 32 } else { 64 },
            kv_heads: 8, head_dim: 128, taps: if flash { 5 } else { 6 }, vocab: 154880,
            conv_group: 16, selector_rank: 256,
        }
    }

    #[test]
    fn checkpoint_default_preserves_bf16_and_quantization_is_explicit() {
        assert_eq!(GlmDraftRepresentation::from_fp8_option(None), GlmDraftRepresentation::Fp8Only);
        assert_eq!(GlmDraftRepresentation::from_fp8_option(Some(false)), GlmDraftRepresentation::Bf16Only);
        assert_eq!(GlmDraftRepresentation::from_fp8_option(Some(true)), GlmDraftRepresentation::Fp8Only);
    }

    #[test]
    fn actual_glm_and_flash_weights_have_one_representation_and_no_head_copy() {
        let cap = GlmDraftCapacity::new(20, 16, 8).unwrap();
        for (flash, bf16, packed, scale, staging) in [
            (false, 4_759_486_464, 2_379_743_232, 74_366_976, 452_984_832),
            (true, 2_183_135_232, 1_091_567_616, 34_111_488, 201_326_592),
        ] {
            let a = GlmDraftRuntimeLayout::new(geometry(flash), GlmDraftRepresentation::Bf16Only, cap, 2048).unwrap();
            let b = GlmDraftRuntimeLayout::new(geometry(flash), GlmDraftRepresentation::Fp8Only, cap, 2048).unwrap();
            assert_eq!((a.weights.bf16_values, a.weights.fp8_values, a.weights.fp8_scales), (bf16, 0, 0));
            assert_eq!((b.weights.bf16_values, b.weights.fp8_values, b.weights.fp8_scales), (0, packed, scale));
            assert_eq!(a.weights.max_load_staging, 0);
            assert_eq!(b.weights.max_load_staging, staging);
            assert_eq!(a.weights.bf16_codebooks, 158_597_120);
            assert_eq!(a.weights.bf16_codebooks, b.weights.bf16_codebooks);
            assert!(a.fp8_scratch.is_none());
            assert!(b.fp8_scratch.as_ref().unwrap().shapes.iter().all(|s| s.n != 154880));
            assert_eq!(b.weights.loading_peak_bytes().unwrap(), b.weights.resident_bytes().unwrap() + staging);
        }
    }

    #[test]
    fn context_slots_and_draft_batch_are_independent_and_wide_updates_are_covered() {
        let cap = GlmDraftCapacity::new(20, 16, 8).unwrap();
        assert_eq!((cap.context_slots, cap.max_batch_sequences, cap.block_rows), (20, 16, 128));
        let l = GlmDraftRuntimeLayout::new(geometry(true), GlmDraftRepresentation::Fp8Only, cap, 2048).unwrap();
        let scratch = l.fp8_scratch.unwrap();
        assert_eq!(scratch.rows, 2048);
        assert!(scratch.shapes.contains(&GlmDraftLinearShape { k: 4096, n: 2048 }));
        let cap = GlmDraftCapacity::new(300, 300, 8).unwrap();
        let l = GlmDraftRuntimeLayout::new(geometry(true), GlmDraftRepresentation::Fp8Only, cap, 2048).unwrap();
        assert_eq!(l.fp8_scratch.unwrap().rows, 2400);
    }

    /// The GLM 5.3 Flash DFlash2 drafter's start-up ledger on an RTX 5090 (170 SMs), FP8 weights:
    /// the 5090 profile at 16 sequences (16 rings, `--draft-linear w8a8`) and an earlier build's
    /// defaults at 16 sequences (20 rings, w8a16), both with draft batches of 16.
    #[test]
    fn device_bytes_reproduce_the_measured_drafter_ledgers() {
        let g = geometry(true);
        let bytes = |slots: usize, linear: GlmDraftLinear| {
            GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Fp8Only, GlmDraftCapacity::new(slots, 16, 8).unwrap(),
                GLM_DRAFT_TAP_ROWS as usize).unwrap().device_bytes(&g, 8, linear, 170).unwrap()
        };
        let candidate = bytes(16, GlmDraftLinear::W8a8);
        assert_eq!(candidate.resident(), 2_081_647_104);
        assert_eq!(candidate.workspace, 174_999_744);
        assert_eq!(candidate.fp8_scratch, 28_394_496);
        let base = bytes(20, GlmDraftLinear::W8a16);
        assert_eq!(base.resident(), 2_249_419_264);
        assert_eq!(base.workspace, 174_999_744);
        assert_eq!(base.fp8_scratch, 14_156_288);
        // Each ring is 2,048 rows x 5 layers x K and V x 1,024 BF16 values.
        assert_eq!(base.rings - candidate.rings, 4 * 41_943_040);
        // `wide` has the same row passes as w8a8 without its activation scales; more SMs split K further.
        assert!(bytes(16, GlmDraftLinear::Wide).fp8_scratch < candidate.fp8_scratch);
        let pro = GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Fp8Only, GlmDraftCapacity::new(16, 16, 8)
            .unwrap(), 2048).unwrap().device_bytes(&g, 8, GlmDraftLinear::W8a8, GLM_DRAFT_SCRATCH_SMS).unwrap();
        assert_eq!(pro.fp8_scratch - candidate.fp8_scratch, 2 << 20);
        // A BF16 drafter has no FP8 scratch.
        let bf16 = GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Bf16Only, GlmDraftCapacity::new(16, 16, 8)
            .unwrap(), 2048).unwrap().device_bytes(&g, 8, GlmDraftLinear::W8a8, 170).unwrap();
        assert_eq!((bf16.fp8_scratch, bf16.rings, bf16.workspace), (0, candidate.rings, candidate.workspace));
    }

    #[test]
    fn the_dflash2_config_gives_the_drafter_geometry() {
        let config = serde_json::json!({"hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 5,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 154880,
            "sliding_window": 2048, "dflash_config": {"block_size": 8, "conv_group_size": 16, "selector_rank": 256,
                "target_layer_ids": [1, 10, 20, 30, 40]}});
        let (g, block) = glm_dflash_geometry(&config).unwrap();
        assert_eq!(block, 8);
        let flash = geometry(true);
        assert_eq!((g.hidden, g.intermediate, g.layers, g.heads, g.kv_heads, g.head_dim, g.taps, g.vocab,
            g.conv_group, g.selector_rank), (flash.hidden, flash.intermediate, flash.layers, flash.heads,
            flash.kv_heads, flash.head_dim, flash.taps, flash.vocab, flash.conv_group, flash.selector_rank));
        // A dSpark (or any other) drafter is not DFlash2.
        assert!(glm_dflash_geometry(&serde_json::json!({"hidden_size": 4096})).is_none());
    }

    #[test]
    fn invalid_capacity_and_overflow_are_named_results() {
        assert!(GlmDraftCapacity::new(0, 16, 8).is_err());
        assert!(GlmDraftCapacity::new(16, 20, 8).is_err());
        assert!(GlmDraftCapacity::new(20, 16, 1).is_err());
        assert!(GlmDraftCapacity::new(1_048_577, 16, 8).is_err());
        let mut g = geometry(false);
        g.layers = u64::MAX;
        assert_eq!(GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Bf16Only,
            GlmDraftCapacity::new(20, 16, 8).unwrap(), 2048).unwrap_err(), GlmDraftStorageError::Overflow);
    }

    #[test]
    fn unsupported_fp8_shapes_do_not_change_bf16_storage() {
        let mut g = geometry(true);
        g.hidden = 4112; // BF16 cuBLAS can read this; FP8 K must align to 128.
        let cap = GlmDraftCapacity::new(20, 16, 8).unwrap();
        assert!(GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Bf16Only, cap, 2048).is_ok());
        assert!(matches!(GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Fp8Only, cap, 2048),
            Err(GlmDraftStorageError::Unsupported(_))));
    }
}
