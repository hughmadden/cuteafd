//! Immutable MiMo DFlash storage and its device-loading contract.
//! The target owns the BF16 vocabulary head; single-copy drafters borrow it.
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoDraftRepresentation {
    Bf16Only,
    Fp8Only,
    /// BF16 FC and complete QKV matrices; FP8 o_proj, gate/up and down.
    /// Every matrix owns one representation and the target head is borrowed.
    Bf16Context,
    /// Unchanged private qualification baseline, not a single-copy mode.
    LegacyDual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MimoDraftCapacity {
    pub context_slots: usize,
    pub max_batch_sequences: usize,
    pub block_rows: usize,
}

impl MimoDraftCapacity {
    pub fn new(
        context_slots: usize,
        max_batch_sequences: usize,
        block: usize,
    ) -> Result<Self, MimoDraftStorageError> {
        if context_slots == 0
            || max_batch_sequences == 0
            || max_batch_sequences > context_slots
            || block < 2
        {
            return Err(MimoDraftStorageError::Unsupported(
                "draft batch must fit nonzero context slots and block >= 2",
            ));
        }
        // Native ring destinations use signed32 row indices (-1 means skip).
        if mul(context_slots as u64, 1024)? > i32::MAX as u64 + 1 {
            return Err(MimoDraftStorageError::Unsupported(
                "context slots exceed native ring row indices",
            ));
        }
        Ok(Self {
            context_slots,
            max_batch_sequences,
            block_rows: mul(max_batch_sequences as u64, block as u64)?
                .try_into()
                .map_err(|_| MimoDraftStorageError::Overflow)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MimoDraftGeometry {
    pub hidden: u64,
    pub intermediate: u64,
    pub layers: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub taps: u64,
    pub vocab: u64,
    pub sinks: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MimoDraftWeightLayout {
    /// Owned BF16 GEMM values, excluding the borrowed target head.
    pub bf16_values: u64,
    pub fp8_values: u64,
    pub fp8_scales: u64,
    /// Norms, sinks and trained mask row, always BF16.
    pub small_bf16: u64,
    /// Only the legacy baseline owns a packed copy of the target head.
    pub head_fp8_values: u64,
    pub head_fp8_scales: u64,
    /// One BF16 source matrix while packing FP8-only storage. Packing is
    /// drained before this allocation is released or the next matrix loads.
    pub max_load_staging: u64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MimoDraftStorageError {
    #[error("unsupported MiMo DFlash storage: {0}")]
    Unsupported(&'static str),
    #[error("MiMo DFlash storage byte count overflow")]
    Overflow,
}

fn mul(a: u64, b: u64) -> Result<u64, MimoDraftStorageError> {
    a.checked_mul(b).ok_or(MimoDraftStorageError::Overflow)
}
fn add(a: u64, b: u64) -> Result<u64, MimoDraftStorageError> {
    a.checked_add(b).ok_or(MimoDraftStorageError::Overflow)
}
fn sum(values: impl IntoIterator<Item = u64>) -> Result<u64, MimoDraftStorageError> {
    values.into_iter().try_fold(0, add)
}

impl MimoDraftWeightLayout {
    pub fn new(
        g: MimoDraftGeometry,
        mode: MimoDraftRepresentation,
    ) -> Result<Self, MimoDraftStorageError> {
        if [
            g.hidden,
            g.intermediate,
            g.layers,
            g.heads,
            g.kv_heads,
            g.head_dim,
            g.taps,
            g.vocab,
        ]
        .contains(&0)
            || g.heads % g.kv_heads != 0
        {
            return Err(MimoDraftStorageError::Unsupported(
                "invalid drafter geometry",
            ));
        }
        let attention = mul(g.heads, g.head_dim)?;
        let kv = mul(g.kv_heads, g.head_dim)?;
        let layer_shapes = [
            (add(attention, mul(2, kv)?)?, g.hidden),
            (g.hidden, attention),
            (mul(2, g.intermediate)?, g.hidden),
            (g.hidden, g.intermediate),
        ];
        let fc = (g.hidden, mul(g.taps, g.hidden)?);
        let uses_fp8 = mode != MimoDraftRepresentation::Bf16Only;
        let matrix =
            |(n, k): (u64, u64), packed: bool| -> Result<(u64, u64, u64), MimoDraftStorageError> {
                if packed && (n % 16 != 0 || k % 128 != 0) {
                    return Err(MimoDraftStorageError::Unsupported(
                        "packed FP8 matrices require N % 16 == 0 and K % 128 == 0",
                    ));
                }
                let values = mul(n, k)?;
                Ok((
                    mul(values, 2)?.max(256),
                    values.max(256),
                    mul(values / 128, 4)?.max(256),
                ))
            };
        let layer = layer_shapes
            .into_iter()
            .enumerate()
            .map(|(index, shape)| {
                matrix(
                    shape,
                    uses_fp8 && !(mode == MimoDraftRepresentation::Bf16Context && index == 0),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let fc_bytes = matrix(fc, uses_fp8 && mode != MimoDraftRepresentation::Bf16Context)?;
        let own = |index: usize| -> Result<u64, MimoDraftStorageError> {
            let get = |m: &(u64, u64, u64)| match index {
                0 => m.0,
                1 => m.1,
                _ => m.2,
            };
            add(get(&fc_bytes), mul(g.layers, sum(layer.iter().map(get))?)?)
        };
        let context_bf16 = add(fc_bytes.0, mul(g.layers, layer[0].0)?)?;
        let rest = |index: usize| -> Result<u64, MimoDraftStorageError> {
            let get = |m: &(u64, u64, u64)| if index == 1 { m.1 } else { m.2 };
            mul(g.layers, sum(layer[1..].iter().map(get))?)
        };
        let small_per_layer = sum([
            mul(2, mul(g.hidden, 2)?.max(256))?,
            mul(2, mul(g.head_dim, 2)?.max(256))?,
            if g.sinks {
                mul(g.heads, 2)?.max(256)
            } else {
                0
            },
        ])?;
        let small_bf16 = add(
            mul(g.layers, small_per_layer)?,
            mul(3, mul(g.hidden, 2)?.max(256))?,
        )?;
        let (head_fp8_values, head_fp8_scales) = if mode == MimoDraftRepresentation::LegacyDual {
            let (_, values, scales) = matrix((g.vocab, g.hidden), true)?;
            (values, scales)
        } else {
            (0, 0)
        };
        Ok(Self {
            bf16_values: match mode {
                MimoDraftRepresentation::Fp8Only => 0,
                MimoDraftRepresentation::Bf16Context => context_bf16,
                _ => own(0)?,
            },
            fp8_values: match mode {
                MimoDraftRepresentation::Bf16Only => 0,
                MimoDraftRepresentation::Bf16Context => rest(1)?,
                _ => own(1)?,
            },
            fp8_scales: match mode {
                MimoDraftRepresentation::Bf16Only => 0,
                MimoDraftRepresentation::Bf16Context => rest(2)?,
                _ => own(2)?,
            },
            small_bf16,
            head_fp8_values,
            head_fp8_scales,
            max_load_staging: match mode {
                MimoDraftRepresentation::Fp8Only => layer
                    .iter()
                    .map(|m| m.0)
                    .chain([fc_bytes.0])
                    .max()
                    .unwrap_or(0),
                MimoDraftRepresentation::Bf16Context => {
                    layer[1..].iter().map(|m| m.0).max().unwrap_or(0)
                }
                _ => 0,
            },
        })
    }

    pub fn resident_bytes(&self) -> Result<u64, MimoDraftStorageError> {
        sum([
            self.bf16_values,
            self.fp8_values,
            self.fp8_scales,
            self.small_bf16,
            self.head_fp8_values,
            self.head_fp8_scales,
        ])
    }

    pub fn loading_peak_bytes(&self) -> Result<u64, MimoDraftStorageError> {
        add(self.resident_bytes()?, self.max_load_staging)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pro() -> MimoDraftGeometry {
        MimoDraftGeometry {
            hidden: 6144,
            intermediate: 16384,
            layers: 5,
            heads: 128,
            kv_heads: 8,
            head_dim: 128,
            taps: 5,
            vocab: 152576,
            sinks: true,
        }
    }
    #[test]
    fn pro_single_copy_counts_and_one_matrix_loading_peak() {
        let legacy =
            MimoDraftWeightLayout::new(pro(), MimoDraftRepresentation::LegacyDual).unwrap();
        let fp8 = MimoDraftWeightLayout::new(pro(), MimoDraftRepresentation::Fp8Only).unwrap();
        let bf16 = MimoDraftWeightLayout::new(pro(), MimoDraftRepresentation::Bf16Only).unwrap();
        assert_eq!(bf16.bf16_values, 5_536_481_280);
        assert_eq!(fp8.fp8_values, 2_768_240_640);
        assert_eq!(fp8.fp8_scales, 86_507_520);
        assert_eq!(fp8.max_load_staging, 402_653_184);
        assert_eq!(
            fp8.bf16_values + fp8.head_fp8_values + fp8.head_fp8_scales,
            0
        );
        assert_eq!(
            bf16.fp8_values + bf16.fp8_scales + bf16.head_fp8_values + bf16.max_load_staging,
            0
        );
        assert_eq!(
            legacy.resident_bytes().unwrap() - fp8.resident_bytes().unwrap(),
            6_503_202_816
        );
        assert_eq!(
            fp8.loading_peak_bytes().unwrap(),
            fp8.resident_bytes().unwrap() + 402_653_184
        );
    }
    #[test]
    fn twenty_context_slots_and_sixteen_block8_members_are_independent() {
        assert_eq!(
            MimoDraftCapacity::new(20, 16, 8).unwrap(),
            MimoDraftCapacity {
                context_slots: 20,
                max_batch_sequences: 16,
                block_rows: 128
            }
        );
        assert!(MimoDraftCapacity::new(16, 20, 8).is_err());
        assert!(MimoDraftCapacity::new(20, 0, 8).is_err());
        assert!(MimoDraftCapacity::new(20, 16, 1).is_err());
        assert!(MimoDraftCapacity::new(2_097_153, 16, 8).is_err());
    }
    #[test]
    fn bf16_context_keeps_one_copy_of_each_complete_matrix() {
        let layout =
            MimoDraftWeightLayout::new(pro(), MimoDraftRepresentation::Bf16Context).unwrap();
        let legacy =
            MimoDraftWeightLayout::new(pro(), MimoDraftRepresentation::LegacyDual).unwrap();
        assert_eq!(layout.bf16_values, 1_509_949_440);
        assert_eq!(layout.fp8_values, 2_013_265_920);
        assert_eq!(layout.fp8_scales, 62_914_560);
        assert_eq!(layout.head_fp8_values + layout.head_fp8_scales, 0);
        assert_eq!(layout.max_load_staging, 402_653_184);
        assert_eq!(
            legacy.resident_bytes().unwrap() - layout.resident_bytes().unwrap(),
            5_771_821_056
        );
        let mut geometry = pro();
        geometry.intermediate = 128;
        geometry.taps = 32;
        // A persistent BF16 FC must not be counted again as temporary FP8
        // packing storage. The only large packed matrix is o_proj here.
        let layout =
            MimoDraftWeightLayout::new(geometry, MimoDraftRepresentation::Bf16Context).unwrap();
        assert_eq!(layout.max_load_staging, 201_326_592);
    }
    #[test]
    fn reject_overflow_and_fp8_alignment_but_preserve_bf16_geometry() {
        let mut g = pro();
        g.hidden = 6145;
        assert!(MimoDraftWeightLayout::new(g, MimoDraftRepresentation::Fp8Only).is_err());
        assert!(MimoDraftWeightLayout::new(g, MimoDraftRepresentation::Bf16Only).is_ok());
        g.hidden = u64::MAX;
        assert_eq!(
            MimoDraftWeightLayout::new(g, MimoDraftRepresentation::Bf16Only),
            Err(MimoDraftStorageError::Overflow)
        );
    }
}
