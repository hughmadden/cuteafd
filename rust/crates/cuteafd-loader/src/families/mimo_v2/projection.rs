//! One immutable resident representation of a target head/output projection.
use std::fmt;

/// BF16 or row-major E4M3 values. Scale banks are metadata, not another weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoProjectionRepresentation { Bf16, Fp8 }

/// Maximum BF16 source rows uploaded during FP8 packing. Each block is drained
/// before its source storage is reused; the final weight owns no BF16 fallback.
pub const MIMO_PROJECTION_STAGING_ROWS: u64 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MimoProjectionLayout {
    pub values: u64,
    pub scales: u64,
    pub max_load_staging: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MimoProjectionLayoutError;
impl fmt::Display for MimoProjectionLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MiMo single-copy projection requires positive rows/K, FP8 K divisible by128, one or two scale banks, and checked byte extents")
    }
}
impl std::error::Error for MimoProjectionLayoutError {}

impl MimoProjectionLayout {
    pub fn new(rows: u64, cols: u64, representation: MimoProjectionRepresentation,
        scale_banks: u64) -> Result<Self, MimoProjectionLayoutError> {
        use MimoProjectionRepresentation::*;
        let error = MimoProjectionLayoutError;
        if rows == 0 || cols == 0 || !(1..=2).contains(&scale_banks)
            || (representation == Fp8 && cols %128 != 0) { return Err(error); }
        let values = rows.checked_mul(cols).and_then(|n| n.checked_mul(if representation == Bf16 { 2 } else { 1 }))
            .ok_or(error)?;
        let scales = if representation == Fp8 {
            rows.checked_mul(cols /128).and_then(|n| n.checked_mul(4))
                .and_then(|n| n.checked_mul(scale_banks)).ok_or(error)?
        } else { 0 };
        let max_load_staging = if representation == Fp8 {
            rows.min(MIMO_PROJECTION_STAGING_ROWS).checked_mul(cols).and_then(|n| n.checked_mul(2)).ok_or(error)?
        } else { 0 };
        Ok(Self { values, scales, max_load_staging })
    }

    pub fn resident_bytes(self) -> Result<u64, MimoProjectionLayoutError> {
        self.values.checked_add(self.scales).ok_or(MimoProjectionLayoutError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pro_head_has_one_weight_and_bounded_source() {
        let fp8 = MimoProjectionLayout::new(152576, 6144, MimoProjectionRepresentation::Fp8, 1).unwrap();
        assert_eq!(fp8.resident_bytes().unwrap(), 966721536);
        assert_eq!(fp8.max_load_staging, 12582912);
        let bf16 = MimoProjectionLayout::new(152576, 6144, MimoProjectionRepresentation::Bf16, 1).unwrap();
        assert_eq!(bf16.resident_bytes().unwrap(), 1874853888);
        assert_eq!(bf16.max_load_staging, 0);
    }
    #[test]
    fn output_has_two_scale_orders_without_overlapping_values() {
        let full = MimoProjectionLayout::new(6144, 16384, MimoProjectionRepresentation::Fp8, 2).unwrap();
        let split = MimoProjectionLayout::new(6144, 8192, MimoProjectionRepresentation::Fp8, 2).unwrap();
        assert_eq!(full.resident_bytes().unwrap(), split.resident_bytes().unwrap() *2);
        assert_eq!(full.max_load_staging, 33554432);
        for (rows, cols, banks) in [(0,128,1),(16,127,1),(16,128,0),(16,128,3),(u64::MAX,128,1)] {
            assert!(MimoProjectionLayout::new(rows,cols,MimoProjectionRepresentation::Fp8,banks).is_err());
        }
    }
}
