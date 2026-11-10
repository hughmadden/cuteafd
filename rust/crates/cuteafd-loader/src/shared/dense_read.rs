//! Loading-only row windows shared by native dense admission and the mount reader.
use anyhow::{ensure, Context, Result};

pub const STEP_BYTES: usize = 32 << 20;
pub const PINNED_FLOOR_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadBatch {
    pub rows: usize,
    pub packed_bytes: usize,
    pub cached_bytes: usize,
}

impl ReadBatch {
    pub fn new(rows: usize, row_bytes: usize, width: usize, limit: usize) -> Result<Self> {
        ensure!(rows > 0 && width > 0 && width <= row_bytes && limit > 0, "invalid dense read window");
        let rows = rows.min((limit / width).max(1));
        Ok(Self { rows,
            packed_bytes: rows.checked_mul(width).context("dense packed read overflow")?,
            cached_bytes: if width == row_bytes { 0 } else {
                rows.checked_mul(row_bytes).context("dense row cache overflow")?
            },
        })
    }

    pub fn pinned_bytes(self) -> usize { self.packed_bytes.max(PINNED_FLOOR_BYTES) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_columns_charge_full_rows_for_the_packed_batch() {
        let batch = ReadBatch::new(16384, 8192, 4096, STEP_BYTES).unwrap();
        assert_eq!(batch.rows, 8192);
        assert_eq!(batch.packed_bytes, STEP_BYTES);
        assert_eq!(batch.cached_bytes, 2 * STEP_BYTES);
        assert_eq!(ReadBatch::new(16384, 8192, 8192, STEP_BYTES).unwrap().cached_bytes, 0);
        assert!(ReadBatch::new(1, 1, 0, STEP_BYTES).is_err());
        assert!(ReadBatch::new(2, usize::MAX, 1, usize::MAX).is_err());
    }
}
