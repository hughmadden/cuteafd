//! Prefill wave geometry inside a link's fixed range. A link admits `slots`
//! return slots (control words, request staging arenas, Spark queue depth)
//! of `rows` rows each, in one range per rank of `slots * rows` rows,
//! registered once: slot `s` lives at row `s * rows`. The cut is fixed for
//! the link's lifetime (TJ 2026-10-28: fixed prefill depth; the per-call
//! re-cut of #762 was removed).

use anyhow::{ensure, Result};

/// A link's cut: return slots in flight and rows per slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Cut {
    pub waves: usize,
    pub rows: usize,
}

/// A link's fixed return-slot cut.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SlotGeometry { cut: Cut }

impl SlotGeometry {
    /// `waves` slots of `rows` rows.
    pub(crate) fn new(waves: usize, rows: usize) -> Result<Self> {
        ensure!((1..=4).contains(&waves) && rows > 0, "a link admits 1..=4 return slots of at least one row");
        Ok(Self { cut: Cut { waves, rows } })
    }
    pub(crate) fn cut(&self) -> Cut { self.cut }
    pub(crate) fn slots(&self) -> usize { self.cut.waves }
    /// Rows of each rank's range.
    pub(crate) fn reserve_rows(&self) -> usize { self.cut.waves * self.cut.rows }

    /// Byte offset and extent of `slot` in a rank's range of `row_bytes` rows.
    pub(crate) fn slice(&self, slot: usize, row_bytes: usize) -> (usize, usize) {
        assert!(slot < self.cut.waves, "slot {slot} outside the cut's {} waves", self.cut.waves);
        (slot * self.cut.rows * row_bytes, self.cut.rows * row_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_contiguous_and_fill_the_range() {
        for (waves, rows) in [(1, 4096), (2, 4096), (3, 2752), (4, 2048), (3, 256)] {
            let g = SlotGeometry::new(waves, rows).unwrap();
            let row_bytes = 6144 * 2;
            let mut end = 0;
            for slot in 0..waves {
                let (offset, bytes) = g.slice(slot, row_bytes);
                assert_eq!(offset, end, "slots are contiguous");
                assert_eq!(bytes, rows * row_bytes);
                end = offset + bytes;
            }
            assert_eq!(end, g.reserve_rows() * row_bytes);
        }
        assert!(SlotGeometry::new(5, 1024).is_err());
        assert!(SlotGeometry::new(0, 1024).is_err());
        assert!(SlotGeometry::new(2, 0).is_err());
    }
}
