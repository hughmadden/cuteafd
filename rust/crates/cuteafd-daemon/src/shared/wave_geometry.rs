//! Runtime prefill wave geometry inside a link's fixed reserve (#762 stages
//! 2–4). A link admits `slots` return slots (control words, request staging
//! arenas, Spark queue depth) and one reserve range per rank of `reserve_rows`
//! rows, registered once. Between prefills the executor may re-cut that
//! range into `waves <= slots` return slots of `rows <= max_rows` rows each:
//! slot `s` lives at row `s * rows`. Nothing is allocated, registered or
//! reconnected; a cut only changes offsets.
//!
//! A re-cut is legal only at rest (no wave outstanding, every landed wave
//! consumed). The new slots overlap the old ones arbitrarily, so the first
//! wave into each new slot also waits for every old slot's last consumed
//! count (already reached at rest; the fence keeps the invariant local).

use anyhow::{ensure, Result};

/// The current cut of a link's reserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Cut {
    pub waves: usize,
    pub rows: usize,
}

/// Admission bounds plus the current cut.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SlotGeometry {
    /// Admitted return slots (control words, arenas, Spark queue depth).
    slots: usize,
    /// Admitted rows of one unit (Spark owner capacity, staging arenas).
    max_rows: usize,
    /// Rows of each rank's reserve range.
    reserve_rows: usize,
    cut: Cut,
    /// Per new slot: whether its first wave since the last cut still owes
    /// the all-slot fence.
    fresh: Vec<bool>,
}

impl SlotGeometry {
    /// `waves` slots of `rows` inside a reserve admitting `slots` slots of up
    /// to `max_rows` rows in `reserve_rows` rows per rank.
    pub(crate) fn new(slots: usize, max_rows: usize, reserve_rows: usize, waves: usize, rows: usize)
        -> Result<Self> {
        ensure!((1..=4).contains(&slots) && max_rows > 0, "a link admits 1..=4 return slots of at least one row");
        ensure!(reserve_rows >= max_rows, "the reserve ({reserve_rows} rows) is narrower than one unit ({max_rows})");
        let mut geometry = Self { slots, max_rows, reserve_rows, cut: Cut { waves: 1, rows: 1 },
            fresh: vec![false; slots] };
        geometry.check(waves, rows)?;
        geometry.cut = Cut { waves, rows };
        Ok(geometry)
    }

    fn check(&self, waves: usize, rows: usize) -> Result<()> {
        ensure!((1..=self.slots).contains(&waves), "{waves} waves exceed the link's {} admitted slots", self.slots);
        ensure!((1..=self.max_rows).contains(&rows), "{rows}-row units exceed the admitted {}-row unit", self.max_rows);
        ensure!(waves * rows <= self.reserve_rows,
            "{waves} x {rows} rows exceed the link's {}-row reserve", self.reserve_rows);
        Ok(())
    }

    pub(crate) fn cut(&self) -> Cut { self.cut }
    pub(crate) fn slots(&self) -> usize { self.slots }
    pub(crate) fn max_rows(&self) -> usize { self.max_rows }
    pub(crate) fn reserve_rows(&self) -> usize { self.reserve_rows }

    /// A one-window call on a single-slot link (k = 1): its slot lands every
    /// call (any admitted rows) in one rotating range, so there is nothing to
    /// re-cut. Errors for more waves, or rows past the admitted unit.
    pub(crate) fn one_window(&self, waves: usize, rows: usize) -> Result<()> {
        ensure!(self.slots == 1 && waves == 1, "a single-slot link runs one window, not {waves}");
        ensure!((1..=self.max_rows).contains(&rows), "{rows}-row call exceeds the admitted {}-row unit", self.max_rows);
        Ok(())
    }

    /// Re-cut the reserve. The caller proves the link is at rest.
    pub(crate) fn set(&mut self, waves: usize, rows: usize) -> Result<bool> {
        self.check(waves, rows)?;
        let changed = self.cut != Cut { waves, rows };
        if changed {
            self.cut = Cut { waves, rows };
            self.fresh = vec![true; self.slots];
        }
        Ok(changed)
    }

    /// Byte offset and extent of `slot` in a rank's range of `row_bytes` rows.
    pub(crate) fn slice(&self, slot: usize, row_bytes: usize) -> (usize, usize) {
        assert!(slot < self.cut.waves, "slot {slot} outside the cut's {} waves", self.cut.waves);
        (slot * self.cut.rows * row_bytes, self.cut.rows * row_bytes)
    }

    /// The consumed fences the wave entering `slot` waits for: its own slot's
    /// last use, or, first after a cut, every admitted slot's last use.
    pub(crate) fn fences(&mut self, slot: usize, uses: &[u32]) -> Vec<(usize, u32)> {
        assert!(slot < self.cut.waves && uses.len() == self.slots);
        if std::mem::replace(&mut self.fresh[slot], false) {
            uses.iter().copied().enumerate().collect()
        } else {
            vec![(slot, uses[slot])]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shared reserve: 8192 rows, plus the 64-row page rounding three 2752-row
    /// units need (8256), admitting four slots of up to 4096 rows.
    fn reserve() -> SlotGeometry { SlotGeometry::new(4, 4096, 8256, 2, 4096).unwrap() }

    /// A k = 1 link (one slot of 4096 rows) accepts every prefill call's
    /// start, full or tail, as a no-op, and refuses more waves or wider units.
    #[test]
    fn a_one_window_link_accepts_every_call_start() {
        let one = SlotGeometry::new(1, 4096, 4096, 1, 4096).unwrap();
        for rows in [4096, 4096, 2112, 1] { one.one_window(1, rows).unwrap(); }
        assert!(one.one_window(2, 2048).is_err());
        assert!(one.one_window(1, 4097).is_err());
        assert!(reserve().one_window(1, 4096).is_err(), "multi-slot links re-cut instead");
    }

    #[test]
    fn every_admitted_cut_fits_the_reserve_without_overlap() {
        let mut g = reserve();
        for (waves, rows) in [(1, 4096), (2, 4096), (3, 2752), (4, 2048), (4, 512), (3, 256)] {
            g.set(waves, rows).unwrap();
            let row_bytes = 6144 * 2;
            let mut end = 0;
            for slot in 0..waves {
                let (offset, bytes) = g.slice(slot, row_bytes);
                assert_eq!(offset, end, "slots are contiguous");
                assert_eq!(bytes, rows * row_bytes);
                end = offset + bytes;
            }
            assert!(end <= g.reserve_rows() * row_bytes);
        }
    }

    #[test]
    fn cuts_beyond_admission_are_refused_and_keep_the_old_cut() {
        let mut g = reserve();
        for (waves, rows) in [(5, 1024), (0, 1024), (3, 4096), (2, 4160), (4, 2112), (3, 2816)] {
            assert!(g.set(waves, rows).is_err(), "{waves} x {rows}");
            assert_eq!(g.cut(), Cut { waves: 2, rows: 4096 });
        }
        // An 8192-row reserve cannot take 3 x 2752: the page rounding is admitted explicitly.
        let mut tight = SlotGeometry::new(4, 4096, 8192, 4, 2048).unwrap();
        assert!(tight.set(3, 2752).is_err());
        // Admission is checked at construction too.
        assert!(SlotGeometry::new(5, 4096, 8256, 2, 4096).is_err());
        assert!(SlotGeometry::new(4, 4096, 2048, 1, 2048).is_err());
        assert!(SlotGeometry::new(2, 2752, 8256, 3, 2752).is_err());
    }

    #[test]
    fn the_first_wave_into_each_slot_after_a_cut_fences_every_old_slot() {
        let mut g = reserve();
        let uses = [7, 7, 3, 0];
        assert_eq!(g.fences(1, &uses), [(1, 7)]);
        assert!(!g.set(2, 4096).unwrap(), "an unchanged cut owes nothing");
        assert_eq!(g.fences(0, &uses), [(0, 7)]);
        assert!(g.set(3, 2752).unwrap());
        for slot in 0..3 {
            assert_eq!(g.fences(slot, &uses), [(0, 7), (1, 7), (2, 3), (3, 0)], "slot {slot}");
            assert_eq!(g.fences(slot, &uses), [(slot, uses[slot])]);
        }
    }
}
