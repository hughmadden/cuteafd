//! Per-engine runtime prefill depth (#762 stages 2–4). An engine whose
//! prefill runs on the shared executor holds one [`PrefillDepth`]: the
//! planner's admitted slots and unit extent, and the shared
//! [`WaveControl`]. At each decode→prefill transition the engine takes
//! [`PrefillDepth::next`] (the depth and unit of this prefill, inside the
//! admitted slots and reserve); after the prefill it reports the Sparks'
//! measured pace with [`PrefillDepth::finish`]. Storage never changes: every
//! depth's slots are views at `slot * unit` inside per-slot planes sized for
//! [`reserve_rows`], and the Spark prefill owner accepts any unit up to the
//! admitted one.

use cuteafd_loader::serving_capacity::wave_control::{reserve_rows, Decision, PrefillPace, WaveControl};
use std::cell::RefCell;

/// The shape of one prefill: `waves` units of at most `unit_rows` in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Depth { pub waves: usize, pub unit_rows: usize }

pub(crate) struct PrefillDepth {
    slots: usize,
    unit_cap: usize,
    control: RefCell<WaveControl>,
}

impl PrefillDepth {
    /// `initial` waves (the planner's estimate) of at most `unit_cap` rows,
    /// within `slots` admitted slots. `pinned` (an explicit wave count)
    /// keeps the initial depth.
    pub(crate) fn new(initial: usize, slots: usize, unit_cap: usize, pinned: bool) -> anyhow::Result<Self> {
        anyhow::ensure!((1..=4).contains(&slots) && (1..=slots).contains(&initial)
            && unit_cap >= 64 && unit_cap % 64 == 0 && unit_cap <= 4096,
            "prefill depth {initial} of {slots} slots x {unit_cap} rows outside the shared reserve");
        let control = WaveControl::new(initial as u64, slots as u64, unit_cap as u64, pinned);
        tracing::info!(slots, unit_cap, initial = control.waves(), pinned, candidates = ?control.candidates(),
            reserve_rows = reserve_rows(slots as u64, unit_cap as u64), "prefill depth control");
        Ok(Self { slots, unit_cap, control: RefCell::new(control) })
    }

    /// Admitted in-flight slots (fixed at startup).
    pub(crate) fn slots(&self) -> usize { self.slots }
    /// The widest unit admitted (fixed at startup).
    pub(crate) fn unit_cap(&self) -> usize { self.unit_cap }
    /// Rows every per-slot plane spans so any depth's slots are views in it.
    pub(crate) fn reserve_rows(&self) -> usize { reserve_rows(self.slots as u64, self.unit_cap as u64) as usize }

    /// The shape of the next prefill.
    pub(crate) fn next(&self) -> Depth {
        let control = self.control.borrow();
        Depth { waves: control.waves() as usize, unit_rows: control.unit_rows() as usize }
    }

    /// A finished prefill that ran `waves` units deep reports the Sparks' pace.
    pub(crate) fn finish(&self, waves: usize, pace: &PrefillPace) -> Decision {
        let before = self.next();
        let d = self.control.borrow_mut().observe(waves as u64, pace);
        if d.waves as usize != before.waves {
            tracing::info!(waves = before.waves, unit_rows = before.unit_rows, next_waves = d.waves,
                next_unit_rows = d.unit_rows, rows = pace.rows, busy_us_per_row = ?d.busy_us_per_row,
                idle_us_per_row = ?d.idle_us_per_row, turnaround_us = ?d.turnaround_us, exact = d.exact,
                reason = d.reason, "prefill depth");
        } else {
            tracing::debug!(waves = before.waves, unit_rows = before.unit_rows, rows = pace.rows,
                busy_us_per_row = ?d.busy_us_per_row, idle_us_per_row = ?d.idle_us_per_row,
                turnaround_us = ?d.turnaround_us, exact = d.exact, reason = d.reason, "prefill depth");
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depths_stay_inside_the_admitted_slots_and_reserve() -> anyhow::Result<()> {
        let depth = PrefillDepth::new(3, 4, 2752, false)?;
        assert_eq!((depth.next(), depth.reserve_rows()), (Depth { waves: 3, unit_rows: 2752 }, 8256));
        let pinned = PrefillDepth::new(2, 4, 4096, true)?;
        let starving = PrefillPace { rows: 60_000, ranks: vec![(1_000_000, 500_000); 6] };
        for _ in 0..4 { pinned.finish(2, &starving); }
        assert_eq!(pinned.next(), Depth { waves: 2, unit_rows: 4096 });
        assert!(PrefillDepth::new(5, 4, 2048, false).is_err());
        assert!(PrefillDepth::new(1, 4, 8192, false).is_err());
        Ok(())
    }
}
