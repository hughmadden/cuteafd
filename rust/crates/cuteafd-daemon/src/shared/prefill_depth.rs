//! Per-engine runtime prefill depth (#762 stages 2–4). An engine whose
//! prefill runs on the shared executor holds one [`PrefillDepth`]: the
//! planner's admitted slots and unit extent, and the shared
//! [`WaveControl`]. Before each prefill call (one window cycle) the engine
//! reports every finished call's pace from its Spark link
//! ([`PrefillDepth::report`]) and takes this call's depth
//! ([`PrefillDepth::start`]): fewer, wider units only at a prefill's opening
//! call, more units at any call. Storage never changes: every depth's slots
//! are views at `slot * unit` inside per-slot planes sized for
//! [`reserve_rows`], and the Spark prefill owner accepts any unit up to the
//! admitted one.

use cuteafd_loader::serving_capacity::wave_control::{call_rows, reserve_rows, Boundary, Decision, PrefillPace,
    WaveControl};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;

/// The shape of one prefill call: `waves` units of at most `unit_rows` in
/// flight, and its id on the controller's clock (passed to the link's
/// `begin_prefill`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Depth { pub waves: usize, pub unit_rows: usize, pub call: u64 }

/// Calls whose pace may still arrive (older ones are dropped unreported).
const PENDING: usize = 16;

pub(crate) struct PrefillDepth {
    slots: usize,
    unit_cap: usize,
    control: RefCell<WaveControl>,
    /// Started calls not yet reported: the decision and the rows the call ran.
    pending: RefCell<VecDeque<(Decision, u64)>>,
    /// The last call's sequence key and end row: a call starting there runs on.
    last: Cell<Option<(u64, usize)>>,
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
        Ok(Self { slots, unit_cap, control: RefCell::new(control), pending: RefCell::new(VecDeque::new()),
            last: Cell::new(None) })
    }

    /// Admitted in-flight slots (fixed at startup).
    pub(crate) fn slots(&self) -> usize { self.slots }
    /// The widest unit admitted (fixed at startup).
    pub(crate) fn unit_cap(&self) -> usize { self.unit_cap }
    /// Rows every per-slot plane spans so any depth's slots are views in it.
    pub(crate) fn reserve_rows(&self) -> usize { reserve_rows(self.slots as u64, self.unit_cap as u64) as usize }
    /// Longest prefill call: whole units at every depth the controller may pick.
    pub(crate) fn call_rows(&self) -> usize {
        if self.control.borrow().pinned() { return self.reserve_rows(); }
        call_rows(self.slots as u64, self.unit_cap as u64) as usize
    }

    /// The current depth (a pinned controller's only one), without starting a call.
    pub(crate) fn current(&self) -> (usize, usize) {
        let control = self.control.borrow();
        (control.waves() as usize, control.unit_rows() as usize)
    }

    /// Finished calls' paces from the link ([`crate::shared::spark_waves::Waves::take_prefill`]).
    pub(crate) fn report(&self, paces: Vec<(u64, PrefillPace)>) {
        for (call, pace) in paces {
            let Some(at) = self.pending.borrow().iter().position(|(decision, _)| decision.call == call) else { continue };
            let (decision, rows) = self.pending.borrow_mut().remove(at).expect("pending call");
            let sample = self.control.borrow_mut().record(&decision, rows, &pace);
            // One line per finished call: its depth, period and both waits.
            tracing::info!(call, ran = sample.waves, rows, boundary = ?sample.boundary, transition = decision.transition,
                counted = sample.counted, wall_us = pace.wall_us, period_us = ?sample.period_us,
                idle_share = ?sample.idle_share, reply_wait_share = ?sample.reply_wait_share, "prefill call");
        }
    }

    /// A prefill call of sequence `key` starts at row `start`: it runs on when
    /// the previous call was the same sequence's and ended there; otherwise
    /// it opens a prefill. Returns its depth.
    pub(crate) fn start(&self, key: u64, start: usize) -> Depth {
        let running = start > 0 && self.last.get() == Some((key, start));
        let boundary = if running { Boundary::Running } else { Boundary::Opening };
        let decision = self.control.borrow_mut().next(boundary);
        let mut pending = self.pending.borrow_mut();
        if pending.len() == PENDING { pending.pop_front(); }
        pending.push_back((decision, 0));
        self.last.set(Some((key, start)));
        tracing::debug!(call = decision.call, waves = decision.waves, unit_rows = decision.unit_rows,
            boundary = ?boundary, transition = decision.transition, reason = decision.reason, "prefill depth");
        Depth { waves: decision.waves as usize, unit_rows: decision.unit_rows as usize, call: decision.call }
    }

    /// Call `depth.call` cut `units` units of `rows` prompt rows in all
    /// (fewer units than its depth when short) and ends at row `end`.
    pub(crate) fn ran(&self, depth: Depth, units: usize, rows: usize) {
        if let Some((decision, ran)) = self.pending.borrow_mut().iter_mut().find(|(d, _)| d.call == depth.call) {
            decision.waves = decision.waves.min(units as u64);
            *ran = rows as u64;
        }
        if let Some((key, start)) = self.last.get() { self.last.set(Some((key, start + rows))); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pace(rows: u64, wall_us: u64, busy: u64, idle: u64) -> PrefillPace {
        PrefillPace { rows, ranks: vec![(busy, idle); 6], wall_us, reply_wait_us: Some(0) }
    }

    #[test]
    fn depths_stay_inside_the_admitted_slots_and_reserve() -> anyhow::Result<()> {
        let depth = PrefillDepth::new(3, 4, 2752, false)?;
        assert_eq!((depth.current(), depth.reserve_rows()), ((3, 2752), 8256));
        assert_eq!(depth.call_rows(), 8192);
        assert_eq!(PrefillDepth::new(3, 3, 2752, true)?.call_rows(), 8256);
        let pinned = PrefillDepth::new(2, 4, 4096, true)?;
        for call in 0..4 {
            let d = pinned.start(1, call * 8192);
            pinned.ran(d, 2, 8192);
            pinned.report(vec![(d.call, pace(8192, 3_000_000, 1_000_000, 500_000))]);
        }
        assert_eq!(pinned.current(), (2, 4096));
        assert!(PrefillDepth::new(5, 4, 2048, false).is_err());
        assert!(PrefillDepth::new(1, 4, 8192, false).is_err());
        Ok(())
    }

    #[test]
    fn calls_open_a_prefill_or_run_on_where_the_last_one_ended() -> anyhow::Result<()> {
        let depth = PrefillDepth::new(3, 4, 4096, false)?;
        // Sparks waiting on the coordinator point at fewer, wider units, but a
        // running call never widens: only the next prompt's opening call does.
        let first = depth.start(7, 0);
        assert_eq!(first.waves, 3);
        depth.ran(first, 3, 8192);
        depth.report(vec![(first.call, pace(8192, 2_400_000, 1_600_000, 400_000))]);
        let second = depth.start(7, 8192);
        assert_eq!(second.waves, 3, "a running call keeps or deepens its depth");
        depth.ran(second, 3, 8192);
        // A different sequence (or a gap) opens: now the wider trial runs.
        let third = depth.start(8, 0);
        assert_eq!(third.waves, 2);
        // A call below 60% of the smallest unit is no sample: the depth stays.
        depth.ran(third, 1, 1000);
        depth.report(vec![(second.call, pace(8192, 2_400_000, 1_600_000, 400_000)),
            (third.call, pace(1000, 1_000_000, 900_000, 100_000))]);
        assert_eq!(depth.start(9, 0).waves, 2);
        Ok(())
    }
}
