//! Runtime prefill wave adaptation (#762 stages 2–4). The planner admits a
//! maximum of `max` in-flight units and a unit extent; at each decode→prefill
//! transition the executor asks [`WaveControl`] for the wave count k′ of the
//! next prefill, within that admission. Each finished full-depth prefill
//! reports what the Sparks measured on their prefill owner ([`PrefillPace`]):
//! per-unit busy time S and the idle gap I before each unit started. The
//! controller adds a wave while the Sparks starve, tries one fewer while they
//! never do, and keeps a move only if the delivered Spark period per row did
//! not regress. Pure arithmetic over the reported counters; no clock, no I/O.

use super::prefill_waves::{unit_rows, MAX_WAVES};

/// Idle share above which the Sparks wait between units ("gaps above zero",
/// above the timing noise floor): the call measures turnaround exactly.
pub const IDLE_ADD: f64 = 0.01;
/// A depth replaces the current one only if its predicted period per row is
/// lower by this share (hysteresis against noise).
pub const MARGIN: f64 = 0.01;
/// Weight of the newest call in each unit's running Spark cost per row.
const WEIGHT: f64 = 0.25;

/// The Sparks' measured cost of one prefill call at one depth: per rank, the
/// summed unit busy time (from the unit's arrival, input copies included, to
/// its completion) and the summed idle gap before each unit; each rank's
/// first unit of the call is excluded (its gap is the decode boundary).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrefillPace {
    /// Prompt rows those units carried.
    pub rows: u64,
    /// Per Spark rank: (Σ busy µs, Σ idle µs).
    pub ranks: Vec<(u64, u64)>,
}

impl PrefillPace {
    /// The critical (busiest) rank's (busy µs per row, idle µs per row).
    pub fn per_row(&self) -> Option<(f64, f64)> {
        if self.rows == 0 { return None; }
        let &(busy, idle) = self.ranks.iter().max_by_key(|(busy, _)| *busy)?;
        (busy > 0).then(|| (busy as f64 / self.rows as f64, idle as f64 / self.rows as f64))
    }
}

/// The unit a depth runs at inside the admitted extent `unit_cap`: the
/// shared reserve's unit for `k`, never wider than what was admitted.
pub fn unit_for(k: u64, unit_cap: u64) -> u64 { unit_rows(k).min(unit_cap) }

/// Rows every per-slot plane spans so any depth up to `slots` fits as views
/// at `slot * unit_for(k)`: the largest `k * unit_for(k)` (8256 at most).
pub fn reserve_rows(slots: u64, unit_cap: u64) -> u64 {
    (1..=slots.clamp(1, MAX_WAVES)).map(|k| k * unit_for(k, unit_cap)).max().unwrap_or(0)
}

/// The depths worth running under `slots` and `unit_cap`: for each distinct
/// unit, its deepest depth (more units in flight at the same unit only hide
/// more turnaround at the same Spark cost per row). Ascending.
pub fn candidates(slots: u64, unit_cap: u64) -> Vec<u64> {
    let max = slots.clamp(1, MAX_WAVES);
    (1..=max).filter(|&k| k == max || unit_for(k + 1, unit_cap) != unit_for(k, unit_cap)).collect()
}

/// One boundary's decision (logged by the executor).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    pub waves: u64,
    pub unit_rows: u64,
    /// The finished call's Spark busy and idle µs per row (critical rank).
    pub busy_us_per_row: Option<f64>,
    pub idle_us_per_row: Option<f64>,
    /// Coordinator turnaround per row the choice used (µs), and whether it
    /// is exact (the Sparks waited) or an upper bound (they never did).
    pub turnaround_us: Option<f64>,
    pub exact: bool,
    pub reason: &'static str,
}

/// Chooses each prefill's depth from measured S and T (#762 stages 2–4).
///
/// Per row of a unit, the Spark is busy s(unit) and the coordinator turns
/// a unit around in τ (attention, routing and norms scale with the unit's
/// rows; τ grows with context). With k units in flight a Spark waits only
/// when τ exceeds the k−1 units queued behind it: idle per row =
/// max(0, τ − (k−1)·s). s depends on the unit's rows, not on the prompt, so
/// it is learned once per admitted unit (each candidate runs once). τ is
/// re-measured by every full-depth call: exactly when the Sparks waited,
/// else bounded by (k−1)·s (the tightest bound since the last exact value). It runs the candidate with the
/// lowest predicted Spark period per row, s + max(0, τ − (k−1)·s). Pinned
/// controllers never move (explicit PREFILL_WAVES, benchmarks, goldens).
#[derive(Debug, Clone, PartialEq)]
pub struct WaveControl {
    k: u64,
    unit_cap: u64,
    pinned: bool,
    candidates: Vec<u64>,
    /// Per depth: Spark busy µs per row.
    busy: [Option<f64>; MAX_WAVES as usize],
    /// The last full-depth call's turnaround per row (µs) and whether it is exact.
    turnaround: Option<(f64, bool)>,
}

impl WaveControl {
    /// Start at the planner's estimate `initial`, moved to the deepest depth
    /// that runs the same unit.
    pub fn new(initial: u64, slots: u64, unit_cap: u64, pinned: bool) -> Self {
        let candidates = candidates(slots, unit_cap);
        let max = *candidates.last().expect("at least one depth");
        let initial = initial.clamp(1, max);
        let k = if pinned { initial } else {
            *candidates.iter().find(|&&k| k >= initial).expect("max is a candidate")
        };
        Self { k, unit_cap, pinned, candidates, busy: [None; MAX_WAVES as usize], turnaround: None }
    }
    /// The wave count of the next prefill.
    pub fn waves(&self) -> u64 { self.k }
    /// Its unit inside the admitted extent.
    pub fn unit_rows(&self) -> u64 { unit_for(self.k, self.unit_cap) }
    pub fn pinned(&self) -> bool { self.pinned }
    pub fn candidates(&self) -> &[u64] { &self.candidates }

    /// Predicted Spark period per row at depth `j` under turnaround `tau` per row.
    fn predict(&self, j: u64, tau: f64) -> Option<f64> {
        let s = self.busy[(j - 1) as usize]?;
        Some(s + (tau - (j as f64 - 1.0) * s).max(0.0))
    }

    /// One finished prefill that ran `waves` deep (fewer when the prompt was
    /// short) reports its Spark pace; only calls at the chosen depth count.
    /// Returns the next prefill's depth.
    pub fn observe(&mut self, waves: u64, pace: &PrefillPace) -> Decision {
        let Some((busy, idle)) = pace.per_row().filter(|_| !self.pinned && waves == self.k) else {
            return self.decision(None, None, if self.pinned { "pinned" } else { "not full depth" });
        };
        let k = self.k;
        let cell = &mut self.busy[(k - 1) as usize];
        *cell = Some(cell.map_or(busy, |old| old + WEIGHT * (busy - old)));
        let queued = (k as f64 - 1.0) * busy;
        // Sparks that waited measure τ exactly. Sparks that never waited only
        // bound it by the work queued behind each unit; a deeper depth's bound
        // is looser, so the tightest bound seen since the last exact value
        // applies (too tight a bound makes the next call wait and corrects it).
        let exact = idle / (busy + idle) > IDLE_ADD;
        self.turnaround = Some(if exact { (idle + queued, true) } else {
            (self.turnaround.map_or(queued, |(previous, _)| previous.min(queued)), false)
        });
        let reason = if let Some(&unknown) = self.candidates.iter().find(|&&j| self.busy[(j - 1) as usize].is_none()) {
            self.k = unknown;
            "measure an admitted unit's Spark cost once"
        } else {
            let (t, _) = self.turnaround.expect("this call set a turnaround");
            let current = self.predict(k, t).expect("this call measured the current depth");
            let best = self.candidates.iter().filter_map(|&j| self.predict(j, t).map(|p| (j, p)))
                .min_by(|a, b| a.1.total_cmp(&b.1)).expect("measured candidates");
            if best.0 != k && best.1 < (1.0 - MARGIN) * current {
                self.k = best.0;
                "lowest predicted Spark period per row"
            } else { "hold" }
        };
        self.decision(Some(busy), Some(idle), reason)
    }

    fn decision(&self, busy: Option<f64>, idle: Option<f64>, reason: &'static str) -> Decision {
        let turnaround = self.turnaround;
        Decision { waves: self.k, unit_rows: self.unit_rows(), busy_us_per_row: busy, idle_us_per_row: idle,
            turnaround_us: turnaround.map(|t| t.0), exact: turnaround.is_some_and(|t| t.1), reason }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One rank of a k-deep pipeline: busy `s(unit)` µs per unit; the
    /// coordinator needs `tau` µs per row of a unit, so each unit after the
    /// first waits max(0, tau·unit − (k−1)·busy). `rows` prompt rows.
    fn pace(k: u64, unit: u64, rows: u64, s: impl Fn(u64) -> f64, tau: f64) -> PrefillPace {
        let busy = s(unit);
        let idle = (tau * unit as f64 - (k as f64 - 1.0) * busy).max(0.0);
        let counted = rows.div_ceil(unit) - 1;
        PrefillPace { rows: rows - unit, ranks: vec![((busy * counted as f64) as u64, (idle * counted as f64) as u64); 6] }
    }

    fn run(control: &mut WaveControl, lengths: &[u64], s: impl Fn(u64) -> f64 + Copy, t: impl Fn(u64) -> f64) -> Vec<u64> {
        lengths.iter().map(|&rows| {
            let k = control.waves();
            control.observe(k, &pace(k, control.unit_rows(), rows, s, t(rows))).waves
        }).collect()
    }

    /// The bench's order: warm and timed 8K, then warm and timed 64K, repeated.
    fn bench(calls: usize) -> Vec<u64> { (0..calls).map(|i| if i % 4 < 2 { 8192 } else { 65536 }).collect() }

    /// Step FP8 max oracle (#762): full-unit compute 4096 rows 20.9 ms, 2752
    /// 16.6 ms, 2048 8.38 ms (the grouped-FP8 BF16 2048→2304 cliff); 4×2048
    /// delivered best.
    fn step_s(unit: u64) -> f64 {
        match unit { 4096 => 20_943.0, 2752 => 16_577.0, _ => 8_383.0 * unit as f64 / 2048.0 }
    }

    #[test]
    fn step_lands_on_four_by_2048_from_every_start_and_admitted_unit() {
        // Oracle turnaround: 3×2048 waited ~1.1 ms per unit, τ ≈ (2·8.38 +
        // 1.13) ms / 2048 rows ≈ 8.7 µs/row. Whether the Sparks starve at the
        // start or not, each admitted unit runs once, then 4×2048 holds.
        for t in [2.0, 8.7, 12.0] {
            for cap in [2048, 2752, 4096] {
                for start in 1..=4 {
                    let mut control = WaveControl::new(start, 4, cap, false);
                    let trace = run(&mut control, &bench(40), step_s, |_| t);
                    let settled = control.candidates().len();
                    assert!(trace[settled..].iter().all(|&k| k == 4), "T {t} cap {cap} start {start}: {trace:?}");
                }
            }
        }
    }

    #[test]
    fn glm_holds_three_by_2752_after_one_look_at_four() {
        // GLM K4 six Sparks: S ≈ 13 ms per 2752 unit, T ≈ 19.5 ms hidden by
        // 3 units; 2048 rows cost 6.5% more per row (#762 row-band audit).
        let s = |unit: u64| if unit == 2048 { 3.674 * 2048.0 * 13_000.0 / (3.448 * 2752.0) }
            else { 13_000.0 * unit as f64 / 2752.0 };
        let mut control = WaveControl::new(3, 4, 2752, false);
        let trace = run(&mut control, &bench(200), s, |_| 19_500.0 / 2752.0);
        assert_eq!(&trace[..2], [4, 3], "{trace:?}");
        assert!(trace[2..].iter().all(|&k| k == 3), "{trace:?}");
        // A context-dependent turnaround that outgrows 3 units at 64K moves
        // 64K prefills deeper when that predicts a lower period, and back.
        let mut control = WaveControl::new(3, 4, 2752, false);
        let trace = run(&mut control, &bench(40), s, |rows| if rows > 16_384 { 22.0 } else { 5.0 });
        assert!(trace.contains(&3) && trace.contains(&4), "{trace:?}");
    }

    #[test]
    fn wider_units_win_when_turnaround_is_hidden_and_cost_per_row_falls() {
        let s = |unit: u64| 1_000.0 + 3.0 * unit as f64;
        let mut control = WaveControl::new(4, 4, 4096, false);
        run(&mut control, &bench(12), s, |_| 1.0);
        assert_eq!(control.waves(), 2);
        // Large turnaround: more units in flight beat wider ones.
        let mut control = WaveControl::new(2, 4, 4096, false);
        run(&mut control, &bench(12), s, |_| 15.0);
        assert_eq!(control.waves(), 4);
    }

    #[test]
    fn short_prompts_and_pins_never_move_the_depth() {
        let mut control = WaveControl::new(3, 4, 2752, false);
        let starving = pace(3, 2752, 65536, |_| 5_000.0, 20.0);
        assert_eq!(control.observe(2, &starving).reason, "not full depth");
        assert_eq!(control.waves(), 3);
        let mut pinned = WaveControl::new(2, 4, 4096, true);
        for _ in 0..10 { pinned.observe(2, &starving); }
        assert_eq!(pinned.waves(), 2);
        assert_eq!(control.observe(3, &PrefillPace::default()).waves, 3);
    }

    #[test]
    fn candidates_and_reserve_stay_inside_the_shared_reserve() {
        assert_eq!(candidates(4, 4096), [2, 3, 4]);
        assert_eq!(candidates(4, 2752), [3, 4]);
        assert_eq!(candidates(4, 2048), [4]);
        assert_eq!(candidates(2, 4096), [2]);
        assert_eq!(candidates(1, 4096), [1]);
        assert_eq!((reserve_rows(4, 4096), reserve_rows(4, 2752), reserve_rows(4, 2048), reserve_rows(2, 4096)),
            (8256, 8256, 8192, 8192));
        for cap in [2048, 2752, 4096] {
            for k in 1..=4 {
                let unit = WaveControl::new(k, 4, cap, true).unit_rows();
                assert!(unit <= cap && unit * k <= reserve_rows(4, cap) && unit % 64 == 0, "cap {cap} k {k}: {unit}");
            }
        }
        assert_eq!(WaveControl::new(1, 4, 2752, false).waves(), 3);
        assert_eq!(WaveControl::new(1, 4, 2752, true).waves(), 1);
        assert_eq!(WaveControl::new(9, 2, 4096, false).waves(), 2);
    }

    #[test]
    fn per_row_uses_the_busiest_rank() {
        let pace = PrefillPace { rows: 100, ranks: vec![(900, 100), (950, 0), (500, 600)] };
        assert_eq!(pace.per_row(), Some((9.5, 0.0)));
        assert!(PrefillPace { rows: 0, ranks: vec![(1, 1)] }.per_row().is_none());
    }
}
