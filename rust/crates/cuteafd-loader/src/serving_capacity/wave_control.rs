//! Runtime prefill wave adaptation (#762 stages 2–4). The planner admits a
//! maximum of `slots` in-flight units and a unit extent; at each
//! decode→prefill transition the executor asks [`WaveControl`] for the depth
//! k′ of the next prefill, within that admission. Each finished full-depth
//! prefill reports what the Sparks measured on their prefill owner
//! ([`PrefillPace`]): per-unit busy time and the idle gap before each unit.
//! Their sum per row is the call's delivered Spark period per row; the
//! controller measures every candidate depth once and then runs the one with
//! the lowest period, re-measuring the current depth on each full-depth call.
//! Pure arithmetic over the reported counters; no clock, no I/O.

use super::prefill_waves::{unit_rows, MAX_WAVES};

/// A depth replaces the current one only if its measured period per row is
/// lower by this share (hysteresis against noise).
pub const MARGIN: f64 = 0.01;
/// Weight of the newest call in each depth's running period per row.
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

/// Rows one prefill call takes so that every candidate depth cuts it into
/// whole units: the smallest `k * unit_for(k)` over [`candidates`] (8192 for
/// four slots of 2752: 3 x 2752 cuts it 2752/2752/2688, 4 x 2048 exactly),
/// never the reserve itself (8256 at 4 x 2048 would cut eight 1088-row units).
pub fn call_rows(slots: u64, unit_cap: u64) -> u64 {
    candidates(slots, unit_cap).into_iter().map(|k| k * unit_for(k, unit_cap)).min().unwrap_or(0)
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
    /// The finished depth's running period per row (busy + idle, µs).
    pub period_us: Option<f64>,
    pub reason: &'static str,
}

/// Chooses each prefill's depth by its measured Spark period per row (#762),
/// separately for short and long contexts.
///
/// The delivered prefill rate is one unit per Spark period, so the Sparks'
/// busy + idle time per row is the call's wall period per row: busy is their
/// work, idle whatever the coordinator did not hide. Both depend on the unit
/// size (busy per row on the kernels' row curve; the coordinator's turnaround
/// per row on its per-unit costs), so no depth is predicted from another:
/// each candidate (the deepest depth per distinct unit) runs once, then the
/// lowest measured period wins, and every full-depth call refreshes its own
/// depth's running period (context length and load move it). Pinned
/// controllers never move (explicit PREFILL_WAVES, benchmarks, goldens).
#[derive(Debug, Clone, PartialEq)]
pub struct WaveControl {
    unit_cap: u64,
    pinned: bool,
    candidates: Vec<u64>,
    /// Per context class: the chosen depth and each depth's running Spark
    /// period µs per row (busy + idle).
    classes: [Class; CLASSES],
    /// The class of the last decision (its depth is `waves()`).
    current: usize,
}

/// One context class's state ([`class_of`]).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Class { k: u64, period: [Option<f64>; MAX_WAVES as usize] }

/// Context classes: a call's period depends on the context its attention
/// reads (64K prefills deliver less per row than 8K), so depths are compared
/// only within a class.
const CLASSES: usize = 2;
/// Calls ending at most this many context rows are the short class.
pub const SHORT_CONTEXT: u64 = 16_384;

/// The class of a call ending at context row `context`.
pub fn class_of(context: u64) -> usize { usize::from(context > SHORT_CONTEXT) }

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
        Self { unit_cap, pinned, candidates, classes: [Class { k, period: [None; MAX_WAVES as usize] }; CLASSES],
            current: 0 }
    }
    /// The wave count of the last chosen prefill.
    pub fn waves(&self) -> u64 { self.classes[self.current].k }
    /// Its unit inside the admitted extent.
    pub fn unit_rows(&self) -> u64 { unit_for(self.waves(), self.unit_cap) }
    pub fn pinned(&self) -> bool { self.pinned }
    pub fn candidates(&self) -> &[u64] { &self.candidates }

    /// The depth of a prefill call ending at context row `context`: its
    /// class's next unmeasured candidate, else its lowest measured period.
    pub fn choose(&mut self, context: u64) -> u64 {
        self.current = class_of(context);
        self.waves()
    }

    /// One finished prefill that ran `waves` deep (fewer when the prompt was
    /// short) and ended at context row `context` reports its Spark pace; only
    /// calls at their class's chosen depth count. Returns that class's next depth.
    pub fn observe(&mut self, waves: u64, context: u64, pace: &PrefillPace) -> Decision {
        let index = class_of(context);
        self.current = index;
        let Some((busy, idle)) = pace.per_row().filter(|_| !self.pinned && waves == self.classes[index].k) else {
            return self.decision(None, None, if self.pinned { "pinned" } else { "not full depth" });
        };
        let candidates = self.candidates.clone();
        let class = &mut self.classes[index];
        let k = class.k;
        let cell = &mut class.period[(k - 1) as usize];
        let measured = busy + idle;
        *cell = Some(cell.map_or(measured, |old| old + WEIGHT * (measured - old)));
        let reason = if let Some(&unknown) = candidates.iter().find(|&&j| class.period[(j - 1) as usize].is_none()) {
            class.k = unknown;
            "measure each candidate depth once"
        } else {
            let current = class.period[(k - 1) as usize].expect("this call measured the current depth");
            let best = candidates.iter().filter_map(|&j| class.period[(j - 1) as usize].map(|p| (j, p)))
                .min_by(|a, b| a.1.total_cmp(&b.1)).expect("measured candidates");
            if best.0 != k && best.1 < (1.0 - MARGIN) * current {
                class.k = best.0;
                "lowest measured Spark period per row"
            } else { "hold" }
        };
        let period = class.period[(k - 1) as usize];
        let mut decision = self.decision(Some(busy), Some(idle), reason);
        decision.period_us = period;
        decision
    }

    fn decision(&self, busy: Option<f64>, idle: Option<f64>, reason: &'static str) -> Decision {
        Decision { waves: self.waves(), unit_rows: self.unit_rows(), busy_us_per_row: busy, idle_us_per_row: idle,
            period_us: None, reason }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One rank of a k-deep pipeline: busy `s(unit)` µs per unit; the
    /// coordinator needs `tau` µs per row of a unit (depending on the unit),
    /// so each unit after the first waits max(0, tau·unit − (k−1)·busy).
    fn pace(k: u64, unit: u64, rows: u64, s: impl Fn(u64) -> f64, tau: f64) -> PrefillPace {
        let busy = s(unit);
        let idle = (tau * unit as f64 - (k as f64 - 1.0) * busy).max(0.0);
        let counted = rows.div_ceil(unit) - 1;
        PrefillPace { rows: rows - unit, ranks: vec![((busy * counted as f64) as u64, (idle * counted as f64) as u64); 6] }
    }

    fn run(control: &mut WaveControl, lengths: &[u64], s: impl Fn(u64) -> f64 + Copy, t: impl Fn(u64) -> f64) -> Vec<u64> {
        run_units(control, lengths, s, |rows, _| t(rows))
    }

    /// [`run`] with the coordinator's turnaround per row depending on the
    /// prompt length and the unit (`t(rows, unit)`).
    fn run_units(control: &mut WaveControl, lengths: &[u64], s: impl Fn(u64) -> f64 + Copy,
        t: impl Fn(u64, u64) -> f64) -> Vec<u64> {
        lengths.iter().map(|&rows| {
            let k = control.choose(rows);
            let unit = control.unit_rows();
            control.observe(k, rows, &pace(k, unit, rows, s, t(rows, unit)));
            k
        }).collect()
    }

    /// Spark busy and coordinator turnaround per row (µs) by unit, from the
    /// #762 cards' adaptive arms (rw762-cards-a, U = 4096): 3 x 2752 busy
    /// 3.71 / idle 0.29 (GLM K4 max), 4.97 / 0.93 (Kimi BF16 max); 2 x 4096
    /// 3.36 / 1.54 and 4.45 / 1.74; 4 x 2048 delivered 7.9% / 11% less than
    /// 3 x 2752 at 8K. τ = idle + (k − 1) · busy at each depth.
    fn measured(family: &str) -> (impl Fn(u64) -> f64 + Copy, impl Fn(u64, u64) -> f64 + Copy) {
        let glm = family == "glm";
        let busy = move |unit: u64| match (glm, unit) {
            (true, 4096) => 3.36, (true, 2752) => 3.71, (true, _) => 3.90,
            (false, 4096) => 4.45, (false, 2752) => 4.97, (false, _) => 5.40 };
        let tau = move |_: u64, unit: u64| match (glm, unit) {
            (true, 4096) => 1.54 + 3.36, (true, 2752) => 0.29 + 2.0 * 3.71, (true, _) => 0.44 + 3.0 * 3.90,
            (false, 4096) => 1.74 + 4.45, (false, 2752) => 0.93 + 2.0 * 4.97, (false, _) => 1.25 + 3.0 * 5.40 };
        (move |unit: u64| busy(unit) * unit as f64, tau)
    }

    #[test]
    fn unit_dependent_turnaround_settles_glm_and_kimi_on_three_by_2752() {
        // The superseded S/T model held 4 x 2048 here: it assumed one
        // turnaround per row for every unit. The measured period does not.
        for family in ["glm", "kimi"] {
            let (s, t) = measured(family);
            for start in 2..=4 {
                let mut control = WaveControl::new(start, 4, 4096, false);
                let trace = run_units(&mut control, &bench(40), s, t);
                assert!(trace[12..].iter().all(|&k| k == 3), "{family} start {start}: {trace:?}");
            }
        }
    }

    /// The bench's order: warm and timed 8K, then warm and timed 64K, repeated.
    fn bench(calls: usize) -> Vec<u64> { (0..calls).map(|i| if i % 4 < 2 { 8192 } else { 65536 }).collect() }

    /// Step FP8 max on the A8 wire (#802 unit runs, warmed prefill tok/s at
    /// 8K / 64K): 4 x 2048 2483 / 2357, 3 x 2752 2720 / 2425, 2 x 4096 2347 /
    /// 2253. A call's period per row is the inverse of its delivered rate;
    /// the Sparks are busy 80% of it and idle the rest.
    fn step_a8(k: u64, rows: u64) -> PrefillPace {
        let rate = match (k, rows > 16_384) {
            (4, false) => 2483.0, (4, true) => 2357.0, (3, false) => 2720.0, (3, true) => 2425.0,
            (_, false) => 2347.0, (_, true) => 2253.0 };
        let period = 1e6 / rate;
        let counted = rows - unit_rows(k);
        PrefillPace { rows: counted, ranks: vec![((0.8 * period * counted as f64) as u64,
            (0.2 * period * counted as f64) as u64); 6] }
    }

    #[test]
    fn step_a8_settles_on_three_by_2752_from_every_start() {
        for start in 2..=4 {
            let mut control = WaveControl::new(start, 4, 4096, false);
            let trace: Vec<u64> = bench(40).iter().map(|&rows| {
                let k = control.choose(rows);
                control.observe(k, rows, &step_a8(k, rows));
                k
            }).collect();
            assert!(trace[12..].iter().all(|&k| k == 3), "start {start}: {trace:?}");
        }
    }

    /// The superseded BF16-input Step oracle: full-unit compute 4096 rows 20.9
    /// ms, 2752 16.6 ms, 2048 8.38 ms (the grouped-FP8 2048→2304 cliff).
    fn step_s(unit: u64) -> f64 {
        match unit { 4096 => 20_943.0, 2752 => 16_577.0, _ => 8_383.0 * unit as f64 / 2048.0 }
    }

    #[test]
    fn a_kernel_row_cliff_moves_the_choice_to_four_by_2048() {
        // Each admitted unit runs once, then the cheapest period holds.
        for t in [2.0, 8.7, 12.0] {
            for start in 2..=4 {
                let mut control = WaveControl::new(start, 4, 4096, false);
                let trace = run(&mut control, &bench(40), step_s, |_| t);
                assert!(trace[12..].iter().all(|&k| k == 4), "T {t} start {start}: {trace:?}");
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
        assert!(trace[8..].iter().all(|&k| k == 3), "{trace:?}");
    }

    #[test]
    fn wider_units_win_when_turnaround_is_hidden_and_cost_per_row_falls() {
        let s = |unit: u64| 1_000.0 + 3.0 * unit as f64;
        let mut control = WaveControl::new(4, 4, 4096, false);
        run(&mut control, &bench(24), s, |_| 1.0);
        assert_eq!(control.waves(), 2);
        // Large turnaround: more units in flight beat wider ones.
        let mut control = WaveControl::new(2, 4, 4096, false);
        run(&mut control, &bench(24), s, |_| 15.0);
        assert_eq!(control.waves(), 4);
    }

    #[test]
    fn short_prompts_and_pins_never_move_the_depth() {
        let mut control = WaveControl::new(3, 4, 2752, false);
        let starving = pace(3, 2752, 65536, |_| 5_000.0, 20.0);
        assert_eq!(control.observe(2, 65536, &starving).reason, "not full depth");
        assert_eq!(control.waves(), 3);
        let mut pinned = WaveControl::new(2, 4, 4096, true);
        for _ in 0..10 { pinned.observe(2, 65536, &starving); }
        assert_eq!(pinned.waves(), 2);
        assert_eq!(control.observe(3, 65536, &PrefillPace::default()).waves, 3);
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
        // One call is whole units at every candidate depth (never 8 x 1088 at 4 x 2048).
        assert_eq!((call_rows(4, 2752), call_rows(4, 4096), call_rows(4, 2048), call_rows(3, 2752), call_rows(2, 4096)),
            (8192, 8192, 8192, 8256, 8192));
    }

    #[test]
    fn per_row_uses_the_busiest_rank() {
        let pace = PrefillPace { rows: 100, ranks: vec![(900, 100), (950, 0), (500, 600)] };
        assert_eq!(pace.per_row(), Some((9.5, 0.0)));
        assert!(PrefillPace { rows: 0, ranks: vec![(1, 1)] }.per_row().is_none());
    }
}
