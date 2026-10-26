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

/// Idle share above which the Sparks starve: a deeper, narrower depth is tried.
pub const IDLE_ADD: f64 = 0.05;
/// Idle share below which the Sparks never starve: a shallower, wider depth may pay.
pub const IDLE_DROP: f64 = 0.01;
/// Consecutive non-starving full-depth prefills before a wider depth is retried.
pub const HOLD: u32 = 3;
/// A move to another depth needs its period per row to beat the current
/// depth's by this factor (hysteresis against noise).
pub const KEEP: f64 = 1.02;
/// Full-depth prefills after which a depth's measurement is stale and may be
/// retried when the idle signal points at it.
pub const STALE: u32 = 64;
/// Weight of the newest call in each depth's running means.
const WEIGHT: f64 = 0.25;

/// The Sparks' measured cost of one prefill call at one depth: per rank, the
/// summed unit busy time and the summed idle gap before each unit (each
/// rank's first unit of the call excluded: its gap is the decode boundary).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrefillPace {
    /// Prompt rows those units carried.
    pub rows: u64,
    /// Per Spark rank: (Σ busy µs, Σ idle µs).
    pub ranks: Vec<(u64, u64)>,
}

impl PrefillPace {
    /// (Spark wall per prompt row in µs, idle share): the slowest rank's busy
    /// + idle time per row, and the busiest rank's idle share (it sets the pace).
    pub fn summary(&self) -> Option<(f64, f64)> {
        if self.rows == 0 { return None; }
        let &(busy, idle) = self.ranks.iter().max_by_key(|(busy, _)| *busy)?;
        let wall = self.ranks.iter().map(|(busy, idle)| busy + idle).max()?;
        if busy + idle == 0 || wall == 0 { return None; }
        Some((wall as f64 / self.rows as f64, idle as f64 / (busy + idle) as f64))
    }
}

/// The unit a depth runs at inside the admitted extent `unit_cap`: the
/// shared reserve's unit for `k`, never wider than what was admitted.
pub fn unit_for(k: u64, unit_cap: u64) -> u64 { unit_rows(k).min(unit_cap) }

/// The depths worth running under `max` waves and `unit_cap` rows: for each
/// distinct unit, its deepest depth (more units in flight at the same unit
/// only hide more turnaround at the same Spark cost per row). Ascending.
pub fn candidates(max: u64, unit_cap: u64) -> Vec<u64> {
    let max = max.clamp(1, MAX_WAVES);
    (1..=max).filter(|&k| k == max || unit_for(k + 1, unit_cap) != unit_for(k, unit_cap)).collect()
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Depth { period: Option<f64>, idle: Option<f64>, age: u32 }

/// One boundary's decision (logged by the executor).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    pub waves: u64,
    pub unit_rows: u64,
    pub period_us_per_row: Option<f64>,
    pub idle_share: Option<f64>,
    pub reason: &'static str,
}

/// Chooses the depth of each prefill among [`candidates`]. Starvation (idle
/// above [`IDLE_ADD`]) tries the next deeper, narrower depth; no starvation
/// for [`HOLD`] calls tries the next wider one; every candidate is measured
/// once, because Spark cost per row is not monotone in unit width (Step's
/// grouped-FP8 2048→2752 cliff, #762). The controller then runs the depth
/// with the lowest measured Spark period per row. A pinned controller never
/// moves (explicit PREFILL_WAVES, benchmarks, goldens).
#[derive(Debug, Clone, PartialEq)]
pub struct WaveControl {
    k: u64,
    unit_cap: u64,
    pinned: bool,
    candidates: Vec<u64>,
    quiet: u32,
    depths: [Depth; MAX_WAVES as usize],
}

impl WaveControl {
    /// Start at the planner's estimate `initial`, moved to the deepest depth
    /// running the same unit.
    pub fn new(initial: u64, max: u64, unit_cap: u64, pinned: bool) -> Self {
        let candidates = candidates(max, unit_cap);
        let max = *candidates.last().expect("at least one depth");
        let initial = initial.clamp(1, max);
        let k = if pinned { initial } else {
            *candidates.iter().find(|&&k| k >= initial).expect("max is a candidate")
        };
        Self { k, unit_cap, pinned, candidates, quiet: 0, depths: Default::default() }
    }
    /// The wave count of the next prefill.
    pub fn waves(&self) -> u64 { self.k }
    /// Its unit inside the admitted extent.
    pub fn unit_rows(&self) -> u64 { unit_for(self.k, self.unit_cap) }
    pub fn pinned(&self) -> bool { self.pinned }
    pub fn candidates(&self) -> &[u64] { &self.candidates }

    fn at(&self, k: u64) -> &Depth { &self.depths[(k - 1) as usize] }

    /// The candidate next to `k` towards deeper (`+1`) or wider (`-1`) units.
    fn neighbour(&self, k: u64, deeper: bool) -> Option<u64> {
        let index = self.candidates.iter().position(|&c| c == k)?;
        if deeper { self.candidates.get(index + 1).copied() } else { index.checked_sub(1).map(|i| self.candidates[i]) }
    }

    /// Never measured, or measured long ago.
    fn open(&self, k: u64) -> bool { let d = self.at(k); d.period.is_none() || d.age >= STALE }

    /// One finished prefill that ran `waves` deep (fewer when the prompt was
    /// short) reports its Spark pace. Only calls at the chosen depth count.
    /// Returns the next prefill's depth.
    ///
    /// Direction comes from the idle signal: a starving depth goes one deeper;
    /// a depth with no idle for [`HOLD`] calls tries one wider. Every admitted
    /// unit is measured once. Otherwise the depth with the lowest delivered
    /// Spark period per row (busy plus idle, the critical rank's wall) runs;
    /// a measurement older than [`STALE`] calls is retried when the idle
    /// signal points at it, so a drifting turnaround is followed.
    pub fn observe(&mut self, waves: u64, pace: &PrefillPace) -> Decision {
        let (period, idle) = match pace.summary() {
            Some(sample) if !self.pinned && waves == self.k => sample,
            _ => return self.decision(None, None, if self.pinned { "pinned" } else { "not full depth" }),
        };
        let k = self.k;
        for depth in self.depths.iter_mut() { depth.age = depth.age.saturating_add(1); }
        let entry = &mut self.depths[(k - 1) as usize];
        // A stale mean restarts from its new sample.
        let stale = entry.age > STALE;
        let mean = |old: Option<f64>, new: f64| Some(match old { Some(old) if !stale => old + WEIGHT * (new - old), _ => new });
        entry.period = mean(entry.period, period);
        entry.idle = mean(entry.idle, idle);
        entry.age = 0;
        self.quiet = if idle < IDLE_DROP { self.quiet + 1 } else { 0 };
        let starving = idle > IDLE_ADD;
        let deeper = self.neighbour(k, true).filter(|_| starving);
        let wider = self.neighbour(k, false).filter(|&w| self.quiet >= HOLD && self.open(w));
        let unmeasured = self.candidates.iter().copied().find(|&c| self.at(c).period.is_none());
        let (next, reason) = if let Some(d) = deeper {
            (d, "Sparks starve: one more, narrower wave")
        } else if let Some(w) = wider {
            (w, "no Spark idle: one fewer, wider wave")
        } else if let Some(u) = unmeasured.filter(|_| !starving) {
            (u, "measure every admitted unit once")
        } else {
            // Exploit: the lowest measured period per row; the current depth
            // keeps near-ties.
            let current = self.at(k).period.unwrap_or(f64::INFINITY);
            let best = self.candidates.iter().copied()
                .filter_map(|c| self.at(c).period.map(|p| (c, p))).min_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                Some((b, p)) if b != k && p * KEEP < current => (b, "lowest measured period per row"),
                _ => (k, "hold"),
            }
        };
        if next != k { self.quiet = 0; }
        self.k = next;
        self.decision(Some(period), Some(idle), reason)
    }

    fn decision(&self, period: Option<f64>, idle: Option<f64>, reason: &'static str) -> Decision {
        Decision { waves: self.k, unit_rows: self.unit_rows(), period_us_per_row: period, idle_share: idle, reason }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A k-deep pipeline over `units` units of `unit` rows on one Spark rank:
    /// busy `s(unit)` per unit; the coordinator returns a unit after its wave
    /// lands plus `t` (turnaround), so the Spark idles max(0, t - (k-1)·s).
    fn pace(k: u64, unit: u64, rows: u64, s: impl Fn(u64) -> f64, t: f64) -> PrefillPace {
        let units = rows.div_ceil(unit);
        let busy = s(unit);
        let idle = (t - (k as f64 - 1.0) * busy).max(0.0);
        // Each rank's first unit is excluded.
        let counted = units - 1;
        PrefillPace { rows: rows - unit, ranks: vec![((busy * counted as f64) as u64, (idle * counted as f64) as u64); 6] }
    }

    /// Runs `calls` 64K prefills from `start` and returns the settled depth.
    fn settle(start: u64, max: u64, cap: u64, s: impl Fn(u64) -> f64 + Copy, t: f64, calls: usize) -> (u64, Vec<u64>) {
        let mut control = WaveControl::new(start, max, cap, false);
        let mut trace = Vec::new();
        for _ in 0..calls {
            let k = control.waves();
            let decision = control.observe(k, &pace(k, control.unit_rows(), 65536, s, t));
            trace.push(decision.waves);
        }
        (control.waves(), trace)
    }

    /// Step FP8 max oracle (#762): 4096 units 20.9 ms, 2752 16.6 ms, 2048
    /// 8.38 ms; preceding gap ~1.1–2.2 ms. 4×2048 delivered best.
    /// Turnaround bounded by the oracle's ~1.1 ms gaps at 3×2048 (2·8.38 + 1.1).
    const STEP_T: f64 = 17_000.0;
    fn step_s(unit: u64) -> f64 {
        match unit { 4096 => 20_943.0, 2752 => 16_577.0, 2048 => 8_383.0, _ => 8_383.0 * unit as f64 / 2048.0 }
    }

    #[test]
    fn step_oracle_settles_on_four_by_2048_from_every_admitted_start() {
        // 2752 units cost twice a 2048 unit for 1.34× the rows (the grouped
        // FP8 BF16 cliff), 4096 units 2.5×: whatever the admitted extent and
        // start, the controller settles on 4×2048 within a few prefills.
        for cap in [2048, 2752, 4096] {
            for start in 1..=4 {
                let (k, trace) = settle(start, 4, cap, step_s, STEP_T, 40);
                assert_eq!(k, 4, "cap {cap} start {start}: {trace:?}");
                // At most one call per wider unit before it settles.
                assert!(trace.iter().skip(3).all(|&k| k == 4), "cap {cap} start {start}: {trace:?}");
            }
        }
    }

    #[test]
    fn glm_estimate_holds_three_waves() {
        // GLM K4 six Sparks: S ≈ 13 ms per 2752 unit, T ≈ 19.5 ms, and 2048
        // rows cost more per row (#762 audit). k=3 hides T (2·13 > 19.5).
        let s = |unit: u64| if unit == 2048 { 3.674 * 2048.0 * 13_000.0 / (3.448 * 2752.0) } else { 13_000.0 * unit as f64 / 2752.0 };
        let (k, trace) = settle(3, 4, 2752, s, 19_500.0, 40);
        assert_eq!(k, 3, "{trace:?}");
    }

    #[test]
    fn short_prompts_and_pins_never_move_the_depth() {
        let mut control = WaveControl::new(3, 4, 2752, false);
        let starving = pace(3, 2752, 65536, |_| 5_000.0, 50_000.0);
        assert_eq!(control.observe(2, &starving).reason, "not full depth");
        assert_eq!(control.waves(), 3);
        let mut pinned = WaveControl::new(2, 4, 4096, true);
        for _ in 0..10 { pinned.observe(2, &starving); }
        assert_eq!(pinned.waves(), 2);
        assert_eq!(control.observe(3, &PrefillPace::default()).waves, 3);
    }

    #[test]
    fn a_worse_explored_depth_is_left_after_one_call() {
        // GLM K4 per-row Spark curve (#762 audit): 2048 rows 3.674 us/row,
        // 2752 rows 3.448; T hidden at both. 4×2048 is explored once, then 3.
        let s = |unit: u64| if unit == 2048 { 3.674 * 2048.0 } else { 3.448 * unit as f64 };
        let mut control = WaveControl::new(3, 4, 2752, false);
        let d = control.observe(3, &pace(3, 2752, 65536, s, 15_000.0));
        assert_eq!((d.waves, d.reason), (4, "measure every admitted unit once"));
        let d = control.observe(4, &pace(4, 2048, 65536, s, 15_000.0));
        assert_eq!((d.waves, d.reason), (3, "lowest measured period per row"));
        for _ in 0..200 { assert_eq!(control.observe(3, &pace(3, 2752, 65536, s, 15_000.0)).waves, 3); }
    }

    #[test]
    fn a_wider_unit_that_starves_is_rejected_and_only_retried_when_stale() {
        // 4096 units are cheapest per row, but 2×4096 cannot hide T (one
        // 12.3 ms unit against 25 ms) while 3×2752 does (2·10.3 ms ≈ 20.5 ms
        // plus its own); 2048 units pay a fixed per-unit cost that 2752 hides.
        let s = |unit: u64| 3.0 * unit as f64 + if unit == 4096 { 0.0 } else { 2_000.0 } + if unit == 2048 { 3_000.0 } else { 0.0 };
        let t = 20_000.0;
        let mut control = WaveControl::new(3, 4, 4096, false);
        let mut trace = Vec::new();
        for _ in 0..300 {
            let k = control.waves();
            trace.push(control.observe(k, &pace(k, control.unit_rows(), 65536, s, t)).waves);
        }
        let at = |k| trace.iter().filter(|&&w| w == k).count();
        assert!(at(2) <= 300 / STALE as usize + 2, "2×4096 retried too often: {trace:?}");
        assert!(at(3) > 250, "{trace:?}");
    }

    #[test]
    fn candidates_are_the_deepest_depth_per_distinct_unit() {
        assert_eq!(candidates(4, 4096), [2, 3, 4]);
        assert_eq!(candidates(4, 2752), [3, 4]);
        assert_eq!(candidates(4, 2048), [4]);
        assert_eq!(candidates(2, 4096), [2]);
        assert_eq!(candidates(1, 4096), [1]);
        assert_eq!(WaveControl::new(1, 4, 2752, false).waves(), 3);
        assert_eq!(WaveControl::new(1, 4, 2752, true).waves(), 1);
    }

    #[test]
    fn units_never_exceed_the_admitted_extent() {
        for cap in [2048, 2752, 4096] {
            for k in 1..=4 {
                let unit = WaveControl::new(k, 4, cap, false).unit_rows();
                assert!(unit <= cap && unit * k <= 8256 && unit % 64 == 0, "cap {cap} k {k}: {unit}");
            }
        }
        assert_eq!(WaveControl::new(9, 2, 4096, false).waves(), 2);
    }

    #[test]
    fn summary_uses_the_busiest_rank_and_the_slowest_wall() {
        let pace = PrefillPace { rows: 100, ranks: vec![(900, 100), (950, 0), (500, 600)] };
        let (period, idle) = pace.summary().unwrap();
        assert!((period - 11.0).abs() < 1e-9);
        assert_eq!(idle, 0.0);
        assert!(PrefillPace { rows: 0, ranks: vec![(1, 1)] }.summary().is_none());
    }
}
