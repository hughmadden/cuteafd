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

/// Idle share above which the Sparks starve: one more, narrower wave is tried.
pub const IDLE_ADD: f64 = 0.05;
/// Idle share below which the Sparks never starve.
pub const IDLE_DROP: f64 = 0.01;
/// Consecutive non-starving full-depth prefills before one fewer, wider wave is tried.
pub const HOLD: u32 = 3;
/// A trial keeps its new depth unless its period per row exceeds the old
/// depth's by this factor.
pub const KEEP: f64 = 1.02;
/// Full-depth prefills a rejected depth sits out before it may be retried.
pub const COOLDOWN: u32 = 64;
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
    /// plus idle time per row, and the busiest rank's idle share (it sets the pace).
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

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Depth { period: Option<f64>, cooldown: u32 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Steady { quiet: u32 },
    /// Running candidate `k` after `from`; decided by `k`'s next full-depth call.
    Trial { from: u64 },
}

/// One boundary's decision (logged by the executor).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    pub waves: u64,
    pub unit_rows: u64,
    pub period_us_per_row: Option<f64>,
    pub idle_share: Option<f64>,
    pub reason: &'static str,
}

/// Chooses the depth of each prefill among [`candidates`] at each
/// decode→prefill transition (#762): Spark idle above [`IDLE_ADD`] tries the
/// next deeper, narrower candidate; idle below [`IDLE_DROP`] for [`HOLD`]
/// calls tries the next wider one. A trial is kept unless its delivered Spark
/// period per row (busy plus idle) exceeds the previous depth's by [`KEEP`];
/// a rejected depth sits out [`COOLDOWN`] calls. Pinned controllers never
/// move (explicit PREFILL_WAVES, benchmarks, goldens).
#[derive(Debug, Clone, PartialEq)]
pub struct WaveControl {
    k: u64,
    unit_cap: u64,
    pinned: bool,
    candidates: Vec<u64>,
    phase: Phase,
    depths: [Depth; MAX_WAVES as usize],
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
        Self { k, unit_cap, pinned, candidates, phase: Phase::Steady { quiet: 0 }, depths: Default::default() }
    }
    /// The wave count of the next prefill.
    pub fn waves(&self) -> u64 { self.k }
    /// Its unit inside the admitted extent.
    pub fn unit_rows(&self) -> u64 { unit_for(self.k, self.unit_cap) }
    pub fn pinned(&self) -> bool { self.pinned }
    pub fn candidates(&self) -> &[u64] { &self.candidates }

    /// The candidate next to `k`, deeper (narrower units) or wider.
    fn neighbour(&self, k: u64, deeper: bool) -> Option<u64> {
        let index = self.candidates.iter().position(|&c| c == k)?;
        if deeper { self.candidates.get(index + 1).copied() } else { index.checked_sub(1).map(|i| self.candidates[i]) }
    }
    fn ready(&self, k: u64) -> bool { self.depths[(k - 1) as usize].cooldown == 0 }

    /// One finished prefill that ran `waves` deep (fewer when the prompt was
    /// short) reports its Spark pace; only calls at the chosen depth count.
    /// Returns the next prefill's depth.
    pub fn observe(&mut self, waves: u64, pace: &PrefillPace) -> Decision {
        let (period, idle) = match pace.summary() {
            Some(sample) if !self.pinned && waves == self.k => sample,
            _ => return self.decision(None, None, if self.pinned { "pinned" } else { "not full depth" }),
        };
        for depth in self.depths.iter_mut() { depth.cooldown = depth.cooldown.saturating_sub(1); }
        let k = self.k;
        let trial = matches!(self.phase, Phase::Trial { .. });
        let entry = &mut self.depths[(k - 1) as usize];
        // A trial's first sample replaces any older mean of its depth.
        entry.period = Some(match entry.period { Some(old) if !trial => old + WEIGHT * (period - old), _ => period });
        let reason = match self.phase {
            Phase::Trial { from } => {
                self.phase = Phase::Steady { quiet: 0 };
                if self.depths[(from - 1) as usize].period.is_some_and(|old| period > KEEP * old) {
                    self.depths[(k - 1) as usize].cooldown = COOLDOWN;
                    self.k = from;
                    "trial regressed: revert"
                } else { "trial kept" }
            }
            Phase::Steady { quiet } => {
                let quiet = if idle < IDLE_DROP { quiet + 1 } else { 0 };
                let deeper = self.neighbour(k, true).filter(|&d| idle > IDLE_ADD && self.ready(d));
                let wider = self.neighbour(k, false).filter(|&w| quiet >= HOLD && self.ready(w));
                if let Some(d) = deeper {
                    self.phase = Phase::Trial { from: k };
                    self.k = d;
                    "Sparks starve: one more, narrower wave"
                } else if let Some(w) = wider {
                    self.phase = Phase::Trial { from: k };
                    self.k = w;
                    "no Spark idle: one fewer, wider wave"
                } else {
                    self.phase = Phase::Steady { quiet };
                    "hold"
                }
            }
        };
        self.decision(Some(period), Some(idle), reason)
    }

    fn decision(&self, period: Option<f64>, idle: Option<f64>, reason: &'static str) -> Decision {
        Decision { waves: self.k, unit_rows: self.unit_rows(), period_us_per_row: period, idle_share: idle, reason }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One rank of a k-deep pipeline: busy `s(unit)` per unit (input copies
    /// included); the coordinator returns a unit `t` after its wave lands and a
    /// fixed per-unit gap `g` remains (exchange/launch), so each unit after the
    /// first idles max(g, t - (k-1)·busy).
    fn pace(k: u64, unit: u64, rows: u64, s: impl Fn(u64) -> f64, t: f64, g: f64) -> PrefillPace {
        let busy = s(unit);
        let idle = (t - (k as f64 - 1.0) * busy).max(g);
        let counted = rows.div_ceil(unit) - 1;
        PrefillPace { rows: rows - unit, ranks: vec![((busy * counted as f64) as u64, (idle * counted as f64) as u64); 6] }
    }

    fn run(control: &mut WaveControl, calls: usize, s: impl Fn(u64) -> f64 + Copy, t: f64, g: f64) -> Vec<u64> {
        (0..calls).map(|_| {
            let k = control.waves();
            control.observe(k, &pace(k, control.unit_rows(), 65536, s, t, g)).waves
        }).collect()
    }

    /// Step FP8 max oracle (#762): full-unit compute 4096 rows 20.9 ms, 2752
    /// 16.6 ms, 2048 8.38 ms (the grouped-FP8 BF16 2048→2304 cliff); 4×2048
    /// delivered best. Turnaround bounded by 3×2048's ~1.1 ms gaps.
    fn step_s(unit: u64) -> f64 {
        match unit { 4096 => 20_943.0, 2752 => 16_577.0, _ => 8_383.0 * unit as f64 / 2048.0 }
    }
    const STEP_T: f64 = 17_000.0;

    #[test]
    fn step_holds_its_measured_four_by_2048_against_wider_trials() {
        for cap in [2048, 2752, 4096] {
            let mut control = WaveControl::new(4, 4, cap, false);
            let trace = run(&mut control, 400, step_s, STEP_T, 0.0);
            let off = trace.iter().filter(|&&k| k != 4).count();
            assert!(off <= 400 / COOLDOWN as usize + 1, "cap {cap}: {off} calls off 4×2048");
            assert_eq!(control.waves(), 4);
        }
    }

    #[test]
    fn step_starving_at_three_by_2752_moves_to_four_by_2048() {
        // 3×2752's measured inter-unit gap (~1.5 ms of 16.6, #762 oracle) is
        // above the add threshold; the narrower trial is far cheaper per row.
        let mut control = WaveControl::new(3, 4, 2752, false);
        let trace = run(&mut control, 20, step_s, STEP_T, 1_500.0);
        assert_eq!(trace[0], 4, "{trace:?}");
        assert!(trace.iter().all(|&k| k == 4), "{trace:?}");
    }

    #[test]
    fn glm_holds_three_by_2752() {
        // GLM K4 six Sparks: S ≈ 13 ms per 2752 unit, T ≈ 19.5 ms hidden by
        // 3 units; 2048 rows cost more per row (#762 row-band audit).
        let s = |unit: u64| if unit == 2048 { 3.674 * 2048.0 * 13_000.0 / (3.448 * 2752.0) }
            else { 13_000.0 * unit as f64 / 2752.0 };
        let mut control = WaveControl::new(3, 4, 2752, false);
        assert!(run(&mut control, 200, s, 19_500.0, 0.0).iter().all(|&k| k == 3));
        // A starving depth whose narrower trial costs 1.5× per row (a kernel
        // cliff) tries it once, finds the period worse, and reverts.
        let steep = |unit: u64| if unit == 2048 { 1.5 * 13_000.0 * 2048.0 / 2752.0 } else { 13_000.0 * unit as f64 / 2752.0 };
        let mut control = WaveControl::new(3, 4, 2752, false);
        let trace = run(&mut control, 10, steep, 30_000.0, 0.0);
        assert_eq!(&trace[..2], [4, 3], "{trace:?}");
        assert!(trace[2..].iter().all(|&k| k == 3), "{trace:?}");
    }

    #[test]
    fn a_wider_unit_is_tried_when_idle_is_zero_and_kept_when_cheaper() {
        // Smooth curve (per-row cost falls with width) and T hidden at 2×4096.
        let s = |unit: u64| 1_000.0 + 3.0 * unit as f64;
        let mut control = WaveControl::new(4, 4, 4096, false);
        let trace = run(&mut control, 12, s, 5_000.0, 0.0);
        assert_eq!(control.waves(), 2, "{trace:?}");
    }

    #[test]
    fn short_prompts_and_pins_never_move_the_depth() {
        let mut control = WaveControl::new(3, 4, 2752, false);
        let starving = pace(3, 2752, 65536, |_| 5_000.0, 50_000.0, 0.0);
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
    fn summary_uses_the_busiest_rank_and_the_slowest_wall() {
        let pace = PrefillPace { rows: 100, ranks: vec![(900, 100), (950, 0), (500, 600)] };
        let (period, idle) = pace.summary().unwrap();
        assert!((period - 11.0).abs() < 1e-9);
        assert_eq!(idle, 0.0);
        assert!(PrefillPace { rows: 0, ranks: vec![(1, 1)] }.summary().is_none());
    }
}
