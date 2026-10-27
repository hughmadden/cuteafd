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

/// A trial depth replaces the current one only if its period per row is
/// lower by this share (hysteresis against noise).
pub const MARGIN: f64 = 0.01;

/// The Sparks' measured cost of one prefill call at one depth: per rank, the
/// summed unit busy time (from the unit's arrival, input copies included, to
/// its completion) and the summed idle gap before each unit; each rank's
/// first unit of the call is excluded (its gap is the decode boundary).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrefillPace {
    /// Prompt rows those units carried (per layer wave).
    pub rows: u64,
    /// Per Spark rank: (Σ busy µs, Σ idle µs).
    pub ranks: Vec<(u64, u64)>,
    /// The call's wall time on the host: its start to its last wave's landing (µs).
    pub wall_us: u64,
}

impl PrefillPace {
    /// The critical (busiest) rank's (busy µs per row, idle µs per row).
    pub fn per_row(&self) -> Option<(f64, f64)> {
        if self.rows == 0 { return None; }
        let &(busy, idle) = self.ranks.iter().max_by_key(|(busy, _)| *busy)?;
        (busy > 0).then(|| (busy as f64 / self.rows as f64, idle as f64 / self.rows as f64))
    }
    /// The critical rank's idle share of its time (Sparks waiting on the coordinator).
    pub fn idle_share(&self) -> Option<f64> {
        let (busy, idle) = self.per_row()?;
        Some(idle / (busy + idle))
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
    /// The reported call's depth.
    pub ran: u64,
    /// Its depth's fresh mean overall period µs per prompt row, and the Sparks' idle share.
    pub period_us: Option<f64>,
    pub idle_share: Option<f64>,
    pub reason: &'static str,
}

/// Idle share above which the Sparks wait on the coordinator: the signal
/// points at fewer, wider units.
pub const IDLE_HIGH: f64 = 0.05;
/// Idle share below which the Sparks never wait: the signal points at more units.
pub const IDLE_LOW: f64 = 0.01;
/// A call is a sample when its rows reach this share of the smallest unit.
pub const SAMPLE_SHARE: f64 = 0.6;
/// Recent periods kept per depth.
const HISTORY: usize = 2;
/// A sample expires after this many prefill calls of any size (one global
/// clock): an expired depth has no evidence again.
pub const EXPIRY_CALLS: u64 = 15;

/// The last `HISTORY` periods of one depth with the call that measured each,
/// newest replacing oldest.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Recent { samples: [(f64, u64); HISTORY], count: usize, next: usize }

impl Recent {
    fn push(&mut self, period: f64, call: u64) {
        self.samples[self.next] = (period, call);
        self.next = (self.next + 1) % HISTORY;
        self.count = (self.count + 1).min(HISTORY);
    }
    /// The mean of the samples still fresh at call `now`.
    fn mean(&self, now: u64) -> Option<f64> {
        let fresh = self.samples[..self.count].iter().filter(|(_, at)| now.saturating_sub(*at) < EXPIRY_CALLS);
        let (sum, n) = fresh.fold((0.0, 0), |(sum, n), (period, _)| (sum + period, n + 1));
        (n > 0).then(|| sum / n as f64)
    }
}

/// Chooses each prefill's depth (#762; TJ 2026-10-27). The metric is the
/// call's overall period per prompt row: its host wall time (Spark, idle and
/// coordinator work, fill and drain) over its rows. Every prefill call of at
/// least 60% of the smallest unit is a sample for the depth it ran; each depth
/// keeps only its last two samples, and a sample expires after 15 prefill
/// calls of any size. After each call the controller runs the depth with the
/// best fresh period, except when the Sparks' idle share points at a
/// neighbouring depth without fresh evidence: waiting on the coordinator
/// (above `IDLE_HIGH`) points at fewer, wider units; never waiting (below
/// `IDLE_LOW`) at more units. Then it steps once that way to measure it. No
/// other exploration. Pinned controllers never move (explicit
/// PREFILL_WAVES, benchmarks, goldens).
#[derive(Debug, Clone, PartialEq)]
pub struct WaveControl {
    k: u64,
    unit_cap: u64,
    pinned: bool,
    candidates: Vec<u64>,
    recent: [Recent; MAX_WAVES as usize],
    /// Prefill calls reported so far (every size): the expiry clock.
    calls: u64,
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
        Self { k, unit_cap, pinned, candidates, recent: [Recent::default(); MAX_WAVES as usize], calls: 0 }
    }
    /// The wave count of the next prefill.
    pub fn waves(&self) -> u64 { self.k }
    /// Its unit inside the admitted extent.
    pub fn unit_rows(&self) -> u64 { unit_for(self.k, self.unit_cap) }
    pub fn pinned(&self) -> bool { self.pinned }
    pub fn candidates(&self) -> &[u64] { &self.candidates }

    fn neighbour(&self, k: u64, up: bool) -> Option<u64> {
        let at = self.candidates.iter().position(|&j| j == k)?;
        if up { self.candidates.get(at + 1).copied() } else { at.checked_sub(1).map(|i| self.candidates[i]) }
    }

    /// A finished prefill call of `rows` prompt rows that ran `waves` deep
    /// reports its pace. Returns the next prefill's depth.
    pub fn observe(&mut self, waves: u64, rows: u64, pace: &PrefillPace) -> Decision {
        self.calls += 1;
        let now = self.calls;
        let smallest = self.candidates.iter().map(|&k| unit_for(k, self.unit_cap)).min().unwrap_or(1);
        let sample = !self.pinned && pace.wall_us > 0 && rows as f64 >= SAMPLE_SHARE * smallest as f64
            && self.candidates.contains(&waves);
        let mut decision = Decision { waves: self.k, unit_rows: self.unit_rows(), ran: waves, period_us: None,
            idle_share: pace.idle_share(), reason: if self.pinned { "pinned" } else { "not a sample" } };
        if !sample { return decision; }
        self.recent[(waves - 1) as usize].push(pace.wall_us as f64 / rows as f64, now);
        decision.period_us = self.recent[(waves - 1) as usize].mean(now);
        let idle = pace.idle_share().unwrap_or(0.0);
        // The neighbour the Sparks' wait points at. At the end of the
        // candidates the only neighbour is the other way (idle Sparks at the
        // shallowest depth also mean too few units in flight).
        let direction = if idle > IDLE_HIGH { Some(false) } else if idle < IDLE_LOW { Some(true) } else { None };
        let pointed = direction.and_then(|up| self.neighbour(waves, up).or_else(|| self.neighbour(waves, !up)));
        let reason = match pointed.filter(|&j| self.recent[(j - 1) as usize].mean(now).is_none()) {
            Some(j) => {
                self.k = j;
                if j < waves { "Sparks wait: measure fewer, wider units" } else { "Sparks saturated: measure more units" }
            }
            None => {
                let current = self.recent[(self.k - 1) as usize].mean(now);
                let best = self.candidates.iter().filter_map(|&j| self.recent[(j - 1) as usize].mean(now).map(|p| (j, p)))
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                match (best, current) {
                    (Some((j, p)), Some(c)) if j != self.k && p < (1.0 - MARGIN) * c => {
                        self.k = j;
                        "best fresh period"
                    }
                    (Some((j, _)), None) => { self.k = j; "best fresh period" }
                    _ => "hold",
                }
            }
        };
        decision.waves = self.k;
        decision.unit_rows = self.unit_rows();
        decision.reason = reason;
        decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A k-deep pipeline over a call of `rows` rows: each unit of `u` rows
    /// costs the Sparks B = s(u)·u and the coordinator T = τ(u)·u. With k units
    /// circulating, one unit completes per max(B, T, (T + B) / k); the Sparks
    /// idle the rest of each period, and the call adds one fill.
    fn pace(k: u64, u: u64, rows: u64, s: f64, tau: f64) -> PrefillPace {
        let (b, t) = (s * u as f64, tau * u as f64);
        let period = b.max(t).max((t + b) / k as f64);
        let units = rows.div_ceil(u) as f64;
        let wall = units * period + (t + b - period).max(0.0);
        PrefillPace { rows, ranks: vec![((units * b) as u64, (units * (period - b)) as u64); 6], wall_us: wall as u64 }
    }

    /// Runs calls of `rows` rows through the controller; `profile(u, call)`
    /// gives (s, τ). Returns the depth each call ran.
    fn run(control: &mut WaveControl, calls: &[u64], profile: impl Fn(u64, usize) -> (f64, f64)) -> Vec<u64> {
        calls.iter().enumerate().map(|(call, &rows)| {
            let k = control.waves();
            let unit = control.unit_rows();
            let (s, tau) = profile(unit, call);
            control.observe(k, rows, &pace(k, unit, rows, s, tau));
            k
        }).collect()
    }

    /// One 8K prompt per call.
    fn short(calls: usize) -> Vec<u64> { vec![8192; calls] }

    /// The cards' GLM K4 / Kimi BF16 figures (rw762-cards-a, per layer-wave
    /// µs per row: Spark busy, and a coordinator turnaround reproducing each
    /// depth's measured period and Spark idle) and Step FP8's A8-wire unit
    /// runs (#802: 3 x 2752 delivered 2720 tok/s at 8K, 4 x 2048 2483,
    /// 2 x 4096 2347): 3 x 2752 is best.
    fn measured(family: &str) -> impl Fn(u64, usize) -> (f64, f64) + Copy {
        let family = match family { "glm" => 0, "kimi" => 1, _ => 2 };
        move |unit: u64, _| match (family, unit) {
            (0, 4096) => (3.36, 4.90), (0, 2752) => (3.71, 4.00), (0, _) => (3.90, 4.34),
            (1, 4096) => (4.45, 6.19), (1, 2752) => (4.97, 5.90), (1, _) => (5.40, 6.65),
            (_, 4096) => (350.0, 426.1), (_, 2752) => (367.6, 300.0), (_, _) => (380.0, 402.7),
        }
    }

    /// The superseded BF16-input Step kernel: 2048-row units cost far less
    /// per row (GEMV) than wider ones; the Sparks are the bottleneck.
    fn cliff(unit: u64) -> (f64, f64) { (match unit { 4096 => 5.1, 2752 => 6.0, _ => 4.1 }, 2.0) }
    /// Fixed per-unit coordinator cost: wider units cost it less per row.
    fn bound(unit: u64) -> (f64, f64) { (1.5, 1.0 + 4_000.0 / unit as f64) }

    #[test]
    fn measured_glm_kimi_and_step_figures_settle_on_three_by_2752() {
        for family in ["glm", "kimi", "step"] {
            for start in 2..=4 {
                let mut control = WaveControl::new(start, 4, 4096, false);
                let trace = run(&mut control, &short(60), measured(family));
                // One call per depth to measure it, then 3 x 2752 holds, leaving
                // it about once per expiry period to re-measure a neighbour.
                let away = trace[4..].iter().filter(|&&k| k != 3).count();
                assert!(away <= trace.len() / EXPIRY_CALLS as usize + 1, "{family} start {start}: {trace:?}");
                assert_eq!(control.waves(), 3, "{family} start {start}: {trace:?}");
            }
        }
    }

    #[test]
    fn a_kernel_row_cliff_settles_on_four_by_2048() {
        for start in 2..=4 {
            let mut control = WaveControl::new(start, 4, 4096, false);
            let trace = run(&mut control, &short(40), |unit, _| cliff(unit));
            assert_eq!(control.waves(), 4, "start {start}: {trace:?}");
        }
    }

    #[test]
    fn a_coordinator_bound_pipeline_steps_down_to_two() {
        let mut control = WaveControl::new(4, 4, 4096, false);
        let trace = run(&mut control, &short(30), |unit, _| bound(unit));
        assert_eq!(control.waves(), 2, "{trace:?}");
        assert!(trace.iter().position(|&k| k == 2).is_some_and(|i| i <= 3), "{trace:?}");
    }

    #[test]
    fn a_workload_shift_reconverges_within_an_expiry_period() {
        // First a Spark-bound workload with a 2048 cliff (best 4), then a
        // coordinator-bound one (best 2): the old evidence expires and the
        // Sparks' wait leads the controller down within ~15 calls.
        let shift = 30;
        let profile = |unit: u64, call: usize| if call < shift { cliff(unit) } else { bound(unit) };
        let mut control = WaveControl::new(3, 4, 4096, false);
        let trace = run(&mut control, &short(shift + 2 * EXPIRY_CALLS as usize), profile);
        assert_eq!(trace[shift - 1], 4, "{trace:?}");
        let first = trace[shift..].iter().position(|&k| k == 2).expect("reaches 2") ;
        assert!(first <= EXPIRY_CALLS as usize, "{trace:?}");
        // Then 2 holds, leaving it about once per expiry period to re-measure.
        let away = trace[shift + first..].iter().filter(|&&k| k != 2).count();
        assert!(away <= 2, "{trace:?}");
    }

    #[test]
    fn fresh_evidence_is_not_re_explored_when_the_signal_agrees() {
        // GLM: 3 x 2752 leaves the Sparks idle 7% of the time (they point at
        // fewer, wider units), but 2 x 4096 measured worse: while that
        // evidence is fresh the controller holds 3 and does not step to 2.
        let mut control = WaveControl::new(3, 4, 4096, false);
        let trace = run(&mut control, &short(EXPIRY_CALLS as usize), measured("glm"));
        assert_eq!(trace.iter().filter(|&&k| k == 2).count(), 1, "{trace:?}");
    }

    #[test]
    fn expired_evidence_is_re_measured_only_when_the_signal_points_at_it() {
        // Once 2 x 4096's samples expire, the Sparks' wait (pointing at fewer,
        // wider units) re-measures it: about once per expiry period.
        let mut control = WaveControl::new(3, 4, 4096, false);
        let trace = run(&mut control, &short(5 * EXPIRY_CALLS as usize), measured("glm"));
        let visits = trace.iter().filter(|&&k| k == 2).count();
        assert!((4..=6).contains(&visits), "{visits} visits: {trace:?}");
        // Calls too small to be samples still advance the clock.
        let mut control = WaveControl::new(3, 4, 4096, false);
        let mut calls = short(3);
        calls.extend(vec![500u64; EXPIRY_CALLS as usize]);
        calls.extend(short(2));
        let trace = run(&mut control, &calls, measured("glm"));
        assert_eq!(trace.last(), Some(&2), "the expired neighbour is re-measured: {trace:?}");
        // Without a signal (idle between the thresholds) nothing is re-measured.
        let quiet = |unit: u64, _| match unit { 4096 => (3.36, 3.36), 2752 => (3.0, 3.06), _ => (3.9, 3.9) };
        let mut control = WaveControl::new(3, 4, 4096, false);
        let trace = run(&mut control, &short(5 * EXPIRY_CALLS as usize), quiet);
        assert!(trace.iter().all(|&k| k == 3), "{trace:?}");
    }

    #[test]
    fn small_calls_and_pins_never_move_the_depth() {
        let mut control = WaveControl::new(4, 4, 4096, false);
        // Calls below 60% of the smallest unit (2048) are not samples.
        let trace = run(&mut control, &vec![1000; 20], |unit, _| bound(unit));
        assert!(trace.iter().all(|&k| k == 4), "{trace:?}");
        let mut pinned = WaveControl::new(2, 4, 4096, true);
        let trace = run(&mut pinned, &short(20), measured("glm"));
        assert!(trace.iter().all(|&k| k == 2), "{trace:?}");
        assert_eq!(control.observe(4, 8192, &PrefillPace::default()).waves, 4);
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
        let pace = PrefillPace { rows: 100, ranks: vec![(900, 100), (950, 0), (500, 600)], wall_us: 1 };
        assert_eq!(pace.per_row(), Some((9.5, 0.0)));
        assert!(PrefillPace { rows: 0, ranks: vec![(1, 1)], wall_us: 1 }.per_row().is_none());
        assert_eq!(PrefillPace { rows: 10, ranks: vec![(90, 10)], wall_us: 1 }.idle_share(), Some(0.1));
    }
}
