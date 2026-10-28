//! Runtime prefill wave adaptation (#762 stages 2–4). The planner admits a
//! maximum of `slots` in-flight units and a unit extent; before each prefill
//! call (one window cycle of at most [`call_rows`] rows) the executor asks
//! [`WaveControl`] for the call's depth k′ within that admission, and every
//! finished call reports its pace ([`PrefillPace`]): its wall time, the
//! Sparks' measured busy/idle time on their prefill owner, and the time the
//! coordinator's consuming stream waited on Spark replies. Pure arithmetic
//! over the reported counters; no clock, no I/O.

use super::prefill_waves::{unit_rows_in, MAX_WAVES, RESERVE_ROWS};

/// A depth replaces the current one only if its ratio is lower by this
/// share (hysteresis against noise).
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
    /// The call's own wall time on the host (µs): from its start, or the
    /// previous call's last landing when later, to its last wave's landing.
    pub wall_us: u64,
    /// GPU time the coordinator's consuming stream waited on the call's
    /// Spark replies (µs; None when not measured).
    pub reply_wait_us: Option<u64>,
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
    /// The share of the call's wall time the coordinator waited on Spark replies.
    pub fn reply_wait_share(&self) -> Option<f64> {
        (self.wall_us > 0).then_some(())?;
        Some(self.reply_wait_us? as f64 / self.wall_us as f64)
    }
    /// Where the waits point (latest sample): `Some(false)` at fewer, wider
    /// units (the Sparks wait on the coordinator), `Some(true)` at more units
    /// (the coordinator waits on the Sparks); the larger share wins.
    /// When both fire, the larger must beat the other by [`SIGNAL_FACTOR`]
    /// or [`SIGNAL_GAP`]; a tie points nowhere.
    pub fn direction(&self) -> Option<bool> {
        let idle = self.idle_share().unwrap_or(0.0);
        let reply = self.reply_wait_share().unwrap_or(0.0);
        let beats = |a: f64, b: f64| a >= SIGNAL_FACTOR * b || a - b >= SIGNAL_GAP;
        match (idle > IDLE_HIGH, reply > REPLY_WAIT_HIGH) {
            (true, true) if beats(reply, idle) => Some(true),
            (true, true) if beats(idle, reply) => Some(false),
            (true, true) => None,
            (true, false) => Some(false),
            (false, true) => Some(true),
            (false, false) => None,
        }
    }
}

/// The unit a depth runs at inside the admitted extent `unit_cap`: the
/// standard reserve's unit for `k`, never wider than what was admitted
/// ([`unit_for_in`] at [`RESERVE_ROWS`]).
pub fn unit_for(k: u64, unit_cap: u64) -> u64 { unit_for_in(k, unit_cap, RESERVE_ROWS) }

/// The unit depth `k` runs at in a `reserve` of rows (TJ's reserve rule:
/// `roundup64(reserve / k)`, depths 1 and 2 both at half), never wider than
/// the admitted `unit_cap`. A card that cannot hold the standard reserve
/// admits a smaller one; its depths cut that reserve.
pub fn unit_for_in(k: u64, unit_cap: u64, reserve: u64) -> u64 { unit_rows_in(k, reserve).min(unit_cap) }

/// Rows every per-slot plane spans so any depth up to `slots` fits as views
/// at `slot * unit_for(k)`: the largest `k * unit_for(k)` (8256 at most).
pub fn reserve_rows(slots: u64, unit_cap: u64) -> u64 { reserve_rows_in(slots, unit_cap, RESERVE_ROWS) }
/// [`reserve_rows`] in a `reserve` of rows.
pub fn reserve_rows_in(slots: u64, unit_cap: u64, reserve: u64) -> u64 {
    (1..=slots.clamp(1, MAX_WAVES)).map(|k| k * unit_for_in(k, unit_cap, reserve)).max().unwrap_or(0)
}

/// Rows one prefill call takes so that every candidate depth cuts it into
/// whole units: the smallest `k * unit_for(k)` over [`candidates`] (8192 for
/// four slots of 2752: 3 x 2752 cuts it 2752/2752/2688, 4 x 2048 exactly),
/// never the reserve itself (8256 at 4 x 2048 would cut eight 1088-row units).
pub fn call_rows(slots: u64, unit_cap: u64) -> u64 { call_rows_in(slots, unit_cap, RESERVE_ROWS) }
/// [`call_rows`] in a `reserve` of rows.
pub fn call_rows_in(slots: u64, unit_cap: u64, reserve: u64) -> u64 {
    candidates_in(slots, unit_cap, reserve).into_iter().map(|k| k * unit_for_in(k, unit_cap, reserve)).min().unwrap_or(0)
}

/// The depths worth running under `slots` and `unit_cap`: for each distinct
/// unit, its deepest depth (more units in flight at the same unit only hide
/// more turnaround at the same Spark cost per row). Ascending.
pub fn candidates(slots: u64, unit_cap: u64) -> Vec<u64> { candidates_in(slots, unit_cap, RESERVE_ROWS) }
/// [`candidates`] in a `reserve` of rows.
pub fn candidates_in(slots: u64, unit_cap: u64, reserve: u64) -> Vec<u64> {
    let max = slots.clamp(1, MAX_WAVES);
    (1..=max).filter(|&k| k == max || unit_for_in(k + 1, unit_cap, reserve) != unit_for_in(k, unit_cap, reserve)).collect()
}

/// Where a call starts. Fewer, wider units (a lower depth) only at an
/// opening call, where nothing is in flight; more units at any call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Boundary {
    /// A prefill's first call (the decode→prefill transition).
    Opening,
    /// A call continuing the previous call's sequence where it ended.
    Running,
}

/// One call's decision (logged by the executor).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    /// The call's id on the controller's clock.
    pub call: u64,
    pub waves: u64,
    pub unit_rows: u64,
    pub boundary: Boundary,
    /// A reference call: an opening at row 0 of at least [`REFERENCE_SHARE`]
    /// of a full call. Only reference calls are evidence.
    pub reference: bool,
    /// A running call whose depth differs from the previous call's.
    pub transition: bool,
    pub reason: &'static str,
}

/// A reported call (logged by the executor).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub call: u64,
    pub waves: u64,
    pub rows: u64,
    pub boundary: Boundary,
    /// Whether it counts as evidence (a reference call that ran its depth in full).
    pub counted: bool,
    pub period_us: Option<f64>,
    pub idle_share: Option<f64>,
    pub reply_wait_share: Option<f64>,
    /// Its period over the current depth's band mean (a trial's measured ratio).
    pub ratio: Option<f64>,
}

/// Spark idle share above which the Sparks wait on the coordinator: the
/// signal points at fewer, wider units.
pub const IDLE_HIGH: f64 = 0.05;
/// Coordinator reply-wait share above which it waits on the Sparks: the
/// signal points at more units.
pub const REPLY_WAIT_HIGH: f64 = 0.05;
/// When both waits fire, the larger must exceed the smaller by this factor
/// or by [`SIGNAL_GAP`] to pick a direction; otherwise it is a tie.
pub const SIGNAL_FACTOR: f64 = 1.5;
pub const SIGNAL_GAP: f64 = 0.10;
/// A reference call is an opening at row 0 with at least this share of a full call's rows.
pub const REFERENCE_SHARE: f64 = 0.9;
/// Reference samples of the current depth kept for its noise band.
pub const BAND_SAMPLES: usize = 4;
/// The band: `BAND_SPREAD` times the samples' spread, within [`BAND_MIN`, `BAND_MAX`].
pub const BAND_MIN: f64 = 0.015;
pub const BAND_MAX: f64 = 0.05;
pub const BAND_SPREAD: f64 = 1.5;
/// Consecutive same-direction out-of-band reference calls that mean the
/// current depth's period changed.
pub const CHANGE_CALLS: usize = 2;
/// Backstop: reference calls at the current depth before its neighbours are
/// re-tested without a change; doubles after each re-test that confirms the
/// old ratios, resets on any change.
pub const BACKSTOP_FIRST: u64 = 64;
pub const BACKSTOP_MAX: u64 = 1 << 16;
/// A dispatched trial unreported after this many calls is forgotten.
const STALE_CALLS: u64 = 16;

/// The noise band (a share of the mean) of a depth's recent reference periods.
fn band_width(samples: &[f64]) -> f64 {
    if samples.len() < 2 { return BAND_MIN; }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let (lo, hi) = samples.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &s| (lo.min(s), hi.max(s)));
    (BAND_SPREAD * (hi - lo) / mean).clamp(BAND_MIN, BAND_MAX)
}

/// The controller's persisted evidence (#762): every depth's period as a
/// ratio to the current depth's, the current depth's band samples and the
/// backstop. Versioned by the caller's cache key.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Evidence {
    /// The current depth.
    pub waves: u64,
    /// (depth, period ratio to the current depth), the current depth at 1.
    pub ratios: Vec<(u64, f64)>,
    /// The current depth's recent reference periods (µs per row), oldest first.
    pub band: Vec<f64>,
    /// Reference calls at the current depth since its neighbours were last measured.
    pub since_test: u64,
    /// Reference calls the backstop waits before re-testing the neighbours.
    pub backstop: u64,
}

/// Chooses each prefill call's depth from relative evidence (#762 round F;
/// TJ 2026-10-27: measured real time per row; a direction signal only where
/// there is no evidence; downsize at any call, upsize only at an opening).
///
/// - **Evidence** is each depth's period as a ratio to the current depth's,
///   measured at matched position: only *reference calls* (an opening at row
///   0 of at least 90% of a full call) count, so a trial and the current
///   depth's band are the same shape. Ratios don't drift with context or
///   power; absolutes do.
/// - **Calibration** (before readiness, or loaded from the persisted
///   evidence) measures every candidate once, so every depth has evidence.
/// - **Re-test on change, no expiry:** the current depth keeps its last four
///   reference periods; two consecutive reference calls outside their noise
///   band, in the same direction, re-base the band and re-test each
///   neighbour once (one per reference call). A backstop re-tests after
///   [`BACKSTOP_FIRST`] quiet reference calls, doubling after each re-test
///   that confirms the old ratios.
/// - **Choice:** each call runs the depth with the lowest ratio among those
///   its boundary allows, with 1% hysteresis.
/// - **Signals** (Spark idle → fewer, wider units; coordinator reply-wait →
///   more units) only trigger a trial toward a neighbour with no evidence,
///   and only when one beats the other by a margin; a tie means no trial.
///
/// Pinned controllers never move (explicit PREFILL_WAVES, benchmarks, goldens).
#[derive(Debug, Clone, PartialEq)]
pub struct WaveControl {
    k: u64,
    unit_cap: u64,
    /// The reserve every depth's units are cut from ([`unit_for_in`]).
    reserve: u64,
    pinned: bool,
    candidates: Vec<u64>,
    /// Per depth: its period over the current depth's (the current at 1).
    ratios: [Option<f64>; MAX_WAVES as usize],
    /// The current depth's recent reference periods, oldest first.
    band: Vec<f64>,
    /// Consecutive out-of-band reference periods at the current depth (same direction).
    outside: Vec<f64>,
    /// Depths to measure at the next reference calls, in order.
    trials: Vec<u64>,
    /// Trials dispatched whose record has not arrived: (call, depth).
    in_flight: Vec<(u64, u64)>,
    /// Ratios before the running re-test round (to judge confirmation), and
    /// whether the round was triggered by a change.
    round: Option<([Option<f64>; MAX_WAVES as usize], bool)>,
    since_test: u64,
    backstop: u64,
    /// Calibration in progress: the remaining forced depths (warm first).
    calibrating: Option<Vec<(u64, bool)>>,
    /// The latest reference sample's direction at the current depth.
    direction: Option<bool>,
    /// The last call's depth: a running call may only deepen from it.
    last: u64,
    /// The calibration warm call (never evidence).
    warm: Option<u64>,
    /// Evidence changed since the last [`Self::evidence`] read.
    dirty: bool,
    /// Prefill calls started so far.
    calls: u64,
}

impl WaveControl {
    /// Start at the planner's estimate `initial`, moved to the deepest depth
    /// that runs the same unit.
    pub fn new(initial: u64, slots: u64, unit_cap: u64, pinned: bool) -> Self {
        Self::in_reserve(initial, slots, unit_cap, RESERVE_ROWS, pinned)
    }
    /// [`Self::new`] for depths cut from a `reserve` of rows ([`unit_for_in`]).
    pub fn in_reserve(initial: u64, slots: u64, unit_cap: u64, reserve: u64, pinned: bool) -> Self {
        let candidates = candidates_in(slots, unit_cap, reserve);
        let max = *candidates.last().expect("at least one depth");
        let initial = initial.clamp(1, max);
        let k = if pinned { initial } else {
            *candidates.iter().find(|&&k| k >= initial).expect("max is a candidate")
        };
        let mut ratios = [None; MAX_WAVES as usize];
        ratios[(k - 1) as usize] = Some(1.0);
        Self { k, unit_cap, reserve, pinned, candidates, ratios, band: Vec::new(), outside: Vec::new(), trials: Vec::new(),
            in_flight: Vec::new(), round: None, since_test: 0, backstop: BACKSTOP_FIRST, calibrating: None,
            direction: None, last: k, warm: None, dirty: false, calls: 0 }
    }
    /// The wave count of the next prefill call.
    pub fn waves(&self) -> u64 { self.k }
    /// Its unit inside the admitted extent.
    pub fn unit_rows(&self) -> u64 { unit_for_in(self.k, self.unit_cap, self.reserve) }
    /// The reserve depths are cut from.
    pub fn reserve(&self) -> u64 { self.reserve }
    pub fn pinned(&self) -> bool { self.pinned }
    pub fn candidates(&self) -> &[u64] { &self.candidates }
    /// Every candidate's ratio to the current depth (None: no evidence).
    pub fn ratios(&self) -> Vec<(u64, Option<f64>)> {
        self.candidates.iter().map(|&j| (j, self.ratios[(j - 1) as usize])).collect()
    }
    /// Whether the controller can move at all (unpinned, more than one candidate).
    pub fn adaptive(&self) -> bool { !self.pinned && self.candidates.len() > 1 }
    /// Whether every candidate has evidence.
    pub fn calibrated(&self) -> bool { self.candidates.iter().all(|&j| self.ratios[(j - 1) as usize].is_some()) }

    /// Starts calibration: the next calls run one warm call at the current
    /// depth (not evidence), then one reference call at each candidate, the
    /// current depth first. The caller makes each a reference call.
    pub fn calibrate(&mut self) -> usize {
        if !self.adaptive() { return 0; }
        let mut plan = vec![(self.k, false), (self.k, true)];
        plan.extend(self.candidates.iter().filter(|&&j| j != self.k).map(|&j| (j, true)));
        let calls = plan.len();
        self.calibrating = Some(plan);
        calls
    }
    /// Ends calibration once every call reported: runs the best measured depth.
    pub fn finish_calibration(&mut self) -> &'static str {
        // Calibration calls still in flight stay trials: a late pace records
        // (or, after a switch, re-queues) its depth.
        self.calibrating = None;
        self.trials.clear();
        self.round = None;
        self.since_test = 0;
        match self.best(Boundary::Opening) {
            Some(j) if j != self.k => { self.switch(j); "calibrated: switched" }
            _ => "calibrated: held",
        }
    }

    /// The persisted evidence, and whether it changed since the last read.
    pub fn evidence(&mut self) -> (Evidence, bool) {
        let dirty = std::mem::take(&mut self.dirty);
        (Evidence { waves: self.k, ratios: self.candidates.iter()
            .filter_map(|&j| self.ratios[(j - 1) as usize].map(|r| (j, r))).collect(),
            band: self.band.clone(), since_test: self.since_test, backstop: self.backstop }, dirty)
    }
    /// Restores persisted evidence; refuses evidence that does not fit these candidates.
    pub fn restore(&mut self, evidence: &Evidence) -> Result<(), &'static str> {
        if self.pinned { return Err("pinned"); }
        if !self.candidates.contains(&evidence.waves) { return Err("current depth is not a candidate"); }
        let mut ratios = [None; MAX_WAVES as usize];
        for &(j, r) in &evidence.ratios {
            if !self.candidates.contains(&j) || !(r.is_finite() && r > 0.0) { return Err("ratio outside the candidates"); }
            ratios[(j - 1) as usize] = Some(r);
        }
        if ratios[(evidence.waves - 1) as usize].is_none_or(|r| (r - 1.0).abs() > 1e-9) {
            return Err("current depth's ratio is not 1");
        }
        if evidence.band.len() > BAND_SAMPLES || evidence.band.iter().any(|p| !(p.is_finite() && *p > 0.0))
            || !(BACKSTOP_FIRST..=BACKSTOP_MAX).contains(&evidence.backstop) {
            return Err("band or backstop out of range");
        }
        self.k = evidence.waves;
        self.last = evidence.waves;
        self.ratios = ratios;
        self.band = evidence.band.clone();
        self.since_test = evidence.since_test.min(evidence.backstop);
        self.backstop = evidence.backstop;
        self.outside.clear();
        self.trials.clear();
        Ok(())
    }

    fn neighbour(&self, k: u64, up: bool) -> Option<u64> {
        let at = self.candidates.iter().position(|&j| j == k)?;
        if up { self.candidates.get(at + 1).copied() } else { at.checked_sub(1).map(|i| self.candidates[i]) }
    }
    fn ratio(&self, j: u64) -> Option<f64> { self.ratios[(j - 1) as usize] }
    /// The shallowest depth a call at `boundary` may run: any at an
    /// opening, at least the previous call's at a running call.
    fn floor(&self, boundary: Boundary) -> u64 { if boundary == Boundary::Opening { 1 } else { self.last } }

    /// The allowed depth with the lowest ratio, if it beats the current by MARGIN.
    fn best(&self, boundary: Boundary) -> Option<u64> {
        let floor = self.floor(boundary);
        let (j, r) = self.candidates.iter().filter(|&&j| j >= floor)
            .filter_map(|&j| self.ratio(j).map(|r| (j, r))).min_by(|a, b| a.1.total_cmp(&b.1))?;
        (j != self.k && r < 1.0 - MARGIN).then_some(j)
    }

    /// Makes `j` the current depth: ratios re-based on it, a fresh band.
    fn switch(&mut self, j: u64) {
        if let Some(base) = self.ratio(j) {
            for r in self.ratios.iter_mut().flatten() { *r /= base; }
        }
        self.ratios[(j - 1) as usize] = Some(1.0);
        self.k = j;
        self.band.clear();
        self.outside.clear();
        self.direction = None;
        self.dirty = true;
    }

    /// A call starts: `boundary` and whether it is a reference call (an
    /// opening at row 0 with at least 90% of a full call's rows). Returns
    /// its depth and id.
    pub fn next(&mut self, boundary: Boundary, reference: bool) -> Decision {
        self.calls += 1;
        let now = self.calls;
        let reference = reference && boundary == Boundary::Opening;
        let (waves, reason) = if self.pinned { (self.k, "pinned") } else { self.decide(boundary, reference, now) };
        let previous = std::mem::replace(&mut self.last, waves);
        Decision { call: now, waves, unit_rows: unit_for_in(waves, self.unit_cap, self.reserve), boundary, reference,
            transition: boundary == Boundary::Running && waves != previous, reason }
    }

    fn decide(&mut self, boundary: Boundary, reference: bool, now: u64) -> (u64, &'static str) {
        // A trial whose pace never arrived (dropped by the executor) is forgotten.
        self.in_flight.retain(|&(call, _)| now.saturating_sub(call) <= STALE_CALLS);
        if let Some(plan) = &mut self.calibrating {
            // Calibration calls run their forced depth (the caller makes them reference calls).
            if !plan.is_empty() {
                let (j, counted) = plan.remove(0);
                if counted { if j != self.k { self.in_flight.push((now, j)); } } else { self.warm = Some(now); }
                return (j, if counted { "calibrate" } else { "calibrate: warm" });
            }
            return (self.k, "calibrate: done");
        }
        // A pending trial runs at the next reference call (an opening: both
        // directions allowed), once the current depth has a band to compare to.
        if reference && !self.band.is_empty() {
            while let Some(j) = self.trials.first().copied() {
                self.trials.remove(0);
                if j != self.k && self.candidates.contains(&j) {
                    self.in_flight.push((now, j));
                    return (j, "re-test");
                }
            }
            // A signal only toward a neighbour with no evidence, and only one
            // that beats the other wait by the margin (a tie: no trial).
            if let Some(j) = self.direction.and_then(|up| self.neighbour(self.k, up))
                .filter(|&j| self.ratio(j).is_none() && !self.in_flight.iter().any(|&(_, d)| d == j)) {
                self.in_flight.push((now, j));
                return (j, if j < self.k { "Sparks wait: measure fewer, wider units" }
                    else { "coordinator waits: measure more units" });
            }
        }
        let floor = self.floor(boundary);
        if self.k < floor {
            // A running call after a deeper trial: the prompt stays at least
            // that deep (fewer units only at an opening); the best such depth.
            let deeper = self.candidates.iter().filter(|&&j| j >= floor)
                .filter_map(|&j| self.ratio(j).map(|r| (j, r))).min_by(|a, b| a.1.total_cmp(&b.1))
                .map_or(floor, |(j, _)| j);
            return (deeper, "running: no wider than the previous call");
        }
        match self.best(boundary) {
            Some(j) => { self.switch(j); (j, "best ratio") }
            None => (self.k, "hold"),
        }
    }

    /// A finished call reports its pace (in any order, any time after it ran).
    /// `full_rows` is a full call's rows (the reference threshold's base).
    pub fn record(&mut self, decision: &Decision, rows: u64, full_rows: u64, pace: &PrefillPace) -> Sample {
        let trial = self.in_flight.iter().position(|&(call, _)| call == decision.call)
            .map(|at| self.in_flight.remove(at).1);
        let warm = self.warm.is_some_and(|call| call == decision.call);
        let counted = !self.pinned && !warm && decision.reference && pace.wall_us > 0
            && rows as f64 >= REFERENCE_SHARE * full_rows as f64 && self.candidates.contains(&decision.waves);
        let period = (pace.wall_us as f64) / rows.max(1) as f64;
        let mut sample = Sample { call: decision.call, waves: decision.waves, rows, boundary: decision.boundary, counted,
            period_us: counted.then_some(period), idle_share: pace.idle_share(),
            reply_wait_share: pace.reply_wait_share(), ratio: None };
        if !counted { return sample; }
        let j = decision.waves;
        if j == self.k {
            self.direction = pace.direction();
            sample.ratio = Some(self.observe(period));
        } else if !self.band.is_empty() {
            // A trial at matched position: its ratio to the current depth's band mean.
            let ratio = period / (self.band.iter().sum::<f64>() / self.band.len() as f64);
            self.ratios[(j - 1) as usize] = Some(ratio);
            sample.ratio = Some(ratio);
            self.dirty = true;
            if trial.is_some() { self.close_round(); }
        } else if trial.is_some() {
            // No band to compare against yet (the depth just switched): measure it again later.
            self.trials.insert(0, j);
            sample.counted = false;
        }
        sample
    }

    /// A reference period at the current depth: the band, or a change.
    fn observe(&mut self, period: f64) -> f64 {
        let mean = if self.band.is_empty() { period } else { self.band.iter().sum::<f64>() / self.band.len() as f64 };
        let width = band_width(&self.band);
        let side = if period > mean * (1.0 + width) { 1 } else if period < mean * (1.0 - width) { -1 } else { 0 };
        self.dirty = true;
        if self.band.len() >= 2 && side != 0 {
            let same = self.outside.last().is_none_or(|&last| (last > mean) == (side > 0));
            if !same { self.outside.clear(); }
            self.outside.push(period);
            if self.outside.len() >= CHANGE_CALLS {
                // The current depth's period moved: re-base and re-test its neighbours.
                self.band = std::mem::take(&mut self.outside);
                self.start_round(true);
            }
            return period / mean;
        }
        self.outside.clear();
        self.band.push(period);
        if self.band.len() > BAND_SAMPLES { self.band.remove(0); }
        self.since_test += 1;
        if self.since_test >= self.backstop && self.trials.is_empty() && self.round.is_none() {
            self.start_round(false);
        }
        period / mean
    }

    fn start_round(&mut self, changed: bool) {
        self.trials = [false, true].iter().filter_map(|&up| self.neighbour(self.k, up)).collect();
        self.since_test = 0;
        if self.trials.is_empty() { return; }
        self.round = Some((self.ratios, changed));
    }

    /// A re-test reported: once the round's trials are all in, the backstop
    /// doubles if it confirmed the old ratios, else resets.
    fn close_round(&mut self) {
        if !self.trials.is_empty() || !self.in_flight.is_empty() { return; }
        let Some((before, changed)) = self.round.take() else { return };
        let width = band_width(&self.band).max(0.02);
        let confirmed = !changed && self.candidates.iter().all(|&j| match (before[(j - 1) as usize], self.ratio(j)) {
            (Some(old), Some(new)) => (new / old - 1.0).abs() <= width,
            _ => true,
        });
        self.backstop = if confirmed { (self.backstop * 2).min(BACKSTOP_MAX) } else { BACKSTOP_FIRST };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A depth's measured behaviour: (µs per row of a full opening call,
    /// Spark idle share, coordinator reply-wait share, context rise: the
    /// period's growth over a 64K prompt).
    type Profile = (f64, f64, f64, f64);

    /// One call of `rows` rows at context `at`: the full-call period grown
    /// with context; a short call pays more fill per row (GLM's 4875-row
    /// opening ran 325 µs/row against 283 for a full one); `scale`
    /// multiplies the wall (power, noise).
    fn pace(unit: u64, rows: u64, at: u64, (period, idle, reply, rise): Profile, scale: f64) -> PrefillPace {
        let per_row = period * (1.0 + rise * at as f64 / 65536.0);
        let fill = 0.4 * period * unit.min(rows) as f64 * (FULL - rows.min(FULL)) as f64 / FULL as f64;
        let wall = (per_row * rows as f64 + fill) * scale;
        let busy = wall * (1.0 - idle);
        PrefillPace { rows, ranks: vec![(busy as u64, (wall * idle) as u64); 6], wall_us: wall as u64,
            reply_wait_us: Some((wall * reply) as u64) }
    }

    /// One call as run.
    #[derive(Debug, Clone, Copy)]
    struct Ran { k: u64, opening: bool, reference: bool, transition: bool, reason: &'static str }

    const FULL: u64 = 8192;

    /// Runs prompts (each cut into full calls) through the controller, each
    /// call reported before the next starts. `profile(k, prompt)` gives the
    /// depth's behaviour; `scale(call)` the wall factor of the n-th call.
    fn run(control: &mut WaveControl, prompts: &[u64], profile: impl Fn(u64, usize) -> Profile,
        scale: impl Fn(usize) -> f64) -> Vec<Ran> {
        let mut trace = Vec::new();
        for (prompt, &length) in prompts.iter().enumerate() {
            let mut at = 0;
            while at < length {
                let rows = FULL.min(length - at);
                let boundary = if at == 0 { Boundary::Opening } else { Boundary::Running };
                let reference = at == 0 && rows as f64 >= REFERENCE_SHARE * FULL as f64;
                let decision = control.next(boundary, reference);
                let pace = pace(decision.unit_rows, rows, at, profile(decision.waves, prompt), scale(trace.len()));
                control.record(&decision, rows, FULL, &pace);
                trace.push(Ran { k: decision.waves, opening: at == 0, reference: decision.reference,
                    transition: decision.transition, reason: decision.reason });
                at += rows;
            }
        }
        trace
    }
    /// Startup calibration: full reference calls at context 0, then the choice.
    fn calibrate(control: &mut WaveControl, profile: impl Fn(u64) -> Profile) -> Vec<u64> {
        let calls = control.calibrate();
        let mut depths = Vec::new();
        for _ in 0..calls {
            let decision = control.next(Boundary::Opening, true);
            control.record(&decision, FULL, FULL, &pace(decision.unit_rows, FULL, 0, profile(decision.waves), 1.0));
            depths.push(decision.waves);
        }
        control.finish_calibration();
        depths
    }
    fn depths(trace: &[Ran]) -> Vec<u64> { trace.iter().map(|ran| ran.k).collect() }
    fn quiet(_: usize) -> f64 { 1.0 }
    /// ±0.75% deterministic noise (cards E: full openings at one depth
    /// spread 1.2–1.6% peak to peak, e.g. GLM 3 x 2752 282.0–285.4 µs/row).
    fn noisy(call: usize) -> f64 { 1.0 + 0.0075 * (((call * 2654435761) % 201) as f64 / 100.0 - 1.0) }

    /// Cards E opening calls (rw762-cards-e coordinator logs, 2 RTX + 6
    /// Sparks): every family's best is 3 x 2752, and at 3 both waits fire
    /// (GLM reply-wait 0.42 over idle 0.07; Step 0.29 and 0.29). GLM's
    /// 2 x 4096 was not run in E (estimated from WS/W3 cards). Kimi's running
    /// calls slow ~3x over a 64K prompt; Step's ~14%; GLM's ~4%.
    fn measured(family: &str) -> fn(u64) -> Profile {
        match family {
            "glm" => |k| match k { 2 => (300.0, 0.15, 0.30, 0.04), 3 => (283.0, 0.071, 0.42, 0.04),
                _ => (305.4, 0.074, 0.45, 0.04) },
            "kimi" => |k| match k { 2 => (552.8, 0.27, 0.11, 3.0), 3 => (515.0, 0.125, 0.003, 3.0),
                _ => (535.3, 0.083, 0.066, 3.0) },
            _ => |k| match k { 2 => (390.3, 0.463, 0.414, 0.14), 3 => (362.0, 0.286, 0.29, 0.14),
                _ => (376.8, 0.194, 0.326, 0.14) },
        }
    }
    /// Spark-bound: more, smaller units win; the coordinator waits on replies.
    fn cliff(k: u64) -> Profile { match k { 2 => (420.0, 0.0, 0.30, 0.0), 3 => (380.0, 0.0, 0.20, 0.0), _ => (340.0, 0.0, 0.10, 0.0) } }
    /// Coordinator-bound: fewer, wider units win; the Sparks idle.
    fn bound(k: u64) -> Profile { match k { 2 => (300.0, 0.10, 0.0, 0.0), 3 => (330.0, 0.25, 0.0, 0.0), _ => (360.0, 0.35, 0.0, 0.0) } }
    /// A mix of short, 8K and 64K prompts.
    fn mixed(prompts: usize) -> Vec<u64> {
        (0..prompts).map(|p| match p % 4 { 0 => 900, 1 | 2 => 8192, _ => 65536 }).collect()
    }

    #[test]
    fn calibration_measures_every_depth_and_picks_the_best() {
        for (family, best) in [("glm", 3), ("kimi", 3), ("step", 3)] {
            for start in [2, 3, 4] {
                let mut control = WaveControl::new(start, 4, 4096, false);
                let ran = calibrate(&mut control, measured(family));
                // One warm call at the start depth, then each depth once.
                assert_eq!(ran.len(), 4, "{family}: {ran:?}");
                assert_eq!(ran[..2], [start, start]);
                assert!(control.calibrated(), "{family}: {:?}", control.ratios());
                assert_eq!(control.waves(), best, "{family} start {start}: {:?}", control.ratios());
            }
        }
        let mut control = WaveControl::new(3, 4, 4096, false);
        calibrate(&mut control, cliff);
        assert_eq!(control.waves(), 4);
        let mut control = WaveControl::new(3, 4, 4096, false);
        calibrate(&mut control, bound);
        assert_eq!(control.waves(), 2);
        // A pinned or single-candidate controller never calibrates.
        assert_eq!(WaveControl::new(3, 4, 4096, true).calibrate(), 0);
        assert_eq!(WaveControl::new(4, 4, 2048, false).calibrate(), 0);
    }

    #[test]
    fn the_measured_optimum_holds_with_both_signals_firing() {
        // GLM, Kimi and Step at 3 x 2752: idle and (GLM, Step) reply-wait both
        // fire, but 4 and 2 have evidence that they lose. After calibration no
        // call runs off 3 in a card-sized run (60 mixed prompts with noise)...
        for family in ["glm", "kimi", "step"] {
            let profile = measured(family);
            let probe = pace(2752, FULL, 0, profile(3), 1.0);
            assert!(probe.idle_share().unwrap() > IDLE_HIGH, "{family}: {probe:?}");
            let mut control = WaveControl::new(3, 4, 4096, false);
            calibrate(&mut control, profile);
            let trace = run(&mut control, &mixed(60), |k, _| profile(k), noisy);
            let off: Vec<_> = trace.iter().enumerate().filter(|(_, ran)| ran.k != 3).collect();
            assert!(off.is_empty(), "{family}: {off:?}");
            // ...and over 2,000 prompts it leaves 3 only for backstop re-tests,
            // each confirming the old ratios (so their interval doubles): a
            // deeper re-test opening a 64K prompt holds to that prompt's end.
            let mut control = WaveControl::new(3, 4, 4096, false);
            calibrate(&mut control, profile);
            let trace = run(&mut control, &mixed(2000), |k, _| profile(k), noisy);
            let rounds = trace.iter().filter(|ran| ran.reason == "re-test" && ran.k == 2).count();
            let off = trace.iter().filter(|ran| ran.k != 3).count();
            assert!((1..=4).contains(&rounds) && off <= rounds * 9, "{family}: {rounds} rounds, {off} off");
            assert!(trace.iter().filter(|ran| ran.k != 3).all(|ran| ran.reason == "re-test"
                || ran.reason.starts_with("running")), "{family}");
        }
    }

    #[test]
    fn a_quiet_run_of_sixty_calls_never_re_tests() {
        for profile in [measured("glm"), measured("step")] {
            let mut control = WaveControl::new(3, 4, 4096, false);
            calibrate(&mut control, profile);
            let trace = run(&mut control, &[8192; 60], |k, _| profile(k), noisy);
            assert!(trace.iter().all(|ran| ran.k == 3 && ran.reason == "hold"), "{:?}", depths(&trace));
        }
    }

    #[test]
    fn single_call_dips_do_not_trigger_a_re_test() {
        // A power-capped call (2x) every 7th call: never two in a row.
        let mut control = WaveControl::new(3, 4, 4096, false);
        calibrate(&mut control, measured("glm"));
        let trace = run(&mut control, &[8192; 50], |k, _| measured("glm")(k),
            |call| if call % 7 == 3 { 2.0 } else { noisy(call) });
        assert!(trace.iter().all(|ran| ran.k == 3), "{:?}", depths(&trace));
    }

    #[test]
    fn a_step_change_in_the_current_depth_re_tests_its_neighbours() {
        // After 20 calls the current depth (3 x 2752) slows 15% (a kernel or
        // power change at that unit only): the controller re-tests 2 and 4
        // once each and moves to the new best.
        let shift = 20;
        let profile = move |k: u64, prompt: usize| {
            let (period, idle, reply, rise) = measured("glm")(k);
            if prompt >= shift && k == 3 { (period * 1.15, idle, reply, rise) } else { (period, idle, reply, rise) }
        };
        let mut control = WaveControl::new(3, 4, 4096, false);
        calibrate(&mut control, measured("glm"));
        let trace = run(&mut control, &[8192; 40], profile, quiet);
        assert!(trace[..shift].iter().all(|ran| ran.k == 3), "{:?}", depths(&trace));
        let tests: Vec<_> = trace.iter().filter(|ran| ran.reason == "re-test").map(|ran| ran.k).collect();
        assert_eq!(tests, [2, 4], "{:?}", depths(&trace));
        // GLM's 2 x 4096 (300 µs/row) now beats 4 x 2048 (305) and 3 (325): it moves to 2.
        assert!(trace[shift + 6..].iter().all(|ran| ran.k == 2), "{:?}", depths(&trace));
        // A uniform slowdown (every depth, e.g. power) re-tests once but stays at 3.
        let mut control = WaveControl::new(3, 4, 4096, false);
        calibrate(&mut control, measured("glm"));
        let trace = run(&mut control, &[8192; 40], |k, _| measured("glm")(k),
            |call| if call >= shift { 1.2 } else { 1.0 });
        let tests = trace.iter().filter(|ran| ran.reason == "re-test").count();
        assert_eq!(tests, 2, "{:?}", depths(&trace));
        assert!(trace.iter().filter(|ran| ran.reason != "re-test").all(|ran| ran.k == 3), "{:?}", depths(&trace));
    }

    #[test]
    fn the_backstop_re_tests_after_quiet_reference_calls_and_doubles() {
        let mut control = WaveControl::new(3, 4, 4096, false);
        calibrate(&mut control, measured("step"));
        let trace = run(&mut control, &[8192; 400], |k, _| measured("step")(k), quiet);
        let at: Vec<usize> = trace.iter().enumerate().filter(|(_, ran)| ran.reason == "re-test").map(|(i, _)| i).collect();
        // The first round after 64 quiet reference calls (two trials), the next after 128 more.
        assert_eq!(at.len(), 4, "{at:?}");
        assert!(at[0] >= BACKSTOP_FIRST as usize && at[2] - at[0] >= 2 * BACKSTOP_FIRST as usize, "{at:?}");
        assert!(trace.iter().filter(|ran| ran.reason != "re-test").all(|ran| ran.k == 3));
    }

    #[test]
    fn signals_trigger_only_toward_a_depth_without_evidence_and_ties_are_no_trial() {
        // Uncalibrated (calibration skipped): GLM's reply-wait (0.42) beats idle
        // (0.07), so the first full opening measures 4, which loses: 3 holds.
        let mut control = WaveControl::new(3, 4, 4096, false);
        let trace = run(&mut control, &[8192; 12], |k, _| measured("glm")(k), quiet);
        assert_eq!(depths(&trace)[..3], [3, 4, 3], "{:?}", depths(&trace));
        assert!(trace[3..].iter().all(|ran| ran.k == 3), "{:?}", depths(&trace));
        // Step's waits tie (0.29 vs 0.29): no trial, 3 holds unmeasured neighbours.
        let mut control = WaveControl::new(3, 4, 4096, false);
        let trace = run(&mut control, &[8192; 30], |k, _| measured("step")(k), quiet);
        assert!(trace.iter().all(|ran| ran.k == 3), "{:?}", depths(&trace));
        // Calibrated, both signals firing never trigger a trial.
        let mut control = WaveControl::new(3, 4, 4096, false);
        calibrate(&mut control, measured("glm"));
        let trace = run(&mut control, &[8192; 30], |k, _| measured("glm")(k), quiet);
        assert!(trace.iter().all(|ran| ran.k == 3), "{:?}", depths(&trace));
    }

    #[test]
    fn the_larger_wait_share_must_beat_the_other_by_the_margin() {
        let p = |idle: u64, reply: u64| PrefillPace { rows: 100, ranks: vec![(1000 - idle, idle)], wall_us: 1000,
            reply_wait_us: Some(reply) };
        assert_eq!(p(100, 0).direction(), Some(false));
        assert_eq!(p(0, 100).direction(), Some(true));
        assert_eq!(p(70, 420).direction(), Some(true));
        assert_eq!(p(300, 100).direction(), Some(false));
        // Within 1.5x and 0.10 of each other: a tie.
        assert_eq!(p(290, 290).direction(), None);
        assert_eq!(p(100, 140).direction(), None);
        assert_eq!(p(20, 20).direction(), None);
        assert_eq!(PrefillPace { reply_wait_us: None, ..p(20, 0) }.direction(), None);
    }

    #[test]
    fn running_calls_deepen_with_the_evidence_and_widen_only_at_openings() {
        // Coordinator-bound start at 4 with calibration: the next opening
        // widens to 2; a running call never widens past the previous call.
        let mut control = WaveControl::new(4, 4, 4096, false);
        calibrate(&mut control, bound);
        assert_eq!(control.waves(), 2);
        let trace = run(&mut control, &[65536; 4], |k, _| bound(k), quiet);
        for pair in trace.windows(2) {
            assert!(pair[1].k >= pair[0].k || pair[1].opening, "widened mid-prompt: {:?}", depths(&trace));
        }
        assert!(trace.iter().all(|ran| ran.k == 2), "{:?}", depths(&trace));
        // Evidence that a deeper depth wins moves a running call deeper (a
        // transition call), and later openings stay there.
        let mut control = WaveControl::new(3, 4, 4096, false);
        calibrate(&mut control, measured("glm"));
        let evidence = Evidence { waves: 3, ratios: vec![(2, 1.1), (3, 1.0), (4, 0.9)], band: vec![282.0; 4],
            since_test: 0, backstop: BACKSTOP_FIRST };
        control.restore(&evidence).unwrap();
        let decision = control.next(Boundary::Running, false);
        assert!(decision.waves == 4 && decision.transition, "{decision:?}");
    }

    #[test]
    fn short_and_running_calls_are_logged_but_never_evidence() {
        let mut control = WaveControl::new(3, 4, 4096, false);
        calibrate(&mut control, measured("kimi"));
        let before = control.ratios();
        // Kimi's running calls slow 4x with context, and short prompts are
        // slow per row: neither moves the evidence.
        let trace = run(&mut control, &[65536, 900, 4000, 65536], |k, _| measured("kimi")(k), quiet);
        assert!(trace.iter().all(|ran| ran.k == 3), "{:?}", depths(&trace));
        assert_eq!(control.ratios(), before);
        assert_eq!(trace.iter().filter(|ran| ran.reference).count(), 2);
        let decision = control.next(Boundary::Opening, true);
        assert!(!control.record(&decision, 4000, FULL, &pace(2752, 4000, 0, measured("kimi")(3), 1.0)).counted);
        let decision = control.next(Boundary::Opening, true);
        assert!(!control.record(&decision, FULL, FULL, &PrefillPace::default()).counted);
        // Pins never move and never count.
        let mut pinned = WaveControl::new(2, 4, 4096, true);
        let trace = run(&mut pinned, &mixed(20), |k, _| cliff(k), quiet);
        assert!(trace.iter().all(|ran| ran.k == 2 && !ran.transition), "{:?}", depths(&trace));
    }

    #[test]
    fn evidence_round_trips_and_mismatched_evidence_is_refused() {
        let mut control = WaveControl::new(3, 4, 4096, false);
        calibrate(&mut control, measured("glm"));
        run(&mut control, &[8192; 5], |k, _| measured("glm")(k), quiet);
        let (evidence, dirty) = control.evidence();
        assert!(dirty && !control.evidence().1);
        let json = serde_json::to_string(&evidence).unwrap();
        let mut restored = WaveControl::new(4, 4, 4096, false);
        restored.restore(&serde_json::from_str(&json).unwrap()).unwrap();
        assert!(restored.calibrated());
        assert_eq!(restored.waves(), control.waves());
        for ((j, a), (i, b)) in restored.ratios().into_iter().zip(control.ratios()) {
            assert!(j == i && (a.unwrap() - b.unwrap()).abs() < 1e-12);
        }
        // Restored, it runs the same depths as the original.
        let a = run(&mut control, &mixed(20), |k, _| measured("glm")(k), quiet);
        let b = run(&mut restored, &mixed(20), |k, _| measured("glm")(k), quiet);
        assert_eq!(depths(&a), depths(&b));
        let bad = |edit: fn(&mut Evidence)| { let mut e = evidence.clone(); edit(&mut e);
            WaveControl::new(3, 4, 4096, false).restore(&e).is_err() };
        assert!(bad(|e| e.waves = 1));
        assert!(bad(|e| e.ratios.push((1, 1.2))));
        assert!(bad(|e| e.ratios[0].1 = f64::NAN));
        assert!(bad(|e| e.ratios.retain(|&(j, _)| j != e.waves)));
        assert!(bad(|e| e.band = vec![1.0; 9]));
        assert!(bad(|e| e.backstop = 0));
        assert!(WaveControl::new(3, 4, 2752, false).restore(&evidence).is_err());
        assert!(WaveControl::new(3, 4, 4096, true).restore(&evidence).is_err());
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

    /// The reserve-aware forms (TJ's reserve rule) reduce to the standard
    /// arithmetic at the standard reserve for every admitted shape, so
    /// families planned at 8192 pick identical depths and units.
    #[test]
    fn standard_reserve_forms_equal_the_fixed_reserve_arithmetic() {
        // The fixed-reserve arithmetic as it was: unit_rows(k) capped.
        let fixed = |k: u64, cap: u64| super::super::prefill_waves::unit_rows(k).min(cap);
        for slots in 1..=4 {
            for cap in (64..=4096).step_by(64) {
                for k in 1..=4 {
                    assert_eq!(unit_for_in(k, cap, RESERVE_ROWS), fixed(k, cap), "k {k} cap {cap}");
                }
                assert_eq!(candidates(slots, cap), candidates_in(slots, cap, RESERVE_ROWS));
                assert_eq!(reserve_rows(slots, cap), reserve_rows_in(slots, cap, RESERVE_ROWS));
                assert_eq!(call_rows(slots, cap), call_rows_in(slots, cap, RESERVE_ROWS));
                for k in 1..=slots {
                    for pinned in [false, true] {
                        let (a, b) = (WaveControl::new(k, slots, cap, pinned),
                            WaveControl::in_reserve(k, slots, cap, RESERVE_ROWS, pinned));
                        assert_eq!((a.waves(), a.unit_rows(), a.candidates()), (b.waves(), b.unit_rows(), b.candidates()));
                    }
                }
            }
        }
    }

    /// A smaller reserve's depths cut it (5090 TP3: 4224 rows, cap 2112):
    /// depth 3 runs 1408-row units, depth 4 1088, every depth inside the reserve.
    #[test]
    fn a_smaller_reserve_cuts_its_own_units() {
        use super::super::prefill_waves::{charge_rows, unit_rows_in};
        for reserve in [4224u64, 6144, 7872, RESERVE_ROWS] {
            let cap = unit_rows_in(2, reserve);
            assert_eq!(candidates_in(4, cap, reserve), [2, 3, 4]);
            for k in 1..=4 {
                let unit = unit_for_in(k, cap, reserve);
                assert_eq!(unit, unit_rows_in(k, reserve));
                assert!(unit * k <= reserve_rows_in(4, cap, reserve) && unit % 64 == 0);
            }
            assert_eq!(reserve_rows_in(4, cap, reserve), charge_rows(reserve));
            assert_eq!(WaveControl::in_reserve(3, 4, cap, reserve, true).unit_rows(), unit_rows_in(3, reserve));
        }
        assert_eq!((unit_for_in(3, 2112, 4224), unit_for_in(4, 2112, 4224)), (1408, 1088));
    }

    #[test]
    fn per_row_uses_the_busiest_rank() {
        let pace = PrefillPace { rows: 100, ranks: vec![(900, 100), (950, 0), (500, 600)], wall_us: 1, reply_wait_us: None };
        assert_eq!(pace.per_row(), Some((9.5, 0.0)));
        assert!(PrefillPace { rows: 0, ranks: vec![(1, 1)], ..pace.clone() }.per_row().is_none());
        assert_eq!(PrefillPace { rows: 10, ranks: vec![(90, 10)], ..pace }.idle_share(), Some(0.1));
    }
}
