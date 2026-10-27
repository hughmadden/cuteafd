//! Runtime prefill wave adaptation (#762 stages 2–4). The planner admits a
//! maximum of `slots` in-flight units and a unit extent; before each prefill
//! call (one window cycle of at most [`call_rows`] rows) the executor asks
//! [`WaveControl`] for the call's depth k′ within that admission, and every
//! finished call reports its pace ([`PrefillPace`]): its wall time, the
//! Sparks' measured busy/idle time on their prefill owner, and the time the
//! coordinator's consuming stream waited on Spark replies. Pure arithmetic
//! over the reported counters; no clock, no I/O.

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
    pub fn direction(&self) -> Option<bool> {
        let idle = self.idle_share().unwrap_or(0.0);
        let reply = self.reply_wait_share().unwrap_or(0.0);
        match (idle > IDLE_HIGH, reply > REPLY_WAIT_HIGH) {
            (true, true) => Some(reply > idle),
            (true, false) => Some(false),
            (false, true) => Some(true),
            (false, false) => None,
        }
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

/// Where a call starts. Fewer, wider units (a lower depth) only at an
/// opening call, where nothing is in flight; more units at any call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// A running call whose depth differs from the previous call's: never a sample.
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
    /// Whether it counts as evidence for its depth.
    pub counted: bool,
    pub period_us: Option<f64>,
    pub idle_share: Option<f64>,
    pub reply_wait_share: Option<f64>,
}

/// Spark idle share above which the Sparks wait on the coordinator: the
/// signal points at fewer, wider units.
pub const IDLE_HIGH: f64 = 0.05;
/// Coordinator reply-wait share above which it waits on the Sparks: the
/// signal points at more units.
pub const REPLY_WAIT_HIGH: f64 = 0.05;
/// A call is a sample when its rows reach this share of the smallest unit.
pub const SAMPLE_SHARE: f64 = 0.6;
/// Recent periods kept per depth and boundary class.
const HISTORY: usize = 2;
/// A sample expires after this many prefill calls of any size (one global
/// clock): an expired depth has no evidence again.
pub const EXPIRY_CALLS: u64 = 15;

/// The last `HISTORY` periods of one depth and class with the call that
/// measured each, newest replacing oldest.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Recent { samples: [(f64, u64); HISTORY], count: usize, next: usize }

impl Recent {
    fn push(&mut self, period: f64, call: u64) {
        self.samples[self.next] = (period, call);
        self.next = (self.next + 1) % HISTORY;
        self.count = (self.count + 1).min(HISTORY);
    }
    fn fresh(&self, now: u64) -> impl Iterator<Item = (f64, u64)> + '_ {
        self.samples[..self.count].iter().copied().filter(move |(_, at)| now.saturating_sub(*at) < EXPIRY_CALLS)
    }
    /// The mean of the samples still fresh at call `now`.
    fn mean(&self, now: u64) -> Option<f64> {
        let (sum, n) = self.fresh(now).fold((0.0, 0), |(sum, n), (period, _)| (sum + period, n + 1));
        (n > 0).then(|| sum / n as f64)
    }
    /// The newest fresh sample.
    fn newest(&self, now: u64) -> Option<f64> {
        self.fresh(now).max_by_key(|&(_, at)| at).map(|(period, _)| period)
    }
}

/// Chooses each prefill call's depth (#762; TJ 2026-10-27). A sample is one
/// call: its own wall period per prompt row, counted for the depth it ran
/// unless it is a transition call or under 60% of the smallest unit. Each
/// depth keeps its last two samples per class (opening, running); a sample
/// expires after 15 calls of any size.
///
/// At each call: the latest sample at the current depth points at a
/// neighbour (Spark idle → fewer, wider units; coordinator reply-wait → more
/// units). When the boundary allows that direction (fewer units only at an
/// opening call) and the neighbour has no fresh evidence at all, the
/// call steps there once to measure it. Otherwise it runs the best fresh
/// depth the boundary allows: at an opening call by the mean of recent
/// opening calls (all start at context 0), at a running call by each depth's
/// newest running call (adjacent calls of one prompt), falling back to the
/// opening calls when the current depth has no running sample yet. A depth
/// without comparable evidence holds until measured. No other exploration.
/// Pinned controllers never move (explicit PREFILL_WAVES, benchmarks, goldens).
#[derive(Debug, Clone, PartialEq)]
pub struct WaveControl {
    k: u64,
    unit_cap: u64,
    pinned: bool,
    candidates: Vec<u64>,
    /// Per depth: [opening, running] evidence.
    recent: [[Recent; 2]; MAX_WAVES as usize],
    /// The newest counted sample: (call, depth, direction).
    latest: Option<(u64, u64, Option<bool>)>,
    /// Prefill calls started so far (every size): the expiry clock.
    calls: u64,
}

fn class(boundary: Boundary) -> usize { match boundary { Boundary::Opening => 0, Boundary::Running => 1 } }

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
        Self { k, unit_cap, pinned, candidates, recent: [[Recent::default(); 2]; MAX_WAVES as usize], latest: None,
            calls: 0 }
    }
    /// The wave count of the next prefill call.
    pub fn waves(&self) -> u64 { self.k }
    /// Its unit inside the admitted extent.
    pub fn unit_rows(&self) -> u64 { unit_for(self.k, self.unit_cap) }
    pub fn pinned(&self) -> bool { self.pinned }
    pub fn candidates(&self) -> &[u64] { &self.candidates }

    fn neighbour(&self, k: u64, up: bool) -> Option<u64> {
        let at = self.candidates.iter().position(|&j| j == k)?;
        if up { self.candidates.get(at + 1).copied() } else { at.checked_sub(1).map(|i| self.candidates[i]) }
    }

    fn evidence(&self, k: u64, boundary: Boundary, now: u64) -> Option<f64> {
        let recent = &self.recent[(k - 1) as usize][class(boundary)];
        match boundary { Boundary::Opening => recent.mean(now), Boundary::Running => recent.newest(now) }
    }

    /// A call of `boundary` starts: returns its depth and id.
    pub fn next(&mut self, boundary: Boundary) -> Decision {
        self.calls += 1;
        let (now, previous) = (self.calls, self.k);
        let reason = if self.pinned { "pinned" } else { self.decide(boundary, now) };
        Decision { call: now, waves: self.k, unit_rows: self.unit_rows(), boundary,
            transition: boundary == Boundary::Running && self.k != previous, reason }
    }

    fn decide(&mut self, boundary: Boundary, now: u64) -> &'static str {
        let k = self.k;
        let allowed = |j: u64| boundary == Boundary::Opening || j >= k;
        // The latest sample at this depth points at a neighbour (none past
        // the end of the candidates: the other signal covers the other way).
        let direction = self.latest.filter(|&(at, depth, _)| depth == k && now.saturating_sub(at) < EXPIRY_CALLS)
            .and_then(|(_, _, up)| up);
        let pointed = direction.and_then(|up| self.neighbour(k, up)).filter(|&j| allowed(j));
        // A trial only where the neighbour has no fresh evidence in either class.
        let unmeasured = |j: u64| [Boundary::Opening, Boundary::Running].iter()
            .all(|&class| self.evidence(j, class, now).is_none());
        if let Some(j) = pointed.filter(|&j| unmeasured(j)) {
            self.k = j;
            return if j < k { "Sparks wait: measure fewer, wider units" } else { "coordinator waits: measure more units" };
        }
        // Compare in the class where the current depth has evidence: a
        // running call by adjacent running calls, or (right after an opening
        // trial) by the opening calls; an opening call only by opening calls.
        let classes: &[Boundary] = match boundary {
            Boundary::Opening => &[Boundary::Opening],
            Boundary::Running => &[Boundary::Running, Boundary::Opening],
        };
        let best = |class: Boundary| self.candidates.iter().filter(|&&j| allowed(j))
            .filter_map(|&j| self.evidence(j, class, now).map(|p| (j, p)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        for &class in classes {
            let Some(current) = self.evidence(k, class, now) else { continue };
            return match best(class) {
                Some((j, p)) if j != k && p < (1.0 - MARGIN) * current => { self.k = j; "best fresh period" }
                _ => "hold",
            };
        }
        // No evidence for the current depth in any comparable class (a trial
        // not yet reported, or a depth reached mid-prompt at its first
        // opening call): hold it, so it is measured.
        "hold"
    }

    /// A finished call reports its pace (in any order, any time after it ran).
    pub fn record(&mut self, decision: &Decision, rows: u64, pace: &PrefillPace) -> Sample {
        let smallest = self.candidates.iter().map(|&k| unit_for(k, self.unit_cap)).min().unwrap_or(1);
        let counted = !self.pinned && !decision.transition && pace.wall_us > 0
            && rows as f64 >= SAMPLE_SHARE * smallest as f64 && self.candidates.contains(&decision.waves);
        let mut sample = Sample { call: decision.call, waves: decision.waves, rows, boundary: decision.boundary, counted,
            period_us: None, idle_share: pace.idle_share(), reply_wait_share: pace.reply_wait_share() };
        if counted {
            let recent = &mut self.recent[(decision.waves - 1) as usize][class(decision.boundary)];
            recent.push(pace.wall_us as f64 / rows as f64, decision.call);
            sample.period_us = Some(pace.wall_us as f64 / rows as f64);
            if self.latest.is_none_or(|(at, _, _)| at < decision.call) {
                self.latest = Some((decision.call, decision.waves, pace.direction()));
            }
        }
        sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One call of `rows` rows through a k-deep pipeline at context `at`:
    /// each unit of `u` rows costs the Sparks B = s(u)·u and the coordinator
    /// T = τ(u)·u·(1 + rise·at/65536) (attention grows with context). With k
    /// units circulating one completes per max(B, T, (T + B) / k); the Sparks
    /// idle and the coordinator waits on replies for the rest of each period,
    /// and the call adds one fill.
    fn pace(k: u64, u: u64, rows: u64, at: u64, (s, tau, rise): (f64, f64, f64)) -> PrefillPace {
        let (b, t) = (s * u as f64, tau * u as f64 * (1.0 + rise * at as f64 / 65536.0));
        let period = b.max(t).max((t + b) / k as f64);
        let units = rows.div_ceil(u) as f64;
        let wall = units * period + (t + b - period).max(0.0);
        PrefillPace { rows, ranks: vec![((units * b) as u64, (units * (period - b)) as u64); 6], wall_us: wall as u64,
            reply_wait_us: Some((units * (period - t)) as u64) }
    }

    /// One call as run: its depth, boundary, transition flag and wall time.
    #[derive(Debug, Clone, Copy)]
    struct Ran { k: u64, opening: bool, transition: bool, wall: u64 }

    /// Runs prompts of the given lengths (each cut into `call_rows` calls)
    /// through the controller, each call reported before the next starts;
    /// `profile(u, prompt)` gives (s, τ, rise). Returns every call.
    fn run(control: &mut WaveControl, prompts: &[u64], profile: impl Fn(u64, usize) -> (f64, f64, f64)) -> Vec<Ran> {
        let call = call_rows(4, 4096);
        let mut trace = Vec::new();
        for (prompt, &length) in prompts.iter().enumerate() {
            let mut at = 0;
            while at < length {
                let rows = call.min(length - at);
                let boundary = if at == 0 { Boundary::Opening } else { Boundary::Running };
                let decision = control.next(boundary);
                let unit = decision.unit_rows;
                let pace = pace(decision.waves, unit, rows, at, profile(unit, prompt));
                control.record(&decision, rows, &pace);
                trace.push(Ran { k: decision.waves, opening: at == 0, transition: decision.transition, wall: pace.wall_us });
                at += rows;
            }
        }
        trace
    }
    fn depths(trace: &[Ran]) -> Vec<u64> { trace.iter().map(|ran| ran.k).collect() }

    /// The cards' GLM K4 / Kimi BF16 figures (rw762-cards-a, per layer-wave
    /// µs per row: Spark busy, and a coordinator turnaround reproducing each
    /// depth's measured period and Spark idle) and Step FP8's A8-wire unit
    /// runs (#802: 3 x 2752 delivered 2720 tok/s at 8K, 4 x 2048 2483,
    /// 2 x 4096 2347): 3 x 2752 is best. Kimi's coordinator cost rises 4x
    /// over a 64K prompt (rw762-cards-d: 526 → 2105 µs/row per call).
    fn measured(family: &str) -> impl Fn(u64, usize) -> (f64, f64, f64) + Copy {
        let family = match family { "glm" => 0, "kimi" => 1, _ => 2 };
        move |unit: u64, _| match (family, unit) {
            (0, 4096) => (3.36, 4.90, 0.0), (0, 2752) => (3.71, 4.00, 0.0), (0, _) => (3.90, 4.34, 0.0),
            (1, 4096) => (4.45, 6.19, 3.0), (1, 2752) => (4.97, 5.90, 3.0), (1, _) => (5.40, 6.65, 3.0),
            (_, 4096) => (350.0, 426.1, 0.1), (_, 2752) => (367.6, 300.0, 0.1), (_, _) => (380.0, 402.7, 0.1),
        }
    }

    /// Spark-bound: the superseded BF16-input Step kernel, where 2048-row
    /// units cost the Sparks far less per row than wider ones.
    fn cliff(unit: u64) -> (f64, f64, f64) { (match unit { 4096 => 5.1, 2752 => 6.0, _ => 4.1 }, 2.0, 0.0) }
    /// Coordinator-bound: a fixed per-unit coordinator cost, cheaper per row in wider units.
    fn bound(unit: u64) -> (f64, f64, f64) { (1.5, 1.0 + 4_000.0 / unit as f64, 0.0) }
    /// A mix of 8K and 64K prompts.
    fn mixed(prompts: usize) -> Vec<u64> { (0..prompts).map(|p| if p % 3 == 2 { 65536 } else { 8192 }).collect() }

    #[test]
    fn measured_glm_kimi_and_step_figures_settle_on_three_by_2752() {
        for family in ["glm", "kimi", "step"] {
            // From the planner's estimate (3) or the deepest depth.
            for start in 3..=4 {
                let mut control = WaveControl::new(start, 4, 4096, false);
                let prompts = mixed(60);
                let trace = run(&mut control, &prompts, measured(family));
                let mut fixed = WaveControl::new(3, 4, 4096, true);
                let wall = |trace: &[Ran]| trace[8..].iter().map(|ran| ran.wall as f64).sum::<f64>();
                let cost = wall(&trace) / wall(&run(&mut fixed, &prompts, measured(family)));
                eprintln!("{family} start {start}: adaptive/static-3 wall {cost:.4}");
                // Settled on 3: most of the last prompts open at 3 (one may be a trial).
                let opened: Vec<u64> = trace.iter().filter(|ran| ran.opening).map(|ran| ran.k).collect();
                assert!(opened[opened.len() - 4..].iter().filter(|&&k| k == 3).count() >= 3,
                    "{family} start {start}: {:?}", depths(&trace));
                // Away from 3 only to measure a neighbour, each after an expiry:
                // a wider trial is one opening call; a narrower one (a running
                // call may only deepen) lasts at most to its prompt's end and
                // one more opening call (eight 8K calls of a 64K prompt + 1).
                let (mut entries, mut away) = (0, 0);
                for pair in trace[8..].windows(2) {
                    if pair[1].k != 3 && pair[0].k == 3 { entries += 1; }
                    away = if pair[1].k == 3 { 0 } else { away + 1 };
                    assert!(away <= 9, "{family}: {:?}", depths(&trace));
                }
                assert!(entries <= 2 * trace.len() / EXPIRY_CALLS as usize + 2, "{family} start {start}: {:?}",
                    depths(&trace));
                assert!(cost < 1.06, "{family} start {start}: {cost}");
            }
        }
    }

    #[test]
    fn a_coordinator_bound_optimum_with_no_reply_wait_holds_without_a_costly_excursion() {
        // GLM at 3 x 2752: the Sparks idle 7% (pointing at fewer, wider units),
        // the coordinator never waits on replies. 2 x 4096 measures worse, so 3
        // holds; each re-measure of 2 (after its evidence expires) costs one
        // opening call, never a whole long prompt.
        let mut control = WaveControl::new(3, 4, 4096, false);
        let trace = run(&mut control, &mixed(60), measured("glm"));
        let trials: Vec<&Ran> = trace.iter().filter(|ran| ran.k == 2).collect();
        assert!(!trials.is_empty() && trials.iter().all(|ran| ran.opening), "{:?}", depths(&trace));
        assert!(trials.len() <= trace.len() / EXPIRY_CALLS as usize + 1, "{} trials: {:?}", trials.len(), depths(&trace));
        assert!(trace.iter().all(|ran| ran.k != 4), "nothing points at more units: {:?}", depths(&trace));
        // Every call after a trial returns to 3 at once (the trial is never held).
        for pair in trace.windows(2) {
            if pair[0].k == 2 { assert_eq!(pair[1].k, 3, "{:?}", depths(&trace)); }
        }
    }

    #[test]
    fn a_spark_bound_pipeline_steps_to_more_windows_inside_a_long_prompt() {
        // Starting at 3, the coordinator waits on the Sparks: the second call
        // of the first 64K prompt already runs 4 x 2048 (a transition call),
        // and the depth stays at 4 for the rest of the prompt and after.
        let mut control = WaveControl::new(3, 4, 4096, false);
        let trace = run(&mut control, &[65536; 6], |unit, _| cliff(unit));
        assert_eq!(depths(&trace[..3]), [3, 4, 4], "{:?}", depths(&trace));
        assert!(trace[1].transition && !trace[2].transition);
        assert!(trace[8..].iter().all(|ran| ran.k == 4), "{:?}", depths(&trace));
        assert_eq!(control.waves(), 4);
    }

    #[test]
    fn a_coordinator_bound_pipeline_widens_only_at_an_opening_call() {
        let mut control = WaveControl::new(4, 4, 4096, false);
        let trace = run(&mut control, &[65536; 8], |unit, _| bound(unit));
        for pair in trace.windows(2) {
            assert!(pair[1].k >= pair[0].k || pair[1].opening, "widened mid-prompt: {:?}", depths(&trace));
        }
        // The whole first prompt runs at 4; the next opening widens to 3, then 2.
        assert!(trace[..8].iter().all(|ran| ran.k == 4), "{:?}", depths(&trace));
        assert_eq!(control.waves(), 2, "{:?}", depths(&trace));
    }

    #[test]
    fn transition_calls_are_never_samples() {
        let mut control = WaveControl::new(3, 4, 4096, false);
        let opening = control.next(Boundary::Opening);
        let spark_bound = pace(3, 2752, 8192, 0, cliff(2752));
        assert!(control.record(&opening, 8192, &spark_bound).counted);
        let transition = control.next(Boundary::Running);
        assert!(transition.transition && transition.waves == 4, "{transition:?}");
        let sample = control.record(&transition, 8192, &pace(4, 2048, 8192, 8192, cliff(2048)));
        assert!(!sample.counted && sample.period_us.is_none());
        // The next call at 4 is the first sample for 4.
        let steady = control.next(Boundary::Running);
        assert!(!steady.transition && steady.waves == 4);
        assert!(control.record(&steady, 8192, &pace(4, 2048, 8192, 16384, cliff(2048))).counted);
        // A depth change at an opening call is a sample (nothing in flight).
        let mut control = WaveControl::new(3, 4, 4096, false);
        let first = control.next(Boundary::Opening);
        control.record(&first, 8192, &pace(3, 2752, 8192, 0, bound(2752)));
        let trial = control.next(Boundary::Opening);
        assert!(trial.waves == 2 && !trial.transition);
        assert!(control.record(&trial, 8192, &pace(2, 4096, 8192, 0, bound(4096))).counted);
    }

    #[test]
    fn a_depth_switch_costs_no_more_than_the_depths_own_calls() {
        // The first call at a new depth (after a switch) costs the same as
        // later calls at that depth and context: no first-use penalty, and a
        // trial costs one call, not a whole prompt.
        let mut control = WaveControl::new(3, 4, 4096, false);
        let trace = run(&mut control, &[8192; 10], |unit, _| bound(unit));
        let at = |k: u64| trace.iter().filter(|ran| ran.k == k).map(|ran| ran.wall).collect::<Vec<_>>();
        for k in [2, 3] {
            let walls = at(k);
            assert!(walls.windows(2).all(|w| w[0] == w[1]), "k {k}: {walls:?}");
        }
        let mut control = WaveControl::new(3, 4, 4096, false);
        let trace = run(&mut control, &mixed(30), measured("kimi"));
        // Kimi's 64K prompts: a trial at 2 is one opening call; the rest of the
        // prompt runs the chosen depth.
        let mut runs = Vec::new();
        for ran in &trace {
            match runs.last_mut() { Some((k, n)) if *k == ran.k => *n += 1, _ => runs.push((ran.k, 1)) }
        }
        assert!(runs.iter().filter(|(k, _)| *k == 2).all(|&(_, n)| n == 1), "{runs:?}");
    }

    #[test]
    fn a_workload_shift_reconverges_within_an_expiry_period() {
        // First Spark-bound (best 4), then coordinator-bound (best 2): the old
        // evidence expires and the waits lead the controller down.
        let shift = 20;
        let profile = |unit: u64, prompt: usize| if prompt < shift { cliff(unit) } else { bound(unit) };
        let mut control = WaveControl::new(3, 4, 4096, false);
        let trace = run(&mut control, &[8192; 20 + 3 * EXPIRY_CALLS as usize], profile);
        assert_eq!(trace[shift - 1].k, 4, "{:?}", depths(&trace));
        let first = trace[shift..].iter().position(|ran| ran.k == 2).expect("reaches 2");
        assert!(first <= EXPIRY_CALLS as usize, "{:?}", depths(&trace));
        assert_eq!(control.waves(), 2);
    }

    #[test]
    fn evidence_expires_and_small_calls_and_pins_hold() {
        // Without a signal (both waits between the thresholds) nothing is re-measured.
        let quiet = |unit: u64, _| match unit { 4096 => (3.36, 3.36, 0.0), 2752 => (3.0, 3.06, 0.0), _ => (3.9, 3.9, 0.0) };
        let mut control = WaveControl::new(3, 4, 4096, false);
        let trace = run(&mut control, &[8192; 60], quiet);
        assert!(trace.iter().all(|ran| ran.k == 3), "{:?}", depths(&trace));
        // Calls below 60% of the smallest unit (2048) are not samples, but advance the clock.
        let mut control = WaveControl::new(4, 4, 4096, false);
        let trace = run(&mut control, &[1000; 20], |unit, _| bound(unit));
        assert!(trace.iter().all(|ran| ran.k == 4), "{:?}", depths(&trace));
        let mut pinned = WaveControl::new(2, 4, 4096, true);
        let trace = run(&mut pinned, &mixed(20), |unit, _| cliff(unit));
        assert!(trace.iter().all(|ran| ran.k == 2), "{:?}", depths(&trace));
        let decision = control.next(Boundary::Opening);
        assert!(!control.record(&decision, 8192, &PrefillPace::default()).counted);
    }

    #[test]
    fn the_larger_wait_share_picks_the_direction() {
        let p = |idle: u64, reply: u64| PrefillPace { rows: 100, ranks: vec![(1000 - idle, idle)], wall_us: 1000,
            reply_wait_us: Some(reply) };
        assert_eq!(p(100, 0).direction(), Some(false));
        assert_eq!(p(0, 100).direction(), Some(true));
        assert_eq!(p(100, 200).direction(), Some(true));
        assert_eq!(p(200, 100).direction(), Some(false));
        assert_eq!(p(20, 20).direction(), None);
        assert_eq!(PrefillPace { reply_wait_us: None, ..p(20, 0) }.direction(), None);
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
        let pace = PrefillPace { rows: 100, ranks: vec![(900, 100), (950, 0), (500, 600)], wall_us: 1, reply_wait_us: None };
        assert_eq!(pace.per_row(), Some((9.5, 0.0)));
        assert!(PrefillPace { rows: 0, ranks: vec![(1, 1)], ..pace.clone() }.per_row().is_none());
        assert_eq!(PrefillPace { rows: 10, ranks: vec![(90, 10)], ..pace }.idle_share(), Some(0.1));
    }
}
