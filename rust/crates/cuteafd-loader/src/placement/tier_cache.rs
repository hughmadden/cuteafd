//! GPU expert tier against a dynamic expert cache: replays a route trace
//! ([`RouteWindow`]s from the census trace) through residency policies at the
//! same number of resident experts, with the tier's step model
//! ([`TierStepModel`]) and the cost of uploading an expert.
//!
//! - **Per window:** the window's counts give each expert's chance to be read
//!   by one step of the window's rows ([`ExpertActivation::from_profile`]),
//!   and the step model the routed-expert time for a resident set.
//! - **Static:** the planner's tier from a profile ([`optimized_order`]),
//!   fixed for the run, as served today.
//! - **Dynamic** policies re-choose residents at each window boundary from the
//!   past alone (windowed LFU through the planner's own ranking, or LRU), or
//!   from the window itself (an oracle, the bound for any predictor). New
//!   residents upload while the window's first steps run on the old set, all
//!   of them slowed by the upload's interference.
//! - **Exactness:** GPU and Spark expert results differ in their low bits,
//!   so a resident change makes a prefix's state depend on when it ran. With
//!   byte-exact prefix restores, every snapshot a change could touch is lost:
//!   [`invalidations`] prices that from the replayed requests
//!   ([`ReplayedRequest`]), as the re-prefill of each turn whose cached
//!   prefix outlived a change.
use super::expert_tier::{optimized_order, ExpertActivation, ExpertRouteProfile, ExpertTierError, TierStepModel};
use super::route_trace::RouteWindow;
use serde::{Deserialize, Serialize};

/// The costs beside the tier's step model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CacheCosts {
    /// Per-layer GPU and Spark expert costs; `active_experts` at
    /// `step_rows` rows scales with a window's rows per step.
    pub model: TierStepModel,
    /// Uploading one expert (both TP2 halves) from pinned host memory.
    pub upload_ns: u64,
    /// How much slower a step runs while an upload is in flight.
    pub upload_slowdown: f64,
    /// The rest of a step: attention, dense layers, the drafter, sampling.
    pub base_step_ns: u64,
}

impl CacheCosts {
    /// Measured on a 48 GB SM89 pair with four Sparks: the tier's C1 verify
    /// model, 0.85 ms per pinned-host upload with a 6% slower wave meanwhile
    /// (charged here to the whole step), and `base_step_ns` from the caller.
    pub fn measured(base_step_ns: u64) -> Self {
        Self { model: TierStepModel::C1_VERIFY_48GB_PAIR_4_SPARKS, upload_ns: 850_000, upload_slowdown: 0.06,
            base_step_ns }
    }
}

/// How residents are chosen.
#[derive(Debug, Clone, PartialEq)]
pub enum CachePolicy {
    /// The planner's tier for these counts, fixed for the run.
    Static(ExpertRouteProfile),
    /// The planner's tier for the last `windows` windows' counts, re-chosen
    /// at every boundary. Residents rank `margin` higher than they are, so a
    /// newcomer must beat them clearly (0: no hysteresis).
    WindowedLfu { windows: usize, margin: f64 },
    /// The experts most recently routed at least `threshold` times in a
    /// window, the most routed first among equals.
    Lru { threshold: u64 },
    /// The planner's tier for each window's own counts, known in advance.
    Oracle,
}

impl CachePolicy {
    pub fn name(&self) -> String {
        match self {
            Self::Static(_) => "static".into(),
            Self::WindowedLfu { windows, margin } => format!("lfu-{windows}w-m{margin}"),
            Self::Lru { threshold } => format!("lru-t{threshold}"),
            Self::Oracle => "oracle".into(),
        }
    }
}

/// One policy's replay.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CacheOutcome {
    pub policy: String,
    pub windows: usize,
    pub steps: u64,
    /// Mean time per step: base, routed experts and upload interference.
    pub step_ns: f64,
    /// Mean routed-expert time per step.
    pub expert_ns: f64,
    /// Mean upload interference per step.
    pub upload_ns: f64,
    /// Experts uploaded, per window.
    pub uploads_per_window: f64,
    /// Share of the expected expert reads that resident experts serve, step-weighted.
    pub hit_rate: f64,
    /// Windows whose residents changed at their start, by position in the
    /// replayed slice.
    #[serde(skip)]
    pub changed: Vec<usize>,
}

/// One window as the replay sees it.
struct Window {
    position: usize,
    steps: u64,
    counts: Vec<Vec<u64>>,
    activation: ExpertActivation,
}

/// Replays `windows` (those with decode steps; `topk` routes per row) through
/// `policy` with at most `capacity` resident experts. Every dynamic policy
/// starts from `initial`'s tier (the static tier it would replace).
pub fn simulate(windows: &[RouteWindow], topk: usize, capacity: usize, costs: &CacheCosts, policy: &CachePolicy,
    initial: &ExpertRouteProfile) -> Result<CacheOutcome, ExpertTierError> {
    let windows = prepare(windows, topk, &costs.model)?;
    let first = windows.first().ok_or_else(|| ExpertTierError::Profile("no window with decode steps".into()))?;
    let (layers, experts) = (first.counts.len(), first.counts[0].len());
    let rows = typical_rows(&windows, topk);
    let plan = |counts: &[Vec<u64>], boost: Option<(&[Vec<u16>], f64)>| -> Result<Vec<Vec<u16>>, ExpertTierError> {
        let mut activation = activation(counts, rows, topk, &costs.model)?;
        if let Some((residents, margin)) = boost {
            for (row, ids) in activation.layers.iter_mut().zip(residents) {
                for &id in ids { row[usize::from(id)] = (row[usize::from(id)] as f64 * (1.0 + margin)) as u64; }
            }
        }
        Ok(tier(&activation, capacity, &costs.model))
    };
    let mut residents = match policy {
        CachePolicy::Static(profile) => plan(&profile.layers, None)?,
        _ => plan(&initial.layers, None)?,
    };
    let mut history: Vec<&Window> = Vec::new();
    let mut last_used = vec![vec![(None::<usize>, 0u64); experts]; layers];
    let (mut time, mut expert_time, mut interference, mut uploads, mut hits, mut steps) = (0.0, 0.0, 0.0, 0u64, 0.0, 0u64);
    let mut changed = Vec::new();
    for (index, window) in windows.iter().enumerate() {
        let next = match policy {
            CachePolicy::Static(_) => residents.clone(),
            CachePolicy::Oracle => plan(&window.counts, None)?,
            CachePolicy::WindowedLfu { windows: span, margin } => {
                let recent = &history[history.len().saturating_sub(*span)..];
                if recent.is_empty() { residents.clone() } else {
                    let mut summed = vec![vec![0u64; experts]; layers];
                    for past in recent {
                        for (total, row) in summed.iter_mut().zip(&past.counts) {
                            for (t, c) in total.iter_mut().zip(row) { *t += c; }
                        }
                    }
                    plan(&summed, Some((&residents, *margin)))?
                }
            }
            CachePolicy::Lru { .. } => if history.is_empty() { residents.clone() } else { recent(&last_used, capacity) },
        };
        let added: u64 = next.iter().zip(&residents)
            .map(|(new, old)| new.iter().filter(|id| !old.contains(id)).count() as u64).sum();
        let old = step_expert_ns(&costs.model, &window.activation, &residents);
        let new = step_expert_ns(&costs.model, &window.activation, &next);
        let base = costs.base_step_ns as f64;
        // The first steps run on the old residents until the uploads land,
        // every one of them slowed by the copies.
        let slowed = (base + old) * (1.0 + costs.upload_slowdown);
        let waiting = ((added * costs.upload_ns) as f64 / slowed).min(window.steps as f64);
        let rest = window.steps as f64 - waiting;
        time += waiting * slowed + rest * (base + new);
        expert_time += waiting * old + rest * new;
        interference += waiting * (base + old) * costs.upload_slowdown;
        hits += waiting * hit_rate(&window.activation, &residents) + rest * hit_rate(&window.activation, &next);
        uploads += added;
        steps += window.steps;
        if added > 0 { changed.push(window.position); }
        residents = next;
        if let CachePolicy::Lru { threshold } = policy {
            let threshold = (*threshold).max(1);
            for (layer, row) in window.counts.iter().enumerate() {
                for (expert, &count) in row.iter().enumerate() {
                    if count >= threshold { last_used[layer][expert] = (Some(index), count); }
                }
            }
        }
        history.push(window);
    }
    let per_step = |total: f64| if steps == 0 { 0.0 } else { total / steps as f64 };
    Ok(CacheOutcome { policy: policy.name(), windows: windows.len(), steps, step_ns: per_step(time),
        expert_ns: per_step(expert_time), upload_ns: per_step(interference),
        uploads_per_window: uploads as f64 / windows.len() as f64, hit_rate: per_step(hits), changed })
}

/// The trace's decode and verify counts summed over its windows: the profile
/// of the best static tier for this traffic (`None` without windows).
pub fn trace_totals(windows: &[RouteWindow]) -> Option<ExpertRouteProfile> {
    let first = windows.first()?;
    let mut layers = vec![vec![0u64; first.layers.first().map_or(0, Vec::len)]; first.layers.len()];
    for window in windows {
        for (total, row) in layers.iter_mut().zip(&window.layers) {
            for (sum, count) in total.iter_mut().zip(row) { *sum += count; }
        }
    }
    Some(ExpertRouteProfile { layers })
}

/// The windows that ran decode steps, with their per-step activations.
fn prepare(windows: &[RouteWindow], topk: usize, model: &TierStepModel) -> Result<Vec<Window>, ExpertTierError> {
    windows.iter().enumerate().filter(|(_, w)| w.step_count() > 0 && w.tokens > 0).map(|(position, w)| {
        let rows = (w.tokens as f64 / w.step_count() as f64).round().max(1.0) as u32;
        Ok(Window { position, steps: w.step_count(), counts: w.layers.clone(),
            activation: activation(&w.layers, rows, topk, model)? })
    }).collect()
}

/// One request as the traffic driver logged it: its client session, its turn
/// in that session, wall-clock milliseconds, and the server's token usage
/// (`cached_tokens`: the prompt prefix served from the prefix cache).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ReplayedRequest {
    pub session: String,
    pub turn: u64,
    pub started_ms: i64,
    pub ended_ms: i64,
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    #[serde(default)]
    pub error: Option<String>,
}

impl ReplayedRequest {
    /// The requests of a driver log (JSON lines), without failed ones.
    pub fn read(path: &std::path::Path) -> Result<Vec<Self>, ExpertTierError> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| ExpertTierError::Profile(format!("reading {}: {error}", path.display())))?;
        let mut requests = Vec::new();
        for (number, line) in text.lines().enumerate().filter(|(_, line)| !line.trim().is_empty()) {
            let request: Self = serde_json::from_str(line).map_err(|error|
                ExpertTierError::Profile(format!("{} line {}: {error}", path.display(), number + 1)))?;
            if request.error.is_none() { requests.push(request); }
        }
        Ok(requests)
    }
}

/// Seconds since the epoch of an RFC 3339 UTC time as the census writes it
/// (`YYYY-MM-DDTHH:MM:SSZ`).
pub fn rfc3339_seconds(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    if bytes.len() != 20 || bytes[4] != b'-' || bytes[7] != b'-' || bytes[10] != b'T' || bytes[13] != b':'
        || bytes[16] != b':' || bytes[19] != b'Z' { return None; }
    let field = |range: std::ops::Range<usize>| text.get(range)?.parse::<i64>().ok();
    let (year, month, day) = (field(0..4)?, field(5..7)?, field(8..10)?);
    let (hour, minute, second) = (field(11..13)?, field(14..16)?, field(17..19)?);
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 60 { return None; }
    // Days from the civil date (Howard Hinnant), March-based years.
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Some((era * 146_097 + day_of_era - 719_468) * 86_400 + hour * 3600 + minute * 60 + second)
}

/// A window's wall-clock span in milliseconds: its open and close seconds,
/// the close second included.
fn span_ms(window: &RouteWindow) -> Option<(i64, i64)> {
    Some((rfc3339_seconds(&window.started)? * 1000, rfc3339_seconds(&window.time)? * 1000 + 999))
}

/// Each window's client session: the one whose requests overlap its span
/// the most (`None` where none does, or the window has no times).
pub fn window_sessions(windows: &[RouteWindow], requests: &[ReplayedRequest]) -> Vec<Option<String>> {
    windows.iter().map(|window| {
        let (open, close) = span_ms(window)?;
        let mut overlap: std::collections::BTreeMap<&str, i64> = Default::default();
        for request in requests {
            let shared = close.min(request.ended_ms) - open.max(request.started_ms);
            if shared > 0 { *overlap.entry(request.session.as_str()).or_default() += shared; }
        }
        overlap.into_iter().max_by_key(|&(session, shared)| (shared, std::cmp::Reverse(session)))
            .map(|(session, _)| session.to_owned())
    }).collect()
}

/// What byte-exact prefix restores cost a policy: the turns whose cached
/// prefix was computed before a resident change and reused after it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Invalidation {
    /// Turns that lose their cached prefix, of the turns that had one.
    pub turns: usize,
    pub cached_turns: usize,
    /// Prompt tokens prefilled again, and the time that takes.
    pub tokens: u64,
    pub seconds: f64,
}

/// Prices `changed` (window positions in `windows` whose residents changed
/// at their start) against `requests`: a turn whose cached prefix came from
/// its session's previous turn loses it when a change falls between that
/// turn's start and its own, and prefills it again at `prefill_tok_s`.
pub fn invalidations(windows: &[RouteWindow], changed: &[usize], requests: &[ReplayedRequest],
    prefill_tok_s: f64) -> Invalidation {
    let mut changes: Vec<i64> = changed.iter().filter_map(|&position|
        windows.get(position).and_then(|window| span_ms(window)).map(|(open, _)| open)).collect();
    changes.sort_unstable();
    let mut sessions: std::collections::BTreeMap<&str, Vec<&ReplayedRequest>> = Default::default();
    for request in requests { sessions.entry(request.session.as_str()).or_default().push(request); }
    let mut priced = Invalidation { turns: 0, cached_turns: 0, tokens: 0, seconds: 0.0 };
    for turns in sessions.values_mut() {
        turns.sort_by_key(|request| (request.turn, request.started_ms));
        for pair in turns.windows(2) {
            let (previous, turn) = (pair[0], pair[1]);
            let cached = turn.cached_tokens.unwrap_or(0);
            if cached == 0 { continue; }
            priced.cached_turns += 1;
            let first = changes.partition_point(|&change| change < previous.started_ms);
            if changes.get(first).is_some_and(|&change| change <= turn.started_ms) {
                priced.turns += 1;
                priced.tokens += cached;
            }
        }
    }
    priced.seconds = if prefill_tok_s > 0.0 { priced.tokens as f64 / prefill_tok_s } else { 0.0 };
    priced
}

/// Rows per step over the trace, for planning from summed counts.
fn typical_rows(windows: &[Window], topk: usize) -> u32 {
    let (rows, steps) = windows.iter().fold((0u64, 0u64), |(rows, steps), w|
        (rows + w.counts[0].iter().sum::<u64>() / topk as u64, steps + w.steps));
    ((rows as f64 / steps.max(1) as f64).round() as u32).max(1)
}

/// A step's activations at `rows` rows: the model's distinct experts per
/// step scale with the rows (more requests touch more experts).
fn activation(counts: &[Vec<u64>], rows: u32, topk: usize, model: &TierStepModel)
    -> Result<ExpertActivation, ExpertTierError> {
    let experts = counts.first().map_or(0, Vec::len);
    let active = (u64::from(model.active_experts) * u64::from(rows) / u64::from(model.step_rows.max(1)))
        .clamp(1, experts as u64) as u32;
    ExpertActivation::from_profile(&ExpertRouteProfile { layers: counts.to_vec() }, counts.len(), experts, rows,
        topk as u32, active)
}

/// The planner's resident sets for `activation`: its order's useful prefix,
/// at most `capacity` experts.
fn tier(activation: &ExpertActivation, capacity: usize, model: &TierStepModel) -> Vec<Vec<u16>> {
    let layers: Vec<usize> = (0..activation.layers.len()).collect();
    let order = optimized_order(&layers, activation, model, 0);
    let mut sets = vec![Vec::new(); layers.len()];
    for slot in &order.order[..capacity.min(order.useful)] { sets[usize::from(slot.layer)].push(slot.expert); }
    sets
}

/// The `capacity` most recently used experts, more routes first among equals.
fn recent(last_used: &[Vec<(Option<usize>, u64)>], capacity: usize) -> Vec<Vec<u16>> {
    let mut slots: Vec<_> = last_used.iter().enumerate().flat_map(|(layer, row)| row.iter().enumerate()
        .filter_map(move |(expert, &(window, count))| window.map(|w| (w, count, layer, expert)))).collect();
    slots.sort_by(|a, b| (b.0, b.1).cmp(&(a.0, a.1)).then((a.2, a.3).cmp(&(b.2, b.3))));
    let mut sets = vec![Vec::new(); last_used.len()];
    for &(_, _, layer, expert) in slots.iter().take(capacity) { sets[layer].push(expert as u16); }
    sets
}

fn step_expert_ns(model: &TierStepModel, activation: &ExpertActivation, residents: &[Vec<u16>]) -> f64 {
    let sets: Vec<&[u16]> = residents.iter().map(Vec::as_slice).collect();
    model.step_ns(activation, &sets)
}

fn hit_rate(activation: &ExpertActivation, residents: &[Vec<u16>]) -> f64 {
    let sets: Vec<&[u16]> = residents.iter().map(Vec::as_slice).collect();
    activation.hit_rate(&sets)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYERS: usize = 4;
    const EXPERTS: usize = 64;
    /// Four resident experts a layer: memory binds before the pools balance.
    const CAPACITY: usize = 4 * LAYERS;

    /// 64 steps of 6 rows (6 routes a row): every expert warm (20 routes),
    /// the four `hot` ones 256 more, in every layer.
    fn window(index: u64, hot: &[usize; 4]) -> RouteWindow {
        let mut row = vec![20u64; EXPERTS];
        for &expert in hot { row[expert] += 256; }
        RouteWindow { index, steps: [index * 64, index * 64 + 64], tokens: 64 * 6, prefill_tokens: 0,
            started: String::new(), time: String::new(), requests: vec![1], finished: vec![],
            layers: vec![row; LAYERS], prefill_layers: vec![vec![0; EXPERTS]; LAYERS] }
    }

    fn totals(windows: &[RouteWindow]) -> ExpertRouteProfile { trace_totals(windows).unwrap() }

    fn run(windows: &[RouteWindow], policy: &CachePolicy, costs: &CacheCosts) -> CacheOutcome {
        simulate(windows, 6, CAPACITY, costs, policy, &totals(windows)).unwrap()
    }

    const A: [usize; 4] = [0, 1, 2, 3];
    const B: [usize; 4] = [40, 41, 42, 43];

    #[test]
    fn steady_traffic_gives_every_policy_the_static_tier() {
        let windows: Vec<_> = (0..8).map(|i| window(i, &[1, 5, 9, 13])).collect();
        let costs = CacheCosts::measured(15_000_000);
        let fixed = run(&windows, &CachePolicy::Static(totals(&windows)), &costs);
        assert_eq!((fixed.windows, fixed.steps, fixed.uploads_per_window), (8, 512, 0.0));
        assert!(fixed.hit_rate > 0.15 && fixed.expert_ns > 0.0, "{fixed:?}");
        for policy in [CachePolicy::WindowedLfu { windows: 1, margin: 0.0 }, CachePolicy::Lru { threshold: 200 },
            CachePolicy::Oracle] {
            let outcome = run(&windows, &policy, &costs);
            assert_eq!(outcome.uploads_per_window, 0.0, "{policy:?}");
            assert!((outcome.step_ns - fixed.step_ns).abs() < 1e-6, "{policy:?}: {outcome:?} vs {fixed:?}");
        }
    }

    /// Hotness that moves every four windows: the static tier keeps one
    /// phase's experts, LFU follows a window late and the oracle at once,
    /// each paying its uploads; uploads that outlast a window lose.
    #[test]
    fn moving_hotness_rewards_following_it_until_uploads_dominate() {
        let windows: Vec<_> = (0..16).map(|i| window(i, if (i / 4) % 2 == 0 { &A } else { &B })).collect();
        let costs = CacheCosts::measured(15_000_000);
        let fixed = run(&windows, &CachePolicy::Static(totals(&windows)), &costs);
        let lfu = run(&windows, &CachePolicy::WindowedLfu { windows: 1, margin: 0.0 }, &costs);
        let lru = run(&windows, &CachePolicy::Lru { threshold: 200 }, &costs);
        let oracle = run(&windows, &CachePolicy::Oracle, &costs);
        assert!(oracle.step_ns < lfu.step_ns && lfu.step_ns < fixed.step_ns, "{oracle:?} {lfu:?} {fixed:?}");
        assert!(oracle.hit_rate > lfu.hit_rate && lfu.hit_rate > fixed.hit_rate);
        // The hot experts are the routes' recent ones, so LRU follows as LFU does.
        assert!((lru.step_ns - lfu.step_ns).abs() < 1e-6 && lru.uploads_per_window == lfu.uploads_per_window);
        // The static tier starts on phase A: three switches of one phase each.
        assert_eq!(oracle.uploads_per_window, (3 * CAPACITY) as f64 / 16.0);
        assert!(oracle.upload_ns > 0.0 && fixed.upload_ns == 0.0);
        // At 85 ms an upload outlasts the window: following loses.
        let dear = CacheCosts { upload_ns: 85_000_000, ..costs };
        let dear_lfu = run(&windows, &CachePolicy::WindowedLfu { windows: 1, margin: 0.0 }, &dear);
        assert!(dear_lfu.step_ns > fixed.step_ns, "{dear_lfu:?} vs {fixed:?}");
    }

    #[test]
    fn a_margin_keeps_residents_through_a_brief_change() {
        let near = [0, 1, 2, 30];
        let windows: Vec<_> = (0..6).map(|i| window(i, if i == 3 { &near } else { &A })).collect();
        let costs = CacheCosts::measured(15_000_000);
        let eager = run(&windows, &CachePolicy::WindowedLfu { windows: 1, margin: 0.0 }, &costs);
        let held = run(&windows, &CachePolicy::WindowedLfu { windows: 1, margin: 4.0 }, &costs);
        // Eager swaps expert 3 for 30 after window 3 and back after window 4.
        assert_eq!((eager.uploads_per_window, held.uploads_per_window), ((2 * LAYERS) as f64 / 6.0, 0.0));
    }

    #[test]
    fn census_times_read_as_epoch_seconds() {
        assert_eq!(rfc3339_seconds("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(rfc3339_seconds("2026-09-21T14:13:20Z"), Some(1_790_000_000));
        assert_eq!(rfc3339_seconds("2024-02-29T23:59:59Z"), Some(1_709_251_199));
        for bad in ["", "2026-09-21 14:13:20Z", "2026-13-01T00:00:00Z", "2026-09-21T14:13:20", "x026-09-21T14:13:20Z"] {
            assert_eq!(rfc3339_seconds(bad), None, "{bad}");
        }
    }

    /// Window `index` spans seconds [10 index, 10 index + 9] of a fixed day.
    fn timed(index: u64, hot: &[usize; 4]) -> RouteWindow {
        // 1_790_000_000 is 2026-09-21T14:13:20Z; the offsets stay inside the hour.
        let at = |offset: u64| { let second = 13 * 60 + 20 + offset;
            format!("2026-09-21T14:{:02}:{:02}Z", second / 60, second % 60) };
        let mut window = window(index, hot);
        (window.started, window.time) = (at(10 * index), at(10 * index + 9));
        window
    }

    fn request(session: &str, turn: u64, from_s: i64, to_s: i64, cached: u64) -> ReplayedRequest {
        let base = 1_790_000_000_000i64;
        ReplayedRequest { session: session.into(), turn, started_ms: base + from_s * 1000, ended_ms: base + to_s * 1000,
            prompt_tokens: Some(cached + 100), cached_tokens: Some(cached), error: None }
    }

    #[test]
    fn windows_join_their_sessions_and_changes_price_the_reused_prefixes() {
        let windows: Vec<_> = (0..4).map(|i| timed(i, &A)).collect();
        assert_eq!(rfc3339_seconds(&windows[1].started), Some(1_790_000_010));
        assert_eq!(rfc3339_seconds(&windows[3].time), Some(1_790_000_039));
        // Session s runs turns in windows 0, 1 and 3; t has window 2.
        let requests = [request("s", 0, 0, 9, 0), request("s", 1, 10, 19, 5000), request("t", 0, 20, 29, 0),
            request("s", 2, 30, 39, 9000)];
        assert_eq!(window_sessions(&windows, &requests), ["s", "s", "t", "s"].map(|s| Some(s.to_owned())));
        // No change: nothing to prefill again.
        assert_eq!(invalidations(&windows, &[], &requests, 1000.0),
            Invalidation { turns: 0, cached_turns: 2, tokens: 0, seconds: 0.0 });
        // A change at window 2 falls between turns 1 and 2 of s: turn 2 loses its 9000-token prefix.
        assert_eq!(invalidations(&windows, &[2], &requests, 1000.0),
            Invalidation { turns: 1, cached_turns: 2, tokens: 9000, seconds: 9.0 });
        // A change at window 1 (turn 1's start) costs turn 1 and turn 2 both.
        assert_eq!(invalidations(&windows, &[1], &requests, 1000.0).tokens, 14_000);
        // Windows without times join nothing and price nothing.
        assert_eq!(window_sessions(&[window(0, &A)], &requests), [None]);
        assert_eq!(invalidations(&[window(0, &A)], &[0], &requests, 1000.0).turns, 0);
    }

    #[test]
    fn a_policy_reports_the_windows_it_changed() {
        let windows: Vec<_> = (0..8).map(|i| window(i, if i < 4 { &A } else { &B })).collect();
        let costs = CacheCosts::measured(15_000_000);
        let lfu = run(&windows, &CachePolicy::WindowedLfu { windows: 1, margin: 0.0 }, &costs);
        // Static starts on A's tier from the totals; LFU moves to B a window late.
        assert_eq!(lfu.changed, vec![5]);
        assert!(run(&windows, &CachePolicy::Static(totals(&windows)), &costs).changed.is_empty());
    }

    #[test]
    fn a_trace_without_decode_steps_is_refused() {
        let mut idle = window(0, &A);
        idle.steps = [5, 5];
        assert!(simulate(&[idle], 6, CAPACITY, &CacheCosts::measured(0), &CachePolicy::Oracle,
            &totals(&[window(0, &A)])).is_err());
    }
}
