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
//! - **Not modelled:** a resident change also changes where a prefix's
//!   experts ran, which the static tier keeps fixed so prefix-cache restores
//!   stay byte-exact.
use super::expert_tier::{optimized_order, ExpertActivation, ExpertRouteProfile, ExpertTierError, TierStepModel};
use super::route_trace::RouteWindow;
use serde::Serialize;

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
}

/// One window as the replay sees it.
struct Window {
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
        uploads_per_window: uploads as f64 / windows.len() as f64, hit_rate: per_step(hits) })
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
    windows.iter().filter(|w| w.step_count() > 0 && w.tokens > 0).map(|w| {
        let rows = (w.tokens as f64 / w.step_count() as f64).round().max(1.0) as u32;
        Ok(Window { steps: w.step_count(), counts: w.layers.clone(), activation: activation(&w.layers, rows, topk, model)? })
    }).collect()
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
    fn a_trace_without_decode_steps_is_refused() {
        let mut idle = window(0, &A);
        idle.steps = [5, 5];
        assert!(simulate(&[idle], 6, CAPACITY, &CacheCosts::measured(0), &CachePolicy::Oracle,
            &totals(&[window(0, &A)])).is_err());
    }
}
