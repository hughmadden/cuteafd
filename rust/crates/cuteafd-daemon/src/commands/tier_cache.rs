//! `cuteafd tier-cache`: replay a route census trace through the static GPU
//! expert tier and dynamic expert caches at the same number of resident
//! experts (`cuteafd_loader::placement::tier_cache`).
use anyhow::{ensure, Context, Result};
use clap::Args;
use cuteafd_loader::placement::tier_cache::{invalidations, simulate, trace_totals, window_sessions, CacheCosts,
    CacheOutcome, CachePolicy, Invalidation, ReplayedRequest};
use cuteafd_loader::placement::{ExpertRouteProfile, RouteTrace, RouteWindow};
use std::path::PathBuf;

#[derive(Debug, Args)]
pub(crate) struct TierCacheArgs {
    /// The census trace (`route-census-HOST.windows.jsonl`); every session's
    /// windows replay in order.
    pub(crate) trace: PathBuf,
    /// Resident experts on the GPUs (a TP2 expert's two halves count once).
    #[arg(long)]
    pub(crate) capacity: usize,
    /// A static tier ranked from this census or profile file, as production
    /// would plan it from other traffic (beside the trace's own hindsight tier).
    #[arg(long)]
    pub(crate) profile: Option<PathBuf>,
    /// The traffic driver's request log (JSON lines: session, turn,
    /// started_ms, ended_ms, cached_tokens): joins windows to client sessions
    /// and prices what byte-exact prefix restores cost each policy.
    #[arg(long)]
    pub(crate) requests: Option<PathBuf>,
    /// Client sessions to plan the static tier from (needs --requests); the
    /// other sessions replay, and every policy is judged on them.
    #[arg(long, value_delimiter = ',')]
    pub(crate) train_sessions: Vec<String>,
    /// Prefill speed for prefixes a resident change invalidates, tokens/s.
    #[arg(long, default_value_t = 3300.0)]
    pub(crate) prefill_tok_s: f64,
    /// The rest of a step in ms: attention, dense layers, the drafter, sampling.
    #[arg(long, default_value_t = 15.0)]
    pub(crate) base_ms: f64,
    /// Uploading one expert from pinned host memory, ms.
    #[arg(long, default_value_t = 0.85)]
    pub(crate) upload_ms: f64,
    /// How much slower a step runs while an upload is in flight.
    #[arg(long, default_value_t = 0.06)]
    pub(crate) upload_slowdown: f64,
    /// Windowed LFU history lengths to compare, in windows.
    #[arg(long, value_delimiter = ',', default_value = "1,4,16")]
    pub(crate) lfu_windows: Vec<usize>,
    /// Windowed LFU hysteresis: residents rank this much higher.
    #[arg(long, default_value_t = 0.25)]
    pub(crate) margin: f64,
    /// LRU: routes in a window that make it a use.
    #[arg(long, default_value_t = 1)]
    pub(crate) lru_threshold: u64,
    #[arg(long)]
    pub(crate) json: bool,
}

/// One policy's line: its replay, and with requests what exact restores cost it.
#[derive(Debug, serde::Serialize)]
struct Row {
    #[serde(flatten)]
    outcome: CacheOutcome,
    changes: usize,
    invalidation: Option<Invalidation>,
}

pub(crate) fn run(args: TierCacheArgs) -> Result<()> {
    ensure!(args.base_ms >= 0.0 && args.upload_ms >= 0.0 && args.upload_slowdown >= 0.0 && args.margin >= 0.0
        && args.prefill_tok_s > 0.0, "tier-cache costs, margin and prefill speed must not be negative");
    let trace = RouteTrace::read(&args.trace)?;
    let topk = trace.sessions.first().map(|(session, _)| session.topk).context("the trace has no session")?;
    ensure!(trace.sessions.iter().all(|(session, _)| session.topk == topk), "the trace's sessions differ in top-k");
    let windows: Vec<RouteWindow> = trace.sessions.iter().flat_map(|(_, windows)| windows.iter().cloned()).collect();
    let requests = args.requests.as_deref().map(ReplayedRequest::read).transpose()?;
    // The windows every policy is judged on, and the held-out static tier.
    let (replayed, requests, held_out) = if args.train_sessions.is_empty() { (windows, requests, None) } else {
        let requests = requests.context("--train-sessions needs --requests")?;
        let sessions = window_sessions(&windows, &requests);
        let train = |session: &Option<String>| session.as_ref().is_some_and(|s| args.train_sessions.contains(s));
        let trained: Vec<RouteWindow> = windows.iter().zip(&sessions).filter(|(_, s)| train(s))
            .map(|(w, _)| w.clone()).collect();
        let replayed: Vec<RouteWindow> = windows.iter().zip(&sessions).filter(|(_, s)| s.is_some() && !train(s))
            .map(|(w, _)| w.clone()).collect();
        ensure!(!trained.is_empty() && !replayed.is_empty(),
            "--train-sessions leaves {} windows to plan from and {} to replay", trained.len(), replayed.len());
        let requests = requests.into_iter().filter(|r| !args.train_sessions.contains(&r.session)).collect();
        (replayed, Some(requests), Some(trace_totals(&trained).context("no training windows")?))
    };
    let hindsight = trace_totals(&replayed).context("the trace has no windows")?;
    let realistic = match (&args.profile, held_out) {
        (Some(path), _) => Some(("static-profile", ExpertRouteProfile::read(path)?)),
        (None, Some(profile)) => Some(("static-held-out", profile)),
        (None, None) => None,
    };
    let costs = CacheCosts { upload_ns: (args.upload_ms * 1e6) as u64, upload_slowdown: args.upload_slowdown,
        ..CacheCosts::measured((args.base_ms * 1e6) as u64) };
    // Dynamic policies replace the tier production would have.
    let initial = realistic.as_ref().map_or(&hindsight, |(_, profile)| profile);
    let mut policies = vec![("static-hindsight".to_owned(), CachePolicy::Static(hindsight.clone()))];
    if let Some((name, profile)) = &realistic { policies.push(((*name).to_owned(), CachePolicy::Static(profile.clone()))); }
    for policy in args.lfu_windows.iter().map(|&windows| CachePolicy::WindowedLfu { windows, margin: args.margin })
        .chain([CachePolicy::Lru { threshold: args.lru_threshold }, CachePolicy::Oracle]) {
        policies.push((policy.name(), policy));
    }
    let rows = policies.iter().map(|(name, policy)| -> Result<Row> {
        let mut outcome = simulate(&replayed, topk, args.capacity, &costs, policy, initial)?;
        outcome.policy = name.clone();
        let invalidation = requests.as_ref()
            .map(|requests| invalidations(&replayed, &outcome.changed, requests, args.prefill_tok_s));
        Ok(Row { changes: outcome.changed.len(), outcome, invalidation })
    }).collect::<Result<Vec<_>>>()?;
    if args.json {
        let report = serde_json::to_string_pretty(&serde_json::json!({
            "trace": args.trace, "sessions": trace.sessions.len(), "capacity": args.capacity, "costs": costs,
            "train_sessions": args.train_sessions, "rows": rows }))?;
        println!("{report}");
        return Ok(());
    }
    let time = |row: &Row| row.outcome.step_ns * row.outcome.steps as f64;
    let reference = |name: &str| rows.iter().find(|row| row.outcome.policy == name).map(time);
    let (hindsight_time, realistic_time) = (reference("static-hindsight").unwrap_or(f64::NAN),
        realistic.as_ref().and_then(|(name, _)| reference(name)));
    let first = &rows[0].outcome;
    println!("{} trace sessions, {} replayed windows with decode steps, {} steps; {} resident experts; \
        base {} ms, upload {} ms (+{}%)", trace.sessions.len(), first.windows, first.steps, args.capacity,
        args.base_ms, args.upload_ms, args.upload_slowdown * 100.0);
    println!("{:<18} {:>8} {:>10} {:>9} {:>14} {:>8} {:>10} {:>10} {:>11} {:>11} {:>10}", "policy", "step ms",
        "experts ms", "upload ms", "uploads/window", "hit", "vs hind.", "vs real", "lost turns", "re-prefill", "net real");
    for row in &rows {
        let gain = |reference: f64, extra: f64| (reference / (time(row) + extra) - 1.0) * 100.0;
        let fmt = |value: Option<f64>| value.map_or("-".to_owned(), |v| format!("{v:+.1}%"));
        let extra = row.invalidation.map_or(0.0, |i| i.seconds * 1e9);
        let lost = row.invalidation.map_or("-".to_owned(), |i| format!("{}/{}", i.turns, i.cached_turns));
        let prefill = row.invalidation.map_or("-".to_owned(), |i| format!("{:.1} s", i.seconds));
        println!("{:<18} {:>8.3} {:>10.3} {:>9.4} {:>14.1} {:>7.1}% {:>10} {:>10} {:>11} {:>11} {:>10}",
            row.outcome.policy, row.outcome.step_ns / 1e6, row.outcome.expert_ns / 1e6, row.outcome.upload_ns / 1e6,
            row.outcome.uploads_per_window, row.outcome.hit_rate * 100.0, fmt(Some(gain(hindsight_time, 0.0))),
            fmt(realistic_time.map(|t| gain(t, 0.0))), lost, prefill,
            fmt(realistic_time.or(Some(hindsight_time)).filter(|_| row.invalidation.is_some()).map(|t| gain(t, extra))));
    }
    Ok(())
}
