//! `cuteafd tier-cache`: replay a route census trace through the static GPU
//! expert tier and dynamic expert caches at the same number of resident
//! experts (`cuteafd_loader::placement::tier_cache`).
use anyhow::{ensure, Context, Result};
use clap::Args;
use cuteafd_loader::placement::tier_cache::{simulate, trace_totals, CacheCosts, CacheOutcome, CachePolicy};
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
    /// Rank the static tier from this census or profile file; by default the
    /// trace's own totals, the best static tier for this traffic.
    #[arg(long)]
    pub(crate) profile: Option<PathBuf>,
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

pub(crate) fn run(args: TierCacheArgs) -> Result<()> {
    ensure!(args.base_ms >= 0.0 && args.upload_ms >= 0.0 && args.upload_slowdown >= 0.0 && args.margin >= 0.0,
        "tier-cache costs and margin must not be negative");
    let trace = RouteTrace::read(&args.trace)?;
    let topk = trace.sessions.first().map(|(session, _)| session.topk).context("the trace has no session")?;
    ensure!(trace.sessions.iter().all(|(session, _)| session.topk == topk), "the trace's sessions differ in top-k");
    let windows: Vec<RouteWindow> = trace.sessions.iter().flat_map(|(_, windows)| windows.iter().cloned()).collect();
    let profile = match &args.profile {
        Some(path) => ExpertRouteProfile::read(path)?,
        None => trace_totals(&windows).context("the trace has no windows")?,
    };
    let costs = CacheCosts { upload_ns: (args.upload_ms * 1e6) as u64, upload_slowdown: args.upload_slowdown,
        ..CacheCosts::measured((args.base_ms * 1e6) as u64) };
    let mut policies = vec![CachePolicy::Static(profile.clone())];
    policies.extend(args.lfu_windows.iter().map(|&windows| CachePolicy::WindowedLfu { windows, margin: args.margin }));
    policies.extend([CachePolicy::Lru { threshold: args.lru_threshold }, CachePolicy::Oracle]);
    let outcomes = policies.iter().map(|policy| simulate(&windows, topk, args.capacity, &costs, policy, &profile))
        .collect::<Result<Vec<CacheOutcome>, _>>()?;
    if args.json {
        let report = serde_json::to_string_pretty(&serde_json::json!({
            "trace": args.trace, "sessions": trace.sessions.len(), "capacity": args.capacity, "costs": costs,
            "outcomes": outcomes }))?;
        println!("{report}");
        return Ok(());
    }
    let first = &outcomes[0];
    println!("{} sessions, {} windows with decode steps, {} steps; {} resident experts; base {} ms, upload {} ms (+{}%)",
        trace.sessions.len(), first.windows, first.steps, args.capacity, args.base_ms, args.upload_ms,
        args.upload_slowdown * 100.0);
    println!("{:<18} {:>8} {:>10} {:>10} {:>15} {:>9} {:>10}", "policy", "step ms", "experts ms", "upload ms",
        "uploads/window", "hit rate", "vs static");
    for outcome in &outcomes {
        println!("{:<18} {:>8.3} {:>10.3} {:>10.4} {:>15.1} {:>8.1}% {:>+9.1}%", outcome.policy,
            outcome.step_ns / 1e6, outcome.expert_ns / 1e6, outcome.upload_ns / 1e6, outcome.uploads_per_window,
            outcome.hit_rate * 100.0, (first.step_ns / outcome.step_ns - 1.0) * 100.0);
    }
    Ok(())
}
