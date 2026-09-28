//! `cuteafd plan`: inspect a checkpoint and report what this build can serve.
use anyhow::{Context, Result};
use cuteafd_loader::plan::{plan, render, PlanOptions};
use cuteafd_loader::{default_hf_home, resolve_snapshot_at_revision};
use std::path::PathBuf;

use crate::cli::PlanArgs;

pub(crate) fn run_plan(args: PlanArgs) -> Result<()> {
    let snapshot = if PathBuf::from(&args.model).is_dir() {
        PathBuf::from(&args.model)
    } else {
        let hf_home = args.hf_home.clone().unwrap_or_else(default_hf_home);
        let resolved = resolve_snapshot_at_revision(&args.model, Some(&hf_home), args.revision.as_deref())?;
        resolved
            .snapshot_path
            .with_context(|| format!("no snapshot of {} under {}", args.model, hf_home.display()))?
    };
    let report = plan(
        &snapshot,
        &PlanOptions {
            spark_ranks: args.spark_ranks,
            spark_budget_bytes: (args.spark_budget_gib * (1u64 << 30) as f64) as u64,
        },
    )?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render(&report));
    }
    if args.require_ready && !report.executable() {
        anyhow::bail!("{} is not servable by this build", args.model);
    }
    Ok(())
}
