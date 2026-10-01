//! `cuteafd plan`: inspect a checkpoint and report what this build can serve.
use anyhow::{Context, Result};
use cuteafd_loader::plan::{budget_bytes, plan, render, ExpertPlacement, PlanError, PlanOptions};
use cuteafd_loader::{default_hf_home, resolve_snapshot_at_revision};
use std::path::PathBuf;

use crate::cli::PlanArgs;

/// The planning options `args` name, validated before any checkpoint is read.
fn options(args: &PlanArgs) -> Result<PlanOptions, PlanError> {
    let options = PlanOptions {
        placement: ExpertPlacement::from_spark_ranks(args.spark_ranks),
        spark_budget_bytes: budget_bytes("--spark-budget-gib", args.spark_budget_gib)?,
        coordinator_budget_bytes: budget_bytes("--coordinator-budget-gib", args.coordinator_budget_gib)?,
    };
    options.validate()?;
    Ok(options)
}

pub(crate) fn run_plan(args: PlanArgs) -> Result<()> {
    let options = options(&args)?;
    let snapshot = if PathBuf::from(&args.model).is_dir() {
        PathBuf::from(&args.model)
    } else {
        let hf_home = args.hf_home.clone().unwrap_or_else(default_hf_home);
        let resolved = match resolve_snapshot_at_revision(&args.model, Some(&hf_home), args.revision.as_deref()) {
            Ok(resolved) => resolved,
            // A stale main ref: describe the only snapshot present (serving still refuses it).
            Err(error) if args.revision.is_none() => {
                let snapshots = cuteafd_loader::model_cache_dir(&hf_home, &args.model).join("snapshots");
                let present: Vec<PathBuf> = std::fs::read_dir(&snapshots).map(|entries| {
                    entries.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect()
                }).unwrap_or_default();
                let [only] = present.as_slice() else { return Err(error) };
                eprintln!("note: {error:#}; describing {}", only.display());
                let revision = only.file_name().and_then(|n| n.to_str()).context("snapshot name")?.to_owned();
                resolve_snapshot_at_revision(&args.model, Some(&hf_home), Some(&revision))?
            }
            Err(error) => return Err(error),
        };
        resolved
            .snapshot_path
            .with_context(|| format!("no snapshot of {} under {}", args.model, hf_home.display()))?
    };
    let report = plan(&snapshot, &options)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::testing::{mimo_flash_config, mimo_flash_tensors, mimo_pro_config, mimo_pro_tensors,
        write_snapshot};

    fn args(model: &std::path::Path, spark_ranks: usize, require_ready: bool) -> PlanArgs {
        PlanArgs {
            model: model.display().to_string(),
            revision: None,
            hf_home: None,
            spark_ranks,
            spark_budget_gib: 100.0,
            coordinator_budget_gib: 80.0,
            json: true,
            require_ready,
        }
    }

    #[test]
    fn require_ready_exits_by_verdict_and_options_fail_typed() {
        let flash = tempfile::tempdir().unwrap();
        write_snapshot(flash.path(), &mimo_flash_config(), &mimo_flash_tensors(), Some(1));
        let pro = tempfile::tempdir().unwrap();
        write_snapshot(pro.path(), &mimo_pro_config(), &mimo_pro_tensors(), Some(8));
        // Complete inventories at packaged worlds are servable.
        run_plan(args(flash.path(), 4, true)).unwrap();
        run_plan(args(pro.path(), 6, true)).unwrap();
        run_plan(args(pro.path(), 0, true)).unwrap();
        // V2.6 Pro has no tp4 MXFP4 layout: a verdict, an error only with --require-ready.
        run_plan(args(pro.path(), 4, false)).unwrap();
        let error = run_plan(args(pro.path(), 4, true)).unwrap_err();
        assert!(error.to_string().contains("is not servable by this build"), "{error:#}");
        // Options that describe no deployment fail before the checkpoint is read.
        for (ranks, budget) in [(1, 100.0), (5, 100.0), (8, 100.0), (4, f64::NAN), (4, 0.0), (4, -3.0)] {
            let error = run_plan(PlanArgs { spark_budget_gib: budget, ..args(flash.path(), ranks, false) }).unwrap_err();
            assert!(matches!(error.downcast_ref::<PlanError>(), Some(PlanError::InvalidOption { .. })),
                "{ranks} ranks, {budget} GiB: {error:#}");
        }
    }
}
