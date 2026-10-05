//! `cuteafd plan`: inspect a checkpoint and report what this build can serve.
use anyhow::{Context, Result};
use cuteafd_loader::plan::{budget_bytes, plan, plan_preferred, render, ExpertPlacement, PlanError, PlanOptions};
use cuteafd_loader::{default_hf_home, resolve_snapshot_at_revision};
use std::path::PathBuf;

use crate::cli::PlanArgs;

/// The planning options `args` name, validated before any checkpoint is read.
fn options(args: &PlanArgs) -> Result<PlanOptions, PlanError> {
    let options = PlanOptions {
        placement: ExpertPlacement::from_spark_ranks(args.spark_ranks.unwrap_or(4)),
        spark_budget_bytes: budget_bytes("--spark-budget-gib", args.spark_budget_gib)?,
        coordinator_budget_bytes: budget_bytes("--coordinator-budget-gib", args.coordinator_budget_gib)?
            .min(if args.layout { budget_bytes("--rtx-budget-gib", args.rtx_gib)? } else { u64::MAX }),
        layout: args.layout.then(|| -> Result<_, PlanError> {
            if !(1..=2).contains(&args.rtx) {
                return Err(PlanError::InvalidOption { option: "--rtx", reason: "1 or 2 coordinator GPUs".into() });
            }
            Ok(cuteafd_loader::plan::layout::LayoutOptions {
                rtx_bytes: vec![budget_bytes("--rtx-budget-gib", args.rtx_gib)?; args.rtx],
                pool_tokens: args.pool_tokens,
                local_expert_layers: args.local_expert_layers,
                context_tokens: args.context_tokens,
                prefill_rows: args.prefill_rows,
                concurrency: args.concurrency,
                prefix_slots: args.prefix_slots,
                native_mtp_layers: args.native_mtp_layers,
                workspace_manifest: args.workspace_manifest.clone(),
                drafter_bytes: if args.drafter_gib > 0.0 { budget_bytes("--drafter-gib", args.drafter_gib)? } else { 0 },
                ..Default::default()
            })
        }).transpose()?,
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
    let report = if args.spark_ranks.is_some() { plan(&snapshot, &options)? }
        else { plan_preferred(&snapshot, &options)? };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render(&report));
        if let Some(layout) = &report.memory_layout {
            print!("\nmemory layout (planner)\n{}", layout.render());
        }
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
            spark_ranks: Some(spark_ranks),
            spark_budget_gib: 100.0,
            coordinator_budget_gib: 80.0,
            json: true,
            require_ready,
            layout: true,
            rtx: 2,
            rtx_gib: 95.5,
            pool_tokens: None,
            drafter_gib: 0.0,
            local_expert_layers: None,
            context_tokens: 262144,
            prefill_rows: 4096,
            concurrency: 8,
            prefix_slots: None,
            native_mtp_layers: 3,
            workspace_manifest: None,
        }
    }

    #[test]
    fn rtx_budget_flag_and_legacy_alias_bound_every_layout_gpu() {
        use clap::Parser;
        for flag in ["--rtx-budget-gib", "--rtx-gib"] {
            let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--layout", "--rtx", "2", flag, "32"])
                .unwrap();
            let crate::cli::Commands::Plan(args) = cli.command else { panic!("plan") };
            let options = options(&args).unwrap();
            assert_eq!(options.layout.unwrap().rtx_bytes, vec![32 << 30; 2]);
            assert_eq!(options.coordinator_budget_bytes, 32 << 30);
        }
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read"]).unwrap();
        let crate::cli::Commands::Plan(plan_args) = cli.command else { panic!("plan") };
        assert!(options(&plan_args).unwrap().layout.is_none());
        for gib in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let dir = tempfile::tempdir().unwrap();
            let error = options(&PlanArgs { rtx_gib: gib, ..args(dir.path(), 4, false) }).unwrap_err();
            assert!(matches!(error, PlanError::InvalidOption { option: "--rtx-budget-gib", .. }));
        }
    }

    #[test]
    fn layout_refuses_full_storage_shortfall_even_when_weights_fit() {
        let snapshot = tempfile::tempdir().unwrap();
        write_snapshot(snapshot.path(), &mimo_flash_config(), &mimo_flash_tensors(), Some(1));
        let tiny = PlanArgs { rtx_gib: 2.0, ..args(snapshot.path(), 4, true) };
        let mut weight_only = options(&tiny).unwrap();
        weight_only.layout = None;
        assert!(plan(snapshot.path(), &weight_only).unwrap().executable(), "weight inventory fits");
        let report = plan(snapshot.path(), &options(&tiny).unwrap()).unwrap();
        assert!(!report.fits && !report.executable());
        assert!(report.hints.iter().any(|h| h.what.contains("full memory layout")
            && h.what.contains("shortfall") && h.what.contains("bytes")));
        let error = run_plan(tiny).unwrap_err();
        assert!(error.to_string().contains("is not servable"), "{error:#}");
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
