//! `cuteafd plan`: inspect a checkpoint and report what this build can serve.
use anyhow::{Context, Result};
use cuteafd_loader::families::glm5::draft_representation::GlmDraftLinear;
use cuteafd_loader::plan::{budget_bytes, plan, plan_preferred, render, ExpertPlacement, PlanError, PlanOptions};
use cuteafd_loader::{default_hf_home, resolve_snapshot_at_revision};
use std::path::PathBuf;

use crate::cli::PlanArgs;

/// The planning options `args` name, validated before any checkpoint is read.
fn options(args: &PlanArgs) -> Result<PlanOptions, PlanError> {
    if !args.layout && args.vision_replicas != 1 {
        return Err(PlanError::InvalidOption { option: "--vision-replicas", reason: "requires --layout to describe replica inventory".into() });
    }
    let options = PlanOptions {
        vision: args.vision,
        audio: args.audio,
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
                vision_replicas: args.vision_replicas as usize,
                host_embedding: args.embedding_placement == crate::shared::token_io::EmbedPlacement::Host,
                local_expert_layers: args.local_expert_layers,
                context_tokens: args.context_tokens,
                prefill_rows: args.prefill_rows,
                prefill_lanes: args.prefill_lanes,
                glmf_decode_rows: args.decode_rows,
                headroom_bytes: budget_bytes("--headroom-gib", args.headroom_gib)?,
                graph_budget_bytes: args.graph_budget_mib.map(|mib| mib << 20),
                glmf_shared_replay: args.replay_records == crate::families::glm5_flash::engine::ReplayRecords::Shared,
                glmf_pool_marks: args.prefix_marks == crate::families::glm5_flash::prefix::PrefixMarks::Pool,
                glmf_index_cache: args.index_cache.into(),
                glmf_kda_state: args.kda_state.into(),
                glmf_kda_fp8: match args.kda_fp8 {
                    crate::families::glm5_flash::fp8::KdaFp8::Off => cuteafd_loader::plan::layout::GlmfKdaFp8::Off,
                    crate::families::glm5_flash::fp8::KdaFp8::Row128 => cuteafd_loader::plan::layout::GlmfKdaFp8::Row128,
                    crate::families::glm5_flash::fp8::KdaFp8::Channel => cuteafd_loader::plan::layout::GlmfKdaFp8::Channel,
                },
                glmf_fp8_head: args.fp8_head,
                glmf_row_buckets: args.decode_row_buckets,
                glmf_startup_graphs: args.startup_graphs == crate::shared::prefix::Toggle::On,
                glmf_draft: args.draft.clone().map(|snapshot| cuteafd_loader::plan::layout::GlmfDraft {
                    snapshot,
                    fp8: args.draft_fp8 != Some(false),
                    linear: match args.draft_linear {
                        crate::shared::fp8_linear::Fp8Rows::W8a16 => GlmDraftLinear::W8a16,
                        crate::shared::fp8_linear::Fp8Rows::Wide => GlmDraftLinear::Wide,
                        crate::shared::fp8_linear::Fp8Rows::W8a8 => GlmDraftLinear::W8a8,
                    },
                    context_slots: args.draft_context_slots,
                    sequences: args.draft_sequences,
                }),
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
            vision_replicas: 1,
            vision: cuteafd_loader::plan::MediaMode::Auto,
            audio: cuteafd_loader::plan::MediaMode::Off,
            model: model.display().to_string(),
            embedding_placement: crate::shared::token_io::EmbedPlacement::Gpu,
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
            prefill_lanes: 0,
            decode_rows: 0,
            headroom_gib: 2.0,
            graph_budget_mib: None,
            replay_records: crate::families::glm5_flash::engine::ReplayRecords::Own,
            concurrency: 8,
            prefix_slots: None,
            prefix_marks: crate::families::glm5_flash::prefix::PrefixMarks::Arena,
            index_cache: crate::families::glm5_flash::engine::IndexCache::Keys,
            kda_state: crate::families::glm5_flash::engine::KdaState::F32,
            kda_fp8: crate::families::glm5_flash::fp8::KdaFp8::Off,
            fp8_head: false,
            decode_row_buckets: false,
            startup_graphs: crate::shared::prefix::Toggle::On,
            draft: None,
            draft_fp8: None,
            draft_linear: crate::shared::fp8_linear::Fp8Rows::W8a16,
            draft_context_slots: None,
            draft_sequences: 16,
            native_mtp_layers: 3,
            workspace_manifest: None,
        }
    }

    /// serve-glmf's memory flags reach the layout under the same names, so a launcher can size
    /// exactly the flags it serves with.
    #[test]
    fn glm_flash_serving_flags_reach_the_layout() {
        use clap::Parser;
        use cuteafd_loader::plan::layout::GlmfKdaFp8;
        use cuteafd_loader::serving_capacity::{GlmfIndexCache, GlmfKdaState};
        let parse = |extra: &[&str]| {
            let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--layout"].into_iter()
                .chain(extra.iter().copied()));
            let crate::cli::Commands::Plan(args) = cli.unwrap().command else { panic!("plan") };
            options(&args).unwrap().layout.unwrap()
        };
        let defaults = parse(&[]);
        assert_eq!((defaults.glmf_index_cache, defaults.glmf_kda_state, defaults.glmf_kda_fp8, defaults.glmf_fp8_head,
            defaults.glmf_row_buckets), (GlmfIndexCache::Keys, GlmfKdaState::F32, GlmfKdaFp8::Off, false, false));
        // Startup graphs, as serve-glmf captures them by default; `off` plans lazily captured ones.
        assert!(defaults.glmf_startup_graphs && !parse(&["--startup-graphs", "off"]).glmf_startup_graphs);
        assert!(defaults.glmf_draft.is_none());
        let profile = parse(&["--index-cache", "compact", "--kda-state", "bf16", "--kda-fp8", "off", "--fp8-head", "false",
            "--decode-row-buckets", "--draft", "/drafter", "--draft-linear", "w8a8", "--prefill-lane-rows", "4096"]);
        assert_eq!((profile.glmf_index_cache, profile.glmf_kda_state, profile.glmf_row_buckets, profile.prefill_rows),
            (GlmfIndexCache::Compact, GlmfKdaState::Bf16, true, 4096));
        let draft = profile.glmf_draft.unwrap();
        assert_eq!((draft.snapshot, draft.fp8, draft.linear, draft.context_slots, draft.sequences),
            (std::path::PathBuf::from("/drafter"), true, GlmDraftLinear::W8a8, None, 16));
        let precise = parse(&["--kda-fp8", "row128", "--fp8-head", "--draft", "/d", "--draft-fp8", "false",
            "--draft-context-slots", "20", "--draft-sequences", "8"]);
        assert_eq!((precise.glmf_kda_fp8, precise.glmf_fp8_head), (GlmfKdaFp8::Row128, true));
        let draft = precise.glmf_draft.unwrap();
        assert_eq!((draft.fp8, draft.context_slots, draft.sequences), (false, Some(20), 8));
        for bad in [&["--index-cache", "tails"][..], &["--kda-state", "f16"], &["--kda-fp8", "row64"]] {
            let parsed = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--layout"].into_iter()
                .chain(bad.iter().copied()));
            assert!(parsed.is_err(), "{bad:?}");
        }
    }

    #[test]
    fn media_modes_parse_on_every_command_and_invalid_modes_fail() {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--vision", "off", "--audio", "auto"]).unwrap();
        assert_eq!(cli.vision, Some(cuteafd_loader::plan::MediaMode::Off));
        assert_eq!(cli.audio, Some(cuteafd_loader::plan::MediaMode::Auto));
        assert!(crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--vision", "bad"]).is_err());
    }

    #[test]
    fn planner_and_glm_flash_share_auto_default_and_explicit_off() {
        use clap::Parser;
        use cuteafd_loader::plan::MediaMode;
        for command in ["plan", "serve-glmf"] {
            for mode in [None, Some("off")] {
                let mut argv = vec!["cuteafd", command, "--snapshot", "/not-read", "--native-lib", "/not-loaded"];
                if command == "plan" { argv = vec!["cuteafd", command, "/not-read"]; }
                if let Some(mode) = mode { argv.extend(["--vision", mode]); }
                let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
                let expected = if mode.is_some() { MediaMode::Off } else { MediaMode::Auto };
                assert_eq!(crate::resolve_vision(cli.vision, None), expected);
                match cli.command {
                    crate::cli::Commands::Plan(args) => assert_eq!(args.vision, MediaMode::Auto),
                    crate::cli::Commands::ServeGlmf(args) => assert_eq!(args.vision, MediaMode::Auto),
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn glm_flash_prefix_marks_reach_the_layout() {
        use clap::Parser;
        let parse = |extra: &[&str]| crate::cli::Cli::try_parse_from(
            ["cuteafd", "plan", "/not-read", "--layout"].into_iter().chain(extra.iter().copied()));
        for (extra, pool) in [(&[][..], false), (&["--prefix-marks", "arena"][..], false),
            (&["--prefix-marks", "pool"][..], true)] {
            let crate::cli::Commands::Plan(args) = parse(extra).unwrap().command else { panic!("plan") };
            assert_eq!(options(&args).unwrap().layout.unwrap().glmf_pool_marks, pool);
        }
        assert!(parse(&["--prefix-marks", "host"]).is_err());
    }

    #[test]
    fn glm_flash_decode_rows_reach_the_layout() {
        use clap::Parser;
        let parse = |extra: &[&str]| crate::cli::Cli::try_parse_from(
            ["cuteafd", "plan", "/not-read", "--layout"].into_iter().chain(extra.iter().copied()));
        for (extra, rows) in [(&[][..], 0), (&["--decode-rows", "64"][..], 64), (&["--decode-rows", "128"][..], 128)] {
            let crate::cli::Commands::Plan(args) = parse(extra).unwrap().command else { panic!("plan") };
            assert_eq!(options(&args).unwrap().layout.unwrap().glmf_decode_rows, rows);
        }
        assert!(parse(&["--decode-rows", "96"]).is_err());
    }

    #[test]
    fn replicas_require_an_explicit_layout_inventory() {
        let mut request = args(std::path::Path::new("/not-read"), 4, false);
        request.layout = false;
        request.vision_replicas = 2;
        assert!(matches!(options(&request), Err(PlanError::InvalidOption { option: "--vision-replicas", .. })));
        request.layout = true;
        assert_eq!(options(&request).unwrap().layout.unwrap().vision_replicas, 2);
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
        for (ranks, budget) in [(5, 100.0), (8, 100.0), (4, f64::NAN), (4, 0.0), (4, -3.0)] {
            let error = run_plan(PlanArgs { spark_budget_gib: budget, ..args(flash.path(), ranks, false) }).unwrap_err();
            assert!(matches!(error.downcast_ref::<PlanError>(), Some(PlanError::InvalidOption { .. })),
                "{ranks} ranks, {budget} GiB: {error:#}");
        }
    }
}
