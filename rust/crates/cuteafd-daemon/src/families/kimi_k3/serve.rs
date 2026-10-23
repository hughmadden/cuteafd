//! Native text serving through the shared scheduler, sampler and exact prefix cache.
use super::{backend::KimiBackend, startup};
use crate::shared::{prefix::PrefixArgs, qwen::scheduler::{self, Backend}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::{chat::kimi::KimiEncoding, ModelEncoding, ModelProfile, NativeLimits, NativeRequest};
use std::{net::SocketAddr, path::PathBuf, sync::{Arc, Mutex}};
use cuteafd_loader::families::kimi_k3::{COMPILED_CONTEXT, DEFAULT_CONTEXT};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    /// Target and drafter share the exact selected MLA record.
    #[arg(long, default_value = "fp8")] pub kv_cache: cuteafd_core::MlaCacheFormat,
    #[arg(long)] pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")] pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")] pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)] pub device: i32,
    #[arg(long, default_value_t = 1)] pub split_device: i32,
    #[arg(long, value_delimiter = ',')] pub peers: Vec<SocketAddr>,
    /// Auto admission grows beyond the selected context floor while memory allows.
    #[arg(long)] pub pool_tokens: Option<u64>,
    /// Native allocation extent and API ceiling; unset takes the checkpoint core's
    /// default (the compiled 1M for a K6 core, 512K for a BF16 core).
    #[arg(long)] pub max_context: Option<usize>,
    #[arg(long, default_value = "auto")] pub rtx_expert_layers: cuteafd_loader::placement::Onboard,
    /// Complete per-rank allocation ceiling, estimated until the first worker census.
    #[arg(long)] pub spark_budget_gib: Option<f64>,
    #[arg(long, default_value = "0.0.0.0:8000")] pub listen: String,
    #[arg(long, default_value_t = 32768)] pub max_output: u32,
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u8).range(1..=64))]
    pub max_sequences: u8,
    #[arg(long)] pub model_id: Option<String>,
    /// Opt in to a checkpoint-native MLA DFlash2 drafter.
    #[arg(long)] pub draft: Option<PathBuf>,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)] pub draft_fp8: bool,
    /// Fixed draft depth; absent lets the shared policy choose up to the checkpoint width.
    #[arg(long, requires = "draft", value_parser = clap::value_parser!(u8).range(1..=7))]
    pub draft_fixed: Option<u8>,
    /// Prefill unit rows (PLAN "Prefill contract"); the lane holds
    /// `--prefill-waves` units, at most 4096 rows.
    #[arg(long, default_value_t = 2048)] pub prefill_rows: usize,
    /// Prefill units in flight: Spark waves outstanding on the one transport.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(1..=4))]
    pub prefill_waves: u8,
    #[command(flatten)] pub prefix: PrefixArgs,
    #[command(flatten)] pub decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)] pub token_io: crate::shared::token_io::TokenIoArgs,
    #[command(flatten)] pub api: crate::shared::api::ApiArgs,
}

impl ServeArgs {
    /// `admitted` is the engine's resolved context extent (the explicit
    /// --max-context, else the checkpoint core's default).
    fn api_limits(&self, admitted: usize) -> Result<NativeLimits> {
        NativeLimits::new(u32::try_from(self.max_context.unwrap_or(admitted).min(admitted))?, self.max_output)
    }

    fn max_context(&self, core: &cuteafd_loader::families::kimi_k3::core::CoreInventory) -> usize {
        self.max_context.unwrap_or_else(|| cuteafd_loader::families::kimi_k3::default_context(core))
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.device >= 0 && self.split_device >= 0 && self.device != self.split_device,
            "Kimi serving requires two distinct coordinator devices");
        ensure!(self.peers.len() == 6 && self.peers.iter().collect::<std::collections::HashSet<_>>().len() == 6,
            "Kimi serving requires six distinct real Spark endpoints");
        ensure!(self.max_context.is_none_or(|context| (1..=COMPILED_CONTEXT).contains(&context)),
            "Kimi context exceeds the compiled 1M extent");
        ensure!(self.spark_budget_gib.is_none_or(|gib| gib.is_finite() && gib > 0.0),
            "Kimi requires a finite positive estimated --spark-budget-gib");
        ensure!(self.prefill_rows % 64 == 0 && (256..=4096).contains(&(self.prefill_rows * usize::from(self.prefill_waves))),
            "Kimi --prefill-rows x --prefill-waves must be page aligned and hold 256..=4096 lane rows");
        ensure!(self.prefix.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Kimi serving requires exact prefix frontiers");
        ensure!(matches!(self.token_io.embed_placement, crate::shared::token_io::EmbedPlacement::Gpu),
            "Kimi serving admits GPU token embeddings only");
        Ok(())
    }
}

pub(crate) async fn run(args: ServeArgs) -> Result<()> {
    args.validate()?;
    let api = args.api.load()?;
    let encoding = Arc::new(KimiEncoding::from_snapshot(&args.snapshot)?);
    let eos = encoding.eos_token_ids();
    let model = args.model_id.clone().or_else(|| args.snapshot.ancestors().find_map(|path| {
        let name = path.file_name()?.to_str()?.strip_prefix("models--")?;
        let (organization, model) = name.split_once("--")?;
        Some(format!("{organization}/{model}"))
    })).context("Kimi serving requires --model-id for a snapshot without HF identity")?;
    let profile = ModelProfile::new(model, ModelEncoding::Kimi(encoding));
    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<usize>>();
    let worker_args = args.clone();
    let worker = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut ready = Some(ready_tx);
        let result = (|| -> Result<()> {
            cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
                .map_err(|_| anyhow::anyhow!("Kimi serving needs an isolated process"))?;
            let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&worker_args.snapshot, false, false)?;
            let cfg = cuteafd_loader::families::kimi_k3::config::KimiK3Config::from_hf(&checkpoint.config)?;
            cfg.release_geometry()?;
            let slots = usize::from(worker_args.max_sequences);
            let mark_bytes = super::state::KdaLayout::new(&cfg, slots, 1, 1)?.mark_bytes()
                .checked_mul(2).context("Kimi TP2 prefix mark extent overflow")?;
            let marks = worker_args.prefix.mark_rule(slots).slots(mark_bytes);
            startup::preflight(&worker_args.manifest, &worker_args.native_lib)?;
            // SAFETY: every native owner drops on this worker before the library.
            let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&worker_args.native_lib)? });
            let programs = library.programs()?.with_manifest(&worker_args.manifest)?;
            let draft = worker_args.draft.as_ref().map(|snapshot| super::draft::Options {
                snapshot: snapshot.clone(), fp8: worker_args.draft_fp8, cache_format: worker_args.kv_cache });
            let loaded = startup::load(&library, &programs, &checkpoint, &cfg, &cfg, startup::Options {
                cache_format: worker_args.kv_cache,
                devices: [worker_args.device, worker_args.split_device], native_lib: &worker_args.native_lib,
                peers: worker_args.peers.clone(), rtx_expert_layers: worker_args.rtx_expert_layers,
                spark_budget_gib: worker_args.spark_budget_gib,
                rows: worker_args.prefill_rows * usize::from(worker_args.prefill_waves),
                waves: usize::from(worker_args.prefill_waves), prefill_rows: Some(worker_args.prefill_rows), slots, marks,
                max_context: worker_args.max_context(&cuteafd_loader::plan::families::kimi::core_inventory(&checkpoint)?),
                pool_tokens: worker_args.pool_tokens.filter(|&tokens| tokens > 0), graphs: true,
                ffn_mode: cuteafd_ffi::KimiFfnMode::default(), hosted_layers: None, draft, verify_rows: 0,
            })?;
            crate::shared::memory_report::release_load_staging(&library);
            let backend = KimiBackend::new(&loaded.engine)?;
            // The shared selector allocates on the calling thread's device and
            // consumes logits on the terminal rank's exact producing stream.
            library.cuda_set_device(worker_args.split_device)?;
            scheduler::schedule(&backend, &worker_args.snapshot, &mut receive, &worker_stats, slots,
                match loaded.engine.native_draft() {
                    Some(draft) => scheduler::Drafts::Neural {
                        depth: worker_args.draft_fixed.map_or(draft.drafts(), usize::from),
                        fixed: worker_args.draft_fixed.is_some() },
                    None => scheduler::Drafts::None,
                }, eos, worker_args.decode_share, &worker_args.prefix,
                worker_args.token_io.token_select, &mut super::super::qwen3_5::serve::text_media(), None, || {
                    loaded.engine.require_live()?;
                    crate::shared::inventory::RuntimeInventory::measure(&library,
                        &[worker_args.device, worker_args.split_device],
                        crate::shared::inventory::InventoryPhase::PostLoad, |_| Ok(()))?
                        .record("kimi_k3-serving-loaded", &loaded.admission_free);
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(Ok(backend.geometry().max_context));
                    }
                    Ok(())
                })
        })();
        if let Some(ready) = ready.take() {
            let error = result.as_ref().err().map(|error| format!("{error:#}"))
                .unwrap_or_else(|| "scheduler exited before readiness".into());
            let _ = ready.send(Err(anyhow::anyhow!("Kimi initialization failed: {error}")));
        }
        result
    });
    let compiled_context = ready_rx.await.context("Kimi worker exited before readiness")??;
    let limits = args.api_limits(compiled_context)?;
    tracing::info!(api_max_context = limits.context(), compiled_max_context = compiled_context,
        "Kimi API and compiled context limits");
    let snapshot = args.snapshot.clone();
    let hub = crate::shared::console::hub(false, || Ok(crate::shared::console::Layout::new(
        "kimi_k3", profile.id.clone(), snapshot.clone())));
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, std::time::Duration::from_secs(25),
        hub.clone(), api.serve(profile, &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    crate::shared::memory_report::log("ready");
    tokio::select! {
        served = axum::serve(listener, api.app(router, hub)
            .into_make_service_with_connect_info::<SocketAddr>()) => served?,
        _ = crate::shared::api::watch_scheduler(worker) => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn args(extra: &[&str]) -> Result<ServeArgs, clap::Error> {
        let mut argv = vec!["cuteafd", "serve-kimi", "--snapshot", "/checkpoint", "--native-lib", "/native.so"];
        argv.extend_from_slice(extra);
        let cli = crate::cli::Cli::try_parse_from(argv)?;
        let crate::cli::Commands::ServeKimi(args) = cli.command else { panic!("Kimi command") };
        Ok(args)
    }

    #[test]
    fn native_draft_is_opt_in_with_explicit_precision_and_bounded_fixed_depth() -> Result<()> {
        let defaults = args(&[])?;
        assert!(defaults.draft.is_none());
        assert!(!defaults.draft_fp8);
        assert!(defaults.draft_fixed.is_none());
        for fp8 in ["false", "true"] {
            let selected = args(&["--draft", "/drafter", "--draft-fp8", fp8, "--draft-fixed", "7"])?;
            assert_eq!(selected.draft.as_deref(), Some(std::path::Path::new("/drafter")));
            assert_eq!(selected.draft_fp8, fp8 == "true");
            assert_eq!(selected.draft_fixed, Some(7));
        }
        assert!(args(&["--draft-fp8"]).is_err());
        assert!(args(&["--draft-fixed", "1"]).is_err());
        for depth in ["0", "8"] {
            assert!(args(&["--draft", "/drafter", "--draft-fixed", depth]).is_err());
        }
        Ok(())
    }

    #[test]
    fn production_defaults_and_preflight_reject_unsupported_geometry() -> Result<()> {
        let peers = "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4,127.0.0.1:5,127.0.0.1:6";
        let defaults = args(&["--peers", peers])?;
        defaults.validate()?;
        // Unset follows the checkpoint core: 1M for a header-native K6 core, 512K for BF16.
        assert_eq!(defaults.max_context, None);
        let bf16 = cuteafd_loader::families::kimi_k3::core::CoreInventory::default();
        assert_eq!(defaults.max_context(&bf16), DEFAULT_CONTEXT);
        let mut k6 = bf16.clone();
        let head = "language_model.lm_head".to_owned();
        k6.projections.insert(head.clone(), cuteafd_loader::families::kimi_k3::core::CoreProjection { stem: head,
            bits: 6, codebook: cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mcg, shape: [163840, 7168],
            stored_outputs: 163840, split: cuteafd_loader::families::kimi_k3::core::CoreSplit::Whole, source_bytes: 0 });
        assert_eq!(defaults.max_context(&k6), COMPILED_CONTEXT);
        assert_eq!(args(&["--peers", peers, "--max-context", "524288"])?.max_context(&k6), 524_288);
        assert_eq!(defaults.kv_cache, cuteafd_core::MlaCacheFormat::Fp8);
        assert_eq!(cuteafd_loader::plan::layout::LayoutOptions::default().mla_kv, defaults.kv_cache);
        assert_eq!(args(&["--peers", peers, "--kv-cache", "bf16"])?.kv_cache, cuteafd_core::MlaCacheFormat::Bf16);
        assert_eq!(args(&["--peers", peers, "--kv-cache", "fp8"])?.kv_cache, cuteafd_core::MlaCacheFormat::Fp8);
        assert!(args(&["--peers", peers, "--kv-cache", "int8"]).is_err());
        assert_eq!(defaults.rtx_expert_layers, cuteafd_loader::placement::Onboard::Auto);
        assert_eq!(defaults.max_sequences, 16);
        assert!(args(&["--peers", peers, "--native-k6"]).is_err());
        assert_eq!(defaults.pool_tokens, None);
        assert_eq!(defaults.spark_budget_gib, None);
        assert_eq!(defaults.api_limits(DEFAULT_CONTEXT)?.context(), DEFAULT_CONTEXT as u32);
        assert_eq!(defaults.api_limits(COMPILED_CONTEXT)?.context(), COMPILED_CONTEXT as u32);
        args(&["--peers", peers, "--max-context", "1048576"])?.validate()?;
        let limited = args(&["--peers", peers, "--max-context", "262144"])?;
        limited.validate()?;
        let limits = limited.api_limits(COMPILED_CONTEXT)?;
        assert_eq!(limits.context(), 262144);
        assert_eq!(limits.output_for_prompt(262143, 32768)?, 1);
        assert!(limits.output_for_prompt(262144, 1).is_err());
        assert_eq!(args(&["--peers", peers, "--rtx-expert-layers", "3"])?.rtx_expert_layers,
            cuteafd_loader::placement::Onboard::Layers(3));
        args(&["--peers", peers, "--rtx-expert-layers", "auto", "--spark-budget-gib", "108"])?.validate()?;
        for budget in ["0", "NaN", "inf"] {
            assert!(args(&["--peers", peers, "--spark-budget-gib", budget])?.validate().is_err());
        }
        assert!(args(&[])?.validate().is_err());
        for extra in [["--split-device", "0"], ["--max-context", "1048577"],
            ["--embedding-placement", "host"], ["--prefix-partial", "on"]] {
            assert!(args(&["--peers", peers, extra[0], extra[1]])?.validate().is_err());
        }
        assert!(!defaults.prefix.host_cache_bytes.enabled(), "host retention remains opt-in");
        for budget in ["0", "auto", "1", "32GiB"] {
            args(&["--peers", peers, "--host-cache-bytes", budget])?.validate()?;
        }
        assert!(args(&["--max-sequences", "65"]).is_err());
        assert!(args(&["--speculator", "mtp"]).is_err());
        Ok(())
    }
}
