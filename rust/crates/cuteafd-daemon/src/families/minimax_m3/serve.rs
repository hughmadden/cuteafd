//! Single coordinator startup: validate, admit, load, bind, schedule.
use super::{admission::{self, RuntimeSizing}, backend::MiniMaxBackend, cache::Cache,
    execution::Execution, media::{ReadyVision, RemoteVision}, routed::Routed, MiniMaxError};
use crate::shared::{memory::device::Device, placement::{SparkAdmission, spark_rings},
    prefix::PrefixArgs, qwen::{Encoder, scheduler::{self, Backend}},
    token_io::{EmbedPlacement, EmbedSource, TokenEmbedding, TokenIoArgs}};
use anyhow::{Context, Result};
use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
use cuteafd_loader::{families::{minimax_m3::config::MiniMaxM3Config, qwen4::Qwen4KvCache},
    plan::{checkpoint::Checkpoint, MediaMode}, placement::SparkWorkingSet};
use cuteafd_transport::{expert::SparkWorld, worker_selection::ModelAdmission, TcpTransportConfig};
use std::{cell::RefCell, net::SocketAddr, path::PathBuf, sync::{Arc, Mutex}};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long)]
    split_device: Option<i32>,
    #[arg(long, value_delimiter = ',')]
    peers: Vec<SocketAddr>,
    #[arg(long, default_value_t = 108 << 30)]
    spark_budget_bytes: u64,
    #[arg(long)]
    pool_tokens: Option<u64>,
    #[arg(long, default_value = "fp8")]
    kv_cache: Qwen4KvCache,
    #[arg(long, default_value_t = 0)]
    max_context: u64,
    #[arg(long, alias = "prefill-batch-tokens", default_value_t = 4096, value_parser = clap::value_parser!(u32).range(1..=4096))]
    prefill_rows: u32,
    #[arg(long, default_value = "auto")]
    rtx_expert_layers: String,
    #[arg(long, default_value = "dspark", value_parser = ["dspark", "off"])]
    speculator: String,
    #[arg(long, default_value = "0.0.0.0:8000")]
    listen: String,
    #[arg(long, default_value_t = 32768)]
    max_output: u32,
    #[arg(long, default_value_t = 64, value_parser = clap::value_parser!(u8).range(1..=64))]
    max_sequences: u8,
    #[arg(long)]
    model_id: Option<String>,
    /// Project every prefill row's logits (prefill-shaped probe scoring);
    /// otherwise prefill projects only the rows a request needs.
    #[arg(long)]
    full_prefill_logits: bool,
    /// DSpark context rings (default: max sequences, at most 8). A request
    /// admitted while every ring is taken decodes target-only.
    #[arg(long)]
    draft_context_slots: Option<usize>,
    /// GPU memory (GiB) kept free for runtime growth, as the planner's --headroom-gib.
    #[arg(long, default_value_t = 2.0)]
    headroom_gib: f64,
    #[arg(skip = MediaMode::Auto)]
    pub(crate) vision: MediaMode,
    #[arg(long)]
    vision_peers: Option<String>,
    #[arg(long)]
    encoder_plan_hash: Option<String>,
    #[arg(long)]
    encoder_revision: Option<String>,
    #[arg(long)]
    media_cache_bytes: Option<u64>,
    #[command(flatten)]
    prefix: PrefixArgs,
    #[command(flatten)]
    decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)]
    token_io: TokenIoArgs,
    #[command(flatten)]
    api: crate::shared::api::ApiArgs,
}
impl ServeArgs {
    pub(crate) fn validate(&self) -> Result<Option<usize>> {
        if self.split_device.is_some() { return Err(MiniMaxError::Unsupported("two-GPU head split").into()); }
        if self.device < 0 || !matches!(self.peers.len(), 2 | 3 | 4)
            || self.peers.iter().enumerate().any(|(i, p)| self.peers[..i].contains(p)) {
            return Err(MiniMaxError::Invalid("one coordinator and distinct TP2/TP3/TP4 peers required").into());
        }
        if self.kv_cache != Qwen4KvCache::Fp8 { return Err(MiniMaxError::Unsupported("BF16 KV has no exported MiniMax kernel").into()); }
        if !matches!(self.speculator.as_str(),"off"|"dspark") { return Err(MiniMaxError::Unsupported("speculator must be dspark or off").into()); }
        if self.speculator=="dspark" && self.prefill_rows<7 {return Err(MiniMaxError::Invalid("DSpark needs at least seven scratch rows").into());}
        if matches!(std::env::var("CUTEAFD_SPARK_WRITE").as_deref(), Ok("1" | "on")) {
            return Err(MiniMaxError::Unsupported("Spark write mode has no MiniMax allocation ledger").into());
        }
        if self.max_sequences as u32 > self.prefill_rows || self.max_context > 1_048_576 {
            return Err(MiniMaxError::Invalid("request context or row capacity").into());
        }
        if self.draft_context_slots.is_some_and(|n| n == 0 || n > self.max_sequences as usize) {
            return Err(MiniMaxError::Invalid("draft context slots outside 1..=max sequences").into());
        }
        if !(self.headroom_gib.is_finite() && (0.0..=64.0).contains(&self.headroom_gib)) {
            return Err(MiniMaxError::Invalid("headroom GiB").into());
        }
        if self.prefix.prefix_partial != crate::shared::prefix::Toggle::Off {
            return Err(MiniMaxError::Unsupported("partial prefix reuse").into());
        }
        if self.rtx_expert_layers == "auto" { return Ok(None); }
        let count = self.rtx_expert_layers.parse::<usize>().map_err(|_| MiniMaxError::Invalid("RTX expert layers: use auto or N"))?;
        if count > 57 { return Err(MiniMaxError::Invalid("RTX expert layers exceed routed layer count").into()); }
        Ok(Some(count))
    }
}

fn request(identity: u64, layer: usize, rows: usize, routes: Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>,
    wire: bytes::Bytes, kind: cuteafd_transport::ExpertV2SourceKind) -> Result<cuteafd_transport::ExpertProtocolV2Request> {
    use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor, ExpertV2Dtype};
    if identity == 0 || !(3..60).contains(&layer) || !(1..=4096).contains(&rows) {
        return Err(MiniMaxError::Invalid("Spark request identity or extent").into());
    }
    let mut request = ExpertProtocolV2Request::new_bytes(identity, 1, layer as u32, 6144,
        ExpertV2Dtype::Bf16, (0..rows as u32).map(|row| ExpertProtocolV2RowDescriptor {
            row_id: row as u64, source_kind: kind, source_request_id: identity, token_position: row as u64,
            route_offset: row * 4, route_count: 4 }).collect(), routes, wire)?;
    request.header.flags |= cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

type Ready = (usize, Option<Arc<cuteafd_api::openai::media::MediaPreparer>>);

fn worker(args: &ServeArgs, receive: &mut tokio::sync::mpsc::Receiver<cuteafd_api::openai::NativeRequest>,
    stats: &Arc<Mutex<serde_json::Value>>, eos: Vec<u32>, initialized: impl FnOnce(Ready) -> Result<()>) -> Result<()> {
    let onboard = args.validate()?;
    let checkpoint = Checkpoint::coordinator(&args.snapshot, false, false)?;
    let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
    cfg.release_geometry()?;
    cuteafd_core::set_expert_geometry(ExpertGeometry::MINIMAX_M3)
        .map_err(|_| MiniMaxError::Invalid("expert geometry already installed"))?;
    let library = Arc::new(cuteafd_ffi::minimax::load_library(&args.native_lib)?);
    library.cuda_set_device(args.device)?;
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    programs.load_matching(|name| name.starts_with("minimax_"))?;
    let (rows, sequences) = (args.prefill_rows as usize, args.max_sequences as usize);
    let rings = args.draft_context_slots.unwrap_or_else(|| cuteafd_loader::families::minimax_m3::dspark::default_ring_slots(sequences));
    let draft_contract=if args.speculator=="dspark" {Some(cuteafd_loader::families::minimax_m3::dspark::DsparkContract::open(&args.snapshot)?)} else {None};
    let draft_layout=draft_contract.as_ref().map(|c|super::dspark::Layout::new(&library,c,rings,rows)).transpose()?;
    let (draft_bytes,draft_rings)=draft_layout.as_ref().map_or((0,0),|l|(l.bytes,l.slots));
    let sizing = RuntimeSizing { rows, decode_rows: rows.min(64), sequences,
        context: if args.max_context == 0 {cfg.max_position as u64} else {args.max_context}, spark_ranks: args.peers.len(),
        logit_rows: RuntimeSizing::head_rows(rows, args.full_prefill_logits, draft_layout.as_ref().map_or(0, |l| l.sequences * 7)) };
    let headroom = (args.headroom_gib * (1u64 << 30) as f64) as u64;
    let manifest = cuteafd_loader::plan::checkpoint::read_json(&args.manifest)?;
    let package = crate::shared::experts::fp8::package_directory(&args.native_lib, 1,
        cuteafd_loader::formats::fp8_experts::ExpertFormat::Nvfp4);
    let local_package = package.join("manifest.json");
    let local_package = local_package.is_file().then(|| cuteafd_loader::plan::checkpoint::read_json(&local_package)).transpose()?;
    let available = || -> Result<u64> {
        library.cuda_set_device(args.device)?;
        let (free, total) = library.cuda_memory_info()?;
        Ok(cuteafd_ffi::coordinator_gpu_budget().map_or(free as u64, |b|
            (free as u64).min(b.0.saturating_sub((total - free) as u64))))
    };
    let remote = RemoteVision::parse(args.vision_peers.as_deref(), args.encoder_plan_hash.as_deref(), args.encoder_revision.as_deref())?;
    let (vision_mode, text_admission) = if args.vision == MediaMode::Auto && remote.is_none() {
        let free = available()?;
        let text = admission::solve(&checkpoint, &manifest, local_package.as_ref(), admission::Request {
            sizing, available: free, requested_pool: None, kv_format: args.kv_cache, onboard,
            spark_budget: args.spark_budget_bytes, headroom, draft_bytes,
            host_embedding: args.token_io.embed_placement == EmbedPlacement::Host })?;
        let vision_bytes = ReadyVision::local_bytes(&args.snapshot, &args.native_lib)?;
        let cache_target = args.pool_tokens.filter(|n| *n > 0).unwrap_or(262_144)
            .checked_mul(text.cache.cache.token_bytes(0)).context("MiniMax vision cache target overflow")?;
        let mode = super::media::local_auto_mode(free, text.cache.fixed_bytes, vision_bytes, cache_target)?;
        tracing::info!(?mode, vision_bytes, cache_target, "MiniMax automatic vision placement preserves text admission");
        (mode, Some(text))
    } else { (args.vision, None) };
    let (vision, prefix) = ReadyVision::load(&args.snapshot, &args.native_lib, &library, args.device,
        vision_mode, &args.prefix, args.media_cache_bytes, remote)?;
    let available = available()?;
    let admitted = if let Some(mut text) = text_admission {
        text.readmit(&cfg, sizing, available, args.pool_tokens, args.kv_cache)?;
        text
    } else {
        let remote_reserve = vision.as_ref().map_or(0, |v| v.spark_reserve_bytes);
        admission::solve(&checkpoint, &manifest, local_package.as_ref(), admission::Request {
            sizing, available, requested_pool: args.pool_tokens, kv_format: args.kv_cache, onboard,
            spark_budget: args.spark_budget_bytes.saturating_sub(remote_reserve), headroom, draft_bytes,
            host_embedding: args.token_io.embed_placement == EmbedPlacement::Host })?
    };
    let mut catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    let endpoint = SparkEndpoint { capacity: args.prefill_rows, ingress: SparkIngress::Bf16 };
    let config = TcpTransportConfig::default();
    let working = SparkWorkingSet { rtx_layers: admitted.local_layers.clone(), spark_layers: admitted.spark_layers.clone(),
        spark_ranks: args.peers.len(), prefill_lanes: 1 };
    let pending = SparkAdmission::plan(&args.snapshot, ExpertGeometry::MINIMAX_M3, Some(&catalog), &working,
        args.peers.len(), None, ModelAdmission::for_clients(ExpertGeometry::MINIMAX_M3, endpoint,
            admitted.spark_capacity as u32, spark_rings(config.max_frame_bytes)?, 1)?)?.start(&args.peers)?;
    let device = Device { library: &library, id: args.device };
    let cache_bytes = admitted.cache.cache_bytes.checked_add(256).context("MiniMax cache reserve overflow")?;
    let token_ids = cuteafd_ffi::minimax::Buffer::new(&library, args.device, admitted.ledger.bytes("token_ids").unwrap() as usize)?;
    let embedded = cuteafd_ffi::minimax::Buffer::new(&library, args.device, admitted.ledger.bytes("embedded").unwrap() as usize)?;
    let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == "language_model.model.embed_tokens.weight")
        .context("MiniMax embedding header missing")?;
    let source = EmbedSource::new(&args.snapshot, &tensor.shard, &tensor.meta, cfg.hidden)?;
    let (embedding, (execution, local)) = crate::shared::placement::load_while_admitting(Some(pending), || {
        TokenEmbedding::load(&library, source, args.token_io.embed_placement, || {
            let mut execution = Execution::load(device, &programs, &checkpoint, &cfg, &admitted.ledger)?;
            if let (Some(contract),Some(layout))=(&draft_contract,draft_layout) {execution.load_draft(contract,layout)?;}
            let local = if admitted.local_layers.is_empty() {None} else {
                let tensors = catalog.fp8().context("MiniMax native expert catalog missing")?.for_coordinator();
                Some(crate::shared::experts::fp8::Fp8Experts::load(&library, &tensors, &package,
                    &admitted.local_layers, 1, 0, sizing.rows, available as usize)?)
            };
            Ok((execution, local))
        })
    })?;
    let mut cache = Cache::new(&library, args.device, &admitted.cache.cache, admitted.cache.pool_pages as usize,
        sizing.sequences, cache_bytes)?;
    cache.context = cache.context.min(admitted.cache.pool_tokens as usize);
    cuteafd_transport::worker_selection::require_peer_clients(&args.peers, endpoint, 1)?;
    let world = SparkWorld::canonical(args.peers.len())?;
    let spark = admitted.spark_layers.first().map(|&first| super::routed::Spark::connect(&library, args.device, world,
        &args.peers, sizing.rows, config, first, request)).transpose()?;
    let expander = vision.as_ref().map(|_| cuteafd_loader::media::minimax::span_expander(&args.snapshot, cfg.vocab_size as u32, false)).transpose()?;
    let backend = MiniMaxBackend { execution: &execution, cache: &cache, embedding: &embedding, token_ids, embedded,
        experts: RefCell::new(Routed::new(local, spark)?), rows: sizing.rows, vocabulary: cfg.vocab_size, expander,
        pending: RefCell::new(None), timing: RefCell::new([0.0; 2]), probe: RefCell::new(crate::shared::draft::routes::RoundRoutes::new(1,4)),
        rings: RefCell::new(super::backend::Rings::new(sizing.sequences, rings)), full_prefill_logits: args.full_prefill_logits };
    let (encoder, preparer, media_bytes) = vision.map_or((Encoder::Off, None, 0), |v| (v.encoder, Some(v.preparer), v.cache_bytes));
    let mut media = cuteafd_engine::media::MediaAdmission::new(cuteafd_engine::media::EmbeddingCache::new(media_bytes), encoder, 16);
    tracing::info!(pool_tokens=admitted.cache.pool_tokens, context=cache.context, local_layers=?admitted.local_layers,
        rows=sizing.rows, logit_rows=sizing.logit_rows, draft_rings,
        "MiniMax native eager target; prefill uses the MSA index-prefill and attention-extend kernels");
    let drafts=if args.speculator=="dspark" {scheduler::Drafts::Neural {depth:7,fixed:false}} else {scheduler::Drafts::None};
    scheduler::schedule(&backend, &args.snapshot, receive, stats, sizing.sequences, drafts,
        eos, args.decode_share, &prefix, args.token_io.token_select, &mut media, None,
        || initialized((backend.geometry().max_context, preparer)))
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    use cuteafd_api::openai::{ModelEncoding, ModelProfile, NativeLimits};
    args.validate()?;
    let snapshot = args.snapshot.clone();
    let config = cuteafd_loader::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
    let architecture = config["architectures"][0].as_str().context("MiniMax architecture missing")?;
    let encoding = ModelEncoding::from_architecture(architecture, &snapshot)?.context("MiniMax API architecture unsupported")?;
    let eos = encoding.eos_token_ids();
    let model = args.model_id.clone().or_else(|| cuteafd_api::openai::snapshot_model_id(&snapshot))
        .context("MiniMax requires --model-id for a snapshot without HF identity")?;
    let mut profile = ModelProfile::new(model, encoding).with_tokenizer(&snapshot)?;
    let api = args.api.load()?;
    let (queue, mut receive) = tokio::sync::mpsc::channel(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let engine_args = args.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<Ready>>();
    let handle = tokio::task::spawn_blocking(move || {
        let mut ready = Some(ready_tx);
        let result = worker(&engine_args, &mut receive, &worker_stats, eos, |state| {
            if let Some(ready) = ready.take() { let _ = ready.send(Ok(state)); } Ok(())
        });
        if let Some(ready) = ready.take() {
            let _ = ready.send(Err(anyhow::anyhow!("MiniMax initialization failed: {:#}",
                result.as_ref().err().context("worker exited before readiness")?)));
        }
        result
    });
    let (context, vision) = ready_rx.await.context("MiniMax worker exited before readiness")??;
    if let Some(preparer) = vision { profile = profile.with_loaded_vision(preparer); }
    let limits = NativeLimits::new(u32::try_from(context)?, args.max_output.min(context as u32))?;
    let hub = crate::shared::console::hub(false, || Ok(crate::shared::console::Layout::new("minimax_m3", profile.id.clone(), snapshot.clone())));
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, std::time::Duration::from_secs(25),
        hub.clone(), api.serve(profile, &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    tokio::select! {
        served = axum::serve(listener, api.app(router, hub).into_make_service_with_connect_info::<SocketAddr>()) => served?,
        _ = crate::shared::api::watch_scheduler(handle) => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    fn args() -> ServeArgs {
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "serve-minimax", "--snapshot", "/model",
            "--native-lib", "/lib.so", "--peers", "127.0.0.1:1,127.0.0.1:2"]).unwrap();
        let crate::cli::Commands::ServeMiniMax(args) = cli.command else {panic!("MiniMax dispatch")}; args
    }
    #[test]
    fn minimax_unsupported_modes_fail_before_native_load() {
        assert_eq!(args().validate().unwrap(), None);
        let mut a=args(); a.split_device=Some(1); assert!(a.validate().unwrap_err().downcast_ref::<MiniMaxError>().is_some());
        let mut a=args(); a.kv_cache=Qwen4KvCache::Bf16; assert!(a.validate().is_err());
        let mut a=args(); a.speculator="mtp".into(); assert!(a.validate().is_err());
        assert_eq!(args().speculator,"dspark");
        let mut a=args(); a.speculator="off".into(); assert!(a.validate().is_ok());
        let mut a=args(); a.token_io.embed_placement=EmbedPlacement::Host; assert!(a.validate().is_ok());
        let mut a=args(); a.rtx_expert_layers="58".into(); assert!(a.validate().is_err());
        let mut a=args(); a.rtx_expert_layers="3".into(); assert_eq!(a.validate().unwrap(),Some(3));
    }
}
