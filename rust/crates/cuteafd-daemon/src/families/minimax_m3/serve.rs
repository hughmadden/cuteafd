//! Single coordinator startup: validate, admit, load, bind, schedule.
use super::{admission::{self, RuntimeSizing}, backend::MiniMaxBackend, cache::Cache,
    execution::Execution, media::ReadyVision, routed::Routed, MiniMaxError};
use crate::shared::{memory::device::{Allocation, Device}, placement::SparkAdmission,
    prefix::PrefixArgs, qwen::{Encoder, scheduler::{self, Backend}},
    token_io::{EmbedPlacement, EmbedSource, TokenEmbedding, TokenIoArgs}};
use anyhow::{Context, Result};
use cuteafd_core::ExpertGeometry;
use cuteafd_loader::{families::{minimax_m3::config::MiniMaxM3Config, qwen4::Qwen4KvCache},
    plan::{checkpoint::Checkpoint, MediaMode}, placement::SparkWorkingSet, serving_capacity::prefill_lanes};
use cuteafd_transport::{expert::SparkWorld, TcpTransportConfig};
use std::{cell::RefCell, net::SocketAddr, path::{Path, PathBuf}, sync::{Arc, Mutex}};

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
    #[arg(long, value_delimiter = ',')]
    peers: Vec<SocketAddr>,
    #[arg(long, default_value_t = cuteafd_loader::plan::families::minimax::admission::SPARK_BUDGET_BYTES)]
    spark_budget_bytes: u64,
    #[arg(long)]
    pool_tokens: Option<u64>,
    #[arg(long, default_value = "fp8")]
    kv_cache: Qwen4KvCache,
    #[arg(long, default_value_t = 0)]
    max_context: u64,
    /// Rows of one prefill unit: one Spark wave, and the most rows a program launch takes.
    #[arg(long, alias = "prefill-batch-tokens", alias = "prefill-lane-rows", default_value_t = 4096,
        value_parser = clap::value_parser!(u32).range(1..=4096))]
    prefill_rows: u32,
    #[arg(long, default_value = "auto")]
    rtx_expert_layers: String,
    /// The bundled DSpark (`dspark/`), a DFlash2 checkpoint (`dflash2`, with --draft) on the
    /// shared drafter, or none.
    #[arg(long, default_value = "dspark", value_parser = ["dspark", "dflash2", "off"])]
    speculator: String,
    /// The DFlash2 drafter snapshot (RadixArk/Minimax-M3.1-Flash-Dflash2) for --speculator dflash2.
    #[arg(long)]
    draft: Option<PathBuf>,
    /// DFlash2: explicit calibration-free E4M3 quantization of the drafter's GEMMs at load.
    /// Unset/true: one FP8 copy (the shared drafter's default); false keeps the checkpoint's BF16.
    #[arg(long, action = clap::ArgAction::Set)]
    draft_fp8: Option<bool>,
    /// DFlash2 context-ring K/V storage.
    #[arg(long, default_value = "fp8")]
    draft_kv_cache: Qwen4KvCache,
    /// DFlash2 block causality, overriding SGLang's mask for the checkpoint (causal rows for its
    /// sliding-window layers): an acceptance A/B; verification keeps the output unchanged.
    #[arg(long, hide = true, action = clap::ArgAction::Set)]
    draft_causal: Option<bool>,
    /// DFlash2 selector scores: FP32 sums of the unary logit and BF16 edge, as SGLang's DFlash2
    /// selector scores (the release's serving reference), or each rounded to BF16 (the GLM
    /// ports'): an acceptance A/B.
    #[arg(long, hide = true, default_value = "fp32", value_parser = ["fp32", "bf16"])]
    draft_selector_score: String,
    /// Fixed draft depth (launcher SPECULATOR_DRAFTS); absent lets the
    /// shared policy choose up to the drafter's seven drafts, as on Kimi K3.
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=7))]
    draft_fixed: Option<u8>,
    #[arg(long, default_value = "0.0.0.0:8000")]
    listen: String,
    #[arg(long, default_value_t = 32768)]
    max_output: u32,
    #[arg(long, default_value_t = 64, value_parser = clap::value_parser!(u8).range(1..=64))]
    max_sequences: u8,
    #[arg(long)]
    model_id: Option<String>,
    /// Spark waves a prefill chunk keeps in flight (1 to 4): the chunk runs in up to this many
    /// lanes of --prefill-rows rows, each lane's routed wave in flight on the one Spark link while
    /// the other lanes' GPU layers run. --prefill-lanes is a deprecated alias.
    #[arg(long, alias = "prefill-lanes", default_value_t = prefill_lanes::DEFAULT_PREFILL_LANES,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..=prefill_lanes::MAX_PREFILL_LANES as u64))]
    prefill_waves: usize,
    /// Project every prefill row's logits (prefill-shaped probe scoring);
    /// otherwise prefill projects only the rows a request needs.
    #[arg(long)]
    full_prefill_logits: bool,
    /// Drafter context rings (default: max sequences, at most half the decode rows, 32 at the
    /// shipped shapes; DFlash2 drafts as many sequences per step). A request admitted while every
    /// ring is taken decodes target-only.
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
        // The Spark worlds the expert package ships (the shared Spark-range policy).
        let worlds = admission::spark_worlds();
        if self.device < 0 || !worlds.contains(&self.peers.len())
            || self.peers.iter().enumerate().any(|(i, p)| self.peers[..i].contains(p)) {
            anyhow::bail!("MiniMax serves one coordinator with distinct peers for a Spark world its expert package \
                ships ({worlds:?}), not {} peers", self.peers.len());
        }
        if self.kv_cache != Qwen4KvCache::Fp8 { return Err(MiniMaxError::Unsupported("BF16 KV has no exported MiniMax kernel").into()); }
        if !matches!(self.speculator.as_str(),"off"|"dspark"|"dflash2") {
            return Err(MiniMaxError::Unsupported("speculator must be dspark, dflash2 or off").into());
        }
        if self.speculator=="dspark" && self.prefill_rows<7 {return Err(MiniMaxError::Invalid("DSpark needs at least seven scratch rows").into());}
        if (self.speculator=="dflash2") != self.draft.is_some() {
            return Err(MiniMaxError::Invalid("--speculator dflash2 takes its drafter snapshot with --draft, and only it does").into());
        }
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

pub(super) fn request(identity: u64, layer: usize, rows: usize, routes: Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>,
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

/// What admission chose for one coordinator, for [`start`]: its sizing, ledger and KV pool, the
/// routed layers it runs locally and on the Sparks, and the drafter (serve only).
pub(super) struct Start<'s> {
    pub snapshot: &'s Path,
    pub checkpoint: &'s Checkpoint,
    pub cfg: &'s MiniMaxM3Config,
    pub sizing: RuntimeSizing,
    pub ledger: &'s admission::RuntimeLedger,
    pub cache: &'s cuteafd_loader::plan::families::minimax::admission::Admission,
    pub local_layers: &'s [usize],
    pub spark_layers: &'s [usize],
    pub peers: &'s [SocketAddr],
    pub catalog: &'s cuteafd_loader::OfficialV41Catalog,
    /// The native TP1 expert package directory (for `local_layers`).
    pub package: &'s Path,
    pub embed: EmbedPlacement,
    /// Free coordinator bytes measured before the weights load.
    pub available: u64,
    pub draft: Option<DraftStart<'s>>,
}

/// The drafter serve admitted, loaded by [`start`].
pub(super) enum DraftStart<'s> {
    Dspark(&'s cuteafd_loader::families::minimax_m3::dspark::DsparkContract, super::dspark::Layout),
    /// A DFlash2 checkpoint and its file read, started while admission and the target load run.
    Dflash(&'s super::dflash::Contract, std::thread::JoinHandle<std::io::Result<Vec<u8>>>),
}

/// The loaded coordinator that serve and golden drive.
pub(super) struct Runtime<'a> {
    pub execution: Execution<'a>,
    pub cache: Cache<'a>,
    pub embedding: TokenEmbedding<'a>,
    pub token_ids: Allocation<'a>,
    pub embedded: Allocation<'a>,
    /// RTX-local and Spark routed experts; `None` when the run executes no routed layer.
    pub routed: Option<Routed<'a>>,
}

/// The start-up after admission, shared by serve and golden: Spark worker admission runs while
/// the target weights, token embedding and local experts load, then the KV cache is allocated and
/// the Spark link connects (one connection per rank; one return slot per prefill lane).
pub(super) fn start<'a>(library_owner: &'a Arc<cuteafd_ffi::NativeLibrary>, device: i32,
    programs: &'a cuteafd_ffi::programs::Programs<'a>, start: Start<'_>) -> Result<Runtime<'a>> {
    let library = library_owner.as_ref();
    let Start { snapshot, checkpoint, cfg, sizing, ledger, cache, local_layers, spark_layers, peers, catalog,
        package, embed, available, draft } = start;
    let config = TcpTransportConfig { timing: cuteafd_transport::protocol_v2_timing_from_env(), ..Default::default() };
    // One link carries decode and every prefill wave: one persistent client per rank, its endpoint a
    // unit's rows, and an execution slot per wave in flight, as the planner charges every rank.
    let model = admission::spark_model(sizing)?;
    let endpoint = sizing.spark_endpoint()?;
    let pending = if spark_layers.is_empty() { None } else {
        let working = SparkWorkingSet { rtx_layers: local_layers.to_vec(), spark_layers: spark_layers.to_vec(),
            spark_ranks: peers.len(), prefill_lanes: 1 };
        Some(SparkAdmission::plan(snapshot, ExpertGeometry::MINIMAX_M3, Some(catalog), &working, peers.len(), None,
            model)?.start(peers)?)
    };
    let buffer = |name: &str| -> Result<Allocation<'a>> {
        let bytes = ledger.bytes(name).with_context(|| format!("MiniMax {name} admission"))?;
        Allocation::new(Device { library, id: device }, usize::try_from(bytes)?)
    };
    let (token_ids, embedded) = (buffer("token_ids")?, buffer("embedded")?);
    let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == super::weights::EMBEDDING)
        .context("MiniMax embedding header missing")?;
    let source = EmbedSource::new(snapshot, &tensor.shard, &tensor.meta, cfg.hidden)?;
    let (dspark, dflash) = match draft {
        Some(DraftStart::Dspark(contract, layout)) => (Some((contract, layout)), None),
        Some(DraftStart::Dflash(contract, file)) => (None, Some((contract, file))),
        None => (None, None),
    };
    let (embedding, (mut execution, local)) = crate::shared::placement::load_while_admitting(pending, || {
        TokenEmbedding::load(library, source, embed, || {
            let mut execution = Execution::load(Device { library, id: device }, programs, checkpoint, cfg, ledger, sizing)?;
            if let Some((contract, layout)) = dspark { execution.load_draft(contract, layout)?; }
            let local = if local_layers.is_empty() {None} else {
                let tensors = catalog.fp8().context("MiniMax native expert catalog missing")?.for_coordinator();
                Some(crate::shared::experts::fp8::Fp8Experts::load(library, &tensors, package,
                    local_layers, 1, 0, sizing.unit_rows, usize::try_from(available)?)?)
            };
            Ok((execution, local))
        })
    })?;
    // The DFlash2 drafter loads once the embedding is resident (its mask row), before the pool.
    if let Some((contract, file)) = dflash {
        let file = file.join().map_err(|_| anyhow::anyhow!("MiniMax DFlash2 drafter read panicked"))??;
        let mask_row = embedding.host_rows(&[contract.cfg.mask_token])?;
        execution.load_dflash(contract, file, mask_row)?;
    }
    let mut kv = Cache::new(library, device, &cache.cache, cache.pool_pages as usize, sizing.sequences,
        cache.cache_bytes.checked_add(256).context("MiniMax cache reserve overflow")?)?;
    kv.context = kv.context.min(cache.pool_tokens as usize);
    let spark = match spark_layers.first() {
        None => None,
        Some(&first) => {
            cuteafd_transport::worker_selection::require_peer_clients(peers, endpoint, 1)?;
            Some(super::routed::Spark::connect(library_owner, device, SparkWorld::canonical(peers.len())?, peers,
                sizing.unit_rows, config, first, request, sizing.waves)?)
        }
    };
    let routed = if local.is_none() && spark.is_none() {None} else {Some(Routed::new(local, spark)?)};
    Ok(Runtime { execution, cache: kv, embedding, token_ids, embedded, routed })
}

type Ready = (usize, Option<Arc<cuteafd_api::openai::media::MediaPreparer>>);

fn worker(args: &ServeArgs, receive: &mut tokio::sync::mpsc::Receiver<cuteafd_api::openai::NativeRequest>,
    stats: &Arc<Mutex<serde_json::Value>>, eos: Vec<u32>, initialized: impl FnOnce(Ready) -> Result<()>) -> Result<()> {
    let local_layers = args.validate()?.map_or(admission::LocalLayers::Fewest, admission::LocalLayers::Bottom);
    let checkpoint = Checkpoint::coordinator(&args.snapshot, false, false)?;
    let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
    cfg.release_geometry()?;
    cuteafd_core::set_expert_geometry(ExpertGeometry::MINIMAX_M3)
        .map_err(|_| MiniMaxError::Invalid("expert geometry already installed"))?;
    let library = Arc::new(cuteafd_ffi::minimax::load_library(&args.native_lib)?);
    library.cuda_set_device(args.device)?;
    super::routed::require_device_landing(&library, args.device)?;
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    programs.load_matching(|name| name.starts_with("minimax_"))?;
    let (rows, sequences) = (args.prefill_rows as usize, args.max_sequences as usize);
    // The prefill shape the launch resolved (the plan's units): `--prefill-waves` units of
    // `--prefill-rows` in flight. Head rows follow once the drafter's block is known.
    let mut sizing = RuntimeSizing { unit_rows: rows, decode_rows: rows.min(cuteafd_loader::plan::families::minimax::admission::DECODE_ROWS),
        sequences, context: if args.max_context == 0 {cfg.max_position as u64} else {args.max_context},
        spark_ranks: args.peers.len(), logit_rows: 0, waves: args.prefill_waves };
    let span = sizing.slot_rows();
    let rings = args.draft_context_slots.unwrap_or_else(|| cuteafd_loader::plan::families::minimax::admission::default_draft_rings(
        sequences, rows.min(cuteafd_loader::plan::families::minimax::admission::DECODE_ROWS)));
    let draft_contract=if args.speculator=="dspark" {Some(cuteafd_loader::families::minimax_m3::dspark::DsparkContract::open(&args.snapshot)?)} else {None};
    let draft_layout=draft_contract.as_ref().map(|c|super::dspark::Layout::new(&library,c,rings,span)).transpose()?;
    let dflash=args.draft.as_deref().map(|snapshot| -> Result<_> {
        let head = checkpoint.tensors.iter().find(|t| t.meta.name == "language_model.lm_head.weight")
            .context("MiniMax lm_head header missing")?;
        super::dflash::Contract::open(snapshot, &head.meta, args.draft_fp8, args.draft_kv_cache, rings,
            library.sm_count()?, args.draft_causal, args.draft_selector_score == "fp32")
    }).transpose()?;
    // The drafter's 4 GB checkpoint is read while admission and the target load run.
    let dflash_file=dflash.as_ref().map(|c| crate::shared::speculation::dflash::prefetch(&c.snapshot));
    let (draft_bytes,draft_rings,draft_staging)=match (&draft_layout,&dflash) {
        (Some(l),_) => (l.bytes,l.slots,0),
        (None,Some(c)) => (c.bytes,c.slots,c.staging),
        (None,None) => (0,0,0),
    };
    sizing.logit_rows = RuntimeSizing::head_rows(span, args.full_prefill_logits, draft_layout.as_ref().map_or(0, |l| l.sequences * 7));
    let sizing = sizing;
    let headroom = (args.headroom_gib * (1u64 << 30) as f64) as u64;
    let manifest = cuteafd_loader::plan::checkpoint::read_json(&args.manifest)?;
    let (package, local_package) = admission::local_package(&args.native_lib)?;
    // The coordinator GPU budget is already applied to the free bytes it reports.
    let available = || -> Result<u64> {
        library.cuda_set_device(args.device)?;
        Ok(library.cuda_memory_info()?.0 as u64)
    };
    let remote = crate::shared::vision::startup::RemotePeers::parse(args.vision, args.vision_peers.as_deref(),
        args.encoder_plan_hash.as_deref(), args.encoder_revision.as_deref())?;
    // `auto`, like the other families: a checkpoint that bundles the tower and names no Spark
    // peers keeps it on the coordinator GPU, and the KV pool takes what is left. The text-only
    // pool is admitted first so startup logs what the tower costs.
    let auto_local = args.vision == MediaMode::Auto && remote.is_none();
    let vision_mode = if auto_local && cuteafd_loader::families::minimax_m3::config::MiniMaxM3VisionConfig::read(&args.snapshot)?.is_none() {
        MediaMode::Off
    } else { args.vision };
    let request = admission::Request { sizing, kv_format: args.kv_cache, local_layers, spark_budget: args.spark_budget_bytes,
        spark_vision_bytes: 0, headroom, draft_bytes, draft_rings, draft_staging,
        host_embedding: args.token_io.embed_placement == EmbedPlacement::Host,
        startup_graphs: false, rings: admission::spark_rings()? };
    let text_admission = (auto_local && vision_mode == MediaMode::Auto).then(|| -> Result<_> {
        admission::solve(&checkpoint, &manifest, local_package.as_ref(), request.clone(), available()?, None)
    }).transpose()?;
    let (vision, prefix) = ReadyVision::load(&args.snapshot, &args.native_lib, &library, args.device,
        vision_mode, &args.prefix, args.media_cache_bytes, remote)?;
    let available = available()?;
    let admitted = if let Some(mut text) = text_admission {
        let text_pool = text.cache.pool_tokens;
        text.readmit(&cfg, sizing, available, None, args.kv_cache).context("the bundled vision tower leaves no room \
            for a KV pool on the coordinator GPU: pass --vision off, or host it on Sparks with --vision-peers")?;
        tracing::info!(text_pool_tokens = text_pool, pool_tokens = text.cache.pool_tokens,
            tower_kv_tokens = text_pool.saturating_sub(text.cache.pool_tokens),
            "MiniMax automatic vision keeps the bundled tower on the coordinator GPU; the KV pool takes what is left");
        if args.pool_tokens.is_some() { text.readmit(&cfg, sizing, available, args.pool_tokens, args.kv_cache)?; }
        text
    } else {
        admission::solve(&checkpoint, &manifest, local_package.as_ref(), admission::Request {
            spark_vision_bytes: vision.as_ref().map_or(0, |v| v.spark_reserve_bytes), ..request }, available, args.pool_tokens)?
    };
    let mut catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    let Runtime { execution, cache, embedding, token_ids, embedded, routed } = start(&library, args.device, &programs, Start {
        snapshot: &args.snapshot, checkpoint: &checkpoint, cfg: &cfg, sizing, ledger: &admitted.ledger,
        cache: &admitted.cache, local_layers: &admitted.local_layers, spark_layers: &admitted.spark_layers,
        peers: &args.peers, catalog: &catalog, package: &package, embed: args.token_io.embed_placement, available,
        draft: match (draft_contract.as_ref().zip(draft_layout), dflash.as_ref().zip(dflash_file)) {
            (Some((contract, layout)), _) => Some(DraftStart::Dspark(contract, layout)),
            (None, Some((contract, file))) => Some(DraftStart::Dflash(contract, file)),
            (None, None) => None,
        } })?;
    let experts = routed.context("MiniMax serving admitted no routed expert owner")?;
    let expander = vision.as_ref().map(|_| cuteafd_loader::media::minimax::span_expander(&args.snapshot, cfg.vocab_size as u32)).transpose()?;
    let backend = MiniMaxBackend { execution: &execution, cache: &cache, embedding: &embedding, token_ids, embedded,
        experts: RefCell::new(experts), rows: span, vocabulary: cfg.vocab_size, expander,
        pending: RefCell::new(None), timing: RefCell::new([0.0; 2]), probe: RefCell::new(crate::shared::draft::routes::RoundRoutes::new(1,4)),
        rings: RefCell::new(super::backend::Rings::new(sizing.sequences, rings)), full_prefill_logits: args.full_prefill_logits,
        trace: RefCell::new(super::diagnostics::LogitTrace::from_env()?),
        startup_captures: std::cell::Cell::new(None),
        selector: match execution.draft() {
            Ok(super::execution::Drafter::Dflash(draft)) => Some(draft.selector_prior()?),
            _ => None,
        } };
    // Decode runs eagerly: the Spark exchange's stream waits and writes are never captured, so no
    // whole step is recorded and serving captures nothing after this point.
    anyhow::ensure!(admitted.graphs.keys.is_empty() && admitted.graphs.inventory.bytes(0) == 0,
        "MiniMax decode runs eagerly and admits no startup decode graphs");
    backend.startup_captures.set(Some(cuteafd_ffi::graph_captures()));
    let (encoder, preparer, media_bytes) = vision.map_or((Encoder::Off, None, 0), |v| (v.encoder, Some(v.preparer), v.cache_bytes));
    let mut media = cuteafd_engine::media::MediaAdmission::new(cuteafd_engine::media::EmbeddingCache::new(media_bytes), encoder, 16);
    tracing::info!(pool_tokens=admitted.cache.pool_tokens, context=cache.context, local_layers=?admitted.local_layers,
        unit_rows=sizing.unit_rows, prefill_waves=sizing.waves, span_rows=span, logit_rows=sizing.logit_rows, draft_rings,
        speculator=%args.speculator, draft_bytes,
        "MiniMax native target; decode runs eagerly, prefill uses the MSA index-prefill and attention-extend kernels");
    let drafts=if args.speculator!="off" {
        scheduler::Drafts::Neural {depth:args.draft_fixed.map_or(7,usize::from),fixed:args.draft_fixed.is_some()}
    } else {scheduler::Drafts::None};
    scheduler::schedule(&backend, &args.snapshot, receive, stats, sizing.sequences, drafts,
        eos, args.decode_share, &prefix, args.token_io.token_select, &mut media, None,
        || initialized((backend.geometry().max_context, preparer)))
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    use cuteafd_api::openai::{ModelEncoding, ModelProfile, NativeLimits};
    args.validate()?;
    let snapshot = args.snapshot.clone();
    let encoding = ModelEncoding::MiniMaxM3(Arc::new(cuteafd_api::openai::chat::minimax_m3::MiniMaxEncoding::from_snapshot(&snapshot)?));
    let eos = encoding.eos_token_ids();
    let model = args.model_id.clone().or_else(|| cuteafd_api::openai::snapshot_model_id(&snapshot))
        .context("MiniMax requires --model-id for a snapshot without HF identity")?;
    let profile = ModelProfile::new(model, encoding);
    let api = args.api.load()?;
    let engine_args = args.clone();
    crate::shared::api::serve_native("MiniMax", "minimax_m3", &args.listen, &snapshot, api,
        move |receive, stats, ready| worker(&engine_args, receive, stats, eos, |state| { ready(state); Ok(()) }),
        |(context, vision): Ready| {
            let profile = match vision { Some(preparer) => profile.with_loaded_vision(preparer), None => profile };
            Ok((profile, NativeLimits::new(u32::try_from(context)?, args.max_output.min(context as u32))?))
        }).await
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
        let mut a=args(); a.kv_cache=Qwen4KvCache::Bf16; assert!(a.validate().unwrap_err().downcast_ref::<MiniMaxError>().is_some());
        let mut a=args(); a.speculator="mtp".into(); assert!(a.validate().is_err());
        assert_eq!(args().speculator,"dspark");
        let mut a=args(); a.speculator="off".into(); assert!(a.validate().is_ok());
        // DFlash2 takes its snapshot with --draft, and only DFlash2 does.
        let mut a=args(); a.speculator="dflash2".into(); assert!(a.validate().is_err());
        a.draft=Some("/draft".into()); assert!(a.validate().is_ok());
        a.speculator="dspark".into(); assert!(a.validate().is_err());
        a.speculator="off".into(); assert!(a.validate().is_err());
        assert_eq!((args().draft_fp8, args().draft_kv_cache), (None, Qwen4KvCache::Fp8));
        let mut a=args(); a.token_io.embed_placement=EmbedPlacement::Host; assert!(a.validate().is_ok());
        let mut a=args(); a.rtx_expert_layers="58".into(); assert!(a.validate().is_err());
        let mut a=args(); a.rtx_expert_layers="3".into(); assert_eq!(a.validate().unwrap(),Some(3));
        assert!(args().draft_fixed.is_none());
    }
    #[test]
    fn minimax_dflash2_takes_its_snapshot_precision_and_ring_storage() {
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "serve-minimax", "--snapshot", "/model",
            "--native-lib", "/lib.so", "--peers", "127.0.0.1:1,127.0.0.1:2", "--speculator", "dflash2",
            "--draft", "/draft", "--draft-fp8", "false", "--draft-kv-cache", "bf16", "--draft-fixed", "7"]).unwrap();
        let crate::cli::Commands::ServeMiniMax(a) = cli.command else { panic!("MiniMax dispatch") };
        assert_eq!((a.speculator.as_str(), a.draft.as_deref()), ("dflash2", Some(Path::new("/draft"))));
        assert_eq!((a.draft_fp8, a.draft_kv_cache, a.draft_fixed), (Some(false), Qwen4KvCache::Bf16, Some(7)));
        assert_eq!(a.draft_selector_score, "fp32");
        assert!(a.validate().is_ok());
        assert!(crate::cli::Cli::try_parse_from(["cuteafd", "serve-minimax", "--snapshot", "/model",
            "--native-lib", "/lib.so", "--peers", "127.0.0.1:1,127.0.0.1:2", "--speculator", "eagle"]).is_err());
    }

    #[test]
    fn minimax_fixed_draft_depth_is_one_to_the_drafter_block() {
        let parse = |depth: &str| crate::cli::Cli::try_parse_from(["cuteafd", "serve-minimax", "--snapshot", "/model",
            "--native-lib", "/lib.so", "--peers", "127.0.0.1:1,127.0.0.1:2", "--draft-fixed", depth]);
        let Ok(crate::cli::Cli { command: crate::cli::Commands::ServeMiniMax(a), .. }) = parse("3") else { panic!("depth 3") };
        assert_eq!(a.draft_fixed, Some(3));
        assert!(parse("0").is_err() && parse("8").is_err());
    }

    /// `--prefill-waves` is the waves the one Spark link keeps in flight (a lane each, today);
    /// `--prefill-lanes` stays a deprecated alias of it.
    #[test]
    fn minimax_prefill_waves_take_the_deprecated_lane_alias() {
        let waves = |extra: &[&str]| match crate::cli::Cli::try_parse_from(["cuteafd", "serve-minimax", "--snapshot",
            "/model", "--native-lib", "/lib.so", "--peers", "127.0.0.1:1,127.0.0.1:2"].into_iter().chain(extra.iter().copied())) {
            Ok(crate::cli::Cli { command: crate::cli::Commands::ServeMiniMax(a), .. }) => Some((a.prefill_rows, a.prefill_waves)),
            _ => None,
        };
        assert_eq!(waves(&[]), Some((4096, prefill_lanes::DEFAULT_PREFILL_LANES)));
        assert_eq!(waves(&["--prefill-rows", "2048", "--prefill-waves", "3"]), Some((2048, 3)));
        assert_eq!(waves(&["--prefill-lanes", "1"]), Some((4096, 1)));
        for refused in ["0", "5"] { assert_eq!(waves(&["--prefill-waves", refused]), None, "{refused} waves"); }
    }

    #[test]
    fn minimax_serves_two_three_four_or_six_distinct_spark_ranks() {
        assert_eq!(admission::spark_worlds(), [2, 3, 4, 6]);
        for count in 0u16..=8 {
            let mut a=args(); a.peers=(1..=count).map(|port| SocketAddr::new(std::net::Ipv4Addr::LOCALHOST.into(), port)).collect();
            assert_eq!(a.validate().is_ok(), admission::spark_worlds().contains(&(count as usize)), "{count} peers");
        }
        let mut a=args(); a.peers=vec![a.peers[0]; 6]; assert!(a.validate().is_err());
    }
}
