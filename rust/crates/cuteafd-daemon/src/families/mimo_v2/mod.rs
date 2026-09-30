//! MiMo V2 (mimo_v2_flash) on the generic engine: weights, the coordinator
//! programs' layer chain, and the golden comparison command.
pub(crate) mod dflash;
pub(crate) mod engine;
pub(crate) mod mtp;
pub(crate) mod serve;
pub(crate) mod weights;

use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::mimo_v2::MimoV2Config;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::time::Instant;

/// What every MiMo command needs to stand up the engine.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct EngineArgs {
    /// Checkpoint snapshot (XiaomiMiMo/MiMo-V2-Flash).
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/DSV4_PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// Run only the first N layers (layer 0 is the only dense layer).
    #[arg(long)]
    pub layers: Option<usize>,
    /// Longest sequence (the RoPE tables and page tables).
    #[arg(long, default_value_t = 32768)]
    pub max_context: usize,
    /// Tokens the full-attention record pool holds across sequences.
    #[arg(long, default_value_t = 131_072)]
    pub pool_tokens: usize,
    /// Sequences with a sliding-window ring.
    #[arg(long, default_value_t = 16)]
    pub rings: usize,
    #[arg(long, default_value_t = 4096)]
    pub prefill_rows: usize,
    /// Spark ranks in TP order (HOST:PORT,...) serving the fp8 expert family, for MoE layers.
    #[arg(long, conflicts_with = "local_experts")]
    pub peers: Option<String>,
    /// Run the MoE layers' routed experts on this GPU (TP1 FP8 package; all
    /// resident layers' experts must fit: 6.4 GiB each).
    #[arg(long)]
    pub local_experts: bool,
    /// DFlash drafter: a directory with dflash_draft_model.safetensors, or a
    /// snapshot with one under dflash/ (MiMo V2.6 Pro); taps the target
    /// layers and drafts on this GPU.
    #[arg(long)]
    pub draft: Option<PathBuf>,
    /// Native MTP drafter: run this many of the checkpoint's MTP stages
    /// (`model.mtp.layers.*`, 0 = off) after each next token.
    #[arg(long, default_value_t = 0)]
    pub mtp: usize,
    /// Sequences the drafter keeps a context for and drafts for at once.
    #[arg(long, default_value_t = 4)]
    pub draft_sequences: usize,
    /// Draft through E4M3 copies of the drafter's GEMM weights and of the LM
    /// head (false: BF16; the committed tokens are the same either way).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub draft_fp8: bool,
    /// Scale rule of the FP8 copies made from BF16 weights at load: amax /
    /// 448, the smallest power of two >= it (pow2), or per block whichever of
    /// the two leaves the smaller error (best).
    #[arg(long, value_enum, default_value_t = crate::shared::fp8_linear::Fp8Scales::Amax)]
    pub fp8_scales: crate::shared::fp8_linear::Fp8Scales,
    #[command(flatten)]
    pub l2: crate::shared::l2_prefetch::L2PrefetchArgs,
    /// With --local-experts: keep only N MoE layers' experts resident and load
    /// each missing layer over the oldest (prefill checks of models whose
    /// experts do not fit one GPU, such as V2.6 Pro).
    #[arg(long, requires = "local_experts")]
    pub expert_window: Option<usize>,
    /// Skip the routed experts (MoE layers add zero): times the coordinator
    /// path alone; logits and cosines are meaningless.
    #[arg(long, conflicts_with_all = ["local_experts", "peers"])]
    pub skip_experts: bool,
    /// TP1 FP8 package layout for --local-experts (default `<libdir>/fp8/fp8-mimo/tp1`).
    #[arg(long)]
    pub fp8_package: Option<PathBuf>,
    /// Decode steps of <= 16 rows read E4M3 copies of the qkv, o and dense FFN
    /// weights (the checkpoint's own FP8 bytes; o_proj quantized per row).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub fp8_decode: bool,
    /// Decode steps of <= 16 rows read an E4M3 copy of the LM head (per row x 128-K scales).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub fp8_head: bool,
    /// With --fp8-decode, also an E4M3 copy of o_proj (BF16 in the checkpoint,
    /// quantized per row and 128-K block at load).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub fp8_o_proj: bool,
    /// Expert input rows sent to the Spark ranks: FP8 K32 wire rows, BF16
    /// (the ranks load the `fp8-mimo-bf16` package too), or BF16 for decode
    /// steps only.
    #[arg(long, value_enum, default_value_t = engine::ExpertInput::Fp8)]
    pub expert_input: engine::ExpertInput,
}

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Directory with tokens.bin, layerNN.bin and logits.bin from
    /// python/reference/families/mimo_v2/mimo_v2/golden.py.
    #[arg(long)]
    pub golden: PathBuf,
    /// Prefill only the first N tokens, then feed the rest through decode
    /// steps of --step-rows rows (teacher-forced), comparing their rows.
    #[arg(long)]
    pub prefill: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub step_rows: usize,
    /// Feed each layer the golden output of the previous one (prefill only),
    /// so every layer's cosine measures that layer alone.
    #[arg(long)]
    pub teacher_force: bool,
    /// Prefill in chunks of this many rows (checks prefill after cached rows).
    #[arg(long)]
    pub prefill_chunk: Option<usize>,
    /// Score every prefill row against the golden logits: top-1 agreement,
    /// next-token accuracy and mean NLL of the prompt's next tokens (the
    /// golden's `mean_nll`).
    #[arg(long)]
    pub nll: bool,
    /// Skip the per-layer comparison (each layer's rows are otherwise copied
    /// back), so prefill and decode times are the engine's.
    #[arg(long)]
    pub timing: bool,
    /// With --draft: run only the drafter on the golden taps at the anchors of
    /// python/reference/families/mimo_v2/mimo_dflash/reference.py's output directory, compare
    /// drafts, and score them against the golden text and greedy targets.
    #[arg(long, requires = "draft")]
    pub draft_oracle: Option<PathBuf>,
    /// With --draft: replay the drafter alone on the golden taps, drafting
    /// after every token from this position on, BF16 and FP8 (acceptance
    /// against the text and the golden greedy picks, draft time).
    #[arg(long)]
    pub draft_replay: Option<usize>,
    /// With --mtp: run only the MTP drafter on the golden's last-layer rows at
    /// the anchors of python/reference/families/mimo_v2/mimo_mtp/reference.py's output
    /// directory, compare with its predictions and score acceptance.
    #[arg(long)]
    pub mtp_oracle: Option<PathBuf>,
}

/// The checkpoint and native library, opened on the calling thread.
pub(crate) struct Opened {
    pub catalog: cuteafd_loader::OfficialV41Catalog,
    pub checkpoint: Checkpoint,
    pub cfg: MimoV2Config,
    pub library: NativeLibrary,
}

pub(crate) fn open(args: &EngineArgs) -> Result<Opened> {
    let checkpoint = Checkpoint::open(&args.snapshot)?;
    ensure!(checkpoint.missing_shards.is_empty(), "checkpoint shards missing: {:?}", checkpoint.missing_shards);
    let cfg = MimoV2Config::read(&args.snapshot)?;
    // The expert geometry is process-wide and must be fixed before the native
    // library loads (its expert helpers size rows from it).
    let catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    cuteafd_core::set_expert_geometry(catalog.routed_experts().geometry()?)
        .map_err(|g| anyhow::anyhow!("expert geometry already {g:?}"))?;
    // SAFETY: the library is the cuteafd native shim built for this engine.
    let library = unsafe { NativeLibrary::load(&args.native_lib) }?;
    library.cuda_set_device(args.device)?;
    Ok(Opened { catalog, checkpoint, cfg, library })
}

impl Opened {
    /// Builds the engine and hands it to `body`.
    pub fn with_engine<T>(&self, args: &EngineArgs, body: impl FnOnce(&engine::MimoEngine<'_>) -> Result<T>) -> Result<T> {
        let programs = self.library.dsv4_programs()?.with_manifest(&args.manifest)?;
        programs.load_all()?;
        let draft_dir = args.draft.as_deref().map(dflash::drafter_dir);
        let draft_file = draft_dir.as_deref().map(dflash::prefetch);
        let stream = self.library.cuda_stream_create()?;
        let started = Instant::now();
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let loader = weights::MimoLoader { library: &self.library, checkpoint: &self.checkpoint, stream,
            checkpoint_tp: cuteafd_loader::mimo_v2::checkpoint_tp(&args.snapshot)?, fp8_decode: args.fp8_decode,
            fp8_head: args.fp8_head, fp8_o_proj: args.fp8_o_proj, fp8_scales: args.fp8_scales };
        let model = loader.model(&self.cfg, layers)?;
        let mtp = if args.mtp > 0 {
            let started = Instant::now();
            let available = self.checkpoint.tensors.iter()
                .filter(|t| t.meta.name.starts_with("model.mtp.layers.") && t.meta.name.ends_with(".eh_proj.weight"))
                .count();
            ensure!(args.mtp <= available, "--mtp {} but the checkpoint has {available} MTP layers", args.mtp);
            let zeroed = |bytes: usize| -> Result<crate::shared::memory::DeviceAllocation<'_>> {
                let allocation = crate::shared::memory::DeviceAllocation::new(&self.library, bytes.max(256))?;
                self.library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
                Ok(allocation)
            };
            let (h, record) = (self.cfg.hidden, self.cfg.record_elems(cuteafd_loader::mimo_v2::MimoAttention::Sliding));
            let stages = (0..args.mtp).map(|k| -> Result<mtp::MtpStage<'_>> {
                let [eh, enorm, hnorm, final_norm] = loader.mtp_extras(k)?;
                Ok(mtp::MtpStage { layer: loader.mtp_layer(&self.cfg, k)?, eh, enorm, hnorm, final_norm,
                    ring: zeroed(args.rings * engine::RING_ROWS * record * 2)? })
            }).collect::<Result<Vec<_>>>()?;
            let rows = engine::DECODE_ROWS;
            let drafter = mtp::MtpDrafter { stages, hidden: zeroed(args.rings * mtp::HIDDEN_ROWS * h * 2)?,
                embed: zeroed(rows * h * 2)?, rows_h: zeroed(rows * h * 2)?, normed_e: zeroed(rows * h * 2)?,
                normed_h: zeroed(rows * h * 2)?, cat: zeroed(rows * 2 * h * 2)?,
                ext: std::cell::RefCell::new(vec![vec![0; args.mtp]; args.rings]) };
            tracing::info!(stages = args.mtp, elapsed_ms = started.elapsed().as_millis() as u64, "MTP drafter resident");
            Some(drafter)
        } else {
            None
        };
        tracing::info!(layers, elapsed_ms = started.elapsed().as_millis() as u64, "MiMo coordinator weights resident");
        let pages = args.pool_tokens.div_ceil(engine::PAGE_ROWS);
        let mut engine = engine::MimoEngine::new(&self.library, &programs, self.cfg.clone(), model, stream,
            args.max_context, args.prefill_rows, pages, args.rings)?;
        engine.mtp = mtp;
        if let Some(dir) = &draft_dir {
            let started = Instant::now();
            let cfg = dflash::DflashConfig::read(dir)?;
            ensure!(cfg.hidden == self.cfg.hidden && cfg.vocab == self.cfg.vocab_size
                && cfg.taps.iter().all(|&l| l < self.cfg.layers), "the DFlash drafter does not fit this target");
            let mask = match dflash::mask_embedding(dir, cfg.hidden)? {
                Some(row) => row,
                None => embed_rows(&self.checkpoint, &[cfg.mask_token], cfg.hidden)?,
            };
            let file = draft_file.context("drafter prefetch")?.join()
                .map_err(|_| anyhow::anyhow!("drafter prefetch panicked"))??;
            let mut drafter = dflash::MimoDrafter::load(&self.library, dir, file, stream, args.draft_sequences,
                args.draft_sequences, mask)?;
            if args.draft_fp8 {
                drafter.enable_fp8(engine.weights.head.buffer.ptr, args.fp8_scales)?;
            }
            engine.drafter = Some(drafter);
            tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "DFlash drafter resident");
        }
        let moe_layers = (0..layers).filter(|&l| !self.cfg.dense[l]).collect::<Vec<_>>();
        engine.expert_input = args.expert_input;
        if let Some(experts) = self.experts(args, &moe_layers)? {
            engine.set_experts(experts);
        }
        if let Some(budget) = args.l2.budget(&self.library, crate::shared::l2_prefetch::OTHER_DEFAULT)? {
            engine.l2 = Some(crate::shared::l2_prefetch::L2Prefetch::new(&self.library, budget, &engine.decode_read_order())?);
        }
        let result = body(&engine);
        drop(engine);
        // SAFETY: the engine that used the stream is gone.
        unsafe { self.library.cuda_stream_destroy(stream)? };
        result
    }
}

impl Opened {
    /// The routed-expert source for `moe_layers`: Spark ranks (`--peers`,
    /// warmed with a full-capacity request) or the local TP1 package.
    fn experts<'s>(&'s self, args: &EngineArgs, moe_layers: &[usize]) -> Result<Option<engine::Experts<'s>>> {
        if moe_layers.is_empty() {
            return Ok(None);
        }
        if args.skip_experts {
            return Ok(Some(engine::Experts::Skip));
        }
        if args.local_experts {
            let tensors = self.catalog.fp8().context("MiMo experts are the checkpoint's FP8 tensors")?;
            let directory = args.fp8_package.clone()
                .unwrap_or_else(|| crate::shared::experts::fp8::package_directory(&args.native_lib, 1));
            let (free, _) = self.library.cuda_memory_info()?;
            if let Some(window) = args.expert_window {
                let experts = crate::shared::experts::fp8::Fp8Experts::load(&self.library, tensors, &directory, 0..0, 1, 0,
                    args.prefill_rows, free.saturating_sub(4 << 30))?;
                return Ok(Some(engine::Experts::Streamed { experts: std::cell::RefCell::new(experts), tensors, window }));
            }
            let started = Instant::now();
            let local = crate::shared::experts::fp8::Fp8Experts::load(&self.library, tensors, &directory,
                moe_layers[0]..moe_layers[moe_layers.len() - 1] + 1, 1, 0, args.prefill_rows,
                free.saturating_sub(4 << 30))?;
            tracing::info!(layers = moe_layers.len(), elapsed_ms = started.elapsed().as_millis() as u64,
                "MiMo FP8 experts resident on this GPU");
            return Ok(Some(engine::Experts::Local(local)));
        }
        let Some(peers) = args.peers.as_deref() else { return Ok(None) };
        let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
        let executors: Vec<u64> = (0..peers.len())
            .map(|rank| cuteafd_transport::v41_expert::v41_spark_executor_id(peers.len(), rank))
            .collect::<Result<_>>()?;
        let mut transport = crate::shared::spark_intake::SparkLink::new(&self.library, &peers, &executors,
            u32::try_from(args.prefill_rows)?, cuteafd_transport::TcpTransportConfig { timing: false,
                timeout: std::time::Duration::from_secs(120), max_frame_bytes: 64 << 20 }, self.cfg.hidden * 2)?;
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        // Connect every rank and register full-size buffers now: the first
        // request otherwise pays seconds of connection setup.
        let started = Instant::now();
        // The first request sizes the registered rings: the widest one first
        // (BF16 rows when prefill sends them), then a one-row BF16 request
        // when decode does, so a rank without the BF16 package fails here.
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let warm = |rows: usize, bf16: bool| -> Result<cuteafd_transport::ExpertProtocolV2Request> {
            let routes = (0..rows * topk).map(|i| cuteafd_transport::ExpertProtocolV2RouteEntry {
                row_index: (i / topk) as u32, expert_id: (i % self.cfg.experts) as u32, gate_weight: 0.0,
            }).collect();
            let (dtype, row_bytes) = if bf16 { (cuteafd_transport::ExpertV2Dtype::Bf16, 2 * h) }
                else { (cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32, h + h / 32) };
            let mut request = cuteafd_transport::ExpertProtocolV2Request::new(1, 17, moe_layers[0] as u32, h as u32,
                dtype,
                (0..rows as u32).map(|row| cuteafd_transport::ExpertProtocolV2RowDescriptor {
                    row_id: u64::from(row), source_kind: cuteafd_transport::ExpertV2SourceKind::Prefill,
                    source_request_id: 1, token_position: u64::from(row), route_offset: row * topk as u32,
                    route_count: topk as u32,
                }).collect(),
                routes, vec![0; rows * row_bytes])?;
            request.header.flags |= cuteafd_transport::v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
            Ok(request)
        };
        let mut warmups = vec![warm(args.prefill_rows, args.expert_input.bf16(false))?];
        if args.expert_input.bf16(true) && !args.expert_input.bf16(false) {
            warmups.push(warm(1, true)?);
        }
        let warm_stream = self.library.cuda_stream_create()?;
        for request in &warmups {
            runtime.block_on(async {
                let wave = transport.dispatch(request)?;
                transport.receive(wave, request.header.row_count as usize, warm_stream).await
            })?;
        }
        // SAFETY: the stream was created above; its waits drain before it goes.
        unsafe {
            self.library.cuda_stream_synchronize(warm_stream)?;
            self.library.cuda_stream_destroy(warm_stream)?;
        }
        tracing::info!(ranks = peers.len(), elapsed_ms = started.elapsed().as_millis() as u64, "Spark expert transport warm");
        Ok(Some(engine::Experts::Spark { transport: std::cell::RefCell::new(transport), runtime }))
    }
}

/// The embedding table's shard, opened once (serving reads rows every step).
pub(crate) struct Embeddings {
    file: std::fs::File,
    offset: u64,
    row: usize,
}

impl Embeddings {
    pub fn open(checkpoint: &Checkpoint, hidden: usize) -> Result<Self> {
        let name = "model.embed_tokens.weight";
        let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("checkpoint has no {name}"))?;
        let tensor = &checkpoint.tensors[at];
        Ok(Self { file: std::fs::File::open(checkpoint.snapshot.join(&tensor.shard))?, offset: tensor.meta.byte_offset,
            row: hidden * 2 })
    }

    /// BF16 rows of `tokens`.
    pub fn rows(&self, tokens: &[u32]) -> Result<Vec<u8>> {
        let mut out = vec![0u8; tokens.len() * self.row];
        for (slot, token) in out.chunks_exact_mut(self.row).zip(tokens) {
            self.file.read_exact_at(slot, self.offset + u64::from(*token) * self.row as u64)?;
        }
        Ok(out)
    }
}

pub(crate) fn embed_rows(checkpoint: &Checkpoint, tokens: &[u32], hidden: usize) -> Result<Vec<u8>> {
    let name = "model.embed_tokens.weight";
    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
        .map_err(|_| anyhow::anyhow!("checkpoint has no {name}"))?;
    let tensor = &checkpoint.tensors[at];
    let file = std::fs::File::open(checkpoint.snapshot.join(&tensor.shard))?;
    let row = hidden * 2;
    let mut out = vec![0u8; tokens.len() * row];
    for (slot, token) in out.chunks_exact_mut(row).zip(tokens) {
        file.read_exact_at(slot, tensor.meta.byte_offset + u64::from(*token) * row as u64)?;
    }
    Ok(out)
}

fn bf16s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect()
}

fn similarity(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
        diff += (x - y) * (x - y);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE), diff.sqrt() / nb.sqrt().max(f64::MIN_POSITIVE))
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || golden(args)).await?
}

fn golden(args: GoldenArgs) -> Result<()> {
    let opened = open(&args.engine)?;
    opened.with_engine(&args.engine, |engine| golden_run(&args, &opened, engine))
}

fn golden_run(args: &GoldenArgs, opened: &Opened, engine: &engine::MimoEngine<'_>) -> Result<()> {
    if let Some(dir) = &args.draft_oracle {
        return draft_oracle(args, opened, engine, dir);
    }
    if let Some(start) = args.draft_replay {
        let drafter = engine.drafter.as_ref().context("--draft-replay needs --draft")?;
        let (tokens, greedy) = crate::families::glm5::dflash::golden_sequence(&args.golden, opened.cfg.vocab_size)?;
        let hidden = opened.cfg.hidden;
        let layers: Vec<Vec<u8>> = drafter.cfg.taps.iter()
            .map(|l| std::fs::read(args.golden.join(format!("layer{l:02}.bin")))).collect::<std::io::Result<_>>()?;
        let row = hidden * 2;
        let taps = |first: usize, n: usize| -> Result<Vec<u8>> {
            let mut taps = vec![0u8; n * layers.len() * row];
            for r in 0..n {
                for (i, layer) in layers.iter().enumerate() {
                    taps[(r * layers.len() + i) * row..][..row].copy_from_slice(&layer[(first + r) * row..][..row]);
                }
            }
            Ok(taps)
        };
        return crate::families::glm5::dflash::replay(drafter, &tokens, &greedy, &taps,
            &|t| embed_rows(&opened.checkpoint, t, hidden), engine.weights.head.buffer.ptr, start);
    }
    if let Some(dir) = &args.mtp_oracle {
        return mtp_oracle(args, opened, engine, dir);
    }
    let cfg = &opened.cfg;
    let layers = engine.weights.layers.len();
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let mut placement = engine::Allocator::new(engine.pages, engine.rings).admit(tokens.len())?;
    let embed = embed_rows(&opened.checkpoint, &tokens, cfg.hidden)?;
    let row = cfg.hidden * 2;
    let prefill = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
    // Rows [first, first + n) of layer `layer`'s golden output.
    let compare = |layer: usize, first: usize, stream: &[u8], worst: &mut Vec<f64>| -> Result<()> {
        let path = args.golden.join(format!("layer{layer:02}.bin"));
        if let Ok(golden) = std::fs::read(&path) {
            ensure!(golden.len() >= (first * row + stream.len()), "golden layer {layer} is short");
            let golden = &golden[first * row..][..stream.len()];
            let (cosine, rel) = similarity(&bf16s(stream), &bf16s(golden));
            if worst.len() <= layer {
                worst.resize(layer + 1, 1.0);
            }
            worst[layer] = worst[layer].min(cosine);
            if first == 0 {
                // Per-row cosines: route flips show as a few bad rows with the
                // median near 1; a systematic error moves the median.
                let (ours, theirs) = (bf16s(stream), bf16s(golden));
                let hidden = row / 2;
                let mut rows: Vec<f64> = ours.chunks_exact(hidden).zip(theirs.chunks_exact(hidden))
                    .map(|(a, b)| similarity(a, b).0).collect();
                rows.sort_by(f64::total_cmp);
                let bad = rows.iter().filter(|&&c| c < 0.999).count();
                println!("layer {layer:2}: cosine {cosine:.6} rel_l2 {rel:.3e} | rows: median {:.6} p1 {:.4} worst {:.4}, \
                    {bad} of {} below 0.999", rows[rows.len() / 2], rows[rows.len() / 100], rows[0], rows.len());
            }
        }
        Ok(())
    };
    let mut worst = Vec::new();
    let started = Instant::now();
    let forced = |layer: usize| -> Option<Vec<u8>> {
        std::fs::read(args.golden.join(format!("layer{layer:02}.bin"))).ok().map(|rows| rows[..prefill * row].to_vec())
    };
    let chunk = args.prefill_chunk.unwrap_or(prefill).clamp(1, engine.prefill_rows);
    ensure!(!args.teacher_force || chunk >= prefill, "--teacher-force needs a single prefill chunk");
    let mut logits = None;
    let mut prefill_logits: Vec<f32> = Vec::new();
    for first in (0..prefill).step_by(chunk) {
        let n = chunk.min(prefill - first);
        let mut prefill_compare = |layer: usize, stream: &[u8]| compare(layer, first, stream, &mut worst);
        logits = engine.prefill_forced(&mut placement, &embed[first * row..(first + n) * row], args.nll,
            (!args.timing).then_some(&mut prefill_compare as &mut dyn FnMut(usize, &[u8]) -> Result<()>),
            args.teacher_force.then_some(&forced as &dyn Fn(usize) -> Option<Vec<u8>>))?;
        if args.nll {
            prefill_logits.extend(logits.as_deref().unwrap_or_default());
        }
    }
    if args.nll {
        logits = prefill_logits.len().checked_sub(cfg.vocab_size).map(|at| prefill_logits[at..].to_vec());
    }
    let prefill_seconds = started.elapsed().as_secs_f64();
    if chunk < prefill {
        println!("prefill in chunks of {chunk}: worst cosine per layer {:?}",
            worst.iter().map(|c| format!("{c:.6}")).collect::<Vec<_>>());
    }
    let started = Instant::now();
    let mut decode_worst = Vec::new();
    let mut decode_logits: Vec<f32> = Vec::new();
    let mut position = prefill;
    while position < tokens.len() {
        let n = args.step_rows.min(tokens.len() - position);
        let first = position;
        let mut step_compare = |layer: usize, stream: &[u8]| compare(layer, first, stream, &mut decode_worst);
        if let Some(logits) = engine.verify(&mut [(&mut placement, n)], &embed[position * row..(position + n) * row],
            (!args.timing).then_some(&mut step_compare as &mut dyn FnMut(usize, &[u8]) -> Result<()>))? {
            decode_logits.extend(logits);
        }
        position += n;
    }
    if prefill < tokens.len() {
        println!("decode: {} rows in steps of {} in {:.2} s; worst row-block cosine per layer {:?}", tokens.len() - prefill,
            args.step_rows, started.elapsed().as_secs_f64(),
            decode_worst.iter().map(|c| format!("{c:.6}")).collect::<Vec<_>>());
    }
    let golden_logits = || -> Result<Vec<f32>> {
        Ok(std::fs::read(args.golden.join("logits.bin"))?
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    };
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
    // Rows of `ours` are the logits of positions `first..`: top-1 agreement
    // with the golden, next-token accuracy and mean NLL of the next tokens.
    let score = |label: &str, ours_all: &[f32], first: usize| -> Result<()> {
        // Numerics A/B between engine configs (benchmarks): the decode rows' logits, F32.
        if let (true, Ok(path)) = (label == "decode", std::env::var("CUTEAFD_DUMP_DECODE_LOGITS")) {
            std::fs::write(&path, ours_all.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        let golden = golden_logits()?;
        let vocab = cfg.vocab_size;
        let (mut agree, mut next_ok, mut golden_next, mut nll, mut kl) = (0usize, 0usize, 0usize, 0f64, 0f64);
        let rows = ours_all.len() / vocab;
        let log_softmax = |l: &[f32]| -> Vec<f64> {
            let top = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = top + l.iter().map(|&x| (x as f64 - top).exp()).sum::<f64>().ln();
            l.iter().map(|&x| x as f64 - lse).collect()
        };
        for r in 0..rows {
            let (ours, theirs) = (&ours_all[r * vocab..][..vocab], &golden[(first + r) * vocab..][..vocab]);
            agree += usize::from(argmax(ours) == argmax(theirs));
            // KL(golden || engine) of the next-token distributions.
            let (p, q) = (log_softmax(theirs), log_softmax(ours));
            kl += p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum::<f64>();
            if let Some(&next) = tokens.get(first + r + 1) {
                next_ok += usize::from(argmax(ours) == next as usize);
                golden_next += usize::from(argmax(theirs) == next as usize);
                let top = ours.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let sum: f64 = ours.iter().map(|&l| (l as f64 - top).exp()).sum();
                nll += top + sum.ln() - ours[next as usize] as f64;
            }
        }
        let scored = tokens.len().saturating_sub(first + 1).min(rows).max(1) as f64;
        println!("{label} logits: top-1 agreement {:.1}% over {rows} rows | next-token accuracy engine {:.1}% \
            golden {:.1}% | mean NLL {:.4} | mean KL(golden||engine) {:.5}", 100.0 * agree as f64 / rows as f64,
            100.0 * next_ok as f64 / scored, 100.0 * golden_next as f64 / scored, nll / scored, kl / rows as f64);
        Ok(())
    };
    if !prefill_logits.is_empty() {
        score("prefill", &prefill_logits, 0)?;
    }
    if !decode_logits.is_empty() {
        score("decode", &decode_logits, prefill)?;
    }
    println!("prefill: {prefill} tokens through {layers} layers in {prefill_seconds:.2} s");
    if let Some(logits) = logits {
        let golden = golden_logits()?;
        let last = &golden[(prefill - 1) * cfg.vocab_size..][..cfg.vocab_size];
        let (cosine, _) = similarity(&logits, last);
        println!("last-row logits: argmax engine {} golden {} cosine {cosine:.6}", argmax(&logits), argmax(last));
    }
    let profile = engine.profile.borrow();
    if profile[1] > 0.0 {
        println!("phases: GPU wait before expert requests {:.2} s, expert exchange / streamed loads {:.2} s",
            profile[0], profile[1]);
    }
    Ok(())
}

/// Runs the drafter alone on the golden taps at reference.py's anchors and
/// compares drafts and final-norm rows; scores drafts against the golden text
/// (accepted prefix) and against the target's greedy predictions along the
/// text (`logits.bin` argmax, counted while the text follows the drafts); and
/// times draft steps of 1..=draft_sequences sequences.
fn draft_oracle(args: &GoldenArgs, opened: &Opened, engine: &engine::MimoEngine<'_>, dir: &std::path::Path)
    -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft-oracle needs --draft")?;
    let (hidden, block, drafts_per, vocab) = (opened.cfg.hidden, drafter.cfg.block, drafter.cfg.drafts(),
        opened.cfg.vocab_size);
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let positions: Vec<usize> = meta["positions"].as_array().context("positions")?.iter()
        .map(|p| p.as_u64().map(|p| p as usize).context("position")).collect::<Result<_>>()?;
    let ref_tokens: Vec<u32> = std::fs::read(dir.join("drafts.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let ref_hidden = bf16s(&std::fs::read(dir.join("hidden.bin"))?);
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let greedy: Vec<u32> = std::fs::read(args.golden.join("logits.bin"))?.chunks_exact(vocab * 4).map(|row| {
        let values = row.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()));
        values.enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).map_or(0, |(i, _)| i as u32)
    }).collect();
    let layers: Vec<Vec<u8>> = drafter.cfg.taps.iter()
        .map(|l| std::fs::read(args.golden.join(format!("layer{l:02}.bin")))).collect::<std::io::Result<_>>()?;
    let row = hidden * 2;
    let width = layers.len() * row;
    let (mut done, mut exact, mut first, mut matched, mut worst) = (0usize, 0, 0, 0, 1f64);
    let (mut text_accepted, mut greedy_accepted) = (0usize, 0usize);
    let mut histogram = vec![0usize; block];
    let mut draft_seconds = 0f64;
    let update = |from: usize, to: usize, slot: usize| -> Result<()> {
        let mut at = from;
        while at < to {
            let n = (to - at).min(dflash::TAP_ROWS);
            let mut taps = vec![0u8; n * width];
            for r in 0..n {
                for (i, layer) in layers.iter().enumerate() {
                    taps[r * width + i * row..][..row].copy_from_slice(&layer[(at + r) * row..][..row]);
                }
            }
            drafter.put_taps(&taps)?;
            drafter.update(&(0..n).map(|r| dflash::ContextRow { tap_row: r, slot, position: at + r })
                .collect::<Vec<_>>())?;
            at += n;
        }
        Ok(())
    };
    for (index, &position) in positions.iter().enumerate() {
        // Only the last RING positions matter to a draft at `position`.
        update(done.max(position.saturating_sub(dflash::RING)), position, 0)?;
        done = position;
        let anchor = tokens[position];
        let timer = Instant::now();
        let draft = drafter.draft(&[dflash::DraftSeq { slot: 0, anchor, position }],
            &embed_rows(&opened.checkpoint, &[anchor], hidden)?, engine.weights.head.buffer.ptr)?.remove(0);
        draft_seconds += timer.elapsed().as_secs_f64();
        let reference = &ref_tokens[index * drafts_per..][..drafts_per];
        exact += usize::from(draft.tokens == reference);
        first += usize::from(draft.tokens[0] == reference[0]);
        matched += draft.tokens.iter().zip(reference).take_while(|(a, b)| a == b).count();
        let (cosine, _) = similarity(&bf16s(&drafter.last_hidden(1)?), &ref_hidden[index * block * hidden..][..block * hidden]);
        worst = worst.min(cosine);
        let text = draft.tokens.iter().enumerate()
            .take_while(|&(j, &d)| tokens.get(position + 1 + j) == Some(&d)).count();
        let target = draft.tokens.iter().enumerate().take_while(|&(j, &d)| {
            greedy.get(position + j) == Some(&d) && (j == 0 || tokens.get(position + j) == Some(&draft.tokens[j - 1]))
        }).count();
        text_accepted += text;
        greedy_accepted += target;
        histogram[text] += 1;
        if draft.tokens != reference {
            println!("position {position}: engine {:?} reference {reference:?}", draft.tokens);
        }
    }
    let n = positions.len();
    println!("draft oracle: {n} anchors, identical drafts {exact}/{n}, first draft {first}/{n}, matching prefix \
        {:.2} of {drafts_per}, worst final-norm cosine {worst:.6}, {:.2} ms/draft", matched as f64 / n as f64,
        draft_seconds * 1e3 / n as f64);
    println!("acceptance (teacher-forced on the golden text): accepted prefix vs text {:.2}, vs target greedy \
        {:.2} per draft of {drafts_per}; text histogram {histogram:?}", text_accepted as f64 / n as f64,
        greedy_accepted as f64 / n as f64);
    // Draft-step cost by sequences (every slot holds the same context).
    let position = *positions.last().context("no anchors")?;
    for slot in 1..drafter.slots {
        update(position.saturating_sub(dflash::RING), position, slot)?;
    }
    let anchor = embed_rows(&opened.checkpoint, &[tokens[position]], hidden)?;
    for sequences in 1..=drafter.slots {
        let seqs: Vec<_> = (0..sequences).map(|slot| dflash::DraftSeq { slot, anchor: tokens[position], position })
            .collect();
        let rows = anchor.repeat(sequences);
        drafter.draft(&seqs, &rows, engine.weights.head.buffer.ptr)?;
        let timer = Instant::now();
        for _ in 0..10 {
            drafter.draft(&seqs, &rows, engine.weights.head.buffer.ptr)?;
        }
        let per = timer.elapsed().as_secs_f64() * 1e2;
        let timer = Instant::now();
        for _ in 0..10 {
            update(position - 8, position, 0)?;
        }
        // SAFETY: the engine owns this stream.
        unsafe { opened.library.cuda_stream_synchronize(engine.stream)? };
        println!("draft step, {sequences} sequences: {per:.2} ms; context update of 8 rows {:.2} ms",
            timer.elapsed().as_secs_f64() * 1e2);
    }
    Ok(())
}

/// Runs the MTP drafter alone on the golden's last-layer rows at the
/// reference's anchors (DFlash's convention: the anchor p is the next token,
/// context 0..p-1) and compares stage k's draft with the reference's
/// teacher-forced prediction at row p - 1 while every earlier draft equals
/// the golden token (then the two chains are the same computation).
fn mtp_oracle(args: &GoldenArgs, opened: &Opened, engine: &engine::MimoEngine<'_>, dir: &std::path::Path)
    -> Result<()> {
    let mtp = engine.mtp.as_ref().context("--mtp-oracle needs --mtp N")?;
    let (hidden, vocab) = (opened.cfg.hidden, opened.cfg.vocab_size);
    let stages = mtp.stages.len();
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let anchors: Vec<usize> = meta["anchors"].as_array().context("anchors")?.iter()
        .map(|p| p.as_u64().map(|p| p as usize).context("anchor")).collect::<Result<_>>()?;
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let t = tokens.len();
    let predictions: Vec<u32> = std::fs::read(dir.join("predictions.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let ref_stages = meta["stages"].as_u64().unwrap_or(0) as usize;
    ensure!(ref_stages >= stages && predictions.len() == ref_stages * t, "reference predictions do not cover {stages} stages");
    let greedy: Vec<u32> = std::fs::read(args.golden.join("logits.bin"))?.chunks_exact(vocab * 4).map(|row| {
        let values = row.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()));
        values.enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).map_or(0, |(i, _)| i as u32)
    }).collect();
    let last = std::fs::read(args.golden.join(format!("layer{:02}.bin", opened.cfg.layers - 1)))?;
    let row = hidden * 2;
    let embed = |ids: &[u32]| embed_rows(&opened.checkpoint, ids, hidden);
    engine.mtp_reset(0, anchors[0]);
    let (mut compared, mut matched, mut text, mut target) = (vec![0usize; stages], vec![0usize; stages], 0usize, 0usize);
    let mut seconds = 0f64;
    let mut filled = 0usize;
    for &p in &anchors {
        // Golden last-layer rows of positions < p into ring 0 of the hidden ring.
        for j in filled.max(p.saturating_sub(mtp::HIDDEN_ROWS))..p {
            let at = (j % mtp::HIDDEN_ROWS) * row;
            opened.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer {
                // SAFETY: one row inside the hidden ring.
                ptr: unsafe { mtp.hidden.buffer.ptr.cast::<u8>().add(at) }.cast(), bytes: row, ..mtp.hidden.buffer },
                &last[j * row..(j + 1) * row])?;
        }
        filled = p;
        let timer = Instant::now();
        let drafts = engine.mtp_draft(&[mtp::MtpSeq { ring: 0, len: p, tokens: &tokens[..=p] }], stages, &embed)?
            .remove(0);
        seconds += timer.elapsed().as_secs_f64();
        for k in 0..stages {
            if (0..k).all(|m| tokens.get(p + 1 + m) == Some(&drafts[m])) {
                compared[k] += 1;
                matched[k] += usize::from(drafts[k] == predictions[k * t + p - 1]);
            }
        }
        text += drafts.iter().enumerate().take_while(|&(k, &d)| tokens.get(p + 1 + k) == Some(&d)).count();
        target += drafts.iter().enumerate().take_while(|&(k, &d)| {
            greedy.get(p + k) == Some(&d) && (k == 0 || tokens.get(p + k) == Some(&drafts[k - 1]))
        }).count();
    }
    let n = anchors.len();
    println!("MTP oracle: {n} anchors, {stages} stages; drafts equal to the reference (where comparable) {:?} of {:?}; \
        accepted prefix vs text {:.2}, vs target greedy {:.2}; {:.2} ms/draft (incl. true-row catch-up)",
        matched, compared, text as f64 / n as f64, target as f64 / n as f64, seconds * 1e3 / n as f64);
    Ok(())
}
