//! MiMo V2 (mimo_v2_flash) on the generic engine: weights, the coordinator
//! programs' layer chain, and the golden comparison command.
pub(crate) mod engine;
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
    /// TP1 FP8 package layout for --local-experts (default `<libdir>/fp8/fp8-mimo/tp1`).
    #[arg(long)]
    pub fp8_package: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Directory with tokens.bin, layerNN.bin and logits.bin from
    /// python/reference/mimo_v2/golden.py.
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
        let stream = self.library.cuda_stream_create()?;
        let started = Instant::now();
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let loader = weights::MimoLoader { library: &self.library, checkpoint: &self.checkpoint, stream };
        let model = loader.model(&self.cfg, layers)?;
        tracing::info!(layers, elapsed_ms = started.elapsed().as_millis() as u64, "MiMo coordinator weights resident");
        let pages = args.pool_tokens.div_ceil(engine::PAGE_ROWS);
        let mut engine = engine::MimoEngine::new(&self.library, &programs, self.cfg.clone(), model, stream,
            args.max_context, args.prefill_rows, pages, args.rings)?;
        let moe_layers = (0..layers).filter(|&l| !self.cfg.dense[l]).collect::<Vec<_>>();
        if let Some(experts) = self.experts(args, &moe_layers)? {
            engine.set_experts(experts);
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
        if args.local_experts {
            let tensors = self.catalog.fp8().context("MiMo experts are the checkpoint's FP8 tensors")?;
            let directory = args.fp8_package.clone()
                .unwrap_or_else(|| crate::v41_experts::fp8::package_directory(&args.native_lib, 1));
            let (free, _) = self.library.cuda_memory_info()?;
            let started = Instant::now();
            let local = crate::v41_experts::fp8::Fp8Experts::load(&self.library, tensors, &directory,
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
        let mut transport = cuteafd_transport::v41_expert::V41Tp4Roce::new_ranks(&peers, &executors, u32::try_from(args.prefill_rows)?,
            cuteafd_transport::TcpTransportConfig { timing: false, timeout: std::time::Duration::from_secs(120),
                max_frame_bytes: 64 << 20 })?;
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        // Connect every rank and register full-size buffers now: the first
        // request otherwise pays seconds of connection setup.
        let started = Instant::now();
        let (rows, h, topk) = (args.prefill_rows, self.cfg.hidden, self.cfg.topk);
        let routes = (0..rows * topk).map(|i| cuteafd_transport::ExpertProtocolV2RouteEntry {
            row_index: (i / topk) as u32, expert_id: (i % self.cfg.experts) as u32, gate_weight: 0.0,
        }).collect();
        let mut request = cuteafd_transport::ExpertProtocolV2Request::new(1, 17, moe_layers[0] as u32, h as u32,
            cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..rows as u32).map(|row| cuteafd_transport::ExpertProtocolV2RowDescriptor {
                row_id: u64::from(row), source_kind: cuteafd_transport::ExpertV2SourceKind::Prefill,
                source_request_id: 1, token_position: u64::from(row), route_offset: row * topk as u32,
                route_count: topk as u32,
            }).collect(),
            routes, vec![0; rows * (h + h / 32)])?;
        request.header.flags |= cuteafd_transport::v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        runtime.block_on(async { transport.execute(&request, |_, _, _| Ok(())).await })?;
        tracing::info!(ranks = peers.len(), elapsed_ms = started.elapsed().as_millis() as u64, "Spark expert transport warm");
        Ok(Some(engine::Experts::Spark { transport: std::cell::RefCell::new(transport), runtime }))
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
    let logits = engine.prefill_forced(&mut placement, &embed[..prefill * row],
        Some(&mut |layer, stream| compare(layer, 0, stream, &mut worst)),
        args.teacher_force.then_some(&forced as &dyn Fn(usize) -> Option<Vec<u8>>))?;
    let prefill_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut decode_worst = Vec::new();
    let mut decode_logits: Vec<f32> = Vec::new();
    let mut position = prefill;
    while position < tokens.len() {
        let n = args.step_rows.min(tokens.len() - position);
        let first = position;
        if let Some(logits) = engine.verify(&mut [(&mut placement, n)], &embed[position * row..(position + n) * row],
            Some(&mut |layer, stream| compare(layer, first, stream, &mut decode_worst)))? {
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
    if !decode_logits.is_empty() {
        let golden = golden_logits()?;
        let vocab = cfg.vocab_size;
        let (mut agree, mut next_ok, mut golden_next, mut nll) = (0usize, 0usize, 0usize, 0f64);
        let rows = decode_logits.len() / vocab;
        for r in 0..rows {
            let (ours, theirs) = (&decode_logits[r * vocab..][..vocab], &golden[(prefill + r) * vocab..][..vocab]);
            agree += usize::from(argmax(ours) == argmax(theirs));
            if let Some(&next) = tokens.get(prefill + r + 1) {
                next_ok += usize::from(argmax(ours) == next as usize);
                golden_next += usize::from(argmax(theirs) == next as usize);
                let top = ours.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let sum: f64 = ours.iter().map(|&l| (l as f64 - top).exp()).sum();
                nll += top + sum.ln() - ours[next as usize] as f64;
            }
        }
        let scored = (rows - 1).max(1) as f64;
        println!("decode logits: top-1 agreement {:.1}% over {rows} rows | next-token accuracy engine {:.1}% \
            golden {:.1}% | mean NLL {:.4}", 100.0 * agree as f64 / rows as f64, 100.0 * next_ok as f64 / scored,
            100.0 * golden_next as f64 / scored, nll / scored);
    }
    println!("prefill: {prefill} tokens through {layers} layers in {prefill_seconds:.2} s");
    if let Some(logits) = logits {
        let golden = golden_logits()?;
        let last = &golden[(prefill - 1) * cfg.vocab_size..][..cfg.vocab_size];
        let (cosine, _) = similarity(&logits, last);
        println!("last-row logits: argmax engine {} golden {} cosine {cosine:.6}", argmax(&logits), argmax(last));
    }
    Ok(())
}
