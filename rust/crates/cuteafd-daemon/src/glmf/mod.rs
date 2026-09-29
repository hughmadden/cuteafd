//! GLM 5.3 Flash (glm5_next) on the generic engine: weights, the coordinator
//! programs' layer chain, and the golden comparison command.
pub(crate) mod engine;
pub(crate) mod fp8;
pub(crate) mod serve;
mod speculate;
pub(crate) mod weights;

use anyhow::{ensure, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::fp8_experts::Fp8ExpertTensors;
use cuteafd_loader::glm_next::GlmNextConfig;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::time::Instant;

/// What every GLM 5.3 Flash command needs to stand up the engine.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct EngineArgs {
    /// Checkpoint snapshot (coordinator weights, config, tokenizer).
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/DSV4_PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// Run only the first N layers (layers 0-2 are the dense ones).
    #[arg(long)]
    pub layers: Option<usize>,
    /// Longest sequence (the exported index top-k covers up to 131072).
    #[arg(long, default_value_t = 65_536)]
    pub max_context: usize,
    /// Tokens the MLA record pools hold across sequences.
    #[arg(long, default_value_t = 32_768)]
    pub pool_tokens: usize,
    /// Sequences with KDA state (136 MiB each).
    #[arg(long, default_value_t = 8)]
    pub slots: usize,
    #[arg(long, default_value_t = 4096)]
    pub prefill_rows: usize,
    /// Spark ranks in TP order (HOST:PORT,...) serving the fp8 expert family.
    #[arg(long, conflicts_with = "local_experts")]
    pub peers: Option<String>,
    /// Run the routed experts on this GPU: EXL3 checkpoints through the
    /// coordinator `exl3-glmf-k<tiers>/rtx-tp1` package, FP8 ones through the
    /// TP1 `fp8-glmf` package. `--experts-snapshot` names another checkpoint
    /// for the experts (the official FP8 one with EXL3 coordinator weights).
    #[arg(long)]
    pub local_experts: bool,
    #[arg(long)]
    pub experts_snapshot: Option<PathBuf>,
    /// TP1 FP8 package directory (default `<libdir>/fp8/fp8-glmf/tp1`).
    #[arg(long)]
    pub fp8_package: Option<PathBuf>,
    /// FP8 expert layers resident at once with --local-experts (7.25 GiB each).
    #[arg(long, default_value_t = 6)]
    pub expert_window: usize,
    /// Decode rows (<= 16 per step) read FP8 copies of the MLA, dense and
    /// shared-expert projections: the official FP8 release's own E4M3 blocks
    /// with --fp8-snapshot, else 128x128 blocks quantized from BF16.
    #[arg(long)]
    pub fp8_decode: bool,
    /// The official FP8 checkpoint (zai-org/GLM-5.3-Flash) for --fp8-decode.
    #[arg(long)]
    pub fp8_snapshot: Option<PathBuf>,
    /// FP8 KDA projections for decode rows, quantized per row at load.
    #[arg(long, value_enum, default_value = "off")]
    pub kda_fp8: fp8::KdaFp8,
    /// Decode rows (<= 16) project to the vocabulary through an FP8 copy of the
    /// LM head (per row x 128-K scales, quantized at load).
    #[arg(long)]
    pub fp8_head: bool,
    /// Numerics gate only: round the KDA projections through NVFP4 (group 16,
    /// E4M3 scales) at load and run them as BF16: `rtn` (amax/6) or `search`.
    #[arg(long, hide = true)]
    pub kda_nvfp4_gate: Option<String>,
    /// Keep every prefill row's logits (glmf-golden --nll; 2.5 GiB at 4096 rows).
    #[arg(long, hide = true)]
    pub full_prefill_logits: bool,
    /// Most EXL3 expert layers resident at once (about 3 GiB each; the free
    /// memory decides first).
    #[arg(long, default_value_t = 64)]
    pub exl3_window: usize,
    /// Prefill projections that run block-FP8 GEMMs (E4M3 activations per row
    /// and 128-K block, FP32 scales): `mla` (q_a|kv_a, q_b, o_proj) and `ffn`
    /// (dense and shared-expert MLPs) over the official FP8 weights (needs
    /// --fp8-decode --fp8-snapshot), `kda-in` / `kda-o` (the KDA in-projection
    /// and o_proj over their per-row copies; needs --kda-fp8 row128).
    #[arg(long, value_enum, value_delimiter = ',')]
    pub fp8_prefill: Vec<Fp8PrefillGroup>,
    /// Profiling only: MoE layers run the router, the expert wire rows and the
    /// shared expert; the routed experts contribute nothing.
    #[arg(long, hide = true)]
    pub skip_experts: bool,
    /// DFlash2 drafter snapshot (incoai/GLM-5.3-Flash-DFlash2): taps the mHC
    /// stream mean after its target layers and drafts on this GPU.
    #[arg(long)]
    pub draft: Option<PathBuf>,
    /// Sequences the drafter keeps a context for and drafts for at once.
    #[arg(long, default_value_t = 8)]
    pub draft_sequences: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Fp8PrefillGroup {
    Mla,
    Ffn,
    KdaIn,
    KdaO,
    /// Every group.
    All,
}

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Directory with tokens.bin, layerNN.bin and logits.bin from
    /// python/reference/glm5_next/golden.py.
    #[arg(long)]
    pub golden: PathBuf,
    /// Prefill only the first N tokens, then feed the rest through decode
    /// steps of --step-rows rows (teacher-forced), comparing their rows.
    #[arg(long)]
    pub prefill: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub step_rows: usize,
    /// Decode steps run speculatively (KDA state untouched, replay rows
    /// recorded) and then commit every row: the verify-by-replay path.
    #[arg(long)]
    pub spec_steps: bool,
    /// Feed each layer the golden output of the previous one (prefill only),
    /// so every layer's cosine measures that layer alone.
    #[arg(long)]
    pub teacher_force: bool,
    /// Score every prefill row's logits against the golden (mean NLL, top-1).
    #[arg(long)]
    pub nll: bool,
    /// Decode steps compare logits only (no per-layer downloads; graphs run).
    #[arg(long)]
    pub logits_only: bool,
    /// After the comparison, time this many greedy single-row decode steps
    /// (no layer downloads) from the prefilled sequence.
    #[arg(long, default_value_t = 0)]
    pub bench_decode: usize,
    /// Time this many more prefills of the golden prompt (up to --prefill-rows
    /// tokens) on fresh sequences, without layer downloads.
    #[arg(long, default_value_t = 0)]
    pub bench_prefill: usize,
    /// Prompt length of --bench-prefill (the golden tokens repeated), in
    /// chunks of the engine's prefill capacity; default the golden prompt up
    /// to one chunk.
    #[arg(long)]
    pub bench_prefill_tokens: Option<usize>,
    /// With --draft: run only the drafter on the golden taps (the layers'
    /// stream means) and compare with python/reference/glm_dflash2/reference.py's
    /// output directory.
    #[arg(long)]
    pub draft_oracle: Option<PathBuf>,
    /// With --draft: after the --prefill tokens, decode N greedy tokens one row
    /// per step, drafting before each, and report how many drafts the target
    /// reproduced (0: teacher-forced on tokens.bin, scoring against it).
    #[arg(long)]
    pub generate: Option<usize>,
    /// Verify-by-replay check: after --prefill tokens, for every kept count k
    /// in 1..=N, compare the KDA state after one speculative N-row verify
    /// committing k rows with the state after k serial single-row steps (and
    /// after a plain N-row verify for k = N); then time spec + commit.
    #[arg(long)]
    pub replay_check: Option<usize>,
    /// Time verify steps of 1..=N rows per sequence (C sequences, see
    /// --bench-sequences) after --prefill tokens: the step cost by rows.
    #[arg(long)]
    pub bench_verify: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub bench_sequences: usize,
}

/// The checkpoint and native library, opened on the calling thread.
pub(crate) struct Opened {
    pub checkpoint: Checkpoint,
    pub fp8_checkpoint: Option<Checkpoint>,
    pub cfg: GlmNextConfig,
    pub library: NativeLibrary,
    /// The FP8 expert catalog for --local-experts.
    pub experts: Option<cuteafd_loader::OfficialV41Catalog>,
}

impl Opened {
    fn fp8(&self) -> Option<&Fp8ExpertTensors> {
        self.experts.as_ref().and_then(|c| c.fp8())
    }
}

pub(crate) fn open(args: &EngineArgs) -> Result<Opened> {
    let checkpoint = Checkpoint::open(&args.snapshot)?;
    ensure!(checkpoint.missing_shards.is_empty(), "checkpoint shards missing: {:?}", checkpoint.missing_shards);
    let cfg = GlmNextConfig::read(&args.snapshot)?;
    // The expert geometry is process-wide and must be fixed before the native
    // library loads (its expert helpers size rows from it).
    let geometry = cuteafd_core::ExpertGeometry::GLM_NEXT;
    ensure!(geometry.hidden as usize == cfg.hidden && geometry.experts as usize == cfg.experts
        && geometry.topk as usize == cfg.topk && geometry.intermediate as usize == cfg.moe_intermediate,
        "checkpoint experts do not match the GLM 5.3 Flash geometry");
    cuteafd_core::set_expert_geometry(geometry).map_err(|g| anyhow::anyhow!("expert geometry already {g:?}"))?;
    let experts = if args.local_experts {
        let source = args.experts_snapshot.as_deref().unwrap_or(&args.snapshot);
        let catalog = cuteafd_loader::read_expert_catalog(source)?;
        ensure!(catalog.fp8().is_some() || catalog.exl3().is_some(),
            "--local-experts runs FP8 or EXL3 experts; {} has neither", source.display());
        Some(catalog)
    } else {
        None
    };
    // SAFETY: the library is the cuteafd native shim built for this engine.
    let library = unsafe { NativeLibrary::load(&args.native_lib) }?;
    library.cuda_set_device(args.device)?;
    let fp8_checkpoint = args.fp8_snapshot.as_deref().map(Checkpoint::open).transpose()?;
    Ok(Opened { checkpoint, fp8_checkpoint, cfg, library, experts })
}

impl Opened {
    /// Builds the engine and hands it to `body`.
    pub fn with_engine<T>(&self, args: &EngineArgs, body: impl FnOnce(&engine::GlmfEngine<'_>) -> Result<T>)
        -> Result<T> {
        let programs = self.library.dsv4_programs()?.with_manifest(&args.manifest)?;
        programs.load_all()?;
        let stream = self.library.cuda_stream_create()?;
        let started = Instant::now();
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let loader = weights::GlmfLoader { library: &self.library, checkpoint: &self.checkpoint, stream,
            fp8_dense: args.fp8_decode, fp8_source: self.fp8_checkpoint.as_ref(), kda_fp8: args.kda_fp8,
            fp8_head: args.fp8_head, kda_nvfp4: args.kda_nvfp4_gate.as_deref().map(|mode| mode == "search") };
        let model = loader.model(&self.cfg, layers)?;
        let resident: usize = model.layers.iter().map(weights::GlmfLayer::bytes).sum();
        tracing::info!(layers, gib = resident as f64 / (1u64 << 30) as f64, fp8_decode = args.fp8_decode,
            fp8_source = self.fp8_checkpoint.is_some(), kda_fp8 = ?args.kda_fp8,
            elapsed_ms = started.elapsed().as_millis() as u64, "GLM 5.3 Flash coordinator weights resident");
        let pages = args.pool_tokens.div_ceil(engine::PAGE_ROWS);
        let mut engine = engine::GlmfEngine::new(&self.library, &programs, self.cfg.clone(), model, stream,
            args.max_context, args.prefill_rows, pages, args.slots)?;
        engine.full_prefill_logits = args.full_prefill_logits;
        let group = |g: Fp8PrefillGroup| args.fp8_prefill.iter().any(|&x| x == g || x == Fp8PrefillGroup::All);
        engine.fp8_prefill = engine::Fp8Prefill { mla: group(Fp8PrefillGroup::Mla), ffn: group(Fp8PrefillGroup::Ffn),
            kda_bits: i32::from(group(Fp8PrefillGroup::KdaIn)) | (i32::from(group(Fp8PrefillGroup::KdaO)) << 1) };
        ensure!(!(engine.fp8_prefill.mla || engine.fp8_prefill.ffn) || args.fp8_decode,
            "--fp8-prefill mla/ffn reads the FP8 copies --fp8-decode loads");
        ensure!(engine.fp8_prefill.kda_bits == 0 || args.kda_fp8 == fp8::KdaFp8::Row128,
            "--fp8-prefill kda reads the per-row FP8 copies --kda-fp8 row128 loads");
        if let Some(snapshot) = &args.draft {
            let started = Instant::now();
            let cfg = crate::glm::dflash::DflashConfig::read(snapshot)?;
            ensure!(cfg.hidden == self.cfg.hidden && cfg.vocab == self.cfg.vocab_size
                && cfg.taps.iter().all(|&l| l < self.cfg.layers), "the DFlash2 drafter does not fit this target");
            let mask = embed_rows(&self.checkpoint, &[cfg.mask_token], self.cfg.hidden)?;
            let file = crate::glm::dflash::prefetch(snapshot).join()
                .map_err(|_| anyhow::anyhow!("drafter read panicked"))??;
            engine.drafter = Some(crate::glm::dflash::GlmDrafter::load(&self.library, snapshot, file, stream,
                args.draft_sequences, args.draft_sequences, mask, true)?);
            tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "DFlash2 drafter resident");
        }
        if (0..layers).any(|l| !self.cfg.dense[l]) {
            if let Some(experts) = self.experts(args)? {
                engine.set_experts(experts);
            }
        }
        let result = body(&engine);
        drop(engine);
        // SAFETY: the engine that used the stream is gone.
        unsafe { self.library.cuda_stream_destroy(stream)? };
        result
    }

    fn experts<'s>(&'s self, args: &EngineArgs) -> Result<Option<engine::Experts<'s>>> {
        if args.skip_experts {
            return Ok(Some(engine::Experts::Skip));
        }
        if let Some(tensors) = self.fp8() {
            let directory = args.fp8_package.clone()
                .unwrap_or_else(|| crate::v41_experts::fp8::package_directory(&args.native_lib, 1));
            let (free, _) = self.library.cuda_memory_info()?;
            // An empty window: the package and its scratch; layers load on first use.
            let experts = crate::v41_experts::fp8::Fp8Experts::load(&self.library, tensors, &directory, 0..0, 1, 0,
                args.prefill_rows, free.saturating_sub(4 << 30))?;
            return Ok(Some(engine::Experts::Local(engine::LocalExperts {
                library: &self.library, tensors, experts: std::cell::RefCell::new(experts),
                window: args.expert_window.max(1), loads: std::cell::RefCell::new(0),
            })));
        }
        if let Some(catalog) = self.experts.as_ref().filter(|c| c.exl3().is_some()) {
            let (free, _) = self.library.cuda_memory_info()?;
            return Ok(Some(engine::Experts::LocalExl3(engine::LocalExl3 {
                library: &self.library, native_lib: args.native_lib.clone(), catalog,
                resident: std::cell::RefCell::new(None), window: args.exl3_window.max(1),
                layers: args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers), max_rows: args.prefill_rows,
                // Room for the step workspace (logits alone are 2.4 GiB at 4096 rows).
                budget: free.saturating_sub(12 << 30), loads: std::cell::RefCell::new(0),
            })));
        }
        let Some(peers) = args.peers.as_deref() else { return Ok(None) };
        let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
        let executors: Vec<u64> = (0..peers.len())
            .map(|rank| cuteafd_transport::v41_expert::v41_spark_executor_id(peers.len(), rank))
            .collect::<Result<_>>()?;
        // One transport per prefill lane: each lane's wave stays in flight on its own QPs.
        let transports = (0..engine::PREFILL_LANES).map(|_| cuteafd_transport::v41_expert::V41Tp4Roce::new_ranks(
            &peers, &executors, u32::try_from(args.prefill_rows)?, cuteafd_transport::TcpTransportConfig { timing: false,
                timeout: std::time::Duration::from_secs(120), max_frame_bytes: 64 << 20 }))
            .collect::<Result<Vec<_>>>()?;
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        Ok(Some(engine::Experts::Spark { transports: std::cell::RefCell::new(transports), runtime }))
    }
}

pub(crate) fn embed_rows(checkpoint: &Checkpoint, tokens: &[u32], hidden: usize) -> Result<Vec<u8>> {
    let name = format!("{}embed_tokens.weight", weights::PREFIX);
    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(&name))
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

pub(crate) async fn run_golden(mut args: GoldenArgs) -> Result<()> {
    args.engine.full_prefill_logits |= args.nll;
    tokio::task::spawn_blocking(move || golden(args)).await?
}

fn golden(args: GoldenArgs) -> Result<()> {
    let opened = open(&args.engine)?;
    opened.with_engine(&args.engine, |engine| golden_run(&args, &opened, engine))
}

/// Mean NLL of `logits` rows against the next tokens, and top-1 agreements.
/// Mean KL(golden || engine) over rows, in float64 (the golden's next-token
/// distribution against the engine's, as the published KL gates compute it).
fn mean_kl(logits: &[f32], golden: &[f32], first: usize, vocab: usize) -> f64 {
    let log_softmax = |l: &[f32]| -> Vec<f64> {
        let top = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = top + l.iter().map(|&x| (x as f64 - top).exp()).sum::<f64>().ln();
        l.iter().map(|&x| x as f64 - lse).collect()
    };
    let rows = logits.len() / vocab;
    let total: f64 = logits.chunks_exact(vocab).enumerate().map(|(r, ours)| {
        let (p, q) = (log_softmax(&golden[(first + r) * vocab..][..vocab]), log_softmax(ours));
        p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum::<f64>()
    }).sum();
    total / rows.max(1) as f64
}

fn score(logits: &[f32], golden: &[f32], tokens: &[u32], first: usize, vocab: usize) -> (usize, usize, usize, f64, usize) {
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
    let (mut agree, mut next_ok, mut golden_next, mut nll, mut scored) = (0, 0, 0, 0f64, 0);
    for (r, ours) in logits.chunks_exact(vocab).enumerate() {
        let theirs = &golden[(first + r) * vocab..][..vocab];
        agree += usize::from(argmax(ours) == argmax(theirs));
        if let Some(&next) = tokens.get(first + r + 1) {
            next_ok += usize::from(argmax(ours) == next as usize);
            golden_next += usize::from(argmax(theirs) == next as usize);
            let top = ours.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = ours.iter().map(|&l| (l as f64 - top).exp()).sum();
            nll += top + sum.ln() - ours[next as usize] as f64;
            scored += 1;
        }
    }
    (agree, next_ok, golden_next, nll, scored)
}

fn golden_run(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmfEngine<'_>) -> Result<()> {
    if let Some(dir) = &args.draft_oracle {
        return speculate::draft_oracle(args, opened, engine, dir);
    }
    if let Some(rows) = args.replay_check {
        return speculate::replay_check(args, opened, engine, rows);
    }
    if let Some(rows) = args.bench_verify {
        return speculate::bench_verify(args, opened, engine, rows);
    }
    if engine.drafter.is_some() {
        return speculate::draft_run(args, opened, engine);
    }
    let cfg = &opened.cfg;
    let layers = engine.weights.layers.len();
    ensure!(!args.nll || engine.full_prefill_logits, "--nll needs full prefill logits");
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let mut placement = engine::Allocator::new(engine.pages, engine.slots).admit(tokens.len() + args.bench_decode)?;
    let embed = embed_rows(&opened.checkpoint, &tokens, cfg.hidden)?;
    let row = cfg.hidden * 2;
    let stream_row = row * 4;
    let prefill = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
    // Rows [first, first + n) of layer `layer`'s golden streams.
    let compare = |layer: usize, first: usize, streams: &[u8], worst: &mut Vec<f64>| -> Result<()> {
        let path = args.golden.join(format!("layer{layer:02}.bin"));
        if let Ok(golden) = std::fs::read(&path) {
            ensure!(golden.len() >= first * stream_row + streams.len(), "golden layer {layer} is short");
            let golden = &golden[first * stream_row..][..streams.len()];
            let (ours, theirs) = (bf16s(streams), bf16s(golden));
            let (cosine, rel) = similarity(&ours, &theirs);
            if worst.len() <= layer {
                worst.resize(layer + 1, 1.0);
            }
            worst[layer] = worst[layer].min(cosine);
            if first == 0 {
                let mut rows: Vec<f64> = ours.chunks_exact(row * 2).zip(theirs.chunks_exact(row * 2))
                    .map(|(a, b)| similarity(a, b).0).collect();
                rows.sort_by(f64::total_cmp);
                let bad = rows.iter().filter(|&&c| c < 0.999).count();
                println!("layer {layer:2} ({:?}): cosine {cosine:.6} rel_l2 {rel:.3e} | rows: median {:.6} p1 {:.6} \
                    worst {:.6}, {bad} of {} below 0.999", cfg.attention[layer], rows[rows.len() / 2],
                    rows[rows.len() / 100], rows[0], rows.len());
            }
        }
        Ok(())
    };
    let mut worst = Vec::new();
    let started = Instant::now();
    let forced = |layer: usize| -> Option<Vec<u8>> {
        std::fs::read(args.golden.join(format!("layer{layer:02}.bin"))).ok().map(|rows| rows[..prefill * stream_row].to_vec())
    };
    // Prefill in chunks of the engine's prefill rows (teacher forcing needs one chunk).
    ensure!(!args.teacher_force || prefill <= engine.prefill_rows, "teacher forcing takes one prefill chunk");
    let mut logits: Option<Vec<f32>> = None;
    let mut done = 0;
    // --logits-only prefills without layer downloads (Spark prefill then runs
    // pipelined in lanes, chunks up to the engine's prefill capacity).
    let chunk_rows = if args.logits_only && !args.teacher_force { engine.prefill_capacity() } else { engine.prefill_rows };
    while done < prefill {
        let n = chunk_rows.min(prefill - done);
        let first = done;
        let mut compare_layer = |layer, streams: &[u8]| compare(layer, first, streams, &mut worst);
        let on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>> =
            if args.logits_only && !args.teacher_force { None } else { Some(&mut compare_layer) };
        let chunk = engine.prefill_forced(&mut placement, &embed[done * row..(done + n) * row], on_layer,
            args.teacher_force.then_some(&forced as &dyn Fn(usize) -> Option<Vec<u8>>), args.nll)?;
        logits = match (logits, chunk) {
            (Some(mut all), Some(more)) if args.nll => {
                all.extend(more);
                Some(all)
            }
            (_, chunk) => chunk,
        };
        done += n;
    }
    let prefill_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut decode_worst = Vec::new();
    let mut decode_logits: Vec<f32> = Vec::new();
    let mut position = prefill;
    while position < tokens.len() {
        let n = args.step_rows.min(tokens.len() - position);
        let first = position;
        let mut compare_layer = |layer, streams: &[u8]| compare(layer, first, streams, &mut decode_worst);
        let on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>> =
            if args.logits_only { None } else { Some(&mut compare_layer) };
        let step = &embed[position * row..(position + n) * row];
        let logits = if args.spec_steps {
            let logits = engine.verify_spec(&mut [(&mut placement, n)], step)?;
            engine.commit(&[(placement.slot, 0, n)])?;
            logits
        } else {
            engine.verify(&mut [(&mut placement, n)], step, on_layer)?
        };
        if let Some(logits) = logits {
            decode_logits.extend(logits);
        }
        position += n;
    }
    let golden_logits = || -> Result<Vec<f32>> {
        Ok(std::fs::read(args.golden.join("logits.bin"))?
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    };
    let vocab = cfg.vocab_size;
    if prefill < tokens.len() {
        println!("decode: {} rows in steps of {} in {:.2} s; worst row-block cosine per layer {:?}",
            tokens.len() - prefill, args.step_rows, started.elapsed().as_secs_f64(),
            decode_worst.iter().map(|c| format!("{c:.6}")).collect::<Vec<_>>());
        if !decode_logits.is_empty() {
            let golden = golden_logits()?;
            let (agree, next_ok, golden_next, nll, scored) = score(&decode_logits, &golden, &tokens, prefill, vocab);
            let (_, _, _, golden_nll, _) =
                score(&golden[prefill * vocab..(prefill * vocab + decode_logits.len())], &golden, &tokens, prefill, vocab);
            let rows = decode_logits.len() / vocab;
            println!("decode logits: top-1 agreement {:.2}% over {rows} rows | next-token accuracy engine {:.1}% \
                golden {:.1}% | mean NLL engine {:.4} golden {:.4} | mean KL(golden||engine) {:.5}",
                100.0 * agree as f64 / rows as f64, 100.0 * next_ok as f64 / scored.max(1) as f64,
                100.0 * golden_next as f64 / scored.max(1) as f64, nll / scored.max(1) as f64,
                golden_nll / scored.max(1) as f64, mean_kl(&decode_logits, &golden, prefill, vocab));
        }
    }
    if args.bench_prefill > 0 {
        *engine.profile.borrow_mut() = [0.0; 3];
        engine.op_profile()?;
        let n = args.bench_prefill_tokens.unwrap_or(prefill.min(engine.prefill_capacity()));
        let long: Vec<u8> = embed.chunks_exact(row).cycle().take(n).flatten().copied().collect();
        let mut allocator = engine::Allocator::new(engine.pages, engine.slots);
        let _held = allocator.admit(tokens.len() + args.bench_decode)?;
        let mut times = Vec::new();
        for _ in 0..args.bench_prefill {
            let mut fresh = allocator.admit(n)?;
            let started = Instant::now();
            for chunk in long.chunks(engine.prefill_capacity() * row) {
                engine.prefill(&mut fresh, chunk, None)?;
            }
            times.push(started.elapsed().as_secs_f64());
            allocator.release(fresh);
        }
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        let phases = std::mem::take(&mut *engine.profile.borrow_mut());
        println!("prefill bench phases per prefill: GPU until the expert exchange {:.1} ms, Spark exchange {:.1} ms",
            1e3 * phases[0] / times.len() as f64, 1e3 * phases[1] / times.len() as f64);
        println!("prefill bench: {n} tokens through {layers} layers, median {:.1} ms ({:.0} tok/s), min {:.1} ms",
            1e3 * median, n as f64 / median, 1e3 * times[0]);
        let ops = engine.op_profile()?;
        if !ops.is_empty() {
            let runs = times.len() as f64;
            let total: f64 = ops.iter().filter(|(k, _)| !k.starts_with("host")).map(|(_, v)| v.0).sum();
            println!("prefill ops per prefill (GPU ms between events, {total:.1} ms total per {runs} runs):");
            let mut rows: Vec<_> = ops.into_iter().collect();
            rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
            for (label, (ms, count)) in rows {
                println!("  {label:44} {:9.2} ms  {:6.1}%  x{}", ms / runs, 100.0 * ms / total, count as f64 / runs);
            }
        }
    }
    if args.bench_decode > 0 {
        *engine.profile.borrow_mut() = [0.0; 3];
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32);
        let mut token = tokens[placement.len.min(tokens.len() - 1)];
        let mut times = Vec::new();
        let mut produced = Vec::new();
        for _ in 0..args.bench_decode {
            let started = Instant::now();
            let row = embed_rows(&opened.checkpoint, &[token], cfg.hidden)?;
            let logits = engine.verify(&mut [(&mut placement, 1)], &row, None)?;
            times.push(started.elapsed().as_secs_f64());
            if let Some(logits) = logits {
                token = argmax(&logits);
                produced.push(token);
            }
        }
        times.sort_by(f64::total_cmp);
        let profile = engine.profile.borrow();
        let steps = times.len() as f64;
        println!("decode bench: {} steps through {layers} layers, median {:.2} ms (min {:.2}, max {:.2}); \
            per step: GPU until the expert exchanges {:.2} ms, Spark exchanges {:.2} ms, head {:.2} ms; \
            tokens {:?}", times.len(),
            1e3 * times[times.len() / 2], 1e3 * times[0], 1e3 * times[times.len() - 1], 1e3 * profile[0] / steps,
            1e3 * profile[1] / steps, 1e3 * profile[2] / steps, &produced[..produced.len().min(16)]);
    }
    let loads = match engine.experts() {
        Some(engine::Experts::Local(local)) => format!(", {} FP8 expert layer loads", local.loads.borrow()),
        Some(engine::Experts::LocalExl3(local)) => format!(", {} EXL3 expert layer loads", local.loads.borrow()),
        _ => String::new(),
    };
    println!("prefill: {prefill} tokens through {layers} layers in {prefill_seconds:.2} s{loads}");
    if let Some(logits) = logits {
        let golden = golden_logits()?;
        if args.nll {
            let (agree, next_ok, golden_next, nll, scored) = score(&logits, &golden, &tokens, 0, vocab);
            let (_, _, _, golden_nll, _) = score(&golden[..prefill * vocab], &golden, &tokens, 0, vocab);
            println!("prefill logits: top-1 agreement {:.1}% over {prefill} rows | next-token accuracy engine {:.1}% \
                golden {:.1}% | mean NLL engine {:.4} golden {:.4}", 100.0 * agree as f64 / prefill as f64,
                100.0 * next_ok as f64 / scored.max(1) as f64, 100.0 * golden_next as f64 / scored.max(1) as f64,
                nll / scored.max(1) as f64, golden_nll / scored.max(1) as f64);
            println!("prefill logits: mean KL(golden||engine) {:.5}", mean_kl(&logits, &golden, 0, vocab));
        }
        let logits = &logits[logits.len() - vocab..];
        let last = &golden[(prefill - 1) * vocab..][..vocab];
        let (cosine, _) = similarity(logits, last);
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        println!("last-row logits: argmax engine {} golden {} cosine {cosine:.6}", argmax(logits), argmax(last));
    }
    Ok(())
}
