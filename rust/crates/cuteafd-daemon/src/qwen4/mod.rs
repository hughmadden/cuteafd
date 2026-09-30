//! Qwen 3.8 Flash Next (qwen4_exp) on the generic engine: weights, the PLE
//! n-gram table, the coordinator programs' layer chain, and the golden
//! comparison command.
pub(crate) mod engine;
mod mtp_golden;
pub(crate) mod mtp_policy;
pub(crate) mod ple;
pub(crate) mod serve;
pub(crate) mod speculate;
pub(crate) mod weights;

use anyhow::{ensure, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::fp8_experts::Fp8ExpertTensors;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_loader::qwen4_exp::Qwen4Config;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::time::Instant;

/// What every Qwen 3.8 Flash Next command needs to stand up the engine.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct EngineArgs {
    /// Checkpoint snapshot (coordinator weights, PLE table, config, tokenizer).
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/DSV4_PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// Run only the first N layers.
    #[arg(long)]
    pub layers: Option<usize>,
    /// Longest sequence (the exported index top-k covers the manifest's max_context).
    #[arg(long, default_value_t = 65_536)]
    pub max_context: usize,
    /// Tokens the K/V record pools hold across sequences.
    #[arg(long, default_value_t = 32_768)]
    pub pool_tokens: usize,
    /// Sequences with GDN/PLE state (about 115 MiB each).
    #[arg(long, default_value_t = 8)]
    pub slots: usize,
    #[arg(long, default_value_t = 4096)]
    pub prefill_rows: usize,
    /// Decode steps of <= 16 rows read E4M3 copies (FP32 128x128 block scales,
    /// quantized at load) of the GDN and attention in/out projections.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub fp8_decode: bool,
    /// Scale rule of the FP8 copies made from BF16 weights at load: amax /
    /// 448, the smallest power of two >= it (pow2), or per block whichever of
    /// the two leaves the smaller error (best).
    #[arg(long, value_enum, default_value_t = crate::fp8_linear::Fp8Scales::Amax)]
    pub fp8_scales: crate::fp8_linear::Fp8Scales,
    #[command(flatten)]
    pub l2: crate::l2_prefetch::L2PrefetchArgs,
    /// Where the PLE n-gram table lives.
    #[arg(long, value_enum, default_value_t = ple::PlePlacement::Host)]
    pub ple: ple::PlePlacement,
    /// Reader threads for the PLE table.
    #[arg(long, default_value_t = 16)]
    pub ple_threads: usize,
    /// Spark ranks in TP order (HOST:PORT,...) serving the routed experts.
    #[arg(long, conflicts_with_all = ["local_experts", "shared_only"])]
    pub peers: Option<String>,
    /// Run the routed experts on this GPU: EXL3 checkpoints through the
    /// coordinator `exl3-qwen4-k45/rtx-tp1` package, FP8 ones through the TP1
    /// `fp8-qwen4` package. `--experts-snapshot` names another checkpoint for
    /// the experts.
    #[arg(long, conflicts_with = "shared_only")]
    pub local_experts: bool,
    /// No routed experts: the MoE output is the shared expert alone (plumbing tests).
    #[arg(long)]
    pub shared_only: bool,
    #[arg(long)]
    pub experts_snapshot: Option<PathBuf>,
    /// TP1 FP8 package directory (default `<libdir>/fp8/fp8-qwen4/tp1`).
    #[arg(long)]
    pub fp8_package: Option<PathBuf>,
    /// FP8 expert layers resident at once with --local-experts (2.4 GiB each).
    #[arg(long, default_value_t = 16)]
    pub expert_window: usize,
    /// Most EXL3 expert layers resident at once (the free memory decides first).
    #[arg(long, default_value_t = 48)]
    pub exl3_window: usize,
    /// GPU memory (GiB) kept free of local experts for step workspaces
    /// (logits alone are 4 GiB at 4096 rows).
    #[arg(long, default_value_t = 12)]
    pub expert_reserve_gib: usize,
    /// Native MTP drafts per step (0: no MTP). Loads the MTP layer (`mtp.*`,
    /// its experts local) and verifies up to this many drafts per sequence.
    #[arg(long, default_value_t = 0)]
    pub mtp: usize,
    /// MTP drafts read an E4M3 copy of lm_head (per-row x 128-K scales, made at
    /// load): half the head's bytes per draft step; verification stays exact.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub mtp_fp8_head: bool,
}

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Directory with tokens.bin, layerNN.bin and logits.bin from
    /// python/reference/qwen4_exp/golden.py.
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
    /// Score every prefill row's logits against the golden (mean NLL, top-1).
    #[arg(long)]
    pub nll: bool,
    /// After the comparison, time this many greedy single-row decode steps.
    #[arg(long, default_value_t = 0)]
    pub bench_decode: usize,
    /// Time this many more prefills of the golden prompt (up to --prefill-rows
    /// tokens) on fresh sequences, without layer downloads.
    #[arg(long, default_value_t = 0)]
    pub bench_prefill: usize,
    /// With --step-rows k: verify each step speculatively and commit only
    /// this many of its rows (the next step starts after them), checking
    /// GDN/PLE verify-by-replay against the golden logits.
    #[arg(long)]
    pub spec_keep: Option<usize>,
    /// Compare the MTP layer (needs --mtp) with the torch reference in this
    /// directory (python/reference/qwen4_exp/mtp.py), teacher forced on the
    /// golden target streams, then stop.
    #[arg(long)]
    pub mtp_oracle: Option<PathBuf>,
    /// Greedy-decode this many tokens after the golden prompt (--prefill N
    /// truncates it) plainly and with MTP speculation at depth --mtp; the
    /// outputs must match. Reports acceptance and step costs, then stops.
    #[arg(long)]
    pub spec_decode: Option<usize>,
}

/// The checkpoint and native library, opened on the calling thread.
pub(crate) struct Opened {
    pub checkpoint: Checkpoint,
    pub cfg: Qwen4Config,
    pub library: NativeLibrary,
    /// The expert catalog for --local-experts.
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
    let cfg = Qwen4Config::read(&args.snapshot)?;
    cfg.check_programs()?;
    // The expert geometry is process-wide and must be fixed before the native
    // library loads (its expert helpers size rows from it).
    let geometry = cuteafd_core::ExpertGeometry::QWEN4_EXP;
    ensure!(geometry.hidden as usize == cfg.hidden && geometry.experts as usize == cfg.experts
        && geometry.topk as usize == cfg.topk && geometry.intermediate as usize == cfg.moe_intermediate,
        "checkpoint experts do not match the Qwen 3.8 Flash Next geometry");
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
    Ok(Opened { checkpoint, cfg, library, experts })
}

impl Opened {
    /// Builds the engine and hands it to `body`.
    pub fn with_engine<T>(&self, args: &EngineArgs, body: impl FnOnce(&engine::Qwen4Engine<'_>) -> Result<T>)
        -> Result<T> {
        let programs = self.library.dsv4_programs()?.with_manifest(&args.manifest)?;
        programs.load_all()?;
        let stream = self.library.cuda_stream_create()?;
        let started = Instant::now();
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let loader = weights::Qwen4Loader { library: &self.library, checkpoint: &self.checkpoint,
            fp8_decode: args.fp8_decode, fp8_scales: args.fp8_scales, stream };
        let model = loader.model(&self.cfg, layers, args.mtp > 0 && layers == self.cfg.layers, args.mtp_fp8_head)?;
        let resident: usize = model.layers.iter().map(weights::Qwen4Layer::bytes).sum::<usize>()
            + model.mtp.as_ref().map_or(0, weights::MtpWeights::bytes);
        tracing::info!(layers, gib = resident as f64 / (1u64 << 30) as f64,
            elapsed_ms = started.elapsed().as_millis() as u64, "Qwen 3.8 Flash Next coordinator weights resident");
        let ple = match self.cfg.ple_layers.first() {
            Some(&layer) if layer < layers => Some(ple::PleTable::load(&self.library, &self.checkpoint, &self.cfg,
                layer, args.ple, args.ple_threads)?),
            _ => None,
        };
        let pages = args.pool_tokens.div_ceil(engine::PAGE_ROWS);
        let mut engine = engine::Qwen4Engine::new(&self.library, &programs, self.cfg.clone(), model, ple, stream,
            args.max_context, args.prefill_rows, pages, args.slots)?;
        if let Some(experts) = self.experts(args, layers)? {
            engine.set_experts(experts);
        }
        if let Some(budget) = args.l2.budget(&self.library, crate::l2_prefetch::OTHER_DEFAULT)? {
            engine.l2 = Some(crate::l2_prefetch::L2Prefetch::new(&self.library, budget, &engine.decode_read_order())?);
        }
        let result = body(&engine);
        drop(engine);
        // SAFETY: the engine that used the stream is gone.
        unsafe { self.library.cuda_stream_destroy(stream)? };
        result
    }

    fn experts<'s>(&'s self, args: &EngineArgs, layers: usize) -> Result<Option<engine::Experts<'s>>> {
        if args.shared_only {
            tracing::warn!("--shared-only: routed experts are skipped (outputs do not match the model)");
            return Ok(Some(engine::Experts::SharedOnly));
        }
        if let Some(tensors) = self.fp8() {
            let directory = args.fp8_package.clone()
                .unwrap_or_else(|| crate::v41_experts::fp8::package_directory(&args.native_lib, 1));
            let (free, _) = self.library.cuda_memory_info()?;
            // An empty window: the package and its scratch; layers load on first use.
            let experts = crate::v41_experts::fp8::Fp8Experts::load(&self.library, tensors, &directory, 0..0, 1, 0,
                args.prefill_rows, free.saturating_sub(args.expert_reserve_gib.min(4) << 30))?;
            return Ok(Some(engine::Experts::Local(engine::LocalExperts {
                library: &self.library, tensors, experts: std::cell::RefCell::new(experts),
                window: args.expert_window.max(1), loads: std::cell::RefCell::new(0),
            })));
        }
        if let Some(catalog) = self.experts.as_ref().filter(|c| c.exl3().is_some()) {
            let (free, _) = self.library.cuda_memory_info()?;
            return Ok(Some(engine::Experts::LocalExl3(engine::LocalExl3 {
                library: &self.library, native_lib: args.native_lib.clone(), catalog,
                resident: std::cell::RefCell::new(None), window: args.exl3_window.max(1), layers,
                mtp: args.mtp > 0 && layers == self.cfg.layers,
                max_rows: args.prefill_rows,
                budget: free.saturating_sub(args.expert_reserve_gib << 30), loads: std::cell::RefCell::new(0),
            })));
        }
        let Some(peers) = args.peers.as_deref() else { return Ok(None) };
        let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
        let executors: Vec<u64> = (0..peers.len())
            .map(|rank| cuteafd_transport::v41_expert::v41_spark_executor_id(peers.len(), rank))
            .collect::<Result<_>>()?;
        let transport = crate::spark_intake::SparkLink::new(&self.library, &peers, &executors,
            u32::try_from(args.prefill_rows)?, cuteafd_transport::TcpTransportConfig { timing: false,
                timeout: std::time::Duration::from_secs(120), max_frame_bytes: 64 << 20 }, self.cfg.hidden * 2)?;
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        Ok(Some(engine::Experts::Spark { transport: std::cell::RefCell::new(transport), runtime }))
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

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || golden(args)).await?
}

fn golden(args: GoldenArgs) -> Result<()> {
    let opened = open(&args.engine)?;
    opened.with_engine(&args.engine, |engine| {
        if let Some(dir) = &args.mtp_oracle {
            return mtp_golden::mtp_oracle(&args, &opened, engine, dir);
        }
        if let Some(count) = args.spec_decode {
            return mtp_golden::spec_decode(&args, &opened, engine, count, args.engine.mtp);
        }
        golden_run(&args, &opened, engine)
    })
}

/// Mean NLL of `logits` rows against the next tokens, and top-1 agreements.
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

fn golden_run(args: &GoldenArgs, opened: &Opened, engine: &engine::Qwen4Engine<'_>) -> Result<()> {
    let cfg = &opened.cfg;
    let layers = engine.weights.layers.len();
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let mut allocator = engine::Allocator::new(engine.pages, engine.slots, cfg);
    let mut placement = allocator.admit(tokens.len() + args.bench_decode)?;
    let embed = embed_rows(&opened.checkpoint, &tokens, cfg.hidden)?;
    let row = cfg.hidden * 2;
    let stream_row = row * 4;
    let prefill = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
    // Rows [first, first + n) of layer `layer`'s golden streams.
    let compare = |layer: usize, first: usize, streams: &[u8], worst: &mut Vec<f64>| -> Result<()> {
        let path = args.golden.join(format!("layer{layer:02}.bin"));
        let Ok(file) = std::fs::File::open(&path) else { return Ok(()) };
        let mut golden = vec![0u8; streams.len()];
        file.read_exact_at(&mut golden, (first * stream_row) as u64)
            .map_err(|e| anyhow::anyhow!("golden layer {layer} is short: {e}"))?;
        let (ours, theirs) = (bf16s(streams), bf16s(&golden));
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
        Ok(())
    };
    let mut worst = Vec::new();
    let started = Instant::now();
    let forced = |layer: usize| -> Option<Vec<u8>> {
        let file = std::fs::File::open(args.golden.join(format!("layer{layer:02}.bin"))).ok()?;
        let mut rows = vec![0u8; prefill * stream_row];
        file.read_exact_at(&mut rows, 0).ok()?;
        Some(rows)
    };
    ensure!(!args.teacher_force || prefill <= engine.prefill_rows, "teacher forcing takes one prefill chunk");
    let mut logits: Option<Vec<f32>> = None;
    let mut done = 0;
    while done < prefill {
        let n = engine.prefill_rows.min(prefill - done);
        let first = done;
        let chunk = engine.prefill_forced(&mut placement, &tokens[done..done + n], &embed[done * row..(done + n) * row],
            Some(&mut |layer, streams| compare(layer, first, streams, &mut worst)),
            args.teacher_force.then_some(&forced as &dyn Fn(usize) -> Option<Vec<u8>>), if args.nll { n } else { 1 })?;
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
    let mut spec_steps = 0usize;
    while position < tokens.len() {
        let n = args.step_rows.min(tokens.len() - position);
        let first = position;
        let rows = &tokens[position..position + n];
        let step_embed = &embed[position * row..(position + n) * row];
        let Some(keep) = args.spec_keep else {
            if let Some(logits) = engine.verify(&mut [(&mut placement, rows)], step_embed,
                Some(&mut |layer, streams| compare(layer, first, streams, &mut decode_worst)))? {
                decode_logits.extend(logits);
            }
            position += n;
            continue;
        };
        // Speculative: verify n rows, keep the first `keep` (the rest are verified again next step).
        let keep = keep.clamp(1, n);
        let history = placement.history.clone();
        let logits = engine.verify_spec(&mut [(&mut placement, rows)], step_embed,
            Some(&mut |layer, streams| compare(layer, first, streams, &mut decode_worst)))?;
        engine.commit(&[(placement.slot, 0, keep)])?;
        engine.rewind(&mut placement, first, history, &rows[..keep])?;
        if let Some(logits) = logits {
            decode_logits.extend_from_slice(&logits[..keep * cfg.vocab_size]);
        }
        spec_steps += 1;
        position += keep;
    }
    if args.spec_keep.is_some() {
        println!("speculative verify: {spec_steps} steps of {} rows, each committing {} (GDN/PLE verify-by-replay)",
            args.step_rows, args.spec_keep.unwrap_or(0));
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
            // Numerics A/B between engine configs (benchmarks): the decode rows' logits, F32.
            if let Ok(path) = std::env::var("CUTEAFD_DUMP_DECODE_LOGITS") {
                std::fs::write(&path, decode_logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            }
            let golden = golden_logits()?;
            let (agree, next_ok, golden_next, nll, scored) = score(&decode_logits, &golden, &tokens, prefill, vocab);
            let rows = decode_logits.len() / vocab;
            println!("decode logits: top-1 agreement {:.1}% over {rows} rows | next-token accuracy engine {:.1}% \
                golden {:.1}% | mean NLL {:.4} | mean KL(golden||engine) {:.5}", 100.0 * agree as f64 / rows as f64,
                100.0 * next_ok as f64 / scored.max(1) as f64, 100.0 * golden_next as f64 / scored.max(1) as f64,
                nll / scored.max(1) as f64, crate::glmf::mean_kl(&decode_logits, &golden, prefill, vocab));
        }
    }
    if args.bench_prefill > 0 {
        let n = prefill.min(engine.prefill_rows);
        let mut times = Vec::new();
        for _ in 0..args.bench_prefill {
            let mut fresh = allocator.admit(n)?;
            let started = Instant::now();
            engine.prefill(&mut fresh, &tokens[..n], &embed[..n * row])?;
            times.push(started.elapsed().as_secs_f64());
            allocator.release(fresh);
        }
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        println!("prefill bench: {n} tokens through {layers} layers, median {:.1} ms ({:.0} tok/s), min {:.1} ms",
            1e3 * median, n as f64 / median, 1e3 * times[0]);
    }
    if args.bench_decode > 0 {
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32);
        let mut token = tokens[placement.len.min(tokens.len() - 1)];
        let mut times = Vec::new();
        let mut produced = Vec::new();
        // FNV-1a over every step's logits bits (bit-identity checks between configs).
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for _ in 0..args.bench_decode {
            let started = Instant::now();
            let row = embed_rows(&opened.checkpoint, &[token], cfg.hidden)?;
            let step = [token];
            let logits = engine.verify(&mut [(&mut placement, &step[..])], &row, None)?;
            times.push(started.elapsed().as_secs_f64());
            if let Some(logits) = logits {
                for v in &logits {
                    digest = (digest ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3);
                }
                token = argmax(&logits);
                produced.push(token);
            }
        }
        times.sort_by(f64::total_cmp);
        let profile = engine.profile.borrow();
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        println!("decode bench: {} steps through {layers} layers, median {:.3} ms (mean {:.3}, min {:.3}, max {:.2}); \
            expert GPU wait {:.1} ms, exchange {:.1} ms total; logits digest {digest:016x}; tokens {:?}", times.len(),
            1e3 * times[times.len() / 2], 1e3 * mean, 1e3 * times[0], 1e3 * times[times.len() - 1], 1e3 * profile[0],
            1e3 * profile[1], &produced[..produced.len().min(16)]);
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
        }
        let logits = &logits[logits.len() - vocab..];
        let last = &golden[(prefill - 1) * vocab..][..vocab];
        let (cosine, _) = similarity(logits, last);
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        println!("last-row logits: argmax engine {} golden {} cosine {cosine:.6}", argmax(logits), argmax(last));
    }
    Ok(())
}
