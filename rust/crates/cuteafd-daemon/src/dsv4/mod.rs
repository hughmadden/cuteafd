//! DeepSeek V4 (Flash / Pro) coordinator engine over the exported b12x programs.
pub(crate) mod engine;
pub(crate) mod metadata;
pub(crate) mod pool;
pub(crate) mod serve;
pub(crate) mod weights;

use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::deepseek_v4::DeepseekV4Config;
use cuteafd_transport::v41_expert::V41Tp4Roce;
use cuteafd_transport::TcpTransportConfig;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// What every DeepSeek V4 command needs to stand up the engine.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct EngineArgs {
    /// Checkpoint snapshot directory.
    #[arg(long)]
    pub snapshot: PathBuf,
    /// Spark expert ranks in TP order, comma-separated HOST:PORT.
    #[arg(long)]
    pub peers: String,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    /// dsv4_programs.json written by the exporter next to the library.
    #[arg(long, default_value = "/opt/cuteafd/share/DSV4_PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    #[arg(long, default_value_t = 188)]
    pub sms: u32,
    /// Sequences that can be resident at once (compressor state slots).
    #[arg(long, default_value_t = 8)]
    pub max_sequences: usize,
    /// Total tokens the compressed-cache pools hold across sequences.
    #[arg(long, default_value_t = 262_144)]
    pub pool_tokens: usize,
}

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Directory with tokens.bin, layerNN.bin and logits.bin from golden.py.
    #[arg(long)]
    pub golden: PathBuf,
    /// Compare only the first N layers' streams (all logits still compared).
    #[arg(long)]
    pub layers: Option<usize>,
    /// Prefill only the first N tokens and decode the rest one at a time
    /// (teacher-forced), comparing every decode row with the golden logits.
    #[arg(long)]
    pub prefill: Option<usize>,
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
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

pub(crate) fn embed_rows(catalog: &cuteafd_loader::OfficialV41Catalog, tokens: &[u32], hidden: usize) -> Result<Vec<u8>> {
    let tensor = catalog.tensor("embed.weight")?;
    let file = std::fs::File::open(catalog.snapshot().join(&tensor.shard))?;
    let row = hidden * 2;
    let mut out = vec![0u8; tokens.len() * row];
    for (slot, token) in out.chunks_exact_mut(row).zip(tokens) {
        file.read_exact_at(slot, tensor.metadata.byte_offset + u64::from(*token) * row as u64)?;
    }
    Ok(out)
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || golden(args)).await?
}

/// Everything the engine borrows, built on the calling (blocking) thread.
pub(crate) struct Loaded {
    pub snapshot: PathBuf,
    pub catalog: cuteafd_loader::OfficialV41Catalog,
    pub library: NativeLibrary,
    pub cfg: DeepseekV4Config,
    pub family: &'static str,
    pub manifest: serde_json::Value,
}

pub(crate) fn load(args: &EngineArgs) -> Result<Loaded> {
    let catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    let geometry = catalog.routed_experts().geometry()?;
    cuteafd_core::set_expert_geometry(geometry).map_err(|g| anyhow::anyhow!("geometry already {g:?}"))?;
    let family = match geometry.family() {
        Some("dsv4f") => "dsv4f",
        Some("dsv4p") => "dsv4p",
        other => anyhow::bail!("not a DeepSeek V4 checkpoint (expert family {other:?})"),
    };
    let cfg = DeepseekV4Config::read(&args.snapshot, 1)?;
    let library = unsafe { NativeLibrary::load(&args.native_lib) }?;
    library.cuda_set_device(args.device)?;
    let manifest = serde_json::from_str(&std::fs::read_to_string(&args.manifest)
        .with_context(|| format!("reading {}", args.manifest.display()))?)?;
    Ok(Loaded { snapshot: args.snapshot.clone(), catalog, library, cfg, family, manifest })
}

/// Builds the engine over `loaded` and hands it, with a Spark transport and a
/// runtime for it, to `body`.
pub(crate) fn with_engine<T>(
    loaded: &Loaded,
    args: &EngineArgs,
    body: impl FnOnce(&engine::Engine<'_>, &mut V41Tp4Roce, &tokio::runtime::Runtime) -> Result<T>,
) -> Result<T> {
    let programs = loaded.library.dsv4_programs()?.with_manifest(&args.manifest)?;
    let caps = &loaded.manifest["capacities"];
    let stream = loaded.library.cuda_stream_create()?;
    let started = Instant::now();
    let loader = weights::WeightLoader {
        library: &loaded.library, catalog: &loaded.catalog, programs: &programs, family: loaded.family, stream,
    };
    let model = loader.model(&loaded.cfg)?;
    tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "DeepSeek V4 coordinator weights resident");
    let max_context = caps["max_context"].as_u64().context("manifest max_context")? as usize;
    let shape = pool::PoolShape::new(
        args.max_sequences,
        max_context,
        caps["prefill_rows"].as_u64().context("prefill_rows")? as usize,
        (args.pool_tokens / 4).div_ceil(64) + args.max_sequences,
        (args.pool_tokens / 128).div_ceil(2) + args.max_sequences,
    );
    let engine = engine::Engine::new(engine::EngineParts {
        library: &loaded.library,
        programs: &programs,
        cfg: loaded.cfg.clone(),
        weights: model,
        family: loaded.family,
        decode_rows: caps["decode_rows"].as_u64().context("decode_rows")? as usize,
        prefill_rows: caps["prefill_rows"].as_u64().context("prefill_rows")? as usize,
        c128_width: max_context.div_ceil(128).div_ceil(64) * 64,
        max_context,
        stream,
        sms: args.sms,
        shape,
    })?;
    let peers = args.peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
    let executors: Vec<u64> = (1..=peers.len() as u64).collect();
    let mut transport = V41Tp4Roce::new_ranks(&peers, &executors, 4096,
        TcpTransportConfig { timing: false, timeout: Duration::from_secs(120), max_frame_bytes: 64 << 20 })?;
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let result = body(&engine, &mut transport, &runtime);
    drop(engine);
    unsafe { loaded.library.cuda_stream_destroy(stream)? };
    result
}

fn golden(args: GoldenArgs) -> Result<()> {
    let loaded = load(&args.engine)?;
    let cfg = loaded.cfg.clone();
    with_engine(&loaded, &args.engine, |engine, transport, runtime| golden_run(&args, &loaded.catalog, &cfg, engine, transport, runtime))
}

fn golden_run(
    args: &GoldenArgs,
    catalog: &cuteafd_loader::OfficialV41Catalog,
    cfg: &DeepseekV4Config,
    engine: &engine::Engine<'_>,
    transport: &mut V41Tp4Roce,
    runtime: &tokio::runtime::Runtime,
) -> Result<()> {
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let embed = embed_rows(catalog, &tokens, cfg.dim)?;
    let started = Instant::now();
    let compare_layers = args.layers.unwrap_or(cfg.n_layers);
    let prefill = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
    let mut allocator = pool::PoolAllocator::new(engine.shape);
    let mut placement = allocator.admit(tokens.len())?;
    let row = cfg.dim * 2;
    let mut logits = engine.prefill(&mut placement, &tokens[..prefill], &embed[..prefill * row], transport, runtime,
        |layer, stream| {
            if layer < compare_layers && prefill == tokens.len() {
                let golden = std::fs::read(args.golden.join(format!("layer{layer:02}.bin")))?;
                ensure!(golden.len() == stream.len(), "golden layer {layer} has {} bytes, engine {}", golden.len(), stream.len());
                let (cosine, rel) = similarity(&bf16s(stream), &bf16s(&golden));
                println!("layer {layer:2}: cosine {cosine:.6} rel_l2 {rel:.3e}");
            }
            Ok(())
        })?;
    let prefill_elapsed = started.elapsed();
    *engine.profile.borrow_mut() = engine::Profile::default();
    let decode_started = Instant::now();
    for position in prefill..tokens.len() {
        let token = tokens[position];
        logits.extend(engine.decode(&mut [(&mut placement, token)], &embed[position * row..][..row], transport, runtime)?);
    }
    let decode_steps = tokens.len() - prefill;
    if decode_steps > 0 {
        println!("decode host phases: {}", engine.profile.borrow().report());
        println!("decode: {decode_steps} steps in {:.2} s ({:.1} ms/token)", decode_started.elapsed().as_secs_f64(),
            decode_started.elapsed().as_secs_f64() * 1e3 / decode_steps as f64);
    }
    println!("prefill: {prefill} tokens in {:.2} s", prefill_elapsed.as_secs_f64());
    let vocab = cfg.vocab_size;
    let golden = f32s(&std::fs::read(args.golden.join("logits.bin"))?);
    ensure!(golden.len() == logits.len(), "golden logits {} vs engine {}", golden.len(), logits.len());
    let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap();
    let (mut agree, mut next_ok, mut golden_next_ok) = (0usize, 0usize, 0usize);
    for (row, (ours, theirs)) in logits.chunks_exact(vocab).zip(golden.chunks_exact(vocab)).enumerate() {
        let (a, b) = (argmax(ours), argmax(theirs));
        agree += usize::from(a == b);
        if row + 1 < tokens.len() {
            next_ok += usize::from(a == tokens[row + 1] as usize);
            golden_next_ok += usize::from(b == tokens[row + 1] as usize);
        }
    }
    let t = tokens.len();
    let (cosine, rel) = similarity(&logits[(t - 1) * vocab..], &golden[(t - 1) * vocab..]);
    println!(
        "logits: top-1 agreement {:.1}% | next-token accuracy engine {:.1}% golden {:.1}% | last row cosine {cosine:.6} rel_l2 {rel:.3e}",
        100.0 * agree as f64 / t as f64,
        100.0 * next_ok as f64 / (t - 1) as f64,
        100.0 * golden_next_ok as f64 / (t - 1) as f64,
    );
    if decode_steps > 0 {
        let (mut decode_agree, mut worst) = (0usize, 1f64);
        for row in prefill..t {
            let (ours, theirs) = (&logits[row * vocab..][..vocab], &golden[row * vocab..][..vocab]);
            decode_agree += usize::from(argmax(ours) == argmax(theirs));
            worst = worst.min(similarity(ours, theirs).0);
        }
        println!("decode rows: top-1 agreement {:.1}% | worst row cosine {worst:.6}",
            100.0 * decode_agree as f64 / decode_steps as f64);
    }
    Ok(())
}
