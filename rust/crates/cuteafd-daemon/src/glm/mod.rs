//! GLM 5.x (glm_moe_dsa) on the generic engine: weights, the coordinator
//! programs' layer chain, and the golden comparison command.
pub(crate) mod engine;
pub(crate) mod weights;

use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::glm_dsa::GlmDsaConfig;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    /// Checkpoint snapshot (the EXL3 publication carries the dense weights).
    #[arg(long)]
    pub snapshot: PathBuf,
    /// Directory with tokens.bin, layerNN.bin and logits.bin from
    /// python/reference/glm_dsa/golden.py.
    #[arg(long)]
    pub golden: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/DSV4_PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// Run only the first N layers (dense layers need no experts).
    #[arg(long)]
    pub layers: Option<usize>,
    #[arg(long, default_value_t = 32768)]
    pub max_context: usize,
    /// Spark expert ranks in TP order (HOST:PORT,...), for MoE layers.
    #[arg(long)]
    pub peers: Option<String>,
}

pub(crate) fn embed_rows(catalog: &cuteafd_loader::OfficialV41Catalog, tokens: &[u32], hidden: usize) -> Result<Vec<u8>> {
    let tensor = catalog.tensor("model.embed_tokens.weight")?;
    let file = std::fs::File::open(catalog.snapshot().join(&tensor.shard))?;
    let row = hidden * 2;
    let mut out = vec![0u8; tokens.len() * row];
    for (slot, token) in out.chunks_exact_mut(row).zip(tokens) {
        file.read_exact_at(slot, tensor.metadata.byte_offset + u64::from(*token) * row as u64)?;
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
    let catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    cuteafd_core::set_expert_geometry(catalog.routed_experts().geometry()?)
        .map_err(|g| anyhow::anyhow!("geometry already {g:?}"))?;
    let cfg = GlmDsaConfig::read(&args.snapshot)?;
    let library = unsafe { NativeLibrary::load(&args.native_lib) }?;
    library.cuda_set_device(args.device)?;
    let programs = library.dsv4_programs()?.with_manifest(&args.manifest)?;
    programs.load_all()?;
    let stream = library.cuda_stream_create()?;
    let started = Instant::now();
    let layers = args.layers.unwrap_or(cfg.layers).min(cfg.layers);
    let loader = weights::GlmLoader { library: &library, catalog: &catalog, stream };
    let model = loader.model(&cfg, layers)?;
    println!("weights for {layers} layers resident in {:.1} s", started.elapsed().as_secs_f64());
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let pages = args.max_context.div_ceil(engine::PAGE_ROWS);
    let engine = engine::GlmEngine::new(&library, &programs, cfg.clone(), model, stream, args.max_context,
        tokens.len().max(1), pages)?;
    let mut placement = engine::GlmPlacement { pages: (0..pages as i32).collect(), len: 0 };
    let embed = embed_rows(&catalog, &tokens, cfg.hidden)?;
    let started = Instant::now();
    let mut compare = |layer: usize, stream: &[u8]| -> Result<()> {
        let path = args.golden.join(format!("layer{layer:02}.bin"));
        if let Ok(golden) = std::fs::read(&path) {
            ensure!(golden.len() == stream.len(), "golden layer {layer} has {} bytes, engine {}", golden.len(), stream.len());
            let (cosine, rel) = similarity(&bf16s(stream), &bf16s(&golden));
            println!("layer {layer:2}: cosine {cosine:.6} rel_l2 {rel:.3e}");
        }
        Ok(())
    };
    let mut transport = match args.peers.as_deref() {
        Some(peers) => {
            let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
            let executors: Vec<u64> = (0..peers.len())
                .map(|rank| cuteafd_transport::v41_expert::v41_spark_executor_id(peers.len(), rank))
                .collect::<Result<_>>()?;
            Some(cuteafd_transport::v41_expert::V41Tp4Roce::new_ranks(&peers, &executors, 4096,
                cuteafd_transport::TcpTransportConfig { timing: false, timeout: std::time::Duration::from_secs(120),
                    max_frame_bytes: 64 << 20 })?)
        }
        None => None,
    };
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let experts = transport.as_mut().map(|t| (t, &runtime));
    let logits = engine.prefill(&mut placement, &embed, experts, Some(&mut compare))?;
    println!("prefill: {} tokens through {layers} layers in {:.2} s", tokens.len(), started.elapsed().as_secs_f64());
    if let Some(logits) = logits {
        let golden: Vec<f32> = std::fs::read(args.golden.join("logits.bin"))?
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let last = &golden[(tokens.len() - 1) * cfg.vocab_size..][..cfg.vocab_size];
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).context("empty");
        let (cosine, _) = similarity(&logits, last);
        println!("last-row logits: argmax engine {} golden {} cosine {cosine:.6}", argmax(&logits)?, argmax(last)?);
    }
    Ok(())
}
