//! Kimi K3 coordinator operands from the installed K6 checkpoint
//! (CUTEAFD_PUSH_KIMI) on real GPUs (CUTEAFD_NATIVE_LIB): both global ranks
//! (embedding; native head, output mix and norm), the dense KDA layer 0, the
//! first routed KDA layer and the first MLA layer in each placement it admits,
//! on both coordinator GPUs. Digests are pinned to the loader as it was before
//! push records (CUTEAFD_KIMI_GOLDEN overrides the golden path;
//! CUTEAFD_KIMI_GOLDEN_WRITE=1 records it, to CUTEAFD_KIMI_GOLDEN_OUT when set).
use super::*;
use crate::shared::memory::device::Device;
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::{placement::FfnMode, plan::Checkpoint};
use sha2::{Digest, Sha256};

pub(crate) type Digests = BTreeMap<String, String>;

fn digest(library: &NativeLibrary, allocation: &Allocation<'_>) -> Result<String> {
    let mut bytes = vec![0u8; allocation.buffer.bytes];
    allocation.device.run(|| library.copy_d2h(&mut bytes, allocation.buffer))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

pub(crate) fn snapshot() -> Option<std::path::PathBuf> {
    std::env::var_os("CUTEAFD_PUSH_KIMI").map(std::path::PathBuf::from)
}

pub(crate) fn library() -> Result<NativeLibrary> {
    let path = std::env::var("CUTEAFD_NATIVE_LIB").context("set CUTEAFD_NATIVE_LIB")?;
    // SAFETY: the slot's native library, loaded once for this test process.
    unsafe { NativeLibrary::load(path) }
}

fn path() -> std::path::PathBuf {
    std::env::var_os("CUTEAFD_KIMI_GOLDEN").map(std::path::PathBuf::from).unwrap_or_else(||
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/kimi_k3/coordinator_golden.json"))
}

pub(crate) fn check(actual: &Digests) -> Result<()> {
    if std::env::var_os("CUTEAFD_KIMI_GOLDEN_WRITE").is_some() {
        let out = std::env::var_os("CUTEAFD_KIMI_GOLDEN_OUT").map(std::path::PathBuf::from).unwrap_or_else(path);
        std::fs::write(&out, serde_json::to_vec_pretty(actual)?)?;
        eprintln!("wrote {} operand digests to {}", actual.len(), out.display());
        return Ok(());
    }
    let golden: Digests = serde_json::from_slice(&std::fs::read(path())?)?;
    ensure!(actual.keys().eq(golden.keys()), "operand inventory changed: {} actual, {} golden",
        actual.len(), golden.len());
    let differ: Vec<_> = actual.iter().filter(|(k, v)| golden.get(*k) != Some(v)).map(|(k, _)| k.clone()).collect();
    ensure!(differ.is_empty(), "{} operands differ from the golden, first {:?}", differ.len(),
        &differ[..differ.len().min(8)]);
    eprintln!("PASS {} operand digests match the golden", actual.len());
    Ok(())
}

/// The golden's layers and their placements: the dense KDA layer 0, the first
/// routed KDA layer, and the first MLA layer split by heads and whole on
/// either GPU.
pub(crate) fn golden_layers(cfg: &KimiK3Config) -> Result<Vec<(usize, LayerMode)>> {
    let kda = (0..cfg.layers).find(|&l| cfg.attention[l] == KimiAttention::Kda && !cfg.dense[l])
        .context("no routed KDA layer")?;
    let mla = (0..cfg.layers).find(|&l| cfg.attention[l] == KimiAttention::Mla).context("no MLA layer")?;
    ensure!(cfg.attention[0] == KimiAttention::Kda && cfg.dense[0], "Kimi layer 0 is a dense KDA layer");
    Ok(vec![(0, LayerMode::HeadSplit), (kda, LayerMode::HeadSplit), (mla, LayerMode::HeadSplit),
        (mla, LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }), (mla, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split })])
}

fn mode_tag(mode: LayerMode) -> String {
    match mode {
        LayerMode::HeadSplit => "heads".into(),
        LayerMode::Whole { gpu, .. } => format!("whole{gpu}"),
        other => format!("{other:?}"),
    }
}

fn core_digests(library: &NativeLibrary, prefix: &str, core: &CoreWeights<'_>, out: &mut Digests) -> Result<()> {
    for (part, allocation) in [("payload", &core.payload), ("suh", &core.suh), ("svh", &core.svh)] {
        out.insert(format!("{prefix}.{part}"), digest(library, allocation)?);
    }
    Ok(())
}

pub(crate) fn global_digests(library: &NativeLibrary, rank: usize, weights: &GlobalWeights<'_>,
    out: &mut Digests) -> Result<()> {
    for (name, allocation) in &weights.operands {
        out.insert(format!("global.{rank}.{name}"), digest(library, allocation)?);
    }
    if let Some(head) = &weights.core_head { core_digests(library, &format!("global.{rank}.head"), head, out)?; }
    Ok(())
}

pub(crate) fn layer_digests(library: &NativeLibrary, prefix: &str, weights: &LayerWeights<'_>,
    out: &mut Digests) -> Result<()> {
    for (name, allocation) in &weights.operands {
        out.insert(format!("{prefix}.{name}"), digest(library, allocation)?);
    }
    for (name, core) in &weights.core { core_digests(library, &format!("{prefix}.{name}"), core, out)?; }
    Ok(())
}

pub(crate) struct Installed {
    pub checkpoint: Checkpoint,
    pub cfg: KimiK3Config,
    pub core: CoreInventory,
}

pub(crate) fn installed(snapshot: &std::path::Path) -> Result<Installed> {
    let checkpoint = Checkpoint::coordinator(snapshot, false, false)?;
    let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
    let core = cuteafd_loader::plan::families::kimi::core_inventory(&checkpoint)?;
    ensure!(!core.projections.is_empty(), "the golden checkpoint carries the K6 core");
    Ok(Installed { checkpoint, cfg, core })
}

/// Every golden operand through the coordinator loader, each owner on its GPU.
pub(crate) fn mount(library: &NativeLibrary, model: &Installed, out: &mut Digests) -> Result<()> {
    let devices = [0, 1].map(|id| Device { library, id });
    let read = |name: &str| read_tensor(&model.checkpoint, name);
    let globals = GlobalLayout::pair_with_core(&model.cfg, &model.core)?;
    for (rank, layout) in globals.iter().enumerate() {
        let weights = layout.upload(devices[rank], layout.pack(&model.cfg, read)?)?;
        global_digests(library, rank, &weights, out)?;
    }
    for (layer, mode) in golden_layers(&model.cfg)? {
        let layouts = LayerLayout::pair_with_core(&model.cfg, layer, mode, &model.core)?;
        let packed = PackedLayer::pair_with_core(&model.cfg, layer, mode, &model.core, read)?;
        for ((rank, packed), layout) in packed.into_iter().enumerate().zip(&layouts) {
            let weights = LayerWeights::upload(devices[rank], packed, layout, layout.bytes()?)?;
            layer_digests(library, &format!("layers.{layer}.{}.{rank}", mode_tag(mode)), &weights, out)?;
        }
    }
    Ok(())
}

fn peers(library: &NativeLibrary) -> Result<()> {
    library.cuda_set_device(0)?;
    for gpu in [0, 1] { Device { library, id: gpu }.run(|| library.cuda_enable_peer(1 - gpu))?; }
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_KIMI, CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let model = installed(&snapshot)?;
    let library = library()?;
    peers(&library)?;
    let mut actual = Digests::new();
    mount(&library, &model, &mut actual)?;
    check(&actual)
}
