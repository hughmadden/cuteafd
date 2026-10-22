//! MiMo coordinator operands from the installed checkpoint (CUTEAFD_PUSH_MIMO)
//! on real RTX GPUs: decoder layers 0 (dense MLP) and 1 (SWA + router), the
//! final norm and head on a two-GPU head split and on one GPU, and MTP layer 0
//! with its extras, under the checkpoint's resolved FP8 formats. Digests are
//! pinned to the mount loader as it was before push recipes, so the push
//! recipes (landings, staged row quantization, scale gathers at Done) must
//! reproduce it bit for bit. CUTEAFD_MIMO_GOLDEN_WRITE=1 records instead.
use super::weights::{MimoLayer, MimoLoader, MimoWeights};
use crate::shared::memory::device::Device;
use anyhow::{Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::mimo_v2::{projection::MimoProjectionRepresentation, MimoV2Config};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub(crate) type Digests = BTreeMap<String, String>;

pub(crate) fn digest(library: &NativeLibrary, buffer: CuteafdDeviceBuffer) -> Result<String> {
    let mut bytes = vec![0u8; buffer.bytes];
    Device { library, id: buffer.device_id }.run(|| library.copy_d2h(&mut bytes, buffer))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

pub(crate) fn layer_digests(library: &NativeLibrary, prefix: &str, layer: &MimoLayer<'_>, out: &mut Digests)
    -> Result<()> {
    for (name, buffer) in layer.buffers() {
        let key = format!("{prefix}.{name}");
        anyhow::ensure!(out.insert(key.clone(), digest(library, buffer)?).is_none(), "duplicate {key}");
    }
    Ok(())
}

pub(crate) fn weights_digests(library: &NativeLibrary, prefix: &str, weights: &MimoWeights<'_>, out: &mut Digests)
    -> Result<()> {
    for (layer, share) in weights.layers.iter().enumerate() {
        layer_digests(library, &format!("{prefix}.layers.{layer}"), share, out)?;
    }
    out.insert(format!("{prefix}.norm"), digest(library, weights.norm.buffer)?);
    for (i, allocation) in weights.head.allocations().into_iter().enumerate() {
        out.insert(format!("{prefix}.head.{i}"), digest(library, allocation.buffer)?);
    }
    Ok(())
}

pub(crate) fn snapshot() -> Option<std::path::PathBuf> {
    std::env::var_os("CUTEAFD_PUSH_MIMO").map(std::path::PathBuf::from)
}

pub(crate) fn library() -> Result<NativeLibrary> {
    let path = std::env::var("CUTEAFD_NATIVE_LIB").context("set CUTEAFD_NATIVE_LIB")?;
    // SAFETY: the slot's native library, loaded once for this test process.
    unsafe { NativeLibrary::load(path) }
}

/// The golden's layers, the formats serving resolves for them, and a loader.
pub(crate) struct Setup {
    pub cfg: MimoV2Config,
    pub checkpoint: cuteafd_loader::plan::checkpoint::Checkpoint,
    pub formats: BTreeMap<String, MimoProjectionRepresentation>,
    pub checkpoint_tp: usize,
}

pub(crate) const LAYERS: usize = 2;

pub(crate) fn setup(snapshot: &std::path::Path) -> Result<Setup> {
    let cfg = MimoV2Config::read(snapshot)?;
    let checkpoint = cuteafd_loader::plan::checkpoint::Checkpoint::coordinator(snapshot, false, true)?;
    let formats = (0..LAYERS).map(|l| format!("model.layers.{l}.self_attn.o_proj.weight"))
        .chain(std::iter::once("model.mtp.layers.0.self_attn.o_proj.weight".to_string()))
        .map(|name| (name, MimoProjectionRepresentation::Fp8)).collect();
    Ok(Setup { cfg, checkpoint_tp: cuteafd_loader::families::mimo_v2::checkpoint_tp(snapshot)?, checkpoint, formats })
}

pub(crate) fn loader<'a>(library: &'a NativeLibrary, setup: &'a Setup, peers: usize) -> MimoLoader<'a> {
    MimoLoader { library, checkpoint: &setup.checkpoint, stream: std::ptr::null_mut(),
        checkpoint_tp: setup.checkpoint_tp, fp8_head: true, output_formats: &setup.formats,
        fp8_scales: crate::shared::fp8_linear::Fp8Scales::Amax, device: 0,
        peers: (1..=peers).map(|device| crate::shared::peer_split::RankDevice { device: device as i32,
            stream: std::ptr::null_mut() }).collect() }
}

/// Every golden operand through `loader`s (two-GPU head split, then one GPU).
pub(crate) fn mount(library: &NativeLibrary, setup: &Setup, out: &mut Digests) -> Result<()> {
    let split = loader(library, setup, 1);
    let (weights, shares) = split.model(&setup.cfg, LAYERS)?;
    weights_digests(library, "split.rank0", &weights, out)?;
    for (rank, share) in shares.iter().enumerate() {
        for (layer, part) in share.iter().enumerate() {
            layer_digests(library, &format!("split.rank{}.layers.{layer}", rank + 1), part, out)?;
        }
    }
    drop((weights, shares));
    let mtp = split.mtp_layer(&setup.cfg, 0)?;
    layer_digests(library, "mtp0", &mtp, out)?;
    for (i, extra) in split.mtp_extras(0)?.iter().enumerate() { out.insert(format!("mtp0.extra{i}"), digest(library, extra.buffer)?); }
    drop(mtp);
    let single = loader(library, setup, 0);
    let (weights, _) = single.model(&setup.cfg, LAYERS)?;
    weights_digests(library, "single", &weights, out)?;
    Ok(())
}

pub(crate) fn path() -> std::path::PathBuf {
    std::env::var_os("CUTEAFD_MIMO_GOLDEN").map(std::path::PathBuf::from).unwrap_or_else(||
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/mimo_v2/coordinator_golden.json"))
}

pub(crate) fn check(actual: &Digests, subset: bool) -> Result<()> {
    if std::env::var_os("CUTEAFD_MIMO_GOLDEN_WRITE").is_some() {
        let out = std::env::var_os("CUTEAFD_MIMO_GOLDEN_OUT").map(std::path::PathBuf::from).unwrap_or_else(path);
        std::fs::write(&out, serde_json::to_vec_pretty(actual)?)?;
        eprintln!("wrote {} operand digests to {}", actual.len(), out.display());
        return Ok(());
    }
    let golden: Digests = serde_json::from_slice(&std::fs::read(path())?)?;
    if !subset {
        anyhow::ensure!(actual.keys().eq(golden.keys()), "operand inventory changed: {} actual, {} golden",
            actual.len(), golden.len());
    }
    let differ = actual.iter().filter(|(k, v)| golden.get(*k) != Some(v)).map(|(k, _)| k.clone()).collect::<Vec<_>>();
    anyhow::ensure!(differ.is_empty(), "{} operands differ from the golden, first {:?}", differ.len(),
        &differ[..differ.len().min(8)]);
    eprintln!("PASS {} operand digests match the golden", actual.len());
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_MIMO, CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let setup = setup(&snapshot)?;
    let cfg = &setup.cfg;
    let first_layer = cfg.dense.iter().position(|dense| !dense).unwrap_or(cfg.layers);
    let _ = cuteafd_core::set_expert_geometry(cuteafd_loader::RoutedExpertShape { layers: cfg.layers, first_layer,
        experts: cfg.experts, topk: cfg.topk, hidden: cfg.hidden, intermediate: cfg.moe_intermediate,
        draft_stages: 0, draft_experts: 0 }.geometry()?);
    let library = library()?;
    library.cuda_set_device(0)?;
    for gpu in [0, 1] { Device { library: &library, id: gpu }.run(|| library.cuda_enable_peer(1 - gpu))?; }
    let mut actual = Digests::new();
    mount(&library, &setup, &mut actual)?;
    check(&actual, false)
}
