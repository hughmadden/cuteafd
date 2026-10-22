//! Step-5 coordinator operands from the installed checkpoint (CUTEAFD_PUSH_STEP5)
//! on real RTX GPUs: every operand `load_operand` builds for representative
//! layers (a head-split SWA layer with its dense MLP, a whole CSA layer with
//! its routed front end, a routed SWA layer) on both ranks of the two-GPU
//! ownership, those layers on one GPU, and MTP stage 0. Digests are pinned to
//! the mount loader as it was before push recipes, so the push recipes
//! (landings, F32 narrowing on the GPU) must reproduce it bit for bit.
//! CUTEAFD_STEP5_GOLDEN_WRITE=1 records instead.
use crate::shared::memory::device::Device;
use anyhow::{Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::plan::{checkpoint::Checkpoint, families::step5::coordinator::RankWeightLayout};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub(crate) type Digests = BTreeMap<String, String>;

pub(crate) fn snapshot() -> Option<std::path::PathBuf> {
    std::env::var_os("CUTEAFD_PUSH_STEP5").map(std::path::PathBuf::from)
}

pub(crate) fn library() -> Result<NativeLibrary> {
    let path = std::env::var("CUTEAFD_NATIVE_LIB").context("set CUTEAFD_NATIVE_LIB")?;
    // SAFETY: the slot's native library, loaded once for this test process.
    unsafe { NativeLibrary::load(path) }
}

/// The golden's layers: dense SWA (0), routed SWA (4), routed CSA (3, 7).
pub(crate) const LAYERS: [usize; 4] = [0, 3, 4, 7];

/// (key prefix, device, layout) of every layout the golden covers.
pub(crate) fn layouts(checkpoint: &Checkpoint) -> Result<Vec<(String, i32, RankWeightLayout)>> {
    let cfg = cuteafd_loader::plan::families::step5::Config::parse(checkpoint)?;
    let ownership = super::golden::split_ownership(&cfg, 16)?;
    let single = cuteafd_loader::plan::families::step5::coordinator::RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?;
    let mut out = Vec::new();
    for rank in 0..2 {
        out.push((format!("split.rank{rank}"), rank as i32, RankWeightLayout::for_execution(checkpoint, rank, &ownership)?));
    }
    out.push(("single".into(), 0, single));
    out.push(("mtp0".into(), 1, RankWeightLayout::for_mtp_stage(checkpoint, 0)?));
    Ok(out)
}

/// Whether the golden covers operand `name` (its layers, the head, MTP stage 0).
pub(crate) fn covered(name: &str) -> bool {
    match name.strip_prefix("model.layers.").and_then(|r| r.split('.').next()).and_then(|l| l.parse::<usize>().ok()) {
        Some(layer) => LAYERS.contains(&layer) || layer >= 92,
        None => !name.contains("embed_tokens"),
    }
}

pub(crate) fn digest(library: &NativeLibrary, buffer: cuteafd_ffi::CuteafdDeviceBuffer) -> Result<String> {
    let mut bytes = vec![0u8; buffer.bytes];
    Device { library, id: buffer.device_id }.run(|| library.copy_d2h(&mut bytes, buffer))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

pub(crate) fn path() -> std::path::PathBuf {
    std::env::var_os("CUTEAFD_STEP5_GOLDEN").map(std::path::PathBuf::from).unwrap_or_else(||
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/step5/coordinator_golden.json"))
}

pub(crate) fn check(actual: &Digests, subset: bool) -> Result<()> {
    if std::env::var_os("CUTEAFD_STEP5_GOLDEN_WRITE").is_some() {
        let out = std::env::var_os("CUTEAFD_STEP5_GOLDEN_OUT").map(std::path::PathBuf::from).unwrap_or_else(path);
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
#[ignore = "requires CUTEAFD_PUSH_STEP5, CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let checkpoint = Checkpoint::coordinator(&snapshot, false, true)?;
    let library = library()?;
    let mut actual = Digests::new();
    for (prefix, device, layout) in layouts(&checkpoint)? {
        let device = Device { library: &library, id: device };
        for operand in layout.operands.iter().filter(|o| covered(&o.name)) {
            let allocation = super::weights::load_operand(device, &checkpoint, &layout, operand)?;
            actual.insert(format!("{prefix}.{}", operand.name), digest(&library, allocation.buffer)?);
        }
    }
    check(&actual, false)
}
