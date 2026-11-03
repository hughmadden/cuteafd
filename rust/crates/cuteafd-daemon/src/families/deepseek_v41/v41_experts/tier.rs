//! The GPU expert tier's resident experts (`cuteafd plan --gpu-expert-tier`):
//! in each backbone layer a fixed subset of routed experts, held as TP2
//! halves, half `r` of every resident expert on GPU `r`. Each GPU's layer is
//! one compact `ExpertWeights` whose slot `i` holds expert `ids[i]`; through
//! the mount the selected halves are read alone, and under a dense recording
//! they record for the coordinator push group instead. Serving wiring (the
//! tier kernel's id-to-slot lookup, the tier wave) comes separately.
// Serving wiring (the tier wave) follows; tests exercise loading and push today.
#![allow(dead_code)]
use super::{ExpertLayer, ExpertWeights};
use crate::shared::memory::device::{Device, DeviceOwner};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::OfficialV41Catalog;
#[cfg(test)]
mod tests;

/// One GPU's resident halves: per backbone layer with resident experts, in
/// layer order, its layer index and the compact half.
pub(crate) type RankHalves<'a> = DeviceOwner<'a, Vec<(usize, ExpertWeights<'a>)>>;

/// Both GPUs' halves of `resident` (per backbone layer, the ascending expert
/// ids the planner's `ExpertTierPlacement::resident` lists), each GPU within
/// its `budgets` (resident bytes plus one expert's load staging).
pub(crate) fn load_halves<'a>(devices: [Device<'a>; 2], catalog: &OfficialV41Catalog, resident: &[Vec<u16>],
    budgets: [usize; 2]) -> Result<[RankHalves<'a>; 2]> {
    ensure!(devices[0].id != devices[1].id && std::ptr::eq(devices[0].library, devices[1].library),
        "tier halves need two GPUs of one library");
    ensure!(resident.len() <= catalog.routed_experts().layers, "tier names more layers than the backbone has");
    let mut halves = [devices[0].own(|| Ok(Vec::new()))?, devices[1].own(|| Ok(Vec::new()))?];
    let mut remaining = budgets;
    for (layer, ids) in resident.iter().enumerate().filter(|(_, ids)| !ids.is_empty()) {
        let ids = ids.iter().map(|&id| usize::from(id)).collect::<Vec<_>>();
        for rank in 0..2 {
            devices[rank].run(|| {
                let half = ExpertWeights::load_resident(devices[rank].library, catalog,
                    ExpertLayer::BackboneTp2 { layer, rank }, &ids, remaining[rank])
                    .with_context(|| format!("tier layer {layer} half {rank}"))?;
                remaining[rank] = remaining[rank].checked_sub(half.budget().resident_bytes)
                    .context("tier halves exceed their GPU budget")?;
                halves[rank].push((layer, half));
                Ok(())
            })?;
        }
    }
    Ok(halves)
}

/// Each GPU's resident bytes for `resident`'s halves (what `load_halves`
/// allocates and a push delivers), without loading anything.
pub(crate) fn resident_bytes(library: &cuteafd_ffi::NativeLibrary, catalog: &OfficialV41Catalog,
    resident: &[Vec<u16>]) -> Result<[usize; 2]> {
    let mut bytes = [0usize; 2];
    for (layer, ids) in resident.iter().enumerate().filter(|(_, ids)| !ids.is_empty()) {
        for (rank, total) in bytes.iter_mut().enumerate() {
            let selection = ExpertLayer::BackboneTp2 { layer, rank };
            let whole = ExpertWeights::plan(library, catalog, selection)?;
            let experts = catalog.routed_experts().experts;
            ensure!(whole.resident_bytes % experts == 0, "tier layer {layer} does not divide by expert");
            *total = total.checked_add(whole.resident_bytes / experts * ids.len()).context("tier bytes overflow")?;
        }
    }
    Ok(bytes)
}
