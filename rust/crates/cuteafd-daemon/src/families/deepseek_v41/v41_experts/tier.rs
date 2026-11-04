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
use cuteafd_core::ExpertResidencySet;
use cuteafd_loader::OfficialV41Catalog;
#[cfg(test)]
mod tests;

/// `slot_of` entry of an expert the tier does not hold: the expert kernel's
/// route plan drops a route whose id is outside `0..experts` (its inverse is
/// -1 and its plane zero), so the route reads no weights.
pub(crate) const NOT_RESIDENT: i32 = -1;

/// The GPU expert tier's resident set, fixed for the server's lifetime (the
/// planner's `ExpertTierPlacement::resident`): per backbone layer, the
/// ascending expert ids the coordinator GPUs hold, slot `i` holding `ids[i]`.
///
/// It is the one source of both views of the set: the host's per-layer masks,
/// which mark the routes a Spark request leaves to the coordinator, and the
/// GPUs' `slot_of` tables, which turn the router's expert ids into compact
/// slots (or [`NOT_RESIDENT`]) for the tier's expert kernel. The startup check
/// ([`TierTables::verify`]) proves the GPU copies agree with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TierResidency {
    experts: usize,
    resident: Vec<Vec<u16>>,
    masks: Vec<ExpertResidencySet>,
}

impl TierResidency {
    /// `resident[layer]` for every backbone layer (empty: none resident),
    /// each ascending and distinct below `experts`.
    pub(crate) fn new(experts: usize, resident: Vec<Vec<u16>>) -> Result<Self> {
        ensure!(experts > 0 && experts <= i32::MAX as usize, "invalid routed expert count {experts}");
        let masks = resident.iter().enumerate().map(|(layer, ids)| {
            ensure!(ids.windows(2).all(|pair| pair[0] < pair[1]),
                "tier layer {layer}: resident experts must be ascending and distinct");
            ExpertResidencySet::from_ids(experts, ids.iter().map(|&id| usize::from(id)))
                .map_err(|error| anyhow::anyhow!("tier layer {layer}: {error}"))
        }).collect::<Result<Vec<_>>>()?;
        Ok(Self { experts, resident, masks })
    }

    /// Routed experts per layer (the `slot_of` tables' width).
    pub(crate) fn experts(&self) -> usize { self.experts }
    /// Backbone layers the set describes.
    pub(crate) fn layers(&self) -> usize { self.resident.len() }
    /// Layer `layer`'s resident ids, ascending (empty past the described layers).
    pub(crate) fn ids(&self, layer: usize) -> &[u16] {
        self.resident.get(layer).map_or(&[], Vec::as_slice)
    }
    /// Every layer's resident ids, in layer order (`tier::load_halves`' input).
    pub(crate) fn resident(&self) -> &[Vec<u16>] { &self.resident }
    /// Backbone layers holding at least one resident expert, ascending.
    pub(crate) fn resident_layers(&self) -> Vec<usize> {
        self.resident.iter().enumerate().filter(|(_, ids)| !ids.is_empty()).map(|(layer, _)| layer).collect()
    }
    /// Resident experts over all layers.
    pub(crate) fn total(&self) -> usize { self.resident.iter().map(Vec::len).sum() }
    /// Layer `layer`'s host mask, or `None` past the described layers.
    pub(crate) fn mask(&self, layer: usize) -> Option<&ExpertResidencySet> { self.masks.get(layer) }
    /// Whether the coordinator holds `expert` of `layer`.
    pub(crate) fn contains(&self, layer: usize, expert: usize) -> bool {
        self.masks.get(layer).is_some_and(|mask| mask.contains(expert))
    }
    /// Layer `layer`'s `slot_of` table: the compact slot of each resident
    /// expert, [`NOT_RESIDENT`] for the rest.
    pub(crate) fn slot_of(&self, layer: usize) -> Vec<i32> {
        let mut table = vec![NOT_RESIDENT; self.experts];
        for (slot, &id) in self.ids(layer).iter().enumerate() {
            table[usize::from(id)] = slot as i32;
        }
        table
    }
    /// Every layer's `slot_of` table, row-major `[layers, experts]`: what each
    /// GPU uploads once.
    pub(crate) fn slot_tables(&self) -> Vec<i32> {
        (0..self.layers()).flat_map(|layer| self.slot_of(layer)).collect()
    }
}

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
