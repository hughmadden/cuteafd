//! The GPU expert tier's resident experts (`cuteafd plan --gpu-expert-tier`):
//! in each backbone layer a fixed subset of routed experts, held as TP2
//! halves, half `r` of every resident expert on GPU `r`. Each GPU's layer is
//! one compact `ExpertWeights` whose slot `i` holds expert `ids[i]`; through
//! the mount the selected halves are read alone, and under a dense recording
//! they record for the coordinator push group instead.
//!
//! At run time each TP2 FFN (a decode lane's, the prefill slots') runs the
//! tier beside its Spark waves, in place of whole TP2 layers: its routed owner
//! pair is over the tier's halves, each GPU turns the router's expert ids into
//! compact slots ([`TierLane`]: `slot_of`, -1 for the experts it does not
//! hold), runs its half of the resident experts on the unchanged TP2 expert
//! kernel, and the halves meet on the Spark reduce's GPU, whose result joins
//! that reduce as one more plane. The Spark request leaves the same experts to the
//! coordinator (the immutable coordinator resident sets), so every route runs
//! exactly once.
use super::{ExpertLayer, ExpertWeights};
use crate::shared::experts::rtx::pair::{OwnerPolicy, OwnerWeights};
use crate::shared::experts::rtx::native::NativeTp2;
use crate::shared::memory::device::{Allocation, Device, DeviceOwner, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_core::ExpertResidencySet;
use cuteafd_ffi::{CuteafdDeviceBuffer, TierSlotKernel, TierSlotShape};
use cuteafd_loader::OfficialV41Catalog;
use std::rc::Rc;
use std::sync::Arc;
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
/// ([`check_views`], run by `NativeTp4Wave::enable_tier`) proves the GPU
/// copies and the Spark requests' marking agree with it.
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
    let ids = resident.iter().map(|ids| ids.iter().map(|&id| usize::from(id)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    if ids.iter().all(Vec::is_empty) {
        return Ok([devices[0].own(|| Ok(Vec::new()))?, devices[1].own(|| Ok(Vec::new()))?]);
    }
    let mut layouts = Vec::with_capacity(2);
    for rank in 0..2 {
        let (layout, staging) = ExpertWeights::resident_arena_layout(devices[rank], catalog,
            ids.iter().enumerate().filter(|(_, ids)| !ids.is_empty())
                .map(|(layer, ids)| (ExpertLayer::BackboneTp2 { layer, rank }, ids.as_slice())))?;
        crate::shared::experts::weight_layout::admit(&layout, staging, 0, budgets[rank])?;
        layouts.push(layout);
    }
    let arenas = layouts.into_iter().enumerate().map(|(rank, layout)|
        crate::shared::memory::weight_arena::DeviceWeightArena::new(devices[rank], layout,
            cuteafd_core::memory_layout::ArenaTag::experts("experts/native-tier")))
        .collect::<Result<Vec<_>>>()?;
    let mut halves = [devices[0].own(|| Ok(Vec::new()))?, devices[1].own(|| Ok(Vec::new()))?];
    for (layer, ids) in ids.iter().enumerate().filter(|(_, ids)| !ids.is_empty()) {
        for rank in 0..2 {
            devices[rank].run(|| {
                let half = ExpertWeights::load_resident_into(devices[rank].library, catalog,
                    ExpertLayer::BackboneTp2 { layer, rank }, ids, budgets[rank], &arenas[rank])
                    .with_context(|| format!("tier layer {layer} half {rank}"))?;
                halves[rank].push((layer, half));
                Ok(())
            })?;
        }
    }
    Ok(halves)
}

/// Each rank's physical aggregate arena bytes, including alignment, without
/// loading or switching devices. Native sizes come from the current device;
/// the TP2 rank selects the half, not a CUDA ordinal to enter.
pub(crate) fn resident_bytes(library: &cuteafd_ffi::NativeLibrary, catalog: &OfficialV41Catalog,
    resident: &[Vec<u16>]) -> Result<[usize; 2]> {
    let ids = resident.iter().map(|ids| ids.iter().map(|&id| usize::from(id)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    if ids.iter().all(Vec::is_empty) { return Ok([0; 2]); }
    let mut bytes = [0; 2];
    for (rank, total) in bytes.iter_mut().enumerate() {
        *total = ExpertWeights::resident_arena_layout_current(library, rank as i32, catalog,
            ids.iter().enumerate().filter(|(_, ids)| !ids.is_empty())
                .map(|(layer, ids)| (ExpertLayer::BackboneTp2 { layer, rank }, ids.as_slice())))?.0.bytes();
    }
    Ok(bytes)
}

/// Both GPUs' tier halves and the set they hold, as owner-pair weights:
/// shared by every lane that runs the tier, each with its own [`TierLane`].
#[derive(Clone)]
pub(crate) struct TierHalves<'a> {
    pub(crate) residency: Arc<TierResidency>,
    pub(crate) weights: [OwnerWeights<'a>; 2],
}

impl<'a> TierHalves<'a> {
    /// [`load_halves`]' (or a push's) halves of `residency`, admitted for
    /// executions of up to `max_rows` rows at the native owner policy. Each
    /// GPU's halves must hold exactly the set's experts of every resident
    /// layer, in slot order.
    pub(crate) fn new(residency: Arc<TierResidency>, halves: [RankHalves<'a>; 2], max_rows: usize) -> Result<Self> {
        let layers = residency.resident_layers();
        let [left, right] = halves;
        let mut weights = Vec::with_capacity(2);
        for (rank, half) in [left, right].into_iter().enumerate() {
            let device = half.device;
            let OwnerPolicy::Native { selection, output, .. } = device.run(|| OwnerPolicy::native(device.library, false))?
                else { unreachable!("the native owner policy") };
            ensure!(half.len() == layers.len() && half.iter().zip(&layers).all(|((layer, weight), &expected)|
                *layer == expected && weight.resident_ids() == Some(residency.ids(expected))),
                "GPU {} tier halves differ from the resident set", device.id);
            // The halves move from one device-scoped owner into another.
            let owned = device.own(|| Ok(half.into_inner().into_iter().map(|(_, weight)| weight).collect()))?;
            weights.push(OwnerWeights::Native(NativeTp2::resident_weights(device, rank as u8, Rc::new(owned),
                max_rows, selection, output)?));
        }
        let weights: [OwnerWeights<'a>; 2] = weights.try_into().ok().expect("two tier halves");
        Ok(Self { residency, weights })
    }
}

/// One lane's GPU expert tier beside the routed owner pair it runs on (the
/// lane's TP2 FFN's, over the tier's halves): on each GPU the uploaded
/// `slot_of` tables and the slots its rank's kernel reads.
pub(crate) struct TierLane<'a> {
    residency: Arc<TierResidency>,
    tables: [Allocation<'a>; 2],
    slots: [Allocation<'a>; 2],
    kernel: TierSlotKernel<'a>,
    shape: TierSlotShape,
    capacity: u32,
}

impl<'a> TierLane<'a> {
    /// What one lane's tier allocates on each GPU beyond its owner pair: the
    /// `slot_of` tables, and the slots of `capacity` rows of six routes.
    fn allocation_sizes(shape: TierSlotShape, capacity: u32) -> [usize; 2] {
        [shape.bytes(), capacity as usize * 6 * 4]
    }

    /// Device bytes of one lane's tier on each GPU beyond its owner pair.
    pub(crate) fn table_bytes(residency: &TierResidency, capacity: u32) -> usize {
        Self::allocation_sizes(TierSlotShape { layers: residency.layers(), experts: residency.experts() }, capacity)
            .iter().sum()
    }

    /// `residency`'s tables and `capacity` rows of slots on both `devices`.
    pub(crate) fn new(residency: Arc<TierResidency>, devices: [Device<'a>; 2], capacity: u32) -> Result<Self> {
        let shape = TierSlotShape { layers: residency.layers(), experts: residency.experts() };
        shape.validate()?;
        let [table_bytes, slot_bytes] = Self::allocation_sizes(shape, capacity);
        let table: Vec<u8> = residency.slot_tables().iter().flat_map(|slot| slot.to_ne_bytes()).collect();
        let tables = devices.map(|device| -> Result<Allocation<'a>> {
            let allocation = Allocation::new(device, table_bytes, cuteafd_core::memory_layout::Tag::workspace("v41/tier/tables"))?;
            device.run(|| device.library.copy_h2d(allocation.buffer, &table))?;
            Ok(allocation)
        });
        let [left, right] = tables;
        let slots = devices.map(|device| Allocation::new(device, slot_bytes, cuteafd_core::memory_layout::Tag::workspace("v41/tier/routes")));
        let [first, second] = slots;
        Ok(Self { residency, tables: [left?, right?], slots: [first?, second?],
            kernel: devices[0].library.v41_tier_slots()?, shape, capacity })
    }

    /// Whether this lane's tier holds experts of `layer` (its Spark requests
    /// then leave them to the coordinator).
    pub(crate) fn holds(&self, layer: usize) -> bool { !self.residency.ids(layer).is_empty() }

    /// Queues on `stream` (GPU `rank`'s) the slots of `rows` rows of the
    /// router's `ids` (U32 `[rows, 6]` on that GPU) in `layer`, and returns
    /// the buffer holding them (this lane's, valid until the next call).
    ///
    /// # Safety
    /// `ids` is complete in `stream`'s order and unchanged until the slots'
    /// readers (this lane's next [`Self::execute`]) complete; the slots'
    /// previous readers finished or are ordered before `stream`.
    pub(crate) unsafe fn slot_routes(&self, rank: usize, layer: usize, rows: u32, ids: CuteafdDeviceBuffer,
        stream: &Stream<'a>) -> Result<CuteafdDeviceBuffer> {
        ensure!(rank < 2 && (1..=self.capacity).contains(&rows) && self.holds(layer)
            && ids.device_id == stream.device.id && stream.device.id == self.slots[rank].device.id,
            "GPU expert tier slot routes: rank {rank}, {rows} rows, layer {layer}");
        let slots = CuteafdDeviceBuffer { bytes: rows as usize * 6 * 4, ..self.slots[rank].buffer };
        // SAFETY: forwarded from this function's contract; the tables are
        // written only at construction.
        stream.device.run(|| unsafe {
            self.kernel.launch(ids, self.tables[rank].buffer, slots, self.shape, layer, rows as usize * 6, stream.raw)
        })?;
        Ok(slots)
    }

    /// Each GPU's slots for `probe` (router ids) in `layer`, through its
    /// uploaded tables and the slot kernel (the startup check; synchronous).
    pub(crate) fn probe_slots(&self, layer: usize, probe: &[u32]) -> Result<[Vec<i32>; 2]> {
        let run = |rank: usize| probe_slots(self.tables[rank].device, &self.kernel, self.tables[rank].buffer,
            self.shape, layer, probe);
        Ok([run(0)?, run(1)?])
    }
}

/// The router ids of the startup probe: every expert once, in rows of six
/// distinct ids (64 rows for 384 experts).
pub(crate) fn probe_ids(experts: usize) -> Vec<u32> {
    (0..experts.next_multiple_of(6)).map(|route| (route % experts) as u32).collect()
}

/// One GPU's slots for `probe` in `layer` through `tables` (its uploaded
/// `slot_of` tables) and `kernel`, synchronously on a stream of its own.
pub(crate) fn probe_slots<'a>(device: Device<'a>, kernel: &TierSlotKernel<'_>, tables: CuteafdDeviceBuffer,
    shape: TierSlotShape, layer: usize, probe: &[u32]) -> Result<Vec<i32>> {
    let ids = Allocation::new(device, probe.len() * 4, cuteafd_core::memory_layout::Tag::workspace("v41/tier/probe"))?;
    let slots = Allocation::new(device, probe.len() * 4, cuteafd_core::memory_layout::Tag::workspace("v41/tier/probe"))?;
    let stream = Stream::new(device)?;
    device.run(|| device.library.copy_h2d(ids.buffer, &probe.iter().flat_map(|id| id.to_ne_bytes()).collect::<Vec<_>>()))?;
    // SAFETY: every buffer is this function's or the caller's live tables;
    // the stream drains before any of them is read or released.
    let launched = device.run(|| unsafe { kernel.launch(ids.buffer, tables, slots.buffer, shape, layer, probe.len(),
        stream.raw) });
    let drained = stream.drain();
    launched.and(drained)?;
    let mut raw = vec![0u8; probe.len() * 4];
    device.run(|| device.library.copy_d2h(&mut raw, slots.buffer))?;
    Ok(raw.as_chunks::<4>().0.iter().map(|word| i32::from_ne_bytes(*word)).collect())
}

/// The startup check of one layer on one GPU: for every probe route, the
/// Spark request leaves it to the coordinator (`coordinator[r]`, the host
/// marking) exactly when the GPU's slot is valid, and a valid slot names that
/// route's expert in the resident set (so the tier's kernel reads that
/// expert's half and no other).
pub(crate) fn check_views(residency: &TierResidency, layer: usize, gpu: i32, probe: &[u32], coordinator: &[bool],
    slots: &[i32]) -> Result<()> {
    ensure!(probe.len() == coordinator.len() && probe.len() == slots.len(),
        "GPU expert tier check: {} probe routes, {} host owners, {} device slots", probe.len(), coordinator.len(),
        slots.len());
    let ids = residency.ids(layer);
    for ((&expert, &host), &slot) in probe.iter().zip(coordinator).zip(slots) {
        let named = usize::try_from(slot).ok().and_then(|slot| ids.get(slot)).map(|&id| u32::from(id));
        ensure!(host == (slot != NOT_RESIDENT) && (slot == NOT_RESIDENT || named == Some(expert))
            && host == residency.contains(layer, expert as usize),
            "GPU expert tier views disagree on layer {layer} expert {expert}: the Spark request {} it, GPU {gpu}'s \
            slot is {slot}{}", if host { "leaves to the coordinator" } else { "keeps" },
            named.map_or(String::new(), |id| format!(" (expert {id})")));
    }
    Ok(())
}
