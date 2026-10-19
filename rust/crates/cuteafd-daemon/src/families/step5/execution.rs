//! Shared immutable weights with private native execution lanes.
use super::{cache::CacheRank, csa::CsaLayer, ffn::{DenseFfn, DensePlan},
    rope::TargetTables, routed::{RoutedFfn, RoutedPlan}, swa::SwaLayer,
    transaction::{Staging, Step}};
use crate::shared::{experts::{rtx::pair::{self, SubmissionScope}, staging::RouteStaging},
    memory::{HostAllocation, device::{Allocation, Device, DeviceOwner}},
    peer_split::{PeerExchange, RankDevice, attention::{self, AttentionStateOwners,
        HeadKernels, HeadMerge, HeadPrecision, HeadProjection, HeadSequence,
        LaneOrder, LayerKernels, PipelineKernels}, hop::HopLink}, spark_intake::SparkLane};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, programs::Programs};
use cuteafd_hostcache::copy::CopyEngine;
use cuteafd_loader::{placement::{Hop, HopPoint, LayerMode}, plan::{checkpoint::Checkpoint,
    families::step5::{coordinator::{CoordinatorContract, RankWeightLayout}, ownership::ExecutionOwnership}}};
use cuteafd_transport::ExpertProtocolV2Request;
use std::{cell::{Cell, RefCell}, ffi::c_void, rc::Rc, time::Duration};

const H: usize = 4096;

/// Exact native allocation bytes, excluding caller-owned cache/transport/code reserves.
/// The final head contributes weights once and a private handle workspace per lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RankAdmission {
    immutable: u64,
    workspace: u64,
}
impl RankAdmission {
    fn bytes(self, lanes: usize) -> Result<u64> {
        ensure!(matches!(lanes, 1 | 2), "Step-5 native lane count must be one or two");
        self.workspace.checked_mul(lanes as u64).and_then(|n| n.checked_add(self.immutable))
            .context("Step-5 native lane admission overflow")
    }

    pub(super) fn check(self, lanes: usize, available: u64, reservations: &[u64]) -> Result<u64> {
        let bytes = self.bytes(lanes)?;
        let total = reservations.iter().try_fold(bytes, |n, &v| n.checked_add(v))
            .context("Step-5 native lane reservation overflow")?;
        ensure!(total <= available,
            "Step-5 native weights and {lanes} lane workspaces need {total} bytes, available {available}");
        Ok(bytes)
    }

    fn with_head(self, head: &super::head::HeadPlan) -> Result<Self> {
        Ok(Self {
            immutable: self.immutable.checked_add(head.immutable_bytes())
                .context("Step-5 native head weight overflow")?,
            workspace: self.workspace.checked_add(head.workspace_bytes())
                .context("Step-5 native head workspace overflow")?,
        })
    }
}

enum Attention<'a> { Sliding(SwaLayer<'a>), Sparse(CsaLayer<'a>) }
enum Ffn<'a> { Dense(DenseFfn<'a>), Routed(RoutedFfn<'a>) }
struct Layer<'a> { attention: Attention<'a>, ffn: Ffn<'a> }
impl<'a> Layer<'a> {
    fn new_lane(&self) -> Result<Self> {
        let attention = match &self.attention {
            Attention::Sliding(owner) => Attention::Sliding(owner.new_lane()?),
            Attention::Sparse(owner) => Attention::Sparse(owner.new_lane()?),
        };
        let ffn = match &self.ffn {
            Ffn::Dense(owner) => Ffn::Dense(owner.new_lane()?),
            Ffn::Routed(owner) => Ffn::Routed(owner.new_lane()?),
        };
        Ok(Self { attention, ffn })
    }

    fn input(&self) -> CuteafdDeviceBuffer {
        match &self.attention { Attention::Sliding(l) => l.input(), Attention::Sparse(l) => l.input() }
    }
    fn post(&self) -> CuteafdDeviceBuffer {
        match &self.attention { Attention::Sliding(l) => l.attention_output(), Attention::Sparse(l) => l.attention_output() }
    }
    fn output(&self) -> CuteafdDeviceBuffer {
        match &self.ffn { Ffn::Dense(l) => l.output(), Ffn::Routed(l) => l.output() }
    }
    fn ffn_input(&self) -> CuteafdDeviceBuffer {
        match &self.ffn { Ffn::Dense(l) => l.input(), Ffn::Routed(l) => l.input() }
    }
}

/// A private lane retains component Rc owners; it never reloads their operands.
struct NativeLane<'a> {
    ranks: [Vec<Option<Layer<'a>>>; 2],
    head: Option<super::head::TargetHeadOwner<'a>>,
}
impl<'a> NativeLane<'a> {
    fn new_lane(&self) -> Result<Self> {
        let mut ranks = [Vec::new(), Vec::new()];
        for (destination, source) in ranks.iter_mut().zip(&self.ranks) {
            *destination = source.iter().map(|layer| layer.as_ref().map(Layer::new_lane).transpose())
                .collect::<Result<_>>()?;
        }
        let head = self.head.as_ref().map(super::head::TargetHeadOwner::new_lane).transpose()?;
        Ok(Self { ranks, head })
    }
}

/// Native rank allocation is atomic with respect to admission: every rank is
/// checked before the first load, and later lanes allocate only private workspace.
struct NativeLanes<'a> {
    lanes: Vec<NativeLane<'a>>,
}
impl<'a> NativeLanes<'a> {
    fn load(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: &ExecutionOwnership, devices: [Device<'a>; 2], rows: usize,
        count: usize, available: [u64; 2], reservations: [u64; 2]) -> Result<Self> {
        ensure!(ownership.ranks() == 2 && matches!(count, 1 | 92), "Step-5 native rank/layer inventory");
        let lanes = usize::try_from(ownership.execution().spec().lanes)?;
        let head_rank = ownership.execution().spec().head_gpu as usize;
        ensure!(head_rank < 2, "Step-5 final head rank");
        let head_plan = (count == 92).then(||
            super::head::HeadPlan::for_execution(programs, checkpoint, rows, ownership)).transpose()?;
        let mut admission = [RankAdmission { immutable: 0, workspace: 0 }; 2];
        for (rank, plan) in admission.iter_mut().enumerate() {
            *plan = Execution::rank_admission(programs, checkpoint, ownership, rank, rows, count)?;
            if rank == head_rank {
                if let Some(head) = &head_plan { *plan = plan.with_head(head)?; }
            }
        }
        admit_native_lanes(admission, lanes, available, reservations, || {
            let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
            let mut ranks: [Vec<Option<Layer<'a>>>; 2] = [Vec::new(), Vec::new()];
            for rank in 0..2 {
                let device = devices[rank];
                let rope = Rc::new(TargetTables::load(device, programs, &contract, "step5_", available[rank], &[])?);
                for layer in 0..count {
                    if ownership.layer(layer, rank)?.query_heads == 0 { ranks[rank].push(None); continue; }
                    let attention = if contract.layers[layer].sliding.is_some() {
                        Attention::Sliding(SwaLayer::for_execution(device, programs, checkpoint, layer,
                            rows, Some(rope.clone()), available[rank], &[], rank, ownership)?)
                    } else {
                        Attention::Sparse(CsaLayer::for_execution(device, programs, checkpoint, layer,
                            rows, Some(rope.clone()), available[rank], &[], rank, ownership)?)
                    };
                    let ffn = if contract.layers[layer].dense_activation.is_some() {
                        let plan = DensePlan::for_execution(programs, checkpoint, layer, rows, u64::MAX, &[], rank, ownership)?;
                        Ffn::Dense(DenseFfn::load_plan(device, programs, checkpoint, plan)?)
                    } else {
                        let plan = RoutedPlan::for_execution(programs, checkpoint, layer, rows, u64::MAX, &[], rank, ownership)?;
                        Ffn::Routed(RoutedFfn::load(device, programs, checkpoint, layer, plan)?)
                    };
                    ranks[rank].push(Some(Layer { attention, ffn }));
                }
            }
            let head = head_plan.map(|plan| super::head::TargetHeadOwner::load(devices[head_rank],
                programs, checkpoint, plan, available[head_rank], &[])).transpose()?;
            let first = NativeLane { ranks, head };
            let mut owners = Vec::with_capacity(lanes);
            for _ in 1..lanes { owners.push(first.new_lane()?); }
            owners.insert(0, first);
            Ok(Self { lanes: owners })
        })
    }
}

fn admit_native_lanes<T>(admission: [RankAdmission; 2], lanes: usize,
    available: [u64; 2], reservations: [u64; 2], allocate: impl FnOnce() -> Result<T>) -> Result<T> {
    for rank in 0..2 { admission[rank].check(lanes, available[rank], &[reservations[rank]])?; }
    allocate()
}

pub(super) struct Execution<'e, 'a> {
    lanes: Vec<LaneExecution<'e, 'a>>,
}

struct LaneExecution<'e, 'a> {
    id: usize,
    ownership: ExecutionOwnership,
    native: NativeLane<'a>,
    caches: [&'e RefCell<CacheRank<'a>>; 2],
    stages: [&'e Staging<'a>; 2],
    peer: PeerExchange<'a>,
    hops: HopLink<'a>,
    sequence: RefCell<HeadSequence>,
    lanes: [RefCell<Option<DeviceOwner<'a, SparkLane<'a>>>>; 2],
    pinned: [RefCell<HostAllocation<'a>>; 2],
    rows: usize,
    inventory: Vec<Inventory>,
    requests: Cell<u64>,
}

struct Inventory { weights: Vec<u8>, kv: Vec<u8>, indexer: Option<Vec<u8>>, compressor: Option<Vec<u8>> }

impl<'e, 'a> Execution<'e, 'a> {
    pub fn required_bytes(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: &ExecutionOwnership, rank: usize, rows: usize) -> Result<u64> {
        Self::required_bytes_count(programs, checkpoint, ownership, rank, rows, 92)
    }

    pub(super) fn required_bytes_count(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: &ExecutionOwnership, rank: usize, rows: usize, count: usize) -> Result<u64> {
        ensure!(matches!(count, 1 | 92), "Step-5 diagnostic layer count");
        ensure!(ownership.ranks() == 2, "Step-5 shared execution requires two ranks");
        Self::rank_admission(programs, checkpoint, ownership, rank, rows, count)?
            .bytes(usize::try_from(ownership.execution().spec().lanes)?)
    }

    /// Includes the vocabulary head on its owning rank. Call for every rank before
    /// allocating any lane; cache, staging, input, transport and code are reserves.
    pub(super) fn serving_admission(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: &ExecutionOwnership, rank: usize, rows: usize) -> Result<RankAdmission> {
        let admission = Self::rank_admission(programs, checkpoint, ownership, rank, rows, 92)?;
        if rank == ownership.execution().spec().head_gpu as usize {
            admission.with_head(&super::head::HeadPlan::for_execution(programs, checkpoint, rows, ownership)?)
        } else { Ok(admission) }
    }

    fn rank_admission(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: &ExecutionOwnership, rank: usize, rows: usize, count: usize) -> Result<RankAdmission> {
        ensure!(matches!(count, 1 | 92) && rank < ownership.ranks(), "Step-5 native rank/layer admission");
        ensure!(rows as u64 <= ownership.execution().spec().rows, "Step-5 native row admission");
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let layout = RankWeightLayout::for_execution(checkpoint, rank, ownership)?;
        let immutable = layout.operands.iter().filter(|w| (0..count).any(|layer| w.name.starts_with(&format!("model.layers.{layer}."))))
            .try_fold(contract.rope_tables.allocation_bytes, |n, w| n.checked_add(w.allocation_bytes))
            .context("Step-5 native rank weight overflow")?;
        let mut workspace = 0u64;
        for layer in 0..count {
            if ownership.layer(layer, rank)?.query_heads == 0 { continue; }
            let attention = if contract.layers[layer].sliding.is_some() {
                SwaLayer::workspace_for_heads(rows, ownership.layer(layer, rank)?.query_heads)? }
                else { CsaLayer::workspace_bytes(rows)? };
            let ffn = if contract.layers[layer].dense_activation.is_some() {
                DensePlan::for_execution(programs, checkpoint, layer, rows, u64::MAX, &[], rank, ownership)?.workspace_bytes()
            } else {
                RoutedPlan::for_execution(programs, checkpoint, layer, rows, u64::MAX, &[], rank, ownership)?.workspace_bytes()
            };
            workspace = workspace.checked_add(attention).and_then(|n| n.checked_add(ffn))
                .context("Step-5 native rank workspace overflow")?;
        }
        Ok(RankAdmission { immutable, workspace })
    }

    pub fn load(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: ExecutionOwnership, caches: [&'e RefCell<CacheRank<'a>>; 2],
        stages: [&'e Staging<'a>; 2], rows: usize, available: [u64; 2],
        reservations: [u64; 2], lanes: [Option<DeviceOwner<'a, SparkLane<'a>>>; 2]) -> Result<Self> {
        Self::load_count(programs, checkpoint, ownership, caches, stages, rows, available,
            reservations, lanes, 92)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_count(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: ExecutionOwnership, caches: [&'e RefCell<CacheRank<'a>>; 2],
        stages: [&'e Staging<'a>; 2], rows: usize, available: [u64; 2],
        reservations: [u64; 2], lanes: [Option<DeviceOwner<'a, SparkLane<'a>>>; 2],
        count: usize) -> Result<Self> {
        Self::load_lanes_count(programs, checkpoint, ownership, caches, vec![stages], rows,
            available, reservations, vec![lanes], count)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_lanes_count(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: ExecutionOwnership, caches: [&'e RefCell<CacheRank<'a>>; 2],
        stages: Vec<[&'e Staging<'a>; 2]>, rows: usize, available: [u64; 2],
        reservations: [u64; 2], transports: Vec<[Option<DeviceOwner<'a, SparkLane<'a>>>; 2]>,
        count: usize) -> Result<Self> {
        ensure!(matches!(count, 1 | 92), "Step-5 execution count must be layer-zero diagnostic or full target");
        let lane_count = usize::try_from(ownership.execution().spec().lanes)?;
        ensure!(ownership.ranks() == 2 && matches!(lane_count, 1 | 2)
            && stages.len() == lane_count && transports.len() == lane_count,
            "Step-5 transport/cache lane inventory");
        for rank in 0..2 {
            ensure!(rows as u64 <= ownership.execution().spec().rows,
                "Step-5 live rows exceed execution plan");
            let cache = caches[rank].borrow();
            ensure!(cache.rank() == rank && cache.layout().ranks.len() == 2,
                "Step-5 cache rank inventory");
            for (lane, stages) in stages.iter().enumerate() {
                stages[rank].validate_cache(&cache)?;
                ensure!(transports[lane][rank].is_some() == (0..count).any(|l|
                    ownership.layer(l, rank).is_ok_and(|local| local.routed_wave)),
                    "Step-5 routed-wave transport inventory");
            }
            if lane_count == 2 {
                ensure!(!Rc::ptr_eq(&stages[0][rank].stream, &stages[1][rank].stream),
                    "Step-5 execution lanes share a staging stream");
            }
        }
        let devices = [caches[0].borrow().stream.device, caches[1].borrow().stream.device];
        ensure!(devices[0].id != devices[1].id && std::ptr::eq(devices[0].library, devices[1].library),
            "Step-5 rank devices/library differ");
        let owners = NativeLanes::load(programs, checkpoint, &ownership, devices,
            rows, count, available, reservations)?;
        let lanes = owners.lanes.into_iter().zip(stages).zip(transports).enumerate()
            .map(|(id, ((native, stages), transports))| LaneExecution::bind(checkpoint,
                ownership.clone(), native, caches, stages, rows, transports, count, id))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { lanes })
    }
}

impl<'e, 'a> LaneExecution<'e, 'a> {
    #[allow(clippy::too_many_arguments)]
    fn bind(checkpoint: &'a Checkpoint, ownership: ExecutionOwnership, native: NativeLane<'a>,
        caches: [&'e RefCell<CacheRank<'a>>; 2], stages: [&'e Staging<'a>; 2], rows: usize,
        lanes: [Option<DeviceOwner<'a, SparkLane<'a>>>; 2], count: usize, id: usize) -> Result<Self> {
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let devices = [caches[0].borrow().stream.device, caches[1].borrow().stream.device];
        let native_ranks = std::array::from_fn(|rank| RankDevice { device: devices[rank].id,
            stream: stages[rank].stream.raw });
        let peer = devices[0].run(|| PeerExchange::new_abortable(devices[0].library, native_ranks, 8, rows * H * 4))?;
        let mut hops = devices[0].run(|| HopLink::new_private_lane(devices[0].library, native_ranks,
            ownership.execution().hops(), ownership.execution().spec(), id))?;
        hops.attach_abort([peer.abort_word(0)?.context("Step-5 lead abort")?,
            peer.abort_word(1)?.context("Step-5 peer abort")?])?;
        hops.validate_binding(ownership.execution(), &peer)?;
        let ranks = &native.ranks;
        let bytes = RouteStaging::new(rows, 8, H * 2)?.bytes;
        let pinned = [RefCell::new(HostAllocation::new(devices[0].library, bytes)?),
            RefCell::new(HostAllocation::new(devices[1].library, bytes)?)];
        let mut inventory = Vec::new();
        for layer in 0..count {
            let mut state = Inventory { weights: Vec::new(), kv: Vec::new(),
                indexer: contract.layers[layer].sliding.is_none().then(Vec::new),
                compressor: contract.layers[layer].sliding.is_none().then(Vec::new) };
            for rank in 0..2 {
                if let Some(local) = &ranks[rank][layer] {
                    match &local.attention {
                        Attention::Sliding(owner) => owner.validate_live_weights(devices[rank])?,
                        Attention::Sparse(owner) => owner.validate_live_weights(devices[rank])?,
                    }
                    match &local.ffn {
                        Ffn::Dense(owner) => owner.validate_live_weights(devices[rank])?,
                        Ffn::Routed(owner) => owner.validate_live_weights(devices[rank])?,
                    }
                    state.weights.push(rank as u8);
                }
                let cache = caches[rank].borrow();
                let layout = &cache.layout().ranks[rank];
                if let Some(ring) = layout.sliding.iter().find(|ring| ring.layer == layer && !ring.mtp) {
                    let key = cache.request_component(ring.key, 0)?;
                    let value = cache.request_component(ring.value, 0)?;
                    ensure!(!key.ptr.is_null() && !value.ptr.is_null() && key.bytes > 0 && value.bytes > 0,
                        "Step-5 live SWA state missing");
                    state.kv.push(rank as u8);
                }
                if let Some(region) = layout.regions.iter().find(|region| region.layer == layer) {
                    let (key, value) = cache.csa_planes(layer)?;
                    ensure!(!key.ptr.is_null() && !value.ptr.is_null() && key.bytes > 0 && value.bytes > 0,
                        "Step-5 live CSA pages missing");
                    state.kv.push(rank as u8);
                    let summary = cache.request_component(region.summaries.context("Step-5 CSA summary owner")?, 0)?;
                    ensure!(!summary.ptr.is_null() && summary.bytes > 0, "Step-5 live indexer state missing");
                    state.indexer.as_mut().context("Step-5 unexpected indexer")?.push(rank as u8);
                    ensure!(stages[rank].owns_compressor(&cache, layer), "Step-5 live compressor missing");
                    state.compressor.as_mut().context("Step-5 unexpected compressor")?.push(rank as u8);
                }
            }
            for owners in [Some(state.weights.as_slice()), Some(state.kv.as_slice()),
                state.indexer.as_deref(), state.compressor.as_deref()].into_iter().flatten() {
                ownership.execution().validate_state_owners(layer, owners)?;
            }
            inventory.push(state);
        }
        let execution = Self { id, ownership, native, caches, stages, peer, hops,
            sequence: RefCell::new(HeadSequence::new(id)?), lanes: lanes.map(RefCell::new),
            pinned, rows, inventory, requests: Cell::new(0) };
        if count == 92 { execution.validate_replications()?; }
        Ok(execution)
    }

    fn layer(&self, rank: usize, layer: usize) -> Result<&Layer<'a>> {
        self.native.ranks[rank].get(layer).and_then(Option::as_ref).context("Step-5 missing live layer owner")
    }
    pub fn output(&self, layer: usize) -> Result<(Device<'a>, CuteafdDeviceBuffer)> {
        let rank = match self.ownership.execution().mode(layer)? {
            LayerMode::Whole { gpu, .. } => gpu as usize, LayerMode::HeadSplit => 0,
            LayerMode::ContextSplit => anyhow::bail!("Step-5 context output unsupported"),
        };
        Ok((self.caches[rank].borrow().stream.device, self.layer(rank, layer)?.output()))
    }
    pub(super) fn head(&self) -> Result<&super::head::TargetHeadOwner<'a>> {
        self.native.head.as_ref().context("Step-5 final head not loaded for this diagnostic")
    }
    pub(super) fn diagnostic_layer0(&self, rank: usize) -> Result<(&SwaLayer<'a>, &DenseFfn<'a>)> {
        let layer = self.layer(rank, 0)?;
        let (Attention::Sliding(attention), Ffn::Dense(ffn)) = (&layer.attention, &layer.ffn)
            else { anyhow::bail!("Step-5 layer zero diagnostic ownership"); };
        Ok((attention, ffn))
    }

    pub(super) fn enqueue_layer0<'s, E: CopyEngine>(&'s self, step: &mut Step<'s, 'a, E>,
        input: &'s Allocation<'a>, directory: &'s std::path::Path) -> Result<()> {
        ensure!(self.native.ranks.iter().all(|layers| layers.len() == 1), "layer-zero diagnostic must load only layer zero");
        step.validate_binding(&self.caches, &self.stages, self.rows)?;
        pair::submit_scoped(|| self.peer.require_live(), || self.shutdown(), |scope| {
            step.in_submission(scope, |step| {
                step.compose(0, input, |_, _| Ok(()))?;
                attention::replicate_terminal(&self.peer, input.buffer, self.layer(1, 0)?.input(), self.rows * H * 4)?;
                let mut binding = Bindings { execution: self, step, scope,
                    current: [None, Some(self.layer(1, 0)?.input())], input: input.buffer,
                    request: super::golden::request, layer: 0, diagnostic: Some(directory) };
                binding.heads(0)?;
                binding.ffn(0, None)?;
                for stage in &self.stages { stage.stream.drain()?; }
                Ok(())
            })
        })
    }

    fn validate_replications(&self) -> Result<()> {
        let mut copies = self.ownership.routed_replications(self.id)?;
        for (layer, copy) in copies.iter_mut().enumerate() {
            if let Some(copy) = copy {
                let Ffn::Routed(source) = &self.layer(copy.source as usize, layer)?.ffn else { anyhow::bail!("Step-5 routed source executor"); };
                let Ffn::Routed(destination) = &self.layer(1 - copy.source as usize, layer)?.ffn else { anyhow::bail!("Step-5 routed receive executor"); };
                let (source, destination) = (source.routed_output(), destination.routed_output());
                ensure!(!source.ptr.is_null() && !destination.ptr.is_null()
                    && source.bytes >= self.rows * H * 2 && destination.bytes >= self.rows * H * 2
                    && source.device_id == self.peer.device(0) && destination.device_id == self.peer.device(1),
                    "Step-5 routed replication live storage/device");
                copy.storage = destination.ptr as usize;
            }
        }
        self.ownership.execution().validate_routed_replications(&copies)?;
        Ok(())
    }
    fn shutdown(&self) -> Result<()> { shutdown_lanes(&[self]) }

    fn quiesce(&self) -> Result<()> {
        collect_lane_results(self.lanes.iter().filter_map(|lane| lane.borrow_mut().as_mut().map(|lane| {
            let device = lane.device;
            device.run(|| lane.get_mut().terminal_quiesce())
        })))
    }

    fn drain_copies(&self) -> Result<()> {
        collect_lane_results(self.lanes.iter().filter_map(|lane| lane.borrow().as_ref().map(|lane|
            lane.device.run(|| lane.intake.terminal_drain_copies()))))?;
        collect_lane_results(self.lanes.iter().filter_map(|lane| lane.borrow_mut().as_mut().map(|lane| {
            let device = lane.device;
            device.run(|| lane.get_mut().terminal_release())
        })))
    }

}
impl Drop for LaneExecution<'_, '_> {
    fn drop(&mut self) { if let Err(error) = self.shutdown() {
        tracing::error!(%error, "Step-5 native pipeline teardown retained queued owners");
    }}
}

// Attempt every owner in a phase, even if an earlier owner fails. In particular,
// every abort publication precedes the first compute drain across all lanes.
fn collect_lane_results(results: impl IntoIterator<Item = Result<()>>) -> Result<()> {
    let failures: Vec<_> = results.into_iter().filter_map(Result::err)
        .map(|error| format!("{error:#}")).collect();
    ensure!(failures.is_empty(), "Step-5 lane owners: {}", failures.join("; "));
    Ok(())
}

fn shutdown_lanes(lanes: &[&LaneExecution<'_, '_>]) -> Result<()> {
    use crate::shared::peer_split::TerminalState;
    if lanes.iter().all(|lane| lane.peer.terminal_state() == TerminalState::Drained) { return Ok(()); }
    let result = pair::teardown(&mut pair::Teardown {
        abort: || collect_lane_results(lanes.iter().map(|lane| lane.peer.publish_abort())),
        ranks: || collect_lane_results(lanes.iter().map(|lane| lane.peer.drain_compute())),
        links: || collect_lane_results(lanes.iter().map(|lane| lane.quiesce())),
        copies: || collect_lane_results(lanes.iter().map(|lane| lane.drain_copies())),
        graphs: || Ok(()),
        finish: |drained| {
            for lane in lanes {
                lane.peer.finish_terminal(drained);
                if !drained { lane.caches[0].borrow().stream.device.library.quarantine_module_after_failed_drain(); }
            }
        },
    });
    crate::shared::decode_graph::fatal_drain(result, "Step-5 pipeline terminal owners");
    Ok(())
}

impl<'e, 'a> Execution<'e, 'a> {
    fn only_lane(&self) -> Result<&LaneExecution<'e, 'a>> {
        ensure!(self.lanes.len() == 1, "Step-5 singleton diagnostic requires one lane");
        Ok(&self.lanes[0])
    }
    pub fn output(&self, layer: usize) -> Result<(Device<'a>, CuteafdDeviceBuffer)> {
        self.only_lane()?.output(layer)
    }
    pub(super) fn head(&self) -> Result<&super::head::TargetHeadOwner<'a>> { self.only_lane()?.head() }
    pub(super) fn lane_output(&self, lane: usize, layer: usize) -> Result<(Device<'a>, CuteafdDeviceBuffer)> {
        self.lanes.get(lane).context("Step-5 output lane outside admission")?.output(layer)
    }
    pub(super) fn lane_head(&self, lane: usize) -> Result<&super::head::TargetHeadOwner<'a>> {
        self.lanes.get(lane).context("Step-5 head lane outside admission")?.head()
    }
    pub(super) fn diagnostic_layer0(&self, rank: usize) -> Result<(&SwaLayer<'a>, &DenseFfn<'a>)> {
        self.only_lane()?.diagnostic_layer0(rank)
    }
    pub(super) fn enqueue_layer0<'s, E: CopyEngine>(&'s self, step: &mut Step<'s, 'a, E>,
        input: &'s Allocation<'a>, directory: &'s std::path::Path) -> Result<()> {
        self.only_lane()?.enqueue_layer0(step, input, directory)
    }
    pub fn enqueue<'s, E: CopyEngine>(&'s self, step: &mut Step<'s, 'a, E>,
        input: &'s Allocation<'a>, request: Request) -> Result<()> {
        self.enqueue_lanes(&mut [step], &[input], request)
    }
    pub(super) fn enqueue_lanes<'s, E: CopyEngine>(&'s self, steps: &mut [&mut Step<'s, 'a, E>],
        inputs: &[&'s Allocation<'a>], request: Request) -> Result<()> {
        ensure!(steps.len() == self.lanes.len() && inputs.len() == steps.len(),
            "Step-5 submission lane inventory");
        for ((lane, step), input) in self.lanes.iter().zip(steps.iter()).zip(inputs) {
            step.validate_binding(&lane.caches, &lane.stages, lane.rows)?;
            lane.validate_replications()?;
            ensure!(!input.buffer.ptr.is_null()
                && input.buffer.device_id == lane.peer.device(lane.ownership.execution().spec().entry_gpu as usize)
                && input.buffer.bytes >= lane.rows * H * 4, "Step-5 live input allocation");
        }
        let lanes: Vec<_> = self.lanes.iter().collect();
        pair::submit_scoped(|| collect_lane_results(lanes.iter().map(|lane| lane.peer.require_live())),
            || shutdown_lanes(&lanes), |scope| Step::in_submissions(scope, steps, |steps| {
                let mut bindings = self.lanes.iter().zip(steps.iter_mut()).zip(inputs)
                    .map(|((lane, step), input)| {
                        step.compose(lane.ownership.execution().spec().entry_gpu as usize, *input, |_, _| Ok(()))?;
                        Ok(Bindings { execution: lane, step, scope, current: [None, None],
                            input: input.buffer, request, layer: 0, diagnostic: None })
                    }).collect::<Result<Vec<_>>>()?;
                let exchanges: Vec<_> = self.lanes.iter().map(|lane| (&lane.peer, &lane.hops)).collect();
                let rows: Vec<_> = self.lanes.iter().map(|lane| lane.rows).collect();
                attention::layer_pipeline_lanes(scope, self.lanes[0].ownership.execution(), &exchanges,
                    LaneOrder::Independent, &rows, &mut bindings)?;
                collect_lane_results(self.lanes.iter().flat_map(|lane| lane.stages.iter())
                    .map(|stage| stage.stream.drain()))
            }))
    }
}
impl Drop for Execution<'_, '_> {
    fn drop(&mut self) {
        let lanes: Vec<_> = self.lanes.iter().collect();
        if let Err(error) = shutdown_lanes(&lanes) {
            tracing::error!(%error, "Step-5 aggregate teardown retained queued owners");
        }
    }
}

fn request_identity(lane: usize, ordinal: u64) -> Result<u64> {
    ensure!(lane < 2, "Step-5 request lane outside admission");
    ordinal.checked_mul(2).and_then(|id| id.checked_add(lane as u64 + 1))
        .context("Step-5 request namespace exhausted")
}

type Request = fn(usize, usize, Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>, bytes::Bytes) -> Result<ExpertProtocolV2Request>;
struct Bindings<'s, 'b, 'e, 'a, 'g, E: CopyEngine> {
    execution: &'s LaneExecution<'e, 'a>, step: &'b mut Step<'s, 'a, E>,
    scope: &'g SubmissionScope<'g>, current: [Option<CuteafdDeviceBuffer>; 2],
    input: CuteafdDeviceBuffer, request: Request, layer: usize,
    diagnostic: Option<&'s std::path::Path>,
}
impl<'s, E: CopyEngine> Bindings<'s, '_, '_, '_, '_, E> {
    fn copy(&mut self, rank: usize, destination: CuteafdDeviceBuffer, source: CuteafdDeviceBuffer) -> Result<()> {
        ensure!(destination.device_id == self.execution.peer.device(rank)
            && source.device_id == destination.device_id && !source.ptr.is_null() && !destination.ptr.is_null()
            && source.bytes >= self.execution.rows * H * 4 && destination.bytes >= self.execution.rows * H * 4,
            "Step-5 residual copy live rank extent");
        if source.ptr == destination.ptr { return Ok(()); }
        self.step.compose(rank, self.execution, |_, meta| meta.stream.device.run(|| {
            // SAFETY: the transaction borrows the complete execution owner; admitted disjoint rank planes survive drainage.
            unsafe { meta.stream.device.library.copy_d2d_async(destination, source, meta.rows * H * 4, meta.stream.raw) }
        }))
    }
    fn attention(&mut self, rank: usize, layer: usize) -> Result<()> {
        let owner = self.execution.layer(rank, layer)?;
        let source = self.current[rank].unwrap_or(self.input);
        self.copy(rank, owner.input(), source)?;
        match &owner.attention {
            Attention::Sliding(owner) => owner.enqueue_attention_step(self.step, rank),
            Attention::Sparse(owner) => owner.enqueue_attention_step(self.step, rank),
        }
    }
}
impl<E: CopyEngine> LayerKernels for Bindings<'_, '_, '_, '_, '_, E> {
    fn pack(&mut self, hop: Hop) -> Result<*const c_void> {
        Ok(self.current[hop.from as usize].unwrap_or(self.input).ptr.cast_const())
    }
    fn entry_destination(&mut self, rank: usize) -> Result<*mut c_void> {
        // Entry hops land directly in the first layer's owned input.
        Ok(self.execution.layer(rank, 0)?.input().ptr)
    }
    fn adopt(&mut self, hop: Hop, residual: *mut c_void) -> Result<()> {
        ensure!(matches!(hop.at, HopPoint::Entry | HopPoint::BeforeLayer(_) | HopPoint::Exit),
            "Step-5 whole native FFN never broadcasts after attention");
        self.current[hop.to as usize] = Some(CuteafdDeviceBuffer { ptr: residual,
            bytes: self.execution.rows * H * 4, device_id: self.execution.peer.device(hop.to as usize), flags: 0 });
        Ok(())
    }
    fn heads(&mut self, layer: usize) -> Result<()> {
        self.layer = layer;
        self.attention(0, layer)?;
        self.attention(1, layer)?;
        if let Some(directory) = self.diagnostic {
            // Capture before merge reuses the normalized input plane.
            for rank in 0..2 {
                self.execution.stages[rank].stream.drain()?;
                let (owner, _) = self.execution.diagnostic_layer0(rank)?;
                let device = self.execution.caches[rank].borrow().stream.device;
                super::golden::dump_plane(directory, &format!("rank{rank}-attention_input"), device,
                    owner.diagnostic_norm(), self.execution.rows, true)?;
                super::golden::dump_plane(directory, &format!("rank{rank}-o_partial"), device,
                    owner.projection_partial()?, self.execution.rows, false)?;
            }
        }
        let execution = self.execution;
        attention::head_partials(self.scope, &execution.peer, &mut execution.sequence.borrow_mut(),
            execution.rows * H, HeadPrecision::F32, self)
    }
    fn context(&mut self, _layer: usize, _ordinal: usize) -> Result<()> {
        anyhow::bail!("Step-5 context split has no qualified binding")
    }
    fn owned(&mut self, layer: usize, rank: usize) -> Result<()> {
        self.layer = layer;
        self.attention(rank, layer)
    }
    fn ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<()> {
        let ticket = self.begin_ffn(layer, owner)?;
        self.finish_ffn(layer, ticket)
    }
}

impl<E: CopyEngine> PipelineKernels for Bindings<'_, '_, '_, '_, '_, E> {
    type FfnTicket = Option<usize>;
    fn state_owners(&self, layer: usize) -> Result<AttentionStateOwners<'_>> {
        let state = self.execution.inventory.get(layer).context("Step-5 live state inventory")?;
        Ok(AttentionStateOwners { weights: &state.weights, kv: &state.kv,
            indexer: state.indexer.as_deref(), compressor: state.compressor.as_deref() })
    }
    fn begin_ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<Self::FfnTicket> {
        let execution = self.execution;
        let mut wave = None;
        for rank in 0..2 {
            if owner.is_some_and(|owner| owner != rank) { continue; }
            let local = execution.layer(rank, layer)?;
            self.copy(rank, local.ffn_input(), local.post())?;
            match &local.ffn {
                Ffn::Dense(ffn) => self.step.compose(rank, ffn, |ffn, meta| ffn.enqueue_on(meta.rows, meta.stream))?,
                Ffn::Routed(ffn) => {
                    let dispatch = execution.ownership.layer(layer, rank)?.routed_wave;
                    self.step.compose(rank, ffn, |ffn, meta| {
                        if dispatch { ffn.frontend_on(meta.rows, meta.stream) }
                        else { ffn.normalize_on(meta.rows, meta.stream) }
                    })?;
                    if dispatch {
                        ensure!(wave.is_none(), "Step-5 duplicate routed Spark wave");
                        let layout = RouteStaging::new(execution.rows, 8, H * 2)?;
                        let host = execution.pinned[rank].borrow();
                        self.step.compose(rank, ffn, |ffn, meta| {
                            meta.stream.device.run(|| {
                                // SAFETY: pinned arena belongs to this lane; next write follows wait and consumed-event drainage.
                                unsafe { layout.enqueue(meta.stream.device.library, host.buffer, ffn.wave_inputs(), meta.stream.raw) }
                            })?;
                            // Both head publications were queued before either merge. No future
                            // publication is needed for this stream's staging completion.
                            meta.stream.drain()
                        })?;
                        let routes = layout.routes(host.bytes())?;
                        let wire = bytes::Bytes::copy_from_slice(layout.wire(host.bytes())?);
                        let request = self.request;
                        let rows = execution.rows;
                        let mut lane = execution.lanes[rank].borrow_mut();
                        let lane = lane.as_mut().context("Step-5 routed wave lane missing")?;
                        let device = lane.device;
                        let ordinal = execution.requests.get();
                        let next = ordinal.checked_add(1).context("Step-5 request sequence exhausted")?;
                        let identity = request_identity(execution.id, ordinal)?;
                        execution.requests.set(next);
                        device.run(|| lane.get_mut().submit(execution.rows, Box::new(move || {
                            let mut wave = request(layer, rows, routes, wire)?;
                            wave.header.request_id = identity;
                            Ok(wave)
                        })))?;
                        wave = Some(rank);
                    }
                    self.step.compose(rank, ffn, |ffn, meta| ffn.shared_on(meta.rows, meta.stream))?;
                }
            }
            self.current[rank] = Some(local.output());
        }
        Ok(wave)
    }
    fn finish_ffn(&mut self, layer: usize, ticket: Self::FfnTicket) -> Result<()> {
        let execution = self.execution;
        let Some(rank) = ticket else { return Ok(()); };
        let Ffn::Routed(ffn) = &execution.layer(rank, layer)?.ffn else { anyhow::bail!("Step-5 routed completion owner"); };
        self.step.compose(rank, ffn, |ffn, meta| {
            let mut lane = execution.lanes[rank].borrow_mut();
            let lane = lane.as_mut().context("Step-5 routed completion lane")?;
            let device = lane.device;
            device.run(|| lane.get_mut().wait(meta.rows, Duration::from_secs(120), meta.stream.raw))?;
            meta.stream.device.run(|| {
                // SAFETY: waited intake planes and routed output are disjoint [rows,4096]
                // BF16 owners. Runtime expert geometry is set before library construction.
                unsafe { meta.stream.device.library.v41_compact_reducer()?.reduce_planes(
                    lane.intake.pointers(), lane.world_size() as u32, std::ptr::null(),
                    ffn.routed_output().ptr.cast(), meta.rows as u32, meta.stream.raw) }
            })?;
            meta.stream.device.run(|| lane.intake.consumed(meta.stream.raw))
        })?;
        if execution.ownership.execution().mode(layer)? == LayerMode::HeadSplit {
            let Ffn::Routed(destination) = &execution.layer(1 - rank, layer)?.ffn else { anyhow::bail!("Step-5 routed replica missing"); };
            let binding = execution.ownership.routed_replications(execution.id)?[layer].context("Step-5 routed publication binding")?;
            attention::replicate_result(self.scope, &execution.peer, rank, binding.slot,
                ffn.routed_output(), destination.routed_output(), execution.rows * H * 2)?;
            for at in 0..2 {
                let Ffn::Routed(ffn) = &execution.layer(at, layer)?.ffn else { anyhow::bail!("Step-5 native routed join missing"); };
                self.step.compose(at, ffn, |ffn, meta| ffn.finish_on(meta.rows, meta.stream))?;
            }
        } else {
            self.step.compose(rank, ffn, |ffn, meta| ffn.finish_on(meta.rows, meta.stream))?;
        }
        Ok(())
    }
}

impl<E: CopyEngine> HeadKernels for Bindings<'_, '_, '_, '_, '_, E> {
    fn project(&mut self, rank: usize) -> Result<HeadProjection> {
        // heads() has queued both projections through the cache transaction.
        let layer = self.layer;
        let Attention::Sliding(owner) = &self.execution.layer(rank, layer)?.attention else { anyhow::bail!("Step-5 head split requires SWA"); };
        Ok(HeadProjection { partial: owner.projection_partial()?, rounded: None })
    }
    fn merge(&mut self, rank: usize, values: HeadMerge) -> Result<()> {
        let layer = self.layer;
        let Attention::Sliding(owner) = &self.execution.layer(rank, layer)?.attention else { anyhow::bail!("Step-5 head split requires SWA"); };
        let HeadMerge::F32(values) = values else { anyhow::bail!("Step-5 head merge requires F32"); };
        let capture = self.diagnostic.map(|directory| (directory, rank));
        self.step.compose(rank, owner, |owner, meta| owner.merge_partial_capture_on(meta.rows, values, meta.stream, capture))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::peer_split::{attention::{LayerProtocol, layer_pipeline_with},
        order::{check_submission_faults, host::{HostSchedule, check_host}}};
    use cuteafd_loader::plan::families::step5::Config;
    use std::cell::Cell;

    struct Trace {
        ownership: ExecutionOwnership,
        host: RefCell<HostSchedule>,
        heads: Cell<usize>,
        waves: Cell<usize>,
    }
    impl Trace {
        fn record(&self, from: usize, exchange: &'static str, slot: usize) {
            let mut host = self.host.borrow_mut();
            host.push(from, exchange, slot, "publication");
            host.wait(1 - from, exchange, slot, "receive");
        }
    }
    impl LayerProtocol for Trace {
        type Ticket = (usize, usize);
        fn require_live(&self) -> Result<()> { Ok(()) }
        fn push_entry(&self, from: usize, _: *const c_void, _: *mut c_void, _: usize) -> Result<()> {
            self.host.borrow_mut().push(from, "entry", 0, "entry push"); Ok(())
        }
        fn wait_entry(&self, to: usize) -> Result<()> {
            self.host.borrow_mut().wait(to, "entry", 0, "entry receive"); Ok(())
        }
        fn send(&self, index: usize, _: usize, _: *const c_void, _: usize) -> Result<Self::Ticket> {
            let hop = self.ownership.execution().hops()[index];
            self.host.borrow_mut().push(hop.from as usize, "hop", index, "hop push");
            Ok((hop.to as usize, index))
        }
        fn land(&self, (to, index): Self::Ticket) -> Result<*mut c_void> {
            self.host.borrow_mut().wait(to, "hop", index, "hop receive");
            Ok(std::ptr::null_mut())
        }
    }
    struct Recorder<'a>(&'a Trace);
    impl LayerKernels for Recorder<'_> {
        fn pack(&mut self, _: Hop) -> Result<*const c_void> { Ok(std::ptr::null()) }
        fn entry_destination(&mut self, _: usize) -> Result<*mut c_void> { Ok(std::ptr::null_mut()) }
        fn adopt(&mut self, _: Hop, _: *mut c_void) -> Result<()> { Ok(()) }
        fn heads(&mut self, _: usize) -> Result<()> {
            let ordinal = self.0.heads.get();
            let mut host = self.0.host.borrow_mut();
            // Mirrors head_partials: both projections/publications precede waits.
            for rank in 0..2 { host.push(rank, "heads", ordinal % 2, "head push"); }
            for rank in 0..2 { host.wait(rank, "heads", ordinal % 2, "native head merge"); }
            self.0.heads.set(ordinal + 1); Ok(())
        }
        fn context(&mut self, _: usize, _: usize) -> Result<()> { anyhow::bail!("no context binding") }
        fn owned(&mut self, _: usize, _: usize) -> Result<()> { Ok(()) }
        fn ffn(&mut self, _: usize, _: Option<usize>) -> Result<()> { unreachable!() }
    }
    impl PipelineKernels for Recorder<'_> {
        type FfnTicket = Option<usize>;
        fn state_owners(&self, layer: usize) -> Result<AttentionStateOwners<'_>> {
            let owners: &[u8] = match self.0.ownership.execution().mode(layer)? {
                LayerMode::Whole { gpu: 0, .. } => &[0],
                LayerMode::Whole { gpu: 1, .. } => &[1],
                LayerMode::HeadSplit => &[0, 1],
                _ => unreachable!(),
            };
            let sparse = layer % 4 == 3;
            Ok(AttentionStateOwners { weights: owners, kv: owners,
                indexer: sparse.then_some(owners), compressor: sparse.then_some(owners) })
        }
        fn begin_ffn(&mut self, layer: usize, _: Option<usize>) -> Result<Self::FfnTicket> {
            let mut wave = None;
            for rank in 0..2 {
                if self.0.ownership.layer(layer, rank)?.routed_wave {
                    assert!(wave.is_none(), "duplicate routed dispatch");
                    self.0.host.borrow_mut().sync(rank, "route D2H before Spark dispatch");
                    self.0.waves.set(self.0.waves.get() + 1);
                    wave = Some(rank);
                }
            }
            Ok(wave)
        }
        fn finish_ffn(&mut self, layer: usize, wave: Option<usize>) -> Result<()> {
            if let Some(rank) = wave {
                if let Some(copy) = self.0.ownership.routed_replications(0)?[layer] {
                    assert_eq!(rank, copy.source as usize);
                    self.0.record(rank, "routed", copy.slot);
                }
            }
            Ok(())
        }
    }
    fn config() -> Config {
        Config { norm_eps: 1e-5, hidden: H, vocab: 128896, layers: 92, mtp: 3,
            dense: 13824, experts: 352, intermediate: 1536, shared: 1536, topk: 8,
            heads: 64, groups: 4, dim: 192, window: 512, max_context: 1 << 20,
            full: (0..95).map(|l| l < 92 && l % 4 == 3).collect(),
            moe: (0..95).map(|l| (3..91).contains(&l)).collect(),
            theta: vec![], rotary: vec![], rope_scaling: None, scale_full: false,
            scale_sliding: false, activation: vec![], shared_activation: vec![] }
    }
    #[test]
    fn request_identities_are_disjoint_checked_and_never_zero() -> Result<()> {
        let mut seen = std::collections::BTreeSet::new();
        for ordinal in 0..1024 {
            for lane in 0..2 {
                let id = request_identity(lane, ordinal)?;
                assert_ne!(id, 0);
                assert!(seen.insert(id));
                assert_eq!((id - 1) % 2, lane as u64);
            }
        }
        assert_eq!(request_identity(0, u64::MAX / 2)?, u64::MAX);
        assert!(request_identity(1, u64::MAX / 2).is_err());
        assert!(request_identity(0, u64::MAX / 2 + 1).is_err());
        assert!(request_identity(2, 0).is_err());
        Ok(())
    }

    #[test]
    fn grouped_terminal_phases_attempt_both_owners_and_abort_before_draining() {
        for fault in [None, Some("abort"), Some("ranks"), Some("links"), Some("copies")] {
            for cancelled in [false, true] {
                let events = RefCell::new(Vec::new());
                let phase = |name| collect_lane_results((0..2).map(|lane| {
                    events.borrow_mut().push((name, lane));
                    ensure!(fault != Some(name) || lane != 0, "injected {name} failure");
                    Ok(())
                }));
                let shutdown = || pair::teardown(&mut pair::Teardown {
                    abort: || phase("abort"), ranks: || phase("ranks"),
                    links: || phase("links"), copies: || phase("copies"),
                    graphs: || Ok(()),
                    finish: |drained| events.borrow_mut().push((if drained { "drained" } else { "retained" }, 2)),
                });
                if cancelled {
                    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _: Result<()> = pair::submit_scoped(|| Ok(()), shutdown, |_| {
                            panic!("injected submission cancellation")
                        });
                    })).is_err());
                } else {
                    let result: Result<()> = pair::submit_scoped(|| Ok(()), shutdown, |_| {
                        anyhow::bail!("injected submission failure")
                    });
                    assert!(result.is_err());
                }
                let mut expected = vec![("abort", 0), ("abort", 1)];
                if fault != Some("abort") { expected.extend([("ranks", 0), ("ranks", 1)]); }
                expected.extend([("links", 0), ("links", 1)]);
                if !matches!(fault, Some("abort" | "ranks" | "links")) {
                    expected.extend([("copies", 0), ("copies", 1)]);
                }
                expected.push((if fault.is_none() { "drained" } else { "retained" }, 2));
                assert_eq!(*events.borrow(), expected, "fault={fault:?} cancelled={cancelled}");
            }
        }
    }

    #[test]
    fn one_lane_native_plan_orders_host_syncs_hops_and_routed_replies_across_odd_head_passes() -> Result<()> {
        let trace = Trace { ownership: super::super::golden::split_ownership(&config(), 64)?,
            host: RefCell::new(HostSchedule::default()), heads: Cell::new(0), waves: Cell::new(0) };
        for pass in 0..2 {
            pair::submit_scoped(|| Ok(()), || Ok(()), |scope| {
                layer_pipeline_with(scope, trace.ownership.execution(), &trace,
                    LaneOrder::Independent, &[64], &mut [Recorder(&trace)])
            })?;
            assert_eq!(trace.heads.get(), (pass + 1) * 69);
            assert_eq!(trace.waves.get(), (pass + 1) * 88);
            check_host(&trace.host.borrow()).map_err(anyhow::Error::msg)?;
        }
        check_submission_faults(&trace.host.borrow().streams()).map_err(anyhow::Error::msg)?;
        Ok(())
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    #[test]
    fn immutable_bytes_are_shared_and_every_lane_workspace_is_charged() -> Result<()> {
        let plan = RankAdmission { immutable: 1_000_000, workspace: 123_456 };
        assert_eq!(plan.check(1, 1_123_456, &[])?, 1_123_456);
        assert_eq!(plan.check(2, 1_380_456, &[133_544])?, 1_246_912);
        Ok(())
    }

    #[test]
    fn final_head_weights_and_private_blas_workspace_are_included() -> Result<()> {
        let head = super::super::head::tests::admission_plan(64);
        let base = RankAdmission { immutable: 1000, workspace: 2000 };
        let plan = base.with_head(&head)?;
        assert_eq!(plan.bytes(1)?, base.bytes(1)? + head.bytes());
        assert_eq!(plan.bytes(2)?, base.bytes(2)? + head.bytes() + head.workspace_bytes());
        assert_eq!(plan.bytes(2)? - plan.bytes(1)?, 2000 + head.workspace_bytes());
        assert!(plan.check(2, plan.bytes(2)? - 1, &[]).is_err());
        assert!(RankAdmission { immutable: u64::MAX, workspace: 0 }.with_head(&head).is_err());
        assert!(RankAdmission { immutable: 0, workspace: u64::MAX }.with_head(&head).is_err());
        Ok(())
    }

    #[test]
    fn all_ranks_are_admitted_before_the_allocation_callback() -> Result<()> {
        let admission = [RankAdmission { immutable: 100, workspace: 20 },
            RankAdmission { immutable: 200, workspace: 30 }];
        let calls = std::cell::Cell::new(0);
        let allocate = || { calls.set(calls.get() + 1); Ok(()) };
        // Rank zero fits, but rank one must fail without loading either rank.
        assert!(admit_native_lanes(admission, 2, [140, 259], [0, 0], allocate).is_err());
        assert_eq!(calls.get(), 0);
        assert!(admit_native_lanes(admission, 2, [140, 260], [0, 1], allocate).is_err());
        assert_eq!(calls.get(), 0);
        assert!(admit_native_lanes(admission, 3, [u64::MAX; 2], [0; 2], allocate).is_err());
        assert_eq!(calls.get(), 0);
        admit_native_lanes(admission, 2, [147, 269], [7, 9], allocate)?;
        assert_eq!(calls.get(), 1);
        Ok(())
    }

    #[test]
    fn lane_admission_refuses_underbudget_invalid_counts_and_overflow() {
        let plan = RankAdmission { immutable: 100, workspace: 20 };
        assert!(plan.check(2, 139, &[]).is_err());
        assert!(plan.check(2, 140, &[1]).is_err());
        assert!(plan.check(0, u64::MAX, &[]).is_err());
        assert!(plan.check(3, u64::MAX, &[]).is_err());
        assert!(plan.check(2, u64::MAX, &[u64::MAX]).is_err());
        assert!(RankAdmission { immutable: 0, workspace: u64::MAX }.bytes(2).is_err());
        assert!(RankAdmission { immutable: u64::MAX, workspace: 1 }.bytes(1).is_err());
    }
}
