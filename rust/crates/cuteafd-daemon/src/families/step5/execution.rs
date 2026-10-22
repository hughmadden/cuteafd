//! One native execution owner with shared per-rank scratch.
use super::{scratch::Arena, cache::CacheRank, csa::CsaLayer, ffn::{DenseFfn, DensePlan},
    rope::TargetTables, routed::{RoutedFfn, RoutedPlan}, swa::SwaLayer,
    transaction::{Staging, Step}};
use crate::shared::{experts::{rtx::pair::{self, SubmissionScope}, staging::RouteStaging},
    memory::{HostAllocation, device::{Allocation, Device, DeviceOwner}},
    peer_split::{PeerExchange, RankDevice, attention::{self, AttentionStateOwners,
        HeadKernels, HeadMerge, HeadPrecision, HeadProjection, HeadSequence,
        LaneOrder, LayerKernels, LayerProtocol, PipelineKernels}, hop::{HopLink, HopTicket}, TerminalState}, spark_intake::SparkLink};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::qwen4::Qwen4KvCache;
use cuteafd_ffi::{CuteafdDeviceBuffer, programs::Programs};
use cuteafd_hostcache::copy::CopyEngine;
use cuteafd_loader::{placement::{Hop, HopPoint, LayerMode}, plan::{checkpoint::Checkpoint,
    families::step5::{coordinator::{CoordinatorContract, RankWeightLayout}, ownership::ExecutionOwnership}}};
use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RouteEntry,
    ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
    expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
use std::{cell::{Cell, RefCell}, ffi::c_void, rc::Rc};

const H: usize = 4096;

/// Exact native allocation bytes, excluding caller-owned cache/transport/code reserves.
/// The final head contributes weights once and shares the sequential scratch arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RankAdmission {
    pub(super) immutable: u64,
    pub(super) workspace: u64,
}
impl RankAdmission {
    fn bytes(self) -> Result<u64> {
        self.workspace.checked_add(self.immutable).context("Step-5 native admission overflow")
    }

    pub(super) fn check(self, available: u64, reservations: &[u64]) -> Result<u64> {
        let bytes = self.bytes()?;
        let total = reservations.iter().try_fold(bytes, |n, &v| n.checked_add(v))
            .context("Step-5 native reservation overflow")?;
        ensure!(total <= available, "Step-5 native weights and workspace need {total} bytes, available {available}");
        Ok(bytes)
    }

    fn with_head(self, head: &super::head::HeadPlan) -> Result<Self> {
        Ok(Self {
            immutable: self.immutable.checked_add(head.immutable_bytes())
                .context("Step-5 native head weight overflow")?,
            workspace: self.workspace.max(head.workspace_bytes()),
        })
    }
}

enum Attention<'a> { Sliding(SwaLayer<'a>), Sparse(CsaLayer<'a>) }
enum Ffn<'a> { Dense(DenseFfn<'a>), Routed(RoutedFfn<'a>) }
struct Layer<'a> { attention: Attention<'a>, ffn: Ffn<'a> }
impl<'a> Layer<'a> {
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

/// The native owner retains component operands and one sequential arena per rank.
struct NativeOwners<'a> {
    ranks: Vec<Vec<Option<Layer<'a>>>>,
    head: Option<super::head::TargetHeadOwner<'a>>,
    arenas: Vec<Arena<'a>>,
    rope: Vec<Rc<TargetTables<'a>>>,
}
impl<'a> NativeOwners<'a> {
    fn load(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: &ExecutionOwnership, devices: &[Device<'a>], rows: usize,
        count: usize, available: &[u64], reservations: &[u64], depth: usize, kv_format: Qwen4KvCache) -> Result<Self> {
        validate_rank_inventory(ownership.ranks(), &[devices.len(), available.len(), reservations.len()])?;
        ensure!(matches!(count, 1 | 92), "Step-5 native layer inventory");
        let head_rank = ownership.execution().spec().head_gpu as usize;
        ensure!(head_rank < ownership.ranks(), "Step-5 final head rank");
        let head_plan = (count == 92).then(||
            super::head::HeadPlan::for_execution(programs, checkpoint, rows, ownership)).transpose()?;
        let mut admission = vec![RankAdmission { immutable: 0, workspace: 0 }; ownership.ranks()];
        for (rank, plan) in admission.iter_mut().enumerate() {
            *plan = Execution::rank_admission(programs, checkpoint, ownership, rank, rows, count)?;
            if rank == head_rank {
                if let Some(head) = &head_plan { *plan = plan.with_head(head)?; }
                plan.workspace = plan.workspace.max(
                    cuteafd_loader::plan::families::step5::native::mtp_forecast_workspace(rows, depth, kv_format)?);
            }
        }
        admit_native(&admission, available, reservations, || {
            let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
            let mut ranks: Vec<Vec<Option<Layer<'a>>>> = (0..ownership.ranks()).map(|_| Vec::new()).collect();
            let mut arenas = Vec::with_capacity(ownership.ranks());
            let mut tables = Vec::with_capacity(ownership.ranks());
            for rank in 0..ownership.ranks() {
                let device = devices[rank];
                let arena = Arena::new(device, usize::try_from(admission[rank].workspace)?)?;
                let rope = Rc::new(TargetTables::load(device, programs, &contract, "step5_", available[rank], &[])?);
                tables.push(rope.clone());
                for layer in 0..count {
                    if ownership.layer(layer, rank)?.query_heads == 0 { ranks[rank].push(None); continue; }
                    let attention = if contract.layers[layer].sliding.is_some() {
                        Attention::Sliding(SwaLayer::for_execution(device, programs, checkpoint, layer,
                            rows, Some(rope.clone()), available[rank], &[], rank, &arena, ownership, kv_format)?)
                    } else {
                        Attention::Sparse(CsaLayer::for_execution(device, programs, checkpoint, layer,
                            rows, Some(rope.clone()), available[rank], &[], rank, &arena, ownership, kv_format)?)
                    };
                    let offset = usize::try_from(match &attention {
                        Attention::Sliding(_) => cuteafd_loader::plan::families::step5::native::plane_bytes(
                            &cuteafd_loader::plan::families::step5::native::swa_planes(rows, ownership.layer(layer, rank)?.query_heads)?)?,
                        Attention::Sparse(_) => cuteafd_loader::plan::families::step5::native::plane_bytes(
                            &cuteafd_loader::plan::families::step5::native::csa_planes(rows)?)?,
                    })?;
                    let ffn = if contract.layers[layer].dense_activation.is_some() {
                        let plan = DensePlan::for_execution(programs, checkpoint, layer, rows, u64::MAX, &[], rank, ownership)?;
                        Ffn::Dense(DenseFfn::load_plan_in(device, programs, checkpoint, plan, Some((&arena, offset)))?)
                    } else {
                        let plan = RoutedPlan::for_execution(programs, checkpoint, layer, rows, u64::MAX, &[], rank, ownership)?;
                        Ffn::Routed(RoutedFfn::load_in(device, programs, checkpoint, layer, plan, Some((&arena, offset)))?)
                    };
                    ranks[rank].push(Some(Layer { attention, ffn }));
                }
                arenas.push(arena);
            }
            let head = head_plan.map(|plan| super::head::TargetHeadOwner::load_in(devices[head_rank],
                programs, checkpoint, plan, available[head_rank], &[], Some((&arenas[head_rank], 0)))).transpose()?;
            Ok(Self { ranks, head, arenas, rope: tables })
        })
    }
}

fn validate_rank_inventory(ranks: usize, lengths: &[usize]) -> Result<()> {
    ensure!(matches!(ranks, 1 | 2) && lengths.iter().all(|&length| length == ranks),
        "Step-5 rank inventory must match one or two admitted ranks");
    Ok(())
}

fn admit_native<T>(admission: &[RankAdmission],
    available: &[u64], reservations: &[u64], allocate: impl FnOnce() -> Result<T>) -> Result<T> {
    validate_rank_inventory(admission.len(), &[available.len(), reservations.len()])?;
    for rank in 0..admission.len() { admission[rank].check(available[rank], &[reservations[rank]])?; }
    allocate()
}

pub(super) struct Execution<'e, 'a> {
    main: MainExecution<'e, 'a>,
    startup_captures: u64,
}

struct MainExecution<'e, 'a> {
    ownership: ExecutionOwnership,
    native: NativeOwners<'a>,
    caches: Vec<&'e RefCell<CacheRank<'a>>>,
    stages: Vec<&'e Staging<'a>>,
    exchange: Exchange<'a>,
    sequence: RefCell<HeadSequence>,
    pending_wave: RefCell<Option<cuteafd_transport::expert::SparkExpertWave>>,
    transport: RefCell<Option<DeviceOwner<'a, SparkLink<'a>>>>,
    runtime: tokio::runtime::Runtime,
    routed_reply: Allocation<'a>,
    pinned: Vec<RefCell<HostAllocation<'a>>>,
    rows: usize,
    inventory: Vec<Inventory>,
    requests: Cell<u64>,
    placement_version: u64,
    terminal_output: Cell<Option<CuteafdDeviceBuffer>>,
    poison_scratch: Cell<bool>,
    diagnostic_outputs: RefCell<std::collections::BTreeMap<usize, Vec<u8>>>,
}

enum Exchange<'a> {
    Local(Cell<TerminalState>),
    Pair { peer: PeerExchange<'a>, hops: HopLink<'a> },
}
impl<'a> Exchange<'a> {
    fn peer(&self) -> Result<&PeerExchange<'a>> {
        match self { Self::Pair { peer, .. } => Ok(peer), Self::Local(_) => anyhow::bail!("Step-5 local execution has no peer") }
    }
    fn terminal_state(&self) -> TerminalState {
        match self { Self::Pair { peer, .. } => peer.terminal_state(), Self::Local(state) => state.get() }
    }
    fn publish_abort(&self) -> Result<()> {
        match self {
            Self::Pair { peer, .. } => peer.publish_abort(),
            Self::Local(state) => {
                ensure!(matches!(state.get(), TerminalState::Active | TerminalState::Drained),
                    "Step-5 local execution is already terminal");
                if state.get() == TerminalState::Active { state.set(TerminalState::Aborting); }
                Ok(())
            }
        }
    }
    fn finish_terminal(&self, drained: bool) {
        match self {
            Self::Pair { peer, .. } => peer.finish_terminal(drained),
            Self::Local(state) => state.set(if drained { TerminalState::Drained } else { TerminalState::Retained }),
        }
    }
}
impl LayerProtocol for Exchange<'_> {
    type Ticket = HopTicket;
    fn require_live(&self) -> Result<()> {
        ensure!(self.terminal_state() == TerminalState::Active, "Step-5 terminal execution cannot be reused");
        Ok(())
    }
    fn push_entry(&self, from: usize, source: *const c_void, destination: *mut c_void, bytes: usize) -> Result<()> {
        self.peer()?.push_to(from, crate::shared::peer_split::DIRECT, source, destination, bytes)
    }
    fn wait_entry(&self, to: usize) -> Result<()> { self.peer()?.wait(to, crate::shared::peer_split::DIRECT) }
    fn send(&self, index: usize, lane: usize, source: *const c_void, rows: usize) -> Result<Self::Ticket> {
        match self { Self::Pair { hops, .. } => hops.send(index, lane, source, rows),
            Self::Local(_) => anyhow::bail!("Step-5 local execution cannot send a peer hop") }
    }
    fn land(&self, ticket: Self::Ticket) -> Result<*mut c_void> {
        match self { Self::Pair { hops, .. } => hops.land(ticket),
            Self::Local(_) => anyhow::bail!("Step-5 local execution cannot receive a peer hop") }
    }
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
        ensure!(matches!(ownership.ranks(), 1 | 2), "Step-5 shared execution rank count");
        Self::rank_admission(programs, checkpoint, ownership, rank, rows, count)?
            .bytes()
    }

    /// Includes the vocabulary head on its owning rank. Call for every rank before
    /// allocating native owners; cache, staging, input, transport and code are reserves.
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
                cuteafd_loader::plan::families::step5::native::plane_bytes(
                            &cuteafd_loader::plan::families::step5::native::swa_planes(rows, ownership.layer(layer, rank)?.query_heads)?)? }
                else { cuteafd_loader::plan::families::step5::native::plane_bytes(
                            &cuteafd_loader::plan::families::step5::native::csa_planes(rows)?)? };
            let ffn = if contract.layers[layer].dense_activation.is_some() {
                DensePlan::for_execution(programs, checkpoint, layer, rows, u64::MAX, &[], rank, ownership)?.workspace_bytes()
            } else {
                RoutedPlan::for_execution(programs, checkpoint, layer, rows, u64::MAX, &[], rank, ownership)?.workspace_bytes()
            };
            workspace = workspace.max(attention.checked_add(ffn)
                .context("Step-5 native rank workspace overflow")?);
        }
        Ok(RankAdmission { immutable, workspace })
    }

    pub fn load(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: ExecutionOwnership, caches: Vec<&'e RefCell<CacheRank<'a>>>,
        stages: Vec<&'e Staging<'a>>, rows: usize, available: Vec<u64>,
        reservations: Vec<u64>, transport: Option<DeviceOwner<'a, SparkLink<'a>>>,
        placement_version: u64) -> Result<Self> {
        Self::load_count(programs, checkpoint, ownership, caches, stages, rows, available,
            reservations, transport, 92, placement_version)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_count(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: ExecutionOwnership, caches: Vec<&'e RefCell<CacheRank<'a>>>,
        stages: Vec<&'e Staging<'a>>, rows: usize, available: Vec<u64>,
        reservations: Vec<u64>, transport: Option<DeviceOwner<'a, SparkLink<'a>>>,
        count: usize, placement_version: u64) -> Result<Self> {
        let native = Self::load_native(programs, checkpoint, &ownership, &caches, &stages, rows, &available,
            &reservations, count)?;
        Self::bind_native(checkpoint, ownership, native, caches, stages, rows, transport, count, placement_version)
    }

    /// The admitted native owners (weights and arenas) alone, before any Spark
    /// transport exists: serving loads them while Spark admission is in flight.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_native(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        ownership: &ExecutionOwnership, caches: &[&'e RefCell<CacheRank<'a>>], stages: &[&'e Staging<'a>],
        rows: usize, available: &[u64], reservations: &[u64], count: usize) -> Result<LoadedNative<'a>> {
        ensure!(matches!(count, 1 | 92), "Step-5 execution count must be layer-zero diagnostic or full target");
        validate_rank_inventory(ownership.ranks(), &[caches.len(), stages.len(), available.len(), reservations.len()])?;
        let kv_format = caches[0].borrow().layout().kv_format;
        for rank in 0..ownership.ranks() {
            ensure!(rows as u64 <= ownership.execution().spec().rows, "Step-5 live rows exceed execution plan");
            let cache = caches[rank].borrow();
            ensure!(cache.layout().kv_format == kv_format, "Step-5 rank cache formats differ");
            ensure!(cache.rank() == rank && cache.layout().ranks.len() == ownership.ranks(), "Step-5 cache rank inventory");
            stages[rank].validate_cache(&cache)?;
        }
        let devices: Vec<_> = caches.iter().map(|cache| cache.borrow().stream.device).collect();
        ensure!(devices.iter().enumerate().all(|(rank, device)|
            std::ptr::eq(device.library, devices[0].library)
                && devices[..rank].iter().all(|other| other.id != device.id)), "Step-5 rank devices/library differ");
        let depth = caches[ownership.execution().spec().head_gpu as usize].borrow()
            .layout().ranks[ownership.execution().spec().head_gpu as usize].mtp_lengths.len();
        Ok(LoadedNative(NativeOwners::load(programs, checkpoint, ownership, &devices, rows, count, available,
            reservations, depth, kv_format)?))
    }

    /// Binds loaded native owners to the caches and the (now connected) transport.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bind_native(checkpoint: &'a Checkpoint, ownership: ExecutionOwnership, native: LoadedNative<'a>,
        caches: Vec<&'e RefCell<CacheRank<'a>>>, stages: Vec<&'e Staging<'a>>, rows: usize,
        transport: Option<DeviceOwner<'a, SparkLink<'a>>>, count: usize, placement_version: u64) -> Result<Self> {
        ensure!(placement_version != 0, "Step-5 execution needs an admitted placement version");
        ensure!(transport.is_some() == (count == 92), "Step-5 main Spark transport inventory");
        let main = MainExecution::bind(checkpoint, ownership, native.0, caches, stages, rows, transport, count,
            placement_version)?;
        Ok(Self { main, startup_captures: cuteafd_ffi::graph_captures() })
    }
}

/// Native owners loaded ahead of their execution binding.
pub(super) struct LoadedNative<'a>(NativeOwners<'a>);

impl<'e, 'a> MainExecution<'e, 'a> {
    #[allow(clippy::too_many_arguments)]
    fn bind(checkpoint: &'a Checkpoint, ownership: ExecutionOwnership, native: NativeOwners<'a>,
        caches: Vec<&'e RefCell<CacheRank<'a>>>, stages: Vec<&'e Staging<'a>>, rows: usize,
        transport: Option<DeviceOwner<'a, SparkLink<'a>>>, count: usize,
        placement_version: u64) -> Result<Self> {
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let devices: Vec<_> = caches.iter().map(|cache| cache.borrow().stream.device).collect();
        let exchange = if ownership.ranks() == 2 {
            let native_ranks = std::array::from_fn(|rank| RankDevice { device: devices[rank].id,
                stream: stages[rank].stream.raw });
            let peer = devices[0].run(|| PeerExchange::new_abortable(devices[0].library, native_ranks, 8, rows * H * 4))?;
            let mut hops = devices[0].run(|| HopLink::new_private_lane(devices[0].library, native_ranks,
                ownership.execution().hops(), ownership.execution().spec(), 0))?;
            hops.attach_abort([peer.abort_word(0)?.context("Step-5 lead abort")?,
                peer.abort_word(1)?.context("Step-5 peer abort")?])?;
            hops.validate_binding(ownership.execution(), &peer)?;
            Exchange::Pair { peer, hops }
        } else {
            ensure!(ownership.execution().hops().is_empty(), "Step-5 local execution cannot own peer hops");
            Exchange::Local(Cell::new(TerminalState::Active))
        };
        let ranks = &native.ranks;
        let bytes = RouteStaging::new(rows, 8, H * 2)?.bytes;
        let pinned = devices.iter().map(|device| HostAllocation::new(device.library, bytes).map(RefCell::new))
            .collect::<Result<Vec<_>>>()?;
        let mut inventory = Vec::new();
        for layer in 0..count {
            let mut state = Inventory { weights: Vec::new(), kv: Vec::new(),
                indexer: contract.layers[layer].sliding.is_none().then(Vec::new),
                compressor: contract.layers[layer].sliding.is_none().then(Vec::new) };
            for rank in 0..ownership.ranks() {
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
        let entry = usize::from(ownership.execution().spec().entry_gpu);
        if let Some(link) = &transport {
            ensure!(link.device.id == devices[entry].id, "Step-5 main intake must be on the entry GPU");
            link.transport.installed_grouping()?;
        }
        let routed_reply = Allocation::new(devices[entry], rows * H * 2)?;
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        let execution = Self { ownership, native, caches, stages, exchange,
            sequence: RefCell::new(HeadSequence::new(0)?), pending_wave: RefCell::new(None), transport: RefCell::new(transport), runtime, routed_reply,
            pinned, rows, inventory, requests: Cell::new(0), placement_version,
            terminal_output: Cell::new(None), poison_scratch: Cell::new(false),
            diagnostic_outputs: RefCell::new(std::collections::BTreeMap::new()) };
        if count == 92 { execution.validate_replications()?; }
        Ok(execution)
    }

    fn device(&self, rank: usize) -> Result<Device<'a>> {
        Ok(self.caches.get(rank).context("Step-5 device outside admitted ranks")?.borrow().stream.device)
    }
    fn layer(&self, rank: usize, layer: usize) -> Result<&Layer<'a>> {
        self.native.ranks.get(rank).and_then(|layers| layers.get(layer)).and_then(Option::as_ref)
            .context("Step-5 missing live layer owner")
    }
    fn drain_compute(&self) -> Result<()> {
        match &self.exchange {
            Exchange::Pair { peer, .. } => peer.drain_compute(),
            Exchange::Local(state) => {
                ensure!(state.get() == TerminalState::Aborting, "Step-5 local drain requires terminal state");
                collect_lane_results(self.stages.iter().map(|stage| stage.stream.drain()))
            }
        }
    }
    fn output(&self, layer: usize) -> Result<(Device<'a>, CuteafdDeviceBuffer)> {
        let rank = match self.ownership.execution().mode(layer)? {
            LayerMode::Whole { gpu, .. } => gpu as usize, LayerMode::HeadSplit => 0,
            LayerMode::ContextSplit => anyhow::bail!("Step-5 context output unsupported"),
        };
        Ok((self.device(rank)?, self.layer(rank, layer)?.output()))
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
        ensure!(self.ownership.ranks() == 2 && self.native.ranks.iter().all(|layers| layers.len() == 1), "layer-zero diagnostic must load only layer zero");
        step.validate_binding(&self.caches, &self.stages, self.rows)?;
        pair::submit_scoped(|| self.exchange.require_live(), || self.shutdown(), |scope| {
            step.in_submission(scope, |step| {
                step.compose(0, input, |_, _| Ok(()))?;
                attention::replicate_terminal(self.exchange.peer()?, input.buffer, self.layer(1, 0)?.input(), step.rows() * H * 4)?;
                let mut binding = Bindings { execution: self, step, scope,
                    current: vec![None, Some(self.layer(1, 0)?.input())], input: input.buffer,
                    kind: ExpertV2SourceKind::Benchmark, layer: 0, diagnostic: Some(directory) };
                binding.heads(0)?;
                binding.ffn(0, None)?;
                for stage in &self.stages { stage.stream.drain()?; }
                Ok(())
            })
        })
    }

    fn validate_replications(&self) -> Result<()> {
        let mut copies = self.ownership.routed_replications()?;
        for (layer, copy) in copies.iter_mut().enumerate() {
            if let Some(copy) = copy {
                let Ffn::Routed(destination) = &self.layer(1 - copy.source as usize, layer)?.ffn else { anyhow::bail!("Step-5 routed receive executor"); };
                let (source, destination) = (self.routed_reply.buffer, destination.routed_output());
                ensure!(!source.ptr.is_null() && !destination.ptr.is_null()
                    && source.bytes >= self.rows * H * 2 && destination.bytes >= self.rows * H * 2
                    && source.device_id == self.device(copy.source as usize)?.id
                    && destination.device_id == self.device(1 - copy.source as usize)?.id,
                    "Step-5 routed replication live storage/device");
                copy.storage = destination.ptr as usize;
            }
        }
        self.ownership.execution().validate_routed_replications(&copies)?;
        Ok(())
    }
    fn shutdown(&self) -> Result<()> {
        if self.exchange.terminal_state() == TerminalState::Drained { return Ok(()); }
        let result = pair::teardown(&mut pair::Teardown {
            abort: || self.exchange.publish_abort(),
            ranks: || self.drain_compute(),
            links: || self.with_transport(|link| link.terminal_quiesce()),
            copies: || self.with_transport(|link| { link.terminal_drain_copies()?; link.terminal_release() }),
            graphs: || Ok(()),
            finish: |drained| {
                self.exchange.finish_terminal(drained);
                if !drained { self.caches[0].borrow().stream.device.library.quarantine_module_after_failed_drain(); }
            },
        });
        crate::shared::decode_graph::fatal_drain(result, "Step-5 pipeline terminal owners");
        Ok(())
    }

    fn with_transport(&self, work: impl FnOnce(&mut SparkLink<'a>) -> Result<()>) -> Result<()> {
        if let Some(link) = self.transport.borrow_mut().as_mut() {
            let device = link.device;
            device.run(|| work(link.get_mut()))?;
        }
        Ok(())
    }
}
impl Drop for MainExecution<'_, '_> {
    fn drop(&mut self) { if let Err(error) = self.shutdown() {
        tracing::error!(%error, "Step-5 native pipeline teardown retained queued owners");
    }}
}

// Attempt every owner in a phase, even if an earlier owner fails. In particular,
// every abort publication precedes the first compute drain across all ranks.
pub(super) fn collect_lane_results(results: impl IntoIterator<Item = Result<()>>) -> Result<()> {
    let failures: Vec<_> = results.into_iter().filter_map(Result::err)
        .map(|error| format!("{error:#}")).collect();
    ensure!(failures.is_empty(), "Step-5 rank owners: {}", failures.join("; "));
    Ok(())
}

impl<'e, 'a> Execution<'e, 'a> {
    pub(super) fn validate_caches(&self, caches: &[&RefCell<CacheRank<'a>>], rows: usize) -> Result<()> {
        let owner = &self.main;
        ensure!(owner.rows == rows && caches.len() == owner.caches.len()
            && caches.iter().zip(&owner.caches).all(|(left, right)| std::ptr::eq(*left, *right)),
            "Step-5 backend cache owners differ from execution");
        Ok(())
    }
    pub(super) fn terminal_output(&self) -> Result<(Device<'a>, CuteafdDeviceBuffer)> {
        let rank = self.main.ownership.execution().spec().head_gpu as usize;
        Ok((self.main.device(rank)?, self.main.terminal_output.take().context("Step-5 terminal wave absent")?))
    }
    pub(super) fn require_live(&self) -> Result<()> {
        check_captures(self.startup_captures, cuteafd_ffi::graph_captures())?;
        self.main.exchange.require_live()?;
        ensure!(self.main.caches.iter().all(|cache| !cache.borrow().poisoned.get()), "Step-5 poisoned cache");
        Ok(())
    }
    pub(super) fn draft_geometry(&self, width: usize, rows: usize) -> Result<cuteafd_core::PolicyGeometry> {
        use crate::shared::draft::resource::{self, LayerHome};
        let transport = self.main.transport.borrow();
        let reports = transport.as_ref().context("Step-5 main transport missing")?.transport.installed_grouping()?;
        let homes = (0..92).map(|layer| {
            if layer < 3 || layer == 91 { Ok(LayerHome::Dense) }
            else { Ok(LayerHome::Remote { slice_bytes: resource::worker_weight_read_bytes(&reports, layer)? as f64 }) }
        }).collect::<Result<Vec<_>>>()?;
        resource::geometry_with_grouping(&homes, 352, 8, rows, width, 0,
            |layer, rows| resource::worker_group_rows_for_input(&reports, layer, rows, true))
    }
    pub(super) fn predictor_rope(&self) -> Rc<TargetTables<'a>> {
        self.main.native.rope[self.ownership().execution().spec().head_gpu as usize].clone()
    }
    pub(super) fn predictor_arena(&self) -> &Arena<'a> {
        &self.main.native.arenas[self.ownership().execution().spec().head_gpu as usize]
    }
    pub(super) fn predictor_stream(&self) -> Rc<crate::shared::memory::device::Stream<'a>> {
        self.main.stages[self.ownership().execution().spec().head_gpu as usize].stream.clone()
    }
    pub(super) fn poison_scratch_between_layers(&self) { self.main.poison_scratch.set(true); }
    pub(super) fn retain_diagnostic_outputs(&self, layers: &[usize]) -> Result<()> {
        ensure!(layers.iter().all(|&layer| layer < self.main.inventory.len()), "Step-5 diagnostic layer outside loaded target");
        *self.main.diagnostic_outputs.borrow_mut() = layers.iter().map(|&layer| (layer, Vec::new())).collect();
        Ok(())
    }
    pub(super) fn diagnostic_output(&self, layer: usize) -> Result<Vec<f32>> {
        let outputs = self.main.diagnostic_outputs.borrow();
        let bytes = outputs.get(&layer).filter(|bytes| !bytes.is_empty())
            .context("Step-5 diagnostic residual was not retained before scratch reuse")?;
        Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }
    pub(super) fn stages(&self) -> Vec<&'e Staging<'a>> { self.main.stages.clone() }
    pub(super) fn ownership(&self) -> &ExecutionOwnership { &self.main.ownership }
    pub(super) fn head(&self) -> Result<&super::head::TargetHeadOwner<'a>> { self.main.head() }
    pub(super) fn diagnostic_layer0(&self, rank: usize) -> Result<(&SwaLayer<'a>, &DenseFfn<'a>)> { self.main.diagnostic_layer0(rank) }
    pub(super) fn enqueue_layer0<'s, E: CopyEngine>(&'s self, step: &mut Step<'s, 'a, E>,
        input: &'s Allocation<'a>, directory: &'s std::path::Path) -> Result<()> { self.main.enqueue_layer0(step, input, directory) }
    pub fn enqueue<'s, E: CopyEngine>(&'s self, step: &mut Step<'s, 'a, E>,
        input: &'s Allocation<'a>, kind: ExpertV2SourceKind) -> Result<()> {
        let main = &self.main;
        main.terminal_output.set(None);
        step.validate_binding(&main.caches, &main.stages, main.rows)?;
        if let Exchange::Pair { peer, hops } = &main.exchange {
            hops.validate_binding(main.ownership.execution(), peer)?;
            hops.validate_dispatch(0, false)?;
        }
        main.validate_replications()?;
        ensure!(!input.buffer.ptr.is_null()
            && input.buffer.device_id == main.device(main.ownership.execution().spec().entry_gpu as usize)?.id
            && input.buffer.bytes >= step.rows() * H * 4, "Step-5 live input allocation");
        pair::submit_scoped(|| main.exchange.require_live(), || main.shutdown(), |scope| step.in_submission(scope, |step| {
            step.compose(main.ownership.execution().spec().entry_gpu as usize, input, |_, _| Ok(()))?;
            let mut binding = Bindings { execution: main, step, scope, current: vec![None; main.ownership.ranks()],
                input: input.buffer, kind, layer: 0, diagnostic: None };
            let rows = binding.step.rows();
            attention::layer_pipeline_with(scope, main.ownership.execution(), &main.exchange,
                LaneOrder::Independent, &[rows], std::slice::from_mut(&mut binding))?;
            let head = main.ownership.execution().spec().head_gpu as usize;
            let output = binding.current[head].context("Step-5 pipeline omitted terminal head output")?;
            ensure!(output.device_id == main.device(head)?.id, "Step-5 terminal output rank mismatch");
            main.terminal_output.set(Some(output));
            collect_lane_results(main.stages.iter().map(|stage| stage.stream.drain()))
        }))
    }
}

fn expert_request(identity: u64, placement_version: u64, layer: usize,
    rows: Vec<ExpertProtocolV2RowDescriptor>, routes: Vec<ExpertProtocolV2RouteEntry>, wire: bytes::Bytes,
) -> Result<ExpertProtocolV2Request> {
    ensure!(identity != 0 && placement_version != 0 && (3..91).contains(&layer),
        "Step-5 routed request identity, placement or target layer");
    let mut request = ExpertProtocolV2Request::new_bytes(identity, placement_version,
        u32::try_from(layer)?, H as u32, ExpertV2Dtype::Bf16, rows, routes, wire)?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

pub(super) fn bootstrap_request(rows: usize, placement_version: u64) -> Result<ExpertProtocolV2Request> {
    ensure!((1..=4096).contains(&rows), "Step-5 bootstrap row capacity");
    let descriptors = (0..rows as u32).map(|row| ExpertProtocolV2RowDescriptor {
        row_id: u64::from(row), source_kind: ExpertV2SourceKind::Prefill, source_request_id: 1,
        token_position: u64::from(row), route_offset: row * 8, route_count: 8,
    }).collect();
    let routes = (0..rows * 8).map(|i| ExpertProtocolV2RouteEntry {
        row_index: (i / 8) as u32, expert_id: (i % 352) as u32, gate_weight: 0.,
    }).collect();
    expert_request(1, placement_version, 3, descriptors, routes, vec![0; rows * H * 2].into())
}

fn check_captures(startup: u64, current: u64) -> Result<()> {
    ensure!(startup == current, "Step-5 zero-steady-capture contract: startup={startup}, current={current}");
    Ok(())
}

fn request_identity(ordinal: u64) -> Result<u64> {
    ordinal.checked_add(1).context("Step-5 request namespace exhausted")
}

struct Bindings<'s, 'b, 'e, 'a, 'g, E: CopyEngine> {
    execution: &'s MainExecution<'e, 'a>, step: &'b mut Step<'s, 'a, E>,
    scope: &'g SubmissionScope<'g>, current: Vec<Option<CuteafdDeviceBuffer>>,
    input: CuteafdDeviceBuffer, kind: ExpertV2SourceKind, layer: usize,
    diagnostic: Option<&'s std::path::Path>,
}
impl<'s, E: CopyEngine> Bindings<'s, '_, '_, '_, '_, E> {
    fn retain_diagnostic_output(&self, layer: usize) -> Result<()> {
        if !self.execution.diagnostic_outputs.borrow().contains_key(&layer) { return Ok(()); }
        let (device, output) = self.execution.output(layer)?;
        let rank = match self.execution.ownership.execution().mode(layer)? {
            LayerMode::Whole { gpu, .. } => gpu as usize,
            LayerMode::HeadSplit => 0,
            LayerMode::ContextSplit => anyhow::bail!("Step-5 context diagnostic unsupported"),
        };
        self.execution.stages[rank].stream.drain()?;
        let mut bytes = vec![0; self.step.rows() * H * 4];
        device.run(|| device.library.copy_d2h(&mut bytes, output))?;
        self.execution.diagnostic_outputs.borrow_mut().insert(layer, bytes);
        Ok(())
    }

    fn copy(&mut self, rank: usize, destination: CuteafdDeviceBuffer, source: CuteafdDeviceBuffer) -> Result<()> {
        ensure!(destination.device_id == self.execution.device(rank)?.id
            && source.device_id == destination.device_id && !source.ptr.is_null() && !destination.ptr.is_null()
            && source.bytes >= self.step.rows() * H * 4 && destination.bytes >= self.step.rows() * H * 4,
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
        if self.execution.poison_scratch.get() {
            self.execution.stages[rank].stream.drain()?;
            self.execution.native.arenas[rank].poison_after(self.step.rows() * H * 4)?;
        }
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
            bytes: self.step.rows() * H * 4, device_id: self.execution.device(hop.to as usize)?.id, flags: 0 });
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
                    owner.diagnostic_norm(), self.step.rows(), true)?;
                super::golden::dump_plane(directory, &format!("rank{rank}-o_partial"), device,
                    owner.projection_partial()?, self.step.rows(), false)?;
            }
        }
        let execution = self.execution;
        attention::head_partials(self.scope, execution.exchange.peer()?, &mut execution.sequence.borrow_mut(),
            self.step.rows() * H, HeadPrecision::F32, self)
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
        for rank in 0..execution.ownership.ranks() {
            if owner.is_some_and(|owner| owner != rank) { continue; }
            let local = execution.layer(rank, layer)?;
            self.copy(rank, local.ffn_input(), local.post())?;
            match &local.ffn {
                Ffn::Dense(ffn) => self.step.compose(rank, ffn, |ffn, meta| {
                    ffn.enqueue_on(meta.rows, meta.stream)
                })?,
                Ffn::Routed(ffn) => {
                    let dispatch = execution.ownership.layer(layer, rank)?.routed_wave;
                    self.step.compose(rank, ffn, |ffn, meta| {
                        if dispatch { ffn.frontend_on(meta.rows, meta.stream) }
                        else { ffn.normalize_on(meta.rows, meta.stream) }
                    })?;
                    if dispatch {
                        ensure!(wave.is_none(), "Step-5 duplicate routed Spark wave");
                        let layout = RouteStaging::new(self.step.rows(), 8, H * 2)?;
                        let host = execution.pinned[rank].borrow();
                        self.step.compose(rank, ffn, |ffn, meta| {
                            meta.stream.device.run(|| {
                                // SAFETY: pinned arena belongs to this rank; next write follows wait and consumed-event drainage.
                                unsafe { layout.enqueue(meta.stream.device.library, host.buffer, ffn.wave_inputs(), meta.stream.raw) }
                            })?;
                            // Both head publications were queued before either merge. No future
                            // publication is needed for this stream's staging completion.
                            meta.stream.drain()
                        })?;
                        let routes = layout.routes(host.bytes())?;
                        let wire = bytes::Bytes::copy_from_slice(layout.wire(host.bytes())?);
                        let descriptors = self.step.expert_rows(self.kind)?;
                        let mut transport = execution.transport.borrow_mut();
                        let link = transport.as_mut().context("Step-5 main transport missing")?;
                        let device = link.device;
                        let ordinal = execution.requests.get();
                        let identity = request_identity(ordinal)?;
                        execution.requests.set(identity);
                        let request = expert_request(identity, execution.placement_version, layer, descriptors, routes, wire)?;
                        ensure!(execution.pending_wave.borrow().is_none(), "Step-5 main wave already pending");
                        let ticket = device.run(|| link.get_mut().dispatch(&request))?;
                        *execution.pending_wave.borrow_mut() = Some(ticket);
                        wave = Some(rank);
                    }
                    self.step.compose(rank, ffn, |ffn, meta| {
                        ffn.shared_on(meta.rows, meta.stream)
                    })?;
                }
            }
            self.current[rank] = Some(local.output());
        }
        Ok(wave)
    }
    fn finish_ffn(&mut self, layer: usize, ticket: Self::FfnTicket) -> Result<()> {
        let execution = self.execution;
        let Some(rank) = ticket else { return self.retain_diagnostic_output(layer); };
        let wave = execution.pending_wave.borrow_mut().take().context("Step-5 main wave absent")?;
        let entry = usize::from(execution.ownership.execution().spec().entry_gpu);
        let Ffn::Routed(ffn) = &execution.layer(rank, layer)?.ffn else { anyhow::bail!("Step-5 routed completion owner"); };
        let split = execution.ownership.execution().mode(layer)? == LayerMode::HeadSplit;
        let output = if rank == entry { ffn.routed_output() } else { execution.routed_reply.buffer };
        self.step.compose(entry, &execution.routed_reply, |_, meta| {
            let mut transport = execution.transport.borrow_mut();
            let link = transport.as_mut().context("Step-5 main transport missing")?;
            let device = link.device;
            device.run(|| execution.runtime.block_on(link.get_mut().receive(wave, meta.rows, meta.stream.raw)))?;
            device.run(|| {
                // SAFETY: intake planes and the admitted BF16 reply owner are
                // disjoint; the submission retains both through their consumers.
                unsafe { device.library.v41_compact_reducer()?.reduce_planes(link.intake.pointers(),
                    link.world_size() as u32, std::ptr::null(), output.ptr.cast(), meta.rows as u32, meta.stream.raw) }
            })?;
            device.run(|| link.intake.consumed(meta.stream.raw))
        })?;
        if split || rank != entry {
            let destination_rank = 1 - entry;
            let Ffn::Routed(destination) = &execution.layer(destination_rank, layer)?.ffn else { anyhow::bail!("Step-5 routed receive owner"); };
            let binding = execution.ownership.routed_replications()?[layer].context("Step-5 routed publication binding")?;
            attention::replicate_result(self.scope, execution.exchange.peer()?, entry, binding.slot,
                output, destination.routed_output(), self.step.rows() * H * 2)?;
        }
        for at in 0..execution.ownership.ranks() {
            if !split && at != rank { continue; }
            let Ffn::Routed(ffn) = &execution.layer(at, layer)?.ffn else { anyhow::bail!("Step-5 native routed join missing"); };
            self.step.compose(at, ffn, |ffn, meta| ffn.finish_on(meta.rows, meta.stream))?;
        }
        self.retain_diagnostic_output(layer)
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
            for rank in 0..self.0.ownership.ranks() {
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
                assert!(self.0.ownership.layer(layer, rank)?.routed_wave);
                if let Some(copy) = self.0.ownership.routed_replications()?[layer] {
                    assert_eq!(copy.source, self.0.ownership.execution().spec().entry_gpu);
                    self.0.record(copy.source as usize, "routed", copy.slot);
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
    fn routed_request_preserves_admitted_version_and_owned_row_metadata() -> Result<()> {
        let rows = vec![ExpertProtocolV2RowDescriptor { row_id: 0,
            source_kind: ExpertV2SourceKind::Decode, source_request_id: 7,
            token_position: 1234, route_offset: 0, route_count: 8 }];
        let routes = (0..8).map(|expert| ExpertProtocolV2RouteEntry {
            row_index: 0, expert_id: expert, gate_weight: 0.125 }).collect::<Vec<_>>();
        let wire = bytes::Bytes::from(vec![0; H * 2]);
        let request = expert_request(39, 812, 88, rows.clone(), routes.clone(), wire.clone())?;
        assert_eq!((request.header.request_id, request.header.placement_version, request.header.layer_id), (39, 812, 88));
        assert_eq!(request.rows, rows);
        assert_ne!(request.header.flags & EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, 0);
        for (id, version, layer) in [(0, 812, 88), (39, 0, 88), (39, 812, 2), (39, 812, 91)] {
            assert!(expert_request(id, version, layer, rows.clone(), routes.clone(), wire.clone()).is_err());
        }
        Ok(())
    }

    #[test]
    fn eager_execution_never_grows_the_capture_count() {
        check_captures(19, 19).unwrap();
        assert!(check_captures(19, 20).is_err());
    }

    #[test]
    fn request_identities_are_checked_and_never_zero() -> Result<()> {
        for ordinal in 0..1024 { assert_eq!(request_identity(ordinal)?, ordinal + 1); }
        assert_eq!(request_identity(u64::MAX - 1)?, u64::MAX);
        assert!(request_identity(u64::MAX).is_err());
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
    fn single_rank_target_uses_common_pipeline() -> Result<()> {
        use cuteafd_loader::placement::*;
        let cfg = config();
        let modes = vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }; cfg.layers];
        let spec = HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: (H * 4) as u64,
            rows: 64, lanes: 1, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::AttentionOwner };
        let placement = Placement { pool_tokens: 0, pool_limits: PoolLimits(vec![]),
            attention_placement: AttentionPlacement::Layers, attention_by_kind: vec![], peer_row_bytes: 0,
            onboard_layers: 0, layers: modes.iter().map(|&mode| LayerAssignment { mode, experts: ExpertHome::Spark }).collect(),
            movables: vec![], expert_ranges: vec![], residual: residual_homes(&modes, &spec)?,
            hops: plan_hops(&modes, &spec)?, tp2: None, items: vec![] };
        let ownership = ExecutionOwnership::new(&cfg, 1, LayerExecutionPlan::new(&placement, spec)?)?;
        let trace = Trace { ownership, host: RefCell::new(HostSchedule::default()),
            heads: Cell::new(0), waves: Cell::new(0) };
        let local = Exchange::Local(Cell::new(TerminalState::Active));
        for order in [LaneOrder::Independent, LaneOrder::OrderedChunks] {
            pair::submit_scoped(|| local.require_live(), || Ok(()), |scope|
                crate::shared::peer_split::attention::layer_pipeline_selected_instances(scope, trace.ownership.execution(), &[&local],
                    &[0], order, &[64], &mut [Recorder(&trace)]))?;
        }
        assert_eq!(trace.heads.get(), 0);
        assert_eq!(trace.waves.get(), 176);
        assert!(trace.ownership.execution().hops().is_empty());
        check_host(&trace.host.borrow()).map_err(anyhow::Error::msg)?;
        Ok(())
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
    fn native_workspace_is_charged_once() -> Result<()> {
        let plan = RankAdmission { immutable: 1_000_000, workspace: 123_456 };
        assert_eq!(plan.check(1_123_456, &[])?, 1_123_456);
        assert!(plan.check(1_123_455, &[]).is_err());
        Ok(())
    }

    #[test]
    fn final_head_weights_and_private_blas_workspace_are_included() -> Result<()> {
        let head = super::super::head::tests::admission_plan(64);
        let base = RankAdmission { immutable: 1000, workspace: 2000 };
        let plan = base.with_head(&head)?;
        assert_eq!(plan.bytes()?, base.immutable + head.immutable_bytes() + head.workspace_bytes());
        assert!(plan.check(plan.bytes()? - 1, &[]).is_err());
        assert!(RankAdmission { immutable: u64::MAX, workspace: 0 }.with_head(&head).is_err());
        assert_eq!(RankAdmission { immutable: 0, workspace: u64::MAX }.with_head(&head)?.workspace, u64::MAX);
        Ok(())
    }

    #[test]
    fn all_ranks_are_admitted_before_the_allocation_callback() -> Result<()> {
        let admission = [RankAdmission { immutable: 100, workspace: 20 },
            RankAdmission { immutable: 200, workspace: 30 }];
        let calls = std::cell::Cell::new(0);
        let allocate = || { calls.set(calls.get() + 1); Ok(()) };
        // Rank zero fits, but rank one must fail without loading either rank.
        assert!(admit_native(&admission, &[120, 229], &[0, 0], allocate).is_err());
        assert_eq!(calls.get(), 0);
        assert!(admit_native(&admission, &[120, 230], &[0, 1], allocate).is_err());
        assert_eq!(calls.get(), 0);
        admit_native(&admission, &[127, 239], &[7, 9], allocate)?;
        assert_eq!(calls.get(), 1);
        Ok(())
    }

    #[test]
    fn rank_inventory_rejects_missing_and_fake_ranks_before_allocation() -> Result<()> {
        for ranks in 1..=2 {
            validate_rank_inventory(ranks, &[ranks, ranks, ranks])?;
            for length in 0..=3 {
                if length != ranks { assert!(validate_rank_inventory(ranks, &[ranks, length]).is_err()); }
            }
        }
        assert!(validate_rank_inventory(0, &[]).is_err());
        assert!(validate_rank_inventory(3, &[3]).is_err());
        let admission = [RankAdmission { immutable: 100, workspace: 20 }];
        let calls = Cell::new(0);
        let allocate = || { calls.set(calls.get() + 1); Ok(()) };
        assert!(admit_native(&admission, &[119], &[0], allocate).is_err());
        assert!(admit_native(&admission, &[120, 120], &[0], allocate).is_err());
        assert_eq!(calls.get(), 0);
        admit_native(&admission, &[120], &[0], allocate)?;
        assert_eq!(calls.get(), 1);
        Ok(())
    }

    #[test]
    fn local_protocol_rejects_peer_operations_and_terminal_reuse() -> Result<()> {
        let local = Exchange::Local(Cell::new(TerminalState::Active));
        local.require_live()?;
        assert!(local.peer().is_err());
        assert!(local.push_entry(0, std::ptr::null(), std::ptr::null_mut(), 1).is_err());
        assert!(local.wait_entry(0).is_err());
        assert!(local.send(0, 1, std::ptr::null(), 1).is_err());
        local.publish_abort()?;
        assert_eq!(local.terminal_state(), TerminalState::Aborting);
        assert!(local.require_live().is_err());
        local.finish_terminal(true);
        assert_eq!(local.terminal_state(), TerminalState::Drained);
        local.publish_abort()?;
        assert!(local.require_live().is_err());
        let failed = Exchange::Local(Cell::new(TerminalState::Active));
        failed.publish_abort()?;
        failed.finish_terminal(false);
        assert_eq!(failed.terminal_state(), TerminalState::Retained);
        assert!(failed.require_live().is_err());
        assert!(failed.publish_abort().is_err());
        Ok(())
    }

    #[test]
    fn native_admission_refuses_underbudget_and_overflow() {
        let plan = RankAdmission { immutable: 100, workspace: 20 };
        assert!(plan.check(119, &[]).is_err());
        assert!(plan.check(120, &[1]).is_err());
        assert!(plan.check(u64::MAX, &[u64::MAX]).is_err());
        assert!(RankAdmission { immutable: u64::MAX, workspace: 1 }.bytes().is_err());
    }
}
