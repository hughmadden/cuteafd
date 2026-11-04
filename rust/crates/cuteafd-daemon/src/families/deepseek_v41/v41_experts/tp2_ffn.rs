//! One serving lane's TP2 encoder FFN, including peer inputs and final addition.
use crate::families::deepseek_v41::v41_backbone_shared::tp2::{
    Wave as SharedWave, Weights as SharedWeights,
};
use crate::shared::experts::rtx::{
    native::NativeTp2,
    pair::{OwnerInputs, OwnerPair, OwnerPolicy, OwnerWeights},
};
use crate::shared::memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, Bf16Add};
use std::rc::Rc;

/// Weight-only budgets have already been admitted separately from every lane's
/// workspace by the family placement request. Shared loading rechecks peak usage.
pub(crate) fn load_routed_weights<'a>(
    devices: [Device<'a>; 2],
    catalog: &cuteafd_loader::OfficialV41Catalog,
    layers: usize,
    capacity: u32,
    budgets: [usize; 2],
) -> Result<[OwnerWeights<'a>; 2]> {
    let nvfp4 = catalog.nvfp4().is_some();
    let OwnerPolicy::Native { selection, output, .. } =
        devices[0].run(|| OwnerPolicy::native(devices[0].library, nvfp4))? else { unreachable!() };
    let workspace = NativeTp2::workspace_bytes_for_format(
        devices[0].library,
        capacity as usize,
        selection,
        output,
        nvfp4,
    )?;
    let budgets = budgets
        .map(|budget| {
            budget
                .checked_add(workspace)
                .ok_or_else(|| anyhow::anyhow!("TP2 native load budget overflow"))
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("two TP2 budgets"))?;
    Ok(NativeTp2::load_pair_weights(
        devices,
        catalog,
        (0..layers).collect(),
        capacity as usize,
        budgets,
        selection,
        output,
    )?
    .map(OwnerWeights::Native))
}

/// [`load_routed_weights`] over native halves a push group delivered.
pub(crate) fn pushed_routed_weights<'a>(
    devices: [Device<'a>; 2],
    catalog: &cuteafd_loader::OfficialV41Catalog,
    layers: usize,
    capacity: u32,
    budgets: [usize; 2],
    pushed: [Vec<crate::shared::experts::layer::ExpertWeights<'a>>; 2],
) -> Result<[OwnerWeights<'a>; 2]> {
    ensure!(catalog.exl3().is_none() && catalog.nvfp4().is_none(), "pushed coordinator halves are native MXFP4");
    let OwnerPolicy::Native { selection, output, .. } =
        devices[0].run(|| OwnerPolicy::native(devices[0].library, false))? else { unreachable!() };
    let workspace = NativeTp2::workspace_bytes_for_format(devices[0].library, capacity as usize, selection, output, false)?;
    let budgets = [budgets[0].checked_add(workspace).context("TP2 native budget overflow")?,
        budgets[1].checked_add(workspace).context("TP2 native budget overflow")?];
    Ok(NativeTp2::pushed_pair_weights(devices, catalog, (0..layers).collect(), capacity as usize, budgets,
        selection, output, pushed)?.map(OwnerWeights::Native))
}

struct PeerInputs<'a> {
    values: Allocation<'a>,
    wire: Allocation<'a>,
    ids: Allocation<'a>,
    routing: Allocation<'a>,
}
impl<'a> PeerInputs<'a> {
    fn allocation_sizes(rows: u32) -> [usize; 4] { [10240,5280,24,24].map(|width| width*rows as usize) }
    fn new(device: Device<'a>, capacity: u32) -> Result<Self> {
        let [values, wire, ids, routing] = Self::allocation_sizes(capacity);
        Ok(Self {
            values: Allocation::new(device, values)?,
            wire: Allocation::new(device, wire)?,
            ids: Allocation::new(device, ids)?,
            routing: Allocation::new(device, routing)?,
        })
    }
    fn buffers(&self) -> [CuteafdDeviceBuffer; 4] {
        [
            self.values.buffer,
            self.wire.buffer,
            self.ids.buffer,
            self.routing.buffer,
        ]
    }
}

/// One real TP2 FFN execution arena per GPU, shared by all prefill slots.
/// Peer values/wire/ids/routing and final sums are reserve planes, since later
/// hooks may read them. OwnerPair rank GEMM/count/offset/reduction arenas are
/// unit scratch guarded through the final add (including both peer readers).
/// Shared TP2 final sums are reserve planes with their existing 1E guards.
/// Submission bookkeeping is mutex-owned, then protected by the GPU release
/// frontier after the host submit completes; host phase/window stays per slot.
/// One arena does not remove measured cross-unit compute overlap: prof19 k2
/// had 0/680 overlapping route-to-output windows per GPU (3113/2245 ms busy);
/// prof20 k3 had 0/475 (2240/1918 ms). Kernel-only overlap was also exactly zero.
/// Both traces used private per-slot OwnerPairs. Keep one unit arena, not k.
#[cfg(test)]
thread_local! { static SKIP_FFN_ACQUIRE: std::cell::Cell<bool> = const {std::cell::Cell::new(false)}; }
pub(crate) struct Storage<'a> {
    execution: tokio::sync::Mutex<Execution<'a>>,
    streams: Rc<[Stream<'a>; 2]>,
    rows: u32,
    unit: u32,
    layers: usize,
    shared_layers: usize,
    /// The GPU expert tier's set once installed ([`Wave::install_tier`]).
    tier: std::cell::OnceCell<std::sync::Arc<super::tier::TierResidency>>,
}
impl<'a> Storage<'a> {
    pub(crate) fn storage_bytes(library: &cuteafd_ffi::NativeLibrary, rows: u32, unit: u32, nvfp4: bool) -> Result<usize> {
        ensure!(unit > 0 && unit <= 4096 && unit <= rows
            && rows as usize <= crate::shared::memory::row_window::MAX_RESERVE_ROWS, "invalid TP2 FFN storage geometry");
        Ok(OwnerPair::device_bytes_for(library, unit, cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
                OwnerPolicy::native(library, nvfp4)?)?
            + crate::families::deepseek_v41::v41_backbone_shared::tp2::Storage::storage_bytes(library, rows, unit)?
            + PeerInputs::allocation_sizes(rows).into_iter().sum::<usize>() + rows as usize * 10240)
    }
    pub(crate) fn new(routed: [OwnerWeights<'a>; 2], shared: [Rc<Vec<SharedWeights<'a>>>; 2],
        layers: usize, rows: u32, unit: u32,
        modules: [Option<Vec<crate::shared::experts::exl3::execution::Exl3Modules<'a>>>; 2]) -> Result<Rc<Self>> {
        ensure!(unit > 0 && unit <= rows && rows as usize <= crate::shared::memory::row_window::MAX_RESERVE_ROWS,
            "invalid TP2 FFN reserve/unit");
        let shared_layers = shared[0].len().min(shared[1].len());
        let execution = Execution::new(routed, shared, layers, unit, rows, modules)?;
        let streams = execution.streams.clone();
        Ok(Rc::new(Self { execution: tokio::sync::Mutex::new(execution), streams, rows, unit, layers, shared_layers,
            tier: std::cell::OnceCell::new() }))
    }
}
pub(crate) struct Wave<'a> {
    storage: Rc<Storage<'a>>,
    window: crate::shared::memory::row_window::RowWindow,
}
impl<'a> Wave<'a> {
    pub(crate) fn device_bytes(library: &cuteafd_ffi::NativeLibrary, capacity: u32) -> Result<usize> {
        Storage::storage_bytes(library, capacity, capacity, false)
    }
    pub(crate) fn nvfp4_device_bytes(library: &cuteafd_ffi::NativeLibrary, capacity: u32) -> Result<usize> {
        Storage::storage_bytes(library, capacity, capacity, true)
    }
    pub(crate) fn new(routed: [OwnerWeights<'a>; 2], shared: [Rc<Vec<SharedWeights<'a>>>; 2],
        layers: usize, capacity: u32,
        modules: [Option<Vec<crate::shared::experts::exl3::execution::Exl3Modules<'a>>>; 2]) -> Result<Self> {
        Ok(Self::on_storage(Storage::new(routed, shared, layers, capacity, capacity, modules)?))
    }
    pub(crate) fn on_storage(storage: Rc<Storage<'a>>) -> Self {
        Self { storage, window: crate::shared::memory::row_window::RowWindow::NONE }
    }
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.end <= self.storage.rows as usize
                && range.len() <= self.storage.unit as usize, "TP2 FFN window exceeds reserve/unit");
        }
        self.window = window; Ok(())
    }
    pub(crate) fn contains(&self, layer: usize) -> bool { layer < self.storage.layers }
    /// Runs the GPU expert tier on this wave's Spark layers: a lane of its
    /// own over `halves` (shared weights) in this wave's execution arena. For
    /// a decode lane's private wave; at rest, once.
    #[allow(dead_code)] // called by the serving start-up of decode lanes (not wired yet)
    pub(crate) fn install_tier(&self, halves: &super::tier::TierHalves<'a>) -> Result<()> {
        ensure!(self.window.range().is_none(), "only a decode lane's private TP2 FFN runs the GPU expert tier");
        let mut execution = self.storage.execution.try_lock().context("TP2 FFN submit still borrowed")?;
        execution.install_tier(halves)?;
        self.storage.tier.set(halves.residency.clone())
            .map_err(|_| anyhow::anyhow!("the GPU expert tier is already installed on this TP2 FFN"))
    }
    /// The GPU expert tier's set, when this wave runs one.
    pub(crate) fn tier(&self) -> Option<&std::sync::Arc<super::tier::TierResidency>> { self.storage.tier.get() }
    /// Whether this wave's tier holds experts of `layer`.
    pub(crate) fn tier_holds(&self, layer: usize) -> bool {
        self.tier().is_some_and(|tier| !tier.ids(layer).is_empty())
    }
    /// Each GPU's tier slots for the router ids `probe` in `layer` (the
    /// startup check, at rest).
    #[allow(dead_code)] // called by the serving start-up of decode lanes (not wired yet)
    pub(crate) fn probe_tier_slots(&self, layer: usize, probe: &[u32]) -> Result<[Vec<i32>; 2]> {
        let execution = self.storage.execution.try_lock().context("TP2 FFN submit still borrowed")?;
        execution.tier.as_ref().context("this TP2 FFN runs no GPU expert tier")?.probe_slots(layer, probe)
    }
    #[cfg(test)]
    pub(crate) fn window(&self) -> &crate::shared::memory::row_window::RowWindow { &self.window }
    pub(crate) fn contains_shared(&self, layer: usize) -> bool { layer < self.storage.shared_layers }
    pub(crate) fn stream(&self, device: usize) -> &Stream<'a> { &self.storage.streams[device] }
    pub(crate) fn release_scratch(&self) -> Result<()> {
        self.storage.execution.try_lock().context("TP2 FFN submit still borrowed")?.released.release()
    }
    pub(crate) fn release_shared_output(&self, destination: usize) -> Result<()> {
        self.storage.execution.try_lock().context("TP2 FFN submit still borrowed")?.release_shared_output(destination)
    }
    /// After the Spark reduce that read the GPU expert tier's result.
    pub(crate) fn release_tier_scratch(&self) -> Result<()> {
        self.storage.execution.try_lock().context("TP2 FFN submit still borrowed")?.release_tier_scratch()
    }
    /// Device-ordered passes of this decode lane push every transfer between
    /// the GPUs (`v41_peer_push`): routed partials, shared partials and the
    /// returned result. `rows` is the decode lane's width.
    pub(crate) fn enable_peer_push(&self, rows: usize) -> Result<()> {
        ensure!(self.window.range().is_none(), "only a decode lane's private TP2 FFN pushes");
        let mut execution = self.storage.execution.try_lock().context("TP2 FFN submit still borrowed")?;
        execution.enable_push(rows)
    }
    async fn execution(&self, rows: u32) -> Result<tokio::sync::MutexGuard<'_, Execution<'a>>> {
        ensure!(self.window.admits(rows as usize), "TP2 FFN rows differ from slot window");
        let mut execution = self.storage.execution.lock().await;
        execution.window = self.window.clone(); execution.shared.select_window(self.window.clone())?;
        Ok(execution)
    }
    /// # Safety
    /// Inputs are complete and retained until the enclosing chain drains.
    pub(crate) async unsafe fn execute_shared(&mut self, layer: usize, rows: u32, values: CuteafdDeviceBuffer) -> Result<CuteafdDeviceBuffer> {
        unsafe { self.execution(rows).await?.execute_shared(layer, rows, values).await }
    }
    /// # Safety
    /// Same input/chain lifetime contract as execute_shared.
    pub(crate) async unsafe fn execute_shared_on(&mut self, layer: usize, rows: u32, values: CuteafdDeviceBuffer, destination: usize) -> Result<CuteafdDeviceBuffer> {
        unsafe { self.execution(rows).await?.execute_shared_on(layer, rows, values, destination).await }
    }
    /// A Spark layer's shared TP2 contribution and the GPU expert tier's
    /// result for `layer` (both BF16 on `destination`, the Spark reduce's GPU).
    /// # Safety
    /// Inputs are complete and retained until the enclosing chain drains; the
    /// tier result stays unchanged until [`Self::release_tier_scratch`] after its reduce.
    pub(crate) async unsafe fn execute_shared_and_tier_on(&mut self, layer: usize, rows: u32, values: CuteafdDeviceBuffer,
        wire: CuteafdDeviceBuffer, ids: CuteafdDeviceBuffer, routing: CuteafdDeviceBuffer, destination: usize)
        -> Result<(CuteafdDeviceBuffer, CuteafdDeviceBuffer)> {
        unsafe { self.execution(rows).await?.execute_shared_and_tier_on(layer, rows, values, wire, ids, routing, destination)
            .await }
    }
    /// # Safety
    /// Source stays immutable and retained until the enclosing chain drains.
    pub(crate) async unsafe fn return_result(&mut self, source: CuteafdDeviceBuffer, destination: usize, rows: u32) -> Result<CuteafdDeviceBuffer> {
        unsafe { self.execution(rows).await?.return_result(source, destination, rows).await }
    }
    /// # Safety
    /// Inputs stay immutable and retained until the enclosing chain drains.
    pub(crate) async unsafe fn execute(&mut self, layer: usize, rows: u32, values: CuteafdDeviceBuffer,
        wire: CuteafdDeviceBuffer, ids: CuteafdDeviceBuffer, routing: CuteafdDeviceBuffer) -> Result<CuteafdDeviceBuffer> {
        unsafe { self.execution(rows).await?.execute(layer, rows, values, wire, ids, routing).await }
    }
}
struct Execution<'a> {
    // Drain uploads and final addition before any referenced allocation drops.
    streams: Rc<[Stream<'a>; 2]>,
    peers: [PeerInputs<'a>; 2],
    shared: SharedWave<'a>,
    routed: OwnerPair<'a>,
    output: [Allocation<'a>; 2],
    add: Bf16Add<'a>,
    /// SM peer copies per destination GPU for device-ordered passes.
    sm: [Option<cuteafd_ffi::V41PeerCopy<'a>>; 2],
    layers: usize,
    capacity: u32,
    window: crate::shared::memory::row_window::RowWindow,
    released: crate::shared::memory::row_window::ScratchGuard<'a>,
    /// Device flags of the pushed transfers ([`Self::enable_push`]): rank =
    /// GPU, flag [`RETURN`] carries the reduced result back (A7).
    flags: Option<crate::shared::peer_split::PeerExchange<'a>>,
    /// The GPU expert tier on this lane's Spark layers ([`Self::install_tier`]).
    tier: Option<super::tier::TierLane<'a>>,
    /// The decode width pushes were enabled for (the tier's push follows it).
    push_rows: Option<usize>,
}
/// Flag of the result a Spark layer's reduce GPU returns to its block's GPU.
const RETURN: usize = 0;
/// Flag of the FFN input planes the attention GPU pushes to the other GPU
/// (A2 on TP2 layers, A5 on Spark layers), one sequence per direction.
const INPUT: usize = 1;
/// Flags of a decode lane's TP2 FFN exchange.
const FLAGS: usize = 2;

/// Bytes of a flat plane pushed in whole 16-byte units: route ids and
/// weights are 24 bytes a row. The rounding stays inside both ends: router
/// planes are allocated in whole units, and a pushing wave's peer inputs
/// hold an even number of rows (checked when pushes are enabled).
fn padded(bytes: usize) -> usize { bytes.next_multiple_of(16) }
/// A pushing wave's capacity: even, so its padded route planes fit (see
/// [`padded`]), and at least the decode rows it pushes.
fn push_capacity(capacity: u32, rows: usize) -> Result<()> {
    ensure!(capacity % 2 == 0 && rows <= capacity as usize,
        "TP2 FFN pushes need an even decode capacity of at least {rows} rows, got {capacity}");
    Ok(())
}
impl Drop for Execution<'_> {
    fn drop(&mut self) {
        for stream in self.streams.iter() { stream.drain_or_abort("TP2 FFN shared storage"); }
    }
}
/// Copies `bytes` from `source` (the other GPU) into `destination` on
/// `stream`: an SM copy ordered by events under a device-ordered pass (no
/// copy-engine queue held behind an event), else a copy-engine peer copy.
unsafe fn peer_copy(
    stream: &Stream<'_>,
    sm: Option<&cuteafd_ffi::V41PeerCopy<'_>>,
    destination: CuteafdDeviceBuffer,
    source: CuteafdDeviceBuffer,
    bytes: usize,
) -> Result<()> {
    // Device-ordered passes copy decode-width transfers on the SMs and
    // prefill-width ones on the copy engines (`device::sm_peer_copy_for`).
    // The issuing stream is this wave's own, ordered by a
    // `cudaStreamWaitEvent` on the chain head, never a memory wait.
    match sm.filter(|_| crate::shared::memory::device::sm_peer_copy_for(crate::shared::memory::chain::deferred(), bytes, crate::shared::memory::device::DECODE_PEER_COPY_BYTES)) {
        Some(sm) => unsafe { sm.launch(destination, source, bytes, stream.raw) },
        None => unsafe {
            stream
                .device
                .library
                .copy_peer_async(destination, source, bytes, stream.raw)
        },
    }
}
/// Whether this pass pushes, so the arena's scratch guard is neither taken
/// nor released (a decode lane's private arena; see `push_inputs`).
fn pushing_guards(flags: &Option<crate::shared::peer_split::PeerExchange<'_>>) -> bool {
    flags.is_some() && crate::shared::memory::chain::deferred()
}

/// A2/A5's push: `planes` (source on `local`, final at the chain head;
/// destination on the other GPU; bytes) as flat planes on `local`'s TP2
/// stream after the head, then the other GPU's TP2 stream waits on the
/// [`INPUT`] flag. Returns the head mark for a receiver on the other GPU that
/// adopts the head. Push before wait, in one synchronous block.
///
/// No slot or acknowledgement: the next layer's push into the same peer
/// inputs follows this layer's readers, because each layer's FFN output
/// gates the next layer's input (layers do not pipeline;
/// `v41_peer_push::schedule_tests`).
///
/// # Safety
/// Inside a device-ordered chain scope; every plane stays live and unaliased
/// until the chain drains.
unsafe fn push_inputs(flags: &crate::shared::peer_split::PeerExchange<'_>, streams: &[Stream<'_>; 2], local: usize,
    planes: &[(CuteafdDeviceBuffer, CuteafdDeviceBuffer, usize)]) -> Result<Option<crate::shared::memory::chain::HeadMark>> {
    use crate::shared::memory::chain;
    let remote = 1 - local;
    let planes = planes.iter().map(|&(source, destination, bytes)| {
        ensure!(source.device_id == streams[local].device.id && destination.device_id == streams[remote].device.id
            && bytes % 16 == 0 && bytes <= destination.bytes, "TP2 FFN input plane devices or extent differ");
        Ok(cuteafd_ffi::peer_exchange::PeerPlane { destination: destination.ptr, source: source.ptr.cast_const(),
            rows: 1, row_bytes: bytes as u64, destination_pitch: bytes as u64, source_pitch: bytes as u64 })
    }).collect::<Result<Vec<_>>>()?;
    let (sender, receiver) = (&streams[local], &streams[remote]);
    let mark = sender.device.run(|| unsafe { chain::join_for_push(sender.device.library, sender.raw) })?;
    flags.push_planes_on(local, sender.raw, INPUT, &planes)?;
    flags.wait_on(remote, receiver.raw, INPUT)?;
    Ok(mark)
}

/// A7's push: `source` (final at the chain head, on the other GPU) into
/// `output` on `destination`, ordered on the device by the exchange's
/// [`RETURN`] flag. Push before wait, in one synchronous block.
///
/// # Safety
/// Inside a device-ordered chain scope; both buffers stay live and unaliased
/// until the chain drains.
unsafe fn return_pushed(flags: &crate::shared::peer_split::PeerExchange<'_>, streams: &[Stream<'_>; 2],
    source: CuteafdDeviceBuffer, output: CuteafdDeviceBuffer, destination: usize) -> Result<()> {
    use crate::shared::memory::chain;
    let from = 1 - destination;
    ensure!(source.device_id == streams[from].device.id && output.device_id == streams[destination].device.id
        && source.bytes >= output.bytes, "TP2 result return devices or extent differ");
    let (sender, receiver) = (&streams[from], &streams[destination]);
    let mark = sender.device.run(|| unsafe { chain::join_for_push(sender.device.library, sender.raw) })?;
    flags.push_to_on(from, sender.raw, RETURN, source.ptr, output.ptr, output.bytes)?;
    receiver.device.run(|| unsafe {
        flags.wait_on(destination, receiver.raw, RETURN)?;
        chain::finish_after_flag(receiver.device.library, receiver.raw, mark)
    })
}

impl<'a> Execution<'a> {
    pub fn new(
        routed: [OwnerWeights<'a>; 2],
        shared: [Rc<Vec<SharedWeights<'a>>>; 2],
        layers: usize,
        capacity: u32,
        reserve: u32,
        modules: [Option<Vec<crate::shared::experts::exl3::execution::Exl3Modules<'a>>>; 2],
    ) -> Result<Self> {
        let devices = [routed[0].device(), routed[1].device()];
        ensure!(
            devices[0].id == 0
                && devices[1].id == 1
                && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid TP2 FFN devices"
        );
        ensure!(
            (1..=40).contains(&layers) && (1..=4096).contains(&capacity),
            "invalid TP2 FFN layer count/capacity"
        );
        for rank in 0..2 {
            ensure!(
                routed[rank].layers().get(..layers).is_some_and(|ids| ids.iter().copied().eq(0..layers))
                    && shared[rank].len() >= layers
                    && shared[rank][0].device().id == devices[rank].id
                    && std::ptr::eq(shared[rank][0].device().library, devices[rank].library),
                "TP2 FFN weights do not cover the requested devices/layers"
            );
        }
        for rank in 0..2 {
            devices[rank].run(|| devices[rank].library.cuda_enable_peer(1 - rank as i32))?;
        }
        Ok(Self {
            streams: Rc::new([Stream::new(devices[0])?, Stream::new(devices[1])?]),
            peers: [
                PeerInputs::new(devices[0], reserve)?,
                PeerInputs::new(devices[1], reserve)?,
            ],
            shared: SharedWave::on_storage(crate::families::deepseek_v41::v41_backbone_shared::tp2::Storage::new(shared, reserve, capacity)?),
            // One resident load serves every arena: each pair is admitted at
            // its own capacity (prefill unit cap, or a decode lane's width).
            routed: OwnerPair::with_modules(
                [routed[0].with_max_rows(capacity as usize)?, routed[1].with_max_rows(capacity as usize)?],
                capacity,
                cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
                std::env::var_os("CUTEAFD_TP2_TIMING").is_some(),
                modules,
            )?,
            output: [
                Allocation::new(devices[0], reserve as usize * 10240)?,
                Allocation::new(devices[1], reserve as usize * 10240)?,
            ],
            add: devices[0].library.bf16_add()?,
            sm: [Some(devices[0].run(|| devices[0].library.v41_peer_copy())?),
                Some(devices[1].run(|| devices[1].library.v41_peer_copy())?)],
            layers,
            capacity, window: crate::shared::memory::row_window::RowWindow::NONE,
            released: crate::shared::memory::row_window::ScratchGuard::new(devices[0].library, &[0,1])?,
            flags: None,
            tier: None,
            push_rows: None,
        })
    }
    /// See [`Wave::install_tier`]. The tier's owner pair reduces its halves
    /// like the routed TP2 layers' (pushed once this lane pushes).
    #[allow(dead_code)] // called by the serving start-up of decode lanes (not wired yet)
    fn install_tier(&mut self, halves: &super::tier::TierHalves<'a>) -> Result<()> {
        use crate::families::deepseek_v41::v41_peer_push::lane_bytes;
        ensure!(self.tier.is_none(), "the GPU expert tier is already installed on this TP2 FFN");
        ensure!(halves.weights.iter().enumerate().all(|(rank, weights)| weights.device().id == self.streams[rank].device.id),
            "the GPU expert tier's halves live on other GPUs than this TP2 FFN");
        let mut tier = super::tier::TierLane::new(halves, self.capacity, std::env::var_os("CUTEAFD_TP2_TIMING").is_some())?;
        if let Some(rows) = self.push_rows { tier.enable_push(lane_bytes(rows, tier.partial_row_bytes())?)?; }
        self.tier = Some(tier);
        Ok(())
    }
    /// See [`Wave::enable_peer_push`].
    fn enable_push(&mut self, rows: usize) -> Result<()> {
        use crate::families::deepseek_v41::v41_peer_push::lane_bytes;
        push_capacity(self.capacity, rows)?;
        self.routed.enable_push(lane_bytes(rows, self.routed.partial_row_bytes())?)?;
        self.shared.enable_push(lane_bytes(rows, 10240)?)?;
        let library = self.streams[0].device.library;
        self.flags = Some(crate::shared::peer_split::PeerExchange::new_flags(library, [0, 1], FLAGS,
            lane_bytes(rows, PeerInputs::allocation_sizes(1).iter().sum::<usize>())?)?);
        if let Some(tier) = self.tier.as_mut() { tier.enable_push(lane_bytes(rows, tier.partial_row_bytes())?)?; }
        self.push_rows = Some(rows);
        Ok(())
    }
    pub fn contains(&self, layer: usize) -> bool {
        layer < self.layers
    }
    /// The lane's stream on GPU `device` (0 or 1).
    pub(crate) fn stream(&self, device: usize) -> &Stream<'a> {
        &self.streams[device]
    }

    pub fn contains_shared(&self, layer: usize) -> bool {
        self.shared.contains(layer)
    }

    /// Execute only the shared TP2 contribution while routed experts
    /// run on Sparks. Reuses the encoder FFN's lane-local input/shared storage.
    /// # Safety
    /// Completed normalized input remains immutable through completion or drain.
    pub async unsafe fn execute_shared(
        &mut self,
        layer: usize,
        rows: u32,
        values: CuteafdDeviceBuffer,
    ) -> Result<CuteafdDeviceBuffer> {
        unsafe {
            self.execute_shared_on(layer, rows, values, values.device_id as usize)
                .await
        }
    }

    /// Same input lifetime as `execute_shared`; select the reduction owner so
    /// Spark collection can remain on its transport GPU for encoder layers too.
    /// # Safety
    /// Input storage remains complete, immutable and live through completion or drain.
    pub async unsafe fn execute_shared_on(
        &mut self,
        layer: usize,
        rows: u32,
        values: CuteafdDeviceBuffer,
        destination: usize,
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            self.contains_shared(layer)
                && destination < 2
                && rows > 0
                && rows <= self.capacity
                && matches!(values.device_id, 0 | 1)
                && !values.ptr.is_null()
                && values.bytes >= rows as usize * 10240,
            "TP2 shared input layer/device/extent differs"
        );
        // Peer DMA is never queued behind an unresolved dependency: settle any
        // chained producer of `values` on the host first.
        if !crate::shared::memory::chain::deferred() {
            crate::shared::memory::chain::settle(self.streams[0].device.library)?;
        }
        let local = values.device_id as usize;
        let remote = 1 - local;
        let upload = &self.streams[remote];
        let mut guard = upload.cancellation_guard("TP2 shared input upload");
        let peer = self.window.apply(self.peers[remote].values.buffer, 10240);
        let pushing = self.flags.as_ref().filter(|_| crate::shared::memory::chain::deferred());
        // A5 below the boundary reduces on the other GPU: its receiver adopts
        // the head this push joined (no cross-device merge).
        let mut adopt = None;
        if let Some(flags) = pushing {
            let mark = unsafe { push_inputs(flags, &self.streams, local, &[(values, peer, rows as usize * 10240)])? };
            adopt = mark.filter(|_| destination != local);
        } else {
            if crate::shared::memory::chain::deferred() {
                // Both GPUs' streams follow the input's producers (the chain head).
                self.streams[0].join_chain()?;
                self.streams[1].join_chain()?;
            } else {
                upload.wait().await?;
            }
            let sm = self.sm[remote].as_ref();
            upload
                .device
                .run(|| unsafe { peer_copy(upload, sm, peer, values, rows as usize * 10240) })?;
        }
        let mut inputs = [values; 2];
        inputs[remote] = peer;
        let output = unsafe {
            self.shared
                .execute(
                    layer,
                    rows,
                    destination,
                    inputs,
                    [&self.streams[0], &self.streams[1]],
                    adopt,
                )
                .await?
        };
        guard.complete = true;
        Ok(output)
    }

    /// Return a completed remote-expert reduction to its block's GPU using
    /// lane-owned scratch. Stream::wait drains this copy on cancellation.
    /// # Safety
    /// Source storage remains immutable and live until the transfer completes or drains.
    pub async unsafe fn return_result(
        &mut self,
        source: CuteafdDeviceBuffer,
        destination: usize,
        rows: u32,
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            destination < 2
                && rows > 0
                && rows <= self.capacity
                && source.bytes >= rows as usize * 10240,
            "invalid TP2 result transfer"
        );
        let mut output = self.window.apply(self.output[destination].buffer, 10240);
        output.bytes = rows as usize * 10240;
        if let Some(flags) = self.flags.as_ref().filter(|_| crate::shared::memory::chain::deferred()) {
            // A7: the reduce's GPU pushes its result, the block's GPU waits on
            // the flag and adopts the chain head without a cross-device merge.
            // No ack: the next push into `output` follows the next layer's
            // reduce, which follows this layer's readers of it (layers do not
            // pipeline; `v41_peer_push::schedule_tests`).
            unsafe { return_pushed(flags, &self.streams, source, output, destination)? };
            return Ok(output);
        }
        if !crate::shared::memory::chain::deferred() {
            crate::shared::memory::chain::settle(self.streams[0].device.library)?;
        }
        let stream = &self.streams[destination];
        let sm = self.sm[destination].as_ref();
        let queued = stream.device.run(|| unsafe {
            if crate::shared::memory::chain::deferred() {
                crate::shared::memory::chain::join(stream.device.library, stream.raw)?;
            }
            peer_copy(stream, sm, output, source, output.bytes)
        });
        let drained = stream.complete().await;
        queued.and(drained)?;
        Ok(output)
    }

    /// # Safety
    /// Local inputs are complete, immutable, and retained through return or
    /// cancellation. They all represent the same normalized FFN rows. Each lane
    /// owns a distinct Wave. Returned storage is valid until its next execute.
    pub async unsafe fn execute(
        &mut self,
        layer: usize,
        rows: u32,
        values: CuteafdDeviceBuffer,
        wire: CuteafdDeviceBuffer,
        ids: CuteafdDeviceBuffer,
        routing: CuteafdDeviceBuffer,
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            self.contains(layer) && rows > 0 && rows <= self.capacity,
            "TP2 FFN layer/rows unavailable"
        );
        ensure!(
            matches!(values.device_id, 0 | 1),
            "invalid TP2 FFN input device"
        );
        let local = values.device_id as usize;
        let remote = 1 - local;
        let inputs = [values, wire, ids, routing];
        let widths = [10240, 5280, 24, 24];
        for (buffer, width) in inputs.into_iter().zip(widths) {
            ensure!(
                buffer.device_id == local as i32
                    && !buffer.ptr.is_null()
                    && buffer.bytes >= rows as usize * width,
                "TP2 FFN input extent/device mismatch"
            );
        }
        #[cfg(test)]
        let skip = SKIP_FFN_ACQUIRE.with(|s|s.get());
        #[cfg(not(test))]
        let skip = false;
        // A decode lane's pushing pass: the flags order every reuse of this
        // lane's private arena, so its scratch guard is not taken.
        let pushing = self.flags.as_ref().filter(|_| crate::shared::memory::chain::deferred());
        if !skip && pushing.is_none() {for stream in self.streams.iter() {
            stream.device.run(|| unsafe { self.released.acquire(stream.raw) })?;
        }}
        let upload = &self.streams[remote];
        let mut guard = upload.cancellation_guard("TP2 FFN input upload");
        let peer = self.peers[remote].buffers().into_iter().zip(widths)
            .map(|(buffer, width)| self.window.apply(buffer, width)).collect::<Vec<_>>();
        let peer: [CuteafdDeviceBuffer; 4] = peer.try_into().ok().unwrap();
        if let Some(flags) = pushing {
            // A2: the four planes in one push; ids and weights as flat
            // 16-byte-rounded planes.
            let planes: Vec<_> = inputs.into_iter().zip(peer).zip(widths)
                .map(|((source, destination), width)| (source, destination, padded(rows as usize * width))).collect();
            unsafe { push_inputs(flags, &self.streams, local, &planes)?; }
        } else {
            // Do not submit DMA behind an unresolved stream dependency: a blocked
            // copy packet can hold up independent lanes on the shared copy engine.
            // Chained producers of these inputs are settled on the host as well.
            if !crate::shared::memory::chain::deferred() {
                crate::shared::memory::chain::settle(upload.device.library)?;
            }
            if crate::shared::memory::chain::deferred() {
                self.streams[0].join_chain()?;
                self.streams[1].join_chain()?;
            } else {
                upload.wait().await?;
            }
            let sm = self.sm[remote].as_ref();
            upload.device.run(|| {
                for ((source, destination), width) in inputs.into_iter().zip(peer).zip(widths) {
                    unsafe {
                        peer_copy(upload, sm, destination, source, rows as usize * width)?;
                    }
                }
                Ok(())
            })?;
        }
        let mut ranks = [inputs; 2];
        ranks[remote] = peer;
        // NVFP4 consumes the BF16 values already broadcast for shared experts,
        // not the native family's FP8 wire payload.
        let routed_input = if self.routed.uses_bf16_input() { 0 } else { 1 };
        // Each backend records its own producer event after the queued copies;
        // local compute can begin before peer input transfers finish.
        let (routed, shared) = tokio::try_join!(
            unsafe {
                self.routed.execute(
                    layer,
                    rows,
                    local,
                    std::array::from_fn(|rank| OwnerInputs {
                        input: ranks[rank][routed_input],
                        ids: ranks[rank][2],
                        weights: ranks[rank][3],
                        producer: &self.streams[rank],
                    }),
                )
            },
            unsafe {
                self.shared.execute(
                    layer,
                    rows,
                    local,
                    [ranks[0][0], ranks[1][0]],
                    [&self.streams[0], &self.streams[1]],
                    None,
                )
            }
        )?;
        guard.complete = true;
        let stream = &self.streams[local];
        let output = self.window.apply(self.output[local].buffer, 10240);
        let queued = stream.device.run(|| unsafe {
            // Deferred: the routed and shared sums were merged into the chain head.
            if crate::shared::memory::chain::deferred() {
                crate::shared::memory::chain::join(stream.device.library, stream.raw)?;
            }
            self.add
                .launch(routed, shared, output, rows as usize * 5120, stream.raw)
        });
        let drained = stream.complete().await;
        queued.and(drained)?;
        // The add above read the shared output: release it for the next layer
        // (it is on the chain head now; no-op outside a chain).
        self.shared.release_output(local)?;
        if !pushing_guards(&self.flags) { self.released.release()?; }
        let mut result = output;
        result.bytes = rows as usize * 10240;
        Ok(result)
    }
    /// The Spark reduce that read the shared output on `destination` finished
    /// on the unit's chain: the next layer's shared add may overwrite it.
    pub(crate) fn release_shared_output(&self, destination: usize) -> Result<()> {
        self.shared.release_output(destination)
    }

    /// A Spark layer of a lane with the GPU expert tier: the shared TP2
    /// contribution and the tier's result for the experts the Spark request
    /// left to the coordinator, both BF16 on `destination` (the Spark
    /// reduce's GPU). The four input planes cross to the other GPU once (as a
    /// TP2 layer's do), each GPU turns the router's ids into its tier slots,
    /// and both halves run beside the shared halves.
    ///
    /// The tier's arenas (slots, its owner pair's partials and result) and
    /// the peer planes are this lane's scratch: under a device-ordered pass
    /// that does not push, they are guarded through the Spark reduce, which
    /// releases them ([`Wave::release_tier_scratch`]) once it read the result.
    /// # Safety
    /// As [`Self::execute`]: the router's outputs are complete, immutable and
    /// retained until the enclosing chain drains (or the cancellation drain).
    pub async unsafe fn execute_shared_and_tier_on(
        &mut self,
        layer: usize,
        rows: u32,
        values: CuteafdDeviceBuffer,
        wire: CuteafdDeviceBuffer,
        ids: CuteafdDeviceBuffer,
        routing: CuteafdDeviceBuffer,
        destination: usize,
    ) -> Result<(CuteafdDeviceBuffer, CuteafdDeviceBuffer)> {
        ensure!(self.contains_shared(layer) && destination < 2 && rows > 0 && rows <= self.capacity
            && self.tier.as_ref().is_some_and(|tier| tier.holds(layer)),
            "TP2 shared and GPU expert tier layer/rows/destination differ");
        ensure!(matches!(values.device_id, 0 | 1), "invalid TP2 FFN input device");
        let local = values.device_id as usize;
        let remote = 1 - local;
        let inputs = [values, wire, ids, routing];
        let widths = [10240, 5280, 24, 24];
        for (buffer, width) in inputs.into_iter().zip(widths) {
            ensure!(buffer.device_id == local as i32 && !buffer.ptr.is_null() && buffer.bytes >= rows as usize * width,
                "TP2 shared and tier input extent/device mismatch");
        }
        let pushing = self.flags.as_ref().filter(|_| crate::shared::memory::chain::deferred());
        if pushing.is_none() {
            for stream in self.streams.iter() {
                stream.device.run(|| unsafe { self.released.acquire(stream.raw) })?;
            }
        }
        let upload = &self.streams[remote];
        let mut guard = upload.cancellation_guard("TP2 shared and tier input upload");
        let peer = self.peers[remote].buffers().into_iter().zip(widths)
            .map(|(buffer, width)| self.window.apply(buffer, width)).collect::<Vec<_>>();
        let peer: [CuteafdDeviceBuffer; 4] = peer.try_into().ok().unwrap();
        // A5 below the boundary reduces on the other GPU: its receiver adopts
        // the head this push joined (as `execute_shared_on`).
        let mut adopt = None;
        if let Some(flags) = pushing {
            let planes: Vec<_> = inputs.into_iter().zip(peer).zip(widths)
                .map(|((source, destination), width)| (source, destination, padded(rows as usize * width))).collect();
            let mark = unsafe { push_inputs(flags, &self.streams, local, &planes)? };
            adopt = mark.filter(|_| destination != local);
        } else {
            if !crate::shared::memory::chain::deferred() {
                crate::shared::memory::chain::settle(upload.device.library)?;
            }
            if crate::shared::memory::chain::deferred() {
                self.streams[0].join_chain()?;
                self.streams[1].join_chain()?;
            } else {
                upload.wait().await?;
            }
            let sm = self.sm[remote].as_ref();
            upload.device.run(|| {
                for ((source, destination), width) in inputs.into_iter().zip(peer).zip(widths) {
                    unsafe { peer_copy(upload, sm, destination, source, rows as usize * width)?; }
                }
                Ok(())
            })?;
        }
        let mut ranks = [inputs; 2];
        ranks[remote] = peer;
        let tier = self.tier.as_mut().context("GPU expert tier missing")?;
        // Each GPU's slots follow its copy of the router's ids on its stream.
        let slots = [0, 1].map(|rank| unsafe { tier.slot_routes(rank, layer, rows, ranks[rank][2], &self.streams[rank]) });
        let [first, second] = slots;
        let slots = [first?, second?];
        let (shared, tiered) = tokio::try_join!(
            unsafe {
                self.shared.execute(layer, rows, destination, [ranks[0][0], ranks[1][0]],
                    [&self.streams[0], &self.streams[1]], adopt)
            },
            unsafe {
                tier.execute(layer, rows, destination, std::array::from_fn(|rank| OwnerInputs {
                    input: ranks[rank][1], ids: slots[rank], weights: ranks[rank][3], producer: &self.streams[rank],
                }))
            }
        )?;
        guard.complete = true;
        Ok((shared, tiered))
    }

    /// The Spark reduce that read the tier's result finished on the chain:
    /// this lane's tier and peer scratch may be rewritten by its next layer.
    pub(crate) fn release_tier_scratch(&self) -> Result<()> {
        if !pushing_guards(&self.flags) { self.released.release()?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// A2's padded route planes stay inside an even-capacity wave's peer
    /// inputs and inside the router's whole-unit planes; odd capacities and
    /// rows past the capacity are refused.
    #[test]
    fn padded_route_planes_fit_even_capacities() {
        assert_eq!([24, 48, 72, 1536].map(super::padded), [32, 48, 80, 1536]);
        for capacity in (2..=256u32).step_by(2) {
            super::push_capacity(capacity, capacity as usize).unwrap();
            for rows in 1..=capacity as usize {
                assert!(super::padded(rows * 24) <= capacity as usize * 24, "{rows} of {capacity}");
            }
        }
        assert!(super::push_capacity(63, 1).is_err());
        assert!(super::push_capacity(64, 65).is_err());
    }
    use super::*;
    #[test]
    #[ignore = "requires combined TP2 CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT, two GPUs"]
    fn real_tp2_ffn_broadcast_and_combine_independent_lanes() -> Result<()> {
        let lib =
            unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?),
        )?;
        lib.cuda_set_device(0)?;
        let devices = [
            Device {
                library: &lib,
                id: 0,
            },
            Device {
                library: &lib,
                id: 1,
            },
        ];
        let routed = load_routed_weights(
            devices,
            &catalog,
            1,
            16,
            [4_000_000_000; 2],
        )?;
        let shared: [Rc<Vec<SharedWeights<'_>>>; 2] = [0, 1]
            .map(|rank| {
                (0..40)
                    .map(|layer| {
                        SharedWeights::load(
                            devices[rank],
                            &catalog,
                            layer,
                            SharedWeights::load_peak_device_bytes(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()
                    .map(Rc::new)
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .ok()
            .expect("two shared ranks");
        let mut lanes = [
            Wave::new(routed.clone(), shared.clone(), 1, 16, [None, None])?,
            Wave::new(routed.clone(), shared.clone(), 1, 16, [None, None])?,
        ];
        let mut reference_routed = OwnerPair::new(
            routed,
            16,
            cuteafd_core::ExpertGeometry::DEEPSEEK_V41,
            std::env::var_os("CUTEAFD_TP2_TIMING").is_some(),
        )?;
        let mut reference_shared = SharedWave::new(shared, 16)?;
        let inputs = [
            PeerInputs::new(devices[0], 16)?,
            PeerInputs::new(devices[1], 16)?,
        ];
        let reference_inputs = [
            PeerInputs::new(devices[0], 16)?,
            PeerInputs::new(devices[1], 16)?,
        ];
        let producers = [Stream::new(devices[0])?, Stream::new(devices[1])?];
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        assert!(lanes[0].contains_shared(39));
        assert!(!lanes[0].contains_shared(40));
        assert!(!lanes[0].contains(20));
        for layer in [20, 39] {
            for rows in [1u32, 16, 1] {
                for local in 0..2 {
                    let host: Vec<u8> = (0..rows as usize * 5120)
                        .flat_map(|i| {
                            let value = ((i + layer + local) % 17) as f32 / 32.0 - 0.25;
                            ((value.to_bits() >> 16) as u16).to_ne_bytes()
                        })
                        .collect();
                    for rank in 0..2 {
                        devices[rank].run(|| {
                            devices[rank]
                                .library
                                .copy_h2d(reference_inputs[rank].values.buffer, &host)
                        })?;
                    }
                    devices[local].run(|| {
                        devices[local]
                            .library
                            .copy_h2d(inputs[local].values.buffer, &host)
                    })?;
                    let actual = runtime.block_on(unsafe {
                        lanes[local].execute_shared(layer, rows, inputs[local].values.buffer)
                    })?;
                    let expected = runtime.block_on(unsafe {
                        reference_shared.execute(
                            layer,
                            rows,
                            local,
                            [
                                reference_inputs[0].values.buffer,
                                reference_inputs[1].values.buffer,
                            ],
                            [&producers[0], &producers[1]],
                            None,
                        )
                    })?;
                    devices[local].run(|| {
                        let mut left = vec![0; actual.bytes];
                        let mut right = vec![0; expected.bytes];
                        devices[local].library.copy_d2h(&mut left, actual)?;
                        devices[local].library.copy_d2h(&mut right, expected)?;
                        assert_eq!(
                            left, right,
                            "decoder shared TP2 broadcast/reduction differs"
                        );
                        Ok(())
                    })?;
                    let returned = runtime.block_on(async {
                        let peer = unsafe {
                            lanes[local]
                                .execute_shared_on(
                                    layer,
                                    rows,
                                    inputs[local].values.buffer,
                                    1 - local,
                                )
                                .await?
                        };
                        unsafe { lanes[local].return_result(peer, local, rows).await }
                    })?;
                    devices[local].run(|| {
                        let mut left = vec![0; returned.bytes];
                        let mut right = vec![0; expected.bytes];
                        lib.copy_d2h(&mut left, returned)?;
                        lib.copy_d2h(&mut right, expected)?;
                        ensure!(
                            left == right,
                            "peer-owned shared reduction and return differs"
                        );
                        Ok(())
                    })?;
                    assert_eq!(lib.cuda_get_device()?, 0);
                }
            }
        }
        for (rows, bf16, fp8) in [
            (1, 0u16, 0u8),
            (16, 0x3f80, 0x38),
            (1, 0x3f00, 0x30),
            (16, 0, 0),
        ] {
            let frames: [[Vec<u8>; 4]; 2] = std::array::from_fn(|rank| {
                let negative = rank == 1 && bf16 != 0;
                let values: Vec<u8> = (bf16 ^ if negative { 0x8000 } else { 0 })
                    .to_ne_bytes()
                    .into_iter()
                    .cycle()
                    .take(16 * 10240)
                    .collect();
                let mut wire = vec![0; 16 * 5280];
                for row in wire.chunks_exact_mut(5280) {
                    row[..5120].fill(fp8 ^ if negative { 0x80 } else { 0 });
                    row[5120..].fill(127);
                }
                let ids: Vec<u8> = (0..96i32)
                    .flat_map(|i| (i % 6 + rank as i32 * 6).to_ne_bytes())
                    .collect();
                let routing: Vec<u8> = (0..96)
                    .flat_map(|i| {
                        ([0.1f32, 0.15, 0.2, 0.25, 0.1, 0.2][(i + rank) % 6]).to_ne_bytes()
                    })
                    .collect();
                [values, wire, ids, routing]
            });
            for rank in 0..2 {
                devices[rank].run(|| {
                    for (buffer, bytes) in inputs[rank].buffers().into_iter().zip(&frames[rank]) {
                        lib.copy_h2d(buffer, bytes)?;
                    }
                    Ok(())
                })?;
            }
            let [left, right] = &mut lanes;
            let a = inputs[0].buffers();
            let b = inputs[1].buffers();
            let (x, y) = runtime.block_on(async {
                tokio::join!(
                    unsafe { left.execute(0, rows, a[0], a[1], a[2], a[3]) },
                    unsafe { right.execute(0, rows, b[0], b[1], b[2], b[3]) }
                )
            });
            let mut actual = [
                vec![0; rows as usize * 10240],
                vec![0; rows as usize * 10240],
            ];
            for (rank, buffer) in [x?, y?].into_iter().enumerate() {
                devices[rank].run(|| lib.copy_d2h(&mut actual[rank], buffer))?;
            }
            assert_eq!(
                actual[0] != actual[1],
                bf16 != 0,
                "different lane inputs must remain independent"
            );
            for lane in 0..2 {
                for rank in 0..2 {
                    devices[rank].run(|| {
                        for (buffer, bytes) in reference_inputs[rank]
                            .buffers()
                            .into_iter()
                            .zip(&frames[lane])
                        {
                            lib.copy_h2d(buffer, bytes)?;
                        }
                        Ok(())
                    })?;
                }
                let route = runtime.block_on(unsafe {
                    reference_routed.execute(
                        0,
                        rows,
                        0,
                        std::array::from_fn(|rank| OwnerInputs {
                            input: reference_inputs[rank].wire.buffer,
                            ids: reference_inputs[rank].ids.buffer,
                            weights: reference_inputs[rank].routing.buffer,
                            producer: &producers[rank],
                        }),
                    )
                })?;
                let shared = runtime.block_on(unsafe {
                    reference_shared.execute(
                        0,
                        rows,
                        0,
                        [
                            reference_inputs[0].values.buffer,
                            reference_inputs[1].values.buffer,
                        ],
                        [&producers[0], &producers[1]],
                        None,
                    )
                })?;
                let mut route_bytes = vec![0; actual[0].len()];
                let mut shared_bytes = vec![0; actual[0].len()];
                lib.copy_d2h(&mut route_bytes, route)?;
                lib.copy_d2h(&mut shared_bytes, shared)?;
                for ((a, b), out) in route_bytes
                    .chunks_exact(2)
                    .zip(shared_bytes.chunks_exact(2))
                    .zip(actual[lane].chunks_exact(2))
                {
                    let decode =
                        |v: &[u8]| f32::from_bits((u16::from_ne_bytes([v[0], v[1]]) as u32) << 16);
                    let sum = decode(a) + decode(b);
                    assert!(sum.is_finite());
                    let bits = sum.to_bits();
                    let expected = ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
                    assert_eq!(u16::from_ne_bytes([out[0], out[1]]), expected);
                }
            }
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        for capacity in [1, 16, 80, 256, 1024, 4096] {
            eprintln!(
                "TP2 FFN capacity={capacity} workspace_bytes_per_gpu_per_lane={}",
                Wave::device_bytes(&lib, capacity)?
            );
        }
        // Deliberately hold one lane's upload while the other completes real
        // kernels. A bounded release avoids hanging even if a join regresses.
        use std::ffi::c_void;
        use std::{
            future::Future,
            sync::atomic::{AtomicBool, Ordering},
            task::{Context, Poll, Waker},
        };
        unsafe extern "C" fn hold(data: *mut c_void) {
            let release = unsafe { &*data.cast::<AtomicBool>() };
            while !release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        }
        struct CudaLibrary(*mut c_void);
        impl Drop for CudaLibrary {
            fn drop(&mut self) {
                unsafe {
                    libc::dlclose(self.0);
                }
            }
        }
        let cuda =
            CudaLibrary(unsafe { libc::dlopen(c"libcudart.so.13".as_ptr(), libc::RTLD_NOW) });
        ensure!(!cuda.0.is_null(), "CUDA runtime unavailable");
        let symbol = unsafe { libc::dlsym(cuda.0, c"cudaLaunchHostFunc".as_ptr()) };
        ensure!(!symbol.is_null(), "CUDA callback API unavailable");
        let launch: unsafe extern "C" fn(
            *mut c_void,
            unsafe extern "C" fn(*mut c_void),
            *mut c_void,
        ) -> i32 = unsafe { std::mem::transmute(symbol) };
        let release = AtomicBool::new(false);
        struct Release<'a>(&'a AtomicBool);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let _release = Release(&release);
        let [left, right] = &mut lanes;
        devices[1].run(|| {
            ensure!(
                unsafe {
                    launch(
                        left.storage.streams[1].raw,
                        hold,
                        (&release as *const AtomicBool).cast_mut().cast(),
                    )
                } == 0,
                "could not stall TP2 upload"
            );
            Ok(())
        })?;
        let a = inputs[0].buffers();
        let b = inputs[1].buffers();
        let _entered = runtime.enter();
        let mut blocked = Box::pin(unsafe { left.execute(0, 16, a[0], a[1], a[2], a[3]) });
        let _unwind_release = Release(&release);
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(blocked.as_mut().poll(&mut context), Poll::Pending));
        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(500));
                release.store(true, Ordering::Release);
            });
            for rank in 0..2 {
                devices[rank].run(|| unsafe {
                    lib.copy_peer_async(
                        right.storage.execution.try_lock()?.output[rank].buffer,
                        inputs[1 - rank].values.buffer,
                        10240,
                        right.storage.streams[rank].raw,
                    )
                })?;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            for rank in 0..2 {
                assert!(
                    devices[rank]
                        .run(|| unsafe { lib.cuda_stream_query(right.storage.streams[rank].raw) })?,
                    "unrelated peer copy stalled on GPU {rank}"
                );
            }
            let started = std::time::Instant::now();
            let mut other = Box::pin(unsafe { right.execute(0, 16, b[0], b[1], b[2], b[3]) });
            let initial = other.as_mut().poll(&mut context);
            eprintln!(
                "independent lane first poll {:?}, pending={}",
                started.elapsed(),
                initial.is_pending()
            );
            let result = match initial {
                Poll::Ready(result) => result,
                Poll::Pending => runtime.block_on(other),
            };
            eprintln!("independent lane completed {:?}", started.elapsed());
            let independent = !release.load(Ordering::Acquire);
            drop(blocked); // Cancellation must drain before releasing input owners.
            assert!(release.load(Ordering::Acquire));
            result?;
            assert!(
                independent,
                "one lane's stalled input upload held the other lane"
            );
            Ok(())
        })?;
        // Reuse the cancelled lane's same streams, events, and scratch.
        runtime.block_on(unsafe { left.execute(0, 16, a[0], a[1], a[2], a[3]) })?;
        release.store(false, Ordering::Release);
        devices[1].run(|| {
            ensure!(
                unsafe {
                    launch(
                        left.storage.streams[1].raw,
                        hold,
                        (&release as *const AtomicBool).cast_mut().cast(),
                    )
                } == 0,
                "stalling shared upload failed"
            );
            Ok(())
        })?;
        let mut blocked_shared = Box::pin(unsafe { left.execute_shared(20, 16, a[0]) });
        let shared_unwind_release = Release(&release);
        assert!(matches!(
            blocked_shared.as_mut().poll(&mut context),
            Poll::Pending
        ));
        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(500));
                release.store(true, Ordering::Release);
            });
            let result = runtime.block_on(unsafe { right.execute_shared(39, 1, b[0]) });
            let independent = !release.load(Ordering::Acquire);
            drop(blocked_shared);
            result?;
            assert!(independent, "decoder shared work joined the other lane");
            Ok(())
        })?;
        drop(shared_unwind_release);
        runtime.block_on(unsafe { left.execute_shared(20, 16, a[0]) })?;
        assert_eq!(lib.cuda_get_device()?, 0);
        Ok(())
    }
}

#[cfg(test)]
mod shared_storage_tests {
    use super::*;
    use crate::shared::memory::{chain::StageChain,row_window::RowWindow};
    fn run(guarded:bool)->Result<()> {
        let lib=unsafe {cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)?};lib.cuda_set_device(0)?;
        let catalog=cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let devices=[Device {library:&lib,id:0},Device {library:&lib,id:1}];
        let routed=load_routed_weights(devices,&catalog,1,16,[4_000_000_000;2])?;
        let shared=[Rc::new(vec![SharedWeights::load(devices[0],&catalog,0,SharedWeights::load_peak_device_bytes())?]),
            Rc::new(vec![SharedWeights::load(devices[1],&catalog,0,SharedWeights::load_peak_device_bytes())?])];
        let inputs=[PeerInputs::new(devices[0],16)?,PeerInputs::new(devices[1],16)?];
        for (seed,input) in inputs.iter().enumerate() {
            let value=if seed==0 {0x3e80u16}else {0xbe80u16};
            let values=value.to_ne_bytes().repeat(16*5120);
            let mut wire=vec![0;16*5280];for row in wire.chunks_exact_mut(5280) {
                row[..5120].fill(if seed==0 {0x28}else {0xa8});row[5120..].fill(127);
            }
            let ids:Vec<u8>=(0..96u32).flat_map(|i|(i%6).to_ne_bytes()).collect();
            let routing=(1.0f32/6.0).to_ne_bytes().repeat(96);
            input.values.device.run(|| {
                for (buffer,host) in input.buffers().into_iter().zip([&values,&wire,&ids,&routing]) {lib.copy_h2d(buffer,host)?;}
                Ok(())
            })?;
        }
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        let mut private=Wave::new(routed.clone(),shared.clone(),1,16,[None,None])?;
        let a=inputs[0].buffers();
        let private_a=runtime.block_on(unsafe {private.execute(0,16,a[0],a[1],a[2],a[3])})?;
        let mut want_a=vec![0;16*10240];devices[0].run(||lib.copy_d2h(&mut want_a,private_a))?;
        let private_rank=private.storage.execution.try_lock().unwrap().routed.rank_output(0);
        let mut want=vec![0;private_rank.bytes];lib.copy_d2h(&mut want,private_rank)?;
        let b=inputs[1].buffers();
        let private_b=runtime.block_on(unsafe {private.execute(0,16,b[0],b[1],b[2],b[3])})?;
        let mut want_b=vec![0;16*10240];devices[1].run(||lib.copy_d2h(&mut want_b,private_b))?;
        let storage=Storage::new(routed,shared,1,48,16,[None,None])?;
        let mut slots=[Wave::on_storage(storage.clone()),Wave::on_storage(storage.clone())];
        for (i,slot) in slots.iter_mut().enumerate() {
            slot.select_window(RowWindow::within(i*16..i*16+16,48)?)?;
            let b=inputs[i].buffers();runtime.block_on(unsafe {slot.execute(0,16,b[0],b[1],b[2],b[3])})?;
        }
        let rank=storage.execution.try_lock().unwrap().routed.rank_output(0);
        let reader=Stream::new(devices[1])?;let saved=Allocation::new(devices[1],want.len())?;
        let reader_copy=devices[1].run(||lib.v41_peer_copy())?;
        devices[1].run(||unsafe {reader_copy.launch(saved.buffer,rank,want.len(),reader.raw)})?;reader.drain()?;
        let chains=[StageChain::on_devices(&lib,&[0,1])?,StageChain::on_devices(&lib,&[0,1])?];
        let gate=devices[1].run(||unsafe {cuteafd_ffi::test_support::CudaStreamGate::new(&lib,reader.raw)})?;
        runtime.block_on(chains[0].handle().scope(async {
            let b=inputs[0].buffers();unsafe {slots[0].execute(0,16,b[0],b[1],b[2],b[3]).await?;}
            devices[1].run(||unsafe {crate::shared::memory::chain::join(&lib,reader.raw)?;
                reader_copy.launch(saved.buffer,rank,want.len(),reader.raw)?;
                crate::shared::memory::chain::finish(&lib,reader.raw)})?;
            slots[0].release_scratch()
        }))?;
        SKIP_FFN_ACQUIRE.with(|s|s.set(!guarded));
        let queued=runtime.block_on(chains[1].handle().scope(async {
            let b=inputs[1].buffers();unsafe {slots[1].execute(0,16,b[0],b[1],b[2],b[3]).await?;}anyhow::Ok(())
        }));SKIP_FFN_ACQUIRE.with(|s|s.set(false));queued?;
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(if guarded {2}else {10});let mut done=false;
        while std::time::Instant::now()<deadline {
            let execution = storage.execution.try_lock().unwrap();
            let stream = execution.routed.rank_stream(0);
            if stream.device.run(||unsafe {lib.cuda_stream_query(stream.raw)})? {done=true;break;}
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let final_done = devices[1].run(||unsafe {lib.cuda_stream_query(storage.streams[1].raw)})?;
        eprintln!("OwnerPair guarded={guarded}: rank0_ready={done}, final_add_ready={final_done} before reader release");
        devices[1].run(|| {drop(gate);Ok(())})?;for chain in &chains {chain.drain()?;}reader.drain()?;
        assert_eq!(done,!guarded,"TP2 FFN B rank completion before A rank-reader release");
        let mut got=vec![0;want.len()];devices[1].run(||lib.copy_d2h(&mut got,saved.buffer))?;
        if guarded {assert_eq!(got,want);}else {assert_ne!(got,want,"unguarded actual OwnerPair did not clobber");}
        let execution=storage.execution.try_lock().unwrap();
        for (slot,expected) in [(0,want_a),(1,want_b)] {
            let output=RowWindow::within(slot*16..slot*16+16,48)?.apply(execution.output[slot].buffer,10240);
            let mut actual=vec![0;expected.len()];
            devices[slot].run(||lib.copy_d2h(&mut actual,output))?;
            assert_eq!(actual,expected,"slot {slot} full FFN output differs from private run");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires actual TP2 routed/shared kernels, snapshot and two GPUs"]
    fn tp2_ffn_shared_owner_preserves_gated_peer_reader()->Result<()> {for _ in 0..3 {run(true)?;}Ok(())}
    #[test]
    #[ignore = "requires actual TP2 routed/shared kernels, snapshot and two GPUs"]
    fn tp2_ffn_without_owner_guard_clobbers_gated_peer_reader()->Result<()> {for _ in 0..3 {run(false)?;}Ok(())}
    /// 2-RTX assembly at k = 2/3/4: one routed load at the unit cap serves the
    /// slots' shared arena (reserve 8192, unit U) and both decode lanes at
    /// their own width, each pair admitted at its arena's capacity.
    #[test]
    #[ignore = "requires actual TP2 routed/shared kernels, snapshot and two GPUs"]
    fn one_routed_load_builds_slot_storage_and_decode_lanes_at_every_k()->Result<()> {
        let lib=unsafe {cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)?};lib.cuda_set_device(0)?;
        let catalog=cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let devices=[Device {library:&lib,id:0},Device {library:&lib,id:1}];
        let decode=crate::families::deepseek_v41::v41_native_serve::DECODE_LANE_ROWS;
        for unit in [4096u32,2752,2048] {
            let routed=load_routed_weights(devices,&catalog,1,unit,[8_000_000_000;2])?;
            let shared=[Rc::new(vec![SharedWeights::load(devices[0],&catalog,0,SharedWeights::load_peak_device_bytes())?]),
                Rc::new(vec![SharedWeights::load(devices[1],&catalog,0,SharedWeights::load_peak_device_bytes())?])];
            let storage=Storage::new(routed.clone(),shared.clone(),1,8192,unit,[None,None])?;
            let lanes=[Wave::new(routed.clone(),shared.clone(),1,decode,[None,None])?,
                Wave::new(routed.clone(),shared.clone(),1,decode,[None,None])?];
            assert_eq!(storage.execution.try_lock().unwrap().capacity,unit);
            for lane in &lanes {assert_eq!(lane.storage.execution.try_lock().unwrap().capacity,decode);}
            assert!(routed.iter().all(|w| w.max_rows()==unit as usize),"re-admission must not change the loaded handle");
            eprintln!("k unit {unit}: slot storage + 2 decode lanes at {decode} built");
        }
        Ok(())
    }
    /// A2/A5: the FFN input planes pushed to the other GPU (ids and weights
    /// as 16-byte-rounded flat planes) equal the pulled planes on every row
    /// the consumers read, both directions, at decode rows 1, 6, 17, 48 and
    /// 64; nothing past the rounded extent changes.
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs with peer access"]
    fn input_planes_push_matches_pull() -> Result<()> {
        use crate::shared::memory::chain::{self, StageChain};
        // SAFETY: the explicitly selected native library outlives every owner below.
        let lib = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
        for gpu in 0..2 { devices[gpu].run(|| lib.cuda_enable_peer(1 - gpu as i32))?; }
        const CAPACITY: usize = 64;
        let widths = [10240usize, 5280, 24, 24];
        let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
        let flags = crate::shared::peer_split::PeerExchange::new_flags(&lib, [0, 1], FLAGS,
            crate::families::deepseek_v41::v41_peer_push::lane_bytes_with(CAPACITY, 15568, None)?)?;
        let sm = [devices[0].run(|| lib.v41_peer_copy())?, devices[1].run(|| lib.v41_peer_copy())?];
        let chain = StageChain::on_devices(&lib, &[0, 1])?;
        chain.set_device_order(true);
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for local in [0usize, 1] {
            let remote = 1 - local;
            let sources = widths.map(|width| Allocation::new(devices[local], CAPACITY * width));
            let sources: Vec<_> = sources.into_iter().collect::<Result<_>>()?;
            let peers: Vec<_> = widths.iter().map(|&width| Allocation::new(devices[remote], CAPACITY * width)).collect::<Result<_>>()?;
            for rows in [1usize, 6, 17, 48, 64] {
                for (plane, (source, &width)) in sources.iter().zip(&widths).enumerate() {
                    let data: Vec<u8> = (0..CAPACITY * width).map(|i| ((i * 31 + plane * 7 + rows) % 253) as u8).collect();
                    devices[local].run(|| lib.copy_h2d(source.buffer, &data))?;
                }
                let mut results = Vec::new();
                for push in [false, true] {
                    for peer in &peers { devices[remote].run(|| lib.copy_h2d(peer.buffer, &vec![0x6b; peer.buffer.bytes]))?; }
                    let planes: Vec<_> = sources.iter().zip(&peers).zip(&widths).map(|((source, peer), &width)|
                        (source.buffer, peer.buffer, if push { padded(rows * width) } else { rows * width })).collect();
                    runtime.block_on(chain.handle().scope(async { unsafe {
                        if push { return push_inputs(&flags, &streams, local, &planes).map(|_| ()); }
                        streams[remote].device.run(|| {
                            chain::join(&lib, streams[remote].raw)?;
                            for &(source, destination, bytes) in &planes {
                                peer_copy(&streams[remote], Some(&sm[remote]), destination, source, bytes)?;
                            }
                            chain::finish(&lib, streams[remote].raw)
                        })
                    } }))?;
                    chain.drain_bounded(std::time::Duration::from_secs(10))?;
                    lib.cuda_device_synchronize()?;
                    let mut landed = Vec::new();
                    for peer in &peers {
                        let mut bytes = vec![0u8; peer.buffer.bytes];
                        devices[remote].run(|| lib.copy_d2h(&mut bytes, peer.buffer))?;
                        landed.push(bytes);
                    }
                    results.push(landed);
                }
                for (plane, &width) in widths.iter().enumerate() {
                    let (pulled, pushed) = (&results[0][plane], &results[1][plane]);
                    ensure!(pulled[..rows * width] == pushed[..rows * width],
                        "A2/A5 plane {plane} differs from the pull at {rows} rows from GPU {local}");
                    ensure!(pushed[padded(rows * width)..].iter().all(|&b| b == 0x6b),
                        "A2/A5 plane {plane} wrote past its rounded extent at {rows} rows");
                }
            }
        }
        eprintln!("A2/A5 input planes: push == pull on every read row, both directions, rows 1, 6, 17, 48, 64");
        Ok(())
    }

    /// A7: the reduced result returned GPU1 -> GPU0 by push equals the pull,
    /// at decode rows 1, 6, 17, 48 and 64, and leaves rows past it untouched.
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs with peer access"]
    fn result_return_push_matches_pull() -> Result<()> {
        use crate::shared::memory::chain::{self, StageChain};
        // SAFETY: the explicitly selected native library outlives every owner below.
        let lib = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
        for gpu in 0..2 { devices[gpu].run(|| lib.cuda_enable_peer(1 - gpu as i32))?; }
        let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
        let flags = crate::shared::peer_split::PeerExchange::new_flags(&lib, [0, 1], FLAGS,
            crate::families::deepseek_v41::v41_peer_push::lane_bytes_with(64, 15568, None)?)?;
        let sm = devices[0].run(|| lib.v41_peer_copy())?;
        let source = Allocation::new(devices[1], 64 * 10240)?;
        let output = Allocation::new(devices[0], 64 * 10240)?;
        let chain = StageChain::on_devices(&lib, &[0, 1])?;
        chain.set_device_order(true);
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for rows in [1usize, 6, 17, 48, 64] {
            let data: Vec<u8> = (0..64 * 10240).map(|i| ((i * 7 + rows * 13) % 251) as u8).collect();
            devices[1].run(|| lib.copy_h2d(source.buffer, &data))?;
            let mut results = Vec::new();
            for push in [false, true] {
                devices[0].run(|| lib.copy_h2d(output.buffer, &vec![0x77; 64 * 10240]))?;
                let src = CuteafdDeviceBuffer { bytes: rows * 10240, ..source.buffer };
                let out = CuteafdDeviceBuffer { bytes: rows * 10240, ..output.buffer };
                runtime.block_on(chain.handle().scope(async { unsafe {
                    if push { return return_pushed(&flags, &streams, src, out, 0); }
                    devices[0].run(|| {
                        chain::join(&lib, streams[0].raw)?;
                        peer_copy(&streams[0], Some(&sm), out, src, out.bytes)?;
                        chain::finish(&lib, streams[0].raw)
                    })
                } }))?;
                chain.drain_bounded(std::time::Duration::from_secs(10))?;
                lib.cuda_device_synchronize()?;
                let mut landed = vec![0u8; 64 * 10240];
                devices[0].run(|| lib.copy_d2h(&mut landed, output.buffer))?;
                results.push(landed);
            }
            ensure!(results[0] == results[1], "A7 push differs from pull at {rows} rows");
            ensure!(results[1][..rows * 10240] == data[..rows * 10240]
                && results[1][rows * 10240..].iter().all(|&b| b == 0x77), "A7 push moved the wrong rows at {rows}");
        }
        eprintln!("A7 result return: push == pull at rows 1, 6, 17, 48, 64");
        Ok(())
    }

    #[test]
    fn peer_input_storage_bytes_equal_planned_allocations() {
        assert_eq!(PeerInputs::allocation_sizes(8192).into_iter().sum::<usize>(),8192*15568);
    }
}
