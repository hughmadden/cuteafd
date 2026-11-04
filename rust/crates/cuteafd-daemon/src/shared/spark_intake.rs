//! Where the Spark ranks' routed-expert partials arrive on the coordinator.
//!
//! Each prefill/decode transport owns a [`SparkIntake`]: one device plane per
//! rank that the compact reducer sums (with the shared expert) after a wave.
//! How the rows reach those planes is chosen once per process
//! ([`IntakeMode`], `CUTEAFD_SPARK_INTAKE`):
//!
//! - `gpu`: the NIC writes payloads straight into the planes (GPUDirect RDMA
//!   over dma-buf; see `cuteafd_transport::DeviceLanding`). No host copy, no
//!   upload.
//! - `pinned`: payloads stay in the transport's pinned receive slots; the
//!   link's exchange lands them in the planes from there
//!   ([`crate::shared::spark_waves`]: an SM pull of the frames in place, or
//!   copy-engine copies once the host saw the wave land).
//! - `host`: zero partials staged in pinned memory and uploaded on the
//!   compute stream (benchmarks without Spark ranks, [`SparkIntake::upload_zeros`]).
//!
//! `auto` (the default) probes dma-buf landing once at startup (a loopback
//! QP pair timing sends into device vs pinned host memory, and pinned host to
//! device copies) and picks `gpu` when the NIC lands in device memory at
//! least [`GPU_SHARE_OF_PINNED`] of the pinned path's store-and-forward rate
//! (NIC into host memory, then the H2D copy), `pinned` otherwise. On raptor
//! the NIC writes GPU memory at ~19.6 GB/s (cross-root-complex peer writes)
//! against ~52 GB/s into host memory, yet landing wins end to end at every
//! wave size measured: device landing takes no host memory bandwidth, CPU,
//! copy engine or receive-slot retention, and prefill lanes overlap each
//! wave's wire time with the other lanes' GPU work, so a wave's standalone
//! landing time (18 ms for V4 Pro TP6's 352 MB) is not on the critical path.
//! An unusable capability probe selects `pinned` and says why. Once `gpu`
//! is selected, range alignment or registration failure refuses startup;
//! the selected path never silently changes to pinned receives.
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_transport::expert::SparkExperts;
use cuteafd_transport::{DeviceLanding, GpuLandingProbe, VerbsHostProtocolV2ResponsePayload};
use crate::shared::wave_geometry::SlotGeometry;
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

/// Most ranks a compact reduction sums.
pub(crate) const MAX_INTAKE_RANKS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum IntakeMode {
    Host,
    Pinned,
    Gpu,
}

impl IntakeMode {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Pinned => "pinned",
            Self::Gpu => "gpu",
        }
    }
}

/// The process-wide intake setting, its probe, and why.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct IntakeChoice {
    /// `CUTEAFD_SPARK_INTAKE` (auto, gpu, pinned or host).
    pub(crate) setting: String,
    /// The mode before a transport's wave size is known: `auto` resolves
    /// `gpu` per transport ([`Self::mode_for`]).
    pub(crate) mode: IntakeMode,
    pub(crate) reason: String,
    pub(crate) probe: Option<GpuLandingProbe>,
    /// Pinned host to device copy rate, GB/s (the pinned path's second leg).
    pub(crate) h2d_gbps: Option<f64>,
}

/// `auto` lands in device memory when the probed landing rate is at least
/// this share of the pinned path's store-and-forward rate
/// ([`pinned_path_gbps`]). Measured end to end, landing won at shares 0.53
/// (loopback on GPU1, GLM 5.3 Flash TP4 8K prefill 566 -> 541 ms, per-wave
/// 6.4 vs 12.0 ms) and 0.71 (raptor GPU0 + Sparks, landing 19.6 GB/s, host
/// 52.4, H2D 57.6: GLM 5.3 Flash TP4 5329 -> 5477 tok/s, V4 Pro TP6 1502 ->
/// 1650 tok/s, C1 decode +2-3%), the win growing with the wave (134 -> 352
/// MB); below half nothing was measured.
pub(crate) const GPU_SHARE_OF_PINNED: f64 = 0.5;

/// The pinned path's rate for a wave: every byte lands in host memory and
/// then crosses to the GPU, and a rank's upload starts only once its rows
/// have landed, so the legs add up per byte.
pub(crate) fn pinned_path_gbps(host_gbps: f64, h2d_gbps: f64) -> f64 {
    1.0 / (1.0 / host_gbps + 1.0 / h2d_gbps)
}

/// Whether `auto` lands in device memory at the probed rates (landing
/// verified), and the rates it compared.
fn gpu_wins(probe: &GpuLandingProbe, h2d_gbps: Option<f64>) -> (bool, String) {
    match h2d_gbps {
        Some(h2d) if probe.gpu_gbps > 0.0 && probe.host_gbps > 0.0 && h2d > 0.0 => {
            let pinned = pinned_path_gbps(probe.host_gbps, h2d);
            let share = probe.gpu_gbps / pinned;
            (share >= GPU_SHARE_OF_PINNED, format!("landing {:.1} GB/s = {share:.2} of the pinned path's {pinned:.1} \
                (host {:.1}, H2D {h2d:.1}; gpu from {GPU_SHARE_OF_PINNED})", probe.gpu_gbps, probe.host_gbps))
        }
        _ => (probe.gpu_gbps > 0.0, format!("landing {:.1} GB/s (pinned path not probed)", probe.gpu_gbps)),
    }
}

impl IntakeChoice {
    /// The mode for a transport whose full wave carries `wave_bytes` (the
    /// same for every wave size; logged with the probed landing time).
    pub(crate) fn mode_for(&self, wave_bytes: usize) -> (IntakeMode, String) {
        match (&self.probe, self.mode) {
            (Some(probe), IntakeMode::Gpu) if probe.gpu_gbps > 0.0 => {
                let ms = wave_bytes as f64 / (probe.gpu_gbps * 1e6);
                (IntakeMode::Gpu, format!("{} ({:.0} MB waves land in {ms:.1} ms)", self.reason, wave_bytes as f64 / 1e6))
            }
            _ => (self.mode, self.reason.clone()),
        }
    }
}

/// The RDMA device, port and source GID a landing probe binds: chosen by the
/// address that reaches the Spark peers, never by device or GID table order.
pub(crate) type LandingRoute<'r> = &'r dyn Fn() -> Result<cuteafd_transport::fabric::RdmaSelection>;

/// Probes dma-buf GPU landing (64 MiB loopback sends) on the current device,
/// over `route`'s device, port and source GID.
pub(crate) fn probe_gpu_landing(library: &NativeLibrary, route: LandingRoute<'_>) -> Result<GpuLandingProbe> {
    ensure!(!std::env::var("CUTEAFD_GPU_LANDING_PROBE").is_ok_and(|v| v == "unavailable"),
        "GPU landing probe forced unavailable");
    let rdma = route()?;
    let probe = cuteafd_transport::gpu_landing_probe(library, &rdma, 64 << 20, 8)?;
    let link = landing_pcie_link(library);
    tracing::info!(cuda_device = probe.cuda_device, rdma_device = %rdma.device, port = rdma.port,
        gid_index = rdma.gid_index, gpu_gbps = probe.gpu_gbps, host_gbps = probe.host_gbps, pcie_generation = ?link.as_ref().and_then(|link| link.generation),
        pcie_width = ?link.as_ref().and_then(|link| link.width), usable = probe.usable(), "GPU landing probe");
    Ok(probe)
}

/// Pinned host to device copy rate on the current device, GB/s: eight
/// 64 MiB copies after one warm-up, on a stream of their own.
pub(crate) fn probe_h2d(library: &NativeLibrary) -> Result<f64> {
    let _memory_scope = cuteafd_ffi::memory_ledger::scope("probe");
    const BYTES: usize = 64 << 20;
    let host = HostAllocation::new(library, BYTES)?;
    let device = DeviceAllocation::new(library, BYTES)?;
    let stream = library.cuda_stream_create()?;
    let timed = (|| -> Result<f64> {
        let mut started = std::time::Instant::now();
        for copy in 0..9 {
            // SAFETY: both buffers hold BYTES and outlive the stream sync below.
            unsafe { library.copy_host_buffer_h2d_async(device.buffer, host.buffer, BYTES, stream)? };
            if copy == 0 {
                // SAFETY: the stream was created above.
                unsafe { library.cuda_stream_synchronize(stream)? };
                started = std::time::Instant::now();
            }
        }
        // SAFETY: as above.
        unsafe { library.cuda_stream_synchronize(stream)? };
        Ok(8.0 * BYTES as f64 / started.elapsed().as_secs_f64() / 1e9)
    })();
    // SAFETY: the stream is idle (synchronized, or its copies failed to queue).
    unsafe {
        let _ = library.cuda_stream_synchronize(stream);
        library.cuda_stream_destroy(stream)?;
    }
    timed
}

/// Exercise the copy engine and PCIe link before timing peer writes.
fn warm_landing_link(library: &NativeLibrary) -> Result<()> {
    probe_h2d(library).map(|_| ())
}

fn resolve_landing_probe(setting: &str, probe: &Result<GpuLandingProbe>, h2d: Option<f64>) -> (IntakeMode, String) {
    match probe {
        Err(error) => (IntakeMode::Pinned, format!("GPU landing probe failed: {error:#}")),
        Ok(p) if !p.usable() => (IntakeMode::Pinned, format!("GPU landing unusable: {}{}",
            p.error.as_deref().unwrap_or(&p.status),
            if p.registered && p.writes_ordering < 100 { " (GPUDirect writes not ordered for kernels)" } else { "" })),
        Ok(p) if setting == "auto" => match gpu_wins(p, h2d) {
            (true, rates) => (IntakeMode::Gpu, format!("dma-buf landing verified, {rates}")),
            (false, rates) => (IntakeMode::Pinned, format!("dma-buf landing verified but slow, {rates}")),
        },
        Ok(p) => (IntakeMode::Gpu, format!("dma-buf landing verified, {:.1} GB/s into GPU vs {:.1} GB/s into host",
            p.gpu_gbps, p.host_gbps)),
    }
}

/// Resolves `CUTEAFD_SPARK_INTAKE` (and probes dma-buf landing over `route`
/// for `auto` and `gpu`) once per process.
pub(crate) fn choose_mode(library: &NativeLibrary, route: LandingRoute<'_>) -> Result<IntakeChoice> {
    static CHOICE: OnceLock<IntakeChoice> = OnceLock::new();
    if let Some(choice) = CHOICE.get() {
        return Ok(choice.clone());
    }
    let setting = std::env::var("CUTEAFD_SPARK_INTAKE").unwrap_or_else(|_| "auto".into());
    let (mode, reason, probe, h2d_gbps) = match setting.as_str() {
        "host" => (IntakeMode::Host, "CUTEAFD_SPARK_INTAKE=host".to_string(), None, None),
        "pinned" => (IntakeMode::Pinned, "CUTEAFD_SPARK_INTAKE=pinned".to_string(), None, None),
        "gpu" | "auto" => {
            let probe = probe_gpu_landing(library, route);
            let h2d = match (&probe, setting.as_str()) {
                (Ok(p), "auto") if p.usable() => probe_h2d(library)
                    .map_err(|error| tracing::warn!("H2D copy probe failed: {error:#}")).ok(),
                _ => None,
            };
            let (mode, reason) = resolve_landing_probe(&setting, &probe, h2d);
            (mode, reason, probe.ok(), h2d)
        }
        other => anyhow::bail!("CUTEAFD_SPARK_INTAKE={other:?} is not one of auto, gpu, pinned, host"),
    };
    let choice = IntakeChoice { setting, mode, reason, probe, h2d_gbps };
    tracing::info!(setting = %choice.setting, mode = choice.mode.name(), reason = %choice.reason, "Spark partial intake");
    Ok(CHOICE.get_or_init(|| choice).clone())
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct LandingPcieLink {
    generation: Option<u32>,
    width: Option<u32>,
}

fn landing_pcie_link(library: &NativeLibrary) -> Option<LandingPcieLink> {
    let bus = library.cuda_device_pci_bus_id(library.cuda_get_device().ok()?)?;
    let nvml = cuteafd_ffi::nvml::Nvml::open().ok()?;
    let devices = nvml.devices().ok()?;
    let device = devices.iter().find(|device| device.identity.pci_bus_id.trim_start_matches('0')
        .eq_ignore_ascii_case(bus.trim_start_matches('0')))?;
    let (generation, width) = device.pcie_link();
    Some(LandingPcieLink { generation, width })
}

/// `cuteafd fabric`'s view of GPU landing on one CUDA device.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct FabricLanding {
    pub(crate) cuda_device: i32,
    pub(crate) pcie_before: Option<LandingPcieLink>,
    pub(crate) pcie_link: Option<LandingPcieLink>,
    pub(crate) probe: Option<GpuLandingProbe>,
    pub(crate) h2d_gbps: Option<f64>,
    pub(crate) error: Option<String>,
}

impl FabricLanding {
    pub(crate) fn summary(&self) -> String {
        match (&self.probe, &self.error) {
            (Some(p), _) if p.usable() => {
                let (gpu, rates) = gpu_wins(p, self.h2d_gbps);
                let link = self.pcie_link.as_ref().map(|link| format!("PCIe gen {} x{}",
                    link.generation.map_or("?".into(), |value| value.to_string()),
                    link.width.map_or("?".into(), |value| value.to_string())))
                    .unwrap_or_else(|| "PCIe unknown".into());
                format!("GPU landing (cuda {}, {}, {link}): dma-buf ok, loopback {rates}; intake auto: {}",
                    self.cuda_device, p.rdma_device, if gpu { "gpu" } else { "pinned" })
            }
            (Some(p), _) => format!("GPU landing (cuda {}): unusable ({}); intake auto: pinned", self.cuda_device,
                p.error.as_deref().unwrap_or(&p.status)),
            (None, Some(error)) => format!("GPU landing (cuda {}): not probed ({error}); intake auto: pinned",
                self.cuda_device),
            (None, None) => "GPU landing: not probed".into(),
        }
    }
}

/// The device, port and source GID `cuteafd fabric` probes landing on: the
/// address that reaches `peer`, or, without one, the only RoCE v2 address on
/// the host. Several candidates and no peer is refused rather than guessed.
fn fabric_landing_route(peer: Option<std::net::SocketAddr>) -> Result<cuteafd_transport::fabric::RdmaSelection> {
    let ports = cuteafd_transport::fabric::discover()?.ports;
    if let Some(peer) = peer {
        return cuteafd_transport::fabric::select_rdma_route(&ports, peer);
    }
    sole_rdma_address(&ports)
}

fn sole_rdma_address(ports: &[cuteafd_transport::fabric::RdmaPort]) -> Result<cuteafd_transport::fabric::RdmaSelection> {
    let candidates: Vec<_> = ports.iter().filter(|port| port.active && port.link_layer == "Ethernet")
        .flat_map(|port| port.roce_v2.iter().map(move |&(gid_index, address)| (port, gid_index, address))).collect();
    match candidates.as_slice() {
        [(port, gid_index, _)] => Ok(cuteafd_transport::fabric::RdmaSelection {
            device: port.device.clone(), port: port.port, gid_index: *gid_index }),
        [] => anyhow::bail!("no active RoCE v2 IPv4 address"),
        many => anyhow::bail!("{} RoCE v2 IPv4 addresses ({}); pass --landing-peer to pick the one that reaches the Sparks",
            many.len(), many.iter().map(|(port, _, address)| format!("{}:{address}", port.device)).collect::<Vec<_>>().join(", ")),
    }
}

/// Runs the GPU landing probe for `cuteafd fabric` when a native library is
/// given (or `CUTEAFD_NATIVE_LIB` names one).
pub(crate) fn fabric_probe(native_lib: Option<&std::path::Path>, device: i32, warmup: bool, repeats: u32,
    peer: Option<std::net::SocketAddr>) -> Vec<FabricLanding> {
    let library = (|| -> Result<NativeLibrary> {
        let path = native_lib.map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("CUTEAFD_NATIVE_LIB").map(std::path::PathBuf::from))
            .context("no native library (pass --native-lib or set CUTEAFD_NATIVE_LIB)")?;
        // SAFETY: a trusted image library, kept loaded across all repeats.
        let library = unsafe { NativeLibrary::load(&path) }?;
        library.cuda_set_device(device)?;
        Ok(library)
    })();
    (0..repeats.clamp(1, 5)).map(|_| {
        let pcie_before = library.as_ref().ok().and_then(landing_pcie_link);
        let result = (|| -> Result<(GpuLandingProbe, Option<f64>, Option<LandingPcieLink>)> {
            let library = library.as_ref().map_err(|error| anyhow::anyhow!("{error:#}"))?;
            if warmup { warm_landing_link(library)?; }
            let probe = probe_gpu_landing(library, &|| fabric_landing_route(peer))?;
            let pcie_link = landing_pcie_link(library);
            let h2d = probe_h2d(library).ok();
            Ok((probe, h2d, pcie_link))
        })();
        match result {
            Ok((probe, h2d_gbps, pcie_link)) => FabricLanding { cuda_device: device, pcie_before, pcie_link, probe: Some(probe), h2d_gbps, error: None },
            Err(error) => FabricLanding { cuda_device: device, pcie_before, pcie_link: None, probe: None, h2d_gbps: None, error: Some(format!("{error:#}")) },
        }
    }).collect()
}

/// One transport's rank planes.
///
/// Each rank owns one range of `waves` return slots of `rows` rows
/// ([`SlotGeometry`], fixed for the intake's lifetime). The link's exchange
/// ([`crate::shared::spark_waves`]) takes the slots in dispatch order, lands
/// each wave in its slot and reuses a slot only once its previous wave was
/// consumed.
pub(crate) struct SparkIntake<'a> {
    library: &'a NativeLibrary,
    mode: IntakeMode,
    ranks: usize,
    row_bytes: usize,
    /// One allocation per rank holding its whole reserve range.
    planes: Vec<IntakePlane<'a>>,
    /// Admitted return slots.
    waves: usize,
    geometry: RefCell<SlotGeometry>,
    /// Host mode: every rank's plane back to back ([`Self::upload_zeros`]).
    staging: Option<RefCell<HostAllocation<'a>>>,
    /// The stream terminal teardown drains ([`Self::terminal_drain_copies`]).
    copy_stream: *mut c_void,
    /// Reply frames a closed exchange still held (its pulls may have read
    /// them): released at the link's reset or terminal release, after its
    /// consuming streams drained.
    held: RefCell<Vec<VerbsHostProtocolV2ResponsePayload>>,
}

/// Owns the raw cudaMalloc while every consumer uses the aligned interior view.
struct IntakePlane<'a> {
    _allocation: DeviceAllocation<'a>,
    buffer: CuteafdDeviceBuffer,
}

impl<'a> SparkIntake<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, mode: IntakeMode, ranks: usize, rows: usize, row_bytes: usize)
        -> Result<Self> {
        Self::with_waves(library, mode, ranks, rows, row_bytes, 1)
    }

    /// `waves` return slots of `rows` rows, registered as one contiguous
    /// range of `waves * rows` rows per rank.
    pub(crate) fn with_waves(library: &'a NativeLibrary, mode: IntakeMode, ranks: usize, rows: usize,
        row_bytes: usize, waves: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("spark-intake");
        ensure!((1..=8).contains(&ranks), "{ranks} Spark ranks are outside the one-through-eight transport worlds");
        ensure!((1..=4).contains(&waves), "Spark intake admits 1..=4 return slots");
        let geometry = SlotGeometry::new(waves, rows)?;
        let reserve_rows = geometry.reserve_rows();
        let plane_bytes = rows.checked_mul(row_bytes).context("Spark intake plane size overflow")?;
        let logical_bytes = reserve_rows.checked_mul(row_bytes).context("Spark intake reserve size overflow")?;
        let reserve_bytes = usize::try_from(cuteafd_core::serving_capacity::spark_intake_allocation_bytes(
            u64::try_from(logical_bytes.max(256))?)?)?;
        let planes = (0..ranks).map(|rank| {
            let allocation = DeviceAllocation::new(library, reserve_bytes)?;
            let (offset, bytes) = cuteafd_core::serving_capacity::spark_intake_export_range(
                allocation.buffer.ptr as u64, u64::try_from(logical_bytes.max(256))?)?;
            let (offset, bytes) = (usize::try_from(offset)?, usize::try_from(bytes)?);
            ensure!(offset.checked_add(bytes).is_some_and(|end| end <= allocation.buffer.bytes),
                "Spark intake aligned range exceeds raw allocation");
            let buffer = CuteafdDeviceBuffer {
                // SAFETY: the checked aligned interior range belongs to the retained raw allocation.
                ptr: unsafe { allocation.buffer.ptr.cast::<u8>().add(offset) }.cast(),
                bytes, ..allocation.buffer
            };
            tracing::info!(rank, waves, rows, row_bytes, logical_bytes, allocation_bytes = reserve_bytes,
                allocation_ptr = ?allocation.buffer.ptr, aligned_offset = offset,
                ptr = ?buffer.ptr, registered_bytes = bytes, "allocated Spark intake reserve");
            Ok(IntakePlane { _allocation: allocation, buffer })
        }).collect::<Result<Vec<_>>>()?;
        // Host mode stages its zero partials; the exchange lands every other wave.
        let staging = match mode {
            IntakeMode::Host => Some(RefCell::new(HostAllocation::new(library, ranks * plane_bytes)?)),
            IntakeMode::Pinned | IntakeMode::Gpu => None,
        };
        let copy_stream = library.cuda_stream_create()?;
        Ok(Self {
            library, mode, ranks, row_bytes, planes, waves, geometry: RefCell::new(geometry), staging, copy_stream,
            held: RefCell::new(Vec::new()),
        })
    }

    /// The device ranges a GPU-mode transport lands in: a single-slot link's
    /// one range, or a multi-slot link's placed reserve range, whose waves
    /// name their slices ([`Self::placed`]).
    pub(crate) fn landing(&self) -> Option<Vec<DeviceLanding>> {
        (self.mode == IntakeMode::Gpu).then(|| self.planes.iter().map(|p| match self.placed() {
            true => DeviceLanding::placed(p.buffer).expect("validated intake reserve"),
            false => DeviceLanding::new(p.buffer),
        }).collect())
    }

    /// Multi-slot links place each wave's GPU landing slice per request; a
    /// single-slot link keeps one rotating range.
    pub(crate) fn placed(&self) -> bool { self.waves > 1 }

    /// Each rank's whole reserve range.
    pub(crate) fn ranges(&self) -> Vec<CuteafdDeviceBuffer> { self.planes.iter().map(|plane| plane.buffer).collect() }
    /// The return slots and their rows.
    pub(crate) fn geometry(&self) -> SlotGeometry { self.geometry.borrow().clone() }

    /// Return slots of the current cut.
    pub(crate) fn waves(&self) -> usize { self.geometry.borrow().cut().waves }
    /// Admitted return slots.
    pub(crate) fn slots(&self) -> usize { self.waves }
    /// Rows of one return slot under the current cut.
    pub(crate) fn rows(&self) -> usize { self.geometry.borrow().cut().rows }
    pub(crate) fn row_bytes(&self) -> usize { self.row_bytes }
    pub(crate) fn ranks(&self) -> usize { self.ranks }
    /// The CUDA device the planes live on.
    pub(crate) fn device(&self) -> i32 { self.planes[0].buffer.device_id }

    /// Reply frames a closed exchange still held (its pulls may have read
    /// them): kept until the link's reset or terminal release, which follow
    /// the drain of its consuming streams.
    pub(crate) fn keep_frames(&self, frames: Vec<VerbsHostProtocolV2ResponsePayload>) {
        self.held.borrow_mut().extend(frames);
    }
    fn slot_planes(&self, slot: usize) -> impl Iterator<Item = CuteafdDeviceBuffer> + '_ {
        let (offset, bytes) = self.geometry.borrow().slice(slot, self.row_bytes);
        self.planes.iter().map(move |plane| CuteafdDeviceBuffer {
            // SAFETY: every slot of the cut lies within this rank's reserve allocation.
            ptr: unsafe { plane.buffer.ptr.cast::<u8>().add(offset) }.cast(),
            bytes,
            ..plane.buffer
        })
    }

    /// Points `transport`'s receives at this intake's planes (GPU mode).
    ///
    /// # Safety
    /// The intake must outlive `transport` (or its landing must be cleared
    /// first), and every dispatch on it must follow [`Self::before_dispatch`].
    pub(crate) unsafe fn attach(&self, transport: &mut SparkExperts) -> Result<()> {
        ensure!(transport.world_size() == self.ranks, "intake ranks differ from the transport's");
        // SAFETY: forwarded from this function's contract: the planes stay
        // allocated while the transport lives and are read only between a
        // wave's receive and the next dispatch.
        unsafe { transport.set_gpu_landing(self.landing()) }
    }

    fn check(&self, t: usize, ranks: usize) -> Result<usize> {
        let rows = self.rows();
        ensure!(ranks == self.ranks && t <= rows,
            "a wave of {t} rows from {ranks} ranks exceeds the intake ({} ranks x {rows} rows)", self.ranks);
        Ok(t * self.row_bytes)
    }

    /// Benchmarks without Spark ranks: zero partials of `t` rows uploaded
    /// through the pinned staging, as a host-mode wave would (host mode).
    pub(crate) fn upload_zeros(&self, t: usize, stream: *mut c_void) -> Result<()> {
        ensure!(self.mode == IntakeMode::Host, "zero partials go through a host-mode intake");
        let plane_bytes = self.check(t, self.ranks)?;
        self.staging.as_ref().context("host intake staging")?.borrow_mut().bytes_mut()[..self.ranks * plane_bytes]
            .fill(0);
        self.upload_staging(0, plane_bytes, stream)
    }

    /// Plane pointers of return slot 0 (a host-mode intake's, which
    /// [`Self::upload_zeros`] fills), for the compact reducer.
    pub(crate) fn pointers(&self) -> [*const u16; MAX_INTAKE_RANKS] {
        self.slot_pointers(0)
    }

    /// Plane pointers of return slot `slot`.
    pub(crate) fn slot_pointers(&self, slot: usize) -> [*const u16; MAX_INTAKE_RANKS] {
        let mut pointers = [std::ptr::null::<u16>(); MAX_INTAKE_RANKS];
        for (pointer, plane) in pointers.iter_mut().zip(self.slot_planes(slot)) {
            *pointer = plane.ptr.cast();
        }
        pointers
    }

    /// Resets the drained transport: the copy stream drains, the wire
    /// resets, and only then go the frames a closed exchange still held (the
    /// NIC or a pull may still have read them until the reset).
    fn reset(&self, transport: &mut SparkExperts) -> Result<()> {
        transport.require_drained_reset()?;
        self.terminal_drain_copies()?;
        transport.reset_drained()?;
        self.held.borrow_mut().clear();
        Ok(())
    }

    /// Terminal ownership only: the transport must first quiesce its QPs and
    /// the engine must drain its compute stream. Drain the independent copy
    /// stream directly, including uploads whose event publication failed.
    /// This method deliberately keeps every held payload until all intakes
    /// and compute streams have drained successfully.
    pub(crate) fn terminal_drain_copies(&self) -> Result<()> {
        if self.copy_stream.is_null() { return Ok(()); }
        // SAFETY: the intake owns the stream; its planes and source owners
        // remain live throughout terminal quiescence and this synchronization.
        unsafe { self.library.cuda_stream_synchronize(self.copy_stream) }
    }

    pub(crate) fn terminal_release_payloads(&self) {
        self.held.borrow_mut().clear();
    }

    fn upload_staging(&self, landed: u8, plane_bytes: usize, stream: *mut c_void) -> Result<()> {
        self.upload_staging_slot(0, landed, plane_bytes, stream)
    }

    fn upload_staging_slot(&self, slot: usize, landed: u8, plane_bytes: usize, stream: *mut c_void) -> Result<()> {
        let staging = self.staging.as_ref().context("host intake staging")?.borrow();
        for rank in (0..self.ranks).filter(|rank| landed & (1 << rank) == 0) {
            let source = CuteafdHostBuffer {
                // SAFETY: rank planes are disjoint slices of the staging buffer.
                ptr: unsafe { staging.buffer.ptr.cast::<u8>().add(rank * plane_bytes) }.cast(),
                bytes: plane_bytes,
                ..staging.buffer
            };
            // SAFETY: pinned source and device plane both hold `plane_bytes`;
            // the caller orders the next rewrite of the staging after this copy.
            unsafe { self.library.copy_host_buffer_h2d_async(self.slot_planes(slot).nth(rank).expect("validated intake rank"), source, plane_bytes, stream)? };
        }
        Ok(())
    }
}

pub(crate) fn upload_offset(ranks: usize, rank: usize, first: u32, row_bytes: usize,
    bytes: usize, plane_bytes: usize) -> Result<usize> {
    ensure!(rank < ranks, "rank {rank} exceeds intake plane count");
    let offset = (first as usize).checked_mul(row_bytes).context("intake row offset overflow")?;
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= plane_bytes),
        "rank {rank} partial rows run past its plane");
    Ok(offset)
}

pub(crate) trait StreamLibrary {
    fn create_stream(&self) -> Result<*mut c_void>;
    /// # Safety
    /// `stream` is a live stream this library created.
    unsafe fn synchronize_stream(&self, stream: *mut c_void) -> Result<()>;
    /// # Safety
    /// `stream` is a live stream this library created; it is not used again.
    unsafe fn destroy_stream(&self, stream: *mut c_void) -> Result<()>;
}

impl StreamLibrary for NativeLibrary {
    fn create_stream(&self) -> Result<*mut c_void> {
        self.cuda_stream_create()
    }
    unsafe fn synchronize_stream(&self, stream: *mut c_void) -> Result<()> {
        // SAFETY: the caller vouches for the stream.
        unsafe { self.cuda_stream_synchronize(stream) }
    }
    unsafe fn destroy_stream(&self, stream: *mut c_void) -> Result<()> {
        // SAFETY: the caller vouches for the stream.
        unsafe { self.cuda_stream_destroy(stream) }
    }
}

/// The temporary stream a start-up Spark transport warm-up receives its waves on. [`Self::finish`]
/// drains and destroys it and returns either failure; any other way out (an early return or `?`)
/// drops it, which drains and destroys it too and logs failures. Declare it after the transports
/// it serves, so that it drops first and their queued uploads drain while the intakes are alive.
pub(crate) struct WarmStream<'a, L: StreamLibrary = NativeLibrary> {
    library: &'a L,
    raw: *mut c_void,
}

impl<'a, L: StreamLibrary> WarmStream<'a, L> {
    pub(crate) fn new(library: &'a L) -> Result<Self> {
        Ok(Self { library, raw: library.create_stream()? })
    }

    pub(crate) fn raw(&self) -> *mut c_void {
        self.raw
    }

    /// Drains the stream, then destroys it, even when draining fails (that error comes first).
    pub(crate) fn finish(self) -> Result<()> {
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: the stream was created in `new`; `ManuallyDrop` keeps `drop` from destroying it again.
        let drained = unsafe { this.library.synchronize_stream(this.raw) };
        let destroyed = unsafe { this.library.destroy_stream(this.raw) };
        drained.and(destroyed)
    }
}

impl<L: StreamLibrary> Drop for WarmStream<'_, L> {
    fn drop(&mut self) {
        // SAFETY: the stream was created in `new`; `finish`, which destroys it, never drops the guard.
        if let Err(error) = unsafe { self.library.synchronize_stream(self.raw) } {
            tracing::error!(%error, "draining a Spark transport warm-up stream");
        }
        if let Err(error) = unsafe { self.library.destroy_stream(self.raw) } {
            tracing::error!(%error, "destroying a Spark transport warm-up stream");
        }
    }
}

#[cfg(test)]
mod warm_stream_tests {
    use super::*;

    /// Records every stream call; the calls named in `fail` return an error.
    #[derive(Default)]
    struct Fake { calls: RefCell<Vec<String>>, fail: Vec<&'static str> }
    impl Fake {
        fn call(&self, name: &'static str, stream: *mut c_void) -> Result<()> {
            self.calls.borrow_mut().push(format!("{name} {:#x}", stream as usize));
            ensure!(!self.fail.contains(&name), "injected {name} failure");
            Ok(())
        }
        fn calls(&self) -> Vec<String> {
            self.calls.borrow().clone()
        }
    }
    impl StreamLibrary for Fake {
        fn create_stream(&self) -> Result<*mut c_void> {
            let stream = 0x5000 as *mut c_void;
            self.call("create", stream)?;
            Ok(stream)
        }
        unsafe fn synchronize_stream(&self, stream: *mut c_void) -> Result<()> {
            self.call("synchronize", stream)
        }
        unsafe fn destroy_stream(&self, stream: *mut c_void) -> Result<()> {
            self.call("destroy", stream)
        }
    }

    /// The warm-ups' shape: create, receive `waves` on the stream (the one at `fail_at` fails), finish.
    fn warm(library: &Fake, waves: usize, fail_at: Option<usize>) -> Result<()> {
        let stream = WarmStream::new(library)?;
        for wave in 0..waves {
            library.call("receive", stream.raw())?;
            ensure!(Some(wave) != fail_at, "wave {wave} failed");
        }
        stream.finish()
    }

    const ALL: [&str; 4] = ["create 0x5000", "receive 0x5000", "synchronize 0x5000", "destroy 0x5000"];

    #[test]
    fn an_early_return_drains_and_destroys_the_warm_up_stream() {
        let library = Fake::default();
        assert_eq!(warm(&library, 3, Some(1)).unwrap_err().to_string(), "wave 1 failed");
        assert_eq!(library.calls(), [ALL[0], ALL[1], ALL[1], ALL[2], ALL[3]]);
        // A failing call inside the loop (`?`) unwinds the same way.
        let library = Fake { fail: vec!["receive"], ..Fake::default() };
        assert_eq!(warm(&library, 3, None).unwrap_err().to_string(), "injected receive failure");
        assert_eq!(library.calls(), ALL);
    }

    #[test]
    fn finishing_drains_then_destroys_the_stream_once() {
        let library = Fake::default();
        warm(&library, 2, None).unwrap();
        assert_eq!(library.calls(), [ALL[0], ALL[1], ALL[1], ALL[2], ALL[3]]);
    }

    #[test]
    fn a_failed_drain_or_destroy_is_reported_and_the_stream_still_goes() {
        for fail in ["synchronize", "destroy"] {
            let library = Fake { fail: vec![fail], ..Fake::default() };
            assert_eq!(warm(&library, 1, None).unwrap_err().to_string(), format!("injected {fail} failure"));
            assert_eq!(library.calls(), ALL, "{fail}");
        }
        // Dropped on an early return, the guard logs a failed drain and still destroys the stream.
        let library = Fake { fail: vec!["synchronize"], ..Fake::default() };
        assert_eq!(warm(&library, 1, Some(0)).unwrap_err().to_string(), "wave 0 failed");
        assert_eq!(library.calls(), ALL);
    }

    #[test]
    fn a_stream_that_was_not_created_is_not_destroyed() {
        let library = Fake { fail: vec!["create"], ..Fake::default() };
        assert_eq!(warm(&library, 1, None).unwrap_err().to_string(), "injected create failure");
        assert_eq!(library.calls(), [ALL[0]]);
    }

    /// GLM 5.3 Flash's and MiMo's start-up warm-ups hold their temporary stream in the guard, so
    /// every way out of them destroys it, and neither handles a raw warm-up stream any more.
    #[test]
    fn the_spark_transport_warm_ups_hold_their_stream_in_the_guard() {
        // The needles are joined here, so that this file's own text does not match them.
        let finished = ["warm_stream.", "finish()?;"].concat();
        for (family, source, library) in [
            ("glm5_flash", include_str!("../families/glm5_flash/mod.rs"), "self.library.as_ref()"),
            ("mimo_v2", include_str!("../families/mimo_v2/mod.rs"), "&*self.library"),
        ] {
            let created = ["let warm_stream = crate::shared::spark_intake::Warm", "Stream::new(", library, ")?;"].concat();
            assert_eq!((source.matches(&created).count(), source.matches(&finished).count()), (1, 1), "{family}");
            let (start, end) = (source.find(&created).unwrap(), source.find(&finished).unwrap());
            assert!(start < end, "{family}");
            let warm_up = &source[start..end];
            assert!(warm_up.contains(&["warm_stream.", "raw()"].concat()), "{family}");
            for raw in ["_create(", "_synchronize(", "_destroy("] {
                assert!(!warm_up.contains(&["cuda_stream", raw].concat()), "{family}: {raw}");
            }
            assert!(!source.contains(&["(warm_", "stream)"].concat()), "{family}: a raw warm-up stream call");
        }
    }

    fn spark_transport_constructor(source: &str) -> Option<usize> {
        ["new", "new_waves"].iter()
            .filter_map(|constructor| source.find(&format!("SparkLink::{constructor}(")))
            .min()
    }

    #[test]
    fn transport_audit_recognizes_single_and_multi_wave_constructors() {
        for constructor in ["new", "new_waves"] {
            let source = format!("let transport = SparkLink::{constructor}(");
            assert_eq!(spark_transport_constructor(&source), Some(16));
        }
        assert_eq!(spark_transport_constructor("SparkLink::new_unchecked("), None);
    }

    /// Every family that builds a Spark transport bootstraps it with a request
    /// at install, so each rank's persistent session is open and its immutable
    /// installed expert grouping is published before anything plans against the
    /// workers. Qwen 4's draft planner reads the grouping that early, and a
    /// transport that had never opened a session answered "worker rank 0 did
    /// not report installed expert grouping" instead of launching.
    ///
    /// The request argument is load-bearing: the request that opens a rank's
    /// session sizes its registered rings, so the bootstrap must hand over the
    /// widest wave the family dispatches (its own start-up warm-up's request).
    #[test]
    fn every_spark_family_bootstraps_its_transport_with_a_wave_request() {
        // The needles are joined here, so this file's own text does not match them.
        let open = [".boot", "strap(&"].concat();
        for (family, source) in [
            ("deepseek_v4", include_str!("../families/deepseek_v4/mod.rs")),
            ("glm5", include_str!("../families/glm5/mod.rs")),
            ("glm5_flash", include_str!("../families/glm5_flash/mod.rs")),
            ("mimo_v2", include_str!("../families/mimo_v2/mod.rs")),
            ("qwen4", include_str!("../families/qwen4/mod.rs")),
            ("step5", include_str!("../families/step5/serve.rs")),
            ("step5 golden", include_str!("../families/step5/golden.rs")),
        ] {
            let transport = spark_transport_constructor(source)
                .unwrap_or_else(|| panic!("{family}: no Spark transport"));
            let bootstrap = source.find(&open)
                .unwrap_or_else(|| panic!("{family}: the Spark transport is never bootstrapped"));
            assert!(bootstrap > transport, "{family}: the bootstrap precedes the transport it opens");
        }
        // The argument-less form is the shape that leaves every session
        // unopened; it must not come back.
        let bare = [".boot", "strap()"].concat();
        for source in [
            include_str!("../families/deepseek_v4/mod.rs"),
            include_str!("../families/deepseek_v4/engine.rs"),
            include_str!("../families/glm5/mod.rs"),
            include_str!("../families/glm5/engine.rs"),
            include_str!("../families/glm5_flash/mod.rs"),
            include_str!("../families/mimo_v2/mod.rs"),
            include_str!("../families/qwen4/mod.rs"),
            include_str!("../families/qwen4/engine.rs"),
        ] {
            assert!(!source.contains(&bare), "a Spark transport is bootstrapped without a wave request");
        }
    }
}

impl Drop for SparkIntake<'_> {
    fn drop(&mut self) {
        // SAFETY: the stream was created by this intake; queued work drains first.
        unsafe {
            if !self.copy_stream.is_null() {
                let _ = self.library.cuda_stream_synchronize(self.copy_stream);
                let _ = self.library.cuda_stream_destroy(self.copy_stream);
            }
        }
    }
}

/// `dst.copy_from_slice(src)`, split over threads for large payloads (one
/// core copies pinned memory at about 20 GB/s).
pub(crate) fn copy_parallel(dst: &mut [u8], src: &[u8]) {
    if src.len() < 2 << 20 {
        dst.copy_from_slice(src);
        return;
    }
    // Persistent pool threads: no spawn per copy.
    cuteafd_transport::copy_pool_parallel(dst, src, 8);
}

/// Waves whose expert input is at least this large go out through the
/// transport's zero-copy egress buffer ([`SparkLink::egress`]); smaller
/// (decode) requests keep their own copy.
const EGRESS_MIN_BYTES: usize = 1 << 20;

fn registered_egress_bytes(request: &cuteafd_transport::ExpertProtocolV2Request) -> Result<usize> {
    let mut geometry = cuteafd_core::expert_geometry();
    geometry.hidden = request.header.hidden_dim;
    ensure!(request.header.row_count > 0 && request.header.route_count % request.header.row_count == 0,
        "egress bootstrap needs a uniform route geometry");
    geometry.topk = request.header.route_count / request.header.row_count;
    geometry.compact_wire_bytes(request.header.row_count,
        request.header.hidden_dtype == cuteafd_transport::ExpertV2Dtype::Bf16)
        .map(|wire| wire.0).context("registered egress geometry overflow")
}

/// Logs once how this coordinator moves expert traffic, from the fabric it
/// detected: large requests leave zero-copy from one registered buffer as
/// chunked WRITEs, partials arrive by the probed intake
/// (`CUTEAFD_SPARK_INTAKE`), and every rank's traffic uses the coordinator's
/// first rail. Measured on raptor (one 400 Gb port, both GPUs a host bridge
/// away from the NIC) with six Sparks: the coordinator's intake (~19 GB/s
/// landed in GPU memory) bounds a wave long before the ranks' links do (six
/// rails of 100-200 Gb), so striping ranks over rails gains nothing there; a
/// split GPU/pinned intake was slower than GPU landing alone.
fn log_transfer_plan(mode: IntakeMode, ranks: usize) {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| match cuteafd_transport::fabric::discover() {
        Ok(report) => {
            let ports = report.rails.rails.iter().map(|rail| format!("{}@{} {:.0}G", rail.device, rail.address,
                rail.effective_gbps)).collect::<Vec<_>>().join(", ");
            tracing::info!(ranks, rdma_tc = report.rdma_traffic_class, dscp = report.rdma_traffic_class >> 2, intake = mode.name(),
                coordinator_rails = %ports, "Spark transfer plan: every rank on the first coordinator rail");
        }
        Err(error) => tracing::warn!("Spark transfer plan: fabric discovery failed: {error:#}"),
    });
}

/// A Spark transport and the intake its waves land in (dropped in that order).
pub(crate) struct SparkLink<'a> {
    intake: std::rc::Rc<SparkIntake<'a>>,
    /// Bytes of the transport's egress buffer (0: none).
    egress_bytes: usize,
    egress_enabled: bool,
    terminal_library: Option<Arc<NativeLibrary>>,
    terminal_released: bool,
    /// The link's persistent exchange worker ([`Self::waves`]), started by
    /// [`Self::bootstrap`]; it owns the transport while it runs.
    exchange: Option<crate::shared::spark_waves::Waves<'a>>,
    /// The transport while no exchange runs (before bootstrap, and while it
    /// is handed back for a reset or terminal teardown).
    transport: Option<SparkExperts>,
    /// Widest wave staging each slot needs (bytes), sized at bootstrap.
    staging_bytes: usize,
    /// The bootstrap request without its (all-zero) hidden payload, and that
    /// payload's length: a reset re-opens the sessions with it before the
    /// exchange restarts, as bootstrap does.
    bootstrap: Option<(cuteafd_transport::ExpertProtocolV2Request, usize)>,
    timing: bool,
    world_size: usize,
    capacity: u32,
}

impl<'a> SparkLink<'a> {
    /// Connects a [`SparkExperts::new_world`] transport whose `capacity`-row
    /// waves land in an intake of the process-wide [`choose_mode`].
    pub(crate) fn new(library: &'a NativeLibrary, world: cuteafd_transport::expert::SparkWorld, peers: &[std::net::SocketAddr], capacity: u32,
        config: cuteafd_transport::TcpTransportConfig, row_bytes: usize) -> Result<Self> {
        Self::new_waves(library, world, peers, capacity, config, row_bytes, 1, capacity)
    }

    /// A link admitting `waves` outstanding waves of up to `capacity` rows,
    /// each with its own return slot ([`SparkIntake::with_waves`]). Waves are
    /// received in dispatch order, with one GPU landing range per slot.
    /// `capacity` bounds every request on the transport; each return slot
    /// holds `wave_rows` (at most `capacity`), the widest wave in flight.
    pub(crate) fn new_waves(library: &'a NativeLibrary, world: cuteafd_transport::expert::SparkWorld,
        peers: &[std::net::SocketAddr], capacity: u32, config: cuteafd_transport::TcpTransportConfig,
        row_bytes: usize, waves: usize, wave_rows: u32) -> Result<Self> {
        ensure!((1..=capacity).contains(&wave_rows), "Spark return slot rows exceed the transport capacity");
        let timing = config.timing;
        let mut transport = SparkExperts::new_world(world, peers, capacity, config, 0)?;
        transport.set_max_waves(waves)?;
        Self::with_egress_waves(library, transport, row_bytes, true, waves, wave_rows, timing)
    }

    /// Return slots.
    pub(crate) fn waves(&self) -> usize { self.intake.waves() }
    pub(crate) fn slots(&self) -> usize { self.intake.slots() }
    pub(crate) fn world_size(&self) -> usize { self.world_size }
    /// The most rows any request on this link may carry.
    pub(crate) fn capacity(&self) -> u32 { self.capacity }
    /// The intake planes' device (the reduce's GPU).
    pub(crate) fn device(&self) -> i32 { self.intake.device() }

    /// Adopts an existing topology-bound or canonical transport without changing
    /// its route contract, rank identities or connection configuration.
    pub(crate) fn from_transport(library: &'a NativeLibrary, transport: SparkExperts,
        row_bytes: usize, timing: bool) -> Result<Self> {
        let capacity = transport.capacity();
        Self::with_egress_waves(library, transport, row_bytes, false, 1, capacity, timing)
    }

    /// [`Self::from_transport`] with `waves` outstanding waves of `wave_rows`
    /// rows (one prefill transport with N waves, PLAN "Prefill contract").
    pub(crate) fn from_transport_waves(library: &'a NativeLibrary, mut transport: SparkExperts,
        row_bytes: usize, waves: usize, wave_rows: u32, timing: bool) -> Result<Self> {
        ensure!((1..=transport.capacity()).contains(&wave_rows), "Spark return slot rows exceed the transport capacity");
        transport.set_max_waves(waves)?;
        Self::with_egress_waves(library, transport, row_bytes, false, waves, wave_rows, timing)
    }

    fn with_egress_waves(library: &'a NativeLibrary, mut transport: SparkExperts,
        row_bytes: usize, enable_egress: bool, waves: usize, wave_rows: u32, timing: bool) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("spark-intake");
        let ranks = transport.world_size();
        let capacity = transport.capacity();
        library.v41_compact_reducer()?.require_plane_reduction(ranks as u32)?;
        let route = || transport.rdma_route();
        let (mode, reason) = choose_mode(library, &route)?.mode_for(ranks * capacity as usize * row_bytes);
        tracing::info!(mode = mode.name(), ranks, capacity, %reason, "Spark transport intake");
        let intake = SparkIntake::with_waves(library, mode, ranks, wave_rows as usize, row_bytes, waves)?;
        // SAFETY: the link drops its transport (and its exchange's) before its
        // intake, and every wave's planes are read only between its landing
        // and its consume, both ordered by the exchange.
        unsafe { intake.attach(&mut transport)? };
        log_transfer_plan(mode, ranks);
        Ok(Self { intake: std::rc::Rc::new(intake), egress_bytes: 0, egress_enabled: enable_egress,
            terminal_library: None, terminal_released: false, exchange: None, transport: Some(transport),
            staging_bytes: 0, bootstrap: None, timing, world_size: ranks, capacity })
    }

    /// The transport while no exchange runs (setup and teardown only).
    fn idle_transport(&mut self) -> Result<&mut SparkExperts> {
        ensure!(self.exchange.is_none(), "the Spark transport belongs to its running exchange");
        self.transport.as_mut().context("Spark link has no transport")
    }

    /// Bind the admitted endpoint before bootstrap starts its exchange worker.
    pub(crate) fn bind_model_endpoint(&mut self, endpoint: u32) -> Result<()> {
        self.idle_transport()?.bind_model_endpoint(endpoint)
    }

    /// Completes every rank's persistent bootstrap without dispatching a wave,
    /// so the workers' immutable installed grouping is readable before the
    /// first request and the rings its widest wave needs are already
    /// registered, then starts the link's exchange worker. Every Spark family
    /// calls this once, right after [`Self::enable_terminal_ownership`] and
    /// before anything plans against the transport.
    ///
    /// `request` must be the widest wave the family dispatches (the one its
    /// start-up warm-up sends): it sizes every rank's registered rings and
    /// each slot's wave staging.
    pub(crate) fn bootstrap(&mut self, request: &cuteafd_transport::ExpertProtocolV2Request) -> Result<()> {
        let staging = crate::shared::experts::staging::RouteStaging::new(request.header.row_count as usize,
            usize::try_from(request.header.route_count / request.header.row_count.max(1))?.max(1),
            request.hidden_payload.len() / (request.header.row_count as usize).max(1))?.bytes;
        let (egress_enabled, egress_bytes, waves) = (self.egress_enabled, self.egress_bytes, self.slots());
        let transport = self.idle_transport()?;
        let mut registered = egress_bytes;
        if egress_enabled && egress_bytes == 0 {
            let bytes = registered_egress_bytes(request)?;
            let _memory_scope = cuteafd_ffi::memory_ledger::scope("spark-intake");
            transport.enable_egress(bytes)?;
            registered = bytes;
            tracing::info!(slot_bytes = bytes, slots = waves, min_bytes = EGRESS_MIN_BYTES,
                registered_bytes = bytes * waves, "Spark registered request egress");
        }
        transport.bootstrap(request)?;
        self.egress_bytes = registered;
        self.staging_bytes = staging;
        let shape = cuteafd_transport::ExpertProtocolV2Request { hidden_payload: bytes::Bytes::new(), ..request.clone() };
        self.bootstrap = Some((shape, request.hidden_payload.len()));
        self.start()
    }

    /// Starts the exchange worker on the idle transport.
    fn start(&mut self) -> Result<()> {
        let intake = &self.intake;
        let transport = self.transport.take().context("Spark link has no transport")?;
        // A link whose registered request ring has a slot per wave, each large
        // enough, stages in those slots: every rank then sends each request
        // straight from its slot (RDMA WRITE chunks or a gathered SEND), with
        // no per-rank copy. Otherwise each slot gets its own pinned arena.
        let egress = match transport.egress_arenas() {
            Ok(egress) => egress,
            Err(error) => { self.transport = Some(transport); return Err(error); }
        };
        let bytes = self.staging_bytes.max(256);
        let slots = intake.slots();
        let registered = egress.len() == slots && egress.iter().all(|arena| arena.bytes() >= bytes);
        let arenas = if registered { Ok(egress) } else {
            let _scope = cuteafd_ffi::memory_ledger::scope("spark-intake");
            (0..slots).map(|_| cuteafd_transport::StagingArena::new(bytes)).collect::<Result<Vec<_>>>()
        };
        let arenas = match arenas {
            Ok(arenas) => arenas,
            Err(error) => { self.transport = Some(transport); return Err(error); }
        };
        tracing::info!(slots, bytes, registered, "Spark exchange staging");
        let landing = transport.gpu_landing_ranks().iter().enumerate()
            .fold(0u8, |bits, (rank, &landed)| if landed { bits | 1 << rank } else { bits });
        let geometry = crate::shared::spark_waves::Intake { ranks: intake.ranks(), row_bytes: intake.row_bytes(),
            ranges: intake.ranges(), geometry: intake.geometry(),
            landing: if intake.mode == IntakeMode::Gpu { landing } else { 0 }, placed: intake.placed() };
        // Read-only worker metadata, kept so callers need not reach the transport.
        let grouping = transport.installed_grouping().unwrap_or_default();
        self.exchange = Some(crate::shared::spark_waves::Waves::start(intake.library, geometry, transport, grouping,
            arenas, self.timing)?);
        Ok(())
    }

    /// The link's exchange: every Spark wave goes through it.
    pub(crate) fn exchange(&mut self) -> Result<&mut crate::shared::spark_waves::Waves<'a>> {
        self.exchange.as_mut().context("Spark link exchange not started (bootstrap first)")
    }

    /// Reuse after cancellation: stops the exchange (its consumers drained),
    /// resets the connections and starts it again.
    pub(crate) fn reset_connections(&mut self) -> Result<()> {
        if let Some(exchange) = self.exchange.take() {
            // A failed exchange already abandoned its waves; a cancelled one
            // gives up the engine's unlanded waves without a failure (the
            // process stays healthy) and releases their waits. Either way the
            // worker hands its transport back (#784).
            let (transport, frames, result) = exchange.cancel::<SparkExperts>()?;
            self.intake.keep_frames(frames);
            self.transport = Some(transport);
            if let Err(error) = result {
                tracing::warn!(error = %format!("{error:#}"), "Spark exchange closed with a failure; resetting its transport");
            }
        }
        let intake = std::rc::Rc::clone(&self.intake);
        let transport = self.transport.as_mut().context("Spark link has no transport")?;
        intake.reset(transport)?;
        // Re-open every rank's session as bootstrap did: the exchange reads
        // which ranks land in device memory from the live sessions.
        if let Some((shape, payload)) = &self.bootstrap {
            let request = cuteafd_transport::ExpertProtocolV2Request {
                hidden_payload: bytes::Bytes::from(vec![0; *payload]), ..shape.clone() };
            transport.bootstrap(&request)?;
        }
        self.start()
    }

    /// Immutable worker metadata from the bootstrap (installed grouping).
    pub(crate) fn installed_grouping(&self) -> Result<Vec<cuteafd_transport::worker_selection::InstalledExpertGrouping>> {
        // The grouping is fixed at bootstrap; the exchange keeps a copy.
        match (&self.exchange, &self.transport) {
            (Some(exchange), _) => Ok(exchange.installed_grouping().to_vec()),
            (None, Some(transport)) => transport.installed_grouping(),
            (None, None) => anyhow::bail!("Spark link has no transport"),
        }
    }

    /// Must precede connection/bootstrap so a temporary failed endpoint can
    /// retain the same external landing and pinned-upload owners as submit.
    pub(crate) fn enable_terminal_ownership(&mut self, library: Arc<NativeLibrary>) -> Result<()> {
        ensure!(std::ptr::eq(library.as_ref(),self.intake.library),"terminal intake must retain its actual native library owner");
        self.intake.library.rdma_rc_endpoint_quiesce_available()?;
        self.idle_transport()?.enable_terminal_ownership()?;
        self.terminal_library = Some(library);
        Ok(())
    }

    /// Terminal teardown, step 1: closes the exchange (releasing every queued
    /// wait), takes its transport back and quiesces the QPs (#784: the
    /// exchange owns the transport once started).
    pub(crate) fn terminal_quiesce(&mut self) -> Result<()> {
        if let Some(exchange) = self.exchange.take() {
            // Teardown may abandon waves; the worker then fails, releases its
            // waits and still hands the transport back.
            let (transport, frames, result) = exchange.close::<SparkExperts>()?;
            self.intake.keep_frames(frames);
            self.transport = Some(transport);
            if let Err(error) = result {
                tracing::warn!(error = %format!("{error:#}"), "Spark exchange closed with a failure; quiescing its transport");
            }
        }
        self.transport.as_mut().context("Spark link has no transport")?.terminal_quiesce()
    }
    pub(crate) fn terminal_drain_copies(&self) -> Result<()> { self.intake.terminal_drain_copies() }
    pub(crate) fn terminal_release(&mut self) -> Result<()> {
        // Validate successful QP quiescence before releasing any held source.
        self.transport.as_mut().context("Spark link has no transport")?.terminal_release()?;
        self.intake.terminal_release_payloads();
        self.terminal_released = true;
        Ok(())
    }
}

impl Drop for SparkLink<'_> {
    fn drop(&mut self) {
        // The exchange joins first; with waves outstanding it fails and
        // releases every queued wait.
        drop(self.exchange.take());
        if self.transport.as_ref().is_some_and(SparkExperts::terminal_owned) && !self.terminal_released {
            // Drop may run inside fallible connection or submission scopes,
            // before the engine regains control. QP quiescence alone cannot
            // prove the compute stream passed these planes/pinned sources.
            // Retain the whole intake until process teardown. The transport
            // separately retains endpoint/library owners if quiescence fails.
            std::mem::forget(self.intake.clone());
            if let Some(library)=self.terminal_library.take() { std::mem::forget(library); }
            tracing::error!("retaining terminal intake whose complete consumer drainage was not proven");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_partial_reserve_keeps_logical_wave_offsets_and_plan_charge() -> Result<()> {
        let (waves, rows, row_bytes, ranks) = (3, 2752, 34_864, 6);
        let geometry = SlotGeometry::new(waves, rows)?;
        let logical = geometry.reserve_rows() * row_bytes;
        let backing = cuteafd_core::serving_capacity::spark_intake_allocation_bytes(logical as u64)?;
        assert!(backing > logical as u64);
        for base in [4096, 4608] {
            let (aligned_offset, registered_bytes) = cuteafd_core::serving_capacity::spark_intake_export_range(base, logical as u64)?;
            assert_eq!((base + aligned_offset) % 4096, 0);
            assert!(aligned_offset + registered_bytes <= backing);
            cuteafd_core::serving_capacity::validate_spark_intake_export(base + aligned_offset, registered_bytes)?;
        }
        for slot in 0..waves {
            let (offset, bytes) = geometry.slice(slot, row_bytes);
            assert_eq!((offset, bytes), (slot * rows * row_bytes, rows * row_bytes));
            assert!((offset + bytes) as u64 <= backing);
        }
        assert_eq!(cuteafd_loader::serving_capacity::prefill_return_bytes(
            waves as u64, rows as u64, ranks, row_bytes as u64 / 2), ranks * backing);
        Ok(())
    }

    fn probe(gpu_gbps: f64) -> GpuLandingProbe {
        GpuLandingProbe {
            rdma_device: "mlx5_0".into(), cuda_device: 0, dma_buf: true, gpudirect_rdma: true,
            writes_ordering: 100, registered: true, gpu_gbps, host_gbps: 52.4, status: String::new(), error: None,
        }
    }

    #[test]
    fn fabric_landing_without_a_peer_needs_a_sole_rdma_address() {
        use cuteafd_transport::fabric::{RdmaPort, RdmaSelection};
        let port = |roce_v2: Vec<(u32, std::net::Ipv4Addr)>| RdmaPort { device: "mlx5_0".into(), port: 1, active: true,
            link_layer: "Ethernet".into(), link_gbps: 400.0, netdev: Some("eno1".into()), mtu: Some(9000), roce_v2,
            subnets: vec![], pci: None };
        let rail = std::net::Ipv4Addr::new(10, 55, 0, 22);
        assert_eq!(sole_rdma_address(&[port(vec![(7, rail)])]).unwrap(),
            RdmaSelection { device: "mlx5_0".into(), port: 1, gid_index: 7 });
        // A management address beside the rail (lower index): no guess.
        let error = sole_rdma_address(&[port(vec![(3, std::net::Ipv4Addr::new(192, 168, 1, 50)), (7, rail)])])
            .unwrap_err().to_string();
        assert!(error.contains("--landing-peer") && error.contains("192.168.1.50"), "{error}");
        assert!(sole_rdma_address(&[port(vec![])]).is_err());
    }

    #[test]
    fn fabric_probe_reports_missing_library_for_each_bounded_repeat() {
        let reports = fabric_probe(Some(std::path::Path::new("/nonexistent-landing-probe-library.so")), 0, true, 3, None);
        assert_eq!(reports.len(), 3);
        for report in reports {
            assert!(report.probe.is_none());
            assert!(report.error.is_some());
            assert!(report.summary().contains("intake auto: pinned"));
        }
    }

    #[test]
    fn registered_egress_matches_widest_bootstrap_request_geometry() -> Result<()> {
        use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor,
            ExpertProtocolV2RouteEntry, ExpertV2SourceKind, ExpertV2Dtype};
        for rows in [64u32, 1024, 2048, 4096] {
            for dtype in [ExpertV2Dtype::Bf16, ExpertV2Dtype::Fp8E4m3Ue8m0K32] {
                let geometry = cuteafd_core::ExpertGeometry::MIMO_V26_PRO;
                let hidden_bytes = if dtype == ExpertV2Dtype::Bf16 { geometry.hidden * 2 }
                    else { geometry.hidden + geometry.hidden / 32 };
                let request = ExpertProtocolV2Request::new(1, 17, 0, geometry.hidden, dtype,
                    (0..rows).map(|row| ExpertProtocolV2RowDescriptor {
                        row_id: u64::from(row), source_kind: ExpertV2SourceKind::Prefill,
                        source_request_id: 1, token_position: u64::from(row),
                        route_offset: row * geometry.topk, route_count: geometry.topk,
                    }).collect(),
                    (0..rows).flat_map(|row| (0..geometry.topk).map(move |expert| ExpertProtocolV2RouteEntry {
                        row_index: row, expert_id: expert, gate_weight: 0.0,
                    })).collect(), vec![0; (rows * hidden_bytes) as usize])?;
                let expected = geometry.compact_wire_bytes(rows, dtype == ExpertV2Dtype::Bf16).unwrap().0;
                assert_eq!(registered_egress_bytes(&request)?, expected);
                assert_eq!(request.encode()?.len(), expected);
            }
        }
        Ok(())
    }

    /// A prefill wave stages its route ids, gate weights and input rows in a
    /// slot of the registered egress ring (spark_waves), so every family's
    /// widest wave must fit the slot its widest bootstrap request sized.
    #[test]
    fn registered_egress_slots_hold_a_whole_prefill_wave_staging() -> Result<()> {
        use cuteafd_core::ExpertGeometry as G;
        for geometry in [G::DEEPSEEK_V41, G::DEEPSEEK_V4_FLASH, G::DEEPSEEK_V4_PRO, G::GLM5, G::GLM5_FLASH,
            G::MIMO_V2_FLASH, G::MIMO_V26_PRO, G::QWEN4, G::KIMI_K3, G::STEP5] {
            for rows in [1u32, 64, 1024, 4096] {
                for bf16 in [false, true] {
                    let (hidden, topk) = (geometry.hidden as usize, geometry.topk as usize);
                    let wire = if bf16 { hidden * 2 } else { hidden + hidden / 32 };
                    let staging = crate::shared::experts::staging::RouteStaging::new(rows as usize, topk, wire)?.bytes;
                    let slot = geometry.compact_wire_bytes(rows, bf16).unwrap().0;
                    assert!(staging <= slot, "{geometry:?} {rows} rows bf16={bf16}: {staging} > {slot}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn upload_bounds_reject_missing_planes_and_overflow_before_pointer_arithmetic() {
        for rank in [0, 1, 2, 4, usize::MAX] {
            assert!(upload_offset(0, rank, 0, 10240, 10240, 10240).is_err());
        }
        assert!(upload_offset(2, 2, 0, 10240, 10240, 10240).is_err());
        assert!(upload_offset(2, 0, 1, 10240, 10240, 10240).is_err());
        assert!(upload_offset(2, 0, 1, usize::MAX, 1, usize::MAX).is_err());
        assert!(upload_offset(2, 0, 2, usize::MAX, 0, usize::MAX).is_err());
        assert_eq!(upload_offset(2, 1, 3, 10240, 20480, 51200).unwrap(), 30720);
    }

    #[test]
    fn auto_lands_on_the_gpu_from_half_the_pinned_path_rate() {
        // raptor GPU0: 19.6 GB/s landing vs 52.4 into host and 57.6 H2D ->
        // 27.4 pinned path, share 0.71: measured faster at 134 and 352 MB waves.
        let pinned = pinned_path_gbps(52.4, 57.6);
        assert!((pinned - 27.4).abs() < 0.1);
        assert!(gpu_wins(&probe(19.6), Some(57.6)).0);
        // Loopback GPU1 per-wave share 0.53 also won end to end.
        assert!(gpu_wins(&probe(0.53 * pinned), Some(57.6)).0);
        // A landing path at a third of the pinned path's rate stays pinned.
        assert!(!gpu_wins(&probe(pinned / 3.0), Some(57.6)).0);
        // Without the H2D probe, a verified landing path is used.
        assert!(gpu_wins(&probe(19.6), None).0);
    }

    #[test]
    fn explicit_gpu_intake_lands_on_the_gpu_only_when_the_probe_can() {
        let measured = probe(19.2);
        for supported in [true, false] {
            let result = if supported { Ok(measured.clone()) }
                else { Err(anyhow::anyhow!("forced unavailable")) };
            let (mode, _) = resolve_landing_probe("gpu", &result, None);
            assert_eq!(mode, if supported { IntakeMode::Gpu } else { IntakeMode::Pinned });
        }
    }

    #[test]
    fn resolved_pinned_intake_cannot_be_upgraded_by_a_usable_probe() {
        let measured = probe(1.0);
        let (mode, reason) = resolve_landing_probe("auto", &Ok(measured.clone()), Some(57.6));
        assert_eq!(mode, IntakeMode::Pinned);
        let choice = IntakeChoice { setting: "auto".into(), mode, reason,
            probe: Some(measured), h2d_gbps: Some(57.6) };
        assert!(choice.probe.as_ref().unwrap().usable());
        assert_eq!(choice.mode_for(4096 * 10240).0, IntakeMode::Pinned);
    }

    #[test]
    fn unavailable_landing_keeps_pinned_intake() {
        for setting in ["auto", "gpu"] {
            let (mode, reason) = resolve_landing_probe(setting, &Err(anyhow::anyhow!("forced unavailable")), None);
            assert!(reason.contains("forced unavailable"));
            let choice = IntakeChoice { setting: setting.into(), mode,
                reason, probe: None, h2d_gbps: None };
            assert_eq!(choice.mode_for(4096 * 10240).0, IntakeMode::Pinned);
        }
    }

    #[test]
    fn mode_for_keeps_the_resolved_mode_for_every_wave_size() {
        let choice = |mode| IntakeChoice { setting: "auto".into(), mode, reason: "probe".into(),
            probe: Some(probe(19.6)), h2d_gbps: Some(57.6) };
        // GLM 5.3 Flash TP4 (134 MB) and V4 Pro TP6 (352 MB) both land on the GPU.
        assert_eq!(choice(IntakeMode::Gpu).mode_for(4 * 4096 * 4096 * 2).0, IntakeMode::Gpu);
        assert_eq!(choice(IntakeMode::Gpu).mode_for(6 * 4096 * 7168 * 2).0, IntakeMode::Gpu);
        assert_eq!(choice(IntakeMode::Pinned).mode_for(1).0, IntakeMode::Pinned);
    }
}
