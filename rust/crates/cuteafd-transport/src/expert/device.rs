//! A Spark transport driven by the GPU (PLAN.md, device-driven Spark
//! exchange). The engine's stream copies a wave's route ids, weights and wire
//! rows into a pinned, device-mapped [`mailbox`] and publishes a sequence
//! number there (`cuteafd_host_signal`); this lane's thread spins on it,
//! builds and posts the request, receives every rank's partials straight
//! into device memory (GPU landing) and publishes the wave's completion
//! sequence, which the stream waits on (`cuteafd_peer_wait`) before its
//! reduce. The inference thread neither synchronizes nor parses per layer,
//! and a step's layers can be queued (and captured) back to back.
//!
//! Waves are strictly sequential on one lane: the stream publishes wave n+1
//! only after it waited for wave n, so one mailbox suffices, and the planes
//! a wave lands in are free again once the stream passed that wave's reduce
//! (the next wave's partials cannot arrive before its request is posted).
use super::SparkExperts;
use crate::{
    DeviceLanding, ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, TcpTransportConfig,
};
use anyhow::{anyhow, ensure, Context, Result};
use std::net::SocketAddr;
use std::sync::atomic::{fence, AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Byte offsets inside the mailbox (pinned, device-mapped host memory).
pub mod mailbox {
    /// u32 sequence the GPU publishes after the wave's payload.
    pub const READY: usize = 0;
    /// u32 sequence the proxy publishes once the wave's partials landed.
    pub const DONE: usize = 64;
    /// u32 [4]: layer, rows, kind ([`super::DeviceWave::kind`]), top-k.
    pub const DESCRIPTOR: usize = 128;
    /// Route ids (u32 [rows * topk]), then gate weights (f32 [rows * topk]).
    pub const ROUTES: usize = 256;

    /// Offset of the gate weights for a mailbox sized for `capacity` rows.
    pub const fn weights(capacity: usize, topk: usize) -> usize {
        ROUTES + capacity * topk * 4
    }

    /// Offset of the wire rows (256-byte aligned).
    pub const fn wire(capacity: usize, topk: usize) -> usize {
        (weights(capacity, topk) + capacity * topk * 4).next_multiple_of(256)
    }

    /// Mailbox bytes for `capacity` rows of `topk` routes and `wire_row_bytes`.
    pub const fn bytes(capacity: usize, topk: usize, wire_row_bytes: usize) -> usize {
        wire(capacity, topk) + capacity * wire_row_bytes
    }
}

/// A wave as the GPU described it in the mailbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceWave {
    pub layer: u32,
    pub rows: u32,
    /// Engine-defined (e.g. the request's source kind).
    pub kind: u32,
}

/// Builds a wave's request on the lane thread from its routes and wire rows.
pub type DeviceBuild =
    Box<dyn FnMut(DeviceWave, Vec<ExpertProtocolV2RouteEntry>, bytes::Bytes) -> Result<ExpertProtocolV2Request> + Send>;

/// Where the proxy's state meets the inference thread.
struct Shared {
    stop: AtomicBool,
    /// A step was queued: spin instead of parking.
    armed: AtomicBool,
    /// Waves completed (each published to the GPU, with or without error).
    completed: AtomicU64,
    /// The first error since the last [`SparkDeviceLane::check`].
    error: Mutex<Option<String>>,
    /// Lane thread busy time, for [`SparkDeviceLane::stats`].
    build_post_ns: AtomicU64,
    receive_ns: AtomicU64,
}

/// Per-wave averages since the lane started.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceLaneStats {
    pub waves: u64,
    /// Mailbox read, request build and RDMA posts, us per wave.
    pub build_post_us: f64,
    /// Post until every rank's partials landed, us per wave.
    pub receive_us: f64,
}

pub struct SparkDeviceLane {
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
    world: usize,
}

/// The pinned mailbox, carried across threads as an integer.
#[derive(Clone, Copy)]
struct Mailbox {
    base: usize,
    bytes: usize,
    capacity: usize,
    topk: usize,
    wire_row_bytes: usize,
}

impl Mailbox {
    fn word(&self, offset: usize) -> &AtomicU32 {
        // SAFETY: `offset` is a 4-byte aligned word inside the live mailbox
        // (checked at spawn); the GPU and this thread access it atomically.
        unsafe { &*((self.base + offset) as *const AtomicU32) }
    }

    fn bytes_at(&self, offset: usize, len: usize) -> Result<&[u8]> {
        ensure!(offset + len <= self.bytes, "mailbox read of {len} bytes at {offset} exceeds {}", self.bytes);
        // SAFETY: inside the live mailbox; the GPU wrote this range before it
        // published the sequence this thread acquired, and rewrites it only
        // after the wave's completion is published.
        Ok(unsafe { std::slice::from_raw_parts((self.base + offset) as *const u8, len) })
    }

    /// The published wave's descriptor, routes and wire rows.
    fn read(&self) -> Result<(DeviceWave, Vec<ExpertProtocolV2RouteEntry>, bytes::Bytes)> {
        let d = |i: usize| self.word(mailbox::DESCRIPTOR + 4 * i).load(Ordering::Relaxed);
        let wave = DeviceWave { layer: d(0), rows: d(1), kind: d(2) };
        let topk = d(3) as usize;
        let rows = wave.rows as usize;
        ensure!(rows > 0 && rows <= self.capacity && topk == self.topk,
            "device wave of {rows} rows x {topk} routes exceeds the mailbox ({} x {})", self.capacity, self.topk);
        let ids = self.bytes_at(mailbox::ROUTES, rows * topk * 4)?;
        let weights = self.bytes_at(mailbox::weights(self.capacity, topk), rows * topk * 4)?;
        let word = |bytes: &[u8], i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        let routes = (0..rows * topk).map(|i| ExpertProtocolV2RouteEntry {
            row_index: (i / topk) as u32,
            expert_id: word(ids, i),
            gate_weight: f32::from_bits(word(weights, i)),
        }).collect();
        let wire = bytes::Bytes::copy_from_slice(
            self.bytes_at(mailbox::wire(self.capacity, topk), rows * self.wire_row_bytes)?);
        Ok((wave, routes, wire))
    }
}

/// How long the proxy keeps spinning after its last wave before it parks.
const SPIN_AFTER: Duration = Duration::from_millis(50);

impl SparkDeviceLane {
    /// Connects a [`SparkExperts::new_ranks`] transport on a new thread whose
    /// waves come from the GPU through `mailbox` (host address of
    /// [`mailbox::bytes`] pinned, device-mapped bytes, zeroed) and land in
    /// `landing` (one device range per rank). A rank that does not land in
    /// device memory has its rows copied there on `cuda_device`. `warm`, a
    /// full-capacity request, is exchanged once at setup (connections and
    /// ring sizes; its partials land in the planes before the engine runs).
    ///
    /// # Safety
    /// The mailbox and every landing range stay allocated until this lane is
    /// dropped. The GPU publishes wave n+1 only after it waited for wave n's
    /// completion and finished reading its planes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn spawn(peers: Vec<SocketAddr>, executors: Vec<u64>, capacity: u32, config: TcpTransportConfig,
        landing: Vec<DeviceLanding>, cuda_device: i32, mailbox_base: usize, mailbox_bytes: usize, topk: usize,
        wire_row_bytes: usize, warm: Option<ExpertProtocolV2Request>, mut build: DeviceBuild) -> Result<Self> {
        ensure!(landing.len() == peers.len(), "a device lane needs one landing range per rank");
        ensure!(mailbox_base % 64 == 0 && mailbox_bytes >= mailbox::bytes(capacity as usize, topk, wire_row_bytes),
            "device lane mailbox is misaligned or smaller than {} bytes", mailbox::bytes(capacity as usize, topk,
            wire_row_bytes));
        let mailbox = Mailbox { base: mailbox_base, bytes: mailbox_bytes, capacity: capacity as usize, topk,
            wire_row_bytes };
        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            armed: AtomicBool::new(false),
            completed: AtomicU64::new(0),
            error: Mutex::new(None),
            build_post_ns: AtomicU64::new(0),
            receive_ns: AtomicU64::new(0),
        });
        let (ready_tx, ready) = std::sync::mpsc::channel::<Result<usize>>();
        let state = Arc::clone(&shared);
        let thread = std::thread::Builder::new().name("spark-device-lane".into()).spawn(move || {
            let setup = (|| -> Result<_> {
                let mut transport = SparkExperts::new_ranks(&peers, &executors, capacity, config)?;
                // SAFETY: forwarded from this constructor's contract.
                unsafe { transport.set_gpu_landing(Some(landing.clone()))? };
                let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
                let library = crate::verbs::load_verbs_host_native_library()?;
                library.cuda_set_device(cuda_device)?;
                if let Some(request) = &warm {
                    let wave = transport.dispatch_wave(request)?;
                    let receipt = runtime.block_on(transport.receive_wave(wave, |_, _, _| Ok(())))?;
                    ensure!(receipt.landed.count_ones() as usize == transport.world_size(),
                        "device lane warm-up: ranks {:#b} of {} landed in GPU memory", receipt.landed,
                        transport.world_size());
                }
                Ok((transport, runtime, library))
            })();
            let (mut transport, runtime, library) = match setup {
                Ok(setup) => setup,
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(transport.world_size()));
            let row_bytes = landing.first().map_or(0, |l| l.bytes / capacity as usize);
            let ready_word = mailbox.word(mailbox::READY);
            let done_word = mailbox.word(mailbox::DONE);
            let mut expected = ready_word.load(Ordering::Acquire);
            done_word.store(expected, Ordering::Release);
            let mut last = Instant::now();
            while !state.stop.load(Ordering::Relaxed) {
                let published = ready_word.load(Ordering::Acquire);
                if published == expected {
                    if state.armed.load(Ordering::Relaxed) || last.elapsed() < SPIN_AFTER {
                        std::hint::spin_loop();
                    } else {
                        std::thread::park_timeout(Duration::from_millis(1));
                    }
                    continue;
                }
                let seen = Instant::now();
                state.armed.store(false, Ordering::Relaxed);
                fence(Ordering::Acquire);
                let result = (|| -> Result<()> {
                    ensure!(published == expected.wrapping_add(1),
                        "device wave sequence {published} skipped past {}", expected.wrapping_add(1));
                    let (wave, routes, wire) = mailbox.read()?;
                    let request = build(wave, routes, wire)?;
                    let pending = transport.dispatch_wave(&request)?;
                    let posted = Instant::now();
                    let rows = wave.rows as usize;
                    runtime.block_on(transport.receive_wave(pending, |rank, first, payload| {
                        // A rank that could not register its plane: copy its
                        // rows there (synchronously; a fallback, not the plan).
                        let at = first as usize * row_bytes;
                        let plane = landing.get(rank).context("partial from an unknown rank")?;
                        ensure!(at + payload.len() <= (rows * row_bytes).min(plane.bytes),
                            "rank {rank} partial rows run past its plane");
                        let target = cuteafd_ffi::CuteafdDeviceBuffer {
                            ptr: (plane.ptr + at) as *mut std::ffi::c_void, bytes: payload.len(), ..Default::default()
                        };
                        library.copy_h2d(target, payload)
                    }))?;
                    let finished = Instant::now();
                    state.build_post_ns.fetch_add((posted - seen).as_nanos() as u64, Ordering::Relaxed);
                    state.receive_ns.fetch_add((finished - posted).as_nanos() as u64, Ordering::Relaxed);
                    Ok(())
                })();
                if let Err(error) = result {
                    tracing::error!("device Spark wave failed: {error:#}");
                    let mut slot = state.error.lock().unwrap_or_else(|p| p.into_inner());
                    slot.get_or_insert_with(|| format!("{error:#}"));
                    transport.reset_connections();
                }
                expected = published;
                // Release: the landed planes (and this thread's copies) are
                // complete before the GPU sees the sequence.
                done_word.store(expected, Ordering::Release);
                state.completed.fetch_add(1, Ordering::Release);
                last = Instant::now();
            }
        })?;
        let world = ready.recv().map_err(|_| anyhow!("device lane thread exited during setup"))??;
        Ok(Self { shared, thread: Some(thread), world })
    }

    pub fn world_size(&self) -> usize {
        self.world
    }

    /// Before queuing a step's waves: the proxy spins until it sees the next
    /// one (it parks after [`SPIN_AFTER`] without waves). Returns the error of
    /// an earlier wave, if any.
    pub fn arm(&self) -> Result<()> {
        self.check()?;
        self.shared.armed.store(true, Ordering::Relaxed);
        if let Some(thread) = &self.thread {
            thread.thread().unpark();
        }
        Ok(())
    }

    /// The first wave error since the last check (its partials are garbage:
    /// the step that waited for it must be discarded).
    pub fn check(&self) -> Result<()> {
        let mut slot = self.shared.error.lock().unwrap_or_else(|p| p.into_inner());
        match slot.take() {
            Some(error) => Err(anyhow!("device Spark exchange: {error}")),
            None => Ok(()),
        }
    }

    /// Waves completed since the lane started.
    pub fn completed(&self) -> u64 {
        self.shared.completed.load(Ordering::Acquire)
    }

    pub fn stats(&self) -> DeviceLaneStats {
        let waves = self.completed();
        let per = |ns: &AtomicU64| ns.load(Ordering::Relaxed) as f64 / 1e3 / waves.max(1) as f64;
        DeviceLaneStats { waves, build_post_us: per(&self.shared.build_post_ns), receive_us: per(&self.shared.receive_ns) }
    }
}

impl Drop for SparkDeviceLane {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mailbox_layout_keeps_routes_and_rows_apart() {
        // DeepSeek V4 Flash decode: 64 rows x 6 routes, 4096 + 128 wire bytes.
        let (capacity, topk, wire) = (64, 6, 4096 + 128);
        assert_eq!(mailbox::weights(capacity, topk), mailbox::ROUTES + 64 * 6 * 4);
        assert!(mailbox::wire(capacity, topk) >= mailbox::weights(capacity, topk) + 64 * 6 * 4);
        assert_eq!(mailbox::wire(capacity, topk) % 256, 0);
        assert_eq!(mailbox::bytes(capacity, topk, wire), mailbox::wire(capacity, topk) + 64 * wire);
        assert!(mailbox::DONE >= mailbox::READY + 64 && mailbox::DESCRIPTOR >= mailbox::DONE + 64);
    }

    #[test]
    fn mailbox_reads_the_published_wave() -> Result<()> {
        let (capacity, topk, wire_row) = (4usize, 2usize, 16usize);
        let mut storage = vec![0u64; mailbox::bytes(capacity, topk, wire_row).div_ceil(8)];
        let base = storage.as_mut_ptr() as usize;
        let mb = Mailbox { base, bytes: storage.len() * 8, capacity, topk, wire_row_bytes: wire_row };
        for (i, v) in [7u32, 2, 1, 2].into_iter().enumerate() {
            mb.word(mailbox::DESCRIPTOR + 4 * i).store(v, Ordering::Relaxed);
        }
        for i in 0..4 {
            mb.word(mailbox::ROUTES + 4 * i).store(10 + i as u32, Ordering::Relaxed);
            mb.word(mailbox::weights(capacity, topk) + 4 * i).store((0.5f32 * i as f32).to_bits(), Ordering::Relaxed);
        }
        // SAFETY: inside `storage`, which outlives `mb`.
        unsafe { std::ptr::write_bytes((base + mailbox::wire(capacity, topk)) as *mut u8, 3, 2 * wire_row) };
        let (wave, routes, wire) = mb.read()?;
        assert_eq!(wave, DeviceWave { layer: 7, rows: 2, kind: 1 });
        assert_eq!(routes.len(), 4);
        assert_eq!((routes[3].row_index, routes[3].expert_id, routes[3].gate_weight), (1, 13, 1.5));
        assert_eq!(wire.len(), 2 * wire_row);
        assert!(wire.iter().all(|&b| b == 3));
        mb.word(mailbox::DESCRIPTOR + 4).store(5, Ordering::Relaxed);
        assert!(mb.read().is_err(), "rows past the mailbox capacity are rejected");
        drop(storage);
        Ok(())
    }
}
