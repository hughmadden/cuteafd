//! Per-engine device counters and an independent, sleeping snapshot worker.
use crate::*;
use cuteafd_core::expert_usage::{Geometry, Reader, Snapshot};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

type Tally =
    unsafe extern "C" fn(*const u32, *mut u64, u32, u32, u32, u32, *const i32, *mut c_void) -> i32;
type SnapshotFn = unsafe extern "C" fn(*mut u64, *mut u64, u32, *mut c_void) -> i32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsagePhase {
    Prefill,
    Decode,
}

/// Only the canonical router calls this descriptor, with its real row count.
/// Replicated TP2 routers pass `canonical = false` and enqueue no work.
#[derive(Clone, Copy)]
pub struct RouteTally<'a> {
    owner: &'a RawOwner,
    layer: usize,
    phase: UsagePhase,
}
impl RouteTally<'_> {
    /// # Safety
    /// `ids` has `rows * topk` U32 entries on the owner's device, ordered before
    /// this call on `stream`; the engine drains its graphs before dropping usage.
    /// Optional `real_rows` points at a live device I32 count for padded graphs.
    pub unsafe fn record(
        &self,
        ids: *const c_void,
        rows: usize,
        real_rows: Option<*const i32>,
        mask: u32,
        canonical: bool,
        stream: *mut c_void,
    ) -> Result<()> {
        if !canonical || rows == 0 {
            return Ok(());
        }
        ensure!(
            !ids.is_null() && real_rows.is_none_or(|p| !p.is_null()),
            "null usage route operand"
        );
        let g = &self.owner.geometry;
        let offset =
            (usize::from(self.phase == UsagePhase::Decode) * g.layers + self.layer) * g.experts;
        // SAFETY: descriptor construction validates layer, and allocation has both phase tables.
        let counts = unsafe { self.owner.counters.ptr.cast::<u64>().add(offset) };
        // SAFETY: device pointers/stream lifetime are the caller's contract, ABI is fixed above.
        let status = unsafe {
            (self.owner.tally)(
                ids.cast(),
                counts,
                u32::try_from(rows)?,
                u32::try_from(g.topk)?,
                u32::try_from(g.experts)?,
                mask,
                real_rows.unwrap_or(std::ptr::null()),
                stream,
            )
        };
        ensure!(status == 0, "expert usage tally failed: {status}");
        Ok(())
    }
}

struct RawOwner {
    _module: Arc<Library>,
    geometry: Geometry,
    counters: CuteafdDeviceBuffer,
    snapshot: CuteafdDeviceBuffer,
    host: CuteafdHostBuffer,
    stream: *mut c_void,
    event: *mut c_void,
    device: i32,
    tally: Tally,
    read: SnapshotFn,
    set_device: unsafe extern "C" fn(i32) -> CuteafdStatus,
    free_device: FreeDeviceBufferFn,
    free_host: FreeHostBufferFn,
    destroy_stream: CudaStreamDestroyFn,
    copy: CopyD2HAsyncFn,
    record_event: CudaEventRecordFn,
    wait_event: CudaEventSynchronizeFn,
    destroy_event: CudaEventDestroyFn,
}
// SAFETY: pointers remain owned until the worker joins. Counter access is atomic;
// only the worker touches snapshot/host storage, on its private device stream.
unsafe impl Send for RawOwner {}
// SAFETY: serving only reads immutable metadata and atomically adds device counters.
unsafe impl Sync for RawOwner {}
impl RawOwner {
    fn sample(&self, elapsed_s: f64) -> Result<Snapshot> {
        // SAFETY: all symbols belong to the retained module, each buffer is owned
        // here, and the private snapshot stream has exactly one worker.
        unsafe {
            ensure!(
                (self.set_device)(self.device) == 0,
                "usage snapshot device selection failed"
            );
            ensure!(
                (self.read)(
                    self.counters.ptr.cast(),
                    self.snapshot.ptr.cast(),
                    u32::try_from(self.geometry.cells().unwrap() * 2)?,
                    self.stream
                ) == 0,
                "usage snapshot kernel failed"
            );
            ensure!(
                (self.copy)(self.host.ptr, self.snapshot, self.host.bytes, self.stream) == 0,
                "usage snapshot D2H failed"
            );
            let recorded = (self.record_event)(self.event, self.stream);
            let waited = if recorded == 0 {
                (self.wait_event)(self.event)
            } else {
                recorded
            };
            ensure!(waited == 0, "usage snapshot completion failed: {waited}");
            let cells = self.geometry.cells().unwrap();
            let words = std::slice::from_raw_parts(self.host.ptr.cast::<u64>(), 2 * cells);
            Ok(Snapshot {
                elapsed_s,
                prefill: words[..cells].to_vec(),
                decode: words[cells..].to_vec(),
            })
        }
    }
}
impl Drop for RawOwner {
    fn drop(&mut self) {
        // SAFETY: worker has joined and the engine drained all counter users.
        // Construction failures have no serving users and no snapshot work.
        unsafe {
            if (self.set_device)(self.device) != 0 {
                return;
            }
            if !self.stream.is_null() {
                let _ = (self.destroy_stream)(self.stream);
            }
            if !self.event.is_null() {
                let _ = (self.destroy_event)(self.event);
            }
            for buffer in [&mut self.snapshot, &mut self.counters] {
                if !buffer.ptr.is_null() {
                    memory_ledger::record_free(buffer.ptr as usize);
                    let _ = (self.free_device)(buffer);
                }
            }
            if !self.host.ptr.is_null() {
                memory_ledger::record_free(self.host.ptr as usize);
                let _ = (self.free_host)(&mut self.host);
            }
        }
    }
}

pub struct ExpertUsage<'a> {
    library: &'a NativeLibrary,
    raw: Option<Arc<RawOwner>>,
    stop: mpsc::Sender<()>,
    worker: Option<JoinHandle<()>>,
    reader: Reader,
}
impl<'a> ExpertUsage<'a> {
    pub fn new(library: &'a NativeLibrary, geometry: Geometry, reader: Reader) -> Result<Self> {
        ensure!(geometry.validate(), "invalid expert usage geometry");
        let bytes = geometry
            .table_bytes()
            .context("expert usage table size overflow")?;
        ensure!(
            geometry.cells().unwrap() * 2 <= u32::MAX as usize,
            "usage geometry exceeds native indexing"
        );
        let _scope = memory_ledger::scope("expert-usage");
        // SAFETY: symbols use the native ABI defined in expert_usage.cu; all
        // function pointers stay private and the raw owner retains the DSO.
        let mut raw = RawOwner {
            _module: Arc::clone(&library.lib),
            geometry,
            counters: Default::default(),
            snapshot: Default::default(),
            host: Default::default(),
            stream: std::ptr::null_mut(),
            event: std::ptr::null_mut(),
            device: library.cuda_get_device()?,
            tally: unsafe { *library.lib.get(b"cuteafd_expert_usage_tally")? },
            read: unsafe { *library.lib.get(b"cuteafd_expert_usage_snapshot")? },
            set_device: library.cuteafd_cuda_set_device()?,
            free_device: library.cuteafd_free_device_buffer()?,
            free_host: library.cuteafd_free_host_buffer()?,
            destroy_stream: library.cuteafd_cuda_stream_destroy()?,
            copy: library.cuteafd_copy_d2h_async()?,
            record_event: library.cuteafd_cuda_event_record()?,
            wait_event: library.cuteafd_cuda_event_synchronize()?,
            destroy_event: library.cuteafd_cuda_event_destroy()?,
        };
        raw.counters = library.alloc_device_buffer(bytes)?;
        library.cuda_zero_bytes(raw.counters, bytes)?;
        raw.snapshot = library.alloc_device_buffer(bytes)?;
        raw.host = library.alloc_host_buffer(bytes)?;
        raw.stream = library.cuda_stream_create()?;
        raw.event = library.cuda_event_create_blocking()?;
        reader
            .initialize(raw.geometry.clone())
            .map_err(anyhow::Error::msg)?;
        let (stop, receive) = mpsc::channel();
        let raw = Arc::new(raw);
        let work = Arc::clone(&raw);
        let output = reader.clone();
        let worker = std::thread::Builder::new()
            .name("expert-usage".into())
            .stack_size(cuteafd_core::expert_usage::WORKER_STACK_BYTES)
            .spawn(move || {
                let start = Instant::now();
                while matches!(
                    receive.recv_timeout(Duration::from_secs(1)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    let result = work
                        .sample(start.elapsed().as_secs_f64())
                        .and_then(|sample| output.publish(sample).map_err(anyhow::Error::msg));
                    if let Err(error) = result {
                        output.fail(format!("{error:#}"));
                        // A failed completion must not free an in-flight snapshot.
                        std::mem::forget(work);
                        return;
                    }
                }
            })?;
        Ok(Self {
            library,
            raw: Some(raw),
            stop,
            worker: Some(worker),
            reader,
        })
    }
    pub fn reader(&self) -> Reader {
        self.reader.clone()
    }
    pub fn routes(&self, layer: usize, phase: UsagePhase) -> Result<RouteTally<'_>> {
        let raw = self.raw.as_ref().unwrap();
        ensure!(
            raw.geometry.routed_layers.binary_search(&layer).is_ok(),
            "usage layer {layer} is not routed"
        );
        Ok(RouteTally {
            owner: raw,
            layer,
            phase,
        })
    }
}
impl Drop for ExpertUsage<'_> {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        if self.library.is_quarantined_after_failed_drain() {
            if let Some(raw) = self.raw.take() {
                std::mem::forget(raw);
            }
        }
    }
}
