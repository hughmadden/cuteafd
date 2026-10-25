//! Mapped-table gathers use fixed startup staging. The issuer reserves an
//! owned slot at begin, then queues ready -> mapped-alias SM copy -> consumed.
//! Only the host gather worker waits for slot reuse; it never calls CUDA.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{GatherPool, MappedTable, TablePrefetcher, TableStatsSnapshot};
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::memory::HostAllocation;
#[path = "mapped_table/control.rs"]
mod control;
pub(crate) use control::{RowBatch, RowSource};

#[derive(Debug, Clone, thiserror::Error)]
pub(crate) enum MappedRowsError {
    #[error("mapped-table admission exceeded: {0} unsubmitted gathers")]
    Admission(usize),
    #[error("mapped-table unit {0} is missing or was already begun")]
    Unit(usize),
    #[error("mapped-table gather worker stopped")]
    Stopped,
    #[error("mapped-table gather panicked")]
    Panicked,
    #[error("mapped-table gather failed: {0}")]
    Gather(String),
    #[error("mapped-table rows were not completed before readback")]
    Incomplete,
}

/// Mapped-table options a family engine takes.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct MappedTableArgs {
    /// Host threads gathering mapped rows.
    #[arg(long = "table-threads", alias = "ple-threads", default_value_t = 16)]
    pub threads: usize,
    /// Advise the page cache about rows known ahead of their step.
    #[arg(long = "table-prefetch", default_value_t = true, action = clap::ArgAction::Set)]
    pub prefetch: bool,
}

const LINE: usize = 64;
#[derive(Clone, Copy)]
struct Words { base: usize }
impl Words {
    fn ptr(self, slot: usize, consumed: bool) -> *mut u32 {
        (self.base + (2 * slot + usize::from(consumed)) * LINE) as *mut u32
    }
    fn load(self, slot: usize, consumed: bool) -> u32 {
        // SAFETY: aligned words in table-owned mapped pinned memory, retained
        // through worker join and the last consumer event. GPU uses stream ops.
        unsafe { (*self.ptr(slot, consumed).cast::<AtomicU32>()).load(Ordering::Acquire) }
    }
    fn ready(self, usage: control::Use) {
        // SAFETY: as load; the Release publishes the completed staging writes.
        unsafe { (*self.ptr(usage.slot, false).cast::<AtomicU32>()).store(usage.sequence, Ordering::Release) }
    }
}

struct RingSlot<'a> {
    host: HostAllocation<'a>,
    alias: CuteafdDeviceBuffer,
    event: *mut c_void,
    pending: bool,
}

impl Drop for RingSlot<'_> {
    fn drop(&mut self) {
        if !self.host.library.is_quarantined_after_failed_drain() {
            // SAFETY: constructor rollback has no readers; normal table Drop
            // drains consumers and joins its worker before dropping the slots.
            if let Err(error) = unsafe { self.host.library.cuda_event_destroy(self.event) } {
                tracing::error!(%error, "destroying mapped-table event");
            }
        }
    }
}

struct GatherBackend {
    table: Arc<MappedTable>,
    pool: GatherPool,
    hosts: Vec<(usize, usize)>,
    words: Words,
}
impl control::Backend for GatherBackend {
    fn consumed(&self, slot: usize) -> u32 { self.words.load(slot, true) }
    fn gather(&mut self, slot: usize, ids: &[i64]) -> std::result::Result<(), MappedRowsError> {
        let rows: Vec<_> = ids.iter().map(|&id| id as u64).collect();
        let bytes = rows.len() * self.table.row_bytes();
        let (host, capacity) = self.hosts[slot];
        assert!(bytes <= capacity, "validated gather exceeds staging");
        // SAFETY: this worker alone writes the slot, after its previous GPU
        // consumed store. The table retains staging until this worker joins.
        let output = unsafe { std::slice::from_raw_parts_mut(host as *mut u8, bytes) };
        self.pool.gather(&self.table, &rows, output).map(|_| ())
            .map_err(|error| MappedRowsError::Gather(error.to_string()))
    }
    fn clear(&mut self, slot: usize, rows: usize) {
        let bytes = rows * self.table.row_bytes();
        // SAFETY: same exclusive slot ownership as gather, including unwind.
        unsafe { std::ptr::write_bytes(self.hosts[slot].0 as *mut u8, 0, bytes); }
    }
    fn publish(&self, usage: control::Use) { self.words.ready(usage); }
}

/// Only batch status is shared with readback, never a raw staging pointer.
/// Errors retire after their step's mandatory drain, so later requests remain
/// usable after a read error. A batched step fails all requests it contains.
#[derive(Default)]
struct Readbacks(Mutex<Vec<Arc<RowBatch>>>);
impl Readbacks {
    fn check(&self, device: i32, stream: usize) -> Result<()> {
        let mut batches = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let mut error = None;
        for batch in batches.iter() {
            if let Err(failure) = batch.readback(device, stream) {
                if error.is_none() { error = Some(failure); }
            }
        }
        batches.retain(|batch| !batch.retired());
        error.map_or(Ok(()), |error| Err(error.into()))
    }
}
fn readbacks() -> &'static Mutex<Vec<Weak<Readbacks>>> {
    static READBACKS: OnceLock<Mutex<Vec<Weak<Readbacks>>>> = OnceLock::new();
    READBACKS.get_or_init(Mutex::default)
}

/// Call after the existing consumer-stream drain, before publishing any rows.
/// No CUDA calls or waits are added at this boundary.
pub(crate) fn check_readback(device: i32, stream: *mut c_void) -> Result<()> {
    let mut registry = readbacks().lock().unwrap_or_else(|p| p.into_inner());
    registry.retain(|entry| entry.strong_count() > 0);
    let mut error = None;
    for owner in registry.iter().filter_map(Weak::upgrade) {
        if let Err(failure) = owner.check(device, stream as usize) {
            if error.is_none() { error = Some(failure); }
        }
    }
    error.map_or(Ok(()), Err)
}

/// A begin-time lease. Drop cancels without waiting for page faults or CUDA.
pub(crate) struct PendingRows {
    lease: control::Lease,
    bytes: usize,
}

pub(crate) struct MappedTableDevice<'a> {
    library: &'a NativeLibrary,
    device: i32,
    name: &'static str,
    table: Arc<MappedTable>,
    prefetcher: Option<TablePrefetcher>,
    control: Arc<control::Control>,
    worker: Option<std::thread::JoinHandle<()>>,
    slots: RefCell<Vec<RingSlot<'a>>>,
    _words: HostAllocation<'a>,
    words: Words,
    prefill_rows: usize,
    decode_rows: usize,
    ledger: Arc<Readbacks>,
    logged: Cell<TableStatsSnapshot>,
}

impl<'a> MappedTableDevice<'a> {
    /// Startup staging charge is exactly (k * prefill_rows + decode_rows) *
    /// row_bytes. k is the admitted deployment's wave count, not its ceiling.
    pub fn new(library: &'a NativeLibrary, name: &'static str, table: MappedTable, args: &MappedTableArgs,
        prefill_rows: usize, prefill_slots: usize, decode_rows: usize) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("mapped-table");
        ensure!(prefill_rows > 0 && decode_rows > 0 && (1..=4).contains(&prefill_slots),
            "invalid mapped-table staging geometry");
        let device = library.cuda_get_device()?;
        table.name_stats(name);
        let max_rows = prefill_rows.max(decode_rows);
        let prefetcher = if args.prefetch {
            Some(TablePrefetcher::new(&format!("{name}-prefetch"), 16, max_rows, 2 * max_rows)?)
        } else { None };
        let mut slots = Vec::new();
        for rows in std::iter::repeat_n(prefill_rows, prefill_slots).chain(std::iter::once(decode_rows)) {
            let bytes = rows.checked_mul(table.row_bytes()).context("mapped-table staging overflow")?;
            let host = HostAllocation::new(library, bytes)?;
            let alias = library.cuda_host_buffer_device_alias(host.buffer)?;
            slots.push(RingSlot { host, alias, event: library.cuda_event_create_ordering()?, pending: false });
        }
        let words_owner = HostAllocation::new(library, slots.len() * 2 * LINE)?;
        let words = Words { base: words_owner.buffer.ptr as usize };
        // Preload the copy kernel before any not-yet-ready stream wait. CUDA
        // lazy loading can otherwise wait for the very stream we just blocked.
        let output = super::memory::DeviceAllocation::new(library, 1)?;
        let warm = super::spark_intake::WarmStream::new(library)?;
        // SAFETY: both regions hold one byte and stay live until warm drains,
        // including a partial enqueue failure.
        let queued = unsafe { library.sm_copy_regions(&[(output.buffer.ptr, slots[0].alias.ptr, 1)], warm.raw()) };
        let drained = warm.finish();
        if drained.is_err() { library.quarantine_module_after_failed_drain(); }
        queued.and(drained)?;
        let table = Arc::new(table);
        let pool = GatherPool::new(&format!("{name}-gather"), args.threads)?;
        let backend = GatherBackend { table: Arc::clone(&table), pool,
            hosts: slots.iter().map(|slot| (slot.host.buffer.ptr as usize, slot.host.buffer.bytes)).collect(), words };
        let (control, receive) = control::Control::new(prefill_slots);
        let worker_control = Arc::clone(&control);
        let worker = std::thread::Builder::new().name(format!("{name}-gather-owner"))
            .spawn(move || control::run(worker_control, receive, backend))?;
        let ledger = Arc::new(Readbacks::default());
        readbacks().lock().unwrap_or_else(|p| p.into_inner()).push(Arc::downgrade(&ledger));
        Ok(Self { library, device, name, table: Arc::clone(&table), prefetcher, control, worker: Some(worker),
            slots: RefCell::new(slots), _words: words_owner, words, prefill_rows, decode_rows, ledger,
            logged: Cell::new(table.stats().snapshot()) })
    }

    /// Retain existing executor-owned unit tables; no copied row-ID job queue.
    pub fn batch(&self, rows: Arc<dyn RowSource>) -> Arc<RowBatch> {
        self.control.batch(rows)
    }

    pub fn begin_unit(&self, batch: &Arc<RowBatch>, unit: usize, decode: bool) -> Result<PendingRows> {
        let rows = batch.rows(unit);
        let capacity = if decode { self.decode_rows } else { self.prefill_rows };
        ensure!(rows.len() <= capacity, "{} {} rows exceed the step's {capacity}", self.name, rows.len());
        ensure!(rows.iter().all(|&row| row >= 0 && (row as u64) < self.table.rows()),
            "{} gather row outside the table", self.name);
        let bytes = rows.len().checked_mul(self.table.row_bytes()).context("mapped-table gather size overflow")?;
        let lease = self.control.begin(Arc::clone(batch), unit, decode)?;
        Ok(PendingRows { lease, bytes })
    }

    /// Ordinary decode or non-pipelined prefill also owns its input tables.
    pub fn begin(&self, rows: Arc<dyn RowSource>, decode: bool) -> Result<PendingRows> {
        let batch = self.batch(rows);
        self.begin_unit(&batch, 0, decode)
    }

    /// Queue ready -> SM copy -> consumed without any gather/reuse host wait.
    ///
    /// # Safety
    /// dst and stream belong to this table's device and remain live through
    /// consumption. All uses of a staging slot have one ordered stream history.
    pub unsafe fn finish(&self, mut pending: PendingRows, dst: CuteafdDeviceBuffer, stream: *mut c_void)
        -> Result<usize> {
        ensure!(dst.device_id == self.device && dst.bytes >= pending.bytes, "mapped-table destination geometry");
        let usage = pending.lease.usage;
        let mut slots = self.slots.borrow_mut();
        let slot = &mut slots[usage.slot];
        // Include the batch before enqueue: even a partial enqueue must retain
        // its owned IDs/status through the error-only drain below.
        {
            let mut batches = self.ledger.0.lock().unwrap_or_else(|p| p.into_inner());
            if !batches.iter().any(|batch| Arc::ptr_eq(batch, &pending.lease.batch)) {
                batches.push(Arc::clone(&pending.lease.batch));
            }
        }
        let queued = (|| -> Result<()> {
            // SAFETY: the table retains staging/words; the worker publishes
            // every issued generation, including failures over zeroed bytes.
            unsafe {
                self.library.cuda_stream_wait_value32(stream, self.words.ptr(usage.slot, false), usage.sequence)?;
                if pending.bytes > 0 {
                    self.library.sm_copy_regions(&[(dst.ptr, slot.alias.ptr, pending.bytes as u64)], stream)?;
                }
                self.library.cuda_stream_write_value32(stream, self.words.ptr(usage.slot, true), usage.sequence)?;
                self.library.cuda_event_record(slot.event, stream)?;
            }
            Ok(())
        })();
        if let Err(error) = queued {
            // A ready wait may already be queued: cancellation must not suppress
            // this generation's worker publication, even if its drain fails.
            pending.lease.submit(self.device, stream as usize);
            pending.lease.failed_enqueue();
            // SAFETY: only the failure path drains. Never host-publish consumed
            // until a partially queued copy has provably stopped reading.
            if unsafe { self.library.cuda_stream_synchronize(stream) }.is_err() {
                self.library.quarantine_module_after_failed_drain();
                return Err(error);
            }
            // SAFETY: no GPU reader remains; advance a possibly missing store.
            unsafe { (*self.words.ptr(usage.slot, true).cast::<AtomicU32>()).store(usage.sequence, Ordering::Release); }
            return Err(error);
        }
        slot.pending = true;
        pending.lease.submit(self.device, stream as usize);
        Ok(pending.bytes)
    }

    /// Owner-hop execution may publish logits on a different device. Those
    /// hops already order the PLE consumer before this terminal stream.
    pub fn bind_readback(&self, device: i32, stream: *mut c_void) {
        for batch in self.ledger.0.lock().unwrap_or_else(|p| p.into_inner()).iter() {
            batch.bind_readback(device, stream as usize);
        }
    }

    pub fn prefetch(&self, ids: &[i64]) -> Result<()> {
        let Some(prefetcher) = &self.prefetcher else { return Ok(()) };
        if ids.is_empty() { return Ok(()); }
        let rows = ids.iter().map(|&id| u64::try_from(id).context("negative table row")).collect::<Result<Vec<_>>>()?;
        if !prefetcher.submit_detached(self.table.clone(), &rows)? { self.table.stats().record_prefetch_dropped(); }
        Ok(())
    }

    pub fn log_interval(&self, event: &str) {
        let now = self.table.stats().snapshot();
        let d = now.since(&self.logged.replace(now));
        if d.rows == 0 && d.prefetch_jobs == 0 { return; }
        tracing::info!(table = self.name, event, rows = d.rows, gathers = d.gathers, mib = d.bytes as f64 / (1 << 20) as f64,
            gather_ms = d.gather_ns as f64 / 1e6, stall_ms = d.stall_ns as f64 / 1e6, major_faults = d.major_faults,
            minor_faults = d.minor_faults, resident = d.resident_rate(), prefetch_pages = d.prefetch_pages,
            prefetch_dropped = d.prefetch_dropped, "mapped table");
    }
}

impl Drop for MappedTableDevice<'_> {
    fn drop(&mut self) {
        // Keep the worker publishing until every queued reader finishes. Only
        // then close/join it, before pinned bytes and control words are freed.
        let drained = (|| -> Result<()> {
            for slot in self.slots.get_mut().iter().filter(|slot| slot.pending) {
                // SAFETY: the last ordered use owns this live event.
                unsafe { self.library.cuda_event_synchronize(slot.event)?; }
            }
            Ok(())
        })();
        if let Err(error) = drained {
            tracing::error!(%error, "draining mapped-table rows; retaining staging");
            self.library.quarantine_module_after_failed_drain();
        }
        self.control.close();
        if self.worker.take().is_some_and(|worker| worker.join().is_err()) {
            self.library.quarantine_module_after_failed_drain();
        }
    }
}
