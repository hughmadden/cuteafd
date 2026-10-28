//! Bounded background engram gathers on the shared [`GatherWorker`]: reusable
//! staging slots and request-owned cancellation.
use crate::{EngramBatchStaging, EngramGatherView, EngramTable, GatherFailure, GatherLease, GatherPoll, GatherTicket,
    GatherWorker};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{EngramBatch, ENGRAM_ROWS};
use std::sync::Arc;
use std::time::Instant;

pub const ENGRAM_DEVICE_ROW_BYTES: usize = 24 * (256 + 8 + 512) + 1;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EngramStagingError {
    #[error("unsupported Engram shape: {units} units of {rows} rows")]
    Shape { units: usize, rows: usize },
    #[error("Engram shape needs {required} host slots, got {available}")]
    Slots { required: usize, available: usize },
}

/// Shape is a requirement, not a memory tuning knob. The executor can begin k
/// units before a slot returns, and layer 1 follows begin immediately. Start a
/// full k-unit burst ahead (including the initial burst before any begin).
/// Each active unit can retain L1 and L14 between gather and H2D completion;
/// k lookahead units retain only L1: live maximum = 2k + k. A stalled copy may
/// attain this maximum; reducing it lets lookahead starve an active L14.
/// OS fault latency is unbounded: readiness remains a correctness fence even
/// with the mandatory burst of lead time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngramStagingShape { rows: usize, units: usize }
impl EngramStagingShape {
    pub fn new(rows: usize, units: usize) -> std::result::Result<Self, EngramStagingError> {
        if !(1..=4096).contains(&rows) || !(1..=4).contains(&units) {
            return Err(EngramStagingError::Shape { rows, units });
        }
        Ok(Self { rows, units })
    }
    pub fn rows(self) -> usize { self.rows }
    pub fn units(self) -> usize { self.units }
    pub fn decode_slots(self) -> usize { 2 * 2 }
    pub fn decode_rows(self) -> usize { self.rows.min(256) }
    pub fn decode_host_resident_bytes(self) -> usize {
        self.decode_slots() * EngramBatchStaging::storage_bytes(self.decode_rows()).expect("validated shape")
    }
    pub fn decode_host_workspace_bytes(self) -> usize {
        EngramBatchStaging::worker_bytes(self.decode_rows()).expect("validated shape")
    }
    pub fn decode_host_bytes(self) -> usize { self.decode_host_resident_bytes() + self.decode_host_workspace_bytes() }
    pub fn prefetch_units(self) -> usize { self.units }
    pub fn host_slots(self) -> usize { 2 * self.units + self.prefetch_units() }
    pub fn host_resident_bytes(self) -> usize {
        self.host_slots() * EngramBatchStaging::storage_bytes(self.rows).expect("validated shape")
    }
    pub fn host_workspace_bytes(self) -> usize { EngramBatchStaging::worker_bytes(self.rows).expect("validated shape") }
    /// Exact persistent pool plus a documented upper bound for its one worker.
    /// Decode has its own pool/worker, so concurrent decode cannot starve prefill.
    pub fn host_bytes(self) -> usize { self.host_resident_bytes() + self.host_workspace_bytes() }
    pub fn device_bytes(self) -> usize { self.units * self.rows * ENGRAM_DEVICE_ROW_BYTES }
    pub fn validate_slots(self, available: usize) -> std::result::Result<(), EngramStagingError> {
        if available != self.host_slots() { return Err(EngramStagingError::Slots {
            required: self.host_slots(), available }); }
        Ok(())
    }
}

/// Recorded only when the runtime's timing trace is enabled at submission.
pub type EngramGatherTiming = crate::GatherTiming;

/// One layer's gather: the exact immutable request batches, in wave order.
pub struct EngramGatherJob {
    table: Arc<EngramTable>,
    layer: usize,
    batches: Vec<Arc<EngramBatch>>,
}

/// A pinned gather source. After enqueue it is retained by the gatherer until
/// its H2D event completes, independently of wave cancellation or gate lifetime.
pub struct EngramGatherLease(GatherLease<EngramBatchStaging, EngramGatherJob>);
impl EngramGatherLease {
    pub fn view(&self) -> Result<EngramGatherView<'_>> {
        self.0.slot().context("engram gather lease is empty")?.view()
    }
    /// Exact immutable batches represented by the concatenated output rows.
    pub fn batches(&self) -> &[Arc<EngramBatch>] {
        &self.0.job().batches
    }
    pub fn timing(&self) -> Option<&EngramGatherTiming> {
        self.0.timing()
    }
    pub fn host_buffer(&self) -> Option<cuteafd_ffi::CuteafdHostBuffer> { self.0.slot()?.host_buffer() }
    /// # Safety
    /// All reads of this lease's pinned bytes are enqueued on the supplied stream.
    pub unsafe fn copied_on(&mut self, stream: *mut std::ffi::c_void) -> Result<()> {
        // SAFETY: caller provides the stream containing every queued reader.
        unsafe { self.0.slot_mut().context("empty gather lease")?.copied_on(stream) }
    }
    fn copy_ready(&mut self) -> Result<bool> {
        self.0.slot_mut().context("empty gather lease")?.copy_ready()
    }
}

pub enum EngramGatherPoll {
    Pending,
    Cancelled,
    Ready(EngramGatherLease),
}
/// Independent of scheduler slot reuse; drop cancels queued/in-progress work.
/// An in-progress OS page fault cannot be interrupted, but its result is discarded.
pub struct EngramGatherTicket(GatherTicket<EngramBatchStaging, EngramGatherJob, anyhow::Error>);
impl EngramGatherTicket {
    pub fn cancel(&mut self) {
        self.0.cancel();
    }
    /// Sleep until completion without retaining a request-bank borrow.
    pub async fn wait(&mut self) -> Result<EngramGatherPoll> {
        Self::result(self.0.wait().await)
    }
    /// Nonblocking polling for a CUDA or scheduler thread; consume at most once.
    pub fn poll(&mut self) -> Result<EngramGatherPoll> {
        Self::result(self.0.poll())
    }
    fn result(result: std::result::Result<GatherPoll<GatherLease<EngramBatchStaging, EngramGatherJob>>,
        GatherFailure<anyhow::Error>>) -> Result<EngramGatherPoll> {
        match result {
            Ok(GatherPoll::Pending) => Ok(EngramGatherPoll::Pending),
            Ok(GatherPoll::Cancelled) => Ok(EngramGatherPoll::Cancelled),
            Ok(GatherPoll::Ready(lease)) => Ok(EngramGatherPoll::Ready(EngramGatherLease(lease))),
            Err(GatherFailure::Job(error)) => Err(error),
            Err(GatherFailure::Worker(error)) => Err(anyhow::Error::new(error).context("engram gather")),
        }
    }
}

/// Deferred copy leases survive wave cancellation. The readiness predicate is
/// queried only from the CUDA thread's existing I/O polls.
pub struct EngramCopyPool<L> { pending: Vec<L>, capacity: usize }
impl<L> EngramCopyPool<L> {
    pub fn new(capacity: usize) -> Self { Self { pending: Vec::with_capacity(capacity), capacity } }
    pub fn retain(&mut self, lease: L) {
        assert!(self.pending.len() < self.capacity, "copy pool exceeds its gather slots");
        self.pending.push(lease);
    }
    pub fn capacity_stalled(&self, capacity_available: bool) -> bool {
        !capacity_available && !self.pending.is_empty()
    }
    pub fn reclaim(&mut self, mut ready: impl FnMut(&mut L) -> Result<bool>) -> Result<()> {
        let mut index = 0;
        while index < self.pending.len() {
            if ready(&mut self.pending[index])? { self.pending.swap_remove(index); }
            else { index += 1; }
        }
        Ok(())
    }
}

pub struct EngramGatherer {
    worker: GatherWorker<EngramBatchStaging, EngramGatherJob, anyhow::Error>,
    capacity: usize,
    copies: std::sync::Mutex<EngramCopyPool<EngramGatherLease>>,
}
impl EngramGatherer {
    /// The slot pool bounds queued, in-flight and completed-but-unconsumed storage.
    pub fn new(slots: usize, capacity: usize, staging_budget: usize) -> Result<Self> {
        ensure!(
            slots > 0 && slots <= 32,
            "engram gather pool requires 1..32 slots"
        );
        let bytes = EngramBatchStaging::storage_bytes(capacity)?
            .checked_mul(slots)
            .context("engram gather pool budget overflow")?;
        ensure!(
            bytes <= staging_budget,
            "engram gather pool exceeds host staging budget"
        );
        let staging = (0..slots).map(|_| EngramBatchStaging::new(capacity)).collect::<Result<Vec<_>>>()?;
        Self::with_staging(capacity, staging)
    }
    pub fn new_pinned(library: &cuteafd_ffi::NativeLibrary, shape: EngramStagingShape) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("v41/engram-prefill");
        let staging = (0..shape.host_slots()).map(|_| EngramBatchStaging::new_pinned(library, shape.rows))
            .collect::<Result<Vec<_>>>()?;
        shape.validate_slots(staging.len())?;
        Self::with_staging(shape.rows, staging)
    }
    pub fn new_decode(library: &cuteafd_ffi::NativeLibrary, shape: EngramStagingShape) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("v41/engram-decode");
        let staging = (0..shape.decode_slots()).map(|_| EngramBatchStaging::new_pinned(library, shape.decode_rows()))
            .collect::<Result<Vec<_>>>()?;
        Self::with_staging(shape.decode_rows(), staging)
    }
    pub fn copy_capacity_stalled(&self) -> Result<bool> {
        let copies = self.copies.lock().unwrap_or_else(|p| p.into_inner());
        Ok(copies.capacity_stalled(self.worker.capacity_available()?))
    }
    fn with_staging(capacity: usize, staging: Vec<EngramBatchStaging>) -> Result<Self> {
        let slots = staging.len();
        let worker = GatherWorker::new("engram-gather", staging, |staging: &mut EngramBatchStaging, job: &EngramGatherJob| {
            let started = Instant::now();
            let batches: Vec<_> = job.batches.iter().map(Arc::as_ref).collect();
            let view = staging.gather(&job.table, &batches, job.layer)?;
            let bytes = view.weights.len() + view.scales.len();
            job.table.weights().stats().record_gather(view.rows * 24, bytes, started.elapsed(), [0; 3]);
            Ok(())
        })
        .context("starting engram gather worker")?;
        Ok(Self { worker, capacity, copies: std::sync::Mutex::new(EngramCopyPool::new(slots)) })
    }
    pub fn retain_copy(&self, lease: EngramGatherLease) {
        self.copies.lock().unwrap_or_else(|p| p.into_inner()).retain(lease);
    }
    /// Called from existing request I/O polling, including cancelled waves.
    pub fn reclaim_copies(&self) -> Result<()> {
        self.copies.lock().unwrap_or_else(|p| p.into_inner()).reclaim(EngramGatherLease::copy_ready)
    }
    /// Submit early when decode/prefill/verification token hashes become available.
    /// None is bounded backpressure; it performs no mapped reads or blocking wait.
    pub fn try_submit(
        &self,
        table: Arc<EngramTable>,
        batches: &[Arc<EngramBatch>],
        layer: usize,
    ) -> Result<Option<EngramGatherTicket>> {
        self.reclaim_copies()?;
        let job = self.job(table, batches, layer)?;
        let timed = tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG);
        Ok(self.worker.try_submit(job, timed)?.map(EngramGatherTicket))
    }
    /// Retain the immutable job while sleeping for bounded staging admission.
    pub async fn submit(&self, table: Arc<EngramTable>, batches: &[Arc<EngramBatch>],
        layer: usize) -> Result<EngramGatherTicket> {
        self.reclaim_copies()?;
        let job = self.job(table, batches, layer)?;
        let timed = tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG);
        Ok(EngramGatherTicket(self.worker.submit(job, timed).await?))
    }
    fn job(&self, table: Arc<EngramTable>, batches: &[Arc<EngramBatch>],
        layer: usize) -> Result<EngramGatherJob> {
        ensure!(
            !batches.is_empty() && batches.len() <= 16,
            "invalid engram request batch count"
        );
        let rows = batches.iter().try_fold(0usize, |sum, batch| {
            sum.checked_add(batch.hashes().len())
                .context("engram batch length overflow")
        })?;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "engram gather exceeds row capacity"
        );
        ensure!(
            table.weights().rows() == *ENGRAM_ROWS.get(layer).context("invalid engram layer")?,
            "engram gather table belongs to another layer"
        );
        Ok(EngramGatherJob { table, layer, batches: batches.to_vec() })
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc, sync::atomic::{AtomicBool, Ordering}};

    struct CopyLease { ready: Arc<AtomicBool>, drops: Rc<RefCell<Vec<usize>>>, id: usize }
    impl Drop for CopyLease { fn drop(&mut self) { self.drops.borrow_mut().push(self.id); } }

    #[test]
    fn engram_copy_pending_pool_reclaims_on_completion_with_bounded_polls() -> Result<()> {
        for k in 1..=4 {
            let shape = EngramStagingShape::new(((8192usize.div_ceil(k.max(2)) + 63) / 64 * 64), k)?;
            let drops = Rc::new(RefCell::new(Vec::new()));
            let mut pool = EngramCopyPool::new(shape.host_slots());
            let mut events = Vec::new();
            for id in 0..shape.host_slots() {
                let ready = Arc::new(AtomicBool::new(false));
                events.push(ready.clone()); pool.retain(CopyLease { ready, drops: drops.clone(), id });
            }
            assert!(pool.capacity_stalled(false));
            let mut polls = 0;
            pool.reclaim(|l| { polls += 1; Ok(l.ready.load(Ordering::Acquire)) })?;
            assert!(pool.capacity_stalled(false) && drops.borrow().is_empty());
            events[k].store(true, Ordering::Release);
            pool.reclaim(|l| { polls += 1; Ok(l.ready.load(Ordering::Acquire)) })?;
            assert!(!pool.capacity_stalled(true), "one completion admits another gather");
            assert_eq!(*drops.borrow(), [k]);
            assert!(polls <= 2 * shape.host_slots());
            for event in events { event.store(true, Ordering::Release); }
            pool.reclaim(|l| Ok(l.ready.load(Ordering::Acquire)))?;
            let mut released = drops.borrow().clone(); released.sort_unstable();
            assert_eq!(released, (0..shape.host_slots()).collect::<Vec<_>>());
        }
        Ok(())
    }

    fn complete<T>(future: impl std::future::Future<Output = T>) -> T {
        use std::task::{Context, Poll, Wake, Waker};
        struct ThreadWake(std::thread::Thread);
        impl Wake for ThreadWake { fn wake(self: Arc<Self>) { self.0.unpark(); } }
        let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
        let mut future = std::pin::pin!(future);
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut Context::from_waker(&waker)) { return value; }
            assert!(Instant::now() < deadline, "fixture wait timed out");
            std::thread::park_timeout(deadline.saturating_duration_since(Instant::now()));
        }
    }
    type TestTicket = GatherTicket<usize, (usize, usize), ()>;
    type TestLease = GatherLease<usize, (usize, usize)>;
    fn ready(ticket: &mut TestTicket) -> TestLease {
        match complete(ticket.wait()) { Ok(GatherPoll::Ready(lease)) => lease, _ => panic!("gather not ready") }
    }
    struct PendingCopy { lease: TestLease, done: Arc<AtomicBool> }

    #[test]
    fn engram_lifecycle_exact_pool_and_adversarial_layer14_admission() -> Result<()> {
        for k in 1..=4 {
            let rows = 8192usize.div_ceil(k.max(2)).div_ceil(64) * 64;
            let shape = EngramStagingShape::new(rows, k)?;
            let n = shape.host_slots();
            assert_eq!(n, 3 * k);
            assert_eq!(shape.host_resident_bytes(), n * EngramBatchStaging::storage_bytes(rows)?);
            assert_eq!(shape.host_bytes(), shape.host_resident_bytes() + rows * 24 * 1024 + (1 << 20));
            assert_eq!(shape.decode_host_bytes(), 4 * EngramBatchStaging::storage_bytes(256)? + 256 * 24 * 1024 + (1 << 20));
            assert_eq!(shape.device_bytes(), k * rows * ENGRAM_DEVICE_ROW_BYTES);
            assert_eq!(shape.validate_slots(n - 1), Err(EngramStagingError::Slots { required: n, available: n - 1 }));
            let order = Arc::new(std::sync::Mutex::new(Vec::new()));
            let observed = order.clone();
            let worker = GatherWorker::new("engram-lifecycle", (0..n).collect::<Vec<_>>(),
                move |_: &mut usize, job: &(usize, usize)| {
                    observed.lock().unwrap().push(*job); Ok::<_, ()>(())
                })?;
            let mut copies = EngramCopyPool::new(n);
            let mut events = Vec::new();
            // Retain active L1 sources even after cancellation, while every
            // lookahead L1 is ready. Remaining k slots must admit active L14.
            for unit in 0..k {
                let mut ticket = worker.try_submit((unit, 0), false)?.unwrap();
                let done = Arc::new(AtomicBool::new(false)); events.push(done.clone());
                copies.retain(PendingCopy { lease: ready(&mut ticket), done });
            }
            let mut future = (k..2*k).map(|unit| worker.try_submit((unit, 0), false).map(Option::unwrap))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let future = future.iter_mut().map(ready).collect::<Vec<_>>();
            for unit in 0..k {
                let mut ticket = worker.try_submit((unit, 1), false)?.expect("lookahead cannot starve L14");
                let done = Arc::new(AtomicBool::new(false)); events.push(done.clone());
                copies.retain(PendingCopy { lease: ready(&mut ticket), done });
            }
            assert!(worker.try_submit((99, 0), false)?.is_none(), "computed live maximum attained");
            let order = order.lock().unwrap();
            for unit in 0..k {
                assert!(order.iter().position(|x| *x == (unit, 0)) < order.iter().position(|x| *x == (unit, 1)));
                assert!(order.iter().position(|x| *x == (2*k-1, 0)) < order.iter().position(|x| *x == (unit, 1)));
            }
            drop(order);
            drop(future); // cancelled ready lookahead returns its slots
            for event in &events { event.store(true, Ordering::Release); }
            copies.reclaim(|copy| { let _ = copy.lease.slot(); Ok(copy.done.load(Ordering::Acquire)) })?;
            // Exactly n slots returned: every admission succeeds once, n+1 fails.
            let held = (0..n).map(|i| worker.try_submit((100+i, 0), false).map(Option::unwrap))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            assert!(worker.try_submit((999, 0), false)?.is_none());
            drop(held);

        }
        Ok(())
    }

    #[test]
    fn engram_cancel_queued_gathering_ready_and_copy_states_return_once() -> Result<()> {
        for k in 1..=4 {
            let n = EngramStagingShape::new(8192usize.div_ceil(k.max(2)).div_ceil(64) * 64, k)?.host_slots();
            // Block the first gather deterministically; the second is queued.
            let (started, seen) = std::sync::mpsc::sync_channel(1);
            let (release, wait) = std::sync::mpsc::sync_channel(1);
            let mut first = true;
            let worker = GatherWorker::new("engram-cancel-states", (0..n).collect::<Vec<_>>(),
                move |_: &mut usize, _: &(usize, usize)| {
                    if first { first = false; let _ = started.send(()); let _ = wait.recv(); }
                    Ok::<_, ()>(())
                })?;
            let mut gathering = worker.try_submit((0, 0), false)?.unwrap();
            seen.recv_timeout(std::time::Duration::from_secs(5))?;
            let mut queued = worker.try_submit((1, 0), false)?.unwrap();
            gathering.cancel(); queued.cancel(); drop(gathering); drop(queued);
            release.send(())?;
            // FIFO barrier proves both cancelled leases have finished recycling.
            let mut barrier = worker.try_submit((2, 0), false)?.unwrap(); drop(ready(&mut barrier));
            let mut ticket = worker.try_submit((3, 0), false)?.unwrap();
            drop(ready(&mut ticket)); // ready cancellation
            let mut ticket = worker.try_submit((4, 0), false)?.unwrap();
            let done = Arc::new(AtomicBool::new(false));
            let mut copies = EngramCopyPool::new(n);
            copies.retain(PendingCopy { lease: ready(&mut ticket), done: done.clone() });
            copies.reclaim(|copy| Ok(copy.done.load(Ordering::Acquire)))?;
            assert_eq!(copies.pending.len(), 1, "cancel cannot return a copy-pending source");
            done.store(true, Ordering::Release);
            copies.reclaim(|copy| Ok(copy.done.load(Ordering::Acquire)))?;
            assert!(copies.pending.is_empty());
            copies.reclaim(|_| panic!("copy-done lease returned twice"))?;
            let mut tickets = (0..n).map(|i| worker.try_submit((i, 1), false).map(Option::unwrap))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            assert!(worker.try_submit((n, 1), false)?.is_none());
            let mut ids = tickets.iter_mut().map(|t| *ready(t).slot().unwrap()).collect::<Vec<_>>();
            ids.sort_unstable(); assert_eq!(ids, (0..n).collect::<Vec<_>>());
        }
        Ok(())
    }

    #[test]
    fn engram_copy_completion_wakes_real_gather_admission() -> Result<()> {
        use std::{future::Future, task::{Context, Poll, Wake, Waker}};
        struct Counter(std::sync::atomic::AtomicUsize);
        impl Wake for Counter { fn wake(self: Arc<Self>) { self.0.fetch_add(1, Ordering::Release); } }
        for k in 1..=4 {
            let n = EngramStagingShape::new(8192usize.div_ceil(k.max(2)).div_ceil(64) * 64, k)?.host_slots();
            let worker = GatherWorker::new("engram-copy-admit", (0..n).collect::<Vec<_>>(),
                |_: &mut usize, _: &(usize, usize)| Ok::<_, ()>(()))?;
            let mut copies = EngramCopyPool::new(n);
            let event = Arc::new(AtomicBool::new(false));
            for i in 0..n {
                let mut ticket = worker.try_submit((i, 0), false)?.unwrap();
                copies.retain(PendingCopy { lease: ready(&mut ticket), done: event.clone() });
            }
            let wake = Arc::new(Counter(std::sync::atomic::AtomicUsize::new(0)));
            let waker = Waker::from(wake.clone());
            let mut cx = Context::from_waker(&waker);
            let mut admission = std::pin::pin!(worker.submit((n, 1), false));
            assert!(admission.as_mut().poll(&mut cx).is_pending());
            assert!(copies.capacity_stalled(worker.capacity_available()?));
            event.store(true, Ordering::Release);
            let mut polls = 0;
            copies.reclaim(|copy| { polls += 1; Ok(copy.done.load(Ordering::Acquire)) })?;
            assert!(polls <= n && wake.0.load(Ordering::Acquire) > 0);
            let Poll::Ready(ticket) = admission.as_mut().poll(&mut cx) else { panic!("copy completion did not admit gather") };
            let mut ticket = ticket?; drop(ready(&mut ticket));
        }
        Ok(())
    }

    #[test]
    fn engram_cancel_mid_copy_mixed_lookahead_admission_progress() -> Result<()> {
        use std::{future::Future, task::{Context, Poll, Wake, Waker}};
        struct Counter(std::sync::atomic::AtomicUsize);
        impl Wake for Counter { fn wake(self: Arc<Self>) { self.0.fetch_add(1, Ordering::Release); } }
        for k in 1..=4 {
            let n = EngramStagingShape::new(8192usize.div_ceil(k.max(2)).div_ceil(64) * 64, k)?.host_slots();
            let worker = GatherWorker::new("engram-mixed-copy-admit", (0..n).collect::<Vec<_>>(),
                |_: &mut usize, _: &(usize, usize)| Ok::<_, ()>(()))?;
            let mut copies = EngramCopyPool::new(n);
            let event = Arc::new(AtomicBool::new(false));
            let mut cancelled = worker.try_submit((0, 0), false)?.unwrap();
            copies.retain(PendingCopy { lease: ready(&mut cancelled), done: event.clone() });
            cancelled.cancel(); drop(cancelled); // lease now belongs to the copy pool
            let mut held = Vec::new();
            for i in 1..n {
                let mut ticket = worker.try_submit((i, 0), false)?.unwrap();
                held.push(ready(&mut ticket)); // refill lookahead and active ready gathers
            }
            let wake = Arc::new(Counter(std::sync::atomic::AtomicUsize::new(0)));
            let waker = Waker::from(wake.clone());
            let mut cx = Context::from_waker(&waker);
            let mut admission = std::pin::pin!(worker.submit((n, 1), false));
            assert!(admission.as_mut().poll(&mut cx).is_pending());
            assert!(copies.capacity_stalled(worker.capacity_available()?));
            copies.reclaim(|copy| Ok(copy.done.load(Ordering::Acquire)))?;
            assert_eq!(wake.0.load(Ordering::Acquire), 0, "worker cannot notify CUDA completion");
            event.store(true, Ordering::Release);
            let mut polls = 0;
            copies.reclaim(|copy| { polls += 1; Ok(copy.done.load(Ordering::Acquire)) })?;
            assert!(polls <= n && wake.0.load(Ordering::Acquire) > 0);
            assert!(!copies.capacity_stalled(worker.capacity_available()?));
            let Poll::Ready(ticket) = admission.as_mut().poll(&mut cx) else { panic!("mixed admission stuck") };
            let mut ticket = ticket?; drop(ready(&mut ticket));
            // A full worker-only pool must use notifications, not event polling.
            let mut ticket = worker.try_submit((n + 1, 0), false)?.unwrap();
            held.push(ready(&mut ticket));
            assert!(!worker.capacity_available()?);
            assert!(!copies.capacity_stalled(false));
        }
        Ok(())
    }

    #[test]
    fn engram_shape_refusal_is_typed_and_never_shrinks_lookahead() {
        for (rows, units) in [(0, 2), (4097, 2), (8192, 1), (4096, 0), (4096, 5)] {
            assert_eq!(EngramStagingShape::new(rows, units), Err(EngramStagingError::Shape { rows, units }));
        }
        for k in 1..=4 { assert_eq!(EngramStagingShape::new(4096, k).unwrap().prefetch_units(), k); }
    }
}
