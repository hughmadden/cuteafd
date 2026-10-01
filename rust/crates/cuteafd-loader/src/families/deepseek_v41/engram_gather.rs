//! Bounded background engram gathers on the shared [`GatherWorker`]: reusable
//! staging slots and request-owned cancellation.
use crate::{EngramBatchStaging, EngramGatherView, EngramTable, GatherFailure, GatherLease, GatherPoll, GatherTicket,
    GatherWorker};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{EngramBatch, ENGRAM_ROWS};
use std::sync::Arc;
use std::time::Instant;

/// Recorded only when the runtime's timing trace is enabled at submission.
pub type EngramGatherTiming = crate::GatherTiming;

/// One layer's gather: the exact immutable request batches, in wave order.
pub struct EngramGatherJob {
    table: Arc<EngramTable>,
    layer: usize,
    batches: Vec<Arc<EngramBatch>>,
}

/// Holds a staging slot until its synchronous GPU upload has consumed the view.
/// Dropping a ready result returns the slot to the pool without blocking.
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
    /// Nonblocking polling for a CUDA or scheduler thread; consume at most once.
    pub fn poll(&mut self) -> Result<EngramGatherPoll> {
        match self.0.poll() {
            Ok(GatherPoll::Pending) => Ok(EngramGatherPoll::Pending),
            Ok(GatherPoll::Cancelled) => Ok(EngramGatherPoll::Cancelled),
            Ok(GatherPoll::Ready(lease)) => Ok(EngramGatherPoll::Ready(EngramGatherLease(lease))),
            Err(GatherFailure::Job(error)) => Err(error),
            Err(GatherFailure::Worker(error)) => Err(anyhow::Error::new(error).context("engram gather")),
        }
    }
}

pub struct EngramGatherer {
    worker: GatherWorker<EngramBatchStaging, EngramGatherJob, anyhow::Error>,
    capacity: usize,
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
        let worker = GatherWorker::new("engram-gather", staging, |staging: &mut EngramBatchStaging, job: &EngramGatherJob| {
            let started = Instant::now();
            let batches: Vec<_> = job.batches.iter().map(Arc::as_ref).collect();
            let view = staging.gather(&job.table, &batches, job.layer)?;
            let bytes = view.weights.len() + view.scales.len();
            job.table.weights().stats().record_gather(view.rows * 24, bytes, started.elapsed(), [0; 3]);
            Ok(())
        })
        .context("starting engram gather worker")?;
        Ok(Self { worker, capacity })
    }
    /// Submit early when decode/prefill/verification token hashes become available.
    /// None is bounded backpressure; it performs no mapped reads or blocking wait.
    pub fn try_submit(
        &self,
        table: Arc<EngramTable>,
        batches: &[Arc<EngramBatch>],
        layer: usize,
    ) -> Result<Option<EngramGatherTicket>> {
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
        let timed = tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG);
        let job = EngramGatherJob { table, layer, batches: batches.to_vec() };
        Ok(self.worker.try_submit(job, timed)?.map(EngramGatherTicket))
    }
}
