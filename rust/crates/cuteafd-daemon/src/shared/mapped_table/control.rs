//! Fixed staging admission and executor-owned gather generations. The worker's
//! channel carries only coalesced wakeups, never row IDs or a queue of jobs.
use super::MappedRowsError;
use std::sync::{Arc, Mutex, Weak, mpsc};
use std::time::Duration;

pub(crate) trait RowSource: Send + Sync {
    fn units(&self) -> usize;
    fn rows(&self, unit: usize) -> &[i64];
}
impl RowSource for Vec<i64> {
    fn units(&self) -> usize { 1 }
    fn rows(&self, unit: usize) -> &[i64] { assert_eq!(unit, 0); self }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Use {
    pub slot: usize,
    pub sequence: u32,
    pub previous: u32,
}
#[derive(Default)]
struct Status {
    usage: Option<Use>,
    cancelled: bool,
    completed: bool,
    stream: Option<(i32, usize)>,
    checked: bool,
    error: Option<MappedRowsError>,
}

pub(crate) struct RowBatch {
    source: Arc<dyn RowSource>,
    statuses: Vec<Mutex<Status>>,
}
impl RowBatch {
    pub(super) fn new(source: Arc<dyn RowSource>) -> Arc<Self> {
        Arc::new(Self { statuses: (0..source.units()).map(|_| Mutex::new(Status::default())).collect(), source })
    }
    pub(super) fn rows(&self, unit: usize) -> &[i64] { self.source.rows(unit) }

    /// Called only after the consumer stream drained. One failed generation
    /// fails its entire step, not the process or a later request.
    pub(super) fn readback(&self, device: i32, stream: usize) -> Result<(), MappedRowsError> {
        let mut first = None;
        for status in &self.statuses {
            let mut status = status.lock().unwrap_or_else(|p| p.into_inner());
            if status.stream == Some((device, stream)) && !status.checked {
                if !status.completed { return Err(MappedRowsError::Incomplete); }
                status.checked = true;
                if first.is_none() { first = status.error.clone(); }
            }
        }
        first.map_or(Ok(()), Err)
    }
    pub(super) fn bind_readback(&self, device: i32, stream: usize) {
        for status in &self.statuses {
            let mut status = status.lock().unwrap_or_else(|p| p.into_inner());
            if status.stream.is_some() && !status.checked { status.stream = Some((device, stream)); }
        }
    }
    pub(crate) fn cancel_unbegun(&self) {
        for status in &self.statuses {
            let mut status = status.lock().unwrap_or_else(|p| p.into_inner());
            if status.usage.is_none() { status.cancelled = true; }
        }
    }
    pub(super) fn retired(&self) -> bool {
        self.statuses.iter().all(|status| {
            let status = status.lock().unwrap_or_else(|p| p.into_inner());
            status.cancelled || status.checked
        })
    }
}

#[derive(Default)]
struct Slot {
    sequence: u32,
    submitted: u32,
    leased: bool,
}
struct Schedule {
    slots: Vec<Slot>,
    batches: Vec<Weak<RowBatch>>,
    closing: bool,
}
pub(super) struct Control {
    schedule: Mutex<Schedule>,
    wake: mpsc::SyncSender<()>,
    prefill_slots: usize,
}
impl Control {
    pub(super) fn new(prefill_slots: usize) -> (Arc<Self>, mpsc::Receiver<()>) {
        let (wake, receive) = mpsc::sync_channel(1);
        (Arc::new(Self { schedule: Mutex::new(Schedule {
            slots: (0..=prefill_slots).map(|_| Slot::default()).collect(), batches: Vec::new(), closing: false,
        }), wake, prefill_slots }), receive)
    }
    pub(super) fn batch(&self, source: Arc<dyn RowSource>) -> Arc<RowBatch> {
        let batch = RowBatch::new(source);
        let mut schedule = self.schedule.lock().unwrap_or_else(|p| p.into_inner());
        schedule.batches.retain(|batch| batch.strong_count() > 0);
        schedule.batches.push(Arc::downgrade(&batch));
        batch
    }
    pub(super) fn begin(self: &Arc<Self>, batch: Arc<RowBatch>, unit: usize, decode: bool)
        -> Result<Lease, MappedRowsError> {
        let mut schedule = self.schedule.lock().unwrap_or_else(|p| p.into_inner());
        if schedule.closing { return Err(MappedRowsError::Stopped); }
        let mut status = batch.statuses.get(unit).ok_or(MappedRowsError::Unit(unit))?
            .lock().unwrap_or_else(|p| p.into_inner());
        if status.usage.is_some() || status.cancelled { return Err(MappedRowsError::Unit(unit)); }
        let range = if decode { self.prefill_slots..self.prefill_slots + 1 } else { 0..self.prefill_slots };
        let slot = range.into_iter().find(|&slot| !schedule.slots[slot].leased)
            .ok_or(MappedRowsError::Admission(if decode { 1 } else { self.prefill_slots }))?;
        let entry = &mut schedule.slots[slot];
        entry.leased = true;
        entry.sequence = entry.sequence.wrapping_add(1);
        let usage = Use { slot, sequence: entry.sequence, previous: entry.submitted };
        status.usage = Some(usage);
        drop(status);
        drop(schedule);
        self.notify();
        Ok(Lease { control: Arc::clone(self), batch, unit, usage, submitted: false })
    }
    fn notify(&self) { let _ = self.wake.try_send(()); }
    pub(super) fn close(&self) {
        self.schedule.lock().unwrap_or_else(|p| p.into_inner()).closing = true;
        self.notify();
    }
}

/// Busy from begin, not just upload: unsubmitted gathers cannot alias. A
/// submitted lease frees host admission immediately; GPU consumed counters
/// guard the next generation's writes to the fixed pinned slot.
pub(super) struct Lease {
    control: Arc<Control>,
    pub batch: Arc<RowBatch>,
    pub unit: usize,
    pub usage: Use,
    submitted: bool,
}
impl Lease {
    pub(super) fn submit(&mut self, device: i32, stream: usize) {
        let mut schedule = self.control.schedule.lock().unwrap_or_else(|p| p.into_inner());
        self.batch.statuses[self.unit].lock().unwrap_or_else(|p| p.into_inner()).stream = Some((device, stream));
        let slot = &mut schedule.slots[self.usage.slot];
        slot.submitted = self.usage.sequence;
        slot.leased = false;
        self.submitted = true;
    }
    pub(super) fn failed_enqueue(&self) {
        let mut status = self.batch.statuses[self.unit].lock().unwrap_or_else(|p| p.into_inner());
        status.checked = true;
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        if !self.submitted {
            let mut schedule = self.control.schedule.lock().unwrap_or_else(|p| p.into_inner());
            self.batch.statuses[self.unit].lock().unwrap_or_else(|p| p.into_inner()).cancelled = true;
            schedule.slots[self.usage.slot].leased = false;
        }
        self.control.notify();
    }
}

pub(super) trait Backend: Send + 'static {
    fn consumed(&self, slot: usize) -> u32;
    fn gather(&mut self, slot: usize, rows: &[i64]) -> Result<(), MappedRowsError>;
    fn clear(&mut self, slot: usize, rows: usize);
    fn publish(&self, usage: Use);
}
fn reached(value: u32, target: u32) -> bool { value.wrapping_sub(target) as i32 >= 0 }

/// Max generations in flight = units the executor holds. Descriptors borrow
/// their IDs through that owned batch, rather than duplicating them in a job
/// queue. Only the k prefill slots and one decode slot contain gathered bytes.
pub(super) fn run<B: Backend>(control: Arc<Control>, receive: mpsc::Receiver<()>, mut backend: B) {
    let mut next = vec![1u32; control.prefill_slots + 1];
    loop {
        let mut selected = None;
        let mut outstanding = false;
        {
            let mut schedule = control.schedule.lock().unwrap_or_else(|p| p.into_inner());
            if schedule.closing { return; }
            schedule.batches.retain(|batch| batch.strong_count() > 0);
            for (slot, expected) in next.iter_mut().enumerate() {
                if !reached(schedule.slots[slot].sequence, *expected) { continue; }
                outstanding = true;
                let job = schedule.batches.iter().filter_map(Weak::upgrade).find_map(|batch| {
                    let unit = batch.statuses.iter().position(|status| status.lock().unwrap_or_else(|p| p.into_inner())
                        .usage.is_some_and(|usage| usage.slot == slot && usage.sequence == *expected))?;
                    Some((batch, unit))
                });
                let Some((batch, unit)) = job else {
                    // A dropped batch contains only cancelled/unsubmitted work.
                    *expected = expected.wrapping_add(1);
                    continue;
                };
                let status = batch.statuses[unit].lock().unwrap_or_else(|p| p.into_inner());
                if status.cancelled {
                    *expected = expected.wrapping_add(1);
                    continue;
                }
                let usage = status.usage.expect("selected generation");
                if reached(backend.consumed(slot), usage.previous) {
                    selected = Some((Arc::clone(&batch), unit, usage));
                    break;
                }
            }
        }
        if let Some((batch, unit, usage)) = selected {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.gather(usage.slot, batch.rows(unit))
            })).unwrap_or(Err(MappedRowsError::Panicked));
            if result.is_err() { backend.clear(usage.slot, batch.rows(unit).len()); }
            {
                let mut status = batch.statuses[unit].lock().unwrap_or_else(|p| p.into_inner());
                status.error = result.err();
                status.completed = true;
            }
            backend.publish(usage);
            next[usage.slot] = next[usage.slot].wrapping_add(1);
            continue;
        }
        // A device-written consumed counter has no host notification. Park
        // rather than burn an idle CPU; incoming begins/cancels wake us early.
        if outstanding {
            let _ = receive.recv_timeout(Duration::from_micros(100));
        } else if receive.recv().is_err() { return; }
    }
}

#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;
