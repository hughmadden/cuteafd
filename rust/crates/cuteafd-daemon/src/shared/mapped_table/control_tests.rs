use super::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

struct Source(Vec<Vec<i64>>);
impl RowSource for Source {
    fn units(&self) -> usize { self.0.len() }
    fn rows(&self, unit: usize) -> &[i64] { &self.0[unit] }
}

struct Fake {
    consumed: Arc<Vec<AtomicU32>>,
    published: mpsc::SyncSender<(Use, Vec<u8>)>,
    rows: Vec<Vec<u8>>,
}
impl Backend for Fake {
    fn consumed(&self, slot: usize) -> u32 { self.consumed[slot].load(Ordering::Acquire) }
    fn gather(&mut self, slot: usize, rows: &[i64]) -> Result<(), MappedRowsError> {
        self.rows[slot] = rows.iter().flat_map(|row| row.to_le_bytes()).collect();
        match rows[0] {
            -1 => Err(MappedRowsError::Gather("injected read failure".into())),
            -2 => panic!("injected gather panic"),
            _ => Ok(()),
        }
    }
    fn clear(&mut self, slot: usize, rows: usize) { self.rows[slot] = vec![0; rows * 8]; }
    fn publish(&self, usage: Use) { self.published.send((usage, self.rows[usage.slot].clone())).unwrap(); }
}
fn worker(k: usize) -> (Arc<Control>, Arc<Vec<AtomicU32>>, mpsc::Receiver<(Use, Vec<u8>)>, std::thread::JoinHandle<()>) {
    let (control, receive) = Control::new(k);
    let consumed = Arc::new((0..=k).map(|_| AtomicU32::new(0)).collect::<Vec<_>>());
    let (published, output) = mpsc::sync_channel(128);
    let backend = Fake { consumed: Arc::clone(&consumed), published, rows: vec![Vec::new(); k + 1] };
    let owner = Arc::clone(&control);
    let thread = std::thread::spawn(move || run(owner, receive, backend));
    (control, consumed, output, thread)
}

#[test]
fn three_and_four_unsubmitted_waves_have_distinct_owned_slots() {
    for k in [3, 4] {
        let (control, _) = Control::new(k);
        let batch = control.batch(Arc::new(Source((0..k + 2).map(|n| vec![n as i64]).collect())));
        let mut leases = (0..k).map(|unit| control.begin(Arc::clone(&batch), unit, false).unwrap()).collect::<Vec<_>>();
        let slots = leases.iter().map(|lease| lease.usage.slot).collect::<std::collections::HashSet<_>>();
        assert_eq!(slots.len(), k);
        assert!(matches!(control.begin(Arc::clone(&batch), k, false), Err(MappedRowsError::Admission(n)) if n == k));
        let decode = control.begin(Arc::clone(&batch), k + 1, true).unwrap();
        assert_eq!(decode.usage.slot, k);
        drop(leases.remove(0));
        assert!(control.begin(Arc::clone(&batch), k, false).is_ok());
        drop(decode);
    }
}

#[test]
fn long_prompt_runs_far_ahead_without_overwriting_unconsumed_bytes() {
    for k in [3, 4] {
        let count = 64;
        let (control, consumed, output, thread) = worker(k);
        let ids = Arc::new(Source((0..count).map(|n| vec![n as i64 + 10, n as i64 + 1000]).collect()));
        let weak = Arc::downgrade(&ids);
        let batch = control.batch(ids.clone());
        drop(ids);
        let started = Instant::now();
        let mut uses = Vec::new();
        // Simulate the executor holding k unsubmitted units, then queuing
        // every unit of a long prompt before the GPU consumes any generation.
        for first in (0..count).step_by(k) {
            let mut live = (first..(first + k).min(count))
                .map(|unit| control.begin(Arc::clone(&batch), unit, false).unwrap()).collect::<Vec<_>>();
            for lease in &mut live {
                uses.push((lease.unit, lease.usage));
                lease.submit(0, 123);
            }
        }
        assert!(started.elapsed() < Duration::from_secs(1), "issuer must not wait for GPU consumption");
        let mut seen = std::collections::HashSet::new();
        for _ in 0..k {
            let (usage, bytes) = output.recv_timeout(Duration::from_secs(5)).unwrap();
            let unit = uses.iter().find(|(_, expected)| *expected == usage).unwrap().0;
            assert_eq!(bytes, batch.rows(unit).iter().flat_map(|row| row.to_le_bytes()).collect::<Vec<_>>());
            seen.insert(unit);
        }
        assert!(output.recv_timeout(Duration::from_millis(10)).is_err(), "no staging slot is free yet");
        for (_, usage) in uses.iter().take(k) { consumed[usage.slot].store(usage.sequence, Ordering::Release); }
        while seen.len() < count {
            let (usage, bytes) = output.recv_timeout(Duration::from_secs(5)).unwrap();
            let unit = uses.iter().find(|(_, expected)| *expected == usage).unwrap().0;
            assert!(seen.insert(unit));
            assert_eq!(bytes, batch.rows(unit).iter().flat_map(|row| row.to_le_bytes()).collect::<Vec<_>>());
            consumed[usage.slot].store(usage.sequence, Ordering::Release);
        }
        batch.readback(0, 123).unwrap();
        assert!(batch.retired());
        control.close();
        thread.join().unwrap();
        drop(batch);
        assert!(weak.upgrade().is_none(), "retired batches release their owned row IDs");
    }
}

#[test]
fn shared_executor_never_exceeds_begin_time_admission() -> anyhow::Result<()> {
    use crate::shared::prefill_pipeline::{prefill_schedule, sequence_units, FfnRoute, Step};
    for k in [1, 2, 3, 4] {
        let mut units = sequence_units(65536, 512, 1, 4)?;
        for unit in &mut units { unit.slot = unit.index % k; }
        let (control, _) = Control::new(k);
        let batch = control.batch(Arc::new(Source(vec![vec![0]; units.len()])));
        let mut leases = (0..units.len()).map(|_| None).collect::<Vec<_>>();
        for step in prefill_schedule(&[FfnRoute::Spark; 4], &units, k)? {
            match step {
                Step::Begin { unit } => leases[unit] = Some(control.begin(Arc::clone(&batch), unit, false)?),
                Step::Attention { layer: 1, unit } => {
                    let mut lease = leases[unit].take().unwrap();
                    lease.submit(0, 123);
                }
                _ => (),
            }
        }
        assert!(leases.iter().all(Option::is_none));
    }
    Ok(())
}

#[test]
fn failure_zeroes_rows_releases_ready_and_does_not_poison_later_steps() {
    let (control, consumed, output, thread) = worker(1);
    for (row, failed) in [(-1, true), (-2, true), (42, false)] {
        let batch = control.batch(Arc::new(vec![row]));
        let mut lease = control.begin(Arc::clone(&batch), 0, true).unwrap();
        let usage = lease.usage;
        lease.submit(0, 777);
        let (published, bytes) = output.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(published, usage);
        assert_eq!(bytes, if failed { vec![0; 8] } else { row.to_le_bytes().to_vec() });
        consumed[usage.slot].store(usage.sequence, Ordering::Release);
        assert!(batch.readback(1, 777).is_ok(), "another device cannot consume this error");
        assert_eq!(batch.readback(0, 777).is_err(), failed);
        assert!(batch.readback(0, 777).is_ok(), "errors consume once at their own readback");
    }
    control.close();
    thread.join().unwrap();
}

#[test]
fn partial_step_drain_retires_error_before_next_requests_readback() {
    let (control, consumed, output, thread) = worker(1);
    let ledger = super::super::Readbacks::default();
    let failed = control.batch(Arc::new(Source(vec![vec![-1], vec![123]])));
    let mut lease = control.begin(Arc::clone(&failed), 0, false).unwrap();
    lease.submit(0, 111);
    ledger.0.lock().unwrap().push(Arc::clone(&failed));
    let (usage, _) = output.recv_timeout(Duration::from_secs(5)).unwrap();
    consumed[usage.slot].store(usage.sequence, Ordering::Release);
    failed.cancel_unbegun();
    // A partial owner-hop step drains all owners, then binds its error to
    // that existing terminal boundary instead of waiting for absent logits.
    failed.bind_readback(1, 222);
    assert!(ledger.check(0, 111).is_ok());
    assert!(ledger.check(1, 222).is_err());
    assert!(ledger.0.lock().unwrap().is_empty());
    assert!(matches!(control.begin(Arc::clone(&failed), 1, false), Err(MappedRowsError::Unit(1))));
    let next = control.batch(Arc::new(vec![42]));
    let mut lease = control.begin(Arc::clone(&next), 0, false).unwrap();
    lease.submit(1, 222);
    ledger.0.lock().unwrap().push(next);
    let (usage, bytes) = output.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(bytes, 42i64.to_le_bytes());
    consumed[usage.slot].store(usage.sequence, Ordering::Release);
    ledger.check(1, 222).unwrap();
    assert!(ledger.0.lock().unwrap().is_empty(), "the previous request cannot poison this readback");
    control.close();
    thread.join().unwrap();
}

#[test]
fn cancellation_does_not_require_a_gpu_consumed_store() {
    let (control, consumed, output, thread) = worker(1);
    let cancelled = control.batch(Arc::new(vec![7]));
    drop(control.begin(Arc::clone(&cancelled), 0, false).unwrap());
    let batch = control.batch(Arc::new(vec![99]));
    let mut lease = control.begin(Arc::clone(&batch), 0, false).unwrap();
    assert_eq!(lease.usage.previous, 0);
    lease.submit(0, 123);
    loop {
        let (usage, bytes) = output.recv_timeout(Duration::from_secs(5)).unwrap();
        if usage == lease.usage {
            assert_eq!(bytes, 99i64.to_le_bytes());
            consumed[usage.slot].store(usage.sequence, Ordering::Release);
            break;
        }
    }
    batch.readback(0, 123).unwrap();
    control.close();
    thread.join().unwrap();
}
