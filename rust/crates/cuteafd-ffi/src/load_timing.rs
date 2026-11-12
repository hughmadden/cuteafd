//! Scoped weight-load accounting. Contexts cross only explicit loader boundaries.
use std::{cell::RefCell, sync::{Arc, Mutex}, time::Instant};

#[derive(Clone, Copy, Default)]
struct Counter { bytes: u64, seconds: f64 }
#[derive(Default)]
struct Totals { counters: [Counter; 4], busy: Vec<(Instant, Instant)> }
#[derive(Clone, Default)]
pub struct Context(Vec<Arc<Mutex<Totals>>>);
thread_local! { static ACTIVE: RefCell<Context> = RefCell::new(Context::default()); }

/// Capture on the parent before spawning; unrelated loads have separate contexts.
pub fn capture() -> Context { ACTIVE.with(|active| active.borrow().clone()) }
impl Context {
    pub fn enter<T>(&self, body: impl FnOnce() -> T) -> T {
        struct Restore(Context);
        impl Drop for Restore {
            fn drop(&mut self) { ACTIVE.with(|active| *active.borrow_mut() = self.0.clone()); }
        }
        let _restore = Restore(ACTIVE.with(|active| active.replace(self.clone())));
        body()
    }
}
fn timed<T>(kind: usize, bytes: usize, body: impl FnOnce() -> T) -> T {
    let context = capture();
    let start = Instant::now();
    let value = body();
    let end = Instant::now();
    for phase in context.0 {
        let mut totals = phase.lock().unwrap_or_else(|e| e.into_inner());
        totals.counters[kind].bytes += bytes as u64;
        totals.counters[kind].seconds += (end - start).as_secs_f64();
        // Quantization is part of transforms, not a second subtraction.
        if kind != 2 { totals.busy.push((start, end)); }
    }
    value
}
pub fn read<T>(bytes: usize, body: impl FnOnce() -> T) -> T { timed(0, bytes, body) }
pub fn upload<T>(bytes: usize, body: impl FnOnce() -> T) -> T { timed(1, bytes, body) }
pub fn quantize<T>(bytes: usize, body: impl FnOnce() -> T) -> T { timed(2, bytes, body) }
pub fn push<T>(bytes: usize, body: impl FnOnce() -> T) -> T { timed(3, bytes, body) }

/// A phase includes its caller and explicitly inherited worker contexts.
/// Nested phases also contribute to their enclosing phase, exactly once.
pub struct Split {
    started: Instant,
    totals: Arc<Mutex<Totals>>,
    // A split is thread-bound; only its Context may cross a thread boundary.
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Split {
    pub fn start() -> Self {
        let totals = Arc::new(Mutex::new(Totals::default()));
        ACTIVE.with(|active| active.borrow_mut().0.push(totals.clone()));
        Self { started: Instant::now(), totals, _thread: Default::default() }
    }
    pub fn read_bytes(&self) -> u64 { self.totals.lock().unwrap_or_else(|e| e.into_inner()).counters[0].bytes }
    pub fn push_bytes(&self) -> u64 { self.totals.lock().unwrap_or_else(|e| e.into_inner()).counters[3].bytes }
    pub fn log(self, label: &str) {
        let elapsed = self.started.elapsed().as_secs_f64();
        let totals = self.totals.lock().unwrap_or_else(|e| e.into_inner());
        let [read, upload, quantize, push] = totals.counters;
        let busy = io_wall_seconds(self.started, &totals.busy);
        tracing::info!(label, elapsed_seconds = elapsed, read_bytes = read.bytes, read_seconds = read.seconds,
            upload_bytes = upload.bytes, upload_seconds = upload.seconds,
            quantize_bytes = quantize.bytes, quantize_seconds = quantize.seconds,
            push_bytes = push.bytes, push_seconds = push.seconds,
            io_wall_seconds = busy, remainder_seconds = (elapsed - busy).max(0.0),
            "coordinator weight load split (operation seconds sum workers; async copies measure submission; remainder includes waits and uninstrumented transforms)");
    }
}
impl Drop for Split {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.borrow_mut().0.retain(|phase| !Arc::ptr_eq(phase, &self.totals)));
    }
}

fn io_wall_seconds(started: Instant, intervals: &[(Instant, Instant)]) -> f64 {
    let mut intervals = intervals.to_vec();
    intervals.sort_unstable_by_key(|&(start, _)| start);
    let mut end = started;
    let mut busy = 0.0;
    for (start, stop) in intervals {
        if stop > end { busy += (stop - start.max(end)).as_secs_f64(); end = stop; }
    }
    busy
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_receiver_threads_aggregate_without_unrelated_phase() {
        let split = Split::start();
        let context = capture();
        // SAFETY: integer results are unaliased and have no device work.
        unsafe { crate::synchronized_load::load_pair(|rank| {
            let nested = Split::start();
            read(1000, || ()); // inherited parent plus receiver-local bracket
            assert_eq!(nested.read_bytes(), 1000);
            context.enter(|| { read(rank + 3, || ()); upload(5, || ()); quantize(7, || ()); push(11, || ()); });
            Ok(rank)
        }) }.unwrap();
        // load_pair inherits automatically: the parent also sees the local work.
        assert_eq!(split.read_bytes(), 2007);
        let totals = split.totals.lock().unwrap();
        assert_eq!(totals.counters.map(|c| c.bytes), [2007, 10, 14, 22]);
        drop(totals);
        std::thread::spawn(|| { let other = Split::start(); read(99, || ()); assert_eq!(other.read_bytes(), 99); }).join().unwrap();
        assert_eq!(split.read_bytes(), 2007);
    }
    #[test]
    fn simultaneous_phases_do_not_cross_account() {
        let rendezvous = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            for bytes in [31usize, 73] {
                let rendezvous = &rendezvous;
                scope.spawn(move || {
                    let phase = Split::start();
                    let context = capture();
                    std::thread::scope(|workers| {
                        workers.spawn(|| context.enter(|| {
                            rendezvous.wait();
                            read(bytes, || ());
                            upload(bytes * 2, || ());
                            quantize(bytes * 3, || ());
                        }));
                    });
                    let totals = phase.totals.lock().unwrap();
                    assert_eq!(totals.counters.map(|counter| counter.bytes),
                        [bytes as u64, bytes as u64 * 2, bytes as u64 * 3, 0]);
                });
            }
        });
    }
    #[test]
    fn overlapping_worker_time_is_subtracted_only_once() {
        use std::time::Duration;
        let start = Instant::now();
        let at = |seconds| start + Duration::from_secs(seconds);
        // Seven summed worker seconds cover only four wall seconds.
        let intervals = [(at(2), at(5)), (at(1), at(4)), (at(2), at(3))];
        assert_eq!(io_wall_seconds(start, &intervals), 4.0);
        assert_eq!(io_wall_seconds(start, &[]), 0.0);
    }
    #[test]
    fn nested_and_panicking_contexts_restore_the_parent() {
        let parent = Split::start();
        {
            let child = Split::start();
            read(4, || ());
            assert_eq!(child.read_bytes(), 4);
        }
        let _ = std::panic::catch_unwind(|| Context::default().enter(|| panic!("fixture")));
        read(3, || ());
        assert_eq!(parent.read_bytes(), 7);
    }
}
