//! Device-side ordering for the stages of one target pass.
//!
//! Each layer used to drain every stage on the host before the next stage was
//! submitted (mHC finish, query, window KV, attention, shared FFN, reduction),
//! leaving the RTX idle while the host issued the following launch. Inside a
//! chain scope, a stage instead joins the chain's CUDA event before its first
//! enqueue and re-records that event after its last one. Consecutive stages on
//! different streams are therefore ordered on the device, and the host waits
//! only where it must read results (routes before dispatch, the head output).
//!
//! Safety argument: every stage of a scoped pass joins the chain, so the chain
//! is a total order over the pass's GPU work. The per-layer route download is
//! a host wait on a stream joined after all earlier stages, hence every stage
//! submitted before it, including pinned-staging uploads, has completed when
//! the host reuses that staging in the next layer. Work outside a scope keeps
//! its original synchronous behaviour.
//!
//! The scope is a thread-local set only while polling the pass future (the
//! same pattern as the device-scoped futures), so interleaved lanes on one
//! executor thread keep independent chains.
use super::LoadStream;
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use std::cell::Cell;
use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

#[derive(Clone)]
struct Current {
    /// One ordering event per device the pass runs on; (device, event).
    events: Rc<[(i32, *mut c_void)]>,
    /// Index of the event holding the chain head, if any stage finished.
    head: Rc<Cell<Option<usize>>>,
    /// Per-device fork events: a stage marks the point where its outputs that
    /// later producers read are complete, before its remaining work.
    forks: Rc<[(i32, *mut c_void)]>,
    fork: Rc<Cell<Option<usize>>>,
    /// Device-ordered passes: per device a stream and two events marking
    /// "everything chained so far" ([`fence_mark`]), and which device marked each.
    fences: Rc<[Fence]>,
    marked: Rc<Cell<[Option<usize>; 2]>>,
}

#[derive(Clone, Copy)]
struct Fence {
    device: i32,
    stream: *mut c_void,
    events: [*mut c_void; 2],
}

thread_local! {
    static CURRENT: std::cell::RefCell<Option<Current>> = const { std::cell::RefCell::new(None) };
}

/// Ordering events for one pass owner, one per participating device.
pub(crate) struct StageChain<'a> {
    library: &'a NativeLibrary,
    events: Rc<[(i32, *mut c_void)]>,
    head: Rc<Cell<Option<usize>>>,
    forks: Rc<[(i32, *mut c_void)]>,
    fork: Rc<Cell<Option<usize>>>,
    fences: Rc<[Fence]>,
    marked: Rc<Cell<[Option<usize>; 2]>>,
}
impl<'a> StageChain<'a> {
    /// A chain for the current device only.
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        let device = library.cuda_get_device()?;
        Self::on_devices(library, &[device])
    }
    /// A chain spanning `devices`; each event is created on its own device.
    pub fn on_devices(library: &'a NativeLibrary, devices: &[i32]) -> Result<Self> {
        let previous = library.cuda_get_device()?;
        let mut events = Vec::with_capacity(devices.len());
        let mut forks = Vec::with_capacity(devices.len());
        let mut fences = Vec::with_capacity(devices.len());
        let created = (|| -> Result<()> {
            for &device in devices {
                library.cuda_set_device(device)?;
                events.push((device, library.cuda_event_create_ordering()?));
                forks.push((device, library.cuda_event_create_ordering()?));
                if device_enabled() {
                    fences.push(Fence { device, stream: library.cuda_stream_create()?,
                        events: [library.cuda_event_create_ordering()?, library.cuda_event_create_ordering()?] });
                }
            }
            Ok(())
        })();
        library.cuda_set_device(previous)?;
        if let Err(error) = created {
            for &(_, event) in events.iter().chain(&forks) { let _ = unsafe { library.cuda_event_destroy(event) }; }
            destroy_fences(library, &fences);
            return Err(error);
        }
        Ok(Self { library, events: events.into(), head: Rc::new(Cell::new(None)),
            forks: forks.into(), fork: Rc::new(Cell::new(None)), fences: fences.into(),
            marked: Rc::new(Cell::new([None; 2])) })
    }
    /// An owned handle that can wrap a future borrowing the chain's owner.
    pub fn handle(&self) -> ChainHandle {
        ChainHandle(Current { events: self.events.clone(), head: self.head.clone(),
            forks: self.forks.clone(), fork: self.fork.clone(), fences: self.fences.clone(),
            marked: self.marked.clone() })
    }
    /// Host wait for everything recorded so far, then forget the head. Call
    /// after the pass (or an aborted pass) before any unscoped consumer.
    pub fn drain(&self) -> Result<()> {
        self.fork.set(None);
        self.marked.set([None; 2]);
        if let Some(head) = self.head.replace(None) {
            unsafe { self.library.cuda_event_synchronize(self.events[head].1)?; }
        }
        Ok(())
    }
}
impl Drop for StageChain<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.drain() {
            tracing::error!(%error, "draining target stage chain");
        }
        for &(_, event) in self.events.iter().chain(self.forks.iter()) {
            if let Err(error) = unsafe { self.library.cuda_event_destroy(event) } {
                tracing::error!(%error, "destroying target stage chain event");
            }
        }
        destroy_fences(self.library, &self.fences);
    }
}

fn destroy_fences(library: &NativeLibrary, fences: &[Fence]) {
    for fence in fences {
        // SAFETY: created by this chain; drained before destruction.
        unsafe {
            let _ = library.cuda_stream_synchronize(fence.stream);
            let _ = library.cuda_stream_destroy(fence.stream);
            for event in fence.events { let _ = library.cuda_event_destroy(event); }
        }
    }
}

/// Device-ordered passes: marks fence `slot` (0 or 1) at the current chain
/// head, without a host wait. A later [`fence_wait`] returns once every
/// stage chained before the mark has completed, so host staging those stages
/// uploaded from may be rewritten.
pub(crate) fn fence_mark(library: &NativeLibrary, slot: usize) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    let Some(head) = current.head.get() else { return Ok(()) };
    let fence = current.fences.iter().position(|f| f.device == current.events[head].0)
        .map(|i| (i, current.fences[i]));
    let Some((index, fence)) = fence else { return Ok(()) };
    let previous = library.cuda_get_device()?;
    library.cuda_set_device(fence.device)?;
    // SAFETY: the fence stream and events belong to this chain's device.
    let marked = unsafe {
        library.cuda_stream_wait_event(fence.stream, current.events[head].1)
            .and_then(|()| library.cuda_event_record(fence.events[slot], fence.stream))
    };
    library.cuda_set_device(previous)?;
    marked?;
    let mut slots = current.marked.get();
    slots[slot] = Some(index);
    current.marked.set(slots);
    Ok(())
}

/// Waits for fence `slot` (no-op when it was not marked since the last wait),
/// yielding between polls so another lane on this thread keeps running.
pub(crate) async fn fence_wait(library: &NativeLibrary, slot: usize) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    let mut slots = current.marked.get();
    let Some(index) = slots[slot].take() else { return Ok(()) };
    current.marked.set(slots);
    let fence = current.fences[index];
    // The fence stream holds only marks, and the next mark is queued after this
    // wait: the stream is idle exactly when this slot's mark has completed.
    loop {
        let previous = library.cuda_get_device()?;
        library.cuda_set_device(fence.device)?;
        // SAFETY: the fence stream belongs to this chain.
        let ready = unsafe { library.cuda_stream_query(fence.stream) };
        library.cuda_set_device(previous)?;
        if ready? {
            return Ok(());
        }
        tokio::task::yield_now().await;
    }
}

pub(crate) struct ChainHandle(Current);
impl ChainHandle {
    /// Poll `future` with this chain installed as the current scope.
    pub fn scope<F: Future>(self, future: F) -> ChainScope<F> {
        ChainScope { current: self.0, future }
    }
}

/// Whether target passes order their stages on the device (default) or drain
/// each stage on the host (`CUTEAFD_STAGE_CHAIN=0`, the pre-v14 behaviour).
pub(crate) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CUTEAFD_STAGE_CHAIN").map_or(true, |v| v != "0"))
}

/// Whether chained V4.1 passes also drop the host waits that only kept
/// copy-engine transfers from queuing behind unresolved events
/// (`CUTEAFD_V41_DEVICE=1`, PLAN.md device-driven exchange, stage D3): peer
/// transfers become SM copies ordered by device events and the host enqueues
/// the next stage at once. Off by default; the default path is unchanged.
pub(crate) fn device_enabled() -> bool {
    device_setting() > 0
}

/// Whether V4.1 remote verification waves also use the device-driven Spark
/// exchange (`CUTEAFD_V41_DEVICE=1`; `chain` keeps them on the host path).
pub(crate) fn device_exchange_enabled() -> bool {
    device_setting() > 1
}

fn device_setting() -> u8 {
    static SETTING: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *SETTING.get_or_init(|| {
        let setting = match std::env::var("CUTEAFD_V41_DEVICE").as_deref() {
            Ok("1" | "on" | "true") => 2,
            Ok("chain") => 1,
            _ => 0,
        };
        if setting > 0 {
            tracing::info!(exchange = setting > 1, "V4.1 device-ordered passes: chained stages enqueue without host waits");
        }
        setting
    })
}

/// Inside a chain scope with [`device_enabled`]: stages that used to wait on
/// the host for a producer stream join the chain instead.
pub(crate) fn deferred() -> bool {
    active() && device_enabled()
}

pub(crate) struct ChainScope<F> {
    current: Current,
    future: F,
}
impl<F: Future> Future for ChainScope<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<F::Output> {
        // The future is never moved after pinning; only borrowed in place.
        let this = unsafe { self.get_unchecked_mut() };
        let previous = CURRENT.with(|c| c.replace(Some(this.current.clone())));
        struct Restore(Option<Current>);
        impl Drop for Restore {
            fn drop(&mut self) { let previous = self.0.take(); CURRENT.with(|c| *c.borrow_mut() = previous); }
        }
        let _restore = Restore(previous);
        unsafe { Pin::new_unchecked(&mut this.future) }.poll(context)
    }
}

fn current() -> Option<Current> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Whether stages are currently device-ordered instead of host-drained.
pub(crate) fn active() -> bool {
    CURRENT.with(|c| c.borrow().is_some())
}

/// Order `stream` after the chain head. Call before a stage's first enqueue.
/// # Safety
/// `stream` is a live stream on the chain's device.
pub(crate) unsafe fn join(library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    // A full join ends any fork window: later stages see the whole chain.
    current.fork.set(None);
    let Some(head) = current.head.get() else { return Ok(()) };
    // Cross-device event waits are permitted; the event lives on its own device.
    unsafe { library.cuda_stream_wait_event(stream, current.events[head].1) }
}

/// Mark a fork on `stream`: work queued so far on it (for example the query
/// stage's normalized layer input) is what fork joiners depend on, while the
/// stage's later work (the query projections) may overlap theirs.
/// # Safety
/// `stream` is a live stream on a chain device, already joined to the head.
pub(crate) unsafe fn mark_fork(library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    let device = library.cuda_get_device()?;
    let Some(index) = current.forks.iter().position(|&(d, _)| d == device) else { return Ok(()) };
    unsafe { library.cuda_event_record(current.forks[index].1, stream)?; }
    current.fork.set(Some(index));
    Ok(())
}

/// Join the current fork instead of the head, so this stage overlaps the rest
/// of the forking stage. Its `finish` still merges the head. Without a fork
/// (none marked since the last full join) this is [`join`].
/// # Safety
/// The stage reads only outputs complete at the fork (plus its own state).
pub(crate) unsafe fn join_fork(library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    match current.fork.get() {
        Some(fork) => unsafe { library.cuda_stream_wait_event(stream, current.forks[fork].1) },
        None => unsafe { join(library, stream) },
    }
}

/// Complete a stage: record the chain head on `stream` inside a scope,
/// otherwise drain `stream` on the host exactly as before.
/// # Safety
/// `stream` holds this stage's queued work on the chain's device.
pub(crate) unsafe fn finish(library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
    let Some(current) = current() else { return unsafe { library.cuda_stream_synchronize(stream) } };
    let device = library.cuda_get_device()?;
    let Some(index) = current.events.iter().position(|&(d, _)| d == device) else {
        // A device outside this chain: complete the stage on the host.
        return unsafe { library.cuda_stream_synchronize(stream) };
    };
    // Merge: the stream first waits for the previous head, so parallel branches
    // (window, compressor, index projection) all precede the new head.
    if let Some(head) = current.head.get() {
        unsafe { library.cuda_stream_wait_event(stream, current.events[head].1)?; }
    }
    unsafe { library.cuda_event_record(current.events[index].1, stream)?; }
    current.head.set(Some(index));
    Ok(())
}

/// Cooperative form of [`finish`]: record inside a scope, otherwise yield
/// until the stream completes.
/// # Safety
/// Same as [`finish`].
pub(crate) async unsafe fn finish_cooperative(stream: &LoadStream<'_>) -> Result<()> {
    if active() {
        unsafe { finish(stream.library, stream.raw) }
    } else {
        stream.wait().await
    }
}

/// Host wait for all chained work before a host-synchronous operation (legacy
/// stream copies do not order with the non-blocking stage streams).
pub(crate) fn settle(library: &NativeLibrary) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    if let Some(head) = current.head.get() {
        unsafe { library.cuda_event_synchronize(current.events[head].1)?; }
    }
    Ok(())
}
