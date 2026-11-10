//! Opt-in routing census (`CUTEAFD_V41_ROUTE_CENSUS=<path>`): per routed
//! layer, how many live rows chose each expert while serving, written as the
//! `cuteafd plan --gpu-expert-tier-profile` file
//! ([`cuteafd_loader::placement::ExpertRouteCensus`]).
//!
//! - **Counting** runs on the GPU that runs each layer's router: one small
//!   kernel after the router reads its top-k ids and adds them into that GPU's
//!   histogram (`cuteafd_ffi::RouteCensusKernels`). The launch is part of the
//!   router's captured graph (fixed pointers, no host synchronisation) and
//!   writes nothing the model reads, so outputs stay byte-identical. It counts
//!   the router's rows, which are exactly the live rows its experts receive.
//! - **Phases:** decode and verify passes count into `layers` (what a GPU
//!   expert tier serves), prefill passes (encoder units, decoder replay) into
//!   `prefill_layers`. Each router stream owns a [`PhaseSelector`] byte that
//!   the kernel reads on the device; the host rewrites it in stream order only
//!   when that stream's pass phase changes, so one captured graph counts every
//!   replay in the phase of the pass replaying it.
//! - **Off by default:** without the variable no router holds a
//!   [`RouteCounter`], so nothing is allocated and routers launch nothing more.
//! - **Reading** happens only where the serve loop has drained every lane:
//!   [`RouteCensus::tick`] every `CUTEAFD_V41_ROUTE_CENSUS_STEPS` decode
//!   steps (default 1000) and [`RouteCensus::finish`] at shutdown. A writer
//!   thread replaces the file by atomic rename.
//! - **Two GPUs:** each layer's router runs on one GPU, which counts it; the
//!   file is the sum over the GPUs.
//! - **Trace:** beside the file, `NAME.windows.jsonl` gets one line per
//!   window ([`cuteafd_loader::placement::RouteTrace`]): the routes counted
//!   since the previous window, read at a drained boundary once the window
//!   spans `CUTEAFD_V41_ROUTE_CENSUS_WINDOW` decode steps (default 64; 0
//!   writes no trace) or where a request finishes, with the requests it saw.
//!   It is the same device counts read more often, so serving is unchanged.
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, RouteCensusInput, RouteCensusKernels, RouteCensusShape};
use cuteafd_loader::placement::{ExpertRouteCensus, RouteTraceLine, RouteTraceSession, RouteWindow};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::SystemTime;

pub(crate) const PATH_ENV: &str = "CUTEAFD_V41_ROUTE_CENSUS";
pub(crate) const STEPS_ENV: &str = "CUTEAFD_V41_ROUTE_CENSUS_STEPS";
const DEFAULT_STEPS: u64 = 1000;
pub(crate) const WINDOW_ENV: &str = "CUTEAFD_V41_ROUTE_CENSUS_WINDOW";
const DEFAULT_WINDOW: u64 = 64;
/// Phase selector bytes per GPU, one per router stream (each decode lane's
/// router storage and the prefill slots' shared one).
const SELECTORS: usize = 64;

/// The pass a router serves: decode and verify (plane 0) or prefill (plane 1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RoutePhase {
    Decode,
    #[default]
    Prefill,
}

impl RoutePhase {
    pub(crate) fn of_pass(decode: bool) -> Self {
        if decode { Self::Decode } else { Self::Prefill }
    }
    fn plane(self) -> usize {
        match self { Self::Decode => 0, Self::Prefill => 1 }
    }
}

/// Where and how often the census is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Settings {
    pub path: PathBuf,
    /// Decode steps between periodic dumps.
    pub steps: u64,
    /// Decode steps a trace window spans at least; 0 writes no trace.
    pub window: u64,
}

impl Settings {
    /// `None` unless [`PATH_ENV`] names a file.
    pub(crate) fn from_vars(get: impl Fn(&str) -> Option<String>) -> Result<Option<Self>> {
        let Some(path) = get(PATH_ENV).filter(|path| !path.is_empty()) else { return Ok(None) };
        let steps = match get(STEPS_ENV).filter(|steps| !steps.is_empty()) {
            None => DEFAULT_STEPS,
            Some(value) => value.parse().ok().filter(|&steps: &u64| steps > 0)
                .with_context(|| format!("{STEPS_ENV} must be a positive decode step count, not {value:?}"))?,
        };
        let window = match get(WINDOW_ENV).filter(|window| !window.is_empty()) {
            None => DEFAULT_WINDOW,
            Some(value) => value.parse().ok().with_context(||
                format!("{WINDOW_ENV} must be a decode step count (0: no trace), not {value:?}"))?,
        };
        Ok(Some(Self { path: path.into(), steps, window }))
    }

    /// The trace beside the census file (`NAME.windows.jsonl` for `NAME.json`).
    pub(crate) fn trace_path(&self) -> Option<PathBuf> {
        (self.window > 0).then(|| self.path.with_extension("windows.jsonl"))
    }
}

struct Gpu<'a> {
    /// `[planes][layers][experts]` U64 counts, then the selector bytes.
    counts: Allocation<'a>,
    /// Selector bytes in use, one bit each.
    selectors: Cell<u64>,
}

struct Shared<'a> {
    library: &'a NativeLibrary,
    kernels: RouteCensusKernels<'a>,
    shape: RouteCensusShape,
    gpus: Vec<Gpu<'a>>,
    /// Decode and verify passes counted (at their first routed layer).
    steps: Cell<u64>,
}

impl Shared<'_> {
    fn counts(&self, gpu: usize) -> CuteafdDeviceBuffer {
        CuteafdDeviceBuffer { bytes: self.shape.bytes(), ..self.gpus[gpu].counts.buffer }
    }
}

/// The census of one serving process: every coordinator GPU's counts.
pub(crate) struct RouteCensus<'a> {
    shared: Rc<Shared<'a>>,
    settings: Settings,
    model: String,
    started: Cell<SystemTime>,
    /// The step count at which the next periodic dump is due.
    due: Cell<u64>,
    writer: RefCell<Option<std::thread::JoinHandle<()>>>,
    trace: Option<Trace>,
}

/// The windowed trace: the open window and the thread appending closed ones.
struct Trace {
    windows: RefCell<Windows>,
    writer: TraceWriter,
}

impl<'a> RouteCensus<'a> {
    /// The census [`PATH_ENV`] asks for, counting the routers on `gpus`
    /// (`None` when it is unset).
    pub(crate) fn from_env(library: &'a NativeLibrary, gpus: &[i32], shape: RouteCensusShape, model: String)
        -> Result<Option<Self>> {
        let Some(settings) = Settings::from_vars(|name| std::env::var(name).ok())? else { return Ok(None) };
        Self::new(library, gpus, shape, model, settings).map(Some)
    }

    /// Zeroed counts and selectors on each of `gpus`, outside every serving
    /// owner's ledger scope (the census is not part of the plan).
    pub(crate) fn new(library: &'a NativeLibrary, gpus: &[i32], shape: RouteCensusShape, model: String,
        settings: Settings) -> Result<Self> {
        shape.validate()?;
        ensure!(!gpus.is_empty() && (1..gpus.len()).all(|i| !gpus[..i].contains(&gpus[i])),
            "route census GPUs must be distinct: {gpus:?}");
        let kernels = library.route_census().context("route census needs a native library with its kernels")?;
        // An unwritable census fails at startup, not at its first dump.
        probe_writable(&settings.path)?;
        let trace = settings.trace_path().map(|path| -> Result<Trace> {
            Ok(Trace { windows: RefCell::new(Windows::new(settings.window, shape, SystemTime::now())),
                writer: TraceWriter::open(&path)? })
        }).transpose()?;
        let gpus = gpus.iter().map(|&id| {
            let device = Device { library, id };
            let _scope = cuteafd_ffi::memory_ledger::scope("diagnostics/route-census");
            let counts = Allocation::new(device, shape.bytes() + SELECTORS)?;
            device.run(|| library.cuda_zero_bytes(counts.buffer, counts.buffer.bytes))?;
            Ok(Gpu { counts, selectors: Cell::new(0) })
        }).collect::<Result<Vec<_>>>()?;
        tracing::info!(path = %settings.path.display(), steps = settings.steps, window = settings.window,
            layers = shape.layers, experts = shape.experts, gpus = gpus.len(), "route census on");
        let due = Cell::new(settings.steps);
        Ok(Self { shared: Rc::new(Shared { library, kernels, shape, gpus, steps: Cell::new(0) }), settings, model,
            started: Cell::new(SystemTime::now()), due, writer: RefCell::new(None), trace })
    }

    /// Routed layer `layer`'s binding, for its router on GPU `gpu`.
    pub(crate) fn counter(&self, gpu: i32, layer: usize) -> Result<RouteCounter<'a>> {
        ensure!(layer < self.shared.shape.layers, "route census layer {layer} outside the model");
        let gpu = self.shared.gpus.iter().position(|g| g.counts.device.id == gpu)
            .with_context(|| format!("route census does not count GPU {gpu}"))?;
        Ok(RouteCounter { shared: self.shared.clone(), gpu, layer })
    }

    /// Serving begins with every lane idle: counting starts from zero here,
    /// so nothing a startup path routed enters the census.
    pub(crate) fn begin(&self) -> Result<()> {
        for (index, gpu) in self.shared.gpus.iter().enumerate() {
            gpu.counts.device.run(|| self.shared.library.cuda_zero_bytes(self.shared.counts(index),
                self.shared.shape.bytes()))?;
        }
        self.shared.steps.set(0);
        self.due.set(self.settings.steps);
        let started = SystemTime::now();
        self.started.set(started);
        if let Some(trace) = &self.trace {
            *trace.windows.borrow_mut() = Windows::new(self.settings.window, self.shared.shape, started);
            let shape = self.shared.shape;
            trace.writer.send(RouteTraceLine::Session(RouteTraceSession { model: self.model.clone(),
                started: cuteafd_bench::report::rfc3339(started), layers: shape.layers, experts: shape.experts,
                topk: shape.topk, window: self.settings.window }));
        }
        Ok(())
    }

    /// At the serve loop's drained boundary, with the requests in serving
    /// (id, finished): close a due trace window, and once another interval
    /// of decode steps ran, hand the counts to a writer thread. Never fails
    /// serving: a failed read is logged and retried at the next boundary.
    pub(crate) fn tick(&self, active: impl IntoIterator<Item = (u64, bool)>) {
        let steps = self.shared.steps.get();
        let window = self.trace.as_ref().is_some_and(|trace| trace.windows.borrow_mut().note(steps, active));
        // A previous file still being written waits for a later boundary.
        let file = steps >= self.due.get() && self.writer.borrow().as_ref().is_none_or(|writer| writer.is_finished());
        if !window && !file { return; }
        let counts = match self.counts() {
            Ok(counts) => counts,
            Err(error) => {
                tracing::warn!(error = %format!("{error:#}"), "reading the route census");
                return;
            }
        };
        let time = SystemTime::now();
        if let Some(trace) = self.trace.as_ref().filter(|_| window) {
            let closed = trace.windows.borrow_mut().close(&counts, steps, time);
            trace.writer.send(RouteTraceLine::Window(closed));
        }
        if !file { return; }
        self.join_writer();
        self.due.set((steps / self.settings.steps + 1) * self.settings.steps);
        let census = assemble(&counts, self.shared.shape, steps, self.model.clone(), self.started.get(), time);
        let path = self.settings.path.clone();
        match std::thread::Builder::new().name("route-census".into()).spawn(move || write_logged(&census, &path)) {
            Ok(writer) => *self.writer.borrow_mut() = Some(writer),
            Err(error) => tracing::warn!(%error, "starting the route census writer"),
        }
    }

    /// Clean shutdown, every lane drained: the final census, written before
    /// this returns.
    pub(crate) fn finish(&self) {
        self.join_writer();
        let counts = match self.counts() {
            Ok(counts) => counts,
            Err(error) => return tracing::warn!(error = %format!("{error:#}"), "reading the final route census"),
        };
        let (steps, time) = (self.shared.steps.get(), SystemTime::now());
        if let Some(trace) = &self.trace {
            let mut windows = trace.windows.borrow_mut();
            if windows.counted(&counts) {
                trace.writer.send(RouteTraceLine::Window(windows.close(&counts, steps, time)));
            }
        }
        write_logged(&assemble(&counts, self.shared.shape, steps, self.model.clone(), self.started.get(), time),
            &self.settings.path);
    }

    /// Every GPU's counts, summed (each layer's routes live on one GPU).
    fn counts(&self) -> Result<Vec<u64>> {
        let shape = self.shared.shape;
        let mut counts = vec![0u64; shape.bytes() / 8];
        let mut bytes = vec![0u8; shape.bytes()];
        for (index, gpu) in self.shared.gpus.iter().enumerate() {
            gpu.counts.device.run(|| self.shared.library.copy_d2h(&mut bytes, self.shared.counts(index)))?;
            for (total, word) in counts.iter_mut().zip(bytes.as_chunks::<8>().0) {
                *total += u64::from_ne_bytes(*word);
            }
        }
        Ok(counts)
    }

    fn join_writer(&self) {
        if let Some(writer) = self.writer.borrow_mut().take() {
            if writer.join().is_err() { tracing::warn!("the route census writer panicked"); }
        }
    }
}

impl Drop for RouteCensus<'_> {
    fn drop(&mut self) { self.join_writer(); }
}

/// One phase's `[layers][experts]` counts of summed device counts
/// (`[planes][layers][experts]`).
fn plane(counts: &[u64], shape: RouteCensusShape, phase: RoutePhase) -> Vec<Vec<u64>> {
    counts[phase.plane() * shape.plane_words()..][..shape.plane_words()]
        .chunks_exact(shape.experts).map(<[u64]>::to_vec).collect()
}

/// Rows routed: every row routes `topk` experts through each layer it reaches.
fn rows(layers: &[Vec<u64>], shape: RouteCensusShape) -> u64 {
    layers.first().map_or(0, |row| row.iter().sum::<u64>() / shape.topk as u64)
}

/// The census file from summed device counts.
fn assemble(counts: &[u64], shape: RouteCensusShape, steps: u64, model: String, started: SystemTime,
    time: SystemTime) -> ExpertRouteCensus {
    let (layers, prefill_layers) = (plane(counts, shape, RoutePhase::Decode), plane(counts, shape, RoutePhase::Prefill));
    ExpertRouteCensus {
        model,
        started: cuteafd_bench::report::rfc3339(started),
        time: cuteafd_bench::report::rfc3339(time),
        steps,
        tokens: rows(&layers, shape),
        prefill_tokens: rows(&prefill_layers, shape),
        experts: shape.experts,
        topk: shape.topk,
        layers,
        prefill_layers,
    }
}

/// The trace's open window: the summed counts and step count it opened at,
/// and the requests the boundaries inside it saw.
struct Windows {
    every: u64,
    shape: RouteCensusShape,
    index: u64,
    counts: Vec<u64>,
    step: u64,
    time: SystemTime,
    requests: BTreeSet<u64>,
    finished: BTreeSet<u64>,
}

impl Windows {
    fn new(every: u64, shape: RouteCensusShape, time: SystemTime) -> Self {
        Self { every, shape, index: 0, counts: vec![0; shape.bytes() / 8], step: 0, time,
            requests: BTreeSet::new(), finished: BTreeSet::new() }
    }

    /// Notes the requests in serving at a drained boundary at `steps`
    /// decode steps: whether the window closes here (it spans `every` steps,
    /// or a request finishes after decode steps in it). A request that
    /// finishes before any step of the window is listed in the next one.
    fn note(&mut self, steps: u64, active: impl IntoIterator<Item = (u64, bool)>) -> bool {
        let mut finishing = false;
        for (id, finished) in active {
            self.requests.insert(id);
            if finished { finishing |= self.finished.insert(id); }
        }
        steps > self.step && (steps - self.step >= self.every || finishing)
    }

    /// Whether `counts` holds routes the window has not reported.
    fn counted(&self, counts: &[u64]) -> bool { counts != self.counts.as_slice() }

    /// Closes the window at summed `counts` and `steps`, and opens the next.
    fn close(&mut self, counts: &[u64], steps: u64, time: SystemTime) -> RouteWindow {
        let delta: Vec<u64> = counts.iter().zip(&self.counts).map(|(now, then)| now.saturating_sub(*then)).collect();
        let (layers, prefill_layers) = (plane(&delta, self.shape, RoutePhase::Decode),
            plane(&delta, self.shape, RoutePhase::Prefill));
        let window = RouteWindow { index: self.index, steps: [self.step, steps],
            tokens: rows(&layers, self.shape), prefill_tokens: rows(&prefill_layers, self.shape),
            started: cuteafd_bench::report::rfc3339(self.time), time: cuteafd_bench::report::rfc3339(time),
            requests: std::mem::take(&mut self.requests).into_iter().collect(),
            finished: std::mem::take(&mut self.finished).into_iter().collect(), layers, prefill_layers };
        self.index += 1;
        self.counts.copy_from_slice(counts);
        (self.step, self.time) = (steps, time);
        window
    }
}

/// Appends trace lines on its own thread, so the serve loop only reads the
/// counts; each line is flushed whole. Dropping it writes what is queued.
struct TraceWriter {
    sender: Option<std::sync::mpsc::Sender<RouteTraceLine>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl TraceWriter {
    fn open(path: &Path) -> Result<Self> {
        use std::io::Write;
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path)
            .with_context(|| format!("opening the route census trace {}", path.display()))?;
        let (sender, receiver) = std::sync::mpsc::channel::<RouteTraceLine>();
        let path = path.to_owned();
        let thread = std::thread::Builder::new().name("route-trace".into()).spawn(move || {
            let mut file = std::io::BufWriter::new(file);
            let mut failed = false;
            for line in receiver {
                let written = serde_json::to_writer(&mut file, &line).map_err(std::io::Error::from)
                    .and_then(|()| file.write_all(b"\n")).and_then(|()| file.flush());
                if let Err(error) = written {
                    // Once: a full disk would otherwise log every window.
                    if !failed { tracing::warn!(path = %path.display(), %error, "writing the route census trace"); }
                    failed = true;
                }
            }
        }).context("starting the route census trace writer")?;
        Ok(Self { sender: Some(sender), thread: Some(thread) })
    }

    fn send(&self, line: RouteTraceLine) {
        if let Some(sender) = &self.sender { let _ = sender.send(line); }
    }
}

impl Drop for TraceWriter {
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() { tracing::warn!("the route census trace writer panicked"); }
        }
    }
}

/// Writes `census` unless it counted nothing: a server stopped before any
/// traffic leaves an earlier census at the same path in place.
fn write_logged(census: &ExpertRouteCensus, path: &Path) {
    if census.layers.iter().chain(&census.prefill_layers).flatten().all(|&count| count == 0) {
        tracing::info!(path = %path.display(), "route census counted no routes; the file is left as it was");
        return;
    }
    match census.write(path) {
        Ok(()) => tracing::info!(path = %path.display(), steps = census.steps, tokens = census.tokens,
            prefill_tokens = census.prefill_tokens, "route census written"),
        Err(error) => tracing::warn!(%error, "writing the route census"),
    }
}

/// Creates and removes a sibling of `path`: its directory exists and is writable.
fn probe_writable(path: &Path) -> Result<()> {
    let name = path.file_name().with_context(|| format!("{PATH_ENV}={} names no file", path.display()))?;
    let probe = path.with_file_name(format!(".{}.{}.probe", name.to_string_lossy(), std::process::id()));
    std::fs::write(&probe, b"")
        .with_context(|| format!("{PATH_ENV}={}: its directory is not writable", path.display()))?;
    std::fs::remove_file(&probe).with_context(|| format!("removing {}", probe.display()))
}

/// One routed layer's census binding, held by its router weights.
pub(crate) struct RouteCounter<'a> {
    shared: Rc<Shared<'a>>,
    gpu: usize,
    layer: usize,
}

impl<'a> RouteCounter<'a> {
    pub(crate) fn device_id(&self) -> i32 { self.shared.gpus[self.gpu].counts.device.id }

    /// A selector byte for one router stream on this layer's GPU, from the
    /// census allocation (nothing is allocated).
    pub(crate) fn selector(&self) -> Result<PhaseSelector<'a>> {
        let taken = &self.shared.gpus[self.gpu].selectors;
        let slot = (!taken.get()).trailing_zeros() as usize;
        ensure!(slot < SELECTORS, "route census: more than {SELECTORS} router streams on GPU {}", self.device_id());
        taken.set(taken.get() | 1 << slot);
        Ok(PhaseSelector { shared: self.shared.clone(), gpu: self.gpu, slot, written: Cell::new(None) })
    }

    /// Queues the count of `rows` rows of `ids` (`[rows][topk]` U32) into
    /// this layer's row of the plane `selector` holds when the kernel runs.
    /// Capturable: fixed pointers, no host synchronisation.
    ///
    /// # Safety
    /// `ids` holds this layer's router output, written earlier on `stream`,
    /// which is `selector`'s router stream on this layer's GPU.
    pub(crate) unsafe fn count(&self, ids: CuteafdDeviceBuffer, rows: u32, selector: &PhaseSelector<'a>,
        stream: *mut std::ffi::c_void) -> Result<()> {
        ensure!(selector.gpu == self.gpu, "route census selector belongs to another GPU");
        // SAFETY: the census owns `counts` and the selector byte; the caller
        // orders `ids` and the selector's writes on `stream`.
        unsafe {
            self.shared.kernels.count(RouteCensusInput { ids, live: None, rows: rows as usize }, selector.byte(),
                self.shared.counts(self.gpu), self.shared.shape, self.layer, stream)
        }
    }

    /// This layer's router ran in a pass of `phase`: the first routed layer
    /// counts the decode steps.
    pub(crate) fn observe(&self, phase: RoutePhase) {
        if self.layer == 0 && phase == RoutePhase::Decode {
            self.shared.steps.set(self.shared.steps.get() + 1);
        }
    }
}

/// One router stream's plane selector: a device byte the count kernel reads.
pub(crate) struct PhaseSelector<'a> {
    shared: Rc<Shared<'a>>,
    gpu: usize,
    slot: usize,
    /// The phase last written on the stream (`None`: not written yet).
    written: Cell<Option<RoutePhase>>,
}

impl PhaseSelector<'_> {
    pub(crate) fn device_id(&self) -> i32 { self.shared.gpus[self.gpu].counts.device.id }

    fn byte(&self) -> CuteafdDeviceBuffer {
        let base = self.shared.gpus[self.gpu].counts.buffer;
        // SAFETY: the allocation holds the counts, then SELECTORS bytes.
        CuteafdDeviceBuffer { ptr: unsafe { base.ptr.cast::<u8>().add(self.shared.shape.bytes() + self.slot) }.cast(),
            bytes: 1, ..base }
    }

    /// Before a router launch on `stream`, never inside a capture: counts
    /// queued after this go to `phase`'s plane. Writes only on a change.
    ///
    /// # Safety
    /// `stream` is this selector's router stream, on its GPU.
    pub(crate) unsafe fn select(&self, phase: RoutePhase, stream: *mut std::ffi::c_void) -> Result<()> {
        if self.written.get() == Some(phase) { return Ok(()); }
        // SAFETY: a one-byte write into the census allocation, in stream order.
        unsafe { self.shared.kernels.select(self.byte(), phase.plane(), stream)?; }
        self.written.set(Some(phase));
        Ok(())
    }
}

impl Drop for PhaseSelector<'_> {
    fn drop(&mut self) {
        // The owning router storage drained its stream first.
        let taken = &self.shared.gpus[self.gpu].selectors;
        taken.set(taken.get() & !(1 << self.slot));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::placement::ExpertRouteProfile;

    const V41: RouteCensusShape = RouteCensusShape { layers: 40, experts: 384, topk: 6 };

    fn settings(path: Option<&str>, steps: Option<&str>) -> Result<Option<Settings>> {
        windowed(path, steps, None)
    }

    fn windowed(path: Option<&str>, steps: Option<&str>, window: Option<&str>) -> Result<Option<Settings>> {
        Settings::from_vars(|name| match name {
            PATH_ENV => path.map(str::to_owned),
            STEPS_ENV => steps.map(str::to_owned),
            WINDOW_ENV => window.map(str::to_owned),
            _ => None,
        })
    }

    #[test]
    fn settings_come_from_the_environment_only_when_named() {
        assert_eq!(settings(None, Some("5")).unwrap(), None);
        assert_eq!(settings(Some(""), None).unwrap(), None);
        assert_eq!(settings(Some("/p/census.json"), None).unwrap(),
            Some(Settings { path: "/p/census.json".into(), steps: 1000, window: 64 }));
        assert_eq!(settings(Some("/p/census.json"), Some("")).unwrap().unwrap().steps, 1000);
        assert_eq!(settings(Some("/p/census.json"), Some("250")).unwrap().unwrap().steps, 250);
        for steps in ["0", "-3", "ten"] {
            assert!(settings(Some("/p/census.json"), Some(steps)).is_err(), "{steps}");
        }
        // The trace: beside the file, every 64 steps unless told otherwise; 0 writes none.
        let census = |window| windowed(Some("/p/route-census-h.json"), None, window).unwrap().unwrap();
        assert_eq!(census(None).trace_path(), Some("/p/route-census-h.windows.jsonl".into()));
        assert_eq!((census(Some("")).window, census(Some("16")).window), (64, 16));
        assert_eq!(census(Some("0")).trace_path(), None);
        for window in ["-1", "many"] {
            assert!(windowed(Some("/p/census.json"), None, Some(window)).is_err(), "{window}");
        }
    }

    /// Windows close at boundaries `every` steps apart or where a request
    /// finishes after steps in the window, never without steps; they hold
    /// the counts since the previous close, so they sum to the file's totals,
    /// and list the requests their boundaries saw.
    #[test]
    fn trace_windows_close_on_steps_or_a_finish_and_sum_to_the_totals() {
        let shape = RouteCensusShape { layers: 2, experts: 4, topk: 2 };
        let start = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_790_000_000);
        let mut windows = Windows::new(4, shape, start);
        let mut counts = vec![0u64; shape.bytes() / 8];
        let mut closed = Vec::new();
        // (steps at the boundary, requests in serving, decode rows routed since the last boundary)
        let boundaries: [(u64, &[(u64, bool)], u64); 7] = [(0, &[(7, false)], 0), (3, &[(7, false)], 3),
            (4, &[(7, false)], 1), (6, &[(7, true)], 2), (6, &[(8, true)], 0), (9, &[(9, false)], 3),
            (13, &[(9, false)], 4)];
        for (round, &(steps, active, rows)) in boundaries.iter().enumerate() {
            // Every decode row takes experts (round % 4) and 3 in both layers; one prefill row per round.
            for layer in 0..2 {
                counts[layer * 4 + round % 4] += rows;
                counts[layer * 4 + 3] += rows;
                counts[shape.plane_words() + layer * 4] += 2;
            }
            if windows.note(steps, active.iter().copied()) {
                closed.push(windows.close(&counts, steps, start + std::time::Duration::from_secs(steps)));
            }
        }
        let spans: Vec<_> = closed.iter().map(|w| (w.index, w.steps, w.tokens, w.requests.clone(), w.finished.clone()))
            .collect();
        assert_eq!(spans, vec![(0, [0, 4], 4, vec![7], vec![]), (1, [4, 6], 2, vec![7], vec![7]),
            (2, [6, 13], 7, vec![8, 9], vec![8])]);
        // Window 1 is round 3 alone: its two rows both route to expert 3, which is also round % 4.
        assert_eq!((closed[1].layers[0].clone(), closed[1].prefill_tokens), (vec![0, 0, 0, 4], 1));
        assert_eq!((closed[0].started.as_str(), closed[1].started.as_str()), ("2026-09-21T14:13:20Z", "2026-09-21T14:13:24Z"));
        // Nothing reported since the last close; the next routes are.
        assert!(!windows.counted(&counts));
        counts[1] += 1;
        assert!(windows.counted(&counts));
        let tail = windows.close(&counts, 13, start);
        closed.push(tail);
        let total = assemble(&counts, shape, 13, "m".into(), start, start);
        for layer in 0..2 {
            for phase in 0..2 {
                let summed: Vec<u64> = (0..4).map(|expert| closed.iter().map(|w|
                    [&w.layers, &w.prefill_layers][phase][layer][expert]).sum()).collect();
                assert_eq!(&summed, &[&total.layers, &total.prefill_layers][phase][layer], "layer {layer} phase {phase}");
            }
        }
    }

    #[test]
    fn phases_fill_their_own_planes_and_gpus_sum() {
        let words = V41.plane_words();
        // Encoder layers on GPU0, decoder layers on GPU1 (the dual RTX split):
        // each layer's routes are on one GPU, and the file sums them.
        let mut gpus = [vec![0u64; 2 * words], vec![0u64; 2 * words]];
        for layer in 0..40 {
            let gpu = &mut gpus[usize::from(layer >= 20)];
            // Decode: 5 rows of 6 routes per layer; prefill: 100 rows in the
            // encoder layers, the 3-row decoder replay in the others.
            gpu[layer * 384 + layer] = 18;
            gpu[layer * 384 + 200] = 12;
            gpu[words + layer * 384 + 7] = if layer < 20 { 600 } else { 18 };
        }
        let counts: Vec<u64> = gpus[0].iter().zip(&gpus[1]).map(|(a, b)| a + b).collect();
        let at = |seconds| std::time::UNIX_EPOCH + std::time::Duration::from_secs(seconds);
        let census = assemble(&counts, V41, 5, "org/model".into(), at(1_790_000_000), at(1_790_000_060));
        assert_eq!((census.steps, census.tokens, census.prefill_tokens), (5, 5, 100));
        assert_eq!((census.started.as_str(), census.time.as_str()), ("2026-09-21T14:13:20Z", "2026-09-21T14:14:20Z"));
        assert_eq!((census.layers.len(), census.prefill_layers.len()), (40, 40));
        for layer in 0..40 {
            let decode = &census.layers[layer];
            assert_eq!((decode[layer], decode[200], decode.iter().sum::<u64>()), (18, 12, 30), "layer {layer}");
            let prefill = &census.prefill_layers[layer];
            assert_eq!((prefill[7], prefill.iter().sum::<u64>()), if layer < 20 { (600, 600) } else { (18, 18) });
        }
    }

    #[test]
    fn a_census_written_from_counts_is_a_valid_tier_profile() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("route-census.json");
        let settings = settings(Some(&path.display().to_string()), None).unwrap().unwrap();
        probe_writable(&settings.path).unwrap();
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0, "the probe leaves nothing");
        assert!(probe_writable(&directory.path().join("absent").join("c.json")).is_err());
        let mut counts = vec![0u64; V41.bytes() / 8];
        for (index, count) in counts.iter_mut().enumerate() { *count = (index % 7) as u64; }
        let census = assemble(&counts, V41, 2, "org/model".into(), SystemTime::now(), SystemTime::now());
        write_logged(&census, &settings.path);
        let profile = ExpertRouteProfile::read(&settings.path).unwrap();
        profile.validate(40, 384).unwrap();
        assert_eq!(profile.layers, census.layers);
        assert_ne!(census.layers, census.prefill_layers);
        // A census that counted nothing does not replace it.
        let empty = assemble(&vec![0; V41.bytes() / 8], V41, 0, "org/model".into(), SystemTime::now(), SystemTime::now());
        write_logged(&empty, &settings.path);
        assert_eq!(ExpertRouteProfile::read(&settings.path).unwrap().layers, census.layers);
    }

    /// The device path: two router streams (a decode lane's and the prefill
    /// slots') with their own selectors, interleaving phases; startup counts
    /// dropped by `begin`; the periodic and final files hold exactly the CPU
    /// histogram of what each stream counted in its phase.
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA; route census device path"]
    fn streams_count_in_their_pass_phase_and_the_file_holds_the_sum() -> Result<()> {
        use crate::shared::memory::DeviceAllocation;
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("route-census.json");
        let census = RouteCensus::new(&library, &[0], V41, "org/model".into(),
            Settings { path: path.clone(), steps: 2, window: 1 })?;
        let counters = [census.counter(0, 0)?, census.counter(0, 39)?];
        let selectors = [counters[0].selector()?, counters[1].selector()?];
        assert_ne!(selectors[0].slot, selectors[1].slot);
        let streams = [library.cuda_stream_create()?, library.cuda_stream_create()?];
        let rows = [6usize, 2752];
        let ids: Vec<Vec<u32>> = rows.iter().enumerate().map(|(seed, &rows)|
            (0..rows * 6).map(|i| ((i * 7919 + seed * 104_729) % 384) as u32).collect()).collect();
        let buffers = ids.iter().map(|ids| -> Result<_> {
            let buffer = DeviceAllocation::new(&library, ids.len() * 4)?;
            library.copy_h2d(buffer.buffer, &ids.iter().flat_map(|id| id.to_ne_bytes()).collect::<Vec<_>>())?;
            Ok(buffer)
        }).collect::<Result<Vec<_>>>()?;
        let run = |phases: [RoutePhase; 2]| -> Result<()> {
            for stream in 0..2 {
                unsafe { selectors[stream].select(phases[stream], streams[stream])?; }
                for counter in &counters {
                    unsafe { counter.count(buffers[stream].buffer, rows[stream] as u32, &selectors[stream], streams[stream])?; }
                    counter.observe(phases[stream]);
                }
            }
            Ok(())
        };
        // Startup traffic before serving begins is not counted.
        run([RoutePhase::Decode, RoutePhase::Prefill])?;
        for &stream in &streams { unsafe { library.cuda_stream_synchronize(stream)?; } }
        census.begin()?;
        let mut expected = vec![0u64; V41.bytes() / 8];
        let mut decode_steps = 0;
        for phases in [[RoutePhase::Decode, RoutePhase::Prefill], [RoutePhase::Decode, RoutePhase::Prefill],
            [RoutePhase::Prefill, RoutePhase::Decode], [RoutePhase::Decode, RoutePhase::Decode]] {
            run(phases)?;
            for (stream, phase) in phases.into_iter().enumerate() {
                let histogram = cuteafd_ffi::route_histogram(&ids[stream], rows[stream], 6, None, 384);
                for layer in [0, 39] {
                    let row = &mut expected[(phase.plane() * 40 + layer) * 384..][..384];
                    for (total, count) in row.iter_mut().zip(&histogram) { *total += count; }
                }
                decode_steps += u64::from(phase == RoutePhase::Decode);
            }
            for &stream in &streams { unsafe { library.cuda_stream_synchronize(stream)?; } }
            // The serve loop's drained boundary: due every two decode steps.
            census.tick([(11, false)]);
        }
        census.finish();
        drop(census);
        let written: ExpertRouteCensus = serde_json::from_slice(&std::fs::read(&path)?)?;
        let plane = |plane: usize| expected[plane * V41.plane_words()..][..V41.plane_words()]
            .as_chunks::<384>().0.iter().map(|row| row.to_vec()).collect::<Vec<_>>();
        assert_eq!((written.layers.clone(), written.prefill_layers.clone()), (plane(0), plane(1)));
        assert_eq!((written.steps, written.model.as_str()), (decode_steps, "org/model"));
        // Decode: the 6-row stream three times, the 2752-row one twice; prefill the other way.
        assert_eq!((written.tokens, written.prefill_tokens), (3 * 6 + 2 * 2752, 6 + 2 * 2752));
        ExpertRouteProfile::read(&path)?.validate(40, 384)?;
        // The trace's windows, one per round with decode steps, sum to the file.
        let trace = cuteafd_loader::placement::RouteTrace::read(&path.with_extension("windows.jsonl"))?;
        let [(session, windows)] = trace.sessions.as_slice() else { anyhow::bail!("one session expected") };
        ensure!(session.window == 1 && windows.len() == 4 && windows.iter().all(|w| w.requests == [11]),
            "trace windows {:?}", windows.iter().map(|w| (w.steps, w.requests.clone())).collect::<Vec<_>>());
        for (phase, planes) in [(0, &written.layers), (1, &written.prefill_layers)] {
            for (layer, row) in planes.iter().enumerate() {
                let summed: Vec<u64> = (0..384).map(|e| windows.iter().map(|w|
                    [&w.layers, &w.prefill_layers][phase][layer][e]).sum()).collect();
                ensure!(&summed == row, "trace phase {phase} layer {layer} differs from the file");
            }
        }
        // A released selector byte is reused by the next stream.
        let slot = selectors[1].slot;
        let [first, second] = selectors;
        drop(second);
        assert_eq!(counters[0].selector()?.slot, slot);
        drop(first);
        for stream in streams { unsafe { library.cuda_stream_destroy(stream)?; } }
        Ok(())
    }
}
