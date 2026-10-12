//! Round timing: per-layer device time and the host round/draft boundaries.
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;
use std::time::Instant;

/// Per-layer device time of one pass from CUDA timing events.
///
/// One event per layer plus an entry event, allocated once per lane or rank.
/// The engine records `mark(layer)` on its stream at each layer's boundary
/// (V4.1: after the layer's FFN finish, outside the stage graphs; a generic
/// engine: at its `console::layer_mark` site, between per-layer graph
/// segments). Recording between graph replays does not touch capture; where
/// one graph spans several layers the engine records inside the capture, and
/// these events must then outlive the graph. Nothing waits on the events until
/// [`read`](Self::read) after the round's existing completion sync.
///
/// A layer's time runs from the previous layer's mark to its own; layer 0 has
/// no predecessor (its time is in the round residual). When a pass hands its
/// residual to another device, the first layer there is timed from
/// [`mark_entry`](Self::mark_entry) instead, which omits only the hop.
pub(crate) struct LayerClock<'a> {
    library: &'a NativeLibrary,
    events: Vec<*mut c_void>,
    recorded: Vec<bool>,
    /// Recorded when a layer's input arrives from another device.
    entry: *mut c_void,
    entry_layer: Option<usize>,
}

impl<'a> LayerClock<'a> {
    /// Timing events for `layers` layers on the current device.
    pub fn new(library: &'a NativeLibrary, layers: usize) -> Result<Self> {
        let mut events = Vec::with_capacity(layers + 1);
        for _ in 0..=layers {
            match library.cuda_event_create() {
                Ok(event) => events.push(event),
                Err(error) => {
                    for &event in &events {
                        // SAFETY: each event was created above and is destroyed once.
                        let _ = unsafe { library.cuda_event_destroy(event) };
                    }
                    return Err(error);
                }
            }
        }
        let entry = events.pop().expect("entry event");
        Ok(Self { library, events, recorded: vec![false; layers], entry, entry_layer: None })
    }
    pub fn layers(&self) -> usize { self.events.len() }
    /// Forget the previous pass's marks.
    pub fn reset(&mut self) {
        self.recorded.fill(false);
        self.entry_layer = None;
    }
    /// Mark the end of `layer` on `stream`. `drain` waits for the event at once
    /// (a stream the engine already drained and must find idle). Returns
    /// whether the mark was recorded.
    ///
    /// # Safety
    /// `stream` is a live stream on this clock's device, and no read of this
    /// layer's previous mark is in flight.
    pub unsafe fn mark(&mut self, layer: usize, stream: *mut c_void, drain: bool) -> bool {
        let Some(&event) = self.events.get(layer) else { return false };
        // SAFETY: the caller guarantees the stream; the event belongs to this clock.
        if !unsafe { self.record(event, stream, drain) } { return false; }
        self.recorded[layer] = true;
        true
    }
    /// Mark the arrival of `layer`'s input from another device.
    ///
    /// # Safety
    /// As [`mark`](Self::mark).
    pub unsafe fn mark_entry(&mut self, layer: usize, stream: *mut c_void, drain: bool) -> bool {
        // SAFETY: the caller guarantees the stream; the event belongs to this clock.
        if !unsafe { self.record(self.entry, stream, drain) } { return false; }
        self.entry_layer = Some(layer);
        true
    }
    unsafe fn record(&self, event: *mut c_void, stream: *mut c_void, drain: bool) -> bool {
        // SAFETY: forwarded from `mark`/`mark_entry`.
        if unsafe { self.library.cuda_event_record(event, stream) }.is_err() { return false; }
        if drain {
            // SAFETY: the event was just recorded on a live stream.
            if let Err(error) = unsafe { self.library.cuda_event_synchronize(event) } {
                tracing::warn!(%error, "layer timing event wait failed");
            }
        }
        true
    }
    /// Device µs from `from`'s mark to `to`'s, once the pass completed.
    pub fn between(&self, from: usize, to: usize) -> Option<f64> {
        if !(*self.recorded.get(from)? && *self.recorded.get(to)?) { return None; }
        // SAFETY: both events were recorded in this pass and its work completed.
        unsafe { self.library.cuda_event_elapsed_ms(self.events[from], self.events[to]) }
            .ok().map(|ms| f64::from(ms) * 1e3)
    }
    /// Device µs from `layer`'s entry mark to its own mark.
    pub fn since_entry(&self, layer: usize) -> Option<f64> {
        if self.entry_layer != Some(layer) || !*self.recorded.get(layer)? { return None; }
        // SAFETY: both events were recorded in this pass and its work completed.
        unsafe { self.library.cuda_event_elapsed_ms(self.entry, self.events[layer]) }
            .ok().map(|ms| f64::from(ms) * 1e3)
    }
    /// Device µs from this layer's owner entry, or its previous layer's mark.
    pub fn layer_us(&self, layer: usize) -> Option<f64> {
        match layer_start(&self.recorded, self.entry_layer, layer)? {
            LayerStart::Entry => self.since_entry(layer),
            LayerStart::Previous(previous) => self.between(previous, layer),
        }
    }
    /// Every recorded layer's device µs; an owner cut uses its entry event.
    pub fn read(&self) -> Vec<Option<f64>> {
        (0..self.layers()).map(|layer| self.layer_us(layer)).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerStart {
    Entry,
    Previous(usize),
}

fn layer_start(recorded: &[bool], entry_layer: Option<usize>, layer: usize) -> Option<LayerStart> {
    if !*recorded.get(layer)? { return None; }
    if entry_layer == Some(layer) { return Some(LayerStart::Entry); }
    let previous = layer.checked_sub(1)?;
    recorded[previous].then_some(LayerStart::Previous(previous))
}

impl Drop for LayerClock<'_> {
    fn drop(&mut self) {
        for &event in self.events.iter().chain([&self.entry]) {
            // SAFETY: every event was created by this clock and is destroyed once.
            if let Err(error) = unsafe { self.library.cuda_event_destroy(event) } {
                tracing::error!(%error, "destroying layer timing event");
            }
        }
    }
}

/// Aggregate host-only probe/policy work without per-round logging or switches.
#[derive(Clone, Copy)]
pub(crate) enum HostStage { Routes, Observe, Select, Publish }

pub(crate) struct HostSpan { stage: HostStage, start: Instant }

impl HostSpan {
    pub(crate) fn new(stage: HostStage) -> Self { Self { stage, start: Instant::now() } }
}

impl Drop for HostSpan {
    fn drop(&mut self) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SAMPLES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
        static MICROS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
        let index = self.stage as usize;
        let micros = self.start.elapsed().as_micros() as u64;
        let total = MICROS[index].fetch_add(micros, Ordering::Relaxed) + micros;
        let samples = SAMPLES[index].fetch_add(1, Ordering::Relaxed) + 1;
        if samples % 64 == 0 {
            let stage = match self.stage { HostStage::Routes => "routes", HostStage::Observe => "observe", HostStage::Select => "select", HostStage::Publish => "publish" };
            tracing::info!(stage, samples, mean_us = total as f64 / samples as f64, "draft policy host timing");
        }
    }
}

/// Host boundaries of one lane round: `total_us` runs from the round's start
/// to its observation, and `draft_us` brackets the draft call alone. Sequence
/// collection, copy lookup and length selection are round work, not draft
/// work. Verify time is not a separate signal: it is the layer sum plus the
/// round residual.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoundClock {
    round_start: Instant,
    draft_us: Option<u64>,
}

impl RoundClock {
    /// A round that started at `round_start`.
    pub fn at(round_start: Instant) -> Self { Self { round_start, draft_us: None } }
    /// Bracket a synchronous draft call.
    pub fn draft<T>(&mut self, call: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let value = call();
        self.draft_us = Some(start.elapsed().as_micros() as u64);
        value
    }
    /// A polled drafter that times itself from issue to completion.
    pub fn drafted_us(&mut self, us: u64) { self.draft_us = Some(us); }
    /// The round's boundaries, observed now.
    pub fn observe(&self) -> RoundTimes {
        RoundTimes { total_us: self.round_start.elapsed().as_micros() as u64, draft_us: self.draft_us }
    }
}

/// One round's observed host times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RoundTimes {
    pub total_us: u64,
    pub draft_us: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_cut_uses_entry_then_same_owner_boundaries() {
        let peer = [false, false, true, true];
        assert_eq!(layer_start(&peer, Some(2), 2), Some(LayerStart::Entry));
        assert_eq!(layer_start(&peer, Some(2), 3), Some(LayerStart::Previous(2)));
        assert_eq!(layer_start(&peer, None, 2), None);
        assert_eq!(layer_start(&peer, Some(2), 1), None);
        assert_eq!(layer_start(&peer, Some(2), 4), None);
        // An explicit arrival takes precedence even if an old-owner mark exists.
        assert_eq!(layer_start(&[true; 4], Some(2), 2), Some(LayerStart::Entry));
        assert_eq!(layer_start(&[false; 4], Some(2), 2), None);
    }

    #[test]
    fn single_owner_clock_boundaries_are_unchanged() {
        let recorded = [true; 4];
        assert_eq!(layer_start(&recorded, None, 0), None);
        for layer in 1..4 {
            assert_eq!(layer_start(&recorded, None, layer), Some(LayerStart::Previous(layer - 1)));
        }
        assert_eq!(layer_start(&[true, false, true], None, 2), None);
    }

    #[test]
    fn round_clock_brackets_the_draft_call_alone() {
        let mut clock = RoundClock::at(Instant::now());
        assert_eq!(clock.observe().draft_us, None);
        std::thread::sleep(std::time::Duration::from_millis(3));
        let value = clock.draft(|| { std::thread::sleep(std::time::Duration::from_millis(2)); 7 });
        assert_eq!(value, 7);
        std::thread::sleep(std::time::Duration::from_millis(3));
        let times = clock.observe();
        let draft = times.draft_us.unwrap();
        assert!((2_000..times.total_us).contains(&draft), "{times:?}");
        assert!(times.total_us >= draft + 6_000, "round work outside the draft call: {times:?}");
        let mut polled = RoundClock::at(Instant::now());
        polled.drafted_us(1_234);
        assert_eq!(polled.observe().draft_us, Some(1_234));
    }
}
