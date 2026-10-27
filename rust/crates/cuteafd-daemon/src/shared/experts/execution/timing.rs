//! Optional, reusable GPU events for the native expert boundary.
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;

pub(crate) struct Event<'a> {
    library: &'a NativeLibrary,
    raw: *mut c_void,
}
impl<'a> Event<'a> {
    fn new(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self {
            library,
            raw: library.cuda_event_create()?,
        })
    }

    fn ordering(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self { library, raw: library.cuda_event_create_ordering()? })
    }
}
impl Drop for Event<'_> {
    fn drop(&mut self) {
        if let Err(error) = unsafe { self.library.cuda_event_destroy(self.raw) } {
            tracing::error!(%error, "destroying expert timing event");
        }
    }
}

/// The owner stream's last completed unit end, shared by every adapter that
/// queues on that stream (FP8 weight groups), so the idle gap follows the
/// stream rather than one group's queue.
pub(crate) type PaceClock<'a> = std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<Event<'a>>>>>;

/// One unit's measured pace on its owner stream (#762): busy from its
/// arrival (before its input copies) to its fence, idle from the previous
/// unit's fence on the same stream to its arrival (zero for the stream's
/// first unit, or when it was queued behind its predecessor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UnitPace { pub busy_us: u32, pub idle_us: u32 }

/// Prefill units: per staging slot an arrival and a fence. A fence is shared
/// with the pace clock, which retains the newest completed one; a slot whose
/// fence the clock still holds takes the spare before it is recorded again.
struct UnitEvents<'a> {
    events: Vec<(Event<'a>, std::rc::Rc<Event<'a>>)>,
    /// Always exactly one spare fence.
    spare: std::rc::Rc<Event<'a>>,
    /// Slot whose arrival was recorded and whose fence is not yet queued.
    arrived: Option<usize>,
    clock: PaceClock<'a>,
}

enum Completion<'a> {
    SingleFlight(*mut c_void),
    UnitEvents(UnitEvents<'a>),
}

/// Decode has one retained unit, so stream idleness is its completion fence
/// (decode owners report no pace). Prefill retains independent unit fences so
/// an earlier reply need not wait for later work, and records each unit's
/// arrival before its input copies, so its pace is read from the completed
/// events after its fence (no added wait). All events are allocated before
/// serving; hot-path submission never allocates.
pub(crate) struct PendingUnits<'a> {
    library: &'a NativeLibrary,
    completion: Completion<'a>,
    free: std::collections::VecDeque<usize>,
    queued: std::collections::VecDeque<(usize, (u64, u32, u32, u64))>,
}
impl<'a> PendingUnits<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, capacity: usize) -> Result<Self> {
        Self::for_owner(library, capacity, cuteafd_core::ExecutionOwner::Prefill)
    }

    pub(crate) fn for_owner(library: &'a NativeLibrary, capacity: usize,
        owner: cuteafd_core::ExecutionOwner) -> Result<Self> {
        Self::with_clock(library, capacity, owner, Default::default())
    }

    /// A queue whose prefill idle gaps follow `clock` (one per owner stream).
    pub(crate) fn with_clock(library: &'a NativeLibrary, capacity: usize, owner: cuteafd_core::ExecutionOwner,
        clock: PaceClock<'a>) -> Result<Self> {
        anyhow::ensure!((1..=4).contains(&capacity), "invalid expert unit queue capacity");
        let completion = match owner {
            cuteafd_core::ExecutionOwner::Decode(_) => {
                anyhow::ensure!(capacity == 1, "decode completion requires a single retained unit");
                Completion::SingleFlight(std::ptr::null_mut())
            },
            cuteafd_core::ExecutionOwner::Prefill => Completion::UnitEvents(UnitEvents {
                events: (0..capacity).map(|_| Ok((Event::new(library)?, std::rc::Rc::new(Event::new(library)?))))
                    .collect::<Result<_>>()?,
                spare: std::rc::Rc::new(Event::new(library)?), arrived: None, clock }),
        };
        Ok(Self { library, completion,
            free: (0..capacity).collect(), queued: std::collections::VecDeque::with_capacity(capacity) })
    }
    /// The staging slot the next unit uses.
    pub(crate) fn next_slot(&self) -> Result<usize> {
        self.free.front().copied().ok_or_else(|| anyhow::anyhow!("expert unit queue exhausted"))
    }
    pub(crate) fn is_empty(&self) -> bool { self.queued.is_empty() }
    pub(crate) fn len(&self) -> usize { self.queued.len() }
    /// Marks the next prefill unit's arrival on `stream`, before its input
    /// copies (decode: nothing).
    /// # Safety
    /// `stream` is the owner stream the unit's fence is later recorded on.
    pub(crate) unsafe fn arrive(&mut self, stream: *mut c_void) -> Result<()> {
        let slot = self.next_slot()?;
        if let Completion::UnitEvents(units) = &mut self.completion {
            // SAFETY: this slot's events are exclusively reserved until the unit is popped.
            unsafe { self.library.cuda_event_record(units.events[slot].0.raw, stream)?; }
            units.arrived = Some(slot);
        }
        Ok(())
    }
    /// # Safety
    /// The stream and every borrowed request/reply owner remain live until pop.
    pub(crate) unsafe fn record(&mut self, identity: (u64, u32, u32, u64), stream: *mut c_void) -> Result<()> {
        let slot = self.next_slot()?;
        match &mut self.completion {
            Completion::SingleFlight(owner_stream) => *owner_stream = stream,
            Completion::UnitEvents(units) => {
                anyhow::ensure!(units.arrived.take() == Some(slot), "expert unit fence without its arrival");
                // The clock may still hold this slot's previous fence (it is the
                // stream's newest completed end): give the slot the spare instead.
                if std::rc::Rc::strong_count(&units.events[slot].1) > 1 {
                    std::mem::swap(&mut units.events[slot].1, &mut units.spare);
                }
                // SAFETY: this event is exclusively reserved until the unit is popped.
                unsafe { self.library.cuda_event_record(units.events[slot].1.raw, stream)?; }
            },
        }
        self.free.pop_front();
        self.queued.push_back((slot, identity));
        Ok(())
    }
    pub(crate) fn ready(&self) -> Result<bool> {
        let Some((slot, _)) = self.queued.front() else { return Ok(true) };
        // SAFETY: the queued unit retains its owner stream or exclusive event.
        unsafe { match &self.completion {
            Completion::SingleFlight(stream) => self.library.cuda_stream_query(*stream),
            Completion::UnitEvents(units) => self.library.cuda_event_query(units.events[*slot].1.raw),
        } }
    }
    /// The oldest unit's identity and measured pace (decode: zero), once its fence completed.
    pub(crate) fn pop(&mut self) -> Result<Option<((u64, u32, u32, u64), UnitPace)>> {
        anyhow::ensure!(self.ready()?, "expert unit completed before its fence");
        let Some((slot, identity)) = self.queued.pop_front() else { return Ok(None) };
        let pace = match &self.completion {
            Completion::SingleFlight(_) => UnitPace { busy_us: 0, idle_us: 0 },
            Completion::UnitEvents(units) => {
                let (arrived, end) = &units.events[slot];
                let elapsed_us = |from: *mut c_void, to: *mut c_void| -> Result<u32> {
                    // SAFETY: both events were recorded on this owner stream and have
                    // completed: the fence is ready and every earlier record precedes it.
                    let ms = unsafe { self.library.cuda_event_elapsed_ms(from, to)? };
                    Ok((f64::from(ms).max(0.0) * 1000.0).min(f64::from(u32::MAX)) as u32)
                };
                let busy_us = elapsed_us(arrived.raw, end.raw)?;
                let idle_us = match units.clock.borrow().as_ref() {
                    Some(previous) => elapsed_us(previous.raw, arrived.raw)?,
                    None => 0,
                };
                *units.clock.borrow_mut() = Some(end.clone());
                UnitPace { busy_us, idle_us }
            }
        };
        self.free.push_back(slot);
        Ok(Some((identity, pace)))
    }
}

fn safe_host(host: &str) -> String {
    let name: String = host.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
        c
    } else { '_' }).collect();
    if name.is_empty() { "unknown".into() } else { name }
}

fn enabled(value: Option<&str>) -> bool { value == Some("1") }

#[derive(serde::Serialize)]
struct WorkerSample<'a> {
    event: &'static str,
    host: &'a str,
    rank: usize,
    request_id: u64,
    layer: u32,
    rows: u32,
    executor_id: u64,
    format: &'a str,
    completed_unix_ns: u64,
    compute_us: f32,
    previous_request_id: Option<u64>,
    previous_layer: Option<u32>,
    gpu_gap_us: Option<f32>,
}
impl WorkerSample<'_> {
    fn write_to(&self, output: &mut impl std::io::Write) -> Result<()> {
        serde_json::to_writer(&mut *output, self)?;
        output.write_all(b"\n")?;
        output.flush()?;
        Ok(())
    }
}

struct WorkerSpans<'a> {
    free: std::collections::VecDeque<usize>,
    recording: Option<usize>,
    queued: std::collections::VecDeque<usize>,
    previous_end: Option<std::rc::Rc<Event<'a>>>,
}

/// Startup-only diagnostics shared by every Spark weight format.
/// The measured pipeline includes device routing and output reduction, not host copies.
pub(crate) struct WorkerTiming<'a> {
    events: Vec<[std::rc::Rc<Event<'a>>; 2]>,
    state: std::cell::RefCell<WorkerSpans<'a>>,
    previous: Option<(u64, u32)>,
    rank: usize,
    host: String,
    output: Option<std::io::BufWriter<std::fs::File>>,
}
impl<'a> WorkerTiming<'a> {
    pub(crate) fn from_env(library: &'a NativeLibrary, rank: usize) -> Result<Option<Self>> {
        if !enabled(std::env::var("CUTEAFD_EXPERT_TIMING").ok().as_deref()) {
            return Ok(None);
        }
        let host = std::env::var("CUTEAFD_PROFILE_HOST").or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "unknown".into());
        let host = safe_host(&host);
        let output = std::env::var_os("CUTEAFD_PROFILE_DIR").map(|directory| {
            let path = std::path::Path::new(&directory).join(format!("expert-timing-{host}-rank{rank}.jsonl"));
            std::fs::OpenOptions::new().create(true).append(true).open(path).map(std::io::BufWriter::new)
        }).transpose()?;
        Ok(Some(Self::new(library, rank, host, output)?))
    }

    fn new(library: &'a NativeLibrary, rank: usize, host: String,
        output: Option<std::io::BufWriter<std::fs::File>>) -> Result<Self> {
        // One extra span retains the preceding end while up to four units queue.
        let count = usize::from(cuteafd_transport::worker_selection::MAX_BUFFERED_PREFILL_UNITS) + 1;
        let events = (0..count).map(|_| Ok([std::rc::Rc::new(Event::new(library)?),
            std::rc::Rc::new(Event::new(library)?)] )).collect::<Result<_>>()?;
        Ok(Self { events, state: std::cell::RefCell::new(WorkerSpans {
            free: (0..count).collect(), recording: None, queued: Default::default(), previous_end: None }),
            previous: None, rank, host, output })
    }

    /// The worker retains its stream and inputs until the existing completion barrier.
    pub(crate) unsafe fn record(&self, index: usize, stream: *mut c_void) -> Result<()> {
        let mut state = self.state.borrow_mut();
        let slot = if index == 0 {
            anyhow::ensure!(state.recording.is_none(), "expert timing span already recording");
            let slot = state.free.pop_front().ok_or_else(|| anyhow::anyhow!("expert timing queue exhausted"))?;
            state.recording = Some(slot);
            slot
        } else {
            state.recording.ok_or_else(|| anyhow::anyhow!("expert timing end without start"))?
        };
        let event = &self.events[slot][usize::from(index != 0)];
        // SAFETY: this event pair remains reserved through this unit's publication.
        unsafe { event.library.cuda_event_record(event.raw, stream)?; }
        if index != 0 { state.recording = None; state.queued.push_back(slot); }
        Ok(())
    }

    /// Called after this unit's completion fence; never adds a GPU wait.
    pub(crate) unsafe fn publish(&mut self, request_id: u64, layer: u32, rows: u32, executor_id: u64,
        format: &str) -> Result<()> {
        let mut state = self.state.borrow_mut();
        let slot = *state.queued.front().ok_or_else(|| anyhow::anyhow!("no completed expert timing span"))?;
        let [start, end] = &self.events[slot];
        // SAFETY: the caller completed this unit's event, not necessarily its stream.
        let compute_us = unsafe { start.library.cuda_event_elapsed_ms(start.raw, end.raw)? } * 1000.0;
        let gpu_gap_us = state.previous_end.as_ref().map(|previous_end| {
            // SAFETY: both retained records completed before this publication.
            unsafe { start.library.cuda_event_elapsed_ms(previous_end.raw, start.raw) }
                .map(|ms| ms * 1000.0)
        }).transpose()?;
        let unix_ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos().min(u64::MAX as u128) as u64;
        if let Some(output) = &mut self.output {
            WorkerSample { event: "expert_compute", host: &self.host, rank: self.rank,
                request_id, layer, rows, executor_id, format, completed_unix_ns: unix_ns,
                compute_us, previous_request_id: self.previous.map(|p| p.0),
                previous_layer: self.previous.map(|p| p.1), gpu_gap_us }.write_to(output)?;
        } else {
            tracing::debug!(target: "cuteafd::expert_timing", host=%self.host, rank=self.rank,
                request_id, layer, rows, executor_id, format, compute_us, completed_unix_ns=unix_ns,
                previous_request_id=?self.previous.map(|p| p.0), previous_layer=?self.previous.map(|p| p.1), ?gpu_gap_us,
                "Spark expert compute");
        }
        self.previous = Some((request_id, layer));
        state.previous_end = Some(end.clone());
        state.queued.pop_front();
        state.free.push_back(slot);
        Ok(())
    }
}

#[cfg(test)]
mod worker_tests {
    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn decode_stream_completion_preserves_credit_identity_and_timing_without_unit_events() -> anyhow::Result<()> {
        use super::*;
        use cuteafd_core::ExecutionOwner;
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        library.cuda_set_device(7)?;
        fixture.configure_memory(&library, 7, 1 << 20)?;
        fixture.queued_experts(&library)?;
        let stream = crate::shared::memory::LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let allocation = crate::shared::memory::DeviceAllocation::new(&library, 1)?;
        let input = [13u8; 1];
        let mut pending = PendingUnits::for_owner(&library, 1, ExecutionOwner::Decode(1))?;
        assert!(matches!(&pending.completion, Completion::SingleFlight(_)));
        assert!(PendingUnits::for_owner(&library, 2, ExecutionOwner::Decode(0)).is_err());
        let prefill = PendingUnits::for_owner(&library, 1, ExecutionOwner::Prefill)?;
        assert!(matches!(&prefill.completion, Completion::UnitEvents(units) if units.events.len() == 1));
        let path = fixture.directory().join("decode-timing.jsonl");
        let output = std::io::BufWriter::new(std::fs::File::create(&path)?);
        let mut timing = WorkerTiming::new(&library, 0, "fixture".into(), Some(output))?;
        for request_id in 0..3 {
            assert!(pending.ready()? && pending.is_empty());
            // SAFETY: the source, buffers, stream and timing records outlive completion.
            unsafe {
                timing.record(0, stream.raw)?;
                library.copy_h2d_async(allocation.buffer, &input, stream.raw)?;
                timing.record(1, stream.raw)?;
                pending.record((request_id, 20, 24, 27), stream.raw)?;
            }
            assert_eq!(pending.len(), 1);
            assert!(pending.next_slot().is_err() && !pending.ready()?);
            assert!(pending.pop().is_err());
            fixture.advance_queued_experts(&library, 1)?;
            assert!(pending.ready()?);
            let (identity, pace) = pending.pop()?.unwrap();
            assert_eq!(identity, (request_id, 20, 24, 27));
            // Decode owners report no pace (#762: prefill owner only).
            assert_eq!(pace, UnitPace { busy_us: 0, idle_us: 0 });
            assert_eq!(pending.next_slot()?, 0);
            // SAFETY: the exclusive owner stream is idle after both timing records.
            unsafe { timing.publish(identity.0, identity.1, identity.2, identity.3, "native")?; }
        }
        let records = std::fs::read_to_string(path)?.lines().map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(records.len(), 3);
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record["request_id"], index as u64);
            assert!((record["compute_us"].as_f64().unwrap() - 1.0).abs() < 0.001);
        }
        Ok(())
    }

    #[test]
    fn typed_owner_selects_completion_for_every_spark_backend() {
        let local = include_str!("../service/local.rs");
        assert!(local.contains("extent.staging_rows as usize, extent.owner)?"));
        for source in [include_str!("../execution.rs"), include_str!("../exl3/worker.rs"),
            include_str!("../service/exlr8.rs")] {
            assert!(source.contains("PendingUnits::for_owner(self.library, units, owner)?"));
        }
        // FP8 weight groups on one owner stream share its pace clock.
        assert!(include_str!("../fp8/worker.rs")
            .contains("PendingUnits::with_clock(self.library, units, owner,\n            self.clock.clone())?"));
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn queued_spans_keep_fifo_identities_and_gpu_gaps_across_event_reuse() -> anyhow::Result<()> {
        use super::*;
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        library.cuda_set_device(7)?;
        fixture.configure_memory(&library, 7, 1 << 20)?;
        fixture.queued_experts(&library)?;
        let path = fixture.directory().join("timing.jsonl");
        let output = std::io::BufWriter::new(std::fs::File::create(&path)?);
        let mut timing = WorkerTiming::new(&library, 0, "fixture".into(), Some(output))?;
        let input = [13u8; 1];
        let allocation = crate::shared::memory::DeviceAllocation::new(&library, 1)?;
        let stream = crate::shared::memory::LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let mut pending = PendingUnits::new(&library, 4)?;
        for round in 0..3 {
            for unit in 0..4 {
                // SAFETY: the source and destination outlive this stream drain.
                unsafe {
                    pending.arrive(stream.raw)?;
                    library.copy_h2d_async(allocation.buffer, &input, stream.raw)?;
                    timing.record(0, stream.raw)?;
                    library.copy_h2d_async(allocation.buffer, &input, stream.raw)?;
                    timing.record(1, stream.raw)?;
                    pending.record((round * 4 + unit, 3, 1, 0), stream.raw)?;
                }
            }
            assert_eq!(pending.len(), 4);
            assert!(pending.next_slot().is_err());
            for unit in 0..4 {
                assert!(!pending.ready()?);
                fixture.advance_queued_experts(&library, 2)?;
                let (identity, pace) = pending.pop()?.unwrap();
                assert_eq!(identity.0, round * 4 + unit);
                // Two queued copies per unit, back to back on the stream.
                assert!(pace.busy_us > 0 && pace.idle_us <= pace.busy_us, "{pace:?}");
                // SAFETY: the unit's fence is ready while later spans may be queued.
                unsafe { timing.publish(identity.0, identity.1, identity.2, identity.3, "fp8_moe")?; }
            }
        }
        let rows = std::fs::read_to_string(path)?.lines().map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(rows.len(), 12);
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(row["request_id"], index as u64);
            assert!((row["compute_us"].as_f64().unwrap() - 1.0).abs() < 0.001);
            if index == 0 { assert!(row["gpu_gap_us"].is_null()); }
            else {
                assert_eq!(row["previous_request_id"], index as u64 - 1);
                assert!((row["gpu_gap_us"].as_f64().unwrap() - 1.0).abs() < 0.001);
            }
        }
        Ok(())
    }

    #[test]
    fn worker_spans_emit_the_same_jsonl_contract_for_exlr8_and_existing_formats() -> anyhow::Result<()> {
        let mut output = Vec::new();
        for format in ["native", "exl3", "fp8_moe", "exlr8"] {
            super::WorkerSample { event: "expert_compute", host: "worker", rank: 2,
                request_id: 17, layer: 3, rows: 65, executor_id: 23, format,
                completed_unix_ns: 29, compute_us: 31.5, previous_request_id: Some(16),
                previous_layer: Some(3), gpu_gap_us: Some(2.5) }.write_to(&mut output)?;
        }
        let text = String::from_utf8(output)?;
        assert!(text.ends_with('\n'));
        let records = text.lines().map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(records.len(), 4);
        for (record, format) in records.iter().zip(["native", "exl3", "fp8_moe", "exlr8"]) {
            assert_eq!(record, &serde_json::json!({"event": "expert_compute", "host": "worker",
                "rank": 2, "request_id": 17, "layer": 3, "rows": 65, "executor_id": 23,
                "format": format, "completed_unix_ns": 29, "compute_us": 31.5,
                "previous_request_id": 16, "previous_layer": 3, "gpu_gap_us": 2.5}));
        }
        Ok(())
    }

    #[test]
    fn timing_requires_explicit_one() {
        assert!(super::enabled(Some("1")));
        for value in [None, Some(""), Some("0"), Some("true")] {
            assert!(!super::enabled(value));
        }
    }

    #[test]
    fn host_cannot_escape_profile_directory() {
        assert_eq!(super::safe_host("ostrich"), "ostrich");
        assert_eq!(super::safe_host("../host/a"), "___host_a");
        assert_eq!(super::safe_host(""), "unknown");
    }
}

/// The wave owns these events and drains its stream before dropping them.
pub(super) struct ExpertTiming<'a> {
    events: [Event<'a>; 3],
}
impl<'a> ExpertTiming<'a> {
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self {
            events: [
                Event::new(library)?,
                Event::new(library)?,
                Event::new(library)?,
            ],
        })
    }
    pub unsafe fn record(&self, index: usize, stream: *mut c_void) -> Result<()> {
        let event = &self.events[index];
        unsafe { event.library.cuda_event_record(event.raw, stream) }
    }
    /// Call only after the owning stream's existing completion barrier.
    pub unsafe fn elapsed_us(&self) -> Result<(f32, f32)> {
        let [start, kernel_end, compact_end] = &self.events;
        Ok(unsafe {
            (
                1000.0
                    * start
                        .library
                        .cuda_event_elapsed_ms(start.raw, kernel_end.raw)?,
                1000.0
                    * start
                        .library
                        .cuda_event_elapsed_ms(kernel_end.raw, compact_end.raw)?,
            )
        })
    }
}
