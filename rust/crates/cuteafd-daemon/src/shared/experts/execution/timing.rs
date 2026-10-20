//! Optional, reusable GPU events for the native expert boundary.
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;

struct Event<'a> {
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
}
impl Drop for Event<'_> {
    fn drop(&mut self) {
        if let Err(error) = unsafe { self.library.cuda_event_destroy(self.raw) } {
            tracing::error!(%error, "destroying expert timing event");
        }
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
}
impl WorkerSample<'_> {
    fn write_to(&self, output: &mut impl std::io::Write) -> Result<()> {
        serde_json::to_writer(&mut *output, self)?;
        output.write_all(b"\n")?;
        output.flush()?;
        Ok(())
    }
}

/// Startup-only diagnostics shared by every Spark weight format.
/// The measured pipeline includes device routing and output reduction, not host copies.
pub(crate) struct WorkerTiming<'a> {
    events: [Event<'a>; 2],
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
        Ok(Some(Self { events: [Event::new(library)?, Event::new(library)?], rank, host, output }))
    }

    /// The worker retains its stream and inputs until the existing completion barrier.
    pub(crate) unsafe fn record(&self, index: usize, stream: *mut c_void) -> Result<()> {
        let event = &self.events[index];
        // SAFETY: the caller retains the event and stream through completion.
        unsafe { event.library.cuda_event_record(event.raw, stream) }
    }

    /// Called after the worker's existing stream completion, never adds a GPU wait.
    pub(crate) unsafe fn publish(&mut self, request_id: u64, layer: u32, rows: u32, executor_id: u64,
        format: &str) -> Result<()> {
        let [start, end] = &self.events;
        // SAFETY: the caller completed the stream containing both event records.
        let compute_us = unsafe { start.library.cuda_event_elapsed_ms(start.raw, end.raw)? } * 1000.0;
        let unix_ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos().min(u64::MAX as u128) as u64;
        if let Some(output) = &mut self.output {
            WorkerSample { event: "expert_compute", host: &self.host, rank: self.rank,
                request_id, layer, rows, executor_id, format, completed_unix_ns: unix_ns,
                compute_us }.write_to(output)?;
        } else {
            tracing::debug!(target: "cuteafd::expert_timing", host=%self.host, rank=self.rank,
                request_id, layer, rows, executor_id, format, compute_us, completed_unix_ns=unix_ns,
                "Spark expert compute");
        }
        Ok(())
    }
}

#[cfg(test)]
mod worker_tests {
    #[test]
    fn worker_spans_emit_the_same_jsonl_contract_for_exlr8_and_existing_formats() -> anyhow::Result<()> {
        let mut output = Vec::new();
        for format in ["native", "exl3", "fp8_moe", "exlr8"] {
            super::WorkerSample { event: "expert_compute", host: "worker", rank: 2,
                request_id: 17, layer: 3, rows: 65, executor_id: 23, format,
                completed_unix_ns: 29, compute_us: 31.5 }.write_to(&mut output)?;
        }
        let text = String::from_utf8(output)?;
        assert!(text.ends_with('\n'));
        let records = text.lines().map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(records.len(), 4);
        for (record, format) in records.iter().zip(["native", "exl3", "fp8_moe", "exlr8"]) {
            assert_eq!(record, &serde_json::json!({"event": "expert_compute", "host": "worker",
                "rank": 2, "request_id": 17, "layer": 3, "rows": 65, "executor_id": 23,
                "format": format, "completed_unix_ns": 29, "compute_us": 31.5}));
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
