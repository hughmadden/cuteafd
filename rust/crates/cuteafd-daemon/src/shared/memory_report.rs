//! Memory reports from the allocation ledger (`cuteafd_ffi::memory_ledger`):
//! per device, the bytes the runtime reports in use, the bytes this process
//! allocated by category, the untracked rest (CUDA context, modules, cuBLAS,
//! graph executables), weights by checkpoint tensor stem and resident format,
//! and any tensor resident in two formats. One JSON line per report under the
//! `cuteafd::memory` target; `scripts/bench/memory-audit.py` tabulates them.

use cuteafd_ffi::memory_ledger::{self, Space};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::time::Duration;

fn meminfo() -> Value {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else { return Value::Null };
    let mut out = Map::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let (Some(key), Some(value)) = (parts.next(), parts.next()) else { continue };
        let key = key.trim_end_matches(':');
        if matches!(key, "MemTotal" | "MemAvailable" | "Cached" | "AnonPages" | "Shmem" | "Mlocked") {
            if let Ok(kib) = value.parse::<u64>() {
                out.insert(key.to_owned(), json!(kib * 1024));
            }
        }
    }
    Value::Object(out)
}

/// Builds one report. Queries each device on the calling thread and restores
/// nothing: call it from a thread that does not launch work, or restore the
/// device afterwards (see [`log`]).
pub(crate) fn report(stage: &str) -> Value {
    let snapshot = memory_ledger::snapshot();
    let mut per_device = Vec::new();
    // Only devices this process allocated on: querying another would create a context there.
    for device in snapshot.devices() {
        let tracked = snapshot.total(Space::Device, device) + snapshot.total(Space::Managed, device);
        let scopes: BTreeMap<&str, usize> = snapshot.by_scope(Space::Device, device);
        let mut entry = json!({
            "device": device,
            "tracked": tracked,
            "peak": snapshot.peak.get(&(Space::Device, device)).copied().unwrap_or(0),
            "scopes": scopes,
        });
        if let Some((free, total)) = memory_ledger::device_memory(device) {
            let used = total - free;
            entry["used"] = json!(used);
            entry["total"] = json!(total);
            entry["untracked"] = json!(used as i64 - tracked as i64);
        }
        per_device.push(entry);
    }
    let weights: Vec<Value> = snapshot.rows.iter().filter(|r| r.key.tensor.is_some())
        .map(|r| json!([r.key.space.label(), r.key.device, r.key.scope, r.key.tensor, r.key.format, r.bytes,
            r.allocations]))
        .collect();
    let dual: Vec<Value> = snapshot.dual_formats().into_iter()
        .map(|(device, tensor, formats)| json!({"device": device, "tensor": tensor, "formats": formats}))
        .collect();
    json!({
        "stage": stage,
        "devices": per_device,
        "pinned": {
            "tracked": snapshot.total(Space::Pinned, -1),
            "peak": snapshot.peak.get(&(Space::Pinned, -1)).copied().unwrap_or(0),
            "scopes": snapshot.by_scope(Space::Pinned, -1),
        },
        "weights": weights,
        "dual_formats": dual,
        "host": meminfo(),
    })
}

/// Logs a report from a helper thread (the caller's device selection is untouched).
pub(crate) fn log(stage: &str) {
    let stage = stage.to_owned();
    let value = std::thread::spawn(move || report(&stage)).join();
    if let Ok(value) = value {
        tracing::info!(target: "cuteafd::memory", report = %value, "memory ledger");
    }
}

/// Logs a report now and again whenever device use or the ledger moves by at
/// least 64 MiB (first graph captures, lazily sized workspaces), checking every
/// `period`. Runs for the life of the process.
pub(crate) fn monitor(stage: &'static str, period: Duration) {
    if std::env::var_os("CUTEAFD_MEMORY_MONITOR").is_some_and(|v| v == "0") {
        return;
    }
    let _ = std::thread::Builder::new().name("memory-report".into()).spawn(move || {
        let mut last: Option<Vec<i64>> = None;
        let mut sequence = 0u64;
        loop {
            let value = report(stage);
            if value["devices"].as_array().is_none_or(Vec::is_empty) && value["pinned"]["tracked"] == 0 {
                std::thread::sleep(period);
                continue;
            }
            let signature: Vec<i64> = value["devices"].as_array().into_iter().flatten()
                .flat_map(|d| [d["used"].as_i64().unwrap_or(0), d["tracked"].as_i64().unwrap_or(0)])
                .chain([value["pinned"]["tracked"].as_i64().unwrap_or(0)])
                .collect();
            let moved = last.as_ref().is_none_or(|previous| previous.iter().zip(&signature)
                .any(|(a, b)| (a - b).abs() >= 64 << 20));
            if moved {
                tracing::info!(target: "cuteafd::memory", sequence, report = %value, "memory ledger");
                sequence += 1;
                last = Some(signature);
            }
            std::thread::sleep(period);
        }
    });
}

/// Frees the pinned upload staging the weight loaders grew (see
/// `NativeLibrary::release_sync_h2d_staging`) once a family's weights are resident.
pub(crate) fn release_load_staging(library: &cuteafd_ffi::NativeLibrary) {
    match library.release_sync_h2d_staging() {
        Ok(0) => {}
        Ok(bytes) => tracing::info!(bytes, "released load-time pinned upload staging"),
        Err(error) => tracing::warn!(%error, "could not release load-time pinned upload staging"),
    }
}
