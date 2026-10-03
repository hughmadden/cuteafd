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

/// The kernel page cache (`Cached` in /proc/meminfo), bytes.
pub(crate) fn cached_bytes() -> Option<u64> {
    meminfo()["Cached"].as_u64()
}

/// One GPU holding KV records: its records per logical token and the bytes
/// that must stay free there after the pool (workspaces, graphs, drafter,
/// headroom) — the planner's per-device costs.
pub(crate) struct KvDevice {
    pub device: i32,
    pub bytes_per_token: u64,
    pub reserve_bytes: u64,
}

/// The largest pool (whole `unit_rows` units, at most `target` tokens) every
/// device can hold in its free memory now, after its reserve. Restores the
/// calling thread's device.
pub(crate) fn auto_pool_tokens(library: &cuteafd_ffi::NativeLibrary, devices: &[KvDevice], unit_rows: u64,
    target: u64) -> anyhow::Result<u64> {
    let current = library.cuda_get_device()?;
    let mut free = Vec::with_capacity(devices.len());
    for device in devices {
        library.cuda_set_device(device.device)?;
        let sample = library.cuda_memory_info();
        library.cuda_set_device(current)?;
        let (available, _) = sample?;
        free.push(available as i64 - device.reserve_bytes as i64);
    }
    let per_token: Vec<u64> = devices.iter().map(|d| d.bytes_per_token).collect();
    let tokens = cuteafd_core::memory_layout::size_pool(&free, &per_token, unit_rows, target);
    tracing::info!(tokens, target, ?free, ?per_token, "automatic KV pool from free memory after fixed costs");
    anyhow::ensure!(tokens >= unit_rows, "no room for a KV pool after fixed costs (free after reserve {free:?} bytes)");
    Ok(tokens)
}

/// Bytes of a checkpoint directory's safetensors shards (a drafter's resident
/// size when it keeps its checkpoint representation).
pub(crate) fn safetensors_bytes(directory: &std::path::Path) -> u64 {
    std::fs::read_dir(directory).map(|entries| entries.flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        .filter_map(|e| std::fs::metadata(e.path()).ok().map(|m| m.len())).sum()).unwrap_or(0)
}

/// The planner's automatic KV pool for an engine about to allocate its cache:
/// each GPU's free memory now, minus what the planner says is still to come
/// there (step workspaces, peer exchange, drafter, recurrent state, prefix
/// marks, graph executables, headroom), over the family's records per token.
/// `devices` lists the KV-owning GPUs, lead first; `drafter` is an external
/// drafter checkpoint the lead GPU will load.
pub(crate) fn planned_pool_tokens(library: &cuteafd_ffi::NativeLibrary, snapshot: &std::path::Path, devices: &[i32],
    drafter: Option<&std::path::Path>, prefill_rows: usize, slots: usize) -> anyhow::Result<usize> {
    use anyhow::Context;
    let checkpoint = cuteafd_loader::plan::Checkpoint::open(snapshot)?;
    let family = cuteafd_loader::plan::family::detect(&checkpoint).context("no family for this checkpoint")?;
    let model = family.open(&checkpoint).map_err(|e| anyhow::anyhow!("{}", e.0))?;
    let geometry = model.cache_geometry(cuteafd_loader::serving_capacity::CacheOptions {
        coordinator_ranks: devices.len(), ..Default::default() })?
        .with_context(|| format!("{} has no cache geometry for {} GPUs", family.id(), devices.len()))?;
    let costs = cuteafd_loader::plan::layout::family_costs(family.id());
    let headroom = cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes;
    let draft = drafter.map_or(0, |d| safetensors_bytes(d) + (1300 << 20));
    let unit = geometry.logical_unit_rows.max(1);
    let split = devices.len() == 2;
    let kv: Vec<KvDevice> = devices.iter().zip(&geometry.ranks).enumerate().map(|(index, (&device, rank))| {
        let role = if !split { 0 } else if index == 0 { 1 } else { 2 };
        let workspace = costs.workspace_bytes[role] * prefill_rows.max(1) as u64 / 4096;
        let state = rank.fixed_state_bytes + rank.active_state_per_sequence_bytes * slots as u64;
        let marks = rank.retained_mark_bytes * costs.mark_slots;
        KvDevice {
            device,
            bytes_per_token: (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes).div_ceil(unit),
            reserve_bytes: workspace + if split { costs.exchange_bytes } else { 0 } + if index == 0 { draft } else { 0 }
                + state + marks + costs.graph_bytes[role] + headroom,
        }
    }).collect();
    let tokens = auto_pool_tokens(library, &kv, unit, cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS)?;
    Ok(usize::try_from(tokens)?)
}
