//! Signed physical-use versus admission-owner reconciliation. Samples are
//! diagnostic: never turn a missing sample or a positive delta into refusal.
use super::*;
use std::sync::{Mutex, OnceLock};

fn records() -> &'static Mutex<Value> {
    static RECORDS: OnceLock<Mutex<Value>> = OnceLock::new();
    RECORDS.get_or_init(|| Mutex::new(json!({"schema_version": 1, "ready": null, "first_request": null})))
}
fn workers() -> &'static Mutex<BTreeMap<u32, Value>> {
    static WORKERS: OnceLock<Mutex<BTreeMap<u32, Value>>> = OnceLock::new();
    WORKERS.get_or_init(Default::default)
}

pub(crate) fn spark_ready(report: &cuteafd_core::ModelAdmissionResponse) {
    workers().lock().unwrap_or_else(|e| e.into_inner()).insert(report.rank,
        json!({"admission_sample": report.runtime.sample,
            "physical_sample": report.runtime.detail.memory_reconciliation}));
}

pub(crate) fn snapshot() -> Value {
    records().lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn row(device: &str, measured: Option<u64>, owners: &cuteafd_core::memory_policy::DeviceBreakdown,
    provenance: Value, reason: Option<&str>) -> Value {
    let padding: u64 = owners.reserve.iter().filter(|owner| owner.name == "release padding")
        .map(|owner| owner.bytes).sum();
    let charged = owners.weights_bytes.saturating_add(owners.workspace_bytes)
        .saturating_add(owners.kv_bytes).saturating_add(owners.other_bytes)
        .saturating_add(owners.reserve_bytes()).saturating_sub(padding);
    let delta = measured.map(|used| i128::from(used) - i128::from(charged));
    json!({"device": device, "measured_bytes": measured, "charged_bytes": charged,
        "padding_bytes": padding, "measured_minus_charged_bytes": delta,
        "beyond_half_padding": delta.map(|delta| delta.abs() > i128::from(padding / 2)),
        "measurement_method": "raw CUDA physical total minus free; device-wide, not process attribution",
        "charge_basis": "admitted named owners excluding release padding", "owners": owners,
        "unavailable_reason": reason, "provenance": provenance})
}

/// Ready runs on a helper thread; first-request runs only once, after completed
/// request retirement. Neither path creates a primary context or synchronizes a
/// partial multi-rank submission.
pub(crate) fn capture(stage: &str) {
    let inputs = memory_ledger::fault::inputs();
    let Some(provenance) = inputs.get("memory_reconciliation_provenance") else { return };
    let Some(devices) = provenance["devices"].as_array() else { return };
    let worker_samples = workers().lock().unwrap_or_else(|e| e.into_inner()).clone();
    let mut rows = Vec::new();
    for (key, value) in inputs.as_object().into_iter().flatten().filter(|(key, _)| key.starts_with("device.")) {
        let Ok(owners) = serde_json::from_value::<cuteafd_core::memory_policy::DeviceBreakdown>(value.clone()) else { continue };
        let id = key.trim_start_matches("device.");
        if let Some(index) = id.strip_prefix("rtx").and_then(|n| n.parse::<usize>().ok()) {
            let physical = devices.get(index).and_then(|value| value["device"].as_i64());
            let measured = physical.and_then(|device| i32::try_from(device).ok())
                .and_then(memory_ledger::fault::device_memory).map(|(free, total)| total.saturating_sub(free) as u64);
            let mut identity = provenance.clone();
            identity["device_sample"] = devices.get(index).cloned().unwrap_or(Value::Null);
            rows.push(row(id, measured, &owners, identity,
                measured.is_none().then_some("no active CUDA primary context sample")));
        } else if let Some(rank) = id.strip_prefix("spark").and_then(|n| n.trim_start_matches('.').parse::<u32>().ok()) {
            let worker = worker_samples.get(&rank);
            let measured = (stage == "ready").then(|| worker.and_then(|value|
                value["physical_sample"]["cuda_used_bytes"].as_u64())).flatten();
            let reason = if stage != "ready" { Some("post-request Spark control-plane telemetry unavailable; see #1631; worker logs its own CUDA sample") }
                else if measured.is_none() { Some("worker omitted optional physical sample during rolling update") } else { None };
            let mut record = row(id, measured, &owners, json!({"coordinator": provenance,
                "worker": worker.and_then(|value| value.get("physical_sample"))}), reason);
            record["rank"] = rank.into();
            // This is the admission-time sample, never labelled post-request.
            record["admission_sample"] = worker.map(|value| value["admission_sample"].clone()).unwrap_or(Value::Null);
            rows.push(record);
        }
    }
    let value = json!({"stage": stage, "devices": rows});
    tracing::info!(target: "cuteafd::memory", record = %value, "memory admission reconciliation");
    records().lock().unwrap_or_else(|e| e.into_inner())[stage] = value;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_delta_keeps_release_padding_separate_and_preserves_missing_samples() {
        let owners = cuteafd_core::memory_policy::DeviceBreakdown { device: "rtx0".into(), cap_bytes: 1000,
            weights_bytes: 60, workspace_bytes: 10, kv_bytes: 20, other_bytes: 0, reserve_overridden: false,
            reserve: vec![cuteafd_core::memory_policy::ReserveComponent { name: "release padding".into(), bytes: 10 }] };
        let measured = row("rtx0", Some(85), &owners, json!({}), None);
        assert_eq!(measured["charged_bytes"], 90);
        assert_eq!(measured["padding_bytes"], 10);
        assert_eq!(measured["measured_minus_charged_bytes"], -5);
        assert_eq!(measured["beyond_half_padding"], false);
        let missing = row("spark0", None, &owners, json!({}), Some("no telemetry"));
        assert!(missing["measured_minus_charged_bytes"].is_null());
        assert!(missing["beyond_half_padding"].is_null());
    }
}
