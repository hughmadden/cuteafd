//! Background NVML sampler for the GPU throttle report (every serving role).
//!
//! One thread, off every request path: the fast tier (temperature, SM clock,
//! power, limit, clock-event reasons, utilization; about 0.1 ms per GPU) at
//! [`thermal::SAMPLE_HZ`], the slow tier (cumulative throttle counters, T.Limit
//! headroom, memory temperature; about 2 ms per GPU) every
//! [`thermal::SLOW_EVERY`] fast samples. Samples go
//! to `cuteafd_core::thermal`, which `/v1/stats` (current values) and
//! `/v1/thermal` (interval summaries for the bench) read. A periodic log line
//! puts current values on the console. `CUTEAFD_THERMAL=off` disables it.
//!
//! GPUs: the devices this process's CUDA context can see (NVML sees every
//! device the container exposes; `CUDA_VISIBLE_DEVICES` names the coordinator's
//! by UUID or index).
//! A host power daemon (gpu-powerd, github.com/tpurtell/gpu-powerd) may publish
//! its state in `CUTEAFD_POWER_GOVERNOR_STATE` (default
//! `/run/gpu-powerd/state.json`); when the file is readable, the report carries
//! it as `governor`.
use cuteafd_core::thermal;
use std::time::{Duration, Instant};

const LOG_EVERY: Duration = Duration::from_secs(30);
const STATE_FILE: &str = "/run/gpu-powerd/state.json";

/// Which NVML devices to sample: those named by `CUDA_VISIBLE_DEVICES` (UUIDs or
/// indices), else all.
fn selected(visible: Option<&str>, uuid: &str, index: u32) -> bool {
    let Some(visible) = visible.map(str::trim).filter(|v| !v.is_empty()) else { return true };
    visible.split(',').map(str::trim).any(|entry| {
        if entry.starts_with("GPU-") { uuid.starts_with(entry) } else { entry.parse::<u32>().is_ok_and(|i| i == index) }
    })
}

fn governor_note() -> Option<serde_json::Value> {
    let path = std::env::var("CUTEAFD_POWER_GOVERNOR_STATE").unwrap_or_else(|_| STATE_FILE.into());
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// Start the sampler thread for `role` (`coordinator`, `expertd`). Never fails the caller.
pub(crate) fn monitor(role: &'static str) {
    if std::env::var("CUTEAFD_THERMAL").is_ok_and(|v| v == "off" || v == "0") {
        tracing::info!(role, "GPU throttle report off (CUTEAFD_THERMAL=off)");
        return;
    }
    let spawned = std::thread::Builder::new().name("thermal".into()).spawn(move || {
        let nvml = match cuteafd_ffi::nvml::Nvml::open() {
            Ok(nvml) => nvml,
            Err(error) => {
                tracing::info!(role, error = %format!("{error:#}"), "GPU throttle report unavailable (no NVML)");
                return;
            }
        };
        let devices = match nvml.devices() {
            Ok(devices) => devices,
            Err(error) => {
                tracing::warn!(role, error = %format!("{error:#}"), "GPU throttle report: no NVML devices");
                return;
            }
        };
        // `CUDA_VISIBLE_DEVICES` index entries refer to the container's NVML order.
        let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        let devices: Vec<_> = devices.into_iter()
            .filter(|d| selected(visible.as_deref(), &d.identity.uuid, d.identity.index)).collect();
        if devices.is_empty() { return; }
        thermal::install(devices.iter().map(|d| d.identity.clone()).collect());
        for d in &devices {
            let first = d.read(true);
            tracing::info!(role, gpu = d.identity.index, uuid = %d.identity.uuid, pci = %d.identity.pci_bus_id,
                name = %d.identity.name, slowdown_tlimit_c = ?d.identity.slowdown_tlimit_c,
                shutdown_tlimit_c = ?d.identity.shutdown_tlimit_c,
                memory_temperature = if first.memory_temperature_c.is_some() { "available" } else { "N/A" },
                tlimit_margin = if first.tlimit_margin_c.is_some() { "available" } else { "N/A" },
                power_limit = if first.power_limit_mw.is_some() { "available" } else { "N/A" },
                "GPU throttle report sampling");
        }
        let period = Duration::from_secs(1) / thermal::SAMPLE_HZ;
        let mut tick = 0u32;
        let mut logged = Instant::now() - LOG_EVERY;
        let mut next = Instant::now();
        loop {
            let slow = tick % thermal::SLOW_EVERY == 0;
            let t = thermal::now_s();
            let readings: Vec<_> = devices.iter().map(|d| d.read(slow)).collect();
            thermal::push(t, &readings);
            if slow { thermal::set_note(governor_note()); }
            if logged.elapsed() >= LOG_EVERY {
                logged = Instant::now();
                let since = (t - LOG_EVERY.as_secs_f64()).max(0.0);
                tracing::info!(target: "cuteafd::thermal", role, current = %thermal::current(),
                    "GPU throttle (last 30 s): {}", thermal::describe(&thermal::window(since, t)));
            }
            tick = tick.wrapping_add(1);
            next += period;
            let now = Instant::now();
            if next > now { std::thread::sleep(next - now); } else { next = now; }
        }
    });
    if let Err(error) = spawned { tracing::warn!(%error, "GPU throttle report thread not started"); }
}

#[cfg(test)]
mod tests {
    use super::selected;

    #[test]
    fn visible_devices_select_by_uuid_prefix_or_index() {
        assert!(selected(None, "GPU-a", 0));
        assert!(selected(Some(""), "GPU-a", 3));
        assert!(selected(Some("GPU-fe5b,GPU-95f8"), "GPU-95f8f212-9131", 1));
        assert!(!selected(Some("GPU-fe5b"), "GPU-95f8f212-9131", 1));
        assert!(selected(Some("1"), "GPU-x", 1));
        assert!(!selected(Some("0"), "GPU-x", 1));
    }
}
