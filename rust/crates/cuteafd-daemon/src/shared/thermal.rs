//! Sampler for the GPU throttle report (every serving role).
//!
//! One thread, off every request path. Source, in order of preference:
//!
//! 1. gpu-powerd's realtime stream (github.com/tpurtell/gpu-powerd): a Unix socket
//!    (`CUTEAFD_POWER_GOVERNOR_STREAM`, default `/run/gpu-powerd/stream.sock`) that
//!    publishes one JSON line per sample with every GPU's readings and plan. With the
//!    stream connected the engine makes no NVML reads at all.
//! 2. NVML at [`thermal::SAMPLE_HZ`] (about 0.1 ms per GPU) when the socket is absent, or
//!    the stream has been silent for [`STALL`] or dropped. The thread keeps polling the
//!    socket and goes back to the stream when it reconnects.
//!
//! Samples go to `cuteafd_core::thermal`, which `/v1/stats` (current values) and
//! `/v1/thermal` (interval summaries for the bench) read. Throttle seconds per cause are
//! estimated from the reason bits in both modes. In NVML mode the governor's state file
//! (`CUTEAFD_POWER_GOVERNOR_STATE`, default `/run/gpu-powerd/state.json`) is still attached
//! as `governor` when readable. A periodic log line puts current values on the console.
//! `CUTEAFD_THERMAL=off` disables it.
//!
//! GPUs: the devices `CUDA_VISIBLE_DEVICES` names (UUIDs or indices); all when unset.
//! The stream's indices are the host's NVML order.
use cuteafd_core::thermal::{self, GpuInfo, Reading, Source};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const LOG_EVERY: Duration = Duration::from_secs(30);
const STATE_FILE: &str = "/run/gpu-powerd/state.json";
const STREAM_SOCKET: &str = "/run/gpu-powerd/stream.sock";
/// A stream silent this long counts as dropped (gpu-powerd publishes every second by default).
const STALL: Duration = Duration::from_secs(3);
const RECONNECT_EVERY: Duration = Duration::from_secs(2);

/// Which devices to sample: those named by `CUDA_VISIBLE_DEVICES` (UUIDs or indices), else all.
fn selected(visible: Option<&str>, uuid: &str, index: u32) -> bool {
    let Some(visible) = visible.map(str::trim).filter(|v| !v.is_empty()) else { return true };
    visible.split(',').map(str::trim).any(|entry| {
        if entry.starts_with("GPU-") { uuid.starts_with(entry) } else { entry.parse::<u32>().is_ok_and(|i| i == index) }
    })
}

fn governor_note() -> Option<Value> {
    let path = std::env::var("CUTEAFD_POWER_GOVERNOR_STATE").unwrap_or_else(|_| STATE_FILE.into());
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// One stream message mapped to the selected GPUs' identities and readings.
struct Message {
    gpus: Vec<(GpuInfo, Reading)>,
    note: Value,
}

fn number(value: &Value) -> Option<f64> { value.as_f64().filter(|v| v.is_finite()) }

fn milli(watts: Option<f64>) -> Option<u32> { watts.map(|w| (w.max(0.0) * 1000.0).round() as u32) }

/// Parse one line of gpu-powerd's stream (its state JSON), keeping the GPUs `visible` names.
/// `None` for a malformed line or one from a daemon that has exited (`running: false`).
fn parse_message(line: &str, visible: Option<&str>) -> Option<Message> {
    let note: Value = serde_json::from_str(line).ok()?;
    if note["running"].as_bool() == Some(false) { return None; }
    let mut gpus = Vec::new();
    for g in note["gpus"].as_array()? {
        let uuid = g["uuid"].as_str()?.to_string();
        let index = g["index"].as_u64()? as u32;
        if !selected(visible, &uuid, index) { continue; }
        let r = &g["reading"];
        let info = GpuInfo { index, uuid, pci_bus_id: g["pci"].as_str().unwrap_or_default().to_string(),
            name: g["name"].as_str().unwrap_or_default().to_string(), ..GpuInfo::default() };
        let reading = Reading {
            temperature_c: number(&r["temperature_c"]).map(|v| v.round() as i32),
            tlimit_margin_c: number(&r["tlimit_c"]).map(|v| v.round() as i32),
            power_mw: milli(number(&r["power_w"])),
            power_limit_mw: milli(number(&r["limit_w"]).or_else(|| number(&g["limit_w"]))),
            sm_clock_mhz: number(&r["sm_clock_mhz"]).map(|v| v.round() as u32),
            mem_clock_mhz: number(&r["mem_clock_mhz"]).map(|v| v.round() as u32),
            utilization_pct: number(&r["utilization_pct"]).map(|v| v.round() as u32),
            fan_pct: number(&r["fan_pct"]).map(|v| v.round() as u32),
            reasons: r["throttle_mask"].as_u64(),
        };
        gpus.push((info, reading));
    }
    if gpus.is_empty() { return None; }
    Some(Message { gpus, note })
}

struct Config {
    socket: PathBuf,
    visible: Option<String>,
    period: Duration,
    stall: Duration,
}

impl Config {
    fn from_env() -> Self {
        Self {
            socket: std::env::var("CUTEAFD_POWER_GOVERNOR_STREAM").unwrap_or_else(|_| STREAM_SOCKET.into()).into(),
            visible: std::env::var("CUDA_VISIBLE_DEVICES").ok(),
            period: Duration::from_secs(1) / thermal::SAMPLE_HZ,
            stall: STALL,
        }
    }

    fn connect(&self) -> Option<UnixStream> {
        let stream = UnixStream::connect(&self.socket).ok()?;
        stream.set_read_timeout(Some(self.stall)).ok()?;
        Some(stream)
    }
}

/// Consume `stream` until it ends or stalls, pushing every sample. Returns the number of samples.
fn stream_phase(config: &Config, stream: UnixStream, stop: &AtomicBool) -> usize {
    let mut samples = 0;
    let mut lines = BufReader::new(stream);
    let mut line = String::new();
    while !stop.load(Ordering::Relaxed) {
        line.clear();
        match lines.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let Some(message) = parse_message(&line, config.visible.as_deref()) else { continue };
        let uuids: Vec<String> = message.gpus.iter().map(|(info, _)| info.uuid.clone()).collect();
        if thermal::uuids() != uuids {
            thermal::install(message.gpus.iter().map(|(info, _)| info.clone()).collect());
        }
        let readings: Vec<Reading> = message.gpus.iter().map(|(_, r)| *r).collect();
        thermal::push(thermal::now_s(), Source::GpuPowerd, &readings);
        thermal::set_note(Some(message.note));
        samples += 1;
    }
    samples
}

/// Sample NVML once per period until `poll` reports a connected stream (returned) or `stop` is set.
fn nvml_phase(config: &Config, poll: &mut dyn FnMut() -> Option<UnixStream>, stop: &AtomicBool,
    role: &'static str) -> Option<UnixStream> {
    let nvml = match cuteafd_ffi::nvml::Nvml::open() {
        Ok(nvml) => Some(nvml),
        Err(error) => {
            tracing::info!(role, error = %format!("{error:#}"), "GPU throttle report: no NVML");
            None
        }
    };
    let devices = nvml.as_ref().and_then(|nvml| match nvml.devices() {
        Ok(devices) => Some(devices),
        Err(error) => {
            tracing::warn!(role, error = %format!("{error:#}"), "GPU throttle report: no NVML devices");
            None
        }
    }).unwrap_or_default();
    // Keep the GPUs a stream already registered (same set, same order); else those the container names.
    let installed = thermal::uuids();
    let devices: Vec<_> = if installed.is_empty() {
        devices.into_iter().filter(|d| selected(config.visible.as_deref(), &d.identity.uuid, d.identity.index)).collect()
    } else {
        let mut by_uuid: Vec<_> = devices.into_iter().map(Some).collect();
        installed.iter().filter_map(|uuid| by_uuid.iter_mut()
            .find(|d| d.as_ref().is_some_and(|d| &d.identity.uuid == uuid)).and_then(Option::take)).collect()
    };
    if installed.is_empty() && !devices.is_empty() {
        thermal::install(devices.iter().map(|d| d.identity.clone()).collect());
    }
    let mut tick = 0u32;
    let mut next = Instant::now();
    let mut polled = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        // Looking for the stream again is a cheap connect, tried every couple of seconds.
        if polled.elapsed() >= RECONNECT_EVERY {
            polled = Instant::now();
            if let Some(stream) = poll() { return Some(stream); }
        }
        let t = thermal::now_s();
        if !devices.is_empty() {
            let readings: Vec<_> = devices.iter().map(|d| d.read()).collect();
            thermal::push(t, Source::Nvml, &readings);
            if tick % thermal::GOVERNOR_EVERY == 0 { thermal::set_note(governor_note()); }
        }
        tick = tick.wrapping_add(1);
        next += config.period;
        let now = Instant::now();
        if next > now { std::thread::sleep(next - now); } else { next = now; }
    }
    None
}

/// The fallback phase: sample until `poll` yields a connected stream, which it returns.
type Fallback<'a> = dyn FnMut(&Config, &mut dyn FnMut() -> Option<UnixStream>) -> Option<UnixStream> + 'a;

fn run(config: &Config, stop: &AtomicBool, role: &'static str, nvml: &mut Fallback) {
    let mut stream = config.connect();
    while !stop.load(Ordering::Relaxed) {
        if let Some(connected) = stream.take() {
            tracing::info!(role, socket = %config.socket.display(), "GPU throttle report: reading gpu-powerd's stream");
            let samples = stream_phase(config, connected, stop);
            if stop.load(Ordering::Relaxed) { return; }
            tracing::warn!(role, samples, "GPU throttle report: gpu-powerd stream ended; sampling NVML");
        } else {
            tracing::info!(role, "GPU throttle report: sampling NVML");
        }
        stream = nvml(config, &mut || config.connect());
        if stream.is_none() && !stop.load(Ordering::Relaxed) {
            // NVML phase ended without a stream (cannot happen unless stopped); avoid a hot loop.
            std::thread::sleep(config.period);
        }
    }
}

/// Start the sampler thread for `role` (`coordinator`, `expertd`). Never fails the caller.
pub(crate) fn monitor(role: &'static str) {
    if std::env::var("CUTEAFD_THERMAL").is_ok_and(|v| v == "off" || v == "0") {
        tracing::info!(role, "GPU throttle report off (CUTEAFD_THERMAL=off)");
        return;
    }
    let spawned = std::thread::Builder::new().name("thermal".into()).spawn(move || {
        let config = Config::from_env();
        let stop = AtomicBool::new(false);
        let logger = std::thread::Builder::new().name("thermal-log".into()).spawn(move || {
            let mut logged = Instant::now() - LOG_EVERY;
            loop {
                std::thread::sleep(Duration::from_secs(1));
                if thermal::active() && logged.elapsed() >= LOG_EVERY {
                    logged = Instant::now();
                    let t = thermal::now_s();
                    let since = (t - LOG_EVERY.as_secs_f64()).max(0.0);
                    tracing::info!(target: "cuteafd::thermal", role, current = %thermal::current(),
                        "GPU throttle (last 30 s): {}", thermal::describe(&thermal::window(since, t)));
                }
            }
        });
        if let Err(error) = logger { tracing::warn!(%error, "GPU throttle log thread not started"); }
        run(&config, &stop, role, &mut |config, poll| nvml_phase(config, poll, &stop, role));
    });
    if let Err(error) = spawned { tracing::warn!(%error, "GPU throttle report thread not started"); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::net::UnixListener;
    use std::sync::{Arc, Mutex};

    #[test]
    fn visible_devices_select_by_uuid_prefix_or_index() {
        assert!(selected(None, "GPU-a", 0));
        assert!(selected(Some(""), "GPU-a", 3));
        assert!(selected(Some("GPU-fe5b,GPU-95f8"), "GPU-95f8f212-9131", 1));
        assert!(!selected(Some("GPU-fe5b"), "GPU-95f8f212-9131", 1));
        assert!(selected(Some("1"), "GPU-x", 1));
        assert!(!selected(Some("0"), "GPU-x", 1));
    }

    fn line(power_w: f64, mask: u64, running: bool) -> String {
        let gpu = |index: u32, uuid: &str| serde_json::json!({
            "index": index, "uuid": uuid, "name": "NVIDIA RTX PRO 6000", "pci": "E1:00.0", "limit_w": 450,
            "reading": {"temperature_c": 71.0, "power_w": power_w, "utilization_pct": 99.0, "mem_utilization_pct": 40.0,
                "limit_w": 450.0, "sm_clock_mhz": 2400.0, "mem_clock_mhz": 14000.0, "throttle_mask": mask,
                "fan_pct": 55.0, "tlimit_c": 12.0},
            "plan": {"bound": "budget", "reason": "share"}});
        let mut text = serde_json::json!({"version": 1, "running": running, "budget_w": 900.0,
            "gpus": [gpu(0, "GPU-aaaa-1"), gpu(1, "GPU-bbbb-2")]}).to_string();
        text.push('\n');
        text
    }

    #[test]
    fn stream_lines_map_to_readings_and_the_uuid_filter_applies() {
        let both = parse_message(&line(400.0, 0x4, true), None).unwrap();
        assert_eq!(both.gpus.len(), 2);
        let (info, r) = &both.gpus[1];
        assert_eq!((info.index, info.uuid.as_str(), info.pci_bus_id.as_str()), (1, "GPU-bbbb-2", "E1:00.0"));
        assert_eq!((r.temperature_c, r.tlimit_margin_c, r.power_mw, r.power_limit_mw), (Some(71), Some(12), Some(400_000), Some(450_000)));
        assert_eq!((r.sm_clock_mhz, r.mem_clock_mhz, r.utilization_pct, r.fan_pct, r.reasons),
            (Some(2400), Some(14000), Some(99), Some(55), Some(0x4)));
        let one = parse_message(&line(400.0, 0, true), Some("GPU-bbbb")).unwrap();
        assert_eq!(one.gpus.len(), 1);
        assert_eq!(one.gpus[0].0.uuid, "GPU-bbbb-2");
        assert!(parse_message(&line(400.0, 0, true), Some("GPU-cccc")).is_none());
        assert!(parse_message(&line(400.0, 0, false), None).is_none(), "an exited daemon's last line is not a sample");
        assert!(parse_message("not json", None).is_none());
    }

    /// Stream, then fallback when the publisher goes away, then the stream again on reconnect.
    /// Shares the process-wide throttle ring, so it is the only test here that touches it.
    #[test]
    fn stream_falls_back_to_nvml_and_returns() {
        let dir = std::env::temp_dir().join(format!("cuteafd-thermal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let socket = dir.join("stream.sock");
        let config = Config { socket: socket.clone(), visible: Some("GPU-bbbb".into()),
            period: Duration::from_millis(20), stall: Duration::from_millis(300) };
        let stop = Arc::new(AtomicBool::new(false));
        let sources: Arc<Mutex<Vec<&'static str>>> = Arc::default();
        let driver = {
            let (stop, sources) = (stop.clone(), sources.clone());
            std::thread::spawn(move || {
                let mut fallback = |config: &Config, poll: &mut dyn FnMut() -> Option<UnixStream>| {
                    sources.lock().unwrap().push("fallback");
                    while !stop.load(Ordering::Relaxed) {
                        // A fake NVML sample for the GPU the stream registered.
                        thermal::push(thermal::now_s(), Source::Nvml, &[Reading { power_limit_mw: Some(325_000), ..Reading::default() }]);
                        if let Some(stream) = poll() { return Some(stream); }
                        std::thread::sleep(config.period);
                    }
                    None
                };
                run(&config, &stop, "test", &mut fallback);
            })
        };
        // No socket yet: the fallback runs (no GPUs registered, so nothing to push into).
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(sources.lock().unwrap().as_slice(), ["fallback"]);

        // The daemon appears: the loop reconnects and reads the stream.
        let serve = |lines: &'static [(f64, u64)]| {
            let listener = UnixListener::bind(&socket).unwrap();
            let socket = socket.clone();
            std::thread::spawn(move || {
                let (mut client, _) = listener.accept().unwrap();
                for &(power, mask) in lines {
                    client.write_all(line(power, mask, true).as_bytes()).unwrap();
                    std::thread::sleep(Duration::from_millis(50));
                }
                drop(client);
                drop(listener);
                let _ = std::fs::remove_file(&socket);
            })
        };
        let first = serve(&[(300.0, 0), (310.0, 0x4), (320.0, 0x4), (330.0, 0)]);
        let wait = |what: &str, ok: &dyn Fn() -> bool| {
            let until = Instant::now() + Duration::from_secs(10);
            while !ok() { assert!(Instant::now() < until, "timed out: {what}"); std::thread::sleep(Duration::from_millis(20)); }
        };
        let window = || thermal::window(0.0, thermal::now_s() + 1.0);
        wait("stream samples", &|| window()["gpus"][0]["source"] == "gpu-powerd");
        assert_eq!(thermal::uuids(), ["GPU-bbbb-2"], "only the visible GPU is registered");
        assert_eq!(window()["source"], "gpu-powerd");
        assert_eq!(window()["governor_budget_w"], 900.0);
        first.join().unwrap();

        // The publisher is gone: back to the fallback, which records NVML samples.
        wait("fallback", &|| window()["gpus"][0]["source"] == "nvml");

        // And it returns to the stream when the socket comes back.
        let second = serve(&[(340.0, 0x4), (350.0, 0x4), (360.0, 0x4)]);
        wait("stream again", &|| window()["gpus"][0]["source"] == "gpu-powerd");
        let w = window();
        // Throttle seconds come from the reason bits: four stream samples then three more,
        // sw_power_cap active in five of them, at 1 s per sample only when spaced 1 s apart;
        // here the spacing is ~50 ms, so the estimate is the summed holds (small, positive).
        assert!(w["gpus"][0]["throttle_s"]["sw_power_cap"].as_f64().unwrap() > 0.0);
        assert_eq!(w["gpus"][0]["tlimit_margin_c"]["min"], 12.0);
        second.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        driver.join().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
