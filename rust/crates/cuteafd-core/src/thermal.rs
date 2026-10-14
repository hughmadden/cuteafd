//! GPU throttle report: a process-wide ring of NVML samples (filled by the
//! daemon's background sampler) and the interval summaries the API serves
//! (`/v1/thermal`, `/v1/stats`) and the bench attaches to each measured phase.
//!
//! Nothing here touches a device; the sampler pushes plain readings, from
//! gpu-powerd's realtime stream or from NVML. A window summary answers whether,
//! why and how much a phase was throttled: seconds in each clock-event reason
//! (sample interval x reason active), the share of samples each reason was
//! active, SM clock while busy, temperature, T.Limit headroom (gpu-powerd only)
//! and power.
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Samples per second the NVML sampler takes (temperature, clock, power, reasons).
pub const SAMPLE_HZ: u32 = 1;
/// The NVML sampler re-reads the governor's state file every this many samples.
pub const GOVERNOR_EVERY: u32 = 10;
/// Samples kept per GPU (two hours at [`SAMPLE_HZ`]); a longer window is summarized over what remains.
const KEEP: usize = 2 * 3600 * SAMPLE_HZ as usize;
/// A sample counts as busy (for the busy-clock statistics) at or above this utilization.
const BUSY_PCT: u32 = 50;

// `nvmlClocksEventReasons` bits.
pub const REASON_SW_POWER_CAP: u64 = 0x4;
pub const REASON_HW_SLOWDOWN: u64 = 0x8;
pub const REASON_SW_THERMAL: u64 = 0x20;
pub const REASON_HW_THERMAL: u64 = 0x40;
pub const REASON_HW_POWER_BRAKE: u64 = 0x80;
const REASONS: [(&str, u64); 4] = [("sw_power_cap", REASON_SW_POWER_CAP), ("sw_thermal", REASON_SW_THERMAL),
    ("hw_thermal", REASON_HW_THERMAL), ("hw_power_brake", REASON_HW_POWER_BRAKE)];

/// A device's identity and fixed limits, recorded once.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GpuInfo {
    pub index: u32,
    pub uuid: String,
    pub pci_bus_id: String,
    pub name: String,
    /// T.Limit offsets of the slowdown, shutdown and maximum-operating points (degrees).
    pub slowdown_tlimit_c: Option<i32>,
    pub shutdown_tlimit_c: Option<i32>,
    pub gpu_max_tlimit_c: Option<i32>,
}

/// One reading; `None` where the board does not expose a value or the tier was not read.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Reading {
    pub temperature_c: Option<i32>,
    pub tlimit_margin_c: Option<i32>,
    pub power_mw: Option<u32>,
    pub power_limit_mw: Option<u32>,
    pub sm_clock_mhz: Option<u32>,
    pub utilization_pct: Option<u32>,
    pub reasons: Option<u64>,
    /// Memory clock and fan speed: only gpu-powerd's stream supplies them.
    pub mem_clock_mhz: Option<u32>,
    pub fan_pct: Option<u32>,
}

/// Where a sample came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Nvml,
    GpuPowerd,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self { Source::Nvml => "nvml", Source::GpuPowerd => "gpu-powerd" }
    }
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    t: f64,
    reading: Reading,
    source: Source,
}

struct Gpu {
    info: GpuInfo,
    samples: VecDeque<Sample>,
    /// The newest reading that carried a T.Limit margin (gpu-powerd only), shown in `current`.
    slow: Reading,
    /// The governor's dominant plan bound per slow tick: (time, bound), oldest first.
    bounds: VecDeque<(f64, String)>,
}

#[derive(Default)]
struct State {
    gpus: Vec<Gpu>,
    /// Extra facts merged into the report (e.g. a host power governor's state).
    note: Option<Value>,
}

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Seconds on the report's clock (process-relative, monotonic): the cursor the bench passes back.
pub fn now_s() -> f64 {
    epoch().elapsed().as_secs_f64()
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::default()))
}

/// Register the sampled GPUs (replaces any earlier set).
pub fn install(gpus: Vec<GpuInfo>) {
    let _ = epoch();
    if let Ok(mut state) = state().lock() {
        state.gpus = gpus.into_iter().map(|info| Gpu { info, samples: VecDeque::new(), slow: Reading::default(), bounds: VecDeque::new() }).collect();
    }
}

/// Whether a sampler has registered any GPU.
pub fn active() -> bool {
    state().lock().map(|s| !s.gpus.is_empty()).unwrap_or(false)
}

/// UUIDs of the registered GPUs, in registration order.
pub fn uuids() -> Vec<String> {
    state().lock().map(|s| s.gpus.iter().map(|g| g.info.uuid.clone()).collect()).unwrap_or_default()
}

/// Append one reading per registered GPU (same order as [`install`]), taken at `t` ([`now_s`]).
pub fn push(t: f64, source: Source, readings: &[Reading]) {
    let Ok(mut state) = state().lock() else { return };
    for (gpu, reading) in state.gpus.iter_mut().zip(readings) {
        if gpu.samples.len() == KEEP { gpu.samples.pop_front(); }
        gpu.samples.push_back(Sample { t, reading: *reading, source });
        if reading.tlimit_margin_c.is_some() { gpu.slow = *reading; }
    }
}

/// Attach a note (e.g. the host power governor's current limit and reason) to every report.
///
/// Also records each GPU's `plan.bound` from the note (`gpus[]`, matched by `index`, else by
/// position) so a window can report the dominant bound.
pub fn set_note(note: Option<Value>) {
    let Ok(mut state) = state().lock() else { return };
    let t = now_s();
    if let Some(entries) = note.as_ref().and_then(|n| n["gpus"].as_array()) {
        for (position, gpu) in state.gpus.iter_mut().enumerate() {
            let entry = entries.iter().find(|e| e["index"].as_u64() == Some(u64::from(gpu.info.index)))
                .or_else(|| entries.get(position));
            if let Some(bound) = entry.and_then(|e| e["plan"]["bound"].as_str()) {
                if gpu.bounds.len() == KEEP { gpu.bounds.pop_front(); }
                gpu.bounds.push_back((t, bound.to_string()));
            }
        }
    }
    state.note = note;
}

/// Mean of `(t, value)` points weighted by the time each value held (until the next point,
/// at most [`MAX_HOLD_S`]; the last point holds one sample period).
pub fn time_weighted_mean(points: &[(f64, f64)]) -> Option<f64> {
    let last = points.len().checked_sub(1)?;
    let (mut sum, mut weight) = (0.0, 0.0);
    for (i, &(t, v)) in points.iter().enumerate() {
        let hold = if i < last { points[i + 1].0 - t } else { 1.0 / f64::from(SAMPLE_HZ) };
        let hold = hold.clamp(0.0, MAX_HOLD_S);
        sum += v * hold;
        weight += hold;
    }
    if weight > 0.0 { Some(sum / weight) } else { Some(points[0].1) }
}

const MAX_HOLD_S: f64 = 2.5 / SAMPLE_HZ as f64;

fn watts(mw: Option<u32>) -> Option<f64> { mw.map(|mw| f64::from(mw) / 1e3) }

fn active_reasons(bits: u64) -> Vec<&'static str> {
    let mut names: Vec<&str> = REASONS.iter().filter(|(_, bit)| bits & bit != 0).map(|(name, _)| *name).collect();
    if bits & REASON_HW_SLOWDOWN != 0 && bits & (REASON_HW_THERMAL | REASON_HW_POWER_BRAKE) == 0 {
        names.push("hw_slowdown");
    }
    names
}

/// The latest reading of every GPU, for the console and `/v1/stats`.
///
/// `queried_s` is when this report was read, on the [`now_s`] clock. Each GPU's
/// `sampled_s` is the time of the reading shown, on that same clock, or `null`
/// when that GPU has no sample yet. A repeated reading keeps `sampled_s` while
/// `queried_s` moves.
pub fn current() -> Value {
    let Ok(state) = state().lock() else { return Value::Null };
    if state.gpus.is_empty() { return Value::Null; }
    let gpus: Vec<Value> = state.gpus.iter().map(|gpu| {
        let sample = gpu.samples.back();
        let last = sample.map(|s| s.reading).unwrap_or_default();
        json!({
            "index": gpu.info.index, "uuid": gpu.info.uuid, "pci_bus_id": gpu.info.pci_bus_id, "name": gpu.info.name,
            "sampled_s": sample.map(|s| s.t),
            "temperature_c": last.temperature_c,
            "tlimit_margin_c": gpu.slow.tlimit_margin_c,
            "mem_clock_mhz": last.mem_clock_mhz, "fan_pct": last.fan_pct,
            "power_w": watts(last.power_mw), "power_limit_w": watts(last.power_limit_mw),
            "sm_clock_mhz": last.sm_clock_mhz, "utilization_pct": last.utilization_pct,
            "throttling": last.reasons.map(active_reasons),
            "slowdown_tlimit_c": gpu.info.slowdown_tlimit_c, "shutdown_tlimit_c": gpu.info.shutdown_tlimit_c,
        })
    }).collect();
    let mut value = json!({"queried_s": now_s(), "gpus": gpus});
    if let Some(note) = &state.note { value["governor"] = note.clone(); }
    value
}

#[derive(Default)]
struct Stat { n: u32, sum: f64, min: f64, max: f64 }

impl Stat {
    fn add(&mut self, value: Option<f64>) {
        let Some(v) = value else { return };
        if self.n == 0 { self.min = v; self.max = v; }
        self.n += 1;
        self.sum += v;
        self.min = self.min.min(v);
        self.max = self.max.max(v);
    }
    fn mean(&self) -> Option<f64> { (self.n > 0).then(|| self.sum / f64::from(self.n)) }
    fn min(&self) -> Option<f64> { (self.n > 0).then_some(self.min) }
    fn max(&self) -> Option<f64> { (self.n > 0).then_some(self.max) }
}

fn round1(v: Option<f64>) -> Value { v.map_or(Value::Null, |v| json!((v * 10.0).round() / 10.0)) }
fn round3(v: Option<f64>) -> Value { v.map_or(Value::Null, |v| json!((v * 1000.0).round() / 1000.0)) }

/// Seconds each sample covers: until the next one (at most [`MAX_HOLD_S`]); the last holds one
/// [`SAMPLE_HZ`] period.
fn holds(times: &[f64]) -> Vec<f64> {
    (0..times.len()).map(|i| match times.get(i + 1) {
        Some(next) => (next - times[i]).clamp(0.0, MAX_HOLD_S),
        None => 1.0 / f64::from(SAMPLE_HZ),
    }).collect()
}

fn summarize_gpu(gpu: &Gpu, from: f64, to: f64) -> Value {
    let samples = &gpu.samples;
    let start = samples.partition_point(|s| s.t < from);
    let end = samples.partition_point(|s| s.t <= to);
    let window = samples.range(start..end);
    let (mut temp, mut margin, mut power, mut limit, mut clock, mut busy_clock) =
        (Stat::default(), Stat::default(), Stat::default(), Stat::default(), Stat::default(), Stat::default());
    let times: Vec<f64> = window.clone().map(|s| s.t).collect();
    let hold = holds(&times);
    let mut limit_points: Vec<(f64, f64)> = Vec::new();
    let mut reason_hits = [0u32; 4];
    let mut reason_seconds = [0.0f64; 4];
    let mut reason_samples = 0u32;
    let (mut n, mut busy) = (0u32, 0u32);
    let (mut first_t, mut last_t) = (None, None);
    let mut source = None;
    for (sample, hold) in window.zip(&hold) {
        let r = &sample.reading;
        n += 1;
        source = Some(sample.source);
        first_t.get_or_insert(sample.t);
        last_t = Some(sample.t);
        temp.add(r.temperature_c.map(f64::from));
        margin.add(r.tlimit_margin_c.map(f64::from));
        power.add(watts(r.power_mw));
        limit.add(watts(r.power_limit_mw));
        if let Some(w) = watts(r.power_limit_mw) { limit_points.push((sample.t, w)); }
        clock.add(r.sm_clock_mhz.map(f64::from));
        if r.utilization_pct.is_some_and(|u| u >= BUSY_PCT) {
            busy += 1;
            busy_clock.add(r.sm_clock_mhz.map(f64::from));
        }
        if let Some(bits) = r.reasons {
            reason_samples += 1;
            for (index, (_, bit)) in REASONS.iter().enumerate() {
                if bits & bit != 0 { reason_hits[index] += 1; reason_seconds[index] += hold; }
            }
        }
    }
    // Throttle seconds per cause: sample interval x reason active (the same estimate from either source).
    let throttle_s = if reason_samples == 0 { Value::Null } else {
        let mut out = serde_json::Map::new();
        for (index, (name, _)) in REASONS.iter().enumerate() { out.insert((*name).into(), round3(Some(reason_seconds[index]))); }
        Value::Object(out)
    };
    let mut reason_fraction = serde_json::Map::new();
    for (index, (name, _)) in REASONS.iter().enumerate() {
        reason_fraction.insert((*name).into(),
            round3((reason_samples > 0).then(|| f64::from(reason_hits[index]) / f64::from(reason_samples))));
    }
    let mut tally: Vec<(&str, u32)> = Vec::new();
    for (_, bound) in gpu.bounds.iter().filter(|(t, _)| *t >= from && *t <= to) {
        match tally.iter_mut().find(|(b, _)| b == bound) {
            Some(entry) => entry.1 += 1,
            None => tally.push((bound, 1)),
        }
    }
    let dominant = tally.iter().max_by_key(|(_, n)| *n).map(|(b, _)| *b);
    json!({
        "index": gpu.info.index, "uuid": gpu.info.uuid, "pci_bus_id": gpu.info.pci_bus_id,
        "governor_bound": dominant,
        "samples": n,
        "source": source.map(Source::name),
        "covered_s": round3(first_t.zip(last_t).map(|(a, b)| b - a)),
        "throttle_s": throttle_s,
        "reason_fraction": Value::Object(reason_fraction),
        "sm_clock_mhz": {"min": round1(clock.min()), "mean": round1(clock.mean())},
        "busy": {"fraction": round3((n > 0).then(|| f64::from(busy) / f64::from(n))),
            "sm_clock_min_mhz": round1(busy_clock.min()), "sm_clock_mean_mhz": round1(busy_clock.mean())},
        "temperature_c": {"max": round1(temp.max()), "mean": round1(temp.mean())},
        "tlimit_margin_c": {"min": round1(margin.min()), "mean": round1(margin.mean())},
        "power_w": {"mean": round1(power.mean()), "max": round1(power.max())},
        "power_limit_w": {"min": round1(limit.min()), "max": round1(limit.max()),
            "mean": round1(time_weighted_mean(&limit_points))},
    })
}

/// Summary of every GPU over `[from, to]` on the [`now_s`] clock; `Value::Null` without a sampler.
pub fn window(from: f64, to: f64) -> Value {
    let Ok(state) = state().lock() else { return Value::Null };
    if state.gpus.is_empty() { return Value::Null; }
    let gpus: Vec<Value> = state.gpus.iter().map(|gpu| summarize_gpu(gpu, from, to)).collect();
    let mut value = json!({"from_s": round3(Some(from)), "to_s": round3(Some(to)),
        "seconds": round3(Some((to - from).max(0.0))), "gpus": gpus});
    if let Some(note) = &state.note {
        value["governor"] = note.clone();
        value["governor_name"] = json!("gpu-powerd");
        value["source"] = json!(state.gpus.iter().find_map(|g| g.samples.back()).map(|s| s.source.name()));
        value["governor_budget_w"] = note["budget_w"].clone();
    }
    value
}

/// One line per GPU for a log or summary table.
pub fn describe(window: &Value) -> String {
    window["gpus"].as_array().into_iter().flatten().map(|g| {
        let t = &g["throttle_s"];
        format!("gpu{} power-cap {}s thermal {}s/{}s brake {}s · busy clock {} MHz (min {}) · max {} C · min headroom {} C · {} W mean",
            g["index"], t["sw_power_cap"], t["sw_thermal"], t["hw_thermal"], t["hw_power_brake"],
            g["busy"]["sm_clock_mean_mhz"], g["busy"]["sm_clock_min_mhz"], g["temperature_c"]["max"],
            g["tlimit_margin_c"]["min"], g["power_w"]["mean"])
    }).collect::<Vec<_>>().join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(samples: &[(f64, Reading)]) -> Gpu {
        Gpu { info: GpuInfo { index: 1, uuid: "GPU-x".into(), ..GpuInfo::default() },
            samples: samples.iter().map(|&(t, reading)| Sample { t, reading, source: Source::Nvml }).collect(), slow: Reading::default(), bounds: VecDeque::new() }
    }

    fn fast(temp: i32, clock: u32, util: u32, watts: u32, bits: u64) -> Reading {
        Reading { temperature_c: Some(temp), sm_clock_mhz: Some(clock), utilization_pct: Some(util),
            power_mw: Some(watts * 1000), power_limit_mw: Some(325_000), reasons: Some(bits), ..Reading::default() }
    }

    fn with_margin(mut r: Reading, margin: i32) -> Reading {
        r.tlimit_margin_c = Some(margin);
        r
    }

    #[test]
    fn throttle_seconds_come_from_reason_bits_times_sample_interval() {
        let g = gpu(&[
            (0.0, with_margin(fast(40, 180, 0, 30, 0), 53)),
            (1.0, fast(60, 2600, 100, 320, REASON_SW_POWER_CAP)),
            (2.0, with_margin(fast(70, 1400, 100, 325, REASON_SW_POWER_CAP), 23)),
            (3.0, fast(72, 1300, 100, 324, REASON_SW_POWER_CAP | REASON_HW_THERMAL)),
            (4.0, with_margin(fast(74, 1200, 100, 324, REASON_SW_POWER_CAP), 19)),
        ]);
        let w = summarize_gpu(&g, 0.5, 4.0);
        assert_eq!(w["samples"], 4);
        assert_eq!(w["source"], "nvml");
        // Four samples at 1 s each, the last holding one sample period.
        assert_eq!(w["throttle_s"]["sw_power_cap"], 4.0);
        assert_eq!(w["throttle_s"]["hw_thermal"], 1.0);
        assert_eq!(w["throttle_s"]["sw_thermal"], 0.0);
        assert_eq!(w["reason_fraction"]["sw_power_cap"], 1.0);
        assert_eq!(w["reason_fraction"]["hw_thermal"], 0.25);
        assert_eq!(w["busy"]["fraction"], 1.0);
        assert_eq!(w["busy"]["sm_clock_min_mhz"], 1200.0);
        assert_eq!(w["temperature_c"]["max"], 74.0);
        assert_eq!(w["tlimit_margin_c"]["min"], 19.0);
        assert_eq!(w["power_limit_w"]["max"], 325.0);
    }

    #[test]
    fn idle_samples_do_not_drag_the_busy_clock_and_a_gap_is_capped() {
        let g = gpu(&[
            (0.0, fast(40, 180, 0, 30, REASON_SW_POWER_CAP)),
            (100.0, fast(41, 2600, 90, 200, 0)),
        ]);
        let w = summarize_gpu(&g, 0.0, 100.0);
        assert_eq!(w["busy"]["sm_clock_min_mhz"], 2600.0);
        assert_eq!(w["sm_clock_mhz"]["min"], 180.0);
        assert_eq!(w["throttle_s"]["sw_power_cap"], MAX_HOLD_S);
        assert_eq!(w["throttle_s"]["hw_thermal"], 0.0);
        assert!(w["tlimit_margin_c"]["min"].is_null(), "NVML mode has no margin");
    }

    #[test]
    fn an_empty_window_has_no_throttle_seconds_and_names_no_reason() {
        let g = gpu(&[(0.0, fast(40, 180, 0, 30, 0))]);
        let w = summarize_gpu(&g, 5.0, 6.0);
        assert_eq!(w["samples"], 0);
        assert!(w["throttle_s"].is_null());
        assert_eq!(active_reasons(REASON_SW_POWER_CAP | REASON_HW_SLOWDOWN | REASON_HW_THERMAL),
            vec!["sw_power_cap", "hw_thermal"]);
        assert_eq!(active_reasons(REASON_HW_SLOWDOWN), vec!["hw_slowdown"]);
    }

    #[test]
    fn time_weighted_mean_weights_by_hold_time() {
        assert_eq!(time_weighted_mean(&[]), None);
        assert_eq!(time_weighted_mean(&[(0.0, 400.0)]), Some(400.0));
        // 600 W held 1 s, 300 W held for the one sample period the last sample covers.
        let m = time_weighted_mean(&[(0.0, 600.0), (1.0, 300.0)]).unwrap();
        assert!((m - 450.0).abs() < 1e-9, "{m}");
        // Uneven spacing: 600 W for 2 s, 300 W for 1 s.
        let m = time_weighted_mean(&[(0.0, 600.0), (2.0, 300.0)]).unwrap();
        assert!((m - 500.0).abs() < 1e-9, "{m}");
        // A gap longer than MAX_HOLD_S is capped (2.5 s of 600 W, 1 s of 300 W).
        let m = time_weighted_mean(&[(0.0, 600.0), (100.0, 300.0)]).unwrap();
        assert!((m - (600.0 * 2.5 + 300.0) / 3.5).abs() < 1e-9, "{m}");
    }

    #[test]
    fn window_reports_mean_limit_and_dominant_bound() {
        let mut g = gpu(&[]);
        for i in 0..10 {
            let mut r = fast(60, 2000, 100, 400, 0);
            r.power_limit_mw = Some(if i < 5 { 600_000 } else { 400_000 });
            g.samples.push_back(Sample { t: f64::from(i), reading: r, source: Source::Nvml });
        }
        g.bounds = VecDeque::from([(0.0, "max".to_string()), (2.0, "budget".to_string()), (4.0, "budget".to_string())]);
        let w = summarize_gpu(&g, 0.0, 10.0);
        assert_eq!(w["power_limit_w"]["mean"], 500.0);
        assert_eq!(w["power_limit_w"]["min"], 400.0);
        assert_eq!(w["governor_bound"], "budget");
    }

    /// `install`/`push` share process state; one test keeps the cases from racing.
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) { install(Vec::new()); }
    }

    #[test]
    fn current_keeps_each_gpus_sample_time_while_the_query_time_advances() {
        let _reset = Reset;
        install(vec![
            GpuInfo { index: 0, uuid: "GPU-a".into(), name: "A".into(), ..GpuInfo::default() },
            GpuInfo { index: 1, uuid: "GPU-b".into(), name: "B".into(), ..GpuInfo::default() },
        ]);

        let none = current();
        assert!(none.get("now_s").is_none(), "the query time is queried_s");
        assert!(none["queried_s"].as_f64().is_some());
        assert!(none["gpus"][0]["sampled_s"].is_null());
        assert!(none["gpus"][1]["sampled_s"].is_null());
        assert!(none["gpus"][0]["temperature_c"].is_null());

        // An empty reading is still a sample: the timestamp is set and the other GPU stays empty.
        let blank = 1_000_000.0;
        push(blank, Source::Nvml, &[Reading::default()]);
        let blanked = current();
        assert_eq!(blanked["gpus"][0]["sampled_s"].as_f64(), Some(blank));
        assert!(blanked["gpus"][0]["temperature_c"].is_null());
        assert!(blanked["gpus"][1]["sampled_s"].is_null());

        let t = 1_000_010.0;
        push(t, Source::Nvml, &[fast(40, 180, 0, 30, 0), fast(55, 200, 10, 40, 0)]);
        let first = current();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let second = current();
        assert_eq!(first["gpus"][0]["sampled_s"].as_f64(), Some(t));
        assert_eq!(second["gpus"][0]["sampled_s"].as_f64(), Some(t));
        assert_eq!(first["gpus"][1]["sampled_s"].as_f64(), Some(t));
        assert_eq!(second["gpus"][1]["sampled_s"].as_f64(), Some(t));
        assert_eq!(second["gpus"][0]["temperature_c"].as_i64(), Some(40));
        assert_eq!(second["gpus"][1]["temperature_c"].as_i64(), Some(55));
        let (q0, q1) = (first["queried_s"].as_f64().unwrap(), second["queried_s"].as_f64().unwrap());
        assert!(q1 > q0, "query time {q0} -> {q1}");
        assert_ne!(second["gpus"][0]["sampled_s"].as_f64(), Some(q1));

        // A shorter push updates only the leading GPUs, so their sample times can differ.
        let later = 1_000_020.0;
        push(later, Source::GpuPowerd, &[fast(41, 180, 0, 31, 0)]);
        let split = current();
        assert_eq!(split["gpus"][0]["sampled_s"].as_f64(), Some(later));
        assert_eq!(split["gpus"][1]["sampled_s"].as_f64(), Some(t));
        assert_eq!(split["gpus"][0]["temperature_c"].as_i64(), Some(41));
        assert_eq!(split["gpus"][1]["temperature_c"].as_i64(), Some(55));
        assert!(split.get("now_s").is_none());
    }
}
