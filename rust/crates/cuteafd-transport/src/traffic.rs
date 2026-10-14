//! Off-path fabric observations. NIC data counters are four-octet words;
//! engine counters are protocol wire bytes (not payloads or receive capacities).
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::sync::{atomic::{AtomicBool, AtomicU64, Ordering}, Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PortSample {
    pub device: String,
    pub port: u32,
    pub state: Option<String>,
    pub link_gbps: Option<f64>,
    /// Missing or unreadable counters are absent, not zero.
    pub counters: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_xmit_wait_ticks_per_s: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineSample {
    pub device: String,
    pub port: u32,
    pub peer: String,
    #[serde(default)]
    pub local_gid: String,
    /// Subset of sent_bytes posted as one-sided writes, including their flags.
    #[serde(default)]
    pub written_sent_bytes: u64,
    pub receives_writes: bool,
    pub sent_bytes: u64,
    pub received_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HostSample {
    /// Process incarnation; a restart must not be interpreted as a counter wrap.
    pub incarnation: String,
    /// Seconds on this host's monotonic clock, never compared across hosts.
    pub sampled_s: f64,
    pub ports: Vec<PortSample>,
    pub engine: Vec<EngineSample>,
    /// Future producers (e.g. storage) use the same cumulative byte shape.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sources: BTreeMap<String, SourceSample>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceSample {
    pub incarnation: String,
    pub counters: Vec<EngineSample>,
}

/// A third producer supplies cumulative bytes keyed by device/port/peer. Sampling
/// is called only by the observer, never by an inference or verbs post path.
pub trait TrafficSource: Send + Sync {
    fn name(&self) -> &'static str;
    fn sample(&self) -> SourceSample;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Snapshot {
    pub hosts: BTreeMap<String, HostSample>,
}

#[derive(Debug, thiserror::Error)]
#[error("served fabric traffic probe")]
pub struct TrafficProbeOnly;

#[derive(Debug, thiserror::Error)]
pub enum TrafficError {
    #[error("fabric control I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("fabric control JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid fabric control reply")]
    InvalidReply,
}

/// One handle per QP, registered before serving. Two relaxed atomic adds on the
/// exchange path, no lookup, allocation, lock, logging or sysfs access.
#[derive(Debug)]
pub(crate) struct WireCounter {
    device: String,
    port: u32,
    peer: OnceLock<String>,
    local_gid: String,
    written: AtomicU64,
    receives_writes: AtomicBool,
    sent: AtomicU64,
    received: AtomicU64,
}
impl WireCounter {
    pub(crate) fn sent(&self, bytes: usize) { self.sent.fetch_add(bytes as u64, Ordering::Relaxed); }
    pub(crate) fn written(&self, bytes: usize) {
        self.sent(bytes);
        self.written.fetch_add(bytes as u64, Ordering::Relaxed);
    }
    pub(crate) fn receives_writes(&self) { self.receives_writes.store(true, Ordering::Relaxed); }
    pub(crate) fn received(&self, bytes: usize) { self.received.fetch_add(bytes as u64, Ordering::Relaxed); }
    pub(crate) fn peer(&self, peer: &str) { let _ = self.peer.set(peer.into()); }
    fn sample(&self) -> EngineSample {
        EngineSample { device: self.device.clone(), port: self.port, peer: self.peer.get().cloned().unwrap_or_default(),
            receives_writes: self.receives_writes.load(Ordering::Relaxed),
            local_gid: self.local_gid.clone(), written_sent_bytes: self.written.load(Ordering::Relaxed),
            sent_bytes: self.sent.load(Ordering::Relaxed), received_bytes: self.received.load(Ordering::Relaxed) }
    }
}

#[derive(Default)]
struct Registry {
    // Handles survive QP retirement: cumulative observations must never decrease.
    counters: Vec<Arc<WireCounter>>,
    peers: BTreeSet<SocketAddr>,
    sources: Vec<Arc<dyn TrafficSource>>,
}
fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

pub(crate) fn register(device: &str, port: u32, peer: &str, local_gid: String) -> Arc<WireCounter> {
    start();
    let counter = Arc::new(WireCounter { device: device.into(), port, peer: { let p = OnceLock::new(); if !peer.is_empty() { let _ = p.set(peer.into()); } p },
        local_gid, written: AtomicU64::new(0), receives_writes: AtomicBool::new(false), sent: AtomicU64::new(0), received: AtomicU64::new(0) });
    registry().lock().unwrap().counters.push(counter.clone());
    counter
}

/// Called during transport setup, not while dispatching requests.
pub fn observe_peers(peers: &[SocketAddr]) {
    registry().lock().unwrap().peers.extend(peers.iter().copied());
    start();
}
pub fn add_source(source: Arc<dyn TrafficSource>) {
    registry().lock().unwrap().sources.push(source);
    start();
}

fn latest() -> &'static Mutex<Snapshot> {
    static LATEST: OnceLock<Mutex<Snapshot>> = OnceLock::new();
    LATEST.get_or_init(|| Mutex::new(Snapshot::default()))
}

pub fn snapshot() -> Snapshot {
    start();
    latest().lock().map(|s| s.clone()).unwrap_or_default()
}
pub fn local() -> HostSample {
    snapshot().hosts.remove("coordinator").unwrap_or_default()
}

/// Read every port and every numeric counter; unsupported files simply disappear.
pub fn read_ports(root: &Path) -> Vec<PortSample> {
    let mut result = Vec::new();
    let Ok(devices) = std::fs::read_dir(root) else { return result };
    for device in devices.flatten() {
        let Ok(ports) = std::fs::read_dir(device.path().join("ports")) else { continue };
        for entry in ports.flatten() {
            let Some(port) = entry.file_name().to_str().and_then(|p| p.parse().ok()) else { continue };
            let path = entry.path();
            let mut counters = BTreeMap::new();
            for directory in ["counters", "hw_counters"] {
                let Ok(files) = std::fs::read_dir(path.join(directory)) else { continue };
                for file in files.flatten() {
                    let name = file.file_name().to_string_lossy().into_owned();
                    let Some(mut value) = std::fs::read_to_string(file.path()).ok()
                        .and_then(|s| s.trim().parse::<u64>().ok()) else { continue };
                    if matches!(name.as_str(), "port_xmit_data" | "port_rcv_data") { value = value.saturating_mul(4); }
                    // Hardware identity must not change when a standard read fails.
                    let key = if directory == "hw_counters" { format!("hw_counters/{name}") } else { name };
                    counters.insert(key, value);
                }
            }
            result.push(PortSample { device: device.file_name().to_string_lossy().into_owned(), port,
                state: std::fs::read_to_string(path.join("state")).ok().map(|s| s.trim().into()),
                link_gbps: std::fs::read_to_string(path.join("rate")).ok()
                    .and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok()).filter(|v| v.is_finite() && *v > 0.0),
                counters, idle_xmit_wait_ticks_per_s: None });
        }
    }
    result.sort_by(|a, b| (&a.device, a.port).cmp(&(&b.device, b.port)));
    result
}

fn read_peer(peer: SocketAddr) -> Result<HostSample, TrafficError> {
    use std::io::{BufRead, BufReader, Write};
    let timeout = Duration::from_millis(150);
    let mut stream = TcpStream::connect_timeout(&peer, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    stream.write_all(b"{\"message\":\"fabric_traffic\"}\n")?;
    let mut line = String::new();
    BufReader::new(std::io::Read::take(stream, 1 << 20)).read_line(&mut line)?;
    let value: serde_json::Value = serde_json::from_str(&line)?;
    if value["message"] != "fabric_traffic" { return Err(TrafficError::InvalidReply); }
    Ok(serde_json::from_value(value["sample"].clone())?)
}

/// Start once per serving process. Sysfs and network reads run on a niced OS
/// thread at 1 Hz. API handlers only clone cached observations.
pub fn background_priority() { cuteafd_ffi::background_thread_priority(); }

pub fn start() {
    static START: OnceLock<()> = OnceLock::new();
    START.get_or_init(|| {
        let _ = std::thread::Builder::new().name("fabric-observer".into()).spawn(|| {
            cuteafd_ffi::background_thread_priority();
            let clock = Instant::now();
            let incarnation = format!("{}-{}", std::process::id(), std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
            let mut previous: Option<HostSample> = None;
            let mut idle = BTreeMap::new();
            loop {
                let at = Instant::now();
                let (counters, peers, sources) = {
                    let state = registry().lock().unwrap();
                    (state.counters.clone(), state.peers.clone(), state.sources.clone())
                };
                let mut local = HostSample { incarnation: incarnation.clone(), sampled_s: 0.0,
                    ports: read_ports(Path::new("/sys/class/infiniband")),
                    engine: counters.iter().map(|counter| counter.sample()).collect(),
                    sources: sources.iter().map(|s| (s.name().into(), s.sample())).collect() };
                local.sampled_s = clock.elapsed().as_secs_f64();
                if let Some(before) = previous.as_ref() {
                    // Freeze the first quiet interval per port as its idle reference.
                    // If serving starts immediately, wait until engine bytes stop moving.
                    let total = |host: &HostSample| host.engine.iter()
                        .fold(0u64, |n, s| n.saturating_add(s.sent_bytes).saturating_add(s.received_bytes));
                    let quiet = total(&local) == total(before);
                    let dt = local.sampled_s - before.sampled_s;
                    for port in &mut local.ports {
                        let key = (port.device.clone(), port.port);
                        if quiet && dt > 0.0 {
                            if let Some(old) = before.ports.iter().find(|p| p.device == port.device && p.port == port.port) {
                                if let Some(delta) = port.counters.get("port_xmit_wait").zip(old.counters.get("port_xmit_wait"))
                                    .and_then(|(a,b)| a.checked_sub(*b)) {
                                    idle.entry(key.clone()).or_insert(delta as f64 / dt);
                                }
                            }
                        }
                        port.idle_xmit_wait_ticks_per_s = idle.get(&key).copied();
                    }
                }
                previous = Some(local.clone());
                // Publish local before remote reads: a peer probe must never wait on another probe.
                latest().lock().unwrap().hosts.insert("coordinator".into(), local);
                let remote: BTreeMap<_, _> = std::thread::scope(|scope| {
                    let jobs: Vec<_> = peers.into_iter().map(|peer| (peer, scope.spawn(move || read_peer(peer)))).collect();
                    jobs.into_iter().filter_map(|(peer, job)| job.join().ok()?.ok().map(|sample| (peer.to_string(), sample))).collect()
                });
                {
                    let mut state = latest().lock().unwrap();
                    state.hosts.retain(|name, _| name == "coordinator");
                    state.hosts.extend(remote);
                }
                std::thread::sleep(Duration::from_secs(1).saturating_sub(at.elapsed()));
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sysfs_words_missing_and_bad_counters() {
        let root = std::env::temp_dir().join(format!("traffic-test-{}", std::process::id()));
        let port = root.join("nic/ports/1");
        std::fs::create_dir_all(port.join("counters")).unwrap();
        std::fs::write(port.join("counters/port_xmit_data"), "123\n").unwrap();
        std::fs::write(port.join("counters/port_rcv_data"), "not supported").unwrap();
        std::fs::create_dir_all(port.join("hw_counters")).unwrap();
        std::fs::write(port.join("hw_counters/port_xmit_data"), "456").unwrap();
        std::fs::write(port.join("rate"), "200 Gb/sec (4X HDR)").unwrap();
        let ports = read_ports(&root);
        assert_eq!(ports[0].counters["port_xmit_data"], 492);
        assert_eq!(ports[0].counters["hw_counters/port_xmit_data"], 1824);
        assert!(!ports[0].counters.contains_key("port_rcv_data"));
        assert_eq!(ports[0].link_gbps, Some(200.0));
        std::fs::write(port.join("counters/port_xmit_data"), "unavailable").unwrap();
        std::fs::write(port.join("hw_counters/port_xmit_data"), "500").unwrap();
        let next = read_ports(&root);
        assert!(!next[0].counters.contains_key("port_xmit_data"));
        assert_eq!(next[0].counters["hw_counters/port_xmit_data"], 2000);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn wire_counter_is_cumulative_and_directional() {
        let counter = WireCounter { device: "nic".into(), port: 1, peer: OnceLock::from("peer".to_string()),
            local_gid: "local".into(), written: AtomicU64::new(0), receives_writes: AtomicBool::new(false), sent: AtomicU64::new(0), received: AtomicU64::new(0) };
        counter.sent(128); counter.sent(4096); counter.written(72); counter.received(64);
        assert_eq!(counter.sample().sent_bytes, 4296);
        assert_eq!(counter.sample().written_sent_bytes, 72);
        assert_eq!(counter.sample().received_bytes, 64);
    }
}
