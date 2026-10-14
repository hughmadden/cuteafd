//! Fabric deltas share the NIC observation clock on each host. A missing,
//! reset, restarted or stale counter is unknown, never invented as zero.
use cuteafd_transport::traffic::{EngineSample, HostSample, Snapshot};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Phase {
    pub name: String,
    pub timed: bool,
    pub ports: Vec<Port>,
    pub assessments: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Port {
    pub host: String,
    pub device: String,
    pub port: u32,
    pub state: Option<String>,
    pub link_gbps: Option<f64>,
    pub seconds: f64,
    /// Actual host-local observation window; phase edges are sampled at 1 Hz.
    pub sampled_start_s: f64,
    pub sampled_end_s: f64,
    pub sent_bytes: Option<u64>,
    pub received_bytes: Option<u64>,
    pub engine_sent_bytes: Option<u64>,
    pub engine_received_bytes: Option<u64>,
    pub foreign_sent_bytes: Option<u64>,
    pub foreign_received_bytes: Option<u64>,
    pub xmit_gbps: Option<f64>,
    pub rcv_gbps: Option<f64>,
    /// Full duplex: max(tx, rx) / link rate, not (tx + rx) / link rate.
    pub utilization: Option<f64>,
    pub counters: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xmit_wait_ticks_per_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idle_xmit_wait_ticks_per_s: Option<f64>,
    pub peers: Vec<EngineSample>,
}

fn delta(before: Option<&u64>, after: Option<&u64>) -> Option<u64> {
    after?.checked_sub(*before?)
}
fn engine(host: &HostSample, device: &str, port: u32) -> BTreeMap<String, (u64, u64)> {
    let mut peers = BTreeMap::new();
    for sample in host.engine.iter().filter(|s| s.device == device && s.port == port) {
        let pair = peers.entry(sample.peer.clone()).or_insert((0u64, 0u64));
        pair.0 = pair.0.saturating_add(sample.sent_bytes);
        pair.1 = pair.1.saturating_add(sample.received_bytes);
    }
    peers
}
fn source_delta(before: &HostSample, after: &HostSample, device: &str, port: u32) -> Option<(u64, u64)> {
    if before.sources.keys().ne(after.sources.keys()) { return None; }
    let mut total = (0u64, 0u64);
    for (name, a) in &after.sources {
        let b = before.sources.get(name)?;
        if a.incarnation != b.incarnation { return None; }
        let totals = |source: &cuteafd_transport::traffic::SourceSample| {
            let mut peers = BTreeMap::new();
            for sample in source.counters.iter().filter(|s| s.device == device && s.port == port) {
                let pair = peers.entry(sample.peer.clone()).or_insert((0u64, 0u64));
                pair.0 = pair.0.saturating_add(sample.sent_bytes);
                pair.1 = pair.1.saturating_add(sample.received_bytes);
            }
            peers
        };
        let ea = totals(a); let eb = totals(b);
        if eb.keys().any(|key| !ea.contains_key(key)) { return None; }
        for (peer, (at, ar)) in ea {
            let (bt, br) = eb.get(&peer).copied().unwrap_or_default();
            total.0 = total.0.saturating_add(at.checked_sub(bt)?);
            total.1 = total.1.saturating_add(ar.checked_sub(br)?);
        }
    }
    Some(total)
}

pub fn interval(name: &str, timed: bool, before: &Snapshot, after: &Snapshot) -> Phase {
    let mut phase = Phase { name: name.into(), timed, ..Default::default() };
    for (host, a) in &after.hosts {
        let Some(b) = before.hosts.get(host) else { continue };
        let seconds = a.sampled_s - b.sampled_s;
        if a.incarnation != b.incarnation || seconds <= 0.0 || !seconds.is_finite() { continue; }
        for port in &a.ports {
            let Some(old) = b.ports.iter().find(|p| p.device == port.device && p.port == port.port) else { continue };
            let counters: BTreeMap<_, _> = port.counters.iter().filter_map(|(key, value)|
                delta(old.counters.get(key), Some(value)).map(|d| (key.clone(), d))).collect();
            // A NIC counter reset invalidates byte attribution for this interval.
            let sent_bytes = counters.get("port_xmit_data").copied();
            let received_bytes = counters.get("port_rcv_data").copied();
            let eb = engine(b, &port.device, port.port);
            let ea = engine(a, &port.device, port.port);
            let mut peers = Vec::new();
            let ea_len = ea.len();
            for (peer, (tx, rx)) in ea {
                let (old_tx, old_rx) = eb.get(&peer).copied().unwrap_or_default();
                if let (Some(sent_bytes), Some(received_bytes)) = (tx.checked_sub(old_tx), rx.checked_sub(old_rx)) {
                    peers.push(EngineSample { peer, device: port.device.clone(), port: port.port, sent_bytes, received_bytes, ..Default::default() });
                }
            }
            let engine_sent_bytes = (peers.len() == ea_len && eb.keys().all(|peer| peers.iter().any(|p| &p.peer == peer)))
                .then(|| peers.iter().fold(0u64, |n, p| n.saturating_add(p.sent_bytes)));
            // One-sided writes have no receive CQE. Difference each actual
            // sender at both edges, not its lifetime total when a QP appears.
            let written_delta = || -> Option<BTreeMap<String, u64>> {
                let pairs: BTreeSet<_> = a.engine.iter().filter(|s| s.device == port.device
                    && s.port == port.port && s.receives_writes)
                    .map(|s| (s.local_gid.as_str(), s.peer.as_str())).collect();
                let mut bytes = BTreeMap::<String, u64>::new();
                for (local, peer) in pairs {
                    if local.is_empty() || peer.is_empty() { return None; }
                    let (name, old) = before.hosts.iter().find(|(_, host)| host.engine.iter()
                        .any(|s| s.local_gid == peer && s.peer == local))?;
                    let new = after.hosts.get(name)?;
                    if old.incarnation != new.incarnation || new.sampled_s <= old.sampled_s
                        || !new.engine.iter().any(|s| s.local_gid == peer && s.peer == local) { return None; }
                    let total = |host: &HostSample| host.engine.iter()
                        .filter(|s| s.local_gid == peer && s.peer == local)
                        .fold(0u64, |n, s| n.saturating_add(s.written_sent_bytes));
                    let entry = bytes.entry(peer.into()).or_default();
                    *entry = entry.saturating_add(total(new).checked_sub(total(old))?);
                }
                Some(bytes)
            };
            let engine_received_bytes = engine_sent_bytes.and_then(|_| {
                let written = written_delta()?;
                for peer in &mut peers {
                    peer.received_bytes = peer.received_bytes.saturating_add(written.get(&peer.peer).copied().unwrap_or(0));
                }
                Some(peers.iter().fold(0u64, |n, p| n.saturating_add(p.received_bytes)))
            });
            let sources = source_delta(b, a, &port.device, port.port);
            let rate = |bytes: u64| bytes as f64 * 8.0 / seconds / 1e9;
            let xmit_gbps = sent_bytes.map(rate);
            let rcv_gbps = received_bytes.map(rate);
            let utilization = port.link_gbps.filter(|r| *r > 0.0).and_then(|link|
                Some(xmit_gbps?.max(rcv_gbps?) / link));
            let foreign_sent_bytes = sent_bytes.zip(sources).zip(engine_sent_bytes).map(|((n, (tx, _)), engine)| n.saturating_sub(engine).saturating_sub(tx));
            let foreign_received_bytes = received_bytes.zip(sources).zip(engine_received_bytes).map(|((n, (_, rx)), engine)| n.saturating_sub(engine).saturating_sub(rx));
            let label = format!("{host} {}/{}", port.device, port.port);
            // Protocol bytes exclude Ethernet/IP/RDMA overhead. Require both
            // 5% of NIC traffic and 1 Gb/s to avoid calling headers foreign work.
            let foreign = foreign_sent_bytes.unwrap_or(0).saturating_add(foreign_received_bytes.unwrap_or(0));
            let nic = sent_bytes.unwrap_or(0).saturating_add(received_bytes.unwrap_or(0));
            if timed && rate(foreign) > 1.0 && foreign as f64 > nic as f64 * 0.05 {
                phase.assessments.push(format!("foreign traffic: {label} {:.2} Gb/s (>5% and >1 Gb/s)", rate(foreign)));
            }
            let signal = |key: &str| counters.get(key).is_some_and(|n| *n > 0);
            let wait_rate = counters.get("port_xmit_wait").map(|ticks| *ticks as f64 / seconds);
            let rising_wait = wait_rate.zip(port.idle_xmit_wait_ticks_per_s)
                .is_some_and(|(rate, idle)| rate > idle + (idle * 0.1).max(1.0));
            if timed && rising_wait {
                phase.assessments.push(format!("xmit_wait rising above idle: {label}"));
            }
            if utilization.is_some_and(|u| u >= 0.7)
                && (rising_wait || ["np_cnp_sent", "rp_cnp_handled", "np_ecn_marked_roce_packets"].iter().any(|key| signal(key))) {
                phase.assessments.push(format!("congestion at high utilization: {label}"));
            }
            if counters.iter().any(|(key, n)| *n > 0 && (key.contains("retrans")
                || matches!(key.as_str(), "out_of_sequence" | "packet_seq_err" | "local_ack_timeout_err"))) {
                phase.assessments.push(format!("retransmit/sequence signal: {label}"));
            }
            phase.ports.push(Port { host: host.clone(), device: port.device.clone(), port: port.port,
                sampled_start_s: b.sampled_s, sampled_end_s: a.sampled_s, state: port.state.clone(), link_gbps: port.link_gbps, seconds, sent_bytes, received_bytes,
                engine_sent_bytes, engine_received_bytes, foreign_sent_bytes, foreign_received_bytes,
                xmit_gbps, rcv_gbps, utilization, counters, xmit_wait_ticks_per_s: wait_rate,
                idle_xmit_wait_ticks_per_s: port.idle_xmit_wait_ticks_per_s, peers });
        }
    }
    phase
}

pub fn header(phases: &[Phase]) -> Option<String> {
    let ports: Vec<_> = phases.iter().flat_map(|phase| &phase.ports).collect();
    let used = |p: &Port| p.engine_sent_bytes.is_some_and(|n| n > 0) || p.engine_received_bytes.is_some_and(|n| n > 0);
    let spark = ports.iter().filter(|p| used(p)).find(|p| p.host != "coordinator").map(|p| p.host.as_str())
        .or_else(|| ports.iter().filter(|p| used(p)).next().map(|p| p.host.as_str()))?;
    let rails: BTreeMap<_, _> = ports.iter().filter(|p| p.host == spark
        && p.state.as_deref().is_some_and(|s| s.contains("ACTIVE"))
        && (p.engine_sent_bytes.is_some_and(|n| n > 0) || p.engine_received_bytes.is_some_and(|n| n > 0)))
        .filter_map(|p| Some(((p.device.as_str(), p.port), p.link_gbps?))).collect();
    if rails.is_empty() { return None; }
    let rates: Vec<_> = rails.values().map(|rate| format!("{rate:.0}")).collect();
    let links = if rates.windows(2).all(|w| w[0] == w[1]) { format!("{} x {} Gb/s", rates.len(), rates[0]) }
        else { format!("{} rails [{}] Gb/s", rates.len(), rates.join(" + ")) };
    let peak = ports.iter().filter_map(|p| p.utilization).reduce(f64::max);
    Some(match peak {
        Some(peak) => format!("RoCE {links} · peak {:.1}%", peak * 100.0),
        None => format!("RoCE {links}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_transport::traffic::PortSample;
    fn snapshot(time: f64, nic: u64, engine: u64) -> Snapshot {
        Snapshot { hosts: BTreeMap::from([("spark".into(), HostSample {
            incarnation: "one".into(), sampled_s: time,
            ports: vec![PortSample { device: "nic".into(), port: 1, link_gbps: Some(100.0), state: Some("4: ACTIVE".into()),
                counters: BTreeMap::from([("port_xmit_data".into(), nic), ("port_rcv_data".into(), nic),
                    ("rp_cnp_handled".into(), nic)]), idle_xmit_wait_ticks_per_s: None }],
            engine: vec![EngineSample { device: "nic".into(), port: 1, peer: "peer".into(),
                sent_bytes: engine, received_bytes: engine, ..Default::default() }], ..Default::default()
        })]) }
    }
    #[test]
    fn directional_utilization_and_foreign_congestion() {
        let phase = interval("decode", true, &snapshot(1.0, 0, 0), &snapshot(2.0, 10_000_000_000, 9_000_000_000));
        let port = &phase.ports[0];
        assert_eq!(port.utilization, Some(0.8));
        assert_eq!(port.foreign_sent_bytes, Some(1_000_000_000));
        assert_eq!(phase.assessments.len(), 2);
        assert_eq!(header(&[phase]).unwrap(), "RoCE 1 x 100 Gb/s · peak 80.0%");
    }
    #[test]
    fn source_restart_and_disappearance_leave_foreign_unknown() {
        let mut before = snapshot(1.0, 0, 0);
        let mut after = snapshot(2.0, 1000, 100);
        let source = cuteafd_transport::traffic::SourceSample { incarnation: "a".into(),
            counters: vec![EngineSample { device: "nic".into(), port: 1, sent_bytes: 100,
                received_bytes: 100, ..Default::default() }] };
        before.hosts.get_mut("spark").unwrap().sources.insert("storage".into(), source.clone());
        assert_eq!(interval("x", true, &before, &after).ports[0].foreign_sent_bytes, None);
        let mut next = source.clone(); next.incarnation = "b".into();
        after.hosts.get_mut("spark").unwrap().sources.insert("storage".into(), next);
        assert_eq!(interval("x", true, &before, &after).ports[0].foreign_sent_bytes, None);
        after.hosts.get_mut("spark").unwrap().sources.insert("storage".into(), source);
        assert_eq!(interval("x", true, &before, &after).ports[0].foreign_sent_bytes, Some(900));
    }
    #[test]
    fn one_sided_receives_come_from_peer_posts() {
        let mut before = snapshot(1.0, 0, 0);
        let mut after = snapshot(2.0, 1000, 100);
        for snapshot in [&mut before, &mut after] {
            snapshot.hosts.get_mut("spark").unwrap().engine[0].local_gid = "local".into();
            snapshot.hosts.get_mut("spark").unwrap().engine[0].receives_writes = true;
            let bytes = if snapshot.hosts["spark"].sampled_s > 1.0 { 700 } else { 0 };
            snapshot.hosts.insert("coordinator".into(), HostSample { incarnation: "peer".into(),
                sampled_s: snapshot.hosts["spark"].sampled_s,
                engine: vec![EngineSample { local_gid: "peer".into(), peer: "local".into(),
                    written_sent_bytes: bytes, sent_bytes: bytes, ..Default::default() }], ..Default::default() });
        }
        let phase = interval("write", true, &before, &after);
        assert_eq!(phase.ports[0].engine_received_bytes, Some(800));
        assert_eq!(phase.ports[0].peers[0].received_bytes, 800);
        assert_eq!(phase.ports[0].foreign_received_bytes, Some(200));
        before.hosts.remove("coordinator");
        assert_eq!(interval("missing", true, &before, &after).ports[0].engine_received_bytes, None);
        after.hosts.remove("coordinator");
        assert_eq!(interval("missing", true, &before, &after).ports[0].foreign_received_bytes, None);
    }
    #[test]
    fn stale_reset_restart_and_missing_are_not_traffic() {
        let before = snapshot(1.0, 500, 500);
        assert!(interval("x", true, &before, &before).ports.is_empty());
        let mut after = snapshot(2.0, 100, 600);
        let phase = interval("x", true, &before, &after);
        assert_eq!(phase.ports[0].sent_bytes, None);
        after.hosts.get_mut("spark").unwrap().incarnation = "two".into();
        assert!(interval("x", true, &before, &after).ports.is_empty());
    }
}
