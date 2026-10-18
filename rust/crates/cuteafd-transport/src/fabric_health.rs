//! Behavioral congestion-control evidence, evaluated only by off-path observers.
use crate::traffic::PortSample;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Retransmit recoveries as a share of packets moved: RoCE recovers a gap by
/// go-back-N, resending the window, and that starts to cost measurable goodput
/// around 0.5% of packets. Below it the fabric is healthy; above it, a warning.
pub const RETRANSMIT_WARN_PCT: f64 = 0.5;
/// A window that moved less than this on a port is too small to judge its
/// retransmit rate (a handful of events on a short phase reads as a big share).
pub const RETRANSMIT_MIN_GB: f64 = 2.0;
pub const HIGH_UTILIZATION: f64 = 0.30;
/// RoCE sequence and recovery counters, reported raw. They overlap: one gap
/// raises out_of_sequence at the receiver and packet_seq_err at the sender, and
/// roce_adp_retrans counts the resulting resend; rnr_nak_retry_err is
/// receiver-not-ready flow control, not a gap.
const RETRANSMIT: [&str; 5] = [
    "out_of_sequence",
    "packet_seq_err",
    "roce_adp_retrans",
    "local_ack_timeout_err",
    "rnr_nak_retry_err",
];

/// The RoCE (InfiniBand) MTU a port's Ethernet MTU carries: the largest IB MTU
/// that fits beside the RoCEv2 headers; 4096 when the netdev MTU is unknown.
pub fn roce_mtu(netdev_mtu: Option<u32>) -> u32 {
    let Some(mtu) = netdev_mtu else { return 4096 };
    [4096, 2048, 1024, 512, 256].into_iter().find(|ib| ib + 58 <= mtu).unwrap_or(256)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Ok,
    Warn,
    Idle,
    Unproven,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Health {
    pub verdict: Verdict,
    pub reasons: Vec<String>,
    pub seconds: Option<f64>,
    /// Decimal GB, tx+rx on this physical port; not unique fabric bytes.
    pub moved_gb: Option<f64>,
    pub utilization: Option<f64>,
    pub ecn_marked_packets: Option<u64>,
    pub cnp_handled: Option<u64>,
    pub ecn_active: Option<bool>,
    /// The raw sequence/recovery counters over the window (overlapping; see `RETRANSMIT`).
    pub retransmit_counters: BTreeMap<String, Option<u64>>,
    /// Go-back-N recoveries without double counting: the larger of
    /// out_of_sequence (receiver) and packet_seq_err (sender) for the same gaps,
    /// plus local_ack_timeout_err (sender timeouts, a separate trigger).
    pub retransmit_events: Option<u64>,
    /// `retransmit_events` as a percentage of packets moved (bytes ÷ `roce_mtu`).
    pub retransmit_pct: Option<f64>,
    pub retransmit_warn_pct: f64,
    /// False when the window moved under `RETRANSMIT_MIN_GB`: shown, never judged.
    pub retransmit_judged: bool,
    pub roce_mtu: u32,
    pub missing_or_reset: Vec<String>,
    pub pause: String,
}

fn counter(port: &PortSample, key: &str) -> Option<u64> {
    if matches!(key, "port_xmit_data" | "port_rcv_data") {
        return port.counters.get(key).copied();
    }
    port.counters
        .get(&format!("hw_counters/{key}"))
        .or_else(|| port.counters.get(key))
        .copied()
}

pub fn evaluate(before: &PortSample, after: &PortSample, seconds: f64) -> Health {
    let mut h = Health {
        retransmit_warn_pct: RETRANSMIT_WARN_PCT,
        roce_mtu: roce_mtu(after.mtu),
        pause: "not collected".into(),
        ..Default::default()
    };
    if !seconds.is_finite()
        || seconds <= 0.0
        || before.device != after.device
        || before.port != after.port
        || before.counter_epoch != after.counter_epoch
    {
        h.reasons
            .push("no continuous advancing observation window (gap/reset or changed port)".into());
        return h;
    }
    h.seconds = Some(seconds);
    let mut difference = |key: &str| {
        let value = counter(after, key)
            .zip(counter(before, key))
            .and_then(|(a, b)| a.checked_sub(b));
        if value.is_none() {
            h.missing_or_reset.push(key.into());
        }
        value
    };
    let tx = difference("port_xmit_data");
    let rx = difference("port_rcv_data");
    h.ecn_marked_packets = difference("np_ecn_marked_roce_packets");
    h.cnp_handled = difference("rp_cnp_handled");
    h.retransmit_counters = RETRANSMIT
        .into_iter()
        .map(|key| (key.into(), difference(key)))
        .collect();
    h.moved_gb = tx.zip(rx).map(|(tx, rx)| (tx as f64 + rx as f64) / 1e9);
    let count = |key: &str| h.retransmit_counters.get(key).copied().flatten();
    // One gap per sequence error pair, plus timeouts; never the overlapping sum.
    h.retransmit_events = count("out_of_sequence").zip(count("packet_seq_err"))
        .map(|(rx, tx)| rx.max(tx)).zip(count("local_ack_timeout_err")).map(|(gaps, timeouts)| gaps + timeouts);
    let packets = h.moved_gb.filter(|gb| *gb > 0.0).map(|gb| gb * 1e9 / f64::from(h.roce_mtu));
    h.retransmit_pct = h.retransmit_events.zip(packets).map(|(n, packets)| 100.0 * n as f64 / packets);
    h.retransmit_judged = h.moved_gb.is_some_and(|gb| gb >= RETRANSMIT_MIN_GB);
    // A changed link rate cannot describe one utilization envelope reliably.
    h.utilization = after
        .link_gbps
        .filter(|r| r.is_finite() && *r > 0.0 && before.link_gbps == Some(*r))
        .and_then(|r| Some(tx?.max(rx?) as f64 * 8.0 / seconds / 1e9 / r));
    h.ecn_active = h
        .ecn_marked_packets
        .zip(h.cnp_handled)
        .map(|(m, c)| m > 0 && c > 0);
    if h.retransmit_judged && h.retransmit_pct.is_some_and(|p| p > RETRANSMIT_WARN_PCT) {
        h.reasons.push(format!(
            "retransmits {} of packets exceed {} (go-back-N resends cost goodput; no data is lost)",
            pct(h.retransmit_pct), pct(Some(RETRANSMIT_WARN_PCT))
        ));
    }
    let loaded = h.utilization.is_some_and(|u| u > HIGH_UTILIZATION);
    if loaded && h.ecn_marked_packets == Some(0) {
        h.reasons.push("no ECN marking under load: switch queue ECN/WRED off, or the NICs aren't sending ECN-capable traffic".into());
    } else if loaded && h.ecn_marked_packets.is_some_and(|n| n > 0) && h.cnp_handled == Some(0) {
        h.reasons.push(
            "ECN marks observed under load but no CNPs handled; sender backoff not demonstrated"
                .into(),
        );
    }
    if !h.reasons.is_empty() {
        h.verdict = Verdict::Warn;
    } else if !h.missing_or_reset.is_empty()
        || h.utilization.is_none()
        || before.counter_epoch != after.counter_epoch
    {
        h.verdict = Verdict::Unknown;
        h.reasons.push(
            "missing/reset counters, observation gap or changed/unavailable link rate".into(),
        );
    } else if h.moved_gb == Some(0.0) {
        h.verdict = Verdict::Idle;
        h.reasons
            .push("idle port; congestion control not exercised".into());
    } else if h.ecn_active == Some(true) {
        h.verdict = Verdict::Ok;
        h.reasons
            .push("ECN marks and handled CNPs observed; retransmits below the warning rate".into());
    } else {
        h.verdict = Verdict::Unproven;
        h.reasons
            .push("low load; ECN/backoff not demonstrated in this window".into());
    }
    h.reasons.push(format!("retransmits {} of packets{} (warning above {})", pct(h.retransmit_pct),
        if h.retransmit_judged || h.moved_gb.is_none() { "" } else { ", too little traffic to judge" }, pct(Some(RETRANSMIT_WARN_PCT))));
    h
}

/// A percentage with two significant digits ("unknown" when absent).
pub fn pct(n: Option<f64>) -> String {
    match n {
        None => "unknown".into(),
        Some(v) if !v.is_finite() => "unknown".into(),
        Some(v) if v == 0.0 => "0%".into(),
        Some(v) if v < 1e-4 => "<0.0001%".into(),
        Some(v) => format!("{}%", format!("{v:.1e}").parse::<f64>().unwrap_or(v)),
    }
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Warn => "warn",
            Self::Idle => "idle",
            Self::Unproven => "unproven",
            Self::Unknown => "unknown",
        }
    }
}

impl Health {
    pub fn line(&self) -> String {
        format!(
            "{}: retransmits {} of packets; {}",
            self.verdict.label(),
            pct(self.retransmit_pct),
            self.reasons
                .first()
                .map(String::as_str)
                .unwrap_or("no evidence")
        )
    }
}

/// Conservative host aggregation: an idle rail never masks a loaded bad rail.
pub fn verdict<'a>(health: impl IntoIterator<Item = &'a Health>) -> Verdict {
    let states: Vec<_> = health.into_iter().map(|h| h.verdict).collect();
    for state in [
        Verdict::Warn,
        Verdict::Unknown,
        Verdict::Unproven,
        Verdict::Ok,
        Verdict::Idle,
    ] {
        if states.contains(&state) {
            return state;
        }
    }
    Verdict::Unknown
}

/// Since the start of a continuous observer lease, not since NIC boot. A missing
/// or decreasing counter stays unknown for this window, even if it recovers.
#[derive(Default)]
pub(crate) struct Window {
    started: Option<f64>,
    baseline: BTreeMap<(String, u32), PortSample>,
    previous: BTreeMap<(String, u32), PortSample>,
    epochs: BTreeMap<(String, u32), u64>,
}
impl Window {
    pub(crate) fn update(&mut self, sampled_s: f64, ports: &mut [PortSample]) -> f64 {
        let started = *self.started.get_or_insert(sampled_s);
        if self.previous.is_empty() && self.baseline.is_empty() && sampled_s == started {
            self.baseline = ports
                .iter()
                .map(|p| ((p.device.clone(), p.port), p.clone()))
                .collect();
        }
        for p in ports.iter_mut() {
            let key = (p.device.clone(), p.port);
            let epoch = self.epochs.entry(key.clone()).or_default();
            if let Some(previous) = self.previous.get(&key) {
                if previous.link_gbps != p.link_gbps
                    || previous.counters.keys().ne(p.counters.keys())
                    || previous
                        .counters
                        .iter()
                        .any(|(k, b)| p.counters.get(k).is_some_and(|a| a < b))
                {
                    *epoch = epoch.saturating_add(1);
                }
            } else if sampled_s != started {
                *epoch = epoch.saturating_add(1);
            }
            p.counter_epoch = *epoch;
            if let Some(base) = self.baseline.get_mut(&key) {
                if let Some(previous) = self.previous.get(&key) {
                    base.counters.retain(|k, _| {
                        previous
                            .counters
                            .get(k)
                            .zip(p.counters.get(k))
                            .is_some_and(|(b, a)| a >= b)
                    });
                    if previous.link_gbps != p.link_gbps {
                        base.link_gbps = None;
                    }
                }
                p.health_since_start = Some(evaluate(base, p, sampled_s - started));
            } else {
                p.health_since_start = Some(evaluate(p, p, 0.0));
            }
        }
        self.baseline
            .retain(|k, _| ports.iter().any(|p| (p.device.clone(), p.port) == *k));
        self.previous = ports
            .iter()
            .map(|p| ((p.device.clone(), p.port), p.clone()))
            .collect();
        started
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn port(bytes: u64, marks: u64, cnp: u64, gaps: u64) -> PortSample {
        let mut p = PortSample {
            device: "nic".into(),
            port: 1,
            link_gbps: Some(100.0),
            ..Default::default()
        };
        for (k, n) in [
            ("port_xmit_data", bytes),
            ("port_rcv_data", bytes),
            ("np_ecn_marked_roce_packets", marks),
            ("rp_cnp_handled", cnp),
        ] {
            p.counters.insert(k.into(), n);
        }
        for (i, k) in RETRANSMIT.into_iter().enumerate() {
            p.counters
                .insert(format!("hw_counters/{k}"), if i == 0 { gaps } else { 0 });
        }
        p
    }
    #[test]
    fn calibrated_post_ecn_and_pre_ecn() {
        let b = port(0, 0, 0, 0);
        let healthy = evaluate(&b, &port(25_000_000_000, 300_000, 4_000, 2_000), 10.0);
        assert_eq!(healthy.verdict, Verdict::Ok);
        // 2000 gaps over 50 GB at a 4 KB RoCE MTU: 0.016% of packets.
        assert_eq!(healthy.retransmit_events, Some(2_000));
        assert_eq!(healthy.pause, "not collected");
        assert!(healthy.line().contains("retransmits 0.016% of packets"), "{}", healthy.line());
        let bad = evaluate(&b, &port(25_000_000_000, 0, 4_000, 2_000_000), 2.0);
        assert_eq!(bad.verdict, Verdict::Warn);
        assert!(bad
            .reasons
            .iter()
            .any(|r| r.starts_with("no ECN marking under load:")));
    }
    #[test]
    fn boundaries_low_load_and_missing_are_not_fabricated_success() {
        let b = port(0, 0, 0, 0);
        assert_eq!(
            evaluate(&b, &port(1_000_000_000, 0, 0, 0), 10.0).verdict,
            Verdict::Unproven
        );
        assert_eq!(evaluate(&b, &b, 1.0).verdict, Verdict::Idle);
        // 2 GB moved (the minimum to judge) is 488,281 packets at 4 KB; 0.5% is 2441 gaps.
        assert_eq!(
            evaluate(&b, &port(1_000_000_000, 1, 1, 2_441), 1.0).verdict,
            Verdict::Ok
        );
        assert_eq!(
            evaluate(&b, &port(1_000_000_000, 1, 1, 2_442), 1.0).verdict,
            Verdict::Warn
        );
        // Under 2 GB moved: shown, never judged, however high the share.
        let small = evaluate(&b, &port(500_000_000, 1, 1, 100_000), 1.0);
        assert_eq!(small.verdict, Verdict::Ok);
        assert!(!small.retransmit_judged && small.retransmit_pct.unwrap() > 0.5);
        assert!(small.line().contains("too little traffic to judge") || small.reasons.iter().any(|r| r.contains("too little traffic")));
        let mut a = port(1_000_000_000, 1, 1, 0);
        a.counters.remove("hw_counters/packet_seq_err");
        assert_eq!(evaluate(&b, &a, 1.0).verdict, Verdict::Unknown);
        assert_eq!(evaluate(&b, &a, 0.0).verdict, Verdict::Unknown);
        assert_eq!(evaluate(&a, &b, 1.0).verdict, Verdict::Unknown);
        assert_eq!(
            evaluate(&b, &port(3_750_000_000, 0, 0, 0), 1.0).verdict,
            Verdict::Unproven
        );
        assert_eq!(
            evaluate(&b, &port(3_750_000_001, 0, 0, 0), 1.0).verdict,
            Verdict::Warn
        );
    }
    #[test]
    fn one_gap_counts_once_and_the_mtu_follows_the_netdev() {
        let b = port(0, 0, 0, 0);
        let mut a = port(10_000_000_000, 1, 1, 0);
        // The same 300 gaps seen at both ends, their resends, 20 timeouts and RNR flow control.
        for (key, n) in [("out_of_sequence", 300), ("packet_seq_err", 280), ("roce_adp_retrans", 300),
            ("local_ack_timeout_err", 20), ("rnr_nak_retry_err", 5_000)] {
            a.counters.insert(format!("hw_counters/{key}"), n);
        }
        let h = evaluate(&b, &a, 10.0);
        assert_eq!(h.retransmit_events, Some(320));
        assert_eq!(h.roce_mtu, 4096);
        assert_eq!(roce_mtu(Some(9000)), 4096);
        assert_eq!(roce_mtu(Some(1500)), 1024);
        assert_eq!(pct(h.retransmit_pct), "0.0066%");
        assert_eq!(pct(Some(0.019)), "0.019%");
        assert_eq!(pct(Some(12.34)), "12%");
    }
    #[test]
    fn reset_inside_window_stays_unknown_and_aggregation_is_conservative() {
        let mut w = Window::default();
        let mut p = vec![port(1_000_000_000, 10, 10, 10)];
        w.update(1.0, &mut p);
        p[0] = port(2_000_000_000, 20, 20, 20);
        w.update(2.0, &mut p);
        assert_eq!(
            p[0].health_since_start.as_ref().unwrap().verdict,
            Verdict::Ok
        );
        p[0] = port(1, 1, 1, 1);
        w.update(3.0, &mut p);
        p[0] = port(1000, 100, 100, 100);
        w.update(4.0, &mut p);
        assert_eq!(
            p[0].health_since_start.as_ref().unwrap().verdict,
            Verdict::Unknown
        );
        let states = [
            Health {
                verdict: Verdict::Idle,
                ..Default::default()
            },
            Health {
                verdict: Verdict::Warn,
                ..Default::default()
            },
        ];
        assert_eq!(verdict(&states), Verdict::Warn);
    }
    #[test]
    fn intermediate_reset_and_missing_port_do_not_bridge() {
        let mut w = Window::default();
        let mut p = vec![port(1_000_000_000, 10, 10, 10)];
        w.update(1.0, &mut p);
        let before = p[0].clone();
        p[0] = port(1, 1, 1, 1);
        w.update(2.0, &mut p);
        p[0] = port(2_000_000_000, 20, 20, 20);
        w.update(3.0, &mut p);
        assert_eq!(evaluate(&before, &p[0], 2.0).verdict, Verdict::Unknown);
        w.update(4.0, &mut []);
        p[0] = port(3_000_000_000, 30, 30, 30);
        w.update(5.0, &mut p);
        assert_eq!(
            p[0].health_since_start.as_ref().unwrap().verdict,
            Verdict::Unknown
        );
        assert_ne!(before.counter_epoch, p[0].counter_epoch);
    }
    #[test]
    fn hardware_alias_is_not_double_counted_and_bytes_stay_standard() {
        let mut b = port(0, 0, 0, 0);
        let mut a = port(1_000_000_000, 1, 1, 10);
        for p in [&mut b, &mut a] {
            p.counters.insert("out_of_sequence".into(), 900_000);
            p.counters.insert("hw_counters/port_xmit_data".into(), 99);
        }
        let h = evaluate(&b, &a, 1.0);
        assert_eq!(h.retransmit_counters["out_of_sequence"], Some(10));
        assert_eq!(h.retransmit_events, Some(10));
        assert_eq!(h.moved_gb, Some(2.0));
        let a = port(5_000_000_000, 1, 0, 0);
        assert_eq!(evaluate(&b, &a, 1.0).verdict, Verdict::Warn);
    }
}
