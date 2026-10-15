//! Behavioral congestion-control evidence, evaluated only by off-path observers.
use crate::traffic::PortSample;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const LOSS_EVENTS_PER_GB_WARN: f64 = 100.0;
pub const HIGH_UTILIZATION: f64 = 0.30;
const LOSS: [&str; 5] = [
    "out_of_sequence",
    "packet_seq_err",
    "roce_adp_retrans",
    "local_ack_timeout_err",
    "rnr_nak_retry_err",
];

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
    /// The events overlap and are not lost packets or retransmitted bytes.
    pub loss_events: BTreeMap<String, Option<u64>>,
    pub loss_events_per_gb: Option<f64>,
    pub loss_events_per_gb_by_counter: BTreeMap<String, Option<f64>>,
    pub sequence_events_per_gb: Option<f64>,
    pub adapter_retransmits_per_gb: Option<f64>,
    pub loss_warning_threshold_per_gb: f64,
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
        loss_warning_threshold_per_gb: LOSS_EVENTS_PER_GB_WARN,
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
    h.loss_events = LOSS
        .into_iter()
        .map(|key| (key.into(), difference(key)))
        .collect();
    h.moved_gb = tx.zip(rx).map(|(tx, rx)| (tx as f64 + rx as f64) / 1e9);
    h.sequence_events_per_gb = h.moved_gb.filter(|gb| *gb > 0.0).and_then(|gb| {
        Some(
            (h.loss_events.get("out_of_sequence")?.as_ref().copied()? as f64
                + h.loss_events.get("packet_seq_err")?.as_ref().copied()? as f64)
                / gb,
        )
    });
    h.adapter_retransmits_per_gb = h
        .moved_gb
        .filter(|gb| *gb > 0.0)
        .and_then(|gb| Some(h.loss_events.get("roce_adp_retrans")?.as_ref().copied()? as f64 / gb));
    h.loss_events_per_gb_by_counter = h
        .loss_events
        .iter()
        .map(|(key, n)| {
            (
                key.clone(),
                h.moved_gb
                    .filter(|gb| *gb > 0.0)
                    .and_then(|gb| Some((*n)? as f64 / gb)),
            )
        })
        .collect();
    // A changed link rate cannot describe one utilization envelope reliably.
    h.utilization = after
        .link_gbps
        .filter(|r| r.is_finite() && *r > 0.0 && before.link_gbps == Some(*r))
        .and_then(|r| Some(tx?.max(rx?) as f64 * 8.0 / seconds / 1e9 / r));
    h.ecn_active = h
        .ecn_marked_packets
        .zip(h.cnp_handled)
        .map(|(m, c)| m > 0 && c > 0);
    h.loss_events_per_gb = h.moved_gb.filter(|gb| *gb > 0.0).and_then(|gb| {
        h.loss_events
            .values()
            .try_fold(0.0, |sum, n| Some(sum + (*n)? as f64))
            .map(|n| n / gb)
    });
    if h.loss_events_per_gb
        .is_some_and(|n| n > LOSS_EVENTS_PER_GB_WARN)
    {
        h.reasons.push(format!(
            "loss signals {:.1} events/GB exceed {:.0}/GB (overlapping events, not lost bytes)",
            h.loss_events_per_gb.unwrap(),
            LOSS_EVENTS_PER_GB_WARN
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
            .push("ECN marks and handled CNPs observed; loss below warning threshold".into());
    } else {
        h.verdict = Verdict::Unproven;
        h.reasons
            .push("low load; ECN/backoff not demonstrated in this window".into());
    }
    h.reasons.push(format!("loss {} events/GB; sequence/reordering {}; adapter retransmits {} (warning >{LOSS_EVENTS_PER_GB_WARN:.0}/GB combined)",
        rate(h.loss_events_per_gb), rate(h.sequence_events_per_gb), rate(h.adapter_retransmits_per_gb)));
    h
}

fn rate(n: Option<f64>) -> String {
    n.map(|n| format!("{n:.1}"))
        .unwrap_or_else(|| "unknown".into())
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
            "{}: loss {} / sequence {} / retransmit {} events/GB; {}",
            self.verdict.label(),
            rate(self.loss_events_per_gb),
            rate(self.sequence_events_per_gb),
            rate(self.adapter_retransmits_per_gb),
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
    fn port(bytes: u64, marks: u64, cnp: u64, loss: u64) -> PortSample {
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
        for (i, k) in LOSS.into_iter().enumerate() {
            p.counters
                .insert(format!("hw_counters/{k}"), if i == 0 { loss } else { 0 });
        }
        p
    }
    #[test]
    fn calibrated_post_ecn_and_pre_ecn() {
        let b = port(0, 0, 0, 0);
        let healthy = evaluate(&b, &port(25_000_000_000, 300_000, 4_000, 2_000), 10.0);
        assert_eq!(healthy.verdict, Verdict::Ok);
        assert_eq!(healthy.loss_events_per_gb, Some(40.0));
        assert_eq!(healthy.pause, "not collected");
        assert_eq!(healthy.sequence_events_per_gb, Some(40.0));
        assert_eq!(healthy.adapter_retransmits_per_gb, Some(0.0));
        assert!(healthy
            .line()
            .contains("loss 40.0 / sequence 40.0 / retransmit 0.0 events/GB"));
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
        assert_eq!(
            evaluate(&b, &port(1_000_000_000, 1, 1, 200), 1.0).verdict,
            Verdict::Ok
        );
        assert_eq!(
            evaluate(&b, &port(1_000_000_000, 1, 1, 201), 1.0).verdict,
            Verdict::Warn
        );
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
        assert_eq!(h.loss_events_per_gb, Some(5.0));
        assert_eq!(
            h.loss_events_per_gb_by_counter["out_of_sequence"],
            Some(5.0)
        );
        assert_eq!(h.moved_gb, Some(2.0));
        let a = port(5_000_000_000, 1, 0, 0);
        assert_eq!(evaluate(&b, &a, 1.0).verdict, Verdict::Warn);
    }
}
