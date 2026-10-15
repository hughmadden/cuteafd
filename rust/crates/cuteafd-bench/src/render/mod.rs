//! Server-side rendering of panels, reports and the share card as
//! self-contained SVG (and PNG through resvg). The dashboard shows these same
//! SVGs, so a chart looks the same in the browser, in an export and on GitHub.
pub mod svg;
pub mod theme;
pub mod bodies;
pub mod report;
pub mod card;
pub mod charts;
pub mod layout;
pub mod panels;
pub mod png;

pub use theme::Theme;

/// Thousands separators: 2415.3 -> "2,415".
pub fn grouped(value: f64) -> String {
    let rounded = value.round() as i64;
    let digits = rounded.abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if rounded < 0 { format!("-{out}") } else { out }
}

/// A rate with sensible precision: 187.3, 41.06, 2,415.
pub fn rate(value: f64) -> String {
    if !value.is_finite() || value <= 0.0 {
        "—".into()
    } else if value >= 1000.0 {
        grouped(value)
    } else if value >= 100.0 {
        format!("{value:.0}")
    } else if value >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

pub fn seconds(value: f64) -> String {
    if !value.is_finite() || value < 0.0 {
        "—".into()
    } else if value < 1.0 {
        format!("{:.0} ms", value * 1e3)
    } else if value < 100.0 {
        format!("{value:.2} s")
    } else if value < 3600.0 {
        format!("{}m {:02}s", (value / 60.0) as u64, (value % 60.0) as u64)
    } else {
        format!("{}h {:02}m", (value / 3600.0) as u64, (value % 3600.0 / 60.0) as u64)
    }
}

/// `2026-10-02` from an RFC 3339 time.
pub fn date(rfc3339: &str) -> &str {
    rfc3339.get(..10).unwrap_or(rfc3339)
}

#[cfg(test)]
mod tests {
    #[test]
    fn number_formats() {
        assert_eq!(super::grouped(2415.3), "2,415");
        assert_eq!(super::grouped(1234567.0), "1,234,567");
        assert_eq!(super::rate(187.34), "187");
        assert_eq!(super::rate(41.06), "41.1");
        assert_eq!(super::rate(2415.3), "2,415");
        assert_eq!(super::seconds(0.4), "400 ms");
        assert_eq!(super::seconds(125.0), "2m 05s");
    }
    #[test]
    fn health_footer_keeps_rates_and_pause_in_card_and_report() {
        let mut report = crate::sample::full_report();
        let h = cuteafd_transport::fabric_health::Health {
            verdict: cuteafd_transport::fabric_health::Verdict::Ok,
            loss_events_per_gb: Some(60.0), adapter_retransmits_per_gb: Some(1.0),
            sequence_events_per_gb: Some(59.0), reasons: vec!["ECN and CNP observed".into()],
            ..Default::default()
        };
        report.fabric = vec![crate::fabric::Phase { name: "decode".into(), timed: true,
            ports: vec![crate::fabric::Port { host: "coordinator".into(), device: "nic".into(), port: 1,
                health: Some(h.clone()), health_since_start: Some(h), mtu: Some(9000), ..Default::default() }], ..Default::default() }];
        let card = super::card::card_svg(&report);
        assert!(card.contains("Fabric OK: peak loss 60.0 / retransmit 1.0"));
        assert!(card.contains("pause not collected"));
        let report = super::report::report_svg(&report);
        assert!(report.contains("loss 60.0 / sequence 59.0 / retransmit 1.0"));
        assert!(report.contains("Since observation start") && report.contains("MTU 9000"));
    }

}
