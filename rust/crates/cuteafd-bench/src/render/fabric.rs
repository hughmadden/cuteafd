//! The run's fabric as small multiples, the same component the dashboard's
//! `fabric.js` draws: one compact stacked-bar chart per used host/port (one
//! bar per timed phase, busier direction, engine / SparkNest / foreign /
//! other Gb/s, link capacity as a thin reference line), idle and spare ports
//! folded into one grey line.
use super::svg::{fit, Anchor, Doc, Font};
use super::Theme;
use crate::fabric::{Phase, Port};
use crate::report::Report;
use std::collections::BTreeMap;

/// Engine, SparkNest, foreign, other: the validated categorical steps of the page.
const COLORS: [&str; 4] = ["#3987e5", "#199e70", "#d95926", "#3a4760"];
const NAMES: [&str; 4] = ["engine", "SparkNest", "foreign", "other"];
const CELL_H: f64 = 136.0;

#[derive(Debug, Clone)]
struct Bar { phase: String, values: [f64; 4], nic: Option<f64> }

#[derive(Debug, Clone)]
struct PortView { name: String, port: String, link: Option<f64>, bars: Vec<Bar>, peak: f64, worst: Option<f64>,
    limit: f64, issue: Option<String>, used: bool }

fn rate(bytes: Option<u64>, seconds: f64) -> Option<f64> {
    (seconds > 0.0).then_some(())?;
    Some(bytes? as f64 * 8.0 / seconds / 1e9)
}

fn bar(phase: &str, p: &Port) -> Bar {
    let tx = rate(p.sent_bytes, p.seconds).or(p.xmit_gbps);
    let rx = rate(p.received_bytes, p.seconds).or(p.rcv_gbps);
    let sent = !matches!((tx, rx), (t, Some(r)) if t.is_none_or(|t| r > t));
    let pick = |a: Option<u64>, b: Option<u64>| rate(if sent { a } else { b }, p.seconds).unwrap_or(0.0);
    let nic = if sent { tx } else { rx };
    let engine = pick(p.engine_sent_bytes, p.engine_received_bytes);
    let storage = pick(p.sparknest_sent_bytes, p.sparknest_received_bytes);
    let foreign = pick(p.foreign_sent_bytes, p.foreign_received_bytes);
    let other = nic.map_or(0.0, |n| (n - engine - storage - foreign).max(0.0));
    Bar { phase: phase.into(), values: [engine, storage, foreign, other], nic }
}

/// Display name of a fabric host: its own hostname, the coordinator's host,
/// the Spark's /etc/hosts name, else "Spark N" by rank. Never an address.
pub fn host_name(report: &Report, port: &Port) -> String {
    if let Some(name) = &port.hostname { return name.clone(); }
    let hw = &report.server.hardware;
    if port.host == "coordinator" { return hw.host.clone().unwrap_or_else(|| "coordinator".into()); }
    let address = port.host.rsplit_once(':').map_or(port.host.as_str(), |(a, _)| a).trim_matches(['[', ']']);
    hw.sparks.iter().find(|s| s.address == address)
        .map(|s| s.name.clone().unwrap_or_else(|| format!("Spark {}", s.rank + 1)))
        .unwrap_or_else(|| "Spark".into())
}

/// A caveat line with every fabric host key (`10.55.0.2:19802`, `coordinator`)
/// replaced by the host's display name, so exports never print addresses.
pub fn named(report: &Report, text: &str) -> String {
    let mut out = text.to_string();
    let mut keys: Vec<&Port> = report.fabric.iter().flat_map(|p| &p.ports).collect();
    keys.sort_by_key(|p| std::cmp::Reverse(p.host.len()));
    keys.dedup_by(|a, b| a.host == b.host);
    for p in keys { out = out.replace(&p.host, &host_name(report, p)); }
    out
}

fn views(report: &Report, phases: &[Phase]) -> Vec<PortView> {
    let mut ports: BTreeMap<(i64, String, String), PortView> = BTreeMap::new();
    let rank = |p: &Port| if p.host == "coordinator" { -1 } else {
        let address = p.host.rsplit_once(':').map_or(p.host.as_str(), |(a, _)| a);
        report.server.hardware.sparks.iter().find(|s| s.address == address).map_or(999, |s| s.rank as i64)
    };
    for phase in phases.iter().filter(|p| p.timed) {
        for p in &phase.ports {
            let key = (rank(p), p.host.clone(), format!("{}/{}", p.device, p.port));
            let view = ports.entry(key.clone()).or_insert_with(|| PortView { name: host_name(report, p), port: key.2.clone(),
                link: p.link_gbps, bars: Vec::new(), peak: 0.0, worst: None, limit: 100.0, issue: None, used: false });
            let b = bar(&phase.name, p);
            view.peak = view.peak.max(b.nic.unwrap_or(0.0));
            view.used |= b.values[0] > 0.0 || b.nic.is_some_and(|n| n > 0.0);
            if let Some(h) = &p.health {
                if h.loss_warning_threshold_per_gb > 0.0 { view.limit = h.loss_warning_threshold_per_gb; }
                if let Some(loss) = h.loss_events_per_gb { view.worst = Some(view.worst.map_or(loss, |w: f64| w.max(loss))); }
            }
            let active = p.state.as_deref().is_some_and(|s| s.contains("ACTIVE"));
            if b.values[0] > 0.0 && !active { view.issue = Some("down".into()); }
            view.link = p.link_gbps.or(view.link);
            view.bars.push(b);
        }
    }
    let mut out: Vec<PortView> = ports.into_values().collect();
    for v in &mut out {
        if v.issue.is_none() { if let Some(w) = v.worst.filter(|w| *w > v.limit) { v.issue = Some(format!("loss {w:.0}/GB")); } }
    }
    out
}

fn gb(v: f64) -> String {
    if !v.is_finite() { "–".into() } else if v >= 100.0 { format!("{v:.0}") } else if v >= 10.0 { format!("{v:.1}") }
    else if v >= 0.1 { format!("{v:.2}") } else if v > 0.0 { format!("{v:.3}") } else { "0".into() }
}

/// Draws the section at (x, y), `w` wide; returns its height (0 without timed fabric phases).
pub fn draw(doc: &mut Doc, t: &Theme, report: &Report, x: f64, y: f64, w: f64) -> f64 {
    height_or_draw(Some(doc), t, report, x, y, w)
}

pub fn height(report: &Report, w: f64) -> f64 {
    height_or_draw(None, &super::theme::NORMAL, report, 0.0, 0.0, w)
}

fn height_or_draw(mut doc: Option<&mut Doc>, t: &Theme, report: &Report, x: f64, y0: f64, w: f64) -> f64 {
    let all = views(report, &report.fabric);
    if all.is_empty() { return 0.0; }
    let (shown, quiet): (Vec<_>, Vec<_>) = all.into_iter().partition(|v| v.used || v.issue.is_some());
    let gap = 10.0;
    let columns = shown.len().clamp(1, ((w + gap) / (150.0 + gap)).floor().max(1.0) as usize);
    let cell_w = ((w - gap * (columns as f64 - 1.0)) / columns as f64).min(260.0);
    let rows = shown.len().div_ceil(columns);
    let mut y = y0;
    let total = 24.0 + rows as f64 * (CELL_H + gap) + if quiet.is_empty() { 0.0 } else { 18.0 } + 6.0;
    let Some(doc) = doc.as_deref_mut() else { return total };
    doc.text(x, y + 13.0, Font::new(13.0, t.ink).weight(600), "Fabric during this run");
    let mut lx = x + 190.0;
    for (name, color) in NAMES.iter().zip(COLORS) {
        doc.rect(lx, y + 4.0, 9.0, 9.0, 2.0, color);
        doc.text(lx + 13.0, y + 12.5, Font::new(10.5, t.ink2), name);
        lx += 26.0 + super::svg::text_width(name, 10.5);
    }
    doc.text(x + w, y + 12.5, Font::new(10.0, t.muted).anchor(Anchor::End), "Gb/s, busier direction, one bar per phase · - - link");
    y += 24.0;
    for (i, v) in shown.iter().enumerate() {
        let cx = x + (i % columns) as f64 * (cell_w + gap);
        let cy = y + (i / columns) as f64 * (CELL_H + gap);
        cell(doc, t, v, cx, cy, cell_w);
    }
    y += rows as f64 * (CELL_H + gap);
    if !quiet.is_empty() {
        let mut counts: Vec<(String, usize)> = Vec::new();
        for v in &quiet {
            match counts.iter_mut().find(|(n, _)| *n == v.name) { Some(c) => c.1 += 1, None => counts.push((v.name.clone(), 1)) }
        }
        let line = counts.iter().map(|(n, c)| if *c > 1 { format!("{n} ×{c}") } else { n.clone() }).collect::<Vec<_>>().join(" · ");
        doc.text(x, y + 10.0, Font::new(10.5, t.muted), &fit(&format!("Idle or spare (normal): {line}"), 10.5, w));
    }
    total
}

fn cell(doc: &mut Doc, t: &Theme, v: &PortView, x: f64, y: f64, w: f64) {
    let border = if v.issue.is_some() { t.bad } else { t.line };
    doc.frame(x, y, w, CELL_H, 8.0, t.well, border);
    let pad = 9.0;
    doc.text(x + pad, y + 16.0, Font::new(11.0, t.ink).weight(600), &fit(&v.name, 11.0, w - 2.0 * pad - 80.0));
    doc.text(x + w - pad, y + 16.0, Font::new(11.0, t.ink).anchor(Anchor::End), &format!("peak {} Gb/s", gb(v.peak)));
    let verdict = v.issue.clone().unwrap_or_else(|| v.worst.map_or("OK".into(), |l| format!("loss {}/GB", gb(l))));
    let short = v.port.trim_start_matches("roce").to_string();
    doc.spans(x + pad, y + 30.0, 10.0, Anchor::Start, &[(&fit(&format!("{short} · "), 10.0, w * 0.6), t.muted, 400),
        (&verdict, if v.issue.is_some() { t.bad } else { t.muted }, if v.issue.is_some() { 600 } else { 400 })]);
    let (top, base) = (y + 40.0, y + CELL_H - 20.0);
    let (px, pw) = (x + pad, w - 2.0 * pad);
    let ceiling = v.peak.max(1e-6) * 1.15;
    let ceiling = match v.link { Some(l) if v.peak >= l * 0.25 => ceiling.max(l * 1.04), _ => ceiling };
    let sy = |val: f64| base - (val / ceiling).min(1.0) * (base - top);
    doc.line(px, base + 0.5, px + pw, base + 0.5, t.line2, 1.0);
    let slots = v.bars.len().max(1) as f64;
    let step = pw / slots;
    let bw = (step * 0.78).clamp(1.0, 22.0);
    for (i, b) in v.bars.iter().enumerate() {
        let bx = px + i as f64 * step + (step - bw) / 2.0;
        doc.titled(&format!("{} · {}: engine {} · SparkNest {} · foreign {} · other {} · NIC {} Gb/s", v.name, b.phase,
            gb(b.values[0]), gb(b.values[1]), gb(b.values[2]), gb(b.values[3]), b.nic.map_or("–".into(), gb)));
        let mut acc = 0.0;
        for (value, color) in b.values.iter().zip(COLORS) {
            if *value <= 0.0 { continue; }
            let (y0, y1) = (sy(acc), sy(acc + value));
            let h = (y0 - y1 - if acc > 0.0 { 1.0 } else { 0.0 }).max(0.0);
            if h > 0.0 { doc.rect(bx, y1, bw, h, (bw / 3.0).min(1.5), color); }
            acc += value;
        }
        doc.rect_attrs(bx - (step - bw) / 2.0, top, step, base - top, 0.0, r#"fill="transparent""#);
        doc.end();
    }
    if let Some(link) = v.link.filter(|l| *l <= ceiling) {
        let ly = sy(link);
        doc.raw(&format!(r#"<line x1="{px}" x2="{}" y1="{ly}" y2="{ly}" stroke="{}" stroke-width="1" stroke-dasharray="3 3" opacity="0.7"/>"#,
            px + pw, t.ink2));
    }
    doc.text(px, y + CELL_H - 6.0, Font::new(9.5, t.muted), &format!("{} phase{}", v.bars.len(), if v.bars.len() == 1 { "" } else { "s" }));
    if let Some(link) = v.link {
        doc.text(px + pw, y + CELL_H - 6.0, Font::new(9.5, t.muted).anchor(Anchor::End), &format!("link {} Gb/s", gb(link)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn port(host: &str, device: &str, engine: u64, nic: u64, state: &str) -> Port {
        Port { host: host.into(), device: device.into(), port: 1, state: Some(state.into()), link_gbps: Some(200.0),
            seconds: 1.0, sent_bytes: Some(nic), received_bytes: Some(0), engine_sent_bytes: Some(engine),
            engine_received_bytes: Some(0), sparknest_sent_bytes: Some(0), sparknest_received_bytes: Some(0),
            foreign_sent_bytes: Some(0), foreign_received_bytes: Some(0), ..Default::default() }
    }

    #[test]
    fn used_ports_chart_by_host_name_and_idle_ports_fold_into_one_line() {
        let mut report = crate::sample::report(false);
        report.server.hardware.host = Some("raptor".into());
        report.server.hardware.sparks = vec![crate::report::Spark { address: "10.55.0.1".into(), name: Some("ostrich".into()), rank: 0 }];
        report.fabric = vec![Phase { name: "decode/c1".into(), timed: true, ports: vec![
            port("coordinator", "mlx5_0", 5_000_000_000, 6_000_000_000, "4: ACTIVE"),
            port("10.55.0.1:19802", "rocep1s0f0", 2_000_000_000, 2_500_000_000, "4: ACTIVE"),
            port("10.55.0.1:19802", "rocep1s0f1", 0, 0, "1: DOWN"),
        ], ..Default::default() }];
        let mut doc = Doc::new(1000.0, 400.0);
        let h = draw(&mut doc, &super::super::theme::NORMAL, &report, 0.0, 0.0, 1000.0);
        assert_eq!(h, height(&report, 1000.0));
        assert!(h < 200.0, "one row of small multiples: {h}");
        let svg = doc.finish();
        for text in ["raptor", "ostrich", "peak 48.0 Gb/s", "Idle or spare (normal): ostrich", "link 200 Gb/s"] {
            assert!(svg.contains(text), "{text}");
        }
        assert!(!svg.contains("10.55.0.1"), "never an address");
    }
}
