//! Chart bodies of the measurement panels. Each reads its panel's newest
//! record (the running pass's partial one while it runs).
use super::bodies::content_color;
use super::charts::{hbars, legend, short, Plot, Scale};
use super::svg::{Anchor, Doc, Font};
use super::{rate, seconds, Theme};
use crate::report::PanelResult;
use serde_json::Value;

fn num(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

fn empty(doc: &mut Doc, t: &Theme, text: &str) -> f64 {
    doc.text(0.0, 16.0, Font::new(12.0, t.ink2), text);
    24.0
}

/// Renders panel `id`'s chart; None for panels without a chart body here.
pub fn body(doc: &mut Doc, t: &Theme, id: &str, panel: &PanelResult, w: f64) -> Option<f64> {
    let latest = panel.latest()?;
    Some(match id {
        "decode_content" => decode_content(doc, t, latest, w),
        "concurrency" => concurrency(doc, t, latest, w),
        "prefill" => prefill(doc, t, latest, w),
        "retained" => retained(doc, t, latest, w),
        "prefix_cache" => prefix_cache(doc, t, latest, w),
        _ => return None,
    })
}

fn decode_content(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let rows: Vec<(String, f64, &str, String)> = v["rows"].as_array().map(|rows| rows.iter().map(|r| {
        let content = r["content"].as_str().unwrap_or("?").to_string();
        let note = match num(r, "acceptance") {
            Some(a) => format!("{:.0}% accepted", 100.0 * a),
            None => format!("{} tokens", r["tokens"].as_u64().unwrap_or(0)),
        };
        let color = content_color(t, &content);
        (content, num(r, "tok_s").unwrap_or(0.0), color, note)
    }).collect()).unwrap_or_default();
    if rows.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    hbars(doc, t, 0.0, 0.0, w, &rows, "tok/s")
}

fn plot_height(w: f64) -> f64 {
    (w * 0.3).clamp(180.0, 280.0)
}

fn concurrency(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let points = v["points"].as_array().cloned().unwrap_or_default();
    if points.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let max_c = points.iter().filter_map(|p| num(p, "c")).fold(1.0, f64::max).max(2.0);
    let agg: Vec<(f64, f64)> = points.iter().filter_map(|p| Some((num(p, "c")?, num(p, "aggregate_tok_s")?))).collect();
    let per: Vec<(f64, f64)> = points.iter().filter_map(|p| Some((num(p, "c")?, num(p, "per_request_tok_s")?))).collect();
    let h = plot_height(w);
    let plot = Plot { x: 46.0, y: 26.0, w: w - 150.0, h, xs: Scale::Log2 { min: 1.0, max: max_c },
        ys: Scale::linear_from(agg.iter().chain(&per).map(|p| p.1), true) };
    legend(doc, t, 46.0, 10.0, &[("aggregate", t.series[0]), ("per request", t.series[1])]);
    plot.axes(doc, t, "concurrent requests", "tok/s", &|x| format!("C{}", x as u64));
    plot.line(doc, &agg, t.series[0], agg.last().map(|p| rate(p.1)).as_deref());
    plot.line(doc, &per, t.series[1], per.last().map(|p| rate(p.1)).as_deref());
    26.0 + h + 36.0
}

fn prefill(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let points = v["points"].as_array().cloned().unwrap_or_default();
    if points.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let cold: Vec<(f64, f64)> = points.iter().filter_map(|p| Some((num(p, "prompt_tokens")?, num(p, "cold_ttft_s")?))).collect();
    let cached: Vec<(f64, f64)> = points.iter().filter_map(|p| Some((num(p, "prompt_tokens")?, num(p, "cached_ttft_s")?))).collect();
    let max_x = cold.iter().map(|p| p.0).fold(2048.0, f64::max);
    let ys: Vec<f64> = cold.iter().chain(&cached).map(|p| p.1).collect();
    let (lo, hi) = (ys.iter().cloned().fold(f64::INFINITY, f64::min).max(0.005), ys.iter().cloned().fold(0.01, f64::max));
    let h = plot_height(w);
    let plot = Plot { x: 46.0, y: 26.0, w: w - 130.0, h, xs: Scale::Log2 { min: 1024.0, max: max_x * 1.05 },
        ys: Scale::Log10 { min: 10f64.powf(lo.log10().floor()), max: 10f64.powf(hi.log10().ceil()) } };
    legend(doc, t, 46.0, 10.0, &[("cold", t.series[3]), ("cached prefix (last 256 tokens new)", t.series[1])]);
    plot.axes(doc, t, "prompt tokens", "TTFT s", &|x| short(x));
    plot.line(doc, &cold, t.series[3], None);
    plot.line(doc, &cached, t.series[1], None);
    for p in &points {
        if let (Some(x), Some(y), Some(rate_v)) = (num(p, "prompt_tokens"), num(p, "cold_ttft_s"), num(p, "cold_tok_s")) {
            doc.text(plot.px(x), plot.py(y) - 8.0, Font::new(9.5, t.ink2).anchor(Anchor::Middle), &rate(rate_v));
        }
    }
    26.0 + h + 36.0
}

fn retained(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let points: Vec<(f64, f64)> = v["points"].as_array().map(|p| p.iter()
        .filter_map(|p| Some((num(p, "prompt_tokens")?, num(p, "decode_tok_s")?))).collect()).unwrap_or_default();
    if points.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let h = plot_height(w) * 0.85;
    let plot = Plot { x: 46.0, y: 18.0, w: w - 110.0, h,
        xs: Scale::linear_from(points.iter().map(|p| p.0), true), ys: Scale::linear_from(points.iter().map(|p| p.1), true) };
    plot.axes(doc, t, "tokens of context", "decode tok/s", &|x| short(x));
    plot.line(doc, &points, t.series[0], None);
    for (x, y) in &points {
        doc.text(plot.px(*x), plot.py(*y) - 9.0, Font::new(10.0, t.ink).anchor(Anchor::Middle), &rate(*y));
    }
    18.0 + h + 36.0
}

fn prefix_cache(doc: &mut Doc, t: &Theme, v: &Value, w: f64) -> f64 {
    let turns = v["turns"].as_array().cloned().unwrap_or_default();
    if turns.is_empty() {
        return empty(doc, t, "no measurements yet");
    }
    let max = turns.iter().filter_map(|x| num(x, "prompt_tokens")).fold(1.0, f64::max);
    let h = 150.0;
    let top = 22.0;
    legend(doc, t, 0.0, 10.0, &[("restored from cache", t.series[1]), ("prefilled", t.series[3])]);
    let slot = w / turns.len().max(6) as f64;
    for (i, turn) in turns.iter().enumerate() {
        let prompt = num(turn, "prompt_tokens").unwrap_or(0.0);
        let cached = num(turn, "cached_tokens").unwrap_or(0.0).min(prompt);
        let x = slot * i as f64 + slot * 0.2;
        let bw = slot * 0.6;
        let total_h = h * prompt / max;
        let cached_h = h * cached / max;
        let base = top + h + 8.0;
        doc.rect(x, base - total_h, bw, total_h - cached_h, 2.0, t.series[3]);
        doc.rect(x, base - cached_h, bw, cached_h, 2.0, t.series[1]);
        let mut label = seconds(num(turn, "ttft_s").unwrap_or(f64::NAN));
        if let Some(cold) = num(turn, "cold_ttft_s") {
            label.push_str(&format!(" / {}", seconds(cold)));
        }
        doc.text(x + bw / 2.0, base - total_h - 6.0, Font::new(10.0, t.ink).anchor(Anchor::Middle), &label);
        doc.text(x + bw / 2.0, base + 14.0, Font::new(10.0, t.muted).anchor(Anchor::Middle),
            &format!("turn {} · {:.0}%", turn["turn"], 100.0 * cached / prompt.max(1.0)));
    }
    doc.text(w, 10.0, Font::new(10.0, t.muted).anchor(Anchor::End), "TTFT / cold TTFT above each turn");
    top + h + 30.0
}

