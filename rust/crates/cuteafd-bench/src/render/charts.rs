//! Chart primitives for panel bodies: axes, line and bar charts, heatmaps,
//! violins and point ranges, drawn into a [`Doc`] with the report theme.
use super::svg::{fit, text_width, Anchor, Doc, Font};
use super::Theme;

/// An axis scale.
#[derive(Debug, Clone, Copy)]
pub enum Scale {
    Linear { min: f64, max: f64 },
    /// log2 (concurrency, context lengths).
    Log2 { min: f64, max: f64 },
    /// log10 (token counts).
    Log10 { min: f64, max: f64 },
}

impl Scale {
    pub fn linear_from(values: impl Iterator<Item = f64>, zero: bool) -> Self {
        let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in values.filter(|v| v.is_finite()) {
            min = min.min(v);
            max = max.max(v);
        }
        if !min.is_finite() {
            return Self::Linear { min: 0.0, max: 1.0 };
        }
        if zero {
            min = min.min(0.0);
        }
        if max <= min {
            max = min + 1.0;
        }
        let (min, max) = nice(min, max + (max - min) * 0.06);
        Self::Linear { min: if zero { min.min(0.0) } else { min }, max }
    }

    /// 0..1 position of `v`.
    pub fn unit(&self, v: f64) -> f64 {
        match *self {
            Self::Linear { min, max } => (v - min) / (max - min),
            Self::Log2 { min, max } => (v.max(1e-9).log2() - min.log2()) / (max.log2() - min.log2()).max(1e-9),
            Self::Log10 { min, max } => (v.max(1e-9).log10() - min.log10()) / (max.log10() - min.log10()).max(1e-9),
        }
    }

    pub fn ticks(&self) -> Vec<f64> {
        match *self {
            Self::Linear { min, max } => {
                let step = nice_step((max - min) / 5.0);
                let mut t = (min / step).ceil() * step;
                let mut out = Vec::new();
                while t <= max + step * 1e-6 {
                    out.push(t);
                    t += step;
                }
                out
            }
            Self::Log2 { min, max } => {
                let mut out = Vec::new();
                let mut t = 2f64.powi(min.log2().ceil() as i32);
                while t <= max * 1.0001 {
                    out.push(t);
                    t *= 2.0;
                }
                out
            }
            Self::Log10 { min, max } => {
                let mut out = Vec::new();
                let mut t = 10f64.powi(min.log10().ceil() as i32);
                while t <= max * 1.0001 {
                    out.push(t);
                    t *= 10.0;
                }
                out
            }
        }
    }
}

fn nice_step(raw: f64) -> f64 {
    if raw <= 0.0 || !raw.is_finite() {
        return 1.0;
    }
    let p = 10f64.powf(raw.log10().floor());
    let f = raw / p;
    p * if f <= 1.0 { 1.0 } else if f <= 2.0 { 2.0 } else if f <= 5.0 { 5.0 } else { 10.0 }
}

fn nice(min: f64, max: f64) -> (f64, f64) {
    let step = nice_step((max - min) / 5.0);
    ((min / step).floor() * step, (max / step).ceil() * step)
}

/// Compact tick labels: 1.5k, 128K tokens, 0.25.
pub fn short(v: f64) -> String {
    let a = v.abs();
    if a >= 1e6 {
        format!("{:.0}M", v / 1e6)
    } else if a >= 1e3 && (v / 1e3).fract().abs() < 1e-9 {
        format!("{:.0}k", v / 1e3)
    } else if a >= 1e3 {
        format!("{:.1}k", v / 1e3)
    } else if a >= 10.0 || v.fract() == 0.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// A plot area with axes.
pub struct Plot {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub xs: Scale,
    pub ys: Scale,
}

impl Plot {
    pub fn px(&self, v: f64) -> f64 {
        self.x + self.w * self.xs.unit(v).clamp(-0.05, 1.05)
    }
    pub fn py(&self, v: f64) -> f64 {
        self.y + self.h * (1.0 - self.ys.unit(v).clamp(-0.05, 1.05))
    }

    /// Grid, ticks and axis titles.
    pub fn axes(&self, doc: &mut Doc, t: &Theme, x_title: &str, y_title: &str, x_label: &dyn Fn(f64) -> String) {
        for tick in self.ys.ticks() {
            let y = self.py(tick);
            doc.line(self.x, y, self.x + self.w, y, t.line, 1.0);
            doc.text(self.x - 6.0, y + 3.5, Font::new(10.0, t.muted).anchor(Anchor::End), &short(tick));
        }
        for tick in self.xs.ticks() {
            let x = self.px(tick);
            doc.line(x, self.y + self.h, x, self.y + self.h + 4.0, t.line2, 1.0);
            doc.text(x, self.y + self.h + 16.0, Font::new(10.0, t.muted).anchor(Anchor::Middle), &x_label(tick));
        }
        doc.line(self.x, self.y + self.h, self.x + self.w, self.y + self.h, t.line2, 1.0);
        if !x_title.is_empty() {
            doc.text(self.x + self.w, self.y + self.h + 30.0, Font::new(10.0, t.muted).anchor(Anchor::End), x_title);
        }
        if !y_title.is_empty() {
            doc.text(self.x - 34.0, self.y - 8.0, Font::new(10.0, t.muted), y_title);
        }
    }

    /// A series as a line with dots and an optional end label.
    pub fn line(&self, doc: &mut Doc, points: &[(f64, f64)], color: &str, label: Option<&str>) {
        let pts: Vec<(f64, f64)> = points.iter().filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|&(x, y)| (self.px(x), self.py(y))).collect();
        doc.polyline(&pts, color, 2.0);
        for &(x, y) in &pts {
            doc.circle(x, y, 3.0, color);
        }
        if let (Some(label), Some(&(x, y))) = (label, pts.last()) {
            doc.text(x + 6.0, y + 3.5, Font::new(10.5, color).weight(600), label);
        }
    }

    /// Vertical whiskers (min..max) with a dot at the center.
    pub fn range(&self, doc: &mut Doc, x: f64, lo: f64, mid: f64, hi: f64, color: &str) {
        let px = self.px(x);
        doc.line(px, self.py(lo), px, self.py(hi), color, 1.5);
        doc.line(px - 4.0, self.py(lo), px + 4.0, self.py(lo), color, 1.5);
        doc.line(px - 4.0, self.py(hi), px + 4.0, self.py(hi), color, 1.5);
        doc.circle(px, self.py(mid), 4.0, color);
    }
}

/// Legend entries in a row; returns the height used.
pub fn legend(doc: &mut Doc, t: &Theme, x: f64, y: f64, items: &[(&str, &str)]) -> f64 {
    let mut cx = x;
    for (label, color) in items {
        doc.rect(cx, y - 8.0, 10.0, 10.0, 2.0, color);
        doc.text(cx + 15.0, y + 1.0, Font::new(11.0, t.ink2), label);
        cx += 30.0 + text_width(label, 11.0);
    }
    if items.is_empty() { 0.0 } else { 18.0 }
}

/// Horizontal bars: rows of (label, value, color, annotation); returns height.
pub fn hbars(doc: &mut Doc, t: &Theme, x: f64, y: f64, w: f64, rows: &[(String, f64, &str, String)], unit: &str) -> f64 {
    let max = rows.iter().map(|r| r.1).fold(1e-9f64, f64::max) * 1.1;
    let label_w = 120.0;
    let value_w = 170.0;
    for (i, (label, value, color, note)) in rows.iter().enumerate() {
        let ry = y + 24.0 * i as f64;
        doc.text(x, ry + 13.0, Font::new(11.5, t.ink2), &fit(label, 11.5, label_w - 8.0));
        let track = w - label_w - value_w;
        doc.rect(x + label_w, ry + 3.0, track, 14.0, 3.0, t.well);
        doc.rect(x + label_w, ry + 3.0, track * (value / max).clamp(0.0, 1.0), 14.0, 3.0, color);
        doc.spans(x + w, ry + 14.0, 11.5, Anchor::End, &[(&format!("{} {unit}", super::rate(*value)), t.ink, 600),
            (&if note.is_empty() { String::new() } else { format!("  {note}") }, t.muted, 400)]);
    }
    24.0 * rows.len() as f64
}

/// A heatmap of cells (x index, y index) -> value in 0..1 with labels.
#[allow(clippy::too_many_arguments)]
pub fn heatmap(doc: &mut Doc, t: &Theme, x: f64, y: f64, w: f64, xs: &[String], ys: &[String],
    cell: &dyn Fn(usize, usize) -> Option<f64>, good: &str, bad: &str) -> f64 {
    let label_w = 70.0;
    let cw = (w - label_w) / xs.len().max(1) as f64;
    let ch = 22.0;
    for (j, ylabel) in ys.iter().enumerate() {
        let cy = y + ch * j as f64;
        doc.text(x + label_w - 8.0, cy + 15.0, Font::new(10.5, t.muted).anchor(Anchor::End), ylabel);
        for i in 0..xs.len() {
            let cx = x + label_w + cw * i as f64;
            match cell(i, j) {
                Some(v) => {
                    let color = if v >= 0.999 { good } else if v <= 0.001 { bad } else { t.warn };
                    let opacity = 0.35 + 0.65 * if v >= 0.5 { v } else { 1.0 - v };
                    doc.rect_attrs(cx + 1.0, cy + 1.0, cw - 2.0, ch - 2.0, 3.0,
                        &format!(r#"fill="{color}" opacity="{opacity:.2}""#));
                    if cw > 34.0 {
                        doc.text(cx + cw / 2.0, cy + 15.0, Font::new(10.0, t.bg).anchor(Anchor::Middle).weight(600),
                            &format!("{:.0}%", 100.0 * v));
                    }
                }
                None => doc.rect(cx + 1.0, cy + 1.0, cw - 2.0, ch - 2.0, 3.0, t.well),
            }
        }
    }
    let ly = y + ch * ys.len() as f64 + 14.0;
    for (i, xlabel) in xs.iter().enumerate() {
        doc.text(x + label_w + cw * (i as f64 + 0.5), ly, Font::new(10.0, t.muted).anchor(Anchor::Middle), xlabel);
    }
    ch * ys.len() as f64 + 22.0
}

/// A violin (kernel density, mirrored) of `values` at center x over `plot`'s y scale.
pub fn violin(doc: &mut Doc, plot: &Plot, x: f64, half_width: f64, values: &[f64], color: &str) {
    let values: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        return;
    }
    let cx = plot.px(x);
    if values.len() == 1 {
        doc.circle(cx, plot.py(values[0]), 3.5, color);
        return;
    }
    // Density in unit (plot) space so log scales look right.
    let units: Vec<f64> = values.iter().map(|&v| plot.ys.unit(v)).collect();
    let n = units.len() as f64;
    let mean = units.iter().sum::<f64>() / n;
    let sd = (units.iter().map(|u| (u - mean).powi(2)).sum::<f64>() / n).sqrt().max(0.02);
    let bw = 1.06 * sd * n.powf(-0.2);
    let (lo, hi) = units.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &u| (a.min(u), b.max(u)));
    let steps = 32;
    let mut density = Vec::with_capacity(steps + 1);
    for s in 0..=steps {
        let u = lo - bw + (hi - lo + 2.0 * bw) * s as f64 / steps as f64;
        let d: f64 = units.iter().map(|&v| (-0.5 * ((u - v) / bw).powi(2)).exp()).sum();
        density.push((u, d));
    }
    let peak = density.iter().map(|d| d.1).fold(1e-9, f64::max);
    let y_of = |u: f64| plot.y + plot.h * (1.0 - u.clamp(-0.05, 1.05));
    let mut path = String::new();
    for (i, (u, d)) in density.iter().enumerate() {
        let dx = half_width * d / peak;
        path.push_str(&format!("{}{:.1},{:.1}", if i == 0 { "M" } else { "L" }, cx + dx, y_of(*u)));
    }
    for (u, d) in density.iter().rev() {
        path.push_str(&format!("L{:.1},{:.1}", cx - half_width * d / peak, y_of(*u)));
    }
    path.push('Z');
    doc.path(&path, &format!(r#"fill="{color}" fill-opacity="0.35" stroke="{color}" stroke-width="1.2""#));
    let mut sorted = values.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    doc.line(cx - half_width * 0.6, plot.py(median), cx + half_width * 0.6, plot.py(median), color, 2.0);
}

/// Median and the min/max spread of a sample.
pub fn spread(values: &[f64]) -> Option<(f64, f64, f64)> {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some((v[0], v[v.len() / 2], v[v.len() - 1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_and_ticks() {
        let s = Scale::linear_from([3.0, 187.0].into_iter(), true);
        assert!(matches!(s, Scale::Linear { min, max } if min == 0.0 && max >= 187.0));
        assert!(s.ticks().len() >= 3);
        let log = Scale::Log2 { min: 1.0, max: 32.0 };
        assert_eq!(log.ticks(), vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0]);
        assert!((log.unit(4.0) - 0.4).abs() < 1e-9);
        assert_eq!(short(131072.0), "131.1k");
        assert_eq!(short(4096.0), "4.1k");
        assert_eq!(short(8000.0), "8k");
        assert_eq!(spread(&[3.0, 1.0, 2.0]), Some((1.0, 2.0, 3.0)));
    }
}
