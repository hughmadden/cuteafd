//! A small SVG writer: self-contained documents (no external fonts, scripts
//! or images) that render the same in a browser, on GitHub and through resvg.
use std::fmt::Write;

/// Monospace stack: the viewer's UI monospace, DejaVu Sans Mono for resvg.
pub const MONO: &str = "ui-monospace, 'SF Mono', 'JetBrains Mono', 'Cascadia Mono', Menlo, Consolas, \
    'DejaVu Sans Mono', monospace";

/// Width of `text` at `size` px in the monospace stack (0.6 em per char).
pub fn text_width(text: &str, size: f64) -> f64 {
    text.chars().count() as f64 * size * 0.6
}

pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c if (c as u32) < 0x20 && c != '\n' && c != '\t' => {}
            c => out.push(c),
        }
    }
    out
}

/// Truncates `text` to fit `width` px at `size`, with an ellipsis.
pub fn fit(text: &str, size: f64, width: f64) -> String {
    let max = (width / (size * 0.6)).floor().max(1.0) as usize;
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    Start,
    Middle,
    End,
}

impl Anchor {
    fn attr(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Middle => "middle",
            Self::End => "end",
        }
    }
}

/// Text style.
#[derive(Debug, Clone, Copy)]
pub struct Font<'a> {
    pub size: f64,
    pub weight: u32,
    pub color: &'a str,
    pub anchor: Anchor,
    pub spacing: f64,
    pub opacity: f64,
}

impl<'a> Font<'a> {
    pub fn new(size: f64, color: &'a str) -> Self {
        Self { size, weight: 400, color, anchor: Anchor::Start, spacing: 0.0, opacity: 1.0 }
    }
    pub fn bold(mut self) -> Self {
        self.weight = 700;
        self
    }
    pub fn weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }
    pub fn anchor(mut self, anchor: Anchor) -> Self {
        self.anchor = anchor;
        self
    }
    pub fn spacing(mut self, spacing: f64) -> Self {
        self.spacing = spacing;
        self
    }
    pub fn opacity(mut self, opacity: f64) -> Self {
        self.opacity = opacity;
        self
    }
}

/// An SVG document under construction.
pub struct Doc {
    pub width: f64,
    pub height: f64,
    defs: String,
    body: String,
}

fn n(v: f64) -> String {
    let rounded = (v * 100.0).round() / 100.0;
    if rounded.fract() == 0.0 { format!("{}", rounded as i64) } else { format!("{rounded}") }
}

impl Doc {
    pub fn new(width: f64, height: f64) -> Self {
        Self { width, height, defs: String::new(), body: String::new() }
    }

    pub fn defs(&mut self, fragment: &str) {
        self.defs.push_str(fragment);
    }

    pub fn raw(&mut self, fragment: &str) {
        self.body.push_str(fragment);
    }

    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, radius: f64, fill: &str) {
        let _ = write!(self.body, r#"<rect x="{}" y="{}" width="{}" height="{}" rx="{}" fill="{fill}"/>"#,
            n(x), n(y), n(w.max(0.0)), n(h.max(0.0)), n(radius));
    }

    pub fn rect_attrs(&mut self, x: f64, y: f64, w: f64, h: f64, radius: f64, attrs: &str) {
        let _ = write!(self.body, r#"<rect x="{}" y="{}" width="{}" height="{}" rx="{}" {attrs}/>"#,
            n(x), n(y), n(w.max(0.0)), n(h.max(0.0)), n(radius));
    }

    /// A stroked, filled box (panel frames).
    pub fn frame(&mut self, x: f64, y: f64, w: f64, h: f64, radius: f64, fill: &str, stroke: &str) {
        let _ = write!(self.body, r#"<rect x="{}" y="{}" width="{}" height="{}" rx="{}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>"#,
            n(x + 0.5), n(y + 0.5), n(w - 1.0), n(h - 1.0), n(radius));
    }

    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, width: f64) {
        let _ = write!(self.body, r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{stroke}" stroke-width="{}"/>"#,
            n(x1), n(y1), n(x2), n(y2), n(width));
    }

    pub fn dashed(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str) {
        let _ = write!(self.body, r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{stroke}" stroke-width="1" stroke-dasharray="3 4"/>"#,
            n(x1), n(y1), n(x2), n(y2));
    }

    pub fn circle(&mut self, x: f64, y: f64, r: f64, fill: &str) {
        let _ = write!(self.body, r#"<circle cx="{}" cy="{}" r="{}" fill="{fill}"/>"#, n(x), n(y), n(r));
    }

    pub fn path(&mut self, d: &str, attrs: &str) {
        let _ = write!(self.body, r#"<path d="{d}" {attrs}/>"#);
    }

    /// A polyline through `points` (data already in pixels).
    pub fn polyline(&mut self, points: &[(f64, f64)], stroke: &str, width: f64) {
        if points.is_empty() {
            return;
        }
        let list: Vec<String> = points.iter().map(|(x, y)| format!("{},{}", n(*x), n(*y))).collect();
        let _ = write!(self.body, r#"<polyline points="{}" fill="none" stroke="{stroke}" stroke-width="{}" stroke-linejoin="round" stroke-linecap="round"/>"#,
            list.join(" "), n(width));
    }

    pub fn text(&mut self, x: f64, y: f64, font: Font<'_>, text: &str) {
        let mut attrs = format!(r#"x="{}" y="{}" font-family="{MONO}" font-size="{}" fill="{}""#, n(x), n(y),
            n(font.size), font.color);
        if font.weight != 400 {
            let _ = write!(attrs, r#" font-weight="{}""#, font.weight);
        }
        if font.anchor != Anchor::Start {
            let _ = write!(attrs, r#" text-anchor="{}""#, font.anchor.attr());
        }
        if font.spacing != 0.0 {
            let _ = write!(attrs, r#" letter-spacing="{}""#, n(font.spacing));
        }
        if font.opacity < 1.0 {
            let _ = write!(attrs, r#" opacity="{}""#, n(font.opacity));
        }
        let _ = write!(self.body, "<text {attrs}>{}</text>", escape(text));
    }

    /// Text made of runs with their own colors and weights, one line.
    pub fn spans(&mut self, x: f64, y: f64, size: f64, anchor: Anchor, runs: &[(&str, &str, u32)]) {
        let mut out = format!(r#"<text x="{}" y="{}" font-family="{MONO}" font-size="{}""#, n(x), n(y), n(size));
        if anchor != Anchor::Start {
            let _ = write!(out, r#" text-anchor="{}""#, anchor.attr());
        }
        out.push('>');
        for (text, color, weight) in runs {
            let _ = write!(out, r#"<tspan fill="{color}"{}>{}</tspan>"#,
                if *weight != 400 { format!(r#" font-weight="{weight}""#) } else { String::new() }, escape(text));
        }
        out.push_str("</text>");
        self.body.push_str(&out);
    }

    /// Starts a translated group; close with [`Self::end`].
    pub fn group(&mut self, x: f64, y: f64) {
        let _ = write!(self.body, r#"<g transform="translate({},{})">"#, n(x), n(y));
    }

    pub fn group_attrs(&mut self, attrs: &str) {
        let _ = write!(self.body, "<g {attrs}>");
    }

    pub fn end(&mut self) {
        self.body.push_str("</g>");
    }

    /// Nests a self-contained `<svg viewBox=…>` fragment at x, y, w, h.
    pub fn nested(&mut self, fragment: &str, x: f64, y: f64, w: f64, h: f64) {
        let placed = fragment.trim().replacen("<svg ", &format!(r#"<svg x="{}" y="{}" width="{}" height="{}" "#,
            n(x), n(y), n(w), n(h)), 1);
        self.body.push_str(&placed);
    }

    pub fn finish(self) -> String {
        let mut out = format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            n(self.width), n(self.height), n(self.width), n(self.height));
        if !self.defs.is_empty() {
            let _ = write!(out, "<defs>{}</defs>", self.defs);
        }
        out.push_str(&self.body);
        out.push_str("</svg>\n");
        out
    }

    /// The document's body so far (to embed in another document).
    pub fn into_parts(self) -> (String, String) {
        (self.defs, self.body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documents_escape_and_close() {
        let mut doc = Doc::new(100.0, 50.0);
        doc.text(1.0, 2.0, Font::new(12.0, "#fff").bold(), "a < b & \"c\"");
        doc.rect(0.0, 0.0, 10.5, 3.333, 2.0, "#000");
        let svg = doc.finish();
        assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(svg.contains("a &lt; b &amp; &quot;c&quot;"));
        assert!(svg.contains(r#"height="3.33""#));
        assert!(svg.trim_end().ends_with("</svg>"));
    }

    #[test]
    fn fit_truncates_with_an_ellipsis() {
        assert_eq!(fit("abcdef", 10.0, 36.0), "abcdef");
        assert_eq!(fit("abcdefgh", 10.0, 36.0), "abcde…");
    }
}
