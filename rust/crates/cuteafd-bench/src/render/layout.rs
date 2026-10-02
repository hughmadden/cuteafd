//! Panel spans and row packing, shared by the whole-report export and the
//! dashboard (which reads the same spans from `/v1/bench/panels`).
use serde::Serialize;

/// A panel's preferred share of a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Span {
    Third,
    Half,
    Full,
}

impl Span {
    /// Units of a six-unit row.
    pub fn units(self) -> usize {
        match self {
            Self::Third => 2,
            Self::Half => 3,
            Self::Full => 6,
        }
    }
}

/// Preferred span and minimum width (px) of panel `id`: full width for wide
/// text and timelines, half or a third for compact charts.
pub fn span(id: &str) -> (Span, f64) {
    match id {
        "decode_content" | "concurrency" | "retained" | "prefill" | "prefix_cache" | "tool_eval" | "structured" =>
            (Span::Half, 420.0),
        "ifeval" | "code" | "math" => (Span::Third, 300.0),
        _ => (Span::Full, 600.0),
    }
}

/// One packed panel: its id, x offset and width inside the content box.
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    pub id: String,
    pub x: f64,
    pub width: f64,
}

/// Packs `ids` (in order) greedily into rows of six units over `width` px
/// with `gap` px between cells. A row that is not full stretches its cells
/// over the whole width; a span whose minimum width does not fit widens.
pub fn pack(ids: &[String], width: f64, gap: f64) -> Vec<Vec<Cell>> {
    let unit = (width - 5.0 * gap) / 6.0;
    let units_for = |id: &str| {
        let (span, min) = span(id);
        let mut units = span.units();
        while units < 6 && (unit * units as f64 + gap * (units - 1) as f64) < min {
            units = if units < 3 { 3 } else { 6 };
        }
        units
    };
    let mut rows: Vec<Vec<(String, usize)>> = Vec::new();
    let mut used = 6;
    for id in ids {
        let units = units_for(id);
        if used + units > 6 {
            rows.push(Vec::new());
            used = 0;
        }
        rows.last_mut().expect("row").push((id.clone(), units));
        used += units;
    }
    rows.into_iter().map(|row| {
        let total: usize = row.iter().map(|(_, u)| u).sum();
        let scale = 6.0 / total as f64;
        let mut x = 0.0;
        let n = row.len();
        row.into_iter().enumerate().map(|(i, (id, units))| {
            let w = if i + 1 == n { width - x } else { (unit * units as f64 * scale) + gap * (units as f64 * scale - 1.0) };
            let cell = Cell { id, x, width: w };
            x += w + gap;
            cell
        }).collect()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn rows_pack_greedily_and_fill_the_width() {
        let rows = pack(&ids(&["hardware", "decode_content", "concurrency", "math", "code", "ifeval", "needle",
            "tool_eval"]), 960.0, 14.0);
        let shape: Vec<Vec<&str>> = rows.iter().map(|r| r.iter().map(|c| c.id.as_str()).collect()).collect();
        assert_eq!(shape, vec![vec!["hardware"], vec!["decode_content", "concurrency"], vec!["math", "code", "ifeval"],
            vec!["needle"], vec!["tool_eval"]]);
        for row in &rows {
            let last = row.last().unwrap();
            assert!((last.x + last.width - 960.0).abs() < 1e-6, "{row:?}");
        }
        // A lone half panel stretches over the row.
        assert!((rows[4][0].width - 960.0).abs() < 1e-6);
    }

    #[test]
    fn narrow_widths_fall_back_to_wider_spans() {
        let rows = pack(&ids(&["math", "code", "decode_content"]), 700.0, 14.0);
        // A third of 700 px is under 300: thirds become halves (343 px suits them); a half
        // panel needing 420 px becomes full.
        let shape: Vec<usize> = rows.iter().map(Vec::len).collect();
        assert_eq!(shape, vec![2, 1]);
    }
}
