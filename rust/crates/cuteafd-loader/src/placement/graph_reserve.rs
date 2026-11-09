//! A graph arena is a named owner, independent of GPU_RESERVE and KV policy.
use cuteafd_core::memory_policy::Size;

/// Per-rank measured startup base plus an enforced lazy-shape policy budget.
pub const GLM_EXTRA: u64 = 64 << 20;
pub fn source(id: &str) -> String { format!("graph reserve: {id}") }
pub fn is_source(source: &str) -> bool { source.starts_with("graph reserve: ") }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved { pub bytes: u64, pub source: &'static str }

/// A bare size overrides each coordinator device, never splits across ranks.
/// Selectors override `gpu0`, `gpu1`, ... or every worker through `spark`.
pub fn resolve(text: Option<&str>, rank: usize, spark: bool, base: u64, extra: u64) -> Result<Resolved, String> {
    let default = || base.checked_add(extra).map(|bytes| Resolved { bytes, source: "family default" })
        .ok_or_else(|| "graph reserve overflows".into());
    let Some(text) = text.filter(|text| !text.trim().is_empty()) else { return default() };
    if !text.contains('=') {
        let size = text.parse::<Size>().map_err(|e| e.to_string())?.0;
        return if spark { default() } else { Ok(Resolved { bytes: size, source: "user override" }) };
    }
    let target = if spark { "spark".to_owned() } else { format!("gpu{rank}") };
    let mut seen = std::collections::HashSet::new();
    let mut selected = None;
    for item in text.split(',') {
        let (key, value) = item.trim().split_once('=').ok_or("expected graph reserve selector=size")?;
        if key != "spark" && key.strip_prefix("gpu").and_then(|n| n.parse::<usize>().ok()).is_none() {
            return Err(format!("invalid graph reserve selector {key}"));
        }
        if !seen.insert(key) { return Err(format!("duplicate graph reserve selector {key}")); }
        let size = value.parse::<Size>().map_err(|e| e.to_string())?.0;
        if key == target { selected = Some(Resolved { bytes: size, source: "user override" }); }
    }
    selected.map_or_else(default, Ok)
}

pub fn current(rank: usize, spark: bool, base: u64, extra: u64) -> Result<Resolved, String> {
    resolve(std::env::var("CUTEAFD_GRAPH_RESERVE").ok().as_deref(), rank, spark, base, extra)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reserve_selectors_are_per_device_and_independent_of_gpu_reserve() {
        assert_eq!(resolve(None, 0, false, 100, 20).unwrap().bytes, 120);
        let text = Some("gpu0=3GiB,gpu1=2GiB,spark=512MiB");
        assert_eq!(resolve(text, 0, false, 100, 20).unwrap().bytes, 3 << 30);
        assert_eq!(resolve(text, 1, false, 100, 20).unwrap().bytes, 2 << 30);
        assert_eq!(resolve(text, 9, true, 100, 20).unwrap().bytes, 512 << 20);
        assert_eq!(resolve(Some("0B"), 1, false, 100, 20).unwrap().bytes, 0);
        assert_eq!(resolve(Some("1GiB"), 1, true, 100, 20).unwrap().bytes, 120);
        assert!(resolve(Some("gpu0=1GiB,gpu0=2GiB"), 0, false, 0, 0).is_err());
    }
}
