//! A graph arena is a named owner, independent of GPU_RESERVE and KV policy.
use cuteafd_core::memory_policy::Size;

/// GLM's measured full startup base plus room for lazy shapes. The extra is a
/// policy budget, not a memory prediction; the runtime enforces the same value.
pub const GLM_EXTRA: u64 = 64 << 20;

pub fn source(id: &str) -> String { format!("graph reserve: {id}") }
pub fn is_source(source: &str) -> bool { source.starts_with("graph reserve: ") }

/// Bare sizes apply everywhere. Qualified entries are `arch/role=size`, e.g.
/// `sm120/coordinator=4GiB,sm121/spark=512MiB`; `*` matches either field.
pub fn resolve(text: Option<&str>, arch: &str, role: &str, base: u64, extra: u64) -> Result<u64, String> {
    let Some(text) = text.filter(|text| !text.trim().is_empty()) else {
        return base.checked_add(extra).ok_or_else(|| "graph reserve overflows".into());
    };
    if !text.contains('=') { return text.parse::<Size>().map(|size| size.0).map_err(|e| e.to_string()); }
    let mut selected = None;
    for item in text.split(',') {
        let (key, value) = item.trim().split_once('=').ok_or("expected graph reserve selector=size")?;
        let (a, r) = key.split_once('/').ok_or("expected graph reserve arch/role")?;
        if !matches!(a, "*" | "sm120" | "sm121") || !matches!(r, "*" | "coordinator" | "spark") {
            return Err(format!("invalid graph reserve selector {key}"));
        }
        let size = value.parse::<Size>().map_err(|e| e.to_string())?.0;
        if (a == "*" || a == arch) && (r == "*" || r == role) {
            let specificity = usize::from(a != "*") + usize::from(r != "*");
            if selected.is_some_and(|(prior, _)| prior == specificity) { return Err("ambiguous graph reserve selectors".into()); }
            if selected.is_none_or(|(prior, _)| specificity > prior) { selected = Some((specificity, size)); }
        }
    }
    selected.map(|(_, size)| size).map_or_else(|| base.checked_add(extra).ok_or_else(|| "graph reserve overflows".into()), Ok)
}

pub fn current(arch: &str, role: &str, base: u64, extra: u64) -> Result<u64, String> {
    resolve(std::env::var("CUTEAFD_GRAPH_RESERVE").ok().as_deref(), arch, role, base, extra)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reserve_selector_is_independent_of_gpu_reserve() {
        assert_eq!(resolve(None, "sm120", "coordinator", 100, 20).unwrap(), 120);
        assert_eq!(resolve(Some("sm120/coordinator=4GiB,sm121/spark=512MiB"), "sm120", "coordinator", 100, 20).unwrap(), 4 << 30);
        assert_eq!(resolve(Some("0B"), "sm121", "spark", 100, 20).unwrap(), 0);
        assert!(resolve(Some("sm999/coordinator=1GiB"), "sm120", "coordinator", 0, 0).is_err());
    }
}
