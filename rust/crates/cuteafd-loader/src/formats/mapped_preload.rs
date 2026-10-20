//! All-or-nothing residency policy for separate per-row lookup-table scales.
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Off by default: #419 measured flat decode tok/s with all V4.1 scales preloaded.
pub const DEFAULT_CAP_BYTES: u64 = 0;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappedPlacement {
    pub mapped: usize,
    pub preloaded: usize,
    pub scale_bytes: u64,
    pub preloaded_bytes: u64,
    pub cap_bytes: u64,
    pub host_budget_bytes: u64,
    pub reason: String,
}

impl MappedPlacement {
    pub fn resolve(tables: usize, scales: usize, scale_bytes: u64, cap_bytes: u64, host_budget_bytes: u64) -> Self {
        let reason = if scales == 0 { "no separate per-row scales" }
            else if cap_bytes == 0 { "disabled" }
            else if scale_bytes > cap_bytes { "all scale tensors exceed preload cap" }
            else if scale_bytes > host_budget_bytes { "all scale tensors exceed available host budget" }
            else { "all scale tensors admitted" };
        let preloaded = if scales > 0 && cap_bytes > 0 && scale_bytes <= cap_bytes && scale_bytes <= host_budget_bytes { scales } else { 0 };
        Self { mapped: tables - preloaded, preloaded, scale_bytes,
            preloaded_bytes: if preloaded > 0 { scale_bytes } else { 0 }, cap_bytes, host_budget_bytes, reason: reason.into() }
    }

    pub fn label(&self) -> String {
        if self.preloaded == 0 { format!("PLE {} mapped", self.mapped) }
        else { format!("PLE {} mapped / {} preloaded", self.mapped, self.preloaded) }
    }

    pub fn fallback(&mut self, error: &impl std::fmt::Display) {
        self.mapped += self.preloaded;
        self.preloaded = 0;
        self.preloaded_bytes = 0;
        self.reason = format!("scale preload unavailable: {error}");
    }
}

/// Startup admission: live available RAM, bounded by every cgroup ancestor,
/// with the same ten-percent headroom used by the host prefix-cache budget.
/// Once allocated, the anonymous buffers are charged by the OS to subsequent
/// prefix-cache admission, not subtracted a second time.
pub fn host_budget_bytes() -> Result<u64> {
    let (total, available) = host_memory()?;
    Ok(available.saturating_sub(total / 10))
}

/// Live host/cgroup capacity before coordinator allocations.
pub fn host_memory() -> Result<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo")?;
    let field = |key| -> Result<u64> {
        let value = text.lines().find_map(|line| line.strip_prefix(key))
            .with_context(|| format!("missing host memory field {key}"))?;
        Ok(value.split_whitespace().next().context("missing memory value")?.parse::<u64>()?.checked_mul(1024).context("host memory overflow")?)
    };
    let mut total = field("MemTotal:")?;
    let mut available = field("MemAvailable:")?;
    let cgroups = std::fs::read_to_string("/proc/self/cgroup")?;
    for line in cgroups.lines() {
        let parts: Vec<_> = line.splitn(3, ':').collect();
        let [_, controllers, relative] = parts.as_slice() else { continue };
        let (root, limit_file, current_file) = if controllers.is_empty() {
            (Path::new("/sys/fs/cgroup"), "memory.max", "memory.current")
        } else if controllers.split(',').any(|c| c == "memory") {
            (Path::new("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes")
        } else { continue };
        let member = root.join(relative.trim_start_matches('/'));
        for path in member.ancestors().take_while(|p| p.starts_with(root)).chain(std::iter::once(root)) {
            if let (Ok(limit), Ok(current)) = (std::fs::read_to_string(path.join(limit_file)), std::fs::read_to_string(path.join(current_file))) {
                if limit.trim() == "max" { continue }
                let (limit, current) = (limit.trim().parse::<u64>()?, current.trim().parse::<u64>()?);
                total = total.min(limit);
                available = available.min(limit.saturating_sub(current));
            }
        }
    }
    Ok((total, available))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_or_nothing_cap_and_host_admission() {
        let bytes = 6 << 30;
        let cap = 8 << 30;
        let admitted = MappedPlacement::resolve(4, 2, bytes, cap, bytes);
        assert_eq!(admitted.label(), "PLE 2 mapped / 2 preloaded");
        assert_eq!(admitted.preloaded_bytes, bytes);
        for (cap, host) in [(DEFAULT_CAP_BYTES, bytes), (bytes - 1, bytes), (cap, bytes - 1)] {
            let rejected = MappedPlacement::resolve(4, 2, bytes, cap, host);
            assert_eq!(rejected.label(), "PLE 4 mapped");
            assert_eq!(rejected.preloaded_bytes, 0);
        }
        assert_eq!(MappedPlacement::resolve(2, 0, 0, DEFAULT_CAP_BYTES, 0).label(), "PLE 2 mapped");
    }
}
