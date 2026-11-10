//! All-or-nothing host residency policy for mapped lookup tables under one
//! cap: every table (rows and scales) when all of them fit, else the separate
//! per-row scales when those fit.
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Off by default: #419 measured flat decode tok/s with all V4.1 scales preloaded.
pub const DEFAULT_CAP_BYTES: u64 = 0;

/// Positioned readers a preload runs, one table at a time.
pub const PRELOAD_READERS: usize = 8;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappedPlacement {
    pub mapped: usize,
    pub preloaded: usize,
    pub scale_bytes: u64,
    pub preloaded_bytes: u64,
    pub cap_bytes: u64,
    /// Every table's bytes, rows and scales, where the runtime can preload
    /// them all ([`Self::admit_tables`]); 0 otherwise.
    #[serde(default)]
    pub table_bytes: u64,
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
            preloaded_bytes: if preloaded > 0 { scale_bytes } else { 0 }, cap_bytes, table_bytes: 0,
            host_budget_bytes, reason: reason.into() }
    }

    /// Preload every table instead, rows and scales (`table_bytes` in all), when
    /// they fit the same cap and the host budget. Otherwise the scale decision
    /// stands; a cap that holds the tables but a budget that does not says so.
    pub fn admit_tables(mut self, table_bytes: u64) -> Self {
        self.table_bytes = table_bytes;
        if self.cap_bytes == 0 || table_bytes == 0 || table_bytes > self.cap_bytes || self.mapped + self.preloaded == 0 {
            return self;
        }
        if table_bytes > self.host_budget_bytes {
            self.reason = format!("all tables exceed available host budget; {}", self.reason);
            return self;
        }
        self.preloaded += std::mem::take(&mut self.mapped);
        self.preloaded_bytes = table_bytes;
        self.reason = "all tables admitted".into();
        self
    }

    /// Every table (rows and scales) is held in host memory.
    pub fn tables_preloaded(&self) -> bool {
        self.table_bytes > 0 && self.mapped == 0 && self.preloaded > 0 && self.preloaded_bytes == self.table_bytes
    }

    pub fn label(&self) -> String {
        if self.preloaded == 0 { format!("PLE {} mapped", self.mapped) }
        else if self.mapped == 0 { format!("PLE {} preloaded", self.preloaded) }
        else { format!("PLE {} mapped / {} preloaded", self.mapped, self.preloaded) }
    }

    pub fn fallback(&mut self, error: &impl std::fmt::Display) {
        let what = if self.tables_preloaded() { "table" } else { "scale" };
        self.mapped += self.preloaded;
        self.preloaded = 0;
        self.preloaded_bytes = 0;
        self.reason = format!("{what} preload unavailable: {error}");
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
    cuteafd_ffi::host_memory()
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

    #[test]
    fn one_cap_takes_every_table_when_all_fit_else_the_scales() {
        let (scales, tables) = (6u64 << 30, 200u64 << 30);
        let place = |cap: u64, host: u64| MappedPlacement::resolve(4, 2, scales, cap, host).admit_tables(tables);
        let all = place(tables, tables);
        assert_eq!((all.label().as_str(), all.preloaded_bytes, all.reason.as_str()),
            ("PLE 4 preloaded", tables, "all tables admitted"));
        assert!(all.tables_preloaded());
        // A cap below the tables keeps today's scale decision, unchanged.
        let scale_only = place(tables - 1, tables);
        assert_eq!((scale_only.label().as_str(), scale_only.preloaded_bytes, scale_only.reason.as_str()),
            ("PLE 2 mapped / 2 preloaded", scales, "all scale tensors admitted"));
        assert!(!scale_only.tables_preloaded());
        // A cap that holds the tables but a host that cannot falls back to the scales and says why.
        let short = place(tables, tables - 1);
        assert_eq!((short.label().as_str(), short.preloaded_bytes), ("PLE 2 mapped / 2 preloaded", scales));
        assert_eq!(short.reason, "all tables exceed available host budget; all scale tensors admitted");
        let off = place(DEFAULT_CAP_BYTES, tables);
        assert_eq!((off.label().as_str(), off.reason.as_str()), ("PLE 4 mapped", "disabled"));
        // A runtime that cannot preload rows passes no table bytes: the scale policy alone.
        let unsupported = MappedPlacement::resolve(4, 2, scales, tables, tables).admit_tables(0);
        assert_eq!(unsupported.label(), "PLE 2 mapped / 2 preloaded");
        let mut failed = all;
        failed.fallback(&"injected");
        assert_eq!((failed.label().as_str(), failed.preloaded_bytes), ("PLE 4 mapped", 0));
        assert_eq!(failed.reason, "table preload unavailable: injected");
    }
}
