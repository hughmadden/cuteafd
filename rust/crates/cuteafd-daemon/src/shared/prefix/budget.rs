//! Startup-only pinned prefix sizing. GPU KV capacity and active request
//! admission are separate: these bytes store exact inactive snapshots.
use super::parse_bytes;
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::FamilyLayout;
use std::path::Path;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostBudget {
    Auto,
    Bytes(u64),
}

impl FromStr for HostBudget {
    type Err = String;
    fn from_str(text: &str) -> std::result::Result<Self, Self::Err> {
        match text.trim() {
            "auto" => Ok(Self::Auto),
            value => parse_bytes(value).map(Self::Bytes),
        }
    }
}

impl HostBudget {
    pub fn explicit_bytes(self) -> u64 {
        match self { Self::Auto => 0, Self::Bytes(bytes) => bytes }
    }

    pub fn enabled(self) -> bool {
        !matches!(self, Self::Bytes(0))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct HostMemory {
    pub total: u64,
    pub available: u64,
}

/// Bound against live available RAM and every readable cgroup ancestor's
/// remaining hard limit. Already-used mapped weights and staging are counted
/// by the operating system, rather than subtracted from total RAM twice.
pub(crate) fn host_memory() -> Result<HostMemory> {
    let text = std::fs::read_to_string("/proc/meminfo").context("reading host memory for automatic prefix budget")?;
    let mut memory = meminfo(&text)?;
    let cgroups = std::fs::read_to_string("/proc/self/cgroup").context("reading prefix-cache cgroup membership")?;
    for line in cgroups.lines() {
        let parts: Vec<_> = line.splitn(3, ':').collect();
        let [_, controllers, relative] = parts.as_slice() else { continue };
        let (root, limit_file, current_file) = if controllers.is_empty() {
            (Path::new("/sys/fs/cgroup"), "memory.max", "memory.current")
        } else if controllers.split(',').any(|c| c == "memory") {
            (Path::new("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes")
        } else { continue };
        let member = root.join(relative.trim_start_matches('/'));
        // The namespace may expose its current cgroup directly at the mount
        // root instead of reproducing the host membership path.
        for path in member.ancestors().take_while(|p| p.starts_with(root)).chain(std::iter::once(root)) {
            let limit = std::fs::read_to_string(path.join(limit_file));
            let current = std::fs::read_to_string(path.join(current_file));
            if let (Ok(limit), Ok(current)) = (limit, current) {
                apply_cgroup(&mut memory, &limit, &current)?;
            }
        }
    }
    Ok(memory)
}

/// Host RAM total for host-side caches sized from it (`EmbeddingCache::default_budget`).
pub(crate) fn host_total() -> Result<u64> { Ok(host_memory()?.total) }

fn meminfo(text: &str) -> Result<HostMemory> {
    let field = |name: &str| -> Result<u64> {
        let value = text.lines().find_map(|line| line.strip_prefix(name))
            .with_context(|| format!("/proc/meminfo lacks {name}"))?;
        let kib: u64 = value.split_whitespace().next().context("memory field without a value")?.parse()?;
        kib.checked_mul(1024).context("host memory field overflows bytes")
    };
    let (total, available) = (field("MemTotal:")?, field("MemAvailable:")?);
    ensure!(total > 0 && available <= total, "invalid MemTotal/MemAvailable for automatic prefix budget");
    Ok(HostMemory { total, available })
}

fn apply_cgroup(memory: &mut HostMemory, limit: &str, current: &str) -> Result<()> {
    if limit.trim() == "max" { return Ok(()) }
    let (limit, current): (u64, u64) = (limit.trim().parse()?, current.trim().parse()?);
    memory.total = memory.total.min(limit);
    memory.available = memory.available.min(limit.saturating_sub(current));
    Ok(())
}

/// Full-context KV for max concurrency plus four; independent of device retention counts.
pub(crate) fn retained_bytes(layout: FamilyLayout, concurrency: usize, max_context: usize, chunk: u64) -> Result<u64> {
    cuteafd_core::prefix::host_cache_required_bytes(max_context as u64, concurrency as u64,
        layout.page_rows as u64, layout.page_bytes as u64,
        layout.mark_bytes.max(1) as u64, layout.draft_bytes as u64, chunk)
        .context("invalid geometry or overflow for automatic host prefix budget")
}

pub(crate) fn automatic_bytes(required: u64, chunk: u64, memory: HostMemory, _headroom: u64) -> u64 {
    required.min(cuteafd_core::prefix::default_host_cache_bytes(memory.total)) / chunk * chunk
}

/// The live ledger is attribution only: those pinned bytes are already absent
/// from MemAvailable. Deferred scoring and the reserve still need admission.
pub(crate) fn admit(bytes: u64, deferred: u64, headroom: u64) -> Result<cuteafd_core::memory_layout::HostLayout> {
    use cuteafd_core::memory_layout::{Basis, Category, HostLayout, Item, ItemLifetime};
    let memory = host_memory()?;
    let snapshot = cuteafd_ffi::memory_ledger::snapshot();
    let mut items: Vec<_> = snapshot.by_scope(cuteafd_ffi::memory_ledger::Space::Pinned, -1).into_iter()
        .map(|(scope, bytes)| Item::new(Category::of_scope(scope), scope, "pinned", bytes as u64, Basis::Exact))
        .collect();
    items.push(Item::new(Category::Workspace, "scoring/full-logits", "FP32", deferred, Basis::Formula)
        .with_lifetime(ItemLifetime::PostReadyReserve));
    let ledger = HostLayout::new(memory.total, memory.available, headroom.max(memory.total / 10), bytes, items)
        .context("host admission byte count overflow")?;
    ensure!(ledger.shortfall_bytes == 0,
        "coordinator host memory cannot admit prefix cache {} plus deferred consumers {} and reserve {}: available {}, total {}, resident {}, shortfall {} bytes",
        bytes, ledger.deferred_bytes, ledger.reserve_bytes, memory.available, memory.total,
        ledger.resident_bytes, ledger.shortfall_bytes);
    tracing::info!(host_budget_bytes = bytes, host_ledger = %serde_json::to_string(&ledger)?,
        "coordinator host memory admitted");
    Ok(ledger)
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;

    #[test]
    fn auto_is_a_fixed_fraction_capped_at_32_gib_and_respects_cgroup_total() -> Result<()> {
        assert_eq!(cuteafd_core::prefix::default_host_cache_bytes(183 * GIB), 32 * GIB);
        assert_eq!(cuteafd_core::prefix::default_host_cache_bytes(64 * GIB), 16 * GIB);
        let mut memory = meminfo("MemTotal: 134217728 kB\nMemAvailable: 100663296 kB\n")?;
        assert_eq!(automatic_bytes(200 * GIB, GIB, memory, 8 * GIB), 32 * GIB);
        apply_cgroup(&mut memory, &(64 * GIB).to_string(), &(40 * GIB).to_string())?;
        assert_eq!(automatic_bytes(200 * GIB, GIB, memory, 8 * GIB), 16 * GIB);
        apply_cgroup(&mut memory, &(48 * GIB).to_string(), &(44 * GIB).to_string())?;
        assert_eq!(automatic_bytes(200 * GIB, GIB, memory, 8 * GIB), 12 * GIB);
        apply_cgroup(&mut memory, "max", "0")?;
        assert_eq!(memory.available, 4 * GIB);
        assert!(meminfo("MemTotal: 100 kB\n").is_err());
        assert!(apply_cgroup(&mut memory, "broken", "0").is_err());
        Ok(())
    }

    #[test]
    fn auto_never_allocates_more_than_full_retention_or_a_fractional_chunk() -> Result<()> {
        let layout = FamilyLayout { page_rows: 64, pages: 100, page_bytes: 512, mark_bytes: 140,
            draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT, mark_store: Default::default(), page_owners: Default::default() };
        // Six full-context snapshots: twelve pages in six chunks, six marks in one.
        assert_eq!(retained_bytes(layout, 2, 65, 1024)?, 7 * 1024);
        let memory = HostMemory { total: 32768, available: 32000 };
        assert_eq!(automatic_bytes(6 * 1024, 1024, memory, 0), 6 * 1024);
        assert_eq!(automatic_bytes(100 * 1024, 1024, memory, 0), 8 * 1024);
        assert!(retained_bytes(layout, usize::MAX, usize::MAX, 1024).is_err());
        Ok(())
    }
}
