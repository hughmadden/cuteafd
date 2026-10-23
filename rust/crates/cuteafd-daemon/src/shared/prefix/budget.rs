//! Startup-only pinned prefix sizing. GPU KV capacity and active request
//! admission are separate: these bytes store exact inactive snapshots.
use super::parse_bytes;
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::FamilyLayout;
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

/// Shared MemAvailable/cgroup working-set inventory for prefix admission.
pub(crate) fn host_memory() -> Result<HostMemory> {
    let (total, available) = cuteafd_loader::formats::mapped_preload::host_memory()?;
    Ok(HostMemory { total, available })
}

/// Host RAM total for host-side caches sized from it (`EmbeddingCache::default_budget`).
pub(crate) fn host_total() -> Result<u64> { Ok(host_memory()?.total) }

/// Full-context KV for max concurrency plus four; independent of device retention counts.
pub(crate) fn retained_bytes(layout: FamilyLayout, concurrency: usize, max_context: usize, chunk: u64) -> Result<u64> {
    cuteafd_core::prefix::host_cache_required_bytes(max_context as u64, concurrency as u64,
        layout.page_rows as u64, layout.page_bytes as u64,
        layout.mark_bytes.max(1) as u64, layout.draft_bytes as u64, chunk)
        .context("invalid geometry or overflow for automatic host prefix budget")
}

pub(crate) fn resolve_bytes(layout: FamilyLayout, explicit: Option<u64>, concurrency: usize,
    max_context: usize, chunk: u64, memory: HostMemory) -> Result<u64> {
    cuteafd_core::prefix::HostCacheGeometry { unit_rows: layout.page_rows as u64,
        page_bytes: layout.page_bytes as u64, mark_bytes: layout.mark_bytes as u64,
        draft_bytes: layout.draft_bytes as u64 }
        .resolve(explicit, max_context as u64, concurrency as u64, memory.total, chunk)
        .context("invalid geometry or overflow for automatic host prefix budget")
}

pub(crate) fn eviction_headroom(layout: FamilyLayout, tokens: u64) -> Result<[u64; 4]> {
    let [pages, marks, draft] = cuteafd_core::prefix::HostCacheGeometry {
        unit_rows: layout.page_rows as u64, page_bytes: layout.page_bytes as u64,
        mark_bytes: layout.mark_bytes as u64, draft_bytes: layout.draft_bytes as u64,
    }.eviction_headroom_bytes(tokens).context("host eviction headroom overflow")?;
    Ok([pages, marks, draft, 0])
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
    fn shared_resolver_charges_two_marks_per_conversation() -> Result<()> {
        // Full-context pages and two marks per conversation, one coordinator, C16,
        // 128 GiB effective RAM, 256 MiB chunks. Default KV representations:
        // V4.1 native+dSpark, MiMo Pro INT8/full + BF16/SWA + warm DFlash,
        // Qwen FP8+MTP+PLE, GLM FP8, V4 Flash native+three dSpark stages.
        // The 4K extent also checks uncapped arithmetic for the large families.
        let fixtures = [
            ("v41-dspark", 512, 455_680, 2_922_816, 1_048_576),
            ("mimo-dflash", 64, 1_843_200, 60_293_128, 1_048_576),
            ("qwen-mtp", 256, 4_526_080, 115_642_368, 262_144),
            ("glm", 64, 3_452_160, 0, 1_048_576),
            ("v4-dspark", 256, 998_208, 27_850_752, 1_048_576),
        ];
        let memory = HostMemory { total: 128 * GIB, available: 128 * GIB };
        let chunk = 256u64 << 20;
        for (family, page_rows, page_bytes, mark_bytes, context) in fixtures {
            let layout = FamilyLayout { page_rows, pages: 8192, page_bytes, mark_bytes,
                draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT,
                mark_store: Default::default(), page_owners: Default::default() };
            for context in [context, 4096] {
                let class = |count: u64, bytes: u64| {
                    count.div_ceil(chunk / bytes) * chunk
                };
                let required = class((context as u64).div_ceil(page_rows as u64) * 20, page_bytes as u64)
                    + class(40, mark_bytes.max(1) as u64);
                let expected = required.min(32 * GIB);
                assert_eq!(resolve_bytes(layout, None, 16, context, chunk, memory)?, expected, "{family}, {context}");
            }
        }
        Ok(())
    }

    #[test]
    fn planner_matches_runtime_for_bf16_qwen_with_native_mtp() -> Result<()> {
        use cuteafd_loader::{families::qwen4::{Qwen4Attention, Qwen4Config, Qwen4KvCache},
            plan::{testing, layout::LayoutOptions, PlanInput, PlanOptions, MediaMode}};
        let dir = tempfile::tempdir()?;
        let mut config = testing::qwen4_config(4);
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        testing::write_snapshot(dir.path(), &config, &testing::qwen4_coordinator(&config, true), None);
        let cfg = Qwen4Config::from_hf(&config)?;
        let input = PlanInput::open(dir.path())?;
        let memory = HostMemory { total: 128 * GIB, available: 128 * GIB };
        let context = 1 << 18;
        let concurrency = 1;
        let mut quotas = Vec::new();
        for (mtp, kv) in [(false, Qwen4KvCache::Fp8), (true, Qwen4KvCache::Fp8), (true, Qwen4KvCache::Bf16)] {
            let options = LayoutOptions { host_memory: Some((memory.total, memory.available)),
                host_cache_bytes: None, context_tokens: context, concurrency,
                native_mtp_layers: usize::from(mtp), qwen_kv: kv, ..Default::default() };
            let report = input.plan(&PlanOptions { layout: Some(options), vision: MediaMode::Off,
                ..Default::default() })?;
            let full = cfg.attention.iter().filter(|&&a| a == Qwen4Attention::Full).count();
            // The runtime allocates these owner-local GDN regions and paged
            // buffers; MTP adds its full-attention pool, not its pending stash.
            let gdn = crate::shared::qwen::GdnLayout::new(cfg.gdn_geometry())?;
            let marks = (cfg.layers - full) * (gdn.conv_slot_bytes + gdn.state_slot_bytes)
                + if cfg.ple_layers.is_empty() { 0 } else { 9 * cfg.hc_count * cfg.hidden * 2 };
            let layout = FamilyLayout { page_rows: 256, pages: 4096,
                page_bytes: (full + usize::from(mtp)) * (256 * (kv.record_bytes(cfg.kv_heads, cfg.head_dim) + 256) + 64 * 256),
                mark_bytes: marks, draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT,
                mark_store: Default::default(), page_owners: Default::default() };
            let runtime = resolve_bytes(layout, None, concurrency as usize, context as usize, 256 << 20, memory)?;
            assert_eq!(report.host_budget_bytes, runtime, "MTP={mtp}, KV={kv:?}, report={report:?}");
            quotas.push(runtime);
        }
        assert!(quotas[0] < quotas[1] && quotas[1] < quotas[2], "geometry must not be hidden by the cap: {quotas:?}");
        Ok(())
    }

    #[test]
    fn auto_is_a_fixed_fraction_capped_at_32_gib_and_respects_cgroup_total() -> Result<()> {
        assert_eq!(cuteafd_core::prefix::default_host_cache_bytes(183 * GIB), 32 * GIB);
        assert_eq!(cuteafd_core::prefix::default_host_cache_bytes(64 * GIB), 16 * GIB);
        let layout = FamilyLayout { page_rows: 1, pages: 1, page_bytes: GIB as usize, mark_bytes: 0,
            draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT, mark_store: Default::default(), page_owners: Default::default() };
        let resolve = |memory| resolve_bytes(layout, None, 16, 200, GIB, memory).unwrap();
        assert_eq!(resolve(HostMemory { total: 128 * GIB, available: 96 * GIB }), 32 * GIB);
        assert_eq!(resolve(HostMemory { total: 64 * GIB, available: 24 * GIB }), 16 * GIB);
        assert_eq!(resolve(HostMemory { total: 48 * GIB, available: 4 * GIB }), 12 * GIB);
        Ok(())
    }

    #[test]
    fn auto_never_allocates_more_than_full_retention_or_a_fractional_chunk() -> Result<()> {
        let layout = FamilyLayout { page_rows: 64, pages: 100, page_bytes: 512, mark_bytes: 140,
            draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT, mark_store: Default::default(), page_owners: Default::default() };
        // Six conversations: twelve pages in six chunks and twelve marks in two.
        assert_eq!(retained_bytes(layout, 2, 65, 1024)?, 8 * 1024);
        let memory = HostMemory { total: 32768, available: 32000 };
        assert_eq!(resolve_bytes(layout, None, 1, 65, 1024, memory)?, 7 * 1024);
        assert_eq!(resolve_bytes(layout, None, 16, 4096, 1024, memory)?, 8 * 1024);
        assert!(retained_bytes(layout, usize::MAX, usize::MAX, 1024).is_err());
        Ok(())
    }
}
