//! Checked byte and reservation inputs shared by native startup admission.
use anyhow::{Context, Result, ensure};
use std::{fmt, str::FromStr, sync::OnceLock};
use cuteafd_loader::placement::ResidentWeights;

static RESIDENT_WEIGHT_LIMIT: OnceLock<ResidentWeights> = OnceLock::new();

pub(crate) fn set_resident_weight_limit(limit: ResidentWeights) -> Result<()> {
    ensure!(limit.0 > 0, "coordinator resident-weight limit must be positive");
    RESIDENT_WEIGHT_LIMIT.set(limit)
        .map_err(|_| anyhow::anyhow!("coordinator resident-weight limit is already installed"))
}

pub(crate) fn resident_weight_limit() -> ResidentWeights {
    RESIDENT_WEIGHT_LIMIT.get().copied().unwrap_or(ResidentWeights::DEFAULT_LIMIT)
}

/// A positive byte size (`--kv-pool-size`), through the one shared memory-size
/// parser (`cuteafd_core::memory_policy::Size`: B, MB, GB, MiB, GiB).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ByteSize(pub usize);
impl FromStr for ByteSize {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        let bytes = usize::try_from(value.parse::<cuteafd_core::memory_policy::Size>()?.0)?;
        ensure!(bytes > 0, "memory size rounds to zero bytes");
        Ok(Self(bytes))
    }
}
impl fmt::Display for ByteSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}B", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalLayers {
    Auto,
    /// Every routed-expert layer of the target model.
    All,
    Count(usize),
}
impl LocalLayers {
    /// Resolve against the target's routed-expert layer count, so `all` is an
    /// exact pin rather than a priority mode.
    pub(crate) fn onboard(self, routed: usize) -> cuteafd_loader::placement::Onboard {
        use cuteafd_loader::placement::Onboard;
        match self {
            Self::Auto => Onboard::Auto,
            Self::All => Onboard::Layers(routed),
            Self::Count(n) => Onboard::Layers(n),
        }
    }
}
impl FromStr for LocalLayers {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => return Ok(Self::Auto),
            "all" => return Ok(Self::All),
            _ => {}
        }
        let count: usize = value
            .parse()
            .context("RTX expert layers must be auto, all or 0..40")?;
        ensure!(count <= 40, "RTX expert layers must be auto, all or 0..40");
        Ok(Self::Count(count))
    }
}

/// Resolve one boot-time C before deriving queues, rings, marks or graph capacities.
pub(crate) fn serving_concurrency(requested: Option<usize>, native_lib: &std::path::Path,
    device: i32) -> Result<usize> {
    let concurrency = resolve_serving_concurrency(requested, || {
        if let Some(budget) = cuteafd_ffi::coordinator_gpu_budget() { return Ok(budget.0); }
        // SAFETY: the library is retained through the device-info query; no device work is launched.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(native_lib) }?;
        Ok(library.cuda_device_info(device)?.total_memory_bytes)
    })?;
    cuteafd_bench::context::set_resolved("max-sequences", &concurrency.to_string());
    tracing::info!(concurrency, "resolved boot-time serving concurrency");
    Ok(concurrency)
}

pub(crate) fn resolve_serving_concurrency(requested: Option<usize>, logical_bytes: impl FnOnce() -> Result<u64>)
    -> Result<usize> {
    match requested {
        Some(value) => Ok(value),
        None => Ok(cuteafd_core::serving_capacity::default_concurrency(logical_bytes()?) as usize),
    }
}

/// Admit the complete future host working set once, before any allocation.
/// Families supply table byte formulas summed across every local GPU, together
/// with the full prefix-cache quota (zero when disabled) and deferred logits.
/// Separate checks for tables and cache against the same live sample are not
/// reservations: they can each fit while their combined working set does not.
pub(crate) fn admit_host_memory(cache: u64, tables: &[(&str, u64)], deferred_logits: u64, headroom: u64)
    -> Result<cuteafd_core::memory_layout::HostLayout> {
    admit_host_memory_with_resident_formulas(cache, tables, &[], deferred_logits, headroom)
}

/// Already allocated native payload and pageable workspace have one formula
/// owner. Replace their pinned attribution, rather than charging them twice.
pub(crate) fn admit_host_memory_with_resident_formulas(cache: u64, tables: &[(&str, u64)],
    formulas: &[(&str, u64)], deferred_logits: u64, headroom: u64)
    -> Result<cuteafd_core::memory_layout::HostLayout> {
    let memory = cuteafd_loader::formats::mapped_preload::host_memory()?;
    let snapshot = cuteafd_ffi::memory_ledger::snapshot();
    let mut resident = host_resident_items(pinned_by_label(&snapshot), formulas);
    // One singleton serves every mapped table and GPU owner in this process.
    // Its reusable buffers are lazy, so retain the full future reservation.
    append_table_reactor(&mut resident, usize::from(
        cuteafd_loader::formats::mapped_table::table_reactor_started()));
    let ledger = host_admission(memory, resident, cache, tables, deferred_logits, headroom)?;
    tracing::info!(host_budget_bytes = cache, host_ledger = %serde_json::to_string(&ledger)?,
        "coordinator host memory admitted");
    Ok(ledger)
}

/// Pinned bytes by ledger label, each with its kind's planner category: the
/// recorded tag when there is one, else the legacy scope-root mapping.
fn pinned_by_label(snapshot: &cuteafd_ffi::memory_ledger::Snapshot)
    -> std::collections::BTreeMap<&'static str, (cuteafd_core::memory_layout::Category, usize)> {
    use cuteafd_core::memory_layout::Category;
    let mut out = std::collections::BTreeMap::new();
    for row in snapshot.rows.iter().filter(|r| r.key.space == cuteafd_ffi::memory_ledger::Space::Pinned) {
        let category = row.key.tag.kind.category();
        let entry = out.entry(row.key.scope).or_insert((category, 0));
        entry.1 += row.bytes;
    }
    out
}

fn append_table_reactor(items: &mut Vec<cuteafd_core::memory_layout::Item>, mapped_tables: usize) {
    use cuteafd_core::memory_layout::{Basis, Category, Item, ItemLifetime};
    let bytes = cuteafd_loader::formats::mapped_table::table_reactor_workspace_bytes(mapped_tables);
    if bytes > 0 {
        items.push(Item::new(Category::Workspace, "workspace/table-read-reactor", "process-wide", bytes as u64,
            Basis::Formula).with_lifetime(ItemLifetime::PostReadyReserve));
    }
}

fn host_resident_items(pinned: std::collections::BTreeMap<&'static str, (cuteafd_core::memory_layout::Category, usize)>,
    formulas: &[(&str, u64)]) -> Vec<cuteafd_core::memory_layout::Item> {
    use cuteafd_core::memory_layout::{Basis, Category, Item};
    let mut items: Vec<_> = pinned.into_iter()
        .filter(|(scope, _)| !formulas.iter().any(|&(name, _)| name == *scope))
        .map(|(scope, (category, bytes))| Item::new(category, scope, "pinned", bytes as u64, Basis::Exact))
        .collect();
    items.extend(formulas.iter().map(|&(name, bytes)| Item::new(Category::Staging, name,
        "pinned payload + pageable sort tuples", bytes, Basis::Formula)));
    items
}

fn host_admission((total, available): (u64, u64),
    mut resident: Vec<cuteafd_core::memory_layout::Item>, cache: u64,
    tables: &[(&str, u64)], deferred_logits: u64, headroom: u64)
    -> Result<cuteafd_core::memory_layout::HostLayout> {
    use cuteafd_core::memory_layout::{Basis, Category, HostLayout, Item, ItemLifetime, host_table_item};
    // Resident bytes are already missing from live availability. Only these
    // future consumers and the reserve are charged against it again.
    resident.extend(tables.iter().map(|&(name, bytes)| host_table_item(name, bytes)));
    resident.push(Item::new(Category::Workspace, "scoring/full-logits", "FP32", deferred_logits, Basis::Formula)
        .with_lifetime(ItemLifetime::PostReadyReserve));
    let ledger = HostLayout::new(total, available, headroom.max(total / 10), cache, resident)
        .context("host admission byte count overflow")?;
    ensure!(ledger.shortfall_bytes == 0,
        "coordinator host memory cannot admit host cache {} plus deferred consumers {} and reserve {}: available {}, total {}, resident {}, shortfall {} bytes",
        cache, ledger.deferred_bytes, ledger.reserve_bytes, available, total,
        ledger.resident_bytes, ledger.shortfall_bytes);
    Ok(ledger)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::memory_layout::{Basis, Category, Item, ItemLifetime};

    #[test]
    fn boot_concurrency_uses_logical_capacity_only_when_unset() -> Result<()> {
        for bytes in [30 << 30, 32 << 30, (32 << 30) + 1, 96 << 30] {
            assert_eq!(resolve_serving_concurrency(None, || Ok(bytes))?,
                cuteafd_core::serving_capacity::default_concurrency(bytes) as usize);
        }
        for requested in [0, 4, 8, 16, 65] {
            assert_eq!(resolve_serving_concurrency(Some(requested), || panic!("explicit C must not query CUDA"))?,
                requested, "family admission validates the resolved value");
        }
        assert!(resolve_serving_concurrency(None, || anyhow::bail!("device query failed"))
            .unwrap_err().to_string().contains("device query failed"));
        Ok(())
    }

    #[test]
    fn reactor_reservation_is_once_per_process_not_per_table() -> Result<()> {
        for count in [0, 1, 2, 8] {
            let mut items = Vec::new();
            append_table_reactor(&mut items, count);
            let ledger = host_admission((2 << 30, 1 << 30), items, 0, &[], 0, 0)?;
            assert_eq!(ledger.items.iter().filter(|item| item.group == "workspace/table-read-reactor").count(), usize::from(count > 0));
            assert_eq!(ledger.deferred_bytes, if count == 0 { 0 } else { 18 << 20 });
        }
        Ok(())
    }

    #[test]
    fn mapped_table_host_workspace_remains_deferred_at_every_depth() -> Result<()> {
        for depth in 1..=4 {
            for row_bytes in [160, 320] {
                let shape = cuteafd_loader::formats::mapped_table::MappedStaging::new(
                    2752 * 16, depth, 64 * 16, row_bytes)?;
                let payload = shape.slot_bytes().sum::<usize>() as u64;
                let workspace = shape.workspace_bytes as u64;
                let items = host_resident_items(std::collections::BTreeMap::from([
                    ("mapped-table", (Category::Staging, payload as usize))]), &[]);
                let demands = [("staging/mapped-table-worker", workspace)];
                let ledger = host_admission((2 << 30, 1 << 30), items.clone(), 0, &demands, 0, 0)?;
                assert_eq!((ledger.resident_bytes, ledger.deferred_bytes), (payload, workspace));
                assert_eq!(ledger.resident_bytes + ledger.deferred_bytes, shape.total_bytes as u64);
                let worker = ledger.items.iter().find(|item| item.group == demands[0].0).unwrap();
                assert_eq!((worker.bytes, worker.lifetime), (workspace, ItemLifetime::PostReadyReserve));
                let available = (2u64 << 30) / 10 + workspace;
                assert!(host_admission((2 << 30, available), items.clone(), 0, &demands, 0, 0).is_ok());
                assert!(host_admission((2 << 30, available - 1), items, 0, &demands, 0, 0).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn engram_host_formula_replaces_payload_attribution_without_double_charge() -> Result<()> {
        for k in 1..=4 {
            let rows = 8192usize.div_ceil(k.max(2)).div_ceil(64) * 64;
            let shape = cuteafd_loader::EngramStagingShape::new(rows, k)?;
            let formulas = [("v41/engram-prefill", shape.host_resident_bytes() as u64),
                ("v41/engram-decode", shape.decode_host_resident_bytes() as u64)];
            let pinned = std::collections::BTreeMap::from([
                ("v41/engram-prefill", (Category::Staging, shape.host_slots() * rows * 6337)),
                ("v41/engram-decode", (Category::Staging, shape.decode_slots() * shape.decode_rows() * 6337)),
                ("unrelated/pinned", (Category::Staging, 7))]);
            let items = host_resident_items(pinned, &formulas);
            assert_eq!(items.len(), 3);
            for &(name, bytes) in &formulas {
                let entry = items.iter().find(|item| item.group == name).unwrap();
                assert_eq!((entry.bytes, entry.basis, entry.lifetime),
                    (bytes, Basis::Formula, ItemLifetime::ReadyResident));
            }
            let workspace = [("v41/engram-prefill-worker", shape.host_workspace_bytes() as u64),
                ("v41/engram-decode-worker", shape.decode_host_workspace_bytes() as u64)];
            let workspace_bytes = workspace.iter().map(|f| f.1).sum::<u64>();
            let ledger = host_admission((2 << 30, 1 << 30), items.clone(), 0, &workspace, 0, 0)?;
            assert_eq!(ledger.resident_bytes, formulas.iter().map(|f| f.1).sum::<u64>() + 7);
            assert_eq!(ledger.deferred_bytes, workspace_bytes);
            assert_eq!(ledger.resident_bytes + ledger.deferred_bytes,
                (shape.host_bytes() + shape.decode_host_bytes()) as u64 + 7);
            // Worker vectors/rings do not exist at readiness: they must consume
            // future availability, not be mislabeled already-resident bytes.
            assert!(host_admission((2 << 30, workspace_bytes - 1), items, 0, &workspace, 0, 0).is_err());
        }
        Ok(())
    }
    #[test]
    fn named_pinned_tables_are_admitted_without_a_prefix_cache() -> Result<()> {
        let memory = (1000, 800);
        let resident = vec![Item::new(Category::Weights, "resident/weight", "pinned", 200, Basis::Exact)];
        let ledger = host_admission(memory, resident, 0,
            &[("synthetic/ordered-table", 300), ("synthetic/other-gpu", 200)], 100, 0)?;
        assert_eq!((ledger.host_budget_bytes, ledger.resident_bytes, ledger.deferred_bytes), (0, 200, 600));
        assert_eq!((ledger.reserve_bytes, ledger.headroom_bytes, ledger.shortfall_bytes), (100, 100, 0));
        for (name, bytes) in [("synthetic/ordered-table", 300), ("synthetic/other-gpu", 200)] {
            let item = ledger.items.iter().find(|item| item.group == name).unwrap();
            assert_eq!((item.category, item.bytes, item.lifetime),
                (Category::Staging, bytes, ItemLifetime::PostReadyReserve));
        }
        let encoded = serde_json::to_value(&ledger)?;
        assert_eq!(serde_json::from_value::<cuteafd_core::memory_layout::HostLayout>(encoded)?, ledger);
        Ok(())
    }

    #[test]
    fn host_admission_charges_future_tables_and_prefix_once_but_not_resident_twice() -> Result<()> {
        let memory = (1000, 700);
        let resident = vec![Item::new(Category::Staging, "resident/table", "pinned", 300, Basis::Exact)];
        let ledger = host_admission(memory, resident, 100, &[("synthetic/table", 200)], 100, 150)?;
        assert_eq!((ledger.resident_bytes, ledger.deferred_bytes, ledger.headroom_bytes), (300, 300, 150));
        assert_eq!(ledger.reserve_bytes, 150);
        assert_eq!(host_admission(memory, vec![], 100, &[], 100, 0)?.deferred_bytes, 100);
        Ok(())
    }

    #[test]
    fn tables_and_prefix_must_be_admitted_in_one_complete_working_set() -> Result<()> {
        let memory = (1000, 600);
        let tables = [("synthetic/table", 300)];
        assert!(host_admission(memory, vec![], 300, &[], 0, 0).is_ok());
        assert!(host_admission(memory, vec![], 0, &tables, 0, 0).is_ok());
        let error = host_admission(memory, vec![], 300, &tables, 0, 0).unwrap_err();
        assert!(error.to_string().contains("shortfall 100 bytes"), "{error}");
        Ok(())
    }

    #[test]
    fn host_admission_refuses_available_total_and_arithmetic_shortfalls() {
        let memory = (1000, 300);
        let error = host_admission(memory, vec![], 0, &[("synthetic/table", 201)], 0, 0).unwrap_err();
        assert!(error.to_string().contains("shortfall 1 bytes"), "{error}");
        let resident = vec![Item::new(Category::Staging, "resident/table", "pinned", 800, Basis::Exact)];
        let error = host_admission((1000, 1000), resident, 0,
            &[("synthetic/table", 101)], 0, 0).unwrap_err();
        assert!(error.to_string().contains("shortfall 1 bytes"), "{error}");
        let memory = (u64::MAX, u64::MAX);
        for (cache, tables, logits, headroom) in [
            (0, vec![("synthetic/a", u64::MAX), ("synthetic/b", 1)], 0, 0),
            (1, vec![("synthetic/a", u64::MAX)], 0, 0),
            (0, vec![("synthetic/a", u64::MAX)], 1, 0),
            (0, vec![], 1, u64::MAX),
        ] {
            let error = host_admission(memory, vec![], cache, &tables, logits, headroom).unwrap_err();
            assert!(error.to_string().contains("overflow"), "{error}");
        }
    }
    #[test]
    fn local_layers_parse_auto_all_and_exact_counts() {
        use cuteafd_loader::placement::Onboard;
        assert_eq!("all".parse::<LocalLayers>().unwrap().onboard(40), Onboard::Layers(40));
        assert_eq!("auto".parse::<LocalLayers>().unwrap().onboard(40), Onboard::Auto);
        assert_eq!("40".parse::<LocalLayers>().unwrap().onboard(40), Onboard::Layers(40));
        for value in ["41", "-1", "max", "50%"] {
            assert!(value.parse::<LocalLayers>().is_err(), "{value}");
        }
    }

    #[test]
    fn sizes_and_reservations_validate_units_and_overflow() {
        for (value, bytes) in [
            ("37GB", 37_000_000_000),
            ("1.5GiB", 1_610_612_736),
            ("1MB", 1_000_000),
            ("1MiB", 1_048_576),
            ("123", 123),
        ] {
            assert_eq!(value.parse::<ByteSize>().unwrap().0, bytes);
        }
        for value in [
            "0",
            "-1GB",
            "NaN",
            "1TB",
            "10%",
            "1.0000001GB",
            "18446744073709551616GB",
        ] {
            assert!(value.parse::<ByteSize>().is_err(), "{value}");
        }
    }
}
