//! Checked byte and reservation inputs shared by native startup admission.
use anyhow::{Context, Result, ensure};
use std::{fmt, str::FromStr};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ByteSize(pub usize);
#[derive(Clone, Copy, Debug)]
pub(crate) enum Reservation {
    Bytes(ByteSize),
    Percent(u64), // millionths of one percent
}
fn decimal(value: &str) -> Result<u64> {
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    ensure!(
        !whole.is_empty()
            && whole.bytes().all(|c| c.is_ascii_digit())
            && fraction.len() <= 6
            && fraction.bytes().all(|c| c.is_ascii_digit()),
        "expected a positive decimal with at most six fractional digits"
    );
    let whole: u64 = whole.parse()?;
    let fractional: u64 = if fraction.is_empty() {
        0
    } else {
        fraction.parse()?
    };
    let scaled = whole
        .checked_mul(1_000_000)
        .and_then(|v| v.checked_add(fractional * 10u64.pow(6 - fraction.len() as u32)))
        .context("memory size overflow")?;
    ensure!(scaled > 0, "memory size must be positive");
    Ok(scaled)
}
impl FromStr for ByteSize {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        let value = value.trim();
        let end = value
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(value.len());
        let amount = decimal(&value[..end])?;
        let unit = value[end..].trim().to_ascii_lowercase();
        let multiplier: u128 = match unit.as_str() {
            "" | "b" => 1,
            "mb" => 1_000_000,
            "gb" => 1_000_000_000,
            "mib" => 1_048_576,
            "gib" => 1_073_741_824,
            _ => anyhow::bail!("memory size unit must be B, MB, GB, MiB or GiB"),
        };
        let bytes = usize::try_from(u128::from(amount) * multiplier / 1_000_000)?;
        ensure!(bytes > 0, "memory size rounds to zero bytes");
        Ok(Self(bytes))
    }
}
impl fmt::Display for ByteSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}B", self.0)
    }
}
impl FromStr for Reservation {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        if let Some(percent) = value.trim().strip_suffix('%') {
            let percent = decimal(percent.trim())?;
            ensure!(
                percent <= 100_000_000,
                "memory percentage must be in (0,100]"
            );
            Ok(Self::Percent(percent))
        } else {
            Ok(Self::Bytes(value.parse()?))
        }
    }
}
impl Reservation {
    pub(crate) fn bytes(self, total: usize) -> Result<usize> {
        let bytes = match self {
            Self::Bytes(size) => size.0,
            Self::Percent(percent) => (total as u128 * u128::from(percent) / 100_000_000) as usize,
        };
        ensure!(
            bytes > 0 && bytes <= total,
            "memory reservation exceeds device total or rounds to zero"
        );
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalLayers {
    Auto,
    ExpertsFirst,
    Count(usize),
}
impl LocalLayers {
    pub(crate) fn onboard(self) -> cuteafd_loader::placement::Onboard {
        use cuteafd_loader::placement::{Onboard, EXPERTS_FIRST_POOL_FLOOR};
        match self {
            Self::Auto => Onboard::Auto,
            Self::ExpertsFirst => Onboard::ExpertsFirst { pool_floor: EXPERTS_FIRST_POOL_FLOOR },
            Self::Count(n) => Onboard::Layers(n),
        }
    }
}
impl FromStr for LocalLayers {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => return Ok(Self::Auto),
            "max" => return Ok(Self::ExpertsFirst),
            _ => {}
        }
        let count: usize = value
            .parse()
            .context("RTX expert layers must be auto, max or 0..40")?;
        ensure!(count <= 40, "RTX expert layers must be auto, max or 0..40");
        Ok(Self::Count(count))
    }
}

/// Admit all future pinned tables for this coordinator process before allocation.
/// Families supply the names and byte formulas, summed across every local GPU.
/// This check is independent of whether retained-prefix caching is enabled.
pub(crate) fn admit_pinned_tables(tables: &[(&str, u64)], deferred_logits: u64, headroom: u64)
    -> Result<cuteafd_core::memory_layout::HostLayout> {
    admit_host_memory(0, tables, deferred_logits, headroom)
}

pub(crate) fn admit_host_memory(cache: u64, tables: &[(&str, u64)], deferred_logits: u64, headroom: u64)
    -> Result<cuteafd_core::memory_layout::HostLayout> {
    use cuteafd_core::memory_layout::{Basis, Category, Item};
    let memory = crate::shared::prefix::budget::host_memory()?;
    let snapshot = cuteafd_ffi::memory_ledger::snapshot();
    let resident = snapshot.by_scope(cuteafd_ffi::memory_ledger::Space::Pinned, -1).into_iter()
        .map(|(scope, bytes)| Item::new(Category::of_scope(scope), scope, "pinned", bytes as u64, Basis::Exact))
        .collect();
    let ledger = host_admission(memory, resident, cache, tables, deferred_logits, headroom)?;
    tracing::info!(host_budget_bytes = cache, host_ledger = %serde_json::to_string(&ledger)?,
        "coordinator host memory admitted");
    Ok(ledger)
}

fn host_admission(memory: crate::shared::prefix::budget::HostMemory,
    mut resident: Vec<cuteafd_core::memory_layout::Item>, cache: u64,
    tables: &[(&str, u64)], deferred_logits: u64, headroom: u64)
    -> Result<cuteafd_core::memory_layout::HostLayout> {
    use cuteafd_core::memory_layout::{Basis, Category, HostLayout, Item, ItemLifetime, host_table_item};
    // Resident bytes are already missing from live availability. Only these
    // future consumers and the reserve are charged against it again.
    resident.extend(tables.iter().map(|&(name, bytes)| host_table_item(name, bytes)));
    resident.push(Item::new(Category::Workspace, "scoring/full-logits", "FP32", deferred_logits, Basis::Formula)
        .with_lifetime(ItemLifetime::PostReadyReserve));
    let ledger = HostLayout::new(memory.total, memory.available, headroom.max(memory.total / 10), cache, resident)
        .context("host admission byte count overflow")?;
    ensure!(ledger.shortfall_bytes == 0,
        "coordinator host memory cannot admit host cache {} plus deferred consumers {} and reserve {}: available {}, total {}, resident {}, shortfall {} bytes",
        cache, ledger.deferred_bytes, ledger.reserve_bytes, memory.available, memory.total,
        ledger.resident_bytes, ledger.shortfall_bytes);
    Ok(ledger)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefix::budget::HostMemory;
    use cuteafd_core::memory_layout::{Basis, Category, Item, ItemLifetime};

    #[test]
    fn named_pinned_tables_are_admitted_without_a_prefix_cache() -> Result<()> {
        let memory = HostMemory { total: 1000, available: 800 };
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
        let memory = HostMemory { total: 1000, available: 700 };
        let resident = vec![Item::new(Category::Staging, "resident/table", "pinned", 300, Basis::Exact)];
        let ledger = host_admission(memory, resident, 100, &[("synthetic/table", 200)], 100, 150)?;
        assert_eq!((ledger.resident_bytes, ledger.deferred_bytes, ledger.headroom_bytes), (300, 300, 150));
        assert_eq!(ledger.reserve_bytes, 150);
        assert_eq!(host_admission(memory, vec![], 100, &[], 100, 0)?.deferred_bytes, 100);
        Ok(())
    }

    #[test]
    fn host_admission_refuses_available_total_and_arithmetic_shortfalls() {
        let memory = HostMemory { total: 1000, available: 300 };
        let error = host_admission(memory, vec![], 0, &[("synthetic/table", 201)], 0, 0).unwrap_err();
        assert!(error.to_string().contains("shortfall 1 bytes"), "{error}");
        let resident = vec![Item::new(Category::Staging, "resident/table", "pinned", 800, Basis::Exact)];
        let error = host_admission(HostMemory { total: 1000, available: 1000 }, resident, 0,
            &[("synthetic/table", 101)], 0, 0).unwrap_err();
        assert!(error.to_string().contains("shortfall 1 bytes"), "{error}");
        let memory = HostMemory { total: u64::MAX, available: u64::MAX };
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
    fn local_layers_max_uses_shared_experts_first_policy() {
        use cuteafd_loader::placement::{Onboard, EXPERTS_FIRST_POOL_FLOOR};
        assert_eq!("max".parse::<LocalLayers>().unwrap().onboard(),
            Onboard::ExpertsFirst { pool_floor: EXPERTS_FIRST_POOL_FLOOR });
        assert_eq!("auto".parse::<LocalLayers>().unwrap().onboard(), Onboard::Auto);
        assert_eq!("40".parse::<LocalLayers>().unwrap().onboard(), Onboard::Layers(40));
        for value in ["41", "-1", "all", "50%"] {
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
        assert_eq!(
            "87.5%"
                .parse::<Reservation>()
                .unwrap()
                .bytes(8_000_000_000)
                .unwrap(),
            7_000_000_000
        );
        for value in ["0%", "100.1%", "inf%"] {
            assert!(value.parse::<Reservation>().is_err());
        }
    }
}
