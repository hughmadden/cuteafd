//! Startup-only source-pool sizing. No allocation policy runs in the token loop.
use crate::families::deepseek_v41::v41_backbone_cache::BackboneCache;
use anyhow::{ensure, Context, Result};
use std::{fmt, str::FromStr};

pub(crate) mod distributed;

// Aggregate source capacity is independent of the retained snapshot count.
const RETAINED_CONTEXTS: usize = 2;
// Extra pages cover retained partial tails and active copy-on-write frontiers.
const MAX_GROUPS: usize = 131_072;
const GROUP_BYTES: usize = 5 * 256 * (68 + cuteafd_ffi::V41Kv::COMPRESSED_ROW_BYTES);
// Request scratch and graph/runtime allocations. Snapshot arenas are already live.
// Kept outside the eagerly allocated cache; this is not a CUDA process quota.
pub(super) const RUNTIME_HEADROOM: usize = 2 * 1024 * 1024 * 1024;

/// Resolve the planner's token budget after fixed owners are live (or charged
/// to a synthetic free-memory sample for deferred TP2 experts). The byte plan
/// is then passed to the existing allocator, which verifies its own formula.
pub(super) fn planned_pool_size(args: &crate::cli::NativeServeArgs,
    memory: &[(usize, usize)]) -> Result<Option<ByteSize>> {
    let Some(requested) = args.pool_tokens else { return Ok(args.kv_pool_size); };
    let automatic = requested == 0;
    let target = if automatic { 14 * 1_048_576 }
        else { requested };
    let config: serde_json::Value = serde_json::from_reader(std::fs::File::open(args.snapshot.join("config.json"))?)?;
    let available: Vec<u64> = memory.iter().enumerate().map(|(gpu, &(free, total))| {
        ensure!(total > 0 && free <= total, "invalid GPU {gpu} memory information");
        let policy_ceiling = args.memory_reservation.map(|r| r.bytes(total)).transpose()?.unwrap_or(total);
        let ceiling = if automatic { policy_ceiling.min((total as u128 * 97 / 100) as usize) }
            else { policy_ceiling };
        // Keep >=3 GiB even after deferred local expert placement consumes
        // its allowed budget. Existing explicit/non-planner paths are intact.
        let runtime = if automatic { 3usize << 30 } else if memory.len() == 1 { RUNTIME_HEADROOM }
            else { distributed::RUNTIME_HEADROOM };
        Ok(ceiling.checked_sub(total - free).and_then(|n| n.checked_sub(runtime))
            .with_context(|| format!("GPU {gpu} planner leaves no room after fixed owners and runtime reserve"))? as u64)
    }).collect::<Result<_>>()?;
    let groups = cuteafd_loader::serving_capacity::deepseek_v41_pool_groups(&config,
        args.concurrency as u64, args.prefix_cache_entries as u64, &available, target, automatic, args.tp2_attention)?;
    let bytes = usize::try_from(groups)?.checked_mul(GROUP_BYTES).context("planner KV byte overflow")?;
    tracing::info!(requested_pool_tokens=requested, admitted_pool_tokens=groups.saturating_sub(
        args.concurrency as u64 + 2 * args.prefix_cache_entries as u64) * 512,
        groups, global_bytes=bytes, ?available, "V4.1 planner source-pool admission");
    Ok(Some(ByteSize(bytes)))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ByteSize(pub usize);
#[derive(Clone, Copy, Debug)]
pub(crate) enum HostBudget { Auto, Bytes(u64) }
impl FromStr for HostBudget {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value.trim() {
            "auto" => Ok(Self::Auto),
            "0" => Ok(Self::Bytes(0)),
            value => Ok(Self::Bytes(value.parse::<ByteSize>()?.0 as u64)),
        }
    }
}
impl HostBudget {
    pub fn explicit_bytes(self) -> u64 {
        match self { Self::Auto => 0, Self::Bytes(bytes) => bytes }
    }
}
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
    fn bytes(self, total: usize) -> Result<usize> {
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

#[derive(Debug)]
pub(super) struct PoolPlan {
    pub pages: [usize; 4],
    pub global_bytes: usize,
    pub cache_bytes: usize,
    pub occupied_before: usize,
    pub reservation_bytes: usize,
}
impl PoolPlan {
    fn from_groups(
        slots: usize,
        groups: usize,
        occupied_before: usize,
        reservation_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_GROUPS).contains(&groups),
            "source pool exceeds physical page capacity"
        );
        let pages = [groups, groups, groups, groups * 2];
        Ok(Self {
            pages,
            global_bytes: groups * GROUP_BYTES,
            cache_bytes: BackboneCache::device_bytes(slots, pages)?,
            occupied_before,
            reservation_bytes,
        })
    }
    pub fn new(
        slots: usize,
        context: usize,
        retained_turns: usize,
        snapshot_bytes: usize,
        exact: Option<ByteSize>,
        reservation: Option<Reservation>,
        free: usize,
        total: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=1_048_576).contains(&context),
            "invalid pool context limit"
        );
        ensure!(
            free <= total && total > 0,
            "invalid device memory information"
        );
        ensure!((1..=16).contains(&slots), "invalid concurrency limit");
        let occupied = total - free;
        let ceiling = reservation
            .map(|r| r.bytes(total))
            .transpose()?
            .unwrap_or(total);
        let available = ceiling.checked_sub(occupied).and_then(|v| v.checked_sub(RUNTIME_HEADROOM))
            .context("memory reservation leaves no space after existing allocations and runtime headroom")?;
        ensure!(retained_turns <= 128, "invalid retained-turn limit");
        // At most two retained frontiers per turn, plus one active tail per slot.
        let spare_groups = slots + 2 * retained_turns;
        let per_context = context.div_ceil(512);
        // Explicit budgets may trade aggregate context capacity for memory.
        // Provision at least one page per active owner plus private tail space.
        let minimum = slots + spare_groups;
        let groups = if let Some(exact) = exact {
            ensure!(
                exact.0 / GROUP_BYTES <= MAX_GROUPS,
                "exact KV pool exceeds physical page capacity"
            );
            exact.0 / GROUP_BYTES
        } else if reservation.is_some() {
            // Tables saturate at the maximum logical per-request context. Find
            // the largest whole page group whose entire cache fits the budget.
            let (mut low, mut high) = (0, MAX_GROUPS);
            while low < high {
                let mid = (low + high).div_ceil(2);
                if Self::from_groups(slots, mid, occupied, ceiling)?.cache_bytes <= available {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            low
        } else {
            // Keep the default total footprint stable when snapshot tails move
            // from demand allocation into startup arenas. Explicit KV/total
            // reservations above retain their own sizing policy.
            (per_context * (slots + RETAINED_CONTEXTS) + spare_groups)
                .saturating_sub(snapshot_bytes.div_ceil(GROUP_BYTES)).max(minimum)
        };
        ensure!(groups >= minimum,
            "KV pool needs at least {} global bytes for {slots} active owners plus copy-on-write headroom; increase the memory budget",
            minimum * GROUP_BYTES);
        let plan = Self::from_groups(slots, groups, occupied, ceiling)?;
        ensure!(plan.cache_bytes <= available,
            "KV cache needs {} bytes but reservation leaves {available} after existing allocations and {} bytes of runtime headroom",
            plan.cache_bytes, RUNTIME_HEADROOM);
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn planner_is_opt_in_and_explicit_pool_sizes_remain_strict() -> Result<()> {
        use clap::{Args, FromArgMatches};
        let parse = |extra: &[&str]| -> Result<crate::cli::NativeServeArgs> {
            let mut values = vec!["fixture", "--snapshot", "/missing", "--native-lib", "/missing",
                "--peers", "127.0.0.1:9000,127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003"];
            values.extend(extra);
            let matches = crate::cli::NativeServeArgs::augment_args(clap::Command::new("fixture"))
                .try_get_matches_from(values)?;
            Ok(crate::cli::NativeServeArgs::from_arg_matches(&matches)?)
        };
        let unchanged = parse(&[])?;
        assert!(unchanged.pool_tokens.is_none());
        assert!(planned_pool_size(&unchanged, &[])?.is_none());
        let exact = parse(&["--kv-pool-size", "1GiB"])?;
        assert_eq!(planned_pool_size(&exact, &[])?.unwrap().0, 1 << 30);
        assert!(parse(&["--kv-pool-size", "1GiB", "--pool-tokens", "0"]).is_err());
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("config.json"), include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json")))?;
        let mut auto = parse(&["--pool-tokens", "0"])?;
        auto.snapshot = directory.path().to_owned();
        let full = planned_pool_size(&auto, &[(96 << 30, 96 << 30)])?.unwrap();
        assert_eq!(full.0 / GROUP_BYTES, 14 * 2048 + 16 + 40);
        let limited = planned_pool_size(&auto, &[(6 << 30, 96 << 30)])?.unwrap();
        assert!(limited.0 < full.0);
        auto.pool_tokens = Some(14 * 1_048_576);
        assert!(planned_pool_size(&auto, &[(6 << 30, 96 << 30)]).is_err());
        Ok(())
    }
    #[test]
    fn planner_cache_formula_matches_allocator_and_replica_owners() -> Result<()> {
        use crate::families::deepseek_v41::v41_backbone_cache::CachePlacement;
        let config: serde_json::Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json")))?;
        for slots in [1, 16] {
            for groups in [2 * slots + 40, 4152, 32768] {
                let pages = [groups, groups, groups, groups * 2];
                let single = cuteafd_loader::serving_capacity::deepseek_v41_cache_bytes(&config, 1,
                    slots as u64, groups as u64, false)?;
                assert_eq!(single, vec![BackboneCache::device_bytes(slots, pages)? as u64]);
                for replicated in [false, true] {
                    let actual = if replicated { BackboneCache::replicated_device_bytes(CachePlacement::encoder_decoder(), slots, pages)? }
                        else { BackboneCache::distributed_device_bytes(CachePlacement::encoder_decoder(), slots, pages)? };
                    let predicted = cuteafd_loader::serving_capacity::deepseek_v41_cache_bytes(&config, 2,
                        slots as u64, groups as u64, replicated)?;
                    assert_eq!(predicted, actual.map(|b| b as u64).to_vec());
                }
            }
        }
        Ok(())
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
    #[test]
    fn default_pool_covers_eighteen_contexts_and_twenty_four_snapshot_tails() {
        let p = PoolPlan::new(16, 1_048_576, 24, 0, None, None, 96 << 30, 96 << 30).unwrap();
        assert_eq!(p.pages, [36_928, 36_928, 36_928, 73_856]);
        assert_eq!(p.global_bytes, 16_827_351_040);
        assert!(p.cache_bytes > p.global_bytes);
        let small = PoolPlan::new(16, 32768, 24, 0, None, None, 8 << 30, 96 << 30).unwrap();
        assert_eq!(small.pages, [1216, 1216, 1216, 2432]);
        assert!(PoolPlan::new(16, 1_048_576, 24, 0, None, None, 32 << 30, 96 << 30).is_ok());
        assert!(PoolPlan::new(16, 1_048_576, 24, 0, None, None, 16 << 30, 96 << 30).is_err());
    }
    #[test]
    fn snapshot_arenas_trade_default_pool_bytes_without_changing_overrides() {
        let tail = crate::families::deepseek_v41::v41_backbone_cache::BackbonePrefix::device_bytes().div_ceil(256) * 256;
        let bytes = 50 * (tail + 3 * cuteafd_ffi::V41DsparkCache::SLOT_BYTES);
        let free = 56 << 30;
        let total = 96 << 30;
        let old = PoolPlan::new(16, 1_048_576, 24, 0, None, None, free, total).unwrap();
        let pooled = PoolPlan::new(16, 1_048_576, 24, bytes, None, None, free-bytes, total).unwrap();
        let returned = old.global_bytes - pooled.global_bytes;
        assert!(returned >= bytes && returned - bytes < GROUP_BYTES);
        let explicit = PoolPlan::new(16, 1_048_576, 24, bytes,
            Some(ByteSize(old.global_bytes)), None, free-bytes, total).unwrap();
        assert_eq!(explicit.pages, old.pages);
        let short = PoolPlan::new(16, 128, 24, bytes, None, None, free-bytes, total).unwrap();
        assert!(short.pages[0] >= 2*16 + 2*24);
    }
    #[test]
    fn exact_and_total_budgets_round_down_without_undercutting_admission() {
        let total = 96 << 30;
        let free = 56 << 30;
        let default = PoolPlan::new(16, 1_048_576, 24, 0, None, None, free, total).unwrap();
        let exact = PoolPlan::new(
            16,
            1_048_576,
            24, 0,
            Some(ByteSize(default.global_bytes + 99)),
            None,
            free,
            total,
        )
        .unwrap();
        assert_eq!(exact.pages, default.pages);
        let small =
            PoolPlan::new(2, 1_048_576, 24, 0, Some(ByteSize(1 << 30)), None, free, total).unwrap();
        assert!(small.global_bytes <= 1 << 30);
        let c2 = PoolPlan::new(2, 1_048_576, 24, 0, None, None, free, total).unwrap();
        assert_eq!(c2.pages[0], 2048 * 4 + 50);
        let reservation = Some("80GiB".parse().unwrap());
        let p = PoolPlan::new(16, 1_048_576, 24, 0, None, reservation, free, total).unwrap();
        assert!(p.cache_bytes + p.occupied_before + RUNTIME_HEADROOM <= 80 << 30);
        let next =
            PoolPlan::from_groups(16, p.pages[0] + 1, p.occupied_before, p.reservation_bytes)
                .unwrap();
        assert!(next.cache_bytes + p.occupied_before + RUNTIME_HEADROOM > 80 << 30);
        assert!(PoolPlan::new(
            16,
            1_048_576,
            24, 0,
            Some(ByteSize(1 << 20)),
            None,
            free,
            total
        )
        .is_err());
        assert!(PoolPlan::new(
            16,
            1_048_576,
            24, 0,
            Some(ByteSize(default.global_bytes)),
            Some("50GiB".parse().unwrap()),
            free,
            total
        )
        .is_err());
        assert!(PoolPlan::new(
            16,
            1_048_576,
            24, 0,
            None,
            Some("101GiB".parse().unwrap()),
            free,
            total
        )
        .is_err());
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalLayers { Auto, Count(usize) }
impl FromStr for LocalLayers {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        if value == "auto" { return Ok(Self::Auto); }
        let count: usize = value.parse().context("RTX expert layers must be auto or 0..40")?;
        ensure!(count <= 40, "RTX expert layers must be auto or 0..40");
        Ok(Self::Count(count))
    }
}
#[derive(Debug)]
pub(super) struct LocalLayerPlan {
    pub layers: usize,
    pub resident_bytes: usize,
    pub workspace_bytes: usize,
    pub peak_bytes: usize,
}
impl LocalLayerPlan {
    /// Free memory is sampled after mandatory weights, KV and both lanes exist.
    /// Reserve both local workspaces and bounded load staging conservatively.
    pub fn new(requested: LocalLayers, budgets: &[crate::families::deepseek_v41::v41_experts::ExpertLoadBudget],
        workspace_bytes: usize, free: usize, total: usize, ceiling: usize) -> Result<Self> {
        ensure!(free <= total && ceiling <= total && budgets.len() <= 40, "invalid local memory inventory");
        let available = ceiling.saturating_sub(total - free).saturating_sub(RUNTIME_HEADROOM);
        let target = match requested { LocalLayers::Auto => budgets.len(), LocalLayers::Count(n) => n };
        ensure!(target <= budgets.len(), "requested RTX layers exceed available layer plans");
        let mut plan = Self { layers: 0, resident_bytes: 0, workspace_bytes: 0, peak_bytes: 0 };
        let mut staging = 0;
        for budget in budgets.iter().take(target) {
            let resident = plan.resident_bytes.checked_add(budget.resident_bytes).context("local weight size overflow")?;
            staging = staging.max(budget.device_staging_bytes);
            let peak = resident.checked_add(workspace_bytes).and_then(|b| b.checked_add(staging))
                .context("local layer peak size overflow")?;
            if peak > available { break; }
            plan = Self { layers: plan.layers + 1, resident_bytes: resident, workspace_bytes, peak_bytes: peak };
        }
        if let LocalLayers::Count(n) = requested {
            ensure!(plan.layers == n, "requested {n} RTX expert layers but only {} fit after KV, workspaces, staging and runtime headroom", plan.layers);
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod local_tests {
    use super::*;
    fn budgets() -> Vec<crate::families::deepseek_v41::v41_experts::ExpertLoadBudget> {
        vec![crate::families::deepseek_v41::v41_experts::ExpertLoadBudget { resident_bytes: 7 << 30,
            device_staging_bytes: 20 << 20, pinned_host_bytes: 0, read_scratch_bytes: 0 }; 40]
    }
    #[test]
    fn local_prefix_respects_workspace_staging_and_ceiling() {
        let b = budgets();
        let p = LocalLayerPlan::new(LocalLayers::Auto, &b, 600 << 20, 38 << 30, 96 << 30, 96 << 30).unwrap();
        assert_eq!(p.layers, 5);
        assert_eq!(p.resident_bytes, 35 << 30);
        let limited = LocalLayerPlan::new(LocalLayers::Auto, &b, 600 << 20, 38 << 30, 96 << 30, 90 << 30).unwrap();
        assert_eq!(limited.layers, 4);
        assert!(LocalLayerPlan::new(LocalLayers::Count(5), &b, 600 << 20, 38 << 30, 96 << 30, 90 << 30).is_err());
        let exact = LocalLayerPlan::new(LocalLayers::Auto, &b, 600 << 20,
            p.peak_bytes + RUNTIME_HEADROOM, 96 << 30, 96 << 30).unwrap();
        assert_eq!(exact.layers, 5);
        let short = LocalLayerPlan::new(LocalLayers::Auto, &b, 600 << 20,
            p.peak_bytes + RUNTIME_HEADROOM - 1, 96 << 30, 96 << 30).unwrap();
        assert_eq!(short.layers, 4);
        let zero = LocalLayerPlan::new(LocalLayers::Auto, &b, 600 << 20, 1 << 30, 96 << 30, 96 << 30).unwrap();
        assert_eq!((zero.layers, zero.peak_bytes), (0, 0));
        assert!("41".parse::<LocalLayers>().is_err());
        assert!("-1".parse::<LocalLayers>().is_err());
        assert_eq!("auto".parse::<LocalLayers>().unwrap(), LocalLayers::Auto);
    }
}
