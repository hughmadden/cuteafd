//! One device memory policy for every family (TJ, 2026-11-07).
//!
//! Each device has one cap. Weights, KV, workspaces and the reserve all live
//! inside it. The planner fixes the layout for a KV target; the pool then
//! grows into what the fixed layout leaves, up to a KV max, and a launch is
//! refused below the KV min.
//!
//! The reserve is a sum of named components. Families supply their measured
//! components (CUDA context, loaded code, graphs, runtime growth), release padding and a
//! `FAMILY_RESERVE` constant; the operator may replace the whole reserve with
//! `GPU_RESERVE` / `SPARK_RESERVE`, which zeroes every built-in component.
//! Installed once per process before planning or allocation, like the Spark
//! reply formats.
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fmt;
use std::str::FromStr;
use std::sync::OnceLock;

use crate::serving_capacity::DEFAULT_GPU_KV_TOKENS;

/// Cards at or below this size take the 1M default KV target.
pub const SMALL_CARD_BYTES: u64 = 32 << 30;
pub const SMALL_CARD_KV_TARGET_TOKENS: u64 = 1 << 20;
/// Agentic context floor: the default KV min when the model allows it.
pub const AGENTIC_FLOOR_TOKENS: u64 = 262_144;
/// Explicit release allowance for unassigned memory, lowered each release as
/// the remaining owners are attributed. Per coordinator GPU, inside its cap.
pub const DEFAULT_MEMORY_PADDING: u64 = 100 << 20;

/// A percentage of each measured node/executable coefficient, or absolute
/// bytes added to each coefficient. Percentage adjustments round to whole bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphPaddingValue { BasisPoints(u64), Bytes(u64) }

impl GraphPaddingValue {
    pub fn bytes_per_node(self, rate: u64) -> Option<u64> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            Self::BasisPoints(points) => ((u128::from(rate) * u128::from(points) + 5000) / 10000).try_into().ok(),
        }
    }
}

impl FromStr for GraphPaddingValue {
    type Err = GraphNodePaddingError;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let number = |s: &str| if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) {
            s.parse::<u64>().map_err(|_| GraphNodePaddingError)
        } else { Err(GraphNodePaddingError) };
        if let Some(percent) = text.strip_suffix('%') {
            let (whole, fraction) = percent.split_once('.').unwrap_or((percent, "0"));
            if fraction.len() > 2 { return Err(GraphNodePaddingError); }
            let fraction = number(fraction)? * if fraction.len() == 1 { 10 } else { 1 };
            let points = number(whole)?.checked_mul(100).and_then(|v| v.checked_add(fraction))
                .ok_or(GraphNodePaddingError)?;
            Ok(Self::BasisPoints(points))
        } else { Ok(Self::Bytes(number(text)?)) }
    }
}

impl fmt::Display for GraphPaddingValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Bytes(bytes) => write!(f, "{bytes}"),
            Self::BasisPoints(points) => write!(f, "{}.{:02}%", points / 100, points % 100) }
    }
}

/// Graph padding is independent of GPU_RESERVE, SPARK_RESERVE and MEMORY_PADDING.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNodePadding {
    pub rtx: GraphPaddingValue,
    pub spark: GraphPaddingValue,
}

impl Default for GraphNodePadding {
    fn default() -> Self { Self { rtx: GraphPaddingValue::BasisPoints(250), spark: GraphPaddingValue::BasisPoints(250) } }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("GRAPH_NODE_PADDING must be a percentage (up to two decimals), integer bytes per coefficient, or rtx=VALUE,spark=VALUE")]
pub struct GraphNodePaddingError;

impl FromStr for GraphNodePadding {
    type Err = GraphNodePaddingError;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if !text.contains('=') {
            let value = text.parse()?;
            return Ok(Self { rtx: value, spark: value });
        }
        let (mut rtx, mut spark) = (None, None);
        for entry in text.split(',') {
            let (key, value) = entry.split_once('=').ok_or(GraphNodePaddingError)?;
            let slot = match key { "rtx" => &mut rtx, "spark" => &mut spark, _ => return Err(GraphNodePaddingError) };
            if slot.replace(value.parse()?).is_some() { return Err(GraphNodePaddingError); }
        }
        Ok(Self { rtx: rtx.ok_or(GraphNodePaddingError)?, spark: spark.ok_or(GraphNodePaddingError)? })
    }
}

impl fmt::Display for GraphNodePadding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "rtx={},spark={}", self.rtx, self.spark) }
}

/// A byte size with an explicit unit: `B`, `MB`, `GB`, `MiB` or `GiB`. No
/// percentages. A bare number is bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Size(pub u64);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SizeError {
    #[error("memory size {0:?}: expected a non-negative decimal with at most six fractional digits")]
    Number(String),
    #[error("memory size {0:?}: unit must be B, MB, GB, MiB or GiB")]
    Unit(String),
    #[error("memory size {0:?} overflows")]
    Overflow(String),
}

impl FromStr for Size {
    type Err = SizeError;
    fn from_str(text: &str) -> Result<Self, SizeError> {
        let value = text.trim();
        let end = value.find(|c: char| !c.is_ascii_digit() && c != '.').unwrap_or(value.len());
        let (number, unit) = (&value[..end], value[end..].trim());
        let (whole, fraction) = number.split_once('.').unwrap_or((number, ""));
        if whole.is_empty() || fraction.len() > 6 || !whole.bytes().all(|c| c.is_ascii_digit())
            || !fraction.bytes().all(|c| c.is_ascii_digit()) {
            return Err(SizeError::Number(text.into()));
        }
        let multiplier: u128 = match unit.to_ascii_lowercase().as_str() {
            "" | "b" => 1,
            "mb" => 1_000_000,
            "gb" => 1_000_000_000,
            "mib" => 1 << 20,
            "gib" => 1 << 30,
            _ => return Err(SizeError::Unit(text.into())),
        };
        let overflow = || SizeError::Overflow(text.into());
        let millionths = u128::from(whole.parse::<u64>().map_err(|_| overflow())?) * 1_000_000
            + if fraction.is_empty() { 0 } else {
                u128::from(fraction.parse::<u64>().map_err(|_| overflow())?) * 10u128.pow(6 - fraction.len() as u32)
            };
        u64::try_from(millionths * multiplier / 1_000_000).map(Size).map_err(|_| overflow())
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}B", self.0)
    }
}

/// Device kind a reserve applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind { Gpu, Spark }

/// The operator's policy, as given. `None` everywhere is the default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPolicy {
    /// Total per-GPU ceiling (`GPU_MEMORY_CAP`); `None` is the full device.
    pub gpu_cap: Option<u64>,
    /// Total per-Spark ceiling (`SPARK_MEMORY_CAP`); `None` is the family's Spark default.
    pub spark_cap: Option<u64>,
    /// Whole GPU reserve (`GPU_RESERVE`); zeroes every built-in GPU component.
    pub gpu_reserve: Option<u64>,
    /// Whole Spark reserve (`SPARK_RESERVE`); zeroes every built-in Spark component.
    pub spark_reserve: Option<u64>,
    /// Per-GPU named reserve (`FAMILY_RESERVE`); `None` is the family constant.
    pub family_reserve: Option<u64>,
    /// Per-GPU release padding (`MEMORY_PADDING`); `None` uses the release default.
    #[serde(default)]
    pub memory_padding: Option<u64>,
    #[serde(default)]
    pub graph_node_padding: GraphNodePadding,
    pub kv_target: Option<u64>,
    pub kv_min: Option<u64>,
    pub kv_max: Option<u64>,
}

static INSTALLED: OnceLock<MemoryPolicy> = OnceLock::new();
thread_local! {
    static SCOPED: RefCell<Option<MemoryPolicy>> = const { RefCell::new(None) };
}

impl MemoryPolicy {
    /// Fixed once, before any planning or allocation.
    pub fn install(self) -> Result<(), &'static str> {
        INSTALLED.set(self).map_err(|_| "memory policy was already installed")
    }

    /// The process policy (a test scope wins on its own thread).
    pub fn current() -> Self {
        SCOPED.with(|scoped| *scoped.borrow()).or_else(|| INSTALLED.get().copied()).unwrap_or_default()
    }

    /// Run `body` with this policy on the current thread only (tests, in-process plans).
    pub fn scoped<T>(self, body: impl FnOnce() -> T) -> T {
        struct Restore(Option<MemoryPolicy>);
        impl Drop for Restore {
            fn drop(&mut self) { SCOPED.with(|scoped| *scoped.borrow_mut() = self.0); }
        }
        let _restore = Restore(SCOPED.with(|scoped| scoped.borrow_mut().replace(self)));
        body()
    }

    /// The per-GPU cap: the smaller of the physical total and `GPU_MEMORY_CAP`.
    pub fn gpu_cap_bytes(&self, physical: u64) -> u64 {
        self.gpu_cap.map_or(physical, |cap| cap.min(physical))
    }

    /// The per-Spark cap, defaulting to the caller's Spark default.
    pub fn spark_cap_bytes(&self, default: u64) -> u64 {
        self.spark_cap.unwrap_or(default)
    }

    pub fn reserve_override(&self, kind: DeviceKind) -> Option<u64> {
        match kind { DeviceKind::Gpu => self.gpu_reserve, DeviceKind::Spark => self.spark_reserve }
    }

    /// A built-in reserve component's charged bytes: zero under an override.
    pub fn component(&self, kind: DeviceKind, bytes: u64) -> u64 {
        if self.reserve_override(kind).is_some() { 0 } else { bytes }
    }

    /// The per-device named reserve the budget keeps free (never allocated):
    /// the whole override when one is set, else `FAMILY_RESERVE` plus release padding on GPUs
    /// (the key, or the family constant) and nothing on Sparks.
    pub fn named_reserve(&self, kind: DeviceKind, family_constant: u64) -> u64 {
        match (self.reserve_override(kind), kind) {
            (Some(bytes), _) => bytes,
            (None, DeviceKind::Gpu) => self.family_reserve.unwrap_or(family_constant).saturating_add(self.release_padding()),
            (None, DeviceKind::Spark) => 0,
        }
    }

    /// Release padding is independent of family reserve; a whole GPU reserve
    /// override replaces both. Sparks never receive this allowance.
    pub fn release_padding(&self) -> u64 {
        self.component(DeviceKind::Gpu, self.memory_padding.unwrap_or(DEFAULT_MEMORY_PADDING))
    }

    /// KV target, min and max for the smallest physical coordinator card.
    /// The default minimum is the smaller of the model context and the
    /// agentic floor, bounded by the target, independent of Spark placement.
    pub fn kv(&self, smallest_card: u64, context: u64) -> Result<KvBounds, PolicyError> {
        let default_target = if smallest_card <= SMALL_CARD_BYTES { SMALL_CARD_KV_TARGET_TOKENS }
            else { DEFAULT_GPU_KV_TOKENS };
        // An explicit target can be below the model maximum for automatic context.
        let target = self.kv_target.unwrap_or(default_target.max(context));
        let min = self.kv_min.unwrap_or(context.min(AGENTIC_FLOOR_TOKENS).min(target));
        let max = self.kv_max.unwrap_or(target).max(if self.kv_max.is_some() { 0 } else { target });
        if target == 0 || min > target || target > max {
            return Err(PolicyError::KvOrder { min, target, max });
        }
        Ok(KvBounds { target, min, max })
    }

    /// Reject inconsistent caps and reserves before any planning.
    pub fn validate(&self) -> Result<(), PolicyError> {
        for (name, cap, reserve) in [("GPU", self.gpu_cap, self.gpu_reserve),
            ("Spark", self.spark_cap, self.spark_reserve)] {
            if cap == Some(0) { return Err(PolicyError::ZeroCap(name)); }
            if let (Some(cap), Some(reserve)) = (cap, reserve) {
                if reserve >= cap { return Err(PolicyError::ReserveFillsCap { device: name, reserve, cap }); }
            }
        }
        self.kv(u64::MAX, 0).map(|_| ())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvBounds {
    pub target: u64,
    pub min: u64,
    pub max: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PolicyError {
    #[error("KV bounds must satisfy 0 < KV_MIN_TOKENS <= KV_TARGET_TOKENS <= KV_MAX_TOKENS (min {min}, target {target}, max {max})")]
    KvOrder { min: u64, target: u64, max: u64 },
    #[error("{0}_MEMORY_CAP must be positive")]
    ZeroCap(&'static str),
    #[error("{device} reserve {reserve} B leaves nothing inside the {cap} B cap")]
    ReserveFillsCap { device: &'static str, reserve: u64, cap: u64 },
    #[error("KV pool {pool} tokens is below KV_MIN_TOKENS {min}: {reason}")]
    BelowMin { pool: u64, min: u64, reason: String },
}

/// One named reserve component on one device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReserveComponent {
    pub name: String,
    pub bytes: u64,
}

/// The startup breakdown of one device: cap, weights, workspaces, the total
/// reserve with every component named and sized, and the KV pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceBreakdown {
    pub device: String,
    pub cap_bytes: u64,
    pub weights_bytes: u64,
    pub workspace_bytes: u64,
    pub reserve: Vec<ReserveComponent>,
    /// Set when `GPU_RESERVE` / `SPARK_RESERVE` replaced the components.
    pub reserve_overridden: bool,
    pub kv_bytes: u64,
    pub other_bytes: u64,
}

impl DeviceBreakdown {
    pub fn reserve_bytes(&self) -> u64 {
        self.reserve.iter().map(|component| component.bytes).sum()
    }
}

/// Pool result plus the policy bounds it was sized against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolBreakdown {
    pub kv: KvBounds,
    pub pool_tokens: u64,
}

const MIB: f64 = (1u64 << 20) as f64;

/// The one-line form shared by the startup log and `cuteafd plan`.
pub fn render_line(device: &DeviceBreakdown, pool: &PoolBreakdown) -> String {
    let mib = |bytes: u64| format!("{:.0}MiB", bytes as f64 / MIB);
    let components = device.reserve.iter()
        .map(|component| format!("{} {}", component.name, mib(component.bytes)))
        .collect::<Vec<_>>().join(", ");
    format!("{}: cap {}; weights {}; workspaces {}; reserve {}{} [{}]; other {}; KV target {} min {} max {}, pool {} tokens / {}",
        device.device, mib(device.cap_bytes), mib(device.weights_bytes), mib(device.workspace_bytes),
        mib(device.reserve_bytes()), if device.reserve_overridden { " (overridden)" } else { "" },
        components, mib(device.other_bytes), pool.kv.target, pool.kv.min, pool.kv.max, pool.pool_tokens,
        mib(device.kv_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;

    #[test]
    fn graph_padding_has_explicit_architectures_and_no_reserve_dependency() {
        for (text, expected) in [("0", GraphNodePadding { rtx: GraphPaddingValue::Bytes(0), spark: GraphPaddingValue::Bytes(0) }),
            ("350", GraphNodePadding { rtx: GraphPaddingValue::Bytes(350), spark: GraphPaddingValue::Bytes(350) }),
            ("rtx=1,spark=20", GraphNodePadding { rtx: GraphPaddingValue::Bytes(1), spark: GraphPaddingValue::Bytes(20) }),
            ("spark=20,rtx=1", GraphNodePadding { rtx: GraphPaddingValue::Bytes(1), spark: GraphPaddingValue::Bytes(20) })] {
            assert_eq!(text.parse::<GraphNodePadding>(), Ok(expected));
            assert_eq!(expected.to_string().parse::<GraphNodePadding>(), Ok(expected));
        }
        assert_eq!("2.5%".parse::<GraphNodePadding>(), Ok(GraphNodePadding::default()));
        assert_eq!(GraphPaddingValue::BasisPoints(250).bytes_per_node(7143), Some(179));
        assert_eq!(GraphPaddingValue::BasisPoints(250).bytes_per_node(2456), Some(61));
        for text in ["", "rtx=1", "rtx=1,rtx=2", "rtx=1,spark=2,", "-1", "1MiB", "2.501%", "+1", "18446744073709551616"] {
            assert!(text.parse::<GraphNodePadding>().is_err(), "{text}");
        }
    }

    #[test]
    fn sizes_take_every_unit_and_reject_percentages() {
        for (text, bytes) in [("0", 0), ("4096", 4096), ("12B", 12), ("1.5MB", 1_500_000), ("2GB", 2_000_000_000),
            ("3MiB", 3 << 20), ("2.5GiB", 5 << 29), (" 1 GiB ", GIB), ("96gib", 96 * GIB)] {
            assert_eq!(text.parse::<Size>(), Ok(Size(bytes)), "{text}");
        }
        for text in ["", "GiB", "97%", "1.0000001GiB", "-1GiB", "1TB", "1.2.3"] {
            assert!(text.parse::<Size>().is_err(), "{text}");
        }
        assert!("99999999999999999999GiB".parse::<Size>().is_err());
    }

    #[test]
    fn override_zeroes_components_and_family_reserve_adds() {
        let auto = MemoryPolicy::default();
        assert_eq!(auto.component(DeviceKind::Gpu, 700), 700);
        assert_eq!(auto.named_reserve(DeviceKind::Gpu, 2 * GIB), 2 * GIB + DEFAULT_MEMORY_PADDING);
        let family = MemoryPolicy { family_reserve: Some(GIB), ..auto };
        // FAMILY_RESERVE replaces only the family constant; components stay.
        assert_eq!((family.component(DeviceKind::Gpu, 700), family.named_reserve(DeviceKind::Gpu, 2 * GIB)), (700, GIB + DEFAULT_MEMORY_PADDING));
        let overridden = MemoryPolicy { gpu_reserve: Some(3 * GIB), family_reserve: Some(GIB), ..auto };
        assert_eq!(overridden.component(DeviceKind::Gpu, 700), 0);
        assert_eq!(overridden.named_reserve(DeviceKind::Gpu, 2 * GIB), 3 * GIB);
        // A GPU override leaves Spark components alone.
        assert_eq!(overridden.component(DeviceKind::Spark, 5), 5);
        let spark = MemoryPolicy { spark_reserve: Some(GIB), ..auto };
        assert_eq!((spark.component(DeviceKind::Spark, 5), spark.named_reserve(DeviceKind::Spark, 0)), (0, GIB));
    }

    #[test]
    fn release_padding_is_explicit_gpu_only_and_whole_reserve_replaces_it() {
        let default = MemoryPolicy::default();
        assert_eq!(default.release_padding(), DEFAULT_MEMORY_PADDING);
        assert_eq!(default.named_reserve(DeviceKind::Gpu, 7), 7 + DEFAULT_MEMORY_PADDING);
        assert_eq!(default.named_reserve(DeviceKind::Spark, 7), 0);
        let disabled = MemoryPolicy { memory_padding: Some(0), ..default };
        assert_eq!(disabled.named_reserve(DeviceKind::Gpu, 7), 7);
        let custom = MemoryPolicy { memory_padding: Some(123), ..default };
        assert_eq!(custom.named_reserve(DeviceKind::Gpu, 7), 130);
        for reserve in [0, 456] {
            let whole = MemoryPolicy { gpu_reserve: Some(reserve), ..custom };
            assert_eq!(whole.release_padding(), 0);
            assert_eq!(whole.named_reserve(DeviceKind::Gpu, 7), reserve);
        }
        let encoded = serde_json::to_value(default).unwrap();
        let mut old = encoded;
        old.as_object_mut().unwrap().remove("memory_padding");
        assert_eq!(serde_json::from_value::<MemoryPolicy>(old).unwrap().release_padding(), DEFAULT_MEMORY_PADDING);
    }

    #[test]
    fn kv_defaults_follow_card_and_context() {
        let policy = MemoryPolicy::default();
        assert_eq!(policy.kv(96 * GIB, 1 << 20).unwrap(), KvBounds { target: 2 << 20, min: AGENTIC_FLOOR_TOKENS, max: 2 << 20 });
        assert_eq!(policy.kv(96 * GIB, 65_536).unwrap(), KvBounds { target: 2 << 20, min: 65_536, max: 2 << 20 });
        assert_eq!(policy.kv(32 * GIB, 65_536).unwrap(), KvBounds { target: 1 << 20, min: 65_536, max: 1 << 20 });
        let short_target = MemoryPolicy { kv_target: Some(131_072), ..policy };
        assert_eq!(short_target.kv(96 * GIB, 1 << 20).unwrap(),
            KvBounds { target: 131_072, min: 131_072, max: 131_072 });
        let grow = MemoryPolicy { kv_max: Some(4 << 20), ..policy };
        assert_eq!(grow.kv(96 * GIB, 1 << 20).unwrap().max, 4 << 20);
        assert!(MemoryPolicy { kv_max: Some(1 << 20), ..policy }.kv(96 * GIB, 0).is_err());
        assert!(MemoryPolicy { kv_min: Some(3 << 20), ..policy }.kv(96 * GIB, 0).is_err());
    }

    #[test]
    fn caps_bound_physical_and_reject_a_full_reserve() {
        let policy = MemoryPolicy { gpu_cap: Some(32 * GIB), ..Default::default() };
        assert_eq!((policy.gpu_cap_bytes(96 * GIB), policy.gpu_cap_bytes(24 * GIB)), (32 * GIB, 24 * GIB));
        assert_eq!(MemoryPolicy::default().spark_cap_bytes(108 * GIB), 108 * GIB);
        assert!(MemoryPolicy { gpu_reserve: Some(32 * GIB), ..policy }.validate().is_err());
        assert!(MemoryPolicy { gpu_cap: Some(0), ..Default::default() }.validate().is_err());
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn scoped_policy_is_thread_local_and_restored() {
        let scoped = MemoryPolicy { gpu_reserve: Some(1), ..Default::default() };
        scoped.scoped(|| {
            assert_eq!(MemoryPolicy::current(), scoped);
            std::thread::spawn(|| assert_eq!(MemoryPolicy::current().gpu_reserve, None)).join().unwrap();
        });
        assert_eq!(MemoryPolicy::current().gpu_reserve, None);
    }

    #[test]
    fn log_line_names_every_component_and_the_override() {
        let device = DeviceBreakdown { device: "rtx0".into(), cap_bytes: 96 * GIB, weights_bytes: 40 * GIB,
            workspace_bytes: 8 * GIB, reserve: vec![
                ReserveComponent { name: "cuda context".into(), bytes: 560 << 20 },
                ReserveComponent { name: "graphs".into(), bytes: GIB },
                ReserveComponent { name: "family".into(), bytes: 0 }],
            reserve_overridden: false, kv_bytes: 30 * GIB, other_bytes: GIB };
        let pool = PoolBreakdown { kv: KvBounds { target: 2 << 20, min: 262_144, max: 2 << 20 }, pool_tokens: 2 << 20 };
        let line = render_line(&device, &pool);
        for part in ["rtx0: cap 98304MiB", "weights 40960MiB", "workspaces 8192MiB", "reserve 1584MiB [cuda context 560MiB, graphs 1024MiB, family 0MiB]",
            "KV target 2097152 min 262144 max 2097152, pool 2097152 tokens / 30720MiB"] {
            assert!(line.contains(part), "{line} lacks {part}");
        }
        let overridden = DeviceBreakdown { reserve_overridden: true, reserve: vec![
            ReserveComponent { name: "GPU_RESERVE".into(), bytes: 3 * GIB }], ..device };
        assert!(render_line(&overridden, &pool).contains("reserve 3072MiB (overridden) [GPU_RESERVE 3072MiB]"));
    }
}
