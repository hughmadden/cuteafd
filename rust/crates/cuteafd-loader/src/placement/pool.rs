//! The one KV pool policy (`cuteafd_core::memory_policy`): the layout is
//! sized for `KV_TARGET_TOKENS` (2M tokens on cards above 32 GiB, 1M at 32 GiB
//! or less by default); the pool then grows into what
//! the fixed layout leaves, up to `KV_MAX_TOKENS`; below `KV_MIN_TOKENS` the
//! launch is refused.
use cuteafd_core::memory_policy::{MemoryPolicy, SMALL_CARD_BYTES};

/// Agentic context floor: the default KV min.
pub use cuteafd_core::memory_policy::AGENTIC_FLOOR_TOKENS;

/// Physical card identity and an independent operator memory ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinatorCard {
    pub physical_bytes: u64,
    pub logical_bytes: u64,
}

impl CoordinatorCard {
    /// Offline caps are not card identities. Absent physical facts, assume
    /// the architecture's default PRO class, and report that assumption.
    pub fn planned(logical_bytes: u64, physical_sms: Option<u32>) -> Self {
        let class = physical_sms.and_then(|sms| super::inventory::ARCH_CONTEXTS.iter()
            .find(|class| class.arch == "sm_120" && class.sms == sms));
        let physical_bytes = class.map_or(super::inventory::PRO_TOTAL_BYTES, |class|
            if class.total_bytes == 0 { SMALL_CARD_BYTES } else { class.total_bytes });
        Self { physical_bytes, logical_bytes }
    }

    pub fn maximum_pool(cards: &[Self], context: u64, requested: Option<u64>, page_rows: u64) -> u64 {
        let physical = cards.iter().map(|card| card.physical_bytes).collect::<Vec<_>>();
        PoolPolicy::resolve(&physical, context, requested, page_rows).maximum_tokens()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolPolicy {
    pub context: super::planner::Context,
    /// Immutable workspace extent, independent of the pool fit floor.
    pub compiled_context: u64,
    /// An explicit pool is strict: admission fails rather than shrink it.
    pub requested: Option<u64>,
    /// The layout (onboarding, buffers) is sized for this pool.
    pub target: u64,
    /// Admission refuses below this (`KV_MIN_TOKENS` and any pinned context).
    pub floor: u64,
    /// The largest pool the build can index; a fixed onboard fills up to it.
    pub ceiling: u64,
    /// After an automatic layout is fixed for the target, the pool grows into
    /// what is left up to this (`KV_MAX_TOKENS`, default the target).
    pub max: u64,
    pub unit_rows: u64,
}

impl PoolPolicy {
    /// Both CLI zero and an omitted pool request select automatic admission.
    pub fn normalize_request(requested: Option<u64>) -> Option<u64> {
        requested.filter(|&tokens| tokens > 0)
    }

    pub fn maximum_tokens(&self) -> u64 {
        self.requested.unwrap_or(self.target)
    }

    /// `card_bytes` are the coordinator GPUs' totals; `context` the compiled
    /// serving context; `requested` 0 or `None` means automatic. KV target,
    /// min and max come from the installed memory policy.
    pub fn resolve(card_bytes: &[u64], context: u64, requested: Option<u64>, unit_rows: u64) -> Self {
        let requested = Self::normalize_request(requested);
        let smallest = card_bytes.iter().copied().min().unwrap_or(u64::MAX);
        let policy = MemoryPolicy::current();
        let target_default = if smallest <= SMALL_CARD_BYTES { cuteafd_core::memory_policy::SMALL_CARD_KV_TARGET_TOKENS }
            else { cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS };
        let target = policy.kv_target.unwrap_or(target_default.max(context));
        let minimum = policy.kv_min.unwrap_or(context.min(AGENTIC_FLOOR_TOKENS).min(target));
        let floor = match requested {
            Some(n) => n.max(context),
            None => context.max(minimum),
        };
        let max = requested.unwrap_or_else(|| policy.kv_max.unwrap_or(target).max(target));
        Self { context: context.into(), compiled_context: context, requested, target, floor, ceiling: u64::MAX, max, unit_rows }
    }

    /// Apply the shared context requirement without changing workspace extents.
    pub fn with_context(mut self, context: super::planner::Context) -> Self {
        self.context = context;
        self.compiled_context = context.maximum;
        self.target = MemoryPolicy::current().kv_target.unwrap_or(self.target).max(context.required());
        let minimum = MemoryPolicy::current().kv_min
            .unwrap_or(context.maximum.min(AGENTIC_FLOOR_TOKENS).min(self.target));
        self.floor = self.requested.unwrap_or(minimum).max(context.required());
        self
    }

    /// Pin a selected pool for re-admission without inventing an operator request.
    pub fn pin_selected(&mut self, tokens: u64) {
        self.target = tokens;
        self.floor = tokens;
        self.ceiling = tokens;
        self.max = tokens;
    }

    pub fn validate_explicit(&self) -> Result<(), super::PlacementError> {
        if let Some(requested) = self.requested.filter(|&n| n > 0 && n < self.context.required()) {
            return Err(super::PlacementError::RequestedBelowContext {
                requested, floor: self.context.required(),
            });
        }
        Ok(())
    }

    pub(crate) fn wanted_units(&self) -> u64 {
        self.requested.unwrap_or(self.target).div_ceil(self.unit_rows.max(1))
    }

    pub(crate) fn ceiling_units(&self) -> u64 {
        self.ceiling / self.unit_rows.max(1)
    }

    /// Automatic growth bound: KV max within the index limit.
    pub(crate) fn max_units(&self) -> u64 {
        self.max.min(self.ceiling) / self.unit_rows.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capped_pro_has_the_same_planned_and_sampled_pool_extent() {
        let planned = CoordinatorCard::planned(32 << 30, Some(188));
        let sampled = CoordinatorCard { physical_bytes: super::super::inventory::PRO_TOTAL_BYTES,
            logical_bytes: 32 << 30 };
        assert_eq!(planned, sampled);
        assert_eq!(CoordinatorCard::maximum_pool(&[planned], 262_144, None, 64), 2 << 20);
        assert_eq!(CoordinatorCard::maximum_pool(&[planned], 262_144, Some(0), 64),
            CoordinatorCard::maximum_pool(&[sampled], 262_144, None, 64));
        assert_eq!(CoordinatorCard::planned(32 << 30, None), planned);
        assert_eq!(CoordinatorCard::maximum_pool(&[CoordinatorCard::planned(28 << 30, Some(170))],
            262_144, None, 64), 1 << 20);
    }

    #[test]
    fn targets_follow_the_smallest_card_and_the_context() {
        let gib = 1u64 << 30;
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, None, 256).target, 2 << 20);
        assert_eq!(PoolPolicy::resolve(&[96 * gib, 32 * gib], 131_072, None, 256).target, 1 << 20);
        assert_eq!(PoolPolicy::resolve(&[32 * gib], 1 << 21, None, 256).target, 1 << 21);
        let explicit = PoolPolicy::resolve(&[96 * gib], 131_072, Some(32_768), 256);
        assert_eq!((explicit.requested, explicit.floor), (Some(32_768), 131_072));
        assert_eq!(explicit.validate_explicit(), Err(super::super::PlacementError::RequestedBelowContext {
            requested: 32_768, floor: 131_072,
        }));
        assert!(PoolPolicy::resolve(&[96 * gib], 32_768, Some(32_768), 256)
            .validate_explicit().is_ok());
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, Some(0), 256).floor, 131_072);
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, Some(0), 256).requested, None);

    }
}
