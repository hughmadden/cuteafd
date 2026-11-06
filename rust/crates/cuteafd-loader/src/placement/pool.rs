//! The one KV pool target: 2M tokens on cards above 32 GiB, 1M at 32 GiB or
//! less, never below the compiled context.
use cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS;

/// Agentic context floor for automatic Spark-free layouts.
pub const AGENTIC_FLOOR_TOKENS: u64 = 262_144;
const SMALL_CARD_BYTES: u64 = 32 << 30;
const SMALL_CARD_POOL_TOKENS: u64 = 1 << 20;

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
        PoolPolicy::resolve(&physical, context, requested, page_rows, false).maximum_tokens()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolPolicy {
    /// Fixed or operator-set context requirement; zero for automatic context.
    /// Independent of the mutable fit floor and the selected pool.
    pub compiled_context: u64,
    /// An explicit pool is strict: admission fails rather than shrink it.
    pub requested: Option<u64>,
    /// Automatic pools stop here.
    pub target: u64,
    /// Spark-free layouts fail below this; pools with Sparks may go lower
    /// (the serving context is then clamped to the pool).
    pub floor: u64,
    /// The largest pool the build can index; a fixed onboard fills up to it.
    pub ceiling: u64,
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
    /// serving context; `requested` 0 or `None` means automatic.
    pub fn resolve(card_bytes: &[u64], context: u64, requested: Option<u64>, unit_rows: u64,
        spark_free: bool) -> Self {
        let requested = Self::normalize_request(requested);
        let small = card_bytes.iter().any(|&bytes| bytes <= SMALL_CARD_BYTES);
        let target = if small { SMALL_CARD_POOL_TOKENS } else { DEFAULT_GPU_KV_TOKENS }.max(context);
        // Explicit pools must cover the compiled context. Only automatic,
        // Spark-free pools additionally retain the agentic floor.
        let floor = match requested {
            Some(n) => n.max(context),
            None if spark_free => context.max(AGENTIC_FLOOR_TOKENS),
            None => context,
        };
        Self { compiled_context: context, requested, target, floor,
            ceiling: u64::MAX, unit_rows }
    }

    /// Automatic checkpoint context may be clamped by serving to an explicit
    /// diagnostic pool. Fixed compiled extents must not use this mode.
    pub fn with_automatic_context(mut self, automatic: bool) -> Self {
        if automatic {
            self.compiled_context = 0;
            if let Some(requested) = self.requested {
                self.floor = requested;
            }
        }
        self
    }

    /// Pin a selected pool for re-admission without inventing an operator request.
    pub fn pin_selected(&mut self, tokens: u64) {
        self.target = tokens;
        self.floor = tokens;
        self.ceiling = tokens;
    }

    pub fn validate_explicit(&self) -> Result<(), super::PlacementError> {
        if let Some(requested) = self.requested.filter(|&n| n > 0 && n < self.compiled_context) {
            return Err(super::PlacementError::RequestedBelowContext {
                requested, floor: self.compiled_context,
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
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, None, 256, false).target, 2 << 20);
        assert_eq!(PoolPolicy::resolve(&[96 * gib, 32 * gib], 131_072, None, 256, false).target, 1 << 20);
        assert_eq!(PoolPolicy::resolve(&[32 * gib], 1 << 21, None, 256, false).target, 1 << 21);
        let explicit = PoolPolicy::resolve(&[96 * gib], 131_072, Some(32_768), 256, true);
        assert_eq!((explicit.requested, explicit.floor), (Some(32_768), 131_072));
        assert_eq!(explicit.validate_explicit(), Err(super::super::PlacementError::RequestedBelowContext {
            requested: 32_768, floor: 131_072,
        }));
        assert!(PoolPolicy::resolve(&[96 * gib], 32_768, Some(32_768), 256, true)
            .validate_explicit().is_ok());
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, Some(0), 256, true).floor, AGENTIC_FLOOR_TOKENS);
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, Some(0), 256, true).requested, None);

    }
}
