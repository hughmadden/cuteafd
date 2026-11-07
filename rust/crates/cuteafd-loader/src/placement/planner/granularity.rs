//! Direct cudaMalloc backing on SM120, CUDA 13.2 LAZY. Requested payload stays
//! with its allocation; this opaque owner charges only additional backing pages.
use super::*;

pub const ALLOCATION_GRANULARITY_GROUP: &str = "allocation granularity";
const PAGE: u64 = 2 << 20;
const SMALL: u64 = 1 << 20;

/// Explicit physical requests behind one logical owner, not its tensor views.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GranularityBacking {
    pub backing: String,
    pub requests: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GranularityEvidence {
    /// Large allocations round independently; small requests share driver blocks.
    /// This is not a packing guarantee (e.g. repeated 576 KiB requests fragment).
    Forecast,
}

/// Qualified only for discrete SM120 direct cudaMalloc, driver/runtime 13020,
/// CUDA_MODULE_LOADING=LAZY. Never apply this physical model to GB10/SM121.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationGranularity {
    pub id: String,
    pub location: Location,
    pub lifetime: Lifetime,
    pub backings: Vec<GranularityBacking>,
    pub evidence: GranularityEvidence,
}

fn sum_requests(requests: impl IntoIterator<Item = u64>) -> Result<u64, Refusal> {
    requests.into_iter().try_fold(0u64, |sum, bytes| {
        if bytes == 0 { return Err(Refusal::InvalidFact("empty cudaMalloc request".into())); }
        sum.checked_add(bytes).ok_or(Refusal::Overflow("cudaMalloc requests"))
    })
}

fn round(bytes: u64) -> Result<u64, Refusal> {
    bytes.div_ceil(PAGE).checked_mul(PAGE).ok_or(Refusal::Overflow("cudaMalloc backing"))
}

/// Sizes-only forecast, calibrated against Kimi's 3,935 core requests:
/// 1,244,796,544 B forecast vs 1,242,699,392 B physical slack (+0.16876%).
/// #1454, 2026-11-08. No multiplier, residual or safety margin.
pub fn allocation_granularity_forecast(requests: &[u64]) -> Result<u64, Refusal> {
    let requested = sum_requests(requests.iter().copied())?;
    let (mut large, mut small) = (0u64, 0u64);
    for &bytes in requests {
        if bytes > SMALL {
            large = large.checked_add(round(bytes)?).ok_or(Refusal::Overflow("large cudaMalloc backing"))?;
        } else {
            small = small.checked_add(bytes).ok_or(Refusal::Overflow("small cudaMalloc backing"))?;
        }
    }
    large.checked_add(round(small)?).and_then(|bytes| bytes.checked_sub(requested))
        .ok_or(Refusal::Overflow("cudaMalloc granularity"))
}

impl AllocationGranularity {
    pub fn requests(&self) -> Vec<u64> {
        self.backings.iter().flat_map(|owner| owner.requests.iter().copied()).collect()
    }

    pub fn bytes(&self) -> Result<u64, Refusal> {
        if self.id.is_empty() || self.backings.is_empty()
            || !matches!(self.location, Location { device: Device::Coordinator(_), domain: Domain::Device }) {
            return Err(Refusal::InvalidFact("SM120 cudaMalloc granularity requires a named coordinator owner".into()));
        }
        allocation_granularity_forecast(&self.requests())
    }

    pub fn basis(&self) -> Basis { Basis::Estimated }

    pub fn source(&self) -> String {
        format!("{} (SM120 direct cudaMalloc, driver/runtime13020 LAZY, #1454 2026-11-08, sizes-only forecast)", self.id)
    }

    pub(super) fn validate_backings(&self, facts: &Facts, seen: &mut BTreeSet<String>) -> Result<(), Refusal> {
        self.bytes()?;
        for backing in &self.backings {
            let owner = facts.allocations.iter().chain(&facts.deferred).find(|owner| owner.id == backing.backing);
            if backing.backing.is_empty() || !seen.insert(backing.backing.clone())
                || !owner.is_some_and(|owner| owner.view.is_none() && owner.location == self.location
                    && owner.lifetime == self.lifetime && sum_requests(backing.requests.iter().copied()) == Ok(owner.bytes)) {
                return Err(Refusal::InvalidFact(format!("invalid cudaMalloc granularity backing {}", backing.backing)));
            }
        }
        Ok(())
    }
}

/// Fixed request inventory, activated with the same finite placement profile as
/// its payload. A logical aggregate alone is never sufficient to call this.
pub fn allocation_granularity_component(owner: AllocationGranularity, maximum: u64) -> Result<Component, Refusal> {
    let bytes = owner.bytes()?;
    Ok(Component { activation: Activation::always(), bytes: ByteCurve::fixed(bytes, maximum),
        kind: ComponentKind::AllocationGranularity(owner) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_kimi_requests_match_physical_slack_within_one_block() {
        let histogram = [(384,69),(512,69),(1024,24),(1280,24),(3072,72),(3584,92),(7168,276),
            (12288,276),(14336,1087),(24576,393),(28672,186),(36864,24),(49152,69),(589824,69),
            (3145728,69),(3211264,69),(3440640,24),(8257536,24),(12582912,48),(12845056,92),
            (19267584,184),(21233664,24),(33030144,276),(66060288,393),(484442112,1),(968884224,1)];
        let requests = histogram.into_iter().flat_map(|(bytes, count)| std::iter::repeat_n(bytes, count)).collect::<Vec<_>>();
        assert_eq!(requests.len(), 3935);
        assert_eq!(sum_requests(requests.iter().copied()).unwrap(), 43_172_882_816);
        let forecast = allocation_granularity_forecast(&requests).unwrap();
        assert_eq!(forecast, 1_244_796_544);
        assert_eq!(forecast - 1_242_699_392, PAGE);
        assert!(forecast - 1_242_699_392 < 1_242_699_392 / 100);
    }

    #[test]
    fn small_requests_share_blocks_and_large_requests_round_independently() {
        assert_eq!(allocation_granularity_forecast(&[]).unwrap(), 0);
        for (requests, physical) in [(vec![SMALL; 2], PAGE), (vec![SMALL + 1; 2], 2 * PAGE),
            (vec![63 << 20, 14 << 10, 24 << 10], 66 << 20)] {
            assert_eq!(allocation_granularity_forecast(&requests).unwrap() + requests.iter().sum::<u64>(), physical);
        }
        assert!(allocation_granularity_forecast(&[0]).is_err());
        assert!(allocation_granularity_forecast(&[u64::MAX]).is_err());
    }
}
