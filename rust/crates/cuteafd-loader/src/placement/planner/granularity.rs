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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceExtent {
    pub address: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GranularityEvidence {
    /// Large allocations round independently; small requests share driver blocks.
    /// This is not a packing guarantee (e.g. repeated 576 KiB requests fragment).
    Forecast,
    /// Complete live native device extents around an allocation-only boundary.
    /// Include earlier owners so their already-paid small blocks are not charged twice.
    Live { before: Vec<DeviceExtent>, after: Vec<DeviceExtent> },
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

fn extent_map(extents: &[DeviceExtent]) -> Result<BTreeMap<u64, u64>, Refusal> {
    let mut map = BTreeMap::new();
    for extent in extents {
        if extent.address == 0 || extent.bytes == 0 || map.insert(extent.address, extent.bytes).is_some() {
            return Err(Refusal::InvalidFact("invalid or duplicate live cudaMalloc extent".into()));
        }
    }
    let mut previous_end = 0;
    for (&address, &bytes) in &map {
        if address < previous_end { return Err(Refusal::InvalidFact("overlapping live cudaMalloc extents".into())); }
        previous_end = address.checked_add(bytes).ok_or(Refusal::Overflow("live cudaMalloc extent"))?;
    }
    Ok(map)
}

fn page_union(map: &BTreeMap<u64, u64>) -> Result<u64, Refusal> {
    let (mut end, mut pages) = (0u64, 0u64);
    for (&address, &bytes) in map {
        let first = address / PAGE;
        let last = address.checked_add(bytes).ok_or(Refusal::Overflow("live cudaMalloc page"))?.div_ceil(PAGE);
        pages = pages.checked_add(last.saturating_sub(first.max(end)))
            .ok_or(Refusal::Overflow("live cudaMalloc page union"))?;
        end = end.max(last);
    }
    pages.checked_mul(PAGE).ok_or(Refusal::Overflow("live cudaMalloc backing"))
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
        let requests = self.requests();
        match &self.evidence {
            GranularityEvidence::Forecast => allocation_granularity_forecast(&requests),
            GranularityEvidence::Live { before, after } => {
                let before = extent_map(before)?;
                let after = extent_map(after)?;
                if before.iter().any(|(address, bytes)| after.get(address) != Some(bytes)) {
                    return Err(Refusal::InvalidFact("cudaMalloc measurement removed or resized a baseline owner".into()));
                }
                let mut added = after.iter().filter(|(address, _)| !before.contains_key(address))
                    .map(|(_, &bytes)| bytes).collect::<Vec<_>>();
                let mut expected = requests;
                expected.sort_unstable();
                added.sort_unstable();
                if added != expected { return Err(Refusal::InvalidFact("cudaMalloc measurement requests differ from declared owners".into())); }
                let requested = sum_requests(expected)?;
                page_union(&after)?.checked_sub(page_union(&before)?).and_then(|bytes| bytes.checked_sub(requested))
                    .ok_or_else(|| Refusal::InvalidFact("live cudaMalloc increment has no nonnegative opaque slack".into()))
            }
        }
    }

    pub fn basis(&self) -> Basis {
        match self.evidence { GranularityEvidence::Forecast => Basis::Estimated, GranularityEvidence::Live { .. } => Basis::Calibrated }
    }

    pub fn source(&self) -> String {
        format!("{} (SM120 direct cudaMalloc, driver/runtime13020 LAZY, #1454 2026-11-08, {})", self.id,
            match self.evidence { GranularityEvidence::Forecast => "sizes-only forecast", GranularityEvidence::Live { .. } => "live 2 MiB page union" })
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

/// Replace only the selected owner's forecast, without reselecting pool/layout.
/// The caller's live sample must not already have paid this increment in its baseline.
pub fn reconcile_allocation_granularity(plan: &mut ResolvedPlan, id: &str,
    before: Vec<DeviceExtent>, after: Vec<DeviceExtent>) -> Result<(), Refusal> {
    let mut facts = plan.facts.clone();
    let owners = facts.allocation_granularity.iter_mut().filter(|owner| owner.id == id).collect::<Vec<_>>();
    let [owner]: [&mut AllocationGranularity; 1] = owners.try_into()
        .map_err(|_| Refusal::InvalidFact("missing or duplicate cudaMalloc granularity owner".into()))?;
    owner.evidence = GranularityEvidence::Live { before, after };
    let phases = evaluate(&facts, &plan.budgets)?;
    plan.facts = facts;
    plan.phases = phases;
    Ok(())
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

    #[test]
    fn live_union_reuses_paid_small_blocks_and_rejects_aliases_or_missing_requests() {
        let before = vec![DeviceExtent { address: 4 * PAGE, bytes: 14 << 10 }];
        let mut after = before.clone();
        after.extend([DeviceExtent { address: 4 * PAGE + (16 << 10), bytes: 24 << 10 },
            DeviceExtent { address: 8 * PAGE, bytes: 3 << 20 }]);
        let mut owner = AllocationGranularity { id: "core".into(), location: Location::coordinator(0),
            lifetime: Lifetime::RESIDENT, backings: vec![GranularityBacking { backing: "payload".into(),
                requests: vec![24 << 10, 3 << 20] }], evidence: GranularityEvidence::Live { before: before.clone(), after: after.clone() } };
        assert_eq!(owner.bytes().unwrap(), (1 << 20) - (24 << 10));
        assert_eq!(owner.basis(), Basis::Calibrated);
        owner.evidence = GranularityEvidence::Forecast;
        assert_eq!(owner.bytes().unwrap(), (3 << 20) - (24 << 10));
        for bad in [vec![after[1]], vec![after[0], after[0]], vec![after[0], DeviceExtent { address: after[0].address + 1, bytes: 1 }]] {
            owner.evidence = GranularityEvidence::Live { before: before.clone(), after: bad };
            assert!(owner.bytes().is_err());
        }
        owner.location = Location { device: Device::Spark(0), domain: Domain::Managed };
        owner.evidence = GranularityEvidence::Forecast;
        assert!(owner.bytes().is_err());
    }
}
