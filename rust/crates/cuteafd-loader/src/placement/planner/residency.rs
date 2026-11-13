//! Admission of an ordered, monotone optional weight set. The recipe supplies
//! physical owners for each prefix; shared policy reserves KV before optional
//! weights and grows KV only after fixing the resident set.
use super::*;

pub struct ResidencyRange {
    pub minimum: usize,
    pub maximum: usize,
    /// An exact prefix is mandatory and the pool is its output.
    pub exact: Option<usize>,
}

struct Candidate<'a, F> {
    count: usize,
    prepare: &'a F,
}
impl<F: Fn(&Request, usize) -> Result<Contract, Refusal>> Recipe for Candidate<'_, F> {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        (self.prepare)(request, self.count)
    }
}

/// Admit one immutable prefix. Costs must be monotone in `count` and decision
/// domains must be identical across prefixes. All ordinary owners (including
/// the preferred encoder) are admitted first and their decisions are frozen.
/// A capacity or weight-limit refusal bounds the search; invalid facts escape.
pub fn solve_residency(
    request: Request,
    budgets: Vec<Budget>,
    range: ResidencyRange,
    prepare: impl Fn(&Request, usize) -> Result<Contract, Refusal>,
) -> Result<(ResolvedPlan, usize), Refusal> {
    if range.minimum > range.maximum
        || range.exact.is_some_and(|n| n < range.minimum || n > range.maximum)
    {
        return Err(Refusal::InvalidFact("invalid ranked residency extent".into()));
    }
    let candidate = |count| Candidate { count, prepare: &prepare };
    if let Some(count) = range.exact {
        return solve(request, budgets, &candidate(count)).map(|plan| (plan, count));
    }
    let (base, selected) = search::solve_with_growth(
        request.clone(), budgets.clone(), &candidate(range.minimum), false, None)?;
    let mut pinned = request.clone();
    pinned.pool = Pin::Exact(base.pool_tokens);
    let (mut low, mut high) = (range.minimum, range.maximum);
    while low < high {
        let count = low + (high - low).div_ceil(2);
        match search::solve_with_growth(pinned.clone(), budgets.clone(),
            &candidate(count), false, Some(&selected)) {
            Ok(_) => low = count,
            Err(Refusal::Capacity { .. } | Refusal::ResidentWeightLimit { .. }) => high = count - 1,
            Err(error) => return Err(error),
        }
    }
    search::solve_with_growth(request, budgets, &candidate(low), true, Some(&selected))
        .map(|(plan, _)| (plan, low))
}
