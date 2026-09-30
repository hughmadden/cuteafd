//! How many MTP drafts each Qwen 3.8 Flash Next sequence verifies, on the
//! shared policy core (`crate::draft_policy`).
//!
//! MTP drafts are sequential: a cycle runs one MTP step over every pending
//! row (the canonical history, plus each drafting sequence's first draft)
//! and one chain step per further draft, then the target verifies every
//! sequence's next token plus its drafts in one step. Acceptance is each
//! sequence's conditional rate per draft position; the allocator prices a
//! cycle as the fitted verify step of its rows plus the MTP steps of its
//! deepest sequence, both refit as the server runs.
use crate::draft_policy::{self, Base, CycleCost, DraftHistory, Drafter, Group};

/// Qwen 3.8 Flash Next EXL3 K4.25, experts local on one RTX PRO 6000 (325 W):
/// speculative verify step ms by rows of one sequence after the 1634-token
/// golden prompt (qwen4-golden --spec-decode, median of 7, mean of two runs).
/// Every row count to 33: the steps are not smooth (17 rows cost 4.5 ms more
/// than 16; 15 and 30 rows take slower programs than 16 and 31).
pub(crate) const RTX_TP1_VERIFY_MS: [(usize, f64); 37] = [
    (1, 9.1), (2, 9.7), (3, 10.9), (4, 11.4), (5, 12.3), (6, 13.1), (7, 14.0), (8, 14.5), (9, 15.2), (10, 16.0),
    (11, 16.5), (12, 17.4), (13, 17.9), (14, 18.4), (15, 20.5), (16, 19.4), (17, 23.9), (18, 24.7), (19, 25.1),
    (20, 25.9), (21, 26.3), (22, 26.8), (23, 27.4), (24, 28.3), (25, 28.9), (26, 29.5), (27, 30.1), (28, 30.5),
    (29, 30.6), (30, 34.7), (31, 31.4), (32, 32.7), (33, 33.4), (40, 41.6), (48, 46.9), (56, 50.9), (64, 55.2)];
/// MTP draft work per drafting cycle and per chained step, served (the
/// first step also runs the pending rows; 0.69 ms per step alone).
pub(crate) const RTX_TP1_MTP_MS: (f64, f64) = (0.1, 0.75);

/// The cycle cost of this deployment before serving refits it.
pub(crate) fn cycle_cost(max_rows: usize) -> CycleCost {
    CycleCost::new(&RTX_TP1_VERIFY_MS, max_rows).drafts(RTX_TP1_MTP_MS.0, RTX_TP1_MTP_MS.1, 0.0)
}

/// Conditional acceptance per draft position of every sequence (up to its
/// limit), from its own history.
pub(crate) fn acceptance(histories: &[&DraftHistory], limits: &[usize]) -> Vec<Vec<f64>> {
    histories.iter().zip(limits).map(|(history, &limit)| history.conditional(limit)).collect()
}

/// Draft depths per sequence from its conditional acceptance per position
/// (as long as it may verify), maximizing expected committed tokens per
/// millisecond; `fixed` verifies that many where allowed.
pub(crate) fn plan(confidence: &[Vec<f64>], fixed: Option<usize>, cost: &CycleCost) -> Vec<usize> {
    if let Some(fixed) = fixed {
        return confidence.iter().map(|c| fixed.min(c.len())).collect();
    }
    let groups: Vec<Group> = confidence.iter()
        .map(|c| Group { confidence: c.clone(), members: 1, minimum: 0 }).collect();
    draft_policy::allocate(&groups, Base::default(), Drafter::Chain, cost).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_follows_acceptance() {
        let cost = cycle_cost(64);
        let (mut good, mut bad) = (DraftHistory::default(), DraftHistory::default());
        for _ in 0..16 {
            good.observe(4, 4);
            bad.observe(4, 0);
        }
        let plan = |h: &[&DraftHistory], limits: &[usize], fixed| plan(&acceptance(h, limits), fixed, &cost);
        let deep = plan(&[&good], &[7], None)[0];
        assert!(deep >= 4, "{deep}");
        assert_eq!(plan(&[&bad], &[7], None), vec![0]);
        let both = plan(&[&good, &bad], &[7, 7], None);
        assert!(both[0] > both[1], "{both:?}");
        assert_eq!(plan(&[&good, &bad], &[2, 7], Some(3)), vec![2, 3]);
    }
}
