//! How many DFlash2 drafts each sequence verifies (glmrt v9's adaptive K1-K7
//! on the shared policy core, `crate::draft_policy`).
//!
//! Each sequence's conditional acceptance per draft position (its last 16
//! outcomes, a 3-in-4 prior) is refined by the frozen logistic calibration
//! from glmrt (fit on 32 fixed-K7 GLM-5.3 K4 requests) with the selector's
//! margin, best probability, entropy and rank. The shared allocator admits
//! draft positions across sequences while expected committed tokens per
//! millisecond improve, priced by a verify-step fit seeded with the
//! deployment's measured single-sequence table. Identical sequences plan as
//! one group. The first four cycles of a sequence verify five drafts; a lone
//! warm sequence keeps five unless the plan beats them by 2%.
pub(crate) use crate::draft_policy::{CycleCost, DraftHistory, DraftSkip};
pub(crate) use crate::draft_policy::Shape;
use crate::draft_policy::{self, Base, Drafter};

pub(crate) const START_DRAFTS: usize = 5;
/// A single sequence keeps five drafts unless the schedule beats it by 2%.
const REFERENCE_MARGIN: f64 = 1.02;

// glmrt dflash2_confidence.rs (frozen fit 6d34604f...).
const MEAN: [f64; 6] = [1.0337757155740168, 1.5042334365117003, 4.105818737585014, 0.5404882231666059,
    0.18709320564567553, 0.36878205711857714];
const SCALE: [f64; 6] = [0.6869594087005682, 0.8183292270430992, 3.7913657744791154, 0.6412433617632998,
    0.4768998480274627, 0.23888156204918426];
const BETA: [f64; 7] = [1.3954686667753908, 0.05344836366609415, -0.3418558408836222, 1.7687177070514315,
    -0.5006880248156605, -0.21189320574726964, -0.09773211967803144];

/// Calibrated conditional acceptance per position from the history rates
/// and the selector features; None when a feature is out of range.
pub(crate) fn calibrated_confidence(history: &[f64], features: &[[f32; 4]]) -> Option<Vec<f64>> {
    if history.is_empty() || history.len() > 7 || features.len() < history.len() {
        return None;
    }
    let logit = |p: f64| {
        let p = p.clamp(1e-4, 1.0 - 1e-4);
        (p / (1.0 - p)).ln()
    };
    history.iter().zip(features).enumerate().map(|(index, (&prior, &[margin, probability, entropy, rank]))| {
        let valid = prior.is_finite() && (0.0..=1.0).contains(&prior)
            && [margin, probability, entropy, rank].iter().all(|x| x.is_finite())
            && margin >= 0.0 && probability > 0.0 && probability <= 1.0
            && (-1e-5..=2.80).contains(&entropy) && (0.0..16.0).contains(&rank) && rank.fract() == 0.0;
        if !valid {
            return None;
        }
        let x = [logit(prior), f64::from(margin).ln_1p(), logit(f64::from(probability)), f64::from(entropy),
            f64::from(rank).ln_1p(), (index + 1) as f64 / 7.0];
        let z = (BETA[0] + (0..6).map(|i| BETA[i + 1] * (x[i] - MEAN[i]) / SCALE[i]).sum::<f64>()).clamp(-40.0, 40.0);
        Some(1.0 / (1.0 + (-z).exp()))
    }).collect()
}

/// Conditional acceptance of `history` refined by the draft's selector features.
pub(crate) fn confidence(history: &DraftHistory, features: &[[f32; 4]]) -> Vec<f64> {
    let rates = history.conditional(features.len());
    calibrated_confidence(&rates, features).unwrap_or(rates)
}

/// Table-scale cost of a row whose routes an identical row already reads
/// (4 identical sequences: 46 ms plain vs 35.6 ms for one).
const DUPLICATE_ROW_MS: f64 = 3.5;

/// The cycle cost of a DFlash deployment from its single-sequence verify
/// table (`points`: (rows, ms), increasing, first at one row): duplicate
/// rows of identical sequences at `DUPLICATE_ROW_MS`, a 5 ms draft step and
/// 0.3 ms of host time to start from; serving refits all of it.
pub(crate) fn step_cost(points: &[(usize, f64)], max_rows: usize) -> CycleCost {
    CycleCost::new(points, max_rows).duplicate_rows(DUPLICATE_ROW_MS).drafts(5.0, 0.0, 0.3)
}

/// One sequence's inputs to a step's draft plan.
pub(crate) struct PlanInput<'h> {
    /// Identical sequences (same position and token digest) share a key:
    /// they route alike and draft alike, so the plan prices them as one group.
    pub key: (usize, u64),
    pub history: &'h DraftHistory,
    /// Selector features of the sequence's DFlash2 draft (None: no draft).
    pub features: Option<&'h [[f32; 4]]>,
    /// Most drafts the sequence may verify this step.
    pub limit: usize,
}

/// DFlash2 draft counts per sequence: `fixed` (within each limit), or the
/// adaptive [`plan`] over groups of identical drafting sequences, priced with
/// the sequences that do not draft.
pub(crate) fn plan_counts(inputs: &[PlanInput<'_>], fixed: Option<usize>, cost: &CycleCost) -> Vec<usize> {
    let indices: Vec<usize> = (0..inputs.len()).filter(|&i| inputs[i].features.is_some()).collect();
    let mut counts = vec![0; inputs.len()];
    if let Some(fixed) = fixed {
        for &i in &indices {
            counts[i] = fixed.min(inputs[i].limit).min(inputs[i].features.map_or(0, <[_]>::len));
        }
        return counts;
    }
    if indices.is_empty() {
        return counts;
    }
    let mut members: Vec<Vec<usize>> = Vec::new();
    for &i in &indices {
        match members.iter_mut().find(|m| inputs[m[0]].key == inputs[i].key) {
            Some(group) => group.push(i),
            None => members.push(vec![i]),
        }
    }
    let groups: Vec<Group<'_>> = members.iter().map(|m| Group {
        history: inputs[m[0]].history,
        confidence: confidence(inputs[m[0]].history, inputs[m[0]].features.unwrap_or(&[])),
        room: m.iter().map(|&i| inputs[i].limit).min().unwrap_or(0),
        members: m.len(),
    }).collect();
    let others: Vec<_> = inputs.iter().filter(|i| i.features.is_none()).map(|i| i.key).collect();
    let distinct = others.iter().collect::<std::collections::HashSet<_>>().len();
    for (m, n) in members.iter().zip(plan(&groups, (others.len(), distinct), cost)) {
        for &i in m {
            counts[i] = n;
        }
    }
    counts
}

/// GLM-5.3 EXL3 K4, 1 RTX PRO 6000 (325 W) + 4 Sparks TP4: verify step ms by
/// rows, the serving fit's prior. Measured on the Sparks (p7 Spark images,
/// coordinator b3521aa: sparse MLA reads only selected tokens, few-row head)
/// with glm-golden --bench-verify 64, 512 tokens of context, median of 7.
pub(crate) const K4_TP4_STEP_MS: [(usize, f64); 15] = [(1, 35.0), (2, 50.5), (3, 60.2), (4, 67.2), (5, 77.6),
    (6, 85.3), (7, 92.8), (8, 102.6), (10, 116.2), (12, 127.8), (16, 148.4), (24, 190.6), (32, 223.7), (48, 269.4),
    (64, 317.8)];

/// Sequences that draft together: identical ones (same tokens, same
/// position) form one group of `members`, with one draft between them.
pub(crate) struct Group<'h> {
    pub history: &'h DraftHistory,
    /// Conditional acceptance per draft position.
    pub confidence: Vec<f64>,
    /// Most drafts the group may verify.
    pub room: usize,
    pub members: usize,
}

/// The draft count of every group (`base`: rows and distinct rows of the
/// sequences that do not draft): cold groups verify `START_DRAFTS`; a lone
/// warm sequence keeps them unless the plan is 2% better.
pub(crate) fn plan(groups: &[Group<'_>], base: (usize, usize), cost: &CycleCost) -> Vec<usize> {
    let base = Base { rows: base.0, distinct: base.1 };
    let core: Vec<draft_policy::Group> = groups.iter().map(|g| {
        let confidence = g.confidence[..g.confidence.len().min(g.room)].to_vec();
        let minimum = if g.history.cold() { START_DRAFTS.min(confidence.len()) } else { 0 };
        draft_policy::Group { confidence, members: g.members, minimum }
    }).collect();
    let (lengths, rate) = draft_policy::allocate(&core, base, Drafter::Block, cost);
    if groups.len() == 1 && groups[0].members == 1 && base.rows == 0 && !groups[0].history.cold() {
        // The reference is exactly START_DRAFTS drafts (glmrt prices K5 alone):
        // a reference free to extend past them would equal any longer best
        // plan and cap every warm sequence at five.
        let n = START_DRAFTS.min(core[0].confidence.len());
        if lengths[0] != n && rate < draft_policy::rate(&core, &[n], base, Drafter::Block, cost) * REFERENCE_MARGIN {
            return vec![n];
        }
    }
    lengths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_frozen_glmrt_calibration() {
        let actual = calibrated_confidence(&[0.75, 0.6, 0.8, 0.4, 0.9, 0.5, 0.7], &[
            [0.0, 0.0625, 2.765625, 0.0], [1.0, 0.5, 1.0, 2.0], [4.0, 0.96875, 0.25, 0.0], [0.125, 0.25, 1.5, 4.0],
            [8.0, 1.0, 0.0, 0.0], [2.0, 0.75, 0.75, 1.0], [3.0, 0.875, 0.5, 0.0],
        ]).unwrap();
        let expected = [0.06228691035179077, 0.27691696975472285, 0.7942950332348667, 0.1161585187943506,
            0.9809519714258056, 0.37597938760974686, 0.5693257689991642];
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!(calibrated_confidence(&[0.75], &[[1.0, 0.5, 1.0, 16.0]]).is_none());
    }

    fn group<'h>(history: &'h DraftHistory, rate: f64, members: usize) -> Group<'h> {
        Group { history, confidence: vec![rate; 7], room: 7, members }
    }

    fn warm(proposed: usize, accepted: usize) -> DraftHistory {
        let mut history = DraftHistory::default();
        for _ in 0..8 {
            history.observe(proposed, accepted);
        }
        history
    }

    #[test]
    fn schedule_stops_where_throughput_peaks() {
        let cost = step_cost(&K4_TP4_STEP_MS, 64);
        let history = warm(3, 2);
        assert_eq!(plan(&[group(&history, 0.99, 1)], (0, 0), &cost), vec![7]);
        assert_eq!(plan(&[group(&history, 0.2, 1)], (0, 0), &cost), vec![0]);
        // The better sequence gets rows first.
        let lengths = plan(&[group(&history, 0.95, 1), group(&history, 0.3, 1)], (0, 0), &cost);
        assert!(lengths[0] > lengths[1]);
    }

    #[test]
    fn warm_confident_sequence_verifies_seven() {
        // glmrt compares the best plan with exactly five drafts; a reference
        // schedule that may extend past five capped warm sequences at five.
        let cost = step_cost(&K4_TP4_STEP_MS, 64);
        let history = warm(7, 7);
        assert_eq!(plan(&[group(&history, 0.97, 1)], (0, 0), &cost), vec![7]);
        assert!(plan(&[group(&history, 0.5, 1)], (0, 0), &cost)[0] < 5);
    }

    #[test]
    fn cold_sequences_verify_five() {
        let cost = step_cost(&K4_TP4_STEP_MS, 64);
        let history = DraftHistory::default();
        assert_eq!(plan(&[group(&history, 0.1, 1)], (0, 0), &cost), vec![5]);
        let narrow = Group { room: 3, ..group(&history, 0.1, 1) };
        assert_eq!(plan(&[narrow], (0, 0), &cost), vec![3]);
    }

    #[test]
    fn identical_sequences_share_their_rows() {
        let cost = step_cost(&K4_TP4_STEP_MS, 64);
        let warm = warm(3, 2);
        // Four identical sequences price a plain step near 46 ms, not 74.
        let distinct: usize = plan(&[group(&warm, 0.6, 1), group(&warm, 0.6, 1), group(&warm, 0.6, 1),
            group(&warm, 0.6, 1)], (0, 0), &cost).iter().sum::<usize>();
        let shared = plan(&[group(&warm, 0.6, 4)], (0, 0), &cost)[0] * 4;
        assert!(distinct > 0 && shared <= distinct, "{distinct} {shared}");
        let ms = |rows, distinct, sequences| cost.cycle_ms(Shape { rows, distinct, sequences }, 0, true);
        assert!((ms(4, 1, 4) - ms(1, 1, 1) - 3.0 * DUPLICATE_ROW_MS).abs() < 1e-9);
        assert!(ms(4, 4, 4) > ms(4, 1, 4) + 20.0);
    }

    #[test]
    fn prior_matches_the_single_ratio_model_before_observations() {
        // Before serving observes anything the fit is the table itself (the
        // previous model at ratio 1): 5 ms draft + 0.3 ms host on top.
        let cost = step_cost(&K4_TP4_STEP_MS, 64);
        for (rows, ms) in K4_TP4_STEP_MS {
            let got = cost.cycle_ms(Shape { rows, distinct: rows, sequences: 1 }, 0, true);
            assert!((got - ms - 5.3).abs() < 1e-9, "{rows}: {got}");
        }
    }
}
