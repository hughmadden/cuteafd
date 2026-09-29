//! How many DFlash2 drafts each sequence verifies (glmrt v9's adaptive K1-K7).
//!
//! Each sequence keeps its last 16 (proposed, accepted) outcomes; a Beta(3, 4)
//! prior per draft position turns them into conditional acceptance rates
//! (a position is observed only when every earlier draft was accepted). The
//! frozen logistic calibration from glmrt (fit on 32 fixed-K7 GLM-5.3 K4
//! requests) refines those rates with the selector's margin, best
//! probability, entropy and rank. The joint schedule then admits draft
//! positions across sequences in order of survival probability (the product
//! of the conditional rates) while expected committed tokens per second,
//! priced by the measured verify-step cost of the total row count, improve.
//! The first four cycles of a sequence verify five drafts.
use std::collections::VecDeque;

const HISTORY: usize = 16;
const COLD_START_CYCLES: usize = 4;
pub(crate) const START_DRAFTS: usize = 5;
const PRIOR_SUCCESSES: usize = 3;
const PRIOR_TRIALS: usize = 4;
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

/// A sequence's recent draft outcomes.
#[derive(Debug, Clone, Default)]
pub(crate) struct DraftHistory {
    outcomes: VecDeque<(usize, usize)>,
}

impl DraftHistory {
    pub fn cold(&self) -> bool {
        self.outcomes.len() < COLD_START_CYCLES
    }

    pub fn observe(&mut self, proposed: usize, accepted: usize) {
        if proposed == 0 {
            return;
        }
        self.outcomes.push_back((proposed, accepted.min(proposed)));
        while self.outcomes.len() > HISTORY {
            self.outcomes.pop_front();
        }
    }

    /// Conditional acceptance of positions 1..=max (censored after a miss).
    pub fn conditional(&self, max: usize) -> Vec<f64> {
        (1..=max).map(|position| {
            let (mut trials, mut successes) = (PRIOR_TRIALS, PRIOR_SUCCESSES);
            for &(proposed, accepted) in &self.outcomes {
                if proposed >= position && accepted + 1 >= position {
                    trials += 1;
                    successes += usize::from(accepted >= position);
                }
            }
            successes as f64 / trials as f64
        }).collect()
    }

    /// Conditional acceptance refined by the draft's selector features.
    pub fn confidence(&self, features: &[[f32; 4]]) -> Vec<f64> {
        let history = self.conditional(features.len());
        calibrated_confidence(&history, features).unwrap_or(history)
    }
}

/// Cycle milliseconds of a verify step plus the draft step. Rows of
/// identical sequences (same tokens at the same position) route alike and
/// share their Spark expert reads, so a step costs the measured
/// single-sequence cost of its distinct rows plus `DUPLICATE_ROW_MS` per
/// duplicate row, scaled by a ratio that tracks what serving observes.
#[derive(Debug, Clone)]
pub(crate) struct StepCost {
    prior: Vec<f64>,
    ratio: f64,
    draft_ms: f64,
}

/// Coordinator-side cost of a row whose routes another row already reads
/// (4 identical sequences: 46 ms plain vs 35.6 ms for one).
const DUPLICATE_ROW_MS: f64 = 3.5;

impl StepCost {
    /// `points` are (rows, ms) with increasing rows, the first at one row.
    pub fn new(points: &[(usize, f64)], max_rows: usize) -> Self {
        let prior = (0..=max_rows).map(|rows| {
            let rows = rows.max(1);
            let upper = points.iter().position(|&(r, _)| r >= rows).unwrap_or(points.len() - 1).max(1);
            let ((r0, m0), (r1, m1)) = (points[upper - 1], points[upper]);
            m0 + (m1 - m0) * (rows as f64 - r0 as f64) / (r1 as f64 - r0 as f64)
        }).collect();
        Self { prior, ratio: 1.0, draft_ms: 5.0 }
    }

    fn model(&self, rows: usize, distinct: usize) -> f64 {
        let distinct = distinct.clamp(1, rows.max(1));
        self.prior[distinct.min(self.prior.len() - 1)] + DUPLICATE_ROW_MS * (rows - distinct) as f64
    }

    /// Cycle ms (draft + verify) of `rows` rows, `distinct` of them distinct.
    pub fn ms(&self, rows: usize, distinct: usize) -> f64 {
        self.ratio * self.model(rows, distinct) + self.draft_ms
    }

    /// Folds one observed draft-step time in.
    pub fn observe_draft(&mut self, ms: f64) {
        if ms.is_finite() && ms > 0.0 {
            self.draft_ms += 0.1 * (ms.min(4.0 * self.draft_ms) - self.draft_ms);
        }
    }

    /// Folds one observed verify-step time in.
    pub fn observe(&mut self, rows: usize, distinct: usize, ms: f64) {
        if ms.is_finite() && ms > 0.0 {
            let ratio = (ms / self.model(rows, distinct)).clamp(0.5, 2.0);
            self.ratio += 0.1 * (ratio - self.ratio);
        }
    }
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
pub(crate) fn plan_counts(inputs: &[PlanInput<'_>], fixed: Option<usize>, cost: &StepCost) -> Vec<usize> {
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
        confidence: inputs[m[0]].history.confidence(inputs[m[0]].features.unwrap_or(&[])),
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

/// Longest run of steps without a draft step after plans that verified none.
const MAX_DRAFT_SKIP: usize = 8;

/// After plans that verify no drafts, skip drafting for a while (doubling up
/// to `MAX_DRAFT_SKIP` steps): a draft step costs about a verified row.
#[derive(Debug, Clone)]
pub(crate) struct DraftSkip {
    skip: usize,
    next: usize,
}

impl Default for DraftSkip {
    fn default() -> Self {
        Self { skip: 0, next: 1 }
    }
}

impl DraftSkip {
    /// Whether this step drafts.
    pub fn drafts(&self) -> bool {
        self.skip == 0
    }

    /// After planning a step: `drafted` when it drafted, `none_planned` when
    /// the plan verifies no drafts (adaptive plans only).
    pub fn after(&mut self, drafted: bool, none_planned: bool) {
        if self.skip > 0 {
            self.skip -= 1;
        } else if drafted {
            if none_planned {
                self.skip = self.next;
                self.next = (self.next * 2).min(MAX_DRAFT_SKIP);
            } else {
                self.next = 1;
            }
        }
    }
}

/// GLM-5.3 EXL3 K4, 1 RTX PRO 6000 (325 W) + 4 Sparks TP4: verify step ms by
/// rows. Measured on the Sparks before the sparse MLA read only selected
/// tokens and the few-row head (35.6 / 111.8 / 260.2 ms at 1 / 8 / 32 rows),
/// less the coordinator time those saved at each row count (glm-golden
/// --bench-verify --skip-routed-experts, 512 tokens of context); re-measure
/// with --bench-verify on the Sparks.
pub(crate) const K4_TP4_STEP_MS: [(usize, f64); 15] = [(1, 34.5), (2, 48.9), (3, 59.5), (4, 69.4), (5, 77.4),
    (6, 85.8), (7, 93.8), (8, 102.1), (10, 118.8), (12, 134.0), (16, 157.9), (24, 197.8), (32, 223.6), (48, 284.7),
    (64, 327.1)];

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

/// Draft counts per group maximizing expected committed tokens per
/// millisecond. `minimum[g]` drafts are mandatory; `base` is the (rows,
/// distinct rows) of sequences that do not draft; `cost(rows, distinct)` a
/// cycle's ms. Positions are admitted in order of survival probability.
pub(crate) fn schedule(groups: &[Group<'_>], minimum: &[usize], base: (usize, usize),
    cost: impl Fn(usize, usize) -> f64) -> (Vec<usize>, f64) {
    let mut lengths = minimum.to_vec();
    let (mut rows, mut distinct) = base;
    let mut expected = base.0 as f64;
    let mut candidates = Vec::new();
    for (g, group) in groups.iter().enumerate() {
        let rates = &group.confidence[..group.confidence.len().min(group.room)];
        rows += group.members * (1 + minimum[g]);
        distinct += 1 + minimum[g];
        expected += group.members as f64;
        let mut survival = 1.0;
        for (position, &rate) in rates.iter().enumerate() {
            survival *= rate.clamp(0.0, 1.0);
            if position < minimum[g] {
                expected += group.members as f64 * survival;
            } else if survival > 0.0 {
                candidates.push((survival, position + 1, g));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut best = (lengths.clone(), expected / cost(rows, distinct));
    for (survival, length, g) in candidates {
        if length != lengths[g] + 1 {
            continue;
        }
        lengths[g] = length;
        rows += groups[g].members;
        distinct += 1;
        expected += groups[g].members as f64 * survival;
        let rate = expected / cost(rows, distinct);
        if rate > best.1 {
            best = (lengths.clone(), rate);
        }
    }
    best
}

/// The draft count of every group: cold groups verify `START_DRAFTS`; a
/// lone warm sequence keeps them unless the schedule is 2% better.
pub(crate) fn plan(groups: &[Group<'_>], base: (usize, usize), cost: &StepCost) -> Vec<usize> {
    let cost = |rows: usize, distinct: usize| cost.ms(rows, distinct);
    let minimum: Vec<usize> = groups.iter()
        .map(|g| if g.history.cold() { START_DRAFTS.min(g.room).min(g.confidence.len()) } else { 0 }).collect();
    let (lengths, rate) = schedule(groups, &minimum, base, cost);
    if groups.len() == 1 && groups[0].members == 1 && base.0 == 0 && !groups[0].history.cold() {
        let reference = [START_DRAFTS.min(groups[0].room).min(groups[0].confidence.len())];
        let (fixed, fixed_rate) = schedule(groups, &reference, base, cost);
        if lengths[0] != reference[0] && rate < fixed_rate * REFERENCE_MARGIN {
            return vec![fixed[0].min(reference[0])];
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

    #[test]
    fn history_censors_positions_after_a_miss() {
        let mut history = DraftHistory::default();
        history.observe(5, 2);
        let rates = history.conditional(5);
        // Positions 1, 2 accepted, 3 rejected, 4 and 5 unobserved.
        assert_eq!(rates, vec![4.0 / 5.0, 4.0 / 5.0, 3.0 / 5.0, 0.75, 0.75]);
    }

    fn group<'h>(history: &'h DraftHistory, rate: f64, members: usize) -> Group<'h> {
        Group { history, confidence: vec![rate; 7], room: 7, members }
    }

    #[test]
    fn schedule_stops_where_throughput_peaks() {
        let cost = StepCost::new(&K4_TP4_STEP_MS, 64);
        let c = |rows, distinct| cost.ms(rows, distinct);
        let history = DraftHistory::default();
        assert_eq!(schedule(&[group(&history, 0.99, 1)], &[0], (0, 0), c).0, vec![7]);
        assert_eq!(schedule(&[group(&history, 0.2, 1)], &[0], (0, 0), c).0, vec![0]);
        // The better sequence gets rows first.
        let (lengths, _) = schedule(&[group(&history, 0.95, 1), group(&history, 0.3, 1)], &[0, 0], (0, 0), c);
        assert!(lengths[0] > lengths[1]);
    }

    #[test]
    fn cold_sequences_verify_five() {
        let cost = StepCost::new(&K4_TP4_STEP_MS, 64);
        let history = DraftHistory::default();
        assert_eq!(plan(&[group(&history, 0.1, 1)], (0, 0), &cost), vec![5]);
        let narrow = Group { room: 3, ..group(&history, 0.1, 1) };
        assert_eq!(plan(&[narrow], (0, 0), &cost), vec![3]);
    }

    #[test]
    fn identical_sequences_share_their_rows() {
        let cost = StepCost::new(&K4_TP4_STEP_MS, 64);
        let warm = DraftHistory { outcomes: [(3, 2); 8].into() };
        // Four identical sequences price a plain step near 46 ms, not 74.
        let distinct: usize = plan(&[group(&warm, 0.6, 1), group(&warm, 0.6, 1), group(&warm, 0.6, 1),
            group(&warm, 0.6, 1)], (0, 0), &cost).iter().sum::<usize>();
        let shared = plan(&[group(&warm, 0.6, 4)], (0, 0), &cost)[0] * 4;
        assert!(distinct > 0 && shared <= distinct, "{distinct} {shared}");
        assert!((cost.ms(4, 1) - cost.ms(1, 1) - 3.0 * DUPLICATE_ROW_MS).abs() < 1e-9);
        assert!(cost.ms(4, 4) > cost.ms(4, 1) + 20.0);
    }
}
