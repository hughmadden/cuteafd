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

/// Cycle milliseconds by sequences and total verify rows: the verify step
/// plus the draft step. Each (sequences, rows) cell starts from the measured
/// single-sequence cost of that many rows (linear between points,
/// extrapolated from the last two) and follows what serving observes, since
/// sequences that route alike (the same prompt) share Spark expert reads.
#[derive(Debug, Clone)]
pub(crate) struct StepCost {
    prior: Vec<f64>,
    /// [sequences][rows] observed ms (NaN until seen).
    seen: Vec<Vec<f64>>,
    draft_ms: f64,
}

impl StepCost {
    /// `points` are (rows, ms) with increasing rows, the first at one row.
    pub fn new(points: &[(usize, f64)], max_rows: usize) -> Self {
        let prior = (0..=max_rows).map(|rows| {
            let rows = rows.max(1);
            let upper = points.iter().position(|&(r, _)| r >= rows).unwrap_or(points.len() - 1).max(1);
            let ((r0, m0), (r1, m1)) = (points[upper - 1], points[upper]);
            m0 + (m1 - m0) * (rows as f64 - r0 as f64) / (r1 as f64 - r0 as f64)
        }).collect();
        Self { prior, seen: vec![vec![f64::NAN; max_rows + 1]; max_rows + 1], draft_ms: 5.0 }
    }

    /// Verify-step ms of `rows` rows over `sequences` sequences.
    pub fn verify_ms(&self, sequences: usize, rows: usize) -> f64 {
        let rows = rows.min(self.prior.len() - 1);
        let seen = self.seen.get(sequences).map_or(f64::NAN, |r| r[rows]);
        if seen.is_nan() { self.prior[rows] } else { seen }.max(1.0)
    }

    /// Cycle ms (draft + verify).
    pub fn ms(&self, sequences: usize, rows: usize) -> f64 {
        self.verify_ms(sequences, rows) + self.draft_ms
    }

    /// Whether plain steps (one row per sequence) at this concurrency are still unobserved.
    pub fn plain_unseen(&self, sequences: usize) -> bool {
        self.seen.get(sequences).is_some_and(|r| r[sequences.min(r.len() - 1)].is_nan())
    }

    /// Folds one observed draft-step time in.
    pub fn observe_draft(&mut self, ms: f64) {
        if ms.is_finite() && ms > 0.0 {
            self.draft_ms += 0.1 * (ms.min(4.0 * self.draft_ms) - self.draft_ms);
        }
    }

    /// Folds one observed verify-step time into its cell.
    pub fn observe(&mut self, sequences: usize, rows: usize, ms: f64) {
        let prior = self.verify_ms(sequences, rows);
        let Some(cell) = self.seen.get_mut(sequences).and_then(|r| r.get_mut(rows)) else { return };
        if !ms.is_finite() || ms <= 0.0 {
            return;
        }
        let ms = ms.clamp(0.25 * prior, 4.0 * prior);
        *cell = if cell.is_nan() { ms } else { *cell + 0.2 * (ms - *cell) };
    }
}

/// GLM-5.3 EXL3 K4, 1 RTX PRO 6000 (325 W) + 4 Sparks TP4: verify step ms by rows.
pub(crate) const K4_TP4_STEP_MS: [(usize, f64); 15] = [(1, 35.6), (2, 50.5), (3, 62.4), (4, 73.6), (5, 83.2),
    (6, 93.1), (7, 102.2), (8, 111.8), (10, 128.1), (12, 146.7), (16, 173.6), (24, 220.8), (32, 260.2), (48, 342.5),
    (64, 402.8)];

/// Draft counts per sequence maximizing expected committed tokens per
/// millisecond; `confidence[s]` are sequence s's conditional rates (its most
/// drafts), `minimum[s]` drafts it must verify.
/// `base_rows` counts the step's rows outside `confidence` (sequences that
/// do not draft); `cost(rows)` is a cycle's ms.
pub(crate) fn schedule(confidence: &[Vec<f64>], minimum: &[usize], base_rows: usize, cost: impl Fn(usize) -> f64)
    -> (Vec<usize>, f64) {
    let mut lengths = minimum.to_vec();
    let mut rows = base_rows + confidence.len() + minimum.iter().sum::<usize>();
    let mut expected = confidence.len() as f64;
    let mut candidates = Vec::new();
    for (s, rates) in confidence.iter().enumerate() {
        let mut survival = 1.0;
        for (position, &rate) in rates.iter().enumerate() {
            survival *= rate.clamp(0.0, 1.0);
            if position < minimum[s] {
                expected += survival;
            } else if survival > 0.0 {
                candidates.push((survival, position + 1, s));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut best = (lengths.clone(), expected / cost(rows));
    for (survival, length, s) in candidates {
        if length != lengths[s] + 1 {
            continue;
        }
        lengths[s] = length;
        rows += 1;
        expected += survival;
        let rate = expected / cost(rows);
        if rate > best.1 {
            best = (lengths.clone(), rate);
        }
    }
    best
}

/// The draft count of every sequence: cold sequences verify
/// `START_DRAFTS`; a lone warm sequence keeps them unless the schedule is 2%
/// better. `room[s]` caps sequence s.
pub(crate) fn plan(histories: &[&DraftHistory], confidence: &[Vec<f64>], room: &[usize], sequences: usize,
    cost: &StepCost) -> Vec<usize> {
    let base = sequences - histories.len();
    let cost = |rows: usize| cost.ms(sequences, rows);
    let rates: Vec<Vec<f64>> = confidence.iter().zip(room).map(|(c, &r)| c[..c.len().min(r)].to_vec()).collect();
    let minimum: Vec<usize> = histories.iter().zip(&rates)
        .map(|(h, r)| if h.cold() { START_DRAFTS.min(r.len()) } else { 0 }).collect();
    let (lengths, rate) = schedule(&rates, &minimum, base, cost);
    if histories.len() == 1 && !histories[0].cold() {
        let reference = [START_DRAFTS.min(rates[0].len())];
        let (fixed, fixed_rate) = schedule(&rates, &reference, base, cost);
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

    #[test]
    fn schedule_stops_where_throughput_peaks() {
        let cost = StepCost::new(&K4_TP4_STEP_MS, 64);
        let sure = vec![vec![0.99; 7]];
        let c1 = |rows| cost.ms(1, rows);
        assert_eq!(schedule(&sure, &[0], 0, c1).0, vec![7]);
        let unsure = vec![vec![0.2; 7]];
        assert_eq!(schedule(&unsure, &[0], 0, c1).0, vec![0]);
        // The better sequence gets rows first.
        let (lengths, _) = schedule(&[vec![0.95; 7], vec![0.3; 7]], &[0, 0], 0, |rows| cost.ms(2, rows));
        assert!(lengths[0] > lengths[1]);
    }

    #[test]
    fn cold_sequences_verify_five() {
        let cost = StepCost::new(&K4_TP4_STEP_MS, 64);
        let history = DraftHistory::default();
        assert_eq!(plan(&[&history], &[vec![0.1; 7]], &[7], 1, &cost), vec![5]);
        assert_eq!(plan(&[&history], &[vec![0.1; 7]], &[3], 1, &cost), vec![3]);
    }

    #[test]
    fn cheap_observed_plain_steps_stop_drafting() {
        let mut cost = StepCost::new(&K4_TP4_STEP_MS, 64);
        let (warm, rates) = (DraftHistory { outcomes: [(3, 2); 8].into() }, vec![vec![0.7; 7]; 4]);
        let histories = [&warm; 4];
        assert!(plan(&histories, &rates, &[7; 4], 4, &cost).iter().sum::<usize>() > 0);
        assert!(cost.plain_unseen(4));
        cost.observe(4, 4, 40.0);
        assert!(!cost.plain_unseen(4));
        assert_eq!(plan(&histories, &rates, &[7; 4], 4, &cost), vec![0; 4]);
    }
}
