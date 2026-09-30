//! How many MTP drafts each Qwen 3.8 Flash Next sequence verifies.
//!
//! MTP drafts are sequential: a cycle runs one MTP step over every pending
//! row (the canonical history, plus each drafting sequence's first draft)
//! and one chain step per further draft, then the target verifies every
//! sequence's next token plus its drafts in one step. Each sequence keeps its
//! last 16 outcomes (glm::dflash_policy's `DraftHistory`: conditional
//! acceptance per draft position, a 3-in-4 prior); the plan admits draft
//! positions across sequences in order of survival probability (the product
//! of the conditional rates) while the expected committed tokens per
//! millisecond improve, pricing a cycle as the verify step of its rows plus
//! its MTP steps. Measured costs fold in as the server runs.
use crate::glm::dflash_policy::DraftHistory;

/// Qwen 3.8 Flash Next EXL3 K4.25, experts local on one RTX PRO 6000 (325 W):
/// speculative verify step ms by rows (one sequence after the 1634-token golden
/// prompt, qwen4-golden --spec-decode), and an MTP step with its E4M3 head and
/// the host round trip (0.71 ms alone, 0.93 ms in a cycle).
pub(crate) const RTX_TP1_VERIFY_MS: [(usize, f64); 13] = [(1, 9.0), (2, 9.7), (3, 10.9), (4, 11.4), (5, 12.3),
    (6, 13.2), (8, 14.5), (12, 17.4), (16, 19.7), (24, 27.8), (32, 32.2), (48, 46.7), (64, 56.3)];
pub(crate) const RTX_TP1_MTP_STEP_MS: f64 = 0.93;

/// Cycle cost: the verify step by rows, scaled by what serving observes, and
/// the MTP steps.
#[derive(Debug, Clone)]
pub(crate) struct MtpCost {
    verify: Vec<f64>,
    ratio: f64,
    step_ms: f64,
}

impl MtpCost {
    /// `points` are (rows, ms) with increasing rows, the first at one row.
    pub fn new(points: &[(usize, f64)], step_ms: f64, max_rows: usize) -> Self {
        let verify = (0..=max_rows).map(|rows| {
            let rows = rows.max(1);
            let upper = points.iter().position(|&(r, _)| r >= rows).unwrap_or(points.len() - 1).max(1);
            let ((r0, m0), (r1, m1)) = (points[upper - 1], points[upper]);
            m0 + (m1 - m0) * (rows as f64 - r0 as f64) / (r1 as f64 - r0 as f64)
        }).collect();
        Self { verify, ratio: 1.0, step_ms }
    }

    /// Cycle ms of a verify step of `rows` rows after `steps` MTP steps.
    pub fn ms(&self, rows: usize, steps: usize) -> f64 {
        self.ratio * self.verify[rows.min(self.verify.len() - 1)] + self.step_ms * steps as f64
    }

    pub fn observe_verify(&mut self, rows: usize, ms: f64) {
        if ms.is_finite() && ms > 0.0 {
            let ratio = (ms / self.verify[rows.min(self.verify.len() - 1)]).clamp(0.5, 2.0);
            self.ratio += 0.1 * (ratio - self.ratio);
        }
    }

    pub fn observe_steps(&mut self, steps: usize, ms: f64) {
        if steps > 0 && ms.is_finite() && ms > 0.0 {
            let per = (ms / steps as f64).min(4.0 * self.step_ms);
            self.step_ms += 0.1 * (per - self.step_ms);
        }
    }
}

/// Draft depths per sequence (`limits`: most drafts each may verify)
/// maximizing expected committed tokens per millisecond; `fixed` verifies
/// that many where allowed.
pub(crate) fn plan(histories: &[&DraftHistory], limits: &[usize], fixed: Option<usize>, cost: &MtpCost) -> Vec<usize> {
    if let Some(fixed) = fixed {
        return limits.iter().map(|&l| fixed.min(l)).collect();
    }
    let mut depths = vec![0usize; histories.len()];
    let mut candidates = Vec::new();
    for (i, (history, &limit)) in histories.iter().zip(limits).enumerate() {
        let mut survival = 1.0;
        for (position, rate) in history.conditional(limit).into_iter().enumerate() {
            survival *= rate.clamp(0.0, 1.0);
            candidates.push((survival, position + 1, i));
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut rows = histories.len();
    let mut expected = histories.len() as f64;
    let mut best = (depths.clone(), expected / cost.ms(rows, 0));
    for (survival, depth, i) in candidates {
        if depth != depths[i] + 1 {
            continue;
        }
        depths[i] = depth;
        rows += 1;
        expected += survival;
        let steps = depths.iter().copied().max().unwrap_or(0);
        let rate = expected / cost.ms(rows, steps);
        if rate > best.1 {
            best = (depths.clone(), rate);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_follows_acceptance() {
        let cost = MtpCost::new(&RTX_TP1_VERIFY_MS, RTX_TP1_MTP_STEP_MS, 64);
        let mut good = DraftHistory::default();
        let mut bad = DraftHistory::default();
        for _ in 0..16 {
            good.observe(4, 4);
            bad.observe(4, 0);
        }
        let deep = plan(&[&good], &[7], None, &cost)[0];
        assert!(deep >= 4, "{deep}");
        assert_eq!(plan(&[&bad], &[7], None, &cost), vec![0]);
        let both = plan(&[&good, &bad], &[7, 7], None, &cost);
        assert!(both[0] > both[1]);
        assert_eq!(plan(&[&good, &bad], &[2, 7], Some(3), &cost), vec![2, 3]);
    }
}
