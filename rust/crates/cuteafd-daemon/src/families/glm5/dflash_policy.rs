//! GLM confidence priors retained by the shared resource binding.
pub(crate) use crate::shared::draft_policy::{DraftHistory, DraftSkip};
pub(crate) const START_DRAFTS: usize = 5;

#[cfg(test)]
fn calibrated_confidence(history: &[f64], features: &[[f32; 4]]) -> Option<Vec<f64>> {
    crate::shared::draft_confidence::SelectorFit::default().confidence(history, features)
}

/// Weight of a dSpark confidence head's prediction against the sequence's
/// history rate, in logit space (`CUTEAFD_DSPARK_HEAD_WEIGHT`, default 0.75).
fn head_weight() -> f64 {
    static WEIGHT: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *WEIGHT.get_or_init(|| std::env::var("CUTEAFD_DSPARK_HEAD_WEIGHT").ok().and_then(|v| v.parse().ok())
        .filter(|w: &f64| (0.0..=1.0).contains(w)).unwrap_or(0.75))
}

/// Conditional acceptance per position from a dSpark confidence head (the
/// predicted acceptance of each drafted token given the ones before it),
/// blended in logit space with the history rate.
pub(crate) fn head_confidence(history: &DraftHistory, head: &[f32]) -> Vec<f64> {
    let rates = history.conditional(head.len());
    let logit = |p: f64| {
        let p = p.clamp(1e-4, 1.0 - 1e-4);
        (p / (1.0 - p)).ln()
    };
    let w = head_weight();
    head.iter().zip(rates).map(|(&c, rate)| {
        let c = f64::from(c);
        if !c.is_finite() {
            return rate;
        }
        let z = w * logit(c) + (1.0 - w) * logit(rate);
        1.0 / (1.0 + (-z).exp())
    }).collect()
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

}
