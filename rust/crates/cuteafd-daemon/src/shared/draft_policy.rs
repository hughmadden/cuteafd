//! Shared request acceptance history.
//! Resource prices, calibration and allocation live in cuteafd-core::draft_policy.
use std::collections::VecDeque;

const HISTORY: usize = 16;
const COLD_START_CYCLES: usize = 4;
const PRIOR_SUCCESSES: usize = 3;
const PRIOR_TRIALS: usize = 4;

/// A sequence's recent draft outcomes.
#[derive(Debug, Clone, Default)]
pub(crate) struct DraftHistory {
    outcomes: VecDeque<(usize, usize)>,
}

impl DraftHistory {
    /// Fewer than four outcomes observed.
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

    /// (successes, trials) of `position` (censored after a miss).
    fn counts(&self, position: usize) -> (usize, usize) {
        let (mut successes, mut trials) = (0, 0);
        for &(proposed, accepted) in &self.outcomes {
            if proposed >= position && accepted + 1 >= position {
                trials += 1;
                successes += usize::from(accepted >= position);
            }
        }
        (successes, trials)
    }

    /// Conditional acceptance of positions 1..=max with a prior of 3
    /// successes in 4 trials per position.
    pub fn conditional(&self, max: usize) -> Vec<f64> {
        (1..=max).map(|position| {
            let (successes, trials) = self.counts(position);
            (PRIOR_SUCCESSES + successes) as f64 / (PRIOR_TRIALS + trials) as f64
        }).collect()
    }

    /// Conditional acceptance of positions 1..=max under a prior pooled
    /// from `others` (the other sequences in the step): their outcomes at
    /// each position on top of 3 in 4, weighing at most `POOL_TRIALS`
    /// trials. Alone, this is [`Self::conditional`].
    pub fn pooled<'h>(&self, max: usize, others: impl Iterator<Item = &'h DraftHistory> + Clone) -> Vec<f64> {
        (1..=max).map(|position| {
            let (pooled, trials) = others.clone().map(|h| h.counts(position))
                .fold((PRIOR_SUCCESSES, PRIOR_TRIALS), |(s, t), (hs, ht)| (s + hs, t + ht));
            let weight = (trials as f64).min(POOL_TRIALS);
            let mean = pooled as f64 / trials as f64;
            let (successes, own) = self.counts(position);
            (weight * mean + successes as f64) / (weight + own as f64)
        }).collect()
    }
}

/// Most prior weight, in trials, the other sequences' outcomes get.
///
/// Served traces (Qwen 3.8 MTP, four sequences) show one sequence's
/// 16-outcome history is a noisy estimate: positions it rated 0.4-0.5 were
/// accepted 60-70% of the time and 0.8-0.9 ones 75-80%, so plans cut drafts
/// that would have been kept. Borrowing the other sequences' outcomes shrinks
/// both tails; a lone sequence keeps its own history (pooling a sequence
/// with its own past made single-sequence plans slower to follow content).
/// Half the history's weight keeps a sequence unlike the others its own.
const POOL_TRIALS: f64 = 8.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_censors_positions_after_a_miss() {
        let mut history = DraftHistory::default();
        history.observe(5, 2);
        // Positions 1, 2 accepted, 3 rejected, 4 and 5 unobserved.
        assert_eq!(history.conditional(5), vec![4.0 / 5.0, 4.0 / 5.0, 3.0 / 5.0, 0.75, 0.75]);
    }

    #[test]
    fn pooled_rates_borrow_other_sequences() {
        let mut me = DraftHistory::default();
        for _ in 0..4 {
            me.observe(3, 0);
        }
        // Alone: the 3-in-4 prior.
        assert_eq!(me.pooled(2, std::iter::empty()), me.conditional(2));
        let mut other = DraftHistory::default();
        for _ in 0..16 {
            other.observe(3, 3);
        }
        // Two others at 16/16 on position 1: prior (3 + 32) / (4 + 32) at 8 trials, then 0 of 4 own.
        let rates = me.pooled(2, [&other, &other].into_iter());
        let prior = 35.0 / 36.0;
        assert!((rates[0] - 8.0 * prior / 12.0).abs() < 1e-12, "{rates:?}");
        // Position 2 is unobserved for me: the prior itself.
        assert!((rates[1] - prior).abs() < 1e-12, "{rates:?}");
    }


}

/// Candidate policy switches deliberately stay off until throughput gates pass.
pub(crate) fn enabled(variable: &str) -> bool {
    std::env::var(variable).is_ok_and(|v| matches!(v.as_str(), "1" | "on" | "true"))
}
