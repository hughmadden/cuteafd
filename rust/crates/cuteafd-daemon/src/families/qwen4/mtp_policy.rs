//! Conditional native-MTP priors; the shared DraftPolicy owns calibration and allocation.
use crate::shared::draft_policy::DraftHistory;

pub(crate) fn acceptance(histories: &[&DraftHistory], limits: &[usize]) -> Vec<Vec<f64>> {
    histories.iter().zip(limits).enumerate().map(|(i, (history, &limit))| {
        history.pooled(limit, histories.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, h)| *h))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn acceptance_preserves_request_history_and_position_limits() {
        let (mut good, mut bad) = (DraftHistory::default(), DraftHistory::default());
        for _ in 0..16 { good.observe(4, 4); bad.observe(4, 0); }
        let rates = acceptance(&[&good, &bad], &[2, 7]);
        assert_eq!((rates[0].len(), rates[1].len()), (2, 7));
        assert!(rates[0][0] > rates[1][0]);
        assert_eq!(rates[0], good.pooled(2, [&bad].into_iter()));
    }
}
