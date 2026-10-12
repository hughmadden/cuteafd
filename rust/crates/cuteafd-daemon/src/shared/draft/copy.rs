//! History copies replace only the neural draft's unconfident continuation.

/// Most copy positions beyond the loaded neural window.
pub(crate) const COPY_EXTENSION: usize = 32;
pub(crate) const MIN_DRAFT_AGREEMENT: usize = 3;
pub(crate) const MIN_ANCHORED_MATCH: usize = 7;
/// Protect moderately likely leading tokens; trace calibration motivates 0.6.
pub(crate) const CONFIDENCE_CUT: f64 = 0.6;

pub(crate) fn tail_prior(position: usize) -> f64 {
    [0.95, 0.9, 0.65, 0.3].get(position).copied().unwrap_or(0.2)
}

pub(crate) struct Continuation {
    /// Complete proposal after the anchor, including the agreeing neural prefix.
    pub tokens: Vec<u32>,
    /// Positions still backed by neural evidence; later positions are copy evidence.
    pub neural: usize,
}

pub(crate) fn continuation(history: &[u32], drafted: &[u32], calibrated: &[f64], room: usize,
    greedy: bool) -> Option<Continuation> {
    if !greedy || drafted.len() < MIN_DRAFT_AGREEMENT || calibrated.len() != drafted.len() { return None; }
    let confident = calibrated.iter().take_while(|&&p| p.is_finite() && p >= CONFIDENCE_CUT).count();
    let neural = confident.max(MIN_DRAFT_AGREEMENT);
    let (_, suffix, copy) = window(history, room.min(drafted.len() + COPY_EXTENSION));
    if suffix + neural < MIN_ANCHORED_MATCH || copy.len() <= neural
        || drafted[..neural] != copy[..neural] { return None; }
    Some(Continuation { tokens: copy.to_vec(), neural })
}

/// Trace-only agreement with the same selected history copy, independent of room.
pub(crate) fn agreement(history: &[u32], drafted: &[u32]) -> usize {
    let (_, _, copy) = window(history, drafted.len());
    copy.iter().zip(drafted).take_while(|(copied, neural)| copied == neural).count()
}

pub(crate) fn trace_window(history: &[u32], limit: usize) -> (Option<usize>, usize, Vec<u32>) {
    let (start, suffix, copy) = window(history, limit);
    (start, suffix, copy.to_vec())
}

/// Latest earlier occurrence of the longest matching history suffix (4-8 tokens).
fn window(history: &[u32], limit: usize) -> (Option<usize>, usize, &[u32]) {
    let len = history.len();
    for n in (4..=8).rev() {
        if len <= n { continue; }
        let tail = &history[len - n..];
        if let Some(start) = (0..len - n).rev().find(|&i| &history[i..i + n] == tail) {
            let from = start + n;
            return (Some(start), n, &history[from..from + limit.min(len - from)]);
        }
    }
    (None, 0, &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history(copy: &[u32], suffix: usize) -> Vec<u32> {
        let tail: Vec<_> = (1..=suffix as u32).collect();
        let mut history = tail.clone();
        history.extend(copy);
        history.extend(100..164);
        history.extend(tail);
        history
    }

    #[test]
    fn uniform_agreement_replaces_low_confidence_for_every_width() {
        for width in [3, 5, 7] {
            let neural: Vec<_> = (20..20 + width as u32).collect();
            let mut copied: Vec<_> = (50..50 + width as u32 + 40).collect();
            copied[..3].copy_from_slice(&neural[..3]);
            let mut confidence = vec![0.1; width];
            confidence[..3].fill(0.9);
            let history = history(&copied, 4);
            let result = continuation(&history, &neural, &confidence, 100, true).unwrap();
            assert_eq!(result.neural, 3);
            assert_eq!(result.tokens, copied[..width + COPY_EXTENSION]);
            assert_eq!(agreement(&history, &neural), 3);
            assert_eq!(continuation(&history, &neural, &confidence, 5, true).unwrap().tokens.len(), 5);
            assert!(continuation(&history, &neural, &confidence, 3, true).is_none());
            assert!(continuation(&history, &neural, &confidence, 100, false).is_none());
        }
    }

    #[test]
    fn protects_every_confident_leading_position_and_at_least_three() {
        let neural = [20, 21, 22, 23, 24, 25, 26];
        for protected in 3..=7 {
            let mut copy: Vec<_> = (50..100).collect();
            copy[..protected].copy_from_slice(&neural[..protected]);
            let mut confidence = vec![0.1; 7];
            confidence[..protected].fill(CONFIDENCE_CUT);
            let h = history(&copy, 4);
            assert_eq!(continuation(&h, &neural, &confidence, 39, true).unwrap().neural, protected);
            for mismatch in 0..protected {
                let mut bad = copy.clone();
                bad[mismatch] = 99;
                assert!(continuation(&history(&bad, 4), &neural, &confidence, 39, true).is_none());
            }
        }
        assert!(continuation(&history(&[20, 21, 50, 51], 8), &neural, &[0.1; 7], 39, true).is_none());
        assert!(continuation(&history(&[20, 21, 22, 50], 4), &neural[..2], &[0.9; 2], 39, true).is_none());
    }

    #[test]
    fn trace_identifies_the_copy_and_decay_has_no_flat_tail() {
        let h = history(&[20, 21, 22, 50, 51], 4);
        assert_eq!(trace_window(&h, 5), (Some(0), 4, vec![20, 21, 22, 50, 51]));
        assert_eq!((0..6).map(tail_prior).collect::<Vec<_>>(), [0.95, 0.9, 0.65, 0.3, 0.2, 0.2]);
    }

    #[test]
    fn no_match_short_copy_or_missing_confidence_cannot_qualify() {
        let neural = [20, 21, 22];
        assert!(continuation(&[], &neural, &[0.9; 3], 35, true).is_none());
        assert!(continuation(&history(&neural, 4)[..7], &neural, &[0.9; 3], 35, true).is_none());
        assert!(continuation(&history(&[20, 21, 22, 50], 4), &neural, &[], 35, true).is_none());
        assert_eq!(agreement(&history(&[20, 21, 99, 50], 4), &neural), 2);
        assert_eq!(agreement(&[], &neural), 0);
    }
}
