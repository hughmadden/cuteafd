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

impl Continuation {
    /// Whether the verified proposal reaches the history-backed tail.
    pub(crate) fn uses_copy(&self, verified_drafts: usize) -> bool {
        verified_drafts.min(self.tokens.len()) > self.neural
    }
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
    if len <= 4 { return (None, 0, &[]); }
    let mut best_suffix = 0;
    let mut best_end = 0;
    // An exclusive end in 4..len is exactly start < len - n for any
    // extended match of length n. Overlaps with the tail remain eligible.
    for end in (4..len).rev() {
        if history[end - 1] != history[len - 1]
            || history[end - 4..end - 1] != history[len - 4..len - 1] { continue; }
        let mut n = 4;
        while n < 8 && n < end && history[end - n - 1] == history[len - n - 1] {
            n += 1;
        }
        // Reverse traversal already selected the latest match at equal length.
        if n > best_suffix {
            best_suffix = n;
            best_end = end;
            if n == 8 { break; }
        }
    }
    if best_suffix == 0 { return (None, 0, &[]); }
    (Some(best_end - best_suffix), best_suffix,
        &history[best_end..best_end + limit.min(len - best_end)])
}

#[cfg(test)]
mod tests {
    use super::*;

    // Keep the original five-pass search as an independent selection oracle.
    fn reference_window(history: &[u32], limit: usize) -> (Option<usize>, usize, &[u32]) {
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

    fn assert_window(history: &[u32], start: Option<usize>, suffix: usize) {
        for limit in (0..=history.len() + 1).chain([usize::MAX]) {
            let copy = start.map_or(&[][..], |start| {
                let from = start + suffix;
                &history[from..from + limit.min(history.len() - from)]
            });
            let expected = (start, suffix, copy);
            assert_eq!(reference_window(history, limit), expected);
            assert_eq!(window(history, limit), expected);
        }
    }

    #[test]
    fn window_keeps_original_bounds_and_longest_latest_selection() {
        for len in 0..=4 { assert_window(&vec![7; len], None, 0); }
        assert_window(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], None, 0);
        // The suffix occurs at zero, and the intervening token limits it to four.
        assert_window(&[1, 2, 3, 4, 20, 1, 2, 3, 4], Some(0), 4);
        assert_window(&[9, 1, 2, 3, 4, 10, 1, 2, 3, 4, 20, 1, 2, 3, 4], Some(6), 4);
        // An earlier eight beats a later four; the first eight wins a tie.
        assert_window(&[1, 2, 3, 4, 5, 6, 7, 8, 99, 5, 6, 7, 8, 98,
            1, 2, 3, 4, 5, 6, 7, 8], Some(0), 8);
        assert_window(&[1, 2, 3, 4, 5, 6, 7, 8, 99, 1, 2, 3, 4, 5, 6, 7, 8, 98,
            1, 2, 3, 4, 5, 6, 7, 8], Some(9), 8);
        // Matches may end one token before the tail ends, even at minimum length.
        assert_window(&[7; 5], Some(0), 4);
        assert_window(&[7; 8], Some(0), 7);
        assert_window(&[7; 12], Some(3), 8);
        assert_window(&[1, 2, 1, 2, 1, 2, 1, 2, 1, 2], Some(0), 8);
    }

    fn next_random(state: &mut u64) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 32) as u32
    }

    #[test]
    fn window_matches_reference_on_dense_random_histories_at_every_limit() {
        let mut state = 0x9e37_79b9_7f4a_7c15;
        for alphabet in 2..=6 {
            for len in (0..=128).chain([256, 512, 1024, 2048, 3072, 4096]) {
                let history: Vec<_> = (0..len).map(|_| next_random(&mut state) % alphabet).collect();
                // All larger limits are equivalent to len; include saturation too.
                for limit in (0..=len + 1).chain([usize::MAX]) {
                    assert_eq!(window(&history, limit), reference_window(&history, limit),
                        "alphabet={alphabet}, len={len}, limit={limit}");
                }
            }
        }
    }

    #[test]
    #[ignore = "CPU history-copy search microbenchmark; run optimized with --ignored --nocapture"]
    fn window_search_microseconds() {
        use std::{hint::black_box, time::Instant};
        let iterations = 500;
        let mut state = 0xd1b5_4a32_d192_ed03;
        for len in [32 * 1024, 128 * 1024, 200 * 1024] {
            let mut history: Vec<_> = (0..len).map(|_| next_random(&mut state) % 65_536).collect();
            // Reserve a token for the tail so the no-match fixture is guaranteed.
            history[len - 1] = u32::MAX;
            for early_match in [false, true] {
                if early_match {
                    let tail: [u32; 8] = history[len - 8..].try_into().unwrap();
                    history[16..24].copy_from_slice(&tail);
                }
                let expected = if early_match { (Some(16), 8) } else { (None, 0) };
                let actual = window(&history, COPY_EXTENSION);
                assert_eq!((actual.0, actual.1), expected);
                assert_eq!(actual, reference_window(&history, COPY_EXTENSION));
                // Warm both searches before timing matched inputs and outputs.
                for _ in 0..10 {
                    black_box(reference_window(black_box(&history), black_box(COPY_EXTENSION)));
                    black_box(window(black_box(&history), black_box(COPY_EXTENSION)));
                }
                let started = Instant::now();
                for _ in 0..iterations {
                    black_box(reference_window(black_box(&history), black_box(COPY_EXTENSION)));
                }
                let before = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;
                let started = Instant::now();
                for _ in 0..iterations {
                    black_box(window(black_box(&history), black_box(COPY_EXTENSION)));
                }
                let after = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;
                println!("history_copy {len} tokens / {}: {before:.2} us -> {after:.2} us ({iterations} iterations)",
                    if early_match { "early eight" } else { "no match" });
            }
        }
    }

    #[test]
    fn verified_neural_prefix_does_not_count_as_a_copy_round() {
        let copy = Continuation { tokens: vec![20, 21, 22, 50, 51], neural: 3 };
        for verified in 0..=3 {
            assert!(!copy.uses_copy(verified));
        }
        for verified in [4, 5, 64] {
            assert!(copy.uses_copy(verified));
        }
        let neural_only = Continuation { tokens: vec![20, 21, 22], neural: 3 };
        assert!(!neural_only.uses_copy(64));
    }

    fn history(copy: &[u32], suffix: usize) -> Vec<u32> {
        let tail: Vec<_> = (1..=suffix as u32).collect();
        let mut history = tail.clone();
        history.extend(copy);
        history.extend(100..164);
        history.extend(tail);
        history
    }

    #[test]
    fn trace_window_matches_runtime_copy_without_changing_selection() {
        let history = [1, 2, 3, 4, 10, 11, 12, 13, 1, 2, 3, 4];
        assert_eq!(trace_window(&history, 3), (Some(0), 4, vec![10, 11, 12]));
        assert_eq!(trace_window(&history, 20).2, window(&history, 20).2);
        assert_eq!(trace_window(&[], 3), (None, 0, Vec::new()));
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
