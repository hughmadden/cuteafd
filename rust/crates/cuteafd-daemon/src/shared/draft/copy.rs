//! Copies extend an agreeing neural draft; they never replace it.

/// Most copy positions appended past the drafter's whole window.
pub(crate) const COPY_EXTENSION: usize = 32;

/// The continuation of a copy that repeats the neural drafter's whole window.
/// Partial agreement, empty drafts, and copies with no continuation do not qualify.
pub(crate) fn copy_extension<'c>(drafted: &[u32], copy: &'c [u32]) -> &'c [u32] {
    if drafted.is_empty() || copy.len() <= drafted.len() || copy[..drafted.len()] != *drafted {
        return &[];
    }
    &copy[drafted.len()..]
}

/// An agreed continuation inside this greedy request's output/context room.
pub(crate) fn extension(history: &[u32], drafted: &[u32], room: usize, greedy: bool) -> Vec<u32> {
    let extra = COPY_EXTENSION.min(room.saturating_sub(drafted.len()));
    if !greedy || drafted.is_empty() || extra == 0 { return Vec::new(); }
    let copy = window(history, drafted.len() + extra);
    copy_extension(drafted, &copy).to_vec()
}

/// Neural-prefix agreement with the same history copy considered for extension.
/// Trace-only: independent of output room, greediness, and policy selection.
pub(crate) fn agreement(history: &[u32], drafted: &[u32]) -> usize {
    if drafted.is_empty() { return 0; }
    let copy = window(history, drafted.len());
    copy.iter().zip(drafted).take_while(|(copied, neural)| copied == neural).count()
}

/// Latest earlier occurrence of the longest matching history suffix (4-8 tokens).
fn window(history: &[u32], limit: usize) -> Vec<u32> {
    let len = history.len();
    for n in (4..=8).rev() {
        if len <= n { continue; }
        let tail = &history[len - n..];
        if let Some(start) = (0..len - n).rev().find(|&i| &history[i..i + n] == tail) {
            let from = start + n;
            return history[from..(from + limit).min(len)].to_vec();
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_extend_only_whole_window_agreement() {
        assert_eq!(copy_extension(&[1, 2, 3], &[1, 2, 3, 4, 5]), [4, 5]);
        assert!(copy_extension(&[1, 2, 3], &[1, 2, 9, 4, 5]).is_empty());
        assert!(copy_extension(&[1, 2, 3], &[1, 2, 3]).is_empty());
        assert!(copy_extension(&[1, 2, 3], &[1, 2]).is_empty());
        assert!(copy_extension(&[], &[1, 2]).is_empty());
    }

    #[test]
    fn extension_obeys_room_greedy_and_every_familys_window() {
        for width in [1usize, 2, 3, 7, 8, 16] {
            let neural: Vec<u32> = (20..20 + width as u32).collect();
            let suffix: Vec<u32> = (1..=8).collect();
            let mut history = suffix.clone();
            history.extend(&neural);
            history.extend(100..164);
            history.extend(&suffix);
            assert_eq!(extension(&history, &neural, width + 100, true).len(), COPY_EXTENSION);
            for room in 0..=width + COPY_EXTENSION {
                assert_eq!(extension(&history, &neural, room, true).len(), room.saturating_sub(width));
            }
            assert!(extension(&history, &neural, width + 10, false).is_empty());
            assert!(extension(&history, &[], width + 10, true).is_empty());
        }
    }

    #[test]
    fn agreement_reports_partial_neural_prefixes() {
        let neural = [20, 21, 22, 23, 24, 25, 26];
        let suffix: Vec<u32> = (1..=8).collect();
        for matched in 0..=neural.len() {
            let mut copy = neural.to_vec();
            if matched < copy.len() { copy[matched] = 99; }
            let mut history = suffix.clone();
            history.extend(copy);
            history.extend(100..164);
            history.extend(&suffix);
            assert_eq!(agreement(&history, &neural), matched);
        }
        assert_eq!(agreement(&[], &neural), 0);
        assert_eq!(agreement(&suffix, &neural), 0);
        assert_eq!(agreement(&suffix, &[]), 0);
        let mut short = suffix.clone();
        short.extend(&neural[..3]);
        short.extend(&suffix);
        assert_eq!(agreement(&short, &neural), 3);
    }

    #[test]
    fn every_neural_window_requires_full_agreement() {
        // Qwen/MiMo MTP chains, DFlash2, and GLM/V4.1 dSpark windows.
        for width in [1, 2, 3, 7, 8, 16] {
            let neural: Vec<_> = (0..width).collect();
            let copy: Vec<_> = (0..width + COPY_EXTENSION as u32).collect();
            assert_eq!(copy_extension(&neural, &copy).len(), COPY_EXTENSION);
            for mismatch in 0..width as usize {
                let mut bad = copy.clone();
                bad[mismatch] = u32::MAX;
                assert!(copy_extension(&neural, &bad).is_empty(), "width {width}, mismatch {mismatch}");
            }
        }
    }
}
