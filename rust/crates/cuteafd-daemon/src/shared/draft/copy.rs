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
