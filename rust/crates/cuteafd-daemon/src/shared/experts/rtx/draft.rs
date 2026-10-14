//! Capture-compatible top-3 native draft experts and ordered rank-pair reduction.
//! Unlike backbone owner reduction, each route sums ranks in FP32 and rounds to
//! BF16 before the top-3 sum and optional shared contribution.
pub(crate) mod pair;
pub(crate) mod rank;
