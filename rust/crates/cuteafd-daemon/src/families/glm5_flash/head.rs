//! The GLM 5.3 Flash vocabulary head: one resident representation, shared by
//! the target's logits and the DFlash drafter.
//!
//! `Bf16`: the checkpoint's `lm_head.weight`. `Fp8` (`--fp8-head`): E4M3 with
//! FP32 scales per output row and 128-wide K block (`[vocab, hidden/128]`),
//! quantized at load; no BF16 copy is kept, and every row count (decode,
//! verify, prefill tails, `--nll` full prefill, draft blocks) runs the
//! `glmf_head_fp8` program in [`FP8_HEAD_ROWS`]-row spans.
//!
//! The target's logits through the BF16 head ([`TargetHeadMode`], `--target-head`): `exact` runs
//! the pedantic FP32 cuBLAS GEMM (CUDA cores) at every row count; `tensor` keeps steps of up to
//! [`TARGET_HEAD_TENSOR_ROWS`]` - 1` rows on it and runs wider ones as a BF16 tensor-core GEMM with
//! FP32 accumulation, which reads the head once for every row count.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Result};
use cuteafd_ffi::programs::{Programs, Scalar};
use std::ffi::c_void;

/// Rows one `glmf_head_fp8` launch takes (its GEMV's M tile).
pub(crate) const FP8_HEAD_ROWS: usize = 16;

/// How the target's logits run through the BF16 head (`--target-head`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum TargetHeadMode {
    /// The pedantic FP32 cuBLAS GEMM (FP32 products and sums on CUDA cores) at every row count.
    #[default]
    Exact,
    /// Up to 8 rows as `exact` (a one-sequence step: one token and up to seven drafts); from
    /// [`TARGET_HEAD_TENSOR_ROWS`] rows a BF16 tensor-core GEMM with FP32 accumulation and logits.
    Tensor,
}

/// Rows from which [`TargetHeadMode::Tensor`] takes tensor cores: one sequence's decode or verify
/// step (its token and a DFlash2 block's seven drafts) keeps its bits.
pub(crate) const TARGET_HEAD_TENSOR_ROWS: usize = 9;

impl TargetHeadMode {
    /// Whether `rows` logit rows run on tensor cores.
    pub fn tensor_cores(self, rows: usize) -> bool {
        self == TargetHeadMode::Tensor && rows >= TARGET_HEAD_TENSOR_ROWS
    }
}

/// `(first row, rows)` spans of at most [`FP8_HEAD_ROWS`] covering `0..rows`.
pub(crate) fn fp8_head_spans(rows: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..rows).step_by(FP8_HEAD_ROWS).map(move |first| (first, FP8_HEAD_ROWS.min(rows - first)))
}

pub(crate) enum GlmfHead<'a> {
    Bf16(DeviceAllocation<'a>),
    Fp8 { values: DeviceAllocation<'a>, scales: DeviceAllocation<'a> },
}

impl<'a> GlmfHead<'a> {
    pub fn allocations(&self) -> Vec<&DeviceAllocation<'a>> {
        match self {
            Self::Bf16(weight) => vec![weight],
            Self::Fp8 { values, scales } => vec![values, scales],
        }
    }

    pub fn bytes(&self) -> usize {
        self.allocations().iter().map(|a| a.buffer.bytes).sum()
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Bf16(_) => "BF16",
            Self::Fp8 { .. } => "FP8",
        }
    }

    /// The checkpoint BF16 head, when that is the resident representation.
    pub fn bf16(&self) -> Option<&DeviceAllocation<'a>> {
        match self {
            Self::Bf16(weight) => Some(weight),
            Self::Fp8 { .. } => None,
        }
    }

    /// FP32 logits `[rows, vocab]` of BF16 rows `x [rows, hidden]` through the
    /// FP8 head, in 16-row spans of `glmf_head_fp8`. Errors on a BF16 head.
    ///
    /// # Safety
    /// `x` and `logits` must hold `rows` rows of `hidden` BF16 and `vocab` FP32
    /// values, live until `stream` reaches these launches.
    pub unsafe fn launch_fp8(&self, programs: &Programs<'_>, x: *const c_void, logits: *mut f32, rows: usize,
        hidden: usize, vocab: usize, stream: *mut c_void) -> Result<()> {
        let Self::Fp8 { values, scales } = self else {
            anyhow::bail!("the GLMF head is BF16; the FP8 head program has no weight");
        };
        ensure!(values.buffer.bytes == vocab * hidden && scales.buffer.bytes >= vocab * (hidden / 128) * 4,
            "FP8 head operands do not match [{vocab}, {hidden}]");
        let program = programs.program("glmf_head_fp8", &["x", "w_fp8", "scale", "logits"])?;
        for (first, n) in fp8_head_spans(rows) {
            let x = x.cast::<u8>().wrapping_add(first * hidden * 2).cast_mut().cast();
            let out = logits.cast::<u8>().wrapping_add(first * vocab * 4).cast();
            // SAFETY: rows first..first + n of the caller's input and logits (its
            // contract) and this head's live operands, ordered on `stream`.
            unsafe { program.launch(&[x, values.buffer.ptr, scales.buffer.ptr, out], &[Scalar::I32(n as i32)],
                stream)? };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `exact` never takes tensor cores; `tensor` takes them from 9 rows, so steps of one
    /// sequence (up to 8 rows) keep the pedantic GEMM's bits.
    #[test]
    fn the_target_head_takes_tensor_cores_from_nine_rows_only_when_asked() {
        assert_eq!(TargetHeadMode::default(), TargetHeadMode::Exact);
        for rows in [1, 8, 9, 64, 127, 128, 4096] {
            assert!(!TargetHeadMode::Exact.tensor_cores(rows), "{rows}");
        }
        for rows in 1..=8 {
            assert!(!TargetHeadMode::Tensor.tensor_cores(rows), "{rows}");
        }
        for rows in [9, 16, 56, 64, 127, 128, 4096] {
            assert!(TargetHeadMode::Tensor.tensor_cores(rows), "{rows}");
        }
    }

    #[test]
    fn spans_cover_every_row_count_without_a_fallback() {
        for rows in 1..=4096 {
            let mut next = 0;
            for (first, n) in fp8_head_spans(rows) {
                assert_eq!(first, next);
                assert!((1..=FP8_HEAD_ROWS).contains(&n));
                next += n;
            }
            assert_eq!(next, rows);
        }
        assert_eq!(fp8_head_spans(0).count(), 0);
    }
}
