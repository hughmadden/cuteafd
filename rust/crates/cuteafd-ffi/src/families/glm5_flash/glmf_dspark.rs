//! dSpark drafter kernels for GLM 5.3 Flash
//! (`native/families/glm5_flash/cuda/glmf_dspark.cu`). Every pointer is device
//! memory of the documented shape on the stream's device; the stream orders them.
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

type P = *const c_void;
type M = *mut c_void;

fn check(status: i32, what: &str) -> Result<()> {
    ensure!(status == 0, "dSpark {what} failed with {status}");
    Ok(())
}

fn i(value: usize) -> Result<i32> {
    Ok(i32::try_from(value)?)
}

impl NativeLibrary {
    /// Per-head (64-wide) RMSNorm + RoPE of `qkv` rows; k/v rows land at
    /// `slots[row]` (row when null, skipped when negative).
    ///
    /// # Safety
    /// Pointers are live device buffers of those shapes; `positions` I64 [rows], `slots` I32 [rows].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glmf_dspark_qk_rope(&self, qkv: P, q_norm: P, k_norm: P, positions: P, slots: P, q_out: M,
        k_out: M, v_out: M, rows: usize, heads: usize, kv_heads: usize, theta: f32, eps: f32, stream: M)
        -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, M, M, M, i32, i32, i32, f32, f32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glmf_dspark_qk_rope") }?;
        check(unsafe {
            f(qkv, q_norm, k_norm, positions, slots, q_out, k_out, v_out, i(rows)?, i(heads)?, i(kv_heads)?, theta,
                eps, stream)
        }, "qk rope")
    }

    /// Workspace bytes [`Self::glmf_dspark_attention`] needs.
    pub fn glmf_dspark_attention_workspace(&self, sequences: usize, kv_heads: usize, max_keys: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32, i32, i32) -> u64;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glmf_dspark_attention_workspace") }?;
        Ok(usize::try_from(unsafe { f(i(sequences)?, i(kv_heads)?, i(max_keys)?) })?)
    }

    /// Block attention over each sequence's ring context and its block rows
    /// (causal inside the block with `causal`).
    ///
    /// # Safety
    /// As [`Self::glmf_dspark_qk_rope`]; the tables are I32 [sequences].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glmf_dspark_attention(&self, q: P, k_block: P, v_block: P, k_ring: P, v_ring: P, seq_slots: P,
        ctx_lengths: P, ctx_ends: P, out: M, workspace: M, sequences: usize, block_rows: usize, heads: usize,
        kv_heads: usize, ring: usize, max_keys: usize, causal: bool, scale: f32, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, P, P, P, M, M, i32, i32, i32, i32, i32, i32, i32, f32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glmf_dspark_attention") }?;
        check(unsafe {
            f(q, k_block, v_block, k_ring, v_ring, seq_slots, ctx_lengths, ctx_ends, out, workspace, i(sequences)?,
                i(block_rows)?, i(heads)?, i(kv_heads)?, i(ring)?, i(max_keys)?, i32::from(causal), scale, stream)
        }, "attention")
    }

    /// `residual_out = residual + delta`, `normalized = RMSNorm(residual_out) * w`.
    ///
    /// # Safety
    /// As [`Self::glmf_dspark_qk_rope`]; `residual_out` may alias `residual`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glmf_dspark_add_rmsnorm(&self, residual: P, delta: P, w: P, residual_out: M, normalized: M,
        rows: usize, width: usize, eps: f32, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, M, M, i32, i32, f32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glmf_dspark_add_rmsnorm") }?;
        check(unsafe { f(residual, delta, w, residual_out, normalized, i(rows)?, i(width)?, eps, stream) }, "add rmsnorm")
    }

    /// Workspace bytes [`Self::glmf_dspark_markov`] needs.
    pub fn glmf_dspark_markov_workspace(&self, sequences: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32) -> u64;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glmf_dspark_markov_workspace") }?;
        Ok(usize::try_from(unsafe { f(i(sequences)?) })?)
    }

    /// Greedy drafts [sequences, block] U32 from head `logits` [sequences *
    /// block, vocab] FP32 plus the Markov bias (`w1`, `w2` BF16 [vocab, 256]);
    /// step 0 conditions on `anchors` [sequences] U32. At most 32 sequences.
    ///
    /// # Safety
    /// As [`Self::glmf_dspark_qk_rope`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glmf_dspark_markov(&self, logits: P, w1: P, w2: P, anchors: P, drafts: M, workspace: M,
        sequences: usize, block: usize, vocab: usize, rank: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, M, M, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glmf_dspark_markov") }?;
        check(unsafe {
            f(logits, w1, w2, anchors, drafts, workspace, i(sequences)?, i(block)?, i(vocab)?, i(rank)?, stream)
        }, "Markov drafts")
    }

    /// Confidence [sequences * block] F32 of each drafted row.
    ///
    /// # Safety
    /// As [`Self::glmf_dspark_qk_rope`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glmf_dspark_confidence(&self, hidden: P, w1: P, weight: P, bias: P, anchors: P, drafts: P, out: M,
        sequences: usize, block: usize, width: usize, rank: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, P, M, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glmf_dspark_confidence") }?;
        check(unsafe {
            f(hidden, w1, weight, bias, anchors, drafts, out, i(sequences)?, i(block)?, i(width)?, i(rank)?, stream)
        }, "confidence")
    }
}
