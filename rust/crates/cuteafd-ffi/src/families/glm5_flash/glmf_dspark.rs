//! dSpark drafter kernels for GLM 5.3 Flash
//! (`native/families/glm5_flash/cuda/glmf_dspark.cu`). Every pointer is device
//! memory of the documented shape on the stream's device; the stream orders them.
//! With `fp8_kv`, retained K/V planes use `align16(kv_heads * (head_dim + 4))`
//! bytes per row: E4M3 codes, FP32 head descales, zero alignment padding.
//! Queries and speculative block K/V remain BF16 in either mode.
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
    /// Shared block dSpark per-head norm and checkpoint RoPE. For 128-wide
    /// heads, `inv_freq` is immutable FP32 [64] on device and `factor` scales
    /// cos/sin before BF16 rounding. The 64-wide default route is unchanged.
    ///
    /// # Safety
    /// Device buffers follow [`Self::glmf_dspark_qk_rope`] with `head_dim`
    /// replacing 64; all remain live until `stream` finishes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn block_dspark_qk_rope(&self, qkv: P, q_norm: P, k_norm: P, positions: P, slots: P,
        inv_freq: P, q_out: M, k_out: M, v_out: M, rows: usize, heads: usize, kv_heads: usize,
        head_dim: usize, theta: f32, factor: f32, eps: f32, fp8_kv: bool, stream: M) -> Result<()> {
        ensure!(rows > 0 && kv_heads > 0, "invalid block dSpark qk geometry");
        ensure!((head_dim == 64 && inv_freq.is_null() && factor == 1.0)
            || (head_dim == 128 && !inv_freq.is_null()), "unsupported block dSpark RoPE");
        type F = unsafe extern "C" fn(P, P, P, P, P, P, M, M, M, i32, i32, i32, i32, f32, f32, f32, M) -> i32;
        // SAFETY: the symbol has the native signature above.
        let symbol: &[u8] = if fp8_kv { b"cuteafd_block_dspark_qk_rope_fp8" } else { b"cuteafd_block_dspark_qk_rope" };
        // SAFETY: both format-specific symbols have the native signature above.
        let f = *unsafe { self.lib.get::<F>(symbol) }?;
        // SAFETY: caller supplies the live device buffers described above.
        // SAFETY: the caller supplies live buffers in the selected record format.
        check(unsafe { f(qkv, q_norm, k_norm, positions, slots, inv_freq, q_out, k_out, v_out,
            i(rows)?, i(heads)?, i(kv_heads)?, i(head_dim)?, theta, factor, eps, stream) }, "block qk rope")
    }

    /// Shared block attention workspace, including all query slices for GQA.
    pub fn block_dspark_attention_workspace(&self, sequences: usize, heads: usize, kv_heads: usize,
        block: usize, max_keys: usize, head_dim: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32) -> u64;
        // SAFETY: the symbol has the native signature above.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_block_dspark_attention_workspace") }?;
        // SAFETY: this metadata-only call takes no pointers.
        let bytes = unsafe { f(i(sequences)?, i(heads)?, i(kv_heads)?, i(block)?, i(max_keys)?, i(head_dim)?) };
        ensure!(bytes > 0, "invalid block dSpark attention geometry");
        Ok(usize::try_from(bytes)?)
    }

    /// Shared attention over context capacity and the noise block. Full-context
    /// callers admit the complete position extent before allocation; no ring
    /// truncation is implied by this API.
    ///
    /// # Safety
    /// As [`Self::glmf_dspark_attention`], with `head_dim` replacing 64 and
    /// workspace sized by [`Self::block_dspark_attention_workspace`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn block_dspark_attention(&self, q: P, k_block: P, v_block: P, k_context: P, v_context: P,
        slots: P, lengths: P, ends: P, out: M, workspace: M, sequences: usize, block: usize, heads: usize,
        kv_heads: usize, capacity: usize, max_keys: usize, causal: bool, head_dim: usize, scale: f32,
        fp8_kv: bool, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, P, P, P, M, M,
            i32, i32, i32, i32, i32, i32, i32, i32, f32, M) -> i32;
        // SAFETY: the symbol has the native signature above.
        let symbol: &[u8] = if fp8_kv { b"cuteafd_block_dspark_attention_fp8" } else { b"cuteafd_block_dspark_attention" };
        // SAFETY: both format-specific symbols have the native signature above.
        let f = *unsafe { self.lib.get::<F>(symbol) }?;
        // SAFETY: caller supplies live and correctly sized device buffers.
        // SAFETY: the caller supplies live buffers in the selected record format.
        check(unsafe { f(q, k_block, v_block, k_context, v_context, slots, lengths, ends, out, workspace,
            i(sequences)?, i(block)?, i(heads)?, i(kv_heads)?, i(capacity)?, i(max_keys)?,
            i32::from(causal), i(head_dim)?, scale, stream) }, "block attention")
    }

    /// Paged full-context attention. I32 page tables are [sequences, table_width],
    /// with 64 rows per page. Ends are relative to each binding's storage origin.
    ///
    /// # Safety
    /// Buffers satisfy [`Self::block_dspark_attention`]; every visible page ID
    /// addresses a live, initialized segment in both retained K/V planes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn block_dspark_attention_paged(&self, q: P, k_block: P, v_block: P,
        k_context: P, v_context: P, slots: P, lengths: P, ends: P, pages: P,
        out: M, workspace: M, sequences: usize, block: usize, heads: usize,
        kv_heads: usize, capacity: usize, max_keys: usize, causal: bool,
        head_dim: usize, table_width: usize, scale: f32, fp8_kv: bool, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, P, P, P, P, M, M,
            i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, M) -> i32;
        let symbol: &[u8] = if fp8_kv { b"cuteafd_block_dspark_attention_paged_fp8" }
            else { b"cuteafd_block_dspark_attention_paged" };
        // SAFETY: both format-specific symbols have the native signature above.
        let f = *unsafe { self.lib.get::<F>(symbol) }?;
        // SAFETY: caller retains all visible pages and workspaces on the owned stream.
        check(unsafe { f(q, k_block, v_block, k_context, v_context, slots, lengths,
            ends, pages, out, workspace, i(sequences)?, i(block)?, i(heads)?,
            i(kv_heads)?, i(capacity)?, i(max_keys)?, i32::from(causal), i(head_dim)?,
            i(table_width)?, scale, stream) }, "paged block attention")
    }

    /// Per-head (64-wide) RMSNorm + RoPE of `qkv` rows; k/v rows land at
    /// `slots[row]` (row when null, skipped when negative).
    ///
    /// # Safety
    /// Pointers are live device buffers of those shapes; `positions` I64 [rows], `slots` I32 [rows].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glmf_dspark_qk_rope(&self, qkv: P, q_norm: P, k_norm: P, positions: P, slots: P, q_out: M,
        k_out: M, v_out: M, rows: usize, heads: usize, kv_heads: usize, theta: f32, eps: f32, fp8_kv: bool, stream: M)
        -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, M, M, M, i32, i32, i32, f32, f32, M) -> i32;
        let symbol: &[u8] = if fp8_kv { b"cuteafd_glmf_dspark_qk_rope_fp8" } else { b"cuteafd_glmf_dspark_qk_rope" };
        // SAFETY: both format-specific symbols have the native signature above.
        let f = *unsafe { self.lib.get::<F>(symbol) }?;
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
        kv_heads: usize, ring: usize, max_keys: usize, causal: bool, scale: f32, fp8_kv: bool, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, P, P, P, M, M, i32, i32, i32, i32, i32, i32, i32, f32, M) -> i32;
        let symbol: &[u8] = if fp8_kv { b"cuteafd_glmf_dspark_attention_fp8" } else { b"cuteafd_glmf_dspark_attention" };
        // SAFETY: both format-specific symbols have the native signature above.
        let f = *unsafe { self.lib.get::<F>(symbol) }?;
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
    pub fn glmf_dspark_markov_workspace(&self, sequences: usize, block: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32, i32) -> u64;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glmf_dspark_markov_workspace") }?;
        Ok(usize::try_from(unsafe { f(i(sequences)?, i(block)?) })?)
    }

    /// `norms` [vocab] F32 = row norms of the Markov projection `w2` (BF16 [vocab, rank]).
    ///
    /// # Safety
    /// As [`Self::glmf_dspark_qk_rope`].
    pub unsafe fn glmf_dspark_markov_norms(&self, w2: P, norms: M, vocab: usize, rank: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, M, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glmf_dspark_markov_norms") }?;
        check(unsafe { f(w2, norms, i(vocab)?, i(rank)?, stream) }, "Markov norms")
    }

    /// Greedy drafts [sequences, block] U32 from head `logits` [sequences *
    /// block, vocab] FP32 plus the Markov bias (`w1`, `w2` BF16 [vocab, 256],
    /// `norms` from [`Self::glmf_dspark_markov_norms`]); step 0 conditions on
    /// `anchors` [sequences] U32. At most 32 sequences.
    ///
    /// # Safety
    /// As [`Self::glmf_dspark_qk_rope`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glmf_dspark_markov(&self, logits: P, w1: P, w2: P, norms: P, anchors: P, drafts: M, workspace: M,
        sequences: usize, block: usize, vocab: usize, rank: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, M, M, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glmf_dspark_markov") }?;
        check(unsafe {
            f(logits, w1, w2, norms, anchors, drafts, workspace, i(sequences)?, i(block)?, i(vocab)?, i(rank)?, stream)
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
