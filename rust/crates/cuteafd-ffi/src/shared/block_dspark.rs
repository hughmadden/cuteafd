//! Block dSpark drafter kernels (`native/families/glm5_flash/cuda/glmf_dspark.cu`),
//! one binding for every drafter that runs them: GLM 5.3 Flash and Qwen 3.5
//! (the shared drafter) and MiniMax M3. Every pointer is device memory of the
//! documented shape on the stream's device; the stream orders them.
//! With `fp8_kv`, retained K/V planes use `align16(kv_heads * (head_dim + 4))`
//! bytes per row: E4M3 codes, FP32 head descales, zero alignment padding.
//! Queries and speculative block K/V remain BF16 in either mode.
//! The draft heads ([`NativeLibrary::dspark_markov_norms`],
//! [`NativeLibrary::dspark_draft_heads`]) check their operands' extents first.
use crate::{CuteafdDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

type P = *const c_void;
type M = *mut c_void;

/// The Markov head's rank, the only one the kernels take.
pub const MARKOV_RANK: usize = 256;
/// Most sequences one Markov launch drafts.
pub const MARKOV_SEQUENCES: usize = 32;

fn check(status: i32, what: &str) -> Result<()> {
    ensure!(status == 0, "dSpark {what} failed with {status}");
    Ok(())
}

fn i(value: usize) -> Result<i32> {
    Ok(i32::try_from(value)?)
}

/// The product of `factors`, refusing overflow.
pub fn extent(factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |n, &factor| n.checked_mul(factor)).context("dSpark operand extent overflow")
}

/// `buffer` holds `bytes` (non-null when nonempty) at `alignment`.
pub fn check_operand(buffer: CuteafdDeviceBuffer, bytes: usize, alignment: usize) -> Result<()> {
    ensure!(bytes <= buffer.bytes && (bytes == 0 || !buffer.ptr.is_null()), "dSpark operand extent");
    ensure!(buffer.ptr as usize % alignment == 0, "dSpark operand alignment");
    ensure!((buffer.ptr as usize).checked_add(bytes).is_some(), "dSpark operand pointer extent overflow");
    Ok(())
}

/// The first `a_bytes` of `a` and `b_bytes` of `b` do not overlap.
pub fn disjoint(a: CuteafdDeviceBuffer, a_bytes: usize, b: CuteafdDeviceBuffer, b_bytes: usize) -> Result<()> {
    check_operand(a, a_bytes, 1)?;
    check_operand(b, b_bytes, 1)?;
    let (a_start, b_start) = (a.ptr as usize, b.ptr as usize);
    ensure!(a_bytes == 0 || b_bytes == 0 || a_start + a_bytes <= b_start || b_start + b_bytes <= a_start,
        "dSpark operands overlap");
    Ok(())
}

/// A draft step's Markov head (`w1`, `w2` BF16 [vocab, 256], `norms` the
/// row norms of `w2`) and confidence head (`weight` BF16 [width + 256],
/// `bias` BF16 [1]).
#[derive(Debug, Clone, Copy)]
pub struct DraftHeads {
    pub w1: CuteafdDeviceBuffer,
    pub w2: CuteafdDeviceBuffer,
    pub norms: CuteafdDeviceBuffer,
    pub weight: CuteafdDeviceBuffer,
    pub bias: CuteafdDeviceBuffer,
}

impl NativeLibrary {
    /// `norms` [vocab] F32 = row norms of `w2` BF16 [vocab, 256], once at load.
    ///
    /// # Safety
    /// Both buffers stay live until `stream` reaches the kernel.
    pub unsafe fn dspark_markov_norms(&self, w2: CuteafdDeviceBuffer, norms: CuteafdDeviceBuffer, vocab: usize,
        stream: M) -> Result<()> {
        // The norm grid counts 32 threads per row in i32.
        ensure!(vocab > 0 && vocab <= (i32::MAX as usize - 255) / 32, "dSpark vocabulary exceeds native norm limits");
        let (weight_bytes, norm_bytes) = (extent(&[vocab, MARKOV_RANK, 2])?, extent(&[vocab, 4])?);
        check_operand(w2, weight_bytes, 16)?;
        check_operand(norms, norm_bytes, 4)?;
        disjoint(w2, weight_bytes, norms, norm_bytes)?;
        // SAFETY: the checked extents hold the fixed rank's rows; the caller's lifetimes.
        unsafe { self.glmf_dspark_markov_norms(w2.ptr, norms.ptr, vocab, MARKOV_RANK, stream) }
    }

    /// One draft step's heads over its `sequences * block` rows: greedy
    /// Markov drafts `tokens` from the head `logits` (F32 [rows, vocab]) and
    /// each sequence's anchor (U32 [sequences], on the device), then their
    /// `confidence` (F32 [rows]) from the final `hidden` rows (BF16
    /// [rows, width]). Both kernels are queued on `stream`; confidence reads
    /// only the bounded Markov result.
    ///
    /// # Safety
    /// Every operand stays live until `stream` reaches the kernels, and the
    /// anchors are token ids below `vocab`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dspark_draft_heads(&self, heads: &DraftHeads, logits: CuteafdDeviceBuffer, hidden: CuteafdDeviceBuffer,
        anchors: CuteafdDeviceBuffer, tokens: CuteafdDeviceBuffer, workspace: CuteafdDeviceBuffer,
        confidence: CuteafdDeviceBuffer, sequences: usize, block: usize, vocab: usize, width: usize, stream: M)
        -> Result<()> {
        ensure!((1..=MARKOV_SEQUENCES).contains(&sequences) && block > 0 && vocab > 0 && width > 0,
            "dSpark draft heads geometry");
        let rows = extent(&[sequences, block])?;
        i(rows)?;
        let workspace_bytes = self.glmf_dspark_markov_workspace(sequences, block)?;
        let operands = [logits, heads.w1, heads.w2, heads.norms, hidden, heads.weight, heads.bias, anchors, tokens,
            workspace, confidence];
        let sizes = [extent(&[rows, vocab, 4])?, extent(&[vocab, MARKOV_RANK, 2])?, extent(&[vocab, MARKOV_RANK, 2])?,
            extent(&[vocab, 4])?, extent(&[rows, width, 2])?, extent(&[width + MARKOV_RANK, 2])?, 2, sequences * 4,
            rows * 4, workspace_bytes, rows * 4];
        for (index, buffer) in operands.iter().enumerate() {
            check_operand(*buffer, sizes[index], match index { 1 | 2 => 16, 4..=6 => 2, _ => 4 })?;
        }
        // Anchors, drafts, workspace and confidence alias nothing before them.
        for out in 7..operands.len() {
            for input in 0..out { disjoint(operands[out], sizes[out], operands[input], sizes[input])?; }
        }
        // SAFETY: checked extents bound every indexed read; invalid proposals
        // fall back to a bounded token before the confidence launch.
        unsafe {
            self.glmf_dspark_markov(logits.ptr, heads.w1.ptr, heads.w2.ptr, heads.norms.ptr, anchors.ptr, tokens.ptr,
                workspace.ptr, sequences, block, vocab, MARKOV_RANK, stream)?;
            self.glmf_dspark_confidence(hidden.ptr, heads.w1.ptr, heads.weight.ptr, heads.bias.ptr, anchors.ptr,
                tokens.ptr, confidence.ptr, sequences, block, width, MARKOV_RANK, stream)
        }
    }


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

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(address: usize, bytes: usize) -> CuteafdDeviceBuffer {
        CuteafdDeviceBuffer { ptr: address as *mut _, bytes, device_id: 0, flags: 0 }
    }

    #[test]
    fn draft_extents_reject_overflow_short_buffers_and_aliases() -> Result<()> {
        assert_eq!(extent(&[7, 64, 128, 2])?, 114688);
        assert!(extent(&[usize::MAX, 2]).is_err());
        assert!(check_operand(buffer(16, 16), 17, 16).is_err());
        assert!(check_operand(buffer(17, 16), 16, 16).is_err());
        assert!(check_operand(buffer(usize::MAX, 1), 1, 1).is_err());
        assert!(disjoint(buffer(16, 16), 16, buffer(24, 16), 16).is_err());
        disjoint(buffer(16, 16), 16, buffer(32, 16), 16)?;
        Ok(())
    }
}
