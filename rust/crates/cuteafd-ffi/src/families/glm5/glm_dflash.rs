//! DFlash2 drafter kernels (GLM 5.3, GLM 5.3 Flash and MiniMax M3.1 Flash)
//! (`native/families/glm5/cuda/glm_dflash.cu`) and the cuBLAS BF16 linear its
//! GEMMs use. Every pointer is device memory of the documented shape on the
//! stream's device; the stream orders them. The `_gemma` kernels are a Gemma-norm
//! draft's (MiniMax): FP32 norms, RoPE and activation rounded once.
//! With `fp8_kv`, retained K/V planes use `align16(kv_heads * (head_dim + 4))`
//! bytes per row: E4M3 codes, FP32 head descales, zero alignment padding.
//! Queries and speculative block K/V remain BF16 in either mode.
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

type P = *const c_void;
type M = *mut c_void;

fn check(status: i32, what: &str) -> Result<()> {
    ensure!(status == 0, "DFlash2 {what} failed with {status}");
    Ok(())
}

fn i(value: usize) -> Result<i32> {
    Ok(i32::try_from(value)?)
}

impl NativeLibrary {
    /// `out` [rows, n] BF16 = `x` [rows, k] BF16 @ `w`^T (`w` [n, k] BF16), FP32 accumulation.
    ///
    /// # Safety
    /// Pointers are live device buffers of those shapes.
    pub unsafe fn linear_bf16(&self, x: P, w: P, out: M, rows: usize, k: usize, n: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, M, usize, usize, usize, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_cuda_linear_bf16_cublas_async") }?;
        check(unsafe { f(x, w, std::ptr::null(), out, rows, k, n, stream) }, "linear")
    }

    /// Qwen3 RMSNorm of `rows` rows of `width` BF16 values.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    pub unsafe fn glm_dflash_rmsnorm(&self, x: P, w: P, out: M, rows: usize, width: usize, eps: f32, stream: M)
        -> Result<()> {
        type F = unsafe extern "C" fn(P, P, M, i32, i32, f32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_rmsnorm") }?;
        check(unsafe { f(x, w, out, i(rows)?, i(width)?, eps, stream) }, "rmsnorm")
    }

    /// Gemma RMSNorm (`x * rsqrt(mean(x^2) + eps) * (1 + w)` in FP32) of `rows` rows of `width` BF16 values.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    pub unsafe fn glm_dflash_rmsnorm_gemma(&self, x: P, w: P, out: M, rows: usize, width: usize, eps: f32, stream: M)
        -> Result<()> {
        type F = unsafe extern "C" fn(P, P, M, i32, i32, f32, M) -> i32;
        // SAFETY: the exported symbol has the native signature `F` above.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_rmsnorm_gemma") }?;
        // SAFETY: the caller's contract (# Safety) covers every pointer for these rows on `stream`;
        // the integer arguments are range-checked by `i`.
        check(unsafe { f(x, w, out, i(rows)?, i(width)?, eps, stream) }, "Gemma rmsnorm")
    }

    /// Side-0 dynamic convolution of block rows.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_conv(&self, source: P, dynamic: P, base: P, out: M, rows: usize, block_rows: usize,
        hidden: usize, group: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, M, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_conv") }?;
        check(unsafe { f(source, dynamic, base, out, i(rows)?, i(block_rows)?, i(hidden)?, i(group)?, stream) }, "conv")
    }

    /// `residual_out = residual + conv1(source)`, `normalized = RMSNorm(residual_out) * w`.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`]; `residual_out` may alias `residual`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_conv_residual_norm(&self, source: P, dynamic: P, base: P, residual: P, w: P,
        residual_out: M, normalized: M, rows: usize, block_rows: usize, hidden: usize, group: usize, eps: f32,
        stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, M, M, i32, i32, i32, i32, f32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_conv_residual_norm") }?;
        check(unsafe {
            f(source, dynamic, base, residual, w, residual_out, normalized, i(rows)?, i(block_rows)?, i(hidden)?,
                i(group)?, eps, stream)
        }, "conv residual norm")
    }

    /// [`Self::glm_dflash_conv_residual_norm`] with the Gemma norm of the FP32 sum
    /// (`residual_out` is its BF16 rounding).
    ///
    /// # Safety
    /// As [`Self::glm_dflash_conv_residual_norm`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_conv_residual_norm_gemma(&self, source: P, dynamic: P, base: P, residual: P, w: P,
        residual_out: M, normalized: M, rows: usize, block_rows: usize, hidden: usize, group: usize, eps: f32,
        stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, M, M, i32, i32, i32, i32, f32, M) -> i32;
        // SAFETY: the exported symbol has the native signature `F` above.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_conv_residual_norm_gemma") }?;
        // SAFETY: the caller's contract (# Safety) covers every pointer for these rows on `stream`;
        // the integer arguments are range-checked by `i`.
        check(unsafe {
            f(source, dynamic, base, residual, w, residual_out, normalized, i(rows)?, i(block_rows)?, i(hidden)?,
                i(group)?, eps, stream)
        }, "Gemma conv residual norm")
    }

    /// Per-head RMSNorm + RoPE of `qkv` rows; k/v rows land at `slots[row]`
    /// (row when null, skipped when negative).
    ///
    /// # Safety
    /// As [`Self::linear_bf16`]; `positions` I64 [rows], `slots` I32 [rows].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_qk_rope(&self, qkv: P, q_norm: P, k_norm: P, positions: P, slots: P, q_out: M,
        k_out: M, v_out: M, rows: usize, heads: usize, kv_heads: usize, theta: f32, eps: f32, fp8_kv: bool, stream: M)
        -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, M, M, M, i32, i32, i32, f32, f32, M) -> i32;
        let symbol: &[u8] = if fp8_kv { b"cuteafd_glm_dflash_qk_rope_fp8" } else { b"cuteafd_glm_dflash_qk_rope" };
        // SAFETY: both format-specific symbols have the native signature above.
        let f = *unsafe { self.lib.get::<F>(symbol) }?;
        check(unsafe {
            f(qkv, q_norm, k_norm, positions, slots, q_out, k_out, v_out, i(rows)?, i(heads)?, i(kv_heads)?, theta,
                eps, stream)
        }, "qk rope")
    }

    /// [`Self::glm_dflash_qk_rope`] with Gemma per-head norms and NeoX RoPE on the first `rotary`
    /// (64 or 128) channels, in FP32 rounded once.
    ///
    /// # Safety
    /// As [`Self::glm_dflash_qk_rope`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_qk_rope_gemma(&self, qkv: P, q_norm: P, k_norm: P, positions: P, slots: P, q_out: M,
        k_out: M, v_out: M, rows: usize, heads: usize, kv_heads: usize, rotary: usize, theta: f32, eps: f32,
        fp8_kv: bool, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, M, M, M, i32, i32, i32, i32, f32, f32, M) -> i32;
        let symbol: &[u8] = if fp8_kv { b"cuteafd_glm_dflash_qk_rope_gemma_fp8" }
            else { b"cuteafd_glm_dflash_qk_rope_gemma" };
        // SAFETY: both format-specific symbols have the native signature above.
        let f = *unsafe { self.lib.get::<F>(symbol) }?;
        // SAFETY: the caller's contract (# Safety) covers every pointer for these rows on `stream`;
        // the integer arguments are range-checked by `i`.
        check(unsafe {
            f(qkv, q_norm, k_norm, positions, slots, q_out, k_out, v_out, i(rows)?, i(heads)?, i(kv_heads)?,
                i(rotary)?, theta, eps, stream)
        }, "Gemma qk rope")
    }

    /// Workspace bytes of [`Self::glm_dflash_block_attention`] for these heads and block rows.
    pub fn glm_dflash_block_attention_workspace(&self, sequences: usize, heads: usize, kv_heads: usize,
        block_rows: usize, max_keys: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32, i32, i32, i32, i32) -> u64;
        // SAFETY: the exported symbol has the native signature `F` above.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_block_attention_workspace") }?;
        // SAFETY: the native workspace query takes only the range-checked sizes.
        let bytes = unsafe { f(i(sequences)?, i(heads)?, i(kv_heads)?, i(block_rows)?, i(max_keys)?) };
        ensure!(bytes > 0, "DFlash2 block attention workspace geometry");
        Ok(usize::try_from(bytes)?)
    }

    /// Block attention over each sequence's ring context and block (`causal`: block row r
    /// sees block rows 0..=r); `window` > 0 limits each block row to context positions within it.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`]; the I32 tables hold one entry per sequence.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_block_attention(&self, q: P, k_block: P, v_block: P, k_ring: P, v_ring: P,
        seq_slots: P, ctx_lengths: P, ctx_ends: P, out: M, workspace: M, sequences: usize, block_rows: usize,
        heads: usize, kv_heads: usize, ring: usize, max_keys: usize, window: usize, causal: bool, scale: f32,
        fp8_kv: bool, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, P, P, P, M, M, i32, i32, i32, i32, i32, i32, i32, i32, f32, M) -> i32;
        let symbol: &[u8] = if fp8_kv { b"cuteafd_glm_dflash_block_attention_fp8" }
            else { b"cuteafd_glm_dflash_block_attention" };
        // SAFETY: both format-specific symbols have the native signature above.
        let f = *unsafe { self.lib.get::<F>(symbol) }?;
        // SAFETY: the caller's contract (# Safety) covers every pointer for these rows on `stream`;
        // the integer arguments are range-checked by `i`.
        check(unsafe {
            f(q, k_block, v_block, k_ring, v_ring, seq_slots, ctx_lengths, ctx_ends, out, workspace, i(sequences)?,
                i(block_rows)?, i(heads)?, i(kv_heads)?, i(ring)?, i(max_keys)?, i(window)?, i32::from(causal),
                scale, stream)
        }, "block attention")
    }

    /// `out` [rows, inter] = silu(gate) * up of `gate_up` [rows, 2 * inter].
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    pub unsafe fn glm_dflash_silu_mul(&self, gate_up: P, out: M, rows: usize, inter: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, M, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_silu_mul") }?;
        check(unsafe { f(gate_up, out, i(rows)?, i(inter)?, stream) }, "silu mul")
    }

    /// `out` [rows, inter] = SwiGLU-OAI of `gate_up` [rows, 2 * inter] ([gate | up]):
    /// `g * sigmoid(alpha g) * (u + 1)`, `g = min(gate, limit)`, `u = clamp(up, ±limit)`.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_swiglu_oai(&self, gate_up: P, out: M, rows: usize, inter: usize, alpha: f32,
        limit: f32, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, M, i32, i32, f32, f32, M) -> i32;
        // SAFETY: the exported symbol has the native signature `F` above.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_swiglu_oai") }?;
        // SAFETY: the caller's contract (# Safety) covers every pointer for these rows on `stream`;
        // the integer arguments are range-checked by `i`.
        check(unsafe { f(gate_up, out, i(rows)?, i(inter)?, alpha, limit, stream) }, "SwiGLU-OAI")
    }

    /// Workspace bytes of [`Self::glm_dflash_topk`] for `rows` drafted rows.
    pub fn glm_dflash_topk_workspace(&self, rows: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32) -> u64;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_topk_workspace") }?;
        Ok(usize::try_from(unsafe { f(i(rows)?) })?)
    }

    /// Top-16 (F32 values of BF16-rounded logits, I32 ids) of block rows
    /// 1..=drafts of each sequence.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_topk(&self, logits: P, unary: M, candidates: M, workspace: M, sequences: usize,
        block_rows: usize, drafts: usize, vocab: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, M, M, M, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_topk") }?;
        check(unsafe {
            f(logits, unary, candidates, workspace, i(sequences)?, i(block_rows)?, i(drafts)?, i(vocab)?, stream)
        }, "top-k")
    }

    /// Greedy candidate-selector walk: tokens [sequences, drafts] U32 and
    /// features [sequences, drafts, 4] F32.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_select(&self, pred_cb: P, succ_cb: P, projected: P, candidates: P, unary: P,
        anchors: P, tokens: M, features: M, sequences: usize, block_rows: usize, drafts: usize, rank: usize,
        stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, P, M, M, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_select") }?;
        check(unsafe {
            f(pred_cb, succ_cb, projected, candidates, unary, anchors, tokens, features, i(sequences)?,
                i(block_rows)?, i(drafts)?, i(rank)?, stream)
        }, "select")
    }

    /// [`Self::glm_dflash_select`] scoring each candidate as the FP32 sum of its unary logit and
    /// BF16 edge (SGLang's DFlash2 selector) rather than its BF16 rounding.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_select_fp32_score(&self, pred_cb: P, succ_cb: P, projected: P, candidates: P,
        unary: P, anchors: P, tokens: M, features: M, sequences: usize, block_rows: usize, drafts: usize,
        rank: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, P, M, M, i32, i32, i32, i32, M) -> i32;
        // SAFETY: the exported symbol has the native signature `F` above.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_select_fp32_score") }?;
        // SAFETY: the caller's contract (# Safety) covers every pointer for these rows on `stream`;
        // the integer arguments are range-checked by `i`.
        check(unsafe {
            f(pred_cb, succ_cb, projected, candidates, unary, anchors, tokens, features, i(sequences)?,
                i(block_rows)?, i(drafts)?, i(rank)?, stream)
        }, "select")
    }

    /// `dst[row, offset..offset + width] = src[row]` for BF16 rows.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_tap(&self, src: P, dst: M, rows: usize, width: usize, stride: usize, offset: usize,
        stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, M, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_tap") }?;
        check(unsafe { f(src, dst, i(rows)?, i(width)?, i(stride)?, i(offset)?, stream) }, "tap")
    }

    /// `dst[row, offset..offset + width] = bf16(mean of the row's hc streams)`
    /// for `src` [rows, hc, width] BF16 (the mHC contraction DFlash2 taps).
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_tap_mean(&self, src: P, dst: M, rows: usize, width: usize, hc: usize, stride: usize,
        offset: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, M, i32, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_tap_mean") }?;
        check(unsafe { f(src, dst, i(rows)?, i(width)?, i(hc)?, i(stride)?, i(offset)?, stream) }, "tap mean")
    }

    /// `dst[row, offset..offset + width] = bf16(src[row])` for F32 rows (an FP32 residual's
    /// tap; round to nearest even).
    ///
    /// # Safety
    /// As [`Self::linear_bf16`]; `src` 16-byte aligned rows of `width` F32, `dst` 8-byte aligned.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_tap_f32(&self, src: P, dst: M, rows: usize, width: usize, stride: usize, offset: usize,
        stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, M, i32, i32, i32, i32, M) -> i32;
        // SAFETY: the exported symbol has the native signature `F` above.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_tap_f32") }?;
        // SAFETY: the caller's contract (# Safety) covers every pointer for these rows on `stream`;
        // the integer arguments are range-checked by `i`.
        check(unsafe { f(src, dst, i(rows)?, i(width)?, i(stride)?, i(offset)?, stream) }, "F32 tap")
    }
}
