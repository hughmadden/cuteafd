//! DFlash drafter kernels of MiMo V2.6 Pro (`native/families/mimo_v2/cuda/mimo_dflash.cu`):
//! per-head norm + partial RoPE with value scale, sink attention over the
//! block and a sliding window of ring context, and the residual add + norm.
//! Every pointer is device memory of the documented shape on the stream's
//! device; the stream orders them.
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

type P = *const c_void;
type M = *mut c_void;

fn check(status: i32, what: &str) -> Result<()> {
    ensure!(status == 0, "MiMo DFlash {what} failed with {status}");
    Ok(())
}

fn i(value: usize) -> Result<i32> {
    Ok(i32::try_from(value)?)
}

impl NativeLibrary {
    /// `qkv` [rows, (heads + 2 kv) * 128]: q heads -> `q_out` [rows, heads, 128]
    /// (normed, RoPE on the first `rope_dim` dims); k (normed, roped) and
    /// bf16(v * v_scale) -> row `slots[row]` (row when null, skipped when
    /// negative) of `k_out`/`v_out` [*, kv_heads, 128].
    ///
    /// # Safety
    /// Pointers are live device buffers of those shapes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mimo_dflash_qk_rope(&self, qkv: P, q_norm: P, k_norm: P, positions: P, slots: P, q_out: M,
        k_out: M, v_out: M, rows: usize, heads: usize, kv_heads: usize, rope_dim: usize, theta: f32, eps: f32,
        v_scale: f32, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, M, M, M, i32, i32, i32, i32, f32, f32, f32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_mimo_dflash_qk_rope") }?;
        check(unsafe { f(qkv, q_norm, k_norm, positions, slots, q_out, k_out, v_out, i(rows)?, i(heads)?,
            i(kv_heads)?, i(rope_dim)?, theta, eps, v_scale, stream) }, "qk_rope")
    }

    pub fn mimo_dflash_attention_workspace(&self, sequences: usize, heads: usize, kv_heads: usize, block_rows: usize,
        max_keys: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32, i32, i32, i32, i32) -> u64;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_mimo_dflash_attention_workspace") }?;
        Ok(unsafe { f(i(sequences)?, i(heads)?, i(kv_heads)?, i(block_rows)?, i(max_keys)?) } as usize)
    }

    /// Block attention with per-head sinks (`sinks` BF16 [heads] or null); see
    /// `cuteafd_mimo_dflash_attention`.
    ///
    /// # Safety
    /// As [`Self::mimo_dflash_qk_rope`]; `workspace` holds
    /// [`Self::mimo_dflash_attention_workspace`] bytes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mimo_dflash_attention(&self, q: P, k_block: P, v_block: P, k_ring: P, v_ring: P, seq_slots: P,
        ctx_lengths: P, ctx_ends: P, sinks: P, out: M, workspace: M, sequences: usize, block_rows: usize,
        heads: usize, kv_heads: usize, ring: usize, max_keys: usize, window: usize, scale: f32, stream: M)
        -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, P, P, P, P, M, M, i32, i32, i32, i32, i32, i32, i32, f32, M)
            -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_mimo_dflash_attention") }?;
        check(unsafe { f(q, k_block, v_block, k_ring, v_ring, seq_slots, ctx_lengths, ctx_ends, sinks, out, workspace,
            i(sequences)?, i(block_rows)?, i(heads)?, i(kv_heads)?, i(ring)?, i(max_keys)?, i(window)?, scale,
            stream) }, "attention")
    }

    /// `h += delta` (BF16), `n = RMSNorm(h) * w`.
    ///
    /// # Safety
    /// As [`Self::mimo_dflash_qk_rope`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mimo_dflash_add_norm(&self, h: M, delta: P, w: P, n: M, rows: usize, width: usize, eps: f32,
        stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(M, P, P, M, i32, i32, f32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_mimo_dflash_add_norm") }?;
        check(unsafe { f(h, delta, w, n, i(rows)?, i(width)?, eps, stream) }, "add_norm")
    }
}
