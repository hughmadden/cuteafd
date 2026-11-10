//! Exact 12-bit BF16 vocabulary heads (`native/shared/cuda/packed_head.cu`):
//! every BF16 weight as a coarse byte (sign, 4-bit exponent code over a per-row
//! base, mantissa bits 6..4) and a fine nibble (mantissa bits 3..0) in MMA
//! fragment order, plus a per-row table of the few values outside their row's
//! 16-exponent window. The exact GEMV reads both planes and the table (the BF16
//! values, FP32 sums); the coarse GEMV reads the coarse plane alone. `planes`
//! hold device pointers in `cuteafd_core::packed_head::Region` order: coarse,
//! fine, base, row_ptr, columns, values. Every pointer is device memory of the
//! documented shape on the stream's device.
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

type P = *const c_void;
type M = *mut c_void;

/// One packed head's six device arrays, in the native `planes` order.
pub type PackedHeadPlanes = [*mut c_void; 6];

impl NativeLibrary {
    /// Packs BF16 `w` [n, k] (n % 16 == 0, k % 128 == 0, n * k < 2^32) into
    /// `planes` with a table of `capacity` entries, then verifies the pack.
    /// `status` receives four u64 words: exceptions, failing rows, the first
    /// failing row and the most exceptions in a row. Asynchronous on `stream`.
    ///
    /// # Safety
    /// `w` is a live [n, k] BF16 buffer; `planes` and `status` are live device
    /// buffers of the sizes `cuteafd_core::packed_head::PackedHeadLayout` gives.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn packed_head_pack(&self, w: P, planes: &PackedHeadPlanes, capacity: usize, n: usize, k: usize,
        status: M, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, *const M, i64, i32, i32, M, M) -> i32;
        // SAFETY: the symbol has this signature in packed_head.cu.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_packed_head_pack") }?;
        // SAFETY: the caller's contract; the native call reads `planes` before it returns.
        let code = unsafe { f(w, planes.as_ptr(), i64::try_from(capacity)?, i32::try_from(n)?, i32::try_from(k)?,
            status, stream) };
        ensure!(code == 0, "packed head pack [{n}, {k}] failed with {code}");
        Ok(())
    }

    /// Re-verifies a packed head against its BF16 source, resetting and
    /// writing status words 1 (failing rows) and 2 (the first one).
    ///
    /// # Safety
    /// As [`Self::packed_head_pack`], with `planes` holding a pack of `w`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn packed_head_verify(&self, w: P, planes: &PackedHeadPlanes, capacity: usize, n: usize, k: usize,
        status: M, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, *const M, i64, i32, i32, M, M) -> i32;
        // SAFETY: the symbol has this signature in packed_head.cu.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_packed_head_verify") }?;
        // SAFETY: the caller's contract; the native call reads `planes` before it returns.
        let code = unsafe { f(w, planes.as_ptr(), i64::try_from(capacity)?, i32::try_from(n)?, i32::try_from(k)?,
            status, stream) };
        ensure!(code == 0, "packed head verify [{n}, {k}] failed with {code}");
        Ok(())
    }

    /// Scratch bytes [`Self::packed_head_linear`] needs for these shapes.
    pub fn packed_head_workspace(&self, rows: usize, k: usize, n: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32, i32, i32) -> usize;
        // SAFETY: the symbol has this signature in packed_head.cu.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_packed_head_workspace") }?;
        // SAFETY: a pure shape query.
        let bytes = unsafe { f(i32::try_from(rows)?, i32::try_from(k)?, i32::try_from(n)?) };
        ensure!(bytes > 0, "packed head workspace: invalid shape [{rows}, {k}] x [{n}, {k}]");
        Ok(bytes)
    }

    /// `out` [rows, n] (BF16, or FP32 with `out_f32`) = `x` [rows, k] BF16 @ W^T
    /// for a head packed by [`Self::packed_head_pack`]: the exact weights, or
    /// with `coarse` the coarse plane alone (mantissa bits 3..0 at 0b1000).
    ///
    /// # Safety
    /// `x`, `out` and `planes` are live device buffers of those shapes;
    /// `workspace` holds `workspace_bytes` >= [`Self::packed_head_workspace`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn packed_head_linear(&self, x: P, planes: &PackedHeadPlanes, out: M, out_f32: bool, rows: usize,
        k: usize, n: usize, coarse: bool, workspace: M, workspace_bytes: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, *const M, M, i32, i32, i32, i32, i32, M, usize, M) -> i32;
        // SAFETY: the symbol has this signature in packed_head.cu.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_packed_head_linear") }?;
        // SAFETY: the caller's contract; the native call reads `planes` before it returns.
        let code = unsafe { f(x, planes.as_ptr(), out, i32::from(out_f32), i32::try_from(rows)?,
            i32::try_from(k)?, i32::try_from(n)?, i32::from(coarse), workspace, workspace_bytes, stream) };
        ensure!(code == 0, "packed head linear ({}) [{rows}, {k}] x [{n}, {k}] failed with {code}",
            if coarse { "coarse" } else { "exact" });
        Ok(())
    }
}
