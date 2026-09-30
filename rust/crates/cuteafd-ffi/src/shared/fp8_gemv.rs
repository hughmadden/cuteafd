//! W8A16 linear for skinny row counts (`native/shared/cuda/fp8_gemv.cu`):
//! E4M3 weights packed in MMA fragment order with FP32 per-row x 128-K block
//! scales, BF16 activations, FP32 accumulation. Every pointer is device
//! memory of the documented shape on the stream's device.
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

type P = *const c_void;
type M = *mut c_void;

impl NativeLibrary {
    /// Packs BF16 `w` [n, k] (n % 16 == 0, k % 128 == 0) into `packed`
    /// (n * k bytes) and `scale` ([n, k / 128] FP32) per row and 128-wide
    /// block under scale `rule`: 0 amax / 448, 1 the smallest power of two >=
    /// it, 2 whichever of the two leaves the smaller squared error.
    ///
    /// # Safety
    /// Pointers are live device buffers of those shapes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_w8a16_pack(&self, w: P, packed: M, scale: M, n: usize, k: usize, rule: i32, stream: M)
        -> Result<()> {
        type F = unsafe extern "C" fn(P, M, M, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_fp8_w8a16_pack") }?;
        let status = unsafe { f(w, packed, scale, i32::try_from(n)?, i32::try_from(k)?, rule, stream) };
        ensure!(status == 0, "FP8 W8A16 pack [{n}, {k}] failed with {status}");
        Ok(())
    }

    /// Scratch bytes [`Self::fp8_w8a16_linear`] needs for these shapes.
    pub fn fp8_w8a16_workspace(&self, rows: usize, k: usize, n: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32, i32, i32) -> usize;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_fp8_w8a16_workspace") }?;
        Ok(unsafe { f(i32::try_from(rows)?, i32::try_from(k)?, i32::try_from(n)?) })
    }

    /// `out` [rows, n] (BF16, or FP32 with `out_f32`) = `x` [rows, k] BF16 @ W^T
    /// for a weight packed by [`Self::fp8_w8a16_pack`].
    ///
    /// # Safety
    /// Pointers are live device buffers of those shapes; `workspace` holds
    /// `workspace_bytes` >= [`Self::fp8_w8a16_workspace`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_w8a16_linear(&self, x: P, packed: P, scale: P, out: M, out_f32: bool, rows: usize, k: usize,
        n: usize, workspace: M, workspace_bytes: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, M, i32, i32, i32, i32, M, usize, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_fp8_w8a16_linear") }?;
        let status = unsafe { f(x, packed, scale, out, i32::from(out_f32), i32::try_from(rows)?, i32::try_from(k)?,
            i32::try_from(n)?, workspace, workspace_bytes, stream) };
        ensure!(status == 0, "FP8 W8A16 linear [{rows}, {k}] x [{n}, {k}] failed with {status}");
        Ok(())
    }
}
