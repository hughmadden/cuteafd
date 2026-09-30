//! E4M3 copies of BF16 GEMM weights for skinny steps (drafters, draft heads):
//! packed at load by `fp8_gemv.cu` (FP32 scales per output row and 128-wide
//! K block, `amax / 448` or with `pow2` the smallest power of two >= it) and
//! applied by its W8A16 tensor-core GEMV (BF16 activations, FP32 sums).
use crate::v41_memory::DeviceAllocation;
use anyhow::{ensure, Result};
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;

/// An E4M3 copy of a `[n, k]` BF16 weight in the GEMV's fragment order.
pub(crate) struct Fp8Weight<'a> {
    packed: DeviceAllocation<'a>,
    scale: DeviceAllocation<'a>,
    pub n: usize,
    pub k: usize,
}

impl<'a> Fp8Weight<'a> {
    /// Packs the live BF16 weight `w` [n, k] on `stream` (the caller
    /// synchronizes before the source is freed).
    pub fn pack(library: &'a NativeLibrary, w: *const c_void, n: usize, k: usize, pow2: bool, stream: *mut c_void)
        -> Result<Self> {
        ensure!(n % 16 == 0 && k % 128 == 0, "FP8 copy of [{n}, {k}]: needs n % 16 == 0 and k % 128 == 0");
        let packed = DeviceAllocation::new(library, n * k)?;
        let scale = DeviceAllocation::new(library, n * k / 128 * 4)?;
        // SAFETY: `w` is a live [n, k] BF16 weight; the new buffers hold the packed copy.
        unsafe { library.fp8_w8a16_pack(w, packed.buffer.ptr, scale.buffer.ptr, n, k, pow2, stream)? };
        Ok(Self { packed, scale, n, k })
    }

    /// Device bytes of the copy and its scales.
    pub fn bytes(&self) -> usize {
        self.n * self.k + self.n * self.k / 128 * 4
    }

    /// `out` [rows, n'] = `x` [rows, k] @ rows `first..first + n'` of the
    /// weight (`first` a multiple of 16), BF16 out or FP32 with `out_f32`.
    ///
    /// # Safety
    /// `x` and `out` are live device buffers of those shapes; `scratch` was
    /// sized for at least `rows` rows of this shape.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn apply(&self, library: &NativeLibrary, x: *const c_void, out: *mut c_void, out_f32: bool, rows: usize,
        first: usize, n: usize, scratch: &DeviceAllocation<'_>, stream: *mut c_void) -> Result<()> {
        ensure!(first % 16 == 0 && first + n <= self.n, "FP8 rows {first}..{} of {}", first + n, self.n);
        // SAFETY: the offsets stay inside the packed copy (16-row tiles are contiguous) and its scales.
        let (packed, scale) = unsafe {
            (self.packed.buffer.ptr.cast::<u8>().add(first * self.k).cast::<c_void>(),
                self.scale.buffer.ptr.cast::<u8>().add(first * self.k / 128 * 4).cast::<c_void>())
        };
        // SAFETY: the caller's contract.
        unsafe { library.fp8_w8a16_linear(x, packed, scale, out, out_f32, rows, self.k, n, scratch.buffer.ptr,
            scratch.buffer.bytes, stream) }
    }
}

/// GEMV scratch for up to `rows` rows of every `(k, n)` shape.
pub(crate) fn scratch<'a>(library: &'a NativeLibrary, rows: usize, shapes: &[(usize, usize)])
    -> Result<DeviceAllocation<'a>> {
    let bytes = shapes.iter().map(|&(k, n)| library.fp8_w8a16_workspace(rows, k, n))
        .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
    DeviceAllocation::new(library, bytes.max(256))
}
