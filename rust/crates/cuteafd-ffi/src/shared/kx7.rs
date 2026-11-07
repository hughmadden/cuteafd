//! KX7 exact Huffman-packed BF16 matrices (`native/shared/cuda/kx7.cu`, ABI in
//! `native/shared/include/cuteafd_kx7.h`; format and packer in `cuteafd_core::kx7`). A pack
//! reaches the native code as one [`Kx7DevicePack`]: the header scalars its kernels need and its
//! device buffers. Format v1.1 has four: the decode tables (u32), the stream offsets (u32 per
//! stream plus the total), the stream lengths (u16) and the streams (bytes, 16-byte aligned,
//! with `cuteafd_core::kx7::READ_AHEAD_BYTES` zeros after them). v0 decodes every stream into a
//! BF16 workspace and runs the BF16 cuBLAS GEMM over it, so a KX7 linear is bit for bit that
//! GEMM over the source weights. Every pointer is device memory of the documented shape on the
//! stream's device.
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

type P = *const c_void;
type M = *mut c_void;

/// `cuteafd_kx7_pack_t`: fields are only appended; `struct_bytes` tells the native code how far
/// the caller's struct goes, so a format that needs another buffer or scalar adds a field here
/// and there without changing any call site.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Kx7DevicePack {
    pub struct_bytes: u32,
    pub format: u32,
    pub n: i32,
    pub k: i32,
    pub slices: i32,
    pub streams_per_row: i32,
    pub symbols_per_stream: i32,
    pub quad_stride: u32,
    pub ragged: u32,
    pub esc_shift: u32,
    pub esc_mask: u32,
    pub table_words: u32,
    pub tables: P,
    pub offsets: P,
    pub lengths: P,
    pub streams: P,
}

/// The device buffers of one pack, in [`Kx7DevicePack`] order: tables, offsets, lengths, streams.
pub type Kx7Buffers = [P; 4];

impl Kx7DevicePack {
    /// `CUTEAFD_KX7_FORMAT_V11`: KX7 v1.1 (`cuteafd_core::kx7::V11_LAYOUT` and `V11_CODE`).
    pub const FORMAT_V11: u32 = 11;
    /// v1.1's bytes between a stream's interleaved quads.
    pub const V11_QUAD_STRIDE: u32 = 512;

    /// A v1.1 pack of an [n, k] matrix over `slices` K slices whose two-class code takes an
    /// ESC of `esc_len` bits and `rare_bits` raw rank bits, with `table_words` of decode tables.
    #[allow(clippy::too_many_arguments)]
    pub fn v11(n: usize, k: usize, slices: usize, esc_len: u32, rare_bits: u32, table_words: usize,
        [tables, offsets, lengths, streams]: Kx7Buffers) -> Result<Self> {
        ensure!(slices > 0 && k % (4 * slices) == 0 && rare_bits > 0 && esc_len + rare_bits <= 31,
            "KX7 v1.1 header: [{n}, {k}] over {slices} slices, ESC {esc_len} + {rare_bits} bits");
        Ok(Self { struct_bytes: std::mem::size_of::<Self>() as u32, format: Self::FORMAT_V11,
            n: i32::try_from(n)?, k: i32::try_from(k)?, slices: i32::try_from(slices)?,
            streams_per_row: i32::try_from(4 * slices)?, symbols_per_stream: i32::try_from(k / (4 * slices))?,
            quad_stride: Self::V11_QUAD_STRIDE, ragged: 1, esc_shift: 31 - esc_len - rare_bits,
            esc_mask: (1u32 << (rare_bits + 1)) - 2, table_words: u32::try_from(table_words)?,
            tables, offsets, lengths, streams })
    }
}

impl NativeLibrary {
    /// Workspace bytes [`Self::kx7_linear`] needs for `rows` rows over an [n, k] pack.
    pub fn kx7_workspace(&self, rows: usize, n: usize, k: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32, i32, i32) -> usize;
        // SAFETY: the symbol has this signature in kx7.cu.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_kx7_workspace") }?;
        // SAFETY: a pure shape query.
        let bytes = unsafe { f(i32::try_from(rows)?, i32::try_from(n)?, i32::try_from(k)?) };
        ensure!(bytes > 0, "KX7 workspace: invalid shape {rows} x [{n}, {k}]");
        Ok(bytes)
    }

    /// Decodes `pack` into BF16 `out` [n, k] (16-byte aligned). Asynchronous on `stream`, graph
    /// capturable.
    ///
    /// # Safety
    /// `pack` names a live pack's device buffers (as `cuteafd_core::kx7::Kx7DeviceBytes` sizes
    /// them); `out` is live device memory of `n * k * 2` bytes.
    pub unsafe fn kx7_decode(&self, pack: &Kx7DevicePack, out: M, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(*const Kx7DevicePack, M, M) -> i32;
        // SAFETY: the symbol has this signature in kx7.cu.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_kx7_decode") }?;
        // SAFETY: the caller's contract; the native call reads `pack` before it returns.
        let code = unsafe { f(pack, out, stream) };
        self.status_to_result("KX7 decode", code)
    }

    /// `out` [rows, n] BF16 = `x` [rows, k] BF16 @ W^T for W = `pack` (FP32 accumulation).
    /// Asynchronous on `stream` and graph capturable (no allocation, no host synchronization).
    ///
    /// # Safety
    /// As [`Self::kx7_decode`]; `x` and `out` are live device buffers of those shapes and
    /// `workspace` holds `workspace_bytes` >= [`Self::kx7_workspace`], unused by any other
    /// stream until the call's work retires.
    pub unsafe fn kx7_linear(&self, x: P, pack: &Kx7DevicePack, out: M, rows: usize, workspace: M,
        workspace_bytes: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, *const Kx7DevicePack, M, i32, M, usize, M) -> i32;
        // SAFETY: the symbol has this signature in kx7.cu.
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_kx7_linear") }?;
        // SAFETY: the caller's contract; the native call reads `pack` before it returns.
        let code = unsafe { f(x, pack, out, i32::try_from(rows)?, workspace, workspace_bytes, stream) };
        self.status_to_result("KX7 linear", code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header matches `cuteafd_kx7_pack_t` on LP64: 48 bytes of scalars, four pointers.
    #[test]
    fn kx7_device_pack_has_the_native_layout() {
        assert_eq!(std::mem::size_of::<Kx7DevicePack>(), 80);
        assert_eq!(std::mem::offset_of!(Kx7DevicePack, table_words), 44);
        assert_eq!(std::mem::offset_of!(Kx7DevicePack, tables), 48);
        assert_eq!(std::mem::offset_of!(Kx7DevicePack, streams), 72);
        let p = std::ptr::without_provenance::<c_void>(16);
        let pack = Kx7DevicePack::v11(8192, 4096, 4, 3, 12, 4096, [p; 4]).unwrap();
        assert_eq!((pack.struct_bytes, pack.format, pack.n, pack.k, pack.slices), (80, 11, 8192, 4096, 4));
        assert_eq!((pack.streams_per_row, pack.symbols_per_stream, pack.quad_stride, pack.ragged), (16, 256, 512, 1));
        assert_eq!((pack.esc_shift, pack.esc_mask, pack.table_words), (16, 0x1ffe, 4096));
        assert!(Kx7DevicePack::v11(1 << 31, 4096, 4, 3, 12, 4096, [p; 4]).is_err());
        assert!(Kx7DevicePack::v11(8192, 4096, 0, 3, 12, 4096, [p; 4]).is_err());
    }

    #[test]
    fn kx7_linear_native_failure_is_tagged() -> Result<()> {
        let library = crate::tests::draft_failure_library()?;
        let p: M = std::ptr::null_mut();
        let pack = Kx7DevicePack::v11(8192, 4096, 4, 3, 12, 4096, [p.cast_const(); 4])?;
        // SAFETY: the CPU fixture export returns an error without touching pointers.
        crate::tests::assert_draft_native_failure(unsafe { library.kx7_linear(p, &pack, p, 1, p, 0, p) });
        Ok(())
    }
}
