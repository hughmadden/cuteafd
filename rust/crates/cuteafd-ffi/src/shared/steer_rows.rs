//! Weight-variant row steering (`native/shared/src/steer_rows_aot.cc`,
//! `native/shared/include/cuteafd_steer_rows.h`): `y -= A (B^T y)` in place
//! on the BF16 rows a variant request owns. Base rows are never written.
//! `cuteafd_core::steer_rows::steer_rows_ref` is the bit-exact host reference.
use crate::{CuteafdDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Context, Result};
use cuteafd_core::steer_rows::check_steer_shape;
use std::ffi::c_void;

type LoadFn = unsafe extern "C" fn() -> i32;
type SteerFn = unsafe extern "C" fn(*mut u16, i64, i32, i32, *const i32, *const f32, *const f32,
    *const i32, i32, *mut c_void) -> i32;

/// The launch shape: `rows` rows of `x` (`[rows, ld]` BF16, first `h`
/// columns), tables `[variants + 1][k_max][h]` FP32 and `k` `[variants + 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SteerRowsShape {
    pub rows: usize,
    pub h: usize,
    pub ld: usize,
    pub k_max: usize,
    /// Configured variants V (table slots 1..=V; slot 0 is base).
    pub variants: usize,
}

impl SteerRowsShape {
    /// Device bytes each operand must hold.
    pub fn x_bytes(self) -> usize {
        if self.rows == 0 { 0 } else { ((self.rows - 1) * self.ld + self.h) * 2 }
    }
    pub fn vidx_bytes(self) -> usize {
        self.rows * 4
    }
    pub fn table_bytes(self) -> usize {
        (self.variants + 1) * self.k_max * self.h * 4
    }
    pub fn k_bytes(self) -> usize {
        (self.variants + 1) * 4
    }

    fn check(self) -> Result<()> {
        check_steer_shape(self.h, self.ld, self.k_max)?;
        ensure!((1..=7).contains(&self.variants), "steer_rows variants={} must be in 1..=7", self.variants);
        ensure!(i32::try_from(self.rows).is_ok() && i64::try_from(self.ld).is_ok(),
            "steer_rows rows={} ld={} exceed the native ABI", self.rows, self.ld);
        Ok(())
    }
}

/// `cuteafd_steer_rows`, with both programs loaded on the binding device.
pub struct SteerRows<'a> {
    _library: &'a NativeLibrary,
    steer: SteerFn,
    device: i32,
}

impl NativeLibrary {
    /// Binds the steering kernel and loads it onto the current device, so a
    /// later launch never loads a module (safe inside graph capture).
    pub fn steer_rows(&self) -> Result<SteerRows<'_>> {
        // SAFETY: symbol types match native/shared/include/cuteafd_steer_rows.h.
        let (load, steer) = unsafe {
            (*self.lib.get::<LoadFn>(b"cuteafd_steer_rows_load")
                .context("native library lacks cuteafd_steer_rows_load (CUTEAFD_ENABLE_STEER_ROWS_AOT)")?,
             *self.lib.get::<SteerFn>(b"cuteafd_steer_rows")
                .context("native library lacks cuteafd_steer_rows (CUTEAFD_ENABLE_STEER_ROWS_AOT)")?)
        };
        // SAFETY: the loader takes no arguments and only loads static modules.
        let status = unsafe { load() };
        ensure!(status == 0, "cuteafd_steer_rows_load failed with CUDA status {status}");
        Ok(SteerRows { _library: self, steer, device: self.cuda_get_device()? })
    }
}

impl SteerRows<'_> {
    /// The device whose modules [`NativeLibrary::steer_rows`] loaded.
    pub fn device(&self) -> i32 {
        self.device
    }

    /// Steers the rows of `x` whose `vidx` names an editing variant; the
    /// buffers are checked against `shape` and the binding device first.
    ///
    /// # Safety
    /// `stream` is a live stream on the binding device, current when called.
    /// The buffers stay allocated until the stream passes this launch, and
    /// `x` does not overlap the other operands.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(&self, x: CuteafdDeviceBuffer, vidx: CuteafdDeviceBuffer, a: CuteafdDeviceBuffer,
        b: CuteafdDeviceBuffer, k: CuteafdDeviceBuffer, shape: SteerRowsShape, stream: *mut c_void) -> Result<()> {
        shape.check()?;
        for (name, buffer, bytes) in [("x", x, shape.x_bytes()), ("vidx", vidx, shape.vidx_bytes()),
            ("a", a, shape.table_bytes()), ("b", b, shape.table_bytes()), ("k", k, shape.k_bytes())] {
            ensure!(!buffer.ptr.is_null() && buffer.bytes >= bytes,
                "steer_rows {name} holds {} bytes, needs {bytes}", buffer.bytes);
            ensure!(buffer.device_id == self.device, "steer_rows {name} is on device {}, bound to {}",
                buffer.device_id, self.device);
        }
        // SAFETY: shapes and sizes were checked above; the caller's contract covers the rest.
        unsafe { self.launch_raw(x.ptr.cast(), vidx.ptr.cast(), a.ptr.cast(), b.ptr.cast(), k.ptr.cast(), shape, stream) }
    }

    /// [`Self::launch`] on raw device pointers (views into larger arenas).
    /// The shape is still checked; the pointers are not.
    ///
    /// # Safety
    /// As [`Self::launch`], and each pointer addresses at least the bytes
    /// [`SteerRowsShape`] names on the binding device; `x`, `a` and `b`
    /// are 16-byte aligned.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_raw(&self, x: *mut u16, vidx: *const i32, a: *const f32, b: *const f32,
        k: *const i32, shape: SteerRowsShape, stream: *mut c_void) -> Result<()> {
        shape.check()?;
        // SAFETY: the caller's contract; the native entry point re-checks shape and alignment.
        let status = unsafe {
            (self.steer)(x, shape.ld as i64, shape.rows as i32, shape.h as i32, vidx, a, b, k,
                shape.k_max as i32, stream)
        };
        ensure!(status == 0, "cuteafd_steer_rows ({} rows, h {}, k_max {}) failed with CUDA status {status}",
            shape.rows, shape.h, shape.k_max);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_sizes_and_checks() {
        let shape = SteerRowsShape { rows: 3, h: 4096, ld: 4104, k_max: 19, variants: 2 };
        assert_eq!(shape.x_bytes(), (2 * 4104 + 4096) * 2);
        assert_eq!(shape.table_bytes(), 3 * 19 * 4096 * 4);
        assert_eq!(shape.k_bytes(), 12);
        assert!(shape.check().is_ok());
        assert!(SteerRowsShape { h: 4000, ..shape }.check().is_err());
        assert!(SteerRowsShape { variants: 0, ..shape }.check().is_err());
        assert!(SteerRowsShape { variants: 8, ..shape }.check().is_err());
        assert_eq!(SteerRowsShape { rows: 0, ..shape }.x_bytes(), 0);
    }
}

#[cfg(test)]
#[path = "steer_rows_gpu_tests.rs"]
mod gpu_tests;
