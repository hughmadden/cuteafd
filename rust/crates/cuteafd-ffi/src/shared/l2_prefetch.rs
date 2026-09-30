//! L2 prefetch of weight ranges (`native/shared/cuda/l2_prefetch.cu`).
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

impl NativeLibrary {
    /// The current device's L2 bytes (0 when the query fails).
    pub fn l2_cache_bytes(&self) -> Result<usize> {
        type F = unsafe extern "C" fn() -> i64;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_l2_cache_bytes") }?;
        Ok(usize::try_from(unsafe { f() }).unwrap_or(0))
    }

    /// The current device's multiprocessor count.
    pub fn sm_count(&self) -> Result<usize> {
        type F = unsafe extern "C" fn() -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_sm_count") }?;
        let n = unsafe { f() };
        ensure!(n > 0, "multiprocessor count query failed");
        Ok(n as usize)
    }

    /// Touches one byte of every `stride` bytes of each range (at most 16)
    /// through L2 from `blocks` blocks on `stream`; writes nothing.
    ///
    /// # Safety
    /// Every range is live device memory on the stream's device.
    pub unsafe fn l2_prefetch(&self, ranges: &[(*const c_void, usize)], stride: usize, blocks: usize,
        stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*const *const c_void, *const i64, i32, i32, i32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_l2_prefetch") }?;
        let ptrs: Vec<*const c_void> = ranges.iter().map(|r| r.0).collect();
        let bytes: Vec<i64> = ranges.iter().map(|r| r.1 as i64).collect();
        let status = unsafe { f(ptrs.as_ptr(), bytes.as_ptr(), i32::try_from(ranges.len())?, i32::try_from(stride)?,
            i32::try_from(blocks)?, stream) };
        ensure!(status == 0, "L2 prefetch of {} ranges failed with {status}", ranges.len());
        Ok(())
    }
}
