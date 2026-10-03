//! CUDA gates for testing cancellation against genuinely queued work.
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use libloading::Library;
use std::ffi::c_void;
use std::sync::{atomic::{AtomicBool, Ordering}, Arc};

/// Holds one CUDA stream until a separate thread releases it. Drop also
/// releases and drains, so an assertion failure cannot strand the callback.
pub struct CudaStreamGate<'a> {
    native: &'a NativeLibrary,
    stream: *mut c_void,
    release: Arc<AtomicBool>,
    _runtime: Library,
}
impl<'a> CudaStreamGate<'a> {
    /// # Safety
    /// `stream` belongs to the loaded CUDA runtime and remains live until this
    /// gate is dropped. The caller must not synchronize it before releasing the
    /// gate, except when an independent thread will release it.
    pub unsafe fn new(native: &'a NativeLibrary, stream: *mut c_void) -> Result<Self> {
        // SAFETY: load the matching runtime; it remains loaded through callback
        // completion, and the symbol uses CUDA's declared C ABI.
        let runtime = unsafe { Library::new("libcudart.so.13")? };
        type Launch = unsafe extern "C" fn(*mut c_void, unsafe extern "C" fn(*mut c_void), *mut c_void) -> i32;
        // SAFETY: cudaLaunchHostFunc has the signature above.
        let launch = unsafe { *runtime.get::<Launch>(b"cudaLaunchHostFunc\0")? };
        let release = Arc::new(AtomicBool::new(false));
        let data = Arc::into_raw(release.clone()).cast_mut().cast();
        // SAFETY: the callback owns this Arc reference; it uses no CUDA APIs.
        let status = unsafe { launch(stream, hold, data) };
        if status != 0 {
            // SAFETY: failed submission never acquired the callback reference.
            drop(unsafe { Arc::from_raw(data.cast::<AtomicBool>()) });
        }
        ensure!(status == 0, "cudaLaunchHostFunc failed: {status}");
        Ok(Self { native, stream, release, _runtime: runtime })
    }

    pub fn release_handle(&self) -> Arc<AtomicBool> { self.release.clone() }
}
impl Drop for CudaStreamGate<'_> {
    fn drop(&mut self) {
        self.release.store(true, Ordering::Release);
        // SAFETY: the constructor requires the stream to outlive the gate.
        if let Err(error) = unsafe { self.native.cuda_stream_synchronize(self.stream) } {
            eprintln!("draining CUDA test gate: {error:#}");
        }
    }
}

unsafe extern "C" fn hold(data: *mut c_void) {
    // SAFETY: successful submission transferred exactly one Arc reference here.
    let release = unsafe { Arc::from_raw(data.cast::<AtomicBool>()) };
    while !release.load(Ordering::Acquire) { std::thread::yield_now(); }
}
