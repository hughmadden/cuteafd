use crate::{CuteafdDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
type Launch = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const f32,
    *const f32,
    *const u8,
    *mut f32,
    *mut u32,
    *mut f32,
    i32,
    i32,
    *mut c_void,
) -> i32;
pub struct V41Router<'a> {
    _library: &'a NativeLibrary,
    launch: Launch,
}
impl NativeLibrary {
    pub fn v41_router(&self) -> Result<V41Router<'_>> {
        let initialize = unsafe {
            *self
                .lib
                .get::<unsafe extern "C" fn() -> i32>(b"cuteafd_v41_router_initialize")?
        };
        let status = unsafe { initialize() };
        ensure!(
            status == 0,
            "native router initialization CUDA status {status}"
        );
        Ok(V41Router {
            _library: self,
            launch: unsafe { *self.lib.get::<Launch>(b"cuteafd_v41_router")? },
        })
    }
}
impl V41Router<'_> {
    /// # Safety
    /// Initialized inputs must belong to the stream device, with finite logits
    /// and biases; all buffers must remain live through completion. Outputs must
    /// not overlap inputs or each other. image_mask requires a valid VL bias.
    pub unsafe fn launch(
        &self,
        hidden: CuteafdDeviceBuffer,
        weight: CuteafdDeviceBuffer,
        bias: CuteafdDeviceBuffer,
        bias_vl: CuteafdDeviceBuffer,
        image_mask: Option<CuteafdDeviceBuffer>,
        scores: CuteafdDeviceBuffer,
        ids: CuteafdDeviceBuffer,
        routing: CuteafdDeviceBuffer,
        rows: usize,
        experts: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && matches!(experts, 128 | 384),
            "invalid router geometry"
        );
        let topk = if experts == 128 { 3 } else { 6 };
        let views = [
            (hidden, rows * 10240),
            (weight, experts * 10240),
            (bias, experts * 4),
            (scores, rows * experts * 4),
            (ids, rows * topk * 4),
            (routing, rows * topk * 4),
        ];
        for (view, bytes) in views {
            ensure!(
                !view.ptr.is_null() && view.bytes >= bytes,
                "invalid router buffer"
            );
        }
        if let Some(mask) = image_mask {
            ensure!(
                !mask.ptr.is_null()
                    && mask.bytes >= rows
                    && !bias_vl.ptr.is_null()
                    && bias_vl.bytes >= experts * 4,
                "invalid router image mask or VL bias"
            );
        }
        let status = unsafe {
            (self.launch)(
                hidden.ptr.cast(),
                weight.ptr.cast(),
                bias.ptr.cast(),
                bias_vl.ptr.cast(),
                image_mask.map_or(std::ptr::null(), |m| m.ptr.cast()),
                scores.ptr.cast(),
                ids.ptr.cast(),
                routing.ptr.cast(),
                rows as i32,
                experts as i32,
                stream,
            )
        };
        self._library.status_to_result("native router", status)?;
        Ok(())
    }
}

#[cfg(test)]
mod draft_failure_tests {
    use super::*;
    #[test]
    fn router_native_failure_is_tagged_but_host_guards_are_not() -> Result<()> {
        let library = crate::tests::draft_failure_library()?;
        unsafe extern "C" fn launch(_: *const u16, _: *const u16, _: *const f32, _: *const f32,
            _: *const u8, _: *mut f32, _: *mut u32, _: *mut f32, _: i32, _: i32, _: *mut c_void) -> i32 { 98 }
        let router = V41Router { _library: &library, launch };
        let b = CuteafdDeviceBuffer { ptr: std::ptr::NonNull::<u16>::dangling().as_ptr().cast(),
            bytes: 4 << 20, device_id: 0, flags: 0 };
        // SAFETY: the injected entry point never dereferences synthetic buffers.
        unsafe {
            crate::tests::assert_draft_native_failure(router.launch(b, b, b, b, None, b, b, b, 1, 128, std::ptr::null_mut()));
            let error = router.launch(b, b, b, b, None, b, b, b, 0, 128, std::ptr::null_mut()).unwrap_err();
            assert!(error.downcast_ref::<crate::NativeStatusError>().is_none());
        }
        Ok(())
    }
}
