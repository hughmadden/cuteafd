//! Exact 12-bit copies of BF16 vocabulary heads (`native/shared/cuda/packed_head.cu`; format in
//! `cuteafd_core::packed_head`): packed and verified at load from the resident BF16 head, then
//! applied by a tensor-core GEMV that reads the BF16 values exactly (both planes and the exception
//! table, FP32 sums) or, for drafters, the coarse plane alone (the traffic of an E4M3 copy).
use crate::shared::memory::DeviceAllocation;
use anyhow::{Context, Result};
use cuteafd_core::packed_head::{PackStatus, PackedHeadLayout, Region, STATUS_WORDS};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, PackedHeadPlanes};
use std::ffi::c_void;

/// An `[n, k]` BF16 head in the exact 12-bit format: one allocation laid out by
/// [`PackedHeadLayout`] (planes, bases, exception table and the pack's status).
pub(crate) struct PackedHead<'a> {
    storage: DeviceAllocation<'a>,
    layout: PackedHeadLayout,
    status: PackStatus,
    pub n: usize,
    pub k: usize,
}

/// The native `planes` array of a head stored at `storage`.
fn planes(storage: CuteafdDeviceBuffer, layout: &PackedHeadLayout) -> PackedHeadPlanes {
    [Region::Coarse, Region::Fine, Region::Base, Region::RowPtr, Region::Columns, Region::Values]
        .map(|region| storage.ptr.wrapping_byte_add(layout.offset(region)))
}

fn status_buffer(storage: CuteafdDeviceBuffer, layout: &PackedHeadLayout) -> CuteafdDeviceBuffer {
    CuteafdDeviceBuffer { ptr: storage.ptr.wrapping_byte_add(layout.offset(Region::Status)), bytes: STATUS_WORDS * 8,
        ..storage }
}

impl<'a> PackedHead<'a> {
    /// Packs the live BF16 weight `w` [n, k] on `stream`, drains it and checks the pack's own
    /// bit-for-bit verification. A failed drain quarantines the destination and the native
    /// module ([`NativeLibrary::is_quarantined_after_failed_drain`]); the caller must then also
    /// retain the source.
    pub fn pack(library: &'a NativeLibrary, w: *const c_void, n: usize, k: usize, stream: *mut c_void)
        -> Result<Self> {
        let layout = PackedHeadLayout::new(n, k)?;
        let storage = DeviceAllocation::new(library, layout.total_bytes())?;
        let status = status_buffer(storage.buffer, &layout);
        // SAFETY: `w` is a live [n, k] BF16 weight; the planes and status are regions of `storage`,
        // sized by `layout` for this shape and `layout.capacity` table entries.
        let launched = unsafe { library.packed_head_pack(w, &planes(storage.buffer, &layout), layout.capacity, n, k,
            status.ptr, stream) };
        // SAFETY: the source and `storage` stay owned until this drain proves the pack retired; a
        // launch error may follow queued work, so drain on that path too.
        if let Err(drain) = unsafe { library.cuda_stream_synchronize(stream) } {
            // Completion is unknown: retain the destination until process teardown.
            library.quarantine_module_after_failed_drain();
            std::mem::forget(storage);
            let error = launched.err().unwrap_or_else(|| anyhow::anyhow!("packed head [{n}, {k}] pack"));
            return Err(error.context(format!(
                "packed head destination and native module quarantined after failed drain: {drain}")));
        }
        launched?;
        let mut words = [0u8; STATUS_WORDS * 8];
        library.copy_d2h(&mut words, status)?;
        let status = PackStatus::from_bytes(&words);
        status.check(layout.capacity).with_context(|| format!("packed head [{n}, {k}]"))?;
        Ok(Self { storage, layout, status, n, k })
    }

    /// Device bytes of the head (the planner's [`PackedHeadLayout::bytes`]).
    pub fn bytes(&self) -> usize { self.layout.total_bytes() }

    /// Values outside their rows' windows, and the most in any row.
    pub fn exceptions(&self) -> (u64, u64) { (self.status.exceptions, self.status.max_row_exceptions) }

    /// `out` [rows, n] (BF16, or FP32 with `out_f32`) = `x` [rows, k] @ W^T: the exact BF16
    /// weights, or with `coarse` the coarse plane alone (mantissa bits 3..0 at their midpoint).
    ///
    /// # Safety
    /// `x` and `out` are live device buffers of those shapes; `scratch` was sized by [`scratch`]
    /// for at least `rows` rows of this head.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn apply(&self, library: &NativeLibrary, x: *const c_void, out: *mut c_void, out_f32: bool, rows: usize,
        coarse: bool, scratch: &DeviceAllocation<'_>, stream: *mut c_void) -> Result<()> {
        // SAFETY: the caller's contract; the planes are this head's live regions.
        unsafe { library.packed_head_linear(x, &planes(self.storage.buffer, &self.layout), out, out_f32, rows, self.k,
            self.n, coarse, scratch.buffer.ptr, scratch.buffer.bytes, stream) }
    }
}

/// GEMV scratch for up to `rows` rows of every `(k, n)` head shape.
pub(crate) fn scratch<'a>(library: &'a NativeLibrary, rows: usize, shapes: &[(usize, usize)])
    -> Result<DeviceAllocation<'a>> {
    let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace/packed-head");
    let bytes = shapes.iter().map(|&(k, n)| library.packed_head_workspace(rows, k, n))
        .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
    DeviceAllocation::new(library, bytes.max(256))
}

#[cfg(test)]
impl PackedHead<'_> {
    pub(crate) fn visit(&self, prefix: &str, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        f(format!("{prefix}packed12"), self.storage.buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The planes array is the layout's first six regions, in the native order, and the
    /// status words follow the table.
    #[test]
    fn planes_follow_the_layout_regions() {
        let layout = PackedHeadLayout::new(64_640, 5120).unwrap();
        let storage = CuteafdDeviceBuffer { ptr: 0x1000_0000usize as *mut c_void, bytes: layout.total_bytes(),
            ..Default::default() };
        let planes = planes(storage, &layout);
        let offsets = planes.map(|p| p as usize - 0x1000_0000);
        assert_eq!(offsets, [Region::Coarse, Region::Fine, Region::Base, Region::RowPtr, Region::Columns,
            Region::Values].map(|region| layout.offset(region)));
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], 64_640 * 5120);
        let status = status_buffer(storage, &layout);
        assert_eq!(status.ptr as usize - 0x1000_0000, layout.offset(Region::Status));
        assert_eq!(status.bytes, 32);
        assert!(layout.offset(Region::Status) + 32 <= layout.total_bytes());
    }
}
