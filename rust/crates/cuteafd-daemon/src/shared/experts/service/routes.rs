//! Packed pinned route planes retained independently through each unit's fence.
use crate::shared::memory::HostAllocation;
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_transport::expert::BackboneRequest;
use std::ffi::c_void;

pub(in crate::shared::experts) struct RouteStaging<'a> {
    slots: Vec<HostAllocation<'a>>,
    plane_bytes: usize,
}

impl<'a> RouteStaging<'a> {
    pub(in crate::shared::experts) fn new(library: &'a NativeLibrary, units: usize,
        rows: usize) -> Result<Self> {
        let mut staging = Self { slots: Vec::new(), plane_bytes: 0 };
        staging.resize(library, units, rows)?;
        Ok(staging)
    }

    pub(in crate::shared::experts) fn resize(&mut self, library: &'a NativeLibrary,
        units: usize, rows: usize) -> Result<()> {
        ensure!(units > 0 && rows > 0, "route staging requires positive units and rows");
        self.plane_bytes = rows.checked_mul(cuteafd_core::expert_geometry().topk as usize)
            .and_then(|routes| routes.checked_mul(4)).context("route staging extent overflow")?;
        let bytes = self.plane_bytes.checked_mul(2).context("route staging size overflow")?;
        self.slots.clear();
        for _ in 0..units {
            self.slots.push(HostAllocation::named(library, bytes, "experts/workspace/route_staging")?);
        }
        Ok(())
    }

    pub(in crate::shared::experts) fn pack(&mut self, slot: usize,
        request: &BackboneRequest<'_>) -> Result<()> {
        ensure!(cfg!(target_endian = "little"), "native routes require little-endian storage");
        let count = request.rows() as usize * cuteafd_core::expert_geometry().topk as usize;
        ensure!(count * 4 <= self.plane_bytes, "route staging exceeds admitted rows");
        let base = self.slots.get_mut(slot).context("route staging slot outside owner queue")?.buffer.ptr.cast::<u8>();
        // SAFETY: pinned storage is aligned, initialized, exclusively borrowed,
        // and the two typed planes are disjoint and within its admitted extent.
        unsafe {
            request.copy_routes_into(std::slice::from_raw_parts_mut(base.cast::<i32>(), count),
                std::slice::from_raw_parts_mut(base.add(self.plane_bytes).cast::<f32>(), count))?;
        }
        Ok(())
    }

    /// # Safety
    /// The unit's slot and both destinations remain owned until its stream
    /// fence completes; the slot must not be packed again before that fence.
    pub(in crate::shared::experts) unsafe fn upload(&self, slot: usize, first: usize,
        count: usize, ids: CuteafdDeviceBuffer, weights: CuteafdDeviceBuffer,
        stream: *mut c_void) -> Result<()> {
        let (offset, bytes) = plane_range(self.plane_bytes, first, count)?;
        let staging = self.slots.get(slot).context("route staging slot outside owner queue")?;
        let source = staging.buffer;
        // SAFETY: plane_range validates both subranges before any work is queued.
        let sources = unsafe { [CuteafdHostBuffer { ptr: source.ptr.cast::<u8>().add(offset).cast(), bytes, ..source },
            CuteafdHostBuffer { ptr: source.ptr.cast::<u8>().add(self.plane_bytes + offset).cast(), bytes, ..source }] };
        // SAFETY: caller retains the packed slot and exclusive destinations through the fence.
        unsafe { staging.library.copy_host_buffers_h2d_batch_async(&[ids, weights],
            &sources, &[bytes, bytes], stream)?; }
        Ok(())
    }
}

fn plane_range(plane_bytes: usize, first: usize, count: usize) -> Result<(usize, usize)> {
    let offset = first.checked_mul(4).context("route staging offset overflow")?;
    let bytes = count.checked_mul(4).context("route staging copy extent overflow")?;
    ensure!(count > 0 && offset.checked_add(bytes).is_some_and(|end| end <= plane_bytes),
        "route staging copy exceeds admitted plane");
    Ok((offset, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_routes_copy_contiguous_chunk_planes_with_checked_extents() {
        assert_eq!(plane_range(32, 0, 8).unwrap(), (0, 32));
        assert_eq!(plane_range(32, 3, 5).unwrap(), (12, 20));
        for (first, count) in [(0, 0), (0, 9), (7, 2), (usize::MAX, 1), (1, usize::MAX)] {
            assert!(plane_range(32, first, count).is_err());
        }
    }

    #[test]
    fn spark_workers_never_deinterleave_routes_with_strided_mapped_copies() {
        for source in [include_str!("../execution.rs"), include_str!("../exl3/worker.rs"),
            include_str!("../fp8/worker.rs"), include_str!("exlr8.rs")] {
            assert!(!source.contains("routes_view") && !source.contains("mapped.routes"));
            assert!(!source.contains("copy_d2d_2d_async"));
        }
    }
}
