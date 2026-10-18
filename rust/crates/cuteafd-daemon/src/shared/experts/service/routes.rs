//! Device-only deinterleave of canonical route records into admitted input planes.
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::{expert::BackboneRequest, EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN};
use std::ffi::c_void;

#[derive(Debug)]
struct RouteCopies {
    sources: [CuteafdDeviceBuffer; 2],
    destinations: [CuteafdDeviceBuffer; 2],
    routes: usize,
}
impl RouteCopies {
    fn plan(source: CuteafdDeviceBuffer, ids: CuteafdDeviceBuffer,
        weights: CuteafdDeviceBuffer, routes: usize) -> Result<Self> {
        ensure!(cfg!(target_endian = "little"), "native routes require little-endian storage");
        ensure!(routes > 0, "device route copy requires nonempty routes");
        let source_bytes = routes.checked_mul(EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN)
            .context("device route source extent overflow")?;
        let destination_bytes = routes.checked_mul(4).context("device route plane extent overflow")?;
        ensure!(source.bytes == source_bytes, "mapped route records differ from canonical route count");
        ensure!(!source.ptr.is_null(), "null mapped route records");
        let source_start = source.ptr as usize;
        let source_end = source_start.checked_add(source_bytes).context("route source address overflow")?;
        let mut destinations = [ids, weights];
        let mut ranges = [(0, 0); 2];
        for (index, destination) in destinations.iter_mut().enumerate() {
            ensure!(!destination.ptr.is_null() && destination.bytes >= destination_bytes,
                "admitted device route plane is too small or null");
            ensure!(destination.device_id == source.device_id,
                "mapped route records and destination must share a device");
            let start = destination.ptr as usize;
            let end = start.checked_add(destination_bytes).context("route destination address overflow")?;
            ensure!(end <= source_start || source_end <= start, "device route source overlaps destination");
            ranges[index] = (start, end);
            destination.bytes = destination_bytes;
        }
        ensure!(ranges[0].1 <= ranges[1].0 || ranges[1].1 <= ranges[0].0,
            "device route destination planes overlap");
        let field = |offset: usize| CuteafdDeviceBuffer {
            ptr: (source_start + offset) as *mut c_void,
            bytes: source_bytes - offset,
            ..source
        };
        Ok(Self { sources: [field(4), field(8)], destinations, routes })
    }
}

/// Copies only expert IDs and F32 gates. Canonical validation already guarantees
/// ascending token-major row indices and a fixed top-k; no host repack is needed.
///
/// # Safety
/// The mapped receive frame and exclusive destinations stay live until `stream`
/// drains. Source arrival must precede this stream's reads. On any launch error,
/// the caller still drains previously queued copies before releasing the frame.
pub(super) unsafe fn copy_mapped_routes(library: &NativeLibrary,
    request: &BackboneRequest<'_>, source: CuteafdDeviceBuffer,
    ids: CuteafdDeviceBuffer, weights: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
    ensure!(!request.is_native_group(), "ownership-tagged routes need an ownership-aware device adapter");
    let count = (request.rows() as usize).checked_mul(cuteafd_core::expert_geometry().topk as usize)
        .context("canonical route count overflow")?;
    let plan = RouteCopies::plan(source, ids, weights, count)?;
    for index in 0..2 {
        // SAFETY: planning validates both source and destination extents before
        // queuing anything; the caller retains their owners through the drain.
        // The local 2D ABI accepts device-visible mapped-host aliases; the peer
        // row-copy ABI deliberately accepts only device-allocation flags.
        unsafe { library.copy_d2d_2d_async(plan.destinations[index], 4, plan.sources[index],
            EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN, 4, plan.routes, stream)?; }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn buffer(address: usize, bytes: usize) -> CuteafdDeviceBuffer {
        CuteafdDeviceBuffer { ptr: address as *mut c_void, bytes, device_id: 1, flags: 0 }
    }
    #[test]
    fn mapped_routes_select_wire_fields_without_host_repacking() {
        for count in [1, 8, 512, 32768] {
            let plan = RouteCopies::plan(buffer(0x100000, count * 12),
                buffer(0x200000, count * 4), buffer(0x300000, count * 4), count).unwrap();
            assert_eq!(plan.sources[0].ptr as usize, 0x100004);
            assert_eq!(plan.sources[1].ptr as usize, 0x100008);
            for (index, offset) in [4, 8].into_iter().enumerate() {
                assert_eq!(plan.sources[index].bytes, count * 12 - offset);
                assert_eq!((count - 1) * 12 + 4, plan.sources[index].bytes - (8 - offset));
                assert_eq!(plan.destinations[index].bytes, count * 4);
            }
            assert_eq!(plan.routes, count);
        }
    }
    #[test]
    fn mapped_routes_refuse_bad_extents_devices_and_aliases_before_copy() {
        let source = buffer(0x1000, 96);
        let ids = buffer(0x2000, 32);
        let weights = buffer(0x3000, 32);
        for bad in [buffer(0x1000, 95), buffer(0x1000, 108), buffer(0, 96),
            buffer(usize::MAX - 32, 96)] {
            assert!(RouteCopies::plan(bad, ids, weights, 8).is_err());
        }
        for bad in [buffer(0x2000, 31), buffer(0, 32), buffer(0x1004, 32),
            buffer(0x3004, 32), buffer(usize::MAX - 16, 32),
            CuteafdDeviceBuffer { device_id: 2, ..ids }] {
            assert!(RouteCopies::plan(source, bad, weights, 8).is_err());
        }
        assert!(RouteCopies::plan(source, ids, weights, 0).is_err());
        assert!(RouteCopies::plan(source, ids, weights, usize::MAX).is_err());
    }
}
