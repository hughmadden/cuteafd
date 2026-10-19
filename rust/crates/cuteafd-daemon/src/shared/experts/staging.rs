//! Default Spark exchange boundary: device ids/weights/wire to one pinned arena.
//! Callers retain ownership and choose the existing stream/event completion point.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_transport::ExpertProtocolV2RouteEntry;
use std::ffi::c_void;

#[derive(Clone, Copy, Debug)]
pub(crate) struct RouteStaging {
    rows: usize,
    topk: usize,
    pub route_bytes: usize,
    pub wire_bytes: usize,
    pub bytes: usize,
}

impl RouteStaging {
    pub fn new(rows: usize, topk: usize, row_bytes: usize) -> Result<Self> {
        ensure!(rows > 0 && topk > 0 && row_bytes > 0, "Spark staging geometry must be positive");
        ensure!(u32::try_from(rows).is_ok() && u32::try_from(topk).is_ok(),
            "Spark staging geometry exceeds protocol fields");
        let routes = rows.checked_mul(topk).context("Spark staging route count overflow")?;
        ensure!(u32::try_from(routes).is_ok(), "Spark staging route offsets exceed protocol fields");
        let route_bytes = routes.checked_mul(4).context("Spark staging route size overflow")?;
        let wire_bytes = rows.checked_mul(row_bytes).context("Spark staging wire size overflow")?;
        let bytes = route_bytes.checked_mul(2).and_then(|n| n.checked_add(wire_bytes))
            .context("Spark staging size overflow")?;
        Ok(Self { rows, topk, route_bytes, wire_bytes, bytes })
    }

    fn regions(&self, host: CuteafdHostBuffer, sources: [CuteafdDeviceBuffer; 3])
        -> Result<[(CuteafdHostBuffer, CuteafdDeviceBuffer, usize); 3]> {
        ensure!(!host.ptr.is_null() && host.bytes >= self.bytes, "Spark pinned staging is short or null");
        ensure!((host.ptr as usize).checked_add(self.bytes).is_some(), "Spark staging address overflow");
        let device = sources[0].device_id;
        let sizes = [self.route_bytes, self.route_bytes, self.wire_bytes];
        for (source, bytes) in sources.iter().zip(sizes) {
            ensure!(device >= 0 && source.device_id == device, "Spark staging sources must share a device");
            ensure!(!source.ptr.is_null() && source.bytes >= bytes, "Spark staging source is short or null");
            ensure!((source.ptr as usize).checked_add(bytes).is_some(), "Spark source address overflow");
        }
        let offsets = [0, self.route_bytes, 2 * self.route_bytes];
        Ok(std::array::from_fn(|i| (CuteafdHostBuffer {
            ptr: host.ptr.cast::<u8>().wrapping_add(offsets[i]).cast(),
            bytes: host.bytes - offsets[i],
            ..host
        }, sources[i], sizes[i])))
    }

    /// Enqueue ids, weights, then wire, without synchronizing or recording events.
    ///
    /// # Safety
    /// All handles name live allocations on the stream's current device. The
    /// pinned arena remains exclusively owned until the caller's completion
    /// point, and all storage survives queued work, including a partial failure.
    pub unsafe fn enqueue(&self, library: &NativeLibrary, host: CuteafdHostBuffer,
        sources: [CuteafdDeviceBuffer; 3], stream: *mut c_void) -> Result<()> {
        let regions = self.regions(host, sources)?;
        for (destination, source, bytes) in regions {
            // SAFETY: regions validates all extents before the first enqueue;
            // allocation liveness and stream ordering are the caller's contract.
            unsafe { library.copy_d2h_host_buffer_async(destination, source, bytes, stream)?; }
        }
        Ok(())
    }

    pub fn routes(&self, staged: &[u8]) -> Result<Vec<ExpertProtocolV2RouteEntry>> {
        ensure!(staged.len() >= self.bytes, "staged routes and wire rows are short");
        let word = |offset: usize, i: usize| u32::from_le_bytes(staged[offset + i * 4..][..4].try_into().unwrap());
        Ok((0..self.rows * self.topk).map(|i| ExpertProtocolV2RouteEntry {
            row_index: (i / self.topk) as u32,
            expert_id: word(0, i),
            gate_weight: f32::from_bits(word(self.route_bytes, i)),
        }).collect())
    }

    pub fn wire<'a>(&self, staged: &'a [u8]) -> Result<&'a [u8]> {
        ensure!(staged.len() >= self.bytes, "staged routes and wire rows are short");
        Ok(&staged[2 * self.route_bytes..self.bytes])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_arena_offsets_copy_order_and_route_bits_are_unchanged() -> Result<()> {
        for (hidden, topk) in [(7168, 8), (6144, 8), (3584, 16)] {
            for rows in [1, 16, 4096] {
                for row_bytes in [hidden + hidden / 32, hidden * 2] {
                    let layout = RouteStaging::new(rows, topk, row_bytes)?;
                    let host = CuteafdHostBuffer { ptr: 0x1000usize as *mut c_void, bytes: layout.bytes + 64, flags: 7 };
                    let sources = std::array::from_fn(|i| CuteafdDeviceBuffer {
                        ptr: (0x2000usize + i * 0x1000) as *mut c_void,
                        bytes: if i < 2 { rows * topk * 4 } else { rows * row_bytes },
                        device_id: 1, flags: 0,
                    });
                    let regions = layout.regions(host, sources)?;
                    for i in 0..3 {
                        let offset = i * rows * topk * 4;
                        assert_eq!(regions[i].0.ptr as usize, host.ptr as usize + offset);
                        assert_eq!(regions[i].0.bytes, host.bytes - offset);
                        assert_eq!(regions[i].0.flags, host.flags);
                        assert_eq!(regions[i].1.ptr, sources[i].ptr);
                        assert_eq!(regions[i].2, sources[i].bytes);
                    }
                    let mut bytes = Vec::new();
                    for i in 0..rows * topk { bytes.extend_from_slice(&((i % 896) as u32).to_le_bytes()); }
                    for i in 0..rows * topk {
                        // Preserve signed zero and payload bits as well as ordinary weights.
                        bytes.extend_from_slice(&[0x80000000u32, 0x3f000000, 0x7fc00001][i % 3].to_le_bytes());
                    }
                    let wire: Vec<_> = (0..rows * row_bytes).map(|i| i as u8).collect();
                    bytes.extend_from_slice(&wire);
                    let routes = layout.routes(&bytes)?;
                    for (i, route) in routes.iter().enumerate() {
                        assert_eq!(route.row_index, (i / topk) as u32);
                        assert_eq!(route.expert_id, (i % 896) as u32);
                        assert_eq!(route.gate_weight.to_bits(), [0x80000000, 0x3f000000, 0x7fc00001][i % 3]);
                    }
                    assert_eq!(layout.wire(&bytes)?, wire);
                    assert!(layout.routes(&bytes[..bytes.len() - 1]).is_err());
                    assert!(layout.wire(&bytes[..bytes.len() - 1]).is_err());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn all_copy_extents_are_checked_before_enqueue() -> Result<()> {
        for geometry in [(0, 8, 4), (1, 0, 4), (1, 8, 0), (usize::MAX, 8, 4),
            (u32::MAX as usize, 2, 4), (1, 8, usize::MAX)] {
            assert!(RouteStaging::new(geometry.0, geometry.1, geometry.2).is_err());
        }
        let layout = RouteStaging::new(16, 16, 3696)?;
        let host = CuteafdHostBuffer { ptr: 0x1000usize as *mut c_void, bytes: layout.bytes, flags: 0 };
        let sources = std::array::from_fn(|i| CuteafdDeviceBuffer {
            ptr: (0x2000usize + i * 0x1000) as *mut c_void,
            bytes: if i < 2 { layout.route_bytes } else { layout.wire_bytes }, device_id: 0, flags: 0,
        });
        assert!(layout.regions(CuteafdHostBuffer { bytes: layout.bytes - 1, ..host }, sources).is_err());
        assert!(layout.regions(CuteafdHostBuffer { ptr: std::ptr::null_mut(), ..host }, sources).is_err());
        assert!(layout.regions(CuteafdHostBuffer { ptr: usize::MAX as *mut c_void, ..host }, sources).is_err());
        for i in 0..3 {
            for invalid in [CuteafdDeviceBuffer { bytes: sources[i].bytes - 1, ..sources[i] },
                CuteafdDeviceBuffer { ptr: std::ptr::null_mut(), ..sources[i] },
                CuteafdDeviceBuffer { device_id: 1, ..sources[i] },
                CuteafdDeviceBuffer { ptr: usize::MAX as *mut c_void, ..sources[i] }] {
                let mut bad = sources;
                bad[i] = invalid;
                assert!(layout.regions(host, bad).is_err());
            }
        }
        Ok(())
    }
}
