//! Executed-route tallies whose real rows are device data. A step copies one
//! `[prefill rows, decode rows]` pair from an immutable pinned table into its
//! workspace, outside capture; the canonical router's tally reads its phase's
//! entry when the launch runs, eager or replayed from a captured graph. Startup
//! work, masked graph warm-up and graph padding upload zero rows, so they count
//! nothing, and no graph is captured again when the rows change.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::expert_usage::{ExpertUsage, UsagePhase};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_loader::placement::expert_usage::{REPLAY_LOOKUP_BYTES, REPLAY_MAX_ROWS, REPLAY_PAIR_BYTES};
use std::ffi::c_void;

/// Every pair a step can upload: the prefill pairs `[rows, 0]`, then the
/// decode pairs `[0, rows]`, for 0..=[`REPLAY_MAX_ROWS`] rows each.
pub(crate) fn lookup() -> Vec<i32> {
    [false, true].into_iter().flat_map(|decode| (0..=REPLAY_MAX_ROWS).flat_map(move |rows| {
        if decode { [0, rows as i32] } else { [rows as i32, 0] }
    })).collect()
}

/// Byte offset of `rows` real rows of one phase in [`lookup`].
pub(crate) fn lookup_offset(decode: bool, rows: usize) -> Result<usize> {
    ensure!(rows <= REPLAY_MAX_ROWS, "expert usage rows {rows} exceed the replay table ({REPLAY_MAX_ROWS})");
    Ok((usize::from(decode) * (REPLAY_MAX_ROWS + 1) + rows) * REPLAY_PAIR_BYTES)
}

/// One engine's immutable pinned pairs. Uploads from it never reuse a pinned
/// source and never wait on the host.
pub(crate) struct ReplayLookup<'a> {
    pairs: crate::shared::memory::HostAllocation<'a>,
}

impl<'a> ReplayLookup<'a> {
    /// Allocates and fills the table under ledger owner `owner`.
    pub(crate) fn new(library: &'a NativeLibrary, owner: &str) -> Result<Self> {
        let _owner = cuteafd_ffi::memory_ledger::owner(owner);
        let pairs = crate::shared::memory::HostAllocation::new(library, REPLAY_LOOKUP_BYTES)?;
        let values = lookup();
        ensure!(values.len() * 4 == REPLAY_LOOKUP_BYTES, "expert usage replay table extent");
        // SAFETY: the fresh pinned owner is private and holds exactly these values;
        // every later reader sees immutable pairs.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), pairs.buffer.ptr.cast::<i32>(), values.len()); }
        Ok(Self { pairs })
    }

    /// Queues the copy of `rows` real rows of one phase into `pair`.
    ///
    /// # Safety
    /// `pair` is a live device pair of at least [`REPLAY_PAIR_BYTES`] on the current
    /// device; its readers are ordered on `stream`; the call is outside capture.
    pub(crate) unsafe fn put(&self, pair: CuteafdDeviceBuffer, decode: bool, rows: usize, stream: *mut c_void)
        -> Result<()> {
        ensure!(pair.bytes >= REPLAY_PAIR_BYTES, "expert usage replay pair is too small");
        let offset = lookup_offset(decode, rows)?;
        let source = CuteafdHostBuffer { ptr: self.pairs.buffer.ptr.cast::<u8>().wrapping_add(offset).cast(),
            bytes: REPLAY_PAIR_BYTES, ..self.pairs.buffer };
        // SAFETY: the immutable table and the caller's pair outlive the stream's consumers.
        unsafe { self.pairs.library.copy_host_buffer_h2d_async(pair, source, REPLAY_PAIR_BYTES, stream) }
    }
}

/// Tallies `rows` selected rows of `ids` for `layer`, bounded at launch by the
/// phase's entry of `pair`. Only the canonical router (rank 0) counts: replica
/// routers and disabled telemetry return before touching any operand or CUDA.
///
/// # Safety
/// `ids` holds `rows * topk` u32 ids ordered before this call on `stream`, on
/// `device`; `ids`, `pair` and `stream` stay live through the retirement of any
/// graph that captures this launch.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn record(usage: Option<&ExpertUsage<'_>>, device: i32, rank: usize, layer: usize, decode: bool,
    ids: *const c_void, rows: usize, pair: Option<*const i32>, stream: *mut c_void) -> Result<()> {
    if rank != 0 { return Ok(()); }
    let Some(usage) = usage else { return Ok(()); };
    let pair = pair.context("canonical expert usage replay pair missing")?;
    let phase = if decode { UsagePhase::Decode } else { UsagePhase::Prefill };
    // SAFETY: the pair holds both phase entries; the caller retains ids, pair and stream.
    unsafe { usage.routes(device, layer, phase)?.record(ids, rows, Some(pair.add(usize::from(decode))),
        u32::MAX, true, stream) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_covers_every_phase_and_width_including_zero() {
        let pairs = lookup();
        assert_eq!(pairs.len() * 4, REPLAY_LOOKUP_BYTES);
        for decode in [false, true] {
            for rows in 0..=REPLAY_MAX_ROWS {
                let first = lookup_offset(decode, rows).unwrap() / 4;
                assert_eq!(&pairs[first..first + 2], if decode { [0, rows as i32] } else { [rows as i32, 0] });
                // A tally reads its own phase's entry: the rows, never the other phase's zero.
                assert_eq!(pairs[first + usize::from(decode)], rows as i32);
                assert_eq!(pairs[first + usize::from(!decode)], 0);
            }
        }
        assert!(lookup_offset(false, REPLAY_MAX_ROWS + 1).is_err());
        assert!(lookup_offset(true, REPLAY_MAX_ROWS + 1).is_err());
    }

    #[test]
    fn disabled_and_replica_routes_do_not_touch_operands_or_cuda() {
        let captures = cuteafd_ffi::graph_capture_sites();
        for (rank, decode) in [(0, false), (0, true), (1, false), (1, true)] {
            // SAFETY: the disabled and replica paths must not inspect any operand or query CUDA.
            unsafe { record(None, -1, rank, usize::MAX, decode, std::ptr::null(), usize::MAX, None,
                std::ptr::null_mut()).unwrap(); }
        }
        assert_eq!(captures, cuteafd_ffi::graph_capture_sites());
    }
}
