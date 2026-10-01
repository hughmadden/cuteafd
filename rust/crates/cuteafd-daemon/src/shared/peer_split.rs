//! Two-GPU head split plumbing shared by the families that split attention
//! heads over the coordinator's two GPUs (MiMo V2.6 Pro, GLM 5.x): the ranks'
//! devices and streams, and both ends of the exchange between them.
//!
//! Exchange: SM pushes into the other GPU's receive buffers with a release
//! flag, and a spinning acquire wait on the receiving stream (the
//! `peer_exchange` kernels; device-side sequence numbers, so graphs replay
//! them). Every slot has its own flag and sequences: each push into a slot is
//! matched, in order, by one wait on that slot on the other GPU, whatever the
//! interleaving of slots (prefill lanes); the host queues a push before it
//! waits on either stream. Callers pick slots by layer parity (and lane), so a
//! push never lands on rows the other GPU may still read. One extra flag
//! (`DIRECT`) serves pushes into buffers outside the slots.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;

/// One GPU of a head split: its device and the stream its work runs on.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RankDevice {
    pub device: i32,
    pub stream: *mut c_void,
}

/// Runs `body` with `device` current, then `home` again (whatever `body` returned).
pub(crate) fn on_device<T>(library: &NativeLibrary, device: i32, home: i32, body: impl FnOnce() -> Result<T>)
    -> Result<T> {
    if device == home {
        return body();
    }
    library.cuda_set_device(device)?;
    let out = body();
    library.cuda_set_device(home)?;
    out
}

/// One GPU's end: receive buffers and, per flag (slot, plus [`DIRECT`]), four
/// control words: `[0]` the sequence the other GPU publishes, `[1..3]` this
/// GPU's send state, `[3]` its receive state.
struct End<'a> {
    recv: Vec<DeviceAllocation<'a>>,
    control: DeviceAllocation<'a>,
}

impl End<'_> {
    fn word(&self, flag: usize, word: usize) -> *mut u32 {
        debug_assert!((flag * 4 + word) * 4 < self.control.buffer.bytes);
        // SAFETY: four words per flag inside the zeroed control allocation (sized in `new`).
        unsafe { self.control.buffer.ptr.cast::<u32>().add(flag * 4 + word) }
    }

    fn flag(&self, flag: usize) -> *mut u32 {
        self.word(flag, 0)
    }

    fn send_state(&self, flag: usize) -> *mut u32 {
        self.word(flag, 1)
    }

    fn recv_state(&self, flag: usize) -> *mut u32 {
        self.word(flag, 3)
    }
}

/// The flag of pushes into buffers outside the receive slots.
pub(crate) const DIRECT: usize = usize::MAX;

/// Both ends of the exchange between ranks 0 and 1.
pub(crate) struct PeerExchange<'a> {
    library: &'a NativeLibrary,
    ranks: [RankDevice; 2],
    ends: [End<'a>; 2],
}

impl<'a> PeerExchange<'a> {
    /// Enables peer access both ways and allocates `slots` zeroed receive
    /// buffers of `slot_bytes` and the control words on each rank; `ranks[0]`'s
    /// device must be current (and is again on return).
    pub fn new(library: &'a NativeLibrary, ranks: [RankDevice; 2], slots: usize, slot_bytes: usize) -> Result<Self> {
        ensure!(ranks[0].device != ranks[1].device && slots > 0 && slot_bytes % 16 == 0,
            "a head split exchanges between two GPUs in 16-byte rows");
        let end = |rank: usize| -> Result<End<'a>> {
            on_device(library, ranks[rank].device, ranks[0].device, || {
                library.cuda_enable_peer(ranks[1 - rank].device)?;
                library.peer_exchange_initialize()?;
                let zeroed = |bytes: usize| -> Result<DeviceAllocation<'a>> {
                    let allocation = DeviceAllocation::new(library, bytes.max(256))?;
                    library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
                    Ok(allocation)
                };
                Ok(End { recv: (0..slots).map(|_| zeroed(slot_bytes)).collect::<Result<_>>()?,
                    control: zeroed((slots + 1) * 16)? })
            })
        };
        Ok(Self { library, ranks, ends: [end(0)?, end(1)?] })
    }

    pub fn stream(&self, rank: usize) -> *mut c_void {
        self.ranks[rank].stream
    }

    pub fn device(&self, rank: usize) -> i32 {
        self.ranks[rank].device
    }

    /// Runs `body` with rank `rank`'s device current.
    pub fn on<T>(&self, rank: usize, body: impl FnOnce() -> Result<T>) -> Result<T> {
        on_device(self.library, self.ranks[rank].device, self.ranks[0].device, body)
    }

    /// Receive buffer `slot` of rank `rank`.
    pub fn recv(&self, rank: usize, slot: usize) -> Result<*mut c_void> {
        Ok(self.ends[rank].recv.get(slot).with_context(|| format!("exchange slot {slot}"))?.buffer.ptr)
    }

    /// Bytes of each receive buffer.
    pub fn slot_bytes(&self) -> usize {
        self.ends[0].recv[0].buffer.bytes
    }

    /// Control index of `slot` ([`DIRECT`]: the one after the slots).
    fn flag_index(&self, slot: usize) -> Result<usize> {
        let slots = self.ends[0].recv.len();
        if slot == DIRECT {
            return Ok(slots);
        }
        ensure!(slot < slots, "exchange slot {slot} of {slots}");
        Ok(slot)
    }

    /// Queues on `from`'s stream: push `bytes` of `source` (rows final on that
    /// stream) into the other GPU's receive slot `slot` and publish that slot's
    /// next sequence.
    pub fn push(&self, from: usize, slot: usize, source: *const c_void, bytes: usize) -> Result<()> {
        ensure!(bytes <= self.slot_bytes(), "push of {bytes} bytes past the exchange slots");
        self.push_to(from, slot, source, self.recv(1 - from, slot)?, bytes)
    }

    /// [`Self::push`] into any `destination` on the other GPU, signalling flag
    /// `slot` (usually [`DIRECT`]): the destination's reader waits on that flag.
    pub fn push_to(&self, from: usize, slot: usize, source: *const c_void, destination: *mut c_void, bytes: usize)
        -> Result<()> {
        let flag = self.flag_index(slot)?;
        tracing::trace!(from, flag, bytes, "peer push");
        let (mine, theirs) = (&self.ends[from], &self.ends[1 - from]);
        // SAFETY: peer access is enabled both ways (new); the source rows are final on
        // this stream and untouched until later work on it; the destination is a live
        // buffer of the other GPU that it reads only after its matching wait.
        self.on(from, || unsafe {
            self.library.peer_push_signal(destination, source, bytes, theirs.flag(flag), mine.send_state(flag), 0,
                self.ranks[from].stream)
        })
    }

    /// Queues on `rank`'s stream: `out = bf16(a + b)` over `count` elements (the same
    /// bits whichever order the two partials come in).
    pub fn add(&self, rank: usize, a: *const c_void, b: *const c_void, out: *mut c_void, count: usize) -> Result<()> {
        // SAFETY: callers pass live, disjoint [count] BF16 buffers of rank `rank`'s GPU whose
        // producers are ordered before on its stream.
        self.on(rank, || unsafe { self.library.peer_add_bf16(a, b, out, count, self.ranks[rank].stream) })
    }

    /// Queues on `at`'s stream: wait for the other GPU's next push on flag `slot`.
    pub fn wait(&self, at: usize, slot: usize) -> Result<()> {
        let flag = self.flag_index(slot)?;
        tracing::trace!(at, flag, "peer wait");
        let end = &self.ends[at];
        // SAFETY: the flag and state are this GPU's control words; every wait is matched by a
        // push the host queues before it waits on either stream.
        self.on(at, || unsafe {
            self.library.peer_wait(end.flag(flag), end.recv_state(flag), self.ranks[at].stream)
        })
    }
}

/// How a head split slices a 2-D weight: by output rows or by input columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Axis {
    Rows,
    Cols,
}

/// `part` of `ranks` equal slices of a row-major `[rows, cols]` tensor of
/// `elem`-byte elements along `axis`, as contiguous bytes.
pub(crate) fn slice_2d(bytes: &[u8], rows: usize, cols: usize, elem: usize, axis: Axis, part: usize, ranks: usize)
    -> Vec<u8> {
    match axis {
        Axis::Rows => {
            let n = rows / ranks;
            bytes[part * n * cols * elem..(part + 1) * n * cols * elem].to_vec()
        }
        Axis::Cols => {
            let n = cols / ranks;
            let mut out = Vec::with_capacity(rows * n * elem);
            for row in bytes.chunks_exact(cols * elem) {
                out.extend_from_slice(&row[part * n * elem..(part + 1) * n * elem]);
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_cover_the_tensor() {
        let bytes: Vec<u8> = (0..24).collect();
        assert_eq!(slice_2d(&bytes, 4, 6, 1, Axis::Rows, 1, 2), (12..24).collect::<Vec<u8>>());
        assert_eq!(slice_2d(&bytes, 4, 6, 1, Axis::Cols, 0, 2), vec![0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20]);
        assert_eq!(slice_2d(&bytes, 2, 6, 2, Axis::Cols, 1, 3), vec![4, 5, 6, 7, 16, 17, 18, 19]);
    }
}
