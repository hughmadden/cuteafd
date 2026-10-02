//! MiMo V2 (Flash, V2.6 Pro) as a prefix-cache family (`cuteafd_engine::prefix`).
//!
//! Paged state: the full-attention layers' BF16 records, 64 rows per page; one page index
//! spans every full layer (Flash 9 x 64 x 2,560 B = 1.47 MB, Pro 10 x 64 x 5,120 B = 3.28 MB).
//! Full pages are shared by reference; nobody writes them again (every sequence appends past
//! its own length), so only a partial tail page is copied.
//!
//! Mark: the sliding-window layers keep a 256-slot ring per sequence that the next step
//! overwrites, and their attention reads keys `p - window + 1 ..= p`, so a sequence continuing
//! at `P` needs ring rows `P - window + 1 .. P`; the mark keeps the last `window` rows of
//! every SWA layer in position order (Flash 39 x 128 x 5,120 B = 25.6 MB, Pro 60 layers =
//! 39.3 MB). With MTP (Flash) it also keeps the last `window + stages + 1` rows of the MTP
//! hidden ring: `mtp_reset` at the restored length makes every stage recompute its own ring
//! from them, exactly as after a full prefill (~1 MiB). Rows go back to the same `position %
//! 256` slots of the new sequence's ring, so the restored ring reads exactly as the captured
//! one. The DFlash drafter (Pro) is not captured: a restored sequence drafts cold with
//! `context_valid_from` at the restore point (drafts only steer speculation).
//!
//! Restores are exact: the deepest retained point whose tokens prefix the request wins
//! (`ReuseRule::EXACT`), and intermediate points (message boundaries, periodic chunk ends; see
//! `cuteafd_engine::prefix::plan_points`) keep one close. A point may lie up to
//! `capture_reach` rows behind the committed length: the rings still hold its window there
//! (256 slots, minus a verify step's 64 rows that may be written past the commit point, minus
//! the rows a mark keeps).
//!
//! Opt-in (`--prefix-partial on`, off by default): V4.1-style partial reuse. A partial match
//! shares the pages below `aligned common - window`, restarts the positional state there and
//! replays the window. The kernels have no per-sequence key floor, so "restarts" means the
//! ring rows before the restart point are zeroed (zero keys and values: an extra zero-logit
//! sink for the first replayed rows); approximate by design, not byte-exact.
//!
//! Every copy is enqueued on the engine stream, in order with the forward passes, and `drain`
//! synchronizes it. Under a head split each GPU keeps its own KV heads: its pages and ring rows
//! are copied on that GPU's stream into its own mark arena (the same slot on both GPUs).
use super::engine::{MimoEngine, MimoPlacement, DECODE_ROWS, PAGE_ROWS, RING_ROWS};
use super::mtp::HIDDEN_ROWS;
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, PrefixFamily, ReuseRule, TailCopy};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_hostcache::copy::DeviceRange;
use cuteafd_loader::families::mimo_v2::MimoAttention;
use std::rc::Rc;

/// One ring-structured state the mark keeps rows of: `rows` rows before the frontier, each
/// `row` bytes, ring `r` position `p` at `(r * ring_rows + p % ring_rows) * row`.
#[derive(Clone, Copy)]
struct RingState {
    /// The head-split rank whose GPU holds the ring (0 without a split).
    rank: usize,
    buffer: CuteafdDeviceBuffer,
    row: usize,
    ring_rows: usize,
    rows: usize,
    /// Byte offset of this state's rows inside a mark.
    offset: usize,
}

pub(crate) struct MimoPrefix<'e, 'a> {
    engine: &'e MimoEngine<'a>,
    /// Full-attention record pools: (rank, buffer, record bytes).
    full: Vec<(usize, CuteafdDeviceBuffer, usize)>,
    states: Vec<RingState>,
    /// Bytes of one mark on each rank's GPU, and their sum.
    rank_mark_bytes: Vec<usize>,
    mark_bytes: usize,
    /// Per rank: its marks' arena.
    arenas: Vec<Rc<Allocation<'a>>>,
    slots: usize,
    partial: bool,
}

/// A byte range inside `buffer` (bounds-checked; pointer arithmetic only).
fn view(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "view {offset}+{bytes} past a {}-byte buffer", buffer.bytes);
    Ok(CuteafdDeviceBuffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

impl<'e, 'a> MimoPrefix<'e, 'a> {
    /// The family over `engine`'s buffers with a device arena of `slots(mark_bytes)` marks;
    /// `partial` opts into V4.1-style partial reuse (approximate).
    pub fn new(engine: &'e MimoEngine<'a>, slots: impl FnOnce(usize) -> usize, partial: bool) -> Result<Self> {
        let (mut full, mut states) = (Vec::new(), Vec::new());
        let mut offsets = vec![0usize; engine.ranks()];
        let window = engine.cfg.window;
        ensure!(window > 0 && window <= RING_ROWS, "SWA window {window} does not fit the {RING_ROWS}-slot ring");
        for rank in 0..engine.ranks() {
            for layer in 0..engine.weights.layers.len() {
                let (attention, buffer, record) = engine.kv_layer_on(rank, layer);
                match attention {
                    MimoAttention::Full => full.push((rank, buffer, record)),
                    MimoAttention::Sliding => {
                        states.push(RingState { rank, buffer, row: record, ring_rows: RING_ROWS, rows: window,
                            offset: offsets[rank] });
                        offsets[rank] += window * record;
                    }
                }
            }
        }
        if let Some(mtp) = &engine.mtp {
            let rows = window + mtp.stages.len() + 1;
            ensure!(rows <= HIDDEN_ROWS, "MTP catch-up of {rows} rows exceeds the {HIDDEN_ROWS}-row hidden ring");
            let row = engine.cfg.hidden * 2;
            states.push(RingState { rank: 0, buffer: mtp.hidden.buffer, row, ring_rows: HIDDEN_ROWS, rows,
                offset: offsets[0] });
            offsets[0] += rows * row;
        }
        let mark_bytes: usize = offsets.iter().sum();
        let slots = slots(mark_bytes);
        let arenas = if slots > 0 && mark_bytes > 0 {
            offsets.iter().enumerate().map(|(rank, &bytes)| {
                let device = Device { library: engine.library, id: engine.kv_layer_on(rank, 0).1.device_id };
                Allocation::new(device, (slots * bytes).max(256)).map(Rc::new)
            }).collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        ensure!(!partial || window % PAGE_ROWS == 0, "partial reuse replays a whole number of pages");
        Ok(Self { engine, full, states, rank_mark_bytes: offsets, mark_bytes, arenas, slots, partial })
    }

    pub fn mark_bytes(&self) -> usize {
        self.mark_bytes
    }

    pub fn page_bytes(&self) -> usize {
        self.full.iter().map(|(_, _, record)| PAGE_ROWS * record).sum()
    }

    pub fn slots(&self) -> usize {
        if self.arenas.is_empty() { 0 } else { self.slots }
    }

    /// Stable allocations the host snapshot tier may copy. Full pages and
    /// per-rank SWA/MTP mark arenas are registered on their actual devices;
    /// the live rings are captured into those arenas before host submission.
    pub fn host_owners(&self) -> Vec<Rc<Allocation<'a>>> {
        self.engine.full_kv_owners().into_iter().chain(self.arenas.iter().cloned()).collect()
    }

    /// Zero every state's rows before `len` (a partial restore's empty window).
    fn empty_window(&self, ring: usize, len: usize) -> Result<()> {
        for state in &self.states {
            for (ring_row, _, rows) in runs(ring, state.ring_rows, state.rows, len) {
                let target = view(state.buffer, ring_row * state.row, rows * state.row)?;
                // SAFETY: the view lies inside a live allocation of the state's GPU; ordered on its stream.
                self.engine.on(state.rank, || unsafe {
                    self.engine.library.cuda_zero_bytes_async(target, target.bytes, self.engine.stream_of(state.rank))
                })?;
            }
        }
        Ok(())
    }

    /// A copy between two views on rank `rank`'s GPU.
    fn copy(&self, rank: usize, dst: CuteafdDeviceBuffer, src: CuteafdDeviceBuffer) -> Result<()> {
        debug_assert_eq!(dst.bytes, src.bytes);
        // SAFETY: both views lie inside live allocations of that GPU (checked by `view`); the copy
        // is ordered on its stream with every forward pass that reads or writes them.
        self.engine.on(rank, || unsafe {
            self.engine.library.copy_d2d_async(dst, src, src.bytes, self.engine.stream_of(rank))
        })
    }

    /// Copy every state's last rows before `len` between ring `ring` and mark `slot`.
    fn move_mark(&self, slot: MarkSlot, ring: usize, len: usize, capture: bool) -> Result<()> {
        ensure!(!self.arenas.is_empty(), "no mark arena");
        ensure!((slot.0 as usize) < self.slots, "mark slot {} of {}", slot.0, self.slots);
        for state in &self.states {
            let arena = self.arenas[state.rank].buffer;
            let base = slot.0 as usize * self.rank_mark_bytes[state.rank];
            for (ring_row, mark_row, rows) in runs(ring, state.ring_rows, state.rows, len) {
                let ring_view = view(state.buffer, ring_row * state.row, rows * state.row)?;
                let mark_view = view(arena, base + state.offset + mark_row * state.row, rows * state.row)?;
                if capture {
                    self.copy(state.rank, mark_view, ring_view)?;
                } else {
                    self.copy(state.rank, ring_view, mark_view)?;
                }
            }
        }
        Ok(())
    }
}

/// The last `min(len, rows)` positions before `len` of ring `ring` as runs whose ring slots do
/// not wrap: (ring row, mark row, rows), mark rows in position order from the first kept one.
fn runs(ring: usize, ring_rows: usize, rows: usize, len: usize) -> Vec<(usize, usize, usize)> {
    let first = len - len.min(rows);
    let mut out = Vec::new();
    let mut position = first;
    while position < len {
        let at = position % ring_rows;
        let run = (len - position).min(ring_rows - at);
        out.push((ring * ring_rows + at, position - first, run));
        position += run;
    }
    out
}

impl PrefixFamily for MimoPrefix<'_, '_> {
    type Placement = MimoPlacement;

    fn layout(&self) -> FamilyLayout {
        FamilyLayout {
            page_rows: PAGE_ROWS,
            pages: self.engine.pages,
            page_bytes: self.page_bytes(),
            mark_bytes: self.mark_bytes,
            draft_bytes: 0,
            rule: if self.partial {
                ReuseRule { align: PAGE_ROWS, replay: Some(self.engine.cfg.window) }
            } else {
                ReuseRule::EXACT
            },
        }
    }

    fn capture_reach(&self) -> usize {
        let kept = self.states.iter().map(|s| s.rows).max().unwrap_or(0);
        RING_ROWS.saturating_sub(DECODE_ROWS + kept)
    }

    fn pages<'p>(&self, placement: &'p MimoPlacement) -> &'p [u32] {
        &placement.pages
    }

    fn commit_point(&self, placement: &MimoPlacement) -> usize {
        placement.len
    }

    fn capture(&self, slot: MarkSlot, placement: &MimoPlacement, len: usize) -> Result<(), BoxError> {
        if len > placement.len || placement.len - len > self.capture_reach() {
            return Err(format!("capture at {len} is out of reach of the committed {} rows", placement.len).into());
        }
        Ok(self.move_mark(slot, placement.ring as usize, len, true)?)
    }

    fn restore(&self, mark: Option<MarkSlot>, placement: &mut MimoPlacement, len: usize) -> Result<(), BoxError> {
        if len.div_ceil(PAGE_ROWS) > placement.pages.len() {
            return Err(format!("restore of {len} rows into {} pages", placement.pages.len()).into());
        }
        match mark {
            Some(slot) => self.move_mark(slot, placement.ring as usize, len, false)?,
            None if self.partial => self.empty_window(placement.ring as usize, len)?,
            None => return Err("MiMo restores exact snapshots with their positional mark".into()),
        }
        placement.len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        for &(rank, buffer, record) in &self.full {
            let page = PAGE_ROWS * record;
            self.copy(rank, view(buffer, copy.to as usize * page, copy.rows * record)?,
                view(buffer, copy.from as usize * page, copy.rows * record)?)?;
        }
        Ok(())
    }

    fn drain(&self) -> Result<(), BoxError> {
        for rank in 0..self.engine.ranks() {
            let device = Device { library: self.engine.library, id: self.engine.kv_layer_on(rank, 0).1.device_id };
            device.run(|| {
                // SAFETY: the engine owns this stream on the selected rank.
                unsafe { self.engine.library.cuda_stream_synchronize(self.engine.stream_of(rank)) }
            })?;
        }
        Ok(())
    }

    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        self.full.iter().map(|&(_, buffer, record)| {
            let bytes = PAGE_ROWS * record;
            DeviceRange { addr: buffer.ptr as u64 + (page as usize * bytes) as u64, bytes }
        }).collect()
    }

    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        self.arenas.iter().zip(&self.rank_mark_bytes).map(|(arena, &bytes)| DeviceRange {
            addr: arena.buffer.ptr as u64 + (slot.0 as usize * bytes) as u64,
            bytes,
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::runs;

    #[test]
    fn mark_rows_follow_positions_across_the_ring_wrap() {
        // Positions 172..300 of ring 2: slots 172..255, then 0..43.
        assert_eq!(runs(2, 256, 128, 300), vec![(512 + 172, 0, 84), (512, 84, 44)]);
        // A short sequence keeps what it has; an aligned one is one run.
        assert_eq!(runs(0, 256, 128, 50), vec![(0, 0, 50)]);
        assert_eq!(runs(1, 256, 128, 256), vec![(256 + 128, 0, 128)]);
        assert!(runs(3, 256, 128, 0).is_empty());
        // The MTP hidden ring keeps window + stages + 1 rows.
        assert_eq!(runs(0, 256, 132, 1000), vec![(100, 0, 132)]);
        assert_eq!(runs(0, 256, 132, 1100), vec![(200, 0, 56), (0, 56, 76)]);
    }
}
