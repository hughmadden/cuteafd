//! What a family tells the prefix cache, and the device work it does for it.
use super::marks::MarkSlot;
use super::pages::TailCopy;
use cuteafd_core::prefix::ReuseRule;
use cuteafd_hostcache::copy::DeviceRange;
use cuteafd_hostcache::pool::Layout;
use serde::Serialize;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// A family's snapshot geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct FamilyLayout {
    /// Token rows per page (the sharing unit).
    pub page_rows: usize,
    /// Pages in the device pool.
    pub pages: usize,
    /// Device bytes of one page index, over every buffer that holds its rows (MiMo: one
    /// record block per full-attention layer). A family with several page geometries lists
    /// them as segments of one page index; there is one page class today.
    pub page_bytes: usize,
    /// Bytes of a positional mark (0: the pages are the whole state).
    pub mark_bytes: usize,
    /// Bytes of drafter state kept with a snapshot (0: drafters restore cold, masked by
    /// `context_valid_from`).
    pub draft_bytes: usize,
    /// How partial matches are reused; a positional mark allows exact frontiers only.
    pub rule: ReuseRule,
}

impl FamilyLayout {
    /// The host tier's slab sizes for this family.
    pub fn host_layout(&self) -> Layout {
        Layout::family(self.page_bytes, self.mark_bytes.max(1), self.draft_bytes)
    }
}

/// The device side of snapshots for one family. Every call only enqueues work on the family's
/// stream (in stream order with its forward passes); [`PrefixFamily::drain`] waits for it. The
/// cache drains before it publishes a snapshot to the host tier and before it releases pages or
/// mark slots, so no queued copy ever reads storage someone else was handed.
pub trait PrefixFamily {
    type Placement;

    fn layout(&self) -> FamilyLayout;
    /// The placement's pages, in row order.
    fn pages<'p>(&self, placement: &'p Self::Placement) -> &'p [u32];
    /// Committed rows: the length a snapshot of this placement may capture (a speculative
    /// family reports the rows its state is consistent at, not rows still being verified).
    fn commit_point(&self, placement: &Self::Placement) -> usize;
    /// Copy the positional state of `placement` at `len` into `slot`.
    fn capture(&self, slot: MarkSlot, placement: &Self::Placement, len: usize) -> Result<(), BoxError>;
    /// Make `placement` continue at `len`: copy `mark` back into its own positional state (none
    /// for a mark-less family) and set its length. Touches tables and buffers only, never graph
    /// shapes or workspaces.
    fn restore(&self, mark: Option<MarkSlot>, placement: &mut Self::Placement, len: usize) -> Result<(), BoxError>;
    /// Copy rows `[0, copy.rows)` of every paged buffer from page `copy.from` to page `copy.to`.
    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError>;
    /// Wait for every copy enqueued so far.
    fn drain(&self) -> Result<(), BoxError>;
    /// Device ranges of one page (host tier), concatenated in this order on the host.
    fn page_segments(&self, page: u32) -> Vec<DeviceRange>;
    /// Device ranges of one mark slot (host tier).
    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange>;
}
