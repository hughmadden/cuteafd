//! Full-context draft pages share one token-budget-sized pool. GPU owners must retire
//! readers, writers and tail copies before committing, rolling back or releasing a lease.
use super::{PoolExhausted, RefPagePool, TailCopy};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

static NEXT_POOL: AtomicU64 = AtomicU64::new(1);

fn pool_id() -> Result<u64> {
    NEXT_POOL.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| DraftPageError::Geometry)
}

pub const DRAFT_PAGE_ROWS: usize = 64;

#[derive(Debug, Error)]
pub enum DraftPageError {
    #[error("invalid draft page geometry")]
    Geometry,
    #[error("draft slot is occupied or outside the admitted layout")]
    Slot,
    #[error("stale draft ownership or reservation")]
    Stale,
    #[error("draft append differs from the committed frontier or context extent")]
    Frontier,
    #[error("draft snapshot is incompatible with the requested frontier")]
    Snapshot,
    #[error(transparent)]
    Exhausted(#[from] PoolExhausted),
}

type Result<T> = std::result::Result<T, DraftPageError>;

/// A request binding cannot survive slot reuse, even when its physical pages survive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftBinding {
    pool: u64,
    slot: usize,
    generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftReservation {
    binding: DraftBinding,
    serial: u64,
    start: usize,
    end: usize,
}

impl DraftReservation {
    pub fn start(self) -> usize { self.start }
    pub fn end(self) -> usize { self.end }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DraftSnapshot { pool: u64, serial: u64 }

/// A page identity is meaningful only in its pool/layout namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftPageId {
    pub index: u32,
    pub generation: u32,
}

/// Plain metadata for a typed snapshot attachment. Full cold-prefill compatibility
/// additionally depends on the shared scheduler's checkpoint and stage-state predicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftPageAttachment {
    pub layout: u64,
    pub origin: usize,
    pub frontier: usize,
    pub pages: Vec<DraftPageId>,
}

struct Sequence {
    binding: DraftBinding,
    origin: usize,
    end: usize,
    pages: Vec<u32>,
    pending: Option<DraftReservation>,
}

struct Snapshot {
    origin: usize,
    end: usize,
    pages: Vec<u32>,
}

/// Physical page IDs address the corresponding segment in every full draft layer.
/// Windowed layer rings do not allocate from this pool. Partial tails belong to one
/// writer: capture and restore eagerly copy them using the existing prefix allocator.
/// No operation allocates storage beyond ceil(target_tokens / 64) pages.
pub struct DraftPagePool {
    id: u64,
    pool: RefPagePool,
    slots: Vec<Option<Sequence>>,
    generations: Vec<u64>,
    snapshots: BTreeMap<DraftSnapshot, Snapshot>,
    serial: u64,
    context: usize,
    layout: u64,
}

impl DraftPagePool {
    pub fn new(target_tokens: usize, slots: usize, context: usize, layout: u64) -> Result<Self> {
        let pages = target_tokens.div_ceil(DRAFT_PAGE_ROWS);
        if target_tokens == 0 || slots == 0 || context == 0 || pages > i32::MAX as usize {
            return Err(DraftPageError::Geometry);
        }
        let id = pool_id()?;
        Ok(Self { id, pool: RefPagePool::new(pages, DRAFT_PAGE_ROWS),
            slots: (0..slots).map(|_| None).collect(), generations: vec![0; slots],
            snapshots: BTreeMap::new(), serial: 0, context, layout })
    }

    pub fn capacity(&self) -> usize { self.pool.capacity() }
    pub fn free(&self) -> usize { self.pool.free() }

    fn vacant(&self, slot: usize) -> Result<()> {
        if self.slots.get(slot).is_some_and(Option::is_none) { Ok(()) }
        else { Err(DraftPageError::Slot) }
    }

    fn sequence(&self, binding: DraftBinding) -> Result<&Sequence> {
        self.slots.get(binding.slot).and_then(Option::as_ref)
            .filter(|s| s.binding == binding).ok_or(DraftPageError::Stale)
    }

    fn next_serial(&mut self) -> Result<u64> {
        self.serial = self.serial.checked_add(1).ok_or(DraftPageError::Stale)?;
        Ok(self.serial)
    }

    fn install(&mut self, slot: usize, origin: usize, end: usize, pages: Vec<u32>) -> DraftBinding {
        // Callers preflight generation overflow before acquiring page references.
        self.generations[slot] += 1;
        let binding = DraftBinding { pool: self.id, slot, generation: self.generations[slot] };
        self.slots[slot] = Some(Sequence { binding, origin, end, pages, pending: None });
        binding
    }

    /// Bind cold(P) without allocating a page or exposing the previous occupant's bytes.
    pub fn bind(&mut self, slot: usize, origin: usize) -> Result<DraftBinding> {
        self.vacant(slot)?;
        if origin > self.context { return Err(DraftPageError::Frontier); }
        self.generations[slot].checked_add(1).ok_or(DraftPageError::Stale)?;
        Ok(self.install(slot, origin, origin, Vec::new()))
    }

    pub fn frontier(&self, binding: DraftBinding) -> Result<(usize, usize)> {
        let s = self.sequence(binding)?;
        Ok((s.origin, s.end))
    }

    /// Reserve all producer destinations before enqueueing any layer's writes.
    pub fn reserve(&mut self, binding: DraftBinding, start: usize, rows: usize) -> Result<DraftReservation> {
        let s = self.sequence(binding)?;
        let end = start.checked_add(rows).ok_or(DraftPageError::Frontier)?;
        if s.pending.is_some() || start != s.end || rows == 0 || end > self.context {
            return Err(DraftPageError::Frontier);
        }
        let total = (end - s.origin).div_ceil(DRAFT_PAGE_ROWS);
        let previous = s.pages.len();
        let serial = self.next_serial()?;
        let fresh = self.pool.alloc(total - previous)?;
        let reservation = DraftReservation { binding, serial, start, end };
        let s = self.slots[binding.slot].as_mut().expect("validated binding");
        s.pages.extend(fresh);
        s.pending = Some(reservation);
        Ok(reservation)
    }

    fn reserved(&self, reservation: DraftReservation) -> Result<&Sequence> {
        self.sequence(reservation.binding)?.pending.filter(|r| *r == reservation)
            .ok_or(DraftPageError::Stale)?;
        self.sequence(reservation.binding)
    }

    /// Absolute positions drive RoPE; destinations are origin-relative physical rows.
    pub fn destinations(&self, reservation: DraftReservation) -> Result<Vec<i64>> {
        let s = self.reserved(reservation)?;
        Ok((reservation.start..reservation.end).map(|position| {
            let relative = position - s.origin;
            i64::from(s.pages[relative / DRAFT_PAGE_ROWS]) * DRAFT_PAGE_ROWS as i64
                + (relative % DRAFT_PAGE_ROWS) as i64
        }).collect())
    }

    /// Includes the provisional suffix. The caller bounds attention with the committed
    /// frontier and query offsets; page membership does not make proposal rows committed.
    pub fn page_table(&self, reservation: DraftReservation) -> Result<Vec<i32>> {
        Ok(self.reserved(reservation)?.pages.iter().map(|&p| p as i32).collect())
    }

    /// Publish only after accepted rows have their canonical target-derived KV and
    /// every queued user of the rejected suffix has retired. accepted=0 is rollback.
    pub fn commit(&mut self, reservation: DraftReservation, accepted: usize) -> Result<()> {
        self.reserved(reservation)?;
        if accepted > reservation.end - reservation.start { return Err(DraftPageError::Frontier); }
        let s = self.slots[reservation.binding.slot].as_mut().expect("validated binding");
        s.end = reservation.start + accepted;
        let retained = (s.end - s.origin).div_ceil(DRAFT_PAGE_ROWS);
        let released = s.pages.split_off(retained);
        s.pending = None;
        self.pool.release(&released);
        Ok(())
    }

    /// Retain a committed prefix. Run the returned tail copy on every full layer,
    /// and complete its owner events before publishing this snapshot to lookup.
    pub fn capture(&mut self, binding: DraftBinding) -> Result<(DraftSnapshot, Option<TailCopy>)> {
        let s = self.sequence(binding)?;
        if s.pending.is_some() { return Err(DraftPageError::Frontier); }
        let (origin, end, source) = (s.origin, s.end, s.pages.clone());
        let id = DraftSnapshot { pool: self.id, serial: self.next_serial()? };
        let len = end - origin;
        let (pages, copy) = if len == 0 { (Vec::new(), None) } else {
            let fork = self.pool.fork(&source, len, source.len())?;
            (fork.pages, fork.copy)
        };
        self.snapshots.insert(id, Snapshot { origin, end, pages });
        Ok((id, copy))
    }

    pub fn attachment(&self, snapshot: DraftSnapshot) -> Result<DraftPageAttachment> {
        let s = self.snapshots.get(&snapshot).ok_or(DraftPageError::Snapshot)?;
        Ok(DraftPageAttachment { layout: self.layout, origin: s.origin, frontier: s.end,
            pages: s.pages.iter().map(|&index| DraftPageId {
                index, generation: self.pool.generation(index) }).collect() })
    }

    /// Reserve a private continuation of an exact snapshot, leaving the destination
    /// unbound on exhaustion. The returned tail copy must retire before any producer.
    pub fn restore(&mut self, slot: usize, snapshot: DraftSnapshot, frontier: usize)
        -> Result<(DraftBinding, Option<TailCopy>)> {
        self.vacant(slot)?;
        self.generations[slot].checked_add(1).ok_or(DraftPageError::Stale)?;
        let s = self.snapshots.get(&snapshot).ok_or(DraftPageError::Snapshot)?;
        if frontier != s.end { return Err(DraftPageError::Snapshot); }
        let (origin, end) = (s.origin, s.end);
        let len = end - origin;
        let (pages, copy) = if len == 0 { (Vec::new(), None) } else {
            let fork = self.pool.fork(&s.pages, len, s.pages.len())?;
            (fork.pages, fork.copy)
        };
        Ok((self.install(slot, origin, end, pages), copy))
    }

    /// Call only after queued copies/readers of the snapshot have retired.
    pub fn release_snapshot(&mut self, snapshot: DraftSnapshot) -> Result<()> {
        let s = self.snapshots.remove(&snapshot).ok_or(DraftPageError::Snapshot)?;
        self.pool.release(&s.pages);
        Ok(())
    }

    /// Cancellation, including a pending proposal. The owner must drain first.
    pub fn release(&mut self, binding: DraftBinding) -> Result<()> {
        self.sequence(binding)?;
        let s = self.slots[binding.slot].take().expect("validated binding");
        self.pool.release(&s.pages);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_from_another_pool_never_alias() {
        let mut a = DraftPagePool::new(128, 1, 1024, 1).unwrap();
        let mut b = DraftPagePool::new(128, 1, 1024, 1).unwrap();
        let binding_a = a.bind(0, 0).unwrap();
        let binding_b = b.bind(0, 0).unwrap();
        let (snapshot_a, _) = a.capture(binding_a).unwrap();
        let (snapshot_b, _) = b.capture(binding_b).unwrap();
        assert!(b.release(binding_a).is_err());
        assert!(b.release_snapshot(snapshot_a).is_err());
        assert!(b.attachment(snapshot_a).is_err());
        assert!(b.attachment(snapshot_b).is_ok());
        assert_eq!(b.frontier(binding_b).unwrap(), (0, 0));
    }

    #[test]
    fn aggregate_budget_is_independent_of_slots_and_context() {
        for slots in [1, 8, 20] {
            for context in [1024, 1 << 20] {
                let pool = DraftPagePool::new(1025, slots, context, 1).unwrap();
                assert_eq!((pool.capacity(), pool.free()), (17, 17));
            }
        }
    }

    #[test]
    fn rejected_suffix_and_slot_reuse_cannot_publish_stale_frontiers() {
        let mut pool = DraftPagePool::new(256, 2, 1 << 20, 7).unwrap();
        let a = pool.bind(0, 131).unwrap();
        let r = pool.reserve(a, 131, 65).unwrap();
        assert_eq!(pool.destinations(r).unwrap(), (0..65).collect::<Vec<i64>>());
        pool.commit(r, 63).unwrap();
        assert_eq!(pool.frontier(a).unwrap(), (131, 194));
        assert_eq!(pool.free(), 3);
        assert!(pool.commit(r, 1).is_err());
        let r = pool.reserve(a, 194, 2).unwrap();
        pool.commit(r, 0).unwrap();
        assert_eq!(pool.frontier(a).unwrap(), (131, 194));
        pool.release(a).unwrap();
        let b = pool.bind(0, 500).unwrap();
        assert!(pool.reserve(a, 194, 1).is_err());
        assert_eq!(pool.frontier(b).unwrap(), (500, 500));
        assert_eq!(pool.free(), 4);
    }

    #[test]
    fn capture_restore_share_full_pages_and_copy_only_the_partial_tail() {
        for rows in [63, 64, 65] {
            let mut pool = DraftPagePool::new(512, 2, 1024, 11).unwrap();
            let a = pool.bind(0, 0).unwrap();
            let r = pool.reserve(a, 0, rows).unwrap();
            pool.commit(r, rows).unwrap();
            let (snapshot, copy) = pool.capture(a).unwrap();
            assert_eq!(copy.map(|c| c.rows), (rows % 64 != 0).then_some(rows % 64));
            let attachment = pool.attachment(snapshot).unwrap();
            assert_eq!((attachment.origin, attachment.frontier, attachment.layout), (0, rows, 11));
            let (b, copy) = pool.restore(1, snapshot, rows).unwrap();
            assert_eq!(copy.map(|c| c.rows), (rows % 64 != 0).then_some(rows % 64));
            let r = pool.reserve(b, rows, 1).unwrap();
            pool.commit(r, 0).unwrap();
            assert_eq!(pool.attachment(snapshot).unwrap(), attachment);
            pool.release(a).unwrap();
            pool.release(b).unwrap();
            pool.release_snapshot(snapshot).unwrap();
            assert_eq!(pool.free(), 8);
            assert!(pool.attachment(snapshot).is_err());
        }
    }

    #[test]
    fn snapshot_bytes_equal_cold_prefill_after_proposals_and_tail_copies() {
        let mut pool = DraftPagePool::new(1024, 3, 4096, 19).unwrap();
        let mut memory = vec![0u64; 1024];
        let a = pool.bind(0, 0).unwrap();
        for end in [1, 63, 64, 65, 127, 191] {
            let (_, start) = pool.frontier(a).unwrap();
            let proposal = pool.reserve(a, start, 8).unwrap();
            for row in pool.destinations(proposal).unwrap() { memory[row as usize] = u64::MAX; }
            pool.commit(proposal, 0).unwrap();
            let append = pool.reserve(a, start, end - start).unwrap();
            for (position, row) in (start..end).zip(pool.destinations(append).unwrap()) {
                memory[row as usize] = position as u64 + 1;
            }
            pool.commit(append, end - start).unwrap();
        }
        let (snapshot, copy) = pool.capture(a).unwrap();
        if let Some(copy) = copy {
            let saved = memory[copy.from as usize * 64..copy.from as usize * 64 + copy.rows].to_vec();
            memory[copy.to as usize * 64..copy.to as usize * 64 + copy.rows].copy_from_slice(&saved);
        }
        let attachment = pool.attachment(snapshot).unwrap();
        let host = attachment.pages.iter().enumerate().flat_map(|(i, page)| {
            let rows = (attachment.frontier - i * 64).min(64);
            memory[page.index as usize * 64..page.index as usize * 64 + rows].to_vec()
        }).collect::<Vec<_>>();
        assert_eq!(host, (1..=191).collect::<Vec<u64>>());
        // The writer can append rejected bytes without changing its snapshot tail.
        let proposal = pool.reserve(a, 191, 8).unwrap();
        for row in pool.destinations(proposal).unwrap() { memory[row as usize] = u64::MAX; }
        pool.commit(proposal, 0).unwrap();
        let (b, copy) = pool.restore(1, snapshot, 191).unwrap();
        if let Some(copy) = copy {
            let saved = memory[copy.from as usize * 64..copy.from as usize * 64 + copy.rows].to_vec();
            memory[copy.to as usize * 64..copy.to as usize * 64 + copy.rows].copy_from_slice(&saved);
        }
        let s = pool.sequence(b).unwrap();
        let restored = (0..191).map(|position| memory[s.pages[position / 64] as usize * 64 + position % 64])
            .collect::<Vec<_>>();
        assert_eq!(restored, host);
        assert_eq!(pool.attachment(snapshot).unwrap(), attachment);
        pool.release(a).unwrap();
        pool.release(b).unwrap();
        pool.release_snapshot(snapshot).unwrap();
        assert_eq!(pool.free(), 16);
    }

    #[test]
    fn exhaustion_and_bad_restore_are_transactional() {
        let mut pool = DraftPagePool::new(128, 2, 1024, 1).unwrap();
        let a = pool.bind(0, 0).unwrap();
        let r = pool.reserve(a, 0, 65).unwrap();
        pool.commit(r, 65).unwrap();
        assert!(pool.capture(a).is_err());
        assert_eq!(pool.free(), 0);
        assert_eq!(pool.frontier(a).unwrap(), (0, 65));
        assert!(pool.reserve(a, 65, 64).is_err());
        let r = pool.reserve(a, 65, 1).unwrap();
        pool.commit(r, 0).unwrap();
        pool.release(a).unwrap();
        let a = pool.bind(0, 0).unwrap();
        let r = pool.reserve(a, 0, 64).unwrap();
        pool.commit(r, 64).unwrap();
        let (snapshot, _) = pool.capture(a).unwrap();
        assert!(pool.restore(1, snapshot, 63).is_err());
        assert!(pool.bind(1, 0).is_ok());
    }

    #[test]
    fn empty_and_cold_origin_snapshots_keep_their_semantics() {
        let mut pool = DraftPagePool::new(64, 2, 1024, 2).unwrap();
        let a = pool.bind(0, 513).unwrap();
        let (snapshot, copy) = pool.capture(a).unwrap();
        assert!(copy.is_none());
        assert_eq!(pool.free(), 1);
        let (b, _) = pool.restore(1, snapshot, 513).unwrap();
        assert_eq!(pool.frontier(b).unwrap(), (513, 513));
        assert_ne!(pool.attachment(snapshot).unwrap().origin, 0);
    }
}
