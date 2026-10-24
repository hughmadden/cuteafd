//! One refcounted device page pool for every generic family (it replaces the per-family free
//! lists). A page holds `page_rows` token rows of every paged layer; a sequence's placement is a
//! list of page indices. Sharing is by reference count: a retained snapshot and every request
//! restored from it share only pages below their earliest writable row. A target-only writer
//! appends past its committed length; a lagging predictor can still write a retained row.
//! Forks eagerly copy every retained page at or beyond that writable frontier.
//!
//! Identity: a page index is reused after it is freed, so every free bumps the index's
//! generation. `(index, generation)` names one allocation for the host tier's page sharing.
//!
//! Reserved pages ([`RefPagePool::with_reserved`]): the leading page indices that no allocation
//! ever hands out, so they keep whatever the family put there at start-up.
//!
//! Parity half pools ([`RefPagePool::parity`], token-split `context` attention, PLAN.md "v3
//! attention placement" section 3): two half pools under one id space. Page id `i` lives on GPU
//! `i & 1` at local index `i >> 1`, and a sequence's page at logical position `j` always gets an
//! id of parity `j % 2` ([`RefPagePool::alloc_for`]). Ownership is a function of the logical
//! position alone, so a forked full page and a copied tail keep their position and therefore
//! their GPU; a sequence of `n` pages takes `ceil(n/2)` pages of GPU0 and `floor(n/2)` of GPU1
//! (at most one page of imbalance per placement or snapshot). Admission needs both halves.
//!
//! Tiered pools ([`RefPagePool::tiered`], MiniMax host K/V placement): one GPU, one id space
//! whose ids below [`RefPagePool::resident`] are device-resident pages and the rest host pages.
//! Two free lists; an allocation counts both and hands its fast pages to the highest logical
//! positions it covers (the end a sequence writes next), so a short request is wholly fast.
use thiserror::Error;

/// Not enough free pages; nothing was allocated. In a parity pool the numbers are those of the
/// first half that is short (`half`), so a waiter compares them with
/// [`RefPagePool::admission_free`].
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("device page pool exhausted: {needed} pages needed, {free} free of {capacity}")]
pub struct PoolExhausted {
    pub needed: usize,
    pub free: usize,
    pub capacity: usize,
}

/// A partial tail copied by a fork: rows `[0, rows)` of page `from` into page `to`. In a parity
/// pool both pages sit at the same logical position, so they share an owner (`owner`, the GPU
/// whose stream runs the copy); `None` in a uniform pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailCopy {
    pub from: u32,
    pub to: u32,
    pub rows: usize,
    pub owner: Option<u8>,
}

/// The pages of a forked sequence and the tail copy the family must run before it writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fork {
    pub pages: Vec<u32>,
    pub copy: Option<TailCopy>,
}

/// A page freed by a release: its index and the generation it had while it was allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FreedPage {
    pub index: u32,
    pub generation: u32,
}

/// Fresh pages an allocation needs from each half pool (a uniform pool counts in `[0]` only).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Need(pub [usize; 2]);

impl Need {
    pub fn total(self) -> usize {
        self.0[0] + self.0[1]
    }
}

impl std::ops::Add for Need {
    type Output = Need;
    fn add(self, other: Need) -> Need {
        Need([self.0[0] + other.0[0], self.0[1] + other.0[1]])
    }
}

pub struct RefPagePool {
    refs: Vec<u32>,
    generation: Vec<u32>,
    /// Free indices per half (a uniform pool uses `[0]`); popped from the end, so a fresh pool
    /// hands out its first unreserved page, then the next, ...
    free: [Vec<u32>; 2],
    /// Page ids never handed out: the leading `reserved` (uniform), or the leading `reserved`
    /// local indices of each half (parity: ids `0..2 * reserved`).
    reserved: usize,
    parity: bool,
    /// Tiered pool: ids below this are the fast tier (`free[0]`), the rest `free[1]`.
    resident: Option<u32>,
    page_rows: usize,
    release_epoch: u64,
}

impl RefPagePool {
    pub fn new(pages: usize, page_rows: usize) -> Self {
        Self::with_reserved(pages, page_rows, 0)
    }

    /// A uniform pool whose ids `0..resident` are the fast tier and `resident..pages` the slow
    /// one (see the module notes). Admission counts both tiers.
    pub fn tiered(pages: usize, page_rows: usize, resident: usize) -> Self {
        assert!(resident <= pages, "{resident} resident pages of {pages}");
        let mut pool = Self::build(pages, page_rows, 0, false);
        pool.free[0].retain(|&page| (page as usize) < resident);
        pool.free[1] = (resident as u32..pages as u32).rev().collect();
        pool.resident = Some(resident as u32);
        pool
    }

    /// The fast tier's id bound in a tiered pool (every id in other pools).
    pub fn resident(&self) -> usize {
        self.resident.map_or(self.refs.len(), |r| r as usize)
    }
    /// Whether `page` is in the fast tier (always outside a tiered pool).
    pub fn is_resident(&self, page: u32) -> bool {
        self.resident.is_none_or(|r| page < r)
    }
    /// Free pages per tier, fast then slow (a pool without tiers counts all as fast).
    pub fn free_tiers(&self) -> [usize; 2] {
        if self.resident.is_some() { [self.free[0].len(), self.free[1].len()] } else { [self.free(), 0] }
    }

    /// A pool of `pages` page indices whose first `reserved` are never handed out (they keep
    /// whatever the family wrote there at start-up).
    pub fn with_reserved(pages: usize, page_rows: usize, reserved: usize) -> Self {
        assert!(reserved <= pages, "{reserved} reserved pages of {pages}");
        Self::build(pages, page_rows, reserved, false)
    }

    /// Two half pools of `pages / 2` local pages each under one id space (`pages` even): id `i`
    /// is local page `i >> 1` of GPU `i & 1`. Each half keeps its first `reserved` local pages
    /// (GLM 5.3 Flash's zero page on every GPU).
    pub fn parity(pages: usize, page_rows: usize, reserved: usize) -> Self {
        assert!(pages % 2 == 0, "a parity pool has two equal halves, not {pages} pages");
        assert!(reserved <= pages / 2, "{reserved} reserved pages of {} per half", pages / 2);
        Self::build(pages, page_rows, reserved, true)
    }

    fn build(pages: usize, page_rows: usize, reserved: usize, parity: bool) -> Self {
        assert!(page_rows > 0, "pages hold at least one row");
        let pages = u32::try_from(pages).expect("page count fits u32");
        let first = if parity { 2 * reserved as u32 } else { reserved as u32 };
        let mut free: [Vec<u32>; 2] = Default::default();
        for page in (first..pages).rev() {
            free[if parity { (page & 1) as usize } else { 0 }].push(page);
        }
        Self {
            refs: vec![0; pages as usize],
            generation: vec![0; pages as usize],
            free,
            reserved,
            parity,
            resident: None,
            page_rows,
            release_epoch: 0,
        }
    }

    /// The free list a freed `page` returns to.
    fn list(&self, page: u32) -> usize {
        if self.parity { (page & 1) as usize } else { usize::from(!self.is_resident(page)) }
    }

    /// Whether this is a parity pool ([`RefPagePool::parity`]).
    pub fn is_parity(&self) -> bool {
        self.parity
    }
    /// The GPU whose half holds `page` (parity pools; `None` in a uniform pool).
    pub fn owner(&self, page: u32) -> Option<u8> {
        self.parity.then_some((page & 1) as u8)
    }
    /// Pages the pool can hand out (reserved ones excluded).
    pub fn capacity(&self) -> usize {
        self.refs.len() - self.reserved_ids()
    }
    /// Pages one half can hand out (the whole capacity in a uniform pool's `[0]`).
    pub fn half_capacity(&self) -> [usize; 2] {
        if self.parity { [self.capacity() / 2; 2] } else { [self.capacity(), 0] }
    }
    /// Leading page indices never handed out (per half in a parity pool).
    pub fn reserved(&self) -> usize {
        self.reserved
    }
    fn reserved_ids(&self) -> usize {
        if self.parity { 2 * self.reserved } else { self.reserved }
    }
    pub fn free(&self) -> usize {
        self.free[0].len() + self.free[1].len()
    }
    /// Free pages per half (a uniform or tiered pool's are all in `[0]`).
    pub fn free_halves(&self) -> [usize; 2] {
        if self.resident.is_some() { [self.free(), 0] } else { [self.free[0].len(), self.free[1].len()] }
    }
    /// The free count a deferred admission compares against [`PoolExhausted::free`]: the whole
    /// pool's, or the scarcer half's in a parity pool.
    pub fn admission_free(&self) -> usize {
        if self.parity { self.free[0].len().min(self.free[1].len()) } else { self.free() }
    }
    pub fn used(&self) -> usize {
        self.capacity() - self.free()
    }
    /// Allocated pages per half.
    pub fn used_halves(&self) -> [usize; 2] {
        let capacity = self.half_capacity();
        let free = self.free_halves();
        [capacity[0] - free[0], capacity[1] - free[1]]
    }
    pub fn page_rows(&self) -> usize {
        self.page_rows
    }
    /// Progress relevant to admission, including a running placement releasing
    /// pages still held by an inactive snapshot (those pages become evictable
    /// without increasing the physical free-page count).
    pub fn release_epoch(&self) -> u64 {
        self.release_epoch
    }
    /// Pages a sequence of `tokens` tokens needs (at least one: a placement always has a page).
    pub fn pages_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.page_rows).max(1)
    }
    /// Pages whose every row is below `len` (the shareable prefix of a snapshot at `len`).
    pub fn full_pages(&self, len: usize) -> usize {
        len / self.page_rows
    }
    pub fn refs(&self, page: u32) -> u32 {
        self.refs[page as usize]
    }
    pub fn generation(&self, page: u32) -> u32 {
        self.generation[page as usize]
    }
    /// Pages referenced more than once.
    pub fn shared(&self) -> usize {
        self.refs.iter().filter(|&&r| r > 1).count()
    }

    /// Fresh pages for logical positions `positions` of a sequence.
    pub fn need_for(&self, positions: std::ops::Range<usize>) -> Need {
        let n = positions.len();
        if !self.parity {
            return Need([n, 0]);
        }
        let even = (positions.start..positions.end).filter(|j| j % 2 == 0).count();
        Need([even, n - even])
    }

    /// Whether `need` fits the free pages of each half (either tier of a tiered pool).
    pub fn fits(&self, need: Need) -> bool {
        let free = self.free_halves();
        need.0[0] <= free[0] && need.0[1] <= free[1]
    }

    /// `n` fresh pages for logical positions `0..n`, each with one reference; all or nothing.
    pub fn alloc(&mut self, n: usize) -> Result<Vec<u32>, PoolExhausted> {
        self.alloc_for(0..n)
    }

    /// Fresh pages for logical positions `positions` (a parity pool hands position `j` a page
    /// of parity `j % 2`; a tiered pool hands its fast pages to the highest positions), each
    /// with one reference; all or nothing.
    pub fn alloc_for(&mut self, positions: std::ops::Range<usize>) -> Result<Vec<u32>, PoolExhausted> {
        let need = self.need_for(positions.clone());
        if !self.fits(need) {
            return Err(self.exhausted(need));
        }
        let pages: Vec<u32> = if self.resident.is_some() {
            let slow = positions.len().saturating_sub(self.free[0].len());
            (0..positions.len()).map(|i| self.free[usize::from(i < slow)].pop().expect("checked")).collect()
        } else {
            positions.map(|j| self.free[if self.parity { j % 2 } else { 0 }].pop().expect("checked")).collect()
        };
        for &page in &pages {
            debug_assert_eq!(self.refs[page as usize], 0);
            self.refs[page as usize] = 1;
        }
        Ok(pages)
    }

    /// One more reference to each of `pages` (which must be allocated).
    pub fn share(&mut self, pages: &[u32]) {
        for &page in pages {
            let refs = &mut self.refs[page as usize];
            assert!(*refs > 0, "shared page {page} is free");
            *refs += 1;
        }
    }

    /// Drop one reference to each of `pages`; returns the pages that became free. The caller
    /// must have drained every queued reader and writer of a page before its last release.
    pub fn release(&mut self, pages: &[u32]) -> Vec<FreedPage> {
        let mut freed = Vec::new();
        for &page in pages {
            let refs = &mut self.refs[page as usize];
            assert!(*refs > 0, "released page {page} is free");
            *refs -= 1;
            if *refs == 0 {
                let generation = self.generation[page as usize];
                self.generation[page as usize] = generation.wrapping_add(1);
                let list = self.list(page);
                self.free[list].push(page);
                freed.push(FreedPage { index: page, generation });
            }
        }
        if !pages.is_empty() {
            self.release_epoch = self.release_epoch.wrapping_add(1);
        }
        freed
    }

    /// Fresh pages a fork of a snapshot at `len` into a sequence of `total` pages needs.
    pub fn fork_cost(&self, len: usize, total: usize) -> usize {
        self.fork_need(len, total).total()
    }

    /// [`RefPagePool::fork_cost`] per half.
    pub fn fork_need(&self, len: usize, total: usize) -> Need {
        self.fork_need_writable(len, total, len)
    }

    /// Fresh pages when rows at or beyond `write_frontier` may still change.
    pub fn fork_need_writable(&self, len: usize, total: usize, write_frontier: usize) -> Need {
        assert!(write_frontier <= len, "write frontier exceeds retained length");
        self.need_for(self.full_pages(write_frontier)..total.max(self.pages_for(len)))
    }

    /// Share complete immutable pages and eagerly copy a partial tail.
    pub fn fork(&mut self, source: &[u32], len: usize, total: usize) -> Result<Fork, PoolExhausted> {
        let (pages, copies) = self.fork_writable(source, len, total, len)?;
        debug_assert!(copies.len() <= 1);
        Ok(Fork { pages, copy: copies.into_iter().next() })
    }

    /// Share only pages strictly before the earliest writable retained row. Copies include
    /// complete retained pages when a predictor lags behind the target's committed frontier.
    pub fn fork_writable(&mut self, source: &[u32], len: usize, total: usize,
        write_frontier: usize) -> Result<(Vec<u32>, Vec<TailCopy>), PoolExhausted> {
        assert!(write_frontier <= len, "write frontier exceeds retained length");
        let shared = self.full_pages(write_frontier);
        let retained = len.div_ceil(self.page_rows);
        assert!(source.len() >= retained, "fork source shorter than {len} rows");
        let fresh = self.alloc_for(shared..total.max(self.pages_for(len)))?;
        self.share(&source[..shared]);
        let copies = (shared..retained).map(|i| {
            let copy = TailCopy { from: source[i], to: fresh[i - shared],
                rows: (len - i * self.page_rows).min(self.page_rows), owner: self.owner(source[i]) };
            debug_assert_eq!(self.owner(copy.from), self.owner(copy.to), "a copy keeps its owner");
            copy
        }).collect();
        let mut pages = source[..shared].to_vec();
        pages.extend(fresh);
        Ok((pages, copies))
    }

    fn exhausted(&self, need: Need) -> PoolExhausted {
        if !self.parity {
            return PoolExhausted { needed: need.0[0], free: self.free(), capacity: self.capacity() };
        }
        let half = usize::from(need.0[0] <= self.free[0].len());
        PoolExhausted { needed: need.0[half], free: self.free[half].len(), capacity: self.capacity() / 2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_is_all_or_nothing_and_release_bumps_generations() {
        let mut pool = RefPagePool::new(4, 64);
        let a = pool.alloc(3).unwrap();
        assert_eq!(a, vec![0, 1, 2]);
        assert_eq!(pool.alloc(2), Err(PoolExhausted { needed: 2, free: 1, capacity: 4 }));
        assert_eq!(pool.free(), 1);
        pool.share(&a[..1]);
        assert_eq!(pool.release(&a), vec![FreedPage { index: 1, generation: 0 }, FreedPage { index: 2, generation: 0 }]);
        assert_eq!((pool.refs(0), pool.generation(1)), (1, 1));
        assert_eq!(pool.release(&a[..1]), vec![FreedPage { index: 0, generation: 0 }]);
        assert_eq!(pool.free(), 4);
    }

    #[test]
    fn reserved_pages_are_never_handed_out() {
        let mut pool = RefPagePool::with_reserved(5, 64, 1);
        assert_eq!((pool.capacity(), pool.free(), pool.reserved()), (4, 4, 1));
        let a = pool.alloc(4).unwrap();
        assert_eq!(a, vec![1, 2, 3, 4]);
        assert_eq!(pool.alloc(1), Err(PoolExhausted { needed: 1, free: 0, capacity: 4 }));
        pool.release(&a);
        // Freed pages come back in any order; page 0 never does.
        let b = pool.alloc(4).unwrap();
        assert!(!b.contains(&0) && pool.used() == 4, "{b:?}");
        let fork = pool.fork(&b, 100, 3);
        assert!(fork.is_err(), "a fork needs two fresh pages and none is free");
        pool.release(&b);
        assert_eq!(pool.free(), 4);
        assert_eq!(RefPagePool::new(3, 64).capacity(), 3);
    }

    #[test]
    fn fork_shares_full_pages_and_copies_the_partial_tail() {
        let mut pool = RefPagePool::new(16, 64);
        let owner = pool.alloc(4).unwrap(); // 0..4, sequence of 200 rows
        let fork = pool.fork(&owner, 200, 6).unwrap();
        assert_eq!(&fork.pages[..3], &owner[..3]);
        assert_eq!(fork.pages.len(), 6);
        assert_eq!(fork.copy, Some(TailCopy { from: owner[3], to: fork.pages[3], rows: 8, owner: None }));
        assert_eq!((pool.refs(owner[0]), pool.refs(owner[3])), (2, 1));
        assert_eq!(pool.fork_cost(200, 6), 3);
        // Page-aligned: nothing to copy.
        let aligned = pool.fork(&owner, 192, 3).unwrap();
        assert_eq!((aligned.pages, aligned.copy), (owner[..3].to_vec(), None));
        // A fork never gets fewer pages than its rows need.
        assert_eq!(pool.fork(&owner, 130, 1).unwrap().pages.len(), 3);
        let before = pool.free();
        assert!(pool.fork(&owner, 64, 64).is_err());
        assert_eq!(pool.free(), before, "a failed fork holds nothing");
        assert_eq!(pool.refs(owner[0]), 4);
    }
}

#[cfg(test)]
mod tiered_tests {
    use super::*;

    #[test]
    fn tiered_pool_counts_both_tiers_and_puts_fast_pages_at_the_written_end() {
        let mut pool = RefPagePool::tiered(8, 64, 3);
        assert_eq!((pool.capacity(), pool.free(), pool.free_tiers(), pool.free_halves()), (8, 8, [3, 5], [8, 0]));
        assert!(!pool.is_parity() && pool.owner(5).is_none() && pool.resident() == 3);
        // Short allocations are wholly resident.
        let a = pool.alloc(2).unwrap();
        assert_eq!(a, vec![0, 1]);
        assert!(a.iter().all(|&p| pool.is_resident(p)));
        // A longer one gets the remaining fast page at its highest position.
        let b = pool.alloc_for(0..4).unwrap();
        assert_eq!(b, vec![3, 4, 5, 2]);
        assert_eq!(pool.free_tiers(), [0, 2]);
        assert!(pool.fits(pool.need_for(0..2)) && !pool.fits(pool.need_for(0..3)));
        assert_eq!(pool.alloc(3), Err(PoolExhausted { needed: 3, free: 2, capacity: 8 }));
        assert_eq!(pool.used_halves(), [6, 0]);
        // Freed pages return to their own tier.
        pool.release(&b);
        assert_eq!(pool.free_tiers(), [1, 5]);
        let fork = pool.fork(&a, 100, 3).unwrap();
        assert_eq!(&fork.pages[..1], &a[..1]);
        assert!(pool.is_resident(fork.pages[2]) && !pool.is_resident(fork.pages[1]));
        pool.release(&a);
        pool.release(&fork.pages);
        assert_eq!((pool.free(), pool.free_tiers()), (8, [3, 5]));
        // Every page may be slow, or every page fast.
        let mut slow = RefPagePool::tiered(4, 64, 0);
        assert_eq!(slow.alloc(4).unwrap(), vec![0, 1, 2, 3]);
        assert!(!slow.is_resident(0));
        let fast = RefPagePool::tiered(4, 64, 4);
        assert_eq!(fast.free_tiers(), [4, 0]);
        assert_eq!(RefPagePool::new(4, 64).free_tiers(), [4, 0]);
    }
}

#[cfg(test)]
mod writable_tests {
    use super::*;

    #[test]
    fn aligned_predictor_tail_is_private_and_multiple_copies_keep_parity() {
        for parity in [false, true] {
            let mut pool = if parity { RefPagePool::parity(16, 256, 0) } else { RefPagePool::new(16, 256) };
            let source = pool.alloc(3).unwrap();
            assert_eq!(pool.fork_need_writable(512, 3, 511), pool.need_for(1..3));
            let (pages, copies) = pool.fork_writable(&source, 512, 3, 511).unwrap();
            assert_eq!(pages[0], source[0]);
            assert_ne!(pages[1], source[1]);
            assert_eq!(copies, vec![TailCopy { from: source[1], to: pages[1], rows: 256,
                owner: pool.owner(source[1]) }]);
            let (wide, copies) = pool.fork_writable(&source, 513, 3, 255).unwrap();
            assert_eq!(copies.iter().map(|c| c.rows).collect::<Vec<_>>(), vec![256, 256, 1]);
            for c in copies { assert_eq!(pool.owner(c.from), pool.owner(c.to)); }
            pool.release(&source);
            pool.release(&pages);
            pool.release(&wide);
            assert_eq!(pool.free(), 16);
        }
    }
}
