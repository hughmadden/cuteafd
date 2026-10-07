//! Host ownership for immutable source prefixes. Active request tables own one
//! reference per page; retained prefixes release their references on eviction.
use std::collections::{HashMap, HashSet};
use std::{cell::RefCell, rc::Rc};

pub(super) struct PagePool {
    pub free: Vec<u32>,
    references: Vec<usize>,
    /// Bumped when a page's last reference goes, so a reused index is a new identity.
    generations: Vec<u32>,
}
impl PagePool {
    pub fn new(pages: usize) -> Self {
        Self {
            free: (0..pages as u32).rev().collect(),
            references: vec![0; pages],
            generations: vec![0; pages],
        }
    }
    pub fn capacity(&self) -> usize {
        self.references.len()
    }
    pub fn generation(&self, page: u32) -> u32 {
        self.generations[page as usize]
    }
    /// Take `count` free pages, each with one reference owned by the caller; `None` if fewer are free.
    pub fn allocate(&mut self, count: usize) -> Option<Vec<u32>> {
        if self.free.len() < count {
            return None;
        }
        let pages = self.free.split_off(self.free.len() - count);
        self.retain(&pages);
        Some(pages)
    }
    pub fn references(&self, page: u32) -> usize {
        self.references[page as usize]
    }
    pub fn retain(&mut self, pages: &[u32]) {
        for &page in pages {
            self.references[page as usize] += 1;
        }
    }
    pub fn release(&mut self, pages: &[u32]) {
        for &page in pages {
            let count = &mut self.references[page as usize];
            assert!(*count > 0, "source page released without ownership");
            *count -= 1;
            if *count == 0 {
                self.generations[page as usize] += 1;
                self.free.push(page);
            }
        }
    }
}

/// What evicting a retained prefix gains one append transaction in its pool, least first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Gain {
    /// Live request tables hold every page, and none is a tail the transaction appends to.
    Nothing,
    /// Live tables hold every page, so none is freed, but the prefix shares a partial tail the
    /// transaction appends to: its reference makes the append copy that page first.
    Copies,
    /// Some page is held only by retained prefixes; evicting can free it.
    Pages,
}

/// One source pool under an append transaction: the pages live request tables hold, the
/// partial tail pages the transaction appends into, and references to those tails counted as
/// dropped (`release`). It belongs to one pool, and a prefix of another pool is refused.
#[derive(Clone)]
pub(crate) struct Pressure {
    pool: Rc<RefCell<PagePool>>,
    held: HashSet<u32>,
    tails: HashSet<u32>,
    released: HashMap<u32, usize>,
}
impl Pressure {
    pub(super) fn new(pool: &Rc<RefCell<PagePool>>, tables: &[Vec<u32>], tails: HashSet<u32>) -> Self {
        Self {
            pool: Rc::clone(pool),
            held: tables.iter().flatten().copied().collect(),
            tails,
            released: HashMap::new(),
        }
    }
    pub(super) fn is_of(&self, pool: &Rc<RefCell<PagePool>>) -> bool {
        Rc::ptr_eq(&self.pool, pool)
    }
    /// References to `page` counted as dropped.
    pub(super) fn released(&self, page: u32) -> usize {
        self.released.get(&page).copied().unwrap_or(0)
    }
    /// Count `prefix`'s references to the appended tails as dropped.
    pub fn release(&mut self, prefix: &SourcePrefix) -> anyhow::Result<()> {
        anyhow::ensure!(self.is_of(&prefix.pool), "source prefix released against another source pool");
        for &page in prefix.pages.iter().filter(|page| self.tails.contains(page)) {
            *self.released.entry(page).or_default() += 1;
        }
        Ok(())
    }
}

pub(crate) struct SourcePrefix {
    pub(super) pool: Rc<RefCell<PagePool>>,
    pub(super) pages: Vec<u32>,
    pub(super) rows: usize,
}
impl SourcePrefix {
    pub fn pages(&self) -> &[u32] {
        &self.pages
    }
    /// What evicting this prefix gains the transaction behind `pressure`. References from
    /// other snapshots do not count as held. An error when `pressure` is another pool's.
    pub fn gain(&self, pressure: &Pressure) -> anyhow::Result<Gain> {
        anyhow::ensure!(pressure.is_of(&self.pool), "source prefix weighed against another source pool");
        Ok(if !self.pages.iter().all(|page| pressure.held.contains(page)) {
            Gain::Pages
        } else if self.pages.iter().any(|page| pressure.tails.contains(page)) {
            Gain::Copies
        } else {
            Gain::Nothing
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Retain a shorter initialized frontier after the original request was
    /// released. Future rows in its physical tail remain owned by the original
    /// snapshot; an appending branch must still use copy-on-write.
    pub fn truncate(&self, rows: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            rows <= self.rows,
            "source prefix truncation exceeds initialized rows"
        );
        let pages = self.pages[..rows.div_ceil(super::PAGE_ROWS)].to_vec();
        self.pool.borrow_mut().retain(&pages);
        Ok(Self {
            pool: Rc::clone(&self.pool),
            pages,
            rows,
        })
    }
}
impl Drop for SourcePrefix {
    fn drop(&mut self) {
        self.pool.borrow_mut().release(&self.pages);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_readmission_keeps_a_snapshot_larger_than_half_the_pool() {
        use cuteafd_core::prefix::{Retention, SnapshotKind};
        let pool = Rc::new(RefCell::new(PagePool::new(10)));
        let pages = pool.borrow_mut().allocate(7).unwrap();
        let mut retained = Retention::new(2);
        retained.bank_mut(SnapshotKind::Prompt).insert(&[1, 2, 3], SourcePrefix {
            pool: pool.clone(), pages: pages.clone(), rows: 7 * super::super::PAGE_ROWS,
        });
        // The lookup borrow pins the source until the fresh lease owns its pages,
        // as in PrefixCache::restore. Re-admission needs only new output pages.
        let (_, _, hit) = retained.lookup_reusable(&[1, 2, 3]).unwrap();
        pool.borrow_mut().retain(hit.pages());
        let held = Pressure::new(&pool, &[hit.pages().to_vec()], HashSet::new());
        assert!(retained.evict_one_where(&|prefix| prefix.gain(&held).unwrap() != Gain::Pages).is_none());
        let output = pool.borrow_mut().allocate(2).unwrap();
        assert!(retained.lookup_reusable(&[1, 2, 3]).is_some());
        assert_eq!(pool.borrow().free.len(), 1);
        pool.borrow_mut().release(&output);
        pool.borrow_mut().release(&pages);
        let idle = Pressure::new(&pool, &[], HashSet::new());
        let (_, snapshot) = retained.evict_one_where(&|prefix| prefix.gain(&idle).unwrap() != Gain::Pages).unwrap();
        drop(snapshot);
        assert_eq!(pool.borrow().free.len(), 10);
    }

    #[test]
    fn sharing_with_other_snapshots_does_not_protect_an_inactive_snapshot() {
        let pool = Rc::new(RefCell::new(PagePool::new(4)));
        let pages = pool.borrow_mut().allocate(3).unwrap();
        let first = SourcePrefix { pool: pool.clone(), pages, rows: 3 * super::super::PAGE_ROWS };
        let second = first.truncate(first.rows()).unwrap();
        assert_eq!(first.gain(&Pressure::new(&pool, &[], HashSet::new())).unwrap(), Gain::Pages);
        let active = Pressure::new(&pool, &[first.pages()[..2].to_vec()], HashSet::new());
        assert_eq!(first.gain(&active).unwrap(), Gain::Pages, "one unshared page is reclaimable");
        drop(first);
        drop(second);
        assert_eq!(pool.borrow().free.len(), 4);
    }

    #[test]
    fn a_held_prefix_gains_copies_only_where_it_shares_an_appended_tail() {
        let rows = super::super::PAGE_ROWS;
        let pool = Rc::new(RefCell::new(PagePool::new(8)));
        // A request table of two full pages and a partial tail, restored from `source`.
        let table = pool.borrow_mut().allocate(3).unwrap();
        pool.borrow_mut().retain(&table);
        let source = SourcePrefix { pool: pool.clone(), pages: table.clone(), rows: 2 * rows + 100 };
        let ancestor = source.truncate(2 * rows).unwrap();
        let tail = HashSet::from([table[2]]);
        let appending = Pressure::new(&pool, &[table.clone()], tail.clone());
        assert_eq!(source.gain(&appending).unwrap(), Gain::Copies);
        assert_eq!(ancestor.gain(&appending).unwrap(), Gain::Nothing, "full pages are never copied");
        let idle = Pressure::new(&pool, &[table.clone()], HashSet::new());
        assert_eq!(source.gain(&idle).unwrap(), Gain::Nothing, "no append, no copy");
        // A second sharer of the tail (a turn snapshot at the same frontier) counts as well.
        let twin = source.truncate(source.rows()).unwrap();
        let mut released = appending.clone();
        for prefix in [&source, &ancestor, &twin] {
            released.release(prefix).unwrap();
        }
        assert_eq!((released.released(table[2]), released.released(table[0])), (2, 0));
        assert_eq!(appending.released(table[2]), 0, "a clone counts its own releases");
        drop((source, ancestor, twin));
        pool.borrow_mut().release(&table);
        assert_eq!(pool.borrow().free.len(), 8);
    }

    #[test]
    fn a_snapshots_sources_are_weighed_against_their_own_pools_in_order() {
        use crate::families::deepseek_v41::v41_compressor::{release_snapshot_tails, snapshot_gain, CompressorPrefix};
        let rows = super::super::PAGE_ROWS + 10;
        // Four compressed sources, each a pool whose request table was restored from the snapshot.
        let pools: Vec<_> = (0..4).map(|_| Rc::new(RefCell::new(PagePool::new(4)))).collect();
        let tables: Vec<Vec<u32>> = pools.iter().map(|pool| {
            let table = pool.borrow_mut().allocate(2).unwrap();
            pool.borrow_mut().retain(&table);
            table
        }).collect();
        let sources: Vec<_> = pools.iter().zip(&tables).map(|(pool, table)| CompressorPrefix::from_parts(1, 2 * rows as u64,
            SourcePrefix { pool: pool.clone(), pages: table.clone(), rows })).collect();
        let mut pressure: Vec<_> = pools.iter().zip(&tables)
            .map(|(pool, table)| Pressure::new(pool, &[table.clone()], HashSet::from([table[1]]))).collect();
        assert_eq!(snapshot_gain(&sources, &pressure).unwrap(), Gain::Copies);
        // The most any source gains: one source whose table no longer holds its pages frees them.
        let mut freeing = pressure.clone();
        freeing[2] = Pressure::new(&pools[2], &[], HashSet::new());
        assert_eq!(snapshot_gain(&sources, &freeing).unwrap(), Gain::Pages);
        // A reordering on either side, or a missing source, is an error rather than a wrong answer.
        pressure.swap(1, 2);
        assert!(snapshot_gain(&sources, &pressure).is_err());
        assert!(release_snapshot_tails(&sources, &mut pressure).is_err());
        assert_eq!(pressure[0].released(tables[0][1]), 0, "a refused release counts nothing");
        pressure.swap(1, 2);
        assert!(snapshot_gain(&sources, &pressure[..3]).is_err());
        release_snapshot_tails(&sources, &mut pressure).unwrap();
        for (pressure, table) in pressure.iter().zip(&tables) {
            assert_eq!((pressure.released(table[1]), pressure.released(table[0])), (1, 0));
        }
        drop(sources);
        for (pool, table) in pools.iter().zip(&tables) {
            pool.borrow_mut().release(table);
            assert_eq!(pool.borrow().free.len(), 4);
        }
    }

    #[test]
    fn a_prefix_is_never_weighed_against_another_pool() {
        let pool = Rc::new(RefCell::new(PagePool::new(2)));
        let other = Rc::new(RefCell::new(PagePool::new(2)));
        let pages = pool.borrow_mut().allocate(1).unwrap();
        let prefix = SourcePrefix { pool: pool.clone(), pages: pages.clone(), rows: 10 };
        let mut foreign = Pressure::new(&other, &[pages.clone()], HashSet::from([pages[0]]));
        assert!(prefix.gain(&foreign).is_err());
        assert!(foreign.release(&prefix).is_err());
        assert!(prefix.gain(&Pressure::new(&pool, &[pages], HashSet::new())).is_ok());
    }

    #[test]
    fn prefix_eviction_frees_only_pages_without_active_owners() {
        let pool = Rc::new(RefCell::new(PagePool::new(3)));
        let pages = {
            let mut pool = pool.borrow_mut();
            vec![pool.free.pop().unwrap(), pool.free.pop().unwrap()]
        };
        pool.borrow_mut().retain(&pages);
        pool.borrow_mut().retain(&pages);
        let prefix = SourcePrefix {
            pool: Rc::clone(&pool),
            pages: pages.clone(),
            rows: 300,
        };
        assert!(pool.borrow().references(pages[0]) > 1);
        pool.borrow_mut().release(&pages[..1]);
        assert_eq!(pool.borrow().free.len(), 1);
        drop(prefix);
        assert_eq!(pool.borrow().free.len(), 2);
        assert_eq!(pool.borrow().references(pages[1]), 1);
        pool.borrow_mut().release(&pages[1..]);
        let mut free = pool.borrow().free.clone();
        free.sort_unstable();
        assert_eq!(free, vec![0, 1, 2]);
    }
}
