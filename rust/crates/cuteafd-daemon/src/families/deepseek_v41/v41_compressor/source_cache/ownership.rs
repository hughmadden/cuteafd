//! Host ownership for immutable source prefixes. Active request tables own one
//! reference per page; retained prefixes release their references on eviction.
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
    pub fn shared(&self, page: u32) -> bool {
        self.references(page) > 1
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

pub(crate) struct SourcePrefix {
    pub(super) pool: Rc<RefCell<PagePool>>,
    pub(super) pages: Vec<u32>,
    pub(super) rows: usize,
}
impl SourcePrefix {
    pub fn pages(&self) -> &[u32] {
        &self.pages
    }
    /// Whether dropping this prefix cannot release any source page because live
    /// request tables own every page. References from other snapshots do not count.
    pub fn held_by(&self, active: &std::collections::HashSet<u32>) -> bool {
        self.pages.iter().all(|page| active.contains(page))
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
        let active = hit.pages().iter().copied().collect();
        assert!(retained.evict_one_where(&|prefix| prefix.held_by(&active)).is_none());
        let output = pool.borrow_mut().allocate(2).unwrap();
        assert!(retained.lookup_reusable(&[1, 2, 3]).is_some());
        assert_eq!(pool.borrow().free.len(), 1);
        pool.borrow_mut().release(&output);
        pool.borrow_mut().release(&pages);
        let (_, snapshot) = retained.evict_one_where(&|prefix| prefix.held_by(&Default::default())).unwrap();
        drop(snapshot);
        assert_eq!(pool.borrow().free.len(), 10);
    }

    #[test]
    fn sharing_with_other_snapshots_does_not_protect_an_inactive_snapshot() {
        let pool = Rc::new(RefCell::new(PagePool::new(4)));
        let pages = pool.borrow_mut().allocate(3).unwrap();
        let first = SourcePrefix { pool: pool.clone(), pages, rows: 3 * super::super::PAGE_ROWS };
        let second = first.truncate(first.rows()).unwrap();
        assert!(!first.held_by(&Default::default()));
        let active = first.pages()[..2].iter().copied().collect();
        assert!(!first.held_by(&active), "one unshared page is reclaimable");
        drop(first);
        drop(second);
        assert_eq!(pool.borrow().free.len(), 4);
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
        assert!(pool.borrow().shared(pages[0]));
        pool.borrow_mut().release(&pages[..1]);
        assert_eq!(pool.borrow().free.len(), 1);
        drop(prefix);
        assert_eq!(pool.borrow().free.len(), 2);
        assert!(!pool.borrow().shared(pages[1]));
        pool.borrow_mut().release(&pages[1..]);
        let mut free = pool.borrow().free.clone();
        free.sort_unstable();
        assert_eq!(free, vec![0, 1, 2]);
    }
}
