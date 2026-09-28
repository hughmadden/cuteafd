//! Paged DeepSeek V4 caches shared by every sequence.
//!
//! Each layer owns one window pool (FP8 584-byte records, 256 per page), one
//! compressed pool per ratio (C4: 64 rows per page with a same-numbered index
//! page; C128: 2 rows per page) and compressor state arrays indexed by a
//! sequence's state slot. Sequences own page lists; the host maps logical
//! positions and groups to physical slots for every step.
//!
//! Window attention reads only the last 128 positions, so a sequence's window
//! pages form a ring large enough for one prefill chunk plus the window.
use super::metadata::{compressed_page_rows, SOURCE_PAGE_TOKENS, WINDOW};
use anyhow::{ensure, Context, Result};

/// Page and slot geometry of the pool (identical for every layer).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoolShape {
    /// Sequences that can hold state at once.
    pub sequences: usize,
    /// Window pages per sequence (ring).
    pub ring_pages: usize,
    pub c4_pages: usize,
    pub c128_pages: usize,
}

impl PoolShape {
    pub fn new(sequences: usize, max_context: usize, prefill_rows: usize, c4_pages: usize, c128_pages: usize) -> Self {
        let _ = max_context;
        Self {
            sequences,
            ring_pages: (prefill_rows + WINDOW).div_ceil(SOURCE_PAGE_TOKENS),
            c4_pages,
            c128_pages,
        }
    }

    pub fn window_pages(&self) -> usize {
        self.sequences * self.ring_pages
    }
}

/// One sequence's placement in the pool.
#[derive(Debug, Clone)]
pub(crate) struct Placement {
    pub state: usize,
    pub c4_pages: Vec<i32>,
    pub c128_pages: Vec<i32>,
    /// Tokens whose KV is in the caches.
    pub len: usize,
}

impl Placement {
    /// Physical window slot of `position` (ring of this sequence's pages).
    pub fn window_slot(&self, shape: &PoolShape, position: usize) -> i64 {
        let page = self.state * shape.ring_pages + (position / SOURCE_PAGE_TOKENS) % shape.ring_pages;
        (page * SOURCE_PAGE_TOKENS + position % SOURCE_PAGE_TOKENS) as i64
    }

    /// Physical compressed slot of group `group` for `ratio` (C4 also names the index slot).
    pub fn group_slot(&self, ratio: usize, group: usize) -> Result<i32> {
        let rows = compressed_page_rows(ratio);
        let pages = if ratio == 4 { &self.c4_pages } else { &self.c128_pages };
        let page = *pages.get(group / rows).with_context(|| format!("group {group} has no C{ratio} page"))?;
        Ok(page * rows as i32 + (group % rows) as i32)
    }
}

/// Host bookkeeping: free state slots and compressed pages.
pub(crate) struct PoolAllocator {
    free_states: Vec<usize>,
    free_c4: Vec<i32>,
    free_c128: Vec<i32>,
}

impl PoolAllocator {
    pub fn new(shape: PoolShape) -> Self {
        Self {
            free_states: (0..shape.sequences).rev().collect(),
            free_c4: (0..shape.c4_pages as i32).rev().collect(),
            free_c128: (0..shape.c128_pages as i32).rev().collect(),
        }
    }

    pub fn free_states(&self) -> usize {
        self.free_states.len()
    }

    /// Admits a sequence that may grow to `capacity` tokens, reserving every
    /// compressed page up front so decode never runs out mid-request.
    pub fn admit(&mut self, capacity: usize) -> Result<Placement> {
        let c4 = (capacity / 4).div_ceil(compressed_page_rows(4)).max(1);
        let c128 = (capacity / 128).div_ceil(compressed_page_rows(128)).max(1);
        ensure!(!self.free_states.is_empty(), "every sequence slot is busy");
        ensure!(self.free_c4.len() >= c4 && self.free_c128.len() >= c128,
            "compressed cache pages exhausted ({c4} C4 / {c128} C128 needed)");
        let state = self.free_states.pop().unwrap();
        let c4_pages = (0..c4).map(|_| self.free_c4.pop().unwrap()).collect();
        let c128_pages = (0..c128).map(|_| self.free_c128.pop().unwrap()).collect();
        Ok(Placement { state, c4_pages, c128_pages, len: 0 })
    }

    pub fn release(&mut self, placement: Placement) {
        self.free_states.push(placement.state);
        self.free_c4.extend(placement.c4_pages);
        self.free_c128.extend(placement.c128_pages);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_and_group_slots() -> Result<()> {
        let shape = PoolShape::new(4, 131_072, 4096, 64, 64);
        assert_eq!(shape.ring_pages, 17);
        let mut pool = PoolAllocator::new(shape);
        let a = pool.admit(1000)?;
        let b = pool.admit(1000)?;
        assert_ne!(a.state, b.state);
        // Positions one ring apart share a slot; the window never spans a ring.
        assert_eq!(a.window_slot(&shape, 5), a.window_slot(&shape, 5 + 17 * 256));
        assert_ne!(a.window_slot(&shape, 5), b.window_slot(&shape, 5));
        assert_eq!(a.c4_pages.len(), 4);
        assert_eq!(a.group_slot(4, 65)?, a.c4_pages[1] * 64 + 1);
        assert_eq!(a.group_slot(128, 3)?, a.c128_pages[1] * 2 + 1);
        pool.release(a);
        assert_eq!(pool.free_states(), 3);
        Ok(())
    }
}
