//! Append-only writes: validate private tails before dispatch, publish after drain.
use super::{prefix::Placement, MiniMaxError};
use cuteafd_engine::prefix::RefPagePool;
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
struct Sequence {
    slot: usize,
    pages: Vec<u32>,
    base: usize,
    tokens: Vec<u32>,
}
#[derive(Debug)]
pub(super) struct Step { sequences: Vec<Sequence>, complete: bool }

impl Step {
    pub fn begin(sequences: &[(&mut Placement, &[u32])], pool: &RefPagePool,
        slots: usize, capacity: usize, context: usize, pages: usize) -> Result<Self, MiniMaxError> {
        if sequences.is_empty() || pool.is_parity() || pool.page_rows() != 64
            || pool.capacity().checked_add(pool.reserved()) != Some(pages) {
            return Err(MiniMaxError::Invalid("transaction pool geometry"));
        }
        let mut seen = BTreeSet::new();
        let mut referenced = BTreeSet::new();
        let mut writable = BTreeSet::new();
        let mut total = 0usize;
        let mut retained = Vec::new();
        for (p, tokens) in sequences {
            let end = p.committed.checked_add(tokens.len()).ok_or(MiniMaxError::Overflow("frontier"))?;
            total = total.checked_add(tokens.len()).ok_or(MiniMaxError::Overflow("wave rows"))?;
            if p.sequence >= slots || !seen.insert(p.sequence) || tokens.is_empty() || end > context
                || total > capacity || end.div_ceil(64) > p.pages.len() || p.pages.len() > context.div_ceil(64) {
                return Err(MiniMaxError::Invalid("transaction sequence extent"));
            }
            let mut unique = BTreeSet::new();
            for (logical, &page) in p.pages.iter().enumerate() {
                if (page as usize) < pool.reserved() || page as usize >= pages || !unique.insert(page) {
                    return Err(MiniMaxError::Invalid("transaction page"));
                }
                let refs = pool.refs(page);
                let write = logical >= p.committed / 64;
                if refs == 0 || writable.contains(&page) || (write && (refs != 1 || referenced.contains(&page))) {
                    return Err(MiniMaxError::Invalid("transaction writable page alias"));
                }
                referenced.insert(page);
                if write { writable.insert(page); }
            }
            retained.push(Sequence { slot: p.sequence, pages: p.pages.clone(), base: p.committed, tokens: tokens.to_vec() });
        }
        Ok(Self { sequences: retained, complete: false })
    }
    pub fn completed(&mut self) { self.complete = true; }
    pub fn publish(self, sequences: &mut [(&mut Placement, usize)]) -> Result<(), MiniMaxError> {
        if !self.complete || sequences.len() != self.sequences.len() {
            return Err(MiniMaxError::Invalid("transaction completion evidence"));
        }
        for (saved, (p, rows)) in self.sequences.iter().zip(sequences.iter()) {
            if saved.slot != p.sequence || saved.pages != p.pages || *rows > saved.tokens.len()
                || !(saved.base..=saved.base + saved.tokens.len()).contains(&p.committed) {
                return Err(MiniMaxError::Invalid("transaction publication identity"));
            }
        }
        for (saved, (p, rows)) in self.sequences.iter().zip(sequences.iter_mut()) {
            p.committed = saved.base + *rows;
        }
        Ok(())
    }
    pub fn matches(&self, slot: usize, base: usize, tokens: &[u32]) -> bool {
        self.sequences.iter().any(|s| s.slot == slot && s.base == base && s.tokens == tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_engine::media::RequestMedia;
    fn placement(pages: Vec<u32>, sequence: usize, committed: usize) -> Placement {
        Placement { pages, sequence, committed, media: RequestMedia::new(Vec::new(), 6144, 0).unwrap() }
    }
    #[test]
    fn minimax_append_completion_cannot_publish_unexecuted_or_wrong_request() {
        let mut pool = RefPagePool::new(8, 64);
        let pages = pool.alloc(2).unwrap();
        let mut p = placement(pages, 0, 0);
        let step = Step::begin(&[(&mut p, &[1, 2])], &pool, 2, 64, 128, 8).unwrap();
        assert!(step.publish(&mut [(&mut p, 2)]).is_err());
        assert_eq!(p.committed, 0);
        let mut step = Step::begin(&[(&mut p, &[1, 2])], &pool, 2, 64, 128, 8).unwrap();
        step.completed();
        assert!(step.matches(0, 0, &[1, 2]));
        assert!(!step.matches(0, 0, &[1, 3]));
        step.publish(&mut [(&mut p, 2)]).unwrap();
        assert_eq!(p.committed, 2);
        let mut step = Step::begin(&[(&mut p, &[3])], &pool, 2, 64, 128, 8).unwrap();
        step.completed(); p.sequence = 1;
        assert!(step.publish(&mut [(&mut p, 1)]).is_err());
        assert_eq!(p.committed, 2);
    }
    #[test]
    fn minimax_append_rejects_aliases_before_writes() {
        let mut pool = RefPagePool::new(8, 64);
        let pages = pool.alloc(2).unwrap();
        let mut p = placement(pages.clone(), 0, 0);
        let mut q = placement(pages, 1, 0);
        assert!(Step::begin(&[(&mut p, &[1]), (&mut q, &[2])], &pool, 2, 64, 128, 8).is_err());
        p.pages[0] = 9;
        assert!(Step::begin(&[(&mut p, &[1])], &pool, 2, 64, 128, 8).is_err());
    }
}
