//! Append-only writes: the shared wave checks before dispatch, publication after drain.
use super::{execution::{BatchView, RequestView}, prefix::Placement};
use crate::shared::prefix::append::{self, SequenceRows};
use anyhow::{ensure, Result};
use cuteafd_engine::prefix::RefPagePool;

#[derive(Debug, Clone)]
struct Sequence {
    slot: usize,
    start: usize,
    pages: Vec<u32>,
    base: usize,
    tokens: Vec<u32>,
}
impl Sequence {
    fn view(&self) -> SequenceRows<'_> {
        SequenceRows { sequence: self.slot as u64, start: self.start, rows: self.tokens.len(), base: self.base,
            pages: &self.pages }
    }
}
#[derive(Debug)]
pub(super) struct Step { sequences: Vec<Sequence>, complete: bool }

impl Step {
    pub fn begin(sequences: &[(&mut Placement, &[u32])], pool: &RefPagePool,
        slots: usize, capacity: usize, context: usize, pages: usize) -> Result<Self> {
        let mut start = 0;
        let retained: Vec<_> = sequences.iter().map(|(p, tokens)| {
            let sequence = Sequence { slot: p.sequence, start, pages: p.pages.clone(), base: p.committed,
                tokens: tokens.to_vec() };
            start += tokens.len();
            sequence
        }).collect();
        append::validate(&retained.iter().map(Sequence::view).collect::<Vec<_>>(), pool, slots, capacity, context, pages)?;
        Ok(Self { sequences: retained, complete: false })
    }
    /// The validated wave as the kernels see it: its packed rows' positions
    /// and physical KV slots, and each sequence's page table.
    pub fn batch(&self, pool_pages: usize) -> BatchView {
        let mut batch = BatchView { positions: Vec::new(), slots: Vec::new(), requests: Vec::new(), pool_pages };
        for s in &self.sequences {
            batch.requests.push(RequestView { row_start: s.start, rows: s.tokens.len(), start_position: s.base,
                pages: s.pages.iter().map(|&p| p as i32).collect() });
            for position in s.base..s.base + s.tokens.len() {
                batch.positions.push(position as i64);
                batch.slots.push(i64::from(s.pages[position / 64]) * 64 + (position % 64) as i64);
            }
        }
        batch
    }
    pub fn completed(&mut self) { self.complete = true; }
    pub fn publish(self, sequences: &mut [(&mut Placement, usize)]) -> Result<()> {
        ensure!(self.complete, "MiniMax publication before its wave completed");
        append::check_publication(self.sequences.iter().map(Sequence::view), sequences)?;
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
    fn minimax_wave_batch_takes_causal_positions_and_physical_slots() -> Result<()> {
        let mut pool = RefPagePool::new(8, 64);
        pool.alloc(5)?;
        // Two rows from 63 over pages 0 and 2, one row at 4 on page 4.
        let (mut a, mut b) = (placement(vec![0, 2], 0, 63), placement(vec![4], 1, 4));
        let batch = Step::begin(&[(&mut a, &[1, 2]), (&mut b, &[3])], &pool, 2, 64, 128, 8)?.batch(8);
        assert_eq!((batch.positions.as_slice(), batch.slots.as_slice()), ([63, 64, 4].as_slice(), [63, 128, 260].as_slice()));
        assert_eq!(batch.requests.iter().map(|r| (r.row_start, r.rows, r.start_position)).collect::<Vec<_>>(),
            [(0, 2, 63), (2, 1, 4)]);
        assert_eq!(batch.validate(64, 128)?, 3);
        assert!(batch.validate(2, 128).is_err() && batch.validate(64, 64).is_err());
        // A page one request writes and another lists needs copy-on-write.
        let mut c = placement(vec![0], 1, 4);
        assert!(Step::begin(&[(&mut a, &[1, 2]), (&mut c, &[3])], &pool, 2, 64, 128, 8).is_err());
        // A complete page both only read is shared; each writes its own page.
        let (mut x, mut y) = (placement(vec![0, 2], 0, 64), placement(vec![0, 4], 1, 64));
        let batch = Step::begin(&[(&mut x, &[1]), (&mut y, &[2])], &pool, 2, 64, 128, 8)?.batch(8);
        assert_eq!(batch.slots, [128, 256]);
        Ok(())
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
    #[test]
    fn minimax_publication_takes_the_accepted_frontier_only() {
        let mut pool = RefPagePool::new(8, 64);
        let mut p = placement(pool.alloc(2).unwrap(), 0, 10);
        // The scheduler advanced past a kept prefix of 2: publish at the kept frontier.
        let mut step = Step::begin(&[(&mut p, &[1, 2, 3])], &pool, 1, 64, 128, 8).unwrap();
        step.completed(); p.committed = 12;
        step.publish(&mut [(&mut p, 2)]).unwrap();
        assert_eq!(p.committed, 12);
        // A finished emission without accepted rows returns the frontier to the wave base.
        let mut step = Step::begin(&[(&mut p, &[4, 5])], &pool, 1, 64, 128, 8).unwrap();
        step.completed(); p.committed = 14;
        step.publish(&mut [(&mut p, 0)]).unwrap();
        assert_eq!(p.committed, 12);
        // A frontier the accepted rows do not explain is refused.
        let mut step = Step::begin(&[(&mut p, &[6, 7, 8])], &pool, 1, 64, 128, 8).unwrap();
        step.completed(); p.committed = 13;
        assert!(step.publish(&mut [(&mut p, 2)]).is_err());
        assert_eq!(p.committed, 13);
    }
}
