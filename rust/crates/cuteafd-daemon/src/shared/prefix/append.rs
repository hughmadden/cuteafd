//! Append-only waves over the shared page pool: a wave's new rows write only
//! pages their sequence holds alone, checked before any input write, and a
//! drained wave publishes only on the evidence it was checked with.
use crate::shared::qwen::scheduler::Placement;
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::RefPagePool;
use std::collections::BTreeSet;

/// One contiguous query slice of a wave: `rows` new rows from position
/// `base`, packed at wave row `start`, in native state slot `sequence` over
/// its admitted `pages`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SequenceRows<'p> {
    pub sequence: u64,
    pub start: usize,
    pub rows: usize,
    pub base: usize,
    pub pages: &'p [u32],
}

/// Checks a wave against the admitted pool (`pages` physical 64-row pages)
/// before any input write: distinct admitted slots, row slices packed back
/// to back within `capacity` rows and `context`, every page allocated, in
/// bounds and listed once per sequence, and every page at or after a
/// sequence's base page (one it writes) referenced by that sequence alone.
/// Returns the wave's rows.
pub(crate) fn validate(sequences: &[SequenceRows<'_>], pool: &RefPagePool,
    slots: usize, capacity: usize, context: usize, pages: usize,
) -> Result<usize> {
    ensure!(!sequences.is_empty() && !pool.is_parity() && pool.page_rows() == 64
        && pool.capacity().checked_add(pool.reserved()) == Some(pages),
        "append wave pool geometry differs from native admission");
    let mut slots_seen = BTreeSet::new();
    let mut referenced = BTreeSet::new();
    let mut writable = BTreeSet::new();
    let mut total = 0usize;
    for sequence in sequences {
        ensure!(sequence.sequence < slots as u64 && slots_seen.insert(sequence.sequence),
            "append wave duplicate or unadmitted sequence slot");
        let end = sequence.base.checked_add(sequence.rows).context("append wave frontier overflow")?;
        ensure!(sequence.rows > 0 && sequence.start == total && end <= context,
            "append wave row slice or context extent");
        total = total.checked_add(sequence.rows).context("append wave rows overflow")?;
        ensure!(total <= capacity && end.div_ceil(64) <= sequence.pages.len()
            && sequence.pages.len() <= context.div_ceil(64),
            "append wave exceeds row or page-table admission");
        let mut unique = BTreeSet::new();
        for (logical, &page) in sequence.pages.iter().enumerate() {
            // RefPagePool::refs indexes directly, so bounds precede every read.
            ensure!((page as usize) >= pool.reserved() && (page as usize) < pages
                && unique.insert(page), "append wave invalid or duplicate physical page");
            let refs = pool.refs(page);
            ensure!(refs > 0, "append wave references an unallocated page");
            let write = logical >= sequence.base / 64;
            ensure!(!writable.contains(&page) && (!write || (!referenced.contains(&page) && refs == 1)),
                "append wave writable page is shared or aliases another sequence");
            referenced.insert(page);
            if write { writable.insert(page); }
        }
    }
    Ok(total)
}

/// Checks a drained wave's publication against the wave it validated: the
/// same slots and pages in order, at most each sequence's rows accepted, and
/// the scheduler's frontier at the base or at base + accepted. With nothing
/// accepted the frontier may stand anywhere in the wave (a failed or finished
/// emission advances it); publication returns it to the base. The caller then
/// commits base + accepted.
pub(crate) fn check_publication<'w, P: Placement>(wave: impl ExactSizeIterator<Item = SequenceRows<'w>>,
    placements: &[(&mut P, usize)]) -> Result<()> {
    ensure!(wave.len() == placements.len(), "append publication inventory differs from the wave");
    for (sequence, (placement, accepted)) in wave.zip(placements) {
        let committed = placement.len();
        ensure!(u64::try_from(placement.slot()).ok() == Some(sequence.sequence)
            && placement.units() == sequence.pages && *accepted <= sequence.rows
            && (committed == sequence.base || committed == sequence.base + accepted
                || (*accepted == 0 && (sequence.base..=sequence.base + sequence.rows).contains(&committed))),
            "append publication frontier differs from retained evidence");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_admission_shares_only_immutable_pages_and_preserves_row_slices() {
        let mut pool = RefPagePool::new(16, 64);
        let source = pool.alloc(2).unwrap();
        let first = pool.fork(&source, 65, 3).unwrap();
        let second = pool.fork(&source, 65, 3).unwrap();
        let sequences = [
            SequenceRows { sequence: 3, start: 0, rows: 1, base: 65, pages: &first.pages },
            SequenceRows { sequence: 1, start: 1, rows: 17, base: 65, pages: &second.pages },
        ];
        assert_eq!(validate(&sequences, &pool, 4, 18, 192, 16).unwrap(), 18);
        assert_eq!(first.pages[0], second.pages[0]);
        assert_ne!(first.pages[1], second.pages[1]);
        assert!(first.copy.is_some() && second.copy.is_some());
        let mut invalid = sequences;
        invalid[1].pages = &first.pages;
        assert!(validate(&invalid, &pool, 4, 18, 192, 16).is_err());
        let readonly = SequenceRows { sequence: 0, start: 0, rows: 1, base: 128, pages: &first.pages };
        let writer = SequenceRows { sequence: 1, start: 1, rows: 1, base: 65, pages: &first.pages };
        assert!(validate(&[readonly, writer], &pool, 4, 18, 192, 16).is_err());
        assert!(validate(&[SequenceRows { start: 0, ..writer }, SequenceRows { start: 1, ..readonly }],
            &pool, 4, 18, 192, 16).is_err());
        let reversed = [SequenceRows { start: 0, ..sequences[1] }, SequenceRows { start: 17, ..sequences[0] }];
        assert_eq!(validate(&reversed, &pool, 4, 18, 192, 16).unwrap(), 18);
        pool.release(&source);
        pool.release(&first.pages);
        pool.release(&second.pages);
        assert_eq!(pool.free(), 16);
    }

    #[test]
    fn wave_admission_rejects_invalid_inventory_before_pool_indexing() {
        let mut pool = RefPagePool::new(8, 64);
        let pages = pool.alloc(3).unwrap();
        let valid = SequenceRows { sequence: 0, start: 0, rows: 2, base: 63, pages: &pages };
        assert_eq!(validate(&[valid], &pool, 2, 16, 192, 8).unwrap(), 2);
        for invalid in [
            SequenceRows { sequence: 2, ..valid },
            SequenceRows { start: 1, ..valid },
            SequenceRows { rows: 0, ..valid },
            SequenceRows { rows: 17, ..valid },
            SequenceRows { base: usize::MAX, ..valid },
            SequenceRows { base: 191, ..valid },
            SequenceRows { pages: &pages[..1], ..valid },
            SequenceRows { pages: &[0, 0], ..valid },
            SequenceRows { pages: &[0, 7], ..valid },
            SequenceRows { pages: &[0, 8], ..valid },
            SequenceRows { pages: &[0, u32::MAX], ..valid },
        ] {
            assert!(validate(&[invalid], &pool, 2, 16, 192, 8).is_err(), "{invalid:?}");
        }
        assert!(validate(&[], &pool, 2, 16, 192, 8).is_err());
        assert!(validate(&[valid], &pool, 2, 16, 192, 9).is_err());
        assert!(validate(&[valid, SequenceRows { start: 2, ..valid }], &pool, 2, 16, 192, 8).is_err());
        pool.share(&pages[1..2]);
        assert!(validate(&[valid], &pool, 2, 16, 192, 8).is_err());
        pool.release(&pages[1..2]);
        let mut reserved = RefPagePool::with_reserved(8, 64, 1);
        let allocation = reserved.alloc(2).unwrap();
        let reserved_valid = SequenceRows { pages: &allocation, ..valid };
        validate(&[reserved_valid], &reserved, 2, 16, 192, 8).unwrap();
        assert!(validate(&[SequenceRows { pages: &[0, 1], ..valid }], &reserved, 2, 16, 192, 8).is_err());
        let wrong_rows = RefPagePool::new(8, 32);
        assert!(validate(&[valid], &wrong_rows, 2, 16, 192, 8).is_err());
        let parity = RefPagePool::parity(8, 64, 0);
        assert!(validate(&[valid], &parity, 2, 16, 192, 8).is_err());
    }

    #[derive(Debug, Clone)]
    struct Fake { pages: Vec<u32>, slot: i32, committed: usize }
    impl Placement for Fake {
        fn len(&self) -> usize { self.committed }
        fn set_len(&mut self, len: usize) { self.committed = len; }
        fn units(&self) -> &[u32] { &self.pages }
        fn slot(&self) -> i32 { self.slot }
        fn spans(&self) -> &[cuteafd_core::MediaSpan] { &[] }
    }

    #[test]
    fn publication_needs_the_validated_slots_pages_and_frontier() {
        let pages = [4u32, 5];
        let wave = [SequenceRows { sequence: 1, start: 0, rows: 3, base: 65, pages: &pages }];
        let check = |slot: i32, pages: &[u32], committed: usize, accepted: usize| {
            let mut fake = Fake { pages: pages.to_vec(), slot, committed };
            check_publication(wave.iter().copied(), &[(&mut fake, accepted)]).is_ok()
        };
        // Untouched, or advanced exactly by the accepted rows.
        assert!(check(1, &pages, 65, 2) && check(1, &pages, 67, 2) && check(1, &pages, 68, 3));
        // Nothing accepted: wherever the emission left the frontier in the wave.
        assert!(check(1, &pages, 66, 0) && check(1, &pages, 68, 0));
        assert!(!check(1, &pages, 66, 2), "advanced by other than the accepted rows");
        assert!(!check(1, &pages, 69, 0) && !check(1, &pages, 64, 0), "outside the wave");
        assert!(!check(1, &pages, 65, 4), "more accepted than the wave's rows");
        assert!(!check(0, &pages, 65, 1) && !check(-1, &pages, 65, 1), "another slot");
        assert!(!check(1, &[4, 6], 65, 1) && !check(1, &[4], 65, 1), "other pages");
        let mut fake = Fake { pages: pages.to_vec(), slot: 1, committed: 65 };
        assert!(check_publication(std::iter::empty(), &[(&mut fake, 0)]).is_err());
        assert!(check_publication(wave.iter().copied(), &[] as &[(&mut Fake, usize)]).is_err());
    }
}
