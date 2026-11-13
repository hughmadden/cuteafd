//! Host metadata of one Muse step (a prefill unit or a decode/verify batch):
//! the rows' positions, the full layers' page tables and the sliding layers'
//! ring addressing, exactly as the attention programs read them
//! (`_mimo_kernels.MimoGqaAttention._resolve`, `MimoRingCommit`):
//!
//! - a sliding row at position `p` of a sequence whose rings start at ring row
//!   `base` (`ring index * ring_rows`) has `ring_slots[row] = base + p % ring_rows`
//!   and `seq_first[row]` = the step row of its sequence's first row; keys at or
//!   past that first row's position come from this step's records, older keys
//!   from ring slot `base + k % ring_rows` (the [`MuseStateLayout`] slots, so
//!   prefix marks and kernels agree on where position `k` lives);
//! - a full row reads key `k` at page `page_table[row * stride + k / 64]`.
//!
//! Rows of one sequence are consecutive with consecutive positions.
//!
//! [`MuseStateLayout`]: cuteafd_loader::families::muse_spark::cache::MuseStateLayout
use anyhow::{ensure, Context, Result};

pub(crate) const PAGE_ROWS: usize = 64;

/// One sequence's rows in a step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StepSequence<'p> {
    /// Its sequence slot (ring index).
    pub slot: usize,
    /// Position of its first row in this step (its committed length).
    pub start: usize,
    pub rows: usize,
    /// Its pages in row order, covering at least `start + rows` positions.
    pub pages: &'p [u32],
}

/// Device-ready metadata of a step, one row per entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StepMetadata {
    pub positions: Vec<i64>,
    pub ring_slots: Vec<i64>,
    pub seq_first: Vec<i32>,
    /// Full layers' K|V record index per row in the paged cache
    /// (`page * 64 + p % 64`), the `kv_slots` of `muse_qk_norm_nope_full`.
    pub page_slots: Vec<i64>,
    /// Page ids: one shared row for prefill, one per query for decode/verify.
    pub page_table: Vec<i32>,
    /// Query-row stride in page ids; zero makes every prefill query share row 0.
    pub table_stride: usize,
    /// Packed prefill members: local query range and its private shared table.
    pub segments: Vec<(std::ops::Range<usize>, Vec<i32>)>,
}

impl StepMetadata {
    /// SWA layers write each row's record into the step's own `kv_step` row.
    pub(crate) fn step_slots(&self) -> Vec<i64> { (0..self.positions.len() as i64).collect() }

    /// Give decode tables a stable admitted row pitch. Positions still bound
    /// attention on-device; sentinel padding is never a visible key.
    pub(crate) fn with_stride(mut self, stride: usize) -> Result<Self> {
        ensure!(self.table_stride > 0 && stride >= self.table_stride,
            "Muse decode table stride {} exceeds admitted {stride}", self.table_stride);
        if stride != self.table_stride {
            let mut table = Vec::with_capacity(self.positions.len() * stride);
            for row in self.page_table.chunks_exact(self.table_stride) {
                table.extend_from_slice(row);
                table.resize(table.len() + stride - self.table_stride, -1);
            }
            self.page_table = table;
            self.table_stride = stride;
        }
        Ok(self)
    }

    /// `ring_rows`: rows of one sequence's ring; `slots`: rings admitted.
    /// `shared_table`: the prefill route reads one table row for its one sequence.
    pub(crate) fn build(sequences: &[StepSequence<'_>], ring_rows: usize, slots: usize, step_capacity: usize,
        shared_table: bool) -> Result<Self> {
        ensure!(!sequences.is_empty() && ring_rows > 0, "Muse step without sequences or rings");
        let rows = sequences.iter().try_fold(0usize, |n, s| n.checked_add(s.rows)).context("Muse step rows")?;
        ensure!((1..=step_capacity).contains(&rows), "Muse step of {rows} rows outside 1..={step_capacity}");
        let stride = sequences.iter().map(|s| s.pages.len()).max().unwrap_or(0).max(1);
        let mut seen = std::collections::HashSet::new();
        let mut out = Self { positions: Vec::with_capacity(rows), ring_slots: Vec::with_capacity(rows),
            seq_first: Vec::with_capacity(rows), page_slots: Vec::with_capacity(rows), page_table: Vec::new(),
            table_stride: if shared_table { 0 } else { stride }, segments: Vec::new() };
        for sequence in sequences {
            ensure!(sequence.rows > 0 && sequence.slot < slots && seen.insert(sequence.slot),
                "Muse step sequence slot {} invalid or repeated", sequence.slot);
            let end = sequence.start.checked_add(sequence.rows).context("Muse step positions")?;
            ensure!(end.div_ceil(PAGE_ROWS) <= sequence.pages.len(),
                "Muse sequence {} lacks pages for {end} positions", sequence.slot);
            // A step's own keys come from its records and the commit runs after attention,
            // so a prefill unit may be longer than the ring (`cache::SWA_RING_STEP`). A
            // rollback step (decode/verify, `!shared_table`) writes past the committed
            // frontier and must leave the visible window intact.
            ensure!(shared_table || sequence.rows as u64 <= cuteafd_loader::families::muse_spark::cache::SWA_RING_STEP,
                "Muse decode/verify step of {} rows exceeds the ring's rollback step", sequence.rows);
            let first = i32::try_from(out.positions.len())?;
            let base = (sequence.slot * ring_rows) as i64;
            let mut table = sequence.pages.iter().map(|&p| i32::try_from(p)).collect::<Result<Vec<_>, _>>()?;
            table.resize(stride, -1);
            for p in sequence.start..end {
                out.positions.push(i64::try_from(p)?);
                out.ring_slots.push(base + (p % ring_rows) as i64);
                out.seq_first.push(if shared_table && sequences.len() > 1 { 0 } else { first });
                out.page_slots.push(i64::from(sequence.pages[p / PAGE_ROWS]) * PAGE_ROWS as i64 + (p % PAGE_ROWS) as i64);
                if !shared_table { out.page_table.extend_from_slice(&table); }
            }
            if shared_table {
                if sequences.len() > 1 { out.segments.push((first as usize..out.positions.len(), table.clone())); }
                if out.page_table.is_empty() { out.page_table = table; }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::families::muse_spark::cache::{MuseCacheLayout, MuseStateLayout};

    /// The kernel's key resolution, on the host: where key `k` of `row` is read.
    fn key_source(meta: &StepMetadata, row: usize, k: i64, ring_rows: usize) -> (bool, i64) {
        let first = meta.seq_first[row] as usize;
        let seq_start = meta.positions[first];
        let p = meta.positions[row];
        let ring_base = meta.ring_slots[row] - p % ring_rows as i64;
        if k >= seq_start { (true, first as i64 + k - seq_start) } else { (false, ring_base + k % ring_rows as i64) }
    }

    #[test]
    fn ring_slots_agree_with_the_prefix_state_layout() -> Result<()> {
        let cfg = cuteafd_loader::families::muse_spark::MuseSparkConfig::from_hf(
            &cuteafd_loader::plan::testing::muse_spark_config())?;
        let cache = MuseCacheLayout::new(&cfg, 1)?;
        let state = MuseStateLayout::new(&cfg, &cache, 16)?;
        let ring = state.ring.rows as usize;
        let pages: Vec<u32> = (100..400).collect();
        for (start, rows) in [(0, 1), (0, 2752), (5000, 2752), (ring * 3 - 7, 64), (10_000, 1), (100, 4096)] {
            let meta = StepMetadata::build(&[StepSequence { slot: 5, start, rows, pages: &pages }], ring, 16, 4096, true)?;
            for (row, &p) in meta.positions.iter().enumerate() {
                // The commit writes p's record where the mark layout expects it.
                assert_eq!(meta.ring_slots[row], (5 * ring) as i64 + p % ring as i64);
                // Every key in the window that predates the step is read from that same ring slot.
                for k in (p - (cfg.window as i64 - 1)).max(0)..=p {
                    let (step, at) = key_source(&meta, row, k, ring);
                    if k < start as i64 {
                        assert!(!step && at == (5 * ring) as i64 + k % ring as i64);
                    } else {
                        assert!(step && meta.positions[at as usize] == k);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn full_attention_query_stride_matches_uploaded_shared_and_per_query_tables() -> Result<()> {
        let pages: Vec<u32> = (0..256).map(|page| 1000 + page * 7).collect();
        for (start, rows) in [(0, 1), (1, 511), (0, 513), (63, 64), (63, 65), (11840, 4), (11840, 512)] {
            for shared in [true, false] {
                if !shared && rows > 64 { continue; }
                let meta = StepMetadata::build(&[StepSequence { slot: 3, start, rows, pages: &pages }],
                    2112, 16, 4096, shared)?;
                assert_eq!(meta.table_stride, if shared { 0 } else { pages.len() });
                assert_eq!(meta.page_table.len(), if shared { pages.len() } else { rows * pages.len() });
                for row in 0..rows {
                    // Match the inherited attention kernel's row * stride + key / 64 addressing.
                    for key in 0..=start + row {
                        let index = row * meta.table_stride + key / PAGE_ROWS;
                        assert_eq!(meta.page_table.get(index).copied(), Some(pages[key / PAGE_ROWS] as i32),
                            "shared={shared}, start={start}, row={row}, key={key}");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn decode_batches_keep_each_sequence_on_its_own_ring_and_pages() -> Result<()> {
        let (a, b): (Vec<u32>, Vec<u32>) = ((0..3).collect(), (10..12).collect());
        let meta = StepMetadata::build(&[StepSequence { slot: 0, start: 130, rows: 4, pages: &a },
            StepSequence { slot: 3, start: 64, rows: 1, pages: &b }], 2112, 4, 64, false)?;
        assert_eq!(meta.positions, [130, 131, 132, 133, 64]);
        assert_eq!(meta.seq_first, [0, 0, 0, 0, 4]);
        assert_eq!(meta.ring_slots[4], 3 * 2112 + 64);
        assert_eq!(meta.table_stride, 3);
        assert_eq!(&meta.page_table[..3], [0, 1, 2]);
        assert_eq!(&meta.page_table[12..], [10, 11, -1]);
        Ok(())
    }

    #[test]
    fn packed_members_rebase_swa_records_but_keep_private_tables_rings_and_marks() -> Result<()> {
        use crate::shared::prefill_packing::{plan, Limits};
        use cuteafd_loader::families::muse_spark::cache::StateAt;
        let cfg = cuteafd_loader::families::muse_spark::MuseSparkConfig::from_hf(
            &cuteafd_loader::plan::testing::muse_spark_config())?;
        for ranks in [1, 2] {
            let cache = MuseCacheLayout::new(&cfg, ranks)?;
            let state = MuseStateLayout::new(&cfg, &cache, 4)?;
            let ring = state.ring.rows as usize;
            let a = (10..60).collect::<Vec<_>>();
            let b = vec![91, 97];
            let sequences = [StepSequence { slot: 1, start: ring + 63, rows: 65, pages: &a },
                StepSequence { slot: 3, start: 7, rows: 29, pages: &b }];
            let segments = plan(&sequences.iter().map(|s| (s.start, s.rows)).collect::<Vec<_>>(),
                &Limits { rows: 128, chunk_rows: 128, max_context: 8192, taps: Some(128) })?;
            let packed = StepMetadata::build(&sequences, ring, 4, 128, true)?;
            assert_eq!(packed.segments.len(), 2);
            assert_ne!(packed.segments[0].1, packed.segments[1].1, "tables must not alias or be reused on host");
            assert_eq!(segments[1].tap_offset, segments[0].tap_rows);
            for (i, sequence) in sequences.iter().enumerate() {
                let (range, table) = &packed.segments[i];
                let alone = StepMetadata::build(&[sequence.clone()], ring, 4, 128, true)?;
                assert_eq!(&packed.positions[range.clone()], alone.positions);
                assert_eq!(&packed.ring_slots[range.clone()], alone.ring_slots);
                assert_eq!(&packed.page_slots[range.clone()], alone.page_slots);
                assert_eq!(&table[..sequence.pages.len()], alone.page_table);
                assert!(packed.seq_first[range.clone()].iter().all(|&first| first == 0));
                for local in 0..range.len() {
                    let position = packed.positions[range.start + local];
                    let step_key = position - sequence.start as i64;
                    assert_eq!(range.start + step_key as usize, range.start + local);
                }
                let mark_slot = (i + 5) as u64;
                for copy in state.capture(sequence.slot as u64, (sequence.start + sequence.rows) as u64, mark_slot)? {
                    let (StateAt::Ring(source), StateAt::Mark(destination)) = (copy.source, copy.destination)
                        else { panic!("capture direction") };
                    assert!(destination >= mark_slot * state.mark_bytes());
                    assert!(destination + copy.bytes <= (mark_slot + 1) * state.mark_bytes());
                    let layer = source / (4 * ring as u64 * state.row_bytes);
                    let base = state.ring_offset(layer as usize, sequence.slot as u64);
                    assert!(source >= base && source + copy.bytes <= base + ring as u64 * state.row_bytes);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn fixed_decode_pitch_preserves_every_visible_key_and_address() -> Result<()> {
        let pages = [7, 2, 9];
        let original = StepMetadata::build(&[StepSequence { slot: 1, start: 127, rows: 2, pages: &pages }],
            2112, 4, 64, false)?;
        let padded = original.clone().with_stride(8)?;
        assert_eq!(padded.positions, original.positions);
        assert_eq!(padded.page_slots, original.page_slots);
        assert_eq!(padded.ring_slots, original.ring_slots);
        assert_eq!(padded.seq_first, original.seq_first);
        for row in 0..2 {
            assert_eq!(&padded.page_table[row * 8..row * 8 + 3], &original.page_table[row * 3..row * 3 + 3]);
            assert!(padded.page_table[row * 8 + 3..(row + 1) * 8].iter().all(|&p| p == -1));
        }
        assert!(original.with_stride(2).is_err());
        Ok(())
    }

    #[test]
    fn malformed_steps_refuse_before_any_launch() {
        let pages: Vec<u32> = vec![0];
        let one = |slot, start, rows| StepSequence { slot, start, rows, pages: &pages };
        assert!(StepMetadata::build(&[], 2112, 4, 64, false).is_err());
        assert!(StepMetadata::build(&[one(4, 0, 1)], 2112, 4, 64, false).is_err(), "slot outside rings");
        assert!(StepMetadata::build(&[one(0, 0, 1), one(0, 1, 1)], 2112, 4, 64, false).is_err(), "repeated slot");
        assert!(StepMetadata::build(&[one(0, 60, 8)], 2112, 4, 64, false).is_err(), "lacks pages");
        assert!(StepMetadata::build(&[one(0, 0, 65)], 2112, 4, 64, false).is_err(), "over capacity");
        let many: Vec<u32> = (0..2).collect();
        assert!(StepMetadata::build(&[StepSequence { slot: 0, start: 0, rows: 65, pages: &many }], 2112, 4, 4096, false)
            .is_err(), "a verify step past the rollback step");
        assert!(StepMetadata::build(&[one(0, 0, 1), one(1, 0, 1)], 2112, 4, 64, true).is_ok());
    }

    /// Records written through `kv_slots` land where the attention reads them:
    /// a full layer's key `k` of a row is read at `page_table[k / 64] * 64 + k % 64`,
    /// which is the slot the row at position `k` wrote; a sliding row's own-step
    /// keys come from `kv_step[seq_first + k - start]`, the slot the SWA record write
    /// used. Two prefills of the same tokens on different pages write identical
    /// records at their own slots (restore byte-identity rests on this).
    #[test]
    fn records_land_where_attention_and_marks_read_them() -> Result<()> {
        let record = |p: i64| -> [u8; 4] { (p as u32).wrapping_mul(2654435761).to_le_bytes() };
        let pages_a: Vec<u32> = vec![7, 2, 9];
        let pages_b: Vec<u32> = vec![4, 11, 0];
        let mut cache = vec![[0u8; 4]; 12 * 64];
        fn write(cache: &mut [[u8; 4]], record: impl Fn(i64) -> [u8; 4], pages: &[u32], start: usize, rows: usize)
            -> Result<StepMetadata> {
            let meta = StepMetadata::build(&[StepSequence { slot: 0, start, rows, pages }], 2112, 1, 4096, true)?;
            for (row, &slot) in meta.page_slots.iter().enumerate() {
                cache[slot as usize] = record(meta.positions[row]);
            }
            Ok(meta)
        }
        let meta = write(&mut cache, record, &pages_a, 0, 130)?;
        // Every key the last row reads (all earlier positions) is its own position's record.
        for k in 0..130i64 {
            let page = meta.page_table[(k / 64) as usize] as usize;
            assert_eq!(cache[page * 64 + (k % 64) as usize], record(k));
        }
        // The same tokens prefilled onto other pages: identical records at their own slots.
        write(&mut cache, record, &pages_b, 0, 100)?;
        write(&mut cache, record, &pages_b, 100, 30)?;
        for k in 0..130usize {
            assert_eq!(cache[pages_a[k / 64] as usize * 64 + k % 64], cache[pages_b[k / 64] as usize * 64 + k % 64]);
        }
        // SWA: own-step keys at kv_step[seq_first + k - start] = the row's step slot.
        let swa = StepMetadata::build(&[StepSequence { slot: 0, start: 500, rows: 8, pages: &[0u32; 9] }], 2112, 1, 64, false)?;
        let slots = swa.step_slots();
        for (row, &p) in swa.positions.iter().enumerate() {
            let first = swa.seq_first[row] as i64;
            assert_eq!(slots[(first + p - swa.positions[first as usize]) as usize], row as i64);
        }
        Ok(())
    }
}
