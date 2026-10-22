//! One admitted rank allocation. Only completed, initialized weights expose
//! execution views; a loading arena exposes destinations solely to the copier.
use super::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::{collections::BTreeMap, ops::Range};

#[derive(Clone, Debug)]
pub(crate) struct ArenaLayout {
    regions: Vec<Range<usize>>,
    bytes: usize,
    validation_bytes: usize,
}

impl ArenaLayout {
    /// Keep each existing layer layout intact, including its own padding.
    pub fn new(sizes: impl IntoIterator<Item = usize>, admitted: usize) -> Result<Self> {
        let mut regions = Vec::new();
        let mut bytes = 0usize;
        for size in sizes {
            ensure!(size > 0, "empty rank weight region");
            bytes = bytes
                .checked_add(255)
                .context("rank arena alignment overflow")?
                & !255;
            let end = bytes
                .checked_add(size)
                .context("rank arena extent overflow")?;
            regions.push(bytes..end);
            bytes = end;
        }
        ensure!(
            !regions.is_empty() && bytes <= admitted,
            "rank weight arena needs {bytes} bytes, admitted {admitted}"
        );
        Ok(Self {
            regions,
            bytes,
            validation_bytes: 0,
        })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn with_validation(mut self, bytes: usize, admitted_host: usize) -> Result<Self> {
        ensure!(
            bytes <= admitted_host && self.bytes.checked_add(bytes).is_some(),
            "host validation tail exceeds admission"
        );
        self.validation_bytes = bytes;
        Ok(self)
    }
    pub fn logical_bytes(&self) -> usize {
        self.bytes + self.validation_bytes
    }
    pub fn region(&self, index: usize) -> Result<Range<usize>> {
        self.regions
            .get(index)
            .cloned()
            .context("rank arena region absent")
    }
}

#[cfg(test)]
#[path = "weight_arena_tests.rs"]
pub(crate) mod real_tests;

const MAX_COVERAGE_INTERVALS: usize = 1_000_000;

#[derive(Default)]
pub(crate) struct Coverage {
    intervals: BTreeMap<u64, u64>,
    bytes: u64,
    peak: usize,
}
impl Coverage {
    pub(crate) fn insert(&mut self, start: u64, bytes: u64) -> Result<()> {
        self.insert_bounded(start, bytes, MAX_COVERAGE_INTERVALS)
    }
    fn insert_bounded(&mut self, start: u64, bytes: u64, cap: usize) -> Result<()> {
        let mut end = start.checked_add(bytes).context("coverage end overflow")?;
        ensure!(bytes > 0, "empty delivery fragment");
        let mut start = start;
        if let Some((&left, &right)) = self.intervals.range(..=start).next_back() {
            if right >= start {
                start = left;
                end = end.max(right);
            }
        }
        while let Some((&left, &right)) = self.intervals.range(start..).next() {
            if left > end {
                break;
            }
            self.intervals.remove(&left);
            self.bytes -= right - left;
            end = end.max(right);
        }
        ensure!(
            self.intervals.len() < cap,
            "push coverage interval cap exceeded"
        );
        self.bytes = self
            .bytes
            .checked_add(end - start)
            .context("coverage total overflow")?;
        self.intervals.insert(start, end);
        self.peak = self.peak.max(self.intervals.len());
        Ok(())
    }
}

/// One plan entry's destination shape: `run` bytes at each point of
/// `dims` (outermost first, `(count, dst_stride)`, unit and dense axes folded).
struct Shape {
    dst_base: u64,
    dst_len: u64,
    packed_base: u64,
    run: u64,
    dims: Vec<(u64, u64)>,
}

/// Exact destination footprint of a checked plan, excluding generated padding.
///
/// Delivery is tracked in the rank's packed space: each entry's runs in order,
/// entries in destination order (`plan.check` keeps a rank's entry extents
/// disjoint). Padding between runs never maps, so a fragment touching it is
/// refused and a missing run stays a hole, while a complete delivery is one
/// interval whatever the padding. The footprint costs O(entries), not O(runs).
#[derive(Default)]
pub(crate) struct ExpectedCoverage {
    shapes: Vec<Shape>,
    bytes: u64,
}
impl ExpectedCoverage {
    pub fn from_plan(
        plan: &cuteafd_loader::push_load::builder::CheckedPlan,
        rank: u32,
    ) -> Result<Self> {
        use cuteafd_loader::push_load::plan::Steps;
        let mut shapes = Vec::new();
        for entry in plan.plan.entries.iter().filter(|e| e.rank == rank) {
            let pattern = &plan.plan.patterns[entry.pattern as usize];
            let mut run = u64::from(pattern.run);
            let mut dims = Vec::new();
            let mut extent = run;
            let mut empty = false;
            for dim in pattern.dims.iter().rev() {
                let count = match &dim.steps {
                    Steps::Count(n) => u64::from(*n),
                    Steps::Indices(v) => v.len() as u64,
                };
                if count == 0 {
                    empty = true;
                    break;
                }
                if count == 1 {
                    continue;
                }
                ensure!(dim.dst_stride >= extent, "plan pattern destination strides overlap");
                if dims.is_empty() && dim.dst_stride == run {
                    run = run.checked_mul(count).context("expected footprint overflow")?;
                    extent = run;
                } else {
                    dims.push((count, dim.dst_stride));
                    extent = (count - 1)
                        .checked_mul(dim.dst_stride)
                        .and_then(|v| v.checked_add(extent))
                        .context("expected extent overflow")?;
                }
            }
            if empty {
                continue;
            }
            dims.reverse();
            shapes.push(Shape { dst_base: entry.dst_base, dst_len: extent, packed_base: 0, run, dims });
        }
        shapes.sort_unstable_by_key(|shape| shape.dst_base);
        let mut bytes = 0u64;
        let mut end = 0u64;
        for shape in &mut shapes {
            ensure!(shape.dst_base >= end, "plan entries overlap in the rank arena");
            end = shape.dst_base.checked_add(shape.dst_len).context("expected end overflow")?;
            shape.packed_base = bytes;
            let runs = shape
                .dims
                .iter()
                .try_fold(1u64, |n, &(count, _)| n.checked_mul(count))
                .context("expected run count overflow")?;
            bytes = runs
                .checked_mul(shape.run)
                .and_then(|v| v.checked_add(bytes))
                .context("expected footprint overflow")?;
        }
        ensure!(
            Some(&bytes) == plan.totals.rank_bytes.get(rank as usize),
            "checked plan destination total differs from footprint"
        );
        Ok(Self { shapes, bytes })
    }

    /// The packed position of destination byte `offset` and the bytes left in
    /// its run, or None when `offset` is padding or outside every entry.
    fn locate(&self, offset: u64) -> Option<(u64, u64)> {
        let index = self.shapes.partition_point(|shape| shape.dst_base <= offset).checked_sub(1)?;
        let shape = &self.shapes[index];
        let mut rel = offset - shape.dst_base;
        if rel >= shape.dst_len {
            return None;
        }
        let mut linear = 0u64;
        for &(count, stride) in &shape.dims {
            let step = rel / stride;
            if step >= count {
                return None;
            }
            rel -= step * stride;
            linear = linear * count + step;
        }
        (rel < shape.run).then(|| (shape.packed_base + linear * shape.run + rel, shape.run - rel))
    }

    /// Packed intervals of a destination fragment, coalesced; None if any
    /// byte is padding or outside the footprint.
    fn packed(&self, start: u64, bytes: u64) -> Option<Vec<(u64, u64)>> {
        let end = start.checked_add(bytes)?;
        let mut out: Vec<(u64, u64)> = Vec::new();
        let mut at = start;
        while at < end {
            let (packed, left) = self.locate(at)?;
            let take = left.min(end - at);
            match out.last_mut() {
                Some((s, n)) if *s + *n == packed => *n += take,
                _ => out.push((packed, take)),
            }
            at += take;
        }
        Some(out)
    }

    /// Record a delivered destination fragment; it must lie in the footprint.
    pub(crate) fn record(&self, coverage: &mut Coverage, start: u64, bytes: u64) -> Result<()> {
        ensure!(bytes > 0, "empty delivery fragment");
        let pieces = self.packed(start, bytes).context("delivered fragment outside checked footprint")?;
        for (packed, length) in pieces {
            coverage.insert(packed, length)?;
        }
        Ok(())
    }
    /// Delivered coverage equals this footprint exactly.
    pub(crate) fn complete(&self, delivered: &Coverage) -> bool {
        delivered.bytes == self.bytes
            && (self.bytes == 0 || delivered.intervals.iter().eq([(&0, &self.bytes)]))
    }
    pub(crate) fn bytes(&self) -> u64 { self.bytes }
    pub(crate) fn contains(&self, start: u64, bytes: u64) -> bool {
        bytes > 0 && self.packed(start, bytes).is_some()
    }
    /// Contiguous ascending destination spans, one entry each (fixtures).
    #[cfg(test)]
    pub(crate) fn spans(spans: &[(u64, u64)]) -> Self {
        let mut bytes = 0;
        let shapes = spans.iter().map(|&(dst_base, len)| {
            let shape = Shape { dst_base, dst_len: len, packed_base: bytes, run: len, dims: Vec::new() };
            bytes += len;
            shape
        }).collect();
        Self { shapes, bytes }
    }
    /// End of the last planned byte (the footprint's destination extent).
    fn end(&self) -> u64 {
        self.shapes.last().map_or(0, |shape| shape.dst_base + shape.dst_len)
    }
    /// Every destination gap below `limit` in ascending order: between
    /// entries, between runs inside each entry, and after the last entry.
    fn gaps(&self, limit: u64, gap: &mut dyn FnMut(u64, u64) -> Result<()>) -> Result<()> {
        fn inner(dims: &[(u64, u64)], run: u64, base: u64,
            gap: &mut dyn FnMut(u64, u64) -> Result<()>) -> Result<()> {
            let Some((&(count, stride), rest)) = dims.split_first() else { return Ok(()) };
            let extent = rest.iter().fold(run, |e, &(n, s)| (n - 1) * s + e);
            for step in 0..count {
                let at = base + step * stride;
                inner(rest, run, at, gap)?;
                if step + 1 < count && stride > extent {
                    gap(at + extent, at + stride)?;
                }
            }
            Ok(())
        }
        let mut cursor = 0u64;
        for shape in &self.shapes {
            if shape.dst_base >= limit { break; }
            gap(cursor, shape.dst_base)?;
            inner(&shape.dims, shape.run, shape.dst_base, &mut |a, b| gap(a.min(limit), b.min(limit)))?;
            cursor = (shape.dst_base + shape.dst_len).min(limit);
        }
        gap(cursor.min(limit), limit)
    }
}

pub(crate) struct LoadingArena<'a> {
    allocation: Allocation<'a>,
    layout: ArenaLayout,
    expected: ExpectedCoverage,
    coverage: Coverage,
    validation: Vec<u8>,
}

/// Post-Done owner: semantic checks and descriptor uploads may now run, but
/// execution still cannot borrow a published view until initialization succeeds.
pub(crate) struct DeliveredArena<'a> {
    allocation: Allocation<'a>,
    layout: ArenaLayout,
    validation: Vec<u8>,
    expected: ExpectedCoverage,
}

pub(crate) struct ReadyArena<'a> {
    allocation: Allocation<'a>,
    layout: ArenaLayout,
}

/// Non-owning, lifetime-bound layer region; it cannot outlive the rank owner.
pub(crate) struct WeightView<'s> {
    buffer: CuteafdDeviceBuffer,
    _owner: std::marker::PhantomData<&'s ReadyArena<'s>>,
}

fn slice(buffer: CuteafdDeviceBuffer, range: Range<usize>) -> Result<CuteafdDeviceBuffer> {
    ensure!(
        range.start <= range.end && range.end <= buffer.bytes,
        "weight view exceeds rank arena"
    );
    // SAFETY: the byte range was checked inside the owner's live allocation.
    let ptr = unsafe { buffer.ptr.cast::<u8>().add(range.start).cast() };
    Ok(CuteafdDeviceBuffer {
        ptr,
        bytes: range.len(),
        ..buffer
    })
}

impl<'a> LoadingArena<'a> {
    pub fn new(
        device: Device<'a>,
        layout: ArenaLayout,
        expected: ExpectedCoverage,
    ) -> Result<Self> {
        ensure!(
            expected.end() <= layout.logical_bytes() as u64,
            "expected footprint exceeds admitted arena"
        );
        let validation = vec![0; layout.validation_bytes];
        let allocation = Allocation::new(device, layout.bytes())?;
        Ok(Self {
            allocation,
            layout,
            expected,
            coverage: Coverage::default(),
            validation,
        })
    }

    pub fn belongs_to(&self, device: Device<'_>) -> bool {
        self.allocation.device.id == device.id
            && std::ptr::eq(self.allocation.device.library, device.library)
    }
    pub fn device_bytes(&self) -> usize { self.layout.bytes() }
    pub fn logical_bytes(&self) -> usize { self.layout.logical_bytes() }
    pub fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> {
        ensure!(bytes > 0 && self.expected.contains(offset, bytes), "fragment outside checked footprint");
        Ok(())
    }

    /// Copy-only destination. The ring consumer must retain this owner and
    /// drain its streams before dropping it on every failure path.
    pub fn destination(&self, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
        slice(
            self.allocation.buffer,
            offset..offset.checked_add(bytes).context("fragment end overflow")?,
        )
    }

    /// Host-tail fragments route directly from the registered ring. Device and
    /// host spans must be split at the logical prefix boundary by the consumer.
    pub fn copy_validation(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        ensure!(
            offset >= self.layout.bytes as u64
                && self.expected.contains(offset, bytes.len() as u64),
            "validation fragment outside checked host tail"
        );
        let start = usize::try_from(offset - self.layout.bytes as u64)?;
        let end = start
            .checked_add(bytes.len())
            .context("validation fragment overflow")?;
        self.validation
            .get_mut(start..end)
            .context("validation fragment exceeds host tail")?
            .copy_from_slice(bytes);
        self.expected.record(&mut self.coverage, offset, bytes.len() as u64)
    }

    /// Called only after a slot's CUDA copy event completes, before Release.
    pub fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> {
        ensure!(
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= self.layout.bytes as u64),
            "completed fragment exceeds rank arena"
        );
        ensure!(
            self.expected.contains(offset, bytes),
            "completed fragment is outside checked plan footprint"
        );
        self.expected.record(&mut self.coverage, offset, bytes)?;
        ensure!(
            self.coverage.bytes <= self.expected.bytes(),
            "rank unique delivery exceeds checked plan"
        );
        Ok(())
    }

    /// Caller observed rank Done and drained every copy. Missing bytes cannot
    /// become execution-visible even when the control socket says Done.
    pub fn done(self) -> Result<DeliveredArena<'a>> {
        ensure!(
            self.expected.complete(&self.coverage),
            "rank Done before all planned bytes completed"
        );
        Ok(DeliveredArena {
            allocation: self.allocation,
            layout: self.layout,
            validation: self.validation,
            expected: self.expected,
        })
    }
}

impl<'a> DeliveredArena<'a> {
    pub fn device(&self) -> Device<'a> { self.allocation.device }
    pub fn buffer(&self, range: Range<usize>) -> Result<CuteafdDeviceBuffer> {
        slice(self.allocation.buffer, range)
    }
    /// Only omitted bytes are generated. Delivered source bytes are never
    /// replaced by zero padding, including replayed or sparse tensor images.
    pub fn zero_padding(&self) -> Result<()> {
        let result = self.device().run(|| {
            let queued = (|| {
                // Fold adjacent equal-width/equal-pitch gaps in constant host
                // memory. A padded plane then needs one enqueue, not one per row.
                let mut pending: Option<(usize, usize, usize, usize)> = None;
                let flush = |(start, width, pitch, rows): (usize, usize, usize, usize)| -> Result<()> {
                    let end = start + (rows - 1) * pitch + width;
                    // SAFETY: only checked complement gaps are selected; this
                    // owner is retained until the unconditional drain below.
                    unsafe { self.device().library.zero_pitched_async(
                        self.buffer(start..end)?, pitch, width, rows, std::ptr::null_mut()) }
                };
                let mut gap = |start: usize, end: usize| -> Result<()> {
                    if start == end { return Ok(()); }
                    let width = end - start;
                    if let Some((base, previous_width, pitch, rows)) = pending {
                        if previous_width == width && (rows == 1 || Some(start - base) == rows.checked_mul(pitch)) {
                            pending = Some((base, width, if rows == 1 { start - base } else { pitch }, rows + 1));
                            return Ok(());
                        }
                        flush((base, previous_width, pitch, rows))?;
                    }
                    pending = Some((start, width, width, 1));
                    Ok(())
                };
                self.expected.gaps(self.layout.bytes as u64, &mut |start, end| {
                    gap(usize::try_from(start)?, usize::try_from(end)?)
                })?;
                if let Some(group) = pending { flush(group)?; }
                Ok(())
            })();
            // A failed enqueue can still leave earlier zero fills in flight.
            // SAFETY: the current device owns the default stream and arena.
            let drained = unsafe { self.device().library.cuda_stream_synchronize(std::ptr::null_mut()) };
            drained?;
            Ok(queued)
        });
        // Device binding/restoration can fail outside the drain closure too.
        if result.is_err() { self.device().library.quarantine_module_after_failed_drain(); }
        result?
    }
    pub fn validation(&self) -> &[u8] {
        &self.validation
    }
    pub fn region(&self, index: usize) -> Result<CuteafdDeviceBuffer> {
        slice(self.allocation.buffer, self.layout.region(index)?)
    }

    /// Preserve semantic validation and generated descriptors, after Done and
    /// before execution. An error consumes and discards the unpublished arena.
    pub fn initialize(
        self,
        initialize: impl FnOnce(&Self) -> Result<()>,
    ) -> Result<ReadyArena<'a>> {
        initialize(&self)?;
        Ok(ReadyArena {
            allocation: self.allocation,
            layout: self.layout,
        })
    }
}

impl ReadyArena<'_> {
    pub fn region(&self, index: usize) -> Result<WeightView<'_>> {
        Ok(WeightView {
            buffer: slice(self.allocation.buffer, self.layout.region(index)?)?,
            _owner: std::marker::PhantomData,
        })
    }
}

impl WeightView<'_> {
    pub fn buffer(&self) -> CuteafdDeviceBuffer {
        self.buffer
    }
    pub fn subregion(&self, offset: usize, bytes: usize) -> Result<WeightView<'_>> {
        Ok(WeightView {
            buffer: slice(
                self.buffer,
                offset
                    ..offset
                        .checked_add(bytes)
                        .context("weight subregion overflow")?,
            )?,
            _owner: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;

    fn expected(bytes: u64) -> Result<ExpectedCoverage> {
        Ok(if bytes == 0 { ExpectedCoverage::default() } else { ExpectedCoverage::spans(&[(0, bytes)]) })
    }

    #[test]
    fn layout_admits_before_allocation_and_keeps_layer_offsets() -> Result<()> {
        let layout = ArenaLayout::new([257, 32], 544)?;
        assert_eq!(layout.region(0)?, 0..257);
        assert_eq!(layout.region(1)?, 512..544);
        assert!(ArenaLayout::new([257, 32], 543).is_err());
        assert!(ArenaLayout::new([usize::MAX, 1], usize::MAX).is_err());
        assert!(ArenaLayout::new([], 0).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn only_done_initialized_arena_publishes_stable_nonowning_views() -> Result<()> {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device {
            library: &library,
            id: 1,
        };
        fixture.configure_memory(&library, 1, 4096)?;
        let mut loading =
            LoadingArena::new(device, ArenaLayout::new([256, 256], 512)?, expected(16)?)?;
        let base = loading.destination(0, 16)?.ptr;
        assert!(loading.destination(510, 4).is_err());
        loading.copied(0, 8)?;
        loading.copied(8, 8)?;
        loading.copied(0, 16)?; // Replayed source file is idempotent.
        assert!(loading.copied(512, 1).is_err());
        let delivered = loading.done()?;
        let ready = delivered.initialize(|arena| {
            assert_eq!(arena.region(0)?.ptr, base);
            Ok(())
        })?;
        let view = ready.region(0)?;
        assert_eq!(view.buffer().ptr, base);
        assert_eq!(view.subregion(8, 8)?.buffer().bytes, 8);
        assert!(view.subregion(250, 8).is_err());
        assert_eq!(fixture.events()?.matches('D').count(), 1);
        assert_eq!(fixture.events()?.matches('d').count(), 0);
        drop(ready);
        assert_eq!(fixture.events()?.matches('d').count(), 1);
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn host_validation_tail_is_admitted_routed_and_checked_only_after_done() -> Result<()> {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device {
            library: &library,
            id: 0,
        };
        let footprint = ExpectedCoverage::spans(&[(0, 4), (256, 4)]);
        let layout = ArenaLayout::new([256], 256)?.with_validation(4, 4)?;
        assert!(ArenaLayout::new([256], 256)?.with_validation(4, 3).is_err());
        let mut loading = LoadingArena::new(device, layout, footprint)?;
        assert!(loading.copied(16, 4).is_err()); // Equal-size padding is not planned payload.
        assert!(loading.copied(256, 4).is_err()); // No device copy to host tail.
        loading.copied(0, 4)?;
        loading.copy_validation(256, &0xcbac1fedu32.to_le_bytes())?;
        let delivered = loading.done()?;
        assert_eq!(delivered.validation(), &0xcbac1fedu32.to_le_bytes());
        let _ready = delivered.initialize(|value| {
            ensure!(
                u32::from_le_bytes(value.validation().try_into()?) == 0xcbac1fed,
                "bad multiplier"
            );
            Ok(())
        })?;
        assert_eq!(fixture.events()?.matches('D').count(), 1);
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn incomplete_delivery_or_failed_initialization_never_publishes() -> Result<()> {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device {
            library: &library,
            id: 0,
        };
        let loading = LoadingArena::new(device, ArenaLayout::new([256], 256)?, expected(4)?)?;
        assert!(loading.done().is_err());
        let delivered =
            LoadingArena::new(device, ArenaLayout::new([256], 256)?, expected(0)?)?.done()?;
        assert!(delivered
            .initialize(|_| anyhow::bail!("semantic check failed"))
            .is_err());
        assert_eq!(fixture.events()?.matches('D').count(), 2);
        assert_eq!(fixture.events()?.matches('d').count(), 2);
        Ok(())
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::*;
    use cuteafd_loader::push_load::{
        builder::CheckedPlan,
        plan::{Dim, Entry, Pattern, Plan, Steps},
    };

    fn checked(patterns: Vec<Pattern>, entries: Vec<Entry>, file: u64, arena: u64) -> Result<CheckedPlan> {
        let plan = Plan { files: vec!["weights".into()], patterns, entries };
        let totals = plan.check(&[file], &[arena])?;
        Ok(CheckedPlan { plan, file_sizes: vec![file], arenas: vec![arena], totals })
    }
    fn entry(pattern: u32, src_base: u64, dst_base: u64) -> Entry {
        Entry { file: 0, pattern, rank: 0, src_base, dst_base }
    }
    /// Every destination run of the rank's entries (oracle).
    fn runs(plan: &CheckedPlan) -> Vec<(u64, u64)> {
        let mut out = plan.plan.entries.iter().flat_map(|e| {
            let p = &plan.plan.patterns[e.pattern as usize];
            p.runs().into_iter().map(move |(_, d)| (e.dst_base + d, u64::from(p.run)))
        }).collect::<Vec<_>>();
        out.sort_unstable();
        out
    }
    /// Gaps must be exactly the complement of the runs below `limit`.
    fn assert_gaps(expected: &ExpectedCoverage, plan: &CheckedPlan, limit: u64) -> Result<()> {
        let mut want = Vec::new();
        let mut cursor = 0;
        for (start, len) in runs(plan) {
            if start.min(limit) > cursor { want.push((cursor, start.min(limit))); }
            cursor = (start + len).min(limit);
        }
        if limit > cursor { want.push((cursor, limit)); }
        let mut got = Vec::new();
        expected.gaps(limit, &mut |a, b| { if a < b { got.push((a, b)); } Ok(()) })?;
        // Adjacent gaps may be reported separately; merge before comparing.
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (a, b) in got {
            match merged.last_mut() { Some((_, e)) if *e == a => *e = b, _ => merged.push((a, b)) }
        }
        assert_eq!(merged, want);
        Ok(())
    }

    #[test]
    fn expected_footprint_excludes_padding_and_merges_dense_axes() -> Result<()> {
        let plan = checked(
            vec![Pattern {
                run: 4,
                dims: vec![
                    Dim { steps: Steps::Count(2), src_stride: 64, dst_stride: 16 },
                    Dim { steps: Steps::Indices(vec![0, 3]), src_stride: 4, dst_stride: 4 },
                ],
            }],
            vec![entry(0, 0, 0)],
            128,
            32,
        )?;
        let expected = ExpectedCoverage::from_plan(&plan, 0)?;
        assert_eq!(expected.bytes(), 16);
        assert!(expected.contains(4, 4));
        assert!(!expected.contains(4, 8));
        assert!(!expected.contains(8, 4));
        assert!(ExpectedCoverage::from_plan(&plan, 2).is_err());
        assert_gaps(&expected, &plan, 32)?;
        Ok(())
    }

    #[test]
    fn padding_never_hides_a_missing_run() -> Result<()> {
        // Four 6-byte rows at an 8-byte pitch: 2 bytes of padding per row.
        let pattern = Pattern { run: 6, dims: vec![Dim { steps: Steps::Count(4), src_stride: 6, dst_stride: 8 }] };
        let plan = checked(vec![pattern], vec![entry(0, 0, 0)], 24, 40)?;
        let expected = ExpectedCoverage::from_plan(&plan, 0)?;
        let mut coverage = Coverage::default();
        for row in [0u64, 1, 3] {
            expected.record(&mut coverage, row * 8, 6)?;
        }
        assert!(!expected.complete(&coverage), "a missing row must stay a hole");
        // Padding is not deliverable, alone or straddling a run.
        assert!(expected.record(&mut coverage, 6, 2).is_err());
        assert!(expected.record(&mut coverage, 20, 6).is_err());
        assert!(expected.record(&mut coverage, 30, 2).is_err());
        expected.record(&mut coverage, 16, 6)?;
        assert!(expected.complete(&coverage));
        assert_eq!(coverage.intervals.len(), 1);
        // Replays merge without changing the total.
        expected.record(&mut coverage, 8, 6)?;
        assert_eq!(coverage.bytes, 24);
        assert_gaps(&expected, &plan, 40)?;
        Ok(())
    }

    /// MiMo V2.6 Pro TP6 Down on a 256-wide rank stored at the padded 384
    /// width: one 256 B run per row at a 384 B pitch, every expert of a layer.
    #[test]
    fn padded_mimo_down_layer_stays_far_under_the_cap() -> Result<()> {
        let (rows, experts, width, pitch, cols) = (6144u64, 384u64, 256u64, 384u64, 2048u64);
        let pattern = Pattern {
            run: width as u32,
            dims: vec![Dim { steps: Steps::Count(rows as u32), src_stride: cols, dst_stride: pitch }],
        };
        let image = rows * pitch;
        let entries = (0..experts).map(|e| entry(0, e * rows * cols + 768, e * image)).collect();
        let plan = checked(vec![pattern], entries, experts * rows * cols, experts * image)?;
        let expected = ExpectedCoverage::from_plan(&plan, 0)?;
        assert_eq!(expected.bytes(), rows * experts * width);
        let mut coverage = Coverage::default();
        // Deliver in file order, one row run at a time, as the source does.
        for e in 0..experts {
            for row in 0..rows {
                expected.record(&mut coverage, e * image + row * pitch, width)?;
            }
            assert!(coverage.intervals.len() <= 1);
        }
        assert!(expected.complete(&coverage));
        assert_eq!(coverage.peak, 1, "2.4M row runs fold into one packed interval");
        Ok(())
    }

    #[test]
    fn gaps_follow_multi_dimensional_and_inter_entry_padding() -> Result<()> {
        let patterns = vec![
            Pattern { run: 3, dims: vec![
                Dim { steps: Steps::Count(2), src_stride: 100, dst_stride: 20 },
                Dim { steps: Steps::Indices(vec![1, 4, 5]), src_stride: 3, dst_stride: 5 },
            ] },
            Pattern { run: 8, dims: vec![] },
        ];
        let plan = checked(patterns, vec![entry(1, 300, 2), entry(0, 0, 12), entry(1, 400, 60)], 512, 80)?;
        let expected = ExpectedCoverage::from_plan(&plan, 0)?;
        assert_gaps(&expected, &plan, 80)?;
        assert_gaps(&expected, &plan, 40)?;
        let mut coverage = Coverage::default();
        for (start, len) in runs(&plan).into_iter().rev() {
            assert!(!expected.complete(&coverage));
            expected.record(&mut coverage, start, len)?;
        }
        assert!(expected.complete(&coverage));
        Ok(())
    }

    #[test]
    fn replay_merges_partial_overlaps_and_cap_fails_closed() -> Result<()> {
        let mut coverage = Coverage::default();
        coverage.insert_bounded(4, 4, 2)?;
        coverage.insert_bounded(16, 4, 2)?;
        assert!(coverage.insert_bounded(32, 4, 2).is_err());
        coverage.insert_bounded(0, 18, 2)?;
        coverage.insert_bounded(0, 20, 2)?;
        assert_eq!(coverage.intervals, [(0, 20)].into());
        assert_eq!(coverage.bytes, 20);
        assert_eq!(coverage.peak, 2);
        Ok(())
    }
}
