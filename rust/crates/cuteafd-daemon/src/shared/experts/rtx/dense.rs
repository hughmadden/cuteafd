//! Coordinator dense operands as push-group ranks, for any family.
//!
//! A loader in push mode allocates every operand exactly as the mount does and
//! records how its bytes arrive instead of reading them:
//! - an [`Operand`] (class 2) is checkpoint rectangles landing unchanged in its
//!   allocation;
//! - a [`Step`] (class 3) is rectangles staged in pinned host memory, then GPU
//!   [`Job`]s (FP8 quantization, BF16 widening, MLA absorption, checked copies)
//!   that write their operands on the step's GPU. Steps hold at most
//!   [`STEP_BYTES`] (whole scale units), so staging stays bounded.
//!
//! The mount runs the same records: rectangles read into the operand, or into a
//! pinned buffer the same jobs read. A rank's logical space is its operands in
//! `[0, base)` (device) and its steps in `[base, base + host_bytes)`, delivered
//! through the consumer's host path; a step's jobs run once its bytes are
//! complete, and its staging returns to a small pool.
#[path = "dense/records.rs"]
mod records;
pub(crate) use records::*;
use crate::shared::memory::{device::Device, push_consumer::PushDestination,
    weight_arena::{Coverage, ExpectedCoverage}, HostAllocation};
use crate::shared::fp8_load_quant::LoadKernels;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::formats::native_dense::{element_bytes, RankWeightLayout, WeightOperand, WeightSlice};
use cuteafd_loader::push_load::builder::{Builder, CheckedPlan, Destination, Source};
use std::{ffi::c_void, path::Path};

/// `rect`'s rows as plan entries landing at `offset` of rank `rank`.
pub(crate) fn append(rect: &Rect, builder: &mut Builder, rank: u32, offset: u64) -> Result<()> {
    let source = Source { path: rect.path.clone(), offset: rect.base,
        bytes: (rect.rows.end * rect.row_bytes) as u64 };
    builder.rows(&source, (rect.rows.start * rect.row_bytes + rect.cols.start) as u64,
        u32::try_from(rect.rows.len())?, u32::try_from(rect.width())?, rect.row_bytes as u64, rect.width() as u64,
        Destination { rank, offset, bytes: rect.bytes() as u64 })?;
    Ok(())
}

/// Where a native dense operand's bytes come from, in allocation order: each
/// source's checkpoint rectangle (the whole tensor, the rank's rows or the
/// rank's column window, in source order) at its operand offset, flagged when
/// an F32 rectangle narrows to BF16 into half its bytes. Also returns the
/// packed bytes, which the header admission must match.
pub(crate) fn operand_lands(snapshot: &Path, layout: &RankWeightLayout, operand: &WeightOperand)
    -> Result<(Vec<(Rect, usize, bool)>, usize)> {
    let elem = element_bytes(&operand.dtype).with_context(|| format!("{}: unsupported precision", operand.name))? as usize;
    let mut lands = Vec::with_capacity(operand.sources.len());
    let mut at = 0usize;
    for source in &operand.sources {
        let meta = &source.meta;
        let narrow = meta.dtype != operand.dtype;
        ensure!(!narrow || (operand.dtype == DType::Bf16 && meta.dtype == DType::F32),
            "{}: native parameter precision changed", meta.name);
        let total = if meta.shape.len() >= 2 { meta.shape[0] } else { 1 };
        let bytes = usize::try_from(meta.byte_length)?;
        ensure!(total > 0 && bytes % total == 0, "{}: rows do not divide the tensor", meta.name);
        let whole = Rect { name: meta.name.clone(), path: snapshot.join(&source.shard), base: meta.byte_offset,
            row_bytes: bytes / total, rows: 0..total, cols: 0..bytes / total };
        let rect = match operand.slice {
            WeightSlice::Replicated => whole,
            WeightSlice::Rows | WeightSlice::Cols => {
                ensure!(operand.sources.len() == 1 && meta.shape.len() == 2, "{}: split source layout", meta.name);
                let (rank, ranks) = (layout.rank, layout.ranks);
                // Narrowing is elementwise: a source element's width selects the window.
                let elem = if narrow { 4 } else { elem };
                if operand.slice == WeightSlice::Rows {
                    ensure!(total % ranks == 0, "{}: rows do not split over {ranks} GPUs", meta.name);
                    whole.sub(rank * total / ranks..(rank + 1) * total / ranks)
                } else {
                    ensure!(meta.shape[1] % ranks == 0, "{}: columns do not split over {ranks} GPUs", meta.name);
                    let width = meta.shape[1] / ranks * elem;
                    Rect { cols: rank * width..(rank + 1) * width, ..whole }
                }
            }
        };
        let landed = if narrow { rect.bytes() / 2 } else { rect.bytes() };
        lands.push((rect, at, narrow));
        at += landed;
    }
    let expected = operand.shape.iter().try_fold(elem, |n: usize, dim| n.checked_mul(*dim))
        .context("packed size overflow")?;
    ensure!(at == expected && expected.max(256) as u64 == operand.allocation_bytes,
        "{}: packed storage disagrees with header admission", operand.name);
    Ok((lands, at))
}

/// `operand`'s bytes as dense records into `buffer` (allocated as the mount
/// allocates it): its [`operand_lands`], the unchanged rectangles as one
/// operand and each narrowed one as a staged step on the GPU. Filled through
/// the mount at once, or recorded for the coordinator push group.
pub(crate) fn load_records(device: Device<'_>, snapshot: &Path, layout: &RankWeightLayout, operand: &WeightOperand,
    buffer: CuteafdDeviceBuffer) -> Result<()> {
    let (lands, bytes) = operand_lands(snapshot, layout, operand)?;
    ensure!(buffer.bytes >= bytes, "{}: allocation smaller than its packed bytes", operand.name);
    let dense = DenseLoad::mount(device.library);
    let mut unchanged = Vec::with_capacity(lands.len());
    for (rect, at, narrow) in lands {
        if !narrow { unchanged.push((rect, at)); continue; }
        // F32 -> BF16 on the GPU: one tensor per step (CSA indexer tensors).
        let n = rect.bytes();
        ensure!(n <= STEP_BYTES, "{}: F32 narrowing step beyond the staging bound", rect.name);
        // The lands keep the narrowed values inside the packed bytes.
        let dst = CuteafdDeviceBuffer { ptr: buffer.ptr.wrapping_byte_add(at), bytes: n / 2, ..buffer };
        device.run(|| dense.step(Step { device: device.id, rects: vec![rect],
            jobs: vec![Job::NarrowF32Bf16 { at: 0, bytes: n, dst }] }, std::ptr::null_mut()))?;
    }
    if !unchanged.is_empty() {
        device.run(|| dense.operand(Operand { device: device.id, buffer, lands: unchanged, bytes }))?;
    }
    Ok(())
}


/// Staged steps a dense rank may hold open at once: one per push source
/// (SparkNest streams each file from one source; a rank's files come from at
/// most this many storage hosts). Pinned staging peaks at this many
/// [`STEP_BYTES`] buffers.
pub(crate) const MAX_OPEN_STEPS: usize = 8;

/// One GPU's operands and steps in its rank's logical space.
pub(crate) struct RankLayout {
    pub device: i32,
    /// (logical start, operand), ascending, device space `[0, base)`.
    pub operands: Vec<(u64, Operand)>,
    /// (start relative to `base`, step), ascending.
    pub steps: Vec<(u64, Step)>,
    pub base: u64,
    pub host_bytes: u64,
    /// Device-to-device transforms of this GPU's landed operands, run at Done.
    pub finishes: Vec<Finish>,
}

impl RankLayout {
    /// `device`'s records in loader order: 256-byte aligned with a gap byte
    /// between neighbours, so no slot fragment spans two of them.
    pub fn new(operands: Vec<Operand>, steps: Vec<Step>, device: i32) -> Result<Self> {
        let mut placed = Vec::new();
        let mut cursor = 0u64;
        for operand in operands.into_iter().filter(|o| o.device == device) {
            ensure!(operand.bytes > 0, "empty pushed operand");
            let start = cursor.next_multiple_of(256);
            cursor = start + operand.bytes as u64 + 1;
            placed.push((start, operand));
        }
        let base = cursor.next_multiple_of(256);
        let mut staged = Vec::new();
        let mut cursor = 0u64;
        for step in steps.into_iter().filter(|s| s.device == device) {
            let start = cursor.next_multiple_of(256);
            cursor = start + step.bytes() as u64 + 1;
            staged.push((start, step));
        }
        Ok(Self { device, operands: placed, steps: staged, base, host_bytes: if cursor == 0 { 0 } else { cursor },
            finishes: Vec::new() })
    }

    /// [`Self::new`] with `device`'s finishes from `finishes`.
    pub fn with_finishes(mut self, finishes: &[Finish]) -> Self {
        self.finishes = finishes.iter().filter(|f| f.device() == self.device).cloned().collect();
        self
    }

    /// The rank's entries as a device-free plan: operands from 0, steps from
    /// the `host_base` the group driver passes (`base` when alone).
    pub fn plan(&self) -> super::push::DensePlan {
        let operands: Vec<(u64, Vec<(Rect, usize)>)> = self.operands.iter()
            .map(|(start, o)| (*start, o.lands.clone())).collect();
        let steps: Vec<(u64, Vec<Rect>)> = self.steps.iter().map(|(start, s)| (*start, s.rects.clone())).collect();
        super::push::DensePlan { base: self.base, host_bytes: self.host_bytes,
            append: Box::new(move |builder, rank, host_base| {
                for (start, lands) in &operands {
                    for (rect, at) in lands { append(rect, builder, rank, start + *at as u64)?; }
                }
                for (start, rects) in &steps {
                    let mut at = host_base + start;
                    for rect in rects { append(rect, builder, rank, at)?; at += rect.bytes() as u64; }
                }
                Ok(())
            }) }
    }

    /// This rank's checked entries alone, in a group of `world` ranks.
    pub fn checked(&self, root: &std::path::Path, world: usize, rank: usize) -> Result<CheckedPlan> {
        let mut arenas = vec![0; world];
        *arenas.get_mut(rank).context("coordinator rank outside group")? = self.base + self.host_bytes;
        let mut builder = Builder::new(root, arenas);
        (self.plan().append)(&mut builder, u32::try_from(rank)?, self.base)?;
        Ok(builder.finish()?)
    }
}

/// A rank's operands as a push destination: device fragments land in the
/// operand allocations, host fragments fill their step's staging (taken from
/// a pool on its first fragment); a complete step runs its jobs at once and
/// returns its staging. Done requires exactly the planned footprint and every
/// step run; nothing reads these operands before commit.
pub(crate) struct DenseArena<'a> {
    device: Device<'a>,
    layout: RankLayout,
    expected: ExpectedCoverage,
    coverage: Coverage,
    kernels: LoadKernels<'a>,
    stream: *mut c_void,
    /// Open steps: index -> (staging, delivered bytes).
    open: std::collections::HashMap<usize, (HostAllocation<'a>, u64)>,
    pool: Vec<HostAllocation<'a>>,
    scratch: Option<crate::shared::memory::DeviceAllocation<'a>>,
    run: usize,
    /// Bytes of staging held by open steps now, and their peak.
    staged: u64,
    peak: u64,
    /// Bytes of staging allocated (open steps and the pool), and their peak.
    held: u64,
    peak_held: u64,
}

impl<'a> DenseArena<'a> {
    pub fn new(device: Device<'a>, layout: RankLayout, expected: ExpectedCoverage) -> Result<Self> {
        ensure!(layout.device == device.id, "operand layout belongs to another GPU");
        let stream = device.run(|| device.library.cuda_stream_create())?;
        let kernels = match LoadKernels::new(device.library) {
            Ok(kernels) => kernels,
            Err(error) => {
                // SAFETY: the stream was just created and has no work.
                device.run(|| unsafe { device.library.cuda_stream_destroy(stream) })?;
                return Err(error);
            }
        };
        Ok(Self { device, layout, expected, coverage: Coverage::default(), kernels, stream,
            open: Default::default(), pool: Vec::new(), scratch: None, run: 0, staged: 0, peak: 0, held: 0, peak_held: 0 })
    }

    fn operand(&self, offset: u64, bytes: u64) -> Result<(CuteafdDeviceBuffer, usize)> {
        let index = self.layout.operands.partition_point(|(start, _)| *start <= offset).checked_sub(1)
            .context("fragment before the first operand")?;
        let (start, operand) = &self.layout.operands[index];
        let at = usize::try_from(offset - start)?;
        ensure!(at.checked_add(usize::try_from(bytes)?).is_some_and(|end| end <= operand.bytes),
            "fragment crosses an operand boundary");
        Ok((operand.buffer, at))
    }

    fn step(&self, offset: u64, bytes: u64) -> Result<(usize, usize)> {
        let relative = offset.checked_sub(self.layout.base).context("host fragment below the steps")?;
        let index = self.layout.steps.partition_point(|(start, _)| *start <= relative).checked_sub(1)
            .context("fragment before the first step")?;
        let (start, step) = &self.layout.steps[index];
        let at = usize::try_from(relative - start)?;
        ensure!(at.checked_add(usize::try_from(bytes)?).is_some_and(|end| end <= step.bytes()),
            "fragment crosses a step boundary");
        Ok((index, at))
    }
}

impl Drop for DenseArena<'_> {
    fn drop(&mut self) {
        let library = self.device.library;
        let stream = self.stream;
        // SAFETY: every job drains its stream before returning; this owner's
        // stream has no other users.
        let _ = self.device.run(|| unsafe {
            library.cuda_stream_synchronize(stream)?;
            library.cuda_stream_destroy(stream)
        }).map_err(|error| tracing::error!(%error, "coordinator dense stream teardown"));
    }
}

impl PushDestination for DenseArena<'_> {
    type Delivered = ();
    fn belongs_to(&self, device: Device<'_>) -> bool {
        self.device.id == device.id && std::ptr::eq(self.device.library, device.library)
    }
    fn logical_bytes(&self) -> usize { (self.layout.base + self.layout.host_bytes) as usize }
    fn device_bytes(&self) -> usize { self.layout.base as usize }
    fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> {
        ensure!(bytes > 0 && self.expected.contains(offset, bytes), "fragment outside checked footprint");
        if offset < self.layout.base { self.operand(offset, bytes).map(|_| ()) } else { self.step(offset, bytes).map(|_| ()) }
    }
    fn destination(&self, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
        let (buffer, at) = self.operand(offset as u64, bytes as u64)?;
        // SAFETY: `at + bytes` lies inside this live operand allocation (checked above).
        Ok(CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(at).cast() }, bytes, ..buffer })
    }
    fn copy_validation(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        let length = bytes.len() as u64;
        ensure!(self.expected.contains(offset, length), "staged fragment outside checked footprint");
        let (index, at) = self.step(offset, length)?;
        let need = self.layout.steps[index].1.bytes();
        if !self.open.contains_key(&index) {
            // Admission planned one staged step per open source: a step whose
            // bytes interleave with others (sources in file order over a
            // layout that scatters a step) would grow pinned staging without
            // bound, so refuse it rather than pin unplanned memory.
            ensure!(self.open.len() < MAX_OPEN_STEPS,
                "coordinator dense rank on GPU {} would open {} staged steps ({} MiB pinned), beyond the planned {}                  (one {} MiB step per source): a step's bytes interleave with others in the push order",
                self.device.id, self.open.len() + 1, (self.open.len() + 1) * (STEP_BYTES >> 20), MAX_OPEN_STEPS,
                STEP_BYTES >> 20);
            // Uniform buffers (every step is within the bound, checked when
            // recorded), so returned staging always fits the next step.
            let staging = match self.pool.iter().position(|s| s.buffer.bytes >= need) {
                Some(found) => self.pool.swap_remove(found),
                None => {
                    let bytes = STEP_BYTES;
                    self.held += bytes as u64;
                    self.peak_held = self.peak_held.max(self.held);
                    HostAllocation::named(self.device.library, bytes, "staging/coordinator-dense")?
                }
            };
            self.staged += staging.buffer.bytes as u64;
            self.peak = self.peak.max(self.staged);
            self.open.insert(index, (staging, 0));
        }
        let (staging, delivered) = self.open.get_mut(&index).context("open step")?;
        staging.bytes_mut()[at..at + bytes.len()].copy_from_slice(bytes);
        *delivered += length;
        self.expected.record(&mut self.coverage, offset, length)?;
        if *delivered == need as u64 {
            let (staging, _) = self.open.remove(&index).context("complete step")?;
            let (library, stream, step) = (self.device.library, self.stream, &self.layout.steps[index].1);
            let (kernels, scratch) = (&mut self.kernels, &mut self.scratch);
            self.device.run(|| step.run(kernels, library, &staging.bytes()[..need], stream, scratch))?;
            self.run += 1;
            self.staged -= staging.buffer.bytes as u64;
            self.pool.push(staging);
        }
        Ok(())
    }
    fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> {
        ensure!(offset + bytes <= self.layout.base && self.expected.contains(offset, bytes),
            "completed fragment is outside checked plan footprint");
        self.expected.record(&mut self.coverage, offset, bytes)
    }
    fn done(self) -> Result<()> {
        ensure!(self.expected.complete(&self.coverage), "rank Done before all planned operand bytes completed");
        ensure!(self.open.is_empty() && self.run == self.layout.steps.len(), "rank Done before every staged step ran");
        let (library, stream) = (self.device.library, self.stream);
        let finished = std::time::Instant::now();
        self.device.run(|| {
            for finish in &self.layout.finishes { finish.run(library, stream)?; }
            // SAFETY: the finishes' buffers are this rank's live operand allocations.
            unsafe { library.cuda_stream_synchronize(stream) }
        })?;
        let finish_ms = finished.elapsed().as_secs_f64() * 1e3;
        tracing::info!(device = self.device.id, operands = self.layout.operands.len(), steps = self.run,
            finishes = self.layout.finishes.len(), finish_ms,
            peak_staged_mib = self.peak as f64 / (1u64 << 20) as f64,
            peak_pinned_mib = self.peak_held as f64 / (1u64 << 20) as f64, "coordinator dense rank delivered");
        #[cfg(test)]
        testing::PEAKS.with(|p| p.borrow_mut().push((self.peak, self.peak_held)));
        Ok(())
    }
}

/// Both of `devices`' (one or two GPUs) plans and destinations, as ranks
/// `sparks..`.
#[allow(clippy::type_complexity)]
pub(crate) fn plan_ranks<'a>(snapshot: &std::path::Path, operands: Vec<Operand>, steps: Vec<Step>,
    devices: &[Device<'a>], sparks: usize)
    -> Result<(Vec<super::push::DensePlan>, Vec<DenseArena<'a>>)> {
    plan_records(snapshot, Records { operands, steps, finishes: Vec::new() }, devices, sparks)
}

/// [`plan_ranks`] for a whole record set (each GPU's finishes run at its Done).
#[allow(clippy::type_complexity)]
pub(crate) fn plan_records<'a>(snapshot: &std::path::Path, records: Records,
    devices: &[Device<'a>], sparks: usize)
    -> Result<(Vec<super::push::DensePlan>, Vec<DenseArena<'a>>)> {
    let Records { operands, steps, finishes } = records;
    ensure!(finishes.iter().all(|f| devices.iter().any(|d| d.id == f.device())),
        "dense finish on a GPU outside the push group");
    let (root, _) = cuteafd_loader::push_load::namespace(snapshot)?;
    let world = sparks + devices.len();
    let mut operands = operands;
    let mut steps = steps;
    let mut out = (Vec::new(), Vec::new());
    for (gpu, device) in devices.iter().enumerate() {
        let (mine, rest): (Vec<_>, Vec<_>) = operands.into_iter().partition(|o| o.device == device.id);
        operands = rest;
        let (my_steps, rest): (Vec<_>, Vec<_>) = steps.into_iter().partition(|s| s.device == device.id);
        steps = rest;
        let layout = RankLayout::new(mine, my_steps, device.id)?.with_finishes(&finishes);
        let checked = layout.checked(&root, world, sparks + gpu)?;
        ensure!(checked.plan.files.iter().all(|f| cuteafd_loader::push_load::is_push_shard(&root.join(f))),
            "coordinator dense source is not a SparkNest shard");
        let expected = ExpectedCoverage::from_plan(&checked, u32::try_from(sparks + gpu)?)?;
        out.0.push(layout.plan());
        out.1.push(DenseArena::new(*device, layout, expected)?);
    }
    ensure!(operands.is_empty() && steps.is_empty(), "dense records on a GPU outside the push group");
    Ok(out)
}

#[cfg(test)]
pub(crate) mod testing {
    use anyhow::Result;

    thread_local! {
        /// (peak staged, peak pinned) bytes of each delivered dense rank on this thread.
        pub(crate) static PEAKS: std::cell::RefCell<Vec<(u64, u64)>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    use std::os::unix::fs::FileExt;

    /// A one-source fake SparkNest for one rank: packs the rank's planned runs
    /// into ring slots in plan order, Filled per slot (each written only after its
    /// previous use was released), then Done.
    pub(crate) struct FakeRank {
        ring: std::fs::File,
        slot_bytes: usize,
        slots: Vec<Vec<u8>>,
        next: std::cell::Cell<usize>,
        released: std::cell::RefCell<std::collections::BTreeSet<u32>>,
        capacity: usize,
    }

    impl FakeRank {
        pub(crate) fn new(plan: &cuteafd_loader::push_load::builder::CheckedPlan, mount: &std::path::Path, rank: u32,
            ring: &cuteafd_ffi::push_ring::RegisteredRing<'_>) -> Result<Self> {
            use cuteafd_loader::push_load::slot;
            let slot_bytes = ring.slot_bytes();
            let sources = plan.plan.files.iter().map(|f| std::fs::File::open(mount.join(f))).collect::<std::io::Result<Vec<_>>>()?;
            let (mut slots, mut packer) = (Vec::new(), slot::Packer::new(vec![0; slot_bytes]));
            for entry in plan.plan.entries.iter().filter(|e| e.rank == rank) {
                let pattern = &plan.plan.patterns[entry.pattern as usize];
                for (src, dst) in pattern.runs() {
                    let mut run = vec![0u8; pattern.run as usize];
                    sources[entry.file as usize].read_exact_at(&mut run, entry.src_base + src)?;
                    let mut bytes = &run[..];
                    let mut at = entry.dst_base + dst;
                    while !bytes.is_empty() {
                        let n = packer.push(at, bytes);
                        if n == 0 { slots.push(packer.finish().0); packer = slot::Packer::new(vec![0; slot_bytes]); continue; }
                        at += n as u64;
                        bytes = &bytes[n..];
                    }
                }
            }
            if !packer.is_empty() { slots.push(packer.finish().0); }
            use std::os::fd::AsFd;
            Ok(Self { ring: std::fs::File::from(ring.fd().as_fd().try_clone_to_owned()?), slot_bytes, slots,
                next: std::cell::Cell::new(0), released: Default::default(), capacity: ring.slots() })
        }
    }

    impl crate::shared::memory::push_consumer::RankEvents for FakeRank {
        fn next(&self) -> std::io::Result<cuteafd_loader::push_load::proto::Event> {
            use cuteafd_loader::push_load::proto::Event;
            let i = self.next.get();
            let Some(bytes) = self.slots.get(i) else { return Ok(Event::Done) };
            let slot = (i % self.capacity) as u32;
            if i >= self.capacity && !self.released.borrow_mut().remove(&slot) {
                return Err(std::io::Error::other("fake source reused an unreleased slot"));
            }
            self.ring.write_all_at(bytes, slot as u64 * self.slot_bytes as u64)?;
            self.next.set(i + 1);
            Ok(Event::Filled { slot, len: bytes.len() as u32 })
        }
        fn try_next(&self) -> std::io::Result<cuteafd_loader::push_load::Poll> {
            // Slots fill only after releases: report empty while every slot is in flight.
            let i = self.next.get();
            if i < self.slots.len() && i >= self.capacity && !self.released.borrow().contains(&((i % self.capacity) as u32)) {
                return Ok(cuteafd_loader::push_load::Poll::Empty);
            }
            self.next().map(cuteafd_loader::push_load::Poll::Ready)
        }
        fn release(&self, slot: u32) -> std::io::Result<()> {
            self.released.borrow_mut().insert(slot);
            Ok(())
        }
    }

    /// Replays `layout`'s checked plan for group rank `rank` (of `world`)
    /// through a ring into a [`super::DenseArena`] on `device`, reading the
    /// sources under `root`.
    pub(crate) fn replay(device: crate::shared::memory::device::Device<'_>, layout: super::RankLayout,
        root: &std::path::Path, world: usize, rank: usize) -> Result<()> {
        let checked = layout.checked(root, world, rank)?;
        let expected = crate::shared::memory::weight_arena::ExpectedCoverage::from_plan(&checked, rank as u32)?;
        device.run(|| -> Result<()> {
            let ring = cuteafd_ffi::push_ring::RegisteredRing::new(device.library, 8,
                cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES)?;
            let fake = FakeRank::new(&checked, root, rank as u32, &ring)?;
            let arena = super::DenseArena::new(device, layout, expected)?;
            crate::shared::memory::push_consumer::consume(device, arena, ring, fake, 8)
        })
    }
}
