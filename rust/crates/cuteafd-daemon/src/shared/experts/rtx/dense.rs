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
use crate::shared::memory::{device::Device, load_timing, push_consumer::PushDestination,
    weight_arena::{Coverage, ExpectedCoverage}, HostAllocation};
use crate::shared::fp8_load_quant::{LoadKernels, Target};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::push_load::builder::{Builder, CheckedPlan, Destination, Source};
use std::{ffi::c_void, ops::Range, os::unix::fs::FileExt, path::PathBuf};

/// The most bytes one step stages (one or more whole scale units).
pub(crate) const STEP_BYTES: usize = 32 << 20;

/// Rows `rows` of a checkpoint tensor (rows of `row_bytes` at `base` in `path`),
/// bytes `cols` of each, packed row by row.
#[derive(Clone, Debug)]
pub(crate) struct Rect {
    pub name: String,
    pub path: PathBuf,
    pub base: u64,
    pub row_bytes: usize,
    pub rows: Range<usize>,
    pub cols: Range<usize>,
}

impl Rect {
    pub fn width(&self) -> usize { self.cols.len() }
    pub fn bytes(&self) -> usize { self.rows.len() * self.cols.len() }

    /// Reads this rectangle into `out` (exactly [`Self::bytes`] long).
    pub fn read(&self, out: &mut [u8]) -> Result<()> { self.read_cached(out, &mut None) }

    /// [`Self::read`], keeping the whole rows of a partial-width read in
    /// `cache` so the next rank's column window of the same rows reads nothing.
    pub fn read_cached(&self, out: &mut [u8], cache: &mut Option<(Rect, Vec<u8>)>) -> Result<()> {
        ensure!(out.len() == self.bytes() && !self.rows.is_empty() && !self.cols.is_empty()
            && self.cols.end <= self.row_bytes, "{}: rectangle outside its tensor", self.name);
        let file = std::fs::File::open(&self.path).with_context(|| format!("opening {}", self.name))?;
        let start = self.base + (self.rows.start * self.row_bytes) as u64;
        if self.width() == self.row_bytes {
            return load_timing::read(out.len(), || file.read_exact_at(out, start))
                .with_context(|| format!("reading {}", self.name));
        }
        let whole = Rect { cols: 0..self.row_bytes, ..self.clone() };
        let hit = cache.as_ref().is_some_and(|(rows, _)| rows.path == whole.path && rows.base == whole.base
            && rows.row_bytes == whole.row_bytes && rows.rows == whole.rows);
        if !hit {
            let mut source = vec![0u8; self.rows.len() * self.row_bytes];
            load_timing::read(source.len(), || file.read_exact_at(&mut source, start))
                .with_context(|| format!("reading {}", self.name))?;
            *cache = Some((whole, source));
        }
        let (_, source) = cache.as_ref().context("row cache")?;
        for (row, chunk) in source.chunks_exact(self.row_bytes).zip(out.chunks_exact_mut(self.width())) {
            chunk.copy_from_slice(&row[self.cols.clone()]);
        }
        Ok(())
    }

    fn append(&self, builder: &mut Builder, rank: u32, offset: u64) -> Result<()> {
        let source = Source { path: self.path.clone(), offset: self.base,
            bytes: (self.rows.end * self.row_bytes) as u64 };
        builder.rows(&source, (self.rows.start * self.row_bytes + self.cols.start) as u64,
            u32::try_from(self.rows.len())?, u32::try_from(self.width())?, self.row_bytes as u64, self.width() as u64,
            Destination { rank, offset, bytes: self.bytes() as u64 })?;
        Ok(())
    }

    /// Rows `rows` (relative to this rectangle) of it.
    pub fn sub(&self, rows: Range<usize>) -> Self {
        Self { rows: self.rows.start + rows.start..self.rows.start + rows.end, ..self.clone() }
    }
}

/// A class-2 operand: rectangles landing unchanged at byte offsets of `buffer`
/// (a descriptor of an allocation the loader owns).
#[derive(Clone)]
pub(crate) struct Operand {
    pub device: i32,
    pub buffer: CuteafdDeviceBuffer,
    pub lands: Vec<(Rect, usize)>,
    pub bytes: usize,
}

/// One GPU transform of a step's staged bytes (byte offsets are into the staging).
#[derive(Clone)]
pub(crate) enum Job {
    /// Staged bytes `at..at + bytes` copied unchanged to `dst`.
    Upload { at: usize, bytes: usize, dst: CuteafdDeviceBuffer },
    /// BF16 `at..at + bytes` widened to F32 into `dst`.
    Widen { at: usize, bytes: usize, dst: CuteafdDeviceBuffer },
    /// Staged BF16 rows `rows` (of `pitch` elements), columns `col0..col0 + cols`,
    /// quantized to FP8 into `target`.
    Quant { rows: Range<usize>, pitch: usize, col0: usize, cols: usize, layout: crate::families::glm5_flash::fp8::Layout,
        rule: crate::shared::fp8_linear::Fp8Scales, target: Target },
    /// The staged `kv_b` rows of `heads` heads absorbed into `uk`/`uv`.
    Absorb { heads: usize, nope: usize, v: usize, lat: usize, uk: CuteafdDeviceBuffer, uv: CuteafdDeviceBuffer },
    /// The staged raw F32 `A_log` keeps `heads` active heads: its padding is +0
    /// and its live values finite.
    ALog { heads: usize },
}

/// A class-3 transform: `rects` staged back to back, then `jobs` on `device`
/// (descriptors of allocations the loader owns).
#[derive(Clone)]
pub(crate) struct Step {
    pub device: i32,
    pub rects: Vec<Rect>,
    pub jobs: Vec<Job>,
}

impl Step {
    pub fn bytes(&self) -> usize { self.rects.iter().map(Rect::bytes).sum() }

    /// Runs the jobs on the current device (`device`) from `staged`.
    pub fn run(&self, kernels: &mut LoadKernels<'_>, library: &NativeLibrary, staged: &[u8], stream: *mut c_void)
        -> Result<()> {
        ensure!(staged.len() >= self.bytes(), "step staging is short");
        for job in &self.jobs {
            match *job {
                Job::Upload { at, bytes, dst } => {
                    ensure!(at + bytes <= staged.len() && bytes <= dst.bytes, "upload outside its step");
                    load_timing::upload(bytes, || library.copy_h2d(CuteafdDeviceBuffer { bytes, ..dst },
                        &staged[at..at + bytes]))?;
                }
                Job::Widen { at, bytes, dst } => {
                    ensure!(at + bytes <= staged.len(), "widening outside its step");
                    load_timing::quantize(bytes, || kernels.widen_from(&staged[at..at + bytes], dst, stream))?;
                }
                Job::Quant { ref rows, pitch, col0, cols, layout, rule, target } => {
                    load_timing::quantize(rows.len() * cols * 2, || kernels.quantize_from(staged, rows.clone(),
                        pitch, col0, cols, layout, rule, target, stream))?;
                }
                Job::Absorb { heads, nope, v, lat, uk, uv } => {
                    load_timing::quantize(heads * (nope + v) * lat * 2, || kernels.absorb_from(staged, heads, 0,
                        heads, nope, v, lat, uk, uv, stream))?;
                }
                Job::ALog { heads } => {
                    cuteafd_loader::kda::narrow_a_log(&staged[..self.bytes()], heads)?;
                }
            }
        }
        Ok(())
    }
}

/// `rects` (rows of equal width, back to back) as steps of at most
/// [`STEP_BYTES`] in whole `unit`-row groups: each step's rectangles and the
/// row range of the whole it covers.
pub(crate) fn windows(rects: &[Rect], unit: usize) -> Result<Vec<(Vec<Rect>, Range<usize>)>> {
    let width = rects.first().context("no rows to stage")?.width();
    ensure!(rects.iter().all(|r| r.width() == width), "staged rows differ in width");
    let total: usize = rects.iter().map(|r| r.rows.len()).sum();
    ensure!(unit > 0 && total % unit == 0, "staged rows are not whole {unit}-row units");
    let per = (STEP_BYTES / width / unit).max(1) * unit;
    let mut out = Vec::new();
    let mut start = 0;
    while start < total {
        let end = (start + per).min(total);
        let (mut parts, mut at) = (Vec::new(), 0);
        for rect in rects {
            let n = rect.rows.len();
            let (lo, hi) = (start.max(at), end.min(at + n));
            if lo < hi { parts.push(rect.sub(lo - at..hi - at)); }
            at += n;
        }
        out.push((parts, start..end));
        start = end;
    }
    Ok(out)
}

/// Push-mode loader output: class-2 operands and class-3 steps, in loader order.
#[derive(Default)]
pub(crate) struct DenseSink {
    operands: std::cell::RefCell<Vec<Operand>>,
    steps: std::cell::RefCell<Vec<Step>>,
}

impl DenseSink {
    pub fn operand(&self, operand: Operand) -> Result<()> {
        ensure!(operand.bytes <= operand.buffer.bytes
            && operand.lands.iter().all(|(r, at)| at + r.bytes() <= operand.bytes), "pushed landing outside its operand");
        self.operands.borrow_mut().push(operand);
        Ok(())
    }
    pub fn step(&self, step: Step) -> Result<()> {
        ensure!(step.bytes() > 0 && step.bytes() <= STEP_BYTES.max(step.rects.iter().map(Rect::bytes).max().unwrap_or(0)),
            "step stages too many bytes");
        self.steps.borrow_mut().push(step);
        Ok(())
    }
    /// The recorded operands and steps (leaves the sink empty).
    pub fn take(&self) -> (Vec<Operand>, Vec<Step>) { (self.operands.take(), self.steps.take()) }
}

/// Mount-path runner: reads into operands, or into one reused pinned buffer
/// for a step's jobs.
pub(crate) struct Mount<'a> {
    library: &'a NativeLibrary,
    kernels: LoadKernels<'a>,
    staging: Option<HostAllocation<'a>>,
    /// The whole rows of the last partial-width read (column shares of one tensor).
    rows: Option<(Rect, Vec<u8>)>,
}

impl<'a> Mount<'a> {
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self { library, kernels: LoadKernels::new(library)?, staging: None, rows: None })
    }

    /// Fills `operand` on the current device.
    pub fn operand(&mut self, operand: &Operand) -> Result<()> {
        let mut bytes = vec![0u8; operand.bytes];
        for (rect, at) in &operand.lands { rect.read_cached(&mut bytes[*at..*at + rect.bytes()], &mut self.rows)?; }
        load_timing::upload(bytes.len(), || self.library.copy_h2d(
            CuteafdDeviceBuffer { bytes: operand.bytes, ..operand.buffer }, &bytes))
    }

    /// Stages and runs `step` on the current device (`stream`).
    pub fn step(&mut self, step: &Step, stream: *mut c_void) -> Result<()> {
        let bytes = step.bytes();
        if self.staging.as_ref().is_none_or(|s| s.buffer.bytes < bytes) {
            drop(self.staging.take());
            self.staging = Some(HostAllocation::named(self.library, bytes.max(1 << 20), "staging/coordinator-dense")?);
        }
        let staging = self.staging.as_mut().context("mount staging")?;
        let mut at = 0;
        for rect in &step.rects {
            rect.read_cached(&mut staging.bytes_mut()[at..at + rect.bytes()], &mut self.rows)?;
            at += rect.bytes();
        }
        step.run(&mut self.kernels, self.library, &staging.bytes()[..bytes], stream)
    }
}

/// Fills recorded operands and runs recorded steps through the mount (the
/// whole push group fell back), each on its GPU; `stream` names a GPU's stream.
pub(crate) fn fill_by_mount(library: &NativeLibrary, operands: &[Operand], steps: &[Step],
    stream: impl Fn(i32) -> *mut c_void) -> Result<()> {
    let mut mount = Mount::new(library)?;
    for operand in operands {
        Device { library, id: operand.device }.run(|| mount.operand(operand))?;
    }
    for step in steps {
        Device { library, id: step.device }.run(|| mount.step(step, stream(step.device)))?;
    }
    Ok(())
}

/// One GPU's operands and steps in its rank's logical space.
pub(crate) struct RankLayout {
    pub device: i32,
    /// (logical start, operand), ascending, device space `[0, base)`.
    pub operands: Vec<(u64, Operand)>,
    /// (start relative to `base`, step), ascending.
    pub steps: Vec<(u64, Step)>,
    pub base: u64,
    pub host_bytes: u64,
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
        Ok(Self { device, operands: placed, steps: staged, base, host_bytes: if cursor == 0 { 0 } else { cursor } })
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
                    for (rect, at) in lands { rect.append(builder, rank, start + *at as u64)?; }
                }
                for (start, rects) in &steps {
                    let mut at = host_base + start;
                    for rect in rects { rect.append(builder, rank, at)?; at += rect.bytes() as u64; }
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
    run: usize,
    staged: u64,
    peak: u64,
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
            open: Default::default(), pool: Vec::new(), run: 0, staged: 0, peak: 0 })
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
            let staging = match self.pool.iter().position(|s| s.buffer.bytes >= need) {
                Some(found) => self.pool.swap_remove(found),
                None => HostAllocation::named(self.device.library, need.max(STEP_BYTES.min(1 << 20)), "staging/coordinator-dense")?,
            };
            self.staged += staging.buffer.bytes as u64;
            self.peak = self.peak.max(self.staged);
            self.open.insert(index, (staging, 0));
        }
        let (staging, delivered) = self.open.get_mut(&index).context("open step")?;
        staging.bytes_mut()[at..at + bytes.len()].copy_from_slice(bytes);
        *delivered += length;
        self.coverage.insert(offset, length)?;
        if *delivered == need as u64 {
            let (staging, _) = self.open.remove(&index).context("complete step")?;
            let (library, stream, step) = (self.device.library, self.stream, &self.layout.steps[index].1);
            let kernels = &mut self.kernels;
            self.device.run(|| step.run(kernels, library, &staging.bytes()[..need], stream))?;
            self.run += 1;
            self.pool.push(staging);
        }
        Ok(())
    }
    fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> {
        ensure!(offset + bytes <= self.layout.base && self.expected.contains(offset, bytes),
            "completed fragment is outside checked plan footprint");
        self.coverage.insert(offset, bytes)
    }
    fn done(self) -> Result<()> {
        ensure!(self.expected.complete(&self.coverage), "rank Done before all planned operand bytes completed");
        ensure!(self.open.is_empty() && self.run == self.layout.steps.len(), "rank Done before every staged step ran");
        tracing::info!(device = self.device.id, operands = self.layout.operands.len(), steps = self.run,
            peak_staged_mib = self.peak as f64 / (1u64 << 20) as f64, "coordinator dense rank delivered");
        Ok(())
    }
}

/// Both of `devices`' (one or two GPUs) plans and destinations, as ranks
/// `sparks..`.
#[allow(clippy::type_complexity)]
pub(crate) fn plan_ranks<'a>(snapshot: &std::path::Path, operands: Vec<Operand>, steps: Vec<Step>,
    devices: &[Device<'a>], sparks: usize)
    -> Result<(Vec<super::push::DensePlan>, Vec<DenseArena<'a>>)> {
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
        let layout = RankLayout::new(mine, my_steps, device.id)?;
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
