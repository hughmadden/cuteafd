//! The coordinator's one stream and the MiniMax program table: every launch,
//! copy and upload of a step is queued here. Operands are views of the
//! execution's own allocations (weights, scratch, KV cache, step input),
//! which outlive the stream's work: every entry point drains before it
//! returns, and the stream drains before any owner is released.
use crate::shared::{memory::{BlockingWait, device::{CancellationDrain, Device, Stream}}, prefix::view};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{minimax::MiniMaxPrograms, programs::{Programs, Scalar}, CuteafdDeviceBuffer, NativeLibrary};
use std::{cell::RefCell, ffi::c_void};

pub(super) struct Native<'a> {
    stream: Stream<'a>,
    programs: MiniMaxPrograms<'a>,
    /// The per-step host wait under `--host-wait`.
    round: RefCell<BlockingWait<'a>>,
}

impl<'a> Native<'a> {
    pub fn new(device: Device<'a>, programs: &'a Programs<'a>) -> Result<Self> {
        let programs = device.run(|| MiniMaxPrograms::load(programs, device.id))?;
        Ok(Self { stream: Stream::new(device)?, programs, round: RefCell::new(BlockingWait::new(device.library)) })
    }
    pub fn device(&self) -> Device<'a> { self.stream.device }
    pub fn library(&self) -> &'a NativeLibrary { self.stream.device.library }
    pub fn raw(&self) -> *mut c_void { self.stream.raw }
    /// A failed drain quarantines the module: storage queued work may still
    /// use is never freed.
    pub fn drain(&self) -> Result<()> {
        let drained = self.stream.drain();
        if drained.is_err() { self.library().quarantine_module_after_failed_drain(); }
        drained
    }
    /// A step's host wait for its queued work, as the process `--host-wait` policy asks (block:
    /// poll a blocking-sync event, then sleep on it; spin: a stream synchronize). A failed wait
    /// drains instead, so the stream is drained or the module quarantined either way.
    pub fn wait(&self) -> Result<()> {
        // SAFETY: the stream is this execution's own, waited on its device.
        let waited = self.device().run(|| unsafe { self.round.borrow_mut().wait(self.stream.raw) });
        match waited {
            Ok(()) => Ok(()),
            Err(error) => self.drain().and(Err(error)),
        }
    }
    /// Drains (or aborts) on unwind unless completed: an entry point's
    /// borrowed operands outlive the work it queued.
    pub fn cancellation_guard(&self, site: &'static str) -> CancellationDrain<'_, 'a> {
        self.stream.cancellation_guard(site)
    }
    pub fn check_device(&self, buffers: &[CuteafdDeviceBuffer]) -> Result<()> { self.programs.check_device(buffers) }
    pub fn capacity(&self, name: &str) -> Result<usize> { self.programs.capacity(name) }
    pub fn projection(&self, projection: &str, rows: usize) -> String { self.programs.projection(projection, rows) }
    pub fn launch(&self, name: &str, operands: &[CuteafdDeviceBuffer], scalars: &[Scalar]) -> Result<()> {
        // SAFETY: the operands are views of this execution's allocations, which
        // outlive the stream's work (module docs).
        unsafe { self.programs.launch(name, operands, scalars, self.stream.raw) }
    }
    pub fn index_stage(&self, cache: CuteafdDeviceBuffer, pages: CuteafdDeviceBuffer, count: usize,
        keys: CuteafdDeviceBuffer, scales: CuteafdDeviceBuffer) -> Result<()> {
        // SAFETY: as `launch`; the page ids are the batch's validated pool pages.
        unsafe { self.programs.index_stage(cache, pages, count, keys, scales, self.stream.raw) }
    }
    pub fn copy(&self, destination: CuteafdDeviceBuffer, source: CuteafdDeviceBuffer, bytes: usize) -> Result<()> {
        ensure!(bytes <= destination.bytes && bytes <= source.bytes, "MiniMax copy extent");
        self.check_device(&[destination, source])?;
        // SAFETY: checked views of live owners, ordered on this stream.
        unsafe { self.library().copy_d2d_async(destination, source, bytes, self.stream.raw) }
    }
    pub fn zero(&self, buffer: CuteafdDeviceBuffer) -> Result<()> {
        self.check_device(&[buffer])?;
        // SAFETY: the whole view belongs to a live owner.
        unsafe { self.library().cuda_zero_bytes_async(buffer, buffer.bytes, self.stream.raw) }
    }
    /// Uploads `bytes` to the start of `buffer` after the stream's queued work. `buffer` is the
    /// whole admitted destination (or its tail from a fixed offset), never a view narrowed to
    /// `bytes`: the first ordered upload into a destination sizes its pinned staging, which a
    /// later, longer upload must fit ([`Self::reserve`] creates it before readiness).
    pub fn upload(&self, buffer: CuteafdDeviceBuffer, bytes: &[u8]) -> Result<()> {
        ensure!(bytes.len() <= buffer.bytes, "MiniMax ordered upload extent");
        self.check_device(&[buffer])?;
        // SAFETY: the ordered upload owns its host staging; the destination's owner is live.
        unsafe { self.library().copy_h2d_ordered(buffer, bytes, self.stream.raw) }
    }
    /// Creates `buffer`'s ordered-upload staging, sized to its whole extent, with no copy: before
    /// readiness, so no upload allocates pinned memory (which can synchronize the device) later.
    pub fn reserve(&self, buffer: CuteafdDeviceBuffer) -> Result<()> {
        self.check_device(&[buffer])?;
        // SAFETY: as `upload`; nothing is copied.
        unsafe { self.library().reserve_h2d_ordered(buffer, self.stream.raw) }
    }
    /// `destination[..count]` (F32) = `source[..count]` (BF16).
    pub fn widen(&self, source: CuteafdDeviceBuffer, destination: CuteafdDeviceBuffer, count: usize) -> Result<()> {
        // Only the live rows are accumulated into, so only they are cleared.
        self.zero(view(destination, 0, count.checked_mul(4).context("MiniMax widen extent")?)?)?;
        self.check_device(&[source, destination])?;
        // SAFETY: the native routine checks both extents; the distinct live
        // allocations are initialized by the preceding stream operations.
        unsafe { self.library().cuda_accumulate_bf16_to_f32_async(source, destination, count, self.stream.raw) }
    }
    pub fn add_bf16(&self, a: CuteafdDeviceBuffer, b: CuteafdDeviceBuffer, out: CuteafdDeviceBuffer, count: usize)
        -> Result<()> {
        self.check_device(&[a, b, out])?;
        // SAFETY: the binding checks shapes, and all three owners are live.
        unsafe { self.library().bf16_add()?.launch(a, b, out, count, self.stream.raw) }
    }
}
