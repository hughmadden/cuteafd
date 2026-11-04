//! `CUTEAFD_V41_CHECK_STAGING=1`: catches the host rewriting pinned staging
//! that a queued upload has not read yet.
//!
//! A device-ordered pass runs the host ahead of the GPU, so a staging buffer
//! is safe to rewrite only once the upload of its previous contents ran. The
//! stage chain's per-layer fences guarantee that when they are marked after
//! those uploads joined the chain; in October 2026 one was marked too early
//! and the host rewrote attention metadata the GPU had not read yet. Each
//! checked staging region records an event after its upload is queued, and
//! the host queries that event, without blocking, before it rewrites the
//! region: an upload still pending is an error naming the site. Off by
//! default (no events, no queries); on in tests and in busy-lane runs.
//! Uploads are queued outside graph capture at every checked site.
use anyhow::{ensure, Result};
use cuteafd_ffi::NativeLibrary;
use std::cell::RefCell;
use std::ffi::c_void;

pub(crate) const ENV: &str = "CUTEAFD_V41_CHECK_STAGING";

/// Whether the process checks staging (`CUTEAFD_V41_CHECK_STAGING=1`), read once.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(ENV).is_ok_and(|value| value.trim() == "1"))
}

struct Upload {
    device: i32,
    event: *mut c_void,
    pending: bool,
}

/// The check for one staging owner with `regions` regions (a
/// [`super::LayerStaging`]'s layers, or 1 for a single buffer).
pub(crate) struct StagingCheck<'a> {
    library: &'a NativeLibrary,
    site: &'static str,
    on: bool,
    uploads: RefCell<Vec<Option<Upload>>>,
}
impl<'a> StagingCheck<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, site: &'static str, regions: usize) -> Self {
        Self::with(library, site, regions, enabled())
    }
    /// [`Self::new`] on or off regardless of the environment (tests).
    pub(crate) fn with(library: &'a NativeLibrary, site: &'static str, regions: usize, on: bool) -> Self {
        Self { library, site, on, uploads: RefCell::new((0..regions.max(1)).map(|_| None).collect()) }
    }
    fn on_device<T>(&self, device: i32, work: impl FnOnce() -> Result<T>) -> Result<T> {
        let previous = self.library.cuda_get_device()?;
        self.library.cuda_set_device(device)?;
        let result = work();
        self.library.cuda_set_device(previous)?;
        result
    }
    /// Before the host rewrites region `region` (modulo the regions): fails
    /// if the upload queued from its previous contents has not run.
    pub(crate) fn before_write(&self, region: usize) -> Result<()> {
        if !self.on { return Ok(()) }
        let mut uploads = self.uploads.borrow_mut();
        let index = region % uploads.len();
        let Some(upload) = uploads[index].as_mut().filter(|upload| upload.pending) else { return Ok(()) };
        // SAFETY: the event belongs to this check, on its recorded device.
        let ran = self.on_device(upload.device, || unsafe { self.library.cuda_event_query(upload.event) })?;
        ensure!(ran, "{}: the host is rewriting pinned staging (region {index}) before the GPU read its \
            previous upload ({ENV})", self.site);
        upload.pending = false;
        Ok(())
    }
    /// After queueing the upload that reads region `region` on `stream`.
    ///
    /// # Safety
    /// `stream` is a live stream of the current device, not capturing.
    pub(crate) unsafe fn uploaded(&self, region: usize, stream: *mut c_void) -> Result<()> {
        if !self.on { return Ok(()) }
        let device = self.library.cuda_get_device()?;
        let mut uploads = self.uploads.borrow_mut();
        let index = region % uploads.len();
        if uploads[index].as_ref().is_some_and(|upload| upload.device != device) {
            let upload = uploads[index].take().expect("checked upload");
            // SAFETY: the event was recorded on its own device; its owner drops it here.
            self.on_device(upload.device, || unsafe { self.library.cuda_event_destroy(upload.event) })?;
        }
        if uploads[index].is_none() {
            uploads[index] = Some(Upload { device, event: self.library.cuda_event_create_ordering()?, pending: false });
        }
        let upload = uploads[index].as_mut().expect("checked upload");
        // SAFETY: the stream and event are live on the current device.
        unsafe { self.library.cuda_event_record(upload.event, stream)? };
        upload.pending = true;
        Ok(())
    }
}
impl Drop for StagingCheck<'_> {
    fn drop(&mut self) {
        for upload in std::mem::take(self.uploads.get_mut()).into_iter().flatten() {
            // SAFETY: each event belongs to this check, on its recorded device.
            let destroyed = self.on_device(upload.device, || unsafe { self.library.cuda_event_destroy(upload.event) });
            if let Err(error) = destroyed { tracing::error!(%error, site = self.site, "destroying staging check event"); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StagingCheck;
    use crate::shared::memory::chain::{self, StageChain};
    use crate::shared::memory::device::{Allocation, Device, Stream};
    use crate::shared::memory::HostAllocation;
    use anyhow::{ensure, Result};
    use cuteafd_ffi::NativeLibrary;
    use std::future::Future;
    use std::sync::atomic::Ordering;

    struct Outcome {
        /// Layer L+1's fence wait was still pending while the gate held.
        fence_held: bool,
        /// What layer L's upload carried.
        uploaded: Vec<u8>,
        /// The staging check's verdict on layer L+1's rewrite.
        check: Result<()>,
    }

    /// The October 2026 device-order corruption, reduced to its mechanism on
    /// one GPU: layer L's attention uploads pinned positions on a stream that
    /// the other lane's queued work holds (a gate), and layer L+1 rewrites the
    /// staging as soon as its schedule's fence wait returns. `fence_after_tail`
    /// marks the fence at the FFN entry, after the attention tail joined the
    /// chain (the fix, `v41_backbone_lane`); otherwise right after preparation,
    /// while the query producer still heads the chain (the bug).
    fn layer_pair(lib: &NativeLibrary, fence_after_tail: bool) -> Result<Outcome> {
        let device = Device { library: lib, id: 0 };
        let chain = StageChain::on_devices(lib, &[0])?;
        chain.set_device_order(true);
        let (query, attention) = (Stream::new(device)?, Stream::new(device)?);
        let mut staging = HostAllocation::new(lib, 64)?;
        let uploaded = Allocation::new(device, 64)?;
        let check = StagingCheck::with(lib, "October regression positions", 1, true);
        let (original, rewritten) = ([0x11u8; 64], [0x77u8; 64]);
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        // SAFETY: the attention stream outlives the gate (dropped first below).
        let gate = unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(lib, attention.raw)? };
        let release = gate.release_handle();
        let outcome = runtime.block_on(chain.handle().scope(async {
            // SAFETY: both streams are this device's; staging and the upload
            // target outlive the drains below.
            unsafe {
                // Layer L: the query producer heads the chain.
                chain::join(lib, query.raw)?;
                chain::finish(lib, query.raw)?;
                // Preparation: stage the positions and queue their upload on
                // the attention stream, behind the other lane's work.
                check.before_write(0)?;
                staging.bytes_mut().copy_from_slice(&original);
                chain::join(lib, attention.raw)?;
                lib.copy_host_buffer_h2d_async(uploaded.buffer, staging.buffer, 64, attention.raw)?;
                check.uploaded(0, attention.raw)?;
                if !fence_after_tail { chain::fence_mark(lib, 0)?; }
                chain::finish(lib, attention.raw)?;
                if fence_after_tail { chain::fence_mark(lib, 0)?; }
            }
            // Layer L+1: its schedule waits for fence 0, polled for 200 ms
            // while the gate holds.
            let mut wait = std::pin::pin!(chain::fence_wait(lib, 0));
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(200);
            let mut fence_held = true;
            while std::time::Instant::now() < deadline {
                let polled = std::future::poll_fn(|context| std::task::Poll::Ready(wait.as_mut().poll(context))).await;
                if let std::task::Poll::Ready(result) = polled { result?; fence_held = false; break; }
                tokio::task::yield_now().await;
            }
            if fence_held {
                release.store(true, Ordering::Release);
                wait.await?;
            }
            // The host rewrites the staging for layer L+1 once the wait returned.
            let check = check.before_write(0);
            staging.bytes_mut().copy_from_slice(&rewritten);
            release.store(true, Ordering::Release);
            anyhow::Ok((fence_held, check))
        }));
        let drained = chain.drain().and(attention.drain()).and(query.drain());
        drop(gate);
        let (fence_held, check) = outcome?;
        drained?;
        let mut bytes = vec![0; 64];
        lib.copy_d2h(&mut bytes, uploaded.buffer)?;
        ensure!(bytes == original || bytes == rewritten, "upload carried neither staging image");
        Ok(Outcome { fence_held, uploaded: bytes, check })
    }

    fn library() -> Result<NativeLibrary> {
        // SAFETY: the explicitly selected native library outlives the test.
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        Ok(lib)
    }

    /// The fix: the next layer's fence wait stays pending until the gate
    /// opens, the upload carries the original positions, and the check passes.
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and a CUDA GPU"]
    fn fence_after_the_attention_tail_keeps_staging_until_its_upload_ran() -> Result<()> {
        let lib = library()?;
        let outcome = layer_pair(&lib, true)?;
        ensure!(outcome.fence_held, "the fence released before the gated upload ran");
        ensure!(outcome.uploaded == [0x11; 64], "the upload carried the rewritten positions");
        outcome.check?;
        eprintln!("fence after the attention tail: wait held, upload intact, staging check clean");
        Ok(())
    }

    /// Negative control (the October placement): the fence releases at once,
    /// the host rewrites first, the upload carries the rewritten positions,
    /// and the staging check names the rewrite.
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and a CUDA GPU"]
    fn fence_before_the_attention_tail_lets_the_host_rewrite_queued_staging() -> Result<()> {
        let lib = library()?;
        let outcome = layer_pair(&lib, false)?;
        ensure!(!outcome.fence_held, "the early fence waited for the gated upload");
        ensure!(outcome.uploaded == [0x77; 64], "the upload escaped the early rewrite");
        let error = outcome.check.err().ok_or_else(|| anyhow::anyhow!("the staging check missed the rewrite"))?;
        ensure!(format!("{error:#}").contains("October regression positions"), "check names another site: {error:#}");
        eprintln!("fence before the attention tail: corrupted as in October, staging check: {error:#}");
        Ok(())
    }
}
