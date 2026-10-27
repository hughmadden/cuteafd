//! Owned native gathered-row upload and BF16 engram embedding production.
pub(crate) mod layer;
pub(crate) mod placement;
use crate::shared::memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use std::rc::Rc;
use crate::shared::memory::row_window::{RowWindow, ScratchGuard, MAX_RESERVE_ROWS};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{
    EngramGatherPoll, EngramGatherTicket, EngramGatherView, EngramPipeline, EngramWave,
};

#[cfg(test)]
thread_local! {
    static SKIP_ACQUIRE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Unit-cap input-hook scratch. Embeddings and text_mask are guarded scratch
/// read by the gate stream; release after that reader joins. Packed weights and
/// scales are plain scratch uploaded/dequantized on the shared upload stream.
/// Gather staging (including its two layer regions) and ready state are per slot.
pub(crate) struct EngramRowsStorage<'a> {
    stream: LoadStream<'a>,
    guard: ScratchGuard<'a>,
    weights: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
    embeddings: DeviceAllocation<'a>,
    text_mask: DeviceAllocation<'a>,
    rows: usize,
    unit: usize,
}
impl Drop for EngramRowsStorage<'_> {
    fn drop(&mut self) {
        // SAFETY: storage owns this stream and all of its scratch/row planes.
        crate::shared::decode_graph::fatal_drain(unsafe {
            self.stream.library.cuda_stream_synchronize(self.stream.raw)
        }, "Engram row storage");
    }
}
pub(crate) struct EngramDeviceRows<'a> {
    storage: Rc<EngramRowsStorage<'a>>,
    window: RowWindow,
    staging: crate::shared::memory::LayerStaging<'a>,
    library: &'a NativeLibrary,
    capacity: usize,
    ready: Option<(usize, usize)>,
}
/// Borrowed buffers: never free or retain across owner reuse/drop.
pub(crate) struct EngramDeviceView {
    pub embeddings: CuteafdDeviceBuffer,
    pub text_mask: CuteafdDeviceBuffer,
    pub rows: usize,
    pub layer_index: usize,
}
pub(crate) enum EngramUploadPoll {
    Pending,
    Cancelled,
    Ready(EngramDeviceView),
}
impl<'a> EngramDeviceRows<'a> {
    pub(crate) fn library(&self) -> &'a cuteafd_ffi::NativeLibrary { self.storage.stream.library }
    /// Validate current request generations before consuming an early gather.
    pub fn poll_wave(
        &mut self,
        pipeline: &EngramPipeline,
        wave: &mut EngramWave,
        histories: &[&cuteafd_core::EngramHistory],
        layer: usize,
    ) -> Result<EngramUploadPoll> {
        self.ready = None;
        Ok(match pipeline.poll(wave, histories, layer)? {
            EngramGatherPoll::Pending => {
                tracing::debug!(target: "cuteafd::timing", layer, upload_owner=self as *const Self as usize, "engram IO pending");
                EngramUploadPoll::Pending
            }
            EngramGatherPoll::Cancelled => EngramUploadPoll::Cancelled,
            EngramGatherPoll::Ready(lease) => {
                let gathered = lease.view()?;
                if let Some(timing) = lease.timing() {
                    tracing::debug!(target: "cuteafd::timing", layer, rows=gathered.rows,
                        upload_owner=self as *const Self as usize,
                        queue_us=timing.queued.as_micros() as u64,
                        gather_us=timing.gather.as_micros() as u64,
                        ready_age_us=timing.completed.elapsed().as_micros() as u64,
                        minor_faults=timing.minor_faults, major_faults=timing.major_faults,
                        input_blocks=timing.input_blocks, "engram IO delivery");
                }
                EngramUploadPoll::Ready(self.upload(&gathered)?)
            }
        })
    }
    /// Poll on the CUDA-owning thread and recycle ready staging after upload.
    /// The scheduler must cancel tickets whose request history has been invalidated.
    pub fn poll_upload(&mut self, ticket: &mut EngramGatherTicket) -> Result<EngramUploadPoll> {
        self.ready = None;
        Ok(match ticket.poll()? {
            EngramGatherPoll::Pending => EngramUploadPoll::Pending,
            EngramGatherPoll::Cancelled => EngramUploadPoll::Cancelled,
            EngramGatherPoll::Ready(lease) => {
                let output = self.upload(&lease.view()?)?;
                // Upload copies into slot-owned pinned staging before the lease is recycled.
                EngramUploadPoll::Ready(output)
            }
        })
    }
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid engram device capacity"
        );
        capacity
            .checked_mul(24 * (256 + 8 + 512) + 1)
            .context("engram device budget overflow")
    }
    pub fn new(
        library: &'a NativeLibrary,
        capacity: usize,
        available_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(capacity)? <= available_bytes,
            "engram rows exceed device budget"
        );
        Self::on_storage(Self::storage(library, capacity, capacity)?)
    }
    pub(crate) fn storage(library: &'a NativeLibrary, rows: usize, unit: usize)
        -> Result<Rc<EngramRowsStorage<'a>>> {
        ensure!((1..=MAX_RESERVE_ROWS).contains(&rows) && (1..=4096).contains(&unit)
            && unit <= rows, "invalid Engram row storage dimensions");
        Ok(Rc::new(EngramRowsStorage {
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            guard: ScratchGuard::new(library, &[library.cuda_get_device()?])?,
            weights: DeviceAllocation::new(library, unit * 24 * 256)?,
            scales: DeviceAllocation::new(library, unit * 24 * 8)?,
            embeddings: DeviceAllocation::new(library, unit * 24 * 512)?,
            text_mask: DeviceAllocation::new(library, unit)?, rows, unit,
        }))
    }
    pub(crate) fn on_storage(storage: Rc<EngramRowsStorage<'a>>) -> Result<Self> {
        let library = storage.stream.library;
        let capacity = storage.unit;
        Ok(Self { storage, library, capacity, window: RowWindow::NONE, ready: None,
            staging: crate::shared::memory::LayerStaging::new(library,
                capacity * (24 * (256 + 8) + 1), 2)?,
        })
    }
    /// The slot must have drained before reusing its pinned gather staging.
    pub(crate) fn select_window(&mut self, window: RowWindow) -> Result<()> {
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.end <= self.storage.rows
                && range.len() <= self.capacity, "Engram row window outside storage or unit");
        }
        self.ready = None;
        self.window = window;
        Ok(())
    }
    /// Release after the gate consumed these rows and joined the unit chain.
    pub(crate) fn release_scratch(&self) -> Result<()> { self.storage.guard.release() }
    #[cfg(test)]
    pub(crate) fn test_writer(&self, skip: bool) -> *mut std::ffi::c_void {
        SKIP_ACQUIRE.with(|value| value.set(skip));
        self.storage.stream.raw
    }
    unsafe fn acquire_scratch(&self, stream: *mut std::ffi::c_void) -> Result<()> {
        #[cfg(test)]
        if SKIP_ACQUIRE.with(|skip| skip.get()) { return Ok(()); }
        // SAFETY: caller supplies a live writer stream on this storage's device.
        unsafe { self.storage.guard.acquire(stream) }
    }
    fn gathered_buffers(&self, gathered: &EngramGatherView<'_>) -> (CuteafdDeviceBuffer, CuteafdDeviceBuffer) {
        let mut scales = self.storage.scales.buffer;
        if gathered.encoding == cuteafd_loader::EngramEncoding::Nvfp4 {
            // Compact FP4 rows leave enough room for their scales in the existing
            // weight allocation, so FP8 and FP4 need identical GPU budgets.
            scales = self.storage.weights.buffer;
            scales.ptr = unsafe { scales.ptr.cast::<u8>().add(gathered.weights.len()).cast() };
            scales.bytes = gathered.scales.len();
        }
        let mut weights = self.storage.weights.buffer;
        weights.bytes = gathered.weights.len();
        (weights, scales)
    }
    unsafe fn dequantize(&self, gathered: &EngramGatherView<'_>) -> Result<()> {
        let (weights, scales) = self.gathered_buffers(gathered);
        let rows = i32::try_from(gathered.rows * 24)?;
        match gathered.encoding {
            cuteafd_loader::EngramEncoding::Fp8 => unsafe {
                self.library.cuda_engram_dequant_bf16_async(weights, scales, self.storage.embeddings.buffer, rows, self.storage.stream.raw)
            },
            cuteafd_loader::EngramEncoding::Nvfp4 => unsafe {
                self.library.cuda_engram_nvfp4_dequant_bf16_async(weights, scales, gathered.global_scale,
                    self.storage.embeddings.buffer, rows, self.storage.stream.raw)
            },
        }
    }
    /// Consume a completed I/O-worker gather; this method never reads mapped tables.
    /// Output is contiguous BF16 [token,24,256] plus a byte-per-token text mask.
    pub fn upload(&mut self, gathered: &EngramGatherView<'_>) -> Result<EngramDeviceView> {
        self.ready = None;
        ensure!(
            gathered.rows > 0 && gathered.rows <= self.capacity && self.window.admits(gathered.rows) && gathered.layer_index < 2,
            "invalid gathered engram dimensions"
        );
        let hash_rows = gathered.rows * 24;
        ensure!(
            gathered.weights.len() == hash_rows * gathered.encoding.weight_bytes()
                && gathered.scales.len() == hash_rows * gathered.encoding.scale_bytes()
                && gathered.text_mask.len() == gathered.rows
                && gathered.text_mask.iter().all(|value| *value <= 1),
            "invalid gathered engram storage"
        );
        let (staging, staged) = self.staging.region(gathered.layer_index);
        let mut offset = 0;
        for bytes in [gathered.weights, gathered.scales, gathered.text_mask] {
            staged[offset..offset + bytes.len()].copy_from_slice(bytes);
            offset += bytes.len();
        }
        let launched = unsafe { self.enqueue_upload(gathered, staging) };
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        // SAFETY: dequantization and both output planes precede publication.
        unsafe { crate::shared::memory::chain::finish(self.library, self.storage.stream.raw)?; }
        self.ready = Some((gathered.rows, gathered.layer_index));
        self.view()
    }
    /// Pinned staging and this upload owner remain exclusive through completion;
    /// cancellation drains before either can be reused. No request owner is borrowed.
    pub async fn upload_cooperative(&mut self, gathered: &EngramGatherView<'_>) -> Result<EngramDeviceView> {
        self.ready = None;
        ensure!(
            gathered.rows > 0 && gathered.rows <= self.capacity && self.window.admits(gathered.rows) && gathered.layer_index < 2,
            "invalid gathered engram dimensions"
        );
        let hash_rows = gathered.rows * 24;
        ensure!(
            gathered.weights.len() == hash_rows * gathered.encoding.weight_bytes()
                && gathered.scales.len() == hash_rows * gathered.encoding.scale_bytes()
                && gathered.text_mask.len() == gathered.rows
                && gathered.text_mask.iter().all(|value| *value <= 1),
            "invalid gathered engram storage"
        );
        let mut offset = 0;
        let (staging, staged) = self.staging.region(gathered.layer_index);
        for bytes in [gathered.weights, gathered.scales, gathered.text_mask] {
            staged[offset..offset + bytes.len()].copy_from_slice(bytes);
            offset += bytes.len();
        }
        let deferred = crate::shared::memory::chain::deferred();
        let launched = unsafe { self.enqueue_upload(gathered, staging) };
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        if deferred {
            // The gate joins the chain head, which now follows this upload.
            unsafe { crate::shared::memory::chain::finish(self.library, self.storage.stream.raw)?; }
        } else {
            self.storage.stream.wait().await?;
        }
        self.ready = Some((gathered.rows, gathered.layer_index));
        self.view()
    }
    /// # Safety
    /// The per-slot layer staging and output window remain live until drain.
    unsafe fn enqueue_upload(&self, gathered: &EngramGatherView<'_>,
        staging: cuteafd_ffi::CuteafdHostBuffer) -> Result<()> {
        // SAFETY: staging belongs to this slot; all scratch use is on this stream.
        unsafe {
            crate::shared::memory::chain::join(self.library, self.storage.stream.raw)?;
            self.acquire_scratch(self.storage.stream.raw)?;
            let (weights, scales) = self.gathered_buffers(gathered);
            let mut offset = 0;
            for (destination, bytes) in [(weights, gathered.weights.len()),
                (scales, gathered.scales.len()),
                (self.storage.text_mask.buffer, gathered.text_mask.len())] {
                let mut host = staging;
                host.ptr = host.ptr.cast::<u8>().wrapping_add(offset).cast();
                host.bytes = bytes;
                self.library.copy_host_buffer_h2d_async(destination, host, bytes, self.storage.stream.raw)?;
                offset += bytes;
            }
            self.dequantize(gathered)
        }
    }
    #[cfg(test)]
    pub async fn check_cooperative(&mut self, gathered: &EngramGatherView<'_>) -> Result<()> {
        use std::{future::Future, task::Poll};
        let read = |lib: &NativeLibrary, view: EngramDeviceView| -> Result<Vec<Vec<u8>>> {
            [view.embeddings, view.text_mask].into_iter().map(|buffer| {
                let mut bytes = vec![0; buffer.bytes]; lib.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
            }).collect()
        };
        let lib = self.library;
        let expected = read(lib, self.upload(gathered)?)?;
        let cancelled = {
            let mut work = std::pin::pin!(self.upload_cooperative(gathered));
            std::future::poll_fn(|cx| Poll::Ready(work.as_mut().poll(cx).is_pending())).await
        };
        if cancelled { assert!(self.view().is_err()); }
        for _ in 0..2 { assert_eq!(read(lib, self.upload_cooperative(gathered).await?)?, expected); }
        eprintln!("PASS queued Engram upload {}: exact dequant/mask parity, pending cancellation={cancelled}, reuse", gathered.layer_index);
        Ok(())
    }
    pub fn view(&self) -> Result<EngramDeviceView> {
        let (rows, layer_index) = self.ready.context("engram device rows are not complete")?;
        let mut embeddings = self.storage.embeddings.buffer;
        embeddings.bytes = rows * 24 * 512;
        let mut text_mask = self.storage.text_mask.buffer;
        text_mask.bytes = rows;
        Ok(EngramDeviceView {
            embeddings,
            text_mask,
            rows,
            layer_index,
        })
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.storage.stream.raw) }
    }
}
impl Drop for EngramDeviceRows<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining native engram row production");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA native library and real packed PLE fixture with CPU reference"]
    fn nvfp4_ple_upload_matches_cpu_reference() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_PLE_FIXTURE")?);
        let fixtures: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for device in 0..2 {
            library.cuda_set_device(device)?;
            let mut owner = EngramDeviceRows::new(&library, 16, EngramDeviceRows::device_bytes(16)?)?;
            let pointers = (owner.storage.weights.buffer.ptr, owner.storage.scales.buffer.ptr, owner.storage.embeddings.buffer.ptr);
            for _ in 0..2 {
                for (index, fixture) in fixtures.as_array().context("fixture array")?.iter().enumerate() {
                    let layer = fixture["layer"].as_u64().context("layer")?;
                    let weights = std::fs::read(root.join(format!("layer{layer}-weights.bin")))?;
                    let scales = std::fs::read(root.join(format!("layer{layer}-scales.bin")))?;
                    let expected = std::fs::read(root.join(format!("layer{layer}-expected.bin")))?;
                    let mask: Vec<u8> = fixture["text_mask"].as_array().context("mask")?.iter()
                        .map(|v| v.as_u64().unwrap() as u8).collect();
                    // Reuse the owner at smaller and larger live sizes; FP4 scale
                    // placement depends on live rows, not allocation capacity.
                    for rows in [1, mask.len(), 3, mask.len()] {
                        let gathered = EngramGatherView {
                            weights: &weights[..rows*24*128], scales: &scales[..rows*24*16],
                            text_mask: &mask[..rows], rows, layer_index: index,
                            encoding: cuteafd_loader::EngramEncoding::Nvfp4,
                            global_scale: fixture["global_scale"].as_f64().context("global scale")? as f32,
                        };
                        for cooperative in [false, true] {
                            let view = if cooperative { runtime.block_on(owner.upload_cooperative(&gathered))? }
                                else { owner.upload(&gathered)? };
                            let mut actual = vec![0; view.embeddings.bytes];
                            library.copy_d2h(&mut actual, view.embeddings)?;
                            assert_eq!(actual, expected[..rows*24*512], "GPU {device}, layer {layer}, rows {rows}");
                            let mut actual_mask = vec![0; rows];
                            library.copy_d2h(&mut actual_mask, view.text_mask)?;
                            assert_eq!(actual_mask, mask[..rows]);
                        }
                        runtime.block_on(owner.check_cooperative(&gathered))?;
                        assert_eq!(pointers, (owner.storage.weights.buffer.ptr, owner.storage.scales.buffer.ptr, owner.storage.embeddings.buffer.ptr));
                    }
                    // Reuse the same allocation with the original FP8 format.
                    let weights = vec![0x38; 24*256];
                    let scales = vec![127; 24*8];
                    let gathered = EngramGatherView {
                        weights: &weights, scales: &scales, text_mask: &[1], rows: 1,
                        layer_index: index, encoding: cuteafd_loader::EngramEncoding::Fp8,
                        global_scale: 1.0,
                    };
                    let view = runtime.block_on(owner.upload_cooperative(&gathered))?;
                    let mut actual = vec![0; view.embeddings.bytes];
                    library.copy_d2h(&mut actual, view.embeddings)?;
                    assert!(actual.chunks_exact(2).all(|v| v == [0x80, 0x3f]));
                }
            }
        }
        Ok(())
    }
}
