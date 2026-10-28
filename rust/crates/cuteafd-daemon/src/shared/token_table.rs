//! Common token-table H2D issue contract. Call only after host readiness and
//! destination reuse gates are satisfied; never park these copies behind waits.
use anyhow::{ensure, Result};
use super::memory::{HostAllocation, device::{Device, Event, Stream}};
use cuteafd_loader::shared::token_table::{UploadCursor, UploadShape, METADATA_SM_COPY_BYTES};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use std::ffi::c_void;

pub(crate) struct CopyPlane { pub destination: CuteafdDeviceBuffer, pub offset: usize, pub bytes: usize }

/// # Safety
/// The source lease and every destination survive the upload fence; stream is
/// on their device, and the caller has observed its destination reuse gate.
pub(crate) unsafe fn enqueue(library: &NativeLibrary, source: CuteafdHostBuffer,
    planes: &[CopyPlane], stream: *mut c_void) -> Result<()> {
    for plane in planes {
        ensure!(plane.offset.checked_add(plane.bytes).is_some_and(|n| n <= source.bytes)
            && plane.bytes <= plane.destination.bytes, "token-table copy plane exceeds storage");
    }
    for plane in planes {
        if plane.bytes == 0 { continue; }
        let host = CuteafdHostBuffer { ptr: source.ptr.cast::<u8>().wrapping_add(plane.offset).cast(), bytes: plane.bytes, ..source };
        // SAFETY: validated disjoint source range and caller-owned destination;
        // the caller records a source fence even after a partial enqueue error.
        unsafe { library.copy_host_buffer_h2d_async(plane.destination, host, plane.bytes, stream)?; }
    }
    Ok(())
}

/// One immutable source range per queued read, over one admitted pinned owner.
/// Each arena serves exactly one consumer stream; only completed sources wrap.
pub(crate) struct RetainedUploads<'a> {
    storage: HostAllocation<'a>,
    alias: CuteafdDeviceBuffer,
    cursor: UploadCursor,
    events: Vec<Event<'a>>,
    upload: Stream<'a>,
    previous_reader: Event<'a>,
    stream: *mut c_void,
    failed: bool,
    exhaustion_waits: u64,
}
impl<'a> RetainedUploads<'a> {
    pub fn new(device: Device<'a>, shape: UploadShape) -> Result<Self> {
        let cursor = UploadCursor::new(shape)?;
        let storage = HostAllocation::new(device.library, shape.capacity_bytes()?)?;
        let alias = device.library.cuda_host_buffer_device_alias(storage.buffer)?;
        let events = (0..shape.ranges).map(|_| Event::new(device)).collect::<Result<Vec<_>>>()?;
        Ok(Self { storage, alias, cursor, events, upload: Stream::new(device)?, previous_reader: Event::new(device)?,
            stream: std::ptr::null_mut(), failed: false, exhaustion_waits: 0 })
    }

    /// # Safety
    /// Destination and stream remain live through the queued read and every
    /// destination consumer/rewrite runs in this stream's order. Large uploads
    /// observe the previous reader before issuing copy-engine work.
    pub unsafe fn upload(&mut self, destination: CuteafdDeviceBuffer, bytes: &[u8], stream: *mut c_void) -> Result<()> {
        if bytes.is_empty() { return Ok(()); }
        ensure!(!self.failed && !stream.is_null() && (self.stream.is_null() || self.stream == stream),
            "retained upload source is failed or changed consumer stream");
        ensure!(bytes.len() <= destination.bytes && destination.device_id == self.upload.device.id,
            "retained metadata exceeds destination admission");
        self.stream = stream;
        let library = self.storage.library;
        let large = bytes.len() > METADATA_SM_COPY_BYTES;
        let copy_stream = if large { self.upload.raw } else { stream };
        if large {
            // SAFETY: previous destination readers all precede this record.
            // Observe completion before issuing copy-engine work, never park
            // the copy behind an unmet device wait.
            unsafe { library.cuda_event_record(self.previous_reader.raw, stream)?; }
            self.upload.device.run(|| unsafe { library.cuda_event_synchronize(self.previous_reader.raw) })?;
        }
        let (fence, range) = loop {
            if let Some(reserved) = self.cursor.reserve(bytes.len())? { break reserved; }
            let oldest = self.cursor.oldest().expect("live source prevents arena reuse");
            let event = &self.events[oldest];
            // SAFETY: every live source owns a successfully recorded event.
            let ready = event.device.run(|| unsafe { library.cuda_event_query(event.raw) })?;
            if !ready {
                self.exhaustion_waits += 1;
                anyhow::bail!("retained metadata arena exceeded admitted source depth: exhaustion {}, upload {} bytes",
                    self.exhaustion_waits, bytes.len());
            }
            self.cursor.retire(oldest)?;
        };
        // SAFETY: UploadCursor proved this sub-range disjoint from every live
        // GPU read. Do not form a mutable slice spanning the whole allocation.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.storage.buffer.ptr.cast::<u8>().add(range.start), bytes.len()); }
        let event = &self.events[fence];
        // SAFETY: admitted destination/source, immutable until this event;
        // the source and every event survive partial enqueue/record failures.
        let copied = if large {
            let source = CuteafdHostBuffer { ptr: self.storage.buffer.ptr.cast::<u8>().wrapping_add(range.start).cast(),
                bytes: bytes.len(), ..self.storage.buffer };
            unsafe { enqueue(library, source, &[CopyPlane { destination, offset: 0, bytes: bytes.len() }], copy_stream) }
        } else {
            unsafe { library.sm_copy_regions(&[(destination.ptr,
                self.alias.ptr.cast::<u8>().wrapping_add(range.start).cast_const().cast(), bytes.len() as u64)], copy_stream) }
        };
        let recorded = event.device.run(|| unsafe { library.cuda_event_record(event.raw, copy_stream) });
        if recorded.is_err() {
            self.failed = true;
            // SAFETY: an unproved record must drain or quarantine the source.
            if unsafe { library.cuda_stream_synchronize(copy_stream) }.is_err() {
                library.quarantine_module_after_failed_drain();
            }
        }
        copied?;
        recorded?;
        if large {
            // SAFETY: the independent upload has been issued; its fence orders
            // later consumers without a copy-engine job behind their wait.
            unsafe { library.cuda_stream_wait_event(stream, event.raw)?; }
        }
        Ok(())
    }
}
impl Drop for RetainedUploads<'_> {
    fn drop(&mut self) {
        while let Some(fence) = self.cursor.oldest() {
            let event = &self.events[fence];
            // SAFETY: source ownership survives until its reader completion.
            if self.failed || event.device.run(|| unsafe {
                self.storage.library.cuda_event_synchronize(event.raw)
            }).is_err() {
                self.storage.library.quarantine_module_after_failed_drain();
                break;
            }
            let _ = self.cursor.retire(fence);
        }
    }
}
