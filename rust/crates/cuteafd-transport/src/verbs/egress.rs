//! Zero-copy request egress. A transport may own one pinned, device-mapped
//! host buffer that the engine writes a wave's expert input into (a D2H copy
//! on its own stream); requests whose hidden payload is a view of that buffer
//! ([`EgressBuffer::payload`]) are sent by every rank's session as a gathered
//! SEND: the header from the session's send-ring slot, the payload straight
//! from the shared buffer. Without it each session copied the whole payload
//! into its own send ring (six 26 MB copies per MiMo V2.6 Pro prefill wave).
use super::*;
use bytes::Bytes;

/// The pinned buffer, freed when the transport and every payload view are gone.
pub(crate) struct EgressBuffer {
    library: Arc<NativeLibrary>,
    host: CuteafdHostBuffer,
    /// Cloned into every payload view; the buffer may be rewritten only when
    /// no view is left ([`Self::writable`]).
    views: Arc<()>,
}

// The allocation is owned here and only read through views (or written by the
// engine's D2H copy once no view and no send is left); the raw pointer is the
// only non-Send part.
unsafe impl Send for EgressBuffer {}
unsafe impl Sync for EgressBuffer {}

impl EgressBuffer {
    pub(crate) fn new(bytes: usize) -> Result<Arc<Self>> {
        anyhow::ensure!(bytes > 0, "empty egress buffer");
        let library = load_verbs_host_native_library()?;
        let host = library.alloc_host_buffer(bytes)?;
        Ok(Arc::new(Self { library, host, views: Arc::new(()) }))
    }

    pub(crate) fn host(&self) -> CuteafdHostBuffer {
        self.host
    }

    /// Whether `payload` lies inside this buffer; its offset if so.
    pub(crate) fn offset_of(&self, payload: &[u8]) -> Option<usize> {
        let base = self.host.ptr as usize;
        let start = payload.as_ptr() as usize;
        (start >= base && start + payload.len() <= base + self.host.bytes).then(|| start - base)
    }

    /// No payload view is alive (requests holding one have been dropped).
    pub(crate) fn writable(&self) -> bool {
        Arc::strong_count(&self.views) == 1
    }

    /// The first `len` bytes as a request payload.
    pub(crate) fn payload(self: &Arc<Self>, len: usize) -> Result<Bytes> {
        anyhow::ensure!(len <= self.host.bytes, "egress payload of {len} bytes exceeds the {}-byte buffer",
            self.host.bytes);
        Ok(Bytes::from_owner(EgressView { buffer: Arc::clone(self), _view: Arc::clone(&self.views), len }))
    }
}

impl Drop for EgressBuffer {
    fn drop(&mut self) {
        let _ = self.library.free_host_buffer(&mut self.host);
    }
}

struct EgressView {
    buffer: Arc<EgressBuffer>,
    _view: Arc<()>,
    len: usize,
}

impl AsRef<[u8]> for EgressView {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: the allocation lives as long as `buffer`; the engine writes
        // it only while no view exists (`EgressBuffer::writable`), so the
        // bytes are not mutated while this shared view is alive.
        unsafe { std::slice::from_raw_parts(self.buffer.host.ptr.cast::<u8>(), self.len) }
    }
}
