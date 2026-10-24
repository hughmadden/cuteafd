//! Pinned host memory from transparent hugepages, for large long-lived pools (the host
//! snapshot tier's slabs).
//!
//! `cudaHostAlloc` faults, zeroes and maps every 4 KiB page it pins. An anonymous mapping
//! advised `MADV_HUGEPAGE` faults one hugepage per extent instead, and `cudaHostRegister`
//! then pins it with the native allocator's flags (portable, mapped). On an RTX 5090 host
//! (THP `madvise`), 8 x 256 MiB pinned at 8.1 GB/s this way against 1.15 GB/s through
//! `cudaHostAlloc`. Registration binds the CUDA runtime already in the process
//! (`RTLD_NOLOAD`, as [`crate::CudaRuntime`] does), never a second one.
use crate::cuda_runtime::{RTLD_NOLOAD, RTLD_NOW, SONAMES};
use crate::{memory_ledger, CuteafdHostBuffer, CUTEAFD_HOST_BUFFER_FLAG_MAPPED, CUTEAFD_HOST_BUFFER_FLAG_PINNED};
use anyhow::{anyhow, bail, ensure, Context, Result};
use libloading::os::unix::Library;
use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};
use std::sync::Arc;

/// `cudaHostRegisterPortable | cudaHostRegisterMapped` (`cuda_runtime_api.h`: 0x01 | 0x02),
/// the registration `cudaHostAllocPortable | cudaHostAllocMapped` gives the native path.
const REGISTER_PORTABLE_MAPPED: c_uint = 0x01 | 0x02;
const THP_ENABLED: &str = "/sys/kernel/mm/transparent_hugepage/enabled";
const THP_SIZE: &str = "/sys/kernel/mm/transparent_hugepage/hpage_pmd_size";

type HostRegisterFn = unsafe extern "C" fn(*mut c_void, usize, c_uint) -> c_int;
type HostUnregisterFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type GetErrorStringFn = unsafe extern "C" fn(c_int) -> *const c_char;

/// The bracketed choice of a sysfs THP mode line (`always [madvise] never` is `madvise`).
fn selected_mode(text: &str) -> Option<&str> {
    let start = text.find('[')? + 1;
    Some(&text[start..start + text[start..].find(']')?])
}

/// This host's transparent hugepage mode (`always`, `madvise` or `never`); `None` without THP.
pub fn transparent_hugepage_mode() -> Option<String> {
    selected_mode(&std::fs::read_to_string(THP_ENABLED).ok()?).map(str::to_owned)
}

/// A hugepage-aligned private anonymous mapping of `len` bytes (a multiple of `huge`).
fn map_window(len: usize, huge: usize) -> Result<*mut c_void> {
    let span = len.checked_add(huge).context("pinned host buffer size overflow")?;
    // SAFETY: a new private anonymous mapping that nothing else refers to.
    let raw = unsafe { libc::mmap(std::ptr::null_mut(), span, libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
    if raw == libc::MAP_FAILED { bail!("mmap of {span} bytes failed: {}", std::io::Error::last_os_error()); }
    let start = (raw as usize).next_multiple_of(huge);
    let head = start - raw as usize;
    // SAFETY: both trims lie inside the fresh mapping and outside the kept window.
    unsafe {
        if head > 0 { libc::munmap(raw, head); }
        if span - head > len { libc::munmap((start + len) as *mut c_void, span - head - len); }
    }
    Ok(start as *mut c_void)
}

/// Host registration through the process's CUDA runtime, for hugepage-backed buffers.
#[derive(Clone)]
pub struct HugePagePinning {
    _library: Arc<Library>, // keepalive for the bound entry points
    register: HostRegisterFn,
    unregister: HostUnregisterFn,
    error_string: GetErrorStringFn,
    huge_page: usize,
}

// Plain function pointers into a library the process already loaded; no device state.
unsafe impl Send for HugePagePinning {}
unsafe impl Sync for HugePagePinning {}

impl HugePagePinning {
    /// `None` unless madvised memory gets transparent hugepages (`always` or `madvise`) and
    /// the process already has a CUDA runtime.
    pub fn load() -> Option<Self> {
        if !matches!(transparent_hugepage_mode()?.as_str(), "always" | "madvise") { return None; }
        let huge_page = std::fs::read_to_string(THP_SIZE).ok().and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|n| n.is_power_of_two() && *n >= 4096).unwrap_or(2 << 20);
        SONAMES.iter().find_map(|soname| {
            // SAFETY: RTLD_NOLOAD only finds a runtime already mapped into the process.
            let library = unsafe { Library::open(Some(soname), RTLD_NOW | RTLD_NOLOAD) }.ok().map(Arc::new)?;
            // SAFETY: these entry points have these signatures in every supported runtime.
            let (register, unregister, error_string) = unsafe { (
                *library.get::<HostRegisterFn>(b"cudaHostRegister").ok()?,
                *library.get::<HostUnregisterFn>(b"cudaHostUnregister").ok()?,
                *library.get::<GetErrorStringFn>(b"cudaGetErrorString").ok()?) };
            Some(Self { _library: library, register, unregister, error_string, huge_page })
        })
    }

    /// Map, fault and pin at least `bytes` (whole hugepages, hugepage-aligned).
    pub fn alloc(&self, bytes: usize) -> Result<CuteafdHostBuffer> {
        ensure!(bytes > 0, "pinned host buffer size is zero");
        let len = bytes.checked_next_multiple_of(self.huge_page).context("pinned host buffer size overflow")?;
        let ptr = map_window(len, self.huge_page)?;
        let unmap = |error: anyhow::Error| {
            // SAFETY: the window is still mapped, unregistered and unused.
            unsafe { libc::munmap(ptr, len) };
            error
        };
        // SAFETY: the window is a live private mapping of `len` bytes.
        if unsafe { libc::madvise(ptr, len, libc::MADV_HUGEPAGE) } != 0 {
            return Err(unmap(anyhow!("madvise(MADV_HUGEPAGE) failed: {}", std::io::Error::last_os_error())));
        }
        for offset in (0..len).step_by(self.huge_page) {
            // SAFETY: inside the writable window; one write faults (and zeroes) one hugepage.
            unsafe { ptr.cast::<u8>().add(offset).write_volatile(0) };
        }
        // SAFETY: the window stays mapped until `free` unregisters it.
        let status = unsafe { (self.register)(ptr, len, REGISTER_PORTABLE_MAPPED) };
        if status != 0 {
            return Err(unmap(anyhow!("cudaHostRegister of {len} bytes failed with status {status}: {}",
                self.error(status))));
        }
        memory_ledger::record_alloc(memory_ledger::Space::Pinned, -1, ptr as usize, len);
        Ok(CuteafdHostBuffer { ptr, bytes: len, flags: CUTEAFD_HOST_BUFFER_FLAG_PINNED | CUTEAFD_HOST_BUFFER_FLAG_MAPPED })
    }

    /// Unpin and unmap a buffer from [`HugePagePinning::alloc`].
    ///
    /// # Safety
    /// `buffer` came from `alloc` and no queued copy still reads or writes it.
    pub unsafe fn free(&self, buffer: &mut CuteafdHostBuffer) -> Result<()> {
        if buffer.ptr.is_null() { return Ok(()); }
        // SAFETY: the caller guarantees this is a live registration with no queued users.
        let status = unsafe { (self.unregister)(buffer.ptr) };
        ensure!(status == 0, "cudaHostUnregister failed with status {status}: {}", self.error(status));
        memory_ledger::record_free(buffer.ptr as usize);
        // SAFETY: the mapping `alloc` kept, now unregistered.
        ensure!(unsafe { libc::munmap(buffer.ptr, buffer.bytes) } == 0, "munmap failed: {}",
            std::io::Error::last_os_error());
        *buffer = CuteafdHostBuffer::default();
        Ok(())
    }

    fn error(&self, status: c_int) -> String {
        // SAFETY: cudaGetErrorString returns a static string or null.
        let raw = unsafe { (self.error_string)(status) };
        if raw.is_null() { return String::new(); }
        // SAFETY: a non-null result is a NUL-terminated static string.
        unsafe { std::ffi::CStr::from_ptr(raw) }.to_string_lossy().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thp_mode_reads_the_bracketed_choice() {
        assert_eq!(selected_mode("always [madvise] never\n"), Some("madvise"));
        assert_eq!(selected_mode("[always] madvise never"), Some("always"));
        assert_eq!(selected_mode("always madvise [never]"), Some("never"));
        assert_eq!(selected_mode("garbage"), None);
    }

    #[test]
    fn hugepage_windows_are_aligned_whole_and_writable() -> Result<()> {
        let huge = 2 << 20;
        for len in [huge, 3 * huge, 128 * huge] {
            let ptr = map_window(len, huge)?;
            assert_eq!(ptr as usize % huge, 0);
            // SAFETY: the window is a live private mapping of `len` bytes.
            unsafe {
                if transparent_hugepage_mode().is_some() { assert_eq!(libc::madvise(ptr, len, libc::MADV_HUGEPAGE), 0); }
                std::ptr::write_bytes(ptr.cast::<u8>(), 0xa5, len);
                assert_eq!(*ptr.cast::<u8>().add(len - 1), 0xa5);
                assert_eq!(libc::munmap(ptr, len), 0);
            }
        }
        Ok(())
    }
}
