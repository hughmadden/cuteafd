//! Pinned host memory from transparent hugepages, for large long-lived pools (the host
//! snapshot tier's slabs).
//!
//! `cudaHostAlloc` faults, zeroes and maps every 4 KiB page it pins. An anonymous mapping
//! advised `MADV_HUGEPAGE` faults one hugepage per extent instead, and `cudaHostRegister`
//! then pins it with the native allocator's flags (portable, mapped). On an RTX 5090 host
//! (THP `madvise`), 8 x 256 MiB pinned at 8.1 GB/s this way against 1.15 GB/s through
//! `cudaHostAlloc`. Registration binds the CUDA runtime already in the process
//! (`RTLD_NOLOAD`, as [`crate::CudaRuntime`] does), never a second one.
//!
//! A madvised fault that finds no free hugepage compacts memory synchronously under THP
//! defrag `always`, `madvise` or `defer+madvise`. With the page cache holding most of RAM
//! (203 GB cached, 5.4 GB free on a 251 GiB host) that ran at ~165 ms per hugepage and
//! pinned 7.1 GiB of a 32 GiB pool in ten minutes. Faults are therefore paced: each window
//! of [`FAULT_WINDOW`] bytes must fault at [`HUGEPAGE_FAULT_FLOOR_BYTES_PER_S`] or better,
//! and the first window that falls behind abandons the buffer with [`SlowHugepageFaults`],
//! so the caller pins the rest with base pages.
use crate::cuda_runtime::{RTLD_NOLOAD, RTLD_NOW, SONAMES};
use crate::{memory_ledger, CuteafdHostBuffer, CUTEAFD_HOST_BUFFER_FLAG_MAPPED, CUTEAFD_HOST_BUFFER_FLAG_PINNED};
use anyhow::{anyhow, bail, ensure, Context, Result};
use libloading::os::unix::Library;
use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `cudaHostRegisterPortable | cudaHostRegisterMapped` (`cuda_runtime_api.h`: 0x01 | 0x02),
/// the registration `cudaHostAllocPortable | cudaHostAllocMapped` gives the native path.
const REGISTER_PORTABLE_MAPPED: c_uint = 0x01 | 0x02;
const THP_ENABLED: &str = "/sys/kernel/mm/transparent_hugepage/enabled";
const THP_DEFRAG: &str = "/sys/kernel/mm/transparent_hugepage/defrag";
const THP_SIZE: &str = "/sys/kernel/mm/transparent_hugepage/hpage_pmd_size";
/// Hugepage faults pay only while they beat the base-page path that replaces them
/// (`cudaHostAlloc`: 0.75-1.15 GB/s on RTX 5090 hosts, against 8.1 GB/s for free hugepages).
pub const HUGEPAGE_FAULT_FLOOR_BYTES_PER_S: u64 = 1_000_000_000;
/// One pacing window: 8 ms of faults at 8 GB/s against a 67 ms budget at the floor, so
/// scheduling noise passes and one compaction stall (~165 ms) ends the window at once.
pub const FAULT_WINDOW: usize = 64 << 20;

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

/// This host's THP defrag policy (`always`, `defer`, `defer+madvise`, `madvise` or `never`):
/// whether a madvised fault without a free hugepage compacts memory before it returns.
pub fn transparent_hugepage_defrag() -> Option<String> {
    selected_mode(&std::fs::read_to_string(THP_DEFRAG).ok()?).map(str::to_owned)
}

/// A hugepage-backed buffer abandoned because its faults fell below
/// [`HUGEPAGE_FAULT_FLOOR_BYTES_PER_S`]: the kernel is compacting memory to find hugepages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlowHugepageFaults {
    /// Start of the window that fell behind.
    pub offset: usize,
    /// Bytes that window faulted before its budget ran out.
    pub faulted: usize,
    pub elapsed: Duration,
    pub len: usize,
    pub floor: u64,
}

impl std::fmt::Display for SlowHugepageFaults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rate = self.faulted as f64 / self.elapsed.as_secs_f64().max(1e-9) / 1e9;
        write!(f, "hugepage faults ran at {rate:.3} GB/s ({} bytes in {:?} from offset {} of a {}-byte buffer), \
            below the {:.2} GB/s floor", self.faulted, self.elapsed, self.offset, self.len, self.floor as f64 / 1e9)
    }
}

impl std::error::Error for SlowHugepageFaults {}

/// The time `bytes` of faults may take at `floor` bytes per second.
fn fault_budget(bytes: usize, floor: u64) -> Duration {
    let nanos = (bytes as u128).saturating_mul(1_000_000_000) / u128::from(floor.max(1));
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}

/// Fault `len` bytes one `huge`-byte extent at a time (`touch(offset)`), in windows of
/// `window` bytes that must each finish within their [`fault_budget`] by `now`. The first
/// extent that leaves its window over budget stops the loop there.
fn fault_paced(len: usize, huge: usize, window: usize, floor: u64, mut touch: impl FnMut(usize),
    mut now: impl FnMut() -> Duration) -> std::result::Result<(), SlowHugepageFaults> {
    let window = window.max(huge);
    let mut start = 0;
    while start < len {
        let end = start.saturating_add(window).min(len);
        let (budget, began) = (fault_budget(end - start, floor), now());
        for offset in (start..end).step_by(huge) {
            touch(offset);
            let elapsed = now().saturating_sub(began);
            if elapsed > budget {
                return Err(SlowHugepageFaults { offset: start, faulted: (offset + huge).min(end) - start,
                    elapsed, len, floor });
            }
        }
        start = end;
    }
    Ok(())
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

    /// Map, fault and pin at least `bytes` (whole hugepages, hugepage-aligned). Fails with
    /// [`SlowHugepageFaults`], leaving nothing mapped, when the faults fall behind the floor.
    pub fn alloc(&self, bytes: usize) -> Result<CuteafdHostBuffer> {
        ensure!(bytes > 0, "pinned host buffer size is zero");
        let len = bytes.checked_next_multiple_of(self.huge_page).context("pinned host buffer size overflow")?;
        let ptr = map_window(len, self.huge_page)?;
        let unmap = |error: anyhow::Error| {
            // SAFETY: the window is still mapped, unregistered and unused.
            unsafe { libc::munmap(ptr, len) };
            error
        };
        // Hugepages for the faults below; and no copy into a forked child, which would
        // otherwise duplicate the pinned pages eagerly.
        for (advice, name) in [(libc::MADV_HUGEPAGE, "MADV_HUGEPAGE"), (libc::MADV_DONTFORK, "MADV_DONTFORK")] {
            // SAFETY: the window is a live private mapping of `len` bytes.
            if unsafe { libc::madvise(ptr, len, advice) } != 0 {
                return Err(unmap(anyhow!("madvise({name}) failed: {}", std::io::Error::last_os_error())));
            }
        }
        let started = Instant::now();
        let paced = fault_paced(len, self.huge_page, FAULT_WINDOW, HUGEPAGE_FAULT_FLOOR_BYTES_PER_S,
            // SAFETY: `fault_paced` passes offsets below `len`, inside the writable window;
            // one write faults (and zeroes) one hugepage.
            |offset| unsafe { ptr.cast::<u8>().add(offset).write_volatile(0) },
            || started.elapsed());
        if let Err(slow) = paced {
            return Err(unmap(slow.into()));
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
        assert_eq!(selected_mode("always defer [defer+madvise] madvise never"), Some("defer+madvise"));
        assert_eq!(selected_mode("garbage"), None);
    }

    const HUGE: usize = 2 << 20;
    const MS: Duration = Duration::from_millis(1);

    /// Runs `fault_paced` with the production window and floor on a clock that advances
    /// `cost(extent)` per fault; returns the result and every touched offset.
    fn paced(len: usize, mut cost: impl FnMut(usize) -> Duration)
        -> (std::result::Result<(), SlowHugepageFaults>, Vec<usize>) {
        let clock = std::cell::Cell::new(Duration::ZERO);
        let mut touched = Vec::new();
        let result = fault_paced(len, HUGE, FAULT_WINDOW, HUGEPAGE_FAULT_FLOOR_BYTES_PER_S, |offset| {
            touched.push(offset);
            clock.set(clock.get() + cost(offset / HUGE));
        }, || clock.get());
        (result, touched)
    }

    #[test]
    fn a_window_at_the_floor_takes_its_byte_time() {
        // 64 MiB at 1 GB/s: 67.1 ms; a 2 MiB hugepage: 2.1 ms.
        assert_eq!(fault_budget(FAULT_WINDOW, HUGEPAGE_FAULT_FLOOR_BYTES_PER_S), Duration::from_nanos(67_108_864));
        assert_eq!(fault_budget(HUGE, HUGEPAGE_FAULT_FLOOR_BYTES_PER_S), Duration::from_nanos(2_097_152));
        assert_eq!(fault_budget(usize::MAX, 1), Duration::from_nanos(u64::MAX));
        assert_eq!(fault_budget(HUGE, 0), Duration::from_nanos(2_097_152_000_000_000), "a zero floor is one byte per second");
    }

    #[test]
    fn free_hugepages_fault_every_extent_once_in_order() {
        // 256 MiB at the measured 8 GB/s (0.26 ms per hugepage), with a 50 ms scheduling hiccup.
        let (result, touched) = paced(256 << 20, |extent| if extent == 70 { 50 * MS } else { MS / 4 });
        assert_eq!(result, Ok(()));
        assert_eq!(touched, (0..256 << 20).step_by(HUGE).collect::<Vec<_>>());
    }

    #[test]
    fn one_compaction_stall_ends_the_buffer_at_that_extent() {
        // The stalled host: fast faults, then ~165 ms per hugepage once free hugepages run out.
        let (result, touched) = paced(256 << 20, |extent| if extent >= 40 { 165 * MS } else { MS / 4 });
        let slow = result.unwrap_err();
        assert_eq!((slow.offset, slow.faulted, slow.len), (FAULT_WINDOW, 9 * HUGE, 256 << 20));
        assert_eq!(slow.elapsed, 8 * MS / 4 + 165 * MS);
        assert_eq!(touched.len(), 41, "no fault after the first stall");
        assert!(slow.to_string().contains("below the 1.00 GB/s floor"), "{slow}");
    }

    #[test]
    fn a_fast_window_lends_no_budget_to_a_slow_one() {
        // Window 0 uses a tenth of its budget; window 1 overruns its own by 2%.
        let budget = fault_budget(FAULT_WINDOW, HUGEPAGE_FAULT_FLOOR_BYTES_PER_S);
        let per_extent = |fraction: f64| budget.mul_f64(fraction / (FAULT_WINDOW / HUGE) as f64);
        let (result, touched) = paced(2 * FAULT_WINDOW, |extent| per_extent(if extent < 32 { 0.1 } else { 1.02 }));
        assert_eq!(result.unwrap_err().offset, FAULT_WINDOW);
        assert_eq!(touched.len(), 64, "the overrun shows only at the window's last extent");
        // At 95% of each window's budget the whole buffer faults.
        assert_eq!(paced(4 * FAULT_WINDOW, |_| per_extent(0.95)).0, Ok(()));
    }

    #[test]
    fn a_short_last_window_has_a_proportional_budget() {
        // 96 MiB: a full window, then 32 MiB allowed half of a window's 67 ms.
        let full = fault_budget(FAULT_WINDOW, HUGEPAGE_FAULT_FLOOR_BYTES_PER_S);
        let tail = |share: f64| move |extent: usize| if extent < 32 { MS / 4 } else { full.mul_f64(share / 16.0) };
        // 90% of a full window's budget is 180% of the short window's: it stops at its ninth fault.
        let (result, touched) = paced(FAULT_WINDOW * 3 / 2, tail(0.9));
        let slow = result.unwrap_err();
        assert_eq!((slow.offset, slow.faulted, touched.len()), (FAULT_WINDOW, 9 * HUGE, 41));
        assert_eq!(paced(FAULT_WINDOW * 3 / 2, tail(0.45)).0, Ok(()));
    }

    #[test]
    fn paced_faults_write_every_extent_of_a_real_window() -> Result<()> {
        let len = 8 * HUGE;
        let ptr = map_window(len, HUGE)?;
        let started = Instant::now();
        // One byte per second: no host is that slow, so every extent faults.
        let result = fault_paced(len, HUGE, FAULT_WINDOW, 1,
            // SAFETY: offsets stay below `len` inside the live private mapping.
            |offset| unsafe { ptr.cast::<u8>().add(offset).write_volatile(0xa5) }, || started.elapsed());
        // SAFETY: the window is a live private mapping of `len` bytes until the munmap.
        let written = (0..len).step_by(HUGE).all(|offset| unsafe { *ptr.cast::<u8>().add(offset) } == 0xa5);
        // SAFETY: nothing refers to the mapping any more.
        unsafe { assert_eq!(libc::munmap(ptr, len), 0) };
        assert_eq!(result, Ok(()));
        assert!(written);
        Ok(())
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
