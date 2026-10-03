//! Page-cache hygiene for loaders that read a checkpoint once.
//!
//! On GB10 the page cache and CUDA allocations share the same 121 GiB, and
//! CUDA allocations do not reclaim cached pages, so a worker that has copied
//! its weights into device memory drops the source pages it no longer needs.

use std::os::fd::AsRawFd;
use std::path::Path;

/// Advises the kernel to drop cached pages of every `*.safetensors` file under
/// `snapshot` (resolving symlinks). Returns the bytes of the files advised.
/// Failures are skipped: this is a hint, never a correctness requirement.
pub fn drop_snapshot_pages(snapshot: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(snapshot) else { return 0 };
    let mut advised = 0u64;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "safetensors") {
            continue;
        }
        let Ok(file) = std::fs::File::open(&path) else { continue };
        let Ok(length) = file.metadata().map(|m| m.len()) else { continue };
        // SAFETY: the descriptor is open for the call's duration; the advice
        // only affects caching, never the file's contents.
        let status = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        if status == 0 {
            advised += length;
        }
    }
    advised
}
