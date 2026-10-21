//! Registered shared mappings for push delivery. No device bytes live here.
use crate::{CuteafdHostBuffer, NativeLibrary};
use anyhow::{ensure, Context, Result};
use libloading::Symbol;
use std::{
    ffi::{c_void, CString},
    os::fd::{AsFd, BorrowedFd, FromRawFd, OwnedFd},
};

pub const DEFAULT_SLOT_BYTES: usize = 1024 * 1024;
pub const DEFAULT_SLOTS: usize = 128;

pub struct RegisteredRing<'a> {
    library: &'a NativeLibrary,
    fd: OwnedFd,
    ptr: *mut c_void,
    bytes: usize,
    slots: usize,
    slot_bytes: usize,
    device: i32,
    registered: bool,
}
impl<'a> RegisteredRing<'a> {
    pub fn new(library: &'a NativeLibrary, slots: usize, slot_bytes: usize) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&slots)
                && slot_bytes >= 4096
                && slot_bytes <= DEFAULT_SLOT_BYTES
                && slot_bytes % 4096 == 0,
            "invalid RDMA push ring geometry"
        );
        let bytes = slots
            .checked_mul(slot_bytes)
            .context("push ring size overflow")?;
        let name = CString::new("weight-push")?;
        // SAFETY: valid NUL-terminated name; ownership of the new fd is transferred once.
        let fd = unsafe {
            libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
        };
        ensure!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
        // SAFETY: successful memfd_create returned a newly owned descriptor.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        use std::os::fd::AsRawFd;
        // SAFETY: owned fd, bounded nonzero extent; sealing prevents remote resize.
        ensure!(
            unsafe { libc::ftruncate(fd.as_raw_fd(), bytes as libc::off_t) } == 0,
            "ring ftruncate failed"
        );
        // SAFETY: sealing resize does not prevent shared payload writes.
        ensure!(
            unsafe {
                libc::fcntl(
                    fd.as_raw_fd(),
                    libc::F_ADD_SEALS,
                    libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL,
                )
            } == 0,
            "ring sealing failed"
        );
        // SAFETY: file has the requested extent and remains owned through unmapping.
        let device = library.cuda_get_device()?;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        ensure!(ptr != libc::MAP_FAILED, "ring mmap failed");
        let mut ring = Self {
            library,
            fd,
            ptr,
            bytes,
            slots,
            slot_bytes,
            device,
            registered: false,
        };
        type Register = unsafe extern "C" fn(*mut c_void, usize) -> i32;
        // SAFETY: native registration accepts this live shared mapping.
        let status = unsafe {
            let register: Symbol<Register> = library.lib.get(b"cuteafd_register_host_mapping")?;
            register(ptr, bytes)
        };
        library.status_to_result("cuteafd_register_host_mapping", status)?;
        ring.registered = true;
        Ok(ring)
    }
    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
    pub fn slots(&self) -> usize {
        self.slots
    }
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    /// The caller holds the server's Filled credit until all references/copies end.
    pub fn slot(&self, index: u32, len: u32) -> Result<&[u8]> {
        ensure!(
            (index as usize) < self.slots && len as usize == self.slot_bytes,
            "invalid Filled slot extent"
        );
        // SAFETY: checked slot is inside the mapped extent; Filled transfers write ownership.
        Ok(unsafe {
            std::slice::from_raw_parts(
                self.ptr.cast::<u8>().add(index as usize * self.slot_bytes),
                len as usize,
            )
        })
    }
    pub fn payload(&self, index: u32, offset: usize, bytes: usize) -> Result<CuteafdHostBuffer> {
        ensure!(
            (index as usize) < self.slots
                && offset
                    .checked_add(bytes)
                    .is_some_and(|n| n <= self.slot_bytes),
            "ring payload outside slot"
        );
        // SAFETY: checked range is inside this registered mapping retained by the consumer.
        Ok(CuteafdHostBuffer {
            ptr: unsafe {
                self.ptr
                    .cast::<u8>()
                    .add(index as usize * self.slot_bytes + offset)
                    .cast()
            },
            bytes,
            flags: 1,
        })
    }
}
impl Drop for RegisteredRing<'_> {
    fn drop(&mut self) {
        if self.library.is_quarantined_after_failed_drain() {
            return;
        }
        if self.registered {
            let result = (|| -> Result<()> {
                let previous = self.library.cuda_get_device()?;
                self.library.cuda_set_device(self.device)?;
                type Unregister = unsafe extern "C" fn(*mut c_void) -> i32;
                // SAFETY: all queued copies must have drained before the ring is dropped.
                let status = unsafe {
                    let unregister: Symbol<Unregister> =
                        self.library.lib.get(b"cuteafd_unregister_host_mapping")?;
                    unregister(self.ptr)
                };
                let result = self
                    .library
                    .status_to_result("cuteafd_unregister_host_mapping", status);
                self.library.cuda_set_device(previous)?;
                result
            })();
            if let Err(error) = result {
                tracing::error!(%error, "retaining registered push mapping");
                return;
            }
        }
        // SAFETY: unique mapping owner, no registration or queued accesses remain.
        unsafe {
            libc::munmap(self.ptr, self.bytes);
        }
    }
}
