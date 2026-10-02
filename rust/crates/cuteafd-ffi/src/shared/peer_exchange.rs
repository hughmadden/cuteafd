//! Two-GPU peer exchange and the P2P probe (`native/shared/cuda/peer_exchange.cu`).
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

/// One `cuteafd_p2p_probe` measurement (see `cuteafd_peer_exchange.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P2pTest {
    CopyEngine,
    SmPull,
    SmPush,
    HostBounce,
    CopyEnginePingPongEvents,
    SmPushPingPongEvents,
    FlagPingPong,
    FlagPingPongGraph,
    FlagExchange,
    FlagExchangeGraph,
    CopyEnginePingPongGraph,
}

impl P2pTest {
    pub const ALL: [P2pTest; 11] = [Self::CopyEngine, Self::SmPull, Self::SmPush, Self::HostBounce,
        Self::CopyEnginePingPongEvents, Self::SmPushPingPongEvents, Self::FlagPingPong, Self::FlagPingPongGraph,
        Self::FlagExchange, Self::FlagExchangeGraph, Self::CopyEnginePingPongGraph];

    pub fn label(self) -> &'static str {
        match self {
            Self::CopyEngine => "copy engine, one-way",
            Self::SmPull => "SM pull, one-way",
            Self::SmPush => "SM push, one-way",
            Self::HostBounce => "pinned-host bounce, one-way (host-timed)",
            Self::CopyEnginePingPongEvents => "copy engine + events, hop",
            Self::SmPushPingPongEvents => "SM push + events, hop",
            Self::FlagPingPong => "SM push + flag, hop",
            Self::FlagPingPongGraph => "SM push + flag, hop (graph)",
            Self::FlagExchange => "SM push + flag, two-way exchange",
            Self::FlagExchangeGraph => "SM push + flag, two-way exchange (graph)",
            Self::CopyEnginePingPongGraph => "copy engine + events, hop (graph)",
        }
    }
}

impl NativeLibrary {
    /// Microseconds per operation of `test` moving `bytes` between devices
    /// `a` and `b` (hops are half a round trip). `ingress` bits 0/1 stream
    /// host->device copies into `a`/`b` meanwhile. Allocates and synchronizes:
    /// a diagnostic, never on a serving path.
    #[allow(clippy::too_many_arguments)]
    pub fn p2p_probe(&self, a: i32, b: i32, bytes: usize, test: P2pTest, ingress: u32, iterations: u32,
        blocks: u32) -> Result<f64> {
        type F = unsafe extern "C" fn(i32, i32, u64, i32, u32, u32, u32, *mut f64) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_p2p_probe") }?;
        let mut us = 0.0;
        // SAFETY: the probe owns every allocation it touches; `us` outlives the call.
        let status = unsafe { f(a, b, bytes as u64, test as i32, ingress, iterations, blocks, &mut us) };
        ensure!(status == 0, "P2P probe {:?} of {bytes} bytes failed with CUDA error {status}", test);
        Ok(us)
    }

    /// Loads the exchange kernels on the current device (before any wait is
    /// queued there; see `cuteafd_peer_exchange_initialize`).
    pub fn peer_exchange_initialize(&self) -> Result<()> {
        type F = unsafe extern "C" fn() -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_peer_exchange_initialize") }?;
        let status = unsafe { f() };
        ensure!(status == 0, "loading the peer exchange kernels failed with CUDA error {status}");
        Ok(())
    }

    /// `out = bf16(a + b)` over `count` BF16 elements on `stream`.
    ///
    /// # Safety
    /// `a`, `b` and `out` are live device buffers of `count` elements on the
    /// stream's device, `out` disjoint from both, producers ordered before.
    pub unsafe fn peer_add_bf16(&self, a: *const c_void, b: *const c_void, out: *mut c_void, count: usize,
        stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, u64, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_peer_add_bf16_async") }?;
        let status = unsafe { f(a, b, out, count as u64, stream) };
        ensure!(status == 0, "BF16 add of {count} elements failed with CUDA error {status}");
        Ok(())
    }

    /// Pushes `bytes` from local `source` to peer `destination` and publishes
    /// the next sequence to the peer's `flag` (see `cuteafd_peer_push_signal`).
    ///
    /// # Safety
    /// `stream` belongs to the current (source) device with peer access to the
    /// destination's device; `destination` and `flag` are live peer memory,
    /// `source` and `send_state` (u32 [2]) live local memory, 16-byte aligned
    /// with `bytes % 16 == 0`; the source stays unchanged and the destination
    /// unread by others until the peer's matching wait.
    pub unsafe fn peer_push_signal(&self, destination: *mut c_void, source: *const c_void, bytes: usize,
        flag: *mut u32, send_state: *mut u32, blocks: u32, stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*mut c_void, *const c_void, u64, *mut u32, *mut u32, u32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_peer_push_signal") }?;
        let status = unsafe { f(destination, source, bytes as u64, flag, send_state, blocks, stream) };
        ensure!(status == 0, "peer push of {bytes} bytes failed with CUDA error {status}");
        Ok(())
    }

    /// Publishes a host mailbox on `stream`: writes `words` to `descriptor`,
    /// then the next sequence to `flag` (see `cuteafd_host_signal`).
    ///
    /// # Safety
    /// `flag` and `descriptor` are live pinned, device-mapped host memory,
    /// `send_state` (u32 [1]) live memory of the stream's device; the host
    /// side reads the mailbox only after it sees the sequence.
    pub unsafe fn host_signal(&self, flag: *mut u32, send_state: *mut u32, descriptor: *mut u32, words: [u32; 4],
        stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*mut u32, *mut u32, *mut u32, *const u32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_host_signal") }?;
        let status = unsafe { f(flag, send_state, descriptor, words.as_ptr(), stream) };
        ensure!(status == 0, "host mailbox signal failed with CUDA error {status}");
        Ok(())
    }

    /// Write-mode Spark completions on `stream` (see `cuteafd_spark_wait_written`).
    ///
    /// # Safety
    /// `flags` (`ranks` u64 words `stride_words` apart) is device memory the
    /// NICs write; `state` and `error` are live memory mapped on the stream's
    /// device; one wave was (or will be) posted per wait.
    pub unsafe fn spark_wait_written(&self, flags: *const u64, ranks: u32, stride_words: u32, state: *mut u32,
        error: *mut u32, stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*const u64, u32, u32, *mut u32, *mut u32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_spark_wait_written") }?;
        let status = unsafe { f(flags, ranks, stride_words, state, error, stream) };
        ensure!(status == 0, "Spark written-completion wait failed with CUDA error {status}");
        Ok(())
    }

    /// Spins on `stream` until `flag` reaches the next sequence.
    ///
    /// # Safety
    /// `flag` and `recv_state` (u32 [1]) are live memory of the stream's
    /// device; a peer push for every wait is (or will be) enqueued, or the
    /// stream never drains.
    pub unsafe fn peer_wait(&self, flag: *const u32, recv_state: *mut u32, stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*const u32, *mut u32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_peer_wait") }?;
        let status = unsafe { f(flag, recv_state, stream) };
        ensure!(status == 0, "peer wait failed with CUDA error {status}");
        Ok(())
    }
}
