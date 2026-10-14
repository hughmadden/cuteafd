//! How a TP2 layer's two routed halves meet.
//!
//! Fused combine numerics (PLAN section 2): each rank forms
//! `p_r = dtype(FP32(routed_r) + FP32(shared_r))`, the FFN exchange carries
//! `p_r` to the other GPU, and both GPUs compute `BF16(FP32(p_0) + FP32(p_1))`
//! in rank order, so both hold the same bits. Spark reduction instead sums
//! routed planes first and adds shared after; this is a numerics change with
//! its own fidelity gate per format.
use super::{PartialDtype, PartialLayout};
use crate::shared::memory::device::{Allocation, Device, Event, PeerTransfer, Stream};
use anyhow::{ensure, Result};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, RtxPartialDtype, RtxTp2Combine, V41Tp2ExpertReducer,
};
use std::ffi::c_void;

/// The dtype of the FFN exchange payload under [`Combine::FusedAllReduce`].
pub(crate) type ExchangeDtype = RtxPartialDtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Combine {
    /// HeadSplit / `Whole{ffn: Split}` layers: each rank adds its routed
    /// partial to its shared/dense half, then the existing FFN slot exchange
    /// sums the two in rank order on both GPUs.
    FusedAllReduce { exchange: ExchangeDtype },
    /// `Whole{ffn: Owner}` layers (V4.1): inputs and routes broadcast, routed
    /// then shared reduced in rank order onto the owner, added in BF16.
    OwnerReduce { owner: u8 },
}

impl Combine {
    /// `CUTEAFD_TP2_EXCHANGE=bf16|f32` overrides the fused exchange dtype.
    pub(crate) fn fused_from_env(default: ExchangeDtype) -> Result<Self> {
        let exchange = match std::env::var("CUTEAFD_TP2_EXCHANGE").ok().as_deref() {
            None | Some("") => default,
            Some("bf16") => ExchangeDtype::Bf16,
            Some("f32") => ExchangeDtype::F32,
            Some(other) => anyhow::bail!("CUTEAFD_TP2_EXCHANGE must be bf16 or f32, got {other:?}"),
        };
        Ok(Self::FusedAllReduce { exchange })
    }
}

/// The fused all-reduce's two kernels on one rank.
pub(crate) struct FusedCombine<'a> {
    kernels: RtxTp2Combine<'a>,
    pub exchange: ExchangeDtype,
}

impl<'a> FusedCombine<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, exchange: ExchangeDtype) -> Result<Self> {
        Ok(Self {
            kernels: library.rtx_tp2_combine()?,
            exchange,
        })
    }

    /// Bytes of one rank's exchange payload for `rows` rows of `hidden`.
    pub(crate) fn payload_bytes(&self, rows: usize, hidden: usize) -> usize {
        rows * hidden * self.exchange.element_bytes()
    }

    /// This rank's payload: its routed partial (FP32, or null for a layer
    /// whose routed experts are not RTX-resident) plus its shared half.
    /// # Safety
    /// As [`RtxTp2Combine::partial`], on the current device.
    pub(crate) unsafe fn partial(
        &self,
        routed: *const f32,
        shared: *const u16,
        out: *mut c_void,
        rows: usize,
        hidden: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe {
            self.kernels
                .partial(routed, shared, out, rows * hidden, self.exchange, stream)
        }
    }

    /// Both payloads summed in rank order into BF16 `out`.
    /// # Safety
    /// As [`RtxTp2Combine::sum`], on the current device.
    pub(crate) unsafe fn sum(
        &self,
        rank0: *const c_void,
        rank1: *const c_void,
        out: *mut u16,
        rows: usize,
        hidden: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe {
            self.kernels
                .sum(rank0, rank1, out, rows * hidden, self.exchange, stream)
        }
    }
}

/// Owner-side routed reduction. Shared contributions remain a separate reduction;
/// the caller adds the two BF16 results only after both complete.
pub(crate) struct OwnerReduce<'a> {
    transfer: PeerTransfer<'a>,
    local_ready: Event<'a>,
    staging: Allocation<'a>,
    output: Allocation<'a>,
    reducer: V41Tp2ExpertReducer<'a>,
    capacity: u32,
    local_rank: u8,
    geometry: cuteafd_core::ExpertGeometry,
}

impl<'a> OwnerReduce<'a> {
    pub(crate) fn workspace_bytes(
        capacity: u32,
        geometry: cuteafd_core::ExpertGeometry,
    ) -> Result<usize> {
        V41Tp2ExpertReducer::require_owner_geometry(geometry)?;
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid owner reduction capacity"
        );
        Ok(capacity as usize * geometry.hidden as usize * (geometry.topk as usize * 4 + 2))
    }

    pub(crate) fn new(
        devices: [Device<'a>; 2],
        policy: Combine,
        capacity: u32,
        geometry: cuteafd_core::ExpertGeometry,
    ) -> Result<Self> {
        Self::workspace_bytes(capacity, geometry)?;
        let Combine::OwnerReduce { owner } = policy else {
            anyhow::bail!("owner reducer requires explicit OwnerReduce policy")
        };
        ensure!(
            owner < 2
                && devices[0].id != devices[1].id
                && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid owner reduction devices"
        );
        let local = devices[owner as usize];
        Ok(Self {
            transfer: PeerTransfer::new(devices[1 - owner as usize], local)?,
            local_ready: Event::new(local)?,
            staging: Allocation::new(
                local,
                capacity as usize * geometry.hidden as usize * geometry.topk as usize * 4,
            )?,
            output: Allocation::new(local, capacity as usize * geometry.hidden as usize * 2)?,
            reducer: local.run(|| local.library.v41_tp2_expert_reducer())?,
            capacity,
            local_rank: owner,
            geometry,
        })
    }

    /// # Safety
    /// Both rank outputs are immutable, live and ordered on their producer
    /// streams through completion or cancellation drainage. Deferred callers
    /// retain them until the enclosing stage chain completes.
    pub(crate) async unsafe fn reduce(
        &mut self,
        local: &Allocation<'a>,
        remote: &Allocation<'a>,
        local_producer: &Stream<'a>,
        remote_producer: &Stream<'a>,
        rows: u32,
        dtype: PartialDtype,
        layout: PartialLayout,
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            (1..=self.capacity).contains(&rows),
            "owner reduction exceeds capacity"
        );
        let columns = match layout {
            PartialLayout::Tokens => 1,
            PartialLayout::Routes { topk } => {
                ensure!(
                    topk == self.geometry.topk as usize,
                    "owner route planes differ from admitted geometry"
                );
                topk
            }
        };
        ensure!(
            dtype != PartialDtype::Bf16 || matches!(layout, PartialLayout::Routes { .. }),
            "BF16 owner reduction requires route planes"
        );
        let bytes = rows as usize
            * self.geometry.hidden as usize
            * columns
            * if dtype == PartialDtype::F32 { 4 } else { 2 };
        ensure!(
            local.device.id == self.output.device.id
                && local.buffer.bytes >= bytes
                && local_producer.device.id == local.device.id
                && std::ptr::eq(local.device.library, self.output.device.library)
                && std::ptr::eq(local_producer.device.library, self.output.device.library),
            "owner reduction local output/producer mismatch"
        );
        let started = tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG)
            .then(std::time::Instant::now);
        if !crate::shared::memory::chain::deferred() {
            remote_producer.wait().await?;
        }
        let producer_us = started.map(|start| start.elapsed().as_micros() as u64);
        self.local_ready.record(local_producer)?;
        let ready = &self.local_ready;
        let output = &self.output;
        let reducer = &self.reducer;
        let local_rank = self.local_rank;
        // SAFETY: the caller retains rank outputs; transfer retains the callback
        // and drains copy/reduction on any partial enqueue or cancellation.
        unsafe {
            self.transfer
                .copy_then(
                    remote,
                    &mut self.staging,
                    remote_producer,
                    bytes,
                    |peer, stream| {
                        output
                            .device
                            .library
                            .cuda_stream_wait_event(stream, ready.raw)?;
                        let (rank0, rank1) = if local_rank == 0 {
                            (local.buffer, peer)
                        } else {
                            (peer, local.buffer)
                        };
                        if dtype == PartialDtype::Bf16 {
                            reducer.reduce_bf16_routes(rank0, rank1, output.buffer, rows, stream)
                        } else {
                            reducer.reduce(
                                rank0,
                                rank1,
                                output.buffer,
                                rows,
                                layout == PartialLayout::Tokens,
                                stream,
                            )
                        }
                    },
                )
                .await?;
        }
        if let Some(started) = started {
            tracing::debug!(target: "cuteafd::timing", rows, token_sums=layout == PartialLayout::Tokens,
                ?layout, producer_us=producer_us.unwrap(),
                copy_reduce_us=started.elapsed().as_micros() as u64-producer_us.unwrap(),
                "TP2 routed completion");
        }
        let mut output = self.output.buffer;
        output.bytes = rows as usize * self.geometry.hidden as usize * 2;
        Ok(output)
    }
}

#[cfg(test)]
mod owner_tests {
    use super::*;
    use cuteafd_core::ExpertGeometry;

    #[test]
    fn owner_workspace_matches_native_abi_and_rejects_unsupported_geometry() -> Result<()> {
        let geometry = ExpertGeometry::DEEPSEEK_V41;
        assert_eq!(
            OwnerReduce::workspace_bytes(16, geometry)?,
            16 * 5120 * (6 * 4 + 2)
        );
        for unsupported in [
            ExpertGeometry::DEEPSEEK_V4_FLASH,
            ExpertGeometry::GLM5,
            ExpertGeometry {
                topk: 3,
                ..geometry
            },
        ] {
            assert!(OwnerReduce::workspace_bytes(16, unsupported).is_err());
        }
        assert!(OwnerReduce::workspace_bytes(0, geometry).is_err());
        assert!(OwnerReduce::workspace_bytes(4097, geometry).is_err());
        Ok(())
    }
}
