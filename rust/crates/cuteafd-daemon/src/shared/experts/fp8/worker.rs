//! Spark request adapter for the FP8 family: FP8 K32 wire rows and routes in,
//! the compact BF16 rank partial out, written straight into the registered
//! send slot when the transport permits it.
use super::{Fp8Experts, Fp8Weights};
use crate::shared::experts::execution::HostExpertExchange;
use crate::shared::memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::{
    expert::BackboneRequest, ExpertProtocolV2DeviceResponseRef, ExpertProtocolV2ResponseRef, ExpertV2Dtype,
    EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
};
use std::rc::Rc;

pub(crate) struct Fp8Worker<'a> {
    // Drain the stream before dropping the experts or the inputs.
    stream: Rc<LoadStream<'a>>,
    experts: Fp8Experts<'a, Rc<Fp8Weights<'a>>>,
    library: &'a NativeLibrary,
    inputs: [DeviceAllocation<'a>; 3],
    output: DeviceAllocation<'a>,
    capacity: usize,
    layer: usize,
    executor_id: u64,
    timing: Option<Rc<std::cell::RefCell<super::super::execution::timing::WorkerTiming<'a>>>>,
    /// Mapped requests return queued; each reply waits only for its unit fence.
    deferred: bool,
    pending: super::super::execution::timing::PendingUnits<'a>,
}

fn validate_worker_layers(layers: &[usize]) -> Result<()> {
    ensure!(!layers.is_empty(), "FP8 worker has no layers");
    super::validate_layer_ids(layers)
}

impl<'a> Fp8Worker<'a> {
    pub(crate) fn capacity(&self) -> usize { self.capacity }
    /// Exact retained input/routes/output bytes (scratch is admitted separately).
    pub(crate) fn io_bytes(info: &cuteafd_ffi::fp8_moe::Fp8MoeInfo, bf16_sibling: bool,
        capacity: usize) -> Result<usize> {
        cuteafd_ffi::fp8_moe::fp8_slot_io_bytes(info.hidden, info.topk, info.wire_input,
            bf16_sibling, capacity)
    }

    pub(crate) fn workspace_bytes(experts: &Fp8Weights<'_>, capacity: usize) -> Result<usize> {
        Self::io_bytes(experts.module.info(), experts.bf16_module.is_some(), capacity)?
            .checked_add(experts.slot_scratch_bytes(capacity)?)
            .context("native slot workspace size overflow")
    }

    pub(crate) fn new(library: &'a NativeLibrary, experts: Rc<Fp8Weights<'a>>, capacity: u32) -> Result<Self> {
        Self::with_scratch(library, experts, capacity, None)
    }

    pub(crate) fn with_scratch(library: &'a NativeLibrary, experts: Rc<Fp8Weights<'a>>, capacity: u32,
        scratch: Option<Rc<DeviceAllocation<'a>>>) -> Result<Self> {
        let geometry = cuteafd_core::expert_geometry();
        let (hidden, topk, capacity) = (geometry.hidden as usize, geometry.topk as usize, capacity as usize);
        validate_worker_layers(&experts.layers.iter().map(|layer| layer.layer).collect::<Vec<_>>())?;
        ensure!(experts.module.info().capacity_for(capacity).is_some(), "FP8 package lacks capacity {capacity}");
        ensure!(experts.wire_input() || matches!(experts.module.info().weights,
            cuteafd_ffi::fp8_moe::Fp8MoeWeights::Nvfp4 | cuteafd_ffi::fp8_moe::Fp8MoeWeights::Bf16),
            "Spark workers need a wire-input package, or native BF16-input weights");
        let experts = match scratch {
            Some(scratch) => Fp8Experts::from_scratch(experts, capacity, scratch)?,
            None => Fp8Experts::from_shared(library, experts, capacity)?,
        };
        let inputs = [
            DeviceAllocation::new(library, capacity * if !experts.wire_input() || experts.bf16_module.is_some() { hidden * 2 }
                else { hidden + hidden / 32 })?,
            DeviceAllocation::new(library, capacity * topk * 4)?,
            DeviceAllocation::new(library, capacity * topk * 4)?,
        ];
        let output = DeviceAllocation::new(library, capacity * hidden * 2)?;
        let executor_id = cuteafd_transport::expert::SparkWorld::canonical(experts.tp)?.topology().executor_id(experts.rank)?;
        let timing = super::super::execution::timing::WorkerTiming::from_env(library, experts.rank)?
            .map(|timing| Rc::new(std::cell::RefCell::new(timing)));
        Ok(Self {
            stream: Rc::new(LoadStream { library, raw: library.cuda_stream_create()? }),
            experts,
            library,
            inputs,
            output,
            capacity,
            layer: 0,
            executor_id,
            timing,
            deferred: false,
            pending: super::super::execution::timing::PendingUnits::new(library, 1)?,
        })
    }

    pub(crate) fn installed_grouping(&self, loaded: cuteafd_transport::worker_selection::WorkerSelection,
        rank: usize) -> Result<cuteafd_transport::worker_selection::InstalledExpertGrouping> {
        self.experts.installed_grouping(loaded, rank, self.capacity)
    }

    /// All homogeneous groups in one slot run serially; their gap clock must
    /// follow the slot, not skip compute from a different weight group.
    pub(crate) fn share_owner(&mut self, other: &Self) {
        self.stream = other.stream.clone();
        self.timing = other.timing.clone();
    }

    /// Leave mapped requests queued at return (see `deferred`).
    pub(crate) fn defer(&mut self, units: usize) -> Result<()> {
        ensure!(self.pending.is_empty(), "cannot resize an active FP8 queue");
        self.pending = super::super::execution::timing::PendingUnits::new(self.library, units)?;
        self.deferred = true;
        Ok(())
    }

    /// Publish deferred diagnostics after the owner's existing completion check.
    pub(crate) fn complete(&mut self) -> Result<()> {
        let Some((request_id, layer, rows, executor_id)) = self.pending.pop()? else { return Ok(()) };
        if let Some(timing) = &mut self.timing {
            // SAFETY: the caller observed this stream drained after both records.
            unsafe { timing.borrow_mut().publish(request_id, layer, rows, executor_id, "fp8_moe")?; }
        }
        Ok(())
    }

    pub(crate) fn unit_ready(&self) -> Result<bool> { self.pending.ready() }

    pub(crate) fn synchronize(&self) -> Result<()> {
        // SAFETY: this owner retains its stream and every queued buffer.
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw)?; }
        Ok(())
    }

    /// Whether every request queued on this execution's stream has completed.
    pub(crate) fn stream_idle(&self) -> Result<bool> {
        // SAFETY: the owned stream is live for the worker's lifetime.
        unsafe { self.library.cuda_stream_query(self.stream.raw) }
    }

    pub(crate) fn bind_layer(&mut self, layer: usize) -> Result<()> {
        ensure!(layer < self.experts.layers.len(), "requested FP8 layer is not resident");
        self.layer = layer;
        Ok(())
    }

    fn execute(&mut self, request: &BackboneRequest<'_>, executor_id: u64, exchange: &mut HostExpertExchange,
        destination: Option<CuteafdDeviceBuffer>, hidden_view: Option<CuteafdDeviceBuffer>,
        routes_view: Option<CuteafdDeviceBuffer>)
        -> Result<CuteafdDeviceBuffer> {
        ensure!(request.layer() as usize == self.experts.layers[self.layer].layer,
            "request does not match the selected FP8 layer");
        ensure!(executor_id == self.executor_id, "FP8 response executor identity mismatch");
        let rows = request.rows() as usize;
        ensure!(rows > 0 && rows <= self.capacity, "FP8 request exceeds capacity");
        // FP8 K32 wire rows, or BF16 rows when the BF16-input package is loaded.
        let bf16 = request.require_input_dtype(ExpertV2Dtype::Fp8E4m3Ue8m0K32 as u32).is_err();
        if bf16 {
            ensure!(!self.experts.wire_input() || self.experts.bf16_module.is_some(),
                "BF16 expert input needs the BF16-input FP8 package (fp8-<family>-bf16) on this rank");
            request.require_input_dtype(ExpertV2Dtype::Bf16 as u32)?;
        }
        ensure!(bf16 || self.experts.wire_input(),
            "native BF16/NVFP4 Spark input must be BF16, not FP8 K32");
        ensure!(request.hidden().len() <= self.inputs[0].buffer.bytes, "FP8 request rows exceed the input buffer");
        let bytes = request.plane_bytes()?;
        let routes = rows * cuteafd_core::expert_geometry().topk as usize;
        if routes_view.is_none() {
            ensure!(self.pending.is_empty(), "host routes cannot overwrite queued expert inputs");
            request.copy_routes_into(&mut exchange.ids, &mut exchange.routing)?;
        }
        self.pending.next_slot()?;
        ensure!(cfg!(target_endian = "little"), "native exchange requires little-endian storage");
        // Every copy follows the previous unit's consumers on this owner stream;
        // the transport retains each mapped source independently until its reply.
        let hidden_bytes = request.hidden().len();
        match hidden_view.filter(|view| view.bytes >= hidden_bytes) {
            // SAFETY: the view is device-visible request storage of at least
            // `hidden_bytes`, retained by the transport until this request's
            // response is emitted, which follows the stream synchronize below.
            Some(view) => unsafe {
                self.library.copy_d2d_async(self.inputs[0].buffer,
                    CuteafdDeviceBuffer { bytes: hidden_bytes, ..view }, hidden_bytes, self.stream.raw)?
            },
            None => self.library.copy_h2d(self.inputs[0].buffer, request.hidden())?,
        }
        if let Some(source) = routes_view {
            // SAFETY: transport retains this unit's mapped records until its fence and send complete.
            unsafe { super::super::service::routes::copy_mapped_routes(self.library, request, source,
                self.inputs[1].buffer, self.inputs[2].buffer, self.stream.raw)?; }
        } else {
            // SAFETY: the host path has no queued unit and these initialized slices live through each copy.
            unsafe {
                self.library.copy_h2d(self.inputs[1].buffer,
                    std::slice::from_raw_parts(exchange.ids.as_ptr().cast::<u8>(), routes * 4))?;
                self.library.copy_h2d(self.inputs[2].buffer,
                    std::slice::from_raw_parts(exchange.routing.as_ptr().cast::<u8>(), routes * 4))?;
            }
        }
        let output = match destination {
            Some(output) => output,
            None => CuteafdDeviceBuffer { bytes, ..self.output.buffer },
        };
        ensure!(output.bytes >= bytes, "FP8 response slot is too small");
        // SAFETY: inputs, output and the resident layer are live device memory of
        // the documented extents; the stream is synchronized below.
        unsafe {
            if let Some(timing) = &self.timing { timing.borrow().record(0, self.stream.raw)?; }
            if bf16 {
                self.experts.run_bf16(self.layer, rows, self.inputs[0].buffer.ptr, self.inputs[1].buffer.ptr,
                    self.inputs[2].buffer.ptr, output.ptr, self.stream.raw)?;
            } else {
                self.experts.run(self.layer, rows, self.inputs[0].buffer.ptr, self.inputs[1].buffer.ptr,
                    self.inputs[2].buffer.ptr, output.ptr, self.stream.raw)?;
            }
            if let Some(timing) = &self.timing { timing.borrow().record(1, self.stream.raw)?; }
            // The transport holds both mapped slots until this unit's fence,
            // independent of later work queued on the owner's stream.
            if self.deferred && destination.is_some() {
                self.pending.record((request.request_id(), request.layer(), request.rows(), executor_id), self.stream.raw)?;
                return Ok(output);
            }
            self.library.cuda_stream_synchronize(self.stream.raw)?;
        }
        if let Some(timing) = &mut self.timing {
            // SAFETY: the existing worker barrier completed both event records.
            unsafe { timing.borrow_mut().publish(request.request_id(), request.layer(), request.rows(), executor_id, "fp8_moe")?; }
        }
        if destination.is_none() {
            ensure!(exchange.partials.len() >= bytes, "FP8 host exchange is too small");
            self.library.copy_d2h(&mut exchange.partials[..bytes], output)?;
        }
        Ok(output)
    }

    /// # Safety
    /// The transport exclusively owns a GPU-accessible send slot on this device
    /// and retains it through the response send completion.
    pub(crate) unsafe fn execute_mapped_request(&mut self, request: &BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, slot: CuteafdDeviceBuffer, hidden: Option<CuteafdDeviceBuffer>)
        -> Result<Option<ExpertProtocolV2DeviceResponseRef<'static>>> {
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let bytes = request.plane_bytes()?;
        if !request.permits_device_response() || slot.bytes < prefix + bytes {
            return Ok(None);
        }
        ensure!(!slot.ptr.is_null(), "null FP8 response slot");
        let output = CuteafdDeviceBuffer { ptr: slot.ptr.cast::<u8>().add(prefix).cast(), bytes, ..slot };
        let response = request.response_device(executor_id, output)?;
        self.execute(request, executor_id, exchange, Some(output), hidden, None)?;
        Ok(Some(response))
    }

    /// # Safety
    /// Transport retains all mapped frames through this unit's completion and send.
    pub(crate) unsafe fn execute_mapped_payload(&mut self, request: &BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, mapped: cuteafd_transport::ProtocolV2RequestDevicePayload)
        -> Result<Option<ExpertProtocolV2DeviceResponseRef<'static>>> {
        let Some(slot) = mapped.response_slot else { return Ok(None) };
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let bytes = request.plane_bytes()?;
        if !request.permits_device_response() || slot.bytes < prefix + bytes { return Ok(None); }
        // SAFETY: checked prefix/plane extent stays in the exclusively retained send slot.
        let output = CuteafdDeviceBuffer { ptr: unsafe { slot.ptr.cast::<u8>().add(prefix).cast() }, bytes, ..slot };
        let response = request.response_device(executor_id, output)?;
        self.execute(request, executor_id, exchange, Some(output), Some(mapped.hidden_payload), Some(mapped.routes))?;
        Ok(Some(response))
    }

    pub(crate) fn execute_host_chunks<F>(&mut self, request: &BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, row_indices: &mut [u32], max_frame_bytes: usize, mut sink: F) -> Result<()>
    where
        F: FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    {
        let chunk_rows = request.response_chunk_rows(max_frame_bytes)?;
        ensure!(row_indices.len() >= chunk_rows as usize, "response row-index scratch is too short");
        self.execute(request, executor_id, exchange, None, None, None)?;
        let stride = cuteafd_core::expert_geometry().row_bytes() as usize;
        for start in (0..request.rows()).step_by(chunk_rows as usize) {
            let end = start.saturating_add(chunk_rows).min(request.rows());
            sink(request.response_chunk(executor_id, start,
                &exchange.partials[start as usize * stride..end as usize * stride], row_indices, max_frame_bytes)
                .context("FP8 response chunk")?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn native_slots_own_distinct_capacity_sized_scratch_and_charge_workspace_once() -> Result<()> {
        use cuteafd_ffi::{fp8_moe::Fp8MoeModule, memory_ledger::{self, Space},
            native_library_lifetime_fixture::Fixture};
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        library.cuda_set_device(7)?;
        fixture.configure_memory(&library, 7, 1 << 30)?;
        std::fs::hard_link(fixture.directory().join("liblifetime_fixture.so"),
            fixture.directory().join("libcuteafd_fp8moe.so"))?;
        let geometry = cuteafd_core::expert_geometry();
        let baseline = memory_ledger::snapshot().total(Space::Device, 7);
        for format in [1u32, 2, 4, 5] {
            for capacity in [1usize, 17] {
                let wire = matches!(format, 1 | 2);
                let mut words = [5 | format << 8 | 1 << 16 | 1 << 24, geometry.hidden,
                    128, 2, geometry.topk, 256, 2, if wire { 7 } else { 1 }, 0,
                    2, 16, 4096, 0, 0, 0, 0];
                fixture.configure_fp8(&library, &words, &[1024, 4096, 0, 0, 0, 0])?;
                // SAFETY: this test compiled the CPU-only native ABI above.
                let module = unsafe { Fp8MoeModule::load(fixture.directory())? };
                let bf16_module = if wire {
                    words[7] = 1;
                    fixture.configure_fp8(&library, &words, &[2048, 8192, 0, 0, 0, 0])?;
                    // SAFETY: the same trusted CPU ABI snapshots independent metadata per module.
                    Some(unsafe { Fp8MoeModule::load(fixture.directory())? })
                } else { None };
                let resident = {
                    let _scope = memory_ledger::scope("experts/weights");
                    DeviceAllocation::new(&library, 512)?
                };
                let weights = Rc::new(Fp8Weights {
                    layers: vec![super::super::Fp8Layer { layer: 3,
                        slice: cuteafd_core::SparkSlice { unit_map: None, layer: 3, first: 0,
                            width: 128, stored_width: 128, tiers: vec![], projection_counts: vec![] },
                        regions: vec![Some(resident.into())] }],
                    module, bf16_module, tp: 2, rank: 0, weight_read_bytes: 256,
                });
                let scratch = match (wire, capacity) { (true, 1) => 2048, (true, _) => 8192,
                    (false, 1) => 1024, (false, _) => 4096 };
                assert_eq!(weights.resident_bytes(), 512);
                assert_eq!(weights.slot_scratch_bytes(capacity)?, scratch);
                let io = Fp8Worker::io_bytes(weights.module.info(), wire, capacity)?;
                let workspace = Fp8Worker::workspace_bytes(&weights, capacity)?;
                assert_eq!(workspace, io + scratch);
                let workers = {
                    let _scope = memory_ledger::scope("experts/workspace/slot");
                    [Fp8Worker::new(&library, weights.clone(), capacity as u32)?,
                        Fp8Worker::new(&library, weights.clone(), capacity as u32)?]
                };
                assert!(Rc::ptr_eq(&workers[0].experts.weights, &workers[1].experts.weights));
                assert_ne!(workers[0].experts.scratch.buffer.ptr, workers[1].experts.scratch.buffer.ptr);
                for worker in &workers {
                    assert_eq!(worker.experts.scratch_bytes(), scratch);
                    assert_eq!(worker.inputs.iter().map(|input| input.buffer.bytes).sum::<usize>()
                        + worker.output.buffer.bytes, io);
                }
                let ledger = memory_ledger::snapshot();
                let scopes = ledger.by_scope(Space::Device, 7);
                assert_eq!(scopes["experts/weights"], 512);
                assert_eq!(scopes["experts/workspace/native_scratch"], 2 * scratch);
                assert_eq!(scopes["experts/workspace/slot"], 2 * io);
                assert_eq!(ledger.total(Space::Device, 7) - baseline, 512 + 2 * workspace);
                let row = ledger.rows.iter().find(|row| row.key.device == 7
                    && row.key.scope == "experts/workspace/native_scratch").unwrap();
                assert_eq!(row.allocations, 2);
                drop(workers);
                assert_eq!(memory_ledger::snapshot().total(Space::Device, 7) - baseline, 512);
                drop(weights);
                assert_eq!(memory_ledger::snapshot().total(Space::Device, 7), baseline);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn three_queued_prefill_units_preserve_distinct_routes_inputs_and_replies() -> Result<()> {
        use crate::shared::memory::HostAllocation;
        use cuteafd_ffi::{fp8_moe::Fp8MoeModule, native_library_lifetime_fixture::Fixture};
        use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RequestView,
            ExpertProtocolV2RowDescriptor, ExpertProtocolV2RouteEntry, ExpertV2SourceKind,
            ProtocolV2RequestDevicePayload, expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        library.cuda_set_device(7)?;
        fixture.configure_memory(&library, 7, 1 << 30)?;
        fixture.queued_experts(&library)?;
        std::fs::hard_link(fixture.directory().join("liblifetime_fixture.so"),
            fixture.directory().join("libcuteafd_fp8moe.so"))?;
        let geometry = cuteafd_core::expert_geometry();
        let words = [5 | 1 << 8 | 1 << 16 | 1 << 24, geometry.hidden,
            128, geometry.experts, geometry.topk, 256, 2, 7, 0,
            2, 16, 4096, 0, 0, 0, 0];
        fixture.configure_fp8(&library, &words, &[1024, 4096, 0, 0, 0, 0])?;
        // SAFETY: this test built the trusted CPU ABI immediately above.
        let module = unsafe { Fp8MoeModule::load(fixture.directory())? };
        let layers = [3, 4].into_iter().map(|layer| super::super::Fp8Layer { layer,
            slice: cuteafd_core::SparkSlice { unit_map: None, layer: layer as u32, first: 0,
                width: 128, stored_width: 128, tiers: vec![], projection_counts: vec![] },
            regions: vec![None, None, None, None, None, None] }).collect();
        let weights = Rc::new(Fp8Weights { layers, module, bf16_module: None,
            tp: 2, rank: 0, weight_read_bytes: 256 });
        let mut worker = Fp8Worker::new(&library, weights, 4096)?;
        worker.defer(3)?;
        let mut exchange = HostExpertExchange::new(4096)?;
        let rows = 2u32;
        let hidden = geometry.hidden as usize;
        let topk = geometry.topk as usize;
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let mut frames = Vec::new();
        let mut replies = Vec::new();
        let mut expected = Vec::new();
        for unit in 0..3usize {
            let layer = 3 + unit % 2;
            let input = 7 + unit as u8 * 4;
            let gate = (unit + 1) as f32;
            let ids = (0..rows as usize * topk).map(|route|
                ((unit * 23 + route * 3) % geometry.experts as usize) as u32).collect::<Vec<_>>();
            let mut owned = ExpertProtocolV2Request::new(91 + unit as u64, 17, layer as u32, geometry.hidden,
                ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                (0..rows).map(|row| ExpertProtocolV2RowDescriptor { row_id: row as u64,
                    source_kind: ExpertV2SourceKind::Prefill, source_request_id: unit as u64,
                    token_position: row as u64, route_offset: row * geometry.topk,
                    route_count: geometry.topk }).collect(),
                ids.iter().enumerate().map(|(route, &expert_id)| ExpertProtocolV2RouteEntry {
                    row_index: (route / topk) as u32, expert_id, gate_weight: gate }).collect(),
                vec![input; rows as usize * (hidden + hidden / 32)])?;
            owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
            let encoded = owned.encode()?;
            let mut frame = HostAllocation::new(&library, encoded.len())?;
            frame.bytes_mut().copy_from_slice(&encoded);
            let mut reply = HostAllocation::new(&library, prefix + rows as usize * hidden * 2 + 32)?;
            reply.bytes_mut().fill(0xa5);
            expected.push((0..rows as usize).flat_map(|row| {
                let value = input as u16 + ids[row * topk..(row + 1) * topk].iter()
                    .map(|id| ((id + 1) as usize * (unit + 1)) as u16).sum::<u16>();
                std::iter::repeat_n(value.to_le_bytes(), hidden).flatten()
            }).collect::<Vec<_>>());
            frames.push(frame);
            replies.push(reply);
        }
        for unit in 0..3 {
            let view = ExpertProtocolV2RequestView::parse(frames[unit].bytes())?;
            let request = BackboneRequest::parse(frames[unit].bytes(), 4096)?;
            let alias = library.cuda_host_buffer_device_alias(frames[unit].buffer)?;
            let field = |bytes: &[u8]| CuteafdDeviceBuffer {
                ptr: bytes.as_ptr().cast_mut().cast(), bytes: bytes.len(), ..alias };
            let mapped = ProtocolV2RequestDevicePayload { routes: field(view.route_bytes()),
                hidden_payload: field(view.hidden_payload()), execution_lane: 0,
                response_slot: Some(library.cuda_host_buffer_device_alias(replies[unit].buffer)?) };
            worker.bind_layer(unit % 2)?;
            // SAFETY: every frame and reply is retained until all queued unit fences complete.
            let response = unsafe { worker.execute_mapped_payload(&request, worker.executor_id,
                &mut exchange, mapped)? }.context("mapped queue request rejected")?;
            assert_eq!(response.header.request_id, 91 + unit as u64);
            assert!(!worker.unit_ready()?);
            if unit == 0 {
                // Input/routes have arrived; the first expert read is still queued.
                // Submit unit 1's different routes before allowing that read to run.
                fixture.advance_queued_experts(&library, 3)?;
                assert!(!worker.unit_ready()?);
            }
        }
        assert_eq!(worker.pending.len(), 3);
        assert!(!worker.stream_idle()?);
        for unit in 0..3 {
            fixture.advance_queued_experts(&library, if unit == 0 { 1 } else { 4 })?;
            assert!(worker.unit_ready()?);
            if unit < 2 { assert!(!worker.stream_idle()?); }
            worker.complete()?;
            assert_eq!(&replies[unit].bytes()[prefix..prefix + expected[unit].len()], expected[unit]);
            assert!(replies[unit].bytes()[..prefix].iter().all(|byte| *byte == 0xa5));
            assert!(replies[unit].bytes()[prefix + expected[unit].len()..].iter().all(|byte| *byte == 0xa5));
        }
        assert!(worker.stream_idle()? && worker.pending.is_empty());
        Ok(())
    }

    #[test]
    fn width_group_accepts_sparse_layers_but_not_empty_or_duplicate_ids() {
        for layers in [&[3][..], &[3, 4, 5][..], &[3, 6, 7, 8, 9, 12][..]] {
            super::validate_worker_layers(layers).unwrap();
        }
        for layers in [&[][..], &[3, 3][..], &[6, 3][..]] {
            assert!(super::validate_worker_layers(layers).is_err());
        }
    }



}
