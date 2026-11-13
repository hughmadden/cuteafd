//! One coordinator wave owns TP route planes through final native reduction.
use crate::shared::memory::Tag;
use super::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_core::CoordinatorExpertResidency;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::{
    expert::{SparkExperts, V41_PARTIAL_ROW_BYTES, V41_ROUTED_EXPERTS},
    ExpertProtocolV2Request,
};

/// For each route of `experts` (rows of six distinct ids) in `layer`, whether
/// the resident set marks it for the coordinator (a probe request, encoded as the
/// link's requests are).
pub(crate) fn coordinator_routes(resident: &[CoordinatorExpertResidency], layer: u32, experts: &[u32])
    -> Result<Vec<bool>> {
    use cuteafd_transport::{ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind};
    ensure!(!experts.is_empty() && experts.len() % 6 == 0, "a tier probe routes whole rows of six experts");
    let rows = (experts.len() / 6) as u32;
    let descriptors = (0..rows).map(|row| ExpertProtocolV2RowDescriptor { row_id: row.into(),
        source_kind: ExpertV2SourceKind::Prefill, source_request_id: 1, token_position: row.into(),
        route_offset: row * 6, route_count: 6 }).collect();
    let routes = experts.iter().enumerate().map(|(index, &expert_id)| ExpertProtocolV2RouteEntry {
        row_index: index as u32 / 6, expert_id, gate_weight: 1.0 / 6.0 }).collect();
    let request = crate::families::deepseek_v41::v41_backbone_router::build_spark_request(
        1, 0, layer, ExpertV2Dtype::Fp8E4m3Ue8m0K32, descriptors, routes,
        vec![0; rows as usize * 5280].into(), Some(resident))?;
    request.routes.iter().map(|route| Ok(cuteafd_transport::expert::CoordinatorRouteWord::decode(
        route.expert_id, true)?.coordinator)).collect()
}

/// A Spark link with V4.1's canonical request contract and optional immutable
/// per-layer resident sets shared with the exchange worker. Decode lanes each own one;
/// prefill units share one link whose return slots hold the in-flight units
/// (PLAN "Prefill contract": one transport, N waves).
pub(crate) struct V41Link<'a> {
    link: crate::shared::spark_intake::SparkLink<'a>,
    resident: Option<std::sync::Arc<Vec<CoordinatorExpertResidency>>>,
    /// The GPU expert tier whose experts this link's requests leave to the
    /// coordinator ([`Self::enable_tier`]).
    tier: Option<std::sync::Arc<super::tier::TierResidency>>,
}
impl<'a> V41Link<'a> {
    pub(crate) fn new(link: crate::shared::spark_intake::SparkLink<'a>) -> Result<Self> {
        Ok(Self { link, resident: None, tier: None })
    }
    /// The one prefill link: `slots` return slots of `unit_rows` rows.
    pub(crate) fn prefill(library: &'a NativeLibrary, transport: SparkExperts, slots: usize, unit_rows: u32)
        -> Result<Self> {
        let ranks = transport.world_size();
        library.v41_compact_reducer()?.require_rank_count(ranks as u32)?;
        let timing = crate::families::deepseek_v41::v41_native_serve::protocol_v2_timing();
        Self::new(crate::shared::spark_intake::SparkLink::from_transport_waves(library, transport,
            V41_PARTIAL_ROW_BYTES as usize, slots, unit_rows,
            timing)?)
    }
    /// Device bytes of the prefill link's intake: every rank's partial-sum
    /// plane over its `reserve_rows` (slots x unit) rows.
    pub(crate) fn prefill_device_bytes(reserve_rows: usize, ranks: usize,
        format: cuteafd_core::SparkReplyFormat) -> Result<usize> {
        ensure!((1..=8).contains(&ranks) && reserve_rows > 0, "invalid V4.1 prefill link geometry");
        usize::try_from(cuteafd_loader::serving_capacity::prefill_return_bytes_with_format(
            1, reserve_rows as u64, ranks as u64, 5120, format)?).context("prefill link budget overflow")
    }
    /// Leave each layer's resident experts to the coordinator, before the first request.
    pub(crate) fn enable_tier(&mut self, tier: std::sync::Arc<super::tier::TierResidency>) -> Result<()> {
        ensure!(self.tier.is_none(), "the GPU expert tier is already enabled on this Spark link");
        // The tier's result is one more plane in the link's reduce (eight at most).
        ensure!(self.link.world_size() < crate::shared::spark_intake::MAX_INTAKE_RANKS,
            "a {}-rank Spark link has no reduce plane left for the GPU expert tier", self.link.world_size());
        ensure!(tier.experts() == V41_ROUTED_EXPERTS, "GPU expert tier must describe all 384 experts");
        let resident = tier.resident().iter()
            .map(|ids| CoordinatorExpertResidency::from_experts(ids))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        self.resident = Some(std::sync::Arc::new(resident));
        self.tier = Some(tier);
        Ok(())
    }
    /// The GPU expert tier this link's requests leave to the coordinator.
    pub(crate) fn tier(&self) -> Option<&std::sync::Arc<super::tier::TierResidency>> { self.tier.as_ref() }
    /// For each route of `experts` (rows of six distinct ids) in `layer`,
    /// whether this link's request marks it for the coordinator: the host
    /// view of the GPU expert tier, through the link's immutable sets.
    pub(crate) fn coordinator_routes(&self, layer: u32, experts: &[u32]) -> Result<Vec<bool>> {
        coordinator_routes(self.resident.as_deref().context("the Spark link has no GPU expert tier")?, layer, experts)
    }
    /// Whether this link's requests leave experts of `layer` to the coordinator.
    pub(crate) fn tier_holds(&self, layer: usize) -> bool {
        self.tier.as_ref().is_some_and(|tier| !tier.ids(layer).is_empty())
    }
    pub(crate) fn link(&mut self) -> &mut crate::shared::spark_intake::SparkLink<'a> { &mut self.link }
    pub(crate) fn capacity(&self) -> u32 { self.link.capacity() }
    pub(crate) fn world_size(&self) -> usize { self.link.world_size() }
    pub(crate) fn reset_connections(&mut self) -> Result<()> { self.link.reset_connections() }
    /// Opens reply planes at capacity without naming or executing an expert layer.
    pub(crate) fn bootstrap(&mut self, nvfp4: bool) -> Result<()> {
        let dtype = if nvfp4 { cuteafd_transport::ExpertV2Dtype::Bf16 }
            else { cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32 };
        let mut shape = cuteafd_transport::SparkSessionShape::new(self.link.capacity(), 5120, 6, dtype)?;
        shape.flags |= cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        if self.resident.is_some() { shape.flags |= cuteafd_transport::expert::V41_COORDINATOR_ROUTES_REQUEST_FLAG; }
        self.link.bootstrap(&shape)
    }

}

pub(crate) struct NativeTp4Wave<'a> {
    /// The lane's own link (decode lanes); `None` for a prefill-only slot,
    /// whose units send on the shared prefill link.
    link: Option<V41Link<'a>>,
    // Drop drains the stream before fields release any GPU allocations.
    stream: LoadStream<'a>,
    output: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    ready_rows: Option<u32>,
    local: Option<super::local::LocalExpertWave<'a>>,
    tp2: Option<Box<super::tp2_ffn::Wave<'a>>>,
}

impl<'a> NativeTp4Wave<'a> {
    fn own_link(&mut self) -> Result<&mut V41Link<'a>> {
        self.link.as_mut().context("prefill-only slot has no Spark link of its own")
    }
    pub(crate) fn spark_world(&self) -> usize { self.link.as_ref().map_or(0, V41Link::world_size) }

    pub fn install_local(&mut self, wave: super::local::LocalExpertWave<'a>) -> Result<()> {
        ensure!(self.local.is_none() && self.tp2.is_none(), "local expert lane already installed");
        self.local = Some(wave);
        Ok(())
    }
    pub fn has_local_layer(&self, layer: usize) -> bool {
        self.local.as_ref().is_some_and(|wave| wave.contains(layer)) || self.has_tp2_layer(layer)
    }
    pub fn install_tp2(&mut self, wave: super::tp2_ffn::Wave<'a>) -> Result<()> {
        ensure!(self.local.is_none() && self.tp2.is_none(), "local expert lane already installed");
        self.tp2 = Some(Box::new(wave));
        Ok(())
    }
    /// Runs the GPU expert tier on this decode lane's Spark layers: its TP2
    /// FFN (`tp2_ffn::Wave::tiered`) runs the tier's halves beside the Spark
    /// wave, and its link's requests leave the same experts (one
    /// `TierResidency`) to the coordinator. A lane's two views of the set can
    /// then never disagree: the Sparks skip exactly the routes the GPUs run.
    /// At rest, before the link's first request.
    pub(crate) fn enable_tier(&mut self) -> Result<()> {
        let tp2 = self.tp2.as_ref().context("the GPU expert tier runs on the lane's TP2 FFN (two RTX)")?;
        let residency = tp2.tier().context("this lane's TP2 FFN runs no GPU expert tier")?.clone();
        ensure!(self.link.is_some(), "the GPU expert tier runs on a decode lane with its own Spark link");
        self.own_link()?.enable_tier(residency)?;
        self.verify_tier()
    }
    /// The startup check ([`super::tier::check_views`]): for every layer, the
    /// routes this lane's Spark requests leave to the coordinator are exactly
    /// those both GPUs' slot tables run, each slot naming its route's expert.
    /// Host marks come from the link's resident sets, device slots from each GPU's
    /// uploaded tables through the slot kernel, as the serving path uses them.
    fn verify_tier(&self) -> Result<()> {
        let link = self.link.as_ref().context("the GPU expert tier runs on a decode lane with its own Spark link")?;
        let tp2 = self.tp2.as_ref().context("the GPU expert tier runs on the lane's TP2 FFN")?;
        let residency = link.tier().context("the Spark link has no GPU expert tier")?;
        ensure!(tp2.tier().is_some_and(|tier| std::sync::Arc::ptr_eq(tier, residency)),
            "the Spark link and the TP2 FFN hold different GPU expert tiers");
        let probe = super::tier::probe_ids(residency.experts());
        for layer in 0..residency.layers() {
            let host = link.coordinator_routes(layer as u32, &probe)?;
            for (gpu, slots) in tp2.probe_tier_slots(layer, &probe)?.iter().enumerate() {
                super::tier::check_views(residency, layer, gpu as i32, &probe, &host, slots)?;
            }
        }
        tracing::info!(layers = residency.resident_layers().len(), experts = residency.total(),
            "GPU expert tier: Spark marking and both GPUs' slot tables agree on every layer");
        Ok(())
    }
    /// Selects a prefill slot's rows in its TP2 FFN's shared planes, at rest
    /// (`shared::memory::row_window`); the reduce output is the slot's own.
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        if let Some(tp2) = self.tp2.as_mut() { tp2.select_window(window)?; }
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn tp2_window(&self) -> Option<&crate::shared::memory::row_window::RowWindow> {
        self.tp2.as_ref().map(|tp2| tp2.window())
    }
    pub fn has_tp2_layer(&self, layer: usize) -> bool {
        self.tp2.as_ref().is_some_and(|wave| wave.contains(layer))
    }
    pub fn has_tp2_shared_layer(&self, layer: usize) -> bool {
        self.tp2.as_ref().is_some_and(|wave| wave.contains_shared(layer))
    }
    /// # Safety
    /// Input and router producers have completed. Both borrowed outputs remain
    /// immutable through this operation, including cancellation draining.
    pub async unsafe fn execute_tp2_ffn(&mut self, input: &crate::families::deepseek_v41::v41_block::FfnInput<'_>,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let binding = routed.binding()?;
        ensure!(binding == input.binding() && input.layer == routed.layer
            && input.tokens.len() == routed.rows as usize && input.tokens == routed.tokens,
            "TP2 FFN input/router identity mismatch");
        let values = unsafe { self.tp2.as_mut().context("TP2 expert lane missing")?
            .execute(input.layer, routed.rows, input.values, routed.expert_input, routed.ids, routed.routing).await? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding, _owner: std::marker::PhantomData })
    }
    /// # Safety
    /// Completed router/shared buffers stay live and unmodified through drain.
    pub unsafe fn execute_local_ffn(&mut self,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let values = unsafe { self.local.as_mut().context("local expert lane missing")?.execute(routed, shared)? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding: routed.binding()?, _owner: std::marker::PhantomData })
    }
    /// # Safety
    /// Completed router/shared inputs remain immutable until completion or drain.
    pub async unsafe fn execute_local_ffn_cooperative(&mut self,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let binding = routed.binding()?;
        let values = unsafe { self.local.as_mut().context("local expert lane missing")?
            .execute_cooperative(routed, shared).await? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding, _owner: std::marker::PhantomData })
    }
    /// Admission invalidates prior output; completed RoCE sessions remain reusable.
    pub fn begin_request(&mut self) {
        self.ready_rows = None;
    }
    /// After a cancelled or failed pass whose streams drained: stops the
    /// exchange, resets the connections and starts it again.
    pub fn reset_connections(&mut self) -> Result<()> {
        self.ready_rows = None;
        match self.link.as_mut() { Some(link) => link.reset_connections(), None => Ok(()) }
    }
    /// Legacy TP4 upper-bound reservation (four rank planes), also sufficient
    /// for a legacy TP2 transport.
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        Self::device_bytes_for(capacity, 4)
    }
    /// Reservation for exactly `ranks` physical rank planes plus the final
    /// output plane. Six-rank layouts need their own, larger reservation.
    pub fn device_bytes_for(capacity: u32, ranks: usize) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid native TP wave capacity"
        );
        ensure!(
            (1..=8).contains(&ranks),
            "native TP wave requires one through eight ranks"
        );
        usize::try_from(cuteafd_loader::placement::families::deepseek_v41::owners::transport_bytes(
            u64::from(capacity), ranks)?).context("native TP wave budget overflow")
    }
    pub fn new(
        library: &'a NativeLibrary,
        transport: SparkExperts,
        available_bytes: usize,
    ) -> Result<Self> {
        let capacity = transport.capacity();
        let world_size = transport.world_size();
        ensure!(
            (1..=8).contains(&world_size),
            "native TP wave requires one through eight ranks"
        );
        ensure!(
            Self::device_bytes_for(capacity, world_size)? <= available_bytes,
            "native TP wave exceeds device budget"
        );
        let reducer = library.v41_compact_reducer()?;
        // Prove the exact physical-rank reduction is available before any plane
        // or output allocation. Legacy two-rank transports also fail here rather
        // than at their first request.
        reducer.require_rank_count(world_size as u32)?;
        let timing = crate::families::deepseek_v41::v41_native_serve::protocol_v2_timing();
        let link = V41Link::new(crate::shared::spark_intake::SparkLink::from_transport(library, transport,
            V41_PARTIAL_ROW_BYTES as usize, timing)?)?;
        let mut wave = Self::unlinked(library, capacity)?;
        wave.link = Some(link);
        Ok(wave)
    }
    /// A prefill-only slot's FFN owner: its reduce output, stream and local or
    /// TP2 expert workspace, without a Spark link (its units send on the
    /// shared prefill link).
    pub fn unlinked(library: &'a NativeLibrary, capacity: u32) -> Result<Self> {
        ensure!(capacity > 0 && capacity <= 4096, "invalid native TP wave capacity");
        Ok(Self {
            link: None,
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            output: DeviceAllocation::new(library, capacity as usize * 5120 * 2, Tag::workspace("v41/expert-wave"))?,
            library,
            ready_rows: None,
            local: None,
            tp2: None,
        })
    }

    /// Opens the lane's own link ([`V41Link::bootstrap`]).
    pub(crate) fn bootstrap(&mut self, nvfp4: bool) -> Result<()> {
        self.own_link()?.bootstrap(nvfp4)
    }
    /// Prefill slot 0 owns the one prefill link (its non-unit work sends on
    /// it). A units call borrows it for every slot's waves
    /// ([`Self::dispatch_unit`]) and returns it ([`Self::restore_link`]).
    pub(crate) fn take_link(&mut self) -> Result<V41Link<'a>> {
        self.ready_rows = None;
        self.link.take().context("prefill slot 0 has no link to lend")
    }
    pub(crate) fn restore_link(&mut self, link: V41Link<'a>) -> Result<()> {
        ensure!(self.link.is_none(), "prefill slot 0 already holds a link");
        self.link = Some(link);
        Ok(())
    }
    /// A prefill slot 0's FFN owner around the one prefill `link`.
    pub(crate) fn with_link(library: &'a NativeLibrary, link: V41Link<'a>) -> Result<Self> {
        let mut wave = Self::unlinked(library, link.capacity())?;
        wave.link = Some(link);
        Ok(wave)
    }

    /// Sends a remote layer's wave on the link's exchange: the router's route
    /// ids, gate weights and expert input rows copy into the wave's staging on
    /// the router stream (joined to the chain), and the exchange worker builds,
    /// posts and lands it. Nothing waits on the host. Finish with
    /// [`NativePendingFfn::finish`] (or its TP2 form).
    ///
    /// # Safety
    /// The router output was produced in stream order on its own stream (the
    /// chain head or `router_stream`) and stays unchanged until the wave's
    /// staging copies ran, which later work on `router_stream` orders.
    pub(crate) unsafe fn dispatch_ffn<'w>(&'w mut self,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        router_stream: *mut std::ffi::c_void, placement: u64,
        rows: &[crate::families::deepseek_v41::v41_backbone_router::ExpertRow]) -> Result<NativePendingFfn<'w, 'a>> {
        self.ready_rows = None;
        let link = self.link.as_mut().context("prefill-only slot has no Spark link of its own")?;
        // The Sparks skip this layer's tier experts exactly when the TP2 FFN
        // runs them (one set: `enable_tier`).
        let tiered = link.tier_holds(routed.layer);
        ensure!(tiered == self.tp2.as_ref().is_some_and(|tp2| tp2.tier_holds(routed.layer)),
            "layer {}: the Spark link and the TP2 FFN disagree on the GPU expert tier", routed.layer);
        let ticket = unsafe { send_on(self.library, &self.stream, link, routed, router_stream, placement, rows)? };
        Ok(NativePendingFfn { ticket: ticket.0, binding: ticket.1, layer: routed.layer, rows: routed.rows,
            library: self.library, stream: &self.stream, link: &mut link.link, output: self.output.buffer,
            ready_rows: &mut self.ready_rows, tp2: self.tp2.as_deref_mut(),
            tier: tiered.then_some([routed.expert_input, routed.ids, routed.routing]) })
    }
    /// A prefill unit's wave on the shared prefill `link` (its return slots
    /// hold the in-flight units): the returned ticket owns nothing borrowed, so
    /// the unit's later land hook collects it after other units' hooks ran
    /// ([`Self::collect_unit`]); this slot's reduce output and stream stay its own.
    ///
    /// # Safety
    /// As [`Self::dispatch_ffn`].
    pub(crate) unsafe fn dispatch_unit(&mut self, link: &mut V41Link<'a>,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        router_stream: *mut std::ffi::c_void, placement: u64,
        rows: &[crate::families::deepseek_v41::v41_backbone_router::ExpertRow]) -> Result<NativeUnitFfn> {
        self.ready_rows = None;
        // As a decode lane's: the Sparks skip the layer's tier experts
        // exactly when the slots' TP2 FFN runs them.
        let tiered = link.tier_holds(routed.layer);
        ensure!(tiered == self.tp2.as_ref().is_some_and(|tp2| tp2.tier_holds(routed.layer)),
            "layer {}: the prefill link and the TP2 FFN disagree on the GPU expert tier", routed.layer);
        let (ticket, binding) = unsafe { send_on(self.library, &self.stream, link, routed, router_stream, placement, rows)? };
        Ok(NativeUnitFfn { ticket, binding, layer: routed.layer, rows: routed.rows,
            owner: self.output.buffer.ptr as usize,
            tier: tiered.then_some([routed.expert_input, routed.ids, routed.routing]) })
    }
    /// Lands a unit's wave from the shared prefill `link` (waves land in
    /// dispatch order: the executor's FIFO) and reduces it with its `shared`
    /// values into this slot's output.
    ///
    /// # Safety
    /// `shared` holds the unit's completed shared contribution for exactly this
    /// wave's rows and stays unchanged until the reduce completes.
    pub(crate) async unsafe fn collect_unit<'w>(&'w mut self, link: &'w mut V41Link<'a>, unit: NativeUnitFfn,
        shared: UnitShared, destination: i32) -> Result<NativeFfnOutput<'w>> {
        ensure!(unit.owner == self.output.buffer.ptr as usize, "prefill unit ticket belongs to another slot");
        let rows = unit.rows as usize;
        ensure!(shared.values.device_id == self.output.buffer.device_id && shared.values.bytes >= rows * 10240
            && shared.tier.is_some() == unit.tier.is_some() && shared.tier.is_none_or(|tier|
                tier.device_id == self.output.buffer.device_id && tier.bytes >= rows * 10240),
            "prefill unit shared contribution differs from its wave");
        let mut values = shared.values;
        values.bytes = rows * 10240;
        let pending = NativePendingFfn { ticket: unit.ticket, binding: unit.binding, layer: unit.layer, rows: unit.rows,
            library: self.library, stream: &self.stream, link: &mut link.link, output: self.output.buffer,
            ready_rows: &mut self.ready_rows, tp2: self.tp2.as_deref_mut(), tier: unit.tier };
        let destination = Some(destination).filter(|&device| device != self.output.buffer.device_id);
        if destination.is_some() {
            let device = crate::shared::memory::device::Device { library: self.library, id: self.output.buffer.device_id };
            device.future(unsafe { pending.finish_values(values, destination, shared.tier) }).await
        } else {
            unsafe { pending.finish_values(values, None, shared.tier).await }
        }
    }
    /// The TP2 shared contribution of `unit` (a prefill unit's wave), and the
    /// GPU expert tier's result when it holds experts of the layer, into this
    /// slot's windows of the TP2 FFN's planes on the reduce GPU.
    /// # Safety
    /// The unit's normalized input and router outputs stay unchanged until it
    /// completes; the returned values stay unchanged until the unit lands.
    pub(crate) async unsafe fn unit_shared(&mut self, input: &crate::families::deepseek_v41::v41_block::FfnInput<'_>,
        unit: &NativeUnitFfn) -> Result<UnitShared> {
        ensure!(matches!(input.values.device_id, 0 | 1) && unit.layer == input.layer
            && unit.rows as usize == input.tokens.len(), "TP2 shared input device or unit differs");
        let destination = self.output.buffer.device_id as usize;
        let tp2 = self.tp2.as_mut().context("TP2 shared workspace missing")?;
        Ok(match unit.tier {
            Some([wire, ids, routing]) => {
                let (values, tier) = unsafe { tp2.execute_shared_and_tier_on(input.layer, unit.rows, input.values,
                    wire, ids, routing, destination).await? };
                UnitShared { values, tier: Some(tier) }
            }
            None => UnitShared { values: unsafe { tp2.execute_shared_on(input.layer, unit.rows, input.values,
                destination).await? }, tier: None },
        })
    }
    /// Borrowed device view; never free it or retain it across wave reuse/drop.
    pub fn output(&self) -> Result<CuteafdDeviceBuffer> {
        let rows = self
            .ready_rows
            .context("native TP wave output is not complete")?;
        let mut output = self.output.buffer;
        output.bytes = rows as usize * 5120 * 2;
        Ok(output)
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }
    }
}
/// Sends a remote layer's wave on `link`'s exchange (see
/// [`NativeTp4Wave::dispatch_ffn`]), staging on `stream` when the router shares
/// the link's GPU.
///
/// # Safety
/// As [`NativeTp4Wave::dispatch_ffn`].
unsafe fn send_on<'a>(library: &'a NativeLibrary, stream: &LoadStream<'a>, link: &mut V41Link<'a>,
    routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
    router_stream: *mut std::ffi::c_void, placement: u64,
    rows: &[crate::families::deepseek_v41::v41_backbone_router::ExpertRow])
    -> Result<(crate::shared::spark_waves::WaveTicket, crate::families::deepseek_v41::v41_attention_binding::QueryBinding)> {
    let capacity = link.link.capacity();
    ensure!(routed.rows > 0 && routed.rows <= capacity, "native dispatched FFN rows exceed capacity");
    let (binding, build) = routed.request_builder(placement, rows,
        link.resident.clone())?;
    let (layout, sources) = routed.wave_sources()?;
    // Stage on this wave's own stream when the router shares its GPU: the
    // lane's owners (the router among them) then carry nothing past their
    // per-layer host waits, which a host-ordered pass relies on when it
    // rebinds them (`LoadStream::require_complete`). A router on the other
    // GPU (TP2 shared, device-ordered passes) stages from its own stream.
    let stage = if sources[0].device_id == link.link.device() { stream.raw } else { router_stream };
    let exchange = link.link.exchange()?;
    // SAFETY: the sources are the router's live device outputs, complete in
    // chain order: `stage` joins the chain head, which the router's stage
    // finished (caller contract).
    let ticket = unsafe {
        crate::shared::memory::chain::join(library, stage)?;
        exchange.send(routed.rows as usize, exchange.compact_payload(routed.rows as usize), layout, sources, stage,
            crate::shared::spark_waves::combined_reply_eligible(rows.iter().map(|row|
                (row.request_id, row.kind == cuteafd_transport::ExpertV2SourceKind::Prefill))), build)?
    };
    Ok((ticket, binding))
}
impl Drop for NativeTp4Wave<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining native coordinator TP wave");
        }
    }
}

/// Complete ordered TP reduction plus the shared expert, borrowed until consumed.
pub(crate) struct NativeFfnOutput<'a> {
    pub values: CuteafdDeviceBuffer,
    binding: crate::families::deepseek_v41::v41_attention_binding::QueryBinding,
    _owner: std::marker::PhantomData<&'a ()>,
}
impl NativeFfnOutput<'_> {
    pub fn binding(&self) -> crate::families::deepseek_v41::v41_attention_binding::QueryBinding {
        self.binding
    }
}

/// A prefill unit's wave between its send and land hooks: owned, so no
/// transport borrow survives a hook. Dropping it unlanded leaves the wave
/// outstanding; the pass fails and resets the link after its streams drain.
pub(crate) struct NativeUnitFfn {
    ticket: crate::shared::spark_waves::WaveTicket,
    binding: crate::families::deepseek_v41::v41_attention_binding::QueryBinding,
    layer: usize,
    rows: u32,
    owner: usize,
    /// The router's FP8-K32 wire, ids and gate weights when the GPU expert
    /// tier holds experts of this layer (the request left them to the coordinator).
    tier: Option<[CuteafdDeviceBuffer; 3]>,
}

/// A prefill unit's coordinator contributions on the reduce GPU, from its
/// send ([`NativeTp4Wave::unit_shared`]) to its land: the TP2 shared values,
/// and the GPU expert tier's result when the tier holds experts of the layer.
#[derive(Clone, Copy)]
pub(crate) struct UnitShared {
    pub(crate) values: CuteafdDeviceBuffer,
    pub(crate) tier: Option<CuteafdDeviceBuffer>,
}

/// A remote layer's wave on the exchange, borrowed until its reduction is
/// queued. Dropping it unfinished leaves the wave unlanded: the owner's pass
/// fails and resets the link (stopping the exchange) after its streams drain.
pub(crate) struct NativePendingFfn<'w, 'a> {
    ticket: crate::shared::spark_waves::WaveTicket,
    binding: crate::families::deepseek_v41::v41_attention_binding::QueryBinding,
    layer: usize,
    rows: u32,
    library: &'a NativeLibrary,
    stream: &'w LoadStream<'a>,
    link: &'w mut crate::shared::spark_intake::SparkLink<'a>,
    output: CuteafdDeviceBuffer,
    ready_rows: &'w mut Option<u32>,
    tp2: Option<&'w mut super::tp2_ffn::Wave<'a>>,
    /// The router's FP8-K32 wire, ids and gate weights when the GPU expert
    /// tier holds experts of this layer (the request left them to the coordinator).
    tier: Option<[CuteafdDeviceBuffer; 3]>,
}
impl<'w> NativePendingFfn<'w, '_> {
    /// # Safety
    /// Shared values hold the completed contribution for this exact request and
    /// remain immutable until the final reduction completes on the stream.
    /// Every wait yields to the executor; cancellation drains before releasing frames.
    pub async unsafe fn finish(self,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'w>> {
        ensure!(self.binding == shared.binding()? && self.layer == shared.layer && self.rows == shared.rows
            && shared.values.bytes == self.rows as usize * 10240 && shared.values.device_id == self.output.device_id,
            "native TP shared contribution differs from routed request");
        ensure!(self.tier.is_none(), "the GPU expert tier runs only beside TP2 shared halves");
        unsafe { self.finish_values(shared.values, None, None).await }
    }
    /// Run shared TP2 after the Spark dispatch, reduce on the transport GPU, and
    /// return the result to the block's GPU. The pending owner retains every lane workspace.
    /// # Safety
    /// Input is the completed normalized FFN input for the dispatched request;
    /// its storage remains immutable through completion or cancellation drain.
    pub async unsafe fn finish_tp2(mut self, input: &crate::families::deepseek_v41::v41_block::FfnInput<'_>) -> Result<NativeFfnOutput<'w>> {
        ensure!(self.binding == input.binding() && self.layer == input.layer
            && self.rows as usize == input.tokens.len() && matches!(input.values.device_id, 0 | 1),
            "TP2 shared input/request/device differs");
        let tp2 = self.tp2.as_mut().context("TP2 shared workspace missing")?;
        let (values, tier) = match self.tier {
            // The tier's halves run beside the shared halves; its result
            // joins the Spark reduce on the reduce's GPU.
            Some([wire, ids, routing]) => {
                let (values, tier) = unsafe { tp2.execute_shared_and_tier_on(input.layer, self.rows, input.values,
                    wire, ids, routing, self.output.device_id as usize).await? };
                (values, Some(tier))
            }
            None => (unsafe { tp2.execute_shared_on(input.layer, self.rows, input.values,
                self.output.device_id as usize).await? }, None),
        };
        let destination = input.values.device_id;
        if destination != self.output.device_id {
            let device = crate::shared::memory::device::Device { library: self.library, id: self.output.device_id };
            device.future(unsafe { self.finish_values(values, Some(destination), tier) }).await
        } else {
            unsafe { self.finish_values(values, Some(destination), tier).await }
        }
    }
    async unsafe fn finish_values(mut self, values: CuteafdDeviceBuffer, destination: Option<i32>,
        tier: Option<CuteafdDeviceBuffer>) -> Result<NativeFfnOutput<'w>> {
        ensure!(values.device_id == self.output.device_id && values.bytes == self.rows as usize * 10240,
            "shared reduction device/extent differs");
        ensure!(tier.is_some() == self.tier.is_some() && tier.is_none_or(|tier| tier.device_id == self.output.device_id
            && tier.bytes >= self.rows as usize * 10240), "GPU expert tier result device/extent differs");
        let timing = std::time::Instant::now();
        let rows = self.rows;
        let slot = self.ticket.slot();
        let exchange = self.link.exchange()?;
        // A host-ordered pass (1 RTX) collects as before the exchange: the
        // host waits for the reply here, where everything this layer can
        // issue (router, staging, shared FFN, L2 prefetch) is already queued,
        // and only then queues the landing and reduce, so no stream sits
        // blocked on a device wait through the round trip. A paced
        // device-ordered pass (both dual lanes busy) collects the same way;
        // any other device-ordered pass queues them at once (the device waits).
        //
        // Every wait here yields to the executor rather than blocking the
        // thread (#815): prefill polls its two encoder lanes on this thread,
        // and a lane that spun or slept through its ~10 ms round trip kept
        // the other lane from issuing its next layer, so only one wave was
        // ever in flight and the Sparks idled half the time. A lone future
        // (decode) re-polls at once, so its wait is still a spin.
        let host_ordered = !crate::shared::memory::chain::deferred();
        if host_ordered || crate::shared::memory::chain::paced() {
            // The shared host-wait pacing still applies (`--host-wait block`
            // sleeps past its spin window), between yields. Pinned ranks that
            // arrived are pulled meanwhile (#815) on the reduce stream (after
            // the slot's previous reduce; no chain work touches the planes),
            // so their PCIe transfer overlaps the ranks still on the wire.
            let mut backoff = cuteafd_transport::wait_policy::Backoff::new();
            while !exchange.landed(&self.ticket) {
                crate::shared::spark_waves::healthy()?;
                exchange.pull_arrived(&self.ticket, self.stream.raw)?;
                backoff.pause();
                tokio::task::yield_now().await;
            }
            // The reply landed: progress for the pacing around the lane loop.
            cuteafd_transport::wait_policy::note_active();
        }
        // SAFETY: the reduce stream joins the chain (the shared contribution's
        // producers), then waits on the device for this wave's planes; the
        // intake planes, shared values and output stay live until the stream
        // completes (the pass drains before releasing them).
        unsafe {
            crate::shared::memory::chain::join(self.library, self.stream.raw)?;
            exchange.land(self.ticket, self.stream.raw)?;
            // The tier's result is the plane after the ranks': every route of
            // the layer ran once, on a Spark or on the coordinator GPUs.
            exchange.reduce_with(slot, tier.map(|tier| tier.ptr.cast_const().cast()), values.ptr.cast(),
                self.output.ptr.cast(), rows as usize, self.stream.raw)?;
        }
        // Inside a chain this records the head; unchained (prefill) it yields
        // until the reduce completes instead of synchronizing the thread.
        // SAFETY: the pending owner retains every input until completion or cancellation.
        unsafe { crate::shared::memory::chain::finish_cooperative(self.stream).await? };
        // The reduce read the TP2 shared output (when this lane computes it):
        // release it now that the reduce is on the chain head.
        if let Some(tp2) = self.tp2.as_deref() {
            tp2.release_shared_output(values.device_id as usize)?;
        }
        if host_ordered && crate::shared::memory::chain::active() {
            // The layer's one host wait for its streams: the reduce joined the
            // chain head (attention, router, shared FFN), so its completion is
            // what the next layer's rebinds check (`require_complete`).
            // SAFETY: this pending owner retains the reduction's queued storage.
            unsafe { crate::shared::memory::chain::finish_or_wait(self.stream, false).await?; }
        }
        tracing::debug!(target: "cuteafd::timing", layer = self.layer, rows,
            collect_us = timing.elapsed().as_micros() as u64, "target collection");
        *self.ready_rows = Some(rows);
        let mut values = self.output;
        values.bytes = rows as usize * 10240;
        if let Some(device) = destination.filter(|&device| device != values.device_id) {
            values = unsafe { self.tp2.as_mut().context("TP2 return workspace missing")?
                .return_result(values, device as usize, rows).await? };
        }
        Ok(NativeFfnOutput {
            values,
            binding: self.binding,
            _owner: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod coordinator_route_tests {
    use super::*;
    use cuteafd_transport::{expert::{BackboneRequest, CoordinatorRouteWord, V41_COORDINATOR_ROUTES_REQUEST_FLAG},
        ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind};

    fn router_request(layer: u32, rows: usize, residents: Option<&[CoordinatorExpertResidency]>)
        -> Result<ExpertProtocolV2Request> {
        crate::families::deepseek_v41::v41_backbone_router::build_spark_request(
            31, 7, layer, ExpertV2Dtype::Bf16,
            (0..rows).map(|r| ExpertProtocolV2RowDescriptor { row_id: r as u64,
                source_kind: ExpertV2SourceKind::Decode, source_request_id: 31,
                token_position: r as u64, route_offset: r as u32 * 6, route_count: 6 }).collect(),
            (0..rows * 6).map(|i| ExpertProtocolV2RouteEntry { row_index: i as u32 / 6,
                expert_id: (i % 384) as u32, gate_weight: (i % 6 + 1) as f32 / 21.0 }).collect(),
            vec![0; rows * 10240].into(), residents)
    }

    #[test]
    fn router_residents_and_worker_unpack_agree_route_for_route() -> Result<()> {
        let mut resident = vec![CoordinatorExpertResidency::empty(); 40];
        resident[3] = CoordinatorExpertResidency::from_experts(&[0, 5, 63, 64, 127, 200, 383])?;
        for rows in [1, 2, 64, 4096] {
            let original = router_request(3, rows, None)?;
            let request = router_request(3, rows, Some(&resident))?;
            let frame = request.encode()?;
            assert_eq!(frame.len(), original.encode()?.len());
            let worker = BackboneRequest::parse(&frame, rows as u32)?;
            let mut ids = vec![-1; rows * 6]; let mut weights = vec![-1.0; rows * 6];
            worker.copy_routes_into(&mut ids, &mut weights)?;
            for (i, route) in original.routes.iter().enumerate() {
                let marked = resident[3].contains(route.expert_id as u16);
                assert_eq!(CoordinatorRouteWord::decode(request.routes[i].expert_id, true)?.coordinator, marked);
                assert_eq!(ids[i], if marked { 384 } else { route.expert_id as i32 });
                assert_eq!(weights[i].to_bits(), if marked { 0.0f32.to_bits() } else { route.gate_weight.to_bits() });
            }
        }
        Ok(())
    }

    #[test]
    fn tier_flag_is_link_wide_and_startup_probe_matches_router() -> Result<()> {
        let resident = vec![CoordinatorExpertResidency::from_experts(&[0, 5])?];
        for layer in [0, 1, 39] {
            let request = router_request(layer, 1, Some(&resident))?;
            assert_ne!(request.header.flags & V41_COORDINATOR_ROUTES_REQUEST_FLAG, 0);
            let mask = coordinator_routes(&resident, layer, &[0, 1, 2, 3, 4, 5])?;
            for (i, route) in request.routes.iter().enumerate() {
                assert_eq!(mask[i], CoordinatorRouteWord::decode(route.expert_id, true)?.coordinator);
            }
        }
        assert_eq!(router_request(0, 1, None)?.header.flags & V41_COORDINATOR_ROUTES_REQUEST_FLAG, 0);
        Ok(())
    }

    #[test]
    fn marking_depends_only_on_layer_residency_and_expert() -> Result<()> {
        let resident = vec![CoordinatorExpertResidency::from_experts(&[0, 5, 383])?];
        for id in [1, 7, u64::MAX] {
            let mut request = router_request(0, 64, None)?;
            request.header.request_id = id;
            request.with_coordinator_routes(&resident[0])?;
            for route in &request.routes {
                let word = CoordinatorRouteWord::decode(route.expert_id, true)?;
                assert_eq!(word.coordinator, resident[0].contains(word.expert_id));
            }
        }
        Ok(())
    }
}
