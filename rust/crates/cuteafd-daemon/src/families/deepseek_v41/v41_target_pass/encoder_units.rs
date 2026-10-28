//! V4.1 encoder prefill on the shared executor (PLAN "Prefill contract"):
//! continuous row units on one FIFO of waves, unit k+1 one layer behind unit k
//! at every layer, units committed in order. Each in-flight slot owns one
//! target pass (lane, index, embedding, Engram gates, taps) and the Spark link
//! its waves land in; decode keeps its own lanes and links.
use super::*;
use crate::families::deepseek_v41::v41_backbone_cache::CacheLease;
use crate::families::deepseek_v41::v41_backbone_execution::{UnitPendingLayer, UnitPreparedLayer};
use crate::families::deepseek_v41::v41_requests::RequestTokens;
use crate::shared::prefill_pipeline::{run_prefill_units_async, AsyncPrefillHooks, FfnRoute, Unit};
use cuteafd_transport::ExpertV2SourceKind;
use std::cell::RefCell;
use crate::shared::memory::row_window::RowWindow;

/// Encoder layers 0..20 publish at attention; the boundary (source 20, the
/// ratio-two carry the next unit's boundary reads) publishes at the unit's end.
pub(super) const BOUNDARY: usize = 20;
/// One ordering event per encoder layer and the boundary, recorded after a unit's publication
/// of that layer. The next unit's attention at the layer waits on it on the
/// device: it reads the request's window ring and sources that publication
/// wrote, and its own publication overwrites that ring after its attention.
pub(super) struct PublishedLayers<'a> {
    pub(super) library: &'a cuteafd_ffi::NativeLibrary,
    pub(super) events: Vec<*mut std::ffi::c_void>,
    pub(super) recorded: Vec<bool>,
}
impl<'a> PublishedLayers<'a> {
    pub(super) fn new(library: &'a cuteafd_ffi::NativeLibrary) -> Result<Self> {
        let device = library.cuda_get_device()?;
        Self::on_devices(library, |_| Ok(device))
    }
    /// Each layer's event on the GPU that publishes it (its attention GPU; the
    /// boundary's is layer 20's): a chain records it on its head device.
    pub(super) fn on_devices(library: &'a cuteafd_ffi::NativeLibrary,
        device: impl Fn(usize) -> Result<i32>) -> Result<Self> {
        let previous = library.cuda_get_device()?;
        let mut events = Vec::with_capacity(BOUNDARY + 1);
        for layer in 0..=BOUNDARY {
            let created = device(layer).and_then(|gpu| { library.cuda_set_device(gpu)?; Ok(library.cuda_event_create_ordering()?) });
            match created {
                Ok(event) => events.push(event),
                Err(error) => {
                    // SAFETY: created above and never recorded.
                    for event in events { unsafe { let _ = library.cuda_event_destroy(event); } }
                    let _ = library.cuda_set_device(previous);
                    return Err(error);
                }
            }
        }
        library.cuda_set_device(previous)?;
        Ok(Self { library, events, recorded: vec![false; BOUNDARY + 1] })
    }
}
impl Drop for PublishedLayers<'_> {
    fn drop(&mut self) {
        for &event in &self.events {
            // SAFETY: owned here; the slots' chains drain before the hooks drop.
            if let Err(error) = unsafe { self.library.cuda_event_destroy(event) } {
                tracing::error!(%error, "destroying prefill unit publication event");
            }
        }
    }
}
impl PublishedLayers<'_> {
    /// Inside a chain scope: record this slot's publication of `layer`.
    pub(super) fn record(&mut self, layer: usize) -> Result<()> {
        // SAFETY: the slot's own event; the chain records on its head device.
        self.recorded[layer] = unsafe { crate::shared::memory::chain::record_into(self.library, self.events[layer])? };
        Ok(())
    }
    pub(super) fn event(&self, layer: usize) -> Option<*mut std::ffi::c_void> {
        self.recorded[layer].then(|| self.events[layer])
    }
}
/// Inside a chain scope: later stages follow `event` (another slot's publication).
pub(super) fn order_after(library: &cuteafd_ffi::NativeLibrary, event: Option<*mut std::ffi::c_void>) -> Result<()> {
    let Some(event) = event else { return Ok(()) };
    // SAFETY: a live event recorded by the previous unit's chain.
    unsafe { crate::shared::memory::chain::depend(library, event) }
}
/// The slot that ran the unit before `unit`, when it is another slot.
pub(super) fn previous_slot(unit: &Unit, waves: usize) -> Option<usize> {
    unit.index.checked_sub(1).map(|index| index % waves).filter(|&slot| slot != unit.slot)
}

/// A unit's rows in its slot's shared planes: slot `k` holds rows
/// `k * unit_rows ..` (`shared::memory::row_window`; every depth's slots are
/// views of the one reserve).
pub(super) fn unit_window(unit: &Unit, unit_rows: usize) -> Result<RowWindow> {
    let start = unit.slot * unit_rows;
    ensure!(unit.rows.len() <= unit_rows, "encoder unit wider than the call's unit");
    Ok(RowWindow::rows(start..start + unit.rows.len()))
}

pub(super) fn encoder_units(chunks: &[&[u32]]) -> Result<Vec<Unit>> {
    ensure!(!chunks.is_empty(), "empty encoder unit span");
    let mut start = 0;
    chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| {
            ensure!(
                !chunk.is_empty() && chunk.len() <= 4096,
                "encoder unit capacity differs"
            );
            let end = start + chunk.len();
            let unit = Unit {
                index,
                sequence: 0,
                rows: start..end,
                layers: 0..20,
                slot: 0,
                barrier: false,
            };
            start = end;
            Ok(unit)
        })
        .collect()
}

pub(in crate::families::deepseek_v41) fn decoder_unit(index: usize, rows: std::ops::Range<usize>) -> Result<Unit> {
    ensure!(!rows.is_empty() && rows.len() <= 128, "decoder replay suffix capacity differs");
    Ok(Unit { index, sequence: 0, rows, layers: 20..40, slot: 0, barrier: true })
}

struct UnitSlot<'s, 'w, 'a> {
    pass: &'s mut TargetPass<'w, 'a>,
    transport: &'s mut NativeTp4Wave<'a>,
    batch: Option<RequestBatch>,
    prepared: Option<UnitPreparedLayer>,
    started: Instant,
    /// Per encoder layer: recorded on the device after this slot's unit
    /// published that layer's window and source KV (its attention hook).
    published: PublishedLayers<'a>,
}
struct UnitHooks<'s, 'r, 'w, 'a> {
    slots: Vec<UnitSlot<'s, 'w, 'a>>,
    /// The call's unit: slot `k`'s window is rows `k * unit_rows` onward.
    unit_rows: usize,
    /// The one prefill transport: every unit's wave lands in its own return slot.
    prefill: &'s mut crate::families::deepseek_v41::v41_experts::coordinator::V41Link<'a>,
    requests: RefCell<&'r mut Requests<'a>>,
    lease: CacheLease,
    chunks: &'s [&'s [u32]],
    suffix: &'s mut EncoderSuffix<'a>,
    keep_running: &'s dyn Fn() -> bool,
    before_unit: &'s dyn Fn() -> Result<()>,
}
impl Drop for UnitHooks<'_, '_, '_, '_> {
    fn drop(&mut self) {
        // Device-ordered units leave queued work on every slot's chain (and
        // cross-slot waits on publication events): drain them all before any
        // batch, cache slot or event is released.
        for slot in &self.slots {
            crate::shared::decode_graph::fatal_drain(slot.pass.drain_chain(), "cancelled prefill unit chain");
        }
        for slot in &mut self.slots {
            if let Some(mut batch) = slot.batch.take() {
                self.requests.borrow_mut().revoke_batch(&mut batch);
                if let Err(error) = slot.pass.discard(&mut batch) {
                    tracing::error!(%error, "discarding cancelled encoder unit");
                }
            }
        }
    }
}
impl<'a> UnitHooks<'_, '_, '_, 'a> {
    /// The previous unit's publication event for `layer`, when that unit ran
    /// in another slot (one slot runs its units in order).
    fn previous_publication(&self, unit: &Unit, layer: usize) -> Option<*mut std::ffi::c_void> {
        self.slots[previous_slot(unit, self.slots.len())?].published.event(layer)
    }
    fn order_after(&self, unit: &Unit, event: Option<*mut std::ffi::c_void>) -> Result<()> {
        order_after(self.slots[unit.slot].pass.library()?, event)
    }
    fn record_publication(&mut self, unit: &Unit, layer: usize) -> Result<()> {
        self.slots[unit.slot].published.record(layer)
    }
    async fn begin_unit_inner(&mut self, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        ensure!(
            slot.batch.is_none() && slot.prepared.is_none(),
            "encoder unit slot occupied"
        );
        ensure!((self.keep_running)(), "client disconnected");
        (self.before_unit)()?;
        slot.pass.resources.upload.reset()?;
        slot.started = Instant::now();
        let chunk = self.chunks[unit.index];
        slot.batch = Some(
            self.requests
                .borrow_mut()
                .reserve_encoder(&[RequestTokens {
                    lease: self.lease,
                    tokens: chunk,
                    image_mask: None,
                    kind: ExpertV2SourceKind::Prefill,
                }])?,
        );
        // Start the next unit's Engram I/O now (#815): see the 2-RTX hook.
        self.requests.borrow_mut().prefetch_encoder(self.lease, &self.chunks[unit.index + 1..])?;
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        if tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG) {
            let positions = batch.cache()?.positions();
            tracing::debug!(target: "cuteafd::timing", unit = unit.index, slot = unit.slot,
                first = positions.first().copied(), last = positions.last().copied(), rows = positions.len(),
                "V4.1 encoder unit cut");
        }
        // The slot is at rest (drained in `begin_unit`): restart its owners,
        // then select this unit's rows in every shared row plane.
        slot.pass
            .resources
            .execution
            .restart_for(CacheStage::Encoder);
        slot.pass.resources.lane.restart()?;
        slot.pass.resources.index.restart()?;
        slot.pass.resources.taps.reset();
        slot.pass.select_window(unit_window(unit, self.unit_rows)?)?;
        slot.pass.state.begin()?;
        // SAFETY: the unit exclusively owns its batch and its rows of the
        // shared planes. Layer 0's query runs in its attention hook: the query
        // outputs are scratch shared by every slot.
        unsafe {
            self.requests.borrow_mut().prepare_unit_input(
                batch,
                &mut slot.pass.resources.embedding,
                &mut slot.pass.resources.lane,
            )?;
        }
        Ok(())
    }
    async fn attention_inner(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        ensure!((self.keep_running)(), "client disconnected");
        ensure!(slot.prepared.is_none(), "encoder attention still pending");
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        // SAFETY: the preceding hook has consumed the previous layer's output.
        unsafe {
            slot.pass
                .prepare_encoder_query(&self.requests, batch, layer)
                .await?;
        }
        // SAFETY: attention completes before its cache/index/lane owners can change.
        let prepared = unsafe {
            self.requests.borrow_mut().prepare_encoder_layer(
                batch,
                &mut slot.pass.resources.execution,
                &mut slot.pass.resources.lane,
                &mut slot.pass.resources.index,
            )?
        };
        slot.prepared = Some(prepared.into_unit().await?);
        // Every fork reader of the attention's shared scratch (query outputs,
        // sparse query/output, index selection) joined the chain head.
        slot.pass.resources.lane.release_scratch()?;
        slot.pass.resources.index.release_scratch()?;
        Ok(())
    }
    async fn local_ffn_inner(&mut self, _layer: usize, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let prepared = slot.prepared.take().context("encoder attention absent")?;
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: the original bound input remains in the exclusive unit slot.
        let completed = unsafe {
            prepared
                .execute(
                    &mut slot.pass.resources.lane,
                    slot.transport,
                    0,
                    batch.image_mask(),
                )
                .await?
        };
        // SAFETY: the transport result and original unit binding remain live.
        unsafe {
            slot.pass.resources.execution.complete_layer_cooperative(
                batch.cache()?,
                &mut slot.pass.resources.lane,
                completed,
            ).await?;
        }
        Ok(())
    }
    async fn send_experts_inner(&mut self, _layer: usize, unit: &Unit) -> Result<UnitPendingLayer> {
        let slot = &mut self.slots[unit.slot];
        let prepared = slot.prepared.take().context("encoder attention absent")?;
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: the slot is retained until the matching FIFO landing hook.
        unsafe {
            prepared
                .send(
                    &mut slot.pass.resources.lane,
                    slot.transport,
                    self.prefill,
                    0,
                    batch.image_mask(),
                )
                .await
        }
    }
    async fn land_experts_inner(
        &mut self,
        _layer: usize,
        unit: &Unit,
        ticket: UnitPendingLayer,
    ) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: no other unit can overwrite this original slot's shared values.
        let completed = unsafe {
            ticket
                .land(&mut slot.pass.resources.lane, slot.transport, self.prefill)
                .await?
        };
        // SAFETY: result and batch belong to the original prepared layer. The
        // mHC finish yields to the executor while it runs (#815).
        unsafe {
            slot.pass.resources.execution.complete_layer_cooperative(
                batch.cache()?,
                &mut slot.pass.resources.lane,
                completed,
            ).await?;
        }
        Ok(())
    }
    async fn end_unit_inner(&mut self, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        ensure!((self.keep_running)(), "client disconnected");
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        // SAFETY: the lane's output stays live until the unit's commit drains its chain.
        unsafe { self.suffix.capture_chained(&slot.pass.resources.lane.output()?, slot.pass.resources.lane.finish_stream())?; }
        slot.pass.resources.lane.advance()?;
        // SAFETY: encoder output is complete and this batch owns its source-20 proposal.
        unsafe {
            slot.pass.resources.lane.begin_prepared()?;
            self.requests.borrow_mut().publish_encoder_boundary(
                batch,
                &mut slot.pass.resources.execution,
                &slot.pass.resources.lane,
            )?;
        }
        slot.pass.state = State::Encoded(batch.cache()?.identity());
        Ok(())
    }
    async fn commit_unit_inner(&mut self, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        slot.pass.commit(
            &mut self.requests.borrow_mut(),
            batch,
            &[unit.rows.len() as u32],
        )?;
        slot.batch = None;
        crate::families::deepseek_v41::v41_native_serve::console::totals::prefill(unit.rows.len());
        crate::families::deepseek_v41::v41_native_serve::console::Prefill::done(
            crate::families::deepseek_v41::v41_native_serve::console::PrefillKind::Chunk,
            unit.slot,
            unit.index,
            self.chunks.len(),
            unit.rows.len(),
            slot.started,
        );
        Ok(())
    }
}
impl<'a> AsyncPrefillHooks for UnitHooks<'_, '_, '_, 'a> {
    type Ticket = UnitPendingLayer;
    fn route(&self, layer: usize) -> FfnRoute {
        if self.slots[0].transport.has_local_layer(layer) {
            FfnRoute::Local
        } else {
            FfnRoute::Spark
        }
    }
    // Every hook polls in its unit's device-ordered chain scope (#815): the
    // stages it queues order on the device and it waits on the host for none
    // of them, so the executor's next hook issues at once.
    async fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        // A slot's previous unit committed without a host drain; its queued
        // H2D copies may still read this pass's pinned staging (positions,
        // destinations, Engram rows), which this unit now rewrites. Drain
        // here, k - 1 units later, where that work has normally finished.
        let slot = &self.slots[unit.slot];
        slot.pass.drain_chain()?;
        crate::shared::spark_waves::healthy()?;
        let chain = slot.pass.unit_chain()?;
        chain.scope(self.begin_unit_inner(unit)).await
    }
    async fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        // Device-ordered (#815 option (i)): no host wait. The attention first
        // orders after the previous unit's publication of this layer (another
        // slot's chain): it reads that unit's window ring rows and sources, and
        // this unit's publication later overwrites the ring. Its source and
        // window publications are chained direct commits; the layer's event
        // records after them for the next unit.
        crate::shared::spark_waves::healthy()?;
        let chain = self.slots[unit.slot].pass.unit_chain()?;
        let event = self.previous_publication(unit, layer);
        chain.scope(async {
            self.order_after(unit, event)?;
            self.attention_inner(layer, unit).await?;
            self.record_publication(unit, layer)
        }).await
    }
    async fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let chain = self.slots[unit.slot].pass.unit_chain()?;
        chain.scope(self.local_ffn_inner(layer, unit)).await
    }
    async fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<Self::Ticket> {
        let chain = self.slots[unit.slot].pass.unit_chain()?;
        chain.scope(self.send_experts_inner(layer, unit)).await
    }
    async fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: Self::Ticket) -> Result<()> {
        let chain = self.slots[unit.slot].pass.unit_chain()?;
        chain.scope(self.land_experts_inner(layer, unit, ticket)).await
    }
    async fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        // Device-ordered (#815 option (i)): the suffix capture and the
        // boundary (source 20) publication chain behind the unit's last layer;
        // the unit's commit drains once.
        crate::shared::spark_waves::healthy()?;
        let chain = self.slots[unit.slot].pass.unit_chain()?;
        let event = self.previous_publication(unit, BOUNDARY);
        chain.scope(async {
            self.order_after(unit, event)?;
            self.end_unit_inner(unit).await?;
            self.record_publication(unit, BOUNDARY)
        }).await
    }
    async fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        // No host drain: every window and source of an encoder unit was
        // published on the device during its layers, so the commit is host
        // bookkeeping only. The slot drains before its next unit reuses the
        // pass, and every slot drains (with a Spark health check: a failed
        // wire releases its waits over garbage planes) before the prompt's
        // decoder replay reads the cache; a failure releases the request.
        crate::shared::spark_waves::healthy()?;
        let slot = &mut self.slots[unit.slot];
        if slot.pass.resources.lane.captures_routes() { slot.pass.drain_chain()?; }
        slot.pass.resources.lane.drain_route_ring()?;
        self.commit_unit_inner(unit).await
    }
}

impl<'w, 'a> TargetPass<'w, 'a> {
    /// Prepare a chunk's next query only when its execution lane is available.
    async unsafe fn prepare_encoder_query(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, layer: usize) -> Result<()> {
        if layer == 0 {
            // Prepared at begin (`Requests::prepare_unit_input`): the query here.
            unsafe { self.resources.lane.begin_prepared_cooperative().await?; }
            return Ok(());
        }
        self.resources.lane.advance()?;
        // The cooperative forms join the unit's chain (its previous layer's
        // mHC output is not complete on the host).
        if (2..14).contains(&layer) && !self.resources.upload.staged(1)
                    && self.resources.upload.can_enqueue(1)? {
            if let cuteafd_loader::EngramGatherPoll::Ready(lease) = requests.borrow().poll_engram_layer(batch, 1)? {
                self.resources.upload.enqueue_lease(lease, batch.engram_wave()?)?;
            }
        }
        if let Some(gate) = [1, 14].iter().position(|&l| l == layer) {
            let rows = if self.resources.upload.staged(gate) {
                self.resources.upload.consume(gate)?
            } else {
                batch.wait_engram_gather(gate, self.resources.engram_timeout).await?;
                    self.resources.upload.wait_enqueue(gate, self.resources.engram_timeout).await?;
                requests.borrow().poll_engram_rows(batch, &mut self.resources.upload, &self.resources.lane)?
                    .context("engram gather wait returned pending")?
            };
            // SAFETY: gate reads this unit's immutable window until release.
            unsafe { self.resources.lane.apply_engram_cooperative(&mut self.resources.gates[gate], &rows).await?; }
            self.resources.upload.release_scratch()?;
            if gate == 0 { batch.start_later_engram()?; }
        }
        unsafe { self.resources.lane.begin_prepared_cooperative().await?; }
        Ok(())
    }


    /// Encoder prefill of one prompt's `chunks` on the shared executor, one
    /// unit per slot in flight: slot k is `slots[k]`, a target pass with its
    /// Spark link (the two decode lanes' passes, then any further prefill slot).
    /// # Safety
    /// Every slot retains its independent lane/TP2/shared/output storage until
    /// completion or drained cancellation. Decode is idle during this span.
    pub(crate) async unsafe fn execute_encoder_units(
        slots: Vec<(&mut Self, &mut NativeTp4Wave<'a>)>, prefill: &mut crate::families::deepseek_v41::v41_experts::coordinator::V41Link<'a>, requests: &mut Requests<'a>,
        lease: CacheLease, chunks: &[&[u32]], suffix: &mut EncoderSuffix<'a>, keep_running: &dyn Fn() -> bool,
        before_unit: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        requests.prefetch_encoder(lease, chunks)?;
        let waves = slots.len();
        ensure!((1..=4).contains(&waves), "V4.1 prefill runs 1..=4 units in flight");
        let mut units = encoder_units(chunks)?;
        for unit in &mut units { unit.slot = unit.index % waves; }
        for layer in 0..20 {
            ensure!(slots.iter().all(|(_, transport)| transport.has_local_layer(layer) == slots[0].1.has_local_layer(layer)),
                "encoder slot expert placement differs");
        }
        let unit_rows = chunks.iter().map(|chunk| chunk.len()).max().unwrap_or(0);
        let mut hooks = UnitHooks {
            prefill,
            unit_rows,
            slots: slots.into_iter().map(|(pass, transport)| {
                pass.set_attention_decode(false);
                Ok(UnitSlot { published: PublishedLayers::new(pass.library()?)?, pass, transport,
                    batch: None, prepared: None, started: Instant::now() })
            }).collect::<Result<Vec<_>>>()?,
            requests: RefCell::new(requests), lease, chunks, suffix, keep_running, before_unit,
        };
        run_prefill_units_async(&mut hooks, &units, waves).await?;
        // Every unit's chained work completes before the decoder replay (or a
        // caller) reads what the units published.
        for slot in &hooks.slots { slot.pass.drain_chain()?; }
        crate::shared::spark_waves::healthy()?;
        // Non-unit work (replay, continuation, scoring) runs at rows from 0.
        for slot in &mut hooks.slots { slot.pass.select_window(RowWindow::NONE)?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefill_pipeline::{prefill_schedule, Step};

    #[test]
    fn suffix_replay_waits_for_every_encoder_commit() -> Result<()> {
        let full = vec![0; 4096];
        let tail = vec![0; 257];
        for waves in 1..=3 {
            let mut units = encoder_units(&[&full, &full, &full, &tail])?;
            for unit in &mut units { unit.slot = unit.index % waves; }
            let replay = units.len();
            units.push(decoder_unit(replay, 12417..12545)?);
            for mask in [0, (1u64 << 40) - 1, 0x5555555555, 0xaaaaaaaaaa, (1u64 << 20) - 1] {
                let routes = (0..40).map(|layer| if mask & (1 << layer) == 0 {
                    FfnRoute::Spark
                } else { FfnRoute::Local }).collect::<Vec<_>>();
                let schedule = prefill_schedule(&routes, &units, waves)?;
                let mut committed = 0;
                let mut pending = std::collections::VecDeque::new();
                let mut replay_layers = Vec::new();
                for step in schedule {
                    match step {
                        Step::Begin { unit } if unit == replay => {
                            assert_eq!(committed, replay, "replay entered before all encoder/source commits");
                            assert!(pending.is_empty());
                        }
                        Step::Attention { unit, layer } if unit == replay => replay_layers.push(layer),
                        Step::Send { unit, layer } => {
                            pending.push_back((unit, layer));
                            assert!(pending.len() <= waves);
                        }
                        Step::Land { unit, layer } => assert_eq!(pending.pop_front(), Some((unit, layer))),
                        Step::End { unit } if unit == replay => {
                            assert_eq!(replay_layers, (20..40).collect::<Vec<_>>());
                            assert!(pending.is_empty());
                        }
                        Step::Commit { unit } => {
                            assert_eq!(unit, committed);
                            committed += 1;
                        }
                        _ => {}
                    }
                }
                assert_eq!(committed, units.len());
            }
        }
        assert!(decoder_unit(0, 0..0).is_err());
        assert!(decoder_unit(0, 0..129).is_err());
        Ok(())
    }

    #[test]
    fn encoder_units_preserve_cuts_and_ced_layer_barrier() -> Result<()> {
        let first = vec![0; 4096];
        let tail = vec![0; 257];
        let units = encoder_units(&[&first, &first, &tail])?;
        assert_eq!(
            units.iter().map(|u| u.rows.clone()).collect::<Vec<_>>(),
            [0..4096, 4096..8192, 8192..8449]
        );
        assert!(units
            .iter()
            .all(|u| u.layers == (0..20) && u.slot == 0 && !u.barrier));
        for mask in [0, 1, (1u32 << 20) - 1, 0x55555, 0xaaaaa] {
            let routes = (0..20)
                .map(|layer| {
                    if mask & (1 << layer) != 0 {
                        FfnRoute::Local
                    } else {
                        FfnRoute::Spark
                    }
                })
                .collect::<Vec<_>>();
            let schedule = prefill_schedule(&routes, &units, 1)?;
            let mut committed = 0;
            let mut pending = None;
            for step in schedule {
                match step {
                    Step::Begin { unit } => assert_eq!(unit, committed),
                    Step::Send { unit, layer } => assert!(pending.replace((unit, layer)).is_none()),
                    Step::Land { unit, layer } => assert_eq!(pending.take(), Some((unit, layer))),
                    Step::Commit { unit } => {
                        assert_eq!(unit, committed);
                        committed += 1;
                    }
                    _ => {}
                }
            }
            assert_eq!(committed, units.len());
            assert!(pending.is_none());
        }
        assert!(encoder_units(&[]).is_err());
        assert!(encoder_units(&[&[]]).is_err());
        assert!(encoder_units(&[&vec![0; 4097]]).is_err());
        Ok(())
    }
}
