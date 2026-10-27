//! Two-RTX V4.1 encoder prefill on the shared executor: the same hooks and
//! order as the single-RTX units (`super::super::encoder_units`), on placed
//! lanes. Each in-flight slot owns one distributed pass and its link.
use super::*;
use crate::families::deepseek_v41::v41_backbone_cache::CacheLease;
use crate::families::deepseek_v41::v41_backbone_execution::{UnitPendingLayer, UnitPreparedLayer};
use crate::families::deepseek_v41::v41_requests::RequestTokens;
use crate::shared::prefill_pipeline::{run_prefill_units_async, AsyncPrefillHooks, FfnRoute, Unit};
use cuteafd_transport::ExpertV2SourceKind;
use std::cell::RefCell;

impl<'w, 'a> DistributedTargetPass<'w, 'a> {
    /// The per-layer preparation `execute_inner` runs before the FFN, for a
    /// reserved encoder unit: the lane advance (with its block hop), Engram at
    /// layers 1 and 14, cache production and the attention, finished.
    /// # Safety
    /// The unit's batch and this pass's owners stay live until the layer lands.
    async unsafe fn prepare_unit_layer(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, layer: usize) -> Result<UnitPreparedLayer> {
        let gpu = self.resources.map.attention(layer)?;
        let device = self.resources.lanes[gpu].device;
        if layer != 0 {
            unsafe { self.advance(layer).await?; }
            if [1, 14].contains(&layer) {
                let gate = usize::from(layer == 14);
                batch.wait_engram_gather(gate, self.resources.timeout).await?;
                let gathered = requests.with_requests(|r| r.poll_engram_gather(batch, &self.resources.lanes[gpu]))?;
                match gathered {
                    cuteafd_loader::EngramGatherPoll::Ready(lease) => unsafe {
                        self.resources.engram.apply(&mut self.resources.lanes[gpu], &lease.view()?).await?;
                    },
                    cuteafd_loader::EngramGatherPoll::Cancelled => anyhow::bail!("distributed Engram gather cancelled"),
                    cuteafd_loader::EngramGatherPoll::Pending => anyhow::bail!("engram gather wait returned pending"),
                }
            }
            device.future(unsafe { self.resources.lanes[gpu].get_mut().begin_prepared_cooperative() }).await?;
        }
        let index = if [2, 8, 14, 20, 24, 28, 32, 36].contains(&layer) { self.resources.indices[gpu].as_mut() } else { None };
        let mut production = unsafe { requests.with_requests(|r| self.resources.execution.enqueue_production(
            r.cache(), batch.cache()?, &self.resources.lanes[gpu], index))? };
        while !unsafe { requests.borrow_mut().poll_distributed_encoder_production(batch, &mut production)? } {
            tokio::task::yield_now().await;
        }
        drop(production);
        let prepared = unsafe { requests.with_requests(|r| self.resources.execution.prepare_layer(r.cache(),
            batch.cache()?, &mut self.resources.lanes[gpu], self.resources.indices[gpu].as_ref().map(DeviceOwner::get)))? };
        prepared.into_unit().await
    }

    /// Completes a unit's layer from its FFN result and publishes its KV.
    /// # Safety
    /// `completed` is this unit's layer result; the batch is reserved.
    async unsafe fn complete_unit_layer(&mut self, requests: &RefCell<&mut Requests<'a>>, batch: &RequestBatch,
        layer: usize, completed: crate::families::deepseek_v41::v41_backbone_execution::CompletedLayer<'_>) -> Result<()> {
        let gpu = self.resources.map.attention(layer)?;
        unsafe { self.resources.execution.complete_layer(batch.cache()?, &mut self.resources.lanes[gpu], completed).await?; }
        unsafe { self.publish_encoder_layer(requests, batch, layer).await }
    }
}

struct UnitSlot<'s, 'w, 'a> {
    pass: &'s mut DistributedTargetPass<'w, 'a>,
    transport: &'s mut DeviceOwner<'a, NativeTp4Wave<'a>>,
    batch: Option<RequestBatch>,
    prepared: Option<UnitPreparedLayer>,
    started: Instant,
}
struct UnitHooks<'s, 'r, 'w, 'a> {
    slots: Vec<UnitSlot<'s, 'w, 'a>>,
    requests: RefCell<&'r mut Requests<'a>>,
    lease: CacheLease,
    chunks: &'s [&'s [u32]],
    suffix: &'s mut DeviceOwner<'a, EncoderSuffix<'a>>,
    keep_running: &'s dyn Fn() -> bool,
    before_unit: &'s dyn Fn() -> Result<()>,
    /// Every unit committed: the request stays admitted.
    complete: bool,
}
impl Drop for UnitHooks<'_, '_, '_, '_> {
    fn drop(&mut self) {
        // A failed or cancelled prefill revokes the whole request (its later
        // units' reservations included) after each slot's work drained.
        let mut requests = self.requests.borrow_mut();
        for slot in &mut self.slots {
            if let Some(mut batch) = slot.batch.take() {
                if let Err(error) = slot.pass.abort_cache_commit(&mut requests) {
                    tracing::error!(%error, "draining cancelled distributed encoder unit");
                }
                requests.revoke_batch(&mut batch);
                if let Err(error) = slot.pass.discard(&mut batch) {
                    tracing::error!(%error, "discarding cancelled distributed encoder unit");
                }
            }
        }
        if !self.complete && requests.cache().request_id(self.lease).is_ok() {
            if let Err(error) = requests.release(self.lease) {
                tracing::error!(%error, "revoking cancelled distributed encoder units");
            }
        }
    }
}
impl<'a> AsyncPrefillHooks for UnitHooks<'_, '_, '_, 'a> {
    type Ticket = UnitPendingLayer;
    fn route(&self, layer: usize) -> FfnRoute {
        if self.slots[0].transport.has_local_layer(layer) { FfnRoute::Local } else { FfnRoute::Spark }
    }
    async fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        ensure!((self.keep_running)(), "client disconnected");
        (self.before_unit)()?;
        let slot = &mut self.slots[unit.slot];
        ensure!(slot.batch.is_none() && slot.prepared.is_none(), "encoder unit slot occupied");
        slot.started = Instant::now();
        let chunk = self.chunks[unit.index];
        let batch = self.requests.borrow_mut().reserve_encoder(&[RequestTokens { lease: self.lease, tokens: chunk,
            image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
        let batch = slot.batch.insert(batch);
        slot.pass.state.begin()?;
        // SAFETY: the unit exclusively owns its batch and the pass's input storage.
        unsafe { slot.pass.begin_stage(&self.requests, batch, CacheStage::Encoder, None).await }
    }
    async fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        ensure!((self.keep_running)(), "client disconnected");
        let slot = &mut self.slots[unit.slot];
        ensure!(slot.prepared.is_none(), "encoder attention still pending");
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        // SAFETY: the preceding hook consumed the previous layer's output.
        slot.prepared = Some(unsafe { slot.pass.prepare_unit_layer(&self.requests, batch, layer).await? });
        Ok(())
    }
    async fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let prepared = slot.prepared.take().context("encoder attention absent")?;
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        let gpu = slot.pass.resources.map.attention(layer)?;
        let device = slot.pass.resources.lanes[gpu].device;
        // SAFETY: the bound input stays in this unit's slot through the FFN.
        let completed = device.future(unsafe { prepared.execute(slot.pass.resources.lanes[gpu].get_mut(),
            slot.transport.get_mut(), 0, batch.image_mask()) }).await?;
        unsafe { slot.pass.complete_unit_layer(&self.requests, batch, layer, completed).await }
    }
    async fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<Self::Ticket> {
        let slot = &mut self.slots[unit.slot];
        let prepared = slot.prepared.take().context("encoder attention absent")?;
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        let gpu = slot.pass.resources.map.attention(layer)?;
        let device = slot.pass.resources.lanes[gpu].device;
        // SAFETY: the slot is retained until the matching FIFO landing hook.
        device.future(unsafe { prepared.send(slot.pass.resources.lanes[gpu].get_mut(), slot.transport.get_mut(),
            0, batch.image_mask()) }).await
    }
    async fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: Self::Ticket) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        let gpu = slot.pass.resources.map.attention(layer)?;
        let device = slot.pass.resources.lanes[gpu].device;
        // SAFETY: no other unit overwrote this slot's shared values or input.
        // As `PlacedPreparedLayer::execute`: on the lane's GPU; the collect
        // switches to the transport's GPU and returns the result here.
        let completed = device.future(unsafe { ticket.land(slot.pass.resources.lanes[gpu].get_mut(),
            slot.transport.get_mut()) }).await?;
        unsafe { slot.pass.complete_unit_layer(&self.requests, batch, layer, completed).await }
    }
    async fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        ensure!((self.keep_running)(), "client disconnected");
        let slot = &mut self.slots[unit.slot];
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        // SAFETY: the unit's encoder output is complete; units end in order.
        unsafe { slot.pass.finish_encoder_unit(&self.requests, batch, self.suffix).await }
    }
    async fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        let rows = [unit.rows.len() as u32];
        slot.pass.enqueue_cache_commit(&self.requests.borrow(), batch, &rows)?;
        while !slot.pass.poll_cache_commit()? { tokio::task::yield_now().await; }
        slot.pass.commit(&mut self.requests.borrow_mut(), batch, &rows)?;
        slot.batch = None;
        crate::families::deepseek_v41::v41_native_serve::console::totals::prefill(unit.rows.len());
        crate::families::deepseek_v41::v41_native_serve::console::Prefill::done(
            crate::families::deepseek_v41::v41_native_serve::console::PrefillKind::Chunk,
            unit.slot, unit.index, self.chunks.len(), unit.rows.len(), slot.started);
        Ok(())
    }
}

impl<'w, 'a> DistributedTargetPass<'w, 'a> {
    /// Encoder prefill of `chunks` (one prompt) as row units on the shared
    /// executor, `waves` in flight: slot k is `[self, other][k]` with its link.
    /// # Safety
    /// Passes and transports have independent mutable storage and the same model,
    /// placement and capacity. Poll on one CUDA-owning executor thread.
    pub(crate) async unsafe fn execute_encoder_units(&mut self, other: &mut Self, requests: &mut Requests<'a>,
        lease: CacheLease, chunks: &[&[u32]], transports: [&mut DeviceOwner<'a, NativeTp4Wave<'a>>; 2],
        suffix: &mut DeviceOwner<'a, EncoderSuffix<'a>>, keep_running: &dyn Fn() -> bool,
        before_unit: &dyn Fn() -> Result<()>, waves: usize) -> Result<()> {
        ensure!((1..=2).contains(&waves), "encoder unit owner count differs");
        ensure!(self.state == State::Idle && other.state == State::Idle, "distributed encoder lane still in use");
        ensure!(self.resources.map == other.resources.map, "distributed encoder lane placement differs");
        self.set_attention_decode(false);
        other.set_attention_decode(false);
        let mut units = super::super::encoder_units::encoder_units(chunks)?;
        for unit in &mut units { unit.slot = unit.index % waves; }
        let [first, second] = transports;
        for layer in 0..20 {
            ensure!(first.has_local_layer(layer) == second.has_local_layer(layer), "encoder slot expert placement differs");
        }
        let mut hooks = UnitHooks {
            slots: vec![
                UnitSlot { pass: self, transport: first, batch: None, prepared: None, started: Instant::now() },
                UnitSlot { pass: other, transport: second, batch: None, prepared: None, started: Instant::now() },
            ],
            requests: RefCell::new(requests), lease, chunks, suffix, keep_running, before_unit, complete: false,
        };
        run_prefill_units_async(&mut hooks, &units, waves).await?;
        hooks.complete = true;
        Ok(())
    }
}
