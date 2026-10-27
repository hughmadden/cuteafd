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
}
struct UnitHooks<'s, 'r, 'w, 'a> {
    slots: Vec<UnitSlot<'s, 'w, 'a>>,
    requests: RefCell<&'r mut Requests<'a>>,
    lease: CacheLease,
    chunks: &'s [&'s [u32]],
    suffix: &'s mut EncoderSuffix<'a>,
    keep_running: &'s dyn Fn() -> bool,
    before_unit: &'s dyn Fn() -> Result<()>,
}
impl Drop for UnitHooks<'_, '_, '_, '_> {
    fn drop(&mut self) {
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
impl<'a> AsyncPrefillHooks for UnitHooks<'_, '_, '_, 'a> {
    type Ticket = UnitPendingLayer;
    fn route(&self, layer: usize) -> FfnRoute {
        if self.slots[0].transport.has_local_layer(layer) {
            FfnRoute::Local
        } else {
            FfnRoute::Spark
        }
    }
    async fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        ensure!(
            slot.batch.is_none() && slot.prepared.is_none(),
            "encoder unit slot occupied"
        );
        ensure!((self.keep_running)(), "client disconnected");
        (self.before_unit)()?;
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
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        if tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG) {
            let positions = batch.cache()?.positions();
            tracing::debug!(target: "cuteafd::timing", unit = unit.index, slot = unit.slot,
                first = positions.first().copied(), last = positions.last().copied(), rows = positions.len(),
                "V4.1 encoder unit cut");
        }
        slot.pass.state.begin()?;
        slot.pass
            .resources
            .execution
            .restart_for(CacheStage::Encoder);
        slot.pass.resources.lane.restart()?;
        slot.pass.resources.index.restart()?;
        slot.pass.resources.taps.reset();
        // SAFETY: the unit exclusively owns its batch and the pass's input storage.
        unsafe {
            self.requests.borrow_mut().begin_input(
                batch,
                &mut slot.pass.resources.embedding,
                &mut slot.pass.resources.lane,
            )?;
        }
        Ok(())
    }
    async fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
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
        Ok(())
    }
    async fn local_ffn(&mut self, _layer: usize, unit: &Unit) -> Result<()> {
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
            slot.pass.resources.execution.complete_layer(
                batch.cache()?,
                &mut slot.pass.resources.lane,
                completed,
            )?;
        }
        Ok(())
    }
    async fn send_experts(&mut self, _layer: usize, unit: &Unit) -> Result<Self::Ticket> {
        let slot = &mut self.slots[unit.slot];
        let prepared = slot.prepared.take().context("encoder attention absent")?;
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: the slot is retained until the matching FIFO landing hook.
        unsafe {
            prepared
                .send(
                    &mut slot.pass.resources.lane,
                    slot.transport,
                    0,
                    batch.image_mask(),
                )
                .await
        }
    }
    async fn land_experts(
        &mut self,
        _layer: usize,
        unit: &Unit,
        ticket: Self::Ticket,
    ) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: no other unit can overwrite this original slot's shared values.
        let completed = unsafe {
            ticket
                .land(&mut slot.pass.resources.lane, slot.transport)
                .await?
        };
        // SAFETY: result and batch belong to the original prepared layer.
        unsafe {
            slot.pass.resources.execution.complete_layer(
                batch.cache()?,
                &mut slot.pass.resources.lane,
                completed,
            )?;
        }
        Ok(())
    }
    async fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        ensure!((self.keep_running)(), "client disconnected");
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        self.suffix.capture(&slot.pass.resources.lane.output()?)?;
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
    async fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
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
impl<'w, 'a> TargetPass<'w, 'a> {
    /// Prepare a chunk's next query only when its execution lane is available.
    async unsafe fn prepare_encoder_query(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, layer: usize) -> Result<()> {
        if layer == 0 { return Ok(()); }
        self.resources.lane.advance()?;
        if let Some(gate) = [1, 14].iter().position(|&l| l == layer) {
            batch.wait_engram_gather(gate, self.resources.engram_timeout).await?;
            ensure!(unsafe { requests.borrow().poll_engram(batch, &mut self.resources.upload,
                &mut self.resources.gates[gate], &mut self.resources.lane)? },
                "engram gather wait returned pending");
        }
        unsafe { self.resources.lane.begin_prepared()?; }
        Ok(())
    }


    /// # Safety
    /// Both slots retain their independent lane/TP2/shared/output storage until
    /// completion or drained cancellation. Decode is idle during this span.
    pub(crate) async unsafe fn execute_encoder_units(
        &mut self, other: &mut Self, requests: &mut Requests<'a>, lease: CacheLease,
        chunks: &[&[u32]], transports: [&mut NativeTp4Wave<'a>; 2],
        suffix: &mut EncoderSuffix<'a>, keep_running: &dyn Fn() -> bool,
        before_unit: &dyn Fn() -> Result<()>, waves: usize,
    ) -> Result<()> {
        ensure!((1..=2).contains(&waves), "encoder unit owner count differs");
        self.set_attention_decode(false);
        other.set_attention_decode(false);
        let mut units = encoder_units(chunks)?;
        for unit in &mut units { unit.slot = unit.index % waves; }
        let [first, second] = transports;
        for layer in 0..20 {
            ensure!(first.has_local_layer(layer) == second.has_local_layer(layer),
                "encoder slot expert placement differs");
        }
        let mut hooks = UnitHooks {
            slots: vec![
                UnitSlot { pass: self, transport: first, batch: None, prepared: None, started: Instant::now() },
                UnitSlot { pass: other, transport: second, batch: None, prepared: None, started: Instant::now() },
            ],
            requests: RefCell::new(requests), lease, chunks, suffix, keep_running, before_unit,
        };
        run_prefill_units_async(&mut hooks, &units, waves).await
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
