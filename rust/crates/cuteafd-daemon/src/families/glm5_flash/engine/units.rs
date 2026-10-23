//! Continuous prefill units on the shared executor. Decode retains its graph path.
use super::*;
use crate::shared::prefill_pipeline::{FfnRoute, PrefillHooks, Unit, run_prefill_units};

enum Ticket { Spark(SparkExpertWave), Mixed(layers::Ticket) }

struct UnitHooks<'e, 'a, 'w> {
    engine: &'e GlmfEngine<'a>,
    steps: &'w [(StepTables, &'w [u32])],
    ranked: Vec<[StepTables; 2]>,
    work: &'w [Workspace<'a>],
    peer: Option<&'w PeerWorkspaces<'w, 'a>>,
    released: Vec<Option<crate::shared::memory::device::Event<'a>>>,
    consumed: Vec<Vec<usize>>,
    waves: usize,
    total: usize,
    wanted: usize,
    device: bool,
    logits: Vec<f32>,
    result: Option<DeviceLogits>,
    media: Option<&'w cuteafd_engine::media::RequestMedia>,
}

impl<'a> UnitHooks<'_, 'a, '_> {
    fn rows(&self, unit: &Unit) -> usize { unit.rows.len() }
    fn peer(&self, unit: &Unit) -> Result<Option<&Workspace<'a>>> {
        self.peer.map(|p| p.get(unit.slot)).transpose()
    }
    fn post(&self, layer: usize, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let w = &self.work[unit.slot];
        let t = self.rows(unit);
        let rows = Scalar::I32(t as i32);
        // Publish rank 1's retained FFN partial only when this unit lands.
        if let Some(w1) = self.peer(unit)? {
            if e.layer_execution.is_none() && !e.tp2_layer(layer) {
                e.exchange_push(e.exchange()?, "main", 1, slot(layer, true, unit.slot),
                    GlmfEngine::peer_ffn_out(w1, &e.peer()?.layers[layer])?, t * e.cfg.hidden * 2)?;
            }
        }
        let out = e.meet_ffn(w, layer, unit.slot, e.weights.layers.len(), t)?;
        if let Some(w1) = self.peer(unit)? { e.peer_post(layer, unit.slot, w1, t, PREFILL_CAP)?; }
        match e.weights.layers.get(layer + 1) {
            Some(next) => e.post_pre_on(0, w, out, 1, next, "attn", "input_norm", rows, PREFILL_CAP)?,
            None => e.mhc_post_at(w, out, 1, &Span { first: 0, rows: t })?,
        }
        if let Some(drafter) = &e.drafter {
            let window = self.total.saturating_sub(crate::shared::speculation::dflash::TAP_ROWS);
            let from = unit.rows.start.max(window);
            if from < unit.rows.end {
                drafter.tap_streams_at(layer, w.streams[0].buffer.ptr, HC,
                    from - unit.rows.start, unit.rows.end - from, from - window)?;
            }
        }
        Ok(())
    }
}

impl PrefillHooks for UnitHooks<'_, '_, '_> {
    type Ticket = Ticket;
    fn route(&self, layer: usize) -> FfnRoute {
        if !self.engine.weights.layers[layer].dense
            && matches!(self.engine.routed_experts(layer), Some(Experts::Spark { .. })) {
            FfnRoute::Spark
        } else { FfnRoute::Local }
    }
    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let w = &self.work[unit.slot];
        if unit.index >= self.waves {
            if let Some(event) = &self.released[unit.slot] {
                // SAFETY: both streams and the per-slot event outlive this call.
                e.on(1, || unsafe { e.library.cuda_event_record(event.raw, e.stream_of(1)) })?;
                unsafe { e.library.cuda_stream_wait_event(e.stream, event.raw)?; }
            }
        }
        let (tables, tokens) = &self.steps[unit.index];
        let ranked = &self.ranked[unit.index];
        e.put_tables(w, &ranked[0])?;
        if let Some(w1) = self.peer(unit)? { e.peer_tables(w1, &ranked[1])?; }
        if e.context.is_some() {
            let start = tables.positions[0] as usize;
            let ids = placement_units(tables);
            for rank in 0..2 {
                let plan = e.context_prefill_tables(rank, unit.slot, start, start, unit.rows.len(), &ids)?;
                e.context_units.borrow_mut()[unit.index][rank] = plan;
            }
        }
        e.load_streams(w, tokens, false)?;
        e.inject_media(w, tables, self.media)?;
        if self.media.is_some_and(|m| !m.spans().is_empty()) { e.put(&w.ids, tokens)?; }
        if let Some(w1) = self.peer(unit)? {
            e.exchange_direct(w.streams[0].buffer.ptr, w1.streams[0].buffer.ptr,
                unit.rows.len() * HC * e.cfg.hidden * 2)?;
        }
        Ok(())
    }
    fn attention(&mut self, index: usize, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let w = &self.work[unit.slot];
        let layer = &e.weights.layers[index];
        let rows = Scalar::I32(unit.rows.len() as i32);
        e.context_unit.set(unit.index);
        if e.context.is_some() && layer.attention == GlmNextAttention::Mla {
            let parity = e.caches.mla_ordinal[index].context("MLA ordinal")? % 2;
            let mut owners = e.staging_owner.borrow_mut();
            e.context_reuse.set(unit.index > 0 && owners[parity] == Some((index, unit.index - 1)));
            owners[parity] = Some((index, unit.index));
        }
        if e.layer_execution.is_some() {
            return e.unit_attention([w, self.peer(unit)?.context("owned peer slot")?],
                &self.ranked[unit.index], unit, index, self.total, &mut self.consumed[unit.index]);
        }
        if let Some(w1) = self.peer(unit)? {
            e.peer_attention(index, unit.slot, w1, unit.rows.len(), PREFILL_CAP, &self.ranked[unit.index][1])?;
        }
        if index == 0 { e.pre(w, &w.streams[0], layer, rows)?; }
        e.attention(0, w, index, layer, rows, PREFILL_CAP, &self.ranked[unit.index][0], None)?;
        let attended = e.meet_attention(w, slot(index, false, unit.slot), unit.rows.len(), layer, PREFILL_CAP)?;
        e.post_pre_on(0, w, attended, 0, layer, "ffn", "post_norm", rows, PREFILL_CAP)
    }
    fn local_ffn(&mut self, index: usize, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let w = &self.work[unit.slot];
        let layer = &e.weights.layers[index];
        let rows = Scalar::I32(unit.rows.len() as i32);
        if e.layer_execution.is_some() {
            let work = [w, self.peer(unit)?.context("owned peer slot")?];
            let ticket = e.unit_ffn_begin(work, &self.ranked[unit.index], unit, index, self.total)?;
            e.unit_ffn_finish(work, &self.ranked[unit.index], unit, index, self.total, ticket)?;
            return self.post(index, unit);
        }
        if layer.dense {
            e.ffn(w, layer, e.cfg.dense_intermediate, PREFILL_CAP, w.delta.buffer.ptr, rows)?;
        } else {
            e.moe_front(w, index, layer, unit.rows.len(), rows, PREFILL_CAP,
                &[Span { first: 0, rows: unit.rows.len() }])?;
            let w1 = self.peer(unit)?;
            if e.tp2_layer(index) {
                e.tp2_pair(index, unit.slot, w, w1, unit.rows.len(), rows, PREFILL_CAP)?;
                e.tp2_publish(index, unit.slot, unit.rows.len(), w1)?;
            } else {
                e.moe_experts(w, index, layer, unit.rows.len(), rows, PREFILL_CAP, false, w1)?;
            }
        }
        self.post(index, unit)
    }
    fn send_experts(&mut self, index: usize, unit: &Unit) -> Result<Self::Ticket> {
        let e = self.engine;
        let w = &self.work[unit.slot];
        let layer = &e.weights.layers[index];
        let rows = Scalar::I32(unit.rows.len() as i32);
        if e.layer_execution.is_some() {
            return Ok(Ticket::Mixed(e.unit_ffn_begin([w, self.peer(unit)?.context("owned peer slot")?],
                &self.ranked[unit.index], unit, index, self.total)?));
        }
        e.moe_front(w, index, layer, unit.rows.len(), rows, PREFILL_CAP,
            &[Span { first: 0, rows: unit.rows.len() }])?;
        let Some(Experts::Spark { transports, .. }) = e.routed_experts(index) else {
            anyhow::bail!("Spark unit without its link");
        };
        e.spark_dispatch(w, index, unit.rows.len(), false, &mut transports.borrow_mut()[0], ||
            e.ffn(w, layer, e.cfg.moe_intermediate, PREFILL_CAP, w.shared.buffer.ptr, rows)).map(Ticket::Spark)
    }
    fn land_experts(&mut self, layer: usize, unit: &Unit, wave: Self::Ticket) -> Result<()> {
        match wave {
            Ticket::Spark(wave) => self.engine.spark_land_lane(&self.work[unit.slot], unit.rows.len(), 0, wave)?,
            Ticket::Mixed(ticket) => self.engine.unit_ffn_finish(
                [&self.work[unit.slot], self.peer(unit)?.context("owned peer slot")?],
                &self.ranked[unit.index], unit, layer, self.total, ticket)?,
        }
        self.post(layer, unit)
    }
    fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let w = &self.work[unit.slot];
        if let Some(w1) = self.peer(unit)? { e.replicate_terminal(w, w1, 0, unit.rows.len())?; }
        let wanted = unit.rows.end.saturating_sub((self.total - self.wanted).max(unit.rows.start));
        if wanted > 0 {
            e.layer_head(w, unit.rows.len(), wanted)?;
            if self.device {
                ensure!(wanted == self.wanted, "device logits must lie in the last unit");
                self.result = Some(e.device_logits(w, wanted, false));
            } else {
                self.logits.extend(e.download(&w.logits, wanted * e.cfg.vocab_size * 4)?
                    .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
            }
        }
        Ok(())
    }
    fn commit_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
}

impl<'a> GlmfEngine<'a> {
    pub(super) fn step_units_inner(&self, scope: &crate::shared::experts::rtx::pair::SubmissionScope<'_>,
        steps: &[(StepTables, &[u32])], logit_rows: usize, device: bool,
        media: Option<&cuteafd_engine::media::RequestMedia>) -> Result<Option<StepLogits>> {
        let waves = self.prefill_waves.min(steps.len()).max(1);
        let work = self.prefill_slots_of(0, waves)?;
        let peer = self.peer_workspaces(false, Some(waves))?;
        for rank in 0..self.ranks() {
            // SAFETY: one call-boundary drain, never a per-dispatch stream wait.
            self.on(rank, || unsafe { self.library.cuda_stream_synchronize(self.stream_of(rank)) })?;
        }
        let total: usize = steps.iter().map(|(t, _)| t.positions.len()).sum();
        ensure!(logit_rows <= total && (self.full_prefill_logits || logit_rows <= DECODE_ROWS),
            "prefill logits exceed admitted rows");
        let mut ranked = Vec::new();
        let mut units = Vec::new();
        let mut first = 0;
        for (index, (tables, tokens)) in steps.iter().enumerate() {
            ensure!(!tokens.is_empty() && tokens.len() == tables.positions.len() && tokens.len() <= self.prefill_rows,
                "invalid prefill unit rows");
            let mut pair = if self.context.is_some() {
                [self.rank_tables(tables, 0)?, self.rank_tables(tables, 1)?]
            } else { [tables.clone(), tables.clone()] };
            for t in &mut pair { t.lane = index % waves; }
            ranked.push(pair);
            units.push(Unit { index, sequence: 0, rows: first..first + tokens.len(),
                layers: 0..self.weights.layers.len(), slot: index % waves, barrier: false });
            first += tokens.len();
        }
        *self.context_units.borrow_mut() = vec![[context::ContextUnit::default(); 2]; steps.len()];
        *self.staging_owner.borrow_mut() = [None; 2];
        self.unit_prefill.set(true);
        let released = (0..waves).map(|_| self.peer.as_ref().map(|p|
            crate::shared::memory::device::Event::new(crate::shared::memory::device::Device {
                library: self.library, id: p.device })).transpose()).collect::<Result<_>>()?;
        // Route diagnostics retain per-slot staging until the call ends: do not alias it
        // across a slot reuse. Normal routing and the expert requests remain checked.
        if steps.len() <= waves {
            self.prepare_route_checks(&steps.iter().map(|(t, _)| t.positions.len()).collect::<Vec<_>>())?;
        }
        let mut hooks = UnitHooks { engine: self, steps, ranked, work: &work, peer: peer.as_ref(), released,
            consumed: (0..steps.len()).map(|_| vec![0; self.layer_execution.as_ref()
                .map_or(0, |execution| execution.plan.hops().len())]).collect(),
            waves, total, wanted: logit_rows, device, logits: Vec::new(), result: None, media };
        let outcome = run_prefill_units(&mut hooks, &units, waves);
        self.unit_prefill.set(false);
        self.context_unit.set(0);
        self.context_reuse.set(false);
        outcome?;
        if steps.len() <= waves { self.finish_route_checks()?; }
        if logit_rows == 0 { self.synchronize()?; return Ok(None); }
        Ok(Some(if device { StepLogits::Device(hooks.result.context("last-unit logits")?) }
            else { StepLogits::Host(hooks.logits) }))
    }
}
