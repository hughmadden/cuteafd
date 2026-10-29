//! Short queued prompts share one row unit, never attention history or compressor state.
use super::*;
use crate::shared::prefill_packing::{self, Limits, MemberOutcome, PackingMode, Segment};
use crate::shared::prefill_pipeline::{self, UnitSegment};

pub(super) struct PackedMember<'a> {
    pub segment: Segment,
    pub tables: StepTables,
    pub units: Vec<u32>,
    pub views: Vec<UnitSlot<'a>>,
}

impl<'a> UnitSlot<'a> {
    fn member(&self, segment: Segment, h: usize, taps: usize) -> Self {
        let at = |dev: &Dev<'a>, stride| dev.rows(segment.first, stride);
        // rows() selects one stride, so enlarge the bounded view to the member extent.
        let rows = |dev: &Dev<'a>, stride: usize| {
            let mut view = at(dev, stride);
            assert!((segment.first + segment.rows) * stride <= dev.buffer.bytes);
            view.buffer.bytes = segment.rows * stride;
            view
        };
        let mut tapped = self.taps.clone();
        if taps > 0 {
            tapped = self.taps.rows(segment.tap_offset, taps * h * 2);
            assert!((segment.tap_offset + segment.tap_rows) * taps * h * 2 <= self.taps.buffer.bytes);
            tapped.buffer.bytes = segment.tap_rows * taps * h * 2;
        }
        Self { index: self.index, stream_a: rows(&self.stream_a, 4 * h * 2),
            stream_b: rows(&self.stream_b, 4 * h * 2), post: rows(&self.post, 16), comb: rows(&self.comb, 64),
            tokens: rows(&self.tokens, 4), shared: rows(&self.shared, h * 2),
            payload: self.payload.as_ref().map(|p| rows(p, h * 4)), tables: self.tables.clone(), taps: tapped }
    }
}

impl<'a> Engine<'a> {
    pub(crate) fn configure_packing(&self, mode: PackingMode, links: &[SparkLink<'_>]) -> Result<()> {
        if mode == PackingMode::Unrestricted { self.packed_row_cap.set(None); return Ok(()); }
        let mut cap = self.prefill_rows;
        // These older interfaces do not attest arithmetic routes. Unknown means singleton,
        // not a guessed cutoff derived from transport grouping or package capacity.
        if !self.skip_routed && self.local_layers() > 0 { cap = 0; }
        if !self.skip_routed && self.local_layers() < self.cfg.n_layers {
            let reports = links.get(1).context("prefill Spark link")?.installed_grouping()?;
            for layer in self.local_layers()..self.cfg.n_layers {
                let mut found = false;
                for report in &reports {
                    if report.layers().iter().any(|r| (r.first as usize..r.end as usize).contains(&layer)) {
                        cap = cap.min(report.packed_row_cap(false));
                        found = true;
                    }
                }
                if !found { cap = 0; }
            }
        }
        self.packed_row_cap.set(Some(cap));
        prefill_packing::log_mode(mode, cap);
        Ok(())
    }

    pub(crate) fn packed_limits(&self) -> Limits {
        Limits { rows: self.packed_row_cap.get().unwrap_or(self.prefill_rows), chunk_rows: self.prefill_rows,
            max_context: self.max_context, taps: self.draft_ready().then_some(cuteafd_loader::serving_capacity::deepseek_v4_tap_rows(self.prefill_rows, true)) }
    }

    pub(crate) fn packed_plan(&self, chunks: &[(usize, usize)]) -> Result<Vec<Segment>> {
        ensure!(chunks.len() <= cuteafd_loader::serving_capacity::V4_PACKED_MEMBERS,
            "packed prefill exceeds admitted member count");
        prefill_packing::plan_with_tap_rows(chunks, &self.packed_limits(), TAP_ROWS)
    }

    pub(crate) fn prefill_packed(&self, sequences: &mut [(&mut Placement, &[u32])],
        links: &mut [SparkLink<'_>], prompt_ends: &[bool], on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome)
        -> Result<Vec<Result<()>>> {
        ensure!(prompt_ends.len() == sequences.len(), "packed prompt boundary count differs");
        let segments = self.packed_plan(&sequences.iter().map(|(p, t)| (p.len, t.len())).collect::<Vec<_>>())?;
        let links = RefCell::new(links);
        crate::shared::experts::rtx::pair::submit(|| self.require_live(),
            || self.shutdown_submission(&mut links.borrow_mut()), || {
                self.prepare_workspaces()?;
                self.begin_route_step();
                let owned = self.prefill_workspace.borrow();
                let w = owned.as_ref().context("prefill workspace")?;
                let peer_owned = self.peer.as_ref().map(|p| p.prefill_workspace.borrow());
                let w1 = peer_owned.as_ref().and_then(|w| w.as_ref());
                let members = sequences.iter().zip(&segments).map(|((p, tokens), &segment)| {
                    let tables = metadata::prefill_step(p, &self.shape, p.len, tokens.len(), self.cfg.index_topk, self.c128_width)?;
                    let views = std::iter::once(w).chain(w1).enumerate().map(|(rank, w)|
                        w.slots[0].member(segment, self.cfg.dim,
                            if rank == 0 && self.draft_ready() { self.cfg.dspark_target_layer_ids.len() } else { 0 }))
                        .collect();
                    Ok(PackedMember { segment, tables, units: p.units.clone(), views })
                }).collect::<Result<Vec<_>>>()?;
                let tokens = sequences.iter().flat_map(|(_, t)| t.iter().copied()).collect::<Vec<_>>();
                let mut units = prefill_pipeline::sequence_units(tokens.len(), self.prefill_rows, 1, self.cfg.n_layers)?;
                units[0].segments = segments.iter().enumerate().map(|(sequence, s)| UnitSegment {
                    sequence, rows: s.first..s.end_row(), tokens: s.start..s.start + s.rows,
                    prompt_end: prompt_ends[sequence], snapshot: false, head: true,
                }).collect();
                let tables = [members[0].tables.clone()];
                if self.context.is_some() { *self.context_units.borrow_mut() = vec![Default::default(); self.prefill_waves]; }
                self.prepare_route_checks(&[tokens.len()])?;
                let mut links = links.borrow_mut();
                let transport = links.get_mut(1).map(|link| link.exchange()).transpose()?;
                let mut hooks = super::prefill::UnitHooks { engine: self, placement: sequences[0].0,
                    tokens: &tokens, tables: &tables, w, w1, transport, logit_rows: 0, download: false,
                    on_layer: None, logits: Vec::new(), landed: 0, packed: Some(&members),
                    context_plans: vec![Default::default(); self.prefill_waves], staging_owner: [None; 2] };
                prefill_pipeline::run_prefill_units(&mut hooks, &units, 1)?;
                self.finish_route_step()?;
                self.context_units.borrow_mut().clear();
                for ((placement, _), s) in sequences.iter_mut().zip(&segments) { placement.len += s.rows; }
                prefill_packing::finish_members(members.len(), |i| {
                    let member = &members[i];
                    self.head_launch(&member.views[0].stream_a, member.segment.rows, 1, 0, w)?;
                    Ok(self.device_logits(w, 1))
                }, on_logits)
            })
    }

    pub(super) fn packed_attention(&self, layer: usize, members: &[PackedMember<'a>],
        w: &Workspace<'a>, w1: Option<&Workspace<'a>>) -> Result<()> {
        let total: usize = members.iter().map(|m| m.segment.rows).sum();
        if layer == 0 && w1.is_some() { self.exchange()?.wait(1, slot(0, false, 0))?; }
        for member in members {
            let rows = Scalar::I32(member.segment.rows as i32);
            for (rank, view) in member.views.iter().enumerate() {
                self.on(rank, || self.fill(&view.tables, &member.tables))?;
                if self.context.is_some() {
                    self.context_units.borrow_mut()[0][rank] = self.put_context_tables(rank, &view.tables,
                        &member.tables, 0, true, &member.units)?;
                }
            }
            self.attention_front(0, layer, &self.weights.layers[layer], &member.tables,
                &member.views[0], w, rows, self.prefill_rows)?;
            if let Some(w1) = w1 {
                self.attention_front(1, layer, &self.peer()?.layers[layer], &member.tables,
                    &member.views[1], w1, rows, self.prefill_rows)?;
                self.attention_sum(0, layer, 0, w, rows)?;
                self.attention_sum(1, layer, 0, w1, rows)?;
            }
            // shared is idle until the packed FFN. Preserve each attention result there
            // while subsequent members reuse query/delta/sum scratch.
            for (rank, work) in std::iter::once(w).chain(w1).enumerate() {
                let source = if w1.is_some() { &work.sum } else { &work.delta };
                self.on(rank, || {
                    // SAFETY: both views hold this member's rows; its owning stream orders reuse.
                    unsafe { self.library.copy_d2d_async(member.views[rank].shared.buffer, source.buffer,
                        member.segment.rows * self.cfg.dim * 2, self.stream_of(rank)) }
                })?;
            }
            if w1.is_some() {
                // Both sums must finish reading the peer receive planes before the next
                // member can overwrite them (publication alone is not a consumption fence).
                self.exchange()?.signal(0, DIRECT)?;
                self.exchange()?.signal(1, DIRECT)?;
                self.exchange()?.wait(0, DIRECT)?;
                self.exchange()?.wait(1, DIRECT)?;
            }
        }
        let rows = Scalar::I32(total as i32);
        for (rank, work) in std::iter::once(w).chain(w1).enumerate() {
            let lane = &work.slots[0];
            self.on(rank, || {
                // SAFETY: packed shared and delta are admitted total-row planes on this stream.
                unsafe { self.library.copy_d2d_async(work.delta.buffer, lane.shared.buffer,
                    total * self.cfg.dim * 2, self.stream_of(rank)) }
            })?;
            let weights = if rank == 0 { &self.weights.layers[layer] } else { &self.peer()?.layers[layer] };
            self.attention_back(rank, weights, lane, work, rows, self.prefill_rows, work.delta.buffer.ptr)?;
            if rank == 1 {
                self.run_on(1, weights.ffn_split, &format!("shared_ffn_m{}", self.prefill_kernel_rows), &[
                    ("x", work.y.buffer.ptr), ("w13", weights.ptr("w13")?), ("w13_scale", weights.ptr("w13_scale")?),
                    ("w2", weights.ptr("w2")?), ("w2_scale", weights.ptr("w2_scale")?),
                    ("out", lane.shared.buffer.ptr), ("scratch", work.scratch.buffer.ptr)], &[rows])?;
                if self.tp2_layer(layer) { self.tp2_experts(1, layer, 0, work, total)?; }
                else { self.exchange()?.push(1, slot(layer, true, 0), lane.shared.buffer.ptr, total * self.cfg.dim * 2)?; }
            }
        }
        Ok(())
    }
}
