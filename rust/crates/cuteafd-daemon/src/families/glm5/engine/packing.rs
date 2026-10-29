//! Shared short-prompt layout and segment-local consumers on ordinary prefill units.
use super::*;
use crate::shared::prefill_packing::{self, Limits, MemberOutcome, PackingMode, Segment};

impl<'a> GlmEngine<'a> {
    pub(crate) fn configure_packing(&self, mode: PackingMode) -> Result<()> {
        if mode == PackingMode::Unrestricted { self.packed_row_cap.set(None); return Ok(()); }
        let mut cap = self.prefill_rows;
        // TP2's erased backend does not publish arithmetic-route bounds yet.
        // Exact validation fails closed; unrestricted serving still packs.
        if self.tp2.is_some() { cap = 0; }
        let reports = self.prefill_link.borrow().as_ref().map(|link| link.installed_grouping()).transpose()?
            .unwrap_or_default();
        for (index, layer) in self.weights.layers.iter().enumerate() {
            if layer.dense || self.tp2_layer(index) || self.skip.is_some() { continue; }
            let mut found = false;
            for report in &reports {
                if report.layers().iter().any(|range| (range.first as usize..range.end as usize).contains(&index)) {
                    cap = cap.min(report.packed_row_cap(self.spark_bf16));
                    found = true;
                }
            }
            if !found { cap = 0; }
        }
        self.packed_row_cap.set(Some(cap));
        prefill_packing::log_mode(mode, cap);
        Ok(())
    }

    pub(crate) fn packed_limits(&self) -> Limits {
        Limits { rows: self.packed_row_cap.get().unwrap_or(self.prefill_rows), chunk_rows: self.prefill_rows,
            max_context: self.max_context,
            taps: self.drafter.as_ref().map(|_| crate::shared::speculation::dflash::TAP_ROWS) }
    }

    /// Consume every head before shared scratch is reused for the next member.
    /// All placements advance after device success, including cancelled clients.
    pub(crate) fn prefill_packed(&self, sequences: &mut [(&mut GlmPlacement, &[u32])],
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome)
        -> Result<(Vec<Segment>, Vec<Result<()>>)> {
        self.prefill_packed_selected(sequences, &mut |_| true, on_logits)
    }

    pub(crate) fn prefill_packed_selected(&self, sequences: &mut [(&mut GlmPlacement, &[u32])],
        needs_head: &mut dyn FnMut(usize) -> bool,
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome)
        -> Result<(Vec<Segment>, Vec<Result<()>>)> {
        let chunks = sequences.iter().map(|(p, tokens)| (p.len, tokens.len())).collect::<Vec<_>>();
        let segments = prefill_packing::plan(&chunks, &self.packed_limits())?;
        let tables = packed_tables(sequences, &segments, self.cfg.index_topk)?;
        let tokens = sequences.iter().flat_map(|(_, tokens)| tokens.iter().copied()).collect::<Vec<_>>();
        if self.prefill_link.borrow().is_some() {
            let mut link = self.prefill_link.borrow_mut();
            let link = link.as_mut().context("packed prefill link")?;
            let runtime = self.prefill_runtime.as_ref().context("packed prefill runtime")?;
            // One retained unit keeps segment heads/taps in slot 0. Multi-wave
            // slicing needs explicit per-segment slot remapping (#847 follow-up).
            self.step_lanes(&[tables], &[&tokens], self.depth, 0, link, runtime)?;
        } else {
            self.step(&tables, &tokens, 0, None, None)?;
        }
        for (placement, tokens) in sequences.iter_mut() { placement.len += tokens.len(); }
        let slots = self.slot_workspaces(1)?;
        let w = &slots[0];
        let outcomes = prefill_packing::finish_members_selected(sequences.len(), needs_head, |i| {
            self.launch_head(w, segments[i].end_row(), 1)?;
            Ok(self.device_logits(w, 1, false))
        }, on_logits)?;
        Ok((segments, outcomes))
    }

    /// Golden-check consumer: score retained rows without widening serving's head storage.
    pub(crate) fn packed_check_logits(&self, segment: Segment,
        consume: &mut dyn FnMut(usize, DeviceLogits) -> Result<()>) -> Result<()> {
        let slots = self.prefill_slots.borrow();
        let w = slots.first().context("packed-check prefill slot")?;
        for first in (0..segment.rows).step_by(DECODE_ROWS) {
            let rows = (segment.rows - first).min(DECODE_ROWS);
            self.launch_head(w, segment.first + first + rows, rows)?;
            consume(first, self.device_logits(w, rows, false))?;
        }
        Ok(())
    }
}

pub(super) fn sequence_tables(placement: &GlmPlacement, start: usize, rows: usize, topk: usize) -> Result<StepTables> {
    let used = (start + rows).div_ceil(PAGE_ROWS);
    ensure!(used <= placement.pages.len(), "prefill past sequence pages");
    Ok(StepTables { segments: Vec::new(), decode: false,
        positions: (start..start + rows).map(|p| p as i64).collect(),
        slots: (start..start + rows).map(|p| placement.slot(p)).collect::<Result<_>>()?,
        page_table: placement.table(used).collect(), table_width: used, table_stride: 0,
        cache_lengths: (start..start + rows).map(|p| (p + 1) as i32).collect(),
        lengths: (start..start + rows).map(|p| (p + 1).min(topk) as i32).collect(), exchange_rows: rows })
}

fn packed_tables(sequences: &[(&mut GlmPlacement, &[u32])], segments: &[Segment], topk: usize) -> Result<StepTables> {
    ensure!(sequences.len() == segments.len() && !segments.is_empty(), "packed layout members differ");
    let mut tables = sequence_tables(sequences[0].0, sequences[0].0.len, 0, topk)?;
    for ((placement, tokens), &segment) in sequences.iter().zip(segments) {
        let local = sequence_tables(placement, placement.len, tokens.len(), topk)?;
        tables.positions.extend_from_slice(&local.positions);
        tables.slots.extend_from_slice(&local.slots);
        tables.cache_lengths.extend_from_slice(&local.cache_lengths);
        tables.lengths.extend_from_slice(&local.lengths);
        tables.segments.push((segment, local));
    }
    // Segment attention uploads its own table in stream order. No combined
    // page table or per-row context matrix is allocated.
    tables.page_table.clear();
    tables.table_width = 0;
    tables.exchange_rows = tables.positions.len();
    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_tables_equal_singletons_and_ignore_partner_pages() -> Result<()> {
        let mut a = GlmPlacement { pages: vec![3, 4, 5], len: 60 };
        let mut b = GlmPlacement { pages: vec![11, 12], len: 1 };
        let tokens_a = [1; 12];
        let tokens_b = [2; 9];
        let limits = Limits { rows: 256, chunk_rows: 256, max_context: 4096, taps: Some(2048) };
        let segments = prefill_packing::plan(&[(60, 12), (1, 9)], &limits)?;
        let tables = packed_tables(&[(&mut a, &tokens_a), (&mut b, &tokens_b)], &segments, 64)?;
        for ((s, actual), reference) in tables.segments.iter().zip([
            sequence_tables(&a, a.len, tokens_a.len(), 64)?, sequence_tables(&b, b.len, tokens_b.len(), 64)?]) {
            assert_eq!(actual.positions, reference.positions);
            assert_eq!(actual.slots, reference.slots);
            assert_eq!(actual.page_table, reference.page_table);
            assert_eq!(actual.cache_lengths, reference.cache_lengths);
            assert_eq!(actual.lengths, reference.lengths);
            assert_eq!(tables.positions[s.first..s.end_row()], reference.positions);
        }
        b.pages = vec![18, 19];
        let changed = packed_tables(&[(&mut a, &tokens_a), (&mut b, &tokens_b)], &segments, 64)?;
        assert_eq!(tables.segments[0].1.page_table, changed.segments[0].1.page_table);
        assert_eq!(tables.segments[0].1.slots, changed.segments[0].1.slots);
        assert_ne!(tables.segments[1].1.slots, changed.segments[1].1.slots);
        assert_eq!(segments.iter().map(|s| (s.tap_offset, s.tap_rows)).collect::<Vec<_>>(), [(0, 12), (12, 9)]);
        Ok(())
    }

    #[test]
    fn packed_pass_retains_one_unit_for_segment_heads_and_taps() {
        let source = include_str!("packing.rs");
        let pass = source.split("pub(crate) fn prefill_packed_selected(").nth(1).unwrap()
            .split("pub(crate) fn packed_check_logits(").next().unwrap();
        assert!(pass.contains("self.step_lanes(&[tables], &[&tokens], self.depth, 0, link, runtime)?"));
        assert!(pass.contains("self.slot_workspaces(1)?"));
        assert!(!pass.contains("retained_logits_units("));
        assert!(!pass.contains(".chunks("));
    }

    #[test]
    fn exact_route_cap_splits_groups_but_never_rechunks_members() {
        let chunks = [(0, 64), (0, 64), (900, 129), (0, 12)];
        let limits = Limits { rows: 128, chunk_rows: 256, max_context: 4096, taps: Some(2048) };
        assert_eq!(prefill_packing::groups(4, |range| prefill_packing::plan(&chunks[range], &limits).is_ok()),
            [0..2, 2..3, 3..4]);
        assert_eq!(prefill_packing::groups(4, |range| prefill_packing::plan(&chunks[range], &Limits { rows: 0, ..limits }).is_ok()),
            [0..1, 1..2, 2..3, 3..4]);
    }

    #[test]
    fn cancelled_head_is_local_and_does_not_skip_other_members() -> Result<()> {
        let mut consumed = Vec::new();
        let outcomes = prefill_packing::finish_members(3, Ok, &mut |member, row| {
            consumed.push(row);
            Ok(if member == 1 { Err(anyhow::anyhow!("client went away")) } else { Ok(()) })
        })?;
        assert_eq!(consumed, [0, 1, 2]);
        let outcomes = prefill_packing::member_outcomes(3, Ok(outcomes));
        assert!(outcomes[0].is_ok() && outcomes[1].is_err() && outcomes[2].is_ok());
        Ok(())
    }
}
