//! CPU contracts for V4 segmented metadata and the shared packed executor.
use super::*;
use crate::shared::prefill_packing::{self, Limits};
use crate::shared::prefill_pipeline::{self, FfnRoute, PrefillHooks, Unit};

fn limits() -> Limits { Limits { rows: 256, chunk_rows: 256, max_context: 4096, taps: Some(TAP_ROWS) } }

#[test]
fn packed_members_keep_singleton_tables_and_disjoint_state() -> Result<()> {
    let shape = PoolShape::new(3, 256, 48);
    let chunks = [(0, 29), (301, 40), (127, 59)];
    let segments = prefill_packing::plan(&chunks, &limits())?;
    let placements: Vec<_> = chunks.iter().enumerate().map(|(i, &(start, _))| {
        let mut p = Placement::new(i, (i as u32 * 16..(i as u32 + 1) * 16).collect());
        p.len = start; p
    }).collect();
    for ((p, &(start, rows)), s) in placements.iter().zip(&chunks).zip(&segments) {
        let packed = metadata::prefill_step(p, &shape, s.start, s.rows, 16, 32)?;
        let alone = metadata::prefill_step(p, &shape, start, rows, 16, 32)?;
        assert_eq!(packed.positions, alone.positions);
        assert_eq!(packed.main_slots, alone.main_slots);
        assert_eq!(packed.swa_indices, alone.swa_indices);
        assert_eq!(packed.c4_tables, alone.c4_tables);
        assert_eq!(packed.c128_tables, alone.c128_tables);
        assert_eq!(packed.c4_page_table, alone.c4_page_table);
        assert_eq!(packed.c128_indices, alone.c128_indices);
        for slot in packed.swa_indices.into_iter().filter(|&slot| slot >= 0) {
            assert_eq!(slot as usize / (shape.ring_pages * 256), p.state);
        }
        for slot in packed.c128_indices.into_iter().filter(|&slot| slot >= 0) {
            assert!(p.units.contains(&(slot as u32 / 2)));
        }
    }
    assert_eq!(segments.iter().map(|s| (s.tap_offset, s.tap_rows)).collect::<Vec<_>>(), [(0, 29), (29, 40), (69, 59)]);
    assert!(prefill_packing::plan(&[(0, 29), (0, 100)], &limits()).is_err());
    Ok(())
}

#[test]
fn independent_draft_tails_allow_sixteen_short_prompts() -> Result<()> {
    let limits = Limits { rows: 4096, chunk_rows: 4096, max_context: 8192,
        taps: Some(cuteafd_loader::serving_capacity::deepseek_v4_tap_rows(4096, true)) };
    let segments = prefill_packing::plan_with_tap_rows(&[(300, 200); 16], &limits, TAP_ROWS)?;
    assert_eq!(segments.last().map(|s| (s.tap_offset, s.tap_rows)), Some((15 * TAP_ROWS, TAP_ROWS)));
    assert_eq!(segments.iter().map(|s| s.rows).sum::<usize>(), 3200);
    assert!(prefill_packing::plan_with_tap_rows(&[(300, 200); 17], &limits, TAP_ROWS).is_err());
    Ok(())
}

struct Model {
    segments: Vec<prefill_packing::Segment>,
    values: Vec<i64>,
    states: Vec<i64>,
    expert_calls: usize,
    spark: bool,
}
impl PrefillHooks for Model {
    type Ticket = ();
    fn route(&self, _: usize) -> FfnRoute { if self.spark { FfnRoute::Spark } else { FfnRoute::Local } }
    fn begin_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
    fn attention(&mut self, layer: usize, _: &Unit) -> Result<()> {
        for (member, s) in self.segments.iter().enumerate() {
            for row in s.first..s.end_row() {
                self.states[member] += self.values[row];
                self.values[row] += self.states[member] + (s.start + row - s.first + layer) as i64;
            }
        }
        Ok(())
    }
    fn local_ffn(&mut self, _: usize, _: &Unit) -> Result<()> {
        self.expert_calls += 1;
        for value in &mut self.values { *value = *value * 3 + 1; }
        Ok(())
    }
    fn send_experts(&mut self, _: usize, _: &Unit) -> Result<()> { Ok(()) }
    fn land_experts(&mut self, layer: usize, unit: &Unit, _: ()) -> Result<()> { self.local_ffn(layer, unit) }
    fn end_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
    fn commit_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
}

#[test]
fn packed_executor_matches_singletons_and_partner_cannot_change_history() -> Result<()> {
    let chunks = [(0, 29), (301, 40), (127, 59)];
    let segments = prefill_packing::plan(&chunks, &limits())?;
    for spark in [false, true] {
        for changed_partner in [false, true] {
            let values: Vec<_> = (0..128).map(|i| if changed_partner && (29..69).contains(&i) { 1000 } else { i as i64 }).collect();
            let mut packed = Model { segments: segments.clone(), values: values.clone(), states: vec![3, 5, 7], expert_calls: 0, spark };
            let units = prefill_pipeline::sequence_units(128, 256, 1, 5)?;
            prefill_pipeline::run_prefill_units(&mut packed, &units, 1)?;
            assert_eq!(packed.expert_calls, 5);
            for (i, s) in segments.iter().enumerate() {
                let singleton = prefill_packing::plan(&[(s.start, s.rows)], &limits())?;
                let mut alone = Model { segments: singleton, values: values[s.first..s.end_row()].to_vec(),
                    states: vec![[3, 5, 7][i]], expert_calls: 0, spark };
                let units = prefill_pipeline::sequence_units(s.rows, 256, 1, 5)?;
                prefill_pipeline::run_prefill_units(&mut alone, &units, 1)?;
                assert_eq!(packed.values[s.first..s.end_row()], alone.values);
                assert_eq!(packed.states[i], alone.states[0]);
            }
        }
    }
    Ok(())
}

#[test]
fn cancellation_stays_member_local_and_exact_unknown_routes_run_alone() -> Result<()> {
    let mut seen = Vec::new();
    let outcomes = prefill_packing::finish_members(3, Ok, &mut |i, row| {
        seen.push(row);
        Ok(if i == 1 { Err(anyhow::anyhow!("cancelled")) } else { Ok(()) })
    })?;
    assert_eq!(seen, [0, 1, 2]);
    assert!(outcomes[0].is_ok() && outcomes[1].is_err() && outcomes[2].is_ok());
    let chunks = [(0, 29), (0, 40), (0, 59)];
    for (cap, expected) in [(0, vec![0..1, 1..2, 2..3]), (69, vec![0..2, 2..3]), (128, vec![0..3])] {
        assert_eq!(prefill_packing::groups(chunks.len(), |r|
            prefill_packing::plan(&chunks[r], &Limits { rows: cap, ..limits() }).is_ok()), expected);
    }
    Ok(())
}
