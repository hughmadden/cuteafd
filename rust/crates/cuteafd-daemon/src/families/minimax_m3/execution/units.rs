//! MiniMax prefill on the shared executor (PLAN "Prefill contract"): a single-sequence call of more than the
//! decode rows runs as row units under [`run_prefill_units`], with up to `waves` units' Spark waves in flight on
//! the one link. MiniMax's per-row buffers hold the whole call (the `slot_rows` span), so every unit views its own
//! rows of each one (Kimi K3's shape): what a hook leaves for a later hook (the residual, the shared expert's
//! output) lives in the unit's rows, and the rest is rewritten inside each hook. `step` uploads every unit's
//! attention tiles before the executor runs; the call publishes its frontier once, and drafter work runs after it.
use super::*;
use crate::shared::prefill_pipeline::{prefill_units, retained_logits_units, run_prefill_units, FfnRoute, PrefillHooks,
    Unit};

/// Fewest rows a split unit keeps: four KV pages (`prefill_units`' `min_rows`).
const MIN_UNIT_ROWS: usize = 4 * PAGE_ROWS as usize;

/// The cuts of a single-sequence prefill call of `rows` rows into units of at most `unit_rows` with `waves` in
/// flight: equal, page-aligned units in multiples of `waves` (`prefill_units`) and, when the head projects more
/// than one row, all of those rows in the last unit (`retained_logits_units`). A call one unit holds at one wave is
/// that unit, also at golden's unaligned unit rows.
pub(super) fn unit_cuts(rows: usize, unit_rows: usize, waves: usize, logit_rows: usize, logit_capacity: usize)
    -> Result<Vec<std::ops::Range<usize>>> {
    ensure!(rows > 0 && unit_rows > 0 && waves > 0, MiniMaxError::Invalid("prefill unit call extent"));
    if waves == 1 && rows <= unit_rows { return Ok(vec![0..rows]); }
    if logit_rows > 1 { retained_logits_units(rows, unit_rows, waves, MIN_UNIT_ROWS, logit_rows, logit_capacity) }
    else { prefill_units(rows, unit_rows, waves, MIN_UNIT_ROWS) }
}

/// The waves a call of `admitted` waves cuts its units with, and the waves it keeps in flight.
pub(super) fn call_waves(admitted: usize, overlap: Overlap) -> Result<(usize, usize)> {
    match overlap {
        Overlap::Admitted => Ok((admitted, admitted)),
        Overlap::Identity { waves } => {
            ensure!((1..=admitted).contains(&waves), MiniMaxError::Invalid("identity waves beyond the admitted waves"));
            Ok((1, waves))
        }
    }
}

/// A call's units from its `cuts`: every layer through `last_layer` of its one request, slots taken in turn.
pub(super) fn call_units(cuts: Vec<std::ops::Range<usize>>, waves: usize, last_layer: usize) -> Vec<Unit> {
    cuts.into_iter().enumerate().map(|(index, rows)| Unit { index, sequence: 0, rows, layers: 0..last_layer + 1,
        slot: index % waves, barrier: false }).collect()
}

/// Drafter work (DSpark's context update borrows the target's scratch from row 0 and queues an ordered upload)
/// runs only between calls, never while a unit call is in flight.
pub(super) fn outside_units(in_units: bool) -> Result<()> {
    ensure!(!in_units, MiniMaxError::Invalid("drafter work inside a prefill unit call"));
    Ok(())
}

/// A unit's Spark wave between its send and land hooks: a token only. The wave's ticket stays in the link's FIFO
/// (`routed::Spark`), so the executor holds no owner and an aborted call leaves nothing here to release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Sent { layer: usize, slot: usize }

/// One unit call's hooks: each unit's tiles and row views, and borrows of the step's operands.
struct UnitHooks<'h, 'a> {
    exec: &'h Execution<'a>,
    /// The call's BF16 rows: the embedding and any media rows, written before the executor ran.
    embedded: CuteafdDeviceBuffer,
    batch: &'h BatchView,
    caches: &'h [LayerCache<'a>],
    experts: &'h mut dyn ExpertExecutor<'a>,
    /// Every unit's attention tiles, uploaded by `step` before the executor.
    plan: &'h AttentionPlan,
    tiles: &'h [UnitTiles],
    views: &'h [Rows<'h>],
    /// The call's rows (DFlash2 taps their tail).
    span: usize,
    units: usize,
    last_layer: usize,
    /// The rows the head projects, inside the last unit.
    logits: std::ops::Range<usize>,
    output: Option<DeviceLogits>,
}

/// Unit `unit`'s tiles and row views, which must hold its rows.
fn rows_of<'h>(tiles: &'h [UnitTiles], views: &'h [Rows<'h>], unit: &Unit) -> Result<(&'h UnitTiles, &'h Rows<'h>)> {
    let (Some(t), Some(s)) = (tiles.get(unit.index), views.get(unit.index)) else {
        return Err(MiniMaxError::Invalid("prefill unit outside its call").into());
    };
    ensure!(t.first == unit.rows.start && t.rows == unit.rows.len(), MiniMaxError::Invalid("prefill unit rows differ from its tiles"));
    Ok((t, s))
}

impl PrefillHooks for UnitHooks<'_, '_> {
    type Ticket = Sent;
    fn route(&self, layer: usize) -> FfnRoute {
        // Dense and RTX-resident layers complete inside one hook; a Spark layer's wave is in flight across hooks.
        if self.exec.cfg.moe[layer] && !self.experts.local(layer) { FfnRoute::Spark } else { FfnRoute::Local }
    }
    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        ensure!(unit.sequence == 0 && unit.layers == (0..self.last_layer + 1),
            MiniMaxError::Invalid("prefill units run every executed layer of one request"));
        let (t, s) = rows_of(self.tiles, self.views, unit)?;
        let embedded = view(self.embedded, t.first * 6144 * 2, t.rows * 6144 * 2)?;
        self.exec.native.widen(embedded, s.get("residual")?, t.rows * 6144)
    }
    fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let (t, s) = rows_of(self.tiles, self.views, unit)?;
        let cache = self.caches.get(layer).context("MiniMax layer cache")?;
        self.exec.attention(layer, t, self.batch.pool_pages, self.batch, cache, self.plan, s, None)
    }
    fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let (t, s) = rows_of(self.tiles, self.views, unit)?;
        if !self.exec.cfg.moe[layer] { return self.exec.dense_ffn(layer, t.first, t.rows, self.span, s, None) }
        // RTX-resident experts: the dispatch and its collect back to back.
        self.exec.ffn_front(layer, unit.slot, t.rows, s, &mut *self.experts, None)?;
        self.exec.ffn_back(layer, unit.slot, t.first, t.rows, self.span, s, &mut *self.experts)
    }
    fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<Sent> {
        let (t, s) = rows_of(self.tiles, self.views, unit)?;
        self.exec.ffn_front(layer, unit.slot, t.rows, s, &mut *self.experts, None)?;
        Ok(Sent { layer, slot: unit.slot })
    }
    fn land_experts(&mut self, layer: usize, unit: &Unit, sent: Sent) -> Result<()> {
        ensure!(sent == Sent { layer, slot: unit.slot }, MiniMaxError::Invalid("prefill unit wave identity"));
        let (t, s) = rows_of(self.tiles, self.views, unit)?;
        self.exec.ffn_back(layer, unit.slot, t.first, t.rows, self.span, s, &mut *self.experts)
    }
    fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        // The head projects the requested rows once, in the last unit (`unit_cuts` keeps them there): every
        // earlier unit has ended by then, and the residual plane keeps every unit's rows.
        if unit.index + 1 < self.units || self.logits.is_empty() || self.last_layer + 1 < self.exec.cfg.layers {
            return Ok(());
        }
        ensure!(unit.rows.start <= self.logits.start && self.logits.end <= unit.rows.end,
            MiniMaxError::Invalid("prefill logits outside the last unit"));
        let e = self.exec;
        self.output = Some(e.head.logits(&e.native, &e.weights, &e.scratch, self.logits.clone())?);
        Ok(())
    }
    fn commit_unit(&mut self, _: &Unit) -> Result<()> {
        // The call publishes its frontier once, after its step drains; the scheduler hands its taps to the
        // drafter after that.
        Ok(())
    }
}

impl<'a> Execution<'a> {
    /// The units of a single-sequence prefill call of `rows` rows through `last_layer`, the head projecting
    /// `logit_rows` of them, and the waves it keeps in flight (`overlap`).
    pub(super) fn units(&self, rows: usize, logit_rows: usize, last_layer: usize, overlap: Overlap)
        -> Result<(Vec<Unit>, usize)> {
        let (cut, waves) = call_waves(self.waves, overlap)?;
        Ok((call_units(unit_cuts(rows, self.unit_rows, cut, logit_rows, self.logit_rows)?, waves, last_layer), waves))
    }

    /// Runs a single-sequence prefill call's `units` on the shared executor with up to `waves` units' Spark waves
    /// in flight. `step` uploaded every unit's metadata (`plan`) before; nothing here uploads. An error returns at
    /// once: `step` stops the exchange before it drains (the hooks own nothing).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill(&self, embedded: CuteafdDeviceBuffer, batch: &BatchView, caches: &[LayerCache<'a>],
        experts: &mut dyn ExpertExecutor<'a>, plan: &AttentionPlan, units: &[Unit], waves: usize, last_layer: usize,
        logits: std::ops::Range<usize>) -> Result<Option<DeviceLogits>> {
        let span = batch.positions.len();
        let starts = units.iter().skip(1).map(|unit| unit.rows.start).collect::<Vec<_>>();
        let tiles = unit_tiles(plan, span, &starts);
        let views = units.iter().map(|unit| self.scratch.rows(unit.rows.start, unit.rows.len())).collect::<Result<Vec<_>>>()?;
        tracing::debug!(rows = span, units = units.len(), waves,
            sizes = ?units.iter().map(|unit| unit.rows.len()).collect::<Vec<_>>(), "MiniMax prefill units");
        let mut hooks = UnitHooks { exec: self, embedded, batch, caches, experts, plan, tiles: &tiles, views: &views,
            span, units: units.len(), last_layer, logits, output: None };
        // Set after the fallible setup and cleared on both paths: drafter work borrows this scratch.
        self.in_units.set(true);
        let run = run_prefill_units(&mut hooks, units, waves);
        self.in_units.set(false);
        run?;
        Ok(hooks.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefill_pipeline::{prefill_schedule, Step};

    /// Unit shapes the planner and launcher admit (the descriptor's 2 x 4096, pinned 3 x 2752 and 4 x 2048, smaller units).
    const SHAPES: [(usize, usize); 6] = [(4096, 1), (4096, 2), (2752, 3), (2048, 4), (1024, 2), (256, 2)];

    fn source(text: &'static str) -> &'static str { text.split("#[cfg(test)]").next().unwrap() }

    /// The body of `fn name(` in `source`, through the next item at the same depth.
    fn body<'s>(source: &'s str, name: &str) -> &'s str {
        let rest = source.split(name).nth(1).unwrap_or_else(|| panic!("{name}"));
        let end = ["\n    fn ", "\n    pub fn ", "\n    pub(super) fn ", "\n}"].iter().filter_map(|n| rest.find(n)).min().unwrap();
        &rest[..end]
    }

    /// The shared cut rule: a serving call (one head row) cuts as `prefill_units` with four-page units; every row's
    /// logits stay in one unit when the call fits one (scoring), and are refused across units with the shared
    /// text; a call one unit holds at one wave is that unit, also unaligned (golden's windows).
    #[test] fn minimax_unit_cuts_follow_the_shared_rule() -> Result<()> {
        for (unit_rows, waves) in SHAPES {
            let span = unit_rows * waves;
            for rows in (65..=span).step_by(29).chain([span, unit_rows]) {
                let cuts = unit_cuts(rows, unit_rows, waves, 1, 64)?;
                let shared = if waves == 1 && rows <= unit_rows { vec![0..rows] } else { prefill_units(rows, unit_rows, waves, 256)? };
                ensure!(cuts == shared, "{waves} x {unit_rows}, {rows} rows: {cuts:?} vs {shared:?}");
                ensure!(cuts.len() <= waves && cuts[0].start == 0 && cuts.last().unwrap().end == rows
                    && cuts.windows(2).all(|w| w[0].end == w[1].start && w[1].start % 64 == 0)
                    && cuts.iter().all(|c| c.len() <= unit_rows), "{waves} x {unit_rows}, {rows} rows: {cuts:?}");
                if rows <= unit_rows { ensure!(unit_cuts(rows, unit_rows, waves, rows, span)? == [0..rows]); }
            }
        }
        let refused = unit_cuts(8192, 4096, 2, 8192, 8192).unwrap_err().to_string();
        ensure!(refused.contains("score in calls of at most 4096 rows"), "{refused}");
        assert_eq!(unit_cuts(100, 100, 1, 100, 100)?, [0..100]);
        assert!(prefill_units(100, 100, 1, 256).is_err(), "the one-wave rule is what admits unaligned windows");
        assert_eq!(unit_cuts(4095, 4096, 2, 1, 64)?, [0..4095]);
        assert_eq!(unit_cuts(4412, 4096, 2, 1, 64)?, [0..2240, 2240..4412]);
        assert_eq!(unit_cuts(8256, 2752, 3, 1, 64)?, [0..2752, 2752..5504, 5504..8256]);
        let units = call_units(unit_cuts(8256, 2752, 3, 1, 64)?, 3, 59);
        assert!(units.iter().enumerate().all(|(i, u)| u.index == i && u.slot == i && u.sequence == 0
            && u.layers == (0..60) && !u.barrier));
        Ok(())
    }

    /// The real schedule over MiniMax's routes and cuts never lets one unit write bytes another unit holds live:
    /// its residual (begin to end) and its shared expert's output (send to land), at the byte extents
    /// `Scratch::rows` views (`row_extent`); every in-flight unit's view of every per-row buffer lies in the
    /// span's allocation, apart from the others. The negative controls (the shared expert's output in one plane
    /// all units share, and overlapping cuts) are caught.
    #[test] fn minimax_real_schedule_keeps_every_units_live_rows() -> Result<()> {
        use cuteafd_loader::plan::families::minimax::admission::ROW_BUFFERS;
        use super::super::super::scratch::row_extent;
        let extent = |plane: &str, rows: &std::ops::Range<usize>, whole: &[&str]| -> Result<std::ops::Range<usize>> {
            if whole.contains(&plane) { return Ok(0..usize::MAX); }
            let (at, bytes) = row_extent(plane, rows.start, rows.len())?.context("a per-row plane")?;
            Ok(at..at + bytes)
        };
        let hazard = |routes: &[FfnRoute], units: &[Unit], waves: usize, whole: &[&str]| -> Result<Option<String>> {
            let mut live: Vec<(usize, &str, std::ops::Range<usize>)> = Vec::new();
            for step in prefill_schedule(routes, units, waves)? {
                let (unit, writes, open, close): (usize, &[&str], Option<&str>, Option<&str>) = match step {
                    Step::Begin { unit } => (unit, &["residual"], Some("residual"), None),
                    Step::Attention { unit, .. } => (unit, &["norm", "q", "kv", "attention", "delta", "residual"], None, None),
                    Step::Local { unit, .. } => (unit, &["norm", "route_ids", "gate_up", "shared", "routed", "delta",
                        "residual"], None, None),
                    Step::Send { unit, .. } => (unit, &["norm", "router_logits", "route_ids", "route_weights", "gate_up",
                        "shared"], Some("shared"), None),
                    Step::Land { unit, .. } => (unit, &["routed", "delta", "residual"], None, Some("shared")),
                    Step::End { unit } => (unit, &[], None, Some("residual")),
                    Step::Commit { .. } => continue,
                };
                let rows = &units[unit].rows;
                if let Some(plane) = close { live.retain(|(owner, held, _)| !(*owner == unit && *held == plane)); }
                for &plane in writes {
                    let bytes = extent(plane, rows, whole)?;
                    if let Some((owner, ..)) = live.iter().find(|(owner, held, range)| *owner != unit && *held == plane
                        && range.start < bytes.end && bytes.start < range.end) {
                        return Ok(Some(format!("{step:?} writes unit {owner}'s live {plane}")));
                    }
                }
                if let Some(plane) = open { live.push((unit, plane, extent(plane, rows, whole)?)); }
            }
            Ok(None)
        };
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |bound: usize| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed % bound as u64) as usize };
        for case in 0..300 {
            let (unit_rows, waves) = SHAPES[next(SHAPES.len())];
            let span = unit_rows * waves;
            let rows = 65 + next(span - 64);
            let last = 2 + next(58);
            let routes = (0..=last).map(|layer| if layer < 3 || next(5) == 0 { FfnRoute::Local } else { FfnRoute::Spark })
                .collect::<Vec<_>>();
            let units = call_units(unit_cuts(rows, unit_rows, waves, 1, 64)?, waves, last);
            let found = hazard(&routes, &units, waves, &[])?;
            ensure!(found.is_none(), "case {case}, {waves} x {unit_rows}, {rows} rows: {found:?}");
            for (i, a) in units.iter().enumerate() {
                for b in &units[i + 1..] {
                    for &(name, ..) in ROW_BUFFERS {
                        let (x, y) = (extent(name, &a.rows, &[])?, extent(name, &b.rows, &[])?);
                        ensure!(x.end <= y.start && y.end <= extent(name, &(0..span), &[])?.end,
                            "case {case}: {name} views of units {} and {} overlap or leave the span", a.index, b.index);
                    }
                }
            }
        }
        // Negative controls: a full 2 x 4096 call with the shared expert's output in one shared plane, and two
        // units cut over the same rows.
        let routes = (0..60).map(|layer| if layer < 3 { FfnRoute::Local } else { FfnRoute::Spark }).collect::<Vec<_>>();
        let units = call_units(unit_cuts(8192, 4096, 2, 1, 64)?, 2, 59);
        ensure!(hazard(&routes, &units, 2, &[])?.is_none());
        let shared = hazard(&routes, &units, 2, &["shared"])?.context("a shared plane at a fixed offset is caught")?;
        ensure!(shared.contains("live shared"), "{shared}");
        let overlapping = call_units(vec![0..4096, 2048..6144], 2, 59);
        let residual = hazard(&routes, &overlapping, 2, &[])?.context("overlapping cuts are caught")?;
        ensure!(residual.contains("live residual"), "{residual}");
        Ok(())
    }

    /// The one-link mapping on the real schedule: each unit's Spark waves go out on the link as `Routed` sends
    /// them (`Wave` by layer, the unit's slot and rows, checked by the link's FIFO `WaveOrder`): at most `waves`
    /// in flight, every land the oldest wave, and wave k's return slot (k % waves, the exchange's rotation) free.
    #[test] fn minimax_unit_waves_land_in_dispatch_order_on_one_link() -> Result<()> {
        use super::super::super::routed::{Wave, WaveOrder};
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |bound: usize| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed % bound as u64) as usize };
        for case in 0..400 {
            let (unit_rows, waves) = SHAPES[next(SHAPES.len())];
            let span = unit_rows * waves;
            let rows = 65 + next(span - 64);
            let last = 3 + next(57);
            let routes = (0..=last).map(|layer| if layer < 3 || next(5) == 0 { FfnRoute::Local } else { FfnRoute::Spark })
                .collect::<Vec<_>>();
            let units = call_units(unit_cuts(rows, unit_rows, waves, 1, 64)?, waves, last);
            let mut order = WaveOrder::new(waves);
            let (mut sent, mut free) = (0u64, vec![true; waves]);
            for step in prefill_schedule(&routes, &units, waves)? {
                match step {
                    Step::Send { layer, unit } => {
                        let (u, slot) = (&units[unit], (sent % waves as u64) as usize);
                        ensure!(free[slot], "case {case}: wave {sent} reuses return slot {slot} before its previous wave landed");
                        free[slot] = false;
                        order.push(Wave { layer, slot: u.slot, rows: u.rows.len() }, sent)?;
                        sent += 1;
                    }
                    Step::Land { layer, unit } => {
                        let u = &units[unit];
                        let wave = order.pop(Wave { layer, slot: u.slot, rows: u.rows.len() })?;
                        free[(wave % waves as u64) as usize] = true;
                    }
                    _ => {}
                }
            }
            ensure!(order.is_empty() && free.iter().all(|&f| f), "case {case}: the call ends with every wave landed");
        }
        Ok(())
    }

    /// Drafter work never runs inside a unit call: the guard refuses it while one is in flight, `prefill` holds
    /// the flag only around the executor and clears it on both paths, the drafter entry points check it, and the
    /// unit hooks never call them.
    #[test] fn minimax_draft_updates_never_run_inside_a_unit_call() {
        outside_units(false).unwrap();
        let refused = outside_units(true).unwrap_err().to_string();
        assert!(refused.contains("drafter work inside a prefill unit call"), "{refused}");
        let execution = source(include_str!("../execution.rs"));
        for entry in ["pub fn update_draft(", "pub fn propose("] {
            assert!(body(execution, entry).contains("outside_units(self.in_units.get())?"), "{entry}");
        }
        let units = source(include_str!("units.rs"));
        let prefill = body(units, "pub(super) fn prefill(");
        let order = ["self.in_units.set(true);", "run_prefill_units(&mut hooks, units, waves)", "self.in_units.set(false);",
            "run?;"].map(|needle| prefill.find(needle).unwrap_or_else(|| panic!("{needle}")));
        assert!(order.windows(2).all(|w| w[0] < w[1]), "{order:?}");
        for banned in ["update_draft(", ".update(", "propose(", "prefill_rows("] {
            assert!(!units.contains(banned), "the unit hooks call {banned}");
        }
    }

    /// Every unit's metadata is uploaded before the executor: `step` plans every unit's tiles first, and neither
    /// the hooks nor the layer bodies they run upload, reserve or copy from the host. The hooks hold no owner: the
    /// send/land ticket is a `Copy` token.
    #[test] fn minimax_unit_calls_upload_every_tile_before_the_executor() {
        let execution = source(include_str!("../execution.rs"));
        let step = body(execution, "fn step(");
        let (plan, prefill) = (step.find("let plan=self.plan_attention(batch,rows,&starts)?;").unwrap(),
            step.find("self.prefill(").unwrap());
        assert!(plan < prefill, "the call's tiles are uploaded before its units run");
        let units = source(include_str!("units.rs"));
        let bodies = ["fn attention(", "fn dense_ffn(", "fn ffn_front(", "fn ffn_back(", "fn tap(&self,layer:usize"].map(|f| body(execution, f));
        for text in bodies.iter().copied().chain([units]) {
            for banned in [".upload(", "copy_h2d", ".reserve("] { assert!(!text.contains(banned), "{banned}"); }
        }
        assert!(units.contains("#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub(super) struct Sent"));
        assert!(!units.contains(&["Wave", "Ticket"].concat()) && !units.contains("Allocation"));
    }

    /// The identity check's arms run the same units at every wave count: cut with one wave, slots taken
    /// in turn among the waves in flight, never more units than the admitted slots; a full-span call cuts as
    /// serving does, so the gate covers serving's cuts. Waves beyond the admitted ones are refused.
    #[test] fn minimax_identity_units_cut_alike_at_every_wave_count() -> Result<()> {
        for (unit_rows, admitted) in SHAPES {
            let span = unit_rows * admitted;
            for rows in (65..=span).step_by(41).chain([span]) {
                let reference = unit_cuts(rows, unit_rows, 1, 1, 64)?;
                for waves in 1..=admitted {
                    let (cut, run) = call_waves(admitted, Overlap::Identity { waves })?;
                    assert_eq!((cut, run), (1, waves));
                    let units = call_units(unit_cuts(rows, unit_rows, cut, 1, 64)?, run, 59);
                    ensure!(units.iter().map(|u| u.rows.clone()).eq(reference.iter().cloned()) && units.len() <= admitted
                        && units.iter().all(|u| u.slot == u.index % waves), "{admitted} x {unit_rows}, {rows} rows at {waves} waves");
                }
            }
            ensure!(unit_cuts(span, unit_rows, admitted, 1, 64)? == unit_cuts(span, unit_rows, 1, 1, 64)?,
                "{admitted} x {unit_rows}: a full call cuts as serving does");
        }
        assert_eq!(call_waves(3, Overlap::Admitted)?, (3, 3));
        assert!(call_waves(2, Overlap::Identity { waves: 3 }).is_err() && call_waves(2, Overlap::Identity { waves: 0 }).is_err());
        Ok(())
    }

    /// Media rows land in the call's embedded rows before the step runs (`MiniMaxBackend::execute`), so a span
    /// across a unit cut is whole; the unit hooks only widen their rows.
    #[test] fn minimax_media_rows_land_before_the_unit_call() {
        let backend = source(include_str!("../backend.rs"));
        let execute = body(backend, "fn execute(");
        let (media, call) = (execute.find("library.copy_h2d(destination").unwrap(), execute.find("self.execution.execute(").unwrap());
        assert!(media < call);
        let units = source(include_str!("units.rs"));
        for banned in ["write_chunk", "copy_h2d", "RequestMedia"] { assert!(!units.contains(banned), "{banned}"); }
    }
}
