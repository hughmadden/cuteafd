//! Text-only target and opt-in native MLA adapter to the shared scheduler.
use super::{engine::Engine, prefix::{KimiPrefix, Placement}};
use crate::shared::{draft::routes::RoundRoutes, prefix::{CudaCopyEngine, PrefixArgs, Toggle},
    qwen::scheduler::{self, Backend}, token_io::DeviceLogits};
use anyhow::{ensure, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, MediaKeys, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use std::{cell::{Cell, RefCell}, path::Path};

#[derive(Debug, Default)]
pub(crate) struct DraftState {
    anchor: Option<u32>,
    ready: bool,
    valid_from: usize,
}
impl scheduler::DraftState for DraftState {
    fn pending_rows(&self) -> usize { usize::from(self.ready && self.anchor.is_some()) }
    fn close(&mut self, token: u32) { self.anchor = Some(token); }
}

pub(crate) struct KimiBackend<'e, 'a> {
    engine: &'e Engine<'a>,
    profile: RefCell<[f64; 2]>,
    geometry: scheduler::Geometry,
    steady_census: Cell<bool>,
    origins: RefCell<Vec<usize>>,
}

impl<'e, 'a> KimiBackend<'e, 'a> {
    pub(crate) fn new(engine: &'e Engine<'a>) -> Result<Self> {
        Ok(Self { engine, profile: RefCell::new([0.0; 2]), geometry: engine.geometry()?,
            steady_census: Cell::new(false), origins: RefCell::new(vec![0; engine.geometry()?.slots]) })
    }
}

impl<'e, 'a> Backend<'a> for KimiBackend<'e, 'a> {
    type Placement = Placement;
    type Draft = DraftState;
    type History = ();
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = super::prefix::SnapshotMeta;
    type Prefix = KimiPrefix<'a>;

    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry { family: "kimi_k3", vocab_size: self.geometry.vocab_size,
            max_context: self.geometry.max_context, slots: self.geometry.slots,
            prefill_rows: self.geometry.prefill_rows, decode_rows: self.geometry.decode_rows,
            full_prefill_logits: false }
    }
    fn message_starts(&self) -> &[&str] { &["<|open|>"] }
    fn library(&self) -> &'a cuteafd_ffi::NativeLibrary { self.engine.library() }
    fn profile(&self) -> &RefCell<[f64; 2]> { &self.profile }
    fn prefix_cache(&self, args: &PrefixArgs, lanes: usize) -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>, Self::Meta>)> {
        ensure!(args.prefix_partial == Toggle::Off, "Kimi restores exact KDA/MLA snapshots only");
        let args = args.with_scoring(self.geometry.decode_rows, self.geometry.prefill_rows, self.geometry.vocab_size)?;
        let family = self.engine.prefix_family()?;
        let layout = family.layout();
        let host = args.host_config(layout, self.geometry.max_context, lanes)?.map(|config|
            CudaCopyEngine::registered_retained(self.library(), family.snapshot_owners(), family.snapshot_retained()).map(|copy| (config, copy)))
            .transpose()?;
        let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
        let config = PrefixConfig { entries: args.prefix_cache_entries, mark_slots: family.slots(),
            keep_logits: true, min_tokens: args.prefix_cache_min_tokens };
        let cache = PrefixCache::new_with_draft_pages(layout, config, host, family.draft_page_bytes())?;
        cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64, layout.pages as u64,
            family.record_format(), host_bytes);
        Ok((family, cache))
    }
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        ensure!(job.media.is_empty(), "Kimi serving currently supports text only");
        let keys = MediaKeys::new(&tokens, u32::try_from(self.geometry.vocab_size)?, &[])?;
        let media = RequestMedia::new(Vec::new(), 7168, tokens.len())?;
        Ok((scheduler::Prompt { job, tokens, keys }, media, Vec::new()))
    }
    fn probe_features(&self, prompt: &scheduler::Prompt, media: &mut RequestMedia,
        _cache: &mut EmbeddingCache, _snapshot: &Path) -> Result<()> {
        ensure!(prompt.job.media.is_empty() && media.spans().is_empty(), "Kimi media is not admitted");
        Ok(())
    }
    fn coordinates(&self, _tokens: &[u32], job: &NativeRequest, media: &RequestMedia) -> Result<()> {
        ensure!(job.media.is_empty() && media.spans().is_empty(), "Kimi text coordinates require no media");
        Ok(())
    }
    fn placement(&self, units: Vec<u32>, slot: i32) -> Placement {
        Placement { units, slot: slot as usize, len: 0, state_len: 0 }
    }
    fn bind_request(&self, placement: &mut Placement, _coordinates: (), _media: RequestMedia,
        _tokens: &[u32], resume: usize) -> Result<()> {
        if let Some(draft) = self.engine.native_draft() {
            let (origin, end) = draft.frontier(placement.slot)?;
            ensure!(end == resume, "Kimi draft binding differs from prefix admission");
            self.origins.borrow_mut()[placement.slot] = origin;
        }
        Ok(())
    }
    fn history(&self, _placement: &Placement) {}
    fn restore_draft(&self, placement: &Placement, frontier: usize, _tokens: &[u32], seq: &mut DraftState) -> Result<()> {
        let draft = self.engine.native_draft().ok_or_else(|| anyhow::anyhow!("Kimi drafter was not admitted"))?;
        let (origin, end) = draft.frontier(placement.slot)?;
        ensure!(end == frontier, "Kimi draft restored frontier differs from target");
        seq.ready = end > origin;
        seq.valid_from = origin;
        Ok(())
    }
    fn needs_snapshot_filter(&self, neural: bool) -> bool { neural }
    fn snapshot_compatible(&self, saved: &Self::Meta, frontier: usize, neural: bool) -> bool {
        !neural || saved.draft == Some((0, frontier))
    }

    fn prefetch(&self, _tokens: &[u32], _next: &[u32]) {}
    fn require_live(&self) -> Result<()> { self.engine.require_live() }
    fn log_table_stats(&self, _phase: &str) {}
    fn prefill_device(&self, placement: &mut Placement, tokens: &[u32], logit_rows: usize,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        self.engine.step_device(&mut [(placement, tokens)], pool, logit_rows.max(1), true)
            .map(|logits| (logit_rows > 0).then_some(logits))
    }
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        if speculative { return self.engine.verify_device(sequences, pool, true).map(Some); }
        let rows = sequences.iter().map(|(_, tokens)| tokens.len()).sum();
        self.engine.step_device(sequences, pool, rows, true).map(Some)
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        _kind: scheduler::UngraphedKind, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        if speculative { return self.engine.verify_device(sequences, pool, false).map(Some); }
        let rows = sequences.iter().map(|(_, tokens)| tokens.len()).sum();
        self.engine.step_device(sequences, pool, rows, false).map(Some)
    }
    fn verify_bucket_rows(&self, rows: usize, _speculative: bool, _diagnostic: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value {
        serde_json::json!({"kind":"kimi_owned_mla_tp2_kda", "rows":rows, "contexts":contexts})
    }
    fn draft_geometry(&self, snapshot: &Path, width: usize) -> Result<cuteafd_core::PolicyGeometry> {
        ensure!(self.engine.native_draft().is_some(), "Kimi drafter was not admitted");
        self.engine.draft_geometry(snapshot, width)
    }
    fn arm_draft_probe(&self) -> Result<()> { self.engine.arm_draft_probe() }
    fn prefill_draft(&self, placement: &Placement, start: usize, chunk: &[u32], next: Option<u32>,
        seq: &mut DraftState) -> Result<()> {
        let draft = self.engine.native_draft().ok_or_else(|| anyhow::anyhow!("Kimi drafter was not admitted"))?;
        ensure!(!chunk.is_empty() && placement.state_len == start + chunk.len(),
            "Kimi draft prefill differs from committed target frontier");
        let rows = (0..chunk.len()).map(|tap_row| crate::shared::speculation::dflash::ContextRow {
            tap_row, slot: placement.slot, position: start + tap_row }).collect::<Vec<_>>();
        draft.update(&rows)?;
        seq.ready = true;
        seq.valid_from = self.origins.borrow()[placement.slot];
        seq.anchor = next;
        Ok(())
    }
    fn draft(&self, sequences: &mut [scheduler::DraftSeq<'_, Placement, DraftState>], timing: &mut scheduler::DraftTiming)
        -> Result<Vec<Vec<u32>>> {
        let draft = self.engine.native_draft().ok_or_else(|| anyhow::anyhow!("Kimi drafter was not admitted"))?;
        let started = std::time::Instant::now();
        let mut output = vec![Vec::new(); sequences.len()];
        let selected = sequences.iter().enumerate().filter(|(_, seq)| seq.depth > 0)
            .map(|(index, seq)| {
                ensure!(seq.seq.ready && seq.depth <= draft.drafts(), "Kimi draft continuation is cold or too wide");
                Ok((index, crate::shared::speculation::dflash::DraftSeq { slot: seq.placement.slot,
                    anchor: seq.seq.anchor.ok_or_else(|| anyhow::anyhow!("Kimi draft anchor absent"))?,
                    position: seq.placement.len, valid_from: seq.seq.valid_from }))
            }).collect::<Result<Vec<_>>>()?;
        for batch in selected.chunks(draft.max_batch_sequences()) {
            let rows = batch.iter().map(|(_, seq)| *seq).collect::<Vec<_>>();
            let proposals = draft.propose(&rows)?;
            ensure!(proposals.len() == batch.len(), "Kimi draft output count differs from batch");
            for ((index, _), proposal) in batch.iter().zip(proposals) {
                output[*index] = proposal.tokens.into_iter().take(sequences[*index].depth).collect();
            }
            timing.steps += 1;
        }
        timing.seconds += started.elapsed().as_secs_f64();
        Ok(output)
    }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, DraftState, ()>], speculative: bool)
        -> Result<()> {
        if speculative {
            self.engine.accept_provisional(verified)?;
        } else {
            for sequence in verified.iter() {
                ensure!(sequence.placement.state_len == sequence.start + sequence.rows.len(),
                    "Kimi non-speculative state did not commit all input rows");
            }
        }
        if verified.iter().any(|sequence| sequence.neural) {
            let drafter = self.engine.native_draft().ok_or_else(|| anyhow::anyhow!("Kimi drafter was not admitted"))?;
            let mut context = Vec::new();
            for sequence in verified.iter_mut() {
                let Some((keep, after)) = sequence.neural_kept() else { continue; };
                ensure!(keep > 0 && keep <= sequence.rows.len()
                    && sequence.placement.state_len == sequence.start + keep,
                    "Kimi draft accept differs from exact committed target state");
                if !sequence.seq.ready {
                    // Admission already bound cold(P) and reserved this request's growth.
                    // Do not reset again: that would release its upfront live credits.
                    let (origin, end) = drafter.frontier(sequence.placement.slot)?;
                    ensure!(origin == sequence.start && end == sequence.start,
                        "Kimi cold draft continuation differs from admitted frontier");
                    sequence.seq.valid_from = origin;
                }
                context.extend((0..keep).map(|offset| crate::shared::speculation::dflash::ContextRow {
                    tap_row: sequence.first_row + offset, slot: sequence.placement.slot,
                    position: sequence.start + offset }));
                sequence.seq.ready = true;
                sequence.seq.anchor = Some(after);
            }
            drafter.update(&context)?;
        }
        // Shared token selection completed the producing stream before accept.
        // Record one steady census, never a per-step sampler or a compute drain.
        if !self.steady_census.replace(true) {
            let devices = self.engine.devices();
            crate::shared::inventory::RuntimeInventory::measure(self.library(), &devices,
                crate::shared::inventory::InventoryPhase::PostLoad, |_| Ok(()))?
                .record("kimi_k3-serving-steady", &[]);
        }
        Ok(())
    }
    fn cancel(&self, placement: &Placement) -> Result<()> {
        self.engine.cancel_provisional(placement)?;
        if let Some(draft) = self.engine.native_draft() { draft.discard(placement.slot)?; }
        Ok(())
    }
    fn probe_begin(&self, rows: usize) { self.engine.probe_begin(rows); }
    fn probe_end(&self) { self.engine.probe_end(); }
    fn probe_finish<T>(&self, observe: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> {
        self.engine.probe_finish(observe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::qwen::scheduler::DraftState as _;

    #[test]
    fn kimi_verify_cycle_observes_simple_wall_time_and_shared_routes() -> Result<()> {
        use crate::shared::draft::{resource::{self, LayerHome}, clock::RoundTimes, evidence::DraftSource};
        use cuteafd_core::{DraftMode, DraftPolicy, PolicySeed};
        // Same scheduler resource handoff as production; simple deliberately has no routes.
        let homes = [LayerHome::Dense, LayerHome::Local { slice_bytes: 1024.0 },
            LayerHome::Remote { slice_bytes: 1024.0 }];
        let geometry = resource::geometry(&homes, 896, 16, 64, 7, 0)?;
        for mode in [DraftMode::Simple, DraftMode::Shared] {
            let mut policy = DraftPolicy::new(geometry.clone(), PolicySeed::weak(7), false)?;
            policy.set_draft_mode(mode);
            let mut routes = RoundRoutes::new(3, 16);
            let layer_us = if policy.needs_routes() {
                routes.begin(4);
                routes.dense(0)?;
                let ids = (0..64).map(|id| id as u32).collect::<Vec<_>>();
                routes.push(1, &ids, u32::MAX)?;
                let staged = ids.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<_>>();
                routes.push_bytes(2, &staged, u32::MAX)?;
                routes.complete()?;
                assert_eq!(routes.flat().len(), 3 * 4 * 16);
                vec![None, Some(20.0), Some(80.0)]
            } else {
                assert!(routes.flat().is_empty());
                Vec::new()
            };
            let prior = [0.9; 3];
            let requests = [resource::Verified { id: 1, rows: 4, accepted: 3,
                source: DraftSource::Neural, prior: Some(&prior), features: None, censor: None }];
            resource::observe_drafted(&mut policy, &requests, &routes, &layer_us,
                RoundTimes { total_us: 200, draft_us: Some(40) }, 7, None, Some(1))?;
            assert_eq!(policy.stats().rounds, 1);
            if mode == DraftMode::Simple { assert!(policy.simple_round().is_some()); }
        }
        Ok(())
    }

    #[test]
    fn whole_prompt_hit_waits_for_first_target_row_before_drafting() {
        let mut seq = DraftState::default();
        assert_eq!(seq.pending_rows(), 0);
        seq.close(42);
        assert_eq!(seq.pending_rows(), 0, "a cached anchor cannot expose stale drafter context");
        seq.ready = true;
        seq.valid_from = 8192;
        assert_eq!(seq.pending_rows(), 1);
        assert_eq!(seq.anchor, Some(42));
        assert_eq!(seq.valid_from, 8192);
        seq.close(43);
        assert_eq!(seq.anchor, Some(43));
        assert_eq!(seq.valid_from, 8192, "closing a round cannot move the cold storage origin");
    }
}
