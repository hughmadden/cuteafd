//! Shared scheduler adapter; a native Step is the only pending verify evidence.
use super::{cache::CacheRank, execution::Execution, prefix::{Frontier, Placement, Step5Prefix},
    transaction::{Pending, SequenceRows, Step}};
use crate::shared::{draft::routes::RoundRoutes, memory::device::{Allocation, Device},
    prefix::{CudaCopyEngine, PrefixArgs, Toggle}, qwen::scheduler::{self, Backend},
    token_io::{DeviceLogits, TokenEmbedding}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, MediaKeys, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use cuteafd_ffi::NativeLibrary;
use std::{cell::{Cell, RefCell}, path::Path};

const H: usize = 4096;
const V: usize = 128896;
type Evidence<'e, 'a> = Step<'e, 'a, CudaCopyEngine<'a>>;

#[derive(Debug)]
struct ImmediateSequence {
    sequence: u64,
    pages: Vec<u32>,
    start: usize,
    tokens: Vec<u32>,
}

/// Cold/scoring callers publish without accept; diagnostic decode may accept
/// that exact already-published wave, never an unrelated matching length.
#[derive(Default)]
struct ImmediateWave(RefCell<Option<Vec<ImmediateSequence>>>);
impl ImmediateWave {
    fn retire(&self) { self.0.borrow_mut().take(); }

    fn record(sequences: &[(&mut Placement, &[u32])]) -> Vec<ImmediateSequence> {
        sequences.iter().map(|(placement, tokens)| ImmediateSequence {
            sequence: placement.sequence, pages: placement.pages.clone(),
            start: placement.committed, tokens: tokens.to_vec(),
        }).collect()
    }

    fn accept(&self, verified: &[scheduler::Verified<'_, Placement, NoDraft, ()>]) -> Result<()> {
        let wave = self.0.borrow_mut().take().context("Step-5 accept has no submitted wave")?;
        ensure!(wave.len() == verified.len(), "Step-5 immediate accept inventory");
        for (submitted, actual) in wave.iter().zip(verified) {
            ensure!(submitted.sequence == actual.placement.sequence
                && submitted.pages == actual.placement.pages && submitted.start == actual.start
                && submitted.tokens == actual.rows,
                "Step-5 immediate accept differs from submitted wave");
            let end = submitted.start + submitted.tokens.len();
            let retained = actual.kept.is_some_and(|(rows, _)| rows == submitted.tokens.len())
                && actual.placement.committed == end;
            let released = actual.finished && actual.kept.is_none()
                && (submitted.start..=end).contains(&actual.placement.committed);
            ensure!(retained || released,
                "Step-5 immediate accept cannot retain a partial frontier");
        }
        Ok(())
    }
}

/// Input storage is admitted and retained outside the backend, like native lanes.
pub(super) struct Inputs<'a> {
    ids: Allocation<'a>,
    bf16: Allocation<'a>,
    residual: Allocation<'a>,
    rows: usize,
}
impl<'a> Inputs<'a> {
    pub(super) fn bytes(rows: usize) -> Result<u64> {
        ensure!((1..=4096).contains(&rows), "Step-5 input row capacity");
        Ok((rows * (4 + H * 6)) as u64)
    }

    pub(super) fn allocate(device: Device<'a>, rows: usize, available: u64) -> Result<Self> {
        ensure!(Self::bytes(rows)? <= available, "Step-5 input storage exceeds admission");
        Ok(Self { ids: Allocation::new(device, rows * 4)?,
            bf16: Allocation::new(device, rows * H * 2)?,
            residual: Allocation::new(device, rows * H * 4)?, rows })
    }
}

#[derive(Default)]
pub(super) struct NoDraft;
impl scheduler::DraftState for NoDraft {
    fn pending_rows(&self) -> usize { 0 }
    fn close(&mut self, _token: u32) {}
}

pub(super) struct Step5Backend<'e, 'a> {
    // Retire the Step before any borrowed native or input owner can disappear.
    pending: Pending<Evidence<'e, 'a>>,
    immediate: ImmediateWave,
    execution: &'e Execution<'e, 'a>,
    ranks: Vec<&'e RefCell<CacheRank<'a>>>,
    embedding: &'e TokenEmbedding<'a>,
    inputs: Vec<&'e Inputs<'a>>,
    profile: RefCell<[f64; 2]>,
    next_lane: Cell<usize>,
    rows: usize,
    context: usize,
    sequences: usize,
    library: &'a NativeLibrary,
}

impl<'e, 'a> Step5Backend<'e, 'a> {
    pub(super) fn new(execution: &'e Execution<'e, 'a>, ranks: Vec<&'e RefCell<CacheRank<'a>>>,
        embedding: &'e TokenEmbedding<'a>, inputs: Vec<&'e Inputs<'a>>) -> Result<Self> {
        execution.require_startup_graphs()?;
        ensure!(ranks.len() == execution.ownership().ranks()
            && inputs.len() == execution.lane_count(), "Step-5 backend owner inventory");
        let first = ranks.first().context("Step-5 backend missing cache")?.borrow();
        let context = usize::try_from(first.layout().context)?;
        let sequences = usize::try_from(first.sequences())?;
        let library = first.stream.device.library;
        drop(first);
        ensure!(sequences <= i32::MAX as usize && embedding.vocab() == V && embedding.hidden() == H,
            "Step-5 scheduler slots or embedding geometry differ from admission");
        let rows = inputs[0].rows;
        let entry = execution.ownership().execution().spec().entry_gpu as usize;
        for (lane, input) in inputs.iter().enumerate() {
            let stages = execution.lane_stages(lane)?;
            ensure!(input.rows == rows && input.residual.device.id == stages[entry].stream.device.id
                && std::ptr::eq(input.residual.device.library, library),
                "Step-5 input owner differs from admitted execution lane");
            execution.validate_lane_caches(lane, &ranks, rows)?;
        }
        Ok(Self { pending: Pending::default(), immediate: ImmediateWave::default(), execution, ranks, embedding, inputs,
            profile: RefCell::new([0.0; 2]), next_lane: Cell::new(0), rows, context, sequences, library })
    }

    fn execute(&self, sequences: &[(&mut Placement, &[u32])], pool: &RefPagePool,
        replay: bool) -> Result<(Evidence<'e, 'a>, DeviceLogits)> {
        self.execution.require_live()?;
        let lane = self.next_lane.get();
        self.next_lane.set((lane + 1) % self.inputs.len());
        let input = self.inputs[lane];
        let mut first_row = 0usize;
        let descriptors = sequences.iter().map(|(placement, tokens)| {
            let descriptor = SequenceRows { sequence: placement.sequence, start: first_row,
                rows: tokens.len(), base: placement.committed, pages: &placement.pages };
            first_row += tokens.len();
            descriptor
        }).collect::<Vec<_>>();
        ensure!(first_row <= input.rows, "Step-5 wave exceeds admitted input rows");
        // begin_wave validates the scheduler's actual COW pages before any input writes.
        let stages = self.execution.lane_stages(lane)?;
        let mut step = Step::begin_wave(self.ranks.clone(), stages.clone(), &descriptors, pool)?;
        let tokens = sequences.iter().flat_map(|(_, tokens)| tokens.iter().copied()).collect::<Vec<_>>();
        let entry = self.execution.ownership().execution().spec().entry_gpu as usize;
        step.compose(entry, input, |input, meta| meta.stream.device.run(|| {
            self.embedding.embed(&tokens, input.ids.buffer, 1, input.bf16.buffer, meta.stream.raw)?;
            // SAFETY: all three admitted input owners are retained by the backend;
            // the Step drains this stream before reuse, including on submission errors.
            unsafe {
                self.library.cuda_zero_bytes_async(input.residual.buffer, meta.rows * H * 4, meta.stream.raw)?;
                self.library.cuda_accumulate_bf16_to_f32_async(input.bf16.buffer,
                    input.residual.buffer, meta.rows * H, meta.stream.raw)
            }
        }))?;
        let _replay = self.execution.replay_scope(replay);
        self.execution.enqueue_selected(&[lane], &mut [&mut step], &[&input.residual], super::golden::request)?;
        let (device, output) = self.execution.lane_terminal_output(lane)?;
        let head_rank = self.execution.ownership().execution().spec().head_gpu as usize;
        let head = self.execution.lane_head(lane)?;
        let logits = step.compose(head_rank, head, |head, meta| {
            ensure!(device.id == meta.stream.device.id, "Step-5 final output/head device mismatch");
            head.device_logits_on(output, meta.rows, meta.stream)
        })?;
        Ok((step, logits))
    }

    fn immediate(&self, sequences: &mut [(&mut Placement, &[u32])], pool: &RefPagePool,
        replay: bool) -> Result<DeviceLogits> {
        self.pending.retire();
        self.immediate.retire();
        let submitted = ImmediateWave::record(sequences);
        let (step, logits) = self.execute(sequences, pool, replay)?;
        let mut accepted = sequences.iter_mut().map(|(placement, tokens)| (&mut **placement, tokens.len()))
            .collect::<Vec<_>>();
        step.publish_wave(&mut accepted)?;
        *self.immediate.0.borrow_mut() = Some(submitted);
        Ok(logits)
    }
}

impl Drop for Step5Backend<'_, '_> {
    fn drop(&mut self) { self.pending.retire(); }
}

impl<'e, 'a> Backend<'a> for Step5Backend<'e, 'a> {
    type Placement = Placement;
    type Draft = NoDraft;
    type History = ();
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = Frontier;
    type Prefix = Step5Prefix<'e, 'a, Evidence<'e, 'a>>;

    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry { family: "step5", vocab_size: V, max_context: self.context,
            slots: self.sequences, prefill_rows: self.rows, decode_rows: self.rows.min(64), full_prefill_logits: true }
    }
    fn message_starts(&self) -> &[&str] { &["<|im_start|>"] }
    fn library(&self) -> &'a NativeLibrary { self.library }
    fn profile(&self) -> &RefCell<[f64; 2]> { &self.profile }
    fn prefix_cache(&self, args: &PrefixArgs, lanes: usize)
        -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>, Frontier>)> {
        ensure!(args.prefix_partial == Toggle::Off, "Step-5 supports exact prefix frontiers only");
        let family = Step5Prefix::with_pending(self.ranks.clone(), self.pending.clone())?;
        let shape = self.geometry();
        let args = args.with_scoring(shape.decode_rows, shape.decode_rows, V)?;
        let layout = family.layout();
        let host = args.host_config(layout, shape.max_context, lanes)?.map(|config|
            CudaCopyEngine::registered_owned(self.library, family.snapshot_owners()).map(|copy| (config, copy))).transpose()?;
        let config = PrefixConfig { entries: args.prefix_cache_entries, mark_slots: family.slots(),
            keep_logits: true, min_tokens: args.prefix_cache_min_tokens };
        Ok((family, PrefixCache::new(layout, config, host)?))
    }
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        ensure!(job.media.is_empty(), "Step-5 bundled media encoder is not yet qualified");
        let keys = MediaKeys::new(&tokens, V as u32, &[])?;
        let media = RequestMedia::new(Vec::new(), H, tokens.len())?;
        Ok((scheduler::Prompt { job, tokens, keys }, media, Vec::new()))
    }
    fn probe_features(&self, prompt: &scheduler::Prompt, media: &mut RequestMedia,
        _cache: &mut EmbeddingCache, _snapshot: &Path) -> Result<()> {
        ensure!(prompt.job.media.is_empty() && media.spans().is_empty(), "Step-5 text target cannot accept media");
        Ok(())
    }
    fn coordinates(&self, _tokens: &[u32], job: &NativeRequest, media: &RequestMedia) -> Result<()> {
        ensure!(job.media.is_empty() && media.spans().is_empty(), "Step-5 text coordinates cannot accept media");
        Ok(())
    }
    fn placement(&self, pages: Vec<u32>, slot: i32) -> Placement {
        Placement { pages, sequence: slot as u64, committed: 0 }
    }
    fn bind_request(&self, _placement: &mut Placement, _coordinates: (), _media: RequestMedia,
        _tokens: &[u32], _resume: usize) { self.pending.retire(); self.immediate.retire(); }
    fn history(&self, _placement: &Placement) {}
    fn prefetch(&self, _tokens: &[u32], _next: &[u32]) {}
    fn require_live(&self) -> Result<()> {
        self.pending.retire(); self.immediate.retire(); self.execution.require_live()
    }
    fn log_table_stats(&self, _phase: &str) {}
    fn prefill_device(&self, placement: &mut Placement, tokens: &[u32], logit_rows: usize,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        self.pending.retire();
        ensure!(logit_rows <= tokens.len(), "Step-5 prefill logits extent");
        let mut logits = self.immediate(&mut [(placement, tokens)], pool, true)?;
        if logit_rows == 0 { return Ok(None); }
        logits.ptr = logits.ptr.wrapping_byte_add((logits.rows - logit_rows) * logits.stride * 4);
        logits.rows = logit_rows;
        Ok(Some(logits))
    }
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], _speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        self.pending.retire();
        self.immediate.retire();
        self.pending.verify(|| self.execute(sequences, pool, true)).map(Some)
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        self.pending.retire();
        self.immediate.retire();
        if speculative { self.pending.verify(|| self.execute(sequences, pool, false)).map(Some) }
        else { self.immediate(sequences, pool, false).map(Some) }
    }
    fn verify_bucket_rows(&self, rows: usize, _speculative: bool, _diagnostic: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value {
        serde_json::json!({"kind":"step5_swa512_csa64", "rows":rows, "contexts":contexts,
            "attention":"eager_per_sequence", "arithmetic":"startup_exact_keys"})
    }
    fn draft_geometry(&self, _snapshot: &Path, _width: usize) -> Result<cuteafd_core::PolicyGeometry> {
        anyhow::bail!("Step-5 native MTP layers 92..94 are not yet qualified")
    }
    fn arm_draft_probe(&self) -> Result<()> { anyhow::bail!("Step-5 MTP is not yet qualified") }
    fn prefill_draft(&self, _placement: &Placement, _start: usize, _chunk: &[u32], _next: Option<u32>,
        _draft: &mut NoDraft) -> Result<()> { anyhow::bail!("Step-5 MTP is not yet qualified") }
    fn draft(&self, _sequences: &mut [scheduler::DraftSeq<'_, Placement, NoDraft>],
        _timing: &mut scheduler::DraftTiming) -> Result<Vec<Vec<u32>>> { anyhow::bail!("Step-5 MTP is not yet qualified") }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, NoDraft, ()>],
        speculative: bool, draft: bool) -> Result<()> {
        let pending = self.pending.take();
        ensure!(!draft, "Step-5 MTP is not yet qualified");
        if let Some(step) = pending {
            let mut accepted = Vec::with_capacity(verified.len());
            for sequence in verified {
                let rows = sequence.kept.map_or(0, |(rows, _)| rows);
                ensure!(rows <= sequence.rows.len(), "Step-5 accepted rows exceed proposal");
                // A failed emission may advance the scheduler's logical length;
                // no accepted evidence means that sequence's native frontier stays put.
                if rows == 0 { sequence.placement.committed = sequence.start; }
                accepted.push((&mut *sequence.placement, rows));
            }
            step.publish_wave(&mut accepted)
        } else {
            ensure!(!speculative, "Step-5 speculative accept requires pending Step evidence");
            self.immediate.accept(verified)
        }
    }
    fn probe_begin(&self, _rows: usize) {}
    fn probe_end(&self) {}
    fn probe_finish<T>(&self, _observe: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immediate_accept_matches_consumes_and_retires_exact_wave() -> Result<()> {
        for fault in 0..8 {
            let wave = ImmediateWave::default();
            let mut placement = Placement { pages: vec![3], sequence: 2, committed: 7 };
            let tokens = [11, 13];
            *wave.0.borrow_mut() = Some(ImmediateWave::record(&[(&mut placement, &tokens)]));
            placement.committed = 9;
            let mut draft = NoDraft;
            let mut actual = scheduler::Verified { placement: &mut placement, seq: &mut draft,
                start: 7, history: (), rows: &tokens, first_row: 0, kept: Some((2, 17)), finished: false };
            match fault {
                1 => actual.placement.sequence = 1,
                2 => actual.placement.pages[0] = 4,
                3 => actual.start = 6,
                4 => actual.rows = &[11, 14],
                5 => actual.placement.committed = 8,
                6 => actual.kept = Some((1, 17)),
                7 => actual.kept = None,
                _ => (),
            }
            let accepted = wave.accept(&[actual]);
            assert_eq!(accepted.is_ok(), fault == 0, "fault {fault}");
            assert!(wave.accept(&[]).is_err(), "accept always consumes its wave");
        }
        // A terminal emission/grammar failure needs no retained frontier. Its
        // already-published state is released, never rolled back approximately.
        for committed in 6..=10 {
            let wave = ImmediateWave::default();
            let mut placement = Placement { pages: vec![3], sequence: 2, committed: 7 };
            *wave.0.borrow_mut() = Some(ImmediateWave::record(&[(&mut placement, &[11, 13])]));
            placement.committed = committed;
            let mut draft = NoDraft;
            let result = wave.accept(&[scheduler::Verified { placement: &mut placement, seq: &mut draft,
                start: 7, history: (), rows: &[11, 13], first_row: 0, kept: None, finished: true }]);
            assert_eq!(result.is_ok(), (7..=9).contains(&committed));
            assert!(wave.accept(&[]).is_err());
            *wave.0.borrow_mut() = Some(ImmediateWave::record(&[(&mut placement, &[17])]));
            wave.retire();
            assert!(wave.accept(&[]).is_err());
        }
        Ok(())
    }

    #[test]
    fn input_admission_checks_row_bounds_and_complete_bridge_storage() -> Result<()> {
        for rows in [1, 16, 4096] { assert_eq!(Inputs::bytes(rows)?, (rows * (4 + H * 6)) as u64); }
        for rows in [0, 4097, usize::MAX] { assert!(Inputs::bytes(rows).is_err()); }
        Ok(())
    }
}
