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

fn verify_kind(speculative: bool) -> cuteafd_transport::ExpertV2SourceKind {
    if speculative { cuteafd_transport::ExpertV2SourceKind::MtpVerify }
    else { cuteafd_transport::ExpertV2SourceKind::Decode }
}

fn ungraphed_kind(speculative: bool, kind: scheduler::UngraphedKind) -> cuteafd_transport::ExpertV2SourceKind {
    if speculative { return verify_kind(true); }
    match kind {
        scheduler::UngraphedKind::Scoring => cuteafd_transport::ExpertV2SourceKind::Prefill,
        scheduler::UngraphedKind::Decode => cuteafd_transport::ExpertV2SourceKind::Decode,
    }
}

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

/// Request-local state shared by every backend entry, without native-owner mocks.
struct BackendState<T> {
    pending: Pending<T>,
    immediate: ImmediateWave,
    fail_verify_once: Cell<bool>,
}
impl<T> BackendState<T> {
    fn new(fail_verify_once: bool) -> Self {
        Self { pending: Pending::default(), immediate: ImmediateWave::default(),
            fail_verify_once: Cell::new(fail_verify_once) }
    }

    fn retire(&self) { self.pending.retire(); self.immediate.retire(); }
    fn take_pending(&self) -> Option<T> { self.pending.take() }
    fn submit<O>(&self, work: impl FnOnce() -> Result<O>) -> Result<O> {
        self.retire();
        let result = work();
        if result.is_err() { self.retire(); }
        result
    }

    fn post_execution<O>(&self, completed: O) -> Result<O> {
        // Real target work succeeded. Dropping its Step drains before either
        // eager publication or pending evidence can advance the frontier.
        if self.fail_verify_once.replace(false) {
            tracing::warn!( "Step-5 injected post-execution verify failure");
            anyhow::bail!("Step-5 injected post-execution verify failure");
        }
        Ok(completed)
    }

    fn verify<O>(&self, work: impl FnOnce() -> Result<(T, O)>) -> Result<O> {
        self.submit(|| self.pending.verify(|| self.post_execution(work()?)))
    }
}
impl<T> Drop for BackendState<T> {
    fn drop(&mut self) { self.retire(); }
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
        cuteafd_loader::plan::families::step5::admission::input_bytes(rows)
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
    state: BackendState<Evidence<'e, 'a>>,
    execution: &'e Execution<'e, 'a>,
    ranks: Vec<&'e RefCell<CacheRank<'a>>>,
    embedding: &'e TokenEmbedding<'a>,
    inputs: &'e Inputs<'a>,
    profile: RefCell<[f64; 2]>,
    rows: usize,
    context: usize,
    sequences: usize,
    library: &'a NativeLibrary,
}

impl<'e, 'a> Step5Backend<'e, 'a> {
    pub(super) fn new(execution: &'e Execution<'e, 'a>, ranks: Vec<&'e RefCell<CacheRank<'a>>>,
        embedding: &'e TokenEmbedding<'a>, inputs: &'e Inputs<'a>) -> Result<Self> {
        execution.require_live()?;
        ensure!(ranks.len() == execution.ownership().ranks(), "Step-5 backend owner inventory");
        let first = ranks.first().context("Step-5 backend missing cache")?.borrow();
        let context = usize::try_from(first.layout().context)?;
        let sequences = usize::try_from(first.sequences())?;
        let library = first.stream.device.library;
        drop(first);
        ensure!(sequences <= i32::MAX as usize && embedding.vocab() == V && embedding.hidden() == H,
            "Step-5 scheduler slots or embedding geometry differ from admission");
        let rows = inputs.rows;
        let entry = execution.ownership().execution().spec().entry_gpu as usize;
        let entry_device = ranks[entry].borrow().stream.device;
        ensure!(embedding.placement() == crate::shared::token_io::EmbedPlacement::Gpu
            && embedding.owner().id == entry_device.id
            && std::ptr::eq(embedding.owner().library, entry_device.library),
            "Step-5 embedding must be GPU-resident on the admitted entry device/library");
        let stages = execution.stages();
        ensure!(inputs.residual.device.id == stages[entry].stream.device.id
            && std::ptr::eq(inputs.residual.device.library, library), "Step-5 input owner differs from execution");
        execution.validate_caches(&ranks, rows)?;
        let fail_verify_once = match std::env::var("CUTEAFD_DIAG_STEP5_FAIL_VERIFY_ONCE") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "1" => true,
            _ => anyhow::bail!("CUTEAFD_DIAG_STEP5_FAIL_VERIFY_ONCE must be unset or 1"),
        };
        Ok(Self { state: BackendState::new(fail_verify_once), execution, ranks, embedding, inputs,
            profile: RefCell::new([0.0; 2]), rows, context, sequences, library })
    }

    fn execute(&self, sequences: &[(&mut Placement, &[u32])], pool: &RefPagePool,
        kind: cuteafd_transport::ExpertV2SourceKind) -> Result<(Evidence<'e, 'a>, DeviceLogits)> {
        self.execution.require_live()?;
        let input = self.inputs;
        let mut first_row = 0usize;
        let descriptors = sequences.iter().map(|(placement, tokens)| {
            let descriptor = SequenceRows { sequence: placement.sequence, start: first_row,
                rows: tokens.len(), base: placement.committed, pages: &placement.pages };
            first_row += tokens.len();
            descriptor
        }).collect::<Vec<_>>();
        ensure!(first_row <= input.rows, "Step-5 wave exceeds admitted input rows");
        // begin_wave validates the scheduler's actual COW pages before any input writes.
        let stages = self.execution.stages();
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
        self.execution.enqueue(&mut step, &input.residual, kind)?;
        let (device, output) = self.execution.terminal_output()?;
        let head_rank = self.execution.ownership().execution().spec().head_gpu as usize;
        let head = self.execution.head()?;
        let logits = step.compose(head_rank, head, |head, meta| {
            ensure!(device.id == meta.stream.device.id, "Step-5 final output/head device mismatch");
            head.device_logits_on(output, meta.rows, meta.stream)
        })?;
        Ok((step, logits))
    }

    fn immediate(&self, sequences: &mut [(&mut Placement, &[u32])], pool: &RefPagePool,
        kind: cuteafd_transport::ExpertV2SourceKind) -> Result<DeviceLogits> {
        self.state.submit(|| {
            let submitted = ImmediateWave::record(sequences);
            let (step, logits) = self.state.post_execution(self.execute( sequences, pool, kind)?)?;
            let mut accepted = sequences.iter_mut().map(|(placement, tokens)| (&mut **placement, tokens.len()))
                .collect::<Vec<_>>();
            step.publish_wave(&mut accepted)?;
            *self.state.immediate.0.borrow_mut() = Some(submitted);
            Ok(logits)
        })
    }
}

impl Drop for Step5Backend<'_, '_> {
    fn drop(&mut self) { self.state.retire(); }
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
        let family = Step5Prefix::with_pending(self.ranks.clone(), self.state.pending.clone())?;
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
        _tokens: &[u32], _resume: usize) -> Result<()> { self.state.retire(); Ok(()) }
    fn history(&self, _placement: &Placement) {}
    fn prefetch(&self, _tokens: &[u32], _next: &[u32]) {}
    fn require_live(&self) -> Result<()> {
        self.state.retire(); self.execution.require_live()
    }
    fn log_table_stats(&self, _phase: &str) {}
    fn prefill_device(&self, placement: &mut Placement, tokens: &[u32], logit_rows: usize,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        self.state.retire();
        ensure!(logit_rows <= tokens.len(), "Step-5 prefill logits extent");
        let mut logits = self.immediate(&mut [(placement, tokens)], pool, cuteafd_transport::ExpertV2SourceKind::Prefill)?;
        if logit_rows == 0 { return Ok(None); }
        logits.ptr = logits.ptr.wrapping_byte_add((logits.rows - logit_rows) * logits.stride * 4);
        logits.rows = logit_rows;
        Ok(Some(logits))
    }
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        self.state.verify(|| self.execute( sequences, pool, verify_kind(speculative))).map(Some)
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        kind: scheduler::UngraphedKind, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        self.state.retire();
        let source = ungraphed_kind(speculative, kind);
        if speculative { self.state.verify(|| self.execute( sequences, pool, source)).map(Some) }
        else { self.immediate(sequences, pool, source).map(Some) }
    }
    fn verify_bucket_rows(&self, rows: usize, _speculative: bool, _diagnostic: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value {
        serde_json::json!({"kind":"step5_swa512_csa64", "rows":rows, "contexts":contexts,
            "attention":"eager_per_sequence", "arithmetic":"eager"})
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
        speculative: bool) -> Result<()> {
        let pending = self.state.take_pending();
        ensure!(!verified.iter().any(|row| row.neural), "Step-5 MTP is not yet qualified");
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
            self.state.immediate.accept(verified)
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
    fn ungraphed_source_metadata_preserves_scoring_decode_and_speculation() {
        use cuteafd_transport::ExpertV2SourceKind::{Decode, MtpVerify, Prefill};
        use scheduler::UngraphedKind;
        assert_eq!(ungraphed_kind(false, UngraphedKind::Scoring), Prefill);
        assert_eq!(ungraphed_kind(false, UngraphedKind::Decode), Decode);
        for kind in [UngraphedKind::Scoring, UngraphedKind::Decode] {
            assert_eq!(ungraphed_kind(true, kind), MtpVerify);
        }
        assert_eq!(verify_kind(false), Decode);
        assert_eq!(verify_kind(true), MtpVerify);
    }

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
                start: 7, history: (), rows: &tokens, first_row: 0, kept: Some((2, 17)), finished: false, neural: false };
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
                start: 7, history: (), rows: &[11, 13], first_row: 0, kept: None, finished: true, neural: false }]);
            assert_eq!(result.is_ok(), (7..=9).contains(&committed));
            assert!(wave.accept(&[]).is_err());
            *wave.0.borrow_mut() = Some(ImmediateWave::record(&[(&mut placement, &[17])]));
            wave.retire();
            assert!(wave.accept(&[]).is_err());
        }
        Ok(())
    }

    #[test]
    fn backend_failure_drains_before_reuse_and_frontier_publication() -> Result<()> {
        use std::rc::Rc;
        struct Evidence(Rc<Cell<usize>>);
        impl Drop for Evidence { fn drop(&mut self) { self.0.set(self.0.get() + 1); } }
        let drains = Rc::new(Cell::new(0));
        let state = BackendState::new(true);
        assert!(state.verify(|| Ok((Evidence(drains.clone()), 7))).is_err());
        assert_eq!(drains.get(), 1);
        assert!(state.take_pending().is_none());
        assert_eq!(state.verify(|| Ok((Evidence(drains.clone()), 8)))?, 8);
        assert_eq!(drains.get(), 1);
        drop(state.take_pending());
        assert_eq!(drains.get(), 2);
        let frontier = Cell::new(0);
        state.fail_verify_once.set(true);
        assert!(state.submit(|| {
            let evidence = state.post_execution(Evidence(drains.clone()))?;
            frontier.set(1);
            drop(evidence);
            Ok(())
        }).is_err());
        assert_eq!(frontier.get(), 0);
        assert_eq!(drains.get(), 3);
        state.submit(|| { frontier.set(1); Ok(()) })?;
        assert_eq!(frontier.get(), 1);
        Ok(())
    }
    #[test]
    fn failed_execution_does_not_consume_post_execution_injection() -> Result<()> {
        let state = BackendState::<()>::new(true);
        let failed: Result<()> = state.submit(|| anyhow::bail!("execution failed"));
        assert!(failed.is_err());
        assert!(state.fail_verify_once.get());
        assert!(state.verify(|| Ok(((), ()))).is_err());
        assert!(!state.fail_verify_once.get());
        state.verify(|| Ok(((), ())))?;
        Ok(())
    }

    #[test]
    fn input_admission_checks_row_bounds_and_complete_bridge_storage() -> Result<()> {
        for rows in [1, 16, 4096] { assert_eq!(Inputs::bytes(rows)?, (rows * (4 + H * 6)) as u64); }
        for rows in [0, 4097, usize::MAX] { assert!(Inputs::bytes(rows).is_err()); }
        Ok(())
    }
}
