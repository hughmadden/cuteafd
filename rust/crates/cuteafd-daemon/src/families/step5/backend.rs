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
        let rows = inputs[0].rows;
        let entry = execution.ownership().execution().spec().entry_gpu as usize;
        for (lane, input) in inputs.iter().enumerate() {
            let stages = execution.lane_stages(lane)?;
            ensure!(input.rows == rows && input.residual.device.id == stages[entry].stream.device.id
                && std::ptr::eq(input.residual.device.library, library),
                "Step-5 input owner differs from admitted execution lane");
            execution.validate_lane_caches(lane, &ranks, rows)?;
        }
        Ok(Self { pending: Pending::default(), execution, ranks, embedding, inputs,
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
        let (step, logits) = self.execute(sequences, pool, replay)?;
        let mut accepted = sequences.iter_mut().map(|(placement, tokens)| (&mut **placement, tokens.len()))
            .collect::<Vec<_>>();
        step.publish_wave(&mut accepted)?;
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
        _tokens: &[u32], _resume: usize) { self.pending.retire(); }
    fn history(&self, _placement: &Placement) {}
    fn prefetch(&self, _tokens: &[u32], _next: &[u32]) {}
    fn require_live(&self) -> Result<()> { self.pending.retire(); self.execution.require_live() }
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
        self.pending.verify(|| self.execute(sequences, pool, true)).map(Some)
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
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
            ensure!(!speculative && verified.iter().all(|sequence|
                sequence.placement.committed == sequence.start + sequence.rows.len()),
                "Step-5 accept without pending evidence requires an immediate ungraphed step");
            Ok(())
        }
    }
    fn probe_begin(&self, _rows: usize) {}
    fn probe_end(&self) {}
    fn probe_finish<T>(&self, _observe: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> { None }
}
