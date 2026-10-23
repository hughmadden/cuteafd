use super::{cache::Cache, execution::{BatchView, Execution, RequestView}, prefix::{MiniMaxPrefix, Placement},
    routed::Routed, transaction::Step, MiniMaxError};
use crate::shared::{draft::routes::RoundRoutes, prefix::{CudaCopyEngine, PrefixArgs, Toggle},
    qwen::{Encoder, scheduler::{self, Backend}}, token_io::{DeviceLogits, TokenEmbedding}};
use anyhow::{Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use cuteafd_ffi::{minimax::Buffer, NativeLibrary};
use cuteafd_loader::media::SpanExpander;
use std::{cell::RefCell, path::Path};

#[derive(Default)]
pub(super) struct NoDraft;
impl scheduler::DraftState for NoDraft {
    fn pending_rows(&self) -> usize { 0 }
    fn close(&mut self, _: u32) {}
}

fn accept_step(step: Step, verified: &mut [scheduler::Verified<'_, Placement, NoDraft, ()>]) -> Result<()> {
    if verified.iter().any(|v| !step.matches(v.placement.sequence, v.start, v.rows) || v.neural
        || (v.kept.is_none() && !v.finished)) {
        return Err(MiniMaxError::Invalid("accepted wave identity").into());
    }
    let mut publish = verified.iter_mut().map(|v| (&mut *v.placement, v.kept.map_or(0, |(n, _)| n))).collect::<Vec<_>>();
    step.publish(&mut publish)?;
    Ok(())
}

pub(super) struct MiniMaxBackend<'e, 'a> {
    pub execution: &'e Execution<'a>,
    pub cache: &'e Cache<'a>,
    pub embedding: &'e TokenEmbedding<'a>,
    pub token_ids: Buffer<'a>,
    pub embedded: Buffer<'a>,
    pub experts: RefCell<Routed<'a>>,
    pub rows: usize,
    pub vocabulary: usize,
    pub expander: Option<SpanExpander>,
    pub pending: RefCell<Option<Step>>,
    pub timing: RefCell<[f64; 2]>,
}
impl<'e, 'a> MiniMaxBackend<'e, 'a> {
    fn execute(&self, sequences: &mut [(&mut Placement, &[u32])], pool: &RefPagePool,
        prefill: bool) -> Result<(Step, DeviceLogits)> {
        self.pending.borrow_mut().take();
        let mut step = Step::begin(sequences, pool, self.cache.sequences, self.rows,
            self.cache.context, self.cache.pages)?;
        let mut positions = Vec::new();
        let mut slots = Vec::new();
        let mut requests = Vec::new();
        let mut tokens = Vec::new();
        for (p, ids) in sequences.iter() {
            requests.push(RequestView { row_start: tokens.len(), rows: ids.len(), start_position: p.committed,
                pages: p.pages.iter().map(|&p| p as i32).collect() });
            tokens.extend_from_slice(ids);
            for position in p.committed..p.committed + ids.len() {
                positions.push(position as i64);
                slots.push(i64::from(p.pages[position / 64]) * 64 + (position % 64) as i64);
            }
        }
        let library = self.cache.library;
        library.cuda_set_device(self.cache.device)?;
        let embedded = self.embedding.embed(&tokens, self.token_ids.raw(), 1, self.embedded.raw(), self.execution.stream());
        self.execution.drain()?;
        embedded?;
        // Features are admitted host leases; injection precedes the device residual conversion.
        for (request, (p, _)) in requests.iter().zip(sequences.iter()) {
            let mut chunk = cuteafd_engine::media::MediaChunk::default();
            p.media.write_chunk(p.committed, p.committed + request.rows, &mut chunk)?;
            for (feature, &row) in chunk.features.chunks_exact(6144 * 2).zip(&chunk.indices) {
                let destination = self.embedded.view((request.row_start + row as usize) * 6144 * 2, 6144 * 2)?;
                library.copy_h2d(destination.raw(), feature)?;
            }
        }
        let batch = BatchView { positions, slots, requests, pool_pages: self.cache.pages };
        let started = std::time::Instant::now();
        let kind = if prefill { cuteafd_transport::ExpertV2SourceKind::Prefill }
            else { cuteafd_transport::ExpertV2SourceKind::Decode };
        let result = self.execution.execute(&self.embedded, &batch, &self.cache.layers, &mut *self.experts.borrow_mut(), kind);
        let logits = result?;
        self.timing.borrow_mut()[usize::from(!prefill)] += started.elapsed().as_secs_f64();
        step.completed();
        Ok((step, logits))
    }
}

impl<'e, 'a> Backend<'a> for MiniMaxBackend<'e, 'a> {
    type Placement = Placement;
    type Draft = NoDraft;
    type History = ();
    type Coordinates = ();
    type Encoder = Encoder;
    type Meta = ();
    type Prefix = MiniMaxPrefix<'e, 'a>;

    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry { family: "minimax_m3", vocab_size: self.vocabulary, max_context: self.cache.context,
            slots: self.cache.sequences, prefill_rows: self.rows, decode_rows: self.rows.min(64), full_prefill_logits: true }
    }
    fn message_starts(&self) -> &[&str] { &["]~b]"] }
    fn library(&self) -> &'a NativeLibrary { self.cache.library }
    fn profile(&self) -> &RefCell<[f64; 2]> { &self.timing }
    fn prefix_cache(&self, args: &PrefixArgs, lanes: usize)
        -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)> {
        if args.prefix_partial != Toggle::Off { return Err(MiniMaxError::Unsupported("partial prefix reuse").into()); }
        let family = MiniMaxPrefix { cache: self.cache };
        let layout = family.layout();
        let args = args.with_scoring(self.rows, self.rows.min(64), self.vocabulary)?;
        let host = args.host_config(layout, self.cache.context, lanes)?.map(|config|
            CudaCopyEngine::registered(self.cache.library, &self.cache.buffers()).map(|copy| (config, copy))).transpose()?;
        let config = PrefixConfig { entries: args.prefix_cache_entries, mark_slots: 0,
            keep_logits: true, min_tokens: args.prefix_cache_min_tokens };
        Ok((family, PrefixCache::new(layout, config, host)?))
    }
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        super::media::prepare(job, tokens, self.expander.as_ref(), self.vocabulary, self.cache.context)
    }
    fn probe_features(&self, _: &scheduler::Prompt, _: &mut RequestMedia,
        _: &mut EmbeddingCache, _: &Path) -> Result<()> { Ok(()) }
    fn coordinates(&self, _: &[u32], _: &NativeRequest, _: &RequestMedia) -> Result<()> { Ok(()) }
    fn placement(&self, pages: Vec<u32>, slot: i32) -> Placement {
        Placement { pages, sequence: slot as usize, committed: 0,
            media: RequestMedia::new(Vec::new(), 6144, 0).expect("fixed text geometry") }
    }
    fn bind_request(&self, p: &mut Placement, _: (), media: RequestMedia, _: &[u32], resume: usize) -> Result<()> {
        self.pending.borrow_mut().take(); p.media = media; p.committed = resume; Ok(())
    }
    fn history(&self, _: &Placement) {}
    fn prefetch(&self, _: &[u32], _: &[u32]) {}
    fn require_live(&self) -> Result<()> { self.pending.borrow_mut().take(); self.execution.drain() }
    fn log_table_stats(&self, _: &str) {}
    fn prefill_device(&self, p: &mut Placement, tokens: &[u32], logit_rows: usize,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        if logit_rows > tokens.len() { return Err(MiniMaxError::Invalid("prefill logits extent").into()); }
        let (step, mut logits) = self.execute(&mut [(p, tokens)], pool, true)?;
        step.publish(&mut [(p, tokens.len())])?;
        if logit_rows == 0 { return Ok(None); }
        logits.ptr = logits.ptr.wrapping_byte_add((logits.rows - logit_rows) * logits.stride * 4);
        logits.rows = logit_rows;
        Ok(Some(logits))
    }
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        if speculative { return Err(MiniMaxError::Unsupported("speculation; use --speculator off").into()); }
        let (step, logits) = self.execute(sequences, pool, false)?;
        *self.pending.borrow_mut() = Some(step);
        Ok(Some(logits))
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        kind: scheduler::UngraphedKind, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        if matches!(kind, scheduler::UngraphedKind::Decode) { return self.verify_device(sequences, speculative, pool); }
        if speculative { return Err(MiniMaxError::Unsupported("speculative scoring").into()); }
        let (step, logits) = self.execute(sequences, pool, true)?;
        let mut publish = sequences.iter_mut().map(|(p, tokens)| (&mut **p, tokens.len())).collect::<Vec<_>>();
        step.publish(&mut publish)?;
        Ok(Some(logits))
    }
    fn verify_bucket_rows(&self, rows: usize, _: bool, _: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value {
        serde_json::json!({"kind":"minimax_msa_paged", "rows":rows, "contexts":contexts,
            "attention":"causal_paged_decode_tiles", "arithmetic":"native_mxfp8_nvfp4a4"})
    }
    fn draft_geometry(&self, _: &Path, _: usize) -> Result<cuteafd_core::PolicyGeometry> {
        Err(MiniMaxError::Unsupported("speculation").into())
    }
    fn arm_draft_probe(&self) -> Result<()> { Err(MiniMaxError::Unsupported("speculation").into()) }
    fn prefill_draft(&self, _: &Placement, _: usize, _: &[u32], _: Option<u32>, _: &mut NoDraft) -> Result<()> {
        Err(MiniMaxError::Unsupported("speculation").into())
    }
    fn draft(&self, _: &mut [scheduler::DraftSeq<'_, Placement, NoDraft>], _: &mut scheduler::DraftTiming) -> Result<Vec<Vec<u32>>> {
        Err(MiniMaxError::Unsupported("speculation").into())
    }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, NoDraft, ()>], speculative: bool) -> Result<()> {
        if speculative { return Err(MiniMaxError::Unsupported("speculation").into()); }
        let step = self.pending.borrow_mut().take().context("MiniMax accept has no completed wave")?;
        accept_step(step, verified)
    }
    fn cancel(&self, _: &Placement) -> Result<()> { self.pending.borrow_mut().take(); self.execution.drain() }
    fn probe_begin(&self, _: usize) {}
    fn probe_end(&self) {}
    fn probe_finish<T>(&self, _: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scheduler::Placement as _;

    #[test]
    fn minimax_scheduler_accepts_only_completed_matching_wave() -> Result<()> {
        let mut pool = RefPagePool::new(8, 64);
        let mut p = Placement { pages: pool.alloc(2)?, sequence: 0, committed: 63,
            media: RequestMedia::new(Vec::new(), 6144, 0)? };
        let mut draft = NoDraft;
        for (completed, token, kept, finished, succeeds) in [
            (false, 7, Some((1, 8)), false, false),
            (true, 9, Some((1, 8)), false, false),
            (true, 7, None, false, false),
            (true, 7, None, true, true),
            (true, 7, Some((1, 8)), false, true),
        ] {
            p.committed = 63;
            let mut step = Step::begin(&[(&mut p, &[7])], &pool, 2, 64, 128, 8)?;
            if completed { step.completed(); }
            p.set_len(64);
            let result = accept_step(step, &mut [scheduler::Verified { placement: &mut p,
                seq: &mut draft, start: 63, history: (), rows: &[token], first_row: 0,
                kept, finished, neural: false }]);
            assert_eq!(result.is_ok(), succeeds);
            if succeeds { assert_eq!(p.committed, if kept.is_some() {64} else {63}); }
        }
        // A failed/cancelled wave leaves append bytes outside the committed prefix;
        // retrying the same position requires fresh completion evidence.
        p.committed = 63;
        let abandoned = Step::begin(&[(&mut p, &[11])], &pool, 2, 64, 128, 8)?;
        drop(abandoned);
        let mut retry = Step::begin(&[(&mut p, &[12])], &pool, 2, 64, 128, 8)?;
        retry.completed();
        retry.publish(&mut [(&mut p, 1)])?;
        assert_eq!(p.committed, 64);
        Ok(())
    }
}
