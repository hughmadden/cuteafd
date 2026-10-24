use super::{cache::Cache, execution::{BatchView, Execution, RequestView}, prefix::{MiniMaxPrefix, Placement},
    routed::Routed, transaction::Step, MiniMaxError};
use crate::shared::{draft::routes::RoundRoutes, prefix::{CudaCopyEngine, PrefixArgs, Toggle},
    qwen::{Encoder, scheduler::{self, Backend}}, token_io::{DeviceLogits, TokenEmbedding}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use cuteafd_ffi::{minimax::Buffer, NativeLibrary};
use cuteafd_loader::media::SpanExpander;
use std::{cell::RefCell, path::Path};

use super::dspark::State;

fn accept_step(step: Step, verified: &mut [scheduler::Verified<'_, Placement, State, ()>]) -> Result<()> {
    if verified.iter().any(|v| !step.matches(v.placement.sequence, v.start, v.rows)
        || (v.kept.is_none() && !v.finished)) {
        return Err(MiniMaxError::Invalid("accepted wave identity").into());
    }
    let mut publish = verified.iter_mut().map(|v| (&mut *v.placement, v.kept.map_or(0, |(n, _)| n))).collect::<Vec<_>>();
    step.publish(&mut publish)?;
    Ok(())
}

/// Drafter context rings by scheduler slot: a request holds one for its life
/// (bound at admission, released on cancel); requests admitted while every
/// ring is taken decode target-only, their draft state never ready.
#[derive(Debug)]
pub(super) struct Rings { owner: Vec<Option<usize>>, free: Vec<usize> }
impl Rings {
    pub fn new(slots: usize, rings: usize) -> Self { Self { owner: vec![None; slots], free: (0..rings).rev().collect() } }
    fn bind(&mut self, slot: usize) -> Result<()> {
        self.release(slot)?;
        if let Some(ring) = self.free.pop() { self.owner[slot] = Some(ring); }
        Ok(())
    }
    fn release(&mut self, slot: usize) -> Result<()> {
        let owner = self.owner.get_mut(slot).ok_or(MiniMaxError::Invalid("draft ring slot"))?;
        if let Some(ring) = owner.take() { self.free.push(ring); }
        Ok(())
    }
    fn of(&self, slot: usize) -> Option<usize> { self.owner.get(slot).copied().flatten() }
}

fn accept_context(verified: &mut [scheduler::Verified<'_, Placement, State, ()>], rings: &Rings,
    update: impl FnOnce(&[crate::shared::speculation::dflash::ContextRow])->Result<()>) -> Result<()> {
        let mut context=Vec::new();
        let mut states=Vec::new();
        for (index,v) in verified.iter().enumerate() {
            let Some((keep,after))=v.neural_kept() else {continue};
            let Some(ring)=rings.of(v.placement.sequence) else {continue};
            ensure!(keep>0 && keep<=v.rows.len(),MiniMaxError::Invalid("draft accepted prefix"));
            let mut state=v.seq.clone(); state.advance(v.start,keep,Some(after))?;
            context.extend((0..keep).map(|offset|crate::shared::speculation::dflash::ContextRow {
                tap_row:v.first_row+offset,slot:ring,position:v.start+offset }));
            states.push((index,state));
        }
        if !context.is_empty() {update(&context)?;}
        for (index,state) in states {*verified[index].seq=state;}
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
    pub probe: RefCell<RoundRoutes>,
    pub rings: RefCell<Rings>,
    pub full_prefill_logits: bool,
    pub trace: RefCell<Option<super::diagnostics::LogitTrace>>,
    /// Graph captures once the startup recordings finished: serving adds none.
    pub startup_captures: std::cell::Cell<Option<u64>>,
}
impl<'e, 'a> MiniMaxBackend<'e, 'a> {
    /// One step; the head projects only the batch rows in `logits`.
    fn execute(&self, sequences: &mut [(&mut Placement, &[u32])], pool: &RefPagePool,
        prefill: bool, logits: std::ops::Range<usize>) -> Result<(Step, Option<DeviceLogits>)> {
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
        let mut chunks = Vec::with_capacity(requests.len());
        for (request, (p, _)) in requests.iter().zip(sequences.iter()) {
            let mut chunk = cuteafd_engine::media::MediaChunk::default();
            p.media.write_chunk(p.committed, p.committed + request.rows, &mut chunk)?;
            chunks.push(chunk);
        }
        // The embedding is stream-ordered; only host feature copies (off the
        // stream) or a failed enqueue need the stream drained first.
        if embedded.is_err() || chunks.iter().any(|c| !c.features.is_empty()) { self.execution.drain()?; }
        embedded?;
        // Features are admitted host leases; injection precedes the device residual conversion.
        for (request, chunk) in requests.iter().zip(&chunks) {
            for (feature, &row) in chunk.features.chunks_exact(6144 * 2).zip(&chunk.indices) {
                let destination = self.embedded.view((request.row_start + row as usize) * 6144 * 2, 6144 * 2)?;
                library.copy_h2d(destination.raw(), feature)?;
            }
        }
        let batch = BatchView { positions, slots, requests, pool_pages: self.cache.pages };
        let started = std::time::Instant::now();
        let kind = if prefill { cuteafd_transport::ExpertV2SourceKind::Prefill }
            else { cuteafd_transport::ExpertV2SourceKind::Decode };
        let rows = logits.clone();
        let result = self.execution.execute(&self.embedded, &batch, &self.cache.layers, &mut *self.experts.borrow_mut(), kind, logits);
        let logits = result?;
        if let (Some(trace), Some(device)) = (self.trace.borrow_mut().as_mut(), logits.as_ref()) {
            trace.record(prefill, &batch.positions[rows], &device.to_host(library)?, device.vocab)?;
        }
        self.timing.borrow_mut()[usize::from(!prefill)] += started.elapsed().as_secs_f64();
        step.completed();
        Ok((step, logits))
    }
}

impl<'e, 'a> Backend<'a> for MiniMaxBackend<'e, 'a> {
    type Placement = Placement;
    type Draft = State;
    type History = ();
    type Coordinates = ();
    type Encoder = Encoder;
    type Meta = ();
    type Prefix = MiniMaxPrefix<'e, 'a>;

    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry { family: "minimax_m3", vocab_size: self.vocabulary, max_context: self.cache.context,
            slots: self.cache.sequences, prefill_rows: self.rows, decode_rows: self.rows.min(64),
            full_prefill_logits: self.full_prefill_logits }
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
        self.pending.borrow_mut().take(); p.media = media; p.committed = resume;
        if self.execution.draft().is_ok() { self.rings.borrow_mut().bind(p.sequence)?; }
        Ok(())
    }
    fn history(&self, _: &Placement) {}
    fn prefetch(&self, _: &[u32], _: &[u32]) {}
    fn require_live(&self) -> Result<()> {
        self.pending.borrow_mut().take();
        if let Some(startup) = self.startup_captures.get() {
            let current = cuteafd_ffi::graph_captures();
            ensure!(current == startup, "MiniMax zero-steady-capture contract: startup={startup}, current={current}");
        }
        self.execution.drain()
    }
    fn log_table_stats(&self, _: &str) {}
    fn prefill_device(&self, p: &mut Placement, tokens: &[u32], logit_rows: usize,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        if logit_rows > tokens.len() { return Err(MiniMaxError::Invalid("prefill logits extent").into()); }
        let (step, logits) = self.execute(&mut [(p, tokens)], pool, true, tokens.len() - logit_rows..tokens.len())?;
        step.publish(&mut [(p, tokens.len())])?;
        Ok(logits)
    }
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        let _ = speculative;
        let rows = sequences.iter().map(|(_, ids)| ids.len()).sum();
        let (step, logits) = self.execute(sequences, pool, false, 0..rows)?;
        *self.pending.borrow_mut() = Some(step);
        Ok(logits)
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        kind: scheduler::UngraphedKind, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        if matches!(kind, scheduler::UngraphedKind::Decode) { return self.verify_device(sequences, speculative, pool); }
        if speculative { return Err(MiniMaxError::Unsupported("speculative scoring").into()); }
        let rows = sequences.iter().map(|(_, ids)| ids.len()).sum();
        let (step, logits) = self.execute(sequences, pool, true, 0..rows)?;
        let mut publish = sequences.iter_mut().map(|(p, tokens)| (&mut **p, tokens.len())).collect::<Vec<_>>();
        step.publish(&mut publish)?;
        Ok(logits)
    }
    fn verify_bucket_rows(&self, rows: usize, _: bool, _: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value {
        serde_json::json!({"kind":"minimax_msa_paged", "rows":rows, "contexts":contexts,
            "attention":"causal_paged_decode_tiles", "arithmetic":"native_mxfp8_nvfp4a4"})
    }
    fn draft_geometry(&self, _: &Path, width: usize) -> Result<cuteafd_core::PolicyGeometry> {
        self.execution.draft()?;
        ensure!(width==7, MiniMaxError::Invalid("draft policy width"));
        // Until per-layer probes are qualified, all measured cost belongs to
        // the shared policy's round fit; no synthetic expert routes are priced.
        Ok(cuteafd_core::PolicyGeometry { layers:vec![cuteafd_core::LayerResource {
            class:None,slice_bytes:0.,group_rows:1,timed:false }], experts:128,topk:4,
            max_requests:self.rows.min(64),max_positions:7,widths:vec![7],
            classes:vec![cuteafd_core::ResourceClass::rtx()],group_rows_by_rows:vec![],regimes:1 })
    }
    fn arm_draft_probe(&self) -> Result<()> { self.execution.draft()?; Ok(()) }
    fn prefill_draft(&self, p: &Placement, start: usize, chunk: &[u32], next: Option<u32>, state: &mut State) -> Result<()> {
        ensure!(start.checked_add(chunk.len())==Some(p.committed),MiniMaxError::Invalid("draft prefill frontier"));
        // Without a ring the request never drafts: its state stays not ready.
        let Some(ring)=self.rings.borrow().of(p.sequence) else { return Ok(()) };
        let mut next_state=state.clone(); next_state.advance(start,chunk.len(),next)?;
        let rows=(0..chunk.len()).map(|tap_row|crate::shared::speculation::dflash::ContextRow {
            tap_row,slot:ring,position:start+tap_row }).collect::<Vec<_>>();
        self.execution.update_draft(&rows)?;
        *state=next_state; Ok(())
    }
    fn draft(&self, sequences: &mut [scheduler::DraftSeq<'_, Placement, State>], timing: &mut scheduler::DraftTiming) -> Result<Vec<Vec<u32>>> {
        let draft=self.execution.draft()?;
        let started=std::time::Instant::now();
        let mut output=vec![Vec::new();sequences.len()];
        let selected=sequences.iter().enumerate().filter(|(_,s)|s.depth>0)
            .map(|(index,s)| {
                ensure!(s.seq.ready && s.seq.end==s.placement.committed && s.depth<=7,
                    MiniMaxError::Invalid("draft continuation"));
                let slot=self.rings.borrow().of(s.placement.sequence).context("MiniMax drafting request holds no ring")?;
                Ok((index,crate::shared::speculation::dflash::DraftSeq {slot,
                    position:s.placement.committed,valid_from:s.seq.valid_from,
                    anchor:s.seq.anchor.context("MiniMax draft anchor missing")?}))
            }).collect::<Result<Vec<_>>>()?;
        for batch in selected.chunks(draft.layout.sequences) {
            // The block always executes seven rows; near the RoPE limit use
            // target-only verification rather than inventing future positions.
            let batch=batch.iter().filter(|(_,s)|s.position+7<=self.cache.context).copied().collect::<Vec<_>>();
            if batch.is_empty() {continue;}
            let proposals=self.execution.propose(&batch.iter().map(|(_,s)|*s).collect::<Vec<_>>(),self.embedding)?;
            ensure!(proposals.len()==batch.len(),MiniMaxError::Invalid("draft output count"));
            for ((index,_),proposal) in batch.into_iter().zip(proposals) {
                sequences[index].seq.confidence=proposal.confidence;
                output[index]=proposal.tokens.into_iter().take(sequences[index].depth).collect();
            }
            timing.steps+=1;
        }
        timing.seconds+=started.elapsed().as_secs_f64(); Ok(output)
    }
    fn draft_confidence<'s>(&self, state: &'s State) -> Option<&'s [f32]> { Some(&state.confidence) }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, State, ()>], _: bool) -> Result<()> {
        let step = self.pending.borrow_mut().take().context("MiniMax accept has no completed wave")?;
        accept_step(step, verified)?;
        accept_context(verified, &self.rings.borrow(), |rows|self.execution.update_draft(rows))
    }
    fn cancel(&self, p: &Placement) -> Result<()> {
        self.pending.borrow_mut().take(); self.execution.drain()?; self.rings.borrow_mut().release(p.sequence)
    }
    fn probe_begin(&self, rows: usize) {
        let mut probe=self.probe.borrow_mut(); probe.begin(rows); probe.dense(0).expect("one aggregate stage");
    }
    fn probe_end(&self) {}
    fn probe_finish<T>(&self, observe: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> {
        Some(observe(&self.probe.borrow(),&[None]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scheduler::Placement as _;

    #[test]
    fn minimax_fake_verifier_commits_only_the_accepted_taps() -> Result<()> {
        use scheduler::DraftState as _;
        let mut pool=RefPagePool::new(8,64);
        let mut p=Placement {pages:pool.alloc(2)?,sequence:0,committed:63,
            media:RequestMedia::new(Vec::new(),6144,0)?};
        let mut state=State::default();
        for keep in 1..=8 {
            p.committed=63;
            state.restored(63);
            state.close(10);
            assert_eq!(state.pending_rows(),0,"prefix restores never expose stale draft KV");
            let rows=[10,11,12,13,14,15,16,17];
            let mut step=Step::begin(&[(&mut p,&rows)],&pool,1,64,128,8)?;
            step.completed();
            p.committed+=keep;
            let mut verified=[scheduler::Verified {placement:&mut p,seq:&mut state,start:63,history:(),
                rows:&rows,first_row:9,kept:Some((keep,99)),finished:false,neural:true}];
            accept_step(step,&mut verified)?;
            let mut fake=Vec::new();
            let mut rings=Rings::new(1,1); rings.bind(0)?;
            accept_context(&mut verified,&rings,|rows| {
                fake.extend(rows.iter().map(|r|(r.tap_row,r.slot,r.position))); Ok(())
            })?;
            assert_eq!(fake,(0..keep).map(|i|(9+i,0,63+i)).collect::<Vec<_>>());
            assert_eq!((state.valid_from,state.end,state.anchor),(63,63+keep,Some(99)));
            assert_eq!(p.committed,63+keep);
            assert_eq!(state.pending_rows(),1);
        }
        Ok(())
    }

    #[test]
    fn minimax_failed_draft_update_does_not_publish_continuation() -> Result<()> {
        let mut p=Placement {pages:vec![],sequence:0,committed:2,
            media:RequestMedia::new(Vec::new(),6144,0)?};
        let mut state=State::default();
        let mut verified=[scheduler::Verified {placement:&mut p,seq:&mut state,start:0,history:(),
            rows:&[1,2,3],first_row:0,kept:Some((2,4)),finished:false,neural:true}];
        let mut rings=Rings::new(1,1); rings.bind(0)?;
        assert!(accept_context(&mut verified,&rings,|_|Err(MiniMaxError::Invalid("fake executor failure").into())).is_err());
        assert!(!state.ready);
        Ok(())
    }

    #[test]
    fn minimax_ringless_requests_never_become_draft_ready() -> Result<()> {
        let mut rings=Rings::new(3,1);
        rings.bind(2)?; rings.bind(0)?;
        assert_eq!((rings.of(2),rings.of(0)),(Some(0),None));
        let mut p=Placement {pages:vec![],sequence:0,committed:2,media:RequestMedia::new(Vec::new(),6144,0)?};
        let mut state=State::default();
        let mut verified=[scheduler::Verified {placement:&mut p,seq:&mut state,start:0,history:(),
            rows:&[1,2,3],first_row:0,kept:Some((2,4)),finished:false,neural:true}];
        accept_context(&mut verified,&rings,|_|panic!("a ringless request updated draft context"))?;
        assert!(!state.ready);
        rings.release(2)?; rings.bind(0)?;
        assert_eq!((rings.of(2),rings.of(0)),(None,Some(0)));
        rings.bind(0)?;
        assert_eq!(rings.of(0),Some(0),"rebinding a slot keeps one ring");
        assert!(rings.bind(3).is_err());
        Ok(())
    }

    #[test]
    fn minimax_scheduler_accepts_only_completed_matching_wave() -> Result<()> {
        let mut pool = RefPagePool::new(8, 64);
        let mut p = Placement { pages: pool.alloc(2)?, sequence: 0, committed: 63,
            media: RequestMedia::new(Vec::new(), 6144, 0)? };
        let mut draft = State::default();
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
