//! One device, one append-only paged MSA pipeline, native checkpoint arithmetic.
use super::{MiniMaxError, diagnostics::{Diagnostics, StepDump}, ffn, head::TargetHeadOwner, native::Native,
    scratch::{Rows, Scratch}, weights::Weights};
use crate::shared::{memory::device::{Allocation, Device}, prefix::view, token_io::DeviceLogits};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{programs::{Programs,Scalar}, CuteafdDeviceBuffer};
use cuteafd_transport::ExpertV2SourceKind;
use cuteafd_loader::{families::minimax_m3::{cache::PAGE_ROWS,config::MiniMaxM3Config},
    plan::{checkpoint::Checkpoint,families::minimax::admission::{RuntimeLedger,RuntimeSizing}},
    serving_capacity::prefill_lanes as lane_plans};

pub(super) struct LayerCache<'a> { pub key: Allocation<'a>, pub value: Allocation<'a>, pub index: Allocation<'a> }
#[derive(Debug,Clone)]
pub(super) struct RequestView { pub row_start:usize, pub rows:usize, pub start_position:usize, pub pages:Vec<i32> }
/// A step's rows as the kernels see them, built from a wave the shared
/// append checks validated ([`super::transaction::Step::batch`]).
#[derive(Debug,Clone)]
pub(super) struct BatchView { pub positions:Vec<i64>, pub slots:Vec<i64>, pub requests:Vec<RequestView>, pub pool_pages:usize }

impl BatchView {
    /// The kernel ABI's limits on the validated wave: rows within the step's
    /// capacity, at most 64 requests, pool page ids in i32, every request
    /// inside the RoPE table.
    pub fn validate(&self, capacity:usize, context:usize) -> Result<usize> {
        let rows = self.positions.len();
        ensure!(rows>0 && rows<=capacity && self.slots.len()==rows && !self.requests.is_empty()
            && self.requests.len()<=64 && self.pool_pages>0 && self.pool_pages<=i32::MAX as usize
            && self.requests.iter().all(|r|r.start_position.checked_add(r.rows).is_some_and(|end|end<=context)),
            MiniMaxError::Invalid("batch extent"));
        Ok(rows)
    }
    fn request(&self,row:usize)->Result<&RequestView> {
        self.requests.iter().find(|r|row>=r.row_start && row<r.row_start+r.rows).context("MiniMax row owner")
    }
}

/// Rows `first..first + rows` of request `request` (step rows from `row`) as
/// one extend launch; `slot` indexes its table, length and offset entries,
/// `bounds` its causal key ends in the index bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Launch { request:usize, row:usize, first:usize, rows:usize, slot:usize, bounds:usize }

/// A step's attention metadata, uploaded once before the layer loop: page
/// tables do not change between layers. Decode steps give every row its own
/// table for the decode programs; prefill steps tile each request through
/// the prefill scorer and the extend kernel.
#[derive(Debug, PartialEq, Eq)]
enum AttentionPlan { Decode { stride:usize }, Extend { launches:Vec<Launch> } }

/// Decode metadata: per-row tables at a fixed `stride` (-1 padded), lengths
/// and packed offsets. The decode kernels visit only each row's live pages
/// (its cache length), so a wider table changes neither work nor results,
/// and every decode step's launches keep one shape at any context.
fn decode_tables(batch:&BatchView,stride:usize)->Result<(Vec<i32>,Vec<i32>,Vec<i32>)> {
    let rows=batch.positions.len();
    let mut tables=vec![-1i32;rows*stride];
    let mut lengths=Vec::with_capacity(rows);
    for row in 0..rows {
        let request=batch.request(row)?;
        let length=batch.positions[row] as usize+1;
        let width=length.div_ceil(64);
        ensure!(width<=stride,MiniMaxError::Invalid("live page-table width"));
        tables[row*stride..row*stride+width].copy_from_slice(&request.pages[..width]);
        lengths.push(length as i32);
    }
    Ok((tables,lengths,(0..=rows as i32).collect()))
}

/// Extend metadata: one launch per request tile of at most `tile` rows, and
/// no tile across a prefill lane's first row (`cuts`, batch rows). Each
/// launch's table row holds its request's pages through the step (padded
/// with -1 to `stride`), its 16-byte entries hold the cache length after its
/// rows and its packed offsets [0, rows], and `bounds` its rows' causal key
/// ends after the `tile` leading zero key starts.
fn extend_tables(batch:&BatchView,tile:usize,stride:usize,cuts:&[usize])->Result<(Vec<Launch>,Vec<i32>,Vec<i32>,Vec<i32>,Vec<i32>)> {
    let (mut launches,mut tables,mut lengths,mut offsets,mut bounds)=(Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new());
    for (request,view) in batch.requests.iter().enumerate() {
        let end=view.start_position+view.rows;
        let width=end.div_ceil(64);
        ensure!(width<=stride,MiniMaxError::Invalid("extend page-table width"));
        let mut first=0;
        while first<view.rows {
            let cut=cuts.iter().map(|&c|c.saturating_sub(view.row_start)).find(|&c|c>first).unwrap_or(view.rows);
            let rows=(view.rows-first).min(tile).min(cut-first);
            launches.push(Launch{request,row:view.row_start+first,first,rows,slot:launches.len(),bounds:tile+bounds.len()});
            tables.extend_from_slice(&view.pages[..width]);
            tables.resize(launches.len()*stride,-1);
            lengths.extend([(view.start_position+first+rows) as i32,0,0,0]);
            offsets.extend([0,rows as i32,0,0]);
            bounds.extend((0..rows).map(|i|(view.start_position+first+i+1) as i32));
            bounds.resize(bounds.len().next_multiple_of(4),0);
            first+=rows;
        }
    }
    Ok((launches,tables,lengths,offsets,bounds))
}

fn le_bytes(values:&[i32])->Vec<u8> { values.iter().flat_map(|v|v.to_le_bytes()).collect() }

/// The step scratch every step uploads into ([`Execution::step`], `plan_attention`; DSpark's
/// updates and drafts upload `positions` too), besides the extend bounds' tail
/// ([`extend_bounds`]): their ordered-upload staging is reserved before readiness, at each
/// destination's whole extent.
const STEP_UPLOADS: [&str; 5] = ["positions","slots","page_table","cache_lengths","cu_seqlens_q"];

/// The extend launches' causal key ends: `index_bounds` after the `tile` leading zero key
/// starts, to the buffer's end (one ordered-upload destination for every step).
fn extend_bounds(bounds:CuteafdDeviceBuffer,tile:usize)->Result<CuteafdDeviceBuffer> {
    let start=tile.checked_mul(4).context("MiniMax extend bounds offset")?;
    view(bounds,start,bounds.bytes.checked_sub(start).context("MiniMax extend bounds admission")?)
}

/// The lane cuts (batch rows) of a step of `rows` rows in up to `lanes` lanes
/// of `lane_rows`: one lane takes at most a program launch's `program_rows`.
fn step_cuts(rows:usize,lanes:usize,lane_rows:usize,program_rows:usize,diagnostic:bool)->Result<Vec<usize>> {
    if lanes<2 {
        ensure!(rows<=program_rows,MiniMaxError::Invalid(if diagnostic {"a diagnostic runs one lane: --prefill-lanes 1"}
            else {"one-lane step rows"}));
        return Ok(Vec::new());
    }
    let sizes=lane_plans::lane_cuts(rows,lanes,lane_rows,PAGE_ROWS as usize)?;
    Ok(sizes.iter().scan(0,|at,&n|{*at+=n; Some(*at)}).take(sizes.len()-1).collect())
}

/// Batch rows `first..first + rows` of a step and their extend launches (a
/// range of the plan's; empty for a decode plan): one prefill lane today, and
/// one prefill unit or a whole one-pass step once units run on the shared
/// executor.
#[derive(Debug, Clone, PartialEq, Eq)]
struct UnitTiles { first:usize, rows:usize, launches:std::ops::Range<usize> }

/// The row pieces of a step cut at `cuts` (each piece's first row after the
/// first), with the launches of `plan` each holds (extend tiles never cross a cut).
fn unit_tiles(plan:&AttentionPlan,rows:usize,cuts:&[usize])->Vec<UnitTiles> {
    let launches=match plan { AttentionPlan::Extend{launches}=>launches.as_slice(), AttentionPlan::Decode{..}=>&[] };
    let mut bounds=vec![0]; bounds.extend_from_slice(cuts); bounds.push(rows);
    let mut at=0;
    bounds.windows(2).map(|w| {
        let start=at;
        while at<launches.len() && launches[at].row<w[1] { at+=1; }
        UnitTiles{first:w[0],rows:w[1]-w[0],launches:start..at}
    }).collect()
}

/// What a step queues next, lane by lane ([`Execution::layers`] runs exactly this order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Visit {
    /// A dense layer, whole, on every lane in turn.
    Dense(usize),
    /// A lane collects routed layer `layer`'s experts: the shared and routed sums into the
    /// residual, and the drafter's tap.
    Collect { layer:usize, lane:usize },
    /// A lane runs routed layer `layer` through attention, the router, the expert dispatch and
    /// the shared expert.
    Dispatch { layer:usize, lane:usize },
}

/// A step's visits through `last_layer` on `lanes` lanes. A routed layer's lane collects the
/// previous routed layer's wave right before dispatching its own, so one lane's exchange is in
/// flight while another lane's GPU work runs; a dense layer, and the step's end, first collect
/// every lane's wave. Waves are therefore collected in dispatch order with at most `lanes` in
/// flight: one exchange link with a return slot per lane carries them.
fn visits(moe:&[bool],last_layer:usize,lanes:usize)->Vec<Visit> {
    let mut visits=Vec::new();
    let mut pending=None;
    for (layer,&routed) in moe.iter().enumerate().take(last_layer+1) {
        if !routed {
            if let Some(previous)=pending.take() { visits.extend((0..lanes).map(|lane|Visit::Collect{layer:previous,lane})); }
            visits.push(Visit::Dense(layer));
            continue;
        }
        for lane in 0..lanes {
            if let Some(previous)=pending { visits.push(Visit::Collect{layer:previous,lane}); }
            visits.push(Visit::Dispatch{layer,lane});
        }
        pending=Some(layer);
    }
    if let Some(previous)=pending { visits.extend((0..lanes).map(|lane|Visit::Collect{layer:previous,lane})); }
    visits
}

/// The drafter an execution taps and drafts with: the bundled DSpark or a DFlash2 checkpoint
/// on the shared drafter.
pub(super) enum Drafter<'a> {
    Dspark(super::dspark::Dspark<'a>),
    Dflash(super::dflash::Dflash<'a>),
}
impl<'a> Drafter<'a> {
    /// Sequences one draft step takes.
    pub fn sequences(&self) -> usize {
        match self { Self::Dspark(d) => d.layout.sequences, Self::Dflash(d) => d.sequences() }
    }
    /// Rows of the block one draft runs at its anchor's position and after (DSpark 7, DFlash2 8).
    pub fn block(&self) -> usize {
        match self { Self::Dspark(_) => 7, Self::Dflash(d) => d.block() }
    }
    /// Whether drafts carry a confidence head's per-token acceptance (DSpark).
    pub fn confidence(&self) -> bool { matches!(self, Self::Dspark(_)) }
    /// The context rows a single-request `rows`-row step from `start` leaves for ring `slot`.
    pub fn prefill_rows(&self, start: usize, rows: usize, slot: usize) -> Vec<crate::shared::speculation::dflash::ContextRow> {
        match self {
            Self::Dspark(_) => (0..rows).map(|tap_row| crate::shared::speculation::dflash::ContextRow {
                tap_row, slot, position: start + tap_row }).collect(),
            Self::Dflash(d) => d.prefill_rows(start, rows, slot),
        }
    }
    /// Taps batch rows `first..first + rows` of a `span`-row step at `layer` (`residual`: their view).
    fn tap(&self, native: &Native<'a>, layer: usize, residual: CuteafdDeviceBuffer, first: usize, rows: usize,
        span: usize) -> Result<()> {
        match self {
            Self::Dspark(d) => d.tap(native, layer, residual, first, rows),
            Self::Dflash(d) => d.tap(native, layer, residual, first, rows, span),
        }
    }
}

pub(super) trait ExpertExecutor<'a> {
    /// Before a step's layers: `routed` routed layers, up to `slots` waves in flight (one per slot), none wider
    /// than `rows` rows.
    fn begin(&mut self, _routed:usize, _slots:usize, _rows:usize, _kind:ExpertV2SourceKind) -> Result<()> { Ok(()) }
    /// Waves that can be in flight at once (the Spark link's return slots).
    fn slots(&self) -> usize { 1 }
    /// Sends routed layer `layer`'s wave of `rows` rows from slot `slot`'s rows.
    fn dispatch(&mut self, layer:usize, slot:usize, rows:usize, ids:CuteafdDeviceBuffer, weights:CuteafdDeviceBuffer,
        input:CuteafdDeviceBuffer, native:&Native<'a>) -> Result<()>;
    /// Lands slot `slot`'s wave of routed layer `layer` into `output`.
    fn collect(&mut self, layer:usize, slot:usize, rows:usize, output:CuteafdDeviceBuffer, native:&Native<'a>) -> Result<()>;
    /// After a step's drain: no wave outstanding, and the expert wire healthy.
    fn finish(&mut self) -> Result<()> { Ok(()) }
    /// A failed step, before its stream drains: stops the expert exchange, which releases every
    /// wait the stream may still hold; the owner never runs again.
    fn abort(&mut self) {}
    /// After [`Self::abort`] and a completed drain: releases what the exchange retained for it.
    fn release(&mut self) {}
}

pub(super) struct Execution<'a> {
    // Drain queued consumers before their programs and allocations are dropped.
    native: Native<'a>,
    head: TargetHeadOwner<'a>,
    weights: Weights<'a>,
    scratch: Scratch<'a>,
    cfg: MiniMaxM3Config,
    /// Rows of the widest step (a prefill chunk of `lanes` lanes of `lane_rows`).
    rows:usize,
    lane_rows:usize,
    lanes:usize,
    /// The most rows a program launch takes (the export's row capacity).
    program_rows:usize,
    logit_rows:usize,
    context:usize,
    decode_rows:usize,
    extend_rows:usize,
    table_stride:usize,
    poisoned:std::cell::Cell<bool>,
    diagnostics:std::cell::RefCell<Option<Diagnostics>>,
    draft:Option<Drafter<'a>>,
}
impl<'a> Execution<'a> {
    pub fn load(device:Device<'a>, programs:&'a Programs<'a>, checkpoint:&Checkpoint,
        cfg:&MiniMaxM3Config, ledger:&RuntimeLedger, sizing:RuntimeSizing) -> Result<Self> {
        cfg.release_geometry()?;
        let diagnostics=std::cell::RefCell::new(Diagnostics::from_env(cfg.layers)?);
        ensure!(cfg.rms_norm_eps==1e-6 && cfg.swiglu_alpha==1.702 && cfg.swiglu_limit==7.0
            && cfg.routed_scale==2.0,MiniMaxError::Unsupported("compiled norm, activation or route arithmetic"));
        ensure!(cfg.moe[..3].iter().all(|v|!*v) && cfg.moe[3..].iter().all(|v|*v),
            MiniMaxError::Unsupported("dense/routed layer ordering"));
        let rows=usize::try_from(ledger.bytes("norm").context("MiniMax norm admission")? / (6144*2))?;
        let logit_rows=usize::try_from(ledger.bytes("logits").context("MiniMax logits admission")? / (cfg.vocab_size as u64*4))?;
        let context=usize::try_from(ledger.bytes("cos").context("MiniMax rope admission")? / 64)?;
        let native=Native::new(device,programs)?;
        let decode_rows=native.capacity("minimax_index_decode")?;
        ensure!(decode_rows>0 && decode_rows<=64 && native.capacity("minimax_attention_decode")?==decode_rows,
            MiniMaxError::Invalid("index/attention capacity stride"));
        let extend_rows=native.capacity("minimax_index_prefill")?;
        ensure!(extend_rows>0 && extend_rows%4==0 && native.capacity("minimax_attention_extend")?==extend_rows,
            MiniMaxError::Invalid("prefill index/attention capacity stride"));
        let program_rows=native.capacity("minimax_norm")?;
        ensure!(rows==sizing.slot_rows() && sizing.unit_rows<=program_rows,MiniMaxError::Invalid("prefill lane rows"));
        let scratch=Scratch::new(device,ledger)?;
        let table_stride=usize::try_from(cuteafd_loader::plan::families::minimax::admission::extend_table_stride(context as u64))?;
        ensure!(scratch.get("page_table")?.bytes>=decode_rows*table_stride*4,MiniMaxError::Invalid("page-table admission"));
        let (weights,head)=device.run(|| {
            let weights=Weights::load(&native,checkpoint)?;
            native.launch("minimax_rope", &[scratch.get("cos")?,scratch.get("sin")?], &[
                Scalar::I32(context as i32),Scalar::F32(cfg.rope_theta as f32),Scalar::F32(1.0),
                Scalar::F32(0.0),Scalar::F32(1.0),Scalar::F32(1.0)])?;
            let ones=vec![1.0f32.to_le_bytes();decode_rows*4].concat();
            native.upload(scratch.get("k_descale")?,&ones)?;
            native.upload(scratch.get("v_descale")?,&ones)?;
            native.upload(scratch.get("batch_ids")?,&vec![0;rows*4])?;
            // Extend launches read their key starts (all zero) from here.
            native.upload(scratch.get("index_bounds")?,&vec![0;extend_rows*4])?;
            for name in STEP_UPLOADS { native.reserve(scratch.get(name)?)?; }
            native.reserve(extend_bounds(scratch.get("index_bounds")?,extend_rows)?)?;
            native.drain()?;
            Ok((weights,TargetHeadOwner::new(&native,&scratch,logit_rows,cfg.vocab_size)?))
        })?;
        Ok(Self{native,head,weights,scratch,cfg:cfg.clone(),rows,lane_rows:sizing.unit_rows,lanes:sizing.waves,program_rows,
            logit_rows,context,decode_rows,extend_rows,table_stride,
            poisoned:std::cell::Cell::new(false),diagnostics,draft:None})
    }
    pub fn load_draft(&mut self, contract:&cuteafd_loader::families::minimax_m3::dspark::DsparkContract,
        layout:super::dspark::Layout)->Result<()> {
        // The released drafter attends its block noncausally from the anchor.
        ensure!(contract.sample_from_anchor && !contract.causal,MiniMaxError::Unsupported("draft mask or anchor mode"));
        self.draft=Some(Drafter::Dspark(self.native.device().run(|| super::dspark::Dspark::load(&self.native,contract,layout))?));
        Ok(())
    }
    /// Loads a DFlash2 drafter (its checkpoint bytes `file`, the mask token's target embedding
    /// `mask_row`) onto this execution's stream.
    pub fn load_dflash(&mut self, contract:&super::dflash::Contract, file:Vec<u8>, mask_row:Vec<u8>)->Result<()> {
        self.draft=Some(Drafter::Dflash(self.native.device().run(||
            super::dflash::Dflash::load(self.native.library(),self.native.raw(),contract,file,mask_row))?));
        Ok(())
    }
    pub fn draft(&self)->Result<&Drafter<'a>> { self.draft.as_ref().context("MiniMax drafter not admitted") }
    pub fn update_draft(&self,rows:&[crate::shared::speculation::dflash::ContextRow])->Result<()> {
        ensure!(!self.poisoned.get(),MiniMaxError::Invalid("previous native execution failure; restart required"));
        let result=self.native.device().run(|| {
            let mut guard=self.native.cancellation_guard("MiniMax draft update");
            let result=match self.draft()? {
                Drafter::Dspark(draft) => draft.update(&self.native,&self.scratch,rows),
                Drafter::Dflash(draft) => draft.update(rows),
            };
            let drained=self.native.drain();
            guard.complete=true;
            result.and(drained)
        });
        if result.is_err() {self.poisoned.set(true);}
        result
    }
    pub fn propose(&self,seqs:&[crate::shared::speculation::dflash::DraftSeq],embedding:&crate::shared::token_io::TokenEmbedding<'a>)
        ->Result<Vec<crate::shared::speculation::dflash::Draft>> {
        ensure!(!self.poisoned.get(),MiniMaxError::Invalid("previous native execution failure; restart required"));
        let block=self.draft()?.block();
        ensure!(seqs.iter().all(|s|s.position.checked_add(block).is_some_and(|end|end<=self.context)),
            MiniMaxError::Invalid("draft RoPE extent"));
        self.native.device().run(|| {
            let mut guard=self.native.cancellation_guard("MiniMax draft step");
            let result=match self.draft()? {
                Drafter::Dspark(draft) => draft.propose(&self.native,&self.scratch,embedding,&self.head,&self.weights,seqs),
                Drafter::Dflash(draft) => draft.propose(seqs,embedding,self.weights.get("language_model.lm_head.weight")?),
            };
            if result.is_err() {
                self.poisoned.set(true);
                let drained=self.native.drain();
                guard.complete=true;
                drained?;
            }
            // A proposal's readback waited for everything it queued.
            guard.complete=true;
            result
        })
    }
    pub fn stream(&self)->*mut std::ffi::c_void { self.native.raw() }
    pub fn drain(&self)->Result<()> { self.native.drain() }
    /// Runs every layer; the head projects only `logits` (batch rows), and an
    /// empty range skips it.
    pub fn execute(&self, embedded:CuteafdDeviceBuffer, batch:&BatchView, caches:&[LayerCache<'a>],
        experts:&mut dyn ExpertExecutor<'a>, kind:ExpertV2SourceKind, logits:std::ops::Range<usize>) -> Result<Option<DeviceLogits>> {
        self.execute_through(embedded,batch,caches,experts,kind,self.cfg.layers-1,logits)
    }
    pub fn hidden(&self,rows:usize)->Result<CuteafdDeviceBuffer> {
        ensure!(rows>0 && rows<=self.rows,MiniMaxError::Invalid("hidden row extent"));
        view(self.scratch.get("residual")?,0,rows*6144*4)
    }
    pub fn execute_through(&self, embedded:CuteafdDeviceBuffer, batch:&BatchView, caches:&[LayerCache<'a>],
        experts:&mut dyn ExpertExecutor<'a>, kind:ExpertV2SourceKind, last_layer:usize,
        logits:std::ops::Range<usize>) -> Result<Option<DeviceLogits>> {
        self.native.device().run(|| self.step(embedded,batch,caches,experts,kind,last_layer,logits))
    }
    #[allow(clippy::too_many_arguments)]
    fn step(&self, embedded:CuteafdDeviceBuffer, batch:&BatchView, caches:&[LayerCache<'a>],
        experts:&mut dyn ExpertExecutor<'a>, kind:ExpertV2SourceKind, last_layer:usize,
        logits:std::ops::Range<usize>) -> Result<Option<DeviceLogits>> {
        ensure!(!self.poisoned.get(),MiniMaxError::Invalid("previous native execution failure; restart required"));
        ensure!(last_layer<self.cfg.layers,MiniMaxError::Invalid("last execution layer"));
        ensure!(logits.end<=batch.positions.len() && logits.len()<=self.logit_rows,
            MiniMaxError::Invalid("head rows exceed the batch or the admitted logits"));
        let mut diagnostics=self.diagnostics.borrow_mut();
        ensure!(diagnostics.as_ref().is_none_or(|d|d.layer<=last_layer),MiniMaxError::Invalid("diagnostic layer not executed"));
        ensure!(matches!(kind,ExpertV2SourceKind::Decode|ExpertV2SourceKind::Prefill),
            MiniMaxError::Unsupported("expert source kind"));
        let rows=batch.validate(self.rows,self.context)?;
        ensure!(caches.len()==self.cfg.layers,MiniMaxError::Invalid("layer cache inventory"));
        for cache in caches { ensure!(cache.key.buffer.bytes>=batch.pool_pages*32768
            && cache.value.buffer.bytes>=batch.pool_pages*32768 && cache.index.buffer.bytes>=batch.pool_pages*8448,
            MiniMaxError::Invalid("physical cache extent")); }
        let routed=self.cfg.moe[..=last_layer].iter().filter(|&&moe|moe).count();
        // Every step runs eagerly: the Spark exchange's stream waits and writes are runtime
        // state, never captured, so no whole decode step is recorded.
        let mut guard=self.native.cancellation_guard("MiniMax step");
        let result=(||{
            let cuts=self.lane_cuts(batch,rows,experts.slots(),diagnostics.is_some())?;
            let widest=cuts.iter().chain([&rows]).scan(0,|at,&cut|{let piece=cut-*at; *at=cut; Some(piece)}).max().unwrap_or(rows);
            experts.begin(routed,cuts.len()+1,widest,kind)?;
            self.native.upload(self.scratch.get("positions")?,&batch.positions.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            self.native.upload(self.scratch.get("slots")?,&batch.slots.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            let plan=self.plan_attention(batch,rows,&cuts)?;
            let lanes=unit_tiles(&plan,rows,&cuts);
            self.layers(embedded,batch,batch.pool_pages,caches,experts,&plan,&lanes,rows,last_layer,logits.clone(),
                &mut diagnostics)
        })();
        // A failed step's stream may hold Spark exchange waits that only the link's worker
        // releases: stop the exchange (releasing them) before the drain.
        if result.is_err() { experts.abort(); }
        // All device/cache/expert borrows end only after queued work retires: the step's host
        // wait (`--host-wait`), or a plain drain after a failure.
        let drained=if result.is_ok() { self.native.wait() } else { self.native.drain() };
        guard.complete=true;
        let completed=drained.is_ok();
        let result=result.and_then(|logits|{drained?;experts.finish()?;Ok(logits)});
        if result.is_err() {
            self.poisoned.set(true);
            experts.abort();
            if completed { experts.release(); }
        }
        result
    }
    /// The step body over its `lanes`: residual widening, every layer through
    /// `last_layer` in [`visits`] order, then the head over `logits` (an empty
    /// range skips it). A lane collects a routed layer's experts only after
    /// the other lanes dispatched theirs, so one lane's exchange is in flight
    /// while another's GPU work runs. One lane (every decode step, and a
    /// diagnostic, which dumps whole layers) is the layer's own order.
    #[allow(clippy::too_many_arguments)]
    fn layers(&self, embedded:CuteafdDeviceBuffer, batch:&BatchView, pool_pages:usize, caches:&[LayerCache<'a>],
        experts:&mut dyn ExpertExecutor<'a>, plan:&AttentionPlan, lanes:&[UnitTiles], rows:usize, last_layer:usize,
        logits:std::ops::Range<usize>, diagnostics:&mut Option<Diagnostics>) -> Result<Option<DeviceLogits>> {
        ensure!(diagnostics.is_none() || lanes.len()==1,MiniMaxError::Invalid("a diagnostic dumps one lane"));
        let views=lanes.iter().map(|lane|self.scratch.rows(lane.first,lane.rows)).collect::<Result<Vec<_>>>()?;
        let begin=|layer:usize,diagnostics:&Option<Diagnostics>| diagnostics.as_ref().filter(|d|d.layer==layer)
            .map(|d|d.begin(batch)).transpose();
        self.native.widen(embedded,self.scratch.get("residual")?,rows*6144)?;
        // The dump of the routed layer whose waves are in flight (a diagnostic's one lane).
        let mut routed:Option<StepDump>=None;
        for visit in visits(&self.cfg.moe,last_layer,lanes.len()) {
            match visit {
                Visit::Dense(layer) => {
                    let cache=caches.get(layer).context("MiniMax layer cache")?;
                    let mut dump=begin(layer,diagnostics)?;
                    for (lane,s) in lanes.iter().zip(&views) {
                        if let Some(d)=&mut dump { d.tensor(&self.native,"input","f32",&[rows,6144],s.get("residual")?)?; }
                        self.attention(layer,lane,pool_pages,batch,cache,plan,s,dump.as_mut())?;
                        self.dense_ffn(layer,lane.first,lane.rows,rows,s,dump.as_mut())?;
                    }
                    if let Some(mut d)=dump {
                        d.tensor(&self.native,"residual","f32",&[rows,6144],self.scratch.get("residual")?)?;
                        diagnostics.as_mut().context("MiniMax diagnostic owner")?.finish(d)?;
                    }
                }
                Visit::Collect{layer,lane} => {
                    let (view,s)=(lanes.get(lane).context("MiniMax lane")?,&views[lane]);
                    self.ffn_back(layer,lane,view.first,view.rows,rows,s,experts)?;
                    // A routed layer's dump ends once its last lane collected.
                    if lane+1==lanes.len() { if let Some(mut d)=routed.take() {
                        d.tensor(&self.native,"residual","f32",&[rows,6144],self.scratch.get("residual")?)?;
                        diagnostics.as_mut().context("MiniMax diagnostic owner")?.finish(d)?;
                    } }
                }
                Visit::Dispatch{layer,lane} => {
                    let (view,s)=(lanes.get(lane).context("MiniMax lane")?,&views[lane]);
                    if lane==0 {
                        ensure!(routed.is_none(),MiniMaxError::Invalid("diagnostic dump of an uncollected layer"));
                        routed=begin(layer,diagnostics)?;
                        if let Some(d)=&mut routed { d.tensor(&self.native,"input","f32",&[rows,6144],s.get("residual")?)?; }
                    }
                    let cache=caches.get(layer).context("MiniMax layer cache")?;
                    self.attention(layer,view,pool_pages,batch,cache,plan,s,routed.as_mut())?;
                    self.ffn_front(layer,lane,view.rows,s,experts,routed.as_mut())?;
                }
            }
        }
        if last_layer+1==self.cfg.layers && !logits.is_empty() {
            self.head.logits(&self.native,&self.weights,&self.scratch,logits).map(Some)
        } else { Ok(None) }
    }
    fn residual(&self,s:&Rows<'_>,rows:usize)->Result<()> {
        self.native.launch("minimax_residual", &[s.get("residual")?,s.get("delta")?,s.get("residual")?], &[Scalar::I32(rows as i32)])
    }
    /// The drafter's tap of batch rows `first..first + rows` (`s` views them) of a `span`-row step.
    fn tap(&self,layer:usize,s:&Rows<'_>,first:usize,rows:usize,span:usize)->Result<()> {
        if let Some(draft)=&self.draft { draft.tap(&self.native,layer,s.get("residual")?,first,rows,span)?; }
        Ok(())
    }
    /// A dense layer's FFN on batch rows `first..first + rows` of a `span`-row step (`s` views
    /// them): the post-attention norm, the dense MLP into the residual, and the drafter's tap.
    fn dense_ffn(&self,layer:usize,first:usize,rows:usize,span:usize,s:&Rows<'_>,mut dump:Option<&mut StepDump>)->Result<()> {
        ffn::norm(&self.native,s.get("residual")?,self.weights.layer(layer,"post_attention_layernorm.weight")?,s.get("norm")?,rows)?;
        if let Some(d)=dump.as_deref_mut() { d.tensor(&self.native,"ffn_norm","bf16",&[rows,6144],s.get("norm")?)?; }
        ffn::dense(&self.native,&self.weights,s,layer,rows,false)?;
        self.residual(s,rows)?;
        self.tap(layer,s,first,rows,span)
    }
    /// A routed layer's FFN front on `rows` rows (`s` views them), after their attention: the
    /// post-attention norm, the router and the expert dispatch into slot `slot`, then the shared
    /// expert while the experts run. The shared expert's output stays in the rows' `shared`.
    fn ffn_front(&self,layer:usize,slot:usize,rows:usize,s:&Rows<'_>,experts:&mut dyn ExpertExecutor<'a>,
        mut dump:Option<&mut StepDump>)->Result<()> {
        ffn::norm(&self.native,s.get("residual")?,self.weights.layer(layer,"post_attention_layernorm.weight")?,s.get("norm")?,rows)?;
        if let Some(d)=dump.as_deref_mut() { d.tensor(&self.native,"ffn_norm","bf16",&[rows,6144],s.get("norm")?)?; }
        self.native.launch("minimax_router_scores", &[s.get("norm")?,
            self.weights.layer(layer,"block_sparse_moe.gate.weight")?,s.get("router_logits")?], &[Scalar::I32(rows as i32)])?;
        self.native.launch("minimax_route", &[s.get("router_logits")?,
            self.weights.layer(layer,"block_sparse_moe.e_score_correction_bias")?,s.get("route_ids")?,s.get("route_weights")?], &[Scalar::I32(rows as i32)])?;
        if let Some(d)=dump.as_deref_mut() {
            d.tensor(&self.native,"router_ids","i32",&[rows,4],s.get("route_ids")?)?;
            d.tensor(&self.native,"router_weights","f32",&[rows,4],s.get("route_weights")?)?;
        }
        experts.dispatch(layer,slot,rows,s.get("route_ids")?,s.get("route_weights")?,s.get("norm")?,&self.native)?;
        ffn::dense(&self.native,&self.weights,s,layer,rows,true)?;
        if let Some(d)=dump { d.tensor(&self.native,"shared","bf16",&[rows,6144],s.get("shared")?)?; }
        Ok(())
    }
    /// Its back on batch rows `first..first + rows` of a `span`-row step: slot `slot`'s routed
    /// experts plus the shared expert into the residual, and the drafter's tap.
    #[allow(clippy::too_many_arguments)]
    fn ffn_back(&self,layer:usize,slot:usize,first:usize,rows:usize,span:usize,s:&Rows<'_>,
        experts:&mut dyn ExpertExecutor<'a>)->Result<()> {
        experts.collect(layer,slot,rows,s.get("routed")?,&self.native)?;
        self.native.add_bf16(s.get("shared")?,s.get("routed")?,s.get("delta")?,rows*6144)?;
        self.residual(s,rows)?;
        self.tap(layer,s,first,rows,span)
    }
    /// Where a step's lanes after the first start (batch rows): a one-request
    /// prefill chunk runs in up to `lanes` lanes of `lane_rows` rows (at most
    /// `links`, one Spark exchange each; `serving_capacity::prefill_lanes`,
    /// cut on KV pages); decode, verify and packed steps run one lane, as
    /// does a diagnostic, which dumps whole layers.
    fn lane_cuts(&self,batch:&BatchView,rows:usize,links:usize,diagnostic:bool)->Result<Vec<usize>> {
        let one=diagnostic || rows<=self.decode_rows || batch.requests.len()>1;
        step_cuts(rows,if one {1} else {self.lanes.min(links)},self.lane_rows,self.program_rows,diagnostic)
    }
    /// The page-table stride of every decode step: wide enough for any live
    /// request (none holds more than the pool's pages or the context's).
    fn decode_stride(&self,batch:&BatchView)->usize { self.table_stride.min(batch.pool_pages) }
    /// The attention route `plan_attention` takes for a `rows`-row step, as the verify trace
    /// names it: every row a causal decode over its paged prefix, or index-prefill and extend tiles.
    pub fn attention_route(&self,rows:usize)->&'static str {
        if rows<=self.decode_rows {"causal_paged_decode"} else {"causal_paged_extend"}
    }
    /// Uploads the step's attention metadata (see [`AttentionPlan`]); extend
    /// tiles end at the lane `cuts`.
    fn plan_attention(&self,batch:&BatchView,rows:usize,cuts:&[usize])->Result<AttentionPlan> {
        let s=&self.scratch;
        if rows<=self.decode_rows {
            let stride=self.decode_stride(batch);
            let (tables,lengths,offsets)=decode_tables(batch,stride)?;
            self.native.upload(s.get("page_table")?,&le_bytes(&tables))?;
            self.native.upload(s.get("cache_lengths")?,&le_bytes(&lengths))?;
            self.native.upload(s.get("cu_seqlens_q")?,&le_bytes(&offsets))?;
            return Ok(AttentionPlan::Decode{stride});
        }
        let (launches,tables,lengths,offsets,bounds)=extend_tables(batch,self.extend_rows,self.table_stride,cuts)?;
        self.native.upload(s.get("page_table")?,&le_bytes(&tables))?;
        self.native.upload(s.get("cache_lengths")?,&le_bytes(&lengths))?;
        self.native.upload(s.get("cu_seqlens_q")?,&le_bytes(&offsets))?;
        self.native.upload(extend_bounds(s.get("index_bounds")?,self.extend_rows)?,&le_bytes(&bounds))?;
        Ok(AttentionPlan::Extend{launches})
    }
    /// One lane's attention for a layer: its rows (`s` views them) and its
    /// extend launches.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, layer:usize,lane:&UnitTiles,pool_pages:usize,batch:&BatchView,cache:&LayerCache<'a>,
        plan:&AttentionPlan,s:&Rows<'_>,mut dump:Option<&mut StepDump>)->Result<()> {
        let rows=lane.rows;
        ffn::norm(&self.native,s.get("residual")?,self.weights.layer(layer,"input_layernorm.weight")?,s.get("norm")?,rows)?;
        if let Some(d)=dump.as_deref_mut() { d.tensor(&self.native,"norm","bf16",&[rows,6144],s.get("norm")?)?; }
        for (projection,stem,out) in [("q","q","q"),("kv","kv","kv"),("index_q","index_q","index_q"),("index_k","index_k","index_k")] {
            ffn::linear(&self.native,&self.weights,s.get("scratch")?,layer,projection,&format!("self_attn.{stem}_proj"),s.get("norm")?,s.get(out)?,rows)?;
        }
        self.native.launch("minimax_kv_split", &[s.get("kv")?,s.get("k")?,s.get("v")?], &[Scalar::I32(rows as i32)])?;
        for stem in ["q","k","index_q","index_k"] {
            let weight=self.weights.layer(layer,&format!("self_attn.{stem}_norm.weight"))?;
            self.native.launch(&format!("minimax_{stem}_norm_rope"), &[s.get(stem)?,weight,weight,s.get("cos")?,s.get("sin")?,s.get("positions")?,s.get(&format!("{stem}_norm"))?], &[Scalar::I32(rows as i32)])?;
        }
        if let Some(d)=dump.as_deref_mut() {
            for (name,source,heads) in [("q","q_norm",64),("k","k_norm",4),("v","v",4),("index_q","index_q_norm",4),("index_k","index_k_norm",1)] {
                d.tensor(&self.native,name,"bf16",&[rows,heads,128],s.get(source)?)?;
            }
        }
        self.native.launch("minimax_index_k_pack", &[s.get("index_k_norm")?,s.get("slots")?,cache.index.buffer,s.get("index_q_scales")?], &[Scalar::I32(rows as i32)])?;
        self.native.launch("minimax_kv_pack", &[s.get("k_norm")?,s.get("v")?,s.get("slots")?,s.get("batch_ids")?,s.get("k_descale")?,s.get("v_descale")?,cache.key.buffer,cache.value.buffer], &[Scalar::I32(rows as i32)])?;
        self.native.launch("minimax_index_q_quant", &[s.get("index_q_norm")?,s.get("slots")?,s.get("index_q_fp8")?,s.get("index_q_scales")?], &[Scalar::I32(rows as i32)])?;
        if let Some(d)=dump.as_deref_mut() {
            d.tensor(&self.native,"index_q_fp8","u8",&[rows,4,128],s.get("index_q_fp8")?)?;
            d.tensor(&self.native,"index_q_scales","f32",&[rows,4],s.get("index_q_scales")?)?;
            let slots=&batch.slots;
            d.cache_rows(&self.native,"index_k_fp8","u8",&[rows,1,128],cache.index.buffer,slots,8448,0,128)?;
            d.cache_rows(&self.native,"index_k_scales","f32",&[rows,1],cache.index.buffer,slots,8448,8192,4)?;
            for (name,source) in [("k_fp8",cache.key.buffer),("v_fp8",cache.value.buffer)] {
                d.cache_rows(&self.native,name,"u8",&[rows,4,128],source,slots,32768,0,512)?;
            }
        }
        match plan {
            // Every row is an independent causal decode request over the exact
            // paged prefix, including prefixes from prior chunks.
            AttentionPlan::Decode{stride} => {
                let stride=*stride;
                self.native.launch("minimax_index_decode", &[view(s.get("index_q_fp8")?,0,rows*512)?,
                    view(s.get("index_q_scales")?,0,rows*16)?,cache.index.buffer,s.get("page_table")?,s.get("cache_lengths")?,s.get("q2k")?,s.get("scratch")?], &[
                    Scalar::I32(rows as i32),Scalar::I32(stride as i32),Scalar::I32(stride as i32),Scalar::I64(pool_pages as i64)])?;
                if let Some(d)=dump.as_deref_mut() { d.selection(&self.native,s.get("q2k")?,rows,self.decode_rows)?; }
                self.native.launch("minimax_attention_decode", &[view(s.get("q_norm")?,0,rows*16384)?,cache.key.buffer,cache.value.buffer,
                    s.get("k_descale")?,s.get("v_descale")?,s.get("page_table")?,s.get("cache_lengths")?,s.get("cu_seqlens_q")?,s.get("q2k")?,
                    view(s.get("attention")?,0,rows*16384)?,s.get("lse")?,s.get("scratch")?], &[
                    Scalar::I32(rows as i32),Scalar::I32(rows as i32),Scalar::I32(stride as i32),Scalar::I64(pool_pages as i64)])?;
            }
            // Each request's index keys through this lane are staged as the
            // contiguous rows the prefill scorer reads; its tiles then select
            // blocks and attend over the paged cache like the decode kernels.
            // A later lane restages a request an earlier one began: no row
            // reads keys past its own causal end.
            AttentionPlan::Extend{launches} => {
                let (stride,tile)=(self.table_stride,self.extend_rows);
                let (tables,bounds)=(s.get("page_table")?,s.get("index_bounds")?);
                for (index,launch) in launches[lane.launches.clone()].iter().enumerate() {
                    let request=&batch.requests[launch.request];
                    let table=view(tables,launch.slot*stride*4,stride*4)?;
                    if launch.first==0 || index==0 {
                        let end=(request.row_start+request.rows).min(lane.first+lane.rows)-request.row_start;
                        let pages=(request.start_position+end).div_ceil(64);
                        self.native.index_stage(cache.index.buffer,view(table,0,pages*4)?,pages,s.get("index_keys")?,s.get("index_scales")?)?;
                    }
                    let (row,n)=(launch.row-lane.first,launch.rows);
                    self.native.launch("minimax_index_prefill", &[view(s.get("index_q_fp8")?,row*512,n*512)?,
                        view(s.get("index_q_scales")?,row*16,n*16)?,s.get("index_keys")?,s.get("index_scales")?,
                        view(bounds,0,n*4)?,view(bounds,launch.bounds*4,n*4)?,s.get("q2k_extend")?,s.get("scratch")?], &[
                        Scalar::I32(n as i32),Scalar::I32((request.start_position+launch.first+n) as i32)])?;
                    if let Some(d)=dump.as_deref_mut() { d.selection(&self.native,s.get("q2k_extend")?,n,tile)?; }
                    self.native.launch("minimax_attention_extend", &[view(s.get("q_norm")?,row*16384,n*16384)?,cache.key.buffer,cache.value.buffer,
                        s.get("k_descale")?,s.get("v_descale")?,table,view(s.get("cache_lengths")?,launch.slot*16,16)?,
                        view(s.get("cu_seqlens_q")?,launch.slot*16,16)?,s.get("q2k_extend")?,
                        view(s.get("attention")?,row*16384,n*16384)?,s.get("lse")?,s.get("scratch")?], &[
                        Scalar::I32(n as i32),Scalar::I32(1),Scalar::I32(stride as i32),Scalar::I64(pool_pages as i64)])?;
                }
            }
        }
        if let Some(d)=dump.as_deref_mut() {
            d.finish_selection()?;
            d.tensor(&self.native,"attention","bf16",&[rows,64,128],s.get("attention")?)?;
        }
        ffn::linear(&self.native,&self.weights,s.get("scratch")?,layer,"o","self_attn.o_proj",s.get("attention")?,s.get("delta")?,rows)?;
        if let Some(d)=dump.as_deref_mut() { d.tensor(&self.native,"attention_projected","bf16",&[rows,6144],s.get("delta")?)?; }
        self.residual(s,rows)?;
        if let Some(d)=dump { d.tensor(&self.native,"post_attention","f32",&[rows,6144],s.get("residual")?)?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn minimax_attention_metadata_is_built_once_per_step() -> Result<()> {
        // Two requests: 3 rows from 62 (pages 0, 2) and 1 row at 4 (page 4).
        let batch=BatchView {positions:vec![62,63,64,4],slots:vec![62,63,128,260],pool_pages:5,
            requests:vec![RequestView{row_start:0,rows:3,start_position:62,pages:vec![0,2]},
                RequestView{row_start:3,rows:1,start_position:4,pages:vec![4]}]};
        batch.validate(64,128)?;
        let (tables,lengths,offsets)=decode_tables(&batch,2)?;
        assert_eq!(tables,[0,-1, 0,-1, 0,2, 4,-1]);
        assert_eq!(lengths,[63,64,65,5]);
        assert_eq!(offsets,[0,1,2,3,4]);
        // Every decode step uses one fixed stride; live entries keep their place.
        let (wide,_,_)=decode_tables(&batch,4)?;
        assert_eq!(wide,[0,-1,-1,-1, 0,-1,-1,-1, 0,2,-1,-1, 4,-1,-1,-1]);
        assert!(decode_tables(&batch,1).is_err(),"a row wider than the stride is refused");
        // A two-row tile splits the first request; launch metadata is 16-byte aligned.
        let (launches,tables,lengths,offsets,bounds)=extend_tables(&batch,2,4,&[])?;
        assert_eq!(launches,[Launch{request:0,row:0,first:0,rows:2,slot:0,bounds:2},
            Launch{request:0,row:2,first:2,rows:1,slot:1,bounds:6},Launch{request:1,row:3,first:0,rows:1,slot:2,bounds:10}]);
        assert_eq!(tables,[0,2,-1,-1, 0,2,-1,-1, 4,-1,-1,-1]);
        assert_eq!(lengths,[64,0,0,0, 65,0,0,0, 5,0,0,0]);
        assert_eq!(offsets,[0,2,0,0, 0,1,0,0, 0,1,0,0]);
        assert_eq!(bounds,[63,64,0,0, 65,0,0,0, 5,0,0,0]);
        assert!(extend_tables(&batch,2,1,&[]).is_err(),"a request wider than the admitted table is refused");
        Ok(())
    }

    fn chunk(start:usize,sizes:&[usize])->BatchView {
        let rows:usize=sizes.iter().sum();
        let mut requests=Vec::new();
        let mut row=0;
        for (i,&n) in sizes.iter().enumerate() {
            let base=(i*100) as i32;
            requests.push(RequestView{row_start:row,rows:n,start_position:start,pages:(0..(start+n).div_ceil(64) as i32).map(|p|base+p).collect()});
            row+=n;
        }
        let positions=sizes.iter().flat_map(|&n|(start..start+n).map(|p|p as i64)).collect();
        BatchView{positions,slots:vec![0;rows],requests,pool_pages:1<<20}
    }

    #[test] fn minimax_prefill_lanes_end_extend_tiles_at_their_cuts() -> Result<()> {
        let lanes=|batch:&BatchView,want:usize,lane_rows:usize|->Result<Vec<(usize,usize,Vec<(usize,usize)>)>> {
            let rows=batch.positions.len();
            let cuts=step_cuts(rows,want,lane_rows,4096,false)?;
            let (launches,..)=extend_tables(batch,1024,1<<12,&cuts)?;
            let plan=AttentionPlan::Extend{launches:launches.clone()};
            Ok(unit_tiles(&plan,rows,&cuts).into_iter().map(|l|(l.first,l.rows,
                launches[l.launches].iter().map(|t|(t.row,t.rows)).collect())).collect())
        };
        // Two lanes of 4096 rows: a full chunk keeps the serial tiles.
        assert_eq!(lanes(&chunk(8192,&[8192]),2,4096)?,[(0,4096,vec![(0,1024),(1024,1024),(2048,1024),(3072,1024)]),
            (4096,4096,vec![(4096,1024),(5120,1024),(6144,1024),(7168,1024)])]);
        // A page-aligned cut ends a tile early and the next lane's first tile starts there.
        assert_eq!(lanes(&chunk(0,&[4412]),2,4096)?,[(0,2240,vec![(0,1024),(1024,1024),(2048,192)]),
            (2240,2172,vec![(2240,1024),(3264,1024),(4288,124)])]);
        // Short chunks and one lane keep one lane and the serial tiles.
        assert_eq!(lanes(&chunk(64,&[300]),2,4096)?,[(0,300,vec![(0,300)])]);
        assert_eq!(lanes(&chunk(0,&[4096]),1,4096)?,[(0,4096,vec![(0,1024),(1024,1024),(2048,1024),(3072,1024)])]);
        assert!(step_cuts(8192,1,4096,4096,false).is_err(),"one lane holds at most a program launch");
        assert!(step_cuts(4097,2,2048,4096,false).is_err(),"two lanes of 2048 rows hold 4096");
        assert!(step_cuts(8192,1,4096,4096,true).unwrap_err().to_string().contains("--prefill-lanes 1"));
        // Decode plans keep one lane.
        assert_eq!(unit_tiles(&AttentionPlan::Decode{stride:4},64,&[]),[UnitTiles{first:0,rows:64,launches:0..0}]);
        Ok(())
    }

    #[test] fn minimax_step_visits_collect_each_routed_layer_before_the_next_dispatch() {
        use Visit::*;
        // Dense 0, routed 1-2, dense 3, routed 4; two lanes.
        let moe=[false,true,true,false,true];
        assert_eq!(visits(&moe,4,2),[Dense(0),Dispatch{layer:1,lane:0},Dispatch{layer:1,lane:1},
            Collect{layer:1,lane:0},Dispatch{layer:2,lane:0},Collect{layer:1,lane:1},Dispatch{layer:2,lane:1},
            Collect{layer:2,lane:0},Collect{layer:2,lane:1},Dense(3),
            Dispatch{layer:4,lane:0},Dispatch{layer:4,lane:1},Collect{layer:4,lane:0},Collect{layer:4,lane:1}]);
        // One lane is the layer's own order; a partial step ends after its last layer.
        assert_eq!(visits(&moe,2,1),[Dense(0),Dispatch{layer:1,lane:0},Collect{layer:1,lane:0},
            Dispatch{layer:2,lane:0},Collect{layer:2,lane:0}]);
        assert_eq!(visits(&moe,0,3),[Dense(0)]);
    }

    /// The one-link mapping: driven through the step's visit order (dense, RTX-local and Spark
    /// layers in any pattern, 1-4 lanes, any last layer), the routed owner's bookkeeping sends
    /// at most `lanes` Spark waves and collects each one as the oldest outstanding, so a link
    /// with one return slot per lane carries every step. Wave k takes slot k % lanes, and the
    /// slot's previous wave was collected (landed and reduced) before it.
    #[test] fn minimax_lane_waves_land_in_dispatch_order_on_one_link() -> Result<()> {
        use super::super::routed::{Wave, WaveOrder};
        let mut seed=0x2545_f491_4f6c_dd1du64;
        let mut next=|bound:u64| { seed^=seed<<13; seed^=seed>>7; seed^=seed<<17; seed%bound };
        for _ in 0..400 {
            let layers=1+next(60) as usize;
            let moe=(0..layers).map(|_|next(4)!=0).collect::<Vec<_>>();
            let local=(0..layers).map(|_|next(5)==0).collect::<Vec<_>>();
            let lanes=1+next(4) as usize;
            let last=next(layers as u64) as usize;
            let mut order=WaveOrder::new(lanes);
            let mut pending:Vec<Option<(usize,Option<u64>)>>=vec![None;lanes];
            let (mut sent,mut slots)=(0u64,vec![true;lanes]);
            for visit in visits(&moe,last,lanes) {
                match visit {
                    Visit::Dense(layer) => {
                        assert!(!moe[layer] && pending.iter().all(Option::is_none) && order.is_empty(),
                            "a dense layer runs with no wave in flight");
                    }
                    Visit::Dispatch{layer,lane} => {
                        assert!(moe[layer] && pending[lane].is_none(),"one wave per lane");
                        let spark=(!local[layer]).then(|| {
                            let slot=(sent%lanes as u64) as usize;
                            assert!(slots[slot],"wave {sent} reuses slot {slot} before its previous wave landed");
                            slots[slot]=false;
                            order.push(Wave{layer,slot:lane,rows:64},sent).unwrap();
                            sent+=1;
                            sent-1
                        });
                        pending[lane]=Some((layer,spark));
                    }
                    Visit::Collect{layer,lane} => {
                        let (dispatched,spark)=pending[lane].take().expect("collect after dispatch");
                        assert_eq!(dispatched,layer);
                        if let Some(wave)=spark {
                            assert_eq!(order.pop(Wave{layer,slot:lane,rows:64})?,wave,"the oldest wave lands first");
                            slots[(wave%lanes as u64) as usize]=true;
                        }
                    }
                }
            }
            assert!(pending.iter().all(Option::is_none) && order.is_empty(),"the step ends with every wave collected");
        }
        Ok(())
    }

    /// Decode runs eagerly (no whole-step recording can hold the exchange's stream memory
    /// operations), and a failed step quiesces the Spark link before its drain, then releases
    /// the link's owners only after a completed drain.
    /// Every ordered upload of a step goes to a whole admitted destination that load reserved
    /// (or the extend bounds' tail, reserved too), never to a view narrowed to its bytes: the
    /// first upload sizes a destination's pinned staging, and no step allocates any.
    #[test] fn minimax_step_uploads_go_to_reserved_whole_destinations() {
        let source=include_str!("execution.rs").split("#[cfg(test)]").next().unwrap();
        let (head,rest)=source.split_at(source.find("    pub fn load(").unwrap());
        let (load,steps)=rest.split_at(rest.find("    pub fn load_draft(").unwrap());
        assert!(!head.contains(".upload(") && load.contains("for name in STEP_UPLOADS { native.reserve(scratch.get(name)?)?; }")
            && load.contains("native.reserve(extend_bounds(scratch.get(\"index_bounds\")?,extend_rows)?)?;"));
        let mut uploads=0;
        for (at,_) in steps.match_indices(".upload(") {
            let destination=&steps[at+".upload(".len()..];
            let destination=&destination[..destination.find(")?,").unwrap()+2];
            let name=destination.split("get(\"").nth(1).and_then(|n|n.split('"').next()).unwrap_or_default();
            assert!(destination=="extend_bounds(s.get(\"index_bounds\")?"
                || (STEP_UPLOADS.contains(&name) && destination.ends_with(&format!("get(\"{name}\")?"))),
                "unreserved or narrowed upload destination {destination}");
            uploads+=1;
        }
        assert_eq!(uploads,9,"every step upload is checked");
        let native=include_str!("native.rs");
        assert!(native.contains("copy_h2d_ordered(buffer, bytes, self.stream.raw)") && !native.contains("bytes: bytes.len()"));
    }

    #[test] fn minimax_extend_bounds_are_the_index_bounds_tail() -> Result<()> {
        let bounds=CuteafdDeviceBuffer{ptr:0x1000usize as *mut std::ffi::c_void,bytes:4096,device_id:0,flags:0};
        let tail=extend_bounds(bounds,256)?;
        assert_eq!((tail.ptr as usize,tail.bytes),(0x1000+1024,4096-1024));
        assert!(extend_bounds(bounds,1025).is_err(),"key starts past the admitted bounds");
        Ok(())
    }

    #[test] fn minimax_step_is_eager_and_quiesces_the_exchange_before_draining() {
        let source=include_str!("execution.rs").split("#[cfg(test)]").next().unwrap();
        for banned in [["cuda_graph_", "begin_capture"].concat(), ["Graph", "Bank"].concat(),
            ["census::", "dispatch"].concat(), ["warm_", "graphs"].concat()] {
            assert!(!source.contains(&banned),"{banned}");
        }
        let step=source.split("fn step(").nth(1).unwrap().split("fn layers(").next().unwrap();
        let order=["if result.is_err() { experts.abort(); }","let drained=if result.is_ok() { self.native.wait() } else { self.native.drain() };",
            "experts.finish()?","if completed { experts.release(); }"]
            .map(|needle|step.find(needle).unwrap_or_else(||panic!("{needle}")));
        assert!(order.windows(2).all(|w|w[0]<w[1]),"{order:?}");
    }
}

