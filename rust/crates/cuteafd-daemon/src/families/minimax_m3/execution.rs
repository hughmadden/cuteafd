//! One device, one append-only paged MSA pipeline, native checkpoint arithmetic.
use super::{MiniMaxError, diagnostics::{Diagnostics, StepDump}, ffn, head::TargetHeadOwner, scratch::Scratch, weights::Weights};
use crate::shared::{memory::device::Device, token_io::DeviceLogits};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{minimax::{Buffer, Commands}, programs::{Programs,Scalar}};
use cuteafd_transport::ExpertV2SourceKind;
use cuteafd_loader::{families::minimax_m3::{cache::PAGE_ROWS,config::MiniMaxM3Config},
    plan::{checkpoint::Checkpoint,families::minimax::admission::{RuntimeLedger,RuntimeSizing}},
    serving_capacity::prefill_lanes as lane_plans};

#[derive(Clone)]
pub(super) struct LayerCache<'a> { pub key: Buffer<'a>, pub value: Buffer<'a>, pub index: Buffer<'a> }
#[derive(Debug,Clone)]
pub(super) struct RequestView { pub row_start:usize, pub rows:usize, pub start_position:usize, pub pages:Vec<i32> }
#[derive(Debug,Clone)]
pub(super) struct BatchView { pub positions:Vec<i64>, pub slots:Vec<i64>, pub requests:Vec<RequestView>, pub pool_pages:usize }

impl BatchView {
    pub fn validate(&self, capacity:usize, context:usize) -> Result<usize> {
        let rows = self.positions.len();
        ensure!(rows>0 && rows<=capacity && self.slots.len()==rows && !self.requests.is_empty()
            && self.requests.len()<=64 && self.pool_pages>0 && self.pool_pages<=i32::MAX as usize,
            MiniMaxError::Invalid("batch extent"));
        let mut cursor = 0;
        for request in &self.requests {
            ensure!(request.rows>0 && request.row_start==cursor && request.rows<=rows-cursor,
                MiniMaxError::Invalid("packed request rows"));
            let end = request.start_position.checked_add(request.rows).context("MiniMax request extent overflow")?;
            ensure!(end<=context && request.pages.len()>=end.div_ceil(64)
                && request.pages.iter().all(|&p|p>=0 && (p as usize)<self.pool_pages),
                MiniMaxError::Invalid("request pages or context"));
            for local in 0..request.rows {
                let position=request.start_position+local;
                let slot=request.pages[position/64] as i64*64+(position%64) as i64;
                ensure!(self.positions[cursor+local]==position as i64 && self.slots[cursor+local]==slot,
                    MiniMaxError::Invalid("append position or slot"));
            }
            cursor+=request.rows;
        }
        ensure!(cursor==rows,MiniMaxError::Invalid("unowned batch rows"));
        ensure!(self.slots.iter().collect::<std::collections::BTreeSet<_>>().len()==rows,
            MiniMaxError::Invalid("multiple append rows share a physical slot"));
        let written:std::collections::BTreeSet<_>=self.slots.iter().map(|slot|(*slot/64) as i32).collect();
        let mut owners=std::collections::BTreeSet::new();
        for request in &self.requests {
            for &page in &request.pages {
                ensure!(owners.insert(page) || !written.contains(&page),
                    MiniMaxError::Invalid("append page is shared; copy-on-write required"));
            }
        }
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

/// Decode metadata: per-row tables (-1 padded), lengths and packed offsets.
fn decode_tables(batch:&BatchView)->Result<(usize,Vec<i32>,Vec<i32>,Vec<i32>)> {
    let rows=batch.positions.len();
    let stride=batch.positions.iter().copied().max().context("MiniMax empty batch")? as usize/64+1;
    let mut tables=vec![-1i32;rows*stride];
    let mut lengths=Vec::with_capacity(rows);
    for row in 0..rows {
        let request=batch.request(row)?;
        let length=batch.positions[row] as usize+1;
        let width=length.div_ceil(64);
        tables[row*stride..row*stride+width].copy_from_slice(&request.pages[..width]);
        lengths.push(length as i32);
    }
    Ok((stride,tables,lengths,(0..=rows as i32).collect()))
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
/// range of the plan's; empty for a decode plan): one prefill lane
/// (`serving_capacity::prefill_lanes`). Each lane has its own expert exchange
/// in flight while another lane's layer runs on the stream.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Lane { first:usize, rows:usize, launches:std::ops::Range<usize> }

/// The lanes of a step cut at `cuts` (each lane's first row after the first),
/// with the launches of `plan` each holds (extend tiles never cross a cut).
fn lanes_at(plan:&AttentionPlan,rows:usize,cuts:&[usize])->Vec<Lane> {
    let launches=match plan { AttentionPlan::Extend{launches}=>launches.as_slice(), AttentionPlan::Decode{..}=>&[] };
    let mut bounds=vec![0]; bounds.extend_from_slice(cuts); bounds.push(rows);
    let mut at=0;
    bounds.windows(2).map(|w| {
        let start=at;
        while at<launches.len() && launches[at].row<w[1] { at+=1; }
        Lane{first:w[0],rows:w[1]-w[0],launches:start..at}
    }).collect()
}

pub(super) trait ExpertExecutor<'a> {
    /// Before a step's layers: `waves` routed layers on each of `lanes` lanes.
    fn begin(&mut self, _waves:usize, _lanes:usize, _kind:ExpertV2SourceKind) -> Result<()> { Ok(()) }
    /// Lanes whose waves can be in flight at once (one exchange each).
    fn lanes(&self) -> usize { 1 }
    fn dispatch(&mut self, layer:usize, lane:usize, rows:usize, ids:&Buffer<'a>, weights:&Buffer<'a>,
        input:&Buffer<'a>, commands:&Commands<'a>) -> Result<()>;
    fn collect(&mut self, layer:usize, lane:usize, rows:usize, output:&Buffer<'a>, commands:&Commands<'a>) -> Result<()>;
    fn finish(&mut self) -> Result<()> { Ok(()) }
    fn abort(&mut self) {}
}

pub(super) struct Execution<'a> {
    // Drain queued consumers before their programs and allocations are dropped.
    commands: Commands<'a>,
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
    draft:Option<super::dspark::Dspark<'a>>,
    draft_causal:bool,
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
        let commands=Commands::new(device.library,device.id,programs)?;
        let decode_rows=commands.capacity("minimax_index_decode")?;
        ensure!(decode_rows>0 && decode_rows<=64 && commands.capacity("minimax_attention_decode")?==decode_rows,
            MiniMaxError::Invalid("index/attention capacity stride"));
        let extend_rows=commands.capacity("minimax_index_prefill")?;
        ensure!(extend_rows>0 && extend_rows%4==0 && commands.capacity("minimax_attention_extend")?==extend_rows,
            MiniMaxError::Invalid("prefill index/attention capacity stride"));
        let program_rows=commands.capacity("minimax_norm")?;
        ensure!(rows==sizing.chunk_rows() && sizing.rows<=program_rows,MiniMaxError::Invalid("prefill lane rows"));
        let scratch=Scratch::new(device,ledger)?;
        let table_stride=usize::try_from(cuteafd_loader::plan::families::minimax::admission::extend_table_stride(context as u64))?;
        ensure!(scratch.get("page_table")?.raw().bytes>=decode_rows*table_stride*4,MiniMaxError::Invalid("page-table admission"));
        let weights=Weights::load(device,checkpoint,&commands)?;
        commands.launch("minimax_rope", &[scratch.get("cos")?,scratch.get("sin")?], &[
            Scalar::I32(context as i32),Scalar::F32(cfg.rope_theta as f32),Scalar::F32(1.0),
            Scalar::F32(0.0),Scalar::F32(1.0),Scalar::F32(1.0)])?;
        let ones=vec![1.0f32.to_le_bytes();decode_rows*4].concat();
        commands.upload(scratch.get("k_descale")?,&ones)?;
        commands.upload(scratch.get("v_descale")?,&ones)?;
        commands.upload(scratch.get("batch_ids")?,&vec![0;rows*4])?;
        // Extend launches read their key starts (all zero) from here.
        commands.upload(scratch.get("index_bounds")?,&vec![0;extend_rows*4])?;
        commands.drain()?;
        let head=TargetHeadOwner::new(&commands,&scratch,logit_rows,cfg.vocab_size)?;
        Ok(Self{commands,head,weights,scratch,cfg:cfg.clone(),rows,lane_rows:sizing.rows,lanes:sizing.lanes,program_rows,
            logit_rows,context,decode_rows,extend_rows,table_stride,
            poisoned:std::cell::Cell::new(false),diagnostics,draft:None,draft_causal:false})
    }
    pub fn load_draft(&mut self, contract:&cuteafd_loader::families::minimax_m3::dspark::DsparkContract,
        layout:super::dspark::Layout)->Result<()> {
        ensure!(contract.sample_from_anchor && !contract.causal,MiniMaxError::Unsupported("draft mask or anchor mode"));
        self.draft=Some(super::dspark::Dspark::load(Device {library:self.commands.library(),id:self.commands.device()},
            &self.commands,contract,layout)?);
        self.draft_causal=contract.causal;
        Ok(())
    }
    pub fn draft(&self)->Result<&super::dspark::Dspark<'a>> { self.draft.as_ref().context("MiniMax drafter not admitted") }
    pub fn update_draft(&self,rows:&[crate::shared::speculation::dflash::ContextRow])->Result<()> {
        ensure!(!self.poisoned.get(),MiniMaxError::Invalid("previous native execution failure; restart required"));
        let result=self.draft()?.update(&self.commands,&self.scratch,rows);
        let drained=self.commands.drain();
        let result=result.and(drained);
        if result.is_err() {self.poisoned.set(true);}
        result
    }
    pub fn propose(&self,seqs:&[crate::shared::speculation::dflash::DraftSeq],embedding:&crate::shared::token_io::TokenEmbedding<'a>)
        ->Result<Vec<crate::shared::speculation::dflash::Draft>> {
        ensure!(!self.poisoned.get(),MiniMaxError::Invalid("previous native execution failure; restart required"));
        ensure!(seqs.iter().all(|s|s.position.checked_add(7).is_some_and(|end|end<=self.context)),
            MiniMaxError::Invalid("draft RoPE extent"));
        let result=self.draft()?.propose(&self.commands,&self.scratch,embedding,&self.head,&self.weights,seqs,self.draft_causal);
        if result.is_err() {
            self.poisoned.set(true);
            self.commands.drain()?;
        }
        result
    }
    pub fn stream(&self)->*mut std::ffi::c_void { self.commands.raw() }
    pub fn drain(&self)->Result<()> { self.commands.drain() }
    /// Runs every layer; the head projects only `logits` (batch rows), and an
    /// empty range skips it.
    pub fn execute(&self, embedded:&Buffer<'a>, batch:&BatchView, caches:&[LayerCache<'a>],
        experts:&mut dyn ExpertExecutor<'a>, kind:ExpertV2SourceKind, logits:std::ops::Range<usize>) -> Result<Option<DeviceLogits>> {
        self.execute_through(embedded,batch,caches,experts,kind,self.cfg.layers-1,logits)
    }
    pub fn hidden(&self,rows:usize)->Result<Buffer<'a>> {
        ensure!(rows>0 && rows<=self.rows,MiniMaxError::Invalid("hidden row extent"));
        self.scratch.get("residual")?.view(0,rows*6144*4)
    }
    pub fn execute_through(&self, embedded:&Buffer<'a>, batch:&BatchView, caches:&[LayerCache<'a>],
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
        for cache in caches { ensure!(cache.key.raw().bytes>=batch.pool_pages*32768
            && cache.value.raw().bytes>=batch.pool_pages*32768 && cache.index.raw().bytes>=batch.pool_pages*8448,
            MiniMaxError::Invalid("physical cache extent")); }
        let mut lane_count=1;
        let result=(||{
            self.commands.upload(self.scratch.get("positions")?,&batch.positions.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            self.commands.upload(self.scratch.get("slots")?,&batch.slots.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            let cuts=self.lane_cuts(batch,rows,experts.lanes(),diagnostics.is_some())?;
            let plan=self.plan_attention(batch,rows,&cuts)?;
            let lanes=lanes_at(&plan,rows,&cuts);
            let views=lanes.iter().map(|lane|self.scratch.rows(lane.first,lane.rows)).collect::<Result<Vec<_>>>()?;
            lane_count=lanes.len();
            experts.begin(self.cfg.moe[..=last_layer].iter().filter(|&&moe|moe).count(),lanes.len(),kind)?;
            self.commands.widen(embedded,self.scratch.get("residual")?,rows*6144)?;
            // Each lane's routed layer runs its attention, router, dispatch and
            // shared expert, then the next lane's does; a lane collects its
            // experts only after the other lanes' halves of the layer, so one
            // lane's exchange is in flight while another's GPU work runs.
            // With one lane this is the layer's own order.
            let mut pending:Option<(usize,Option<StepDump>)>=None;
            for (layer,cache) in caches.iter().enumerate().take(last_layer+1) {
                if !self.cfg.moe[layer] {
                    if let Some((previous,dump))=pending.take() {
                        self.finish_routed(previous,rows,&lanes,&views,experts,dump,diagnostics.as_mut())?;
                    }
                    // A dense layer runs lane after lane (a diagnostic's one lane).
                    let mut dump=diagnostics.as_ref().filter(|d|d.layer==layer).map(|d|d.begin(batch)).transpose()?;
                    for (lane,s) in lanes.iter().zip(&views) {
                        if let Some(d)=&mut dump { d.tensor(&self.commands,"input","f32",&[rows,6144],s.get("residual")?)?; }
                        self.attention(layer,lane,batch,cache,&plan,s,dump.as_mut())?;
                        ffn::norm(&self.commands,s.get("residual")?,self.weights.layer(layer,"post_attention_layernorm.weight")?,s.get("norm")?,lane.rows)?;
                        if let Some(d)=&mut dump { d.tensor(&self.commands,"ffn_norm","bf16",&[rows,6144],s.get("norm")?)?; }
                        ffn::dense(&self.commands,&self.weights,s,layer,lane.rows,false)?;
                        self.residual(s,lane.rows)?;
                        if let Some(draft)=&self.draft { draft.tap(&self.commands,layer,s.get("residual")?,lane.first,lane.rows)?; }
                    }
                    if let Some(mut d)=dump {
                        d.tensor(&self.commands,"residual","f32",&[rows,6144],self.scratch.get("residual")?)?;
                        diagnostics.as_mut().context("MiniMax diagnostic owner")?.finish(d)?;
                    }
                    continue;
                }
                let mut dump=None;
                for (index,(lane,s)) in lanes.iter().zip(&views).enumerate() {
                    if let Some((previous,previous_dump))=&mut pending {
                        self.routed_back(*previous,index,lane,s,experts)?;
                        if let Some(mut d)=previous_dump.take() {
                            d.tensor(&self.commands,"residual","f32",&[rows,6144],s.get("residual")?)?;
                            diagnostics.as_mut().context("MiniMax diagnostic owner")?.finish(d)?;
                        }
                    }
                    if index==0 {
                        dump=diagnostics.as_ref().filter(|d|d.layer==layer).map(|d|d.begin(batch)).transpose()?;
                        if let Some(d)=&mut dump { d.tensor(&self.commands,"input","f32",&[rows,6144],s.get("residual")?)?; }
                    }
                    self.routed_front(layer,index,lane,s,batch,cache,&plan,experts,dump.as_mut())?;
                }
                pending=Some((layer,dump));
            }
            if let Some((previous,dump))=pending.take() {
                self.finish_routed(previous,rows,&lanes,&views,experts,dump,diagnostics.as_mut())?;
            }
            if last_layer+1==self.cfg.layers && !logits.is_empty() {
                self.head.logits(&self.commands,&self.weights,&self.scratch,logits).map(Some)
            } else { Ok(None) }
        })();
        // All device/cache/expert borrows end only after queued work retires.
        let drained=self.commands.drain();
        let finished=experts.finish();
        let result=result.and_then(|value|{drained?;finished?;Ok(value)});
        if result.is_err() { self.poisoned.set(true); experts.abort(); }
        else if let Err(error)=super::diagnostics::step_digest(&self.commands,self.scratch.get("residual")?,rows,lane_count) {
            tracing::warn!(%error,"MiniMax step digest");
        }
        result
    }
    fn residual(&self,s:&Scratch<'a>,rows:usize)->Result<()> {
        self.commands.launch("minimax_residual", &[s.get("residual")?,s.get("delta")?,s.get("residual")?], &[Scalar::I32(rows as i32)])
    }
    /// A routed layer's first half on one lane: attention, the router and the
    /// expert dispatch, then the shared expert while the experts run.
    #[allow(clippy::too_many_arguments)]
    fn routed_front(&self,layer:usize,index:usize,lane:&Lane,s:&Scratch<'a>,batch:&BatchView,cache:&LayerCache<'a>,
        plan:&AttentionPlan,experts:&mut dyn ExpertExecutor<'a>,mut dump:Option<&mut StepDump>)->Result<()> {
        let rows=lane.rows;
        self.attention(layer,lane,batch,cache,plan,s,dump.as_deref_mut())?;
        ffn::norm(&self.commands,s.get("residual")?,self.weights.layer(layer,"post_attention_layernorm.weight")?,s.get("norm")?,rows)?;
        if let Some(d)=dump.as_deref_mut() { d.tensor(&self.commands,"ffn_norm","bf16",&[rows,6144],s.get("norm")?)?; }
        self.commands.launch("minimax_router_scores", &[s.get("norm")?,
            self.weights.layer(layer,"block_sparse_moe.gate.weight")?,s.get("router_logits")?], &[Scalar::I32(rows as i32)])?;
        self.commands.launch("minimax_route", &[s.get("router_logits")?,
            self.weights.layer(layer,"block_sparse_moe.e_score_correction_bias")?,s.get("route_ids")?,s.get("route_weights")?], &[Scalar::I32(rows as i32)])?;
        if let Some(d)=dump.as_deref_mut() {
            d.tensor(&self.commands,"router_ids","i32",&[rows,4],s.get("route_ids")?)?;
            d.tensor(&self.commands,"router_weights","f32",&[rows,4],s.get("route_weights")?)?;
        }
        experts.dispatch(layer,index,rows,s.get("route_ids")?,s.get("route_weights")?,s.get("norm")?,&self.commands)?;
        ffn::dense(&self.commands,&self.weights,s,layer,rows,true)?;
        if let Some(d)=dump { d.tensor(&self.commands,"shared","bf16",&[rows,6144],s.get("shared")?)?; }
        Ok(())
    }
    /// Its second half: the lane's routed experts plus the shared expert into
    /// the residual, and the drafter's tap.
    fn routed_back(&self,layer:usize,index:usize,lane:&Lane,s:&Scratch<'a>,experts:&mut dyn ExpertExecutor<'a>)->Result<()> {
        experts.collect(layer,index,lane.rows,s.get("routed")?,&self.commands)?;
        self.commands.add_bf16(s.get("shared")?,s.get("routed")?,s.get("delta")?,lane.rows*6144)?;
        self.residual(s,lane.rows)?;
        if let Some(draft)=&self.draft { draft.tap(&self.commands,layer,s.get("residual")?,lane.first,lane.rows)?; }
        Ok(())
    }
    /// Every lane's second half of routed layer `layer`, then the end of its
    /// diagnostic dump (`rows`, one lane).
    #[allow(clippy::too_many_arguments)]
    fn finish_routed(&self,layer:usize,rows:usize,lanes:&[Lane],views:&[Scratch<'a>],experts:&mut dyn ExpertExecutor<'a>,
        dump:Option<StepDump>,diagnostics:Option<&mut Diagnostics>)->Result<()> {
        for (index,(lane,s)) in lanes.iter().zip(views).enumerate() { self.routed_back(layer,index,lane,s,experts)?; }
        if let Some(mut d)=dump {
            d.tensor(&self.commands,"residual","f32",&[rows,6144],self.scratch.get("residual")?)?;
            diagnostics.context("MiniMax diagnostic owner")?.finish(d)?;
        }
        Ok(())
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
    /// Uploads the step's attention metadata (see [`AttentionPlan`]); extend
    /// tiles end at the lane `cuts`.
    fn plan_attention(&self,batch:&BatchView,rows:usize,cuts:&[usize])->Result<AttentionPlan> {
        let s=&self.scratch;
        if rows<=self.decode_rows {
            let (stride,tables,lengths,offsets)=decode_tables(batch)?;
            ensure!(stride<=self.table_stride,MiniMaxError::Invalid("live page-table width"));
            self.commands.upload(s.get("page_table")?,&le_bytes(&tables))?;
            self.commands.upload(s.get("cache_lengths")?,&le_bytes(&lengths))?;
            self.commands.upload(s.get("cu_seqlens_q")?,&le_bytes(&offsets))?;
            return Ok(AttentionPlan::Decode{stride});
        }
        let (launches,tables,lengths,offsets,bounds)=extend_tables(batch,self.extend_rows,self.table_stride,cuts)?;
        self.commands.upload(s.get("page_table")?,&le_bytes(&tables))?;
        self.commands.upload(s.get("cache_lengths")?,&le_bytes(&lengths))?;
        self.commands.upload(s.get("cu_seqlens_q")?,&le_bytes(&offsets))?;
        self.commands.upload(&s.get("index_bounds")?.view(self.extend_rows*4,bounds.len()*4)?,&le_bytes(&bounds))?;
        Ok(AttentionPlan::Extend{launches})
    }
    /// A lane's attention: its rows (`s` views them) and its extend launches.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, layer:usize,lane:&Lane,batch:&BatchView,cache:&LayerCache<'a>,plan:&AttentionPlan,
        s:&Scratch<'a>,mut dump:Option<&mut StepDump>)->Result<()> {
        let rows=lane.rows;
        ffn::norm(&self.commands,s.get("residual")?,self.weights.layer(layer,"input_layernorm.weight")?,s.get("norm")?,rows)?;
        if let Some(d)=dump.as_deref_mut() { d.tensor(&self.commands,"norm","bf16",&[rows,6144],s.get("norm")?)?; }
        for (projection,stem,out) in [("q","q","q"),("kv","kv","kv"),("index_q","index_q","index_q"),("index_k","index_k","index_k")] {
            ffn::linear(&self.commands,&self.weights,s,layer,projection,&format!("self_attn.{stem}_proj"),s.get("norm")?,s.get(out)?,rows)?;
        }
        self.commands.launch("minimax_kv_split", &[s.get("kv")?,s.get("k")?,s.get("v")?], &[Scalar::I32(rows as i32)])?;
        for stem in ["q","k","index_q","index_k"] {
            let weight=self.weights.layer(layer,&format!("self_attn.{stem}_norm.weight"))?;
            self.commands.launch(&format!("minimax_{stem}_norm_rope"), &[s.get(stem)?,weight,weight,s.get("cos")?,s.get("sin")?,s.get("positions")?,s.get(&format!("{stem}_norm"))?], &[Scalar::I32(rows as i32)])?;
        }
        if let Some(d)=dump.as_deref_mut() {
            for (name,source,heads) in [("q","q_norm",64),("k","k_norm",4),("v","v",4),("index_q","index_q_norm",4),("index_k","index_k_norm",1)] {
                d.tensor(&self.commands,name,"bf16",&[rows,heads,128],s.get(source)?)?;
            }
        }
        self.commands.launch("minimax_index_k_pack", &[s.get("index_k_norm")?,s.get("slots")?,&cache.index,s.get("index_q_scales")?], &[Scalar::I32(rows as i32)])?;
        self.commands.launch("minimax_kv_pack", &[s.get("k_norm")?,s.get("v")?,s.get("slots")?,s.get("batch_ids")?,s.get("k_descale")?,s.get("v_descale")?,&cache.key,&cache.value], &[Scalar::I32(rows as i32)])?;
        self.commands.launch("minimax_index_q_quant", &[s.get("index_q_norm")?,s.get("slots")?,s.get("index_q_fp8")?,s.get("index_q_scales")?], &[Scalar::I32(rows as i32)])?;
        if let Some(d)=dump.as_deref_mut() {
            d.tensor(&self.commands,"index_q_fp8","u8",&[rows,4,128],s.get("index_q_fp8")?)?;
            d.tensor(&self.commands,"index_q_scales","f32",&[rows,4],s.get("index_q_scales")?)?;
            d.cache_rows(&self.commands,"index_k_fp8","u8",&[rows,1,128],&cache.index,&batch.slots,8448,0,128)?;
            d.cache_rows(&self.commands,"index_k_scales","f32",&[rows,1],&cache.index,&batch.slots,8448,8192,4)?;
            for (name,source) in [("k_fp8",&cache.key),("v_fp8",&cache.value)] {
                d.cache_rows(&self.commands,name,"u8",&[rows,4,128],source,&batch.slots,32768,0,512)?;
            }
        }
        match plan {
            // Every row is an independent causal decode request over the exact
            // paged prefix, including prefixes from prior chunks.
            AttentionPlan::Decode{stride} => {
                let stride=*stride;
                self.commands.launch("minimax_index_decode", &[&s.get("index_q_fp8")?.view(0,rows*512)?,
                    &s.get("index_q_scales")?.view(0,rows*16)?,&cache.index,s.get("page_table")?,s.get("cache_lengths")?,s.get("q2k")?,s.get("scratch")?], &[
                    Scalar::I32(rows as i32),Scalar::I32(stride as i32),Scalar::I32(stride as i32),Scalar::I64(batch.pool_pages as i64)])?;
                if let Some(d)=dump.as_deref_mut() { d.selection(&self.commands,s.get("q2k")?,rows,self.decode_rows)?; }
                self.commands.launch("minimax_attention_decode", &[&s.get("q_norm")?.view(0,rows*16384)?,&cache.key,&cache.value,
                    s.get("k_descale")?,s.get("v_descale")?,s.get("page_table")?,s.get("cache_lengths")?,s.get("cu_seqlens_q")?,s.get("q2k")?,
                    &s.get("attention")?.view(0,rows*16384)?,s.get("lse")?,s.get("scratch")?], &[
                    Scalar::I32(rows as i32),Scalar::I32(rows as i32),Scalar::I32(stride as i32),Scalar::I64(batch.pool_pages as i64)])?;
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
                    let table=tables.view(launch.slot*stride*4,stride*4)?;
                    if launch.first==0 || index==0 {
                        let end=(request.row_start+request.rows).min(lane.first+lane.rows)-request.row_start;
                        let pages=(request.start_position+end).div_ceil(64);
                        self.commands.index_stage(&cache.index,&table.view(0,pages*4)?,pages,s.get("index_keys")?,s.get("index_scales")?)?;
                    }
                    let (row,n)=(launch.row-lane.first,launch.rows);
                    self.commands.launch("minimax_index_prefill", &[&s.get("index_q_fp8")?.view(row*512,n*512)?,
                        &s.get("index_q_scales")?.view(row*16,n*16)?,s.get("index_keys")?,s.get("index_scales")?,
                        &bounds.view(0,n*4)?,&bounds.view(launch.bounds*4,n*4)?,s.get("q2k_extend")?,s.get("scratch")?], &[
                        Scalar::I32(n as i32),Scalar::I32((request.start_position+launch.first+n) as i32)])?;
                    if let Some(d)=dump.as_deref_mut() { d.selection(&self.commands,s.get("q2k_extend")?,n,tile)?; }
                    self.commands.launch("minimax_attention_extend", &[&s.get("q_norm")?.view(row*16384,n*16384)?,&cache.key,&cache.value,
                        s.get("k_descale")?,s.get("v_descale")?,&table,&s.get("cache_lengths")?.view(launch.slot*16,16)?,
                        &s.get("cu_seqlens_q")?.view(launch.slot*16,16)?,s.get("q2k_extend")?,
                        &s.get("attention")?.view(row*16384,n*16384)?,s.get("lse")?,s.get("scratch")?], &[
                        Scalar::I32(n as i32),Scalar::I32(1),Scalar::I32(stride as i32),Scalar::I64(batch.pool_pages as i64)])?;
                }
            }
        }
        if let Some(d)=dump.as_deref_mut() {
            d.finish_selection()?;
            d.tensor(&self.commands,"attention","bf16",&[rows,64,128],s.get("attention")?)?;
        }
        ffn::linear(&self.commands,&self.weights,s,layer,"o","self_attn.o_proj",s.get("attention")?,s.get("delta")?,rows)?;
        if let Some(d)=dump.as_deref_mut() { d.tensor(&self.commands,"attention_projected","bf16",&[rows,6144],s.get("delta")?)?; }
        self.residual(s,rows)?;
        if let Some(d)=dump { d.tensor(&self.commands,"post_attention","f32",&[rows,6144],s.get("residual")?)?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn minimax_batch_checks_causal_lengths_and_physical_slots() -> Result<()> {
        let batch=BatchView {positions:vec![63,64,4],slots:vec![63,128,260],pool_pages:5,
            requests:vec![RequestView{row_start:0,rows:2,start_position:63,pages:vec![0,2]},
                RequestView{row_start:2,rows:1,start_position:4,pages:vec![4]}]};
        assert_eq!(batch.validate(64,128)?,3);
        let mut bad=batch.clone(); bad.slots[1]=64; assert!(bad.validate(64,128).is_err());
        let mut bad=batch.clone(); bad.requests[1].row_start=1; assert!(bad.validate(64,128).is_err());
        let mut bad=batch.clone(); bad.positions[1]=65; assert!(bad.validate(64,128).is_err());
        let mut shared=batch.clone(); shared.requests[1].pages=vec![0]; shared.slots[2]=4;
        assert!(shared.validate(64,128).is_err());
        let shared=BatchView {positions:vec![64,64],slots:vec![128,256],pool_pages:5,
            requests:vec![RequestView{row_start:0,rows:1,start_position:64,pages:vec![0,2]},
                RequestView{row_start:1,rows:1,start_position:64,pages:vec![0,4]}]};
        assert_eq!(shared.validate(64,128)?,2);
        assert!(batch.validate(2,128).is_err());
        assert!(batch.validate(64,64).is_err());
        Ok(())
    }

    #[test] fn minimax_attention_metadata_is_built_once_per_step() -> Result<()> {
        // Two requests: 3 rows from 62 (pages 0, 2) and 1 row at 4 (page 4).
        let batch=BatchView {positions:vec![62,63,64,4],slots:vec![62,63,128,260],pool_pages:5,
            requests:vec![RequestView{row_start:0,rows:3,start_position:62,pages:vec![0,2]},
                RequestView{row_start:3,rows:1,start_position:4,pages:vec![4]}]};
        batch.validate(64,128)?;
        let (stride,tables,lengths,offsets)=decode_tables(&batch)?;
        assert_eq!(stride,2);
        assert_eq!(tables,[0,-1, 0,-1, 0,2, 4,-1]);
        assert_eq!(lengths,[63,64,65,5]);
        assert_eq!(offsets,[0,1,2,3,4]);
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
            Ok(lanes_at(&plan,rows,&cuts).into_iter().map(|l|(l.first,l.rows,
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
        assert_eq!(lanes_at(&AttentionPlan::Decode{stride:4},64,&[]),[Lane{first:0,rows:64,launches:0..0}]);
        Ok(())
    }
}
