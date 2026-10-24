//! One device, one append-only paged MSA pipeline, native checkpoint arithmetic.
use super::{MiniMaxError, diagnostics::{Diagnostics, StepDump}, ffn, head::TargetHeadOwner, scratch::Scratch, weights::Weights};
use crate::shared::{decode_graph::{census, GraphBank, GraphDecision, GraphOwner, GraphPolicy},
    memory::device::Device, token_io::DeviceLogits};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{minimax::{Buffer, Commands, Retention}, programs::{Programs,Scalar}};
use cuteafd_transport::ExpertV2SourceKind;
use cuteafd_loader::{families::minimax_m3::config::MiniMaxM3Config,
    plan::{checkpoint::Checkpoint,families::minimax::admission::RuntimeLedger}};

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

/// Decode metadata: per-row tables at a fixed `stride` (-1 padded), lengths
/// and packed offsets. The decode kernels visit only each row's live pages
/// (its cache length), so a wider table changes neither work nor results,
/// and one stride for every step lets a recorded step replay at any context.
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

/// Census bank of a recorded whole decode step: `CUTEAFD_GRAPH_EAGER=minimax_step`
/// runs every step eagerly while the recordings stay admitted (an A/B arm).
const STEP_BANK: &std::ffi::CStr = c"minimax_step";

/// What the startup recordings were made against (the keys are row counts
/// only): the step input, cache, expert owner, pool and decode table stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Binding { embedded:usize, cache:usize, experts:usize, pool_pages:usize, stride:usize }

/// Whole decode steps recorded at startup, one per admitted row count
/// ([`Execution::warm_graphs`]); each owner keeps what its launches retained.
/// Declared after `Execution::commands`, whose drop drains the stream first.
struct StepGraphs<'a> {
    bank: GraphBank<usize, GraphOwner<'a, Retention<'a>>>,
    binding: Option<Binding>,
    /// `CUTEAFD_MINIMAX_GRAPH_VERIFY=1`: each replayed step runs again eagerly
    /// and the two steps' logits are compared bit for bit (a diagnostic).
    verify: bool,
    replays: u64,
}

/// Extend metadata: one launch per request tile of at most `tile` rows. Each
/// launch's table row holds its request's pages through the step (padded
/// with -1 to `stride`), its 16-byte entries hold the cache length after its
/// rows and its packed offsets [0, rows], and `bounds` its rows' causal key
/// ends after the `tile` leading zero key starts.
fn extend_tables(batch:&BatchView,tile:usize,stride:usize)->Result<(Vec<Launch>,Vec<i32>,Vec<i32>,Vec<i32>,Vec<i32>)> {
    let (mut launches,mut tables,mut lengths,mut offsets,mut bounds)=(Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new());
    for (request,view) in batch.requests.iter().enumerate() {
        let end=view.start_position+view.rows;
        let width=end.div_ceil(64);
        ensure!(width<=stride,MiniMaxError::Invalid("extend page-table width"));
        for first in (0..view.rows).step_by(tile) {
            let rows=(view.rows-first).min(tile);
            launches.push(Launch{request,row:view.row_start+first,first,rows,slot:launches.len(),bounds:tile+bounds.len()});
            tables.extend_from_slice(&view.pages[..width]);
            tables.resize(launches.len()*stride,-1);
            lengths.extend([(view.start_position+first+rows) as i32,0,0,0]);
            offsets.extend([0,rows as i32,0,0]);
            bounds.extend((0..rows).map(|i|(view.start_position+first+i+1) as i32));
            bounds.resize(bounds.len().next_multiple_of(4),0);
        }
    }
    Ok((launches,tables,lengths,offsets,bounds))
}

fn le_bytes(values:&[i32])->Vec<u8> { values.iter().flat_map(|v|v.to_le_bytes()).collect() }

pub(super) trait ExpertExecutor<'a> {
    fn begin(&mut self, _waves:usize, _kind:ExpertV2SourceKind) -> Result<()> { Ok(()) }
    /// Before and after recording a step at startup (its dispatches and
    /// collects are recorded, not run): announces nothing and drops what a
    /// failed recording left uncollected.
    fn record(&mut self, _kind:ExpertV2SourceKind) -> Result<()> { Ok(()) }
    fn dispatch(&mut self, layer:usize, rows:usize, ids:&Buffer<'a>, weights:&Buffer<'a>,
        input:&Buffer<'a>, commands:&Commands<'a>) -> Result<()>;
    fn collect(&mut self, layer:usize, rows:usize, output:&Buffer<'a>, commands:&Commands<'a>) -> Result<()>;
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
    rows:usize,
    logit_rows:usize,
    context:usize,
    decode_rows:usize,
    extend_rows:usize,
    table_stride:usize,
    poisoned:std::cell::Cell<bool>,
    diagnostics:std::cell::RefCell<Option<Diagnostics>>,
    draft:Option<super::dspark::Dspark<'a>>,
    draft_causal:bool,
    graphs:std::cell::RefCell<StepGraphs<'a>>,
}
impl<'a> Execution<'a> {
    pub fn load(device:Device<'a>, programs:&'a Programs<'a>, checkpoint:&Checkpoint,
        cfg:&MiniMaxM3Config, ledger:&RuntimeLedger) -> Result<Self> {
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
        let graphs=std::cell::RefCell::new(StepGraphs{bank:GraphBank::with_policy(GraphPolicy::FixedStartup),binding:None,
            verify:matches!(std::env::var("CUTEAFD_MINIMAX_GRAPH_VERIFY").as_deref(),Ok("1")),replays:0});
        Ok(Self{commands,head,weights,scratch,cfg:cfg.clone(),rows,logit_rows,context,decode_rows,extend_rows,table_stride,
            poisoned:std::cell::Cell::new(false),diagnostics,draft:None,draft_causal:false,graphs})
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
        let waves=self.cfg.moe[..=last_layer].iter().filter(|&&moe|moe).count();
        // A full-model decode step replays the recording of its row count.
        let replay=(kind==ExpertV2SourceKind::Decode && rows<=self.decode_rows && last_layer+1==self.cfg.layers
            && logits==(0..rows)).then(|| self.binding(embedded,caches,batch.pool_pages,&*experts)).transpose()?.flatten();
        let result=(||{
            experts.begin(waves,kind)?;
            self.commands.upload(self.scratch.get("positions")?,&batch.positions.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            self.commands.upload(self.scratch.get("slots")?,&batch.slots.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            let plan=self.plan_attention(batch,rows)?;
            if replay.is_some() {
                let mut graphs=self.graphs.borrow_mut();
                if let (GraphDecision::Replay,Some(owner))=graphs.bank.enqueue(&rows) {
                    let raw=owner.raw;
                    graphs.replays+=1;
                    drop(graphs);
                    // SAFETY: the bank keeps the executable and every operand it
                    // recorded; this step's metadata uploads precede it on the stream.
                    unsafe { census::dispatch(self.commands.library(),self.commands.raw(),STEP_BANK,rows,Some(raw),
                        || self.layers(embedded,Some(batch),batch.pool_pages,caches,experts,&plan,rows,last_layer,
                            logits.clone(),&mut None).map(|_|()))? };
                    return self.head.output(&self.commands,&self.scratch,rows).map(|logits|(Some(logits),true));
                }
            }
            self.layers(embedded,Some(batch),batch.pool_pages,caches,experts,&plan,rows,last_layer,logits.clone(),
                &mut diagnostics).map(|logits|(logits,false))
        })();
        // All device/cache/expert borrows end only after queued work retires.
        let drained=self.commands.drain();
        let finished=experts.finish();
        let result=result.and_then(|value|{drained?;finished?;Ok(value)});
        if result.is_err() { self.poisoned.set(true); experts.abort(); }
        let (logits,replayed)=result?;
        if replayed && self.graphs.borrow().verify { self.verify(embedded,batch,caches,experts,waves,last_layer)?; }
        Ok(logits)
    }
    /// The recordings' binding when this step's input, cache, expert owner and
    /// pool are theirs; `None` before the warm-up. Any other binding is refused.
    fn binding(&self, embedded:&Buffer<'a>, caches:&[LayerCache<'a>], pool_pages:usize,
        experts:&dyn ExpertExecutor<'a>) -> Result<Option<Binding>> {
        let step=Binding { embedded:embedded.raw().ptr as usize, cache:caches[0].key.raw().ptr as usize,
            experts:std::ptr::from_ref(experts).cast::<()>() as usize, pool_pages, stride:self.table_stride.min(pool_pages) };
        match self.graphs.borrow().binding {
            None => Ok(None),
            Some(binding) => {
                ensure!(binding==step,MiniMaxError::Invalid("decode step outside its recordings' binding"));
                Ok(Some(binding))
            }
        }
    }
    /// Records the admitted decode inventory before ready: one whole step per
    /// row count in `keys`, each on the drained stream against the server's
    /// step input, cache, pool and expert owner, the Spark exchange included
    /// (its sequence numbers live in the pinned mailbox). Nothing executes; a
    /// replay follows each step's metadata uploads. Returns the device bytes
    /// the recordings took.
    pub fn warm_graphs(&self, keys:&[usize], embedded:&Buffer<'a>, caches:&[LayerCache<'a>], pool_pages:usize,
        experts:&mut dyn ExpertExecutor<'a>) -> Result<u64> {
        ensure!(!self.poisoned.get(),MiniMaxError::Invalid("previous native execution failure; restart required"));
        ensure!(self.diagnostics.borrow().is_none(),MiniMaxError::Unsupported("decode graphs beside diagnostic dumps"));
        ensure!(caches.len()==self.cfg.layers && keys.iter().all(|rows|(1..=self.decode_rows).contains(rows)),
            MiniMaxError::Invalid("decode graph inventory"));
        let stride=self.table_stride.min(pool_pages);
        let binding=Binding { embedded:embedded.raw().ptr as usize, cache:caches[0].key.raw().ptr as usize,
            experts:std::ptr::from_ref(&*experts).cast::<()>() as usize, pool_pages, stride };
        let library=self.commands.library();
        let device=Device { library, id:self.commands.device() };
        self.commands.drain()?;
        let (free,_)=library.cuda_memory_info()?;
        let mut graphs=self.graphs.borrow_mut();
        ensure!(graphs.binding.is_none(),MiniMaxError::Invalid("decode graphs already recorded"));
        let started=std::time::Instant::now();
        graphs.bank.warm(keys,|&rows| {
            ensure!(self.commands.idle(),MiniMaxError::Invalid("recording on a busy stream"));
            let (before,_)=library.cuda_memory_info()?;
            experts.record(ExpertV2SourceKind::Decode)?;
            // SAFETY: startup owns this drained stream; capture records and runs nothing.
            device.run(|| unsafe { library.cuda_graph_begin_capture(self.commands.raw()) })?;
            let queued=self.layers(embedded,None,pool_pages,caches,experts,&AttentionPlan::Decode{stride},rows,
                self.cfg.layers-1,0..rows,&mut None);
            // SAFETY: ends the capture begun above, even when recording failed.
            let ended=device.run(|| unsafe { library.cuda_graph_end_capture(self.commands.raw()) });
            let pins=self.commands.take_retained();
            let reset=experts.record(ExpertV2SourceKind::Decode);
            match (queued,ended) {
                (Ok(_),Ok(raw)) => {
                    // SAFETY: the owner keeps every recorded operand; Execution
                    // drains its stream before the bank (or the pins) go.
                    let owner=unsafe { GraphOwner::new(library,device.id,raw,pins) }?;
                    reset?;
                    let (after,_)=library.cuda_memory_info()?;
                    Ok((owner,Some(before.saturating_sub(after) as u64)))
                }
                (Err(error),Ok(raw)) => {
                    // SAFETY: the failed recording was never launched.
                    device.run(|| unsafe { library.cuda_graph_exec_destroy(raw) })?;
                    Err(error)
                }
                (Err(error),Err(ended)) => Err(error.context(format!("ending the failed recording: {ended:#}"))),
                (Ok(_),Err(error)) => Err(error),
            }
        })?;
        graphs.binding=Some(binding);
        let (after,_)=library.cuda_memory_info()?;
        let bytes=free.saturating_sub(after) as u64;
        tracing::info!(executables=keys.len(),bytes,ms=started.elapsed().as_secs_f64()*1e3,
            "MiniMax decode steps recorded at startup");
        Ok(bytes)
    }
    /// The step body: residual widening, every layer through `last_layer`,
    /// then the head over `logits` (an empty range skips it).
    #[allow(clippy::too_many_arguments)]
    fn layers(&self, embedded:&Buffer<'a>, batch:Option<&BatchView>, pool_pages:usize, caches:&[LayerCache<'a>],
        experts:&mut dyn ExpertExecutor<'a>, plan:&AttentionPlan, rows:usize, last_layer:usize,
        logits:std::ops::Range<usize>, diagnostics:&mut Option<Diagnostics>) -> Result<Option<DeviceLogits>> {
        self.commands.widen(embedded,self.scratch.get("residual")?,rows*6144)?;
        for (layer,cache) in caches.iter().enumerate().take(last_layer+1) {
            let mut dump=diagnostics.as_ref().filter(|d|d.layer==layer)
                .map(|d|d.begin(batch.context("MiniMax diagnostic step needs its batch")?)).transpose()?;
            if let Some(d)=&mut dump { d.tensor(&self.commands,"input","f32",&[rows,6144],self.scratch.get("residual")?)?; }
            self.attention(layer,rows,pool_pages,batch,cache,plan,dump.as_mut())?;
            ffn::norm(&self.commands,self.scratch.get("residual")?,self.weights.layer(layer,"post_attention_layernorm.weight")?,self.scratch.get("norm")?,rows)?;
            if let Some(d)=&mut dump { d.tensor(&self.commands,"ffn_norm","bf16",&[rows,6144],self.scratch.get("norm")?)?; }
            if self.cfg.moe[layer] {
                self.commands.launch("minimax_router_scores", &[self.scratch.get("norm")?,
                    self.weights.layer(layer,"block_sparse_moe.gate.weight")?,self.scratch.get("router_logits")?], &[Scalar::I32(rows as i32)])?;
                self.commands.launch("minimax_route", &[self.scratch.get("router_logits")?,
                    self.weights.layer(layer,"block_sparse_moe.e_score_correction_bias")?,self.scratch.get("route_ids")?,self.scratch.get("route_weights")?], &[Scalar::I32(rows as i32)])?;
                if let Some(d)=&mut dump {
                    d.tensor(&self.commands,"router_ids","i32",&[rows,4],self.scratch.get("route_ids")?)?;
                    d.tensor(&self.commands,"router_weights","f32",&[rows,4],self.scratch.get("route_weights")?)?;
                }
                experts.dispatch(layer,rows,self.scratch.get("route_ids")?,self.scratch.get("route_weights")?,self.scratch.get("norm")?,&self.commands)?;
                ffn::dense(&self.commands,&self.weights,&self.scratch,layer,rows,true)?;
                if let Some(d)=&mut dump { d.tensor(&self.commands,"shared","bf16",&[rows,6144],self.scratch.get("shared")?)?; }
                experts.collect(layer,rows,self.scratch.get("routed")?,&self.commands)?;
                self.commands.add_bf16(self.scratch.get("shared")?,self.scratch.get("routed")?,self.scratch.get("delta")?,rows*6144)?;
            } else { ffn::dense(&self.commands,&self.weights,&self.scratch,layer,rows,false)?; }
            self.residual(rows)?;
            if let Some(draft)=&self.draft { draft.tap(&self.commands,layer,self.scratch.get("residual")?,rows)?; }
            if let Some(mut d)=dump {
                d.tensor(&self.commands,"residual","f32",&[rows,6144],self.scratch.get("residual")?)?;
                diagnostics.as_mut().context("MiniMax diagnostic owner")?.finish(d)?;
            }
        }
        if last_layer+1==self.cfg.layers && !logits.is_empty() {
            self.head.logits(&self.commands,&self.weights,&self.scratch,logits).map(Some)
        } else { Ok(None) }
    }
    /// `CUTEAFD_MINIMAX_GRAPH_VERIFY=1`: runs the replayed step again eagerly
    /// (same inputs: the cache rows it appends are rewritten identically) and
    /// compares the logits bit for bit.
    fn verify(&self, embedded:&Buffer<'a>, batch:&BatchView, caches:&[LayerCache<'a>],
        experts:&mut dyn ExpertExecutor<'a>, waves:usize, last_layer:usize) -> Result<()> {
        let library=self.commands.library();
        let rows=batch.positions.len();
        let replayed=self.head.output(&self.commands,&self.scratch,rows)?.to_host(library)?;
        let eager=(||{
            experts.begin(waves,ExpertV2SourceKind::Decode)?;
            self.layers(embedded,Some(batch),batch.pool_pages,caches,experts,&AttentionPlan::Decode{stride:self.decode_stride(batch)},
                rows,last_layer,0..rows,&mut None)?.context("MiniMax verified step has no logits")
        })();
        let drained=self.commands.drain();
        let finished=experts.finish();
        let eager=eager.and_then(|logits|{drained?;finished?;logits.to_host(library)});
        let eager=match eager { Ok(eager)=>eager, Err(error)=>{ self.poisoned.set(true); experts.abort(); return Err(error); } };
        let differing=replayed.iter().zip(&eager).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
        let max=replayed.iter().zip(&eager).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
        let replays=self.graphs.borrow().replays;
        if differing==0 { tracing::info!(rows,replays,"MiniMax decode graph replay matches eager bit for bit"); }
        else { tracing::error!(rows,replays,differing,max_abs=max,"MiniMax decode graph replay differs from eager"); }
        Ok(())
    }
    fn residual(&self,rows:usize)->Result<()> {
        self.commands.launch("minimax_residual", &[self.scratch.get("residual")?,self.scratch.get("delta")?,self.scratch.get("residual")?], &[Scalar::I32(rows as i32)])
    }
    /// The page-table stride of every decode step: wide enough for any live
    /// request (none holds more than the pool's pages or the context's).
    fn decode_stride(&self,batch:&BatchView)->usize { self.table_stride.min(batch.pool_pages) }
    /// Uploads the step's attention metadata (see [`AttentionPlan`]).
    fn plan_attention(&self,batch:&BatchView,rows:usize)->Result<AttentionPlan> {
        let s=&self.scratch;
        if rows<=self.decode_rows {
            let stride=self.decode_stride(batch);
            let (tables,lengths,offsets)=decode_tables(batch,stride)?;
            self.commands.upload(s.get("page_table")?,&le_bytes(&tables))?;
            self.commands.upload(s.get("cache_lengths")?,&le_bytes(&lengths))?;
            self.commands.upload(s.get("cu_seqlens_q")?,&le_bytes(&offsets))?;
            return Ok(AttentionPlan::Decode{stride});
        }
        let (launches,tables,lengths,offsets,bounds)=extend_tables(batch,self.extend_rows,self.table_stride)?;
        self.commands.upload(s.get("page_table")?,&le_bytes(&tables))?;
        self.commands.upload(s.get("cache_lengths")?,&le_bytes(&lengths))?;
        self.commands.upload(s.get("cu_seqlens_q")?,&le_bytes(&offsets))?;
        self.commands.upload(&s.get("index_bounds")?.view(self.extend_rows*4,bounds.len()*4)?,&le_bytes(&bounds))?;
        Ok(AttentionPlan::Extend{launches})
    }
    /// One layer's attention; `batch` is absent only while recording a decode
    /// step, which has neither extend launches nor dumps.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, layer:usize,rows:usize,pool_pages:usize,batch:Option<&BatchView>,cache:&LayerCache<'a>,
        plan:&AttentionPlan,mut dump:Option<&mut StepDump>)->Result<()> {
        let s=&self.scratch;
        let step=|| batch.context("MiniMax extend launches and dumps need their batch");
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
            let slots=&step()?.slots;
            d.cache_rows(&self.commands,"index_k_fp8","u8",&[rows,1,128],&cache.index,slots,8448,0,128)?;
            d.cache_rows(&self.commands,"index_k_scales","f32",&[rows,1],&cache.index,slots,8448,8192,4)?;
            for (name,source) in [("k_fp8",&cache.key),("v_fp8",&cache.value)] {
                d.cache_rows(&self.commands,name,"u8",&[rows,4,128],source,slots,32768,0,512)?;
            }
        }
        match plan {
            // Every row is an independent causal decode request over the exact
            // paged prefix, including prefixes from prior chunks.
            AttentionPlan::Decode{stride} => {
                let stride=*stride;
                self.commands.launch("minimax_index_decode", &[&s.get("index_q_fp8")?.view(0,rows*512)?,
                    &s.get("index_q_scales")?.view(0,rows*16)?,&cache.index,s.get("page_table")?,s.get("cache_lengths")?,s.get("q2k")?,s.get("scratch")?], &[
                    Scalar::I32(rows as i32),Scalar::I32(stride as i32),Scalar::I32(stride as i32),Scalar::I64(pool_pages as i64)])?;
                if let Some(d)=dump.as_deref_mut() { d.selection(&self.commands,s.get("q2k")?,rows,self.decode_rows)?; }
                self.commands.launch("minimax_attention_decode", &[&s.get("q_norm")?.view(0,rows*16384)?,&cache.key,&cache.value,
                    s.get("k_descale")?,s.get("v_descale")?,s.get("page_table")?,s.get("cache_lengths")?,s.get("cu_seqlens_q")?,s.get("q2k")?,
                    &s.get("attention")?.view(0,rows*16384)?,s.get("lse")?,s.get("scratch")?], &[
                    Scalar::I32(rows as i32),Scalar::I32(rows as i32),Scalar::I32(stride as i32),Scalar::I64(pool_pages as i64)])?;
            }
            // Each request's index keys through this step are staged as the
            // contiguous rows the prefill scorer reads; its tiles then select
            // blocks and attend over the paged cache like the decode kernels.
            AttentionPlan::Extend{launches} => {
                let (stride,tile)=(self.table_stride,self.extend_rows);
                let (tables,bounds)=(s.get("page_table")?,s.get("index_bounds")?);
                let batch=step()?;
                for launch in launches {
                    let request=&batch.requests[launch.request];
                    let table=tables.view(launch.slot*stride*4,stride*4)?;
                    if launch.first==0 {
                        let pages=(request.start_position+request.rows).div_ceil(64);
                        self.commands.index_stage(&cache.index,&table.view(0,pages*4)?,pages,s.get("index_keys")?,s.get("index_scales")?)?;
                    }
                    let (row,n)=(launch.row,launch.rows);
                    self.commands.launch("minimax_index_prefill", &[&s.get("index_q_fp8")?.view(row*512,n*512)?,
                        &s.get("index_q_scales")?.view(row*16,n*16)?,s.get("index_keys")?,s.get("index_scales")?,
                        &bounds.view(0,n*4)?,&bounds.view(launch.bounds*4,n*4)?,s.get("q2k_extend")?,s.get("scratch")?], &[
                        Scalar::I32(n as i32),Scalar::I32((request.start_position+launch.first+n) as i32)])?;
                    if let Some(d)=dump.as_deref_mut() { d.selection(&self.commands,s.get("q2k_extend")?,n,tile)?; }
                    self.commands.launch("minimax_attention_extend", &[&s.get("q_norm")?.view(row*16384,n*16384)?,&cache.key,&cache.value,
                        s.get("k_descale")?,s.get("v_descale")?,&table,&s.get("cache_lengths")?.view(launch.slot*16,16)?,
                        &s.get("cu_seqlens_q")?.view(launch.slot*16,16)?,s.get("q2k_extend")?,
                        &s.get("attention")?.view(row*16384,n*16384)?,s.get("lse")?,s.get("scratch")?], &[
                        Scalar::I32(n as i32),Scalar::I32(1),Scalar::I32(stride as i32),Scalar::I64(pool_pages as i64)])?;
                }
            }
        }
        if let Some(d)=dump.as_deref_mut() {
            d.finish_selection()?;
            d.tensor(&self.commands,"attention","bf16",&[rows,64,128],s.get("attention")?)?;
        }
        ffn::linear(&self.commands,&self.weights,s,layer,"o","self_attn.o_proj",s.get("attention")?,s.get("delta")?,rows)?;
        if let Some(d)=dump.as_deref_mut() { d.tensor(&self.commands,"attention_projected","bf16",&[rows,6144],s.get("delta")?)?; }
        self.residual(rows)?;
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
        let (tables,lengths,offsets)=decode_tables(&batch,2)?;
        assert_eq!(tables,[0,-1, 0,-1, 0,2, 4,-1]);
        assert_eq!(lengths,[63,64,65,5]);
        assert_eq!(offsets,[0,1,2,3,4]);
        // Every decode step uses one fixed stride; live entries keep their place.
        let (wide,_,_)=decode_tables(&batch,4)?;
        assert_eq!(wide,[0,-1,-1,-1, 0,-1,-1,-1, 0,2,-1,-1, 4,-1,-1,-1]);
        assert!(decode_tables(&batch,1).is_err(),"a row wider than the stride is refused");
        // A two-row tile splits the first request; launch metadata is 16-byte aligned.
        let (launches,tables,lengths,offsets,bounds)=extend_tables(&batch,2,4)?;
        assert_eq!(launches,[Launch{request:0,row:0,first:0,rows:2,slot:0,bounds:2},
            Launch{request:0,row:2,first:2,rows:1,slot:1,bounds:6},Launch{request:1,row:3,first:0,rows:1,slot:2,bounds:10}]);
        assert_eq!(tables,[0,2,-1,-1, 0,2,-1,-1, 4,-1,-1,-1]);
        assert_eq!(lengths,[64,0,0,0, 65,0,0,0, 5,0,0,0]);
        assert_eq!(offsets,[0,2,0,0, 0,1,0,0, 0,1,0,0]);
        assert_eq!(bounds,[63,64,0,0, 65,0,0,0, 5,0,0,0]);
        assert!(extend_tables(&batch,2,1).is_err(),"a request wider than the admitted table is refused");
        Ok(())
    }
}
