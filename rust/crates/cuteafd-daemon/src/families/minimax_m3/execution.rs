//! One device, one append-only paged MSA pipeline, native checkpoint arithmetic.
use super::{MiniMaxError, ffn, head::TargetHeadOwner, scratch::Scratch, weights::Weights};
use crate::shared::{memory::device::Device, token_io::DeviceLogits};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{minimax::{Buffer, Commands}, programs::{Programs,Scalar}};
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

pub(super) trait ExpertExecutor<'a> {
    fn begin(&mut self, _waves:usize, _kind:ExpertV2SourceKind) -> Result<()> { Ok(()) }
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
    context:usize,
    decode_rows:usize,
    table_stride:usize,
    poisoned:std::cell::Cell<bool>,
}
impl<'a> Execution<'a> {
    pub fn load(device:Device<'a>, programs:&'a Programs<'a>, checkpoint:&Checkpoint,
        cfg:&MiniMaxM3Config, ledger:&RuntimeLedger) -> Result<Self> {
        cfg.release_geometry()?;
        ensure!(cfg.rms_norm_eps==1e-6 && cfg.swiglu_alpha==1.702 && cfg.swiglu_limit==7.0
            && cfg.routed_scale==2.0,MiniMaxError::Unsupported("compiled norm, activation or route arithmetic"));
        ensure!(cfg.moe[..3].iter().all(|v|!*v) && cfg.moe[3..].iter().all(|v|*v),
            MiniMaxError::Unsupported("dense/routed layer ordering"));
        let rows=usize::try_from(ledger.bytes("norm").context("MiniMax norm admission")? / (6144*2))?;
        let context=usize::try_from(ledger.bytes("cos").context("MiniMax rope admission")? / 64)?;
        let commands=Commands::new(device.library,device.id,programs)?;
        let decode_rows=commands.capacity("minimax_index_decode")?;
        ensure!(decode_rows>0 && decode_rows<=64 && commands.capacity("minimax_attention_decode")?==decode_rows,
            MiniMaxError::Invalid("index/attention capacity stride"));
        let scratch=Scratch::new(device,ledger)?;
        let table_stride=scratch.get("page_table")?.raw().bytes/(decode_rows*4);
        ensure!(table_stride>=context.div_ceil(64),MiniMaxError::Invalid("page-table admission"));
        let weights=Weights::load(device,checkpoint,&commands)?;
        commands.launch("minimax_rope", &[scratch.get("cos")?,scratch.get("sin")?], &[
            Scalar::I32(context as i32),Scalar::F32(cfg.rope_theta as f32),Scalar::F32(1.0),
            Scalar::F32(0.0),Scalar::F32(1.0),Scalar::F32(1.0)])?;
        let ones=vec![1.0f32.to_le_bytes();decode_rows*4].concat();
        commands.upload(scratch.get("k_descale")?,&ones)?;
        commands.upload(scratch.get("v_descale")?,&ones)?;
        commands.upload(scratch.get("batch_ids")?,&vec![0;rows*4])?;
        commands.drain()?;
        let head=TargetHeadOwner::new(&commands,&scratch,rows,cfg.vocab_size)?;
        Ok(Self{commands,head,weights,scratch,cfg:cfg.clone(),rows,context,decode_rows,table_stride,
            poisoned:std::cell::Cell::new(false)})
    }
    pub fn stream(&self)->*mut std::ffi::c_void { self.commands.raw() }
    pub fn drain(&self)->Result<()> { self.commands.drain() }
    pub fn execute(&self, embedded:&Buffer<'a>, batch:&BatchView, caches:&[LayerCache<'a>],
        experts:&mut dyn ExpertExecutor<'a>, kind:ExpertV2SourceKind) -> Result<DeviceLogits> {
        ensure!(!self.poisoned.get(),MiniMaxError::Invalid("previous native execution failure; restart required"));
        ensure!(matches!(kind,ExpertV2SourceKind::Decode|ExpertV2SourceKind::Prefill),
            MiniMaxError::Unsupported("expert source kind"));
        let rows=batch.validate(self.rows,self.context)?;
        ensure!(caches.len()==self.cfg.layers,MiniMaxError::Invalid("layer cache inventory"));
        for cache in caches { ensure!(cache.key.raw().bytes>=batch.pool_pages*32768
            && cache.value.raw().bytes>=batch.pool_pages*32768 && cache.index.raw().bytes>=batch.pool_pages*8448,
            MiniMaxError::Invalid("physical cache extent")); }
        let result=(||{
            experts.begin(self.cfg.moe_layers(),kind)?;
            self.commands.upload(self.scratch.get("positions")?,&batch.positions.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            self.commands.upload(self.scratch.get("slots")?,&batch.slots.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            self.commands.widen(embedded,self.scratch.get("residual")?,rows*6144)?;
            for (layer,cache) in caches.iter().enumerate() {
                self.attention(layer,rows,batch,cache)?;
                ffn::norm(&self.commands,self.scratch.get("residual")?,self.weights.layer(layer,"post_attention_layernorm.weight")?,self.scratch.get("norm")?,rows)?;
                if self.cfg.moe[layer] {
                    self.commands.launch("minimax_router_scores", &[self.scratch.get("norm")?,
                        self.weights.layer(layer,"block_sparse_moe.gate.weight")?,self.scratch.get("router_logits")?], &[Scalar::I32(rows as i32)])?;
                    self.commands.launch("minimax_route", &[self.scratch.get("router_logits")?,
                        self.weights.layer(layer,"block_sparse_moe.e_score_correction_bias")?,self.scratch.get("route_ids")?,self.scratch.get("route_weights")?], &[Scalar::I32(rows as i32)])?;
                    experts.dispatch(layer,rows,self.scratch.get("route_ids")?,self.scratch.get("route_weights")?,self.scratch.get("norm")?,&self.commands)?;
                    ffn::dense(&self.commands,&self.weights,&self.scratch,layer,rows,true)?;
                    experts.collect(layer,rows,self.scratch.get("routed")?,&self.commands)?;
                    self.commands.add_bf16(self.scratch.get("shared")?,self.scratch.get("routed")?,self.scratch.get("delta")?,rows*6144)?;
                } else { ffn::dense(&self.commands,&self.weights,&self.scratch,layer,rows,false)?; }
                self.residual(rows)?;
            }
            self.head.logits(&self.commands,&self.weights,&self.scratch,rows)
        })();
        // All device/cache/expert borrows end only after queued work retires.
        let drained=self.commands.drain();
        let finished=experts.finish();
        let result=result.and_then(|value|{drained?;finished?;Ok(value)});
        if result.is_err() { self.poisoned.set(true); experts.abort(); }
        result
    }
    fn residual(&self,rows:usize)->Result<()> {
        self.commands.launch("minimax_residual", &[self.scratch.get("residual")?,self.scratch.get("delta")?,self.scratch.get("residual")?], &[Scalar::I32(rows as i32)])
    }
    fn attention(&self, layer:usize,rows:usize,batch:&BatchView,cache:&LayerCache<'a>)->Result<()> {
        let s=&self.scratch;
        ffn::norm(&self.commands,s.get("residual")?,self.weights.layer(layer,"input_layernorm.weight")?,s.get("norm")?,rows)?;
        for (projection,stem,out) in [("q","q","q"),("kv","kv","kv"),("index_q","index_q","index_q"),("index_k","index_k","index_k")] {
            ffn::linear(&self.commands,&self.weights,s,layer,projection,&format!("self_attn.{stem}_proj"),s.get("norm")?,s.get(out)?,rows)?;
        }
        self.commands.launch("minimax_kv_split", &[s.get("kv")?,s.get("k")?,s.get("v")?], &[Scalar::I32(rows as i32)])?;
        for stem in ["q","k","index_q","index_k"] {
            let weight=self.weights.layer(layer,&format!("self_attn.{stem}_norm.weight"))?;
            self.commands.launch(&format!("minimax_{stem}_norm_rope"), &[s.get(stem)?,weight,weight,s.get("cos")?,s.get("sin")?,s.get("positions")?,s.get(&format!("{stem}_norm"))?], &[Scalar::I32(rows as i32)])?;
        }
        self.commands.launch("minimax_index_k_pack", &[s.get("index_k_norm")?,s.get("slots")?,&cache.index,s.get("index_q_scales")?], &[Scalar::I32(rows as i32)])?;
        self.commands.launch("minimax_kv_pack", &[s.get("k_norm")?,s.get("v")?,s.get("slots")?,s.get("batch_ids")?,s.get("k_descale")?,s.get("v_descale")?,&cache.key,&cache.value], &[Scalar::I32(rows as i32)])?;
        self.commands.launch("minimax_index_q_quant", &[s.get("index_q_norm")?,s.get("slots")?,s.get("index_q_fp8")?,s.get("index_q_scales")?], &[Scalar::I32(rows as i32)])?;
        // Every tile treats its queries as independent causal requests. It
        // reads the exact paged prefix, including prefixes from prior chunks.
        for start in (0..rows).step_by(self.decode_rows) {
            let count=(rows-start).min(self.decode_rows);
            let stride=batch.positions[start..start+count].iter().copied().max().unwrap() as usize/64+1;
            ensure!(stride<=self.table_stride,MiniMaxError::Invalid("live page-table width"));
            let mut tables=vec![-1i32;count*stride];
            let mut lengths=Vec::with_capacity(count);
            for row in 0..count {
                let request=batch.request(start+row)?;
                let length=batch.positions[start+row] as usize+1;
                let width=length.div_ceil(64);
                tables[row*stride..row*stride+width].copy_from_slice(&request.pages[..width]);
                lengths.push(length as i32);
            }
            self.commands.upload(s.get("page_table")?,&tables.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            self.commands.upload(s.get("cache_lengths")?,&lengths.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            self.commands.upload(s.get("cu_seqlens_q")?,&(0..=count as i32).flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
            self.commands.launch("minimax_index_decode", &[&s.get("index_q_fp8")?.view(start*512,count*512)?,
                &s.get("index_q_scales")?.view(start*16,count*16)?,&cache.index,s.get("page_table")?,s.get("cache_lengths")?,s.get("q2k")?,s.get("scratch")?], &[
                Scalar::I32(count as i32),Scalar::I32(stride as i32),Scalar::I32(stride as i32),Scalar::I64(batch.pool_pages as i64)])?;
            self.commands.launch("minimax_attention_decode", &[&s.get("q_norm")?.view(start*16384,count*16384)?,&cache.key,&cache.value,
                s.get("k_descale")?,s.get("v_descale")?,s.get("page_table")?,s.get("cache_lengths")?,s.get("cu_seqlens_q")?,s.get("q2k")?,
                &s.get("attention")?.view(start*16384,count*16384)?,s.get("lse")?,s.get("scratch")?], &[
                Scalar::I32(count as i32),Scalar::I32(count as i32),Scalar::I32(stride as i32),Scalar::I64(batch.pool_pages as i64)])?;
        }
        ffn::linear(&self.commands,&self.weights,s,layer,"o","self_attn.o_proj",s.get("attention")?,s.get("delta")?,rows)?;
        self.residual(rows)
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
}
