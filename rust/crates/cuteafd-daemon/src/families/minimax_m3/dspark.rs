//! MiniMax adapter to shared sliding attention and Markov heads.
//! Semantics: vllm PR 63232 at 306db61dc16a2c7d2655e4d2b29f7d6a6c4409d6.
use super::{ffn, head::TargetHeadOwner, scratch::Scratch, weights::Weights, MiniMaxError};
use crate::shared::{memory::device::Device, qwen::scheduler,
    speculation::dflash::{ContextRow, Draft, DraftSeq}, token_io::TokenEmbedding};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{minimax::{Buffer, Commands}, programs::Scalar, NativeLibrary};
use cuteafd_loader::{families::minimax_m3::dspark::{DraftMemory, DsparkContract, PREFIX, TARGET_LAYERS},
    plan::checkpoint::Checkpoint};
use std::collections::BTreeMap;

const H: usize = 6144;
const KV: usize = 512;
const BLOCK: usize = 7;
const WINDOW: usize = 4096;
const VOCAB: usize = 200064;

#[derive(Debug, Default, Clone)]
pub(super) struct State {
    pub anchor: Option<u32>,
    pub valid_from: usize,
    pub end: usize,
    pub ready: bool,
    pub confidence: Vec<f32>,
}
impl scheduler::DraftState for State {
    fn pending_rows(&self) -> usize { usize::from(self.ready && self.anchor.is_some()) }
    fn close(&mut self, token: u32) { self.anchor = Some(token); }
    fn restored(&mut self, frontier: usize) {
        *self = Self { valid_from: frontier, end: frontier, ..Self::default() };
    }
}
impl State {
    pub fn advance(&mut self, start: usize, rows: usize, after: Option<u32>) -> Result<()> {
        ensure!(rows>0 && (!self.ready || self.end==start), MiniMaxError::Invalid("draft context frontier"));
        let end=start.checked_add(rows).context("MiniMax draft frontier overflow")?;
        if !self.ready { self.valid_from=start; }
        self.end=end; self.ready=true; self.anchor=after; self.confidence.clear(); Ok(())
    }
}

fn context_spans(rows:&[ContextRow],first:usize,slots:usize,capacity:usize)->Result<Vec<(usize,usize,usize)>> {
    let mut spans:Vec<(usize,usize,usize)>=Vec::new();
    for row in rows {
        ensure!(row.slot<slots && row.tap_row>=first && row.tap_row<capacity,
            MiniMaxError::Invalid("draft context span"));
        let source=row.tap_row-first;
        let destination=row.slot*WINDOW+row.position%WINDOW;
        if let Some((src,dst,n))=spans.last_mut() {
            if *src+*n==source && *dst+*n==destination && *dst/WINDOW==destination/WINDOW {
                *n+=1; continue;
            }
        }
        spans.push((source,destination,1));
    }
    Ok(spans)
}

pub(super) struct Layout {
    pub buffers: BTreeMap<&'static str, usize>,
    pub bytes: u64,
    pub sequences: usize,
    pub slots: usize,
    pub rows: usize,
    pub ring_plane_bytes: usize,
}
impl Layout {
    pub fn new(library: &NativeLibrary, contract: &DsparkContract, slots: usize, rows: usize) -> Result<Self> {
        let memory=DraftMemory::new(contract,slots,rows,library.sm_count()?)?;
        ensure!(memory.weights_bytes==Weights::draft_layout(contract)?.resident_bytes
            && memory.buffers["markov_workspace"]==library.glmf_dspark_markov_workspace(memory.sequences,BLOCK)?.max(256)
            && memory.buffers["attention_workspace"]==library.block_dspark_attention_workspace(
                memory.sequences,64,4,BLOCK,WINDOW+BLOCK,128)?.max(256),
            MiniMaxError::Invalid("draft allocation ledger differs from native inventory"));
        Ok(Self {buffers:memory.buffers,bytes:memory.bytes,sequences:memory.sequences,slots,rows,
            ring_plane_bytes:memory.ring_plane_bytes})
    }
}

pub(super) struct Dspark<'a> {
    weights: Weights<'a>,
    buffers: BTreeMap<&'static str, Buffer<'a>>,
    caches: Vec<(Buffer<'a>,Buffer<'a>)>,
    pub layout: Layout,
}
impl<'a> Dspark<'a> {
    pub fn load(device: Device<'a>, commands: &Commands<'a>, contract: &DsparkContract, layout: Layout) -> Result<Self> {
        let checkpoint=Checkpoint::open(&contract.snapshot)?;
        let weights=Weights::load_layout(device,&checkpoint,commands,&Weights::draft_layout(contract)?)?;
        let buffers=layout.buffers.iter().map(|(&name,&bytes)|Ok((name,Buffer::new(device.library,device.id,bytes)?)))
            .collect::<Result<_>>()?;
        let caches=(0..5).map(|_|Ok((Buffer::new(device.library,device.id,layout.ring_plane_bytes)?,
            Buffer::new(device.library,device.id,layout.ring_plane_bytes)?))).collect::<Result<_>>()?;
        let draft=Self {weights,buffers,caches,layout};
        commands.draft_markov_norms(draft.weight("markov_head.markov_w2.weight")?,draft.get("markov_norms")?,VOCAB)?;
        commands.drain()?;
        Ok(draft)
    }
    fn get(&self,name:&str)->Result<&Buffer<'a>> { self.buffers.get(name).context("MiniMax missing draft buffer") }
    fn weight(&self,name:&str)->Result<&Buffer<'a>> { self.weights.get(&format!("{PREFIX}.{name}")) }
    fn layer(&self,layer:usize,name:&str)->Result<&Buffer<'a>> { self.weight(&format!("layers.{layer}.decoder_layer.{name}")) }
    fn linear(&self,c:&Commands<'a>,s:&Scratch<'a>,layer:usize,projection:&str,stem:&str,
        input:&Buffer<'a>,output:&Buffer<'a>,rows:usize)->Result<()> {
        c.launch(&format!("minimax_mxfp8_{projection}"), &[input,self.layer(layer,&format!("{stem}.weight"))?,
            self.layer(layer,&format!("{stem}.weight_scale_inv"))?,s.get("scratch")?,output], &[Scalar::I32(rows as i32),Scalar::I32(0)])
    }
    pub fn tap(&self,c:&Commands<'a>,layer:usize,residual:&Buffer<'a>,rows:usize)->Result<()> {
        let Some(index)=TARGET_LAYERS.iter().position(|&l|l==layer) else {return Ok(())};
        ensure!(rows<=self.layout.rows,MiniMaxError::Invalid("draft tap capacity"));
        c.narrow(residual,self.get("tap")?,rows*H)?;
        c.copy_rows(&self.get("taps")?.view(index*H*2,self.get("taps")?.raw().bytes-index*H*2)?,5*H*2,
            self.get("tap")?,H*2,H*2,rows)
    }
    pub fn update(&self,c:&Commands<'a>,s:&Scratch<'a>,rows:&[ContextRow])->Result<()> {
        if rows.is_empty() {return Ok(())}
        let first=rows.iter().map(|r|r.tap_row).min().unwrap();
        let last=rows.iter().map(|r|r.tap_row).max().unwrap();
        ensure!(last<self.layout.rows && rows.iter().all(|r|r.slot<self.layout.slots),MiniMaxError::Invalid("draft context rows"));
        let n=last-first+1;
        let mut positions=vec![0i64;n];
        for row in rows {positions[row.tap_row-first]=i64::try_from(row.position)?;}
        c.upload(s.get("positions")?,&positions.iter().flat_map(|p|p.to_le_bytes()).collect::<Vec<_>>())?;
        c.bf16_linear(&self.get("taps")?.view(first*5*H*2,n*5*H*2)?,self.weight("fc.weight")?,s.get("delta")?,n,H,5*H)?;
        c.widen(s.get("delta")?,s.get("residual")?,n*H)?;
        ffn::norm(c,s.get("residual")?,self.weight("hidden_norm.weight")?,s.get("norm")?,n)?;
        for (layer,(key,value)) in self.caches.iter().enumerate() {
            self.linear(c,s,layer,"kv","self_attn.kv_proj",s.get("norm")?,s.get("kv")?,n)?;
            c.launch("minimax_kv_split", &[s.get("kv")?,s.get("k")?,s.get("v")?], &[Scalar::I32(n as i32)])?;
            self.rope(c,s,layer,"k",n)?;
            for (source,dest,count) in context_spans(rows,first,self.layout.slots,self.layout.rows)? {
                let bytes=count*KV*2;
                c.copy(&key.view(dest*KV*2,bytes)?,&s.get("k_norm")?.view(source*KV*2,bytes)?,bytes)?;
                c.copy(&value.view(dest*KV*2,bytes)?,&s.get("v")?.view(source*KV*2,bytes)?,bytes)?;
            }
        }
        Ok(())
    }
    fn rope(&self,c:&Commands<'a>,s:&Scratch<'a>,layer:usize,stem:&str,rows:usize)->Result<()> {
        let weight=self.layer(layer,&format!("self_attn.{stem}_norm.weight"))?;
        c.launch(&format!("minimax_{stem}_norm_rope"), &[s.get(stem)?,weight,weight,s.get("cos")?,s.get("sin")?,
            s.get("positions")?,s.get(&format!("{stem}_norm"))?], &[Scalar::I32(rows as i32)])
    }
    pub fn propose(&self,c:&Commands<'a>,s:&Scratch<'a>,embedding:&TokenEmbedding<'a>,head:&TargetHeadOwner<'a>,
        target:&Weights<'a>,seqs:&[DraftSeq],causal:bool)->Result<Vec<Draft>> {
        let count=seqs.len();
        ensure!(count>0 && count<=self.layout.sequences,MiniMaxError::Invalid("draft batch"));
        let rows=count*BLOCK;
        let mut tables=vec![0i32;3*count];
        let mut positions=Vec::new();
        let mut ids=Vec::new();
        let mut anchors=Vec::new();
        for (i,seq) in seqs.iter().enumerate() {
            ensure!(seq.slot<self.layout.slots && seq.valid_from<=seq.position && seq.anchor<VOCAB as u32,
                MiniMaxError::Invalid("draft sequence"));
            tables[i]=seq.slot as i32;
            tables[count+i]=(seq.position-seq.valid_from).min(WINDOW) as i32;
            tables[2*count+i]=i32::try_from(seq.position)?;
            positions.extend((seq.position..seq.position+BLOCK).map(|p|p as i64));
            ids.push(seq.anchor); ids.extend(std::iter::repeat_n(200058,BLOCK-1)); anchors.push(seq.anchor);
        }
        c.retain(&[self.get("ids")?,self.get("input")?])?;
        embedding.embed(&ids,self.get("ids")?.raw(),1,self.get("input")?.raw(),c.raw())?;
        c.upload(s.get("positions")?,&positions.iter().flat_map(|p|p.to_le_bytes()).collect::<Vec<_>>())?;
        c.widen(self.get("input")?,s.get("residual")?,rows*H)?;
        for (layer,(key,value)) in self.caches.iter().enumerate() {
            // Fused Gemma norm sees the unrounded sum; its carried residual is BF16.
            ffn::norm(c,s.get("residual")?,self.layer(layer,"input_layernorm.weight")?,s.get("norm")?,rows)?;
            c.narrow(s.get("residual")?,self.get("input")?,rows*H)?;
            self.linear(c,s,layer,"q","self_attn.q_proj",s.get("norm")?,s.get("q")?,rows)?;
            self.linear(c,s,layer,"kv","self_attn.kv_proj",s.get("norm")?,s.get("kv")?,rows)?;
            c.launch("minimax_kv_split", &[s.get("kv")?,s.get("k")?,s.get("v")?], &[Scalar::I32(rows as i32)])?;
            self.rope(c,s,layer,"q",rows)?; self.rope(c,s,layer,"k",rows)?;
            let max_keys=tables[count..2*count].iter().copied().max().unwrap() as usize+BLOCK;
            c.draft_attention(s.get("q_norm")?,s.get("k_norm")?,s.get("v")?,key,value,
                (self.get("tables")?,&tables),s.get("attention")?,self.get("attention_workspace")?,count,BLOCK,
                self.layout.slots,WINDOW,max_keys,causal)?;
            self.linear(c,s,layer,"o","self_attn.o_proj",s.get("attention")?,s.get("delta")?,rows)?;
            c.widen(self.get("input")?,s.get("residual")?,rows*H)?;
            c.launch("minimax_residual", &[s.get("residual")?,s.get("delta")?,s.get("residual")?], &[Scalar::I32(rows as i32)])?;
            ffn::norm(c,s.get("residual")?,self.layer(layer,"post_attention_layernorm.weight")?,s.get("norm")?,rows)?;
            c.narrow(s.get("residual")?,self.get("input")?,rows*H)?;
            self.linear(c,s,layer,"dense_gate_up","mlp.gate_up_proj",s.get("norm")?,s.get("gate_up")?,rows)?;
            c.launch("minimax_swiglu_i12288", &[s.get("gate_up")?,s.get("activation")?], &[Scalar::I32(rows as i32)])?;
            self.linear(c,s,layer,"dense_down","mlp.down_proj",s.get("activation")?,s.get("delta")?,rows)?;
            c.widen(self.get("input")?,s.get("residual")?,rows*H)?;
            c.launch("minimax_residual", &[s.get("residual")?,s.get("delta")?,s.get("residual")?], &[Scalar::I32(rows as i32)])?;
        }
        ffn::norm(c,s.get("residual")?,self.weight("final_norm.weight")?,s.get("norm")?,rows)?;
        head.project(c,target,s,s.get("norm")?,rows)?;
        c.draft_heads(s.get("logits")?,self.weight("markov_head.markov_w1.weight")?,self.weight("markov_head.markov_w2.weight")?,
            self.get("markov_norms")?,s.get("norm")?,self.weight("confidence_head.proj.weight")?,self.weight("confidence_head.proj.bias")?,
            (self.get("anchors")?,&anchors),self.get("tokens")?,self.get("markov_workspace")?,self.get("confidence")?,count,BLOCK,VOCAB)?;
        let tokens=c.download(self.get("tokens")?,rows*4)?;
        let confidence=c.download(self.get("confidence")?,rows*4)?;
        let tokens=tokens.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        let confidence=confidence.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        ensure!(tokens.iter().all(|&t|t<VOCAB as u32),MiniMaxError::Invalid("draft token outside vocabulary"));
        Ok((0..count).map(|i|Draft { tokens:tokens[i*BLOCK..(i+1)*BLOCK].to_vec(),
            confidence:confidence[i*BLOCK..(i+1)*BLOCK].to_vec(),features:vec![[0.;4];BLOCK] }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn minimax_context_copies_split_at_ring_wraps_and_rejected_rows() -> Result<()> {
        let rows=[ContextRow {tap_row:2,slot:1,position:4095},ContextRow {tap_row:3,slot:1,position:4096},
            ContextRow {tap_row:4,slot:1,position:4097},ContextRow {tap_row:9,slot:0,position:12}];
        assert_eq!(context_spans(&rows,2,2,16)?,[(0,8191,1),(1,4096,2),(7,12,1)]);
        assert!(context_spans(&rows,3,2,16).is_err());
        assert!(context_spans(&rows,2,1,16).is_err());
        assert!(context_spans(&rows,2,2,9).is_err());
        Ok(())
    }
    #[test]
    fn minimax_taps_name_outputs_and_context_frontiers_are_contiguous() -> Result<()> {
        assert_eq!(TARGET_LAYERS,[3,17,31,45,59]);
        assert_eq!(TARGET_LAYERS.map(|layer|layer+1),[4,18,32,46,60]);
        let mut state=State::default();
        state.advance(4090,6,Some(5))?;
        assert!(state.advance(4097,1,Some(6)).is_err());
        state.advance(4096,2,Some(6))?;
        assert_eq!((state.valid_from,state.end),(4090,4098));
        assert!(state.advance(4098,0,Some(7)).is_err());
        Ok(())
    }
}
