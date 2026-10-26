//! MiniMax adapter to shared sliding attention and Markov heads.
//! Semantics: vllm PR 63232 at 306db61dc16a2c7d2655e4d2b29f7d6a6c4409d6.
use super::{ffn, head::TargetHeadOwner, native::Native, scratch::Scratch, weights::Weights, MiniMaxError};
use crate::shared::{memory::{BlockingWait, HostAllocation, device::Allocation}, prefix::view, qwen::scheduler,
    speculation::{block_dspark, dflash::{ContextRow, Draft, DraftSeq}}, token_io::TokenEmbedding};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{block_dspark::{check_operand as check, disjoint, extent, DraftHeads}, programs::Scalar,
    CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
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
    /// DFlash2: the selector features of the last proposal's drafted positions.
    pub features: Vec<[f32; 4]>,
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
        self.end=end; self.ready=true; self.anchor=after; self.confidence.clear(); self.features.clear(); Ok(())
    }
}

/// Copies `(tap row - first, ring row, rows)` of the context `rows` (tap rows
/// below `capacity`), runs that stay contiguous in both merged.
fn context_spans(rows:&[ContextRow],first:usize,slots:usize,capacity:usize)->Result<Vec<(usize,usize,usize)>> {
    let mut spans:Vec<(usize,usize,usize)>=Vec::new();
    for row in rows {
        ensure!(row.tap_row>=first && row.tap_row<capacity, MiniMaxError::Invalid("draft context span"));
        let source=row.tap_row-first;
        let destination=block_dspark::context_row(true,row.slot,row.position,slots,WINDOW)? as usize;
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
    buffers: BTreeMap<&'static str, Allocation<'a>>,
    caches: Vec<(Allocation<'a>,Allocation<'a>)>,
    pub layout: Layout,
    /// Pinned landing of a round's tokens then confidence: both download behind its one host wait.
    landing: HostAllocation<'a>,
    /// The draft round's host wait (`--host-wait`).
    round: std::cell::RefCell<BlockingWait<'a>>,
}
impl<'a> Dspark<'a> {
    pub fn load(native: &Native<'a>, contract: &DsparkContract, layout: Layout) -> Result<Self> {
        let device=native.device();
        let checkpoint=Checkpoint::open(&contract.snapshot)?;
        let weights=Weights::load_layout(native,&checkpoint,&Weights::draft_layout(contract)?)?;
        let buffers=layout.buffers.iter().map(|(&name,&bytes)|Ok((name,Allocation::new(device,bytes)?)))
            .collect::<Result<_>>()?;
        let caches=(0..5).map(|_|Ok((Allocation::new(device,layout.ring_plane_bytes)?,
            Allocation::new(device,layout.ring_plane_bytes)?))).collect::<Result<_>>()?;
        // Pinned memory is allocated before readiness: its allocation can synchronize the device.
        let landing=HostAllocation::named(native.library(),landing_bytes(layout.sequences)?,"minimax-dspark-landing")?;
        let round=std::cell::RefCell::new(BlockingWait::new(native.library()));
        let draft=Self {weights,buffers,caches,layout,landing,round};
        // The draft metadata's ordered uploads (positions is the target's step scratch, reserved
        // with it) get their whole-extent staging now.
        for name in DRAFT_UPLOADS { native.reserve(draft.get(name)?)?; }
        let (w2,norms)=(draft.weight("markov_head.markov_w2.weight")?,draft.get("markov_norms")?);
        native.check_device(&[w2,norms])?;
        // SAFETY: the drafter owns both buffers; the drain below completes the kernel.
        unsafe { native.library().dspark_markov_norms(w2,norms,VOCAB,native.raw())? };
        native.drain()?;
        Ok(draft)
    }
    fn get(&self,name:&str)->Result<CuteafdDeviceBuffer> { self.buffers.get(name).map(|a|a.buffer).context("MiniMax missing draft buffer") }
    fn weight(&self,name:&str)->Result<CuteafdDeviceBuffer> { self.weights.get(&format!("{PREFIX}.{name}")) }
    fn layer(&self,layer:usize,name:&str)->Result<CuteafdDeviceBuffer> { self.weight(&format!("layers.{layer}.decoder_layer.{name}")) }
    #[allow(clippy::too_many_arguments)]
    fn linear(&self,n:&Native<'a>,s:&Scratch<'a>,layer:usize,projection:&str,stem:&str,
        input:CuteafdDeviceBuffer,output:CuteafdDeviceBuffer,rows:usize)->Result<()> {
        ffn::project(n,projection,self.layer(layer,&format!("{stem}.weight"))?,
            self.layer(layer,&format!("{stem}.weight_scale_inv"))?,s.get("scratch")?,input,output,rows)
    }
    /// Taps `rows` residual rows (batch rows from `first`) at a target layer.
    pub fn tap(&self,n:&Native<'a>,layer:usize,residual:CuteafdDeviceBuffer,first:usize,rows:usize)->Result<()> {
        let Some(index)=TARGET_LAYERS.iter().position(|&l|l==layer) else {return Ok(())};
        ensure!(first+rows<=self.layout.rows,MiniMaxError::Invalid("draft tap capacity"));
        narrow(n,residual,self.get("tap")?,rows*H)?;
        let at=(first*5+index)*H*2;
        copy_rows(n,view(self.get("taps")?,at,self.get("taps")?.bytes-at)?,5*H*2,
            self.get("tap")?,H*2,H*2,rows)
    }
    pub fn update(&self,n:&Native<'a>,s:&Scratch<'a>,rows:&[ContextRow])->Result<()> {
        if rows.is_empty() {return Ok(())}
        // A program launch takes at most `capacity` rows: a prefill chunk of
        // several lanes updates in pieces, in row order (each row is its own).
        let capacity=n.capacity("minimax_kv_split")?;
        let first=rows.iter().map(|r|r.tap_row).min().unwrap();
        let last=rows.iter().map(|r|r.tap_row).max().unwrap();
        for start in (first..=last).step_by(capacity) {
            let piece=rows.iter().filter(|r|(start..start+capacity).contains(&r.tap_row)).copied().collect::<Vec<_>>();
            if !piece.is_empty() { self.update_rows(n,s,&piece)?; }
        }
        Ok(())
    }
    fn update_rows(&self,n:&Native<'a>,s:&Scratch<'a>,rows:&[ContextRow])->Result<()> {
        let first=rows.iter().map(|r|r.tap_row).min().unwrap();
        let last=rows.iter().map(|r|r.tap_row).max().unwrap();
        ensure!(last<self.layout.rows && rows.iter().all(|r|r.slot<self.layout.slots),MiniMaxError::Invalid("draft context rows"));
        let count=last-first+1;
        let mut positions=vec![0i64;count];
        for row in rows {positions[row.tap_row-first]=i64::try_from(row.position)?;}
        n.upload(s.get("positions")?,&positions.iter().flat_map(|p|p.to_le_bytes()).collect::<Vec<_>>())?;
        bf16_linear(n,view(self.get("taps")?,first*5*H*2,count*5*H*2)?,self.weight("fc.weight")?,s.get("delta")?,count,H,5*H)?;
        n.widen(s.get("delta")?,s.get("residual")?,count*H)?;
        ffn::norm(n,s.get("residual")?,self.weight("hidden_norm.weight")?,s.get("norm")?,count)?;
        for (layer,(key,value)) in self.caches.iter().enumerate() {
            self.linear(n,s,layer,"kv","self_attn.kv_proj",s.get("norm")?,s.get("kv")?,count)?;
            n.launch("minimax_kv_split", &[s.get("kv")?,s.get("k")?,s.get("v")?], &[Scalar::I32(count as i32)])?;
            self.rope(n,s,layer,"k",count)?;
            for (source,dest,span) in context_spans(rows,first,self.layout.slots,self.layout.rows)? {
                let bytes=span*KV*2;
                n.copy(view(key.buffer,dest*KV*2,bytes)?,view(s.get("k_norm")?,source*KV*2,bytes)?,bytes)?;
                n.copy(view(value.buffer,dest*KV*2,bytes)?,view(s.get("v")?,source*KV*2,bytes)?,bytes)?;
            }
        }
        Ok(())
    }
    fn rope(&self,n:&Native<'a>,s:&Scratch<'a>,layer:usize,stem:&str,rows:usize)->Result<()> {
        let weight=self.layer(layer,&format!("self_attn.{stem}_norm.weight"))?;
        n.launch(&format!("minimax_{stem}_norm_rope"), &[s.get(stem)?,weight,weight,s.get("cos")?,s.get("sin")?,
            s.get("positions")?,s.get(&format!("{stem}_norm"))?], &[Scalar::I32(rows as i32)])
    }
    #[allow(clippy::too_many_arguments)]
    pub fn propose(&self,n:&Native<'a>,s:&Scratch<'a>,embedding:&TokenEmbedding<'a>,head:&TargetHeadOwner<'a>,
        target:&Weights<'a>,seqs:&[DraftSeq])->Result<Vec<Draft>> {
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
        n.check_device(&[self.get("ids")?,self.get("input")?])?;
        embedding.embed(&ids,self.get("ids")?,1,self.get("input")?,n.raw())?;
        n.upload(s.get("positions")?,&positions.iter().flat_map(|p|p.to_le_bytes()).collect::<Vec<_>>())?;
        n.widen(self.get("input")?,s.get("residual")?,rows*H)?;
        for (layer,(key,value)) in self.caches.iter().enumerate() {
            // Fused Gemma norm sees the unrounded sum; its carried residual is BF16.
            ffn::norm(n,s.get("residual")?,self.layer(layer,"input_layernorm.weight")?,s.get("norm")?,rows)?;
            narrow(n,s.get("residual")?,self.get("input")?,rows*H)?;
            self.linear(n,s,layer,"q","self_attn.q_proj",s.get("norm")?,s.get("q")?,rows)?;
            self.linear(n,s,layer,"kv","self_attn.kv_proj",s.get("norm")?,s.get("kv")?,rows)?;
            n.launch("minimax_kv_split", &[s.get("kv")?,s.get("k")?,s.get("v")?], &[Scalar::I32(rows as i32)])?;
            self.rope(n,s,layer,"q",rows)?; self.rope(n,s,layer,"k",rows)?;
            let max_keys=tables[count..2*count].iter().copied().max().unwrap() as usize+BLOCK;
            draft_attention(n,s.get("q_norm")?,s.get("k_norm")?,s.get("v")?,key.buffer,value.buffer,
                (self.get("tables")?,&tables),s.get("attention")?,self.get("attention_workspace")?,count,BLOCK,
                self.layout.slots,WINDOW,max_keys)?;
            self.linear(n,s,layer,"o","self_attn.o_proj",s.get("attention")?,s.get("delta")?,rows)?;
            n.widen(self.get("input")?,s.get("residual")?,rows*H)?;
            n.launch("minimax_residual", &[s.get("residual")?,s.get("delta")?,s.get("residual")?], &[Scalar::I32(rows as i32)])?;
            ffn::norm(n,s.get("residual")?,self.layer(layer,"post_attention_layernorm.weight")?,s.get("norm")?,rows)?;
            narrow(n,s.get("residual")?,self.get("input")?,rows*H)?;
            self.linear(n,s,layer,"dense_gate_up","mlp.gate_up_proj",s.get("norm")?,s.get("gate_up")?,rows)?;
            n.launch("minimax_swiglu_i12288", &[s.get("gate_up")?,s.get("activation")?], &[Scalar::I32(rows as i32)])?;
            self.linear(n,s,layer,"dense_down","mlp.down_proj",s.get("activation")?,s.get("delta")?,rows)?;
            n.widen(self.get("input")?,s.get("residual")?,rows*H)?;
            n.launch("minimax_residual", &[s.get("residual")?,s.get("delta")?,s.get("residual")?], &[Scalar::I32(rows as i32)])?;
        }
        ffn::norm(n,s.get("residual")?,self.weight("final_norm.weight")?,s.get("norm")?,rows)?;
        head.project(n,target,s,s.get("norm")?,rows)?;
        let heads=DraftHeads { w1:self.weight("markov_head.markov_w1.weight")?, w2:self.weight("markov_head.markov_w2.weight")?,
            norms:self.get("markov_norms")?, weight:self.weight("confidence_head.proj.weight")?,
            bias:self.weight("confidence_head.proj.bias")? };
        draft_heads(n,&heads,s.get("logits")?,s.get("norm")?,(self.get("anchors")?,&anchors),self.get("tokens")?,
            self.get("markov_workspace")?,self.get("confidence")?,count)?;
        // The round's results download on its stream; one host wait then covers the round and
        // both copies.
        let (tokens,confidence)=(self.get("tokens")?,self.get("confidence")?);
        let [token_landing,confidence_landing]=landing_halves(self.landing.buffer,rows)?;
        check(tokens,rows*4,4)?; check(confidence,rows*4,4)?;
        n.check_device(&[tokens,confidence])?;
        // SAFETY: the device buffers hold `rows` words and the drafter's pinned landing both
        // halves; the landing is read only after the wait, which covers both copies.
        unsafe {
            n.library().copy_d2h_host_buffer_async(token_landing,tokens,rows*4,n.raw())?;
            n.library().copy_d2h_host_buffer_async(confidence_landing,confidence,rows*4,n.raw())?;
            self.round.borrow_mut().wait(n.raw())?;
        }
        let (tokens,confidence)=self.landing.bytes()[..rows*8].split_at(rows*4);
        let tokens=tokens.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        let confidence=confidence.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        ensure!(tokens.iter().all(|&t|t<VOCAB as u32),MiniMaxError::Invalid("draft token outside vocabulary"));
        Ok((0..count).map(|i|Draft { tokens:tokens[i*BLOCK..(i+1)*BLOCK].to_vec(),
            confidence:confidence[i*BLOCK..(i+1)*BLOCK].to_vec(),features:vec![[0.;4];BLOCK] }).collect())
    }
}

const HEADS: usize = 64;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;

fn scalar(value: usize) -> Result<()> {
    ensure!(value > 0 && value <= i32::MAX as usize,
        MiniMaxError::Invalid("draft scalar is outside the native range"));
    Ok(())
}

fn pitch_span(pitch: usize, width: usize, rows: usize) -> Result<usize> {
    ensure!(width <= pitch, MiniMaxError::Invalid("draft copy width exceeds pitch"));
    if rows == 0 || width == 0 { return Ok(0); }
    (rows-1).checked_mul(pitch).and_then(|n|n.checked_add(width))
        .ok_or_else(||MiniMaxError::Invalid("draft copy span overflow").into())
}

fn draft_rows(sequences: usize, block: usize, vocab: usize) -> Result<usize> {
    ensure!((1..=32).contains(&sequences) && (1..=7).contains(&block),
        MiniMaxError::Invalid("draft batch or block exceeds native limits"));
    scalar(vocab)?;
    ensure!(vocab <= (i32::MAX as usize-255)/32,
        MiniMaxError::Invalid("draft vocabulary exceeds native norm limits"));
    extent(&[sequences,block])
}

fn attention_tables(tables: &[i32], sequences: usize, block: usize, slots: usize,
    capacity: usize, max_keys: usize) -> Result<()> {
    draft_rows(sequences,block,1)?; scalar(slots)?;
    ensure!((1..=WINDOW).contains(&capacity) && max_keys >= block && max_keys <= capacity+block,
        MiniMaxError::Invalid("draft attention context extent"));
    ensure!(tables.len() == sequences*3, MiniMaxError::Invalid("draft attention table shape"));
    for s in 0..sequences {
        let (slot,length,end)=(tables[s],tables[sequences+s],tables[2*sequences+s]);
        ensure!(slot >= 0 && (slot as usize) < slots && length >= 0 && (length as usize) <= capacity
            && end >= length && end <= i32::MAX-block as i32 && length as usize+block <= max_keys,
            MiniMaxError::Invalid("draft attention table value"));
    }
    Ok(())
}

fn anchor_ids(anchors: &[u32], sequences: usize, vocab: usize) -> Result<()> {
    ensure!(anchors.len() == sequences && anchors.iter().all(|&token|(token as usize)<vocab),
        MiniMaxError::Invalid("draft anchor token is outside the vocabulary"));
    Ok(())
}

/// `destination[..count]` (BF16) = `source[..count]` (F32).
fn narrow(n: &Native<'_>, source: CuteafdDeviceBuffer, destination: CuteafdDeviceBuffer, count: usize) -> Result<()> {
    let (source_bytes,destination_bytes)=(extent(&[count,4])?,extent(&[count,2])?);
    check(source,source_bytes,4)?; check(destination,destination_bytes,2)?;
    disjoint(source,source_bytes,destination,destination_bytes)?;
    n.check_device(&[source,destination])?;
    if count == 0 { return Ok(()); }
    // SAFETY: checked distinct spans have the native alignment and live owners.
    unsafe { n.library().cuda_f32_to_bf16_async(source,destination,count,n.raw()) }
}

fn copy_rows(n: &Native<'_>, destination: CuteafdDeviceBuffer, destination_pitch: usize, source: CuteafdDeviceBuffer,
    source_pitch: usize, width: usize, rows: usize) -> Result<()> {
    let destination_bytes=pitch_span(destination_pitch,width,rows)?;
    let source_bytes=pitch_span(source_pitch,width,rows)?;
    disjoint(destination,destination_bytes,source,source_bytes)?;
    n.check_device(&[destination,source])?;
    if destination_bytes == 0 { return Ok(()); }
    // SAFETY: both pitched spans are checked, disjoint and owned through the ordered copy.
    unsafe { n.library().copy_d2d_2d_async(destination,destination_pitch,source,source_pitch,width,rows,n.raw()) }
}

fn bf16_linear(n: &Native<'_>, input: CuteafdDeviceBuffer, weight: CuteafdDeviceBuffer, output: CuteafdDeviceBuffer,
    rows: usize, outputs: usize, k: usize) -> Result<()> {
    for value in [rows,outputs,k] { scalar(value)?; }
    let sizes=[extent(&[rows,k,2])?,extent(&[outputs,k,2])?,extent(&[rows,outputs,2])?];
    for (buffer,bytes) in [input,weight,output].into_iter().zip(sizes) { check(buffer,bytes,16)?; }
    disjoint(input,sizes[0],output,sizes[2])?;
    disjoint(weight,sizes[1],output,sizes[2])?;
    n.check_device(&[input,weight,output])?;
    // SAFETY: cuBLAS dimensions and aligned, disjoint output extents are checked; owners are live.
    unsafe { n.library().linear_bf16(input.ptr,weight.ptr,output.ptr,rows,k,outputs,n.raw()) }
}

/// Tables are three contiguous I32 arrays: slot, context length, exclusive context end.
/// The release uses a per-query sliding window and a noncausal draft block.
#[allow(clippy::too_many_arguments)]
fn draft_attention(n: &Native<'_>, q: CuteafdDeviceBuffer, k: CuteafdDeviceBuffer, v: CuteafdDeviceBuffer,
    context_k: CuteafdDeviceBuffer, context_v: CuteafdDeviceBuffer, tables: (CuteafdDeviceBuffer, &[i32]),
    output: CuteafdDeviceBuffer, workspace: CuteafdDeviceBuffer, sequences: usize, block: usize, slots: usize,
    capacity: usize, max_keys: usize) -> Result<()> {
    attention_tables(tables.1,sequences,block,slots,capacity,max_keys)?;
    let rows=extent(&[sequences,block])?;
    let q_bytes=extent(&[rows,HEADS,HEAD_DIM,2])?;
    let kv_bytes=extent(&[rows,KV_HEADS,HEAD_DIM,2])?;
    let context_bytes=extent(&[slots,capacity,KV_HEADS,HEAD_DIM,2])?;
    let library=n.library();
    let workspace_bytes=library.block_dspark_attention_workspace(sequences,HEADS,KV_HEADS,block,max_keys,HEAD_DIM)?;
    let operands=[q,k,v,context_k,context_v,tables.0,output,workspace];
    let sizes=[q_bytes,kv_bytes,kv_bytes,context_bytes,context_bytes,sequences*12,q_bytes,workspace_bytes];
    for (i,buffer) in operands.iter().enumerate() { check(*buffer,sizes[i],if i==5 {4} else {16})?; }
    for out in [5,6,7] { for input in 0..out {
        disjoint(operands[out],sizes[out],operands[input],sizes[input])?;
    } }
    n.check_device(&operands)?;
    n.upload(tables.0,&tables.1.iter().flat_map(|value|value.to_ne_bytes()).collect::<Vec<_>>())?;
    let table=tables.0.ptr.cast::<i32>();
    // SAFETY: uploaded table values bound all ring reads; all aligned buffers and scratch are live.
    unsafe { library.mimo_dflash_attention(q.ptr,
        k.ptr,v.ptr,context_k.ptr,context_v.ptr,table.cast(),
        table.add(sequences).cast(),table.add(2*sequences).cast(),std::ptr::null(),output.ptr,workspace.ptr,
        sequences,block,HEADS,KV_HEADS,capacity,max_keys,WINDOW,(HEAD_DIM as f32).sqrt().recip(),false,n.raw()) }
}

/// The shared draft heads over the block's `sequences * 7` rows, from each
/// sequence's (checked, then uploaded) anchor.
#[allow(clippy::too_many_arguments)]
fn draft_heads(n: &Native<'_>, heads: &DraftHeads, logits: CuteafdDeviceBuffer, hidden: CuteafdDeviceBuffer,
    anchors: (CuteafdDeviceBuffer, &[u32]), tokens: CuteafdDeviceBuffer, workspace: CuteafdDeviceBuffer,
    confidence: CuteafdDeviceBuffer, sequences: usize) -> Result<()> {
    draft_rows(sequences,BLOCK,VOCAB)?;
    anchor_ids(anchors.1,sequences,VOCAB)?;
    n.check_device(&[logits,heads.w1,heads.w2,heads.norms,hidden,heads.weight,heads.bias,anchors.0,tokens,workspace,confidence])?;
    n.upload(anchors.0,&anchors.1.iter().flat_map(|value|value.to_ne_bytes()).collect::<Vec<_>>())?;
    // SAFETY: the drafter and scratch own every operand through the readback's
    // drain; the anchors were checked against the vocabulary.
    unsafe { n.library().dspark_draft_heads(heads,logits,hidden,anchors.0,tokens,workspace,confidence,
        sequences,BLOCK,VOCAB,H,n.raw()) }
}

/// The draft buffers a round uploads into (besides the target's `positions`).
const DRAFT_UPLOADS: [&str; 2] = ["tables","anchors"];

/// Pinned landing bytes for `sequences` draft blocks: tokens (u32) then confidence (f32).
fn landing_bytes(sequences: usize) -> Result<usize> { extent(&[draft_rows(sequences,BLOCK,VOCAB)?,8]) }

/// A round of `rows` draft rows lands its tokens in the landing's first `rows` words and its
/// confidence in the next `rows`.
fn landing_halves(landing: CuteafdHostBuffer, rows: usize) -> Result<[CuteafdHostBuffer; 2]> {
    let half=extent(&[rows,4])?;
    ensure!(rows>0 && !landing.ptr.is_null() && landing.bytes>=2*half, MiniMaxError::Invalid("draft landing extent"));
    Ok([0,half].map(|offset|CuteafdHostBuffer { ptr:landing.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes:half, ..landing }))
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

    #[test] fn draft_pitched_copy_checks_the_last_row() -> Result<()> {
        assert_eq!(pitch_span(64,16,3)?,144);
        assert_eq!(pitch_span(64,0,usize::MAX)?,0);
        assert_eq!(pitch_span(64,16,0)?,0);
        assert!(pitch_span(16,17,1).is_err());
        assert!(pitch_span(usize::MAX,1,3).is_err());
        assert!(pitch_span(usize::MAX,usize::MAX,2).is_err());
        Ok(())
    }

    #[test] fn draft_metadata_bounds_every_ring_read() -> Result<()> {
        attention_tables(&[0,2,0,4096,0,8192],2,7,3,4096,4103)?;
        for tables in [[3,1,1],[0,-1,1],[0,2,1],[0,4097,8192],[0,1,i32::MAX]] {
            assert!(attention_tables(&tables,1,7,3,4096,4103).is_err());
        }
        assert!(attention_tables(&[0,4096,8192],1,7,1,4096,4102).is_err());
        assert!(attention_tables(&[0,1,1],1,7,1,4097,4104).is_err());
        assert!(attention_tables(&[0,1],1,7,1,4096,4103).is_err());
        Ok(())
    }

    /// A round's tokens and confidence land in the two halves of one pinned landing, sized at
    /// load for the widest draft batch; the uploads it queues go to reserved draft buffers.
    #[test] fn minimax_draft_round_lands_in_pinned_halves() -> Result<()> {
        assert_eq!(landing_bytes(32)?,32*BLOCK*8);
        assert!(landing_bytes(33).is_err());
        let landing=CuteafdHostBuffer{ptr:0x4000usize as *mut std::ffi::c_void,bytes:landing_bytes(32)?,flags:3};
        let [tokens,confidence]=landing_halves(landing,2*BLOCK)?;
        assert_eq!((tokens.ptr as usize,tokens.bytes,tokens.flags),(0x4000,56,3));
        assert_eq!((confidence.ptr as usize,confidence.bytes),(0x4000+56,56));
        assert!(landing_halves(landing,32*BLOCK+1).is_err() && landing_halves(landing,0).is_err());
        let source=include_str!("dspark.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("for name in DRAFT_UPLOADS { native.reserve(draft.get(name)?)?; }"));
        for upload in ["n.upload(tables.0,","n.upload(anchors.0,"] { assert!(source.contains(upload)); }
        assert!(source.contains("(self.get(\"tables\")?,&tables)") && source.contains("(self.get(\"anchors\")?,&anchors)"));
        assert!(!source.contains(&["copy_", "d2h(&mut"].concat()),"no synchronous readback");
        Ok(())
    }

    #[test] fn draft_batch_and_anchor_limits_match_native_heads() -> Result<()> {
        assert_eq!(draft_rows(32,7,200064)?,224);
        for shape in [(0,7,200064),(33,7,200064),(1,8,200064),(1,0,200064),(1,7,0),
            (1,7,i32::MAX as usize)] { assert!(draft_rows(shape.0,shape.1,shape.2).is_err()); }
        anchor_ids(&[0,200063],2,200064)?;
        assert!(anchor_ids(&[200064],1,200064).is_err());
        assert!(anchor_ids(&[0],2,200064).is_err());
        Ok(())
    }
}
