use super::{on_device, Buffer, Commands, MiniMaxNativeError};
use crate::CuteafdDeviceBuffer;
use anyhow::{ensure, Result};

const HIDDEN: usize = 6144;
const RANK: usize = 256;
const HEADS: usize = 64;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const WINDOW: usize = 4096;

fn extent(factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |n, &factor| n.checked_mul(factor)
        .ok_or_else(|| MiniMaxNativeError::Operand("draft extent overflow").into()))
}

fn scalar(value: usize) -> Result<()> {
    ensure!(value > 0 && value <= i32::MAX as usize,
        MiniMaxNativeError::Operand("draft scalar is outside the native range"));
    Ok(())
}

fn check(buffer: CuteafdDeviceBuffer, bytes: usize, alignment: usize) -> Result<()> {
    ensure!(bytes <= buffer.bytes && (bytes == 0 || !buffer.ptr.is_null()),
        MiniMaxNativeError::Operand("draft buffer extent"));
    ensure!(buffer.ptr as usize % alignment == 0,
        MiniMaxNativeError::Operand("draft buffer alignment"));
    ensure!((buffer.ptr as usize).checked_add(bytes).is_some(),
        MiniMaxNativeError::Operand("draft pointer extent overflow"));
    Ok(())
}

fn disjoint(a: CuteafdDeviceBuffer, a_bytes: usize, b: CuteafdDeviceBuffer, b_bytes: usize) -> Result<()> {
    check(a,a_bytes,1)?; check(b,b_bytes,1)?;
    let (a_start,b_start)=(a.ptr as usize,b.ptr as usize);
    ensure!(a_bytes == 0 || b_bytes == 0 || a_start+a_bytes <= b_start || b_start+b_bytes <= a_start,
        MiniMaxNativeError::Operand("draft buffers overlap"));
    Ok(())
}

fn pitch_span(pitch: usize, width: usize, rows: usize) -> Result<usize> {
    ensure!(width <= pitch, MiniMaxNativeError::Operand("draft copy width exceeds pitch"));
    if rows == 0 || width == 0 { return Ok(0); }
    (rows-1).checked_mul(pitch).and_then(|n|n.checked_add(width))
        .ok_or_else(||MiniMaxNativeError::Operand("draft copy span overflow").into())
}

fn draft_rows(sequences: usize, block: usize, vocab: usize) -> Result<usize> {
    ensure!((1..=32).contains(&sequences) && (1..=7).contains(&block),
        MiniMaxNativeError::Operand("draft batch or block exceeds native limits"));
    scalar(vocab)?;
    ensure!(vocab <= (i32::MAX as usize-255)/32,
        MiniMaxNativeError::Operand("draft vocabulary exceeds native norm limits"));
    extent(&[sequences,block])
}

fn attention_tables(tables: &[i32], sequences: usize, block: usize, slots: usize,
    capacity: usize, max_keys: usize) -> Result<()> {
    draft_rows(sequences,block,1)?; scalar(slots)?;
    ensure!((1..=WINDOW).contains(&capacity) && max_keys >= block && max_keys <= capacity+block,
        MiniMaxNativeError::Operand("draft attention context extent"));
    ensure!(tables.len() == sequences*3, MiniMaxNativeError::Operand("draft attention table shape"));
    for s in 0..sequences {
        let (slot,length,end)=(tables[s],tables[sequences+s],tables[2*sequences+s]);
        ensure!(slot >= 0 && (slot as usize) < slots && length >= 0 && (length as usize) <= capacity
            && end >= length && end <= i32::MAX-block as i32 && length as usize+block <= max_keys,
            MiniMaxNativeError::Operand("draft attention table value"));
    }
    Ok(())
}

fn anchor_ids(anchors: &[u32], sequences: usize, vocab: usize) -> Result<()> {
    ensure!(anchors.len() == sequences && anchors.iter().all(|&token|(token as usize)<vocab),
        MiniMaxNativeError::Operand("draft anchor token is outside the vocabulary"));
    Ok(())
}

impl<'a> Commands<'a> {
    pub fn narrow(&self, source: &Buffer<'a>, destination: &Buffer<'a>, count: usize) -> Result<()> {
        let (source_bytes,destination_bytes)=(extent(&[count,4])?,extent(&[count,2])?);
        check(source.raw(),source_bytes,4)?; check(destination.raw(),destination_bytes,2)?;
        disjoint(source.raw(),source_bytes,destination.raw(),destination_bytes)?;
        self.keep(&[source,destination])?;
        if count == 0 { return Ok(()); }
        // SAFETY: checked distinct spans have the native alignment and remain retained until drain.
        on_device(self.library,self.device,||unsafe { self.library.cuda_f32_to_bf16_async(
            source.raw(),destination.raw(),count,self.stream) })
    }

    pub fn copy_rows(&self, destination: &Buffer<'a>, destination_pitch: usize, source: &Buffer<'a>,
        source_pitch: usize, width: usize, rows: usize) -> Result<()> {
        let destination_bytes=pitch_span(destination_pitch,width,rows)?;
        let source_bytes=pitch_span(source_pitch,width,rows)?;
        disjoint(destination.raw(),destination_bytes,source.raw(),source_bytes)?;
        self.keep(&[destination,source])?;
        if destination_bytes == 0 { return Ok(()); }
        // SAFETY: both pitched spans are checked, disjoint and retained through the ordered copy.
        on_device(self.library,self.device,||unsafe { self.library.copy_d2d_2d_async(destination.raw(),
            destination_pitch,source.raw(),source_pitch,width,rows,self.stream) })
    }

    pub fn bf16_linear(&self, input: &Buffer<'a>, weight: &Buffer<'a>, output: &Buffer<'a>,
        rows: usize, n: usize, k: usize) -> Result<()> {
        for value in [rows,n,k] { scalar(value)?; }
        let sizes=[extent(&[rows,k,2])?,extent(&[n,k,2])?,extent(&[rows,n,2])?];
        for (buffer,bytes) in [input,weight,output].into_iter().zip(sizes) { check(buffer.raw(),bytes,16)?; }
        disjoint(input.raw(),sizes[0],output.raw(),sizes[2])?;
        disjoint(weight.raw(),sizes[1],output.raw(),sizes[2])?;
        self.keep(&[input,weight,output])?;
        // SAFETY: cuBLAS dimensions and aligned, disjoint output extents are checked; owners are retained.
        on_device(self.library,self.device,||unsafe { self.library.linear_bf16(input.raw().ptr,
            weight.raw().ptr,output.raw().ptr,rows,k,n,self.stream) })
    }

    /// Tables are three contiguous I32 arrays: slot, context length, exclusive context end.
    /// The release uses a per-query sliding window and a noncausal draft block.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_attention(&self, q: &Buffer<'a>, k: &Buffer<'a>, v: &Buffer<'a>, context_k: &Buffer<'a>,
        context_v: &Buffer<'a>, tables: (&Buffer<'a>, &[i32]), output: &Buffer<'a>, workspace: &Buffer<'a>,
        sequences: usize, block: usize, slots: usize, capacity: usize, max_keys: usize, causal: bool) -> Result<()> {
        ensure!(!causal, MiniMaxNativeError::Operand("MiniMax draft attention is noncausal"));
        attention_tables(tables.1,sequences,block,slots,capacity,max_keys)?;
        let rows=extent(&[sequences,block])?;
        let q_bytes=extent(&[rows,HEADS,HEAD_DIM,2])?;
        let kv_bytes=extent(&[rows,KV_HEADS,HEAD_DIM,2])?;
        let context_bytes=extent(&[slots,capacity,KV_HEADS,HEAD_DIM,2])?;
        let workspace_bytes=on_device(self.library,self.device,||self.library.block_dspark_attention_workspace(
            sequences,HEADS,KV_HEADS,block,max_keys,HEAD_DIM))?;
        let operands=[q,k,v,context_k,context_v,tables.0,output,workspace];
        let sizes=[q_bytes,kv_bytes,kv_bytes,context_bytes,context_bytes,sequences*12,q_bytes,workspace_bytes];
        for (i,buffer) in operands.iter().enumerate() { check(buffer.raw(),sizes[i],if i==5 {4} else {16})?; }
        for out in [5,6,7] { for input in 0..out {
            disjoint(operands[out].raw(),sizes[out],operands[input].raw(),sizes[input])?;
        } }
        self.keep(&operands)?;
        self.upload(tables.0,&tables.1.iter().flat_map(|value|value.to_ne_bytes()).collect::<Vec<_>>())?;
        let table=tables.0.raw().ptr.cast::<i32>();
        // SAFETY: uploaded table values bound all ring reads; all aligned buffers and scratch are retained.
        on_device(self.library,self.device,||unsafe { self.library.mimo_dflash_attention(q.raw().ptr,
            k.raw().ptr,v.raw().ptr,context_k.raw().ptr,context_v.raw().ptr,table.cast(),
            table.add(sequences).cast(),table.add(2*sequences).cast(),std::ptr::null(),output.raw().ptr,workspace.raw().ptr,
            sequences,block,HEADS,KV_HEADS,capacity,max_keys,WINDOW,(HEAD_DIM as f32).sqrt().recip(),false,self.stream) })
    }

    pub fn draft_markov_norms(&self, w2: &Buffer<'a>, norms: &Buffer<'a>, vocab: usize) -> Result<()> {
        draft_rows(1,1,vocab)?;
        let (weight_bytes,norm_bytes)=(extent(&[vocab,RANK,2])?,extent(&[vocab,4])?);
        check(w2.raw(),weight_bytes,16)?; check(norms.raw(),norm_bytes,4)?;
        disjoint(w2.raw(),weight_bytes,norms.raw(),norm_bytes)?;
        self.keep(&[w2,norms])?;
        // SAFETY: the fixed rank, native grid bound and distinct retained extents are checked.
        on_device(self.library,self.device,||unsafe { self.library.glmf_dspark_markov_norms(w2.raw().ptr,
            norms.raw().ptr,vocab,RANK,self.stream) })
    }

    /// Run both heads together so confidence consumes only the bounded Markov result.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_heads(&self, logits: &Buffer<'a>, w1: &Buffer<'a>, w2: &Buffer<'a>, norms: &Buffer<'a>,
        hidden: &Buffer<'a>, weight: &Buffer<'a>, bias: &Buffer<'a>, anchors: (&Buffer<'a>, &[u32]),
        tokens: &Buffer<'a>, workspace: &Buffer<'a>, output: &Buffer<'a>, sequences: usize,
        block: usize, vocab: usize) -> Result<()> {
        let rows=draft_rows(sequences,block,vocab)?;
        anchor_ids(anchors.1,sequences,vocab)?;
        let workspace_bytes=self.library.glmf_dspark_markov_workspace(sequences,block)?;
        let operands=[logits,w1,w2,norms,hidden,weight,bias,anchors.0,tokens,workspace,output];
        let sizes=[extent(&[rows,vocab,4])?,extent(&[vocab,RANK,2])?,extent(&[vocab,RANK,2])?,
            extent(&[vocab,4])?,extent(&[rows,HIDDEN,2])?,(HIDDEN+RANK)*2,2,sequences*4,rows*4,
            workspace_bytes,rows*4];
        for (i,buffer) in operands.iter().enumerate() {
            let alignment=match i {1|2=>16,4..=6=>2,_=>4};
            check(buffer.raw(),sizes[i],alignment)?;
        }
        for out in 7..operands.len() { for input in 0..out {
            disjoint(operands[out].raw(),sizes[out],operands[input].raw(),sizes[input])?;
        } }
        self.keep(&operands)?;
        self.upload(anchors.0,&anchors.1.iter().flat_map(|value|value.to_ne_bytes()).collect::<Vec<_>>())?;
        self.draft_markov(logits,w1,w2,norms,anchors,tokens,workspace,sequences,block,vocab)?;
        self.draft_confidence(hidden,w1,weight,bias,anchors,tokens,output,sequences,block)
    }

    // Invalid proposals fall back to a bounded token before the confidence launch.
    #[allow(clippy::too_many_arguments)]
    fn draft_markov(&self, logits: &Buffer<'a>, w1: &Buffer<'a>, w2: &Buffer<'a>, norms: &Buffer<'a>,
        anchors: (&Buffer<'a>, &[u32]), tokens: &Buffer<'a>, workspace: &Buffer<'a>, sequences: usize,
        block: usize, vocab: usize) -> Result<()> {
        // SAFETY: checked anchors and native token bounds guard every indexed read; all buffers are retained.
        on_device(self.library,self.device,||unsafe { self.library.glmf_dspark_markov(logits.raw().ptr,
            w1.raw().ptr,w2.raw().ptr,norms.raw().ptr,anchors.0.raw().ptr,tokens.raw().ptr,workspace.raw().ptr,
            sequences,block,vocab,RANK,self.stream) })
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_confidence(&self, hidden: &Buffer<'a>, w1: &Buffer<'a>, weight: &Buffer<'a>, bias: &Buffer<'a>,
        anchors: (&Buffer<'a>, &[u32]), tokens: &Buffer<'a>, output: &Buffer<'a>, sequences: usize,
        block: usize) -> Result<()> {
        // SAFETY: the retained Markov result bounds previous-token reads; checked anchors and fixed widths agree.
        on_device(self.library,self.device,||unsafe { self.library.glmf_dspark_confidence(hidden.raw().ptr,
            w1.raw().ptr,weight.raw().ptr,bias.raw().ptr,anchors.0.raw().ptr,tokens.raw().ptr,output.raw().ptr,
            sequences,block,HIDDEN,RANK,self.stream) })
    }

    pub fn download(&self, buffer: &Buffer<'a>, bytes: usize) -> Result<Vec<u8>> {
        check(buffer.raw(),bytes,1)?;
        self.keep(&[buffer])?; self.drain()?;
        let mut output=vec![0;bytes];
        if bytes > 0 { on_device(self.library,self.device,||self.library.copy_d2h(&mut output,
            CuteafdDeviceBuffer{bytes,..buffer.raw()}))?; }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(address: usize, bytes: usize) -> CuteafdDeviceBuffer {
        CuteafdDeviceBuffer{ptr:address as *mut _,bytes,device_id:0,flags:0}
    }

    #[test] fn draft_extents_reject_overflow_short_buffers_and_aliases() -> Result<()> {
        assert_eq!(extent(&[7,64,128,2])?,114688);
        assert!(extent(&[usize::MAX,2]).is_err());
        assert!(check(buffer(16,16),17,16).is_err());
        assert!(check(buffer(17,16),16,16).is_err());
        assert!(check(buffer(usize::MAX,1),1,1).is_err());
        assert!(disjoint(buffer(16,16),16,buffer(24,16),16).is_err());
        disjoint(buffer(16,16),16,buffer(32,16),16)?;
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
