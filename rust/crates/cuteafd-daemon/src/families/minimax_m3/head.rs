use super::{ffn, native::Native, scratch::Scratch, weights::Weights};
use anyhow::{ensure, Result};
use cuteafd_ffi::{programs::VocabularyHead, CuteafdDeviceBuffer};
use crate::shared::{prefix::view, target_head::{self, TargetHead}, token_io::DeviceLogits};

/// The BF16 target head over the admitted `head_workspace`, which the
/// execution's scratch keeps until after the head is dropped.
pub(super) struct TargetHeadOwner<'a> { head: VocabularyHead<'a>, rows: usize, vocab: usize }
impl<'a> TargetHeadOwner<'a> {
    pub fn new(native: &Native<'a>, scratch: &Scratch<'a>, rows: usize, vocab: usize) -> Result<Self> {
        let workspace = scratch.get("head_workspace")?;
        ensure!(workspace.bytes >= 4<<20 && rows > 0 && rows <= 4096, "MiniMax head admission");
        native.check_device(&[workspace])?;
        // SAFETY: the scratch keeps the workspace live until the head is dropped.
        let head = unsafe { native.library().vocabulary_head_rows(workspace.ptr,6144,rows as u32,vocab as u32)? };
        Ok(Self { head, rows, vocab })
    }
    /// Final norm and head over residual `rows` only (prefill needs its last rows).
    pub fn logits(&self, native: &Native<'a>, weights: &Weights<'a>, scratch: &Scratch<'a>, rows: std::ops::Range<usize>) -> Result<DeviceLogits> {
        let residual=view(scratch.get("residual")?,rows.start*6144*4,rows.len()*6144*4)?;
        ffn::norm(native,residual,weights.get("language_model.model.norm.weight")?,scratch.get("head_input")?,rows.len())?;
        self.project(native,weights,scratch,scratch.get("head_input")?,rows.len())
    }
    pub fn project(&self, native: &Native<'a>, weights: &Weights<'a>, scratch: &Scratch<'a>, input: CuteafdDeviceBuffer, rows: usize) -> Result<DeviceLogits> {
        let (weight, output) = (weights.get("language_model.lm_head.weight")?, scratch.get("logits")?);
        ensure!(rows>0 && rows<=self.rows && input.bytes>=rows*6144*2
            && weight.bytes>=self.vocab*6144*2 && output.bytes>=rows*self.vocab*4, "MiniMax head operands");
        native.check_device(&[input,weight,output])?;
        // SAFETY: the head's fixed geometry and the live extents agree; the
        // execution keeps every operand's owner through the stream's drain.
        unsafe { target_head::launch(TargetHead::configured(),&self.head,input.ptr,weight.ptr,output.ptr.cast(),rows,native.raw())? };
        self.output(native,scratch,rows)
    }
    /// The logits `rows` projections leave (a replayed step's, too).
    pub fn output(&self, native: &Native<'a>, scratch: &Scratch<'a>, rows: usize) -> Result<DeviceLogits> {
        let output = scratch.get("logits")?;
        Ok(DeviceLogits { ptr: output.ptr.cast_const(), rows, vocab:self.vocab,
            stride:self.vocab,stream:native.raw(),greedy:None })
    }
}
