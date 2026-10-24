use super::{ffn, scratch::Scratch, weights::Weights};
use anyhow::Result;
use cuteafd_ffi::minimax::{Buffer,Commands,Head};
use crate::shared::{target_head::TargetHead,token_io::DeviceLogits};

pub(super) struct TargetHeadOwner<'a> { head: Head<'a>, vocab: usize }
impl<'a> TargetHeadOwner<'a> {
    pub fn new(commands: &Commands<'a>, scratch: &Scratch<'a>, rows: usize, vocab: usize) -> Result<Self> {
        Ok(Self { head: Head::new(commands,scratch.get("head_workspace")?.clone(),rows,vocab)?, vocab })
    }
    /// Final norm and head over residual `rows` only (prefill needs its last rows).
    pub fn logits(&self, commands: &Commands<'a>, weights: &Weights<'a>, scratch: &Scratch<'a>, rows: std::ops::Range<usize>) -> Result<DeviceLogits> {
        let residual=scratch.get("residual")?.view(rows.start*6144*4,rows.len()*6144*4)?;
        ffn::norm(commands,&residual,weights.get("language_model.model.norm.weight")?,scratch.get("head_input")?,rows.len())?;
        self.project(commands,weights,scratch,scratch.get("head_input")?,rows.len())
    }
    pub fn project(&self, commands: &Commands<'a>, weights: &Weights<'a>, scratch: &Scratch<'a>, input: &Buffer<'a>, rows: usize) -> Result<DeviceLogits> {
        self.head.launch(commands,input,weights.get("language_model.lm_head.weight")?,
            scratch.get("logits")?,rows,TargetHead::configured().tensor_cores())?;
        self.output(commands,scratch,rows)
    }
    /// The logits `rows` projections leave (a replayed step's, too).
    pub fn output(&self, commands: &Commands<'a>, scratch: &Scratch<'a>, rows: usize) -> Result<DeviceLogits> {
        let output = scratch.get("logits")?;
        Ok(DeviceLogits { ptr: output.raw().ptr.cast_const(), rows, vocab:self.vocab,
            stride:self.vocab,stream:commands.raw(),greedy:None })
    }
}
