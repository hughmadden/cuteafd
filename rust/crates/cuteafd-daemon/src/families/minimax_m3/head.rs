use super::{ffn, scratch::Scratch, weights::Weights};
use anyhow::Result;
use cuteafd_ffi::minimax::{Buffer,Commands,Head};
use crate::shared::{target_head::TargetHead,token_io::DeviceLogits};

pub(super) struct TargetHeadOwner<'a> { head: Head<'a>, vocab: usize }
impl<'a> TargetHeadOwner<'a> {
    pub fn new(commands: &Commands<'a>, scratch: &Scratch<'a>, rows: usize, vocab: usize) -> Result<Self> {
        Ok(Self { head: Head::new(commands,scratch.get("head_workspace")?.clone(),rows,vocab)?, vocab })
    }
    pub fn logits(&self, commands: &Commands<'a>, weights: &Weights<'a>, scratch: &Scratch<'a>, rows: usize) -> Result<DeviceLogits> {
        ffn::norm(commands,scratch.get("residual")?,weights.get("language_model.model.norm.weight")?,scratch.get("head_input")?,rows)?;
        self.project(commands,weights,scratch,scratch.get("head_input")?,rows)
    }
    pub fn project(&self, commands: &Commands<'a>, weights: &Weights<'a>, scratch: &Scratch<'a>, input: &Buffer<'a>, rows: usize) -> Result<DeviceLogits> {
        let output = scratch.get("logits")?;
        self.head.launch(commands,input,weights.get("language_model.lm_head.weight")?,
            output,rows,TargetHead::configured().tensor_cores())?;
        Ok(DeviceLogits { ptr: output.raw().ptr.cast_const(), rows, vocab:self.vocab,
            stride:self.vocab,stream:commands.raw(),greedy:None })
    }
}
