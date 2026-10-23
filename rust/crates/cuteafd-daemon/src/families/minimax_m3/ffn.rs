use super::{scratch::Scratch, weights::Weights};
use anyhow::Result;
use cuteafd_ffi::{minimax::{Buffer,Commands}, programs::Scalar};

pub(super) fn linear<'a>(commands: &Commands<'a>, weights: &Weights<'a>, scratch: &Scratch<'a>,
    layer: usize, projection: &str, stem: &str, input: &Buffer<'a>, output: &Buffer<'a>, rows: usize) -> Result<()> {
    commands.launch(&format!("minimax_mxfp8_{projection}"), &[input,
        weights.layer(layer,&format!("{stem}.weight"))?, weights.layer(layer,&format!("{stem}.weight_scale_inv"))?,
        scratch.get("scratch")?,output], &[Scalar::I32(rows as i32),Scalar::I32(0)])
}

pub(super) fn norm<'a>(commands: &Commands<'a>, input: &Buffer<'a>, weight: &Buffer<'a>, output: &Buffer<'a>, rows: usize) -> Result<()> {
    commands.launch("minimax_norm", &[input,weight,weight,weight,weight,weight,output], &[Scalar::I32(rows as i32)])
}

pub(super) fn dense<'a>(commands: &Commands<'a>, weights: &Weights<'a>, scratch: &Scratch<'a>, layer: usize, rows: usize, shared: bool) -> Result<()> {
    let (prefix, stem, width, output) = if shared { ("shared", "block_sparse_moe.shared_experts",3072,"shared") }
        else { ("dense", "mlp",12288,"delta") };
    linear(commands,weights,scratch,layer,&format!("{prefix}_gate_up"),&format!("{stem}.gate_up_proj"),
        scratch.get("norm")?,scratch.get("gate_up")?,rows)?;
    commands.launch(&format!("minimax_swiglu_i{width}"), &[scratch.get("gate_up")?,scratch.get("activation")?], &[Scalar::I32(rows as i32)])?;
    linear(commands,weights,scratch,layer,&format!("{prefix}_down"),&format!("{stem}.down_proj"),
        scratch.get("activation")?,scratch.get(output)?,rows)
}
