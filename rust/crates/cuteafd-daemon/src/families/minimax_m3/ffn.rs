use super::{native::Native, scratch::Rows, weights::Weights};
use anyhow::Result;
use cuteafd_ffi::{programs::Scalar, CuteafdDeviceBuffer};

/// One native MXFP8 projection; decode-sized rows take its split-K program.
#[allow(clippy::too_many_arguments)]
pub(super) fn project(native: &Native<'_>, projection: &str, weight: CuteafdDeviceBuffer, scales: CuteafdDeviceBuffer,
    scratch: CuteafdDeviceBuffer, input: CuteafdDeviceBuffer, output: CuteafdDeviceBuffer, rows: usize) -> Result<()> {
    native.launch(&native.projection(projection, rows), &[input, weight, scales, scratch, output],
        &[Scalar::I32(rows as i32),Scalar::I32(0)])
}

/// [`project`] through a target layer's `{stem}.weight` and its scales.
#[allow(clippy::too_many_arguments)]
pub(super) fn linear(native: &Native<'_>, weights: &Weights<'_>, scratch: CuteafdDeviceBuffer,
    layer: usize, projection: &str, stem: &str, input: CuteafdDeviceBuffer, output: CuteafdDeviceBuffer, rows: usize) -> Result<()> {
    project(native, projection, weights.layer(layer,&format!("{stem}.weight"))?,
        weights.layer(layer,&format!("{stem}.weight_scale_inv"))?, scratch, input, output, rows)
}

pub(super) fn norm(native: &Native<'_>, input: CuteafdDeviceBuffer, weight: CuteafdDeviceBuffer, output: CuteafdDeviceBuffer,
    rows: usize) -> Result<()> {
    native.launch("minimax_norm", &[input,weight,weight,weight,weight,weight,output], &[Scalar::I32(rows as i32)])
}

pub(super) fn dense(native: &Native<'_>, weights: &Weights<'_>, s: &Rows<'_>, layer: usize, rows: usize, shared: bool) -> Result<()> {
    let (prefix, stem, width, output) = if shared { ("shared", "block_sparse_moe.shared_experts",3072,"shared") }
        else { ("dense", "mlp",12288,"delta") };
    linear(native,weights,s.get("scratch")?,layer,&format!("{prefix}_gate_up"),&format!("{stem}.gate_up_proj"),
        s.get("norm")?,s.get("gate_up")?,rows)?;
    native.launch(&format!("minimax_swiglu_i{width}"), &[s.get("gate_up")?,s.get("activation")?], &[Scalar::I32(rows as i32)])?;
    linear(native,weights,s.get("scratch")?,layer,&format!("{prefix}_down"),&format!("{stem}.down_proj"),
        s.get("activation")?,s.get(output)?,rows)
}
