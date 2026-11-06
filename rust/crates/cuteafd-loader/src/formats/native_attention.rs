//! Native attention geometry shared by checkpoint inventories and runtime owners.
use super::trellis_core::CoreSplit;
use anyhow::{Context, Result};

/// Unrotated selected-view attention with the same logical TP axes as K6.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeProjection { pub shape: [usize; 2], pub fp8: bool, pub split: CoreSplit }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeLayout {
    pub stem: String, pub shape: [usize; 2], pub fp8: bool,
    pub rank: usize, pub ranks: usize, pub split: CoreSplit,
}
impl NativeLayout {
    pub fn bytes(&self) -> Result<u64> {
        let [n, k] = self.shape;
        let values = n.checked_mul(k).and_then(|n| n.checked_mul(if self.fp8 { 1 } else { 2 }))
            .context("native attention projection extent overflow")?.max(256);
        let scales = if self.fp8 { n.div_ceil(128).checked_mul(k.div_ceil(128)).and_then(|n| n.checked_mul(4))
            .context("native attention scale extent overflow")?.max(256) } else { 0 };
        Ok(values.checked_add(scales).context("native attention projection byte overflow")? as u64)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{tensor}: native FP8 attention requires one declared F32 block-128 scale tensor; {reason}")]
pub struct InvalidFp8Scale { pub tensor: String, pub reason: &'static str }

/// Resolve only tensors in the selected index, preserving each publication's name.
pub fn fp8_scale<'a>(selected: &'a crate::plan::checkpoint::SelectedProjection, stem: &str)
    -> Result<&'a crate::plan::checkpoint::CheckpointTensor> {
    let invalid = |reason| InvalidFp8Scale { tensor: format!("{stem}.weight"), reason };
    let crate::plan::checkpoint::ProjectionStorage::Native { dtype: cuteafd_core::DType::F8E4M3, shape }
        = &selected.storage else { return Err(invalid("source is not native E4M3").into()); };
    let mut scales = selected.tensors.iter().filter(|tensor| tensor.meta.name.strip_prefix(stem)
        .is_some_and(|suffix| matches!(suffix, ".weight_scale_inv" | ".qscale_weight")));
    let scale = scales.next().ok_or_else(|| invalid("scale is missing"))?;
    if scales.next().is_some() { return Err(invalid("scale is ambiguous").into()); }
    if shape.len() != 2 || scale.meta.dtype != cuteafd_core::DType::F32
        || scale.meta.shape != [shape[0].div_ceil(128), shape[1].div_ceil(128)]
        || scale.meta.byte_length != scale.meta.shape.iter().product::<usize>() as u64 * 4 {
        return Err(invalid("scale dtype or geometry differs from the selected weight").into());
    }
    Ok(scale)
}

/// FP8-compact is a conversion of native BF16, never a requantization of native FP8.
#[derive(Debug, thiserror::Error)]
#[error("{tensor}: HQA attention is already FP8; --fp8-compact-attention applies only to BF16 HQA attention (found {found})")]
pub struct UnsupportedFp8Conversion { pub tensor: String, pub found: String }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{plan::checkpoint::{CheckpointTensor, ProjectionStorage, SelectedProjection}, SafetensorsTensorMetadata};
    use cuteafd_core::DType;

    #[test]
    fn selected_fp8_scales_preserve_both_names_and_refuse_missing_or_ambiguous() -> Result<()> {
        for suffix in ["weight_scale_inv", "qscale_weight"] {
            let stem = "layers.0.attention.wq";
            let scale = CheckpointTensor { shard: "native.safetensors".into(), meta: SafetensorsTensorMetadata {
                name: format!("{stem}.{suffix}"), dtype: DType::F32, shape: vec![2, 4], byte_offset: 512, byte_length: 32 } };
            let mut selected = SelectedProjection { storage: ProjectionStorage::Native { dtype: DType::F8E4M3,
                shape: vec![256, 512] }, tensors: vec![scale.clone()], source_bytes: 32 };
            assert_eq!(fp8_scale(&selected, stem)?.meta.name, scale.meta.name);
            selected.tensors[0].meta.dtype = DType::Bf16;
            assert!(fp8_scale(&selected, stem).unwrap_err().is::<InvalidFp8Scale>());
            selected.tensors.clear();
            assert!(fp8_scale(&selected, stem).unwrap_err().is::<InvalidFp8Scale>());
            selected.tensors.push(scale.clone());
            let mut alias = scale;
            alias.meta.name = format!("{stem}.{}", if suffix == "qscale_weight" { "weight_scale_inv" } else { "qscale_weight" });
            selected.tensors.push(alias);
            assert!(fp8_scale(&selected, stem).unwrap_err().is::<InvalidFp8Scale>());
        }
        Ok(())
    }
}
