//! Native HQA restores Mistral's original E4M3 weights and F32 block scales.
use crate::plan::Checkpoint;
use cuteafd_core::DType;

use super::Ml4Config;

#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("Mistral Large 4 HQA {tensor}: expected {expected}")]
    NativeTensor { tensor: String, expected: String },
    #[error("{0}")]
    UnsupportedFp8Conversion(#[from] crate::formats::native_attention::UnsupportedFp8Conversion),
}

/// Check the complete restored wq/wo inventory before runtime admission. The
/// official FP8 source has no HQA declaration and remains inventory-readable.
pub fn validate(checkpoint: &Checkpoint, cfg: &Ml4Config) -> Result<(), AttentionError> {
    if checkpoint.config["quantization_config"].get("attention_view").is_none() {
        return Ok(());
    }
    for layer in 0..cfg.layers {
        for (projection, shape) in [
            ("wq", [cfg.heads * cfg.qk_head_dim(), cfg.hidden]),
            ("wo", [cfg.hidden, cfg.heads * cfg.v_head_dim]),
        ] {
            let stem = format!("layers.{layer}.attention.{projection}");
            for (suffix, dtype, shape) in [
                ("weight", DType::F8E4M3, shape),
                ("qscale_weight", DType::F32, shape.map(|n| n.div_ceil(128))),
            ] {
                let name = format!("{stem}.{suffix}");
                let width = if dtype == DType::F32 { 4 } else { 1 };
                let expected = shape.iter().try_fold(width, |bytes: usize, &n| bytes.checked_mul(n));
                let valid = checkpoint.tensors.iter().find(|tensor| tensor.meta.name == name)
                    .is_some_and(|tensor| tensor.meta.dtype == dtype && tensor.meta.shape == shape
                        && expected == usize::try_from(tensor.meta.byte_length).ok());
                if !valid {
                    return Err(AttentionError::NativeTensor {
                        tensor: name, expected: format!("{dtype:?} {shape:?}"),
                    });
                }
            }
            if checkpoint.contains_tensor(&format!("{stem}.trellis")) {
                return Err(AttentionError::NativeTensor {
                    tensor: stem, expected: "native FP8 attention without EXL3 companions".into(),
                });
            }
        }
    }
    if checkpoint.attention_mode() == crate::plan::checkpoint::AttentionMode::Fp8Compact {
        return Err(crate::formats::native_attention::UnsupportedFp8Conversion {
            tensor: "layers.0.attention.wq.weight".into(), found: "E4M3".into(),
        }.into());
    }
    Ok(())
}
