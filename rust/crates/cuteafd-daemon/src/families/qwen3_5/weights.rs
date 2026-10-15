//! Native dense BF16 operands; no fabricated MoE, mHC, PLE or index weights.
use crate::shared::memory::device::Allocation;
use crate::shared::qwen::weights::WeightReader;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_loader::families::qwen::HybridAttention;
use cuteafd_loader::families::qwen3_5::Qwen35Config;
use std::{collections::HashMap, ffi::c_void};

const PREFIX: &str = "model.language_model.";

pub(crate) struct DenseLayer<'a> {
    pub(crate) attention: HybridAttention,
    operands: HashMap<&'static str, Allocation<'a>>,
}

impl DenseLayer<'_> {
    pub(crate) fn ptr(&self, name: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(name).with_context(|| format!("dense Qwen layer has no operand {name}"))?.buffer.ptr)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
    }
}

pub(crate) struct DenseWeights<'a> {
    pub(crate) layers: Vec<DenseLayer<'a>>,
    pub(crate) norm: Allocation<'a>,
    pub(crate) head: Allocation<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Storage { Bf16, F32, NativeNorm }

#[derive(Debug, PartialEq, Eq)]
struct Operand {
    name: &'static str,
    tensors: Vec<String>,
    storage: Storage,
}

fn layer_operands(prefix: &str, attention: HybridAttention) -> Vec<Operand> {
    let mut out = Vec::new();
    let mut add = |name, tensors: &[&str], storage| out.push(Operand { name,
        tensors: tensors.iter().map(|t| format!("{prefix}.{t}")).collect(), storage });
    for (operand, tensor) in [("attn.norm", "input_layernorm.weight"), ("ffn.norm", "post_attention_layernorm.weight"),
        ("ffn.w_down", "mlp.down_proj.weight")] {
        add(operand, &[tensor], Storage::Bf16);
    }
    add("ffn.w_gate_up", &["mlp.gate_proj.weight", "mlp.up_proj.weight"], Storage::Bf16);
    match attention {
        HybridAttention::Gdn => {
            add("w_in", &["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"], Storage::Bf16);
            for (operand, tensor) in [("conv_w", "conv1d.weight"), ("a_log", "A_log"), ("dt_bias", "dt_bias")] {
                add(operand, &[&format!("linear_attn.{tensor}")], Storage::F32);
            }
            add("norm_w", &["linear_attn.norm.weight"], Storage::NativeNorm);
            add("w_out", &["linear_attn.out_proj.weight"], Storage::Bf16);
        }
        HybridAttention::Full => {
            add("w_in", &["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"], Storage::Bf16);
            for (operand, tensor) in [("q_norm", "q_norm.weight"), ("k_norm", "k_norm.weight"), ("w_o", "o_proj.weight")] {
                add(operand, &[&format!("self_attn.{tensor}")], Storage::Bf16);
            }
        }
    }
    out
}

pub(crate) struct DenseWeightBytes {
    pub(crate) layers: Vec<u64>,
    pub(crate) norm: u64,
    pub(crate) head: u64,
    pub(crate) embedding: u64,
}

impl<'a> WeightReader<'a> {
    /// Header-only native allocation inventory, including widened tiny tensors.
    pub(crate) fn dense_bf16_bytes(&self, cfg: &Qwen35Config, norm_dtype: DType) -> Result<DenseWeightBytes> {
        ensure!(matches!(norm_dtype, DType::Bf16 | DType::F32), "dense Qwen GDN norm ABI must be BF16 or FP32");
        let specs: Vec<_> = cfg.attention.iter().enumerate().map(|(layer, &kind)|
            layer_operands(&format!("{PREFIX}layers.{layer}"), kind)).collect();
        for spec in specs.iter().flatten() {
            for name in &spec.tensors {
                let dtype = &self.tensor(name)?.meta.dtype;
                let expected = match spec.storage { Storage::NativeNorm => &norm_dtype, _ => &DType::Bf16 };
                ensure!(dtype == expected || (spec.storage == Storage::F32 && *dtype == DType::F32),
                    "{name}: dense BF16 export expects {expected:?}, found {dtype:?}; native FP8 requires its own export");
            }
        }
        let norm_name = format!("{PREFIX}norm.weight");
        let head_name = if cfg.tied_embeddings { format!("{PREFIX}embed_tokens.weight") } else { "lm_head.weight".into() };
        for name in [&norm_name, &head_name] {
            ensure!(self.tensor(name)?.meta.dtype == DType::Bf16, "{name}: expected BF16");
        }
        let layers = specs.iter().map(|operands| operands.iter().try_fold(0u64, |total, spec| {
            let bytes = spec.tensors.iter().try_fold(0u64, |total, name| {
                let tensor = self.tensor(name)?;
                let bytes = if spec.storage == Storage::F32 && tensor.meta.dtype == DType::Bf16 {
                    tensor.meta.byte_length.checked_mul(2).context("dense widened weight size overflow")?
                } else { tensor.meta.byte_length };
                total.checked_add(bytes).context("dense packed weight size overflow")
            })?.max(256);
            total.checked_add(bytes).context("dense layer weight size overflow")
        })).collect::<Result<_>>()?;
        let embedding = self.tensor(&format!("{PREFIX}embed_tokens.weight"))?;
        ensure!(embedding.meta.dtype == DType::Bf16, "dense embedding must be native BF16");
        Ok(DenseWeightBytes { layers, norm: self.tensor(&norm_name)?.meta.byte_length.max(256),
            head: self.tensor(&head_name)?.meta.byte_length.max(256), embedding: embedding.meta.byte_length })
    }

    /// Validate every header before uploading any weights.
    pub(crate) fn dense_bf16(&self, cfg: &Qwen35Config, norm_dtype: DType) -> Result<DenseWeights<'a>> {
        self.dense_bf16_bytes(cfg, norm_dtype)?;
        let norm_name = format!("{PREFIX}norm.weight");
        let head_name = if cfg.tied_embeddings { format!("{PREFIX}embed_tokens.weight") } else { "lm_head.weight".into() };
        let specs = cfg.attention.iter().enumerate().map(|(layer, &kind)|
            layer_operands(&format!("{PREFIX}layers.{layer}"), kind));
        let layers = cfg.attention.iter().zip(specs).map(|(&attention, specs)| {
            let operands = specs.into_iter().map(|spec| {
                let weight = match spec.storage {
                    Storage::Bf16 if spec.tensors.len() > 1 => self.rows(&spec.tensors, 0)?,
                    Storage::Bf16 => self.bf16(&spec.tensors[0])?,
                    Storage::F32 => self.f32(&spec.tensors[0])?,
                    Storage::NativeNorm => {
                        let (bytes, _, _) = self.raw(&spec.tensors[0])?;
                        self.upload(&bytes)?
                    }
                };
                Ok((spec.name, weight))
            }).collect::<Result<_>>()?;
            Ok(DenseLayer { attention, operands })
        }).collect::<Result<_>>()?;
        Ok(DenseWeights { layers, norm: self.bf16(&norm_name)?, head: self.bf16(&head_name)? })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_operands_keep_native_projection_order_and_norm_precision() {
        for kind in [HybridAttention::Gdn, HybridAttention::Full] {
            let specs = layer_operands("layer", kind);
            let names = |operand: &str| specs.iter().find(|s| s.name == operand).unwrap().tensors.clone();
            assert_eq!(names("ffn.w_gate_up"), ["layer.mlp.gate_proj.weight", "layer.mlp.up_proj.weight"]);
            if kind == HybridAttention::Gdn {
                assert_eq!(names("w_in"), ["layer.linear_attn.in_proj_qkv.weight", "layer.linear_attn.in_proj_z.weight",
                    "layer.linear_attn.in_proj_b.weight", "layer.linear_attn.in_proj_a.weight"]);
                assert_eq!(specs.iter().find(|s| s.name == "norm_w").unwrap().storage, Storage::NativeNorm);
                assert_eq!(specs.iter().filter(|s| s.storage == Storage::F32).count(), 3);
            } else {
                assert_eq!(names("w_in"), ["layer.self_attn.q_proj.weight", "layer.self_attn.k_proj.weight", "layer.self_attn.v_proj.weight"]);
            }
            assert!(specs.iter().all(|s| !s.name.contains("expert") && !s.name.contains("hc") && !s.name.contains("index")));
        }
    }
}
