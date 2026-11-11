//! Mathematical K3.1 CPU oracle, not an installed device execution route.
//! The native indexer weights remain BF16 in every dense-core view. The
//! oracle uses F32 activations, matching `python/reference/.../k31.py`.
use super::config::KimiDsaConfig;
use anyhow::{ensure, Context, Result};

/// Flat checkpoint order is also the packed arena order; no core conversion,
/// head slicing or transposition applies to the native indexer operands.
#[derive(Debug, Clone, Copy)]
pub enum IndexerOperand { Query, Key, Heads, Norm, Bias }

impl IndexerOperand {
    pub fn arena_name(self) -> &'static str {
        match self { Self::Query => "mla.indexer.q", Self::Key => "mla.indexer.k",
            Self::Heads => "mla.indexer.weights", Self::Norm => "mla.indexer.norm", Self::Bias => "mla.indexer.bias" }
    }
}

pub fn load_native(cfg: &super::config::KimiK3Config, layer: usize, operand: IndexerOperand,
    mut read: impl FnMut(&str) -> Result<crate::tensors::LoadedTensor>) -> Result<crate::tensors::LoadedTensor> {
    ensure!(cfg.attention.get(layer) == Some(&super::config::KimiAttention::Mla), "Kimi indexer requires a DSA layer");
    let index = cfg.dsa.context("Kimi indexer requires K3.1 DSA config")?;
    index.geometry().validate()?;
    let width = index.heads.checked_mul(index.head_dim).context("Kimi index query extent overflow")?;
    let (leaf, shape) = match operand {
        IndexerOperand::Query => ("wq_b.weight", vec![width, cfg.q_lora_rank]),
        IndexerOperand::Key => ("wk.weight", vec![index.head_dim, cfg.hidden]),
        IndexerOperand::Heads => ("weights_proj.weight", vec![index.heads, cfg.hidden]),
        IndexerOperand::Norm => ("k_norm.weight", vec![index.head_dim]),
        IndexerOperand::Bias => ("k_norm.bias", vec![index.head_dim]),
    };
    let bytes = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d)).context("Kimi indexer operand extent overflow")?;
    let name = format!("language_model.model.layers.{layer}.self_attn.indexer.{leaf}");
    let tensor = read(&name).with_context(|| format!("read native indexer {name}"))?;
    ensure!(tensor.info.name == name && tensor.info.dtype == cuteafd_core::DType::Bf16
        && tensor.info.shape == shape && tensor.bytes.len() == bytes && tensor.info.byte_length == bytes as u64,
        "{name}: expected native BF16 indexer {shape:?} ({bytes} bytes)");
    Ok(tensor)
}

pub struct IndexerWeights<'a> {
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub heads: &'a [f32],
    pub norm: &'a [f32],
    pub bias: &'a [f32],
}

pub struct IndexRow {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub weights: Vec<f32>,
}

fn linear(input: &[f32], weights: &[f32], outputs: usize) -> Result<Vec<f32>> {
    ensure!(!input.is_empty() && outputs.checked_mul(input.len()) == Some(weights.len()),
        "Kimi DSA projection shape mismatch");
    ensure!(input.iter().chain(weights).all(|v| v.is_finite()), "nonfinite Kimi DSA projection operand");
    let output: Vec<f32> = weights.chunks_exact(input.len()).map(|row| row.iter().zip(input).map(|(w,x)| w*x).sum()).collect();
    ensure!(output.iter().all(|v| v.is_finite()), "Kimi DSA projection overflow");
    Ok(output)
}

impl KimiDsaConfig {
    /// The latent is the MLA's already-normalized q-LoRA residual, not hidden
    /// states and not a second input LayerNorm. Offset 1 never removes self.
    pub fn project(&self, hidden: &[f32], latent: &[f32], weights: IndexerWeights<'_>, eps: f32) -> Result<IndexRow> {
        self.geometry().validate()?;
        ensure!(eps.is_finite() && eps > 0.0 && weights.norm.len() == self.head_dim
            && weights.bias.len() == self.head_dim, "Kimi DSA LayerNorm geometry");
        ensure!(weights.norm.iter().chain(weights.bias).all(|v| v.is_finite()), "nonfinite Kimi DSA normalization operand");
        let query_width = self.heads.checked_mul(self.head_dim).context("Kimi index query width overflow")?;
        let query = linear(latent, weights.query, query_width)?;
        let key = linear(hidden, weights.key, self.head_dim)?;
        let mean = key.iter().sum::<f32>() / self.head_dim as f32;
        let variance = key.iter().map(|x| (x-mean).powi(2)).sum::<f32>() / self.head_dim as f32;
        let inv = (variance + eps).sqrt().recip();
        let key: Vec<f32> = key.iter().zip(weights.norm).zip(weights.bias).map(|((x,w),b)| (x-mean)*inv*w+b).collect();
        ensure!(key.iter().all(|v| v.is_finite()), "Kimi DSA normalization overflow");
        let scale = (self.heads as f32).sqrt().recip() * (self.head_dim as f32).sqrt().recip();
        let weights = linear(hidden, weights.heads, self.heads)?.iter().map(|w| w*scale).collect();
        Ok(IndexRow { query, key, weights })
    }
}

/// One attention head over explicit selected logical tokens. Produces the
/// sigmoid-gated value before the ordinary output projection. The caller owns
/// MLA query scaling; K3.1 uses `1/sqrt(qk_nope_dim + qk_rope_dim)`.
pub fn selected_attention(query: &[f32], keys: &[f32], values: &[f32], gate: &[f32],
    selected: &[usize], position: usize, scale: f32) -> Result<Vec<f32>> {
    ensure!(!query.is_empty() && !gate.is_empty() && keys.len() % query.len() == 0
        && values.len() % gate.len() == 0 && keys.len()/query.len() == values.len()/gate.len(),
        "Kimi sparse MLA shape mismatch");
    let tokens = keys.len()/query.len();
    ensure!(!selected.is_empty() && position < tokens && scale.is_finite() && scale > 0.0
        && selected.iter().all(|&id| id <= position)
        && selected.windows(2).all(|ids| ids[0] < ids[1]), "Kimi sparse MLA selected ids must be unique, ascending and causal");
    let scores: Vec<f32> = selected.iter().map(|&id| query.iter()
        .zip(&keys[id*query.len()..(id+1)*query.len()]).map(|(q,k)| q*k).sum::<f32>()*scale).collect();
    ensure!(scores.iter().chain(values).chain(gate).all(|v| v.is_finite()), "nonfinite Kimi sparse MLA operand");
    let max = scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);
    let probabilities: Vec<f32> = scores.iter().map(|s| (s-max).exp()).collect();
    let total = probabilities.iter().sum::<f32>();
    Ok(gate.iter().enumerate().map(|(d,g)| {
        selected.iter().zip(&probabilities).map(|(&id,p)| p/total*values[id*gate.len()+d]).sum::<f32>()
            / (1.0+(-g).exp())
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn k31_indexer_loader_preserves_native_bits_and_rejects_conversion() -> Result<()> {
        use cuteafd_core::{DType, TensorInfo, TensorRole};
        use crate::tensors::LoadedTensor;
        let mut cfg = super::super::config::KimiK3Config::from_hf(&super::super::config::tests::k31_config())?;
        cfg.hidden = 4; cfg.q_lora_rank = 4;
        cfg.dsa = Some(KimiDsaConfig { heads: 2, head_dim: 4, topk: 2 });
        for (operand, shape) in [(IndexerOperand::Query, vec![8,4]), (IndexerOperand::Key, vec![4,4]),
            (IndexerOperand::Heads, vec![2,4]), (IndexerOperand::Norm, vec![4]), (IndexerOperand::Bias, vec![4])] {
            let bytes: Vec<u8> = (0..shape.iter().product::<usize>() as u16).flat_map(u16::to_le_bytes).collect();
            let read = |name: &str| -> Result<LoadedTensor> {
                Ok(LoadedTensor { info: TensorInfo { name: name.into(), file: "test".into(), dtype: DType::Bf16,
                    shape: shape.clone(), byte_offset: 0, byte_length: bytes.len() as u64, role: TensorRole::Other,
                    layer_id: Some(3), expert_id: None, is_quantization_metadata: false },
                    source_path: "test".into(), bytes: bytes.clone(), elapsed_micros: 0, sha256: String::new() })
            };
            let loaded = load_native(&cfg, 3, operand, read)?;
            assert_eq!(loaded.bytes, bytes);
            assert_eq!(loaded.info.shape, shape);
            assert!(operand.arena_name().starts_with("mla.indexer."));
            assert!(load_native(&cfg, 3, operand, |name| {
                let mut tensor = read(name)?; tensor.info.dtype = DType::F8E4M3; Ok(tensor)
            }).is_err());
            assert!(load_native(&cfg, 3, operand, |name| {
                let mut tensor = read(name)?; tensor.bytes.pop(); Ok(tensor)
            }).is_err());
            assert!(load_native(&cfg, 0, operand, |_| unreachable!()).is_err());
        }
        cfg.dsa = None;
        assert!(load_native(&cfg, 3, IndexerOperand::Query, |_| unreachable!()).is_err());
        Ok(())
    }

    #[test]
    fn k31_indexer_and_selected_attention_match_python_reference_vectors() -> Result<()> {
        let fixture: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../../../scripts/tests/fixtures/kimi-k31-dsa.json")))?;
        let array = |key: &str| -> Vec<f32> { fixture[key].as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap() as f32).collect() };
        let (x,latent,wq,wk,wh,norm,bias) = (array("hidden"),array("latent"),array("wq"),array("wk"),
            array("wh"),array("norm"),array("bias"));
        let cfg = KimiDsaConfig { heads: 2, head_dim: 4, topk: 2 };
        let mut rows = Vec::new();
        let mut keys = Vec::new();
        for (hidden,latent) in x.chunks_exact(4).zip(latent.chunks_exact(4)) {
            let row = cfg.project(hidden,latent,IndexerWeights { query:&wq,key:&wk,heads:&wh,norm:&norm,bias:&bias },1e-5)?;
            keys.extend_from_slice(&row.key);
            rows.push(row);
        }
        let close = |actual: &[f32], expected: &[f32]| {
            assert_eq!(actual.len(),expected.len());
            for (a,e) in actual.iter().zip(expected) { assert!((a-e).abs() <= 3e-6 + 3e-5*e.abs(), "{a} != {e}"); }
        };
        close(&keys,&array("index_keys"));
        close(&rows.iter().flat_map(|r| r.query.iter().copied()).collect::<Vec<_>>(), &array("index_queries"));
        let (q,k,v,g) = (array("mla_query"),array("mla_keys"),array("mla_values"),array("gate"));
        let mut attended = Vec::new();
        for (position,row) in rows.iter().enumerate() {
            let scores = cfg.geometry().scores(&row.query,&keys,Some(&row.weights))?;
            let selected = cfg.geometry().select(&scores,position)?;
            let expected: Vec<i32> = fixture["selected"][position].as_array().unwrap().iter().map(|v|v.as_i64().unwrap() as i32).collect();
            assert_eq!(selected[0],expected);
            let valid: Vec<usize> = selected[0].iter().copied().filter(|&id| id >= 0).map(|id| id as usize).collect();
            attended.extend(selected_attention(&q[position*4..position*4+4],&k,&v,&g[position*2..position*2+2],
                &valid,position,0.5)?);
        }
        close(&attended,&array("gated_values"));
        assert!(selected_attention(&q[..4],&k,&v,&g[..2],&[1],0,0.5).is_err());
        Ok(())
    }
}
