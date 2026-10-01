//! GLM 5.3 Flash (glm5_next) model arguments from the Hugging Face `config.json`
//! (`text_config` of `Glm5NextForConditionalGeneration`).
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::plan::checkpoint::read_json;

/// A layer's attention: Kimi Delta Attention (linear, recurrent state) or
/// MLA without RoPE with a DSA indexer over 4-token key pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmNextAttention {
    Kda,
    Mla,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlmNextConfig {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub attention: Vec<GlmNextAttention>,
    /// `dense[layer]`: a SwiGLU MLP instead of the MoE.
    pub dense: Vec<bool>,
    pub dense_intermediate: usize,
    pub experts: usize,
    pub topk: usize,
    pub moe_intermediate: usize,
    pub routed_scale: f64,
    pub swiglu_limit: f64,
    pub rms_norm_eps: f64,
    pub hc_mult: usize,
    pub kda_heads: usize,
    pub kda_head_dim: usize,
    pub heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_dim: usize,
    pub v_head_dim: usize,
    pub index_topk: usize,
    pub index_kpool: usize,
    pub eos: Vec<u32>,
}

impl GlmNextConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        let v = root.get("text_config").unwrap_or(root);
        ensure!(v["model_type"] == "glm5_next_text" || root["model_type"] == "glm5_next", "not a glm5_next config");
        let int = |value: &Value, key: &str| -> Result<usize> {
            value[key].as_u64().map(|x| x as usize).with_context(|| format!("glm5_next config lacks {key}"))
        };
        let layers = int(v, "num_hidden_layers")?;
        let types = v["layer_types"].as_array().context("glm5_next config lacks layer_types")?;
        ensure!(types.len() == layers, "layer_types has {} entries for {layers} layers", types.len());
        let attention = types.iter().map(|t| match t.as_str() {
            Some("linear_attention") => Ok(GlmNextAttention::Kda),
            Some("deepseek_sparse_attention") => Ok(GlmNextAttention::Mla),
            other => anyhow::bail!("unknown glm5_next layer type {other:?}"),
        }).collect::<Result<Vec<_>>>()?;
        let first_k = v.get("first_k_dense_replace").filter(|x| !x.is_null())
            .map(|x| x.as_u64().map(|x| x as usize).context("first_k_dense_replace must be an unsigned integer"))
            .transpose()?;
        let dense: Vec<bool> = match v.get("mlp_layer_types").filter(|x| !x.is_null()) {
            Some(types) => {
                let types = types.as_array().context("mlp_layer_types must be a list")?;
                ensure!(types.len() == layers, "mlp_layer_types has {} entries for {layers} layers", types.len());
                let dense = types.iter().map(|t| match t.as_str() {
                    Some("dense") => Ok(true),
                    Some("sparse") => Ok(false),
                    other => anyhow::bail!("unknown glm5_next MLP type {other:?}"),
                }).collect::<Result<Vec<_>>>()?;
                if let Some(k) = first_k {
                    ensure!(dense.iter().enumerate().all(|(l, d)| *d == (l < k)),
                        "mlp_layer_types and first_k_dense_replace ({k}) disagree");
                }
                dense
            }
            None => (0..layers).map(|l| l < first_k.unwrap_or(0)).collect(),
        };
        let linear = &v["linear_attn_config"];
        ensure!(v["mla_use_nope"].as_bool().unwrap_or(false) && int(v, "qk_rope_head_dim").unwrap_or(0) == 0,
            "the glmf programs are built for MLA without RoPE (mla_use_nope)");
        ensure!(v["indexer_types"].as_array().map_or(true, |t| t.iter().all(|x| x == "full")),
            "shared-indexer layers are not supported yet");
        ensure!(v["scoring_func"].as_str().unwrap_or("sigmoid") == "sigmoid"
            && v["n_group"].as_u64().unwrap_or(1) == 1 && v["norm_topk_prob"].as_bool().unwrap_or(true),
            "glm5_next routing must be normalized sigmoid noaux_tc without groups");
        ensure!(linear["gate_lower_bound"].as_f64() == Some(-5.0) && int(linear, "short_conv_kernel_size")? == 4,
            "the glmf KDA program is built for a -5 gate bound and a 4-tap short conv");
        let eos = match &v["eos_token_id"] {
            Value::Array(ids) => ids.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
            Value::Number(id) => id.as_u64().map(|x| x as u32).into_iter().collect(),
            _ => Vec::new(),
        };
        Ok(Self {
            vocab_size: int(v, "vocab_size")?,
            hidden: int(v, "hidden_size")?,
            layers,
            attention,
            dense,
            dense_intermediate: int(v, "intermediate_size")?,
            experts: int(v, "n_routed_experts")?,
            topk: int(v, "num_experts_per_tok")?,
            moe_intermediate: int(v, "moe_intermediate_size")?,
            routed_scale: v["routed_scaling_factor"].as_f64().unwrap_or(1.0),
            swiglu_limit: v["swiglu_limit"].as_f64().unwrap_or(0.0),
            rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-5),
            hc_mult: int(v, "hc_mult")?,
            kda_heads: int(linear, "num_heads")?,
            kda_head_dim: int(linear, "head_dim")?,
            heads: int(v, "num_attention_heads")?,
            q_lora_rank: int(v, "q_lora_rank")?,
            kv_lora_rank: int(v, "kv_lora_rank")?,
            qk_nope_dim: int(v, "qk_nope_head_dim")?,
            v_head_dim: int(v, "v_head_dim")?,
            index_topk: int(v, "index_topk")?,
            index_kpool: int(v, "index_kpool").unwrap_or(1),
            eos,
        })
    }

    /// Longest context whose DSA selection is every earlier token (no indexer needed).
    pub fn dense_context(&self) -> usize {
        self.index_topk + self.index_kpool - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm53_flash_config() -> Result<()> {
        let types: Vec<&str> = (0..45).map(|l| if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" })
            .collect();
        let mlp: Vec<&str> = (0..45).map(|l| if l < 3 { "dense" } else { "sparse" }).collect();
        let cfg = GlmNextConfig::from_hf(&serde_json::json!({
            "model_type": "glm5_next", "text_config": {
                "model_type": "glm5_next_text", "vocab_size": 154880, "hidden_size": 4096, "num_hidden_layers": 45,
                "layer_types": types, "mlp_layer_types": mlp, "intermediate_size": 12288, "n_routed_experts": 288,
                "num_experts_per_tok": 8, "moe_intermediate_size": 2048, "routed_scaling_factor": 2.5,
                "swiglu_limit": 10.0, "rms_norm_eps": 1e-5, "hc_mult": 4, "mla_use_nope": true,
                "qk_rope_head_dim": 0, "num_attention_heads": 64, "q_lora_rank": 1536, "kv_lora_rank": 512,
                "qk_nope_head_dim": 256, "v_head_dim": 256, "index_topk": 2048, "index_kpool": 4,
                "eos_token_id": [154820, 154827, 154829],
                "linear_attn_config": {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4,
                                       "gate_lower_bound": -5.0}}}))?;
        assert_eq!(cfg.attention.iter().filter(|a| **a == GlmNextAttention::Kda).count(), 34);
        assert_eq!(cfg.attention[3], GlmNextAttention::Mla);
        assert_eq!(cfg.dense.iter().filter(|d| **d).count(), 3);
        assert_eq!(cfg.dense_context(), 2051);
        Ok(())
    }
}
