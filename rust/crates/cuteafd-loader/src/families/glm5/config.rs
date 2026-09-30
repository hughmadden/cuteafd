//! GLM 5.x (glm_moe_dsa) model arguments from the Hugging Face `config.json`.
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::plan::checkpoint::read_json;

/// Whether a layer runs its own DSA indexer or reuses the previous full
/// layer's top-k selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmIndexer {
    Full,
    Shared,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlmDsaConfig {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub indexers: Vec<GlmIndexer>,
    /// Layers `0..first_moe_layer` are dense MLPs.
    pub first_moe_layer: usize,
    pub dense_intermediate: usize,
    pub experts: usize,
    pub topk: usize,
    pub moe_intermediate: usize,
    pub shared_experts: usize,
    pub routed_scale: f64,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub mtp_layers: usize,
    pub eos_tokens: Vec<u32>,
}

impl GlmDsaConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(v: &Value) -> Result<Self> {
        let v = v.get("text_config").unwrap_or(v);
        ensure!(v["model_type"] == "glm_moe_dsa", "not a glm_moe_dsa config");
        let int = |key: &str| -> Result<usize> {
            v[key].as_u64().map(|x| x as usize).with_context(|| format!("glm_moe_dsa config lacks {key}"))
        };
        let float = |key: &str| -> Result<f64> {
            v[key].as_f64().with_context(|| format!("glm_moe_dsa config lacks {key}"))
        };
        let layers = int("num_hidden_layers")?;
        let indexers = match v["indexer_types"].as_array() {
            Some(types) => types.iter().map(|t| match t.as_str() {
                Some("full") => Ok(GlmIndexer::Full),
                Some("shared") => Ok(GlmIndexer::Shared),
                other => anyhow::bail!("unknown indexer type {other:?}"),
            }).collect::<Result<Vec<_>>>()?,
            None => vec![GlmIndexer::Full; layers],
        };
        ensure!(indexers.len() >= layers && indexers[0] == GlmIndexer::Full,
            "indexer_types must cover every layer and start with a full indexer");
        let first_moe_layer = match v["mlp_layer_types"].as_array() {
            Some(types) => types.iter().position(|t| t == "sparse").context("no sparse layer in mlp_layer_types")?,
            None => int("first_k_dense_replace")?,
        };
        ensure!(v["scoring_func"] == "sigmoid" && v["topk_method"] == "noaux_tc"
            && v["n_group"].as_u64().unwrap_or(1) == 1, "GLM routing must be sigmoid noaux_tc without groups");
        ensure!(v["rope_interleave"].as_bool().unwrap_or(true), "GLM RoPE is interleaved");
        let rope_theta = v["rope_parameters"]["rope_theta"].as_f64()
            .or_else(|| v["rope_theta"].as_f64()).context("glm_moe_dsa config lacks rope_theta")?;
        ensure!(v["rope_parameters"]["rope_type"].as_str().unwrap_or("default") == "default",
            "scaled RoPE is not supported for GLM yet");
        let eos_tokens = match &v["eos_token_id"] {
            Value::Array(ids) => ids.iter().filter_map(|id| id.as_u64().map(|id| id as u32)).collect(),
            id => id.as_u64().map(|id| vec![id as u32]).unwrap_or_default(),
        };
        Ok(Self {
            vocab_size: int("vocab_size")?,
            hidden: int("hidden_size")?,
            layers,
            heads: int("num_attention_heads")?,
            q_lora_rank: int("q_lora_rank")?,
            kv_lora_rank: int("kv_lora_rank")?,
            qk_nope_head_dim: int("qk_nope_head_dim")?,
            qk_rope_head_dim: int("qk_rope_head_dim")?,
            v_head_dim: int("v_head_dim")?,
            index_heads: int("index_n_heads")?,
            index_head_dim: int("index_head_dim")?,
            index_topk: int("index_topk")?,
            indexers: indexers[..layers].to_vec(),
            first_moe_layer,
            dense_intermediate: int("intermediate_size")?,
            experts: int("n_routed_experts")?,
            topk: int("num_experts_per_tok")?,
            moe_intermediate: int("moe_intermediate_size")?,
            shared_experts: int("n_shared_experts")?,
            routed_scale: float("routed_scaling_factor")?,
            rope_theta,
            rms_norm_eps: float("rms_norm_eps")?,
            mtp_layers: v["num_nextn_predict_layers"].as_u64().unwrap_or(0) as usize,
            eos_tokens,
        })
    }

    /// The full layer whose top-k selection `layer` uses.
    pub fn index_source(&self, layer: usize) -> usize {
        (0..=layer).rev().find(|&l| self.indexers[l] == GlmIndexer::Full).unwrap_or(0)
    }

    /// Softmax scale of the MLA attention (qk_head_dim^-1/2).
    pub fn softmax_scale(&self) -> f64 {
        ((self.qk_nope_head_dim + self.qk_rope_head_dim) as f64).powf(-0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm53_shape_and_index_sources() -> Result<()> {
        let mut types: Vec<&str> = vec!["full"; 3];
        for layer in 3..78 {
            types.push(if (layer - 2) % 4 == 0 { "full" } else { "shared" });
        }
        let v = serde_json::json!({
            "model_type": "glm_moe_dsa", "vocab_size": 154880, "hidden_size": 6144, "num_hidden_layers": 78,
            "num_attention_heads": 64, "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64, "v_head_dim": 256, "index_n_heads": 32, "index_head_dim": 128,
            "index_topk": 2048, "indexer_types": types, "first_k_dense_replace": 3, "intermediate_size": 12288,
            "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
            "n_shared_experts": 1, "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
            "topk_method": "noaux_tc", "n_group": 1, "rope_interleave": true, "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
            "num_nextn_predict_layers": 1, "eos_token_id": [154820, 154827, 154829],
        });
        let cfg = GlmDsaConfig::from_hf(&v)?;
        assert_eq!((cfg.first_moe_layer, cfg.topk, cfg.eos_tokens.len()), (3, 8, 3));
        assert_eq!(cfg.index_source(5), 2);
        assert_eq!(cfg.index_source(6), 6);
        assert!((cfg.softmax_scale() - 1.0 / 16.0).abs() < 1e-12);
        Ok(())
    }
}
