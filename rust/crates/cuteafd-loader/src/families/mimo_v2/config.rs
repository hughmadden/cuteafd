//! MiMo V2 (mimo_v2_flash) and V2.6 Pro (mimo_v2) model arguments from the
//! Hugging Face `config.json`.
//!
//! The hub config uses the checkpoint's own keys (`hybrid_layer_pattern`,
//! `moe_layer_freq`, `swa_*`, `layernorm_epsilon`); transformers' native
//! config uses `layer_types` / `mlp_layer_types` and doubles the full layers'
//! KV heads on SWA layers. Both spellings are read.
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::plan::checkpoint::read_json;

/// A layer's attention: full causal GQA or sliding-window GQA with sinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoAttention {
    Full,
    Sliding,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MimoV2Config {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub full_kv_heads: usize,
    pub swa_kv_heads: usize,
    pub head_dim: usize,
    pub v_head_dim: usize,
    /// Rotated leading dims of each query/key head (`int(head_dim * partial_rotary_factor)`).
    pub rope_dim: usize,
    pub full_rope_theta: f64,
    pub swa_rope_theta: f64,
    pub window: usize,
    pub attention: Vec<MimoAttention>,
    pub full_sinks: bool,
    pub swa_sinks: bool,
    /// `dense[layer]`: a SwiGLU MLP instead of the MoE.
    pub dense: Vec<bool>,
    pub dense_intermediate: usize,
    pub experts: usize,
    pub topk: usize,
    pub moe_intermediate: usize,
    pub routed_scale: f64,
    pub rms_norm_eps: f64,
    pub v_scale: f64,
}

impl MimoV2Config {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(v: &Value) -> Result<Self> {
        let v = v.get("text_config").unwrap_or(v);
        ensure!(v["model_type"] == "mimo_v2_flash" || v["model_type"] == "mimo_v2", "not a mimo_v2 config");
        let int = |key: &str| -> Result<usize> {
            v[key].as_u64().map(|x| x as usize).with_context(|| format!("mimo_v2 config lacks {key}"))
        };
        let layers = int("num_hidden_layers")?;
        let attention = if let Some(pattern) = v["hybrid_layer_pattern"].as_array() {
            pattern.iter().map(|p| match p.as_u64() {
                Some(0) => Ok(MimoAttention::Full),
                Some(1) => Ok(MimoAttention::Sliding),
                other => anyhow::bail!("unknown hybrid_layer_pattern entry {other:?}"),
            }).collect::<Result<Vec<_>>>()?
        } else if let Some(types) = v["layer_types"].as_array() {
            types.iter().map(|t| match t.as_str() {
                Some("full_attention") => Ok(MimoAttention::Full),
                Some("sliding_attention") => Ok(MimoAttention::Sliding),
                other => anyhow::bail!("unknown layer type {other:?}"),
            }).collect::<Result<Vec<_>>>()?
        } else {
            (0..layers).map(|l| if l == 0 || (l + 1) % 6 == 0 { MimoAttention::Full } else { MimoAttention::Sliding })
                .collect()
        };
        ensure!(attention.len() >= layers, "the layer pattern must cover every layer");
        let dense = if let Some(freq) = v["moe_layer_freq"].as_array() {
            freq.iter().map(|f| f.as_u64() == Some(0)).collect::<Vec<_>>()
        } else if let Some(types) = v["mlp_layer_types"].as_array() {
            types.iter().map(|t| t == "dense").collect()
        } else {
            (0..layers).map(|l| l == 0).collect()
        };
        ensure!(dense.len() >= layers, "the MLP pattern must cover every layer");
        ensure!(v["scoring_func"].as_str().unwrap_or("sigmoid") == "sigmoid"
            && v["topk_method"].as_str().unwrap_or("noaux_tc") == "noaux_tc"
            && v["n_group"].as_u64().unwrap_or(1) == 1 && v["norm_topk_prob"].as_bool().unwrap_or(true),
            "MiMo routing must be normalized sigmoid noaux_tc without groups");
        let full_kv_heads = int("num_key_value_heads")?;
        let head_dim = int("head_dim")?;
        let partial = v["partial_rotary_factor"].as_f64()
            .or_else(|| v["rope_parameters"]["full_attention"]["partial_rotary_factor"].as_f64())
            .unwrap_or(0.334);
        let full_rope_theta = v["rope_theta"].as_f64()
            .or_else(|| v["rope_parameters"]["full_attention"]["rope_theta"].as_f64())
            .context("mimo_v2 config lacks rope_theta")?;
        let swa_rope_theta = v["swa_rope_theta"].as_f64()
            .or_else(|| v["rope_parameters"]["sliding_attention"]["rope_theta"].as_f64())
            .context("mimo_v2 config lacks swa_rope_theta")?;
        ensure!(v.get("rope_scaling").map_or(true, Value::is_null), "scaled RoPE is not supported for MiMo yet");
        ensure!(int("swa_head_dim").unwrap_or(head_dim) == head_dim
            && int("swa_v_head_dim").unwrap_or(int("v_head_dim")?) == int("v_head_dim")?
            && int("swa_num_attention_heads").unwrap_or(int("num_attention_heads")?) == int("num_attention_heads")?,
            "SWA and full layers must share head counts and dims");
        Ok(Self {
            vocab_size: int("vocab_size")?,
            hidden: int("hidden_size")?,
            layers,
            heads: int("num_attention_heads")?,
            full_kv_heads,
            swa_kv_heads: int("swa_num_key_value_heads").unwrap_or(2 * full_kv_heads),
            head_dim,
            v_head_dim: int("v_head_dim")?,
            rope_dim: (head_dim as f64 * partial) as usize,
            full_rope_theta,
            swa_rope_theta,
            window: int("sliding_window_size").or_else(|_| int("sliding_window"))?,
            attention: attention[..layers].to_vec(),
            full_sinks: v["add_full_attention_sink_bias"].as_bool().unwrap_or(false),
            swa_sinks: v["add_swa_attention_sink_bias"].as_bool().unwrap_or(true),
            dense: dense[..layers].to_vec(),
            dense_intermediate: int("intermediate_size")?,
            experts: int("n_routed_experts")?,
            topk: int("num_experts_per_tok")?,
            moe_intermediate: int("moe_intermediate_size")?,
            routed_scale: v["routed_scaling_factor"].as_f64().unwrap_or(1.0),
            rms_norm_eps: v["layernorm_epsilon"].as_f64().or_else(|| v["rms_norm_eps"].as_f64()).unwrap_or(1e-5),
            v_scale: v["attention_value_scale"].as_f64().unwrap_or(1.0),
        })
    }

    /// The coordinator program family built for this attention geometry
    /// (`b12x.integration.cuteafd` MiMoGeometry): `mimo` (V2 Flash: hidden 4096,
    /// 64 heads, 4/8 KV heads) or `mimop` (V2.6 Pro: hidden 6144, 128 heads, 8/8).
    pub fn program_family(&self) -> Result<&'static str> {
        match (self.hidden, self.heads, self.full_kv_heads, self.swa_kv_heads, self.experts) {
            (4096, 64, 4, 8, 256) => Ok("mimo"),
            (6144, 128, 8, 8, 384) => Ok("mimop"),
            other => anyhow::bail!("no mimo program geometry for (hidden, heads, full KV, SWA KV, experts) {other:?}: \
                add a MiMoGeometry to b12x.integration.cuteafd._common and an exporter entry"),
        }
    }

    /// Rows between key heads in the coordinator's qkv layout: 192, or 256 for
    /// `mimop` (keys zero-padded to whole 128-row blocks; see `FusedQkvLayout`).
    pub fn qkv_key_stride(&self) -> usize {
        if self.program_family().ok() == Some("mimop") { 256 } else { self.head_dim }
    }

    pub fn kv_heads(&self, attention: MimoAttention) -> usize {
        match attention {
            MimoAttention::Full => self.full_kv_heads,
            MimoAttention::Sliding => self.swa_kv_heads,
        }
    }

    /// BF16 elements of one token's KV record (keys then values of every KV head).
    pub fn record_elems(&self, attention: MimoAttention) -> usize {
        self.kv_heads(attention) * (self.head_dim + self.v_head_dim)
    }

    pub fn rope_theta(&self, attention: MimoAttention) -> f64 {
        match attention {
            MimoAttention::Full => self.full_rope_theta,
            MimoAttention::Sliding => self.swa_rope_theta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mimo_v2_flash_hub_config() -> Result<()> {
        let pattern: Vec<u64> = (0..48).map(|l| u64::from(!(l == 0 || (l + 1) % 6 == 0))).collect();
        let freq: Vec<u64> = (0..48).map(|l| u64::from(l != 0)).collect();
        let v = serde_json::json!({
            "model_type": "mimo_v2_flash", "vocab_size": 152576, "hidden_size": 4096, "num_hidden_layers": 48,
            "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 192, "v_head_dim": 128,
            "swa_num_attention_heads": 64, "swa_num_key_value_heads": 8, "swa_head_dim": 192,
            "swa_v_head_dim": 128, "partial_rotary_factor": 0.334, "rope_theta": 5000000,
            "swa_rope_theta": 10000, "sliding_window": 128, "sliding_window_size": 128,
            "hybrid_layer_pattern": pattern, "moe_layer_freq": freq, "add_swa_attention_sink_bias": true,
            "add_full_attention_sink_bias": false, "intermediate_size": 16384, "n_routed_experts": 256,
            "num_experts_per_tok": 8, "moe_intermediate_size": 2048, "routed_scaling_factor": null,
            "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 1, "norm_topk_prob": true,
            "layernorm_epsilon": 1e-5, "attention_value_scale": 0.707,
        });
        let cfg = MimoV2Config::from_hf(&v)?;
        assert_eq!((cfg.rope_dim, cfg.swa_kv_heads, cfg.window), (64, 8, 128));
        assert_eq!(cfg.attention.iter().filter(|a| **a == MimoAttention::Full).count(), 9);
        assert_eq!(cfg.attention[5], MimoAttention::Full);
        assert_eq!(cfg.dense.iter().filter(|d| **d).count(), 1);
        assert_eq!((cfg.record_elems(MimoAttention::Full), cfg.record_elems(MimoAttention::Sliding)), (1280, 2560));
        assert!((cfg.routed_scale - 1.0).abs() < 1e-12 && !cfg.full_sinks && cfg.swa_sinks);
        Ok(())
    }

    #[test]
    fn mimo_v26_pro_hub_config() -> Result<()> {
        let pattern: Vec<u64> = (0..70).map(|l| u64::from(![0, 7, 15, 23, 31, 39, 47, 55, 62, 69].contains(&l))).collect();
        let freq: Vec<u64> = (0..70).map(|l| u64::from(l != 0)).collect();
        let v = serde_json::json!({
            "model_type": "mimo_v2", "vocab_size": 152576, "hidden_size": 6144, "num_hidden_layers": 70,
            "num_attention_heads": 128, "num_key_value_heads": 8, "head_dim": 192, "v_head_dim": 128,
            "swa_num_attention_heads": 128, "swa_num_key_value_heads": 8, "swa_head_dim": 192,
            "swa_v_head_dim": 128, "partial_rotary_factor": 0.334, "rope_theta": 10000000,
            "swa_rope_theta": 10000, "sliding_window": 128, "sliding_window_size": 128,
            "hybrid_layer_pattern": pattern, "moe_layer_freq": freq, "add_swa_attention_sink_bias": true,
            "add_full_attention_sink_bias": false, "intermediate_size": 16384, "n_routed_experts": 384,
            "num_experts_per_tok": 8, "moe_intermediate_size": 2048, "routed_scaling_factor": null,
            "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 1, "norm_topk_prob": true,
            "layernorm_epsilon": 1e-5, "attention_value_scale": 0.612,
        });
        let cfg = MimoV2Config::from_hf(&v)?;
        assert_eq!((cfg.rope_dim, cfg.full_kv_heads, cfg.swa_kv_heads), (64, 8, 8));
        assert_eq!(cfg.attention.iter().filter(|a| **a == MimoAttention::Full).count(), 10);
        assert_eq!(cfg.record_elems(MimoAttention::Full), 2560);
        assert_eq!(cfg.program_family()?, "mimop");
        Ok(())
    }
}
