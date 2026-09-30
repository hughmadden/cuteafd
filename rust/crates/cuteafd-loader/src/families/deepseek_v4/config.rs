//! DeepSeek V4 (Flash 0731, Pro 0813) model arguments.
//!
//! Field names follow the official inference `ModelArgs`. A snapshot's
//! `inference/config.json` uses those keys directly; publications that only
//! ship the Hugging Face `config.json` are mapped key by key. The number of
//! dSpark stages is taken from the checkpoint, because the HF key
//! `num_nextn_predict_layers` says 1 while every V4 checkpoint carries three.
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::plan::checkpoint::read_json;

#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekV4Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub moe_inter_dim: usize,
    pub n_layers: usize,
    pub n_hash_layers: usize,
    pub n_mtp_layers: usize,
    pub n_heads: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub n_activated_experts: usize,
    pub score_func: String,
    pub route_scale: f64,
    pub swiglu_limit: f64,
    pub q_lora_rank: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub norm_eps: f64,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    /// One entry per backbone layer (trailing MTP entries removed).
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: f64,
    pub original_seq_len: usize,
    pub rope_theta: f64,
    pub rope_factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f64,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: usize,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: usize,
}

/// How one backbone layer attends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V4LayerAttention {
    /// Sliding window only.
    Window,
    /// Overlapping ratio-4 compression with its own indexer (top-k).
    Indexed4,
    /// Ratio-128 compression; attends every completed compressed entry.
    Dense128,
}

impl DeepseekV4Config {
    pub fn read(snapshot: &Path, mtp_stages: usize) -> Result<Self> {
        let inference = snapshot.join("inference/config.json");
        if inference.is_file() {
            Self::from_model_args(&read_json(&inference)?, mtp_stages)
        } else {
            Self::from_hf(&read_json(&snapshot.join("config.json"))?, mtp_stages)
        }
    }

    pub fn from_model_args(v: &Value, mtp_stages: usize) -> Result<Self> {
        let get = |k: &str| field(v, k);
        Self::build(
            |k| get(k),
            v.get("norm_eps").and_then(Value::as_f64).unwrap_or(1e-6),
            v.get("hc_eps").and_then(Value::as_f64).unwrap_or(1e-6),
            mtp_stages,
        )
    }

    pub fn from_hf(v: &Value, mtp_stages: usize) -> Result<Self> {
        ensure!(
            v.get("model_type").and_then(Value::as_str) == Some("deepseek_v4"),
            "not a deepseek_v4 config"
        );
        let rope = v.get("rope_scaling").cloned().unwrap_or(Value::Null);
        let map = |k: &str| -> Result<Value> {
            let hf = match k {
                "dim" => "hidden_size",
                "moe_inter_dim" => "moe_intermediate_size",
                "n_layers" => "num_hidden_layers",
                "n_hash_layers" => "num_hash_layers",
                "n_heads" => "num_attention_heads",
                "n_activated_experts" => "num_experts_per_tok",
                "score_func" => "scoring_func",
                "route_scale" => "routed_scaling_factor",
                "rope_head_dim" => "qk_rope_head_dim",
                "window_size" => "sliding_window",
                "original_seq_len" => return field(&rope, "original_max_position_embeddings"),
                "rope_factor" => return field(&rope, "factor"),
                "beta_fast" | "beta_slow" => return field(&rope, k),
                other => other,
            };
            field(v, hf)
        };
        Self::build(
            map,
            v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6),
            v.get("hc_eps").and_then(Value::as_f64).unwrap_or(1e-6),
            mtp_stages,
        )
    }

    fn build(get: impl Fn(&str) -> Result<Value>, norm_eps: f64, hc_eps: f64, mtp_stages: usize) -> Result<Self> {
        let u = |k: &str| -> Result<usize> {
            get(k)?.as_u64().map(|x| x as usize).with_context(|| format!("{k} is not an unsigned integer"))
        };
        let f = |k: &str| -> Result<f64> { get(k)?.as_f64().with_context(|| format!("{k} is not a number")) };
        let list = |k: &str| -> Result<Vec<usize>> {
            get(k)?
                .as_array()
                .with_context(|| format!("{k} is not a list"))?
                .iter()
                .map(|x| x.as_u64().map(|x| x as usize).with_context(|| format!("{k} has a non-integer")))
                .collect()
        };
        let n_layers = u("n_layers")?;
        let mut ratios = list("compress_ratios")?;
        ensure!(ratios.len() >= n_layers, "compress_ratios has {} entries for {n_layers} layers", ratios.len());
        ratios.truncate(n_layers);
        if let Some(bad) = ratios.iter().find(|r| !matches!(r, 0 | 4 | 128)) {
            bail!("unsupported V4 compress ratio {bad} (expected 0, 4 or 128)");
        }
        let config = Self {
            vocab_size: u("vocab_size")?,
            dim: u("dim")?,
            moe_inter_dim: u("moe_inter_dim")?,
            n_layers,
            n_hash_layers: u("n_hash_layers").unwrap_or(0),
            n_mtp_layers: mtp_stages,
            n_heads: u("n_heads")?,
            n_routed_experts: u("n_routed_experts")?,
            n_shared_experts: u("n_shared_experts")?,
            n_activated_experts: u("n_activated_experts")?,
            score_func: get("score_func")?.as_str().context("score_func")?.to_owned(),
            route_scale: f("route_scale")?,
            swiglu_limit: f("swiglu_limit")?,
            q_lora_rank: u("q_lora_rank")?,
            head_dim: u("head_dim")?,
            rope_head_dim: u("rope_head_dim")?,
            norm_eps,
            o_groups: u("o_groups")?,
            o_lora_rank: u("o_lora_rank")?,
            window_size: u("window_size")?,
            compress_ratios: ratios,
            compress_rope_theta: f("compress_rope_theta")?,
            original_seq_len: u("original_seq_len")?,
            rope_theta: f("rope_theta")?,
            rope_factor: f("rope_factor")?,
            beta_fast: f("beta_fast")?,
            beta_slow: f("beta_slow")?,
            index_n_heads: u("index_n_heads")?,
            index_head_dim: u("index_head_dim")?,
            index_topk: u("index_topk")?,
            hc_mult: u("hc_mult")?,
            hc_sinkhorn_iters: u("hc_sinkhorn_iters")?,
            hc_eps,
            dspark_block_size: u("dspark_block_size").unwrap_or(0),
            dspark_noise_token_id: u("dspark_noise_token_id").unwrap_or(0),
            dspark_target_layer_ids: list("dspark_target_layer_ids").unwrap_or_default(),
            dspark_markov_rank: u("dspark_markov_rank").unwrap_or(0),
        };
        ensure!(config.score_func == "sqrtsoftplus", "unsupported V4 score_func {}", config.score_func);
        ensure!(config.head_dim > config.rope_head_dim, "head_dim must exceed rope_head_dim");
        ensure!(config.n_heads % config.o_groups == 0, "n_heads must divide into o_groups");
        Ok(config)
    }

    pub fn layer_attention(&self, layer: usize) -> V4LayerAttention {
        match self.compress_ratios[layer] {
            0 => V4LayerAttention::Window,
            4 => V4LayerAttention::Indexed4,
            _ => V4LayerAttention::Dense128,
        }
    }

    pub fn is_hash_layer(&self, layer: usize) -> bool {
        layer < self.n_hash_layers
    }

    /// Latent dimensions stored as FP8 (the rest is the BF16 RoPE tail).
    pub fn nope_dim(&self) -> usize {
        self.head_dim - self.rope_head_dim
    }
}

fn field(v: &Value, key: &str) -> Result<Value> {
    v.get(key).cloned().with_context(|| format!("config has no {key}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object(pairs: &[(&str, Value)]) -> Value {
        Value::Object(pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect())
    }

    fn flash_model_args() -> Value {
        let mut ratios = vec![0u64, 0];
        ratios.extend((2..43).map(|i| if i % 2 == 0 { 4 } else { 128 }));
        ratios.extend([0, 0, 0]);
        object(&[
            ("vocab_size", json!(129280)), ("dim", json!(4096)), ("moe_inter_dim", json!(2048)),
            ("n_layers", json!(43)), ("n_hash_layers", json!(3)), ("n_mtp_layers", json!(3)),
            ("dspark_block_size", json!(5)), ("dspark_noise_token_id", json!(128799)),
            ("dspark_target_layer_ids", json!([40, 41, 42])), ("dspark_markov_rank", json!(256)),
            ("n_heads", json!(64)), ("n_routed_experts", json!(256)), ("n_shared_experts", json!(1)),
            ("n_activated_experts", json!(6)), ("score_func", json!("sqrtsoftplus")),
            ("route_scale", json!(1.5)), ("swiglu_limit", json!(10.0)), ("q_lora_rank", json!(1024)),
            ("head_dim", json!(512)), ("rope_head_dim", json!(64)), ("o_groups", json!(8)),
            ("o_lora_rank", json!(1024)), ("window_size", json!(128)), ("original_seq_len", json!(65536)),
            ("rope_theta", json!(10000)), ("rope_factor", json!(16)), ("beta_fast", json!(32)),
            ("beta_slow", json!(1)), ("index_n_heads", json!(64)), ("index_head_dim", json!(128)),
            ("index_topk", json!(512)), ("hc_mult", json!(4)), ("hc_sinkhorn_iters", json!(20)),
            ("compress_rope_theta", json!(160000)), ("compress_ratios", json!(ratios)),
        ])
    }

    #[test]
    fn flash_model_args_describe_the_layer_schedule() {
        let c = DeepseekV4Config::from_model_args(&flash_model_args(), 3).unwrap();
        assert_eq!(c.compress_ratios.len(), 43);
        assert_eq!(c.layer_attention(0), V4LayerAttention::Window);
        assert_eq!(c.layer_attention(2), V4LayerAttention::Indexed4);
        assert_eq!(c.layer_attention(3), V4LayerAttention::Dense128);
        assert_eq!(c.layer_attention(42), V4LayerAttention::Indexed4);
        assert!(c.is_hash_layer(2) && !c.is_hash_layer(3));
        assert_eq!(c.nope_dim(), 448);
        assert_eq!(c.norm_eps, 1e-6);
    }

    #[test]
    fn pro_hf_config_maps_to_model_args() {
        let ratios: Vec<u64> = (0..64).map(|i| if i < 2 { 128 } else if i >= 61 { 0 } else if i % 2 == 0 { 4 } else { 128 }).collect();
        let hf = object(&[
            ("model_type", json!("deepseek_v4")), ("vocab_size", json!(129280)), ("hidden_size", json!(7168)),
            ("moe_intermediate_size", json!(3072)), ("num_hidden_layers", json!(61)), ("num_hash_layers", json!(3)),
            ("num_attention_heads", json!(128)), ("n_routed_experts", json!(384)), ("n_shared_experts", json!(1)),
            ("num_experts_per_tok", json!(6)), ("scoring_func", json!("sqrtsoftplus")),
            ("routed_scaling_factor", json!(2.5)), ("swiglu_limit", json!(10.0)), ("q_lora_rank", json!(1536)),
            ("head_dim", json!(512)), ("qk_rope_head_dim", json!(64)), ("o_groups", json!(16)),
            ("o_lora_rank", json!(1024)), ("sliding_window", json!(128)), ("rope_theta", json!(10000)),
            ("compress_rope_theta", json!(160000)),
            ("rope_scaling", json!({"beta_fast": 32, "beta_slow": 1, "factor": 16, "original_max_position_embeddings": 65536})),
            ("index_n_heads", json!(64)), ("index_head_dim", json!(128)), ("index_topk", json!(1024)),
            ("hc_mult", json!(4)), ("hc_sinkhorn_iters", json!(20)), ("hc_eps", json!(1e-6)),
            ("rms_norm_eps", json!(1e-6)), ("dspark_block_size", json!(5)), ("dspark_markov_rank", json!(512)),
            ("dspark_noise_token_id", json!(128799)), ("dspark_target_layer_ids", json!([58, 59, 60])),
            ("compress_ratios", json!(ratios)),
        ]);
        let c = DeepseekV4Config::from_hf(&hf, 3).unwrap();
        assert_eq!((c.dim, c.n_heads, c.o_groups, c.q_lora_rank), (7168, 128, 16, 1536));
        assert_eq!(c.route_scale, 2.5);
        assert_eq!(c.layer_attention(0), V4LayerAttention::Dense128);
        assert_eq!(c.n_mtp_layers, 3);
        assert_eq!(c.original_seq_len, 65536);
    }

    #[test]
    fn unknown_ratio_is_rejected() {
        let mut v = flash_model_args();
        v["compress_ratios"][5] = json!(2);
        assert!(DeepseekV4Config::from_model_args(&v, 3).is_err());
    }
}
