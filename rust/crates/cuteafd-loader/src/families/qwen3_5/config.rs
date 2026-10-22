//! Official dense Qwen3.5/3.8 configuration, without fabricated MoE fields.
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::path::Path;
use crate::families::qwen::{GdnGeometry, GdnOutputGate, HybridAttention};
use crate::plan::checkpoint::read_json;

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub attention: Vec<HybridAttention>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub gdn: GdnGeometry,
    pub rope_dim: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub max_context: usize,
    pub mtp_layers: usize,
    pub tied_embeddings: bool,
}

impl Qwen35Config {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        let v = root.get("text_config").unwrap_or(root);
        ensure!(v["model_type"] == "qwen3_5_text", "not a dense qwen3_5_text config");
        let int = |key: &str| -> Result<usize> {
            let n = v[key].as_u64().and_then(|x| usize::try_from(x).ok())
                .with_context(|| format!("qwen3_5 config lacks integer {key}"))?;
            ensure!(n > 0 && n <= 16_777_216, "qwen3_5 {key} is out of range: {n}");
            Ok(n)
        };
        for key in ["attn_output_gate", "mtp_use_dedicated_embeddings", "tie_word_embeddings"] {
            ensure!(v.get(key).is_none_or(Value::is_boolean), "qwen3_5 {key} must be boolean");
        }
        for key in ["mamba_ssm_dtype", "output_gate_type"] {
            ensure!(v.get(key).is_none_or(Value::is_string), "qwen3_5 {key} must be a string");
        }
        ensure!(v.get("attention_dropout").is_none_or(Value::is_number), "attention_dropout must be numeric");
        ensure!(root.get("tie_word_embeddings").is_none_or(Value::is_boolean), "tie_word_embeddings must be boolean");
        let layers = int("num_hidden_layers")?;
        let types = v["layer_types"].as_array().context("qwen3_5 config lacks layer_types")?;
        ensure!(layers == types.len(), "layer_types length differs from num_hidden_layers");
        let attention = types.iter().map(|kind| match kind.as_str() {
            Some("linear_attention") => Ok(HybridAttention::Gdn),
            Some("full_attention") => Ok(HybridAttention::Full),
            _ => anyhow::bail!("unknown qwen3_5 attention type {kind}"),
        }).collect::<Result<Vec<_>>>()?;
        if let Some(interval) = v.get("full_attention_interval") {
            let interval = interval.as_u64().filter(|&n| n > 0).context("invalid full_attention_interval")?;
            ensure!(attention.iter().enumerate().all(|(i, &kind)|
                (kind == HybridAttention::Full) == ((i as u64 + 1) % interval == 0)),
                "layer_types disagrees with full_attention_interval");
        }
        ensure!(v["hidden_act"].as_str() == Some("silu"), "dense Qwen FFN requires SiLU");
        ensure!(v["attention_bias"].as_bool() == Some(false), "dense Qwen requires bias-free attention");
        ensure!(v["attn_output_gate"].as_bool().unwrap_or(true), "dense Qwen requires gated full attention");
        ensure!(v["attention_dropout"].as_f64().unwrap_or(0.0) == 0.0, "attention dropout must be zero");
        ensure!(v["mamba_ssm_dtype"].as_str().unwrap_or("float32") == "float32", "GDN state must be FP32");
        // Transformers 5.13.0 Qwen3_5Attention ignores this extra field and uses
        // sigmoid. Qwen3_5RMSNormGated uses FP32 SiLU. Do not switch the GQA gate.
        ensure!(matches!(v["output_gate_type"].as_str().unwrap_or("swish"), "swish" | "silu"),
            "unreconciled dense Qwen output_gate_type {}", v["output_gate_type"]);
        let heads = int("num_attention_heads")?;
        let kv_heads = int("num_key_value_heads")?;
        ensure!(heads % kv_heads == 0, "GQA heads must divide by KV heads");
        let gdn = GdnGeometry {
            key_heads: int("linear_num_key_heads")?, value_heads: int("linear_num_value_heads")?,
            key_dim: int("linear_key_head_dim")?, value_dim: int("linear_value_head_dim")?,
            conv_kernel: int("linear_conv_kernel_dim")?, output_gate: GdnOutputGate::Silu,
        };
        ensure!(gdn.value_heads % gdn.key_heads == 0, "GDN value heads must divide by key heads");
        ensure!(gdn.key_dim == 128 && gdn.value_dim == 128 && gdn.conv_kernel == 4
            && gdn.key_heads <= 1024 && gdn.value_heads <= 1024 && heads <= 1024
            && int("head_dim")? == 256, "unsupported dense Qwen GDN/GQA geometry");
        let rope = &v["rope_parameters"];
        ensure!(rope["rope_type"].as_str() == Some("default"), "unsupported dense Qwen RoPE type");
        ensure!(rope["mrope_section"] == serde_json::json!([11, 11, 10])
            && rope["mrope_interleaved"].as_bool() == Some(true), "unsupported dense Qwen M-RoPE layout");
        let head_dim = int("head_dim")?;
        let partial = rope["partial_rotary_factor"].as_f64().context("missing partial_rotary_factor")?;
        let rope_width = head_dim as f64 * partial;
        ensure!(rope_width.is_finite() && rope_width > 0.0 && rope_width <= head_dim as f64
            && rope_width.fract() == 0.0 && rope_width as usize == 64, "unsupported dense Qwen rotary width");
        let rope_theta = rope["rope_theta"].as_f64().context("missing rope_theta")?;
        let rms_norm_eps = v["rms_norm_eps"].as_f64().context("missing rms_norm_eps")?;
        ensure!(rope_theta.is_finite() && rope_theta > 0.0 && rms_norm_eps.is_finite() && rms_norm_eps > 0.0,
            "invalid rope_theta or rms_norm_eps");
        let mtp_layers = match v.get("mtp_num_hidden_layers") {
            None => 0,
            Some(value) => value.as_u64().and_then(|x| usize::try_from(x).ok())
                .context("mtp_num_hidden_layers must be a nonnegative integer")?,
        };
        ensure!(mtp_layers <= 1 && !v["mtp_use_dedicated_embeddings"].as_bool().unwrap_or(false),
            "only one shared-embedding dense MTP layer is described");
        Ok(Self { hidden: int("hidden_size")?, intermediate: int("intermediate_size")?, vocab: int("vocab_size")?,
            attention, heads, kv_heads, head_dim, gdn, rope_dim: rope_width as usize, rope_theta, rms_norm_eps,
            max_context: int("max_position_embeddings")?, mtp_layers,
            tied_embeddings: v["tie_word_embeddings"].as_bool().or(root["tie_word_embeddings"].as_bool()).unwrap_or(false) })
    }

    pub fn full_kv_bytes_per_token(&self, format: crate::families::qwen4::Qwen4KvCache) -> usize {
        self.attention.iter().filter(|&&a| a == HybridAttention::Full).count() * format.record_bytes(self.kv_heads, self.head_dim)
    }

    pub fn recurrent_state_bytes(&self) -> usize {
        self.attention.iter().filter(|&&a| a == HybridAttention::Gdn).count() * self.gdn.state_bytes()
    }
}
