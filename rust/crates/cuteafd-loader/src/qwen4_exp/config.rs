//! Qwen 3.8 Flash Next model arguments from the Hugging Face `config.json`
//! (`text_config` of `Qwen4ExpForConditionalGeneration`).
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::plan::checkpoint::read_json;

/// A layer's attention: Gated DeltaNet (linear, recurrent state) or gated GQA
/// with the QSA block indexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen4Attention {
    Gdn,
    Full,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4Config {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub attention: Vec<Qwen4Attention>,
    pub experts: usize,
    pub topk: usize,
    pub moe_intermediate: usize,
    pub shared_intermediate: usize,
    pub rms_norm_eps: f64,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Rotated dims of each full-attention head (head_dim x partial_rotary_factor).
    pub rope_dim: usize,
    pub rope_theta: f64,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_budget: usize,
    pub index_block: usize,
    pub gdn_key_heads: usize,
    pub gdn_value_heads: usize,
    pub gdn_head_dim: usize,
    pub conv_kernel: usize,
    /// Decoder layers that apply PLE before their attention site (ple_layer_ids - 1).
    pub ple_layers: Vec<usize>,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ple_dim: usize,
    pub ple_conv: usize,
    /// The token that separates n-gram segments and pads their history.
    pub eos: u32,
    pub mtp_layers: usize,
}

impl Qwen4Config {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        let v = root.get("text_config").unwrap_or(root);
        ensure!(v["model_type"] == "qwen4_exp_text" || root["model_type"] == "qwen4_exp", "not a qwen4_exp config");
        let int = |value: &Value, key: &str| -> Result<usize> {
            value[key].as_u64().map(|x| x as usize).with_context(|| format!("qwen4_exp config lacks {key}"))
        };
        let layers = int(v, "num_hidden_layers")?;
        let types = v["layer_types"].as_array().context("qwen4_exp config lacks layer_types")?;
        let attention = types.iter().take(layers).map(|t| match t.as_str() {
            Some("linear_attention") => Ok(Qwen4Attention::Gdn),
            Some("full_attention") => Ok(Qwen4Attention::Full),
            other => anyhow::bail!("unknown qwen4_exp layer type {other:?}"),
        }).collect::<Result<Vec<_>>>()?;
        ensure!(attention.len() == layers, "layer_types must cover every layer");
        let rope = &v["rope_parameters"];
        let partial = rope["partial_rotary_factor"].as_f64().or(v["partial_rotary_factor"].as_f64()).unwrap_or(1.0);
        let head_dim = int(v, "head_dim")?;
        ensure!(rope["rope_type"].as_str().unwrap_or("default") == "default", "qwen4_exp RoPE must be the default type");
        ensure!(v["norm_topk_prob"].as_bool().unwrap_or(true), "qwen4_exp routing must renormalize the top-k");
        ensure!(v["output_gate_type"].as_str().unwrap_or("sigmoid") == "sigmoid",
            "the qwen4 programs are built for sigmoid output gates");
        ensure!(v["hidden_act"].as_str().unwrap_or("silu") == "silu", "qwen4_exp experts must use SiLU");
        let eos = match &v["eos_token_id"] {
            Value::Array(ids) => ids.first().and_then(Value::as_u64),
            Value::Number(id) => id.as_u64(),
            _ => None,
        }
        .context("qwen4_exp config lacks eos_token_id (PLE needs it)")? as u32;
        let ple_layers = v["ple_layer_ids"].as_array().map(|ids| {
            ids.iter().filter_map(Value::as_u64).map(|id| id as usize).collect::<Vec<_>>()
        }).unwrap_or_default();
        ensure!(ple_layers.iter().all(|&id| id >= 1 && id <= layers), "ple_layer_ids out of range");
        Ok(Self {
            vocab_size: int(v, "vocab_size")?,
            hidden: int(v, "hidden_size")?,
            layers,
            attention,
            experts: int(v, "num_experts")?,
            topk: int(v, "num_experts_per_tok")?,
            moe_intermediate: int(v, "moe_intermediate_size")?,
            shared_intermediate: int(v, "shared_expert_intermediate_size")?,
            rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            hc_count: int(v, "hc_count")?,
            hc_lowrank: int(v, "hc_lowrank")?,
            heads: int(v, "num_attention_heads")?,
            kv_heads: int(v, "num_key_value_heads")?,
            head_dim,
            rope_dim: (head_dim as f64 * partial) as usize,
            rope_theta: rope["rope_theta"].as_f64().or(v["rope_theta"].as_f64()).context("qwen4_exp lacks rope_theta")?,
            index_heads: int(v, "indexer_n_heads")?,
            index_head_dim: int(v, "indexer_head_dim")?,
            index_budget: int(v, "indexer_budget")?,
            index_block: int(v, "indexer_compress_ratio")?,
            gdn_key_heads: int(v, "linear_num_key_heads")?,
            gdn_value_heads: int(v, "linear_num_value_heads")?,
            gdn_head_dim: int(v, "linear_key_head_dim")?,
            conv_kernel: int(v, "linear_conv_kernel_dim")?,
            ple_layers: ple_layers.into_iter().map(|id| id - 1).collect(),
            ngram_size: int(v, "ngram_size").unwrap_or(3),
            heads_per_ngram: int(v, "heads_per_ngram").unwrap_or(8),
            ple_dim: int(v, "ple_embed_dim").unwrap_or(0),
            ple_conv: int(v, "ple_conv_kernel_size").unwrap_or(4),
            eos,
            mtp_layers: int(v, "mtp_num_hidden_layers").unwrap_or(0),
        })
    }

    /// Visible tokens up to which the QSA indexer selects every token.
    pub fn dense_context(&self) -> usize {
        self.index_budget + self.index_block - 1
    }

    /// The width of all hyper-connection streams of one row.
    pub fn hc_width(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// PLE n-gram rows per token.
    pub fn ple_rows(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Checks the shapes the qwen4 programs are compiled for.
    pub fn check_programs(&self) -> Result<()> {
        ensure!(self.hidden == 2560 && self.hc_count == 4 && self.hc_lowrank == 320,
            "the qwen4 programs are built for hidden 2560 and 4 hyper-connection streams of rank 320");
        ensure!(self.heads == 24 && self.kv_heads == 2 && self.head_dim == 256 && self.rope_dim == 64,
            "the qwen4 programs are built for GQA 24/2 x 256 with 64 rotary dims");
        ensure!(self.index_heads == 4 && self.index_head_dim == 128 && self.index_budget == 2048
            && self.index_block == 4, "the qwen4 programs are built for a 4 x 128 QSA indexer over 4-token blocks");
        ensure!(self.gdn_key_heads == 16 && self.gdn_value_heads == 48 && self.gdn_head_dim == 128
            && self.conv_kernel == 4, "the qwen4 programs are built for GDN 16/48 x 128 with a 4-tap conv");
        ensure!(self.experts == 512 && self.topk == 10 && self.moe_intermediate == 640 && self.shared_intermediate == 640,
            "the qwen4 programs are built for 512 experts top-10 of 640 and a 640 shared expert");
        ensure!(self.ple_layers.len() <= 1 && (self.ple_layers.is_empty() || (self.ple_dim == 2560
            && self.ple_rows() == 16 && self.ple_conv == 4 && self.ngram_size == 3)),
            "the qwen4 PLE program is built for one layer of 16 n-gram rows of 160 and a 4-tap conv at dilation 3");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_flash_next_config() -> Result<()> {
        let types: Vec<&str> = (0..48).map(|l| if l % 4 == 3 { "full_attention" } else { "linear_attention" }).collect();
        let cfg = Qwen4Config::from_hf(&serde_json::json!({
            "model_type": "qwen4_exp", "text_config": {
                "model_type": "qwen4_exp_text", "vocab_size": 248320, "hidden_size": 2560, "num_hidden_layers": 48,
                "layer_types": types, "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640, "rms_norm_eps": 1e-6, "hc_count": 4, "hc_lowrank": 320,
                "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
                "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
                "indexer_n_heads": 4, "indexer_head_dim": 128, "indexer_budget": 2048, "indexer_compress_ratio": 4,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                "linear_conv_kernel_dim": 4, "ple_layer_ids": [2], "ngram_size": 3, "heads_per_ngram": 8,
                "ple_embed_dim": 2560, "ple_conv_kernel_size": 4, "eos_token_id": 248044,
                "output_gate_type": "sigmoid", "mtp_num_hidden_layers": 1}}))?;
        assert_eq!(cfg.attention.iter().filter(|a| **a == Qwen4Attention::Gdn).count(), 36);
        assert_eq!(cfg.attention[3], Qwen4Attention::Full);
        assert_eq!(cfg.ple_layers, vec![1]);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.dense_context(), 2051);
        cfg.check_programs()
    }
}
