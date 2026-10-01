//! MiMo V2 (mimo_v2_flash) and V2.6 Pro (mimo_v2) model arguments from the
//! Hugging Face `config.json`.
//!
//! The hub config uses the checkpoint's own keys (`hybrid_layer_pattern`,
//! `moe_layer_freq`, `swa_*`, `layernorm_epsilon`); transformers' native
//! config uses `layer_types` / `mlp_layer_types` and doubles the full layers'
//! KV heads on SWA layers. Both spellings are read; each present pattern must
//! cover exactly every layer, and two present spellings must agree. This is
//! the one reader: serve-mimo and `cuteafd plan` both derive from it.
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

/// One per-layer pattern in either spelling: each spelling present must be a
/// list covering exactly `layers` entries of known values, and when both are
/// present they must agree. `None` when neither is present.
fn pattern<T: PartialEq + Copy + std::fmt::Debug>(v: &Value, layers: usize, spellings: [(&str, &dyn Fn(&Value) -> Option<T>); 2])
    -> Result<Option<Vec<T>>> {
    let mut found: Option<(&str, Vec<T>)> = None;
    for (key, decode) in spellings {
        let Some(raw) = v.get(key).filter(|x| !x.is_null()) else { continue };
        let list = raw.as_array().with_context(|| format!("{key} must be a list of {layers} entries, found {raw}"))?;
        ensure!(list.len() == layers, "{key} has {} entries for {layers} layers", list.len());
        let values = list.iter().enumerate()
            .map(|(layer, x)| decode(x).with_context(|| format!("{key}[{layer}] = {x} is not a known entry")))
            .collect::<Result<Vec<T>>>()?;
        if let Some((other, previous)) = &found {
            ensure!(*previous == values, "{other} and {key} disagree");
        }
        found = Some((key, values));
    }
    Ok(found.map(|(_, values)| values))
}

/// `hybrid_layer_pattern` (0 full, 1 sliding) or `layer_types`; default: full
/// attention on layer 0 and every sixth layer.
fn attention_pattern(v: &Value, layers: usize) -> Result<Vec<MimoAttention>> {
    let hybrid = |x: &Value| match x.as_u64() {
        Some(0) => Some(MimoAttention::Full),
        Some(1) => Some(MimoAttention::Sliding),
        _ => None,
    };
    let types = |x: &Value| match x.as_str() {
        Some("full_attention") => Some(MimoAttention::Full),
        Some("sliding_attention") => Some(MimoAttention::Sliding),
        _ => None,
    };
    Ok(pattern(v, layers, [("hybrid_layer_pattern", &hybrid), ("layer_types", &types)])?.unwrap_or_else(|| {
        (0..layers).map(|l| if l == 0 || (l + 1) % 6 == 0 { MimoAttention::Full } else { MimoAttention::Sliding }).collect()
    }))
}

/// `dense[layer]` from `moe_layer_freq` (0 dense, 1 MoE) or `mlp_layer_types`
/// ("dense" / "sparse"); default: layer 0 dense.
fn dense_pattern(v: &Value, layers: usize) -> Result<Vec<bool>> {
    let freq = |x: &Value| match x.as_u64() {
        Some(0) => Some(true),
        Some(1) => Some(false),
        _ => None,
    };
    let types = |x: &Value| match x.as_str() {
        Some("dense") => Some(true),
        Some("sparse") => Some(false),
        _ => None,
    };
    Ok(pattern(v, layers, [("moe_layer_freq", &freq), ("mlp_layer_types", &types)])?
        .unwrap_or_else(|| (0..layers).map(|l| l == 0).collect()))
}

/// `sliding_window_size` or `sliding_window` (equal when both are present).
fn window(v: &Value) -> Result<usize> {
    let read = |key: &str| v.get(key).filter(|x| !x.is_null()).map(|x| {
        x.as_u64().map(|x| x as usize).with_context(|| format!("{key} must be an unsigned integer, found {x}"))
    }).transpose();
    match (read("sliding_window_size")?, read("sliding_window")?) {
        (Some(a), Some(b)) => {
            ensure!(a == b, "sliding_window_size {a} and sliding_window {b} disagree");
            Ok(a)
        }
        (Some(w), None) | (None, Some(w)) => Ok(w),
        (None, None) => anyhow::bail!("mimo_v2 config lacks sliding_window_size / sliding_window"),
    }
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
        let attention = attention_pattern(v, layers)?;
        let dense = dense_pattern(v, layers)?;
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
            window: window(v)?,
            attention,
            full_sinks: v["add_full_attention_sink_bias"].as_bool().unwrap_or(false),
            swa_sinks: v["add_swa_attention_sink_bias"].as_bool().unwrap_or(true),
            dense,
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
            // One GPU of V2.6 Pro's two-GPU head split (`head_split(2)`).
            (6144, 64, 4, 4, 384) => Ok("mimop2"),
            other => anyhow::bail!("no mimo program geometry for (hidden, heads, full KV, SWA KV, experts) {other:?}: \
                add a MiMoGeometry to b12x.integration.cuteafd._common and an exporter entry"),
        }
    }

    /// Rows between key heads in the coordinator's qkv layout: 192, or 256 for
    /// `mimop` (keys zero-padded to whole 128-row blocks; see `FusedQkvLayout`).
    pub fn qkv_key_stride(&self) -> usize {
        if matches!(self.program_family().ok(), Some("mimop" | "mimop2")) { 256 } else { self.head_dim }
    }

    /// One GPU's share of a head split over `ranks` GPUs: `heads / ranks` query
    /// heads with the KV heads they read (`kv / ranks`, partitioned, not
    /// replicated) and `dense_intermediate / ranks` of the dense MLP. Rank `r`
    /// owns query heads `r * heads / ranks ..`, KV heads `r * kv / ranks ..` (GQA
    /// groups stay whole) and the matching o_proj columns.
    pub fn head_split(&self, ranks: usize) -> Result<Self> {
        ensure!(ranks > 0 && self.heads % ranks == 0 && self.full_kv_heads % ranks == 0
            && self.swa_kv_heads % ranks == 0 && self.dense_intermediate % (ranks * 128) == 0,
            "{} query / {}+{} KV heads and a {}-wide dense MLP do not split over {ranks} GPUs", self.heads,
            self.full_kv_heads, self.swa_kv_heads, self.dense_intermediate);
        Ok(Self { heads: self.heads / ranks, full_kv_heads: self.full_kv_heads / ranks,
            swa_kv_heads: self.swa_kv_heads / ranks, dense_intermediate: self.dense_intermediate / ranks, ..self.clone() })
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

    fn two_layers() -> serde_json::Value {
        serde_json::json!({
            "model_type": "mimo_v2_flash", "vocab_size": 64, "hidden_size": 4096, "num_hidden_layers": 2,
            "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 192, "v_head_dim": 128,
            "rope_theta": 5000000, "swa_rope_theta": 10000, "sliding_window": 128, "intermediate_size": 16384,
            "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
        })
    }

    #[test]
    fn both_spellings_read_alike_and_partial_patterns_are_refused() -> Result<()> {
        let mut hub = two_layers();
        hub["hybrid_layer_pattern"] = serde_json::json!([0, 1]);
        hub["moe_layer_freq"] = serde_json::json!([0, 1]);
        let mut hf = two_layers();
        hf["layer_types"] = serde_json::json!(["full_attention", "sliding_attention"]);
        hf["mlp_layer_types"] = serde_json::json!(["dense", "sparse"]);
        let mut both = hub.clone();
        both["layer_types"] = hf["layer_types"].clone();
        let (a, b, c) = (MimoV2Config::from_hf(&hub)?, MimoV2Config::from_hf(&hf)?, MimoV2Config::from_hf(&both)?);
        assert!(a == b && b == c && a.swa_sinks && !a.full_sinks);
        assert_eq!((a.attention.clone(), a.dense.clone()), (vec![MimoAttention::Full, MimoAttention::Sliding], vec![true, false]));
        // The defaults equal the hub pattern for these two layers.
        assert_eq!(MimoV2Config::from_hf(&two_layers())?, a);
        for (key, value) in [("moe_layer_freq", serde_json::json!([0])), ("moe_layer_freq", serde_json::json!(1)),
            ("layer_types", serde_json::json!(["full_attention", "full_attention"])),
            ("hybrid_layer_pattern", serde_json::json!([0, 2])), ("sliding_window_size", serde_json::json!(64))] {
            let mut config = hub.clone();
            config[key] = value;
            assert!(MimoV2Config::from_hf(&config).is_err(), "{key}");
        }
        Ok(())
    }
}
