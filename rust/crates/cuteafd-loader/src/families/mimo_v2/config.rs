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

/// Storage format of the full-attention KV records (the paged pool shared by sequences and
/// prefix snapshots). SWA rings and steps stay BF16: they hold 256 rows per sequence, and
/// widening 8-bit ring records cost a V2 Flash decode step ~1.3% (39 SWA layers).
///
/// E4M3 records (scales per 64 dims) are not offered: on the goldens they raised
/// KL(golden||engine) by 0.235 (V2 Flash) and 0.012 (V2.6 Pro) over BF16, int8 by
/// -0.0002 and 0.0015 at about the same size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MimoKvCache {
    /// BF16 keys and values.
    Bf16,
    /// Signed 8-bit keys and values with FP32 scales `amax / 127` per 32 dims of each head's key
    /// and value (sparkinfer `_mimo_kernels`, `kv8="s8"`).
    #[default]
    Int8,
}

impl MimoKvCache {
    /// Program-name tag of the attention programs over these records (`full_producer{tag}_m64`).
    pub fn program_tag(self) -> &'static str {
        match self {
            Self::Bf16 => "",
            Self::Int8 => "_kvint8",
        }
    }

    /// The format of `attention` layers' records: `self` for full attention, BF16 for SWA.
    pub fn of(self, attention: MimoAttention) -> Self {
        match attention {
            MimoAttention::Full => self,
            MimoAttention::Sliding => Self::Bf16,
        }
    }

    /// Dims of a key or value sharing one FP32 scale (0 for BF16).
    pub fn scale_group(self) -> usize {
        match self {
            Self::Bf16 => 0,
            Self::Int8 => 32,
        }
    }
}

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
    /// FP32 hi/lo router (V2 Flash), or the native BF16 router (V2.6).
    pub router_fp32: bool,
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
            router_fp32: match v["moe_router_dtype"].as_str() {
                Some("float32") => true,
                Some("bfloat16") => false,
                None => int("hidden_size")? == 4096 && v["model_type"] == "mimo_v2_flash",
                Some(dtype) => anyhow::bail!("unsupported MiMo moe_router_dtype {dtype}"),
            },
        })
    }

    /// Select only programs matching every checkpoint arithmetic constant.
    /// Layer patterns and RoPE tables are dynamic; their theta values still
    /// identify the qualified checkpoint variant rather than aliasing V2 Flash.
    pub fn program_family(&self) -> Result<&'static str> {
        let (base, split, value, epsilon, theta, fp32) = match (self.hidden, self.experts) {
            (4096, 256) if self.full_rope_theta == 5.0e6 => ("mimo", "mimo2", 0.707, 1e-5, 5.0e6, true),
            (4096, 256) => ("mimof", "mimof2", 0.707, 1e-6, 1.0e7, false),
            (6144, 384) => ("mimop", "mimop2", 0.612, 1e-5, 1.0e7, false),
            _ => anyhow::bail!("no MiMo program for hidden {} / {} experts: add a MiMoGeometry and exporter entry",
                self.hidden, self.experts),
        };
        let full_heads = if self.hidden == 4096 { (64, 4, 8) } else { (128, 8, 8) };
        let share_heads = (full_heads.0 / 2, full_heads.1 / 2, full_heads.2 / 2);
        let heads = (self.heads, self.full_kv_heads, self.swa_kv_heads);
        let (family, dense) = if heads == full_heads { (base, 16384) }
            else if heads == share_heads { (split, 8192) }
            else { anyhow::bail!("no {base} program for query/full KV/SWA KV heads {heads:?}") };
        for (key, actual, expected) in [
            ("head_dim", self.head_dim, 192), ("v_head_dim", self.v_head_dim, 128),
            ("rope_dim", self.rope_dim, 64), ("sliding_window", self.window, 128),
            ("intermediate_size", self.dense_intermediate, dense),
            ("moe_intermediate_size", self.moe_intermediate, 2048), ("num_experts_per_tok", self.topk, 8),
        ] {
            ensure!(actual == expected, "{family} manifest requires {key}={expected}, config has {actual}; add a matching export");
        }
        for (key, actual, expected) in [
            ("layernorm_epsilon", self.rms_norm_eps, epsilon), ("attention_value_scale", self.v_scale, value),
            ("rope_theta", self.full_rope_theta, theta), ("swa_rope_theta", self.swa_rope_theta, 1.0e4),
            ("routed_scaling_factor", self.routed_scale, 1.0),
        ] {
            ensure!(actual == expected, "{family} manifest requires {key}={expected}, config has {actual}; add a matching export");
        }
        ensure!(self.router_fp32 == fp32, "{family} manifest requires moe_router_dtype={}, config disagrees",
            if fp32 { "float32" } else { "bfloat16" });
        ensure!(!self.full_sinks && self.swa_sinks, "{family} requires sinks on SWA layers only");
        Ok(family)
    }

    /// Check the loaded export, including constants not used by family selection
    /// (the FP8 head's vocabulary, and cache page/ring extents).
    pub fn validate_program_manifest(&self, manifest: &Value, ranks: usize) -> Result<()> {
        let mut configs = vec![self.clone()];
        if ranks > 1 { configs.push(self.head_split(ranks)?); }
        for cfg in configs {
            let family = cfg.program_family()?;
            let geometry = &manifest["families"][family];
            for (key, expected) in [
                ("hidden", cfg.hidden), ("heads", cfg.heads), ("full_kv_heads", cfg.full_kv_heads),
                ("swa_kv_heads", cfg.swa_kv_heads), ("qk_head_dim", cfg.head_dim),
                ("v_head_dim", cfg.v_head_dim), ("rope_dim", cfg.rope_dim), ("window", cfg.window),
                ("dense_inter", cfg.dense_intermediate), ("moe_inter", cfg.moe_intermediate),
                ("routed_experts", cfg.experts), ("top_k", cfg.topk), ("vocab_size", cfg.vocab_size),
                ("qkv_k_stride", cfg.qkv_key_stride()), ("page_rows", 64), ("ring_rows", 256),
            ] {
                ensure!(geometry[key].as_u64() == Some(expected as u64),
                    "{family} manifest {key}={} disagrees with config/runtime {expected}; add a matching export",
                    geometry[key]);
            }
            for (key, expected) in [("norm_eps", cfg.rms_norm_eps), ("v_scale", cfg.v_scale),
                ("full_rope_theta", cfg.full_rope_theta), ("swa_rope_theta", cfg.swa_rope_theta)] {
                ensure!(geometry[key].as_f64() == Some(expected),
                    "{family} manifest {key}={} disagrees with config {expected}; add a matching export",
                    geometry[key]);
            }
            ensure!(geometry["router_fp32"].as_bool() == Some(cfg.router_fp32),
                "{family} manifest router_fp32 disagrees with checkpoint router dtype");
        }
        Ok(())
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

    /// Bytes of one token's KV record in `kv`: BF16 (`record_elems * 2`), or int8 (sparkinfer
    /// `_mimo_kernels`): signed bytes for keys and values, then one FP32 scale per
    /// `kv.scale_group()` dims of each KV head's key and value, padded to 16 bytes.
    pub fn record_bytes(&self, attention: MimoAttention, kv: MimoKvCache) -> usize {
        self.record_bytes_of(self.kv_heads(attention), kv.of(attention))
    }

    /// [`Self::record_bytes`] of a record holding `heads` KV heads.
    pub fn record_bytes_of(&self, heads: usize, kv: MimoKvCache) -> usize {
        match kv {
            MimoKvCache::Bf16 => heads * (self.head_dim + self.v_head_dim) * 2,
            MimoKvCache::Int8 => {
                let raw = heads * (self.head_dim + self.v_head_dim)
                    + 4 * heads * (self.head_dim + self.v_head_dim) / kv.scale_group();
                raw.div_ceil(16) * 16
            }
        }
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
        assert_eq!((cfg.record_bytes(MimoAttention::Full, MimoKvCache::Int8),
            cfg.record_bytes(MimoAttention::Sliding, MimoKvCache::Int8)), (1440, 5120));
        assert_eq!(cfg.head_split(2)?.record_bytes(MimoAttention::Full, MimoKvCache::Int8), 720);
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
        assert_eq!(cfg.record_bytes(MimoAttention::Full, MimoKvCache::Int8), 2560 + 320);
        assert_eq!(cfg.head_split(2)?.record_bytes(MimoAttention::Full, MimoKvCache::Int8), 1280 + 160);
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

    fn flash_mopd() -> MimoV2Config {
        let mut value = crate::plan::testing::mimo_flash_config();
        value["model_type"] = serde_json::json!("mimo_v2");
        value["vocab_size"] = serde_json::json!(152576);
        value["rope_theta"] = serde_json::json!(1.0e7);
        value["layernorm_epsilon"] = serde_json::json!(1.0e-6);
        value["moe_router_dtype"] = serde_json::json!("bfloat16");
        MimoV2Config::from_hf(&value).unwrap()
    }

    #[test]
    fn flash_mopd_constants_do_not_alias_old_flash_or_pro() -> Result<()> {
        let cfg = flash_mopd();
        assert_eq!(cfg.program_family()?, "mimof");
        assert_eq!(cfg.head_split(2)?.program_family()?, "mimof2");
        let mut variants = Vec::new();
        macro_rules! wrong { ($field:ident, $value:expr) => {{
            let mut wrong = cfg.clone(); wrong.$field = $value; variants.push(wrong);
        }}; }
        wrong!(rms_norm_eps, 1e-5); wrong!(v_scale, 0.612); wrong!(full_rope_theta, 5e6);
        wrong!(swa_rope_theta, 2e4); wrong!(router_fp32, true); wrong!(routed_scale, 2.0);
        wrong!(head_dim, 128); wrong!(v_head_dim, 192); wrong!(rope_dim, 128); wrong!(window, 256);
        wrong!(dense_intermediate, 8192); wrong!(moe_intermediate, 4096); wrong!(topk, 4);
        wrong!(full_sinks, true); wrong!(swa_sinks, false);
        for wrong in variants { assert!(wrong.program_family().is_err(), "{wrong:?}"); }
        Ok(())
    }

    fn manifest_geometry(cfg: &MimoV2Config) -> Value {
        serde_json::json!({"hidden": cfg.hidden, "heads": cfg.heads,
            "full_kv_heads": cfg.full_kv_heads, "swa_kv_heads": cfg.swa_kv_heads,
            "qk_head_dim": cfg.head_dim, "v_head_dim": cfg.v_head_dim, "rope_dim": cfg.rope_dim,
            "window": cfg.window, "dense_inter": cfg.dense_intermediate, "moe_inter": cfg.moe_intermediate,
            "routed_experts": cfg.experts, "top_k": cfg.topk, "vocab_size": cfg.vocab_size,
            "qkv_k_stride": cfg.qkv_key_stride(), "page_rows": 64, "ring_rows": 256,
            "norm_eps": cfg.rms_norm_eps, "v_scale": cfg.v_scale,
            "full_rope_theta": cfg.full_rope_theta, "swa_rope_theta": cfg.swa_rope_theta,
            "router_fp32": cfg.router_fp32})
    }

    #[test]
    fn loaded_manifests_must_match_all_constants_before_allocation() -> Result<()> {
        let cfg = flash_mopd();
        let manifest = serde_json::json!({"families": {"mimof": manifest_geometry(&cfg),
            "mimof2": manifest_geometry(&cfg.head_split(2)?)}});
        cfg.validate_program_manifest(&manifest, 1)?;
        cfg.validate_program_manifest(&manifest, 2)?;
        for family in ["mimof", "mimof2"] {
            for key in manifest["families"][family].as_object().unwrap().keys() {
                let mut wrong = manifest.clone();
                wrong["families"][family][key] = Value::Null;
                let error = cfg.validate_program_manifest(&wrong, 2).unwrap_err().to_string();
                assert!(error.contains(key), "{error}");
            }
        }
        let mut wrong_vocab = cfg.clone(); wrong_vocab.vocab_size -= 128;
        assert!(wrong_vocab.validate_program_manifest(&manifest, 1).unwrap_err().to_string().contains("vocab_size"));
        assert!(cfg.validate_program_manifest(&serde_json::json!({}), 1).is_err());
        Ok(())
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
