//! Checkpoint-native MLA DFlash2 geometry shared by admission and loading.
use super::block_dspark::{DsparkConfigError, DsparkRope, DsparkRotaryPlan};
use serde::Deserialize;
use std::path::Path;

type Result<T> = std::result::Result<T, DsparkConfigError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftGeometry {
    pub hidden: usize,
    pub heads: usize,
    pub q_rank: usize,
    pub kv_rank: usize,
    pub nope: usize,
    pub rope: usize,
    pub value: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaDflashConfig {
    pub mla: MlaDraftGeometry,
    pub intermediate: usize,
    pub eps: f64,
    pub block: usize,
    pub group: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub taps: Vec<usize>,
    pub vocab: usize,
    pub mask_token: u32,
    pub max_context: usize,
    /// None means full context; each sliding layer keeps its own exact window.
    pub windows: Vec<Option<usize>>,
    pub rotary: DsparkRotaryPlan,
}

#[derive(Deserialize)]
struct Raw {
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    rms_norm_eps: f64,
    vocab_size: usize,
    draft_vocab_size: usize,
    max_position_embeddings: usize,
    num_target_layers: usize,
    layer_types: Vec<String>,
    sliding_window: Option<usize>,
    use_sliding_window: bool,
    is_causal: bool,
    mla_use_output_gate: bool,
    attention_bias: bool,
    hidden_act: String,
    fc_norm: bool,
    rope_parameters: DsparkRope,
    dflash_config: Draft,
}

#[derive(Deserialize)]
struct Draft {
    attention_mode: String,
    block_size: usize,
    conv_kernel_size: usize,
    conv_group_size: usize,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
    mask_token_id: u32,
}

fn unsupported(message: &str) -> DsparkConfigError {
    DsparkConfigError::Unsupported(message.into())
}

impl MlaDflashConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_json(&serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?)
    }

    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let raw: Raw = serde_json::from_value(value.clone())?;
        let d = raw.dflash_config;
        if !raw.architectures.iter().any(|a| a == "DFlash2DraftModel")
            || d.attention_mode != "mla" || raw.is_causal || raw.mla_use_output_gate
            || raw.attention_bias || raw.hidden_act != "silu" || raw.fc_norm {
            return Err(unsupported("MLA DFlash2 requires noncausal, ungated, bias-free SiLU and fc_norm=false"));
        }
        let mla = MlaDraftGeometry { hidden: raw.hidden_size, heads: raw.num_attention_heads,
            q_rank: raw.q_lora_rank, kv_rank: raw.kv_lora_rank, nope: raw.qk_nope_head_dim,
            rope: raw.qk_rope_head_dim, value: raw.v_head_dim };
        if mla != (MlaDraftGeometry { hidden: 7168, heads: 64, q_rank: 1536,
            kv_rank: 512, nope: 128, rope: 64, value: 128 })
            || raw.num_key_value_heads != mla.heads || raw.rms_norm_eps != 1e-5 {
            return Err(unsupported("MLA draft exports require H7168, 64 heads, Q1536/KV512, NoPE128/RoPE64/V128, eps=1e-5"));
        }
        if raw.intermediate_size == 0 || raw.intermediate_size > 1_000_000
            || raw.intermediate_size % 128 != 0 || raw.num_hidden_layers == 0
            || raw.num_hidden_layers > 1024 || raw.vocab_size == 0 || raw.vocab_size > 1_000_000
            || raw.vocab_size != raw.draft_vocab_size || d.mask_token_id as usize >= raw.vocab_size
            || raw.max_position_embeddings == 0 || raw.max_position_embeddings > 1_048_576
            || raw.num_target_layers == 0 || raw.num_target_layers > 1_000_000
            || d.target_layer_ids.is_empty() || d.target_layer_ids.len() > 1024
            || d.target_layer_ids.iter().any(|&tap| tap >= raw.num_target_layers)
            || d.target_layer_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(unsupported("invalid MLA draft dimensions, context, vocabulary or ordered post-layer taps"));
        }
        if d.block_size != 8 || d.conv_kernel_size != 2 || d.conv_group_size != 16
            || d.selector_rank != 256 || d.selector_top_k != 16 {
            return Err(unsupported("DFlash2 requires block8, two-tap group16 convolutions, rank256/top16 selector"));
        }
        if raw.layer_types.len() != raw.num_hidden_layers {
            return Err(unsupported("MLA draft layer_types count differs from num_hidden_layers"));
        }
        let windows = raw.layer_types.iter().map(|kind| match kind.as_str() {
            "full_attention" => Ok(None),
            "sliding_attention" if raw.use_sliding_window && raw.sliding_window == Some(4096) => Ok(Some(4096)),
            _ => Err(unsupported("MLA draft exports support full attention or a native 4096-token window")),
        }).collect::<Result<Vec<_>>>()?;
        let rotary = raw.rope_parameters.plan(mla.rope)?;
        Ok(Self { mla, intermediate: raw.intermediate_size, eps: raw.rms_norm_eps,
            block: d.block_size, group: d.conv_group_size, selector_rank: d.selector_rank,
            selector_top_k: d.selector_top_k, taps: d.target_layer_ids, vocab: raw.vocab_size,
            mask_token: d.mask_token_id, max_context: raw.max_position_embeddings, windows, rotary })
    }

    pub fn attention_scale(&self) -> f32 {
        ((self.mla.nope + self.mla.rope) as f64).sqrt().recip() as f32 * self.rotary.softmax_factor
    }

    /// Exact BF16 source shapes. Embedding and vocabulary head are borrowed
    /// from the target, matching the checkpoint's reference execution.
    pub fn tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let h = self.mla.hidden;
        let mut shapes = vec![
            ("fc.weight".into(), vec![h, self.taps.len() * h]),
            ("hidden_norm.weight".into(), vec![h]),
            ("norm.weight".into(), vec![h]),
            ("candidate_selector.hidden_projection.weight".into(), vec![self.selector_rank, h]),
            ("candidate_selector.predecessor_codebook".into(), vec![self.vocab, self.selector_rank]),
            ("candidate_selector.successor_codebook".into(), vec![self.vocab, self.selector_rank]),
        ];
        for layer in 0..self.windows.len() {
            let p = format!("layers.{layer}");
            for leaf in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                shapes.push((format!("{p}.{leaf}"), vec![h]));
            }
            for conv in ["attention_conv", "mlp_conv"] {
                shapes.push((format!("{p}.{conv}.kernel_projection.weight"), vec![4 * h / self.group, h]));
                shapes.push((format!("{p}.{conv}.base_kernel"), vec![2, 2, h]));
            }
            for (leaf, shape) in [
                ("q_a_proj", vec![self.mla.q_rank, h]),
                ("kv_a_proj_with_mqa", vec![self.mla.kv_rank + self.mla.rope, h]),
                ("q_a_layernorm", vec![self.mla.q_rank]),
                ("kv_a_layernorm", vec![self.mla.kv_rank]),
                ("q_b_proj", vec![self.mla.heads * (self.mla.nope + self.mla.rope), self.mla.q_rank]),
                ("kv_b_proj", vec![self.mla.heads * (self.mla.nope + self.mla.value), self.mla.kv_rank]),
                ("o_proj", vec![h, self.mla.heads * self.mla.value]),
            ] {
                shapes.push((format!("{p}.self_attn.{leaf}.weight"), shape));
            }
            for (leaf, shape) in [
                ("gate_proj", vec![self.intermediate, h]),
                ("up_proj", vec![self.intermediate, h]),
                ("down_proj", vec![h, self.intermediate]),
            ] {
                shapes.push((format!("{p}.mlp.{leaf}.weight"), shape));
            }
        }
        shapes
    }

    pub fn bf16_weight_bytes(&self) -> Result<u64> {
        self.tensor_shapes().iter().try_fold(0u64, |total, (_, shape)| {
            let bytes = shape.iter().try_fold(2u64, |n, &d| n.checked_mul(d as u64))
                .ok_or_else(|| unsupported("MLA draft weight byte overflow"))?;
            total.checked_add(bytes).ok_or_else(|| unsupported("MLA draft weight total overflow"))
        })
    }

    /// Latent BF16 records only, not block/workspace/staging admission. Full
    /// layers grow to the selected context; sliding layers do not silently grow.
    pub fn cache_bytes(&self, slots: usize, context: usize) -> Result<u64> {
        if slots == 0 || context == 0 || context > self.max_context {
            return Err(unsupported("MLA draft cache slots/context outside checkpoint extent"));
        }
        let rows = self.windows.iter().try_fold(0u64, |sum, window| {
            sum.checked_add(window.unwrap_or(context) as u64)
                .ok_or_else(|| unsupported("MLA draft cache row overflow"))
        })?;
        rows.checked_mul(slots as u64).and_then(|n| n.checked_mul((self.mla.kv_rank + self.mla.rope) as u64))
            .and_then(|n| n.checked_mul(2)).ok_or_else(|| unsupported("MLA draft cache byte overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> serde_json::Value {
        json!({"architectures":["DFlash2DraftModel"], "hidden_size":7168,
            "intermediate_size":14336, "num_hidden_layers":5, "num_attention_heads":64,
            "num_key_value_heads":64, "q_lora_rank":1536, "kv_lora_rank":512,
            "qk_nope_head_dim":128, "qk_rope_head_dim":64, "v_head_dim":128,
            "rms_norm_eps":1e-5, "vocab_size":163840, "draft_vocab_size":163840,
            "max_position_embeddings":1048576, "num_target_layers":93,
            "layer_types":["sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","full_attention"], "sliding_window":4096,
            "use_sliding_window":true, "is_causal":false, "mla_use_output_gate":false,
            "attention_bias":false, "hidden_act":"silu", "fc_norm":false,
            "rope_parameters":{"rope_type":"yarn", "rope_theta":50000.0, "factor":32.0,
                "original_max_position_embeddings":32768, "beta_fast":32, "beta_slow":1,
                "mscale":1.0, "mscale_all_dim":1.0},
            "dflash_config":{"attention_mode":"mla", "block_size":8,
                "conv_kernel_size":2, "conv_group_size":16, "selector_rank":256,
                "selector_top_k":16, "target_layer_ids":[19,37,66,78,90], "mask_token_id":163592}})
    }

    #[test]
    fn native_mla_weights_and_full_sliding_cache_are_exact() {
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        assert_eq!(cfg.windows, vec![Some(4096), Some(4096), Some(4096), Some(4096), None]);
        assert_eq!(cfg.taps, [19,37,66,78,90]);
        assert_eq!(cfg.bf16_weight_bytes().unwrap(), 5_036_945_408);
        assert!(!cfg.tensor_shapes().iter().any(|(name, _)| name.contains("embed_tokens") || name.contains("lm_head")));
        assert_eq!(cfg.cache_bytes(8, 262144).unwrap(), 2_566_914_048);
        assert_eq!(cfg.cache_bytes(8, 1048576).unwrap(), 9_814_671_360);
        assert_eq!(cfg.rotary.attention_factor, 1.0);
        let expected = (192.0f64).sqrt().recip() * (1.0 + 0.1 * 32.0f64.ln()).powi(2);
        assert!((cfg.attention_scale() as f64 - expected).abs() < 2e-8);
        assert!(cfg.cache_bytes(0, 262144).is_err());
        assert!(cfg.cache_bytes(8, 1048577).is_err());
        assert!(cfg.cache_bytes(usize::MAX, 1048576).is_err());
    }

    #[test]
    fn mla_config_refuses_unsupported_native_contracts() {
        for (field, value) in [
            ("mla_use_output_gate", json!(true)), ("is_causal", json!(true)),
            ("fc_norm", json!(true)), ("attention_bias", json!(true)),
            ("rms_norm_eps", json!(1e-6)), ("num_attention_heads", json!(96)),
            ("qk_rope_head_dim", json!(128)), ("sliding_window", json!(2048)),
            ("num_hidden_layers", json!(6)), ("vocab_size", json!(0)),
            ("intermediate_size", json!(usize::MAX)), ("max_position_embeddings", json!(0)),
        ] {
            let mut raw = config(); raw[field] = value;
            assert!(MlaDflashConfig::from_json(&raw).is_err(), "{field}");
        }
        for (field, value) in [("target_layer_ids", json!([19,19])),
            ("target_layer_ids", json!([37,19])), ("target_layer_ids", json!([93])),
            ("mask_token_id", json!(163840)), ("conv_group_size", json!(0)),
            ("block_size", json!(64)), ("attention_mode", json!("gqa"))] {
            let mut raw = config(); raw["dflash_config"][field] = value;
            assert!(MlaDflashConfig::from_json(&raw).is_err(), "{field}");
        }
    }
}
