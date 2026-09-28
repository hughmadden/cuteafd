//! Xiaomi MiMo V2: hybrid full / sliding-window GQA with attention sinks,
//! sigmoid top-8 experts without shared experts, dense-FFN MTP layers.
use anyhow::Result;
use serde_json::Value;

use crate::plan::checkpoint::{opt_usize_field, usize_field, Checkpoint};
use crate::plan::family::{Family, Hint, RuntimeStatus};
use crate::plan::names::indexed;
use crate::plan::spec::*;

pub struct MiMo;
pub static MIMO_V2: MiMo = MiMo;

const ARCHITECTURES: &[&str] = &["MiMoV2FlashForCausalLM", "MiMoV2ForCausalLM"];

fn usize_list(config: &Value, key: &str) -> Vec<usize> {
    config
        .get(key)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
        .unwrap_or_default()
}

impl Family for MiMo {
    fn id(&self) -> &'static str {
        "mimo_v2"
    }
    fn runtime(&self) -> RuntimeStatus {
        RuntimeStatus::Planned
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|arch| ARCHITECTURES.contains(&arch.as_str()))
    }

    fn spec(&self, checkpoint: &Checkpoint) -> Result<ModelSpec> {
        let text = checkpoint.text_config();
        let layers = usize_field(text, "num_hidden_layers")?;
        let hybrid = usize_list(text, "hybrid_layer_pattern");
        let moe_freq = usize_list(text, "moe_layer_freq");
        let window = opt_usize_field(text, "sliding_window_size")
            .or_else(|| opt_usize_field(text, "sliding_window"))
            .unwrap_or(0);
        let sinks = text.get("add_swa_attention_sink_bias").and_then(Value::as_bool).unwrap_or(false);
        let layer_specs = (0..layers)
            .map(|layer| LayerSpec {
                attention: if hybrid.get(layer) == Some(&1) {
                    AttentionKind::SlidingGqa {
                        heads: opt_usize_field(text, "swa_num_attention_heads").unwrap_or(0),
                        kv_heads: opt_usize_field(text, "swa_num_key_value_heads").unwrap_or(0),
                        head_dim: opt_usize_field(text, "swa_head_dim").unwrap_or(0),
                        window,
                        sinks,
                    }
                } else {
                    AttentionKind::Gqa {
                        heads: opt_usize_field(text, "num_attention_heads").unwrap_or(0),
                        kv_heads: opt_usize_field(text, "num_key_value_heads").unwrap_or(0),
                        head_dim: opt_usize_field(text, "head_dim").unwrap_or(0),
                    }
                },
                ffn: if moe_freq.get(layer) == Some(&1) {
                    FfnKind::Moe
                } else {
                    FfnKind::Dense { intermediate: opt_usize_field(text, "intermediate_size").unwrap_or(0) }
                },
            })
            .collect();
        let mtp = checkpoint
            .tensors
            .iter()
            .filter_map(|t| indexed(&t.meta.name, "model.mtp.layers.").map(|(i, _)| i))
            .max()
            .map_or(0, |max| max + 1);
        Ok(ModelSpec {
            family: "mimo_v2",
            architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
            hidden: usize_field(text, "hidden_size")?,
            vocab: usize_field(text, "vocab_size")?,
            layers: layer_specs,
            moe: Some(MoeSpec {
                experts: usize_field(text, "n_routed_experts")?,
                top_k: usize_field(text, "num_experts_per_tok")?,
                intermediate: usize_field(text, "moe_intermediate_size")?,
                shared_experts: 0,
                shared_intermediate: 0,
                scoring: text.get("scoring_func").and_then(Value::as_str).unwrap_or("sigmoid").into(),
                routed_scaling: text.get("routed_scaling_factor").and_then(Value::as_f64),
                groups: None,
            }),
            speculator: (mtp > 0).then_some(SpeculatorSpec::NativeMtp { layers: mtp }),
            tables: Vec::new(),
            vision: checkpoint.config.get("vision_config").is_some(),
            notes: vec![format!(
                "partial rotary {}, attention value scale {}",
                text.get("partial_rotary_factor").cloned().unwrap_or(Value::Null),
                text.get("attention_value_scale").cloned().unwrap_or(Value::Null)
            )],
        })
    }

    fn classify(&self, _spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        use Component::*;
        match name {
            "lm_head.weight" => return Some(TensorRole::new(LmHead)),
            "model.embed_tokens.weight" => return Some(TensorRole::new(Embedding)),
            "model.norm.weight" => return Some(TensorRole::new(Norm)),
            _ => {}
        }
        if let Some((layer, _)) = indexed(name, "model.mtp.layers.") {
            return Some(TensorRole::layer(Speculator, layer));
        }
        if name.starts_with("visual.") || name.starts_with("model.visual.") || name.starts_with("audio")
            || name.starts_with("speech_") || name.starts_with("model.audio")
        {
            return Some(TensorRole::new(Vision));
        }
        let (layer, rest) = indexed(name, "model.layers.")?;
        if let Some((expert, _)) = indexed(rest, "mlp.experts.") {
            return Some(TensorRole::expert(RoutedExpert, layer, expert));
        }
        let component = if rest.starts_with("self_attn.") {
            Attention
        } else if rest.starts_with("mlp.gate.") {
            Router
        } else if rest.starts_with("mlp.") {
            DenseFfn
        } else if rest.ends_with("layernorm.weight") {
            Norm
        } else {
            Other
        };
        Some(TensorRole::layer(component, layer))
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        let (what, how) = match component {
            Component::Attention => (
                "GQA full attention plus 128-token sliding-window GQA with learned sink bias",
                "b12x attention/paged (FP8 KV) and varlen cover GQA; add sink-bias support to the \
                 softmax (it is a per-head additive logit, like V4.1 attn_sink). Head dims differ for \
                 QK (192) and V (128); partial rotary 0.334. Sliding layers need only a 128-token ring.",
            ),
            Component::RoutedExpert => (
                "sigmoid top-8 routed experts, no shared expert",
                "Export b12x fused_moe at this geometry. MiMo V2 Flash is FP8 128x128 block \
                 (w8a8 or dequantize-on-load to MXFP4); V2.6 Pro stores MXFP4 with E8M0 scales typed U8.",
            ),
            Component::Router => (
                "sigmoid noaux_tc router with e_score_correction_bias (router weight FP32 on Flash)",
                "GLM's top-8 sigmoid router path covers this; no routed scaling factor.",
            ),
            Component::Speculator => (
                "three dense-FFN MTP layers",
                "Each MTP layer is a full attention block plus dense FP8 FFN (eh_proj/enorm/hnorm).",
            ),
            _ => return None,
        };
        Some(Hint { what: what.into(), how: how.into() })
    }
}
