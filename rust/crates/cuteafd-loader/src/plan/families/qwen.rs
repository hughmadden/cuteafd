//! Qwen 3.8 Flash Next (qwen4_exp): Gated DeltaNet linear attention with a
//! full-attention layer every fourth (with an indexer), fused expert tensors,
//! shared expert with a sigmoid gate, hyper-connections and PLE n-gram tables.
use anyhow::Result;
use serde_json::Value;

use crate::plan::checkpoint::{opt_usize_field, usize_field, Checkpoint};
use crate::plan::family::{Family, Hint, RuntimeStatus};
use crate::plan::format::WeightFormat;
use crate::plan::names::indexed;
use crate::plan::spec::*;

pub struct Qwen;
pub static QWEN4: Qwen = Qwen;

impl Family for Qwen {
    fn id(&self) -> &'static str {
        "qwen4"
    }
    fn runtime(&self) -> RuntimeStatus {
        RuntimeStatus::Serving
    }

    fn executes(&self, component: Component, format: &WeightFormat) -> bool {
        use WeightFormat::*;
        // qwen4 programs: BF16 coordinator tensors; the PLE table in BF16 or
        // E4M3 with one scale; routed experts from EXL3 K4/K5 packages
        // (qwen4:exl3-k45) or the checkpoint's FP8 128x128 blocks (qwen4:fp8).
        match component {
            Component::RoutedExpert => matches!(format, Exl3 { bits: 4..=5 } | Fp8Block { block: (128, 128) }),
            Component::MappedTable => matches!(format, Bf16 | Fp8PerChannel | Int),
            Component::Speculator | Component::SpeculatorExpert | Component::Vision => false,
            _ => matches!(format, Bf16 | F32),
        }
    }

    fn optional(&self, component: Component) -> bool {
        // Text serving runs without the native MTP layer and the vision tower.
        matches!(component, Component::Speculator | Component::SpeculatorExpert | Component::Vision)
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint
            .architectures()
            .iter()
            .any(|arch| arch == "Qwen4ExpForConditionalGeneration" || arch == "Qwen4ExpForCausalLM")
    }

    fn spec(&self, checkpoint: &Checkpoint) -> Result<ModelSpec> {
        let text = checkpoint.text_config();
        let layers = usize_field(text, "num_hidden_layers")?;
        let types: Vec<String> = text
            .get("layer_types")
            .and_then(Value::as_array)
            .map(|v| v.iter().filter_map(|t| t.as_str().map(str::to_owned)).collect())
            .unwrap_or_default();
        let layer_specs = (0..layers)
            .map(|layer| LayerSpec {
                attention: if types.get(layer).map(String::as_str) == Some("linear_attention") {
                    AttentionKind::GatedDeltaNet
                } else {
                    AttentionKind::Gqa {
                        heads: opt_usize_field(text, "num_attention_heads").unwrap_or(0),
                        kv_heads: opt_usize_field(text, "num_key_value_heads").unwrap_or(0),
                        head_dim: opt_usize_field(text, "head_dim").unwrap_or(0),
                    }
                },
                ffn: FfnKind::Moe,
            })
            .collect();
        let ple_layers: Vec<usize> = text
            .get("ple_layer_ids")
            .and_then(Value::as_array)
            .map(|v| v.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
            .unwrap_or_default();
        let mtp = opt_usize_field(text, "mtp_num_hidden_layers").unwrap_or(0);
        Ok(ModelSpec {
            family: "qwen4",
            architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
            hidden: usize_field(text, "hidden_size")?,
            vocab: usize_field(text, "vocab_size")?,
            layers: layer_specs,
            moe: Some(MoeSpec {
                experts: usize_field(text, "num_experts")?,
                top_k: usize_field(text, "num_experts_per_tok")?,
                intermediate: usize_field(text, "moe_intermediate_size")?,
                shared_experts: 1,
                shared_intermediate: opt_usize_field(text, "shared_expert_intermediate_size").unwrap_or(0),
                scoring: "softmax".into(),
                routed_scaling: None,
                groups: None,
            }),
            speculator: (mtp > 0).then_some(SpeculatorSpec::NativeMtp { layers: mtp }),
            tables: if ple_layers.is_empty() {
                Vec::new()
            } else {
                vec![MappedTableSpec { name: "ple-ngram".into(), layers: ple_layers }]
            },
            vision: checkpoint.config.get("vision_config").is_some(),
            notes: vec![format!(
                "hyper-connections {} (low rank {}), indexer budget {}",
                opt_usize_field(text, "hc_count").unwrap_or(0),
                opt_usize_field(text, "hc_lowrank").unwrap_or(0),
                opt_usize_field(text, "indexer_budget").unwrap_or(0)
            )],
        })
    }

    fn classify(&self, _spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        use Component::*;
        if name == "lm_head.weight" {
            return Some(TensorRole::new(LmHead));
        }
        if name.starts_with("mtp.") {
            if name.contains(".mlp.experts.") {
                return Some(TensorRole::new(SpeculatorExpert));
            }
            return Some(TensorRole::new(Speculator));
        }
        if name.starts_with("model.visual.") || name.starts_with("visual.") {
            return Some(TensorRole::new(Vision));
        }
        let name = name.strip_prefix("model.language_model.")?;
        match name {
            "embed_tokens.weight" => return Some(TensorRole::new(Embedding)),
            "norm.weight" => return Some(TensorRole::new(Norm)),
            _ => {}
        }
        if name.starts_with("hyper_connection_mixer.") {
            return Some(TensorRole::new(HyperConnection));
        }
        let (layer, rest) = indexed(name, "layers.")?;
        let component = if rest.starts_with("mlp.experts.") {
            // Fused [experts, ...] tensors: one tensor per projection per layer.
            RoutedExpert
        } else if rest.starts_with("ple.ple_embedding.") {
            MappedTable
        } else if rest.starts_with("ple.") {
            TableProjection
        } else if rest.starts_with("self_attn.indexer.") {
            Indexer
        } else if rest.starts_with("self_attn.") || rest.starts_with("linear_attn.") {
            Attention
        } else if rest.starts_with("mlp.gate.") {
            Router
        } else if rest.starts_with("mlp.shared_expert") {
            SharedExpert
        } else if rest.contains("hyper_connection.") {
            HyperConnection
        } else if rest.ends_with("layernorm.weight") || rest.ends_with("norm.weight") {
            Norm
        } else {
            Other
        };
        Some(TensorRole::layer(component, layer))
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        let (what, how) = match component {
            Component::RoutedExpert => (
                "512 experts top-10 (hidden 2560, intermediate 640, SiLU unclamped) in a format without a \
                 routed kernel family"
                    .to_string(),
                "Executable: EXL3 K4/K5 (qwen4:exl3-k45, python/tools/aot/package_exl3_aot.py --geometry qwen4; \
                 exl3_cross_sm121.py for Sparks) and FP8 128x128 blocks (qwen4:fp8, native/cmake/shared/fp8_moe.cmake). \
                 BF16 fused [512, 1280, 2560] experts: serve the FP8 or EXL3 K4.25 publication, or add a BF16 \
                 routed family. NVFP4 (ModelOpt E2M1 + E4M3 per-16 scales + FP32 global): export b12x fused_moe \
                 NVFP4 at this geometry (W4A16 for accuracy) and teach read_expert_catalog its tensors."
                    .to_string(),
            ),
            Component::Speculator | Component::SpeculatorExpert => (
                "native MTP layer (full attention + 512 experts, hyper-connection feedback) is not run".to_string(),
                "serve-qwen4 verifies copy-window drafts only; MTP needs the mtp.* weights as one more qwen4 \
                 layer fed by fc_hidden/fc_embedding (see ../qflashrt quantization/qwen_model.py mtp_feedback) \
                 and its experts served (FP8/EXL3 packages)."
                    .to_string(),
            ),
            Component::Vision => (
                "the vision tower is not run".to_string(),
                "Text only; images would need the Qwen4Exp vision encoder and mRoPE positions.".to_string(),
            ),
            Component::MappedTable | Component::TableProjection => (
                "PLE n-gram table in a format other than BF16 or E4M3 with one scale".to_string(),
                "The qwen4_ple_{bf16,fp8} programs gather 16 rows of 160 per token from a pinned host (or GPU) \
                 table; add a gather variant for the new format."
                    .to_string(),
            ),
            _ => (
                format!("{} tensors in a format other than BF16", component.label()),
                "The qwen4 coordinator programs take BF16 weights; dequantize at load (weights.rs) or add \
                 an FP8 program variant."
                    .to_string(),
            ),
        };
        Some(Hint { what, how })
    }
}
