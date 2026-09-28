//! Qwen 3.8 Flash Next (qwen4_exp): Gated DeltaNet linear attention with a
//! full-attention layer every fourth (with an indexer), fused expert tensors,
//! shared expert with a sigmoid gate, hyper-connections and PLE n-gram tables.
use anyhow::Result;
use serde_json::Value;

use crate::plan::checkpoint::{opt_usize_field, usize_field, Checkpoint};
use crate::plan::family::{Family, Hint, RuntimeStatus};
use crate::plan::names::indexed;
use crate::plan::spec::*;

pub struct Qwen;
pub static QWEN4_EXP: Qwen = Qwen;

impl Family for Qwen {
    fn id(&self) -> &'static str {
        "qwen4_exp"
    }
    fn runtime(&self) -> RuntimeStatus {
        RuntimeStatus::Planned
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
            family: "qwen4_exp",
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
        let qflash = "../qflashrt (single-device Qwen3.8-Flash-Next runtime)";
        let (what, how) = match component {
            Component::Attention => (
                "Gated DeltaNet linear attention (36 layers) plus gated full attention every 4th layer".to_string(),
                format!("b12x sequence/gdn_prefill + gdn_decode; per-request recurrent state pool. \
                 {qflash} has a working port including the conv1d short convolution and output gate."),
            ),
            Component::RoutedExpert => (
                "512 experts top-10 stored as fused [experts, out, in] BF16 tensors".to_string(),
                "The Spark reader must slice fused expert tensors along the expert axis before TP \
                 slicing; quantize at load (MXFP4/NVFP4) or use a published EXL3 K4.25 build \
                 (wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-*).".to_string(),
            ),
            Component::MappedTable | Component::TableProjection => (
                "PLE n-gram embedding: 128 shards of 2.5M x 160 BF16 (~95 GB)".to_string(),
                format!("Host-mapped table with prefetch, like V4.1 engram (loader engram_*). \
                 b12x sequence/ple, ple_hash, ple_embedding; {qflash} has the hashing and gather."),
            ),
            Component::HyperConnection => (
                "low-rank hyper-connections (hc_count 4, rank 320)".to_string(),
                "b12x norm/hyperconnection; differs from DeepSeek mHC (no Sinkhorn).".to_string(),
            ),
            _ => return None,
        };
        Some(Hint { what, how })
    }
}
