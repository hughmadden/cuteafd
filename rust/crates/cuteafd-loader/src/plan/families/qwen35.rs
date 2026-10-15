//! Dense Qwen3.5/3.8: recognized header contract, not yet an executable family.
use crate::families::qwen::HybridAttention;
use crate::families::qwen3_5::Qwen35Checkpoint;
use crate::plan::{checkpoint::Checkpoint, family::*, format, names::indexed, spec::*};

pub struct Qwen35;
pub static QWEN35: Qwen35 = Qwen35;

impl Family for Qwen35 {
    fn id(&self) -> &'static str { "qwen3_5" }
    fn runtime(&self) -> RuntimeStatus { RuntimeStatus::Planned }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|a| a == "Qwen3_5ForConditionalGeneration")
            || checkpoint.model_type() == Some("qwen3_5")
    }
    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        let header = Qwen35Checkpoint::open(checkpoint).map_err(ConfigError::from_anyhow)?;
        let cfg = &header.config;
        let spec = ModelSpec {
            family: self.id(), architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
            hidden: cfg.hidden, vocab: cfg.vocab,
            layers: cfg.attention.iter().map(|&kind| LayerSpec {
                attention: match kind {
                    HybridAttention::Gdn => AttentionKind::GatedDeltaNet,
                    HybridAttention::Full => AttentionKind::Gqa { heads: cfg.heads, kv_heads: cfg.kv_heads, head_dim: cfg.head_dim },
                },
                ffn: FfnKind::Dense { intermediate: cfg.intermediate },
                rope: (kind == HybridAttention::Full).then_some(RopeSpec { dims: cfg.rope_dim, theta: cfg.rope_theta }),
            }).collect(),
            moe: None, speculator: (cfg.mtp_layers > 0).then_some(SpeculatorSpec::NativeMtp { layers: cfg.mtp_layers }),
            tables: vec![], vision: checkpoint.config.get("vision_config").is_some(),
            notes: vec!["0-Spark dense target; no router, experts, mHC, PLE or QSA indexer".into(),
                "Transformers 5.13.0 reference: GDN SiLU, full-attention sigmoid; output_gate_type=swish is not a GQA gate selector".into(),
                format!("BF16 full KV {} bytes/token; exact FP32 GDN + BF16 convolution state {} bytes/sequence or retained mark; native context {} tokens",
                    cfg.full_kv_bytes_per_token(), cfg.recurrent_state_bytes(), cfg.max_context),
                "Header inventory only: runtime workspace/graph/prefix admission and dense executor are not implemented; no serving-fit claim".into()],
        };
        Ok(Box::new(Model { header, spec }))
    }
    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        let (stem, _) = format::split_stem(name);
        let component = match stem {
            "lm_head" => Some(Component::LmHead),
            "model.language_model.embed_tokens" => Some(Component::Embedding),
            "model.language_model.norm" => Some(Component::Norm),
            _ => None,
        };
        if let Some(component) = component { return Some(TensorRole::new(component)); }
        if stem.starts_with("mtp.") { return Some(TensorRole::new(Component::Speculator)); }
        if stem.starts_with("model.visual.") { return Some(TensorRole::new(Component::Vision)); }
        let (layer, rest) = indexed(stem.strip_prefix("model.language_model.")?, "layers.")?;
        if layer >= spec.layers.len() { return None; }
        let component = if matches!(rest, "input_layernorm" | "post_attention_layernorm") { Component::Norm }
            else if rest.starts_with("self_attn.") || rest.starts_with("linear_attn.") { Component::Attention }
            else if matches!(rest, "mlp.gate_proj" | "mlp.up_proj" | "mlp.down_proj") { Component::DenseFfn }
            else { return None; };
        Some(TensorRole::layer(component, layer))
    }
    fn component_hint(&self, component: Component) -> Option<Hint> {
        let missing = match component {
            Component::Attention => "SiLU-gated GDN and unrestricted causal paged GQA (no QSA), M-RoPE, dense geometry exports",
            Component::DenseFfn => "single-stream residual + dense SiLU gate/up/down projections",
            Component::Vision => "bundled Qwen tower merger/insertion at H4096/H5120; no multimodal parity claim",
            Component::Speculator => "dense MTP feedback fc [H,2H], normalized hidden/token input order and dense block; MTP off for first golden",
            _ => "single-stream zero-centered norms, final norm/head, admission and exact prefix-state lifecycle",
        };
        Some(Hint { what: format!("qwen3_5 {} is planned, not executable", component.label()),
            how: format!("Implement {missing} using shared Qwen operators; gate against scripts/qualify/qwen3_5/reference.py before enabling runtime.") })
    }
    fn optional(&self, component: Component) -> bool {
        matches!(component, Component::Vision | Component::Speculator)
    }
}

struct Model { header: Qwen35Checkpoint, spec: ModelSpec }
impl FamilyModel for Model {
    fn spec(&self) -> &ModelSpec { &self.spec }
    fn accepts(&self, _role: &TensorRole, stem: &str, operand: &mut format::QuantOperand) -> Result<(), String> {
        self.header.accepts(stem, operand)
    }
}
