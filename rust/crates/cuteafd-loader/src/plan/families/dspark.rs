//! Speculators dSpark block drafters (`DSparkDraftModel`, e.g.
//! RedHatAI/GLM-5.3-Flash-speculator.dspark-preview): Qwen3 sliding-window
//! layers over taps of a target's hidden states, a Markov logit-bias head and
//! a confidence head. Not a standalone model: the plan describes it so a
//! target engine can attach it as its speculator (serve-glmf --draft).
use anyhow::Result;
use serde_json::Value;

use crate::plan::checkpoint::{opt_usize_field, usize_field, Checkpoint};
use crate::plan::family::{ConfigError, Described, Family, FamilyModel, Hint, RuntimeStatus};
use crate::plan::spec::*;

pub struct Dspark;
pub static DSPARK: Dspark = Dspark;

impl Family for Dspark {
    fn id(&self) -> &'static str {
        "dspark"
    }
    fn runtime(&self) -> RuntimeStatus {
        RuntimeStatus::Serving
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.config.get("speculators_model_type").and_then(Value::as_str) == Some("dspark")
            || checkpoint.architectures().iter().any(|arch| arch == "DSparkDraftModel")
    }

    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        self.spec(checkpoint).map(|spec| Box::new(Described(spec)) as Box<dyn FamilyModel>)
            .map_err(ConfigError::from_anyhow)
    }

    fn classify(&self, _spec: &ModelSpec, _name: &str) -> Option<TensorRole> {
        // Every tensor belongs to the drafter; it runs on the coordinator.
        Some(TensorRole::new(Component::Speculator))
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        (component == Component::Speculator).then(|| Hint {
            what: "dSpark block drafter attached to a target engine".into(),
            how: "GLM 5.3 Flash drafts with it (serve-glmf --draft SNAP, SPECULATOR=dspark; \
                  cuteafd-daemon families/glm5_flash/dspark.rs, native/families/glm5_flash/cuda/glmf_dspark.cu). \
                  The kernels take 64-wide heads, a vanilla rank-256 Markov head, sample_from_anchor and a \
                  Markov confidence head; other targets need their taps wired into the drafter."
                .into(),
        })
    }
}

impl Dspark {
    fn spec(&self, checkpoint: &Checkpoint) -> Result<ModelSpec> {
        let c = &checkpoint.config;
        let t = c.get("transformer_layer_config").cloned().unwrap_or(Value::Null);
        let layer = LayerSpec {
            rope: None,
            attention: AttentionKind::SlidingGqa {
                heads: usize_field(&t, "num_attention_heads")?,
                kv_heads: opt_usize_field(&t, "num_key_value_heads").unwrap_or(0),
                head_dim: opt_usize_field(&t, "head_dim").unwrap_or(0),
                window: opt_usize_field(&t, "sliding_window").unwrap_or(0),
                sinks: false,
            },
            ffn: FfnKind::Dense { intermediate: opt_usize_field(&t, "intermediate_size").unwrap_or(0) },
        };
        let aux: Vec<u64> = c.get("aux_hidden_state_layer_ids").and_then(Value::as_array)
            .map(|v| v.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
        let verifier = c.pointer("/speculators_config/verifier/architectures/0").and_then(Value::as_str).unwrap_or("?");
        let notes = vec![
            format!("block drafter for {verifier}: block {} (every row drafts: sample_from_anchor {}), aux taps \
                     {aux:?} (target layer outputs {:?}), mask token {}",
                opt_usize_field(c, "block_size").unwrap_or(0),
                c.get("sample_from_anchor").and_then(Value::as_bool).unwrap_or(false),
                aux.iter().map(|l| l.saturating_sub(1)).collect::<Vec<_>>(),
                opt_usize_field(c, "mask_token_id").unwrap_or(0)),
            format!("{} Markov head rank {}, confidence head {} (with Markov features {})",
                c.get("markov_head_type").and_then(Value::as_str).unwrap_or("?"),
                opt_usize_field(c, "markov_rank").unwrap_or(0),
                c.get("enable_confidence_head").and_then(Value::as_bool).unwrap_or(false),
                c.get("confidence_head_with_markov").and_then(Value::as_bool).unwrap_or(false)),
        ];
        Ok(ModelSpec {
            family: "dspark",
            architecture: "DSparkDraftModel".into(),
            hidden: usize_field(&t, "hidden_size")?,
            vocab: usize_field(&t, "vocab_size")?,
            layers: vec![layer; usize_field(&t, "num_hidden_layers")?],
            moe: None,
            speculator: None,
            tables: Vec::new(),
            vision: false,
            notes,
        })
    }
}
