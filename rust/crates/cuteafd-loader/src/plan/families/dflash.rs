//! DFlash2 block drafters (`DFlash2DraftModel`): a few Qwen3-style
//! sliding-window layers that read taps of a target model's hidden states
//! and draft a block of tokens in one pass. Not a standalone model: the plan
//! describes it so a target engine can attach it as its speculator.
use anyhow::Result;
use serde_json::Value;

use crate::plan::checkpoint::{opt_usize_field, usize_field, Checkpoint};
use crate::plan::family::{Family, Hint, RuntimeStatus};
use crate::plan::spec::*;

pub struct DFlash2;
pub static DFLASH2: DFlash2 = DFlash2;

impl Family for DFlash2 {
    fn id(&self) -> &'static str {
        "dflash2"
    }
    fn runtime(&self) -> RuntimeStatus {
        RuntimeStatus::Planned
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|arch| arch == "DFlash2DraftModel")
    }

    fn spec(&self, checkpoint: &Checkpoint) -> Result<ModelSpec> {
        let c = &checkpoint.config;
        let dflash = c.get("dflash_config").cloned().unwrap_or(Value::Null);
        let window = opt_usize_field(c, "sliding_window").unwrap_or(0);
        let layer = LayerSpec {
            attention: AttentionKind::SlidingGqa {
                heads: usize_field(c, "num_attention_heads")?,
                kv_heads: opt_usize_field(c, "num_key_value_heads").unwrap_or(0),
                head_dim: opt_usize_field(c, "head_dim").unwrap_or(0),
                window,
                sinks: false,
            },
            ffn: FfnKind::Dense { intermediate: opt_usize_field(c, "intermediate_size").unwrap_or(0) },
        };
        let taps: Vec<u64> = dflash
            .get("target_layer_ids")
            .and_then(Value::as_array)
            .map(|v| v.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        let notes = vec![
            format!(
                "block drafter: block {}, taps {taps:?} of a {}-layer target (fc input {} x hidden)",
                opt_usize_field(&dflash, "block_size").unwrap_or(0),
                opt_usize_field(c, "num_target_layers").unwrap_or(0),
                taps.len(),
            ),
            format!(
                "candidate selector rank {} top-{}, two-tap attention/MLP convolutions (kernel {}, groups of {}), \
                 mask token {}",
                opt_usize_field(&dflash, "selector_rank").unwrap_or(0),
                opt_usize_field(&dflash, "selector_top_k").unwrap_or(0),
                opt_usize_field(&dflash, "conv_kernel_size").unwrap_or(0),
                opt_usize_field(&dflash, "conv_group_size").unwrap_or(0),
                opt_usize_field(&dflash, "mask_token_id").unwrap_or(0),
            ),
        ];
        Ok(ModelSpec {
            family: "dflash2",
            architecture: "DFlash2DraftModel".into(),
            hidden: usize_field(c, "hidden_size")?,
            vocab: usize_field(c, "vocab_size")?,
            layers: vec![layer; usize_field(c, "num_hidden_layers")?],
            moe: None,
            speculator: None,
            tables: Vec::new(),
            vision: false,
            notes,
        })
    }

    fn classify(&self, _spec: &ModelSpec, _name: &str) -> Option<TensorRole> {
        // Every tensor belongs to the drafter; it runs on the coordinator.
        Some(TensorRole::new(Component::Speculator))
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        (component == Component::Speculator).then(|| Hint {
            what: "DFlash2 block drafter attached to a target engine".into(),
            how: "cuteafd branch work/glm-dflash2 drafts for GLM 5.3 (serve-glm --draft, native/cuda/kernels/\
                  glm_dflash.cu). This variant adds attention_conv/mlp_conv two-tap kernels and taps the mean of \
                  the target's mHC streams; license is CC-BY-NC-ND (check before shipping)."
                .into(),
        })
    }
}
