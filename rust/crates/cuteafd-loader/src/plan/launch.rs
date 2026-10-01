//! The launch description of a checkpoint: its family and the decoder layers
//! whose routed experts the Spark ranks serve, derived from the same
//! canonical configuration readers the runtimes use.
//! `scripts/lib/checkpoint-family.py` prints the same line for the launch
//! scripts (which run without a cuteafd binary); both are checked against
//! `tests/fixtures/launch-families.json`.
use anyhow::{bail, Result};
use serde::Serialize;
use serde_json::Value;

use crate::families::{glm5::GlmDsaConfig, glm5_flash::GlmNextConfig, mimo_v2::MimoV2Config, qwen4::Qwen4Config};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LaunchDescription {
    pub family: &'static str,
    pub model_type: String,
    /// First decoder layer with routed experts.
    pub first_expert_layer: usize,
    /// Last decoder layer the Spark ranks serve; `None` through the end
    /// (MTP/nextn layers included).
    pub last_expert_layer: Option<usize>,
}

impl LaunchDescription {
    /// `FAMILY MODEL_TYPE FIRST LAST` (LAST -1: through the end), as
    /// checkpoint-family.py prints it.
    pub fn line(&self) -> String {
        let last = self.last_expert_layer.map_or("-1".to_string(), |l| l.to_string());
        format!("{} {} {} {last}", self.family, self.model_type, self.first_expert_layer)
    }
}

/// The launch description of `config.json`.
pub fn describe(config: &Value) -> Result<LaunchDescription> {
    let model_type = config.get("model_type").and_then(Value::as_str).unwrap_or("?").to_owned();
    let (family, first, last) = match model_type.as_str() {
        "deepseek_v41" => ("deepseek_v41", 0, None),
        "deepseek_v4" => ("deepseek_v4", 0, None),
        "glm_moe_dsa" => ("glm5", GlmDsaConfig::from_hf(config)?.first_moe_layer, None),
        "glm5_next" => {
            let cfg = GlmNextConfig::from_hf(config)?;
            let first = cfg.dense.iter().position(|dense| !dense).unwrap_or(cfg.layers);
            // serve-glmf does not run the MTP layer: its experts stay off the Sparks.
            ("glm5_flash", first, Some(cfg.layers - 1))
        }
        "mimo_v2_flash" | "mimo_v2" => {
            let cfg = MimoV2Config::from_hf(config)?;
            ("mimo_v2", cfg.dense.iter().position(|dense| !dense).unwrap_or(cfg.layers), None)
        }
        "qwen4_exp" => {
            let cfg = Qwen4Config::from_hf(config)?;
            ("qwen4", 0, Some(cfg.layers - 1))
        }
        other => bail!("unsupported model_type {other:?}"),
    };
    Ok(LaunchDescription { family, model_type, first_expert_layer: first, last_expert_layer: last })
}
