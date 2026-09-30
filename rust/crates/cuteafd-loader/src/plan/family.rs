//! Family registry: detection, spec derivation and tensor classification.
use anyhow::Result;

use super::checkpoint::Checkpoint;
use super::format::WeightFormat;
use super::spec::{Component, ModelSpec, TensorRole};

/// Where a family's implementation stands in this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStatus {
    /// Serves end to end.
    Serving,
    /// Recognized and planned; the execution path is not written yet.
    Planned,
}

/// Guidance a code agent needs to make one requirement work.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Hint {
    pub what: String,
    pub how: String,
}

pub trait Family: Sync {
    fn id(&self) -> &'static str;
    fn detect(&self, checkpoint: &Checkpoint) -> bool;
    fn runtime(&self) -> RuntimeStatus;
    fn spec(&self, checkpoint: &Checkpoint) -> Result<ModelSpec>;
    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole>;
    /// Whether this build executes `component` stored as `format`.
    fn executes(&self, _component: Component, _format: &WeightFormat) -> bool {
        false
    }
    /// Implementation notes for components this build cannot execute yet.
    fn component_hint(&self, component: Component) -> Option<Hint>;

    /// `component_hint` knowing the checkpoint's spec and the component's
    /// detected formats (labels); families whose checkpoints differ in
    /// storage (MiMo V2 Flash FP8 vs V2.6 Pro MXFP4) override it.
    fn component_hint_for(&self, _spec: &ModelSpec, component: Component, _formats: &[String]) -> Option<Hint> {
        self.component_hint(component)
    }

    /// Components the engine can serve without (a speculator it does not run yet).
    fn optional(&self, _component: Component) -> bool {
        false
    }
}

static REGISTRY: [&dyn Family; 7] = [
    &super::families::deepseek::DEEPSEEK_V41,
    &super::families::deepseek::DEEPSEEK_V4,
    &super::families::glm::GLM5,
    &super::families::glm::GLM5_FLASH,
    &super::families::mimo::MIMO_V2,
    &super::families::qwen::QWEN4,
    &super::families::dflash::DFLASH2,
];

pub fn registry() -> &'static [&'static dyn Family] {
    &REGISTRY
}

pub fn detect(checkpoint: &Checkpoint) -> Option<&'static dyn Family> {
    registry().iter().copied().find(|family| family.detect(checkpoint))
}
