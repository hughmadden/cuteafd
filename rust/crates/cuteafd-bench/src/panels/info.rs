//! Hardware and Configuration: what the server runs on and how it is set up.
//! Instant passes; their records are the report's server info.
use super::{Ctx, Panel, Rates};
use crate::report::ServerInfo;
use serde_json::{json, Value};

pub struct Hardware;
pub struct ConfigurationPanel;
pub static HARDWARE: Hardware = Hardware;
pub static CONFIGURATION: ConfigurationPanel = ConfigurationPanel;

impl Panel for Hardware {
    fn id(&self) -> &'static str { "hardware" }
    fn title(&self) -> &'static str { "Hardware" }
    fn description(&self) -> &'static str {
        "GPUs (model, count, power cap, SMs, memory), Sparks, fabric rails and link rates, driver and CUDA."
    }
    fn estimate_s(&self, _rates: &Rates, _info: &ServerInfo) -> f64 { 0.2 }
    fn run(&self, ctx: &Ctx<'_>) -> anyhow::Result<Value> {
        Ok(json!({"hardware": ctx.info.hardware, "readiness_s": ctx.info.readiness_s}))
    }
}

impl Panel for ConfigurationPanel {
    fn id(&self) -> &'static str { "configuration" }
    fn title(&self) -> &'static str { "Configuration" }
    fn description(&self) -> &'static str {
        "Model, checkpoint revision, quantization per tensor group, speculator, layout, options that differ from \
         the defaults, and the build."
    }
    fn estimate_s(&self, _rates: &Rates, _info: &ServerInfo) -> f64 { 0.2 }
    fn run(&self, ctx: &Ctx<'_>) -> anyhow::Result<Value> {
        Ok(json!({"model": ctx.info.model, "family": ctx.info.family, "revision": ctx.info.revision,
            "configuration": ctx.info.configuration, "build": ctx.info.build}))
    }
}
