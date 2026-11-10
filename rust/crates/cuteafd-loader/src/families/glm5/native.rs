//! GLM composition buffers beside the format-level native linear scratch.
use super::GlmDsaConfig;
use crate::plan::exlr8::dense::{DensePlan, DenseRecord, scratch::LinearScratchLayout};
use anyhow::{Context, Result};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct ScratchLayout {
    pub linear: LinearScratchLayout,
    pub stages: BTreeMap<&'static str, usize>,
}

impl ScratchLayout {
    pub fn new(cfg: &GlmDsaConfig, heads: usize, inter: usize, rows: usize,
        linear: LinearScratchLayout) -> Result<Self> {
        let bf16 = |width: usize| rows.checked_mul(width).and_then(|n| n.checked_mul(2))
            .map(|n| n.max(256)).context("GLM native stage overflow");
        let add = |a: usize, b| a.checked_add(b).context("GLM native stage width overflow");
        let mul = |a: usize, b| a.checked_mul(b).context("GLM native stage width overflow");
        Ok(Self { linear, stages: BTreeMap::from([
            ("qkv", bf16(add(add(cfg.q_lora_rank, cfg.kv_lora_rank)?, cfg.qk_rope_head_dim)?)?),
            ("q", bf16(mul(heads, add(cfg.qk_nope_head_dim, cfg.qk_rope_head_dim)?)?)?),
            ("gate", bf16(inter)?), ("up", bf16(inter)?), ("hidden", bf16(inter)?),
            ("kw", bf16(add(cfg.index_head_dim, cfg.index_heads)?)?),
            ("iq", bf16(mul(cfg.index_heads, cfg.index_head_dim)?)?),
            ("values", bf16(mul(heads, cfg.v_head_dim)?)?),
        ]) })
    }
    pub fn bytes(&self) -> Result<usize> {
        self.stages.values().try_fold(self.linear.bytes()?, |sum, &bytes|
            sum.checked_add(bytes).context("GLM native scratch total overflow"))
    }
}

pub fn scratch_layouts(cfg: &GlmDsaConfig, records: &[DenseRecord], plan: &DensePlan,
    decode_rows: usize, prefill_rows: usize, a8: bool) -> Result<[ScratchLayout; 2]> {
    let (heads, inter) = (cfg.heads / plan.world, cfg.dense_intermediate / plan.world);
    Ok([
        ScratchLayout::new(cfg, heads, inter, decode_rows,
            LinearScratchLayout::from_plan(records, plan, decode_rows, false)?)?,
        ScratchLayout::new(cfg, heads, inter, prefill_rows,
            LinearScratchLayout::from_plan(records, plan, prefill_rows, a8)?)?,
    ])
}

pub fn scratch_bytes(layouts: &[ScratchLayout; 2]) -> Result<u64> {
    u64::try_from(layouts[0].bytes()?.checked_add(layouts[1].bytes()?)
        .context("GLM native decode/prefill scratch overflow")?).context("GLM scratch exceeds u64")
}
