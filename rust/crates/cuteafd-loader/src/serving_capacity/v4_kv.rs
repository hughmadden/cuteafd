//! The single physical V4 pool request, shared by admission and execution.
use crate::families::deepseek_v4::DeepseekV4Config;
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct V4KvLayout {
    bytes: usize,
    views: BTreeMap<String, std::ops::Range<usize>>,
}

impl V4KvLayout {
    /// Every record is a multiple of 64 bytes. Keeping that alignment (above
    /// the kernels' 16-byte requirement) preserves an affine pool-size cost.
    #[allow(clippy::too_many_arguments)]
    pub fn new(cfg: &DeepseekV4Config, sequences: usize, ring_pages: usize, units: usize,
        context: bool, stages: usize, positions: usize, prefill_waves: usize, decode_rows: usize) -> Result<Self> {
        ensure!(sequences > 0 && ring_pages > 0 && units > 0 && prefill_waves > 0 && decode_rows > 0,
            "empty V4 KV geometry");
        ensure!(!context || units % 2 == 0, "V4 context pool needs even units");
        let mut layout = Self { bytes: 0, views: BTreeMap::new() };
        let mul = |a: usize, b: usize| a.checked_mul(b).context("V4 KV extent overflow");
        let c4_units = if context { units / 2 + 1 } else { units };
        for layer in 0..cfg.n_layers.checked_add(stages).context("V4 cached layer overflow")? {
            layout.push(format!("layer{layer}.main"), mul(mul(sequences, ring_pages)?, 149_760)?)?;
            let ratio = cfg.compress_ratios.get(layer).copied().unwrap_or(0);
            let carry: &[usize] = match ratio {
                4 => {
                    layout.push(format!("layer{layer}.compressed"), mul(c4_units, 37_440)?)?;
                    layout.push(format!("layer{layer}.index"), mul(c4_units, 8_448)?)?;
                    &[16 * 1024 * 4, 16 * 1024 * 4, 16 * 256 * 4, 16 * 256 * 4]
                }
                128 => {
                    layout.push(format!("layer{layer}.compressed"), mul(units, 1_728)?)?;
                    &[256 * 512 * 4, 256 * 512 * 4]
                }
                0 => &[],
                _ => anyhow::bail!("unsupported V4 compression ratio {ratio}"),
            };
            for (index, &bytes) in carry.iter().enumerate() {
                layout.push(format!("layer{layer}.carry{index}"), mul(sequences, bytes)?)?;
            }
        }
        let rope = mul(mul(positions.max(128), cfg.rope_head_dim)?, 4)?;
        layout.push("rope.window".into(), rope)?;
        layout.push("rope.compressed".into(), rope)?;
        for lane in 0..prefill_waves {
            layout.push(format!("tables.prefill{lane}"), mul(mul(super::V4_PACKED_MEMBERS,
                units.checked_add(3).context("V4 KV table overflow")?)?, 4)?)?;
        }
        // Last: arbitrary decode row counts need no alignment of a following view.
        layout.push("tables.decode".into(), mul(mul(decode_rows, units)?, 4)?)?;
        Ok(layout)
    }

    fn push(&mut self, name: String, bytes: usize) -> Result<()> {
        let bytes = bytes.max(256);
        let start = self.bytes.checked_add(63).context("V4 KV alignment overflow")? & !63;
        self.bytes = start.checked_add(bytes).context("V4 KV extent overflow")?;
        ensure!(self.views.insert(name, start..self.bytes).is_none(), "duplicate V4 KV view");
        Ok(())
    }

    pub fn bytes(&self) -> usize { self.bytes }
    pub fn views(&self) -> &BTreeMap<String, std::ops::Range<usize>> { &self.views }
    pub fn view(&self, name: &str) -> Result<std::ops::Range<usize>> {
        self.views.get(name).cloned().with_context(|| format!("missing V4 KV view {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_views_have_one_disjoint_backing_and_preserve_payloads() -> Result<()> {
        let cfg = DeepseekV4Config::from_model_args(&serde_json::json!({
            "vocab_size":129280,"dim":4096,"moe_inter_dim":2048,"n_layers":3,"n_hash_layers":1,
            "n_heads":64,"n_routed_experts":256,"n_shared_experts":1,"n_activated_experts":6,
            "score_func":"sqrtsoftplus","route_scale":1.5,"swiglu_limit":10.0,"q_lora_rank":1024,
            "head_dim":512,"rope_head_dim":64,"o_groups":8,"o_lora_rank":1024,"window_size":128,
            "original_seq_len":65536,"rope_theta":10000,"rope_factor":16,"beta_fast":32,"beta_slow":1,
            "index_n_heads":64,"index_head_dim":128,"index_topk":512,"hc_mult":4,"hc_sinkhorn_iters":20,
            "compress_rope_theta":160000,"compress_ratios":[0,4,128]
        }), 3)?;
        for context in [false, true] {
            let layout = V4KvLayout::new(&cfg, 16, 17, 8194, context, 3, 131072, 3, 64)?;
            let mut views = layout.views().values().cloned().collect::<Vec<_>>();
            views.sort_by_key(|r| r.start);
            assert!(views.iter().all(|r| r.start % 64 == 0));
            assert!(views.windows(2).all(|w| w[0].end <= w[1].start));
            assert_eq!(views.last().unwrap().end, layout.bytes());
            assert_eq!(layout.view("layer1.compressed")?.len(), if context { 4098 } else { 8194 } * 37_440);
            assert_eq!(layout.view("layer2.compressed")?.len(), 8194 * 1_728);
            assert_eq!(layout.view("layer5.main")?.len(), 16 * 17 * 149_760);
            assert_eq!(layout.view("tables.prefill2")?.len(), 16 * (8194 + 3) * 4);
            assert_eq!(layout.view("tables.decode")?.len(), 64 * 8194 * 4);
        }
        Ok(())
    }
}
