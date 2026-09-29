//! MiMo V2.6 Pro's fused `self_attn.qkv_proj`: FP8 E4M3 rows stored
//! interleaved for the checkpoint's tensor-parallel degree (`tp_size` in the
//! index metadata, 8). Each of the `tp` row shards is `[q | k | v]` of its
//! heads (`heads/tp` query heads, `max(1, kv/tp)` KV heads) with its own
//! 128x128 FP32 block grid, so a shard's 192-row key is a 128-row block then
//! a 64-row block. SGLang's loader (`load_mimo_v2_qkv_proj_weight`,
//! `_deinterleave_qkv_shards`) reads it the same way; the reference modeling
//! code splits the de-interleaved `[q; k; v]`.
use super::config::{MimoAttention, MimoV2Config};
use anyhow::{ensure, Context, Result};
use std::path::Path;

/// One run of rows that shares uniform 128-row scale blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkvSegment {
    /// First row in the checkpoint tensor.
    pub source_row: usize,
    /// First row of its 128-row blocks in the scale grid.
    pub scale_row: usize,
    /// First row in the de-interleaved `[q; k; v]`.
    pub dest_row: usize,
    pub rows: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedQkvLayout {
    pub shards: usize,
    /// Rows of q, k and v in one shard.
    pub q: usize,
    pub k: usize,
    pub v: usize,
}

impl FusedQkvLayout {
    pub fn new(cfg: &MimoV2Config, attention: MimoAttention, tp: usize) -> Result<Self> {
        let (heads, kv_heads) = (cfg.heads, cfg.kv_heads(attention));
        ensure!(tp > 0 && heads % tp == 0 && (kv_heads % tp == 0 || tp % kv_heads == 0),
            "{heads} query / {kv_heads} KV heads do not split over checkpoint TP {tp}");
        // SGLang replicates a KV head over the ranks sharing it (tp > kv): not stored that way here.
        ensure!(tp <= kv_heads, "fused qkv with replicated KV heads (checkpoint TP {tp} > {kv_heads}) is not supported");
        Ok(Self { shards: tp, q: heads / tp * cfg.head_dim, k: kv_heads / tp * cfg.head_dim,
            v: kv_heads / tp * cfg.v_head_dim })
    }

    pub fn rows(&self) -> usize {
        self.shards * (self.q + self.k + self.v)
    }

    pub fn scale_rows(&self) -> usize {
        self.shards * (self.q.div_ceil(128) + self.k.div_ceil(128) + self.v.div_ceil(128))
    }

    /// Every (shard, part) run, in checkpoint order.
    pub fn segments(&self) -> Vec<QkvSegment> {
        let (q_all, k_all) = (self.shards * self.q, self.shards * self.k);
        let mut out = Vec::with_capacity(3 * self.shards);
        let (mut source_row, mut scale_row) = (0, 0);
        for shard in 0..self.shards {
            for (rows, dest_row) in [(self.q, shard * self.q), (self.k, q_all + shard * self.k),
                (self.v, q_all + k_all + shard * self.v)] {
                out.push(QkvSegment { source_row, scale_row, dest_row, rows });
                source_row += rows;
                scale_row += rows.div_ceil(128);
            }
        }
        out
    }
}

/// The checkpoint's tensor-parallel degree (`metadata.tp_size` of the index; 1 when absent).
pub fn checkpoint_tp(snapshot: &Path) -> Result<usize> {
    let index = crate::plan::checkpoint::read_json(&snapshot.join("model.safetensors.index.json"))
        .context("reading model.safetensors.index.json")?;
    Ok(index["metadata"]["tp_size"].as_u64().map_or(1, |tp| tp as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pro_layout_matches_the_checkpoint_grid() -> Result<()> {
        let v = serde_json::json!({
            "model_type": "mimo_v2", "vocab_size": 152576, "hidden_size": 6144, "num_hidden_layers": 1,
            "num_attention_heads": 128, "num_key_value_heads": 8, "head_dim": 192, "v_head_dim": 128,
            "swa_num_key_value_heads": 8, "rope_theta": 1e7, "swa_rope_theta": 1e4, "sliding_window": 128,
            "hybrid_layer_pattern": [0], "moe_layer_freq": [0], "intermediate_size": 16384,
            "n_routed_experts": 384, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
        });
        let cfg = MimoV2Config::from_hf(&v)?;
        let layout = FusedQkvLayout::new(&cfg, MimoAttention::Full, 8)?;
        assert_eq!((layout.rows(), layout.scale_rows()), (27136, 216));
        let segments = layout.segments();
        assert_eq!(segments[1], QkvSegment { source_row: 3072, scale_row: 24, dest_row: 24576, rows: 192 });
        assert_eq!(segments[5], QkvSegment { source_row: 3392 + 3264, scale_row: 27 + 26, dest_row: 26112 + 128, rows: 128 });
        Ok(())
    }
}
