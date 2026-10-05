//! MiMo V2.6 fused `self_attn.qkv_proj`: FP8 E4M3 rows interleaved for
//! `metadata.tp_size` in the checkpoint index (Flash TP4, Pro TP8).
//! Every shard stores `[q | k | v]` with an independent 128x128 FP32 scale
//! grid for each segment. Flash SWA shards contain two contiguous key heads;
//! Pro has one key head per shard and pads it only in the program layout.
//! The reference modeling code splits the de-interleaved `[q; k; v]`.
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

    /// Projection rows and copy runs for the selected program. Flash keeps
    /// multi-head checkpoint shards contiguous; only Pro pads one-head keys.
    pub fn program_segments(&self, cfg: &MimoV2Config) -> Result<(usize, Vec<QkvSegment>)> {
        let stride = cfg.qkv_key_stride();
        if stride == cfg.head_dim {
            return Ok((self.rows(), self.segments()));
        }
        ensure!(self.k == cfg.head_dim && stride % 128 == 0 && self.q % 128 == 0 && self.v % 128 == 0,
            "padded keys (stride {stride}) need one KV head per checkpoint shard and 128-row q/v shards");
        Ok((self.padded_rows(stride), self.segments_with_key_stride(stride)))
    }

    pub fn rows(&self) -> usize {
        self.shards * (self.q + self.k + self.v)
    }

    pub fn scale_rows(&self) -> usize {
        self.shards * (self.q.div_ceil(128) + self.k.div_ceil(128) + self.v.div_ceil(128))
    }

    /// Every (shard, part) run, in checkpoint order.
    pub fn segments(&self) -> Vec<QkvSegment> {
        self.segments_with_key_stride(self.k)
    }

    /// Rows of the de-interleaved `[q; k; v]` whose keys sit `key_stride` rows
    /// apart (`key_stride` >= a shard's key rows; the gap is zero padding).
    pub fn padded_rows(&self, key_stride: usize) -> usize {
        self.shards * (self.q + key_stride + self.v)
    }

    /// `segments` into the layout whose shard keys sit `key_stride` rows apart:
    /// with one KV head per shard and `key_stride` a multiple of 128, every
    /// checkpoint 128x128 block lands on a whole 128-row block (V2.6 Pro: 256).
    pub fn segments_with_key_stride(&self, key_stride: usize) -> Vec<QkvSegment> {
        let (q_all, k_all) = (self.shards * self.q, self.shards * key_stride);
        let mut out = Vec::with_capacity(3 * self.shards);
        let (mut source_row, mut scale_row) = (0, 0);
        for shard in 0..self.shards {
            for (rows, dest_row) in [(self.q, shard * self.q), (self.k, q_all + shard * key_stride),
                (self.v, q_all + k_all + shard * self.v)] {
                out.push(QkvSegment { source_row, scale_row, dest_row, rows });
                source_row += rows;
                scale_row += rows.div_ceil(128);
            }
        }
        out
    }
}

/// The index's positive integer `metadata.tp_size`; required for fused QKV.
pub fn checkpoint_tp(snapshot: &Path) -> Result<usize> {
    let index = crate::plan::checkpoint::read_json(&snapshot.join("model.safetensors.index.json"))
        .context("reading model.safetensors.index.json")?;
    let fused = index["weight_map"].as_object().is_some_and(|weights|
        weights.keys().any(|name| name.ends_with("self_attn.qkv_proj.weight")));
    let raw = &index["metadata"]["tp_size"];
    if raw.is_null() && !fused { return Ok(1); }
    let tp = raw.as_u64().filter(|&tp| tp > 0)
        .context("fused qkv_proj requires positive integer metadata.tp_size in the checkpoint index")?;
    usize::try_from(tp).context("metadata.tp_size does not fit usize")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_tp4_keeps_two_swa_key_heads_contiguous_and_scales_segmented() -> Result<()> {
        let mut value = crate::plan::testing::mimo_flash_config();
        value["model_type"] = serde_json::json!("mimo_v2");
        value["rope_theta"] = serde_json::json!(1e7);
        value["layernorm_epsilon"] = serde_json::json!(1e-6);
        value["moe_router_dtype"] = serde_json::json!("bfloat16");
        let cfg = MimoV2Config::from_hf(&value)?;
        for (kind, rows, scales, key_rows) in [(MimoAttention::Full, 13568, 108, 192),
            (MimoAttention::Sliding, 14848, 116, 384)] {
            let layout = FusedQkvLayout::new(&cfg, kind, 4)?;
            let (width, segments) = layout.program_segments(&cfg)?;
            assert_eq!((width, layout.scale_rows(), layout.k), (rows, scales, key_rows));
            // Label every checkpoint row by its own restarted scale-grid row,
            // and prove de-interleaving leaves no holes or duplicate writes.
            let mut dest = vec![usize::MAX; width];
            for segment in &segments {
                for row in 0..segment.rows {
                    let at = segment.dest_row + row;
                    assert_eq!(dest[at], usize::MAX);
                    dest[at] = segment.scale_row + row / 128;
                }
            }
            assert!(dest.iter().all(|&row| row < scales));
            let q_all = 64 * 192;
            for shard in 0..4 {
                let key = segments[shard * 3 + 1];
                assert_eq!(key.dest_row, q_all + shard * key_rows);
                assert_eq!(dest[key.dest_row + key_rows - 1], key.scale_row + (key_rows - 1) / 128);
            }
            let share = cfg.head_split(2)?;
            let half = FusedQkvLayout::new(&share, kind, 2)?;
            assert_eq!((half.program_segments(&share)?.0 * 2, half.scale_rows() * 2), (width, scales));
            for (whole, split) in segments[..6].iter().zip(half.segments()) {
                assert_eq!((whole.source_row, whole.scale_row, whole.rows), (split.source_row, split.scale_row, split.rows));
            }
        }
        Ok(())
    }

    #[test]
    fn fused_checkpoint_requires_positive_integer_tp_metadata() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model.safetensors.index.json");
        for tp in [serde_json::Value::Null, serde_json::json!(0), serde_json::json!("4"),
            serde_json::json!(-1), serde_json::json!(4.5)] {
            let index = serde_json::json!({"metadata": {"tp_size": tp}, "weight_map": {
                "model.layers.0.self_attn.qkv_proj.weight": "model.safetensors"}});
            std::fs::write(&path, serde_json::to_vec(&index)?)?;
            assert!(checkpoint_tp(dir.path()).unwrap_err().to_string().contains("metadata.tp_size"));
        }
        std::fs::write(&path, serde_json::to_vec(&serde_json::json!({"metadata": {"tp_size": 4},
            "weight_map": {"model.layers.0.self_attn.qkv_proj.weight": "model.safetensors"}}))?)?;
        assert_eq!(checkpoint_tp(dir.path())?, 4);
        std::fs::write(&path, b"{\"weight_map\": {}}")?;
        assert_eq!(checkpoint_tp(dir.path())?, 1);
        Ok(())
    }

    #[test]
    fn pro_layout_matches_the_checkpoint_grid() -> Result<()> {
        let v = serde_json::json!({
            "model_type": "mimo_v2", "vocab_size": 152576, "hidden_size": 6144, "num_hidden_layers": 1,
            "num_attention_heads": 128, "num_key_value_heads": 8, "head_dim": 192, "v_head_dim": 128,
            "swa_num_key_value_heads": 8, "rope_theta": 1e7, "swa_rope_theta": 1e4, "sliding_window": 128,
            "hybrid_layer_pattern": [0], "moe_layer_freq": [0], "intermediate_size": 16384,
            "n_routed_experts": 384, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
            "attention_value_scale": 0.612,
        });
        let cfg = MimoV2Config::from_hf(&v)?;
        let layout = FusedQkvLayout::new(&cfg, MimoAttention::Full, 8)?;
        assert_eq!((layout.rows(), layout.scale_rows()), (27136, 216));
        let segments = layout.segments();
        assert_eq!(segments[1], QkvSegment { source_row: 3072, scale_row: 24, dest_row: 24576, rows: 192 });
        assert_eq!(segments[5], QkvSegment { source_row: 3392 + 3264, scale_row: 27 + 26, dest_row: 26112 + 128, rows: 128 });
        // A two-GPU head split: each GPU takes four whole checkpoint shards (64 query heads,
        // 4 KV heads), the same per-shard layout at half the rows and grid rows.
        let share = cfg.head_split(2)?;
        assert_eq!((share.heads, share.full_kv_heads, share.dense_intermediate), (64, 4, 8192));
        assert_eq!((share.program_family()?, share.qkv_key_stride()), ("mimop2", 256));
        let half = FusedQkvLayout::new(&share, MimoAttention::Full, 4)?;
        assert_eq!((half.rows() * 2, half.scale_rows() * 2), (layout.rows(), layout.scale_rows()));
        // Rank 0's sources are the first four shards' (destinations follow the half layout).
        let source = |s: &[QkvSegment]| s.iter().map(|s| (s.source_row, s.scale_row, s.rows)).collect::<Vec<_>>();
        assert_eq!(source(&half.segments()), source(&segments[..12]));
        assert_eq!(half.segments()[1].dest_row, 64 * 192);
        assert!(cfg.head_split(3).is_err());
        Ok(())
    }
}
