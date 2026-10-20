//! Checkpoint-native MLA DFlash2 geometry shared by admission and loading.
use super::{block_dspark::{DsparkConfigError, DsparkRope, DsparkRotaryPlan}, mla::MlaGeometry};
use serde::Deserialize;
use std::path::Path;

type Result<T> = std::result::Result<T, DsparkConfigError>;

#[derive(Debug, Clone, PartialEq)]
pub struct MlaDflashConfig {
    pub mla: MlaGeometry,
    pub intermediate: usize,
    pub eps: f64,
    pub block: usize,
    pub group: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub taps: Vec<usize>,
    pub vocab: usize,
    pub mask_token: u32,
    pub max_context: usize,
    /// None means full context; each sliding layer keeps its own exact window.
    pub windows: Vec<Option<usize>>,
    pub rotary: DsparkRotaryPlan,
}

/// One layer's slot-isolated physical cache and page-table capacity. The
/// sliding ring includes an admitted private block; committed rows stay intact
/// until verification decides which rows to retain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftCacheLayer {
    slots: usize,
    context: usize,
    block: usize,
    page_rows: usize,
    window: Option<usize>,
    rows_per_slot: usize,
    table_width: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftCacheLayout {
    pub layers: Vec<MlaDraftCacheLayer>,
    pub cache_bytes: u64,
    pub page_table_bytes: u64,
    /// Physical padding and private block rows above retained-record capacity.
    pub margin_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaDraftCacheView {
    /// Page-aligned absolute start; kernel lengths/queries are relative to it.
    pub base_position: usize,
    pub cache_seqlen: usize,
    pub pages: Vec<i32>,
}

impl MlaDraftCacheLayer {
    pub fn rows_per_slot(&self) -> usize { self.rows_per_slot }
    pub fn table_width(&self) -> usize { self.table_width }
    pub fn slots(&self) -> usize { self.slots }

    /// Absolute positions keep RoPE unchanged; only storage addresses wrap.
    pub fn row(&self, slot: usize, position: usize) -> Result<usize> {
        if slot >= self.slots || position >= self.context {
            return Err(unsupported("MLA draft row outside admitted slot/context"));
        }
        Ok(slot * self.rows_per_slot + position % self.rows_per_slot)
    }

    /// No state is published here. A rejected block can be overwritten without
    /// rolling back the committed ring or changing its live frontier.
    pub fn block_view(&self, slot: usize, anchor: usize, rows: usize) -> Result<MlaDraftCacheView> {
        let end = anchor.checked_add(rows).ok_or_else(|| unsupported("MLA draft block overflow"))?;
        if slot >= self.slots || rows == 0 || rows > self.block || end > self.context {
            return Err(unsupported("MLA draft block outside admitted slot/context"));
        }
        let begin = self.window.map_or(0, |window| anchor.saturating_sub(window));
        let base_position = begin / self.page_rows * self.page_rows;
        let pages_per_slot = self.rows_per_slot / self.page_rows;
        let first = base_position / self.page_rows;
        let count = (end - base_position).div_ceil(self.page_rows);
        if count > self.table_width {
            return Err(unsupported("MLA draft view exceeds admitted page table"));
        }
        // An unaligned live span may name the same physical page at both ends;
        // attention masks the stale prefix and suffix of those virtual pages.
        let pages = (first..first + count).map(|page|
            (slot * pages_per_slot + page % pages_per_slot) as i32).collect();
        Ok(MlaDraftCacheView { base_position, cache_seqlen: end - base_position, pages })
    }
}

#[derive(Deserialize)]
struct Raw {
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    rms_norm_eps: f64,
    vocab_size: usize,
    draft_vocab_size: usize,
    max_position_embeddings: usize,
    num_target_layers: usize,
    layer_types: Vec<String>,
    sliding_window: Option<usize>,
    use_sliding_window: bool,
    is_causal: bool,
    mla_use_output_gate: bool,
    attention_bias: bool,
    hidden_act: String,
    fc_norm: bool,
    rope_parameters: DsparkRope,
    dflash_config: Draft,
}

#[derive(Deserialize)]
struct Draft {
    attention_mode: String,
    block_size: usize,
    conv_kernel_size: usize,
    conv_group_size: usize,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
    mask_token_id: u32,
}

fn unsupported(message: &str) -> DsparkConfigError {
    DsparkConfigError::Unsupported(message.into())
}

impl MlaDflashConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_json(&serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?)
    }

    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let raw: Raw = serde_json::from_value(value.clone())?;
        let d = raw.dflash_config;
        if !raw.architectures.iter().any(|a| a == "DFlash2DraftModel")
            || d.attention_mode != "mla" || raw.is_causal || raw.mla_use_output_gate
            || raw.attention_bias || raw.hidden_act != "silu" || raw.fc_norm {
            return Err(unsupported("MLA DFlash2 requires noncausal, ungated, bias-free SiLU and fc_norm=false"));
        }
        let mla = MlaGeometry { hidden: raw.hidden_size, heads: raw.num_attention_heads,
            q_rank: raw.q_lora_rank, kv_rank: raw.kv_lora_rank, nope: raw.qk_nope_head_dim,
            rope: raw.qk_rope_head_dim, value: raw.v_head_dim };
        if mla != (MlaGeometry { hidden: 7168, heads: 64, q_rank: 1536,
            kv_rank: 512, nope: 128, rope: 64, value: 128 })
            || raw.num_key_value_heads != mla.heads || raw.rms_norm_eps != 1e-5 {
            return Err(unsupported("MLA draft exports require H7168, 64 heads, Q1536/KV512, NoPE128/RoPE64/V128, eps=1e-5"));
        }
        if raw.intermediate_size == 0 || raw.intermediate_size > 1_000_000
            || raw.intermediate_size % 128 != 0 || raw.num_hidden_layers == 0
            || raw.num_hidden_layers > 1024 || raw.vocab_size == 0 || raw.vocab_size > 1_000_000
            || raw.vocab_size != raw.draft_vocab_size || d.mask_token_id as usize >= raw.vocab_size
            || raw.max_position_embeddings == 0 || raw.max_position_embeddings > 1_048_576
            || raw.num_target_layers == 0 || raw.num_target_layers > 1_000_000
            || d.target_layer_ids.is_empty() || d.target_layer_ids.len() > 1024
            || d.target_layer_ids.iter().any(|&tap| tap >= raw.num_target_layers)
            || d.target_layer_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(unsupported("invalid MLA draft dimensions, context, vocabulary or ordered post-layer taps"));
        }
        if d.block_size != 8 || d.conv_kernel_size != 2 || d.conv_group_size != 16
            || d.selector_rank != 256 || d.selector_top_k != 16 {
            return Err(unsupported("DFlash2 requires block8, two-tap group16 convolutions, rank256/top16 selector"));
        }
        if raw.layer_types.len() != raw.num_hidden_layers {
            return Err(unsupported("MLA draft layer_types count differs from num_hidden_layers"));
        }
        let windows = raw.layer_types.iter().map(|kind| match kind.as_str() {
            "full_attention" => Ok(None),
            "sliding_attention" if raw.use_sliding_window && raw.sliding_window == Some(4096) => Ok(Some(4096)),
            _ => Err(unsupported("MLA draft exports support full attention or a native 4096-token window")),
        }).collect::<Result<Vec<_>>>()?;
        let rotary = raw.rope_parameters.plan(mla.rope)?;
        Ok(Self { mla, intermediate: raw.intermediate_size, eps: raw.rms_norm_eps,
            block: d.block_size, group: d.conv_group_size, selector_rank: d.selector_rank,
            selector_top_k: d.selector_top_k, taps: d.target_layer_ids, vocab: raw.vocab_size,
            mask_token: d.mask_token_id, max_context: raw.max_position_embeddings, windows, rotary })
    }

    pub fn attention_scale(&self) -> f32 {
        ((self.mla.nope + self.mla.rope) as f64).sqrt().recip() as f32 * self.rotary.softmax_factor
    }

    /// Exact BF16 source shapes. Embedding and vocabulary head are borrowed
    /// from the target, matching the checkpoint's reference execution.
    pub fn tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let h = self.mla.hidden;
        let mut shapes = vec![
            ("fc.weight".into(), vec![h, self.taps.len() * h]),
            ("hidden_norm.weight".into(), vec![h]),
            ("norm.weight".into(), vec![h]),
            ("candidate_selector.hidden_projection.weight".into(), vec![self.selector_rank, h]),
            ("candidate_selector.predecessor_codebook".into(), vec![self.vocab, self.selector_rank]),
            ("candidate_selector.successor_codebook".into(), vec![self.vocab, self.selector_rank]),
        ];
        for layer in 0..self.windows.len() {
            let p = format!("layers.{layer}");
            for leaf in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                shapes.push((format!("{p}.{leaf}"), vec![h]));
            }
            for conv in ["attention_conv", "mlp_conv"] {
                shapes.push((format!("{p}.{conv}.kernel_projection.weight"), vec![4 * h / self.group, h]));
                shapes.push((format!("{p}.{conv}.base_kernel"), vec![2, 2, h]));
            }
            for (leaf, shape) in [
                ("q_a_proj", vec![self.mla.q_rank, h]),
                ("kv_a_proj_with_mqa", vec![self.mla.kv_rank + self.mla.rope, h]),
                ("q_a_layernorm", vec![self.mla.q_rank]),
                ("kv_a_layernorm", vec![self.mla.kv_rank]),
                ("q_b_proj", vec![self.mla.heads * (self.mla.nope + self.mla.rope), self.mla.q_rank]),
                ("kv_b_proj", vec![self.mla.heads * (self.mla.nope + self.mla.value), self.mla.kv_rank]),
                ("o_proj", vec![h, self.mla.heads * self.mla.value]),
            ] {
                shapes.push((format!("{p}.self_attn.{leaf}.weight"), shape));
            }
            for (leaf, shape) in [
                ("gate_proj", vec![self.intermediate, h]),
                ("up_proj", vec![self.intermediate, h]),
                ("down_proj", vec![h, self.intermediate]),
            ] {
                shapes.push((format!("{p}.mlp.{leaf}.weight"), shape));
            }
        }
        shapes
    }

    pub fn bf16_weight_bytes(&self) -> Result<u64> {
        self.tensor_shapes().iter().try_fold(0u64, |total, (_, shape)| {
            let bytes = shape.iter().try_fold(2u64, |n, &d| n.checked_mul(d as u64))
                .ok_or_else(|| unsupported("MLA draft weight byte overflow"))?;
            total.checked_add(bytes).ok_or_else(|| unsupported("MLA draft weight total overflow"))
        })
    }

    /// Cache admission shared by planner and runtime. Full layers admit the
    /// selected context (blocks must end inside it); sliding layers additionally
    /// admit a private block and page-offset slack in their virtual tables.
    pub fn cache_layout(&self, slots: usize, context: usize) -> Result<MlaDraftCacheLayout> {
        const PAGE: usize = 64;
        if slots == 0 || context == 0 || context > self.max_context || self.block == 0 {
            return Err(unsupported("MLA draft cache slots/context outside checkpoint extent"));
        }
        let overflow = || unsupported("MLA draft cache admission overflow");
        let record = self.mla.kv_rank.checked_add(self.mla.rope)
            .and_then(|n| n.checked_mul(2)).ok_or_else(overflow)? as u64;
        let mut layers = Vec::with_capacity(self.windows.len());
        let (mut cache_bytes, mut page_table_bytes, mut margin_bytes) = (0u64, 0u64, 0u64);
        for &window in &self.windows {
            if window == Some(0) { return Err(unsupported("MLA draft window must be nonzero")); }
            let retained = window.unwrap_or(context).min(context);
            let live = match window {
                Some(w) => w.checked_add(self.block).ok_or_else(overflow)?.min(context),
                None => context,
            };
            let rows_per_slot = live.checked_add(PAGE - 1).ok_or_else(overflow)? / PAGE * PAGE;
            let table_width = match window {
                Some(_) => live.checked_add(PAGE - 1).ok_or_else(overflow)?.div_ceil(PAGE),
                None => rows_per_slot / PAGE,
            };
            let total_rows = rows_per_slot.checked_mul(slots).ok_or_else(overflow)?;
            if total_rows / PAGE > i32::MAX as usize || total_rows > i64::MAX as usize {
                return Err(unsupported("MLA draft cache exceeds native page/row indices"));
            }
            let bytes = (total_rows as u64).checked_mul(record).ok_or_else(overflow)?;
            let margin = ((rows_per_slot - retained) as u64).checked_mul(slots as u64)
                .and_then(|n| n.checked_mul(record)).ok_or_else(overflow)?;
            let tables = table_width.checked_mul(slots).and_then(|n| n.checked_mul(4))
                .ok_or_else(overflow)? as u64;
            cache_bytes = cache_bytes.checked_add(bytes).ok_or_else(overflow)?;
            margin_bytes = margin_bytes.checked_add(margin).ok_or_else(overflow)?;
            page_table_bytes = page_table_bytes.checked_add(tables).ok_or_else(overflow)?;
            layers.push(MlaDraftCacheLayer { slots, context, block: self.block, page_rows: PAGE,
                window, rows_per_slot, table_width });
        }
        Ok(MlaDraftCacheLayout { layers, cache_bytes, page_table_bytes, margin_bytes })
    }

    /// Latent BF16 records only, not block/workspace/staging admission. Full
    /// layers grow to the selected context; sliding layers do not silently grow.
    pub fn cache_bytes(&self, slots: usize, context: usize) -> Result<u64> {
        if slots == 0 || context == 0 || context > self.max_context {
            return Err(unsupported("MLA draft cache slots/context outside checkpoint extent"));
        }
        let rows = self.windows.iter().try_fold(0u64, |sum, window| {
            sum.checked_add(window.unwrap_or(context) as u64)
                .ok_or_else(|| unsupported("MLA draft cache row overflow"))
        })?;
        rows.checked_mul(slots as u64).and_then(|n| n.checked_mul((self.mla.kv_rank + self.mla.rope) as u64))
            .and_then(|n| n.checked_mul(2)).ok_or_else(|| unsupported("MLA draft cache byte overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> serde_json::Value {
        json!({"architectures":["DFlash2DraftModel"], "hidden_size":7168,
            "intermediate_size":14336, "num_hidden_layers":5, "num_attention_heads":64,
            "num_key_value_heads":64, "q_lora_rank":1536, "kv_lora_rank":512,
            "qk_nope_head_dim":128, "qk_rope_head_dim":64, "v_head_dim":128,
            "rms_norm_eps":1e-5, "vocab_size":163840, "draft_vocab_size":163840,
            "max_position_embeddings":1048576, "num_target_layers":93,
            "layer_types":["sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","full_attention"], "sliding_window":4096,
            "use_sliding_window":true, "is_causal":false, "mla_use_output_gate":false,
            "attention_bias":false, "hidden_act":"silu", "fc_norm":false,
            "rope_parameters":{"rope_type":"yarn", "rope_theta":50000.0, "factor":32.0,
                "original_max_position_embeddings":32768, "beta_fast":32, "beta_slow":1,
                "mscale":1.0, "mscale_all_dim":1.0},
            "dflash_config":{"attention_mode":"mla", "block_size":8,
                "conv_kernel_size":2, "conv_group_size":16, "selector_rank":256,
                "selector_top_k":16, "target_layer_ids":[19,37,66,78,90], "mask_token_id":163592}})
    }

    #[test]
    fn native_mla_weights_and_full_sliding_cache_are_exact() {
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        assert_eq!(cfg.windows, vec![Some(4096), Some(4096), Some(4096), Some(4096), None]);
        assert_eq!(cfg.taps, [19,37,66,78,90]);
        assert_eq!(cfg.bf16_weight_bytes().unwrap(), 5_036_945_408);
        assert!(!cfg.tensor_shapes().iter().any(|(name, _)| name.contains("embed_tokens") || name.contains("lm_head")));
        assert_eq!(cfg.cache_bytes(8, 262144).unwrap(), 2_566_914_048);
        assert_eq!(cfg.cache_bytes(8, 1048576).unwrap(), 9_814_671_360);
        assert_eq!(cfg.rotary.attention_factor, 1.0);
        let expected = (192.0f64).sqrt().recip() * (1.0 + 0.1 * 32.0f64.ln()).powi(2);
        assert!((cfg.attention_scale() as f64 - expected).abs() < 2e-8);
        assert!(cfg.cache_bytes(0, 262144).is_err());
        assert!(cfg.cache_bytes(8, 1048577).is_err());
        assert!(cfg.cache_bytes(usize::MAX, 1048576).is_err());
    }

    #[test]
    fn admitted_cache_charges_private_blocks_and_virtual_page_slack_per_slot() {
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        for (context, tables) in [(262144, 139520), (1048576, 532736)] {
            let layout = cfg.cache_layout(8, context).unwrap();
            assert_eq!(layout.margin_bytes, 2_359_296);
            assert_eq!(layout.cache_bytes, cfg.cache_bytes(8, context).unwrap() + layout.margin_bytes);
            assert_eq!(layout.page_table_bytes, tables);
            let demands = layout.demands(1).unwrap();
            assert_eq!(demands.iter().map(|d| d.bytes).sum::<u64>(), layout.cache_bytes + tables);
            assert_eq!(demands[1].bytes, layout.margin_bytes);
            let movable = crate::placement::draft::DrafterDemand::itemized(demands, 1).movable(2).unwrap();
            assert_eq!(movable.allowed, [1]);
            assert_eq!(movable.conditional[0].demands.iter().map(|d| d.bytes).sum::<u64>(),
                layout.cache_bytes + tables);
            for layer in &layout.layers[..4] {
                assert_eq!((layer.rows_per_slot(), layer.table_width(), layer.slots()), (4160, 66, 8));
            }
            let full = &layout.layers[4];
            assert_eq!(full.rows_per_slot(), context);
            assert_eq!(full.table_width(), context / 64);
            assert!(full.block_view(7, context - cfg.block, cfg.block).is_ok());
            assert!(full.block_view(7, context - cfg.block + 1, cfg.block).is_err());
            assert!(full.row(8, 0).is_err());
            assert!(full.row(0, context).is_err());
            assert!(full.block_view(0, 0, 0).is_err());
            assert!(full.block_view(0, 0, cfg.block + 1).is_err());
            assert!(full.block_view(0, usize::MAX, cfg.block).is_err());
        }
        assert!(cfg.cache_layout(0, 262144).is_err());
        assert!(cfg.cache_layout(8, 1048577).is_err());
        assert!(cfg.cache_layout(usize::MAX, 1048576).is_err());
        let small = cfg.cache_layout(2, 7).unwrap();
        assert!(small.layers.iter().all(|l| l.rows_per_slot() == 64));
        assert!(small.layers[0].block_view(0, 0, 7).is_ok());
        assert!(small.layers[0].block_view(0, 0, 8).is_err());
    }

    #[test]
    fn full_block_at_window_edge_preserves_live_rows_across_wraps_and_slots() {
        let cfg = MlaDflashConfig::from_json(&config()).unwrap();
        let layout = cfg.cache_layout(3, 262144).unwrap();
        let layer = &layout.layers[0];
        let capacity = layer.rows_per_slot();
        let mut saw_repeated_page = false;
        let tag = |slot: usize, pos: usize| ((slot as u64 + 1) << 32) | pos as u64;
        for base in [0, 4096 - 64, capacity - 64, 2 * capacity - 64, 3 * capacity - 64] {
            for anchor in base..base + 128 {
                let begin = anchor.saturating_sub(4096);
                for slot in 0..3 {
                    let mut ring = vec![u64::MAX; 3 * capacity];
                    for neighbor in 0..3 {
                        for pos in begin..anchor {
                            ring[layer.row(neighbor, pos).unwrap()] = tag(neighbor, pos);
                        }
                    }
                    // Two rejected proposals at the same frontier must both
                    // leave every committed row and every neighboring slot exact.
                    for attempt in 0..2 {
                        for pos in anchor..anchor + cfg.block {
                            ring[layer.row(slot, pos).unwrap()] = (attempt as u64) << 48 | tag(slot, pos);
                        }
                        for neighbor in 0..3 {
                            for pos in begin..anchor {
                                assert_eq!(ring[layer.row(neighbor, pos).unwrap()], tag(neighbor, pos),
                                    "anchor {anchor}, slot {slot}, neighbor {neighbor}, live row {pos}");
                            }
                        }
                        let view = layer.block_view(slot, anchor, cfg.block).unwrap();
                        assert!(view.pages.len() <= layer.table_width());
                        assert_eq!(view.cache_seqlen, anchor + cfg.block - view.base_position);
                        saw_repeated_page |= view.pages.first() == view.pages.last() && view.pages.len() > 1;
                        for pos in begin..anchor + cfg.block {
                            let relative = pos - view.base_position;
                            let row = view.pages[relative / 64] as usize * 64 + relative % 64;
                            assert_eq!(row, layer.row(slot, pos).unwrap());
                        }
                    }
                }
            }
        }
        assert!(saw_repeated_page, "unaligned wrapped spans exercise virtual page duplication");
    }

    #[test]
    fn mla_config_refuses_unsupported_native_contracts() {
        for (field, value) in [
            ("mla_use_output_gate", json!(true)), ("is_causal", json!(true)),
            ("fc_norm", json!(true)), ("attention_bias", json!(true)),
            ("rms_norm_eps", json!(1e-6)), ("num_attention_heads", json!(96)),
            ("qk_rope_head_dim", json!(128)), ("sliding_window", json!(2048)),
            ("num_hidden_layers", json!(6)), ("vocab_size", json!(0)),
            ("intermediate_size", json!(usize::MAX)), ("max_position_embeddings", json!(0)),
        ] {
            let mut raw = config(); raw[field] = value;
            assert!(MlaDflashConfig::from_json(&raw).is_err(), "{field}");
        }
        for (field, value) in [("target_layer_ids", json!([19,19])),
            ("target_layer_ids", json!([37,19])), ("target_layer_ids", json!([93])),
            ("mask_token_id", json!(163840)), ("conv_group_size", json!(0)),
            ("block_size", json!(64)), ("attention_mode", json!("gqa"))] {
            let mut raw = config(); raw["dflash_config"][field] = value;
            assert!(MlaDflashConfig::from_json(&raw).is_err(), "{field}");
        }
    }
}
