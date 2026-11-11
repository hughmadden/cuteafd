//! Header-native Muse Spark core trellis inventory: the dense matrices an
//! EXL3 Core-K6 publication may store as K6 MCG trellis. Everything outside
//! this keep-list (norms, residual gates, router, embedding, towers) keeps
//! its checkpoint precision. The attention output gate (`self_attn.gate_proj`)
//! is on the list but optional: a publication may keep it BF16.
use super::config::MuseSparkConfig;
use crate::{formats::trellis_core::{read_projections, CoreContract}, plan::Checkpoint};
pub use crate::formats::trellis_core::{CoreProjection, CoreSplit};
use anyhow::{ensure, Context, Result};
pub use crate::formats::native_attention::{NativeLayout, NativeProjection};
use std::collections::BTreeMap;

pub const PREFIX: &str = "model.language_model.layers.";
pub const HEAD: &str = "lm_head";
/// Required attention operands, resolved independently in the selected index.
const ATTENTION: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];

/// TP2 ownership of each eligible projection: attention heads split; the MoE
/// frontend splits its outputs over E and the backend its inputs (single
/// residency, a rank-ordered K-split reduction). The LM head stays whole.
pub fn split(cfg: &MuseSparkConfig, stem: &str) -> Option<CoreSplit> {
    if stem == HEAD { return Some(CoreSplit::Whole); }
    let (layer, leaf) = stem.strip_prefix(PREFIX)?.split_once('.')?;
    if layer.parse::<usize>().ok()? >= cfg.layers { return None; }
    match leaf {
        "self_attn.q_proj" | "self_attn.k_proj" | "self_attn.v_proj" | "self_attn.gate_proj" => Some(CoreSplit::Output),
        "self_attn.o_proj" => Some(CoreSplit::Input),
        "mlp.pre_expert_proj" => Some(CoreSplit::Output),
        "mlp.post_expert_proj" => Some(CoreSplit::Input),
        _ => None,
    }
}

/// Logical [N, K] of every keep-list matrix.
pub fn expected(cfg: &MuseSparkConfig) -> BTreeMap<String, Vec<usize>> {
    let (h, q, kv, e) = (cfg.hidden, cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim, cfg.expert_hidden);
    let mut shapes = BTreeMap::from([(format!("{HEAD}.weight"), vec![cfg.vocab_size, h])]);
    for layer in 0..cfg.layers {
        let root = format!("{PREFIX}{layer}");
        for (leaf, n, k) in [("self_attn.q_proj", q, h), ("self_attn.k_proj", kv, h), ("self_attn.v_proj", kv, h),
            ("self_attn.o_proj", h, q), ("self_attn.gate_proj", q, h),
            ("mlp.pre_expert_proj", e, h), ("mlp.post_expert_proj", h, e)] {
            shapes.insert(format!("{root}.{leaf}.weight"), vec![n, k]);
        }
    }
    shapes
}

#[derive(Debug, Default, Clone)]
pub struct CoreInventory {
    pub projections: BTreeMap<String, CoreProjection>,
    pub native: BTreeMap<String, NativeProjection>,
}

impl CoreInventory {
    pub fn from_checkpoint(cfg: &MuseSparkConfig, checkpoint: &Checkpoint) -> Result<Self> {
        let selected = |name: &str| name.starts_with("lm_head.") || name.strip_prefix(PREFIX)
            .and_then(|rest| rest.split_once('.'))
            .is_some_and(|(layer, leaf)| layer.parse::<usize>().is_ok_and(|layer| layer < cfg.layers)
                && !leaf.starts_with("mlp.experts."));
        // The K6 head is stored at whole 128-row output blocks (202,112 for the
        // logical 202,048 vocabulary); the engine trims logits before the softcap.
        let projections = read_projections(checkpoint, &expected(cfg), selected,
            |stem| split(cfg, stem).map(|split| CoreContract { family: "Muse", widths: &[4, 6, 8], split, padded_outputs: stem == HEAD }))?;
        let mut inventory = Self { projections, native: BTreeMap::new() };
        for (name, shape) in expected(cfg) {
            let stem = name.strip_suffix(".weight").unwrap();
            if !stem.contains(".self_attn.") || stem.ends_with(".gate_proj")
                || !checkpoint.contains_tensor(&name) { continue; }
            let selected = checkpoint.projection(stem)?;
            let crate::plan::checkpoint::ProjectionStorage::Native { ref dtype, shape: ref actual } = selected.storage else { continue; };
            ensure!(*actual == shape, "{stem}: native Muse attention shape {actual:?}, expected {shape:?}");
            ensure!(matches!(dtype, cuteafd_core::DType::Bf16 | cuteafd_core::DType::F8E4M3),
                "{stem}: native Muse attention requires BF16 or block E4M3");
            if *dtype == cuteafd_core::DType::F8E4M3 {
                if checkpoint.core_view() == crate::plan::checkpoint::CoreView::Fp8Compact {
                    return Err(crate::formats::native_attention::UnsupportedFp8Conversion {
                        tensor: name.clone(), found: "E4M3".into(),
                    }.into());
                }
                crate::formats::native_attention::fp8_scale(&selected, stem)?;
            }
            inventory.native.insert(stem.into(), NativeProjection {
                shape: [shape[0], shape[1]], split: split(cfg, stem).context("Muse attention split")?,
                fp8: *dtype == cuteafd_core::DType::F8E4M3
                    || checkpoint.core_view() == crate::plan::checkpoint::CoreView::Fp8Compact,
            });
        }
        for layer in 0..cfg.layers {
            let root = format!("{PREFIX}{layer}");
            for (module, leaves) in [("self_attn", &ATTENTION[..]), ("mlp", &["pre_expert_proj", "post_expert_proj"][..])] {
                let present = leaves.iter().filter(|leaf|
                    inventory.projections.contains_key(&format!("{root}.{module}.{leaf}"))).count();
                let mixed_complete = module == "self_attn" && leaves.iter().all(|leaf| {
                    let stem = format!("{root}.{module}.{leaf}");
                    inventory.projections.contains_key(&stem) || inventory.native.contains_key(&stem)
                });
                ensure!(present == 0 || present == leaves.len() || mixed_complete,
                    "{root}.{module}: incomplete native Muse Spark operator group");
            }
        }
        Ok(inventory)
    }

    pub fn get_weight(&self, name: &str) -> Option<&CoreProjection> {
        self.projections.get(name.strip_suffix(".weight").unwrap_or(name))
    }

    /// Whether every layer's attention and MoE projections are native K6.
    pub fn complete(&self, cfg: &MuseSparkConfig) -> bool {
        (0..cfg.layers).all(|layer| ["self_attn.q_proj", "mlp.pre_expert_proj"].iter()
            .all(|leaf| self.projections.contains_key(&format!("{PREFIX}{layer}.{leaf}"))))
    }
}
