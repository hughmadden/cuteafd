//! Header-native Muse Spark core trellis inventory: the dense matrices an
//! EXL3 Core-K6 publication may store as K6 MCG trellis. Everything outside
//! this keep-list (norms, residual gates, router, embedding, towers) keeps
//! its checkpoint precision. The attention output gate (`self_attn.gate_proj`)
//! is on the list but optional: a publication may keep it BF16.
use super::config::MuseSparkConfig;
use crate::{formats::trellis_core::{read_projections, CoreContract}, plan::Checkpoint};
pub use crate::formats::trellis_core::{CoreProjection, CoreSplit};
use anyhow::{ensure, Context, Result};
pub use crate::families::glm5_flash::core::{NativeLayout, NativeProjection};
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
            |stem| split(cfg, stem).map(|split| CoreContract { split, bits: 6, padded_outputs: stem == HEAD }))?;
        let mut inventory = Self { projections, native: BTreeMap::new() };
        // Inventory-only planning has selected names but no native headers yet.
        let missing = expected(cfg).keys().filter(|name| name.contains(".self_attn.")
            && !name.contains(".gate_proj.") && checkpoint.contains_tensor(name)
            && !checkpoint.tensors.iter().any(|tensor| tensor.meta.name == **name))
            .cloned().collect::<std::collections::BTreeSet<_>>();
        let native_checkpoint;
        let native_source = if missing.is_empty() { checkpoint } else {
            native_checkpoint = Checkpoint::open_view_for_role(checkpoint.view.as_ref()
                .context("Muse native attention requires a selected checkpoint view")?,
                "Muse native attention", |name| missing.contains(name)
                    || name.strip_suffix("_scale_inv").is_some_and(|weight| missing.contains(weight)))?;
            &native_checkpoint
        };
        for (name, shape) in expected(cfg) {
            let stem = name.strip_suffix(".weight").unwrap();
            if !stem.contains(".self_attn.") || stem.ends_with(".gate_proj")
                || !checkpoint.contains_tensor(&name) { continue; }
            let selected = native_source.projection(stem)?;
            let crate::plan::checkpoint::ProjectionStorage::Native { dtype, shape: actual } = selected.storage else { continue; };
            ensure!(actual == shape, "{stem}: native Muse attention shape {actual:?}, expected {shape:?}");
            ensure!(matches!(dtype, cuteafd_core::DType::Bf16 | cuteafd_core::DType::F8E4M3),
                "{stem}: native Muse attention requires BF16 or block E4M3");
            if dtype == cuteafd_core::DType::F8E4M3 {
                let scale = selected.tensors.iter().find(|t| t.meta.name == format!("{name}_scale_inv"))
                    .context("native Muse FP8 attention lacks block scales")?;
                ensure!(scale.meta.dtype == cuteafd_core::DType::F32
                    && scale.meta.shape == [shape[0].div_ceil(128), shape[1].div_ceil(128)],
                    "{stem}: native Muse FP8 requires F32 128x128 block scales");
            }
            inventory.native.insert(stem.into(), NativeProjection {
                shape: [shape[0], shape[1]], split: split(cfg, stem).context("Muse attention split")?,
                fp8: dtype == cuteafd_core::DType::F8E4M3
                    || checkpoint.attention_mode() == crate::plan::checkpoint::AttentionMode::Fp8Compact,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{checkpoint::{AttentionMode, CheckpointView}, testing};

    fn mixed(dir: &std::path::Path) {
        let (mut tensors, _) = testing::muse_spark_tensors(&[]);
        tensors.retain(|(name, _, _)| !name.contains(".self_attn.o_proj."));
        for layer in 0..62 {
            tensors.push((format!("{PREFIX}{layer}.self_attn.o_proj.weight"), "BF16", vec![8192, 8192]));
        }
        let mut config = testing::muse_spark_config();
        config["quantization_config"] = serde_json::json!({"quant_method":"exl3"});
        testing::write_snapshot(dir, &config, &tensors, None);
        for (source, destination) in [("config.json", crate::plan::checkpoint::view::COMPACT_CONFIG),
            ("model.safetensors.index.json", crate::plan::checkpoint::view::COMPACT_INDEX)] {
            std::fs::copy(dir.join(source), dir.join(destination)).unwrap();
        }
        std::fs::write(dir.join(crate::plan::checkpoint::view::COMPACT_QUANT), "{}").unwrap();
    }

    #[test]
    fn mixed_attention_is_per_projection_and_has_exact_tp_residency() -> Result<()> {
        let dir = tempfile::tempdir()?;
        mixed(dir.path());
        for mode in [AttentionMode::Hqa, AttentionMode::Fp8Compact] {
            let view = CheckpointView::select(dir.path(), mode)?;
            // Plan's inventory path has no native headers; reads remain selected-index scoped.
            let checkpoint = Checkpoint::inventory_view(&view)?;
            let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
            let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
            assert_eq!(core.native.len(), 62);
            assert!(!core.projections.contains_key(&format!("{PREFIX}0.self_attn.o_proj")));
            assert!(core.projections.contains_key(&format!("{PREFIX}0.self_attn.q_proj")));
            for ranks in [1, 2] {
                for (rank, layout) in super::super::weight_layout::LayerLayout::ranks(&cfg, 0, ranks, &core)?.iter().enumerate() {
                    assert!(!layout.core.contains_key("attn.o"));
                    assert!(!layout.operands.contains_key("attn.o"));
                    let native = &layout.native["attn.o"];
                    assert_eq!(native.shape, [8192, 8192 / ranks]);
                    assert_eq!((native.rank, native.ranks), (rank, ranks));
                    let expected = if mode == AttentionMode::Fp8Compact {
                        8192 * (8192 / ranks) + 64 * (64 / ranks) * 4
                    } else { 8192 * (8192 / ranks) * 2 };
                    assert_eq!(native.bytes()?, expected as u64);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn incomplete_attention_or_moe_groups_still_refuse() -> Result<()> {
        let dir = tempfile::tempdir()?;
        mixed(dir.path());
        let mut checkpoint = Checkpoint::open(dir.path())?;
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
        let name = format!("{PREFIX}0.self_attn.o_proj.weight");
        checkpoint.weight_map.remove(&name);
        checkpoint.tensors.retain(|tensor| tensor.meta.name != name);
        assert!(CoreInventory::from_checkpoint(&cfg, &checkpoint).is_err());
        let mut checkpoint = Checkpoint::open(dir.path())?;
        let prefix = format!("{PREFIX}0.mlp.post_expert_proj.");
        checkpoint.weight_map.retain(|name, _| !name.starts_with(&prefix));
        checkpoint.tensors.retain(|tensor| !tensor.meta.name.starts_with(&prefix));
        assert!(CoreInventory::from_checkpoint(&cfg, &checkpoint).is_err());
        Ok(())
    }
}
