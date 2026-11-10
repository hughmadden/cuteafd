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
            |stem| split(cfg, stem).map(|split| CoreContract { family: "Muse", widths: &[6], split, padded_outputs: stem == HEAD }))?;
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
                if checkpoint.attention_mode() == crate::plan::checkpoint::AttentionMode::Fp8Compact {
                    return Err(crate::formats::native_attention::UnsupportedFp8Conversion {
                        tensor: name.clone(), found: "E4M3".into(),
                    }.into());
                }
                crate::formats::native_attention::fp8_scale(&selected, stem)?;
            }
            inventory.native.insert(stem.into(), NativeProjection {
                shape: [shape[0], shape[1]], split: split(cfg, stem).context("Muse attention split")?,
                fp8: *dtype == cuteafd_core::DType::F8E4M3
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

    fn mixed(dir: &std::path::Path, expert_layers: &[usize]) {
        let (mut tensors, projections) = testing::muse_spark_tensors(expert_layers);
        tensors.retain(|tensor| !crate::media::muse_vision::is_tensor(&tensor.0)
            && !tensor.0.starts_with("model.audio_tower."));
        let compact = tensors.clone();
        tensors.retain(|(name, _, _)| !name.contains(".self_attn.o_proj."));
        for layer in 0..62 {
            tensors.push((format!("{PREFIX}{layer}.self_attn.o_proj.weight"), "BF16", vec![8192, 8192]));
        }
        let mut config = testing::muse_spark_config();
        config.as_object_mut().unwrap().remove("audio_config");
        config.as_object_mut().unwrap().remove("vision_config");
        config["quantization_config"].as_object_mut().unwrap().remove("per_module_metadata");
        testing::write_snapshot(dir, &config, &tensors, None);
        std::fs::rename(dir.join("model-00001-of-00001.safetensors"), dir.join("hqa.safetensors")).unwrap();
        let index = |tensors: &[testing::Tensor], shard: &str| serde_json::json!({"weight_map":
            tensors.iter().map(|(name, ..)| (name.clone(), serde_json::json!(shard)))
                .collect::<serde_json::Map<_, _>>() });
        std::fs::write(dir.join("model.safetensors.index.json"), index(&tensors, "hqa.safetensors").to_string()).unwrap();
        testing::write_safetensors(&dir.join("compact.safetensors"), &compact);
        std::fs::write(dir.join(crate::plan::checkpoint::view::COMPACT_CONFIG), config.to_string()).unwrap();
        std::fs::write(dir.join(crate::plan::checkpoint::view::COMPACT_INDEX),
            index(&compact, "compact.safetensors").to_string()).unwrap();
        let mut quant = testing::exl3_compact(4);
        quant.as_object_mut().unwrap().extend(config["quantization_config"].as_object().unwrap().clone());
        let quant = testing::exl3_manifest(&quant, &projections);
        std::fs::write(dir.join(crate::plan::checkpoint::view::COMPACT_QUANT), quant.to_string()).unwrap();
        let attention_view = serde_json::json!({"schema":"cuteafd-attention-views-v1", "mode":"hqa",
            "views":{"hqa":{"config":"config.json", "index":"model.safetensors.index.json",
                "quantize":"quantize_config.json"}, "compact-attention":{
                "config":crate::plan::checkpoint::view::COMPACT_CONFIG,
                "index":crate::plan::checkpoint::view::COMPACT_INDEX,
                "quantize":crate::plan::checkpoint::view::COMPACT_QUANT}},
            "native_modules":"quantize_config.json:native_attention"});
        config["quantization_config"]["attention_view"] = attention_view.clone();
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let native_projections = projections.into_iter().filter(|(name, ..)|
            !name.ends_with(".self_attn.o_proj")).collect::<Vec<_>>();
        let mut native_quant = testing::exl3_compact(4);
        native_quant.as_object_mut().unwrap().extend(config["quantization_config"].as_object().unwrap().clone());
        let mut manifest = testing::exl3_manifest(&native_quant, &native_projections);
        manifest["attention_view"] = attention_view;
        manifest["native_attention"] = serde_json::Value::Object((0..62).map(|layer| {
            let module = format!("{PREFIX}{layer}.self_attn.o_proj");
            let weight = format!("{module}.weight");
            (module, serde_json::json!({"format":"bf16", "shard":"hqa.safetensors",
                "compact_attention_bits":6, "tensors":{weight:{"dtype":"BF16", "shape":[8192,8192]}}}))
        }).collect());
        testing::write_quantize_config(dir, &manifest);
    }

    #[test]
    fn mixed_attention_is_per_projection_and_has_exact_tp_residency() -> Result<()> {
        let dir = tempfile::tempdir()?;
        mixed(dir.path(), &[]);
        for mode in [AttentionMode::Hqa, AttentionMode::Fp8Compact] {
            let view = CheckpointView::select(dir.path(), mode)?;
            // Shared inventory hydration supplies native headers from the selected index.
            let checkpoint = Checkpoint::inventory_view(&view)?;
            assert!((0..62).all(|layer|
                checkpoint.has_header(&format!("{PREFIX}{layer}.self_attn.o_proj.weight"))));
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
    fn compact_is_parent_k6_and_does_not_open_native_shards() -> Result<()> {
        let dir = tempfile::tempdir()?;
        mixed(dir.path(), &[]);
        let hqa = CheckpointView::select(dir.path(), AttentionMode::Hqa)?;
        let compact = CheckpointView::select(dir.path(), AttentionMode::Compact)?;
        let fp8 = CheckpointView::select(dir.path(), AttentionMode::Fp8Compact)?;
        assert_ne!(hqa.fingerprint(), compact.fingerprint());
        assert_ne!(hqa.fingerprint(), fp8.fingerprint());
        assert_ne!(compact.fingerprint(), fp8.fingerprint());
        let identities = [&hqa, &compact, &fp8].into_iter().map(|view|
            crate::plan::checkpoint::identity::CheckpointIdentity::from_view(view, None)
                .map(|identity| identity.fingerprint)).collect::<Result<std::collections::BTreeSet<_>>>()?;
        assert_eq!(identities.len(), 3, "bootstrap identities must separate every attention mode");
        std::fs::remove_file(dir.path().join("hqa.safetensors"))?;
        let checkpoint = Checkpoint::open_view(&compact)?;
        assert!(checkpoint.missing_shards.is_empty());
        assert!(checkpoint.tensors.iter().all(|tensor| tensor.shard == "compact.safetensors"));
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        assert!(core.native.is_empty());
        assert!(core.projections.contains_key(&format!("{PREFIX}0.self_attn.o_proj")));
        let family = crate::plan::family::detect(&checkpoint).context("Muse fixture family")?;
        family.open(&checkpoint)?;
        Ok(())
    }

    #[test]
    fn selected_modes_name_native_residency_and_load_only_owners() -> Result<()> {
        use crate::plan::{layout::LayoutOptions, ExpertPlacement, PlanInput, PlanOptions};
        let dir = tempfile::tempdir()?;
        mixed(dir.path(), &(0..62).collect::<Vec<_>>());
        for mode in [AttentionMode::Hqa, AttentionMode::Fp8Compact, AttentionMode::Compact] {
            let view = CheckpointView::select(dir.path(), mode)?;
            let input = PlanInput::open_view(&view)?;
            let report = input.plan(&PlanOptions {
                placement: ExpertPlacement::from_spark_ranks(6),
                layout: Some(LayoutOptions { host_memory: Some((256 << 30, 256 << 30)),
                    rtx_bytes: vec![crate::placement::inventory::PRO_TOTAL_BYTES; 2],
                    head_split: true, host_cache_bytes: Some(0), ..Default::default() }),
                ..Default::default()
            })?;
            assert!(report.config_error.is_none(), "{:?}", report.config_error);
            assert!(report.components.iter().all(|component| component.rejected == 0), "{:?}", report.components);
            let ledger = report.phase_ledger.as_ref().context("Muse selected mode phase ledger")?;
            let host = ledger.budgets.iter().find(|budget| budget.id == "host").context("Muse host budget")?;
            use crate::placement::planner::{Location, Phase};
            assert!(host.locations.contains(&Location::PINNED));
            assert!(ledger.budgets.iter().filter(|budget| budget.id.starts_with("rtx"))
                .all(|budget| !budget.locations.contains(&Location::PINNED)));
            let dense_bound = if mode == AttentionMode::Compact { 2 << 30 } else { 17_716_740_096 };
            let host_load = ledger.phases.iter().find(|phase| phase.budget == "host" && phase.phase == Phase::Load)
                .context("Muse host load demand")?;
            assert_eq!(host_load.opaque,
                crate::placement::families::muse_spark::HOST_RUNTIME.bytes + 2 * dense_bound);
            let placement = report.solver_placement.context("Muse selected mode placement")?;
            let checkpoint = input.checkpoint();
            let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
            let core = CoreInventory::from_checkpoint(&cfg, checkpoint)?;
            let global = super::super::weight_layout::GlobalLayout::new(&cfg, &core)?;
            let layers = (0..cfg.layers).map(|layer|
                super::super::weight_layout::LayerLayout::ranks(&cfg, layer, 2, &core))
                .collect::<Result<Vec<_>>>()?;
            for (gpu, rank) in placement.items.iter().enumerate() {
                assert!(!rank.iter().any(|item| item.group == "Muse native attention values and block scales"));
                let local = layers.iter().map(|ranks| &ranks[gpu]).collect::<Vec<_>>();
                let arena = super::super::weight_layout::coordinator_arena(gpu, 2, &global, &local)?;
                let root = rank.iter().find(|item| item.group == "weight arena").context("target arena root")?;
                assert_eq!(root.bytes, arena.bytes() as u64);
                let native = local.iter().flat_map(|layout| layout.native.values())
                    .map(|layout| layout.bytes()).sum::<Result<u64>>()?;
                let expected = match mode {
                    AttentionMode::Compact => 0,
                    AttentionMode::Hqa => 62 * 8192 * 4096 * 2,
                    AttentionMode::Fp8Compact => 62 * (8192 * 4096 + 64 * 32 * 4),
                };
                assert_eq!(native, expected);
            }
            let layouts = super::super::weight_layout::LayerLayout::ranks(&cfg, 0, 2, &core)?;
            let load = super::super::weight_layout::native_mount_load(layouts[0].native.values())?;
            let owners = load.allocations("muse.native.load.0");
            assert_eq!(owners.is_empty(), mode == AttentionMode::Compact);
            assert!(owners.iter().all(|owner| owner.lifetime == crate::placement::planner::Lifetime {
                first: crate::placement::planner::Phase::Load, last: crate::placement::planner::Phase::Load }));
        }
        Ok(())
    }

    #[test]
    fn incomplete_attention_or_moe_groups_still_refuse() -> Result<()> {
        let dir = tempfile::tempdir()?;
        mixed(dir.path(), &[]);
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
