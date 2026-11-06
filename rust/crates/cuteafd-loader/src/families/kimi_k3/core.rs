//! Header-native Kimi core trellis inventory. Source bytes and resident bytes
//! differ: a TP shard replicates the opposite-side H128 sign/scale vector.
use super::config::KimiK3Config;
use crate::plan::checkpoint::Checkpoint;
pub use crate::formats::trellis_core::{CoreProjection, CoreSplit, ProjectionScratchLayout, ProjectionStorageLayout};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// Only this explicit list can replace a BF16 core matrix. Everything else
/// (including layer-zero dense FFN, kv_b, KDA recurrence, router and embedding)
/// retains its checkpoint-native precision.
pub fn split(stem: &str) -> Option<CoreSplit> {
    if stem == "language_model.lm_head" { return Some(CoreSplit::Whole); }
    let rest = stem.strip_prefix("language_model.model.layers.")?;
    let (layer, leaf) = rest.split_once('.')?;
    let layer = layer.parse::<usize>().ok()?;
    match leaf {
        "self_attn.q_proj" | "self_attn.k_proj" | "self_attn.v_proj" |
        "self_attn.g_proj" | "self_attn.q_b_proj" => Some(CoreSplit::Output),
        "self_attn.o_proj" => Some(CoreSplit::Input),
        "self_attn.q_a_proj" | "self_attn.kv_a_proj_with_mqa" => Some(CoreSplit::Whole),
        "block_sparse_moe.shared_experts.gate_proj" | "block_sparse_moe.shared_experts.up_proj" if layer > 0 => Some(CoreSplit::Output),
        "block_sparse_moe.shared_experts.down_proj" if layer > 0 => Some(CoreSplit::Input),
        "block_sparse_moe.routed_expert_down_proj" | "block_sparse_moe.routed_expert_up_proj" if layer > 0 => Some(CoreSplit::Whole),
        _ => None,
    }
}

/// Named split-stage outputs retained beside the fused lane's existing arena.
/// Arena stages live within one unit's hook (`unit` rows); the peer meet
/// planes are row planes of the whole call (`rows`, windowed per unit).
pub fn intermediate_extents(rows: usize, unit: usize) -> Result<BTreeMap<&'static str, usize>> {
    intermediate_extents_for_ranks(rows, unit, 2)
}

pub fn intermediate_extents_for_ranks(rows: usize, unit: usize, ranks: usize) -> Result<BTreeMap<&'static str, usize>> {
    ensure!(matches!(ranks, 1 | 2), "Kimi K6 physical rank count");
    [("K6 q",12288/ranks,2),("K6 k",12288/ranks,2),("K6 v",12288/ranks,2),("K6 gate",12288,2),
        ("K6 q_a",1536,2),("K6 kv_a",640,2),("K6 q_b",18432,2),("K6 normalized",12288,2),
        ("K6 FFN gate",6144/ranks,2),("K6 FFN up",6144/ranks,2),("K6 attention partial",7168,2),
        ("K6 FFN partial",7168,2),("K6 reduced",7168,4)].into_iter()
        .map(|(name,width,element)| {
            let rows = if super::allocation::CORE_MEET_PLANES.contains(&name) { rows } else { unit };
            Ok((name,rows.checked_mul(width*element).context("K6 lane intermediate overflow")?))
        })
        .collect()
}

/// Split entrypoints can need separate scratch larger than the fused plane.
pub fn lane_extents(rows: usize, unit: usize, base_kda: usize, base_mla: usize,
    split_kda: usize, split_mla: usize) -> Result<BTreeMap<&'static str, usize>> {
    lane_extents_for_ranks(rows, unit, base_kda, base_mla, split_kda, split_mla, 2)
}

pub fn lane_extents_for_ranks(rows: usize, unit: usize, base_kda: usize, base_mla: usize,
    split_kda: usize, split_mla: usize, ranks: usize) -> Result<BTreeMap<&'static str, usize>> {
    let mut extents=intermediate_extents_for_ranks(rows, unit, ranks)?;
    if split_kda>base_kda {extents.insert("K6 KDA scratch",split_kda);}
    if split_mla>base_mla {extents.insert("K6 MLA scratch",split_mla);}
    Ok(extents)
}

/// Inert installed-package inspection, with no CUDA load or payload read.
pub fn package_lane_extents(manifest: &serde_json::Value, rows: usize, unit: usize) -> Result<BTreeMap<&'static str,usize>> {
    package_lane_extents_for_ranks(manifest, rows, unit, 2)
}

pub fn package_lane_extents_for_ranks(manifest: &serde_json::Value, rows: usize, unit: usize,
    ranks: usize) -> Result<BTreeMap<&'static str,usize>> {
    ensure!(matches!(ranks, 1 | 2), "Kimi K6 package physical rank count");
    let programs=manifest["programs"].as_array().context("Kimi K6 package program list")?;
    let scratch=|name: &str| -> Result<usize> {
        let entries=programs.iter().filter(|p|p["name"]==name).collect::<Vec<_>>();
        ensure!(entries.len()==1,"Kimi K6 package requires exactly one {name}");
        let scratch=entries[0]["scratch_bytes_at_capacity"].as_object().context("Kimi program scratch map")?;
        scratch.values().try_fold(0usize,|sum,value|sum.checked_add(usize::try_from(value.as_u64()?).ok()?))
            .context("Kimi package scratch extent")
    };
    let mut base_kda=0;let mut split_kda=0;let mut base_mla=0;let mut split_mla=0;
    for capacity in [64,4096] {
        let prefix = if ranks == 1 { "kda_fused_h96" } else { "kimik3_kda" };
        base_kda=base_kda.max(scratch(&format!("{prefix}_m{capacity}"))?);
        split_kda=split_kda.max(scratch(&format!("kda_core_h{}_m{capacity}", 96 / ranks))?);
        for heads in [48,96] {
            let suffix=if heads==48 {""}else{"_h96"};
            base_mla=base_mla.max(scratch(&format!("kimik3_mla_producer{suffix}_m{capacity}"))?);
            for stride in [576,640] {for cache in ["","_fp8"] {
                split_mla=split_mla.max(scratch(&format!("mla_rank_norm_h{heads}_kv{stride}{cache}_m{capacity}"))?);
            }}
        }
    }
    lane_extents_for_ranks(rows,unit,base_kda,base_mla,split_kda,split_mla,ranks)
}

pub fn exchange_recv_bytes(rows: usize) -> Result<usize> {
    ensure!((1..=4096).contains(&rows), "Kimi exchange row capacity");
    Ok((rows * 7168 * 2).max(256))
}

#[derive(Debug,Clone,PartialEq,Eq)]
pub struct CoreCharges {
    pub projection: Vec<Option<ProjectionScratchLayout>>,
    pub stages: BTreeMap<&'static str,usize>,
    pub stage_bytes: usize,
    pub receive: usize,
}
impl CoreCharges {
    /// `rows` lane rows of `unit`-row units: projections and the peer
    /// receive serve one unit's launch; meet planes span the lane.
    pub fn from_package(cfg: &KimiK3Config, core: &CoreInventory, rows: usize, unit: usize,
        manifest: &serde_json::Value) -> Result<Self> {
        Self::from_package_for_ranks(cfg, core, rows, unit, manifest, 2)
    }

    pub fn from_package_for_ranks(cfg: &KimiK3Config, core: &CoreInventory, rows: usize, unit: usize,
        manifest: &serde_json::Value, ranks: usize) -> Result<Self> {
        ensure!(matches!(ranks, 1 | 2), "Kimi K6 charge physical rank count");
        let stages = if core.projections.is_empty() { BTreeMap::new() }
            else { package_lane_extents_for_ranks(manifest,rows,unit,ranks)? };
        let stage_bytes = if stages.is_empty() { 0 } else {
            let base = super::allocation::NativeLaneLayout::for_ranks(cfg, rows, unit, super::COMPILED_CONTEXT,
                super::allocation::NativeScratch::from_manifest_for_ranks(manifest, ranks)?, ranks)?;
            base.clone().with_core_stages(stages.clone())?.bytes()?.checked_sub(base.bytes()?)
                .context("Kimi core stage charge underflow")?
        };
        Ok(Self {
            projection: projection_scratch_for_ranks(cfg,core,unit,&manifest["families"]["dense_trellis_k6"],ranks)?,
            stages, stage_bytes,
            receive: if ranks == 2 { exchange_recv_bytes(unit)? } else { 0 },
        })
    }
    pub fn demands(&self) -> Result<Vec<crate::placement::Demand>> {
        use cuteafd_core::memory_layout::{Basis,Category};
        let mut demands = Vec::new();
        for (gpu, projection) in self.projection.iter().enumerate() {
            demands.push(crate::placement::Demand::new(gpu as u8,Category::Workspace,
                "Kimi K6 projection scratch",projection.as_ref().map_or(Ok(0),|layout|layout.bytes())? as u64,Basis::Exact));
            demands.push(crate::placement::Demand::new(gpu as u8,Category::Workspace,
                "Kimi K6 split-stage operands",self.stage_bytes as u64,Basis::Exact));
            if self.receive != 0 {
                demands.push(crate::placement::Demand::new(gpu as u8,Category::Workspace,
                    "Kimi K6 peer receive",self.receive as u64,Basis::Exact));
            }
        }
        Ok(demands)
    }
}

/// The package selects native projection geometry. Headers select projections; no second export table.
pub fn projection_scratch(cfg: &KimiK3Config, core: &CoreInventory, rows: usize,
    metadata: &serde_json::Value) -> Result<[Option<ProjectionScratchLayout>; 2]> {
    let layouts = projection_scratch_for_ranks(cfg, core, rows, metadata, 2)?;
    Ok(layouts.try_into().expect("validated pair scratch inventory"))
}

pub fn projection_scratch_for_ranks(cfg: &KimiK3Config, core: &CoreInventory, rows: usize,
    metadata: &serde_json::Value, ranks: usize) -> Result<Vec<Option<ProjectionScratchLayout>>> {
    ensure!(matches!(ranks, 1 | 2), "Kimi K6 physical rank count");
    if core.projections.is_empty() { return Ok(vec![None; ranks]); }
    core.validate_runtime_groups(cfg)?;
    ensure!(metadata["abi_version"] == 1, "Kimi K6 row-policy ABI");
    let policies = metadata["row_policies"].as_array().context("Kimi K6 row-policy list")?;
    let mut combined: Vec<Option<ProjectionScratchLayout>> = vec![None; ranks];
    for projection in core.projections.values() {
        let head = projection.stem == "language_model.lm_head";
        let mla = projection.stem.contains(".self_attn.") && projection.stem
            .strip_prefix("language_model.model.layers.").and_then(|s| s.split_once('.'))
            .and_then(|(layer,_)| layer.parse::<usize>().ok()).is_some_and(|layer|
                cfg.attention[layer] == super::config::KimiAttention::Mla);
        let worlds: &[usize] = if ranks == 1 || projection.split == CoreSplit::Whole { &[1] }
            else if mla { &[1,2] } else { &[2] };
        for &world in worlds {
            let (mut n, mut k) = (projection.stored_outputs, projection.shape[1]);
            let side = if world == 1 || projection.split != CoreSplit::Input {1} else {2};
            if world == 2 { if side == 1 { n /= 2; } else { k /= 2; } }
            let book = match projection.codebook {
                crate::formats::exl3_storage::Exl3Codebook::Mcg => "mcg",
                crate::formats::exl3_storage::Exl3Codebook::Mul1 => "mul1",
            };
            let matching = policies.iter().filter(|p| p["input_features"] == k && p["output_features"] == n
                && p["sharded_side"] == side && p["codebook"] == book).collect::<Vec<_>>();
            ensure!(matching.len() == 1, "{}: expected one K6 row policy for K{k}/N{n}/side{side}/{book}, found {}",
                projection.stem, matching.len());
            let policy = matching[0];
            let mut layout = ProjectionScratchLayout::new(k,n,if head {rows.min(64)} else {rows})?;
            if let Some(grouped) = policy.get("grouped") {
                let groups = grouped["groups"].as_u64().and_then(|v| usize::try_from(v).ok())
                    .context("Kimi K6 grouped row-policy groups")?;
                layout.merge(&ProjectionScratchLayout::grouped(k, n, rows, groups)?)?;
            }
            // The last physical owner projects vocabulary; body projections run on every owner.
            for target in combined.iter_mut().skip(if head { ranks - 1 } else { 0 }) {
                if let Some(combined) = target {
                    if !head { combined.rows = rows; }
                    let mut required = layout.clone();
                    // A body owner has full-row storage; a head-only owner stays bounded.
                    required.rows = combined.rows;
                    combined.merge(&required)?;
                } else { *target = Some(layout.clone()); }
            }
        }
    }
    Ok(combined)
}

/// The drafter's own K6 vocabulary-head scratch at its `rows` (<= 64) head
/// rows, or `None` when the head is BF16. The drafter projects through the
/// target's K6 head on its bridge stream; the target lane scratch is never shared.
pub fn draft_head_scratch(core: &CoreInventory, rows: usize, metadata: &serde_json::Value)
    -> Result<Option<ProjectionScratchLayout>> {
    let Some(head) = core.projections.get("language_model.lm_head") else { return Ok(None) };
    ensure!((1..=64).contains(&rows), "Kimi K6 draft head rows outside 1..=64");
    ensure!(metadata["abi_version"] == 1, "Kimi K6 row-policy ABI");
    let (n, k) = (head.stored_outputs, head.shape[1]);
    let book = match head.codebook {
        crate::formats::exl3_storage::Exl3Codebook::Mcg => "mcg",
        crate::formats::exl3_storage::Exl3Codebook::Mul1 => "mul1",
    };
    let policies = metadata["row_policies"].as_array().context("Kimi K6 row-policy list")?;
    let matching = policies.iter().filter(|p| p["input_features"] == k && p["output_features"] == n
        && p["sharded_side"] == 1 && p["codebook"] == book).collect::<Vec<_>>();
    ensure!(matching.len() == 1, "lm_head: expected one K6 row policy for K{k}/N{n}/{book}, found {}", matching.len());
    Ok(Some(ProjectionScratchLayout::new(k, n, rows)?))
}

/// Admission charge of [`draft_head_scratch`] on the head rank (GPU1).
pub fn draft_head_demand(core: &CoreInventory, rows: usize, metadata: &serde_json::Value)
    -> Result<Option<crate::placement::Demand>> {
    use cuteafd_core::memory_layout::{Basis, Category};
    Ok(draft_head_scratch(core, rows, metadata)?.map(|layout| layout.bytes().map(|bytes|
        crate::placement::Demand::new(1, Category::Drafter, "Kimi K6 draft head scratch", bytes as u64, Basis::Exact)))
        .transpose()?)
}

#[derive(Debug, Default, Clone)]
pub struct CoreInventory {
    pub projections: BTreeMap<String, CoreProjection>,
    pub native: BTreeMap<String, crate::families::glm5_flash::core::NativeProjection>,
}

impl CoreInventory {
    /// Expected logical shapes come from the family geometry, not a side file.
    /// Both native MCG and MUL1 keep their own unpacked H128 vectors.
    pub fn from_checkpoint(cfg: &KimiK3Config, checkpoint: &Checkpoint,
        expected: &BTreeMap<String, Vec<usize>>) -> Result<Self> {
        let projections = crate::formats::trellis_core::read_projections(checkpoint, expected,
            |name| !name.contains(".experts.")
                && (name.starts_with("language_model.model.") || name.starts_with("language_model.lm_head.")),
            |stem| split(stem).map(|split| crate::formats::trellis_core::CoreContract {
                split, bits: 6, padded_outputs: stem.ends_with(".self_attn.kv_a_proj_with_mqa")
                    && expected.get(&format!("{stem}.weight")).is_some_and(|shape|
                        shape.first() == Some(&(cfg.kv_lora_rank + cfg.qk_rope_dim))),
            }))?;
        let mut native = BTreeMap::new();
        for (name, shape) in expected {
            let Some(stem) = name.strip_suffix(".weight") else { continue; };
            if !stem.contains(".self_attn.") || split(stem).is_none()
                || projections.contains_key(stem) || !checkpoint.contains_tensor(name) { continue; }
            let group = stem.rsplit_once('.').context("Kimi attention group")?.0;
            if !projections.keys().any(|s| s.rsplit_once('.').is_some_and(|(parent, _)| parent == group)) {
                continue;
            }
            let selected = checkpoint.projection(stem)?;
            let crate::plan::checkpoint::ProjectionStorage::Native { dtype, shape: actual } = selected.storage else { continue; };
            ensure!(actual == *shape && shape.len() == 2 && dtype == cuteafd_core::DType::Bf16,
                "{stem}: Kimi native attention requires BF16 {shape:?}, found {dtype:?} {actual:?}");
            native.insert(stem.into(), crate::families::glm5_flash::core::NativeProjection {
                shape: [shape[0], shape[1]], split: split(stem).unwrap(),
                fp8: checkpoint.attention_mode() == crate::plan::checkpoint::AttentionMode::Fp8Compact });
        }
        let inventory = Self { projections, native };
        inventory.validate_runtime_groups(cfg)?;
        Ok(inventory)
    }

    /// Attention dispatches each validated projection independently. Shared
    /// FFNs and latent projections retain their existing whole-group contract.
    pub fn validate_runtime_groups(&self,cfg:&KimiK3Config)->Result<()> {
        for layer in 0..cfg.layers {
            let root=format!("language_model.model.layers.{layer}");
            let attention:&[&str]=match cfg.attention[layer] {
                super::config::KimiAttention::Kda=>&["q_proj","k_proj","v_proj","g_proj","o_proj"],
                super::config::KimiAttention::Mla=>&["q_a_proj","kv_a_proj_with_mqa","q_b_proj","g_proj","o_proj"],
            };
            let stems = attention.iter().map(|leaf| format!("{root}.self_attn.{leaf}")).collect::<Vec<_>>();
            let staged = stems.iter().any(|stem| self.projections.contains_key(stem) || self.native.contains_key(stem));
            ensure!(!staged || stems.iter().all(|stem| self.projections.contains_key(stem) || self.native.contains_key(stem)),
                "Kimi attention group lacks a validated per-projection K6/BF16 operand: {root}.self_attn");
            if !cfg.dense[layer] {
                self.complete_group("shared FFN",["gate_proj","up_proj","down_proj"].iter()
                    .map(|leaf|format!("{root}.block_sparse_moe.shared_experts.{leaf}")))?;
                self.complete_group("latent",["routed_expert_down_proj","routed_expert_up_proj"].iter()
                    .map(|leaf|format!("{root}.block_sparse_moe.{leaf}")))?;
            }
        }
        Ok(())
    }
    fn complete_group(&self,group:&str,stems:impl IntoIterator<Item=String>)->Result<()> {
        let (native,bf16):(Vec<_>,Vec<_>)=stems.into_iter().partition(|stem|self.get_weight(stem).is_some());
        ensure!(native.is_empty() || bf16.is_empty(),
            "Kimi {group} mixes per-tensor K6/BF16 storage; only whole-group fallback is supported; K6: {}; BF16: {}",
            native.join(", "),bf16.join(", "));
        Ok(())
    }

    pub fn get_weight(&self, weight: &str) -> Option<&CoreProjection> {
        self.projections.get(weight.strip_suffix(".weight").unwrap_or(weight))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_keep_list_is_not_an_implicit_all_linear_rule() {
        for leaf in ["self_attn.kv_b_proj", "self_attn.f_a_proj", "self_attn.f_b_proj", "self_attn.b_proj",
            "self_attn.q_conv1d", "self_attn.k_conv1d", "self_attn.v_conv1d", "self_attn.o_norm",
            "block_sparse_moe.gate", "block_sparse_moe.routed_expert_norm", "mlp.gate_proj", "mlp.up_proj",
            "mlp.down_proj", "input_layernorm", "self_attention_res_proj", "mlp_res_proj"] {
            assert_eq!(split(&format!("language_model.model.layers.0.{leaf}")), None, "{leaf}");
        }
        for stem in ["language_model.model.embed_tokens", "language_model.model.norm", "vision_tower.proj", "mm_projector.proj"] {
            assert_eq!(split(stem), None);
        }
    }

    #[test]
    fn runtime_groups_allow_bf16_kda_fallback_but_name_partial_groups() -> Result<()> {
        let cfg=KimiK3Config::from_hf(&crate::plan::testing::kimi_k3_config())?;
        let mut core=CoreInventory::default();
        core.validate_runtime_groups(&cfg)?;
        let mut add=|leaf:&str| {
            let stem=format!("language_model.model.layers.1.{leaf}");
            core.projections.insert(stem.clone(),CoreProjection {stem:stem.clone(),bits:6,
                codebook:crate::formats::exl3_storage::Exl3Codebook::Mcg,shape:[128,128],stored_outputs:128,
                split:split(&stem).unwrap(),source_bytes:0});
        };
        for leaf in ["gate_proj","up_proj","down_proj"] {add(&format!("block_sparse_moe.shared_experts.{leaf}"));}
        // No KDA projections selected: the unchanged fused BF16 attention is valid.
        core.validate_runtime_groups(&cfg)?;
        for (layer,leaves) in [(1,vec!["self_attn.q_proj"]),
            (3,vec!["self_attn.q_a_proj"]),
            (2,vec!["block_sparse_moe.shared_experts.gate_proj"]),
            (2,vec!["block_sparse_moe.routed_expert_down_proj"])] {
            let mut partial=core.clone();
            let stem=format!("language_model.model.layers.{layer}.{}",leaves[0]);
            partial.projections.insert(stem.clone(),CoreProjection {stem:stem.clone(),bits:6,
                codebook:crate::formats::exl3_storage::Exl3Codebook::Mcg,shape:[128,128],stored_outputs:128,
                split:split(&stem).unwrap(),source_bytes:0});
            let error=partial.validate_runtime_groups(&cfg).unwrap_err().to_string();
            assert!(if leaves[0].starts_with("self_attn.") {
                error.contains("validated per-projection")
            } else {
                error.contains("only whole-group fallback") && error.contains(&stem) && error.contains("BF16:")
            }, "{error}");
        }
        Ok(())
    }

    #[test]
    fn hqa_attention_inventory_and_mode_charges_follow_selected_projections() -> Result<()> {
        use crate::plan::{checkpoint::{AttentionMode, CheckpointView}, testing};
        use crate::placement::{LayerMode, FfnMode};
        let dir = tempfile::tempdir()?;
        testing::write_kimi_core_snapshot(dir.path(), false);
        let compact_index = std::fs::read(dir.path().join("model.safetensors.index.json"))?;
        let mut index: serde_json::Value = serde_json::from_slice(&compact_index)?;
        let mut restored = Vec::new();
        for (name, dtype, shape) in testing::kimi_k3_tensors(false) {
            if !name.contains(".self_attn.") || !(name.ends_with(".o_proj.weight") || name.ends_with(".q_b_proj.weight")) {
                continue;
            }
            let stem = name.strip_suffix(".weight").unwrap();
            for suffix in ["trellis", "suh", "svh", "mcg"] {
                index["weight_map"].as_object_mut().unwrap().remove(&format!("{stem}.{suffix}"));
            }
            index["weight_map"][&name] = serde_json::json!("hqa.safetensors");
            restored.push((name, dtype, shape));
        }
        testing::write_safetensors(&dir.path().join("hqa.safetensors"), &restored);
        std::fs::write(dir.path().join("model.safetensors.index.json"), serde_json::to_vec(&index)?)?;
        let expected = testing::kimi_k3_tensors(false).into_iter().map(|(n, _, s)| (n, s)).collect();
        let mut charges = Vec::new();
        for mode in [AttentionMode::Hqa, AttentionMode::Fp8Compact] {
            let checkpoint = Checkpoint::open_view(&CheckpointView::select(dir.path(), mode)?)?;
            let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
            let core = CoreInventory::from_checkpoint(&cfg, &checkpoint, &expected)?;
            assert_eq!(core.native.len(), restored.len());
            assert!(core.native.values().all(|p| p.fp8 == (mode == AttentionMode::Fp8Compact)));
            let mut bytes = 0;
            for layer in 0..cfg.layers {
                let pair = super::super::weight_layout::LayerLayout::pair_with_core(&cfg, layer, LayerMode::HeadSplit, &core)?;
                for layout in pair {
                    for (name, native) in &layout.native {
                        assert!(!layout.operands.contains_key(name) && !layout.core.contains_key(name));
                        assert_eq!(native.ranks, 2);
                        assert_eq!(native.shape, if *name == "mla.q_b" { [9216, 1536] } else { [7168, 6144] });
                        bytes += native.bytes()?;
                    }
                }
                if cfg.attention[layer] == super::super::config::KimiAttention::Mla {
                    let whole = super::super::weight_layout::LayerLayout::pair_with_core(&cfg, layer,
                        LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }, &core)?;
                    assert!(whole[0].native.is_empty());
                    assert!(whole[1].native.values().all(|p| p.rank == 0 && p.ranks == 1));
                }
            }
            charges.push(bytes);
        }
        assert!(charges[1] < charges[0]);
        Ok(())
    }

    #[test]
    fn tp_bytes_include_the_replicated_opposite_rotation() {
        for split in [CoreSplit::Output, CoreSplit::Input, CoreSplit::Whole] {
            let p = CoreProjection { stem: "test".into(), bits: 6, codebook: crate::formats::exl3_storage::Exl3Codebook::Mcg, shape: [12288,7168], stored_outputs: 12288,
                split, source_bytes: 66099204 };
            let (s, r) = p.residency();
            assert_eq!(p.resident_bytes(2).unwrap(), s / 2 + r);
            assert_eq!(p.resident_bytes(1).unwrap(), s + r);
            assert!(p.resident_bytes(3).is_err());
            if split != CoreSplit::Whole { assert!(p.resident_bytes(2).unwrap() > (p.source_bytes - 4) / 2); }
        }
    }
}

#[cfg(test)]
mod scratch_admission_tests {
    use super::*;
    #[test]
    fn production_merged_scratch_includes_grouped_owners_once() -> Result<()> {
        for (rows,expected) in [(2048,612_368_864),(4096,931_135_968)] {
            let mut layout=ProjectionScratchLayout::grouped(7168,6144,rows,4)?;
            // Hook-local owners use the unit ceiling, not the in-flight wave span.
            assert_eq!(layout.extents["rotated"],rows*57_344);
            assert_eq!(layout.extents["accumulator"],rows*98_304);
            for (k,n) in [(7168,6144),(6144,7168),(7168,3072),(3072,7168),(7168,3584),
                (3584,7168),(7168,1536),(7168,640),(1536,18432),(1536,9216),
                (7168,12288),(12288,7168)] {
                layout.merge(&ProjectionScratchLayout::new(k,n,rows)?)?;
            }
            let body=layout.bytes()?;
            let mut head=ProjectionScratchLayout::new(7168,163840,64)?;
            head.rows=rows;
            layout.merge(&head)?;
            assert_eq!(layout.bytes()?,expected);
            assert_eq!(layout.bytes()? - body,204_832_768);
            assert_eq!(layout.extents["partial"],293_601_280);
        }
        Ok(())
    }
    #[test]
    fn whole_lane_retains_full_kda_and_shared_ffn_operands() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&crate::plan::testing::kimi_k3_config())?;
        let scratch = super::super::allocation::NativeScratch { kda: 1024, mla: [4096; 3] };
        for (rows, unit) in [(64, 64), (8256, 2752)] {
            let stages = intermediate_extents_for_ranks(rows, unit, 1)?;
            for name in ["K6 q", "K6 k", "K6 v"] { assert_eq!(stages[name], unit * 12288 * 2); }
            assert_eq!(stages["K6 FFN gate"], unit * 6144 * 2);
            let whole = super::super::allocation::NativeLaneLayout::for_ranks(&cfg, rows, unit,
                1_048_576, scratch, 1)?;
            let half = super::super::allocation::NativeLaneLayout::new(&cfg, rows, unit, 1_048_576, scratch)?;
            assert_eq!(whole.operands["KDA normalized"], 2 * half.operands["KDA normalized"]);
            assert_eq!(whole.operands["gate-up"], 2 * half.operands["gate-up"]);
            assert_eq!(whole.operands["activated"], 2 * half.operands["activated"]);
            let native = whole.with_core_stages(stages)?;
            assert!(native.bytes()? > half.bytes()?);
            assert!(native.core_phases().iter().flatten().all(|name| native.operands.contains_key(name)));
        }
        assert!(intermediate_extents_for_ranks(64, 64, 0).is_err());
        assert!(super::super::allocation::NativeLaneLayout::for_ranks(&cfg, 64, 64,
            1_048_576, scratch, 3).is_err());
        Ok(())
    }

    #[test]
    fn retained_partials_are_bf16_but_meet_accumulates_f32() -> Result<()> {
        for (rows,unit) in [(1,1),(64,64),(4096,4096),(8192,4096),(8256,2752)] {
            let stages=intermediate_extents(rows,unit)?;
            for name in ["K6 attention partial","K6 FFN partial"] {
                assert_eq!(stages[name],rows*7168*2);
            }
            assert_eq!(stages["K6 reduced"],rows*7168*4);
            assert_eq!(stages["K6 FFN gate"],unit*3072*2);
            assert_eq!(exchange_recv_bytes(unit)?,unit*7168*2);
            let delta=lane_extents(rows,unit,1024,4096,2048,2048)?;
            let cfg=KimiK3Config::from_hf(&crate::plan::testing::kimi_k3_config())?;
            let base=super::super::allocation::NativeLaneLayout::new(&cfg,rows,unit,1_048_576,
                super::super::allocation::NativeScratch{kda:1024,mla:[4096;3]})?;
            let native=base.clone().with_core_stages(delta)?;
            for name in super::super::allocation::CORE_MEET_PLANES {
                assert!(!native.arena_offsets.contains_key(name),"retained meet plane must not alias scratch");
            }
            let owners=native.bytes()?;
            // The three retained row planes alone require 2 + 2 + 4 bytes per element.
            assert!(owners>=base.bytes()?+rows*7168*8);
        }
        assert!(exchange_recv_bytes(0).is_err());
        assert!(exchange_recv_bytes(4097).is_err());
        Ok(())
    }
    #[test]
    fn native_scratch_bounds_partials_and_removes_materialized_weights() -> Result<()> {
        for rows in [1,8,16,63,64,129,130,4096] {
            let layout=ProjectionScratchLayout::new(7168,6144,rows)?;
            assert_eq!(layout.extents["partial"],rows.min(129)*7*6144*4);
            assert_eq!(layout.extents.len(),5);
            assert_eq!(layout.storage()?.owners["route arena"],layout.extents["partial"]);
            assert_eq!(exchange_recv_bytes(rows)?,rows*7168*2);
        }
        assert!(ProjectionScratchLayout::new(7168,6144,0).is_err());
        let grouped=ProjectionScratchLayout::grouped(7168,6144,4096,4)?;
        assert_eq!(grouped.extents["rotated"],4*4096*7168*2);
        assert_eq!(grouped.extents["table"],32);
        assert_eq!(grouped.extents["partial"],4*129*7*6144*4);
        assert_eq!(grouped.extents["accumulator"],4*4096*6144*4);
        let mut merged=ProjectionScratchLayout::new(7168,6144,4096)?;
        merged.merge(&grouped)?;
        assert_eq!(merged,grouped);
        assert!(ProjectionScratchLayout::grouped(7168,6144,8,2).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod draft_head_tests {
    use super::*;
    use crate::families::kimi_k3::weight_layout::GlobalLayout;
    use crate::formats::dense_trellis::TrellisSide;

    /// The published native head geometry: K7168/N163840.
    fn head_policy(book: &str) -> serde_json::Value {
        serde_json::json!({"abi_version": 1, "row_policies": [{"input_features": 7168, "output_features": 163840,
            "sharded_side": 1, "codebook": book}]})
    }

    /// Real-geometry K6 core (every core projection K6, the head included, as the
    /// published Core-K6 checkpoint ships). Neither TP2 rank holds a BF16 head:
    /// rank 0 keeps the BF16 embedding, rank 1 holds only the K6 head, and a
    /// drafter must own a K6 head scratch on rank 1 instead of borrowing one.
    #[test]
    fn k6_head_has_no_bf16_operand_on_either_rank_and_charges_the_drafter() -> Result<()> {
        let dir = tempfile::tempdir()?;
        crate::plan::testing::write_kimi_core_snapshot(dir.path(), false);
        let checkpoint = Checkpoint::open(dir.path())?;
        let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
        let core = crate::plan::families::kimi::core_inventory(&checkpoint)?;
        let head = &core.projections["language_model.lm_head"];
        assert_eq!((head.bits, head.shape, head.stored_outputs), (6, [163840, 7168], 163840));
        let pair = GlobalLayout::pair_with_core(&cfg, &core)?;
        assert!(pair.iter().all(|rank| !rank.operands.contains_key("head")));
        assert!(pair[0].operands.contains_key("embedding") && pair[0].core_head.is_none());
        let resident = pair[1].core_head.as_ref().context("rank 1 K6 head")?;
        assert_eq!((resident.rank, resident.ranks, resident.side), (0, 1, TrellisSide::Output));
        let book = match head.codebook { crate::formats::exl3_storage::Exl3Codebook::Mcg => "mcg",
            crate::formats::exl3_storage::Exl3Codebook::Mul1 => "mul1" };
        // C8 x block 8: the drafter's head rows.
        let demand = draft_head_demand(&core, 64, &head_policy(book))?.context("K6 draft head demand")?;
        let expected = ProjectionScratchLayout::new(7168, 163840, 64)?.bytes()?;
        assert_eq!((demand.gpu, demand.bytes), (1, expected as u64));
        assert!(draft_head_demand(&core, 65, &head_policy(book)).is_err());
        assert!(draft_head_demand(&core, 64, &head_policy(if book == "mcg" { "mul1" } else { "mcg" })).is_err());
        assert!(draft_head_demand(&CoreInventory::default(), 64, &head_policy(book))?.is_none());
        let mut head_only=core.clone();
        head_only.projections.retain(|stem,_| stem=="language_model.lm_head");
        for rows in [1,64,256,4096] {
            let layouts=projection_scratch(&cfg,&head_only,rows,&head_policy(book))?;
            assert!(layouts[0].is_none());
            let expected=ProjectionScratchLayout::new(7168,163840,rows.min(64))?;
            assert_eq!(layouts[1].as_ref().map(|p|(&p.extents,p.rows)),Some((&expected.extents,expected.rows)));
        }
        Ok(())
    }
}
