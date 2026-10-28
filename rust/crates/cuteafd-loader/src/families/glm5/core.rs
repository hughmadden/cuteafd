//! GLM's explicit header-native dense core contract. Routed experts and the
//! disabled native MTP stage are outside the coordinator core inventory.
use super::GlmDsaConfig;
use crate::{formats::trellis_core::{CoreContract, read_projections}, plan::Checkpoint};
pub use crate::formats::trellis_core::{CoreProjection, CoreSplit, ProjectionScratchLayout};
use anyhow::{ensure, Result};
use std::collections::BTreeMap;

pub fn split(cfg: &GlmDsaConfig, stem: &str) -> Option<CoreSplit> {
    if stem == "lm_head" { return Some(CoreSplit::Whole); }
    let (layer, leaf) = stem.strip_prefix("model.layers.")?.split_once('.')?;
    let layer = layer.parse::<usize>().ok()?;
    if layer >= cfg.layers { return None; }
    match leaf {
        "self_attn.q_a_proj" | "self_attn.kv_a_proj_with_mqa" => Some(CoreSplit::Whole),
        "self_attn.q_b_proj" => Some(CoreSplit::Output),
        "self_attn.o_proj" => Some(CoreSplit::Input),
        "mlp.gate_proj" | "mlp.up_proj" if layer < cfg.first_moe_layer => Some(CoreSplit::Output),
        "mlp.down_proj" if layer < cfg.first_moe_layer => Some(CoreSplit::Input),
        "mlp.shared_experts.gate_proj" | "mlp.shared_experts.up_proj" if layer >= cfg.first_moe_layer => Some(CoreSplit::Output),
        "mlp.shared_experts.down_proj" if layer >= cfg.first_moe_layer => Some(CoreSplit::Input),
        _ => None,
    }
}

pub fn expected(cfg: &GlmDsaConfig) -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::from([("lm_head.weight".into(), vec![cfg.vocab_size, cfg.hidden])]);
    for layer in 0..cfg.layers {
        let root = format!("model.layers.{layer}");
        for (leaf, n, k) in [
            ("q_a_proj", cfg.q_lora_rank, cfg.hidden),
            ("kv_a_proj_with_mqa", cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden),
            ("q_b_proj", cfg.heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), cfg.q_lora_rank),
            ("o_proj", cfg.hidden, cfg.heads * cfg.v_head_dim),
        ] { shapes.insert(format!("{root}.self_attn.{leaf}.weight"), vec![n, k]); }
        let (mlp, inter) = if layer < cfg.first_moe_layer { ("mlp", cfg.dense_intermediate) }
            else { ("mlp.shared_experts", cfg.moe_intermediate * cfg.shared_experts) };
        for (leaf, n, k) in [("gate_proj", inter, cfg.hidden), ("up_proj", inter, cfg.hidden),
            ("down_proj", cfg.hidden, inter)] {
            shapes.insert(format!("{root}.{mlp}.{leaf}.weight"), vec![n, k]);
        }
    }
    shapes
}

#[derive(Debug, Default, Clone)]
pub struct CoreInventory { pub projections: BTreeMap<String, CoreProjection> }

impl CoreInventory {
    pub fn from_checkpoint(cfg: &GlmDsaConfig, checkpoint: &Checkpoint) -> Result<Self> {
        let selected = |name: &str| name.starts_with("lm_head.") || name.strip_prefix("model.layers.")
            .and_then(|rest| rest.split_once('.'))
            .is_some_and(|(layer, leaf)| layer.parse::<usize>().is_ok_and(|layer| layer < cfg.layers)
                && !leaf.contains(".experts."));
        let projections = read_projections(checkpoint, &expected(cfg), selected,
            |stem| split(cfg, stem).map(|split| CoreContract { split, bits: 6,
                padded_outputs: stem.ends_with(".self_attn.kv_a_proj_with_mqa") }))?;
        let inventory = Self { projections };
        inventory.validate_groups(cfg)?;
        Ok(inventory)
    }

    pub fn get_weight(&self, name: &str) -> Option<&CoreProjection> {
        self.projections.get(name.strip_suffix(".weight").unwrap_or(name))
    }

    pub fn validate_groups(&self, cfg: &GlmDsaConfig) -> Result<()> {
        for layer in 0..cfg.layers {
            let root = format!("model.layers.{layer}");
            for (module, leaves) in [
                ("self_attn", &["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj"][..]),
                (if layer < cfg.first_moe_layer { "mlp" } else { "mlp.shared_experts" },
                    &["gate_proj", "up_proj", "down_proj"][..]),
            ] {
                let stems = leaves.iter().map(|leaf| format!("{root}.{module}.{leaf}")).collect::<Vec<_>>();
                let present = stems.iter().filter(|stem| self.projections.contains_key(*stem)).count();
                ensure!(present == 0 || present == stems.len(),
                    "GLM {root}.{module} mixes native and EXL3 storage; complete operator groups required");
            }
        }
        Ok(())
    }

    /// Reusable per-lane projection scratch. The head uses at most 64 rows;
    /// all transforms retain the complete stored N, including padded kv_a.
    pub fn projection_scratch(&self, ranks: usize, rows: usize, owned_attention: bool)
        -> Result<Option<ProjectionScratchLayout>> {
        let mut combined: Option<ProjectionScratchLayout> = None;
        for projection in self.projections.values() {
            let world = if projection.split == CoreSplit::Whole
                || owned_attention && projection.stem.contains(".self_attn.") { 1 } else { ranks };
            projection.resident_bytes(world)?;
            let mut n = projection.stored_outputs;
            let mut k = projection.shape[1];
            if projection.split == CoreSplit::Output { n /= world; }
            if projection.split == CoreSplit::Input { k /= world; }
            let mut layout = ProjectionScratchLayout::new(k, n,
                if projection.stem == "lm_head" { rows.min(64) } else { rows })?;
            layout.rows = rows;
            if let Some(combined) = &mut combined { combined.merge(&layout)?; }
            else { combined = Some(layout); }
        }
        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(cfg: &GlmDsaConfig) -> crate::plan::Checkpoint {
        use crate::plan::checkpoint::{Checkpoint, CheckpointTensor};
        use crate::SafetensorsTensorMetadata;
        use cuteafd_core::DType;
        let mut tensors=Vec::new();
        for (name,shape) in expected(cfg) {
            let stem=name.strip_suffix(".weight").unwrap();
            let n=if stem.ends_with(".kv_a_proj_with_mqa") {shape[0].next_multiple_of(128)} else {shape[0]};
            let k=shape[1];
            for (suffix,dtype,shape,width) in [("trellis",DType::I16,vec![k/16,n/16,96],2),
                ("suh",DType::F16,vec![k],2),("svh",DType::F16,vec![n],2),("mcg",DType::I32,vec![1],4)] {
                tensors.push(CheckpointTensor {shard:"model.safetensors".into(),meta:SafetensorsTensorMetadata {
                    name:format!("{stem}.{suffix}"),dtype,byte_offset:0,
                    byte_length:width*shape.iter().product::<usize>() as u64,shape}});
            }
        }
        for (name, dtype, shape) in crate::plan::testing::glm5_tensors(|_, _, _| Vec::new()) {
            let stem = name.strip_suffix(".weight");
            if stem.is_some_and(|stem| split(cfg, stem).is_some()) || name.contains("scale") { continue; }
            let dtype = if dtype == "F32" { DType::F32 } else { DType::Bf16 };
            let width = if dtype == DType::F32 { 4 } else { 2 };
            tensors.push(CheckpointTensor { shard: "model.safetensors".into(), meta: SafetensorsTensorMetadata {
                name, dtype, byte_offset: 0, byte_length: width * shape.iter().product::<usize>() as u64, shape } });
        }
        let mut config = crate::plan::testing::glm5_config();
        config["vocab_size"] = serde_json::json!(cfg.vocab_size);
        Checkpoint {snapshot:Default::default(),config,quantize_config:None,
            weight_map:tensors.iter().map(|t|(t.meta.name.clone(),t.shard.clone())).collect(),tensors,
            missing_shards:Vec::new(),shard_bytes:0}
    }

    #[test]
    fn core_inventory_charges_native_payload_vectors_and_full_padded_crop() -> Result<()> {
        let mut cfg=GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config())?;
        cfg.vocab_size=256;
        let checkpoint=checkpoint(&cfg);
        let core=CoreInventory::from_checkpoint(&cfg,&checkpoint)?;
        assert_eq!(core.projections.len(),15);
        let kv=core.get_weight("model.layers.0.self_attn.kv_a_proj_with_mqa.weight").unwrap();
        assert_eq!(kv.shape,[576,6144]);assert_eq!(kv.stored_outputs,640);
        assert_eq!(kv.resident_bytes(2)?,6144*640*6/8+2*(6144+640));
        for p in core.projections.values() {
            let (split,replicated)=p.residency();
            assert_eq!(p.resident_bytes(2)?,split/2+replicated);
            assert_eq!(p.resident_bytes(1)?,p.source_bytes-4);
        }
        let scratch=core.projection_scratch(2,4096,false)?.unwrap();
        assert!(scratch.extents["rotated"]>=4096*cfg.hidden*2);
        assert!(scratch.extents["accumulator"]>=4096*8192*4);
        assert!(scratch.bytes()?>0);
        let resident = |ranks, owners: Option<&[u8]>| -> Result<u64> {
            Ok(crate::plan::layout::glm5::resident_owned(&checkpoint, ranks, cfg.layers, owners)?
                .iter().flatten().filter(|(_, format, _)| format == "exl3-k6")
                .map(|(_, _, bytes)| bytes).sum())
        };
        assert_eq!(resident(1, None)?, core.projections.values()
            .map(|p| p.resident_bytes(1).unwrap()).sum::<u64>());
        assert_eq!(resident(2, None)?, core.projections.values().map(|p|
            p.resident_bytes(2).unwrap() * if p.stem == "lm_head" { 1 } else { 2 }).sum::<u64>());
        let owners = [0, 1];
        assert_eq!(resident(2, Some(&owners))?, core.projections.values().map(|p| {
            if p.stem.contains(".self_attn.") || p.stem == "lm_head" { p.resident_bytes(1).unwrap() }
            else { p.resident_bytes(2).unwrap() * 2 }
        }).sum::<u64>());
        let mut duplicate=checkpoint;
        duplicate.weight_map.insert("model.layers.0.self_attn.q_a_proj.weight".into(),"other.safetensors".into());
        assert!(CoreInventory::from_checkpoint(&cfg,&duplicate).is_err());
        Ok(())
    }

    #[test]
    fn glm_split_table_preserves_native_keep_list_and_ignores_mtp() -> Result<()> {
        let cfg = GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config())?;
        assert_eq!(split(&cfg, "lm_head"), Some(CoreSplit::Whole));
        for (leaf, expected) in [("q_a_proj", CoreSplit::Whole), ("kv_a_proj_with_mqa", CoreSplit::Whole),
            ("q_b_proj", CoreSplit::Output), ("o_proj", CoreSplit::Input)] {
            assert_eq!(split(&cfg, &format!("model.layers.0.self_attn.{leaf}")), Some(expected));
        }
        for leaf in ["self_attn.kv_b_proj", "self_attn.indexer.wq_b", "mlp.gate", "input_layernorm"] {
            assert_eq!(split(&cfg, &format!("model.layers.0.{leaf}")), None);
        }
        assert_eq!(split(&cfg, &format!("model.layers.{}.self_attn.q_a_proj", cfg.layers)), None);
        Ok(())
    }
}
