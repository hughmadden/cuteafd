//! GLM's explicit header-native dense core contract. Routed experts and the
//! disabled native MTP stage are outside the coordinator core inventory.
use super::GlmDsaConfig;
use crate::{formats::trellis_core::{CoreContract, read_projections}, plan::Checkpoint};
pub use crate::formats::trellis_core::{CoreProjection, CoreSplit, ProjectionScratchLayout};
use anyhow::{ensure, Context, Result};
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

/// Serialized BF16 stage operands beside the shared projection arena. Retained
/// BF16 partials and their F32 sum live in row slots and outlive the producer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreStageLayout { pub extents: BTreeMap<&'static str, usize> }
impl CoreStageLayout {
    pub fn new(cfg: &GlmDsaConfig, rows: usize, ranks: usize, owned_attention: bool) -> Result<Self> {
        ensure!((1..=4096).contains(&rows) && matches!(ranks, 1 | 2), "GLM K6 stage geometry");
        let heads = cfg.heads / if owned_attention { 1 } else { ranks };
        let intermediate = cfg.dense_intermediate.max(cfg.moe_intermediate * cfg.shared_experts) / ranks;
        let bf16 = |width: usize| rows.checked_mul(width).and_then(|n| n.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("GLM K6 stage extent overflow"));
        Ok(Self { extents: BTreeMap::from([
            ("qkv", bf16(cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim)?),
            ("q", bf16((heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim))
                .max(cfg.q_lora_rank).max((cfg.kv_lora_rank + cfg.qk_rope_head_dim).next_multiple_of(128)))?),
            ("gate", bf16(intermediate)?), ("up", bf16(intermediate)?),
            ("hidden", bf16(intermediate)?), ("values", bf16(heads * cfg.v_head_dim)?),
        ]) })
    }
    pub fn bytes(&self) -> Result<usize> {
        self.extents.values().try_fold(0usize, |total, &n| total.checked_add(n)
            .ok_or_else(|| anyhow::anyhow!("GLM K6 stage total overflow")))
    }
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

    /// Header-selected compressed owners for one layer. Whole attention follows
    /// its existing owner; FFNs keep the ordinary intermediate split.
    pub fn layer_weights(&self, cfg: &GlmDsaConfig, layer: usize, ranks: usize,
        owner: Option<u8>) -> Result<Vec<BTreeMap<&'static str, crate::formats::trellis_core::CoreWeightLayout>>> {
        use crate::formats::{dense_trellis::TrellisSide, trellis_core::CoreWeightLayout};
        ensure!(layer < cfg.layers && matches!(ranks, 1 | 2)
            && owner.is_none_or(|gpu| ranks == 2 && usize::from(gpu) < ranks),
            "GLM native core layer ownership");
        let root = format!("model.layers.{layer}");
        let mlp = if layer < cfg.first_moe_layer { "mlp" } else { "mlp.shared_experts" };
        let mut layouts = (0..ranks).map(|_| BTreeMap::new()).collect::<Vec<_>>();
        for (key, leaf) in [("q_a", "self_attn.q_a_proj".to_owned()),
            ("kv_a", "self_attn.kv_a_proj_with_mqa".to_owned()),
            ("q_b", "self_attn.q_b_proj".to_owned()), ("o", "self_attn.o_proj".to_owned()),
            ("gate", format!("{mlp}.gate_proj")), ("up", format!("{mlp}.up_proj")),
            ("down", format!("{mlp}.down_proj"))] {
            let Some(projection) = self.get_weight(&format!("{root}.{leaf}")) else { continue; };
            let attention = leaf.starts_with("self_attn.");
            let world = if projection.split == CoreSplit::Whole || attention && owner.is_some() { 1 } else { ranks };
            let side = if projection.split == CoreSplit::Input && world > 1 { TrellisSide::Input }
                else { TrellisSide::Output };
            for (rank, layouts) in layouts.iter_mut().enumerate() {
                if attention && owner.is_some_and(|gpu| usize::from(gpu) != rank) { continue; }
                layouts.insert(key, CoreWeightLayout::new(projection, side,
                    if world == 1 { 0 } else { rank }, world)?);
            }
        }
        Ok(layouts)
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

}

/// Native storage beside the existing GLM workspaces. Head scratch is never
/// merged into the serialized body arena: target and drafter retain owners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreRankLayout {
    pub body: Vec<crate::formats::trellis_core::CoreWeightLayout>,
    pub stages: Option<[CoreStageLayout; 2]>,
    pub decode: Option<ProjectionScratchLayout>,
    pub prefill: Option<ProjectionScratchLayout>,
    /// Decode and prefill target launches are serialized on the rank-0 stream.
    pub target_head: Option<ProjectionScratchLayout>,
    pub draft_head: Option<ProjectionScratchLayout>,
    pub retained_rows: usize,
    pub receive_suffix: usize,
}
impl CoreRankLayout {
    pub fn bytes(&self, hidden: usize) -> Result<u64> {
        let mut bytes = self.receive_suffix;
        let mut add = |n: usize| -> Result<()> {
            bytes = bytes.checked_add(n).context("GLM K6 allocation total overflow")?;
            Ok(())
        };
        for layout in self.decode.iter().chain(&self.prefill).chain(&self.target_head).chain(&self.draft_head) {
            add(layout.bytes()?)?;
        }
        if let Some(stages) = &self.stages { for stage in stages { add(stage.bytes()?)?; } }
        // Two BF16 retained partials and one F32 ordered reduction output.
        add(self.retained_rows.checked_mul(hidden).and_then(|n| n.checked_mul(2 + 2 + 4))
            .context("GLM K6 retained plane extent overflow")?)?;
        Ok(bytes as u64)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreCharges { pub ranks: Vec<CoreRankLayout>, pub programs: Vec<String> }
impl CoreCharges {
    pub fn from_package(cfg: &GlmDsaConfig, core: &CoreInventory, layers: usize,
        ranks: usize, owners: Option<&[u8]>, unit: usize, slots: usize,
        draft_rows: Option<usize>, manifest: &serde_json::Value) -> Result<Self> {
        use crate::formats::{dense_trellis::TrellisSide, trellis_core::CoreWeightLayout};
        ensure!(layers > 0 && layers <= cfg.layers && matches!(ranks, 1 | 2)
            && (1..=4096).contains(&unit) && (1..=4).contains(&slots)
            && owners.is_none_or(|o| ranks == 2 && o.len() == layers), "GLM K6 admission geometry");
        let span = usize::try_from(crate::placement::families::glm5::prefill_span(unit as u64, slots as u64))?;
        let mut bodies = vec![Vec::new(); ranks];
        for layer in 0..layers {
            for (rank, weights) in core.layer_weights(cfg, layer, ranks, owners.map(|o| o[layer]))?.into_iter().enumerate() {
                bodies[rank].extend(weights.into_values());
            }
        }
        let head = core.get_weight("lm_head").map(|p| CoreWeightLayout::new(p, TrellisSide::Output, 0, 1)).transpose()?;
        let mut programs = std::collections::BTreeSet::new();
        let mut geometries = std::collections::BTreeSet::new();
        for weight in bodies.iter().flatten().chain(head.iter()) {
            let side = if weight.side == TrellisSide::Input { 2 } else { 1 };
            if geometries.insert((weight.shape, side, weight.codebook.marker())) {
                for name in policy_programs(weight, manifest)? { programs.insert(name); }
            }
        }
        if bodies.iter().any(|body| !body.is_empty()) {
            let entries = manifest["programs"].as_array().context("GLM K6 stage program inventory")?;
            for prefix in if owners.is_some() { vec!["glm", "glm2"] }
                else if ranks == 2 { vec!["glm2"] } else { vec!["glm"] } {
                for inter in [cfg.dense_intermediate / ranks, cfg.moe_intermediate * cfg.shared_experts / ranks] {
                    let name = format!("{prefix}_stage_swiglu_i{inter}");
                    // Whole-attention stages still use the split FFN namespace.
                    if prefix == "glm" && ranks == 2 { continue; }
                    ensure!(entries.iter().filter(|p| p["name"] == name).count() == 1, "GLM K6 requires {name}");
                }
                let context = prefix == "glm2" && owners.is_none()
                    && crate::placement::families::glm5::context_programs(
                        entries.iter().filter_map(|p| p["name"].as_str()),
                        |name| entries.iter().any(|p| p["name"] == name).then_some(0)).is_some();
                for cap in ["m64", "m4096"] {
                    for stage in ["rank_pack", "absorb_bf16", "query_rope", "values_bf16"].into_iter()
                        .chain(context.then_some("absorb_bf16_context"))
                        .chain(context.then_some("query_rope_context")) {
                        let name = format!("{prefix}_stage_{stage}_{cap}");
                        ensure!(entries.iter().filter(|p| p["name"] == name).count() == 1, "GLM K6 requires {name}");
                    }
                }
            }
        }
        let split = bodies.iter().flatten().any(|w| w.side == TrellisSide::Input);
        let scratch = |body: &[CoreWeightLayout], rows| -> Result<Option<ProjectionScratchLayout>> {
            let mut result: Option<ProjectionScratchLayout> = None;
            for weight in body {
                let layout = ProjectionScratchLayout::new(weight.shape[1], weight.shape[0], rows)?;
                if let Some(result) = &mut result { result.merge(&layout)?; } else { result = Some(layout); }
            }
            Ok(result)
        };
        let head_scratch = |rows| head.as_ref().map(|head|
            ProjectionScratchLayout::new(head.shape[1], head.shape[0], rows)).transpose();
        let mut layouts = Vec::new();
        for (rank, body) in bodies.into_iter().enumerate() {
            let stages = if body.is_empty() { None } else {
                Some([CoreStageLayout::new(cfg, 64, ranks, owners.is_some())?,
                    CoreStageLayout::new(cfg, unit, ranks, owners.is_some())?])
            };
            let target_head = if rank == 0 { head_scratch(64)? } else { None };
            let draft_head = if rank == 0 { draft_rows.map(head_scratch).transpose()?.flatten() } else { None };
            layouts.push(CoreRankLayout {
                decode: scratch(&body, 64)?, prefill: scratch(&body, unit)?, stages,
                retained_rows: if body.is_empty() { 0 } else { 64usize.checked_add(span).context("GLM K6 row span")? },
                receive_suffix: if ranks == 2 && split { 4usize.checked_mul(slots)
                    .and_then(|n| n.checked_mul(unit.max(64))).and_then(|n| n.checked_mul(cfg.hidden))
                    .and_then(|n| n.checked_mul(2)).context("GLM K6 receive suffix overflow")? } else { 0 },
                body, target_head, draft_head,
            });
        }
        let result = Self { ranks: layouts, programs: programs.into_iter().collect() };
        result.bytes(cfg.hidden)?;
        Ok(result)
    }
    pub fn bytes(&self, hidden: usize) -> Result<Vec<u64>> {
        self.ranks.iter().map(|rank| rank.bytes(hidden)).collect()
    }
    /// Native GLM explicitly selects its BF16 stages. Add only the neutral
    /// policies actually selected by the checkpoint, not every Kimi export.
    pub fn program_set(&self, manifest: &serde_json::Value, split: bool) -> crate::placement::ProgramSet {
        let mut set = super::DensePrograms::Trellis.program_set(manifest, split);
        set.names.extend(self.programs.iter().cloned());
        set.names.sort(); set.names.dedup();
        set
    }
}

fn policy_programs(weight: &crate::formats::trellis_core::CoreWeightLayout,
    manifest: &serde_json::Value) -> Result<[String; 3]> {
    use cuteafd_ffi::dense_trellis::{validate_package_geometry, TrellisDType};
    let book = match weight.codebook { crate::formats::exl3_storage::Exl3Codebook::Mcg => "mcg",
        crate::formats::exl3_storage::Exl3Codebook::Mul1 => "mul1" };
    let side = if weight.side == crate::formats::dense_trellis::TrellisSide::Input { 2 } else { 1 };
    validate_package_geometry(manifest, weight.shape[1], weight.shape[0], side, book, 4096,
        if weight.stem == "lm_head" { TrellisDType::F32 } else { TrellisDType::Bf16 })
        .with_context(|| format!("{}: GLM native K6 package", weight.stem))
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

    fn package(cfg: &GlmDsaConfig, core: &CoreInventory) -> Result<serde_json::Value> {
        use crate::formats::{dense_trellis::TrellisSide, trellis_core::CoreWeightLayout};
        use serde_json::json;
        let mut weights = Vec::new();
        for ranks in [1, 2] { for owner in if ranks == 1 { vec![None] } else { vec![None, Some(0), Some(1)] } {
            for layer in 0..cfg.layers { weights.extend(core.layer_weights(cfg, layer, ranks, owner)?.into_iter()
                .flat_map(BTreeMap::into_values)); }
        } }
        if let Some(head) = core.get_weight("lm_head") {
            weights.push(CoreWeightLayout::new(head, TrellisSide::Output, 0, 1)?);
        }
        let mut policies = BTreeMap::new();
        let mut programs = BTreeMap::new();
        for w in weights {
            let [n, k] = w.shape;
            let side = if w.side == TrellisSide::Input { 2 } else { 1 };
            let book = w.codebook.marker();
            let shape = format!("k{k}_n{n}_s{side}_{book}");
            let routes = [format!("input_{shape}"), format!("core_{shape}"), format!("output_{shape}")];
            policies.insert(shape.clone(), json!({"shape":shape,"input_features":k,"output_features":n,
                "sharded_side":side,"codebook":book,"native_input":routes[0],"native_program":routes[1],
                "output_program":routes[2]}));
            let output = if w.stem == "lm_head" { "f32" } else { "bf16" };
            let geometries = [json!({"features":k,"input_dtype":"bf16","max_rows":4096}),
                json!({"abi_version":1,"input_features":k,"output_features":n,"bits":6,"codebook":book,
                    "sharded_side":side,"rotation_count":1,"max_rows":4096,"split_k_blocks":8,
                    "splits":k.div_ceil(1024),"ctas_per_sm":4}),
                json!({"features":n,"output_dtype":output,"max_rows":4096})];
            let pointers = [vec!["input", "suh", "rotated", "input_block_scales"],
                vec!["rotated_inputs", "payload", "input_block_scales", "partial", "output"],
                vec!["input", "svh", "output"]];
            for i in 0..3 {
                let name = format!("dense_trellis_k6_{}", routes[i]);
                programs.insert(name.clone(), json!({"name":name,"capacity_rows":4096,"geometry":geometries[i],
                    "pointers":pointers[i].iter().map(|name| json!({"name":name})).collect::<Vec<_>>(),
                    "scalars":vec![json!({"type":"int32"}); if i == 1 { 3 } else { 2 }],
                    "scratch_bytes_at_capacity":if i == 1 { json!({"partial":4096*k.div_ceil(1024)*n*4}) } else { json!({}) }}));
            }
        }
        for (prefix, ranks) in [("glm", 1), ("glm2", 2)] {
            for inter in [cfg.dense_intermediate / ranks, cfg.moe_intermediate * cfg.shared_experts / ranks] {
                let name = format!("{prefix}_stage_swiglu_i{inter}");
                programs.insert(name.clone(), json!({"name":name}));
            }
            for cap in ["m64", "m4096"] { for stage in ["rank_pack", "absorb_bf16", "query_rope", "values_bf16",
                "absorb_bf16_context", "query_rope_context"] {
                let name = format!("{prefix}_stage_{stage}_{cap}");
                programs.insert(name.clone(), json!({"name":name}));
            } }
        }
        Ok(json!({"families":{"dense_trellis_k6":{"abi_version":1,
            "row_policies":policies.into_values().collect::<Vec<_>>()}},"programs":programs.into_values().collect::<Vec<_>>()}))
    }

    #[test]
    fn native_admission_separates_body_heads_row_reserve_and_peer_suffix() -> Result<()> {
        let mut cfg = GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config())?;
        cfg.vocab_size = 256;
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint(&cfg))?;
        let manifest = package(&cfg, &core)?;
        for ranks in [1, 2] { for owners in if ranks == 1 { vec![None] } else { vec![None, Some([0, 1])] } {
            for (unit, slots) in [(1, 1), (32, 2), (64, 1), (2048, 2), (2752, 3), (4096, 4)] {
                for draft in [None, Some(128)] {
                    let charge = CoreCharges::from_package(&cfg, &core, cfg.layers, ranks, owners.as_ref().map(|o| o.as_slice()),
                        unit, slots, draft, &manifest)?;
                    let bytes = charge.bytes(cfg.hidden)?;
                    for (rank, layout) in charge.ranks.iter().enumerate() {
                        let mut expected = 0;
                        for (actual, rows) in [layout.decode.as_ref().unwrap(), layout.prefill.as_ref().unwrap()]
                            .into_iter().zip([64, unit]) {
                            let mut scratch: Option<ProjectionScratchLayout> = None;
                            for weight in &layout.body {
                                let next = ProjectionScratchLayout::new(weight.shape[1], weight.shape[0], rows)?;
                                if let Some(scratch) = &mut scratch { scratch.merge(&next)?; } else { scratch = Some(next); }
                            }
                            assert_eq!(Some(actual), scratch.as_ref());
                            expected += actual.storage()?.owners.values().sum::<usize>();
                        }
                        assert!(layout.decode.as_ref().unwrap().extents["accumulator"] <= 64 * 16384 * 4);
                        for (stage, rows) in layout.stages.as_ref().unwrap().iter().zip([64, unit]) {
                            assert_eq!(stage, &CoreStageLayout::new(&cfg, rows, ranks, owners.is_some())?);
                            expected += stage.extents.values().sum::<usize>();
                        }
                        assert_eq!(layout.retained_rows, 64 + crate::placement::families::glm5::prefill_span(unit as u64, slots as u64) as usize);
                        expected += layout.retained_rows * cfg.hidden * (2 + 2 + 4);
                        let suffix = if ranks == 2 { 4 * slots * unit.max(64) * cfg.hidden * 2 } else { 0 };
                        assert_eq!(layout.receive_suffix, suffix);
                        expected += suffix;
                        assert_eq!(layout.target_head.is_some(), rank == 0);
                        if let Some(head) = &layout.target_head { assert_eq!(head.rows, 64); expected += head.bytes()?; }
                        assert_eq!(layout.draft_head.as_ref().map(|h| h.rows), if rank == 0 { draft } else { None });
                        expected += layout.draft_head.as_ref().map(|h| h.bytes()).transpose()?.unwrap_or(0);
                        assert_eq!(bytes[rank], expected as u64);
                    }
                    let set = charge.program_set(&manifest, ranks == 2);
                    assert!(charge.programs.iter().all(|name| set.names.contains(name)));
                }
            }
        } }
        Ok(())
    }

    #[test]
    fn native_admission_rejects_missing_or_corrupt_executable_inventory() -> Result<()> {
        let mut cfg = GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config())?;
        cfg.vocab_size = 256;
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint(&cfg))?;
        let manifest = package(&cfg, &core)?;
        let admit = |m: &serde_json::Value| CoreCharges::from_package(&cfg, &core, cfg.layers, 2, None, 2752, 3, Some(128), m);
        admit(&manifest)?;
        for corruption in ["policy", "duplicate-policy", "program", "duplicate-program", "capacity", "geometry", "dtype", "scratch", "pointer", "scalar", "stage"] {
            let mut m = manifest.clone();
            match corruption {
                "policy" => { m["families"]["dense_trellis_k6"]["row_policies"].as_array_mut().unwrap().clear(); }
                "duplicate-policy" => { let p = m["families"]["dense_trellis_k6"]["row_policies"][0].clone();
                    m["families"]["dense_trellis_k6"]["row_policies"].as_array_mut().unwrap().push(p); }
                "stage" => m["programs"].as_array_mut().unwrap().retain(|p| p["name"] != "glm2_stage_swiglu_i1024"),
                _ => {
                    let entries = m["programs"].as_array_mut().unwrap();
                    let index = entries.iter().position(|p| p["geometry"]["sharded_side"] == 2).unwrap();
                    match corruption {
                        "program" => { entries.remove(index); }
                        "duplicate-program" => { let p = entries[index].clone(); entries.push(p); }
                        "capacity" => entries[index]["capacity_rows"] = serde_json::json!(64),
                        "geometry" => entries[index]["geometry"]["bits"] = serde_json::json!(4),
                        "dtype" => entries[index]["geometry"]["input_dtype"] = serde_json::json!("bf16"),
                        "scratch" => entries[index]["scratch_bytes_at_capacity"]["partial"] = serde_json::json!(8),
                        "pointer" => entries[index]["pointers"][0]["name"] = serde_json::json!("wrong"),
                        "scalar" => entries[index]["scalars"][0]["type"] = serde_json::json!("int64"),
                        _ => unreachable!(),
                    }
                }
            }
            assert!(admit(&m).is_err(), "accepted {corruption}");
        }
        for (ranks, unit, slots, draft) in [(0, 64, 1, None), (3, 64, 1, None), (2, 0, 1, None),
            (2, 4097, 1, None), (2, 64, 0, None), (2, 64, 5, None), (2, 64, 1, Some(4097))] {
            assert!(CoreCharges::from_package(&cfg, &core, cfg.layers, ranks, None, unit, slots, draft, &manifest).is_err());
        }
        Ok(())
    }

    #[test]
    fn native_admission_tracks_head_only_body_only_and_partial_layer_loads() -> Result<()> {
        let mut cfg = GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config())?;
        cfg.vocab_size = 256;
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint(&cfg))?;
        let manifest = package(&cfg, &core)?;
        for which in ["head", "body", "none", "partial"] {
            let mut selected = core.clone();
            selected.projections.retain(|stem, _| match which { "head" => stem == "lm_head", "body" => stem != "lm_head",
                "none" => false, _ => true });
            let layers = if which == "partial" { 1 } else { cfg.layers };
            let c = CoreCharges::from_package(&cfg, &selected, layers, 2, Some(&[0, 1][..layers]), 2048, 2, Some(128), &manifest)?;
            for (rank, c) in c.ranks.iter().enumerate() {
                let body = which == "body" || which == "partial";
                assert_eq!(c.decode.is_some(), body);
                assert_eq!(c.prefill.is_some(), body);
                assert_eq!(c.stages.is_some(), body);
                assert_eq!(c.retained_rows > 0, body);
                assert_eq!(c.receive_suffix > 0, body);
                assert_eq!(c.target_head.is_some(), rank == 0 && matches!(which, "head" | "partial"));
                if which == "none" { assert_eq!(c.bytes(cfg.hidden)?, 0); }
            }
        }
        Ok(())
    }

    #[test]
    fn native_operand_admission_preserves_logical_crop_and_keep_list() -> Result<()> {
        use crate::plan::{family::Family, FamilyModel, format::{Encoding, QuantOperand}, spec::{Component, TensorRole}};
        let mut cfg = GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config())?;
        cfg.vocab_size = 256;
        let checkpoint = checkpoint(&cfg);
        let model = crate::plan::families::glm::GLM5.open(&checkpoint)?;
        let operand = |logical| QuantOperand { encoding: Encoding::Exl3 { bits: 6 }, container: "I16".into(), logical,
            stored: vec![], scale: None, global_scale: None, input_scale: None, companions: vec![] };
        for (name, shape) in expected(&cfg) {
            let stem = name.strip_suffix(".weight").unwrap();
            let role = crate::plan::families::glm::GLM5.classify(model.spec(), &name).unwrap();
            let mut op = operand(if stem.ends_with("kv_a_proj_with_mqa") { vec![640, cfg.hidden] } else { shape.clone() });
            model.accepts(&role, stem, &mut op).map_err(anyhow::Error::msg)?;
            assert_eq!(op.logical, shape);
        }
        for stem in ["model.layers.0.self_attn.kv_b_proj", "model.layers.0.self_attn.indexer.wq_b", "model.layers.0.mlp.gate"] {
            assert!(model.accepts(&TensorRole::new(Component::Attention), stem, &mut operand(vec![128, 6144])).is_err());
        }
        assert!(model.accepts(&TensorRole::new(Component::Speculator), "model.layers.2.self_attn.q_a_proj", &mut operand(vec![2048, 6144])).is_err());
        Ok(())
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
    fn native_layer_owners_match_residency_for_head_and_owned_attention() -> Result<()> {
        use crate::formats::dense_trellis::TrellisSide;
        let mut cfg = GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config())?;
        cfg.vocab_size = 256;
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint(&cfg))?;
        for layer in 0..cfg.layers {
            for (ranks, owner) in [(1, None), (2, None), (2, Some(0)), (2, Some(1))] {
                let layouts = core.layer_weights(&cfg, layer, ranks, owner)?;
                assert_eq!(layouts.len(), ranks);
                for (rank, weights) in layouts.iter().enumerate() {
                    let owns_attention = owner.is_none_or(|gpu| usize::from(gpu) == rank);
                    assert_eq!(weights.len(), if owns_attention { 7 } else { 3 });
                    for (&key, weight) in weights {
                        let projection = &core.projections[&weight.stem];
                        assert_eq!(weight.bytes as u64, projection.resident_bytes(weight.ranks)?);
                        assert_eq!(weight.rank, if weight.ranks == 1 { 0 } else { rank });
                        assert_eq!(weight.side, if ranks == 2 && (key == "down" || key == "o" && owner.is_none()) {
                            TrellisSide::Input
                        } else { TrellisSide::Output });
                        assert!(!weight.stem.contains("indexer") && !weight.stem.contains("kv_b"));
                    }
                    if owns_attention { assert_eq!(weights["kv_a"].shape, [640, cfg.hidden]); }
                }
            }
        }
        for (layer, ranks, owner) in [(cfg.layers, 1, None), (0, 0, None), (0, 1, Some(0)), (0, 2, Some(2))] {
            assert!(core.layer_weights(&cfg, layer, ranks, owner).is_err());
        }
        let ordinary = CoreInventory::default().layer_weights(&cfg, 0, 2, None)?;
        assert!(ordinary.iter().all(BTreeMap::is_empty));
        Ok(())
    }

    #[test]
    fn stage_layout_tracks_whole_attention_but_keeps_ffns_split() -> Result<()> {
        let cfg = GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config())?;
        for rows in [1, 8, 64, 4096] {
            for ranks in [1, 2] {
                let split = CoreStageLayout::new(&cfg, rows, ranks, false)?;
                let owned = CoreStageLayout::new(&cfg, rows, ranks, true)?;
                let h = cfg.heads / ranks;
                assert_eq!(split.extents["values"], rows * h * cfg.v_head_dim * 2);
                assert_eq!(owned.extents["values"], rows * cfg.heads * cfg.v_head_dim * 2);
                for name in ["gate", "up", "hidden"] {
                    assert_eq!(split.extents[name], rows * cfg.dense_intermediate / ranks * 2);
                    assert_eq!(owned.extents[name], split.extents[name]);
                }
                assert_eq!(split.extents["qkv"], rows * 2624 * 2);
                assert!(split.extents["q"] >= rows * 640 * 2);
                assert_eq!(split.bytes()?, split.extents.values().sum::<usize>());
            }
        }
        for (rows, ranks) in [(0, 1), (4097, 1), (64, 0), (64, 3)] {
            assert!(CoreStageLayout::new(&cfg, rows, ranks, false).is_err());
        }
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
