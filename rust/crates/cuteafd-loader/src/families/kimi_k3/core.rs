//! Header-native Kimi core trellis inventory. Source bytes and resident bytes
//! differ: a TP shard replicates the opposite-side H128 sign/scale vector.
use super::config::KimiK3Config;
use crate::{formats::exl3_storage::exl3_module, plan::{checkpoint::Checkpoint, format}};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// How the current TP2 engine owns each eligible projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreSplit { Output, Input, Whole }

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

#[derive(Debug, Clone)]
pub struct CoreProjection {
    pub stem: String,
    pub bits: usize,
    pub codebook: crate::formats::exl3_storage::Exl3Codebook,
    /// Logical checkpoint shape [N,K], before any explicit output padding.
    pub shape: [usize; 2],
    pub stored_outputs: usize,
    pub split: CoreSplit,
    pub source_bytes: u64,
}

impl CoreProjection {
    /// Payload and vectors only; the four-byte codebook marker stays on host.
    pub fn resident_bytes(&self, ranks: usize) -> Result<u64> {
        ensure!(matches!(ranks, 1 | 2), "Kimi core supports one or two coordinator ranks");
        let [_, k] = self.shape;
        let n = self.stored_outputs;
        let payload = k as u64 * n as u64 * self.bits as u64 / 8;
        Ok(match self.split {
            CoreSplit::Whole => payload + 2 * (k + n) as u64,
            CoreSplit::Output => {
                ensure!(n % ranks == 0, "{} output cannot split TP{ranks}", self.stem);
                payload / ranks as u64 + 2 * (k + n / ranks) as u64
            }
            CoreSplit::Input => {
                ensure!(k % ranks == 0, "{} input cannot split TP{ranks}", self.stem);
                payload / ranks as u64 + 2 * (k / ranks + n) as u64
            }
        })
    }

    /// Model residency's split total and replicated amount. A Whole projection
    /// is later assigned to the existing MLA or latent owner, never duplicated.
    pub fn residency(&self) -> (u64, u64) {
        let [_, k] = self.shape;
        let n = self.stored_outputs;
        let payload = k as u64 * n as u64 * self.bits as u64 / 8;
        match self.split {
            CoreSplit::Output => (payload + 2 * n as u64, 2 * k as u64),
            CoreSplit::Input => (payload + 2 * k as u64, 2 * n as u64),
            CoreSplit::Whole => (0, payload + 2 * (k + n) as u64),
        }
    }
}

/// Physical owners and bounded route-exclusive views for serialized projections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionStorageLayout {
    pub owners: BTreeMap<&'static str, usize>,
    pub route_offsets: BTreeMap<&'static str, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionScratchLayout {
    pub extents: BTreeMap<&'static str, usize>,
    pub rows: usize,
}
impl ProjectionScratchLayout {
    pub fn new(k: usize, n: usize, rows: usize, chunk: usize, weights: usize) -> Result<Self> {
        ensure!((1..=4096).contains(&rows) && k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0,
            "Kimi K6 scratch geometry");
        ensure!(chunk > 0 && chunk % 128 == 0 && n % chunk == 0
            && k.checked_mul(chunk).and_then(|v| v.checked_mul(2)) == Some(weights)
            && weights <= 256 * 1024 * 1024, "Kimi K6 chunk policy geometry");
        let product = |values: &[usize]| values.iter().try_fold(1usize, |a, b| a.checked_mul(*b))
            .context("Kimi K6 scratch overflow");
        let mut extents = BTreeMap::from([
            ("rotated", product(&[rows, k, 2])?), ("table", 8), ("scales", k / 128 * 2),
            ("partial", product(&[rows.min(63), k.div_ceil(1024), n, 4])?),
            ("accumulator", product(&[rows, n, 4])?),
        ]);
        if rows >= 64 {
            extents.insert("chunk weights", weights);
            extents.insert("chunk output", product(&[rows, chunk, 4])?);
        }
        let layout = Self { extents, rows };
        layout.bytes()?;
        Ok(layout)
    }
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        ensure!(self.rows == other.rows, "Kimi K6 scratch row capacities differ");
        for (&name, &bytes) in &other.extents {
            self.extents.entry(name).and_modify(|v| *v = (*v).max(bytes)).or_insert(bytes);
        }
        self.bytes()?;
        Ok(())
    }
    pub fn storage(&self) -> Result<ProjectionStorageLayout> {
        let mut owners = self.extents.clone();
        let partial = owners.remove("partial").context("Kimi K6 native partial absent")?;
        let weights = owners.remove("chunk weights");
        let output = owners.remove("chunk output");
        let mut route_offsets = BTreeMap::from([("partial", 0)]);
        let wide = match (weights, output) {
            (Some(weights), Some(output)) => {
                let offset = weights.checked_add(255).map(|n| n / 256 * 256)
                    .context("Kimi K6 chunk alignment overflow")?;
                route_offsets.insert("chunk weights", 0);
                route_offsets.insert("chunk output", offset);
                offset.checked_add(output).context("Kimi K6 chunk arena overflow")?
            }
            (None, None) => 0,
            _ => anyhow::bail!("Kimi K6 incomplete chunk scratch"),
        };
        // Native M<64 and chunk M>=64 never execute in the same projection.
        owners.insert("route arena", partial.max(wide));
        Ok(ProjectionStorageLayout { owners, route_offsets })
    }
    pub fn bytes(&self) -> Result<usize> {
        self.storage()?.owners.values().try_fold(0usize, |a, b| a.checked_add(*b))
            .context("Kimi K6 scratch sum overflow")
    }
}

/// Named split-stage outputs retained beside the fused lane's existing arena.
pub fn intermediate_extents(rows: usize) -> Result<BTreeMap<&'static str, usize>> {
    [("K6 q",6144,2),("K6 k",6144,2),("K6 v",6144,2),("K6 gate",12288,2),
        ("K6 q_a",1536,2),("K6 kv_a",640,2),("K6 q_b",18432,2),("K6 normalized",12288,2),
        ("K6 FFN gate",3072,2),("K6 FFN up",3072,2),("K6 attention partial",7168,4),
        ("K6 FFN partial",7168,4),("K6 reduced",7168,4)].into_iter()
        .map(|(name,width,element)| Ok((name,rows.checked_mul(width*element).context("K6 lane intermediate overflow")?)))
        .collect()
}

/// Split entrypoints can need separate scratch larger than the fused plane.
pub fn lane_extents(rows: usize, base_kda: usize, base_mla: usize,
    split_kda: usize, split_mla: usize) -> Result<BTreeMap<&'static str, usize>> {
    let mut extents=intermediate_extents(rows)?;
    if split_kda>base_kda {extents.insert("K6 KDA scratch",split_kda);}
    if split_mla>base_mla {extents.insert("K6 MLA scratch",split_mla);}
    Ok(extents)
}

/// Inert installed-package inspection, with no CUDA load or payload read.
pub fn package_lane_extents(manifest: &serde_json::Value, rows: usize) -> Result<BTreeMap<&'static str,usize>> {
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
        base_kda=base_kda.max(scratch(&format!("kimik3_kda_m{capacity}"))?);
        split_kda=split_kda.max(scratch(&format!("kda_core_h48_m{capacity}"))?);
        for heads in [48,96] {
            let suffix=if heads==48 {""}else{"_h96"};
            base_mla=base_mla.max(scratch(&format!("kimik3_mla_producer{suffix}_m{capacity}"))?);
            for stride in [576,640] {
                split_mla=split_mla.max(scratch(&format!("mla_rank_norm_h{heads}_kv{stride}_m{capacity}"))?);
            }
        }
    }
    lane_extents(rows,base_kda,base_mla,split_kda,split_mla)
}

pub fn needs_f32_partials(cfg: &KimiK3Config, core: &CoreInventory) -> bool {
    (0..cfg.layers).any(|layer| ["self_attn.o_proj", "block_sparse_moe.shared_experts.down_proj"]
        .iter().any(|leaf| core.get_weight(&format!("language_model.model.layers.{layer}.{leaf}")).is_some()))
}

pub fn exchange_recv_bytes(rows: usize, f32: bool) -> Result<usize> {
    ensure!((1..=4096).contains(&rows), "Kimi exchange row capacity");
    Ok((rows * 7168 * if f32 {4} else {2}).max(256))
}

#[derive(Debug,Clone,PartialEq,Eq)]
pub struct CoreCharges {
    pub projection: Option<ProjectionScratchLayout>,
    pub stages: BTreeMap<&'static str,usize>,
    pub receive: usize,
}
impl CoreCharges {
    pub fn from_package(cfg: &KimiK3Config, core: &CoreInventory, rows: usize,
        manifest: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            projection: projection_scratch(cfg,core,rows,&manifest["families"]["kimi_k6_dense"] )?,
            stages: if core.projections.is_empty() {BTreeMap::new()}else{package_lane_extents(manifest,rows)?},
            receive: exchange_recv_bytes(rows,needs_f32_partials(cfg,core))?,
        })
    }
    pub fn demands(&self) -> Result<Vec<crate::placement::Demand>> {
        use cuteafd_core::memory_layout::{Basis,Category};
        let projection=self.projection.as_ref().map_or(Ok(0),|layout|layout.bytes())?;
        let stages=self.stages.values().try_fold(0usize,|sum,bytes|sum.checked_add(*bytes))
            .context("K6 stage scratch overflow")?;
        Ok((0..2).flat_map(|gpu|[
            crate::placement::Demand::new(gpu,Category::Workspace,"Kimi K6 projection scratch",projection as u64,Basis::Exact),
            crate::placement::Demand::new(gpu,Category::Workspace,"Kimi K6 split-stage operands",stages as u64,Basis::Exact),
            crate::placement::Demand::new(gpu,Category::Workspace,"Kimi K6 peer receive",self.receive as u64,Basis::Exact),
        ]).collect())
    }
}

/// The package owns chunk geometry. Headers select projections; no second export table.
pub fn projection_scratch(cfg: &KimiK3Config, core: &CoreInventory, rows: usize,
    metadata: &serde_json::Value) -> Result<Option<ProjectionScratchLayout>> {
    if core.projections.is_empty() { return Ok(None); }
    core.validate_runtime_groups(cfg)?;
    ensure!(metadata["abi_version"] == 1, "Kimi K6 row-policy ABI");
    let policies = metadata["row_policies"].as_array().context("Kimi K6 row-policy list")?;
    let mut combined: Option<ProjectionScratchLayout> = None;
    for projection in core.projections.values() {
        let head = projection.stem == "language_model.lm_head";
        let mla = projection.stem.contains(".self_attn.") && projection.stem
            .strip_prefix("language_model.model.layers.").and_then(|s| s.split_once('.'))
            .and_then(|(layer,_)| layer.parse::<usize>().ok()).is_some_and(|layer|
                cfg.attention[layer] == super::config::KimiAttention::Mla);
        let worlds: &[usize] = if projection.split == CoreSplit::Whole { &[1] }
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
            ensure!(policy["threshold_rows"] == 64 && policy["below"] == "shared_native"
                && policy["at_or_above"] == "chunk_dequant_bf16" && policy["chunk_output_dtype"] == "f32"
                && policy["stitch_before_output_transform"] == true, "Kimi K6 row-policy route");
            let integer = |key: &str| policy[key].as_u64().and_then(|v| usize::try_from(v).ok())
                .with_context(|| format!("Kimi K6 row-policy {key}"));
            let chunk = integer("chunk_columns")?;
            ensure!(integer("chunk_output_scratch_bytes_per_row")? == chunk * 4, "Kimi K6 chunk output extent");
            let mut layout = ProjectionScratchLayout::new(k,n,if head {rows.min(64)} else {rows},
                chunk,integer("weight_scratch_bytes")?)?;
            layout.rows = rows;
            if let Some(combined) = &mut combined { combined.merge(&layout)?; } else { combined = Some(layout); }
        }
    }
    Ok(combined)
}

#[derive(Debug, Default, Clone)]
pub struct CoreInventory {
    pub projections: BTreeMap<String, CoreProjection>,
}

impl CoreInventory {
    /// Expected logical shapes come from the family geometry, not a side file.
    /// Both native MCG and MUL1 keep their own unpacked H128 vectors.
    pub fn from_checkpoint(cfg: &KimiK3Config, checkpoint: &Checkpoint,
        expected: &BTreeMap<String, Vec<usize>>) -> Result<Self> {
        let mut inventory = Self::default();
        let is_core = |name: &str| !name.contains(".experts.")
            && (name.starts_with("language_model.model.") || name.starts_with("language_model.lm_head."));
        let mut selected = checkpoint.tensors.iter().filter(|t| is_core(&t.meta.name))
            .cloned().collect::<Vec<_>>();
        // Role selection starts from an index-only inventory. Read each missing
        // core shard's bounded header once; never read weight payloads or use a side map.
        let loaded = selected.iter().map(|t| format::split_stem(&t.meta.name).0)
            .collect::<std::collections::BTreeSet<_>>();
        let missing = checkpoint.weight_map.keys().filter(|name| is_core(name))
            .filter_map(|name| name.strip_suffix(".trellis"))
            .filter(|stem| !loaded.contains(stem)).collect::<std::collections::BTreeSet<_>>();
        let mut shards = BTreeMap::<&str, std::collections::BTreeSet<&str>>::new();
        for (name, shard) in &checkpoint.weight_map {
            if missing.contains(format::split_stem(name).0) {
                shards.entry(shard).or_default().insert(name);
            }
        }
        for (shard, mut names) in shards {
            for meta in crate::read_safetensors_metadata(&checkpoint.snapshot.join(shard))? {
                if names.remove(meta.name.as_str()) {
                    selected.push(crate::plan::checkpoint::CheckpointTensor { shard: shard.into(), meta });
                }
            }
            ensure!(names.is_empty(), "Kimi core shard {shard} lacks indexed tensors: {}",
                names.into_iter().collect::<Vec<_>>().join(", "));
        }
        for (stem, members) in format::group_by_stem(&selected) {
            if !members.contains_key("trellis") { continue; }
            let split = split(&stem).with_context(|| format!("{stem}: Kimi BF16/native keep-list forbids core quantization"))?;
            let shape = expected.get(&format!("{stem}.weight"))
                .with_context(|| format!("{stem}: outside the Kimi core tensor geometry"))?;
            ensure!(shape.len() == 2, "{stem}: core projection must be a matrix");
            let suffixes = members.keys().map(String::as_str).collect::<Vec<_>>();
            let module = exl3_module(&stem, &suffixes, |suffix| members.get(suffix).map(|t| &t.meta))?;
            ensure!(module.bits == 6, "{stem}: Kimi core requires K6, found K{}", module.bits);
            let padded = stem.ends_with(".self_attn.kv_a_proj_with_mqa") && shape[0] == cfg.kv_lora_rank + cfg.qk_rope_dim
                && module.output_features == shape[0].next_multiple_of(128);
            ensure!(module.input_features == shape[1] && (module.output_features == shape[0] || padded),
                "{stem}: core header shape [{},{}], expected {shape:?} (only MLA kv_a permits explicit N128 padding)",
                module.output_features, module.input_features);
            ensure!(!checkpoint.contains_tensor(&format!("{stem}.weight")),
                "{stem}: checkpoint contains both BF16 and K6 storage");
            inventory.projections.insert(stem.clone(), CoreProjection { stem, bits: module.bits, codebook: module.codebook,
                shape: [shape[0], shape[1]], stored_outputs: module.output_features, split,
                source_bytes: members.values().map(|t| t.meta.byte_length).sum() });
        }
        Ok(inventory)
    }

    /// The current projected engine admits complete operator groups only. The
    /// quantizer's KDA fallback keeps all five projections BF16, preserving the
    /// original fused path; partial groups require independently exported BF16
    /// partner projections and are deliberately not admitted.
    pub fn validate_runtime_groups(&self,cfg:&KimiK3Config)->Result<()> {
        for layer in 0..cfg.layers {
            let root=format!("language_model.model.layers.{layer}");
            let attention:&[&str]=match cfg.attention[layer] {
                super::config::KimiAttention::Kda=>&["q_proj","k_proj","v_proj","g_proj","o_proj"],
                super::config::KimiAttention::Mla=>&["q_a_proj","kv_a_proj_with_mqa","q_b_proj","g_proj","o_proj"],
            };
            self.complete_group("attention",attention.iter().map(|leaf|format!("{root}.self_attn.{leaf}")))?;
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
            assert!(error.contains("only whole-group fallback") && error.contains(&stem)
                && error.contains("BF16:"),"{error}");
        }
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
    fn route_arena_drops_wide_native_partials_and_keeps_chunk_views_disjoint() -> Result<()> {
        let mut layout=ProjectionScratchLayout::new(7168,12288,4096,12288,7168*12288*2)?;
        layout.merge(&ProjectionScratchLayout::new(12288,7168,4096,7168,12288*7168*2)?)?;
        layout.merge(&ProjectionScratchLayout::new(1536,18432,4096,18432,1536*18432*2)?)?;
        assert_eq!(layout.extents["partial"],21_676_032);
        assert_eq!(layout.bytes()?,880_804_040);
        assert_eq!(2_290_090_184usize-layout.bytes()?,1_409_286_144);
        let mut head=ProjectionScratchLayout::new(7168,163840,64,16384,7168*16384*2)?;
        head.rows=4096;
        layout.merge(&head)?;
        assert_eq!(layout.extents["partial"],289_013_760);
        assert_eq!(layout.bytes()?,939_524_296);
        let storage=layout.storage()?;
        assert_eq!(storage.route_offsets["partial"],0);
        assert_eq!(storage.route_offsets["chunk weights"],0);
        let output=storage.route_offsets["chunk output"];
        assert_eq!(output%256,0);
        assert!(output>=layout.extents["chunk weights"]);
        assert!(output+layout.extents["chunk output"]<=storage.owners["route arena"]);
        assert!(!storage.owners.contains_key("partial"));
        assert!(!storage.owners.contains_key("chunk weights"));
        assert!(!storage.owners.contains_key("chunk output"));
        assert!(ProjectionScratchLayout {rows:64,extents:BTreeMap::from([
            ("partial",16),("chunk weights",usize::MAX),("chunk output",16)])}.bytes().is_err());
        Ok(())
    }
    #[test]
    fn row_policy_scratch_uses_native_ceiling_and_adds_chunk_only_at_64() -> Result<()> {
        for rows in [1,8,16,63,64,4096] {
            let layout=ProjectionScratchLayout::new(7168,6144,rows,6144,7168*6144*2)?;
            assert_eq!(layout.extents["partial"],rows.min(63)*7*6144*4);
            assert_eq!(layout.extents.contains_key("chunk weights"),rows>=64);
            assert_eq!(layout.extents.contains_key("chunk output"),rows>=64);
            assert_eq!(exchange_recv_bytes(rows,true)?,rows*7168*4);
            assert_eq!(exchange_recv_bytes(rows,false)?,rows*7168*2);
        }
        assert!(ProjectionScratchLayout::new(7168,6144,0,6144,7168*6144*2).is_err());
        assert!(ProjectionScratchLayout::new(7168,6144,64,128,1).is_err());
        Ok(())
    }
}
