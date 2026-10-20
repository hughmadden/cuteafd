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
