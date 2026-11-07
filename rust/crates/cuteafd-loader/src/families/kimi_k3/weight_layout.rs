//! Packed device weight extents shared by checkpoint admission and runtime owners.
use crate::formats::trellis_core::CoreWeightLayout;
use super::{config::{KimiAttention, KimiK3Config}, core::CoreInventory, kda};
use crate::{placement::{Demand, LayerMode, ResidentWeights}, formats::dense_trellis::TrellisSide};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightAttention { Absent, Whole, Split }
impl WeightAttention {
    fn bind(mode: LayerMode, ranks: usize) -> Result<Vec<Self>> {
        ensure!(matches!(ranks, 1 | 2), "Kimi weight layout requires one or two ranks");
        match mode {
            LayerMode::HeadSplit | LayerMode::ContextSplit => Ok(vec![Self::Split; ranks]),
            LayerMode::Whole { gpu, .. } if usize::from(gpu) < ranks => {
                let mut shares = vec![Self::Absent; ranks];
                shares[usize::from(gpu)] = Self::Whole;
                Ok(shares)
            }
            _ => anyhow::bail!("Kimi attention weight owner outside TP2"),
        }
    }
    pub fn present(self) -> bool { self != Self::Absent }
}
pub fn routed_owner(cfg: &KimiK3Config, layer: usize) -> Result<usize> {
    routed_owner_for(cfg, layer, 2)
}
pub fn routed_owner_for(cfg: &KimiK3Config, layer: usize, ranks: usize) -> Result<usize> {
    ensure!(matches!(ranks, 1 | 2), "Kimi routed owner rank count");
    ensure!(cfg.dense.get(layer) == Some(&false), "Kimi front end requires a routed layer");
    Ok(cfg.dense[..layer].iter().filter(|&&dense| !dense).count() % ranks)
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalLayout {
    pub rank: usize, pub hidden: usize, pub vocab: usize,
    pub operands: BTreeMap<&'static str, usize>, pub core_head: Option<CoreWeightLayout>,
}
impl GlobalLayout {
    pub fn ranks_with_core(cfg: &KimiK3Config, inventory: &CoreInventory, ranks: usize) -> Result<Vec<Self>> {
        ensure!(matches!(ranks, 1 | 2), "Kimi global rank count");
        let [mut entry, terminal] = Self::pair_with_core(cfg, inventory)?;
        if ranks == 2 { return Ok(vec![entry, terminal]); }
        entry.operands.extend(terminal.operands);
        entry.core_head = terminal.core_head;
        Ok(vec![entry])
    }

    pub fn pair(cfg: &KimiK3Config) -> Result<[Self; 2]> {
        ensure!(!cfg.tie_embeddings, "Kimi global weights require the checkpoint's untied head");
        let matrix = extent(&[cfg.vocab_size, cfg.hidden, 2])?;
        Ok([Self { rank: 0, hidden: cfg.hidden, vocab: cfg.vocab_size,
                core_head: None, operands: BTreeMap::from([("embedding", matrix)]) },
            Self { rank: 1, hidden: cfg.hidden, vocab: cfg.vocab_size,
                core_head: None, operands: BTreeMap::from([("head", matrix), ("output.score", extent(&[cfg.hidden, 4])?),
                    ("output.norm", extent(&[cfg.hidden, 2])?)]) }])
    }

    pub fn pair_with_core(cfg: &KimiK3Config, inventory: &CoreInventory) -> Result<[Self; 2]> {
        let mut pair=Self::pair(cfg)?;
        if let Some(head)=inventory.get_weight("language_model.lm_head") {
            ensure!(head.shape==[cfg.vocab_size,cfg.hidden],"Kimi K6 head inventory geometry");
            pair[1].operands.remove("head");
            pair[1].core_head=Some(CoreWeightLayout::new(head,TrellisSide::Output,0,1)?);
        }
        Ok(pair)
    }

    pub fn bytes(&self) -> Result<usize> {
        self.operands.values().copied().chain(self.core_head.iter().map(|head|head.bytes))
            .try_fold(0usize, |sum, bytes| sum.checked_add(bytes)
            .context("Kimi global weight extent overflow"))
    }

    pub fn allocation_extents(&self) -> Vec<u64> {
        self.operands.values().map(|&bytes| bytes as u64)
            .chain(self.core_head.iter().flat_map(CoreWeightLayout::allocation_extents)).collect()
    }

    pub fn demand(&self) -> Result<Demand> {
        let bytes = self.bytes()? as u64;
        Ok(Demand::new(self.rank as u8, Category::Weights, "Kimi global weights", bytes, Basis::Exact)
            .with_resident_weights(ResidentWeights(bytes)))
    }

}
#[derive(Debug, PartialEq, Eq)]
pub struct LayerLayout {
    pub operands: BTreeMap<&'static str, usize>,
    pub attention: WeightAttention,
    pub core: BTreeMap<&'static str, CoreWeightLayout>,
    pub native: BTreeMap<&'static str, crate::formats::native_attention::NativeLayout>,
}
impl LayerLayout {
    pub fn pair(cfg: &KimiK3Config, layer: usize, mode: LayerMode) -> Result<[Self; 2]> {
        Ok(Self::ranks(cfg, layer, mode, 2)?.try_into().expect("two rank layouts"))
    }

    pub fn ranks(cfg: &KimiK3Config, layer: usize, mode: LayerMode, ranks: usize) -> Result<Vec<Self>> {
        let kind = *cfg.attention.get(layer).context("Kimi attention layer out of range")?;
        ensure!(cfg.layers == cfg.attention.len() && cfg.dense.len() == cfg.layers,
            "Kimi dense/attention inventories differ");
        ensure!(ranks == 1 || !matches!(mode, LayerMode::Whole { ffn: crate::placement::FfnMode::Owner, .. }),
            "Kimi FFNs require TP2 intermediate slices");
        ensure!(ranks != 1 || matches!(mode, LayerMode::Whole { gpu: 0, ffn: crate::placement::FfnMode::Owner }),
            "Kimi single-rank layout requires whole attention and FFN ownership");
        ensure!(ranks == 1 || kind != KimiAttention::Kda || mode == LayerMode::HeadSplit,
            "Kimi KDA requires TP2 head-split placement");
        let shares = WeightAttention::bind(mode, ranks)?;
        let intermediate = if cfg.dense[layer] { cfg.dense_intermediate }
            else { extent(&[cfg.moe_intermediate, cfg.shared_experts])? };
        ensure!(intermediate > 0 && intermediate % ranks == 0, "Kimi FFN intermediate does not divide TP2");
        let local = intermediate / ranks;
        let sum = |a: usize, b: usize| a.checked_add(b).context("Kimi weight extent overflow");
        let mut pair = shares.into_iter().map(|attention| Self { operands: BTreeMap::new(), attention, core: BTreeMap::new(), native: BTreeMap::new() }).collect::<Vec<_>>();
        for (index, rank) in pair.iter_mut().enumerate() {
            for name in ["attention.score", "ffn.score"] { rank.insert(name, &[cfg.hidden, 4])?; }
            for name in ["attention.norm", "ffn.norm"] { rank.insert(name, &[cfg.hidden, 2])?; }
            rank.insert("ffn.gate_up", &[2, local, cfg.hidden, 2])?;
            rank.insert("ffn.down", &[cfg.hidden, local, 2])?;
            if !cfg.dense[layer] && index == routed_owner_for(cfg, layer, ranks)? {
                for name in ["latent.down", "latent.up"] { rank.insert(name, &[cfg.latent, cfg.hidden, 2])?; }
                rank.insert("latent.norm", &[cfg.latent, 2])?;
                rank.insert("router.weight", &[cfg.experts, cfg.hidden, 2])?;
                rank.insert("router.bias", &[cfg.experts, 4])?;
            }
            match kind {
                KimiAttention::Kda => {
                    kda::geometry(cfg).validate(ranks)?;
                    let heads = cfg.kda_heads / ranks;
                    let width = extent(&[heads, cfg.kda_head_dim])?;
                    let input = sum(sum(extent(&[4, width])?, cfg.kda_head_dim)?, heads)?;
                    rank.insert("kda.input", &[input, cfg.hidden, 2])?;
                    rank.insert("kda.forget", &[width, cfg.kda_head_dim, 2])?;
                    rank.insert("kda.conv", &[3, width, cfg.kda_conv, 4])?;
                    rank.insert("kda.a_log", &[heads, 4])?;
                    rank.insert("kda.dt_bias", &[width, 4])?;
                    rank.insert("kda.norm", &[cfg.kda_head_dim, 4])?;
                    rank.insert("kda.output", &[cfg.hidden, width, 2])?;
                }
                KimiAttention::Mla if rank.attention.present() => {
                    ensure!(cfg.heads > 0 && cfg.heads % 2 == 0, "Kimi MLA heads do not divide TP2");
                    let heads = if rank.attention == WeightAttention::Whole { cfg.heads } else { cfg.heads / 2 };
                    let input = sum(cfg.q_lora_rank, sum(cfg.kv_lora_rank, cfg.qk_rope_dim)?)?;
                    rank.insert("mla.qkv_a", &[input, cfg.hidden, 2])?;
                    rank.insert("mla.q_norm", &[cfg.q_lora_rank, 2])?;
                    rank.insert("mla.kv_norm", &[cfg.kv_lora_rank, 2])?;
                    rank.insert("mla.q_b", &[heads, sum(cfg.qk_nope_dim, cfg.qk_rope_dim)?, cfg.q_lora_rank, 2])?;
                    rank.insert("mla.uk", &[heads, cfg.kv_lora_rank, cfg.qk_nope_dim, 2])?;
                    rank.insert("mla.uv", &[heads, cfg.v_head_dim, cfg.kv_lora_rank, 2])?;
                    rank.insert("mla.gate", &[heads, cfg.v_head_dim, cfg.hidden, 2])?;
                    rank.insert("mla.output", &[cfg.hidden, heads, cfg.v_head_dim, 2])?;
                }
                KimiAttention::Mla => {}
            }
        }
        Ok(pair)
    }

    /// The BF16-only layout remains exactly the original fused layout. A mixed
    /// group stores individual BF16 matrices beside its native K6 projections;
    /// it never allocates a dequantized shadow of a compressed source tensor.
    pub fn pair_with_core(cfg: &KimiK3Config, layer: usize, mode: LayerMode,
        inventory: &CoreInventory) -> Result<[Self; 2]> {
        Ok(Self::ranks_with_core(cfg, layer, mode, inventory, 2)?.try_into().expect("two rank layouts"))
    }

    pub fn ranks_with_core(cfg: &KimiK3Config, layer: usize, mode: LayerMode,
        inventory: &CoreInventory, ranks: usize) -> Result<Vec<Self>> {
        let mut pair = Self::ranks(cfg, layer, mode, ranks)?;
        let root = format!("language_model.model.layers.{layer}");
        let shared = format!("{root}.block_sparse_moe.shared_experts");
        let attn = format!("{root}.self_attn");
        let native = |stem: &str| inventory.get_weight(stem).is_some();
        let ffn_split = !cfg.dense[layer] && ["gate_proj", "up_proj", "down_proj"].iter()
            .any(|leaf| native(&format!("{shared}.{leaf}")));
        let attention_split = ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "q_a_proj",
            "q_b_proj", "kv_a_proj_with_mqa"].iter().any(|leaf| native(&format!("{attn}.{leaf}")));
        let attention_ranks = pair.iter().filter(|layout| layout.attention.present()).count();
        for (rank, layout) in pair.iter_mut().enumerate() {
            if ffn_split {
                layout.operands.remove("ffn.gate_up");
                layout.operands.remove("ffn.down");
                let width = cfg.moe_intermediate * cfg.shared_experts;
                for (name, leaf, shape, side) in [
                    ("ffn.gate", "gate_proj", [width, cfg.hidden], TrellisSide::Output),
                    ("ffn.up", "up_proj", [width, cfg.hidden], TrellisSide::Output),
                    ("ffn.down", "down_proj", [cfg.hidden, width], TrellisSide::Input),
                ] { layout.projection(name, &format!("{shared}.{leaf}"), shape, side, rank, ranks, inventory)?; }
            }
            if !cfg.dense[layer] && rank==routed_owner_for(cfg,layer,ranks)? {
                let root=format!("{root}.block_sparse_moe");
                for (name,leaf,shape) in [("latent.down","routed_expert_down_proj",[cfg.latent,cfg.hidden]),
                    ("latent.up","routed_expert_up_proj",[cfg.hidden,cfg.latent])] {
                    if native(&format!("{root}.{leaf}")) {
                        layout.operands.remove(name);
                        layout.projection(name,&format!("{root}.{leaf}"),shape,TrellisSide::Output,0,1,inventory)?;
                    }
                }
            }
            if !attention_split || !layout.attention.present() { continue; }
            let shard = if attention_ranks == 1 { 0 } else { rank };
            match cfg.attention[layer] {
                KimiAttention::Kda => {
                    layout.operands.remove("kda.input");
                    layout.operands.remove("kda.output");
                    layout.insert("kda.fab", &[cfg.kda_head_dim + cfg.kda_heads / ranks, cfg.hidden, 2])?;
                    let width = cfg.kda_heads * cfg.kda_head_dim;
                    for (name, leaf) in [("kda.q", "q_proj"), ("kda.k", "k_proj"),
                        ("kda.v", "v_proj"), ("kda.gate", "g_proj")] {
                        layout.projection(name, &format!("{attn}.{leaf}"), [width, cfg.hidden],
                            TrellisSide::Output, shard, ranks, inventory)?;
                    }
                    layout.projection("kda.output", &format!("{attn}.o_proj"), [cfg.hidden, width],
                        TrellisSide::Input, shard, ranks, inventory)?;
                }
                KimiAttention::Mla => {
                    for name in ["mla.qkv_a", "mla.q_b", "mla.gate", "mla.output"] { layout.operands.remove(name); }
                    for (name, leaf, shape, side, count, index) in [
                        ("mla.q_a", "q_a_proj", [cfg.q_lora_rank, cfg.hidden], TrellisSide::Output, 1, 0),
                        ("mla.kv_a", "kv_a_proj_with_mqa", [cfg.kv_lora_rank + cfg.qk_rope_dim, cfg.hidden], TrellisSide::Output, 1, 0),
                        ("mla.q_b", "q_b_proj", [cfg.heads * (cfg.qk_nope_dim + cfg.qk_rope_dim), cfg.q_lora_rank], TrellisSide::Output, attention_ranks, shard),
                        ("mla.gate", "g_proj", [cfg.heads * cfg.v_head_dim, cfg.hidden], TrellisSide::Output, attention_ranks, shard),
                        ("mla.output", "o_proj", [cfg.hidden, cfg.heads * cfg.v_head_dim], TrellisSide::Input, attention_ranks, shard),
                    ] { layout.projection(name, &format!("{attn}.{leaf}"), shape, side, index, count, inventory)?; }
                }
            }
        }
        Ok(pair)
    }

    pub fn core_layouts(&self) -> impl Iterator<Item=&CoreWeightLayout> { self.core.values() }

    fn projection(&mut self, name: &'static str, stem: &str, shape: [usize; 2], side: TrellisSide,
        rank: usize, ranks: usize, inventory: &CoreInventory) -> Result<()> {
        if let Some(projection) = inventory.get_weight(stem) {
            ensure!(projection.shape == shape, "Kimi {stem} core inventory differs from layer geometry");
            let storage = if ranks == 1 { TrellisSide::Output } else { side };
            self.core.insert(name, CoreWeightLayout::new(projection, storage, rank, ranks)?);
        } else {
            let [mut n, mut k] = shape;
            match side { TrellisSide::Output => n /= ranks, TrellisSide::Input => k /= ranks }
            if let Some(projection) = inventory.native.get(stem) {
                ensure!(projection.shape == shape, "Kimi {stem} native inventory differs from layer geometry");
                self.native.insert(name, crate::formats::native_attention::NativeLayout {
                    stem: stem.into(), shape: [n, k], fp8: projection.fp8,
                    rank, ranks, split: projection.split });
            } else {
                self.insert(name, &[n, k, 2])?;
            }
        }
        Ok(())
    }

    fn insert(&mut self, name: &'static str, dimensions: &[usize]) -> Result<()> {
        ensure!(self.operands.insert(name, extent(dimensions)?).is_none(), "duplicate Kimi operand {name}");
        Ok(())
    }

    pub fn bytes(&self) -> Result<usize> {
        let base = self.operands.values().copied().chain(self.core.values().map(|layout| layout.bytes))
            .try_fold(0usize, |n, size| n.checked_add(size).context("Kimi layer admission extent overflow"))?;
        self.native.values().try_fold(base, |sum, layout| sum.checked_add(usize::try_from(layout.bytes()?)?)
            .context("Kimi native attention extent overflow"))
    }

    /// None selects every physical owner; Some partitions the same requests
    /// into attention and non-attention logical planner groups, without aliases.
    pub fn allocation_extents(&self, attention: Option<bool>) -> Result<Vec<u64>> {
        let selected = |name: &&str| attention.is_none_or(|wanted|
            wanted == (name.starts_with("mla.") || name.starts_with("kda.")));
        let mut requests = self.operands.iter().filter(|(name, _)| selected(name))
            .map(|(_, &bytes)| bytes as u64)
            .chain(self.core.iter().filter(|(name, _)| selected(name))
                .flat_map(|(_, layout)| layout.allocation_extents())).collect::<Vec<_>>();
        for (name, layout) in &self.native {
            if selected(name) { requests.extend(layout.allocation_extents()?); }
        }
        Ok(requests)
    }

    pub fn attention_bytes(&self) -> Result<usize> {
        let base = self.operands.iter().filter(|(name, _)| name.starts_with("mla.") || name.starts_with("kda."))
            .map(|(_, bytes)| *bytes).chain(self.core.iter().filter(|(name, _)|
                name.starts_with("mla.") || name.starts_with("kda.")).map(|(_, layout)| layout.bytes))
            .try_fold(0usize, |n, bytes| n.checked_add(bytes).context("Kimi attention extent overflow"))?;
        self.native.values().try_fold(base, |sum, layout| sum.checked_add(usize::try_from(layout.bytes()?)?)
            .context("Kimi native attention extent overflow"))
    }

}
fn extent(dimensions: &[usize]) -> Result<usize> {
    ensure!(dimensions.iter().all(|n| *n > 0), "Kimi weight extent is empty");
    dimensions.iter().try_fold(1usize, |n, dim| n.checked_mul(*dim)).context("Kimi weight extent overflow")
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::FfnMode;

    #[test]
    fn physical_weight_requests_partition_exact_payloads_for_bf16_k6_and_native_attention() -> Result<()> {
        let dir = tempfile::tempdir()?;
        crate::plan::testing::write_kimi_core_snapshot(dir.path(), false);
        let checkpoint = crate::plan::checkpoint::Checkpoint::open(dir.path())?;
        let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
        let core = crate::plan::families::kimi::core_inventory(&checkpoint)?;
        for inventory in [&CoreInventory::default(), &core] {
            for ranks in [1, 2] {
                for global in GlobalLayout::ranks_with_core(&cfg, inventory, ranks)? {
                    assert_eq!(global.allocation_extents().iter().sum::<u64>(), global.bytes()? as u64);
                }
                let modes = if ranks == 1 { vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] }
                    else { vec![LayerMode::HeadSplit, LayerMode::ContextSplit,
                        LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }] };
                for mode in modes {
                    for layer in 0..cfg.layers {
                        for mut layout in LayerLayout::ranks_with_core(&cfg, layer, mode, inventory, ranks)? {
                            // Both native storage variants use the same physical partition.
                            for fp8 in [false, true] {
                                layout.native.insert("mla.test", crate::formats::native_attention::NativeLayout {
                                    stem: "test".into(), shape: [256, 512], fp8, rank: 0, ranks: 1,
                                    split: crate::formats::trellis_core::CoreSplit::Whole,
                                });
                                let mut all = layout.allocation_extents(None)?;
                                let mut attention = layout.allocation_extents(Some(true))?;
                                assert_eq!(all.iter().sum::<u64>(), layout.bytes()? as u64);
                                assert_eq!(attention.iter().sum::<u64>(), layout.attention_bytes()? as u64);
                                attention.extend(layout.allocation_extents(Some(false))?);
                                all.sort_unstable();
                                attention.sort_unstable();
                                assert_eq!(all, attention);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn one_rank_owns_full_attention_ffns_and_all_routed_frontends() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&super::super::config::tests::release_config())?;
        let mode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
        for layer in 0..cfg.layers {
            let mut ranks = LayerLayout::ranks(&cfg, layer, mode, 1)?;
            assert_eq!(ranks.len(), 1);
            let layout = ranks.pop().unwrap();
            assert_eq!(layout.attention, WeightAttention::Whole);
            let intermediate = if cfg.dense[layer] { cfg.dense_intermediate }
                else { cfg.moe_intermediate * cfg.shared_experts };
            assert_eq!(layout.operands["ffn.down"], cfg.hidden * intermediate * 2);
            if cfg.attention[layer] == KimiAttention::Kda {
                assert_eq!(layout.operands["kda.output"], cfg.hidden * cfg.kda_heads * cfg.kda_head_dim * 2);
                assert_eq!(layout.operands["kda.a_log"], cfg.kda_heads * 4);
            } else {
                assert_eq!(layout.operands["mla.uk"], cfg.heads * cfg.kv_lora_rank * cfg.qk_nope_dim * 2);
            }
            assert_eq!(layout.operands.contains_key("router.weight"), !cfg.dense[layer]);
            if !cfg.dense[layer] { assert_eq!(routed_owner_for(&cfg, layer, 1)?, 0); }
        }
        let globals = GlobalLayout::ranks_with_core(&cfg, &CoreInventory::default(), 1)?;
        assert_eq!(globals.len(), 1);
        assert_eq!(globals[0].rank, 0);
        assert!(globals[0].operands.contains_key("embedding") && globals[0].operands.contains_key("head"));
        assert!(LayerLayout::ranks(&cfg, 0, LayerMode::HeadSplit, 1).is_err());
        assert!(LayerLayout::ranks(&cfg, 0, mode, 0).is_err());
        Ok(())
    }

    #[test]
    fn one_rank_preserves_native_k6_without_bf16_shadows() -> Result<()> {
        let dir = tempfile::tempdir()?;
        crate::plan::testing::write_kimi_core_snapshot(dir.path(), false);
        let checkpoint = crate::plan::checkpoint::Checkpoint::open(dir.path())?;
        let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
        let core = crate::plan::families::kimi::core_inventory(&checkpoint)?;
        for layer in [1, 3] {
            let ranks = LayerLayout::ranks_with_core(&cfg, layer,
                LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }, &core, 1)?;
            let layout = &ranks[0];
            assert!(!layout.core.is_empty());
            assert!(layout.core.values().all(|p| p.rank == 0 && p.ranks == 1 && p.side == TrellisSide::Output));
            assert!(layout.core.keys().all(|name| !layout.operands.contains_key(name)));
            assert!(!layout.operands.contains_key("ffn.gate_up"));
            assert_eq!(layout.core["ffn.down"].shape, [cfg.hidden, cfg.moe_intermediate * cfg.shared_experts]);
            if layer == 1 {
                assert_eq!(layout.core["kda.output"].shape, [cfg.hidden, cfg.kda_heads * cfg.kda_head_dim]);
                assert_eq!(layout.operands["kda.fab"], (cfg.kda_head_dim + cfg.kda_heads) * cfg.hidden * 2);
            }
        }
        let globals = GlobalLayout::ranks_with_core(&cfg, &core, 1)?;
        assert!(globals[0].core_head.is_some() && !globals[0].operands.contains_key("head"));
        Ok(())
    }
}
