//! Final, header-derived requests for the independently retired target weight set.
//! The daemon binds exactly these views and admission charges exactly these layouts.
use super::{core::CoreInventory, resident::GlmfHeadFormat, GlmNextAttention, GlmNextConfig};
use crate::{placement::LayerMode, plan::Checkpoint, serving_capacity::WeightArenaLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use std::collections::BTreeMap;

/// Checkpoint tensor-name prefix of the text model.
pub const PREFIX: &str = "model.language_model.";

pub fn joined(names: &[String]) -> String { names.join("+") }

/// `kda_fp8` selects FP8 KDA input/output projections; the FP8 scale layout
/// (row128 or channel) does not change any extent.
#[allow(clippy::too_many_arguments)]
pub fn target(checkpoint: &Checkpoint, cfg: &GlmNextConfig, modes: &[LayerMode],
    ranks: usize, kda_fp8: bool, output_shard: bool, head: GlmfHeadFormat,
    embedding: Option<(usize, usize)>) -> Result<Vec<WeightArenaLayout>> {
    ensure!(matches!(ranks, 1 | 2) && modes.len() <= cfg.layers, "Flash arena rank/layer extent");
    let core = CoreInventory::from_checkpoint(cfg, checkpoint)?;
    let tensor = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
        .with_context(|| format!("Flash arena missing {name}"));
    let elements = |name: &str| -> Result<usize> { tensor(name)?.meta.shape.iter().try_fold(1usize,
        |n, &d| n.checked_mul(d).context("Flash arena element overflow")) };
    let mut requests = vec![BTreeMap::new(); ranks];
    let insert = |out: &mut BTreeMap<String, usize>, name: String, bytes: usize| -> Result<()> {
        ensure!(out.insert(name.clone(), bytes.max(256)).is_none(), "duplicate Flash arena request {name}");
        Ok(())
    };
    let one = |out: &mut BTreeMap<String, usize>, name: String| -> Result<()> {
        insert(out, name.clone(), usize::try_from(tensor(&name)?.meta.byte_length)?)
    };
    let f32 = |out: &mut BTreeMap<String, usize>, name: String| -> Result<()> {
        insert(out, name.clone(), elements(&name)?.checked_mul(4).context("Flash F32 extent overflow")?)
    };
    let fp8 = |out: &mut BTreeMap<String, usize>, names: Vec<String>, divisor: usize, row: bool| -> Result<()> {
        let (mut values, mut scales) = (0usize, 0usize);
        for name in &names {
            let shape = &tensor(name)?.meta.shape;
            ensure!(shape.len() == 2, "Flash FP8 matrix shape: {name}");
            values = values.checked_add(shape[0].checked_mul(shape[1]).context("Flash FP8 overflow")?)
                .context("Flash FP8 concatenation overflow")?;
            scales += if row { shape[0] * shape[1].div_ceil(128) * 4 }
                else { shape[0].div_ceil(128) * shape[1].div_ceil(128) * 4 };
        }
        let key = joined(&names);
        insert(out, format!("{key}.payload"), values / divisor)?;
        insert(out, format!("{key}.scales"), scales / divisor)
    };
    if let Some((rank, bytes)) = embedding {
        ensure!(rank < ranks, "Flash embedding arena rank");
        insert(&mut requests[rank], "embedding.table".into(), bytes)?;
    }
    one(&mut requests[0], format!("{PREFIX}norm.weight"))?;
    if let Some(head) = core.get_weight("lm_head") {
        let layout = crate::formats::trellis_core::CoreWeightLayout::new(head,
            crate::formats::dense_trellis::TrellisSide::Output, 0, 1)?;
        for (name, bytes) in layout.arena_tensors(&layout.stem)? {
            ensure!(requests[0].insert(name.clone(), bytes).is_none(), "duplicate Flash K6 view {name}");
        }
    } else {
        match head {
            GlmfHeadFormat::Fp8 => fp8(&mut requests[0], vec!["lm_head.weight".into()], 1, true)?,
            // The exact 12-bit copy (its BF16 source is transient, outside the arena).
            GlmfHeadFormat::Packed => {
                let shape = &tensor("lm_head.weight")?.meta.shape;
                ensure!(shape.len() == 2, "Flash head shape");
                insert(&mut requests[0], "packed12".into(),
                    cuteafd_core::packed_head::PackedHeadLayout::new(shape[0], shape[1])?.total_bytes())?;
            }
            GlmfHeadFormat::Bf16 => one(&mut requests[0], "lm_head.weight".into())?,
        }
    }
    for (layer, &mode) in modes.iter().enumerate() {
        let p = format!("{PREFIX}layers.{layer}");
        let a = |leaf: &str| format!("{p}.self_attn.{leaf}");
        let owner = match mode { LayerMode::Whole { gpu, .. } if ranks == 2 => Some(usize::from(gpu)), _ => None };
        let native_core = core.layer_weights(cfg, layer, ranks, owner.map(|r| r as u8))?;
        let native_attention = core.layer_native(cfg, layer, ranks, owner.map(|r| r as u8))?;
        let native = core.projections.keys().any(|stem| stem.starts_with(&format!("{p}.")));
        ensure!(owner.is_none_or(|owner| owner < ranks), "Flash arena attention owner outside ranks");
        for (rank, out) in requests.iter_mut().enumerate() {
            for site in ["attn", "ffn"] { for suffix in ["fn", "scale", "base"] {
                f32(out, format!("{p}.hc_{site}_{suffix}"))?;
            }}
            for leaf in ["input_layernorm.weight", "post_attention_layernorm.weight"] { one(out, format!("{p}.{leaf}"))?; }
            for layout in native_core[rank].values() { for (name, bytes) in layout.arena_tensors(&layout.stem)? {
                ensure!(out.insert(name.clone(), bytes).is_none(), "duplicate Flash K6 view {name}");
            }}
            for layout in native_attention[rank].values() {
                let [n, k] = layout.shape;
                insert(out, format!("{}.payload", layout.stem), n*k*if layout.fp8 { 1 } else { 2 })?;
                if layout.fp8 { insert(out, format!("{}.scales", layout.stem), n/128*(k/128)*4)?; }
            }
            match cfg.attention[layer] {
                GlmNextAttention::Kda => {
                    if native {
                        insert(out, format!("{p}.w_aux"), (elements(&a("f_a_proj.weight"))?
                            + elements(&a("g_a_proj.weight"))? + elements(&a("b_proj.weight"))?/ranks)*2)?;
                    } else {
                        let inputs = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "f_a_proj.weight", "g_a_proj.weight", "b_proj.weight"]
                            .map(|leaf| a(leaf)).to_vec();
                        let values = inputs.iter().enumerate().map(|(i, name)| elements(name)
                            .map(|bytes| bytes / if matches!(i, 3 | 4) { 1 } else { ranks })).collect::<Result<Vec<_>>>()?
                            .into_iter().sum::<usize>();
                        if kda_fp8 {
                            let key = joined(&inputs);
                            insert(out, format!("{key}.payload"), values)?;
                            insert(out, format!("{key}.scales"), values/128*4)?;
                            fp8(out, vec![a("o_proj.weight")], if output_shard { 1 } else { ranks }, true)?;
                        } else {
                            insert(out, format!("{p}.w_in"), values*2)?;
                            // TP1's unchanged output uses its checkpoint name.
                            insert(out, if ranks == 1 { a("o_proj.weight") } else { format!("{p}.w_o") }, elements(&a("o_proj.weight"))?*2/ranks)?;
                        }
                    }
                    for (key, bytes) in [
                        ("w_fg", (elements(&a("f_b_proj.weight"))?+elements(&a("g_b_proj.weight"))?)*2/ranks),
                        ("conv_w", ["q", "k", "v"].iter().map(|s| elements(&a(&format!("{s}_conv1d.weight"))))
                            .collect::<Result<Vec<_>>>()?.into_iter().sum::<usize>()*4/ranks),
                        ("a_log", cfg.kda_heads*4/ranks),
                        ("dt_bias", elements(&a("dt_bias"))?*4/ranks),
                        ("o_norm", usize::try_from(tensor(&a("o_norm.weight"))?.meta.byte_length)?),
                    ] { insert(out, format!("{p}.{key}"), bytes)?; }
                }
                GlmNextAttention::Mla if owner.is_none_or(|owner| owner == rank) => {
                    let parts = if owner.is_some() { 1 } else { ranks };
                    for leaf in ["q_a_layernorm.weight", "kv_a_layernorm.weight", "indexer.wq_b.weight",
                        "indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.index_kpool_compress_ape"] { one(out, a(leaf))?; }
                    insert(out, format!("{}.uk", a("kv_b_proj.weight")), cfg.heads*cfg.qk_nope_dim*cfg.kv_lora_rank*2/parts)?;
                    insert(out, format!("{}.uv", a("kv_b_proj.weight")), cfg.heads*cfg.v_head_dim*cfg.kv_lora_rank*2/parts)?;
                    if !native {
                        fp8(out, vec![a("q_a_proj.weight"), a("kv_a_proj_with_mqa.weight")], 1, false)?;
                        fp8(out, vec![a("q_b_proj.weight")], parts, false)?;
                        fp8(out, vec![a("o_proj.weight")], parts, false)?;
                    }
                    let names = ["wk.weight", "weights_proj.weight", "index_kpool_compress_gate"].map(|leaf| a(&format!("indexer.{leaf}"))).to_vec();
                    let bytes = names.iter().map(|name| elements(name)).collect::<Result<Vec<_>>>()?.into_iter().sum::<usize>()*2;
                    insert(out, joined(&names), bytes)?;
                }
                GlmNextAttention::Mla => {}
            }
            if !native {
                let mlp = if cfg.dense[layer] { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
                if cfg.dense[layer] && tensor(&format!("{mlp}.gate_proj.weight"))?.meta.dtype == DType::U8 {
                    if rank == 0 { for proj in ["gate_proj", "up_proj", "down_proj"] {
                        one(out, format!("{mlp}.{proj}.weight"))?;
                        insert(out, format!("{mlp}.{proj}.weight_scale"), usize::try_from(tensor(&format!("{mlp}.{proj}.weight_scale"))?.meta.byte_length)?+8)?;
                    }}
                } else {
                    fp8(out, vec![format!("{mlp}.gate_proj.weight"), format!("{mlp}.up_proj.weight")], ranks, false)?;
                    fp8(out, vec![format!("{mlp}.down_proj.weight")], ranks, false)?;
                }
            }
            if !cfg.dense[layer] {
                one(out, format!("{p}.mlp.gate.weight"))?;
                f32(out, format!("{p}.mlp.gate.e_score_correction_bias"))?;
            }
        }
    }
    requests.into_iter().enumerate().map(|(rank, requests)| WeightArenaLayout::new(
        format!("glm5_flash/target/{rank}"), requests)).collect()
}

/// Complete target alternatives for every legal embedding home. The offline
/// planner and serving admission use this same header-derived recipe.
pub fn target_options(checkpoint: &Checkpoint, cfg: &GlmNextConfig, modes: &[LayerMode],
    ranks: usize, representation: super::resident::GlmfRepresentation, embedding_bytes: u64,
    native: Option<&crate::plan::exlr8::native::DenseFacts>) -> Result<Vec<Vec<WeightArenaLayout>>> {
    let homes = if embedding_bytes == 0 { 1 } else { ranks };
    if let Some(facts) = native {
        use crate::plan::exlr8::dense::requests::NativeDenseRecords;
        let owners = modes.iter().enumerate().filter_map(|(layer, mode)| match mode {
            LayerMode::Whole { gpu, .. } if cfg.attention[layer] == GlmNextAttention::Mla && ranks > 1 =>
                Some((layer, usize::from(*gpu))), _ => None,
        }).collect();
        let dense = (0..ranks).map(|rank| NativeDenseRecords::from_records(&facts.records, &facts.selection,
            modes.len(), ranks, rank, &owners)).collect::<Result<Vec<_>>>()?;
        return (0..homes).map(|home| native_target(facts, &dense, cfg, home)).collect();
    }
    (0..homes).map(|home| target(checkpoint, cfg, modes, ranks, representation.kda_fp8,
        representation.output_shard, representation.head,
        (embedding_bytes > 0).then_some((home, usize::try_from(embedding_bytes)?)))).collect()
}

/// Native EXLR8 target arenas, one per rank: every layer's records (KDA conv
/// weights in one plain pack), rank zero's final norm, the row-FP8 embedding on
/// `embedding_rank` and head on rank zero, and the embedding facade on rank
/// zero. Shared experts are a separate format-level arena.
pub fn native_target(facts: &crate::plan::exlr8::native::DenseFacts,
    dense: &[crate::plan::exlr8::dense::requests::NativeDenseRecords], cfg: &GlmNextConfig,
    embedding_rank: usize) -> Result<Vec<WeightArenaLayout>> {
    use crate::plan::exlr8::dense::{requests::{layer_requests, row_requests, vocabulary_records, PlainPack},
        DensePlan, DenseTransform};
    ensure!(!dense.is_empty() && embedding_rank < dense.len(), "native Flash arena rank");
    let mut requests = vec![Vec::new(); dense.len()];
    for (rank, records) in dense.iter().enumerate() {
        for source in &records.layers {
            let p = format!("{PREFIX}layers.{}", source.layer);
            let packs = if cfg.attention[source.layer] == GlmNextAttention::Kda {
                vec![PlainPack { name: "conv_w".into(), components: ["q", "k", "v"].iter()
                    .map(|kind| format!("{p}.self_attn.{kind}_conv1d.weight")).collect() }]
            } else { vec![] };
            requests[rank].extend(layer_requests(source, &packs)?);
        }
    }
    requests[0].push((format!("{PREFIX}norm.weight"), DenseTransform::Identity.allocation_bytes(&dense[0].norm)?));
    let vocabulary = DensePlan::for_rank(&vocabulary_records(&facts.records, &facts.selection,
        cfg.hidden, cfg.vocab_size)?, dense.len(), 0)?;
    for (i, slice) in vocabulary.tensors.iter().enumerate() {
        let (rank, name) = if i == 0 { (embedding_rank, "embedding") } else { (0, "head") };
        requests[rank].extend(row_requests(slice)?.into_iter().map(|(suffix, bytes)| (format!("{name}.{suffix}"), bytes)));
    }
    // The embedding facade has its own final placeholder, not a BF16 vocabulary source.
    requests[0].push(("embedding.table".into(), 256));
    requests.into_iter().enumerate().map(|(rank, mut requests)| {
        requests.sort_by(|left, right| left.0.cmp(&right.0));
        WeightArenaLayout::new(format!("glm5_flash/target/{rank}"), requests)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(layout: &WeightArenaLayout) {
        let mut views = layout.views().values().collect::<Vec<_>>();
        views.sort_by_key(|view| view.offset());
        let mut end = 0;
        for view in views {
            assert_eq!(view.offset() % 256, 0);
            assert!(view.offset() >= end);
            end = view.offset() + view.bytes();
        }
        assert_eq!(end, layout.bytes());
        assert_eq!(layout.payload_bytes() + layout.padding_bytes(), layout.bytes());
        assert!(!layout.views().keys().any(|name| name.contains("staging") || name.contains("temporary")));
    }

    #[test]
    fn final_requests_match_residency_for_tp_and_selected_representations() -> Result<()> {
        use crate::plan::testing::{glm5_flash_config, glm5_flash_tensors, write_snapshot};
        use super::super::resident::{resident_weights, GlmfRepresentation};
        let root = tempfile::tempdir()?;
        let config = glm5_flash_config(6);
        write_snapshot(root.path(), &config, &glm5_flash_tensors(&config), None);
        let checkpoint = Checkpoint::open(root.path())?;
        let cfg = GlmNextConfig::from_hf(&config)?;
        for ranks in [1, 2] { for kda_fp8 in [false, true] {
            for head in [GlmfHeadFormat::Bf16, GlmfHeadFormat::Fp8, GlmfHeadFormat::Packed] { for output_shard in [false, true] {
                if output_shard && (ranks == 1 || !kda_fp8) { continue; }
                if head == GlmfHeadFormat::Packed && ranks == 2 { continue; }
                let repr = GlmfRepresentation { kda_fp8, head, output_shard };
                let old = resident_weights(&checkpoint, &cfg, cfg.layers, ranks, repr).map_err(anyhow::Error::msg)?;
                let layouts = target(&checkpoint, &cfg, &vec![LayerMode::HeadSplit; cfg.layers], ranks,
                    kda_fp8, output_shard, head, Some((0, old[0].embedding as usize)))?;
                for (rank, layout) in layouts.iter().enumerate() {
                    check(layout);
                    assert_eq!(layout.payload_bytes() as u64, old[rank].embedding + old[rank].weights,
                        "rank={rank} ranks={ranks} repr={repr:?}");
                }
            }}
        }}
        Ok(())
    }

    #[test]
    fn whole_mla_and_k6_keep_rank_local_views() -> Result<()> {
        use crate::{plan::testing::{glm5_flash_config, glm5_flash_tensors, write_snapshot, t},
            placement::{AttentionPlacement, FfnMode}};
        use super::super::{core, resident::{resident_weights_placed, GlmfRepresentation}};
        let root = tempfile::tempdir()?;
        let mut config = glm5_flash_config(4);
        config["text_config"]["vocab_size"] = serde_json::json!(256);
        let cfg = GlmNextConfig::from_hf(&config)?;
        let expected = core::expected(&cfg);
        let mut tensors = glm5_flash_tensors(&config);
        tensors.retain(|tensor| !expected.keys().any(|name| {
            let stem = name.strip_suffix(".weight").unwrap();
            tensor.0 == *name || tensor.0 == format!("{name}_scale_inv") || tensor.0.starts_with(&format!("{stem}.trellis"))
        }));
        for (name, shape) in expected {
            let stem = name.strip_suffix(".weight").unwrap();
            let (n, k) = (shape[0], shape[1]);
            tensors.extend([t(format!("{stem}.trellis"), "I16", &[k/16, n/16, 96]),
                t(format!("{stem}.suh"), "F16", &[k]), t(format!("{stem}.svh"), "F16", &[n]),
                t(format!("{stem}.mcg"), "I32", &[1])]);
        }
        write_snapshot(root.path(), &config, &tensors, None);
        let checkpoint = Checkpoint::open(root.path())?;
        for ranks in [1, 2] { for first_owner in 0..ranks {
            let attention = if ranks == 2 { AttentionPlacement::Layers } else { AttentionPlacement::Heads };
            let mut owner = first_owner;
            let modes = cfg.attention.iter().map(|&kind| {
                if ranks == 2 && kind == GlmNextAttention::Mla {
                    let mode = LayerMode::Whole { gpu: owner as u8, ffn: FfnMode::Split };
                    owner ^= 1;
                    mode
                } else { LayerMode::HeadSplit }
            }).collect::<Vec<_>>();
            let old = resident_weights_placed(&checkpoint, &cfg, cfg.layers, ranks,
                GlmfRepresentation::default(), attention, first_owner).map_err(anyhow::Error::msg)?;
            let layouts = target(&checkpoint, &cfg, &modes, ranks, true, false, GlmfHeadFormat::Fp8, None)?;
            for (rank, layout) in layouts.iter().enumerate() {
                check(layout);
                assert_eq!(layout.payload_bytes() as u64, old[rank].weights, "native rank {rank}/{ranks}");
                for (layer, mode) in modes.iter().enumerate() {
                    if matches!(mode, LayerMode::Whole { gpu, .. } if usize::from(*gpu) != rank) {
                        assert!(!layout.views().keys().any(|name| name.starts_with(&format!("{PREFIX}layers.{layer}.self_attn."))));
                    }
                }
            }
        }}
        Ok(())
    }

    #[test]
    fn installed_cpu_inventory() -> Result<()> {
        let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_GLMF") else { return Ok(()) };
        let checkpoint = Checkpoint::coordinator(std::path::Path::new(&snapshot), false, false)?;
        let cfg = GlmNextConfig::read(std::path::Path::new(&snapshot))?;
        for ranks in [1, 2] {
            let embedding = checkpoint.tensors.iter().find(|t| t.meta.name == format!("{PREFIX}embed_tokens.weight"))
                .context("installed embedding header")?.meta.byte_length as usize;
            let layouts = target(&checkpoint, &cfg, &vec![LayerMode::HeadSplit; cfg.layers], ranks,
                true, ranks == 2, GlmfHeadFormat::Fp8, Some((0, embedding)))?;
            for (rank, layout) in layouts.iter().enumerate() {
                check(layout);
                let large = layout.views().values().filter(|view| view.bytes() > 1 << 20).collect::<Vec<_>>();
                let slack: usize = large.iter().map(|v| v.bytes().div_ceil(2 << 20)*(2 << 20)-v.bytes()).sum();
                let tail = layout.bytes().div_ceil(2 << 20)*(2 << 20)-layout.bytes();
                let old = super::super::resident::resident_weights(&checkpoint, &cfg, cfg.layers, ranks,
                    super::super::resident::GlmfRepresentation::default()).map_err(anyhow::Error::msg)?;
                let old = old[rank].weights + old[rank].embedding;
                assert_eq!(old, layout.payload_bytes() as u64);
                eprintln!("FLASH_INVENTORY ranks={ranks} rank={rank} allocations={} payload={} large_count={} large_bytes={} old_slack={slack} arena={} padding={} tail={tail}",
                    layout.views().len(), layout.payload_bytes(), large.len(), large.iter().map(|v| v.bytes()).sum::<usize>(), layout.bytes(), layout.padding_bytes());
                eprintln!("FLASH_CHARGE target ranks={ranks} rank={rank} old={old} B={} tail={tail} new={} delta={} old_rounding=no",
                    layout.bytes(), layout.bytes() + tail, (layout.bytes() + tail) as i128 - i128::from(old));
            }
        }
        Ok(())
    }
}

