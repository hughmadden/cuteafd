//! GLM coordinator operands after GlmLoader's packing and rank ownership.
use crate::families::glm5::{GlmDsaConfig, GlmIndexer};
use crate::plan::Checkpoint;
use cuteafd_core::DType;
use std::collections::BTreeMap;

pub fn resident(checkpoint: &Checkpoint, ranks: usize, layers: usize) -> anyhow::Result<Vec<Vec<(String, String, u64)>>> {
    anyhow::ensure!((1..=2).contains(&ranks), "GLM coordinator ranks");
    let cfg = GlmDsaConfig::from_hf(&checkpoint.config)?;
    anyhow::ensure!(layers <= cfg.layers, "GLM resident layer extent exceeds checkpoint");
    let tensors: BTreeMap<_, _> = checkpoint.tensors.iter().map(|t| (t.meta.name.as_str(), &t.meta)).collect();
    let tensor = |name: &str| tensors.get(name).copied().ok_or_else(|| anyhow::anyhow!("missing GLM operand {name}"));
    let mut groups = vec![BTreeMap::<(String, String), u64>::new(); ranks];
    let mut put = |rank: usize, group: &str, format: &str, bytes: u64| {
        *groups[rank].entry((group.into(), format.into())).or_default() += bytes.max(256);
    };
    let native = std::env::var("CUTEAFD_GLM_BF16").is_ok_and(|v| v == "native");
    for layer in 0..layers {
        let p = format!("model.layers.{layer}");
        let attn = format!("{p}.self_attn");
        let qkv = [format!("{attn}.q_a_proj.weight"), format!("{attn}.kv_a_proj_with_mqa.weight")];
        let attention = [qkv[0].clone(), qkv[1].clone(), format!("{attn}.q_b_proj.weight"),
            format!("{attn}.kv_b_proj.weight"), format!("{attn}.o_proj.weight")];
        let bf16 = native && attention.iter().all(|n| tensor(n).is_ok_and(|t| t.dtype == DType::Bf16));
        let qkv_rows: u64 = qkv.iter().map(|n| tensor(n).map(|t| t.shape[0] as u64)).sum::<anyhow::Result<_>>()?;
        let h = cfg.hidden as u64;
        let projection = |name: &str| -> anyhow::Result<(u64, u64)> {
            let t = tensor(name)?;
            anyhow::ensure!(t.shape.len() == 2, "GLM matrix {name}");
            Ok((t.shape[0] as u64 * t.shape[1] as u64, (t.shape[0] as u64).div_ceil(128) * (t.shape[1] as u64).div_ceil(128) * 4))
        };
        for rank in 0..ranks {
            for name in [format!("{p}.input_layernorm.weight"), format!("{p}.post_attention_layernorm.weight"),
                format!("{attn}.q_a_layernorm.weight"), format!("{attn}.kv_a_layernorm.weight")] {
                put(rank, "norm", "bf16", tensor(&name)?.byte_length);
            }
            put(rank, "attention", if bf16 { "bf16" } else { "fp8" }, qkv_rows * h * if bf16 { 2 } else { 1 });
            if !bf16 {
                let grid = qkv.iter().map(|n| projection(n).map(|(_, s)| s)).sum::<anyhow::Result<u64>>()?;
                put(rank, "attention", "fp8-scale", grid);
                put(rank, "attention", "fp8-scale", qkv_rows * h.div_ceil(128) * 4);
            }
            for name in [format!("{attn}.q_b_proj.weight"), format!("{attn}.o_proj.weight")] {
                let (values, scales) = projection(&name)?;
                put(rank, "attention", if bf16 { "bf16" } else { "fp8" }, values / ranks as u64 * if bf16 { 2 } else { 1 });
                if !bf16 { put(rank, "attention", "fp8-scale", scales / ranks as u64); }
            }
            for width in [cfg.qk_nope_head_dim, cfg.v_head_dim] {
                let values = cfg.heads as u64 / ranks as u64 * cfg.kv_lora_rank as u64 * width as u64;
                put(rank, "attention", if bf16 { "bf16" } else { "fp8" }, values * if bf16 { 2 } else { 1 });
                if !bf16 { put(rank, "attention", "fp8-scale", values / 64 * 4); }
            }
            if cfg.indexers[layer] == GlmIndexer::Full {
                let name = format!("{attn}.indexer.wq_b.weight");
                let index_bf16 = native && tensor(&name)?.dtype == DType::Bf16;
                let (values, scales) = projection(&name)?;
                put(rank, "indexer", if index_bf16 { "bf16" } else { "fp8" }, values * if index_bf16 { 2 } else { 1 });
                if !index_bf16 { put(rank, "indexer", "fp8-scale", scales); }
                let ik = ["wk.weight", "weights_proj.weight"].into_iter().map(|suffix| tensor(&format!("{attn}.indexer.{suffix}"))
                    .map(|t| t.shape.iter().product::<usize>() as u64 * 2)).sum::<anyhow::Result<u64>>()?;
                put(rank, "indexer", "bf16", ik);
                for suffix in ["k_norm.weight", "k_norm.bias"] {
                    put(rank, "indexer", "bf16", tensor(&format!("{attn}.indexer.{suffix}"))?.byte_length);
                }
            }
            let dense = layer < cfg.first_moe_layer;
            let mlp = if dense { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
            let names = [format!("{mlp}.gate_proj.weight"), format!("{mlp}.up_proj.weight"), format!("{mlp}.down_proj.weight")];
            let shared_bf16 = !dense && native && names.iter().all(|n| tensor(n).is_ok_and(|t| t.dtype == DType::Bf16));
            let group = if dense { "dense" } else { "shared_expert" };
            let gu = names[..2].iter().map(|n| projection(n)).collect::<anyhow::Result<Vec<_>>>()?;
            let down = projection(&names[2])?;
            for (values, scales) in [(gu[0].0 + gu[1].0, gu[0].1 + gu[1].1), down] {
                put(rank, group, if shared_bf16 { "bf16" } else { "fp8" }, values / ranks as u64 * if shared_bf16 { 2 } else { 1 });
                if !shared_bf16 { put(rank, group, "fp8-scale", scales / ranks as u64); }
            }
            if dense && names.iter().all(|n| tensor(n).is_ok_and(|t| t.dtype == DType::F8E4M3)
                && tensors.contains_key(format!("{n}_scale").as_str())) {
                put(rank, group, "fp8-tensor-scale", 48);
                put(rank, group, "fp8-tensor-scale", 48);
            }
            if !dense && rank == 0 {
                for suffix in ["weight", "e_score_correction_bias"] {
                    put(rank, "router", "bf16", tensor(&format!("{p}.mlp.gate.{suffix}"))?.byte_length);
                }
            }
        }
    }
    for (name, group) in [("model.embed_tokens.weight", "embedding"), ("model.norm.weight", "norm"), ("lm_head.weight", "lm_head")] {
        let t = tensor(name)?;
        let bytes = if t.dtype == DType::F8E4M3 && t.shape.len() == 2 { t.shape.iter().product::<usize>() as u64 * 2 } else { t.byte_length };
        put(0, group, "bf16", bytes);
    }
    Ok(groups.into_iter().map(|g| g.into_iter().map(|((group, format), bytes)| (group, format, bytes)).collect()).collect())
}

/// The planner's packed owners, excluding a host-mapped embedding when selected.
pub fn device_bytes(checkpoint: &Checkpoint, ranks: usize, layers: usize, host_embedding: bool)
    -> anyhow::Result<Vec<u64>> {
    resident(checkpoint, ranks, layers)?.into_iter().map(|groups| groups.into_iter()
        .filter(|(group, _, _)| !host_embedding || group != "embedding")
        .try_fold(0u64, |sum, (_, _, bytes)| sum.checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("GLM retained weight bytes overflow")))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing::{fp8, glm5_config, glm5_tensors, write_snapshot};

    #[test]
    fn resident_headers_follow_rank_ownership_and_absorbed_kv_packing() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &glm5_config(), &glm5_tensors(|n, r, k| fp8(n, r, k, None)), None);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let layers = GlmDsaConfig::from_hf(&checkpoint.config).unwrap().layers;
        let one = resident(&checkpoint, 1, layers).unwrap();
        let two = resident(&checkpoint, 2, layers).unwrap();
        let bytes = |rank: &Vec<(String, String, u64)>, group: &str| rank.iter()
            .filter(|(g, _, _)| g == group).map(|(_, _, b)| *b).sum::<u64>();
        assert_eq!(bytes(&one[0], "router"), bytes(&two[0], "router"));
        assert_eq!(bytes(&two[1], "router"), 0);
        for group in ["embedding", "lm_head"] {
            assert_eq!(bytes(&one[0], group), bytes(&two[0], group));
            assert_eq!(bytes(&two[1], group), 0);
        }
        assert_eq!(bytes(&two[0], "indexer"), bytes(&one[0], "indexer"));
        assert_eq!(bytes(&two[1], "indexer"), bytes(&one[0], "indexer"));
        for group in ["dense", "shared_expert"] {
            assert_eq!(bytes(&two[0], group) + bytes(&two[1], group), bytes(&one[0], group));
        }
        // uk/uv keep per-64K scales; qkv keeps both block and per-row K-major grids.
        let scales = |rank: &Vec<(String, String, u64)>| rank.iter()
            .filter(|(g, f, _)| g == "attention" && f == "fp8-scale").map(|(_, _, b)| *b).sum::<u64>();
        assert!(scales(&two[0]) > 2 * 64 * 512 * (192 + 256) / 2 / 64 * 4);
        assert_eq!(scales(&two[0]), scales(&two[1]));
    }

    #[test]
    fn retained_headers_respect_selected_extent_and_host_embedding() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &glm5_config(), &glm5_tensors(|n, r, k| fp8(n, r, k, None)), None);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let layers = GlmDsaConfig::from_hf(&checkpoint.config).unwrap().layers;
        for ranks in [1, 2] {
            let globals = device_bytes(&checkpoint, ranks, 0, true).unwrap();
            assert!(globals[0] > 0);
            assert!(globals[1..].iter().all(|&bytes| bytes == 0));
            let mut previous = globals;
            for extent in 1..=layers {
                let gpu = device_bytes(&checkpoint, ranks, extent, false).unwrap();
                let host = device_bytes(&checkpoint, ranks, extent, true).unwrap();
                let embedding = checkpoint.tensors.iter().find(|t| t.meta.name == "model.embed_tokens.weight").unwrap();
                assert_eq!(gpu[0] - host[0], embedding.meta.byte_length);
                assert_eq!(gpu[1..], host[1..]);
                assert!(host.iter().zip(&previous).all(|(next, before)| next > before));
                previous = host;
            }
            assert!(resident(&checkpoint, ranks, layers + 1).is_err());
        }
    }
}
