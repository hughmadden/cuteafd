//! Native source-weight contract for the MiniMax M3 coordinator: every
//! RTX-resident text operand in its checkpoint format (MXFP8 payload and UE8M0
//! scale codes land unchanged), laid out by the shared dense builder on the
//! coordinator's one GPU. Routed experts, vision and the drafter are separate
//! owners.
use super::Model;
use crate::formats::native_dense::{Hold, NativeDenseLayout, RankWeightLayout, GATE_UP};
use crate::plan::{checkpoint::Checkpoint, spec::Component};
use anyhow::Result;
use cuteafd_core::DType;

const EMBEDDING: &str = "language_model.model.embed_tokens.weight";
/// The fused pairs of the target and the drafter: gate then up rows (the
/// SwiGLU-OAI order) and K then V rows (one KV projection).
const FUSED: [(&str, &str, &str); 2] = [GATE_UP, (".self_attn.k_proj.", ".self_attn.v_proj.", ".self_attn.kv_proj.")];

/// Every RTX-resident text operand, validated against its checkpoint header.
pub fn native_weights(checkpoint: &Checkpoint) -> Result<NativeDenseLayout> {
    let model = Model::open(checkpoint)?;
    let mut declarations = Vec::new();
    for (stem, contract) in &model.contracts {
        if matches!(contract.role.component, Component::RoutedExpert | Component::Vision | Component::Speculator) { continue; }
        match contract.storage {
            super::Storage::Mxfp8 => {
                let [n, k] = contract.shape[..] else { anyhow::bail!("{stem}: MXFP8 operand must be a matrix") };
                declarations.push((format!("{stem}.weight"), vec![n, k], DType::F8E4M3));
                declarations.push((format!("{stem}.weight_scale_inv"), vec![n, k / 32], DType::U8));
            }
            super::Storage::Bf16 | super::Storage::F32 => {
                let dtype = if contract.storage == super::Storage::Bf16 { DType::Bf16 } else { DType::F32 };
                let name = if stem.ends_with("e_score_correction_bias") { stem.clone() } else { format!("{stem}.weight") };
                declarations.push((name, contract.shape.clone(), dtype));
            }
            super::Storage::Nvfp4 => anyhow::bail!("{stem}: NVFP4 outside the routed experts"),
        }
    }
    Ok(NativeDenseLayout::from_headers(checkpoint, declarations)?)
}

/// Every operand of `weights` whole on one GPU, gate/up and K/V fused: the
/// target's layout and the drafter's.
pub fn single_gpu(weights: &NativeDenseLayout) -> Result<RankWeightLayout> {
    RankWeightLayout::from_native(weights, 0, 1, &FUSED, |source| Ok(Some(Hold::whole(source))))
}

/// The coordinator's resident text operands on its one GPU, as the planner and
/// the server admit them. With `host_embedding`, mapped token lookup owns the
/// embedding; the untied output head stays resident.
pub fn resident_weights(checkpoint: &Checkpoint, host_embedding: bool) -> Result<RankWeightLayout> {
    let weights = single_gpu(&native_weights(checkpoint)?)?;
    if !host_embedding { return Ok(weights); }
    checkpoint.require_untied_embedding(EMBEDDING)?;
    weights.retain(|operand| operand.name != EMBEDDING)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::native_dense::WeightSlice;
    use crate::plan::testing;

    #[test]
    fn coordinator_operands_fuse_gate_up_and_kv_on_one_gpu() {
        let dir = tempfile::tempdir().unwrap();
        testing::write_minimax_m3_snapshot(dir.path(), true, true);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let weights = native_weights(&checkpoint).unwrap();
        assert!(weights.tensors.iter().all(|t| !t.meta.name.contains(".experts.") && !t.meta.name.starts_with("vision")));
        let one = resident_weights(&checkpoint, false).unwrap();
        let find = |name: &str| one.operands.iter().find(|o| o.name == name).cloned().unwrap();
        let shared = find("language_model.model.layers.3.block_sparse_moe.shared_experts.gate_up_proj.weight");
        assert_eq!((shared.shape.as_slice(), shared.sources.len()), (&[6144, 6144][..], 2));
        let scales = find("language_model.model.layers.0.mlp.gate_up_proj.weight_scale_inv");
        assert_eq!((scales.dtype, scales.shape.as_slice()), (DType::U8, &[24576, 192][..]));
        for (suffix, dtype, cols) in [("weight", DType::F8E4M3, 6144), ("weight_scale_inv", DType::U8, 192)] {
            let kv = find(&format!("language_model.model.layers.5.self_attn.kv_proj.{suffix}"));
            assert_eq!((kv.dtype, kv.shape.as_slice()), (dtype, &[1024, cols][..]));
            assert!(kv.sources[0].meta.name.contains(".k_proj.") && kv.sources[1].meta.name.contains(".v_proj."));
        }
        assert!(!one.operands.iter().any(|o| o.name.contains(".k_proj.") || o.name.contains(".v_proj.")));
        assert_eq!(find("language_model.model.layers.5.self_attn.index_k_proj.weight").sources.len(), 1);
        // Every operand is whole and charged exactly its sources' bytes.
        assert!(one.operands.iter().all(|o| o.slice == WeightSlice::Replicated && o.allocation_bytes
            == o.sources.iter().map(|s| s.meta.byte_length).sum::<u64>().max(256)));
        assert_eq!(one.resident_bytes, one.operands.iter().map(|o| o.allocation_bytes).sum::<u64>());
        assert_eq!(one.operands.iter().map(|o| o.sources.len()).sum::<usize>(), weights.tensors.len());
        assert_eq!((one.rank, one.ranks), (0, 1));
    }

    #[test]
    fn minimax_host_embedding_removes_only_the_input_table() -> Result<()> {
        let dir = tempfile::tempdir()?;
        testing::write_minimax_m3_snapshot(dir.path(), false, false);
        let mut checkpoint = Checkpoint::open(dir.path())?;
        let gpu = resident_weights(&checkpoint, false)?;
        let host = resident_weights(&checkpoint, true)?;
        let embedding = gpu.operands.iter().find(|o| o.name.ends_with("embed_tokens.weight")).unwrap();
        assert_eq!(gpu.resident_bytes - host.resident_bytes, embedding.allocation_bytes);
        assert_eq!(host.operands.len() + 1, gpu.operands.len());
        assert!(!host.operands.iter().any(|o| o.name.ends_with("embed_tokens.weight")));
        let head = |layout: &RankWeightLayout| layout.operands.iter()
            .find(|o| o.name == "language_model.lm_head.weight").unwrap().allocation_bytes;
        assert_eq!(head(&gpu), head(&host));
        checkpoint.config["tie_word_embeddings"] = serde_json::Value::Bool(true);
        assert!(resident_weights(&checkpoint, true).is_err());
        Ok(())
    }
}
