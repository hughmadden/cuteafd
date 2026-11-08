//! Final V4.1 target operands, planned before allocation or payload reads.
use super::resident::Fp8Head;
use crate::{OfficialV41Catalog, SafetensorsTensorMetadata, serving_capacity::WeightArenaLayout};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

pub struct TargetLayout { pub ranks: Vec<WeightArenaLayout> }
impl TargetLayout {
    pub fn new(catalog: &OfficialV41Catalog, placement: Option<[usize; 40]>,
        gpu_embedding: bool, head: Fp8Head, split_query: bool, split_output: bool) -> Result<Self> {
        Self::from_headers(catalog.tensors().iter().map(|tensor| &tensor.metadata),
            placement, gpu_embedding, head, split_query, split_output)
    }

    pub fn from_headers<'a>(tensors: impl IntoIterator<Item = &'a SafetensorsTensorMetadata>,
        placement: Option<[usize; 40]>, gpu_embedding: bool, head: Fp8Head,
        split_query: bool, split_output: bool) -> Result<Self> {
        let tensors: BTreeMap<_, _> = tensors.into_iter().map(|tensor| (tensor.name.as_str(), tensor)).collect();
        ensure!(placement.is_some() || !(split_query || split_output), "projection TP2 requires two target GPUs");
        ensure!(placement.as_ref().is_none_or(|map| map.iter().all(|&rank| rank < 2)), "invalid target weight placement");
        let count = if placement.is_some() { 2 } else { 1 };
        let head_tensor = tensors.get("head.weight").context("missing vocabulary checkpoint operand")?;
        ensure!(head_tensor.dtype == cuteafd_core::DType::Bf16 && head_tensor.shape == [129280, 5120]
            && head_tensor.byte_length == 129280 * 10240, "unexpected vocabulary checkpoint geometry");
        let mut requests = vec![BTreeMap::new(); count];
        for layer in 0..40 {
            let rank = placement.map(|map| map[layer]).unwrap_or(0);
            let mut names = Vec::new();
            for kind in ["attn", "ffn"] {
                names.extend([format!("layers.{layer}.hc_{kind}_fn"),
                    format!("layers.{layer}.hc_{kind}_scale"), format!("layers.{layer}.hc_{kind}_base"),
                    format!("layers.{layer}.{kind}_norm.weight")]);
            }
            names.extend(["weight", "bias", "bias_vl"].map(|suffix| format!("layers.{layer}.ffn.gate.{suffix}")));
            names.extend(["wkv.weight", "wkv.scale", "kv_norm.weight", "wq_a.weight", "wq_a.scale",
                "q_norm.weight", "wo_a.weight", "wo_a.scale", "attn_sink"]
                .map(|suffix| format!("layers.{layer}.attn.{suffix}")));
            for (split, stem) in [(split_query, "wq_b"), (split_output, "wo_b")] {
                if !split { names.extend(["weight", "scale"].map(|suffix| format!("layers.{layer}.attn.{stem}.{suffix}"))); }
            }
            if [2, 8, 14, 20].contains(&layer) {
                names.extend(["wkv", "norm"].map(|stem| format!("layers.{layer}.attn.compressor.{stem}.weight")));
                if layer != 20 { names.push(format!("layers.{layer}.attn.compressor.wgate.weight")); }
                names.extend(["wk.weight", "k_norm.weight"].map(|suffix| format!("layers.{layer}.attn.indexer.{suffix}")));
            }
            if [2, 8, 14, 20, 24, 28, 32, 36].contains(&layer) {
                names.extend(["wq_b.weight", "wq_b.scale", "weights_proj.weight"]
                    .map(|suffix| format!("layers.{layer}.attn.indexer.{suffix}")));
            }
            if placement.is_none() {
                for stem in ["w1", "w3", "w2"] {
                    names.extend(["weight", "scale"].map(|suffix| format!("layers.{layer}.ffn.shared_experts.{stem}.{suffix}")));
                }
            }
            if [1, 14].contains(&layer) {
                names.extend(["wkv.weight", "wkv.scale", "q_weight", "k_weight"]
                    .map(|suffix| format!("layers.{layer}.engram.{suffix}")));
            }
            for name in names {
                let tensor = tensors.get(name.as_str()).with_context(|| format!("missing target operand {name}"))?;
                let bytes = usize::try_from(tensor.byte_length)?;
                insert(&mut requests[rank], name.clone(), bytes)?;
                if name.ends_with(".weight") && tensor.dtype == cuteafd_core::DType::F8E4M3 {
                    // Grouped wo_a uses its stored group-local extent, not expanded geometry.
                    insert(&mut requests[rank], packed_scale_name(&name), bytes / 32)?;
                }
            }
            if placement.is_some() {
                for requests in &mut requests {
                    for stem in ["w1", "w3", "w2"] {
                        let name = format!("layers.{layer}.ffn.shared_experts.{stem}.weight");
                        let shape = if stem == "w2" { [5120, 2304] } else { [2304, 5120] };
                        let bytes = fp8_half_bytes(&tensors, &name, shape)?;
                        insert(requests, name.clone(), bytes)?;
                        insert(requests, packed_scale_name(&name), bytes / 32)?;
                    }
                    for (split, stem, shape) in [(split_query, "wq_b", [32768, 1280]), (split_output, "wo_b", [5120, 8192])] {
                        if split {
                            let name = format!("layers.{layer}.attn.{stem}.weight");
                            let bytes = fp8_half_bytes(&tensors, &name, shape)?;
                            insert(requests, name.clone(), bytes)?;
                            insert(requests, packed_scale_name(&name), bytes / 32)?;
                        }
                    }
                }
            }
        }
        if gpu_embedding {
            insert(&mut requests[0], "embed.weight".into(), usize::try_from(tensors.get("embed.weight").context("missing embedding")?.byte_length)?)?;
        }
        insert(&mut requests[count - 1], "norm.weight".into(), usize::try_from(tensors.get("norm.weight").context("missing norm")?.byte_length)?)?;
        for rank in &mut requests {
            for (name, bytes) in head_requests(129280 / count, head)? { insert(rank, name, bytes)?; }
        }
        Ok(Self { ranks: requests.into_iter().enumerate().map(|(rank, requests)|
            WeightArenaLayout::new(format!("v41/target/gpu{rank}"), requests)).collect::<Result<_>>()? })
    }

    /// Extra conversion storage beside the entire preallocated final backing.
    pub fn vocabulary_transient_bytes(head: Fp8Head, ranks: usize) -> Result<usize> {
        ensure!(matches!(ranks, 1 | 2), "invalid V4.1 target rank count");
        Ok(match head {
            Fp8Head::Packed => (129280 / ranks) * 10240,
            Fp8Head::All => 16 * 10240,
            Fp8Head::Off | Fp8Head::Draft => 0,
        })
    }
}

/// Independently retired dense dSpark weights; routed experts have their own arena.
pub fn draft_layout(catalog: &OfficialV41Catalog) -> Result<WeightArenaLayout> {
    draft_layout_from_headers(catalog.tensors().iter().map(|tensor| &tensor.metadata))
}

/// The one dSpark dense recipe the planner and the serving loader both charge:
/// checkpoint headers only, so planning needs no catalog or payload read.
pub fn draft_layout_from_headers<'a>(tensors: impl IntoIterator<Item = &'a SafetensorsTensorMetadata>)
    -> Result<WeightArenaLayout> {
    let mut requests = BTreeMap::new();
    for meta in tensors {
        let name = &meta.name;
        if !name.starts_with("mtp.") || name.contains(".ffn.experts.") { continue; }
        let bytes = usize::try_from(meta.byte_length)?;
        insert(&mut requests, name.clone(), bytes)?;
        if name.ends_with(".weight") && meta.dtype == cuteafd_core::DType::F8E4M3
            && (!name.ends_with(".main_proj.weight") || name == "mtp.0.main_proj.weight") {
            insert(&mut requests, packed_scale_name(name), bytes / 32)?;
        }
    }
    WeightArenaLayout::new("v41/dspark/dense", requests)
}

fn fp8_half_bytes(tensors: &BTreeMap<&str, &SafetensorsTensorMetadata>, name: &str,
    shape: [usize; 2]) -> Result<usize> {
    let tensor = tensors.get(name).with_context(|| format!("missing TP2 weight {name}"))?;
    ensure!(tensor.dtype == cuteafd_core::DType::F8E4M3 && tensor.shape == shape
        && tensor.byte_length == (shape[0] * shape[1]) as u64, "unexpected TP2 weight geometry: {name}");
    let scale = name.strip_suffix(".weight").context("TP2 matrix name lacks weight suffix")?;
    let scale_name = format!("{scale}.scale");
    let scale = tensors.get(scale_name.as_str()).with_context(|| format!("missing TP2 scales {scale_name}"))?;
    ensure!(scale.shape == [shape[0] / 32, shape[1] / 32]
        && scale.dtype == cuteafd_core::DType::F8E8M0
        && scale.byte_length == (shape[0] * shape[1] / 1024) as u64,
        "unexpected TP2 scale geometry: {scale_name}");
    Ok(usize::try_from(tensor.byte_length)? / 2)
}

fn insert(requests: &mut BTreeMap<String, usize>, name: String, bytes: usize) -> Result<()> {
    ensure!(bytes > 0 && requests.insert(name.clone(), bytes).is_none(), "invalid or duplicate V4.1 weight {name}");
    Ok(())
}
pub fn packed_scale_name(weight: &str) -> String { format!("{weight}.packed_scales") }
pub fn head_requests(rows: usize, mode: Fp8Head) -> Result<Vec<(String, usize)>> {
    let mut requests = Vec::new();
    if matches!(mode, Fp8Head::Off | Fp8Head::Draft) { requests.push(("head.bf16".into(), rows * 10240)); }
    if matches!(mode, Fp8Head::All | Fp8Head::Draft) {
        requests.extend([("head.fp8.payload".into(), rows * 5120), ("head.fp8.scales".into(), rows * 160)]);
    }
    if mode == Fp8Head::Packed {
        requests.push(("head.packed12".into(), cuteafd_core::packed_head::PackedHeadLayout::new(rows, 5120)?.total_bytes()));
    }
    Ok(requests)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(name: &str, dtype: cuteafd_core::DType, shape: &[usize], bytes: u64) -> SafetensorsTensorMetadata {
        SafetensorsTensorMetadata { name: name.into(), dtype, shape: shape.to_vec(), byte_offset: 0, byte_length: bytes }
    }

    #[test]
    fn tp2_extents_come_from_validated_native_headers() -> Result<()> {
        let weight = tensor("layers.0.attn.wq_b.weight", cuteafd_core::DType::F8E4M3, &[32768, 1280], 32768 * 1280);
        let scale = tensor("layers.0.attn.wq_b.scale", cuteafd_core::DType::F8E8M0, &[1024, 40], 1024 * 40);
        let headers = BTreeMap::from([(weight.name.as_str(), &weight), (scale.name.as_str(), &scale)]);
        assert_eq!(fp8_half_bytes(&headers, &weight.name, [32768, 1280])?, 16384 * 1280);
        assert!(fp8_half_bytes(&headers, &weight.name, [1280, 32768]).is_err());
        let mut invalid = scale.clone();
        invalid.byte_length *= 4;
        let headers = BTreeMap::from([(weight.name.as_str(), &weight), (invalid.name.as_str(), &invalid)]);
        assert!(fp8_half_bytes(&headers, &weight.name, [32768, 1280]).is_err());
        Ok(())
    }

    #[test]
    fn target_rejects_invalid_placement_and_missing_vocabulary() {
        let empty = Vec::<SafetensorsTensorMetadata>::new();
        assert!(TargetLayout::from_headers(&empty, None, false, Fp8Head::All, true, false).is_err());
        assert!(TargetLayout::from_headers(&empty, Some([2; 40]), false, Fp8Head::All, false, false).is_err());
        assert!(TargetLayout::from_headers(&empty, None, false, Fp8Head::All, false, false).is_err());
    }

    #[test]
    fn head_transient_matches_final_backing_chronology() -> Result<()> {
        for ranks in [1, 2] {
            assert_eq!(TargetLayout::vocabulary_transient_bytes(Fp8Head::Packed, ranks)?, 129280 / ranks * 10240);
            assert_eq!(TargetLayout::vocabulary_transient_bytes(Fp8Head::All, ranks)?, 16 * 10240);
            for head in [Fp8Head::Off, Fp8Head::Draft] {
                assert_eq!(TargetLayout::vocabulary_transient_bytes(head, ranks)?, 0);
            }
        }
        assert!(TargetLayout::vocabulary_transient_bytes(Fp8Head::All, 3).is_err());
        Ok(())
    }
}
