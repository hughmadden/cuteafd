//! Header-only final target operands, before any allocation or payload read.
use super::{v41_backbone_cache::CachePlacement, v41_tensors::Fp8Head};
use anyhow::{ensure, Result};
use cuteafd_loader::{OfficialV41Catalog, serving_capacity::WeightArenaLayout};
use std::collections::BTreeMap;
use crate::shared::memory::{DeviceAllocation, device::Device,
    weight_arena::{DeviceWeightArena, weight_allocation}};

pub(crate) fn allocation<'a>(library: &'a cuteafd_ffi::NativeLibrary,
    arena: Option<&DeviceWeightArena<'a>>, name: &str, bytes: usize) -> Result<DeviceAllocation<'a>> {
    Ok(DeviceAllocation::from_allocation(weight_allocation(
        Device { library, id: library.cuda_get_device()? }, arena, name, bytes)?))
}

pub(crate) struct TargetArenas<'a> { arenas: Vec<(i32, DeviceWeightArena<'a>)> }
impl<'a> TargetArenas<'a> {
    pub fn new(library: &'a cuteafd_ffi::NativeLibrary, layout: TargetLayout) -> Result<Self> {
        let original = library.cuda_get_device()?;
        let count = layout.ranks.len();
        let arenas = layout.ranks.into_iter().enumerate().map(|(rank, layout)| {
            let id = if count == 1 { original } else { rank as i32 };
            Ok((id, DeviceWeightArena::new(Device { library, id }, layout)?))
        }).collect::<Result<_>>()?;
        Ok(Self { arenas })
    }
    pub fn get(&self, id: i32) -> Result<&DeviceWeightArena<'a>> {
        self.arenas.iter().find(|(device, _)| *device == id).map(|(_, arena)| arena)
            .ok_or_else(|| anyhow::anyhow!("V4.1 target arena device {id} missing"))
    }
}


pub(crate) struct TargetLayout {
    pub ranks: Vec<WeightArenaLayout>,
}

impl TargetLayout {
    pub fn new(catalog: &OfficialV41Catalog, placement: Option<CachePlacement>,
        gpu_embedding: bool, head: Fp8Head, split_query: bool, split_output: bool) -> Result<Self> {
        use super::{v41_attention_query::AttentionQueryWeights, v41_attention_output::AttentionOutputWeights,
            v41_backbone_hc::BackboneHcWeights, v41_backbone_router::BackboneRouterWeights,
            v41_backbone_shared::BackboneSharedWeights, v41_window::WindowWeights,
            v41_compressor::CompressorWeights, v41_index_query::IndexQueryWeights};
        ensure!(placement.is_some() || !(split_query || split_output), "projection TP2 requires two target GPUs");
        let count = if placement.is_some() { 2 } else { 1 };
        let mut requests = vec![BTreeMap::new(); count];
        for layer in 0..40 {
            let rank = placement.map(|map| map.attention(layer)).transpose()?.unwrap_or(0);
            let mut names = [BackboneHcWeights::names(layer, true)?.to_vec(),
                BackboneHcWeights::names(layer, false)?.to_vec(),
                BackboneRouterWeights::names(layer)?, WindowWeights::names(layer)?.to_vec()].concat();
            let query = AttentionQueryWeights::names(layer)?;
            names.extend(query.into_iter().enumerate().filter(|(i, _)| !split_query || !matches!(i, 2 | 3)).map(|(_, name)| name));
            let output = AttentionOutputWeights::names(layer)?;
            names.extend(output.into_iter().take(if split_output { 2 } else { 4 }));
            names.push(format!("layers.{layer}.attn.attn_sink"));
            if [2, 8, 14, 20].contains(&layer) { names.extend(CompressorWeights::names(layer)?); }
            if [2, 8, 14, 20, 24, 28, 32, 36].contains(&layer) { names.extend(IndexQueryWeights::names(layer)?); }
            if placement.is_none() { names.extend(BackboneSharedWeights::names(layer)?); }
            if [1, 14].contains(&layer) {
                names.extend(["wkv.weight", "wkv.scale", "q_weight", "k_weight"]
                    .map(|suffix| format!("layers.{layer}.engram.{suffix}")));
            }
            for name in names {
                let bytes = usize::try_from(catalog.device_tensor_bytes(&name, None)?)?;
                insert(&mut requests[rank], name.clone(), bytes)?;
                if name.ends_with(".weight") && catalog.tensor(&name)?.metadata.dtype == cuteafd_core::DType::F8E4M3 {
                    // The exporter uses stored (group-local for wo_a) values / 32.
                    insert(&mut requests[rank], packed_scale_name(&name), bytes / 32)?;
                }
            }
            if placement.is_some() {
                for rank in 0..2 {
                    for (stem, bytes) in [("w1", 5120 * 1152), ("w3", 5120 * 1152), ("w2", 1152 * 5120)] {
                        let name = format!("layers.{layer}.ffn.shared_experts.{stem}.weight");
                        insert(&mut requests[rank], name.clone(), bytes)?;
                        insert(&mut requests[rank], packed_scale_name(&name), bytes / 32)?;
                    }
                    for (split, stem, bytes) in [(split_query, "wq_b", 1280 * 16384), (split_output, "wo_b", 8192 * 2560)] {
                        if split {
                            let name = format!("layers.{layer}.attn.{stem}.weight");
                            insert(&mut requests[rank], name.clone(), bytes)?;
                            insert(&mut requests[rank], packed_scale_name(&name), bytes / 32)?;
                        }
                    }
                }
            }
        }
        if gpu_embedding {
            insert(&mut requests[0], "embed.weight".into(), usize::try_from(catalog.device_tensor_bytes("embed.weight", None)?)?)?;
        }
        insert(&mut requests[count - 1], "norm.weight".into(), usize::try_from(catalog.device_tensor_bytes("norm.weight", None)?)?)?;
        let rows = 129280 / count;
        for rank in &mut requests {
            for (name, bytes) in head_requests(rows, head)? { insert(rank, name, bytes)?; }
        }
        Ok(Self { ranks: requests.into_iter().enumerate().map(|(rank, requests)|
            WeightArenaLayout::new(format!("v41/target/gpu{rank}"), requests)).collect::<Result<_>>()? })
    }
}

fn insert(requests: &mut BTreeMap<String, usize>, name: String, bytes: usize) -> Result<()> {
    ensure!(bytes > 0 && requests.insert(name.clone(), bytes).is_none(), "invalid or duplicate V4.1 weight {name}");
    Ok(())
}

pub(crate) fn packed_scale_name(weight: &str) -> String { format!("{weight}.packed_scales") }

pub(crate) fn head_requests(rows: usize, mode: Fp8Head) -> Result<Vec<(String, usize)>> {
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
    #[test]
    fn installed_target_arenas_inventory() -> Result<()> {
        let Some(snapshot) = super::super::coordinator_golden_tests::snapshot() else { return Ok(()) };
        let catalog = cuteafd_loader::read_official_v41_catalog(&snapshot)?;
        for (placement, gpu_embedding) in [(None, false), (Some(CachePlacement::encoder_decoder()), true)] {
            for mode in [Fp8Head::Off, Fp8Head::Draft, Fp8Head::All, Fp8Head::Packed] {
                let layout = TargetLayout::new(&catalog, placement, gpu_embedding, mode, false, false)?;
                for (rank, arena) in layout.ranks.iter().enumerate() {
                    let large: Vec<_> = arena.views().values().map(|view| view.bytes()).filter(|&bytes| bytes > 1 << 20).collect();
                    let slack: usize = large.iter().map(|&bytes| bytes.div_ceil(2 << 20) * (2 << 20) - bytes).sum();
                    let tail = arena.bytes().div_ceil(2 << 20) * (2 << 20) - arena.bytes();
                    eprintln!("V41_ARENA ranks={} gpu={rank} embedding={gpu_embedding} head={mode:?} allocations={} payload={} large_count={} large_bytes={} old_slack={slack} arena={} padding={} tail={tail}",
                        layout.ranks.len(), arena.views().len(), arena.payload_bytes(), large.len(), large.iter().sum::<usize>(), arena.bytes(), arena.padding_bytes());
                    let mut views = arena.views().values().collect::<Vec<_>>();
                    views.sort_by_key(|view| view.offset());
                    assert!(views.iter().all(|view| view.offset() % 256 == 0));
                    assert!(views.windows(2).all(|views| views[0].offset() + views[0].bytes() <= views[1].offset()));
                    assert!(arena.views().keys().all(|name| !name.contains(".embed.") && !name.contains(".ffn.experts.") && !name.starts_with("mtp.") && !name.starts_with("vision.")));
                    assert!(!matches!(mode, Fp8Head::All | Fp8Head::Packed) || arena.view("head.bf16").is_none());
                }
            }
        }
        for (query, output) in [(true, false), (false, true), (true, true)] {
            let layout = TargetLayout::new(&catalog, Some(CachePlacement::encoder_decoder()), true, Fp8Head::All, query, output)?;
            for rank in &layout.ranks {
                for (enabled, stem, bytes) in [(query, "wq_b", 1280 * 16384), (output, "wo_b", 8192 * 2560)] {
                    if enabled { assert_eq!(rank.view(&format!("layers.0.attn.{stem}.weight")).unwrap().bytes(), bytes); }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn vocabulary_retains_only_selected_destinations() -> Result<()> {
        for rows in [64640, 129280] {
            for mode in [Fp8Head::Off, Fp8Head::Draft, Fp8Head::All, Fp8Head::Packed] {
                let requests = head_requests(rows, mode)?;
                assert_eq!(requests.iter().map(|(_, bytes)| bytes).sum::<usize>(), mode.resident((rows * 10240) as u64).0 as usize);
                assert_eq!(requests.iter().any(|(name, _)| name == "head.bf16"), matches!(mode, Fp8Head::Off | Fp8Head::Draft));
            }
        }
        Ok(())
    }
}
