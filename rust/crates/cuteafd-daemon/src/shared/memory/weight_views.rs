//! Format adapters retain the mount loader's region ordering and descriptors.
//! These aliases borrow the ready rank owner; none owns or frees device memory.
use super::weight_arena::WeightView;
use anyhow::{ensure, Result};
use cuteafd_loader::{
    formats::fp8_experts::{Fp8ExpertTensors, Fp8Projection, Slicing},
    plan::exlr8::residency::Residency,
    V41Exl3Residency,
};

pub(crate) struct RegionLayout {
    regions: Vec<(usize, usize)>,
    bytes: usize,
}
impl RegionLayout {
    fn packed(sizes: impl IntoIterator<Item = usize>) -> Result<Self> {
        let mut cursor = 0usize;
        let mut regions = Vec::new();
        for size in sizes {
            cursor = cursor
                .checked_add(255)
                .ok_or_else(|| anyhow::anyhow!("plane alignment overflow"))?
                & !255;
            regions.push((cursor, size));
            cursor = cursor
                .checked_add(size)
                .ok_or_else(|| anyhow::anyhow!("plane extent overflow"))?;
        }
        Ok(Self {
            regions,
            bytes: cursor,
        })
    }
    pub fn exl3(residency: &V41Exl3Residency) -> Result<Self> {
        let (offsets, bytes) = residency.device_arena_layout()?;
        Ok(Self {
            regions: offsets
                .into_iter()
                .zip(residency.buffers.iter().map(|b| b.bytes.max(16)))
                .collect(),
            bytes,
        })
    }
    pub fn fp8(
        tensors: &Fp8ExpertTensors,
        tp: usize,
        rank: usize,
        slicing: Slicing,
    ) -> Result<Self> {
        let mut sizes = Vec::new();
        for projection in Fp8Projection::ALL {
            let (weight, _) = tensors.slice_bytes_with(projection, tp, rank, slicing)?;
            sizes.push(
                weight
                    .checked_mul(tensors.shape().experts)
                    .ok_or_else(|| anyhow::anyhow!("FP8 weights overflow"))?,
            );
            sizes.push(tensors.scale_region_bytes_with(projection, tp, rank, slicing)?);
        }
        Self::packed(sizes)
    }
    pub fn native(hidden: usize, intermediate: usize, experts: usize, nvfp4: bool) -> Result<Self> {
        let planes = cuteafd_loader::placement::families::deepseek_v41::resident_planes(
            hidden as u64,
            intermediate as u64,
            nvfp4,
        )?;
        let mut sizes = planes
            .into_iter()
            .map(|n| {
                usize::try_from(n)?
                    .checked_mul(experts)
                    .ok_or_else(|| anyhow::anyhow!("native weight plane overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        if nvfp4 {
            sizes.extend(
                [experts
                    .checked_mul(4)
                    .ok_or_else(|| anyhow::anyhow!("NVFP4 vectors overflow"))?; 4],
            );
        }
        Self::packed(sizes)
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn regions<'s>(&self, owner: &'s WeightView<'s>) -> Result<Vec<Option<WeightView<'s>>>> {
        ensure!(
            owner.buffer().bytes >= self.bytes,
            "format view exceeds ready layer region"
        );
        self.regions
            .iter()
            .map(|&(offset, bytes)| {
                if bytes == 0 {
                    Ok(None)
                } else {
                    Ok(Some(owner.subregion(offset, bytes)?))
                }
            })
            .collect()
    }
    pub fn addresses(&self, base: u64) -> Result<Vec<u64>> {
        ensure!(
            base != 0 && base.checked_add(self.bytes as u64).is_some(),
            "invalid format view base"
        );
        Ok(self
            .regions
            .iter()
            .map(|&(offset, bytes)| if bytes == 0 { 0 } else { base + offset as u64 })
            .collect())
    }
}

pub(crate) struct Exlr8Descriptors {
    pub pointers: Vec<u64>,
    pub bits: Vec<i32>,
    pub compact: (Vec<i32>, Vec<i32>, usize),
}
impl Exlr8Descriptors {
    pub fn relocate(residency: &Residency, base: u64) -> Result<Self> {
        let (pointers, bits) = residency.descriptors(base)?;
        Ok(Self {
            pointers,
            bits,
            compact: residency.compact_descriptors()?,
        })
    }
}

/// ADR-063 deliberately omits EXLR8 per-unit checksums on delivered push bytes.
/// Mount still calls Unit::read_verified; this mode never performs mount reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exlr8Integrity {
    PushCheckedPlanNoUnitHash,
}
pub(crate) fn exlr8_push_integrity() -> Exlr8Integrity {
    Exlr8Integrity::PushCheckedPlanNoUnitHash
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_exl3_and_exlr8_descriptors_match_mount_at_same_base() -> Result<()> {
        if let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_EXL3") {
            let catalog = cuteafd_loader::read_expert_catalog(std::path::Path::new(&snapshot))?;
            for world in [4, 6] {
                for rank in 0..world {
                    let residency = catalog.exl3().unwrap().balanced_residency(
                        cuteafd_loader::V41Exl3Layer::Backbone(
                            catalog.routed_experts().first_layer,
                        ),
                        world,
                        rank,
                    )?;
                    let (offsets, bytes) = residency.device_arena_layout()?;
                    let layout = RegionLayout::exl3(&residency)?;
                    assert_eq!(layout.bytes(), bytes);
                    assert_eq!(
                        layout.addresses(0x10000000)?,
                        offsets
                            .iter()
                            .map(|&offset| 0x10000000 + offset as u64)
                            .collect::<Vec<_>>()
                    );
                    for (index, buffer) in residency.buffers.iter().enumerate() {
                        assert_eq!(layout.regions[index].1, buffer.bytes.max(16));
                    }
                }
            }
        }
        if let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_EXLR8") {
            for world in [4, 6] {
                for rank in 0..world {
                    let residency = Residency::plan(
                        std::path::Path::new(&snapshot),
                        "up1",
                        3,
                        world,
                        rank,
                        cuteafd_loader::plan::exlr8::residency::ExpertSet::Routed,
                    )?;
                    let (mount_pointers, mount_bits) = residency.descriptors(0x10000000)?;
                    let push = Exlr8Descriptors::relocate(&residency, 0x10000000)?;
                    assert_eq!(push.pointers, mount_pointers);
                    assert_eq!(push.bits, mount_bits);
                    assert_eq!(push.compact, residency.compact_descriptors()?);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn real_fp8_family_views_preserve_mount_region_order_and_sizes() -> Result<()> {
        for variable in [
            "CUTEAFD_PUSH_FP8_TP6",
            "CUTEAFD_PUSH_NVFP4",
            "CUTEAFD_PUSH_MXFP4",
        ] {
            let Some(snapshot) = std::env::var_os(variable) else {
                continue;
            };
            let snapshot = std::path::Path::new(&snapshot);
            let config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
            let config = config.get("text_config").unwrap_or(&config);
            let get =
                |names: &[&str]| names.iter().find_map(|n| config[n].as_u64()).unwrap() as usize;
            let shape = cuteafd_loader::RoutedExpertShape {
                layers: get(&["num_hidden_layers"]),
                first_layer: config["first_k_dense_replace"]
                    .as_u64()
                    .map(|n| n as usize)
                    .or_else(|| {
                        config["moe_layer_freq"]
                            .as_array()
                            .and_then(|v| v.iter().position(|n| n.as_u64() == Some(1)))
                    })
                    .unwrap_or(0),
                experts: get(&["n_routed_experts", "num_experts"]),
                topk: get(&["num_experts_per_tok"]),
                hidden: get(&["hidden_size"]),
                intermediate: get(&["moe_intermediate_size"]),
                draft_stages: 0,
                draft_experts: 0,
            };
            let tensors =
                Fp8ExpertTensors::read_layers(snapshot, shape, &[shape.first_layer].into())?;
            for world in [4, 6] {
                for rank in 0..world {
                    let slicing = tensors.exact_slicing(world);
                    let layout = RegionLayout::fp8(&tensors, world, rank, slicing)?;
                    for (index, projection) in Fp8Projection::ALL.into_iter().enumerate() {
                        let (weight, _) =
                            tensors.slice_bytes_with(projection, world, rank, slicing)?;
                        assert_eq!(layout.regions[2 * index].1, weight * shape.experts);
                        assert_eq!(
                            layout.regions[2 * index + 1].1,
                            tensors.scale_region_bytes_with(projection, world, rank, slicing)?
                        );
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn native_mount_plane_order_is_unchanged_for_mxfp4_and_nvfp4() -> Result<()> {
        for nvfp4 in [false, true] {
            let sizes = cuteafd_loader::placement::families::deepseek_v41::resident_planes(
                5120, 512, nvfp4,
            )?;
            let layout = RegionLayout::native(5120, 512, 256, nvfp4)?;
            let addresses = layout.addresses(0x10000000)?;
            let mut offset = 0usize;
            for (index, bytes) in sizes.into_iter().map(|n| n as usize * 256).enumerate() {
                offset = (offset + 255) & !255;
                assert_eq!(addresses[index], 0x10000000 + offset as u64);
                assert_eq!(layout.regions[index].1, bytes);
                offset += bytes;
            }
            assert_eq!(addresses.len(), if nvfp4 { 8 } else { 4 });
        }
        assert_eq!(
            exlr8_push_integrity(),
            Exlr8Integrity::PushCheckedPlanNoUnitHash
        );
        Ok(())
    }
}
