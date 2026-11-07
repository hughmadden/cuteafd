//! Final resident expert views, planned before the first upload. Load staging
//! and early-release raw scales never belong to an onboarding arena.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::serving_capacity::WeightArenaLayout;
use crate::shared::memory::{device::Device, weight_arena::DeviceWeightArena};

pub(crate) fn name(layer: super::layer::ExpertLayer, plane: usize) -> String {
    format!("{layer:?}.{plane}")
}

pub(crate) fn arena<'a>(device: Device<'a>, format: &str,
    regions: impl IntoIterator<Item = (String, usize)>) -> Result<DeviceWeightArena<'a>> {
    DeviceWeightArena::new(device, layout(device.id, format, regions)?)
}

pub(crate) fn layout(device: i32, format: &str,
    regions: impl IntoIterator<Item = (String, usize)>) -> Result<WeightArenaLayout> {
    let regions = regions.into_iter().collect::<Vec<_>>();
    let id = cuteafd_loader::serving_capacity::expert_weight_arena_id(format, device, &regions)?;
    WeightArenaLayout::new(id, regions)
}

#[cfg(test)]
pub(crate) fn assert_receipts(receipts: &[cuteafd_loader::placement::planner::Receipt],
    layouts: &[WeightArenaLayout]) -> Result<()> {
    use cuteafd_core::memory_layout::Category;
    use std::collections::BTreeMap;
    let roots = receipts.iter().filter(|receipt| receipt.allocation.view.is_none()
        && receipt.allocation.category == Category::Weights && receipt.allocation.group == "weight arena")
        .map(|receipt| (receipt.allocation.id.as_str(), receipt.allocation.bytes)).collect::<Vec<_>>();
    let actual = roots.iter().copied().collect::<BTreeMap<_, _>>();
    let expected = layouts.iter().map(|layout| (layout.id(), layout.bytes() as u64)).collect::<BTreeMap<_, _>>();
    ensure!(actual.len() == roots.len() && expected.len() == layouts.len(), "duplicate expert backing roots");
    ensure!(actual == expected, "expert backing receipts differ from selected layouts: {actual:?} != {expected:?}");
    let total = |bytes: Vec<u64>| bytes.into_iter().try_fold(0u64, |sum, bytes|
        sum.checked_add(bytes).context("expert backing byte sum overflow"));
    ensure!(total(actual.into_values().collect())? == total(layouts.iter().map(|l| l.bytes() as u64).collect())?,
        "expert backing receipt sum differs from arena layouts");
    Ok(())
}

pub(crate) fn admit(layout: &WeightArenaLayout, staging: usize, workspace: usize,
    available: usize) -> Result<()> {
    let peak = layout.bytes().checked_add(staging).and_then(|n| n.checked_add(workspace))
        .context("expert arena startup peak overflow")?;
    ensure!(peak <= available,
        "expert arena {} needs {} resident + {staging} staging + {workspace} workspace bytes, admitted {available}",
        layout.id(), layout.bytes());
    Ok(())
}

