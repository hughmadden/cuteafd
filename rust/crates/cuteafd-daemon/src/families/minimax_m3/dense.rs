//! One caller-admitted serial projection workspace per lane or prefill owner.
use crate::shared::memory::{device::{Allocation, Device}, receipts::Inventory};
use anyhow::Result;
use cuteafd_core::memory_layout::Tag;
use cuteafd_loader::families::minimax_m3::weight_layout::{CoordinatorWeights, DenseWorkspace};

pub(crate) struct Workspace<'a> {
    pub layout: DenseWorkspace,
    pub inputs: Allocation<'a>,
    pub records: Allocation<'a>,
    pub output: Allocation<'a>,
}

impl<'a> Workspace<'a> {
    /// Admission must include these three physical requests and their tails.
    /// Inputs are a one-entry BF16 pointer table, not an activation conversion.
    pub fn new(device: Device<'a>, weights: &CoordinatorWeights, rows: u32) -> Result<Self> {
        let layout = weights.dense_workspace(rows)?;
        let records_bytes = usize::try_from(layout.records)?;
        let output_bytes = usize::try_from(layout.output)?;
        let inputs = Allocation::new(device, 8, Tag::workspace("minimax/dense/inputs"))?;
        let records = Allocation::new(device, records_bytes, Tag::workspace("minimax/dense/records"))?;
        let output = Allocation::new(device, output_bytes, Tag::workspace("minimax/dense/output"))?;
        Ok(Self { layout, inputs, records, output })
    }

    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, allocation) in [("inputs", &self.inputs), ("records", &self.records), ("output", &self.output)] {
            allocation.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::{families::minimax_m3::weight_layout::Mxfp8Projection,
        placement::planner::{ComponentKind, CoordinatorDevices, Lifetime}, serving_capacity::WeightArenaLayout};
    use cuteafd_core::memory_layout::Category;

    fn weights() -> Result<CoordinatorWeights> {
        Ok(CoordinatorWeights { rank: 0, layers: vec![0],
            arena: WeightArenaLayout::new("fixture", [("unused".into(), 256)])?,
            projections: vec![Mxfp8Projection { stem: "fixture".into(), input: 128, output: 32 }] })
    }

    #[test]
    fn serial_workspace_receipts_match_admitted_requests_and_restore_device() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.strict_device_free(&library, true)?;
        fixture.configure_memory(&library, 1, 1 << 20)?;
        let device = Device { library: &library, id: 1 };
        let weights = weights()?;
        let workspace = Workspace::new(device, &weights, 17)?;
        assert_eq!(workspace.layout, DenseWorkspace { rows: 17, records: 2 * 4 * 528, output: 17 * 32 * 4 });
        assert_eq!(library.cuda_get_device()?, 0);
        let devices = [1];
        let mut inventory = Inventory::new(CoordinatorDevices::new(&devices)?, "minimax", Lifetime::TARGET,
            Category::Workspace, "native serial dense projection");
        workspace.inventory(&mut inventory, "dense")?;
        assert_eq!(inventory.receipts.len(), 3);
        let expected = workspace.layout.allocations("minimax.dense", 0, Lifetime::TARGET);
        for (actual, expected) in inventory.receipts.iter().zip(expected) {
            assert_eq!(actual.allocation, expected);
            assert!(actual.address != 0);
        }
        assert_eq!(workspace.inputs.buffer.bytes, 8);
        assert_eq!(fixture.events()?.matches('D').count(), 3);
        let bytes = 8 + workspace.layout.records as usize + workspace.layout.output as usize;
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, (1 << 20) - bytes);
        // Distinct live owners must not share their writable quantization records.
        let other = Workspace::new(device, &weights, 17)?;
        assert_ne!(workspace.records.buffer.ptr, other.records.buffer.ptr);
        drop(other);
        drop(workspace);
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 1 << 20);
        assert_eq!(fixture.events()?.matches('d').count(), 6);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        let wide = weights.dense_workspace(4096)?;
        let components = wide.components("dense", 0, Lifetime::TARGET, 32)?;
        assert_eq!(components.iter().filter(|c| matches!(c.kind, ComponentKind::Allocation { .. })).count(), 3);
        Ok(())
    }

    #[test]
    fn invalid_dense_owner_refuses_before_allocation() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 1 };
        let mut weights = weights()?;
        for rows in [0, 4097] { assert!(Workspace::new(device, &weights, rows).is_err()); }
        weights.projections.clear();
        assert!(Workspace::new(device, &weights, 1).is_err());
        assert_eq!(fixture.events()?.matches('D').count(), 0);
        assert_eq!(library.cuda_get_device()?, 0);
        Ok(())
    }
}
