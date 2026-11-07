//! Offline calibration for #1178's TP6 allocation budget, not a live inventory.
//! Other worlds and wider exports retain the conservative package bound.
use anyhow::{Context, Result};
use cuteafd_core::{OwnerBytes, OwnerExtent, SparkExecutions};

fn calibrated(world: usize, executions: &SparkExecutions) -> bool {
    runtime_bytes(world, executions) != cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_BYTES
}

/// Allocation allowance shared by coordinator admission and worker load/serve
/// checks. MemAvailable already excludes OS/driver occupancy; the calibrated
/// TP6 ledger retains only the synchronous upload allocation instead.
pub fn runtime_bytes(world: usize, executions: &SparkExecutions) -> u64 {
    cuteafd_core::memory_layout::spark_runtime_bytes(Some((cuteafd_core::ExpertGeometry::KIMI_K3,
        world, executions.compiled_capacity())))
}

/// The 2026-11-04 TP6 census retained m1/m80 decode and m1/m80/m256/m1024
/// prefill owners. Use those maxima for smaller admitted exports too; never
/// extrapolate this rank-width measurement to other Spark worlds.
pub fn owner_bytes(world: usize, extent: OwnerExtent) -> Result<OwnerBytes> {
    if world != 6 || extent.compiled > 1024 { return super::spark_owner_bytes(extent); }
    Ok(OwnerBytes {
        scratch: if extent.compiled <= 80 { 29_782_116 } else { 219_531_260 },
        io: u64::from(extent.capacity) * (3584 + 3584 / 32 + 16 * 8),
        pinned_staging: u64::from(extent.staging_rows) * 16 * 8,
    })
}

/// This is an allocation budget: OS/driver memory is outside it. Keep legacy
/// endpoint I/O as a conservative extra, but replace the generic 2.5 GiB
/// runtime allowance with the measured 4 KiB synchronous upload staging.
/// Source: #1178, issuecomment-6621259666 (all six worker allocation ledgers).
pub fn reserve_bytes(world: usize, waves: u64, unit_rows: u64) -> Result<u64> {
    let executions = super::spark_executions(waves, unit_rows)?;
    if !calibrated(world, &executions) { return super::spark_reserve_bytes(waves, unit_rows); }
    executions.workspace_bytes(|extent| owner_bytes(world, extent))?
        .and_then(|bytes| bytes.total()).and_then(|bytes| bytes.checked_add(runtime_bytes(world, &executions)))
        .context("Kimi calibrated Spark reserve overflow")
}

/// Use the same calibration in pre-placement admission and the published ledger.
pub fn charge(device: &mut cuteafd_core::memory_layout::DeviceLayout, world: usize,
    executions: &SparkExecutions) -> Result<()> {
    use cuteafd_core::memory_layout::{Basis, Category, Item, SPARK_RUNTIME_RESIDUAL_GROUP};
    device.items.retain(|item| item.group != "Kimi synchronous upload staging");
    if calibrated(world, executions) {
        device.items.retain(|item| item.group != SPARK_RUNTIME_RESIDUAL_GROUP);
        device.items.push(Item::new(Category::Staging, "Kimi synchronous upload staging", "pinned",
            runtime_bytes(world, executions), Basis::Calibrated));
    }
    crate::plan::layout::spark_slots::charge_owners(device, executions, |extent| {
        if calibrated(world, executions) { owner_bytes(world, extent) } else { super::spark_owner_bytes(extent) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::memory_layout::{DeviceKind, DeviceLayout, Item, Basis, Category};

    #[test]
    fn measured_tp6_zero_onboard_fits_110_gib_without_weakening_other_worlds() -> Result<()> {
        let executions = super::super::spark_executions(3, 1024)?;
        let retained = executions.workspace_bytes(|extent| owner_bytes(6, extent))?.unwrap();
        assert_eq!(retained.device_bytes, 253_473_888);
        assert_eq!(retained.pinned_bytes, 401_408);
        let resident = 117_692_170_240;
        let rings = 134_217_728;
        let legacy_io = 1024 * ((3584 + 112 + 128 + 7168) + (128 + 7168) + 4) + 1024 * 4 + 128;
        let conservative_total = resident + rings + legacy_io + reserve_bytes(6, 3, 1024)?;
        assert_eq!(conservative_total, 118_099_002_592);
        let inventory = crate::placement::SparkBudgetInventory {
            layers: (1..=92).collect(), resident: vec![vec![1_279_262_720; 92]; 6],
            overhead: vec![conservative_total - resident; 6], budget: 110 << 30,
        };
        assert_eq!(inventory.minimum_rtx_layers()?, 0);
        for world in [1, 2, 3, 4, 5, 7, 8] {
            assert_eq!(reserve_bytes(world, 3, 1024)?, super::super::spark_reserve_bytes(3, 1024)?);
        }
        assert_eq!(reserve_bytes(6, 3, 2752)?, super::super::spark_reserve_bytes(3, 2752)?);
        Ok(())
    }

    #[test]
    fn calibration_recharges_the_ledger_without_duplicate_upload_or_runtime_reserve() -> Result<()> {
        let executions = super::super::spark_executions(3, 1024)?;
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0,
            capacity_bytes: 121 << 30, worker_budget_bytes: 110 << 30,
            kv_tokens: 0, items: vec![] };
        for _ in 0..2 {
            device.items.push(cuteafd_core::memory_layout::spark_runtime_residual());
            charge(&mut device, 6, &executions)?;
            assert_eq!(device.items.iter().map(|item| item.bytes).sum::<u64>(), reserve_bytes(6, 3, 1024)?);
        }
        // Other worlds keep their original residual and owner accounting.
        let mut device = DeviceLayout { items: vec![Item::new(Category::Workspace,
            cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_GROUP, "", 5 << 29, Basis::Calibrated)], ..device };
        charge(&mut device, 5, &executions)?;
        assert_eq!(device.items.iter().map(|item| item.bytes).sum::<u64>(), reserve_bytes(5, 3, 1024)?);
        Ok(())
    }
}
