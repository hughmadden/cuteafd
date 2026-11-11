//! One caller-admitted serial projection workspace per lane or prefill owner.
use crate::shared::memory::{device::{Allocation, Device}, receipts::Inventory};
use anyhow::{ensure, Result};
use cuteafd_ffi::{native_dense::DenseLinear, native_dense_glue::{Companion, Op}, programs::Programs};
use std::collections::BTreeMap;
use cuteafd_core::memory_layout::Tag;
use cuteafd_loader::families::minimax_m3::weight_layout::{CoordinatorWeights, DenseWorkspace};

/// Loaded handles borrow the installed program table; no weights or writable
/// workspaces are allocated here. Each physical device binds its own handles.
pub(crate) struct NativePrograms<'a> {
    pub device: Device<'a>,
    pub linears: BTreeMap<String, DenseLinear<'a>>,
    pub quantizers: BTreeMap<usize, Companion<'a>>,
}

impl<'a> NativePrograms<'a> {
    pub fn bind(device: Device<'a>, programs: &'a Programs<'a>, weights: &CoordinatorWeights,
        manifest: &serde_json::Value, rows: u32) -> Result<Self> {
        ensure!(std::ptr::eq(device.library, programs.library()), "MiniMax dense program library differs from device");
        weights.dense_workspace(rows)?;
        let specs = weights.resolve_dense_programs(manifest, rows)?;
        // Validate every selected entry before loading any module. A separately
        // validated manifest is not proof that this library has the same ABI.
        for expected in &specs {
            let actual = programs.spec(&expected.name)?;
            ensure!(actual.pointers == expected.pointers && actual.scalar_kinds == expected.scalar_kinds
                && actual.scratch == expected.scratch && actual.capacity_rows == expected.capacity_rows
                && actual.geometry == expected.geometry,
                "MiniMax installed program {} differs from admitted manifest", expected.name);
        }
        device.run(|| {
            let mut linears = BTreeMap::new();
            let mut quantizers = BTreeMap::new();
            for projection in &weights.projections {
                let name = projection.program_name();
                if !linears.contains_key(&name) {
                    let linear = DenseLinear::bind(programs, &name)?;
                    ensure!(linear.geometry() == projection.geometry(), "MiniMax bound dense geometry differs");
                    linears.insert(name, linear);
                }
                if !quantizers.contains_key(&projection.input) {
                    let name = format!("exlr8_dense_quant_k{}_bf16_a8", projection.input);
                    let op = Op::Mx8QuantBf16 { features: u32::try_from(projection.input)? };
                    quantizers.insert(projection.input, Companion::bind(programs, &name, op)?);
                }
            }
            Ok(Self { device, linears, quantizers })
        })
    }
}

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

    fn manifest() -> serde_json::Value {
        use serde_json::json;
        let scalars = json!([{"name":"rows","type":"int32"},
            {"name":"rotation_id","type":"int32"}, {"name":"cta_limit","type":"int32"}]);
        json!({"programs":[
            {"name":"exlr8_dense_mxfp8_k128_n32_s1_a8","op":"dense_mxfp8",
                "capacity_rows":4096,"scratch_bytes_at_capacity":{},"scalars":scalars,
                "geometry":{"input_features":128,"output_features":32,"sharded_side":1,"encoding":"mxfp8",
                    "input_dtype":"bf16","rotation_count":1,"max_rows":4096,"abi_version":2},
                "pointers":[{"name":"rotated_inputs","dtype":"int64","shape":"[1]","role":"in"},
                    {"name":"payload","dtype":"uint8","shape":"[4096]","role":"in"},
                    {"name":"scales","dtype":"uint8","shape":"[128]","role":"in"},
                    {"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"in"},
                    {"name":"output","dtype":"float32","shape":"[rows,32]","role":"out"}]},
            {"name":"exlr8_dense_quant_k128_bf16_a8","op":"dense_mxfp8_quant",
                "capacity_rows":4096,"scratch_bytes_at_capacity":{},"scalars":scalars,
                "geometry":{"input_features":128,"input_dtype":"bf16","rotation_count":1,"max_rows":4096,"abi_version":2},
                "pointers":[{"name":"rotated_inputs","dtype":"int64","shape":"[1]","role":"in"},
                    {"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"out"}]}
        ]})
    }

    #[test]
    fn binds_selected_native_programs_on_owner_device_without_workspace_allocation() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 1 };
        let mut weights = weights()?;
        // Multiple checkpoint projections share the same shape's loaded handles.
        weights.projections.push(weights.projections[0].clone());
        let manifest = manifest();
        fixture.register_programs(&library, manifest["programs"].as_array().unwrap())?;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("PROGRAMS.json");
        std::fs::write(&path, serde_json::to_vec(&manifest)?)?;
        let programs = library.programs()?.with_manifest(&path)?;
        let bound = NativePrograms::bind(device, &programs, &weights, &manifest, 17)?;
        assert_eq!(bound.device.id, 1);
        assert_eq!(bound.linears.len(), 1);
        assert_eq!(bound.quantizers.len(), 1);
        assert_eq!(bound.linears.values().next().unwrap().geometry(), weights.projections[0].geometry());
        assert_eq!(bound.quantizers[&128].op, Op::Mx8QuantBf16 { features: 128 });
        assert_eq!(library.cuda_get_device()?, 0);
        assert_eq!(fixture.events()?.matches('D').count(), 0);

        let mut wrong = manifest.clone();
        wrong["programs"][0]["geometry"]["admission_marker"] = serde_json::json!(1);
        let error = NativePrograms::bind(device, &programs, &weights, &wrong, 17).err().unwrap();
        assert!(error.to_string().contains("differs from admitted manifest"), "{error:#}");
        let mut missing = manifest.clone();
        missing["programs"].as_array_mut().unwrap().pop();
        assert!(NativePrograms::bind(device, &programs, &weights, &missing, 17).is_err());
        assert!(NativePrograms::bind(device, &programs, &weights, &manifest, 0).is_err());
        weights.projections.clear();
        assert!(NativePrograms::bind(device, &programs, &weights, &manifest, 17).is_err());
        assert_eq!(library.cuda_get_device()?, 0);
        assert_eq!(fixture.events()?.matches('D').count(), 0);
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
