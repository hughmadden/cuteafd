use super::*;

#[test]
fn local_activation_contract_selects_bf16_only_for_nvfp4() -> Result<()> {
    for rows in [1, 3, 16, 64, 4096] {
        let bf16 = CuteafdDeviceBuffer {
            ptr: 0x1000usize as *mut c_void,
            bytes: rows as usize * 10240,
            device_id: 0,
            flags: 0,
        };
        let fp8 = CuteafdDeviceBuffer {
            ptr: 0x2000usize as *mut c_void,
            bytes: rows as usize * LocalInputFormat::Fp8K32.row_bytes(ExpertGeometry::DEEPSEEK_V41),
            device_id: 0,
            flags: 0,
        };
        for (nvfp4, expected) in [(false, fp8), (true, bf16)] {
            let format = LocalInputFormat::for_nvfp4(nvfp4);
            let selected = format.select(rows, bf16, fp8, ExpertGeometry::DEEPSEEK_V41)?;
            assert_eq!(selected.ptr, expected.ptr);
            assert_eq!(selected.bytes, expected.bytes);
            assert!(format
                .select(rows, fp8, bf16, ExpertGeometry::DEEPSEEK_V41)
                .is_err());
        }
    }
    Ok(())
}

#[test]
fn local_geometry_rejects_unsupported_native_abi_before_planning() {
    let g = ExpertGeometry::DEEPSEEK_V41;
    assert!(LocalExpertWave::require_geometry(g).is_ok());
    assert!(LocalExpertWave::require_geometry(ExpertGeometry { layers: 1, ..g }).is_ok());
    for geometry in [
        ExpertGeometry { hidden: 4096, ..g },
        ExpertGeometry { topk: 3, ..g },
        ExpertGeometry { layers: 0, ..g },
    ] {
        assert!(LocalExpertWave::require_geometry(geometry).is_err());
    }
}

#[test]
fn shared_local_exl3_retains_exact_capacity_set() -> Result<()> {
    for rows in 1..=64 {
        let expected = if rows == 1 { vec![1] } else { vec![1, 16, 64] };
        assert_eq!(LocalExpertWave::exl3_capacities(rows)?, expected);
    }
    for rows in [256, 1024, 4096] {
        assert_eq!(LocalExpertWave::exl3_capacities(rows)?, vec![1, 16, 64, u64::from(rows)]);
    }
    for rows in [0, 65, 2048, 2752, 4097] {
        assert!(LocalExpertWave::exl3_capacities(rows).is_err());
    }
    Ok(())
}

#[test]
fn native_local_and_tp2_freeze_decode_recipe_selection() -> Result<()> {
    use crate::shared::experts::rtx::{native::kernel_capacities, RtxSelectionPolicy};
    use cuteafd_ffi::V41ExpertReduction;

    for (role, intermediate) in [(2, 2304u32), (3, 1152u32)] {
        let recipe = V41ExpertReduction {
            version: 1,
            slice_width: 192,
            fc2_k_steps: 6,
            plane_count: intermediate.div_ceil(192),
            plane_fold: 1,
            implementation: 1,
            mma_k: 32,
            block_threads: 128,
            fc1_k_stage: 128,
        };
        let incompatible = V41ExpertReduction {
            slice_width: 64,
            fc2_k_steps: 2,
            plane_count: intermediate / 64,
            ..recipe
        };
        let prefill = V41ExpertReduction {
            plane_fold: 2,
            ..recipe
        };
        let capacities = if role == 2 {
            LocalExpertWave::capacities(4096)?
        } else {
            kernel_capacities(4096, &cuteafd_core::V41_TARGET_CAPACITIES.map(|n| n as u32))?
        };
        let mut variants = capacities
            .iter()
            .map(|&capacity| {
                (
                    capacity as usize,
                    match capacity {
                        16 => incompatible,
                        1 | 64 => recipe,
                        _ => prefill,
                    },
                )
            })
            .collect::<Vec<_>>();
        let plan = ExpertReductionPlan::new(&variants, 64)?;
        for rows in 1..=64 {
            let index = if role == 2 {
                plan.execution_index(rows, rows, variants.iter().map(|v| v.0))?
            } else {
                RtxSelectionPolicy::CanonicalDecode.execution_index(
                    rows,
                    rows,
                    variants.iter().map(|v| v.0),
                    || plan.execution_index(rows, rows, variants.iter().map(|v| v.0)),
                )?
            };
            assert_eq!(variants[index].1, recipe, "role {role}, rows {rows}");
            assert_eq!(variants[index].0, if rows == 1 { 1 } else { 64 });
        }
        for (rows, expected) in [(65, 256), (256, 256), (257, 1024), (4096, 4096)] {
            let index = plan.execution_index(rows, rows, variants.iter().map(|v| v.0))?;
            assert_eq!(variants[index], (if role == 2 { 4096 } else { expected }, prefill));
        }
        // Freeze at admission: neither later metadata edits nor re-planning
        // may silently substitute incompatible arithmetic for the live owner.
        variants[2].1 = incompatible;
        assert_eq!(
            plan.execution_index(64, 64, variants.iter().map(|v| v.0))?,
            2
        );
        let error = ExpertReductionPlan::new(&variants, 64).unwrap_err().to_string();
        assert!(error.contains("64 decode rows") && error.contains("slice_width"));
        assert!(ExpertReductionPlan::new(&variants[1..], 64).is_err());
        let no_wide = [(1, recipe), (16, incompatible), (4096, prefill)];
        assert!(ExpertReductionPlan::new(&no_wide, 64).is_err());
    }
    Ok(())
}
