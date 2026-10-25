//! Metadata-only EXLR8 execution arenas, shared by admission and the runtime.
use super::{residency::{execution_workspace_layout, ExpertSet}, PrefillActivations};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::ProgramSpec;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionVariant { SharedDecode, DecodeCompact, DecodeGrouped, Prefill }

impl ExecutionVariant {
    pub fn program(self, activations: PrefillActivations) -> &'static str {
        match self {
            Self::SharedDecode => "exlr8_shared_a16_serving",
            Self::DecodeCompact => "exlr8_a16_serving",
            Self::DecodeGrouped => "exlr8_a16_grouped64",
            Self::Prefill => match activations {
                PrefillActivations::A16 => "exlr8_a16_grouped",
                PrefillActivations::A8 => "exlr8_a8_grouped",
            },
        }
    }
}

#[derive(Debug)]
pub struct ExecutionWorkspace {
    pub ranges: [(usize, usize); 11],
    pub bytes: usize,
    pub rows: usize,
    pub top_k: usize,
    pub max_groups: usize,
    pub hidden: usize,
    pub units: usize,
    pub unit_limit: usize,
    pub intermediate_bytes: usize,
    pub compact: bool,
    pub unit_tables: bool,
}

/// The only owned-unit scratch sizing rule. Compact FC1 uses owned units,
/// compact intermediate retains the global stride; grouped FC1/intermediate
/// both use owned units. FC2 always spans the complete hidden domain.
pub fn execution_workspace(spec: &ProgramSpec, rows: usize, set: ExpertSet,
    variant: ExecutionVariant, unit_limit: usize, activations: PrefillActivations)
    -> Result<ExecutionWorkspace> {
    cuteafd_ffi::exlr8::validate(spec)?;
    ensure!((set == ExpertSet::Shared) == (variant == ExecutionVariant::SharedDecode),
        "EXLR8 shared decode requires the shared expert set");
    ensure!(spec.name == variant.program(activations) && spec.geometry["abi_version"] == 3,
        "EXLR8 execution requires fused rotation ABI v3");
    ensure!(rows > 0 && rows <= spec.capacity_rows as usize, "EXLR8 rows outside admitted K1 capacity");
    ensure!(variant != ExecutionVariant::Prefill || rows > cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS,
        "EXLR8 prefill instance is only admitted above the K1 row limit");
    let dimension = |name| -> Result<usize> {
        usize::try_from(spec.geometry[name].as_u64().context("missing K1 geometry")?)
            .context("K1 geometry exceeds address space")
    };
    let hidden = dimension("hidden")?;
    let top_k = dimension("top_k")?;
    let max_groups = dimension("max_groups")?;
    let units = dimension("units")?;
    let compact = set == ExpertSet::Routed;
    let unit_tables = matches!(variant, ExecutionVariant::DecodeGrouped | ExecutionVariant::Prefill);
    ensure!(!compact || (1..=units).contains(&unit_limit), "invalid compact unit limit");
    let scratch = |name: &str| -> Result<usize> {
        let bytes = usize::try_from(*spec.scratch.get(name).context("missing K1 scratch")?)?;
        let owned = match variant {
            ExecutionVariant::DecodeCompact => name == "fc1",
            ExecutionVariant::DecodeGrouped | ExecutionVariant::Prefill => name != "fc2",
            ExecutionVariant::SharedDecode => false,
        };
        if compact && owned {
            ensure!(bytes % units == 0, "compact scratch must divide by global units");
            (bytes / units).checked_mul(unit_limit).context("compact scratch overflow")
        } else { Ok(bytes) }
    };
    let intermediate_bytes = scratch("intermediate")?;
    let (ranges, bytes) = execution_workspace_layout(rows, spec.capacity_rows as usize,
        spec.geometry["group_rows"].as_u64().unwrap_or(16) as usize,
        hidden, top_k, max_groups, scratch("fc1")?, intermediate_bytes, scratch("fc2")?, set)?;
    Ok(ExecutionWorkspace { ranges, bytes, rows, top_k, max_groups, hidden, units,
        unit_limit: if compact { unit_limit } else { units }, intermediate_bytes, compact, unit_tables })
}

fn program_spec(programs: &Value, name: &str) -> Result<ProgramSpec> {
    let entries = programs["programs"].as_array().context("PROGRAMS.json programs must be an array")?;
    let matches = entries.iter().filter(|entry| entry["name"] == name).collect::<Vec<_>>();
    ensure!(matches.len() == 1, "PROGRAMS.json requires one {name}");
    let entry = matches[0];
    let pointers = entry["pointers"].as_array().context("EXLR8 pointers missing")?.iter()
        .map(|p| p["name"].as_str().map(String::from).context("EXLR8 pointer name invalid")).collect::<Result<_>>()?;
    let scalars = entry["scalars"].as_array().context("EXLR8 scalars missing")?.iter()
        .map(|s| match s["type"].as_str() {
            Some("int32") => Ok(b'i'), Some("float32") => Ok(b'f'),
            _ => anyhow::bail!("EXLR8 scalar dtype invalid"),
        }).collect::<Result<_>>()?;
    let scratch = entry["scratch_bytes_at_capacity"].as_object().context("EXLR8 scratch missing")?.iter()
        .map(|(name, bytes)| Ok((name.clone(), bytes.as_u64().context("EXLR8 scratch invalid")?)))
        .collect::<Result<_>>()?;
    Ok(ProgramSpec { name: name.into(), pointers, scalar_kinds: scalars, scratch,
        capacity_rows: u32::try_from(entry["capacity_rows"].as_u64().context("EXLR8 capacity missing")?)?,
        geometry: entry["geometry"].clone() })
}

/// The stream-serialized TP2 backend owns both decode variants, one prefill
/// execution and a stable BF16 output. Pair's per-lane payloads are separate.
pub fn routed_workspace(programs: &Value, rows: u64, unit_limit: usize,
    activations: PrefillActivations) -> Result<u64> {
    ensure!((1..=4096).contains(&rows), "EXLR8 routed lane geometry");
    let rows = usize::try_from(rows)?;
    let decode_rows = cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS;
    let mut total = 0usize;
    let mut hidden = None;
    for (variant, rows, count) in [(ExecutionVariant::DecodeCompact, decode_rows, 1),
        (ExecutionVariant::DecodeGrouped, decode_rows, 1),
        (ExecutionVariant::Prefill, rows, if rows > decode_rows { 1 } else { 0 })] {
        if count == 0 { continue; }
        let spec = program_spec(programs, variant.program(activations))?;
        let workspace = execution_workspace(&spec, rows, ExpertSet::Routed, variant, unit_limit, activations)?;
        ensure!(workspace.hidden == 6144 && workspace.units == 16 && workspace.top_k == 8,
            "EXLR8 routed TP2 workspace requires hidden6144/units16/top8");
        ensure!(hidden.is_none_or(|h| h == workspace.hidden), "EXLR8 routed workspace hidden mismatch");
        hidden = Some(workspace.hidden);
        total = total.checked_add(workspace.bytes.checked_mul(usize::try_from(count)?)
            .context("EXLR8 routed lane workspace overflow")?).context("EXLR8 routed workspace overflow")?;
    }
    let output = rows.max(decode_rows).checked_mul(hidden.context("EXLR8 missing decode workspace")?)
        .and_then(|bytes| bytes.checked_mul(2)).context("EXLR8 routed output overflow")?;
    Ok(u64::try_from(total.checked_add(output).context("EXLR8 routed workspace overflow")?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(variant: ExecutionVariant, activations: PrefillActivations) -> Value {
        let prefill = variant == ExecutionVariant::Prefill;
        let grouped = matches!(variant, ExecutionVariant::DecodeGrouped | ExecutionVariant::Prefill);
        let (rows, groups) = if prefill { (4096u64, 4096u64) } else { (64, 512) };
        let mut pointers = vec!["x", "pointers", "bits", "group_expert", "route_slots", "route_group_lane",
            "route_weights", "fc1", "intermediate", "fc2", "signs_out", "output", "group_count"];
        if grouped { pointers.extend(["unit_count", "unit_map"]); }
        let mut value = json!({"name":variant.program(activations), "capacity_rows":rows,
            "pointers":pointers.iter().map(|name| json!({"name":name})).collect::<Vec<_>>(),
            "scalars":(0..if grouped {3} else {2}).map(|_| json!({"type":"int32"})).collect::<Vec<_>>(),
            "geometry":{"abi_version":3,"hidden":6144,"units":16,"top_k":8,
                "max_groups":groups,"max_rows":rows,"group_rows":16,"split_k_blocks":8,
                "fc1_splits":6,"ctas_per_sm":4},
            "scratch_bytes_at_capacity":{"fc1":groups*16*2*6*16*128*4,
                "intermediate":groups*16*16*128*2,"fc2":groups*16*6144*4}});
        if variant == ExecutionVariant::DecodeCompact { value["geometry"]["serving_variant"] = json!("k1_k23_coalesced"); }
        if variant == ExecutionVariant::DecodeGrouped { value["geometry"]["fc2_partials"] = json!("f32"); }
        if activations == PrefillActivations::A8 && prefill { value["geometry"]["activations"] = json!("a8"); }
        value
    }

    #[test]
    fn adaptive_units_size_row_planes_without_shrinking_exported_scratch() {
        for activations in [PrefillActivations::A16, PrefillActivations::A8] {
            let programs = json!({"programs":[entry(ExecutionVariant::Prefill, activations)]});
            let spec = program_spec(&programs, ExecutionVariant::Prefill.program(activations)).unwrap();
            let full = execution_workspace(&spec, 4096, ExpertSet::Routed,
                ExecutionVariant::Prefill, 8, activations).unwrap();
            for rows in [2048, 2752] {
                let unit = execution_workspace(&spec, rows, ExpertSet::Routed,
                    ExecutionVariant::Prefill, 8, activations).unwrap();
                assert_eq!(unit.rows, rows);
                assert_eq!(unit.ranges[0].1, rows * (6144 + 6144 / 32));
                assert_eq!(unit.ranges[1].1, rows * 8 * 4);
                assert_eq!(unit.ranges[3].1, rows * 6144 * 2);
                for index in [4, 5, 8, 9] {
                    assert_eq!(unit.ranges[index].1, full.ranges[index].1);
                }
                assert_eq!(unit.max_groups, full.max_groups);
                assert!(unit.bytes < full.bytes);
            }
        }
    }

    #[test]
    fn routed_backend_charges_both_decode_arenas_prefill_and_stable_output() {
        for activations in [PrefillActivations::A16, PrefillActivations::A8] {
            let programs = json!({"programs":[entry(ExecutionVariant::DecodeCompact, activations),
                entry(ExecutionVariant::DecodeGrouped, activations), entry(ExecutionVariant::Prefill, activations)]});
            for rows in [1u64, 64, 65, 1024, 2048, 2752, 4096] {
                let mut expected = rows.max(64) * 6144 * 2;
                for variant in [ExecutionVariant::DecodeCompact, ExecutionVariant::DecodeGrouped, ExecutionVariant::Prefill] {
                    if variant == ExecutionVariant::Prefill && rows <= 64 { continue; }
                    let spec = program_spec(&programs, variant.program(activations)).unwrap();
                    let live = if variant == ExecutionVariant::Prefill { rows as usize } else { 64 };
                    let owned = execution_workspace(&spec, live, ExpertSet::Routed, variant, 8, activations).unwrap();
                    expected += owned.bytes as u64;
                    assert_eq!(owned.compact, true);
                    assert_eq!(owned.intermediate_bytes, spec.scratch["intermediate"] as usize
                        / if variant == ExecutionVariant::DecodeCompact { 1 } else { 2 });
                    let full = execution_workspace(&spec, live, ExpertSet::Routed, variant, 16, activations).unwrap();
                    assert!(full.bytes > owned.bytes);
                }
                assert_eq!(routed_workspace(&programs, rows, 8, activations).unwrap(), expected);
            }
            for units in [0, 17, usize::MAX] { assert!(routed_workspace(&programs, 4096, units, activations).is_err()); }
            assert!(routed_workspace(&programs, 4097, 8, activations).is_err());
            let mut invalid = programs.clone();
            invalid["programs"][1]["scratch_bytes_at_capacity"]["fc2"] = json!(1);
            assert!(routed_workspace(&invalid, 4096, 8, activations).is_err());
            invalid = programs.clone();
            invalid["programs"].as_array_mut().unwrap().remove(1);
            assert!(routed_workspace(&invalid, 4096, 8, activations).is_err());
        }
    }
}
