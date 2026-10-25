//! Conservative installed-package bounds until manifest-backed planning (#776).
use super::*;
use crate::formats::fp8_experts::ExpertFormat;
use crate::placement::SparkWorkingSet;

/// Compiled 4096-row scratch maxima from the installed-package census in
/// #778, issuecomment-6401148082. Scratch does not scale with live rows: both
/// 2048 and 2752 select the 4096-row export. MiMo/Step include BF16-input siblings.
fn group_scratch(family: &str, format: ExpertFormat) -> anyhow::Result<u64> {
    use ExpertFormat::*;
    let bytes = match (family, format) {
        ("glm5", Fp8Block128) => 604_246_016,
        ("glm5", Nvfp4) => 667_162_624,
        ("glm5_flash", Fp8Block128) => 470_028_288,
        ("glm5_flash", Nvfp4) => 469_243_904,
        ("mimo_v2", Fp8Block128) => 704_977_920,
        ("mimo_v2", Mxfp4) => 478_417_920,
        ("qwen4", Nvfp4) => 401_745_920,
        ("step5", Fp8Block128) => 538_279_936,
        ("step5", Nvfp4) => 464_821_248,
        ("step5", Bf16) => 580_747_264,
        _ => anyhow::bail!("no calibrated Spark slot scratch for {family}/{format:?}; add the package census or manifest-backed sizing (#776)"),
    };
    Ok(bytes)
}

fn group_slot_bytes(family: &str, format: ExpertFormat, geometry: cuteafd_core::ExpertGeometry,
    rows: u64) -> anyhow::Result<u64> {
    let hidden = u64::from(geometry.hidden);
    // BF16-input siblings are installed for block-FP8 MiMo/Step; NVFP4 and
    // BF16 weights consume BF16 directly. Other packages use FP8 K32 ingress.
    let bf16 = matches!(format, ExpertFormat::Nvfp4 | ExpertFormat::Bf16)
        || (matches!(family, "mimo_v2" | "step5") && format == ExpertFormat::Fp8Block128);
    let input = if bf16 { 2 * hidden } else { hidden + hidden / 32 };
    Ok(group_scratch(family, format)?
        + rows * (input + 8 * u64::from(geometry.topk) + 2 * hidden))
}

/// Match the worker's retained groups, not the number of layers. Unequal
/// rotated widths and mixed weight formats each retain separate workers.
pub(crate) fn native_slot_bytes(input: &super::super::PlanInput, family: &str,
    working: &SparkWorkingSet, rank: usize, rows: u64) -> anyhow::Result<Option<u64>> {
    if !working.selected_ranks().contains(&rank) { return Ok(None); }
    let catalog = input.expert_catalog()?;
    let Some(tensors) = catalog.fp8() else { return Ok(None); };
    let shape = tensors.shape();
    let geometry = cuteafd_core::ExpertGeometry { hidden: u32::try_from(shape.hidden)?,
        experts: u32::try_from(shape.experts)?, topk: u32::try_from(shape.topk)?,
        intermediate: u32::try_from(shape.intermediate)?, layers: u32::try_from(shape.layers)? };
    let mut groups = Vec::new();
    let mut total = 0;
    for &layer in &working.spark_layers {
        if !tensors.has_layer(layer) { continue; }
        let view = tensors.for_layer(layer)?;
        let format = view.format();
        let package_family = match family {
            "glm5" => "glm", "glm5_flash" => "glmf",
            "mimo_v2" => if format == ExpertFormat::Mxfp4 {
                if shape.hidden == 6144 { "mimop" } else { "mimof" }
            } else { "mimo" },
            other => other,
        };
        let format_name = match format {
            ExpertFormat::Nvfp4 => "nvfp4", ExpertFormat::Bf16 => "bf16", _ => "fp8",
        };
        let recipe = super::super::experts::spark_recipe(&format!("{package_family}:{format_name}"),
            shape.intermediate, working.spark_ranks)?;
        let width = view.rank_width(working.spark_ranks, rank, recipe.slicing)?;
        let key = (format, view.activation(layer), width);
        if !groups.contains(&key) {
            groups.push(key);
            total += group_slot_bytes(family, format, geometry, rows)?;
        }
    }
    anyhow::ensure!(!groups.is_empty(), "native Spark slot has no selected expert groups");
    Ok(Some(total))
}

/// Every native slot owns scratch and I/O. Only bytes actually retained in the
/// old first-slot residual can be credited; extra mixed/rotated groups cannot.
pub(crate) fn charge_native_slots(device: &mut DeviceLayout, slots: u8, per_slot: u64) {
    device.items.retain(|item| item.group != "execution slots");
    let residual = device.items.iter().filter(|item| item.group == "expert waves")
        .map(|item| item.bytes).sum::<u64>();
    let credit = residual.min(per_slot);
    device.items.push(Item::new(Category::Workspace, "execution slots", format!("{slots} total; {credit} first-slot bytes already reserved"),
        per_slot * u64::from(slots) - credit, Basis::Calibrated));
}

fn exl3_slot_bytes(family: &str, rows: u64) -> u64 {
    // EXL3 retains the arena maxima plus private state for every capacity.
    // These bounds include the new 2048/2752 capacities; native V4 stays 208 MiB.
    let compiled: u64 = match family {
        "glm5_flash" => 607 * (1 << 20),
        "deepseek_v4" | "deepseek_v41" => 810 * (1 << 20),
        _ => return spark_slot_bytes(family, rows),
    };
    compiled.saturating_mul(rows.max(4096)).div_ceil(4096)
}

pub(crate) fn charge_selected_slots(device: &mut DeviceLayout, input: &super::super::PlanInput,
    working: &SparkWorkingSet, family: &str, slots: u8, rows: u64) -> anyhow::Result<()> {
    if !working.selected_ranks().contains(&(device.index as usize)) { return Ok(()); }
    match native_slot_bytes(input, family, working, device.index as usize, rows)? {
        Some(bytes) => charge_native_slots(device, slots, bytes),
        None if input.expert_catalog()?.exl3().is_some() =>
            charge_native_slots(device, slots, exl3_slot_bytes(family, rows)),
        None => charge_spark_slots(device, working, family, slots, rows),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_bounds_every_shipped_native_package_census() {
        let census: serde_json::Value = serde_json::from_str(include_str!("spark_slot_census.json")).unwrap();
        let capacities = [1, 16, 80, 256, 1024, 2048, 2752, 4096];
        for row in census.as_array().unwrap() {
            let package = row["package"].as_str().unwrap();
            let family = row["family"].as_str().unwrap();
            let format = match row["format"].as_str().unwrap() {
                "fp8" => ExpertFormat::Fp8Block128, "mxfp4" => ExpertFormat::Mxfp4,
                "nvfp4" => ExpertFormat::Nvfp4, "bf16" => ExpertFormat::Bf16, _ => unreachable!(),
            };
            let hidden = row["hidden"].as_u64().unwrap();
            let topk = row["topk"].as_u64().unwrap();
            let input = if row["bf16"].as_bool().unwrap() { 2 * hidden } else { hidden + hidden / 32 };
            let geometry = cuteafd_core::ExpertGeometry { hidden: hidden as u32, topk: topk as u32,
                experts: 1, intermediate: 1, layers: 1 };
            for (index, rows) in capacities.into_iter().enumerate() {
                let exact = row["scratch"][index].as_u64().unwrap() + rows * (input + 8 * topk + 2 * hidden);
                let bound = group_slot_bytes(family, format, geometry, rows).unwrap();
                assert!(bound >= exact, "{package} capacity {rows}: planner {bound} < census {exact}");
            }
        }
    }

    #[test]
    fn compiled_scratch_is_not_scaled_by_live_unit_rows() {
        let geometry = cuteafd_core::ExpertGeometry::GLM5;
        let full = group_slot_bytes("glm5", ExpertFormat::Nvfp4, geometry, 4096).unwrap();
        let partial = group_slot_bytes("glm5", ExpertFormat::Nvfp4, geometry, 2752).unwrap();
        assert!(partial > full * 2752 / 4096);
        assert_eq!(full - partial, (4096 - 2752) * (4 * 6144 + 8 * 8));
    }

    #[test]
    fn rotated_widths_sum_worker_owners_but_repeated_layers_do_not() -> anyhow::Result<()> {
        use crate::plan::testing;
        let root = tempfile::tempdir()?;
        let mut cfg = testing::mimo_pro_config();
        cfg["num_hidden_layers"] = serde_json::json!(7);
        cfg["hybrid_layer_pattern"] = serde_json::json!([0, 1, 1, 1, 1, 1, 1]);
        cfg["moe_layer_freq"] = serde_json::json!([0, 1, 1, 1, 1, 1, 1]);
        let mut tensors = testing::mimo_pro_tensors();
        let layer = tensors.iter().filter(|(name, ..)| name.starts_with("model.layers.1."))
            .cloned().collect::<Vec<_>>();
        for index in 2..7 {
            tensors.extend(layer.iter().map(|(name, dtype, shape)|
                (name.replacen("model.layers.1.", &format!("model.layers.{index}."), 1), *dtype, shape.clone())));
        }
        testing::write_snapshot(root.path(), &cfg, &tensors, None);
        let input = crate::plan::PlanInput::open(root.path())?;
        let mut working = SparkWorkingSet { spark_layers: vec![1], spark_ranks: 6,
            prefill_lanes: 2, ..Default::default() };
        let one = native_slot_bytes(&input, "mimo_v2", &working, 0, 4096)?.unwrap();
        working.spark_layers = (1..7).collect();
        let two = native_slot_bytes(&input, "mimo_v2", &working, 0, 4096)?.unwrap();
        assert_eq!(two, one * 2, "w352 and w320 retain distinct workers, not six layers");
        let exact = 472_126_464 + 465_835_008 + 2 * 76_546_048;
        assert!(two >= exact);
        Ok(())
    }

    #[test]
    fn exl3_bounds_include_private_capacities_and_low_row_alias_owners() {
        use serde_json::json;
        // v3.0.1 SM121 TP2 maxima. rotation_up aliases rotation_gate at
        // m4096, but smaller capacities retain a separate owner of that name.
        for (family, hidden, topk, arenas, expected) in [
            ("glm5_flash", 4096, 8, vec![("rotation_gate", 268_435_456), ("rotation_up", 5_242_880),
                ("fc1", 134_217_728), ("activated", 67_108_864), ("output", 33_554_432),
                ("packed_route_indices", 203_648), ("block_expert_ids", 3_184), ("packed_route_count", 16),
                ("expert_offsets", 1_156), ("expert_counts", 1_152),
                ("fc1_scratch", 12_582_912), ("fc2_scratch", 12_582_912)], 635_693_380),
            ("deepseek_v4", 7168, 6, vec![("rotation_gate", 352_321_536), ("rotation_up", 6_881_280),
                ("fc1", 150_994_944), ("activated", 75_497_472), ("output", 58_720_256),
                ("packed_route_indices", 195_072), ("block_expert_ids", 3_048), ("packed_route_count", 16),
                ("expert_offsets", 1_540), ("expert_counts", 1_536),
                ("fc1_scratch", 12_582_912), ("fc2_scratch", 12_582_912)], 847_635_004),
            ("deepseek_v41", 5120, 6, vec![("rotation_gate", 251_658_240), ("rotation_up", 251_658_240),
                ("fc1", 113_246_208), ("activated", 56_623_104), ("output", 41_943_040),
                ("packed_route_indices", 109_056), ("block_expert_ids", 13_632), ("packed_route_count", 16),
                ("expert_offsets", 1_540), ("expert_counts", 1_536),
                ("fc1_scratch", 3_145_728), ("fc2_scratch", 3_145_728)], 848_604_052),
        ] {
            let manifests = cuteafd_core::EXL3_CAPACITIES.into_iter().map(|rows| {
                let mut buffers = serde_json::Map::new();
                for &(name, bytes) in &arenas {
                    buffers.insert(name.into(), json!({"allocation": name, "bytes": bytes,
                        "dtype": "census", "zero_on_create": false}));
                }
                buffers.insert("workspace".into(), json!({"allocation": "workspace", "bytes": 776,
                    "dtype": "i32", "zero_on_create": true}));
                json!({"capacity": rows, "hidden": hidden, "trellis_lut": {"bytes": 4096}, "buffers": buffers})
            }).collect::<Vec<_>>();
            let retained = crate::serving_capacity::exl3_workspace_bytes(&manifests, true).unwrap();
            let wire_routes = 4096 * (hidden + hidden / 32 + 8 * topk);
            assert_eq!(retained + wire_routes, expected, "{family}");
            for rows in [64, 2048, 2752, 4096] {
                assert!(exl3_slot_bytes(family, rows) >= expected, "{family} rows {rows}");
            }
        }
    }

    #[test]
    fn selected_catalog_uses_exl3_floor_without_changing_native_v4() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        crate::plan::testing::write_v41_exl3_snapshot(root.path(), 3);
        let input = crate::plan::PlanInput::open(root.path())?;
        let working = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4,
            prefill_lanes: 2, ..Default::default() };
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0, capacity_bytes: 121 * GIB,
            items: vec![], kv_tokens: 0 };
        charge_selected_slots(&mut device, &input, &working, "deepseek_v41", 3, 2752)?;
        assert_eq!(device.items[0].bytes, 3 * 810 * (1 << 20));
        assert_eq!(spark_slot_bytes("deepseek_v41", 2752), 208 * (1 << 20));
        assert_eq!(spark_slot_bytes("deepseek_v4", 2752), 208 * (1 << 20));
        Ok(())
    }

    #[test]
    fn native_slots_credit_only_the_existing_first_slot_residual() {
        for residual in [0, 100, 1000] {
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0, capacity_bytes: GIB,
                kv_tokens: 0, items: vec![Item::new(Category::Workspace, "expert waves", "", residual, Basis::Calibrated)] };
            charge_native_slots(&mut device, 3, 200);
            let charged = device.items.iter().filter(|item| item.group == "execution slots")
                .map(|item| item.bytes).sum::<u64>();
            assert_eq!(charged, 600 - residual.min(200));
            assert!(charged + residual >= 600);
        }
    }
}
