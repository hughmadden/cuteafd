//! Resident storage and measured recipes shared with native packaging.
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct StorageHold { hidden: usize, width: usize, stored: usize }
#[derive(Deserialize)]
struct SliceRecipe { hidden: usize, width: usize, sm: u32, capacities: Vec<u32>, slices: Vec<usize>,
    #[serde(default)] prefill: Vec<u32> }
#[derive(Deserialize)]
struct Ownership { intermediate: usize, world: usize, unit: usize }
#[derive(Deserialize)]
struct StreamPrefill { sm: u32, min_work: Option<u64> }
#[derive(Deserialize)]
struct AtomicSlices { hidden: usize, min_capacity: u32 }
#[derive(Deserialize)]
struct Layouts { schema: u32, decode_capacities: Vec<u32>, prefill_capacities: Vec<u32>, stream_prefill: Vec<StreamPrefill>, atomic_slices: Vec<AtomicSlices>, storage_holds: Vec<StorageHold>, native_slices: Vec<SliceRecipe>,
    ownership: Vec<Ownership>, generic_storage_holds: Vec<StorageHold> }

fn validate(data: &Layouts) -> Result<(), &'static str> {
    if data.schema != 1 { return Err("unsupported MXFP4 layout schema"); }
    if data.decode_capacities.first() != Some(&1)
        || !data.decode_capacities.windows(2).all(|p| p[0] < p[1])
        || data.decode_capacities.iter().any(|&n| n > 80)
        || data.prefill_capacities != [256, 1024, 4096] {
        return Err("invalid native MXFP4 capacity recipe");
    }
    let mut sms = std::collections::HashSet::new();
    for row in &data.stream_prefill {
        if ![120, 121].contains(&row.sm) || !sms.insert(row.sm) || row.min_work == Some(0) {
            return Err("invalid native MXFP4 STREAM policy");
        }
    }
    if sms.len() != 2 { return Err("native MXFP4 STREAM policy must name both hardware classes"); }
    let mut hidden = std::collections::HashSet::new();
    for row in &data.atomic_slices {
        if row.hidden == 0 || row.hidden % 256 != 0 || !hidden.insert(row.hidden)
            || ![256, 1024, 4096].contains(&row.min_capacity) {
            return Err("invalid native MXFP4 atomic policy");
        }
    }
    let mut keys = std::collections::HashSet::new();
    for row in &data.native_slices {
        if !keys.insert((row.hidden, row.width, row.sm))
            || row.prefill.iter().enumerate().any(|(i, c)| ![256, 1024, 4096].contains(c) || row.prefill[..i].contains(c))
            || (row.capacities.is_empty() && row.prefill.is_empty()) {
            return Err("invalid native MXFP4 prefill recipe");
        }
        if row.capacities.len() != row.slices.len()
            || row.capacities.iter().enumerate().any(|(i, capacity)| row.capacities[..i].contains(capacity)) {
            return Err("native MXFP4 capacities must pair uniquely with slices");
        }
        let decode: Vec<_> = row.capacities.iter().zip(&row.slices)
            .filter_map(|(&capacity, &slice)| (capacity <= 80).then_some(slice)).collect();
        if (!row.capacities.is_empty() && decode.is_empty()) || !decode.windows(2).all(|pair| pair[0] == pair[1]) {
            return Err("native MXFP4 decode capacities must share one slice recipe");
        }
        if row.slices.iter().any(|slice| ![32, 64, 128, 192].contains(slice)) {
            return Err("invalid native MXFP4 slice width");
        }
    }
    for row in &data.ownership {
        if ![32, 64, 128].contains(&row.unit) || row.intermediate % row.unit != 0
            || !(1..=8).contains(&row.world) || row.intermediate / row.unit < row.world {
            return Err("invalid measured MXFP4 ownership");
        }
    }
    Ok(())
}

fn layouts() -> &'static Layouts {
    static DATA: OnceLock<Layouts> = OnceLock::new();
    DATA.get_or_init(|| {
        let data: Layouts = serde_json::from_str(include_str!(
            "../../../../scripts/build/mxfp4-layouts.json")).expect("checked MXFP4 layouts");
        validate(&data).expect("checked MXFP4 layout policy");
        data
    })
}

/// Build-recipe fallback where coordinator images do not carry Spark manifests.
/// Exporters declare this same contract; runtime reads the installed declaration.
pub fn native_export_capacities(family: &str, role: &str) -> crate::spark_executions::ExpertExportCapacities {
    let mut decode = if family == "v41" && !matches!(role, "coordinator" | "dspark_tp2") {
        crate::V41_DECODE_CAPACITIES.map(|n| n as u32).to_vec()
    } else { layouts().decode_capacities.clone() };
    if family == "v41" && matches!(role, "coordinator" | "dspark_tp2") {
        decode.push(40);
        decode.sort_unstable();
    }
    let mut capacities = decode.clone();
    capacities.extend(&layouts().prefill_capacities);
    capacities.sort_unstable();
    crate::spark_executions::ExpertExportCapacities { capacities, decode }
}

/// Measured adjacent ownership; K32 remains the default scale-aligned unit.
pub fn ownership_unit(intermediate: usize, world: usize) -> usize {
    layouts().ownership.iter().find(|row| row.intermediate == intermediate && row.world == world)
        .map_or(32, |row| row.unit)
}

/// Generic packages may retain a padded extent without changing source ownership.
pub fn generic_stored_width(hidden: usize, width: usize) -> usize {
    layouts().generic_storage_holds.iter().find(|row| row.hidden == hidden && row.width == width)
        .map_or(width, |row| row.stored)
}

/// The same SM121 recipe the native exporter selects. Unmeasured capacities
/// retain their matching tile; measured decode recipes are fixed through C80.
pub fn native_slice_width(hidden: usize, width: usize, sm: u32, capacity: u32,
    matching: usize) -> usize {
    layouts().native_slices.iter().find(|row| row.hidden == hidden && row.width == width && row.sm == sm)
        .and_then(|row| row.capacities.iter().position(|&rows| rows == capacity).map(|index| row.slices[index]))
        .unwrap_or(matching)
}

/// Measured work per checkpoint expert, with a separate explicit hardware hold.
pub fn native_prefill_work(sm: u32, rows: u64, width: usize, experts: usize, topk: usize) -> bool {
    let Some(cutoff) = layouts().stream_prefill.iter().find(|row| row.sm == sm)
        .and_then(|row| row.min_work) else { return false };
    rows > 0 && width > 0 && experts > 0 && topk > 0
        && u128::from(rows) * topk as u128 * width as u128 >= u128::from(cutoff) * experts as u128
}

/// Static export route at the default admitted wave, not its padded capacity.
/// Custom larger waves retain the same correct, default-tuned implementation.
pub fn native_prefill(hidden: usize, width: usize, sm: u32, capacity: u32,
    experts: usize, topk: usize) -> bool {
    let rows = u64::from(capacity).min(crate::serving_capacity::default_prefill_unit_rows());
    native_prefill_work(sm, rows, width, experts, topk)
        && layouts().native_slices.iter().any(|row| row.hidden == hidden && row.width == width
            && row.sm == sm && row.prefill.contains(&capacity))
}

/// Describe the exact export-time route selection in planner output.
pub fn native_prefill_note(hidden: usize, widths: impl IntoIterator<Item = usize>,
    sm: u32, capacity: u32, experts: usize, topk: usize) -> String {
    let rows = u64::from(capacity).min(crate::serving_capacity::default_prefill_unit_rows());
    let routes = widths.into_iter().map(|width| {
        let route = if !native_prefill_work(sm, rows, width, experts, topk) {
            "slices (work gate/hold)"
        } else if !native_prefill(hidden, width, sm, capacity, experts, topk) {
            "slices (shape table)"
        } else { "STREAM" };
        format!("I{width}:{route}")
    }).collect::<Vec<_>>().join(", ");
    format!("native MXFP4 SM{sm} prefill C{capacity}, default-admitted {rows} rows: {routes}; custom waves retain this export-time route")
}

/// The seven STREAM planes and two native ABI per-expert trailing planes.
pub fn native_prefill_scratch_bytes(hidden: usize, width: usize, experts: usize,
    topk: usize, capacity: u32) -> Option<u64> {
    stored_width(hidden, width)?;
    if hidden % 512 != 0 || experts == 0 || topk == 0 || ![256, 1024, 4096].contains(&capacity) { return None; }
    let (h, w, e) = (hidden as u64, width as u64, experts as u64);
    let pairs = u64::from(capacity).checked_mul(topk as u64)?;
    let tiles = pairs.div_ceil(128).checked_add(e.min(pairs))?;
    let row = w.checked_add(w / 32)?.checked_add(15)?.checked_div(16)?.checked_mul(16)?;
    let meta = 4u64.checked_add(e.checked_mul(3)?)?.checked_add(tiles)?.checked_mul(4)?;
    [meta, pairs.checked_mul(4)?, pairs.checked_mul(4)?,
        tiles.checked_mul(128)?.checked_mul(row)?, pairs.checked_mul(4)?, pairs.checked_mul(4)?,
        pairs.checked_mul(h)?.checked_mul(4)?, e.checked_mul(4)?, e.checked_mul(4)?]
        .into_iter().try_fold(0u64, |offset, size| offset.checked_add(15)?
            .checked_div(16)?.checked_mul(16)?.checked_add(size))
}

/// Standard equal-role slice exports with atomic token output; adjacent widths
/// and all other geometries retain ordered routes. STREAM always stays ordered.
pub fn native_atomic_slices(hidden: usize, role: u32, capacity: u32) -> bool {
    matches!(role, 1 | 2 | 3 | 5 | 6 | 7 | 8) && layouts().atomic_slices.iter()
        .any(|row| row.hidden == hidden && capacity >= row.min_capacity)
}

/// `export_b12x_slices_aot.py`'s 16-byte-aligned scratch planes, excluding
/// shared owner I/O. Qualified prefill uses original-route STREAM planes;
/// unqualified geometries retain their existing native slice recipe.
pub fn native_scratch_bytes(hidden: usize, experts: usize, topk: usize,
    width: usize, role: u32, capacity: u32, coded: bool) -> Option<u64> {
    native_scratch_bytes_for_sm(hidden, experts, topk, width, role, capacity, coded, 121)
}

/// Exact static export census for either native hardware class, excluding I/O.
pub fn native_scratch_bytes_for_sm(hidden: usize, experts: usize, topk: usize,
    width: usize, role: u32, capacity: u32, coded: bool, sm: u32) -> Option<u64> {
    stored_width(hidden, width)?;
    if experts == 0 || topk == 0 || ![1, 16, crate::V41_VERIFY_ROWS as u32, 80, 256, 1024, 4096].contains(&capacity) { return None; }
    if native_prefill(hidden, width, sm, capacity, experts, topk) {
        return native_prefill_scratch_bytes(hidden, width, experts, topk, capacity);
    }
    let matching = if coded && (crate::tp_units::native_role_width(role).is_none() || width % 192 == 0) {
        192
    } else { 128 };
    let slice = if sm == 121 { native_slice_width(hidden, width, sm, capacity, matching) } else { matching } as u64;
    let atomic = native_atomic_slices(hidden, role, capacity);
    let (hidden, experts, routes) = (hidden as u64, experts as u64,
        u64::from(capacity).checked_mul(topk as u64)?);
    // FC1 activation storage is independent of the weights' scale encoding.
    let partial_cols = slice / 4 + slice / 32;
    let partial = if atomic { 4 } else {
        (width as u64).div_ceil(slice).checked_mul(routes)?.checked_mul(partial_cols)?.checked_mul(4)?
    };
    let output = (if atomic { u64::from(capacity) } else { routes }).checked_mul(hidden)?.checked_mul(4)?;
    [4, experts.checked_mul(routes)?.checked_mul(4)?, experts.checked_mul(4)?,
        experts.checked_mul(8)?, routes.checked_mul(76)?, routes.checked_mul(4)?,
        routes.checked_mul(4)?, partial, output, experts.checked_mul(4)?, experts.checked_mul(4)?]
        .into_iter().try_fold(0u64, |offset, size| offset.checked_add(15)?
            .checked_div(16)?.checked_mul(16)?.checked_add(size))
}

/// Storage decisions depend on matrix geometry, not a family-specific switch.
/// Holds retain the measured layout pending an explicit qualification decision.
pub fn stored_width(hidden: usize, width: usize) -> Option<usize> {
    if hidden == 0 || hidden % 256 != 0 || width == 0 || width % 32 != 0 || width > 8192 {
        return None;
    }
    Some(layouts().storage_holds.iter().find(|row| row.hidden == hidden && row.width == width)
        .map_or(width, |row| row.stored))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scratch_matches_exporter_plane_offsets_and_capacity_recipe() {
        // Independent exporter specs: live/count/prefix/metadata/group/inverse,
        // route partials, output, and two trailing per-expert planes.
        let planes = |hidden: u64, experts: u64, topk: u64, width: u64,
            tile: u64, capacity: u64, atomic: bool| {
            let routes = capacity * topk;
            // One byte/activation plus one four-byte scale word per K32 block.
            let partial = if atomic { 4 } else {
                width.div_ceil(tile) * routes * (tile + tile / 8)
            };
            let mut end = 0u64;
            for size in [4, experts * routes * 4, experts * 4, experts * 8,
                routes * 19 * 4, routes * 4, routes * 4, partial,
                if atomic { capacity } else { routes } * hidden * 4, experts * 4, experts * 4] {
                end = end.div_ceil(16) * 16 + size;
            }
            end
        };
        for (hidden, experts, topk, width, role, coded, tiles) in [
            (5120, 384, 6, 576, 1, true, [128, 128, 128, 192, 192, 192]),
            (4096, 256, 6, 352, 65888, false, [32, 32, 32, 128, 128, 128]),
            (5120, 384, 6, 288, 8, true, [128, 128, 128, 192, 192, 192]),
            (5120, 384, 6, 512, 66048, true, [128; 6]),
            (5120, 384, 6, 448, 65984, true, [128; 6]),
            (5120, 384, 6, 352, 65888, true, [128; 6]),
            (4096, 256, 6, 512, 1, false, [128; 6]),
            (4096, 256, 6, 704, 66240, false, [128; 6]),
            (7168, 384, 6, 608, 66144, false, [128; 6]),
        ] {
            for (capacity, tile) in [1, 16, 64, 80, 256, 1024, 4096].into_iter()
                .zip([tiles[0], tiles[1], tiles[2], tiles[2], tiles[3], tiles[4], tiles[5]]) {
                // Only V4.1 hidden 5120 inherits m80's recipe at m64; not a tuned m64 result.
                // Follow-up: screen E384/top6 on kakapo/maxq.
                if capacity == 64 && hidden != 5120 { continue; }
                let atomic = hidden == 7168 && role < 65536 && capacity >= 256;
                let expected = if native_prefill(hidden as usize, width as usize, 121, capacity as u32, experts as usize, topk as usize) {
                    native_prefill_scratch_bytes(hidden as usize, width as usize, experts as usize,
                        topk as usize, capacity as u32).unwrap()
                } else { planes(hidden, experts, topk, width, tile, capacity, atomic) };
                assert_eq!(native_scratch_bytes(hidden as usize, experts as usize, topk as usize,
                    width as usize, role, capacity as u32, coded), Some(expected));
            }
        }
        for (hidden, experts, topk, width, capacity) in [
            (255, 384, 6, 288, 1), (5120, 0, 6, 288, 1),
            (5120, 384, 0, 288, 1), (5120, 384, 6, 31, 1), (5120, 384, 6, 288, 65),
        ] {
            assert_eq!(native_scratch_bytes(hidden, experts, topk, width, 8, capacity, true), None);
        }
        assert_eq!(native_scratch_bytes(5120, usize::MAX, 6, 288, 8, 4096, true), None);
    }

    #[test]
    fn prefill_route_uses_checkpoint_work_and_default_live_wave() {
        assert_eq!(crate::serving_capacity::default_prefill_unit_rows(), 2752);
        for (rows, width, experts) in [(4096, 384, 384), (3072, 512, 384), (2048, 512, 256)] {
            assert!(!native_prefill_work(121, rows - 1, width, experts, 6));
            assert!(native_prefill_work(121, rows, width, experts, 6));
            assert!(native_prefill_work(121, rows + 1, width, experts, 6));
            assert!(!native_prefill_work(120, rows + 1, width, experts, 6));
        }
        for (hidden, experts, width, stream) in [(5120, 384, 384, false),
            (4096, 256, 512, true), (7168, 384, 768, true), (7168, 384, 608, false)] {
            for capacity in [256, 1024, 4096] {
                assert_eq!(native_prefill(hidden, width, 121, capacity, experts, 6), stream && capacity == 4096);
                assert!(!native_prefill(hidden, width, 120, capacity, experts, 6));
            }
        }
        assert!(native_prefill_note(5120, [384], 121, 4096, 384, 6).contains("I384:slices"));
        assert!(native_prefill_note(4096, [512], 121, 4096, 256, 6).contains("I512:STREAM"));
        for (width, stream) in [(384, false), (576, true), (768, false), (1152, true)] {
            assert_eq!(native_prefill(5120, width, 121, 4096, 384, 6), stream);
        }
        assert!(native_prefill_note(5120, [384], 121, 4096, 384, 6).contains("slices (work gate/hold)"));
        assert!(native_prefill_note(5120, [768], 121, 4096, 384, 6).contains("slices (shape table)"));
    }

    #[test]
    fn every_prefill_row_is_selected_by_a_shipped_artifact() {
        for row in layouts().native_slices.iter().filter(|row| !row.prefill.is_empty()) {
            let experts = match row.hidden { 4096 => 256, 5120 | 7168 => 384, _ => panic!("unshipped geometry") };
            let intermediate = match row.hidden { 4096 => 2048, 5120 => 2304, _ => 3072 };
            assert!((1..=8).any(|world| crate::tp_units::package_width_keys(
                intermediate, 32, world, false, true, false).iter().any(|&(_, width)| width == row.width)),
                "unshipped width H{}/I{}", row.hidden, row.width);
            assert!(row.prefill.iter().any(|&capacity| native_prefill(
                row.hidden, row.width, row.sm, capacity, experts, 6)),
                "dead prefill row H{}/I{}/SM{}", row.hidden, row.width, row.sm);
        }
    }

    #[test]
    fn policy_rejects_invalid_stream_and_atomic_sections() {
        let bundled = include_str!("../../../../scripts/build/mxfp4-layouts.json");
        for failure in 0..8 {
            let mut data: Layouts = serde_json::from_str(bundled).unwrap();
            match failure {
                0 => data.stream_prefill[1].sm = 120,
                1 => data.stream_prefill[1].sm = 119,
                2 => data.stream_prefill[1].min_work = Some(0),
                3 => { data.stream_prefill.pop(); }
                4 => data.atomic_slices.push(AtomicSlices { hidden: 7168, min_capacity: 256 }),
                5 => data.atomic_slices[0].hidden = 0,
                6 => data.atomic_slices[0].hidden = 7169,
                _ => data.atomic_slices[0].min_capacity = 80,
            }
            assert!(validate(&data).is_err());
        }
    }

    #[test]
    fn stream_scratch_matches_independent_plane_census() {
        for row in &layouts().native_slices {
            for &capacity in &row.prefill {
                let (h, w, e, pairs) = (row.hidden as u64, row.width as u64,
                    if row.hidden == 4096 { 256u64 } else { 384u64 }, u64::from(capacity) * 6);
                let chunks = pairs.div_ceil(128) + e.min(pairs);
                let mut expected = 0u64;
                for size in [(4 + 3 * e + chunks) * 4, pairs * 4, pairs * 4,
                    chunks * 128 * (w + w / 32).div_ceil(16) * 16, pairs * 4, pairs * 4,
                    pairs * h * 4, e * 4, e * 4] {
                    expected = expected.div_ceil(16) * 16 + size;
                }
                assert_eq!(native_prefill_scratch_bytes(row.hidden, row.width, e as usize, 6, capacity), Some(expected));
                for decode in [1, 16, 80] { assert!(!native_prefill(row.hidden, row.width, row.sm, decode, e as usize, 6)); }
            }
        }
        assert!(!native_prefill(7168, 768, 121, 256, 384, 6));
        assert!(!native_prefill(5120, 288, 120, 256, 384, 6));
    }

    #[test]
    fn policy_rejects_capacity_dependent_decode_tiling_and_invalid_tiles() {
        let bundled = include_str!("../../../../scripts/build/mxfp4-layouts.json");
        for failure in 0..4 {
            let mut data: Layouts = serde_json::from_str(bundled).unwrap();
            match failure {
                0 => data.native_slices[0].slices[1] = 64,
                1 => data.native_slices[0].slices[2] = 256,
                2 => data.native_slices[0].capacities[1] = 1,
                _ => { data.native_slices[0].slices.pop(); }
            }
            assert!(validate(&data).is_err());
        }
        assert_eq!(native_slice_width(5120, 288, 120, 1, 192), 192);
        assert_eq!(native_slice_width(5120, 288, 121, 1, 192), 128);
        assert_eq!(native_slice_width(5120, 288, 121, 256, 192), 192);
        assert!(!native_prefill(5120, 288, 121, 256, 384, 6));
    }

    #[test]
    fn geometry_selects_exact_storage_or_an_explicit_measurement_hold() {
        assert_eq!(stored_width(5120, 576), Some(576));
        assert_eq!(ownership_unit(2304, 5), 64);
        assert_eq!(ownership_unit(2304, 7), 32);
        for width in [320, 352, 1024, 2048] {
            assert_eq!(generic_stored_width(6144, width), width);
        }
        assert_eq!(generic_stored_width(4096, 352), 352);
        for width in [288, 320, 352, 448, 480, 768, 1152, 2304] {
            assert_eq!(stored_width(5120, width), Some(width));
        }
        for (hidden, widths) in [(4096, vec![288, 320, 352, 416, 672, 704]),
                                (7168, vec![416, 448, 608])] {
            for width in widths {
                assert_eq!(stored_width(hidden, width), Some(width));
            }
        }
        for (hidden, width) in [(0, 288), (255, 288), (5120, 0), (5120, 31), (5120, 8224)] {
            assert_eq!(stored_width(hidden, width), None);
        }
    }
}
