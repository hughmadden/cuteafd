//! Programs belonging to one coordinator, including unsplit draft/target work.

// Literal registry consumed by release-common.sh as well as the serving owners.
pub const V41_PREFILL_MIN_ROWS: u32 = 80;
pub const V41_PREFILL_MAX_ROWS: u32 = 4096;
pub const V41_LIVE_MIN_ROWS: u32 = 256;
pub const V41_DECODER_REPLAY_ROWS: u32 = 128;
pub const V41_SERVING_AOT_ROWS: [u32; 3] = [256, 1024, 4096];

pub fn v41_live_rows(chunk: u32) -> Option<u32> {
    (V41_PREFILL_MIN_ROWS..=V41_PREFILL_MAX_ROWS).contains(&chunk)
        .then_some(chunk.max(V41_LIVE_MIN_ROWS).max(V41_DECODER_REPLAY_ROWS))
}

pub fn v41_aot_rows(chunk: u32) -> Option<u32> {
    let live = v41_live_rows(chunk)?;
    V41_SERVING_AOT_ROWS.into_iter().find(|&rows| rows >= live)
}
#[derive(Debug, Clone, Copy)]
pub struct CoordinatorPrograms<'a> {
    pub family: &'a str,
    pub split_family: Option<&'a str>,
}

impl CoordinatorPrograms<'_> {
    /// Namespace boundaries matter: `glmf` must not select `glmf2` or `glm`.
    pub fn contains(self, name: &str) -> bool {
        name.split_once('_').is_some_and(|(family, _)| {
            family == self.family || self.split_family == Some(family)
        })
    }

    /// Shared arenas exclude index top-k, which owns a separate persistent buffer.
    pub fn shared_scratch<'a>(self, sizes: impl IntoIterator<Item = (&'a str, u64)>) -> u64 {
        sizes.into_iter().filter(|(name, _)| self.contains(name) && !name.contains("index_topk"))
            .map(|(_, bytes)| bytes).max().unwrap_or(0)
    }

    pub fn validate_v4(self, prefill_rows: u64, decode_rows: u64,
        available: impl IntoIterator<Item = impl AsRef<str>>) -> Result<(), MissingCoordinatorProgram> {
        let available: std::collections::HashSet<String> = available.into_iter()
            .map(|name| name.as_ref().to_string()).collect();
        for name in self.v4_required(prefill_rows, decode_rows) {
            if !self.contains(&name) || !available.contains(&name) {
                return Err(MissingCoordinatorProgram(name));
            }
        }
        Ok(())
    }

    /// All V4 dispatch routes, including the unsplit dSpark target in split mode.
    /// Validate this list before weights, arenas or graph captures are allocated.
    pub fn v4_required(self, prefill_rows: u64, decode_rows: u64) -> Vec<String> {
        let mut names = Vec::new();
        let mut add = |family: &str, suffix: String| names.push(format!("{family}_{suffix}"));
        for suffix in ["mhc_pre", "mhc_post", "mhc_head", "router_scores", "expert_input_quant",
            "block_fp8_scale_prep"] {
            add(self.family, suffix.to_string());
        }
        for ratio in [4, 128] {
            for mode in ["decode", "prefill", "continuation"] {
                add(self.family, format!("compressor_{mode}_c{ratio}"));
            }
        }
        for (mode, rows) in [("decode", decode_rows), ("prefill", prefill_rows)] {
            for suffix in [format!("mhc_post_pre_m{rows}"), format!("index_producer_m{rows}"),
                format!("index_topk_{mode}_m{rows}")] {
                add(self.family, suffix);
            }
            for family in std::iter::once(self.family).chain(self.split_family) {
                for suffix in ["producer", "wo", "shared_ffn"] {
                    add(family, format!("{suffix}_m{rows}"));
                }
                for attention in ["win", "c4", "c128"] {
                    add(family, format!("sparse_mla_{mode}_{attention}_m{rows}"));
                }
            }
        }
        names
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingCoordinatorProgram(pub String);

impl std::fmt::Display for MissingCoordinatorProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "V4 startup requires selected program {}", self.0)
    }
}

impl std::error::Error for MissingCoordinatorProgram {}

/// Select the same 170/188-SM program recipe as native load, before admission.
/// The retained plans allow another device to select its own scratch envelope.
pub fn select_card_plans(manifest: &mut serde_json::Value, sms: u32) -> Result<(), CardPlanError> {
    let Some(programs) = manifest["programs"].as_array_mut() else { return Ok(()) };
    for program in programs {
        if program.get("card_plans").is_none() { continue; }
        let name = program["name"].as_str().unwrap_or("<unnamed>").to_string();
        let invalid = || CardPlanError { name: name.clone() };
        let plans = program["card_plans"].as_array().ok_or_else(invalid)?;
        if sms == 0 || plans.len() != 2 { return Err(invalid()); }
        let selected_sms = if sms >= 188 { 188 } else { 170 };
        let mut selected = None;
        for expected in [170, 188] {
            let matching: Vec<_> = plans.iter().filter(|p| p["sm_count"].as_u64() == Some(expected)).collect();
            if matching.len() != 1 { return Err(invalid()); }
            let plan = matching[0];
            let scratch = plan["scratch_bytes_at_capacity"].as_object().ok_or_else(invalid)?;
            if scratch.values().any(|v| v.as_u64().is_none()) || !plan["geometry"].is_object() {
                return Err(invalid());
            }
            if expected == selected_sms { selected = Some(plan.clone()); }
        }
        let plan = selected.ok_or_else(invalid)?;
        program["scratch_bytes_at_capacity"] = plan["scratch_bytes_at_capacity"].clone();
        program["geometry"] = plan["geometry"].clone();
    }
    Ok(())
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("program {name} has an invalid 170/188-SM card-plan contract")]
pub struct CardPlanError {
    pub name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dual_manifest() -> serde_json::Value {
        serde_json::json!({"programs": [{"name": "dsv4p_sparse_mla_decode_c128_m64",
            "scratch_bytes_at_capacity": {"scratch": 1094778880}, "geometry": {},
            "card_plans": [
                {"sm_count": 170, "scratch_bytes_at_capacity": {"scratch": 1094778880}, "geometry": {"num_splits": 130}},
                {"sm_count": 188, "scratch_bytes_at_capacity": {"scratch": 8421376}, "geometry": {"num_splits": 1}}
            ]}, {"name": "dsv4p_wo", "scratch_bytes_at_capacity": {"scratch": 9}}]})
    }

    #[test]
    fn card_plan_scratch_tracks_native_selection_without_losing_the_other_plan() {
        let mut manifest = dual_manifest();
        for (sms, bytes, splits) in [(188, 8421376, 1), (170, 1094778880, 130), (188, 8421376, 1)] {
            select_card_plans(&mut manifest, sms).unwrap();
            assert_eq!(manifest["programs"][0]["scratch_bytes_at_capacity"]["scratch"], bytes);
            assert_eq!(manifest["programs"][0]["geometry"]["num_splits"], splits);
            assert_eq!(manifest["programs"][1]["scratch_bytes_at_capacity"]["scratch"], 9);
        }
    }

    #[test]
    fn malformed_card_plans_fail_closed_instead_of_using_the_capacity_envelope() {
        for defect in 0..5 {
            let mut manifest = dual_manifest();
            match defect {
                0 => manifest["programs"][0]["card_plans"] = serde_json::json!([]),
                1 => manifest["programs"][0]["card_plans"][1]["sm_count"] = 170.into(),
                2 => manifest["programs"][0]["card_plans"][1]["scratch_bytes_at_capacity"]["scratch"] = (-1).into(),
                3 => manifest["programs"][0]["card_plans"][1]["geometry"] = serde_json::Value::Null,
                _ => {},
            }
            let error = select_card_plans(&mut manifest, if defect == 4 { 0 } else { 188 }).unwrap_err();
            assert_eq!(error.name, "dsv4p_sparse_mla_decode_c128_m64");
        }
    }

    #[test]
    fn v41_serving_rows_cover_live_chunks_and_decoder_replay() {
        for chunk in V41_PREFILL_MIN_ROWS..=V41_PREFILL_MAX_ROWS {
            let live = v41_live_rows(chunk).unwrap();
            let aot = v41_aot_rows(chunk).unwrap();
            assert!(live >= chunk && live >= V41_DECODER_REPLAY_ROWS && live >= V41_LIVE_MIN_ROWS);
            assert!(aot >= live && V41_SERVING_AOT_ROWS.contains(&aot));
        }
        for invalid in [0, 79, 4097, u32::MAX] {
            assert_eq!(v41_live_rows(invalid), None);
            assert_eq!(v41_aot_rows(invalid), None);
        }
    }

    #[test]
    fn namespace_selection_keeps_only_serving_and_split_families() {
        let families = ["dsv4f", "dsv4p", "glm", "glmf", "mimo", "mimof", "mimop", "qwen4"];
        for family in families {
            let split_family = format!("{family}2");
            for split in [false, true] {
                let selected = CoordinatorPrograms { family, split_family: split.then_some(split_family.as_str()) };
                assert!(selected.contains(&format!("{family}_target")));
                assert_eq!(selected.contains(&format!("{family}2_target")), split);
                for other in families.into_iter().filter(|other| *other != family) {
                    assert!(!selected.contains(&format!("{other}_target")));
                    assert!(!selected.contains(&format!("{other}2_target")));
                }
                assert!(!selected.contains(family));
            }
        }
    }

    #[test]
    fn split_v4_keeps_unsplit_draft_programs() {
        for family in ["dsv4f", "dsv4p"] {
            let split_family = format!("{family}2");
            let selected = CoordinatorPrograms { family, split_family: Some(&split_family) };
            let names = selected.v4_required(4096, 64);
            assert!(names.iter().all(|name| selected.contains(name)));
            for suffix in ["producer_m64", "producer_m4096", "wo_m64", "sparse_mla_decode_win_m64"] {
                assert!(names.contains(&format!("{family}_{suffix}")));
                assert!(names.contains(&format!("{split_family}_{suffix}")));
            }
        }
    }
}
