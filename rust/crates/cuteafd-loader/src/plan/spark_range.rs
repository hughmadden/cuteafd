//! Checkpoint-derived Spark count policy, independent of export installation.
use super::{Component, ExpertPlacement, PlanError, PlanInput, PlanOptions, PlanReport};
use cuteafd_core::tp_units::UnitRule;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

pub use cuteafd_core::MAX_SPARK_WORLD as MAX_SPARKS;
pub const EXPERT_BYTES_PER_SPARK: u64 = 16 << 30;
pub const MIN_UNITS_PER_SPARK: usize = 2;

/// Existing releases are a compatibility floor, not a second range policy.
pub fn shipped_worlds(package: &str) -> Vec<usize> {
    static FLOOR: OnceLock<BTreeMap<String, Vec<usize>>> = OnceLock::new();
    let floor = FLOOR.get_or_init(|| serde_json::from_str(include_str!(
        "../../../../../scripts/build/spark-release-floor.json")).expect("checked release floor"));
    let key = package.split_whitespace().next().unwrap_or(package);
    let Some((family, format)) = key.split_once(':') else { return vec![]; };
    let format = if format.starts_with("exl3") { "exl3" } else { format };
    floor.get(&format!("{family}:{format}")).or_else(|| floor.get(&format!("{family}:*")))
        .cloned().unwrap_or_default()
}
pub const MIN_SIZE_CAP: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingCap { Size, Width, Cluster, Minimum, Shipped }

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SparkRange {
    pub minimum: usize,
    pub maximum: usize,
    pub size_cap: usize,
    pub width_cap: usize,
    pub formula_maximum: usize,
    pub shipped_maximum: usize,
    /// All binding caps are retained when the limits tie.
    pub binding: Vec<BindingCap>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, thiserror::Error)]
pub enum RangeRefusal {
    #[error("unsupported Spark count {requested}: supported range {minimum}..={maximum}, binding cap {binding:?}")]
    Unsupported { requested: usize, minimum: usize, maximum: usize, binding: Vec<BindingCap> },
    #[error("invalid Spark range geometry: {0}")]
    Geometry(String),
}

impl SparkRange {
    pub fn new(minimum: usize, routed_bytes: u64, intermediate: usize, unit_columns: usize,
        shipped_maximum: usize) -> Result<Self, RangeRefusal> {
        if minimum > MAX_SPARKS || shipped_maximum > MAX_SPARKS || unit_columns == 0
            || intermediate == 0 || intermediate % unit_columns != 0 {
            return Err(RangeRefusal::Geometry("invalid minimum, shipped ceiling or recipe units".into()));
        }
        let size_cap = ((routed_bytes / EXPERT_BYTES_PER_SPARK) as usize).clamp(MIN_SIZE_CAP, MAX_SPARKS);
        let width_cap = intermediate / unit_columns / MIN_UNITS_PER_SPARK;
        let formula_maximum = MAX_SPARKS.min(size_cap).min(width_cap);
        let maximum = minimum.max(shipped_maximum).max(formula_maximum);
        let mut binding = Vec::new();
        if minimum == maximum && minimum > formula_maximum { binding.push(BindingCap::Minimum); }
        if shipped_maximum == maximum && shipped_maximum > formula_maximum { binding.push(BindingCap::Shipped); }
        if maximum == formula_maximum {
            if size_cap == maximum { binding.push(BindingCap::Size); }
            if width_cap == maximum { binding.push(BindingCap::Width); }
            if MAX_SPARKS == maximum { binding.push(BindingCap::Cluster); }
        }
        Ok(Self { minimum, maximum, size_cap, width_cap, formula_maximum, shipped_maximum, binding })
    }

    pub fn admit(&self, requested: usize) -> Result<(), RangeRefusal> {
        if (self.minimum..=self.maximum).contains(&requested) { return Ok(()); }
        Err(RangeRefusal::Unsupported { requested, minimum: self.minimum, maximum: self.maximum,
            binding: self.binding.clone() })
    }

    pub fn counts(&self) -> std::ops::RangeInclusive<usize> { self.minimum..=self.maximum }

    pub fn width_classes(&self, intermediate: usize, unit_columns: usize) -> Result<Vec<usize>, RangeRefusal> {
        let mut classes = BTreeSet::new();
        for world in self.counts().filter(|world| *world > 0) {
            classes.extend(width_classes(intermediate, unit_columns, world)?);
        }
        Ok(classes.into_iter().collect())
    }
}

/// Classes are in units, not columns. Rotated experts can use both on one rank.
pub fn width_classes(intermediate: usize, unit_columns: usize, world: usize) -> Result<Vec<usize>, RangeRefusal> {
    if unit_columns == 0 || intermediate == 0 || intermediate % unit_columns != 0
        || world == 0 || world > MAX_SPARKS || intermediate / unit_columns < world {
        return Err(RangeRefusal::Geometry(format!("intermediate {intermediate}, unit {unit_columns}, world {world}")));
    }
    let units = intermediate / unit_columns;
    let mut classes = vec![units / world];
    if units % world != 0 { classes.push(units.div_ceil(world)); }
    Ok(classes)
}

pub fn balanced_rule(intermediate: usize, unit_columns: usize, world: usize) -> UnitRule {
    if intermediate / unit_columns % world == 0 { UnitRule::Contiguous }
    else if unit_columns == 32 { UnitRule::LayerRotation }
    else { UnitRule::Rotated }
}

#[derive(Debug, Clone, Serialize)]
pub struct CountLayout {
    pub sparks: usize,
    pub units_per_expert: Vec<usize>,
    pub unit_columns: usize,
    /// Source partition granularity for padded storage (not kernel tile width).
    pub source_block_columns: usize,
    pub rule: UnitRule,
    pub expert_bytes_per_rank: Option<Vec<u64>>,
    /// Installed recipe widths take precedence over prospective balanced widths.
    pub resolved_width_columns: Option<Vec<usize>>,
    pub padded: bool,
    pub shipped_exception: bool,
    /// Current export coverage is separate from the requested support policy.
    pub currently_launchable: bool,
    pub executable_blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColumnRange {
    pub package: String,
    pub rtx_gpus: usize,
    pub routed_expert_bytes: u64,
    pub intermediate: usize,
    pub unit_columns: usize,
    /// None means that no complete, capacity-safe geometric witness exists.
    pub range: Option<SparkRange>,
    pub executable_minimum: Option<usize>,
    pub geometric_witness_rtx_gpus: Option<usize>,
    /// Formula and compatibility ceiling remain inspectable if N_min is unresolved.
    pub policy: SparkRange,
    /// Column widths remain unambiguous when recipes use different unit sizes.
    pub width_classes_columns: Vec<usize>,
    pub width_classes: Vec<usize>,
    pub counts: Vec<CountLayout>,
}

fn geometry(report: &PlanReport) -> Option<(u64, usize, usize)> {
    let moe = report.spec.as_ref()?.moe.as_ref()?;
    let routed = report.components.iter().filter(|component| component.component == Component::RoutedExpert)
        .map(|component| component.bytes).sum();
    let contract = report.experts.as_ref()?;
    let recipe = (1..=MAX_SPARKS).find_map(|world|
        super::experts::spark_recipe(&contract.package, moe.intermediate, world).ok())?;
    let unit = match recipe.slicing {
        crate::formats::fp8_experts::Slicing::Blocks(unit) => unit,
        // Padded recipes use stored_slice's 128-column storage alignment,
        // not the checkpoint's quantization scale-group size.
        crate::formats::fp8_experts::Slicing::Padded => 128,
    };
    Some((routed, moe.intermediate, unit))
}

/// Missing recipes may refuse execution after constructing a complete ledger.
/// Never treat an adapter's empty/partial failure ledger as proof of fit.
fn geometric_fit(report: &PlanReport) -> bool {
    use cuteafd_core::memory_layout::{Category, DeviceKind};
    let Some(memory) = &report.memory_layout else { return false; };
    if !report.fits || report.config_error.is_some() || !report.unclassified.is_empty()
        || !report.missing_shards.is_empty() || report.components.iter().any(|c| c.rejected > 0)
        || memory.devices.iter().any(|device| device.used_bytes() > device.capacity_bytes) {
        return false;
    }
    let expert_kind = if report.spark_ranks == 0 { DeviceKind::Rtx } else { DeviceKind::Spark };
    let experts = memory.devices.iter().filter(|device| device.kind == expert_kind)
        .flat_map(|device| &device.items).filter(|item| item.category == Category::Experts)
        .map(|item| item.bytes).sum::<u64>();
    // The solver's selected placement proves all expert layers are assigned;
    // wip_launchable also qualifies legacy complete-ledger adapters.
    experts > 0 && (report.solver_placement.is_some() || report.wip_launchable())
}

impl PlanInput {
    /// Open headers once, then ask the same complete planner for both card columns.
    /// No GPU queries, tensor payloads, or package-coverage assumptions are made.
    pub fn spark_ranges(&self, options: &PlanOptions) -> Result<Vec<ColumnRange>, PlanError> {
        let mut columns = Vec::new();
        let mut one_rtx_minimum = None;
        for rtx_gpus in [1, 2] {
            let mut options = options.clone();
            let layout = options.layout.get_or_insert_with(Default::default);
            let budget = layout.rtx_bytes.first().copied().unwrap_or(crate::placement::inventory::PRO_TOTAL_BYTES);
            layout.rtx_bytes = vec![budget; rtx_gpus];
            let reports = (0..=MAX_SPARKS).map(|sparks| {
                options.placement = ExpertPlacement::from_spark_ranks(sparks);
                super::plan_unrestricted(self, &options)
            }).collect::<Result<Vec<_>, _>>()?;
            let Some((routed, intermediate, unit_columns)) = reports.iter().find_map(geometry) else { continue; };
            let executable_minimum = reports.iter().position(PlanReport::wip_launchable);
            let own_minimum = reports.iter().position(geometric_fit);
            let (minimum, geometric_witness_rtx_gpus) = match (own_minimum, one_rtx_minimum) {
                (Some(own), Some(single)) if single < own => (Some(single), Some(1)),
                (Some(own), _) => (Some(own), Some(rtx_gpus)),
                (None, Some(single)) => (Some(single), Some(1)),
                (None, None) => (None, None),
            };
            if rtx_gpus == 1 { one_rtx_minimum = minimum; }
            let package = &reports.iter().find(|report| report.experts.is_some()).unwrap().experts.as_ref().unwrap().package;
            let shipped_maximum = shipped_worlds(package).into_iter().max().unwrap_or(0);
            let policy = SparkRange::new(0, routed, intermediate, unit_columns, shipped_maximum)?;
            let range = minimum.map(|minimum| SparkRange::new(minimum, routed, intermediate,
                unit_columns, shipped_maximum)).transpose()?;
            let mut counts = Vec::new();
            {
                let range = range.as_ref().unwrap_or(&policy);
                for sparks in range.counts() {
                    let recipe = super::experts::spark_recipe(package, intermediate, sparks).ok();
                    let padded = recipe.as_ref().is_some_and(|recipe|
                        recipe.slicing == crate::formats::fp8_experts::Slicing::Padded);
                    let count_unit = recipe.as_ref().map(|recipe| match recipe.slicing {
                        crate::formats::fp8_experts::Slicing::Blocks(unit) => unit,
                        crate::formats::fp8_experts::Slicing::Padded => unit_columns,
                    }).unwrap_or(unit_columns);
                    let source_block_columns = if padded { reports[sparks].experts.as_ref().unwrap().block } else { count_unit };
                    let rule = if padded || sparks == 0 { UnitRule::Contiguous }
                        else { balanced_rule(intermediate, count_unit, sparks) };
                    let classes = if sparks == 0 { vec![] } else if padded {
                        let mut classes = recipe.as_ref().unwrap().widths.iter().map(|width| width / count_unit).collect::<Vec<_>>();
                        classes.sort_unstable(); classes.dedup(); classes
                    } else { width_classes(intermediate, count_unit, sparks)? };
                    let expert_bytes_per_rank = if sparks == 0 { Some(vec![]) } else {
                        // The catalog's residency knows mixed EXL3 tiers and scales;
                        // report its actual expert ledger, not a padded byte estimate.
                        reports[sparks].memory_layout.as_ref().map(|memory| memory.devices.iter()
                            .filter(|device| device.kind == cuteafd_core::memory_layout::DeviceKind::Spark)
                            .map(|device| device.items.iter().filter(|item| item.category == cuteafd_core::memory_layout::Category::Experts)
                                .map(|item| item.bytes).sum()).collect::<Vec<_>>())
                            .filter(|bytes| bytes.len() == sparks)
                    };
                    counts.push(CountLayout { sparks, units_per_expert: classes, unit_columns: count_unit,
                        source_block_columns, rule, expert_bytes_per_rank,
                        resolved_width_columns: recipe.map(|recipe| recipe.widths), padded,
                        shipped_exception: sparks > range.formula_maximum && sparks <= shipped_maximum,
                        currently_launchable: reports[sparks].wip_launchable(),
                        executable_blockers: if reports[sparks].wip_launchable() { vec![] }
                            else { reports[sparks].hints.iter().map(|hint| hint.what.clone()).collect() } });
                }
            }
            let classes = counts.iter().filter(|count| count.unit_columns == unit_columns)
                .flat_map(|count| count.units_per_expert.iter().copied())
                .collect::<BTreeSet<_>>().into_iter().collect();
            let width_classes_columns = counts.iter().flat_map(|count| count.units_per_expert.iter()
                .map(|units| units * count.unit_columns)).collect::<BTreeSet<_>>().into_iter().collect();
            columns.push(ColumnRange { package: package.clone(), rtx_gpus, routed_expert_bytes: routed, intermediate, unit_columns,
                range, executable_minimum, geometric_witness_rtx_gpus, policy,
                width_classes_columns, width_classes: classes, counts });
        }
        Ok(columns)
    }
}

/// Only a request above the policy ceiling needs to search for the minimum
/// override. Ordinary admission keeps its one-plan readiness cost.
pub(super) fn admit(input: &PlanInput, options: &PlanOptions, report: &PlanReport) -> Result<(), PlanError> {
    let Some((routed, intermediate, unit)) = geometry(report) else { return Ok(()); };
    let shipped = shipped_worlds(&report.experts.as_ref().unwrap().package).into_iter().max().unwrap_or(0);
    let requested = options.placement.spark_ranks();
    let policy = SparkRange::new(0, routed, intermediate, unit, shipped)?;
    if requested <= policy.maximum { return Ok(()); }
    let mut candidate = options.clone();
    let mut minimum = None;
    for sparks in 0..=MAX_SPARKS {
        candidate.placement = ExpertPlacement::from_spark_ranks(sparks);
        let fits = if sparks == requested { geometric_fit(report) }
            else { geometric_fit(&super::plan_unrestricted(input, &candidate)?) };
        if fits { minimum = Some(sparks); break; }
    }
    SparkRange::new(minimum.unwrap_or(0), routed, intermediate, unit, shipped)?.admit(requested)?;
    Ok(())
}

pub fn render(columns: &[ColumnRange]) -> String {
    use std::fmt::Write;
    let mut output = String::new();
    for column in columns {
        let _ = writeln!(output, "{} RTX: {:.2} GiB routed experts, intermediate {}, unit {} columns",
            column.rtx_gpus, column.routed_expert_bytes as f64 / (1u64 << 30) as f64,
            column.intermediate, column.unit_columns);
        if column.range.is_none() {
            let _ = writeln!(output, "  geometric minimum unresolved: no complete capacity-safe ledger; counts below are policy candidates, not a supported-range claim");
        }
        let range = column.range.as_ref().unwrap_or(&column.policy);
        let minimum = column.range.as_ref().map(|range| range.minimum.to_string()).unwrap_or_else(|| "unresolved".into());
        let executable = column.executable_minimum.map(|n| n.to_string()).unwrap_or_else(|| "unresolved".into());
        let _ = writeln!(output, "  geometric N_min={} executable N_min={} N_max={} binding={:?}; size_cap={} width_cap={} shipped_max={}; width columns {:?}",
            minimum, executable, range.maximum, range.binding, range.size_cap, range.width_cap,
            range.shipped_maximum, column.width_classes_columns);
        for count in &column.counts {
            let gib = count.expert_bytes_per_rank.as_ref().map(|bytes| bytes.iter()
                .map(|bytes| format!("{:.2}", *bytes as f64 / (1u64 << 30) as f64)).collect::<Vec<_>>().join(", "))
                .unwrap_or_else(|| "unavailable".into());
            let exception = if count.shipped_exception { " shipped exception (below min tile)" } else { "" };
            let _ = writeln!(output, "  N={} units/expert {:?} x {} columns; expert GiB/rank [{}]; padded={}; executable today={}{}",
                count.sparks, count.units_per_expert, count.unit_columns, gib, count.padded, count.currently_launchable, exception);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_accepts_seven_and_eight_but_not_nine() {
        for ranks in [7, 8] {
            PlanOptions { placement: ExpertPlacement::Sparks { ranks }, ..Default::default() }.validate().unwrap();
        }
        assert!(PlanOptions { placement: ExpertPlacement::Sparks { ranks: 9 }, ..Default::default() }.validate().is_err());
    }

    #[test]
    fn policy_thresholds_and_minimum_override() {
        for (bytes, intermediate, minimum, maximum) in [
            (1 << 30, 2048, 0, 4), (128 << 30, 2048, 2, 8),
            (128 << 30, 1536, 2, 6), (16 << 30, 640, 1, 2),
            (16 << 30, 640, 3, 3), (127 << 30, 2048, 0, 7),
        ] {
            let range = SparkRange::new(minimum, bytes, intermediate, 128, 0).unwrap();
            assert_eq!(range.maximum, maximum);
            for count in minimum..=maximum { range.admit(count).unwrap(); }
            assert!(matches!(range.admit(maximum + 1), Err(RangeRefusal::Unsupported { .. })));
        }
    }

    #[test]
    fn shipped_qwen_exceptions_keep_native_padded_geometry() {
        for (package, maximum) in [("qwen4:exl3-k45", 4), ("qwen4:nvfp4", 6)] {
            let shipped = shipped_worlds(package).into_iter().max().unwrap();
            let range = SparkRange::new(0, 16 << 30, 640, 128, shipped).unwrap();
            assert_eq!((range.width_cap, range.maximum), (2, maximum));
            assert_eq!(range.binding, [BindingCap::Shipped]);
            for world in range.counts().filter(|world| *world > 0) {
                if let Ok(recipe) = super::super::experts::spark_recipe(package, 640, world) {
                    assert!(recipe.widths.iter().all(|width| *width > 0));
                    if world == 6 {
                        assert_eq!(recipe.slicing, crate::formats::fp8_experts::Slicing::Padded);
                        assert_eq!(recipe.widths, [128; 6]);
                    }
                }
            }
        }
    }

    #[test]
    fn resolved_matrix_recipes_never_have_empty_rank_slices() {
        for (package, intermediate) in [
            ("v41:mxfp4", 2304), ("dsv4f:mxfp4", 2048), ("dsv4p:exl3-k23", 3072),
            ("glmf:exl3-k34", 2048), ("glmf:nvfp4", 2048), ("mimof:fp8", 2048),
            ("mimop:fp8", 2048), ("qwen4:exl3-k45", 640), ("qwen4:nvfp4", 640),
            ("glm:exl3-k45", 2048), ("glm:nvfp4", 2048), ("kimik3:exl3-k2", 3072),
            ("step5:fp8", 1536), ("step5:bf16", 1536),
        ] {
            for world in 1..=MAX_SPARKS {
                if let Ok(recipe) = super::super::experts::spark_recipe(package, intermediate, world) {
                    assert_eq!(recipe.widths.len(), world, "{package} TP{world}");
                    assert!(recipe.widths.iter().all(|width| *width > 0), "{package} TP{world}");
                }
            }
        }
    }

    #[test]
    fn whole_range_classes_cover_seven_and_eight_without_new_exports() {
        let range = SparkRange::new(1, 128 << 30, 2048, 128, 0).unwrap();
        assert_eq!(range.width_classes(2048, 128).unwrap(), [2, 3, 4, 5, 6, 8, 16]);
        let through_six = SparkRange { maximum: 6, ..range.clone() };
        assert_eq!(range.width_classes(2048, 128), through_six.width_classes(2048, 128));
        for intermediate in [640, 1536, 2048, 3072] {
            for unit in [32, 128] {
                let range = SparkRange::new(1, 128 << 30, intermediate, unit, 0).unwrap();
                let packaged = range.width_classes(intermediate, unit).unwrap();
                for world in range.counts() {
                    let rule = balanced_rule(intermediate, unit, world);
                    for layer in [0, 3, 77] {
                        for rank in 0..world {
                            let map = cuteafd_core::tp_units::RankUnits::new(rule, 7, intermediate, unit, layer, world, rank).unwrap();
                            for width in map.width_classes().keys() { assert!(packaged.contains(&(width / unit))); }
                        }
                    }
                }
            }
        }
    }
}
