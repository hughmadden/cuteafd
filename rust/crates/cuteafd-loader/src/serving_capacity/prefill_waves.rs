//! Shared prefill admission (#762). A family supplies its measured base estimate
//! and hard executor capacities; the shared planner scales Spark compute to the
//! deployed rank count and admits at most ceil(T / S) + 1 waves. Units share one
//! 8192-row reserve. Runtime S/T adaptation is a separate executor change.

pub const RESERVE_ROWS: u64 = 8192;
/// A capped reserve is a multiple of lcm(2, 3, 4) pages, so every k = 2..4
/// split stays on 64-row pages (TJ 2026-10-27).
pub const RESERVE_STEP_ROWS: u64 = 12 * 64;

/// Reserves a planner may admit, largest first: the full reserve, then every
/// smaller multiple of [`RESERVE_STEP_ROWS`]. The planner takes the largest one
/// that fits the card; the KV pool takes what remains and never shrinks it.
pub fn reserve_candidates() -> Vec<u64> {
    std::iter::once(RESERVE_ROWS)
        .chain((1..=(RESERVE_ROWS - 1) / RESERVE_STEP_ROWS).rev().map(|n| n * RESERVE_STEP_ROWS))
        .collect()
}
pub const MAX_WAVES: u64 = 4;
pub const MIN_FILL: f64 = 0.5;

/// Exact shared capacities inside the reserve (three units round to 64-row pages).
pub fn unit_rows(waves: u64) -> u64 {
    match waves {
        0..=2 => 4096,
        3 => 2752,
        _ => 2048,
    }
}

/// Units of an admitted reserve: the full reserve's exact capacities
/// (`unit_rows`), else `reserve / waves`, exact on 64-row pages.
pub fn reserve_unit_rows(reserve: u64, waves: u64) -> u64 {
    let waves = waves.max(1);
    if reserve >= RESERVE_ROWS { return unit_rows(waves); }
    reserve.div_ceil(64 * waves) * 64
}

/// Family-owned evidence, never inferred from the family's name or rank count.
/// A measured-wave fallback does not pretend to supply independent S/T timings.
#[derive(Debug, Clone, Copy, Default)]
pub enum PrefillEstimate {
    Timing {
        spark_ms: f64,
        turnaround_ms: f64,
        reference_ranks: usize,
        reference_unit_rows: u64,
        source: &'static str,
    },
    MeasuredWaves {
        waves: u64,
        reference_ranks: usize,
        /// Unmeasured larger topologies may borrow a related family's estimate.
        larger_topology_waves: u64,
        source: &'static str,
    },
    /// No timing or wave comparison yet; retain the conservative two-wave estimate.
    #[default]
    Unmeasured,
}

impl PrefillEstimate {
    pub fn waves(self, spark_ranks: usize) -> u64 {
        if spark_ranks == 0 { return 1; }
        match self {
            Self::Timing { spark_ms, turnaround_ms, reference_ranks, .. } =>
                waves_for(turnaround_ms, spark_ms * reference_ranks as f64 / spark_ranks as f64),
            Self::MeasuredWaves { waves, reference_ranks, larger_topology_waves, .. } =>
                if spark_ranks > reference_ranks { larger_topology_waves } else { waves }.clamp(1, MAX_WAVES),
            Self::Unmeasured => 2,
        }
    }
}

/// Hard capacities and a base estimate consumed identically by plan/serve/golden.
/// Memory admission supplies the available arena extent, not a scheduling row cap.
#[derive(Debug, Clone, Copy)]
pub struct PrefillLimits {
    /// Measured starting unit for automatic selection; explicit unit pins bypass it.
    pub initial_unit_rows: Option<u64>,
    pub max_unit_rows: u64,
    pub max_span_rows: u64,
    pub max_waves: u64,
    pub capacities: &'static [u32],
    pub estimate: PrefillEstimate,
}

impl Default for PrefillLimits {
    fn default() -> Self {
        Self { initial_unit_rows: None, max_unit_rows: 4096, max_span_rows: RESERVE_ROWS + 64,
            max_waves: MAX_WAVES, capacities: &cuteafd_core::EXL3_CAPACITIES,
            estimate: PrefillEstimate::Unmeasured }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PrefillError {
    #[error("unsupported prefill shape: {waves} waves x {unit_rows} rows; units must be page-aligned 64..=4096 rows and waves 1..=4")]
    InvalidShape { unit_rows: u64, waves: u64 },
    #[error("unsupported prefill shape: {waves} waves x {unit_rows} rows; executor supports at most {max_waves} waves x {max_unit_rows} rows")]
    ExecutorCapacity { unit_rows: u64, waves: u64, max_unit_rows: u64, max_waves: u64 },
    #[error("unsupported prefill shape: {waves} waves x {unit_rows} rows exceeds the admitted {reserve_rows}-row live-slot reserve")]
    ReserveExceeded { unit_rows: u64, waves: u64, reserve_rows: u64 },
    #[error("unsupported prefill span: {span_rows} rows must equal {slot_rows} live slot rows within the {max_span_rows}-row shared arena")]
    IntakeSpan { span_rows: u64, slot_rows: u64, max_span_rows: u64 },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PrefillOverrides {
    pub unit_rows: Option<u64>,
    pub waves: Option<u64>,
    pub span_rows: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillPlan {
    pub unit_rows: u64,
    pub waves: u64,
    pub span_rows: u64,
}

impl PrefillPlan {
    pub fn units(self) -> super::PrefillShape {
        super::PrefillShape { unit_rows: self.unit_rows, waves: self.waves }
    }
}

pub fn prefill_shape(spark_ranks: usize, limits: PrefillLimits,
    explicit: PrefillOverrides) -> Result<PrefillPlan, PrefillError> {
    let maximum = limits.estimate.waves(spark_ranks).min(limits.max_waves).min(MAX_WAVES);
    let mut waves = explicit.waves.unwrap_or(maximum);
    loop {
        let budget = unit_rows(waves).min(limits.max_unit_rows)
            .min(limits.max_span_rows.min(RESERVE_ROWS + 64) / waves.max(1));
        let automatic_budget = budget.min(limits.initial_unit_rows.unwrap_or(budget));
        let rows = explicit.unit_rows.unwrap_or_else(|| limits.capacities.iter().copied()
            .map(u64::from).filter(|&rows| rows >= 64 && rows % 64 == 0 && rows <= automatic_budget)
            .max().unwrap_or(0));
        let shape = validate_shape(rows, waves, limits, None);
        if shape.is_ok() { return validate_shape(rows, waves, limits, explicit.span_rows); }
        if explicit.waves.is_some() || waves <= 1 { return shape; }
        // Only automatic depth may shrink to fit the admitted arena. Pins never change.
        waves -= 1;
    }
}

fn validate_shape(rows: u64, waves: u64, limits: PrefillLimits,
    explicit_span: Option<u64>) -> Result<PrefillPlan, PrefillError> {
    let units = super::PrefillShape { unit_rows: rows, waves };
    if units.validate().is_err() {
        return Err(PrefillError::InvalidShape { unit_rows: rows, waves });
    }
    if waves > limits.max_waves || rows > limits.max_unit_rows {
        return Err(PrefillError::ExecutorCapacity { unit_rows: rows, waves,
            max_unit_rows: limits.max_unit_rows, max_waves: limits.max_waves });
    }
    let reserve_rows = RESERVE_ROWS.div_ceil(64 * waves) * (64 * waves);
    let slot_rows = units.slot_rows();
    if slot_rows > reserve_rows {
        return Err(PrefillError::ReserveExceeded { unit_rows: rows, waves, reserve_rows });
    }
    let span_rows = explicit_span.unwrap_or(slot_rows);
    let max_span_rows = limits.max_span_rows.min(RESERVE_ROWS + 64);
    if span_rows != slot_rows || span_rows > max_span_rows {
        return Err(PrefillError::IntakeSpan { span_rows, slot_rows, max_span_rows });
    }
    Ok(PrefillPlan { unit_rows: rows, waves, span_rows })
}

/// ceil(T / S) + 1; invalid/unavailable measurements retain two waves.
pub fn waves_for(turnaround: f64, spark: f64) -> u64 {
    if !(turnaround.is_finite() && spark.is_finite()) || spark <= 0.0 || turnaround <= 0.0 {
        return 2;
    }
    ((turnaround / spark).ceil() as u64).saturating_add(1).clamp(2, MAX_WAVES)
}

/// Use fewer waves when a short prompt cannot keep each unit half full.
pub fn waves_used(rows: u64, waves: u64, unit: u64) -> u64 {
    if rows == 0 || unit == 0 || waves <= 1 { return 1; }
    let cover = rows.div_ceil(unit);
    let filled = (rows as f64 / (MIN_FILL * unit as f64)).floor() as u64;
    cover.max(waves.min(filled)).min(waves).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(waves: u64) -> PrefillLimits {
        PrefillLimits { estimate: PrefillEstimate::MeasuredWaves { waves, reference_ranks: 6, larger_topology_waves: waves, source: "test comparison" },
            ..Default::default() }
    }

    #[test]
    fn units_are_the_exact_capacities_inside_the_reserve() {
        for waves in 2..=MAX_WAVES {
            let unit = unit_rows(waves);
            assert!(cuteafd_core::EXL3_CAPACITIES.contains(&(unit as u32)));
            assert_eq!(unit % 64, 0);
            assert!(unit * waves <= RESERVE_ROWS + 64 * waves);
            assert!(unit * waves >= RESERVE_ROWS);
        }
        assert_eq!((2..=4).map(unit_rows).collect::<Vec<_>>(), [4096, 2752, 2048]);
        assert_eq!((2..=4).map(|k| reserve_unit_rows(RESERVE_ROWS, k)).collect::<Vec<_>>(), [4096, 2752, 2048]);
        let candidates = reserve_candidates();
        assert_eq!(candidates[..3], [8192, 7680, 6912]);
        assert_eq!(*candidates.last().unwrap(), 768);
        for &reserve in &candidates[1..] {
            for waves in 2..=MAX_WAVES {
                let unit = reserve_unit_rows(reserve, waves);
                assert_eq!((unit * waves, unit % 64), (reserve, 0), "{reserve} / {waves}");
            }
        }
        assert_eq!((2..=4).map(|k| reserve_unit_rows(3072, k)).collect::<Vec<_>>(), [1536, 1024, 768]);
    }

    #[test]
    fn estimates_scale_compute_not_a_rank_threshold() {
        let estimate = PrefillEstimate::Timing { spark_ms: 10.0, turnaround_ms: 15.0,
            reference_ranks: 4, reference_unit_rows: 2752, source: "test timing" };
        assert_eq!((0..=8).map(|n| estimate.waves(n)).collect::<Vec<_>>(), [1, 2, 2, 3, 3, 3, 4, 4, 4]);
        let other = PrefillEstimate::Timing { spark_ms: 10.0, turnaround_ms: 2.0,
            reference_ranks: 4, reference_unit_rows: 2752, source: "other test timing" };
        assert_eq!(other.waves(8), 2);
    }

    #[test]
    fn measured_fallback_does_not_invent_spark_scaling() {
        for waves in 1..=MAX_WAVES {
            for ranks in 0..=8 {
                let shape = prefill_shape(ranks, measured(waves), Default::default()).unwrap();
                let k = if ranks == 0 { 1 } else { waves };
                assert_eq!(shape, PrefillPlan { unit_rows: unit_rows(k), waves: k, span_rows: k * unit_rows(k) });
            }
        }
    }

    #[test]
    fn estimate_is_bounded_by_hard_capacity_and_memory() {
        let limits = PrefillLimits { max_waves: 2, ..measured(4) };
        assert_eq!(prefill_shape(6, limits, Default::default()).unwrap().waves, 2);
        let limits = PrefillLimits { max_unit_rows: 1024, max_span_rows: 2048, ..measured(3) };
        let shape = prefill_shape(6, limits, Default::default()).unwrap();
        assert_eq!(shape, PrefillPlan { unit_rows: 256, waves: 3, span_rows: 768 });
        let shape = prefill_shape(6, limits, PrefillOverrides { unit_rows: Some(1024), ..Default::default() }).unwrap();
        assert_eq!(shape, PrefillPlan { unit_rows: 1024, waves: 2, span_rows: 2048 });
        let limits = PrefillLimits { max_span_rows: 128, capacities: &[256, 1024, 4096], ..measured(4) };
        assert!(prefill_shape(6, limits, Default::default()).is_err());
    }

    #[test]
    fn initial_unit_preference_only_changes_automatic_rows() {
        let limits = PrefillLimits { initial_unit_rows: Some(2048), ..measured(3) };
        assert_eq!(prefill_shape(6, measured(3), Default::default()).unwrap(),
            PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256 });
        assert_eq!(prefill_shape(6, limits, Default::default()).unwrap(),
            PrefillPlan { unit_rows: 2048, waves: 3, span_rows: 6144 });
        for (rows, waves) in [(256, 3), (2752, 3), (4096, 2)] {
            assert_eq!(prefill_shape(6, limits, PrefillOverrides {
                unit_rows: Some(rows), waves: Some(waves), ..Default::default()
            }).unwrap(), PrefillPlan { unit_rows: rows, waves, span_rows: rows * waves });
        }
        assert!(matches!(prefill_shape(6, limits, PrefillOverrides {
            unit_rows: Some(4096), waves: Some(3), ..Default::default()
        }), Err(PrefillError::ReserveExceeded { .. })));
        let capped = PrefillLimits { max_unit_rows: 1024, ..limits };
        assert_eq!(prefill_shape(6, capped, Default::default()).unwrap().unit_rows, 1024);
        assert!(matches!(prefill_shape(6, capped, PrefillOverrides {
            unit_rows: Some(2048), waves: Some(3), ..Default::default()
        }), Err(PrefillError::ExecutorCapacity { .. })));
        let shallow = PrefillLimits { max_waves: 2, ..limits };
        assert_eq!(prefill_shape(6, shallow, Default::default()).unwrap().waves, 2);
        assert!(matches!(prefill_shape(6, shallow, PrefillOverrides {
            waves: Some(3), ..Default::default()
        }), Err(PrefillError::ExecutorCapacity { .. })));
    }

    #[test]
    fn initial_unit_preference_still_obeys_memory_and_export_limits() {
        let limits = PrefillLimits { initial_unit_rows: Some(2048), max_span_rows: 2048,
            ..measured(3) };
        assert_eq!(prefill_shape(6, limits, Default::default()).unwrap(),
            PrefillPlan { unit_rows: 256, waves: 3, span_rows: 768 });
        let limits = PrefillLimits { initial_unit_rows: Some(2048),
            capacities: &[256, 4096], ..measured(3) };
        assert_eq!(prefill_shape(6, limits, Default::default()).unwrap().unit_rows, 256);
    }

    #[test]
    fn pins_override_estimates_but_not_hard_limits() {
        for (waves, rows) in [(1, 4096), (2, 4096), (3, 2752), (4, 2048)] {
            let shape = prefill_shape(6, measured(2), PrefillOverrides {
                unit_rows: Some(rows), waves: Some(waves), ..Default::default()
            }).unwrap();
            assert_eq!((shape.unit_rows, shape.waves), (rows, waves));
        }
        let error = prefill_shape(6, PrefillLimits { max_waves: 2, ..measured(4) },
            PrefillOverrides { waves: Some(3), ..Default::default() }).unwrap_err();
        assert!(matches!(error, PrefillError::ExecutorCapacity { waves: 3, max_waves: 2, .. }));
        let error = prefill_shape(6, PrefillLimits { max_unit_rows: 1024, ..measured(3) },
            PrefillOverrides { unit_rows: Some(2048), waves: Some(2), ..Default::default() }).unwrap_err();
        assert!(matches!(error, PrefillError::ExecutorCapacity { unit_rows: 2048, .. }));
    }

    #[test]
    fn explicit_shapes_cannot_grow_the_shared_live_reserve() {
        for (waves, rows) in [(1, 4096), (2, 4096), (3, 2752), (4, 2048), (4, 512)] {
            let shape = prefill_shape(6, measured(3), PrefillOverrides {
                waves: Some(waves), unit_rows: Some(rows), ..Default::default()
            }).unwrap();
            assert_eq!((shape.waves, shape.unit_rows, shape.span_rows), (waves, rows, waves * rows));
        }
        for (waves, rows) in [(3, 2816), (3, 4096), (4, 2112), (4, 4096)] {
            assert!(matches!(prefill_shape(6, measured(3), PrefillOverrides {
                waves: Some(waves), unit_rows: Some(rows), ..Default::default()
            }), Err(PrefillError::ReserveExceeded { .. })));
        }
    }

    #[test]
    fn intake_is_exactly_the_shared_live_slot_extent() {
        let limits = PrefillLimits { max_span_rows: u64::MAX, ..measured(3) };
        assert_eq!(prefill_shape(6, limits, Default::default()).unwrap(),
            PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256 });
        for span in [0, 8192, 8257, 16384] {
            assert!(matches!(prefill_shape(6, limits, PrefillOverrides {
                waves: Some(3), span_rows: Some(span), ..Default::default()
            }), Err(PrefillError::IntakeSpan { .. })));
        }
    }

    #[test]
    fn explicit_shape_errors_are_typed_and_never_clamped() {
        for (rows, waves) in [(0, 2), (65, 2), (8192, 1), (4096, 0), (2048, 5)] {
            assert_eq!(prefill_shape(6, measured(3), PrefillOverrides {
                unit_rows: Some(rows), waves: Some(waves), ..Default::default()
            }).unwrap_err(), PrefillError::InvalidShape { unit_rows: rows, waves });
        }
    }

    #[test]
    fn qwen_mapped_ple_ring_is_a_hard_two_wave_constraint() {
        let limits = crate::families::qwen4::prefill_limits(false);
        assert_eq!(prefill_shape(6, limits, Default::default()).unwrap(),
            PrefillPlan { unit_rows: 4096, waves: 2, span_rows: 8192 });
        for waves in [3, 4] {
            assert!(matches!(prefill_shape(6, limits, PrefillOverrides { waves: Some(waves), ..Default::default() }),
                Err(PrefillError::ExecutorCapacity { max_waves: 2, .. })));
        }
        assert_eq!(prefill_shape(0, crate::families::qwen4::prefill_limits(true), Default::default()).unwrap().waves, 1);
    }

    #[test]
    fn waves_hide_turnaround_behind_spark_compute() {
        for (t, s, k) in [(15.0, 10.0, 3), (10.0, 10.0, 2), (5.0, 10.0, 2), (30.0, 10.0, 4), (f64::MAX, f64::MIN_POSITIVE, 4)] {
            assert_eq!(waves_for(t, s), k);
        }
        for (t, s) in [(0.0, 1.0), (1.0, 0.0), (f64::NAN, 1.0), (1.0, f64::INFINITY)] {
            assert_eq!(waves_for(t, s), 2);
        }
    }

    #[test]
    fn short_prompts_keep_units_at_least_half_full() {
        let (k, unit) = (3, 2752);
        assert_eq!(waves_used(1, k, unit), 1);
        assert_eq!(waves_used(2752, k, unit), 2);
        assert_eq!(waves_used(2 * 1376 - 1, k, unit), 1);
        assert_eq!(waves_used(3 * 1376 - 1, k, unit), 2);
        assert_eq!(waves_used(3 * 1376, k, unit), 3);
        assert_eq!(waves_used(8256, k, unit), 3);
        assert_eq!(waves_used(65536, k, unit), 3);
        assert_eq!(waves_used(65536, 1, unit), 1);
        assert_eq!(waves_used(4095, 2, 4096), 1);
        assert_eq!(waves_used(4096, 2, 4096), 2);
        assert_eq!(waves_used(4097, 2, 4096), 2);
        assert_eq!(waves_used(4095, 4, 2048), 3);
        assert_eq!(waves_used(4096, 4, 2048), 4);
        assert_eq!(waves_used(3 * 1024, 4, 2048), 3);
        assert_eq!(waves_used(3 * 1024 - 1, 4, 2048), 2);
        assert_eq!(waves_used(0, 3, 2752), 1);
    }
}
