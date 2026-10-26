//! Prefill wave count (#762, TJ 2026-10-25). Every family admits units inside
//! the same 8192-row reserve. The policy splits it into k waves on exact EXL3
//! capacities (2 -> 4096, 3 -> 2752, 4 -> 2048), bounded by the family's
//! measured automatic unit limit. k hides raptor's turnaround T behind the Sparks'
//! per-unit compute S: a Spark idles between units only if T > (k - 1) * S,
//! so k = ceil(T / S) + 1, at most [`MAX_WAVES`].
//!
//! The planner fixes k per topology (return slots, landing rotation and Spark
//! execution slots are sized at startup). A prompt that would cut units
//! below [`MIN_FILL`] of a unit runs on fewer waves; a long prompt runs k
//! full units. The executor never reallocates or recaptures.

/// Rows every family reserves for prefill units in flight.
pub const RESERVE_ROWS: u64 = 8192;
/// Most waves in flight (TJ: a fifth adds slots for the least benefit).
pub const MAX_WAVES: u64 = 4;
/// A short prompt never cuts units below this share of a unit (TJ
/// 2026-10-25: 50%; 75% was the alternative considered).
pub const MIN_FILL: f64 = 0.5;

/// Unit rows of `waves` waves sharing the reserve: the exact EXL3 capacity
/// per wave count (`EXL3_CAPACITIES`). Three waves take 2752 rows (8256 in
/// flight, the reserve rounded up to whole 64-row pages).
pub fn unit_rows(waves: u64) -> u64 {
    match waves {
        0 | 1 => 4096,
        2 => 4096,
        3 => 2752,
        _ => 2048,
    }
}

/// Measured scheduling ceilings, separate from hard executor limits. Explicit
/// pins may exceed these ceilings, but must still satisfy every hard limit.
#[derive(Debug, Clone, Copy)]
pub struct PrefillAutomatic {
    pub max_unit_rows: u64,
    pub max_waves: u64,
}

impl Default for PrefillAutomatic {
    fn default() -> Self {
        Self { max_unit_rows: 4096, max_waves: MAX_WAVES }
    }
}

/// Hard kernel/arena limits and the family's measured automatic ceilings.
#[derive(Debug, Clone, Copy)]
pub struct PrefillLimits {
    pub max_unit_rows: u64,
    /// Capacity of the intake arena, bounded by the shared live-slot reserve.
    pub max_span_rows: u64,
    pub max_waves: u64,
    pub automatic: PrefillAutomatic,
}

impl Default for PrefillLimits {
    fn default() -> Self {
        Self { max_unit_rows: 4096, max_span_rows: RESERVE_ROWS + 64, max_waves: MAX_WAVES,
            automatic: PrefillAutomatic::default() }
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

/// The shape admitted before allocating workspace, Spark execution slots or
/// landing rotation. Every family shares the topology policy and measured-cap
/// resolution; only its declared automatic ceilings differ.
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
    let waves = explicit.waves.unwrap_or_else(|| planned_waves(spark_ranks).min(limits.automatic.max_waves));
    let rows = explicit.unit_rows.unwrap_or_else(|| unit_rows(waves).min(limits.automatic.max_unit_rows));
    let units = super::PrefillShape { unit_rows: rows, waves };
    if units.validate().is_err() {
        return Err(PrefillError::InvalidShape { unit_rows: rows, waves });
    }
    if waves > limits.max_waves || rows > limits.max_unit_rows {
        return Err(PrefillError::ExecutorCapacity { unit_rows: rows, waves,
            max_unit_rows: limits.max_unit_rows, max_waves: limits.max_waves });
    }
    // Rounding each share to a whole page admits 3 x 2752, not 3 x 4096.
    // Intake and execution share exactly this extent; no larger family ceiling.
    let reserve_rows = RESERVE_ROWS.div_ceil(64 * waves) * (64 * waves);
    let slot_rows = units.slot_rows();
    if slot_rows > reserve_rows {
        return Err(PrefillError::ReserveExceeded { unit_rows: rows, waves, reserve_rows });
    }
    let span_rows = explicit.span_rows.unwrap_or(slot_rows);
    let max_span_rows = limits.max_span_rows.min(RESERVE_ROWS + 64);
    if span_rows != slot_rows || span_rows > max_span_rows {
        return Err(PrefillError::IntakeSpan { span_rows, slot_rows, max_span_rows });
    }
    Ok(PrefillPlan { unit_rows: rows, waves, span_rows })
}

/// Waves that hide `turnaround` (raptor's reply drain, reduce, attention and
/// routing between a unit's waves) behind `spark` (one unit's expert compute):
/// ceil(T / S) + 1, clamped to 2..=[`MAX_WAVES`].
pub fn waves_for(turnaround: f64, spark: f64) -> u64 {
    if !(turnaround.is_finite() && spark.is_finite()) || spark <= 0.0 || turnaround <= 0.0 {
        return 2;
    }
    ((turnaround / spark).ceil() as u64 + 1).clamp(2, MAX_WAVES)
}

/// The planner's wave count for a topology before any measurement: three for
/// six or more Sparks (S ~ 13 ms vs T ~ 19.5 ms on GLM 5.3 K4, 2026-10-25),
/// two below that (fewer ranks each compute more, so S already exceeds T).
/// No Sparks run serially.
pub fn planned_waves(spark_ranks: usize) -> u64 {
    match spark_ranks {
        0 => 1,
        1..=5 => 2,
        _ => 3,
    }
}

/// Waves in flight for a `rows`-row prompt with at most `waves` units of
/// `unit` rows admitted: enough units to cover it, otherwise as many as the
/// admitted count allows while each unit keeps [`MIN_FILL`] of `unit`.
/// min(k, max(ceil(P / unit), min(k, floor(P / (MIN_FILL * unit))))), >= 1;
/// a prompt longer than k units streams k full units at a time.
pub fn waves_used(rows: u64, waves: u64, unit: u64) -> u64 {
    if rows == 0 || unit == 0 || waves <= 1 { return 1; }
    let cover = rows.div_ceil(unit);
    let filled = (rows as f64 / (MIN_FILL * unit as f64)).floor() as u64;
    cover.max(waves.min(filled)).min(waves).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_are_the_exact_capacities_inside_the_reserve() {
        for waves in 2..=MAX_WAVES {
            let unit = unit_rows(waves);
            assert!(cuteafd_core::EXL3_CAPACITIES.contains(&(unit as u32)), "{unit}");
            assert_eq!(unit % 64, 0);
            // Inside the reserve rounded up to whole pages.
            assert!(unit * waves <= RESERVE_ROWS + 64 * waves, "{waves} x {unit}");
            assert!(unit * waves >= RESERVE_ROWS);
        }
        assert_eq!((2..=4).map(unit_rows).collect::<Vec<_>>(), [4096, 2752, 2048]);
    }

    #[test]
    fn automatic_units_consume_supplied_constraints_without_changing_topology() {
        for ceiling in [1024, 2048, 2752, 4096] {
            let limits = PrefillLimits { automatic: PrefillAutomatic {
                max_unit_rows: ceiling, ..Default::default() }, ..Default::default() };
            for ranks in 0..=8 {
                let shape = prefill_shape(ranks, limits, PrefillOverrides::default()).unwrap();
                let waves = planned_waves(ranks);
                let rows = unit_rows(waves).min(ceiling);
                assert_eq!(shape, PrefillPlan { unit_rows: rows, waves, span_rows: rows * waves },
                    "ceiling={ceiling} ranks={ranks}");
            }
            for waves in 1..=MAX_WAVES {
                let shape = prefill_shape(6, limits, PrefillOverrides {
                    waves: Some(waves), ..Default::default()
                }).unwrap();
                assert_eq!(shape.unit_rows, unit_rows(waves).min(ceiling), "ceiling={ceiling} waves={waves}");
            }
        }
    }

    #[test]
    fn measured_ceilings_do_not_rewrite_explicit_pins_or_hard_refusals() {
        for ceiling in [2048, 2752] {
            let limits = PrefillLimits { automatic: PrefillAutomatic {
                max_unit_rows: ceiling, ..Default::default() }, ..Default::default() };
            for (waves, rows) in [(1, 4096), (2, 4096), (3, 2752), (4, 2048)] {
                let shape = prefill_shape(6, limits, PrefillOverrides {
                    unit_rows: Some(rows), waves: Some(waves), ..Default::default()
                }).unwrap();
                assert_eq!((shape.unit_rows, shape.waves), (rows, waves));
            }
            let limits = PrefillLimits { max_unit_rows: 1024, ..limits };
            let error = prefill_shape(6, limits, PrefillOverrides::default()).unwrap_err();
            assert!(matches!(error, PrefillError::ExecutorCapacity { max_unit_rows: 1024, .. }), "{error}");
            let error = prefill_shape(6, limits, PrefillOverrides {
                unit_rows: Some(2048), waves: Some(2), ..Default::default()
            }).unwrap_err();
            assert!(matches!(error, PrefillError::ExecutorCapacity { unit_rows: 2048, .. }), "{error}");
            let error = prefill_shape(6, PrefillLimits { max_unit_rows: 4096, ..limits }, PrefillOverrides {
                unit_rows: Some(4096), waves: Some(3), ..Default::default()
            }).unwrap_err();
            assert!(matches!(error, PrefillError::ReserveExceeded { .. }), "{error}");
        }
    }

    #[test]
    fn measured_wave_ceilings_only_bound_automatic_selection() {
        let limits = PrefillLimits { automatic: PrefillAutomatic { max_unit_rows: 2048, max_waves: 2 },
            ..Default::default() };
        let shape = prefill_shape(6, limits, PrefillOverrides::default()).unwrap();
        assert_eq!(shape, PrefillPlan { unit_rows: 2048, waves: 2, span_rows: 4096 });
        let shape = prefill_shape(6, limits, PrefillOverrides { waves: Some(3), ..Default::default() }).unwrap();
        assert_eq!(shape, PrefillPlan { unit_rows: 2048, waves: 3, span_rows: 6144 });
        let error = prefill_shape(6, PrefillLimits { max_waves: 2, ..limits }, PrefillOverrides {
            waves: Some(3), ..Default::default()
        }).unwrap_err();
        assert!(matches!(error, PrefillError::ExecutorCapacity { waves: 3, max_waves: 2, .. }));
    }

    #[test]
    fn explicit_shapes_cannot_grow_the_shared_live_reserve() {
        for (waves, rows) in [(1, 4096), (2, 4096), (3, 2752), (4, 2048), (4, 512)] {
            let shape = prefill_shape(6, PrefillLimits::default(), PrefillOverrides {
                waves: Some(waves), unit_rows: Some(rows), ..Default::default()
            }).unwrap();
            assert_eq!((shape.waves, shape.unit_rows, shape.span_rows), (waves, rows, waves * rows));
        }
        for (waves, rows) in [(3, 2816), (3, 4096), (4, 2112), (4, 4096)] {
            let error = prefill_shape(6, PrefillLimits::default(), PrefillOverrides {
                waves: Some(waves), unit_rows: Some(rows), ..Default::default()
            }).unwrap_err();
            assert!(matches!(error, PrefillError::ReserveExceeded { .. }), "{error}");
        }
    }

    #[test]
    fn intake_is_exactly_the_shared_live_slot_extent() {
        let limits = PrefillLimits { max_span_rows: u64::MAX, ..Default::default() };
        let shape = prefill_shape(6, limits, PrefillOverrides::default()).unwrap();
        assert_eq!(shape, PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256 });
        assert_eq!(shape.units().slot_rows(), 8256);
        for span in [0, 8192, 8257, 16384] {
            let error = prefill_shape(6, limits, PrefillOverrides {
                span_rows: Some(span), ..Default::default()
            }).unwrap_err();
            assert!(matches!(error, PrefillError::IntakeSpan { .. }), "{error}");
        }
        let error = prefill_shape(6, PrefillLimits::default(), PrefillOverrides {
            span_rows: Some(16384), ..Default::default()
        }).unwrap_err();
        assert!(matches!(error, PrefillError::IntakeSpan { .. }));
    }

    #[test]
    fn executor_and_shape_errors_are_typed_and_never_clamped() {
        for (rows, waves) in [(0, 2), (65, 2), (8192, 1), (4096, 0), (2048, 5)] {
            let error = prefill_shape(6, PrefillLimits::default(), PrefillOverrides {
                unit_rows: Some(rows), waves: Some(waves), ..Default::default()
            }).unwrap_err();
            assert_eq!(error, PrefillError::InvalidShape { unit_rows: rows, waves });
        }
        let error = prefill_shape(6, PrefillLimits { max_waves: 2, ..Default::default() },
            PrefillOverrides::default()).unwrap_err();
        assert!(matches!(error, PrefillError::ExecutorCapacity { waves: 3, max_waves: 2, .. }));
    }

    #[test]
    fn qwen_fixed_ple_ring_admits_adaptive_spark_waves() {
        let limits = crate::families::qwen4::prefill_limits(false);
        assert_eq!(prefill_shape(6, limits, PrefillOverrides::default()).unwrap(),
            PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256 });
        for waves in [3, 4] {
            let shape = prefill_shape(6, limits, PrefillOverrides {
                waves: Some(waves), ..Default::default()
            }).unwrap();
            assert_eq!(shape.waves, waves);
        }
        assert_eq!(prefill_shape(0, crate::families::qwen4::prefill_limits(true),
            PrefillOverrides::default()).unwrap().waves, 1);
    }

    #[test]
    fn waves_hide_turnaround_behind_spark_compute() {
        // GLM 5.3 K4 max: S ~ 13 ms, T ~ 19.5 ms -> 3 waves (2 leave the Sparks idle).
        assert_eq!(waves_for(19.5, 13.0), 3);
        assert_eq!(waves_for(13.0, 13.0), 2);
        assert_eq!(waves_for(5.0, 13.0), 2);
        assert_eq!(waves_for(30.0, 13.0), 4);
        assert_eq!(waves_for(100.0, 1.0), MAX_WAVES);
        for (t, s) in [(0.0, 1.0), (1.0, 0.0), (f64::NAN, 1.0), (1.0, f64::INFINITY)] {
            assert_eq!(waves_for(t, s), 2);
        }
    }

    #[test]
    fn planned_waves_by_spark_count() {
        assert_eq!(planned_waves(0), 1);
        assert_eq!((1..=5).map(planned_waves).collect::<Vec<_>>(), [2; 5]);
        assert_eq!(planned_waves(6), 3);
        assert_eq!(planned_waves(8), 3);
    }

    #[test]
    fn short_prompts_keep_units_at_least_half_full() {
        // k = 3, unit 2752: half a unit is 1376 rows.
        let (k, unit) = (3, 2752);
        assert_eq!(waves_used(1, k, unit), 1);
        // Exactly 50%: two units of 1376 rows are allowed, three are not.
        assert_eq!(waves_used(2752, k, unit), 2);
        assert_eq!(waves_used(2 * 1376 - 1, k, unit), 1);
        assert_eq!(waves_used(3 * 1376 - 1, k, unit), 2);
        assert_eq!(waves_used(3 * 1376, k, unit), 3);
        // Long prompts run every admitted wave; beyond k units, the stream continues.
        assert_eq!(waves_used(8256, k, unit), 3);
        assert_eq!(waves_used(65536, k, unit), 3);
        assert_eq!(waves_used(65536, 1, unit), 1);
        // Two waves of 4096: a prompt under 4096 rows splits only at >= 2 x 2048.
        assert_eq!(waves_used(4095, 2, 4096), 1);
        assert_eq!(waves_used(4096, 2, 4096), 2);
        assert_eq!(waves_used(4097, 2, 4096), 2);
        // Four waves of 2048.
        assert_eq!(waves_used(4095, 4, 2048), 3);
        assert_eq!(waves_used(4096, 4, 2048), 4);
        assert_eq!(waves_used(3 * 1024, 4, 2048), 3);
        assert_eq!(waves_used(3 * 1024 - 1, 4, 2048), 2);
        assert_eq!(waves_used(0, 3, 2752), 1);
    }
}
