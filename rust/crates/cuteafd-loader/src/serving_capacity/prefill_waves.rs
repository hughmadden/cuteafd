//! Shared prefill admission (#762). A family supplies its measured base estimate
//! and hard executor capacities; the shared planner scales Spark compute to the
//! deployed rank count and admits at most ceil(T / S) + 1 waves. Units are
//! automatic splits of one prefill reserve (TJ 2026-10-27): the standard 8192
//! rows where the card holds it, else the largest 64-row multiple that fits.

/// The standard prefill reserve.
pub const RESERVE_ROWS: u64 = 8192;
pub const MAX_WAVES: u64 = 4;
pub const MIN_FILL: f64 = 0.5;
/// Prefill rows are 64-row pages.
pub const ROW_PAGE: u64 = 64;

/// The unit of a `waves`-way split of `reserve` rows: reserve / k rounded up
/// to a 64-row page (one and two waves both run reserve / 2). It runs as live
/// rows on the smallest compiled capacity at least that wide.
pub fn unit_rows_in(waves: u64, reserve: u64) -> u64 {
    reserve.div_ceil(waves.clamp(2, MAX_WAVES)).div_ceil(ROW_PAGE) * ROW_PAGE
}

/// The standard reserve's units: 4096 / 2752 / 2048 at two, three and four waves.
pub fn unit_rows(waves: u64) -> u64 { unit_rows_in(waves, RESERVE_ROWS) }

/// Rows a reserve charges: the widest split of it (k x unit(k) over k = 2..4;
/// 8256 for 8192, at most reserve + 192).
pub fn charge_rows(reserve: u64) -> u64 {
    (2..=MAX_WAVES).map(|k| k * unit_rows_in(k, reserve)).max().unwrap_or(0)
}

/// The largest reserve (a multiple of 64, at most the standard one) whose
/// charge fits in `rows`.
pub fn reserve_within(rows: u64) -> u64 {
    let mut reserve = RESERVE_ROWS.min(rows) / ROW_PAGE * ROW_PAGE;
    while reserve > 0 && charge_rows(reserve) > rows { reserve -= ROW_PAGE; }
    reserve
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
    /// The executor re-cuts its slots per prefill (#762 stages 2–4): the plan
    /// admits `max_waves` slots of the chosen unit inside the shared reserve,
    /// and the estimate is only the starting depth.
    pub adaptive: bool,
}

impl Default for PrefillLimits {
    fn default() -> Self {
        Self { initial_unit_rows: None, max_unit_rows: 4096, max_span_rows: RESERVE_ROWS + 64,
            max_waves: MAX_WAVES, capacities: &cuteafd_core::EXL3_CAPACITIES,
            estimate: PrefillEstimate::Unmeasured, adaptive: false }
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
    #[error("unsupported prefill slots: {slots} admitted slots for {waves} waves (an adaptive family admits {waves}..={max_waves}; a static one exactly its waves)")]
    Slots { slots: u64, waves: u64, max_waves: u64 },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PrefillOverrides {
    pub unit_rows: Option<u64>,
    pub waves: Option<u64>,
    pub span_rows: Option<u64>,
    /// Admitted slots of a resolved deployment (#762): `waves` starts the
    /// depth, more slots let an adaptive executor choose. A launch passes the
    /// deployment's value back; equal to `waves` it pins the depth.
    pub slots: Option<u64>,
    /// An adaptive family's reserve below the standard one (TJ's reserve
    /// rule: the planner's fit cap); its depths cut it (`wave_control::unit_for_in`).
    pub reserve_rows: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillPlan {
    /// The admitted (widest) unit.
    pub unit_rows: u64,
    /// The starting depth (the estimate, or the pin).
    pub waves: u64,
    /// Rows of every slot-owned plane: the reserve every admitted depth is cut from.
    pub span_rows: u64,
    /// Admitted slots: the deepest depth the executor may run (#762). Equal
    /// to `waves` for a static or pinned plan.
    pub slots: u64,
}

impl PrefillPlan {
    /// The starting depth's units: `waves` of the admitted unit's cut at that
    /// depth (`wave_control::unit_for`; the unit itself when static).
    pub fn units(self) -> super::PrefillShape {
        super::PrefillShape { unit_rows: super::wave_control::unit_for(self.waves, self.unit_rows), waves: self.waves }
    }
    /// The admitted maximum: `slots` slots of at most the admitted unit.
    pub fn admitted(self) -> super::PrefillShape {
        super::PrefillShape { unit_rows: self.unit_rows, waves: self.slots }
    }
    /// Whether the executor may change the depth at runtime.
    pub fn adaptive(self) -> bool { self.slots > self.waves || super::wave_control::candidates(self.slots, self.unit_rows).len() > 1 }
}

/// The widest unit an adaptive family admits: half the shared reserve, so its
/// controller can run any depth from 2 to 4 (#762; TJ 2026-10-27).
pub const ADAPTIVE_UNIT_ROWS: u64 = RESERVE_ROWS / 2;

pub fn prefill_shape(spark_ranks: usize, limits: PrefillLimits,
    explicit: PrefillOverrides) -> Result<PrefillPlan, PrefillError> {
    let maximum = limits.estimate.waves(spark_ranks).min(limits.max_waves).min(MAX_WAVES);
    let slots_cap = limits.max_waves.min(MAX_WAVES);
    // An adaptive family with Sparks: an unpinned launch, or a resolved
    // deployment passing back more slots than waves, admits `slots` slots of
    // the widest unit (reserve / 2) and starts at the estimate's depth (#762).
    let resolved_adaptive = explicit.slots.is_some_and(|slots| explicit.waves.is_some_and(|waves| slots > waves));
    let unpinned = explicit.waves.is_none() && explicit.span_rows.is_none() && explicit.slots.is_none();
    if limits.adaptive && spark_ranks > 0 && slots_cap > 1 && (unpinned || resolved_adaptive) {
        let slots = explicit.slots.unwrap_or(slots_cap);
        let waves = explicit.waves.unwrap_or(maximum).min(slots);
        if slots > slots_cap || waves < 1 {
            return Err(PrefillError::Slots { slots, waves, max_waves: slots_cap });
        }
        let reserve = explicit.reserve_rows.unwrap_or(RESERVE_ROWS);
        if reserve > RESERVE_ROWS || reserve % ROW_PAGE != 0 || reserve == 0 {
            return Err(PrefillError::IntakeSpan { span_rows: reserve, slot_rows: reserve, max_span_rows: RESERVE_ROWS });
        }
        let cap = unit_rows_in(2, reserve).min(ADAPTIVE_UNIT_ROWS).min(limits.max_unit_rows);
        // A planner-fit reserve admits its own widest unit (depth 2's) exactly;
        // the standard reserve the widest compiled capacity under half of it.
        let unit = explicit.unit_rows.unwrap_or_else(|| if explicit.reserve_rows.is_some() { cap } else {
            limits.capacities.iter().copied().map(u64::from)
                .filter(|&rows| rows >= 64 && rows % 64 == 0 && rows <= cap).max().unwrap_or(0) });
        if (super::PrefillShape { unit_rows: unit, waves: 1 }).validate().is_err() {
            return Err(PrefillError::InvalidShape { unit_rows: unit, waves });
        }
        if unit > cap {
            return Err(PrefillError::ExecutorCapacity { unit_rows: unit, waves,
                max_unit_rows: cap, max_waves: slots_cap });
        }
        // The starting depth runs a valid static shape at its own cut.
        let start = validate_shape(super::wave_control::unit_for_in(waves, unit, reserve), waves, limits, None)?;
        let span = super::wave_control::reserve_rows_in(slots, unit, reserve);
        let max_span_rows = limits.max_span_rows.min(RESERVE_ROWS + 64);
        if explicit.span_rows.is_some_and(|given| given != span) || span > max_span_rows {
            return Err(PrefillError::IntakeSpan { span_rows: explicit.span_rows.unwrap_or(span),
                slot_rows: span, max_span_rows });
        }
        return Ok(PrefillPlan { unit_rows: unit, waves: start.waves, span_rows: span, slots });
    }
    let mut waves = explicit.waves.unwrap_or(maximum);
    loop {
        let budget = unit_rows(waves).min(limits.max_unit_rows)
            .min(limits.max_span_rows.min(RESERVE_ROWS + 64) / waves.max(1));
        let automatic_budget = budget.min(limits.initial_unit_rows.unwrap_or(budget));
        let rows = explicit.unit_rows.unwrap_or_else(|| limits.capacities.iter().copied()
            .map(u64::from).filter(|&rows| rows >= 64 && rows % 64 == 0 && rows <= automatic_budget)
            .max().unwrap_or(0));
        let shape = validate_shape(rows, waves, limits, None);
        if shape.is_ok() {
            // A resolved deployment's slots equal its waves (pinned or static);
            // more slots are only an adaptive family's (above).
            if let Some(slots) = explicit.slots.filter(|&slots| slots != waves) {
                return Err(PrefillError::Slots { slots, waves, max_waves: slots_cap });
            }
            return validate_shape(rows, waves, limits, explicit.span_rows);
        }
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
    Ok(PrefillPlan { unit_rows: rows, waves, span_rows, slots: waves })
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

    /// TJ 2026-10-28: a Spark-free prefill path (no Spark ranks in prefill:
    /// GLM Flash / dense Qwen 0-Spark, or a family planning its fully
    /// onboarded prefill path as such) runs one window at the widest unit,
    /// for static and adaptive families alike; an explicit depth still pins.
    #[test]
    fn spark_free_prefill_is_one_window_at_the_widest_unit() {
        for adaptive in [false, true] {
            let limits = PrefillLimits { adaptive, ..Default::default() };
            let plan = prefill_shape(0, limits, PrefillOverrides::default()).unwrap();
            assert_eq!((plan.unit_rows, plan.waves, plan.slots, plan.span_rows), (4096, 1, 1, 4096), "adaptive={adaptive}");
            let pinned = prefill_shape(0, limits, PrefillOverrides { waves: Some(3), ..Default::default() }).unwrap();
            assert_eq!((pinned.unit_rows, pinned.waves, pinned.slots), (2752, 3, 3), "adaptive={adaptive}");
        }
        // Depth 1 is a valid explicit pin with Sparks too.
        let one = prefill_shape(4, PrefillLimits { adaptive: true, ..Default::default() },
            PrefillOverrides { unit_rows: Some(4096), waves: Some(1), ..Default::default() }).unwrap();
        assert_eq!((one.unit_rows, one.waves, one.slots, one.span_rows), (4096, 1, 1, 4096));
    }

    fn measured(waves: u64) -> PrefillLimits {
        PrefillLimits { estimate: PrefillEstimate::MeasuredWaves { waves, reference_ranks: 6, larger_topology_waves: waves, source: "test comparison" },
            ..Default::default() }
    }

    #[test]
    fn units_are_automatic_splits_of_the_reserve() {
        // The standard reserve reproduces the shipped units and its 8256-row charge.
        assert_eq!((2..=4).map(|k| unit_rows_in(k, RESERVE_ROWS)).collect::<Vec<_>>(), [4096, 2752, 2048]);
        assert_eq!(charge_rows(RESERVE_ROWS), 8256);
        assert_eq!(reserve_within(u64::MAX), RESERVE_ROWS);
        assert_eq!(reserve_within(8256), RESERVE_ROWS);
        // Any reserve: page-aligned units covering it, charge at most reserve + 192.
        for reserve in (64..=RESERVE_ROWS).step_by(64) {
            for k in 2..=MAX_WAVES {
                let unit = unit_rows_in(k, reserve);
                assert!(unit % ROW_PAGE == 0 && k * unit >= reserve && unit <= reserve, "{reserve}/{k}: {unit}");
            }
            assert!(charge_rows(reserve) <= reserve + 3 * ROW_PAGE, "{reserve}");
            // The largest reserve whose charge fits a span is never below what the span holds.
            let fit = reserve_within(charge_rows(reserve));
            assert!(fit >= reserve && charge_rows(fit) <= charge_rows(reserve), "{reserve} -> {fit}");
        }
        // A card holding 8000 rows admits 7872: units 3968 / 2624 / 1984 (charge 7936).
        assert_eq!(reserve_within(8000), 7872);
        assert_eq!((2..=4).map(|k| unit_rows_in(k, 7872)).collect::<Vec<_>>(), [3968, 2624, 1984]);
        assert_eq!(charge_rows(7872), 7936);
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
        assert_eq!((2..=4).map(|k| unit_rows_in(k, 3840)).collect::<Vec<_>>(), [1920, 1280, 960]);
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
                assert_eq!(shape, PrefillPlan { unit_rows: unit_rows(k), waves: k, span_rows: k * unit_rows(k), slots: k });
            }
        }
    }

    #[test]
    fn estimate_is_bounded_by_hard_capacity_and_memory() {
        let limits = PrefillLimits { max_waves: 2, ..measured(4) };
        assert_eq!(prefill_shape(6, limits, Default::default()).unwrap().waves, 2);
        let limits = PrefillLimits { max_unit_rows: 1024, max_span_rows: 2048, ..measured(3) };
        let shape = prefill_shape(6, limits, Default::default()).unwrap();
        assert_eq!(shape, PrefillPlan { unit_rows: 256, waves: 3, span_rows: 768, slots: 3 });
        let shape = prefill_shape(6, limits, PrefillOverrides { unit_rows: Some(1024), ..Default::default() }).unwrap();
        assert_eq!(shape, PrefillPlan { unit_rows: 1024, waves: 2, span_rows: 2048, slots: 2 });
        let limits = PrefillLimits { max_span_rows: 128, capacities: &[256, 1024, 4096], ..measured(4) };
        assert!(prefill_shape(6, limits, Default::default()).is_err());
    }

    #[test]
    fn initial_unit_preference_only_changes_automatic_rows() {
        let limits = PrefillLimits { initial_unit_rows: Some(2048), ..measured(3) };
        assert_eq!(prefill_shape(6, measured(3), Default::default()).unwrap(),
            PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256, slots: 3 });
        assert_eq!(prefill_shape(6, limits, Default::default()).unwrap(),
            PrefillPlan { unit_rows: 2048, waves: 3, span_rows: 6144, slots: 3 });
        for (rows, waves) in [(256, 3), (2752, 3), (4096, 2)] {
            assert_eq!(prefill_shape(6, limits, PrefillOverrides {
                unit_rows: Some(rows), waves: Some(waves), ..Default::default()
            }).unwrap(), PrefillPlan { unit_rows: rows, waves, span_rows: rows * waves, slots: waves });
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
            PrefillPlan { unit_rows: 256, waves: 3, span_rows: 768, slots: 3 });
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
            PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256, slots: 3 });
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
    fn adaptive_plans_admit_every_slot_inside_the_same_reserve() {
        // Four slots of the widest unit (reserve / 2), starting at the estimate.
        let limits = PrefillLimits { adaptive: true, ..measured(3) };
        let plan = prefill_shape(6, limits, Default::default()).unwrap();
        assert_eq!(plan, PrefillPlan { unit_rows: 4096, waves: 3, span_rows: 8256, slots: 4 });
        assert_eq!(plan.units(), super::super::PrefillShape { unit_rows: 2752, waves: 3 });
        assert_eq!(plan.admitted(), super::super::PrefillShape { unit_rows: 4096, waves: 4 });
        assert!(plan.adaptive());
        // Every family starts where its estimate says, admitting the same unit.
        for (waves, start) in [(2, 4096), (4, 2048)] {
            let plan = prefill_shape(6, PrefillLimits { adaptive: true, ..measured(waves) }, Default::default()).unwrap();
            assert_eq!((plan.unit_rows, plan.waves, plan.span_rows, plan.slots, plan.units().unit_rows),
                (4096, waves, 8256, 4, start));
        }
        // A smaller compiled capacity bounds the admitted unit.
        let capped = PrefillLimits { max_unit_rows: 2048, ..limits };
        assert_eq!(prefill_shape(6, capped, Default::default()).unwrap(),
            PrefillPlan { unit_rows: 2048, waves: 3, span_rows: 8192, slots: 4 });
        // Pins and spans stay static; no Sparks, one wave.
        assert_eq!(prefill_shape(6, limits, PrefillOverrides { waves: Some(3), ..Default::default() }).unwrap(),
            PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256, slots: 3 });
        assert_eq!(prefill_shape(0, limits, Default::default()).unwrap().slots, 1);
        let shallow = PrefillLimits { max_waves: 2, ..limits };
        assert_eq!(prefill_shape(6, shallow, Default::default()).unwrap(),
            PrefillPlan { unit_rows: 4096, waves: 2, span_rows: 8192, slots: 2 });
        // A resolved deployment's slots come back with its waves: adaptive
        // when more, pinned when equal, refused for static families.
        let resolved = PrefillOverrides { unit_rows: Some(4096), waves: Some(3), span_rows: Some(8256), slots: Some(4), reserve_rows: None };
        assert_eq!(prefill_shape(6, limits, resolved).unwrap(), plan);
        let pinned = PrefillOverrides { unit_rows: Some(2752), waves: Some(3), span_rows: Some(8256), slots: Some(3), reserve_rows: None };
        assert_eq!(prefill_shape(6, limits, pinned).unwrap().slots, 3);
        assert!(prefill_shape(6, measured(3), resolved).is_err(), "a static family has no 4096-row three-wave shape");
        assert!(matches!(prefill_shape(6, measured(3), PrefillOverrides { slots: Some(4), ..pinned }),
            Err(PrefillError::Slots { .. })));
        assert!(matches!(prefill_shape(6, limits, PrefillOverrides { span_rows: Some(8192), ..resolved }),
            Err(PrefillError::IntakeSpan { .. })));
        assert!(matches!(prefill_shape(6, limits, PrefillOverrides { slots: Some(5), ..resolved }),
            Err(PrefillError::Slots { .. })));
        assert!(prefill_shape(6, limits, PrefillOverrides { unit_rows: Some(8192), ..resolved }).is_err());
        assert!(prefill_shape(6, limits, PrefillOverrides { unit_rows: Some(2753), ..resolved }).is_err());
    }

    #[test]
    fn qwen_mapped_ple_ring_is_a_hard_two_wave_constraint() {
        let limits = crate::families::qwen4::prefill_limits(false);
        assert_eq!(prefill_shape(6, limits, Default::default()).unwrap(),
            PrefillPlan { unit_rows: 4096, waves: 2, span_rows: 8192, slots: 2 });
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
