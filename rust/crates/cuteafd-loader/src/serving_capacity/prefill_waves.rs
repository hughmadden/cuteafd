//! Prefill wave count (#762, TJ 2026-10-25). Every family reserves the same
//! 8192 prefill rows (today's 2 x 4096). The reserve is split into k waves of
//! `reserve / k` rows, each run on its exact EXL3 capacity (2 -> 4096,
//! 3 -> 2752, 4 -> 2048). k hides raptor's turnaround T behind the Sparks'
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
