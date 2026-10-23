//! BF16 coordinator receive planes shared by Spark-backed families, and the
//! shared prefill unit shape (PLAN "Prefill contract").
/// One full-capacity BF16 partial plane per rank and independently queued lane.
pub fn spark_intake_bytes(lanes: u64, ranks: u64, rows: u64, hidden: u64) -> u64 {
    lanes * ranks * rows * hidden * 2
}

/// Return slots for `waves` outstanding prefill waves of `unit_rows` each on
/// one transport: one BF16 plane per rank per slot.
pub fn prefill_return_bytes(waves: u64, unit_rows: u64, ranks: u64, hidden: u64) -> u64 {
    spark_intake_bytes(waves, ranks, unit_rows, hidden)
}

/// Prefill row units and how many are in flight on one lane and transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillShape {
    pub unit_rows: u64,
    pub waves: u64,
}

impl PrefillShape {
    /// The contract default: 2048-row units, two in flight.
    pub const DEFAULT: Self = Self { unit_rows: 2048, waves: 2 };

    pub fn validate(self) -> anyhow::Result<Self> {
        anyhow::ensure!((64..=4096).contains(&self.unit_rows) && self.unit_rows % 64 == 0
            && (1..=4).contains(&self.waves), "prefill unit {} rows x {} waves outside 64..=4096 (page aligned) x 1..=4",
            self.unit_rows, self.waves);
        Ok(self)
    }

    /// Rows of every slot-owned plane: one unit per wave in flight.
    pub fn slot_rows(self) -> u64 { self.unit_rows * self.waves }

    /// The admission ladder from this default: halve the unit (2048 → 1024 →
    /// 512) at the same wave count until it admits. Never below 512 rows.
    pub fn ladder(self) -> impl Iterator<Item = Self> {
        std::iter::successors(Some(self), |shape| (shape.unit_rows > 512)
            .then(|| Self { unit_rows: (shape.unit_rows / 2).max(512), waves: shape.waves }))
    }
}

/// Select the largest admitting unit without changing the number of waves.
/// Only fully automatic context/pool choices enforce the agentic pool floor;
/// explicit requests retain the family's strict admission behavior.
pub fn select_prefill_shape<T>(
    shape: PrefillShape,
    automatic_pool: bool,
    mut admit: impl FnMut(PrefillShape) -> anyhow::Result<T>,
    pool_tokens: impl Fn(&T) -> u64,
    is_fit_error: impl Fn(&anyhow::Error) -> bool,
) -> anyhow::Result<(PrefillShape, T)> {
    use anyhow::Context;
    use crate::placement::AGENTIC_FLOOR_TOKENS;
    let mut failure = None;
    for candidate in shape.validate()?.ladder() {
        let error = match admit(candidate) {
            Ok(result) => {
                let pool = pool_tokens(&result);
                if !automatic_pool || pool >= AGENTIC_FLOOR_TOKENS {
                    return Ok((candidate, result));
                }
                anyhow::anyhow!("prefill {} rows x {} waves admits an automatic pool of {pool} tokens, below the {AGENTIC_FLOOR_TOKENS}-token agentic floor",
                    candidate.unit_rows, candidate.waves)
            }
            Err(error) if is_fit_error(&error) => error,
            Err(error) => return Err(error),
        };
        tracing::info!(unit_rows = candidate.unit_rows, waves = candidate.waves, %error,
            "prefill shape did not admit; trying the next shared ladder unit");
        failure = Some(error);
    }
    Err(failure.context("prefill admission ladder is empty")?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn glm_flash_intake_charge_is_unchanged() {
        for lanes in [1, 2, 4] { for ranks in [0, 4, 6] { for rows in [64, 4096] {
            assert_eq!(spark_intake_bytes(lanes, ranks, rows, 6144), lanes * ranks * rows * 6144 * 2);
        } } }
        assert_eq!(spark_intake_bytes(1, 6, 4096, 3584), 176_160_768);
    }

    #[test]
    fn two_2048_row_return_slots_cost_one_4096_row_intake() {
        assert_eq!(prefill_return_bytes(2, 2048, 6, 3584), spark_intake_bytes(1, 6, 4096, 3584));
        assert_eq!(PrefillShape::DEFAULT.slot_rows(), 4096);
    }

    #[test]
    fn automatic_pool_steps_down_but_explicit_pool_keeps_the_largest_unit() {
        use crate::placement::AGENTIC_FLOOR_TOKENS;
        for automatic_pool in [true, false] {
            let mut tried = Vec::new();
            let (selected, pool) = select_prefill_shape(PrefillShape { unit_rows: 4096, waves: 2 },
                automatic_pool, |shape| {
                    tried.push(shape);
                    Ok(if shape.unit_rows == 4096 { 200_000 } else { AGENTIC_FLOOR_TOKENS })
                }, |&pool| pool, |_| false).unwrap();
            let expected = if automatic_pool { 2048 } else { 4096 };
            assert_eq!(selected, PrefillShape { unit_rows: expected, waves: 2 });
            assert_eq!(pool, if automatic_pool { AGENTIC_FLOOR_TOKENS } else { 200_000 });
            assert_eq!(tried.len(), if automatic_pool { 2 } else { 1 });
        }
    }

    #[test]
    fn automatic_floor_exhaustion_and_non_fit_errors_fail_closed() {
        let mut tried = Vec::new();
        let error = select_prefill_shape(PrefillShape { unit_rows: 4096, waves: 3 }, true,
            |shape| { tried.push(shape); Ok(200_000) }, |&pool| pool, |_| false).unwrap_err();
        assert!(error.to_string().contains("262144-token agentic floor"));
        assert_eq!(tried.iter().map(|shape| shape.unit_rows).collect::<Vec<_>>(), [4096, 2048, 1024, 512]);
        assert!(tried.iter().all(|shape| shape.waves == 3));
        for automatic_pool in [true, false] {
            let mut calls = 0;
            let error = select_prefill_shape(PrefillShape::DEFAULT, automatic_pool, |_| {
                calls += 1;
                Err::<u64, _>(anyhow::anyhow!("explicit pool does not fit"))
            }, |&pool| pool, |_| false).unwrap_err();
            assert_eq!(calls, 1);
            assert_eq!(error.to_string(), "explicit pool does not fit");
        }
    }

    #[test]
    fn ladder_halves_the_unit_down_to_512() -> anyhow::Result<()> {
        let rows = |shape: PrefillShape| shape.ladder().map(|s| (s.unit_rows, s.waves)).collect::<Vec<_>>();
        assert_eq!(rows(PrefillShape::DEFAULT), [(2048, 2), (1024, 2), (512, 2)]);
        assert_eq!(rows(PrefillShape { unit_rows: 4096, waves: 1 }), [(4096, 1), (2048, 1), (1024, 1), (512, 1)]);
        assert_eq!(rows(PrefillShape { unit_rows: 512, waves: 3 }), [(512, 3)]);
        assert!(PrefillShape { unit_rows: 100, waves: 2 }.validate().is_err());
        assert!(PrefillShape { unit_rows: 2048, waves: 5 }.validate().is_err());
        PrefillShape::DEFAULT.validate()?;
        Ok(())
    }
}
