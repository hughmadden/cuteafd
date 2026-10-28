//! The prefill shape an engine runs on the shared executor: `waves` units of
//! `unit_rows` in flight, fixed at startup (TJ 2026-10-28: three units of the
//! reserve with Sparks; the adaptive controller of #762 was removed). One
//! prefill call is one window cycle of at most `waves * unit_rows` rows.

/// The prefill shape of an engine (fixed for its lifetime).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillDepth { pub waves: usize, pub unit_rows: usize }

impl PrefillDepth {
    pub(crate) fn new(waves: usize, unit_rows: usize) -> anyhow::Result<Self> {
        anyhow::ensure!((1..=4).contains(&waves) && (64..=4096).contains(&unit_rows) && unit_rows % 64 == 0,
            "prefill shape {waves} x {unit_rows} rows outside 1..=4 x 64..=4096 (page aligned)");
        tracing::info!(waves, unit_rows, span_rows = waves * unit_rows, "prefill shape");
        Ok(Self { waves, unit_rows })
    }
    /// Rows every per-slot plane spans: one unit per slot.
    pub(crate) fn reserve_rows(&self) -> usize { self.waves * self.unit_rows }
    /// Longest prefill call: the slots' units.
    pub(crate) fn call_rows(&self) -> usize { self.reserve_rows() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_call_is_the_slots_units() -> anyhow::Result<()> {
        let depth = PrefillDepth::new(3, 2752)?;
        assert_eq!((depth.reserve_rows(), depth.call_rows()), (8256, 8256));
        assert!(PrefillDepth::new(5, 2048).is_err());
        assert!(PrefillDepth::new(3, 2753).is_err());
        assert!(PrefillDepth::new(1, 8192).is_err());
        Ok(())
    }
}
