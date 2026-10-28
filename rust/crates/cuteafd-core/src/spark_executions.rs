//! Typed Spark compute owners: one small decode execution per decode lane and
//! exactly one prefill execution per Spark. Planner, serve and golden build
//! the same value through [`SparkExecutions::admit`]; the worker allocates each
//! owner once and validates every request against its owner before launch.
use serde::{Deserialize, Serialize};

/// Most decode lanes a family runs (each owns one decode execution).
pub const MAX_SPARK_DECODE_LANES: usize = 2;
/// Most prefill units retained in flight on the single prefill owner.
pub const MAX_BUFFERED_PREFILL_UNITS: u8 = 4;

/// One independently scheduled decode lane. Capacities come from the family's
/// admitted decode extent and its covering package export, never prefill rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeExecution {
    pub capacity: u32,
    pub compiled_capacity: u32,
}

/// Exactly one scratch/stream owner per Spark. Network units retain independent
/// request/reply storage but enqueue on this stream without awaiting earlier replies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillExecution {
    /// Largest unit this owner can run. Static admission uses `unit_rows`;
    /// a future dynamic repartition must explicitly admit a larger extent.
    pub capacity: u32,
    /// Family package's covering export for that admitted extent.
    pub compiled_capacity: u32,
    /// Rows of one selected unit (the reserve divided by k). Per-unit input
    /// landing and reply staging are sized here, and prefill requests are
    /// bounded here.
    pub unit_rows: u32,
    /// Retained in-flight I/O units (k); one stream/scratch owner. Eligible
    /// segmented recipes size that arena at the admitted sum, not k arenas.
    pub buffered_units: u8,
}

/// Typed compute owners, independent of persistent endpoints and network rings.
/// The daemon allocates each decode entry once and the singleton prefill owner
/// once. Per-unit completion events retain all mapped I/O until its own reply.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparkExecutions {
    /// Vector index is the request's execution lane for Decode and MtpVerify.
    pub decode: Vec<DecodeExecution>,
    /// Prefill and Benchmark rows use this owner, irrespective of endpoint lane.
    pub prefill: PrefillExecution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionOwner { Decode(usize), Prefill }

/// One owner's extents for workspace accounting: singleton scratch at `compiled`,
/// stream-ordered device I/O at `capacity`, and pinned route staging of
/// `staging_rows` retained for each of `buffered_units` queued units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerExtent {
    pub owner: ExecutionOwner,
    pub capacity: u32,
    pub compiled: u32,
    pub staging_rows: u32,
    pub buffered_units: u8,
}

/// Segmented code can address four full wire units; allocations use only the
/// admitted row sum, independently of this compile-time metadata bound.
pub const SEGMENT_MAX_ROWS: u32 = 4 * 4096;

impl OwnerExtent {
    pub fn segment_rows(self) -> Option<u32> {
        match self.owner {
            ExecutionOwner::Prefill if self.buffered_units > 1 =>
                self.staging_rows.checked_mul(u32::from(self.buffered_units))
                    .filter(|&rows| rows <= SEGMENT_MAX_ROWS),
            _ => None,
        }
    }
}

/// Exact streamed intermediate arena with the compiled tile-table offset.
/// `act_row_bytes` is installed recipe metadata (BF16 or MXFP8 down input).
/// This does not include row descriptors, wire frames or singleton scratch.
pub fn segment_scratch_bytes(hidden: u32, experts: u32, topk: u32, act_row_bytes: u32,
    rows: u32, maximum: u32) -> Option<u64> {
    if hidden == 0 || experts == 0 || topk == 0 || act_row_bytes == 0
        || rows == 0 || rows > maximum || maximum > SEGMENT_MAX_ROWS { return None; }
    let align = |n: u64| n.checked_add(1023).map(|n| n / 1024 * 1024);
    let tiles = |p: u64| p.div_ceil(128) + u64::from(experts).min(p);
    let pairs = u64::from(rows) * u64::from(topk);
    let bound = u64::from(maximum) * u64::from(topk);
    let meta = align((4 + 3 * u64::from(experts) + tiles(bound)).checked_mul(4)?)?;
    let routes = align(pairs.checked_mul(4)?)?.checked_mul(2)?;
    let act = align(tiles(pairs).checked_mul(128)?.checked_mul(u64::from(act_row_bytes))?)?;
    let output = align(pairs.checked_mul(u64::from(hidden))?.checked_mul(2)?)?;
    meta.checked_add(routes)?.checked_add(act)?.checked_add(output)
}

/// Bytes one owner holds at its extents, from the planner's census or the
/// worker's installed artifacts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OwnerBytes {
    pub scratch: u64,
    pub io: u64,
    /// Per queued unit; multiplied by `buffered_units` only.
    pub pinned_staging: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExecutionAdmissionError {
    #[error("Spark decode lane count {lanes} must be 1..={maximum}")]
    DecodeLanes { lanes: usize, maximum: usize },
    #[error("{owner} live capacity {capacity} requires a covering export, got {compiled}")]
    Capacity { owner: &'static str, capacity: u32, compiled: u32 },
    #[error("{owner} capacity {rows} has no covering export in the package ladder {exports:?}")]
    Uncovered { owner: &'static str, rows: u32, exports: Vec<u32> },
    #[error("Spark prefill unit rows {unit_rows} must be 1..={capacity}")]
    UnitRows { unit_rows: u32, capacity: u32 },
    #[error("Spark prefill buffered units {units} must be 1..={maximum}")]
    BufferedUnits { units: u8, maximum: u8 },
    #[error("Spark decode lane {lane} is outside {lanes} admitted lanes")]
    DecodeLane { lane: u32, lanes: usize },
    #[error("{owner} rows {rows} exceed admitted capacity {capacity}, or are empty")]
    Rows { owner: &'static str, rows: u32, capacity: u32 },
}

/// The smallest export in `exports` that covers `rows` live rows.
pub fn covering_export(exports: &[u32], rows: u32) -> Option<u32> {
    exports.iter().copied().filter(|&export| export >= rows).min()
}

impl SparkExecutions {
    /// The one admission constructor: `decode_rows` per decode lane, the
    /// admitted prefill extent (`prefill_rows`), the selected unit and the
    /// retained unit count, with every compiled capacity taken from
    /// the package's export ladder. An uncovered capacity is a typed refusal,
    /// never a silently larger or smaller owner.
    pub fn admit(decode_rows: &[u32], prefill_rows: u32, unit_rows: u32, buffered_units: u8, exports: &[u32])
        -> Result<Self, ExecutionAdmissionError> {
        let cover = |owner, rows| covering_export(exports, rows).filter(|_| rows > 0)
            .ok_or_else(|| ExecutionAdmissionError::Uncovered { owner, rows, exports: exports.to_vec() });
        let decode = decode_rows.iter().map(|&rows| Ok(DecodeExecution { capacity: rows,
            compiled_capacity: cover("decode", rows)? })).collect::<Result<Vec<_>, _>>()?;
        let plan = Self { decode, prefill: PrefillExecution { capacity: prefill_rows,
            compiled_capacity: cover("prefill", prefill_rows)?, unit_rows, buffered_units } };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), ExecutionAdmissionError> {
        if self.decode.is_empty() || self.decode.len() > MAX_SPARK_DECODE_LANES {
            return Err(ExecutionAdmissionError::DecodeLanes { lanes: self.decode.len(), maximum: MAX_SPARK_DECODE_LANES });
        }
        let check = |owner, capacity, compiled| {
            if capacity == 0 || compiled < capacity || compiled > 4096 {
                Err(ExecutionAdmissionError::Capacity { owner, capacity, compiled })
            } else { Ok(()) }
        };
        for lane in &self.decode { check("decode", lane.capacity, lane.compiled_capacity)?; }
        check("prefill", self.prefill.capacity, self.prefill.compiled_capacity)?;
        if !(1..=self.prefill.capacity).contains(&self.prefill.unit_rows) {
            return Err(ExecutionAdmissionError::UnitRows { unit_rows: self.prefill.unit_rows,
                capacity: self.prefill.capacity });
        }
        if !(1..=MAX_BUFFERED_PREFILL_UNITS).contains(&self.prefill.buffered_units) {
            return Err(ExecutionAdmissionError::BufferedUnits { units: self.prefill.buffered_units,
                maximum: MAX_BUFFERED_PREFILL_UNITS });
        }
        Ok(())
    }

    /// Every owner's extents: each decode owner holds one unit of its own
    /// capacity; the prefill owner holds `buffered_units` units of `unit_rows`.
    pub fn owners(&self) -> impl Iterator<Item = OwnerExtent> + '_ {
        self.decode.iter().enumerate().map(|(lane, owner)| OwnerExtent { owner: ExecutionOwner::Decode(lane),
            capacity: owner.capacity, compiled: owner.compiled_capacity, staging_rows: owner.capacity, buffered_units: 1 })
            .chain(std::iter::once(OwnerExtent { owner: ExecutionOwner::Prefill, capacity: self.prefill.capacity,
                compiled: self.prefill.compiled_capacity, staging_rows: self.prefill.unit_rows,
                buffered_units: self.prefill.buffered_units }))
    }

    /// The one workspace sum shared by planner and worker: scratch and device
    /// I/O once per owner, pinned staging times its queued units. `None` on
    /// overflow; `bytes` errors propagate.
    pub fn workspace_bytes<E>(&self, mut bytes: impl FnMut(OwnerExtent) -> Result<OwnerBytes, E>)
        -> Result<Option<crate::expert_geometry::SparkWorkspaceBytes>, E> {
        let owners = self.owners().map(|extent| bytes(extent).map(|owner| crate::expert_geometry::SparkOwnerWorkspace {
            scratch_bytes: owner.scratch, io_bytes: owner.io, pinned_staging_bytes: owner.pinned_staging,
            buffered_units: extent.buffered_units })).collect::<Result<Vec<_>, E>>()?;
        Ok(crate::expert_geometry::spark_owner_workspace_bytes(owners))
    }

    /// Largest live rows any owner admits.
    pub fn capacity(&self) -> u32 {
        self.decode.iter().map(|lane| lane.capacity).fold(self.prefill.capacity, u32::max)
    }

    /// Largest compiled export any owner loads.
    pub fn compiled_capacity(&self) -> u32 {
        self.decode.iter().map(|lane| lane.compiled_capacity).fold(self.prefill.compiled_capacity, u32::max)
    }

    /// Validate a request against its typed owner before any GPU launch:
    /// `decode_lane` is `Some` for Decode/MtpVerify rows and `None` for
    /// Prefill/Benchmark rows. An endpoint lane never selects prefill scratch.
    pub fn owner(&self, decode_lane: Option<u32>, rows: u32) -> Result<ExecutionOwner, ExecutionAdmissionError> {
        self.validate()?;
        let (owner, capacity, name) = match decode_lane {
            None => (ExecutionOwner::Prefill, self.prefill.unit_rows, "prefill"),
            Some(lane) => {
                let entry = self.decode.get(lane as usize)
                    .ok_or(ExecutionAdmissionError::DecodeLane { lane, lanes: self.decode.len() })?;
                (ExecutionOwner::Decode(lane as usize), entry.capacity, "decode")
            }
        };
        if rows == 0 || rows > capacity {
            return Err(ExecutionAdmissionError::Rows { owner: name, rows, capacity });
        }
        Ok(owner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LADDER: [u32; 8] = crate::EXL3_CAPACITIES;

    #[test]
    fn segmented_allocation_uses_live_sum_and_keeps_decode_unchanged() {
        let plan = SparkExecutions::admit(&[64], 4096, 2752, 3, &LADDER).unwrap();
        let owners = plan.owners().collect::<Vec<_>>();
        assert_eq!(owners[0].segment_rows(), None);
        assert_eq!(owners[1].segment_rows(), Some(8256));
        let size = |rows| segment_scratch_bytes(4096, 256, 8, 768, rows, SEGMENT_MAX_ROWS).unwrap();
        assert!(size(8256) < size(SEGMENT_MAX_ROWS));
        assert_eq!(size(8256), 617_493_504);
        assert!(segment_scratch_bytes(4096, 256, 8, 768, 16385, SEGMENT_MAX_ROWS).is_none());
        assert!(segment_scratch_bytes(4096, 256, 8, 768, 0, SEGMENT_MAX_ROWS).is_none());
    }

    #[test]
    fn admission_takes_covering_exports_from_the_package_ladder() {
        let plan = SparkExecutions::admit(&[64, 80], 4096, 2752, 3, &LADDER).unwrap();
        assert_eq!(plan.decode, [DecodeExecution { capacity: 64, compiled_capacity: 80 },
            DecodeExecution { capacity: 80, compiled_capacity: 80 }]);
        assert_eq!(plan.prefill, PrefillExecution { capacity: 4096, compiled_capacity: 4096,
            unit_rows: 2752, buffered_units: 3 });
        assert_eq!((plan.capacity(), plan.compiled_capacity()), (4096, 4096));
        let encoded = serde_json::to_vec(&plan).unwrap();
        assert_eq!(serde_json::from_slice::<SparkExecutions>(&encoded).unwrap(), plan);
        // Native packages export only m1/m80/main.
        let native = SparkExecutions::admit(&[64], 2752, 2752, 1, &[1, 80, 4096]).unwrap();
        assert_eq!((native.decode[0].compiled_capacity, native.prefill.compiled_capacity), (80, 4096));
    }

    #[test]
    fn uncovered_decode_or_prefill_capacity_is_a_typed_refusal() {
        assert_eq!(SparkExecutions::admit(&[128], 4096, 4096, 1, &[1, 80, 4096]).unwrap().decode[0].compiled_capacity, 4096);
        assert!(matches!(SparkExecutions::admit(&[96], 1024, 1024, 1, &[1, 80, 1024]).unwrap().decode[0],
            DecodeExecution { compiled_capacity: 1024, .. }));
        assert_eq!(SparkExecutions::admit(&[81], 64, 64, 1, &[1, 64, 80]).unwrap_err(),
            ExecutionAdmissionError::Uncovered { owner: "decode", rows: 81, exports: vec![1, 64, 80] });
        assert!(matches!(SparkExecutions::admit(&[64], 4097, 4097, 1, &LADDER),
            Err(ExecutionAdmissionError::Uncovered { owner: "prefill", rows: 4097, .. })));
        assert!(matches!(SparkExecutions::admit(&[0], 4096, 4096, 1, &LADDER),
            Err(ExecutionAdmissionError::Uncovered { owner: "decode", rows: 0, .. })));
        assert!(matches!(SparkExecutions::admit(&[], 4096, 4096, 1, &LADDER), Err(ExecutionAdmissionError::DecodeLanes { .. })));
        assert!(matches!(SparkExecutions::admit(&[64; 3], 4096, 4096, 1, &LADDER), Err(ExecutionAdmissionError::DecodeLanes { .. })));
        for units in [0, 5] {
            assert!(matches!(SparkExecutions::admit(&[64], 4096, 2048, units, &LADDER),
                Err(ExecutionAdmissionError::BufferedUnits { .. })));
        }
        assert_eq!(SparkExecutions::admit(&[64], 2048, 2752, 3, &LADDER).unwrap_err(),
            ExecutionAdmissionError::UnitRows { unit_rows: 2752, capacity: 2048 });
    }

    #[test]
    fn workspace_charges_one_prefill_scratch_and_divides_the_reserve_into_units() {
        let bytes = |plan: &SparkExecutions| plan.workspace_bytes(|extent| Ok::<_, ()>(OwnerBytes {
            scratch: u64::from(extent.compiled) * 1000, io: u64::from(extent.capacity) * 10,
            pinned_staging: u64::from(extent.staging_rows) })).unwrap().unwrap();
        let mut totals = Vec::new();
        for (unit, units) in [(4096, 1), (4096, 2), (2752, 3), (2048, 4)] {
            let plan = SparkExecutions::admit(&[64, 64], 4096, unit, units, &LADDER).unwrap();
            let total = bytes(&plan);
            assert_eq!(total.device_bytes, 2 * (80_000 + 640) + 4_096_000 + 40_960);
            assert_eq!(total.pinned_bytes, 2 * 64 + u64::from(unit) * u64::from(units));
            totals.push(total.device_bytes);
        }
        // More waves never add scratch: the reserve is divided, not multiplied.
        assert!(totals.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn typed_owners_keep_decode_prefill_and_buffered_io_independent() {
        let plan = SparkExecutions { decode: vec![
            DecodeExecution { capacity: 80, compiled_capacity: 80 },
            DecodeExecution { capacity: 64, compiled_capacity: 256 }],
            prefill: PrefillExecution { capacity: 4096, compiled_capacity: 4096, unit_rows: 2752, buffered_units: 3 } };
        plan.validate().unwrap();
        assert_eq!(plan.owner(Some(0), 80).unwrap(), ExecutionOwner::Decode(0));
        assert_eq!(plan.owner(Some(1), 64).unwrap(), ExecutionOwner::Decode(1));
        assert_eq!(plan.owner(None, 2752).unwrap(), ExecutionOwner::Prefill);
        assert!(matches!(plan.owner(Some(1), 65), Err(ExecutionAdmissionError::Rows { .. })));
        assert!(matches!(plan.owner(Some(2), 1), Err(ExecutionAdmissionError::DecodeLane { .. })));
        assert!(matches!(plan.owner(None, 2753), Err(ExecutionAdmissionError::Rows { capacity: 2752, .. })));
        assert!(plan.owner(None, 0).is_err());
        let mut invalid = plan.clone(); invalid.decode[0].compiled_capacity = 64;
        assert!(matches!(invalid.validate(), Err(ExecutionAdmissionError::Capacity { owner: "decode", .. })));
        invalid = plan.clone(); invalid.prefill.compiled_capacity = 2048;
        assert!(matches!(invalid.validate(), Err(ExecutionAdmissionError::Capacity { owner: "prefill", .. })));
        invalid = plan; invalid.decode.clear();
        assert!(matches!(invalid.validate(), Err(ExecutionAdmissionError::DecodeLanes { .. })));
    }
}
