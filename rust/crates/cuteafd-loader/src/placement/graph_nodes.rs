//! Family-owned launch counts and per-family, per-architecture node/executable
//! rates. Combine every bank on one physical device before the single pack.
//! Fingerprints validate coverage; they never select a memory price.
use cuteafd_core::memory_policy::{GraphNodePadding, GraphPaddingValue};
use serde::{Deserialize, Serialize};
use super::inventory::{DeclaredGraphSet, GraphRank};

pub const DRIVER_CHUNK: u64 = 2 << 20;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GraphNodeError {
    #[error("missing graph measurement: family={family}, architecture={architecture}; collect a graph census on this architecture")]
    MissingRate { family: String, architecture: String },
    #[error("graph-node arithmetic overflow: {0}")]
    Overflow(&'static str),
    #[error("invalid graph-node measurement: {0}")]
    InvalidRate(String),
    #[error("cannot derive graph nodes: {0}")]
    Nodes(String),
}

/// Unknown driver fields stay `None`; the CUDA API version is not an NVML
/// driver release string. Record both on future captures.
#[derive(Debug, Clone, Copy)]
pub struct GraphRateProvenance {
    pub engine_sha: &'static str,
    pub sparkinfer_sha: &'static str,
    pub cuda_driver_version: Option<u32>,
    pub nvml_driver_version: Option<&'static str>,
    pub evidence: &'static str,
    pub limitation: &'static str,
}

/// Whole-byte coefficients rounded from the family's ordered-block fit.
/// Executable overhead is not another node class or another driver chunk tail.
#[derive(Debug, Clone, Copy)]
pub struct GraphNodeRate {
    pub family: &'static str,
    pub architecture: &'static str,
    pub bytes_per_node: u64,
    pub bytes_per_executable: u64,
    pub provenance: GraphRateProvenance,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphNodeCounts {
    pub nodes: u64,
    pub executables: u64,
}

impl GraphNodeCounts {
    /// Combine banks before pricing, so a physical device gets one chunk tail.
    pub fn add(&mut self, other: Self) -> Result<(), GraphNodeError> {
        let nodes = self.nodes.checked_add(other.nodes).ok_or(GraphNodeError::Overflow("node count"))?;
        let executables = self.executables.checked_add(other.executables)
            .ok_or(GraphNodeError::Overflow("executable count"))?;
        *self = Self { nodes, executables };
        Ok(())
    }

    /// Families supply a pure key-to-launch-count function, not a captured table.
    pub fn declared<K: std::hash::Hash + Eq + Clone + std::fmt::Debug>(
        declared: &DeclaredGraphSet<K>,
        mut count: impl FnMut(&K) -> Result<u64, GraphNodeError>,
    ) -> Result<Self, GraphNodeError> {
        let mut total = Self::default();
        for key in declared.iter() {
            total.add(Self { nodes: count(key)?, executables: 1 })?;
        }
        Ok(total)
    }
}

/// Both rate terms and their padding belong to one graph owner, not a second
/// allocation or reserve. Only the combined raw sum receives a chunk tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNodeCharge {
    pub nodes: u64,
    pub executables: u64,
    pub rate_bytes: u64,
    pub padding: GraphPaddingValue,
    pub padding_bytes: u64,
    pub packed_bytes: u64,
}

impl GraphNodeCharge {
    pub fn rank(self) -> GraphRank {
        GraphRank { node_charge: Some(self), ..GraphRank::estimated(self.executables, self.packed_bytes) }
    }
}

pub fn charge(family: &str, architecture: &str, counts: &GraphNodeCounts,
    rates: &[GraphNodeRate], padding: GraphNodePadding) -> Result<GraphNodeCharge, GraphNodeError> {
    let missing = || GraphNodeError::MissingRate { family: family.into(), architecture: architecture.into() };
    let padding = match architecture {
        "sm_120" => padding.rtx,
        "sm_121" => padding.spark,
        _ => return Err(missing()),
    };
    let (mut rate_bytes, mut padding_bytes) = (0u64, 0u64);
    if counts.nodes != 0 || counts.executables != 0 {
        if counts.executables == 0 {
            return Err(GraphNodeError::Nodes("nodes without graph executables".into()));
        }
        let mut matching = rates.iter().filter(|rate| rate.family == family && rate.architecture == architecture);
        let rate = matching.next().ok_or_else(missing)?;
        if matching.next().is_some() || rate.bytes_per_node == 0 || rate.bytes_per_executable == 0
            || rate.provenance.engine_sha.is_empty() || rate.provenance.sparkinfer_sha.is_empty()
            || rate.provenance.evidence.is_empty() {
            return Err(GraphNodeError::InvalidRate(format!("{family}/{architecture}")));
        }
        for (count, coefficient) in [(counts.nodes, rate.bytes_per_node),
            (counts.executables, rate.bytes_per_executable)] {
            let adjustment = padding.bytes_per_node(coefficient).ok_or(GraphNodeError::Overflow("padding rate"))?;
            padding_bytes = count.checked_mul(adjustment).and_then(|bytes| padding_bytes.checked_add(bytes))
                .ok_or(GraphNodeError::Overflow("padding bytes"))?;
            rate_bytes = count.checked_mul(coefficient).and_then(|bytes| rate_bytes.checked_add(bytes))
                .ok_or(GraphNodeError::Overflow("rate bytes"))?;
        }
    }
    let raw = rate_bytes.checked_add(padding_bytes).ok_or(GraphNodeError::Overflow("combined bytes"))?;
    let packed_bytes = raw.div_ceil(DRIVER_CHUNK).checked_mul(DRIVER_CHUNK)
        .ok_or(GraphNodeError::Overflow("driver chunks"))?;
    Ok(GraphNodeCharge { nodes: counts.nodes, executables: counts.executables,
        rate_bytes, padding, padding_bytes, packed_bytes })
}

/// Emit a discrepancy without rejecting successfully instantiated CUDA graphs.
/// Coverage and owner validation remain separate, hard correctness checks.
pub fn reconcile(family: &str, device: usize, modeled_bytes: u64, measured_bytes: u64) -> u64 {
    let gap = measured_bytes.saturating_sub(modeled_bytes);
    if gap != 0 {
        tracing::warn!(family, device, modeled_bytes, measured_bytes, gap_bytes = gap,
            "startup graph memory exceeds node/executable model; capture succeeded, continuing; increase GRAPH_NODE_PADDING for this architecture on future launches");
    }
    gap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving_capacity::glm_graph_nodes::{GLM_SM120_NODE_RATES, GLMF_SM120_NODE_RATES};
    fn counts(nodes: u64, executables: u64) -> GraphNodeCounts { GraphNodeCounts { nodes, executables } }

    #[test]
    fn glm_full_calibration_and_independent_small_context() {
        // #1615 sealed same-build captures. The small capture is independent,
        // not the old 404,160-node prefix of the full-context warm order.
        for (nodes, executables, bytes) in [(463350, 16827, 2984247296), (402240, 15168, 2648702976)] {
            let charge = charge("glm5", "sm_120", &counts(nodes, executables), &GLM_SM120_NODE_RATES,
                "0".parse().unwrap()).unwrap();
            assert_eq!(charge.packed_bytes, bytes);
            assert_eq!(charge.rank().executables, executables);
        }
    }

    #[test]
    fn flash_archived_block_fit_is_within_one_chunk() {
        let charge = charge("glm5_flash", "sm_120", &counts(328880, 13800), &GLMF_SM120_NODE_RATES,
            "0".parse().unwrap()).unwrap();
        // Same ordered-256-graph-block fit as GLM; retain its measured residual.
        assert_eq!(charge.packed_bytes, 2313158656 + DRIVER_CHUNK);
    }

    #[test]
    fn padding_applies_to_both_terms_and_survives_reserve_override() {
        use cuteafd_core::memory_policy::MemoryPolicy;
        let counts = counts(463350, 16827);
        let policy = MemoryPolicy { graph_node_padding: "rtx=350,spark=22".parse().unwrap(), ..Default::default() };
        let calculate = || charge("glm5", "sm_120", &counts, &GLM_SM120_NODE_RATES,
            MemoryPolicy::current().graph_node_padding).unwrap();
        let expected = policy.scoped(calculate);
        assert_eq!(expected.padding_bytes, (counts.nodes + counts.executables) * 350);
        assert_eq!(MemoryPolicy { gpu_reserve: Some(0), ..policy }.scoped(calculate), expected);
        let spark_rates = GLM_SM120_NODE_RATES.map(|rate| GraphNodeRate { architecture: "sm_121", ..rate });
        let spark = charge("glm5", "sm_121", &self::counts(3, 2), &spark_rates, policy.graph_node_padding).unwrap();
        assert_eq!(spark.padding_bytes, 5 * 22);
        assert_eq!(spark.packed_bytes, DRIVER_CHUNK);
    }

    #[test]
    fn percentage_padding_uses_node_and_executable_rates() {
        let glm = charge("glm5", "sm_120", &counts(463350, 16827), &GLM_SM120_NODE_RATES, Default::default()).unwrap();
        assert_eq!(glm.padding_bytes, 463350 * 67 + 16827 * 2592);
        let flash = charge("glm5_flash", "sm_120", &counts(328880, 13800), &GLMF_SM120_NODE_RATES, Default::default()).unwrap();
        assert_eq!(flash.padding_bytes, 328880 * 44 + 13800 * 3132);
    }

    #[test]
    fn declared_banks_are_combined_before_one_pack() {
        let keys = DeclaredGraphSet::new(&[1u64, 2]).unwrap();
        let bank = GraphNodeCounts::declared(&keys, |&key| Ok(key)).unwrap();
        assert_eq!(bank, counts(3, 2));
        let mut total = bank;
        total.add(bank).unwrap();
        let priced = charge("glm5", "sm_120", &total, &GLM_SM120_NODE_RATES, "0".parse().unwrap()).unwrap();
        assert_eq!(priced.executables, 4);
        assert_eq!(priced.packed_bytes, DRIVER_CHUNK);
        assert_eq!(charge("glm5", "sm_120", &GraphNodeCounts::default(), &[], Default::default()).unwrap().packed_bytes, 0);
        // Even a zero-node executable retains its executable overhead.
        assert_eq!(charge("glm5", "sm_120", &counts(0, 1), &GLM_SM120_NODE_RATES,
            "0".parse().unwrap()).unwrap().rate_bytes, 103662);
    }

    #[test]
    fn unknown_rates_invalid_inputs_and_overflow_are_typed() {
        for (family, arch) in [("glm5", "sm_121"), ("glm5", "sm_999"), ("glm5_flash", "sm_120")] {
            assert!(matches!(charge(family, arch, &counts(1, 1), &GLM_SM120_NODE_RATES, Default::default()),
                Err(GraphNodeError::MissingRate { .. })));
        }
        for counts in [counts(u64::MAX, 1), counts(1, u64::MAX)] {
            assert!(matches!(charge("glm5", "sm_120", &counts, &GLM_SM120_NODE_RATES, Default::default()),
                Err(GraphNodeError::Overflow(_))));
        }
        assert!(matches!(charge("glm5", "sm_120", &counts(1, 0), &GLM_SM120_NODE_RATES, Default::default()),
            Err(GraphNodeError::Nodes(_))));
        let duplicate = [GLM_SM120_NODE_RATES[0]; 2];
        assert!(matches!(charge("glm5", "sm_120", &counts(1, 1), &duplicate, Default::default()),
            Err(GraphNodeError::InvalidRate(_))));
    }

    #[test]
    fn successful_capture_overrun_is_only_a_warning() {
        assert_eq!(reconcile("fixture", 0, 10, 1000), 990);
        assert_eq!(reconcile("fixture", 0, 1000, 10), 0);
    }
}
