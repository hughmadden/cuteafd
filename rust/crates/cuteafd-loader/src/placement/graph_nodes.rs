//! Family-owned launch counts, architecture-specific measured rates, one pack
//! per physical device (combine all banks before calling `charge`). Fingerprints
//! validate capture coverage; they never select a memory price.
use std::collections::BTreeMap;
use cuteafd_core::memory_policy::{GraphNodePadding, GraphPaddingValue};
use serde::{Deserialize, Serialize};
use super::inventory::{DeclaredGraphSet, GraphRank};

pub const DRIVER_CHUNK: u64 = 2 << 20;
pub const ORDINARY: &str = "ordinary";
pub const DSA_PAGED: &str = "dsa-paged-logits-and-tiled-topk";

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GraphNodeError {
    #[error("missing graph-node measurement: family={family}, architecture={architecture}, class={class}; collect a graph census on this architecture")]
    MissingRate { family: String, architecture: String, class: String },
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

#[derive(Debug, Clone, Copy)]
pub struct GraphNodeRate {
    pub architecture: &'static str,
    pub class: &'static str,
    pub bytes_per_node: u64,
    pub provenance: GraphRateProvenance,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphNodeCounts(pub BTreeMap<&'static str, u64>);

impl GraphNodeCounts {
    pub fn add(&mut self, class: &'static str, nodes: u64) -> Result<(), GraphNodeError> {
        let count = self.0.entry(class).or_default();
        *count = count.checked_add(nodes).ok_or(GraphNodeError::Overflow("node count"))?;
        Ok(())
    }

    /// Counts a declared set, not a captured table. Families supply a pure
    /// key-to-launch-count function and combine banks sharing a device here.
    pub fn declared<K: std::hash::Hash + Eq + Clone + std::fmt::Debug>(
        declared: &DeclaredGraphSet<K>,
        mut count: impl FnMut(&K) -> Result<Self, GraphNodeError>,
    ) -> Result<Self, GraphNodeError> {
        let mut total = Self::default();
        for key in declared.iter() {
            for (class, nodes) in count(key)?.0 { total.add(class, nodes)?; }
        }
        Ok(total)
    }
}

/// The padding is an identified part of the single graph owner, not a second
/// allocation or reserve. Only the combined raw sum receives a chunk tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNodeCharge {
    pub nodes: u64,
    pub rate_bytes: u64,
    pub padding: GraphPaddingValue,
    pub padding_bytes: u64,
    pub packed_bytes: u64,
}

impl GraphNodeCharge {
    pub fn rank(self, executables: u64) -> GraphRank {
        GraphRank { node_charge: Some(self), ..GraphRank::estimated(executables, self.packed_bytes) }
    }
}

pub fn charge(family: &str, architecture: &str, counts: &GraphNodeCounts,
    rates: &[GraphNodeRate], padding: GraphNodePadding) -> Result<GraphNodeCharge, GraphNodeError> {
    let missing = |class: &str| GraphNodeError::MissingRate { family: family.into(),
        architecture: architecture.into(), class: class.into() };
    let padding = match architecture {
        "sm120" => padding.rtx,
        "sm121" => padding.spark,
        _ => return Err(missing("architecture")),
    };
    let (mut nodes, mut rate_bytes, mut padding_bytes) = (0u64, 0u64, 0u64);
    for (&class, &count) in &counts.0 {
        if count == 0 { continue; }
        let mut matching = rates.iter().filter(|rate| rate.architecture == architecture && rate.class == class);
        let rate = matching.next().ok_or_else(|| missing(class))?;
        if matching.next().is_some() || rate.bytes_per_node == 0
            || rate.provenance.engine_sha.is_empty() || rate.provenance.sparkinfer_sha.is_empty()
            || rate.provenance.evidence.is_empty() {
            return Err(GraphNodeError::InvalidRate(format!("{family}/{architecture}/{class}")));
        }
        let per_node = padding.bytes_per_node(rate.bytes_per_node).ok_or(GraphNodeError::Overflow("padding rate"))?;
        padding_bytes = count.checked_mul(per_node).and_then(|bytes| padding_bytes.checked_add(bytes))
            .ok_or(GraphNodeError::Overflow("padding bytes"))?;
        nodes = nodes.checked_add(count).ok_or(GraphNodeError::Overflow("total nodes"))?;
        rate_bytes = count.checked_mul(rate.bytes_per_node).and_then(|bytes| rate_bytes.checked_add(bytes))
            .ok_or(GraphNodeError::Overflow("rate bytes"))?;
    }
    let raw = rate_bytes.checked_add(padding_bytes).ok_or(GraphNodeError::Overflow("combined bytes"))?;
    let packed_bytes = raw.div_ceil(DRIVER_CHUNK).checked_mul(DRIVER_CHUNK)
        .ok_or(GraphNodeError::Overflow("driver chunks"))?;
    Ok(GraphNodeCharge { nodes, rate_bytes, padding, padding_bytes, packed_bytes })
}

/// Emit a discrepancy without rejecting successfully instantiated CUDA graphs.
/// Coverage and owner validation remain separate, hard correctness checks.
pub fn reconcile(family: &str, device: usize, modeled_bytes: u64, measured_bytes: u64) -> u64 {
    let gap = measured_bytes.saturating_sub(modeled_bytes);
    if gap != 0 {
        tracing::warn!(family, device, modeled_bytes, measured_bytes, gap_bytes = gap,
            "startup graph memory exceeds node model; capture succeeded, continuing; increase GRAPH_NODE_PADDING for this architecture on future launches");
    }
    gap
}

#[cfg(test)]
mod tests {
    use super::*;
    const PROVENANCE: GraphRateProvenance = GraphRateProvenance {
        engine_sha: "archived", sparkinfer_sha: "851c42833da26711c68868014cad2fc043916416",
        cuda_driver_version: None, nvml_driver_version: None,
        evidence: "#1605 archived census, VALIDATION.md", limitation: "inferred capture-key join; single rank",
    };
    fn rates(ordinary: u64, dsa: u64) -> [GraphNodeRate; 2] {
        [GraphNodeRate { architecture: "sm120", class: ORDINARY, bytes_per_node: ordinary, provenance: PROVENANCE },
         GraphNodeRate { architecture: "sm120", class: DSA_PAGED, bytes_per_node: dsa, provenance: PROVENANCE }]
    }
    fn counts(ordinary: u64, dsa: u64) -> GraphNodeCounts {
        GraphNodeCounts(BTreeMap::from([(ORDINARY, ordinary), (DSA_PAGED, dsa)]))
    }
    #[test]
    fn archived_totals_and_smaller_contexts() {
        for (family, rates, full, small, bytes, small_bytes) in [
            ("glm5", rates(6680, 2792), counts(434790, 28560), counts(392064, 12096), 2984247296, 2652897280),
            ("glm5_flash", rates(7143, 2456), counts(321180, 7700), counts(288996, 4620), 2313158656, 2076180480),
        ] {
            let zero = "0".parse().unwrap();
            let full = charge(family, "sm120", &full, &rates, zero).unwrap();
            let small = charge(family, "sm120", &small, &rates, zero).unwrap();
            assert_eq!(full.packed_bytes, bytes);
            assert_eq!(small.packed_bytes, small_bytes);
            assert!(small.nodes < full.nodes && small.packed_bytes < full.packed_bytes);
        }
    }
    #[test]
    fn padding_is_per_node_per_device_and_independent_of_reserve() {
        use cuteafd_core::memory_policy::MemoryPolicy;
        let rates = rates(6680, 2792);
        let policy = MemoryPolicy { graph_node_padding: "rtx=350,spark=22".parse().unwrap(), ..Default::default() };
        let calculate = || charge("glm5", "sm120", &counts(434790, 28560), &rates,
            MemoryPolicy::current().graph_node_padding).unwrap();
        let expected = policy.scoped(calculate);
        assert_eq!(expected.padding_bytes, 463350 * 350);
        assert_eq!(expected.packed_bytes, (expected.rate_bytes + expected.padding_bytes).div_ceil(DRIVER_CHUNK) * DRIVER_CHUNK);
        assert_eq!(MemoryPolicy { gpu_reserve: Some(0), ..policy }.scoped(calculate), expected);
        let spark_rates = rates.map(|rate| GraphNodeRate { architecture: "sm121", ..rate });
        let spark = charge("fixture", "sm121", &counts(3, 2), &spark_rates, policy.graph_node_padding).unwrap();
        assert_eq!(spark.padding_bytes, 5 * 22);
        assert_eq!(spark.packed_bytes, DRIVER_CHUNK); // not one chunk per class
    }
    #[test]
    fn percentage_padding_uses_each_class_rate_and_architecture() {
        let glm = charge("glm5", "sm120", &counts(434790, 28560), &rates(6680, 2792), Default::default()).unwrap();
        assert_eq!(glm.padding_bytes, 434790 * 167 + 28560 * 70);
        let flash = charge("glm5_flash", "sm120", &counts(321180, 7700), &rates(7143, 2456), Default::default()).unwrap();
        assert_eq!(flash.padding_bytes, 321180 * 179 + 7700 * 61);
        let spark_rates = rates(1000, 2000).map(|r| GraphNodeRate { architecture: "sm121", ..r });
        let spark = charge("fixture", "sm121", &counts(2, 3), &spark_rates, Default::default()).unwrap();
        assert_eq!(spark.padding_bytes, 2 * 25 + 3 * 50);
    }

    #[test]
    fn unknown_rates_and_overflow_are_typed() {
        for arch in ["sm121", "sm999"] {
            assert!(matches!(charge("glm5", arch, &counts(1, 1), &rates(6680, 2792), Default::default()),
                Err(GraphNodeError::MissingRate { .. })));
        }
        assert!(matches!(charge("glm5", "sm120", &counts(u64::MAX, 1), &rates(6680, 2792), Default::default()),
            Err(GraphNodeError::Overflow(_))));
    }
    #[test]
    fn successful_capture_overrun_is_only_a_warning() {
        assert_eq!(reconcile("fixture", 0, 10, 1000), 990);
        assert_eq!(reconcile("fixture", 0, 1000, 10), 0);
    }
}
