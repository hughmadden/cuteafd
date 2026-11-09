//! V4's decode graph cache inputs: one prewarm list and graph reserve per
//! resolved coordinator shape. V4 keys a captured segment by its exact
//! geometry, so a prewarm entry is a stable `(rows, table_width)` pair; the
//! local-expert startup set uses width 0, decode segments a C4 width class.
//!
//! A shape is `(coordinator GPUs, Sparks attached)`: a card without Sparks
//! captures no onboarding graphs and one GPU captures no TP2 peer segments, so
//! the lists differ by shape rather than by card name. A shape with no trace
//! falls back to the bootstrap startup set and `traced: false`, and startup
//! warns that its list is provisional.
//!
//! **Untraced shapes are provisional.** Until V4's own trace cards run, the
//! reserve keeps the graph growth charge this family admitted before the
//! bounded cache (#1617/#1614): 0.35 GiB per GPU on the 1x shape and
//! 0.40/0.25 GiB on the 2x shape's lead/peer ranks, exactly `GRAPH_BYTES[role]`.
//! All of it is `reserve_extra_bytes` because no measured prewarm arena exists
//! to separate a base from capture headroom, and the bootstrap is a one-graph
//! estimate. A trace card replaces these numbers, the entries and the pins
//! through `scripts/dev/graph-prewarm-from-trace.py`; the consumer schema does
//! not change.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The coordinator shape a prewarm list and reserve belong to: one or two RTX
/// GPUs, with or without Sparks attached. Admission knows both before it sizes
/// the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V4GraphLayout {
    Rtx1Spark,
    Rtx1Local,
    Rtx2Spark,
    Rtx2Local,
}

impl V4GraphLayout {
    pub fn of(gpus: usize, sparks: bool) -> Result<Self, String> {
        match (gpus, sparks) {
            (1, true) => Ok(Self::Rtx1Spark),
            (1, false) => Ok(Self::Rtx1Local),
            (2, true) => Ok(Self::Rtx2Spark),
            (2, false) => Ok(Self::Rtx2Local),
            _ => Err(format!("V4 graph cache admits one or two coordinator GPUs, not {gpus}")),
        }
    }

    /// The shape key in the checked-in list, and in a trace's file names.
    pub fn key(self) -> &'static str {
        match self {
            Self::Rtx1Spark => "rtx1-spark",
            Self::Rtx1Local => "rtx1-local",
            Self::Rtx2Spark => "rtx2-spark",
            Self::Rtx2Local => "rtx2-local",
        }
    }

    pub fn ranks(self) -> usize {
        match self {
            Self::Rtx1Spark | Self::Rtx1Local => 1,
            Self::Rtx2Spark | Self::Rtx2Local => 2,
        }
    }
}

/// One captured geometry. `table_width` is the decode table's C4 width class,
/// or 0 for the local-expert startup set and the segment tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V4PrewarmEntry {
    pub rows: usize,
    pub table_width: usize,
    pub pin: bool,
}

/// One shape's reserve and prewarm set. Every array is indexed by coordinator
/// rank: the lead and peer capture different graphs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V4LayoutPrewarm {
    /// Whether a trace card has replaced the provisional numbers.
    pub traced: bool,
    /// Measured prewarm arena high-water per rank; a best-effort warm target,
    /// never a seal.
    pub reserve_base_bytes: Vec<u64>,
    /// One largest capture of extra room for lazy captures, per rank.
    pub reserve_extra_bytes: Vec<u64>,
    /// The estimate a device uses until it has a measured capture, per rank.
    pub bootstrap_bytes: Vec<u64>,
    pub entries: Vec<V4PrewarmEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V4PrewarmList {
    pub schema_version: u32,
    pub family: String,
    pub layouts: BTreeMap<String, V4LayoutPrewarm>,
}

pub fn v4_prewarm_list() -> V4PrewarmList {
    let list: V4PrewarmList = serde_json::from_str(include_str!("v4_prewarm.json"))
        .expect("checked-in V4 prewarm list");
    assert_eq!(list.schema_version, 1);
    assert_eq!(list.family, "deepseek_v4");
    list
}

fn layout_entry(layout: V4GraphLayout) -> V4LayoutPrewarm {
    v4_prewarm_list().layouts.remove(layout.key())
        .unwrap_or_else(|| panic!("checked-in V4 prewarm list lacks {}", layout.key()))
}

/// The shape selected by the resolved coordinator GPUs and Spark attachment.
pub fn v4_layout(gpus: usize, sparks: bool) -> Result<V4GraphLayout, String> {
    V4GraphLayout::of(gpus, sparks)
}

/// Whether this shape's numbers come from its own trace card. Startup warns
/// when they do not.
pub fn v4_traced(gpus: usize, sparks: bool) -> Result<bool, String> {
    Ok(layout_entry(v4_layout(gpus, sparks)?).traced)
}

/// Per-rank graph reserve: the prewarm arena plus one capture of headroom.
/// A bare `GPU_GRAPH_RESERVE` replaces this on every coordinator device, and
/// `gpuN=` selectors override individual ranks.
pub fn v4_reserves(gpus: usize, sparks: bool) -> Result<Vec<u64>, String> {
    Ok(v4_reserve_policies(gpus, sparks)?.into_iter().map(|reserve| reserve.bytes).collect())
}

/// The same reserves with where each number came from, for startup logs.
pub fn v4_reserve_policies(gpus: usize, sparks: bool)
    -> Result<Vec<crate::placement::graph_reserve::Resolved>, String> {
    let layout = v4_layout(gpus, sparks)?;
    let entry = layout_entry(layout);
    let (base, extra) = (entry.reserve_base_bytes, entry.reserve_extra_bytes);
    if base.len() != layout.ranks() || extra.len() != layout.ranks() {
        return Err(format!("V4 prewarm list {} ranks disagree", layout.key()));
    }
    (0..layout.ranks()).map(|rank| crate::placement::graph_reserve::current(
        rank, false, base[rank], extra[rank])).collect()
}

/// Per-rank bootstrap estimate, for a device with no measured capture yet.
pub fn v4_bootstraps(gpus: usize, sparks: bool) -> Result<Vec<u64>, String> {
    let layout = v4_layout(gpus, sparks)?;
    let entry = layout_entry(layout);
    if entry.bootstrap_bytes.len() != layout.ranks() {
        return Err(format!("V4 prewarm list {} bootstraps disagree", layout.key()));
    }
    Ok(entry.bootstrap_bytes.clone())
}

/// The startup prewarm set for this shape: entries whose `table_width` is 0
/// are the local-expert row classes the engine captures before serving, best
/// effort within the reserve. A width outside 0 is a decode segment the trace
/// showed as always hot.
pub fn v4_prewarm(gpus: usize, sparks: bool) -> Result<Vec<V4PrewarmEntry>, String> {
    Ok(layout_entry(v4_layout(gpus, sparks)?).entries)
}

/// The pinned geometries this shape keeps resident, as `(rows, table_width)`:
/// a graph the trace shows as always hot is captured pinned, so churn elsewhere
/// cannot evict it. V4's startup set is the local-expert row classes.
pub fn v4_pinned(gpus: usize, sparks: bool) -> Result<Vec<(usize, usize)>, String> {
    Ok(layout_entry(v4_layout(gpus, sparks)?).entries.into_iter()
        .filter(|entry| entry.pin)
        .map(|entry| (entry.rows, entry.table_width))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_list_covers_every_shape_and_rank() {
        let list = v4_prewarm_list();
        assert_eq!(list.layouts.len(), 4);
        for (gpus, sparks) in [(1, true), (1, false), (2, true), (2, false)] {
            let layout = v4_layout(gpus, sparks).unwrap();
            let entry = layout_entry(layout);
            assert_eq!(entry.reserve_base_bytes.len(), layout.ranks());
            assert_eq!(entry.reserve_extra_bytes.len(), layout.ranks());
            assert_eq!(entry.bootstrap_bytes.len(), layout.ranks());
            for rank in 0..layout.ranks() {
                assert!(entry.reserve_base_bytes[rank] + entry.reserve_extra_bytes[rank] > 0);
                assert!(entry.bootstrap_bytes[rank] > 0);
                assert!(entry.bootstrap_bytes[rank] <= entry.reserve_base_bytes[rank]
                    .saturating_add(entry.reserve_extra_bytes[rank]));
            }
            let mut identities = std::collections::HashSet::new();
            for graph in &entry.entries {
                assert!(graph.rows > 0);
                assert!(identities.insert((graph.rows, graph.table_width)));
            }
        }
    }

    #[test]
    fn provisional_reserve_matches_the_replaced_growth_charge() {
        // The numbers retired from `families::deepseek_v4::GRAPH_BYTES`.
        assert_eq!(v4_reserves(1, true).unwrap(), [gib(35)]);
        assert_eq!(v4_reserves(2, false).unwrap(), [gib(40), gib(25)]);
        assert_eq!(v4_bootstraps(1, true).unwrap(), [8 << 20]);
        assert!(v4_reserves(3, true).is_err());
        assert!(v4_bootstraps(0, true).is_err());
        // The one-GPU shapes leave every backbone expert on Sparks, so their
        // startup set is empty; the two-GPU ones prewarm the decode row classes.
        for sparks in [true, false] {
            assert!(v4_prewarm(1, sparks).unwrap().is_empty());
            assert!(v4_pinned(1, sparks).unwrap().is_empty());
        }
        let rows: Vec<_> = v4_prewarm(2, false).unwrap().iter().map(|entry| entry.rows).collect();
        assert_eq!(rows, (1..=16).collect::<Vec<_>>());
        // The never-evicted startup set stays pinned through a trace refresh.
        assert_eq!(v4_pinned(2, false).unwrap(), rows.iter().map(|&rows| (rows, 0)).collect::<Vec<_>>());
        assert!(v4_pinned(3, false).is_err());
    }

    #[test]
    fn every_shape_selects_its_own_trace_state() {
        let mut keys = std::collections::HashSet::new();
        for (gpus, sparks) in [(1, true), (1, false), (2, true), (2, false)] {
            let layout = v4_layout(gpus, sparks).unwrap();
            assert!(keys.insert(layout.key()));
            assert_eq!(v4_traced(gpus, sparks).unwrap(), layout_entry(layout).traced);
            // Both traced shapes attach Sparks; the shipped 2-RTX local card
            // and the 1-GPU shapes stay provisional until their own trace.
            assert_eq!(layout.ranks(), gpus);
        }
        assert!(v4_traced(3, true).is_err());
        assert!(v4_traced(0, false).is_err());
    }

    const fn gib(hundredths: u64) -> u64 { hundredths * (1 << 30) / 100 }
}
