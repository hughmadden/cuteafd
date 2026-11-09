//! V4's decode graph cache inputs: one prewarm list and graph reserve per
//! resolved coordinator shape. V4 keys a captured segment by its exact
//! geometry, so a prewarm entry is a stable `(rows, table_width)` pair: width 0
//! is the local-expert/TP2 startup set (decode segments), and a width above 0
//! is a verify geometry the startup warm captures at that C4 width class.
//!
//! A shape is `(coordinator GPUs, Sparks attached)`: a card without Sparks
//! captures no onboarding graphs and one GPU captures no TP2 peer segments, so
//! the lists differ by shape rather than by card name. A shape with no trace
//! falls back to the bootstrap startup set and `traced: false`, and startup
//! warns that its list is provisional.
//!
//! **Untraced shapes are provisional.** A shape without its own trace card
//! keeps the graph growth charge this family admitted before the bounded cache
//! (#1617/#1614): 0.35 GiB per GPU on the 1x shapes and 0.40/0.25 GiB on the 2x
//! shape's lead/peer ranks, exactly `GRAPH_BYTES[role]`, all of it
//! `reserve_extra_bytes` because no measured prewarm arena exists to separate a
//! base from headroom, plus a one-graph bootstrap estimate. Today that is only
//! `rtx1-local`; startup names the missing trace. The shared capture-headroom
//! policy then charges that provisional base like any other family base: with
//! no user key it adds `min(500 MB, 100% of the base)`.
//!
//! A traced shape carries its own cards' measurements: every geometry the trace
//! observed is an entry, so `reserve_base_bytes` is the startup warm's whole
//! arena per rank — the width-0 row classes plus one capture per layer segment
//! at each verify geometry (the trace's remaining charged arena is lazy
//! captures, mainly the never-prewarmed tail), `reserve_extra_bytes` is that
//! arena's 2 MiB allocation tail, and `bootstrap_bytes` is its largest measured
//! capture. `scripts/dev/graph-prewarm-from-trace.py` replaces one shape at a
//! time; the consumer schema does not change.

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
    /// The emitted prewarm set's own measured captures per rank; a best-effort
    /// warm target, never a seal. A trace's charged arena also holds lazy
    /// captures, which are not part of this base.
    pub reserve_base_bytes: Vec<u64>,
    /// The prewarm arena's allocation tail per rank. Capture headroom for lazy
    /// graphs is a separate shared policy key, never hand-picked here.
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

/// Per-rank graph reserve: the prewarm arena plus its tail, further widened by
/// the shared capture-headroom policy. A bare `GPU_GRAPH_RESERVE` replaces this
/// on every coordinator device, and `gpuN=` selectors override individual ranks.
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
        rank, false, base[rank], extra[rank]).map_err(|error| error.to_string())).collect()
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
/// effort within the reserve. A width above 0 is a verify geometry the startup
/// warm captures on a synthetic placement at that C4 class.
pub fn v4_prewarm(gpus: usize, sparks: bool) -> Result<Vec<V4PrewarmEntry>, String> {
    Ok(layout_entry(v4_layout(gpus, sparks)?).entries)
}

/// The pinned geometries this shape keeps resident, as `(rows, table_width)`:
/// a graph the trace shows as always hot is captured pinned, so churn elsewhere
/// cannot evict it. V4's startup set is the width-0 row classes; a verify
/// geometry is pinned only where its own trace showed sustained heat.
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
    fn the_untraced_shape_keeps_the_replaced_growth_charge() {
        // The numbers retired from `families::deepseek_v4::GRAPH_BYTES`; the
        // shared headroom policy charges this provisional base like any other.
        let entry = &v4_prewarm_list().layouts["rtx1-local"];
        assert!(!entry.traced);
        assert_eq!(entry.reserve_base_bytes, [0]);
        assert_eq!(entry.reserve_extra_bytes, [gib(35)]);
        assert_eq!(entry.bootstrap_bytes, [8 << 20]);
        // No trace card for this shape, so there is no startup set to warm.
        assert!(entry.entries.is_empty());
        assert!(v4_prewarm(1, false).unwrap().is_empty());
        assert!(v4_pinned(1, false).unwrap().is_empty());
        assert!(v4_reserves(3, true).is_err());
        assert!(v4_bootstraps(0, true).is_err());
        assert!(v4_pinned(3, false).is_err());
    }

    #[test]
    fn traced_layouts_carry_their_cards_measured_numbers() {
        let list = v4_prewarm_list();
        let one = &list.layouts["rtx1-spark"];
        assert!(one.traced);
        // The startup warm's whole arena on the min card: 112 verify geometries
        // and 51 width-0 row classes, its arena tail, one measured capture.
        assert_eq!(one.reserve_base_bytes, [1_172_307_968]);
        assert_eq!(one.reserve_extra_bytes, [2 << 20]);
        assert_eq!(one.bootstrap_bytes, [2 << 20]);
        let spark = &list.layouts["rtx2-spark"];
        assert!(spark.traced);
        assert_eq!(spark.reserve_base_bytes, [1_509_949_440, 1_524_629_504]);
        let local = &list.layouts["rtx2-local"];
        assert!(local.traced);
        assert_eq!(local.reserve_base_bytes, [1_530_920_960, 1_547_698_176]);
        for entry in [one, spark, local] {
            assert_eq!(entry.reserve_extra_bytes, vec![2 << 20; entry.reserve_base_bytes.len()]);
            assert_eq!(entry.bootstrap_bytes, vec![2 << 20; entry.reserve_base_bytes.len()]);
        }
    }

    #[test]
    fn every_traced_shape_covers_its_working_set_at_reachable_rows() {
        for (gpus, sparks) in [(1, true), (2, true), (2, false)] {
            let entries = v4_prewarm(gpus, sparks).unwrap();
            // Every width-0 row class the trace observed is the boot-warmed
            // startup set, captured before serving, so it is pinned.
            let startup: Vec<_> = entries.iter().filter(|entry| entry.table_width == 0).collect();
            assert!(startup.len() >= 50);
            assert!(startup.iter().all(|entry| entry.pin));
            assert!(startup.iter().any(|entry| entry.rows == 1));
            assert!(startup.iter().any(|entry| entry.rows == 64));
            // No emitted geometry outruns the decode row capacity the traces ran.
            assert!(entries.iter().all(|entry| entry.rows <= 64));
            assert_eq!(v4_pinned(gpus, sparks).unwrap().len(),
                entries.iter().filter(|entry| entry.pin).count());
        }
    }

    #[test]
    fn verify_geometries_are_prewarmed_unpinned_except_sustained_heat() {
        // The full width ladder the traces observed, at every row count.
        for (gpus, sparks) in [(1, true), (2, true), (2, false)] {
            let verify: Vec<_> = v4_prewarm(gpus, sparks).unwrap().into_iter()
                .filter(|entry| entry.table_width > 0).collect();
            assert!(verify.len() > 100, "{gpus}x{sparks}: {} verify geometries", verify.len());
            assert_eq!(verify.iter().map(|entry| entry.table_width).collect::<std::collections::BTreeSet<_>>(),
                [1usize, 2, 4, 8, 32, 64, 128, 256].into_iter().collect());
            // A pin is permanent residency, so only trace-proven heat earns one.
            let pinned: Vec<_> = verify.iter().filter(|entry| entry.pin)
                .map(|entry| (entry.rows, entry.table_width)).collect();
            let expected = if (gpus, sparks) == (1, true) { vec![(48, 128)] } else { vec![] };
            assert_eq!(pinned, expected);
        }
    }

    #[test]
    fn every_shape_selects_its_own_trace_state() {
        let mut keys = std::collections::HashSet::new();
        for (gpus, sparks) in [(1, true), (1, false), (2, true), (2, false)] {
            let layout = v4_layout(gpus, sparks).unwrap();
            assert!(keys.insert(layout.key()));
            // Every shape but `rtx1-local` has its own trace card.
            assert_eq!(v4_traced(gpus, sparks).unwrap(), layout_entry(layout).traced);
            assert_eq!(layout.ranks(), gpus);
        }
        assert!(v4_traced(3, true).is_err());
        assert!(v4_traced(0, false).is_err());
    }

    const fn gib(hundredths: u64) -> u64 { hundredths * (1 << 30) / 100 }
}
