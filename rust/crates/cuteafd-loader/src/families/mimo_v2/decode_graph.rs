//! MiMo's decode graph site and reserve inputs.
//!
//! MiMo captures one executable per layer segment between the Spark exchanges,
//! plus both tail head variants. The startup set is a provisional prewarm, not
//! a sealed inventory: serving misses capture-or-run-eager through the shared
//! bounded device cache, which charges one reserve per physical device.
use crate::serving_capacity::mimo_graphs::{mimo_prewarm_layout, mimo_prewarm_pinned, mimo_prewarm_rows};
use crate::serving_capacity::CacheGeometryError;
use cuteafd_core::serving_capacity::MemoryReservation;

pub const MIMO_DECODE_ROWS: usize = 64;
/// Both full-row logits and retained subset logits are reachable in serving.
pub const MIMO_DECODE_TAIL_HEAD_VARIANTS: [bool; 2] = [true, false];

/// Provisional bootstrap observation of the whole 64-shape Pro capture on SM120
/// (driver/runtime 13020, CUDA 13.2, 2026-11-08): 616 MiB over 4544 lead
/// executables (70 layers plus one tail-head variant per shape) and 566 MiB
/// over 4480 peer executables (70 layers per shape). The lead set now carries
/// the second, headless tail variant; the measured maximum per-shape increment
/// was 12 MiB. These are the checked-in prewarm reserve numbers until the
/// family's trace card replaces them.
pub const MIMO_GRAPH_LEAD_OBSERVED: (u64, u64) = (4608, (616 + 12) << 20);
pub const MIMO_GRAPH_PEER_OBSERVED: (u64, u64) = (4480, 566 << 20);

/// One resolved card layout's graph inputs: its prewarm rows, per-rank reserve
/// and the per-graph class estimate used until the first measured capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimoGraphLayout {
    pub enabled: bool,
    /// Prewarm row shapes, ascending.
    pub rows: Vec<usize>,
    /// Row shapes that stay pinned in the cache.
    pub pinned: Vec<usize>,
    /// Lead rank has each layer plus both full/partial-head tail variants.
    /// A peer has each attention layer and no vocabulary tail.
    pub segments_per_rank: Vec<usize>,
    pub executables_per_rank: Vec<usize>,
    /// Charged once per physical device, before KV sizing.
    pub reserves: Vec<u64>,
    /// Where each rank's reserve came from (`family default` or `user override`).
    pub sources: Vec<&'static str>,
    /// Per-graph bootstrap estimate until the device has a measured capture.
    pub bootstraps: Vec<u64>,
}

impl MimoGraphLayout {
    /// The retained graph owner of one physical rank. With graphs off nothing
    /// is charged and no device cache is attached.
    pub fn reservations(&self, rank: usize) -> Result<Vec<MemoryReservation>, CacheGeometryError> {
        let &bytes = self.reserves.get(rank).ok_or(CacheGeometryError::Unsupported {
            family: "mimo_v2",
            what: "decode graph physical rank",
        })?;
        if bytes == 0 {
            return Ok(Vec::new());
        }
        Ok(vec![MemoryReservation { name: "graph reserve".into(), bytes }])
    }

    pub fn pinned_rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.pinned.iter().copied()
    }
}

pub fn graph_layout(layers: usize, ranks: usize, enabled: bool) -> Result<MimoGraphLayout, CacheGeometryError> {
    if !(1..=2).contains(&ranks) || layers == 0 {
        return Err(CacheGeometryError::Unsupported {
            family: "mimo_v2",
            what: "segmented decode graph layer/rank geometry",
        });
    }
    if !enabled {
        return Ok(MimoGraphLayout {
            enabled: false,
            rows: Vec::new(),
            pinned: Vec::new(),
            segments_per_rank: vec![0; ranks],
            executables_per_rank: vec![0; ranks],
            reserves: vec![0; ranks],
            sources: vec!["disabled"; ranks],
            bootstraps: vec![0; ranks],
        });
    }
    let lead = layers
        .checked_add(MIMO_DECODE_TAIL_HEAD_VARIANTS.len())
        .ok_or(CacheGeometryError::Overflow("MiMo decode graph tail segments"))?;
    let mut segments_per_rank = vec![lead];
    if ranks == 2 {
        segments_per_rank.push(layers);
    }
    let rows = mimo_prewarm_rows(ranks);
    if rows.is_empty() {
        return Err(CacheGeometryError::Unsupported {
            family: "mimo_v2",
            what: "decode graph prewarm layout",
        });
    }
    let executables_per_rank = segments_per_rank
        .iter()
        .map(|&segments| {
            segments
                .checked_mul(rows.len())
                .ok_or(CacheGeometryError::Overflow("MiMo decode graph executable count"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let prewarm = mimo_prewarm_layout(ranks);
    let (mut reserves, mut sources, mut bootstraps) = (Vec::new(), Vec::new(), Vec::new());
    for (rank, &executables) in executables_per_rank.iter().enumerate() {
        let base = prewarm.reserve_base_bytes[rank];
        let extra = prewarm.reserve_extra_bytes[rank];
        let resolved = crate::placement::graph_reserve::current(rank, false, base, extra)
            .map_err(CacheGeometryError::GraphReserve)?;
        reserves.push(resolved.bytes);
        sources.push(resolved.source);
        bootstraps.push(resolved.bytes.div_ceil(executables.max(1) as u64));
    }
    Ok(MimoGraphLayout {
        enabled: true,
        rows,
        pinned: mimo_prewarm_pinned(ranks),
        segments_per_rank,
        executables_per_rank,
        reserves,
        sources,
        bootstraps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_layouts_charge_one_reserve_per_physical_rank() {
        let one = graph_layout(70, 1, true).unwrap();
        assert_eq!(one.segments_per_rank, vec![72]);
        assert_eq!(one.executables_per_rank, vec![4608]);
        assert_eq!(one.reserves, vec![640 << 20]);
        assert_eq!(one.sources, vec!["family default"]);
        assert_eq!(one.bootstraps, vec![(640u64 << 20).div_ceil(4608)]);
        assert_eq!(one.reservations(0).unwrap(),
            vec![MemoryReservation { name: "graph reserve".into(), bytes: 640 << 20 }]);

        let two = graph_layout(70, 2, true).unwrap();
        assert_eq!(two.segments_per_rank, vec![72, 70]);
        assert_eq!(two.executables_per_rank, vec![4608, 4480]);
        assert_eq!(two.reserves, vec![640 << 20, 578 << 20]);
        assert_eq!(two.reservations(0).unwrap()[0].name, "graph reserve");
        assert_eq!(two.reservations(1).unwrap()[0].bytes, 578 << 20);
        assert!(two.reservations(2).is_err());
    }

    #[test]
    fn the_reserve_covers_the_measured_prewarm_arena_and_one_more_capture() {
        assert_eq!(MIMO_GRAPH_LEAD_OBSERVED.1, 628 << 20);
        assert_eq!(MIMO_GRAPH_PEER_OBSERVED.1, 566 << 20);
        let two = graph_layout(70, 2, true).unwrap();
        assert_eq!(two.reserves[0] - MIMO_GRAPH_LEAD_OBSERVED.1, 12 << 20);
        assert_eq!(two.reserves[1] - MIMO_GRAPH_PEER_OBSERVED.1, 12 << 20);
    }

    #[test]
    fn every_reachable_row_shape_is_warmed_and_one_row_is_pinned() {
        let layout = graph_layout(70, 2, true).unwrap();
        assert_eq!(layout.rows, (1..=MIMO_DECODE_ROWS).collect::<Vec<_>>());
        assert_eq!(layout.pinned_rows().collect::<Vec<_>>(), vec![1]);
        assert!(layout.rows.contains(&MIMO_DECODE_ROWS));
    }

    #[test]
    fn disabled_graphs_charge_nothing_and_attach_no_device_cache() {
        let layout = graph_layout(70, 2, false).unwrap();
        assert!(!layout.enabled);
        assert!(layout.rows.is_empty());
        assert_eq!(layout.executables_per_rank, vec![0, 0]);
        assert!(layout.reservations(0).unwrap().is_empty());
        assert!(layout.reservations(1).unwrap().is_empty());
    }

    #[test]
    fn invalid_graph_geometry_is_named() {
        for (layers, ranks) in [(0, 1), (70, 0), (70, 3)] {
            assert!(matches!(graph_layout(layers, ranks, true),
                Err(CacheGeometryError::Unsupported { .. })));
        }
        assert!(matches!(graph_layout(usize::MAX - 1, 1, true),
            Err(CacheGeometryError::Overflow(_))));
    }
}
