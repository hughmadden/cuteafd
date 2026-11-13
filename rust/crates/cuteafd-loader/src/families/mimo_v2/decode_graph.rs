//! MiMo's decode graph site and reserve inputs.
//!
//! MiMo captures one executable per layer segment between the Spark exchanges,
//! plus the full-row head tail. Headless tails execute eagerly. The startup set is a provisional prewarm, not
//! a sealed inventory: serving misses capture-or-run-eager through the shared
//! bounded device cache, which charges one reserve per physical device.
use crate::serving_capacity::mimo_graphs::mimo_prewarm_layout;
use super::config::MimoV2Config;
use crate::serving_capacity::CacheGeometryError;
use cuteafd_core::serving_capacity::MemoryReservation;

pub const MIMO_DECODE_ROWS: usize = 64;
/// Only full-row heads are captured; retained subset logits use an eager tail.
pub const MIMO_DECODE_TAIL_HEAD_VARIANTS: [bool; 1] = [true];

/// Provisional bootstrap observation of the whole 64-shape Pro capture on SM120
/// (driver/runtime 13020, CUDA 13.2, 2026-11-08): 616 MiB over 4544 lead
/// executables (70 layers plus one tail-head variant per shape) and 566 MiB
/// over 4480 peer executables (70 layers per shape). Headless tails are eager;
/// no estimated second-tail increment is included. These are bootstrap inputs until the
/// family's trace card replaces them.
pub const MIMO_GRAPH_LEAD_OBSERVED: (u64, u64) = (4544, 616 << 20);
pub const MIMO_GRAPH_PEER_OBSERVED: (u64, u64) = (4480, 566 << 20);

/// One resolved card layout's graph inputs: its prewarm rows, per-rank reserve
/// and the per-graph class estimate used until the first measured capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimoGraphLayout {
    pub enabled: bool,
    /// Prewarm row shapes: pinned first, then descending request replay demand.
    pub rows: Vec<usize>,
    /// Row shapes that stay pinned in the cache.
    pub pinned: Vec<usize>,
    /// Lead rank has each layer plus the full-row head tail.
    /// A peer has each attention layer and no vocabulary tail.
    pub segments_per_rank: Vec<usize>,
    pub executables_per_rank: Vec<usize>,
    /// Charged once per physical device, before KV sizing.
    pub reserves: Vec<u64>,
    /// Where each rank's reserve came from (`family default` or `user override`).
    pub sources: Vec<&'static str>,
    /// Per-graph bootstrap estimate until the device has a measured capture.
    pub bootstraps: Vec<u64>,
    /// Admission and runtime telemetry share the same resolved per-device policy.
    pub policies: Vec<crate::placement::graph_reserve::Resolved>,
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

pub fn graph_layout(cfg: &MimoV2Config, ranks: usize, sparks: bool, enabled: bool) -> Result<MimoGraphLayout, CacheGeometryError> {
    let layers = cfg.layers;
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
            policies: Vec::new(),
        });
    }
    let lead = layers
        .checked_add(MIMO_DECODE_TAIL_HEAD_VARIANTS.len())
        .ok_or(CacheGeometryError::Overflow("MiMo decode graph tail segments"))?;
    let mut segments_per_rank = vec![lead];
    if ranks == 2 {
        segments_per_rank.push(layers);
    }
    let prewarm = mimo_prewarm_layout(cfg, ranks, sparks);
    let executables_per_rank = segments_per_rank
        .iter()
        .map(|&segments| {
            segments
                .checked_mul(prewarm.entries.len())
                .ok_or(CacheGeometryError::Overflow("MiMo decode graph executable count"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (mut reserves, mut sources, mut bootstraps, mut policies) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (rank, &executables) in executables_per_rank.iter().enumerate() {
        let base = prewarm.reserve_base_bytes[rank];
        let extra = prewarm.reserve_extra_bytes[rank];
        let resolved = crate::placement::graph_reserve::current(rank, false, base, extra)
            .map_err(CacheGeometryError::GraphReserve)?;
        reserves.push(resolved.bytes);
        sources.push(resolved.source);
        bootstraps.push(base.div_ceil(executables.max(1) as u64));
        policies.push(resolved);
    }
    let bases: Vec<_> = policies.iter().map(|policy| policy.base_bytes).collect();
    let selected = select_prewarm(&prewarm, &bases);
    if selected.len() < prewarm.entries.len() {
        tracing::warn!(skipped_entries = prewarm.entries.len() - selected.len(), base_bytes = ?bases,
            "MiMo prewarm exceeds base; skipped rows remain runtime misses (pinned set is indivisible)");
    }
    let rows = selected.iter().map(|&index| prewarm.entries[index].rows).collect::<Vec<_>>();
    let pinned = selected.iter().filter_map(|&index| {
        let entry = prewarm.entries[index];
        entry.pin.then_some(entry.rows)
    }).collect();
    let executables_per_rank = segments_per_rank.iter().map(|segments| segments * rows.len()).collect();
    Ok(MimoGraphLayout {
        enabled: true,
        rows,
        pinned,
        segments_per_rank,
        executables_per_rank,
        reserves,
        sources,
        bootstraps,
        policies,
    })
}

fn select_prewarm(prewarm: &crate::serving_capacity::mimo_graphs::MimoLayoutPrewarm, bases: &[u64]) -> Vec<usize> {
    let pins: Vec<_> = prewarm.entries.iter().map(|entry| entry.pin).collect();
    crate::placement::graph_reserve::prewarm_selection(&pins, bases, |selected, bases| {
        bases.iter().enumerate().all(|(rank, base)| {
            let bytes = if let Some(shares) = prewarm.entry_capture_bytes.get(rank) {
                selected.iter().fold(0u64, |sum, &index| sum.saturating_add(shares[index]))
            } else {
                prewarm.reserve_base_bytes[rank].div_ceil(prewarm.entries.len().max(1) as u64)
                    .saturating_mul(selected.len() as u64)
            };
            let arena = bytes.div_ceil(2 << 20).saturating_mul(2 << 20);
            arena <= *base
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MimoV2Config {
        MimoV2Config::from_hf(&crate::plan::testing::mimo_pro_config()).unwrap()
    }

    #[test]
    fn layouts_charge_one_reserve_per_physical_rank() {
        let cfg = config();
        let lead_segments = cfg.layers + MIMO_DECODE_TAIL_HEAD_VARIANTS.len();
        for ranks in [1, 2] {
            let layout = graph_layout(&cfg, ranks, true, true).unwrap();
            assert_eq!(layout.segments_per_rank[0], lead_segments);
            assert_eq!(layout.executables_per_rank[0], lead_segments * layout.rows.len());
            if ranks == 2 {
                assert_eq!(layout.segments_per_rank[1], cfg.layers);
                assert_eq!(layout.executables_per_rank[1], cfg.layers * layout.rows.len());
            }
            assert_eq!(layout.reserves.len(), ranks);
            for rank in 0..ranks {
                assert_eq!(layout.reservations(rank).unwrap()[0].bytes, layout.reserves[rank]);
                assert_eq!(layout.reservations(rank).unwrap()[0].name, "graph reserve");
            }
            assert!(layout.reservations(ranks).is_err());
        }
    }

    #[test]
    fn prewarm_fits_pinned_set_then_prefix_on_every_device_without_runtime_extra() {
        use crate::serving_capacity::mimo_graphs::{MimoLayoutPrewarm, MimoPrewarmEntry};
        let list = MimoLayoutPrewarm {
            variant: "mimop-70".into(), ranks: 2, sparks: true,
            reserve_base_bytes: vec![20 << 20; 2], reserve_extra_bytes: vec![2 << 20; 2],
            entry_capture_bytes: vec![vec![8 << 20, 2 << 20, 8 << 20, 2 << 20],
                vec![2 << 20, 4 << 20, 2 << 20, 4 << 20]],
            entries: (1..=4).map(|rows| MimoPrewarmEntry { rows, pin: rows % 2 == 0 }).collect(),
        };
        assert_eq!(select_prewarm(&list, &[12 << 20, 10 << 20]), vec![1, 3, 0]);
        assert_eq!(select_prewarm(&list, &[20 << 20, 6 << 20]), Vec::<usize>::new());
        assert_eq!(select_prewarm(&list, &[0, 20 << 20]), Vec::<usize>::new());
        let lazy = crate::placement::graph_reserve::resolve_with_extra(None, Some("500MB"), None,
            0, false, 0, 0).unwrap();
        assert!(lazy.bytes > 0);
        assert!(select_prewarm(&list, &[lazy.base_bytes; 2]).is_empty());
    }

    #[test]
    fn disabled_graphs_charge_nothing() {
        let layout = graph_layout(&config(), 2, true, false).unwrap();
        assert!(!layout.enabled);
        assert!(layout.rows.is_empty());
        assert_eq!(layout.executables_per_rank, vec![0, 0]);
        assert!(layout.reservations(0).unwrap().is_empty());
        assert!(layout.reservations(1).unwrap().is_empty());
    }

    #[test]
    fn invalid_graph_geometry_is_named() {
        for (layers, ranks) in [(0, 1), (70, 0), (70, 3)] {
            let mut cfg = config();
            cfg.layers = layers;
            assert!(matches!(graph_layout(&cfg, ranks, true, true),
                Err(CacheGeometryError::Unsupported { .. })));
        }
        let mut cfg = config();
        cfg.layers = usize::MAX;
        assert!(matches!(graph_layout(&cfg, 1, true, true), Err(CacheGeometryError::Overflow(_))));
    }
}
