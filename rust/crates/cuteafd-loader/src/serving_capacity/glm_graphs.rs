//! GLM owned-attention captures: one census for warm-up, replay and admission.
use super::{glm_decode_row_buckets, glm_decode_table_widths};
use crate::placement::inventory::{GraphSet, Lifetime, StartupGraphs, GLM_OWNED_SM120_GRAPH_BYTES};

/// These values are the persistent graph-key phase, not a launch count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlmOwnedPhase {
    Attention = 0,
    FfnFront = 1,
    FfnJoin = 2,
    RemoteNorm = 3,
}

impl GlmOwnedPhase {
    pub fn key(self, layer: usize) -> usize { 4 * layer + self as usize }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlmOwnedSegment {
    pub rank: usize,
    pub layer: usize,
    pub phase: GlmOwnedPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlmGraphShape {
    pub rows: usize,
    pub table_width: usize,
}

/// The head and TP2 expert fronts are currently uncaptured. Every layer has
/// owner attention, peer normalization and two joins; non-TP2 layers also
/// capture both FFN fronts. Keep this definition shared with the bindings.
pub fn glm_owned_segments(owners: &[u8], tp2: &[bool]) -> Result<Vec<GlmOwnedSegment>, String> {
    if owners.is_empty() || owners.len() != tp2.len() || owners.iter().any(|&rank| rank > 1) {
        return Err("invalid GLM owned graph geometry".into());
    }
    let mut segments = Vec::new();
    for (layer, (&owner, &local)) in owners.iter().zip(tp2).enumerate() {
        for rank in 0..2 {
            segments.push(GlmOwnedSegment { rank, layer, phase: if rank == owner as usize {
                GlmOwnedPhase::Attention
            } else { GlmOwnedPhase::RemoteNorm } });
            if !local { segments.push(GlmOwnedSegment { rank, layer, phase: GlmOwnedPhase::FfnFront }); }
            segments.push(GlmOwnedSegment { rank, layer, phase: GlmOwnedPhase::FfnJoin });
        }
    }
    Ok(segments)
}

pub fn glm_graph_shapes(context: usize, pool_pages: usize) -> Vec<GlmGraphShape> {
    glm_decode_table_widths(context, pool_pages).into_iter().flat_map(|table_width| {
        glm_decode_row_buckets(table_width).iter().map(move |&rows| GlmGraphShape { rows, table_width })
    }).collect()
}

/// Measured SM120 owned-phase calibration with a separate 10% growth margin
/// (64 MiB minimum). Planner and startup use the same executable census.
pub fn glm_owned_startup_graphs(context: usize, pool_pages: usize, owners: &[u8], tp2: &[bool])
    -> Result<StartupGraphs<GlmGraphShape>, String> {
    let segments = glm_owned_segments(owners, tp2)?;
    let keys = glm_graph_shapes(context, pool_pages);
    let mut counts = [0u64; 2];
    for segment in segments {
        counts[segment.rank] = counts[segment.rank].checked_add(keys.len() as u64)
            .ok_or("GLM graph count overflow")?;
    }
    GraphSet::new(&counts, GLM_OWNED_SM120_GRAPH_BYTES, 10, 64 << 20, keys.len() as u64, Lifetime::Startup).startup(keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn owned_census_counts_phases_not_whole_engine_allowances() {
        for owners in [vec![0, 0, 1, 1], vec![1, 1, 0, 0]] {
            let tp2 = [false, true, false, true];
            let segments = glm_owned_segments(&owners, &tp2).unwrap();
            assert_eq!(segments.len(), 20);
            assert_eq!(segments.iter().copied().collect::<HashSet<_>>().len(), segments.len());
            for (layer, &owner) in owners.iter().enumerate() {
                assert!(segments.contains(&GlmOwnedSegment { rank: owner as usize, layer, phase: GlmOwnedPhase::Attention }));
                assert!(segments.contains(&GlmOwnedSegment { rank: 1 - owner as usize, layer, phase: GlmOwnedPhase::RemoteNorm }));
                for rank in 0..2 {
                    assert_eq!(segments.contains(&GlmOwnedSegment { rank, layer, phase: GlmOwnedPhase::FfnFront }), !tp2[layer]);
                }
            }
            let startup = glm_owned_startup_graphs(1 << 20, 32768, &owners, &tp2).unwrap();
            for rank in 0..2 {
                assert_eq!(startup.inventory.ranks[rank].executables, startup.keys.len() as u64 * 10);
                assert_eq!(startup.inventory.at_ready(rank), startup.keys.len() as u64 * 10 * GLM_OWNED_SM120_GRAPH_BYTES);
                assert_eq!(startup.inventory.at_ready(rank) + startup.inventory.growth(rank), startup.inventory.bytes(rank));
            }
        }
        assert!(glm_owned_segments(&[2], &[false]).is_err());
        assert!(glm_owned_segments(&[0], &[]).is_err());
    }

    #[test]
    fn owned_sm120_ready_calibration_matches_recorded_capture_census() {
        // RTX PRO 6000 SM120, GLM 5.3 EXL3 K4, two RTX + six expert ranks,
        // 2M pool, 1M context, 78 layers (TP2 layers 3..5), 2026-10-15.
        // Both launches measured the same deltas. The 213 shapes use widths
        // 16..16384 and the shared short/long decode-row buckets (up to 64).
        // Recheck on RTX 5090 or a changed census; neither is measured here.
        let owners: Vec<_> = (0..78).map(|layer| (layer % 2) as u8).collect();
        let tp2: Vec<_> = (0..78).map(|layer| (3..5).contains(&layer)).collect();
        let startup = glm_owned_startup_graphs(1 << 20, 32768, &owners, &tp2).unwrap();
        assert_eq!(startup.keys.len(), 213);
        for (rank, observed) in [5_790_236_672u64, 5_762_973_696].into_iter().enumerate() {
            assert_eq!(startup.inventory.ranks[rank].executables, 49_416);
            assert_eq!(startup.inventory.at_ready(rank), 5_781_672_000);
            assert!(startup.inventory.at_ready(rank).abs_diff(observed) <= 64 << 20);
            assert_eq!(startup.inventory.growth(rank), 578_167_200);
        }
    }

    #[test]
    fn warm_shapes_match_every_capped_table_and_row_bucket() {
        for context in [64, 131072, 1 << 20] {
            for pages in [1, 1024, 32768] {
                let shapes = glm_graph_shapes(context, pages);
                assert_eq!(shapes.iter().copied().collect::<HashSet<_>>().len(), shapes.len());
                for width in glm_decode_table_widths(context, pages) {
                    for &rows in glm_decode_row_buckets(width) {
                        assert!(shapes.contains(&GlmGraphShape { rows, table_width: width }));
                    }
                }
            }
        }
    }
}
