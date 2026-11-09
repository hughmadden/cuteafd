//! GLM owned-attention captures: one census for warm-up, replay and admission.
use super::{glm_decode_row_buckets, glm_decode_table_widths};
use crate::families::glm5::GlmVariant;
use crate::placement::attention::AttentionPlacement;
use crate::placement::inventory::DeclaredGraphSet;

/// Which checkpoint's trace-derived startup work a glm5-engine layout uses.
/// Both are served by the same engine, but they do not capture the same graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmPrewarmFamily {
    /// GLM 5.x: one list covers every layout it ships.
    Glm5,
    /// Mistral Large 4: one and two coordinator ranks resolve different
    /// attention placements (heads, then page-partitioned context), so each
    /// layout has its own traced list.
    MistralLarge4,
}

impl GlmPrewarmFamily {
    pub fn of(variant: &GlmVariant) -> Self {
        if variant.is_glm() { Self::Glm5 } else { Self::MistralLarge4 }
    }
    /// The list's `family` identity.
    pub fn name(self) -> &'static str {
        match self { Self::Glm5 => "glm5", Self::MistralLarge4 => "mistral_large4" }
    }
    /// The coordinator layout this checkpoint's list is keyed by, if any.
    /// GLM shares one list; Mistral Large 4 keys it by rank count.
    fn layout(self, ranks: usize) -> Option<usize> {
        match self { Self::Glm5 => None, Self::MistralLarge4 => Some(ranks.clamp(1, 2)) }
    }
}

/// The width one rank's graph key carries for a step built with `table_width`.
/// Checked-in entries store the width a step is built with; under the context
/// placement each rank captures its half table (the engine's `rank_tables`).
pub fn prewarm_key_width(table_width: usize, context: bool) -> usize {
    if context { table_width.div_ceil(2) } else { table_width }
}

/// What precedes a captured decode segment's residual norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum GlmGraphPrevious {
    First,
    Delta,
    Planes(usize),
}

/// Stable launch geometry, shared by the declaration and the executor's lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct GlmGraphKey {
    pub layer: usize,
    pub rows: usize,
    pub table_width: usize,
    pub table_stride: usize,
    pub previous: GlmGraphPrevious,
    pub head: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct GlmGraphLayer {
    pub dense: bool,
    /// Whole or paired coordinator-local FFN; both produce a reduced delta.
    pub local: bool,
    pub attention_owner: Option<u8>,
}

/// Declare every retained executable before capture. This describes keys only,
/// not graph memory or observed nodes. Rank 0 also captures the terminal head;
/// rank 1 consumes rank 0's reduced FFN output, never Spark return planes.
pub fn glm_declared_graphs(shapes: &[GlmGraphShape], attention: AttentionPlacement, gpus: usize,
    layers: &[GlmGraphLayer], return_ranks: usize) -> Result<Vec<DeclaredGraphSet<GlmGraphKey>>, String> {
    if !(1..=2).contains(&gpus) || layers.is_empty()
        || (attention != AttentionPlacement::Heads && gpus != 2)
        || layers.iter().any(|layer| (layer.local && layer.dense)
            || match attention {
                AttentionPlacement::Layers => layer.attention_owner.is_none_or(|owner| owner >= 2),
                _ => layer.attention_owner.is_some(),
            }) {
        return Err("invalid GLM declared graph layout".into());
    }
    if shapes.iter().any(|shape| shape.rows == 0 || shape.table_width == 0) {
        return Err("invalid GLM declared graph shape".into());
    }
    let mut keys = vec![Vec::new(); gpus];
    if attention == AttentionPlacement::Layers {
        let owners: Vec<_> = layers.iter().map(|layer| layer.attention_owner.unwrap()).collect();
        let tp2: Vec<_> = layers.iter().map(|layer| layer.local).collect();
        for segment in glm_owned_segments(&owners, &tp2)? {
            for shape in shapes {
                keys[segment.rank].push(GlmGraphKey { layer: segment.phase.key(segment.layer), rows: shape.rows,
                    table_width: shape.table_width, table_stride: shape.table_width,
                    previous: GlmGraphPrevious::First, head: false });
            }
        }
    } else {
        for (rank, keys) in keys.iter_mut().enumerate() {
            for shape in shapes {
                let width = if attention == AttentionPlacement::Context { shape.table_width.div_ceil(2) }
                    else { shape.table_width };
                for layer in 0..layers.len() + usize::from(rank == 0) {
                    let previous = match layer.checked_sub(1).map(|index| layers[index]) {
                        None => GlmGraphPrevious::First,
                        Some(before) if before.dense || before.local => GlmGraphPrevious::Delta,
                        Some(_) => GlmGraphPrevious::Planes(if rank == 0 { return_ranks } else { 0 }),
                    };
                    keys.push(GlmGraphKey { layer, rows: shape.rows, table_width: width, table_stride: width,
                        previous, head: rank == 0 && layer == layers.len() });
                }
            }
        }
    }
    keys.iter().map(|rank| DeclaredGraphSet::new(rank)).collect()
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

/// Checked-in trace-derived startup work. A shape warms every segment sharing
/// its stable row/table geometry; pinning is explicit, not inferred from order.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmPrewarmEntry {
    #[serde(flatten)]
    pub shape: GlmGraphShape,
    pub pin: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmPrewarmList {
    pub schema_version: u32,
    pub family: String,
    /// The coordinator layout the list was traced on; absent when a family's
    /// layouts share one list.
    #[serde(default)]
    pub layout: Option<usize>,
    pub reserve_base_bytes: u64,
    pub reserve_extra_bytes: u64,
    pub entries: Vec<GlmPrewarmEntry>,
}

fn checked_in(source: &str, family: GlmPrewarmFamily, layout: Option<usize>) -> GlmPrewarmList {
    let list: GlmPrewarmList = serde_json::from_str(source)
        .unwrap_or_else(|error| panic!("checked-in {} prewarm list: {error}", family.name()));
    assert_eq!(list.schema_version, 1, "{} prewarm schema", family.name());
    assert_eq!(list.family, family.name(), "prewarm family mismatch");
    assert_eq!(list.layout, layout, "{} prewarm layout mismatch", family.name());
    list
}

/// GLM 5.3's list, measured on its 1-RTX/4-Spark EXL3 baseline plus the
/// C1/2/4/8/16 sweep (docs/architecture/shared-engine.md, Graph cache).
pub fn glm_prewarm_list() -> &'static GlmPrewarmList {
    static LIST: std::sync::OnceLock<GlmPrewarmList> = std::sync::OnceLock::new();
    LIST.get_or_init(|| checked_in(include_str!("glm_prewarm.json"), GlmPrewarmFamily::Glm5, None))
}

/// Mistral Large 4's list for the resolved coordinator layout. One rank runs
/// head-split attention and captures no peer segment; two ranks run the
/// context placement, which adds rank-1 segments and halves each table width.
///
/// Each list was traced on its own card (1 rank + 5 Sparks, 2 ranks + 6
/// Sparks) at KV target 2M / max 100M with the C1/2/4/8/16 sweep and the 32K
/// shared-prefix point. `reserve_base_bytes` is that layout's measured arena
/// high-water (the larger device under the context placement); `reserve_extra`
/// is its largest measured boundary capture, one 2 MiB chunk. Neither trace
/// evicted or recaptured a graph, so the only pin is the C1 decode geometry
/// (rows 1, width 16) that the qualified run pinned, as in `glm_prewarm.json`.
pub fn ml4_prewarm_list(ranks: usize) -> &'static GlmPrewarmList {
    static ONE: std::sync::OnceLock<GlmPrewarmList> = std::sync::OnceLock::new();
    static TWO: std::sync::OnceLock<GlmPrewarmList> = std::sync::OnceLock::new();
    let family = GlmPrewarmFamily::MistralLarge4;
    if ranks <= 1 {
        ONE.get_or_init(|| checked_in(include_str!("ml4_prewarm_1.json"), family, Some(1)))
    } else {
        TWO.get_or_init(|| checked_in(include_str!("ml4_prewarm_2.json"), family, Some(2)))
    }
}

/// The checked-in list a resolved glm5-engine layout uses.
pub fn glm_prewarm(family: GlmPrewarmFamily, ranks: usize) -> &'static GlmPrewarmList {
    match family {
        GlmPrewarmFamily::Glm5 => glm_prewarm_list(),
        GlmPrewarmFamily::MistralLarge4 => ml4_prewarm_list(ranks),
    }
}

/// Entries hold the width a step is built with, so they filter against the
/// engine's table-width buckets unchanged in both placements.
pub fn glm_prewarm_shapes(family: GlmPrewarmFamily, ranks: usize, context: usize,
    pool_pages: usize) -> Vec<GlmPrewarmEntry> {
    let allowed = glm_graph_shapes(context, pool_pages);
    glm_prewarm(family, ranks).entries.iter().filter(|entry| allowed.contains(&entry.shape)).cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn checked_in_prewarm_list_roundtrips_and_filters_to_supported_shapes() {
        let list = glm_prewarm_list();
        assert!(!list.entries.is_empty());
        let mut identities = HashSet::new();
        for entry in &list.entries {
            assert!(identities.insert((entry.shape.rows, entry.shape.table_width)), "duplicate prewarm geometry");
            assert!(entry.shape.rows > 0 && entry.shape.table_width > 0);
        }
        assert!(list.entries.iter().any(|entry| entry.pin && entry.shape == GlmGraphShape { rows: 1, table_width: 16 }));
        assert_eq!(list.reserve_base_bytes % (2 << 20), 0);
        assert_eq!(list.reserve_extra_bytes % (2 << 20), 0);
        assert_eq!(list.reserve_base_bytes, 335_544_320);
        assert_eq!(list.reserve_extra_bytes, 2_097_152);
        assert_eq!(list.reserve_base_bytes + list.reserve_extra_bytes, 337_641_472);
        let _: GlmPrewarmList = serde_json::from_value(serde_json::to_value(&list).unwrap()).unwrap();
        let allowed = glm_graph_shapes(4096, 32);
        for entry in glm_prewarm_shapes(GlmPrewarmFamily::Glm5, 1, 4096, 32) {
            assert!(allowed.contains(&entry.shape));
        }
    }

    #[test]
    fn every_family_layout_selects_its_own_checked_in_list() {
        let glm = glm_prewarm(GlmPrewarmFamily::Glm5, 1);
        assert_eq!(glm.family, "glm5");
        assert_eq!(glm.layout, None);
        // GLM shares one list across its layouts.
        assert!(std::ptr::eq(glm_prewarm(GlmPrewarmFamily::Glm5, 2), glm));
        for layout in [1usize, 2] {
            let list = glm_prewarm(GlmPrewarmFamily::MistralLarge4, layout);
            assert_eq!(list.family, GlmPrewarmFamily::MistralLarge4.name());
            assert_eq!(list.layout, Some(layout));
            assert!(!list.entries.is_empty());
            assert!(list.reserve_base_bytes > 0 && list.reserve_extra_bytes > 0);
            // The ranks key selects by layout, not by an off-by-one count.
            assert!(std::ptr::eq(glm_prewarm(GlmPrewarmFamily::MistralLarge4, layout), list));
        }
        assert!(std::ptr::eq(glm_prewarm(GlmPrewarmFamily::MistralLarge4, 0),
            glm_prewarm(GlmPrewarmFamily::MistralLarge4, 1)));
        let allowed = glm_graph_shapes(1 << 20, 16384);
        let warmed = glm_prewarm_shapes(GlmPrewarmFamily::MistralLarge4, 2, 1 << 20, 16384);
        assert!(!warmed.is_empty());
        for entry in warmed {
            assert!(allowed.contains(&entry.shape));
        }
        let one = glm_prewarm(GlmPrewarmFamily::MistralLarge4, 1);
        let two = glm_prewarm(GlmPrewarmFamily::MistralLarge4, 2);
        assert_eq!((one.reserve_base_bytes, one.reserve_extra_bytes), (713_031_680, 2_097_152));
        assert_eq!((two.reserve_base_bytes, two.reserve_extra_bytes), (912_261_120, 2_097_152));
        for list in [one, two] {
            let pins: Vec<_> = list.entries.iter().filter(|entry| entry.pin).collect();
            assert_eq!(pins.len(), 1, "{} layout {}", list.family, list.layout.unwrap());
            assert_eq!(pins[0].shape, GlmGraphShape { rows: 1, table_width: 16 });
        }
        // Checked-in entries hold the width a step is built with; the context
        // placement's per-rank key is its half table.
        assert_eq!(prewarm_key_width(16, false), 16);
        assert_eq!(prewarm_key_width(16, true), 8);
        assert_eq!(prewarm_key_width(17, true), 9);
    }

    #[test]
    fn split_declarations_match_rank_local_segments_and_terminal_head() {
        let layers = [
            GlmGraphLayer { dense: true, local: false, attention_owner: None },
            GlmGraphLayer { dense: false, local: true, attention_owner: None },
            GlmGraphLayer { dense: false, local: false, attention_owner: None },
        ];
        let shapes = [GlmGraphShape { rows: 4, table_width: 17 }];
        for attention in [AttentionPlacement::Heads, AttentionPlacement::Context] {
            let declared = glm_declared_graphs(&shapes, attention, 2, &layers, 6).unwrap();
            let width = if attention == AttentionPlacement::Context { 9 } else { 17 };
            for (rank, previous) in [
                vec![GlmGraphPrevious::First, GlmGraphPrevious::Delta, GlmGraphPrevious::Delta, GlmGraphPrevious::Planes(6)],
                vec![GlmGraphPrevious::First, GlmGraphPrevious::Delta, GlmGraphPrevious::Delta],
            ].into_iter().enumerate() {
                let actual: Vec<_> = previous.into_iter().enumerate().map(|(layer, previous)| GlmGraphKey {
                    layer, rows: 4, table_width: width, table_stride: width, previous, head: layer == 3 }).collect();
                declared[rank].verify(actual.iter()).unwrap();
                assert_eq!(declared[rank].len(), actual.len());
                let mut changed = actual.clone();
                changed[0].table_stride += 1;
                assert!(declared[rank].verify(changed.iter()).is_err());
            }
        }
        let mut remote = layers;
        remote[1].local = false;
        let declared = glm_declared_graphs(&shapes, AttentionPlacement::Heads, 2, &remote, 4).unwrap();
        for (rank, planes) in [(0, 4), (1, 0)] {
            assert!(declared[rank].contains(&GlmGraphKey { layer: 2, rows: 4,
                table_width: 17, table_stride: 17, previous: GlmGraphPrevious::Planes(planes), head: false }));
        }
        let one = glm_declared_graphs(&shapes, AttentionPlacement::Heads, 1, &remote[..1], 0).unwrap();
        assert_eq!(one.len(), 1);
        assert!(one[0].contains(&GlmGraphKey { layer: 1, rows: 4, table_width: 17, table_stride: 17,
            previous: GlmGraphPrevious::Delta, head: true }));
    }

    #[test]
    fn whole_local_graphs_use_delta_without_a_pair_or_return_planes() {
        let shapes = [GlmGraphShape { rows: 4, table_width: 17 }];
        let mut layers = [
            GlmGraphLayer { dense: true, local: false, attention_owner: None },
            GlmGraphLayer { dense: false, local: true, attention_owner: None },
            GlmGraphLayer { dense: false, local: false, attention_owner: None },
        ];
        let declared = glm_declared_graphs(&shapes, AttentionPlacement::Heads, 1, &layers, 4).unwrap();
        let actual = [GlmGraphPrevious::First, GlmGraphPrevious::Delta,
            GlmGraphPrevious::Delta, GlmGraphPrevious::Planes(4)].into_iter().enumerate()
            .map(|(layer, previous)| GlmGraphKey { layer, rows: 4, table_width: 17,
                table_stride: 17, previous, head: layer == layers.len() }).collect::<Vec<_>>();
        declared[0].verify(actual.iter()).unwrap();
        layers[1].local = false;
        let remote = glm_declared_graphs(&shapes, AttentionPlacement::Heads, 1, &layers, 4).unwrap();
        assert!(remote[0].verify(actual.iter()).is_err());
        layers[1].local = true;
        layers[2].local = true;
        let all_local = glm_declared_graphs(&shapes, AttentionPlacement::Heads, 1, &layers, 0).unwrap();
        assert!(all_local[0].contains(&GlmGraphKey { layer: 3, previous: GlmGraphPrevious::Delta,
            ..actual[3] }));
    }

    #[test]
    fn declarations_reproduce_min_max_counts_without_observed_nodes_or_bytes() {
        let shapes = glm_graph_shapes(1 << 20, 32768);
        assert_eq!(shapes.len(), 213);
        let min: Vec<_> = (0..78).map(|layer| GlmGraphLayer {
            dense: layer < 3, local: false, attention_owner: None }).collect();
        let mut max = min.clone();
        for layer in &mut max[3..9] { layer.local = true; }
        let min = glm_declared_graphs(&shapes, AttentionPlacement::Heads, 1, &min, 4).unwrap();
        let max = glm_declared_graphs(&shapes, AttentionPlacement::Context, 2, &max, 6).unwrap();
        assert_eq!(min[0].len(), 16_827);
        assert_eq!(max.iter().map(DeclaredGraphSet::len).collect::<Vec<_>>(), [16_827, 16_614]);
        assert_ne!(min[0].fingerprint(0).unwrap(), max[0].fingerprint(0).unwrap());
        assert_ne!(max[0].fingerprint(0).unwrap(), max[1].fingerprint(1).unwrap());
    }

    #[test]
    fn min_max_declarations_cover_every_reachable_verify_shape_and_depth() {
        use crate::serving_capacity::{glm_decode_row_bucket, glm_decode_table_width};
        // verify_device_hooked accepts a total of 1..=64 rows. Every speculative
        // depth and batch partition is a member of this domain, not a new key.
        let mut totals = HashSet::new();
        for depth in 1..=64 {
            for batch in 1..=64 / depth {
                for tail in 0..depth {
                    let rows = batch * depth + tail;
                    if rows <= 64 { totals.insert(rows); }
                }
            }
        }
        assert_eq!(totals.len(), 64);
        for (attention, gpus, ranks, pool_tokens, onboard) in [
            (AttentionPlacement::Heads, 1, 4, 1_257_856usize, 0),
            (AttentionPlacement::Context, 2, 6, 2_097_152, 6),
        ] {
            let context = 1 << 20;
            let pages = pool_tokens / super::super::GLM_PAGE_ROWS;
            let layers: Vec<_> = (0..78).map(|layer| GlmGraphLayer {
                dense: layer < 3, local: (3..3 + onboard).contains(&layer), attention_owner: None,
            }).collect();
            let declared = glm_declared_graphs(&glm_graph_shapes(context, pages), attention,
                gpus, &layers, ranks).unwrap();
            // Enumerate real page demands rather than trusting the warm inventory
            // to enumerate itself. Normalize exactly as decode dispatch does.
            let mut reachable = HashSet::new();
            for needed in 1..=context / super::super::GLM_PAGE_ROWS {
                let width = glm_decode_table_width(needed, context, pages);
                for &rows in &totals {
                    reachable.insert((glm_decode_row_bucket(rows, width).unwrap(), width));
                }
            }
            assert_eq!(reachable.len(), 213);
            for (rows, width) in reachable {
                let width = if attention == AttentionPlacement::Context { width.div_ceil(2) } else { width };
                for (rank, inventory) in declared.iter().enumerate() {
                    for layer in 0..layers.len() + usize::from(rank == 0) {
                        let previous = if layer == 0 { GlmGraphPrevious::First }
                            else if layer <= 3 || (3..3 + onboard).contains(&(layer - 1)) { GlmGraphPrevious::Delta }
                            else { GlmGraphPrevious::Planes(if rank == 0 { ranks } else { 0 }) };
                        let key = GlmGraphKey { layer, rows, table_width: width, table_stride: width,
                            previous, head: rank == 0 && layer == layers.len() };
                        assert!(inventory.contains(&key), "reachable rank {rank} key is undeclared: {key:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn owned_declaration_covers_exact_phases_and_changes_with_ownership() {
        let shapes = glm_graph_shapes(131072, 4096);
        let layers = [
            GlmGraphLayer { dense: true, local: false, attention_owner: Some(0) },
            GlmGraphLayer { dense: false, local: true, attention_owner: Some(1) },
        ];
        let declared = glm_declared_graphs(&shapes, AttentionPlacement::Layers, 2, &layers, 4).unwrap();
        for (rank, phases) in [
            [(0, GlmOwnedPhase::Attention), (0, GlmOwnedPhase::FfnFront), (0, GlmOwnedPhase::FfnJoin),
                (1, GlmOwnedPhase::RemoteNorm), (1, GlmOwnedPhase::FfnJoin)],
            [(0, GlmOwnedPhase::RemoteNorm), (0, GlmOwnedPhase::FfnFront), (0, GlmOwnedPhase::FfnJoin),
                (1, GlmOwnedPhase::Attention), (1, GlmOwnedPhase::FfnJoin)],
        ].into_iter().enumerate() {
            let actual: Vec<_> = shapes.iter().flat_map(|shape| phases.iter().map(move |&(layer, phase)| GlmGraphKey {
                layer: phase.key(layer), rows: shape.rows, table_width: shape.table_width, table_stride: shape.table_width,
                previous: GlmGraphPrevious::First, head: false })).collect();
            declared[rank].verify(actual.iter()).unwrap();
            assert!(declared[rank].verify(actual[..actual.len() - 1].iter()).is_err());
        }
        let mut swapped = layers;
        swapped[0].attention_owner = Some(1);
        let swapped = glm_declared_graphs(&shapes, AttentionPlacement::Layers, 2, &swapped, 4).unwrap();
        assert_ne!(declared[0].fingerprint(0).unwrap(), swapped[0].fingerprint(0).unwrap());
    }

    #[test]
    fn invalid_or_duplicate_graph_declarations_fail_closed() {
        let shape = GlmGraphShape { rows: 1, table_width: 16 };
        let mut layer = GlmGraphLayer { dense: true, local: false, attention_owner: None };
        for gpus in [0, 3] {
            assert!(glm_declared_graphs(&[shape], AttentionPlacement::Heads, gpus, &[layer], 0).is_err());
        }
        for attention in [AttentionPlacement::Context, AttentionPlacement::Layers] {
            assert!(glm_declared_graphs(&[shape], attention, 1, &[layer], 0).is_err());
        }
        assert!(glm_declared_graphs(&[shape], AttentionPlacement::Heads, 1, &[], 0).is_err());
        assert!(glm_declared_graphs(&[shape, shape], AttentionPlacement::Heads, 1, &[layer], 0).is_err());
        assert!(glm_declared_graphs(&[GlmGraphShape { rows: 0, ..shape }], AttentionPlacement::Heads, 1, &[layer], 0).is_err());
        assert!(glm_declared_graphs(&[GlmGraphShape { table_width: 0, ..shape }], AttentionPlacement::Heads, 1, &[layer], 0).is_err());
        layer.local = true;
        assert!(glm_declared_graphs(&[shape], AttentionPlacement::Heads, 2, &[layer], 0).is_err());
        layer.local = false;
        layer.attention_owner = Some(2);
        assert!(glm_declared_graphs(&[shape], AttentionPlacement::Layers, 2, &[layer], 0).is_err());
        layer.attention_owner = Some(0);
        assert!(glm_declared_graphs(&[shape], AttentionPlacement::Heads, 2, &[layer], 0).is_err());
        layer.attention_owner = None;
        assert!(glm_declared_graphs(&[shape], AttentionPlacement::Layers, 2, &[layer], 0).is_err());
    }

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
        }
        assert!(glm_owned_segments(&[2], &[false]).is_err());
        assert!(glm_owned_segments(&[0], &[]).is_err());
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
