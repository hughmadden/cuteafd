//! GLM Flash startup decode graph inventory, shared by capture and CPU admission.
use super::glmf_table_pages;
pub const PAGE_ROWS: usize = 64;
pub const UNIT_PAGES: usize = 4;
pub const UNIT_ROWS: usize = PAGE_ROWS * UNIT_PAGES;
pub const DECODE_ROWS: usize = 64;
pub const PLAIN_DECODE_BUCKETS: [usize; 6] = [1, 4, 8, 16, 32, 64];
pub const SPEC_DECODE_BUCKETS: [usize; 6] = [2, 4, 8, 16, 32, 64];
use serde::{Deserialize, Serialize};

pub fn verify_budget(decode_rows: usize, sms: usize) -> usize {
    if decode_rows <= DECODE_ROWS { decode_rows } else { (3 * sms / 4).clamp(DECODE_ROWS, decode_rows) }
}

/// The row buckets padded decode steps run at (startup graphs): the plain buckets (a plain step has
/// one row per sequence, at most 64) and the speculative ones, which with the wide programs end at
/// the verify budget: `[2, 4, 8, 16, 32, 64, 127]` on an RTX 5090, `[.., 64, 128]` on 188 SMs,
/// `[.., 64, 99]` on 132. A budget of 64 keeps the 64-row sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeBuckets {
    pub plain: Vec<usize>,
    pub spec: Vec<usize>,
}

impl DecodeBuckets {
    /// The sets of a verify budget of `verify_rows` rows ([`verify_budget`]).
    pub fn new(verify_rows: usize) -> Self {
        let mut spec = SPEC_DECODE_BUCKETS.to_vec();
        if verify_rows > DECODE_ROWS {
            spec.push(verify_rows);
        }
        Self { plain: PLAIN_DECODE_BUCKETS.to_vec(), spec }
    }

    /// The bucket a step of `rows` rows pads to (the rows themselves past the largest).
    pub fn bucket(&self, rows: usize, spec: bool) -> usize {
        let set = if spec { &self.spec } else { &self.plain };
        set.iter().copied().find(|&bucket| bucket >= rows).unwrap_or(rows)
    }
}

/// One checked-in prewarm shape: a masked decode step's row, spec flag and
/// table geometry. A shape warms every layer segment that shares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmfPrewarmEntry {
    pub rows: usize,
    pub spec: bool,
    pub long: bool,
    pub pool_width: usize,
    pub page_stride: usize,
    pub pool_stride: usize,
    pub pin: bool,
}

impl GlmfPrewarmEntry {
    pub fn shape(&self) -> (usize, bool, GraphGeometry) {
        (self.rows, self.spec, GraphGeometry { pool_width: self.pool_width,
            page_stride: self.page_stride, pool_stride: self.pool_stride, long: self.long })
    }
}

/// The card layouts GLM Flash serves: the min column (one coordinator device
/// with the Sparks) and the max column (two coordinator devices, 0 Sparks).
/// Each layout captures its own graphs, so each reads its own traced entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmfLayout { Min, Max }

impl GlmfLayout {
    /// The layout a launch with `gpus` coordinator devices serves: one device
    /// is the min column, more than one the max column.
    pub fn for_gpus(gpus: usize) -> Self { if gpus > 1 { Self::Max } else { Self::Min } }

    pub fn name(self) -> &'static str { match self { Self::Min => "min", Self::Max => "max" } }
}

/// One layout's prewarm set: the request demand a device graph trace of that
/// layout recorded on default adaptive drafts, the always-hot shapes pinned to
/// run in the startup prewarm, and the arena both sides need priced at the
/// layout's measured bytes per graph. Provenance lives in the checked-in
/// document (`glmf_prewarm.json`); pinning is explicit, never inferred from
/// order, and a new trace replaces the set.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmfPrewarmLayout {
    pub measured_bytes: u64,
    pub measured_executables: u64,
    pub reserve_extra_bytes: u64,
    pub entries: Vec<GlmfPrewarmEntry>,
}

impl GlmfPrewarmLayout {
    /// Measured bytes per executable, rounded up: the per-capture bootstrap a
    /// device uses until its own sample replaces it.
    pub fn executable_bytes(&self) -> u64 {
        self.measured_bytes.div_ceil(self.measured_executables.max(1)).max(1)
    }

    /// The measured arena scaled to `executables` captures, without per-graph
    /// rounding: exactly `measured_bytes` at `measured_executables`.
    pub fn arena_bytes(&self, executables: u64) -> u64 {
        let measured = u64::from(self.measured_executables.max(1));
        (u128::from(executables) * u128::from(self.measured_bytes)).div_ceil(u128::from(measured)) as u64
    }
}

/// The checked-in document: one traced set per card layout.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmfPrewarmList {
    pub schema_version: u32,
    pub family: String,
    pub layouts: std::collections::BTreeMap<String, GlmfPrewarmLayout>,
}

impl GlmfPrewarmList {
    /// This layout's traced set.
    pub fn layout(&self, layout: GlmfLayout) -> &GlmfPrewarmLayout {
        &self.layouts[layout.name()]
    }
}

/// The checked-in list, parsed once.
pub fn glmf_prewarm_list() -> &'static GlmfPrewarmList {
    static LIST: std::sync::OnceLock<GlmfPrewarmList> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let list: GlmfPrewarmList = serde_json::from_str(include_str!("glmf_prewarm.json"))
            .expect("checked-in GLM Flash prewarm list");
        assert_eq!(list.schema_version, 2);
        assert_eq!(list.family, "glm5_flash");
        for layout in [GlmfLayout::Min, GlmfLayout::Max] {
            assert!(list.layouts.contains_key(layout.name()), "missing {} layout", layout.name());
        }
        list
    })
}

/// The prewarm shapes this launch can request, in the startup set's order:
/// the checked-in pinned entries of `layout` restricted to the launch's own
/// geometries and row buckets. A context or row set the list does not cover
/// warms nothing.
pub fn glmf_prewarm_shapes(context: usize, pool_tokens: usize, dense: usize, sequences: usize,
    speculation: bool, buckets: &DecodeBuckets, layout: GlmfLayout) -> Vec<(usize, bool, GraphGeometry)> {
    let allowed = serving_graph_shapes(context, pool_tokens.div_ceil(PAGE_ROWS), dense,
        sequences, speculation, buckets);
    let list = glmf_prewarm_list().layout(layout);
    allowed.into_iter().filter(|shape| list.entries.iter()
        .any(|entry| entry.pin && entry.shape() == *shape)).collect()
}

/// Per-device graph reserve: the packed measured arena of this launch's
/// prewarm set (the reserve base), the layout's packed capture headroom and
/// the per-capture bootstrap estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmfGraphReserve {
    /// The layout whose trace priced this reserve.
    pub layout: GlmfLayout,
    /// Packed measured arena per coordinator device (a rank 0 head split
    /// carries its extra head segment).
    pub reserves: Vec<u64>,
    pub executables: Vec<u64>,
    pub extra: u64,
    pub bootstraps: Vec<u64>,
    pub shapes: Vec<(usize, bool, GraphGeometry)>,
}

pub const GRAPH_CHUNK: u64 = 2 << 20;

fn packed(bytes: u64) -> u64 { bytes.div_ceil(GRAPH_CHUNK).saturating_mul(GRAPH_CHUNK) }

/// The graph reserve one launch charges per coordinator device, before KV
/// sizing, from its layout's checked-in prewarm set and measured per-graph price.
#[allow(clippy::too_many_arguments)]
pub fn glmf_graph_reserve(context: usize, pool_tokens: usize, dense: usize, sequences: usize,
    speculation: bool, layers: usize, gpus: usize, decode_rows: usize, sms: usize) -> GlmfGraphReserve {
    let buckets = DecodeBuckets::new(verify_budget(decode_rows, sms));
    let layout = GlmfLayout::for_gpus(gpus);
    let shapes = glmf_prewarm_shapes(context, pool_tokens, dense, sequences.min(decode_rows),
        speculation, &buckets, layout);
    let list = glmf_prewarm_list().layout(layout);
    let each = list.executable_bytes();
    let extra = packed(list.reserve_extra_bytes);
    let executables: Vec<u64> = (0..gpus).map(|rank|
        shapes.len() as u64 * (layers + usize::from(rank == 0)) as u64).collect();
    let reserves = executables.iter().map(|&count| packed(list.arena_bytes(count))).collect();
    GlmfGraphReserve { layout, reserves, executables, extra, bootstraps: vec![each; gpus], shapes }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphGeometry {
    pub pool_width: usize,
    pub page_stride: usize,
    pub pool_stride: usize,
    pub long: bool,
}

impl GraphGeometry {
    /// The table geometry a decode step's graphs are keyed by. The pool top-k (`index_topk`, which
    /// runs only when a row is long) is the only launch that reads the pool table's width and
    /// stride, so a short step keys neither: short steps that differ only there launch the same
    /// programs with the same pointers and scalars, and share their graphs.
    pub fn keyed(pool_width: usize, page_stride: usize, pool_stride: usize, long: bool) -> Self {
        if long { Self { pool_width, page_stride, pool_stride, long } }
        else { Self { pool_width: 0, page_stride, pool_stride: 0, long } }
    }
}

/// The narrowest decode page-table stride, in MLA pages (4,096 tokens): sequences up to that size
/// share one table shape, and so their decode graphs. The index expansion reads only a row's own
/// pages, so a wider row changes nothing but its upload. A long step (a row past the 2,051-token
/// dense context) already has at least 36 pages, so the floor never moves its stride.
pub const MIN_PAGE_STRIDE: usize = 64;

/// A decode step's (page-table, pool-table) strides for sequences of at most `pages` and
/// `pool_pages` pages: powers of two (they bound the graphs a growing batch captures), the page
/// stride from its floor, both at most the pool's pages and a table row's columns (a sequence
/// holds at most `max_context` tokens' pages). Only a long step keys the pool stride.
pub fn decode_strides(pages: usize, pool_pages: usize, pool: (usize, usize), table: (usize, usize)) -> (usize, usize) {
    (pages.max(1).next_power_of_two().max(MIN_PAGE_STRIDE).min(pool.0).min(table.0),
        pool_pages.max(1).next_power_of_two().min(pool.1).min(table.1))
}

pub fn graph_geometries(context: usize, pages: usize, dense: usize) -> Vec<GraphGeometry> {
    let pools = pages / UNIT_PAGES;
    let (table_pages, table_pools) = glmf_table_pages(context as u64);
    let table = (table_pages as usize, table_pools as usize);
    let mut geometries = Vec::new();
    for units in 1..=context.div_ceil(UNIT_ROWS).min(pools) {
        let capacity = (units * UNIT_ROWS).min(context);
        let mut width = 1;
        while width / 2 * UNIT_ROWS < capacity {
            let low = if width == 1 { 1 } else { width / 2 * UNIT_ROWS + 1 };
            let high = (width * UNIT_ROWS).min(capacity);
            for long in [false, true] {
                if (!long && low <= high.min(dense)) || (long && low.max(dense + 1) <= high) {
                    let live_units = units.min(high.max(131_072).div_ceil(UNIT_ROWS).next_power_of_two());
                    let (page_stride, pool_stride) = decode_strides(
                        live_units * UNIT_PAGES, live_units, (pages, pools), table);
                    let geometry = GraphGeometry::keyed(width.min(pool_stride), page_stride, pool_stride, long);
                    if !geometries.contains(&geometry) { geometries.push(geometry); }
                }
            }
            width *= 2;
        }
    }
    geometries
}

pub fn serving_graph_shapes(context: usize, pages: usize, dense: usize, sequences: usize, speculation: bool,
    buckets: &DecodeBuckets) -> Vec<(usize, bool, GraphGeometry)> {
    let plain = buckets.bucket(sequences.clamp(16, DECODE_ROWS), false);
    graph_geometries(context, pages, dense).into_iter().flat_map(|geometry| {
        buckets.plain.iter().copied().filter(move |&rows| rows <= plain)
            .map(move |rows| (rows, false, geometry))
            .chain(buckets.spec.iter().copied().filter(move |_| speculation)
                .map(move |rows| (rows, true, geometry)))
    }).collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn exchange_matches_peer_slots_and_control_floor() {
        assert_eq!(peer_exchange_bytes(2, 4096, 4096, 2, true), 402_653_440);
        assert_eq!(peer_exchange_bytes(2, 4096, 4096, 2, false), 268_435_712);
    }

    /// The checked-in set is the sizing trace's demand (644 graphs at the
    /// reference launch), its three always-hot shapes are the pinned startup
    /// set, and its reserve prices the whole demand plus one largest capture.
    #[test]
    fn traced_prewarm_holds_the_pinned_shape_and_prices_the_traced_demand() {
        let buckets = DecodeBuckets::new(verify_budget(64, 188));
        let shapes = serving_graph_shapes(1_048_576, 2_097_152 / PAGE_ROWS, 2051, 16, true, &buckets);
        assert_eq!(shapes.len(), 300);
        // The list pins the trace's always-hot shapes, not the launch's whole set.
        let warm = glmf_prewarm_shapes(1_048_576, 2_097_152, 2051, 16, true, &buckets, GlmfLayout::Min);
        assert_eq!(warm.len(), 3);
        assert!(warm.iter().all(|shape| shapes.contains(shape)));
        let plan = glmf_graph_reserve(1_048_576, 2_097_152, 2051, 16, true, 45, 1, 64, 188);
        assert_eq!(plan.shapes, warm);
        // The pinned set's share (138 graphs) is the base; the rest of the
        // traced demand (644 graphs less those 138) plus one largest capture
        // is the capture headroom.
        assert_eq!(plan.reserves, [23_068_672]);
        assert_eq!(plan.executables, [138]);
        assert_eq!(plan.extra, 85_983_232);
        assert_eq!(plan.reserves[0] + plan.extra, 109_051_904);
        assert_eq!(plan.bootstraps, [166_079]);
        assert_eq!(verify_budget(128, 170), 127);
        assert_eq!(verify_budget(128, 132), 99);
    }

    /// Each card layout reads its own traced entry: the min column (one
    /// coordinator device) and the max column (two) never share a prewarm or a
    /// price, and the reserve names the layout that sized it.
    #[test]
    fn each_card_layout_reads_its_own_traced_entry() {
        let list = glmf_prewarm_list();
        assert_eq!(GlmfLayout::for_gpus(1), GlmfLayout::Min);
        assert_eq!(GlmfLayout::for_gpus(2), GlmfLayout::Max);
        for layout in [GlmfLayout::Min, GlmfLayout::Max] {
            let priced = list.layout(layout);
            assert!(!priced.entries.is_empty(), "{} has no shapes", layout.name());
            assert!(priced.entries.iter().any(|entry| entry.pin), "{} pins nothing", layout.name());
            assert!(priced.measured_executables >= priced.entries.len() as u64);
        }
        let min = glmf_graph_reserve(1_048_576, 2_097_152, 2051, 16, true, 45, 1, 64, 188);
        assert_eq!(min.layout, GlmfLayout::Min);
        let max = glmf_graph_reserve(1_048_576, 2_097_152, 2051, 16, true, 45, 2, 64, 188);
        assert_eq!(max.layout, GlmfLayout::Max);
        assert_eq!(max.extra, packed(list.layout(GlmfLayout::Max).reserve_extra_bytes));
        assert_eq!(max.executables.len(), 2);
    }

    #[test]
    fn prewarm_shapes_filter_to_the_launch_and_stay_unique() {
        let list = glmf_prewarm_list();
        assert_eq!(list.schema_version, 2);
        assert_eq!(list.family, "glm5_flash");
        let min = list.layout(GlmfLayout::Min);
        assert_eq!(min.executable_bytes(), 166_079);
        assert_eq!(min.reserve_extra_bytes, 85_983_232);
        let mut identities = HashSet::new();
        for entry in &min.entries {
            assert!(identities.insert(entry.shape()), "duplicate prewarm geometry");
        }
        assert_eq!(min.entries.iter().filter(|entry| entry.pin).count(), 3,
            "a trace chooses pins; the rest capture on first use");
        // A launch that cannot key the pinned shapes warms nothing, never a shape
        // outside its own set: every pin is a speculation shape.
        let buckets = DecodeBuckets::new(64);
        assert!(glmf_prewarm_shapes(65_536, 65_536, 2051, 4, false, &buckets, GlmfLayout::Min).is_empty());
        assert_eq!(glmf_prewarm_shapes(65_536, 65_536, 2051, 16, true, &buckets, GlmfLayout::Min).len(), 3);
        let plan = glmf_graph_reserve(65_536, 65_536, 2051, 4, false, 45, 2, 64, 188);
        assert_eq!(plan.executables, [0, 0]);
        // Rank 0 also captures its head segment; chunk rounding can hide the
        // 138-vs-135 difference in the packed counts.
        let buckets = DecodeBuckets::new(verify_budget(64, 188));
        let plan = glmf_graph_reserve(1_048_576, 2_097_152, 2051, 16, true, 45, 2, 64, 188);
        assert_eq!(plan.executables, [138, 135]);
        assert_eq!(plan.reserves.len(), 2);
        assert!(plan.reserves[0] >= plan.reserves[1]);
    }
}

/// Stable launch geometry shared by admission and the executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphKey {
    pub segment: usize,
    pub rows: usize,
    pub spec: bool,
    pub long: bool,
    pub pool_width: usize,
    pub page_stride: usize,
    pub pool_stride: usize,
}

impl GraphKey {
    pub fn new(segment: usize, rows: usize, spec: bool, geometry: GraphGeometry) -> Self {
        let geometry = GraphGeometry::keyed(geometry.pool_width, geometry.page_stride, geometry.pool_stride, geometry.long);
        Self { segment, rows, spec, long: geometry.long, pool_width: geometry.pool_width,
            page_stride: geometry.page_stride, pool_stride: geometry.pool_stride }
    }
}

/// Every exchange slot owns its payload, completion word and native control storage.
pub fn peer_exchange_bytes(lanes: u64, rows: u64, hidden: u64, partial_bytes: u64, output_shard: bool) -> u64 {
    let slots = lanes * if output_shard { 6 } else { 4 };
    slots * (rows.max(64) * hidden * partial_bytes).max(256) + ((slots + 1) * 16).max(256)
}
