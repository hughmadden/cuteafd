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

/// The resolved shape a launch warms graphs for: its coordinator device count
/// and whether Spark ranks serve its routed experts. Expert onboarding
/// (Sparks) and TP2 (two devices) each add their own captured segments, so
/// every combination reads its own traced entry — never another shape's list.
/// A key with no trace warms [`BOOTSTRAP_SPEC_ROWS`] instead. A head split's
/// second coordinator GPU is not a Spark attachment: a 2-RTX launch whose
/// experts stay local TP2 (its Sparks idle) is `2`, not `2-sparks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlmfLayout {
    pub devices: usize,
    pub sparks: bool,
}

impl GlmfLayout {
    /// The key a launch with `devices` coordinator GPUs, with or without
    /// Sparks attached, serves under.
    pub fn resolved(devices: usize, sparks: bool) -> Self { Self { devices, sparks } }

    /// The checked-in key: `<devices>`, or `<devices>-sparks` with Sparks.
    pub fn key(self) -> String {
        if self.sparks { format!("{}-sparks", self.devices) } else { self.devices.to_string() }
    }
}

impl std::fmt::Display for GlmfLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.key()) }
}

/// The always-hot buckets every GLM Flash launch treats as its own: a
/// single-row plain step and the narrow speculative buckets, in the short
/// geometry. A layout with no traced key warms exactly these, unpinned
/// (no trace measured a pin), and lets ARC decide.
pub const BOOTSTRAP_PLAIN_ROWS: [usize; 1] = [1];
pub const BOOTSTRAP_SPEC_ROWS: [usize; 3] = [4, 8, 16];

fn is_bootstrap_shape(shape: &(usize, bool, GraphGeometry)) -> bool {
    let (rows, spec, geometry) = *shape;
    !geometry.long && if spec { BOOTSTRAP_SPEC_ROWS.contains(&rows) } else { BOOTSTRAP_PLAIN_ROWS.contains(&rows) }
}

/// One layout's prewarm set: the request demand a device graph trace of that
/// layout recorded on default adaptive drafts, the always-hot shapes pinned
/// against eviction, and the arena both sides need priced at the layout's
/// measured bytes per graph. Startup prewarms every entry, so requests hit
/// their graphs instead of capturing on latency. Provenance lives in the
/// checked-in document (`glmf_prewarm.json`); pinning is explicit, never
/// inferred from order, and a new trace replaces the set.
///
/// The entries are the same for every quant of the checkpoint: a shape is a
/// pure function of the launch geometry, and one decode site captures it
/// whatever the expert format. The price is not: a captured graph's
/// executable memory follows its kernels' node structure, so a layout traced
/// on more than one quant charges the costlier quant's measured arena.
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

/// The checked-in document: one traced set per resolved card shape, keyed
/// `<devices>[-sparks]`. A key the document does not hold warms the bootstrap
/// set instead of borrowing another shape's list. A layout traced on more than
/// one quant of the checkpoint holds one entry set (the shapes are
/// quant-independent) priced at the costlier quant measured on that layout.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmfPrewarmList {
    pub schema_version: u32,
    pub family: String,
    pub layouts: std::collections::BTreeMap<String, GlmfPrewarmLayout>,
}

impl GlmfPrewarmList {
    /// This layout's traced set, or `None` when that resolved shape has no trace.
    pub fn layout(&self, layout: GlmfLayout) -> Option<&GlmfPrewarmLayout> {
        self.layouts.get(&layout.key())
    }

    /// The measured price an untraced layout borrows for its bootstrap set: the
    /// dearest measured per-graph cost among the traced keys, never a shape
    /// list. Provenance for that estimate is the traced entry it came from.
    pub fn priciest(&self) -> &GlmfPrewarmLayout {
        self.layouts.values().max_by_key(|layout| layout.executable_bytes())
            .expect("a checked-in GLM Flash list holds at least one traced layout")
    }
}

/// The checked-in list, parsed once.
pub fn glmf_prewarm_list() -> &'static GlmfPrewarmList {
    static LIST: std::sync::OnceLock<GlmfPrewarmList> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let list: GlmfPrewarmList = serde_json::from_str(include_str!("glmf_prewarm.json"))
            .expect("checked-in GLM Flash prewarm list");
        assert_eq!(list.schema_version, 3);
        assert_eq!(list.family, "glm5_flash");
        assert!(!list.layouts.is_empty(), "a checked-in list holds at least one traced layout");
        for (key, layout) in &list.layouts {
            assert!(matches!(key.as_str(), "1" | "1-sparks" | "2" | "2-sparks"),
                "unknown GLM Flash layout key {key}");
            assert!(!layout.entries.is_empty(), "{key} traces no shapes");
            assert!(layout.measured_executables >= layout.entries.len() as u64,
                "{key} prices fewer executables than it warms");
            let mut shapes = std::collections::HashSet::new();
            for entry in &layout.entries {
                assert!(shapes.insert(entry.shape()), "{key} repeats a prewarm geometry");
            }
        }
        list
    })
}

/// Warn once per process that a layout has no trace, naming the missing key.
fn warn_untraced(layout: GlmfLayout) {
    use std::sync::{Mutex, OnceLock};
    static WARNED: OnceLock<Mutex<std::collections::BTreeSet<String>>> = OnceLock::new();
    let mut warned = WARNED.get_or_init(Default::default).lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if warned.insert(layout.key()) {
        tracing::warn!(layout = %layout, "no GLM Flash graph prewarm trace for this layout; \
            warming the bootstrap startup set — trace this shape (RTX count and Sparks attached) to price it");
    }
}

/// The prewarm shapes this launch can request: every checked-in entry of
/// `layout` restricted to the launch's own geometries and row buckets, in the
/// list's order. A context or row set the list does not cover warms nothing.
/// A layout with no trace warms its bootstrap set at the launch's geometries.
pub fn glmf_prewarm_shapes(context: usize, pool_tokens: usize, dense: usize, sequences: usize,
    speculation: bool, buckets: &DecodeBuckets, layout: GlmfLayout) -> Vec<(usize, bool, GraphGeometry)> {
    let allowed = serving_graph_shapes(context, pool_tokens.div_ceil(PAGE_ROWS), dense,
        sequences, speculation, buckets);
    match glmf_prewarm_list().layout(layout) {
        Some(traced) => allowed.into_iter().filter(|shape| traced.entries.iter()
            .any(|entry| entry.shape() == *shape)).collect(),
        None => allowed.into_iter().filter(is_bootstrap_shape).collect(),
    }
}

/// Whether `layout` pins `shape` against eviction: only a trace-derived entry
/// pins. A layout with no trace warms its bootstrap set unpinned and lets ARC
/// decide, so an untraced shape never claims a pin nothing measured.
pub fn glmf_pinned(layout: GlmfLayout, shape: &(usize, bool, GraphGeometry)) -> bool {
    match glmf_prewarm_list().layout(layout) {
        Some(traced) => traced.entries.iter().any(|entry| entry.pin && entry.shape() == *shape),
        None => false,
    }
}

/// Per-device graph reserve: the packed measured arena of this launch's
/// prewarm set (the reserve base), the layout's packed capture headroom and
/// the per-capture bootstrap estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmfGraphReserve {
    /// The resolved shape whose trace (or bootstrap set) priced this reserve.
    pub layout: GlmfLayout,
    /// Whether the checked-in list holds this layout's own trace.
    pub traced: bool,
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
/// sizing, from its layout's checked-in prewarm set and measured per-graph
/// price. A layout with no trace charges its bootstrap set at the dearest
/// measured per-graph cost, so an untraced shape never borrows another
/// shape's list or a smaller price.
#[allow(clippy::too_many_arguments)]
pub fn glmf_graph_reserve(context: usize, pool_tokens: usize, dense: usize, sequences: usize,
    speculation: bool, layers: usize, gpus: usize, decode_rows: usize, sms: usize, sparks: bool)
    -> GlmfGraphReserve {
    let buckets = DecodeBuckets::new(verify_budget(decode_rows, sms));
    let layout = GlmfLayout::resolved(gpus, sparks);
    let shapes = glmf_prewarm_shapes(context, pool_tokens, dense, sequences.min(decode_rows),
        speculation, &buckets, layout);
    let list = glmf_prewarm_list();
    let traced = list.layout(layout);
    if traced.is_none() { warn_untraced(layout); }
    let price = traced.unwrap_or_else(|| list.priciest());
    let each = price.executable_bytes();
    let extra = packed(price.reserve_extra_bytes);
    let executables: Vec<u64> = (0..gpus).map(|rank|
        shapes.len() as u64 * (layers + usize::from(rank == 0)) as u64).collect();
    let reserves = executables.iter().map(|&count| packed(price.arena_bytes(count))).collect();
    GlmfGraphReserve { layout, traced: traced.is_some(), reserves, executables, extra,
        bootstraps: vec![each; gpus], shapes }
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

    /// The checked-in set is the sizing trace's demand (690 graphs at the
    /// reference launch), every traced shape prewarms at startup, its two
    /// always-hot shapes are pinned against eviction, and its reserve prices
    /// the whole demand plus one largest capture.
    #[test]
    fn traced_prewarm_holds_the_pinned_shape_and_prices_the_traced_demand() {
        let buckets = DecodeBuckets::new(verify_budget(64, 188));
        let shapes = serving_graph_shapes(1_048_576, 2_097_152 / PAGE_ROWS, 2051, 16, true, &buckets);
        assert_eq!(shapes.len(), 300);
        let layout = GlmfLayout::resolved(1, true);
        // Startup prewarms the trace's shapes, not the launch's whole set.
        let warm = glmf_prewarm_shapes(1_048_576, 2_097_152, 2051, 16, true, &buckets, layout);
        assert_eq!(warm.len(), 15);
        assert!(warm.iter().all(|shape| shapes.contains(shape)));
        let list = glmf_prewarm_list().layout(layout).expect("the min column is traced");
        assert!(list.entries.iter().filter(|entry| entry.pin)
            .all(|entry| warm.contains(&entry.shape())), "every pin prewarms");
        let pins: HashSet<_> = list.entries.iter().filter(|entry| entry.pin)
            .map(|entry| entry.shape()).collect();
        assert!(warm.iter().all(|shape| glmf_pinned(layout, shape) == pins.contains(shape)));
        let plan = glmf_graph_reserve(1_048_576, 2_097_152, 2051, 16, true, 45, 1, 64, 188, true);
        assert_eq!(plan.layout, layout);
        assert!(plan.traced);
        assert_eq!(plan.shapes, warm);
        // The traced demand (690 graphs) is the base; one packed largest
        // capture is headroom for a shape the trace did not cover.
        assert_eq!(plan.reserves, [119_537_664]);
        assert_eq!(plan.executables, [690]);
        assert_eq!(plan.extra, 2_097_152);
        assert_eq!(plan.reserves[0] + plan.extra, 121_634_816);
        assert_eq!(plan.bootstraps, [173_243]);
        assert_eq!(verify_budget(128, 170), 127);
        assert_eq!(verify_budget(128, 132), 99);
    }

    /// A key with no trace warms only the bootstrap set at its own geometries,
    /// pins nothing (only a trace pins), and charges it at the dearest measured
    /// per-graph cost — never another shape's list and never a smaller price.
    #[test]
    fn untraced_layout_warms_only_the_bootstrap_set() {
        let layout = GlmfLayout::resolved(1, false);
        assert!(glmf_prewarm_list().layout(layout).is_none(), "the probe layout has no trace");
        let buckets = DecodeBuckets::new(verify_budget(64, 188));
        let admitted = serving_graph_shapes(1_048_576, 2_097_152 / PAGE_ROWS, 2051, 16, true, &buckets);
        let warm = glmf_prewarm_shapes(1_048_576, 2_097_152, 2051, 16, true, &buckets, layout);
        assert!(!warm.is_empty());
        assert!(warm.iter().all(|shape| admitted.contains(shape)));
        assert!(warm.iter().all(|&(rows, spec, geometry)| !geometry.long
            && if spec { BOOTSTRAP_SPEC_ROWS.contains(&rows) } else { rows == 1 }));
        assert!(!warm.iter().any(|shape| glmf_pinned(layout, shape)),
            "an untraced layout pins nothing; only a trace-derived entry pins");
        assert!(!warm.iter().any(|&(rows, spec, _)| spec && rows > 16));
        let plan = glmf_graph_reserve(1_048_576, 2_097_152, 2051, 16, true, 45, 1, 64, 188, false);
        assert_eq!(plan.layout, layout);
        assert!(!plan.traced);
        assert_eq!(plan.shapes, warm);
        assert_eq!(plan.executables, [warm.len() as u64 * 46]);
        let dearest = glmf_prewarm_list().priciest();
        assert_eq!(plan.bootstraps, [dearest.executable_bytes()]);
        assert_eq!(plan.extra, packed(dearest.reserve_extra_bytes));
    }

    /// Each resolved card shape reads only its own traced entry: one device
    /// with Sparks (the min column) never shares a prewarm or a price with
    /// another shape, and the reserve names the layout that sized it.
    #[test]
    fn each_resolved_shape_reads_only_its_own_entry() {
        let list = glmf_prewarm_list();
        assert_eq!(list.schema_version, 3);
        assert_eq!(list.family, "glm5_flash");
        for (key, priced) in &list.layouts {
            assert!(!priced.entries.is_empty(), "{key} has no shapes");
            assert!(priced.entries.iter().any(|entry| entry.pin), "{key} pins nothing");
            assert!(priced.measured_executables >= priced.entries.len() as u64);
            assert!(priced.measured_bytes >= priced.measured_executables);
        }
        // The shipping min column is traced; its absence would silently price a
        // bootstrap set, so the key must never drift out of the document.
        let min = glmf_graph_reserve(1_048_576, 2_097_152, 2051, 16, true, 45, 1, 64, 188, true);
        assert_eq!(min.layout, GlmfLayout::resolved(1, true));
        assert!(min.traced);
        let two = glmf_graph_reserve(1_048_576, 2_097_152, 2051, 16, true, 45, 2, 64, 188, false);
        assert_eq!(two.layout, GlmfLayout::resolved(2, false));
        assert!(two.traced, "a 2-RTX launch with local TP2 experts is traced");
        assert_eq!(two.executables.len(), 2);
        // The `2` base is its selected entries' measured capture bytes on the
        // costlier device (K6 locally: 136,314,880 B for 690 executables), and
        // its head segment prices the second device's 675 executables.
        assert_eq!(two.reserves, [136_314_880, 134_217_728]);
        assert_eq!(two.bootstraps, [197_558, 197_558]);
        // Sparks idle behind a 2-RTX local-expert launch are not an expert
        // attachment: that launch keys `2`, the same resolved shape it serves.
        assert_eq!(GlmfLayout::resolved(2, false).key(), two.layout.key());
    }

    #[test]
    fn prewarm_shapes_filter_to_the_launch_and_stay_unique() {
        let list = glmf_prewarm_list();
        let min = list.layout(GlmfLayout::resolved(1, true)).expect("the min column is traced");
        assert_eq!(min.executable_bytes(), 173_243);
        assert_eq!(min.reserve_extra_bytes, 2_097_152);
        let mut identities = HashSet::new();
        for entry in &min.entries {
            assert!(identities.insert(entry.shape()), "duplicate prewarm geometry");
        }
        assert_eq!(min.entries.iter().filter(|entry| entry.pin).count(), 2,
            "a trace chooses pins; every other traced shape stays evictable");
        // A launch warms exactly the traced shapes its own geometries and row
        // buckets admit: the rows=1 plain shape at both its short and long
        // geometries (two long pool widths), and every traced shape the
        // speculative set can request.
        let layout = GlmfLayout::resolved(1, true);
        let buckets = DecodeBuckets::new(64);
        let plain = serving_graph_shapes(65_536, 65_536, 2051, 4, false, &buckets);
        let plain_warm = glmf_prewarm_shapes(65_536, 65_536, 2051, 4, false, &buckets, layout);
        assert_eq!(plain_warm.len(), 3);
        assert!(plain_warm.iter().all(|shape| shape.0 == 1), "one plain row bucket");
        assert!(plain_warm.iter().all(|shape| plain.contains(shape)));
        let speculative = serving_graph_shapes(65_536, 65_536, 2051, 16, true, &buckets);
        let spec_warm = glmf_prewarm_shapes(65_536, 65_536, 2051, 16, true, &buckets, layout);
        assert_eq!(spec_warm.len(), min.entries.iter()
            .filter(|entry| speculative.contains(&entry.shape())).count());
        assert!(min.entries.iter().filter(|entry| entry.pin)
            .all(|entry| spec_warm.contains(&entry.shape())), "every admitted pin prewarms");
        let plan = glmf_graph_reserve(65_536, 65_536, 2051, 4, false, 45, 1, 64, 188, true);
        assert_eq!(plan.executables, [3 * 46]);
        // Rank 0 also captures its head segment, so it holds one more
        // executable per shape than its peer.
        let plan = glmf_graph_reserve(1_048_576, 2_097_152, 2051, 16, true, 45, 2, 64, 188, true);
        let shapes = plan.shapes.len() as u64;
        assert_eq!(plan.executables, [shapes * 46, shapes * 45]);
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
