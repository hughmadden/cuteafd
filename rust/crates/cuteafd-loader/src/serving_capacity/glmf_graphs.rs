//! GLM 5.3 Flash's serving decode graphs: the row buckets a padded decode step runs at, the table
//! geometries a step keys its graphs by, and the bytes the startup graph set takes. The engine
//! captures and keys its graphs by these; the planner reserves the same set.

/// Rows of the `_m64` decode programs.
const DECODE_ROWS: usize = 64;
/// Plain (one row per sequence) and speculative decode buckets, each aligned to the projections'
/// skinny/wide crossovers.
pub const GLMF_PLAIN_DECODE_BUCKETS: [usize; 6] = [1, 4, 8, 16, 32, 64];
pub const GLMF_SPEC_DECODE_BUCKETS: [usize; 6] = [2, 4, 8, 16, 32, 64];
/// Tokens of one allocation unit (four 64-row MLA pages, one 256-token pool page), and its MLA pages.
const UNIT_ROWS: usize = 256;
const UNIT_PAGES: usize = 4;
/// The narrowest decode page-table stride, in MLA pages (4,096 tokens), and pool-page stride:
/// sequences up to that size share one table shape, and so their decode graphs.
pub const GLMF_MIN_PAGE_STRIDE: usize = 64;
pub const GLMF_MIN_POOL_STRIDE: usize = GLMF_MIN_PAGE_STRIDE / UNIT_PAGES;

/// Physical bytes of one captured decode-segment graph, by GPU target (its SM count):
/// - RTX PRO 6000, 188 SMs (WP9 official-startup-dflash2-20261006-v1, 2026-10-06): 11,040 TP1
///   graphs, 1,616,904,192 physical bytes after fixed workspaces (146,458.713 B/graph).
/// - RTX 5090, 170 SMs (provisional): 1,004,535,808 B of device growth over the 7,018 executables
///   a lazily captured 16-sequence run held (143,136.4 B/graph); the startup capture logs its own.
///
/// Another target takes the larger figure. Remeasure for another graph implementation; round up.
pub const GLMF_GRAPH_BYTES: u64 = 146_459;
pub const GLMF_GRAPH_BYTES_RTX5090: u64 = 143_137;
pub const GLMF_GRAPH_MARGIN_PERCENT: u64 = 20;
pub const GLMF_GRAPH_RANK_MARGIN_BYTES: u64 = 64 << 20;

/// [`GLMF_GRAPH_BYTES`] for a GPU of `sms` SMs.
pub fn glmf_graph_bytes(sms: usize) -> u64 {
    if sms == 170 { GLMF_GRAPH_BYTES_RTX5090 } else { GLMF_GRAPH_BYTES }
}

/// Bytes `graphs` decode-segment graphs take on a GPU of `sms` SMs, with the margins.
pub fn glmf_graph_reserve_bytes(graphs: usize, sms: usize) -> u64 {
    let measured = graphs as u64 * glmf_graph_bytes(sms);
    measured + (measured * GLMF_GRAPH_MARGIN_PERCENT).div_ceil(100) + GLMF_GRAPH_RANK_MARGIN_BYTES
}

/// The rows a speculative step of `rows` rows runs as with the fine row buckets
/// (`--decode-row-buckets`): exact to 16 rows, then whole steps of 4 rows to 64 and of 8 to 128,
/// never past the verify budget `verify_rows` and never fewer than its own rows.
pub fn glmf_fine_bucket(rows: usize, verify_rows: usize) -> usize {
    let bucket = match rows {
        0..=16 => rows,
        17..=DECODE_ROWS => rows.next_multiple_of(4),
        _ => rows.next_multiple_of(8),
    };
    bucket.min(verify_rows).max(rows)
}

/// The row buckets padded decode steps run at: plain and speculative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmfDecodeBuckets {
    pub plain: Vec<usize>,
    pub spec: Vec<usize>,
    /// `--decode-rows 128`: speculative buckets past 64 rows run the `_m128` programs.
    pub wide: bool,
}

impl GlmfDecodeBuckets {
    /// The default buckets; with `fine` (`--decode-row-buckets`) the speculative ones are
    /// `glmf_fine_bucket`'s. With `--decode-rows 128` the speculative set goes on to the verify
    /// budget `verify_rows` (127 on an RTX 5090, 128 on 188 SMs: a step's whole sparse MLA waves).
    pub fn new(decode_rows: usize, verify_rows: usize, fine: bool) -> Self {
        let spec: Vec<usize> = if fine {
            let set: std::collections::BTreeSet<usize> = (1..=verify_rows)
                .map(|rows| glmf_fine_bucket(rows, verify_rows)).collect();
            set.into_iter().collect()
        } else {
            GLMF_SPEC_DECODE_BUCKETS.into_iter().filter(|&rows| rows < verify_rows).chain([verify_rows]).collect()
        };
        Self { plain: GLMF_PLAIN_DECODE_BUCKETS.to_vec(), spec, wide: decode_rows > DECODE_ROWS }
    }

    /// The smallest bucket that holds `rows` (a step past every bucket keeps its rows).
    pub fn bucket(&self, rows: usize, spec: bool) -> usize {
        let set = if spec { &self.spec } else { &self.plain };
        set.iter().copied().find(|&bucket| bucket >= rows).unwrap_or(rows)
    }

    /// The plain buckets a serving loop of `sequences` reaches (every one through 16).
    pub fn plain_for(&self, sequences: usize) -> Vec<usize> {
        let cap = self.bucket(sequences.clamp(16, DECODE_ROWS), false);
        self.plain.iter().copied().filter(|&rows| rows <= cap).collect()
    }

    /// The startup graphs' row shapes: the plain buckets of `sequences`, then the speculative ones.
    pub fn shapes(&self, sequences: usize, speculation: bool) -> Vec<(usize, bool)> {
        self.plain_for(sequences).into_iter().map(|rows| (rows, false))
            .chain(self.spec.iter().copied().filter(|_| speculation).map(|rows| (rows, true))).collect()
    }
}

/// A decode step's (page-table, pool-table) strides for sequences of at most `pages` and
/// `pool_pages` pages: powers of two from a floor (they bound the graphs a growing batch captures),
/// at most the pool's pages and a table row's columns (a sequence holds at most `max_context`
/// tokens' pages).
pub fn glmf_decode_strides(pages: usize, pool_pages: usize, pool: (usize, usize), table: (usize, usize))
    -> (usize, usize) {
    (pages.max(1).next_power_of_two().max(GLMF_MIN_PAGE_STRIDE).min(pool.0).min(table.0),
        pool_pages.max(1).next_power_of_two().max(GLMF_MIN_POOL_STRIDE).min(pool.1).min(table.1))
}

/// What a decode graph's tables bake in besides its segment and rows. A step whose rows are all
/// short (none past `dense`, the indexer-free context) keys no pool top-k width or stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlmfGraphGeometry {
    pub pool_width: usize,
    pub page_stride: usize,
    pub pool_stride: usize,
    pub long: bool,
}

/// Every geometry a decode step over a pool of `pages` pages reaches, for sequences of up to
/// `context` tokens whose tables have `table` columns: power-of-two strides from their floor, and
/// widths keyed only when long.
pub fn glmf_graph_geometries(context: usize, pages: usize, dense: usize, table: (usize, usize))
    -> Vec<GlmfGraphGeometry> {
    let pools = pages / UNIT_PAGES;
    let mut geometries = Vec::new();
    for units in 1..=context.div_ceil(UNIT_ROWS).min(pools) {
        let (page_stride, pool_stride) = glmf_decode_strides(units * UNIT_PAGES, units, (pages, pools), table);
        let capacity = (units * UNIT_ROWS).min(context);
        let mut width = 1;
        while width / 2 * UNIT_ROWS < capacity {
            let low = if width == 1 { 1 } else { width / 2 * UNIT_ROWS + 1 };
            let high = (width * UNIT_ROWS).min(capacity);
            for long in [false, true] {
                if (!long && low <= high.min(dense)) || (long && low.max(dense + 1) <= high) {
                    let geometry = if long {
                        GlmfGraphGeometry { pool_width: width.min(pool_stride), page_stride, pool_stride, long }
                    } else {
                        GlmfGraphGeometry { pool_width: 0, page_stride, pool_stride: 0, long }
                    };
                    if !geometries.contains(&geometry) { geometries.push(geometry); }
                }
            }
            width *= 2;
        }
    }
    geometries
}

/// The startup graph set's step shapes: every geometry at every bucket a serving loop of
/// `sequences` reaches.
pub fn glmf_graph_shapes(context: usize, pages: usize, dense: usize, table: (usize, usize),
    buckets: &GlmfDecodeBuckets, sequences: usize, speculation: bool) -> Vec<(usize, bool, GlmfGraphGeometry)> {
    let rows = buckets.shapes(sequences, speculation);
    glmf_graph_geometries(context, pages, dense, table).into_iter()
        .flat_map(|geometry| rows.iter().map(move |&(rows, spec)| (rows, spec, geometry))).collect()
}

/// The bytes serve-glmf's startup graph set takes on one GPU (`cfg.layers + 1` segments a shape)
/// at `context` tokens, a pool of `pool_tokens` (an automatic pool's target bound), `sequences`
/// with drafts (DFlash2 or copy windows), `decode_rows` decode rows (the verify budget of a 188-SM
/// card) and the `fine` buckets: planned at the larger card's bytes per graph.
pub fn glmf_startup_graph_bytes(cfg: &crate::families::glm5_flash::GlmNextConfig, context: usize, pool_tokens: usize,
    sequences: usize, decode_rows: usize, fine: bool) -> u64 {
    let (table_pages, table_pools) = super::glmf_table_pages(context as u64);
    let buckets = GlmfDecodeBuckets::new(decode_rows, decode_rows, fine);
    let shapes = glmf_graph_shapes(context, pool_tokens.div_ceil(64), cfg.dense_context(),
        (table_pages as usize, table_pools as usize), &buckets, sequences.clamp(1, DECODE_ROWS), true).len();
    glmf_graph_reserve_bytes(shapes * (cfg.layers + 1), 188)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_buckets_end_at_the_verify_budget() {
        assert_eq!(GlmfDecodeBuckets::new(64, 64, false).spec, GLMF_SPEC_DECODE_BUCKETS);
        assert_eq!(GlmfDecodeBuckets::new(128, 127, false).spec, [2, 4, 8, 16, 32, 64, 127]);
        let fine = GlmfDecodeBuckets::new(128, 127, true);
        assert_eq!((fine.spec.len(), fine.spec[15], fine.spec.last()), (36, 16, Some(&127)));
        assert_eq!(GlmfDecodeBuckets::new(64, 64, true).spec.len(), 16 + 12);
    }

    /// 32K context, 16 sequences, drafts: 14 geometries (four short ones by page stride, ten long),
    /// 140 shapes; 131,072 tokens: 27 geometries.
    #[test]
    fn short_steps_share_their_geometries() {
        let buckets = GlmfDecodeBuckets::new(64, 64, false);
        assert_eq!(glmf_graph_geometries(32_768, 4096, 2051, (4096, 1024)).len(), 14);
        assert_eq!(glmf_graph_shapes(32_768, 4096, 2051, (4096, 1024), &buckets, 16, true).len(), 140);
        assert_eq!(glmf_graph_geometries(131_072, 32_768, 2051, (2048, 512)).len(), 27);
        assert_eq!(glmf_graph_reserve_bytes(6440, 188), 1_198_944_016);
        assert_eq!(glmf_graph_reserve_bytes(13_662, 170), 2_413_754_097);
    }
}
