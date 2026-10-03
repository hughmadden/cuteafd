//! GLM 5.3 Flash (glm5_next) coordinator over the exported glmf_* programs.
//!
//! Four mHC streams (BF16 `[rows, 4, H]`) run through every layer: the
//! attention site's collapse and input norm (`mhc_pre`, or the previous
//! layer's fused `mhc_post_pre`), the attention sublayer, `mhc_post_pre`
//! back into the streams and onto the FFN site, the FFN, and the next fused
//! post/pre (`mhc_post` after the last layer, then the stream-mean head).
//!
//! Attention: KDA layers keep per-sequence recurrent state (FP32
//! `[64, 128, 128]`) and short-conv state (the last three q/k/v inputs) in
//! slot pools, one slot per sequence shared by every KDA layer (every KDA
//! layer's pool back to back, so `glmf_kda_commit` reaches all of them in one
//! launch). A speculative verify step (`verify_spec`) leaves that state alone
//! and records each row's replay inputs; `commit` then applies the accepted
//! rows with the recurrent step's own arithmetic. MLA layers
//! keep FP8 528-byte latent records in 64-row pages, and their DSA indexer
//! keeps per-token BF16 keys and gates beside the records and one FP8 key
//! per completed 4-token pool in pool pages (64 pools per page). The
//! indexer selects every earlier token up to 2051 tokens; past that the top
//! 512 pools (glmf_index_topk) expand to tokens plus the open tail pool.
//!
//! FFN: dense SwiGLU (clamped at 10), or the MoE: FP32 router logits, the
//! native sigmoid top-8 select, the shared expert, and routed experts from
//! the checkpoint's FP8/NVFP4 on this GPU (the TP1 package, all routed layers
//! resident by default; a paging window only when explicitly requested for
//! diagnostics) or on the Sparks.
use super::head::GlmfHead;
use super::weights::{GlmfLayer, GlmfWeights};
use crate::families::glm5::dflash::TargetHead;
use crate::shared::experts::fp8::{Fp8Experts, Fp8Layer};
use crate::shared::launch_grid::Fp8QuantizeGrid;
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use crate::shared::token_io::{DeviceLogits, EmbedPlacement, TokenEmbedding};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Programs, Scalar, VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors;
use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
use crate::shared::spark_intake::SparkLink;
use cuteafd_transport::expert::{SparkExpertWave, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};
use std::cell::RefCell;
use std::ffi::c_void;

type Dev<'a> = DeviceAllocation<'a>;

pub(crate) const PAGE_ROWS: usize = 64;
/// Rows of the decode-route programs (`_m64`).
pub(crate) const DECODE_ROWS: usize = 64;
/// Selected-slot row width of the sparse MLA programs (2048 + 3, padded to 64).
pub(crate) const SPARSE_TOPK: usize = 2112;
/// Most decode rows the FP8 GEMVs take (the programs' FP8_ROWS).
const FP8_ROWS: i32 = 16;
/// FP8 latent record bytes (512 E4M3 + 4 FP32 group scales).
pub(crate) const RECORD_BYTES: usize = 528;
/// Rows a speculative step records per KDA layer (`REPLAY_ROWS` of the fork's
/// `_glmf_kernels.py`; the decode programs' rows).
const REPLAY_ROWS: usize = DECODE_ROWS;

/// Bytes of one KDA layer's replay record (`kda_replay_layout`): k | decay | v
/// FP32 per row and head, beta FP32, the q/k/v in-projection row BF16.
fn replay_bytes(heads: usize, channels: usize) -> usize {
    REPLAY_ROWS * heads * 3 * 128 * 4 + REPLAY_ROWS * heads * 4 + REPLAY_ROWS * channels * 2
}
const MAX_RANKS: usize = 6;
/// Lanes a long Spark prefill chunk splits into (one lane's GPU layers run
/// while the other lane's Spark wave is in flight), and the fewest rows per
/// lane worth a second exchange per layer.
pub(crate) const PREFILL_LANES: usize = 2;
const MIN_LANE_ROWS: usize = 256;

/// Multi-lane splits must land on MLA page boundaries. A narrow workspace
/// still accepts a single unpadded tail, but cannot advertise unusable padding.
fn prefill_lane_capacity(rows: usize) -> usize {
    rows.max(PREFILL_LANES * (rows / PAGE_ROWS) * PAGE_ROWS)
}

fn prefill_lane_plan(tokens: usize, rows: usize) -> Result<(usize, usize)> {
    ensure!(tokens > 0 && tokens <= prefill_lane_capacity(rows),
        "prefill of {tokens} tokens exceeds the {rows}-row lane workspaces");
    let lanes = if tokens <= rows && tokens < PREFILL_LANES * MIN_LANE_ROWS { 1 } else { PREFILL_LANES };
    let per_lane = if lanes == 1 { tokens } else { tokens.div_ceil(lanes).next_multiple_of(PAGE_ROWS) };
    ensure!(per_lane <= rows, "prefill lane of {per_lane} tokens exceeds {rows} rows");
    Ok((lanes, per_lane))
}
const HC: usize = 4;

/// Routed experts on this GPU from the TP1 package. All layers stay resident
/// unless an explicit diagnostic paging window was requested.
pub(crate) struct LocalExperts<'a> {
    pub library: &'a NativeLibrary,
    pub tensors: &'a Fp8ExpertTensors,
    pub experts: RefCell<Fp8Experts<'a>>,
    /// Explicit diagnostic paging; absent for the fully resident serving path.
    pub window: Option<usize>,
    pub loads: RefCell<usize>,
}

impl LocalExperts<'_> {
    fn index_of(&self, layer: usize) -> Result<usize> {
        let mut experts = self.experts.borrow_mut();
        if let Ok(index) = experts.index_of(layer) {
            return Ok(index);
        }
        let window = self.window.context("local expert layer missing from the admitted resident set")?;
        if experts.layers.len() >= window {
            experts.layers.remove(0);
        }
        let started = std::time::Instant::now();
        experts.layers.push(Fp8Layer::load(self.library, self.tensors, layer, 1, 0)?);
        *self.loads.borrow_mut() += 1;
        tracing::debug!(layer, elapsed_ms = started.elapsed().as_millis() as u64, "FP8 expert layer loaded");
        Ok(experts.layers.len() - 1)
    }
}

/// Routed experts on this GPU from the EXL3 checkpoint (the coordinator's
/// `exl3-glmf-k<tiers>/rtx-tp1` package): a window of resident layers,
/// reloaded as the step walks the layers. The package reads FP8 K32 wire
/// rows (as the Sparks do) and its reducer adds the shared expert.
pub(crate) struct LocalExl3<'a> {
    pub library: &'a NativeLibrary,
    pub native_lib: std::path::PathBuf,
    pub catalog: &'a cuteafd_loader::OfficialV41Catalog,
    pub resident: RefCell<Option<(std::ops::Range<usize>, crate::families::deepseek_v4::local::LocalExperts<'a>)>>,
    pub window: usize,
    /// Layers past the engine's last one never load.
    pub layers: usize,
    pub max_rows: usize,
    pub budget: usize,
    pub loads: RefCell<usize>,
}

impl LocalExl3<'_> {
    /// Makes `layer` resident (with the next `window - 1` layers).
    fn ensure(&self, layer: usize, stream: *mut c_void) -> Result<()> {
        if self.resident.borrow().as_ref().is_some_and(|(range, _)| range.contains(&layer)) {
            return Ok(());
        }
        // SAFETY: the engine owns this stream; the old window's launches drain first.
        unsafe { self.library.cuda_stream_synchronize(stream)? };
        *self.resident.borrow_mut() = None;
        let range = layer..(layer + self.window).min(self.layers);
        let started = std::time::Instant::now();
        let local = crate::families::deepseek_v4::local::LocalExperts::load_range(self.library, &self.native_lib, self.catalog, 0,
            range.clone(), self.max_rows, self.budget, stream)?
            .context("no coordinator EXL3 package for this checkpoint (build glmf:exl3-k<tiers>)")?;
        let range = layer..layer + local.layers();
        ensure!(range.contains(&layer), "EXL3 expert layer {layer} does not fit the budget");
        *self.loads.borrow_mut() += range.len();
        tracing::debug!(?range, elapsed_ms = started.elapsed().as_millis() as u64, "EXL3 expert window resident");
        *self.resident.borrow_mut() = Some((range, local));
        Ok(())
    }
}

/// Where the routed experts run.
pub(crate) enum Experts<'a> {
    Local(LocalExperts<'a>),
    LocalExl3(LocalExl3<'a>),
    /// Spark ranks serving the routed experts over RoCE (one BF16 partial per
    /// rank); one transport per prefill lane (decode uses the first).
    Spark { transports: RefCell<Vec<SparkLink<'a>>>, runtime: tokio::runtime::Runtime },
    /// Profiling only: the router, the wire rows and the shared expert run,
    /// the routed experts contribute nothing (the coordinator's own work).
    Skip,
}

/// Per-program GPU times (CUTEAFD_GLMF_PROFILE_OPS=1): timing events around
/// every launch, read back by [`GlmfEngine::op_profile`].
#[derive(Default)]
pub(crate) struct OpTimes {
    pending: Vec<(String, *mut c_void, *mut c_void)>,
    pool: Vec<*mut c_void>,
    pub totals: std::collections::BTreeMap<String, (f64, usize)>,
}

/// Host tables of one step.
#[derive(Default)]
struct StepTables {
    decode: bool,
    positions: Vec<i64>,
    /// MLA latent record slot per row (also the row's token-key slot).
    kv_slots: Vec<i64>,
    /// KDA state slot per row.
    kda_slots: Vec<i32>,
    /// First step row of each row's sequence.
    seq_first: Vec<i32>,
    /// Pool index-cache slot of the pool a row completes, else -1.
    pool_slots: Vec<i64>,
    /// Complete pools each row sees.
    cache_lengths: Vec<i32>,
    /// Record pages and pool pages: one shared table (prefill) or one padded row per step row.
    page_table: Vec<i32>,
    pool_table: Vec<i32>,
    page_stride: usize,
    pool_stride: usize,
    /// Pool-table columns the top-k reads.
    pool_width: usize,
    /// Whether any row sees more than 2051 tokens (the pool top-k runs).
    long: bool,
    /// A speculative verify: KDA state stays, replay rows are recorded.
    spec: bool,
}

/// Tokens per DSA index pool, and pools per pool-cache page.
pub(crate) const KPOOL: usize = 4;
const POOL_PAGE_TOKENS: usize = KPOOL * PAGE_ROWS;

/// MLA pages per allocation unit: a unit is four consecutive 64-row MLA pages (256 tokens) and
/// the pool-cache page of the same index (64 pools of 4 tokens), so one refcounted unit index
/// names every paged byte of 256 positions (the prefix cache's page).
pub(crate) const UNIT_PAGES: usize = KPOOL;
pub(crate) const UNIT_ROWS: usize = POOL_PAGE_TOKENS;

/// A sequence's allocation units (and the MLA and pool pages they expand to), its KDA state
/// slot, its length and the rows its KDA state holds.
#[derive(Debug, Clone)]
pub(crate) struct GlmfPlacement {
    pub units: Vec<u32>,
    pub pages: Vec<i32>,
    pub pool_pages: Vec<i32>,
    pub slot: i32,
    pub len: usize,
    /// Rows the KDA recurrent and conv state has consumed: `len` after a prefill or a plain
    /// verify; a speculative verify leaves it until the caller commits the kept rows.
    pub kda_len: usize,
}

impl GlmfPlacement {
    /// A fresh sequence over `units` with KDA state slot `slot`.
    pub fn new(units: Vec<u32>, slot: i32) -> Self {
        let pages = units.iter().flat_map(|&u| (0..UNIT_PAGES as i32).map(move |i| u as i32 * UNIT_PAGES as i32 + i))
            .collect();
        let pool_pages = units.iter().map(|&u| u as i32).collect();
        Self { units, pages, pool_pages, slot, len: 0, kda_len: 0 }
    }

    pub fn record(&self, position: usize) -> Result<i64> {
        let page = *self.pages.get(position / PAGE_ROWS).context("position past the sequence's pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + (position % PAGE_ROWS) as i64)
    }

    /// Pool-cache slot of the pool `position` completes, or -1.
    pub fn pool_slot(&self, position: usize) -> Result<i64> {
        if position % KPOOL != KPOOL - 1 {
            return Ok(-1);
        }
        let page = *self.pool_pages.get(position / POOL_PAGE_TOKENS).context("position past the pool pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + ((position / KPOOL) % PAGE_ROWS) as i64)
    }
}

/// Refcounted allocation units and free KDA state slots (goldens and benches; serving takes
/// its units from the prefix cache's pool).
pub(crate) struct Allocator {
    units: cuteafd_engine::prefix::RefPagePool,
    slots: Vec<i32>,
}

impl Allocator {
    /// Over an engine of `pages` MLA pages (a whole number of units) and `slots` KDA slots.
    pub fn new(pages: usize, slots: usize) -> Self {
        Self { units: cuteafd_engine::prefix::RefPagePool::new(pages / UNIT_PAGES, UNIT_ROWS),
            slots: (0..slots as i32).rev().collect() }
    }

    /// Reserves every unit a sequence of up to `capacity` tokens needs and a state slot (the
    /// engine zeroes the slot and maps the pool pages before the first step).
    pub fn admit(&mut self, capacity: usize) -> Result<GlmfPlacement> {
        let slot = self.slots.pop().context("KDA state slots exhausted")?;
        match self.units.alloc(self.units.pages_for(capacity)) {
            Ok(units) => Ok(GlmfPlacement::new(units, slot)),
            Err(error) => {
                self.slots.push(slot);
                Err(error).context("cache pages exhausted")
            }
        }
    }

    /// A second sequence starting as `source`'s first `len` rows: full units shared, the
    /// partial tail unit copied by the caller (the returned copy), and its own KDA slot.
    pub fn fork(&mut self, source: &GlmfPlacement, len: usize, capacity: usize)
        -> Result<(GlmfPlacement, Option<cuteafd_engine::prefix::TailCopy>)> {
        let slot = self.slots.pop().context("KDA state slots exhausted")?;
        match self.units.fork(&source.units, len, self.units.pages_for(capacity)) {
            Ok(fork) => Ok((GlmfPlacement::new(fork.pages, slot), fork.copy)),
            Err(error) => {
                self.slots.push(slot);
                Err(error).context("cache pages exhausted")
            }
        }
    }

    /// A spare KDA state slot (a speculative verify's backup).
    pub fn spare_slot(&mut self) -> Result<i32> {
        self.slots.pop().context("KDA state slots exhausted")
    }

    pub fn release_slot(&mut self, slot: i32) {
        self.slots.push(slot);
    }

    pub fn release(&mut self, placement: GlmfPlacement) {
        self.units.release(&placement.units);
        self.slots.push(placement.slot);
    }
}

struct Workspace<'a> {
    rows: usize,
    streams: [Dev<'a>; 2],
    post: Dev<'a>,
    comb: Dev<'a>,
    x: Dev<'a>,
    delta: Dev<'a>,
    shared: Dev<'a>,
    routed: Dev<'a>,
    query: Dev<'a>,
    q_resid: Dev<'a>,
    latent: Dev<'a>,
    positions: Dev<'a>,
    kv_slots: Dev<'a>,
    kda_slots: Dev<'a>,
    seq_first: Dev<'a>,
    pool_slots: Dev<'a>,
    cache_lengths: Dev<'a>,
    page_table: Dev<'a>,
    pool_table: Dev<'a>,
    q_fp8: Dev<'a>,
    head_weights: Dev<'a>,
    pools: Dev<'a>,
    indices: Dev<'a>,
    lengths: Dev<'a>,
    scratch: Dev<'a>,
    /// The pool top-k's scratch: zeroed once, restored by every launch.
    topk_scratch: Dev<'a>,
    logits: Dev<'a>,
    /// The step's token ids (U32, gathered from the device embedding table).
    ids: Dev<'a>,
    /// Greedy selection of the logits rows inside the decode graph: U32 ids, then U32 statuses.
    select: Dev<'a>,
    router_logits: Dev<'a>,
    route_ids: Dev<'a>,
    route_weights: Dev<'a>,
    wire: Dev<'a>,
    router_host: RefCell<HostAllocation<'a>>,
    head: VocabularyHead<'a>,
    _head_workspace: Dev<'a>,
}

/// ModelOpt NVFP4 dense MLPs (nvidia/GLM-5.3-Flash-NVFP4 layers 0-2) on the
/// `fp8-glmfdense-nvfp4` package: one always-selected expert (ids 0, weight 1),
/// so the route sum is the MLP output exactly.
pub(crate) struct DenseNvfp4<'a> {
    pub module: cuteafd_ffi::fp8_moe::Fp8MoeModule,
    pub scratch: crate::shared::memory::DeviceAllocation<'a>,
    pub ids: crate::shared::memory::DeviceAllocation<'a>,
    pub weights: crate::shared::memory::DeviceAllocation<'a>,
}

impl<'a> DenseNvfp4<'a> {
    /// Loads the package at `directory` with scratch, ids and weights for `rows` rows.
    pub fn load(library: &'a NativeLibrary, directory: &std::path::Path, cfg: &GlmNextConfig, rows: usize)
        -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights/dense-nvfp4");
        // SAFETY: a trusted package for the current device; the engine drains its
        // stream before dropping it.
        let module = unsafe { cuteafd_ffi::fp8_moe::Fp8MoeModule::load(directory) }
            .with_context(|| format!("dense NVFP4 package {} (glmfdense:nvfp4)", directory.display()))?;
        let info = module.info().clone();
        ensure!(info.experts == 1 && info.topk == 1 && info.hidden == cfg.hidden
            && info.intermediate == cfg.dense_intermediate && !info.wire_input
            && matches!(info.weights, cuteafd_ffi::fp8_moe::Fp8MoeWeights::Nvfp4 { .. }),
            "{} ({info:?}) is not the dense NVFP4 MLP package", directory.display());
        let top = info.capacity_for(rows).with_context(|| format!("dense NVFP4 package has no capacity for {rows} rows"))?;
        let scratch = crate::shared::memory::DeviceAllocation::new(library, module.scratch_bytes(top)?.max(256))?;
        let ids = crate::shared::memory::DeviceAllocation::new(library, rows * 4)?;
        library.copy_h2d(ids.buffer, &vec![0u8; rows * 4])?;
        let weights = crate::shared::memory::DeviceAllocation::new(library, rows * 4)?;
        let ones: Vec<u8> = (0..rows).flat_map(|_| 1f32.to_le_bytes()).collect();
        library.copy_h2d(weights.buffer, &ones)?;
        Ok(Self { module, scratch, ids, weights })
    }
}

pub(crate) struct GlmfEngine<'a> {
    quantize_grid: Fp8QuantizeGrid,
    pub library: &'a NativeLibrary,
    pub programs: &'a Programs<'a>,
    pub cfg: GlmNextConfig,
    pub weights: GlmfWeights<'a>,
    pub stream: *mut c_void,
    pub max_context: usize,
    pub prefill_rows: usize,
    pub pages: usize,
    pub slots: usize,
    /// Per MLA layer (None for KDA): the latent record pool.
    kv: Vec<Option<Dev<'a>>>,
    /// Per layer: its index among the KDA layers (None for MLA).
    kda_ordinal: Vec<Option<usize>>,
    /// Every KDA layer's pools back to back: FP32 recurrent state
    /// `[layers, slots, 64, 128, 128]`, BF16 conv state `[layers, slots, 3, 3D]`
    /// and the speculative replay records (`replay_bytes` per layer).
    kda_state: Dev<'a>,
    kda_conv: Dev<'a>,
    kda_replay: Dev<'a>,
    /// `glmf_kda_commit` tables (slot, first row, kept rows per sequence).
    commit_tables: Dev<'a>,
    /// The DFlash2 drafter: every step taps its target layers.
    pub drafter: Option<crate::families::glm5::dflash::GlmDrafter<'a>>,
    /// L2 prefetch of the next layer's weights during decode exchanges.
    pub l2: Option<crate::shared::l2_prefetch::L2Prefetch>,
    /// Per MLA layer (None for KDA): per-token indexer keys | gates (BF16
    /// [record slots, 256]) and the FP8 pool-key cache.
    index: Vec<Option<(Dev<'a>, Dev<'a>)>>,
    /// Logical page of each pool-cache page within its sequence, and its host copy (a shared
    /// pool page sits at the same logical page in every sequence that holds it).
    pool_logical: Dev<'a>,
    pool_logical_host: RefCell<Vec<i32>>,
    /// Pool-cache pages: one per allocation unit (`pages / UNIT_PAGES`).
    pub pool_pages: usize,
    workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    /// Pipelined Spark prefill: one workspace of `prefill_rows` rows per lane.
    lane_workspaces: RefCell<Vec<Workspace<'a>>>,
    experts: Option<Experts<'a>>,
    dense_nvfp4: Option<DenseNvfp4<'a>>,
    /// Host seconds: GPU wait before expert exchanges, the exchanges.
    /// Host seconds per phase: GPU work until each expert exchange (waiting
    /// for the routes and wire rows), the Spark exchanges, and the head (final
    /// norm, vocabulary projection, logits download).
    pub profile: RefCell<[f64; 3]>,
    /// Prefill steps keep every row's logits (golden scoring); otherwise the
    /// prefill workspace holds logits for at most `DECODE_ROWS` rows.
    pub full_prefill_logits: bool,
    /// Captured decode segments (CUTEAFD_GLMF_GRAPHS=0 runs decode eagerly).
    graphs: RefCell<std::collections::HashMap<GraphKey, GraphExec<'a>>>,
    use_graphs: bool,
    /// Spark prefill runs in lanes (CUTEAFD_GLMF_PREFILL_LANES, default on).
    lanes: bool,
    /// CUTEAFD_GLMF_PREFILL_LANES=subset: lanes even with a `--layers`
    /// subset (timing runs against loopback ranks that hold only those layers).
    subset_lanes: bool,
    /// Recorded after a layer's routes and wire rows reach the host staging.
    routes_ready: *mut c_void,
    ops: Option<RefCell<OpTimes>>,
    /// Prefill projections that run block-FP8 GEMMs (the layers need FP8 copies).
    pub fp8_prefill: Fp8Prefill,
    /// The token embedding table (resident on this GPU or read from its shard).
    pub embedding: TokenEmbedding<'a>,
}

/// Which prefill projections run W8A8 block-FP8 GEMMs (E4M3 activations per
/// row and 128-K block): the MLA projections, the dense and shared-expert MLPs
/// (FP8-only weights: W8A16 otherwise), the KDA in-projection and o_proj
/// (per-row x 128-K copies: BF16 otherwise).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Fp8Prefill {
    pub mla: bool,
    pub ffn: bool,
    /// KDA projections that run FP8 (bit 0 the in-projection, bit 1 o_proj).
    pub kda_bits: i32,
}

/// What a captured decode segment baked in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GraphKey {
    segment: usize,
    rows: usize,
    spec: bool,
    long: bool,
    pool_width: usize,
    page_stride: usize,
    pool_stride: usize,
}

struct GraphExec<'a>(*mut c_void, &'a NativeLibrary);

impl Drop for GraphExec<'_> {
    fn drop(&mut self) {
        // SAFETY: the executable graph is owned here and no longer launched.
        let _ = unsafe { self.1.cuda_graph_exec_destroy(self.0) };
    }
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl<'a> GlmfEngine<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, cfg: GlmNextConfig, weights: GlmfWeights<'a>,
        stream: *mut c_void, max_context: usize, prefill_rows: usize, pages: usize, slots: usize,
        embedding: TokenEmbedding<'a>) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("kv");
        let quantize_grid = Fp8QuantizeGrid::new(library.sm_count()?, None)?;
        ensure!(embedding.hidden() == cfg.hidden, "embedding rows of {} for hidden {}", embedding.hidden(), cfg.hidden);
        ensure!(cfg.hc_mult == HC && cfg.kv_lora_rank == 512 && cfg.kda_head_dim == 128 && cfg.heads == 64,
            "the glmf programs are built for 4 mHC streams, a 512 latent, 64 MLA heads and 128-wide KDA heads");
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let d = cfg.kda_heads * cfg.kda_head_dim;
        // Whole allocation units: four MLA pages and one pool page each.
        let pages = pages.max(1).next_multiple_of(UNIT_PAGES);
        let pool_pages = pages / UNIT_PAGES;
        let mut kv = Vec::new();
        let mut kda_ordinal = Vec::new();
        let mut index = Vec::new();
        let mut kda_layers = 0;
        for layer in &weights.layers {
            match layer.attention {
                GlmNextAttention::Mla => {
                    kv.push(Some(zeroed(pages * PAGE_ROWS * RECORD_BYTES)?));
                    kda_ordinal.push(None);
                    index.push(Some((zeroed(pages * PAGE_ROWS * 512)?, zeroed(pool_pages * PAGE_ROWS * 132)?)));
                }
                GlmNextAttention::Kda => {
                    kv.push(None);
                    kda_ordinal.push(Some(kda_layers));
                    kda_layers += 1;
                    index.push(None);
                }
            }
        }
        let kda_state = zeroed(kda_layers * slots * d * cfg.kda_head_dim * 4)?;
        let kda_conv = zeroed(kda_layers * slots * 3 * 3 * d * 2)?;
        let kda_replay = zeroed(kda_layers * replay_bytes(cfg.kda_heads, 3 * d))?;
        let pool_logical = zeroed(pool_pages * 4)?;
        Ok(Self { quantize_grid, library, programs, cfg, weights, stream, max_context, prefill_rows, pages, slots, kv, kda_ordinal,
            kda_state, kda_conv, kda_replay, commit_tables: zeroed(3 * DECODE_ROWS * 4)?, drafter: None, index,
            pool_logical, pool_logical_host: RefCell::new(vec![0; pool_pages]), pool_pages, workspace: RefCell::new(None), decode_workspace: RefCell::new(None),
            lane_workspaces: RefCell::new(Vec::new()),
            experts: None, dense_nvfp4: None, profile: RefCell::new([0.0; 3]), graphs: RefCell::new(std::collections::HashMap::new()),
            use_graphs: std::env::var("CUTEAFD_GLMF_GRAPHS").map_or(true, |v| v != "0"),
            lanes: std::env::var("CUTEAFD_GLMF_PREFILL_LANES").map_or(true, |v| v != "1"),
            subset_lanes: std::env::var("CUTEAFD_GLMF_PREFILL_LANES").is_ok_and(|v| v == "subset"),
            full_prefill_logits: false, routes_ready: library.cuda_event_create_ordering()?,
            ops: std::env::var("CUTEAFD_GLMF_PROFILE_OPS").is_ok_and(|v| v == "1").then(RefCell::default),
            fp8_prefill: Fp8Prefill::default(), l2: None, embedding })
    }

    /// Serves MoE layers from `experts` (without, the engine stops at the first MoE layer).
    /// The one-expert NVFP4 package for ModelOpt NVFP4 dense MLPs.
    pub fn set_dense_nvfp4(&mut self, dense: DenseNvfp4<'a>) {
        self.dense_nvfp4 = Some(dense);
    }

    pub fn set_experts(&mut self, experts: Experts<'a>) {
        self.experts = Some(experts);
    }

    pub fn experts(&self) -> Option<&Experts<'a>> {
        self.experts.as_ref()
    }

    /// Before a sequence's first step: zeroes its KDA state and maps its pool pages.
    fn start(&self, placement: &GlmfPlacement) -> Result<()> {
        self.map_pools(placement)?;
        self.reset_slot(placement.slot)
    }

    /// Records each of `placement`'s pool pages' logical page (the index expansion reads it).
    /// A restored sequence maps its pages before its first step as a fresh one does; shared
    /// pages keep the value they have.
    pub fn map_pools(&self, placement: &GlmfPlacement) -> Result<()> {
        let mut host = self.pool_logical_host.borrow_mut();
        let mut changed = false;
        for (logical, &page) in placement.pool_pages.iter().enumerate() {
            let page = usize::try_from(page)?;
            ensure!(page < self.pool_pages, "pool page {page} out of range");
            changed |= std::mem::replace(&mut host[page], logical as i32) != logical as i32;
        }
        if changed {
            // Entries other sequences' queued steps read keep their values.
            self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: host.len() * 4, ..self.pool_logical.buffer },
                bytes_of(&host[..]))?;
        }
        Ok(())
    }

    /// Copies a sequence's KDA recurrent and conv state from slot `from` to
    /// slot `to` on the engine stream (the backup a speculative verify
    /// restores before replaying its accepted rows).
    pub fn copy_slot(&self, from: i32, to: i32) -> Result<()> {
        let (from, to) = (usize::try_from(from)?, usize::try_from(to)?);
        ensure!(from < self.slots && to < self.slots && from != to, "KDA slots {from} -> {to} out of range");
        for (at_from, at_to) in self.slot_regions(from).into_iter().zip(self.slot_regions(to)) {
            // SAFETY: both slot regions are live and disjoint; the stream orders the copy.
            unsafe { self.library.copy_d2d_async(at_to, at_from, at_from.bytes, self.stream)? };
        }
        Ok(())
    }

    /// Every KDA layer's recurrent and conv state regions of `slot`.
    pub(crate) fn slot_regions(&self, slot: usize) -> Vec<cuteafd_ffi::CuteafdDeviceBuffer> {
        let layers = self.kda_ordinal.iter().flatten().count().max(1);
        let mut out = Vec::new();
        for pool in [&self.kda_state, &self.kda_conv] {
            let per = pool.buffer.bytes / layers / self.slots;
            for layer in 0..layers {
                out.push(cuteafd_ffi::CuteafdDeviceBuffer {
                    // SAFETY: layer < layers and slot < slots: the region lies inside the pool.
                    ptr: unsafe { pool.buffer.ptr.cast::<u8>().add((layer * self.slots + slot) * per) }.cast(),
                    bytes: per,
                    ..pool.buffer
                });
            }
        }
        out
    }

    /// Every MLA layer's paged buffers: latent records (528 B per row), indexer token keys
    /// (512 B per row), both in 64-row MLA pages, and the pool-key cache (8448 B per pool page).
    pub(crate) fn paged_buffers(&self) -> Vec<[cuteafd_ffi::CuteafdDeviceBuffer; 3]> {
        self.kv.iter().zip(&self.index).filter_map(|(kv, index)| match (kv, index) {
            (Some(kv), Some((keys, pools))) => Some([kv.buffer, keys.buffer, pools.buffer]),
            _ => None,
        }).collect()
    }

    /// Zeroes a sequence's KDA recurrent and conv state (before its first step).
    pub fn reset_slot(&self, slot: i32) -> Result<()> {
        let slot = usize::try_from(slot)?;
        ensure!(slot < self.slots, "KDA slot {slot} out of range");
        for region in self.slot_regions(slot) {
            self.library.cuda_zero_bytes(region, region.bytes)?;
        }
        Ok(())
    }

    /// Every KDA layer's recurrent then conv state of `slot` (host copy; checks).
    pub fn slot_state(&self, slot: i32) -> Result<Vec<u8>> {
        let slot = usize::try_from(slot)?;
        ensure!(slot < self.slots, "KDA slot {slot} out of range");
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        let mut out = Vec::new();
        for region in self.slot_regions(slot) {
            let mut bytes = vec![0u8; region.bytes];
            self.library.copy_d2h(&mut bytes, region)?;
            out.extend(bytes);
        }
        Ok(out)
    }

    /// After a speculative verify step (`verify_spec`): applies each
    /// sequence's first `keep` rows (from step row `first`) to the KDA state
    /// of `slot` in every layer, as serial steps over those rows would have.
    /// Callers set the committed placements' `kda_len` to their kept length.
    pub fn commit(&self, sequences: &[(i32, usize, usize)]) -> Result<()> {
        if sequences.is_empty() {
            return Ok(());
        }
        ensure!(sequences.len() <= DECODE_ROWS && sequences.iter().all(|&(slot, first, keep)|
            slot >= 0 && (slot as usize) < self.slots && first + keep <= REPLAY_ROWS), "commit of {sequences:?}");
        let n = sequences.len();
        let mut tables = vec![0i32; 3 * n];
        for (i, &(slot, first, keep)) in sequences.iter().enumerate() {
            tables[i] = slot;
            tables[n + i] = first as i32;
            tables[2 * n + i] = keep as i32;
        }
        self.put(&self.commit_tables, &tables)?;
        let layers = self.kda_ordinal.iter().flatten().count();
        self.run("kda_commit", &[("state", self.kda_state.buffer.ptr), ("conv_state", self.kda_conv.buffer.ptr),
            ("replay", self.kda_replay.buffer.ptr), ("tables", self.commit_tables.buffer.ptr)],
            &[Scalar::I32(n as i32), Scalar::I32(layers as i32), Scalar::I32(self.slots as i32)])
    }

    fn alloc(&self, bytes: usize) -> Result<Dev<'a>> {
        DeviceAllocation::new(self.library, bytes.max(256))
    }

    fn run(&self, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar]) -> Result<()> {
        let name = format!("glmf_{name}");
        let names: Vec<&str> = pointers.iter().map(|(n, _)| *n).collect();
        let program = self.programs.program(&name, &names)?;
        let raw: Vec<*mut c_void> = pointers.iter().map(|(_, p)| *p).collect();
        self.timed(&name, || {
            // SAFETY: every pointer names a live allocation sized for the rows in
            // `scalars`; the stream orders all launches of this engine.
            unsafe { program.launch(&raw, scalars, self.stream) }.with_context(|| format!("{name} with {scalars:?}"))
        })
    }

    /// Runs `body` (stream work) between two timing events when profiling ops.
    fn timed<T>(&self, label: &str, body: impl FnOnce() -> Result<T>) -> Result<T> {
        let Some(ops) = &self.ops else { return body() };
        let event = |ops: &mut OpTimes| -> Result<*mut c_void> {
            match ops.pool.pop() {
                Some(event) => Ok(event),
                None => self.library.cuda_event_create(),
            }
        };
        let (start, end) = {
            let mut ops = ops.borrow_mut();
            (event(&mut ops)?, event(&mut ops)?)
        };
        // SAFETY: both events are live timing events; the engine owns the stream.
        unsafe { self.library.cuda_event_record(start, self.stream)? };
        let out = body()?;
        // SAFETY: as above.
        unsafe { self.library.cuda_event_record(end, self.stream)? };
        ops.borrow_mut().pending.push((label.to_owned(), start, end));
        Ok(out)
    }

    /// Drains the stream and returns the per-label GPU milliseconds and
    /// launches accumulated since the last call (empty unless profiling ops).
    pub fn op_profile(&self) -> Result<std::collections::BTreeMap<String, (f64, usize)>> {
        let Some(ops) = &self.ops else { return Ok(Default::default()) };
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        self.flush_ops()?;
        Ok(std::mem::take(&mut ops.borrow_mut().totals))
    }

    /// Adds host wall time since `since` under `label` when profiling ops.
    fn host_op(&self, label: &str, since: std::time::Instant) {
        if let Some(ops) = &self.ops {
            let mut ops = ops.borrow_mut();
            let entry = ops.totals.entry(label.to_owned()).or_default();
            entry.0 += since.elapsed().as_secs_f64() * 1e3;
            entry.1 += 1;
        }
    }

    fn flush_ops(&self) -> Result<()> {
        let Some(ops) = &self.ops else { return Ok(()) };
        let mut ops = ops.borrow_mut();
        let pending = std::mem::take(&mut ops.pending);
        for (label, start, end) in pending {
            // SAFETY: both events were recorded on the drained stream.
            let ms = unsafe { self.library.cuda_event_elapsed_ms(start, end)? };
            let entry = ops.totals.entry(label).or_default();
            entry.0 += f64::from(ms);
            entry.1 += 1;
            ops.pool.extend([start, end]);
        }
        Ok(())
    }

    fn scratch(&self, name: &str) -> Result<usize> {
        Ok(self.programs.spec(&format!("glmf_{name}"))?.scratch.get("scratch").copied().unwrap_or(0) as usize)
    }

    fn workspace(&self, t: usize, decode: bool) -> Result<Workspace<'a>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace");
        let (h, n, lat) = (self.cfg.hidden, self.cfg.heads, self.cfg.kv_lora_rank);
        let (cap, mode) = if decode { ("m64", "decode") } else { ("m4096", "prefill") };
        let mut scratch = 0;
        for name in [format!("mhc_post_pre_{cap}"), format!("kda_{cap}"), format!("mla_producer_{cap}"),
            format!("sparse_mla_{mode}_{cap}"), format!("o_{cap}"), format!("ffn_i2048_{cap}"),
            format!("ffn_i12288_{cap}"), format!("index_producer_{cap}"), "mhc_pre".into()] {
            scratch = scratch.max(self.scratch(&name)?);
        }
        if self.weights.layers.iter().any(|layer| layer.has("w_in_fp8")) {
            scratch = scratch.max(self.scratch(&format!("kda_w8_{cap}"))?);
        }
        let topk_scratch = self.scratch(&format!("index_topk_{mode}_{cap}"))?;
        let pools = self.cfg.index_topk / KPOOL;
        let table_rows = if decode { t } else { 1 };
        let head_workspace = self.alloc(VOCABULARY_HEAD_WORKSPACE)?;
        let spark = matches!(self.experts, Some(Experts::Spark { .. }));
        let topk = self.cfg.topk;
        Ok(Workspace {
            rows: t,
            streams: [self.alloc(t * HC * h * 2)?, self.alloc(t * HC * h * 2)?],
            post: self.alloc(t * HC * 4)?,
            comb: self.alloc(t * HC * HC * 4)?,
            x: self.alloc(t * h * 2)?,
            delta: self.alloc(t * h * 2)?,
            shared: self.alloc(t * h * 2)?,
            routed: self.alloc(t * h * 2)?,
            query: self.alloc(t * n * lat * 2)?,
            q_resid: self.alloc(t * self.cfg.q_lora_rank * 2)?,
            latent: self.alloc(t * n * lat * 2)?,
            positions: self.alloc(t * 8)?,
            kv_slots: self.alloc(t * 8)?,
            kda_slots: self.alloc(t * 4)?,
            seq_first: self.alloc(t * 4)?,
            pool_slots: self.alloc(t * 8)?,
            cache_lengths: self.alloc(t * 4)?,
            page_table: self.alloc(table_rows * self.pages * 4)?,
            pool_table: self.alloc(table_rows * self.pool_pages * 4)?,
            q_fp8: self.alloc(t * 32 * 128)?,
            head_weights: self.alloc(t * 32 * 4)?,
            pools: self.alloc(t * pools * 4)?,
            indices: self.alloc(t * SPARSE_TOPK * 4)?,
            lengths: self.alloc(t * 4)?,
            scratch: self.alloc(scratch)?,
            topk_scratch: {
                let zero = self.alloc(topk_scratch)?;
                self.library.cuda_zero_bytes(zero.buffer, zero.buffer.bytes)?;
                zero
            },
            logits: self.alloc(if decode || self.full_prefill_logits { t } else { t.min(DECODE_ROWS) }
                * self.cfg.vocab_size * 4)?,
            ids: self.alloc(t * 4)?,
            select: self.alloc(t * 8)?,
            router_logits: self.alloc(t * self.cfg.experts * 4)?,
            route_ids: self.alloc(t * topk * 4)?,
            route_weights: self.alloc(t * topk * 4)?,
            wire: self.alloc(t * (h + h / 32))?,
            router_host: RefCell::new(HostAllocation::new(self.library,
                if spark { t * (topk * 8 + h + h / 32) } else { 256 })?),
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, h as u32, t as u32,
                self.cfg.vocab_size as u32)? },
            _head_workspace: head_workspace,
        })
    }

    /// Streams start as four copies of each token's embedding. With the
    /// device table the ids go up and the rows are gathered into every stream
    /// slot (unless `defer_gather`: the decode graph's first segment gathers);
    /// otherwise the shard's rows go up once (into the second stream buffer)
    /// and are copied into each stream slot on the device.
    fn load_streams(&self, w: &Workspace<'_>, tokens: &[u32], defer_gather: bool) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.len() <= w.rows, "{} tokens exceed the workspace", tokens.len());
        if self.embedding.placement() == EmbedPlacement::Gpu {
            let host = std::time::Instant::now();
            self.embedding.check(tokens)?;
            self.put(&w.ids, tokens)?;
            self.host_op("host: token ids upload", host);
            if !defer_gather {
                self.timed("embedding gather", || self.gather_streams(w, tokens.len()))?;
            }
            return Ok(());
        }
        let embed = self.embedding.host_rows(tokens)?;
        let row = self.cfg.hidden * 2;
        let t = tokens.len();
        let host = std::time::Instant::now();
        let staged = cuteafd_ffi::CuteafdDeviceBuffer { bytes: embed.len(), ..w.streams[1].buffer };
        self.library.copy_h2d(staged, &embed)?;
        self.host_op("host: embedding upload", host);
        self.timed("stream expansion", || {
            for s in 0..HC {
                let dst = cuteafd_ffi::CuteafdDeviceBuffer {
                    // SAFETY: slot `s` of row 0 lies inside the [t, 4, H] stream buffer.
                    ptr: unsafe { w.streams[0].buffer.ptr.cast::<u8>().add(s * row) }.cast(),
                    bytes: w.streams[0].buffer.bytes - s * row,
                    ..w.streams[0].buffer
                };
                // SAFETY: both buffers are live workspace buffers of at least `t` pitched
                // rows; the upload above completed before the copies are queued.
                unsafe { self.library.copy_device_rows_async(dst, staged, row, t, HC * row, row, self.stream)? };
            }
            Ok(())
        })
    }

    /// The device table's rows of the `t` staged ids, four copies each, into stream buffer 0.
    fn gather_streams(&self, w: &Workspace<'_>, t: usize) -> Result<()> {
        // SAFETY: the ids are on the device (a completed copy) and the streams hold t x 4 rows.
        unsafe { self.embedding.gather(w.ids.buffer.ptr, std::ptr::null(), t, HC, std::ptr::null(),
            w.streams[0].buffer.ptr, self.stream) }
    }

    /// Greedy tokens of the first `rows` logits rows into `select`.
    fn select_greedy(&self, w: &Workspace<'_>, rows: usize) -> Result<()> {
        let vocab = self.cfg.vocab_size;
        // SAFETY: the logits rows and the select buffer (ids, then statuses) are live buffers of these shapes.
        unsafe {
            self.library.cuda_logits_greedy_f32_async(w.logits.buffer.ptr, rows, vocab, vocab, w.select.buffer.ptr,
                std::ptr::null_mut(), w.select.buffer.ptr.cast::<u8>().add(rows * 4).cast(), self.stream)
        }
    }

    /// The first `rows` logits rows as device logits (`greedy`: with the rows'
    /// selection from [`Self::select_greedy`]).
    fn device_logits(&self, w: &Workspace<'_>, rows: usize, greedy: bool) -> DeviceLogits {
        let vocab = self.cfg.vocab_size;
        DeviceLogits { ptr: w.logits.buffer.ptr, rows, vocab, stride: vocab, stream: self.stream,
            // SAFETY: the statuses follow the ids inside the rows x 8-byte select buffer.
            greedy: greedy.then(|| (w.select.buffer.ptr.cast_const(),
                unsafe { w.select.buffer.ptr.cast::<u8>().add(rows * 4) }.cast_const().cast())) }
    }

    fn put<T: Copy>(&self, dev: &Dev<'_>, values: &[T]) -> Result<()> {
        let bytes = bytes_of(values);
        ensure!(bytes.len() <= dev.buffer.bytes, "table exceeds its buffer");
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: bytes.len(), ..dev.buffer }, bytes)
    }

    fn download(&self, dev: &Dev<'_>, bytes: usize) -> Result<Vec<u8>> {
        ensure!(bytes <= dev.buffer.bytes, "download of {bytes} bytes exceeds {}-byte buffer", dev.buffer.bytes);
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        let mut out = vec![0u8; bytes];
        self.library.copy_d2h(&mut out, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..dev.buffer })?;
        Ok(out)
    }

    /// Per-row positions, record and pool slots, and the pools each row sees.
    fn rows(&self, placement: &GlmfPlacement, positions: std::ops::Range<usize>, first: i32, tables: &mut StepTables)
        -> Result<()> {
        for position in positions {
            ensure!(position < self.max_context, "position {position} past the context {}", self.max_context);
            tables.positions.push(position as i64);
            tables.kv_slots.push(placement.record(position)?);
            tables.pool_slots.push(placement.pool_slot(position)?);
            tables.kda_slots.push(placement.slot);
            tables.seq_first.push(first);
            tables.cache_lengths.push(((position + 1) / KPOOL) as i32);
            tables.long |= position + 1 > self.cfg.dense_context();
            tables.pool_width = tables.pool_width.max((position + 1).div_ceil(POOL_PAGE_TOKENS));
        }
        Ok(())
    }

    /// Prefills a sequence from its length through every resident layer and
    /// returns the last row's logits when all layers are resident.
    /// `on_layer` receives each layer's output streams (BF16 [t, 4, hidden]).
    /// With `forced`, `forced(l)` (when it returns rows) replaces the streams
    /// after layer `l`, so each layer's comparison measures that layer alone.
    /// With `all_logits`, returns every row's logits instead of the last.
    pub fn prefill_forced(&self, placement: &mut GlmfPlacement, tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>, all_logits: bool) -> Result<Option<Vec<f32>>> {
        self.prefill_step(placement, tokens, on_layer, forced, all_logits, false)?
            .map(|logits| logits.into_host(self.library)).transpose()
    }

    /// [`Self::prefill`] leaving the last row's logits on the device.
    pub fn prefill_device(&self, placement: &mut GlmfPlacement, tokens: &[u32]) -> Result<Option<DeviceLogits>> {
        self.prefill_step(placement, tokens, None, None, false, true)?.map(StepLogits::device).transpose()
    }

    fn prefill_step(&self, placement: &mut GlmfPlacement, tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>, all_logits: bool, device: bool)
        -> Result<Option<StepLogits>> {
        let (t, start) = (tokens.len(), placement.len);
        if on_layer.is_none() && forced.is_none() && self.pipelined() {
            return self.prefill_lanes(placement, tokens, all_logits, device);
        }
        ensure!(t > 0 && t <= self.prefill_rows && start + t <= self.max_context, "prefill of {t} rows at {start}");
        if start == 0 {
            self.start(placement)?;
        }
        let mut tables = StepTables { page_table: placement.pages.clone(), pool_table: placement.pool_pages.clone(),
            ..Default::default() };
        self.rows(placement, start..start + t, 0, &mut tables)?;
        let logits = self.step(&tables, tokens, if all_logits { t } else { 1 }, on_layer, forced, None)?;
        placement.len += t;
        placement.kda_len = placement.len;
        Ok(logits.map(StepLogits::Device))
    }

    /// Whether prefill runs as Spark lanes (a transport per lane, every layer resident).
    /// CUTEAFD_GLMF_PREFILL_LANES=1 keeps the serial one-workspace prefill (A/B runs).
    fn pipelined(&self) -> bool {
        self.lanes && (self.weights.layers.len() == self.cfg.layers || self.subset_lanes) && matches!(&self.experts,
            Some(Experts::Spark { transports, .. }) if transports.borrow().len() >= PREFILL_LANES)
    }

    /// Longest chunk one prefill call takes: a lane of `prefill_rows` rows
    /// each when Spark prefill is pipelined.
    pub fn prefill_capacity(&self) -> usize {
        if self.pipelined() { prefill_lane_capacity(self.prefill_rows) } else { self.prefill_rows }
    }

    /// A Spark prefill chunk as up to [`PREFILL_LANES`] lanes of consecutive
    /// rows (see [`Self::step_lanes`]).
    fn prefill_lanes(&self, placement: &mut GlmfPlacement, tokens: &[u32], all_logits: bool, device: bool)
        -> Result<Option<StepLogits>> {
        let (start, t) = (placement.len, tokens.len());
        let (_, per_lane) = prefill_lane_plan(t, self.prefill_rows)?;
        // Lanes split at a multiple of 64 rows (an MLA page), so each lane's
        // pools and pages start where the previous lane's end.
        ensure!(t > 0 && per_lane <= self.prefill_rows && start + t <= self.max_context,
            "prefill of {t} rows at {start} exceeds {} rows per lane or the context", self.prefill_rows);
        if start == 0 {
            self.start(placement)?;
        }
        let mut steps = Vec::new();
        let mut first = 0;
        while first < t {
            let n = per_lane.min(t - first);
            let mut tables = StepTables { page_table: placement.pages.clone(), pool_table: placement.pool_pages.clone(),
                ..Default::default() };
            self.rows(placement, start + first..start + first + n, 0, &mut tables)?;
            steps.push((tables, &tokens[first..first + n]));
            first += n;
        }
        let logits = self.step_lanes(&steps, if all_logits { t } else { 1 }, device)?;
        placement.len += t;
        placement.kda_len = placement.len;
        Ok(logits)
    }

    pub fn prefill(&self, placement: &mut GlmfPlacement, tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        self.prefill_forced(placement, tokens, on_layer, None, false)
    }

    /// Appends each sequence's tokens (one for decode, several for a verify)
    /// at its length in one decode-shaped step; returns every row's logits.
    /// KDA state advances in place: a caller rejecting a suffix must replay
    /// (or verify with [`Self::verify_spec`] and commit what it keeps).
    pub fn verify(&self, sequences: &mut [(&mut GlmfPlacement, usize)], tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        self.decode_step(sequences, tokens, on_layer, false, None)?.map(|l| l.to_host(self.library)).transpose()
    }

    pub fn verify_trace(&self, sequences: &mut [(&mut GlmfPlacement, usize)], tokens: &[u32],
        on_layer: &mut dyn FnMut(usize, &[u8]) -> Result<()>, dir: &std::path::Path) -> Result<Option<Vec<f32>>> {
        self.decode_step(sequences, tokens, Some(on_layer), false, Some(dir))?
            .map(|l| l.to_host(self.library)).transpose()
    }

    /// Diagnostic snapshot inside a verify's layer callback. The callback
    /// already disables graphs and synchronizes the layer's output download.
    /// Router buffers still belong to this layer, even though mHC has prepared
    /// the next attention input. Ordinary serving never calls this method.
    pub fn trace_decode_layer(&self, layer: usize, rows: usize, streams: &[u8], dir: &std::path::Path)
        -> Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("streams.bin"), streams)?;
        let workspace = self.decode_workspace.borrow();
        let w = workspace.as_ref().context("decode trace without a workspace")?;
        for (name, buffer, bytes) in [("ffn.bin", &w.delta, rows * self.cfg.hidden * 2),
            ("next_input.bin", &w.x, rows * self.cfg.hidden * 2)] {
            std::fs::write(dir.join(name), self.download(buffer, bytes)?)?;
        }
        if !self.weights.layers[layer].dense {
            for (name, buffer, bytes) in [("route_ids.bin", &w.route_ids, rows * self.cfg.topk * 4),
                ("route_weights.bin", &w.route_weights, rows * self.cfg.topk * 4),
                ("router_logits.bin", &w.router_logits, rows * self.cfg.experts * 4),
                ("wire.bin", &w.wire, rows * (self.cfg.hidden + self.cfg.hidden / 32)),
                ("shared.bin", &w.shared, rows * self.cfg.hidden * 2)] {
                std::fs::write(dir.join(name), self.download(buffer, bytes)?)?;
            }
        }
        Ok(())
    }

    /// A speculative verify: as [`Self::verify`], but the KDA state stays at
    /// every sequence's start and each row's replay inputs are recorded; the
    /// caller then passes every sequence's kept rows to [`Self::commit`]
    /// (MLA records past a sequence's kept length are rewritten by later
    /// steps). Placements advance by all rows; callers set the kept length.
    pub fn verify_spec(&self, sequences: &mut [(&mut GlmfPlacement, usize)], tokens: &[u32])
        -> Result<Option<Vec<f32>>> {
        self.decode_step(sequences, tokens, None, true, None)?.map(|l| l.to_host(self.library)).transpose()
    }

    /// [`Self::verify`] (`spec`: [`Self::verify_spec`]) leaving every row's
    /// logits on the device, with the decode graph's greedy selection of them.
    pub fn verify_device(&self, sequences: &mut [(&mut GlmfPlacement, usize)], tokens: &[u32], spec: bool)
        -> Result<Option<DeviceLogits>> {
        self.decode_step(sequences, tokens, None, spec, None)
    }

    fn decode_step(&self, sequences: &mut [(&mut GlmfPlacement, usize)], tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>, spec: bool,
        trace: Option<&std::path::Path>) -> Result<Option<DeviceLogits>> {
        let rows: usize = sequences.iter().map(|(_, n)| n).sum();
        ensure!(rows > 0 && rows <= DECODE_ROWS && tokens.len() == rows, "decode step of {rows} rows");
        // Power-of-two strides and widths bound the graphs a growing batch captures.
        let page_stride = sequences.iter().map(|(p, _)| p.pages.len()).max().unwrap_or(1).next_power_of_two()
            .min(self.pages);
        let pool_stride = sequences.iter().map(|(p, _)| p.pool_pages.len()).max().unwrap_or(1).next_power_of_two()
            .min(self.pool_pages);
        let mut tables = StepTables { decode: true, page_stride, pool_stride, spec, ..Default::default() };
        for (placement, count) in sequences.iter() {
            if placement.len == 0 {
                self.start(placement)?;
            }
            let first = tables.kv_slots.len() as i32;
            self.rows(placement, placement.len..placement.len + count, first, &mut tables)?;
            for _ in 0..*count {
                let mut pages = placement.pages.clone();
                pages.resize(page_stride, 0);
                tables.page_table.extend(pages);
                let mut pools = placement.pool_pages.clone();
                pools.resize(pool_stride, 0);
                tables.pool_table.extend(pools);
            }
        }
        tables.pool_width = tables.pool_width.next_power_of_two().min(pool_stride);
        let logits = self.step(&tables, tokens, rows, on_layer, None, trace)?;
        for (placement, count) in sequences.iter_mut() {
            placement.len += *count;
            if !spec {
                placement.kda_len = placement.len;
            }
        }
        Ok(logits)
    }

    fn step(&self, tables: &StepTables, tokens: &[u32], logit_rows: usize,
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>, trace: Option<&std::path::Path>) -> Result<Option<DeviceLogits>> {
        let (h, t) = (self.cfg.hidden, tables.kv_slots.len());
        let (slot, capacity) = if tables.decode { (&self.decode_workspace, DECODE_ROWS) } else { (&self.workspace, self.prefill_rows) };
        if slot.borrow().is_none() {
            *slot.borrow_mut() = Some(self.workspace(capacity, tables.decode)?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        ensure!(t <= w.rows && logit_rows <= t, "step exceeds the workspace");
        ensure!(tables.decode || self.full_prefill_logits || logit_rows <= DECODE_ROWS,
            "prefill logits past {DECODE_ROWS} rows need full_prefill_logits");
        self.put(&w.positions, &tables.positions)?;
        self.put(&w.kv_slots, &tables.kv_slots)?;
        self.put(&w.kda_slots, &tables.kda_slots)?;
        self.put(&w.seq_first, &tables.seq_first)?;
        self.put(&w.pool_slots, &tables.pool_slots)?;
        self.put(&w.cache_lengths, &tables.cache_lengths)?;
        self.put(&w.page_table, &tables.page_table)?;
        self.put(&w.pool_table, &tables.pool_table)?;
        let row = h * 2;
        ensure!(tokens.len() == t, "{} tokens for a {t}-row step", tokens.len());
        let graphed = self.use_graphs && tables.decode && on_layer.is_none() && forced.is_none();
        self.load_streams(w, tokens, graphed)?;
        let rows = Scalar::I32(t as i32);
        if graphed {
            return self.decode_graphed(w, tables, t, rows, logit_rows);
        }
        let cap = if tables.decode { "m64" } else { "m4096" };
        let layers = &self.weights.layers;
        let mut cur = 0usize;
        self.pre(w, &w.streams[cur], &layers[0], rows)?;
        for (index, layer) in layers.iter().enumerate() {
            if let Some(dir) = trace {
                let dir = dir.join(format!("layer{index:02}"));
                std::fs::create_dir_all(&dir)?;
                for (name, buffer, bytes) in [("attention_input.bin", &w.x, t * h * 2),
                    ("attention_post.bin", &w.post, t * HC * 4), ("attention_comb.bin", &w.comb, t * HC * HC * 4)] {
                    std::fs::write(dir.join(name), self.download(buffer, bytes)?)?;
                }
            }
            match layer.attention {
                GlmNextAttention::Kda => self.kda(w, index, layer, rows, cap, tables.spec)?,
                GlmNextAttention::Mla => self.mla(w, index, layer, rows, cap, tables,
                    trace.map(|dir| dir.join(format!("layer{index:02}"))).as_deref())?,
            }
            if let Some(dir) = trace {
                let dir = dir.join(format!("layer{index:02}"));
                std::fs::write(dir.join("attention.bin"), self.download(&w.delta, t * h * 2)?)?;
                if layer.attention == GlmNextAttention::Kda {
                    let d = self.cfg.kda_heads * self.cfg.kda_head_dim;
                    // The in-projection's output width (q|k|v, f_a, g_a, b), whatever its weight format.
                    let p = 3 * d + 2 * self.cfg.kda_head_dim + self.cfg.kda_heads;
                    // Decode KDA AOT layout: BF16 in-projection, f|gate,
                    // convolved q|k|v, recurrent output and gated-norm output;
                    // each region is 1024-byte aligned in the pinned manifest.
                    let bytes: usize = [t * p * 2, t * 4 * d, t * 6 * d, t * 2 * d, t * 2 * d]
                        .iter().map(|n| n.next_multiple_of(1024)).sum();
                    std::fs::write(dir.join("kda_scratch.bin"), self.download(&w.scratch, bytes)?)?;
                    std::fs::write(dir.join("kda_meta.json"), serde_json::to_vec(&serde_json::json!({
                        "rows": t, "width": d, "in_width": p, "alignment": 1024 }))?)?;
                }
            }
            // Attention back into the streams, then the FFN site's collapse + norm.
            self.post_pre(w, cur, layer, "ffn", "post_norm", rows, cap)?;
            cur ^= 1;
            if let Some(dir) = trace {
                std::fs::write(dir.join(format!("layer{index:02}/ffn_input.bin")), self.download(&w.x, t * h * 2)?)?;
            }
            if layer.dense {
                self.ffn(w, layer, self.cfg.dense_intermediate, cap, w.delta.buffer.ptr, rows)?;
            } else {
                self.moe(w, index, layer, t, rows, cap, tables.decode)?;
            }
            match layers.get(index + 1) {
                Some(next) => {
                    self.post_pre(w, cur, next, "attn", "input_norm", rows, cap)?;
                    cur ^= 1;
                }
                None => {
                    self.run("mhc_post", &[("x", w.delta.buffer.ptr), ("residual", w.streams[cur].buffer.ptr),
                        ("prev_post", w.post.buffer.ptr), ("prev_comb", w.comb.buffer.ptr),
                        ("out", w.streams[cur ^ 1].buffer.ptr)], &[rows])?;
                    cur ^= 1;
                }
            }
            if let Some(drafter) = &self.drafter {
                let n = t.min(crate::families::glm5::dflash::TAP_ROWS);
                drafter.tap_streams(index, w.streams[cur].buffer.ptr, HC, t - n, n)?;
            }
            if let Some(on_layer) = on_layer.as_mut() {
                on_layer(index, &self.download(&w.streams[cur], t * HC * row)?)?;
            }
            if let Some(rows_forced) = forced.and_then(|f| f(index)) {
                ensure!(rows_forced.len() == t * HC * row, "teacher-forced streams of the wrong size");
                self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: rows_forced.len(),
                    ..w.streams[cur].buffer }, &rows_forced)?;
                if let Some(next) = layers.get(index + 1) {
                    self.pre(w, &w.streams[cur], next, rows)?;
                }
            }
            crate::shared::console::layer_mark(index);
        }
        if layers.len() < self.cfg.layers {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        let timer = std::time::Instant::now();
        self.run("head", &[("streams", w.streams[cur].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
            ("out", w.x.buffer.ptr)], &[rows])?;
        self.logits(w, t, logit_rows, tables.decode)?;
        self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
        Ok(Some(self.device_logits(w, logit_rows, false)))
    }

    /// A decode step as captured segments: segment `i` posts layer `i - 1`'s
    /// FFN output into the streams with layer `i`'s attention-site collapse,
    /// then runs layer `i` up to its routed experts, which run (local or on
    /// the Sparks) between segments. Streams start and end in buffer 0.
    fn decode_graphed(&self, w: &Workspace<'_>, tables: &StepTables, t: usize, rows: Scalar, logit_rows: usize)
        -> Result<Option<DeviceLogits>> {
        let layers = &self.weights.layers;
        // Every layer resident: the last segment ends in the head and the greedy selection.
        let head = layers.len() == self.cfg.layers && logit_rows == t;
        let gather = self.embedding.placement() == EmbedPlacement::Gpu;
        for index in 0..=layers.len() {
            let key = GraphKey { segment: index, rows: t, spec: tables.spec, long: tables.long,
                pool_width: tables.pool_width, page_stride: tables.page_stride, pool_stride: tables.pool_stride };
            self.replay(key, || -> Result<()> {
                // Layer `index - 1`'s output streams land in buffer 0 first thing.
                let tap = || match (&self.drafter, index.checked_sub(1)) {
                    (Some(drafter), Some(previous)) => drafter.tap_streams(previous, w.streams[0].buffer.ptr, HC, 0, t),
                    _ => Ok(()),
                };
                let Some(layer) = layers.get(index) else {
                    self.run("mhc_post", &[("x", w.delta.buffer.ptr), ("residual", w.streams[1].buffer.ptr),
                        ("prev_post", w.post.buffer.ptr), ("prev_comb", w.comb.buffer.ptr),
                        ("out", w.streams[0].buffer.ptr)], &[rows])?;
                    tap()?;
                    if head {
                        self.run("head", &[("streams", w.streams[0].buffer.ptr),
                            ("weight", self.weights.norm.buffer.ptr), ("out", w.x.buffer.ptr)], &[rows])?;
                        self.logits(w, t, t, true)?;
                        self.select_greedy(w, t)?;
                    }
                    return Ok(());
                };
                if index == 0 {
                    if gather {
                        self.gather_streams(w, t)?;
                    }
                    self.pre(w, &w.streams[0], layer, rows)?;
                } else {
                    self.post_pre(w, 1, layer, "attn", "input_norm", rows, "m64")?;
                    tap()?;
                }
                match layer.attention {
                    GlmNextAttention::Kda => self.kda(w, index, layer, rows, "m64", tables.spec)?,
                    GlmNextAttention::Mla => self.mla(w, index, layer, rows, "m64", tables, None)?,
                }
                self.post_pre(w, 0, layer, "ffn", "post_norm", rows, "m64")?;
                if layer.dense {
                    self.ffn(w, layer, self.cfg.dense_intermediate, "m64", w.delta.buffer.ptr, rows)
                } else {
                    self.moe_front(w, index, layer, t, rows, "m64")
                }
            })?;
            if layers.get(index).is_some_and(|layer| !layer.dense) {
                self.moe_experts(w, index, &layers[index], t, rows, "m64", true)?;
            }
            if index < layers.len() {
                crate::shared::console::layer_mark(index);
            }
        }
        if layers.len() < self.cfg.layers {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        if !head {
            let timer = std::time::Instant::now();
            self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
                ("out", w.x.buffer.ptr)], &[rows])?;
            self.logits(w, t, logit_rows, tables.decode)?;
            self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
        }
        Ok(Some(self.device_logits(w, logit_rows, head)))
    }

    /// The vocabulary projection of the last `logit_rows` normalized rows into
    /// `w.logits` through the one resident head: BF16, or the FP8 head
    /// (--fp8-head) in 16-row spans for every row count.
    fn logits(&self, w: &Workspace<'_>, t: usize, logit_rows: usize, _decode: bool) -> Result<()> {
        let h = self.cfg.hidden;
        // SAFETY: rows t - logit_rows.. of the normalized rows lie inside `w.x`.
        let x = unsafe { w.x.buffer.ptr.cast::<u8>().add((t - logit_rows) * h * 2) }.cast::<c_void>();
        match &self.weights.head {
            GlmfHead::Fp8 { .. } => self.timed("glmf_head_fp8", || {
                // SAFETY: `x` holds `logit_rows` normalized rows and `w.logits` their
                // FP32 logits (the workspace's logit capacity); the engine stream orders them.
                unsafe { self.weights.head.launch_fp8(self.programs, x, w.logits.buffer.ptr.cast(), logit_rows, h,
                    self.cfg.vocab_size, self.stream) }
            }),
            // SAFETY: the head's input and operands are live buffers of these shapes.
            GlmfHead::Bf16(head) => unsafe {
                w.head.launch(x.cast(), head.buffer.ptr.cast(), w.logits.buffer.ptr.cast(), logit_rows as u32,
                    self.stream)
            },
        }
    }

    /// The target head as the DFlash drafter borrows it (the same resident copy).
    pub fn draft_head(&self) -> TargetHead<'_> {
        match &self.weights.head {
            GlmfHead::Bf16(head) => TargetHead::Bf16(head),
            fp8 @ GlmfHead::Fp8 { .. } => TargetHead::Launch(Box::new(move |x, logits, rows, stream| {
                // SAFETY: the drafter passes its live normalized rows and logits
                // workspace for `rows` rows, on the stream it launches the head on.
                unsafe { fp8.launch_fp8(self.programs, x, logits, rows, self.cfg.hidden, self.cfg.vocab_size, stream) }
            })),
        }
    }

    /// Launches `segment` through a graph captured the first time `key` is seen.
    fn replay(&self, key: GraphKey, segment: impl FnOnce() -> Result<()>) -> Result<()> {
        if let Some(graph) = self.graphs.borrow().get(&key) {
            // SAFETY: the graph's pointers are persistent engine buffers.
            return unsafe { self.library.cuda_graph_launch(graph.0, self.stream) };
        }
        // SAFETY: capture records this stream's launches; nothing in a segment
        // synchronizes the host or allocates.
        unsafe { self.library.cuda_graph_begin_capture(self.stream)? };
        let captured = segment();
        // SAFETY: ends the capture begun above on the same stream.
        let exec = unsafe { self.library.cuda_graph_end_capture(self.stream) };
        captured?;
        let exec = exec?;
        // SAFETY: the new graph reads and writes persistent engine buffers.
        unsafe { self.library.cuda_graph_launch(exec, self.stream)? };
        self.graphs.borrow_mut().insert(key, GraphExec(exec, self.library));
        Ok(())
    }

    /// Attention-site collapse and input norm of `layer` from `streams`.
    fn pre(&self, w: &Workspace<'_>, streams: &Dev<'_>, layer: &GlmfLayer<'_>, rows: Scalar) -> Result<()> {
        self.run("mhc_pre", &[("residual", streams.buffer.ptr), ("fn", layer.ptr("attn.fn")?),
            ("scale", layer.ptr("attn.scale")?), ("base", layer.ptr("attn.base")?), ("norm", layer.ptr("input_norm")?),
            ("post", w.post.buffer.ptr), ("comb", w.comb.buffer.ptr), ("y", w.x.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])
    }

    /// `delta` back into streams `cur` (into the other buffer), then the
    /// `site` collapse of `layer` normalized by its `norm`.
    #[allow(clippy::too_many_arguments)]
    fn post_pre(&self, w: &Workspace<'_>, cur: usize, layer: &GlmfLayer<'_>, site: &str, norm: &str,
        rows: Scalar, cap: &str) -> Result<()> {
        self.run(&format!("mhc_post_pre_{cap}"), &[("x", w.delta.buffer.ptr),
            ("residual", w.streams[cur].buffer.ptr), ("prev_post", w.post.buffer.ptr),
            ("prev_comb", w.comb.buffer.ptr), ("fn", layer.ptr(&format!("{site}.fn"))?),
            ("scale", layer.ptr(&format!("{site}.scale"))?), ("base", layer.ptr(&format!("{site}.base"))?),
            ("norm", layer.ptr(norm)?), ("residual_out", w.streams[cur ^ 1].buffer.ptr),
            ("post", w.post.buffer.ptr), ("comb", w.comb.buffer.ptr), ("y", w.x.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])
    }

    fn kda(&self, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, rows: Scalar, cap: &str, spec: bool)
        -> Result<()> {
        let ordinal = self.kda_ordinal[index].context("KDA layer without a state pool")?;
        let at = |pool: &Dev<'_>, per: usize| -> *mut c_void {
            // SAFETY: ordinal < KDA layers, so the layer's region lies inside the pool.
            unsafe { pool.buffer.ptr.cast::<u8>().add(ordinal * per) }.cast()
        };
        let d = self.cfg.kda_heads * self.cfg.kda_head_dim;
        let conv_state = at(&self.kda_conv, self.slots * 3 * 3 * d * 2);
        let state = at(&self.kda_state, self.slots * d * self.cfg.kda_head_dim * 4);
        let replay = at(&self.kda_replay, replay_bytes(self.cfg.kda_heads, 3 * d));
        let decode = cap == "m64";
        if layer.has("w_in_fp8") {
            return self.kda_w8(w, layer, rows, cap, spec, [conv_state, state, replay]);
        }
        let mut pointers = vec![("x", w.x.buffer.ptr), ("w_in", layer.ptr("w_in")?)];
        // Decode programs read per-row scales [N, K/128]; prefill ones K-block major.
        let (in_scale, o_scale) = if decode { ("w_in_scale", "w_o_scale") } else { ("w_in_kscale", "w_o_kscale") };
        pointers.extend([("w_in_fp8", layer.ptr_or("w_in_fp8", "w_in")?), (in_scale, layer.ptr_or(in_scale, "w_in")?)]);
        pointers.extend([("w_fg", layer.ptr("w_fg")?), ("conv_w", layer.ptr("conv_w")?), ("a_log", layer.ptr("a_log")?),
            ("dt_bias", layer.ptr("dt_bias")?), ("o_norm", layer.ptr("o_norm")?), ("w_o", layer.ptr("w_o")?)]);
        pointers.extend([("w_o_fp8", layer.ptr_or("w_o_fp8", "w_o")?), (o_scale, layer.ptr_or(o_scale, "w_o")?)]);
        pointers.extend([("conv_state", conv_state), ("state", state), ("slots", w.kda_slots.buffer.ptr),
            ("seq_first", w.seq_first.buffer.ptr), ("out", w.delta.buffer.ptr)]);
        let mut scalars = self.fp8_scalars(rows, decode, layer.has(if decode { "w_in_fp8" } else { "w_in_kscale" }));
        if !decode && layer.has("w_in_kscale") {
            // Prefill fp8_rows bits: 1 the in-projection, 2 o_proj.
            scalars[1] = Scalar::I32(self.fp8_prefill.kda_bits);
        }
        if decode {
            pointers.push(("replay", replay));
            scalars.push(Scalar::I32(i32::from(spec)));
        } else {
            ensure!(!spec, "speculative steps are decode-shaped");
        }
        pointers.push(("scratch", w.scratch.buffer.ptr));
        self.run(&format!("kda_{cap}"), &pointers, &scalars)
    }

    /// KDA over the layer's only (FP8, per-row x 128-K, K-major scales) in/out
    /// projections: `kda_w8_{cap}`. Decode rows up to 16 run the FP8 GEMV, wider
    /// verify steps W8A16; prefill runs W8A8 on the `--fp8-prefill kda-*` bits, else W8A16.
    fn kda_w8(&self, w: &Workspace<'_>, layer: &GlmfLayer<'_>, rows: Scalar, cap: &str, spec: bool,
        [conv_state, state, replay]: [*mut c_void; 3]) -> Result<()> {
        let decode = cap == "m64";
        let mut pointers = vec![("x", w.x.buffer.ptr), ("w_in_fp8", layer.ptr("w_in_fp8")?),
            ("w_in_kscale", layer.ptr("w_in_kscale")?), ("w_fg", layer.ptr("w_fg")?), ("conv_w", layer.ptr("conv_w")?),
            ("a_log", layer.ptr("a_log")?), ("dt_bias", layer.ptr("dt_bias")?), ("o_norm", layer.ptr("o_norm")?),
            ("w_o_fp8", layer.ptr("w_o_fp8")?), ("w_o_kscale", layer.ptr("w_o_kscale")?), ("conv_state", conv_state),
            ("state", state), ("slots", w.kda_slots.buffer.ptr), ("seq_first", w.seq_first.buffer.ptr),
            ("out", w.delta.buffer.ptr)];
        let mut scalars = vec![rows, Scalar::I32(if decode { FP8_ROWS } else { self.fp8_prefill.kda_bits })];
        if decode {
            pointers.push(("replay", replay));
            scalars.push(Scalar::I32(i32::from(spec)));
        } else {
            ensure!(!spec, "speculative steps are decode-shaped");
        }
        pointers.push(("scratch", w.scratch.buffer.ptr));
        self.run(&format!("kda_w8_{cap}"), &pointers, &scalars)
    }

    /// `[rows, fp8]`: the decode programs' `fp8_rows` (16 when the layer has
    /// the FP8 copy, else 0), or the prefill programs' `fp8` switch (1: rows
    /// past the skinny GEMV run the block-FP8 GEMMs).
    fn fp8_scalars(&self, rows: Scalar, decode: bool, fp8: bool) -> Vec<Scalar> {
        vec![rows, Scalar::I32(match (fp8, decode) {
            (false, _) => 0,
            (true, true) => FP8_ROWS,
            (true, false) => 1,
        })]
    }

    /// SwiGLU MLP (dense layer or shared expert) of intermediate `inter` into `out`.
    fn ffn(&self, w: &Workspace<'_>, layer: &GlmfLayer<'_>, inter: usize, cap: &str, out: *mut c_void, rows: Scalar)
        -> Result<()> {
        if layer.has("nvfp4_w1") {
            let dense = self.dense_nvfp4.as_ref().context("an NVFP4 dense layer needs the fp8-glmfdense-nvfp4 package")?;
            let Scalar::I32(rows) = rows else { anyhow::bail!("row count scalar") };
            let pointers = [w.x.buffer.ptr, dense.ids.buffer.ptr, dense.weights.buffer.ptr, layer.ptr("nvfp4_w1")?,
                layer.ptr("nvfp4_s1")?, layer.ptr("nvfp4_w3")?, layer.ptr("nvfp4_s3")?, layer.ptr("nvfp4_w2")?,
                layer.ptr("nvfp4_s2")?, out, dense.scratch.buffer.ptr];
            // SAFETY: the input rows, the layer's NVFP4 operands, the constant ids/weights
            // (prefill capacity rows), the output and the scratch are live device buffers
            // used on the engine stream.
            return self.timed("ffn (NVFP4 dense)", || unsafe {
                dense.module.launch(&pointers, usize::try_from(rows)?, self.stream)
            });
        }
        let decode = cap == "m64";
        let pointers = [("x", w.x.buffer.ptr), ("w_gate_up_fp8", layer.ptr("w_gate_up_fp8")?),
            ("w_gate_up_scale", layer.ptr("w_gate_up_scale")?), ("w_down_fp8", layer.ptr("w_down_fp8")?),
            ("w_down_scale", layer.ptr("w_down_scale")?), ("out", out), ("scratch", w.scratch.buffer.ptr)];
        // FP8-only weights: decode rows up to `fp8_rows` on the GEMV; prefill W8A8 or W8A16.
        self.run(&format!("ffn_i{inter}_{cap}"), &pointers, &self.fp8_scalars(rows, decode, decode || self.fp8_prefill.ffn))
    }

    fn mla(&self, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, rows: Scalar, cap: &str,
        tables: &StepTables, trace: Option<&std::path::Path>) -> Result<()> {
        let t = tables.positions.len();
        // These scratch layouts are diagnostic-only and match the pinned AOT
        // decode programs. Stop at the first MLA layer; later layers' inputs
        // already differ and cannot identify the original numerical cause.
        let trace = trace.filter(|_| tables.decode && index == 3);
        let mode = if tables.decode { "decode" } else { "prefill" };
        let cache = self.kv[index].as_ref().context("MLA layer without a record pool")?.buffer.ptr;
        let (keys, pool_cache) = self.index[index].as_ref().context("MLA layer without an index cache")?;
        let decode = tables.decode;
        // FP8-only weights: decode rows up to `fp8_rows` on the GEMV; prefill W8A8 or W8A16.
        let fp8 = decode || self.fp8_prefill.mla;
        let mut pointers = vec![("x", w.x.buffer.ptr), ("kv_slots", w.kv_slots.buffer.ptr),
            ("w_qkv_a_fp8", layer.ptr("w_qkv_a_fp8")?), ("w_qkv_a_scale", layer.ptr("w_qkv_a_scale")?),
            ("q_a_norm", layer.ptr("q_a_norm")?), ("kv_a_norm", layer.ptr("kv_a_norm")?),
            ("w_q_b_fp8", layer.ptr("w_q_b_fp8")?), ("w_q_b_scale", layer.ptr("w_q_b_scale")?)];
        pointers.extend([("w_uk", layer.ptr("w_uk")?), ("kv_cache", cache), ("query", w.query.buffer.ptr),
            ("q_resid", w.q_resid.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
        self.run(&format!("mla_producer_{cap}"), &pointers, &self.fp8_scalars(rows, decode, fp8))?;
        if let Some(dir) = trace {
            let qkv_width = self.cfg.q_lora_rank + self.cfg.kv_lora_rank;
            let q_width = self.cfg.heads * self.cfg.qk_nope_dim;
            let bytes = (t * qkv_width * 2).next_multiple_of(1024)
                + (t * q_width * 2).next_multiple_of(1024);
            for (name, buffer, bytes) in [("mla_producer_scratch.bin", &w.scratch, bytes),
                ("mla_query.bin", &w.query, t * self.cfg.heads * self.cfg.kv_lora_rank * 2),
                ("mla_q_resid.bin", &w.q_resid, t * self.cfg.q_lora_rank * 2)] {
                std::fs::write(dir.join(name), self.download(buffer, bytes)?)?;
            }
            std::fs::write(dir.join("mla_meta.json"), serde_json::to_vec(&serde_json::json!({
                "rows": t, "qkv_width": qkv_width, "q_width": q_width, "alignment": 1024 }))?)?;
        }
        self.run(&format!("index_producer_{cap}"), &[("x", w.x.buffer.ptr), ("q_resid", w.q_resid.buffer.ptr),
            ("slots", w.kv_slots.buffer.ptr), ("pool_slots", w.pool_slots.buffer.ptr), ("w_iq", layer.ptr("w_iq")?),
            ("w_ik", layer.ptr("w_ik")?), ("k_norm_w", layer.ptr("k_norm_w")?), ("k_norm_b", layer.ptr("k_norm_b")?),
            ("ape", layer.ptr("ape")?), ("token_keys", keys.buffer.ptr), ("index_cache", pool_cache.buffer.ptr),
            ("q_fp8", w.q_fp8.buffer.ptr), ("head_weights", w.head_weights.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        if tables.long {
            self.run(&format!("index_topk_{mode}_{cap}"), &[("q_fp8", w.q_fp8.buffer.ptr),
                ("weights", w.head_weights.buffer.ptr), ("index_k_cache", pool_cache.buffer.ptr),
                ("page_table", w.pool_table.buffer.ptr), ("cache_lengths", w.cache_lengths.buffer.ptr),
                ("output_indices", w.pools.buffer.ptr), ("scratch", w.topk_scratch.buffer.ptr)],
                &[rows, Scalar::I32(tables.pool_width.max(1) as i32), Scalar::I32(tables.pool_stride as i32)])?;
        }
        self.run("index_expand", &[("positions", w.positions.buffer.ptr), ("pools", w.pools.buffer.ptr),
            ("pool_logical", self.pool_logical.buffer.ptr), ("page_table", w.page_table.buffer.ptr),
            ("indices", w.indices.buffer.ptr), ("lengths", w.lengths.buffer.ptr)],
            &[rows, Scalar::I32(tables.page_stride as i32)])?;
        if let Some(dir) = trace {
            for (name, buffer, bytes) in [("mla_indices.bin", &w.indices, t * SPARSE_TOPK * 4),
                ("mla_lengths.bin", &w.lengths, t * 4)] {
                std::fs::write(dir.join(name), self.download(buffer, bytes)?)?;
            }
            let kv = self.kv[index].as_ref().context("MLA trace without a record pool")?;
            std::fs::write(dir.join("mla_kv.bin"), self.download(kv, kv.buffer.bytes)?)?;
        }
        if let (false, Some(kernel)) = (tables.decode, crate::families::glm5::engine::native_mla_prefill()) {
            let scale = (self.cfg.qk_nope_dim as f32).powf(-0.5);
            self.timed("glm_mla_prefill (native)", || {
                // SAFETY: query, record cache, indices, lengths and the latent output
                // are live buffers of the step's rows on the engine stream.
                unsafe {
                    self.library.glm_mla_prefill(w.query.buffer.ptr, cache, w.indices.buffer.ptr, w.lengths.buffer.ptr,
                        w.latent.buffer.ptr, tables.positions.len(), self.cfg.heads, SPARSE_TOPK, RECORD_BYTES,
                        scale * std::f32::consts::LOG2_E, kernel, self.stream)
                }
            })?;
            if crate::families::glm5::engine::mla_prefill_check() {
                // SAFETY: as above; the check synchronizes the stream.
                let stats = unsafe {
                    self.library.glm_mla_prefill_check(w.query.buffer.ptr, cache, w.indices.buffer.ptr,
                        w.lengths.buffer.ptr, tables.positions.len(), self.cfg.heads, SPARSE_TOPK, RECORD_BYTES,
                        scale * std::f32::consts::LOG2_E, self.stream)
                }?;
                crate::families::glm5::engine::print_mla_check(index, &stats);
            }
        } else {
            self.run(&format!("sparse_mla_{mode}_{cap}"), &[("q", w.query.buffer.ptr), ("kv_cache", cache),
                ("indices", w.indices.buffer.ptr), ("lengths", w.lengths.buffer.ptr), ("out", w.latent.buffer.ptr),
                ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        }
        if let Some(dir) = trace {
            std::fs::write(dir.join("mla_latent.bin"), self.download(&w.latent,
                t * self.cfg.heads * self.cfg.kv_lora_rank * 2)?)?;
            std::fs::write(dir.join("mla_sparse_scratch.bin"), self.download(&w.scratch,
                self.scratch("sparse_mla_decode_m64")?)?)?;
        }
        let pointers = [("attn", w.latent.buffer.ptr), ("w_uv", layer.ptr("w_uv")?),
            ("w_o_fp8", layer.ptr("w_o_fp8")?), ("w_o_scale", layer.ptr("w_o_scale")?),
            ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)];
        self.run(&format!("o_{cap}"), &pointers, &self.fp8_scalars(rows, decode, fp8))?;
        if let Some(dir) = trace {
            std::fs::write(dir.join("mla_values.bin"), self.download(&w.scratch,
                t * self.cfg.heads * self.cfg.v_head_dim * 2)?)?;
        }
        Ok(())
    }

    /// Router, shared expert and routed experts; leaves `bf16(routed + shared)` in `delta`.
    #[allow(clippy::too_many_arguments)]
    fn moe(&self, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, t: usize, rows: Scalar, cap: &str,
        decode: bool) -> Result<()> {
        self.moe_front(w, index, layer, t, rows, cap)?;
        self.moe_experts(w, index, layer, t, rows, cap, decode)
    }

    /// Router logits, the sigmoid top-8, the shared expert (into `shared`)
    /// and, for wire-fed experts, the FP8 K32 wire rows. No host sync.
    fn moe_front(&self, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, t: usize, rows: Scalar, cap: &str)
        -> Result<()> {
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let experts = self.experts.as_ref().with_context(|| format!(
            "layer {index} is an MoE layer: pass --local-experts (FP8 package) or Spark --peers \
             (run --layers 3 for the dense layers alone)"))?;
        self.run("router_scores", &[("x", w.x.buffer.ptr), ("w", layer.ptr("gate")?),
            ("logits", w.router_logits.buffer.ptr)], &[rows])?;
        self.timed("router_select", || {
            // SAFETY: logits, bias and route outputs are live buffers of `t` rows.
            unsafe {
                self.library.router_select(w.router_logits.buffer.ptr, layer.ptr("gate.bias")?, std::ptr::null(),
                    std::ptr::null(), w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, t, self.cfg.experts, topk,
                    self.cfg.routed_scale as f32, true, self.stream)
            }
        })?;
        if !matches!(experts, Experts::Local(_)) {
            let grid = self.quantize_grid.blocks(t, h);
            self.run("expert_input_quant", &[("source_ptr", w.x.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
                // SAFETY: the scale rows follow the payload inside each wire row.
                ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
                ("scale_mma_ptr", w.delta.buffer.ptr)], &[rows, Scalar::I32(grid as i32)])?;
        }
        Ok(())
    }

    /// Per MoE layer, the weights a decode step reads after its routed
    /// experts are out, in read order: the next layer's attention site and
    /// attention (E4M3 copies where the decode programs read them), FFN site,
    /// router and shared expert; after the last layer the final norm and head.
    pub fn decode_read_order(&self) -> Vec<Vec<crate::shared::l2_prefetch::Range>> {
        let layers = &self.weights.layers;
        (0..layers.len()).map(|i| match layers.get(i + 1) {
            Some(next) => {
                let attention: &[&str] = match next.attention {
                    GlmNextAttention::Kda if next.has("w_in_fp8") => &["w_in_fp8", "w_in_kscale", "w_fg", "conv_w",
                        "a_log", "dt_bias", "o_norm", "w_o_fp8", "w_o_kscale"],
                    GlmNextAttention::Kda => &["w_in", "w_fg", "conv_w", "a_log", "dt_bias", "o_norm", "w_o"],
                    GlmNextAttention::Mla => &["w_qkv_a_fp8", "w_qkv_a_scale", "q_a_norm", "kv_a_norm", "w_q_b_fp8",
                        "w_q_b_scale", "w_iq", "w_ik", "k_norm_w", "k_norm_b", "ape", "w_uk", "w_uv", "w_o_fp8",
                        "w_o_scale"],
                };
                let names: Vec<&str> = ["attn.fn", "attn.scale", "attn.base", "input_norm"].iter().chain(attention)
                    .chain(&["ffn.fn", "ffn.scale", "ffn.base", "post_norm", "gate", "gate.bias", "w_gate_up_fp8",
                        "w_gate_up_scale", "w_down_fp8", "w_down_scale"])
                    .copied().collect();
                crate::shared::l2_prefetch::operands(&names, |n| next.range(n))
            }
            None => {
                std::iter::once(&self.weights.norm).chain(self.weights.head.allocations())
                    .map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes)).collect()
            }
        }).collect()
    }

    /// After layer `index`'s shared expert is queued in a one-lane decode or
    /// verify step (`decode`): the L2 prefetch of what the step reads next;
    /// with no real exchange (`local`), only under CUTEAFD_EMULATE_EXCHANGE_US
    /// (benchmarks), with a Spark-like wait.
    fn exchange_window(&self, index: usize, decode: bool, local: bool) -> Result<()> {
        if !decode {
            return Ok(());
        }
        let mark = if local { crate::shared::l2_prefetch::exchange_mark(self.library, self.stream)? } else { None };
        if local && mark.is_none() {
            return Ok(());
        }
        if let Some(l2) = &self.l2 {
            l2.issue(self.library, index, self.stream)?;
        }
        crate::shared::l2_prefetch::exchange_wait(self.library, mark)
    }

    /// The routed experts of layer `index` (the front ran); leaves
    /// `bf16(routed + shared)` in `delta`.
    /// The shared expert runs here, after the routes are on their way (on the
    /// Spark path while the ranks compute).
    #[allow(clippy::too_many_arguments)]
    fn moe_experts(&self, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, t: usize, rows: Scalar, cap: &str,
        decode: bool) -> Result<()> {
        let h = self.cfg.hidden;
        let experts = self.experts.as_ref().context("MoE layer without experts")?;
        let shared = || {
            self.ffn(w, layer, self.cfg.moe_intermediate, cap, w.shared.buffer.ptr, rows)?;
            self.exchange_window(index, decode, !matches!(experts, Experts::Spark { .. }))
        };
        match experts {
            Experts::Skip => {
                self.ffn(w, layer, self.cfg.moe_intermediate, cap, w.delta.buffer.ptr, rows)?;
                return self.exchange_window(index, decode, true);
            }
            Experts::Local(local) => {
                shared()?;
                let resident = local.index_of(index)?;
                let fp8 = local.experts.borrow();
                ensure!(!fp8.wire_input(), "the coordinator FP8 package takes BF16 rows");
                // SAFETY: input rows, route ids, weights and the output are live
                // buffers of `t` rows on this engine's stream.
                unsafe {
                    fp8.run(resident, t, w.x.buffer.ptr, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr,
                        w.routed.buffer.ptr, self.stream)?;
                }
                if local.window.is_some() {
                    // Diagnostic paging may drop this layer before the stream drains.
                    // SAFETY: the engine owns this stream.
                    unsafe { self.library.cuda_stream_synchronize(self.stream)? };
                }
            }
            Experts::LocalExl3(local) => {
                shared()?;
                local.ensure(index, self.stream)?;
                let mut resident = local.resident.borrow_mut();
                let (_, experts) = resident.as_mut().context("EXL3 window")?;
                // SAFETY: wire rows, routes and the shared-expert rows are complete in
                // stream order; the output is copied before the window can change.
                unsafe {
                    experts.run(crate::families::deepseek_v4::local::LocalLayer::Backbone(index), t, w.wire.buffer.ptr,
                        w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, w.shared.buffer.ptr, self.stream)?;
                    self.library.copy_d2d_async(w.delta.buffer, experts.output.buffer, t * h * 2, self.stream)?;
                }
                return Ok(());
            }
            Experts::Spark { transports, runtime } => {
                // The compact reducer adds the shared expert plane to the rank partials.
                let mut transports = transports.borrow_mut();
                let transport = transports.first_mut().context("no Spark transport")?;
                let wave = self.spark_dispatch(w, index, t, decode, transport, shared)?;
                return runtime.block_on(self.spark_land(w, t, transport, wave));
            }
        }
        self.run("add", &[("a", w.routed.buffer.ptr), ("b", w.shared.buffer.ptr), ("out", w.delta.buffer.ptr)], &[rows])
    }

    /// Routes and wire rows down and one request to every Spark rank; the
    /// shared expert (`shared`) queues behind the copies and runs while the
    /// ranks compute. Complete with [`Self::spark_land`].
    fn spark_dispatch(&self, w: &Workspace<'_>, index: usize, t: usize, decode: bool, transport: &mut SparkLink<'_>,
        shared: impl FnOnce() -> Result<()>) -> Result<SparkExpertWave> {
        let kind = if decode { ExpertV2SourceKind::Decode } else { ExpertV2SourceKind::Prefill };
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let (route_bytes, wire_bytes) = (t * topk * 4, t * (h + h / 32));
        let staging = w.router_host.borrow_mut();
        let host = staging.buffer;
        let at = |offset: usize| cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: ids, weights and wire rows are consecutive inside the pinned buffer.
            ptr: unsafe { host.ptr.cast::<u8>().add(offset) }.cast(),
            bytes: host.bytes - offset,
            ..host
        };
        let timer = std::time::Instant::now();
        // SAFETY: the pinned regions are large enough; the sync completes them.
        unsafe {
            self.library.copy_d2h_host_buffer_async(at(0), w.route_ids.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(route_bytes), w.route_weights.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(2 * route_bytes), w.wire.buffer, wire_bytes, self.stream)?;
            self.library.cuda_event_record(self.routes_ready, self.stream)?;
        }
        // The shared expert queues behind the copies and runs during the exchange;
        // the host waits for the copies only (they also order after this
        // workspace's previous plane uploads, which frees its pinned plane staging).
        shared()?;
        // SAFETY: the event was recorded on this engine's stream above.
        unsafe { self.library.cuda_event_synchronize(self.routes_ready)? };
        self.profile.borrow_mut()[0] += timer.elapsed().as_secs_f64();
        let staged = staging.bytes();
        let word = |offset: usize, i: usize| u32::from_le_bytes(staged[offset + i * 4..][..4].try_into().unwrap());
        let routes = (0..t * topk).map(|i| ExpertProtocolV2RouteEntry {
            row_index: (i / topk) as u32, expert_id: word(0, i), gate_weight: f32::from_bits(word(route_bytes, i)),
        }).collect();
        let wire = staged[2 * route_bytes..2 * route_bytes + wire_bytes].to_vec();
        drop(staging);
        let mut request = ExpertProtocolV2Request::new(index as u64 + 1, 17, index as u32, h as u32,
            ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..t as u32).map(|row| ExpertProtocolV2RowDescriptor {
                row_id: u64::from(row), source_kind: kind, source_request_id: 1,
                token_position: u64::from(row), route_offset: row * topk as u32, route_count: topk as u32,
            }).collect(),
            routes, wire)?;
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        transport.dispatch(&request)
    }

    /// Receives `wave`'s BF16 rank partials into its transport's intake planes
    /// and sums them with the shared expert into `delta`.
    async fn spark_land(&self, w: &Workspace<'_>, t: usize, transport: &mut SparkLink<'_>, wave: SparkExpertWave)
        -> Result<()> {
        let ranks = transport.world_size();
        ensure!(ranks <= MAX_RANKS, "{ranks} Spark ranks exceed the reduction planes");
        let timer = std::time::Instant::now();
        transport.receive(wave, t, self.stream).await?;
        self.profile.borrow_mut()[1] += timer.elapsed().as_secs_f64();
        // SAFETY: the shared-expert plane and `delta` are live [t, h] BF16
        // buffers; the planes are ordered after the wave by `receive`.
        unsafe { transport.reduce(w.shared.buffer.ptr.cast(), w.delta.buffer.ptr.cast(), t, self.stream) }
    }

    /// Pipelined Spark prefill of consecutive-row lanes of one sequence (each
    /// with its own workspace and transport). Units (layer, lane) run
    /// layer-major: a lane's attention needs its own previous layer posted and
    /// the previous lane's same layer done, which stream order gives (KDA
    /// state and conv windows, MLA records and DSA token/pool keys all pass
    /// through the engine's caches). Each lane's Spark wave stays in flight
    /// while the other lane's GPU layers run and is received after the next
    /// wave is dispatched, so the ranks hold the next request when they finish
    /// one. Returns the logits of the last `logit_rows` rows across the lanes.
    fn step_lanes(&self, lanes: &[(StepTables, &[u32])], logit_rows: usize, device: bool)
        -> Result<Option<StepLogits>> {
        let Some(Experts::Spark { transports, runtime }) = &self.experts else {
            anyhow::bail!("pipelined prefill needs Spark experts");
        };
        let mut transports = transports.borrow_mut();
        ensure!(transports.len() >= lanes.len(), "{} lanes need as many Spark transports", lanes.len());
        {
            let mut slots = self.lane_workspaces.borrow_mut();
            while slots.len() < lanes.len() {
                slots.push(self.workspace(self.prefill_rows, false)?);
            }
        }
        let workspaces = self.lane_workspaces.borrow();
        let total: usize = lanes.iter().map(|(t, _)| t.kv_slots.len()).sum();
        ensure!(logit_rows <= total && (self.full_prefill_logits || logit_rows <= DECODE_ROWS),
            "prefill logits past {DECODE_ROWS} rows need full_prefill_logits");
        for ((tables, tokens), w) in lanes.iter().zip(workspaces.iter()) {
            let t = tables.kv_slots.len();
            ensure!(t <= w.rows, "lane of {t} rows exceeds its workspace");
            self.put(&w.positions, &tables.positions)?;
            self.put(&w.kv_slots, &tables.kv_slots)?;
            self.put(&w.kda_slots, &tables.kda_slots)?;
            self.put(&w.seq_first, &tables.seq_first)?;
            self.put(&w.pool_slots, &tables.pool_slots)?;
            self.put(&w.cache_lengths, &tables.cache_lengths)?;
            self.put(&w.page_table, &tables.page_table)?;
            self.put(&w.pool_table, &tables.pool_table)?;
            self.load_streams(w, tokens, false)?;
        }
        let layers = &self.weights.layers;
        let cap = "m4096";
        let rows_of = |lane: usize| Scalar::I32(lanes[lane].0.kv_slots.len() as i32);
        // The drafter taps the chunk's last TAP_ROWS rows: lane `lane`'s part of
        // that window, at its offset in the tap rows.
        let tap_rows = total.min(crate::families::glm5::dflash::TAP_ROWS);
        let lane_first: Vec<usize> = lanes.iter().scan(0, |first, (t, _)| {
            let here = *first;
            *first += t.kv_slots.len();
            Some(here)
        }).collect();
        // Streams sit in buffer 0 at a layer's start and in buffer 1 mid-layer.
        let attention = |(layer, lane): (usize, usize)| -> Result<()> {
            let (w, weights, rows) = (&workspaces[lane], &layers[layer], rows_of(lane));
            let t = lanes[lane].0.kv_slots.len();
            if layer == 0 {
                self.pre(w, &w.streams[0], weights, rows)?;
            }
            match weights.attention {
                GlmNextAttention::Kda => self.kda(w, layer, weights, rows, cap, false)?,
                GlmNextAttention::Mla => self.mla(w, layer, weights, rows, cap, &lanes[lane].0, None)?,
            }
            self.post_pre(w, 0, weights, "ffn", "post_norm", rows, cap)?;
            if weights.dense {
                self.ffn(w, weights, self.cfg.dense_intermediate, cap, w.delta.buffer.ptr, rows)
            } else {
                self.moe_front(w, layer, weights, t, rows, cap)
            }
        };
        let post = |(layer, lane): (usize, usize)| -> Result<()> {
            let (w, rows) = (&workspaces[lane], rows_of(lane));
            match layers.get(layer + 1) {
                Some(next) => self.post_pre(w, 1, next, "attn", "input_norm", rows, cap)?,
                None => self.run("mhc_post", &[("x", w.delta.buffer.ptr), ("residual", w.streams[1].buffer.ptr),
                    ("prev_post", w.post.buffer.ptr), ("prev_comb", w.comb.buffer.ptr),
                    ("out", w.streams[0].buffer.ptr)], &[rows])?,
            }
            if let Some(drafter) = &self.drafter {
                let (first, t) = (lane_first[lane], lanes[lane].0.kv_slots.len());
                let window = total - tap_rows;
                let from = first.max(window);
                if from < first + t {
                    drafter.tap_streams_at(layer, w.streams[0].buffer.ptr, HC, from - first, first + t - from,
                        from - window)?;
                }
            }
            Ok(())
        };
        let units: Vec<(usize, usize)> = (0..layers.len()).flat_map(|l| (0..lanes.len()).map(move |k| (l, k))).collect();
        runtime.block_on(async {
            let mut inflight: Option<((usize, usize), SparkExpertWave)> = None;
            attention(units[0])?;
            for (index, &unit) in units.iter().enumerate() {
                let (layer, lane) = unit;
                let w = &workspaces[lane];
                if layers[layer].dense {
                    post(unit)?;
                } else {
                    let t = lanes[lane].0.kv_slots.len();
                    let shared = || self.ffn(w, &layers[layer], self.cfg.moe_intermediate, cap, w.shared.buffer.ptr,
                        rows_of(lane));
                    let wave = self.spark_dispatch(w, layer, t, false, &mut transports[lane], shared)?;
                    if let Some((previous, wave)) = inflight.take() {
                        let t = lanes[previous.1].0.kv_slots.len();
                        self.spark_land(&workspaces[previous.1], t, &mut transports[previous.1], wave).await?;
                        post(previous)?;
                    }
                    inflight = Some((unit, wave));
                }
                let next = units.get(index + 1).copied();
                // The next unit reads this unit's post when both are on one lane.
                if next.is_none_or(|(_, next_lane)| next_lane == lane) {
                    if let Some((current, wave)) = inflight.take() {
                        let t = lanes[current.1].0.kv_slots.len();
                        self.spark_land(&workspaces[current.1], t, &mut transports[current.1], wave).await?;
                        post(current)?;
                    }
                }
                if let Some(next) = next {
                    attention(next)?;
                }
            }
            anyhow::Ok(())
        })?;
        if logit_rows == 0 {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        let timer = std::time::Instant::now();
        let last = lanes.len() - 1;
        if device {
            // The last rows sit in the last lane: their logits stay in its workspace.
            let t = lanes[last].0.kv_slots.len();
            ensure!(logit_rows <= t, "device logits of {logit_rows} rows across prefill lanes");
            let w = &workspaces[last];
            self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
                ("out", w.x.buffer.ptr)], &[Scalar::I32(t as i32)])?;
            self.logits(w, t, logit_rows, false)?;
            self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
            return Ok(Some(StepLogits::Device(self.device_logits(w, logit_rows, false))));
        }
        let mut logits = Vec::with_capacity(logit_rows * self.cfg.vocab_size);
        for (((tables, _), w), &first) in lanes.iter().zip(workspaces.iter()).zip(&lane_first) {
            let t = tables.kv_slots.len();
            // Rows of this lane inside the last `logit_rows` of the chunk.
            let wanted = (first + t).saturating_sub((total - logit_rows).max(first));
            if wanted == 0 {
                continue;
            }
            self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
                ("out", w.x.buffer.ptr)], &[Scalar::I32(t as i32)])?;
            self.logits(w, t, wanted, false)?;
            let bytes = self.download(&w.logits, wanted * self.cfg.vocab_size * 4)?;
            logits.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
        }
        self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
        Ok(Some(StepLogits::Host(logits)))
    }
}

/// A prefill's logits: on the device (one workspace), or gathered across lanes on the host.
pub(crate) enum StepLogits {
    Device(DeviceLogits),
    Host(Vec<f32>),
}

impl StepLogits {
    fn into_host(self, library: &NativeLibrary) -> Result<Vec<f32>> {
        match self {
            Self::Device(logits) => logits.to_host(library),
            Self::Host(logits) => Ok(logits),
        }
    }

    fn device(self) -> Result<DeviceLogits> {
        match self {
            Self::Device(logits) => Ok(logits),
            Self::Host(_) => anyhow::bail!("prefill lanes left their logits on the host"),
        }
    }
}

impl Drop for GlmfEngine<'_> {
    fn drop(&mut self) {
        // SAFETY: the engine owns this stream and its resident weights. Drain
        // queued work, including a failed step, before their storage drops.
        unsafe {
            let _ = self.library.cuda_stream_synchronize(self.stream);
            let _ = self.library.cuda_event_destroy(self.routes_ready);
        }
        if let Some(ops) = self.ops.take() {
            let ops = ops.into_inner();
            for event in ops.pool.into_iter().chain(ops.pending.into_iter().flat_map(|(_, a, b)| [a, b])) {
                // SAFETY: as above; no launch references these events any more.
                let _ = unsafe { self.library.cuda_event_destroy(event) };
            }
        }
    }
}

#[cfg(test)]
mod prefill_lane_tests {
    use super::{prefill_lane_capacity, prefill_lane_plan, PAGE_ROWS, PREFILL_LANES};

    #[test]
    fn every_advertised_prefill_tail_fits_its_lane_workspaces() {
        for rows in [1, 63, 64, 65, 96, 127, 128, 255, 256, 511, 512, 1024, 1536, 2047, 2048, 4096] {
            let capacity = prefill_lane_capacity(rows);
            for tokens in 1..=capacity {
                let (lanes, per_lane) = prefill_lane_plan(tokens, rows).unwrap();
                assert!(lanes <= PREFILL_LANES && per_lane <= rows, "rows={rows}, tokens={tokens}");
                let starts: Vec<_> = (0..tokens).step_by(per_lane).collect();
                assert!(starts.len() <= lanes);
                assert_eq!(starts.iter().map(|&s| per_lane.min(tokens - s)).sum::<usize>(), tokens);
                assert!(starts.iter().all(|&s| per_lane.min(tokens - s) <= rows));
                assert!(starts.iter().skip(1).all(|s| s % PAGE_ROWS == 0));
            }
            assert!(prefill_lane_plan(capacity + 1, rows).is_err());
        }
    }

    #[test]
    fn narrow_workspace_splits_the_short_tool_prompt() {
        assert_eq!(prefill_lane_plan(214, 128).unwrap(), (2, 128));
        assert_eq!(prefill_lane_plan(1, 1).unwrap(), (1, 1));
        assert!(prefill_lane_plan(0, 128).is_err());
        assert!(prefill_lane_plan(1, 0).is_err());
        // Keep the qualified default's lane threshold and advertised width.
        assert_eq!(prefill_lane_capacity(4096), 8192);
        assert_eq!(prefill_lane_plan(511, 4096).unwrap(), (1, 511));
        assert_eq!(prefill_lane_plan(512, 4096).unwrap(), (2, 256));
    }
}
