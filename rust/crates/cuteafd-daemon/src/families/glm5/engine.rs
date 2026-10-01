//! GLM 5.x (glm_moe_dsa) coordinator over the exported glm_* programs.
//!
//! One layer: input norm (fused with the previous layer's residual add),
//! MLA producer (writes the FP8 656-byte latent record), DSA index producer
//! and top-k on full-indexer layers (shared layers reuse the last full
//! layer's selection), sparse MLA, W_UV + o_proj, post-attention norm, then
//! the dense MLP or the MoE (router, expert input quantization, shared
//! expert, routed experts). The latent and index caches share page ids.
//!
//! A long Spark prefill chunk runs as two lanes of consecutive rows, each with
//! its own workspace and transport ([`GlmEngine::prefill`]): one lane's Spark
//! wave stays in flight while the other lane's GPU layers run.
use super::weights::{GlmLayer, GlmWeights};
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use crate::shared::token_io::{DeviceLogits, EmbedPlacement, TokenEmbedding};
use crate::shared::spark_intake::{copy_parallel, IntakeMode, SparkIntake, SparkLane, SparkLink};
use cuteafd_transport::expert::{
    SparkExpertWave, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16,
};
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Programs, Scalar, VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::glm5::GlmDsaConfig;
use std::cell::RefCell;
use std::ffi::c_void;

type Dev<'a> = DeviceAllocation<'a>;

pub(crate) const PAGE_ROWS: usize = 64;
pub(crate) const RECORD_BYTES: usize = 656;
const RECORD_PAGE_BYTES: usize = PAGE_ROWS * RECORD_BYTES;
pub(crate) const INDEX_PAGE_BYTES: usize = 8448;
/// Most Spark ranks a step's partials come from (the compact reducer's limit).
const MAX_RANKS: usize = 6;
/// Ranks whose (zero) partials a skipped exchange uploads: the TP4 layout.
const SKIP_RANKS: usize = 4;
/// Rows of the decode-route programs (`_m64`).
pub(crate) const DECODE_ROWS: usize = 64;
/// Most lanes a long Spark prefill chunk splits into (CUTEAFD_GLM_PREFILL_LANES,
/// default [`DEFAULT_LANES`]; 1 is the serial path), and the fewest rows per
/// lane worth another exchange per layer.
pub(crate) const PREFILL_LANES: usize = 4;
const DEFAULT_LANES: usize = 3;
const MIN_LANE_ROWS: usize = 256;

/// What precedes a decode segment's residual norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Previous {
    First,
    Delta,
    Planes(usize),
}

/// Everything a captured decode segment bakes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GraphKey {
    layer: usize,
    rows: usize,
    table_width: usize,
    table_stride: usize,
    previous: Previous,
    /// The last segment ends in the head and the greedy selection.
    head: bool,
}

struct GraphExec<'a>(*mut c_void, &'a NativeLibrary);

impl Drop for GraphExec<'_> {
    fn drop(&mut self) {
        // SAFETY: the exec came from end_capture and is destroyed once.
        let _ = unsafe { self.1.cuda_graph_exec_destroy(self.0) };
    }
}

/// Host tables of one step.
struct StepTables {
    decode: bool,
    positions: Vec<i64>,
    slots: Vec<i64>,
    /// Prefill: the sequence's pages; decode: one padded row per step row.
    page_table: Vec<i32>,
    table_width: usize,
    table_stride: usize,
    cache_lengths: Vec<i32>,
    /// Selected entries per row the sparse MLA reads (every earlier token
    /// below the index top-k; the selection leads each indices row).
    lengths: Vec<i32>,
}

/// A sequence's pages (shared by the latent and index caches) and length.
#[derive(Debug, Clone)]
pub(crate) struct GlmPlacement {
    pub pages: Vec<u32>,
    pub len: usize,
}

impl GlmPlacement {
    pub fn slot(&self, position: usize) -> Result<i64> {
        let page = *self.pages.get(position / PAGE_ROWS).context("position past the sequence's pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + (position % PAGE_ROWS) as i64)
    }

    /// The first `count` pages as a page-table row.
    fn table(&self, count: usize) -> impl Iterator<Item = i32> + '_ {
        self.pages[..count].iter().map(|&p| p as i32)
    }
}

/// Refcounted pages of the shared latent/index cache pool (goldens and benches; serving takes
/// its pages from the prefix cache's pool).
pub(crate) struct PageAllocator {
    pool: cuteafd_engine::prefix::RefPagePool,
}

impl PageAllocator {
    pub fn new(pages: usize) -> Self {
        Self { pool: cuteafd_engine::prefix::RefPagePool::new(pages, PAGE_ROWS) }
    }

    /// Reserves every page a sequence of up to `capacity` tokens needs.
    pub fn admit(&mut self, capacity: usize) -> Result<GlmPlacement> {
        let pages = self.pool.alloc(self.pool.pages_for(capacity)).context("cache pages exhausted")?;
        Ok(GlmPlacement { pages, len: 0 })
    }

    /// A second sequence starting as `source`'s first `len` rows: full pages shared, the
    /// partial tail page copied by the caller (the returned copy).
    pub fn fork(&mut self, source: &GlmPlacement, len: usize, capacity: usize)
        -> Result<(GlmPlacement, Option<cuteafd_engine::prefix::TailCopy>)> {
        let fork = self.pool.fork(&source.pages, len, self.pool.pages_for(capacity)).context("cache pages exhausted")?;
        Ok((GlmPlacement { pages: fork.pages, len: 0 }, fork.copy))
    }

    pub fn release(&mut self, placement: GlmPlacement) {
        self.pool.release(&placement.pages);
    }
}

struct Workspace<'a> {
    rows: usize,
    h: Dev<'a>,
    x: Dev<'a>,
    query: Dev<'a>,
    q_resid: Dev<'a>,
    q_fp8: Dev<'a>,
    head_weights: Dev<'a>,
    indices: Dev<'a>,
    lengths: Dev<'a>,
    attn: Dev<'a>,
    delta: Dev<'a>,
    positions: Dev<'a>,
    slots: Dev<'a>,
    page_table: Dev<'a>,
    cache_lengths: Dev<'a>,
    scratch: Dev<'a>,
    topk_scratch: Dev<'a>,
    logits: Dev<'a>,
    /// MoE: router logits (FP32), routes, wire rows, shared-expert output,
    /// rank partial planes, and their pinned staging.
    router_logits: Dev<'a>,
    route_ids: Dev<'a>,
    route_weights: Dev<'a>,
    wire: Dev<'a>,
    shared: Dev<'a>,
    router_host: RefCell<HostAllocation<'a>>,
    /// The step's token ids (U32, gathered from the device embedding table).
    ids: Dev<'a>,
    /// Greedy selection of the logits rows inside the decode graph: U32 ids, then U32 statuses.
    select: Dev<'a>,
    head: VocabularyHead<'a>,
    _head_workspace: Dev<'a>,
}

/// A step's logits: on the device (one workspace's rows), or downloaded
/// (rows spanning several prefill lanes).
enum StepLogits {
    Device(DeviceLogits),
    Host(Vec<f32>),
}

pub(crate) struct GlmEngine<'a> {
    pub library: &'a NativeLibrary,
    pub programs: &'a Programs<'a>,
    pub cfg: GlmDsaConfig,
    pub weights: GlmWeights<'a>,
    pub stream: *mut c_void,
    pub max_context: usize,
    pub prefill_rows: usize,
    pub pages: usize,
    kv: Vec<Dev<'a>>,
    index: Vec<Option<Dev<'a>>>,
    cos_sin: Dev<'a>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    /// One prefill workspace per lane (a serial prefill uses the first) and
    /// one Spark transport thread per lane: a lane's request assembly,
    /// post, polling and partial copies run there while this thread queues
    /// GPU work. Without lane transports, or with CUTEAFD_GLM_PREFILL_LANES=1,
    /// prefill is serial on the caller's transport.
    lane_workspaces: RefCell<Vec<Workspace<'a>>>,
    /// Lanes a long prefill chunk splits into.
    prefill_lanes: usize,
    pub lanes: RefCell<Vec<SparkLane<'a>>>,
    /// Prefill steps keep every row's logits (golden scoring); otherwise the
    /// prefill workspace holds logits for at most `DECODE_ROWS` rows.
    pub full_prefill_logits: bool,
    /// Benchmarks: MoE layers skip the Spark exchange (see
    /// `EngineArgs::skip_routed_experts`) and reduce zero partials of
    /// [`SKIP_RANKS`] ranks uploaded through this host-mode intake.
    skip: Option<SparkIntake<'a>>,
    /// Host time per phase since the last reset: GPU wait before the expert
    /// request, the Spark exchange, the logits download.
    pub profile: RefCell<[f64; 3]>,
    /// Host seconds inside the exchanges: building and posting requests,
    /// copying received partials into the pinned staging.
    pub exchange_host: RefCell<[f64; 4]>,
    graphs: RefCell<std::collections::HashMap<GraphKey, GraphExec<'a>>>,
    /// The DFlash2 drafter; every step taps its target layers.
    pub drafter: Option<super::dflash::GlmDrafter<'a>>,
    /// L2 prefetch of the next layer's weights during decode exchanges.
    pub l2: Option<crate::shared::l2_prefetch::L2Prefetch>,
    /// The token embedding table (resident on this GPU or read from its shard).
    pub embedding: TokenEmbedding<'a>,
    /// Prefill projections over the FP8 weights run W8A8 (E4M3 activations per
    /// row and 128-K block, the official FP8 release's served numerics); false:
    /// W8A16 (bitwise the former BF16 prefill over dequantized weights).
    pub prefill_w8a8: bool,
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl<'a> GlmEngine<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, cfg: GlmDsaConfig,
        weights: GlmWeights<'a>, stream: *mut c_void, max_context: usize, prefill_rows: usize, pages: usize,
        embedding: TokenEmbedding<'a>) -> Result<Self> {
        ensure!(embedding.hidden() == cfg.hidden, "embedding rows of {} for hidden {}", embedding.hidden(), cfg.hidden);
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let layers = weights.layers.len();
        let kv = (0..layers).map(|_| zeroed(pages * RECORD_PAGE_BYTES)).collect::<Result<Vec<_>>>()?;
        let index = weights.layers.iter()
            .map(|l| l.full_indexer.then(|| zeroed(pages * INDEX_PAGE_BYTES)).transpose())
            .collect::<Result<Vec<_>>>()?;
        // cos | sin of position * theta^(-2i/64), FP32 like the reference's inv_freq.
        let dim = cfg.qk_rope_head_dim;
        let mut table = vec![0f32; max_context * dim];
        let inv: Vec<f32> = (0..dim / 2).map(|i| 1.0 / (cfg.rope_theta as f32).powf((2 * i) as f32 / dim as f32)).collect();
        for p in 0..max_context {
            for (i, f) in inv.iter().enumerate() {
                let angle = p as f32 * f;
                table[p * dim + i] = angle.cos();
                table[p * dim + dim / 2 + i] = angle.sin();
            }
        }
        let cos_sin = DeviceAllocation::new(library, table.len() * 4)?;
        library.copy_h2d(cos_sin.buffer, bytes_of(&table))?;
        Ok(Self { library, programs, cfg, weights, stream, max_context, prefill_rows, pages, kv, index, cos_sin,
            decode_workspace: RefCell::new(None), lane_workspaces: RefCell::new(Vec::new()),
            prefill_lanes: configured_lanes(),
            lanes: RefCell::new(Vec::new()), full_prefill_logits: false, skip: None, l2: None, profile: RefCell::new([0.0; 3]),
            exchange_host: RefCell::new([0.0; 4]),
            graphs: RefCell::new(std::collections::HashMap::new()), drafter: None, embedding, prefill_w8a8: true })
    }

    /// Every layer's latent record pool (656 B per row, 64-row pages) and, on full-indexer
    /// layers, its index-key pool (8448 B per page: 64 x 128 E4M3 keys, then 64 FP32 scales).
    pub(crate) fn paged_buffers(&self) -> Vec<(cuteafd_ffi::CuteafdDeviceBuffer, Option<cuteafd_ffi::CuteafdDeviceBuffer>)> {
        self.kv.iter().zip(&self.index).map(|(kv, index)| (kv.buffer, index.as_ref().map(|i| i.buffer))).collect()
    }

    /// Benchmarks: MoE layers skip the Spark exchange from now on.
    pub fn skip_routed_experts(&mut self) -> Result<()> {
        let rows = self.prefill_rows.max(DECODE_ROWS);
        self.skip = Some(SparkIntake::new(self.library, IntakeMode::Host, SKIP_RANKS, rows, self.cfg.hidden * 2)?);
        Ok(())
    }

    /// Whether MoE layers skip the Spark exchange.
    pub fn skip_routed(&self) -> bool {
        self.skip.is_some()
    }

    fn alloc(&self, bytes: usize) -> Result<Dev<'a>> {
        DeviceAllocation::new(self.library, bytes.max(256))
    }

    fn run(&self, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar]) -> Result<()> {
        let names: Vec<&str> = pointers.iter().map(|(n, _)| *n).collect();
        let program = self.programs.program(name, &names)?;
        let raw: Vec<*mut c_void> = pointers.iter().map(|(_, p)| *p).collect();
        // SAFETY: every pointer names a live allocation sized for the rows in
        // `scalars`; the stream orders all launches of this engine.
        unsafe { program.launch(&raw, scalars, self.stream) }.with_context(|| format!("{name} with {scalars:?}"))
    }

    fn scratch(&self, name: &str) -> Result<usize> {
        Ok(self.programs.spec(name)?.scratch.get("scratch").copied().unwrap_or(0) as usize)
    }

    fn workspace(&self, t: usize, decode: bool) -> Result<Workspace<'a>> {
        let (h, heads) = (self.cfg.hidden, self.cfg.heads);
        let (cap, mode) = if decode { ("m64", "decode") } else { ("m4096", "prefill") };
        let mut scratch = 0usize;
        for name in [format!("glm_producer_{cap}"), format!("glm_index_producer_{cap}"),
            format!("glm_sparse_mla_{mode}_{cap}"), format!("glm_o_{cap}"), format!("glm_ffn_i2048_{cap}"),
            format!("glm_ffn_i12288_{cap}")] {
            scratch = scratch.max(self.scratch(&name)?);
        }
        let head_workspace = self.alloc(VOCABULARY_HEAD_WORKSPACE)?;
        let topk = self.alloc(self.scratch(&format!("glm_index_topk_{mode}_{cap}"))?)?;
        self.library.cuda_zero_bytes(topk.buffer, topk.buffer.bytes)?;
        let lengths: Vec<i32> = vec![self.cfg.index_topk as i32; t];
        let lengths_dev = self.alloc(t * 4)?;
        self.library.copy_h2d(lengths_dev.buffer, bytes_of(&lengths))?;
        Ok(Workspace {
            rows: t,
            h: self.alloc(t * h * 2)?,
            x: self.alloc(t * h * 2)?,
            query: self.alloc(t * heads * 576 * 2)?,
            q_resid: self.alloc(t * self.cfg.q_lora_rank * 2)?,
            q_fp8: self.alloc(t * self.cfg.index_heads * self.cfg.index_head_dim)?,
            head_weights: self.alloc(t * self.cfg.index_heads * 4)?,
            indices: self.alloc(t * self.cfg.index_topk * 4)?,
            lengths: lengths_dev,
            attn: self.alloc(t * heads * 512 * 2)?,
            delta: self.alloc(t * h * 2)?,
            positions: self.alloc(t * 8)?,
            slots: self.alloc(t * 8)?,
            page_table: self.alloc(if decode { t * self.pages * 4 } else { self.pages * 4 })?,
            cache_lengths: self.alloc(t * 4)?,
            scratch: self.alloc(scratch)?,
            topk_scratch: topk,
            logits: self.alloc(if decode || self.full_prefill_logits { t } else { t.min(DECODE_ROWS) }
                * self.cfg.vocab_size * 4)?,
            router_logits: self.alloc(t * self.cfg.experts * 4)?,
            route_ids: self.alloc(t * self.cfg.topk * 4)?,
            route_weights: self.alloc(t * self.cfg.topk * 4)?,
            wire: self.alloc(t * (h + h / 32))?,
            shared: self.alloc(t * h * 2)?,
            router_host: RefCell::new(HostAllocation::new(self.library, t * (self.cfg.topk * 8 + h + h / 32))?),
            ids: self.alloc(t * 4)?,
            select: self.alloc(t * 8)?,
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, h as u32, t as u32,
                self.cfg.vocab_size as u32)? },
            _head_workspace: head_workspace,
        })
    }

    fn put<T: Copy>(&self, dev: &Dev<'_>, values: &[T]) -> Result<()> {
        let bytes = bytes_of(values);
        ensure!(bytes.len() <= dev.buffer.bytes, "table exceeds its buffer");
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: bytes.len(), ..dev.buffer }, bytes)
    }

    fn download(&self, dev: &Dev<'_>, bytes: usize) -> Result<Vec<u8>> {
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        let mut out = vec![0u8; bytes];
        self.library.copy_d2h(&mut out, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..dev.buffer })?;
        Ok(out)
    }

    /// Prefills a sequence from its length through every resident layer and
    /// returns the last row's logits when all layers are resident.
    /// `on_layer` receives each layer's output rows (BF16 [t, hidden]).
    pub fn prefill(&self, placement: &mut GlmPlacement, tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        self.prefill_rows_logits(placement, tokens, experts, on_layer, 1)
    }

    /// [`Self::prefill`] leaving the last row's logits on the device.
    pub fn prefill_device(&self, placement: &mut GlmPlacement, tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>) -> Result<Option<DeviceLogits>> {
        match self.prefill_step(placement, tokens, experts, None, 1)? {
            Some(StepLogits::Device(logits)) => Ok(Some(logits)),
            Some(StepLogits::Host(_)) => anyhow::bail!("the last prefill row's logits span lanes"),
            None => Ok(None),
        }
    }

    /// Downloads a step's logits (counted in the logits phase).
    fn host_logits(&self, logits: Option<StepLogits>) -> Result<Option<Vec<f32>>> {
        let timer = std::time::Instant::now();
        let out = match logits {
            Some(StepLogits::Device(logits)) => Some(logits.to_host(self.library)?),
            Some(StepLogits::Host(logits)) => Some(logits),
            None => None,
        };
        self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
        Ok(out)
    }

    /// Whether a Spark prefill runs as lanes (a lane transport, every layer resident).
    fn pipelined(&self) -> bool {
        self.prefill_lanes > 1 && self.lanes.borrow().len() >= self.prefill_lanes
            && self.weights.layers.len() == self.cfg.layers
    }

    /// Longest chunk one prefill call takes (a lane of `prefill_rows` each when pipelined).
    pub fn prefill_capacity(&self) -> usize {
        if self.pipelined() { self.prefill_lanes * self.prefill_rows } else { self.prefill_rows }
    }

    fn prefill_tables(&self, placement: &GlmPlacement, start: usize, t: usize) -> Result<StepTables> {
        let used = (start + t).div_ceil(PAGE_ROWS);
        Ok(StepTables {
            decode: false,
            positions: (start..start + t).map(|p| p as i64).collect(),
            slots: (start..start + t).map(|p| placement.slot(p)).collect::<Result<_>>()?,
            page_table: placement.table(used).collect(),
            table_width: used,
            table_stride: 0,
            cache_lengths: (start..start + t).map(|p| (p + 1) as i32).collect(),
            lengths: (start..start + t).map(|p| (p + 1).min(self.cfg.index_topk) as i32).collect(),
        })
    }

    /// [`Self::prefill`] returning the logits of the chunk's last `logit_rows` rows.
    /// Without `on_layer`, a Spark chunk of at least 2 x 256 rows runs as
    /// [`PREFILL_LANES`] lanes (see [`Self::step_lanes`]).
    pub fn prefill_rows_logits(&self, placement: &mut GlmPlacement, tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>, logit_rows: usize) -> Result<Option<Vec<f32>>> {
        let logits = self.prefill_step(placement, tokens, experts, on_layer, logit_rows)?;
        self.host_logits(logits)
    }

    fn prefill_step(&self, placement: &mut GlmPlacement, tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>, logit_rows: usize) -> Result<Option<StepLogits>> {
        let (start, t) = (placement.len, tokens.len());
        if on_layer.is_none() && self.pipelined() && t >= 2 * MIN_LANE_ROWS && experts.is_some() {
            // Lanes split at 64-row pages, so each lane's pages start where the previous lane's end.
            let count = self.prefill_lanes.min(t / MIN_LANE_ROWS);
            let per_lane = t.div_ceil(count).next_multiple_of(PAGE_ROWS);
            ensure!(per_lane <= self.prefill_rows && start + t <= self.max_context,
                "prefill of {t} rows at {start} exceeds {} rows per lane or the context", self.prefill_rows);
            let (mut lanes, mut chunks) = (Vec::new(), Vec::new());
            let mut first = 0;
            while first < t {
                let n = per_lane.min(t - first);
                lanes.push(self.prefill_tables(placement, start + first, n)?);
                chunks.push(&tokens[first..first + n]);
                first += n;
            }
            let logits = self.step_lanes(&lanes, &chunks, logit_rows)?;
            placement.len += t;
            return Ok(logits);
        }
        ensure!(t > 0 && t <= self.prefill_rows && start + t <= self.max_context, "prefill of {t} rows at {start}");
        let tables = self.prefill_tables(placement, start, t)?;
        let logits = self.step(&tables, tokens, logit_rows, experts, on_layer)?;
        placement.len += t;
        Ok(logits.map(StepLogits::Device))
    }

    /// Appends each sequence's tokens (one for decode, several for a
    /// speculative verify) at its length in one decode-shaped step; returns
    /// every row's logits. A caller that rejects a suffix sets `len` back:
    /// GLM keeps no recurrent state, and rejected records are overwritten.
    pub fn verify(&self, sequences: &mut [(&mut GlmPlacement, usize)], tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        let logits = self.verify_device_hooked(sequences, tokens, experts, on_layer)?;
        self.host_logits(logits.map(StepLogits::Device))
    }

    /// [`Self::verify`] leaving every row's logits on the device, with the
    /// decode graph's greedy selection of them.
    pub fn verify_device(&self, sequences: &mut [(&mut GlmPlacement, usize)], tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>) -> Result<Option<DeviceLogits>> {
        self.verify_device_hooked(sequences, tokens, experts, None)
    }

    fn verify_device_hooked(&self, sequences: &mut [(&mut GlmPlacement, usize)], tokens: &[u32],
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<DeviceLogits>> {
        let rows: usize = sequences.iter().map(|(_, n)| n).sum();
        ensure!(rows > 0 && rows <= DECODE_ROWS && tokens.len() == rows, "decode step of {rows} rows");
        let stride = sequences.iter().map(|(p, _)| p.pages.len()).max().unwrap_or(1);
        let mut tables = StepTables { decode: true, positions: Vec::new(), slots: Vec::new(), page_table: Vec::new(),
            table_width: 1, table_stride: stride, cache_lengths: Vec::new(), lengths: Vec::new() };
        for (placement, count) in sequences.iter() {
            for position in placement.len..placement.len + count {
                ensure!(position < self.max_context, "decode at {position} past the context");
                tables.positions.push(position as i64);
                tables.slots.push(placement.slot(position)?);
                tables.cache_lengths.push((position + 1) as i32);
                // The index top-k selects every earlier token up to its k,
                // leading the row: the sparse MLA reads only those.
                tables.lengths.push((position + 1).min(self.cfg.index_topk) as i32);
                // Power-of-two widths bound the graphs a growing context captures.
                tables.table_width = tables.table_width.max((position + 1).div_ceil(PAGE_ROWS).next_power_of_two().min(stride));
                tables.page_table.extend(placement.table(placement.pages.len()));
                tables.page_table.extend(std::iter::repeat_n(0, stride - placement.pages.len()));
            }
        }
        let logits = self.step(&tables, tokens, rows, experts, on_layer)?;
        for (placement, count) in sequences.iter_mut() {
            placement.len += *count;
        }
        Ok(logits)
    }

    fn step(&self, tables: &StepTables, tokens: &[u32], logit_rows: usize,
        mut experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<DeviceLogits>> {
        let (h, t) = (self.cfg.hidden, tables.positions.len());
        // Decode steps have their own workspace; a serial prefill uses the first lane's.
        if tables.decode && self.decode_workspace.borrow().is_none() {
            *self.decode_workspace.borrow_mut() = Some(self.workspace(DECODE_ROWS, true)?);
        }
        if !tables.decode && self.lane_workspaces.borrow().is_empty() {
            let first = self.workspace(self.prefill_rows, false)?;
            self.lane_workspaces.borrow_mut().push(first);
        }
        let (decode_workspace, lane_workspaces) = (self.decode_workspace.borrow(), self.lane_workspaces.borrow());
        let w = if tables.decode { decode_workspace.as_ref().context("workspace")? } else { &lane_workspaces[0] };
        ensure!(t <= w.rows && logit_rows <= t, "step exceeds the workspace");
        ensure!(tables.decode || self.full_prefill_logits || logit_rows <= DECODE_ROWS,
            "prefill logits past {DECODE_ROWS} rows need full_prefill_logits");
        self.put(&w.positions, &tables.positions)?;
        self.put(&w.slots, &tables.slots)?;
        self.put(&w.page_table, &tables.page_table)?;
        self.put(&w.cache_lengths, &tables.cache_lengths)?;
        self.put(&w.lengths, &tables.lengths)?;
        ensure!(tokens.len() == t, "{} tokens for a {t}-row step", tokens.len());
        let rows = Scalar::I32(t as i32);
        let cap = if tables.decode { "m64" } else { "m4096" };
        let layers = &self.weights.layers;
        if tables.decode && on_layer.is_none() && layers.len() == self.cfg.layers {
            // The first captured segment gathers the rows from the staged ids;
            // the last runs the head and the greedy selection.
            let gather = self.embedding.placement() == EmbedPlacement::Gpu;
            if gather {
                self.embedding.check(tokens)?;
                self.put(&w.ids, tokens)?;
            } else {
                self.embedding.embed(tokens, w.ids.buffer, 1, w.h.buffer, self.stream)?;
            }
            let head = logit_rows == t;
            self.decode_layers(w, tables, t, &mut experts, gather, head)?;
            if !head {
                self.launch_head(w, t, logit_rows)?;
            }
            return Ok(Some(self.device_logits(w, logit_rows, head)));
        }
        self.embedding.embed(tokens, w.ids.buffer, 1, w.h.buffer, self.stream)?;
        self.norm(w, &layers[0], "input_norm", 0, rows)?;
        for (index, layer) in layers.iter().enumerate() {
            self.attention(w, index, layer, rows, cap, tables)?;
            // h += attention; x = post_attention_layernorm(h)
            self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                ("delta1", w.delta.buffer.ptr), ("weight", layer.ptr("post_norm")?), ("out", w.x.buffer.ptr)],
                &[rows, Scalar::I32(1)])?;
            if layer.dense {
                self.ffn(w, layer, self.cfg.dense_intermediate, cap, w.delta.buffer.ptr, rows)?;
            } else if let Some(skip) = &self.skip {
                self.moe_front(w, layer, t)?;
                let ranks = self.moe_skip(w, index, layer, t, cap)?;
                self.reduce(skip.pointers(), w, ranks, t)?;
            } else {
                let (transport, runtime) = experts.as_mut()
                    .with_context(|| format!("layer {index} is an MoE layer; pass Spark peers for its routed experts"))?;
                self.moe(w, index, layer, t, cap, tables.decode, transport, runtime)?;
            }
            // h += ffn; x = next input_layernorm(h) (or the final norm).
            let weight = match layers.get(index + 1) {
                Some(next) => next.ptr("input_norm")?,
                None => self.weights.norm.buffer.ptr,
            };
            self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                ("delta1", w.delta.buffer.ptr), ("weight", weight), ("out", w.x.buffer.ptr)],
                &[rows, Scalar::I32(1)])?;
            if let Some(drafter) = &self.drafter {
                let n = t.min(super::dflash::TAP_ROWS);
                drafter.tap(index, w.h.buffer.ptr, t - n, n)?;
            }
            if let Some(on_layer) = on_layer.as_mut() {
                on_layer(index, &self.download(&w.h, t * h * 2)?)?;
            }
        }
        if layers.len() < self.cfg.layers {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        self.launch_head(w, t, logit_rows)?;
        Ok(Some(self.device_logits(w, logit_rows, false)))
    }

    /// Logits of the last `n` of `t` rows of the final norm's output into the workspace's logits.
    fn launch_head(&self, w: &Workspace<'_>, t: usize, n: usize) -> Result<()> {
        let h = self.cfg.hidden;
        // SAFETY: the final norm's output and the head operands are live buffers of these shapes.
        unsafe {
            super::launch_head(self.library, &w.head, w.x.buffer.ptr.cast::<u8>().add((t - n) * h * 2).cast(),
                self.weights.head.buffer.ptr, w.logits.buffer.ptr.cast(), n, h, self.cfg.vocab_size, self.stream)
        }
    }

    /// `bytes` at `offset` inside `dev`.
    fn region(dev: &Dev<'_>, offset: usize, bytes: usize) -> cuteafd_ffi::CuteafdDeviceBuffer {
        debug_assert!(offset + bytes <= dev.buffer.bytes);
        cuteafd_ffi::CuteafdDeviceBuffer {
            // SAFETY: callers pass offsets inside the buffer.
            ptr: unsafe { dev.buffer.ptr.cast::<u8>().add(offset) }.cast(),
            bytes,
            ..dev.buffer
        }
    }

    /// Greedy tokens of the first `rows` logits rows into `select`.
    fn select_greedy(&self, w: &Workspace<'_>, rows: usize) -> Result<()> {
        let vocab = self.cfg.vocab_size;
        // SAFETY: the logits rows and the select buffer (ids, then statuses) are live buffers of these shapes.
        unsafe {
            self.library.cuda_logits_greedy_f32_async(w.logits.buffer.ptr, rows, vocab, vocab, w.select.buffer.ptr,
                std::ptr::null_mut(), Self::region(&w.select, rows * 4, rows * 4).ptr, self.stream)
        }
    }

    /// The first `rows` logits rows as device logits (`greedy`: with the rows'
    /// selection from [`Self::select_greedy`]).
    fn device_logits(&self, w: &Workspace<'_>, rows: usize, greedy: bool) -> DeviceLogits {
        let vocab = self.cfg.vocab_size;
        DeviceLogits { ptr: w.logits.buffer.ptr, rows, vocab, stride: vocab, stream: self.stream,
            greedy: greedy.then(|| (w.select.buffer.ptr.cast_const(), Self::region(&w.select, rows * 4, rows * 4).ptr
                .cast_const())) }
    }

    /// Decode layers as captured graph segments: each replays the previous
    /// layer's reduce + residual norm, this layer's attention and
    /// post-attention norm, and the dense FFN or the MoE front (router,
    /// select, wire rows); the Spark exchange runs between segments.
    fn decode_layers(&self, w: &Workspace<'_>, tables: &StepTables, t: usize,
        experts: &mut Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>, gather: bool, head: bool) -> Result<()> {
        let rows = Scalar::I32(t as i32);
        let layers = &self.weights.layers;
        // The decode transport's intake planes (or the skip intake's); the
        // replayed graphs bake them in.
        let planes = match (experts.as_ref(), &self.skip) {
            (_, Some(skip)) => skip.pointers(),
            (Some((transport, _)), None) => transport.intake.pointers(),
            (None, None) => [std::ptr::null(); 6],
        };
        // Previous layer's FFN output: none (first layer), in `delta`, or Spark planes.
        let mut previous = Previous::First;
        for index in 0..=layers.len() {
            let layer = layers.get(index);
            let segment = || -> Result<()> {
                if index == 0 && gather {
                    // SAFETY: the step's ids are in `ids` before the replay; `h` holds its rows.
                    unsafe { self.embedding.gather(w.ids.buffer.ptr, std::ptr::null(), t, 1, std::ptr::null(),
                        w.h.buffer.ptr, self.stream)? };
                }
                if let Previous::Planes(ranks) = previous {
                    self.reduce(planes, w, ranks, t)?;
                }
                let weight = match layer {
                    Some(layer) => layer.ptr("input_norm")?,
                    None => self.weights.norm.buffer.ptr,
                };
                let deltas = if matches!(previous, Previous::First) { 0 } else { 1 };
                self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                    ("delta1", w.delta.buffer.ptr), ("weight", weight), ("out", w.x.buffer.ptr)],
                    &[rows, Scalar::I32(deltas)])?;
                // `h` now holds the previous layer's output.
                if let (Some(drafter), Some(previous)) = (&self.drafter, index.checked_sub(1)) {
                    drafter.tap(previous, w.h.buffer.ptr, 0, t)?;
                }
                let Some(layer) = layer else {
                    if head {
                        self.launch_head(w, t, t)?;
                        self.select_greedy(w, t)?;
                    }
                    return Ok(());
                };
                self.attention(w, index, layer, rows, "m64", tables)?;
                self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                    ("delta1", w.delta.buffer.ptr), ("weight", layer.ptr("post_norm")?), ("out", w.x.buffer.ptr)],
                    &[rows, Scalar::I32(1)])?;
                if layer.dense {
                    self.ffn(w, layer, self.cfg.dense_intermediate, "m64", w.delta.buffer.ptr, rows)
                } else {
                    self.moe_front(w, layer, t)
                }
            };
            let key = GraphKey { layer: index, rows: t, table_width: tables.table_width,
                table_stride: tables.table_stride, previous, head: head && layer.is_none() };
            self.replay(key, segment)?;
            previous = match layer {
                None => break,
                Some(layer) if layer.dense => Previous::Delta,
                Some(layer) if self.skip.is_some() => Previous::Planes(self.moe_skip(w, index, layer, t, "m64")?),
                Some(layer) => {
                    let (transport, runtime) = experts.as_mut()
                        .with_context(|| format!("layer {index} is an MoE layer; pass Spark peers for its routed experts"))?;
                    Previous::Planes(self.moe_exchange(w, index, layer, t, "m64", true, transport, runtime)?)
                }
            };
        }
        Ok(())
    }

    /// Launches `segment` through a graph captured the first time `key` is seen.
    fn replay(&self, key: GraphKey, segment: impl FnOnce() -> Result<()>) -> Result<()> {
        if let Some(graph) = self.graphs.borrow().get(&key) {
            // SAFETY: the graph's pointers are persistent engine buffers.
            return unsafe { self.library.cuda_graph_launch(graph.0, self.stream) };
        }
        // SAFETY: capture records launches on the engine stream; nothing in the
        // segment synchronizes the host.
        unsafe { self.library.cuda_graph_begin_capture(self.stream)? };
        let captured = segment();
        let exec = unsafe { self.library.cuda_graph_end_capture(self.stream) };
        captured?;
        let exec = exec?;
        unsafe { self.library.cuda_graph_launch(exec, self.stream)? };
        self.graphs.borrow_mut().insert(key, GraphExec(exec, self.library));
        Ok(())
    }

    /// Per layer, the weights a decode step reads after its routed experts
    /// are out, in read order: the next layer's attention (E4M3 weights and
    /// their scales), post-attention norm, router and shared expert (or dense
    /// MLP); after the last layer the final norm and head.
    pub fn decode_read_order(&self) -> Vec<Vec<crate::shared::l2_prefetch::Range>> {
        let layers = &self.weights.layers;
        (0..layers.len()).map(|i| match layers.get(i + 1) {
            Some(next) => crate::shared::l2_prefetch::operands(&["input_norm", "w_qkv_a_fp8", "w_qkv_a_scale",
                "q_a_norm", "kv_a_norm", "w_q_b_fp8", "w_q_b_scale", "w_iq_fp8", "w_iq_scale", "w_ik", "k_norm_w",
                "k_norm_b", "w_uk_fp8", "w_uk_scale", "w_uv_fp8", "w_uv_scale", "w_o_fp8", "w_o_scale", "post_norm",
                "gate", "gate.bias", "w_gate_up_fp8", "w_gate_up_scale", "w_down_fp8", "w_down_scale"],
                |n| next.range(n)),
            None => [&self.weights.norm, &self.weights.head].iter()
                .map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes)).collect(),
        }).collect()
    }

    /// After layer `index`'s shared expert is queued in a one-lane decode or
    /// verify step: the L2 prefetch of what the step reads next.
    fn prefetch(&self, index: usize) -> Result<()> {
        match &self.l2 {
            Some(l2) => l2.issue(self.library, index, self.stream),
            None => Ok(()),
        }
    }

    /// Router, shared expert and the Spark routed experts; leaves
    /// routed + shared in `delta` for the next norm's residual add.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn moe(&self, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, t: usize, cap: &str, decode: bool,
        transport: &mut SparkLink<'_>, runtime: &tokio::runtime::Runtime) -> Result<()> {
        self.moe_front(w, layer, t)?;
        let ranks = self.moe_exchange(w, index, layer, t, cap, decode, transport, runtime)?;
        self.reduce(transport.intake.pointers(), w, ranks, t)
    }

    /// Router scores, the sigmoid top-k selection and the wire rows (device only).
    fn moe_front(&self, w: &Workspace<'_>, layer: &GlmLayer<'_>, t: usize) -> Result<()> {
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let rows = Scalar::I32(t as i32);
        self.run("glm_router_scores", &[("x", w.x.buffer.ptr), ("w", layer.ptr("gate")?),
            ("logits", w.router_logits.buffer.ptr)], &[rows])?;
        // SAFETY: logits, bias and route outputs are live buffers of `t` rows.
        unsafe {
            self.library.router_select(w.router_logits.buffer.ptr, layer.ptr("gate.bias")?, std::ptr::null(),
                std::ptr::null(), w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, t, self.cfg.experts, topk,
                self.cfg.routed_scale as f32, true, self.stream)?;
        }
        let grid = (t * h.div_ceil(256)).div_ceil(8).clamp(1, 4 * 188);
        self.run("glm_expert_input_quant", &[("source_ptr", w.x.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
            // SAFETY: the scale rows follow the payload inside each wire row.
            ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
            ("scale_mma_ptr", w.delta.buffer.ptr)], &[rows, Scalar::I32(grid as i32)])
    }

    /// Routes and wire rows down, the shared expert, the Spark exchange and
    /// the rank planes' uploads; returns the rank count the reduce needs.
    /// The next exchange's download sync completes the uploads before the
    /// staging is rewritten.
    #[allow(clippy::too_many_arguments)]
    fn moe_exchange(&self, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, t: usize, cap: &str, decode: bool,
        transport: &mut SparkLink<'_>, runtime: &tokio::runtime::Runtime) -> Result<usize> {
        self.moe_stage(w, layer, t, cap)?;
        if decode {
            self.prefetch(index)?;
        }
        let wave = self.moe_send(w, index, t, decode, transport)?;
        runtime.block_on(self.moe_land(t, transport, wave))
    }

    /// [`Self::moe_exchange`] without the Spark request (benchmarks): the
    /// stage, then zero partials of [`SKIP_RANKS`] ranks uploaded through
    /// the skip intake's pinned staging (a host-mode wave's cost).
    /// With CUTEAFD_EMULATE_EXCHANGE_US a decode step waits that long after
    /// the shared expert (and the L2 prefetch) as a Spark exchange would.
    fn moe_skip(&self, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, t: usize, cap: &str) -> Result<usize> {
        let skip = self.skip.as_ref().context("routed experts are not skipped")?;
        self.moe_stage(w, layer, t, cap)?;
        if cap == "m64" {
            let mark = crate::shared::l2_prefetch::exchange_mark(self.library, self.stream)?;
            self.prefetch(index)?;
            crate::shared::l2_prefetch::exchange_wait(self.library, mark)?;
        }
        skip.upload_zeros(t, self.stream)?;
        Ok(SKIP_RANKS)
    }

    /// Routes and wire rows down to this workspace's pinned staging (the host
    /// waits for them), then the shared expert queued on the GPU; send the
    /// request with [`Self::moe_send`].
    fn moe_stage(&self, w: &Workspace<'_>, layer: &GlmLayer<'_>, t: usize, cap: &str) -> Result<()> {
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
        // SAFETY: the pinned regions are large enough; the sync completes them
        // (and this workspace's previous plane uploads, freeing its staging).
        unsafe {
            self.library.copy_d2h_host_buffer_async(at(0), w.route_ids.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(route_bytes), w.route_weights.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(2 * route_bytes), w.wire.buffer, wire_bytes, self.stream)?;
            self.library.cuda_stream_synchronize(self.stream)?;
        }
        self.profile.borrow_mut()[0] += timer.elapsed().as_secs_f64();
        // The shared expert runs on the GPU while the host sends and the Sparks compute.
        self.ffn(w, layer, self.cfg.moe_intermediate, cap, w.shared.buffer.ptr, Scalar::I32(t as i32))
    }

    /// One request to every Spark rank from the staged routes and wire rows
    /// ([`Self::moe_stage`]); complete it with [`Self::moe_land`].
    fn moe_send(&self, w: &Workspace<'_>, index: usize, t: usize, decode: bool, transport: &mut SparkLink<'_>)
        -> Result<SparkExpertWave> {
        let timer = std::time::Instant::now();
        let request = {
            let staging = w.router_host.borrow();
            expert_request(staging.bytes(), index, t, self.cfg.hidden, self.cfg.topk, decode)?
        };
        self.exchange_host.borrow_mut()[2] += timer.elapsed().as_secs_f64();
        let post = std::time::Instant::now();
        let wave = transport.dispatch(&request)?;
        self.exchange_host.borrow_mut()[3] += post.elapsed().as_secs_f64();
        self.exchange_host.borrow_mut()[0] += timer.elapsed().as_secs_f64();
        Ok(wave)
    }

    /// Hands a prefill unit's wave to its lane thread: the request is built
    /// from the staged routes and wire rows ([`Self::moe_stage`]), posted, and
    /// its rank partials gathered into the lane's intake, all off this
    /// thread; [`Self::lane_land`] waits for it.
    fn lane_send(&self, w: &Workspace<'_>, index: usize, t: usize, lane: &mut SparkLane<'_>) -> Result<()> {
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let ranks = lane.world_size();
        ensure!(ranks <= MAX_RANKS, "{ranks} Spark ranks exceed the reduction planes");
        let staged_bytes = t * topk * 8 + t * (h + h / 32);
        let staged = w.router_host.borrow().buffer;
        ensure!(staged_bytes <= staged.bytes, "lane wave exceeds its staging");
        // The address crosses to the lane thread as an integer; see the SAFETY note below.
        let staged = staged.ptr as usize;
        let build = Box::new(move || {
            // SAFETY: the pinned router staging holds this unit's routes and
            // wire rows (the stage synchronized their copies) and nothing
            // writes it until this lane's next stage, which follows the wait
            // for this wave.
            let staged = unsafe { std::slice::from_raw_parts(staged as *const u8, staged_bytes) };
            expert_request(staged, index, t, h, topk, false)
        });
        lane.submit(t, build)
    }

    /// Waits for a lane's wave and orders the stream after its intake;
    /// returns the rank count the reduce needs.
    fn lane_land(&self, t: usize, lane: &mut SparkLane<'_>) -> Result<usize> {
        let timer = std::time::Instant::now();
        let times = lane.wait(t, std::time::Duration::from_secs(120), self.stream)?;
        self.profile.borrow_mut()[1] += timer.elapsed().as_secs_f64();
        self.exchange_host.borrow_mut()[0] += times.build_post;
        Ok(lane.world_size())
    }

    /// Receives `wave`'s BF16 rank partials into the transport's intake
    /// planes; returns the rank count the reduce needs.
    async fn moe_land(&self, t: usize, transport: &mut SparkLink<'_>, wave: SparkExpertWave) -> Result<usize> {
        let ranks = transport.world_size();
        ensure!(ranks <= MAX_RANKS, "{ranks} Spark ranks exceed the reduction planes");
        let timer = std::time::Instant::now();
        transport.receive(wave, t, self.stream).await?;
        self.profile.borrow_mut()[1] += timer.elapsed().as_secs_f64();
        Ok(ranks)
    }

    /// Pipelined Spark prefill of consecutive-row lanes of one sequence, each
    /// with its own workspace and transport thread. Units (layer, lane) run
    /// layer-major, so a lane's attention follows its own previous layer and
    /// the earlier lanes' same layer in stream order: later lanes' MLA and
    /// DSA index top-k read earlier lanes' latent and index records from the
    /// caches, and each lane's shared-indexer layers reuse the `indices` its
    /// own workspace kept from its last full layer. A lane's request is built,
    /// posted and received on its thread while this thread queues the other
    /// lanes' GPU layers. Returns the last `logit_rows` rows' logits.
    fn step_lanes(&self, lanes: &[StepTables], chunks: &[&[u32]], logit_rows: usize) -> Result<Option<StepLogits>> {
        {
            let mut slots = self.lane_workspaces.borrow_mut();
            while slots.len() < lanes.len() {
                slots.push(self.workspace(self.prefill_rows, false)?);
            }
        }
        let owned = self.lane_workspaces.borrow();
        let workspaces: Vec<&Workspace<'_>> = owned.iter().collect();
        let mut transports = self.lanes.borrow_mut();
        let planes: Vec<[*const u16; 6]> = transports.iter().map(|lane| lane.intake.pointers()).collect();
        ensure!(transports.len() >= lanes.len(), "{} lanes need as many lane transports", lanes.len());
        let counts: Vec<usize> = lanes.iter().map(|l| l.positions.len()).collect();
        let starts: Vec<usize> = counts.iter().scan(0, |first, &n| {
            let here = *first;
            *first += n;
            Some(here)
        }).collect();
        let total: usize = counts.iter().sum();
        ensure!(logit_rows <= total && (self.full_prefill_logits || logit_rows <= DECODE_ROWS),
            "prefill logits past {DECODE_ROWS} rows need full_prefill_logits");
        for ((tables, tokens), w) in lanes.iter().zip(chunks).zip(&workspaces) {
            ensure!(tables.positions.len() <= w.rows && tokens.len() == tables.positions.len(),
                "lane exceeds its workspace");
            self.put(&w.positions, &tables.positions)?;
            self.put(&w.slots, &tables.slots)?;
            self.put(&w.page_table, &tables.page_table)?;
            self.put(&w.cache_lengths, &tables.cache_lengths)?;
            self.put(&w.lengths, &tables.lengths)?;
            self.embedding.embed(tokens, w.ids.buffer, 1, w.h.buffer, self.stream)?;
        }
        let layers = &self.weights.layers;
        let cap = "m4096";
        let rows_of = |lane: usize| Scalar::I32(counts[lane] as i32);
        // The drafter taps the chunk's last TAP_ROWS rows: each lane's part of
        // that window, at its offset among the tap rows.
        let window = total - total.min(super::dflash::TAP_ROWS);
        let attention = |(index, lane): (usize, usize)| -> Result<()> {
            let (w, layer, rows) = (workspaces[lane], &layers[index], rows_of(lane));
            if index == 0 {
                self.norm(w, layer, "input_norm", 0, rows)?;
            }
            self.attention(w, index, layer, rows, cap, &lanes[lane])?;
            self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                ("delta1", w.delta.buffer.ptr), ("weight", layer.ptr("post_norm")?), ("out", w.x.buffer.ptr)],
                &[rows, Scalar::I32(1)])?;
            if layer.dense {
                self.ffn(w, layer, self.cfg.dense_intermediate, cap, w.delta.buffer.ptr, rows)
            } else {
                self.moe_front(w, layer, counts[lane])
            }
        };
        // h += ffn (routed partials reduced first); x = the next input norm.
        let post = |(index, lane): (usize, usize), ranks: Option<usize>| -> Result<()> {
            let (w, rows) = (workspaces[lane], rows_of(lane));
            if let Some(ranks) = ranks {
                self.reduce(planes[lane], w, ranks, counts[lane])?;
            }
            let weight = match layers.get(index + 1) {
                Some(next) => next.ptr("input_norm")?,
                None => self.weights.norm.buffer.ptr,
            };
            self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                ("delta1", w.delta.buffer.ptr), ("weight", weight), ("out", w.x.buffer.ptr)],
                &[rows, Scalar::I32(1)])?;
            if let Some(drafter) = &self.drafter {
                let begin = starts[lane];
                let from = begin.max(window);
                if from < begin + counts[lane] {
                    drafter.tap_at(index, w.h.buffer.ptr, from - begin, begin + counts[lane] - from, from - window)?;
                }
            }
            Ok(())
        };
        let count = lanes.len();
        let units: Vec<(usize, usize)> = (0..layers.len()).flat_map(|l| (0..count).map(move |k| (l, k))).collect();
        // Waves in flight, oldest first. A unit's request goes out as soon as
        // its routes are down; waves land (and post) only when the next
        // unit's attention needs its lane's previous layer, so up to
        // `lanes - 1` waves overlap the GPU's other lanes.
        let mut inflight: std::collections::VecDeque<(usize, usize)> = std::collections::VecDeque::new();
        let land = |unit: (usize, usize), transports: &mut Vec<SparkLane<'_>>| -> Result<()> {
            let ranks = self.lane_land(counts[unit.1], &mut transports[unit.1])?;
            post(unit, Some(ranks))
        };
        attention(units[0])?;
        for (position, &unit) in units.iter().enumerate() {
            let (index, lane) = unit;
            if layers[index].dense {
                post(unit, None)?;
            } else {
                self.moe_stage(workspaces[lane], &layers[index], counts[lane], cap)?;
                self.lane_send(workspaces[lane], index, counts[lane], &mut transports[lane])?;
                inflight.push_back(unit);
            }
            match units.get(position + 1).copied() {
                Some(next) => {
                    while inflight.iter().any(|u| u.1 == next.1) {
                        let oldest = inflight.pop_front().context("in-flight waves")?;
                        land(oldest, &mut transports)?;
                    }
                    attention(next)?;
                }
                None => {
                    while let Some(oldest) = inflight.pop_front() {
                        land(oldest, &mut transports)?;
                    }
                }
            }
        }
        if logit_rows == 0 {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        let last = counts.len() - 1;
        if logit_rows <= counts[last] {
            // The wanted rows are the last lane's: they stay on the device.
            self.launch_head(workspaces[last], counts[last], logit_rows)?;
            return Ok(Some(StepLogits::Device(self.device_logits(workspaces[last], logit_rows, false))));
        }
        let mut logits = Vec::with_capacity(logit_rows * self.cfg.vocab_size);
        for ((w, &t), &begin) in workspaces.iter().zip(&counts).zip(&starts) {
            // Rows of this lane inside the last `logit_rows` of the chunk.
            let wanted = (begin + t).saturating_sub((total - logit_rows).max(begin));
            if wanted > 0 {
                self.launch_head(w, t, wanted)?;
                logits.extend(self.device_logits(w, wanted, false).to_host(self.library)?);
            }
        }
        Ok(Some(StepLogits::Host(logits)))
    }

    /// A transport's rank planes + shared expert into `delta`. The next
    /// wave on that transport is dispatched only after its stage's stream
    /// sync, which follows this reduce.
    fn reduce(&self, planes: [*const u16; 6], w: &Workspace<'_>, ranks: usize, t: usize) -> Result<()> {
        // SAFETY: the transport's intake planes, shared and delta are live
        // [t, h] BF16 buffers ordered after the wave's intake.
        unsafe {
            self.library.v41_compact_reducer()?.reduce_planes(planes, ranks as u32, w.shared.buffer.ptr.cast(),
                w.delta.buffer.ptr.cast(), t as u32, self.stream)
        }
    }

    fn norm(&self, w: &Workspace<'_>, layer: &GlmLayer<'_>, weight: &str, deltas: i32, rows: Scalar) -> Result<()> {
        self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
            ("delta1", w.delta.buffer.ptr), ("weight", layer.ptr(weight)?), ("out", w.x.buffer.ptr)],
            &[rows, Scalar::I32(deltas)])
    }

    fn attention(&self, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, rows: Scalar, cap: &str,
        tables: &StepTables) -> Result<()> {
        let mode = if tables.decode { "decode" } else { "prefill" };
        let kv = &self.kv[index];
        let mut producer = vec![("x", w.x.buffer.ptr), ("positions", w.positions.buffer.ptr),
            ("kv_slots", w.slots.buffer.ptr), ("cos_sin", self.cos_sin.buffer.ptr)];
        producer.extend(weight(layer, "w_qkv_a", cap)?);
        producer.extend([("q_a_norm", layer.ptr("q_a_norm")?), ("kv_a_norm", layer.ptr("kv_a_norm")?)]);
        producer.extend(weight(layer, "w_q_b", cap)?);
        producer.extend(weight(layer, "w_uk", cap)?);
        producer.extend([("kv_cache", kv.buffer.ptr), ("query", w.query.buffer.ptr),
            ("q_resid", w.q_resid.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
        let scalars = self.projection_scalars(rows, cap);
        self.run(&format!("glm_producer_{cap}"), &producer, &scalars)?;
        if let Some(index_cache) = &self.index[index] {
            let mut pointers = vec![("x", w.x.buffer.ptr), ("q_resid", w.q_resid.buffer.ptr),
                ("positions", w.positions.buffer.ptr), ("index_slots", w.slots.buffer.ptr),
                ("cos_sin", self.cos_sin.buffer.ptr)];
            pointers.extend(weight(layer, "w_iq", cap)?);
            pointers.extend([("w_ik", layer.ptr("w_ik")?), ("k_norm_w", layer.ptr("k_norm_w")?),
                ("k_norm_b", layer.ptr("k_norm_b")?), ("index_cache", index_cache.buffer.ptr),
                ("q_fp8", w.q_fp8.buffer.ptr), ("head_weights", w.head_weights.buffer.ptr),
                ("scratch", w.scratch.buffer.ptr)]);
            self.run(&format!("glm_index_producer_{cap}"), &pointers, &scalars)?;
            self.run(&format!("glm_index_topk_{mode}_{cap}"), &[
                ("q_fp8", w.q_fp8.buffer.ptr), ("weights", w.head_weights.buffer.ptr),
                ("index_k_cache", index_cache.buffer.ptr), ("page_table", w.page_table.buffer.ptr),
                ("cache_lengths", w.cache_lengths.buffer.ptr), ("output_indices", w.indices.buffer.ptr),
                ("scratch", w.topk_scratch.buffer.ptr)],
                &[rows, Scalar::I32(tables.table_width as i32), Scalar::I32(tables.table_stride as i32)])?;
        }
        // Shared-indexer layers read the previous full layer's `indices`.
        if !tables.decode && native_mla_prefill() {
            let scale = ((self.cfg.qk_nope_head_dim + self.cfg.qk_rope_head_dim) as f32).powf(-0.5);
            // SAFETY: query, cache, indices, lengths and the attention output are
            // live buffers of the step's rows on the engine stream.
            unsafe {
                self.library.glm_mla_prefill(w.query.buffer.ptr, kv.buffer.ptr, w.indices.buffer.ptr,
                    w.lengths.buffer.ptr, w.attn.buffer.ptr, tables.positions.len(), self.cfg.heads,
                    self.cfg.index_topk, 656, scale * std::f32::consts::LOG2_E, self.stream)?;
            }
        } else {
            self.run(&format!("glm_sparse_mla_{mode}_{cap}"), &[
                ("q", w.query.buffer.ptr), ("kv_cache", kv.buffer.ptr), ("indices", w.indices.buffer.ptr),
                ("lengths", w.lengths.buffer.ptr), ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
                &[rows])?;
        }
        let mut o = vec![("attn", w.attn.buffer.ptr)];
        o.extend(weight(layer, "w_uv", cap)?);
        o.extend(weight(layer, "w_o", cap)?);
        o.extend([("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
        self.run(&format!("glm_o_{cap}"), &o, &scalars)
    }

    /// Scalars of the programs over FP8 weights: `rows`, plus the prefill
    /// programs' `fp8_rows` (1: W8A8, 0: W8A16).
    fn projection_scalars(&self, rows: Scalar, cap: &str) -> Vec<Scalar> {
        if cap == "m64" {
            vec![rows]
        } else {
            vec![rows, Scalar::I32(i32::from(self.prefill_w8a8))]
        }
    }

    /// SwiGLU MLP (dense layers, or the shared expert) into `out`.
    fn ffn(&self, w: &Workspace<'_>, layer: &GlmLayer<'_>, intermediate: usize, cap: &str, out: *mut c_void,
        rows: Scalar) -> Result<()> {
        let mut pointers = vec![("x", w.x.buffer.ptr)];
        pointers.extend(weight(layer, "w_gate_up", cap)?);
        pointers.extend(weight(layer, "w_down", cap)?);
        pointers.extend([("out", out), ("scratch", w.scratch.buffer.ptr)]);
        self.run(&format!("glm_ffn_i{intermediate}_{cap}"), &pointers, &self.projection_scalars(rows, cap))
    }
}

/// Prefill sparse MLA on the native F16 kernel (glm_mla_prefill.cu) unless
/// CUTEAFD_MLA_PREFILL=b12x selects the b12x program.
pub(crate) fn native_mla_prefill() -> bool {
    static NATIVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *NATIVE.get_or_init(|| std::env::var("CUTEAFD_MLA_PREFILL").map_or(true, |v| v != "b12x"))
}

/// Prefill lanes: CUTEAFD_GLM_PREFILL_LANES (1 = serial), default [`DEFAULT_LANES`].
pub(crate) fn configured_lanes() -> usize {
    std::env::var("CUTEAFD_GLM_PREFILL_LANES").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_LANES)
        .clamp(1, PREFILL_LANES)
}

/// The expert request of `t` staged rows: expert ids (U32) and gate
/// weights (F32) `[t, topk]`, then the FP8 wire rows, as `moe_stage` lays
/// them out in the pinned router staging.
fn expert_request(staged: &[u8], index: usize, t: usize, h: usize, topk: usize, decode: bool)
    -> Result<ExpertProtocolV2Request> {
    let kind = if decode { ExpertV2SourceKind::Decode } else { ExpertV2SourceKind::Prefill };
    let (route_bytes, wire_bytes) = (t * topk * 4, t * (h + h / 32));
    ensure!(staged.len() >= 2 * route_bytes + wire_bytes, "staged routes and wire rows are short");
    let word = |offset: usize, i: usize| u32::from_le_bytes(staged[offset + i * 4..][..4].try_into().unwrap());
    let routes = (0..t * topk).map(|i| ExpertProtocolV2RouteEntry {
        row_index: (i / topk) as u32, expert_id: word(0, i), gate_weight: f32::from_bits(word(route_bytes, i)),
    }).collect();
    let mut wire = vec![0u8; wire_bytes];
    copy_parallel(&mut wire, &staged[2 * route_bytes..2 * route_bytes + wire_bytes]);
    let mut request = ExpertProtocolV2Request::new(index as u64 + 1, 17, index as u32, h as u32,
        ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        (0..t as u32).map(|row| ExpertProtocolV2RowDescriptor {
            row_id: u64::from(row), source_kind: kind, source_request_id: 1,
            token_position: u64::from(row), route_offset: row * topk as u32, route_count: topk as u32,
        }).collect(),
        routes, wire)?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

/// An FP8 weight's program pointers: its E4M3 copy and FP32 scales (prefill
/// programs read `w_qkv_a`'s per-row scales, K-block major).
fn weight(layer: &GlmLayer<'_>, name: &'static str, cap: &str) -> Result<Vec<(&'static str, *mut c_void)>> {
    let (fp8, scale, kscale) = super::weights::fp8_operand_names(name);
    let scale = if cap != "m64" && !kscale.is_empty() { kscale } else { scale };
    Ok(vec![(fp8, layer.ptr(fp8)?), (scale, layer.ptr(scale)?)])
}
