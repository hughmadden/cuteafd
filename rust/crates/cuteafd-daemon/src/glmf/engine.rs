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
//! slot pools, one slot per sequence shared by every KDA layer; MLA layers
//! keep FP8 528-byte latent records in 64-row pages, and their DSA indexer
//! keeps per-token BF16 keys and gates beside the records and one FP8 key
//! per completed 4-token pool in pool pages (64 pools per page). The
//! indexer selects every earlier token up to 2051 tokens; past that the top
//! 512 pools (glmf_index_topk) expand to tokens plus the open tail pool.
//!
//! FFN: dense SwiGLU (clamped at 10), or the MoE: FP32 router logits, the
//! native sigmoid top-8 select, the shared expert, and routed experts from
//! the checkpoint's FP8 on this GPU (the `fp8-glmf` TP1 package, a window of
//! resident layers reloaded as the step walks the layers) or on the Sparks.
use super::weights::{GlmfLayer, GlmfWeights};
use crate::v41_experts::fp8::{Fp8Experts, Fp8Layer};
use crate::v41_memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::dsv4::{Dsv4Programs, Dsv4Scalar, VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::fp8_experts::Fp8ExpertTensors;
use cuteafd_loader::glm_next::{GlmNextAttention, GlmNextConfig};
use cuteafd_transport::v41_expert::{V41Tp4Roce, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
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
/// FP8 latent record bytes (512 E4M3 + 4 FP32 group scales).
const RECORD_BYTES: usize = 528;
const MAX_RANKS: usize = 6;
const HC: usize = 4;

/// Routed experts on this GPU from the TP1 FP8 package: a window of
/// resident layers, reloaded from the checkpoint when a step reaches a layer
/// outside it (layer-major prefill loads each layer once per step).
pub(crate) struct LocalExperts<'a> {
    pub library: &'a NativeLibrary,
    pub tensors: &'a Fp8ExpertTensors,
    pub experts: RefCell<Fp8Experts<'a>>,
    /// Most layers resident at once.
    pub window: usize,
    pub loads: RefCell<usize>,
}

impl LocalExperts<'_> {
    fn index_of(&self, layer: usize) -> Result<usize> {
        let mut experts = self.experts.borrow_mut();
        if let Ok(index) = experts.index_of(layer) {
            return Ok(index);
        }
        if experts.layers.len() >= self.window {
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
    pub resident: RefCell<Option<(std::ops::Range<usize>, crate::dsv4::local::LocalExperts<'a>)>>,
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
        let local = crate::dsv4::local::LocalExperts::load_range(self.library, &self.native_lib, self.catalog, 0,
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
    /// Spark ranks serving the `fp8` family over RoCE (one BF16 partial per rank).
    Spark { transport: RefCell<V41Tp4Roce>, runtime: tokio::runtime::Runtime },
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
}

/// Tokens per DSA index pool, and pools per pool-cache page.
const KPOOL: usize = 4;
const POOL_PAGE_TOKENS: usize = KPOOL * PAGE_ROWS;

/// A sequence's MLA pages, its pool pages, its KDA state slot and its length.
#[derive(Debug, Clone)]
pub(crate) struct GlmfPlacement {
    pub pages: Vec<i32>,
    pub pool_pages: Vec<i32>,
    pub slot: i32,
    pub len: usize,
}

impl GlmfPlacement {
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

/// Free MLA pages, pool pages and KDA state slots.
pub(crate) struct Allocator {
    pages: Vec<i32>,
    pool_pages: Vec<i32>,
    slots: Vec<i32>,
}

impl Allocator {
    pub fn new(pages: usize, slots: usize) -> Self {
        let pool_pages = pages.div_ceil(KPOOL);
        Self { pages: (0..pages as i32).rev().collect(), pool_pages: (0..pool_pages as i32).rev().collect(),
            slots: (0..slots as i32).rev().collect() }
    }

    /// Reserves every page a sequence of up to `capacity` tokens needs and a
    /// state slot (the engine zeroes the slot and maps the pool pages before
    /// the first step).
    pub fn admit(&mut self, capacity: usize) -> Result<GlmfPlacement> {
        let pages = capacity.div_ceil(PAGE_ROWS).max(1);
        let pool_pages = capacity.div_ceil(POOL_PAGE_TOKENS).max(1);
        ensure!(self.pages.len() >= pages && self.pool_pages.len() >= pool_pages,
            "cache pages exhausted ({pages} + {pool_pages} pool pages needed, {} + {} free)", self.pages.len(),
            self.pool_pages.len());
        let slot = self.slots.pop().context("KDA state slots exhausted")?;
        Ok(GlmfPlacement {
            pages: (0..pages).map(|_| self.pages.pop().unwrap()).collect(),
            pool_pages: (0..pool_pages).map(|_| self.pool_pages.pop().unwrap()).collect(),
            slot,
            len: 0,
        })
    }

    /// A spare KDA state slot (a speculative verify's backup).
    pub fn spare_slot(&mut self) -> Result<i32> {
        self.slots.pop().context("KDA state slots exhausted")
    }

    pub fn release_slot(&mut self, slot: i32) {
        self.slots.push(slot);
    }

    pub fn release(&mut self, placement: GlmfPlacement) {
        self.pages.extend(placement.pages);
        self.pool_pages.extend(placement.pool_pages);
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
    router_logits: Dev<'a>,
    route_ids: Dev<'a>,
    route_weights: Dev<'a>,
    wire: Dev<'a>,
    planes: Vec<Dev<'a>>,
    router_host: RefCell<HostAllocation<'a>>,
    planes_host: RefCell<HostAllocation<'a>>,
    head: VocabularyHead<'a>,
    _head_workspace: Dev<'a>,
}

pub(crate) struct GlmfEngine<'a> {
    pub library: &'a NativeLibrary,
    pub programs: &'a Dsv4Programs<'a>,
    pub cfg: GlmNextConfig,
    pub weights: GlmfWeights<'a>,
    pub stream: *mut c_void,
    pub max_context: usize,
    pub prefill_rows: usize,
    pub pages: usize,
    pub slots: usize,
    /// Per layer: the latent record pool (MLA) or the conv state pool (KDA).
    kv: Vec<Dev<'a>>,
    /// Per KDA layer (None for MLA): the FP32 recurrent state pool.
    state: Vec<Option<Dev<'a>>>,
    /// Per MLA layer (None for KDA): per-token indexer keys | gates (BF16
    /// [record slots, 256]) and the FP8 pool-key cache.
    index: Vec<Option<(Dev<'a>, Dev<'a>)>>,
    /// Logical page of each pool-cache page within its sequence.
    pool_logical: Dev<'a>,
    pub pool_pages: usize,
    workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    experts: Option<Experts<'a>>,
    /// Host seconds: GPU wait before expert exchanges, the exchanges.
    pub profile: RefCell<[f64; 2]>,
    /// Captured decode segments (CUTEAFD_GLMF_GRAPHS=0 runs decode eagerly).
    graphs: RefCell<std::collections::HashMap<GraphKey, GraphExec<'a>>>,
    use_graphs: bool,
}

/// What a captured decode segment baked in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GraphKey {
    segment: usize,
    rows: usize,
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
    pub fn new(library: &'a NativeLibrary, programs: &'a Dsv4Programs<'a>, cfg: GlmNextConfig, weights: GlmfWeights<'a>,
        stream: *mut c_void, max_context: usize, prefill_rows: usize, pages: usize, slots: usize) -> Result<Self> {
        ensure!(cfg.hc_mult == HC && cfg.kv_lora_rank == 512 && cfg.kda_head_dim == 128 && cfg.heads == 64,
            "the glmf programs are built for 4 mHC streams, a 512 latent, 64 MLA heads and 128-wide KDA heads");
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let d = cfg.kda_heads * cfg.kda_head_dim;
        let pool_pages = pages.div_ceil(KPOOL);
        let mut kv = Vec::new();
        let mut state = Vec::new();
        let mut index = Vec::new();
        for layer in &weights.layers {
            match layer.attention {
                GlmNextAttention::Mla => {
                    kv.push(zeroed(pages * PAGE_ROWS * RECORD_BYTES)?);
                    state.push(None);
                    index.push(Some((zeroed(pages * PAGE_ROWS * 512)?, zeroed(pool_pages * PAGE_ROWS * 132)?)));
                }
                GlmNextAttention::Kda => {
                    kv.push(zeroed(slots * 3 * 3 * d * 2)?);
                    state.push(Some(zeroed(slots * d * cfg.kda_head_dim * 4)?));
                    index.push(None);
                }
            }
        }
        let pool_logical = zeroed(pool_pages * 4)?;
        Ok(Self { library, programs, cfg, weights, stream, max_context, prefill_rows, pages, slots, kv, state, index,
            pool_logical, pool_pages, workspace: RefCell::new(None), decode_workspace: RefCell::new(None),
            experts: None, profile: RefCell::new([0.0; 2]), graphs: RefCell::new(std::collections::HashMap::new()),
            use_graphs: std::env::var("CUTEAFD_GLMF_GRAPHS").map_or(true, |v| v != "0") })
    }

    /// Serves MoE layers from `experts` (without, the engine stops at the first MoE layer).
    pub fn set_experts(&mut self, experts: Experts<'a>) {
        self.experts = Some(experts);
    }

    pub fn experts(&self) -> Option<&Experts<'a>> {
        self.experts.as_ref()
    }

    /// Before a sequence's first step: zeroes its KDA state and maps its pool pages.
    fn start(&self, placement: &GlmfPlacement) -> Result<()> {
        for (logical, &page) in placement.pool_pages.iter().enumerate() {
            let page = usize::try_from(page)?;
            ensure!(page < self.pool_pages, "pool page {page} out of range");
            let at = cuteafd_ffi::CuteafdDeviceBuffer {
                // SAFETY: page < pool_pages, so the entry lies inside the table.
                ptr: unsafe { self.pool_logical.buffer.ptr.cast::<u8>().add(page * 4) }.cast(),
                bytes: 4,
                ..self.pool_logical.buffer
            };
            self.library.copy_h2d(at, &(logical as i32).to_le_bytes())?;
        }
        self.reset_slot(placement.slot)
    }

    /// Copies a sequence's KDA recurrent and conv state from slot `from` to
    /// slot `to` on the engine stream (the backup a speculative verify
    /// restores before replaying its accepted rows).
    pub fn copy_slot(&self, from: i32, to: i32) -> Result<()> {
        let (from, to) = (usize::try_from(from)?, usize::try_from(to)?);
        ensure!(from < self.slots && to < self.slots && from != to, "KDA slots {from} -> {to} out of range");
        for (kv, state) in self.kv.iter().zip(&self.state) {
            if let Some(state) = state {
                for pool in [kv, state] {
                    let per = pool.buffer.bytes / self.slots;
                    let at = |slot: usize| cuteafd_ffi::CuteafdDeviceBuffer {
                        // SAFETY: slot < slots, so the slot's bytes lie inside the pool.
                        ptr: unsafe { pool.buffer.ptr.cast::<u8>().add(slot * per) }.cast(),
                        bytes: per,
                        ..pool.buffer
                    };
                    // SAFETY: both slot regions are live and disjoint; the stream orders the copy.
                    unsafe { self.library.copy_d2d_async(at(to), at(from), per, self.stream)? };
                }
            }
        }
        Ok(())
    }

    /// Zeroes a sequence's KDA recurrent and conv state (before its first step).
    pub fn reset_slot(&self, slot: i32) -> Result<()> {
        let slot = usize::try_from(slot)?;
        ensure!(slot < self.slots, "KDA slot {slot} out of range");
        for (kv, state) in self.kv.iter().zip(&self.state) {
            if let Some(state) = state {
                for pool in [kv, state] {
                    let per = pool.buffer.bytes / self.slots;
                    let at = cuteafd_ffi::CuteafdDeviceBuffer {
                        // SAFETY: slot < slots, so the slot's bytes lie inside the pool.
                        ptr: unsafe { pool.buffer.ptr.cast::<u8>().add(slot * per) }.cast(),
                        bytes: per,
                        ..pool.buffer
                    };
                    self.library.cuda_zero_bytes(at, per)?;
                }
            }
        }
        Ok(())
    }

    fn alloc(&self, bytes: usize) -> Result<Dev<'a>> {
        DeviceAllocation::new(self.library, bytes.max(256))
    }

    fn run(&self, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Dsv4Scalar]) -> Result<()> {
        let name = format!("glmf_{name}");
        let names: Vec<&str> = pointers.iter().map(|(n, _)| *n).collect();
        let program = self.programs.program(&name, &names)?;
        let raw: Vec<*mut c_void> = pointers.iter().map(|(_, p)| *p).collect();
        // SAFETY: every pointer names a live allocation sized for the rows in
        // `scalars`; the stream orders all launches of this engine.
        unsafe { program.launch(&raw, scalars, self.stream) }.with_context(|| format!("{name} with {scalars:?}"))
    }

    fn scratch(&self, name: &str) -> Result<usize> {
        Ok(self.programs.spec(&format!("glmf_{name}"))?.scratch.get("scratch").copied().unwrap_or(0) as usize)
    }

    fn workspace(&self, t: usize, decode: bool) -> Result<Workspace<'a>> {
        let (h, n, lat) = (self.cfg.hidden, self.cfg.heads, self.cfg.kv_lora_rank);
        let (cap, mode) = if decode { ("m64", "decode") } else { ("m4096", "prefill") };
        let mut scratch = 0;
        for name in [format!("mhc_post_pre_{cap}"), format!("kda_{cap}"), format!("mla_producer_{cap}"),
            format!("sparse_mla_{mode}_{cap}"), format!("o_{cap}"), format!("ffn_i2048_{cap}"),
            format!("ffn_i12288_{cap}"), format!("index_producer_{cap}"), "mhc_pre".into()] {
            scratch = scratch.max(self.scratch(&name)?);
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
            logits: self.alloc(t * self.cfg.vocab_size * 4)?,
            router_logits: self.alloc(t * self.cfg.experts * 4)?,
            route_ids: self.alloc(t * topk * 4)?,
            route_weights: self.alloc(t * topk * 4)?,
            wire: self.alloc(t * (h + h / 32))?,
            planes: if spark { (0..MAX_RANKS).map(|_| self.alloc(t * h * 2)).collect::<Result<_>>()? } else { Vec::new() },
            router_host: RefCell::new(HostAllocation::new(self.library,
                if spark { t * (topk * 8 + h + h / 32) } else { 256 })?),
            planes_host: RefCell::new(HostAllocation::new(self.library, if spark { MAX_RANKS * t * h * 2 } else { 256 })?),
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
    pub fn prefill_forced(&self, placement: &mut GlmfPlacement, embed: &[u8],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>, all_logits: bool) -> Result<Option<Vec<f32>>> {
        let (t, start) = (embed.len() / (self.cfg.hidden * 2), placement.len);
        ensure!(t > 0 && t <= self.prefill_rows && start + t <= self.max_context, "prefill of {t} rows at {start}");
        if start == 0 {
            self.start(placement)?;
        }
        let mut tables = StepTables { page_table: placement.pages.clone(), pool_table: placement.pool_pages.clone(),
            ..Default::default() };
        self.rows(placement, start..start + t, 0, &mut tables)?;
        let logits = self.step(&tables, embed, if all_logits { t } else { 1 }, on_layer, forced)?;
        placement.len += t;
        Ok(logits)
    }

    pub fn prefill(&self, placement: &mut GlmfPlacement, embed: &[u8],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        self.prefill_forced(placement, embed, on_layer, None, false)
    }

    /// Appends each sequence's tokens (one for decode, several for a verify)
    /// at its length in one decode-shaped step; returns every row's logits.
    /// KDA state advances in place: a caller rejecting a suffix must replay.
    pub fn verify(&self, sequences: &mut [(&mut GlmfPlacement, usize)], embed: &[u8],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        let rows: usize = sequences.iter().map(|(_, n)| n).sum();
        ensure!(rows > 0 && rows <= DECODE_ROWS && embed.len() == rows * self.cfg.hidden * 2, "decode step of {rows} rows");
        // Power-of-two strides and widths bound the graphs a growing batch captures.
        let page_stride = sequences.iter().map(|(p, _)| p.pages.len()).max().unwrap_or(1).next_power_of_two()
            .min(self.pages);
        let pool_stride = sequences.iter().map(|(p, _)| p.pool_pages.len()).max().unwrap_or(1).next_power_of_two()
            .min(self.pool_pages);
        let mut tables = StepTables { decode: true, page_stride, pool_stride, ..Default::default() };
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
        let logits = self.step(&tables, embed, rows, on_layer, None)?;
        for (placement, count) in sequences.iter_mut() {
            placement.len += *count;
        }
        Ok(logits)
    }

    fn step(&self, tables: &StepTables, embed: &[u8], logit_rows: usize,
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>) -> Result<Option<Vec<f32>>> {
        let (h, t) = (self.cfg.hidden, tables.kv_slots.len());
        let (slot, capacity) = if tables.decode { (&self.decode_workspace, DECODE_ROWS) } else { (&self.workspace, self.prefill_rows) };
        if slot.borrow().is_none() {
            *slot.borrow_mut() = Some(self.workspace(capacity, tables.decode)?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        ensure!(t <= w.rows && logit_rows <= t, "step exceeds the workspace");
        self.put(&w.positions, &tables.positions)?;
        self.put(&w.kv_slots, &tables.kv_slots)?;
        self.put(&w.kda_slots, &tables.kda_slots)?;
        self.put(&w.seq_first, &tables.seq_first)?;
        self.put(&w.pool_slots, &tables.pool_slots)?;
        self.put(&w.cache_lengths, &tables.cache_lengths)?;
        self.put(&w.page_table, &tables.page_table)?;
        self.put(&w.pool_table, &tables.pool_table)?;
        // Streams start as four copies of the embedding.
        let row = h * 2;
        let mut streams = vec![0u8; t * HC * row];
        for (r, e) in embed.chunks_exact(row).enumerate() {
            for s in 0..HC {
                streams[(r * HC + s) * row..][..row].copy_from_slice(e);
            }
        }
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: streams.len(), ..w.streams[0].buffer }, &streams)?;
        let rows = Dsv4Scalar::I32(t as i32);
        if self.use_graphs && tables.decode && on_layer.is_none() && forced.is_none() {
            return self.decode_graphed(w, tables, t, rows, logit_rows);
        }
        let cap = if tables.decode { "m64" } else { "m4096" };
        let layers = &self.weights.layers;
        let mut cur = 0usize;
        self.pre(w, &w.streams[cur], &layers[0], rows)?;
        for (index, layer) in layers.iter().enumerate() {
            match layer.attention {
                GlmNextAttention::Kda => self.kda(w, index, layer, rows, cap)?,
                GlmNextAttention::Mla => self.mla(w, index, layer, rows, cap, tables)?,
            }
            // Attention back into the streams, then the FFN site's collapse + norm.
            self.post_pre(w, cur, layer, "ffn", "post_norm", rows, cap)?;
            cur ^= 1;
            if layer.dense {
                self.run(&format!("ffn_i{}_{cap}", self.cfg.dense_intermediate), &[("x", w.x.buffer.ptr),
                    ("w_gate_up", layer.ptr("w_gate_up")?), ("w_down", layer.ptr("w_down")?),
                    ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])?;
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
        }
        if layers.len() < self.cfg.layers {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        self.run("head", &[("streams", w.streams[cur].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
            ("out", w.x.buffer.ptr)], &[rows])?;
        // SAFETY: the head's input and operands are live buffers of these shapes.
        unsafe {
            w.head.launch(w.x.buffer.ptr.cast::<u8>().add((t - logit_rows) * h * 2).cast(),
                self.weights.head.buffer.ptr.cast(), w.logits.buffer.ptr.cast(), logit_rows as u32, self.stream)?;
        }
        let logits = self.download(&w.logits, logit_rows * self.cfg.vocab_size * 4)?;
        Ok(Some(logits.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()))
    }

    /// A decode step as captured segments: segment `i` posts layer `i - 1`'s
    /// FFN output into the streams with layer `i`'s attention-site collapse,
    /// then runs layer `i` up to its routed experts, which run (local or on
    /// the Sparks) between segments. Streams start and end in buffer 0.
    fn decode_graphed(&self, w: &Workspace<'_>, tables: &StepTables, t: usize, rows: Dsv4Scalar, logit_rows: usize)
        -> Result<Option<Vec<f32>>> {
        let layers = &self.weights.layers;
        for index in 0..=layers.len() {
            let key = GraphKey { segment: index, rows: t, long: tables.long, pool_width: tables.pool_width,
                page_stride: tables.page_stride, pool_stride: tables.pool_stride };
            self.replay(key, || -> Result<()> {
                let Some(layer) = layers.get(index) else {
                    return self.run("mhc_post", &[("x", w.delta.buffer.ptr), ("residual", w.streams[1].buffer.ptr),
                        ("prev_post", w.post.buffer.ptr), ("prev_comb", w.comb.buffer.ptr),
                        ("out", w.streams[0].buffer.ptr)], &[rows]);
                };
                if index == 0 {
                    self.pre(w, &w.streams[0], layer, rows)?;
                } else {
                    self.post_pre(w, 1, layer, "attn", "input_norm", rows, "m64")?;
                }
                match layer.attention {
                    GlmNextAttention::Kda => self.kda(w, index, layer, rows, "m64")?,
                    GlmNextAttention::Mla => self.mla(w, index, layer, rows, "m64", tables)?,
                }
                self.post_pre(w, 0, layer, "ffn", "post_norm", rows, "m64")?;
                if layer.dense {
                    self.run(&format!("ffn_i{}_m64", self.cfg.dense_intermediate), &[("x", w.x.buffer.ptr),
                        ("w_gate_up", layer.ptr("w_gate_up")?), ("w_down", layer.ptr("w_down")?),
                        ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])
                } else {
                    self.moe_front(w, index, layer, t, rows, "m64")
                }
            })?;
            if layers.get(index).is_some_and(|layer| !layer.dense) {
                self.moe_experts(w, index, t, rows, true)?;
            }
        }
        if layers.len() < self.cfg.layers {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        let h = self.cfg.hidden;
        self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
            ("out", w.x.buffer.ptr)], &[rows])?;
        // SAFETY: the head's input and operands are live buffers of these shapes.
        unsafe {
            w.head.launch(w.x.buffer.ptr.cast::<u8>().add((t - logit_rows) * h * 2).cast(),
                self.weights.head.buffer.ptr.cast(), w.logits.buffer.ptr.cast(), logit_rows as u32, self.stream)?;
        }
        let logits = self.download(&w.logits, logit_rows * self.cfg.vocab_size * 4)?;
        Ok(Some(logits.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()))
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
    fn pre(&self, w: &Workspace<'_>, streams: &Dev<'_>, layer: &GlmfLayer<'_>, rows: Dsv4Scalar) -> Result<()> {
        self.run("mhc_pre", &[("residual", streams.buffer.ptr), ("fn", layer.ptr("attn.fn")?),
            ("scale", layer.ptr("attn.scale")?), ("base", layer.ptr("attn.base")?), ("norm", layer.ptr("input_norm")?),
            ("post", w.post.buffer.ptr), ("comb", w.comb.buffer.ptr), ("y", w.x.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])
    }

    /// `delta` back into streams `cur` (into the other buffer), then the
    /// `site` collapse of `layer` normalized by its `norm`.
    #[allow(clippy::too_many_arguments)]
    fn post_pre(&self, w: &Workspace<'_>, cur: usize, layer: &GlmfLayer<'_>, site: &str, norm: &str,
        rows: Dsv4Scalar, cap: &str) -> Result<()> {
        self.run(&format!("mhc_post_pre_{cap}"), &[("x", w.delta.buffer.ptr),
            ("residual", w.streams[cur].buffer.ptr), ("prev_post", w.post.buffer.ptr),
            ("prev_comb", w.comb.buffer.ptr), ("fn", layer.ptr(&format!("{site}.fn"))?),
            ("scale", layer.ptr(&format!("{site}.scale"))?), ("base", layer.ptr(&format!("{site}.base"))?),
            ("norm", layer.ptr(norm)?), ("residual_out", w.streams[cur ^ 1].buffer.ptr),
            ("post", w.post.buffer.ptr), ("comb", w.comb.buffer.ptr), ("y", w.x.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])
    }

    fn kda(&self, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, rows: Dsv4Scalar, cap: &str) -> Result<()> {
        let state = self.state[index].as_ref().context("KDA layer without a state pool")?;
        self.run(&format!("kda_{cap}"), &[("x", w.x.buffer.ptr), ("w_in", layer.ptr("w_in")?),
            ("w_fg", layer.ptr("w_fg")?), ("conv_w", layer.ptr("conv_w")?), ("a_log", layer.ptr("a_log")?),
            ("dt_bias", layer.ptr("dt_bias")?), ("o_norm", layer.ptr("o_norm")?), ("w_o", layer.ptr("w_o")?),
            ("conv_state", self.kv[index].buffer.ptr), ("state", state.buffer.ptr),
            ("slots", w.kda_slots.buffer.ptr), ("seq_first", w.seq_first.buffer.ptr), ("out", w.delta.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])
    }

    fn mla(&self, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, rows: Dsv4Scalar, cap: &str,
        tables: &StepTables) -> Result<()> {
        let mode = if tables.decode { "decode" } else { "prefill" };
        let cache = self.kv[index].buffer.ptr;
        let (keys, pool_cache) = self.index[index].as_ref().context("MLA layer without an index cache")?;
        self.run(&format!("mla_producer_{cap}"), &[("x", w.x.buffer.ptr), ("kv_slots", w.kv_slots.buffer.ptr),
            ("w_qkv_a", layer.ptr("w_qkv_a")?), ("q_a_norm", layer.ptr("q_a_norm")?),
            ("kv_a_norm", layer.ptr("kv_a_norm")?), ("w_q_b", layer.ptr("w_q_b")?), ("w_uk", layer.ptr("w_uk")?),
            ("kv_cache", cache), ("query", w.query.buffer.ptr), ("q_resid", w.q_resid.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])?;
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
                &[rows, Dsv4Scalar::I32(tables.pool_width.max(1) as i32), Dsv4Scalar::I32(tables.pool_stride as i32)])?;
        }
        self.run("index_expand", &[("positions", w.positions.buffer.ptr), ("pools", w.pools.buffer.ptr),
            ("pool_logical", self.pool_logical.buffer.ptr), ("page_table", w.page_table.buffer.ptr),
            ("indices", w.indices.buffer.ptr), ("lengths", w.lengths.buffer.ptr)],
            &[rows, Dsv4Scalar::I32(tables.page_stride as i32)])?;
        self.run(&format!("sparse_mla_{mode}_{cap}"), &[("q", w.query.buffer.ptr), ("kv_cache", cache),
            ("indices", w.indices.buffer.ptr), ("lengths", w.lengths.buffer.ptr), ("out", w.latent.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        self.run(&format!("o_{cap}"), &[("attn", w.latent.buffer.ptr), ("w_uv", layer.ptr("w_uv")?),
            ("w_o", layer.ptr("w_o")?), ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])
    }

    /// Router, shared expert and routed experts; leaves `bf16(routed + shared)` in `delta`.
    #[allow(clippy::too_many_arguments)]
    fn moe(&self, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, t: usize, rows: Dsv4Scalar, cap: &str,
        decode: bool) -> Result<()> {
        self.moe_front(w, index, layer, t, rows, cap)?;
        self.moe_experts(w, index, t, rows, decode)
    }

    /// Router logits, the sigmoid top-8, the shared expert (into `shared`)
    /// and, for wire-fed experts, the FP8 K32 wire rows. No host sync.
    fn moe_front(&self, w: &Workspace<'_>, index: usize, layer: &GlmfLayer<'_>, t: usize, rows: Dsv4Scalar, cap: &str)
        -> Result<()> {
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let experts = self.experts.as_ref().with_context(|| format!(
            "layer {index} is an MoE layer: pass --local-experts (FP8 package) or Spark --peers \
             (run --layers 3 for the dense layers alone)"))?;
        self.run("router_scores", &[("x", w.x.buffer.ptr), ("w", layer.ptr("gate")?),
            ("logits", w.router_logits.buffer.ptr)], &[rows])?;
        // SAFETY: logits, bias and route outputs are live buffers of `t` rows.
        unsafe {
            self.library.router_select(w.router_logits.buffer.ptr, layer.ptr("gate.bias")?, std::ptr::null(),
                std::ptr::null(), w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, t, self.cfg.experts, topk,
                self.cfg.routed_scale as f32, true, self.stream)?;
        }
        self.run(&format!("ffn_i{}_{cap}", self.cfg.moe_intermediate), &[("x", w.x.buffer.ptr),
            ("w_gate_up", layer.ptr("w_gate_up")?), ("w_down", layer.ptr("w_down")?),
            ("out", w.shared.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        if !matches!(experts, Experts::Local(_)) {
            let grid = (t * h.div_ceil(256)).div_ceil(8).clamp(1, 4 * 188);
            self.run("expert_input_quant", &[("source_ptr", w.x.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
                // SAFETY: the scale rows follow the payload inside each wire row.
                ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
                ("scale_mma_ptr", w.delta.buffer.ptr)], &[rows, Dsv4Scalar::I32(grid as i32)])?;
        }
        Ok(())
    }

    /// The routed experts of layer `index` (the front ran); leaves
    /// `bf16(routed + shared)` in `delta`.
    fn moe_experts(&self, w: &Workspace<'_>, index: usize, t: usize, rows: Dsv4Scalar, decode: bool) -> Result<()> {
        let h = self.cfg.hidden;
        let experts = self.experts.as_ref().context("MoE layer without experts")?;
        match experts {
            Experts::Local(local) => {
                let resident = local.index_of(index)?;
                let fp8 = local.experts.borrow();
                ensure!(!fp8.wire_input(), "the coordinator FP8 package takes BF16 rows");
                // SAFETY: input rows, route ids, weights and the output are live
                // buffers of `t` rows on this engine's stream.
                unsafe {
                    fp8.run(resident, t, w.x.buffer.ptr, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr,
                        w.routed.buffer.ptr, self.stream)?;
                }
                // The window may drop this layer before the stream drains.
                // SAFETY: the engine owns this stream.
                unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            }
            Experts::LocalExl3(local) => {
                local.ensure(index, self.stream)?;
                let mut resident = local.resident.borrow_mut();
                let (_, experts) = resident.as_mut().context("EXL3 window")?;
                // SAFETY: wire rows, routes and the shared-expert rows are complete in
                // stream order; the output is copied before the window can change.
                unsafe {
                    experts.run(crate::dsv4::local::LocalLayer::Backbone(index), t, w.wire.buffer.ptr,
                        w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, w.shared.buffer.ptr, self.stream)?;
                    self.library.copy_d2d_async(w.delta.buffer, experts.output.buffer, t * h * 2, self.stream)?;
                }
                return Ok(());
            }
            Experts::Spark { transport, runtime } => {
                // The compact reducer adds the shared expert plane to the rank partials.
                return self.spark_moe(w, index, t, decode, &mut transport.borrow_mut(), runtime);
            }
        }
        self.run("add", &[("a", w.routed.buffer.ptr), ("b", w.shared.buffer.ptr), ("out", w.delta.buffer.ptr)], &[rows])
    }

    /// Routes and wire rows down, one request to every Spark rank, the BF16
    /// rank partials and the shared expert summed into `delta`.
    fn spark_moe(&self, w: &Workspace<'_>, index: usize, t: usize, decode: bool, transport: &mut V41Tp4Roce,
        runtime: &tokio::runtime::Runtime) -> Result<()> {
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
            self.library.cuda_stream_synchronize(self.stream)?;
        }
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
        let ranks = transport.world_size();
        ensure!(ranks <= MAX_RANKS, "{ranks} Spark ranks exceed the reduction planes");
        let (row_bytes, plane_bytes) = (h * 2, t * h * 2);
        let mut staging = w.planes_host.borrow_mut();
        let bytes = staging.bytes_mut();
        let timer = std::time::Instant::now();
        runtime.block_on(async {
            transport.execute(&request, |rank, first, payload| {
                let offset = rank * plane_bytes + first as usize * row_bytes;
                ensure!(first as usize * row_bytes + payload.len() <= plane_bytes, "partial rows exceed the step");
                bytes[offset..offset + payload.len()].copy_from_slice(payload);
                Ok(())
            }).await
        })?;
        self.profile.borrow_mut()[1] += timer.elapsed().as_secs_f64();
        let mut pointers = [std::ptr::null::<u16>(); MAX_RANKS];
        for rank in 0..ranks {
            let source = cuteafd_ffi::CuteafdHostBuffer {
                // SAFETY: rank planes are disjoint slices of the staging buffer.
                ptr: unsafe { staging.buffer.ptr.cast::<u8>().add(rank * plane_bytes) }.cast(),
                bytes: plane_bytes,
                ..staging.buffer
            };
            // SAFETY: pinned source and device plane both hold `plane_bytes`.
            unsafe { self.library.copy_host_buffer_h2d_async(w.planes[rank].buffer, source, plane_bytes, self.stream)? };
            pointers[rank] = w.planes[rank].buffer.ptr.cast();
        }
        // SAFETY: planes, the shared-expert plane and `delta` are live [t, h] BF16
        // buffers ordered after the uploads.
        unsafe {
            self.library.v41_compact_reducer()?.reduce_planes(pointers, ranks as u32, w.shared.buffer.ptr.cast(),
                w.delta.buffer.ptr.cast(), t as u32, self.stream)?;
            self.library.cuda_stream_synchronize(self.stream)
        }
    }
}
