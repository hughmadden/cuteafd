//! Qwen 3.8 Flash Next (qwen4_exp) coordinator over the exported qwen4_* programs.
//!
//! Four hyper-connection streams (BF16 `[rows, 4, H]`) run through every
//! layer: the attention site's input (`hc_pre`, or the previous layer's fused
//! `hc_post_pre`), the attention sublayer, `hc_post_pre` into the streams and
//! onto the MLP site, the MoE, and the next fused post/pre (`hc_post` after
//! the last layer, then the stream mixer `head` and lm_head). PLE (layer 1)
//! adds its n-gram features to the streams before that layer's attention site.
//!
//! Attention: Gated DeltaNet layers keep per-sequence FP32 recurrent state
//! and short-conv state (the last three q/k/v inputs) in slot pools, one slot
//! per sequence shared by every GDN layer (and the PLE conv state); full
//! attention layers keep BF16 K/V records in 64-row pages, per-token raw
//! index keys beside them, and one pooled index key per completed 4-token
//! block in pool pages (64 blocks per page). Up to 2051 visible tokens every
//! token is attended; past that the top 512 blocks (qwen4_index_topk) expand
//! to tokens plus the open tail block.
//!
//! MoE: FP32 router logits, the native softmax top-10 (logits and weights
//! rounded to BF16 as the reference), the shared expert with its sigmoid
//! gate, and routed experts on this GPU (FP8 or EXL3 packages, a window of
//! resident layers) or on the Sparks.
use super::weights::{Qwen4Layer, Qwen4Weights};
use crate::shared::experts::fp8::{Fp8Experts, Fp8Layer};
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Programs, Scalar, VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors;
use cuteafd_loader::families::qwen4::{NgramHistory, Qwen4Attention, Qwen4Config};
use crate::shared::spark_intake::SparkLink;
use cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};
use std::cell::RefCell;
use std::ffi::c_void;

type Dev<'a> = DeviceAllocation<'a>;

pub(crate) const PAGE_ROWS: usize = 64;
/// Rows of the decode-shaped programs (`_m64`).
pub(crate) const DECODE_ROWS: usize = 64;
/// Selected-slot row width of the sparse attention (2048 + 3, padded to 64).
pub(crate) const SPARSE_TOPK: usize = 2112;
/// BF16 K/V record of one token: K [2, 256] then V [2, 256].
const RECORD_BYTES: usize = 2048;
const INDEX_DIM: usize = 128;
const MAX_RANKS: usize = 6;
const HC: usize = 4;
/// Tokens per QSA index block, and blocks per pool-cache page.
const BLOCK: usize = 4;
const POOL_PAGE_TOKENS: usize = BLOCK * PAGE_ROWS;
/// Rows of the PLE conv state ((taps - 1) x dilation).
const PLE_STATE_ROWS: usize = 9;
/// Most rows the E4M3 draft head (`qwen4_head_fp8`) takes.
const FP8_HEAD_ROWS: usize = 16;
/// Rows a speculative step records per GDN layer (the fork's `REPLAY_ROWS`).
pub(crate) const REPLAY_ROWS: usize = 64;
/// Target rows a sequence may hold for its MTP canonical history before the
/// next draft step (the MTP stash, per state slot).
pub(crate) const MTP_PENDING_ROWS: usize = 64;

/// Bytes of one GDN layer's replay record (`gdn_replay_layout` in the fork):
/// normalized keys and values, decay and beta, and the conv inputs.
fn gdn_replay_bytes(cfg: &Qwen4Config) -> usize {
    let (kh, vh, d) = (cfg.gdn_key_heads, cfg.gdn_value_heads, cfg.gdn_head_dim);
    let bytes = REPLAY_ROWS * (kh * d * 4 + vh * d * 4 + vh * 2 * 4 + cfg.gdn_conv_width() * 2);
    bytes.div_ceil(1024) * 1024
}

/// Routed experts on this GPU from the TP1 FP8 package: a window of resident
/// layers, reloaded when a step reaches a layer outside it.
pub(crate) struct LocalExperts<'a> {
    pub library: &'a NativeLibrary,
    pub tensors: &'a Fp8ExpertTensors,
    pub experts: RefCell<Fp8Experts<'a>>,
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
/// `exl3-qwen4-k45/rtx-tp1` package): a window of resident layers. The
/// package reads FP8 K32 wire rows and its reducer adds the shared expert.
pub(crate) struct LocalExl3<'a> {
    pub library: &'a NativeLibrary,
    pub native_lib: std::path::PathBuf,
    pub catalog: &'a cuteafd_loader::OfficialV41Catalog,
    pub resident: RefCell<Option<(std::ops::Range<usize>, crate::families::deepseek_v4::local::LocalExperts<'a>)>>,
    pub window: usize,
    pub layers: usize,
    /// The MTP layer's experts (draft stage 0) stay resident with every window;
    /// expert layer `layers` names them.
    pub mtp: bool,
    pub max_rows: usize,
    pub budget: usize,
    pub loads: RefCell<usize>,
}

impl LocalExl3<'_> {
    fn ensure(&self, layer: usize, stream: *mut c_void) -> Result<()> {
        if self.resident.borrow().as_ref().is_some_and(|(range, _)| range.contains(&layer) || layer == self.layers) {
            ensure!(layer < self.layers || self.mtp, "MTP experts are not loaded");
            return Ok(());
        }
        let layer = if layer == self.layers { 0 } else { layer };
        // SAFETY: the engine owns this stream; the old window's launches drain first.
        unsafe { self.library.cuda_stream_synchronize(stream)? };
        *self.resident.borrow_mut() = None;
        let range = layer..(layer + self.window).min(self.layers);
        let started = std::time::Instant::now();
        let local = crate::families::deepseek_v4::local::LocalExperts::load_range(self.library, &self.native_lib, self.catalog,
            usize::from(self.mtp), range.clone(), self.max_rows, self.budget, stream)?
            .context("no coordinator EXL3 package for this checkpoint (build qwen4:exl3-k45)")?;
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
    /// Spark ranks over RoCE (one BF16 partial plane per rank).
    Spark { transport: RefCell<SparkLink<'a>>, runtime: tokio::runtime::Runtime },
    /// No routed experts (plumbing tests only: the MoE output is the shared expert alone).
    SharedOnly,
}

/// Host tables of one step.
#[derive(Default)]
struct StepTables {
    decode: bool,
    /// A speculative verify: GDN and PLE record replay inputs instead of
    /// advancing their state (`commit` applies the accepted rows).
    spec: bool,
    positions: Vec<i64>,
    /// K/V record slot per row (also the row's raw index-key slot).
    kv_slots: Vec<i64>,
    /// GDN / PLE state slot per row.
    slots: Vec<i32>,
    /// First step row of each row's sequence.
    seq_first: Vec<i32>,
    /// Pool index-cache slot of the block a row completes, else -1.
    pool_slots: Vec<i64>,
    /// Complete blocks each row sees.
    cache_lengths: Vec<i32>,
    /// Record pages and pool pages: one shared table (prefill) or one padded row per step row.
    page_table: Vec<i32>,
    pool_table: Vec<i32>,
    /// Row stride of the page tables (0: every row reads one shared table).
    page_stride: usize,
    pool_stride: usize,
    /// Pages of the shared table (prefill) or of each padded row (decode).
    page_width: usize,
    pool_width: usize,
    /// Whether any row sees more than 2051 tokens (the block top-k runs).
    long: bool,
    /// PLE table rows [rows, 16].
    ple_ids: Vec<i64>,
}

/// A sequence's K/V pages, pool pages, state slot, length and n-gram history.
#[derive(Debug, Clone)]
pub(crate) struct Qwen4Placement {
    pub pages: Vec<i32>,
    pub pool_pages: Vec<i32>,
    pub slot: i32,
    pub len: usize,
    pub history: NgramHistory,
}

impl Qwen4Placement {
    pub fn record(&self, position: usize) -> Result<i64> {
        let page = *self.pages.get(position / PAGE_ROWS).context("position past the sequence's pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + (position % PAGE_ROWS) as i64)
    }

    /// Pool-cache slot of the block `position` completes, or -1.
    pub fn pool_slot(&self, position: usize) -> Result<i64> {
        if position % BLOCK != BLOCK - 1 {
            return Ok(-1);
        }
        let page = *self.pool_pages.get(position / POOL_PAGE_TOKENS).context("position past the pool pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + ((position / BLOCK) % PAGE_ROWS) as i64)
    }
}

/// Free record pages, pool pages and state slots.
pub(crate) struct Allocator {
    pages: Vec<i32>,
    pool_pages: Vec<i32>,
    slots: Vec<i32>,
    eos: u32,
    context: usize,
}

impl Allocator {
    pub fn new(pages: usize, slots: usize, cfg: &Qwen4Config) -> Self {
        let pool_pages = pages.div_ceil(BLOCK);
        Self { pages: (0..pages as i32).rev().collect(), pool_pages: (0..pool_pages as i32).rev().collect(),
            slots: (0..slots as i32).rev().collect(), eos: cfg.eos, context: cfg.ngram_size - 1 }
    }

    /// Reserves every page a sequence of up to `capacity` tokens needs and a
    /// state slot (the engine zeroes the slot and maps the pool pages before
    /// the first step).
    pub fn admit(&mut self, capacity: usize) -> Result<Qwen4Placement> {
        let pages = capacity.div_ceil(PAGE_ROWS).max(1);
        let pool_pages = capacity.div_ceil(POOL_PAGE_TOKENS).max(1);
        ensure!(self.pages.len() >= pages && self.pool_pages.len() >= pool_pages,
            "cache pages exhausted ({pages} + {pool_pages} pool pages needed, {} + {} free)", self.pages.len(),
            self.pool_pages.len());
        let slot = self.slots.pop().context("state slots exhausted")?;
        Ok(Qwen4Placement {
            pages: (0..pages).map(|_| self.pages.pop().unwrap()).collect(),
            pool_pages: (0..pool_pages).map(|_| self.pool_pages.pop().unwrap()).collect(),
            slot,
            len: 0,
            history: NgramHistory(vec![self.eos; self.context]),
        })
    }

    pub fn release(&mut self, placement: Qwen4Placement) {
        self.pages.extend(placement.pages);
        self.pool_pages.extend(placement.pool_pages);
        self.slots.push(placement.slot);
    }
}

struct Workspace<'a> {
    rows: usize,
    streams: [Dev<'a>; 2],
    inject: Dev<'a>,
    x: Dev<'a>,
    delta: Dev<'a>,
    shared: Dev<'a>,
    routed: Dev<'a>,
    positions: Dev<'a>,
    kv_slots: Dev<'a>,
    slots: Dev<'a>,
    seq_first: Dev<'a>,
    pool_slots: Dev<'a>,
    cache_lengths: Dev<'a>,
    page_table: Dev<'a>,
    pool_table: Dev<'a>,
    ple_ids: Dev<'a>,
    query: Dev<'a>,
    gate: Dev<'a>,
    index_q: Dev<'a>,
    attn: Dev<'a>,
    blocks: Dev<'a>,
    indices: Dev<'a>,
    lengths: Dev<'a>,
    scratch: Dev<'a>,
    topk_scratch: Dev<'a>,
    logits: Dev<'a>,
    logit_rows: usize,
    router_logits: Dev<'a>,
    route_ids: Dev<'a>,
    route_weights: Dev<'a>,
    wire: Dev<'a>,
    router_host: RefCell<HostAllocation<'a>>,
    /// MTP steps: the source row of each row's feedback, and the greedy draft
    /// (U32 token, FP32 logit) of each head row.
    hidden_rows: Dev<'a>,
    argmax: Dev<'a>,
    /// Pinned staging of a step's host tables and input rows (async uploads)
    /// and its fill level.
    staging: RefCell<(HostAllocation<'a>, usize)>,
    /// Pinned landing of the logits rows.
    logits_host: RefCell<HostAllocation<'a>>,
    head: VocabularyHead<'a>,
    _head_workspace: Dev<'a>,
}

/// One MTP row: the pair (target or draft pre-mixer streams at `position`,
/// the token at `position + 1`), its streams read from row `source` of the step's source.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MtpRow {
    pub position: usize,
    pub token: u32,
    pub source: i32,
}

/// A sequence's rows of an MTP step (contiguous, in position order).
pub(crate) struct MtpGroup<'p> {
    pub placement: &'p Qwen4Placement,
    pub rows: Vec<MtpRow>,
}

/// Where an MTP step reads its rows' streams.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MtpSource {
    /// The stash of target rows (row `slot * MTP_PENDING_ROWS + i`).
    Pending,
    /// The previous MTP step's output streams in the same workspace.
    Chain,
    /// The workspace's final target streams of its last step (prefill).
    Target,
    /// A caller buffer of `[rows, 4, H]` streams (golden checks).
    Buffer(*mut c_void),
}

pub(crate) struct Qwen4Engine<'a> {
    pub library: &'a NativeLibrary,
    pub programs: &'a Programs<'a>,
    pub cfg: Qwen4Config,
    pub weights: Qwen4Weights<'a>,
    pub ple: Option<super::ple::PleTable<'a>>,
    pub stream: *mut c_void,
    pub max_context: usize,
    pub prefill_rows: usize,
    pub pages: usize,
    pub slots: usize,
    /// Per full layer: the K/V record pool.
    kv: Vec<Option<Dev<'a>>>,
    /// Per layer: its ordinal among the GDN layers.
    gdn_ord: Vec<Option<usize>>,
    /// Every GDN layer's pools back to back (the commit program's layout):
    /// conv state BF16 [layers, slots, 3, C], FP32 recurrent state [layers,
    /// slots, 48, 128, 128], and the speculative replay records [layers, record].
    gdn_conv: Option<Dev<'a>>,
    gdn_state: Option<Dev<'a>>,
    gdn_replay: Option<Dev<'a>>,
    gdn_layers: usize,
    /// Per full layer: raw per-token index keys (BF16 [record slots, 128]) and pooled block keys.
    index: Vec<Option<(Dev<'a>, Dev<'a>)>>,
    /// PLE conv state pool (BF16 [slots, 9, 4H]) and its speculative replay record ([64, 4H]).
    ple_state: Option<Dev<'a>>,
    ple_replay: Option<Dev<'a>>,
    /// Commit tables (I32 [3, sequences]).
    commit_tables: Dev<'a>,
    /// The MTP layer's K/V records and index caches, and the stash of target
    /// pre-mixer stream rows awaiting the MTP (BF16 [slots, 64, 4, H]).
    mtp_kv: Option<(Dev<'a>, Dev<'a>, Dev<'a>)>,
    mtp_pending: Option<Dev<'a>>,
    /// Where the last step left its final streams (decode, stream buffer).
    last_streams: std::cell::Cell<(bool, usize)>,
    /// Where the last MTP step left its output streams (decode, stream buffer).
    mtp_streams: std::cell::Cell<(bool, usize)>,
    /// Logical page of each pool-cache page within its sequence.
    pool_logical: Dev<'a>,
    pub pool_pages: usize,
    workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    experts: Option<Experts<'a>>,
    /// Host seconds: GPU wait before expert exchanges, the exchanges.
    pub profile: RefCell<[f64; 2]>,
    graphs: RefCell<std::collections::HashMap<GraphKey, GraphExec<'a>>>,
    use_graphs: bool,
    /// Recorded after a Spark exchange's device-to-host copies: the host
    /// waits on it while the shared expert runs behind it.
    routes_ready: *mut c_void,
    /// L2 prefetch of the next layer's weights during decode exchanges.
    pub l2: Option<crate::shared::l2_prefetch::L2Prefetch>,
}

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

/// Where the step's streams live after each layer (for callbacks and forcing).
type LayerHook<'h> = Option<&'h mut dyn FnMut(usize, &[u8]) -> Result<()>>;
type Forced<'h> = Option<&'h dyn Fn(usize) -> Option<Vec<u8>>>;

impl<'a> Qwen4Engine<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, cfg: Qwen4Config, weights: Qwen4Weights<'a>,
        ple: Option<super::ple::PleTable<'a>>, stream: *mut c_void, max_context: usize, prefill_rows: usize,
        pages: usize, slots: usize) -> Result<Self> {
        cfg.check_programs()?;
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let pool_pages = pages.div_ceil(BLOCK);
        let (mut kv, mut gdn_ord, mut index) = (Vec::new(), Vec::new(), Vec::new());
        let mut gdn_layers = 0;
        for layer in &weights.layers {
            match layer.attention {
                Qwen4Attention::Full => {
                    kv.push(Some(zeroed(pages * PAGE_ROWS * RECORD_BYTES)?));
                    gdn_ord.push(None);
                    index.push(Some((zeroed(pages * PAGE_ROWS * INDEX_DIM * 2)?,
                        zeroed(pool_pages * PAGE_ROWS * INDEX_DIM * 2)?)));
                }
                Qwen4Attention::Gdn => {
                    kv.push(None);
                    gdn_ord.push(Some(gdn_layers));
                    gdn_layers += 1;
                    index.push(None);
                }
            }
        }
        let (gdn_conv, gdn_state, gdn_replay) = if gdn_layers > 0 {
            (Some(zeroed(gdn_layers * slots * Self::conv_slot_bytes(&cfg))?),
             Some(zeroed(gdn_layers * slots * Self::state_slot_bytes(&cfg))?),
             Some(zeroed(gdn_layers * gdn_replay_bytes(&cfg))?))
        } else {
            (None, None, None)
        };
        let (ple_state, ple_replay) = if weights.layers.len() > cfg.ple_layers.first().copied().unwrap_or(usize::MAX) {
            (Some(zeroed(slots * PLE_STATE_ROWS * cfg.hc_width() * 2)?),
             Some(zeroed(REPLAY_ROWS * cfg.hc_width() * 2)?))
        } else {
            (None, None)
        };
        let (mtp_kv, mtp_pending) = if weights.mtp.is_some() {
            (Some((zeroed(pages * PAGE_ROWS * RECORD_BYTES)?, zeroed(pages * PAGE_ROWS * INDEX_DIM * 2)?,
                zeroed(pool_pages * PAGE_ROWS * INDEX_DIM * 2)?)),
             Some(zeroed(slots * MTP_PENDING_ROWS * HC * cfg.hidden * 2)?))
        } else {
            (None, None)
        };
        let pool_logical = zeroed(pool_pages * 4)?;
        Ok(Self { library, programs, cfg, weights, ple, stream, max_context, prefill_rows, pages, slots, kv, gdn_ord,
            gdn_conv, gdn_state, gdn_replay, gdn_layers, index, ple_state, ple_replay,
            commit_tables: zeroed(3 * DECODE_ROWS * 4)?, mtp_kv, mtp_pending,
            last_streams: std::cell::Cell::new((false, 0)), mtp_streams: std::cell::Cell::new((false, 0)), pool_logical, pool_pages, workspace: RefCell::new(None),
            decode_workspace: RefCell::new(None), experts: None, profile: RefCell::new([0.0; 2]),
            graphs: RefCell::new(std::collections::HashMap::new()),
            use_graphs: std::env::var("CUTEAFD_QWEN4_GRAPHS").map_or(true, |v| v != "0"),
            routes_ready: library.cuda_event_create_ordering()?, l2: None })
    }

    pub fn set_experts(&mut self, experts: Experts<'a>) {
        self.experts = Some(experts);
    }

    pub fn experts(&self) -> Option<&Experts<'a>> {
        self.experts.as_ref()
    }

    /// Per MoE layer, the weights a decode step reads after its routed
    /// experts, in read order: the next layer's attention site, attention
    /// (the E4M3 copies where the decode programs read them), MLP site,
    /// router and shared expert; after the last layer the mixer and LM head.
    pub fn decode_read_order(&self) -> Vec<Vec<crate::shared::l2_prefetch::Range>> {
        let layers = &self.weights.layers;
        (0..layers.len()).map(|i| match layers.get(i + 1) {
            Some(next) => {
                let attention: &[&str] = match next.attention {
                    Qwen4Attention::Gdn => &["w_in", "conv_w", "a_log", "dt_bias", "norm_w", "w_out"],
                    Qwen4Attention::Full => &["w_in", "q_norm", "k_norm", "iq_norm", "ik_norm", "w_o"],
                };
                let names: Vec<&str> = ["attn.norm", "attn.w_di", "attn.w_up"].iter().chain(attention)
                    .chain(&["mlp.norm", "mlp.w_di", "mlp.w_up", "gate", "shared.w_gate_up", "shared.w_down"])
                    .copied().collect();
                crate::shared::l2_prefetch::operands(&names, |n| next.range(n))
            }
            None => self.weights.mixer.iter().chain([&self.weights.head])
                .map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes)).collect(),
        }).collect()
    }

    /// After layer `index`'s shared expert is queued in a decode step: the
    /// L2 prefetch of what the step reads next; with local experts only under
    /// CUTEAFD_EMULATE_EXCHANGE_US (benchmarks), with a Spark-like wait.
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

    /// Before a sequence's first step: zeroes its state slot and maps its pool pages.
    pub(crate) fn start(&self, placement: &Qwen4Placement) -> Result<()> {
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

    fn conv_slot_bytes(cfg: &Qwen4Config) -> usize {
        (cfg.conv_kernel - 1) * cfg.gdn_conv_width() * 2
    }

    fn state_slot_bytes(cfg: &Qwen4Config) -> usize {
        cfg.gdn_value_heads * cfg.gdn_head_dim * cfg.gdn_head_dim * 4
    }

    /// `bytes` at `offset` inside `pool`.
    fn region(pool: &Dev<'_>, offset: usize, bytes: usize) -> cuteafd_ffi::CuteafdDeviceBuffer {
        debug_assert!(offset + bytes <= pool.buffer.bytes);
        cuteafd_ffi::CuteafdDeviceBuffer {
            // SAFETY: callers pass offsets inside the pool.
            ptr: unsafe { pool.buffer.ptr.cast::<u8>().add(offset) }.cast(),
            bytes,
            ..pool.buffer
        }
    }

    /// Every per-slot state region of `slot` (GDN conv + recurrent state per
    /// GDN layer, PLE conv state).
    fn slot_regions(&self, slot: usize) -> Vec<cuteafd_ffi::CuteafdDeviceBuffer> {
        let mut regions = Vec::new();
        let (conv, state) = (Self::conv_slot_bytes(&self.cfg), Self::state_slot_bytes(&self.cfg));
        for ord in 0..self.gdn_layers {
            if let (Some(c), Some(s)) = (&self.gdn_conv, &self.gdn_state) {
                regions.push(Self::region(c, (ord * self.slots + slot) * conv, conv));
                regions.push(Self::region(s, (ord * self.slots + slot) * state, state));
            }
        }
        if let Some(ple) = &self.ple_state {
            let per = ple.buffer.bytes / self.slots;
            regions.push(Self::region(ple, slot * per, per));
        }
        regions
    }

    /// GDN layer `ord`'s conv state pool, recurrent state pool and replay record.
    fn gdn_pools(&self, ord: usize) -> Result<[*mut c_void; 3]> {
        let (c, s, r) = (self.gdn_conv.as_ref(), self.gdn_state.as_ref(), self.gdn_replay.as_ref());
        let (Some(c), Some(s), Some(r)) = (c, s, r) else { anyhow::bail!("GDN layer without state pools") };
        let slots = self.slots;
        Ok([Self::region(c, ord * slots * Self::conv_slot_bytes(&self.cfg), 0).ptr,
            Self::region(s, ord * slots * Self::state_slot_bytes(&self.cfg), 0).ptr,
            Self::region(r, ord * gdn_replay_bytes(&self.cfg), 0).ptr])
    }

    /// Zeroes a sequence's state (before its first step).
    pub fn reset_slot(&self, slot: i32) -> Result<()> {
        let slot = usize::try_from(slot)?;
        ensure!(slot < self.slots, "state slot {slot} out of range");
        for region in self.slot_regions(slot) {
            self.library.cuda_zero_bytes(region, region.bytes)?;
        }
        Ok(())
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

    fn workspace(&self, t: usize, decode: bool, logit_rows: usize) -> Result<Workspace<'a>> {
        let h = self.cfg.hidden;
        let cap = if decode { "m64" } else { "m4096" };
        let ple = if self.ple.as_ref().is_some_and(|p| p.fp8) { "qwen4_ple_fp8" } else { "qwen4_ple_bf16" };
        let mut scratch = 0;
        for name in ["qwen4_hc_pre".to_string(), "qwen4_hc_post_pre".into(), "qwen4_head".into(),
            "qwen4_shared".into(), ple.into(), "qwen4_mtp_feedback".into(), format!("qwen4_gdn_{cap}"),
            format!("qwen4_attn_producer_{cap}"),
            format!("qwen4_sparse_gqa_{cap}"), format!("qwen4_attn_o_{cap}")] {
            if let Ok(bytes) = self.scratch(&name) {
                scratch = usize::max(scratch, bytes);
            }
        }
        let topk_scratch = self.scratch(&format!("qwen4_index_topk_{cap}")).unwrap_or(0);
        let blocks = self.cfg.index_budget / BLOCK;
        let table_rows = if decode { t } else { 1 };
        let head_workspace = self.alloc(VOCABULARY_HEAD_WORKSPACE)?;
        let spark = matches!(self.experts, Some(Experts::Spark { .. }));
        let (topk, heads, hd) = (self.cfg.topk, self.cfg.heads, self.cfg.head_dim);
        Ok(Workspace {
            rows: t,
            streams: [self.alloc(t * HC * h * 2)?, self.alloc(t * HC * h * 2)?],
            inject: self.alloc(t * HC * 2)?,
            x: self.alloc(t * h * 2)?,
            delta: self.alloc(t * h * 2)?,
            shared: self.alloc(t * h * 2)?,
            routed: self.alloc(t * h * 2)?,
            positions: self.alloc(t * 8)?,
            kv_slots: self.alloc(t * 8)?,
            slots: self.alloc(t * 4)?,
            seq_first: self.alloc(t * 4)?,
            pool_slots: self.alloc(t * 8)?,
            cache_lengths: self.alloc(t * 4)?,
            page_table: self.alloc(table_rows * self.pages * 4)?,
            pool_table: self.alloc(table_rows * self.pool_pages * 4)?,
            ple_ids: self.alloc(t * self.cfg.ple_rows() * 8)?,
            query: self.alloc(t * heads * hd * 2)?,
            gate: self.alloc(t * heads * hd * 2)?,
            index_q: self.alloc(t * self.cfg.index_heads * INDEX_DIM * 2)?,
            attn: self.alloc(t * heads * hd * 2)?,
            blocks: self.alloc(t * blocks * 4)?,
            indices: self.alloc(t * SPARSE_TOPK * 4)?,
            lengths: self.alloc(t * 4)?,
            scratch: self.alloc(scratch)?,
            topk_scratch: {
                let zero = self.alloc(topk_scratch)?;
                self.library.cuda_zero_bytes(zero.buffer, zero.buffer.bytes)?;
                zero
            },
            logits: self.alloc(logit_rows * self.cfg.vocab_size * 4)?,
            logit_rows,
            router_logits: self.alloc(t * self.cfg.experts * 4)?,
            route_ids: self.alloc(t * topk * 4)?,
            route_weights: self.alloc(t * topk * 4)?,
            wire: self.alloc(t * (h + h / 32))?,
            router_host: RefCell::new(HostAllocation::new(self.library,
                if spark { t * (topk * 8 + h + h / 32) } else { 256 })?),
            hidden_rows: self.alloc(t * 4)?,
            argmax: self.alloc(logit_rows * 8)?,
            logits_host: RefCell::new(HostAllocation::new(self.library, logit_rows * self.cfg.vocab_size * 4)?),
            staging: RefCell::new((HostAllocation::new(self.library, 16 * 16
                + t * (8 * 3 + 4 * 4 + self.cfg.ple_rows() * 8 + (HC + 1) * h * 2)
                + table_rows * (self.pages + self.pool_pages) * 4)?, 0)),
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, h as u32, logit_rows as u32,
                self.cfg.vocab_size as u32)? },
            _head_workspace: head_workspace,
        })
    }

    fn put<T: Copy>(&self, dev: &Dev<'_>, values: &[T]) -> Result<()> {
        let bytes = bytes_of(values);
        ensure!(bytes.len() <= dev.buffer.bytes, "table exceeds its buffer");
        if bytes.is_empty() {
            return Ok(());
        }
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: bytes.len(), ..dev.buffer }, bytes)
    }

    /// Starts a step's staged uploads (the previous step's have drained).
    fn begin_staging(&self, w: &Workspace<'_>) -> Result<()> {
        // SAFETY: the engine owns this stream; its earlier copies read the staging bytes.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        w.staging.borrow_mut().1 = 0;
        Ok(())
    }

    /// Queues `bytes` into `dst` through the workspace's pinned staging.
    fn stage(&self, w: &Workspace<'_>, dst: cuteafd_ffi::CuteafdDeviceBuffer, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        ensure!(bytes.len() <= dst.bytes, "staged upload exceeds its buffer");
        let mut staging = w.staging.borrow_mut();
        let at = staging.1;
        ensure!(at + bytes.len() <= staging.0.buffer.bytes, "step inputs exceed the staging buffer");
        staging.0.bytes_mut()[at..at + bytes.len()].copy_from_slice(bytes);
        let source = cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: `at` lies inside the pinned staging buffer.
            ptr: unsafe { staging.0.buffer.ptr.cast::<u8>().add(at) }.cast(),
            bytes: bytes.len(),
            ..staging.0.buffer
        };
        // SAFETY: the staged bytes stay untouched until `begin_staging` drains the stream.
        unsafe { self.library.copy_host_buffer_h2d_async(dst, source, bytes.len(), self.stream)? };
        staging.1 = (at + bytes.len()).div_ceil(16) * 16;
        Ok(())
    }

    fn stage_table<T: Copy>(&self, w: &Workspace<'_>, dev: &Dev<'_>, values: &[T]) -> Result<()> {
        self.stage(w, dev.buffer, bytes_of(values))
    }

    fn download(&self, dev: &Dev<'_>, bytes: usize) -> Result<Vec<u8>> {
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        let mut out = vec![0u8; bytes];
        self.library.copy_d2h(&mut out, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..dev.buffer })?;
        Ok(out)
    }

    /// Per-row positions, record and pool slots, blocks seen, PLE rows.
    fn rows(&self, placement: &mut Qwen4Placement, tokens: &[u32], first: i32, tables: &mut StepTables) -> Result<()> {
        let start = placement.len;
        for (i, _) in tokens.iter().enumerate() {
            let position = start + i;
            ensure!(position < self.max_context, "position {position} past the context {}", self.max_context);
            tables.positions.push(position as i64);
            tables.kv_slots.push(placement.record(position)?);
            tables.pool_slots.push(placement.pool_slot(position)?);
            tables.slots.push(placement.slot);
            tables.seq_first.push(first);
            tables.cache_lengths.push(((position + 1) / BLOCK) as i32);
            tables.long |= position + 1 > self.cfg.dense_context();
            tables.pool_width = tables.pool_width.max((position + 1).div_ceil(POOL_PAGE_TOKENS));
        }
        if let Some(ple) = &self.ple {
            ple.hasher.hash(&mut placement.history, tokens, &mut tables.ple_ids)?;
        }
        Ok(())
    }

    /// Prefills a sequence from its length through every resident layer and
    /// returns the logits of the last `logit_rows` rows when all layers are
    /// resident. `on_layer` receives each layer's output streams (BF16 [t, 4,
    /// H]); with `forced`, `forced(l)` (when it returns rows) replaces the
    /// streams after layer `l`, so each layer's comparison measures that layer alone.
    pub fn prefill_forced(&self, placement: &mut Qwen4Placement, tokens: &[u32], embed: &[u8], on_layer: LayerHook<'_>,
        forced: Forced<'_>, logit_rows: usize) -> Result<Option<Vec<f32>>> {
        let (t, start) = (tokens.len(), placement.len);
        ensure!(t > 0 && t <= self.prefill_rows && start + t <= self.max_context, "prefill of {t} rows at {start}");
        ensure!(embed.len() == t * self.cfg.hidden * 2, "embedding rows do not match the tokens");
        if start == 0 {
            self.start(placement)?;
        }
        let mut tables = StepTables { page_table: placement.pages.clone(), pool_table: placement.pool_pages.clone(),
            page_stride: 0, pool_stride: 0, page_width: placement.pages.len(), ..Default::default() };
        self.rows(placement, tokens, 0, &mut tables)?;
        let logits = self.step(&tables, embed, logit_rows.clamp(1, t), on_layer, forced)?;
        placement.len += t;
        Ok(logits)
    }

    pub fn prefill(&self, placement: &mut Qwen4Placement, tokens: &[u32], embed: &[u8]) -> Result<Option<Vec<f32>>> {
        self.prefill_forced(placement, tokens, embed, None, None, 1)
    }

    /// Appends each sequence's tokens (one for decode, several for a verify)
    /// at its length in one decode-shaped step; returns every row's logits.
    /// GDN and PLE state advance in place: a caller rejecting a suffix must replay.
    pub fn verify(&self, sequences: &mut [(&mut Qwen4Placement, &[u32])], embed: &[u8], on_layer: LayerHook<'_>)
        -> Result<Option<Vec<f32>>> {
        self.verify_step(sequences, embed, on_layer, false)
    }

    /// [`Self::verify`] as a speculative step: the GDN and PLE state stay as
    /// they were and each row's replay inputs are recorded; [`Self::commit`]
    /// then applies each sequence's accepted rows (and the caller rewinds the
    /// placements with [`Self::rewind`]). K/V records past the accepted rows
    /// are overwritten when those positions come again.
    pub fn verify_spec(&self, sequences: &mut [(&mut Qwen4Placement, &[u32])], embed: &[u8],
        on_layer: LayerHook<'_>) -> Result<Option<Vec<f32>>> {
        self.verify_step(sequences, embed, on_layer, true)
    }

    fn verify_step(&self, sequences: &mut [(&mut Qwen4Placement, &[u32])], embed: &[u8], on_layer: LayerHook<'_>,
        spec: bool) -> Result<Option<Vec<f32>>> {
        let rows: usize = sequences.iter().map(|(_, t)| t.len()).sum();
        ensure!(rows > 0 && rows <= DECODE_ROWS && embed.len() == rows * self.cfg.hidden * 2, "decode step of {rows} rows");
        let page_stride = sequences.iter().map(|(p, _)| p.pages.len()).max().unwrap_or(1).next_power_of_two()
            .min(self.pages);
        let pool_stride = sequences.iter().map(|(p, _)| p.pool_pages.len()).max().unwrap_or(1).next_power_of_two()
            .min(self.pool_pages);
        let mut tables = StepTables { decode: true, spec, page_stride, pool_stride, page_width: page_stride,
            ..Default::default() };
        for (placement, tokens) in sequences.iter_mut() {
            if placement.len == 0 {
                self.start(placement)?;
            }
            let first = tables.kv_slots.len() as i32;
            self.rows(placement, tokens, first, &mut tables)?;
            for _ in 0..tokens.len() {
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
        for (placement, tokens) in sequences.iter_mut() {
            placement.len += tokens.len();
        }
        Ok(logits)
    }

    /// Applies each sequence's accepted rows of the last speculative step to
    /// the GDN and PLE state: `(state slot, first step row, accepted rows)`.
    pub fn commit(&self, accepted: &[(i32, usize, usize)]) -> Result<()> {
        let n = accepted.len();
        if n == 0 {
            return Ok(());
        }
        ensure!(n <= DECODE_ROWS, "commit of {n} sequences");
        let mut table = vec![0i32; 3 * n];
        for (i, &(slot, first, keep)) in accepted.iter().enumerate() {
            ensure!(first + keep <= REPLAY_ROWS && usize::try_from(slot).is_ok_and(|s| s < self.slots),
                "commit of rows {first}+{keep} in slot {slot}");
            (table[i], table[n + i], table[2 * n + i]) = (slot, i32::try_from(first)?, i32::try_from(keep)?);
        }
        // SAFETY: the engine owns this stream; the previous commit's table is consumed.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        self.put(&self.commit_tables, &table)?;
        let i32s = |v: usize| -> Result<Scalar> { Ok(Scalar::I32(i32::try_from(v)?)) };
        if let (Some(conv), Some(state), Some(replay)) = (&self.gdn_conv, &self.gdn_state, &self.gdn_replay) {
            self.run("qwen4_gdn_commit", &[("state", state.buffer.ptr), ("conv_state", conv.buffer.ptr),
                ("replay", replay.buffer.ptr), ("tables", self.commit_tables.buffer.ptr)],
                &[i32s(n)?, i32s(self.gdn_layers)?, i32s(self.slots)?])?;
        }
        if let (Some(state), Some(replay)) = (&self.ple_state, &self.ple_replay) {
            self.run("qwen4_ple_commit", &[("conv_state", state.buffer.ptr), ("replay", replay.buffer.ptr),
                ("tables", self.commit_tables.buffer.ptr)], &[i32s(n)?])?;
        }
        Ok(())
    }

    /// After a speculative step: `placement` keeps the `kept` tokens it verified
    /// from `start`, whose n-gram history was `history` before the step.
    pub fn rewind(&self, placement: &mut Qwen4Placement, start: usize, history: NgramHistory, kept: &[u32])
        -> Result<()> {
        placement.len = start + kept.len();
        placement.history = history;
        if let Some(ple) = &self.ple {
            ple.hasher.hash(&mut placement.history, kept, &mut Vec::new())?;
        }
        Ok(())
    }

    /// Copies target stream rows of the last step (`decode`: the decode
    /// workspace, else the prefill one) into the MTP stash: `(state slot,
    /// first step row, rows, first stash row)` per sequence.
    pub fn mtp_stash(&self, decode: bool, rows: &[(i32, usize, usize, usize)]) -> Result<()> {
        let pending = self.mtp_pending.as_ref().context("MTP is not loaded")?;
        let (last_decode, cur) = self.last_streams.get();
        ensure!(last_decode == decode, "the stash reads the last step's workspace");
        let workspace = if decode { self.decode_workspace.borrow() } else { self.workspace.borrow() };
        let w = workspace.as_ref().context("no step ran")?;
        let row = HC * self.cfg.hidden * 2;
        for &(slot, first, count, at) in rows {
            let slot = usize::try_from(slot)?;
            ensure!(slot < self.slots && at + count <= MTP_PENDING_ROWS && first + count <= w.rows,
                "MTP stash of rows {first}+{count} at {at}");
            if count == 0 {
                continue;
            }
            let dst = Self::region(pending, (slot * MTP_PENDING_ROWS + at) * row, count * row);
            let src = Self::region(&w.streams[cur], first * row, count * row);
            // SAFETY: both regions lie inside live buffers; the stream orders the copy.
            unsafe { self.library.copy_d2d_async(dst, src, count * row, self.stream)? };
        }
        Ok(())
    }

    /// One MTP step over `groups` (each sequence's rows contiguous) in the
    /// decode workspace (`decode`, at most 64 rows) or the prefill one.
    /// Rows read their streams from `source`; the output streams stay in the
    /// workspace for a following [`MtpSource::Chain`] step. With `heads`
    /// (step rows), returns each head row's greedy draft (token, logit) and,
    /// with `logits`, the head rows' FP32 logits.
    #[allow(clippy::type_complexity)]
    pub fn mtp_step(&self, decode: bool, groups: &[MtpGroup<'_>], source: MtpSource, heads: &[usize],
        embed: &[u8], logits: bool) -> Result<(Vec<(u32, f32)>, Option<Vec<f32>>)> {
        let mtp = self.weights.mtp.as_ref().context("MTP is not loaded (--mtp)")?;
        let (kv, keys, blocks) = self.mtp_kv.as_ref().context("MTP pools")?;
        let (h, vocab) = (self.cfg.hidden, self.cfg.vocab_size);
        let t: usize = groups.iter().map(|g| g.rows.len()).sum();
        let capacity = if decode { DECODE_ROWS } else { self.prefill_rows };
        ensure!(t > 0 && t <= capacity && heads.len() <= t && embed.len() == t * h * 2, "MTP step of {t} rows");
        let mut tables = StepTables { decode, ..Default::default() };
        let mut hidden_rows = Vec::with_capacity(t);
        if decode {
            let stride = |n: usize, total: usize| n.next_power_of_two().min(total);
            tables.page_stride = stride(groups.iter().map(|g| g.placement.pages.len()).max().unwrap_or(1), self.pages);
            tables.pool_stride = stride(groups.iter().map(|g| g.placement.pool_pages.len()).max().unwrap_or(1),
                self.pool_pages);
            tables.page_width = tables.page_stride;
        } else {
            ensure!(groups.len() == 1, "a prefill-shaped MTP step takes one sequence");
            tables.page_table = groups[0].placement.pages.clone();
            tables.pool_table = groups[0].placement.pool_pages.clone();
            tables.page_width = groups[0].placement.pages.len();
        }
        for group in groups {
            let first = tables.kv_slots.len() as i32;
            let placement = group.placement;
            for row in &group.rows {
                let position = row.position;
                ensure!(position < self.max_context, "MTP position {position} past the context");
                tables.positions.push(position as i64);
                tables.kv_slots.push(placement.record(position)?);
                tables.pool_slots.push(placement.pool_slot(position)?);
                tables.slots.push(placement.slot);
                tables.seq_first.push(first);
                tables.cache_lengths.push(((position + 1) / BLOCK) as i32);
                tables.long |= position + 1 > self.cfg.dense_context();
                tables.pool_width = tables.pool_width.max((position + 1).div_ceil(POOL_PAGE_TOKENS));
                hidden_rows.push(match source {
                    MtpSource::Pending => i32::try_from(usize::try_from(placement.slot)? * MTP_PENDING_ROWS)?
                        + row.source,
                    _ => row.source,
                });
                if decode {
                    let mut pages = placement.pages.clone();
                    pages.resize(tables.page_stride, 0);
                    tables.page_table.extend(pages);
                    let mut pools = placement.pool_pages.clone();
                    pools.resize(tables.pool_stride, 0);
                    tables.pool_table.extend(pools);
                }
            }
        }
        if decode {
            tables.pool_width = tables.pool_width.next_power_of_two().min(tables.pool_stride);
        }
        let slot = if decode { &self.decode_workspace } else { &self.workspace };
        if slot.borrow().is_none() {
            *slot.borrow_mut() = Some(self.workspace(capacity, decode, if decode { DECODE_ROWS } else { 1 })?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        ensure!(heads.len() <= w.logit_rows, "{} MTP head rows exceed the workspace's {}", heads.len(), w.logit_rows);
        // Source streams, and the buffer the feedback writes (never the source).
        let (src, dst) = match source {
            MtpSource::Pending => (self.mtp_pending.as_ref().context("MTP stash")?.buffer.ptr, 0),
            MtpSource::Buffer(ptr) => (ptr, 0),
            MtpSource::Chain => {
                let (d, cur) = self.mtp_streams.get();
                ensure!(d == decode, "an MTP chain step follows an MTP step in the same workspace");
                (w.streams[cur].buffer.ptr, cur ^ 1)
            }
            MtpSource::Target => {
                let (d, cur) = self.last_streams.get();
                ensure!(d == decode, "the MTP reads the last target step of the same workspace");
                (w.streams[cur].buffer.ptr, cur ^ 1)
            }
        };
        self.begin_staging(w)?;
        self.stage_table(w, &w.positions, &tables.positions)?;
        self.stage_table(w, &w.kv_slots, &tables.kv_slots)?;
        self.stage_table(w, &w.slots, &tables.slots)?;
        self.stage_table(w, &w.seq_first, &tables.seq_first)?;
        self.stage_table(w, &w.pool_slots, &tables.pool_slots)?;
        self.stage_table(w, &w.cache_lengths, &tables.cache_lengths)?;
        self.stage_table(w, &w.page_table, &tables.page_table)?;
        self.stage_table(w, &w.pool_table, &tables.pool_table)?;
        self.stage_table(w, &w.hidden_rows, &hidden_rows)?;
        self.stage(w, w.x.buffer, embed)?;
        let rows = Scalar::I32(t as i32);
        let cap = if decode { "m64" } else { "m4096" };
        self.run("qwen4_mtp_feedback", &[("hidden", src), ("hidden_rows", w.hidden_rows.buffer.ptr),
            ("embed", w.x.buffer.ptr), ("norm_hidden", mtp.norm_hidden.buffer.ptr),
            ("norm_embed", mtp.norm_embed.buffer.ptr), ("fc_hidden", mtp.fc_hidden.buffer.ptr),
            ("fc_embed", mtp.fc_embed.buffer.ptr), ("streams", w.streams[dst].buffer.ptr),
            ("delta", w.delta.buffer.ptr), ("inject", w.inject.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
            &[rows])?;
        // The embedding branch joins every stream (unit injection), then the attention site.
        self.post_pre(w, dst, &mtp.layer, "attn", rows)?;
        self.attend(w, &mtp.layer, kv.buffer.ptr, keys.buffer.ptr, blocks.buffer.ptr, rows, cap, &tables)?;
        self.post_pre(w, dst ^ 1, &mtp.layer, "mlp", rows)?;
        self.moe(w, self.cfg.layers, &mtp.layer, t, rows, decode)?;
        let mut out = dst;
        self.post(w, &mut out, rows)?;
        self.mtp_streams.set((decode, out));
        if heads.is_empty() {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok((Vec::new(), None));
        }
        let [norm, down, up] = &mtp.mixer;
        self.run("qwen4_head", &[("streams", w.streams[out].buffer.ptr), ("norm", norm.buffer.ptr),
            ("w_down", down.buffer.ptr), ("w_up", up.buffer.ptr), ("out", w.x.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        // The head rows, gathered contiguously into `delta`.
        let row = h * 2;
        for (i, &r) in heads.iter().enumerate() {
            ensure!(r < t, "MTP head row {r} of {t}");
            // SAFETY: row r of `x` and row i of `delta` lie inside [t, H] buffers.
            unsafe {
                self.library.copy_d2d_async(Self::region(&w.delta, i * row, row), Self::region(&w.x, r * row, row),
                    row, self.stream)?;
            }
        }
        let n = heads.len();
        match &mtp.head_fp8 {
            Some((q, scale)) if n <= FP8_HEAD_ROWS => self.run("qwen4_head_fp8", &[("x", w.delta.buffer.ptr),
                ("w_fp8", q.buffer.ptr), ("scale", scale.buffer.ptr), ("logits", w.logits.buffer.ptr)],
                &[Scalar::I32(n as i32)])?,
            // SAFETY: the gathered rows, the shared head and the logits are live buffers of these shapes.
            _ => unsafe {
                w.head.launch(w.delta.buffer.ptr.cast(), self.weights.head.buffer.ptr.cast(),
                    w.logits.buffer.ptr.cast(), n as u32, self.stream)?;
            },
        }
        // SAFETY: the logits and the argmax outputs are live buffers of these shapes.
        unsafe {
            self.library.cuda_logits_argmax_checked_f32_async(
                cuteafd_ffi::CuteafdDeviceBuffer { bytes: n * vocab * 4, ..w.logits.buffer },
                Self::region(&w.argmax, 0, n * 4), Self::region(&w.argmax, n * 4, n * 4), n, vocab, self.stream)?;
        }
        let best = self.download(&w.argmax, n * 8)?;
        let word = |i: usize| u32::from_le_bytes(best[i * 4..i * 4 + 4].try_into().unwrap());
        let drafts = (0..n).map(|i| (word(i), f32::from_bits(word(n + i)))).collect();
        let logits = if logits { Some(self.download_logits(w, n)?) } else { None };
        Ok((drafts, logits))
    }

    /// The last MTP step's output streams (BF16 [rows, 4, H]).
    pub fn mtp_output(&self, rows: usize) -> Result<Vec<u8>> {
        let (decode, cur) = self.mtp_streams.get();
        let workspace = if decode { self.decode_workspace.borrow() } else { self.workspace.borrow() };
        let w = workspace.as_ref().context("no MTP step ran")?;
        self.download(&w.streams[cur], rows * HC * self.cfg.hidden * 2)
    }

    fn step(&self, tables: &StepTables, embed: &[u8], logit_rows: usize, mut on_layer: LayerHook<'_>,
        forced: Forced<'_>) -> Result<Option<Vec<f32>>> {
        let (h, t) = (self.cfg.hidden, tables.kv_slots.len());
        let (slot, capacity) = if tables.decode { (&self.decode_workspace, DECODE_ROWS) } else { (&self.workspace, self.prefill_rows) };
        if slot.borrow().as_ref().is_some_and(|w| w.logit_rows < logit_rows) {
            *slot.borrow_mut() = None;
        }
        if slot.borrow().is_none() {
            let rows = if tables.decode { DECODE_ROWS } else { logit_rows.max(1) };
            *slot.borrow_mut() = Some(self.workspace(capacity, tables.decode, rows)?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        ensure!(t <= w.rows && logit_rows <= t, "step exceeds the workspace");
        self.begin_staging(w)?;
        self.stage_table(w, &w.positions, &tables.positions)?;
        self.stage_table(w, &w.kv_slots, &tables.kv_slots)?;
        self.stage_table(w, &w.slots, &tables.slots)?;
        self.stage_table(w, &w.seq_first, &tables.seq_first)?;
        self.stage_table(w, &w.pool_slots, &tables.pool_slots)?;
        self.stage_table(w, &w.cache_lengths, &tables.cache_lengths)?;
        self.stage_table(w, &w.page_table, &tables.page_table)?;
        self.stage_table(w, &w.pool_table, &tables.pool_table)?;
        self.stage_table(w, &w.ple_ids, &tables.ple_ids)?;
        // Streams start as four copies of the embedding.
        let row = h * 2;
        let mut streams = vec![0u8; t * HC * row];
        for (r, e) in embed.chunks_exact(row).enumerate() {
            for s in 0..HC {
                streams[(r * HC + s) * row..][..row].copy_from_slice(e);
            }
        }
        self.stage(w, w.streams[0].buffer, &streams)?;
        let rows = Scalar::I32(t as i32);
        if self.use_graphs && tables.decode && on_layer.is_none() && forced.is_none() {
            return self.decode_graphed(w, tables, t, rows, logit_rows);
        }
        let cap = if tables.decode { "m64" } else { "m4096" };
        let layers = &self.weights.layers;
        let mut cur = 0usize;
        let spec = tables.spec;
        self.enter(w, &mut cur, None, &layers[0], 0, rows, spec)?;
        for (index, layer) in layers.iter().enumerate() {
            match layer.attention {
                Qwen4Attention::Gdn => self.gdn(w, index, layer, rows, cap, spec)?,
                Qwen4Attention::Full => self.full(w, index, layer, rows, cap, tables)?,
            }
            // Attention back into the streams, then the MLP site's input.
            self.post_pre(w, cur, layer, "mlp", rows)?;
            cur ^= 1;
            self.moe(w, index, layer, t, rows, tables.decode)?;
            let forced_rows = forced.and_then(|f| f(index));
            match layers.get(index + 1) {
                Some(next) => {
                    let has_ple = self.cfg.ple_layers.contains(&(index + 1));
                    if has_ple || on_layer.is_some() || forced_rows.is_some() {
                        // Materialize this layer's output first (PLE and hooks see it).
                        self.post(w, &mut cur, rows)?;
                        if let Some(on_layer) = on_layer.as_mut() {
                            on_layer(index, &self.download(&w.streams[cur], t * HC * row)?)?;
                        }
                        if let Some(rows_forced) = forced_rows {
                            ensure!(rows_forced.len() == t * HC * row, "teacher-forced streams of the wrong size");
                            self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: rows_forced.len(),
                                ..w.streams[cur].buffer }, &rows_forced)?;
                        }
                        self.enter(w, &mut cur, None, next, index + 1, rows, spec)?;
                    } else {
                        self.enter(w, &mut cur, Some(()), next, index + 1, rows, spec)?;
                    }
                }
                None => {
                    self.post(w, &mut cur, rows)?;
                    if let Some(on_layer) = on_layer.as_mut() {
                        on_layer(index, &self.download(&w.streams[cur], t * HC * row)?)?;
                    }
                }
            }
        }
        self.last_streams.set((tables.decode, cur));
        if layers.len() < self.cfg.layers {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        self.logits(w, &w.streams[cur], t, rows, logit_rows).map(Some)
    }

    /// The stream mixer and lm_head over the last `logit_rows` rows.
    fn logits(&self, w: &Workspace<'_>, streams: &Dev<'_>, t: usize, rows: Scalar, logit_rows: usize)
        -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        let [norm, down, up] = &self.weights.mixer;
        self.run("qwen4_head", &[("streams", streams.buffer.ptr), ("norm", norm.buffer.ptr),
            ("w_down", down.buffer.ptr), ("w_up", up.buffer.ptr), ("out", w.x.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        // SAFETY: the head's input and operands are live buffers of these shapes.
        unsafe {
            w.head.launch(w.x.buffer.ptr.cast::<u8>().add((t - logit_rows) * h * 2).cast(),
                self.weights.head.buffer.ptr.cast(), w.logits.buffer.ptr.cast(), logit_rows as u32, self.stream)?;
        }
        self.download_logits(w, logit_rows)
    }

    /// The first `rows` logits rows through the pinned landing buffer.
    fn download_logits(&self, w: &Workspace<'_>, rows: usize) -> Result<Vec<f32>> {
        let n = rows * self.cfg.vocab_size;
        let host = w.logits_host.borrow();
        ensure!(n * 4 <= host.buffer.bytes, "logits rows exceed the landing buffer");
        // SAFETY: the pinned buffer holds n floats; the sync completes the copy before the read.
        unsafe {
            self.library.copy_d2h_host_buffer_async(host.buffer, w.logits.buffer, n * 4, self.stream)?;
            self.library.cuda_stream_synchronize(self.stream)?;
            Ok(std::slice::from_raw_parts(host.buffer.ptr.cast::<f32>(), n).to_vec())
        }
    }

    /// A decode step as captured segments: segment `i` finishes layer `i - 1`
    /// (its MoE output into the streams) and runs layer `i` up to its routed
    /// experts, which run (local or on the Sparks) between segments. Every
    /// post flips the stream buffer: segment 0 flips once (the MLP site),
    /// later layers twice (their attention entry, then the MLP site), and the
    /// final segment once; the parity is the same every step, so replays
    /// track it without running the closures.
    fn decode_graphed(&self, w: &Workspace<'_>, tables: &StepTables, t: usize, rows: Scalar, logit_rows: usize)
        -> Result<Option<Vec<f32>>> {
        let layers = &self.weights.layers;
        let mut cur = 0usize;
        for index in 0..=layers.len() {
            let key = GraphKey { segment: index, rows: t, spec: tables.spec, long: tables.long, pool_width: tables.pool_width,
                page_stride: tables.page_stride, pool_stride: tables.pool_stride };
            let start = cur;
            self.replay(key, || -> Result<()> {
                let mut c = start;
                let Some(layer) = layers.get(index) else {
                    return self.post(w, &mut c, rows);
                };
                let spec = tables.spec;
                if index == 0 {
                    self.enter(w, &mut c, None, layer, index, rows, spec)?;
                } else if self.cfg.ple_layers.contains(&index) {
                    self.post(w, &mut c, rows)?;
                    self.enter(w, &mut c, None, layer, index, rows, spec)?;
                } else {
                    self.enter(w, &mut c, Some(()), layer, index, rows, spec)?;
                }
                match layer.attention {
                    Qwen4Attention::Gdn => self.gdn(w, index, layer, rows, "m64", spec)?,
                    Qwen4Attention::Full => self.full(w, index, layer, rows, "m64", tables)?,
                }
                self.post_pre(w, c, layer, "mlp", rows)?;
                self.moe_front(w, index, layer, t, rows)
            })?;
            cur ^= if index == 0 || index == layers.len() { 1 } else { 0 };
            if index < layers.len() {
                self.moe_experts(w, index, t, rows, true)?;
            }
        }
        self.last_streams.set((true, cur));
        if layers.len() < self.cfg.layers {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        self.logits(w, &w.streams[cur], t, rows, logit_rows).map(Some)
    }

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

    fn site<'l>(layer: &'l Qwen4Layer<'_>, site: &str) -> Result<[*mut c_void; 3]> {
        Ok([layer.ptr(&format!("{site}.norm"))?, layer.ptr(&format!("{site}.w_di"))?,
            layer.ptr(&format!("{site}.w_up"))?])
    }

    /// Into layer `index`'s attention site: with `posted` = None the streams
    /// in `cur` are this layer's input (PLE first when it has one, then
    /// `hc_pre`); with Some the previous MoE output is still in `delta` and
    /// `hc_post_pre` posts it (into the other buffer) and enters.
    #[allow(clippy::too_many_arguments)]
    fn enter(&self, w: &Workspace<'_>, cur: &mut usize, posted: Option<()>, layer: &Qwen4Layer<'_>, index: usize,
        rows: Scalar, spec: bool) -> Result<()> {
        let [norm, di, up] = Self::site(layer, "attn")?;
        if posted.is_some() {
            self.post_pre(w, *cur, layer, "attn", rows)?;
            *cur ^= 1;
            return Ok(());
        }
        if self.cfg.ple_layers.contains(&index) {
            self.ple(w, &w.streams[*cur], layer, rows, spec)?;
        }
        self.run("qwen4_hc_pre", &[("residual", w.streams[*cur].buffer.ptr), ("norm", norm), ("w_di", di),
            ("w_up", up), ("y", w.x.buffer.ptr), ("inject", w.inject.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
            &[rows])
    }

    /// `delta` (the finished sublayer's output) into streams `cur` (written
    /// to the other buffer), then `site`'s input from them.
    fn post_pre(&self, w: &Workspace<'_>, cur: usize, layer: &Qwen4Layer<'_>, site: &str, rows: Scalar) -> Result<()> {
        let [norm, di, up] = Self::site(layer, site)?;
        self.run("qwen4_hc_post_pre", &[("x", w.delta.buffer.ptr), ("residual", w.streams[cur].buffer.ptr),
            ("inject", w.inject.buffer.ptr), ("norm", norm), ("w_di", di), ("w_up", up),
            ("residual_out", w.streams[cur ^ 1].buffer.ptr), ("y", w.x.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
            &[rows])
    }

    /// `delta` into streams `cur` alone (written to the other buffer).
    fn post(&self, w: &Workspace<'_>, cur: &mut usize, rows: Scalar) -> Result<()> {
        self.run("qwen4_hc_post", &[("x", w.delta.buffer.ptr), ("residual", w.streams[*cur].buffer.ptr),
            ("inject", w.inject.buffer.ptr), ("out", w.streams[*cur ^ 1].buffer.ptr)], &[rows])?;
        *cur ^= 1;
        Ok(())
    }

    fn ple(&self, w: &Workspace<'_>, streams: &Dev<'_>, layer: &Qwen4Layer<'_>, rows: Scalar, spec: bool)
        -> Result<()> {
        let table = self.ple.as_ref().context("PLE layer without the n-gram table (--ple)")?;
        let state = self.ple_state.as_ref().context("PLE conv state")?;
        let replay = self.ple_replay.as_ref().context("PLE replay record")?;
        let name = if table.fp8 { "qwen4_ple_fp8" } else { "qwen4_ple_bf16" };
        self.run(name, &[("streams", streams.buffer.ptr), ("ids", w.ple_ids.buffer.ptr), ("table", table.table),
            ("scale", table.scale.buffer.ptr), ("w_kv", layer.ptr("ple.w_kv")?),
            ("norm_key", layer.ptr("ple.norm_key")?), ("norm_query", layer.ptr("ple.norm_query")?),
            ("norm_conv", layer.ptr("ple.norm_conv")?), ("conv_w", layer.ptr("ple.conv_w")?),
            ("conv_state", state.buffer.ptr), ("slots", w.slots.buffer.ptr), ("seq_first", w.seq_first.buffer.ptr),
            ("replay", replay.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows, Scalar::I32(i32::from(spec))])
    }

    /// Decode-shaped steps use the E4M3 programs when the layer carries E4M3 copies.
    fn fp8(&self, layer: &Qwen4Layer<'_>, cap: &str) -> bool {
        cap == "m64" && layer.has("w_in_fp8")
    }

    fn gdn(&self, w: &Workspace<'_>, index: usize, layer: &Qwen4Layer<'_>, rows: Scalar, cap: &str, spec: bool)
        -> Result<()> {
        let ord = self.gdn_ord[index].context("GDN layer without a state pool")?;
        let [conv, state, replay] = self.gdn_pools(ord)?;
        let mut pointers = vec![("x", w.x.buffer.ptr), ("w_in", layer.ptr("w_in")?)];
        let fp8 = self.fp8(layer, cap);
        if fp8 {
            pointers.extend([("w_in_fp8", layer.ptr("w_in_fp8")?), ("w_in_scale", layer.ptr("w_in_scale")?)]);
        }
        pointers.extend([("conv_w", layer.ptr("conv_w")?), ("a_log", layer.ptr("a_log")?),
            ("dt_bias", layer.ptr("dt_bias")?), ("norm_w", layer.ptr("norm_w")?), ("w_out", layer.ptr("w_out")?)]);
        if fp8 {
            pointers.extend([("w_out_fp8", layer.ptr("w_out_fp8")?), ("w_out_scale", layer.ptr("w_out_scale")?)]);
        }
        pointers.extend([("conv_state", conv), ("state", state), ("slots", w.slots.buffer.ptr),
            ("seq_first", w.seq_first.buffer.ptr), ("out", w.delta.buffer.ptr)]);
        // Decode capacities record speculative replay inputs (spec) or advance the state.
        let decode = cap == "m64";
        if decode {
            pointers.push(("replay", replay));
        }
        pointers.push(("scratch", w.scratch.buffer.ptr));
        let name = if fp8 { format!("qwen4_gdn_fp8_{cap}") } else { format!("qwen4_gdn_{cap}") };
        if decode {
            self.run(&name, &pointers, &[rows, Scalar::I32(i32::from(spec))])
        } else {
            ensure!(!spec, "speculative steps take the decode programs");
            self.run(&name, &pointers, &[rows])
        }
    }

    fn full(&self, w: &Workspace<'_>, index: usize, layer: &Qwen4Layer<'_>, rows: Scalar, cap: &str,
        tables: &StepTables) -> Result<()> {
        let cache = self.kv[index].as_ref().context("full attention layer without a record pool")?;
        let (keys, blocks) = self.index[index].as_ref().context("full attention layer without an index cache")?;
        self.attend(w, layer, cache.buffer.ptr, keys.buffer.ptr, blocks.buffer.ptr, rows, cap, tables)
    }

    /// Full attention of `layer` over the record pool `cache`, raw index keys
    /// `keys` and pooled block keys `blocks` (a target layer's or the MTP layer's).
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, w: &Workspace<'_>, layer: &Qwen4Layer<'_>, cache: *mut c_void, keys: *mut c_void,
        blocks: *mut c_void, rows: Scalar, cap: &str, tables: &StepTables) -> Result<()> {
        let fp8 = self.fp8(layer, cap);
        let mut pointers = vec![("x", w.x.buffer.ptr), ("w_in", layer.ptr("w_in")?)];
        if fp8 {
            pointers.extend([("w_in_fp8", layer.ptr("w_in_fp8")?), ("w_in_scale", layer.ptr("w_in_scale")?)]);
        }
        pointers.extend([("q_norm", layer.ptr("q_norm")?), ("k_norm", layer.ptr("k_norm")?),
            ("iq_norm", layer.ptr("iq_norm")?), ("ik_norm", layer.ptr("ik_norm")?),
            ("positions", w.positions.buffer.ptr), ("kv_slots", w.kv_slots.buffer.ptr),
            ("pool_slots", w.pool_slots.buffer.ptr), ("kv_cache", cache), ("token_keys", keys),
            ("index_cache", blocks), ("query", w.query.buffer.ptr), ("gate", w.gate.buffer.ptr),
            ("index_q", w.index_q.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
        let name = if fp8 { format!("qwen4_attn_producer_fp8_{cap}") } else { format!("qwen4_attn_producer_{cap}") };
        self.run(&name, &pointers, &[rows])?;
        if tables.long {
            self.run(&format!("qwen4_index_topk_{cap}"), &[("index_q", w.index_q.buffer.ptr),
                ("positions", w.positions.buffer.ptr), ("index_cache", blocks),
                ("page_table", w.pool_table.buffer.ptr), ("output_indices", w.blocks.buffer.ptr),
                ("scratch", w.topk_scratch.buffer.ptr)], &[rows, Scalar::I32(tables.pool_stride as i32)])?;
        }
        self.run("qwen4_index_expand", &[("positions", w.positions.buffer.ptr), ("blocks", w.blocks.buffer.ptr),
            ("indices", w.indices.buffer.ptr), ("lengths", w.lengths.buffer.ptr)], &[rows])?;
        self.run(&format!("qwen4_sparse_gqa_{cap}"), &[("query", w.query.buffer.ptr), ("kv_cache", cache),
            ("positions", w.positions.buffer.ptr), ("page_table", w.page_table.buffer.ptr),
            ("indices", w.indices.buffer.ptr), ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
            &[rows, Scalar::I32(tables.page_width as i32), Scalar::I32(tables.page_stride as i32)])?;
        let mut pointers = vec![("attn", w.attn.buffer.ptr), ("gate", w.gate.buffer.ptr), ("w_o", layer.ptr("w_o")?)];
        if fp8 {
            pointers.extend([("w_o_fp8", layer.ptr("w_o_fp8")?), ("w_o_scale", layer.ptr("w_o_scale")?)]);
        }
        pointers.extend([("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
        let name = if fp8 { format!("qwen4_attn_o_fp8_{cap}") } else { format!("qwen4_attn_o_{cap}") };
        self.run(&name, &pointers, &[rows])
    }

    /// Router, shared expert and routed experts; leaves `bf16(routed + shared)` in `delta`.
    fn moe(&self, w: &Workspace<'_>, index: usize, layer: &Qwen4Layer<'_>, t: usize, rows: Scalar, decode: bool)
        -> Result<()> {
        self.moe_front(w, index, layer, t, rows)?;
        self.moe_experts(w, index, t, rows, decode)
    }

    /// Router logits, the softmax top-10, the shared expert (into `shared`)
    /// and, for wire-fed experts, the FP8 K32 wire rows. No host sync.
    fn moe_front(&self, w: &Workspace<'_>, index: usize, layer: &Qwen4Layer<'_>, t: usize, rows: Scalar)
        -> Result<()> {
        let h = self.cfg.hidden;
        let experts = self.experts.as_ref().with_context(|| format!(
            "layer {index} needs routed experts: pass --local-experts, Spark --peers or --shared-only"))?;
        self.run("qwen4_router_scores", &[("x", w.x.buffer.ptr), ("w", layer.ptr("gate")?),
            ("logits", w.router_logits.buffer.ptr)], &[rows])?;
        // SAFETY: logits and route outputs are live buffers of `t` rows.
        unsafe {
            self.library.router_select_softmax(w.router_logits.buffer.ptr, w.route_ids.buffer.ptr,
                w.route_weights.buffer.ptr, t, self.cfg.experts, self.cfg.topk, 1.0, true, self.stream)?;
        }
        // Spark layers run the shared expert during the exchange (spark_moe).
        if !matches!(experts, Experts::Spark { .. }) {
            self.shared(w, layer, rows)?;
        }
        if matches!(experts, Experts::LocalExl3(_) | Experts::Spark { .. }) {
            let grid = (t * h.div_ceil(256)).div_ceil(8).clamp(1, 4 * 188);
            self.run("qwen4_expert_input_quant", &[("source_ptr", w.x.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
                // SAFETY: the scale rows follow the payload inside each wire row.
                ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
                ("scale_mma_ptr", w.delta.buffer.ptr)], &[rows, Scalar::I32(grid as i32)])?;
        }
        Ok(())
    }

    fn shared(&self, w: &Workspace<'_>, layer: &Qwen4Layer<'_>, rows: Scalar) -> Result<()> {
        self.run("qwen4_shared", &[("x", w.x.buffer.ptr), ("w_gate_up", layer.ptr("shared.w_gate_up")?),
            ("w_down", layer.ptr("shared.w_down")?), ("out", w.shared.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
            &[rows])
    }

    /// The routed experts of layer `index` (the front ran); leaves
    /// `bf16(routed + shared)` in `delta`.
    fn moe_experts(&self, w: &Workspace<'_>, index: usize, t: usize, rows: Scalar, decode: bool) -> Result<()> {
        let h = self.cfg.hidden;
        match self.experts.as_ref().context("MoE layer without experts")? {
            Experts::Local(local) => {
                self.exchange_window(index, decode, true)?;
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
                self.exchange_window(index, decode, true)?;
                local.ensure(index, self.stream)?;
                let mut resident = local.resident.borrow_mut();
                let (_, experts) = resident.as_mut().context("EXL3 window")?;
                let layer = if index == self.cfg.layers {
                    crate::families::deepseek_v4::local::LocalLayer::Stage(0)
                } else {
                    crate::families::deepseek_v4::local::LocalLayer::Backbone(index)
                };
                // SAFETY: wire rows, routes and the shared-expert rows are complete in
                // stream order; the output is copied before the window can change.
                unsafe {
                    experts.run(layer, t, w.wire.buffer.ptr,
                        w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, w.shared.buffer.ptr, self.stream)?;
                    self.library.copy_d2d_async(w.delta.buffer, experts.output.buffer, t * h * 2, self.stream)?;
                }
                return Ok(());
            }
            Experts::Spark { transport, runtime } => {
                ensure!(index < self.cfg.layers, "the Spark ranks do not serve the MTP layer's experts");
                return self.spark_moe(w, index, t, rows, decode, &mut transport.borrow_mut(), runtime);
            }
            Experts::SharedOnly => {
                // SAFETY: both are live [t, H] BF16 buffers ordered on the stream.
                unsafe { self.library.copy_d2d_async(w.delta.buffer, w.shared.buffer, t * h * 2, self.stream)? };
                return Ok(());
            }
        }
        self.run("qwen4_add", &[("a", w.routed.buffer.ptr), ("b", w.shared.buffer.ptr), ("out", w.delta.buffer.ptr)],
            &[rows])
    }

    /// Routes and wire rows down, one request to every Spark rank, the BF16
    /// rank partials and the shared expert summed into `delta`.
    #[allow(clippy::too_many_arguments)]
    fn spark_moe(&self, w: &Workspace<'_>, index: usize, t: usize, rows: Scalar, decode: bool,
        transport: &mut SparkLink<'_>,
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
            self.library.cuda_event_record(self.routes_ready, self.stream)?;
        }
        // The shared expert queues behind the copies and runs during the
        // exchange (the L2 prefetch behind it); the host waits for the copies only.
        self.shared(w, &self.weights.layers[index], rows)?;
        self.exchange_window(index, decode, false)?;
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
        let ranks = transport.world_size();
        ensure!(ranks <= MAX_RANKS, "{ranks} Spark ranks exceed the reduction planes");
        let timer = std::time::Instant::now();
        runtime.block_on(async {
            let wave = transport.dispatch(&request)?;
            transport.receive(wave, t, self.stream).await
        })?;
        self.profile.borrow_mut()[1] += timer.elapsed().as_secs_f64();
        // SAFETY: the shared-expert plane and `delta` are live [t, h] BF16
        // buffers; the intake planes are ordered after the wave by `receive`.
        unsafe {
            transport.reduce(w.shared.buffer.ptr.cast(), w.delta.buffer.ptr.cast(), t, self.stream)?;
            self.library.cuda_stream_synchronize(self.stream)
        }
    }
}

impl Drop for Qwen4Engine<'_> {
    fn drop(&mut self) {
        // SAFETY: the engine's stream is drained by its owner before the engine drops.
        let _ = unsafe { self.library.cuda_event_destroy(self.routes_ready) };
    }
}
