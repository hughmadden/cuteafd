//! MiMo V2 (mimo_v2_flash, V2.6 Pro mimo_v2) coordinator over the exported
//! mimo_* (Flash) or mimop_* (V2.6 Pro) programs.
//!
//! One layer: input norm (fused with the previous layer's residual add),
//! the QKV producer (RoPE, KV record), GQA attention, o_proj, the
//! post-attention norm, then the dense MLP or the MoE: router scores (FP32
//! weight as BF16 hi + lo), the native sigmoid top-8 select, FP8 K32 wire
//! rows, then the routed experts in the checkpoint's own FP8 (the `fp8`
//! family): on the Sparks (one BF16 partial per TP rank, summed on the GPU,
//! as GLM) or, with local experts, on this GPU from the TP1 package.
//!
//! KV state: full layers keep one record per token (keys then values of the 4
//! KV heads) in a paged pool shared by sequences (64 rows per page); SWA layers
//! keep a 256-slot ring per sequence (8 KV heads). Full-attention records are
//! int8 by default (an FP32 scale `amax / 127` per 32 dims of each head's key and
//! value: 1440 bytes on V2 Flash) or BF16 (`--kv-cache bf16`: 2560 bytes); the
//! `full_*_kvint8` programs read and write the int8 layout, and int8 prefill widens
//! the sequence's records once into `kv_wide`. SWA records are BF16. An SWA
//! step's records go to a step buffer first; the attention program reads
//! in-step keys from it, older keys from the ring, and commits the step to the
//! ring afterwards.
use super::weights::{MimoLayer, MimoWeights};
use crate::shared::experts::fp8::{Fp8Experts, Fp8Layer};
use cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors;
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use crate::shared::token_io::{DeviceLogits, TokenEmbedding};
use crate::shared::spark_intake::SparkLink;
use cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Programs, Scalar, VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::mimo_v2::{MimoAttention, MimoKvCache, MimoV2Config};
use crate::shared::peer_split::{PeerExchange, RankDevice, DIRECT};
use std::cell::RefCell;
use std::ffi::c_void;

type Dev<'a> = DeviceAllocation<'a>;

pub(crate) const PAGE_ROWS: usize = 64;
pub(crate) const RING_ROWS: usize = 256;
/// Rows of the decode-route programs (`_m64`).
pub(crate) const DECODE_ROWS: usize = 64;
/// Programs of one layer's attention: the qkv producer, attention, o_proj.
const ATTENTION_PARTS: usize = 3;

/// Most Spark ranks a step's partials come from (the compact reducer's limit).
const MAX_RANKS: usize = 6;

/// What the coordinator sends the Spark ranks as expert input rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum ExpertInput {
    /// FP8 K32 wire rows (E4M3 + UE8M0 per 32; `H + H/32` bytes per row).
    Fp8,
    /// The BF16 rows themselves (`2H` bytes; the ranks need the `-bf16` package).
    Bf16,
    /// BF16 for decode-shaped steps, FP8 wire rows for prefill.
    Bf16Decode,
}

impl ExpertInput {
    pub fn bf16(self, decode: bool) -> bool {
        match self {
            Self::Fp8 => false,
            Self::Bf16 => true,
            Self::Bf16Decode => decode,
        }
    }
}

/// Where the routed experts run.
pub(crate) enum Experts<'a> {
    /// The TP1 FP8 package on the coordinator GPU (resident MoE layers).
    Local(Fp8Experts<'a>),
    /// The TP1 package on the coordinator GPU with a window of resident MoE
    /// layers, loading each missing layer over the oldest (the model's experts
    /// do not fit: MiMo V2.6 Pro's are 495 GiB). For prefill checks; a decode
    /// step would reload every layer.
    Streamed { experts: RefCell<Fp8Experts<'a>>, tensors: &'a Fp8ExpertTensors, window: usize },
    /// No routed experts: MoE layers add zero (coordinator timing only; the
    /// outputs are not the model's).
    Skip,
    /// Spark ranks serving the `fp8` family over RoCE.
    Spark { transport: RefCell<SparkLink<'a>>, runtime: tokio::runtime::Runtime },
}

/// Most rows the FP8 LM head program takes (MmaFp8Gemv's M tile).
pub(crate) const FP8_ROWS: i32 = 16;

/// Most rows a decode program reads the E4M3 qkv / o / dense FFN copies for
/// (sparkinfer `MIMO_FP8_ROWS`: one 16-row GEMV tile up to 16 rows, two above),
/// so DFlash verify steps of up to 32 rows stay off the BF16 projections.
pub(crate) const FP8_DECODE_ROWS: i32 = 32;

/// Most rows of one sequence an MTP drafting pass takes (older true rows
/// catch up in passes of their own first).
const MTP_STEP_ROWS: usize = 16;

/// KV splits of the full-attention decode program (its compiled maximum is 32).
const DECODE_SPLITS: i32 = 16;

/// Host tables of one step.
struct StepTables {
    decode: bool,
    positions: Vec<i64>,
    /// Full layers: paged record slot per row.
    slots: Vec<i64>,
    /// SWA layers: ring slot per row (ring * 256 + position % 256).
    ring_slots: Vec<i64>,
    /// Index of the first step row of each row's sequence.
    seq_first: Vec<i32>,
    /// Prefill: the sequence's pages; decode: one padded row per step row.
    page_table: Vec<i32>,
    table_stride: usize,
}

/// A sequence's full-attention pages (from the refcounted pool: full pages may be
/// shared with retained prefix snapshots and other sequences, which never write them),
/// its SWA ring and its length.
#[derive(Debug, Clone)]
pub(crate) struct MimoPlacement {
    pub pages: Vec<u32>,
    pub ring: i32,
    pub len: usize,
}

impl MimoPlacement {
    pub fn slot(&self, position: usize) -> Result<i64> {
        let page = *self.pages.get(position / PAGE_ROWS).context("position past the sequence's pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + (position % PAGE_ROWS) as i64)
    }

    pub fn ring_slot(&self, position: usize) -> i64 {
        i64::from(self.ring) * RING_ROWS as i64 + (position % RING_ROWS) as i64
    }
}

/// Pages of the full-attention pool (the refcounted pool the prefix cache shares) and
/// free SWA rings, for callers without a prefix cache (the golden command).
pub(crate) struct Allocator {
    pages: cuteafd_engine::prefix::RefPagePool,
    rings: Vec<i32>,
}

impl Allocator {
    pub fn new(pages: usize, rings: usize) -> Self {
        Self { pages: cuteafd_engine::prefix::RefPagePool::new(pages, PAGE_ROWS), rings: (0..rings as i32).rev().collect() }
    }

    /// Reserves every page a sequence of up to `capacity` tokens needs, and a ring.
    pub fn admit(&mut self, capacity: usize) -> Result<MimoPlacement> {
        let pages = self.pages.alloc(self.pages.pages_for(capacity))?;
        let Some(ring) = self.rings.pop() else {
            self.pages.release(&pages);
            anyhow::bail!("SWA rings exhausted");
        };
        Ok(MimoPlacement { pages, ring, len: 0 })
    }

    /// A second sequence starting as `source`'s first `len` rows: full pages shared, the
    /// partial tail page copied into its own page by the caller (the returned copy).
    pub fn fork(&mut self, source: &MimoPlacement, len: usize, capacity: usize)
        -> Result<(MimoPlacement, Option<cuteafd_engine::prefix::TailCopy>)> {
        let fork = self.pages.fork(&source.pages, len, self.pages.pages_for(capacity))?;
        let Some(ring) = self.rings.pop() else {
            self.pages.release(&fork.pages);
            anyhow::bail!("SWA rings exhausted");
        };
        Ok((MimoPlacement { pages: fork.pages, ring, len: 0 }, fork.copy))
    }

    /// Returns a finished sequence's pages and ring.
    pub fn release(&mut self, placement: MimoPlacement) {
        self.pages.release(&placement.pages);
        self.rings.push(placement.ring);
    }
}

struct Workspace<'a> {
    rows: usize,
    h: Dev<'a>,
    x: Dev<'a>,
    query: Dev<'a>,
    attn: Dev<'a>,
    delta: Dev<'a>,
    kv_step: Dev<'a>,
    /// 8-bit KV prefill: BF16 copy of one sequence's full-attention records (`max_context` rows).
    kv_wide: Dev<'a>,
    positions: Dev<'a>,
    slots: Dev<'a>,
    step_slots: Dev<'a>,
    ring_slots: Dev<'a>,
    seq_first: Dev<'a>,
    page_table: Dev<'a>,
    scratch: Dev<'a>,
    logits: Dev<'a>,
    router_logits: Dev<'a>,
    route_ids: Dev<'a>,
    route_weights: Dev<'a>,
    wire: Dev<'a>,
    /// Spark exchange: rank partial planes, a zero shared-expert plane (MiMo
    /// has none), and pinned staging for routes, wire rows and partials.
    zero_plane: Dev<'a>,
    router_host: RefCell<HostAllocation<'a>>,
    /// The step's token ids (U32), gathered from the device embedding table.
    ids: Dev<'a>,
    /// Greedy selection of logits rows (MTP drafts): U32 ids, then U32 statuses.
    select: Dev<'a>,
    /// The cuBLAS LM head (rank 0 only).
    head: Option<VocabularyHead<'a>>,
    _head_workspace: Dev<'a>,
}

/// The second GPU of a two-GPU head split (see `MimoV2Config::head_split`):
/// its share of every layer (half the query heads with the KV heads they read,
/// half the dense MLP), its KV records, RoPE tables and workspaces.
pub(crate) struct Peer<'a> {
    pub device: i32,
    pub stream: *mut c_void,
    pub layers: Vec<MimoLayer<'a>>,
    kv: Vec<Dev<'a>>,
    cos_sin_full: Dev<'a>,
    cos_sin_swa: Dev<'a>,
    workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
}

/// Exchange slot of layer `index`'s attention partials (`ffn` false) or its
/// FFN partials / routed-expert sum: by layer parity, so a push for layer `l`
/// never lands on rows the other GPU may still read (it consumed layer `l - 2`'s
/// before it sent layer `l - 1`'s attention partial, which this GPU waited for).
fn slot(index: usize, ffn: bool) -> usize {
    2 * (index % 2) + usize::from(ffn)
}

/// cos | sin of position * theta^(-2i/dim) for `max_context` positions, FP32
/// like the reference's inv_freq, on the current device.
fn rope_table<'a>(library: &'a NativeLibrary, dim: usize, theta: f64, max_context: usize) -> Result<Dev<'a>> {
    let inv: Vec<f32> = (0..dim / 2).map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / dim as f32)).collect();
    let mut values = vec![0f32; max_context * dim];
    for p in 0..max_context {
        for (i, f) in inv.iter().enumerate() {
            let angle = p as f32 * f;
            values[p * dim + i] = angle.cos();
            values[p * dim + dim / 2 + i] = angle.sin();
        }
    }
    let allocation = DeviceAllocation::new(library, values.len() * 4)?;
    library.copy_h2d(allocation.buffer, bytes_of(&values))?;
    Ok(allocation)
}

pub(crate) struct MimoEngine<'a> {
    pub library: &'a NativeLibrary,
    pub programs: &'a Programs<'a>,
    pub cfg: MimoV2Config,
    pub weights: MimoWeights<'a>,
    pub stream: *mut c_void,
    pub max_context: usize,
    pub prefill_rows: usize,
    pub pages: usize,
    pub rings: usize,
    /// Program family of the checkpoint's geometry (`mimo`, `mimop`).
    family: &'static str,
    /// This engine's GPU (rank 0 of a head split).
    pub device: i32,
    /// Program family of one GPU's share under a head split (`mimop2`): the
    /// layers' producer, attention, o_proj and dense MLP programs.
    split_family: Option<&'static str>,
    /// The head split's second GPU and both ends of its exchange (rank 0's, the peer's).
    peer: Option<Peer<'a>>,
    exchange: Option<PeerExchange<'a>>,
    /// Per layer: the paged record pool (full) or the rings (SWA).
    kv: Vec<Dev<'a>>,
    cos_sin_full: Dev<'a>,
    cos_sin_swa: Dev<'a>,
    workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    experts: Option<Experts<'a>>,
    pub expert_input: ExpertInput,
    /// Host time per phase: GPU wait before the expert request, the Spark exchange.
    pub profile: RefCell<[f64; 2]>,
    /// `CUTEAFD_MIMO_WAVE_TIMING=1`: one line per Spark wave with its host
    /// phases (diagnostics; read once at construction).
    wave_timing: bool,
    /// The DFlash drafter (V2.6 Pro's dflash/): every step taps its target layers.
    pub drafter: Option<super::dflash::MimoDrafter<'a>>,
    /// L2 prefetch of the next layer's weights during decode exchanges.
    pub l2: Option<crate::shared::l2_prefetch::L2Prefetch>,
    /// The native MTP drafter: every step taps the last layer's rows.
    pub mtp: Option<super::mtp::MtpDrafter<'a>>,
    /// The token embedding table (resident on this GPU or read from its shard).
    pub embedding: TokenEmbedding<'a>,
    /// Pinned staging of queued MTP passes' tables (async uploads) and its fill level.
    mtp_staging: RefCell<(HostAllocation<'a>, usize)>,
    /// Prefill qkv and dense-FFN projections run W8A8 (E4M3 activations per
    /// row and 128-K block, the official FP8 release's served numerics); false:
    /// W8A16 (bitwise the former BF16 prefill over dequantized weights).
    pub prefill_w8a8: bool,
    /// KV record format of every layer (and the MTP rings).
    kv_cache: MimoKvCache,
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// `[rows]`, plus the decode programs' `fp8_rows` (`FP8_DECODE_ROWS` when the layer has the FP8 copy, else 0).
fn fp8_scalars(rows: Scalar, decode: bool, fp8: bool) -> Vec<Scalar> {
    let mut scalars = vec![rows];
    if decode {
        scalars.push(Scalar::I32(if fp8 { FP8_DECODE_ROWS } else { 0 }));
    }
    scalars
}

/// An FP8-only weight's scales: row major for decode programs, K-block major for prefill ones.
fn scale(name: &str, decode: bool, layer: &MimoLayer<'_>) -> Result<(&'static str, *mut c_void)> {
    let operand: &'static str = match (name, decode) {
        ("w_qkv", true) => "w_qkv_scale",
        ("w_qkv", false) => "w_qkv_kscale",
        ("w_gate_up", true) => "w_gate_up_scale",
        ("w_gate_up", false) => "w_gate_up_kscale",
        ("w_down", true) => "w_down_scale",
        _ => "w_down_kscale",
    };
    Ok((operand, layer.ptr(operand)?))
}

fn kind(attention: MimoAttention) -> &'static str {
    match attention {
        MimoAttention::Full => "full",
        MimoAttention::Sliding => "swa",
    }
}

impl<'a> MimoEngine<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, cfg: MimoV2Config,
        weights: MimoWeights<'a>, stream: *mut c_void, max_context: usize, prefill_rows: usize, pages: usize,
        rings: usize, embedding: TokenEmbedding<'a>, kv_cache: MimoKvCache) -> Result<Self> {
        let family = cfg.program_family()?;
        // Layers loaded as head-split shares carry half the heads (see `attach_peer`).
        let ranks = if weights.layers.iter().any(|l| l.split) { 2 } else { 1 };
        let share = cfg.head_split(ranks)?;
        let split_family = if ranks > 1 { Some(share.program_family()?) } else { None };
        let device = library.cuda_get_device()?;
        ensure!(embedding.hidden() == cfg.hidden, "embedding rows of {} for hidden {}", embedding.hidden(), cfg.hidden);
        ensure!(cfg.rope_dim == 64 && cfg.head_dim == 192 && cfg.v_head_dim == 128 && cfg.window <= RING_ROWS - DECODE_ROWS,
            "the mimo programs are built for 192/128 heads, 64 RoPE dims and a window of at most {}",
            RING_ROWS - DECODE_ROWS);
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let kv = weights.layers.iter().map(|layer| {
            let record = if layer.split { &share } else { &cfg }.record_bytes(layer.attention, kv_cache);
            zeroed(match layer.attention {
                MimoAttention::Full => pages * PAGE_ROWS * record,
                MimoAttention::Sliding => rings * RING_ROWS * record,
            })
        }).collect::<Result<Vec<_>>>()?;
        let table = |theta: f64| rope_table(library, cfg.rope_dim, theta, max_context);
        let (cos_sin_full, cos_sin_swa) = (table(cfg.full_rope_theta)?, table(cfg.swa_rope_theta)?);
        Ok(Self { library, programs, cfg, weights, stream, max_context, prefill_rows, pages, rings, family, device,
            split_family, peer: None, exchange: None, kv, cos_sin_full,
            cos_sin_swa, workspace: RefCell::new(None), decode_workspace: RefCell::new(None), experts: None,
            expert_input: ExpertInput::Fp8,
            profile: RefCell::new([0.0; 2]),
            wave_timing: std::env::var("CUTEAFD_MIMO_WAVE_TIMING").is_ok_and(|v| v == "1"), drafter: None, mtp: None, l2: None, embedding,
            mtp_staging: RefCell::new((HostAllocation::new(library, 1 << 20)?, 0)), prefill_w8a8: true, kv_cache })
    }

    /// Layer `layer`'s KV storage: the paged record pool (full attention; page `p` holds
    /// rows at `p * PAGE_ROWS * record`) or the SWA rings (ring `r` at `r * RING_ROWS *
    /// record`), with its record bytes.
    pub(crate) fn kv_layer(&self, layer: usize) -> (MimoAttention, cuteafd_ffi::CuteafdDeviceBuffer, usize) {
        self.kv_layer_on(0, layer)
    }

    /// [`Self::kv_layer`] of rank `rank`'s share (0: this GPU, 1: the head split's peer).
    pub(crate) fn kv_layer_on(&self, rank: usize, layer: usize) -> (MimoAttention, cuteafd_ffi::CuteafdDeviceBuffer, usize) {
        let (layers, kv) = match (rank, &self.peer) {
            (1, Some(peer)) => (&peer.layers, &peer.kv),
            _ => (&self.weights.layers, &self.kv),
        };
        let attention = layers[layer].attention;
        (attention, kv[layer].buffer, self.record_bytes(&layers[layer]))
    }

    /// The KV record format.
    pub fn kv_cache(&self) -> MimoKvCache {
        self.kv_cache
    }

    /// Bytes of `layer`'s KV record on the GPU holding it (its KV heads only under a head split).
    fn record_bytes(&self, layer: &MimoLayer<'_>) -> usize {
        let heads = self.cfg.kv_heads(layer.attention) / if layer.split { 2 } else { 1 };
        self.cfg.record_bytes_of(heads, self.kv_cache.of(layer.attention))
    }

    /// GPUs this engine runs on: 2 under a head split.
    pub fn ranks(&self) -> usize {
        1 + usize::from(self.peer.is_some())
    }

    /// Attaches the head split's second GPU: `device` with `stream`, holding
    /// `layers` (every layer's rank-1 share, see `MimoLoader::model`). Enables
    /// peer access both ways, loads the programs there, and allocates its KV
    /// records, RoPE tables and both ends of the exchange.
    pub fn attach_peer(&mut self, device: i32, stream: *mut c_void, layers: Vec<MimoLayer<'a>>) -> Result<()> {
        ensure!(self.split_family.is_some() && layers.len() == self.weights.layers.len()
            && layers.iter().all(|l| l.split), "attach_peer needs the head-split shares of every loaded layer");
        let library = self.library;
        let rows = self.prefill_rows.max(DECODE_ROWS);
        let exchange = PeerExchange::new(library, [RankDevice { device: self.device, stream: self.stream },
            RankDevice { device, stream }], 4, rows * self.cfg.hidden * 2)?;
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let peer = exchange.on(1, || -> Result<Peer<'a>> {
            self.programs.load_all()?;
            let kv = layers.iter().map(|layer| {
                let record = self.record_bytes(layer);
                zeroed(match layer.attention {
                    MimoAttention::Full => self.pages * PAGE_ROWS * record,
                    MimoAttention::Sliding => self.rings * RING_ROWS * record,
                })
            }).collect::<Result<Vec<_>>>()?;
            let table = |theta: f64| rope_table(library, self.cfg.rope_dim, theta, self.max_context);
            Ok(Peer { device, stream, kv, cos_sin_full: table(self.cfg.full_rope_theta)?,
                cos_sin_swa: table(self.cfg.swa_rope_theta)?, layers, workspace: RefCell::new(None),
                decode_workspace: RefCell::new(None) })
        })?;
        self.peer = Some(peer);
        self.exchange = Some(exchange);
        Ok(())
    }

    /// The stream of rank `rank`.
    pub(crate) fn stream_of(&self, rank: usize) -> *mut c_void {
        match (rank, &self.peer) {
            (1, Some(peer)) => peer.stream,
            _ => self.stream,
        }
    }

    /// Runs `body` with rank `rank`'s device current (this engine's device again after).
    pub(crate) fn on<T>(&self, rank: usize, body: impl FnOnce() -> Result<T>) -> Result<T> {
        let Some(peer) = self.peer.as_ref().filter(|_| rank == 1) else { return body() };
        self.library.cuda_set_device(peer.device)?;
        let out = body();
        self.library.cuda_set_device(self.device)?;
        out
    }

    /// Receive slot `slot` of rank `rank` (null without a head split).
    fn recv(&self, rank: usize, slot: usize) -> *mut c_void {
        self.exchange.as_ref().and_then(|e| e.recv(rank, slot).ok()).unwrap_or(std::ptr::null_mut())
    }

    fn exchange(&self) -> Result<&PeerExchange<'a>> {
        self.exchange.as_ref().context("no head-split exchange")
    }

    /// Queues on `from`'s stream: push `bytes` of `source` into the other GPU's slot `slot`.
    fn push(&self, from: usize, slot: usize, source: *const c_void, bytes: usize) -> Result<()> {
        self.exchange()?.push(from, slot, source, bytes)
    }

    /// Queues on `at`'s stream: wait for the other GPU's next push into slot `slot`.
    fn wait(&self, at: usize, slot: usize) -> Result<()> {
        self.exchange()?.wait(at, slot)
    }


    /// Serves MoE layers from `experts` (without, the engine stops at the first MoE layer).
    pub fn set_experts(&mut self, experts: Experts<'a>) {
        self.experts = Some(experts);
    }

    pub fn has_experts(&self) -> bool {
        self.experts.is_some()
    }

    fn alloc(&self, bytes: usize) -> Result<Dev<'a>> {
        DeviceAllocation::new(self.library, bytes.max(256))
    }

    /// `mimo_*` program names of this checkpoint's program family (`mimo` for
    /// V2 Flash, `mimop` for V2.6 Pro).
    fn program_name(&self, name: &str, split: bool) -> String {
        let family = if split { self.split_family.unwrap_or(self.family) } else { self.family };
        match name.strip_prefix("mimo_") {
            Some(rest) => format!("{family}_{rest}"),
            None => name.to_string(),
        }
    }

    fn run(&self, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar]) -> Result<()> {
        self.run_on(0, false, name, pointers, scalars)
    }

    /// Launches `name` on rank `rank`'s stream; `split` picks the head-split share's program.
    fn run_on(&self, rank: usize, split: bool, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar])
        -> Result<()> {
        let name = &self.program_name(name, split);
        let names: Vec<&str> = pointers.iter().map(|(n, _)| *n).collect();
        let program = self.programs.program(name, &names)?;
        let raw: Vec<*mut c_void> = pointers.iter().map(|(_, p)| *p).collect();
        // SAFETY: every pointer names a live allocation of rank `rank`'s GPU sized for
        // the rows in `scalars`; that rank's stream orders all its launches.
        self.on(rank, || unsafe { program.launch(&raw, scalars, self.stream_of(rank)) })
            .with_context(|| format!("{name} with {scalars:?}"))
    }

    fn scratch(&self, name: &str, split: bool) -> Result<usize> {
        Ok(self.programs.spec(&self.program_name(name, split))?.scratch.get("scratch").copied().unwrap_or(0) as usize)
    }

    /// Rank `rank`'s workspace for steps of up to `t` rows (rank 1 holds the
    /// attention-side buffers only); allocated on that rank's GPU.
    fn workspace(&self, rank: usize, t: usize, decode: bool) -> Result<Workspace<'a>> {
        self.on(rank, || self.workspace_here(rank, t, decode))
    }

    fn workspace_here(&self, rank: usize, t: usize, decode: bool) -> Result<Workspace<'a>> {
        let (h, heads) = (self.cfg.hidden, self.cfg.heads);
        let (cap, mode) = if decode { ("m64", "decode") } else { ("m4096", "prefill") };
        let lead = rank == 0;
        let mut scratch = if lead { self.scratch("mimo_router_scores", false)? } else { 0 };
        // Whole-model programs (MTP layers on rank 0) and the split share's.
        let families: &[bool] = match (self.split_family, lead) {
            (None, _) => &[false],
            (Some(_), true) => &[false, true],
            (Some(_), false) => &[true],
        };
        for &split in families {
            let kv = self.kv_cache.program_tag();
            for name in [format!("mimo_full_producer{kv}_{cap}"), format!("mimo_swa_producer_{cap}"),
                format!("mimo_full_attention{kv}_{mode}_{cap}"), format!("mimo_swa_attention_{mode}_{cap}"),
                format!("mimo_ffn_{cap}")] {
                scratch = scratch.max(self.scratch(&name, split)?);
            }
        }
        let head_workspace = self.alloc(if lead { VOCABULARY_HEAD_WORKSPACE } else { 256 })?;
        let spark = lead && matches!(self.experts, Some(Experts::Spark { .. }));
        let identity: Vec<i64> = (0..t as i64).collect();
        let step_slots = self.alloc(t * 8)?;
        self.library.copy_h2d(step_slots.buffer, bytes_of(&identity))?;
        let record = self.cfg.record_bytes(MimoAttention::Sliding, self.kv_cache)
            .max(self.cfg.record_bytes(MimoAttention::Full, self.kv_cache));
        // Rank 1 never runs the router, experts, head or drafters.
        let lead_only = |bytes: usize| if lead { bytes } else { 256 };
        Ok(Workspace {
            rows: t,
            h: self.alloc(t * h * 2)?,
            x: self.alloc(t * h * 2)?,
            query: self.alloc(t * heads * self.cfg.head_dim * 2)?,
            attn: self.alloc(t * heads * self.cfg.v_head_dim * 2)?,
            delta: self.alloc(t * h * 2)?,
            kv_step: self.alloc(t * record)?,
            kv_wide: self.alloc(if decode || self.kv_cache == MimoKvCache::Bf16 { 256 } else {
                self.max_context * self.cfg.record_bytes(MimoAttention::Full, MimoKvCache::Bf16) })?,
            positions: self.alloc(t * 8)?,
            slots: self.alloc(t * 8)?,
            step_slots,
            ring_slots: self.alloc(t * 8)?,
            seq_first: self.alloc(t * 4)?,
            page_table: self.alloc(if decode { t * self.pages * 4 } else { self.pages * 4 })?,
            scratch: self.alloc(scratch)?,
            logits: self.alloc(lead_only(t * self.cfg.vocab_size * 4))?,
            router_logits: self.alloc(lead_only(t * self.cfg.experts * 4))?,
            route_ids: self.alloc(lead_only(t * self.cfg.topk * 4))?,
            route_weights: self.alloc(lead_only(t * self.cfg.topk * 4))?,
            wire: self.alloc(lead_only(t * (h + h / 32)))?,
            zero_plane: {
                let zero = self.alloc(if spark { t * h * 2 } else { 256 })?;
                self.library.cuda_zero_bytes(zero.buffer, zero.buffer.bytes)?;
                zero
            },
            router_host: RefCell::new(HostAllocation::new(self.library,
                if spark { t * (self.cfg.topk * 8 + 2 * h) } else { 256 })?),
            ids: self.alloc(t * 4)?,
            select: self.alloc(t * 8)?,
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: if lead {
                Some(unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, h as u32, t as u32,
                    self.cfg.vocab_size as u32)? })
            } else {
                None
            },
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
    /// returns the last row's logits when all layers are resident, with teacher forcing: after layer `l`, `forced(l)` (when it
    /// returns rows) replaces the residual before layer `l + 1`, so each
    /// layer's comparison measures that layer alone. `all_logits` returns
    /// every row's logits instead of the last row's.
    pub fn prefill_forced(&self, placement: &mut MimoPlacement, tokens: &[u32], all_logits: bool,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>) -> Result<Option<Vec<f32>>> {
        self.prefill_device(placement, tokens, all_logits, on_layer, forced)?
            .map(|logits| logits.to_host(self.library)).transpose()
    }

    /// [`Self::prefill_forced`] leaving the logits on the device.
    pub fn prefill_device(&self, placement: &mut MimoPlacement, tokens: &[u32], all_logits: bool,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>) -> Result<Option<DeviceLogits>> {
        let (t, start) = (tokens.len(), placement.len);
        ensure!(t > 0 && t <= self.prefill_rows && start + t <= self.max_context, "prefill of {t} rows at {start}");
        let used = (start + t).div_ceil(PAGE_ROWS);
        let tables = StepTables {
            decode: false,
            positions: (start..start + t).map(|p| p as i64).collect(),
            slots: (start..start + t).map(|p| placement.slot(p)).collect::<Result<_>>()?,
            ring_slots: (start..start + t).map(|p| placement.ring_slot(p)).collect(),
            seq_first: vec![0; t],
            page_table: placement.pages[..used].iter().map(|&page| page as i32).collect(),
            table_stride: 0,
        };
        let logits = self.step(&tables, tokens, if all_logits { t } else { 1 }, on_layer, forced)?;
        placement.len += t;
        Ok(logits)
    }

    /// Appends each sequence's tokens (one for decode, several for a
    /// speculative verify) at its length in one decode-shaped step; returns
    /// every row's logits. A caller that rejects a suffix sets `len` back:
    /// the 256-slot rings keep every key a later step can still need.
    pub fn verify(&self, sequences: &mut [(&mut MimoPlacement, usize)], tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        self.verify_device(sequences, tokens, on_layer)?.map(|logits| logits.to_host(self.library)).transpose()
    }

    /// [`Self::verify`] leaving every row's logits on the device.
    pub fn verify_device(&self, sequences: &mut [(&mut MimoPlacement, usize)], tokens: &[u32],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<DeviceLogits>> {
        let rows: usize = sequences.iter().map(|(_, n)| n).sum();
        ensure!(rows > 0 && rows <= DECODE_ROWS && tokens.len() == rows, "decode step of {rows} rows");
        let stride = sequences.iter().map(|(p, _)| p.pages.len()).max().unwrap_or(1);
        let mut tables = StepTables { decode: true, positions: Vec::new(), slots: Vec::new(), ring_slots: Vec::new(),
            seq_first: Vec::new(), page_table: Vec::new(), table_stride: stride };
        for (placement, count) in sequences.iter() {
            let first = tables.positions.len() as i32;
            for position in placement.len..placement.len + count {
                ensure!(position < self.max_context, "decode at {position} past the context");
                tables.positions.push(position as i64);
                tables.slots.push(placement.slot(position)?);
                tables.ring_slots.push(placement.ring_slot(position));
                tables.seq_first.push(first);
                let mut pages: Vec<i32> = placement.pages.iter().map(|&page| page as i32).collect();
                pages.resize(stride, 0);
                tables.page_table.extend(pages);
            }
        }
        let logits = self.step(&tables, tokens, rows, on_layer, None)?;
        for (placement, count) in sequences.iter_mut() {
            placement.len += *count;
        }
        Ok(logits)
    }

    fn step(&self, tables: &StepTables, tokens: &[u32], logit_rows: usize,
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>) -> Result<Option<DeviceLogits>> {
        let (h, t) = (self.cfg.hidden, tables.positions.len());
        let (cell, capacity) = if tables.decode { (&self.decode_workspace, DECODE_ROWS) } else { (&self.workspace, self.prefill_rows) };
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(self.workspace(0, capacity, tables.decode)?);
        }
        let workspace = cell.borrow();
        let w = workspace.as_ref().context("workspace")?;
        ensure!(t <= w.rows && logit_rows <= t && tokens.len() == t, "step exceeds the workspace");
        // The head split's second GPU: its workspace of the same shape.
        let peer_workspace = match &self.peer {
            Some(peer) => {
                let cell = if tables.decode { &peer.decode_workspace } else { &peer.workspace };
                if cell.borrow().is_none() {
                    *cell.borrow_mut() = Some(self.workspace(1, capacity, tables.decode)?);
                }
                Some((peer, cell.borrow()))
            }
            None => None,
        };
        let split = match &peer_workspace {
            Some((peer, ws)) => Some((*peer, ws.as_ref().context("peer workspace")?)),
            None => None,
        };
        // The tables and ids go up synchronously: the previous step (which no
        // longer ends in a logits download) must be done reading them.
        // SAFETY: the engine owns these streams; every exchange wait queued on
        // them was matched by a push the previous step queued, so they drain.
        unsafe {
            self.library.cuda_stream_synchronize(self.stream)?;
            if let Some((peer, _)) = split {
                self.library.cuda_stream_synchronize(peer.stream)?;
            }
        }
        for (rank, w) in std::iter::once(w).chain(split.map(|(_, w1)| w1)).enumerate() {
            self.on(rank, || {
                self.put(&w.positions, &tables.positions)?;
                self.put(&w.slots, &tables.slots)?;
                self.put(&w.ring_slots, &tables.ring_slots)?;
                self.put(&w.seq_first, &tables.seq_first)?;
                self.put(&w.page_table, &tables.page_table)
            })?;
        }
        self.embedding.embed(tokens, w.ids.buffer, 1, w.h.buffer, self.stream)?;
        let rows = Scalar::I32(t as i32);
        let cap = if tables.decode { "m64" } else { "m4096" };
        let layers = &self.weights.layers;
        let bytes = t * h * 2;
        // Rank 1 needs nothing from the host once queued (every input is a push from
        // rank 0), so without teacher forcing (which rewrites its rows from the host
        // mid-step) its next layer is queued before rank 0 blocks in the Spark exchange:
        // once rank 0 forwards the routed sum, rank 1 runs on without waiting for the
        // host to queue its work.
        let ahead = forced.is_none();
        if let Some((_, w1)) = split {
            // The embedded rows to the second GPU's residual stream.
            self.exchange()?.push_to(0, DIRECT, w.h.buffer.ptr, w1.h.buffer.ptr, bytes)?;
            self.peer_layer(0, w1, rows, cap, tables, bytes)?;
        }
        self.norm(w, layers[0].ptr("input_norm")?, 0, rows)?;
        for (index, layer) in layers.iter().enumerate() {
            let last = index + 1 == layers.len();
            if let (Some((_, w1)), false, true) = (split, ahead, index > 0) {
                self.peer_layer(index, w1, rows, cap, tables, bytes)?;
            }
            // h += attention; x = post_attention_layernorm(h); under a head split both
            // GPUs add both partials in the same order (identical residual streams).
            self.attention_on(0, w, self.kv[index].buffer.ptr, layer, rows, cap, tables)?;
            let (attended, ffn) = (slot(index, false), slot(index, true));
            if split.is_some() {
                self.push(0, attended, w.delta.buffer.ptr, bytes)?;
                self.wait(0, attended)?;
                self.norm_on(0, w, layer.ptr("post_norm")?, 2, rows, self.recv(0, attended))?;
                if let (Some((_, w1)), true, false) = (split, ahead, last) {
                    self.peer_layer(index + 1, w1, rows, cap, tables, bytes)?;
                }
            } else {
                self.norm(w, layer.ptr("post_norm")?, 1, rows)?;
            }
            // How the next norm takes the FFN output: one delta, or this GPU's partial plus the other's.
            let mut deltas = 1;
            if layer.dense {
                self.dense_ffn(0, w, layer, rows, cap, tables.decode)?;
                if split.is_some() {
                    self.push(0, ffn, w.delta.buffer.ptr, bytes)?;
                    self.wait(0, ffn)?;
                    deltas = 2;
                }
            } else {
                // The routed experts' sum also goes to the second GPU (not after the last layer).
                let forward = (split.is_some() && !last).then_some(ffn);
                self.moe(w, index, layer, t, tables.decode, forward)?;
            }
            // h += ffn; x = next input_layernorm(h) (or the final norm).
            let weight = match layers.get(index + 1) {
                Some(next) => next.ptr("input_norm")?,
                None => self.weights.norm.buffer.ptr,
            };
            let second = if deltas == 2 { self.recv(0, ffn) } else { w.delta.buffer.ptr };
            self.norm_on(0, w, weight, deltas, rows, second)?;
            if let Some(drafter) = &self.drafter {
                // The step's last TAP_ROWS rows (a prefill's tail holds every
                // context row later drafts can see).
                let n = t.min(super::dflash::TAP_ROWS);
                drafter.tap(index, w.h.buffer.ptr, t - n, n)?;
            }
            if let (Some(mtp), true) = (&self.mtp, index + 1 == self.cfg.layers) {
                self.mtp_tap(mtp, w, tables)?;
            }
            if let Some(on_layer) = on_layer.as_mut() {
                on_layer(index, &self.download(&w.h, t * h * 2)?)?;
            }
            if let Some(rows_forced) = forced.and_then(|f| f(index)) {
                ensure!(rows_forced.len() == t * h * 2, "teacher-forced rows of the wrong size");
                self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: rows_forced.len(), ..w.h.buffer },
                    &rows_forced)?;
                self.norm(w, weight, 0, rows)?;
                if let (Some((peer, w1)), false) = (split, last) {
                    // SAFETY: the engine owns the peer stream (queued only through this layer:
                    // `ahead` is off); drained before the synchronous copy.
                    unsafe { self.library.cuda_stream_synchronize(peer.stream)? };
                    self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: rows_forced.len(), ..w1.h.buffer },
                        &rows_forced)?;
                    self.norm_on(1, w1, peer.layers[index + 1].ptr("input_norm")?, 0, rows, w1.delta.buffer.ptr)?;
                }
            }
        }
        if layers.len() < self.cfg.layers {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        // SAFETY: the rows start inside the final norm's output.
        let x = unsafe { w.x.buffer.ptr.cast::<u8>().add((t - logit_rows) * h * 2) }.cast::<c_void>();
        match &self.weights.head_fp8 {
            Some((q, scale)) if tables.decode && logit_rows <= FP8_ROWS as usize => {
                self.run("mimo_head_fp8", &[("x", x), ("w_fp8", q.buffer.ptr), ("scale", scale.buffer.ptr),
                    ("logits", w.logits.buffer.ptr)], &[Scalar::I32(logit_rows as i32)])?;
            }
            // SAFETY: the final norm's output and the head operands are live buffers of these shapes.
            _ => unsafe {
                w.head.as_ref().context("LM head")?.launch(x.cast(), self.weights.head.buffer.ptr.cast(), w.logits.buffer.ptr.cast(),
                    logit_rows as u32, self.stream)?;
            },
        }
        let vocab = self.cfg.vocab_size;
        Ok(Some(DeviceLogits { ptr: w.logits.buffer.ptr, rows: logit_rows, vocab, stride: vocab, stream: self.stream,
            greedy: None }))
    }

    /// Queues rank 1's share of layer `index` (the head split's second GPU),
    /// mirroring rank 0's pushes in `step` one for one: the embedded rows before
    /// layer 0 (its first wait), then per layer its attention partial out and rank
    /// 0's in, and the dense MLP's partials or the routed experts' sum in. Ends
    /// with the next layer's input norm (nothing after the last layer).
    #[allow(clippy::too_many_arguments)]
    fn peer_layer(&self, index: usize, w1: &Workspace<'_>, rows: Scalar, cap: &str, tables: &StepTables, bytes: usize)
        -> Result<()> {
        let peer = self.peer.as_ref().context("no head-split peer")?;
        let share = &peer.layers[index];
        if index == 0 {
            self.wait(1, DIRECT)?;
            self.norm_on(1, w1, share.ptr("input_norm")?, 0, rows, w1.delta.buffer.ptr)?;
        }
        self.attention_on(1, w1, peer.kv[index].buffer.ptr, share, rows, cap, tables)?;
        let (attended, ffn) = (slot(index, false), slot(index, true));
        self.push(1, attended, w1.delta.buffer.ptr, bytes)?;
        self.wait(1, attended)?;
        self.norm_on(1, w1, share.ptr("post_norm")?, 2, rows, self.recv(1, attended))?;
        let next = peer.layers.get(index + 1);
        if share.dense {
            self.dense_ffn(1, w1, share, rows, cap, tables.decode)?;
            self.push(1, ffn, w1.delta.buffer.ptr, bytes)?;
            self.wait(1, ffn)?;
        } else if next.is_some() {
            self.wait(1, ffn)?;
        }
        let Some(next) = next else { return Ok(()) };
        let (deltas, first) = if share.dense { (2, w1.delta.buffer.ptr) } else { (1, self.recv(1, ffn)) };
        self.norm_full(1, w1, next.ptr("input_norm")?, deltas, rows, first, self.recv(1, ffn))
    }

    /// `residual (h) += delta` when `deltas` is 1, then `x = weight * RMSNorm(h)`.
    fn norm(&self, w: &Workspace<'_>, weight: *mut c_void, deltas: i32, rows: Scalar) -> Result<()> {
        self.norm_full(0, w, weight, deltas, rows, w.delta.buffer.ptr, w.delta.buffer.ptr)
    }

    /// On rank `rank`: `h += bf16(delta + second)` when `deltas` is 2 (the local
    /// `delta` first, so both GPUs of a head split sum identically), `h += delta`
    /// when 1, then `x = weight * RMSNorm(h)`.
    #[allow(clippy::too_many_arguments)]
    fn norm_on(&self, rank: usize, w: &Workspace<'_>, weight: *mut c_void, deltas: i32, rows: Scalar,
        second: *mut c_void) -> Result<()> {
        self.norm_full(rank, w, weight, deltas, rows, w.delta.buffer.ptr, second)
    }

    #[allow(clippy::too_many_arguments)]
    fn norm_full(&self, rank: usize, w: &Workspace<'_>, weight: *mut c_void, deltas: i32, rows: Scalar,
        first: *mut c_void, second: *mut c_void) -> Result<()> {
        self.run_on(rank, false, "mimo_norm", &[("residual", w.h.buffer.ptr), ("delta0", first), ("delta1", second),
            ("weight", weight), ("out", w.x.buffer.ptr)], &[rows, Scalar::I32(deltas)])
    }

    /// Copies this step's last-layer rows (pre-norm) into the MTP hidden ring
    /// at (ring, position % 256); a prefill's last 256 rows.
    fn mtp_tap(&self, mtp: &super::mtp::MtpDrafter<'_>, w: &Workspace<'_>, tables: &StepTables) -> Result<()> {
        use super::mtp::HIDDEN_ROWS;
        let (h, t) = (self.cfg.hidden, tables.positions.len());
        let row = h * 2;
        let first = t.saturating_sub(HIDDEN_ROWS);
        let mut r = first;
        while r < t {
            // A run of rows whose ring slots advance by one (one sequence, no wrap).
            let slot = tables.ring_slots[r] as usize;
            let mut n = 1;
            while r + n < t && tables.ring_slots[r + n] as usize == slot + n && (slot + n) % HIDDEN_ROWS != 0 {
                n += 1;
            }
            let ring = slot / RING_ROWS;
            let dest = (ring * HIDDEN_ROWS + slot % RING_ROWS % HIDDEN_ROWS) * row;
            // SAFETY: rows r..r+n of `w.h` and ring rows dest.. lie inside their buffers; stream-ordered.
            unsafe {
                self.library.copy_d2d_async(
                    cuteafd_ffi::CuteafdDeviceBuffer { ptr: mtp.hidden.buffer.ptr.cast::<u8>().add(dest).cast(),
                        bytes: n * row, ..mtp.hidden.buffer },
                    cuteafd_ffi::CuteafdDeviceBuffer { ptr: w.h.buffer.ptr.cast::<u8>().add(r * row).cast(),
                        bytes: n * row, ..w.h.buffer },
                    n * row, self.stream)?;
            }
            r += n;
        }
        Ok(())
    }

    /// A new sequence in `ring` whose first `len` tokens are processed: MTP
    /// stages start their true rows within a window of the end.
    pub fn mtp_reset(&self, ring: usize, len: usize) {
        if let Some(mtp) = &self.mtp {
            let mut ext = mtp.ext.borrow_mut();
            let start = len.saturating_sub(self.cfg.window + mtp.stages.len() + 1);
            ext[ring] = vec![start; mtp.stages.len()];
        }
    }

    /// Up to `stages` MTP drafts after each sequence's next token (see
    /// `mtp`). Every pass embeds its tokens on the device (later stages read
    /// the earlier stages' drafts there), so the passes queue back to back
    /// and the drafts come back once, at the end.
    pub fn mtp_draft(&self, seqs: &[super::mtp::MtpSeq<'_>], stages: usize) -> Result<Vec<Vec<u32>>> {
        use super::mtp::Token;
        let mtp = self.mtp.as_ref().context("no MTP drafter")?;
        let stages = stages.min(mtp.stages.len());
        ensure!(seqs.len() <= DECODE_ROWS, "MTP drafts of {} sequences", seqs.len());
        // The previous step is done with the staging.
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        self.mtp_staging.borrow_mut().1 = 0;
        for k in 0..stages {
            // Catch-up passes over true rows while a sequence's rows exceed the step.
            loop {
                let mut groups = Vec::new();
                let mut rows = 0;
                for seq in seqs {
                    let ext = mtp.ext.borrow()[seq.ring][k];
                    ensure!(ext <= seq.len, "MTP ring {} is ahead of its sequence (reset it at admission)", seq.ring);
                    // True rows are those whose token t_{j+k+1} is known (j <= len - k - 1).
                    // Catch-up stops before row len - 1: the drafting pass needs that row (its
                    // argmax is the draft), so stage 0 must not consume it here.
                    let true_end = seq.len.saturating_sub(k.max(1));
                    if seq.len - ext > MTP_STEP_ROWS && ext < true_end && rows < DECODE_ROWS {
                        let n = (true_end - ext).min(DECODE_ROWS - rows);
                        groups.push((seq.ring, ext, (ext..ext + n).map(|j| Token::Known(seq.tokens[j + k + 1]))
                            .collect::<Vec<_>>()));
                        rows += n;
                    }
                }
                if groups.is_empty() {
                    break;
                }
                self.mtp_pass(mtp, k, &groups, None)?;
                let mut ext = mtp.ext.borrow_mut();
                for (ring, first, tokens) in &groups {
                    ext[*ring][k] = ext[*ring][k].max(first + tokens.len());
                }
            }
            // The drafting pass: rows ext..len of every sequence (in groups within the step).
            let mut index = 0;
            while index < seqs.len() {
                let mut groups = Vec::new();
                let mut members = Vec::new();
                let mut rows = 0;
                while index < seqs.len() {
                    let seq = &seqs[index];
                    let ext = mtp.ext.borrow()[seq.ring][k];
                    ensure!(seq.tokens.len() > seq.len, "MTP sequence needs its next token");
                    let n = seq.len - ext;
                    if rows + n > DECODE_ROWS && !groups.is_empty() {
                        break;
                    }
                    ensure!(n <= DECODE_ROWS, "MTP stage {k}: {n} pending rows");
                    let tokens: Vec<Token> = (ext..seq.len).map(|j| {
                        let at = j + k + 1;
                        if at <= seq.len { Token::Known(seq.tokens[at]) }
                        else { Token::Draft { stage: at - seq.len - 1, member: index } }
                    }).collect();
                    groups.push((seq.ring, ext, tokens));
                    members.push(index);
                    rows += n;
                    index += 1;
                }
                self.mtp_pass(mtp, k, &groups, Some(&members))?;
                let mut ext = mtp.ext.borrow_mut();
                for (&i, (ring, _, _)) in members.iter().zip(&groups) {
                    ext[*ring][k] = ext[*ring][k].max(seqs[i].len.saturating_sub(k));
                }
            }
        }
        if stages == 0 {
            return Ok(vec![Vec::new(); seqs.len()]);
        }
        let bytes = self.download(&mtp.ids, (1 + stages) * DECODE_ROWS * 4)?;
        let word = |at: usize| u32::from_le_bytes(bytes[at * 4..at * 4 + 4].try_into().unwrap());
        Ok((0..seqs.len()).map(|i| (0..stages).map(|k| word((1 + k) * DECODE_ROWS + i)).collect()).collect())
    }

    /// Queues `bytes` into `dst` through the MTP staging (waits for the
    /// stream only when the staging is full).
    fn stage_async(&self, dst: cuteafd_ffi::CuteafdDeviceBuffer, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        ensure!(bytes.len() <= dst.bytes, "staged upload exceeds its buffer");
        let mut staging = self.mtp_staging.borrow_mut();
        if staging.1 + bytes.len() > staging.0.buffer.bytes {
            // SAFETY: the engine owns this stream; its queued copies read the staging.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            staging.1 = 0;
        }
        let at = staging.1;
        ensure!(at + bytes.len() <= staging.0.buffer.bytes, "MTP pass inputs exceed the staging buffer");
        staging.0.bytes_mut()[at..at + bytes.len()].copy_from_slice(bytes);
        let source = cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: `at` lies inside the pinned staging buffer.
            ptr: unsafe { staging.0.buffer.ptr.cast::<u8>().add(at) }.cast(),
            bytes: bytes.len(),
            ..staging.0.buffer
        };
        // SAFETY: the staged bytes stay untouched until the stream drains (see above).
        unsafe { self.library.copy_host_buffer_h2d_async(dst, source, bytes.len(), self.stream)? };
        staging.1 = (at + bytes.len()).div_ceil(16) * 16;
        Ok(())
    }

    /// One MTP stage over `groups` of (ring, first row, tokens t_{j+k+1});
    /// with `members`, each group's last row's draft becomes stage `k`'s draft
    /// of that member (on the device). Queued: no host wait.
    fn mtp_pass(&self, mtp: &super::mtp::MtpDrafter<'_>, k: usize,
        groups: &[(usize, usize, Vec<super::mtp::Token>)], members: Option<&[usize]>) -> Result<()> {
        use super::mtp::{Token, HIDDEN_ROWS};
        let stage = &mtp.stages[k];
        let (h, vocab) = (self.cfg.hidden, self.cfg.vocab_size);
        let mut tables = StepTables { decode: true, positions: Vec::new(), slots: Vec::new(), ring_slots: Vec::new(),
            seq_first: Vec::new(), page_table: vec![0], table_stride: 1 };
        let (mut known, mut index) = (Vec::new(), Vec::new());
        for (ring, first, tokens) in groups {
            ensure!(*ring < self.rings, "MTP ring {ring} of {}", self.rings);
            let start = tables.positions.len() as i32;
            for (j, token) in (*first..first + tokens.len()).zip(tokens) {
                tables.positions.push(j as i64);
                tables.slots.push(-1);
                tables.ring_slots.push((ring * RING_ROWS + j % RING_ROWS) as i64);
                tables.seq_first.push(start);
                let r = known.len();
                match *token {
                    Token::Known(id) => {
                        known.push(id);
                        index.push(r as u32);
                    }
                    Token::Draft { stage, member } => {
                        ensure!(stage < k && member < DECODE_ROWS, "MTP stage {k} reads draft {stage} of {member}");
                        known.push(0);
                        index.push(((1 + stage) * DECODE_ROWS + member) as u32);
                    }
                }
            }
        }
        let t = tables.positions.len();
        ensure!(t > 0 && t <= DECODE_ROWS, "MTP pass of {t} rows");
        if self.decode_workspace.borrow().is_none() {
            *self.decode_workspace.borrow_mut() = Some(self.workspace(0, DECODE_ROWS, true)?);
        }
        let workspace = self.decode_workspace.borrow();
        let w = workspace.as_ref().context("workspace")?;
        self.stage_async(w.positions.buffer, bytes_of(&tables.positions))?;
        self.stage_async(w.ring_slots.buffer, bytes_of(&tables.ring_slots))?;
        self.stage_async(w.seq_first.buffer, bytes_of(&tables.seq_first))?;
        self.stage_async(w.page_table.buffer, bytes_of(&tables.page_table))?;
        self.embedding.check(&known)?;
        self.stage_async(mtp.ids.buffer, bytes_of(&known))?;
        self.stage_async(mtp.index.buffer, bytes_of(&index))?;
        let row = h * 2;
        // SAFETY: the ids (known tokens, earlier stages' drafts) and the indices are
        // ordered on this stream before the gather; `embed` holds DECODE_ROWS rows.
        unsafe { self.embedding.embed_device_ids(mtp.ids.buffer, Some((mtp.index.buffer.ptr.cast_const(), &index)),
            t, 1, None, mtp.embed.buffer, self.stream)? };
        // The target's hidden rows of the same positions.
        let mut at = 0;
        for (ring, first, tokens) in groups {
            for j in *first..first + tokens.len() {
                let src = (ring * HIDDEN_ROWS + j % HIDDEN_ROWS) * row;
                // SAFETY: one row inside each buffer; stream-ordered.
                unsafe {
                    self.library.copy_d2d_async(
                        cuteafd_ffi::CuteafdDeviceBuffer { ptr: mtp.rows_h.buffer.ptr.cast::<u8>().add(at * row).cast(),
                            bytes: row, ..mtp.rows_h.buffer },
                        cuteafd_ffi::CuteafdDeviceBuffer { ptr: mtp.hidden.buffer.ptr.cast::<u8>().add(src).cast(),
                            bytes: row, ..mtp.hidden.buffer }, row, self.stream)?;
                }
                at += 1;
            }
        }
        let rows = Scalar::I32(t as i32);
        for (source, weight, out) in [(&mtp.embed, &stage.enorm, &mtp.normed_e), (&mtp.rows_h, &stage.hnorm, &mtp.normed_h)] {
            self.run("mimo_norm", &[("residual", source.buffer.ptr), ("delta0", source.buffer.ptr),
                ("delta1", source.buffer.ptr), ("weight", weight.buffer.ptr), ("out", out.buffer.ptr)],
                &[rows, Scalar::I32(0)])?;
        }
        // SAFETY: [t, H] sources into [t, 2H] halves, then eh_proj into the residual stream.
        unsafe {
            self.library.glm_dflash_tap(mtp.normed_e.buffer.ptr, mtp.cat.buffer.ptr, t, h, 2 * h, 0, self.stream)?;
            self.library.glm_dflash_tap(mtp.normed_h.buffer.ptr, mtp.cat.buffer.ptr, t, h, 2 * h, h, self.stream)?;
            self.library.linear_bf16(mtp.cat.buffer.ptr, stage.eh.buffer.ptr, w.h.buffer.ptr, t, 2 * h, h, self.stream)?;
        }
        let layer = &stage.layer;
        self.norm(w, layer.ptr("input_norm")?, 0, rows)?;
        self.attention(w, stage.ring.buffer.ptr, layer, rows, "m64", &tables)?;
        self.norm(w, layer.ptr("post_norm")?, 1, rows)?;
        self.dense_ffn(0, w, layer, rows, "m64", true)?;
        self.norm(w, stage.final_norm.buffer.ptr, 1, rows)?;
        let Some(members) = members else { return Ok(()) };
        ensure!(members.len() == groups.len(), "a member per drafting group");
        match &self.weights.head_fp8 {
            Some((q, scale)) if t <= FP8_ROWS as usize => {
                self.run("mimo_head_fp8", &[("x", w.x.buffer.ptr), ("w_fp8", q.buffer.ptr), ("scale", scale.buffer.ptr),
                    ("logits", w.logits.buffer.ptr)], &[rows])?;
            }
            // SAFETY: the final norm's output and the head operands are live buffers of these shapes.
            _ => unsafe {
                w.head.as_ref().context("LM head")?.launch(w.x.buffer.ptr.cast(), self.weights.head.buffer.ptr.cast(), w.logits.buffer.ptr.cast(),
                    t as u32, self.stream)?;
            },
        }
        let region = |dev: &Dev<'_>, offset: usize, bytes: usize| cuteafd_ffi::CuteafdDeviceBuffer {
            // SAFETY: callers pass offsets inside the allocation.
            ptr: unsafe { dev.buffer.ptr.cast::<u8>().add(offset) }.cast(), bytes, ..dev.buffer };
        // SAFETY: the logits rows, the select buffer (ids, then statuses) and the
        // drafts region are live buffers of these shapes; stream-ordered.
        unsafe {
            self.library.cuda_logits_greedy_f32_async(w.logits.buffer.ptr, t, vocab, vocab, w.select.buffer.ptr,
                std::ptr::null_mut(), region(&w.select, t * 4, t * 4).ptr, self.stream)?;
            let mut end = 0;
            for ((_, _, tokens), &member) in groups.iter().zip(members) {
                end += tokens.len();
                self.library.copy_d2d_async(region(&mtp.ids, ((1 + k) * DECODE_ROWS + member) * 4, 4),
                    region(&w.select, (end - 1) * 4, 4), 4, self.stream)?;
            }
        }
        Ok(())
    }

    /// `kv`: the layer's paged record pool (full) or its rings (SWA).
    fn attention(&self, w: &Workspace<'_>, kv: *mut c_void, layer: &MimoLayer<'_>, rows: Scalar, cap: &str,
        tables: &StepTables) -> Result<()> {
        self.attention_on(0, w, kv, layer, rows, cap, tables)
    }

    /// Attention of `layer` (rank `rank`'s share under a head split: its heads,
    /// a partial o_proj sum) into `delta`, on that rank's GPU.
    #[allow(clippy::too_many_arguments)]
    fn attention_on(&self, rank: usize, w: &Workspace<'_>, kv: *mut c_void, layer: &MimoLayer<'_>, rows: Scalar,
        cap: &str, tables: &StepTables) -> Result<()> {
        for part in 0..ATTENTION_PARTS {
            self.attention_part(rank, part, w, kv, layer, rows, cap, tables)?;
        }
        Ok(())
    }

    /// Part `part` of [`Self::attention_on`]: 0 the qkv producer, 1 attention,
    /// 2 o_proj (a head split enqueues the two GPUs' parts alternately, so
    /// neither waits for the host to queue the other's whole layer).
    #[allow(clippy::too_many_arguments)]
    fn attention_part(&self, rank: usize, part: usize, w: &Workspace<'_>, kv: *mut c_void, layer: &MimoLayer<'_>,
        rows: Scalar, cap: &str, tables: &StepTables) -> Result<()> {
        let mode = if tables.decode { "decode" } else { "prefill" };
        let k = kind(layer.attention);
        let (full, swa) = match (rank, &self.peer) {
            (1, Some(peer)) => (&peer.cos_sin_full, &peer.cos_sin_swa),
            _ => (&self.cos_sin_full, &self.cos_sin_swa),
        };
        let (slots, cos_sin) = match layer.attention {
            MimoAttention::Full => (w.slots.buffer.ptr, full.buffer.ptr),
            MimoAttention::Sliding => (w.step_slots.buffer.ptr, swa.buffer.ptr),
        };
        let split = layer.split;
        let records = match layer.attention {
            MimoAttention::Full => kv,
            MimoAttention::Sliding => w.kv_step.buffer.ptr,
        };
        let decode = tables.decode;
        if part == 0 {
            let pointers = [("x", w.x.buffer.ptr), ("positions", w.positions.buffer.ptr), ("kv_slots", slots),
                ("cos_sin", cos_sin), ("w_qkv_fp8", layer.ptr("w_qkv_fp8")?), scale("w_qkv", decode, layer)?,
                ("kv_cache", records), ("query", w.query.buffer.ptr), ("scratch", w.scratch.buffer.ptr)];
            let kv = self.kv_cache.of(layer.attention).program_tag();
            return self.run_on(rank, split, &format!("mimo_{k}_producer{kv}_{cap}"), &pointers, &self.w8_scalars(rows, decode));
        }
        let name = format!("mimo_{k}_attention{}_{mode}_{cap}", self.kv_cache.of(layer.attention).program_tag());
        match layer.attention {
            _ if part != 1 => {}
            MimoAttention::Full => {
                let mut scalars = vec![rows, Scalar::I32(tables.table_stride as i32)];
                let mut pointers = vec![("q", w.query.buffer.ptr), ("kv_cache", kv),
                    ("positions", w.positions.buffer.ptr), ("page_table", w.page_table.buffer.ptr)];
                if tables.decode {
                    scalars.push(Scalar::I32(DECODE_SPLITS));
                } else if self.kv_cache != MimoKvCache::Bf16 {
                    // 8-bit prefill: the sequence's keys are widened once into `kv_wide`.
                    let keys = tables.positions.last().map_or(0, |&p| p + 1);
                    ensure!(keys as usize <= self.max_context, "prefill past max_context ({keys} keys)");
                    pointers.push(("kv_wide", w.kv_wide.buffer.ptr));
                    scalars.push(Scalar::I32(keys as i32));
                }
                pointers.extend([("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
                self.run_on(rank, split, &name, &pointers, &scalars)?;
            }
            MimoAttention::Sliding => {
                self.run_on(rank, split, &name, &[("q", w.query.buffer.ptr), ("kv_step", w.kv_step.buffer.ptr),
                    ("ring", kv), ("positions", w.positions.buffer.ptr),
                    ("ring_slots", w.ring_slots.buffer.ptr), ("seq_first", w.seq_first.buffer.ptr),
                    ("sinks", layer.ptr("sinks")?), ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
                    &[rows])?;
            }
        }
        if part != 2 {
            return Ok(());
        }
        let mut pointers = vec![("attn", w.attn.buffer.ptr), ("w_o", layer.ptr("w_o")?)];
        if decode {
            pointers.extend([("w_o_fp8", layer.ptr_or("w_o_fp8", "w_o")?), ("w_o_scale", layer.ptr_or("w_o_scale", "w_o")?)]);
        }
        pointers.push(("out", w.delta.buffer.ptr));
        self.run_on(rank, split, &format!("mimo_o_{cap}"), &pointers, &fp8_scalars(rows, decode, layer.has("w_o_fp8")))
    }

    /// The dense SwiGLU MLP (layer 0, MTP layers) over its FP8-only weights into `delta`.
    #[allow(clippy::too_many_arguments)]
    fn dense_ffn(&self, rank: usize, w: &Workspace<'_>, layer: &MimoLayer<'_>, rows: Scalar, cap: &str, decode: bool)
        -> Result<()> {
        let pointers = [("x", w.x.buffer.ptr), ("w_gate_up_fp8", layer.ptr("w_gate_up_fp8")?),
            scale("w_gate_up", decode, layer)?, ("w_down_fp8", layer.ptr("w_down_fp8")?), scale("w_down", decode, layer)?,
            ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)];
        self.run_on(rank, layer.split, &format!("mimo_ffn_{cap}"), &pointers, &self.w8_scalars(rows, decode))
    }

    /// `[rows, fp8_rows]` of the programs over FP8-only weights: decode rows up
    /// to `FP8_DECODE_ROWS` on the FP8 GEMVs (W8A16 GEMMs above); prefill 1
    /// (W8A8) or 0 (W8A16, `prefill_w8a8` off).
    fn w8_scalars(&self, rows: Scalar, decode: bool) -> [Scalar; 2] {
        [rows, Scalar::I32(if decode { FP8_DECODE_ROWS } else { i32::from(self.prefill_w8a8) })]
    }

    /// Per MoE layer, the weights a decode step reads after its routed
    /// experts are out, in read order: the next layer's attention (E4M3
    /// copies where the decode programs read them), norms, router and dense
    /// MLP; after the last layer the final norm and head.
    pub fn decode_read_order(&self) -> Vec<Vec<crate::shared::l2_prefetch::Range>> {
        let layers = &self.weights.layers;
        (0..layers.len()).map(|i| match layers.get(i + 1) {
            Some(next) => crate::shared::l2_prefetch::operands(&["input_norm", "w_qkv", "sinks", "w_o", "post_norm",
                "w_router", "w_hilo", "gate.bias", "w_gate_up", "w_down"], |n| next.range(n)),
            None => {
                let head = match &self.weights.head_fp8 {
                    Some((q, scale)) => vec![q, scale],
                    None => vec![&self.weights.head],
                };
                std::iter::once(&self.weights.norm).chain(head).map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))
                    .collect()
            }
        }).collect()
    }

    /// In a decode step with no real exchange (local experts), under
    /// CUTEAFD_EMULATE_EXCHANGE_US (benchmarks): the L2 prefetch and a
    /// Spark-like wait before the experts run.
    fn emulated_exchange(&self, index: usize, decode: bool) -> Result<()> {
        if !decode {
            return Ok(());
        }
        let Some(mark) = crate::shared::l2_prefetch::exchange_mark(self.library, self.stream)? else { return Ok(()) };
        if let Some(l2) = &self.l2 {
            l2.issue(self.library, index, self.stream)?;
        }
        crate::shared::l2_prefetch::exchange_wait(self.library, Some(mark))
    }

    /// Router scores, the sigmoid top-k select and the FP8 wire rows, then the
    /// routed experts (local or Spark); leaves their sum in `delta` for the
    /// next norm's residual add.
    /// With `forward`, the sum is also pushed into that exchange slot of the head split's
    /// second GPU (from the Spark path right after the reduce, before the host wait).
    fn moe(&self, w: &Workspace<'_>, index: usize, layer: &MimoLayer<'_>, t: usize, decode: bool,
        forward: Option<usize>) -> Result<()> {
        let bytes = t * self.cfg.hidden * 2;
        let spark = matches!(self.experts, Some(Experts::Spark { .. }));
        self.moe_local(w, index, layer, t, decode, forward.filter(|_| spark))?;
        match forward {
            Some(slot) if !spark => self.push(0, slot, w.delta.buffer.ptr, bytes),
            _ => Ok(()),
        }
    }

    fn moe_local(&self, w: &Workspace<'_>, index: usize, layer: &MimoLayer<'_>, t: usize, decode: bool,
        forward: Option<usize>) -> Result<()> {
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let experts = self.experts.as_ref().with_context(|| format!(
            "layer {index} is an MoE layer: pass Spark --peers serving the fp8 family, or --local-experts \
             (run --layers 1 for the dense layer alone)"))?;
        let rows = Scalar::I32(t as i32);
        self.run("mimo_router_scores", &[("x", w.x.buffer.ptr), layer.router_operand()?,
            ("logits", w.router_logits.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        // SAFETY: logits, bias and route outputs are live buffers of `t` rows.
        unsafe {
            self.library.router_select(w.router_logits.buffer.ptr, layer.ptr("gate.bias")?, std::ptr::null(),
                std::ptr::null(), w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, t, self.cfg.experts, topk,
                self.cfg.routed_scale as f32, true, self.stream)?;
        }
        let bf16_input = matches!(experts, Experts::Spark { .. }) && self.expert_input.bf16(decode);
        let grid = (t * h.div_ceil(256)).div_ceil(8).clamp(1, 4 * 188);
        if !bf16_input {
            self.run("mimo_expert_input_quant", &[("source_ptr", w.x.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
                // SAFETY: the scale rows follow the payload inside each wire row.
                ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
                ("scale_mma_ptr", w.delta.buffer.ptr)], &[rows, Scalar::I32(grid as i32)])?;
        }
        if !matches!(experts, Experts::Spark { .. }) {
            self.emulated_exchange(index, decode)?;
        }
        match experts {
            Experts::Local(local) => {
                let resident = local.index_of(index)?;
                // Coordinator packages take the BF16 rows themselves (exact, no wire).
                let input = if local.wire_input() { w.wire.buffer.ptr } else { w.x.buffer.ptr };
                // SAFETY: input rows, route ids (u32 = i32 for ids < 256), weights and
                // delta are live buffers of `t` rows on this engine's stream.
                unsafe {
                    local.run(resident, t, input, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr,
                        w.delta.buffer.ptr, self.stream)
                }
            }
            Experts::Streamed { experts, tensors, window } => {
                let mut local = experts.borrow_mut();
                if local.index_of(index).is_err() {
                    // SAFETY: the engine owns this stream; draining it retires every
                    // launch that read the layer about to be evicted.
                    unsafe { self.library.cuda_stream_synchronize(self.stream)? };
                    if local.layers.len() >= (*window).max(1) {
                        local.layers.remove(0);
                    }
                    let started = std::time::Instant::now();
                    local.layers.push(Fp8Layer::load(self.library, tensors, index, 1, 0)?);
                    self.profile.borrow_mut()[1] += started.elapsed().as_secs_f64();
                }
                let resident = local.index_of(index)?;
                let input = if local.wire_input() { w.wire.buffer.ptr } else { w.x.buffer.ptr };
                // SAFETY: as for `Local`; the layer stays resident until a later step evicts it
                // after draining the stream.
                unsafe {
                    local.run(resident, t, input, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr,
                        w.delta.buffer.ptr, self.stream)
                }
            }
            // SAFETY: `delta` holds `t` rows on this engine's stream.
            Experts::Skip => unsafe {
                self.library.cuda_zero_bytes_async(w.delta.buffer, t * h * 2, self.stream)
            },
            Experts::Spark { transport, runtime } => {
                self.spark_moe(w, index, t, decode, bf16_input, &mut transport.borrow_mut(), runtime, forward)
            }
        }
    }

    /// Routes and wire rows down, one request to every Spark rank, the BF16
    /// rank partials summed into `delta` (GLM's exchange, no shared expert).
    #[allow(clippy::too_many_arguments)]
    fn spark_moe(&self, w: &Workspace<'_>, index: usize, t: usize, decode: bool, bf16_input: bool,
        transport: &mut SparkLink<'_>, runtime: &tokio::runtime::Runtime, forward: Option<usize>) -> Result<()> {
        let kind = if decode { ExpertV2SourceKind::Decode } else { ExpertV2SourceKind::Prefill };
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let (route_bytes, wire_bytes) = (t * topk * 4, if bf16_input { t * h * 2 } else { t * (h + h / 32) });
        let (input, dtype) = if bf16_input { (&w.x, ExpertV2Dtype::Bf16) } else { (&w.wire, ExpertV2Dtype::Fp8E4m3Ue8m0K32) };
        let staging = w.router_host.borrow_mut();
        let host = staging.buffer;
        let at = |offset: usize| cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: ids, weights and wire rows are consecutive inside the pinned buffer.
            ptr: unsafe { host.ptr.cast::<u8>().add(offset) }.cast(),
            bytes: host.bytes - offset,
            ..host
        };
        let timer = std::time::Instant::now();
        // Prefill rows go straight into the transport's registered egress
        // buffer and out from there to every rank; small waves keep a copy.
        let egress = transport.egress(wire_bytes)?;
        // SAFETY: the pinned regions are large enough; the sync completes them.
        unsafe {
            self.library.copy_d2h_host_buffer_async(at(0), w.route_ids.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(route_bytes), w.route_weights.buffer, route_bytes, self.stream)?;
            let target = egress.unwrap_or(at(2 * route_bytes));
            self.library.copy_d2h_host_buffer_async(target, input.buffer, wire_bytes, self.stream)?;
        }
        // In a decode step the L2 prefetch queues behind the copies; the host waits for the copies only.
        match self.l2.as_ref().filter(|_| decode) {
            Some(l2) => {
                let mark = crate::shared::l2_prefetch::mark(self.library, self.stream)?;
                l2.issue(self.library, index, self.stream)?;
                crate::shared::l2_prefetch::reached(self.library, mark)?;
            }
            // SAFETY: the engine owns this stream.
            None => unsafe { self.library.cuda_stream_synchronize(self.stream)? },
        }
        let gpu_wait = timer.elapsed().as_secs_f64();
        self.profile.borrow_mut()[0] += gpu_wait;
        let built = std::time::Instant::now();
        let staged = staging.bytes();
        let word = |offset: usize, i: usize| u32::from_le_bytes(staged[offset + i * 4..][..4].try_into().unwrap());
        let routes = (0..t * topk).map(|i| ExpertProtocolV2RouteEntry {
            row_index: (i / topk) as u32, expert_id: word(0, i), gate_weight: f32::from_bits(word(route_bytes, i)),
        }).collect();
        let wire = match egress {
            Some(_) => transport.egress_payload(wire_bytes)?,
            None => staged[2 * route_bytes..2 * route_bytes + wire_bytes].to_vec().into(),
        };
        drop(staging);
        let mut request = ExpertProtocolV2Request::new_bytes(index as u64 + 1, 17, index as u32, h as u32, dtype,
            (0..t as u32).map(|row| ExpertProtocolV2RowDescriptor {
                row_id: u64::from(row), source_kind: kind, source_request_id: 1,
                token_position: u64::from(row), route_offset: row * topk as u32, route_count: topk as u32,
            }).collect(),
            routes, wire)?;
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        let ranks = transport.world_size();
        ensure!(ranks <= MAX_RANKS, "{ranks} Spark ranks exceed the reduction planes");
        let build = built.elapsed().as_secs_f64();
        let timer = std::time::Instant::now();
        let mut dispatched = 0.0;
        runtime.block_on(async {
            let wave = transport.dispatch(&request)?;
            dispatched = timer.elapsed().as_secs_f64();
            transport.receive(wave, t, self.stream).await
        })?;
        let exchange = timer.elapsed().as_secs_f64();
        self.profile.borrow_mut()[1] += exchange;
        let reduced = std::time::Instant::now();
        // SAFETY: the zero plane and delta are live [t, h] BF16 buffers; the
        // intake planes are ordered after the wave by `receive`.
        unsafe {
            transport.reduce(w.zero_plane.buffer.ptr.cast(), w.delta.buffer.ptr.cast(), t, self.stream)?;
            if let Some(slot) = forward {
                self.push(0, slot, w.delta.buffer.ptr, t * h * 2)?;
            }
            // The next layer's request staging is rewritten only after this drains.
            self.library.cuda_stream_synchronize(self.stream)?;
        }
        if self.wave_timing {
            eprintln!("mimo_wave layer={index} rows={t} gpu_wait_ms={:.3} build_ms={:.3} dispatch_ms={:.3} \
                receive_ms={:.3} reduce_ms={:.3}", gpu_wait * 1e3, build * 1e3, dispatched * 1e3,
                (exchange - dispatched) * 1e3, reduced.elapsed().as_secs_f64() * 1e3);
        }
        Ok(())
    }
}
