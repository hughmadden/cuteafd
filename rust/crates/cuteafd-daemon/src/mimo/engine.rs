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
//! KV state: full layers keep BF16 records (keys then values of the 4 KV
//! heads, 1280 elements) in a paged pool shared by sequences (64 rows per
//! page); SWA layers keep a 256-slot ring per sequence (8 KV heads, 2560
//! elements). An SWA step's records go to a step buffer first; the attention
//! program reads in-step keys from it, older keys from the ring, and commits
//! the step to the ring afterwards.
use super::weights::{MimoLayer, MimoWeights};
use crate::v41_experts::fp8::{Fp8Experts, Fp8Layer};
use cuteafd_loader::fp8_experts::Fp8ExpertTensors;
use crate::v41_memory::{DeviceAllocation, HostAllocation};
use cuteafd_transport::v41_expert::{V41Tp4Roce, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::dsv4::{Dsv4Programs, Dsv4Scalar, VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::mimo_v2::{MimoAttention, MimoV2Config};
use std::cell::RefCell;
use std::ffi::c_void;

type Dev<'a> = DeviceAllocation<'a>;

pub(crate) const PAGE_ROWS: usize = 64;
pub(crate) const RING_ROWS: usize = 256;
/// Rows of the decode-route programs (`_m64`).
pub(crate) const DECODE_ROWS: usize = 64;
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
    Spark { transport: RefCell<V41Tp4Roce>, runtime: tokio::runtime::Runtime },
}

/// Most rows a decode program reads the FP8 weight copies for (MmaFp8Gemv's M tile).
pub(crate) const FP8_ROWS: i32 = 16;

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

/// A sequence's full-attention pages, its SWA ring and its length.
#[derive(Debug, Clone)]
pub(crate) struct MimoPlacement {
    pub pages: Vec<i32>,
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

/// Free pages of the full-attention pool and free SWA rings.
pub(crate) struct Allocator {
    pages: Vec<i32>,
    rings: Vec<i32>,
}

impl Allocator {
    pub fn new(pages: usize, rings: usize) -> Self {
        Self { pages: (0..pages as i32).rev().collect(), rings: (0..rings as i32).rev().collect() }
    }

    /// Reserves every page a sequence of up to `capacity` tokens needs, and a ring.
    pub fn admit(&mut self, capacity: usize) -> Result<MimoPlacement> {
        let pages = capacity.div_ceil(PAGE_ROWS).max(1);
        ensure!(self.pages.len() >= pages, "cache pages exhausted ({pages} needed, {} free)", self.pages.len());
        let ring = self.rings.pop().context("SWA rings exhausted")?;
        Ok(MimoPlacement { pages: (0..pages).map(|_| self.pages.pop().unwrap()).collect(), ring, len: 0 })
    }

    /// Returns a finished sequence's pages and ring.
    pub fn release(&mut self, placement: MimoPlacement) {
        self.pages.extend(placement.pages);
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
    planes: Vec<Dev<'a>>,
    zero_plane: Dev<'a>,
    router_host: RefCell<HostAllocation<'a>>,
    planes_host: RefCell<HostAllocation<'a>>,
    head: VocabularyHead<'a>,
    _head_workspace: Dev<'a>,
}

pub(crate) struct MimoEngine<'a> {
    pub library: &'a NativeLibrary,
    pub programs: &'a Dsv4Programs<'a>,
    pub cfg: MimoV2Config,
    pub weights: MimoWeights<'a>,
    pub stream: *mut c_void,
    pub max_context: usize,
    pub prefill_rows: usize,
    pub pages: usize,
    pub rings: usize,
    /// Program family of the checkpoint's geometry (`mimo`, `mimop`).
    family: &'static str,
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
    /// The DFlash drafter (V2.6 Pro's dflash/): every step taps its target layers.
    pub drafter: Option<super::dflash::MimoDrafter<'a>>,
    /// The native MTP drafter: every step taps the last layer's rows.
    pub mtp: Option<super::mtp::MtpDrafter<'a>>,
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// `[rows]`, plus the decode programs' `fp8_rows` (16 when the layer has the FP8 copy, else 0).
fn fp8_scalars(rows: Dsv4Scalar, decode: bool, fp8: bool) -> Vec<Dsv4Scalar> {
    let mut scalars = vec![rows];
    if decode {
        scalars.push(Dsv4Scalar::I32(if fp8 { FP8_ROWS } else { 0 }));
    }
    scalars
}

fn kind(attention: MimoAttention) -> &'static str {
    match attention {
        MimoAttention::Full => "full",
        MimoAttention::Sliding => "swa",
    }
}

impl<'a> MimoEngine<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(library: &'a NativeLibrary, programs: &'a Dsv4Programs<'a>, cfg: MimoV2Config,
        weights: MimoWeights<'a>, stream: *mut c_void, max_context: usize, prefill_rows: usize, pages: usize,
        rings: usize) -> Result<Self> {
        let family = cfg.program_family()?;
        ensure!(cfg.rope_dim == 64 && cfg.head_dim == 192 && cfg.v_head_dim == 128 && cfg.window <= RING_ROWS - DECODE_ROWS,
            "the mimo programs are built for 192/128 heads, 64 RoPE dims and a window of at most {}",
            RING_ROWS - DECODE_ROWS);
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let kv = weights.layers.iter().map(|layer| {
            let record = cfg.record_elems(layer.attention) * 2;
            zeroed(match layer.attention {
                MimoAttention::Full => pages * PAGE_ROWS * record,
                MimoAttention::Sliding => rings * RING_ROWS * record,
            })
        }).collect::<Result<Vec<_>>>()?;
        // cos | sin of position * theta^(-2i/64), FP32 like the reference's inv_freq.
        let table = |theta: f64| -> Result<Dev<'a>> {
            let dim = cfg.rope_dim;
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
        };
        let (cos_sin_full, cos_sin_swa) = (table(cfg.full_rope_theta)?, table(cfg.swa_rope_theta)?);
        Ok(Self { library, programs, cfg, weights, stream, max_context, prefill_rows, pages, rings, family, kv, cos_sin_full,
            cos_sin_swa, workspace: RefCell::new(None), decode_workspace: RefCell::new(None), experts: None,
            expert_input: ExpertInput::Fp8,
            profile: RefCell::new([0.0; 2]), drafter: None, mtp: None })
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
    fn program_name(&self, name: &str) -> String {
        match name.strip_prefix("mimo_") {
            Some(rest) => format!("{}_{rest}", self.family),
            None => name.to_string(),
        }
    }

    fn run(&self, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Dsv4Scalar]) -> Result<()> {
        let name = &self.program_name(name);
        let names: Vec<&str> = pointers.iter().map(|(n, _)| *n).collect();
        let program = self.programs.program(name, &names)?;
        let raw: Vec<*mut c_void> = pointers.iter().map(|(_, p)| *p).collect();
        // SAFETY: every pointer names a live allocation sized for the rows in
        // `scalars`; the stream orders all launches of this engine.
        unsafe { program.launch(&raw, scalars, self.stream) }.with_context(|| format!("{name} with {scalars:?}"))
    }

    fn scratch(&self, name: &str) -> Result<usize> {
        Ok(self.programs.spec(&self.program_name(name))?.scratch.get("scratch").copied().unwrap_or(0) as usize)
    }

    fn workspace(&self, t: usize, decode: bool) -> Result<Workspace<'a>> {
        let (h, heads) = (self.cfg.hidden, self.cfg.heads);
        let (cap, mode) = if decode { ("m64", "decode") } else { ("m4096", "prefill") };
        let mut scratch = self.scratch("mimo_router_scores")?;
        for name in [format!("mimo_full_producer_{cap}"), format!("mimo_swa_producer_{cap}"),
            format!("mimo_full_attention_{mode}_{cap}"), format!("mimo_swa_attention_{mode}_{cap}"),
            format!("mimo_ffn_{cap}")] {
            scratch = scratch.max(self.scratch(&name)?);
        }
        let head_workspace = self.alloc(VOCABULARY_HEAD_WORKSPACE)?;
        let spark = matches!(self.experts, Some(Experts::Spark { .. }));
        let identity: Vec<i64> = (0..t as i64).collect();
        let step_slots = self.alloc(t * 8)?;
        self.library.copy_h2d(step_slots.buffer, bytes_of(&identity))?;
        let record = self.cfg.record_elems(MimoAttention::Sliding).max(self.cfg.record_elems(MimoAttention::Full));
        Ok(Workspace {
            rows: t,
            h: self.alloc(t * h * 2)?,
            x: self.alloc(t * h * 2)?,
            query: self.alloc(t * heads * self.cfg.head_dim * 2)?,
            attn: self.alloc(t * heads * self.cfg.v_head_dim * 2)?,
            delta: self.alloc(t * h * 2)?,
            kv_step: self.alloc(t * record * 2)?,
            positions: self.alloc(t * 8)?,
            slots: self.alloc(t * 8)?,
            step_slots,
            ring_slots: self.alloc(t * 8)?,
            seq_first: self.alloc(t * 4)?,
            page_table: self.alloc(if decode { t * self.pages * 4 } else { self.pages * 4 })?,
            scratch: self.alloc(scratch)?,
            logits: self.alloc(t * self.cfg.vocab_size * 4)?,
            router_logits: self.alloc(t * self.cfg.experts * 4)?,
            route_ids: self.alloc(t * self.cfg.topk * 4)?,
            route_weights: self.alloc(t * self.cfg.topk * 4)?,
            wire: self.alloc(t * (h + h / 32))?,
            planes: if spark { (0..MAX_RANKS).map(|_| self.alloc(t * h * 2)).collect::<Result<_>>()? } else { Vec::new() },
            zero_plane: {
                let zero = self.alloc(if spark { t * h * 2 } else { 256 })?;
                self.library.cuda_zero_bytes(zero.buffer, zero.buffer.bytes)?;
                zero
            },
            router_host: RefCell::new(HostAllocation::new(self.library,
                if spark { t * (self.cfg.topk * 8 + 2 * h) } else { 256 })?),
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

    /// Prefills a sequence from its length through every resident layer and
    /// returns the last row's logits when all layers are resident.
    /// `on_layer` receives each layer's output rows (BF16 [t, hidden]).
    pub fn prefill(&self, placement: &mut MimoPlacement, embed: &[u8],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        self.prefill_forced(placement, embed, false, on_layer, None)
    }

    /// `prefill` with teacher forcing: after layer `l`, `forced(l)` (when it
    /// returns rows) replaces the residual before layer `l + 1`, so each
    /// layer's comparison measures that layer alone. `all_logits` returns
    /// every row's logits instead of the last row's.
    pub fn prefill_forced(&self, placement: &mut MimoPlacement, embed: &[u8], all_logits: bool,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>) -> Result<Option<Vec<f32>>> {
        let (t, start) = (embed.len() / (self.cfg.hidden * 2), placement.len);
        ensure!(t > 0 && t <= self.prefill_rows && start + t <= self.max_context, "prefill of {t} rows at {start}");
        let used = (start + t).div_ceil(PAGE_ROWS);
        let tables = StepTables {
            decode: false,
            positions: (start..start + t).map(|p| p as i64).collect(),
            slots: (start..start + t).map(|p| placement.slot(p)).collect::<Result<_>>()?,
            ring_slots: (start..start + t).map(|p| placement.ring_slot(p)).collect(),
            seq_first: vec![0; t],
            page_table: placement.pages[..used].to_vec(),
            table_stride: 0,
        };
        let logits = self.step(&tables, embed, if all_logits { t } else { 1 }, on_layer, forced)?;
        placement.len += t;
        Ok(logits)
    }

    /// Appends each sequence's tokens (one for decode, several for a
    /// speculative verify) at its length in one decode-shaped step; returns
    /// every row's logits. A caller that rejects a suffix sets `len` back:
    /// the 256-slot rings keep every key a later step can still need.
    pub fn verify(&self, sequences: &mut [(&mut MimoPlacement, usize)], embed: &[u8],
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        let rows: usize = sequences.iter().map(|(_, n)| n).sum();
        ensure!(rows > 0 && rows <= DECODE_ROWS && embed.len() == rows * self.cfg.hidden * 2, "decode step of {rows} rows");
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
                let mut pages = placement.pages.clone();
                pages.resize(stride, 0);
                tables.page_table.extend(pages);
            }
        }
        let logits = self.step(&tables, embed, rows, on_layer, None)?;
        for (placement, count) in sequences.iter_mut() {
            placement.len += *count;
        }
        Ok(logits)
    }

    fn step(&self, tables: &StepTables, embed: &[u8], logit_rows: usize,
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>) -> Result<Option<Vec<f32>>> {
        let (h, t) = (self.cfg.hidden, tables.positions.len());
        let (slot, capacity) = if tables.decode { (&self.decode_workspace, DECODE_ROWS) } else { (&self.workspace, self.prefill_rows) };
        if slot.borrow().is_none() {
            *slot.borrow_mut() = Some(self.workspace(capacity, tables.decode)?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        ensure!(t <= w.rows && logit_rows <= t, "step exceeds the workspace");
        self.put(&w.positions, &tables.positions)?;
        self.put(&w.slots, &tables.slots)?;
        self.put(&w.ring_slots, &tables.ring_slots)?;
        self.put(&w.seq_first, &tables.seq_first)?;
        self.put(&w.page_table, &tables.page_table)?;
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: embed.len(), ..w.h.buffer }, embed)?;
        let rows = Dsv4Scalar::I32(t as i32);
        let cap = if tables.decode { "m64" } else { "m4096" };
        let layers = &self.weights.layers;
        self.norm(w, layers[0].ptr("input_norm")?, 0, rows)?;
        for (index, layer) in layers.iter().enumerate() {
            self.attention(w, self.kv[index].buffer.ptr, layer, rows, cap, tables)?;
            // h += attention; x = post_attention_layernorm(h)
            self.norm(w, layer.ptr("post_norm")?, 1, rows)?;
            if layer.dense {
                let mut pointers = vec![("x", w.x.buffer.ptr), ("w_gate_up", layer.ptr("w_gate_up")?)];
                if tables.decode {
                    pointers.extend([("w_gate_up_fp8", layer.ptr_or("w_gate_up_fp8", "w_gate_up")?),
                        ("w_gate_up_scale", layer.ptr_or("w_gate_up_scale", "w_gate_up")?)]);
                }
                pointers.push(("w_down", layer.ptr("w_down")?));
                if tables.decode {
                    pointers.extend([("w_down_fp8", layer.ptr_or("w_down_fp8", "w_down")?),
                        ("w_down_scale", layer.ptr_or("w_down_scale", "w_down")?)]);
                }
                pointers.extend([("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
                self.run(&format!("mimo_ffn_{cap}"), &pointers, &fp8_scalars(rows, tables.decode, layer.has("w_down_fp8")))?;
            } else {
                self.moe(w, index, layer, t, tables.decode)?;
            }
            // h += ffn; x = next input_layernorm(h) (or the final norm).
            let weight = match layers.get(index + 1) {
                Some(next) => next.ptr("input_norm")?,
                None => self.weights.norm.buffer.ptr,
            };
            self.norm(w, weight, 1, rows)?;
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
                    ("logits", w.logits.buffer.ptr)], &[Dsv4Scalar::I32(logit_rows as i32)])?;
            }
            // SAFETY: the final norm's output and the head operands are live buffers of these shapes.
            _ => unsafe {
                w.head.launch(x.cast(), self.weights.head.buffer.ptr.cast(), w.logits.buffer.ptr.cast(),
                    logit_rows as u32, self.stream)?;
            },
        }
        let logits = self.download(&w.logits, logit_rows * self.cfg.vocab_size * 4)?;
        Ok(Some(logits.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()))
    }

    /// `residual (h) += delta` when `deltas` is 1, then `x = weight * RMSNorm(h)`.
    fn norm(&self, w: &Workspace<'_>, weight: *mut c_void, deltas: i32, rows: Dsv4Scalar) -> Result<()> {
        self.run("mimo_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
            ("delta1", w.delta.buffer.ptr), ("weight", weight), ("out", w.x.buffer.ptr)],
            &[rows, Dsv4Scalar::I32(deltas)])
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
    /// `mtp`); `embed` maps tokens to BF16 embedding rows.
    pub fn mtp_draft(&self, seqs: &[super::mtp::MtpSeq<'_>], stages: usize,
        embed: &dyn Fn(&[u32]) -> Result<Vec<u8>>) -> Result<Vec<Vec<u32>>> {
        let mtp = self.mtp.as_ref().context("no MTP drafter")?;
        let stages = stages.min(mtp.stages.len());
        let mut drafts: Vec<Vec<u32>> = vec![Vec::new(); seqs.len()];
        for k in 0..stages {
            // Catch-up passes over true rows while a sequence's rows exceed the step.
            loop {
                let mut groups = Vec::new();
                let mut rows = 0;
                for seq in seqs {
                    let ext = mtp.ext.borrow()[seq.ring][k];
                    ensure!(ext <= seq.len, "MTP ring {} is ahead of its sequence (reset it at admission)", seq.ring);
                    // True rows are those whose token t_{j+k+1} is known (j <= len - k - 1).
                    let true_end = seq.len.saturating_sub(k);
                    if seq.len - ext > MTP_STEP_ROWS && ext < true_end && rows < DECODE_ROWS {
                        let n = (true_end - ext).min(DECODE_ROWS - rows);
                        groups.push((seq.ring, ext, (ext..ext + n).map(|j| seq.tokens[j + k + 1]).collect::<Vec<_>>()));
                        rows += n;
                    }
                }
                if groups.is_empty() {
                    break;
                }
                self.mtp_pass(mtp, k, &groups, embed, false)?;
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
                    let tokens: Vec<u32> = (ext..seq.len).map(|j| {
                        let at = j + k + 1;
                        if at <= seq.len { seq.tokens[at] } else { drafts[index][at - seq.len - 1] }
                    }).collect();
                    groups.push((seq.ring, ext, tokens));
                    members.push(index);
                    rows += n;
                    index += 1;
                }
                let tops = self.mtp_pass(mtp, k, &groups, embed, true)?;
                let mut ext = mtp.ext.borrow_mut();
                for ((&i, (ring, _, _)), top) in members.iter().zip(&groups).zip(tops) {
                    drafts[i].push(top);
                    ext[*ring][k] = ext[*ring][k].max(seqs[i].len.saturating_sub(k));
                }
            }
        }
        Ok(drafts)
    }

    /// One MTP stage over `groups` of (ring, first row, tokens t_{j+k+1});
    /// with `logits`, returns the argmax of each group's last row.
    fn mtp_pass(&self, mtp: &super::mtp::MtpDrafter<'_>, k: usize, groups: &[(usize, usize, Vec<u32>)],
        embed: &dyn Fn(&[u32]) -> Result<Vec<u8>>, logits: bool) -> Result<Vec<u32>> {
        use super::mtp::HIDDEN_ROWS;
        let stage = &mtp.stages[k];
        let (h, vocab) = (self.cfg.hidden, self.cfg.vocab_size);
        let mut tables = StepTables { decode: true, positions: Vec::new(), slots: Vec::new(), ring_slots: Vec::new(),
            seq_first: Vec::new(), page_table: vec![0], table_stride: 1 };
        let mut all_tokens = Vec::new();
        for (ring, first, tokens) in groups {
            ensure!(*ring < self.rings, "MTP ring {ring} of {}", self.rings);
            let start = tables.positions.len() as i32;
            for j in *first..first + tokens.len() {
                tables.positions.push(j as i64);
                tables.slots.push(-1);
                tables.ring_slots.push((ring * RING_ROWS + j % RING_ROWS) as i64);
                tables.seq_first.push(start);
            }
            all_tokens.extend_from_slice(tokens);
        }
        let t = tables.positions.len();
        ensure!(t > 0 && t <= DECODE_ROWS, "MTP pass of {t} rows");
        if self.decode_workspace.borrow().is_none() {
            *self.decode_workspace.borrow_mut() = Some(self.workspace(DECODE_ROWS, true)?);
        }
        let workspace = self.decode_workspace.borrow();
        let w = workspace.as_ref().context("workspace")?;
        self.put(&w.positions, &tables.positions)?;
        self.put(&w.ring_slots, &tables.ring_slots)?;
        self.put(&w.seq_first, &tables.seq_first)?;
        self.put(&w.page_table, &tables.page_table)?;
        let row = h * 2;
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: t * row, ..mtp.embed.buffer },
            &embed(&all_tokens)?)?;
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
        let rows = Dsv4Scalar::I32(t as i32);
        for (source, weight, out) in [(&mtp.embed, &stage.enorm, &mtp.normed_e), (&mtp.rows_h, &stage.hnorm, &mtp.normed_h)] {
            self.run("mimo_norm", &[("residual", source.buffer.ptr), ("delta0", source.buffer.ptr),
                ("delta1", source.buffer.ptr), ("weight", weight.buffer.ptr), ("out", out.buffer.ptr)],
                &[rows, Dsv4Scalar::I32(0)])?;
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
        let mut pointers = vec![("x", w.x.buffer.ptr), ("w_gate_up", layer.ptr("w_gate_up")?),
            ("w_gate_up_fp8", layer.ptr_or("w_gate_up_fp8", "w_gate_up")?),
            ("w_gate_up_scale", layer.ptr_or("w_gate_up_scale", "w_gate_up")?), ("w_down", layer.ptr("w_down")?),
            ("w_down_fp8", layer.ptr_or("w_down_fp8", "w_down")?), ("w_down_scale", layer.ptr_or("w_down_scale", "w_down")?)];
        pointers.extend([("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
        self.run("mimo_ffn_m64", &pointers, &fp8_scalars(rows, true, layer.has("w_down_fp8")))?;
        self.norm(w, stage.final_norm.buffer.ptr, 1, rows)?;
        if !logits {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(Vec::new());
        }
        match &self.weights.head_fp8 {
            Some((q, scale)) if t <= FP8_ROWS as usize => {
                self.run("mimo_head_fp8", &[("x", w.x.buffer.ptr), ("w_fp8", q.buffer.ptr), ("scale", scale.buffer.ptr),
                    ("logits", w.logits.buffer.ptr)], &[rows])?;
            }
            // SAFETY: the final norm's output and the head operands are live buffers of these shapes.
            _ => unsafe {
                w.head.launch(w.x.buffer.ptr.cast(), self.weights.head.buffer.ptr.cast(), w.logits.buffer.ptr.cast(),
                    t as u32, self.stream)?;
            },
        }
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        let mut tops = Vec::with_capacity(groups.len());
        let mut end = 0;
        let mut bytes = vec![0u8; vocab * 4];
        for (_, _, tokens) in groups {
            end += tokens.len();
            self.library.copy_d2h(&mut bytes, cuteafd_ffi::CuteafdDeviceBuffer {
                // SAFETY: row end - 1 lies inside the logits buffer.
                ptr: unsafe { w.logits.buffer.ptr.cast::<u8>().add((end - 1) * vocab * 4) }.cast(),
                bytes: vocab * 4, ..w.logits.buffer })?;
            let best = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1)).map_or(0, |(i, _)| i);
            tops.push(best as u32);
        }
        Ok(tops)
    }

    /// `kv`: the layer's paged record pool (full) or its rings (SWA).
    fn attention(&self, w: &Workspace<'_>, kv: *mut c_void, layer: &MimoLayer<'_>, rows: Dsv4Scalar, cap: &str,
        tables: &StepTables) -> Result<()> {
        let mode = if tables.decode { "decode" } else { "prefill" };
        let k = kind(layer.attention);
        let (slots, cos_sin) = match layer.attention {
            MimoAttention::Full => (w.slots.buffer.ptr, self.cos_sin_full.buffer.ptr),
            MimoAttention::Sliding => (w.step_slots.buffer.ptr, self.cos_sin_swa.buffer.ptr),
        };
        let records = match layer.attention {
            MimoAttention::Full => kv,
            MimoAttention::Sliding => w.kv_step.buffer.ptr,
        };
        let decode = tables.decode;
        let mut pointers = vec![("x", w.x.buffer.ptr), ("positions", w.positions.buffer.ptr), ("kv_slots", slots),
            ("cos_sin", cos_sin), ("w_qkv", layer.ptr("w_qkv")?)];
        if decode {
            pointers.extend([("w_qkv_fp8", layer.ptr_or("w_qkv_fp8", "w_qkv")?),
                ("w_qkv_scale", layer.ptr_or("w_qkv_scale", "w_qkv")?)]);
        }
        pointers.extend([("kv_cache", records), ("query", w.query.buffer.ptr), ("scratch", w.scratch.buffer.ptr)]);
        self.run(&format!("mimo_{k}_producer_{cap}"), &pointers, &fp8_scalars(rows, decode, layer.has("w_qkv_fp8")))?;
        let name = format!("mimo_{k}_attention_{mode}_{cap}");
        match layer.attention {
            MimoAttention::Full => {
                let mut scalars = vec![rows, Dsv4Scalar::I32(tables.table_stride as i32)];
                if tables.decode {
                    scalars.push(Dsv4Scalar::I32(DECODE_SPLITS));
                }
                self.run(&name, &[("q", w.query.buffer.ptr), ("kv_cache", kv),
                    ("positions", w.positions.buffer.ptr), ("page_table", w.page_table.buffer.ptr),
                    ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &scalars)?;
            }
            MimoAttention::Sliding => {
                self.run(&name, &[("q", w.query.buffer.ptr), ("kv_step", w.kv_step.buffer.ptr),
                    ("ring", kv), ("positions", w.positions.buffer.ptr),
                    ("ring_slots", w.ring_slots.buffer.ptr), ("seq_first", w.seq_first.buffer.ptr),
                    ("sinks", layer.ptr("sinks")?), ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
                    &[rows])?;
            }
        }
        let mut pointers = vec![("attn", w.attn.buffer.ptr), ("w_o", layer.ptr("w_o")?)];
        if decode {
            pointers.extend([("w_o_fp8", layer.ptr_or("w_o_fp8", "w_o")?), ("w_o_scale", layer.ptr_or("w_o_scale", "w_o")?)]);
        }
        pointers.push(("out", w.delta.buffer.ptr));
        self.run(&format!("mimo_o_{cap}"), &pointers, &fp8_scalars(rows, decode, layer.has("w_o_fp8")))
    }

    /// Router scores, the sigmoid top-k select and the FP8 wire rows, then the
    /// routed experts (local or Spark); leaves their sum in `delta` for the
    /// next norm's residual add.
    fn moe(&self, w: &Workspace<'_>, index: usize, layer: &MimoLayer<'_>, t: usize, decode: bool) -> Result<()> {
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let experts = self.experts.as_ref().with_context(|| format!(
            "layer {index} is an MoE layer: pass Spark --peers serving the fp8 family, or --local-experts \
             (run --layers 1 for the dense layer alone)"))?;
        let rows = Dsv4Scalar::I32(t as i32);
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
                ("scale_mma_ptr", w.delta.buffer.ptr)], &[rows, Dsv4Scalar::I32(grid as i32)])?;
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
                self.spark_moe(w, index, t, decode, bf16_input, &mut transport.borrow_mut(), runtime)
            }
        }
    }

    /// Routes and wire rows down, one request to every Spark rank, the BF16
    /// rank partials summed into `delta` (GLM's exchange, no shared expert).
    #[allow(clippy::too_many_arguments)]
    fn spark_moe(&self, w: &Workspace<'_>, index: usize, t: usize, decode: bool, bf16_input: bool,
        transport: &mut V41Tp4Roce, runtime: &tokio::runtime::Runtime) -> Result<()> {
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
        // SAFETY: the pinned regions are large enough; the sync completes them.
        unsafe {
            self.library.copy_d2h_host_buffer_async(at(0), w.route_ids.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(route_bytes), w.route_weights.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(2 * route_bytes), input.buffer, wire_bytes, self.stream)?;
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
        let mut request = ExpertProtocolV2Request::new(index as u64 + 1, 17, index as u32, h as u32, dtype,
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
        // SAFETY: planes, the zero plane and delta are live [t, h] BF16 buffers ordered after the uploads.
        unsafe {
            self.library.v41_compact_reducer()?.reduce_planes(pointers, ranks as u32, w.zero_plane.buffer.ptr.cast(),
                w.delta.buffer.ptr.cast(), t as u32, self.stream)?;
            // The staging is rewritten by the next layer only after this upload drains.
            self.library.cuda_stream_synchronize(self.stream)
        }
    }
}
