//! MiMo V2 (mimo_v2_flash) coordinator over the exported mimo_* programs.
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
use crate::v41_experts::fp8::Fp8Experts;
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

/// Where the routed experts run.
pub(crate) enum Experts<'a> {
    /// The TP1 FP8 package on the coordinator GPU (resident MoE layers).
    Local(Fp8Experts<'a>),
    /// Spark ranks serving the `fp8` family over RoCE.
    Spark { transport: RefCell<V41Tp4Roce>, runtime: tokio::runtime::Runtime },
}

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
    /// Per layer: the paged record pool (full) or the rings (SWA).
    kv: Vec<Dev<'a>>,
    cos_sin_full: Dev<'a>,
    cos_sin_swa: Dev<'a>,
    workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    experts: Option<Experts<'a>>,
    /// Host time per phase: GPU wait before the expert request, the Spark exchange.
    pub profile: RefCell<[f64; 2]>,
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
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
        Ok(Self { library, programs, cfg, weights, stream, max_context, prefill_rows, pages, rings, kv, cos_sin_full,
            cos_sin_swa, workspace: RefCell::new(None), decode_workspace: RefCell::new(None), experts: None,
            profile: RefCell::new([0.0; 2]) })
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

    fn run(&self, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Dsv4Scalar]) -> Result<()> {
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
                if spark { t * (self.cfg.topk * 8 + h + h / 32) } else { 256 })?),
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
            self.attention(w, index, layer, rows, cap, tables)?;
            // h += attention; x = post_attention_layernorm(h)
            self.norm(w, layer.ptr("post_norm")?, 1, rows)?;
            if layer.dense {
                self.run(&format!("mimo_ffn_{cap}"), &[("x", w.x.buffer.ptr), ("w_gate_up", layer.ptr("w_gate_up")?),
                    ("w_down", layer.ptr("w_down")?), ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
                    &[rows])?;
            } else {
                self.moe(w, index, layer, t, tables.decode)?;
            }
            // h += ffn; x = next input_layernorm(h) (or the final norm).
            let weight = match layers.get(index + 1) {
                Some(next) => next.ptr("input_norm")?,
                None => self.weights.norm.buffer.ptr,
            };
            self.norm(w, weight, 1, rows)?;
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
        // SAFETY: the final norm's output and the head operands are live buffers of these shapes.
        unsafe {
            w.head.launch(w.x.buffer.ptr.cast::<u8>().add((t - logit_rows) * h * 2).cast(),
                self.weights.head.buffer.ptr.cast(), w.logits.buffer.ptr.cast(), logit_rows as u32, self.stream)?;
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

    fn attention(&self, w: &Workspace<'_>, index: usize, layer: &MimoLayer<'_>, rows: Dsv4Scalar, cap: &str,
        tables: &StepTables) -> Result<()> {
        let mode = if tables.decode { "decode" } else { "prefill" };
        let k = kind(layer.attention);
        let (slots, cos_sin) = match layer.attention {
            MimoAttention::Full => (w.slots.buffer.ptr, self.cos_sin_full.buffer.ptr),
            MimoAttention::Sliding => (w.step_slots.buffer.ptr, self.cos_sin_swa.buffer.ptr),
        };
        let records = match layer.attention {
            MimoAttention::Full => self.kv[index].buffer.ptr,
            MimoAttention::Sliding => w.kv_step.buffer.ptr,
        };
        self.run(&format!("mimo_{k}_producer_{cap}"), &[("x", w.x.buffer.ptr), ("positions", w.positions.buffer.ptr),
            ("kv_slots", slots), ("cos_sin", cos_sin), ("w_qkv", layer.ptr("w_qkv")?), ("kv_cache", records),
            ("query", w.query.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        let name = format!("mimo_{k}_attention_{mode}_{cap}");
        match layer.attention {
            MimoAttention::Full => {
                let mut scalars = vec![rows, Dsv4Scalar::I32(tables.table_stride as i32)];
                if tables.decode {
                    scalars.push(Dsv4Scalar::I32(DECODE_SPLITS));
                }
                self.run(&name, &[("q", w.query.buffer.ptr), ("kv_cache", self.kv[index].buffer.ptr),
                    ("positions", w.positions.buffer.ptr), ("page_table", w.page_table.buffer.ptr),
                    ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &scalars)?;
            }
            MimoAttention::Sliding => {
                self.run(&name, &[("q", w.query.buffer.ptr), ("kv_step", w.kv_step.buffer.ptr),
                    ("ring", self.kv[index].buffer.ptr), ("positions", w.positions.buffer.ptr),
                    ("ring_slots", w.ring_slots.buffer.ptr), ("seq_first", w.seq_first.buffer.ptr),
                    ("sinks", layer.ptr("sinks")?), ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)],
                    &[rows])?;
            }
        }
        self.run(&format!("mimo_o_{cap}"), &[("attn", w.attn.buffer.ptr), ("w_o", layer.ptr("w_o")?),
            ("out", w.delta.buffer.ptr)], &[rows])
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
        self.run("mimo_router_scores", &[("x", w.x.buffer.ptr), ("w_hilo", layer.ptr("w_hilo")?),
            ("logits", w.router_logits.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        // SAFETY: logits, bias and route outputs are live buffers of `t` rows.
        unsafe {
            self.library.router_select(w.router_logits.buffer.ptr, layer.ptr("gate.bias")?, std::ptr::null(),
                std::ptr::null(), w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, t, self.cfg.experts, topk,
                self.cfg.routed_scale as f32, true, self.stream)?;
        }
        let grid = (t * h.div_ceil(256)).div_ceil(8).clamp(1, 4 * 188);
        self.run("mimo_expert_input_quant", &[("source_ptr", w.x.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
            // SAFETY: the scale rows follow the payload inside each wire row.
            ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
            ("scale_mma_ptr", w.delta.buffer.ptr)], &[rows, Dsv4Scalar::I32(grid as i32)])?;
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
            Experts::Spark { transport, runtime } => {
                self.spark_moe(w, index, t, decode, &mut transport.borrow_mut(), runtime)
            }
        }
    }

    /// Routes and wire rows down, one request to every Spark rank, the BF16
    /// rank partials summed into `delta` (GLM's exchange, no shared expert).
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
        // SAFETY: planes, the zero plane and delta are live [t, h] BF16 buffers ordered after the uploads.
        unsafe {
            self.library.v41_compact_reducer()?.reduce_planes(pointers, ranks as u32, w.zero_plane.buffer.ptr.cast(),
                w.delta.buffer.ptr.cast(), t as u32, self.stream)?;
            // The staging is rewritten by the next layer only after this upload drains.
            self.library.cuda_stream_synchronize(self.stream)
        }
    }
}
