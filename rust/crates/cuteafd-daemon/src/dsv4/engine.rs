//! DeepSeek V4 coordinator: one sequence prefilled from position 0 through the
//! exported programs, routed experts on the Spark ranks.
//!
//! This is the correctness path the serving engine grows from: every layer
//! runs the prototype's op sequence (mHC pre, producer, compressor, index
//! top-k, sparse MLA, wo, mHC post_pre, router, experts + shared FFN, mHC
//! post) and each stage's buffers are sized for the prompt.
use super::metadata::{self, StepTables, INDEX_PAGE_BYTES, MAIN_PAGE_BYTES};
use super::pool::{Placement, PoolShape};
use std::cell::RefCell;
use super::weights::{LayerWeights, ModelWeights};
use crate::v41_memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::dsv4::{Dsv4Program, Dsv4Programs, Dsv4Scalar};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::deepseek_v4::DeepseekV4Config;
use cuteafd_transport::v41_expert::{V41Tp4Roce, V41Tp4RoceWave, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype,
    ExpertV2SourceKind,
};
use std::ffi::c_void;
use std::time::Instant;

type Dev<'a> = DeviceAllocation<'a>;

pub(crate) struct Engine<'a> {
    pub library: &'a NativeLibrary,
    pub programs: &'a Dsv4Programs<'a>,
    pub cfg: DeepseekV4Config,
    pub weights: ModelWeights<'a>,
    pub family: &'static str,
    pub decode_rows: usize,
    pub prefill_rows: usize,
    pub c128_width: usize,
    /// Longest sequence the exported programs' cache extents cover.
    pub max_context: usize,
    pub stream: *mut c_void,
    pub sms: u32,
    pub shape: PoolShape,
    pools: Vec<LayerCache<'a>>,
    rope_window: Dev<'a>,
    rope_compressed: Dev<'a>,
    /// Prefill and decode workspaces, reused across steps.
    prefill_workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
    pub profile: RefCell<Profile>,
    graphs: RefCell<std::collections::HashMap<GraphKey, GraphExec<'a>>>,
    /// Expert layers resident on this GPU (they skip the Spark exchange).
    pub local: RefCell<Option<super::local::LocalExperts<'a>>>,
}

/// `exchange` result for a layer whose experts ran on this GPU.
const LOCAL_EXPERTS: usize = usize::MAX;

/// Lanes a long prefill chunk splits into, and the fewest rows per lane worth
/// a second Spark exchange per layer.
pub(crate) const PREFILL_LANES: usize = 2;
const MIN_LANE_ROWS: usize = 256;

/// One lane's inputs for a step.
struct LaneStep<'s> {
    tables: &'s StepTables,
    tokens: &'s [u32],
    embed: &'s [u8],
}

/// Everything a captured decode segment bakes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GraphKey {
    layer: usize,
    rows: usize,
    attention: &'static str,
    table_width: usize,
    table_stride: usize,
    previous: usize,
}

struct GraphExec<'a>(*mut c_void, &'a NativeLibrary);

impl Drop for GraphExec<'_> {
    fn drop(&mut self) {
        // SAFETY: the exec came from end_capture and is destroyed once.
        let _ = unsafe { self.1.cuda_graph_exec_destroy(self.0) };
    }
}

/// Host-visible phases of a step, for profiling.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Phase {
    /// Waiting for the GPU through the router and input quantizer.
    RouterSync,
    Routing,
    /// Dispatch, remote compute and partials landing in device planes.
    Experts,
    /// Waiting for the head and downloading logits.
    Head,
}

#[derive(Debug, Default)]
pub(crate) struct Profile {
    pub seconds: [f64; 4],
}

impl Profile {
    fn add(&mut self, phase: Phase, since: Instant) {
        self.seconds[phase as usize] += since.elapsed().as_secs_f64();
    }

    pub fn report(&self) -> String {
        let names = ["router_sync", "routing", "experts", "head"];
        names.iter().zip(self.seconds).map(|(n, s)| format!("{n} {:.1} ms", s * 1e3)).collect::<Vec<_>>().join(", ")
    }
}

/// Everything an [`Engine`] needs besides its pools and workspaces.
pub(crate) struct EngineParts<'a> {
    pub library: &'a NativeLibrary,
    pub programs: &'a Dsv4Programs<'a>,
    pub cfg: DeepseekV4Config,
    pub weights: ModelWeights<'a>,
    pub family: &'static str,
    pub decode_rows: usize,
    pub prefill_rows: usize,
    pub c128_width: usize,
    pub max_context: usize,
    pub stream: *mut c_void,
    pub sms: u32,
    pub shape: PoolShape,
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// Per-layer caches and compressor state for one sequence.
pub(crate) struct LayerCache<'a> {
    pub main: Dev<'a>,
    pub compressed: Option<Dev<'a>>,
    pub index: Option<Dev<'a>>,
    pub states: Vec<Dev<'a>>,
}

/// What one lane of rows carries from layer to layer: the residual streams,
/// the mHC post/comb of its current layer, its token ids and step tables.
struct Lane<'a> {
    stream_a: Dev<'a>,
    stream_b: Dev<'a>,
    post: Dev<'a>,
    comb: Dev<'a>,
    /// Token ids (hash-layer routing).
    tokens: Dev<'a>,
    /// Shared-expert output of the lane's current layer (its post reads it
    /// after the other lane's shared expert ran).
    shared: Dev<'a>,
    tables: StepBuffers<'a>,
}

/// Step workspace sized for `rows` rows per lane; everything but the lanes is
/// consumed within one (layer, lane) unit and shared.
struct Workspace<'a> {
    lanes: Vec<Lane<'a>>,
    y: Dev<'a>,
    query: Dev<'a>,
    q_rank: Dev<'a>,
    attn_out: Dev<'a>,
    delta: Dev<'a>,
    index_query: Dev<'a>,
    index_weights: Dev<'a>,
    selected: Dev<'a>,
    topk_scratch: Dev<'a>,
    logits: Dev<'a>,
    /// Routes: expert ids (U32) and weights (FP32), [rows, topk].
    route_ids: Dev<'a>,
    route_weights: Dev<'a>,
    wire: Dev<'a>,
    planes: Vec<Dev<'a>>,
    scratch: Dev<'a>,
    dummy: Dev<'a>,
    vocab_logits: Dev<'a>,
    /// Pinned staging: routes + wire rows down, rank partials up.
    router_host: HostAllocation<'a>,
    planes_host: RefCell<HostAllocation<'a>>,
    // Drops before its workspace below.
    head: cuteafd_ffi::dsv4::VocabularyHead<'a>,
    _head_workspace: Dev<'a>,
}

/// Device copies of one step's tables.
struct StepBuffers<'a> {
    /// Rows these buffers hold; tables are copied in per step.
    rows: usize,
    positions: Dev<'a>,
    main_slots: Dev<'a>,
    swa_indices: Dev<'a>,
    swa_lengths: Dev<'a>,
    c4: Vec<Dev<'a>>,
    c128: Vec<Dev<'a>>,
    c4_page_table: Dev<'a>,
    c4_visible: Dev<'a>,
    c4_indexed_lengths: Dev<'a>,
    c128_indices: Dev<'a>,
    c128_lengths: Dev<'a>,
}

impl<'a> Engine<'a> {
    fn alloc(&self, bytes: usize) -> Result<Dev<'a>> {
        DeviceAllocation::new(self.library, bytes.max(256))
    }

    fn zeroed(&self, bytes: usize) -> Result<Dev<'a>> {
        let allocation = self.alloc(bytes)?;
        self.library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
        Ok(allocation)
    }

    fn program(&self, name: &str, pointers: &[&str]) -> Result<Dsv4Program<'a>> {
        self.programs.program(&format!("{}_{name}", self.family), pointers)
    }

    fn run(&self, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Dsv4Scalar]) -> Result<()> {
        let names: Vec<&str> = pointers.iter().map(|(n, _)| *n).collect();
        let program = self.program(name, &names)?;
        let raw: Vec<*mut c_void> = pointers.iter().map(|(_, p)| *p).collect();
        // SAFETY: every pointer names a live allocation sized for the rows in
        // `scalars`; the stream orders all launches of this engine.
        unsafe { program.launch(&raw, scalars, self.stream) }
            .with_context(|| format!("{name} with scalars {scalars:?}"))
    }

    /// Largest scratch of every program in the table (one shared region;
    /// launches are ordered on one stream).
    fn scratch_bytes(&self) -> Result<usize> {
        let mut bytes = 0usize;
        for name in self.programs.names() {
            if name.contains("index_topk") {
                continue;
            }
            for value in self.programs.spec(name)?.scratch.values() {
                bytes = bytes.max(*value as usize);
            }
        }
        Ok(bytes)
    }

    fn pool_layer(parts: &EngineParts<'a>, layer: usize) -> Result<LayerCache<'a>> {
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(parts.library, bytes.max(256))?;
            parts.library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let shape = parts.shape;
        let sequences = shape.sequences;
        let main = zeroed(shape.window_pages() * MAIN_PAGE_BYTES)?;
        let (compressed, index, states) = match parts.cfg.compress_ratios[layer] {
            4 => (
                Some(zeroed(shape.c4_pages * metadata::compressed_page_bytes(4))?),
                Some(zeroed(shape.c4_pages * INDEX_PAGE_BYTES)?),
                vec![
                    zeroed(sequences * 16 * 1024 * 4)?,
                    zeroed(sequences * 16 * 1024 * 4)?,
                    zeroed(sequences * 16 * 256 * 4)?,
                    zeroed(sequences * 16 * 256 * 4)?,
                ],
            ),
            128 => (
                Some(zeroed(shape.c128_pages * metadata::compressed_page_bytes(128))?),
                None,
                vec![zeroed(sequences * 256 * 512 * 4)?, zeroed(sequences * 256 * 512 * 4)?],
            ),
            _ => (None, None, Vec::new()),
        };
        Ok(LayerCache { main, compressed, index, states })
    }

    /// Allocates the cache pools and RoPE tables for `parts.shape`.
    pub fn new(parts: EngineParts<'a>) -> Result<Self> {
        let pools = (0..parts.cfg.n_layers).map(|l| Self::pool_layer(&parts, l)).collect::<Result<Vec<_>>>()?;
        let table = |compressed: bool| -> Result<Dev<'a>> {
            let values = metadata::rope_table(&parts.cfg, compressed, parts.max_context.max(metadata::WINDOW));
            let allocation = DeviceAllocation::new(parts.library, values.len() * 4)?;
            parts.library.copy_h2d(allocation.buffer, bytes_of(&values))?;
            Ok(allocation)
        };
        Ok(Self {
            rope_window: table(false)?,
            rope_compressed: table(true)?,
            pools,
            library: parts.library,
            programs: parts.programs,
            cfg: parts.cfg,
            weights: parts.weights,
            family: parts.family,
            decode_rows: parts.decode_rows,
            prefill_rows: parts.prefill_rows,
            c128_width: parts.c128_width,
            max_context: parts.max_context,
            stream: parts.stream,
            sms: parts.sms,
            shape: parts.shape,
            prefill_workspace: RefCell::new(None),
            decode_workspace: RefCell::new(None),
            profile: RefCell::new(Profile::default()),
            graphs: RefCell::new(std::collections::HashMap::new()),
            local: RefCell::new(None),
        })
    }

    fn workspace(&self, t: usize, lanes: usize) -> Result<Workspace<'a>> {
        let h = self.cfg.dim;
        let heads = self.cfg.n_heads;
        let (experts, topk) = (self.cfg.n_routed_experts, self.cfg.n_activated_experts);
        let head_workspace = self.alloc(cuteafd_ffi::dsv4::VOCABULARY_HEAD_WORKSPACE)?;
        let mut topk_scratch = 0usize;
        for route in [format!("decode_m{}", self.decode_rows), format!("prefill_m{}", self.prefill_rows)] {
            let spec = self.programs.spec(&format!("{}_index_topk_{route}", self.family))?;
            topk_scratch = topk_scratch.max(spec.scratch.get("scratch").copied().unwrap_or(0) as usize);
        }
        let lane = || -> Result<Lane<'a>> {
            Ok(Lane {
                stream_a: self.alloc(t * 4 * h * 2)?,
                stream_b: self.alloc(t * 4 * h * 2)?,
                post: self.alloc(t * 4 * 4)?,
                comb: self.alloc(t * 16 * 4)?,
                tokens: self.alloc(t * 4)?,
                shared: self.alloc(t * h * 2)?,
                tables: self.step_buffers(t)?,
            })
        };
        Ok(Workspace {
            lanes: (0..lanes).map(|_| lane()).collect::<Result<_>>()?,
            y: self.alloc(t * h * 2)?,
            query: self.alloc(t * heads * 512 * 2)?,
            q_rank: self.alloc(t * self.cfg.q_lora_rank * 2)?,
            attn_out: self.alloc(t * heads * 512 * 2)?,
            delta: self.alloc(t * h * 2)?,
            index_query: self.alloc(t * self.cfg.index_n_heads * self.cfg.index_head_dim)?,
            index_weights: self.alloc(t * self.cfg.index_n_heads * 4)?,
            selected: self.alloc(t * self.cfg.index_topk * 4)?,
            topk_scratch: self.zeroed(topk_scratch)?,
            logits: self.alloc(t * experts * 4)?,
            route_ids: self.alloc(t * topk * 4)?,
            route_weights: self.alloc(t * topk * 4)?,
            wire: self.alloc(t * (h + h / 32))?,
            planes: (0..4).map(|_| self.alloc(t * h * 2)).collect::<Result<_>>()?,
            scratch: self.alloc(self.scratch_bytes()?)?,
            dummy: self.zeroed(4096)?,
            vocab_logits: self.alloc(t * self.cfg.vocab_size * 4)?,
            router_host: HostAllocation::new(self.library, t * (topk * 8 + h + h / 32))?,
            planes_host: RefCell::new(HostAllocation::new(self.library, 4 * t * h * 2)?),
            // SAFETY: the workspace buffer lives in the same struct and drops
            // after the head (field order).
            head: unsafe { self.library.vocabulary_head(head_workspace.buffer.ptr, h as u32, t as u32)? },
            _head_workspace: head_workspace,
        })
    }

    /// Persistent table buffers for up to `rows` rows.
    fn step_buffers(&self, rows: usize) -> Result<StepBuffers<'a>> {
        let ints = |count: usize| self.alloc(count * 4);
        let metadata = |_: usize| -> Result<Vec<Dev<'a>>> { (0..9).map(|_| ints(rows + 2)).collect() };
        Ok(StepBuffers {
            rows,
            positions: self.alloc(rows * 8)?,
            main_slots: self.alloc(rows * 8)?,
            swa_indices: ints(rows * metadata::WINDOW)?,
            swa_lengths: ints(rows)?,
            c4: metadata(4)?,
            c128: metadata(128)?,
            c4_page_table: ints(rows.max(1) * self.shape.c4_pages)?,
            c4_visible: ints(rows)?,
            c4_indexed_lengths: ints(rows)?,
            c128_indices: ints(rows * self.c128_width)?,
            c128_lengths: ints(rows)?,
        })
    }

    fn fill(&self, m: &StepBuffers<'_>, tables: &StepTables) -> Result<()> {
        ensure!(tables.rows <= m.rows, "step of {} rows exceeds its buffers ({})", tables.rows, m.rows);
        let put = |buffer: &Dev<'_>, bytes: &[u8]| -> Result<()> {
            ensure!(bytes.len() <= buffer.buffer.bytes, "step table exceeds its buffer");
            if !bytes.is_empty() {
                self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: bytes.len(), ..buffer.buffer }, bytes)?;
            }
            Ok(())
        };
        put(&m.positions, bytes_of(&tables.positions))?;
        put(&m.main_slots, bytes_of(&tables.main_slots))?;
        put(&m.swa_indices, bytes_of(&tables.swa_indices))?;
        put(&m.swa_lengths, bytes_of(&tables.swa_lengths))?;
        for (buffers, entries) in [(&m.c4, &tables.c4_tables), (&m.c128, &tables.c128_tables)] {
            for (buffer, (_, values)) in buffers.iter().zip(entries) {
                put(buffer, bytes_of(values))?;
            }
        }
        put(&m.c4_page_table, bytes_of(&tables.c4_page_table))?;
        put(&m.c4_visible, bytes_of(&tables.c4_visible))?;
        put(&m.c4_indexed_lengths, bytes_of(&tables.c4_indexed_lengths))?;
        put(&m.c128_indices, bytes_of(&tables.c128_indices))?;
        put(&m.c128_lengths, bytes_of(&tables.c128_lengths))
    }

    fn sync(&self) -> Result<()> {
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream) }
    }

    fn download(&self, allocation: &Dev<'_>, bytes: usize) -> Result<Vec<u8>> {
        self.sync()?;
        let mut out = vec![0u8; bytes];
        self.library.copy_d2h(&mut out, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..allocation.buffer })?;
        Ok(out)
    }

    /// Longest chunk one [`Self::prefill`] call takes.
    pub fn prefill_capacity(&self) -> usize {
        PREFILL_LANES * self.prefill_rows
    }

    /// Prefills the next chunk of a sequence (rows continue at its length) and
    /// returns FP32 logits of its last `logit_rows` rows [logit_rows, vocab].
    /// `on_layer` receives each layer's stream.
    ///
    /// A long chunk without `on_layer` runs as [`PREFILL_LANES`] lanes of
    /// consecutive rows, so one lane's attention overlaps the other's experts.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &self,
        placement: &mut Placement,
        tokens: &[u32],
        embed: &[u8],
        transports: &mut [V41Tp4Roce],
        runtime: &tokio::runtime::Runtime,
        logit_rows: usize,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
    ) -> Result<Vec<f32>> {
        let start = placement.len;
        let t = tokens.len();
        let lanes = if on_layer.is_none() && t >= PREFILL_LANES * MIN_LANE_ROWS { PREFILL_LANES } else { 1 };
        let per_lane = t.div_ceil(lanes);
        ensure!(t > 0 && per_lane <= self.prefill_rows && start + t <= self.max_context,
            "prefill chunk of {t} tokens at {start} exceeds {} rows per lane or the {}-token context",
            self.prefill_rows, self.max_context);
        let row = self.cfg.dim * 2;
        let tables = (0..lanes).map(|lane| {
            let (first, end) = (lane * per_lane, ((lane + 1) * per_lane).min(t));
            metadata::prefill_step(placement, &self.shape, start + first, end - first, self.cfg.index_topk, self.c128_width)
        }).collect::<Result<Vec<_>>>()?;
        let steps: Vec<LaneStep<'_>> = tables.iter().enumerate().map(|(lane, tables)| {
            let (first, end) = (lane * per_lane, ((lane + 1) * per_lane).min(t));
            LaneStep { tables, tokens: &tokens[first..end], embed: &embed[first * row..end * row] }
        }).collect();
        let logits = self.step(&steps, transports, runtime, logit_rows, on_layer)?;
        placement.len += t;
        Ok(logits)
    }

    /// One decode row per sequence: appends each token and returns the
    /// logits rows [rows, vocab] in order.
    pub fn decode(
        &self,
        rows: &mut [(&mut Placement, u32)],
        embed: &[u8],
        transport: &mut V41Tp4Roce,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<Vec<f32>> {
        ensure!(!rows.is_empty() && rows.len() <= self.decode_rows, "decode batch of {} rows", rows.len());
        for (placement, _) in rows.iter() {
            ensure!(placement.len > 0 && placement.len < self.max_context, "decode at {} outside the context", placement.len);
        }
        let tokens: Vec<u32> = rows.iter().map(|(_, token)| *token).collect();
        let steps: Vec<(&Placement, usize)> = rows.iter().map(|(p, _)| (&**p, p.len)).collect();
        let tables = metadata::decode_step(&steps, &self.shape, self.cfg.index_topk, self.c128_width)?;
        let step = LaneStep { tables: &tables, tokens: &tokens, embed };
        let logits = self.step(&[step], std::slice::from_mut(transport), runtime, tokens.len(), None)?;
        for (placement, _) in rows.iter_mut() {
            placement.len += 1;
        }
        Ok(logits)
    }

    /// Runs every layer for `lanes` (one decode lane, or prefill lanes of
    /// consecutive rows of one sequence) and returns the logits of the last
    /// `logit_rows` rows across the lanes.
    fn step(
        &self,
        lanes: &[LaneStep<'_>],
        transports: &mut [V41Tp4Roce],
        runtime: &tokio::runtime::Runtime,
        logit_rows: usize,
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
    ) -> Result<Vec<f32>> {
        let h = self.cfg.dim;
        let decode = lanes[0].tables.decode;
        ensure!(!lanes.is_empty(), "a step needs a lane");
        let total: usize = lanes.iter().map(|l| l.tables.rows).sum();
        ensure!(logit_rows <= total && (on_layer.is_none() || lanes.len() == 1)
            && lanes.iter().all(|l| l.tokens.len() == l.tables.rows && l.embed.len() == l.tables.rows * h * 2)
            && (!decode || lanes.len() == 1), "step lanes disagree");
        let slot = if decode { &self.decode_workspace } else { &self.prefill_workspace };
        if slot.borrow().is_none() {
            let (rows, count) = if decode { (self.decode_rows, 1) } else { (self.prefill_rows, PREFILL_LANES) };
            *slot.borrow_mut() = Some(self.workspace(rows, count)?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        ensure!(lanes.len() <= w.lanes.len(), "{} lanes exceed the workspace", lanes.len());
        // Tables are copied in only after the previous step's last read (the
        // head download synchronized the stream).
        for (step, lane) in lanes.iter().zip(&w.lanes) {
            self.fill(&lane.tables, step.tables)?;
            let mut expanded = Vec::with_capacity(step.embed.len() * 4);
            for row in step.embed.chunks_exact(h * 2) {
                for _ in 0..4 {
                    expanded.extend_from_slice(row);
                }
            }
            self.library.copy_h2d(lane.stream_a.buffer, &expanded)?;
            let token_bytes: Vec<u8> = step.tokens.iter().flat_map(|token| token.to_le_bytes()).collect();
            self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: token_bytes.len(), ..lane.tokens.buffer }, &token_bytes)?;
        }
        let cap = if decode { self.decode_rows } else { self.prefill_rows };
        let rows_of = |lane: usize| Dsv4Scalar::I32(lanes[lane].tables.rows as i32);
        if decode {
            let (tables, lane) = (lanes[0].tables, &w.lanes[0]);
            let (t, rows) = (tables.rows, rows_of(0));
            let mut ranks = 0usize;
            for (layer, weights) in self.weights.layers.iter().enumerate() {
                let previous = ranks;
                // One segment: the previous layer's FFN reduce + mHC post, then
                // this layer's attention, router and expert input quantization,
                // replayed as a CUDA graph keyed by everything it bakes in.
                let segment = || -> Result<()> {
                    if previous > 0 {
                        self.post(w, lane, previous, rows, layer - 1)?;
                    }
                    self.attention(layer, weights, tables, lane, w, rows, cap)
                };
                let key = GraphKey {
                    layer,
                    rows: t,
                    attention: self.attention_kind(weights.ratio, tables),
                    table_width: tables.c4_table_width,
                    table_stride: tables.c4_table_stride,
                    previous,
                };
                self.replay(key, segment)?;
                ranks = self.decode_experts(layer, t, w, lane, cap, transports.first_mut().context("no transport")?, runtime)?;
            }
            if ranks > 0 {
                self.post(w, lane, ranks, rows, self.weights.layers.len() - 1)?;
            }
        } else {
            // Units run layer-major. A unit's attention needs its own lane's
            // previous layer posted and the previous lane's same layer (KV and
            // compressor state) done. With a transport per lane, each Spark
            // wave stays in flight while the next unit's attention runs and is
            // received after the next wave is dispatched, so the Sparks always
            // hold the next request when they finish one.
            let units: Vec<(usize, usize)> = (0..self.weights.layers.len())
                .flat_map(|layer| (0..lanes.len()).map(move |lane| (layer, lane))).collect();
            let attention = |(layer, lane): (usize, usize)| {
                self.attention(layer, &self.weights.layers[layer], lanes[lane].tables, &w.lanes[lane], w, rows_of(lane), cap)
            };
            let post = |(layer, lane): (usize, usize), ranks: usize| self.post(w, &w.lanes[lane], ranks, rows_of(lane), layer);
            let local_layers = self.local_layers();
            let pipelined = transports.len() >= lanes.len() && lanes.len() > 1;
            runtime.block_on(async {
                let mut inflight: Option<((usize, usize), V41Tp4RoceWave)> = None;
                attention(units[0])?;
                for (index, &unit) in units.iter().enumerate() {
                    let (layer, lane) = unit;
                    let t = lanes[lane].tables.rows;
                    if layer < local_layers {
                        self.local_experts(layer, t, w, &w.lanes[lane], cap)?;
                        post(unit, LOCAL_EXPERTS)?;
                    } else {
                        let request = self.stage_request(layer, t, w, ExpertV2SourceKind::Prefill)?;
                        let slot = if pipelined { lane } else { 0 };
                        let wave = transports[slot].dispatch_wave(&request)?;
                        // The shared expert runs on the GPU while the Sparks compute.
                        self.shared_ffn(layer, w, &w.lanes[lane], rows_of(lane), cap)?;
                        if let Some((previous, wave)) = inflight.take() {
                            let slot = if pipelined { previous.1 } else { 0 };
                            let ranks = self.land(&mut transports[slot], wave, lanes[previous.1].tables.rows, w).await?;
                            post(previous, ranks)?;
                        }
                        inflight = Some((unit, wave));
                    }
                    let next = units.get(index + 1).copied();
                    // The next unit's input is this unit's post when both are
                    // on one lane, or without a transport per lane.
                    if !pipelined || next.is_none_or(|(_, next_lane)| next_lane == lane) {
                        if let Some((current, wave)) = inflight.take() {
                            let slot = if pipelined { current.1 } else { 0 };
                            let ranks = self.land(&mut transports[slot], wave, lanes[current.1].tables.rows, w).await?;
                            post(current, ranks)?;
                        }
                    }
                    if let Some(on_layer) = on_layer.as_mut() {
                        on_layer(layer, &self.download(&w.lanes[lane].stream_a, t * 4 * h * 2)?)?;
                    }
                    if let Some(next) = next {
                        attention(next)?;
                    }
                }
                anyhow::Ok(())
            })?;
        }
        if logit_rows == 0 {
            // The next step rewrites the tables only after this one drains.
            self.sync()?;
            return Ok(Vec::new());
        }
        let mut logits = Vec::with_capacity(logit_rows * self.cfg.vocab_size);
        let mut first = 0;
        for (step, lane) in lanes.iter().zip(&w.lanes) {
            let rows = step.tables.rows;
            // Rows of this lane inside the last `logit_rows` of the step.
            let wanted = (first + rows).saturating_sub((total - logit_rows).max(first));
            if wanted > 0 {
                logits.extend(self.head(&lane.stream_a, rows, wanted, w)?);
            }
            first += rows;
        }
        Ok(logits)
    }

    /// Which sparse attention a layer runs this step.
    fn attention_kind(&self, ratio: usize, tables: &StepTables) -> &'static str {
        match ratio {
            4 if tables.c4_groups > 0 => "c4",
            128 if tables.c128_groups > 0 => "c128",
            _ => "win",
        }
    }

    /// Launches `segment` through a captured graph for `key`, capturing it the
    /// first time.
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

    /// The layer's routed partials + shared expert, reduced, then mHC post
    /// into stream a.
    fn post(&self, w: &Workspace<'_>, lane: &Lane<'_>, ranks: usize, rows: Dsv4Scalar, layer: usize) -> Result<()> {
        if ranks == LOCAL_EXPERTS {
            let output = self.local.borrow().as_ref().context("local experts")?.output.buffer.ptr;
            return self.run("mhc_post", &[
                ("x", output), ("residual", lane.stream_b.buffer.ptr), ("prev_post", lane.post.buffer.ptr),
                ("prev_comb", lane.comb.buffer.ptr), ("out", lane.stream_a.buffer.ptr),
            ], &[rows]);
        }
        let Dsv4Scalar::I32(count) = rows else { unreachable!() };
        let reducer = self.library.v41_compact_reducer()?;
        let mut pointers = [std::ptr::null::<u16>(); 6];
        for (slot, plane) in pointers.iter_mut().zip(&w.planes[..ranks]) {
            *slot = plane.buffer.ptr.cast();
        }
        // SAFETY: planes, shared and delta are live [rows, h] BF16 buffers on
        // this device, ordered after the plane uploads and shared FFN.
        unsafe {
            reducer.reduce_planes(pointers, ranks as u32, lane.shared.buffer.ptr.cast(),
                w.delta.buffer.ptr.cast(), count as u32, self.stream)
                .with_context(|| format!("layer {layer} expert reduction"))?;
        }
        self.run("mhc_post", &[
            ("x", w.delta.buffer.ptr), ("residual", lane.stream_b.buffer.ptr), ("prev_post", lane.post.buffer.ptr),
            ("prev_comb", lane.comb.buffer.ptr), ("out", lane.stream_a.buffer.ptr),
        ], &[rows])
    }

    /// mHC pre, producer, compressor/indexer, sparse MLA, wo, mHC post_pre,
    /// router scores and expert input quantization (stream a -> stream b, y).
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: usize,
        weights: &LayerWeights<'_>,
        tables: &StepTables,
        lane: &Lane<'_>,
        w: &Workspace<'_>,
        rows: Dsv4Scalar,
        cap: usize,
    ) -> Result<()> {
        let m = &lane.tables;
        let cache = &self.pools[layer];
        let ratio = weights.ratio;
        let rope = if ratio == 0 { &self.rope_window } else { &self.rope_compressed };
        let mode = if tables.decode { "decode" } else { "prefill" };
        let (a, b) = (&lane.stream_a, &lane.stream_b);
        self.run("mhc_pre", &[
            ("residual", a.buffer.ptr), ("fn", weights.ptr("attn.fn")?), ("scale", weights.ptr("attn.scale")?),
            ("base", weights.ptr("attn.base")?), ("norm", weights.ptr("attn.norm")?), ("post", lane.post.buffer.ptr),
            ("comb", lane.comb.buffer.ptr), ("y", w.y.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        self.run(&format!("producer_m{cap}"), &[
            ("hidden", w.y.buffer.ptr), ("positions", m.positions.buffer.ptr), ("main_slots", m.main_slots.buffer.ptr),
            ("cos_sin", rope.buffer.ptr), ("w_qkv", weights.ptr("w_qkv")?), ("w_qkv_scale", weights.ptr("w_qkv_scale")?),
            ("w_q", weights.ptr("w_q")?), ("w_q_scale", weights.ptr("w_q_scale")?), ("q_norm", weights.ptr("q_norm")?),
            ("kv_norm", weights.ptr("kv_norm")?), ("main_kv_cache", cache.main.buffer.ptr), ("query", w.query.buffer.ptr),
            ("q_rank", w.q_rank.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        let (attention, indexed_cache, indexed_indices, indexed_lengths) =
            self.compress(layer, weights, cache, tables, m, w, rope, rows, cap)?;
        debug_assert_eq!(attention, self.attention_kind(ratio, tables));
        self.run(&format!("sparse_mla_{mode}_{attention}_m{cap}"), &[
            ("q", w.query.buffer.ptr), ("swa_cache", cache.main.buffer.ptr), ("swa_indices", m.swa_indices.buffer.ptr),
            ("swa_lengths", m.swa_lengths.buffer.ptr), ("indexed_cache", indexed_cache),
            ("indexed_indices", indexed_indices), ("indexed_lengths", indexed_lengths),
            ("attn_sink", weights.ptr("attn_sink")?), ("out", w.attn_out.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        self.run(&format!("wo_m{cap}"), &[
            ("o", w.attn_out.buffer.ptr), ("positions", m.positions.buffer.ptr), ("cos_sin", rope.buffer.ptr),
            ("wo_a", weights.ptr("wo_a")?), ("wo_a_scale", weights.ptr("wo_a_scale")?), ("wo_b", weights.ptr("wo_b")?),
            ("wo_b_scale", weights.ptr("wo_b_scale")?), ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        self.run(&format!("mhc_post_pre_m{cap}"), &[
            ("x", w.delta.buffer.ptr), ("residual", a.buffer.ptr), ("prev_post", lane.post.buffer.ptr),
            ("prev_comb", lane.comb.buffer.ptr), ("fn", weights.ptr("ffn.fn")?), ("scale", weights.ptr("ffn.scale")?),
            ("base", weights.ptr("ffn.base")?), ("norm", weights.ptr("ffn.norm")?), ("residual_out", b.buffer.ptr),
            ("post", lane.post.buffer.ptr), ("comb", lane.comb.buffer.ptr), ("y", w.y.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        let Dsv4Scalar::I32(t) = rows else { unreachable!() };
        let h = self.cfg.dim;
        self.run("router_scores", &[
            ("x", w.y.buffer.ptr), ("w", weights.ptr("gate")?), ("logits", w.logits.buffer.ptr),
        ], &[rows])?;
        let (bias, tid2eid) = if weights.hash {
            (std::ptr::null_mut(), weights.ptr("gate.tid2eid")?)
        } else {
            (weights.ptr("gate.bias")?, std::ptr::null_mut())
        };
        // SAFETY: logits, routing tables, tokens and route outputs are live
        // device buffers sized for this step's rows.
        unsafe {
            self.library.dsv4_router_select(w.logits.buffer.ptr, bias, tid2eid, lane.tokens.buffer.ptr,
                w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, t as usize, self.cfg.n_routed_experts,
                self.cfg.n_activated_experts, self.cfg.route_scale as f32, self.stream)?;
        }
        let grid = (t as usize * h.div_ceil(256)).div_ceil(8).min(4 * self.sms as usize).max(1);
        self.run("expert_input_quant", &[
            ("source_ptr", w.y.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
            // SAFETY: the scale rows follow the payload inside each wire row.
            ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
            ("scale_mma_ptr", w.dummy.buffer.ptr),
        ], &[rows, Dsv4Scalar::I32(grid as i32)])
    }

    #[allow(clippy::too_many_arguments)]
    fn compress(
        &self,
        layer: usize,
        weights: &LayerWeights<'_>,
        cache: &LayerCache<'_>,
        tables: &StepTables,
        m: &StepBuffers<'_>,
        w: &Workspace<'_>,
        rope: &Dev<'_>,
        rows: Dsv4Scalar,
        cap: usize,
    ) -> Result<(&'static str, *mut c_void, *mut c_void, *mut c_void)> {
        let dummy = w.dummy.buffer.ptr;
        let window = ("win", dummy, dummy, dummy);
        let ratio = weights.ratio;
        if ratio == 0 {
            return Ok(window);
        }
        let (groups, metadata, names) = if ratio == 4 {
            (tables.c4_groups, &m.c4, &tables.c4_tables)
        } else {
            (tables.c128_groups, &m.c128, &tables.c128_tables)
        };
        let compressed = cache.compressed.as_ref().context("compressed cache")?.buffer.ptr;
        let mut pointers: Vec<(&str, *mut c_void)> = vec![("hidden", w.y.buffer.ptr)];
        pointers.extend(names.iter().zip(metadata).map(|((name, _), buffer)| (*name, buffer.buffer.ptr)));
        pointers.extend([
            ("cos_sin", rope.buffer.ptr), ("joint_projection", weights.ptr("joint_projection")?),
            ("main_ape", weights.ptr("main_ape")?), ("main_norm", weights.ptr("main_norm")?),
            ("compressed_cache", compressed), ("main_kv_state", cache.states[0].buffer.ptr),
            ("main_score_state", cache.states[1].buffer.ptr),
        ]);
        if ratio == 4 {
            pointers.extend([
                ("index_ape", weights.ptr("index_ape")?), ("index_norm", weights.ptr("index_norm")?),
                ("index_cache", cache.index.as_ref().context("index cache")?.buffer.ptr),
                ("index_kv_state", cache.states[2].buffer.ptr), ("index_score_state", cache.states[3].buffer.ptr),
            ]);
        }
        pointers.push(("scratch", w.scratch.buffer.ptr));
        if tables.decode {
            self.run(&format!("compressor_decode_c{ratio}"), &pointers, &[rows])
        } else {
            let completed = names[0].1[0].max(1);
            let program = if tables.start == 0 { "prefill" } else { "continuation" };
            self.run(&format!("compressor_{program}_c{ratio}"), &pointers,
                &[rows, Dsv4Scalar::I32(completed), Dsv4Scalar::I32(1)])
        }
        .with_context(|| format!("layer {layer} compressor"))?;
        if groups == 0 {
            return Ok(window);
        }
        if ratio == 128 {
            return Ok(("c128", compressed, m.c128_indices.buffer.ptr, m.c128_lengths.buffer.ptr));
        }
        self.run(&format!("index_producer_m{cap}"), &[
            ("q_rank", w.q_rank.buffer.ptr), ("hidden", w.y.buffer.ptr), ("positions", m.positions.buffer.ptr),
            ("cos_sin", rope.buffer.ptr), ("w_q", weights.ptr("index_w_q")?), ("w_q_scale", weights.ptr("index_w_q_scale")?),
            ("w_proj", weights.ptr("index_w_proj")?), ("query", w.index_query.buffer.ptr),
            ("head_weights", w.index_weights.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        let mode = if tables.decode { "decode" } else { "prefill" };
        self.run(&format!("index_topk_{mode}_m{cap}"), &[
            ("q_fp8", w.index_query.buffer.ptr), ("weights", w.index_weights.buffer.ptr),
            ("index_k_cache", cache.index.as_ref().context("index cache")?.buffer.ptr),
            ("page_table", m.c4_page_table.buffer.ptr), ("cache_lengths", m.c4_visible.buffer.ptr),
            ("output_indices", w.selected.buffer.ptr), ("scratch", w.topk_scratch.buffer.ptr),
        ], &[rows, Dsv4Scalar::I32(tables.c4_table_width as i32), Dsv4Scalar::I32(tables.c4_table_stride as i32)])?;
        Ok(("c4", compressed, w.selected.buffer.ptr, m.c4_indexed_lengths.buffer.ptr))
    }

    /// Router download, shared expert launch, host routing, Spark exchange
    /// and plane uploads; returns the Spark rank count.
    #[allow(clippy::too_many_arguments)]
    /// One Spark request for `rows` wire rows with `topk` routes each.
    fn expert_request(&self, layer: usize, rows: usize, routes: Vec<ExpertProtocolV2RouteEntry>, wire: Vec<u8>,
        kind: ExpertV2SourceKind) -> Result<ExpertProtocolV2Request> {
        let topk = self.cfg.n_activated_experts as u32;
        let mut request = ExpertProtocolV2Request::new(
            layer as u64 + 1, 17, layer as u32, self.cfg.dim as u32, ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..rows as u32).map(|row| ExpertProtocolV2RowDescriptor {
                row_id: u64::from(row), source_kind: kind, source_request_id: 1,
                token_position: u64::from(row), route_offset: row * topk, route_count: topk,
            }).collect(),
            routes, wire,
        )?;
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        Ok(request)
    }

    /// Connects every Spark rank and registers full-size buffers with one
    /// prefill-sized request of zero rows, so the first real request does not
    /// pay for connection setup (about 0.6 s).
    pub fn warm_transport(&self, transport: &mut V41Tp4Roce, runtime: &tokio::runtime::Runtime) -> Result<()> {
        let (rows, h, experts, topk) = (self.prefill_rows, self.cfg.dim, self.cfg.n_routed_experts, self.cfg.n_activated_experts);
        let routes = (0..rows * topk).map(|i| ExpertProtocolV2RouteEntry {
            row_index: (i / topk) as u32, expert_id: (i % experts) as u32, gate_weight: 0.0,
        }).collect();
        let request = self.expert_request(self.cfg.n_layers - 1, rows, routes, vec![0; rows * (h + h / 32)],
            ExpertV2SourceKind::Prefill)?;
        runtime.block_on(async { transport.execute(&request, |_, _, _| Ok(())).await })
    }

    fn local_layers(&self) -> usize {
        self.local.borrow().as_ref().map_or(0, |l| l.layers())
    }

    /// The shared expert on the unit's FFN input `y` into the lane's `shared`.
    fn shared_ffn(&self, layer: usize, w: &Workspace<'_>, lane: &Lane<'_>, rows: Dsv4Scalar, cap: usize) -> Result<()> {
        let weights = &self.weights.layers[layer];
        self.run(&format!("shared_ffn_m{cap}"), &[
            ("x", w.y.buffer.ptr), ("w13", weights.ptr("w13")?), ("w13_scale", weights.ptr("w13_scale")?),
            ("w2", weights.ptr("w2")?), ("w2_scale", weights.ptr("w2_scale")?), ("out", lane.shared.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr),
        ], &[rows])
    }

    /// Shared and routed experts of a coordinator-resident layer: routes,
    /// wire rows and results stay on the device, the host only enqueues.
    fn local_experts(&self, layer: usize, t: usize, w: &Workspace<'_>, lane: &Lane<'_>, cap: usize) -> Result<()> {
        self.shared_ffn(layer, w, lane, Dsv4Scalar::I32(t as i32), cap)?;
        let timer = Instant::now();
        let mut local = self.local.borrow_mut();
        let local = local.as_mut().context("local experts")?;
        // SAFETY: wire, routes and shared rows are complete in stream order.
        unsafe {
            local.run(layer, t, w.wire.buffer.ptr, w.route_ids.buffer.ptr, w.route_weights.buffer.ptr,
                lane.shared.buffer.ptr, self.stream)?
        };
        self.profile.borrow_mut().add(Phase::Experts, timer);
        Ok(())
    }

    /// Waits for the unit's routes and wire rows and builds its Spark request.
    fn stage_request(&self, layer: usize, t: usize, w: &Workspace<'_>, kind: ExpertV2SourceKind)
        -> Result<ExpertProtocolV2Request> {
        let (h, topk) = (self.cfg.dim, self.cfg.n_activated_experts);
        let timer = Instant::now();
        let (route_bytes, wire_bytes) = (t * topk * 4, t * (h + h / 32));
        let host = w.router_host.buffer;
        let at = |offset: usize| cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: ids, weights and wire rows are consecutive inside the pinned buffer.
            ptr: unsafe { host.ptr.cast::<u8>().add(offset) }.cast(),
            bytes: host.bytes - offset,
            ..host
        };
        // SAFETY: the pinned regions are large enough; the sync below completes them.
        unsafe {
            self.library.copy_d2h_host_buffer_async(at(0), w.route_ids.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(route_bytes), w.route_weights.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(2 * route_bytes), w.wire.buffer, wire_bytes, self.stream)?;
        }
        self.sync()?;
        self.profile.borrow_mut().add(Phase::RouterSync, timer);
        let timer = Instant::now();
        let staged = w.router_host.bytes();
        let word = |offset: usize, i: usize| u32::from_le_bytes(staged[offset + i * 4..][..4].try_into().unwrap());
        let routes = (0..t * topk).map(|i| ExpertProtocolV2RouteEntry {
            row_index: (i / topk) as u32,
            expert_id: word(0, i),
            gate_weight: f32::from_bits(word(route_bytes, i)),
        }).collect();
        let wire = staged[2 * route_bytes..2 * route_bytes + wire_bytes].to_vec();
        let request = self.expert_request(layer, t, routes, wire, kind);
        self.profile.borrow_mut().add(Phase::Routing, timer);
        request
    }

    /// Receives a unit's partial rows into the pinned staging and queues one
    /// upload per rank plane. The reduce that reads the planes is ordered
    /// after the uploads, and the next `stage_request` sync completes them
    /// before the staging is rewritten.
    async fn land(&self, transport: &mut V41Tp4Roce, wave: V41Tp4RoceWave, t: usize, w: &Workspace<'_>) -> Result<usize> {
        let timer = Instant::now();
        let row_bytes = self.cfg.dim * 2;
        let ranks = transport.world_size();
        ensure!(ranks <= w.planes.len(), "{ranks} Spark ranks exceed the reduction planes");
        let plane_bytes = t * row_bytes;
        let mut staging = w.planes_host.borrow_mut();
        ensure!(ranks * plane_bytes <= staging.buffer.bytes, "{ranks} rank planes exceed the pinned staging");
        let bytes = staging.bytes_mut();
        transport.receive_wave(wave, |rank, first, payload| {
            let offset = rank * plane_bytes + first as usize * row_bytes;
            ensure!(first as usize * row_bytes + payload.len() <= plane_bytes, "partial rows exceed the step");
            bytes[offset..offset + payload.len()].copy_from_slice(payload);
            Ok(())
        }).await?;
        for rank in 0..ranks {
            let source = cuteafd_ffi::CuteafdHostBuffer {
                // SAFETY: rank planes are disjoint slices of the staging buffer.
                ptr: unsafe { staging.buffer.ptr.cast::<u8>().add(rank * plane_bytes) }.cast(),
                bytes: plane_bytes,
                ..staging.buffer
            };
            // SAFETY: pinned source and device plane both hold `plane_bytes`.
            unsafe { self.library.copy_host_buffer_h2d_async(w.planes[rank].buffer, source, plane_bytes, self.stream)? };
        }
        self.profile.borrow_mut().add(Phase::Experts, timer);
        Ok(ranks)
    }

    /// A decode unit's experts; returns the rank count its post needs
    /// ([`LOCAL_EXPERTS`] for a coordinator-resident layer).
    #[allow(clippy::too_many_arguments)]
    fn decode_experts(
        &self,
        layer: usize,
        t: usize,
        w: &Workspace<'_>,
        lane: &Lane<'_>,
        cap: usize,
        transport: &mut V41Tp4Roce,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<usize> {
        if layer < self.local_layers() {
            self.local_experts(layer, t, w, lane, cap)?;
            return Ok(LOCAL_EXPERTS);
        }
        // Decode rows poll without the prefill spin quantum.
        let request = self.stage_request(layer, t, w, ExpertV2SourceKind::Decode)?;
        let wave = transport.dispatch_wave(&request)?;
        // The shared expert runs on the GPU while the Sparks compute.
        self.shared_ffn(layer, w, lane, Dsv4Scalar::I32(t as i32), cap)?;
        runtime.block_on(self.land(transport, wave, t, w))
    }

    /// Logits of the last `n` of `t` rows.
    fn head(&self, stream: &Dev<'_>, t: usize, n: usize, w: &Workspace<'_>) -> Result<Vec<f32>> {
        let h = self.cfg.dim;
        let vocab = self.cfg.vocab_size;
        self.run("mhc_head", &[
            ("residual", stream.buffer.ptr), ("fn", self.weights.head_fn.buffer.ptr),
            ("scale", self.weights.head_scale.buffer.ptr), ("base", self.weights.head_base.buffer.ptr),
            ("norm", self.weights.norm.buffer.ptr), ("collapsed", w.delta.buffer.ptr), ("out", w.y.buffer.ptr),
        ], &[Dsv4Scalar::I32(t as i32)])?;
        // SAFETY: input, weights and logits are live buffers of the head's
        // shape; the last `n` rows start `t - n` rows into `y`.
        unsafe {
            w.head.launch(w.y.buffer.ptr.cast::<u8>().add((t - n) * h * 2).cast(), self.weights.head.buffer.ptr.cast(),
                w.vocab_logits.buffer.ptr.cast(), n as u32, self.stream)?;
        }
        let timer = Instant::now();
        let logits = self.download(&w.vocab_logits, n * vocab * 4)?;
        self.profile.borrow_mut().add(Phase::Head, timer);
        Ok(logits
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }
}
