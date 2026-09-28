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
use cuteafd_transport::v41_expert::{V41Tp4Roce, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
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

/// Prefill workspace sized for `tokens` rows.
struct Workspace<'a> {
    stream_a: Dev<'a>,
    stream_b: Dev<'a>,
    post: Dev<'a>,
    comb: Dev<'a>,
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
    wire: Dev<'a>,
    shared: Dev<'a>,
    planes: Vec<Dev<'a>>,
    scratch: Dev<'a>,
    dummy: Dev<'a>,
    vocab_logits: Dev<'a>,
    tables: StepBuffers<'a>,
    /// Pinned staging: router logits + wire rows down, rank partials up.
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

    fn workspace(&self, t: usize) -> Result<Workspace<'a>> {
        let h = self.cfg.dim;
        let heads = self.cfg.n_heads;
        let experts = self.cfg.n_routed_experts;
        let head_workspace = self.alloc(cuteafd_ffi::dsv4::VOCABULARY_HEAD_WORKSPACE)?;
        let mut topk_scratch = 0usize;
        for route in [format!("decode_m{}", self.decode_rows), format!("prefill_m{}", self.prefill_rows)] {
            let spec = self.programs.spec(&format!("{}_index_topk_{route}", self.family))?;
            topk_scratch = topk_scratch.max(spec.scratch.get("scratch").copied().unwrap_or(0) as usize);
        }
        Ok(Workspace {
            stream_a: self.alloc(t * 4 * h * 2)?,
            stream_b: self.alloc(t * 4 * h * 2)?,
            post: self.alloc(t * 4 * 4)?,
            comb: self.alloc(t * 16 * 4)?,
            y: self.alloc(t * h * 2)?,
            query: self.alloc(t * heads * 512 * 2)?,
            q_rank: self.alloc(t * self.cfg.q_lora_rank * 2)?,
            attn_out: self.alloc(t * heads * 512 * 2)?,
            delta: self.alloc(t * h * 2)?,
            index_query: self.alloc(t * 64 * 128)?,
            index_weights: self.alloc(t * 64 * 4)?,
            selected: self.alloc(t * self.cfg.index_topk * 4)?,
            topk_scratch: self.zeroed(topk_scratch)?,
            logits: self.alloc(t * experts * 4)?,
            wire: self.alloc(t * (h + h / 32))?,
            shared: self.alloc(t * h * 2)?,
            planes: (0..4).map(|_| self.alloc(t * h * 2)).collect::<Result<_>>()?,
            scratch: self.alloc(self.scratch_bytes()?)?,
            dummy: self.zeroed(4096)?,
            vocab_logits: self.alloc(t * self.cfg.vocab_size * 4)?,
            tables: self.step_buffers(t)?,
            router_host: HostAllocation::new(self.library, t * (experts * 4 + h + h / 32))?,
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

    /// Prefills the next chunk of a sequence (rows continue at its length) and
    /// returns FP32 logits [T, vocab]. `on_layer` receives each layer's stream.
    pub fn prefill(
        &self,
        placement: &mut Placement,
        tokens: &[u32],
        embed: &[u8],
        transport: &mut V41Tp4Roce,
        runtime: &tokio::runtime::Runtime,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
    ) -> Result<Vec<f32>> {
        let start = placement.len;
        ensure!(!tokens.is_empty() && tokens.len() <= self.prefill_rows && start + tokens.len() <= self.max_context,
            "prefill chunk of {} tokens at {start} exceeds {} rows or the {}-token context",
            tokens.len(), self.prefill_rows, self.max_context);
        let tables = metadata::prefill_step(placement, &self.shape, start, tokens.len(), self.cfg.index_topk, self.c128_width)?;
        let logits = self.step(&tables, tokens, embed, transport, runtime, on_layer)?;
        placement.len += tokens.len();
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
        let logits = self.step(&tables, &tokens, embed, transport, runtime, None)?;
        for (placement, _) in rows.iter_mut() {
            placement.len += 1;
        }
        Ok(logits)
    }

    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        tables: &StepTables,
        tokens: &[u32],
        embed: &[u8],
        transport: &mut V41Tp4Roce,
        runtime: &tokio::runtime::Runtime,
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
    ) -> Result<Vec<f32>> {
        let t = tables.rows;
        let h = self.cfg.dim;
        ensure!(tokens.len() == t && embed.len() == t * h * 2, "step rows disagree");
        let slot = if tables.decode { &self.decode_workspace } else { &self.prefill_workspace };
        if slot.borrow().is_none() {
            let rows = if tables.decode { self.decode_rows } else { self.prefill_rows };
            *slot.borrow_mut() = Some(self.workspace(rows)?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        // Tables are copied in only after the previous step's last read (the
        // head download synchronized the stream).
        self.fill(&w.tables, tables)?;
        let m = &w.tables;
        let mut expanded = Vec::with_capacity(t * 4 * h * 2);
        for row in embed.chunks_exact(h * 2) {
            for _ in 0..4 {
                expanded.extend_from_slice(row);
            }
        }
        self.library.copy_h2d(w.stream_a.buffer, &expanded)?;
        let rows = Dsv4Scalar::I32(t as i32);
        let cap = if tables.decode { self.decode_rows } else { self.prefill_rows };
        let mut ranks = 0usize;
        for (layer, weights) in self.weights.layers.iter().enumerate() {
            let previous = ranks;
            // One segment: the previous layer's FFN reduce + mHC post, then this
            // layer's attention, router and expert input quantization. Decode
            // replays it as a CUDA graph keyed by everything it bakes in.
            let segment = || -> Result<()> {
                if previous > 0 {
                    self.post(&w, previous, rows, layer - 1)?;
                }
                self.attention(layer, weights, tables, m, &w, rows, cap)
            };
            if tables.decode {
                let key = GraphKey {
                    layer,
                    rows: t,
                    attention: self.attention_kind(weights.ratio, tables),
                    table_width: tables.c4_table_width,
                    table_stride: tables.c4_table_stride,
                    previous,
                };
                self.replay(key, segment)?;
            } else {
                segment()?;
            }
            ranks = self.exchange(layer, weights, tokens, &w, cap, transport, runtime)?;
            if !tables.decode {
                // The stream a layer hands on is complete only after its post.
                self.post(&w, ranks, rows, layer)?;
                if let Some(on_layer) = on_layer.as_mut() {
                    on_layer(layer, &self.download(&w.stream_a, t * 4 * h * 2)?)?;
                }
                ranks = 0;
            }
        }
        if ranks > 0 {
            self.post(&w, ranks, rows, self.weights.layers.len() - 1)?;
        }
        self.head(&w.stream_a, t, &w)
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
    fn post(&self, w: &Workspace<'_>, ranks: usize, rows: Dsv4Scalar, layer: usize) -> Result<()> {
        if ranks == LOCAL_EXPERTS {
            let output = self.local.borrow().as_ref().context("local experts")?.output.buffer.ptr;
            return self.run("mhc_post", &[
                ("x", output), ("residual", w.stream_b.buffer.ptr), ("prev_post", w.post.buffer.ptr),
                ("prev_comb", w.comb.buffer.ptr), ("out", w.stream_a.buffer.ptr),
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
            reducer.reduce_planes(pointers, ranks as u32, w.shared.buffer.ptr.cast(),
                w.delta.buffer.ptr.cast(), count as u32, self.stream)
                .with_context(|| format!("layer {layer} expert reduction"))?;
        }
        self.run("mhc_post", &[
            ("x", w.delta.buffer.ptr), ("residual", w.stream_b.buffer.ptr), ("prev_post", w.post.buffer.ptr),
            ("prev_comb", w.comb.buffer.ptr), ("out", w.stream_a.buffer.ptr),
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
        m: &StepBuffers<'_>,
        w: &Workspace<'_>,
        rows: Dsv4Scalar,
        cap: usize,
    ) -> Result<()> {
        let cache = &self.pools[layer];
        let ratio = weights.ratio;
        let rope = if ratio == 0 { &self.rope_window } else { &self.rope_compressed };
        let mode = if tables.decode { "decode" } else { "prefill" };
        let (a, b) = (&w.stream_a, &w.stream_b);
        self.run("mhc_pre", &[
            ("residual", a.buffer.ptr), ("fn", weights.ptr("attn.fn")?), ("scale", weights.ptr("attn.scale")?),
            ("base", weights.ptr("attn.base")?), ("norm", weights.ptr("attn.norm")?), ("post", w.post.buffer.ptr),
            ("comb", w.comb.buffer.ptr), ("y", w.y.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
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
            ("x", w.delta.buffer.ptr), ("residual", a.buffer.ptr), ("prev_post", w.post.buffer.ptr),
            ("prev_comb", w.comb.buffer.ptr), ("fn", weights.ptr("ffn.fn")?), ("scale", weights.ptr("ffn.scale")?),
            ("base", weights.ptr("ffn.base")?), ("norm", weights.ptr("ffn.norm")?), ("residual_out", b.buffer.ptr),
            ("post", w.post.buffer.ptr), ("comb", w.comb.buffer.ptr), ("y", w.y.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        let Dsv4Scalar::I32(t) = rows else { unreachable!() };
        let h = self.cfg.dim;
        self.run("router_scores", &[
            ("x", w.y.buffer.ptr), ("w", weights.ptr("gate")?), ("logits", w.logits.buffer.ptr),
        ], &[rows])?;
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
    fn exchange(
        &self,
        layer: usize,
        weights: &LayerWeights<'_>,
        tokens: &[u32],
        w: &Workspace<'_>,
        cap: usize,
        transport: &mut V41Tp4Roce,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<usize> {
        let t = tokens.len();
        let (h, experts, topk) = (self.cfg.dim, self.cfg.n_routed_experts, self.cfg.n_activated_experts);
        let rows = Dsv4Scalar::I32(t as i32);
        let timer = Instant::now();
        let (logit_bytes, wire_bytes) = (t * experts * 4, t * (h + h / 32));
        let host = w.router_host.buffer;
        let wire_host = cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: the wire rows follow the logits inside the pinned buffer.
            ptr: unsafe { host.ptr.cast::<u8>().add(logit_bytes) }.cast(),
            bytes: host.bytes - logit_bytes,
            ..host
        };
        // SAFETY: both pinned regions are large enough; the sync below completes them.
        unsafe {
            self.library.copy_d2h_host_buffer_async(host, w.logits.buffer, logit_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(wire_host, w.wire.buffer, wire_bytes, self.stream)?;
        }
        self.sync()?;
        let staged = w.router_host.bytes();
        let logits: Vec<f32> = staged[..logit_bytes]
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let wire = staged[logit_bytes..logit_bytes + wire_bytes].to_vec();
        self.profile.borrow_mut().add(Phase::RouterSync, timer);
        // The shared expert runs on the GPU while the Sparks compute.
        self.run(&format!("shared_ffn_m{cap}"), &[
            ("x", w.y.buffer.ptr), ("w13", weights.ptr("w13")?), ("w13_scale", weights.ptr("w13_scale")?),
            ("w2", weights.ptr("w2")?), ("w2_scale", weights.ptr("w2_scale")?), ("out", w.shared.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        let timer = Instant::now();
        // Routing on the host: sqrtsoftplus scores; hash layers take ids from
        // tid2eid[token], score layers the top-k of score + bias; weights are
        // the unbiased scores at those ids, sum-normalized, times route_scale.
        let mut routes = Vec::with_capacity(t * topk);
        for (row, token) in tokens.iter().enumerate() {
            let scores: Vec<f32> = logits[row * experts..][..experts].iter()
                .map(|&x| (if x > 20.0 { x } else { x.exp().ln_1p() }).sqrt()).collect();
            let ids: Vec<usize> = if weights.hash {
                weights.tid2eid[*token as usize * topk..][..topk].iter().map(|&e| e as usize).collect()
            } else {
                let mut order: Vec<usize> = (0..experts).collect();
                order.sort_by(|&a, &b| (scores[b] + weights.gate_bias[b]).total_cmp(&(scores[a] + weights.gate_bias[a])).then(a.cmp(&b)));
                order.truncate(topk);
                order
            };
            let total: f32 = ids.iter().map(|&e| scores[e]).sum();
            for &e in &ids {
                routes.push(ExpertProtocolV2RouteEntry {
                    row_index: row as u32,
                    expert_id: e as u32,
                    gate_weight: scores[e] / total * self.cfg.route_scale as f32,
                });
            }
        }
        let local_layers = self.local.borrow().as_ref().map_or(0, |l| l.layers());
        if layer < local_layers {
            let ids: Vec<u32> = routes.iter().map(|r| r.expert_id).collect();
            let weights: Vec<f32> = routes.iter().map(|r| r.gate_weight).collect();
            self.profile.borrow_mut().add(Phase::Routing, timer);
            let timer = Instant::now();
            let mut local = self.local.borrow_mut();
            let local = local.as_mut().context("local experts")?;
            // SAFETY: wire and shared rows are complete in stream order.
            unsafe { local.run(layer, t, w.wire.buffer.ptr, &ids, &weights, w.shared.buffer.ptr, self.stream)? };
            self.profile.borrow_mut().add(Phase::Experts, timer);
            return Ok(LOCAL_EXPERTS);
        }
        let mut request = ExpertProtocolV2Request::new(
            layer as u64 + 1, 17, layer as u32, h as u32, ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..t as u32).map(|row| ExpertProtocolV2RowDescriptor {
                row_id: u64::from(row), source_kind: ExpertV2SourceKind::Prefill, source_request_id: 1,
                token_position: u64::from(row), route_offset: row * topk as u32, route_count: topk as u32,
            }).collect(),
            routes, wire,
        )?;
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        let row_bytes = h * 2;
        let ranks = transport.world_size();
        ensure!(ranks <= w.planes.len(), "{ranks} Spark ranks exceed the reduction planes");
        self.profile.borrow_mut().add(Phase::Routing, timer);
        let timer = Instant::now();
        // Each rank's partial rows land in its device plane straight from the
        // pinned receive slot.
        let plane_bytes = t * row_bytes;
        let mut staging = w.planes_host.borrow_mut();
        ensure!(ranks * plane_bytes <= staging.buffer.bytes, "{ranks} rank planes exceed the pinned staging");
        let bytes = staging.bytes_mut();
        runtime.block_on(async {
            let pending = transport.dispatch(&request).await?;
            pending.receive(|rank, first, payload| {
                let offset = rank * plane_bytes + first as usize * row_bytes;
                ensure!(first as usize * row_bytes + payload.len() <= plane_bytes, "partial rows exceed the step");
                bytes[offset..offset + payload.len()].copy_from_slice(payload);
                Ok(())
            }).await
        })?;
        // One async upload per plane; the next segment's reduce is ordered
        // after them, and the next router sync completes them before the
        // staging is rewritten.
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

    fn head(&self, stream: &Dev<'_>, t: usize, w: &Workspace<'_>) -> Result<Vec<f32>> {
        let h = self.cfg.dim;
        let vocab = self.cfg.vocab_size;
        self.run("mhc_head", &[
            ("residual", stream.buffer.ptr), ("fn", self.weights.head_fn.buffer.ptr),
            ("scale", self.weights.head_scale.buffer.ptr), ("base", self.weights.head_base.buffer.ptr),
            ("norm", self.weights.norm.buffer.ptr), ("collapsed", w.delta.buffer.ptr), ("out", w.y.buffer.ptr),
        ], &[Dsv4Scalar::I32(t as i32)])?;
        let _ = h;
        // SAFETY: input, weights and logits are live buffers of the head's shape.
        unsafe {
            w.head.launch(w.y.buffer.ptr.cast(), self.weights.head.buffer.ptr.cast(),
                w.vocab_logits.buffer.ptr.cast(), t as u32, self.stream)?;
        }
        let timer = Instant::now();
        let logits = self.download(&w.vocab_logits, t * vocab * 4)?;
        self.profile.borrow_mut().add(Phase::Head, timer);
        Ok(logits
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }
}
