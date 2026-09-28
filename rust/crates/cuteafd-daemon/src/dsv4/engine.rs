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
use crate::v41_memory::DeviceAllocation;
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
}

/// Device copies of one step's tables.
struct StepBuffers<'a> {
    positions: Dev<'a>,
    main_slots: Dev<'a>,
    swa_indices: Dev<'a>,
    swa_lengths: Dev<'a>,
    c4: Vec<(&'static str, Dev<'a>)>,
    c128: Vec<(&'static str, Dev<'a>)>,
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

    fn upload<T: Copy>(&self, values: &[T]) -> Result<Dev<'a>> {
        let allocation = self.alloc(std::mem::size_of_val(values))?;
        if !values.is_empty() {
            self.library.copy_h2d(allocation.buffer, bytes_of(values))?;
        }
        Ok(allocation)
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
        })
    }

    fn workspace(&self, t: usize) -> Result<Workspace<'a>> {
        let h = self.cfg.dim;
        let heads = self.cfg.n_heads;
        let experts = self.cfg.n_routed_experts;
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
        })
    }

    fn step_buffers(&self, tables: &StepTables) -> Result<StepBuffers<'a>> {
        let list = |entries: &[(&'static str, Vec<i32>)]| -> Result<Vec<(&'static str, Dev<'a>)>> {
            entries.iter().map(|(name, values)| Ok((*name, self.upload(values)?))).collect()
        };
        Ok(StepBuffers {
            positions: self.upload(&tables.positions)?,
            main_slots: self.upload(&tables.main_slots)?,
            swa_indices: self.upload(&tables.swa_indices)?,
            swa_lengths: self.upload(&tables.swa_lengths)?,
            c4: list(&tables.c4_tables)?,
            c128: list(&tables.c128_tables)?,
            c4_page_table: self.upload(&tables.c4_page_table)?,
            c4_visible: self.upload(&tables.c4_visible)?,
            c4_indexed_lengths: self.upload(&tables.c4_indexed_lengths)?,
            c128_indices: self.upload(&tables.c128_indices)?,
            c128_lengths: self.upload(&tables.c128_lengths)?,
        })
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

    /// Prefills `tokens` for a freshly admitted sequence and returns FP32
    /// logits [T, vocab]. `on_layer` receives each layer's output stream.
    pub fn prefill(
        &self,
        placement: &mut Placement,
        tokens: &[u32],
        embed: &[u8],
        transport: &mut V41Tp4Roce,
        runtime: &tokio::runtime::Runtime,
        on_layer: impl FnMut(usize, &[u8]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        ensure!(placement.len == 0, "prefill continues only from an empty sequence");
        ensure!(!tokens.is_empty() && tokens.len() <= self.prefill_rows && tokens.len() <= self.max_context,
            "prefill of {} tokens is outside 1..={}", tokens.len(), self.prefill_rows.min(self.max_context));
        let tables = metadata::prefill_step(placement, &self.shape, tokens.len(), self.cfg.index_topk, self.c128_width)?;
        let logits = self.step(&tables, tokens, embed, transport, runtime, on_layer)?;
        placement.len = tokens.len();
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
        let logits = self.step(&tables, &tokens, embed, transport, runtime, |_, _| Ok(()))?;
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
        mut on_layer: impl FnMut(usize, &[u8]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        let t = tables.rows;
        let h = self.cfg.dim;
        ensure!(tokens.len() == t && embed.len() == t * h * 2, "step rows disagree");
        let m = self.step_buffers(tables)?;
        let slot = if tables.decode { &self.decode_workspace } else { &self.prefill_workspace };
        if slot.borrow().is_none() {
            let rows = if tables.decode { self.decode_rows } else { self.prefill_rows };
            *slot.borrow_mut() = Some(self.workspace(rows)?);
        }
        let workspace = slot.borrow();
        let w = workspace.as_ref().context("workspace")?;
        let mut expanded = Vec::with_capacity(t * 4 * h * 2);
        for row in embed.chunks_exact(h * 2) {
            for _ in 0..4 {
                expanded.extend_from_slice(row);
            }
        }
        self.library.copy_h2d(w.stream_a.buffer, &expanded)?;
        let rows = Dsv4Scalar::I32(t as i32);
        let cap = if tables.decode { self.decode_rows } else { self.prefill_rows };
        let mode = if tables.decode { "decode" } else { "prefill" };
        let (mut current, mut next) = (&w.stream_a, &w.stream_b);
        for (layer, weights) in self.weights.layers.iter().enumerate() {
            let cache = &self.pools[layer];
            let ratio = weights.ratio;
            let rope = if ratio == 0 { &self.rope_window } else { &self.rope_compressed };
            // 1. mHC pre (attention) with attn_norm.
            self.run("mhc_pre", &[
                ("residual", current.buffer.ptr), ("fn", weights.ptr("attn.fn")?), ("scale", weights.ptr("attn.scale")?),
                ("base", weights.ptr("attn.base")?), ("norm", weights.ptr("attn.norm")?), ("post", w.post.buffer.ptr),
                ("comb", w.comb.buffer.ptr), ("y", w.y.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
            ], &[rows])?;
            // 2. producer: q/kv projections, window cache pack, query.
            self.run(&format!("producer_m{cap}"), &[
                ("hidden", w.y.buffer.ptr), ("positions", m.positions.buffer.ptr), ("main_slots", m.main_slots.buffer.ptr),
                ("cos_sin", rope.buffer.ptr), ("w_qkv", weights.ptr("w_qkv")?), ("w_qkv_scale", weights.ptr("w_qkv_scale")?),
                ("w_q", weights.ptr("w_q")?), ("w_q_scale", weights.ptr("w_q_scale")?), ("q_norm", weights.ptr("q_norm")?),
                ("kv_norm", weights.ptr("kv_norm")?), ("main_kv_cache", cache.main.buffer.ptr), ("query", w.query.buffer.ptr),
                ("q_rank", w.q_rank.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
            ], &[rows])?;
            // 3. compressor, index query and top-k.
            let (attention, indexed_cache, indexed_indices, indexed_lengths) =
                self.compress(layer, weights, cache, tables, &m, &w, rope, rows, cap)?;
            // 4. sparse MLA over window + indexed slots with sink.
            self.run(&format!("sparse_mla_{mode}_{attention}_m{cap}"), &[
                ("q", w.query.buffer.ptr), ("swa_cache", cache.main.buffer.ptr), ("swa_indices", m.swa_indices.buffer.ptr),
                ("swa_lengths", m.swa_lengths.buffer.ptr), ("indexed_cache", indexed_cache),
                ("indexed_indices", indexed_indices), ("indexed_lengths", indexed_lengths),
                ("attn_sink", weights.ptr("attn_sink")?), ("out", w.attn_out.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
            ], &[rows])?;
            // 5. wo with inverse RoPE.
            self.run(&format!("wo_m{cap}"), &[
                ("o", w.attn_out.buffer.ptr), ("positions", m.positions.buffer.ptr), ("cos_sin", rope.buffer.ptr),
                ("wo_a", weights.ptr("wo_a")?), ("wo_a_scale", weights.ptr("wo_a_scale")?), ("wo_b", weights.ptr("wo_b")?),
                ("wo_b_scale", weights.ptr("wo_b_scale")?), ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
            ], &[rows])?;
            // 6. mHC post (attention) fused with pre (FFN) + ffn_norm.
            self.run(&format!("mhc_post_pre_m{cap}"), &[
                ("x", w.delta.buffer.ptr), ("residual", current.buffer.ptr), ("prev_post", w.post.buffer.ptr),
                ("prev_comb", w.comb.buffer.ptr), ("fn", weights.ptr("ffn.fn")?), ("scale", weights.ptr("ffn.scale")?),
                ("base", weights.ptr("ffn.base")?), ("norm", weights.ptr("ffn.norm")?), ("residual_out", next.buffer.ptr),
                ("post", w.post.buffer.ptr), ("comb", w.comb.buffer.ptr), ("y", w.y.buffer.ptr), ("scratch", w.scratch.buffer.ptr),
            ], &[rows])?;
            std::mem::swap(&mut current, &mut next);
            // 7. MoE: routed experts on the Sparks plus the shared expert.
            self.ffn(layer, weights, tokens, &w, cap, transport, runtime)?;
            // 8. mHC post (FFN) into the next layer's stream.
            self.run("mhc_post", &[
                ("x", w.delta.buffer.ptr), ("residual", current.buffer.ptr), ("prev_post", w.post.buffer.ptr),
                ("prev_comb", w.comb.buffer.ptr), ("out", next.buffer.ptr),
            ], &[rows])?;
            std::mem::swap(&mut current, &mut next);
            if !tables.decode {
                on_layer(layer, &self.download(current, t * 4 * h * 2)?)?;
            }
        }
        self.head(current, t, &w)
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
        let (groups, metadata) = if ratio == 4 { (tables.c4_groups, &m.c4) } else { (tables.c128_groups, &m.c128) };
        let compressed = cache.compressed.as_ref().context("compressed cache")?.buffer.ptr;
        let mut pointers: Vec<(&str, *mut c_void)> = vec![("hidden", w.y.buffer.ptr)];
        pointers.extend(metadata.iter().map(|(name, buffer)| (*name, buffer.buffer.ptr)));
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
            let grid = Dsv4Scalar::I32(groups.max(1) as i32);
            self.run(&format!("compressor_prefill_c{ratio}"), &pointers, &[rows, grid, Dsv4Scalar::I32(1)])
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

    fn ffn(
        &self,
        layer: usize,
        weights: &LayerWeights<'_>,
        tokens: &[u32],
        w: &Workspace<'_>,
        cap: usize,
        transport: &mut V41Tp4Roce,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<()> {
        let t = tokens.len();
        let (h, experts, topk) = (self.cfg.dim, self.cfg.n_routed_experts, self.cfg.n_activated_experts);
        let rows = Dsv4Scalar::I32(t as i32);
        self.run("router_scores", &[
            ("x", w.y.buffer.ptr), ("w", weights.ptr("gate")?), ("logits", w.logits.buffer.ptr),
        ], &[rows])?;
        let grid = (t * h.div_ceil(256)).div_ceil(8).min(4 * self.sms as usize).max(1);
        self.run("expert_input_quant", &[
            ("source_ptr", w.y.buffer.ptr), ("values_ptr", w.wire.buffer.ptr),
            // SAFETY: the scale rows follow the payload inside each wire row.
            ("scale_rows_ptr", unsafe { w.wire.buffer.ptr.cast::<u8>().add(h) }.cast()),
            ("scale_mma_ptr", w.dummy.buffer.ptr),
        ], &[rows, Dsv4Scalar::I32(grid as i32)])?;
        self.run(&format!("shared_ffn_m{cap}"), &[
            ("x", w.y.buffer.ptr), ("w13", weights.ptr("w13")?), ("w13_scale", weights.ptr("w13_scale")?),
            ("w2", weights.ptr("w2")?), ("w2_scale", weights.ptr("w2_scale")?), ("out", w.shared.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr),
        ], &[rows])?;
        // Routing on the host: sqrtsoftplus scores; hash layers take ids from
        // tid2eid[token], score layers the top-k of score + bias; weights are
        // the unbiased scores at those ids, sum-normalized, times route_scale.
        let logits: Vec<f32> = self.download(&w.logits, t * experts * 4)?
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
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
        let wire = self.download(&w.wire, t * (h + h / 32))?;
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
        let mut planes = vec![vec![0u8; t * row_bytes]; transport.world_size()];
        runtime.block_on(transport.execute(&request, |rank, first, payload| {
            let start = first as usize * row_bytes;
            planes[rank][start..start + payload.len()].copy_from_slice(payload);
            Ok(())
        }))?;
        for (plane, bytes) in w.planes.iter().zip(&planes) {
            self.library.copy_h2d(plane.buffer, bytes)?;
        }
        let reducer = self.library.v41_compact_reducer()?;
        let mut pointers = [std::ptr::null::<u16>(); 6];
        for (slot, plane) in pointers.iter_mut().zip(&w.planes) {
            *slot = plane.buffer.ptr.cast();
        }
        // SAFETY: planes, shared and delta are live [t, h] BF16 buffers on this
        // device; the stream orders the shared FFN before the reduction.
        unsafe {
            reducer.reduce_planes(pointers, planes.len() as u32, w.shared.buffer.ptr.cast(),
                w.delta.buffer.ptr.cast(), t as u32, self.stream)?;
        }
        Ok(())
    }

    fn head(&self, stream: &Dev<'_>, t: usize, w: &Workspace<'_>) -> Result<Vec<f32>> {
        let h = self.cfg.dim;
        let vocab = self.cfg.vocab_size;
        self.run("mhc_head", &[
            ("residual", stream.buffer.ptr), ("fn", self.weights.head_fn.buffer.ptr),
            ("scale", self.weights.head_scale.buffer.ptr), ("base", self.weights.head_base.buffer.ptr),
            ("norm", self.weights.norm.buffer.ptr), ("collapsed", w.delta.buffer.ptr), ("out", w.y.buffer.ptr),
        ], &[Dsv4Scalar::I32(t as i32)])?;
        let workspace = self.alloc(cuteafd_ffi::dsv4::VOCABULARY_HEAD_WORKSPACE)?;
        let logits = self.alloc(t * vocab * 4)?;
        // SAFETY: workspace outlives the head, which drops at the end of scope.
        let head = unsafe { self.library.vocabulary_head(workspace.buffer.ptr, h as u32, t as u32)? };
        unsafe {
            head.launch(w.y.buffer.ptr.cast(), self.weights.head.buffer.ptr.cast(),
                logits.buffer.ptr.cast(), t as u32, self.stream)?;
        }
        Ok(self.download(&logits, t * vocab * 4)?
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }
}
