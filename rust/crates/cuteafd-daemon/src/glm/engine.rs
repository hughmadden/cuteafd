//! GLM 5.x (glm_moe_dsa) coordinator over the exported glm_* programs.
//!
//! One layer: input norm (fused with the previous layer's residual add),
//! MLA producer (writes the FP8 656-byte latent record), DSA index producer
//! and top-k on full-indexer layers (shared layers reuse the last full
//! layer's selection), sparse MLA, W_UV + o_proj, post-attention norm, then
//! the dense MLP or the MoE (router, expert input quantization, shared
//! expert, routed experts). The latent and index caches share page ids.
use super::weights::{GlmLayer, GlmWeights};
use crate::v41_memory::{DeviceAllocation, HostAllocation};
use cuteafd_transport::v41_expert::{V41Tp4Roce, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::dsv4::{Dsv4Programs, Dsv4Scalar, VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::glm_dsa::GlmDsaConfig;
use std::cell::RefCell;
use std::ffi::c_void;

type Dev<'a> = DeviceAllocation<'a>;

pub(crate) const PAGE_ROWS: usize = 64;
const RECORD_PAGE_BYTES: usize = PAGE_ROWS * 656;
const INDEX_PAGE_BYTES: usize = 8448;
/// Most Spark ranks a step's partials come from (the compact reducer's limit).
const MAX_RANKS: usize = 6;
/// Rows of the decode-route programs (`_m64`).
pub(crate) const DECODE_ROWS: usize = 64;

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
}

/// A sequence's pages (shared by the latent and index caches) and length.
#[derive(Debug, Clone)]
pub(crate) struct GlmPlacement {
    pub pages: Vec<i32>,
    pub len: usize,
}

impl GlmPlacement {
    pub fn slot(&self, position: usize) -> Result<i64> {
        let page = *self.pages.get(position / PAGE_ROWS).context("position past the sequence's pages")?;
        Ok(i64::from(page) * PAGE_ROWS as i64 + (position % PAGE_ROWS) as i64)
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
    planes: Vec<Dev<'a>>,
    router_host: RefCell<HostAllocation<'a>>,
    planes_host: RefCell<HostAllocation<'a>>,
    head: VocabularyHead<'a>,
    _head_workspace: Dev<'a>,
}

pub(crate) struct GlmEngine<'a> {
    pub library: &'a NativeLibrary,
    pub programs: &'a Dsv4Programs<'a>,
    pub cfg: GlmDsaConfig,
    pub weights: GlmWeights<'a>,
    pub stream: *mut c_void,
    pub max_context: usize,
    pub prefill_rows: usize,
    pub pages: usize,
    kv: Vec<Dev<'a>>,
    index: Vec<Option<Dev<'a>>>,
    cos_sin: Dev<'a>,
    workspace: RefCell<Option<Workspace<'a>>>,
    decode_workspace: RefCell<Option<Workspace<'a>>>,
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl<'a> GlmEngine<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(library: &'a NativeLibrary, programs: &'a Dsv4Programs<'a>, cfg: GlmDsaConfig,
        weights: GlmWeights<'a>, stream: *mut c_void, max_context: usize, prefill_rows: usize, pages: usize)
        -> Result<Self> {
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
            workspace: RefCell::new(None), decode_workspace: RefCell::new(None) })
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
            logits: self.alloc(t * self.cfg.vocab_size * 4)?,
            router_logits: self.alloc(t * self.cfg.experts * 4)?,
            route_ids: self.alloc(t * self.cfg.topk * 4)?,
            route_weights: self.alloc(t * self.cfg.topk * 4)?,
            wire: self.alloc(t * (h + h / 32))?,
            shared: self.alloc(t * h * 2)?,
            planes: (0..MAX_RANKS).map(|_| self.alloc(t * h * 2)).collect::<Result<_>>()?,
            router_host: RefCell::new(HostAllocation::new(self.library, t * (self.cfg.topk * 8 + h + h / 32))?),
            planes_host: RefCell::new(HostAllocation::new(self.library, MAX_RANKS * t * h * 2)?),
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
    pub fn prefill(&self, placement: &mut GlmPlacement, embed: &[u8],
        experts: Option<(&mut V41Tp4Roce, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        let (t, start) = (embed.len() / (self.cfg.hidden * 2), placement.len);
        ensure!(t > 0 && t <= self.prefill_rows && start + t <= self.max_context, "prefill of {t} rows at {start}");
        let used = (start + t).div_ceil(PAGE_ROWS);
        let tables = StepTables {
            decode: false,
            positions: (start..start + t).map(|p| p as i64).collect(),
            slots: (start..start + t).map(|p| placement.slot(p)).collect::<Result<_>>()?,
            page_table: placement.pages[..used].to_vec(),
            table_width: used,
            table_stride: 0,
            cache_lengths: (start..start + t).map(|p| (p + 1) as i32).collect(),
        };
        let logits = self.step(&tables, embed, 1, experts, on_layer)?;
        placement.len += t;
        Ok(logits)
    }

    /// Appends each sequence's tokens (one for decode, several for a
    /// speculative verify) at its length in one decode-shaped step; returns
    /// every row's logits. A caller that rejects a suffix sets `len` back:
    /// GLM keeps no recurrent state, and rejected records are overwritten.
    pub fn verify(&self, sequences: &mut [(&mut GlmPlacement, usize)], embed: &[u8],
        experts: Option<(&mut V41Tp4Roce, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        let rows: usize = sequences.iter().map(|(_, n)| n).sum();
        ensure!(rows > 0 && rows <= DECODE_ROWS && embed.len() == rows * self.cfg.hidden * 2, "decode step of {rows} rows");
        let stride = sequences.iter().map(|(p, _)| p.pages.len()).max().unwrap_or(1);
        let mut tables = StepTables { decode: true, positions: Vec::new(), slots: Vec::new(), page_table: Vec::new(),
            table_width: 1, table_stride: stride, cache_lengths: Vec::new() };
        for (placement, count) in sequences.iter() {
            for position in placement.len..placement.len + count {
                ensure!(position < self.max_context, "decode at {position} past the context");
                tables.positions.push(position as i64);
                tables.slots.push(placement.slot(position)?);
                tables.cache_lengths.push((position + 1) as i32);
                tables.table_width = tables.table_width.max((position + 1).div_ceil(PAGE_ROWS));
                let mut pages = placement.pages.clone();
                pages.resize(stride, 0);
                tables.page_table.extend(pages);
            }
        }
        let logits = self.step(&tables, embed, rows, experts, on_layer)?;
        for (placement, count) in sequences.iter_mut() {
            placement.len += *count;
        }
        Ok(logits)
    }

    fn step(&self, tables: &StepTables, embed: &[u8], logit_rows: usize,
        mut experts: Option<(&mut V41Tp4Roce, &tokio::runtime::Runtime)>,
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
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
        self.put(&w.page_table, &tables.page_table)?;
        self.put(&w.cache_lengths, &tables.cache_lengths)?;
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: embed.len(), ..w.h.buffer }, embed)?;
        let rows = Dsv4Scalar::I32(t as i32);
        let cap = if tables.decode { "m64" } else { "m4096" };
        let layers = &self.weights.layers;
        self.norm(w, &layers[0], "input_norm", 0, rows)?;
        for (index, layer) in layers.iter().enumerate() {
            self.attention(w, index, layer, rows, cap, tables)?;
            // h += attention; x = post_attention_layernorm(h)
            self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                ("delta1", w.delta.buffer.ptr), ("weight", layer.ptr("post_norm")?), ("out", w.x.buffer.ptr)],
                &[rows, Dsv4Scalar::I32(1)])?;
            if layer.dense {
                self.run(&format!("glm_ffn_i{}_{cap}", self.cfg.dense_intermediate), &[
                    ("x", w.x.buffer.ptr), ("w_gate_up", layer.ptr("w_gate_up")?), ("w_down", layer.ptr("w_down")?),
                    ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])?;
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
                &[rows, Dsv4Scalar::I32(1)])?;
            if let Some(on_layer) = on_layer.as_mut() {
                on_layer(index, &self.download(&w.h, t * h * 2)?)?;
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

    /// Router, shared expert and the Spark routed experts; leaves
    /// routed + shared in `delta` for the next norm's residual add.
    #[allow(clippy::too_many_arguments)]
    fn moe(&self, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, t: usize, cap: &str, decode: bool,
        transport: &mut V41Tp4Roce, runtime: &tokio::runtime::Runtime) -> Result<()> {
        let kind = if decode { ExpertV2SourceKind::Decode } else { ExpertV2SourceKind::Prefill };
        let (h, topk) = (self.cfg.hidden, self.cfg.topk);
        let rows = Dsv4Scalar::I32(t as i32);
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
            ("scale_mma_ptr", w.delta.buffer.ptr)], &[rows, Dsv4Scalar::I32(grid as i32)])?;
        // Routes and wire rows down to the host for the request.
        let (route_bytes, wire_bytes) = (t * topk * 4, t * (h + h / 32));
        let mut staging = w.router_host.borrow_mut();
        let host = staging.buffer;
        let at = |offset: usize| cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: ids, weights and wire rows are consecutive inside the pinned buffer.
            ptr: unsafe { host.ptr.cast::<u8>().add(offset) }.cast(),
            bytes: host.bytes - offset,
            ..host
        };
        // SAFETY: the pinned regions are large enough; the sync completes them.
        unsafe {
            self.library.copy_d2h_host_buffer_async(at(0), w.route_ids.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(route_bytes), w.route_weights.buffer, route_bytes, self.stream)?;
            self.library.copy_d2h_host_buffer_async(at(2 * route_bytes), w.wire.buffer, wire_bytes, self.stream)?;
            self.library.cuda_stream_synchronize(self.stream)?;
        }
        // The shared expert runs on the GPU while the Sparks compute.
        self.run(&format!("glm_ffn_i{}_{cap}", self.cfg.moe_intermediate), &[
            ("x", w.x.buffer.ptr), ("w_gate_up", layer.ptr("w_gate_up")?), ("w_down", layer.ptr("w_down")?),
            ("out", w.shared.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])?;
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
        runtime.block_on(async {
            transport.execute(&request, |rank, first, payload| {
                let offset = rank * plane_bytes + first as usize * row_bytes;
                ensure!(first as usize * row_bytes + payload.len() <= plane_bytes, "partial rows exceed the step");
                bytes[offset..offset + payload.len()].copy_from_slice(payload);
                Ok(())
            }).await
        })?;
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
        // SAFETY: planes, shared and delta are live [t, h] BF16 buffers ordered after the uploads.
        unsafe {
            self.library.v41_compact_reducer()?.reduce_planes(pointers, ranks as u32, w.shared.buffer.ptr.cast(),
                w.delta.buffer.ptr.cast(), t as u32, self.stream)?;
            // The staging is rewritten by the next layer only after this upload drains.
            self.library.cuda_stream_synchronize(self.stream)
        }
    }

    fn norm(&self, w: &Workspace<'_>, layer: &GlmLayer<'_>, weight: &str, deltas: i32, rows: Dsv4Scalar) -> Result<()> {
        self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
            ("delta1", w.delta.buffer.ptr), ("weight", layer.ptr(weight)?), ("out", w.x.buffer.ptr)],
            &[rows, Dsv4Scalar::I32(deltas)])
    }

    fn attention(&self, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, rows: Dsv4Scalar, cap: &str,
        tables: &StepTables) -> Result<()> {
        let mode = if tables.decode { "decode" } else { "prefill" };
        let kv = &self.kv[index];
        self.run(&format!("glm_producer_{cap}"), &[
            ("x", w.x.buffer.ptr), ("positions", w.positions.buffer.ptr), ("kv_slots", w.slots.buffer.ptr),
            ("cos_sin", self.cos_sin.buffer.ptr), ("w_qkv_a", layer.ptr("w_qkv_a")?), ("q_a_norm", layer.ptr("q_a_norm")?),
            ("kv_a_norm", layer.ptr("kv_a_norm")?), ("w_q_b", layer.ptr("w_q_b")?), ("w_uk", layer.ptr("w_uk")?),
            ("kv_cache", kv.buffer.ptr), ("query", w.query.buffer.ptr), ("q_resid", w.q_resid.buffer.ptr),
            ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        if let Some(index_cache) = &self.index[index] {
            self.run(&format!("glm_index_producer_{cap}"), &[
                ("x", w.x.buffer.ptr), ("q_resid", w.q_resid.buffer.ptr), ("positions", w.positions.buffer.ptr),
                ("index_slots", w.slots.buffer.ptr), ("cos_sin", self.cos_sin.buffer.ptr), ("w_iq", layer.ptr("w_iq")?),
                ("w_ik", layer.ptr("w_ik")?), ("k_norm_w", layer.ptr("k_norm_w")?), ("k_norm_b", layer.ptr("k_norm_b")?),
                ("index_cache", index_cache.buffer.ptr), ("q_fp8", w.q_fp8.buffer.ptr),
                ("head_weights", w.head_weights.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])?;
            self.run(&format!("glm_index_topk_{mode}_{cap}"), &[
                ("q_fp8", w.q_fp8.buffer.ptr), ("weights", w.head_weights.buffer.ptr),
                ("index_k_cache", index_cache.buffer.ptr), ("page_table", w.page_table.buffer.ptr),
                ("cache_lengths", w.cache_lengths.buffer.ptr), ("output_indices", w.indices.buffer.ptr),
                ("scratch", w.topk_scratch.buffer.ptr)],
                &[rows, Dsv4Scalar::I32(tables.table_width as i32), Dsv4Scalar::I32(tables.table_stride as i32)])?;
        }
        // Shared-indexer layers read the previous full layer's `indices`.
        self.run(&format!("glm_sparse_mla_{mode}_{cap}"), &[
            ("q", w.query.buffer.ptr), ("kv_cache", kv.buffer.ptr), ("indices", w.indices.buffer.ptr),
            ("lengths", w.lengths.buffer.ptr), ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        self.run(&format!("glm_o_{cap}"), &[
            ("attn", w.attn.buffer.ptr), ("w_uv", layer.ptr("w_uv")?), ("w_o", layer.ptr("w_o")?),
            ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])
    }
}
