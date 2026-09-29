//! GLM 5.x (glm_moe_dsa) coordinator over the exported glm_* programs.
//!
//! One layer: input norm (fused with the previous layer's residual add),
//! MLA producer (writes the FP8 656-byte latent record), DSA index producer
//! and top-k on full-indexer layers (shared layers reuse the last full
//! layer's selection), sparse MLA, W_UV + o_proj, post-attention norm, then
//! the dense MLP or the MoE (router, expert input quantization, shared
//! expert, routed experts). The latent and index caches share page ids.
use super::weights::{GlmLayer, GlmWeights};
use crate::v41_memory::DeviceAllocation;
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
            workspace: RefCell::new(None) })
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

    fn workspace(&self, t: usize) -> Result<Workspace<'a>> {
        let (h, heads) = (self.cfg.hidden, self.cfg.heads);
        let cap = "m4096";
        let mut scratch = 0usize;
        for name in [format!("glm_producer_{cap}"), format!("glm_index_producer_{cap}"),
            format!("glm_sparse_mla_prefill_{cap}"), format!("glm_o_{cap}"), format!("glm_ffn_i2048_{cap}"),
            format!("glm_ffn_i12288_{cap}")] {
            scratch = scratch.max(self.scratch(&name)?);
        }
        let head_workspace = self.alloc(VOCABULARY_HEAD_WORKSPACE)?;
        let topk = self.alloc(self.scratch(&format!("glm_index_topk_prefill_{cap}"))?)?;
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
            page_table: self.alloc(self.pages * 4)?,
            cache_lengths: self.alloc(t * 4)?,
            scratch: self.alloc(scratch)?,
            topk_scratch: topk,
            logits: self.alloc(t * self.cfg.vocab_size * 4)?,
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
        mut on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<Vec<f32>>> {
        let (h, t, start) = (self.cfg.hidden, embed.len() / (self.cfg.hidden * 2), placement.len);
        ensure!(t > 0 && t <= self.prefill_rows && start + t <= self.max_context, "prefill of {t} rows at {start}");
        if self.workspace.borrow().is_none() {
            *self.workspace.borrow_mut() = Some(self.workspace(self.prefill_rows)?);
        }
        let workspace = self.workspace.borrow();
        let w = workspace.as_ref().context("workspace")?;
        ensure!(t <= w.rows, "prefill exceeds the workspace");
        let positions: Vec<i64> = (start..start + t).map(|p| p as i64).collect();
        let slots = (start..start + t).map(|p| placement.slot(p)).collect::<Result<Vec<_>>>()?;
        let used = (start + t).div_ceil(PAGE_ROWS);
        let lengths: Vec<i32> = (start..start + t).map(|p| (p + 1) as i32).collect();
        self.put(&w.positions, &positions)?;
        self.put(&w.slots, &slots)?;
        self.put(&w.page_table, &placement.pages[..used])?;
        self.put(&w.cache_lengths, &lengths)?;
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: embed.len(), ..w.h.buffer }, embed)?;
        let rows = Dsv4Scalar::I32(t as i32);
        let cap = "m4096";
        let layers = &self.weights.layers;
        self.norm(w, &layers[0], "input_norm", 0, rows)?;
        for (index, layer) in layers.iter().enumerate() {
            self.attention(w, index, layer, rows, cap, used)?;
            // h += attention; x = post_attention_layernorm(h)
            self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
                ("delta1", w.delta.buffer.ptr), ("weight", layer.ptr("post_norm")?), ("out", w.x.buffer.ptr)],
                &[rows, Dsv4Scalar::I32(1)])?;
            if layer.dense {
                self.run(&format!("glm_ffn_i{}_{cap}", self.cfg.dense_intermediate), &[
                    ("x", w.x.buffer.ptr), ("w_gate_up", layer.ptr("w_gate_up")?), ("w_down", layer.ptr("w_down")?),
                    ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])?;
            } else {
                anyhow::bail!("layer {index} is an MoE layer; routed experts are not wired into the GLM engine yet");
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
        placement.len += t;
        if layers.len() < self.cfg.layers {
            // SAFETY: the engine owns this stream.
            unsafe { self.library.cuda_stream_synchronize(self.stream)? };
            return Ok(None);
        }
        // SAFETY: the final norm's output and the head operands are live buffers of these shapes.
        unsafe {
            w.head.launch(w.x.buffer.ptr.cast::<u8>().add((t - 1) * h * 2).cast(), self.weights.head.buffer.ptr.cast(),
                w.logits.buffer.ptr.cast(), 1, self.stream)?;
        }
        let logits = self.download(&w.logits, self.cfg.vocab_size * 4)?;
        Ok(Some(logits.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()))
    }

    fn norm(&self, w: &Workspace<'_>, layer: &GlmLayer<'_>, weight: &str, deltas: i32, rows: Dsv4Scalar) -> Result<()> {
        self.run("glm_norm", &[("residual", w.h.buffer.ptr), ("delta0", w.delta.buffer.ptr),
            ("delta1", w.delta.buffer.ptr), ("weight", layer.ptr(weight)?), ("out", w.x.buffer.ptr)],
            &[rows, Dsv4Scalar::I32(deltas)])
    }

    fn attention(&self, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>, rows: Dsv4Scalar, cap: &str,
        used_pages: usize) -> Result<()> {
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
            self.run(&format!("glm_index_topk_prefill_{cap}"), &[
                ("q_fp8", w.q_fp8.buffer.ptr), ("weights", w.head_weights.buffer.ptr),
                ("index_k_cache", index_cache.buffer.ptr), ("page_table", w.page_table.buffer.ptr),
                ("cache_lengths", w.cache_lengths.buffer.ptr), ("output_indices", w.indices.buffer.ptr),
                ("scratch", w.topk_scratch.buffer.ptr)],
                &[rows, Dsv4Scalar::I32(used_pages as i32), Dsv4Scalar::I32(0)])?;
        }
        // Shared-indexer layers read the previous full layer's `indices`.
        self.run(&format!("glm_sparse_mla_prefill_{cap}"), &[
            ("q", w.query.buffer.ptr), ("kv_cache", kv.buffer.ptr), ("indices", w.indices.buffer.ptr),
            ("lengths", w.lengths.buffer.ptr), ("out", w.attn.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])?;
        self.run(&format!("glm_o_{cap}"), &[
            ("attn", w.attn.buffer.ptr), ("w_uv", layer.ptr("w_uv")?), ("w_o", layer.ptr("w_o")?),
            ("out", w.delta.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[rows])
    }
}
