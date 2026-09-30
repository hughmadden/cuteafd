//! DFlash block drafter of MiMo V2.6 Pro (the snapshot's `dflash/`: five
//! Qwen3-style layers of 128 query / 8 KV heads, attention sinks, value
//! scale 0.612, NeoX RoPE on the first 64 of 128 dims, a 1024-position
//! sliding window, block 8).
//!
//! Every target step taps the outputs (pre-norm residual) of the
//! `target_layer_ids` layers (0, 15, 31, 47, 69) into [`MimoDrafter::taps`].
//! Once a step's rows are committed, [`MimoDrafter::update`] projects their
//! taps (`hidden_norm(fc(taps))`) into every draft layer's context K/V (a ring
//! of the sequence's last 1024 positions). [`MimoDrafter::draft`] runs the
//! layers over the block `[anchor, mask x 7]` at the anchor's position (the
//! mask token's row is `mask_embedding.pt`'s trained vector) non-causally
//! against the ring, and takes the argmax of the target head's logits for
//! block rows 1..8. Semantics: python/reference/families/mimo_v2/mimo_dflash/reference.py
//! (SGLang's DFlash path for MiMo). Drafts only steer speculation; verify
//! steps keep output identical to plain greedy decoding.
use crate::shared::fp8_linear::{self, Fp8Weight};
use crate::families::glm5::dflash::FP8_ROWS;
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{read_safetensors_metadata, SafetensorsTensorMetadata};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};

type Dev<'a> = DeviceAllocation<'a>;

/// Context entries per sequence (the checkpoint's sliding window).
pub(crate) const RING: usize = 1024;
/// Tapped rows one target step keeps (a prefill's tail, or a verify step).
pub(crate) const TAP_ROWS: usize = RING;
const WEIGHTS: &str = "dflash_draft_model.safetensors";

#[derive(Debug, Clone, serde::Deserialize)]
struct RawConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    #[serde(default)]
    v_head_dim: Option<usize>,
    #[serde(default = "one")]
    partial_rotary_factor: f64,
    rope_theta: f64,
    rms_norm_eps: f64,
    sliding_window: usize,
    vocab_size: usize,
    #[serde(default)]
    is_causal: bool,
    block_size: usize,
    dflash_config: RawDflash,
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RawDflash {
    target_layer_ids: Vec<usize>,
    mask_token_id: u32,
    #[serde(default)]
    attention_value_scale: Option<f64>,
    #[serde(default)]
    attention_sink_bias: bool,
}

/// The drafter geometry the kernels are written for, read from `dflash/config.json`.
#[derive(Debug, Clone)]
pub(crate) struct DflashConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub theta: f32,
    pub eps: f32,
    pub block: usize,
    pub mask_token: u32,
    pub taps: Vec<usize>,
    pub vocab: usize,
    pub window: usize,
    pub v_scale: f32,
    pub sinks: bool,
}

impl DflashConfig {
    pub fn read(dir: &Path) -> Result<Self> {
        let raw: RawConfig = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .context("parsing dflash/config.json")?;
        let d = &raw.dflash_config;
        let rope_dim = (raw.head_dim as f64 * raw.partial_rotary_factor) as usize;
        ensure!(raw.head_dim == 128 && raw.v_head_dim.unwrap_or(128) == 128 && matches!(rope_dim, 64 | 128)
            && raw.num_key_value_heads > 0 && raw.num_attention_heads % raw.num_key_value_heads == 0 && !raw.is_causal,
            "MiMo DFlash kernels take 128-wide heads, RoPE on 64 or 128 dims and non-causal blocks");
        ensure!(raw.sliding_window == RING, "DFlash window {} (the ring holds {RING})", raw.sliding_window);
        Ok(Self {
            hidden: raw.hidden_size,
            intermediate: raw.intermediate_size,
            layers: raw.num_hidden_layers,
            heads: raw.num_attention_heads,
            kv_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rope_dim,
            theta: raw.rope_theta as f32,
            eps: raw.rms_norm_eps as f32,
            block: raw.block_size,
            mask_token: d.mask_token_id,
            taps: d.target_layer_ids.clone(),
            vocab: raw.vocab_size,
            window: raw.sliding_window,
            v_scale: d.attention_value_scale.unwrap_or(1.0) as f32,
            sinks: d.attention_sink_bias,
        })
    }

    pub fn drafts(&self) -> usize {
        self.block - 1
    }

    fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    fn qkv_width(&self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }
}

/// The drafter directory of a MiMo snapshot (`SNAP/dflash`), or `path` itself.
pub(crate) fn drafter_dir(path: &Path) -> PathBuf {
    if path.join(WEIGHTS).exists() { path.to_path_buf() } else { path.join("dflash") }
}

/// The FP8 copies of one draft layer's GEMM weights.
struct Fp8Layer<'a> {
    qkv: Fp8Weight<'a>,
    o: Fp8Weight<'a>,
    gate_up: Fp8Weight<'a>,
    down: Fp8Weight<'a>,
}

/// Every FP8 copy (and of the target's LM head) and the GEMV's scratch.
struct Fp8Weights<'a> {
    fc: Fp8Weight<'a>,
    head: Fp8Weight<'a>,
    layers: Vec<Fp8Layer<'a>>,
    workspace: Dev<'a>,
}

struct DraftLayer<'a> {
    input_norm: Dev<'a>,
    post_norm: Dev<'a>,
    /// q | k | v rows; the context update reads the k | v rows.
    qkv: Dev<'a>,
    q_norm: Dev<'a>,
    k_norm: Dev<'a>,
    sinks: Option<Dev<'a>>,
    o: Dev<'a>,
    gate_up: Dev<'a>,
    down: Dev<'a>,
    k_ring: Dev<'a>,
    v_ring: Dev<'a>,
}

/// Buffers of one draft step for up to `sequences` sequences.
struct Workspace<'a> {
    sequences: usize,
    h: Dev<'a>,
    n: Dev<'a>,
    qkv: Dev<'a>,
    q: Dev<'a>,
    k: Dev<'a>,
    v: Dev<'a>,
    attn: Dev<'a>,
    delta: Dev<'a>,
    gate_up: Dev<'a>,
    act: Dev<'a>,
    logits: Dev<'a>,
    unary: Dev<'a>,
    candidates: Dev<'a>,
    positions: Dev<'a>,
    tables: Dev<'a>,
    attention_workspace: Dev<'a>,
    topk_workspace: Dev<'a>,
    head: VocabularyHead<'a>,
    _head_workspace: Dev<'a>,
}

/// One sequence to draft for: its ring slot, the token at `position` whose
/// target step has not run yet, and `position` itself (the context length).
#[derive(Debug, Clone, Copy)]
pub(crate) struct DraftSeq {
    pub slot: usize,
    pub anchor: u32,
    pub position: usize,
}

/// One committed tapped row: its row in the last step's taps, the sequence's
/// ring slot and the row's position.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ContextRow {
    pub tap_row: usize,
    pub slot: usize,
    pub position: usize,
}

/// The drafted tokens after one anchor and per-token features (margin of the
/// top logit, its probability and the entropy over the top 16, rank 0).
#[derive(Debug, Clone)]
pub(crate) struct Draft {
    pub tokens: Vec<u32>,
    pub features: Vec<[f32; 4]>,
}

pub(crate) struct MimoDrafter<'a> {
    library: &'a NativeLibrary,
    pub cfg: DflashConfig,
    stream: *mut c_void,
    /// Ring slots (sequences with a drafter context).
    pub slots: usize,
    max_sequences: usize,
    fc: Dev<'a>,
    hidden_norm: Dev<'a>,
    norm: Dev<'a>,
    layers: Vec<DraftLayer<'a>>,
    /// [TAP_ROWS, taps * hidden] BF16: the last step's tapped rows.
    pub taps: Dev<'a>,
    fused: Dev<'a>,
    fused_norm: Dev<'a>,
    context_kv: Dev<'a>,
    context_positions: Dev<'a>,
    context_slots: Dev<'a>,
    /// The mask token's (trained) embedding row.
    mask_row: Vec<u8>,
    workspace: RefCell<Option<Workspace<'a>>>,
    fp8: Option<Fp8Weights<'a>>,
    /// Whether draft steps use `fp8` (a replay toggles it).
    use_fp8: std::cell::Cell<bool>,
}

fn at(dev: &Dev<'_>, bytes: usize) -> *mut c_void {
    // SAFETY: callers stay inside the allocation.
    unsafe { dev.buffer.ptr.cast::<u8>().add(bytes) }.cast()
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes for host->device copies.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

struct Checkpoint {
    data: Vec<u8>,
    tensors: HashMap<String, SafetensorsTensorMetadata>,
}

impl Checkpoint {
    fn bytes(&self, name: &str, shape: &[usize]) -> Result<&[u8]> {
        let t = self.tensors.get(name).with_context(|| format!("DFlash checkpoint has no {name}"))?;
        ensure!(t.shape == shape && t.byte_length as usize == shape.iter().product::<usize>() * 2,
            "{name}: shape {:?}, expected BF16 {shape:?}", t.shape);
        let range = t.byte_offset as usize..(t.byte_offset + t.byte_length) as usize;
        self.data.get(range).with_context(|| format!("{name} lies past the file"))
    }
}

/// Reads the drafter's safetensors file on a thread (while the target loads).
pub(crate) fn prefetch(dir: &Path) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>> {
    let path = dir.join(WEIGHTS);
    std::thread::spawn(move || std::fs::read(path))
}

/// The trained mask embedding (`mask_embedding.pt`: a torch zip whose
/// `*/data/0` entry holds `hidden` BF16 values, stored uncompressed), found
/// through the zip's central directory.
pub(crate) fn mask_embedding(dir: &Path, hidden: usize) -> Result<Option<Vec<u8>>> {
    let path = dir.join("mask_embedding.pt");
    let Ok(bytes) = std::fs::read(&path) else { return Ok(None) };
    let u16_at = |o: usize| -> Result<usize> {
        Ok(usize::from(u16::from_le_bytes(bytes.get(o..o + 2).context("truncated zip")?.try_into().unwrap())))
    };
    let u32_at = |o: usize| -> Result<usize> {
        Ok(u32::from_le_bytes(bytes.get(o..o + 4).context("truncated zip")?.try_into().unwrap()) as usize)
    };
    let end = (0..bytes.len().saturating_sub(21)).rev().find(|&o| bytes[o..o + 4] == [0x50, 0x4b, 0x05, 0x06])
        .with_context(|| format!("{}: not a zip file", path.display()))?;
    let (count, mut at) = (u16_at(end + 10)?, u32_at(end + 16)?);
    for _ in 0..count {
        ensure!(u32_at(at)? == 0x0201_4b50, "{}: bad central directory", path.display());
        let (method, size) = (u16_at(at + 10)?, u32_at(at + 20)?);
        let (name_len, extra_len, comment_len, local) = (u16_at(at + 28)?, u16_at(at + 30)?, u16_at(at + 32)?,
            u32_at(at + 42)?);
        let name = std::str::from_utf8(bytes.get(at + 46..at + 46 + name_len).context("truncated zip")?).unwrap_or("");
        if name.ends_with("/data/0") {
            let data = local + 30 + u16_at(local + 26)? + u16_at(local + 28)?;
            ensure!(method == 0 && size == hidden * 2 && data + size <= bytes.len(),
                "{}: expected {} stored bytes in {name}, found method {method} size {size}", path.display(), hidden * 2);
            return Ok(Some(bytes[data..data + size].to_vec()));
        }
        at += 46 + name_len + extra_len + comment_len;
    }
    anyhow::bail!("{}: no data/0 entry", path.display())
}

impl<'a> MimoDrafter<'a> {
    /// Loads the drafter's weights from `file` (its safetensors bytes, see
    /// [`prefetch`]) and allocates `slots` ring contexts; draft steps take up
    /// to `max_sequences` sequences. `mask_row` is the mask token's embedding.
    pub fn load(library: &'a NativeLibrary, dir: &Path, file: Vec<u8>, stream: *mut c_void, slots: usize,
        max_sequences: usize, mask_row: Vec<u8>) -> Result<Self> {
        let cfg = DflashConfig::read(dir)?;
        ensure!(mask_row.len() == cfg.hidden * 2, "mask row of {} bytes", mask_row.len());
        let checkpoint = Checkpoint {
            data: file,
            tensors: read_safetensors_metadata(&dir.join(WEIGHTS))?.into_iter().map(|t| (t.name.clone(), t)).collect(),
        };
        let upload = |bytes: &[u8]| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.len().max(256))?;
            library.copy_h2d(allocation.buffer, bytes)?;
            Ok(allocation)
        };
        let zeroed = |bytes: usize| -> Result<Dev<'a>> {
            let allocation = DeviceAllocation::new(library, bytes.max(256))?;
            library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
            Ok(allocation)
        };
        let (h, kv, inter) = (cfg.hidden, cfg.kv_width(), cfg.intermediate);
        let tensor = |name: &str, shape: &[usize]| checkpoint.bytes(name, shape).and_then(upload);
        let concat = |parts: &[(&str, usize)], cols: usize| -> Result<Dev<'a>> {
            let total: usize = parts.iter().map(|(_, rows)| rows * cols * 2).sum();
            let allocation = DeviceAllocation::new(library, total)?;
            let mut offset = 0;
            for (name, rows) in parts {
                let bytes = checkpoint.bytes(name, &[*rows, cols])?;
                library.copy_h2d(CuteafdDeviceBuffer { ptr: at(&allocation, offset), bytes: bytes.len(),
                    ..allocation.buffer }, bytes)?;
                offset += bytes.len();
            }
            Ok(allocation)
        };
        let layers = (0..cfg.layers).map(|l| -> Result<DraftLayer<'a>> {
            let p = format!("layers.{l}");
            let a = format!("{p}.self_attn");
            Ok(DraftLayer {
                input_norm: tensor(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_norm: tensor(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                qkv: concat(&[(&format!("{a}.q_proj.weight"), cfg.heads * cfg.head_dim),
                    (&format!("{a}.k_proj.weight"), kv), (&format!("{a}.v_proj.weight"), kv)], h)?,
                q_norm: tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim])?,
                k_norm: tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim])?,
                sinks: if cfg.sinks { Some(tensor(&format!("{a}.attention_sink_bias"), &[cfg.heads])?) } else { None },
                o: tensor(&format!("{a}.o_proj.weight"), &[h, cfg.heads * cfg.head_dim])?,
                gate_up: concat(&[(&format!("{p}.mlp.gate_proj.weight"), inter),
                    (&format!("{p}.mlp.up_proj.weight"), inter)], h)?,
                down: tensor(&format!("{p}.mlp.down_proj.weight"), &[h, inter])?,
                k_ring: zeroed(slots * RING * kv * 2)?,
                v_ring: zeroed(slots * RING * kv * 2)?,
            })
        }).collect::<Result<Vec<_>>>()?;
        let taps = cfg.taps.len() * h;
        Ok(Self {
            library,
            stream,
            slots,
            max_sequences,
            fc: tensor("fc.weight", &[h, taps])?,
            hidden_norm: tensor("hidden_norm.weight", &[h])?,
            norm: tensor("norm.weight", &[h])?,
            layers,
            taps: zeroed(TAP_ROWS * taps * 2)?,
            fused: zeroed(TAP_ROWS * h * 2)?,
            fused_norm: zeroed(TAP_ROWS * h * 2)?,
            context_kv: zeroed(TAP_ROWS * 2 * kv * 2)?,
            context_positions: zeroed(TAP_ROWS * 8)?,
            context_slots: zeroed(TAP_ROWS * 4)?,
            mask_row,
            workspace: RefCell::new(None),
            fp8: None,
            use_fp8: std::cell::Cell::new(false),
            cfg,
        })
    }

    pub fn tap_index(&self, layer: usize) -> Option<usize> {
        self.cfg.taps.iter().position(|&l| l == layer)
    }

    /// Copies target layer output rows `[first, first + n)` of `hidden`
    /// ([rows, hidden] BF16) into tap rows `0..n` when `layer` is tapped.
    pub fn tap(&self, layer: usize, hidden: *const c_void, first: usize, n: usize) -> Result<()> {
        let Some(index) = self.tap_index(layer) else { return Ok(()) };
        let h = self.cfg.hidden;
        ensure!(n <= TAP_ROWS, "{n} tapped rows exceed {TAP_ROWS}");
        // SAFETY: `hidden` holds first + n rows; the tap buffer TAP_ROWS rows.
        unsafe {
            self.library.glm_dflash_tap(hidden.cast::<u8>().add(first * h * 2).cast(), self.taps.buffer.ptr, n, h,
                self.cfg.taps.len() * h, index * h, self.stream)
        }
    }

    /// Makes E4M3 copies of every GEMM weight and of the target's LM head
    /// `head` ([vocab, hidden] BF16) and drafts through them from now on (see
    /// [`crate::families::glm5::dflash::GlmDrafter::enable_fp8`]).
    pub fn enable_fp8(&mut self, head: *const c_void, scales: fp8_linear::Fp8Scales) -> Result<()> {
        let started = std::time::Instant::now();
        let (library, stream) = (self.library, self.stream);
        let pack = |w: *const c_void, n: usize, k: usize| Fp8Weight::pack(library, w, n, k, scales, stream);
        let c = &self.cfg;
        let (h, inter, attention) = (c.hidden, c.intermediate, c.heads * c.head_dim);
        let layers = self.layers.iter().map(|l| -> Result<Fp8Layer<'a>> {
            Ok(Fp8Layer {
                qkv: pack(l.qkv.buffer.ptr, c.qkv_width(), h)?,
                o: pack(l.o.buffer.ptr, h, attention)?,
                gate_up: pack(l.gate_up.buffer.ptr, 2 * inter, h)?,
                down: pack(l.down.buffer.ptr, h, inter)?,
            })
        }).collect::<Result<Vec<_>>>()?;
        let fc = pack(self.fc.buffer.ptr, h, c.taps.len() * h)?;
        let head = pack(head, c.vocab, h)?;
        let mut shapes = vec![(fc.k, fc.n), (head.k, head.n), (h, 2 * c.kv_width())];
        for l in &layers {
            for w in [&l.qkv, &l.o, &l.gate_up, &l.down] {
                shapes.push((w.k, w.n));
            }
        }
        let workspace = fp8_linear::scratch(library, FP8_ROWS, &shapes)?;
        // SAFETY: the packing kernels ran on this stream.
        unsafe { library.cuda_stream_synchronize(stream)? };
        let resident: usize = [&fc, &head].into_iter()
            .chain(layers.iter().flat_map(|l| [&l.qkv, &l.o, &l.gate_up, &l.down])).map(Fp8Weight::bytes).sum();
        tracing::info!(gib = resident as f64 / (1u64 << 30) as f64, ?scales, elapsed_ms = started.elapsed().as_millis() as u64,
            "DFlash drafter FP8 copies (and FP8 LM head) resident");
        self.fp8 = Some(Fp8Weights { fc, head, layers, workspace });
        self.use_fp8.set(true);
        Ok(())
    }

    /// Drafts through the FP8 copies (when made) or the BF16 weights.
    pub fn set_fp8(&self, on: bool) {
        self.use_fp8.set(on && self.fp8.is_some());
    }

    /// `out` [rows, n] = `x` [rows, k] @ `w`^T: the FP8 copy `w8` (from row
    /// `first`) for up to [`FP8_ROWS`] rows while FP8 drafting is on, else cuBLAS BF16.
    ///
    /// # Safety
    /// Pointers are live device buffers of those shapes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn linear(&self, x: *const c_void, w: *const c_void, w8: Option<(&Fp8Weight<'_>, usize)>, out: *mut c_void,
        rows: usize, k: usize, n: usize) -> Result<()> {
        match (self.fp8.as_ref(), w8) {
            (Some(fp8), Some((w8, first))) if self.use_fp8.get() && rows <= FP8_ROWS => {
                // SAFETY: the caller's contract; the scratch was sized for FP8_ROWS rows of every shape.
                unsafe { w8.apply(self.library, x, out, false, rows, first, n, &fp8.workspace, self.stream) }
            }
            // SAFETY: the caller's contract.
            _ => unsafe { self.library.linear_bf16(x, w, out, rows, k, n, self.stream) },
        }
    }

    /// Writes the context K/V of committed tapped rows.
    pub fn update(&self, rows: &[ContextRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let (first, last) = (rows.iter().map(|r| r.tap_row).min().unwrap(), rows.iter().map(|r| r.tap_row).max().unwrap());
        let n = last - first + 1;
        ensure!(last < TAP_ROWS, "tap row {last} past {TAP_ROWS}");
        let (mut positions, mut slots) = (vec![0i64; n], vec![-1i32; n]);
        for row in rows {
            ensure!(row.slot < self.slots, "ring slot {} of {}", row.slot, self.slots);
            positions[row.tap_row - first] = row.position as i64;
            slots[row.tap_row - first] = (row.slot * RING + row.position % RING) as i32;
        }
        self.put(&self.context_positions, bytes_of(&positions))?;
        self.put(&self.context_slots, bytes_of(&slots))?;
        let (h, kv, c) = (self.cfg.hidden, self.cfg.kv_width(), &self.cfg);
        let width = c.taps.len() * h;
        let s = self.stream;
        // SAFETY: every buffer holds TAP_ROWS rows of its width; the stream orders the chain.
        unsafe {
            let fp8 = self.fp8.as_ref();
            self.linear(at(&self.taps, first * width * 2), self.fc.buffer.ptr, fp8.map(|f| (&f.fc, 0)),
                self.fused.buffer.ptr, n, width, h)?;
            self.library.glm_dflash_rmsnorm(self.fused.buffer.ptr, self.hidden_norm.buffer.ptr,
                self.fused_norm.buffer.ptr, n, h, c.eps, s)?;
            for (index, layer) in self.layers.iter().enumerate() {
                let q_rows = c.heads * c.head_dim;
                let kv_rows = at(&layer.qkv, q_rows * h * 2);
                self.linear(self.fused_norm.buffer.ptr, kv_rows, fp8.map(|f| (&f.layers[index].qkv, q_rows)),
                    self.context_kv.buffer.ptr, n, h, 2 * kv)?;
                self.library.mimo_dflash_qk_rope(self.context_kv.buffer.ptr, layer.q_norm.buffer.ptr,
                    layer.k_norm.buffer.ptr, self.context_positions.buffer.ptr, self.context_slots.buffer.ptr,
                    std::ptr::null_mut(), layer.k_ring.buffer.ptr, layer.v_ring.buffer.ptr, n, 0, c.kv_heads,
                    c.rope_dim, c.theta, c.eps, c.v_scale, s)?;
            }
        }
        Ok(())
    }

    fn put(&self, dev: &Dev<'_>, bytes: &[u8]) -> Result<()> {
        ensure!(bytes.len() <= dev.buffer.bytes, "table exceeds its buffer");
        self.library.copy_h2d(CuteafdDeviceBuffer { bytes: bytes.len(), ..dev.buffer }, bytes)
    }

    fn workspace(&self, sequences: usize) -> Result<Workspace<'a>> {
        let c = &self.cfg;
        let rows = sequences * c.block;
        let drafted = sequences * c.drafts();
        let alloc = |bytes: usize| DeviceAllocation::new(self.library, bytes.max(256));
        let head_workspace = alloc(VOCABULARY_HEAD_WORKSPACE)?;
        Ok(Workspace {
            sequences,
            h: alloc(rows * c.hidden * 2)?,
            n: alloc(rows * c.hidden * 2)?,
            qkv: alloc(rows * c.qkv_width() * 2)?,
            q: alloc(rows * c.heads * c.head_dim * 2)?,
            k: alloc(rows * c.kv_width() * 2)?,
            v: alloc(rows * c.kv_width() * 2)?,
            attn: alloc(rows * c.heads * c.head_dim * 2)?,
            delta: alloc(rows * c.hidden * 2)?,
            gate_up: alloc(rows * 2 * c.intermediate * 2)?,
            act: alloc(rows * c.intermediate * 2)?,
            logits: alloc(rows * c.vocab * 4)?,
            unary: alloc(drafted * 16 * 4)?,
            candidates: alloc(drafted * 16 * 4)?,
            positions: alloc(rows * 8)?,
            tables: alloc(3 * sequences * 4)?,
            attention_workspace: alloc(self.library.mimo_dflash_attention_workspace(sequences, c.heads, c.kv_heads,
                c.block, RING + c.block)?)?,
            topk_workspace: alloc(self.library.glm_dflash_topk_workspace(drafted)?)?,
            // SAFETY: the workspace buffer lives in the same struct and drops after the head.
            head: unsafe { self.library.vocabulary_head_rows(head_workspace.buffer.ptr, c.hidden as u32,
                rows as u32, c.vocab as u32)? },
            _head_workspace: head_workspace,
        })
    }

    /// Drafts `block - 1` tokens after each sequence's anchor. `anchor_rows`
    /// holds the anchors' embedding rows; `head` is the target's vocabulary
    /// head [vocab, hidden] BF16.
    pub fn draft(&self, sequences: &[DraftSeq], anchor_rows: &[u8], head: *const c_void) -> Result<Vec<Draft>> {
        let c = &self.cfg;
        let (s_count, block, h) = (sequences.len(), c.block, c.hidden);
        ensure!(s_count > 0 && s_count <= self.max_sequences && anchor_rows.len() == s_count * h * 2,
            "draft step of {s_count} sequences");
        let rows = s_count * block;
        let mut slot = self.workspace.borrow_mut();
        if slot.as_ref().is_none_or(|w| w.sequences < s_count) {
            *slot = None;
            *slot = Some(self.workspace(self.max_sequences)?);
        }
        let w = slot.as_ref().context("draft workspace")?;
        let mut embed = Vec::with_capacity(rows * h * 2);
        let mut positions = Vec::with_capacity(rows);
        let mut tables = vec![0i32; 3 * s_count];
        for (i, seq) in sequences.iter().enumerate() {
            ensure!(seq.slot < self.slots, "ring slot {} of {}", seq.slot, self.slots);
            embed.extend_from_slice(&anchor_rows[i * h * 2..(i + 1) * h * 2]);
            for _ in 1..block {
                embed.extend_from_slice(&self.mask_row);
            }
            positions.extend((seq.position..seq.position + block).map(|p| p as i64));
            tables[i] = seq.slot as i32;
            tables[s_count + i] = seq.position.min(RING) as i32;
            tables[2 * s_count + i] = seq.position as i32;
        }
        self.put(&w.h, &embed)?;
        self.put(&w.positions, bytes_of(&positions))?;
        self.put(&w.tables, bytes_of(&tables))?;
        let (s, l, eps, inter) = (self.stream, self.library, c.eps, c.intermediate);
        let attention_width = c.heads * c.head_dim;
        let fp8 = self.fp8.as_ref().filter(|_| self.use_fp8.get());
        // SAFETY: every workspace buffer holds `rows` rows of its width and the
        // weights their checkpoint shapes; the stream orders the chain.
        unsafe {
            l.glm_dflash_rmsnorm(w.h.buffer.ptr, self.layers[0].input_norm.buffer.ptr, w.n.buffer.ptr, rows, h, eps, s)?;
            for (index, layer) in self.layers.iter().enumerate() {
                let f8 = fp8.map(|f| &f.layers[index]);
                self.linear(w.n.buffer.ptr, layer.qkv.buffer.ptr, f8.map(|f| (&f.qkv, 0)), w.qkv.buffer.ptr, rows, h,
                    c.qkv_width())?;
                l.mimo_dflash_qk_rope(w.qkv.buffer.ptr, layer.q_norm.buffer.ptr, layer.k_norm.buffer.ptr,
                    w.positions.buffer.ptr, std::ptr::null(), w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, rows,
                    c.heads, c.kv_heads, c.rope_dim, c.theta, eps, c.v_scale, s)?;
                l.mimo_dflash_attention(w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, layer.k_ring.buffer.ptr,
                    layer.v_ring.buffer.ptr, w.tables.buffer.ptr, at(&w.tables, s_count * 4),
                    at(&w.tables, 2 * s_count * 4), layer.sinks.as_ref().map_or(std::ptr::null(), |t| t.buffer.ptr),
                    w.attn.buffer.ptr, w.attention_workspace.buffer.ptr, s_count, block, c.heads, c.kv_heads, RING,
                    RING + block, c.window, 1.0 / (c.head_dim as f32).sqrt(), s)?;
                self.linear(w.attn.buffer.ptr, layer.o.buffer.ptr, f8.map(|f| (&f.o, 0)), w.delta.buffer.ptr, rows,
                    attention_width, h)?;
                l.mimo_dflash_add_norm(w.h.buffer.ptr, w.delta.buffer.ptr, layer.post_norm.buffer.ptr, w.n.buffer.ptr,
                    rows, h, eps, s)?;
                self.linear(w.n.buffer.ptr, layer.gate_up.buffer.ptr, f8.map(|f| (&f.gate_up, 0)), w.gate_up.buffer.ptr,
                    rows, h, 2 * inter)?;
                l.glm_dflash_silu_mul(w.gate_up.buffer.ptr, w.act.buffer.ptr, rows, inter, s)?;
                self.linear(w.act.buffer.ptr, layer.down.buffer.ptr, f8.map(|f| (&f.down, 0)), w.delta.buffer.ptr, rows,
                    inter, h)?;
                let next = self.layers.get(index + 1).map_or(self.norm.buffer.ptr, |n| n.input_norm.buffer.ptr);
                l.mimo_dflash_add_norm(w.h.buffer.ptr, w.delta.buffer.ptr, next, w.n.buffer.ptr, rows, h, eps, s)?;
            }
            match fp8 {
                Some(f) if rows <= FP8_ROWS => f.head.apply(l, w.n.buffer.ptr, w.logits.buffer.ptr, true, rows, 0,
                    c.vocab, &f.workspace, s)?,
                _ => w.head.launch(w.n.buffer.ptr.cast(), head.cast(), w.logits.buffer.ptr.cast(), rows as u32, s)?,
            }
            l.glm_dflash_topk(w.logits.buffer.ptr, w.unary.buffer.ptr, w.candidates.buffer.ptr,
                w.topk_workspace.buffer.ptr, s_count, block, c.drafts(), c.vocab, s)?;
            l.cuda_stream_synchronize(s)?;
        }
        let drafted = s_count * c.drafts();
        let mut unary = vec![0u8; drafted * 64];
        let mut candidates = vec![0u8; drafted * 64];
        l.copy_d2h(&mut unary, CuteafdDeviceBuffer { bytes: drafted * 64, ..w.unary.buffer })?;
        l.copy_d2h(&mut candidates, CuteafdDeviceBuffer { bytes: drafted * 64, ..w.candidates.buffer })?;
        let word = |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        Ok((0..s_count).map(|i| {
            let rows = (0..c.drafts()).map(|j| i * c.drafts() + j);
            Draft {
                tokens: rows.clone().map(|r| word(&candidates, r * 16)).collect(),
                features: rows.map(|r| {
                    let values: Vec<f32> = (0..16).map(|k| f32::from_bits(word(&unary, r * 16 + k))).collect();
                    features(&values)
                }).collect(),
            }
        }).collect())
    }

    /// The last draft step's final-norm rows [sequences * block, hidden] BF16.
    pub fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        let slot = self.workspace.borrow();
        let w = slot.as_ref().context("no draft step ran")?;
        let bytes = sequences * self.cfg.block * self.cfg.hidden * 2;
        let mut out = vec![0u8; bytes];
        self.library.copy_d2h(&mut out, CuteafdDeviceBuffer { bytes, ..w.n.buffer })?;
        Ok(out)
    }

    /// Copies BF16 tap rows (host, [n, taps * hidden]) into the tap buffer.
    pub fn put_taps(&self, rows: &[u8]) -> Result<()> {
        self.put(&self.taps, rows)
    }
}

/// (margin, top probability, entropy, rank 0) over one row's top-16 logits
/// (largest first); probabilities are renormalized over the 16.
pub(crate) fn features(top: &[f32]) -> [f32; 4] {
    let best = top[0];
    let mass: Vec<f32> = top.iter().map(|&u| (u - best).exp()).collect();
    let total: f32 = mass.iter().sum();
    let entropy = total.ln() - top.iter().zip(&mass).map(|(&u, &m)| m * (u - best)).sum::<f32>() / total;
    [best - top.get(1).copied().unwrap_or(f32::NEG_INFINITY), 1.0 / total, entropy.max(0.0), 0.0]
}

impl crate::families::glm5::dflash::ReplayDrafter for MimoDrafter<'_> {
    fn block(&self) -> usize {
        self.cfg.block
    }

    fn sequences(&self) -> usize {
        self.max_sequences.min(self.slots)
    }

    fn has_fp8(&self) -> bool {
        self.fp8.is_some()
    }

    fn set_fp8(&self, on: bool) {
        MimoDrafter::set_fp8(self, on);
    }

    fn context(&self, taps: &[u8], first: usize) -> Result<()> {
        let n = taps.len() / (self.cfg.taps.len() * self.cfg.hidden * 2);
        self.put_taps(taps)?;
        self.update(&(0..n).map(|r| ContextRow { tap_row: r, slot: 0, position: first + r }).collect::<Vec<_>>())
    }

    fn draft_tokens(&self, seqs: &[(usize, u32, usize)], anchor_rows: &[u8], head: *const c_void)
        -> Result<Vec<Vec<u32>>> {
        let seqs: Vec<DraftSeq> = seqs.iter().map(|&(slot, anchor, position)| DraftSeq { slot, anchor, position }).collect();
        Ok(self.draft(&seqs, anchor_rows, head)?.into_iter().map(|d| d.tokens).collect())
    }

    fn tap_rows(&self) -> usize {
        TAP_ROWS
    }

    fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        MimoDrafter::last_hidden(self, sequences)
    }
}
