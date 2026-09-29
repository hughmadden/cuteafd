//! DFlash2 block drafter for GLM 5.3 (incoai/GLM-5.3-DFlash2: six layers,
//! 64 query heads) and GLM 5.3 Flash (incoai/GLM-5.3-Flash-DFlash2: five
//! layers, 32 query heads over 8 KV heads).
//!
//! Every target step taps the outputs of the `target_layer_ids` layers
//! (GLM 5.3: 5, 19, 33, 47, 61, 75; Flash: 5, 14, 24, 33, 42, where the tap is
//! the mean of the four mHC streams, [`GlmDrafter::tap_streams`]) into
//! [`GlmDrafter::taps`]. Once a
//! step's rows are committed, [`GlmDrafter::update`] projects their taps
//! (`hidden_norm(fc(taps))`) into every draft layer's context K/V, a ring of
//! the sequence's last 2048 positions. [`GlmDrafter::draft`] then runs the
//! Qwen3-style layers (two-tap dynamic convolutions around attention and
//! MLP) over the block `[anchor, mask x 7]` at the anchor's position,
//! non-causally against the ring, takes the top 16 of the target head's
//! logits per drafted row and walks the candidate selector greedily, all on
//! the device. Drafts only steer speculation; the verify step keeps output
//! identical to plain greedy decoding.
use crate::v41_memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::dsv4::{VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{read_safetensors_metadata, SafetensorsTensorMetadata};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

type Dev<'a> = DeviceAllocation<'a>;

/// Context entries each draft layer attends to (the checkpoint's sliding window).
pub(crate) const RING: usize = 2048;
/// Tapped rows one target step keeps (a prefill chunk's tail, or a verify step).
pub(crate) const TAP_ROWS: usize = RING;

#[derive(Debug, Clone, serde::Deserialize)]
struct RawConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    sliding_window: usize,
    vocab_size: usize,
    rope_parameters: RawRope,
    dflash_config: RawDflash,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RawRope {
    rope_theta: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RawDflash {
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    mask_token_id: u32,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
}

/// The drafter geometry the kernels are written for, read from `config.json`.
#[derive(Debug, Clone)]
pub(crate) struct DflashConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub theta: f32,
    pub block: usize,
    pub group: usize,
    pub mask_token: u32,
    pub rank: usize,
    pub taps: Vec<usize>,
    pub vocab: usize,
    /// The sliding window (the ring's length).
    pub window: usize,
    /// Mask each block row's context to its own window (upstream's
    /// `|p - q| < window`); off, every block row sees the last `window`
    /// context entries (the GLM 5.3 port's measured behavior).
    pub row_window: bool,
}

impl DflashConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        let raw: RawConfig = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)
            .context("parsing the DFlash2 config.json")?;
        let d = &raw.dflash_config;
        ensure!(raw.head_dim == 128 && raw.num_key_value_heads > 0
            && raw.num_attention_heads % raw.num_key_value_heads == 0
            && raw.num_attention_heads / raw.num_key_value_heads * d.block_size <= 64,
            "DFlash2 kernels take 128-wide heads and at most 64 queries per kv head and block \
             (got {} heads, {} kv, block {})", raw.num_attention_heads, raw.num_key_value_heads, d.block_size);
        ensure!(d.conv_kernel_size == 2 && d.selector_rank == 256 && d.selector_top_k == 16,
            "DFlash2 kernels take two-tap convolutions and a rank-256 top-16 selector");
        ensure!(raw.sliding_window == RING, "DFlash2 window {} (the ring holds {RING})", raw.sliding_window);
        Ok(Self {
            hidden: raw.hidden_size,
            intermediate: raw.intermediate_size,
            layers: raw.num_hidden_layers,
            heads: raw.num_attention_heads,
            kv_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            eps: raw.rms_norm_eps as f32,
            theta: raw.rope_parameters.rope_theta as f32,
            block: d.block_size,
            group: d.conv_group_size,
            mask_token: d.mask_token_id,
            rank: d.selector_rank,
            taps: d.target_layer_ids.clone(),
            vocab: raw.vocab_size,
            window: raw.sliding_window,
            row_window: false,
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

    fn conv_width(&self) -> usize {
        4 * self.hidden / self.group
    }
}

struct DraftLayer<'a> {
    input_norm: Dev<'a>,
    post_norm: Dev<'a>,
    attn_conv: Dev<'a>,
    attn_base: Dev<'a>,
    mlp_conv: Dev<'a>,
    mlp_base: Dev<'a>,
    /// q | k | v rows; the context update reads the k | v rows.
    qkv: Dev<'a>,
    q_norm: Dev<'a>,
    k_norm: Dev<'a>,
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
    conv: Dev<'a>,
    dynamic: Dev<'a>,
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
    projected: Dev<'a>,
    anchors: Dev<'a>,
    tokens: Dev<'a>,
    features: Dev<'a>,
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

/// The drafted tokens after one anchor and the selector's per-token
/// features (margin, best probability, entropy, rank among the 16).
#[derive(Debug, Clone)]
pub(crate) struct Draft {
    pub tokens: Vec<u32>,
    pub features: Vec<[f32; 4]>,
}

pub(crate) struct GlmDrafter<'a> {
    library: &'a NativeLibrary,
    pub cfg: DflashConfig,
    stream: *mut c_void,
    /// Ring slots (sequences with a drafter context).
    pub slots: usize,
    max_sequences: usize,
    fc: Dev<'a>,
    hidden_norm: Dev<'a>,
    norm: Dev<'a>,
    projection: Dev<'a>,
    predecessor: Dev<'a>,
    successor: Dev<'a>,
    layers: Vec<DraftLayer<'a>>,
    /// [TAP_ROWS, taps * hidden] BF16: the last step's tapped rows.
    pub taps: Dev<'a>,
    /// Context update scratch: [TAP_ROWS, hidden] twice, [TAP_ROWS, 2 kv] and the tables.
    fused: Dev<'a>,
    fused_norm: Dev<'a>,
    context_kv: Dev<'a>,
    context_positions: Dev<'a>,
    context_slots: Dev<'a>,
    /// The mask token's embedding row.
    mask_row: Vec<u8>,
    workspace: RefCell<Option<Workspace<'a>>>,
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

/// Reads the drafter's safetensors file on a thread (while the target loads).
pub(crate) fn prefetch(snapshot: &Path) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>> {
    let path = snapshot.join("model.safetensors");
    std::thread::spawn(move || std::fs::read(path))
}

impl Checkpoint {
    fn bytes(&self, name: &str, shape: &[usize]) -> Result<&[u8]> {
        let t = self.tensors.get(name).with_context(|| format!("DFlash2 checkpoint has no {name}"))?;
        ensure!(t.shape == shape && t.byte_length as usize == shape.iter().product::<usize>() * 2,
            "{name}: shape {:?}, expected BF16 {shape:?}", t.shape);
        let range = t.byte_offset as usize..(t.byte_offset + t.byte_length) as usize;
        self.data.get(range).with_context(|| format!("{name} lies past the file"))
    }
}

impl<'a> GlmDrafter<'a> {
    /// Loads the drafter's weights from `file` (its safetensors bytes, see
    /// [`prefetch`]) and allocates `slots` ring contexts; draft steps take up
    /// to `max_sequences` sequences. `mask_row` is the target embedding of
    /// the mask token.
    #[allow(clippy::too_many_arguments)]
    pub fn load(library: &'a NativeLibrary, snapshot: &Path, file: Vec<u8>, stream: *mut c_void, slots: usize,
        max_sequences: usize, mask_row: Vec<u8>, row_window: bool) -> Result<Self> {
        let cfg = DflashConfig { row_window, ..DflashConfig::read(snapshot)? };
        let path = snapshot.join("model.safetensors");
        let checkpoint = Checkpoint {
            data: file,
            tensors: read_safetensors_metadata(&path)?.into_iter().map(|t| (t.name.clone(), t)).collect(),
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
                attn_conv: tensor(&format!("{p}.attention_conv.kernel_projection.weight"), &[cfg.conv_width(), h])?,
                attn_base: tensor(&format!("{p}.attention_conv.base_kernel"), &[2, 2, h])?,
                mlp_conv: tensor(&format!("{p}.mlp_conv.kernel_projection.weight"), &[cfg.conv_width(), h])?,
                mlp_base: tensor(&format!("{p}.mlp_conv.base_kernel"), &[2, 2, h])?,
                qkv: concat(&[(&format!("{a}.q_proj.weight"), cfg.heads * cfg.head_dim),
                    (&format!("{a}.k_proj.weight"), kv), (&format!("{a}.v_proj.weight"), kv)], h)?,
                q_norm: tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim])?,
                k_norm: tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim])?,
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
            projection: tensor("candidate_selector.hidden_projection.weight", &[cfg.rank, h])?,
            predecessor: tensor("candidate_selector.predecessor_codebook", &[cfg.vocab, cfg.rank])?,
            successor: tensor("candidate_selector.successor_codebook", &[cfg.vocab, cfg.rank])?,
            layers,
            taps: zeroed(TAP_ROWS * taps * 2)?,
            fused: zeroed(TAP_ROWS * h * 2)?,
            fused_norm: zeroed(TAP_ROWS * h * 2)?,
            context_kv: zeroed(TAP_ROWS * 2 * kv * 2)?,
            context_positions: zeroed(TAP_ROWS * 8)?,
            context_slots: zeroed(TAP_ROWS * 4)?,
            mask_row,
            workspace: RefCell::new(None),
            cfg,
        })
    }

    /// Index of `layer` among the tapped target layers.
    pub fn tap_index(&self, layer: usize) -> Option<usize> {
        self.cfg.taps.iter().position(|&l| l == layer)
    }

    /// Copies target layer output rows `[first, first + n)` of `hidden`
    /// ([rows, hidden] BF16) into tap rows `0..n` when `layer` is tapped.
    pub fn tap(&self, layer: usize, hidden: *const c_void, first: usize, n: usize) -> Result<()> {
        self.tap_at(layer, hidden, first, n, 0)
    }

    /// [`Self::tap`] into tap rows `to..to + n`.
    pub fn tap_at(&self, layer: usize, hidden: *const c_void, first: usize, n: usize, to: usize) -> Result<()> {
        let Some(index) = self.tap_index(layer) else { return Ok(()) };
        let (h, width) = (self.cfg.hidden, self.cfg.taps.len() * self.cfg.hidden);
        ensure!(to + n <= TAP_ROWS, "tap rows {to}..{} exceed {TAP_ROWS}", to + n);
        // SAFETY: `hidden` holds first + n rows; the tap buffer TAP_ROWS rows.
        unsafe {
            self.library.glm_dflash_tap(hidden.cast::<u8>().add(first * h * 2).cast(),
                self.taps.buffer.ptr.cast::<u8>().add(to * width * 2).cast(), n, h, width, index * h, self.stream)
        }
    }

    /// Taps the mean of the `hc` streams of rows `[first, first + n)` of
    /// `streams` ([rows, hc, hidden] BF16) into tap rows `0..n` when `layer`
    /// is tapped (GLM 5.3 Flash: the mHC contraction upstream captures).
    pub fn tap_streams(&self, layer: usize, streams: *const c_void, hc: usize, first: usize, n: usize) -> Result<()> {
        self.tap_streams_at(layer, streams, hc, first, n, 0)
    }

    /// [`Self::tap_streams`] into tap rows `to..to + n`.
    pub fn tap_streams_at(&self, layer: usize, streams: *const c_void, hc: usize, first: usize, n: usize, to: usize)
        -> Result<()> {
        let Some(index) = self.tap_index(layer) else { return Ok(()) };
        let (h, width) = (self.cfg.hidden, self.cfg.taps.len() * self.cfg.hidden);
        ensure!(to + n <= TAP_ROWS, "tap rows {to}..{} exceed {TAP_ROWS}", to + n);
        // SAFETY: `streams` holds first + n rows of hc streams; the tap buffer TAP_ROWS rows.
        unsafe {
            self.library.glm_dflash_tap_mean(streams.cast::<u8>().add(first * hc * h * 2).cast(),
                self.taps.buffer.ptr.cast::<u8>().add(to * width * 2).cast(), n, h, hc, width, index * h, self.stream)
        }
    }

    /// Writes the context K/V of committed tapped rows (in tap-row order).
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
        let (h, kv) = (self.cfg.hidden, self.cfg.kv_width());
        let width = self.cfg.taps.len() * h;
        let s = self.stream;
        // SAFETY: every buffer holds TAP_ROWS rows of its width; the stream orders the chain.
        unsafe {
            self.library.linear_bf16(at(&self.taps, first * width * 2), self.fc.buffer.ptr, self.fused.buffer.ptr, n,
                width, h, s)?;
            self.library.glm_dflash_rmsnorm(self.fused.buffer.ptr, self.hidden_norm.buffer.ptr,
                self.fused_norm.buffer.ptr, n, h, self.cfg.eps, s)?;
            for layer in &self.layers {
                let kv_rows = at(&layer.qkv, self.cfg.heads * self.cfg.head_dim * h * 2);
                self.library.linear_bf16(self.fused_norm.buffer.ptr, kv_rows, self.context_kv.buffer.ptr, n, h, 2 * kv, s)?;
                self.library.glm_dflash_qk_rope(self.context_kv.buffer.ptr, layer.q_norm.buffer.ptr,
                    layer.k_norm.buffer.ptr, self.context_positions.buffer.ptr, self.context_slots.buffer.ptr,
                    std::ptr::null_mut(), layer.k_ring.buffer.ptr, layer.v_ring.buffer.ptr, n, 0, self.cfg.kv_heads,
                    self.cfg.theta, self.cfg.eps, s)?;
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
            conv: alloc(rows * c.hidden * 2)?,
            dynamic: alloc(rows * c.conv_width() * 2)?,
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
            projected: alloc(rows * c.rank * 2)?,
            anchors: alloc(sequences * 4)?,
            tokens: alloc(drafted * 4)?,
            features: alloc(drafted * 16)?,
            positions: alloc(rows * 8)?,
            tables: alloc(3 * sequences * 4)?,
            attention_workspace: alloc(self.library.glm_dflash_attention_workspace(sequences, c.kv_heads,
                RING + c.block)?)?,
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
        let anchors: Vec<u32> = sequences.iter().map(|s| s.anchor).collect();
        self.put(&w.anchors, bytes_of(&anchors))?;
        let (s, l) = (self.stream, self.library);
        let (eps, group, inter) = (c.eps, c.group, c.intermediate);
        let attention_width = c.heads * c.head_dim;
        // SAFETY: every workspace buffer holds `rows` rows of its width and the
        // weights their checkpoint shapes; the stream orders the chain.
        unsafe {
            l.glm_dflash_rmsnorm(w.h.buffer.ptr, self.layers[0].input_norm.buffer.ptr, w.n.buffer.ptr, rows, h, eps, s)?;
            for (index, layer) in self.layers.iter().enumerate() {
                l.linear_bf16(w.n.buffer.ptr, layer.attn_conv.buffer.ptr, w.dynamic.buffer.ptr, rows, h, c.conv_width(), s)?;
                l.glm_dflash_conv(w.n.buffer.ptr, w.dynamic.buffer.ptr, layer.attn_base.buffer.ptr, w.conv.buffer.ptr,
                    rows, block, h, group, s)?;
                l.linear_bf16(w.conv.buffer.ptr, layer.qkv.buffer.ptr, w.qkv.buffer.ptr, rows, h, c.qkv_width(), s)?;
                l.glm_dflash_qk_rope(w.qkv.buffer.ptr, layer.q_norm.buffer.ptr, layer.k_norm.buffer.ptr,
                    w.positions.buffer.ptr, std::ptr::null(), w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, rows,
                    c.heads, c.kv_heads, c.theta, eps, s)?;
                l.glm_dflash_attention(w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, layer.k_ring.buffer.ptr,
                    layer.v_ring.buffer.ptr, w.tables.buffer.ptr, at(&w.tables, s_count * 4), at(&w.tables, 2 * s_count * 4),
                    w.attn.buffer.ptr, w.attention_workspace.buffer.ptr, s_count, block, c.heads, c.kv_heads, RING,
                    RING + block, if c.row_window { c.window } else { 0 }, 1.0 / (c.head_dim as f32).sqrt(), s)?;
                l.linear_bf16(w.attn.buffer.ptr, layer.o.buffer.ptr, w.delta.buffer.ptr, rows, attention_width, h, s)?;
                l.glm_dflash_conv_residual_norm(w.delta.buffer.ptr, w.dynamic.buffer.ptr, layer.attn_base.buffer.ptr,
                    w.h.buffer.ptr, layer.post_norm.buffer.ptr, w.h.buffer.ptr, w.n.buffer.ptr, rows, block, h, group, eps, s)?;
                l.linear_bf16(w.n.buffer.ptr, layer.mlp_conv.buffer.ptr, w.dynamic.buffer.ptr, rows, h, c.conv_width(), s)?;
                l.glm_dflash_conv(w.n.buffer.ptr, w.dynamic.buffer.ptr, layer.mlp_base.buffer.ptr, w.conv.buffer.ptr,
                    rows, block, h, group, s)?;
                l.linear_bf16(w.conv.buffer.ptr, layer.gate_up.buffer.ptr, w.gate_up.buffer.ptr, rows, h, 2 * inter, s)?;
                l.glm_dflash_silu_mul(w.gate_up.buffer.ptr, w.act.buffer.ptr, rows, inter, s)?;
                l.linear_bf16(w.act.buffer.ptr, layer.down.buffer.ptr, w.delta.buffer.ptr, rows, inter, h, s)?;
                let next = self.layers.get(index + 1).map_or(self.norm.buffer.ptr, |n| n.input_norm.buffer.ptr);
                l.glm_dflash_conv_residual_norm(w.delta.buffer.ptr, w.dynamic.buffer.ptr, layer.mlp_base.buffer.ptr,
                    w.h.buffer.ptr, next, w.h.buffer.ptr, w.n.buffer.ptr, rows, block, h, group, eps, s)?;
            }
            w.head.launch(w.n.buffer.ptr.cast(), head.cast(), w.logits.buffer.ptr.cast(), rows as u32, s)?;
            l.glm_dflash_topk(w.logits.buffer.ptr, w.unary.buffer.ptr, w.candidates.buffer.ptr,
                w.topk_workspace.buffer.ptr, s_count, block, c.drafts(), c.vocab, s)?;
            l.linear_bf16(w.n.buffer.ptr, self.projection.buffer.ptr, w.projected.buffer.ptr, rows, h, c.rank, s)?;
            l.glm_dflash_select(self.predecessor.buffer.ptr, self.successor.buffer.ptr, w.projected.buffer.ptr,
                w.candidates.buffer.ptr, w.unary.buffer.ptr, w.anchors.buffer.ptr, w.tokens.buffer.ptr,
                w.features.buffer.ptr, s_count, block, c.drafts(), c.rank, s)?;
            l.cuda_stream_synchronize(s)?;
        }
        let drafted = s_count * c.drafts();
        let mut tokens = vec![0u8; drafted * 4];
        let mut features = vec![0u8; drafted * 16];
        l.copy_d2h(&mut tokens, CuteafdDeviceBuffer { bytes: drafted * 4, ..w.tokens.buffer })?;
        l.copy_d2h(&mut features, CuteafdDeviceBuffer { bytes: drafted * 16, ..w.features.buffer })?;
        let word = |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        Ok((0..s_count).map(|i| Draft {
            tokens: (0..c.drafts()).map(|j| word(&tokens, i * c.drafts() + j)).collect(),
            features: (0..c.drafts()).map(|j| {
                let n = (i * c.drafts() + j) * 4;
                [0, 1, 2, 3].map(|k| f32::from_bits(word(&features, n + k)))
            }).collect(),
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
