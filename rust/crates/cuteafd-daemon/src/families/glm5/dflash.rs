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
//!
//! FP8 drafting ([`GlmDrafter::enable_fp8`], on by default in the serve and
//! golden commands, `--draft-fp8 off` keeps BF16): E4M3 copies of every GEMM
//! weight and of the target's LM head (FP32 scales per output row and
//! 128-wide K block, made at load) run the draft and small context updates
//! through the W8A16 tensor-core GEMV (`fp8_gemv.cu`) up to [`FP8_ROWS`]
//! rows; larger updates (prefill tails) keep cuBLAS BF16. The committed
//! tokens do not change: the target verifies every draft. After Hugh
//! Madden's glm53f-afd FP8 drafter (MIT, v1.1.0 16de2a6).
use crate::shared::fp8_linear::{self, Fp8Weight};
use crate::shared::memory::DeviceAllocation;
use crate::shared::token_io::TokenEmbedding;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::programs::{VocabularyHead, VOCABULARY_HEAD_WORKSPACE};
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
/// Most rows a GEMM runs on the FP8 copies (the GEMV reads the weights once
/// per 64 rows: cuBLAS BF16 wins on prefill-sized updates). A GLM 5.3 draft
/// step of 16 sequences (128 rows) is 16.1 ms BF16, 8-9 ms FP8.
pub(crate) const FP8_ROWS: usize = 128;

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

/// The FP8 copies of one draft layer's GEMM weights.
struct Fp8Layer<'a> {
    attn_conv: Fp8Weight<'a>,
    qkv: Fp8Weight<'a>,
    o: Fp8Weight<'a>,
    mlp_conv: Fp8Weight<'a>,
    gate_up: Fp8Weight<'a>,
    down: Fp8Weight<'a>,
}

/// Every FP8 copy and the GEMV's scratch.
struct Fp8Weights<'a> {
    fc: Fp8Weight<'a>,
    projection: Fp8Weight<'a>,
    head: Fp8Weight<'a>,
    layers: Vec<Fp8Layer<'a>>,
    workspace: Dev<'a>,
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
    /// The block's input ids (anchor, then mask tokens) for the device embedding gather.
    ids: Dev<'a>,
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
/// target step has not run yet, `position` itself (the context length), and
/// the first position whose context entry the ring holds for this sequence
/// (`context_valid_from`: 0 after a full prefill; a prefix-cache restore at
/// `P` starts the drafter cold at `P`, and ring entries before it belong to
/// the slot's previous sequence).
#[derive(Debug, Clone, Copy)]
pub(crate) struct DraftSeq {
    pub slot: usize,
    pub anchor: u32,
    pub position: usize,
    pub valid_from: usize,
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

/// Where a draft step's input rows come from.
#[derive(Clone, Copy)]
enum DraftInput<'r, 'e> {
    /// The anchors' embedding rows (host BF16); the mask rows are the drafter's own copy.
    Rows(&'r [u8]),
    /// Gathered on the device from the target's embedding table by token id.
    Table(&'r TokenEmbedding<'e>),
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

/// Reads the drafter's safetensors file on a thread (while the target loads).
pub(crate) fn prefetch(snapshot: &Path) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>> {
    let path = snapshot.join("model.safetensors");
    std::thread::spawn(move || std::fs::read(path))
}

impl Checkpoint {
    fn bytes(&self, name: &str, shape: &[usize]) -> Result<&[u8]> {
        let t = self.tensors.get(name).with_context(|| format!("DFlash2 checkpoint has no {name}"))?;
        ensure!(t.dtype == DType::Bf16, "{name}: DFlash2 weights must be BF16, found {:?}", t.dtype);
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
            fp8: None,
            use_fp8: std::cell::Cell::new(false),
            cfg,
        })
    }

    /// Makes E4M3 copies of every GEMM weight and of the target's LM head
    /// `head` ([vocab, hidden] BF16), with scales `amax / 448` per output row
    /// and 128-wide K block (or `scales`' other rules),
    /// and drafts through them from now on.
    pub fn enable_fp8(&mut self, head: *const c_void, scales: fp8_linear::Fp8Scales) -> Result<()> {
        let started = std::time::Instant::now();
        let (library, stream) = (self.library, self.stream);
        let pack = |w: *const c_void, n: usize, k: usize| Fp8Weight::pack(library, w, n, k, scales, stream);
        let c = &self.cfg;
        let (h, inter, conv, attention) = (c.hidden, c.intermediate, c.conv_width(), c.heads * c.head_dim);
        let layers = self.layers.iter().map(|l| -> Result<Fp8Layer<'a>> {
            Ok(Fp8Layer {
                attn_conv: pack(l.attn_conv.buffer.ptr, conv, h)?,
                qkv: pack(l.qkv.buffer.ptr, c.qkv_width(), h)?,
                o: pack(l.o.buffer.ptr, h, attention)?,
                mlp_conv: pack(l.mlp_conv.buffer.ptr, conv, h)?,
                gate_up: pack(l.gate_up.buffer.ptr, 2 * inter, h)?,
                down: pack(l.down.buffer.ptr, h, inter)?,
            })
        }).collect::<Result<Vec<_>>>()?;
        let fc = pack(self.fc.buffer.ptr, h, c.taps.len() * h)?;
        let projection = pack(self.projection.buffer.ptr, c.rank, h)?;
        let head = pack(head, c.vocab, h)?;
        let mut shapes = vec![(fc.k, fc.n), (projection.k, projection.n), (head.k, head.n), (h, 2 * c.kv_width())];
        for l in &layers {
            for w in [&l.attn_conv, &l.qkv, &l.o, &l.mlp_conv, &l.gate_up, &l.down] {
                shapes.push((w.k, w.n));
            }
        }
        let workspace = fp8_linear::scratch(library, FP8_ROWS, &shapes)?;
        // SAFETY: the packing kernels ran on this stream.
        unsafe { library.cuda_stream_synchronize(stream)? };
        let resident: usize = [&fc, &projection, &head].into_iter()
            .chain(layers.iter().flat_map(|l| [&l.attn_conv, &l.qkv, &l.o, &l.mlp_conv, &l.gate_up, &l.down]))
            .map(Fp8Weight::bytes).sum();
        tracing::info!(gib = resident as f64 / (1u64 << 30) as f64, ?scales, elapsed_ms = started.elapsed().as_millis() as u64,
            "DFlash2 drafter FP8 copies (and FP8 LM head) resident");
        self.fp8 = Some(Fp8Weights { fc, projection, head, layers, workspace });
        self.use_fp8.set(true);
        Ok(())
    }

    /// Drafts through the FP8 copies (when made) or the BF16 weights.
    pub fn set_fp8(&self, on: bool) {
        self.use_fp8.set(on && self.fp8.is_some());
    }

    /// `out` [rows, n] = `x` [rows, k] @ `w`^T: the FP8 copy `w8` for up to
    /// [`FP8_ROWS`] rows while FP8 drafting is on, else cuBLAS BF16.
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
            let fp8 = self.fp8.as_ref();
            self.linear(at(&self.taps, first * width * 2), self.fc.buffer.ptr, fp8.map(|f| (&f.fc, 0)),
                self.fused.buffer.ptr, n, width, h)?;
            self.library.glm_dflash_rmsnorm(self.fused.buffer.ptr, self.hidden_norm.buffer.ptr,
                self.fused_norm.buffer.ptr, n, h, self.cfg.eps, s)?;
            for (index, layer) in self.layers.iter().enumerate() {
                let q_rows = self.cfg.heads * self.cfg.head_dim;
                let kv_rows = at(&layer.qkv, q_rows * h * 2);
                self.linear(self.fused_norm.buffer.ptr, kv_rows, fp8.map(|f| (&f.layers[index].qkv, q_rows)),
                    self.context_kv.buffer.ptr, n, h, 2 * kv)?;
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
            ids: alloc(rows * 4)?,
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
        self.draft_from(sequences, DraftInput::Rows(anchor_rows), head)
    }

    /// [`Self::draft`] with the block's input rows gathered from the target's
    /// embedding table by token id: each anchor, then the mask token (the
    /// drafter's mask row is that table row).
    pub fn draft_device(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>, head: *const c_void)
        -> Result<Vec<Draft>> {
        self.draft_from(sequences, DraftInput::Table(embedding), head)
    }

    fn draft_from(&self, sequences: &[DraftSeq], input: DraftInput<'_, '_>, head: *const c_void) -> Result<Vec<Draft>> {
        let c = &self.cfg;
        let (s_count, block, h) = (sequences.len(), c.block, c.hidden);
        let rows_ok = match input {
            DraftInput::Rows(r) => r.len() == s_count * h * 2,
            DraftInput::Table(_) => true,
        };
        ensure!(s_count > 0 && s_count <= self.max_sequences && rows_ok, "draft step of {s_count} sequences");
        let rows = s_count * block;
        let mut slot = self.workspace.borrow_mut();
        if slot.as_ref().is_none_or(|w| w.sequences < s_count) {
            *slot = None;
            *slot = Some(self.workspace(self.max_sequences)?);
        }
        let w = slot.as_ref().context("draft workspace")?;
        let mut positions = Vec::with_capacity(rows);
        let mut tables = vec![0i32; 3 * s_count];
        for (i, seq) in sequences.iter().enumerate() {
            ensure!(seq.slot < self.slots, "ring slot {} of {}", seq.slot, self.slots);
            positions.extend((seq.position..seq.position + block).map(|p| p as i64));
            tables[i] = seq.slot as i32;
            tables[s_count + i] = seq.position.saturating_sub(seq.valid_from).min(RING) as i32;
            tables[2 * s_count + i] = seq.position as i32;
        }
        match input {
            DraftInput::Rows(anchor_rows) => {
                let mut embed = Vec::with_capacity(rows * h * 2);
                for i in 0..s_count {
                    embed.extend_from_slice(&anchor_rows[i * h * 2..(i + 1) * h * 2]);
                    for _ in 1..block {
                        embed.extend_from_slice(&self.mask_row);
                    }
                }
                self.put(&w.h, &embed)?;
            }
            DraftInput::Table(embedding) => {
                let ids: Vec<u32> = sequences.iter()
                    .flat_map(|seq| std::iter::once(seq.anchor).chain(std::iter::repeat_n(c.mask_token, block - 1)))
                    .collect();
                embedding.embed(&ids, w.ids.buffer, 1, w.h.buffer, self.stream)?;
            }
        }
        self.put(&w.positions, bytes_of(&positions))?;
        self.put(&w.tables, bytes_of(&tables))?;
        let anchors: Vec<u32> = sequences.iter().map(|s| s.anchor).collect();
        self.put(&w.anchors, bytes_of(&anchors))?;
        let (s, l) = (self.stream, self.library);
        let (eps, group, inter) = (c.eps, c.group, c.intermediate);
        let attention_width = c.heads * c.head_dim;
        let fp8 = self.fp8.as_ref().filter(|_| self.use_fp8.get());
        // SAFETY: every workspace buffer holds `rows` rows of its width and the
        // weights their checkpoint shapes; the stream orders the chain.
        unsafe {
            l.glm_dflash_rmsnorm(w.h.buffer.ptr, self.layers[0].input_norm.buffer.ptr, w.n.buffer.ptr, rows, h, eps, s)?;
            for (index, layer) in self.layers.iter().enumerate() {
                let f8 = fp8.map(|f| &f.layers[index]);
                self.linear(w.n.buffer.ptr, layer.attn_conv.buffer.ptr, f8.map(|f| (&f.attn_conv, 0)), w.dynamic.buffer.ptr,
                    rows, h, c.conv_width())?;
                l.glm_dflash_conv(w.n.buffer.ptr, w.dynamic.buffer.ptr, layer.attn_base.buffer.ptr, w.conv.buffer.ptr,
                    rows, block, h, group, s)?;
                self.linear(w.conv.buffer.ptr, layer.qkv.buffer.ptr, f8.map(|f| (&f.qkv, 0)), w.qkv.buffer.ptr, rows, h,
                    c.qkv_width())?;
                l.glm_dflash_qk_rope(w.qkv.buffer.ptr, layer.q_norm.buffer.ptr, layer.k_norm.buffer.ptr,
                    w.positions.buffer.ptr, std::ptr::null(), w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, rows,
                    c.heads, c.kv_heads, c.theta, eps, s)?;
                l.glm_dflash_attention(w.q.buffer.ptr, w.k.buffer.ptr, w.v.buffer.ptr, layer.k_ring.buffer.ptr,
                    layer.v_ring.buffer.ptr, w.tables.buffer.ptr, at(&w.tables, s_count * 4), at(&w.tables, 2 * s_count * 4),
                    w.attn.buffer.ptr, w.attention_workspace.buffer.ptr, s_count, block, c.heads, c.kv_heads, RING,
                    RING + block, if c.row_window { c.window } else { 0 }, 1.0 / (c.head_dim as f32).sqrt(), s)?;
                self.linear(w.attn.buffer.ptr, layer.o.buffer.ptr, f8.map(|f| (&f.o, 0)), w.delta.buffer.ptr, rows,
                    attention_width, h)?;
                l.glm_dflash_conv_residual_norm(w.delta.buffer.ptr, w.dynamic.buffer.ptr, layer.attn_base.buffer.ptr,
                    w.h.buffer.ptr, layer.post_norm.buffer.ptr, w.h.buffer.ptr, w.n.buffer.ptr, rows, block, h, group, eps, s)?;
                self.linear(w.n.buffer.ptr, layer.mlp_conv.buffer.ptr, f8.map(|f| (&f.mlp_conv, 0)), w.dynamic.buffer.ptr,
                    rows, h, c.conv_width())?;
                l.glm_dflash_conv(w.n.buffer.ptr, w.dynamic.buffer.ptr, layer.mlp_base.buffer.ptr, w.conv.buffer.ptr,
                    rows, block, h, group, s)?;
                self.linear(w.conv.buffer.ptr, layer.gate_up.buffer.ptr, f8.map(|f| (&f.gate_up, 0)), w.gate_up.buffer.ptr,
                    rows, h, 2 * inter)?;
                l.glm_dflash_silu_mul(w.gate_up.buffer.ptr, w.act.buffer.ptr, rows, inter, s)?;
                self.linear(w.act.buffer.ptr, layer.down.buffer.ptr, f8.map(|f| (&f.down, 0)), w.delta.buffer.ptr, rows,
                    inter, h)?;
                let next = self.layers.get(index + 1).map_or(self.norm.buffer.ptr, |n| n.input_norm.buffer.ptr);
                l.glm_dflash_conv_residual_norm(w.delta.buffer.ptr, w.dynamic.buffer.ptr, layer.mlp_base.buffer.ptr,
                    w.h.buffer.ptr, next, w.h.buffer.ptr, w.n.buffer.ptr, rows, block, h, group, eps, s)?;
            }
            match fp8 {
                Some(f) if rows <= FP8_ROWS => f.head.apply(l, w.n.buffer.ptr, w.logits.buffer.ptr, true, rows, 0,
                    c.vocab, &f.workspace, s)?,
                _ => super::launch_head(l, &w.head, w.n.buffer.ptr, head, w.logits.buffer.ptr.cast(), rows, h, c.vocab, s)?,
            }
            l.glm_dflash_topk(w.logits.buffer.ptr, w.unary.buffer.ptr, w.candidates.buffer.ptr,
                w.topk_workspace.buffer.ptr, s_count, block, c.drafts(), c.vocab, s)?;
            self.linear(w.n.buffer.ptr, self.projection.buffer.ptr, fp8.map(|f| (&f.projection, 0)),
                w.projected.buffer.ptr, rows, h, c.rank)?;
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

/// Teacher-forced drafter replay on a golden sequence (after Hugh Madden's
/// glm53f-afd draft_record / draft_replay): the drafter's context follows
/// the golden taps one row at a time and it drafts after every token from
/// `start` on, once through the BF16 weights and once through the FP8
/// copies (when made). Prints, per mode, the drafts accepted as a prefix of
/// the text and of the target's greedy picks (`greedy[p]`: the argmax of
/// the golden logits after token p; a draft past a miss of the text is not
/// scored against them), the first draft's greedy agreement and the draft
/// time; then how often the two modes drafted the same tokens.
/// `taps(first, n)` returns tap rows [n, taps * hidden] BF16.
/// What [`replay`] needs of a block drafter (GLM DFlash2, MiMo DFlash).
pub(crate) trait ReplayDrafter {
    fn block(&self) -> usize;
    /// Sequences a draft step takes (and ring slots).
    fn sequences(&self) -> usize;
    fn has_fp8(&self) -> bool;
    fn set_fp8(&self, on: bool);
    /// Ring context of slot 0 from tap rows [n, taps * hidden] at positions `first..first + n`.
    fn context(&self, taps: &[u8], first: usize) -> Result<()>;
    /// Draft tokens after each (slot, anchor, position).
    fn draft_tokens(&self, seqs: &[(usize, u32, usize)], anchor_rows: &[u8], head: *const c_void)
        -> Result<Vec<Vec<u32>>>;
    /// Longest context update one call takes.
    fn tap_rows(&self) -> usize;
    /// The last draft step's final-norm rows [sequences * block, hidden] BF16.
    fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>>;
}

impl ReplayDrafter for GlmDrafter<'_> {
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
        GlmDrafter::set_fp8(self, on);
    }

    fn context(&self, taps: &[u8], first: usize) -> Result<()> {
        let n = taps.len() / (self.cfg.taps.len() * self.cfg.hidden * 2);
        self.put_taps(taps)?;
        self.update(&(0..n).map(|r| ContextRow { tap_row: r, slot: 0, position: first + r }).collect::<Vec<_>>())
    }

    fn draft_tokens(&self, seqs: &[(usize, u32, usize)], anchor_rows: &[u8], head: *const c_void)
        -> Result<Vec<Vec<u32>>> {
        let seqs: Vec<DraftSeq> = seqs.iter().map(|&(slot, anchor, position)| DraftSeq { slot, anchor, position, valid_from: 0 }).collect();
        Ok(self.draft(&seqs, anchor_rows, head)?.into_iter().map(|d| d.tokens).collect())
    }

    fn tap_rows(&self) -> usize {
        TAP_ROWS
    }

    fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        GlmDrafter::last_hidden(self, sequences)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn replay(drafter: &impl ReplayDrafter, tokens: &[u32], greedy: &[u32],
    taps: &dyn Fn(usize, usize) -> Result<Vec<u8>>, embed: &dyn Fn(&[u32]) -> Result<Vec<u8>>, head: *const c_void,
    start: usize) -> Result<()> {
    let (block, drafts) = (drafter.block(), drafter.block() - 1);
    ensure!(tokens.len() > start + block && greedy.len() >= tokens.len(), "replay needs more than {} tokens", start + block);
    let anchors: Vec<usize> = (start..tokens.len() - block).collect();
    let modes: Vec<bool> = if drafter.has_fp8() { vec![false, true] } else { vec![false] };
    let mut outputs: Vec<Vec<Vec<u32>>> = Vec::new();
    // Final-norm rows of the first 64 anchors per mode.
    let mut hidden: Vec<Vec<Vec<f32>>> = Vec::new();
    for &fp8 in &modes {
        drafter.set_fp8(fp8);
        let (mut done, mut seconds) = (0usize, Vec::with_capacity(anchors.len()));
        let mut out = Vec::with_capacity(anchors.len());
        let mut rows_seen = Vec::new();
        let (mut text, mut greedy_ok, mut first) = (0usize, 0usize, 0usize);
        for &p in &anchors {
            while done < p {
                let n = (p - done).min(drafter.tap_rows());
                drafter.context(&taps(done, n)?, done)?;
                done += n;
            }
            let rows = embed(&[tokens[p]])?;
            let timer = std::time::Instant::now();
            let draft = drafter.draft_tokens(&[(0, tokens[p], p)], &rows, head)?.remove(0);
            seconds.push(timer.elapsed().as_secs_f64());
            if rows_seen.len() < 64 {
                rows_seen.push(drafter.last_hidden(1)?.chunks_exact(2)
                    .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect::<Vec<f32>>());
            }
            text += draft.iter().zip(&tokens[p + 1..]).take_while(|(d, t)| d == t).count();
            let mut kept = 0;
            while kept < drafts && draft[kept] == greedy[p + kept] && (kept == 0 || draft[kept - 1] == tokens[p + kept]) {
                kept += 1;
            }
            greedy_ok += kept;
            first += usize::from(draft[0] == greedy[p]);
            out.push(draft);
        }
        seconds.sort_by(f64::total_cmp);
        let n = anchors.len() as f64;
        println!("draft replay {}: {} anchors, accepted vs text {:.3}, vs greedy {:.3} of {drafts}, first draft = \
            greedy {:.1}%, draft median {:.3} ms (p10 {:.3}, p90 {:.3})", if fp8 { "FP8 " } else { "BF16" },
            anchors.len(), text as f64 / n, greedy_ok as f64 / n, 100.0 * first as f64 / n,
            1e3 * seconds[seconds.len() / 2], 1e3 * seconds[seconds.len() / 10], 1e3 * seconds[seconds.len() * 9 / 10]);
        outputs.push(out);
        hidden.push(rows_seen);
        // Draft steps of several sequences (slot i drafts at the last anchor's position).
        let p = *anchors.last().unwrap();
        let mut line = String::new();
        for count in [1usize, 2, 4, 8, 16].into_iter().filter(|&c| c <= drafter.sequences()) {
            let seqs: Vec<(usize, u32, usize)> = (0..count).map(|slot| (slot, tokens[p], p)).collect();
            let rows = embed(&vec![tokens[p]; count])?;
            let mut times = Vec::new();
            for run in 0..9 {
                let timer = std::time::Instant::now();
                drafter.draft_tokens(&seqs, &rows, head)?;
                if run >= 2 {
                    times.push(timer.elapsed().as_secs_f64());
                }
            }
            times.sort_by(f64::total_cmp);
            line += &format!(" {count}: {:.2}", 1e3 * times[times.len() / 2]);
        }
        println!("draft replay {} step ms by sequences (median of 7):{line}", if fp8 { "FP8 " } else { "BF16" });
    }
    if let [bf16, fp8] = &outputs[..] {
        let same = bf16.iter().zip(fp8).filter(|(a, b)| a == b).count();
        let prefix: usize = bf16.iter().zip(fp8).map(|(a, b)| a.iter().zip(b).take_while(|(x, y)| x == y).count()).sum();
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
            let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            dot / (norm(a) * norm(b)).max(1e-30)
        };
        let worst = hidden[0].iter().zip(&hidden[1]).map(|(a, b)| cosine(a, b)).fold(1f64, f64::min);
        println!("draft replay BF16 vs FP8: identical drafts {same}/{} ({:.1}%), common prefix {:.2} of {drafts}, \
            worst final-norm cosine over the first {} anchors {worst:.6}", bf16.len(), 100.0 * same as f64 / bf16.len() as f64,
            prefix as f64 / bf16.len() as f64, hidden[0].len());
    }
    drafter.set_fp8(true);
    Ok(())
}

/// The golden sequence and its greedy picks (the argmax of each row of
/// `logits.bin`, [tokens, vocab] F32) from a golden directory.
pub(crate) fn golden_sequence(dir: &Path, vocab: usize) -> Result<(Vec<u32>, Vec<u32>)> {
    use std::io::Read;
    let tokens: Vec<u32> = std::fs::read(dir.join("tokens.bin"))?.chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let mut file = std::io::BufReader::with_capacity(1 << 24, std::fs::File::open(dir.join("logits.bin"))?);
    let mut row = vec![0u8; vocab * 4];
    let mut greedy = Vec::with_capacity(tokens.len());
    for _ in 0..tokens.len() {
        file.read_exact(&mut row)?;
        let (mut best, mut value) = (0u32, f32::NEG_INFINITY);
        for (i, b) in row.chunks_exact(4).enumerate() {
            let v = f32::from_le_bytes(b.try_into().unwrap());
            if v > value {
                (best, value) = (i as u32, v);
            }
        }
        greedy.push(best);
    }
    Ok((tokens, greedy))
}

#[cfg(test)]
mod checkpoint_header_tests {
    use super::*;

    fn fixture(dtype: DType) -> Checkpoint {
        Checkpoint { data: vec![0x80, 0x3f, 0, 0x40, 0x40, 0x40, 0x80, 0x40],
            tensors: HashMap::from([("fc.weight".into(), SafetensorsTensorMetadata {
                name: "fc.weight".into(), dtype, shape: vec![2, 2], byte_offset: 0, byte_length: 8,
            })]) }
    }

    #[test]
    fn bf16_payload_is_read_without_conversion() {
        let checkpoint = fixture(DType::Bf16);
        assert_eq!(checkpoint.bytes("fc.weight", &[2, 2]).unwrap(), checkpoint.data);
    }

    #[test]
    fn equal_width_other_dtypes_cannot_be_reinterpreted_as_bf16() {
        for dtype in [DType::F16, DType::I16] {
            let checkpoint = fixture(dtype.clone());
            let error = checkpoint.bytes("fc.weight", &[2, 2]).unwrap_err().to_string();
            assert!(error.contains("fc.weight") && error.contains("must be BF16"));
            assert!(error.contains(&format!("{dtype:?}")));
        }
    }

    #[test]
    fn bf16_dtype_does_not_bypass_shape_or_storage_guards() {
        let mut checkpoint = fixture(DType::Bf16);
        assert!(checkpoint.bytes("fc.weight", &[4, 1]).is_err());
        checkpoint.tensors.get_mut("fc.weight").unwrap().byte_length = 6;
        assert!(checkpoint.bytes("fc.weight", &[2, 2]).is_err());
    }
}
