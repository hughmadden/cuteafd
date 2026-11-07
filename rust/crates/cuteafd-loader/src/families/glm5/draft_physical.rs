//! Checkpoint-native standard GQA DFlash owners; graph/program memory is opaque.
use super::draft_representation::{GlmDraftCapacity, GlmDraftGeometry, GlmDraftRepresentation, GlmDraftRuntimeLayout};
use anyhow::{ensure, Context, Result};
use std::path::Path;

/// Context entries each draft layer attends to (the checkpoint's sliding window).
pub const RING: usize = 2048;
/// Tapped rows one target step keeps (a prefill chunk's tail, or a verify step).
pub const TAP_ROWS: usize = RING;
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
    /// Newer configs nest the base in `rope_parameters`; others keep a flat `rope_theta`.
    #[serde(default)]
    rope_parameters: Option<RawRope>,
    #[serde(default)]
    rope_theta: Option<f64>,
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
pub struct GlmDflashConfig {
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
    /// Every block row is a prediction (dSpark's anchor-as-first: `block`
    /// drafts); off, the anchor row only carries context (`block - 1`).
    pub sample_from_anchor: bool,
}

impl GlmDflashConfig {
    /// GQA heads, retained KV and MLP neurons are independent rank-local owners.
    pub fn for_tp(&self, ranks: usize) -> Result<Self> {
        ensure!(matches!(ranks, 1 | 2) && self.heads % ranks == 0
            && self.kv_heads % ranks == 0 && self.intermediate % ranks == 0,
            "DFlash head/MLP geometry cannot split across {ranks} ranks");
        Ok(Self { heads: self.heads / ranks, kv_heads: self.kv_heads / ranks,
            intermediate: self.intermediate / ranks, ..self.clone() })
    }

    pub fn read(snapshot: &Path) -> Result<Self> {
        let raw: RawConfig = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)
            .context("parsing the DFlash2 config.json")?;
        let d = &raw.dflash_config;
        ensure!(matches!(raw.head_dim, 128 | 64) && raw.num_key_value_heads > 0
            && raw.num_attention_heads % raw.num_key_value_heads == 0
            && (raw.num_attention_heads / raw.num_key_value_heads).checked_mul(d.block_size).is_some_and(|rows| rows <= 64),
            "DFlash2 kernels take 128- or 64-wide heads and at most 64 queries per kv head and block \
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
            theta: raw.rope_parameters.as_ref().map(|r| r.rope_theta).or(raw.rope_theta)
                .context("DFlash2 config lacks rope_parameters.rope_theta and rope_theta")? as f32,
            block: d.block_size,
            group: d.conv_group_size,
            mask_token: d.mask_token_id,
            rank: d.selector_rank,
            taps: d.target_layer_ids.clone(),
            vocab: raw.vocab_size,
            window: raw.sliding_window,
            row_window: false,
            sample_from_anchor: false,
        })
    }

    pub fn mla(cfg: &crate::formats::dflash::MlaDflashConfig) -> Self {
        Self { hidden: cfg.mla.hidden, intermediate: cfg.intermediate, layers: cfg.windows.len(),
            heads: cfg.mla.heads, kv_heads: cfg.mla.heads, head_dim: cfg.mla.nope + cfg.mla.rope,
            eps: cfg.eps as f32, theta: 0.0, block: cfg.block, group: cfg.group,
            mask_token: cfg.mask_token, rank: cfg.selector_rank, taps: cfg.taps.clone(), vocab: cfg.vocab,
            window: 0, row_window: false, sample_from_anchor: cfg.drafts() == cfg.block }
    }

    pub fn drafts(&self) -> usize {
        if self.sample_from_anchor { self.block } else { self.block - 1 }
    }

    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub fn qkv_width(&self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }

    pub fn conv_width(&self) -> usize {
        4 * self.hidden / self.group
    }

    pub fn tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let (h, kv, inter) = (self.hidden, self.kv_width(), self.intermediate);
        let mut shapes = vec![
            ("fc.weight".into(), vec![h, self.taps.len() * h]),
            ("hidden_norm.weight".into(), vec![h]),
            ("norm.weight".into(), vec![h]),
            ("candidate_selector.hidden_projection.weight".into(), vec![self.rank, h]),
            ("candidate_selector.predecessor_codebook".into(), vec![self.vocab, self.rank]),
            ("candidate_selector.successor_codebook".into(), vec![self.vocab, self.rank]),
        ];
        for l in 0..self.layers {
            let p = format!("layers.{l}");
            let a = format!("{p}.self_attn");
            shapes.extend([
                (format!("{p}.input_layernorm.weight"), vec![h]),
                (format!("{p}.post_attention_layernorm.weight"), vec![h]),
                (format!("{p}.attention_conv.kernel_projection.weight"), vec![self.conv_width(), h]),
                (format!("{p}.attention_conv.base_kernel"), vec![2, 2, h]),
                (format!("{p}.mlp_conv.kernel_projection.weight"), vec![self.conv_width(), h]),
                (format!("{p}.mlp_conv.base_kernel"), vec![2, 2, h]),
                (format!("{a}.q_proj.weight"), vec![self.heads * self.head_dim, h]),
                (format!("{a}.k_proj.weight"), vec![kv, h]),
                (format!("{a}.v_proj.weight"), vec![kv, h]),
                (format!("{a}.q_norm.weight"), vec![self.head_dim]),
                (format!("{a}.k_norm.weight"), vec![self.head_dim]),
                (format!("{a}.o_proj.weight"), vec![h, self.heads * self.head_dim]),
                (format!("{p}.mlp.gate_proj.weight"), vec![inter, h]),
                (format!("{p}.mlp.up_proj.weight"), vec![inter, h]),
                (format!("{p}.mlp.down_proj.weight"), vec![h, inter]),
            ]);
        }
        shapes
    }

    pub fn geometry(&self) -> GlmDraftGeometry {
        GlmDraftGeometry { hidden: self.hidden as u64, intermediate: self.intermediate as u64,
            layers: self.layers as u64, heads: self.heads as u64, kv_heads: self.kv_heads as u64,
            head_dim: self.head_dim as u64, taps: self.taps.len() as u64, vocab: self.vocab as u64,
            conv_group: self.group as u64, selector_rank: self.rank as u64 }
    }

    pub fn runtime_layout(&self, mode: GlmDraftRepresentation, capacity: GlmDraftCapacity)
        -> Result<GlmDraftRuntimeLayout> {
        Ok(GlmDraftRuntimeLayout::new(self.geometry(), mode, capacity, TAP_ROWS)?)
    }
}


pub use super::draft_physical_layout::GlmDflashPhysicalLayout;
