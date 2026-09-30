//! MiMo's native MTP drafter (`model.mtp.layers.0..2`; MiMo V2 Flash has no
//! other drafter): draft step `k` runs MTP layer `k` as one SWA decoder layer
//! with a dense MLP over `eh_proj(cat(enorm(embed(t)), hnorm(h)))`, then
//! `final_layernorm` and the target head.
//!
//! Stage `k` at row `j` takes the target's last-layer output `h_j` (pre-norm)
//! and the token `t_{j+k+1}` and predicts `t_{j+k+2}` (every stage reads the
//! target's hidden state: chaining a stage's own output measures far worse,
//! python/reference/mimo_mtp/reference.py). Each stage keeps its own SWA ring
//! per sequence (the engine's ring layout and programs). A draft at a
//! sequence of length `len` (tokens `0..len` processed, `t_len` = next) runs
//! stage `k` over its rows `ext[k]..len`: rows `j <= len - k - 1` with their
//! true tokens (committed to the ring for good), rows past with the drafts
//! `d_1..d_k` (committed too, recomputed by the next draft once true). The
//! draft is the argmax of row `len - 1`.
//!
//! The engine taps every step's last-layer rows into a hidden ring (256
//! positions per sequence ring) that the stages read.
use crate::v41_memory::DeviceAllocation;
use super::weights::MimoLayer;

pub(crate) type Dev<'a> = DeviceAllocation<'a>;

/// Positions of target hidden rows kept per sequence ring.
pub(crate) const HIDDEN_ROWS: usize = 256;

/// One MTP stage: its SWA decoder layer, extra weights and per-ring SWA records.
pub(crate) struct MtpStage<'a> {
    pub layer: MimoLayer<'a>,
    /// `eh_proj` BF16 [H, 2H], `enorm`, `hnorm`, `final_layernorm`.
    pub eh: Dev<'a>,
    pub enorm: Dev<'a>,
    pub hnorm: Dev<'a>,
    pub final_norm: Dev<'a>,
    /// [rings * 256, record] SWA records.
    pub ring: Dev<'a>,
}

pub(crate) struct MtpDrafter<'a> {
    pub stages: Vec<MtpStage<'a>>,
    /// [rings * HIDDEN_ROWS, H] BF16: target last-layer outputs by (ring, position % 256).
    pub hidden: Dev<'a>,
    /// Scratch [DECODE_ROWS, H] x 3 and [DECODE_ROWS, 2H].
    pub embed: Dev<'a>,
    pub rows_h: Dev<'a>,
    pub normed_e: Dev<'a>,
    pub normed_h: Dev<'a>,
    pub cat: Dev<'a>,
    /// Per ring and stage: rows `0..ext` of the stage's ring hold true tokens.
    pub ext: std::cell::RefCell<Vec<Vec<usize>>>,
}

/// One sequence's MTP draft request: its ring, its length (`t_len` next) and
/// its tokens `0..=len` (prompt, generated and the next token).
#[derive(Debug, Clone)]
pub(crate) struct MtpSeq<'t> {
    pub ring: usize,
    pub len: usize,
    pub tokens: &'t [u32],
}
