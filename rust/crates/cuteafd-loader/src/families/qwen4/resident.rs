//! Resident representations of the Qwen 3.8 Flash Next projections whose
//! precision serve-qwen4 selects (`--fp8-decode`, `--mtp-fp8-head`). Exactly
//! one representation of each weight is resident: the checkpoint's BF16, or an
//! E4M3 copy made at load (the BF16 source is load staging only).
//!
//! * GDN `w_in` / `w_out` and full-attention `w_in` / `w_o` (target layers and
//!   the MTP layer): BF16, or E4M3 with FP32 128x128 block scales
//!   (`[ceil(N/128), K/128]`), one layout for every decode and prefill program.
//! * `lm_head`, shared by the target and the MTP drafts: BF16, or E4M3 with FP32
//!   per-row x 128-K scales (`[V, H/128]`), run in 16-row `qwen4_head_fp8` spans.
use super::{Qwen4Attention, Qwen4Config};

/// The projection representations a serve configuration selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Qwen4Representation {
    /// GDN and attention in/out projections held as E4M3 only.
    pub fp8_projections: bool,
    /// The shared LM head held as E4M3 only.
    pub fp8_head: bool,
}

/// One selectable weight: its rows and columns and where it sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen4Projection {
    /// Target layer index, or `None` for the MTP layer / the head.
    pub layer: Option<usize>,
    pub operand: &'static str,
    pub rows: usize,
    pub cols: usize,
}

/// Device bytes of a BF16 `[rows, cols]` weight.
pub fn bf16_bytes(rows: usize, cols: usize) -> usize {
    rows * cols * 2
}

/// Device bytes of an E4M3 `[rows, cols]` weight with FP32 128x128 block scales.
pub fn fp8_block_bytes(rows: usize, cols: usize) -> usize {
    rows * cols + fp8_block_scale_bytes(rows, cols)
}

/// Bytes of the FP32 128x128 block-scale grid of a `[rows, cols]` weight.
pub fn fp8_block_scale_bytes(rows: usize, cols: usize) -> usize {
    rows.div_ceil(128) * cols.div_ceil(128) * 4
}

/// Device bytes of an E4M3 `[rows, cols]` weight with FP32 per-row x 128-K scales.
pub fn fp8_row_bytes(rows: usize, cols: usize) -> usize {
    rows * cols + rows * cols.div_ceil(128) * 4
}

/// Selected and alternative bytes of the selectable weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Qwen4ResidentBytes {
    /// The projections as selected, and as BF16 / as FP8.
    pub projections: usize,
    pub projections_bf16: usize,
    pub projections_fp8: usize,
    /// The head as selected, and as BF16 / as FP8.
    pub head: usize,
    pub head_bf16: usize,
    pub head_fp8: usize,
    /// Largest transient device staging of a conversion (one BF16 projection
    /// before its E4M3 copy replaces it; the head converts on the host).
    pub max_device_staging: usize,
}

impl Qwen4ResidentBytes {
    pub fn selected(&self) -> usize {
        self.projections + self.head
    }

    /// What two resident formats (BF16 plus FP8 copies) of the same weights would hold.
    pub fn dual(&self) -> usize {
        self.projections_bf16 + self.projections_fp8 + self.head_bf16 + self.head_fp8
    }
}

/// The two selectable projections of a layer with `attention` (packed as the
/// qwen4 programs take them: GDN `w_in = [qkv; z; b; a]`, attention `w_in =
/// [q; k; v; index_qk]`).
pub fn layer_projections(cfg: &Qwen4Config, attention: Qwen4Attention, layer: Option<usize>) -> [Qwen4Projection; 2] {
    match attention {
        Qwen4Attention::Gdn => [
            Qwen4Projection { layer, operand: "w_in", rows: cfg.gdn_conv_width() + cfg.gdn_value_width()
                + 2 * cfg.gdn_value_heads, cols: cfg.hidden },
            Qwen4Projection { layer, operand: "w_out", rows: cfg.hidden, cols: cfg.gdn_value_width() },
        ],
        Qwen4Attention::Full => [
            Qwen4Projection { layer, operand: "w_in", rows: cfg.attn_in_width(), cols: cfg.hidden },
            Qwen4Projection { layer, operand: "w_o", rows: cfg.hidden, cols: cfg.heads * cfg.head_dim },
        ],
    }
}

/// The selectable projections of target layers `0..layers` (and the MTP layer with `mtp`).
pub fn projections(cfg: &Qwen4Config, layers: usize, mtp: bool) -> Vec<Qwen4Projection> {
    let mut out: Vec<Qwen4Projection> = (0..layers.min(cfg.layers))
        .flat_map(|l| layer_projections(cfg, cfg.attention[l], Some(l))).collect();
    if mtp {
        out.extend(layer_projections(cfg, Qwen4Attention::Full, None));
    }
    out
}

/// Resident bytes of the selectable weights under `selected`.
pub fn resident_bytes(cfg: &Qwen4Config, layers: usize, mtp: bool, selected: Qwen4Representation)
    -> Qwen4ResidentBytes {
    let projections = projections(cfg, layers, mtp);
    let projections_bf16 = projections.iter().map(|p| bf16_bytes(p.rows, p.cols)).sum();
    let projections_fp8 = projections.iter().map(|p| fp8_block_bytes(p.rows, p.cols)).sum();
    let head_bf16 = bf16_bytes(cfg.vocab_size, cfg.hidden);
    let head_fp8 = fp8_row_bytes(cfg.vocab_size, cfg.hidden);
    Qwen4ResidentBytes {
        projections: if selected.fp8_projections { projections_fp8 } else { projections_bf16 },
        projections_bf16,
        projections_fp8,
        head: if selected.fp8_head { head_fp8 } else { head_bf16 },
        head_bf16,
        head_fp8,
        max_device_staging: if selected.fp8_projections {
            projections.iter().map(|p| bf16_bytes(p.rows, p.cols)).max().unwrap_or(0)
        } else {
            0
        },
    }
}

/// The BF16 operand names a layer must not hold when its projections are FP8-only.
pub const BF16_PROJECTION_OPERANDS: [&str; 3] = ["w_in", "w_out", "w_o"];

/// The FP8-only operand pair (E4M3 values, FP32 scales) replacing a BF16 projection operand.
pub fn fp8_operands(operand: &str) -> Option<(&'static str, &'static str)> {
    match operand {
        "w_in" => Some(("w_in_fp8", "w_in_scale")),
        "w_out" => Some(("w_out_fp8", "w_out_scale")),
        "w_o" => Some(("w_o_fp8", "w_o_scale")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Qwen4Config {
        let types: Vec<&str> = (0..48).map(|l| if l % 4 == 3 { "full_attention" } else { "linear_attention" }).collect();
        Qwen4Config::from_hf(&serde_json::json!({
            "model_type": "qwen4_exp", "text_config": {
                "model_type": "qwen4_exp_text", "vocab_size": 248320, "hidden_size": 2560, "num_hidden_layers": 48,
                "layer_types": types, "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640, "rms_norm_eps": 1e-6, "hc_count": 4, "hc_lowrank": 320,
                "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
                "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
                "indexer_n_heads": 4, "indexer_head_dim": 128, "indexer_budget": 2048, "indexer_compress_ratio": 4,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                "linear_conv_kernel_dim": 4, "ple_layer_ids": [2], "ngram_size": 3, "heads_per_ngram": 8,
                "ple_embed_dim": 2560, "ple_conv_kernel_size": 4, "eos_token_id": 248044,
                "output_gate_type": "sigmoid", "mtp_num_hidden_layers": 1}})).unwrap()
    }

    #[test]
    fn program_shapes_and_single_copy_bytes() {
        let cfg = cfg();
        let p = projections(&cfg, 48, true);
        assert_eq!(p.len(), 2 * 48 + 2);
        assert_eq!((p[0].rows, p[0].cols), (16480, 2560));
        assert_eq!((p[1].rows, p[1].cols), (2560, 6144));
        assert_eq!((p[6].rows, p[6].cols, p[6].operand), (13952, 2560, "w_in"));
        assert_eq!((p[7].rows, p[7].cols, p[7].operand), (2560, 6144, "w_o"));
        assert_eq!(fp8_block_scale_bytes(16480, 2560), 129 * 20 * 4);
        let bf16 = resident_bytes(&cfg, 48, true, Qwen4Representation::default());
        let fp8 = resident_bytes(&cfg, 48, true, Qwen4Representation { fp8_projections: true, fp8_head: true });
        assert_eq!(bf16.projections, bf16.projections_bf16);
        assert_eq!(fp8.projections, fp8.projections_fp8);
        assert_eq!(fp8.head, 248320 * 2560 + 248320 * 20 * 4);
        // FP8 single copies hold about half the BF16 bytes, never both formats.
        assert!(fp8.selected() * 2 < bf16.selected() + bf16.selected() / 50);
        assert!(fp8.selected() < fp8.dual() && bf16.selected() < bf16.dual());
        assert_eq!(fp8.max_device_staging, bf16_bytes(16480, 2560));
        assert_eq!(bf16.max_device_staging, 0);
    }
}
