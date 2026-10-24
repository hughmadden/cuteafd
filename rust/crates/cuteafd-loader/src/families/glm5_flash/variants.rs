//! GLM-5.3 Flash weight variants: which tensors become which sites, and the
//! accept bars per format.
//!
//! Sites per layer `L < num_hidden_layers` (the native MTP layer is never
//! run, so its edits are ignored):
//! - `self_attn.o_proj` → `Attn` (BF16 on KDA layers, FP8 128×128 blocks on MLA layers);
//! - MoE layers' `mlp.shared_experts.down_proj` → `SharedExpert`;
//! - dense layers' (0..first_k_dense_replace) `mlp.down_proj` → `DenseFfn`.
//!
//! A tensor that is not a low-rank output-side edit (dealignai's L44) becomes
//! a stored delta, `y += ΔW x` on variant rows, instead of making the variant
//! unavailable.
use crate::variants::{DeriveOpts, SiteTensor, TensorSource};
use cuteafd_core::variants::{SiteKey, SiteKind};
use cuteafd_core::DType;

pub const PREFIX: &str = "model.language_model.layers.";

/// Accept bars by the variant tensor's format. Pilot (dealignai against the
/// official FP8 checkpoint, 2026-10-24):
/// - BF16 (KDA `o_proj`): next share ≤ 0.0009, |cos| ≥ 0.994, leftover ≤ 0.195·‖ΔW‖;
///   the not-low-rank L44 still holds 0.014 in its fifth direction.
/// - FP8 (MLA `o_proj`, shared down): the rank-1 edit under requantization noise, so
///   the leftover is half of ΔW and bounded against ‖W‖ instead: next share ≤ 0.0153,
///   |cos| ≥ 0.893, leftover ≤ 0.0185·‖W‖; L44's shared down has |cos| 0.18 and 0.027·‖W‖.
pub fn variant_opts(dtype: Option<&DType>) -> DeriveOpts {
    let base = DeriveOpts { k_max: 4, edit_floor: 0.002, delta_fallback: true, ..DeriveOpts::default() };
    match dtype {
        Some(DType::F8E4M3) => DeriveOpts { rank_share: 0.03, max_r_delta: None, max_r_w: Some(0.022), min_proj_cos: 0.85, ..base },
        _ => DeriveOpts { rank_share: 0.01, max_r_delta: Some(0.25), max_r_w: None, min_proj_cos: 0.99, ..base },
    }
}

/// The site tensors the variant carries.
pub fn variant_sites(layers: usize, first_moe_layer: usize, variant: &TensorSource) -> Vec<SiteTensor> {
    let mut sites = Vec::new();
    for layer in 0..layers {
        let attn = format!("{PREFIX}{layer}.self_attn.o_proj.weight");
        let (ffn, kind) = if layer < first_moe_layer {
            (format!("{PREFIX}{layer}.mlp.down_proj.weight"), SiteKind::DenseFfn)
        } else {
            (format!("{PREFIX}{layer}.mlp.shared_experts.down_proj.weight"), SiteKind::SharedExpert)
        };
        for (tensor, kind) in [(attn, SiteKind::Attn), (ffn, kind)] {
            if variant.contains(&tensor) {
                sites.push(SiteTensor { key: SiteKey { layer: layer as u16, kind }, opts: variant_opts(variant.dtype(&tensor)), tensor });
            }
        }
    }
    sites
}

/// Tensor names (with FP8 scale partners) a GLM Flash derivation reads from either side.
pub fn wants_tensor(name: &str) -> bool {
    let stem = name.strip_suffix("_scale_inv").unwrap_or(name);
    stem.starts_with(PREFIX) && (stem.ends_with(".self_attn.o_proj.weight")
        || stem.ends_with(".mlp.down_proj.weight") || stem.ends_with(".mlp.shared_experts.down_proj.weight"))
}

#[cfg(test)]
mod tests;
