//! GLM-5.3 full weight variants: which tensors become which steering sites,
//! and the accept bars for an FP8 base (128×128 blocks, F32 `scale_inv`)
//! against a BF16/FP16-mastered published variant (the alternate-weight checkpoint).
//!
//! Sites per layer `L < num_hidden_layers` (the MTP layer is never run, so
//! its edits are ignored):
//! - `self_attn.o_proj` → `Attn`;
//! - dense layers' `mlp.down_proj` → `DenseFfn`;
//! - MoE layers' `mlp.shared_experts.down_proj` → `Ffn` (mode `ffn`: the
//!   projection on the combined routed + shared output, the reference checkpoint's
//!   reference edit) or `SharedExpert` (mode `shared-only`: the shared expert
//!   alone, the alternate-weight checkpoint's edit).
use crate::variants::{DeriveOpts, SiteTensor, TensorSource};
use cuteafd_core::variants::{SiteKey, SiteKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlmVariantMode {
    #[default]
    Ffn,
    SharedOnly,
}
impl std::str::FromStr for GlmVariantMode {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, String> {
        match text {
            "ffn" => Ok(Self::Ffn),
            "shared-only" => Ok(Self::SharedOnly),
            other => Err(format!("GLM-5.3 variant mode {other:?}: expected ffn or shared-only")),
        }
    }
}

/// Accept bars per tensor (pilot, 2026-10-24): o_proj leftover sits at the
/// BF16-versus-FP8 floor (0.0265·‖W‖ in every layer), down_proj at FP16
/// rounding (0.0003·‖W‖); unedited tensors re-encoded from FP8 differ by 0.0002.
pub fn variant_opts(tensor: &str) -> DeriveOpts {
    let base = DeriveOpts { k_max: 4, rank_share: 0.01, max_r_delta: None, min_proj_cos: 0.999, edit_floor: 0.005,
        ..DeriveOpts::default() };
    if tensor.ends_with("self_attn.o_proj.weight") { DeriveOpts { max_r_w: Some(0.03), ..base } }
    else { DeriveOpts { max_r_w: Some(0.001), ..base } }
}

/// The site tensors the published variant carries as plain weights.
pub fn variant_sites(layers: usize, first_moe_layer: usize, mode: GlmVariantMode, variant: &TensorSource) -> Vec<SiteTensor> {
    let mut sites = Vec::new();
    for layer in 0..layers {
        let attn = format!("model.layers.{layer}.self_attn.o_proj.weight");
        let (ffn, kind) = if layer < first_moe_layer {
            (format!("model.layers.{layer}.mlp.down_proj.weight"), SiteKind::DenseFfn)
        } else {
            (format!("model.layers.{layer}.mlp.shared_experts.down_proj.weight"),
                if mode == GlmVariantMode::Ffn { SiteKind::Ffn } else { SiteKind::SharedExpert })
        };
        for (tensor, kind) in [(attn, SiteKind::Attn), (ffn, kind)] {
            if variant.contains(&tensor) {
                sites.push(SiteTensor { key: SiteKey { layer: layer as u16, kind }, opts: variant_opts(&tensor), tensor });
            }
        }
    }
    sites
}

/// Tensor names (with FP8 scale partners) a GLM variant derivation reads from either side.
pub fn wants_tensor(name: &str) -> bool {
    let stem = name.strip_suffix("_scale_inv").unwrap_or(name);
    stem.starts_with("model.layers.") && (stem.ends_with(".self_attn.o_proj.weight")
        || stem.ends_with(".mlp.down_proj.weight") || stem.ends_with(".mlp.shared_experts.down_proj.weight"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sites_follow_layer_kind_and_mode() {
        let dir = tempfile::tempdir().unwrap();
        let mut header = serde_json::Map::new();
        for name in ["model.layers.2.mlp.down_proj.weight", "model.layers.2.self_attn.o_proj.weight",
            "model.layers.3.mlp.shared_experts.down_proj.weight", "model.layers.78.self_attn.o_proj.weight"] {
            header.insert(name.into(), serde_json::json!({"dtype":"F16","shape":[1,1],"data_offsets":[0,2]}));
        }
        let text = serde_json::Value::Object(header).to_string();
        let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
        bytes.extend(text.as_bytes());
        bytes.extend([0, 0]);
        std::fs::write(dir.path().join("v.safetensors"), bytes).unwrap();
        let source = TensorSource::from_files([dir.path().join("v.safetensors")]).unwrap();
        let keys = |mode| variant_sites(78, 3, mode, &source).into_iter().map(|s| (s.key.layer, s.key.kind)).collect::<Vec<_>>();
        assert_eq!(keys(GlmVariantMode::Ffn), [(2, SiteKind::Attn), (2, SiteKind::DenseFfn), (3, SiteKind::Ffn)]);
        assert_eq!(keys(GlmVariantMode::SharedOnly), [(2, SiteKind::Attn), (2, SiteKind::DenseFfn), (3, SiteKind::SharedExpert)]);
        assert_eq!("shared-only".parse::<GlmVariantMode>().unwrap(), GlmVariantMode::SharedOnly);
        assert!("routed".parse::<GlmVariantMode>().is_err());
        assert!(wants_tensor("model.layers.9.self_attn.o_proj.weight_scale_inv"));
        assert!(!wants_tensor("model.layers.9.mlp.experts.0.down_proj.weight"));
    }
}
