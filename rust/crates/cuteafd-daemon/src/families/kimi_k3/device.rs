//! The coordinator backend's operation table: each op names the export that
//! binds it. Until an export lands its op refuses with the work item that
//! owns it (#273), never a substitute arithmetic.
use super::block::{MixSite, Ops, Routes};
use anyhow::{bail, ensure, Result};
use cuteafd_ffi::{programs::Programs, KimiFfn, KimiFfnBatch, KimiPrefixMode, KimiResidual,
    KimiResidualBatch, KimiResidualSite};
use cuteafd_loader::families::kimi_k3::config::KimiK3Config;

/// A device-resident batch: `rows` BF16 rows of `width` in one plane.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plane {
    pub(crate) ptr: *mut std::ffi::c_void,
    pub(crate) rows: usize,
    pub(crate) width: usize,
}

/// The exported KDA programs (fork #29, exporter geometry `kimik3`, 48 heads
/// per TP2 rank): `kda_m{rows}` writes the head-local `o_norm(o, g)` output;
/// `o_proj` is not part of the program (`w_o` stays in the packed operand set
/// for the binding).
pub(crate) const KDA_PROGRAMS: [&str; 3] = ["kimik3_kda_m64", "kimik3_kda_m4096", "kimik3_kda_commit"];

/// Fixed-arena schedule for the fused residual update/mix/norm programs.
/// FFN output is folded into the next attention site (or the output site),
/// while a block-boundary FFN replaces the prefix with attention output.
pub(crate) fn residual_site(cfg: &KimiK3Config, site: MixSite) -> Result<KimiResidualSite> {
    ensure!(cfg.residual_block > 0 && cfg.max_residual_blocks() <= 8,
        "Kimi residual arena requires at most eight stored blocks");
    let (candidates, prefix, append) = match site {
        MixSite::Attention(layer) => {
            ensure!(layer < cfg.layers, "Kimi residual attention layer {layer} out of range");
            if layer == 0 { (1, KimiPrefixMode::Keep, true) }
            else { (layer.div_ceil(cfg.residual_block) + 1, KimiPrefixMode::Add,
                layer % cfg.residual_block == 0) }
        }
        MixSite::Ffn(layer) => {
            ensure!(layer < cfg.layers, "Kimi residual FFN layer {layer} out of range");
            (layer / cfg.residual_block + 2,
                if layer % cfg.residual_block == 0 { KimiPrefixMode::Replace } else { KimiPrefixMode::Add }, false)
        }
        MixSite::Output => (cfg.max_residual_blocks() + 1, KimiPrefixMode::Add, false),
    };
    Ok(KimiResidualSite { candidates, prefix, append })
}

/// Resolve each used residual specialization and both dense/shared FFNs once.
/// W9 supplies admitted allocations and TP2 projection slices to these calls.
pub(crate) struct W8<'a> {
    residuals: Vec<(KimiResidualSite, KimiResidual<'a>)>,
    dense: KimiFfn<'a>,
    shared: KimiFfn<'a>,
    cfg: KimiK3Config,
}
impl<'a> W8<'a> {
    pub(crate) fn load(cfg: &KimiK3Config, programs: &'a Programs<'a>) -> Result<Self> {
        ensure!(cfg.hidden == 7168 && cfg.dense_intermediate == 33792
            && cfg.moe_intermediate * cfg.shared_experts == 6144,
            "unsupported Kimi W8 hidden/dense/shared geometry");
        let mut residuals = Vec::new();
        for site in (0..cfg.layers).flat_map(|l| [MixSite::Attention(l), MixSite::Ffn(l)])
            .chain(std::iter::once(MixSite::Output)) {
            let key = residual_site(cfg, site)?;
            if !residuals.iter().any(|(existing, _)| *existing == key) {
                residuals.push((key, KimiResidual::load(programs, key)?));
            }
        }
        Ok(Self { residuals, dense: KimiFfn::load(programs, cfg.dense_intermediate / 2)?,
            shared: KimiFfn::load(programs, cfg.moe_intermediate * cfg.shared_experts / 2)?, cfg: cfg.clone() })
    }

    pub(crate) fn residual(&self, site: MixSite, batch: KimiResidualBatch,
        stream: *mut std::ffi::c_void) -> Result<()> {
        let key = residual_site(&self.cfg, site)?;
        let (_, program) = self.residuals.iter().find(|(existing, _)| *existing == key)
            .ok_or_else(|| anyhow::anyhow!("unresolved Kimi residual site"))?;
        program.launch(batch, stream)?;
        Ok(())
    }

    pub(crate) fn ffn(&self, layer: usize, batch: KimiFfnBatch,
        stream: *mut std::ffi::c_void) -> Result<()> {
        let dense = self.cfg.dense.get(layer).ok_or_else(|| anyhow::anyhow!("Kimi FFN layer out of range"))?;
        if *dense { self.dense.launch(batch, stream)?; } else { self.shared.launch(batch, stream)?; }
        Ok(())
    }
}

/// Which export each op waits on.
pub(crate) const PENDING: &[(&str, &str)] = &[
    ("residual_mix", "W8 shared residual mixer"),
    ("norm", "W8/W9 RMSNorm at hidden 7168 (fused with the residual mix)"),
    ("kda", "W6 exports kimik3_kda_m64 (decode/verify, replay), kimik3_kda_m4096 (prefill) and kimik3_kda_commit; \
        their output is the head-local gated norm, then o_proj by columns and the TP2 partial-sum meet (W9 binding)"),
    ("mla", "W7 NoPE MLA producer/dense attention/gated output exports and mla::Mla caller; W9 binds owned cache/weights and TP2 meet"),
    ("dense_ffn", "W8 Situ dense FFN"),
    ("route", "W5 sigmoid router E896/K16, scale 1.0"),
    ("latent_down", "W9 BF16 GEMM 7168->3584"),
    ("experts", "W4 EXL3 kimik3:exl3-k23 packages + W5 reducer width 3584"),
    ("latent_up", "W9 RMSNorm(3584) + BF16 GEMM 3584->7168"),
    ("shared_ffn", "W8 Situ shared FFN (TP2 over intermediate 6144)"),
    ("add", "W9 BF16 add (fused into the next mix)"),
];

/// The not-yet-bound coordinator backend.
pub(crate) struct Pending;

fn pending<T>(op: &str) -> Result<T> {
    let owner = PENDING.iter().find(|(name, _)| *name == op).map_or("unassigned", |(_, owner)| owner);
    bail!("Kimi K3 {op}: no coordinator export bound yet ({owner}; #273)")
}

impl Ops for Pending {
    type Rows = Plane;
    fn residual_mix(&mut self, _: MixSite, _: &[&Plane], _: &Plane) -> Result<Plane> { pending("residual_mix") }
    fn norm(&mut self, _: Option<usize>, _: bool, _: &Plane) -> Result<Plane> { pending("norm") }
    fn kda(&mut self, _: usize, _: &Plane) -> Result<Plane> { pending("kda") }
    fn mla(&mut self, _: usize, _: &Plane) -> Result<Plane> { pending("mla") }
    fn dense_ffn(&mut self, _: usize, _: &Plane) -> Result<Plane> { pending("dense_ffn") }
    fn route(&mut self, _: usize, _: &Plane) -> Result<Vec<Routes>> { pending("route") }
    fn latent_down(&mut self, _: usize, _: &Plane) -> Result<Plane> { pending("latent_down") }
    fn experts(&mut self, _: usize, _: &Plane, _: &[Routes]) -> Result<Plane> { pending("experts") }
    fn latent_up(&mut self, _: usize, _: &Plane) -> Result<Plane> { pending("latent_up") }
    fn shared_ffn(&mut self, _: usize, _: &Plane) -> Result<Plane> { pending("shared_ffn") }
    fn add(&mut self, _: &Plane, _: &Plane) -> Result<Plane> { pending("add") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_residual_schedule_matches_every_layer_and_output_frontier() {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap();
        let mut stored = 0;
        for layer in 0..cfg.layers {
            let attention = residual_site(&cfg, MixSite::Attention(layer)).unwrap();
            assert_eq!(attention.candidates, stored + 1);
            assert_eq!(attention.prefix, if layer == 0 { KimiPrefixMode::Keep } else { KimiPrefixMode::Add });
            assert_eq!(attention.append, layer % cfg.residual_block == 0);
            if attention.append { stored += 1; }
            let ffn = residual_site(&cfg, MixSite::Ffn(layer)).unwrap();
            assert_eq!(ffn.candidates, stored + 1);
            assert_eq!(ffn.prefix, if attention.append { KimiPrefixMode::Replace } else { KimiPrefixMode::Add });
            assert!(!ffn.append);
        }
        let output = residual_site(&cfg, MixSite::Output).unwrap();
        assert_eq!(output.candidates, 9);
        assert_eq!(output.prefix, KimiPrefixMode::Add);
        assert!(!output.append);
        assert!(residual_site(&cfg, MixSite::Attention(cfg.layers)).is_err());
        assert!(residual_site(&cfg, MixSite::Ffn(cfg.layers)).is_err());
    }

    #[test]
    fn unbound_ops_name_their_work_item() {
        let cfg = cuteafd_loader::families::kimi_k3::config::KimiK3Config::from_hf(
            &cuteafd_loader::plan::testing::kimi_k3_config()).unwrap();
        let plane = Plane { ptr: std::ptr::null_mut(), rows: 1, width: cfg.hidden };
        let error = super::super::block::forward(&cfg, &mut Pending, plane, cfg.layers).unwrap_err().to_string();
        // Layer 0 runs no attention mix: the first op is its input norm.
        assert!(error.contains("Kimi K3 norm") && error.contains("#273"), "{error}");
        let error = Pending.kda(0, &plane).unwrap_err().to_string();
        assert!(error.contains("W6"), "{error}");
    }
}
