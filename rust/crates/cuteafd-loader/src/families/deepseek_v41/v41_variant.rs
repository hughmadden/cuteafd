//! DeepSeek V4.1 weight variants (contract §2–§3): read a published edit of
//! `layers.N.attn.wo_b` as an auxiliary source beside the served stock
//! snapshot, and derive the per-row steering op `y -= α r (rᵀ y)` it encodes.
//!
//! The strict V4.1 catalog refuses a grafted overlay, so the variant is opened
//! on its own: a sidecar safetensors file, or a snapshot directory holding
//! exactly one file of FP8 `wo_b` tensors. The edited sites come from that
//! file's own tensor names; the base tensors come from the served catalog, so
//! the Spark worker identity never changes.
//!
//! Each site is fitted by the shared library ([`crate::variants::derive_site`]).
//! On top, the V4.1 recipe (measured on the alternate-weight 1017 overlay, 2026-10-24):
//! - all rank-1 sites whose directions agree share one direction, stored once;
//! - α comes from the bit-exact FP8 rebuild against the variant's own scales:
//!   the least-squares α is biased low by FP8 requantization noise in ΔW;
//! - every site is checked by rebuilding the variant's FP8 bytes with the op
//!   that will be served.
use crate::variants::{derive_site, e4m3_to_f32, f32_to_e4m3, ue8m0_to_f32, DeriveOpts, Matrix, SiteTensor, TensorSource, VerifyReport};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_core::variants::{SiteKey, SiteKind, SiteOp, VariantError};
use cuteafd_core::DType;
use rayon::prelude::*;
use serde::Serialize;
use std::path::{Path, PathBuf};

/// UE8M0 scale block edge of the official FP8 `wo_b`.
pub const V41_FP8_BLOCK: usize = 32;

/// One FP8 E4M3 matrix with UE8M0 32×32 block scales, as stored.
#[derive(Clone)]
pub struct Fp8Block32 {
    pub rows: usize,
    pub cols: usize,
    pub weight: Vec<u8>,
    pub scale: Vec<u8>,
}
impl Fp8Block32 {
    pub fn new(rows: usize, cols: usize, weight: Vec<u8>, scale: Vec<u8>) -> Result<Self> {
        ensure!(rows % V41_FP8_BLOCK == 0 && cols % V41_FP8_BLOCK == 0, "FP8 shape {rows}x{cols} is not 32-blocked");
        ensure!(weight.len() == rows * cols && scale.len() == (rows / V41_FP8_BLOCK) * (cols / V41_FP8_BLOCK),
            "FP8 payload {}/{} bytes does not match {rows}x{cols} with 32x32 scales", weight.len(), scale.len());
        Ok(Self { rows, cols, weight, scale })
    }
    fn scale_at(&self, row: usize, col: usize) -> f32 {
        ue8m0_to_f32(self.scale[(row / V41_FP8_BLOCK) * (self.cols / V41_FP8_BLOCK) + col / V41_FP8_BLOCK])
    }
    /// Row-major FP32 values (`w · scale`, exact).
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.rows * self.cols];
        out.par_chunks_mut(self.cols).enumerate().for_each(|(i, row)| {
            for (j, value) in row.iter_mut().enumerate() {
                *value = e4m3_to_f32(self.weight[i * self.cols + j]) * self.scale_at(i, j);
            }
        });
        out
    }
}

/// Where raw tensor bytes come from (the served catalog, a safetensors file, a fixture).
pub trait TensorBytes: Sync {
    /// `(dtype, shape, bytes)` of tensor `name`.
    fn read(&self, name: &str) -> Result<(DType, Vec<usize>, Vec<u8>)>;
}

impl TensorBytes for crate::OfficialV41Catalog {
    fn read(&self, name: &str) -> Result<(DType, Vec<usize>, Vec<u8>)> {
        let tensor = self.tensor(name)?;
        let mut bytes = vec![0u8; usize::try_from(tensor.metadata.byte_length)?];
        self.read_projection_once(name, &mut bytes)?;
        Ok((tensor.metadata.dtype.clone(), tensor.metadata.shape.clone(), bytes))
    }
}

/// Read `<name>.weight` and `<name>.scale` as one FP8 block-32 matrix.
pub fn read_fp8_block32(source: &dyn TensorBytes, name: &str) -> Result<Fp8Block32> {
    let (dtype, shape, weight) = source.read(&format!("{name}.weight"))?;
    ensure!(dtype == DType::F8E4M3 && shape.len() == 2, "{name}.weight is {dtype:?} {shape:?}, expected 2-D F8_E4M3");
    let (scale_dtype, scale_shape, scale) = source.read(&format!("{name}.scale"))?;
    ensure!(scale_dtype == DType::F8E8M0 && scale_shape == [shape[0] / V41_FP8_BLOCK, shape[1] / V41_FP8_BLOCK],
        "{name}.scale is {scale_dtype:?} {scale_shape:?}, expected UE8M0 32x32 blocks of {shape:?}");
    Fp8Block32::new(shape[0], shape[1], weight, scale)
}

/// The variant's edited tensors, opened directly (never through the strict catalog).
pub struct V41VariantSource {
    pub file: PathBuf,
    tensors: Vec<crate::SafetensorsTensorMetadata>,
    /// Edited `layers.N.attn.wo_b` layers, ascending.
    pub layers: Vec<u16>,
}
impl TensorBytes for V41VariantSource {
    fn read(&self, name: &str) -> Result<(DType, Vec<usize>, Vec<u8>)> {
        use std::os::unix::fs::FileExt;
        let tensor = self.tensors.iter().find(|t| t.name == name)
            .with_context(|| format!("variant {} has no tensor {name}", self.file.display()))?;
        let mut bytes = vec![0u8; usize::try_from(tensor.byte_length)?];
        std::fs::File::open(&self.file)?.read_exact_at(&mut bytes, tensor.byte_offset)
            .with_context(|| format!("reading {name} from {}", self.file.display()))?;
        Ok((tensor.dtype.clone(), tensor.shape.clone(), bytes))
    }
}

/// `layers.N.attn.wo_b.{weight,scale}` → `(N, suffix)`.
fn wo_b_site(name: &str) -> Option<(u16, &str)> {
    let rest = name.strip_prefix("layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let suffix = rest.strip_prefix("attn.wo_b.")?;
    Some((layer.parse().ok()?, suffix))
}

/// Tensors a V4.1 variant derivation reads from either side ([`crate::variants::family_variants`]).
pub fn wants_tensor(name: &str) -> bool { wo_b_site(name).is_some_and(|(_, s)| s == "weight" || s == "scale") }

/// Shared-library accept bars for an FP8 `wo_b` edit (alternate-weight 1017 pilot): the
/// edit is rank 1 with the second direction at 0.003–0.004 of ‖ΔW‖²; the
/// leftover is FP8 requantization noise (r_ΔW 0.31–0.38, r_W ~0.02), so no
/// leftover bar applies here and the FP8 rebuild is the floor check;
/// unedited layers re-read from the same bytes differ by 0.
pub fn variant_opts() -> DeriveOpts {
    DeriveOpts { k_max: 4, rank_share: 0.005, max_r_delta: None, max_r_w: None, min_proj_cos: 0.98, edit_floor: 0.005,
        ..DeriveOpts::default() }
}

/// The `wo_b` sites below `layers` the variant carries (for `cuteafd plan --variant`).
pub fn variant_sites(layers: usize, variant: &TensorSource) -> Vec<SiteTensor> {
    (0..layers).map(|layer| format!("layers.{layer}.attn.wo_b.weight")).enumerate()
        .filter(|(_, tensor)| variant.contains(tensor))
        .map(|(layer, tensor)| SiteTensor { key: SiteKey { layer: layer as u16, kind: SiteKind::Attn }, opts: variant_opts(), tensor })
        .collect()
}

impl V41VariantSource {
    /// Open a sidecar file, or the one file of FP8 `wo_b` tensors in a snapshot
    /// directory (other formats, such as EXL3 sidecars, are skipped). Every
    /// tensor in the chosen file must be an FP8 `wo_b` weight or its scale of a
    /// backbone layer below `layers`: an edit elsewhere is refused, never dropped.
    pub fn open(path: &Path, layers: u16) -> Result<Self> {
        let file = if path.is_dir() {
            let mut found = Vec::new();
            for entry in std::fs::read_dir(path).with_context(|| format!("listing {}", path.display()))? {
                let file = entry?.path();
                if file.extension().is_none_or(|e| e != "safetensors") { continue; }
                let tensors = crate::read_safetensors_metadata(&file)?;
                if tensors.iter().any(|t| t.dtype == DType::F8E4M3 && wo_b_site(&t.name).is_some_and(|(_, s)| s == "weight")) {
                    found.push(file);
                }
            }
            found.sort();
            match found.as_slice() {
                [one] => one.clone(),
                [] => bail!("{} holds no FP8 layers.N.attn.wo_b tensors", path.display()),
                many => bail!("{} holds several FP8 wo_b files ({}); pass one with source=<file>", path.display(),
                    many.iter().map(|f| f.file_name().unwrap_or_default().to_string_lossy()).collect::<Vec<_>>().join(", ")),
            }
        } else { path.to_path_buf() };
        let tensors = crate::read_safetensors_metadata(&file)?;
        let mut sites = std::collections::BTreeMap::<u16, [bool; 2]>::new();
        for tensor in &tensors {
            let Some((layer, suffix)) = wo_b_site(&tensor.name).filter(|(l, _)| *l < layers) else {
                bail!("variant {} edits {}: only layers.0..{layers}.attn.wo_b is supported for DeepSeek V4.1",
                    file.display(), tensor.name);
            };
            let slot = match suffix { "weight" => 0, "scale" => 1, _ => bail!("variant tensor {} is not weight/scale", tensor.name) };
            sites.entry(layer).or_default()[slot] = true;
        }
        ensure!(!sites.is_empty(), "variant {} holds no tensors", file.display());
        if let Some((layer, _)) = sites.iter().find(|(_, both)| !both.iter().all(|&b| b)) {
            bail!("variant {} has layers.{layer}.attn.wo_b without both weight and scale", file.display());
        }
        Ok(Self { file, tensors, layers: sites.into_keys().collect() })
    }
}

/// The V4.1 recipe's bars and search on top of the shared fit.
#[derive(Debug, Clone, Serialize)]
pub struct V41VariantOpts {
    /// Shared-library fit and accept bars ([`variant_opts`]).
    pub derive: DeriveOpts,
    /// Fraction of the variant's FP8 bytes reproduced by `q(W − A BᵀW)` (measured 0.84–0.87).
    pub rebuild_min: f32,
    /// Rank-1 sites whose directions all agree this well (|cos|) share one.
    pub shared_cos_min: f32,
    /// α search for the rebuild: least-squares α ± `alpha_span` on a grid of `alpha_step`.
    pub alpha_span: f32,
    pub alpha_step: f32,
    /// Rows sampled by the α search (every `stride`-th row); the final check reads all rows.
    pub alpha_row_stride: usize,
}
impl Default for V41VariantOpts {
    fn default() -> Self {
        Self { derive: variant_opts(), rebuild_min: 0.8, shared_cos_min: 0.999, alpha_span: 0.4, alpha_step: 0.02,
            alpha_row_stride: 8 }
    }
}

/// Measured facts about one edited tensor.
#[derive(Debug, Clone, Serialize)]
pub struct V41SiteReport {
    pub layer: u16,
    /// The shared library's fit (least-squares α).
    pub verify: VerifyReport,
    /// α the FP8 rebuild prefers for this site's own direction (rank 1).
    pub alpha_rebuild: Option<f32>,
    /// FP8 bytes equal after `q(W − A BᵀW)` with the served op, and for stock W.
    pub rebuild_exact: f32,
    pub stock_exact: f32,
    pub accept: bool,
    pub reason: Option<String>,
}

/// A derived V4.1 variant: one steering op per edited site.
#[derive(Debug, Clone, Serialize)]
pub struct V41Derivation {
    pub hidden: u32,
    /// Served α per site, or one when the direction is shared.
    pub alpha: Vec<f32>,
    pub shared_direction: bool,
    /// Smallest pairwise |cos| between the rank-1 sites' directions.
    pub direction_cos_min: f32,
    pub sites: Vec<V41SiteReport>,
    /// Layers the variant carries unchanged against the base (no site).
    pub unchanged: Vec<u16>,
    #[serde(skip)]
    pub ops: Vec<(SiteKey, SiteOp)>,
    pub accept: bool,
    pub reason: Option<String>,
}

fn dot(a: &[f32], b: &[f32]) -> f64 { a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum() }
fn normalize(v: &mut [f32]) {
    let n = dot(v, v).sqrt();
    if n > 0.0 { v.iter_mut().for_each(|x| *x = (f64::from(*x) / n) as f32); }
}
/// `uᵀ M` for row-major `M [rows][cols]`.
fn left_mul(m: &[f32], cols: usize, u: &[f32]) -> Vec<f32> {
    m.par_chunks(cols * 64).zip(u.par_chunks(64)).map(|(block, u)| {
        let mut acc = vec![0f64; cols];
        for (row, &ui) in block.chunks(cols).zip(u) {
            for (a, &x) in acc.iter_mut().zip(row) { *a += f64::from(ui) * f64::from(x); }
        }
        acc
    }).reduce(|| vec![0f64; cols], |mut a, b| { a.iter_mut().zip(b).for_each(|(x, y)| *x += y); a })
        .into_iter().map(|v| v as f32).collect()
}

/// Fraction of `variant` bytes equal to `q(W − α u pᵀ)` (`p = uᵀW`) with the
/// variant's own scales, over every `stride`-th row (`alpha = 0`: stock W requantized).
fn rebuild_exact(w: &[f32], variant: &Fp8Block32, u: &[f32], p: &[f32], alpha: f32, stride: usize) -> f32 {
    let cols = variant.cols;
    let (equal, total) = (0..variant.rows).into_par_iter().step_by(stride).map(|i| {
        let mut equal = 0usize;
        for j in 0..cols {
            let x = w[i * cols + j] - alpha * (u[i] * p[j]);
            equal += usize::from(f32_to_e4m3(x / variant.scale_at(i, j)) == variant.weight[i * cols + j]);
        }
        (equal, cols)
    }).reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    equal as f32 / total as f32
}
/// The same for a general op (`q(rebuild(W, op))`).
fn rebuild_exact_op(w: &Matrix, variant: &Fp8Block32, op: &SiteOp) -> f32 {
    let x = crate::variants::rebuild(w, op);
    let equal: usize = x.data.par_iter().enumerate().map(|(k, &v)| {
        usize::from(f32_to_e4m3(v / variant.scale_at(k / variant.cols, k % variant.cols)) == variant.weight[k])
    }).sum();
    equal as f32 / x.data.len() as f32
}

/// The α near `start` that the variant's FP8 bytes reproduce best (the middle
/// of the best plateau: neighbouring α often requantize to the same bytes).
fn rebuild_alpha(w: &[f32], variant: &Fp8Block32, u: &[f32], p: &[f32], start: f32, opts: &V41VariantOpts) -> f32 {
    let steps = (opts.alpha_span / opts.alpha_step).round() as i32;
    let start = (start / opts.alpha_step).round() * opts.alpha_step;
    let scored: Vec<(f32, f32)> = (-steps..=steps).map(|s| start + s as f32 * opts.alpha_step)
        .map(|a| (a, rebuild_exact(w, variant, u, p, a, opts.alpha_row_stride))).collect();
    let best = scored.iter().map(|s| s.1).fold(f32::MIN, f32::max);
    let plateau: Vec<f32> = scored.iter().filter(|s| s.1 == best).map(|s| s.0).collect();
    plateau[plateau.len() / 2]
}

/// Derive the V4.1 variant: fit each edited `wo_b` against the served base
/// with the shared library, then apply the recipe (shared direction, α by FP8
/// rebuild) and check every site with the op that will be served. A rejected
/// site makes the variant unavailable (`accept == false`) with the reason;
/// base serving is unaffected.
pub fn derive_v41_variant(base: &dyn TensorBytes, variant: &V41VariantSource, hidden: usize,
    opts: &V41VariantOpts) -> Result<V41Derivation> {
    struct Fit { layer: u16, w: Matrix, edit: Fp8Block32, op: SiteOp, verify: VerifyReport, stock: f32, best_alpha: Option<f32> }
    let mut fits = Vec::with_capacity(variant.layers.len());
    let mut unchanged = Vec::new();
    for &layer in &variant.layers {
        let name = format!("layers.{layer}.attn.wo_b");
        let stock = read_fp8_block32(base, &name).with_context(|| format!("base {name}"))?;
        let edit = read_fp8_block32(variant, &name).with_context(|| format!("variant {name}"))?;
        ensure!(stock.rows == hidden && edit.rows == stock.rows && edit.cols == stock.cols,
            "{name}: variant {}x{} differs from base {}x{} (hidden {hidden})", edit.rows, edit.cols, stock.rows, stock.cols);
        let w = Matrix::new(stock.rows, stock.cols, stock.dequantize());
        let v = Matrix::new(edit.rows, edit.cols, edit.dequantize());
        let (op, verify) = match derive_site(&w, &v, &opts.derive) {
            Ok(fit) => fit,
            Err(VariantError::Unchanged) => { unchanged.push(layer); continue; }
            Err(error) => bail!("{name}: {error}"),
        };
        drop(v);
        let stock_exact = rebuild_exact(&w.data, &edit, &vec![0.0; w.rows], &vec![0.0; w.cols], 0.0, 1);
        let best_alpha = match &op {
            SiteOp::Steer { k: 1, b, .. } => {
                let p = left_mul(&w.data, w.cols, b);
                Some(rebuild_alpha(&w.data, &edit, b, &p, verify.alphas[0], opts))
            }
            _ => None,
        };
        tracing::info!(layer, k = verify.k, rel_delta = verify.rel_delta, top = verify.top_share, next = verify.next_share,
            alpha_ls = ?verify.alphas, alpha_rebuild = ?best_alpha, cos = verify.proj_cos, "V4.1 variant site fit");
        fits.push(Fit { layer, w, edit, op, verify, stock: stock_exact, best_alpha });
    }
    ensure!(!fits.is_empty(), "variant {}: every wo_b tensor is unchanged against the base", variant.file.display());
    // One direction for every site when all are rank 1 and agree (sign-aligned mean).
    let directions: Vec<&[f32]> = fits.iter().filter_map(|f| match &f.op { SiteOp::Steer { k: 1, b, .. } => Some(b.as_slice()), _ => None }).collect();
    let mut cos_min = 1.0f32;
    for (i, a) in directions.iter().enumerate() {
        for b in &directions[i + 1..] { cos_min = cos_min.min(dot(a, b).abs() as f32); }
    }
    let shared = directions.len() == fits.len() && fits.len() > 1 && cos_min >= opts.shared_cos_min;
    let served: Vec<SiteOp> = if shared {
        let mut r = vec![0f32; hidden];
        for u in &directions {
            let sign = dot(u, directions[0]).signum() as f32;
            r.iter_mut().zip(u.iter()).for_each(|(x, y)| *x += sign * y);
        }
        normalize(&mut r);
        let mut alphas: Vec<f32> = fits.iter().filter_map(|f| f.best_alpha).collect();
        alphas.sort_by(f32::total_cmp);
        vec![SiteOp::rank1(alphas[alphas.len() / 2], &r); fits.len()]
    } else {
        fits.iter().map(|f| match (&f.op, f.best_alpha) {
            (SiteOp::Steer { k: 1, b, .. }, Some(alpha)) => SiteOp::rank1(alpha, b),
            (op, _) => op.clone(),
        }).collect()
    };
    let mut reports = Vec::with_capacity(fits.len());
    for (fit, op) in fits.iter().zip(&served) {
        let exact = match op {
            SiteOp::Steer { k: 1, a, b } => {
                let p = left_mul(&fit.w.data, fit.w.cols, b);
                rebuild_exact(&fit.w.data, &fit.edit, b, &p, dot(a, b) as f32, 1)
            }
            op => rebuild_exact_op(&fit.w, &fit.edit, op),
        };
        let reason = if !fit.verify.accept { fit.verify.reason.clone() }
            else if exact < opts.rebuild_min { Some(format!("FP8 rebuild reproduces {exact:.4} of the bytes < {}", opts.rebuild_min)) }
            else { None };
        reports.push(V41SiteReport { layer: fit.layer, verify: fit.verify.clone(), alpha_rebuild: fit.best_alpha,
            rebuild_exact: exact, stock_exact: fit.stock, accept: reason.is_none(), reason });
    }
    let reason = reports.iter().find(|r| !r.accept)
        .map(|r| format!("layers.{}.attn.wo_b: {}", r.layer, r.reason.as_deref().unwrap_or("")));
    let ops = fits.iter().zip(&served).map(|(fit, op)| (SiteKey { layer: fit.layer, kind: SiteKind::Attn }, op.clone())).collect();
    let mut alpha: Vec<f32> = served.iter().map(|op| match op { SiteOp::Steer { a, b, .. } => dot(a, b) as f32, SiteOp::Delta {} => 0.0 }).collect();
    if shared { alpha.truncate(1); }
    Ok(V41Derivation { hidden: hidden as u32, alpha, shared_direction: shared, direction_cos_min: cos_min,
        accept: reason.is_none(), reason, sites: reports, unchanged, ops })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mem(Vec<(String, DType, Vec<usize>, Vec<u8>)>);
    impl TensorBytes for Mem {
        fn read(&self, name: &str) -> Result<(DType, Vec<usize>, Vec<u8>)> {
            let t = self.0.iter().find(|t| t.0 == name).context("missing")?;
            Ok((t.1.clone(), t.2.clone(), t.3.clone()))
        }
    }
    fn quantize(w: &[f32], rows: usize, cols: usize, scale: u8) -> Vec<u8> {
        let s = ue8m0_to_f32(scale);
        w.iter().take(rows * cols).map(|&x| f32_to_e4m3(x / s)).collect()
    }
    fn tensors(name: &str, w: &[f32], rows: usize, cols: usize, scale: u8) -> [(String, DType, Vec<usize>, Vec<u8>); 2] {
        [(format!("{name}.weight"), DType::F8E4M3, vec![rows, cols], quantize(w, rows, cols, scale)),
            (format!("{name}.scale"), DType::F8E8M0, vec![rows / 32, cols / 32], vec![scale; rows * cols / 1024])]
    }

    /// A synthetic overlay `q((I − 3.5 r rᵀ) W)` on three layers with one
    /// direction reproduces the recipe: rank 1, shared, α from the rebuild.
    #[test]
    fn synthetic_rank1_overlay_derives_one_shared_direction() -> Result<()> {
        let (rows, cols) = (64usize, 96usize);
        let mut r: Vec<f32> = (0..rows).map(|i| ((i * 37 % 17) as f32 - 8.0) / 8.0).collect();
        normalize(&mut r);
        let mut base = Vec::new();
        let mut edit = Vec::new();
        for layer in [10u16, 11, 12] {
            let w: Vec<f32> = (0..rows * cols).map(|i| ((i as f32 * 0.7 + f32::from(layer)).sin()) * 0.05).collect();
            let wq: Vec<f32> = quantize(&w, rows, cols, 120).iter().map(|&c| e4m3_to_f32(c) * ue8m0_to_f32(120)).collect();
            let p = left_mul(&wq, cols, &r);
            let edited: Vec<f32> = (0..rows * cols).map(|k| wq[k] - 3.5 * (r[k / cols] * p[k % cols])).collect();
            let name = format!("layers.{layer}.attn.wo_b");
            base.extend(tensors(&name, &wq, rows, cols, 120));
            edit.extend(tensors(&name, &edited, rows, cols, 120));
        }
        let dir = tempfile::tempdir()?;
        let file = dir.path().join("wo_b.safetensors");
        write_safetensors(&file, &edit)?;
        let source = V41VariantSource::open(dir.path(), 40)?;
        assert_eq!(source.layers, [10, 11, 12]);
        let derived = derive_v41_variant(&Mem(base), &source, rows, &V41VariantOpts::default())?;
        assert!(derived.accept, "{:?}", derived.reason);
        assert!(derived.shared_direction && derived.alpha.len() == 1, "{derived:?}");
        // A 64x96 toy pins α only to a few rebuild steps; the fixture test pins it on real tensors.
        assert!((derived.alpha[0] - 3.5).abs() < 0.06, "alpha {} {:?}", derived.alpha[0], derived.sites);
        assert_eq!(derived.ops.len(), 3);
        let SiteOp::Steer { k, b, .. } = &derived.ops[0].1 else { panic!() };
        assert_eq!(*k, 1);
        assert!(dot(b, &r).abs() > 0.999);
        assert!(derived.ops.iter().all(|(key, op)| key.kind == SiteKind::Attn && op == &derived.ops[0].1));
        // The shared library's site map (`cuteafd plan --variant`) reads the same sites.
        let config = serde_json::json!({"model_type": "deepseek_v41",
            "text_config": {"model_type": "deepseek_v41_text", "hidden_size": rows, "num_hidden_layers": 40}});
        let family = crate::variants::family_variants(&config, None)?;
        assert_eq!((family.family, family.hidden), ("deepseek_v41", rows));
        let sites = family.sites(&TensorSource::from_files([file])?);
        assert_eq!(sites.iter().map(|s| s.key.layer).collect::<Vec<_>>(), [10, 11, 12]);
        assert!(crate::variants::family_variants(&config, Some("ffn")).is_err(), "V4.1 has no modes");
        Ok(())
    }

    #[test]
    fn source_refuses_other_tensors_and_ambiguous_directories() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let w = vec![0.01f32; 32 * 32];
        let mut bad = tensors("layers.3.attn.wo_b", &w, 32, 32, 127).to_vec();
        bad.push(("layers.3.ffn.experts.0.w2.weight".into(), DType::F8E4M3, vec![32, 32], vec![0; 1024]));
        write_safetensors(&dir.path().join("a.safetensors"), &bad)?;
        assert!(V41VariantSource::open(&dir.path().join("a.safetensors"), 40).is_err());
        let missing_scale = vec![tensors("layers.3.attn.wo_b", &w, 32, 32, 127)[0].clone()];
        write_safetensors(&dir.path().join("a.safetensors"), &missing_scale)?;
        assert!(V41VariantSource::open(&dir.path().join("a.safetensors"), 40).is_err());
        let past = tensors("layers.40.attn.wo_b", &w, 32, 32, 127);
        write_safetensors(&dir.path().join("a.safetensors"), &past)?;
        assert!(V41VariantSource::open(&dir.path().join("a.safetensors"), 40).is_err());
        // An EXL3 sidecar beside it is skipped; two FP8 files are ambiguous.
        let good = tensors("layers.3.attn.wo_b", &w, 32, 32, 127);
        write_safetensors(&dir.path().join("a.safetensors"), &good)?;
        write_safetensors(&dir.path().join("exl3.safetensors"),
            &[("layers.3.attn.wo_b.suh".into(), DType::F16, vec![32], vec![0; 64])])?;
        assert_eq!(V41VariantSource::open(dir.path(), 40)?.layers, [3]);
        write_safetensors(&dir.path().join("b.safetensors"), &good)?;
        assert!(V41VariantSource::open(dir.path(), 40).is_err());
        Ok(())
    }

    fn write_safetensors(path: &Path, tensors: &[(String, DType, Vec<usize>, Vec<u8>)]) -> Result<()> {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, dtype, shape, bytes) in tensors {
            let dtype = match dtype { DType::F8E4M3 => "F8_E4M3", DType::F8E8M0 => "F8_E8M0", DType::F16 => "F16", _ => unreachable!() };
            header.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes.len()]}));
            offset += bytes.len();
        }
        let header = serde_json::to_vec(&header)?;
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend(header);
        for (.., bytes) in tensors { out.extend(bytes); }
        Ok(std::fs::write(path, out)?)
    }

    /// Fixture-backed: the published alternate-weight 1017 overlay against the stock `wo_b`
    /// bytes reproduces the pilot (rank 1, one direction, α ≈ 3.3–3.5, accept).
    /// Needs the fixtures on pg; run with
    /// `cargo test -p cuteafd-loader --release alternate_1017 -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads 1.1 GB of local fixtures"]
    fn alternate_1017_overlay_reproduces_the_pilot() -> Result<()> {
        let stock = std::env::var("V41_VARIANT_STOCK")
            .unwrap_or_else(|_| shellexpand("~/.cache/weight-variants/dsf1017-wo_b-stock"));
        let overlay = std::env::var("V41_VARIANT_OVERLAY").unwrap_or_else(|_| shellexpand(
            "~/.cache/huggingface/hub/models--example-org--DeepSeek-V4.1-Flash-1017-alternate-weights/snapshots/de412d2249632903dace23399526e00a145aada5"));
        let base = BinFixture::open(Path::new(&stock))?;
        let source = V41VariantSource::open(Path::new(&overlay), 40)?;
        assert_eq!(source.layers, (10..=35).collect::<Vec<u16>>(), "sites come from the overlay's own tensors");
        let started = std::time::Instant::now();
        let derived = derive_v41_variant(&base, &source, 5120, &V41VariantOpts::default())?;
        eprintln!("{}", serde_json::to_string_pretty(&derived)?);
        eprintln!("derived {} sites in {:.1}s", derived.sites.len(), started.elapsed().as_secs_f32());
        assert!(derived.accept, "{:?}", derived.reason);
        assert!(derived.unchanged.is_empty());
        assert!(derived.shared_direction && derived.direction_cos_min > 0.999);
        assert_eq!(derived.alpha.len(), 1);
        assert!((3.3..=3.55).contains(&derived.alpha[0]), "alpha {}", derived.alpha[0]);
        for site in &derived.sites {
            let v = &site.verify;
            assert!(v.k == 1 && (0.84..=0.92).contains(&v.top_share) && v.next_share < 0.005, "{site:?}");
            assert!((3.25..=3.45).contains(&v.alphas[0]) && v.proj_cos > 0.985, "{site:?}");
            assert!(site.rebuild_exact >= 0.8 && site.rebuild_exact > site.stock_exact + 0.2, "{site:?}");
        }
        Ok(())
    }

    fn shellexpand(path: &str) -> String {
        path.replacen('~', &std::env::var("HOME").unwrap_or_default(), 1)
    }
    /// The pilot's stock fixture: `<name>.bin` raw bytes plus `meta.json`.
    struct BinFixture { dir: PathBuf, meta: serde_json::Value }
    impl BinFixture {
        fn open(dir: &Path) -> Result<Self> {
            Ok(Self { dir: dir.into(), meta: serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)? })
        }
    }
    impl TensorBytes for BinFixture {
        fn read(&self, name: &str) -> Result<(DType, Vec<usize>, Vec<u8>)> {
            let meta = &self.meta[name];
            let dtype = DType::from_safetensors(meta["dtype"].as_str().context("fixture dtype")?);
            let shape = serde_json::from_value(meta["shape"].clone())?;
            Ok((dtype, shape, std::fs::read(self.dir.join(format!("{name}.bin")))?))
        }
    }
}
