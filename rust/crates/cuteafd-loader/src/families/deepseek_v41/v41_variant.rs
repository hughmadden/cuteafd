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
use crate::variants::{derive_site, e4m3_to_f32, f32_to_e4m3, ue8m0_to_f32, DeriveOpts, Matrix, SiteTensor, SiteVerdict, TensorSource,
    VariantOutcome, VerifyReport};
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
    /// Rows the α search and the rank-1 byte check read (every `stride`-th row).
    pub alpha_row_stride: usize,
    /// Sites sharing a direction share one α only when every site's rebuilt α is within this
    /// fraction of their median. The FP8 noise band, measured on the alternate-weight 1017 overlay
    /// (2026-10-24): one published λ = 3.5 for all 26 layers, rebuilt α 3.50–3.54 per layer
    /// (median 3.52, largest deviation 0.6%), so 2% leaves 3× margin and still separates
    /// strengths a few percent apart. Otherwise each site keeps its own α.
    pub alpha_band: f32,
    /// The served op's residual along its own direction(s), relative to the edit along them:
    /// `‖Bᵀ(W′ − (W − A BᵀW))‖ / ‖Bᵀ(W′ − W)‖`. A wrong strength adds about |Δα|/α and fails
    /// this even when the whole-matrix byte fraction passes (a sparse direction changes few
    /// bytes). Its floor is the direction estimate's angle: alternate-weight 1017 measures 0.09–0.15 per
    /// layer (2026-10-24) with the strength right, so 0.3 keeps 2× margin; the α band above
    /// is the finer strength check.
    pub r_gap_max: f32,
}
impl Default for V41VariantOpts {
    fn default() -> Self {
        Self { derive: variant_opts(), rebuild_min: 0.8, shared_cos_min: 0.999, alpha_span: 0.4, alpha_step: 0.02,
            alpha_row_stride: 8, alpha_band: 0.02, r_gap_max: 0.3 }
    }
}

/// Measured facts about one edited tensor.
#[derive(Debug, Clone, Serialize)]
pub struct V41SiteReport {
    pub layer: u16,
    /// The shared library's fit (least-squares α).
    pub verify: VerifyReport,
    /// α the FP8 rebuild prefers along the serving direction (rank 1).
    pub alpha_rebuild: Option<f32>,
    /// α actually served (rank 1).
    pub alpha_served: Option<f32>,
    /// FP8 bytes equal after `q(W − A BᵀW)` with the served op, and for stock W.
    pub rebuild_exact: f32,
    /// The served op's residual along its direction(s), relative to the edit there ([`V41VariantOpts::r_gap_max`]).
    pub r_gap: f32,
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
    /// Every site serves one α (the rebuilt α agree within [`V41VariantOpts::alpha_band`]).
    pub shared_alpha: bool,
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

/// `‖t − s + α s‖ / ‖t − s‖` for `s = rᵀW`, `t = rᵀW′` (unit r): the rank-1 op's residual along r.
fn r_gap(s: &[f32], t: &[f32], alpha: f32) -> f32 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&s, &t) in s.iter().zip(t) {
        let edit = f64::from(t) - f64::from(s);
        num += (edit + f64::from(alpha) * f64::from(s)).powi(2);
        den += edit.powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt() as f32
}
/// The same for a rank-k op: `‖Bᵀ(W′ − W + A BᵀW)‖ / ‖Bᵀ(W′ − W)‖`.
fn r_gap_op(w: &Matrix, v: &Matrix, op: &SiteOp) -> f32 {
    let SiteOp::Steer { k, a, b } = op else { return f32::INFINITY };
    let (h, k) = (w.rows, usize::from(*k));
    let s: Vec<Vec<f32>> = (0..k).map(|j| left_mul(&w.data, w.cols, &b[j * h..][..h])).collect();
    let t: Vec<Vec<f32>> = (0..k).map(|j| left_mul(&v.data, v.cols, &b[j * h..][..h])).collect();
    let (mut num, mut den) = (0f64, 0f64);
    for j in 0..k {
        let g: Vec<f64> = (0..k).map(|i| dot(&b[j * h..][..h], &a[i * h..][..h])).collect();
        for c in 0..w.cols {
            let edit = f64::from(t[j][c]) - f64::from(s[j][c]);
            let applied: f64 = (0..k).map(|i| g[i] * f64::from(s[i][c])).sum();
            num += (edit + applied).powi(2);
            den += edit.powi(2);
        }
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt() as f32
}

/// The rebuild curve: (α, fraction of the variant's bytes reproduced) on the α grid around `start`.
fn rebuild_curve(w: &[f32], variant: &Fp8Block32, u: &[f32], p: &[f32], start: f32, opts: &V41VariantOpts) -> Vec<(f32, f32)> {
    let steps = (opts.alpha_span / opts.alpha_step).round() as i32;
    let start = (start / opts.alpha_step).round();
    (-steps..=steps).map(|s| (start + s as f32) * opts.alpha_step)
        .map(|a| (a, rebuild_exact(w, variant, u, p, a, opts.alpha_row_stride))).collect()
}
/// The α the variant's FP8 bytes reproduce best (the middle of the best plateau:
/// neighbouring α often requantize to the same bytes).
fn best_alpha(curve: &[(f32, f32)]) -> f32 {
    let best = curve.iter().map(|s| s.1).fold(f32::MIN, f32::max);
    let plateau: Vec<f32> = curve.iter().filter(|s| s.1 == best).map(|s| s.0).collect();
    plateau[plateau.len() / 2]
}
/// The curve's byte fraction at grid point `alpha` (None off the grid).
fn curve_at(curve: &[(f32, f32)], alpha: f32, step: f32) -> Option<f32> {
    curve.iter().find(|(a, _)| (a - alpha).abs() < step / 4.0).map(|(_, f)| *f)
}

/// Derive the V4.1 variant: fit each edited `wo_b` against the served base
/// with the shared library, then apply the recipe and check every site with
/// the op that will be served. A rejected site makes the variant unavailable
/// (`accept == false`) with the reason; base serving is unaffected.
///
/// Recipe: sites whose rank-1 directions all agree share their sign-aligned
/// mean direction. Along the serving direction each site's α is the FP8 rebuild
/// optimum; the sites share one α only when those agree within
/// [`V41VariantOpts::alpha_band`], else each keeps its own (equal tables still
/// dedupe on the device). Every site must reproduce the variant's bytes
/// ([`V41VariantOpts::rebuild_min`]) and its edit along the served direction
/// ([`V41VariantOpts::r_gap_max`]). Two passes, one layer in memory at a time.
pub fn derive_v41_variant(base: &dyn TensorBytes, variant: &V41VariantSource, hidden: usize,
    opts: &V41VariantOpts) -> Result<V41Derivation> {
    let read = |layer: u16| -> Result<(Matrix, Matrix, Fp8Block32)> {
        let name = format!("layers.{layer}.attn.wo_b");
        let stock = read_fp8_block32(base, &name).with_context(|| format!("base {name}"))?;
        let edit = read_fp8_block32(variant, &name).with_context(|| format!("variant {name}"))?;
        ensure!(stock.rows == hidden && edit.rows == stock.rows && edit.cols == stock.cols,
            "{name}: variant {}x{} differs from base {}x{} (hidden {hidden})", edit.rows, edit.cols, stock.rows, stock.cols);
        Ok((Matrix::new(stock.rows, stock.cols, stock.dequantize()), Matrix::new(edit.rows, edit.cols, edit.dequantize()), edit))
    };
    // Pass 1: the shared-library fit per site.
    struct Fit { layer: u16, op: SiteOp, verify: VerifyReport, stock: f32 }
    let mut fits = Vec::with_capacity(variant.layers.len());
    let mut unchanged = Vec::new();
    for &layer in &variant.layers {
        let (w, v, edit) = read(layer)?;
        let (op, verify) = match derive_site(&w, &v, &opts.derive) {
            Ok(fit) => fit,
            Err(VariantError::Unchanged) => { unchanged.push(layer); continue; }
            Err(error) => bail!("layers.{layer}.attn.wo_b: {error}"),
        };
        let stock = rebuild_exact(&w.data, &edit, &vec![0.0; w.rows], &vec![0.0; w.cols], 0.0, 1);
        tracing::info!(layer, k = verify.k, rel_delta = verify.rel_delta, top = verify.top_share, next = verify.next_share,
            alpha_ls = ?verify.alphas, cos = verify.proj_cos, "V4.1 variant site fit");
        fits.push(Fit { layer, op, verify, stock });
    }
    ensure!(!fits.is_empty(), "variant {}: every wo_b tensor is unchanged against the base", variant.file.display());
    // The serving direction: one sign-aligned mean when every site is rank 1 and they agree.
    let directions: Vec<&[f32]> = fits.iter().filter_map(|f| match &f.op { SiteOp::Steer { k: 1, b, .. } => Some(b.as_slice()), _ => None }).collect();
    let mut cos_min = 1.0f32;
    for (i, a) in directions.iter().enumerate() {
        for b in &directions[i + 1..] { cos_min = cos_min.min(dot(a, b).abs() as f32); }
    }
    let shared = directions.len() == fits.len() && fits.len() > 1 && cos_min >= opts.shared_cos_min;
    let mean = shared.then(|| {
        let mut r = vec![0f32; hidden];
        for u in &directions {
            let sign = dot(u, directions[0]).signum() as f32;
            r.iter_mut().zip(u.iter()).for_each(|(x, y)| *x += sign * y);
        }
        normalize(&mut r);
        r
    });
    // Pass 2: along each site's serving direction, the rebuild curve and the edit's r-components.
    enum Along { Rank1 { r: Vec<f32>, curve: Vec<(f32, f32)>, s: Vec<f32>, t: Vec<f32> }, General { exact: f32, gap: f32 } }
    let mut along = Vec::with_capacity(fits.len());
    for fit in &fits {
        let (w, v, edit) = read(fit.layer)?;
        along.push(match &fit.op {
            SiteOp::Steer { k: 1, b, .. } => {
                let r = mean.clone().unwrap_or_else(|| b.clone());
                let (s, t) = (left_mul(&w.data, w.cols, &r), left_mul(&v.data, v.cols, &r));
                // Least squares along r starts the grid: α = −(t − s)·s / s·s.
                let start = -(dot(&t, &s) - dot(&s, &s)) / dot(&s, &s);
                let curve = rebuild_curve(&w.data, &edit, &r, &s, start as f32, opts);
                Along::Rank1 { r, curve, s, t }
            }
            op => Along::General { exact: rebuild_exact_op(&w, &edit, op), gap: r_gap_op(&w, &v, op) },
        });
    }
    let rebuilt: Vec<Option<f32>> = along.iter().map(|a| match a { Along::Rank1 { curve, .. } => Some(best_alpha(curve)), _ => None }).collect();
    let mut sorted: Vec<f32> = rebuilt.iter().flatten().copied().collect();
    sorted.sort_by(f32::total_cmp);
    let median = sorted.get(sorted.len() / 2).copied();
    let shared_alpha = shared && median.is_some_and(|m| sorted.iter().all(|a| (a - m).abs() <= opts.alpha_band * m.abs()));
    let mut reports = Vec::with_capacity(fits.len());
    let mut ops = Vec::with_capacity(fits.len());
    for ((fit, along), rebuilt) in fits.iter().zip(&along).zip(&rebuilt) {
        let (op, served, exact, gap) = match along {
            Along::Rank1 { r, curve, s, t } => {
                let alpha = if shared_alpha { median.unwrap() } else { rebuilt.unwrap() };
                let exact = match curve_at(curve, alpha, opts.alpha_step) {
                    Some(exact) => exact,
                    None => { let (w, _, edit) = read(fit.layer)?; rebuild_exact(&w.data, &edit, r, s, alpha, opts.alpha_row_stride) }
                };
                (SiteOp::rank1(alpha, r), Some(alpha), exact, r_gap(s, t, alpha))
            }
            Along::General { exact, gap } => (fit.op.clone(), None, *exact, *gap),
        };
        let reason = if !fit.verify.accept { fit.verify.reason.clone() }
            else if exact < opts.rebuild_min { Some(format!("FP8 rebuild reproduces {exact:.4} of the bytes < {}", opts.rebuild_min)) }
            else if gap > opts.r_gap_max { Some(format!("the served op misses the edit along its direction by {gap:.4} > {}", opts.r_gap_max)) }
            else { None };
        reports.push(V41SiteReport { layer: fit.layer, verify: fit.verify.clone(), alpha_rebuild: *rebuilt, alpha_served: served,
            rebuild_exact: exact, r_gap: gap, stock_exact: fit.stock, accept: reason.is_none(), reason });
        ops.push((SiteKey { layer: fit.layer, kind: SiteKind::Attn }, op));
    }
    let reason = reports.iter().find(|r| !r.accept)
        .map(|r| format!("layers.{}.attn.wo_b: {}", r.layer, r.reason.as_deref().unwrap_or("")));
    let mut alpha: Vec<f32> = ops.iter().map(|(_, op)| match op { SiteOp::Steer { a, b, .. } => dot(a, b) as f32, SiteOp::Delta { .. } => 0.0 }).collect();
    if shared_alpha { alpha.truncate(1); }
    Ok(V41Derivation { hidden: hidden as u32, alpha, shared_direction: shared, shared_alpha, direction_cos_min: cos_min,
        accept: reason.is_none(), reason, sites: reports, unchanged, ops })
}

impl TensorBytes for TensorSource {
    fn read(&self, name: &str) -> Result<(DType, Vec<usize>, Vec<u8>)> {
        let (bytes, meta) = self.bytes(name)?;
        Ok((meta.dtype.clone(), meta.shape.clone(), bytes))
    }
}

/// One `--variant` derived the V4.1 way, as the shared [`VariantOutcome`]: `cuteafd plan
/// --variant` (through [`crate::variants::derive_variant`]) and serving both call this, so
/// what plan prints is what serves. The base is the served snapshot or `base=`; an edited
/// tensor the base lacks makes the variant unavailable.
pub fn derive_v41_outcome(served: &Path, served_id: &str, index: u8, spec: &cuteafd_core::variants::VariantSpec)
    -> Result<(VariantOutcome, Option<V41Derivation>)> {
    let started = std::time::Instant::now();
    ensure!(spec.mode.is_none(), "variant {}: DeepSeek V4.1 has no variant modes (got mode={})",
        spec.name, spec.mode.as_deref().unwrap_or_default());
    let config = crate::plan::checkpoint::read_json(&served.join("config.json"))?;
    let text = config.get("text_config").unwrap_or(&config);
    let int = |key: &str| text[key].as_u64().map(|v| v as usize).with_context(|| format!("config.json {key}"));
    let (hidden, layers) = (int("hidden_size")?, int("num_hidden_layers")?);
    let source = V41VariantSource::open(Path::new(&spec.source), u16::try_from(layers)?)
        .with_context(|| format!("variant {}", spec.name))?;
    let base_path = spec.base.as_deref().map_or_else(|| served.to_path_buf(), PathBuf::from);
    let base = crate::variants::open_source(&base_path, "variant-base", wants_tensor)?;
    let public_id = spec.public_id(served_id);
    let outcome = |plan, rejection, verdicts, missing_in_base| VariantOutcome { name: spec.name.clone(), public_id: public_id.clone(),
        plan, rejection, verdicts, missing_in_base, seconds: started.elapsed().as_secs_f64() };
    let missing: Vec<String> = source.layers.iter().flat_map(|l| [format!("layers.{l}.attn.wo_b.weight"), format!("layers.{l}.attn.wo_b.scale")])
        .filter(|t| !base.contains(t)).collect();
    if let Some(first) = missing.first() {
        let reason = format!("the base lacks {} edited tensor(s), first {first}", missing.len());
        return Ok((outcome(None, Some(reason), Vec::new(), missing), None));
    }
    let derivation = derive_v41_variant(&base, &source, hidden, &V41VariantOpts::default())?;
    let mut verdicts: Vec<SiteVerdict> = derivation.sites.iter().map(|s| SiteVerdict {
        site: SiteKey { layer: s.layer, kind: SiteKind::Attn }.to_string(), tensor: format!("layers.{}.attn.wo_b.weight", s.layer),
        report: Some(VerifyReport { accept: s.accept, reason: s.reason.clone(), ..s.verify.clone() }) }).collect();
    verdicts.extend(derivation.unchanged.iter().map(|&l| SiteVerdict { site: SiteKey { layer: l, kind: SiteKind::Attn }.to_string(),
        tensor: format!("layers.{l}.attn.wo_b.weight"), report: None }));
    let hidden = u32::try_from(hidden)?;
    let plan = derivation.accept.then(|| cuteafd_core::variants::VariantPlan::new(index, spec.name.clone(), public_id.clone(),
        hidden, derivation.ops.clone())).transpose()?;
    Ok((outcome(plan, derivation.reason.clone(), verdicts, Vec::new()), Some(derivation)))
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

    /// A synthetic overlay `q((I − α_l r rᵀ) W_l)` per (layer, α_l), FP8 with one block scale.
    fn overlay(r: &[f32], cols: usize, layers: &[(u16, f32)]) -> Result<(Mem, tempfile::TempDir, PathBuf)> {
        let rows = r.len();
        let mut base = Vec::new();
        let mut edit = Vec::new();
        for &(layer, alpha) in layers {
            let w: Vec<f32> = (0..rows * cols).map(|i| ((i as f32 * 0.7 + f32::from(layer)).sin()) * 0.05).collect();
            let wq: Vec<f32> = quantize(&w, rows, cols, 120).iter().map(|&c| e4m3_to_f32(c) * ue8m0_to_f32(120)).collect();
            let p = left_mul(&wq, cols, r);
            let edited: Vec<f32> = (0..rows * cols).map(|k| wq[k] - alpha * (r[k / cols] * p[k % cols])).collect();
            let name = format!("layers.{layer}.attn.wo_b");
            base.extend(tensors(&name, &wq, rows, cols, 120));
            edit.extend(tensors(&name, &edited, rows, cols, 120));
        }
        let dir = tempfile::tempdir()?;
        let file = dir.path().join("wo_b.safetensors");
        write_safetensors(&file, &edit)?;
        Ok((Mem(base), dir, file))
    }

    /// Astra's case: one direction (r = e0, a single output row) at strengths 1 and 3. The
    /// direction is shared, the strengths are not; a forced shared α would still reproduce
    /// most FP8 bytes (only row 0 changes) but misses the edit along r, and is refused.
    #[test]
    fn same_direction_different_strengths_keep_their_own_alpha() -> Result<()> {
        let (rows, cols) = (64usize, 96usize);
        let r: Vec<f32> = (0..rows).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let (base, dir, _) = overlay(&r, cols, &[(10, 1.0), (11, 3.0)])?;
        let source = V41VariantSource::open(dir.path(), 40)?;
        let derived = derive_v41_variant(&base, &source, rows, &V41VariantOpts::default())?;
        assert!(derived.accept, "{:?}", derived.reason);
        assert!(derived.shared_direction && !derived.shared_alpha, "{derived:?}");
        assert_eq!(derived.alpha.len(), 2);
        assert!((derived.alpha[0] - 1.0).abs() < 0.06 && (derived.alpha[1] - 3.0).abs() < 0.06, "{:?}", derived.alpha);
        assert_ne!(derived.ops[0].1, derived.ops[1].1, "different strengths are different tables");
        assert!(derived.sites.iter().all(|s| s.r_gap < 0.05), "{:?}", derived.sites);
        // Forcing one α: the byte fraction alone would pass, the r-component check does not.
        let forced = V41VariantOpts { alpha_band: f32::INFINITY, ..V41VariantOpts::default() };
        let derived = derive_v41_variant(&base, &source, rows, &forced)?;
        assert!(derived.shared_alpha && !derived.accept, "{derived:?}");
        let wrong = derived.sites.iter().find(|s| !s.accept).unwrap();
        assert!(wrong.rebuild_exact >= forced.rebuild_min, "the byte check alone passes: {wrong:?}");
        assert!(wrong.reason.as_deref().unwrap().contains("along its direction"), "{wrong:?}");
        Ok(())
    }

    /// `cuteafd plan --variant` (the shared `derive_variant`) and serving (`derive_v41_outcome`)
    /// agree on availability, rejection reason, operators and namespace.
    #[test]
    fn plan_and_serving_derive_the_same_variant() -> Result<()> {
        let (rows, cols) = (64usize, 96usize);
        let r: Vec<f32> = (0..rows).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let cases: [(&str, &[(u16, f32)], Option<u16>, bool); 3] =
            [("shared", &[(10, 3.5), (11, 3.5)], None, true), ("strengths", &[(10, 1.0), (11, 3.0)], None, true),
             ("missing", &[(10, 3.5), (11, 3.5)], Some(11), false)];
        for (case, layers, drop, available) in cases {
            let (Mem(base), _vdir, file) = overlay(&r, cols, layers)?;
            let served = tempfile::tempdir()?;
            std::fs::write(served.path().join("config.json"), serde_json::to_vec(&serde_json::json!({"model_type": "deepseek_v41",
                "text_config": {"model_type": "deepseek_v41_text", "hidden_size": rows, "num_hidden_layers": 40}}))?)?;
            let kept: Vec<_> = base.into_iter().filter(|t| drop.is_none_or(|l| !t.0.starts_with(&format!("layers.{l}.")))).collect();
            write_safetensors(&served.path().join("base.safetensors"), &kept)?;
            let spec: cuteafd_core::variants::VariantSpec = format!("name=ab,source={}", file.display()).parse()?;
            let plan = crate::variants::derive_variant(served.path(), "m", 1, &spec, 1)?;
            let (serve, derivation) = derive_v41_outcome(served.path(), "m", 1, &spec)?;
            assert_eq!((plan.plan.is_some(), serve.plan.is_some()), (available, available), "{case}: {:?}", serve.rejection);
            assert_eq!(plan.rejection, serve.rejection, "{case}");
            assert_eq!(plan.plan.as_ref().map(|p| (&p.sites, p.namespace)), serve.plan.as_ref().map(|p| (&p.sites, p.namespace)), "{case}");
            assert_eq!(plan.public_id, "m-ab");
            assert_eq!(derivation.is_some(), drop.is_none(), "{case}");
            if case == "missing" { assert!(serve.rejection.as_deref().unwrap().contains("base lacks"), "{:?}", serve.rejection); }
        }
        Ok(())
    }

    /// A synthetic overlay `q((I − 3.5 r rᵀ) W)` on three layers with one
    /// direction reproduces the recipe: rank 1, shared direction and α, α from the rebuild.
    #[test]
    fn synthetic_rank1_overlay_derives_one_shared_direction() -> Result<()> {
        let (rows, cols) = (64usize, 96usize);
        let mut r: Vec<f32> = (0..rows).map(|i| ((i * 37 % 17) as f32 - 8.0) / 8.0).collect();
        normalize(&mut r);
        let (base, dir, file) = overlay(&r, cols, &[(10, 3.5), (11, 3.5), (12, 3.5)])?;
        let source = V41VariantSource::open(dir.path(), 40)?;
        assert_eq!(source.layers, [10, 11, 12]);
        let derived = derive_v41_variant(&base, &source, rows, &V41VariantOpts::default())?;
        assert!(derived.accept, "{:?}", derived.reason);
        assert!(derived.shared_direction && derived.shared_alpha && derived.alpha.len() == 1, "{derived:?}");
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
        assert!(derived.shared_direction && derived.shared_alpha && derived.direction_cos_min > 0.999);
        assert_eq!(derived.alpha.len(), 1);
        assert!((3.3..=3.55).contains(&derived.alpha[0]), "alpha {}", derived.alpha[0]);
        for site in &derived.sites {
            let v = &site.verify;
            assert!(v.k == 1 && (0.84..=0.92).contains(&v.top_share) && v.next_share < 0.005, "{site:?}");
            assert!((3.25..=3.45).contains(&v.alphas[0]) && v.proj_cos > 0.985, "{site:?}");
            assert!(site.rebuild_exact >= 0.8 && site.rebuild_exact > site.stock_exact + 0.2, "{site:?}");
            assert!(site.r_gap <= 0.2, "{site:?}");
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
