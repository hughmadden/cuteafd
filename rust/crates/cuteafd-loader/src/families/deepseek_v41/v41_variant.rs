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
//! V4.1 recipe (measured on the Keys 1017 overlay, 2026-10-24): every edit is
//! a rank-1 output projection, and all 26 layers share one direction. α is
//! taken from the bit-exact FP8 rebuild against the variant's own scales; the
//! least-squares α is biased low by FP8 requantization noise in ΔW.
use anyhow::{bail, ensure, Context, Result};
use cuteafd_core::variants::{SiteKey, SiteKind, SiteOp};
use cuteafd_core::DType;
use rayon::prelude::*;
use serde::Serialize;
use std::path::{Path, PathBuf};

/// UE8M0 scale block edge of the official FP8 `wo_b`.
pub const V41_FP8_BLOCK: usize = 32;

/// FP8 E4M3 (fn) value of `code`; 0x7f/0xff are NaN.
pub fn e4m3_to_f32(code: u8) -> f32 { E4M3[usize::from(code & 0x7f)].copysign(if code & 0x80 != 0 { -1.0 } else { 1.0 }) }

static E4M3: std::sync::LazyLock<[f32; 128]> = std::sync::LazyLock::new(|| {
    std::array::from_fn(|code| {
        let (e, m) = ((code >> 3) as i32, (code & 7) as f32);
        if code == 0x7f { f32::NAN } else if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) }
    })
});

/// Round to the nearest E4M3 (ties to even), saturating at ±448, keeping the
/// sign of zero: the same bytes as torch's `clamp(-448, 448).to(float8_e4m3fn)`.
pub fn f32_to_e4m3(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let a = f64::from(value.abs().min(448.0));
    // Largest finite code whose value is <= a (codes are monotone in value).
    let (mut lo, mut hi) = (0usize, 126usize);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if f64::from(E4M3[mid]) <= a { lo = mid } else { hi = mid - 1 }
    }
    let code = if lo == 126 { 126 } else {
        let half = (f64::from(E4M3[lo]) + f64::from(E4M3[lo + 1])) / 2.0;
        if a > half || (a == half && lo % 2 == 1) { lo + 1 } else { lo }
    };
    sign | code as u8
}

/// UE8M0 scale value `2^(code − 127)`.
pub fn e8m0_to_f32(code: u8) -> f32 { if code == 0xff { f32::NAN } else { 2f32.powi(i32::from(code) - 127) } }

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
        e8m0_to_f32(self.scale[(row / V41_FP8_BLOCK) * (self.cols / V41_FP8_BLOCK) + col / V41_FP8_BLOCK])
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

impl V41VariantSource {
    /// Open a sidecar file, or the one file of FP8 `wo_b` tensors in a snapshot
    /// directory (other formats, such as EXL3 sidecars, are skipped). Every
    /// tensor in the chosen file must be an FP8 `wo_b` weight or its scale of a
    /// backbone layer below `layers`.
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

/// Per-recipe accept bars and search for the V4.1 FP8 `wo_b` edit.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct V41VariantOpts {
    /// Second singular value's share of ‖ΔW‖² (measured 0.003–0.004).
    pub second_share_max: f32,
    /// Fraction of the variant's FP8 bytes reproduced by `q(W − α r rᵀW)` (measured 0.84–0.86).
    pub rebuild_min: f32,
    /// Projection form `|cos(rᵀΔW, rᵀW)|` (measured 0.989–0.997).
    pub projection_cos_min: f32,
    /// Sites whose directions all agree this well (|cos|) share one.
    pub shared_cos_min: f32,
    /// α search for the rebuild: least-squares α ± `alpha_span`, step `alpha_step`.
    pub alpha_span: f32,
    pub alpha_step: f32,
    /// Rows sampled by the α search (every `stride`-th row); the final check reads all rows.
    pub alpha_row_stride: usize,
}
impl Default for V41VariantOpts {
    fn default() -> Self {
        Self { second_share_max: 0.005, rebuild_min: 0.8, projection_cos_min: 0.98, shared_cos_min: 0.999,
            alpha_span: 0.4, alpha_step: 0.02, alpha_row_stride: 8 }
    }
}

/// Measured facts about one edited tensor.
#[derive(Debug, Clone, Serialize)]
pub struct V41SiteReport {
    pub layer: u16,
    /// ‖ΔW‖ / ‖W‖.
    pub rel_delta: f32,
    /// Top two singular values' shares of ‖ΔW‖².
    pub top_share: f32,
    pub second_share: f32,
    /// Least-squares α for this site's own direction.
    pub alpha_ls: f32,
    pub projection_cos: f32,
    /// ‖ΔW − (−α u uᵀW)‖ / ‖ΔW‖ and / ‖W‖ at the least-squares α.
    pub r_delta: f32,
    pub r_w: f32,
    /// FP8 bytes equal after `q(W − α r rᵀW)` with the served op, and for stock W.
    pub rebuild_exact: f32,
    pub stock_exact: f32,
    pub accept: bool,
    pub reason: Option<String>,
}

/// A derived V4.1 variant: one steering op per edited site.
#[derive(Debug, Clone, Serialize)]
pub struct V41Derivation {
    pub hidden: u32,
    /// Served α and whether every site uses one shared direction.
    pub alpha: Vec<f32>,
    pub shared_direction: bool,
    /// Smallest pairwise |cos| between the sites' directions.
    pub direction_cos_min: f32,
    pub sites: Vec<V41SiteReport>,
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
/// `M v`.
fn right_mul(m: &[f32], cols: usize, v: &[f32]) -> Vec<f32> {
    m.par_chunks(cols).map(|row| dot(row, v) as f32).collect()
}

/// Rank-1 fit of `D = Wv − W` (H×K): top left singular vector `u` of D by
/// power iteration on `D Dᵀ`, the top two singular shares, α by least squares
/// and the projection-form checks (the pilot's `rank1.py` measures).
struct Rank1 { u: Vec<f32>, p: Vec<f32>, alpha: f32, top: f32, second: f32, cos: f32, r_delta: f32, r_w: f32, rel: f32 }
fn rank1(w: &[f32], d: &[f32], rows: usize, cols: usize) -> Rank1 {
    let energy: f64 = d.par_iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let w_norm: f64 = w.par_iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    let power = |deflate: Option<(&[f32], f64)>| -> (Vec<f32>, f64) {
        let mut u: Vec<f32> = (0..rows).map(|i| 1.0 + ((i * 7919) % 101) as f32 / 101.0).collect();
        if let Some((v, _)) = deflate { let c = dot(&u, v) as f32; u.iter_mut().zip(v).for_each(|(x, y)| *x -= c * y); }
        normalize(&mut u);
        let mut sigma2 = 0.0;
        for _ in 0..200 {
            let mut next = right_mul(d, cols, &left_mul(d, cols, &u));
            if let Some((v, s2)) = deflate {
                let c = (s2 * dot(v, &u)) as f32;
                next.iter_mut().zip(v).for_each(|(x, y)| *x -= c * y);
            }
            let estimate = dot(&next, &u);
            normalize(&mut next);
            u = next;
            if (estimate - sigma2).abs() <= 1e-7 * estimate.abs() { sigma2 = estimate; break; }
            sigma2 = estimate;
        }
        (u, sigma2)
    };
    let (u, s1) = power(None);
    let (_, s2) = power(Some((&u, s1)));
    let p = left_mul(w, cols, &u);
    let q = left_mul(d, cols, &u);
    let (qp, pp, qq) = (dot(&q, &p), dot(&p, &p), dot(&q, &q));
    let alpha = -(qp / pp);
    // ‖D + α u pᵀ‖² = ‖D‖² + 2α qᵀp + α² ‖p‖² (u is unit).
    let residual = (energy + 2.0 * alpha * qp + alpha * alpha * pp).max(0.0).sqrt();
    Rank1 { u, alpha: alpha as f32, top: (s1 / energy) as f32, second: (s2 / energy) as f32,
        cos: (qp.abs() / (qq.sqrt() * pp.sqrt())) as f32, r_delta: (residual / energy.sqrt()) as f32,
        r_w: (residual / w_norm) as f32, rel: (energy.sqrt() / w_norm) as f32, p }
}

/// Fraction of `variant` bytes equal to `q(W − α u pᵀ)` with the variant's own
/// scales, over every `stride`-th row (`alpha = 0`: stock W requantized).
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

/// Derive the V4.1 variant: fit each edited `wo_b` against the served base,
/// detect a shared direction, choose α by FP8 rebuild, and check every site
/// with the op that will be served. A rejected site makes the variant
/// unavailable (`accept == false`) with the reason; base serving is unaffected.
pub fn derive_v41_variant(base: &dyn TensorBytes, variant: &V41VariantSource, hidden: usize,
    opts: &V41VariantOpts) -> Result<V41Derivation> {
    struct Fit { layer: u16, w: Vec<f32>, edit: Fp8Block32, fit: Rank1, stock: f32, best_alpha: f32 }
    let mut fits = Vec::with_capacity(variant.layers.len());
    for &layer in &variant.layers {
        let name = format!("layers.{layer}.attn.wo_b");
        let stock = read_fp8_block32(base, &name).with_context(|| format!("base {name}"))?;
        let edit = read_fp8_block32(variant, &name).with_context(|| format!("variant {name}"))?;
        ensure!(stock.rows == hidden && edit.rows == stock.rows && edit.cols == stock.cols,
            "{name}: variant {}x{} differs from base {}x{} (hidden {hidden})", edit.rows, edit.cols, stock.rows, stock.cols);
        let w = stock.dequantize();
        let mut d = edit.dequantize();
        d.par_iter_mut().zip(w.par_iter()).for_each(|(v, &b)| *v -= b);
        let fit = rank1(&w, &d, stock.rows, stock.cols);
        drop(d);
        let stock_exact = rebuild_exact(&w, &edit, &fit.u, &fit.p, 0.0, 1);
        let best_alpha = rebuild_alpha(&w, &edit, &fit.u, &fit.p, fit.alpha, opts);
        tracing::info!(layer, rel_delta = fit.rel, top = fit.top, second = fit.second, alpha_ls = fit.alpha,
            alpha_rebuild = best_alpha, cos = fit.cos, "V4.1 variant site fit");
        fits.push(Fit { layer, w, edit, fit, stock: stock_exact, best_alpha });
    }
    // One direction for every site when they all agree (sign-aligned mean).
    let first = fits[0].fit.u.clone();
    let mut cos_min = 1.0f32;
    for (i, a) in fits.iter().enumerate() {
        for b in &fits[i + 1..] { cos_min = cos_min.min(dot(&a.fit.u, &b.fit.u).abs() as f32); }
    }
    let shared = fits.len() > 1 && cos_min >= opts.shared_cos_min;
    let served: Vec<(Vec<f32>, f32)> = if shared {
        let mut r = vec![0f32; hidden];
        for fit in &fits {
            let sign = dot(&fit.fit.u, &first).signum() as f32;
            r.iter_mut().zip(&fit.fit.u).for_each(|(x, y)| *x += sign * y);
        }
        normalize(&mut r);
        let mut alphas: Vec<f32> = fits.iter().map(|f| f.best_alpha).collect();
        alphas.sort_by(f32::total_cmp);
        let alpha = alphas[alphas.len() / 2];
        vec![(r, alpha); fits.len()]
    } else {
        fits.iter().map(|f| (f.fit.u.clone(), f.best_alpha)).collect()
    };
    let mut reports = Vec::with_capacity(fits.len());
    for (fit, (r, alpha)) in fits.iter().zip(&served) {
        let p = left_mul(&fit.w, fit.edit.cols, r);
        let exact = rebuild_exact(&fit.w, &fit.edit, r, &p, *alpha, 1);
        let f = &fit.fit;
        let reason = if f.second > opts.second_share_max {
            Some(format!("second singular share {:.4} > {} (not rank 1)", f.second, opts.second_share_max))
        } else if f.cos < opts.projection_cos_min {
            Some(format!("projection |cos| {:.4} < {} (not an output projection)", f.cos, opts.projection_cos_min))
        } else if exact < opts.rebuild_min {
            Some(format!("FP8 rebuild reproduces {exact:.4} of the bytes < {}", opts.rebuild_min))
        } else { None };
        reports.push(V41SiteReport { layer: fit.layer, rel_delta: f.rel, top_share: f.top, second_share: f.second,
            alpha_ls: f.alpha, projection_cos: f.cos, r_delta: f.r_delta, r_w: f.r_w, rebuild_exact: exact,
            stock_exact: fit.stock, accept: reason.is_none(), reason });
    }
    let rejected = reports.iter().find(|r| !r.accept);
    let reason = rejected.map(|r| format!("layers.{}.attn.wo_b: {}", r.layer, r.reason.as_deref().unwrap_or("")));
    let ops = fits.iter().zip(&served)
        .map(|(fit, (r, alpha))| (SiteKey { layer: fit.layer, kind: SiteKind::Attn }, SiteOp::rank1(*alpha, r)))
        .collect();
    let mut alpha: Vec<f32> = served.iter().map(|(_, a)| *a).collect();
    if shared { alpha.truncate(1); }
    Ok(V41Derivation { hidden: hidden as u32, alpha, shared_direction: shared, direction_cos_min: cos_min,
        accept: reason.is_none(), reason, sites: reports, ops })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_round_trips_every_code_and_rounds_ties_to_even() {
        for code in 0u8..=255 {
            if code & 0x7f == 0x7f { continue; }
            assert_eq!(f32_to_e4m3(e4m3_to_f32(code)), code, "code {code:#x}");
        }
        assert_eq!(e4m3_to_f32(0x7e), 448.0);
        assert_eq!(f32_to_e4m3(1e9), 0x7e, "saturates");
        assert_eq!(f32_to_e4m3(-1e9), 0xfe);
        // 1.0 (0x38) and 1.125 (0x39): the tie 1.0625 goes to the even code.
        assert_eq!(f32_to_e4m3(1.0625), 0x38);
        // 1.125 (0x39) and 1.25 (0x3a): the tie 1.1875 goes to 0x3a.
        assert_eq!(f32_to_e4m3(1.1875), 0x3a);
        assert_eq!(f32_to_e4m3(1.07), 0x39);
        // Smallest subnormal 2^-9 and half of it.
        assert_eq!(f32_to_e4m3(2f32.powi(-9)), 0x01);
        assert_eq!(f32_to_e4m3(2f32.powi(-10)), 0x00, "tie to even zero");
        assert_eq!(f32_to_e4m3(-0.0), 0x80);
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(120), 2f32.powi(-7));
    }

    struct Mem(Vec<(String, DType, Vec<usize>, Vec<u8>)>);
    impl TensorBytes for Mem {
        fn read(&self, name: &str) -> Result<(DType, Vec<usize>, Vec<u8>)> {
            let t = self.0.iter().find(|t| t.0 == name).context("missing")?;
            Ok((t.1.clone(), t.2.clone(), t.3.clone()))
        }
    }
    fn quantize(w: &[f32], rows: usize, cols: usize, scale: u8) -> Vec<u8> {
        let s = e8m0_to_f32(scale);
        let _ = rows;
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
            let wq: Vec<f32> = quantize(&w, rows, cols, 120).iter().map(|&c| e4m3_to_f32(c) * e8m0_to_f32(120)).collect();
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

    /// Fixture-backed: the published Keys 1017 overlay against the stock `wo_b`
    /// bytes reproduces the pilot (rank 1, one direction, α ≈ 3.3–3.5, accept).
    /// Needs the fixtures on pg; run with
    /// `cargo test -p cuteafd-loader --release keys_1017 -- --ignored --nocapture`.
    #[test]
    #[ignore = "reads 1.1 GB of local fixtures"]
    fn keys_1017_overlay_reproduces_the_pilot() -> Result<()> {
        let stock = std::env::var("V41_VARIANT_STOCK")
            .unwrap_or_else(|_| shellexpand("~/.cache/weight-variants/dsf1017-wo_b-stock"));
        let overlay = std::env::var("V41_VARIANT_OVERLAY").unwrap_or_else(|_| shellexpand(
            "~/.cache/huggingface/hub/models--drowzeys--keys-DeepSeek-V4.1-Flash-1017-Abliterated/snapshots/de412d2249632903dace23399526e00a145aada5"));
        let base = BinFixture::open(Path::new(&stock))?;
        let source = V41VariantSource::open(Path::new(&overlay), 40)?;
        assert_eq!(source.layers, (10..=35).collect::<Vec<u16>>(), "sites come from the overlay's own tensors");
        let started = std::time::Instant::now();
        let derived = derive_v41_variant(&base, &source, 5120, &V41VariantOpts::default())?;
        eprintln!("{}", serde_json::to_string_pretty(&derived)?);
        eprintln!("derived {} sites in {:.1}s", derived.sites.len(), started.elapsed().as_secs_f32());
        assert!(derived.accept, "{:?}", derived.reason);
        assert!(derived.shared_direction && derived.direction_cos_min > 0.999);
        assert_eq!(derived.alpha.len(), 1);
        assert!((3.3..=3.55).contains(&derived.alpha[0]), "alpha {}", derived.alpha[0]);
        for site in &derived.sites {
            assert!((0.84..=0.92).contains(&site.top_share) && site.second_share < 0.005, "{site:?}");
            assert!((3.25..=3.45).contains(&site.alpha_ls) && site.projection_cos > 0.985, "{site:?}");
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
