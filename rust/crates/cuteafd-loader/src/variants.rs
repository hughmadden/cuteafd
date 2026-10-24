//! Weight-variant derivation and verification (contract §3), shared by the
//! family loaders and `cuteafd plan`.
//!
//! A published variant edits a residual-writing matrix on its output side:
//! `W' = (I − A Bᵀ) W`. Per edited tensor this finds the edit's left
//! singular subspace `U` (randomized subspace iteration on `ΔW = W' − W`,
//! Rayleigh–Ritz on the small Gram), fits `C` by least squares in
//! `ΔW ≈ −U C (Uᵀ W)`, and checks the projection form. The site operator is
//! then `a = U C`, `b = U`, which the steering kernel applies as
//! `y -= a (bᵀ y)`. A rank-1 projection comes out as `a = α r`, `b = r`.
//!
//! Families supply the dequantized FP32 matrices ([`TensorSource::matrix`]
//! decodes BF16, FP16, FP32 and FP8 E4M3 with F32 `weight_scale_inv` or
//! UE8M0 `.scale` blocks) and set the accept bars per format in
//! [`DeriveOpts`].
use anyhow::{bail, ensure, Context, Result};
use cuteafd_core::variants::{SiteKey, SiteOp, VariantError, MAX_STEER_RANK};
use cuteafd_core::DType;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeMap;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// A dense row-major FP32 matrix: `rows` = H (output), `cols` = K (input).
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}
impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data does not match {rows}x{cols}");
        Self { rows, cols, data }
    }
    pub fn zeros(rows: usize, cols: usize) -> Self { Self::new(rows, cols, vec![0.0; rows * cols]) }
    pub fn row(&self, r: usize) -> &[f32] { &self.data[r * self.cols..][..self.cols] }
    pub fn frobenius2(&self) -> f64 { self.data.par_iter().map(|&v| f64::from(v) * f64::from(v)).sum() }
}

/// Accept bars and search limits; set per family and format.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeriveOpts {
    /// Largest rank searched (≤ 32).
    pub k_max: usize,
    /// A singular direction is kept while its share of ‖ΔW‖² exceeds this;
    /// the first one not kept must be at or below it.
    pub rank_share: f32,
    /// Leftover after the fit, relative to ‖ΔW‖ (BF16/FP16 bar: 0.2).
    pub max_r_delta: Option<f32>,
    /// Leftover after the fit, relative to ‖W‖ (the format's noise floor).
    pub max_r_w: Option<f32>,
    /// Projection-form check: |cos(UᵀΔW, −C UᵀW)| must reach this.
    pub min_proj_cos: f32,
    /// A tensor whose ‖ΔW‖/‖W‖ is at or below this is unchanged (a
    /// re-encoded but unedited tensor carries only format noise).
    pub edit_floor: f32,
    /// Extra subspace columns beyond `k_max + 1`, and power iterations.
    pub oversample: usize,
    pub iterations: usize,
}
impl Default for DeriveOpts {
    fn default() -> Self {
        Self { k_max: MAX_STEER_RANK, rank_share: 0.01, max_r_delta: Some(0.2), max_r_w: None, min_proj_cos: 0.98,
            edit_floor: 0.0, oversample: 8, iterations: 4 }
    }
}

/// What the derivation measured for one tensor.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VerifyReport {
    /// Rank kept.
    pub k: u8,
    /// Diagonal of the fitted `C` (α for rank 1).
    pub alphas: Vec<f32>,
    /// Top singular value's share of ‖ΔW‖².
    pub top_share: f32,
    /// Share of the first direction not kept.
    pub next_share: f32,
    pub proj_cos: f32,
    /// ‖ΔW‖ / ‖W‖.
    pub rel_delta: f32,
    /// ‖leftover‖ / ‖ΔW‖ and ‖leftover‖ / ‖W‖.
    pub r_delta: f32,
    pub r_w: f32,
    pub accept: bool,
    pub reason: Option<String>,
}

fn derive_error(reason: impl Into<String>) -> VariantError { VariantError::Derive(reason.into()) }

/// Derive the steering operator that turns `base` into `variant` and verify it.
/// A verification failure is `Ok` with `accept == false` and a reason, so a
/// planner can print the measurements; shape errors and unchanged tensors are errors.
pub fn derive_site(base: &Matrix, variant: &Matrix, opts: &DeriveOpts) -> Result<(SiteOp, VerifyReport), VariantError> {
    if (base.rows, base.cols) != (variant.rows, variant.cols) {
        return Err(derive_error(format!("shape {}x{} against variant {}x{}", base.rows, base.cols, variant.rows, variant.cols)));
    }
    if opts.k_max == 0 || opts.k_max > MAX_STEER_RANK {
        return Err(derive_error(format!("k_max {} is outside 1..={MAX_STEER_RANK}", opts.k_max)));
    }
    let (h, kk) = (base.rows, base.cols);
    let delta = Matrix::new(h, kk, base.data.par_iter().zip(&variant.data).map(|(w, v)| v - w).collect());
    let d2 = delta.frobenius2();
    let w2 = base.frobenius2();
    if d2 == 0.0 { return Err(VariantError::Unchanged); }
    if !d2.is_finite() || !w2.is_finite() || w2 == 0.0 { return Err(derive_error("non-finite or zero base")); }
    if (d2 / w2).sqrt() <= f64::from(opts.edit_floor) { return Err(VariantError::Unchanged); }

    // Top singular subspace of ΔW: m columns, enough to see the share of direction k_max + 1.
    let m = (opts.k_max + 1 + opts.oversample).min(h).min(kk);
    let (u_all, sigma2) = top_left_subspace(&delta, m, opts.iterations);
    let shares: Vec<f64> = sigma2.iter().map(|s| s / d2).collect();
    let mut k = 1;
    while k < opts.k_max.min(m) && shares[k] > f64::from(opts.rank_share) { k += 1; }
    let next_share = shares.get(k).copied().unwrap_or(0.0);
    let u = Matrix::new(h, k, (0..h).flat_map(|r| u_all.row(r)[..k].to_vec()).collect());

    // One pass: Qd = UᵀΔW and P = UᵀW (k×K), reduced to k×k Grams in f64.
    let qd = ut_times(&u, &delta);
    let p = ut_times(&u, base);
    let m1 = gram(&qd, &p, k); // Qd Pᵀ
    let m2 = gram(&p, &p, k); // P Pᵀ
    let qq: f64 = qd.iter().map(|v| v * v).sum();
    // C = −M1 M2⁻¹  (minimizes ‖ΔW + U C P‖).
    let c = solve_right(&m1, &m2, k).ok_or_else(|| derive_error("UᵀW is singular: the edit is not on the output side"))?;
    let c: Vec<f64> = c.iter().map(|v| -v).collect();
    let tr_cm1 = trace_abt(&c, &m1, k); // tr(C M1ᵀ)
    let tr_cm2c = quad(&c, &m2, k); // tr(C M2 Cᵀ)
    let r2 = (d2 + 2.0 * tr_cm1 + tr_cm2c).max(0.0);
    let proj_cos = if qq > 0.0 && tr_cm2c > 0.0 { (-tr_cm1 / (qq.sqrt() * tr_cm2c.sqrt())).abs() } else { 0.0 };
    let report = VerifyReport {
        k: k as u8,
        alphas: (0..k).map(|i| c[i * k + i] as f32).collect(),
        top_share: shares[0] as f32,
        next_share: next_share as f32,
        proj_cos: proj_cos as f32,
        rel_delta: (d2 / w2).sqrt() as f32,
        r_delta: (r2 / d2).sqrt() as f32,
        r_w: (r2 / w2).sqrt() as f32,
        accept: true,
        reason: None,
    };
    let report = judge(report, opts);
    // a_j = Σ_i U_i C_ij,  b_j = U_j, stored [k][H].
    let mut a = vec![0f32; k * h];
    let mut b = vec![0f32; k * h];
    for r in 0..h {
        let row = u.row(r);
        for j in 0..k {
            b[j * h + r] = row[j];
            a[j * h + r] = (0..k).map(|i| f64::from(row[i]) * c[i * k + j]).sum::<f64>() as f32;
        }
    }
    Ok((SiteOp::Steer { k: k as u8, a, b }, report))
}

fn judge(mut report: VerifyReport, opts: &DeriveOpts) -> VerifyReport {
    let fail = if report.next_share > opts.rank_share {
        Some(format!("rank above {}: direction {} still holds {:.4} of ‖ΔW‖² (bar {})", report.k, u32::from(report.k) + 1,
            report.next_share, opts.rank_share))
    } else if report.proj_cos < opts.min_proj_cos {
        Some(format!("not an output-side projection: |cos| {:.4} < {}", report.proj_cos, opts.min_proj_cos))
    } else if opts.max_r_delta.is_some_and(|bar| report.r_delta > bar) {
        Some(format!("leftover {:.4}·‖ΔW‖ above {}", report.r_delta, opts.max_r_delta.unwrap()))
    } else if opts.max_r_w.is_some_and(|bar| report.r_w > bar) {
        Some(format!("leftover {:.5}·‖W‖ above the floor {}", report.r_w, opts.max_r_w.unwrap()))
    } else { None };
    report.accept = fail.is_none();
    report.reason = fail;
    report
}

/// `W − A (Bᵀ W)` for a steering op: the matrix the variant should equal.
pub fn rebuild(base: &Matrix, op: &SiteOp) -> Matrix {
    let SiteOp::Steer { k, a, b } = op else { panic!("rebuild needs a steering op") };
    let (h, kk, k) = (base.rows, base.cols, usize::from(*k));
    let bt = Matrix::new(h, k, (0..h).flat_map(|r| (0..k).map(move |j| b[j * h + r])).collect());
    let p = ut_times(&bt, base);
    let mut out = base.clone();
    out.data.par_chunks_mut(kk).enumerate().for_each(|(r, row)| {
        for j in 0..k {
            let aj = f64::from(a[j * h + r]);
            for (c, v) in row.iter_mut().enumerate() { *v = (f64::from(*v) - aj * p[j * kk + c]) as f32; }
        }
    });
    out
}

// ---- linear algebra (enough for m ≤ ~48 columns) -----------------------------------------

/// Top `m` left singular vectors (columns of an H×m matrix) and their σ², descending.
fn top_left_subspace(d: &Matrix, m: usize, iterations: usize) -> (Matrix, Vec<f64>) {
    let (h, kk) = (d.rows, d.cols);
    // Y = D Ω with a fixed pseudo-random Ω (K×m): deterministic across runs.
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let omega: Vec<f32> = (0..kk * m).map(|_| gaussianish(&mut seed)).collect();
    let mut q = orthonormalize(d_times(d, &omega, m), h, m, &mut seed);
    for _ in 0..iterations {
        let z = dt_times(d, &q, m);
        q = orthonormalize(d_times(d, &z, m), h, m, &mut seed);
    }
    // Rayleigh–Ritz: B = Qᵀ D, G = B Bᵀ (m×m); eigenvectors rotate Q.
    let z = dt_times(d, &q, m); // K×m = (QᵀD)ᵀ
    let mut g = vec![0f64; m * m];
    let partial: Vec<Vec<f64>> = z.par_chunks(m * 1024).map(|chunk| {
        let mut g = vec![0f64; m * m];
        for row in chunk.chunks(m) {
            for i in 0..m { let zi = f64::from(row[i]); for j in i..m { g[i * m + j] += zi * f64::from(row[j]); } }
        }
        g
    }).collect();
    for part in partial { for (a, b) in g.iter_mut().zip(part) { *a += b; } }
    for i in 0..m { for j in 0..i { g[i * m + j] = g[j * m + i]; } }
    let (values, vectors) = jacobi_eigen(g, m);
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    let qv: Vec<f64> = q.iter().map(|&v| v as f64).collect();
    let mut u = vec![0f32; h * m];
    u.par_chunks_mut(m).enumerate().for_each(|(r, out)| {
        let qr = &qv[r * m..][..m];
        for (o, &col) in out.iter_mut().zip(&order) {
            *o = (0..m).map(|i| qr[i] * vectors[i * m + col]).sum::<f64>() as f32;
        }
    });
    (Matrix::new(h, m, u), order.iter().map(|&i| values[i].max(0.0)).collect())
}

fn gaussianish(state: &mut u64) -> f32 {
    // Sum of four uniforms (Irwin–Hall), centred: plenty for a range finder.
    let mut s = 0f32;
    for _ in 0..4 {
        *state ^= *state << 13; *state ^= *state >> 7; *state ^= *state << 17;
        s += (*state >> 40) as f32 / (1u64 << 24) as f32;
    }
    s - 2.0
}

/// D (H×K) times X (K×m) → H×m.
fn d_times(d: &Matrix, x: &[f32], m: usize) -> Vec<f32> {
    let mut y = vec![0f32; d.rows * m];
    y.par_chunks_mut(m).enumerate().for_each(|(r, out)| {
        let mut acc = vec![0f64; m];
        for (c, &v) in d.row(r).iter().enumerate() {
            if v == 0.0 { continue; }
            let xr = &x[c * m..][..m];
            for j in 0..m { acc[j] += f64::from(v) * f64::from(xr[j]); }
        }
        for j in 0..m { out[j] = acc[j] as f32; }
    });
    y
}

/// Dᵀ (K×H) times Q (H×m) → K×m, parallel over column blocks of D.
fn dt_times(d: &Matrix, q: &[f32], m: usize) -> Vec<f32> {
    const BLOCK: usize = 256;
    let (h, kk) = (d.rows, d.cols);
    let mut z = vec![0f32; kk * m];
    z.par_chunks_mut(BLOCK * m).enumerate().for_each(|(blk, out)| {
        let c0 = blk * BLOCK;
        let width = out.len() / m;
        let mut acc = vec![0f64; width * m];
        for r in 0..h {
            let qr = &q[r * m..][..m];
            let drow = &d.data[r * kk + c0..][..width];
            for (ci, &v) in drow.iter().enumerate() {
                if v == 0.0 { continue; }
                let a = &mut acc[ci * m..][..m];
                for j in 0..m { a[j] += f64::from(v) * f64::from(qr[j]); }
            }
        }
        for (o, a) in out.iter_mut().zip(acc) { *o = a as f32; }
    });
    z
}

/// Uᵀ X for U (H×k) and X (H×K) → k×K in f64.
fn ut_times(u: &Matrix, x: &Matrix) -> Vec<f64> {
    const BLOCK: usize = 512;
    let (h, k, kk) = (x.rows, u.cols, x.cols);
    let mut out = vec![0f64; k * kk];
    // Column blocks of X; each block owns its k×width slice, laid out per j below.
    let blocks: Vec<(usize, Vec<f64>)> = (0..kk.div_ceil(BLOCK)).into_par_iter().map(|blk| {
        let c0 = blk * BLOCK;
        let width = BLOCK.min(kk - c0);
        let mut acc = vec![0f64; k * width];
        for r in 0..h {
            let ur = u.row(r);
            let xr = &x.data[r * kk + c0..][..width];
            for j in 0..k {
                let uj = f64::from(ur[j]);
                if uj == 0.0 { continue; }
                let a = &mut acc[j * width..][..width];
                for (a, &v) in a.iter_mut().zip(xr) { *a += uj * f64::from(v); }
            }
        }
        (c0, acc)
    }).collect();
    for (c0, acc) in blocks {
        let width = acc.len() / k;
        for j in 0..k { out[j * kk + c0..][..width].copy_from_slice(&acc[j * width..][..width]); }
    }
    out
}

/// X Yᵀ for row-major k×K operands → k×k.
fn gram(x: &[f64], y: &[f64], k: usize) -> Vec<f64> {
    let kk = x.len() / k;
    debug_assert_eq!(x.len(), y.len());
    let mut out = vec![0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            out[i * k + j] = x[i * kk..][..kk].par_iter().zip(&y[j * kk..][..kk]).map(|(a, b)| a * b).sum();
        }
    }
    out
}

/// Orthonormalize the m columns of an H×m matrix (two passes of modified Gram–Schmidt).
fn orthonormalize(y: Vec<f32>, h: usize, m: usize, seed: &mut u64) -> Vec<f32> {
    let mut cols: Vec<Vec<f64>> = (0..m).map(|j| (0..h).map(|r| f64::from(y[r * m + j])).collect()).collect();
    let scale = cols.iter().map(|c| norm(c)).fold(0.0, f64::max).max(f64::MIN_POSITIVE);
    for j in 0..m {
        for _pass in 0..2 {
            let (done, rest) = cols.split_at_mut(j);
            let cj = &mut rest[0];
            for ci in done.iter() {
                let dot: f64 = ci.par_iter().zip(cj.par_iter()).map(|(a, b)| a * b).sum();
                cj.par_iter_mut().zip(ci.par_iter()).for_each(|(b, a)| *b -= dot * a);
            }
        }
        let n = norm(&cols[j]);
        if n <= scale * 1e-12 {
            // Rank-deficient: replace with a fresh random direction orthogonal to the rest.
            let mut fresh: Vec<f64> = (0..h).map(|_| f64::from(gaussianish(seed))).collect();
            for _pass in 0..2 {
                for ci in &cols[..j] {
                    let dot: f64 = ci.iter().zip(&fresh).map(|(a, b)| a * b).sum();
                    fresh.iter_mut().zip(ci).for_each(|(b, a)| *b -= dot * a);
                }
            }
            let n = norm(&fresh);
            cols[j] = fresh.into_iter().map(|v| v / n).collect();
        } else {
            cols[j].par_iter_mut().for_each(|v| *v /= n);
        }
    }
    let mut out = vec![0f32; h * m];
    for (j, col) in cols.iter().enumerate() { for r in 0..h { out[r * m + j] = col[r] as f32; } }
    out
}
fn norm(v: &[f64]) -> f64 { v.par_iter().map(|x| x * x).sum::<f64>().sqrt() }

/// Cyclic Jacobi eigen-decomposition of a symmetric n×n matrix: (values, vectors as columns).
fn jacobi_eigen(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0f64; n * n];
    for i in 0..n { v[i * n + i] = 1.0; }
    let total: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j]).sum::<f64>().sqrt();
        if off <= total * 1e-15 || off == 0.0 { break; }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() <= total * 1e-300 { continue; }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// X = A M⁻¹ for k×k A and symmetric positive definite M (Gaussian elimination with pivoting).
fn solve_right(a: &[f64], m: &[f64], k: usize) -> Option<Vec<f64>> {
    // Solve Mᵀ Xᵀ = Aᵀ; M is symmetric, so solve M Xᵀ = Aᵀ.
    let mut aug: Vec<Vec<f64>> = (0..k).map(|r| {
        let mut row = m[r * k..][..k].to_vec();
        row.extend((0..k).map(|c| a[c * k + r]));
        row
    }).collect();
    let scale = m.iter().fold(0f64, |s, v| s.max(v.abs()));
    for col in 0..k {
        let pivot = (col..k).max_by(|&x, &y| aug[x][col].abs().total_cmp(&aug[y][col].abs()))?;
        if aug[pivot][col].abs() <= scale * 1e-14 { return None; }
        aug.swap(col, pivot);
        let p = aug[col][col];
        for v in aug[col].iter_mut() { *v /= p; }
        for r in 0..k {
            if r == col { continue; }
            let f = aug[r][col];
            if f == 0.0 { continue; }
            let pivot_row = aug[col].clone();
            for (v, pv) in aug[r].iter_mut().zip(pivot_row) { *v -= f * pv; }
        }
    }
    // aug[:, k..] is Xᵀ.
    let mut x = vec![0f64; k * k];
    for r in 0..k { for c in 0..k { x[c * k + r] = aug[r][k + c]; } }
    Some(x)
}
/// tr(A Bᵀ) for k×k operands.
fn trace_abt(a: &[f64], b: &[f64], _k: usize) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
/// tr(C M Cᵀ).
fn quad(c: &[f64], m: &[f64], k: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..k { for p in 0..k { for q in 0..k { s += c[i * k + p] * m[p * k + q] * c[i * k + q]; } } }
    s
}

// ---- tensor decode --------------------------------------------------------------------------

/// FP8 E4M3 (fn: no infinities, 0x7f/0xff NaN) to f32.
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (byte >> 3) & 0x0f;
    let man = f32::from(byte & 0x07);
    if exp == 0x0f && byte & 0x07 == 0x07 { return f32::NAN; }
    if exp == 0 { return sign * man * (2f32).powi(-9); }
    sign * (1.0 + man / 8.0) * (2f32).powi(i32::from(exp) - 7)
}
/// f32 to FP8 E4M3, round to nearest even, saturating at ±448 (torch's clamp-then-convert).
pub fn f32_to_e4m3(value: f32) -> u8 {
    if value.is_nan() { return 0x7f; }
    let sign = if value.is_sign_negative() { 0x80u8 } else { 0 };
    let x = value.abs().min(448.0);
    // Exhaustive nearest search over the 127 finite magnitudes keeps this obviously right.
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0u8..0x7f {
        let err = (e4m3_to_f32(code) - x).abs();
        if err < best_err || (err == best_err && code & 1 == 0 && best & 1 == 1) { best = code; best_err = err; }
    }
    sign | best
}
pub fn ue8m0_to_f32(byte: u8) -> f32 { if byte == 0xff { f32::NAN } else { (2f32).powi(i32::from(byte) - 127) } }
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (bits >> 10) & 0x1f;
    let man = f32::from(bits & 0x3ff);
    match exp {
        0 => sign * man * (2f32).powi(-24),
        0x1f => if man == 0.0 { sign * f32::INFINITY } else { f32::NAN },
        e => sign * (1.0 + man / 1024.0) * (2f32).powi(i32::from(e) - 15),
    }
}

/// Where a set of named tensors lives: a checkpoint's index, or a bag of shard files.
#[derive(Debug, Clone, Default)]
pub struct TensorSource {
    tensors: BTreeMap<String, (PathBuf, crate::SafetensorsTensorMetadata)>,
}
impl TensorSource {
    /// Every tensor of the given safetensors files (no index needed).
    pub fn from_files(files: impl IntoIterator<Item = PathBuf>) -> Result<Self> {
        let mut tensors = BTreeMap::new();
        for path in files {
            for meta in crate::read_safetensors_metadata(&path)? {
                tensors.insert(meta.name.clone(), (path.clone(), meta));
            }
        }
        Ok(Self { tensors })
    }
    /// The headers a role read with `Checkpoint::open_for_role`.
    pub fn from_checkpoint(checkpoint: &crate::plan::checkpoint::Checkpoint) -> Self {
        let tensors = checkpoint.tensors.iter()
            .map(|t| (t.meta.name.clone(), (checkpoint.snapshot.join(&t.shard), t.meta.clone()))).collect();
        Self { tensors }
    }
    pub fn contains(&self, name: &str) -> bool { self.tensors.contains_key(name) }
    pub fn names(&self) -> impl Iterator<Item = &str> { self.tensors.keys().map(String::as_str) }
    pub fn shape(&self, name: &str) -> Option<&[usize]> { self.tensors.get(name).map(|(_, m)| m.shape.as_slice()) }

    fn bytes(&self, name: &str) -> Result<(Vec<u8>, &crate::SafetensorsTensorMetadata)> {
        let (path, meta) = self.tensors.get(name).with_context(|| format!("tensor {name} is not in this source"))?;
        let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // Header offsets are absolute file offsets (`read_safetensors_metadata` adds the data start).
        let mut bytes = vec![0u8; usize::try_from(meta.byte_length)?];
        file.read_exact_at(&mut bytes, meta.byte_offset)
            .with_context(|| format!("reading {name} from {}", path.display()))?;
        Ok((bytes, meta))
    }

    /// The dequantized FP32 matrix for a 2-D `.weight` tensor. FP8 E4M3 needs
    /// a block-scale partner: `<name>_scale_inv` (F32, GLM) or `<module>.scale`
    /// (UE8M0, DeepSeek V4.1); block sizes come from the two shapes.
    pub fn matrix(&self, name: &str) -> Result<Matrix> {
        let (bytes, meta) = self.bytes(name)?;
        ensure!(meta.shape.len() == 2, "{name} has shape {:?}, expected 2-D", meta.shape);
        let (rows, cols) = (meta.shape[0], meta.shape[1]);
        let data: Vec<f32> = match meta.dtype {
            DType::F32 => bytes.par_chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            DType::Bf16 => bytes.par_chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect(),
            DType::F16 => bytes.par_chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
            DType::F8E4M3 => {
                let (scale, srows, scols) = self.block_scale(name)?;
                let (br, bc) = (rows.div_ceil(srows), cols.div_ceil(scols));
                let lut: Vec<f32> = (0..=255u8).map(e4m3_to_f32).collect();
                let mut out = vec![0f32; rows * cols];
                out.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
                    let src = &bytes[r * cols..][..cols];
                    let srow = &scale[(r / br) * scols..][..scols];
                    for (c, (o, &b)) in row.iter_mut().zip(src).enumerate() { *o = lut[usize::from(b)] * srow[c / bc]; }
                });
                out
            }
            ref other => bail!("{name}: dtype {other:?} is not a supported variant format"),
        };
        ensure!(data.len() == rows * cols, "{name}: {} values for shape {rows}x{cols}", data.len());
        Ok(Matrix::new(rows, cols, data))
    }

    fn block_scale(&self, name: &str) -> Result<(Vec<f32>, usize, usize)> {
        let module = name.strip_suffix(".weight").unwrap_or(name);
        let candidates = [format!("{name}_scale_inv"), format!("{module}.weight_scale_inv"), format!("{module}.scale")];
        let scale_name = candidates.iter().find(|c| self.contains(c))
            .with_context(|| format!("{name}: FP8 weight without a block scale ({})", candidates.join(", ")))?;
        let (bytes, meta) = self.bytes(scale_name)?;
        ensure!(meta.shape.len() == 2, "{scale_name} is not 2-D");
        let values = match meta.dtype {
            DType::F32 => bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            DType::F8E8M0 | DType::U8 => bytes.iter().map(|&b| ue8m0_to_f32(b)).collect(),
            DType::Bf16 => bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect(),
            ref other => bail!("{scale_name}: scale dtype {other:?} is not supported"),
        };
        Ok((values, meta.shape[0], meta.shape[1]))
    }
}

/// One site a family wants derived: which tensor (in both sources) feeds which site.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteTensor {
    pub key: SiteKey,
    pub tensor: String,
    pub opts: DeriveOpts,
}

/// The outcome for one site tensor.
#[derive(Debug, Clone, Serialize)]
pub struct SiteVerdict {
    pub site: String,
    pub tensor: String,
    /// `None` when the tensor is unchanged (no site).
    pub report: Option<VerifyReport>,
}

/// Derive every listed site, in parallel across tensors (`parallel` at a
/// time, bounding FP32 memory). Unchanged tensors yield no op; a rejected
/// tensor is returned in the verdicts and makes the variant unavailable.
pub fn derive_sites(base: &TensorSource, variant: &TensorSource, sites: &[SiteTensor], parallel: usize)
    -> Result<(Vec<(SiteKey, SiteOp)>, Vec<SiteVerdict>)> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(parallel.max(1)).build()?;
    let results: Vec<Result<(Option<(SiteKey, SiteOp)>, SiteVerdict)>> = pool.install(|| sites.par_iter().map(|site| {
        let w = base.matrix(&site.tensor).with_context(|| format!("base {}", site.tensor))?;
        let v = variant.matrix(&site.tensor).with_context(|| format!("variant {}", site.tensor))?;
        Ok(match derive_site(&w, &v, &site.opts) {
            Ok((op, report)) => (report.accept.then_some((site.key, op)),
                SiteVerdict { site: site.key.to_string(), tensor: site.tensor.clone(), report: Some(report) }),
            Err(VariantError::Unchanged) => (None, SiteVerdict { site: site.key.to_string(), tensor: site.tensor.clone(), report: None }),
            Err(error) => bail!("{}: {error}", site.tensor),
        })
    }).collect());
    let mut ops = Vec::new();
    let mut verdicts = Vec::new();
    for result in results {
        let (op, verdict) = result?;
        ops.extend(op);
        verdicts.push(verdict);
    }
    Ok((ops, verdicts))
}

/// The first rejected site, as a planning result: "tensor T: reason".
pub fn first_rejection(verdicts: &[SiteVerdict]) -> Option<String> {
    verdicts.iter().find_map(|v| v.report.as_ref().filter(|r| !r.accept)
        .map(|r| format!("{} ({}): {}", v.tensor, v.site, r.reason.as_deref().unwrap_or("rejected"))))
}

/// A family's variant site map: which tensors to read and which sites they feed.
pub struct FamilyVariants {
    pub family: &'static str,
    pub hidden: usize,
    /// Tensor names (and scale partners) to read from both sides.
    pub wants: fn(&str) -> bool,
    sites: Box<dyn Fn(&TensorSource) -> Vec<SiteTensor> + Send + Sync>,
}
impl FamilyVariants {
    pub fn sites(&self, variant: &TensorSource) -> Vec<SiteTensor> { (self.sites)(variant) }
}

/// The site map for the served checkpoint's family; `mode` is the `--variant` `mode=`.
pub fn family_variants(config: &serde_json::Value, mode: Option<&str>) -> Result<FamilyVariants> {
    let text = config.get("text_config").unwrap_or(config);
    let int = |key: &str| text[key].as_u64().map(|v| v as usize).with_context(|| format!("config.json {key}"));
    match text["model_type"].as_str().unwrap_or_default() {
        "glm_moe_dsa" => {
            use crate::families::glm5::variants::{variant_sites, wants_tensor, GlmVariantMode};
            let mode: GlmVariantMode = mode.map(str::parse).transpose().map_err(anyhow::Error::msg)?.unwrap_or_default();
            let (layers, first_moe) = (int("num_hidden_layers")?, int("first_k_dense_replace")?);
            Ok(FamilyVariants { family: "glm5", hidden: int("hidden_size")?, wants: wants_tensor,
                sites: Box::new(move |variant| variant_sites(layers, first_moe, mode, variant)) })
        }
        "deepseek_v41_text" => {
            use crate::families::deepseek_v41::v41_variant::{variant_sites, wants_tensor};
            ensure!(mode.is_none(), "DeepSeek V4.1 has no variant modes");
            let layers = int("num_hidden_layers")?;
            Ok(FamilyVariants { family: "deepseek_v41", hidden: int("hidden_size")?, wants: wants_tensor,
                sites: Box::new(move |variant| variant_sites(layers, variant)) })
        }
        other => bail!("weight variants are not supported for model_type {other:?} yet"),
    }
}

/// Open a variant or base source: a single `.safetensors` file, a snapshot
/// read through its own index for the tensors `wants` names, or a bare
/// folder of shards.
pub fn open_source(path: &Path, role: &str, wants: fn(&str) -> bool) -> Result<TensorSource> {
    if path.is_file() { return TensorSource::from_files([path.to_path_buf()]); }
    if !path.join("model.safetensors.index.json").is_file() && !path.join("model.safetensors").is_file() {
        // A bare folder of shards (e.g. tensors range-read from a published checkpoint).
        let files = safetensors_files(path)?;
        let mut source = TensorSource::from_files(files)?;
        source.tensors.retain(|name, _| wants(name));
        return Ok(source);
    }
    let checkpoint = crate::plan::checkpoint::Checkpoint::open_for_role(path, role, wants)
        .with_context(|| format!("opening {} for {role}", path.display()))?;
    Ok(TensorSource::from_checkpoint(&checkpoint))
}

/// One variant derived against its base: the plan when every site verified,
/// otherwise the planning result naming the rejected tensor.
#[derive(Debug, Clone)]
pub struct VariantOutcome {
    pub name: String,
    pub public_id: String,
    pub plan: Option<cuteafd_core::variants::VariantPlan>,
    pub rejection: Option<String>,
    pub verdicts: Vec<SiteVerdict>,
    /// Site tensors the variant carries but the base does not (not derived).
    pub missing_in_base: Vec<String>,
    pub seconds: f64,
}
impl VariantOutcome {
    /// The verification summary `/v1/models` carries.
    pub fn verify_summary(&self) -> serde_json::Value {
        let reports: Vec<&VerifyReport> = self.verdicts.iter().filter_map(|v| v.report.as_ref()).collect();
        let max = |f: fn(&VerifyReport) -> f32| reports.iter().map(|r| f(r)).fold(0f32, f32::max);
        let min = |f: fn(&VerifyReport) -> f32| reports.iter().map(|r| f(r)).fold(f32::INFINITY, f32::min);
        serde_json::json!({"tensors": reports.len(), "unchanged": self.verdicts.len() - reports.len(),
            "r_w_max": max(|r| r.r_w), "next_share_max": max(|r| r.next_share), "proj_cos_min": min(|r| r.proj_cos),
            "alpha_min": min(|r| r.alphas.iter().copied().fold(f32::INFINITY, f32::min)),
            "alpha_max": max(|r| r.alphas.iter().copied().fold(0f32, f32::max)),
            "derive_seconds": self.seconds})
    }
}

/// Derive and verify one `--variant` against `served` (or its `base=`).
pub fn derive_variant(served: &Path, served_id: &str, index: u8, spec: &cuteafd_core::variants::VariantSpec,
    parallel: usize) -> Result<VariantOutcome> {
    let started = std::time::Instant::now();
    let config = crate::plan::checkpoint::read_json(&served.join("config.json"))?;
    let family = family_variants(&config, spec.mode.as_deref())?;
    let base_path = spec.base.as_deref().map_or_else(|| served.to_path_buf(), PathBuf::from);
    let variant = open_source(Path::new(&spec.source), "variant", family.wants)?;
    let base = open_source(&base_path, "variant-base", family.wants)?;
    let (sites, missing): (Vec<SiteTensor>, Vec<SiteTensor>) = family.sites(&variant).into_iter().partition(|s| base.contains(&s.tensor));
    let missing_in_base: Vec<String> = missing.into_iter().map(|s| s.tensor).collect();
    ensure!(!sites.is_empty(), "variant {}: {} holds none of the {} site tensors the base has", spec.name, spec.source, family.family);
    let (ops, verdicts) = derive_sites(&base, &variant, &sites, parallel)?;
    let rejection = first_rejection(&verdicts);
    let public_id = spec.public_id(served_id);
    let plan = if rejection.is_none() {
        ensure!(!ops.is_empty(), "variant {}: every site tensor is unchanged against the base", spec.name);
        Some(cuteafd_core::variants::VariantPlan::new(index, spec.name.clone(), public_id.clone(),
            u32::try_from(family.hidden)?, ops)?)
    } else { None };
    Ok(VariantOutcome { name: spec.name.clone(), public_id, plan, rejection, verdicts, missing_in_base,
        seconds: started.elapsed().as_secs_f64() })
}

/// Path helper for families: the `.safetensors` files of a directory or a single file.
pub fn safetensors_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() { return Ok(vec![path.to_path_buf()]); }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path).with_context(|| format!("reading {}", path.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors")).collect();
    files.sort();
    ensure!(!files.is_empty(), "{} holds no .safetensors files", path.display());
    Ok(files)
}

#[cfg(test)]
mod tests;
