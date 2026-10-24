//! Host reference for the weight-variant row steering kernel
//! (`cuteafd_steer_rows`, weight-variants contract section 4).
//!
//! For each row with `v = vidx[r] > 0` and `kv = min(k[v], k_max) > 0`:
//! `t_j = sum_h b[v][j][h] x[h]`, then
//! `x[h] = bf16_rne(x[h] - sum_j a[v][j][h] t_j)`. Other rows are untouched.
//! The reduction order copies the SparkInfer export exactly, so the kernel
//! must match this function bit for bit:
//!
//! - One 512-thread CTA per row. Thread `t` owns the 8-element vectors
//!   `q = t` and `q = t + 512` with `8q < h`; its partial is an `fma` chain
//!   from `+0` over its elements in ascending order.
//! - Each 32-lane warp combines partials with a butterfly at offsets
//!   1, 2, 4, 8, 16 (`v[l] = v[l] + v[l ^ offset]`).
//! - `t_j = ((+0 + w_0) + w_1) + ... + w_15` over the 16 warp sums.
//! - `s` starts at `+0` and takes `s = fma(a_j, t_j, s)` for `j` ascending.
//!
//! A row's result depends only on its own inputs (batch-invariant).
//!
//! [`add_rows_ref`] is the reference of `cuteafd_add_rows`, the stored-delta
//! site: `y = bf16_rne(y + d)` on the rows of variants that carry a delta.

/// Threads per row in the kernel.
pub const STEER_THREADS: usize = 512;
/// Widest supported row (`h`), a multiple of [`STEER_HIDDEN_MULTIPLE`].
pub const STEER_MAX_HIDDEN: usize = 8192;
pub const STEER_HIDDEN_MULTIPLE: usize = 128;
/// Largest `k_max` (the `k_max == 1` export is the fast path).
pub const STEER_MAX_K: usize = 32;

const LANES: usize = 32;
const WARPS: usize = STEER_THREADS / LANES;
const VECTOR: usize = 8;

/// A shape the kernel rejects (it returns `cudaErrorInvalidValue`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SteerShapeError {
    #[error("steer_rows h={0} must be a multiple of 128 in 128..=8192")]
    Hidden(usize),
    #[error("steer_rows ld={ld} must be >= h={h} and a multiple of 8")]
    Stride { ld: usize, h: usize },
    #[error("steer_rows k_max={0} must be in 1..=32")]
    KMax(usize),
}

/// Checks the launch shape exactly as the native entry point does.
pub fn check_steer_shape(h: usize, ld: usize, k_max: usize) -> Result<(), SteerShapeError> {
    if h < STEER_HIDDEN_MULTIPLE || h > STEER_MAX_HIDDEN || h % STEER_HIDDEN_MULTIPLE != 0 {
        return Err(SteerShapeError::Hidden(h));
    }
    if ld < h || ld % VECTOR != 0 {
        return Err(SteerShapeError::Stride { ld, h });
    }
    if k_max == 0 || k_max > STEER_MAX_K {
        return Err(SteerShapeError::KMax(k_max));
    }
    Ok(())
}

/// BF16 bits to FP32 (exact).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// FP32 to BF16 with round-to-nearest-even and no saturation, as PTX
/// `cvt.rn.bf16x2.f32`: overflow rounds to infinity, NaN becomes `0x7FFF`.
pub fn f32_to_bf16_rne(value: f32) -> u16 {
    if value.is_nan() {
        return 0x7FFF;
    }
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// The kernel's `t = b . x` over one row, in its fixed order.
fn dot(b: &[f32], x: &[f32], h: usize) -> f32 {
    let vectors = h / VECTOR;
    let mut partial = [0.0f32; STEER_THREADS];
    for (t, slot) in partial.iter_mut().enumerate() {
        let mut p = 0.0f32;
        for q in [t, t + STEER_THREADS] {
            if q < vectors {
                for e in q * VECTOR..(q + 1) * VECTOR {
                    p = b[e].mul_add(x[e], p);
                }
            }
        }
        *slot = p;
    }
    let mut total = 0.0f32;
    for warp in partial.chunks_exact(LANES) {
        let mut lanes = [0.0f32; LANES];
        lanes.copy_from_slice(warp);
        for offset in [1, 2, 4, 8, 16] {
            let previous = lanes;
            for (l, lane) in lanes.iter_mut().enumerate() {
                *lane = previous[l] + previous[l ^ offset];
            }
        }
        total += lanes[0];
    }
    total
}

/// Applies the steering to `rows` BF16 rows of `x` (`[rows, ld]`, bits) in
/// place. `a` and `b` are `[V+1][k_max][h]`, `k` is `[V+1]`; slot 0 is unused.
///
/// # Panics
/// On a shape [`check_steer_shape`] rejects, a short slice, or a
/// `vidx` value at or beyond `k.len()`.
#[allow(clippy::too_many_arguments)]
pub fn steer_rows_ref(x: &mut [u16], ld: usize, rows: usize, h: usize, vidx: &[i32],
    a: &[f32], b: &[f32], k: &[i32], k_max: usize) {
    if let Err(err) = check_steer_shape(h, ld, k_max) {
        panic!("{err}");
    }
    assert!(vidx.len() >= rows, "vidx holds {} rows, needs {rows}", vidx.len());
    let table = k.len() * k_max * h;
    assert!(a.len() >= table && b.len() >= table, "steer tables must hold [V+1][k_max][h]");
    debug_assert_eq!(WARPS * LANES, STEER_THREADS);
    let mut values = vec![0.0f32; h];
    let mut t = vec![0.0f32; k_max];
    for (r, &v) in vidx[..rows].iter().enumerate() {
        if v <= 0 {
            continue;
        }
        let v = v as usize;
        let kv = (k[v].max(0) as usize).min(k_max);
        if kv == 0 {
            continue;
        }
        let row = &mut x[r * ld..r * ld + h];
        for (value, &bits) in values.iter_mut().zip(row.iter()) {
            *value = bf16_to_f32(bits);
        }
        for (j, tj) in t[..kv].iter_mut().enumerate() {
            let start = (v * k_max + j) * h;
            *tj = dot(&b[start..start + h], &values, h);
        }
        for (e, out) in row.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for (j, &tj) in t[..kv].iter().enumerate() {
                s = a[(v * k_max + j) * h + e].mul_add(tj, s);
            }
            *out = f32_to_bf16_rne(values[e] - s);
        }
    }
}

/// Host reference of `cuteafd_add_rows`: `y[r][i] = bf16_rne(y[r][i] +
/// d[r][i])` on the first `h` columns of each row with `v = vidx[r] > 0` and
/// `on[v] != 0`; other rows are untouched. One FP32 add per element.
///
/// # Panics
/// On short slices, `ld_y < h`, `ld_d < h`, or a `vidx` value at or beyond
/// `on.len()`.
#[allow(clippy::too_many_arguments)]
pub fn add_rows_ref(y: &mut [u16], ld_y: usize, d: &[f32], ld_d: usize, rows: usize, h: usize,
    vidx: &[i32], on: &[i32]) {
    assert!(ld_y >= h && ld_d >= h && vidx.len() >= rows, "add_rows_ref shape");
    for (r, &v) in vidx[..rows].iter().enumerate() {
        if v <= 0 || on[v as usize] == 0 {
            continue;
        }
        let delta = &d[r * ld_d..r * ld_d + h];
        for (out, &dv) in y[r * ld_y..r * ld_y + h].iter_mut().zip(delta) {
            *out = f32_to_bf16_rne(bf16_to_f32(*out) + dv);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn rounding_matches_rne_without_saturation() {
        assert_eq!(f32_to_bf16_rne(1.0), 0x3F80);
        // Ties go to even: 1 + 2^-8 sits halfway between 1 and 1 + 2^-7.
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(f32_to_bf16_rne(f32::MAX), 0x7F80);
        assert_eq!(f32_to_bf16_rne(f32::NEG_INFINITY), 0xFF80);
        assert_eq!(f32_to_bf16_rne(f32::NAN), 0x7FFF);
        assert_eq!(f32_to_bf16_rne(-0.0), 0x8000);
    }

    #[test]
    fn shapes_follow_the_native_checks() {
        assert!(check_steer_shape(4096, 4096, 1).is_ok());
        assert!(check_steer_shape(8192, 8200, 32).is_ok());
        assert_eq!(check_steer_shape(4000, 4096, 1), Err(SteerShapeError::Hidden(4000)));
        assert_eq!(check_steer_shape(8320, 8320, 1), Err(SteerShapeError::Hidden(8320)));
        assert_eq!(check_steer_shape(128, 132, 1), Err(SteerShapeError::Stride { ld: 132, h: 128 }));
        assert_eq!(check_steer_shape(128, 128, 33), Err(SteerShapeError::KMax(33)));
        assert_eq!(check_steer_shape(128, 128, 0), Err(SteerShapeError::KMax(0)));
    }

    #[test]
    fn rank_one_projection_removes_the_direction() {
        // W' = (I - r r^T) W with unit r: the steered row is orthogonal to r.
        let h = 640;
        let mut seed = 7;
        let mut r: Vec<f32> = (0..h).map(|_| lcg(&mut seed)).collect();
        let norm = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        r.iter_mut().for_each(|v| *v /= norm);
        let a = [vec![0.0; h], r.clone()].concat();
        let b = a.clone();
        let mut x: Vec<u16> = (0..h).map(|_| f32_to_bf16_rne(lcg(&mut seed) * 4.0)).collect();
        let before: f32 = x.iter().zip(&r).map(|(&v, &d)| bf16_to_f32(v) * d).sum();
        steer_rows_ref(&mut x, h, 1, h, &[1], &a, &b, &[0, 1], 1);
        let after: f32 = x.iter().zip(&r).map(|(&v, &d)| bf16_to_f32(v) * d).sum();
        assert!(before.abs() > 0.1, "test direction must start with a component");
        assert!(after.abs() < 0.02 * before.abs(), "component {before} -> {after}");
    }

    #[test]
    fn base_and_unedited_rows_are_untouched() {
        let (h, ld, k_max) = (256, 264, 2);
        let mut seed = 11;
        let table: Vec<f32> = (0..3 * k_max * h).map(|_| lcg(&mut seed)).collect();
        let mut x: Vec<u16> = (0..4 * ld).map(|_| f32_to_bf16_rne(lcg(&mut seed))).collect();
        let original = x.clone();
        // Row 0 base, row 1 a variant with k = 0, row 2 negative, row 3 steered.
        steer_rows_ref(&mut x, ld, 4, h, &[0, 2, -1, 1], &table, &table, &[0, 2, 0], k_max);
        assert_eq!(x[..3 * ld], original[..3 * ld]);
        assert_ne!(x[3 * ld..3 * ld + h], original[3 * ld..3 * ld + h]);
        // Columns past h are never written.
        assert_eq!(x[3 * ld + h..], original[3 * ld + h..]);
    }

    #[test]
    fn k_above_k_max_is_clamped() {
        let (h, k_max) = (128, 2);
        let mut seed = 3;
        let table: Vec<f32> = (0..2 * k_max * h).map(|_| lcg(&mut seed)).collect();
        let row: Vec<u16> = (0..h).map(|_| f32_to_bf16_rne(lcg(&mut seed))).collect();
        let (mut clamped, mut exact) = (row.clone(), row);
        steer_rows_ref(&mut clamped, h, 1, h, &[1], &table, &table, &[0, 9], k_max);
        steer_rows_ref(&mut exact, h, 1, h, &[1], &table, &table, &[0, 2], k_max);
        assert_eq!(clamped, exact);
    }

    #[test]
    fn add_rows_adds_only_enabled_variant_rows() {
        let (h, ld_y, ld_d) = (16, 24, 20);
        let mut seed = 21;
        let mut y: Vec<u16> = (0..4 * ld_y).map(|_| f32_to_bf16_rne(lcg(&mut seed))).collect();
        let d: Vec<f32> = (0..4 * ld_d).map(|_| lcg(&mut seed) * 0.25).collect();
        let original = y.clone();
        // Row 0 base, row 1 variant 1 (on), row 2 variant 2 (off), row 3 negative.
        add_rows_ref(&mut y, ld_y, &d, ld_d, 4, h, &[0, 1, 2, -1], &[0, 1, 0]);
        for r in [0, 2, 3] {
            assert_eq!(y[r * ld_y..(r + 1) * ld_y], original[r * ld_y..(r + 1) * ld_y]);
        }
        for i in 0..h {
            let want = f32_to_bf16_rne(bf16_to_f32(original[ld_y + i]) + d[ld_d + i]);
            assert_eq!(y[ld_y + i], want);
        }
        assert_eq!(y[ld_y + h..2 * ld_y], original[ld_y + h..2 * ld_y]);
    }
}
