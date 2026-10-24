/*
MIT License

Copyright (c) 2026 Turquoise Bay AI Pty Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Ported from Hugh Madden, mimo26f-afd v1.3.0, crates/mimo26-image/src/.
//! Pillow `Image.resize(size, Image.BICUBIC)` for 8-bit RGB (Resample.c).
//!
//! Coefficients are computed in f64 with Pillow's exact operation order
//! (`precompute_coeffs`), normalized per output pixel, then converted to
//! 22-bit fixed point (`PRECISION_BITS = 32 - 8 - 2`) rounding half away from
//! zero. The horizontal pass runs first over only the input rows the vertical
//! pass needs; each pass rounds (`+ 1 << 21`) and clips to 0..255. A pass is
//! skipped when its dimension is unchanged, as in `ImagingResampleInner`.

use super::RgbImage;

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn bicubic(x: f64) -> f64 {
    // Pillow bicubic_filter with a = -0.5.
    const A: f64 = -0.5;
    let x = if x < 0.0 { -x } else { x };
    if x < 1.0 {
        return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    }
    if x < 2.0 {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * A;
    }
    0.0
}

/// Per-output-pixel taps: first input index, fixed-point weights.
struct Coeffs {
    precision: u32,
    ksize: usize,
    bounds: Vec<(usize, usize)>, // (xmin, count)
    k: Vec<i32>,                 // out_size * ksize
}

/// Normalized bicubic taps per output pixel, in the operation order Pillow's
/// `precompute_coeffs` and torch's antialiased `_compute_indices_weights_aa` share (f64,
/// box from 0, C truncation), then fixed point rounding half away from zero: Pillow at 22
/// bits, torch at the most bits that keep every weight in int16.
fn precompute(in_size: usize, out_size: usize, torch: bool) -> Coeffs {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut weights = vec![0.0f64; out_size * ksize];
    let mut maximum = 0.0f64;
    for xx in 0..out_size {
        let center = scale * (xx as f64 + 0.5);
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as usize).min(in_size);
        let count = xmax.saturating_sub(xmin).min(ksize);
        let row = &mut weights[xx * ksize..xx * ksize + count];
        let mut total = 0.0;
        for (j, weight) in row.iter_mut().enumerate() {
            *weight = bicubic((j as f64 + xmin as f64 - center + 0.5) * (1.0 / filterscale));
            total += *weight;
        }
        if total != 0.0 { for weight in row { *weight /= total; maximum = maximum.max(*weight); } }
        bounds.push((xmin, count));
    }
    let mut precision = PRECISION_BITS;
    if torch {
        precision = 0;
        while precision < 22 && ((0.5 + maximum * (1u32 << (precision + 1)) as f64) as i32) < 1 << 15 {
            precision += 1;
        }
    }
    let k = weights.into_iter().map(|weight| {
        let value = weight * (1u32 << precision) as f64;
        (if value < 0.0 { value - 0.5 } else { value + 0.5 }) as i32
    }).collect();
    Coeffs { precision, ksize, bounds, k }
}

#[inline(always)]
fn clip8(v: i32, precision: u32) -> u8 {
    (v >> precision).clamp(0, 255) as u8
}

pub(crate) fn resize_bicubic(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    resize_bicubic_impl(img, width, height, false)
}

/// PyTorch CPU uint8 antialiased bicubic uses int16 coefficients with
/// per-axis adaptive precision (UpSampleKernel.cpp).
pub(crate) fn resize_bicubic_torch(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    resize_bicubic_impl(img, width, height, true)
}

fn resize_bicubic_impl(img: &RgbImage, width: u32, height: u32, torch: bool) -> RgbImage {
    let (iw, ih) = (img.width as usize, img.height as usize);
    let (ow, oh) = (width as usize, height as usize);
    if ow == 0 || oh == 0 {
        return RgbImage {
            width,
            height,
            data: Vec::new(),
        };
    }
    if iw == 0 || ih == 0 {
        // No input taps: Pillow's accumulator stays at the rounding bias -> 0.
        return RgbImage {
            width,
            height,
            data: vec![0; ow * oh * 3],
        };
    }
    // Tolerate an inconsistent buffer (public fields) without panicking.
    let padded;
    let src: &[u8] = if img.data.len() >= iw * ih * 3 {
        &img.data[..iw * ih * 3]
    } else {
        let mut v = img.data.clone();
        v.resize(iw * ih * 3, 0);
        padded = v;
        &padded
    };
    if iw == ow && ih == oh {
        return RgbImage {
            width,
            height,
            data: src.to_vec(),
        };
    }
    let need_h = ow != iw;
    let need_v = oh != ih;
    let cv = precompute(ih, oh, torch);
    let ybox_first = cv.bounds[0].0;
    let (lx, lc) = cv.bounds[oh - 1];
    let ybox_last = lx + lc;

    // Horizontal pass over input rows [ybox_first, ybox_last).
    let (tmp, tmp_w, row_off) = if need_h {
        let ch = precompute(iw, ow, torch);
        let rows = ybox_last - ybox_first;
        let mut t = vec![0u8; ow * rows * 3];
        for y in 0..rows {
            let s = &src[(y + ybox_first) * iw * 3..(y + ybox_first + 1) * iw * 3];
            let d = &mut t[y * ow * 3..(y + 1) * ow * 3];
            horizontal_row(s, d, &ch);
        }
        (t, ow, ybox_first)
    } else {
        (Vec::new(), iw, 0)
    };
    let hsrc: &[u8] = if need_h { &tmp } else { src };
    if !need_v {
        return RgbImage {
            width,
            height,
            data: hsrc.to_vec(),
        };
    }
    let mut out = vec![0u8; tmp_w * oh * 3];
    let rowlen = tmp_w * 3;
    let mut acc = vec![0i32; rowlen];
    for yy in 0..oh {
        let (ymin, cnt) = cv.bounds[yy];
        let ymin = ymin - row_off;
        let kk = &cv.k[yy * cv.ksize..yy * cv.ksize + cnt];
        acc.fill(1 << (cv.precision - 1));
        for (j, &kw) in kk.iter().enumerate() {
            let r = &hsrc[(ymin + j) * rowlen..(ymin + j + 1) * rowlen];
            for (a, &p) in acc.iter_mut().zip(r) {
                *a = a.wrapping_add((p as i32).wrapping_mul(kw));
            }
        }
        for (d, &a) in out[yy * rowlen..(yy + 1) * rowlen].iter_mut().zip(&acc) {
            *d = clip8(a, cv.precision);
        }
    }
    RgbImage {
        width,
        height,
        data: out,
    }
}

fn horizontal_row(s: &[u8], d: &mut [u8], c: &Coeffs) {
    for (xx, px) in d.chunks_exact_mut(3).enumerate() {
        let (xmin, cnt) = c.bounds[xx];
        let kk = &c.k[xx * c.ksize..xx * c.ksize + cnt];
        let base = 1i32 << (c.precision - 1);
        let (mut s0, mut s1, mut s2) = (base, base, base);
        for (x, &kw) in kk.iter().enumerate() {
            let p = (xmin + x) * 3;
            s0 = s0.wrapping_add((s[p] as i32).wrapping_mul(kw));
            s1 = s1.wrapping_add((s[p + 1] as i32).wrapping_mul(kw));
            s2 = s2.wrapping_add((s[p + 2] as i32).wrapping_mul(kw));
        }
        px[0] = clip8(s0, c.precision);
        px[1] = clip8(s1, c.precision);
        px[2] = clip8(s2, c.precision);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torch_uint8_antialiased_bicubic_matches_cpu_reference() {
        use sha2::{Digest, Sha256};
        // torch 2.8 CPU interpolate(mode="bicubic", align_corners=False, antialias=True).
        for (iw, ih, ow, oh, expected) in [
            (37,59,56,84,"387ecc751b91aa035cdd183852f8509164b22c1ede63719b3b0a77b7944e01e4"),
            (1008,672,812,532,"99d94ead507ea6846d130a34094233263b57182912358b19caf97870a3cc2dfb"),
            (19,31,56,56,"31b8d76da4ef9f557aac71bc3636b2d5ae0be4450c6f53e08938e3c8e71a990c"),
            (128,99,84,56,"52d8cab52651adb6a8332f7064ff1f96a0a2a7da0172ab1d792ad91d882ed44b"),
            (56,56,56,56,"d54bb962452bcc9bfafce83eea69a8df17c1a967c47b58a236df0e21d58a9b0d"),
            (1,1,56,56,"294cd6182c19878407fafefbb1e7d06cadbd725706a7bdb1cc234ed6b9b079d5"),
            (3,501,28,1008,"2a30d74e60e1e2c0f862ff2852cc6f3cb975e2996458bc534d181e65169aada1"),
        ] {
            let image = RgbImage { width:iw, height:ih,
                data:(0..iw as usize*ih as usize*3).map(|i| ((i*37+11)%256) as u8).collect() };
            let resized = resize_bicubic_torch(&image,ow,oh);
            assert_eq!(format!("{:x}",Sha256::digest(&resized.data)), expected, "{iw}x{ih} -> {ow}x{oh}");
        }
    }

    #[test]
    fn identity_and_degenerate_sizes() {
        let img = RgbImage {
            width: 2,
            height: 1,
            data: vec![1, 2, 3, 4, 5, 6],
        };
        assert_eq!(resize_bicubic(&img, 2, 1), img);
        assert!(resize_bicubic(&img, 0, 5).data.is_empty());
        let one = RgbImage {
            width: 1,
            height: 1,
            data: vec![9, 8, 7],
        };
        let up = resize_bicubic(&one, 3, 2);
        assert_eq!(up.data, [9, 8, 7].repeat(6));
    }

    #[test]
    fn weights_sum_to_one_in_fixed_point() {
        for ((i, o), torch) in [(10, 3), (3, 10), (1000, 999), (7, 1), (1, 7)].into_iter()
            .flat_map(|size| [(size, false), (size, true)]) {
            let c = precompute(i, o, torch);
            for xx in 0..o {
                let (_, cnt) = c.bounds[xx];
                let s: i64 = c.k[xx * c.ksize..xx * c.ksize + cnt]
                    .iter()
                    .map(|&v| v as i64)
                    .sum();
                assert!(
                    (s - (1 << c.precision)).abs() <= cnt as i64,
                    "{i}->{o} torch={torch} px {xx}: {s}"
                );
            }
        }
    }
}
