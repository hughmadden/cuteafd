//! FP8 (E4M3) decode copies of GLM 5.3 Flash coordinator weights.
//!
//! Three scale layouts, all FP32 with 128-wide K blocks:
//! - `Block`: the checkpoint's 128x128 grid `[N/128, K/128]` (MLA, dense and
//!   shared experts: the official FP8 release's own E4M3 bytes and scales).
//! - `Row128`: one scale per output row and 128-wide K block, `[N, K/128]`.
//! - `Channel`: one scale per output row (W8A16), stored repeated as `[N, K/128]`.
//!
//! Quantization of a BF16 weight: `s = amax / 448`, `q = e4m3_rn(w / s)`
//! (saturating), the same recipe as the checkpoint's own blocks.

/// E4M3 (fn: no infinities, 0x7F NaN) of `x`, round to nearest even, saturating at 448.
pub(crate) fn e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return 0x7F;
    }
    // [448, 464) rounds to 448; above that it would round to 480 (NaN): saturate.
    if a >= 464.0 {
        return sign | 0x7E;
    }
    if a < 0.015625 {
        // Subnormals: multiples of 2^-9; 8 * 2^-9 = 2^-6 encodes as the first normal (0x08).
        return sign | (a * 512.0).round_ties_even() as u8;
    }
    let bits = a.to_bits();
    let rounded = bits + 0x7_FFFF + ((bits >> 20) & 1);
    let exponent = ((rounded >> 23) & 0xFF) as i32 - 127;
    let mantissa = ((rounded >> 20) & 7) as u8;
    sign | (((exponent + 7) as u8) << 3) | mantissa
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum KdaFp8 {
    /// BF16 KDA projections (the checkpoint's own precision).
    Off,
    /// One scale per output row.
    Channel,
    /// One scale per output row and 128-wide K block.
    Row128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    Block,
    Row128,
    Channel,
}

fn bf16(bytes: &[u8], i: usize) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])) << 16)
}

/// Quantizes a BF16 `[n, k]` weight; returns E4M3 bytes and FP32 scales in `layout`.
pub(crate) fn quantize(weight: &[u8], n: usize, k: usize, layout: Layout) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(weight.len(), n * k * 2);
    assert!(k % 128 == 0 && (layout != Layout::Block || n % 128 == 0));
    let kb = k / 128;
    let (scale_rows, rows_per_scale) = match layout {
        Layout::Block => (n / 128, 128),
        Layout::Row128 | Layout::Channel => (n, 1),
    };
    let mut values = vec![0u8; n * k];
    let mut scales = vec![0f32; scale_rows * kb];
    let threads = std::thread::available_parallelism().map_or(8, |p| p.get()).min(32);
    let per = scale_rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for ((values, scales), first) in values.chunks_mut(per * rows_per_scale * k)
            .zip(scales.chunks_mut(per * kb)).zip((0..scale_rows).step_by(per)) {
            scope.spawn(move || {
                for (local, srow) in (first..(first + per).min(scale_rows)).enumerate() {
                    let rows = srow * rows_per_scale..(srow + 1) * rows_per_scale;
                    // Scales for this scale row.
                    let mut s = vec![0f32; kb];
                    for row in rows.clone() {
                        for (b, sb) in s.iter_mut().enumerate() {
                            for c in b * 128..(b + 1) * 128 {
                                *sb = sb.max(bf16(weight, row * k + c).abs());
                            }
                        }
                    }
                    if layout == Layout::Channel {
                        let amax = s.iter().copied().fold(0f32, f32::max);
                        s.fill(amax);
                    }
                    for sb in &mut s {
                        *sb = if *sb > 0.0 { *sb / 448.0 } else { 1.0 };
                    }
                    for row in rows {
                        let out = &mut values[(row - first * rows_per_scale) * k..][..k];
                        for (c, q) in out.iter_mut().enumerate() {
                            *q = e4m3(bf16(weight, row * k + c) / s[c / 128]);
                        }
                    }
                    scales[local * kb..(local + 1) * kb].copy_from_slice(&s);
                }
            });
        }
    });
    (values, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(q: u8) -> f32 {
        let sign = if q & 0x80 != 0 { -1.0 } else { 1.0 };
        let (e, m) = (((q >> 3) & 0xF) as i32, (q & 7) as f32);
        sign * if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) }
    }

    #[test]
    fn e4m3_round_trips_and_rounds_to_nearest() {
        for q in 0u8..=0xFE {
            if q & 0x7F == 0x7F {
                continue;
            }
            assert_eq!(e4m3(decode(q)), q, "{q:#x} {}", decode(q));
        }
        assert_eq!(e4m3(500.0), 0x7E);
        assert_eq!(e4m3(450.0), 0x7E);
        // Halfway between 1.0 (0x38) and 1.125 (0x39) rounds to even (1.0).
        assert_eq!(e4m3(1.0625), 0x38);
        assert_eq!(e4m3(1.0626), 0x39);
        assert_eq!(e4m3(-0.0), 0x80);
    }

    #[test]
    fn row_layouts_bound_the_error() {
        let (n, k) = (4, 256);
        let w: Vec<f32> = (0..n * k).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01 * (1 + i / k) as f32).collect();
        let bytes: Vec<u8> = w.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect();
        for layout in [Layout::Row128, Layout::Channel] {
            let (q, s) = quantize(&bytes, n, k, layout);
            for i in 0..n * k {
                let x = bf16(&bytes, i);
                let y = decode(q[i]) * s[(i / k) * 2 + (i % k) / 128];
                assert!((x - y).abs() <= x.abs() * 0.0625 + 1e-6, "{layout:?} {i}: {x} vs {y}");
            }
        }
    }
}
