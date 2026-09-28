//! CPU oracle for EXL3 routed experts: each trellis tile is decoded exactly
//! as the kernels do (MCG codebook, tail-biting 16-bit windows, MMA fragment
//! order) and applied with the exllamav3 transform
//! `y = had128(had128(x * suh) @ W) * svh` in FP32, so the reference uses the
//! checkpoint's own dequantized weights without materializing them.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{OfficialV41Catalog, V41Exl3ProjectionKind};
use cuteafd_transport::ExpertProtocolV2RouteEntry;
use std::collections::BTreeMap;
use std::os::unix::fs::FileExt;

const MCG_MULTIPLIER: u32 = 0xcbac_1fed;
const SWIGLU_LIMIT: f32 = 10.0;

/// `draft` selects dSpark stage `layer` (`mtp.{layer}`) instead of backbone layer `layer`.
pub(super) fn oracle(
    catalog: &OfficialV41Catalog,
    draft: bool,
    layer: usize,
    input: &[f32],
    routes: &[ExpertProtocolV2RouteEntry],
    rows: usize,
) -> Result<Vec<f32>> {
    let manifest = catalog.exl3().context("EXL3 oracle requires an EXL3 checkpoint")?;
    let hidden = catalog.routed_experts().hidden;
    let mut by_expert: BTreeMap<u32, Vec<&ExpertProtocolV2RouteEntry>> = BTreeMap::new();
    for route in routes {
        by_expert.entry(route.expert_id).or_default().push(route);
    }
    let experts: Vec<_> = by_expert.into_iter().collect();
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).min(experts.len().max(1));
    let partials = std::thread::scope(|scope| -> Result<Vec<Vec<f32>>> {
        let workers: Vec<_> = (0..threads)
            .map(|worker| {
                let experts = &experts;
                scope.spawn(move || -> Result<Vec<f32>> {
                    let mut out = vec![0f32; rows * hidden];
                    for (expert, routes) in experts.iter().skip(worker).step_by(threads) {
                        let name = |kind| manifest.naming.projection(draft, layer, *expert as usize, kind);
                        let gate = Projection::read(catalog, &name(V41Exl3ProjectionKind::Gate))?;
                        let up = Projection::read(catalog, &name(V41Exl3ProjectionKind::Up))?;
                        let down = Projection::read(catalog, &name(V41Exl3ProjectionKind::Down))?;
                        let xs: Vec<&[f32]> = routes
                            .iter()
                            .map(|route| &input[route.row_index as usize * hidden..][..hidden])
                            .collect();
                        let g = gate.apply(&xs)?;
                        let u = up.apply(&xs)?;
                        let mids: Vec<Vec<f32>> = g
                            .iter()
                            .zip(&u)
                            .map(|(g, u)| {
                                g.iter()
                                    .zip(u)
                                    .map(|(&g, &u)| {
                                        let g = g.min(SWIGLU_LIMIT);
                                        let u = u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
                                        g / (1.0 + (-g).exp()) * u
                                    })
                                    .collect()
                            })
                            .collect();
                        let ys = down.apply(&mids.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
                        for (route, y) in routes.iter().zip(ys) {
                            let row = &mut out[route.row_index as usize * hidden..][..hidden];
                            for (value, y) in row.iter_mut().zip(y) {
                                *value += route.gate_weight * y;
                            }
                        }
                    }
                    Ok(out)
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().map_err(|_| anyhow::anyhow!("EXL3 oracle worker panicked"))?)
            .collect()
    })?;
    let mut out = vec![0f32; rows * hidden];
    for partial in partials {
        for (value, part) in out.iter_mut().zip(partial) {
            *value += part;
        }
    }
    Ok(out)
}

/// One projection's stored tensors: trellis tiles `[in/16, out/16, 16*bits]`
/// and the FP16 input/output sign-scale vectors.
struct Projection {
    input: usize,
    output: usize,
    bits: usize,
    trellis: Vec<u8>,
    suh: Vec<f32>,
    svh: Vec<f32>,
}

impl Projection {
    fn read(catalog: &OfficialV41Catalog, name: &str) -> Result<Self> {
        Self::from_tensors(name, |tensor| {
            let tensor = catalog.tensor(tensor)?;
            Ok((catalog.snapshot().join(&tensor.shard), tensor.metadata.byte_offset, tensor.metadata.byte_length))
        })
    }

    /// `locate` maps a tensor name to its (file, byte offset, byte length).
    fn from_tensors(
        name: &str,
        locate: impl Fn(&str) -> Result<(std::path::PathBuf, u64, u64)>,
    ) -> Result<Self> {
        let read = |suffix: &str| -> Result<Vec<u8>> {
            let (path, offset, length) = locate(&format!("{name}.{suffix}"))?;
            let mut data = vec![0u8; usize::try_from(length)?];
            std::fs::File::open(path)?
                .read_exact_at(&mut data, offset)
                .with_context(|| format!("reading {name}.{suffix}"))?;
            Ok(data)
        };
        let mcg = read("mcg")?;
        ensure!(
            mcg.len() == 4 && u32::from_le_bytes(mcg[..4].try_into()?) == MCG_MULTIPLIER,
            "{name} is not an MCG trellis"
        );
        let halves = |bytes: Vec<u8>| -> Vec<f32> {
            bytes.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()
        };
        let suh = halves(read("suh")?);
        let svh = halves(read("svh")?);
        let trellis = read("trellis")?;
        let (input, output) = (suh.len(), svh.len());
        ensure!(input % 128 == 0 && output % 128 == 0, "{name} is not H128 aligned");
        let bits = trellis.len() * 8 / (input * output);
        ensure!(
            (1..=8).contains(&bits) && trellis.len() == input * output * bits / 8,
            "{name} trellis size disagrees with its rotations"
        );
        Ok(Self { input, output, bits, trellis, suh, svh })
    }

    /// `had128(had128(x * suh) @ W) * svh` for every input row.
    fn apply(&self, xs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        let rotated: Vec<Vec<f32>> = xs
            .iter()
            .map(|x| {
                let mut a: Vec<f32> = x.iter().zip(&self.suh).map(|(x, s)| x * s).collect();
                hadamard128(&mut a);
                a
            })
            .collect();
        let mut ys = vec![vec![0f32; self.output]; xs.len()];
        let tile_bytes = 32 * self.bits;
        let (k_tiles, n_tiles) = (self.input / 16, self.output / 16);
        let mut block = [[0f32; 16]; 16];
        for k_tile in 0..k_tiles {
            for n_tile in 0..n_tiles {
                let offset = (k_tile * n_tiles + n_tile) * tile_bytes;
                decode_tile(&self.trellis[offset..offset + tile_bytes], self.bits, &mut block);
                for (a, y) in rotated.iter().zip(&mut ys) {
                    let a = &a[k_tile * 16..][..16];
                    let y = &mut y[n_tile * 16..][..16];
                    for (row, &value) in block.iter().zip(a) {
                        for (out, weight) in y.iter_mut().zip(row) {
                            *out += value * weight;
                        }
                    }
                }
            }
        }
        for y in &mut ys {
            hadamard128(y);
            for (value, scale) in y.iter_mut().zip(&self.svh) {
                *value *= scale;
            }
        }
        Ok(ys)
    }
}

/// Decodes one 16x16 tile (`block[k][n]`) from `16 * bits` little-endian
/// int16 words: the 256 weights read overlapping 16-bit windows of a cyclic
/// bitstream, and lane/weight order follows the MMA fragment layout.
fn decode_tile(tile: &[u8], bits: usize, block: &mut [[f32; 16]; 16]) {
    let width = 8 * bits;
    let word = |index: usize| -> u64 {
        let i = (index % width) * 4;
        u64::from(u32::from_le_bytes([tile[i], tile[i + 1], tile[i + 2], tile[i + 3]]))
    };
    for lane in 0..32 {
        let row0 = (lane % 4) * 2;
        let rows = [row0, row0 + 1, row0 + 8, row0 + 9];
        let (col0, parity) = (lane / 8, (lane >> 2) & 1);
        for weight in 0..8 {
            let end_bit = (lane * 8 + weight + 257) * bits;
            let start_bit = end_bit - 16;
            let (first, last) = (start_bit / 32, (end_bit - 1) / 32);
            let shift = (last + 1) * 32 - end_bit;
            let merged = (word(first) << 32) | word(last);
            let window = ((merged >> shift) & 0xffff) as u32;
            let column = 2 * if weight < 4 { col0 } else { col0 + 4 } + parity;
            block[rows[weight % 4]][column] = decode_mcg(window);
        }
    }
}

/// MCG codebook: multiply, mask/xor into two FP16 halves, add them in FP16.
fn decode_mcg(window: u32) -> f32 {
    let x = (window.wrapping_mul(MCG_MULTIPLIER) & 0x8fff_8fff) ^ 0x3b60_3b60;
    let sum = f16_to_f32(x as u16) + f16_to_f32((x >> 16) as u16);
    f16_to_f32(f32_to_f16(sum))
}

/// In-place normalized Sylvester Hadamard over each 128-value block.
fn hadamard128(values: &mut [f32]) {
    let scale = 1.0 / 128f32.sqrt();
    for block in values.chunks_exact_mut(128) {
        let mut span = 1;
        while span < 128 {
            for start in (0..128).step_by(2 * span) {
                for i in start..start + span {
                    let (a, b) = (block[i], block[i + span]);
                    block[i] = a + b;
                    block[i + span] = a - b;
                }
            }
            span *= 2;
        }
        for value in block {
            *value *= scale;
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// Round-to-nearest-even FP32 -> FP16 (finite inputs within FP16 range).
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let magnitude = value.abs();
    if magnitude.is_nan() {
        return sign | 0x7e00;
    }
    if magnitude >= 65520.0 {
        return sign | 0x7c00;
    }
    // Scale into the FP16 integer grid of the value's binade and round there.
    let binade = |value: f32| ((value.to_bits() >> 23) & 0xff) as i32 - 127;
    let exponent = if magnitude < 2f32.powi(-14) { -14 } else { binade(magnitude) };
    let quantum = 2f32.powi(exponent - 10);
    let steps = (magnitude / quantum).round_ties_even();
    let rounded = steps * quantum;
    if rounded == 0.0 {
        return sign;
    }
    let exponent = if rounded < 2f32.powi(-14) { -15 } else { binade(rounded) };
    if exponent < -14 {
        return sign | (rounded / 2f32.powi(-24)) as u16;
    }
    let mantissa = (rounded / 2f32.powi(exponent) - 1.0) * 1024.0;
    sign | (((exponent + 15) as u16) << 10) | mantissa as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decoder cross-check against independent weights: a V4.1 EXL3 projection
    /// applied through this oracle must track the official FP4 projection it
    /// was quantized from (K3/K4 trellis: cosine well above 0.99).
    #[test]
    #[ignore = "requires CUTEAFD_EXL3_SNAPSHOT (V4.1 EXL3) and CUTEAFD_NATIVE_SNAPSHOT (official V4.1)"]
    fn exl3_projection_tracks_the_official_weights() -> Result<()> {
        // Located through the raw index and headers: only the trellis bytes
        // matter here, not the snapshot's packaging metadata.
        let exl3 = std::path::PathBuf::from(std::env::var("CUTEAFD_EXL3_SNAPSHOT")?);
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(exl3.join("model.safetensors.index.json"))?)?;
        let mut tensors = BTreeMap::new();
        let shards: std::collections::BTreeSet<_> = index["weight_map"]
            .as_object()
            .context("index weight_map")?
            .values()
            .filter_map(|shard| shard.as_str())
            .collect();
        for shard in shards {
            for tensor in cuteafd_loader::read_safetensors_metadata(&exl3.join(shard))? {
                tensors.insert(tensor.name.clone(), (exl3.join(shard), tensor.byte_offset, tensor.byte_length));
            }
        }
        let locate = |name: &str| tensors.get(name).cloned().with_context(|| format!("missing {name}"));
        let native = cuteafd_loader::read_expert_catalog(std::path::Path::new(&std::env::var("CUTEAFD_NATIVE_SNAPSHOT")?))?;
        let shape = *native.routed_experts();
        let naming = cuteafd_loader::V41Exl3Naming::CheckpointNative;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        for (layer, expert) in [(3usize, 17usize), (30, 200)] {
            for (kind, stem) in [
                (V41Exl3ProjectionKind::Gate, "w1"),
                (V41Exl3ProjectionKind::Up, "w3"),
                (V41Exl3ProjectionKind::Down, "w2"),
            ] {
                let projection = Projection::from_tensors(&naming.projection(false, layer, expert, kind), locate)?;
                let (rows, cols) = (projection.output, projection.input);
                let reference = super::super::dequantize(
                    &native,
                    &format!("layers.{layer}.ffn.experts.{expert}.{stem}"),
                    rows,
                    cols,
                )?;
                assert_eq!((rows, cols), if stem == "w2" { (shape.hidden, shape.intermediate) } else { (shape.intermediate, shape.hidden) });
                let x: Vec<f32> = (0..cols).map(|_| random()).collect();
                let y = projection.apply(&[&x])?.remove(0);
                let expected: Vec<f32> = (0..rows).map(|r| super::super::dot(&x, &reference[r * cols..][..cols])).collect();
                let dot: f64 = y.iter().zip(&expected).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
                let norm = |v: &[f32]| v.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
                let cosine = dot / (norm(&y) * norm(&expected));
                println!("layer {layer} expert {expert} {stem} K{}: cosine {cosine:.5}", projection.bits);
                assert!(cosine > 0.98, "EXL3 {stem} disagrees with the official weights: {cosine}");
            }
        }
        Ok(())
    }

    #[test]
    fn half_conversions_round_trip_every_finite_code() {
        for code in 0u16..=0xffff {
            if (code >> 10) & 0x1f == 0x1f {
                continue;
            }
            let value = f16_to_f32(code);
            let back = f32_to_f16(value);
            assert!(back == code || (value == 0.0 && back & 0x7fff == 0), "{code:#06x} -> {value} -> {back:#06x}");
        }
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00); // tie to even
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
    }

    #[test]
    fn hadamard_is_orthonormal_and_self_inverse() {
        let original: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut values = original.clone();
        hadamard128(&mut values);
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();
        assert!((norm(&values) - norm(&original)).abs() < 1e-3);
        hadamard128(&mut values);
        for (a, b) in values.iter().zip(&original) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    /// Mirrors sparkinfer's `_pack_native_tiles`/`_decode_lane` reference: pack
    /// 256 K2 symbols, then every decoded weight equals the MCG value of its
    /// cyclic 16-bit history at the fragment position.
    #[test]
    fn tile_decode_matches_the_reference_packing() {
        let bits = 2;
        let symbols: Vec<u32> = (0..256u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 7) & 3).collect();
        // Bitstream: symbol i occupies bits [i*bits, (i+1)*bits), MSB first,
        // grouped into 16-bit words whose pairs are swapped into LE u32 words.
        let mut stream = vec![0u8; 256 * bits];
        for (i, &symbol) in symbols.iter().enumerate() {
            for b in 0..bits {
                stream[i * bits + b] = ((symbol >> (bits - 1 - b)) & 1) as u8;
            }
        }
        let halfwords: Vec<u16> = stream
            .chunks(16)
            .map(|chunk| chunk.iter().fold(0u16, |word, &bit| (word << 1) | u16::from(bit)))
            .collect();
        let mut tile = Vec::new();
        for pair in halfwords.chunks(2) {
            tile.extend_from_slice(&pair[1].to_le_bytes());
            tile.extend_from_slice(&pair[0].to_le_bytes());
        }
        let mut block = [[0f32; 16]; 16];
        decode_tile(&tile, bits, &mut block);
        // Reference: state of weight j is the 16-bit history ending at symbol j
        // (cyclic), i.e. symbols j-7..=j for K2, oldest first.
        for lane in 0..32 {
            let row0 = (lane % 4) * 2;
            let rows = [row0, row0 + 1, row0 + 8, row0 + 9];
            for weight in 0..8 {
                let j = lane * 8 + weight;
                let state = (0..8).fold(0u32, |state, lag| {
                    (state << bits) | symbols[(j + 256 + 1 + lag - 8) % 256]
                });
                let column = 2 * if weight < 4 { lane / 8 } else { lane / 8 + 4 } + ((lane >> 2) & 1);
                assert_eq!(block[rows[weight % 4]][column], decode_mcg(state), "lane {lane} weight {weight}");
            }
        }
    }
}
