//! Live check of a Spark expert deployment against a CPU oracle.
//!
//! Sends one layer's routed rows to every Spark rank over the serving transport
//! and compares the rank-summed partials with the checkpoint math done on the
//! CPU: FP4 (E2M1) weights with E8M0 K32 scales, BF16 gate/up with the SwiGLU
//! clamp, route weight, MXFP8 re-quantization of the intermediate, then W2.
//! Inputs are random FP8 K32 wire rows, so the oracle sees exactly the values
//! the kernels do.
use crate::cli::ExpertProbeArgs;
use anyhow::{ensure, Context, Result};
use cuteafd_transport::{
    v41_expert::{V41Tp4Roce, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16},
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
    ExpertV2Dtype, ExpertV2SourceKind, TcpTransportConfig,
};
use std::collections::BTreeMap;
use std::os::unix::fs::FileExt;
use std::time::{Duration, Instant};

pub(crate) async fn run_expert_probe(args: ExpertProbeArgs) -> Result<()> {
    let catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    let shape = *catalog.routed_experts();
    let geometry = shape.geometry()?;
    cuteafd_core::set_expert_geometry(geometry)
        .map_err(|fixed| anyhow::anyhow!("expert geometry already {fixed:?}"))?;
    ensure!(args.layer < shape.layers, "layer {} is outside 0..{}", args.layer, shape.layers);
    let (hidden, topk, rows) = (shape.hidden, shape.topk, args.rows as usize);
    ensure!(rows > 0 && rows <= 4096, "rows must be 1..=4096");

    let mut rng = args.seed | 1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    // Wire rows: FP8 E4M3 payload (finite codes), then UE8M0 scales near 2^-4.
    let mut wire = Vec::with_capacity(rows * (hidden + hidden / 32));
    let mut input = vec![0f32; rows * hidden];
    for row in 0..rows {
        let scales: Vec<u8> = (0..hidden / 32).map(|_| 120 + (next() % 5) as u8).collect();
        for col in 0..hidden {
            let code = loop {
                let code = (next() & 0xff) as u8;
                if code & 0x7f != 0x7f {
                    break code;
                }
            };
            wire.push(code);
            input[row * hidden + col] = e4m3(code) * e8m0(scales[col / 32]);
        }
        wire.extend(&scales);
    }
    let mut routes = Vec::with_capacity(rows * topk);
    for row in 0..rows {
        let mut chosen: Vec<u32> = Vec::with_capacity(topk);
        while chosen.len() < topk {
            let expert = (next() % shape.experts as u64) as u32;
            if !chosen.contains(&expert) {
                chosen.push(expert);
            }
        }
        for expert in chosen {
            let weight = 0.05 + (next() % 1000) as f32 / 4000.0;
            routes.push(ExpertProtocolV2RouteEntry { row_index: row as u32, expert_id: expert, gate_weight: weight });
        }
    }
    let mut request = ExpertProtocolV2Request::new(
        args.seed,
        17,
        args.layer as u32,
        hidden as u32,
        ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        (0..rows as u32)
            .map(|row| ExpertProtocolV2RowDescriptor {
                row_id: u64::from(row),
                source_kind: ExpertV2SourceKind::Prefill,
                source_request_id: args.seed,
                token_position: u64::from(row),
                route_offset: row * topk as u32,
                route_count: topk as u32,
            })
            .collect(),
        routes.clone(),
        wire,
    )?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;

    let peers = args
        .peers
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()
        .context("--peers takes comma-separated HOST:PORT addresses")?;
    let config = TcpTransportConfig { timing: false, timeout: Duration::from_secs(60), max_frame_bytes: 64 << 20 };
    let executors: Vec<u64> = (1..=peers.len() as u64).collect();
    let mut client = V41Tp4Roce::new_ranks(&peers, &executors, args.capacity, config)?;
    let mut actual = vec![0f32; rows * hidden];
    let started = Instant::now();
    client
        .execute(&request, |_rank, first, payload| {
            for (index, pair) in payload.chunks_exact(2).enumerate() {
                let value = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
                actual[first as usize * hidden + index] += value;
            }
            Ok(())
        })
        .await?;
    let remote = started.elapsed();

    let expected = oracle(&catalog, args.layer, hidden, &input, &routes, rows)?;
    let (mut dot, mut na, mut nb, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (a, e) in actual.iter().zip(&expected) {
        let (a, e) = (f64::from(*a), f64::from(*e));
        dot += a * e;
        na += a * a;
        nb += e * e;
        diff += (a - e) * (a - e);
    }
    let cosine = dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE);
    let rel_l2 = diff.sqrt() / nb.sqrt().max(f64::MIN_POSITIVE);
    let pass = cosine > 0.9999 && rel_l2 < 0.01;
    println!(
        "{} layer {} rows {} ranks {}: cosine {cosine:.6} rel_l2 {rel_l2:.2e} remote {:.2} ms",
        if pass { "PASS" } else { "FAIL" },
        args.layer,
        rows,
        peers.len(),
        remote.as_secs_f64() * 1e3,
    );
    ensure!(pass, "Spark experts disagree with the CPU oracle");
    Ok(())
}

fn oracle(
    catalog: &cuteafd_loader::OfficialV41Catalog,
    layer: usize,
    hidden: usize,
    input: &[f32],
    routes: &[ExpertProtocolV2RouteEntry],
    rows: usize,
) -> Result<Vec<f32>> {
    let intermediate = catalog.routed_experts().intermediate;
    let mut by_expert: BTreeMap<u32, Vec<&ExpertProtocolV2RouteEntry>> = BTreeMap::new();
    for route in routes {
        by_expert.entry(route.expert_id).or_default().push(route);
    }
    let mut out = vec![0f32; rows * hidden];
    for (expert, routes) in by_expert {
        let prefix = format!("layers.{layer}.ffn.experts.{expert}");
        let w1 = dequantize(catalog, &format!("{prefix}.w1"), intermediate, hidden)?;
        let w3 = dequantize(catalog, &format!("{prefix}.w3"), intermediate, hidden)?;
        let w2 = dequantize(catalog, &format!("{prefix}.w2"), hidden, intermediate)?;
        for route in routes {
            let x = &input[route.row_index as usize * hidden..][..hidden];
            let mut mid: Vec<f32> = (0..intermediate)
                .map(|n| {
                    let gate = bf16(dot(x, &w1[n * hidden..][..hidden])).min(10.0);
                    let up = bf16(dot(x, &w3[n * hidden..][..hidden])).clamp(-10.0, 10.0);
                    bf16(gate / (1.0 + (-gate).exp()) * up * route.gate_weight)
                })
                .collect();
            mxfp8_roundtrip(&mut mid);
            let row = &mut out[route.row_index as usize * hidden..][..hidden];
            for (h, value) in row.iter_mut().enumerate() {
                *value += dot(&mid, &w2[h * intermediate..][..intermediate]);
            }
        }
    }
    Ok(out)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum::<f64>() as f32
}

/// Row-major [rows, cols] from packed E2M1 (low nibble first) and E8M0 K32 scales.
fn dequantize(
    catalog: &cuteafd_loader::OfficialV41Catalog,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let read = |suffix: &str, bytes: usize| -> Result<Vec<u8>> {
        let tensor = catalog.tensor(&format!("{name}.{suffix}"))?;
        ensure!(tensor.metadata.byte_length as usize == bytes, "{name}.{suffix} has an unexpected size");
        let mut data = vec![0u8; bytes];
        std::fs::File::open(catalog.snapshot().join(&tensor.shard))?
            .read_exact_at(&mut data, tensor.metadata.byte_offset)?;
        Ok(data)
    };
    let packed = read("weight", rows * cols / 2)?;
    let scales = read("scale", rows * cols / 32)?;
    let mut out = vec![0f32; rows * cols];
    for (index, value) in out.iter_mut().enumerate() {
        let code = (packed[index / 2] >> (4 * (index % 2))) & 15;
        let magnitude = E2M1[usize::from(code & 7)];
        let row = index / cols;
        *value = if code & 8 != 0 { -magnitude } else { magnitude }
            * e8m0(scales[row * (cols / 32) + (index % cols) / 32]);
    }
    Ok(out)
}

fn e8m0(code: u8) -> f32 {
    f32::from_bits(u32::from(code) << 23)
}

fn e4m3(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((code >> 3) & 15);
    let mantissa = f32::from(code & 7);
    sign * if exponent == 0 {
        mantissa / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    }
}

fn bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// Quantizes each 32-value block to E4M3 under a power-of-two scale (amax
/// floor 1e-4), as the kernel does before FC2, and replaces it with the result.
fn mxfp8_roundtrip(values: &mut [f32]) {
    let grid: Vec<f32> = (0u8..0x7f).map(e4m3).collect();
    for block in values.chunks_mut(32) {
        let amax = block.iter().fold(1e-4f32, |max, v| max.max(v.abs()));
        let scale = 2f32.powf((amax / 448.0).log2().ceil());
        for value in block {
            let scaled = (*value / scale).abs();
            let index = grid.partition_point(|&g| g < scaled).min(grid.len() - 1);
            let nearest = if index > 0 && (scaled - grid[index - 1]) <= (grid[index] - scaled) {
                // Ties go to the even code.
                if (scaled - grid[index - 1]) == (grid[index] - scaled) && index % 2 == 0 { index } else { index - 1 }
            } else {
                index
            };
            *value = grid[nearest].copysign(*value) * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_formats() {
        assert_eq!(e4m3(0x7e), 448.0);
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert_eq!(e8m0(127), 1.0);
        assert_eq!(bf16(1.0 + 2f32.powi(-9)), 1.0);
        let mut block = vec![1.0f32; 32];
        block[0] = 3.3;
        mxfp8_roundtrip(&mut block);
        assert_eq!(block[1], 1.0);
        assert!((block[0] - 3.25).abs() < 1e-6, "{}", block[0]);
    }
}
