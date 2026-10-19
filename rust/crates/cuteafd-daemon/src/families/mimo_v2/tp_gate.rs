//! Real-Spark MXFP4 gate: source-native reference and the serving rank reducer.
use anyhow::{ensure, Context, Result};
use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
use cuteafd_loader::formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors, Fp8Projection};
use cuteafd_transport::{expert::SparkWorld, worker_selection::ModelAdmission,
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
    ExpertV2Dtype, ExpertV2SourceKind, TcpTransportConfig,
    protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
use crate::shared::{memory::{DeviceAllocation, LoadStream}, placement::{spark_rings, SparkAdmission},
    spark_intake::SparkLink};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

const H: usize = 6144;
const I: usize = 2048;
const K: usize = 8;
const OUTPUTS: [usize; 16] = [0, 1, 31, 32, 127, 255, 511, 1023, 1535, 2047, 3071, 4095, 5119, 6000, 6142, 6143];

fn bf16(value: f64) -> f64 {
    let bits = (value as f32).to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000) as f64
}

fn element(weights: &[u8], scales: &[u8], columns: usize, row: usize, col: usize) -> Result<f64> {
    let packed = weights[row * (columns / 2) + col / 2];
    let code = (packed >> (4 * (col % 2))) & 15;
    let exponent = scales[row * (columns / 32) + col / 32];
    ensure!(exponent != 255, "MXFP4 reference encountered a NaN scale");
    let value = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(code & 7) as usize];
    Ok(if code & 8 == 0 { value } else { -value } * 2f64.powi(exponent as i32 - 127))
}

fn input(row: usize) -> Vec<(usize, f64)> {
    (0..8).map(|slot| ((slot * 769 + 31) % H,
        if (row + slot) % 2 == 0 { 0.5 } else { -0.25 })).collect()
}

fn expert(layer: usize, row: usize, slot: usize) -> usize {
    (layer * 17 + (row % 2) * 8 + slot * 37) % 384
}

fn request(layer: usize, rows: usize) -> Result<ExpertProtocolV2Request> {
    let mut wire = vec![0u8; rows * H * 2];
    for row in 0..rows {
        for (col, value) in input(row) {
            let bits = ((value as f32).to_bits() >> 16) as u16;
            wire[(row * H + col) * 2..][..2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    let routes = (0..rows).flat_map(|row| (0..K).map(move |slot| ExpertProtocolV2RouteEntry {
        row_index: row as u32, expert_id: expert(layer, row, slot) as u32,
        gate_weight: (slot + 1) as f32 / 36.,
    })).collect();
    let descriptors = (0..rows).map(|row| ExpertProtocolV2RowDescriptor {
        row_id: row as u64, source_kind: ExpertV2SourceKind::Prefill,
        source_request_id: 1, token_position: row as u64, route_offset: (row * K) as u32,
        route_count: K as u32,
    }).collect();
    let mut request = ExpertProtocolV2Request::new_bytes(layer as u64 + rows as u64, 521,
        layer as u32, H as u32, ExpertV2Dtype::Bf16, descriptors, routes, wire.into())?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

/// Independent full-intermediate FFN, not a sum of the candidate's slices.
/// Sparse nonzero BF16 inputs make FP64 source math cheap without a GPU oracle.
fn reference(tensors: &Fp8ExpertTensors, layer: usize) -> Result<Vec<Vec<f64>>> {
    let mut result = vec![vec![0.; OUTPUTS.len()]; 2];
    for (row, output) in result.iter_mut().enumerate() {
        for slot in 0..K {
            let id = expert(layer, row, slot);
            let mut projections = Vec::new();
            for projection in Fp8Projection::ALL {
                let (w, s) = tensors.projection_bytes(projection);
                let (mut weights, mut scales) = (vec![0; w], vec![0; s]);
                tensors.read_projection_once(layer, id, projection, &mut weights, &mut scales)?;
                projections.push((weights, scales));
            }
            let mut hidden = vec![0.; I];
            for (col, hidden) in hidden.iter_mut().enumerate() {
                let mut gu = [0.; 2];
                for projection in 0..2 {
                    for (index, value) in input(row) {
                        gu[projection] += element(&projections[projection].0,
                            &projections[projection].1, H, col, index)? * value;
                    }
                }
                // The package's calibrated activation is silu_bf16_product.
                let (g, u) = (bf16(gu[0]), bf16(gu[1]));
                *hidden = bf16(bf16(g / (1. + (-g).exp())) * u);
            }
            for (&column, output) in OUTPUTS.iter().zip(output.iter_mut()) {
                let mut sum = 0.;
                for (col, &value) in hidden.iter().enumerate() {
                    sum += element(&projections[2].0, &projections[2].1, I, column, col)? * value;
                }
                *output += sum * ((slot + 1) as f32 / 36.) as f64;
            }
        }
    }
    Ok(result)
}

fn error(actual: &[f64], expected: &[f64]) -> Result<(f64, f64)> {
    ensure!(actual.len() == expected.len() && !actual.is_empty(), "reference shape");
    ensure!(actual.iter().chain(expected).all(|v| v.is_finite()), "nonfinite routed output");
    let norm = expected.iter().map(|v| v * v).sum::<f64>();
    let actual_norm = actual.iter().map(|v| v * v).sum::<f64>();
    ensure!(norm > 1e-20 && actual_norm > 1e-20, "degenerate routed output");
    let relative = (actual.iter().zip(expected).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / norm).sqrt();
    let cosine = actual.iter().zip(expected).map(|(a, b)| a * b).sum::<f64>() / (norm * actual_norm).sqrt();
    Ok((relative, cosine))
}

#[test]
fn source_reference_decodes_nibbles_scales_and_rejects_missing_sum() -> Result<()> {
    let weights = [0xf2; 16];
    assert_eq!(element(&weights, &[128], 32, 0, 0)?, 2.);
    assert_eq!(element(&weights, &[128], 32, 0, 1)?, -12.);
    assert!(element(&weights, &[255], 32, 0, 0).is_err());
    assert_eq!(bf16(1.003), 1.);
    let (relative, cosine) = error(&[1., 2., 3.], &[1., 2., 3.])?;
    assert!(relative == 0. && (cosine - 1.).abs() < 1e-12);
    assert!(error(&[0.; 3], &[1., 2., 3.]).is_err());
    assert!(error(&[0.8, 1.6, 2.4], &[1., 2., 3.])?.0 > 0.05);
    Ok(())
}

#[test]
#[ignore = "requires sealed MiMo MXFP4 packages and six real Spark workers via golden-run"]
fn native_mimo_tp_rank_sum() -> Result<()> {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    cuteafd_core::set_expert_geometry(ExpertGeometry::MIMO_V26_PRO)
        .map_err(|_| anyhow::anyhow!("rank-sum process already selected another expert geometry"))?;
    let snapshot = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?);
    let layers = std::env::var("CUTEAFD_GOLDEN_EXPERT_LAYERS")?.split(',')
        .map(str::parse).collect::<std::result::Result<Vec<usize>, _>>()?;
    ensure!(layers.len() >= 6 && layers.windows(2).all(|w| w[1] == w[0] + 1),
        "rank-sum gate needs at least six consecutive absolute routed layers");
    let peers = std::env::var("CUTEAFD_GOLDEN_PEERS")?.split(',')
        .map(str::parse).collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(peers.len() == 6, "rank-sum gate requires TP6");
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    let tensors = catalog.fp8().context("rank-sum gate requires native MXFP4")?;
    ensure!(tensors.format() == ExpertFormat::Mxfp4 && tensors.shape().hidden == H
        && tensors.shape().intermediate == I && tensors.shape().experts == 384, "MiMo Pro gate geometry");
    let mut references = BTreeMap::new();
    for &layer in &layers { references.insert(layer, reference(&tensors.for_layer(layer)?, layer)?); }
    // SAFETY: the official runner supplies the verified sealed native artifact.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? });
    library.cuda_set_device(0)?;
    let config = TcpTransportConfig::default();
    SparkAdmission::plan(&snapshot, ExpertGeometry::MIMO_V26_PRO, Some(&catalog),
        &cuteafd_loader::placement::SparkWorkingSet { rtx_layers: vec![], spark_layers: layers.clone(),
            spark_ranks: 6, prefill_lanes: 1 }, 6, None,
        ModelAdmission { capacity: 16, compiled_capacity: 4096,
            endpoints: vec![SparkEndpoint { capacity: 16, ingress: SparkIngress::Bf16 }],
            rings: spark_rings(config.max_frame_bytes)? })?.select(&peers)?;
    let zero = DeviceAllocation::new(&library, 16 * H * 2)?;
    let output = DeviceAllocation::new(&library, 16 * H * 2)?;
    library.cuda_zero_bytes(zero.buffer, zero.buffer.bytes)?;
    let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
    let mut link = SparkLink::new(&library, SparkWorld::canonical(6)?, &peers, 16, config, H * 2)?;
    link.enable_terminal_ownership(library.clone())?;
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let result = runtime.block_on(async {
        link.bootstrap(&request(layers[0], 16)?)?;
        let out = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_OUT")?);
        let mut metrics = Vec::new();
        for &layer in &layers {
            for rows in [1, 16] {
                let request = request(layer, rows)?;
                let wave = link.dispatch(&request)?;
                link.receive(wave, rows, stream.raw).await?;
                // SAFETY: full-width buffers and the rank planes remain owned
                // until the stream drains; receive orders all six planes first.
                unsafe { link.reduce(zero.buffer.ptr.cast(), output.buffer.ptr.cast(), rows, stream.raw)?; }
                // SAFETY: stream and all queued storage remain live here.
                unsafe { library.cuda_stream_synchronize(stream.raw)?; }
                let mut bytes = vec![0; output.buffer.bytes];
                library.copy_d2h(&mut bytes, output.buffer)?;
                let mut actual = Vec::new();
                let mut expected = Vec::new();
                for row in 0..rows {
                    for (sample, &col) in OUTPUTS.iter().enumerate() {
                        let index = (row * H + col) * 2;
                        actual.push(f32::from_bits((u16::from_le_bytes([bytes[index], bytes[index + 1]]) as u32) << 16) as f64);
                        expected.push(references[&layer][row % 2][sample]);
                    }
                }
                let (relative, cosine) = error(&actual, &expected)?;
                ensure!(relative <= 0.05 && cosine >= 0.999,
                    "rank-sum layer={layer} rows={rows}: relative L2={relative} cosine={cosine}");
                eprintln!("rank-sum layer={layer} rows={rows} relative_l2={relative:.6} cosine={cosine:.8}");
                std::fs::write(out.join(format!("rank-sum-l{layer}-m{rows}.bf16")), &bytes[..rows * H * 2])?;
                metrics.push(serde_json::json!({"layer":layer,"rows":rows,"relative_l2":relative,"cosine":cosine}));
            }
        }
        std::fs::write(out.join("rank-sum.json"), serde_json::to_vec_pretty(&metrics)?)?;
        Ok::<_, anyhow::Error>(())
    });
    let quiesce = link.terminal_quiesce();
    let copies = link.terminal_drain_copies();
    // SAFETY: allocations, stream and intake remain alive on every result path.
    crate::shared::decode_graph::fatal_drain(unsafe { library.cuda_stream_synchronize(stream.raw) }
        .map_err(anyhow::Error::from), "MiMo rank-sum gate");
    quiesce?;
    copies?;
    link.terminal_release()?;
    result
}
