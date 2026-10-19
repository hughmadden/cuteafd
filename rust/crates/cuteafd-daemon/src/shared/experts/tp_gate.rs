//! Real-Spark source-native gate through the serving handshake and rank reducer.
use anyhow::{ensure, Result};
use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
use cuteafd_loader::formats::fp8_experts::ExpertFormat;
use cuteafd_transport::{expert::{SparkTopology, SparkWorld}, worker_selection::ModelAdmission,
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
    ExpertV2Dtype, ExpertV2SourceKind, TcpTransportConfig,
    protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
use crate::shared::{memory::{DeviceAllocation, LoadStream}, placement::{spark_rings, SparkAdmission},
    spark_intake::SparkLink};
use super::tp_reference::{error, expert, input, outputs, reference, route_weight};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

fn request(geometry: ExpertGeometry, layer: usize, rows: usize) -> Result<ExpertProtocolV2Request> {
    let (h, k) = (geometry.hidden as usize, geometry.topk as usize);
    let mut wire = vec![0u8; rows * h * 2];
    for row in 0..rows {
        for (col, value) in input(geometry, row) {
            let bits = ((value as f32).to_bits() >> 16) as u16;
            wire[(row * h + col) * 2..][..2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    let routes = (0..rows).flat_map(|row| (0..k).map(move |slot| ExpertProtocolV2RouteEntry {
        row_index: row as u32, expert_id: expert(geometry, layer, row, slot) as u32,
        gate_weight: route_weight(geometry, slot),
    })).collect();
    let descriptors = (0..rows).map(|row| ExpertProtocolV2RowDescriptor {
        row_id: row as u64, source_kind: ExpertV2SourceKind::Prefill,
        source_request_id: 1, token_position: row as u64, route_offset: (row * k) as u32,
        route_count: k as u32,
    }).collect();
    let mut request = ExpertProtocolV2Request::new_bytes(layer as u64 + rows as u64, 521,
        layer as u32, geometry.hidden, ExpertV2Dtype::Bf16, descriptors, routes, wire.into())?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

fn validate_source(catalog: &cuteafd_loader::OfficialV41Catalog, family: &str, format: &str)
    -> Result<ExpertGeometry> {
    let geometry = catalog.routed_experts().geometry()?;
    ensure!(geometry.family() == Some(family), "rank-sum family {family:?} disagrees with checkpoint {geometry:?}");
    let actual = if catalog.exl3().is_some() { "exl3" }
        else if let Some(tensors) = catalog.fp8() {
            match tensors.format() {
                ExpertFormat::Mxfp4 => "mxfp4", ExpertFormat::Fp8Block128 => "fp8",
                other => anyhow::bail!("rank-sum source decoder not qualified for {other:?}"),
            }
        } else {
            ensure!(geometry.family() == Some("v41") && catalog.nvfp4().is_none(),
                "rank-sum source decoder requires native V4.1, FP8, MXFP4 or EXL3");
            "mxfp4"
        };
    ensure!(actual == format, "rank-sum format {format:?} disagrees with checkpoint {actual}");
    ensure!(matches!(family, "v41" | "glm" | "mimo" | "mimop"),
        "rank-sum activation oracle is not qualified for family {family:?}");
    Ok(geometry)
}

fn checkpoint_world(catalog: &cuteafd_loader::OfficialV41Catalog, ranks: usize) -> Result<SparkWorld> {
    let native = catalog.exl3().is_none() && catalog.fp8().is_none();
    let topology = if native && ranks != 4 {
        Some(SparkTopology::new(u8::try_from(ranks)?, 1)?)
    } else {
        crate::shared::spark_topology::for_checkpoint(None, ranks,
            catalog.exl3().is_some() && catalog.routed_experts().geometry()?.family() == Some("v41"))
    };
    let world = crate::shared::spark_topology::world(topology, ranks)?;
    crate::shared::spark_topology::require_wire(world, catalog)?;
    Ok(world)
}

#[test]
fn checkpoint_world_preserves_generic_wire_and_native_tp6_ownership() -> Result<()> {
    use cuteafd_loader::plan::testing as fixture;
    let root = tempfile::tempdir()?;
    fixture::write_v41_snapshot(root.path());
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    assert_eq!(checkpoint_world(&catalog, 4)?, SparkWorld::canonical(4)?);
    assert_eq!(checkpoint_world(&catalog, 6)?, SparkWorld::owned(SparkTopology::NATIVE_TP6_EP1)?);
    let root = tempfile::tempdir()?;
    fixture::write_v41_exl3_snapshot(root.path(), 3);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    assert_eq!(checkpoint_world(&catalog, 3)?, SparkWorld::owned(SparkTopology::NATIVE_TP3_EP1)?);
    let root = tempfile::tempdir()?;
    let mut config = fixture::glm5_config();
    config["quantization_config"] = fixture::exl3_compact(4);
    let tensors = fixture::glm5_tensors(|name, n, k| (0..256).flat_map(|expert|
        fixture::exl3(&name.replace(".experts.0.", &format!(".experts.{expert}.")), n, k, 4)).collect());
    fixture::write_snapshot(root.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    assert_eq!(checkpoint_world(&catalog, 6)?, SparkWorld::canonical(6)?);
    Ok(())
}

#[test]
fn requests_use_checkpoint_geometry_and_two_repeated_oracle_rows() -> Result<()> {
    for geometry in [ExpertGeometry::DEEPSEEK_V41, ExpertGeometry::GLM5,
        ExpertGeometry::MIMO_V2_FLASH, ExpertGeometry::MIMO_V26_PRO] {
        let small = request(geometry, 3, 1)?;
        let wide = request(geometry, 3, 16)?;
        assert_eq!(small.header.hidden_dim, geometry.hidden);
        assert_eq!(wide.header.hidden_dim, geometry.hidden);
    }
    Ok(())
}

#[test]
#[ignore = "requires sealed packages and real Spark workers via golden-run; family, format and world are explicit"]
fn native_tp_rank_sum() -> Result<()> {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    let snapshot = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?);
    let family = std::env::var("CUTEAFD_GOLDEN_FAMILY")?;
    let format = std::env::var("CUTEAFD_GOLDEN_FORMAT")?;
    let world: usize = std::env::var("CUTEAFD_GOLDEN_SPARK_WORLD")?.parse()?;
    ensure!(matches!(world, 2 | 3 | 4 | 6), "rank-sum requires TP2/3/4/6");
    let layers = std::env::var("CUTEAFD_GOLDEN_EXPERT_LAYERS")?.split(',')
        .map(str::parse).collect::<std::result::Result<Vec<usize>, _>>()?;
    ensure!(layers.len() >= world && layers.windows(2).all(|w| w[1] == w[0] + 1),
        "rank-sum needs at least one full world of consecutive absolute routed layers");
    let peers = std::env::var("CUTEAFD_GOLDEN_PEERS")?.split(',')
        .map(str::parse).collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(peers.len() == world, "rank-sum world disagrees with launched peers");
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    let geometry = validate_source(&catalog, &family, &format)?;
    let spark_world = checkpoint_world(&catalog, world)?;
    cuteafd_core::set_expert_geometry(geometry)
        .map_err(|_| anyhow::anyhow!("rank-sum process already selected another expert geometry"))?;
    ensure!(layers.iter().all(|&l| l >= catalog.routed_experts().first_layer && l < geometry.layers as usize),
        "rank-sum layer is outside the checkpoint's routed set");
    let columns = outputs(geometry);
    let mut references = BTreeMap::new();
    for &layer in &layers { references.insert(layer, reference(&catalog, geometry, layer, &columns)?); }
    // SAFETY: the official runner supplies the verified sealed native artifact.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? });
    library.cuda_set_device(0)?;
    let config = TcpTransportConfig::default();
    SparkAdmission::plan(&snapshot, geometry, Some(&catalog),
        &cuteafd_loader::placement::SparkWorkingSet { rtx_layers: vec![], spark_layers: layers.clone(),
            spark_ranks: world, prefill_lanes: 1 }, world, spark_world.ownership_topology(),
        ModelAdmission { capacity: 16, compiled_capacity: 4096,
            endpoints: vec![SparkEndpoint { capacity: 16, ingress: SparkIngress::Bf16 }],
            rings: spark_rings(config.max_frame_bytes)? })?.select(&peers)?;
    let h = geometry.hidden as usize;
    let zero = DeviceAllocation::new(&library, 16 * h * 2)?;
    let output = DeviceAllocation::new(&library, 16 * h * 2)?;
    library.cuda_zero_bytes(zero.buffer, zero.buffer.bytes)?;
    let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
    let mut link = SparkLink::new(&library, spark_world, &peers, 16, config, h * 2)?;
    link.enable_terminal_ownership(library.clone())?;
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let result = runtime.block_on(async {
        link.bootstrap(&request(geometry, layers[0], 16)?)?;
        let out = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_OUT")?);
        let mut metrics = Vec::new();
        for &layer in &layers {
            for rows in [1, 16] {
                let request = request(geometry, layer, rows)?;
                let wave = link.dispatch(&request)?;
                link.receive(wave, rows, stream.raw).await?;
                // SAFETY: full-width buffers and rank planes stay owned until
                // the stream drains; receive orders every rank plane first.
                unsafe { link.reduce(zero.buffer.ptr.cast(), output.buffer.ptr.cast(), rows, stream.raw)?; }
                // SAFETY: stream and all queued storage remain live here.
                unsafe { library.cuda_stream_synchronize(stream.raw)?; }
                let mut bytes = vec![0; output.buffer.bytes];
                library.copy_d2h(&mut bytes, output.buffer)?;
                let mut actual = Vec::new();
                let mut expected = Vec::new();
                for row in 0..rows {
                    for (sample, &col) in columns.iter().enumerate() {
                        let index = (row * h + col) * 2;
                        actual.push(f32::from_bits((u16::from_le_bytes([bytes[index], bytes[index + 1]]) as u32) << 16) as f64);
                        expected.push(references[&layer][row % 2][sample]);
                    }
                }
                let (relative, cosine) = error(&actual, &expected)?;
                ensure!(relative <= 0.05 && cosine >= 0.999,
                    "rank-sum layer={layer} rows={rows}: relative L2={relative} cosine={cosine}");
                eprintln!("rank-sum family={family} format={format} world={world} layer={layer} rows={rows} relative_l2={relative:.6} cosine={cosine:.8}");
                std::fs::write(out.join(format!("rank-sum-l{layer}-m{rows}.bf16")), &bytes[..rows * h * 2])?;
                metrics.push(serde_json::json!({"family":family,"format":format,"world":world,
                    "layer":layer,"rows":rows,"columns":columns,"relative_l2":relative,"cosine":cosine}));
            }
        }
        std::fs::write(out.join("rank-sum.json"), serde_json::to_vec_pretty(&metrics)?)?;
        Ok::<_, anyhow::Error>(())
    });
    let quiesce = link.terminal_quiesce();
    let copies = link.terminal_drain_copies();
    // SAFETY: allocations, stream and intake remain alive on every result path.
    crate::shared::decode_graph::fatal_drain(unsafe { library.cuda_stream_synchronize(stream.raw) }
        .map_err(anyhow::Error::from), "shared rank-sum gate");
    quiesce?;
    copies?;
    link.terminal_release()?;
    result
}
