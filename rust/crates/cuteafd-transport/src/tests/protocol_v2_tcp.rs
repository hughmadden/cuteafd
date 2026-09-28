use anyhow::Result;
use cuteafd_core::{DType, ExpertBatch, ExpertBatchRoute, ExpertBatchRow, ExpertGraphInstancePool, ExpertHostBatchSet, GraphBucket, LayerId, LayerWaveMode, ModelFacts, PlacementPolicy, PlacementVersion, PositionId, RequestId, RowSourceKind, DS4_FLASH_HIDDEN_BF16_BYTES, DS4_FLASH_HIDDEN_SIZE, DS4_FLASH_ROUTED_EXPERTS};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

use super::common::{protocol_v2_request, request_with_rows, spawn_protocol_v2_server};
use crate::{expert_protocol_v2_compact_id, handle_protocol_v2_synthetic_connection, protocol_v2_echo_loopback_response, protocol_v2_inproc_expert_request_roundtrip, protocol_v2_inproc_roundtrip, protocol_v2_inproc_roundtrip_arena_response_view, protocol_v2_synthetic_response, serve_protocol_v2_tcp_listener_with_executor, tcp_protocol_v2_expert_request_roundtrip, tcp_protocol_v2_roundtrip, tcp_protocol_v2_roundtrip_arena_response_view, tcp_protocol_v2_roundtrip_response_view, EchoExecutor, ExpertProtocolV2FrameArena, ExpertProtocolV2FrameBuffer, ExpertProtocolV2Request, ExpertV2Dtype, ExpertV2SourceKind, TcpProtocolV2PersistentClient, TcpTransportConfig, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN, EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN, EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN, PROTOCOL_V2_ECHO_EXECUTOR, PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR};

const PROTOCOL_V2_SPARSE_MOE_CHAIN_HOPS: usize = 43;

#[tokio::test]
async fn protocol_v2_tcp_decode_roundtrip_matches_inproc_binary_reference() -> Result<()> {
    let (addr, shutdown) = spawn_protocol_v2_server().await?;
    let request = protocol_v2_request(500, 1, ExpertV2SourceKind::Decode)?;

    let inproc = protocol_v2_inproc_roundtrip(&request).await?;
    let tcp = tcp_protocol_v2_roundtrip(addr, &request, TcpTransportConfig::default()).await?;

    assert_eq!(
        request.wire_stats().logical_payload_bytes,
        DS4_FLASH_HIDDEN_BF16_BYTES
    );
    assert_eq!(tcp.encode()?, inproc.encode()?);
    let _ = shutdown.send(());
    Ok(())
}

#[tokio::test]
async fn protocol_v2_tcp_mtp_roundtrip_matches_inproc_binary_reference() -> Result<()> {
    let (addr, shutdown) = spawn_protocol_v2_server().await?;
    for (idx, row_count) in [1, 2, 4, 8].into_iter().enumerate() {
        let request =
            protocol_v2_request(510 + idx as u64, row_count, ExpertV2SourceKind::MtpVerify)?;

        let inproc = protocol_v2_inproc_roundtrip(&request).await?;
        let tcp = tcp_protocol_v2_roundtrip(addr, &request, TcpTransportConfig::default()).await?;

        assert_eq!(
            request.wire_stats().logical_payload_bytes,
            row_count * DS4_FLASH_HIDDEN_BF16_BYTES
        );
        assert_eq!(tcp.encode()?, inproc.encode()?);
    }
    let _ = shutdown.send(());
    Ok(())
}

#[tokio::test]
async fn protocol_v2_tcp_prefill_roundtrip_matches_inproc_binary_reference() -> Result<()> {
    let (addr, shutdown) = spawn_protocol_v2_server().await?;
    for (idx, row_count) in [16, 64, 256, 512].into_iter().enumerate() {
        let request =
            protocol_v2_request(520 + idx as u64, row_count, ExpertV2SourceKind::Prefill)?;

        let inproc = protocol_v2_inproc_roundtrip(&request).await?;
        let tcp = tcp_protocol_v2_roundtrip(addr, &request, TcpTransportConfig::default()).await?;

        assert_eq!(
            request.wire_stats().logical_payload_bytes,
            row_count * DS4_FLASH_HIDDEN_BF16_BYTES
        );
        assert!(request.wire_stats().wire_bytes > request.wire_stats().logical_payload_bytes);
        assert_eq!(tcp.encode()?, inproc.encode()?);
    }
    let _ = shutdown.send(());
    Ok(())
}

#[tokio::test]
async fn protocol_v2_tcp_expert_request_bridge_matches_inproc_route_executor() -> Result<()> {
    let (addr, shutdown) = spawn_protocol_v2_server().await?;
    let mut request = request_with_rows(545, 4, 32, LayerWaveMode::MtpVerify);
    request
        .wave
        .as_mut()
        .expect("test request carries wave metadata")
        .logical_bf16_payload_bytes = request.rows.len() * request.hidden_dim as usize * 2;

    let inproc = protocol_v2_inproc_expert_request_roundtrip(&request).await?;
    let tcp =
        tcp_protocol_v2_expert_request_roundtrip(addr, &request, TcpTransportConfig::default())
            .await?;

    assert_eq!(tcp, inproc);
    assert_eq!(tcp.partial_outputs.len(), request.rows.len());
    assert_eq!(tcp.partial_outputs[0].len(), request.hidden_dim as usize);
    assert_ne!(tcp.partial_outputs[0], request.rows[0].hidden);

    let mut changed_routes = request.clone();
    changed_routes.rows[0].routes[0].expert_id += 1;
    let changed = tcp_protocol_v2_expert_request_roundtrip(
        addr,
        &changed_routes,
        TcpTransportConfig::default(),
    )
    .await?;
    assert_ne!(tcp.partial_outputs[0], changed.partial_outputs[0]);

    let _ = shutdown.send(());
    Ok(())
}





#[tokio::test]
async fn protocol_v2_persistent_client_reconnects_after_stale_request_id_response() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let accepted_connections = Arc::new(AtomicUsize::new(0));
    let observed_requests = Arc::new(AtomicUsize::new(0));
    let accepted_connections_server = Arc::clone(&accepted_connections);
    let observed_requests_server = Arc::clone(&observed_requests);
    let server = tokio::spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await?;
            accepted_connections_server.fetch_add(1, Ordering::SeqCst);
            let observed_requests = Arc::clone(&observed_requests_server);
            tokio::spawn(async move {
                while let Ok(request) = read_protocol_v2_request_from_stream(&mut stream).await {
                    let request_index = observed_requests.fetch_add(1, Ordering::SeqCst);
                    let mut response = protocol_v2_synthetic_response(&request)?;
                    if request_index == 0 {
                        response.header.request_id = request.header.request_id.saturating_sub(1);
                    }
                    let response_frame = response.encode()?;
                    stream.write_all(&response_frame).await?;
                    stream.flush().await?;
                }
                Ok::<(), anyhow::Error>(())
            });
        }
        #[allow(unreachable_code)]
        Ok::<(), anyhow::Error>(())
    });

    let mut client = TcpProtocolV2PersistentClient::new(addr, TcpTransportConfig::default());
    let request = protocol_v2_request(710, 1, ExpertV2SourceKind::Decode)?;
    let response = client.roundtrip(&request).await?;
    assert_eq!(response.header.request_id, request.header.request_id);
    assert_eq!(accepted_connections.load(Ordering::SeqCst), 2);
    assert_eq!(observed_requests.load(Ordering::SeqCst), 2);

    let next_request = protocol_v2_request(711, 1, ExpertV2SourceKind::Decode)?;
    let next_response = client.roundtrip(&next_request).await?;
    server.abort();
    assert_eq!(
        next_response.header.request_id,
        next_request.header.request_id
    );
    assert_eq!(accepted_connections.load(Ordering::SeqCst), 2);
    assert_eq!(observed_requests.load(Ordering::SeqCst), 3);
    Ok(())
}

async fn read_protocol_v2_request_from_stream(
    stream: &mut TcpStream,
) -> Result<ExpertProtocolV2Request> {
    let mut frame = vec![0_u8; crate::EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN];
    stream.read_exact(&mut frame).await?;
    let wire_bytes = ExpertProtocolV2Request::wire_bytes_from_header(&frame)?;
    frame.resize(wire_bytes, 0);
    stream
        .read_exact(&mut frame[crate::EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN..])
        .await?;
    ExpertProtocolV2Request::decode(&frame)
}

#[tokio::test]
async fn protocol_v2_tcp_listener_uses_injected_executor() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let server = tokio::spawn(async move {
        let _ =
            serve_protocol_v2_tcp_listener_with_executor(listener, Arc::new(EchoExecutor)).await;
    });
    let request = protocol_v2_request(565, 2, ExpertV2SourceKind::Decode)?.with_debug_checksum();

    let expected = protocol_v2_echo_loopback_response(&request)?;
    let tcp = tcp_protocol_v2_roundtrip(addr, &request, TcpTransportConfig::default()).await?;
    let synthetic = protocol_v2_inproc_roundtrip(&request).await?;

    assert_eq!(
        tcp.header.executor_id,
        expert_protocol_v2_compact_id(PROTOCOL_V2_ECHO_EXECUTOR)
    );
    assert_eq!(tcp.encode()?, expected.encode()?);
    assert_eq!(tcp.partial_output_payload, request.hidden_payload);
    assert_ne!(tcp.encode()?, synthetic.encode()?);

    server.abort();
    Ok(())
}


#[tokio::test]
async fn protocol_v2_tcp_response_view_reuses_frame_buffer() -> Result<()> {
    let (addr, shutdown) = spawn_protocol_v2_server().await?;
    let request = protocol_v2_request(540, 2, ExpertV2SourceKind::MtpVerify)?.with_debug_checksum();
    let expected = protocol_v2_inproc_roundtrip(&request).await?;
    let mut response_buffer =
        ExpertProtocolV2FrameBuffer::with_capacity(request.wire_stats().wire_bytes);

    let first_payload_ptr = {
        let view = tcp_protocol_v2_roundtrip_response_view(
            addr,
            &request,
            TcpTransportConfig::default(),
            &mut response_buffer,
        )
        .await?;
        assert_eq!(view.header.request_id, request.header.request_id);
        assert_eq!(view.header.output_dim, request.header.hidden_dim);
        assert_eq!(
            view.header.output_row_stride_bytes,
            request.header.hidden_row_stride_bytes
        );
        assert_eq!(
            view.partial_output_payload(),
            expected.partial_output_payload.as_slice()
        );
        assert_eq!(
            view.partial_output_row_payload(1)?,
            &expected.partial_output_payload
                [DS4_FLASH_HIDDEN_BF16_BYTES..2 * DS4_FLASH_HIDDEN_BF16_BYTES]
        );
        view.verify_checksum()?;
        view.partial_output_payload().as_ptr()
    };
    let first_capacity = response_buffer.capacity();

    let second_payload_ptr = {
        let view = tcp_protocol_v2_roundtrip_response_view(
            addr,
            &request,
            TcpTransportConfig::default(),
            &mut response_buffer,
        )
        .await?;
        assert_eq!(
            view.partial_output_payload(),
            expected.partial_output_payload.as_slice()
        );
        assert_eq!(
            view.partial_output_row_payload(1)?,
            &expected.partial_output_payload
                [DS4_FLASH_HIDDEN_BF16_BYTES..2 * DS4_FLASH_HIDDEN_BF16_BYTES]
        );
        view.verify_checksum()?;
        view.partial_output_payload().as_ptr()
    };

    assert_eq!(second_payload_ptr, first_payload_ptr);
    assert_eq!(response_buffer.capacity(), first_capacity);
    let _ = shutdown.send(());
    Ok(())
}


#[tokio::test]
async fn protocol_v2_tcp_arena_response_view_matches_inproc_arena_response_view() -> Result<()> {
    let (addr, shutdown) = spawn_protocol_v2_server().await?;
    for (request_id, row_count, source_kind, with_debug_checksum) in [
        (570, 1, ExpertV2SourceKind::Decode, false),
        (571, 4, ExpertV2SourceKind::MtpVerify, true),
        (572, 16, ExpertV2SourceKind::Prefill, true),
    ] {
        let request = if with_debug_checksum {
            protocol_v2_request(request_id, row_count, source_kind)?.with_debug_checksum()
        } else {
            protocol_v2_request(request_id, row_count, source_kind)?
        };
        let mut inproc_arena = ExpertProtocolV2FrameArena::with_capacities(
            request.wire_stats().wire_bytes,
            request.wire_stats().wire_bytes,
        );
        let mut tcp_arena = ExpertProtocolV2FrameArena::with_capacities(
            request.wire_stats().wire_bytes,
            request.wire_stats().wire_bytes,
        );
        let inproc_response_ptr = inproc_arena.response_ptr();
        let tcp_response_ptr = tcp_arena.response_ptr();
        let last_row_start = (row_count - 1) * DS4_FLASH_HIDDEN_BF16_BYTES;
        let last_row_end = last_row_start + DS4_FLASH_HIDDEN_BF16_BYTES;

        let (inproc_header, inproc_payload, inproc_wire_stats) = {
            let view =
                protocol_v2_inproc_roundtrip_arena_response_view(&request, &mut inproc_arena)
                    .await?;
            assert_eq!(view.header.request_id, request.header.request_id);
            assert_eq!(view.header.output_dim, request.header.hidden_dim);
            assert_eq!(
                view.header.output_row_stride_bytes,
                request.header.hidden_row_stride_bytes
            );
            assert_eq!(view.debug_checksum_enabled(), with_debug_checksum);
            if with_debug_checksum {
                view.verify_checksum()?;
            }
            assert_eq!(view.partial_output_payload().as_ptr(), unsafe {
                inproc_response_ptr.add(view.header_len())
            });
            (
                view.header.clone(),
                view.partial_output_payload().to_vec(),
                view.wire_stats(),
            )
        };

        let view = tcp_protocol_v2_roundtrip_arena_response_view(
            addr,
            &request,
            TcpTransportConfig::default(),
            &mut tcp_arena,
        )
        .await?;
        assert_eq!(view.header, inproc_header);
        assert_eq!(view.wire_stats(), inproc_wire_stats);
        assert_eq!(view.partial_output_payload(), inproc_payload.as_slice());
        assert_eq!(
            view.partial_output_row_payload(row_count - 1)?,
            &inproc_payload[last_row_start..last_row_end]
        );
        assert_eq!(view.debug_checksum_enabled(), with_debug_checksum);
        if with_debug_checksum {
            view.verify_checksum()?;
        }
        assert_eq!(view.partial_output_payload().as_ptr(), unsafe {
            tcp_response_ptr.add(view.header_len())
        });
    }
    let _ = shutdown.send(());
    Ok(())
}

#[tokio::test]
async fn protocol_v2_tcp_43_layer_hot_view_chains_match_inproc() -> Result<()> {
    let (addr, shutdown) = spawn_protocol_v2_server().await?;
    for (request_base, row_count, source_kind) in [
        (600, 1, ExpertV2SourceKind::Decode),
        (700, 8, ExpertV2SourceKind::MtpVerify),
        (800, 16, ExpertV2SourceKind::Prefill),
        (900, 32, ExpertV2SourceKind::Prefill),
    ] {
        let capacity_request = protocol_v2_request(request_base, row_count, source_kind)?;
        let mut inproc_arena = ExpertProtocolV2FrameArena::with_capacities(
            capacity_request.wire_stats().wire_bytes,
            capacity_request.wire_stats().wire_bytes,
        );
        let mut tcp_arena = ExpertProtocolV2FrameArena::with_capacities(
            capacity_request.wire_stats().wire_bytes,
            capacity_request.wire_stats().wire_bytes,
        );
        let inproc_request_ptr = inproc_arena.request_ptr();
        let inproc_response_ptr = inproc_arena.response_ptr();
        let tcp_request_ptr = tcp_arena.request_ptr();
        let tcp_response_ptr = tcp_arena.response_ptr();
        let inproc_request_capacity = inproc_arena.request_capacity();
        let inproc_response_capacity = inproc_arena.response_capacity();
        let tcp_request_capacity = tcp_arena.request_capacity();
        let tcp_response_capacity = tcp_arena.response_capacity();
        let mut request_wire_bytes = 0_usize;
        let mut response_wire_bytes = 0_usize;
        let mut logical_payload_bytes = 0_usize;

        for hop in 0..PROTOCOL_V2_SPARSE_MOE_CHAIN_HOPS {
            let request = protocol_v2_request(request_base + hop as u64, row_count, source_kind)?;
            let (inproc_header, inproc_payload, inproc_wire_stats) = {
                let view =
                    protocol_v2_inproc_roundtrip_arena_response_view(&request, &mut inproc_arena)
                        .await?;
                assert!(!view.debug_checksum_enabled());
                assert_eq!(view.header.output_dim, request.header.hidden_dim);
                assert_eq!(
                    view.header.output_row_stride_bytes,
                    request.header.hidden_row_stride_bytes
                );
                assert_eq!(view.partial_output_payload().as_ptr(), unsafe {
                    inproc_response_ptr.add(view.header_len())
                });
                (
                    view.header.clone(),
                    view.partial_output_payload().to_vec(),
                    view.wire_stats(),
                )
            };

            let view = tcp_protocol_v2_roundtrip_arena_response_view(
                addr,
                &request,
                TcpTransportConfig::default(),
                &mut tcp_arena,
            )
            .await?;
            assert!(!view.debug_checksum_enabled());
            assert_eq!(view.header, inproc_header);
            assert_eq!(view.wire_stats(), inproc_wire_stats);
            assert_eq!(view.partial_output_payload(), inproc_payload.as_slice());
            assert_eq!(view.partial_output_payload().as_ptr(), unsafe {
                tcp_response_ptr.add(view.header_len())
            });

            request_wire_bytes += request.wire_stats().wire_bytes;
            response_wire_bytes += view.wire_stats().wire_bytes;
            logical_payload_bytes += view.wire_stats().logical_payload_bytes;
        }

        assert_eq!(
            request_wire_bytes,
            capacity_request.wire_stats().wire_bytes * PROTOCOL_V2_SPARSE_MOE_CHAIN_HOPS
        );
        assert_eq!(
            response_wire_bytes,
            (EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN + row_count * DS4_FLASH_HIDDEN_BF16_BYTES)
                * PROTOCOL_V2_SPARSE_MOE_CHAIN_HOPS
        );
        assert_eq!(
            logical_payload_bytes,
            row_count * DS4_FLASH_HIDDEN_BF16_BYTES * PROTOCOL_V2_SPARSE_MOE_CHAIN_HOPS
        );
        assert_eq!(inproc_arena.request_ptr(), inproc_request_ptr);
        assert_eq!(inproc_arena.response_ptr(), inproc_response_ptr);
        assert_eq!(tcp_arena.request_ptr(), tcp_request_ptr);
        assert_eq!(tcp_arena.response_ptr(), tcp_response_ptr);
        assert_eq!(inproc_arena.request_capacity(), inproc_request_capacity);
        assert_eq!(inproc_arena.response_capacity(), inproc_response_capacity);
        assert_eq!(tcp_arena.request_capacity(), tcp_request_capacity);
        assert_eq!(tcp_arena.response_capacity(), tcp_response_capacity);
    }
    let _ = shutdown.send(());
    Ok(())
}

#[tokio::test]
async fn protocol_v2_tcp_request_view_rejects_bad_debug_checksum() -> Result<()> {
    let (addr, shutdown) = spawn_protocol_v2_server().await?;
    let request = protocol_v2_request(560, 2, ExpertV2SourceKind::MtpVerify)?.with_debug_checksum();
    let mut frame = request.encode()?;
    let payload_start = request.header_len()
        + request.rows.len() * EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN
        + request.routes.len() * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN;
    frame[payload_start] ^= 0x5a;

    let mut stream = TcpStream::connect(addr).await?;
    stream.write_all(&frame).await?;
    stream.flush().await?;

    let mut response_first_byte = [0_u8; 1];
    let read = timeout(
        Duration::from_secs(5),
        stream.read(&mut response_first_byte),
    )
    .await??;

    assert_eq!(read, 0);
    let _ = shutdown.send(());
    Ok(())
}

#[tokio::test]
async fn protocol_v2_tcp_truncated_response_is_rejected() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        let Ok((mut stream, _peer)) = listener.accept().await else {
            return;
        };
        let _ = stream.write_all(b"too-short").await;
    });
    let request = protocol_v2_request(530, 1, ExpertV2SourceKind::Decode)?;
    let err = tcp_protocol_v2_roundtrip(addr, &request, TcpTransportConfig::default())
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("ProtocolV2 response header"));
    Ok(())
}

#[tokio::test]
async fn native_v41_fp32_routes_and_partial_planes_survive_persistent_tcp() -> Result<()> {
    struct RoutePlaneFixture;
    impl crate::ProtocolV2ExpertExecutor for RoutePlaneFixture {
        fn name(&self) -> &'static str {
            "v41-fp32-wire-fixture"
        }
        fn execute(
            &self,
            request: &crate::ExpertProtocolV2RequestView<'_>,
        ) -> Result<crate::ExpertProtocolV2Response> {
            let mut payload = Vec::new();
            for route in 0..request.header.route_count as usize {
                let weight = request.route(route)?.gate_weight;
                for _ in 0..5120 {
                    payload.extend_from_slice(&weight.to_le_bytes());
                }
            }
            crate::ExpertProtocolV2Response::new(
                request.header.request_id,
                request.header.placement_version,
                request.header.layer_id,
                request.header.row_count,
                6 * 5120,
                ExpertV2Dtype::F32,
                crate::ExpertProtocolV2Status::Ok,
                payload,
            )
        }
    }
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(serve_protocol_v2_tcp_listener_with_executor(
        listener,
        Arc::new(RoutePlaneFixture),
    ));
    let mut client = TcpProtocolV2PersistentClient::new(address, TcpTransportConfig::default());
    for iteration in 0..2_u32 {
        let rows = (0..16)
            .map(|row| crate::ExpertProtocolV2RowDescriptor {
                row_id: row as u64,
                source_kind: ExpertV2SourceKind::MtpVerify,
                source_request_id: 1000 + row as u64,
                token_position: 91,
                route_offset: row * 6,
                route_count: 6,
            })
            .collect();
        let routes = (0..96)
            .map(|route| crate::ExpertProtocolV2RouteEntry {
                row_index: route / 6,
                expert_id: (route * 7) % 384,
                gate_weight: f32::from_bits(0x3e80_0001 + iteration * 1000 + route),
            })
            .collect();
        let request = ExpertProtocolV2Request::new(
            9100 + iteration as u64,
            41,
            13,
            5120,
            ExpertV2Dtype::Bf16,
            rows,
            routes,
            vec![0; 16 * 5120 * 2],
        )?
        .with_debug_checksum();
        let response = client.roundtrip(&request).await?;
        assert_eq!(response.header.output_dtype, ExpertV2Dtype::F32);
        assert_eq!(response.partial_output_payload.len(), 16 * 6 * 5120 * 4);
        for (route, plane) in response
            .partial_output_payload
            .chunks_exact(5120 * 4)
            .enumerate()
        {
            for word in plane.chunks_exact(4) {
                assert_eq!(word, request.routes[route].gate_weight.to_le_bytes());
            }
        }
    }
    server.abort();
    Ok(())
}

struct DirectSubmitExecutor(u8);
impl crate::ProtocolV2ExpertExecutor for DirectSubmitExecutor {
    fn name(&self) -> &'static str {
        "direct-submit-test"
    }
    fn execute(
        &self,
        _: &crate::ExpertProtocolV2RequestView<'_>,
    ) -> Result<crate::ExpertProtocolV2Response> {
        panic!("direct submission must bypass the blocking adapter")
    }
    fn tcp_response_chunks(&self) -> bool {
        true
    }
    fn submit_tcp_chunks(
        &self,
        request: &crate::ExpertProtocolV2RequestView<'_>,
    ) -> Option<Result<tokio::sync::mpsc::Receiver<Result<crate::ExpertProtocolV2Response>>>> {
        Some((|| {
            let (sender, receiver) = tokio::sync::mpsc::channel(1);
            let mut response = crate::ProtocolV2ExpertExecutor::execute(&EchoExecutor, request)?;
            let mode = self.0;
            if mode == 2 {
                response.header.request_id += 1;
            }
            tokio::spawn(async move {
                match mode {
                    1 => {} // Closing before a final frame must fail.
                    3 => {
                        let _ = sender
                            .send(Err(anyhow::anyhow!("injected execution failure")))
                            .await;
                    }
                    _ => {
                        let _ = sender.send(Ok(response)).await;
                    }
                }
            });
            Ok(receiver)
        })())
    }
}

#[tokio::test]
async fn protocol_v2_tcp_direct_submission_reuses_connection() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await?;
        crate::protocol_v2_tcp::handle_protocol_v2_connection_with_executor(
            socket,
            Arc::new(DirectSubmitExecutor(0)),
        )
        .await
    });
    let mut client = TcpProtocolV2PersistentClient::new(address, TcpTransportConfig::default());
    for id in [123, 124] {
        let request = protocol_v2_request(id, 1, ExpertV2SourceKind::Decode)?;
        let expected = protocol_v2_echo_loopback_response(&request)?;
        let mut actual = None;
        client
            .roundtrip_chunks(&request, 1, |bytes| {
                actual = Some(crate::ExpertProtocolV2Response::decode(bytes)?);
                Ok(())
            })
            .await?;
        assert_eq!(
            actual.unwrap().partial_output_payload,
            expected.partial_output_payload
        );
    }
    drop(client);
    timeout(Duration::from_secs(2), server).await???;
    Ok(())
}

#[tokio::test]
async fn protocol_v2_tcp_direct_submission_rejects_invalid_completion() -> Result<()> {
    for (mode, message) in [
        (1, "without a final"),
        (2, "different request"),
        (3, "injected execution failure"),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await?;
            crate::protocol_v2_tcp::handle_protocol_v2_connection_with_executor(
                socket,
                Arc::new(DirectSubmitExecutor(mode)),
            )
            .await
        });
        let mut client = TcpProtocolV2PersistentClient::new(address, TcpTransportConfig::default());
        let request = protocol_v2_request(125, 1, ExpertV2SourceKind::Decode)?;
        assert!(client
            .roundtrip_chunks(&request, 1, |_| Ok(()))
            .await
            .is_err());
        let error = timeout(Duration::from_secs(2), server).await??.unwrap_err();
        assert!(error.to_string().contains(message), "{error:#}");
    }
    Ok(())
}
