//! CPU large-frame wire boundaries and byte-exact replicated owner-map proof.

use anyhow::{ensure, Result};
use cuteafd_transport::expert::{
    BackboneRequest, SparkTopology, V41Tp4ChunkReceiver,
    V41_NATIVE_GROUP_REQUEST_FLAG, V41_NATIVE_UNASSIGNED_EXPERT_ID, V41_PARTIAL_ROW_BYTES,
    V41_ROUTED_EXPERTS, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16,
};
use cuteafd_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RequestView, ExpertProtocolV2RouteEntry,
    ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind,
    EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
};

const HIDDEN: usize = 5120;
const K32_ROW_BYTES: usize = 5280;
const REQUEST_ROW_BYTES: usize = 40 + 6 * 12 + K32_ROW_BYTES;
const RESPONSE_ROW_BYTES: usize = 4 + V41_PARTIAL_ROW_BYTES as usize;
const SLOT: usize = 8 << 20;
const DEPTH: usize = 8;
/// One-direction floor span: 8 MiB slot x depth.
const FLOOR_DIRECTION_SPAN: usize = SLOT * DEPTH;
/// Whole floor endpoint: request + response spans, the value the worker charges.
const FLOOR_ENDPOINT_SPAN: usize = 2 * FLOOR_DIRECTION_SPAN;
const MAX_FRAME: usize = 64 << 20;
const CAPACITY: u32 = 4096;
fn request_wire(rows: u32) -> usize {
    EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN + rows as usize * REQUEST_ROW_BYTES
}
fn response_wire(rows: u32) -> usize {
    EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN + rows as usize * RESPONSE_ROW_BYTES
}

fn build_native_request(rows: u32, request_id: u64) -> Result<ExpertProtocolV2Request> {
    ensure!(rows > 0 && rows <= CAPACITY, "rows out of range");
    let mut rng = 0x5eed_1234_9abc_def0u64;
    let mut next = || {
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let descriptors: Vec<ExpertProtocolV2RowDescriptor> = (0..rows)
        .map(|row| ExpertProtocolV2RowDescriptor {
            row_id: row as u64,
            source_kind: ExpertV2SourceKind::Prefill,
            source_request_id: request_id,
            token_position: row as u64,
            route_offset: row * 6,
            route_count: 6,
        })
        .collect();
    let mut routes = Vec::with_capacity(rows as usize * 6);
    for row in 0..rows {
        let mut chosen = [0u32; 6];
        for slot in 0..6 {
            loop {
                let expert = (next() % V41_ROUTED_EXPERTS as u64) as u32;
                if !chosen[..slot].contains(&expert) {
                    chosen[slot] = expert;
                    break;
                }
            }
        }
        for &expert in &chosen {
            routes.push(ExpertProtocolV2RouteEntry {
                row_index: row,
                expert_id: expert,
                gate_weight: 1.0 + (row % 5) as f32,
            });
        }
    }
    let mut request = ExpertProtocolV2Request::new(
        request_id,
        7,
        20,
        HIDDEN as u32,
        ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        descriptors,
        routes,
        vec![0u8; rows as usize * K32_ROW_BYTES],
    )?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

/// CPU: boundaries plus a real TP2EP2 owner-map round trip at M=819/1556.
#[test]
fn large_frame_boundaries_and_wire_roundtrip() -> Result<()> {
    assert_eq!(request_wire(128), 96 + 128 * 5392);
    assert_eq!(response_wire(128), 96 + 128 * 4 + 128 * 10240);
    assert!(response_wire(818) <= SLOT, "818 rows must fit the 8 MiB slot");
    assert!(response_wire(819) > SLOT, "819 rows must force response growth");
    assert!(request_wire(1555) <= SLOT, "1555 rows must fit the 8 MiB slot");
    assert!(request_wire(1556) > SLOT, "1556 rows must force request growth");
    assert!(response_wire(128) < SLOT && request_wire(128) < SLOT);
    assert_eq!(FLOOR_ENDPOINT_SPAN, 134_217_728);

    // TP2EP2 with an explicit owner map: real replicated-group coverage.
    let topology = SparkTopology::NATIVE_TP2_EP2;
    let mut owners = vec![0u8; V41_ROUTED_EXPERTS];
    for (expert, owner) in owners.iter_mut().enumerate() {
        *owner = (expert % topology.group_count() as usize) as u8;
    }
    for rows in [819u32, 1556] {
        let request = build_native_request(rows, 100 + rows as u64)?;
        let canonical: Vec<(u32, u32)> = request
            .routes
            .iter()
            .map(|route| (route.expert_id, route.gate_weight.to_bits()))
            .collect();
        let mut flagged = request.clone();
        flagged.with_native_group_owners(&owners, topology)?;
        ensure!(flagged.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG != 0);
        let frame = flagged.encode()?;
        ensure!(frame.len() == request_wire(rows));
        let native = BackboneRequest::parse_native_group(&frame, CAPACITY, topology)?;

        let mut owned_per_route = vec![0usize; canonical.len()];
        for rank in 0..topology.world_size() {
            let group = topology.group(rank)?;
            let mut ids = vec![-1i32; canonical.len()];
            let mut weights = vec![-1.0f32; canonical.len()];
            native.copy_native_group_routes_into(&mut ids, &mut weights, group)?;
            for (index, (expert, weight)) in canonical.iter().enumerate() {
                if owners[*expert as usize] == group {
                    ensure!(ids[index] == *expert as i32);
                    ensure!(weights[index].to_bits() == *weight);
                    owned_per_route[index] += 1;
                } else {
                    ensure!(ids[index] == V41_NATIVE_UNASSIGNED_EXPERT_ID);
                    ensure!(weights[index] == 0.0);
                }
            }
        }
        // Exactly-once per TP shard: TP ranks of the one owning group.
        for owned in &owned_per_route {
            ensure!(*owned == topology.tp() as usize);
        }
        // Chunked response coverage still holds at these sizes.
        let executors = topology.executor_ids();
        let chunk_cap = native.response_chunk_rows(MAX_FRAME)?;
        let mut receiver = V41Tp4ChunkReceiver::new_ranks(&native, &executors, MAX_FRAME)?;
        for rank in (0..executors.len()).rev() {
            let mut indices = vec![0u32; chunk_cap as usize];
            let mut row = 0u32;
            while row < rows {
                let chunk_rows = (rows - row).min(chunk_cap);
                let payload = vec![0u8; chunk_rows as usize * V41_PARTIAL_ROW_BYTES as usize];
                let chunk = native
                    .response_chunk(executors[rank], row, &payload, &mut indices, MAX_FRAME)?
                    .to_owned()?
                    .encode()?;
                receiver.push(&chunk, |actual, start, bytes| {
                    ensure!(actual == rank && start == row);
                    ensure!(bytes.len() == payload.len());
                    Ok(())
                })?;
                row += chunk_rows;
            }
        }
        ensure!(receiver.complete());
        ensure!(receiver.received_rows_slice() == &[rows; 6][..executors.len()]);
    }
    Ok(())
}
