//! Coordinator-resident routes on the canonical V4.1 Spark wire.
use anyhow::{ensure, Result};
use cuteafd_core::CoordinatorExpertResidency;

pub const V41_COORDINATOR_ROUTES_REQUEST_FLAG: u32 = 1 << 30;
pub const V41_COORDINATOR_ROUTE_BIT: u32 = 1 << 12;
pub const V41_UNASSIGNED_EXPERT_ID: i32 = 384;
const EXPERT_MASK: u32 = (1 << 9) - 1;

pub(super) fn require_v41_geometry() -> Result<()> {
    let geometry = cuteafd_core::expert_geometry();
    ensure!(geometry.hidden == super::V41_HIDDEN
        && geometry.experts as usize == super::V41_ROUTED_EXPERTS
        && geometry.topk == super::V41_BACKBONE_TOPK,
        "coordinator routes require V4.1 geometry");
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoordinatorRouteWord {
    pub expert_id: u16,
    pub coordinator: bool,
}
impl CoordinatorRouteWord {
    pub fn decode(word: u32, permitted: bool) -> Result<Self> {
        let allowed = EXPERT_MASK | if permitted { V41_COORDINATOR_ROUTE_BIT } else { 0 };
        ensure!(word & !allowed == 0, "reserved coordinator route bits");
        let expert_id = (word & EXPERT_MASK) as u16;
        ensure!(expert_id < 384, "coordinator route expert is out of range");
        Ok(Self { expert_id, coordinator: word & V41_COORDINATOR_ROUTE_BIT != 0 })
    }
}
impl crate::ExpertProtocolV2Request {
    /// Mark a canonical batch transactionally. The flag describes the link even
    /// when this batch contains no resident route; row/route order is unchanged.
    pub fn with_coordinator_routes(&mut self, resident: &CoordinatorExpertResidency) -> Result<()> {
        super::BackboneRequest::validate_owned(self, 4096)?;
        ensure!(self.header.flags & V41_COORDINATOR_ROUTES_REQUEST_FLAG == 0,
            "coordinator routes are already marked");
        require_v41_geometry()?;
        for route in &mut self.routes {
            if resident.contains(route.expert_id as u16) { route.expert_id |= V41_COORDINATOR_ROUTE_BIT; }
        }
        self.header.flags |= V41_COORDINATOR_ROUTES_REQUEST_FLAG;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{expert::{BackboneRequest, SparkTopology, V41Tp4ChunkReceiver},
        ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
        ExpertV2Dtype, ExpertV2SourceKind};

    fn request(rows: usize) -> ExpertProtocolV2Request {
        let mut request = ExpertProtocolV2Request::new(31, 7, 3, 5120, ExpertV2Dtype::Bf16,
            (0..rows).map(|r| ExpertProtocolV2RowDescriptor { row_id: r as u64,
                source_kind: ExpertV2SourceKind::Decode, source_request_id: 31, token_position: r as u64,
                route_offset: r as u32 * 6, route_count: 6 }).collect(),
            (0..rows * 6).map(|i| ExpertProtocolV2RouteEntry { row_index: i as u32 / 6,
                expert_id: (i % 384) as u32, gate_weight: (i % 6 + 1) as f32 / 21.0 }).collect(),
            vec![0; rows * 10240]).unwrap();
        request.header.flags |= super::super::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        request
    }

    #[test]
    fn coordinator_bit_requires_flag_and_all_other_reserved_bits_are_refused() -> Result<()> {
        for id in 0..384 {
            let word = id | V41_COORDINATOR_ROUTE_BIT;
            assert!(CoordinatorRouteWord::decode(word, false).is_err());
            assert_eq!(CoordinatorRouteWord::decode(word, true)?, CoordinatorRouteWord {
                expert_id: id as u16, coordinator: true });
        }
        for bit in (9..32).filter(|&b| b != 12) {
            assert!(CoordinatorRouteWord::decode(1 << bit, true).is_err());
            let mut bad = request(1);
            bad.header.flags |= V41_COORDINATOR_ROUTES_REQUEST_FLAG;
            bad.routes[0].expert_id |= 1 << bit;
            assert!(BackboneRequest::validate_owned(&bad, 1).is_err());
            assert!(BackboneRequest::parse(&bad.encode()?, 1).is_err());
        }
        let mut missing = request(1);
        missing.routes[0].expert_id |= V41_COORDINATOR_ROUTE_BIT;
        assert!(BackboneRequest::validate_owned(&missing, 1).is_err());
        assert!(BackboneRequest::parse(&missing.encode()?, 1).is_err());
        Ok(())
    }

    #[test]
    fn marked_routes_unpack_as_unassigned_without_changing_frame_or_weights() -> Result<()> {
        let residents = CoordinatorExpertResidency::from_experts(&[0, 3, 63, 64, 383])?;
        for rows in [1, 2, 64, 4096] {
            let mut marked = request(rows);
            let original = marked.clone();
            marked.with_coordinator_routes(&residents)?;
            let frame = marked.encode()?;
            assert_eq!(frame.len(), original.encode()?.len());
            let parsed = BackboneRequest::parse(&frame, rows as u32)?;
            let mut ids = vec![-2; rows * 6]; let mut weights = vec![-1.0; rows * 6];
            parsed.copy_routes_into(&mut ids, &mut weights)?;
            for (i, route) in original.routes.iter().enumerate() {
                let coordinator = residents.contains(route.expert_id as u16);
                assert_eq!(marked.routes[i].row_index, route.row_index);
                assert_eq!(marked.routes[i].gate_weight.to_bits(), route.gate_weight.to_bits());
                assert_eq!(ids[i], if coordinator { 384 } else { route.expert_id as i32 });
                assert_eq!(weights[i].to_bits(), if coordinator { 0.0f32.to_bits() } else { route.gate_weight.to_bits() });
            }
        }
        Ok(())
    }

    #[test]
    fn tier_link_flags_empty_batches_and_marking_is_transactional() -> Result<()> {
        let mut empty = request(1);
        empty.with_coordinator_routes(&CoordinatorExpertResidency::empty())?;
        assert_ne!(empty.header.flags & V41_COORDINATOR_ROUTES_REQUEST_FLAG, 0);
        assert!(empty.routes.iter().all(|r| r.expert_id < 384));
        assert!(empty.with_coordinator_routes(&CoordinatorExpertResidency::empty()).is_err());
        let mut bad = request(1); bad.routes[5].expert_id = 384;
        let before = bad.clone();
        assert!(bad.with_coordinator_routes(&CoordinatorExpertResidency::from_experts(&[0])?).is_err());
        assert_eq!(bad.header.flags, before.header.flags);
        assert_eq!(bad.routes[0].expert_id, before.routes[0].expert_id);
        Ok(())
    }

    #[test]
    fn flagged_chunk_receivers_admit_plain_responses_at_every_rank_count() -> Result<()> {
        for ranks in 1..=8 {
            for format in [0, crate::EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_UE8M0_K32,
                crate::EXPERT_PROTOCOL_V2_FLAG_RESPONSE_NVFP4_E2M1_FP8_E4M3] {
                let mut req = request(2); req.header.flags |= format;
                req.with_coordinator_routes(&CoordinatorExpertResidency::from_experts(&[1, 8])?)?;
                let frame = req.encode()?; let parsed = BackboneRequest::parse(&frame, 2)?;
                let topology = SparkTopology::new(ranks)?;
                let executors = topology.executor_ids();
                let mut receiver = V41Tp4ChunkReceiver::from_owned_ranks(&req, 2, &executors, 64 << 20)?;
                for id in executors {
                    let bytes = vec![0; parsed.plane_bytes()?];
                    let response = parsed.response(id, &bytes)?;
                    assert_eq!(response.header.flags & V41_COORDINATOR_ROUTES_REQUEST_FLAG, 0);
                    let encoded = response.to_owned()?.encode()?;
                    receiver.push(&encoded, |_, _, _| Ok(()))?;
                }
                assert!(receiver.complete());
            }
        }
        Ok(())
    }

    #[test]
    fn flagged_chunked_and_checksum_responses_keep_only_response_flags() -> Result<()> {
        for checksum in [false, true] {
            let mut req = request(8);
            if checksum { req = req.with_debug_checksum(); }
            req.with_coordinator_routes(&CoordinatorExpertResidency::from_experts(&[0, 9])?)?;
            let frame = req.encode()?;
            let parsed = BackboneRequest::parse(&frame, 8)?;
            let mut receiver = V41Tp4ChunkReceiver::new_ranks(&parsed, &[1], 64 << 20)?;
            let mut first = 0;
            let mut indices = vec![0; 8];
            while first < 8 {
                let bytes = vec![0; 10240];
                let response = parsed.response_chunk(1, first, &bytes, &mut indices, 11000)?;
                assert_eq!(response.header.flags & V41_COORDINATOR_ROUTES_REQUEST_FLAG, 0);
                assert_eq!(response.header.flags & crate::EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM != 0, checksum);
                let count = response.header.row_count;
                receiver.push(&response.to_owned()?.encode()?, |_, _, _| Ok(()))?;
                first += count;
            }
            assert!(receiver.complete());
        }
        let mut duplicate = request(1);
        duplicate.header.flags |= V41_COORDINATOR_ROUTES_REQUEST_FLAG;
        duplicate.routes[1].expert_id = duplicate.routes[0].expert_id | V41_COORDINATOR_ROUTE_BIT;
        assert!(BackboneRequest::validate_owned(&duplicate, 1).is_err());
        assert!(BackboneRequest::parse(&duplicate.encode()?, 1).is_err());
        Ok(())
    }

    #[test]
    fn coordinator_flag_is_request_only_and_group_flag_stays_unknown() -> Result<()> {
        let mut req = request(1); req.header.flags |= 1 << 18;
        assert!(req.encode().is_err());
        req.header.flags &= !(1 << 18);
        req.with_coordinator_routes(&CoordinatorExpertResidency::empty())?;
        let frame = req.encode()?; let parsed = BackboneRequest::parse(&frame, 1)?;
        let bytes = vec![0; parsed.plane_bytes()?]; let mut response = parsed.response(1, &bytes)?;
        response.header.flags |= V41_COORDINATOR_ROUTES_REQUEST_FLAG;
        assert!(response.validate().is_err());
        req.header.flags |= crate::EXPERT_PROTOCOL_V2_FLAG_STREAM_PLAN;
        assert!(req.encode().is_err());
        Ok(())
    }
}
