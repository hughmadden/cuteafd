//! Spark tensor-parallel topology and executor identity.
//!
//! Every Spark rank holds a tensor-parallel slice of every routed expert of its
//! layers and receives the same canonical request; the coordinator sums the
//! rank planes in rank order. There are no replicated expert groups, so the
//! topology is just the rank count.
//!
//! The 12-byte route entry carries the expert id in bits `0..8`; bits `9..31`
//! are reserved, free for future use, and rejected by the canonical parser
//! through its `expert_id < experts` check. Request flag bit `18` is reserved
//! and rejected by the protocol flag validator.
//!
//! Topology is bound into response identity: each rank count owns a disjoint
//! executor-id namespace (see [`SparkTopology::executor_id`]), so a stale
//! worker launched for another world cannot satisfy this one's receiver.

use anyhow::{bail, ensure, Result};

/// Routed backbone experts in the official V4.1 checkpoint.
pub const V41_ROUTED_EXPERTS: usize = 384;

/// Validated Spark tensor-parallel world of 1..=8 physical ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SparkTopology {
    tp: u8,
}

impl SparkTopology {
    /// Validates a rank count; no rank is invented.
    pub fn new(tp: u8) -> Result<Self> {
        ensure!(tp >= 1, "Spark TP degree must be at least one");
        let topology = Self { tp };
        topology.executor_base()?;
        Ok(topology)
    }

    /// Tensor-parallel degree: the physical Spark rank count.
    pub const fn tp(self) -> u8 {
        self.tp
    }

    /// Physical Spark ranks.
    pub const fn world_size(self) -> usize {
        self.tp as usize
    }

    /// Canonical executor identity for a physical rank.
    ///
    /// Namespaces are disjoint across rank counts and keep their historical
    /// identities (TP4 = 1..=4, TP2 = 5..=6, TP3 = 7..=9, TP6 = 27..=32,
    /// TP1 = 33), so a response from a worker launched for another world is
    /// rejected by identity. This binds world and rank, not checkpoint or
    /// deployment identity.
    pub fn executor_id(self, rank: usize) -> Result<u64> {
        ensure!(rank < self.world_size(), "Spark rank out of range");
        Ok(self.executor_base()? + rank as u64)
    }

    /// Canonical executor identities for every physical rank, in rank order.
    pub fn executor_ids(self) -> Vec<u64> {
        (0..self.world_size())
            .map(|rank| self.executor_id(rank).expect("validated topology rank fits its namespace"))
            .collect()
    }

    /// Physical rank named by an executor identity, or `None` for another
    /// world's (or an unknown) identity.
    pub fn rank_of_executor(self, executor_id: u64) -> Option<usize> {
        let base = self.executor_base().ok()?;
        let offset = executor_id.checked_sub(base)?;
        (offset < self.world_size() as u64).then_some(offset as usize)
    }

    fn executor_base(self) -> Result<u64> {
        match self.tp {
            4 => Ok(1),
            2 => Ok(5),
            3 => Ok(7),
            6 => Ok(27),
            1 => Ok(33),
            5 => Ok(104),
            7 => Ok(120),
            8 => Ok(128),
            tp => bail!("unsupported Spark TP{tp}: supported are TP1..TP8"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert::{tests::request, BackboneRequest, SparkExperts, V41Tp4ChunkReceiver, V41Tp4Planes,
        V41_PARTIAL_ROW_BYTES};
    use crate::{ExpertProtocolV2Response, TcpTransportConfig};
    use std::collections::BTreeSet;
    use std::net::SocketAddr;

    #[test]
    fn one_through_eight_ranks_have_disjoint_executor_namespaces() -> Result<()> {
        let mut ids = BTreeSet::new();
        for world in 1..=cuteafd_core::MAX_SPARK_WORLD {
            let topology = SparkTopology::new(world as u8)?;
            assert_eq!(topology.world_size(), world);
            for rank in 0..world {
                let id = topology.executor_id(rank)?;
                assert!(ids.insert(id), "executor namespace overlaps at {id}");
                assert_eq!(topology.rank_of_executor(id), Some(rank));
            }
            assert!(topology.executor_id(world).is_err());
            assert_eq!(topology.rank_of_executor(0), None);
        }
        for tp in [0, 9, u8::MAX] { assert!(SparkTopology::new(tp).is_err(), "TP{tp}"); }
        // Historical identities stay fixed.
        assert_eq!(SparkTopology::new(4)?.executor_ids(), [1, 2, 3, 4]);
        assert_eq!(SparkTopology::new(2)?.executor_ids(), [5, 6]);
        assert_eq!(SparkTopology::new(3)?.executor_ids(), [7, 8, 9]);
        assert_eq!(SparkTopology::new(6)?.executor_ids(), [27, 28, 29, 30, 31, 32]);
        assert_eq!(SparkTopology::new(1)?.executor_ids(), [33]);
        Ok(())
    }

    #[test]
    fn route_owner_bits_and_flag_18_are_rejected() -> Result<()> {
        // Bits 9..11 (the retired owner field) and every higher bit are
        // reserved: the canonical parser rejects any expert word outside the
        // checkpoint's expert range.
        for bit in 9..32 {
            let mut owned = request(1);
            owned.routes[0].expert_id |= 1 << bit;
            let frame = owned.encode()?;
            assert!(BackboneRequest::parse(&frame, 1).is_err(), "route bit {bit}");
            assert!(BackboneRequest::validate_owned(&owned, 1).is_err(), "route bit {bit}");
        }
        let mut flagged = request(1);
        flagged.header.flags |= 1 << 18;
        assert!(flagged.encode().is_err(), "request flag 1 << 18 is reserved");
        Ok(())
    }

    #[test]
    fn every_world_rejects_stale_and_foreign_responses() -> Result<()> {
        for world in 1..=cuteafd_core::MAX_SPARK_WORLD {
            let topology = SparkTopology::new(world as u8)?;
            let frame = request(2).encode()?;
            let native = BackboneRequest::parse(&frame, 2)?;
            let executors = topology.executor_ids();
            let mut receiver = V41Tp4ChunkReceiver::new_ranks(&native, &executors, 200_000)?;
            let payload = vec![0x11; native.plane_bytes()?];
            let mut stale = native.response(executors[0], &payload)?.to_owned()?;
            stale.header.request_id += 1;
            assert!(receiver.push(&stale.encode()?, |_, _, _| panic!("stale response")).is_err());
            for other in (1..=cuteafd_core::MAX_SPARK_WORLD).filter(|other| *other != world) {
                let foreign = SparkTopology::new(other as u8)?.executor_id(0)?;
                let response = native.response(foreign, &payload)?.to_owned()?.encode()?;
                assert!(receiver.push(&response, |_, _, _| panic!("foreign world")).is_err());
            }
            for rank in (0..world).rev() {
                let response = native.response(executors[rank], &payload)?.to_owned()?.encode()?;
                receiver.push(&response, |actual, start, bytes| {
                    assert_eq!((actual, start), (rank, 0));
                    assert_eq!(bytes, payload);
                    Ok(())
                })?;
                assert!(receiver.push(&response, |_, _, _| panic!("duplicate response")).is_err());
                assert_eq!(receiver.complete(), rank == 0);
            }
            assert_eq!(receiver.received_rows_slice(), vec![2; world]);
        }
        Ok(())
    }

    #[test]
    fn six_rank_chunked_coverage_rejects_reordered_overlapping_and_bad_final_markers() -> Result<()> {
        let topology = SparkTopology::new(6)?;
        let frame = request(3).encode()?;
        let native = BackboneRequest::parse(&frame, 3)?;
        let executors = topology.executor_ids();
        let budget = 200_000;
        let mut receiver = V41Tp4ChunkReceiver::new_ranks(&native, &executors, budget)?;
        let chunk = |rank: usize, start: u32, rows: u32| -> Result<Vec<u8>> {
            let payload = vec![rank as u8 + 1; rows as usize * V41_PARTIAL_ROW_BYTES as usize];
            let mut indices = vec![0u32; rows as usize];
            native.response_chunk(executors[rank], start, &payload, &mut indices, budget)?.to_owned()?.encode()
        };
        assert!(receiver.push(&chunk(0, 1, 2)?, |_, _, _| panic!("reordered rows")).is_err());
        let mut early = ExpertProtocolV2Response::decode(&chunk(0, 0, 2)?)?;
        early.header.flags &= !crate::EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS;
        assert!(receiver.push(&early.encode()?, |_, _, _| panic!("early final marker")).is_err());
        let mut late = ExpertProtocolV2Response::decode(&chunk(0, 2, 1)?)?;
        late.header.flags |= crate::EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS;
        assert!(receiver.push(&late.encode()?, |_, _, _| panic!("missing final marker")).is_err());
        for rank in [5usize, 0, 3, 1, 4, 2] {
            for (start, rows) in [(0u32, 2u32), (2, 1)] {
                let frame = chunk(rank, start, rows)?;
                let expected = vec![rank as u8 + 1; rows as usize * V41_PARTIAL_ROW_BYTES as usize];
                receiver.push(&frame, |actual, first, bytes| {
                    assert_eq!((actual, first), (rank, start));
                    assert_eq!(bytes, expected);
                    Ok(())
                })?;
            }
        }
        assert!(receiver.complete());
        assert_eq!(receiver.received_rows_slice(), [3u32; 6].as_slice());
        Ok(())
    }

    #[test]
    fn planes_collection_supports_three_and_six_ranks() -> Result<()> {
        for world in [3u8, 6] {
            let topology = SparkTopology::new(world)?;
            let frame = request(1).encode()?;
            let native = BackboneRequest::parse(&frame, 1)?;
            let executors = topology.executor_ids();
            let world = topology.world_size();
            let payloads: Vec<Vec<u8>> = (0..world)
                .map(|rank| vec![rank as u8 + 1; native.plane_bytes().unwrap()]).collect();
            let responses: Vec<Vec<u8>> = (0..world).map(|rank| native.response(executors[rank], &payloads[rank])
                .unwrap().to_owned().unwrap().encode().unwrap()).collect();
            let mut planes = V41Tp4Planes::new_ranks(&native, &executors)?;
            assert!(!planes.complete());
            assert_eq!(planes.world_size(), world);
            assert!(planes.planes().is_err());
            for rank in (0..world).rev() {
                assert_eq!(planes.insert(&responses[rank])?, rank);
                assert!(planes.insert(&responses[rank]).is_err());
            }
            assert!(planes.complete());
            for rank in 0..world { assert_eq!(planes.plane(rank)?, payloads[rank].as_slice()); }
            assert!(planes.plane(world).is_err());
        }
        Ok(())
    }

    #[test]
    fn roce_constructors_validate_world_and_identity() -> Result<()> {
        let peers: Vec<SocketAddr> = (0..9)
            .map(|index| format!("127.0.0.1:{}", 24_000 + index).parse())
            .collect::<std::result::Result<_, _>>()?;
        let config = TcpTransportConfig { timing: false, timeout: std::time::Duration::from_secs(1),
            max_frame_bytes: 200_000 };
        for world in 1..=cuteafd_core::MAX_SPARK_WORLD {
            let mut client = SparkExperts::new_world(crate::expert::SparkWorld::canonical(world)?,
                &peers[..world], 80, config.clone(), 0)?;
            assert_eq!(client.world_size(), world);
            assert_eq!(client.capacity(), 80);
            client.reset_connections();
        }
        assert!(crate::expert::SparkWorld::canonical(9).is_err());
        assert!(SparkExperts::new_world(crate::expert::SparkWorld::canonical(3)?, &peers[..2], 80, config.clone(), 0).is_err());
        assert!(SparkExperts::new_world(crate::expert::SparkWorld::canonical(3)?, &[peers[0]; 3], 80, config.clone(), 0).is_err());
        assert!(SparkExperts::new_world(crate::expert::SparkWorld::canonical(6)?, &peers[..6], 0, config, 0).is_err());
        Ok(())
    }
}
