//! Coordinator-owned routes of a GPU expert tier.
//!
//! A tier keeps some routed experts of each layer resident on the coordinator GPU while the
//! Sparks keep every expert. Per layer and step the coordinator runs every active resident
//! expert itself and marks those routes in the Spark request: bit 31 of the route word, legal
//! only under [`EXPERT_PROTOCOL_V2_FLAG_COORDINATOR_ROUTES`]. The batch keeps its canonical
//! shape (top-k routes per row, order and FP32 weights unchanged, same frame size); a worker
//! unpacks a marked route as the unassigned expert (`experts`, one past the last id) with zero
//! weight, which the expert kernels' route packing drops, so a resident expert costs the
//! Sparks no weight bytes. Responses never carry the flag.
use crate::{ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, EXPERT_PROTOCOL_V2_FLAG_COORDINATOR_ROUTES};
use anyhow::{ensure, Result};

/// Route-word bit of a coordinator-owned route.
pub const COORDINATOR_ROUTE_BIT: u32 = 1 << 31;

/// Request-only admission flags: never echoed in a response.
pub(crate) const REQUEST_ONLY_FLAGS: u32 =
    super::V41_NATIVE_GROUP_REQUEST_FLAG | EXPERT_PROTOCOL_V2_FLAG_COORDINATOR_ROUTES;

/// One decoded route word: the expert id and whether the coordinator owns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinatorRouteWord {
    pub expert_id: u32,
    pub coordinator: bool,
}

impl CoordinatorRouteWord {
    /// Decode `word` from a request with `flags`. Without the flag the word is the plain id.
    pub fn decode(word: u32, flags: u32) -> Self {
        if flags & EXPERT_PROTOCOL_V2_FLAG_COORDINATOR_ROUTES != 0 {
            Self { expert_id: word & !COORDINATOR_ROUTE_BIT, coordinator: word & COORDINATOR_ROUTE_BIT != 0 }
        } else {
            Self { expert_id: word, coordinator: false }
        }
    }

    pub fn encode(self) -> u32 {
        if self.coordinator { self.expert_id | COORDINATOR_ROUTE_BIT } else { self.expert_id }
    }
}

/// Whether `request` admits coordinator-owned routes.
pub fn coordinator_routes(flags: u32) -> bool {
    flags & EXPERT_PROTOCOL_V2_FLAG_COORDINATOR_ROUTES != 0
}

/// Mark every route of `routes` whose expert `resident` accepts as coordinator-owned and
/// return how many were marked. Ids, order and gate weights are unchanged.
pub fn mark_coordinator_routes(routes: &mut [ExpertProtocolV2RouteEntry], experts: u32,
    resident: impl Fn(u32) -> bool) -> Result<usize> {
    let mut marked = 0;
    for route in routes {
        ensure!(route.expert_id < experts && route.expert_id & COORDINATOR_ROUTE_BIT == 0,
            "coordinator route marking needs plain expert ids below {experts}");
        if resident(route.expert_id) {
            route.expert_id |= COORDINATOR_ROUTE_BIT;
            marked += 1;
        }
    }
    Ok(marked)
}

/// Flag `request` for coordinator-owned routes and mark its resident ones (see
/// [`mark_coordinator_routes`]). A request already flagged is refused.
pub fn with_coordinator_routes(request: &mut ExpertProtocolV2Request, experts: u32,
    resident: impl Fn(u32) -> bool) -> Result<usize> {
    ensure!(!coordinator_routes(request.header.flags), "request already carries coordinator routes");
    let marked = mark_coordinator_routes(&mut request.routes, experts, resident)?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_COORDINATOR_ROUTES;
    Ok(marked)
}

/// The id and weight a worker's kernels receive for one route: a coordinator route becomes
/// the unassigned expert `experts` with zero weight.
pub(crate) fn unpacked(route: &ExpertProtocolV2RouteEntry, flags: u32, experts: u32) -> (i32, f32) {
    let word = CoordinatorRouteWord::decode(route.expert_id, flags);
    if word.coordinator { (experts as i32, 0.0) } else { (word.expert_id as i32, route.gate_weight) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind};

    fn request(rows: u32, topk: u32, experts: u32) -> ExpertProtocolV2Request {
        let hidden = 64;
        ExpertProtocolV2Request::new(9, 17, 3, hidden, ExpertV2Dtype::Bf16,
            (0..rows).map(|row| ExpertProtocolV2RowDescriptor { row_id: u64::from(row),
                source_kind: ExpertV2SourceKind::Decode, source_request_id: 1, token_position: u64::from(row),
                route_offset: row * topk, route_count: topk }).collect(),
            (0..rows * topk).map(|route| ExpertProtocolV2RouteEntry { row_index: route / topk,
                expert_id: (route * 7 + route / topk) % experts, gate_weight: 0.125 + route as f32 }).collect(),
            vec![0; (rows * hidden * 2) as usize]).unwrap()
    }

    #[test]
    fn route_words_round_trip_and_ignore_the_bit_without_the_flag() {
        let flag = EXPERT_PROTOCOL_V2_FLAG_COORDINATOR_ROUTES;
        for expert_id in [0, 1, 287, 383, 1023] {
            for coordinator in [false, true] {
                let word = CoordinatorRouteWord { expert_id, coordinator };
                assert_eq!(CoordinatorRouteWord::decode(word.encode(), flag), word);
            }
            assert_eq!(CoordinatorRouteWord::decode(expert_id, 0), CoordinatorRouteWord { expert_id, coordinator: false });
            // Without the flag a marked word is just an out-of-range id the validators refuse.
            assert_eq!(CoordinatorRouteWord::decode(expert_id | COORDINATOR_ROUTE_BIT, 0).expert_id,
                expert_id | COORDINATOR_ROUTE_BIT);
        }
    }

    #[test]
    fn marking_keeps_shape_order_and_weights_and_unpacks_to_the_unassigned_expert() {
        let experts = 288;
        let mut marked = request(5, 8, experts);
        let plain = marked.clone();
        let resident = |id: u32| id % 3 == 0;
        let count = with_coordinator_routes(&mut marked, experts, resident).unwrap();
        assert_eq!(count, plain.routes.iter().filter(|r| resident(r.expert_id)).count());
        assert!(count > 0 && count < plain.routes.len());
        assert_eq!(marked.header.flags, plain.header.flags | EXPERT_PROTOCOL_V2_FLAG_COORDINATOR_ROUTES);
        assert_eq!(marked.routes.len(), plain.routes.len());
        assert_eq!(marked.encode().unwrap().len(), plain.encode().unwrap().len(), "frame size is unchanged");
        for (a, b) in marked.routes.iter().zip(&plain.routes) {
            assert_eq!(a.row_index, b.row_index);
            assert_eq!(a.gate_weight.to_bits(), b.gate_weight.to_bits());
            let word = CoordinatorRouteWord::decode(a.expert_id, marked.header.flags);
            assert_eq!((word.expert_id, word.coordinator), (b.expert_id, resident(b.expert_id)));
            let (id, weight) = unpacked(a, marked.header.flags, experts);
            if resident(b.expert_id) {
                assert_eq!((id, weight.to_bits()), (experts as i32, 0f32.to_bits()));
            } else {
                assert_eq!((id, weight.to_bits()), (b.expert_id as i32, b.gate_weight.to_bits()));
            }
        }
        assert!(with_coordinator_routes(&mut marked, experts, resident).is_err(), "marking twice is refused");
        let mut wide = request(1, 8, experts);
        wide.routes[0].expert_id = experts;
        assert!(with_coordinator_routes(&mut wide, experts, |_| true).is_err(), "ids must be in range");
    }
}
