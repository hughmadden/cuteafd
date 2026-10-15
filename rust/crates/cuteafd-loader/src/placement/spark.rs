//! Component selection and per-selected-rank expert transport accounting.
use super::{Demand, PlacementError};
use cuteafd_core::ExpertGeometry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkingSetComponent { Always, RtxExperts, SparkExperts }

#[derive(Debug, Clone)]
pub struct ComponentDemand {
    pub component: WorkingSetComponent,
    pub demand: Demand,
}

impl ComponentDemand {
    pub fn new(component: WorkingSetComponent, gpu: u8,
        category: cuteafd_core::memory_layout::Category, group: &str, bytes: u64,
        basis: cuteafd_core::memory_layout::Basis) -> Self {
        Self { component, demand: Demand::new(gpu, category, group, bytes, basis) }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SparkWorkingSet {
    pub rtx_layers: Vec<usize>,
    pub spark_layers: Vec<usize>,
    pub spark_ranks: usize,
    pub prefill_lanes: u64,
}

impl SparkWorkingSet {
    /// Endpoint rings are summed; one GPU owner shares intake and exchange
    /// across all its QPs, so those allocations are maxima, not lane sums.
    pub fn rank_endpoints_io(&self, rank: usize, geometry: ExpertGeometry,
        endpoints: &[cuteafd_core::SparkEndpoint], rings: cuteafd_core::SparkRings)
        -> Result<SparkIoBytes, PlacementError> {
        if !self.selected_ranks().contains(&rank) { return Ok(SparkIoBytes::default()); }
        if endpoints.is_empty() || endpoints.len() > 8 {
            return Err(PlacementError::Inventory("selected Spark needs 1..=8 endpoints"));
        }
        let mut bytes = SparkIoBytes::default();
        for endpoint in endpoints {
            let ring = SparkRingGeometry { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
                slot_bytes: rings.slot_bytes, alignment: rings.alignment, endpoints: 1 };
            let bf16 = endpoint.ingress == cuteafd_core::SparkIngress::Bf16;
            let io = self.rank_io(rank, geometry, endpoint.capacity, bf16, ring)?;
            bytes.intake = bytes.intake.max(io.intake);
            if let cuteafd_core::SparkIngress::Bf16Decode { rows } = endpoint.ingress {
                let decode = self.rank_io(rank, geometry, rows, true, ring)?;
                bytes.intake = bytes.intake.max(decode.intake);
            }
            bytes.host_exchange = bytes.host_exchange.max(io.host_exchange);
            bytes.row_indices = bytes.row_indices.max(io.row_indices);
            let mapped = endpoint.ring_bytes(geometry, rings)
                .ok_or(PlacementError::Inventory("invalid Spark endpoint ring geometry"))? as u64;
            bytes.mapped_rings = bytes.mapped_rings.checked_add(mapped)
                .ok_or(PlacementError::Overflow("Spark endpoint rings"))?;
        }
        Ok(bytes)
    }

    pub fn includes(&self, component: WorkingSetComponent) -> bool {
        match component {
            WorkingSetComponent::Always => true,
            WorkingSetComponent::RtxExperts => !self.rtx_layers.is_empty(),
            WorkingSetComponent::SparkExperts => !self.spark_layers.is_empty() && self.spark_ranks > 0,
        }
    }

    pub fn selected_ranks(&self) -> std::ops::Range<usize> {
        0..if self.includes(WorkingSetComponent::SparkExperts) { self.spark_ranks } else { 0 }
    }

    pub fn demands<'a>(&'a self, rows: &'a [ComponentDemand]) -> impl Iterator<Item = &'a Demand> + 'a {
        rows.iter().filter(|row| self.includes(row.component)).map(|row| &row.demand)
    }

    /// Coordinator staging is shared across ranks, but not across prefill lanes.
    pub fn coordinator_staging(&self, geometry: ExpertGeometry, capacity: u32, bf16: bool)
        -> Result<u64, PlacementError> {
        if self.selected_ranks().is_empty() { return Ok(0); }
        if self.prefill_lanes == 0 {
            return Err(PlacementError::Inventory("selected Spark staging needs nonzero lanes"));
        }
        let hidden = geometry.hidden as u64;
        let input = if bf16 { hidden * 2 } else { hidden + hidden / 32 };
        (input + geometry.topk as u64 * 8).checked_mul(capacity as u64)
            .and_then(|bytes| bytes.checked_mul(self.prefill_lanes))
            .ok_or(PlacementError::Overflow("Spark coordinator staging"))
    }

    /// Each selected rank owns its intake/exchange and the stated endpoint rings.
    /// Unselected ranks (including every rank of an empty set) charge nothing.
    pub fn rank_io(&self, rank: usize, geometry: ExpertGeometry, capacity: u32,
        bf16: bool, rings: SparkRingGeometry) -> Result<SparkIoBytes, PlacementError> {
        if !self.selected_ranks().contains(&rank) { return Ok(SparkIoBytes::default()); }
        let rows = capacity as u64;
        let hidden = geometry.hidden as u64;
        let routes = geometry.topk as u64 * 8;
        let input = if bf16 { hidden * 2 } else { hidden + hidden / 32 };
        let multiply = |stride: u64| rows.checked_mul(stride)
            .ok_or(PlacementError::Overflow("Spark intake/exchange"));
        let mapped = geometry.compact_ring_bytes(capacity, rings.max_frame_bytes,
            rings.depth, rings.slot_bytes, rings.alignment, rings.endpoints, bf16)
            .ok_or(PlacementError::Inventory("Spark ring geometry exceeds frame budget"))?;
        Ok(SparkIoBytes {
            intake: multiply(input + routes + hidden * 2)?,
            host_exchange: multiply(routes + hidden * 2)?,
            row_indices: multiply(4)?,
            mapped_rings: mapped as u64,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SparkRingGeometry {
    pub max_frame_bytes: usize,
    pub depth: usize,
    pub slot_bytes: usize,
    pub alignment: usize,
    pub endpoints: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SparkIoBytes {
    /// Input rows, expert IDs, gate weights and compact BF16 output rows.
    pub intake: u64,
    pub host_exchange: u64,
    pub row_indices: u64,
    pub mapped_rings: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::memory_layout::{Basis, Category};

    #[test]
    fn endpoint_io_sums_rings_but_shares_owner_buffers() {
        use cuteafd_core::{SparkEndpoint, SparkIngress, SparkRings};
        let set = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 6,
            prefill_lanes: 3, ..Default::default() };
        let geometry = ExpertGeometry::MIMO_V26_PRO;
        let rings = SparkRings { max_frame_bytes: 64 << 20, depth: 8,
            slot_bytes: 8 << 20, alignment: 4096 };
        let main = SparkEndpoint { capacity: 4096, ingress: SparkIngress::Bf16Decode { rows: 80 } };
        let lane = SparkEndpoint { capacity: 1024, ingress: SparkIngress::Fp8K32 };
        let io = set.rank_endpoints_io(0, geometry, &[main, lane], rings).unwrap();
        assert_eq!(io.mapped_rings, (main.ring_bytes(geometry, rings).unwrap()
            + lane.ring_bytes(geometry, rings).unwrap()) as u64);
        let full_fp8 = set.rank_io(0, geometry, 4096, false, two_endpoints()).unwrap();
        assert_eq!(io.intake, full_fp8.intake);
        assert_eq!(io.host_exchange, full_fp8.host_exchange);
        assert_eq!(io.row_indices, full_fp8.row_indices);
        assert!(main.ring_bytes(geometry, rings).unwrap()
            < SparkEndpoint { ingress: SparkIngress::Bf16, ..main }.ring_bytes(geometry, rings).unwrap());
        assert!(set.rank_endpoints_io(0, geometry, &[], rings).is_err());
        assert!(set.rank_endpoints_io(0, geometry, &[SparkEndpoint {
            ingress: SparkIngress::Bf16Decode { rows: 4097 }, ..main }], rings).is_err());
        assert_eq!(set.rank_endpoints_io(6, geometry, &[], rings).unwrap(), SparkIoBytes::default());
    }

    fn two_endpoints() -> SparkRingGeometry {
        SparkRingGeometry { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20,
            alignment: 4096, endpoints: 2 }
    }

    #[test]
    fn empty_and_unselected_components_charge_zero() {
        let empty = SparkWorkingSet::default();
        assert!(empty.selected_ranks().is_empty());
        assert_eq!(empty.rank_io(0, ExpertGeometry::GLM5, 4096, true,
            two_endpoints()).unwrap(), SparkIoBytes::default());
        assert_eq!(empty.coordinator_staging(ExpertGeometry::GLM5, 4096, true).unwrap(), 0);
        let rows: Vec<_> = [WorkingSetComponent::Always, WorkingSetComponent::RtxExperts,
            WorkingSetComponent::SparkExperts].into_iter().map(|component|
            ComponentDemand { component, demand: Demand::new(0, Category::Workspace,
                "selected rows", 100, Basis::Formula) }).collect();
        assert_eq!(empty.demands(&rows).map(|d| d.bytes).sum::<u64>(), 100);
        let local = SparkWorkingSet { rtx_layers: vec![3], ..empty };
        assert_eq!(local.demands(&rows).map(|d| d.bytes).sum::<u64>(), 200);
    }

    #[test]
    fn bf16_io_is_per_selected_rank_not_world_scaled_or_lane_replicated() {
        for geometry in [ExpertGeometry::DEEPSEEK_V41, ExpertGeometry::GLM5,
            ExpertGeometry::GLM5_FLASH, ExpertGeometry::QWEN4] {
            for ranks in [2, 3, 4, 6] {
                let set = SparkWorkingSet { spark_layers: vec![3], spark_ranks: ranks,
                    prefill_lanes: 2, ..SparkWorkingSet::default() };
                let io = set.rank_io(0, geometry, 4096, true, two_endpoints()).unwrap();
                assert_eq!(io.intake, 4096 * (geometry.hidden as u64 * 4 + geometry.topk as u64 * 8));
                assert_eq!(io.host_exchange, 4096 * (geometry.hidden as u64 * 2 + geometry.topk as u64 * 8));
                assert_eq!(io.mapped_rings, geometry.compact_ring_bytes(4096, 64 << 20,
                    8, 8 << 20, 4096, 2, true).unwrap() as u64);
                assert_eq!(set.rank_io(ranks - 1, geometry, 4096, true,
                    two_endpoints()).unwrap(), io);
                assert_eq!(set.rank_io(ranks, geometry, 4096, true,
                    two_endpoints()).unwrap(), SparkIoBytes::default());
                let mut serial = set.clone(); serial.prefill_lanes = 1;
                assert_eq!(set.coordinator_staging(geometry, 4096, true).unwrap(),
                    serial.coordinator_staging(geometry, 4096, true).unwrap() * 2);
            }
        }
    }

    #[test]
    fn selected_coordinator_staging_requires_nonzero_lanes() {
        let set = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4,
            ..SparkWorkingSet::default() };
        assert!(set.coordinator_staging(ExpertGeometry::GLM5, 4096, true).is_err());
        let unselected = SparkWorkingSet { spark_layers: vec![], ..set };
        assert_eq!(unselected.coordinator_staging(ExpertGeometry::GLM5, 4096, true).unwrap(), 0);
    }

    #[test]
    fn legacy_wire_and_bad_geometry_do_not_silently_become_bf16() {
        let set = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4,
            prefill_lanes: 1, ..SparkWorkingSet::default() };
        let g = ExpertGeometry::GLM5;
        let bf16 = set.rank_io(0, g, 4096, true, two_endpoints()).unwrap();
        let wire = set.rank_io(0, g, 4096, false, two_endpoints()).unwrap();
        assert_eq!(bf16.intake - wire.intake, 4096 * (6144 - 6144 / 32));
        assert!(bf16.mapped_rings > wire.mapped_rings);
        assert!(set.rank_io(0, g, 4096, true, SparkRingGeometry {
            max_frame_bytes: 1024, ..two_endpoints() }).is_err());
        assert!(set.rank_io(0, g, 4096, true, SparkRingGeometry {
            alignment: 0, ..two_endpoints() }).is_err());
    }
}
