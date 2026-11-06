//! Shared, opt-in expert telemetry owners, independent of family placement.
use super::planner::{Allocation, ByteCurve, Component, Lifetime, Location, Refusal};
use cuteafd_core::{expert_usage::Geometry, memory_layout::Category};

/// Counter and atomic snapshot tables live on the canonical router's GPU only.
/// The pinned copy and bounded host history do not scale with the KV pool.
pub fn components(geometry: &Geometry, gpu: u8, maximum: u64) -> Result<Vec<Component>, Refusal> {
    if !geometry.validate() {
        return Err(Refusal::InvalidFact("invalid expert usage geometry".into()));
    }
    let table = geometry
        .table_bytes()
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(Refusal::Overflow("expert usage tables"))?;
    let host = geometry
        .host_bytes()
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(Refusal::Overflow("expert usage host history"))?;
    Ok([
        (
            "counters",
            Location::coordinator(gpu),
            table,
            Category::Workspace,
        ),
        (
            "snapshot",
            Location::coordinator(gpu),
            table,
            Category::Workspace,
        ),
        ("pinned", Location::PINNED, table, Category::Staging),
        ("history", Location::HOST, host, Category::Workspace),
    ]
    .into_iter()
    .map(|(name, location, bytes, category)| {
        Component::allocation(
            Allocation::buffer(
                format!("expert-usage.{name}"),
                location,
                bytes,
                Lifetime::TARGET,
                category,
                "expert usage",
            ),
            ByteCurve::fixed(bytes, maximum),
            false,
            maximum,
        )
    })
    .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::planner::ComponentKind;

    #[test]
    fn all_retained_buffers_are_fixed_owners_on_the_canonical_rank() {
        let geometry = Geometry {
            layers: 4,
            experts: 8,
            topk: 2,
            routed_layers: vec![1, 2, 3],
        };
        let owners = components(&geometry, 1, 100).unwrap();
        assert_eq!(owners.len(), 4);
        let mut device = 0;
        let mut pinned = 0;
        let mut host = 0;
        for owner in owners {
            let ComponentKind::Allocation { template, .. } = owner.kind else {
                panic!("allocation");
            };
            assert_eq!(owner.bytes.at(1).unwrap(), owner.bytes.at(100).unwrap());
            if template.location == Location::coordinator(1) {
                device += template.bytes;
            } else if template.location == Location::PINNED {
                pinned += template.bytes;
            } else if template.location == Location::HOST {
                host += template.bytes;
            } else {
                panic!("unexpected telemetry location");
            }
        }
        assert_eq!(device, 2 * geometry.table_bytes().unwrap() as u64);
        assert_eq!(pinned, geometry.table_bytes().unwrap() as u64);
        assert_eq!(host, geometry.host_bytes().unwrap() as u64);
    }
}
