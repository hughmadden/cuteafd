//! Shared, opt-in expert telemetry owners, independent of family placement.
use super::planner::{Allocation, ByteCurve, Component, Lifetime, Location, Refusal};
use cuteafd_core::{
    expert_usage::{device_owner, Geometry},
    memory_layout::Category,
};

/// One counters/snapshot/pinned set per possible canonical router producer,
/// plus one merged bounded host history. Replicated router peers need no set.
/// `gpus` are logical coordinator ranks, not CUDA ordinals.
pub fn components(
    geometry: &Geometry,
    gpus: &[u8],
    maximum: u64,
) -> Result<Vec<Component>, Refusal> {
    if !geometry.validate()
        || gpus.is_empty()
        || gpus
            .iter()
            .enumerate()
            .any(|(i, gpu)| gpus[..i].contains(gpu))
    {
        return Err(Refusal::InvalidFact(
            "invalid expert usage geometry/devices".into(),
        ));
    }
    let table = geometry
        .table_bytes()
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(Refusal::Overflow("expert usage tables"))?;
    let host = geometry
        .host_bytes(gpus.len())
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(Refusal::Overflow("expert usage host history"))?;
    let mut owners = Vec::new();
    let mut add = |id, location, bytes, category| {
        owners.push(Component::allocation(
            Allocation::buffer(
                id,
                location,
                bytes,
                Lifetime::TARGET,
                category,
                "expert usage",
            ),
            ByteCurve::fixed(bytes, maximum),
            false,
            maximum,
        ))
    };
    for &gpu in gpus {
        add(
            device_owner(gpu, "counters"),
            Location::coordinator(gpu),
            table,
            Category::Workspace,
        );
        add(
            device_owner(gpu, "snapshot"),
            Location::coordinator(gpu),
            table,
            Category::Workspace,
        );
        add(
            device_owner(gpu, "pinned"),
            Location::PINNED,
            table,
            Category::Staging,
        );
    }
    add(
        "expert-usage.history".into(),
        Location::HOST,
        host,
        Category::Workspace,
    );
    Ok(owners)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::planner::ComponentKind;

    #[test]
    fn all_retained_buffers_are_fixed_owners_on_canonical_devices() {
        let geometry = Geometry {
            layers: 4,
            experts: 8,
            topk: 2,
            routed_layers: vec![1, 2, 3],
        };
        for gpus in [&[1][..], &[0, 1][..]] {
            let owners = components(&geometry, gpus, 100).unwrap();
            assert_eq!(owners.len(), 3 * gpus.len() + 1);
            let mut device = 0;
            let mut pinned = 0;
            let mut host = 0;
            for owner in owners {
                let ComponentKind::Allocation { template, .. } = owner.kind else {
                    panic!("allocation");
                };
                assert_eq!(owner.bytes.at(1).unwrap(), owner.bytes.at(100).unwrap());
                if gpus
                    .iter()
                    .any(|&gpu| template.location == Location::coordinator(gpu))
                {
                    device += template.bytes;
                } else if template.location == Location::PINNED {
                    pinned += template.bytes;
                } else if template.location == Location::HOST {
                    host += template.bytes;
                } else {
                    panic!("unexpected telemetry location");
                }
            }
            assert_eq!(
                device,
                gpus.len() as u64 * 2 * geometry.table_bytes().unwrap() as u64
            );
            assert_eq!(
                pinned,
                gpus.len() as u64 * geometry.table_bytes().unwrap() as u64
            );
            assert_eq!(host, geometry.host_bytes(gpus.len()).unwrap() as u64);
        }
        assert!(components(&geometry, &[], 100).is_err());
        assert!(components(&geometry, &[0, 0], 100).is_err());
    }
}
