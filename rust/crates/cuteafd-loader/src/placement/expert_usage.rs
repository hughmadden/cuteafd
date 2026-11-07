//! Shared expert telemetry owners, independent of family placement.
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

/// Rows of one phase the replay table spans: the widest prefill unit any family admits.
pub const REPLAY_MAX_ROWS: usize = 4096;
/// One `[prefill rows, decode rows]` pair of I32s.
pub const REPLAY_PAIR_BYTES: usize = 8;
/// The immutable pinned table of every pair a step can upload, both phases.
pub const REPLAY_LOOKUP_BYTES: usize = 2 * (REPLAY_MAX_ROWS + 1) * REPLAY_PAIR_BYTES;
/// A device pair as allocated (the allocation floor).
const REPLAY_PAIR_ALLOCATION: u64 = 256;

/// Replay metadata of a canonical router on logical GPU 0 whose tallies may be
/// captured in graphs: `{prefix}.lookup` (pinned), and one device pair for each of
/// `decode` decode and `prefill` prefill workspaces, `{prefix}.{decode|prefill}.{slot}`.
pub fn replay_components(prefix: &str, decode: u64, prefill: u64, maximum: u64)
    -> Result<Vec<Component>, Refusal> {
    let lanes = 1..=cuteafd_core::serving_capacity::PREFILL_MAX_WAVES;
    if !(1..=2).contains(&decode) || !lanes.contains(&prefill) {
        return Err(Refusal::InvalidFact("expert usage replay workspaces".into()));
    }
    let mut owners = vec![Allocation::buffer(format!("{prefix}.lookup"), Location::PINNED,
        REPLAY_LOOKUP_BYTES as u64, Lifetime::TARGET, Category::Staging, "expert usage replay metadata")];
    for (kind, slots) in [("decode", decode), ("prefill", prefill)] {
        for slot in 0..slots {
            owners.push(Allocation::buffer(format!("{prefix}.{kind}.{slot}"), Location::coordinator(0),
                REPLAY_PAIR_ALLOCATION, Lifetime::TARGET, Category::Workspace, "expert usage replay metadata"));
        }
    }
    Ok(owners.into_iter().map(|owner| {
        let bytes = owner.bytes;
        Component::allocation(owner, ByteCurve::fixed(bytes, maximum), false, maximum)
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::planner::ComponentKind;

    #[test]
    fn replay_metadata_is_one_pinned_table_and_a_fixed_pair_per_canonical_workspace() {
        let owners = replay_components("family.usage", 1, 3, 100).unwrap();
        let ids = owners.iter().map(|owner| {
            let ComponentKind::Allocation { template, .. } = &owner.kind else { panic!("allocation") };
            assert_eq!(owner.bytes.at(1).unwrap(), owner.bytes.at(100).unwrap());
            (template.id.clone(), template.location, template.bytes)
        }).collect::<Vec<_>>();
        assert_eq!(ids, [("family.usage.lookup".to_string(), Location::PINNED, REPLAY_LOOKUP_BYTES as u64),
            ("family.usage.decode.0".into(), Location::coordinator(0), 256),
            ("family.usage.prefill.0".into(), Location::coordinator(0), 256),
            ("family.usage.prefill.1".into(), Location::coordinator(0), 256),
            ("family.usage.prefill.2".into(), Location::coordinator(0), 256)]);
        assert_eq!(REPLAY_LOOKUP_BYTES, 2 * 4097 * 8);
        for (decode, prefill) in [(0, 1), (3, 1), (1, 0), (1, 5)] {
            assert!(replay_components("family.usage", decode, prefill, 100).is_err());
        }
    }

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
