use super::tests::{land, snapshot};
use super::*;

#[test]
fn real_v41_native_tp4_tp6_and_coordinator_tp2_match_staging() {
    let Some(snapshot) = snapshot("CUTEAFD_PUSH_V41") else {
        return;
    };
    let catalog = crate::read_official_v41_catalog(&snapshot).unwrap();
    for world in [4, 6] {
        let mut sizes = Vec::new();
        let mut staging = Vec::new();
        for rank in 0..world {
            let plan = catalog
                .expert_staging(crate::V41ExpertSelection::BackboneTp {
                    layer: 0,
                    expert: 0,
                    world,
                    rank,
                })
                .unwrap();
            sizes.push(plan.staging_bytes() as u64);
            staging.push(plan);
        }
        let mut builder = Builder::new(&snapshot, sizes.clone());
        for (rank, plan) in staging.iter().enumerate() {
            builder
                .v41_regions(
                    &catalog,
                    plan.tensor_names(),
                    plan.tensor_ranges(),
                    world as u32,
                    rank as u32,
                    Destination {
                        rank: rank as u32,
                        offset: 0,
                        bytes: sizes[rank],
                    },
                )
                .unwrap();
        }
        let plan = builder.finish().unwrap();
        let landed = land(&plan, &snapshot);
        for (rank, staging) in staging.iter().enumerate() {
            let mut expected = vec![0; staging.staging_bytes()];
            let mut scratch = vec![0; staging.minimum_read_scratch_bytes()];
            staging.read_into(&mut expected, &mut scratch).unwrap();
            assert_eq!(expected, landed[rank], "V41 MXFP4 TP{world} rank {rank}");
        }
        eprintln!(
            "V41 MXFP4 TP{world}: {} entries byte-exact",
            plan.plan.entries.len()
        );
    }
    let tensor = catalog
        .tensors()
        .iter()
        .find(|t| {
            matches!(t.placement, crate::V41TensorPlacement::CoordinatorRtx)
                && t.metadata.shape.len() == 2
                && t.metadata.shape.iter().all(|n| n % 2 == 0)
                && t.metadata.byte_length < 64 << 20
        })
        .unwrap();
    let name = &tensor.metadata.name;
    let source = Builder::v41_source(&catalog, name).unwrap();
    for axis in 0..2 {
        let mut builder = Builder::new(&snapshot, vec![source.bytes / 2; 2]);
        for rank in 0..2 {
            builder
                .matrix_partition(
                    &source,
                    tensor.metadata.shape[0] as u32,
                    axis,
                    2,
                    rank,
                    Destination {
                        rank,
                        offset: 0,
                        bytes: source.bytes / 2,
                    },
                )
                .unwrap();
        }
        let plan = builder.finish().unwrap();
        for (rank, landed) in land(&plan, &snapshot).iter().enumerate() {
            let mut expected = vec![0; landed.len()];
            let mut scratch = vec![0; source.bytes as usize / tensor.metadata.shape[0]];
            catalog
                .read_coordinator_tp2_into(name, axis, rank, &mut expected, &mut scratch)
                .unwrap();
            assert_eq!(
                &expected, landed,
                "coordinator {name} TP2 axis{axis} rank{rank}"
            );
        }
    }
    let mut builder = Builder::new(&snapshot, vec![source.bytes]);
    builder
        .contiguous(
            &source,
            0,
            source.bytes,
            Destination {
                rank: 0,
                offset: 0,
                bytes: source.bytes,
            },
        )
        .unwrap();
    let landed = land(&builder.finish().unwrap(), &snapshot).remove(0);
    let mut expected = vec![0; source.bytes as usize];
    catalog.read_projection_once(name, &mut expected).unwrap();
    assert_eq!(landed, expected);
    eprintln!("Coordinator whole and TP2 axes: byte-exact");
}

#[test]
fn real_v41_nvfp4_tp4_tp6_match_staging() {
    let Some(snapshot) = snapshot("CUTEAFD_PUSH_V41_NVFP4") else {
        return;
    };
    let catalog = crate::read_official_v41_catalog(&snapshot).unwrap();
    for world in [4, 6] {
        let staging = (0..world)
            .map(|rank| {
                catalog
                    .nvfp4_expert_staging(crate::V41ExpertSelection::BackboneTp {
                        layer: 0,
                        expert: 0,
                        world,
                        rank,
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let sizes = staging
            .iter()
            .map(|s| s.staging_bytes() as u64)
            .collect::<Vec<_>>();
        let mut builder = Builder::new(&snapshot, sizes.clone());
        for (rank, s) in staging.iter().enumerate() {
            builder
                .v41_regions(
                    &catalog,
                    s.tensor_names(),
                    s.tensor_ranges(),
                    world as u32,
                    rank as u32,
                    Destination {
                        rank: rank as u32,
                        offset: 0,
                        bytes: sizes[rank],
                    },
                )
                .unwrap();
        }
        let plan = builder.finish().unwrap();
        for (s, landed) in staging.iter().zip(land(&plan, &snapshot)) {
            let mut expected = vec![0; s.staging_bytes()];
            let mut scratch = vec![0; s.minimum_read_scratch_bytes()];
            s.read_into(&mut expected, &mut scratch).unwrap();
            assert_eq!(landed, expected);
        }
        eprintln!(
            "V41 NVFP4 TP{world}: {} entries byte-exact",
            plan.plan.entries.len()
        );
    }
}
