use super::*;

#[test]
fn native_plane_geometry_keeps_tp4_tp6_mount_admission() -> Result<()> {
    for nvfp4 in [false, true] {
        for (world, stored) in [(4, 640), (6, 384)] {
            let planes = NativePlanes::new(5120, 2304 / world, stored, 384, nvfp4)?;
            let resident = cuteafd_loader::placement::families::deepseek_v41::resident_bytes(
                5120, stored as u64, 384, nvfp4)?;
            assert_eq!(planes.bytes as u64, resident);
            let layout = ArenaLayout::new([planes.bytes, planes.bytes], planes.bytes * 2)?;
            assert_eq!(layout.region(1)?.start, planes.bytes);
            assert!(ArenaLayout::new([planes.bytes], planes.bytes - 1).is_err());
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn installed_native_rank_plan_tp4_tp6_rejects_changed_worker_placement() -> Result<()> {
    let root = std::env::var_os("CUTEAFD_PUSH_V41").context("set CUTEAFD_PUSH_V41")?;
    let root = Path::new(&root);
    let catalog = cuteafd_loader::read_expert_catalog(root)?;
    let geometry = catalog.routed_experts();
    for world in [4, 6] {
        for rank in 0..world {
            let width = geometry.intermediate as usize / world;
            let stored = width.div_ceil(128) * 128;
            let slice = SparkSlice { unit_map: None, layer: 0, first: (rank * width) as u32,
                width: width as u32, stored_width: stored as u32, tiers: vec![], projection_counts: vec![] };
            let resident = usize::try_from(cuteafd_loader::placement::families::deepseek_v41::resident_bytes(
                geometry.hidden as u64, stored as u64, geometry.experts as u64, false)?)?;
            let plan = RankPlan::new(&catalog, &[slice.clone()], world, rank, resident)?;
            let checked = plan.checked(&catalog, root, world, rank)?;
            checked.plan.check(&checked.file_sizes, &checked.arenas)?;
            assert_eq!(plan.placement[0].slice, slice);
            let assignment = cuteafd_loader::placement::SparkAssignment { rank: rank as u32,
                resident_bytes: resident as u64, slices: vec![slice.clone()], io: Default::default() };
            let offer = Offer { rank, host: format!("rank-{rank}"),
                arena_bytes: plan.layout.logical_bytes() as u64, placement: plan.placement.clone() };
            super::super::push_exl3::validate_offer(&offer, rank, &assignment, &plan.placement, offer.arena_bytes)?;
            let mut bad = offer.clone();
            bad.placement[0].slice.first += 32;
            assert!(super::super::push_exl3::validate_offer(&bad, rank, &assignment, &plan.placement, offer.arena_bytes).is_err());
            bad = offer.clone();
            bad.placement[0].fingerprint.push('0');
            assert!(super::super::push_exl3::validate_offer(&bad, rank, &assignment, &plan.placement, offer.arena_bytes).is_err());
            let mut changed = slice.clone();
            changed.first += 32;
            assert!(RankPlan::new(&catalog, &[changed], world, rank, resident).is_err());
            let mut changed = slice;
            changed.width -= 32;
            assert!(RankPlan::new(&catalog, &[changed], world, rank, resident).is_err());
            assert!(RankPlan::new(&catalog, &[plan.placement[0].slice.clone()], world, rank, resident - 1).is_err());
            eprintln!("native TP{world} rank={rank}: checked {} entries, resident={resident}, scratch={}",
                checked.plan.entries.len(), plan.scratch);
        }
    }
    Ok(())
}
