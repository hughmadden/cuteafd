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

#[test]
fn coordinator_suffix_preserves_native_spark_plans() -> Result<()> {
    use cuteafd_loader::plan::testing::{t, write_snapshot};
    let root = tempfile::tempdir()?;
    let config = serde_json::json!({"model_type":"deepseek_v4"});
    let args = serde_json::json!({
        "vocab_size":64,"dim":256,"moe_inter_dim":768,"n_layers":1,"n_heads":4,
        "n_routed_experts":2,"n_shared_experts":1,"n_activated_experts":1,
        "score_func":"sqrtsoftplus","route_scale":1.5,"swiglu_limit":10.0,
        "q_lora_rank":128,"head_dim":128,"rope_head_dim":64,"o_groups":1,
        "o_lora_rank":128,"window_size":128,"original_seq_len":65536,
        "rope_theta":10000,"rope_factor":16,"beta_fast":32,"beta_slow":1,
        "index_n_heads":4,"index_head_dim":128,"index_topk":512,"hc_mult":4,
        "hc_sinkhorn_iters":20,"compress_rope_theta":160000,"compress_ratios":[0]
    });
    let mut tensors = Vec::new();
    for expert in 0..2 {
        for (projection, rows, cols) in [("w1", 768, 256), ("w3", 768, 256), ("w2", 256, 768)] {
            let prefix = format!("layers.0.ffn.experts.{expert}.{projection}");
            tensors.push(t(format!("{prefix}.weight"), "I8", &[rows, cols / 2]));
            tensors.push(t(format!("{prefix}.scale"), "F8_E8M0", &[rows, cols / 32]));
        }
    }
    write_snapshot(root.path(), &config, &tensors, None);
    std::fs::create_dir(root.path().join("inference"))?;
    std::fs::write(root.path().join("inference/config.json"), serde_json::to_vec(&args)?)?;
    let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
    for world in [4, 6] {
        let plans = (0..world).map(|rank| {
            let width: usize = 768 / world;
            let stored = width.div_ceil(128) * 128;
            let slice = SparkSlice { unit_map: None, layer: 0, first: (rank * width) as u32,
                width: width as u32, stored_width: stored as u32, tiers: vec![], projection_counts: vec![] };
            let bytes = NativePlanes::new(256, width, stored, 2, false)?.bytes;
            let plan = RankPlan::new(&catalog, &[slice], world, rank, bytes)?;
            Ok((plan.layout.bytes() as u64, plan.placement.clone(),
                plan.checked(&catalog, root.path(), world, rank)?))
        }).collect::<Result<Vec<_>>>()?;
        super::super::push_exl3::tests::coordinator_suffix(root.path(), &catalog, &plans,
            |assignments, offers, coordinator| checked_plan(root.path(), &catalog, assignments, offers, coordinator))?;
    }
    Ok(())
}
