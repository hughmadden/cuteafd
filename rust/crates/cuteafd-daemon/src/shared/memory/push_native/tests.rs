use super::*;
use std::os::unix::fs::FileExt;

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
    let nvfp4 = catalog.nvfp4().is_some();
    for world in [4, 6] {
        let mut plans = Vec::new();
        for rank in 0..world {
            let width = geometry.intermediate as usize / world;
            let stored = width.div_ceil(128) * 128;
            let slice = SparkSlice { unit_map: None, layer: 0, first: (rank * width) as u32,
                width: width as u32, stored_width: stored as u32, tiers: vec![], projection_counts: vec![] };
            let resident = usize::try_from(cuteafd_loader::placement::families::deepseek_v41::resident_bytes(
                geometry.hidden as u64, stored as u64, geometry.experts as u64, nvfp4)?)?;
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
            plans.push((resident as u64, plan.placement.clone(), checked));
        }
        super::super::push_exl3::tests::coordinator_suffix(root, &catalog, &plans,
            |assignments, offers, coordinator| checked_plan(root, &catalog, assignments, offers, coordinator))?;
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

/// The coordinator's native TP2 halves (a coordinator rank of any push group)
/// land the mount's own BackboneTp2 staging bytes in each expert's plane
/// slots before the post-Done pack: W3 then W1 rows of the half (each padded
/// to the kernel width), the W2 column half, and their scales. Synthetic
/// MXFP4 headers, and the installed checkpoint when CUTEAFD_PUSH_V41 is set.
#[test]
fn coordinator_native_tp2_halves_land_the_mount_staging_bytes() -> Result<()> {
    use cuteafd_loader::plan::testing::{t, write_snapshot};
    let synthetic = tempfile::tempdir()?;
    let config = serde_json::json!({"model_type":"deepseek_v4"});
    let args = serde_json::json!({
        "vocab_size":64,"dim":256,"moe_inter_dim":512,"n_layers":2,"n_heads":4,
        "n_routed_experts":2,"n_shared_experts":1,"n_activated_experts":1,
        "score_func":"sqrtsoftplus","route_scale":1.5,"swiglu_limit":10.0,
        "q_lora_rank":128,"head_dim":128,"rope_head_dim":64,"o_groups":1,
        "o_lora_rank":128,"window_size":128,"original_seq_len":65536,
        "rope_theta":10000,"rope_factor":16,"beta_fast":32,"beta_slow":1,
        "index_n_heads":4,"index_head_dim":128,"index_topk":512,"hc_mult":4,
        "hc_sinkhorn_iters":20,"compress_rope_theta":160000,"compress_ratios":[0, 0]
    });
    let mut tensors = Vec::new();
    for layer in 0..2 {
        for expert in 0..2 {
            for (projection, rows, cols) in [("w1", 512, 256), ("w3", 512, 256), ("w2", 256, 512)] {
                let prefix = format!("layers.{layer}.ffn.experts.{expert}.{projection}");
                tensors.push(t(format!("{prefix}.weight"), "I8", &[rows, cols / 2]));
                tensors.push(t(format!("{prefix}.scale"), "F8_E8M0", &[rows, cols / 32]));
            }
        }
    }
    write_snapshot(synthetic.path(), &config, &tensors, None);
    std::fs::create_dir(synthetic.path().join("inference"))?;
    std::fs::write(synthetic.path().join("inference/config.json"), serde_json::to_vec(&args)?)?;
    // Distinct bytes everywhere, so a misplaced run cannot match.
    for entry in std::fs::read_dir(synthetic.path())? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "safetensors") {
            let mut bytes = std::fs::read(&path)?;
            let header = 8 + u64::from_le_bytes(bytes[..8].try_into()?) as usize;
            for (i, b) in bytes[header..].iter_mut().enumerate() { *b = (i.wrapping_mul(131) % 251) as u8; }
            std::fs::write(&path, bytes)?;
        }
    }
    let mut roots = vec![(synthetic.path().to_path_buf(), vec![0usize, 1])];
    if let Some(root) = std::env::var_os("CUTEAFD_PUSH_V41") { roots.push((root.into(), vec![0, 7])); }
    for (root, layers) in roots {
        let catalog = cuteafd_loader::read_expert_catalog(&root)?;
        for gpu in 0..2 {
            let plan = super::super::push_coordinator::Tp2ExpertPlan::new(&catalog, &layers, gpu, usize::MAX)?;
            assert!(!plan.native.is_empty() && plan.validation.bytes() == 0);
            let world = 6;
            let checked = plan.checked(&catalog, &root, world, 4 + gpu)?;
            checked.plan.check(&checked.file_sizes, &checked.arenas)?;
            // Land every run of the plan into a host image of the rank.
            let mut image = vec![0u8; plan.layout.logical_bytes()];
            let mut files = std::collections::HashMap::new();
            for entry in checked.plan.entries.iter().filter(|e| e.rank as usize == 4 + gpu) {
                let file = files.entry(entry.file).or_insert_with(||
                    std::fs::File::open(root.join(&checked.plan.files[entry.file as usize])).unwrap());
                let pattern = &checked.plan.patterns[entry.pattern as usize];
                for (src, dst) in pattern.runs() {
                    let at = (entry.dst_base + dst) as usize;
                    file.read_exact_at(&mut image[at..at + pattern.run as usize], entry.src_base + src)?;
                }
            }
            for (region, (&layer, planes)) in layers.iter().zip(&plan.native).enumerate() {
                let base = plan.layout.region(region)?.start;
                for expert in [0, planes.experts - 1] {
                    let staging = catalog.expert_staging(cuteafd_loader::V41ExpertSelection::BackboneTp2 {
                        layer, expert, rank: gpu })?;
                    let mut staged = vec![0u8; staging.staging_bytes()];
                    let mut scratch = vec![0u8; staging.minimum_read_scratch_bytes().max(1 << 20)];
                    staging.read_into(&mut staged, &mut scratch)?;
                    let tensor = |slot: usize| &staged[staging.tensor_ranges()[slot].clone()];
                    for (plane, (first, second)) in [(0, (1, 0)), (1, (4, 3))] {
                        let (offset, bytes) = planes.expert_plane(plane, expert)?;
                        let slot = &image[base + offset..base + offset + bytes];
                        let half = bytes / 2;
                        let (a, b) = (tensor(first), tensor(second));
                        assert_eq!(&slot[..a.len()], a, "layer {layer} expert {expert} plane {plane} up rows");
                        assert_eq!(&slot[half..half + b.len()], b, "layer {layer} expert {expert} plane {plane} gate rows");
                    }
                    for (plane, source) in [(2, 2), (3, 5)] {
                        let (offset, bytes) = planes.expert_plane(plane, expert)?;
                        let expected = tensor(source);
                        assert_eq!(&image[base + offset..base + offset + expected.len()], expected,
                            "layer {layer} expert {expert} plane {plane}");
                        assert!(expected.len() <= bytes);
                    }
                }
            }
            eprintln!("{}: gpu{gpu} native TP2 halves land mount staging ({} entries)", root.display(),
                checked.plan.entries.len());
        }
    }
    Ok(())
}
