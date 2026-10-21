use super::*;

#[test]
fn changed_worker_placement_is_rejected_before_plan_upload() -> Result<()> {
    let slice = SparkSlice { unit_map: Some("balanced-units".into()), layer: 3, first: 0,
        width: 256, stored_width: 256, tiers: vec![3, 4], projection_counts: vec![[1; 3]; 2] };
    let assignment = cuteafd_loader::placement::SparkAssignment { rank: 1, resident_bytes: 4096,
        slices: vec![slice.clone()], io: Default::default() };
    let expected = vec![Placement { slice, fingerprint: "source-layout-seal".into() }];
    let good = Offer { rank: 1, host: "rank-one".into(), arena_bytes: 4100, placement: expected.clone() };
    validate_offer(&good, 1, &assignment, &expected, 4100)?;
    for field in 0..6 {
        let mut bad = good.clone();
        match field {
            0 => bad.rank = 0,
            1 => bad.arena_bytes += 256,
            2 => bad.placement[0].slice.unit_map = Some("different-balanced-units".into()),
            3 => bad.placement[0].slice.first = 128,
            4 => bad.placement[0].fingerprint = "different-destination-pitch".into(),
            _ => bad.placement.reverse(),
        }
        if field == 5 { bad.placement.clear(); }
        assert!(validate_offer(&bad, 1, &assignment, &expected, 4100).is_err());
    }
    Ok(())
}

#[test]
fn rendezvous_rejects_changed_worker_placement_for_the_complete_group() -> Result<()> {
    use cuteafd_transport::worker_selection::push;
    use std::{net::{TcpListener, TcpStream}, thread, time::Duration};
    let slice = SparkSlice { unit_map: Some("balanced-units".into()), layer: 3, first: 0,
        width: 256, stored_width: 256, tiers: vec![3, 4], projection_counts: vec![[1; 3]; 2] };
    let expected = vec![Placement { slice: slice.clone(), fingerprint: "layout-seal".into() }];
    let mut clients = Vec::new();
    let mut jobs = Vec::new();
    for rank in 0..2 {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let client = TcpStream::connect(listener.local_addr()?)?;
        let (mut worker, _) = listener.accept()?;
        for stream in [&client, &worker] {
            stream.set_read_timeout(Some(Duration::from_secs(5)))?;
            stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        }
        clients.push(client);
        let mut placement = expected.clone();
        if rank == 1 { placement[0].slice.unit_map = Some("different-source-units".into()); }
        jobs.push(thread::spawn(move || -> Result<()> {
            assert!(push::offer(&mut worker, Some(Offer { rank, host: format!("rank-{rank}"),
                arena_bytes: 4100, placement }))?.is_none(), "mismatched group joined push");
            Ok(())
        }));
    }
    let mut checked = 0;
    push::coordinate(&mut clients, |offers| {
        for (rank, offer) in offers.iter().enumerate() {
            checked += 1;
            let assignment = cuteafd_loader::placement::SparkAssignment { rank: rank as u32,
                resident_bytes: 4096, slices: vec![slice.clone()], io: Default::default() };
            validate_offer(offer, rank, &assignment, &expected, 4100)?;
        }
        anyhow::bail!("fixture must reject before creating or uploading a push group")
    })?;
    assert_eq!(checked, 2, "rendezvous did not validate the changed worker report");
    for job in jobs { job.join().unwrap()?; }
    Ok(())
}

#[test]
#[ignore = "requires an installed EXL3 checkpoint selected by CUTEAFD_PUSH_EXL3"]
fn installed_exl3_complete_rank_plans_fit_admission_and_control_frames() -> Result<()> {
    let snapshot = std::path::PathBuf::from(std::env::var_os("CUTEAFD_PUSH_EXL3")
        .expect("set CUTEAFD_PUSH_EXL3 to the installed snapshot"));
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    let shape = catalog.routed_experts();
    let layers: Vec<_> = (shape.first_layer..shape.layers).collect();
    // Header-only validation also works before the trial namespace exposes
    // ADR-064; production activation still requires that attribute.
    let root = cuteafd_loader::push_load::namespace(&snapshot).map(|(root, _)| root)
        .unwrap_or_else(|_| snapshot.clone());
    for world in [4, 6] {
        for rank in 0..world {
            let plan = RankPlan::new(&catalog, &layers, world, rank, usize::MAX)?;
            let admitted = plan.residencies.iter().map(|r| r.device_arena_layout().map(|(_, n)| n))
                .collect::<Result<Vec<_>, _>>()?.into_iter().sum::<usize>();
            assert_eq!(plan.layout.bytes(), admitted);
            let offer = Offer { rank, host: "rank-host".into(),
                arena_bytes: plan.layout.logical_bytes() as u64, placement: plan.placement.clone() };
            let frame = serde_json::to_vec(&serde_json::json!({"message":"offer", "offer":offer}))?;
            assert!(frame.len() + 1 <= 64 * 1024, "complete rank offer exceeds control bound");
            let checked = plan.checked(&catalog, &root, world, rank)?;
            eprintln!("EXL3 TP{world} rank {rank}: {} layers, {} device bytes, {} validation bytes, {} entries, {} offer bytes",
                layers.len(), admitted, plan.validation.bytes(), checked.plan.entries.len(), frame.len() + 1);
        }
    }
    Ok(())
}

#[test]
fn reported_rank_plans_match_balanced_layout_and_checked_sources() -> Result<()> {
    use cuteafd_loader::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
    let directory = tempfile::tempdir()?;
    let (tensors, _) = qwen4_exl3(1, 4);
    let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
        || tensor.0.contains(".experts.0.")).collect();
    let mut config = qwen4_config(1);
    config["text_config"]["num_experts"] = serde_json::json!(1);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
    write_snapshot(directory.path(), &config, &tensors, None);
    let catalog = cuteafd_loader::read_expert_catalog(directory.path())?;
    for world in [2, 4, 5] {
        for rank in 0..world {
            let plan = RankPlan::new(&catalog, &[0], world, rank, usize::MAX)?;
            let mounted = catalog.exl3().unwrap()
                .balanced_residency(V41Exl3Layer::Backbone(0), world, rank)?;
            assert_eq!(serde_json::to_value(&mounted)?, serde_json::to_value(&plan.residencies[0])?);
            assert_eq!(mounted.unit_map.as_ref().map(|m| m.fingerprint()), plan.placement[0].slice.unit_map);
            let checked = plan.checked(&catalog, directory.path(), world, rank)?;
            assert!(!checked.plan.entries.is_empty());
            assert_eq!(checked.arenas[rank], plan.layout.logical_bytes() as u64);
        }
    }
    Ok(())
}
