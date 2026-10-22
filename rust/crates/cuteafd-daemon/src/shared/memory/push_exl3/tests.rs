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
    for world in [2, 4, 6] {
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
            if world == 2 {
                let mut payload = 0u64;
                for (region, &layer) in layers.iter().enumerate() {
                    let range = plan.layout.region(region)?;
                    let bytes = checked.plan.entries.iter().filter(|entry|
                        entry.dst_base >= range.start as u64 && entry.dst_base < range.end as u64)
                        .map(|entry| checked.plan.patterns[entry.pattern as usize].check()
                            .map(|extent| extent.bytes).map_err(anyhow::Error::msg))
                        .collect::<Result<Vec<_>>>()?.into_iter().sum::<u64>();
                    payload += bytes;
                    eprintln!("EXL3 TP{world} rank {rank} layer {layer}: device={}..{}, payload={bytes}, cumulative_payload={payload}",
                        range.start, range.end);
                }
            }
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
        let mut assignments = Vec::new();
        let mut offers = Vec::new();
        let mut serial = Builder::new(directory.path(), vec![u64::MAX; world]);
        for rank in 0..world {
            let plan = RankPlan::new(&catalog, &[0], world, rank, usize::MAX)?;
            let mounted = catalog.exl3().unwrap()
                .balanced_residency(V41Exl3Layer::Backbone(0), world, rank)?;
            assert_eq!(serde_json::to_value(&mounted)?, serde_json::to_value(&plan.residencies[0])?);
            assert_eq!(mounted.unit_map.as_ref().map(|m| m.fingerprint()), plan.placement[0].slice.unit_map);
            let checked = plan.checked(&catalog, directory.path(), world, rank)?;
            assert!(!checked.plan.entries.is_empty());
            assert_eq!(checked.arenas[rank], plan.layout.logical_bytes() as u64);
            assignments.push(cuteafd_loader::placement::SparkAssignment { rank: rank as u32,
                resident_bytes: plan.layout.bytes() as u64,
                slices: plan.placement.iter().map(|p| p.slice.clone()).collect(), io: Default::default() });
            offers.push(Offer { rank, host: format!("rank-{rank}"),
                arena_bytes: plan.layout.logical_bytes() as u64, placement: plan.placement.clone() });
            plan.append(&mut serial, &catalog, rank as u32)?;
        }
        let mut parallel = Builder::new(directory.path(), vec![u64::MAX; world]);
        for (rank, plan) in rank_plans(&catalog, &[0], &assignments, &offers)?.iter().enumerate() {
            assert_eq!(plan.placement, offers[rank].placement);
            plan.append(&mut parallel, &catalog, rank as u32)?;
        }
        assert_eq!(serde_json::to_vec(&serial.finish()?.plan)?, serde_json::to_vec(&parallel.finish()?.plan)?);
        offers[world - 1].placement[0].fingerprint.push('x');
        assert!(rank_plans(&catalog, &[0], &assignments, &offers).is_err());
    }
    Ok(())
}

#[test]
#[ignore = "requires an installed EXL3 checkpoint selected by CUTEAFD_PUSH_EXL3"]
fn installed_exl3_tp6_prepare_serial_parallel_equivalence_and_timing() -> Result<()> {
    let snapshot = std::path::PathBuf::from(std::env::var_os("CUTEAFD_PUSH_EXL3")
        .expect("set CUTEAFD_PUSH_EXL3 to the installed snapshot"));
    let opened = std::time::Instant::now();
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    eprintln!("catalog_ms={}", opened.elapsed().as_millis());
    let shape = catalog.routed_experts();
    let layers: Vec<_> = (shape.first_layer..shape.layers).collect();
    let root = cuteafd_loader::push_load::namespace(&snapshot).map(|(root, _)| root)
        .unwrap_or_else(|_| snapshot.clone());
    let mut assignments = Vec::new();
    let mut offers = Vec::new();
    let started = std::time::Instant::now();
    let serial = (0..6).map(|rank| RankPlan::new(&catalog, &layers, 6, rank, usize::MAX))
        .collect::<Result<Vec<_>>>()?;
    let serial_rank_ms = started.elapsed().as_millis();
    for (rank, plan) in serial.iter().enumerate() {
        assignments.push(cuteafd_loader::placement::SparkAssignment { rank: rank as u32,
            resident_bytes: plan.layout.bytes() as u64,
            slices: plan.placement.iter().map(|p| p.slice.clone()).collect(), io: Default::default() });
        offers.push(Offer { rank, host: format!("rank-{rank}"),
            arena_bytes: plan.layout.logical_bytes() as u64, placement: plan.placement.clone() });
    }
    let mut expected = None;
    for (name, plans, rank_ms) in [("serial", serial, serial_rank_ms), ("parallel", {
        let started = std::time::Instant::now();
        let plans = rank_plans(&catalog, &layers, &assignments, &offers)?;
        eprintln!("parallel_rank_ms={}", started.elapsed().as_millis());
        plans
    }, 0)] {
        let started = std::time::Instant::now();
        let mut builder = Builder::new(&root, offers.iter().map(|o| o.arena_bytes).collect());
        for (rank, plan) in plans.iter().enumerate() { plan.append(&mut builder, &catalog, rank as u32)?; }
        let appended = started.elapsed();
        let checked = builder.finish()?;
        let check_ms = (started.elapsed() - appended).as_millis();
        let seal = format!("{:x}", Sha256::digest(serde_json::to_vec(&checked.plan)?));
        eprintln!("{name}: rank_ms={rank_ms}, append_ms={}, check_ms={check_ms}, entries={}, seal={seal}",
            appended.as_millis(), checked.plan.entries.len());
        if let Some(expected) = &expected { assert_eq!(&seal, expected); } else { expected = Some(seal); }
    }
    Ok(())
}

/// Exercise each production group builder with and without a coordinator suffix.
pub(crate) fn coordinator_suffix(
    root: &Path, catalog: &OfficialV41Catalog,
    plans: &[(u64, Vec<Placement>, CheckedPlan)],
    build: impl Fn(&[cuteafd_loader::placement::SparkAssignment], &[Offer], Option<&CoordinatorRanks>) -> Result<CheckedPlan>,
) -> Result<()> {
    use cuteafd_loader::push_load::builder::Source;
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    let sparks = plans.len();
    let assignments = plans.iter().enumerate().map(|(rank, (resident, placement, _))|
        cuteafd_loader::placement::SparkAssignment { rank: rank as u32, resident_bytes: *resident,
            slices: placement.iter().map(|p| p.slice.clone()).collect(), io: Default::default() }).collect::<Vec<_>>();
    let mut offers = plans.iter().enumerate().map(|(rank, (_, placement, checked))| Offer {
        rank, host: format!("spark-{rank}"), arena_bytes: checked.arenas[rank], placement: placement.clone(),
    }).collect::<Vec<_>>();
    let plain = build(&assignments, &offers, None)?;
    // Compare the complete Spark group with independently checked rank plans.
    let records = |checked: &CheckedPlan, rank: usize| checked.plan.entries.iter()
        .filter(|e| e.rank as usize == rank).map(|e| serde_json::to_vec(&(
            &checked.plan.files[e.file as usize], &checked.plan.patterns[e.pattern as usize],
            e.src_base, e.dst_base, e.rank))).collect::<std::result::Result<Vec<_>, _>>();
    for (rank, (_, _, checked)) in plans.iter().enumerate() {
        assert_eq!(records(&plain, rank)?, records(checked, rank)?);
    }
    // Reuse a source header so installed-checkpoint proofs remain read-only.
    let source = root.join(&plain.plan.files[0]);
    let source_bytes = plain.file_sizes[0];
    assert!(source_bytes >= 16);
    for gpu in 0..2 {
        offers.push(Offer { rank: sparks + gpu, host: "coordinator".into(), arena_bytes: 16, placement: vec![] });
    }
    let suffix = offers[sparks..].to_vec();
    let snapshot = catalog.snapshot().to_path_buf();
    let calls = Arc::new(AtomicUsize::new(0));
    let appended = calls.clone();
    let coordinator = CoordinatorRanks { ranks: 2, append: Box::new(move |builder, seen_catalog, offers| {
        assert_eq!(seen_catalog.snapshot(), snapshot);
        assert_eq!(serde_json::to_vec(offers)?, serde_json::to_vec(&suffix)?);
        appended.fetch_add(1, Ordering::SeqCst);
        for offer in offers {
            builder.contiguous(&Source { path: source.clone(), offset: 0, bytes: source_bytes }, 0, 16,
                Destination { rank: offer.rank as u32, offset: 0, bytes: 16 })?;
        }
        Ok(())
    }) };
    let joined = build(&assignments, &offers, Some(&coordinator))?;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let n = plain.plan.entries.len();
    assert_eq!(&joined.plan.entries[..n], plain.plan.entries);
    assert_eq!(&joined.plan.patterns[..plain.plan.patterns.len()], plain.plan.patterns);
    assert_eq!(&joined.plan.files[..plain.plan.files.len()], plain.plan.files);
    assert_eq!(&joined.arenas[..sparks], plain.arenas);
    assert_eq!(joined.plan.entries[n..].iter().map(|e| e.rank).collect::<Vec<_>>(),
        [sparks as u32, sparks as u32 + 1]);
    assert!(build(&assignments, &offers, None).is_err(), "unclaimed suffix rejected");
    assert!(build(&assignments, &offers[..sparks + 1], Some(&coordinator)).is_err(), "missing suffix rejected");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "bad worlds fail before append");
    Ok(())
}
