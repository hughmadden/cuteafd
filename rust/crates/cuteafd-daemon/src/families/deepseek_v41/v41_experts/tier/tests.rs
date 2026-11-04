//! The tier's resident halves: the planner's ids per layer record on each
//! GPU at the planner's charge (header-only), and on two GPUs they push
//! byte-exact against the mount and address exactly the whole layer's experts.
use super::*;
use crate::shared::experts::{layer::{native_expert_steps, tp2_half_tests::{expert_fixture, expert_fixture_in,
    fake_planes, pattern_expert, planned_bytes, slot_slices}}, rtx::dense::Job};
use cuteafd_loader::{placement::{ExpertTierPlacement, ExpertTierSetting, ExpertTierShard}, V41ExpertSelection};

/// The tier `cuteafd plan --layout --gpu-expert-tier N` admits on two SM89
/// cards beside four Sparks.
fn planned_tier(snapshot: &std::path::Path, per_layer: usize) -> Result<ExpertTierPlacement> {
    let report = planned_bytes(snapshot, false, ExpertTierSetting::PerLayer(per_layer))?;
    let tier = report.solver_placement.as_ref().and_then(|placement| placement.expert_tier.clone())
        .context("the plan admitted no GPU expert tier")?;
    ensure!(tier.shard == ExpertTierShard::Tp2 && tier.experts == per_layer * 40, "unexpected tier {tier:?}");
    Ok(tier)
}

/// Each GPU's recorded halves of the planner's resident ids: slot `i` of a
/// layer reads only expert `ids[i]`'s half, every job lands in that slot's
/// planes, and the planes sum to the planner's resident bytes for the GPU.
#[test]
fn tier_halves_record_the_planned_ids_at_the_planned_bytes() -> Result<()> {
    let (_dir, catalog) = expert_fixture()?;
    let tier = planned_tier(catalog.snapshot(), 6)?;
    let half = catalog.routed_experts().intermediate / 2;
    for rank in 0..2usize {
        let mut resident = 0u64;
        for (layer, ids) in tier.resident.iter().enumerate().filter(|(_, ids)| !ids.is_empty()) {
            let (stride, planes) = fake_planes(rank as i32, (layer + 1) << 40, ids.len(), half)?;
            resident += planes.iter().map(|plane| plane.bytes as u64).sum::<u64>();
            for (slot, &id) in ids.iter().enumerate() {
                let slices = slot_slices(stride, planes, slot);
                let selection = V41ExpertSelection::BackboneTp2 { layer, expert: usize::from(id), rank };
                let prefix = format!("layers.{layer}.ffn.experts.{id}.");
                for step in native_expert_steps(&catalog, selection, rank as i32, slices)? {
                    ensure!(step.rects.iter().all(|rect| rect.name.starts_with(&prefix)),
                        "layer {layer} slot {slot} reads another expert than {id}");
                    for job in &step.jobs {
                        let dst = match *job {
                            Job::NativeWindow { dst, .. } | Job::ScaleCodes { dst, .. } => dst,
                            _ => anyhow::bail!("unexpected job in a resident expert recipe"),
                        };
                        ensure!(slices.iter().any(|slice| slice.ptr == dst.ptr && slice.bytes == dst.bytes),
                            "layer {layer} slot {slot} writes outside its planes");
                    }
                }
            }
        }
        ensure!(resident == tier.resident_bytes[rank],
            "GPU {rank}: recorded {resident} B, the planner charges {}", tier.resident_bytes[rank]);
        eprintln!("GPU {rank}: {} resident halves, {resident} B (planner {})", tier.experts, tier.resident_bytes[rank]);
    }
    Ok(())
}

/// The planner's tier (two experts in every layer) as TP2 halves on two
/// GPUs: recorded while loading and pushed through the coordinator push
/// protocol, every plane equals the mount load's; in the first and last
/// layers every slot equals the whole TP2 layer's expert it names; and each
/// GPU receives the planner's resident bytes.
#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB (V4.1 TP2 expert exports), two CUDA GPUs and CUTEAFD_PUSH_FIXTURE_DIR on NVMe"]
fn tier_halves_push_byte_exact_on_two_gpus() -> Result<()> {
    use crate::shared::{experts::rtx::{dense::Recorder, push::tests::{assert_bytes_equal, assert_planes_equal,
        mark_namespace, push_records_alone}}};
    let _ = tracing_subscriber::fmt().with_env_filter("info").with_test_writer().try_init();
    // SAFETY: the explicitly supplied architecture-matched library lives through the test.
    let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    let fixtures = std::path::PathBuf::from(std::env::var("CUTEAFD_PUSH_FIXTURE_DIR")?);
    let (root, catalog) = expert_fixture_in(Some(&fixtures))?;
    let snapshot = root.path().canonicalize()?;
    mark_namespace(&snapshot)?;
    let tier = planned_tier(&snapshot, 2)?;
    // Pattern every resident expert and the next one: a slot that read a
    // neighbour's half would differ.
    for (layer, ids) in tier.resident.iter().enumerate() {
        for expert in ids.iter().flat_map(|&id| [usize::from(id), usize::from(id) + 1]).filter(|&e| e < 384) {
            pattern_expert(&catalog, V41ExpertSelection::BackboneFull { layer, expert }, layer * 384 + expert)?;
        }
    }
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let budgets = [40usize << 30; 2];
    let mounted = load_halves(devices, &catalog, &tier.resident, budgets)?;
    let recorder = Recorder::start(true)?.context("dense recorder")?;
    let pushed = load_halves(devices, &catalog, &tier.resident, budgets);
    let records = recorder.finish();
    let pushed = pushed?;
    let sockets = tempfile::tempdir_in(&fixtures)?;
    push_records_alone(&snapshot, sockets.path(), devices, records)?;
    let charged = resident_bytes(&library, &catalog, &tier.resident)?;
    for rank in 0..2 {
        let bytes = |halves: &RankHalves<'_>| halves.iter().map(|(_, half)| half.budget().resident_bytes).sum::<usize>();
        ensure!(bytes(&pushed[rank]) as u64 == tier.resident_bytes[rank] && bytes(&mounted[rank]) == charged[rank]
            && charged[rank] as u64 == tier.resident_bytes[rank],
            "GPU {rank}: pushed {} / mounted {} / runtime {} B against the planner's {}",
            bytes(&pushed[rank]), bytes(&mounted[rank]), charged[rank], tier.resident_bytes[rank]);
        ensure!(pushed[rank].iter().map(|(layer, half)| (*layer, half.resident_ids().map(<[u16]>::to_vec)))
            .eq(tier.resident.iter().enumerate().filter(|(_, ids)| !ids.is_empty())
                .map(|(layer, ids)| (layer, Some(ids.clone())))), "GPU {rank}: resident ids differ from the plan");
        assert_planes_equal(&library, pushed[rank].iter().map(|(_, half)| half).zip(mounted[rank].iter().map(|(_, half)| half)),
            &format!("GPU {rank} tier layer"))?;
        eprintln!("GPU {rank}: {} B of tier halves pushed byte-exact against the mount (planner {})",
            bytes(&pushed[rank]), tier.resident_bytes[rank]);
    }
    drop(mounted);
    for layer in [0, 39] {
        for rank in 0..2 {
            let whole = devices[rank].run(|| ExpertWeights::load(&library, &catalog,
                ExpertLayer::BackboneTp2 { layer, rank }, 40 << 30))?;
            let (_, half) = pushed[rank].iter().find(|(at, _)| *at == layer).context("tier layer absent")?;
            for (slot, &id) in half.resident_ids().context("resident ids")?.iter().enumerate() {
                for plane in 0..4 {
                    let (a, w) = (half.buffers[plane].buffer, whole.buffers[plane].buffer);
                    let stride = a.bytes / half.resident_ids().map_or(1, <[u16]>::len);
                    ensure!(w.bytes == stride * 384, "layer {layer} plane {plane}: whole extent differs");
                    // SAFETY (offsets): slot `slot` and expert `id` lie inside their planes.
                    let at = |b: cuteafd_ffi::CuteafdDeviceBuffer, index: usize| cuteafd_ffi::CuteafdDeviceBuffer {
                        ptr: unsafe { b.ptr.cast::<u8>().add(index * stride) }.cast(), bytes: stride, ..b };
                    assert_bytes_equal(&library, at(a, slot), at(w, usize::from(id)),
                        &format!("GPU {rank} layer {layer} slot {slot} (expert {id}) plane {plane}"))?;
                }
            }
        }
        eprintln!("layer {layer}: every resident slot equals its expert in the whole TP2 layer on both GPUs");
    }
    Ok(())
}

/// The host masks and the `slot_of` tables are one view of the set: a slot
/// exists exactly for a resident expert, and slot `i` names `ids[i]`.
#[test]
fn residency_masks_and_slot_tables_name_the_same_experts() -> Result<()> {
    let residency = TierResidency::new(384, vec![vec![], vec![3, 9, 383], vec![0]])?;
    assert_eq!((residency.layers(), residency.total()), (3, 4));
    assert_eq!(residency.resident_layers(), vec![1, 2]);
    for layer in 0..3 {
        let table = residency.slot_of(layer);
        assert_eq!(table.len(), 384);
        for (expert, &slot) in table.iter().enumerate() {
            assert_eq!(slot != NOT_RESIDENT, residency.contains(layer, expert), "layer {layer} expert {expert}");
            if slot != NOT_RESIDENT {
                assert_eq!(usize::from(residency.ids(layer)[slot as usize]), expert);
            }
        }
        assert_eq!(residency.mask(layer).unwrap().iter().collect::<Vec<_>>(),
            residency.ids(layer).iter().map(|&id| usize::from(id)).collect::<Vec<_>>());
    }
    let tables = residency.slot_tables();
    assert_eq!(tables.len(), 3 * 384);
    assert_eq!(&tables[384..768], residency.slot_of(1).as_slice());
    // Past the described layers nothing is resident.
    assert!(residency.ids(7).is_empty() && !residency.contains(7, 0) && residency.mask(7).is_none());
    for bad in [vec![vec![5u16, 3]], vec![vec![3, 3]], vec![vec![384]]] {
        assert!(TierResidency::new(384, bad).is_err());
    }
    Ok(())
}

/// Seeded inputs of one expert wave: FP8-K32 wire rows (finite E4M3 values,
/// UE8M0 scales 2^-7..2^0), six distinct router expert ids per row and
/// positive gate weights.
fn wave_inputs(seed: u64, rows: usize) -> (Vec<u8>, Vec<u32>, Vec<f32>) {
    let mut state = seed | 1;
    let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
    let (hidden, topk) = (5120usize, 6usize);
    let mut wire = vec![0u8; rows * (hidden + hidden / 32)];
    for row in wire.chunks_exact_mut(hidden + hidden / 32) {
        for value in &mut row[..hidden] {
            let byte = (next() >> 32) as u8;
            *value = if byte & 0x7f == 0x7f { byte ^ 1 } else { byte };
        }
        for scale in &mut row[hidden..] { *scale = 120 + (next() >> 40) as u8 % 8; }
    }
    let mut ids = Vec::with_capacity(rows * topk);
    for _ in 0..rows {
        let mut chosen = Vec::with_capacity(topk);
        while chosen.len() < topk {
            let id = (next() % 384) as u32;
            if !chosen.contains(&id) { chosen.push(id); }
        }
        ids.extend(chosen);
    }
    let weights = (0..rows * topk).map(|_| 0.05 + (next() >> 40) as f32 / (1u64 << 24) as f32).collect();
    (wire, ids, weights)
}

/// One GPU's tier half against the whole TP2 layer half on the same GPU: the
/// router's ids through the device slot kernel over the compact resident
/// planes give rank partials byte-identical to the whole layer's with every
/// non-resident route dropped (id -1), at decode, verify and prefill widths,
/// for both halves and both output policies, so the GPU result the tier adds
/// to the reduce is exactly the resident experts' share of the layer.
/// Synthetic coded-layout weights; one GPU (each half in turn).
#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB (V4.1 TP2 expert exports for this GPU) and one CUDA GPU"]
fn tier_half_equals_the_whole_layer_half_with_other_routes_dropped() -> Result<()> {
    use crate::shared::experts::rtx::{native::{NativeOutputPolicy, NativeTp2}, ExpertInput, Routes, RtxExpertLayer,
        RtxSelectionPolicy};
    use crate::shared::memory::device::{Allocation, Stream};
    use std::rc::Rc;
    // SAFETY: the operator-selected native library outlives every owner below.
    let lib = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    lib.cuda_set_device(0)?;
    let device = Device { library: &lib, id: 0 };
    let slot_kernel = lib.v41_tier_slots()?;
    let (layer, capacity) = (11usize, 256usize);
    // The planner's uniform spread plus a contiguous block: 64 residents.
    let resident: Vec<u16> = (0..384u16).filter(|e| e % 7 == 3 || (100..110).contains(e)).collect();
    let mut layers = vec![Vec::new(); 40];
    layers[layer] = resident.clone();
    let residency = TierResidency::new(384, layers)?;
    let shape = cuteafd_ffi::TierSlotShape { layers: residency.layers(), experts: residency.experts() };
    let tables = Allocation::new(device, shape.bytes())?;
    lib.copy_h2d(tables.buffer, &residency.slot_tables().iter().flat_map(|slot| slot.to_ne_bytes()).collect::<Vec<_>>())?;
    let ids_bytes = capacity * 6 * 4;
    let [wire, ids, masked, slots, routing] = [capacity * 5280, ids_bytes, ids_bytes, ids_bytes, ids_bytes]
        .map(|bytes| Allocation::new(device, bytes));
    let (wire, ids, masked, slots, routing) = (wire?, ids?, masked?, slots?, routing?);
    let stream = Stream::new(device)?;
    let host = |buffer: cuteafd_ffi::CuteafdDeviceBuffer, bytes: usize| -> Result<Vec<u8>> {
        let mut data = vec![0u8; bytes];
        lib.copy_d2h(&mut data, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..buffer })?;
        Ok(data)
    };
    let words = |values: &[u32]| values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
    let mut compared = 0;
    for rank in 0..2u8 {
        let selection = ExpertLayer::BackboneTp2 { layer, rank: usize::from(rank) };
        let whole = Rc::new(device.own(|| Ok(vec![ExpertWeights::synthetic_coded(&lib, selection, 0x71e2 + u64::from(rank))?]))?);
        let ids_usize: Vec<usize> = resident.iter().map(|&id| usize::from(id)).collect();
        let compact = Rc::new(device.own(|| Ok(vec![whole[0].resident_copy(&lib, &ids_usize)?]))?);
        ensure!(compact[0].experts() == resident.len() && compact[0].resident_ids() == Some(resident.as_slice()));
        for policy in [NativeOutputPolicy::TokenSums, NativeOutputPolicy::KernelPlanes] {
            let workspace = NativeTp2::workspace_bytes_for_format(&lib, capacity, RtxSelectionPolicy::CanonicalDecode,
                policy, false)?;
            let mut full = NativeTp2::execution(device, rank, vec![layer], capacity, workspace, whole.clone(),
                RtxSelectionPolicy::CanonicalDecode, policy)?;
            let mut tier = NativeTp2::resident_weights(device, rank, compact.clone(), capacity,
                RtxSelectionPolicy::CanonicalDecode, policy)?.execution()?;
            for (case, rows) in [1usize, 6, 16, 40, 80, 97, 256].into_iter().enumerate() {
                let (wire_host, ids_host, weights_host) = wave_inputs(0x51ab + case as u64 * 31 + u64::from(rank), rows);
                // The whole layer's reference drops every non-resident route.
                let dropped: Vec<u32> = ids_host.iter().map(|&id| if residency.contains(layer, id as usize) { id }
                    else { u32::MAX }).collect();
                lib.copy_h2d(wire.buffer, &wire_host)?;
                lib.copy_h2d(ids.buffer, &words(&ids_host))?;
                lib.copy_h2d(masked.buffer, &words(&dropped))?;
                lib.copy_h2d(routing.buffer, &weights_host.iter().flat_map(|w| w.to_ne_bytes()).collect::<Vec<_>>())?;
                let columns = if policy == NativeOutputPolicy::KernelPlanes { 6 } else { 1 };
                let bytes = rows * 5120 * 4 * columns;
                // SAFETY: the fixture owns every input and output and drains the stream before reading.
                let (reference, actual) = unsafe {
                    full.enqueue(layer, rows, ExpertInput::Fp8K32(wire.buffer.ptr),
                        Routes { ids: masked.buffer.ptr, weights: routing.buffer.ptr }, stream.raw, &mut || {})?;
                    stream.drain()?;
                    let reference = host(cuteafd_ffi::CuteafdDeviceBuffer { ptr: full.output(), ..wire.buffer }, bytes)?;
                    slot_kernel.launch(ids.buffer, tables.buffer, slots.buffer, shape, layer, rows * 6, stream.raw)?;
                    tier.enqueue(layer, rows, ExpertInput::Fp8K32(wire.buffer.ptr),
                        Routes { ids: slots.buffer.ptr, weights: routing.buffer.ptr }, stream.raw, &mut || {})?;
                    stream.drain()?;
                    (reference, host(cuteafd_ffi::CuteafdDeviceBuffer { ptr: tier.output(), ..wire.buffer }, bytes)?)
                };
                let floats = |raw: &[u8]| raw.as_chunks::<4>().0.iter().map(|word| f32::from_ne_bytes(*word)).collect::<Vec<_>>();
                let values = floats(&actual);
                ensure!(values.iter().all(|value| value.is_finite()), "rank {rank} {policy:?} rows {rows}: non-finite partials");
                let resident_routes = dropped.iter().filter(|&&id| id != u32::MAX).count();
                ensure!((resident_routes == 0) == values.iter().all(|&value| value == 0.0),
                    "rank {rank} {policy:?} rows {rows}: {resident_routes} resident routes but partials {}",
                    if resident_routes == 0 { "nonzero" } else { "all zero" });
                if reference != actual {
                    let differing = floats(&reference).iter().zip(&values).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    anyhow::bail!("rank {rank} {policy:?} rows {rows}: {differing} of {} partials differ from the whole layer",
                        values.len());
                }
                eprintln!("tier half rank {rank} {policy:?} rows {rows}: {resident_routes} of {} routes resident, \
                    byte-identical to the whole layer", rows * 6);
                compared += 1;
            }
        }
    }
    ensure!(compared == 2 * 2 * 7);
    Ok(())
}
