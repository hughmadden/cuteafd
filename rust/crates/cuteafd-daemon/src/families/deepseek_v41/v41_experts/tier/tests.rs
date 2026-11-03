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
