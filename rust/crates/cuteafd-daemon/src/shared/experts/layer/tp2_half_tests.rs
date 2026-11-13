//! Recorded TP2 halves of native experts (header-only, no GPU): each rank's
//! sources are exactly what the mount stages for it, the two ranks partition
//! every tensor, and a GPU's recorded planes are what the planner charges it.
use super::{native_expert_rects, native_expert_steps};
use crate::shared::experts::rtx::dense::{HostStaging, Job, TRANSFORM_BYTES};
use anyhow::{ensure, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::{OfficialV41Catalog, V41ExpertSelection};
use std::os::unix::fs::FileExt;

/// The planner's official V4.1 snapshot of holes (with its synthetic
/// workspace manifests) under `parent`, catalogued for its routed experts.
pub(crate) fn expert_fixture_in(parent: Option<&std::path::Path>) -> Result<(tempfile::TempDir, OfficialV41Catalog)> {
    let dir = match parent { Some(parent) => tempfile::tempdir_in(parent)?, None => tempfile::tempdir()? };
    let snapshot = dir.path().canonicalize()?;
    cuteafd_loader::plan::testing::write_v41_snapshot(&snapshot);
    let programs = cuteafd_loader::plan::testing::write_v41_workspaces(&snapshot);
    // The selected workspace manifest must carry its program inventory even
    // when the fixture needs no optional compressed tier-reply capability.
    std::fs::write(programs, serde_json::to_vec(&serde_json::json!({"programs": []}))?)?;
    let catalog = cuteafd_loader::read_official_v41_role_catalog(&snapshot, "experts",
        |name| name.contains(".ffn.experts."))?;
    ensure!(catalog.mxfp4_scale_codes(), "the official fixture uses coded scales");
    Ok((dir, catalog))
}

pub(crate) fn expert_fixture() -> Result<(tempfile::TempDir, OfficialV41Catalog)> { expert_fixture_in(None) }

/// `group`'s bytes on `gpu` in a two-RTX SM89 V4.1 plan of `snapshot` (four
/// Sparks), as `cuteafd plan --layout` charges them.
pub(crate) fn planned_bytes(snapshot: &std::path::Path, tp2_dspark_experts: bool,
    tier: cuteafd_loader::placement::ExpertTierSetting) -> Result<cuteafd_loader::plan::PlanReport> {
    use cuteafd_loader::plan::{layout::LayoutOptions, ExpertPlacement, PlanOptions};
    let card = cuteafd_loader::placement::ArchContext::reference_total_bytes(cuteafd_core::GpuArch::Sm89);
    let options = PlanOptions { layout: Some(LayoutOptions { host_memory: Some((256 << 30, 256 << 30)),
        rtx_bytes: vec![card; 2], coordinator_arch: cuteafd_core::GpuArch::Sm89,
        workspace_manifest: Some(snapshot.join("share/PROGRAMS.json")), pool_tokens: Some(0),
        context_tokens: 1 << 20, expert_tier: tier, tp2_dspark_experts, ..Default::default() }),
        placement: ExpertPlacement::from_spark_ranks(4), mapped_scale_host_budget_bytes: Some(64 << 30),
        ..PlanOptions::default() };
    let report = cuteafd_loader::plan::testing::BudgetedPlanInput::open(snapshot)?.plan_spark_count(&options, 6, Some(4))?;
    ensure!(report.executable(), "{}", cuteafd_loader::plan::render(&report));
    Ok(report)
}

pub(crate) fn group_bytes(report: &cuteafd_loader::plan::PlanReport, gpu: usize, group: &str) -> u64 {
    report.memory_layout.as_ref().map_or(0, |layout| layout.devices[gpu].items.iter()
        .filter(|item| item.group == group).map(|item| item.bytes).sum())
}

/// Writes a distinct byte pattern over one expert's six source tensors.
pub(crate) fn pattern_expert(catalog: &OfficialV41Catalog, selection: V41ExpertSelection, seed: usize) -> Result<()> {
    let staging = catalog.expert_staging(selection)?;
    for (slot, name) in staging.tensor_names().iter().enumerate() {
        let tensor = catalog.tensor(name)?;
        let bytes = (0..tensor.metadata.byte_length as usize).map(|i| if slot >= 3 {
            // UE8M0 codes 2^-3..2^0: any span a coded row accepts.
            124 + ((i * 7 + slot + seed) % 4) as u8
        } else { ((i * 13 + i / 4093 + slot * 31 + seed * 17) % 251) as u8 }).collect::<Vec<_>>();
        std::fs::OpenOptions::new().write(true).open(catalog.snapshot().join(&tensor.shard))?
            .write_all_at(&bytes, tensor.metadata.byte_offset)?;
    }
    Ok(())
}

#[test]
fn tp2_expert_halves_record_what_the_mount_stages_and_partition_every_tensor() -> Result<()> {
    let (_dir, catalog) = expert_fixture()?;
    let cases = [
        (V41ExpertSelection::Dspark { stage: 2, expert: 127 },
            [0, 1].map(|rank| V41ExpertSelection::DsparkTp2 { stage: 2, expert: 127, rank })),
        (V41ExpertSelection::BackboneFull { layer: 39, expert: 383 },
            [0, 1].map(|rank| V41ExpertSelection::BackboneTp2 { layer: 39, expert: 383, rank })),
        (V41ExpertSelection::BackboneFull { layer: 0, expert: 5 },
            [0, 1].map(|rank| V41ExpertSelection::BackboneTp2 { layer: 0, expert: 5, rank })),
    ];
    for (seed, (whole, halves)) in cases.into_iter().enumerate() {
        pattern_expert(&catalog, whole, seed)?;
        let whole_rects = native_expert_rects(&catalog, whole)?;
        let mut covered = whole_rects.iter().map(|r| vec![0u8; r.bytes()]).collect::<Vec<_>>();
        for selection in [whole].into_iter().chain(halves) {
            let staging = catalog.expert_staging(selection)?;
            let mut staged = vec![0u8; staging.staging_bytes()];
            let mut scratch = vec![0u8; staging.minimum_read_scratch_bytes()];
            staging.read_into(&mut staged, &mut scratch)?;
            for (slot, rect) in native_expert_rects(&catalog, selection)?.iter().enumerate() {
                let mut bytes = vec![0u8; rect.bytes()];
                rect.read(&mut bytes)?;
                assert_eq!(bytes, staged[staging.tensor_ranges()[slot].clone()], "{selection:?} slot {slot}");
                if selection == whole { continue; }
                // Mark every source byte this half takes from the whole tensor.
                let whole = &whole_rects[slot];
                assert_eq!((&rect.path, rect.base, rect.row_bytes), (&whole.path, whole.base, whole.row_bytes));
                for row in rect.rows.clone() {
                    for byte in &mut covered[slot][row * whole.row_bytes + rect.cols.start..row * whole.row_bytes + rect.cols.end] {
                        *byte += 1;
                    }
                }
            }
        }
        for (slot, marks) in covered.iter().enumerate() {
            assert!(marks.iter().all(|&n| n == 1), "{whole:?} slot {slot}: TP2 halves do not partition the tensor");
        }
    }
    // Spark shards (TP4 and generic TP) are not recorded as coordinator experts.
    assert!(native_expert_rects(&catalog, V41ExpertSelection::Backbone { layer: 3, expert: 0, rank: 1 }).is_err());
    Ok(())
}

/// Fake device planes (addresses only; nothing dereferences them) for
/// `experts` slots of a coded `intermediate`-wide layer on `device`.
pub(crate) fn fake_planes(device: i32, base: usize, experts: usize, intermediate: usize)
    -> Result<([usize; 4], [CuteafdDeviceBuffer; 4])> {
    let stride = cuteafd_loader::formats::scale_codes::expert_planes(5120, intermediate,
        cuteafd_core::mxfp4_layouts::stored_width(5120, intermediate).unwrap()).unwrap();
    let mut at = base;
    let planes = stride.map(|bytes| {
        let buffer = CuteafdDeviceBuffer { ptr: at as *mut _, bytes: bytes * experts, device_id: device, flags: 0 };
        at += (bytes * experts).next_multiple_of(1 << 20);
        buffer
    });
    Ok((stride, planes))
}

/// `slot`'s slices of `planes` (as the loader carves them).
pub(crate) fn slot_slices(stride: [usize; 4], planes: [CuteafdDeviceBuffer; 4], slot: usize) -> [CuteafdDeviceBuffer; 4] {
    std::array::from_fn(|i| CuteafdDeviceBuffer { ptr: (planes[i].ptr as usize + slot * stride[i]) as *mut _,
        bytes: stride[i], ..planes[i] })
}

/// Every recorded step of a TP2 half lands on its own GPU inside its expert's
/// planes; the windows tile each projection once; each GPU stages exactly
/// half of every draft expert's bytes; and the planes a GPU receives sum to
/// the planner's `--tp2-dspark-experts` charge for it.
#[test]
fn dspark_tp2_recipes_land_each_gpu_on_the_charged_halves() -> Result<()> {
    let (_dir, catalog) = expert_fixture()?;
    let shape = catalog.routed_experts();
    let (stages, experts) = (shape.draft_stages, shape.draft_experts);
    assert_eq!((stages, experts, shape.hidden, shape.intermediate), (3, 128, 5120, 2304));
    let half = shape.intermediate / 2;
    // The planner's charges (cuteafd plan --layout): GPU0's half is its own
    // item; GPU1's routed-expert owner loses the other half when split.
    let snapshot = catalog.snapshot().to_path_buf();
    let whole_plan = planned_bytes(&snapshot, false, Default::default())?;
    let split_plan = planned_bytes(&snapshot, true, Default::default())?;
    let charge = group_bytes(&split_plan, 0, "dSpark TP2 expert half");
    let (whole_planes, _) = fake_planes(1, 0, 1, shape.intermediate)?;
    let whole_resident = (stages * experts * whole_planes.iter().sum::<usize>()) as u64;
    // Each independently coded half retains its own scale bases, so moving
    // GPU0's half out saves slightly less than that half's resident extent.
    assert_eq!(charge, 3_506_405_376, "the three-stage TP2 expert half charge");
    assert_eq!(whole_resident - charge, 3_504_439_296, "GPU1's original net weight reduction");
    assert_eq!(group_bytes(&whole_plan, 1, "dSpark experts"), whole_resident);
    assert_eq!(group_bytes(&split_plan, 1, "dSpark experts"), charge);
    assert_eq!(group_bytes(&whole_plan, 1, "dSpark experts") - group_bytes(&split_plan, 1, "dSpark experts"),
        whole_resident - charge, "GPU1 keeps one half of the drafter's experts");
    let whole_sources = (0..stages).map(|stage| -> Result<u64> {
        let rects = native_expert_rects(&catalog, V41ExpertSelection::Dspark { stage, expert: 0 })?;
        Ok(rects.iter().map(|r| r.bytes() as u64).sum::<u64>() * experts as u64)
    }).sum::<Result<u64>>()?;
    for rank in 0..2usize {
        let device = rank as i32;
        let (mut resident, mut sourced) = (0u64, 0u64);
        for stage in 0..stages {
            let (stride, planes) = fake_planes(device, (stage + 1) << 40, experts, half)?;
            resident += planes.iter().map(|p| p.bytes as u64).sum::<u64>();
            for expert in 0..experts {
                let slices = slot_slices(stride, planes, expert);
                let steps = native_expert_steps(&catalog, V41ExpertSelection::DsparkTp2 { stage, expert, rank },
                    device, slices)?;
                let mut rows = [vec![0u8; half], vec![0u8; half], vec![0u8; shape.hidden]];
                let mut scale_jobs = 0;
                for step in &steps {
                    assert_eq!(step.device, device);
                    assert!(step.bytes() <= TRANSFORM_BYTES);
                    sourced += step.bytes() as u64;
                    for job in &step.jobs {
                        match *job {
                            Job::NativeWindow { intermediate, projection, first, rows: count, dst } => {
                                assert_eq!(intermediate as usize, half);
                                assert_eq!(dst.ptr, slices[if projection == 2 { 2 } else { 0 }].ptr);
                                for row in &mut rows[projection as usize][first as usize..(first + count) as usize] { *row += 1; }
                            }
                            Job::ScaleCodes { dst, .. } => {
                                assert!(dst.ptr == slices[1].ptr || dst.ptr == slices[3].ptr);
                                assert!(dst.bytes == slices[1].bytes || dst.bytes == slices[3].bytes);
                                scale_jobs += 1;
                            }
                            _ => anyhow::bail!("unexpected job in a draft expert recipe"),
                        }
                    }
                }
                assert_eq!(scale_jobs, 3);
                assert!(rows.iter().all(|r| r.iter().all(|&n| n == 1)), "windows do not tile each projection once");
            }
        }
        assert_eq!(resident, charge, "GPU {rank} recorded planes differ from the planner's charge");
        assert_eq!(sourced * 2, whole_sources, "GPU {rank} must stage exactly half of the draft expert bytes");
        eprintln!("GPU {rank}: {resident} B of draft expert halves (planner {charge}), {sourced} source bytes");
    }
    // One GPU's whole draft half streams through bounded staging in file order.
    let mut steps = Vec::new();
    for stage in 0..stages { for expert in 0..experts {
        let (stride, planes) = fake_planes(0, 1 << 40, experts, half)?;
        steps.extend(native_expert_steps(&catalog, V41ExpertSelection::DsparkTp2 { stage, expert, rank: 0 }, 0,
            slot_slices(stride, planes, expert))?);
    } }
    let bound = HostStaging::for_steps(steps.iter().map(|s| s.rects.as_slice()), 1 << 20)?;
    assert!(bound.bytes()? + (128 << 20) <= 512 << 20, "TP2 draft staging {} B", bound.bytes()?);
    Ok(())
}
