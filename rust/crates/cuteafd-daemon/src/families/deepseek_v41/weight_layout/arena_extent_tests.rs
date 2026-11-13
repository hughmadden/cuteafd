//! CPU-only admission/consumer contract, using the official synthetic headers.
use super::*;
use crate::families::deepseek_v41::v41_tensors::allocation_bytes;
use cuteafd_core::GpuArch;
use cuteafd_loader::placement::families::deepseek_v41::weight_layout as loader;
use cuteafd_loader::serving_capacity::WeightArenaLayout;

fn check_views(layout: &WeightArenaLayout, arch: GpuArch,
    expected: impl Fn(&str) -> Result<usize>) -> Result<usize> {
    for (name, view) in layout.views() {
        let bytes = if let Some(weight) = name.strip_suffix(".packed_scales") {
            // Independent of the loader's packed_scale_bytes implementation.
            let divisor = if arch == GpuArch::Sm89 { 1024 } else { 32 };
            expected(weight)? / divisor
        } else { expected(name)? };
        assert_eq!(view.bytes(), bytes, "{arch} {}:{name}", layout.id());
        assert_eq!(allocation_bytes(Some(layout), name, bytes)?, view.bytes(),
            "daemon request {arch} {}:{name}", layout.id());
        // An old standalone size cannot override the named admitted extent.
        assert_eq!(allocation_bytes(Some(layout), name, 1048576)?, view.bytes());
        assert_eq!(allocation_bytes(None, name, bytes)?, bytes);
    }
    assert!(allocation_bytes(Some(layout), "missing.weight", 1048576).is_err());
    Ok(layout.views().len())
}

#[test]
fn v41_weight_arena_views_match_loader_extents() -> Result<()> {
    let dir = tempfile::tempdir()?;
    cuteafd_loader::plan::testing::write_v41_snapshot(dir.path());
    let catalog = cuteafd_loader::read_official_v41_role_catalog(dir.path(), "arena-extents",
        |name| !name.contains(".ffn.experts."))?;
    let headers = || catalog.tensors().iter().map(|tensor| &tensor.metadata);
    let header_bytes = |name: &str| -> Result<usize> {
        Ok(usize::try_from(catalog.tensor(name)?.metadata.byte_length)?)
    };
    let vision = loader::vision_layout_from_headers(headers())?;
    let mut count = 0;
    for arch in [GpuArch::Sm89, GpuArch::Sm120] {
        count += check_views(&vision, arch, header_bytes)?;
        let draft = loader::draft_layout_from_headers_for_arch(headers(), arch)?;
        count += check_views(&draft, arch, header_bytes)?;
        for stage in 0..3 {
            let name = format!("mtp.{stage}.attn.wo_a.weight.packed_scales");
            let bytes = draft.view(&name).unwrap().bytes();
            assert_eq!(bytes, if arch == GpuArch::Sm89 { 32768 } else { 1048576 });
            if arch == GpuArch::Sm89 {
                assert_ne!(bytes, 1048576, "the first-light request used the SM120 extent");
            }
        }
        for placement in [None, Some(CachePlacement::encoder_decoder())] {
            for embedding in [false, true] {
                for head in [Fp8Head::Off, Fp8Head::Draft, Fp8Head::All, Fp8Head::Packed] {
                    for (query, output) in [(false, false), (true, false), (false, true), (true, true)] {
                        if placement.is_none() && (query || output) { continue; }
                        let target = target_layout_for_arch(&catalog, placement, embedding, head,
                            query, output, arch)?;
                        let ranks = target.ranks.len();
                        for layout in &target.ranks {
                            count += check_views(layout, arch, |name| {
                                let rows = 129280 / ranks;
                                Ok(match name {
                                    "head.bf16" => rows * 5120 * 2,
                                    "head.fp8.payload" => rows * 5120,
                                    "head.fp8.scales" => rows * 160,
                                    "head.packed12" => cuteafd_core::packed_head::PackedHeadLayout::new(rows, 5120)?.total_bytes(),
                                    _ => {
                                        let shard = ranks == 2 && (name.contains(".ffn.shared_experts.")
                                            || (query && name.ends_with(".attn.wq_b.weight"))
                                            || (output && name.ends_with(".attn.wo_b.weight")));
                                        header_bytes(name)? / if shard { 2 } else { 1 }
                                    }
                                })
                            })?;
                        }
                    }
                }
            }
        }
        count += expert_views(arch, &catalog)?;
        count += nvfp4_views(arch, &catalog)?;
    }
    // The native library is deliberately not loaded in this test. Pin the
    // first-light call's library-info binding as well as the CPU choke point,
    // so restoring its old literal fails this guard even with arena authority.
    let source = include_str!("../v41_experts/dspark/attention_output.rs");
    let pack = source.split("pub(super) fn pack_grouped_scales").nth(1).unwrap()
        .split("impl DsparkAttentionOutput").next().unwrap();
    assert!(pack.contains("grouped.info().packed_weight_scale_bytes"));
    assert!(!pack.contains("1048576"));
    eprintln!("V4.1 arena extent guard checked {count} views across SM89 and SM120");
    Ok(())
}

/// Native routed and tier planes share their exact library layout. Exercise
/// whole/TP2 backbone, whole/TP2 dSpark and compact aggregate tier inventories
/// using the existing header-only plane fixture and production subset sizing.
fn expert_views(arch: GpuArch, catalog: &OfficialV41Catalog) -> Result<usize> {
    use crate::shared::experts::{layer::{ExpertLayer, ExpertLoadBudget, resident_subset,
        tp2_half_tests::fake_planes}, weight_layout};
    let shape = catalog.routed_experts();
    assert_eq!(shape.hidden, 5120, "native plane fixture geometry");
    let mut count = 0;
    for rank in 0..2 {
        for (half, tier) in [(false, false), (true, false), (true, true)] {
            let mut expected = std::collections::BTreeMap::new();
            for layer in shape.first_layer..shape.layers {
                let kind = if half { ExpertLayer::BackboneTp2 { layer, rank } }
                    else { ExpertLayer::BackboneFull { layer } };
                let (_, planes) = fake_planes(rank as i32, 1 << 40, shape.experts, shape.intermediate / if half { 2 } else { 1 })?;
                let whole = planes.map(|plane| plane.bytes);
                let sizes = if tier {
                    resident_subset(&whole, ExpertLoadBudget { resident_bytes: whole.iter().sum(),
                        device_staging_bytes: 0, pinned_host_bytes: 0, read_scratch_bytes: 0 },
                        shape.experts, &[0, 3, shape.experts - 1])?.0
                } else { whole };
                for (plane, bytes) in sizes.into_iter().enumerate() {
                    expected.insert(weight_layout::name(kind, plane), bytes);
                }
            }
            if !tier {
                for stage in 0..shape.draft_stages {
                    let kind = if half { ExpertLayer::DsparkTp2 { stage, rank } }
                        else { ExpertLayer::Dspark { stage } };
                    let (_, planes) = fake_planes(rank as i32, 1 << 40, shape.draft_experts, shape.intermediate / if half { 2 } else { 1 })?;
                    for (plane, bytes) in planes.map(|plane| plane.bytes).into_iter().enumerate() {
                        expected.insert(weight_layout::name(kind, plane), bytes);
                    }
                }
            }
            let layout = weight_layout::layout(rank as i32, "extent-fixture", expected.clone())?;
            count += check_views(&layout, arch, |name| Ok(expected[name]))?;
        }
    }
    Ok(count)
}

/// The optional NVFP4 backbone has four native planes and four FP32 vectors.
/// Its producer and consumer use the same runtime-selected stored width.
fn nvfp4_views(arch: GpuArch, catalog: &OfficialV41Catalog) -> Result<usize> {
    use crate::shared::experts::{layer::{ExpertLayer, nvfp4}, weight_layout};
    let shape = catalog.routed_experts();
    let mut count = 0;
    for half in [false, true] {
        let stored = (shape.intermediate / if half { 2 } else { 1 }).div_ceil(128) * 128;
        let admitted = cuteafd_loader::placement::families::deepseek_v41::resident_planes(
            shape.hidden as u64, stored as u64, true)?.map(|bytes| bytes as usize * shape.experts);
        let requested = nvfp4::plane_sizes(stored).map(|bytes| bytes * shape.experts);
        assert_eq!(requested, admitted);
        for rank in 0..if half { 2 } else { 1 } {
            let mut expected = std::collections::BTreeMap::new();
            for layer in shape.first_layer..shape.layers {
                let kind = if half { ExpertLayer::BackboneTp2 { layer, rank } }
                    else { ExpertLayer::BackboneFull { layer } };
                for (plane, bytes) in admitted.into_iter().chain([shape.experts * 4; 4]).enumerate() {
                    expected.insert(weight_layout::name(kind, plane), bytes);
                }
            }
            let layout = weight_layout::layout(rank as i32, "nvfp4-extent-fixture", expected.clone())?;
            count += check_views(&layout, arch, |name| Ok(expected[name]))?;
        }
    }
    Ok(count)
}
