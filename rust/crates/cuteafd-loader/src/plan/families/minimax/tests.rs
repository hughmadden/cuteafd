use super::*;
use crate::placement::inventory::PRO_TOTAL_BYTES;
use crate::plan::{layout, testing, ExpertPlacement, PlanOptions, Status};
use cuteafd_core::memory_layout::DeviceKind;

fn snapshot(vision: bool, draft: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    testing::write_minimax_m3_snapshot(dir.path(), vision, draft);
    dir
}

/// An RTX 5090's usable logical budget (`--coordinator-gpu-budget-gib 31.8`): refused by name.
const RTX_5090: u64 = 34_144_990_003;

fn with_layout(rtx: u64, ranks: usize) -> PlanOptions {
    PlanOptions { placement: ExpertPlacement::from_spark_ranks(ranks),
        layout: Some(layout::LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![rtx],
            ..Default::default() }), ..Default::default() }
}

#[test]
fn release_headers_satisfy_the_contract() {
    let dir = snapshot(true, true);
    let report = testing::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert_eq!(report.family.as_deref(), Some("minimax_m3"));
    assert_eq!(report.runtime, Some(RuntimeStatus::Planned));
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    assert!(report.unclassified.is_empty(), "{:?}", &report.unclassified[..report.unclassified.len().min(5)]);
    let spec = report.spec.as_ref().unwrap();
    assert_eq!(spec.moe_layers(), 57);
    assert!(spec.layers.iter().all(|l| matches!(l.attention,
        AttentionKind::BlockSparseGqa { heads: 64, kv_heads: 4, head_dim: 128, block: 128, topk_blocks: 16, index_dim: 128 })));
    assert!(matches!(&spec.speculator, Some(SpeculatorSpec::Dspark { stages: 5, target_layers, .. }) if target_layers == &[3, 17, 31, 45, 59]));
    for component in [Component::Attention, Component::Indexer, Component::Router, Component::RoutedExpert,
        Component::SharedExpert, Component::DenseFfn, Component::Embedding, Component::LmHead, Component::Vision] {
        let plan = report.components.iter().find(|c| c.component == component).unwrap_or_else(|| panic!("{component:?}"));
        assert_eq!((plan.status, plan.rejected), (Status::Planned, 0), "{component:?}: {:?}", plan.rejections);
    }
    let formats = |c| report.components.iter().find(|p| p.component == c).unwrap().formats.keys().cloned().collect::<Vec<_>>();
    assert_eq!(formats(Component::RoutedExpert), ["nvfp4ct-g16"]);
    assert_eq!(formats(Component::Attention), ["bf16", "fp8-block1x32/ue8m0"]);
    assert_eq!(report.experts.as_ref().unwrap().package, PACKAGE);
    assert!(!report.executable());
}

#[test]
fn compressed_nvfp4_is_distinct_from_modelopt() {
    let dir = snapshot(false, false);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let w1 = checkpoint.tensors.iter().filter(|t| t.meta.name.contains(".layers.3.block_sparse_moe.experts.0.w1."))
        .cloned().collect::<Vec<_>>();
    let operand = format::detect(format::group_by_stem(&w1).values().next().unwrap()).unwrap();
    assert!(operand.is_compressed_nvfp4() && !operand.is_nvfp4());
    assert_eq!((operand.logical.as_slice(), operand.stored.as_slice()), (&[3072, 6144][..], &[3072, 3072][..]));
    assert_eq!(operand.global_scale.unwrap().direction, format::ScaleDirection::Divide);
    let model = MINIMAX_M3.open(&checkpoint).unwrap();
    let contract = model.experts(&operand).unwrap();
    assert_eq!((contract.package.as_str(), contract.block, contract.spark_worlds), (PACKAGE, 16, vec![2, 3, 4, 6]));
    // ModelOpt's multiplying scalars are not this family's experts.
    let other_dir = tempfile::tempdir().unwrap();
    testing::write_snapshot(other_dir.path(), &serde_json::json!({}), &testing::nvfp4("x", 3072, 6144), None);
    let other = Checkpoint::open(other_dir.path()).unwrap();
    let other = format::detect(format::group_by_stem(&other.tensors).values().next().unwrap()).unwrap();
    assert!(other.is_nvfp4() && !other.is_compressed_nvfp4() && model.experts(&other).is_none());
}

#[test]
fn drifted_headers_are_named() {
    let dir = tempfile::tempdir().unwrap();
    testing::write_minimax_m3_snapshot(dir.path(), false, false);
    let shard = dir.path().join("model-00005.safetensors");
    let mut tensors: Vec<testing::Tensor> = testing::minimax_m3_tensors(false).into_iter()
        .filter(|t| t.0.contains(".layers.4.")).collect();
    for tensor in &mut tensors {
        if tensor.0.ends_with(".experts.7.w2.weight_scale") { tensor.1 = "U8"; }
    }
    testing::write_safetensors(&shard, &tensors);
    let report = testing::plan(dir.path(), &PlanOptions::default()).unwrap();
    let experts = report.components.iter().find(|c| c.component == Component::RoutedExpert).unwrap();
    assert_eq!(experts.rejected, 1);
    assert!(experts.rejections[0].tensor.contains("layers.4.block_sparse_moe.experts.7.w2"), "{:?}", experts.rejections);
}

#[test]
fn vision_uses_encoder_owner_and_reserves_weights_and_scratch() {
    use crate::plan::{encoder::EncoderKind, MediaMode};
    let dir = snapshot(true, false);
    let programs = dir.path().join("PROGRAMS.json");
    std::fs::write(&programs, admission::tests::programs_manifest().to_string()).unwrap();
    for (mode, expected) in [(MediaMode::Rtx(Some(0)), EncoderKind::Rtx { gpu: 0 }),
        (MediaMode::Spark(Some(2)), EncoderKind::Spark { rank: 2 }), (MediaMode::Off, EncoderKind::Off)] {
        let mut options = with_layout(PRO_TOTAL_BYTES, 4);
        let layout = options.layout.as_mut().unwrap();
        (layout.workspace_manifest, layout.native_mtp_layers, layout.prefill_rows, layout.concurrency) =
            (Some(programs.clone()), 0, 256, 16);
        options.vision = mode;
        let report = testing::plan(dir.path(), &options).unwrap();
        assert!(report.placement_supported, "{mode:?}: {:?}", report.hints);
        let encoder = report.encoder.as_ref().unwrap();
        assert_eq!(encoder.kind, expected);
        let items: Vec<_> = report.memory_layout.as_ref().unwrap().devices.iter().flat_map(|d| &d.items)
            .filter(|i| i.group.starts_with("vision")).collect();
        if expected == EncoderKind::Off { assert!(items.is_empty()); assert_eq!(encoder.admitted_bytes(), 0); }
        else {
            assert_eq!(items.len(), 2);
            // One 576-token image's tower scratch with the BLAS workspace (the native ledger's).
            assert_eq!(encoder.scratch, 210_841_856);
            assert!(encoder.weights > report.components.iter().find(|c| c.component == Component::Vision).unwrap().bytes);
            assert_eq!(items.iter().map(|i| i.bytes).sum::<u64>(), encoder.admitted_bytes());
        }
    }
}

#[test]
fn native_manifest_uses_shared_allocation_ledger() {
    let dir = snapshot(false, false);
    let programs = dir.path().join("PROGRAMS.json");
    std::fs::write(&programs, admission::tests::programs_manifest().to_string()).unwrap();
    let experts = dir.path().join("fp8/fp8-minimax-nvfp4a4");
    std::fs::create_dir_all(&experts).unwrap();
    std::fs::write(experts.join("manifest.json"), admission::tests::expert_manifest().to_string()).unwrap();
    let mut options = with_layout(PRO_TOTAL_BYTES, 3);
    let layout = options.layout.as_mut().unwrap();
    layout.workspace_manifest = Some(programs.clone());
    layout.prefill_rows = 256;
    layout.concurrency = 16;
    layout.pool_tokens = Some(16384);
    layout.native_mtp_layers = 0;
    let report = testing::plan(dir.path(), &options).unwrap();
    assert!(report.placement_supported, "{:?}", report.hints);
    assert!(report.wip_launchable() && !report.executable(), "{:?}", report.hints);
    let memory = report.memory_layout.as_ref().unwrap();
    assert_eq!(memory.pool_tokens, 16384);
    assert!(memory.notes.iter().any(|n| n.contains("native allocation ledger")));
    let rtx = memory.devices.iter().find(|d| d.kind == DeviceKind::Rtx).unwrap();
    let item = |name: &str| rtx.items.iter().find(|i| i.group == name).unwrap();
    // Four admitted 256-row slots by default (#762): the per-row buffers hold their 1024-row reserve.
    assert_eq!((report.prefill_unit_rows, report.prefill_waves, report.prefill_slots, report.prefill_span_rows),
        (Some(256), Some(2), Some(4), Some(1024)));
    assert_eq!(item("MiniMax residual").bytes, 4 * 256 * 6144 * 4);
    // Every program shares the scratch arena; the prefill scorer's is the largest.
    assert_eq!(item("MiniMax scratch").bytes, 134_799_872);
    assert!(rtx.used_bytes() <= PRO_TOTAL_BYTES);
    options.layout.as_mut().unwrap().minimax_kv = crate::families::qwen4::Qwen4KvCache::Bf16;
    let refused = testing::plan(dir.path(), &options).unwrap();
    assert!(!refused.placement_supported && refused.hints.iter().any(|h| h.what.contains("BF16 KV")));
    options.layout.as_mut().unwrap().minimax_kv = crate::families::qwen4::Qwen4KvCache::Fp8;
    options.layout.as_mut().unwrap().rtx_bytes.push(PRO_TOTAL_BYTES);
    let refused = testing::plan(dir.path(), &options).unwrap();
    assert!(!refused.placement_supported && refused.hints.iter().any(|h| h.what.contains("two-GPU")));
}

#[test]
fn native_manifest_admits_bundled_dspark_and_charges_its_full_layout() {
    use crate::families::minimax_m3::dspark::{DraftMemory, DsparkContract};
    use cuteafd_core::memory_layout::Category;
    let dir = snapshot(false, true);
    let programs = dir.path().join("PROGRAMS.json");
    std::fs::write(&programs, admission::tests::programs_manifest().to_string()).unwrap();
    let mut options = with_layout(PRO_TOTAL_BYTES, 3);
    let layout = options.layout.as_mut().unwrap();
    layout.workspace_manifest = Some(programs);
    layout.prefill_rows = 256;
    layout.concurrency = 16;
    layout.pool_tokens = Some(16384);
    layout.physical_sms = Some(188);
    layout.native_mtp_layers = 0;
    let off = testing::plan(dir.path(), &options).unwrap();
    assert!(off.placement_supported, "{:?}", off.hints);
    options.layout.as_mut().unwrap().native_mtp_layers = 5;
    let on = testing::plan(dir.path(), &options).unwrap();
    assert!(on.placement_supported && on.wip_launchable(), "{:?}", on.hints);
    let contract = DsparkContract::open(dir.path()).unwrap();
    // Sixteen sequences each hold a context ring (at most 64 decode rows / 2); the taps hold the
    // span: the 1024-row reserve of four admitted 256-row slots (#762).
    let draft = DraftMemory::new(&contract, admission::default_draft_rings(16, 64), 4 * 256, 188).unwrap();
    assert_eq!(draft.slots, 16);
    let rtx = |report: &crate::plan::PlanReport| report.memory_layout.as_ref().unwrap().devices.iter()
        .find(|d| d.kind == DeviceKind::Rtx).unwrap().clone();
    let (off_rtx, on_rtx) = (rtx(&off), rtx(&on));
    assert!(off_rtx.items.iter().all(|i| i.category != Category::Drafter));
    // Decode runs eagerly: neither plan reserves startup decode graphs.
    let graphs = |device: &cuteafd_core::memory_layout::DeviceLayout| device.items.iter()
        .filter(|i| ["graphs", crate::placement::inventory::GRAPH_GROWTH].contains(&i.group.as_str())).map(|i| i.bytes).sum::<u64>();
    assert_eq!((graphs(&off_rtx), graphs(&on_rtx)), (0, 0));
    // A DSpark step projects its sequences' drafts through the target head: sixteen drafting
    // sequences take 112 head rows where the decode step takes 64.
    let head = |device: &cuteafd_core::memory_layout::DeviceLayout| device.items.iter()
        .filter(|i| ["MiniMax logits", "MiniMax head_input"].contains(&i.group.as_str())).map(|i| i.bytes).sum::<u64>();
    assert!(head(&on_rtx) > head(&off_rtx));
    assert_eq!(on_rtx.used_bytes() - off_rtx.used_bytes(), draft.bytes + head(&on_rtx) - head(&off_rtx));
    assert_eq!(on_rtx.items.iter().filter(|i| i.category == Category::Drafter).map(|i| i.bytes).sum::<u64>(), draft.bytes);
    assert_eq!(on.bytes_by_owner["bundled DSpark weights"], draft.weights_bytes);
    assert_eq!(on.memory_layout.as_ref().unwrap().pool_tokens, 16384);
    assert!(on.memory_layout.as_ref().unwrap().notes.iter().any(|n| n.ends_with("speculation bundled DSpark")));
    // The plan states the rings' share of the pool.
    let ring_mib = draft.ring_bytes as f64 / draft.slots as f64 / (1u64 << 20) as f64;
    assert!(on.memory_layout.as_ref().unwrap().notes.iter()
        .any(|n| n.starts_with(&format!("16 drafter context rings of {ring_mib:.1} MiB each"))), "{:?}",
        on.memory_layout.as_ref().unwrap().notes);
    assert!(off.memory_layout.as_ref().unwrap().notes.iter().all(|n| !n.contains("drafter context rings")));
    assert!(off.memory_layout.as_ref().unwrap().notes.iter().any(|n| n.ends_with("speculation off")));
    options.layout.as_mut().unwrap().physical_sms = Some(170);
    let smaller = testing::plan(dir.path(), &options).unwrap();
    assert!(smaller.placement_supported, "{:?}", smaller.hints);
    // The SM-sized draft attention arena: 532,480 bytes more per drafting sequence at 188 SMs.
    assert_eq!(on_rtx.used_bytes() - rtx(&smaller).used_bytes(), 16 * 532_480);
    // A unit below a KV page is no prefill shape at all (the drafter's taps would need seven rows).
    options.layout.as_mut().unwrap().prefill_rows = 6;
    options.layout.as_mut().unwrap().concurrency = 1;
    let refused = testing::plan(dir.path(), &options).unwrap();
    assert!(!refused.placement_supported);
    assert!(refused.hints.iter().any(|h| h.what.contains("MiniMax prefill shape")), "{:?}", refused.hints);
    options.layout.as_mut().unwrap().prefill_rows = 256;
    options.layout.as_mut().unwrap().concurrency = 16;
    options.layout.as_mut().unwrap().drafter_bytes = 1;
    let refused = testing::plan(dir.path(), &options).unwrap();
    assert!(!refused.placement_supported);
    assert!(refused.hints.iter().any(|h| h.what.contains("external drafter")), "{:?}", refused.hints);
}

#[test]
fn native_dspark_requires_a_bundled_checkpoint() {
    let dir = snapshot(false, false);
    let programs = dir.path().join("PROGRAMS.json");
    std::fs::write(&programs, admission::tests::programs_manifest().to_string()).unwrap();
    let mut options = with_layout(PRO_TOTAL_BYTES, 3);
    let layout = options.layout.as_mut().unwrap();
    layout.workspace_manifest = Some(programs);
    layout.native_mtp_layers = 5;
    let refused = testing::plan(dir.path(), &options).unwrap();
    assert!(!refused.placement_supported);
    assert!(refused.hints.iter().any(|h| h.what.contains("MiniMax bundled DSpark")), "{:?}", refused.hints);
}

#[test]
fn a_missing_native_manifest_is_refused() {
    let dir = snapshot(false, false);
    let mut options = with_layout(PRO_TOTAL_BYTES, 3);
    options.layout.as_mut().unwrap().workspace_manifest = Some(dir.path().join("missing-PROGRAMS.json"));
    options.layout.as_mut().unwrap().native_mtp_layers = 0;
    let report = testing::plan(dir.path(), &options).unwrap();
    assert!(!report.placement_supported);
    assert!(report.memory_layout.unwrap().devices.is_empty());
    assert!(report.hints.iter().any(|h| h.what.contains("PROGRAMS.json")));
}

/// TP2 with 1024-row units and 64 sequences at a 108 GiB Spark budget: 55 Spark layers fit it while
/// serving but not while loading. On hardware the worker refused them with "load 116371564032 serve
/// 115379228928 budget 115964116992": its load peak adds the largest pinned upload and read scratch
/// of a layer to the residents, package scratch and runtime residual.
#[test]
fn tp2_onboarding_sizes_the_sparks_for_their_load_peak() {
    let dir = snapshot(false, false);
    let programs = dir.path().join("PROGRAMS.json");
    std::fs::write(&programs, admission::tests::programs_manifest().to_string()).unwrap();
    let experts = dir.path().join("fp8/fp8-minimax-nvfp4a4");
    std::fs::create_dir_all(&experts).unwrap();
    let mut package = admission::tests::expert_manifest();
    package["layouts"].as_object_mut().unwrap().retain(|name, _| name == "tp1");
    std::fs::write(experts.join("manifest.json"), package.to_string()).unwrap();
    let mut options = with_layout(PRO_TOTAL_BYTES, 2);
    options.vision = crate::plan::MediaMode::Off;
    let layout = options.layout.as_mut().unwrap();
    layout.workspace_manifest = Some(programs);
    layout.native_mtp_layers = 0;
    layout.prefill_rows = 1024;
    layout.concurrency = 64;
    layout.spark_allocation_budget_bytes = Some(108 << 30);
    let report = testing::plan(dir.path(), &options).unwrap();
    assert!(report.wip_launchable(), "{:?}", report.hints);
    let memory = report.memory_layout.as_ref().unwrap();
    assert!(memory.notes.iter().any(|n| n.starts_with("3 routed layers on the RTX")), "{:?}", memory.notes);
}

/// The default Spark budget, measured on headless Sparks, holds every routed layer at TP2 with the
/// default prefill (four admitted 4096-row slots on each rank's typed owners, #762; 64 sequences):
/// nothing onboards onto the RTX.
#[test]
fn tp2_default_budget_keeps_every_routed_layer_on_the_sparks() {
    let dir = snapshot(false, false);
    let programs = dir.path().join("PROGRAMS.json");
    std::fs::write(&programs, admission::tests::programs_manifest().to_string()).unwrap();
    let experts = dir.path().join("fp8/fp8-minimax-nvfp4a4");
    std::fs::create_dir_all(&experts).unwrap();
    let mut package = admission::tests::expert_manifest();
    package["layouts"].as_object_mut().unwrap().retain(|name, _| name == "tp1");
    std::fs::write(experts.join("manifest.json"), package.to_string()).unwrap();
    let mut options = with_layout(PRO_TOTAL_BYTES, 2);
    options.vision = crate::plan::MediaMode::Off;
    let layout = options.layout.as_mut().unwrap();
    layout.workspace_manifest = Some(programs);
    layout.native_mtp_layers = 0;
    layout.concurrency = 64;
    let report = testing::plan(dir.path(), &options).unwrap();
    assert!(report.wip_launchable(), "{:?}", report.hints);
    let memory = report.memory_layout.as_ref().unwrap();
    assert!(memory.notes.iter().any(|n| n.starts_with("0 routed layers on the RTX")), "{:?}", memory.notes);
    let spark = memory.devices.iter().find(|d| d.kind == DeviceKind::Spark).unwrap();
    assert!(spark.items.iter().any(|i| i.group == "execution owners"
        && i.format.starts_with("1 decode + 1 prefill (m4096 scratch once; 4 x 4096-row units)")), "{:?}", spark.items);
}

/// TP2 on the RTX PRO 6000 with automatic vision: the bundled tower stays on the coordinator GPU
/// beside the routed layers a 108 GiB Spark budget onboards, and text admission keeps its pool.
#[test]
fn native_tp2_auto_vision_preserves_text_admission() {
    use crate::plan::{encoder::EncoderKind, MediaMode};
    let dir = snapshot(true, false);
    let programs = dir.path().join("PROGRAMS.json");
    std::fs::write(&programs, admission::tests::programs_manifest().to_string()).unwrap();
    let experts = dir.path().join("fp8/fp8-minimax-nvfp4a4");
    std::fs::create_dir_all(&experts).unwrap();
    let mut package = admission::tests::expert_manifest();
    package["layouts"].as_object_mut().unwrap().retain(|name, _| name == "tp1");
    std::fs::write(experts.join("manifest.json"), package.to_string()).unwrap();
    let mut options = with_layout(PRO_TOTAL_BYTES, 2);
    options.vision = MediaMode::Auto;
    let layout = options.layout.as_mut().unwrap();
    layout.workspace_manifest = Some(programs);
    layout.native_mtp_layers = 0;
    // Two pinned 1024-row waves (a static shape: their slots only).
    (layout.prefill_rows, layout.prefill_lanes) = (1024, 2);
    layout.concurrency = 64;
    // A 108 GiB Spark budget, so TP2 onboards layers (the load peak takes three).
    layout.spark_allocation_budget_bytes = Some(108 << 30);
    let report = testing::plan(dir.path(), &options).unwrap();
    assert!(report.wip_launchable(), "{:?}", report.hints);
    assert_eq!(report.encoder.as_ref().unwrap().kind, EncoderKind::Rtx { gpu: 0 });
    let memory = report.memory_layout.as_ref().unwrap();
    assert!(memory.pool_tokens >= 64);
    assert!(memory.devices.iter().all(|d| d.used_bytes() <= d.capacity_bytes));
    assert!(memory.notes.iter().any(|n| n.starts_with("3 routed layers on the RTX")), "{:?}", memory.notes);
    assert!(!report.hints.iter().any(|h| h.what.starts_with("routed experts need ")));
    // Logits cover decode rows, not every prefill row, so one 4096-row unit fits too, and it holds
    // more per-row buffers than two 1024-row waves.
    options.layout.as_mut().unwrap().prefill_rows = 4096;
    options.layout.as_mut().unwrap().prefill_lanes = 1;
    let wide = testing::plan(dir.path(), &options).unwrap();
    assert!(wide.wip_launchable(), "{:?}", wide.hints);
    assert!(wide.memory_layout.as_ref().unwrap().pool_tokens < memory.pool_tokens);
    (options.layout.as_mut().unwrap().prefill_rows, options.layout.as_mut().unwrap().prefill_lanes) = (1024, 2);
    options.layout.as_mut().unwrap().onboard = Some(crate::placement::Onboard::Layers(0));
    let refused = testing::plan(dir.path(), &options).unwrap();
    assert!(!refused.fits && !refused.placement_supported);
    assert!(refused.hints.iter().any(|h| h.what.contains("MiniMax allocation needs")));
}

/// An RTX PRO 6000 plan with the image's program and expert package manifests, vision off and the
/// bundled DSpark.
fn native_options(dir: &std::path::Path, ranks: usize) -> PlanOptions {
    let programs = dir.join("PROGRAMS.json");
    std::fs::write(&programs, admission::tests::programs_manifest().to_string()).unwrap();
    let experts = dir.join("fp8/fp8-minimax-nvfp4a4");
    std::fs::create_dir_all(&experts).unwrap();
    std::fs::write(experts.join("manifest.json"), admission::tests::expert_manifest().to_string()).unwrap();
    let mut options = with_layout(PRO_TOTAL_BYTES, ranks);
    options.vision = crate::plan::MediaMode::Off;
    let layout = options.layout.as_mut().unwrap();
    layout.workspace_manifest = Some(programs);
    layout.physical_sms = Some(188);
    layout.native_mtp_layers = 5;
    options
}

/// The one Spark link's return slots: a GPU range per rank holding every admitted slot, where the NIC
/// lands every reply (the shared `prefill_return_bytes`): a pinned shape's waves of its unit, or the
/// unpinned plan's 8256-row reserve for four 4096-row slots (#762). Host admission charges the same
/// rows on one transport.
#[test]
fn spark_intake_is_every_waves_return_slots_on_the_gpu() {
    let dir = snapshot(false, false);
    for (ranks, pins, (rows, waves, slots, span)) in [(3u64, (0, 0), (4096u64, 2u64, 4u64, 8256u64)),
        (6, (0, 0), (4096, 2, 4, 8256)), (3, (4096, 1), (4096, 1, 1, 4096)), (3, (4096, 2), (4096, 2, 2, 8192)),
        (2, (2048, 4), (2048, 4, 4, 8192)), (6, (2752, 3), (2752, 3, 3, 8256)), (4, (1024, 3), (1024, 3, 3, 3072))] {
        let mut options = native_options(dir.path(), ranks as usize);
        let layout = options.layout.as_mut().unwrap();
        (layout.prefill_rows, layout.prefill_lanes, layout.native_mtp_layers) = (pins.0, pins.1, 0);
        (layout.concurrency, layout.context_tokens, layout.pool_tokens) = (16, 65536, Some(65536));
        let report = testing::plan(dir.path(), &options).unwrap();
        assert!(report.placement_supported, "{:?}", report.hints);
        assert_eq!((report.prefill_unit_rows, report.prefill_waves, report.prefill_slots, report.prefill_span_rows,
            report.prefill_transports), (Some(rows), Some(waves), Some(slots), Some(span), Some(1)), "TP{ranks} {pins:?}");
        let gpu = report.memory_layout.as_ref().unwrap().devices.iter().find(|d| d.kind == DeviceKind::Rtx).unwrap()
            .items.iter().find(|i| i.group == "MiniMax spark_intake").unwrap().bytes;
        assert_eq!(gpu, crate::serving_capacity::prefill_return_bytes(1, span, ranks, 6144), "TP{ranks} {pins:?}");
        let host = report.host_ledger.as_ref().unwrap().items.iter().find(|i| i.group == "spark-intake").unwrap().bytes;
        assert_eq!(host, gpu, "TP{ranks} {pins:?}: host-mode bound of the same slots");
    }
}

/// MiniMax on an RTX 5090 is not supported on this engine yet: `cuteafd plan` names it, with a
/// layout or without.
#[test]
fn rtx_5090_is_refused_by_name() {
    let dir = snapshot(false, false);
    let named = |report: &crate::plan::PlanReport| report.hints.iter()
        .any(|h| h.what.contains("MiniMax on an RTX 5090 is not supported on this engine yet"));
    for ranks in [2, 3, 4, 6] {
        let mut options = native_options(dir.path(), ranks);
        (options.layout.as_mut().unwrap().rtx_bytes, options.layout.as_mut().unwrap().native_mtp_layers) = (vec![RTX_5090], 0);
        let report = testing::plan(dir.path(), &options).unwrap();
        assert!(!report.placement_supported && !report.wip_launchable() && named(&report), "{:?}", report.hints);
        options.layout.as_mut().unwrap().rtx_bytes = vec![PRO_TOTAL_BYTES];
        let pro = testing::plan(dir.path(), &options).unwrap();
        assert!(pro.wip_launchable() && !named(&pro), "{:?}", pro.hints);
    }
    let inventory = testing::plan(dir.path(), &PlanOptions { coordinator_budget_bytes: RTX_5090,
        placement: ExpertPlacement::from_spark_ranks(4), ..Default::default() }).unwrap();
    assert!(named(&inventory), "{:?}", inventory.hints);
}

/// The RTX PRO 6000 cards (the example configs' shapes: 64 sequences, the checkpoint's context, FP8
/// K/V, DFlash2 with its default rings, automatic vision, the default Spark budget) on the units
/// contract: #762's adaptive prefill (four 4096-row slots in the 8256-row reserve, starting at two
/// waves), one Spark transport, the typed owners (one decode, one prefill buffering a unit per slot) on
/// every rank, and every rank within its budget. The coordinator image's expert package carries its TP1
/// layout only, so Spark scratch is the estimate, as on a real launch.
#[test]
fn pro6000_plans_tp2_tp3_tp4_and_tp6_on_prefill_units() {
    use crate::plan::{encoder::EncoderKind, MediaMode};
    let dir = snapshot(true, false);
    let drafter = tempfile::tempdir().unwrap();
    std::fs::write(drafter.path().join("config.json"), testing::minimax_dflash2_config().to_string()).unwrap();
    for (ranks, unit, waves, slots, span) in [(2usize, 4096u64, 2u64, 4u64, 8256u64), (3, 4096, 2, 4, 8256),
        (4, 4096, 2, 4, 8256), (6, 4096, 2, 4, 8256)] {
        let mut options = native_options(dir.path(), ranks);
        let mut package = admission::tests::expert_manifest();
        package["layouts"].as_object_mut().unwrap().retain(|name, _| name == "tp1");
        std::fs::write(dir.path().join("fp8/fp8-minimax-nvfp4a4/manifest.json"), package.to_string()).unwrap();
        options.vision = MediaMode::Auto;
        let layout = options.layout.as_mut().unwrap();
        (layout.native_mtp_layers, layout.concurrency) = (0, 64);
        layout.minimax_drafter_snapshot = Some(drafter.path().to_path_buf());
        let report = testing::plan(dir.path(), &options).unwrap();
        assert!(report.wip_launchable(), "TP{ranks}: {:?}", report.hints);
        assert_eq!((report.prefill_unit_rows, report.prefill_waves, report.prefill_slots, report.prefill_span_rows,
            report.prefill_transports), (Some(unit), Some(waves), Some(slots), Some(span), Some(1)), "TP{ranks}");
        let memory = report.memory_layout.as_ref().unwrap();
        assert!(memory.pool_tokens > 1_000_000, "TP{ranks}: {} pool tokens", memory.pool_tokens);
        assert!(memory.devices.iter().all(|d| d.used_bytes() <= d.capacity_bytes), "TP{ranks}");
        assert!(memory.notes.iter().any(|n| n.starts_with("0 routed layers on the RTX")), "TP{ranks}: {:?}", memory.notes);
        let sparks = memory.devices.iter().filter(|d| d.kind == DeviceKind::Spark).collect::<Vec<_>>();
        assert_eq!(sparks.len(), ranks);
        let sizing = admission::RuntimeSizing { unit_rows: unit as usize, decode_rows: 64, sequences: 64, context: 1 << 20,
            spark_ranks: ranks, logit_rows: 64, waves: waves as usize, slots: slots as usize };
        let owners = sizing.spark_executions().unwrap()
            .workspace_bytes(|extent| admission::spark_owner_bytes(None, ranks, extent)).unwrap().unwrap();
        for spark in &sparks {
            let owner = spark.items.iter().find(|i| i.group == "execution owners").unwrap();
            assert_eq!(owner.format, format!("1 decode + 1 prefill (m4096 scratch once; {slots} x {unit}-row units); \
                0 bytes already reserved"), "TP{ranks}");
            let item = |name: &str| spark.items.iter().find(|i| i.group == name).unwrap().bytes;
            assert_eq!((owner.bytes, item("execution staging")), (owners.device_bytes, owners.pinned_bytes), "TP{ranks}");
        }
        // One endpoint's rings per rank on the coordinator host.
        let host = report.host_ledger.as_ref().unwrap();
        let rings = host.items.iter().find(|i| i.group == "transport/rdma-rings").unwrap().bytes;
        let endpoint = cuteafd_core::SparkEndpoint { ring_depth: None, capacity: unit as u32, ingress: cuteafd_core::SparkIngress::Bf16 };
        let geometry = cuteafd_core::ExpertGeometry::MINIMAX_M3;
        let (request, response) = endpoint.wire_bytes(geometry).unwrap();
        let ring = cuteafd_core::SparkRings { max_frame_bytes: (64 << 20).max(request).max(response), depth: 8,
            slot_bytes: 8 << 20, alignment: 4096 };
        assert_eq!(rings, endpoint.ring_bytes(geometry, ring).unwrap() as u64 * ranks as u64, "TP{ranks}");
        // The tower: on the GPU at TP2, on Spark rank 0 from TP3 (the ranks have room beside the experts).
        assert_eq!(report.encoder.as_ref().unwrap().kind,
            if ranks == 2 { EncoderKind::Rtx { gpu: 0 } } else { EncoderKind::Spark { rank: 0 } }, "TP{ranks}");
    }
}

#[test]
fn minimax_host_embedding_and_c1_rows_preserve_draft_and_increase_pool() {
    use cuteafd_core::memory_layout::Category;
    let dir = snapshot(false, true);
    let options = native_options(dir.path(), 4);
    let baseline = testing::plan(dir.path(), &options).unwrap();
    assert!(baseline.placement_supported, "{:?}", baseline.hints);
    // One sequence of 256-row prefill chunks: what C1 serving needs, DSpark kept.
    let c1 = |options: &mut PlanOptions| {
        let layout = options.layout.as_mut().unwrap();
        (layout.prefill_rows, layout.concurrency) = (256, 1);
    };
    let mut sized = options.clone();
    c1(&mut sized);
    let sized = testing::plan(dir.path(), &sized).unwrap();
    assert!(sized.placement_supported, "{:?}", sized.hints);
    assert_eq!(sized.prefill_unit_rows, Some(256));
    assert!(sized.memory_layout.as_ref().unwrap().notes.iter().any(|n| n.contains("sequences 1")));
    let mut mapped = options.clone();
    mapped.layout.as_mut().unwrap().host_embedding = true;
    let host = testing::plan(dir.path(), &mapped).unwrap();
    assert!(host.placement_supported, "{:?}", host.hints);
    let embedding = 200064 * 6144 * 2;
    assert_eq!(host.bytes_by_owner["host-mapped embedding"], embedding);
    assert_eq!(baseline.bytes_by_owner["coordinator text weights"] - host.bytes_by_owner["coordinator text weights"], embedding);
    let pool = |r: &crate::plan::PlanReport| r.memory_layout.as_ref().unwrap().pool_tokens;
    assert!(pool(&sized) > pool(&baseline));
    assert!(pool(&host) > pool(&baseline));
    for ranks in [2, 3, 4, 6] {
        let mut combined = native_options(dir.path(), ranks);
        c1(&mut combined);
        combined.layout.as_mut().unwrap().host_embedding = true;
        let report = testing::plan(dir.path(), &combined).unwrap();
        assert!(report.placement_supported, "TP{ranks}: {:?}", report.hints);
        assert!(pool(&report) > 0);
        let memory = report.memory_layout.as_ref().unwrap();
        assert!(memory.devices.iter().all(|d| d.used_bytes() <= d.capacity_bytes));
        let draft: u64 = memory.devices.iter().flat_map(|d| &d.items)
            .filter(|i| i.category == Category::Drafter).map(|i| i.bytes).sum();
        assert!(draft > report.bytes_by_owner["bundled DSpark weights"]);
        assert_eq!(report.bytes_by_owner["bundled DSpark weights"], baseline.bytes_by_owner["bundled DSpark weights"]);
        if ranks == 4 { assert!(pool(&report) > pool(&sized) && pool(&report) > pool(&host)); }
    }
}

#[test]
fn minimax_draft_context_slots_honor_explicit_shapes() {
    let dir = snapshot(false, true);
    let mut options = native_options(dir.path(), 4);
    let layout = options.layout.as_mut().unwrap();
    layout.prefill_rows = 512;
    layout.concurrency = 4;
    layout.draft_context_slots = Some(4);
    let report = testing::plan(dir.path(), &options).unwrap();
    assert!(report.placement_supported, "{:?}", report.hints);
    assert_eq!(report.prefill_unit_rows, Some(512));
    assert!(report.memory_layout.as_ref().unwrap().notes.iter().any(|n| n.contains("sequences 4")));
    // Four sequences default to four rings (64 decode rows hold four drafting requests and more).
    options.layout.as_mut().unwrap().draft_context_slots = None;
    let derived = testing::plan(dir.path(), &options).unwrap();
    assert_eq!(derived.memory_layout.as_ref().unwrap().pool_tokens, report.memory_layout.as_ref().unwrap().pool_tokens);
    assert!(derived.memory_layout.as_ref().unwrap().notes.iter().any(|n| n.starts_with("4 drafter context rings")));
    options.layout.as_mut().unwrap().draft_context_slots = Some(1);
    let single = testing::plan(dir.path(), &options).unwrap();
    assert!(single.placement_supported, "{:?}", single.hints);
    let pool = |report: &crate::plan::PlanReport| report.memory_layout.as_ref().unwrap().pool_tokens;
    assert!(pool(&single) > pool(&report), "fewer context rings leave more pool");
    options.layout.as_mut().unwrap().draft_context_slots = Some(5);
    let refused = testing::plan(dir.path(), &options).unwrap();
    assert!(!refused.placement_supported && refused.hints.iter().any(|h| h.what.contains("draft context slots")));
}

/// Spark TP6 on the RTX PRO 6000: every routed layer in six equal 512-wide slices, smaller than
/// a TP4 rank's, each rank charged the package's exported TP6 scratch, and the automatic tower on
/// Spark rank 0.
#[test]
fn tp6_holds_every_routed_layer_in_six_equal_slices() {
    use crate::plan::{encoder::EncoderKind, MediaMode};
    let dir = snapshot(true, false);
    let planned = |ranks| {
        let mut options = native_options(dir.path(), ranks);
        options.vision = MediaMode::Auto;
        let layout = options.layout.as_mut().unwrap();
        (layout.rtx_bytes, layout.physical_sms, layout.native_mtp_layers) = (vec![PRO_TOTAL_BYTES], Some(188), 0);
        testing::plan(dir.path(), &options).unwrap()
    };
    let (tp4, tp6) = (planned(4), planned(6));
    assert!(tp6.wip_launchable(), "{:?}", tp6.hints);
    let memory = tp6.memory_layout.as_ref().unwrap();
    assert!(memory.notes.iter().any(|n| n.starts_with("0 routed layers on the RTX") && n.ends_with("57 on 6 Sparks")),
        "{:?}", memory.notes);
    let sparks = |report: &crate::plan::PlanReport| report.memory_layout.as_ref().unwrap().devices.iter()
        .filter(|d| d.kind == DeviceKind::Spark).cloned().collect::<Vec<_>>();
    let item = |device: &cuteafd_core::memory_layout::DeviceLayout, name: &str|
        device.items.iter().find(|i| i.group == name).unwrap().bytes;
    let ranks = sparks(&tp6);
    assert_eq!(ranks.len(), 6);
    let experts = item(&ranks[0], "routed expert weights");
    assert!(ranks.iter().all(|d| item(d, "routed expert weights") == experts));
    assert!(experts < item(&sparks(&tp4)[1], "routed expert weights"));
    // The package exports tp6: each rank's owners take its capacities (decode m256, prefill m4096), not the estimate.
    assert_eq!((tp6.prefill_unit_rows, tp6.prefill_waves, tp6.prefill_slots), (Some(4096), Some(2), Some(4)));
    let owners = admission::RuntimeSizing { unit_rows: 4096, decode_rows: 64, sequences: 64, context: 1 << 20, spark_ranks: 6,
        logit_rows: 64, waves: 2, slots: 4 }.spark_executions().unwrap();
    assert_eq!((owners.decode[0].capacity, owners.prefill.unit_rows, owners.prefill.buffered_units), (64, 4096, 4));
    let row = 6144 * 2 + 4 * 8 + 6144 * 2;
    assert!(ranks.iter().all(|d| item(d, "execution owners") == 102_837_248 + 64 * row + 987_654_321 + 4096 * row));
    assert_eq!(tp6.encoder.as_ref().unwrap().kind, EncoderKind::Spark { rank: 0 });
}
