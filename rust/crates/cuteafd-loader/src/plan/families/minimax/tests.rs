use super::*;
use crate::placement::inventory::PRO_TOTAL_BYTES;
use crate::plan::{self, layout, testing, ExpertPlacement, PlanOptions, Status};
use cuteafd_core::memory_layout::DeviceKind;

fn snapshot(vision: bool, draft: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    testing::write_minimax_m3_snapshot(dir.path(), vision, draft);
    dir
}

/// The 5090's usable logical budget (`--coordinator-gpu-budget-gib 31.8`).
const RTX_5090: u64 = 34_144_990_003;

fn with_layout(rtx: u64, ranks: usize) -> PlanOptions {
    PlanOptions { placement: ExpertPlacement::from_spark_ranks(ranks),
        layout: Some(layout::LayoutOptions { rtx_bytes: vec![rtx], ..Default::default() }), ..Default::default() }
}

#[test]
fn release_headers_satisfy_the_contract() {
    let dir = snapshot(true, true);
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
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
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    let experts = report.components.iter().find(|c| c.component == Component::RoutedExpert).unwrap();
    assert_eq!(experts.rejected, 1);
    assert!(experts.rejections[0].tensor.contains("layers.4.block_sparse_moe.experts.7.w2"), "{:?}", experts.rejections);
}

#[test]
fn card_layouts_on_5090_and_pro_6000() {
    let dir = snapshot(false, true);
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let mut seen = Vec::new();
    for (card, rtx) in [("5090", RTX_5090), ("pro6000", PRO_TOTAL_BYTES)] {
        for ranks in [2, 3, 4] {
            let report = plan::plan(dir.path(), &with_layout(rtx, ranks)).unwrap();
            let memory = report.memory_layout.as_ref().unwrap();
            let spark = memory.devices.iter().filter(|d| d.kind == DeviceKind::Spark).map(|d| d.used_bytes()).max().unwrap();
            let rtx_used = memory.devices.iter().find(|d| d.kind == DeviceKind::Rtx).unwrap().used_bytes();
            let onboarded = memory.notes.iter().find_map(|n| n.split_once(" routed layers on the RTX").map(|(n, _)| n.parse::<usize>().unwrap())).unwrap();
            assert!(spark <= placement::SPARK_BUDGET_BYTES, "{card} TP{ranks}: Spark {:.1} GiB", gib(spark));
            seen.push((card, ranks, onboarded, memory.pool_tokens, rtx_used <= rtx));
        }
    }
    let row = |card, ranks| *seen.iter().find(|r| r.0 == card && r.1 == ranks).unwrap();
    // TP3 and TP4 hold every routed layer on the Sparks.
    for ranks in [3, 4] {
        assert_eq!(row("5090", ranks).2, 0);
        assert_eq!(row("pro6000", ranks).2, 0);
    }
    // TP2 (108.2 GiB of experts per rank) onboards bottom layers to the RTX.
    assert!(row("pro6000", 2).2 > 0 && row("pro6000", 2).4);
    // The PRO 6000 holds a full 1M context beside its weights; the 5090 a quarter of one.
    assert!(row("pro6000", 4).3 >= 1 << 20, "{seen:?}");
    // 9.2 GiB of FP8 KV + index keys at 69,360 B/token beside 22.6 GiB of weights,
    // drafter, workspace estimate and headroom.
    assert!((100_000..200_000).contains(&row("5090", 4).3), "{seen:?}");
    assert_eq!(row("5090", 3).3, row("5090", 4).3);
    // 5090 + 2 Sparks: the two onboarded layers leave no 256K pool.
    let refused = plan::plan(dir.path(), &with_layout(RTX_5090, 2)).unwrap();
    assert!(!refused.fits && refused.hints.iter().any(|h| h.what.contains("below one 256K request")), "{:?}", refused.hints);
}

#[test]
fn vision_uses_encoder_owner_and_reserves_weights_and_scratch() {
    use crate::plan::{encoder::EncoderKind, MediaMode};
    let dir = snapshot(true, false);
    for (mode, expected) in [(MediaMode::Rtx(Some(1)), EncoderKind::Rtx { gpu: 1 }),
        (MediaMode::Spark(Some(2)), EncoderKind::Spark { rank: 2 }), (MediaMode::Off, EncoderKind::Off)] {
        let mut options = with_layout(PRO_TOTAL_BYTES, 4);
        options.layout.as_mut().unwrap().rtx_bytes = vec![PRO_TOTAL_BYTES; 2];
        options.vision = mode;
        let report = plan::plan(dir.path(), &options).unwrap();
        let encoder = report.encoder.as_ref().unwrap();
        assert_eq!(encoder.kind, expected);
        let items: Vec<_> = report.memory_layout.as_ref().unwrap().devices.iter().flat_map(|d| &d.items)
            .filter(|i| i.group.starts_with("vision")).collect();
        if expected == EncoderKind::Off { assert!(items.is_empty()); assert_eq!(encoder.admitted_bytes(), 0); }
        else {
            assert_eq!(items.len(), 2);
            assert_eq!(encoder.scratch, 762_383_616);
            assert!(encoder.weights > report.components.iter().find(|c| c.component == Component::Vision).unwrap().bytes);
            assert_eq!(items.iter().map(|i| i.bytes).sum::<u64>(), encoder.admitted_bytes());
        }
    }
}
