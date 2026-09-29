use super::*;
use serde_json::json;
use std::fs;
use std::io::Write;

/// Writes a safetensors file whose payload is zero bytes of the right length.
fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>)]) {
    let width = |dtype: &str| match dtype {
        "BF16" | "F16" | "I16" => 2,
        "F32" | "I32" => 4,
        "I64" => 8,
        _ => 1,
    };
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, dtype, shape) in tensors {
        let bytes = shape.iter().product::<usize>() as u64 * width(dtype);
        header.insert(
            (*name).into(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        offset += bytes;
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&vec![0u8; offset as usize]).unwrap();
}

fn snapshot(config: serde_json::Value, tensors: &[(&str, &str, Vec<usize>)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    write_safetensors(&dir.path().join("model.safetensors"), tensors);
    dir
}

fn v41_config() -> serde_json::Value {
    json!({
        "architectures": ["DeepseekV41ForCausalLM"],
        "text_config": {
            "hidden_size": 64, "vocab_size": 128, "num_hidden_layers": 2,
            "n_routed_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 32,
            "n_shared_experts": 1, "compress_ratios": [0, 2], "scoring_func": "sqrtsoftplus",
            "engram_layer_ids": [1]
        }
    })
}

#[test]
fn deepseek_v41_components_formats_and_placement() {
    let dir = snapshot(
        v41_config(),
        &[
            ("embed.weight", "BF16", vec![128, 64]),
            ("head.weight", "BF16", vec![128, 64]),
            ("norm.weight", "BF16", vec![64]),
            ("layers.0.attn.wq_a.weight", "F8_E4M3", vec![64, 64]),
            ("layers.0.attn.wq_a.scale", "F8_E8M0", vec![2, 2]),
            ("layers.0.attn.attn_sink", "F32", vec![4]),
            ("layers.1.attn.indexer.wq_b.weight", "F8_E4M3", vec![64, 64]),
            ("layers.1.attn.indexer.wq_b.scale", "F8_E8M0", vec![2, 2]),
            ("layers.0.ffn.gate.weight", "BF16", vec![2, 64]),
            ("layers.0.ffn.experts.0.w1.weight", "I8", vec![32, 32]),
            ("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", vec![32, 2]),
            ("layers.1.engram.embed.weight", "F8_E4M3", vec![16, 64]),
            ("layers.1.engram.embed.scale", "F8_E8M0", vec![16, 2]),
            ("mtp.0.ffn.experts.0.w1.weight", "I8", vec![32, 32]),
            ("mtp.0.ffn.experts.0.w1.scale", "F8_E8M0", vec![32, 2]),
        ],
    );
    let report = plan(dir.path(), &PlanOptions::default()).unwrap();
    assert_eq!(report.family.as_deref(), Some("deepseek_v41"));
    assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
    let find = |component| report.components.iter().find(|c| c.component == component).unwrap();
    assert_eq!(find(Component::RoutedExpert).owner, Owner::SparkSliced);
    assert!(find(Component::RoutedExpert).formats.contains_key("mxfp4-g32"));
    assert!(find(Component::Attention).formats.contains_key("fp8-block32x32"));
    assert_eq!(find(Component::MappedTable).owner, Owner::HostMapped);
    assert!(find(Component::MappedTable).formats.contains_key("fp8-block1x32"));
    assert_eq!(find(Component::SpeculatorExpert).owner, Owner::Rtx);
    assert!(report.executable(), "{}", render(&report));
}

#[test]
fn v41_rejects_an_unsupported_expert_format_with_a_hint() {
    let dir = snapshot(
        v41_config(),
        &[
            ("embed.weight", "BF16", vec![128, 64]),
            ("layers.0.ffn.experts.0.w1.weight", "BF16", vec![32, 64]),
        ],
    );
    let report = plan(dir.path(), &PlanOptions::default()).unwrap();
    let experts = report.components.iter().find(|c| c.component == Component::RoutedExpert).unwrap();
    assert_eq!(experts.status, Status::MissingKernel);
    assert!(!report.executable());
    assert!(report.hints.iter().any(|hint| hint.what.contains("routed_expert")));
}

#[test]
fn unknown_architecture_is_reported_not_fatal() {
    let dir = snapshot(json!({"architectures": ["SomethingNew"], "model_type": "new"}),
        &[("w", "BF16", vec![2])]);
    let report = plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.family.is_none());
    assert!(report.hints[0].what.contains("SomethingNew"));
}

#[test]
fn exl3_bits_and_partial_fp8_blocks_are_detected() {
    let dir = snapshot(
        json!({"architectures": ["GlmMoeDsaForCausalLM"], "hidden_size": 64, "vocab_size": 8,
               "num_hidden_layers": 1, "n_routed_experts": 1, "num_experts_per_tok": 1,
               "moe_intermediate_size": 32}),
        &[
            ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "F8_E4M3", vec![576, 256]),
            ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv", "F32", vec![5, 2]),
            ("model.layers.0.mlp.experts.0.gate_proj.trellis", "I16", vec![4, 2, 48]),
            ("model.layers.0.mlp.experts.0.gate_proj.suh", "F16", vec![64]),
            ("model.layers.0.mlp.experts.0.gate_proj.svh", "F16", vec![32]),
            ("model.layers.0.mlp.experts.0.gate_proj.mcg", "I32", vec![1]),
        ],
    );
    let report = plan(dir.path(), &PlanOptions::default()).unwrap();
    let find = |component| report.components.iter().find(|c| c.component == component).unwrap();
    assert!(find(Component::Attention).formats.contains_key("fp8-block128x128"));
    assert_eq!(find(Component::RoutedExpert).formats.get("exl3-k3"), Some(&1));
}

#[test]
fn mimo_pro_segmented_fp8_grid_and_mxfp4_experts_are_described() {
    let dir = snapshot(
        json!({"architectures": ["MiMoV2ForCausalLM"], "model_type": "mimo_v2", "hidden_size": 256,
               "vocab_size": 8, "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
               "head_dim": 192, "v_head_dim": 128, "hybrid_layer_pattern": [0], "moe_layer_freq": [1],
               "n_routed_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 64}),
        &[
            // Two row shards of [q (2 x 192) | k (1 x 192) | v (1 x 128)]: 3 + 2 + 1 blocks each.
            ("model.layers.0.self_attn.qkv_proj.weight", "F8_E4M3", vec![1408, 256]),
            ("model.layers.0.self_attn.qkv_proj.weight_scale_inv", "F32", vec![12, 2]),
            ("model.layers.0.mlp.experts.0.gate_proj.weight", "U8", vec![64, 128]),
            ("model.layers.0.mlp.experts.0.gate_proj.weight_scale", "U8", vec![64, 8]),
        ],
    );
    let report = plan(dir.path(), &PlanOptions::default()).unwrap();
    let find = |component| report.components.iter().find(|c| c.component == component).unwrap();
    assert!(find(Component::Attention).formats.contains_key("fp8-block128x128-segmented"));
    assert!(find(Component::RoutedExpert).formats.contains_key("mxfp4-g32"));
    assert!(report.hints.iter().any(|h| h.how.contains("mimop:fp8")), "{:?}", report.hints);
}

#[test]
fn capacity_suggests_a_supported_rank_count() {
    let dir = snapshot(v41_config(), &[("layers.0.ffn.experts.0.w1.weight", "I8", vec![1024, 1024])]);
    let report = plan(dir.path(), &PlanOptions { spark_ranks: 4, spark_budget_bytes: 200 << 10 }).unwrap();
    // 1 MiB of experts at 200 KiB per rank needs 6 ranks (5 is not a TP size).
    assert_eq!(report.min_spark_ranks, 6);
    assert!(!report.fits);
}

#[test]
fn glm_next_facts_and_dflash2_drafter_are_described() {
    let dir = snapshot(
        json!({"architectures": ["Glm5NextForConditionalGeneration"], "text_config": {
            "hidden_size": 64, "vocab_size": 8, "num_hidden_layers": 2, "n_routed_experts": 2,
            "num_experts_per_tok": 1, "moe_intermediate_size": 32, "kv_lora_rank": 512, "qk_rope_head_dim": 0,
            "qk_nope_head_dim": 256, "v_head_dim": 256, "index_topk": 2048, "index_kpool": 4,
            "index_kpool_always_select_tail": true, "hc_mult": 4, "hc_sinkhorn_iters": 20, "swiglu_limit": 10.0,
            "layer_types": ["linear_attention", "deepseek_sparse_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "linear_attn_config": {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4,
                                   "gate_lower_bound": -5.0}}}),
        &[("model.language_model.layers.0.self_attn.A_log", "F32", vec![64])],
    );
    let report = plan(dir.path(), &PlanOptions::default()).unwrap();
    let notes = report.spec.as_ref().unwrap().notes.join("\n");
    assert!(notes.contains("record 528 B"), "{notes}");
    assert!(notes.contains("dense causal up to 2051 tokens"), "{notes}");
    assert!(notes.contains("recurrent state 4.0 MiB"), "{notes}");
    assert!(notes.contains("SwiGLU clamp 10"), "{notes}");

    let draft = snapshot(
        json!({"architectures": ["DFlash2DraftModel"], "hidden_size": 64, "vocab_size": 8,
               "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16,
               "sliding_window": 2048, "intermediate_size": 128, "num_target_layers": 45,
               "dflash_config": {"block_size": 8, "target_layer_ids": [5, 14]}}),
        &[("fc.weight", "BF16", vec![64, 128])],
    );
    let report = plan(draft.path(), &PlanOptions::default()).unwrap();
    assert_eq!(report.family.as_deref(), Some("dflash2"));
    assert!(report.spec.unwrap().notes[0].contains("taps [5, 14] of a 45-layer target"));
}
