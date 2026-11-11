use super::*;
use crate::plan::{self, testing, PlanOptions};

fn fixture() -> (tempfile::TempDir, serde_json::Value, Vec<testing::Tensor>) {
    let dir = tempfile::tempdir().unwrap();
    let cfg = crate::families::minimax_m3::config::tests::release_config();
    // Use the checkpoint's real shape recipe without writing 89k sparse records per test.
    let mut small = cfg;
    small["text_config"]["num_hidden_layers"] = 4.into();
    small["text_config"]["num_local_experts"] = 4.into();
    for key in ["moe_layer_freq"] {
        small["text_config"][key].as_array_mut().unwrap().truncate(4);
    }
    for key in ["sparse_attention_freq", "sparse_disable_index_value"] {
        small["text_config"]["sparse_attention_config"][key].as_array_mut().unwrap().truncate(4);
    }
    // Construct headers from the independent Python reference's spelling and shapes.
    let c = MiniMaxM3Config::from_hf(&small).unwrap();
    let mut tensors = Vec::new();
    let mut plain = |name: String, dtype, shape: Vec<usize>| tensors.push((name, dtype, shape));
    plain(format!("{TEXT}embed_tokens.weight"), "BF16", vec![c.vocab_size, c.hidden]);
    plain(format!("{HEAD}weight"), "BF16", vec![c.vocab_size, c.hidden]);
    plain(format!("{TEXT}norm.weight"), "BF16", vec![c.hidden]);
    for layer in 0..4 {
        let p = format!("{TEXT}layers.{layer}.");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            plain(format!("{p}{norm}.weight"), "BF16", vec![c.hidden]);
        }
        let mut matrix = |stem: String, n, k| {
            plain(format!("{stem}.weight"), "F8_E4M3", vec![n, k]);
            plain(format!("{stem}.weight_scale_inv"), "U8", vec![n, k / 32]);
        };
        for (name, n, k) in [("q_proj", 8192, 6144), ("k_proj", 512, 6144), ("v_proj", 512, 6144),
            ("o_proj", 6144, 8192), ("index_q_proj", 512, 6144), ("index_k_proj", 128, 6144)] {
            matrix(format!("{p}self_attn.{name}"), n, k);
        }
        let (stem, inter) = if layer == 3 { ("block_sparse_moe.shared_experts", 3072) } else { ("mlp", 12288) };
        for (proj, n, k) in [("gate", inter, 6144), ("up", inter, 6144), ("down", 6144, inter)] {
            matrix(format!("{p}{stem}.{proj}_proj"), n, k);
        }
        for norm in ["q_norm", "k_norm", "index_q_norm", "index_k_norm"] {
            plain(format!("{p}self_attn.{norm}.weight"), "BF16", vec![128]);
        }
        if layer == 3 {
            plain(format!("{p}block_sparse_moe.gate.weight"), "F32", vec![4, 6144]);
            plain(format!("{p}block_sparse_moe.e_score_correction_bias"), "F32", vec![4]);
            for e in 0..4 {
                for (w, n, k) in [("w1", 3072, 6144), ("w3", 3072, 6144), ("w2", 6144, 3072)] {
                    let stem = format!("{p}block_sparse_moe.experts.{e}.{w}");
                    plain(format!("{stem}.weight_packed"), "U8", vec![n, k / 2]);
                    plain(format!("{stem}.weight_scale"), "F8_E4M3", vec![n, k / 16]);
                    for scale in ["weight_global_scale", "input_global_scale"] {
                        plain(format!("{stem}.{scale}"), "F32", vec![]);
                    }
                }
            }
        }
    }
    small.as_object_mut().unwrap().remove("vision_config");
    testing::write_snapshot(dir.path(), &small, &tensors, None);
    (dir, small, tensors)
}

#[test]
fn minimax_native_inventory_is_planned_not_executable() {
    let (dir, _, _) = fixture();
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    assert_eq!(report.family.as_deref(), Some("minimax_m3"));
    assert_eq!(report.runtime, Some(RuntimeStatus::Planned));
    assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
    assert!(report.components.iter().all(|c| c.rejected == 0), "{:?}", report.components);
    assert!(!report.executable());
    let spec = report.spec.unwrap();
    assert_eq!(spec.moe_layers(), 1);
    assert!(matches!(spec.layers[0].attention, AttentionKind::BlockSparseGqa { block: 128, topk_blocks: 16, .. }));
    assert_eq!(spec.layers[0].rope.unwrap().dims, 64);
}

#[test]
fn generic_native_validation_rejects_missing_and_drifted_minimax_tensors_early() {
    let (dir, cfg, mut tensors) = fixture();
    let name = format!("{TEXT}layers.0.self_attn.index_k_proj.weight_scale_inv");
    tensors.iter_mut().find(|t| t.0 == name).unwrap().2[1] += 1;
    testing::write_snapshot(dir.path(), &cfg, &tensors, None);
    let diagnostic = match plan::plan(dir.path(), &PlanOptions::default()).unwrap_err() {
        plan::PlanError::Checkpoint(error) => error.to_string(),
        error => panic!("expected generic checkpoint validation, got {error:?}"),
    };
    assert!(diagnostic.contains(&name), "{diagnostic}");
    tensors.retain(|t| t.0 != name);
    testing::write_snapshot(dir.path(), &cfg, &tensors, None);
    let diagnostic = match plan::plan(dir.path(), &PlanOptions::default()).unwrap_err() {
        plan::PlanError::Checkpoint(error) => error.to_string(),
        error => panic!("expected generic checkpoint validation, got {error:?}"),
    };
    assert!(diagnostic.contains(&name), "{diagnostic}");
}

#[test]
#[ignore = "requires the installed vendor checkpoint"]
fn installed_minimax_headers_match_native_contract() {
    // Explicitly ignored: requires the installed checkpoint, but never reads payloads or a GPU.
    let path = std::path::Path::new("/mnt/sparknest/hf-home/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e");
    let checkpoint = Checkpoint::open(path).unwrap();
    let model = Model::open(&checkpoint).unwrap();
    assert_eq!(model.cfg.moe_layers(), 57);
    assert_eq!(checkpoint.tensors.len(), 89_632);
    assert!(checkpoint.spark_input_is_bf16());
    use crate::plan::checkpoint::{ModuleNumerics, ProjectionStorage};
    for module in [format!("{TEXT}layers.0.self_attn.q_proj"),
        format!("{TEXT}layers.59.self_attn.index_k_proj"),
        format!("{TEXT}layers.3.block_sparse_moe.shared_experts.down_proj")] {
        assert_eq!(checkpoint.module_numerics(&module).unwrap(), ModuleNumerics::W8A8, "{module}");
    }
    let expert = format!("{TEXT}layers.3.block_sparse_moe.experts.0.w1");
    assert_eq!(checkpoint.module_numerics(&expert).unwrap(), ModuleNumerics::W4A4);
    assert_eq!(checkpoint.projection(&expert).unwrap().storage,
        ProjectionStorage::CompressedNvfp4 { shape: vec![3072, 6144] });
    assert_eq!(checkpoint.module_numerics(&format!("{TEXT}layers.3.block_sparse_moe.gate")).unwrap(),
        ModuleNumerics::CheckpointPrecision);
}
