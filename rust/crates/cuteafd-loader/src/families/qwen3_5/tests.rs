use super::*;
use crate::plan::{self, testing::{t, write_snapshot, Tensor}, ExpertPlacement, PlanOptions, RuntimeStatus};
use crate::families::qwen::{GdnOutputGate, HybridAttention};
use serde_json::{json, Value};

fn config(hidden: usize, intermediate: usize, heads: usize, values: usize, layers: usize) -> Value {
    json!({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": {
        "model_type": "qwen3_5_text", "hidden_size": hidden, "intermediate_size": intermediate,
        "num_hidden_layers": layers, "vocab_size": 64, "num_attention_heads": heads, "num_key_value_heads": 4,
        "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": values,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "layer_types": (0..layers).map(|l| if l % 4 == 3 { "full_attention" } else { "linear_attention" }).collect::<Vec<_>>(),
        "full_attention_interval": 4, "hidden_act": "silu", "attention_bias": false, "attn_output_gate": true,
        "output_gate_type": "swish", "mamba_ssm_dtype": "float32", "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144, "mtp_num_hidden_layers": 0, "tie_word_embeddings": false,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10], "mrope_interleaved": true}}})
}

fn tensors(fp8: bool) -> Vec<Tensor> {
    let mut out = vec![t("model.language_model.embed_tokens.weight", "BF16", &[64, 4096]),
        t("lm_head.weight", "BF16", &[64, 4096]), t("model.language_model.norm.weight", "BF16", &[4096])];
    for layer in 0..4 {
        let p = format!("model.language_model.layers.{layer}");
        for name in ["input_layernorm", "post_attention_layernorm"] {
            out.push(t(format!("{p}.{name}.weight"), "BF16", &[4096]));
        }
        let mut projection = |name: &str, n: usize, k: usize| {
            out.push(t(format!("{p}.{name}.weight"), if fp8 { "F8_E4M3" } else { "BF16" }, &[n, k]));
            if fp8 { out.push(t(format!("{p}.{name}.weight_scale_inv"), "BF16", &[n.div_ceil(128), k.div_ceil(128)])); }
        };
        projection("mlp.gate_proj", 12288, 4096);
        projection("mlp.up_proj", 12288, 4096);
        projection("mlp.down_proj", 4096, 12288);
        if layer == 3 {
            projection("self_attn.q_proj", 8192, 4096);
            projection("self_attn.k_proj", 1024, 4096);
            projection("self_attn.v_proj", 1024, 4096);
            projection("self_attn.o_proj", 4096, 4096);
            for name in ["q_norm", "k_norm"] { out.push(t(format!("{p}.self_attn.{name}.weight"), "BF16", &[256])); }
        } else {
            projection("linear_attn.in_proj_qkv", 8192, 4096);
            projection("linear_attn.in_proj_z", 4096, 4096);
            projection("linear_attn.out_proj", 4096, 4096);
            for name in ["in_proj_a", "in_proj_b"] { out.push(t(format!("{p}.linear_attn.{name}.weight"), "BF16", &[32, 4096])); }
            out.push(t(format!("{p}.linear_attn.conv1d.weight"), "BF16", &[8192, 1, 4]));
            out.push(t(format!("{p}.linear_attn.A_log"), "F32", &[32]));
            out.push(t(format!("{p}.linear_attn.dt_bias"), "BF16", &[32]));
            out.push(t(format!("{p}.linear_attn.norm.weight"), "F32", &[128]));
        }
    }
    out
}

#[test]
fn dense_geometry_and_native_gate_sites() {
    for (h, i, heads, values, layers, kv, state) in [(4096, 12288, 16, 32, 32, 32768, 51511296),
        (5120, 17408, 24, 48, 64, 65536, 153944064)] {
        let cfg = Qwen35Config::from_hf(&config(h, i, heads, values, layers)).unwrap();
        assert_eq!(cfg.gdn.output_gate, GdnOutputGate::Silu);
        assert_eq!(cfg.full_kv_bytes_per_token(), kv);
        assert_eq!(cfg.recurrent_state_bytes(), state);
        assert_eq!(cfg.attention[3], HybridAttention::Full);
    }
}

#[test]
fn malformed_config_is_not_a_flash_next_alias() {
    let base = config(4096, 12288, 16, 32, 4);
    for (field, value) in [("output_gate_type", json!("sigmoid")), ("hidden_act", json!("gelu")),
        ("num_key_value_heads", json!(3)), ("linear_conv_kernel_dim", json!(0)),
        ("linear_value_head_dim", json!(64)), ("attention_bias", json!(true)),
        ("mamba_ssm_dtype", json!("bfloat16")), ("layer_types", json!(["full_attention"])),
        ("mtp_num_hidden_layers", json!(-1)), ("mtp_num_hidden_layers", json!("1")),
        ("attn_output_gate", json!("true")), ("attention_dropout", json!("zero")),
        ("output_gate_type", json!(false)), ("mamba_ssm_dtype", json!(32)),
        ("tie_word_embeddings", json!("false")), ("mtp_use_dedicated_embeddings", json!(null))] {
        let mut bad = base.clone(); bad["text_config"][field] = value;
        assert!(Qwen35Config::from_hf(&bad).is_err(), "{field}");
    }
}

#[test]
fn bf16_and_native_block_fp8_headers_remain_planned_and_zero_spark() {
    for fp8 in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &config(4096, 12288, 16, 32, 4), &tensors(fp8), None);
        Qwen35Checkpoint::read(dir.path()).unwrap();
        let report = plan::plan(dir.path(), &PlanOptions { placement: ExpertPlacement::Local, ..Default::default() }).unwrap();
        assert_eq!(report.family.as_deref(), Some("qwen3_5"));
        assert_eq!(report.runtime, Some(RuntimeStatus::Planned));
        assert!(!report.fits && !report.placement_supported && !report.executable());
        assert_eq!(report.spark_ranks, 0);
        assert!(report.experts.is_none() && report.spec.as_ref().unwrap().moe.is_none());
        assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
        assert!(report.components.iter().all(|c| c.rejected == 0), "{:?}", report.components);
    }
}

#[test]
fn missing_or_malformed_projection_is_named() {
    let cfg = config(4096, 12288, 16, 32, 4);
    let dir = tempfile::tempdir().unwrap();
    let mut weights = tensors(false);
    let missing = "model.language_model.layers.0.mlp.up_proj.weight";
    weights.retain(|(name, ..)| name != missing);
    write_snapshot(dir.path(), &cfg, &weights, None);
    assert!(format!("{:#}", Qwen35Checkpoint::read(dir.path()).err().unwrap()).contains(missing));
    let mut weights = tensors(false);
    weights.iter_mut().find(|(name, ..)| name.ends_with("self_attn.q_proj.weight")).unwrap().2 = vec![4096, 4096];
    write_snapshot(dir.path(), &cfg, &weights, None);
    assert!(format!("{:#}", Qwen35Checkpoint::read(dir.path()).err().unwrap()).contains("q_proj.weight"));
    let mut weights = tensors(true);
    weights.iter_mut().find(|(name, ..)| name.ends_with("linear_attn.in_proj_a.weight")).unwrap().1 = "F8_E4M3".into();
    write_snapshot(dir.path(), &cfg, &weights, None);
    assert!(format!("{:#}", Qwen35Checkpoint::read(dir.path()).err().unwrap()).contains("in_proj_a"));
}
