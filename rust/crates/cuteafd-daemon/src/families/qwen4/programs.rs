//! Checkpoint/AOT geometry agreement before loading Qwen modules or allocating state.
use anyhow::{bail, ensure, Result};
use cuteafd_ffi::programs::{ProgramSpec, Programs};
use cuteafd_loader::families::qwen4::Qwen4Config;
use serde_json::{json, Value};

pub(super) fn validate_all(cfg: &Qwen4Config, programs: &Programs<'_>, context: usize) -> Result<()> {
    cfg.check_programs()?;
    for name in programs.names().filter(|name| name.starts_with("qwen4_")) {
        validate(cfg, context, programs.spec(name)?)?;
    }
    Ok(())
}

fn expected(cfg: &Qwen4Config, name: &str) -> Result<Value> {
    let op = name.strip_prefix("qwen4_").unwrap_or(name);
    let (stem, rows) = match op.rsplit_once("_m") {
        Some((stem, rows)) => (stem, Some(rows.parse::<usize>()?)),
        None => (op, None),
    };
    let kv_fp8 = stem.contains("_kv_fp8");
    let stem = stem.trim_end_matches("_kv_fp8");
    let (stem, weights) = if let Some(stem) = stem.strip_suffix("_w8") { (stem, json!("only")) }
        else if let Some(stem) = stem.strip_suffix("_fp8").filter(|_| rows.is_some()) { (stem, json!(true)) }
        else { (stem, json!(false)) };
    let replay = crate::shared::qwen::GdnLayout::new(cfg.gdn_geometry())?.replay_layer_bytes;
    let state_rows = (cfg.ple_conv - 1) * cfg.ngram_size;
    let width = cfg.dense_context().next_multiple_of(64);
    let mut geometry = match stem {
        "hc_pre" | "hc_post_pre" | "hc_post" | "head" => json!({"hidden":cfg.hidden,
            "streams":cfg.hc_count,"lowrank":cfg.hc_lowrank,"eps":cfg.rms_norm_eps}),
        "head_fp8" => json!({"hidden":cfg.hidden,"vocab":cfg.vocab_size,"max_rows":64}),
        "mtp_feedback" => json!({"hidden":cfg.hidden,"streams":cfg.hc_count,"eps":cfg.rms_norm_eps}),
        "router_scores" => json!({"hidden":cfg.hidden,"experts":cfg.experts,"top_k":cfg.topk}),
        "shared" => json!({"hidden":cfg.hidden,"inter":cfg.shared_intermediate,"rows":2 * cfg.shared_intermediate + 16}),
        "add" => json!({"hidden":cfg.hidden}),
        "expert_input_quant" => json!({"hidden":cfg.hidden,"row_bytes":cfg.hidden + cfg.hidden / 32,
            "expected_m":80,"amax_floor":1e-4}),
        "ple_bf16" | "ple_fp8" => json!({"hidden":cfg.hidden,"streams":cfg.hc_count,
            "rows_per_token":cfg.ple_rows(),"row_dim":cfg.ple_dim / cfg.ple_rows(),"taps":cfg.ple_conv,
            "dilation":cfg.ngram_size,"state_rows":state_rows,"fp8":stem == "ple_fp8"}),
        "ple_commit" => json!({"channels":cfg.hidden * cfg.hc_count,"state_rows":state_rows,"replay_rows":64}),
        "gdn_commit" => json!({"value_heads":cfg.gdn_value_heads,"key_heads":cfg.gdn_key_heads,
            "channels":cfg.gdn_conv_width(),"replay_rows":64,"replay_bytes":replay}),
        "gdn" => json!({"hidden":cfg.hidden,"key_heads":cfg.gdn_key_heads,"value_heads":cfg.gdn_value_heads,
            "head_dim":cfg.gdn_head_dim,"in_width":cfg.gdn_conv_width() + cfg.gdn_value_width() + 2 * cfg.gdn_value_heads,
            "eps":cfg.rms_norm_eps,"fp8_weights":weights}),
        "attn_producer" => json!({"hidden":cfg.hidden,"heads":cfg.heads,"kv_heads":cfg.kv_heads,
            "head_dim":cfg.head_dim,"index_heads":cfg.index_heads,"rope_dim":cfg.rope_dim,
            "rope_theta":cfg.rope_theta,"mrope_section":[11,11,10],"page_rows":64,"eps":cfg.rms_norm_eps,
            "fp8_weights":weights}),
        "attn_o" => json!({"hidden":cfg.hidden,"width":cfg.heads * cfg.head_dim,"fp8_weights":weights}),
        "index_topk" => json!({"index_heads":cfg.index_heads,"index_head_dim":cfg.index_head_dim,
            "blocks":cfg.index_budget / cfg.index_block,"block":cfg.index_block}),
        "index_expand" => json!({"blocks":cfg.index_budget / cfg.index_block,"width":width,"dense_limit":cfg.dense_context()}),
        "sparse_gqa" => json!({"heads":cfg.heads,"kv_heads":cfg.kv_heads,"head_dim":cfg.head_dim,
            "width":width,"softmax_scale":(cfg.head_dim as f64).sqrt().recip()}),
        _ => bail!("{name}: no Qwen geometry contract"),
    };
    if let Some(rows) = rows {
        ensure!([64, 4096].contains(&rows), "{name}: unsupported Qwen row capacity {rows}");
        geometry["max_rows"] = json!(rows);
        if weights == "only" || stem == "sparse_gqa" {
            geometry[if stem == "sparse_gqa" { "route" } else { "mode" }] = json!(if rows == 64 { "decode" } else { "prefill" });
        }
        if stem == "gdn" {
            geometry["replay_rows"] = if rows == 64 { json!(64) } else { Value::Null };
            geometry["replay_bytes"] = if rows == 64 { json!(replay) } else { Value::Null };
        }
    }
    if matches!(stem, "attn_producer" | "sparse_gqa") {
        geometry["kv_format"] = json!(if kv_fp8 { "fp8" } else { "bf16" });
        geometry["record_bytes"] = json!(2 * cfg.kv_heads * if kv_fp8 { cfg.head_dim + 4 } else { cfg.head_dim * 2 });
    }
    Ok(geometry)
}

fn validate(cfg: &Qwen4Config, context: usize, spec: &ProgramSpec) -> Result<()> {
    let expected = expected(cfg, &spec.name)?;
    for (key, value) in expected.as_object().expect("geometry object") {
        let actual = spec.geometry.get(key);
        // JSON preserves integer vs float representation; numerical fields need value equality.
        let equal = actual.is_some_and(|actual| actual == value ||
            actual.as_f64().zip(value.as_f64()).is_some_and(|(a, b)| a == b));
        ensure!(equal, "{}: geometry.{key} must be {value}, got {actual:?}", spec.name);
    }
    if spec.name.rsplit_once("_m").is_some_and(|(_, rows)| rows.parse::<usize>().is_ok()) {
        let rows = expected["max_rows"].as_u64().unwrap();
        ensure!(u64::from(spec.capacity_rows) == rows, "{}: manifest row capacity disagrees with geometry", spec.name);
    }
    if spec.name.starts_with("qwen4_index_topk_") {
        ensure!(context > 0 && spec.geometry["max_context"].as_u64().is_some_and(|n| n >= context as u64),
            "{}: compiled index context does not cover {context}", spec.name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_exporter_attention_geometry_is_accepted_independently() -> Result<()> {
        let cfg = Qwen4Config::from_hf(&cuteafd_loader::plan::testing::qwen4_config(48))?;
        // Literal fields from compile_qwen4_attn_producer_aot, not expected().
        let mut spec = ProgramSpec { name: "qwen4_attn_producer_w8_kv_fp8_m64".into(),
            pointers: vec![], scalar_kinds: vec![], scratch: Default::default(), capacity_rows: 64,
            geometry: json!({"hidden":2560,"heads":24,"kv_heads":2,"head_dim":256,
                "index_heads":4,"rope_dim":64,"rope_theta":10000000,"mrope_section":[11,11,10],
                "record_bytes":1040,"kv_format":"fp8","page_rows":64,"eps":1e-6,
                "max_rows":64,"fp8_weights":"only","mode":"decode"}) };
        validate(&cfg, 262144, &spec)?;
        spec.geometry["record_bytes"] = json!(2048);
        assert!(validate(&cfg, 262144, &spec).is_err());
        spec.geometry["record_bytes"] = json!(1040);
        spec.geometry.as_object_mut().unwrap().remove("mrope_section");
        assert!(validate(&cfg, 262144, &spec).is_err());
        Ok(())
    }

    #[test]
    fn qwen_geometry_rejects_stale_fields_and_missing_context() -> Result<()> {
        let cfg = Qwen4Config::from_hf(&cuteafd_loader::plan::testing::qwen4_config(48))?;
        for op in ["hc_pre", "hc_post_pre", "hc_post", "head", "head_fp8", "mtp_feedback", "router_scores",
            "shared", "add", "expert_input_quant", "ple_bf16", "ple_fp8", "ple_commit", "gdn_commit",
            "index_expand", "gdn_m64", "gdn_m4096", "gdn_w8_m64", "gdn_w8_m4096", "gdn_fp8_m64",
            "attn_producer_m64", "attn_producer_kv_fp8_m4096", "attn_producer_w8_kv_fp8_m64",
            "attn_o_fp8_m64", "attn_o_w8_m4096", "index_topk_m64", "index_topk_m4096",
            "sparse_gqa_m64", "sparse_gqa_kv_fp8_m4096"] {
            let name = format!("qwen4_{op}");
            let mut spec = ProgramSpec { name, pointers: vec![], scalar_kinds: vec![], scratch: Default::default(),
                capacity_rows: 4096, geometry: Value::Null };
            spec.geometry = expected(&cfg, &spec.name)?;
            if op.contains("_m") { spec.capacity_rows = spec.geometry["max_rows"].as_u64().unwrap() as u32; }
            if op.starts_with("index_topk") { spec.geometry["max_context"] = json!(1_048_576); }
            validate(&cfg, 262144, &spec)?;
            for key in spec.geometry.as_object().unwrap().keys().cloned().collect::<Vec<_>>() {
                let saved = spec.geometry[&key].clone();
                spec.geometry[&key] = json!("stale");
                assert!(validate(&cfg, 262144, &spec).is_err(), "{op}.{key}");
                spec.geometry[&key] = saved;
            }
            if op.contains("_m") {
                spec.capacity_rows = 1;
                assert!(validate(&cfg, 262144, &spec).is_err(), "{op}: wrong capacity");
            }
            if op.starts_with("index_topk") {
                spec.capacity_rows = spec.geometry["max_rows"].as_u64().unwrap() as u32;
                spec.geometry["max_context"] = json!(131072);
                assert!(validate(&cfg, 262144, &spec).is_err());
            }
        }
        Ok(())
    }
}
