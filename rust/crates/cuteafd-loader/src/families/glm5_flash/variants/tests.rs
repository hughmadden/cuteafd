use super::*;
use cuteafd_core::variants::{SiteOp, VariantSpec};
use std::path::PathBuf;

fn header_only(names: &[(&str, &str)]) -> (tempfile::TempDir, TensorSource) {
    let dir = tempfile::tempdir().unwrap();
    let mut header = serde_json::Map::new();
    for (name, dtype) in names {
        header.insert((*name).into(), serde_json::json!({"dtype": dtype, "shape": [1, 1], "data_offsets": [0, 2]}));
    }
    let text = serde_json::Value::Object(header).to_string();
    let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
    bytes.extend(text.as_bytes());
    bytes.extend([0, 0]);
    std::fs::write(dir.path().join("v.safetensors"), bytes).unwrap();
    let source = TensorSource::from_files([dir.path().join("v.safetensors")]).unwrap();
    (dir, source)
}

#[test]
fn sites_follow_layer_kind_and_skip_the_mtp_layer() {
    let (_dir, source) = header_only(&[
        ("model.language_model.layers.2.mlp.down_proj.weight", "F8_E4M3"),
        ("model.language_model.layers.12.self_attn.o_proj.weight", "BF16"),
        ("model.language_model.layers.15.self_attn.o_proj.weight", "F8_E4M3"),
        ("model.language_model.layers.20.mlp.shared_experts.down_proj.weight", "F8_E4M3"),
        ("model.language_model.layers.20.mlp.experts.3.down_proj.weight", "F8_E4M3"),
        ("model.language_model.layers.45.self_attn.o_proj.weight", "F8_E4M3"),
        ("model.language_model.layers.45.mlp.shared_experts.down_proj.weight", "F8_E4M3"),
    ]);
    let sites = variant_sites(45, 3, &source);
    let keys: Vec<_> = sites.iter().map(|s| (s.key.layer, s.key.kind)).collect();
    assert_eq!(keys, [(2, SiteKind::DenseFfn), (12, SiteKind::Attn), (15, SiteKind::Attn), (20, SiteKind::SharedExpert)],
        "routed experts are never sites; layer 45 (MTP) is never run");
    assert!(sites.iter().all(|s| s.opts.delta_fallback));
    assert_eq!(sites[1].opts.max_r_delta, Some(0.25), "BF16 bars");
    assert_eq!(sites[2].opts.max_r_w, Some(0.022), "FP8 bars");
    assert!(wants_tensor("model.language_model.layers.9.self_attn.o_proj.weight_scale_inv"));
    assert!(wants_tensor("model.language_model.layers.30.mlp.shared_experts.down_proj.weight"));
    assert!(!wants_tensor("model.language_model.layers.30.mlp.experts.0.down_proj.weight"));
    assert!(!wants_tensor("model.layers.30.self_attn.o_proj.weight"));
}

#[test]
fn family_map_resolves_glm_flash_configs() {
    let config = serde_json::json!({"model_type": "glm5_next", "text_config": {"model_type": "glm5_next_text",
        "hidden_size": 4096, "num_hidden_layers": 45, "first_k_dense_replace": 3}});
    let family = crate::variants::family_variants(&config, None).unwrap();
    assert_eq!((family.family, family.hidden), ("glm5_flash", 4096));
    assert!(crate::variants::family_variants(&config, Some("ffn")).is_err(), "GLM Flash has no modes");
}

/// A local checkpoint or evidence folder named by `env`; the test skips without it.
fn local(env: &str) -> Option<PathBuf> {
    let path = std::env::var_os(env).map(PathBuf::from).filter(|p| p.exists());
    if path.is_none() { eprintln!("skipping: set {env} to a local copy"); }
    path
}

/// The whole example-org variant through the production entry point, against the
/// pilot (`glmf-alternate-rank.json`): 56 rank-1 sites (L12–43), the two L44
/// tensors as deltas, the MTP layer's edits never read, every other site unchanged.
#[test]
#[ignore = "needs CUTEAFD_GLMF_BASE, CUTEAFD_GLMF_VARIANT and CUTEAFD_VARIANT_EVIDENCE; --release, ~30 s"]
fn alternate_variant_reproduces_the_pilot_verdicts() {
    let Some(base) = local("CUTEAFD_GLMF_BASE") else { return };
    let Some(variant) = local("CUTEAFD_GLMF_VARIANT") else { return };
    let Some(evidence) = local("CUTEAFD_VARIANT_EVIDENCE") else { return };
    let pilot: serde_json::Value = serde_json::from_slice(&std::fs::read(evidence.join("glmf-alternate-rank.json")).unwrap()).unwrap();
    let spec = VariantSpec { name: "alternate".into(), source: variant.display().to_string(), base: None, id: None, mode: None };
    let parallel = std::thread::available_parallelism().map_or(4, |n| (n.get() / 8).clamp(1, 4));
    let outcome = crate::variants::derive_variant(&base, "zai-org/GLM-5.3-Flash", 1, &spec, parallel).unwrap();
    for v in &outcome.verdicts {
        if let Some(r) = &v.report {
            eprintln!("{:<18} k{} a{:.3} top{:.4} next{:.5} cos{:.4} rD{:.3} rW{:.5} {} {}", v.site, r.k,
                r.alphas.first().copied().unwrap_or(0.0), r.top_share, r.next_share, r.proj_cos, r.r_delta, r.r_w,
                if v.delta { "DELTA" } else if r.accept { "ok" } else { "REJECT" }, r.reason.as_deref().unwrap_or(""));
        }
    }
    assert_eq!(outcome.rejection, None);
    let plan = outcome.plan.as_ref().expect("variant available");
    let steer: Vec<_> = plan.sites.iter().filter(|(_, op)| !op.is_delta()).collect();
    let delta: Vec<_> = plan.sites.iter().filter(|(_, op)| op.is_delta()).map(|(k, _)| *k).collect();
    assert_eq!(steer.len(), 56, "56 rank-1 sites");
    assert!(steer.iter().all(|(k, op)| (12..=43).contains(&k.layer) && op.rank() == 1), "rank 1 in L12-43");
    assert_eq!(steer.iter().filter(|(k, _)| k.kind == SiteKind::Attn).count(), 32);
    assert_eq!(delta, [SiteKey { layer: 44, kind: SiteKind::Attn }, SiteKey { layer: 44, kind: SiteKind::SharedExpert }]);
    assert!(plan.sites.iter().all(|(k, _)| k.layer < 45), "MTP edits ignored");
    assert_eq!(outcome.verdicts.iter().filter(|v| v.report.is_none()).count(), outcome.verdicts.len() - 58, "the rest unchanged");
    assert_eq!(plan.delta_values(), 4096 * 8192 + 4096 * 2048);
    assert_eq!(plan.table_bytes(), 56 * 2 * 4096 * 4);
    // α per tensor against the pilot.
    for v in outcome.verdicts.iter().filter(|v| !v.delta) {
        let Some(r) = &v.report else { continue };
        let row = pilot["rows"].as_array().unwrap().iter().find(|row| row["tensor"] == v.tensor.as_str()).unwrap();
        let alpha = row["alpha"].as_f64().unwrap() as f32;
        assert!((r.alphas[0] - alpha).abs() < 0.02 * alpha.max(1.0), "{}: alpha {} vs pilot {alpha}", v.tensor, r.alphas[0]);
    }
    if let SiteOp::Delta { rows, cols, .. } = plan.site(SiteKey { layer: 44, kind: SiteKind::Attn }).unwrap() {
        assert_eq!((*rows, *cols), (4096, 8192));
    }
}
