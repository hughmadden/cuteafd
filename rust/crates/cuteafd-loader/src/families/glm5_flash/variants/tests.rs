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

/// One shard: (name, dtype, shape, bytes) in file order.
fn write_shard(path: &std::path::Path, tensors: &[(String, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, dtype, shape, bytes) in tensors {
        header.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes.len()]}));
        offset += bytes.len();
    }
    let text = serde_json::Value::Object(header).to_string();
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend(text.as_bytes());
    for (_, _, _, bytes) in tensors { out.extend(bytes); }
    std::fs::write(path, out).unwrap();
}

/// A GLM Flash snapshot directory: config, index and shards. The index also lists a routed-expert
/// shard that is not on disk, so a derivation that opened anything but the site tensors' shards fails.
fn write_snapshot(dir: &std::path::Path, shards: &[(&str, Vec<(String, &str, Vec<usize>, Vec<u8>)>)]) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::json!({"model_type": "glm5_next", "text_config": {
        "model_type": "glm5_next_text", "hidden_size": 128, "num_hidden_layers": 2, "first_k_dense_replace": 0}}).to_string()).unwrap();
    let mut map = serde_json::Map::new();
    for (file, tensors) in shards {
        write_shard(&dir.join(file), tensors);
        for (name, ..) in tensors { map.insert(name.clone(), (*file).into()); }
    }
    map.insert(format!("{PREFIX}1.mlp.experts.0.down_proj.weight"), "model-00009-of-00009.safetensors".into());
    std::fs::write(dir.join("model.safetensors.index.json"), serde_json::json!({"weight_map": map}).to_string()).unwrap();
}

/// The shipped path: `source=` and the base are published snapshot directories, read through their
/// own index and shard headers (only the shards holding site tensors are opened).
#[test]
fn snapshot_directories_derive_through_their_own_index() {
    use crate::variants::{e4m3_to_f32, f32_to_e4m3};
    let (h, k) = (128usize, 256usize);
    let value = |i: usize, seed: usize| ((i * 2654435761 + seed * 40503) % 2001) as f32 / 1000.0 - 1.0;
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let w0: Vec<f32> = (0..h * k).map(|i| value(i, 1)).collect();
    let w1: Vec<f32> = (0..h * k).map(|i| value(i, 2)).collect();
    // L1 o_proj: a rank-1 output-side projection, alpha 1.
    let r: Vec<f32> = { let r: Vec<f32> = (0..h).map(|i| value(i, 3)).collect(); let n = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        r.into_iter().map(|x| x / n).collect() };
    let mut w1v = w1.clone();
    for c in 0..k {
        let t: f32 = (0..h).map(|o| r[o] * w1[o * k + c]).sum();
        for o in 0..h { w1v[o * k + c] -= r[o] * t; }
    }
    // L1 shared down: FP8 with a 1x2 grid of 128x128 F32 scales; the variant rewrites it (full rank).
    let codes = |seed: usize| (0..h * k).map(|i| f32_to_e4m3(value(i, seed) * 400.0)).collect::<Vec<u8>>();
    let scale = f32s(&[0.002, 0.003]);
    let name = |l: usize, t: &str| format!("{PREFIX}{l}.{t}.weight");
    let base = |shared: Vec<u8>, attn1: &[f32]| vec![
        ("model-00001-of-00009.safetensors", vec![(name(0, "self_attn.o_proj"), "F32", vec![h, k], f32s(&w0)),
            (name(1, "self_attn.o_proj"), "F32", vec![h, k], f32s(attn1))]),
        ("model-00002-of-00009.safetensors", vec![(name(1, "mlp.shared_experts.down_proj"), "F8_E4M3", vec![h, k], shared)]),
        // The scale lives in another shard than its weight: found through the index.
        ("model-00003-of-00009.safetensors", vec![(format!("{}_scale_inv", name(1, "mlp.shared_experts.down_proj")), "F32",
            vec![1, 2], scale.clone()), (name(2, "self_attn.o_proj"), "F32", vec![h, k], f32s(&w0))])];
    let dir = tempfile::tempdir().unwrap();
    let (served, variant) = (dir.path().join("base"), dir.path().join("variant"));
    write_snapshot(&served, &base(codes(4), &w1));
    let mut edited = base(codes(5), &w1v);
    // Layer 2 is the MTP layer (num_hidden_layers 2): its edit is never read.
    edited[2].1[1].3 = f32s(&w1);
    write_snapshot(&variant, &edited);
    let spec = VariantSpec { name: "v".into(), source: variant.display().to_string(), base: None, id: None, mode: None };
    let outcome = crate::variants::derive_variant(&served, "test/glmf", 1, &spec, 2).unwrap();
    assert_eq!(outcome.rejection, None);
    let plan = outcome.plan.expect("available");
    let keys: Vec<_> = plan.sites.iter().map(|(key, op)| (key.layer, key.kind, op.is_delta())).collect();
    assert_eq!(keys, [(1, SiteKind::Attn, false), (1, SiteKind::SharedExpert, true)]);
    assert_eq!(plan.sites[0].1.rank(), 1);
    assert_eq!(outcome.verdicts.len(), 3, "L0 o_proj (unchanged), L1 o_proj, L1 shared down; never L2");
    assert!(outcome.verdicts.iter().any(|v| v.tensor == name(0, "self_attn.o_proj") && v.report.is_none()));
    // The delta is the dequantized difference, scales read through the index.
    let SiteOp::Delta { weight, .. } = &plan.sites[1].1 else { unreachable!() };
    let (b, v) = (codes(4), codes(5));
    for i in [0, 1, k - 1, k, h * k - 1] {
        let s = [0.002f32, 0.003][(i % k) / 128];
        assert_eq!(weight[i], cuteafd_core::variants::bf16_rne(e4m3_to_f32(v[i]) * s - e4m3_to_f32(b[i]) * s), "element {i}");
    }
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
