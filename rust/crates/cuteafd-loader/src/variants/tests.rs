use super::*;
use cuteafd_core::variants::SiteKind;

/// Deterministic pseudo-random values in roughly [-1, 1].
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn matrix(&mut self, rows: usize, cols: usize) -> Matrix { Matrix::new(rows, cols, (0..rows * cols).map(|_| self.next()).collect()) }
    fn unit(&mut self, n: usize) -> Vec<f32> { normalize((0..n).map(|_| self.next()).collect()) }
}
fn normalize(v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / n).collect()
}
/// k orthonormal H-vectors.
fn orthonormal(rng: &mut Rng, h: usize, k: usize) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = Vec::new();
    while out.len() < k {
        let mut v: Vec<f64> = (0..h).map(|_| f64::from(rng.next())).collect();
        for _ in 0..2 {
            for u in &out {
                let d: f64 = u.iter().zip(&v).map(|(a, b)| f64::from(*a) * b).sum();
                v.iter_mut().zip(u).for_each(|(b, a)| *b -= d * f64::from(*a));
            }
        }
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        out.push(v.into_iter().map(|x| (x / n) as f32).collect());
    }
    out
}
/// W' = (I − Σ α_j u_j u_jᵀ) W.
fn project(w: &Matrix, dirs: &[Vec<f32>], alphas: &[f32]) -> Matrix {
    let op = SiteOp::Steer { k: dirs.len() as u8,
        a: dirs.iter().zip(alphas).flat_map(|(u, a)| u.iter().map(move |x| x * a)).collect(),
        b: dirs.iter().flatten().copied().collect() };
    rebuild(w, &op)
}
fn max_abs_diff(a: &Matrix, b: &Matrix) -> f32 { a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }

#[test]
fn rank1_projection_is_recovered() {
    let mut rng = Rng(7);
    let w = rng.matrix(256, 384);
    let r = rng.unit(256);
    let variant = project(&w, &[r.clone()], &[3.0]);
    let (op, report) = derive_site(&w, &variant, &DeriveOpts::default()).unwrap();
    assert!(report.accept, "{report:?}");
    assert_eq!(report.k, 1);
    assert!((report.alphas[0] - 3.0).abs() < 1e-3, "{report:?}");
    assert!(report.top_share > 0.9999 && report.proj_cos > 0.9999 && report.r_delta < 1e-3, "{report:?}");
    let SiteOp::Steer { a, b, .. } = &op else { unreachable!() };
    let cos: f32 = b.iter().zip(&r).map(|(x, y)| x * y).sum();
    assert!(cos.abs() > 0.9999, "direction {cos}");
    assert!(a.iter().zip(b).all(|(a, b)| (a - 3.0 * b).abs() < 1e-3), "a = alpha r");
    assert!(max_abs_diff(&rebuild(&w, &op), &variant) < 1e-3);
}

#[test]
fn rank_19_and_32_projections_are_recovered() {
    for k in [19usize, 32] {
        let mut rng = Rng(11 + k as u64);
        let w = rng.matrix(192, 320);
        let dirs = orthonormal(&mut rng, 192, k);
        let alphas: Vec<f32> = (0..k).map(|j| 1.0 + j as f32 / (2 * k) as f32).collect();
        let variant = project(&w, &dirs, &alphas);
        let (op, report) = derive_site(&w, &variant, &DeriveOpts::default()).unwrap();
        assert!(report.accept, "k={k}: {report:?}");
        assert_eq!(usize::from(report.k), k, "{report:?}");
        assert!(report.r_delta < 1e-3 && report.proj_cos > 0.9999, "k={k}: {report:?}");
        assert!(max_abs_diff(&rebuild(&w, &op), &variant) < 2e-3, "k={k}");
    }
}

#[test]
fn rank_40_edit_is_rejected() {
    let mut rng = Rng(40);
    let w = rng.matrix(192, 320);
    let dirs = orthonormal(&mut rng, 192, 40);
    let variant = project(&w, &dirs, &[1.0; 40]);
    let (_, report) = derive_site(&w, &variant, &DeriveOpts::default()).unwrap();
    assert!(!report.accept, "{report:?}");
    assert_eq!(report.k, 32);
    assert!(report.reason.as_deref().unwrap().contains("rank above 32"), "{report:?}");
}

#[test]
fn norm_rescaled_edit_is_rejected() {
    // Norm-preserving abliteration: project, then rescale every output row back to its old norm.
    let mut rng = Rng(5);
    let (h, k) = (128, 256);
    let r = rng.unit(h);
    // A model whose rows carry the edited direction strongly (as a refusal direction does).
    let g: Vec<f32> = (0..k).map(|_| rng.next()).collect();
    let noise = rng.matrix(h, k);
    let w = Matrix::new(h, k, (0..h * k).map(|i| 3.0 * r[i / k] * h as f32 / 8.0 * g[i % k] + noise.data[i]).collect());
    let projected = project(&w, &[r.clone()], &[1.0]);
    let mut variant = projected.clone();
    for row in 0..h {
        let before = w.row(row).iter().map(|x| x * x).sum::<f32>().sqrt();
        let after = projected.row(row).iter().map(|x| x * x).sum::<f32>().sqrt();
        variant.data[row * k..][..k].iter_mut().for_each(|x| *x *= before / after);
    }
    let (_, report) = derive_site(&w, &variant, &DeriveOpts::default()).unwrap();
    assert!(!report.accept, "{report:?}");
}

#[test]
fn input_side_edit_is_rejected() {
    // W' = W (I − α c cᵀ): rank 1, but on the input side, so no output-side operator matches.
    let mut rng = Rng(9);
    let w = rng.matrix(160, 256);
    let c = rng.unit(256);
    let wc: Vec<f32> = (0..160).map(|r| w.row(r).iter().zip(&c).map(|(a, b)| a * b).sum()).collect();
    let variant = Matrix::new(160, 256, (0..160 * 256).map(|i| w.data[i] - 3.0 * wc[i / 256] * c[i % 256]).collect());
    let (_, report) = derive_site(&w, &variant, &DeriveOpts::default()).unwrap();
    assert_eq!(report.k, 1, "{report:?}");
    assert!(!report.accept, "{report:?}");
    assert!(report.reason.as_deref().unwrap().contains("output-side"), "{report:?}");
}

#[test]
fn unchanged_and_mismatched_tensors_are_errors() {
    let mut rng = Rng(3);
    let w = rng.matrix(64, 96);
    assert_eq!(derive_site(&w, &w, &DeriveOpts::default()).unwrap_err(), VariantError::Unchanged);
    let mut noisy = w.clone();
    noisy.data[5] += 1e-4;
    let floor = DeriveOpts { edit_floor: 0.001, ..DeriveOpts::default() };
    assert_eq!(derive_site(&w, &noisy, &floor).unwrap_err(), VariantError::Unchanged);
    assert!(matches!(derive_site(&w, &rng.matrix(64, 95), &DeriveOpts::default()), Err(VariantError::Derive(_))));
}

#[test]
fn fp8_base_with_bf16_variant_fits_under_the_fp8_floor() {
    // GLM o_proj: the base is FP8 (128-blocks, F32 scales), the published variant is the BF16
    // master with the edit. The leftover is the FP8 conversion error, at or below the floor.
    let mut rng = Rng(21);
    let (h, k) = (256, 384);
    let master = rng.matrix(h, k);
    let (scales, fp8) = quantize_fp8_blocks(&master, 128);
    let base = dequantize_fp8_blocks(&fp8, &scales, h, k, 128);
    let floor = (master.data.iter().zip(&base.data).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>()
        / base.frobenius2()).sqrt() as f32;
    let r = rng.unit(h);
    let edited = project(&master, &[r], &[3.0]);
    let variant = Matrix::new(h, k, edited.data.iter().map(|&v| f32::from_bits(u32::from(cuteafd_core::variants::bf16_rne(v)) << 16)).collect());
    let opts = DeriveOpts { k_max: 4, max_r_delta: None, max_r_w: Some(floor * 1.25 + 0.004), min_proj_cos: 0.999, ..DeriveOpts::default() };
    let (_, report) = derive_site(&base, &variant, &opts).unwrap();
    assert!(report.accept, "floor {floor}: {report:?}");
    assert!((report.alphas[0] - 3.0).abs() < 0.02, "{report:?}");
}

fn quantize_fp8_blocks(m: &Matrix, block: usize) -> (Vec<f32>, Vec<u8>) {
    let (sr, sc) = (m.rows.div_ceil(block), m.cols.div_ceil(block));
    let mut scales = vec![0f32; sr * sc];
    for r in 0..m.rows { for c in 0..m.cols {
        let s = &mut scales[(r / block) * sc + c / block];
        *s = s.max(m.data[r * m.cols + c].abs() / 448.0);
    } }
    let bytes = (0..m.rows * m.cols).map(|i| f32_to_e4m3(m.data[i] / scales[(i / m.cols / block) * sc + (i % m.cols) / block])).collect();
    (scales, bytes)
}
fn dequantize_fp8_blocks(bytes: &[u8], scales: &[f32], rows: usize, cols: usize, block: usize) -> Matrix {
    let sc = cols.div_ceil(block);
    Matrix::new(rows, cols, (0..rows * cols).map(|i| e4m3_to_f32(bytes[i]) * scales[(i / cols / block) * sc + (i % cols) / block]).collect())
}

#[test]
fn fp8_and_half_decoders() {
    for code in 0u8..=255 {
        let v = e4m3_to_f32(code);
        if v.is_nan() { assert!(code & 0x7f == 0x7f); continue; }
        if code == 0x80 { continue; } // -0 encodes back as -0
        assert_eq!(f32_to_e4m3(v), code, "code {code:#x} = {v}");
    }
    assert_eq!(e4m3_to_f32(0x7e), 448.0);
    assert_eq!(f32_to_e4m3(1000.0), 0x7e);
    assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
    assert_eq!(ue8m0_to_f32(127), 1.0);
    assert_eq!(ue8m0_to_f32(130), 8.0);
    assert_eq!(f16_to_f32(0x3c00), 1.0);
    assert_eq!(f16_to_f32(0xc000), -2.0);
    assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
}

/// Write a safetensors file from (name, dtype, shape, bytes).
fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, dtype, shape, bytes) in tensors {
        header.insert(name.to_string(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes.len()]}));
        offset += bytes.len();
    }
    let text = serde_json::Value::Object(header).to_string();
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend(text.as_bytes());
    for (_, _, _, bytes) in tensors { out.extend(bytes); }
    std::fs::write(path, out).unwrap();
}

#[test]
fn tensor_source_decodes_fp8_blocks_bf16_and_f16() {
    let dir = tempfile::tempdir().unwrap();
    let (rows, cols) = (4usize, 6usize);
    // FP8 with 2x3 blocks of F32 scale_inv (GLM layout) and with UE8M0 `.scale` (V4.1 layout).
    let codes: Vec<u8> = (0..rows * cols).map(|i| f32_to_e4m3(i as f32 * 0.25 - 2.0)).collect();
    let f32_scale: Vec<u8> = [1.0f32, 2.0, 0.5, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let e8_scale = vec![127u8, 128, 126, 129];
    let bf16: Vec<u8> = (0..rows * cols).flat_map(|i| cuteafd_core::variants::bf16_rne(i as f32).to_le_bytes()).collect();
    let f16: Vec<u8> = (0..rows * cols).flat_map(|_| 0x3c00u16.to_le_bytes()).collect();
    write_safetensors(&dir.path().join("a.safetensors"), &[
        ("g.weight", "F8_E4M3", vec![rows, cols], codes.clone()),
        ("g.weight_scale_inv", "F32", vec![2, 2], f32_scale),
        ("d.weight", "F8_E4M3", vec![rows, cols], codes.clone()),
        ("d.scale", "F8_E8M0", vec![2, 2], e8_scale),
        ("b.weight", "BF16", vec![rows, cols], bf16),
        ("h.weight", "F16", vec![rows, cols], f16),
    ]);
    let source = TensorSource::from_files(safetensors_files(dir.path()).unwrap()).unwrap();
    let block = |r: usize, c: usize| (r / 2) * 2 + c / 3;
    let g = source.matrix("g.weight").unwrap();
    let d = source.matrix("d.weight").unwrap();
    for r in 0..rows { for c in 0..cols {
        let raw = e4m3_to_f32(codes[r * cols + c]);
        assert_eq!(g.data[r * cols + c], raw * [1.0, 2.0, 0.5, 4.0][block(r, c)]);
        assert_eq!(d.data[r * cols + c], raw * [1.0, 2.0, 0.5, 4.0][block(r, c)]);
    } }
    assert_eq!(source.matrix("b.weight").unwrap().data[7], 7.0);
    assert!(source.matrix("h.weight").unwrap().data.iter().all(|&v| v == 1.0));
}

#[test]
fn derive_sites_skips_unchanged_and_reports_rejections() {
    let dir = tempfile::tempdir().unwrap();
    let mut rng = Rng(77);
    let (h, k) = (64usize, 96usize);
    let w = rng.matrix(h, k);
    let edited = project(&w, &[rng.unit(h)], &[2.0]);
    let input_side = { let c = rng.unit(k); let mut m = w.clone();
        for r in 0..h { let t: f32 = w.row(r).iter().zip(&c).map(|(a, b)| a * b).sum();
            for j in 0..k { m.data[r * k + j] -= 2.0 * t * c[j]; } } m };
    let bytes = |m: &Matrix| m.data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    write_safetensors(&dir.path().join("base.safetensors"), &[("l1.weight", "F32", vec![h, k], bytes(&w)),
        ("l2.weight", "F32", vec![h, k], bytes(&w)), ("l3.weight", "F32", vec![h, k], bytes(&w))]);
    write_safetensors(&dir.path().join("variant.safetensors"), &[("l1.weight", "F32", vec![h, k], bytes(&edited)),
        ("l2.weight", "F32", vec![h, k], bytes(&w)), ("l3.weight", "F32", vec![h, k], bytes(&input_side))]);
    let base = TensorSource::from_files([dir.path().join("base.safetensors")]).unwrap();
    let variant = TensorSource::from_files([dir.path().join("variant.safetensors")]).unwrap();
    let site = |layer: u16, name: &str| SiteTensor { key: SiteKey { layer, kind: SiteKind::Attn }, tensor: name.into(), opts: DeriveOpts::default() };
    let (ops, verdicts) = derive_sites(&base, &variant, &[site(1, "l1.weight"), site(2, "l2.weight")], 2).unwrap();
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].0.layer, 1);
    assert!(verdicts[1].report.is_none(), "unchanged tensor yields no site");
    assert!(first_rejection(&verdicts).is_none());
    let (_, verdicts) = derive_sites(&base, &variant, &[site(1, "l1.weight"), site(3, "l3.weight")], 2).unwrap();
    let rejection = first_rejection(&verdicts).unwrap();
    assert!(rejection.starts_with("l3.weight (L3.attn)"), "{rejection}");
}

// ---- pilot fixtures (local tensors on pg; run with --release -- --ignored) --------------------

fn fixture_dir(env: &str, default: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(env).unwrap_or_else(|_| default.into()));
    if path.exists() { Some(path) } else { eprintln!("skipping: {} is absent", path.display()); None }
}
fn evidence(file: &str) -> Option<serde_json::Value> {
    let dir = fixture_dir("CUTEAFD_VARIANT_EVIDENCE", "/home/turq/dev/recipes/cuteafd-upstream/evidence/2026-10-24-weight-variant-verify")?;
    Some(serde_json::from_slice(&std::fs::read(dir.join(file)).unwrap()).unwrap())
}

/// GLM-5.3 full: official FP8 dense tensors against the Keys pack's FP16 tensors
/// (`glm53-keys-exl3-ablit.json`): α 3.0, down_proj leftover at FP16 rounding,
/// o_proj leftover at the BF16-versus-FP8 floor, layer 50 unedited.
#[test]
#[ignore = "needs the GLM-5.3 FP8 dense tensors and the Keys pack (pg)"]
fn glm53_keys_pack_reproduces_the_pilot() {
    let Some(base_dir) = fixture_dir("CUTEAFD_GLM53_FP8_DENSE", "/home/turq/.cache/weight-variants/glm53-fp8-dense") else { return };
    let Some(pack) = fixture_dir("CUTEAFD_GLM53_KEYS", "/home/turq/models/drowzeys/keys-GLM-5.3-EXL3-Abliterated") else { return };
    let Some(pilot) = evidence("glm53-keys-exl3-ablit.json") else { return };
    let base = TensorSource::from_files(safetensors_files(&base_dir).unwrap()).unwrap();
    let names = ["model.layers.2.mlp.down_proj.weight", "model.layers.2.self_attn.o_proj.weight",
        "model.layers.3.mlp.shared_experts.down_proj.weight", "model.layers.3.self_attn.o_proj.weight",
        "model.layers.49.self_attn.o_proj.weight", "model.layers.50.self_attn.o_proj.weight"];
    let wanted = |n: &str| names.contains(&n);
    let checkpoint = crate::plan::checkpoint::Checkpoint::open_for_role(&pack, "variant", wanted).unwrap();
    let variant = TensorSource::from_checkpoint(&checkpoint);
    for name in names {
        let opts = crate::families::glm5::variants::variant_opts(name);
        let w = base.matrix(name).unwrap();
        let v = variant.matrix(name).unwrap();
        let result = derive_site(&w, &v, &opts);
        let row = pilot["rows"].as_array().unwrap().iter().find(|r| r["tensor"] == name).unwrap();
        if name.contains(".50.") {
            assert!(row["rel_dW"].as_f64().unwrap() < 0.001, "pilot control");
            assert_eq!(result.unwrap_err(), VariantError::Unchanged, "{name}: the unedited control");
            continue;
        }
        let (_, report) = result.unwrap();
        eprintln!("{name}: {report:?}");
        assert!(report.accept, "{name}: {report:?}");
        let f = |key: &str| row[key].as_f64().unwrap() as f32;
        assert!((report.alphas[0] - f("alpha")).abs() < 0.01, "{name}: alpha {} vs pilot {}", report.alphas[0], f("alpha"));
        assert!((report.top_share - row["energy_top4"][0].as_f64().unwrap() as f32).abs() < 0.01, "{name}");
        assert!((report.r_w / f("r_W") - 1.0).abs() < 0.15, "{name}: r_W {} vs pilot {}", report.r_w, f("r_W"));
        assert!(report.proj_cos >= f("cos_uD_uW") - 1e-3, "{name}");
    }
}

/// GLM-5.3 Flash: official FP8 against dealignai's UNCENSORED-FP8
/// (`glmf-dealign-rank.json`): BF16 L12 o_proj is rank 1; L44 o_proj is not low rank.
#[test]
#[ignore = "needs zai-org/GLM-5.3-Flash and dealignai/GLM-5.3-Flash-UNCENSORED-FP8 (pg)"]
fn glm_flash_dealign_reproduces_the_pilot() {
    let Some(base_dir) = fixture_dir("CUTEAFD_GLMF_BASE", "/home/turq/models/zai-org/GLM-5.3-Flash") else { return };
    let Some(variant_dir) = fixture_dir("CUTEAFD_GLMF_DEALIGN", "/home/turq/models/dealignai/GLM-5.3-Flash-UNCENSORED-FP8") else { return };
    let Some(pilot) = evidence("glmf-dealign-rank.json") else { return };
    let names = ["model.language_model.layers.12.self_attn.o_proj.weight", "model.language_model.layers.44.self_attn.o_proj.weight"];
    let open = |dir: &Path| {
        let names = names.map(|n| n.to_string());
        let wanted = move |n: &str| names.iter().any(|name| n == name || n == format!("{name}_scale_inv"));
        TensorSource::from_checkpoint(&crate::plan::checkpoint::Checkpoint::open_for_role(dir, "variant", wanted).unwrap())
    };
    let (base, variant) = (open(&base_dir), open(&variant_dir));
    // BF16 bars from the contract: next share ≤ 0.01, leftover ≤ 0.2·‖ΔW‖.
    let opts = DeriveOpts::default();
    for name in names {
        let row = pilot["rows"].as_array().unwrap().iter().find(|r| r["tensor"] == name).unwrap();
        let (_, report) = derive_site(&base.matrix(name).unwrap(), &variant.matrix(name).unwrap(), &opts).unwrap();
        eprintln!("{name}: {report:?}");
        if name.contains(".44.") {
            assert!(!report.accept, "L44 is not a low-rank edit: {report:?}");
        } else {
            assert!(report.accept, "{report:?}");
            assert_eq!(report.k, 1);
            assert!((report.alphas[0] - row["alpha"].as_f64().unwrap() as f32).abs() < 0.01, "{report:?}");
            assert!((report.r_delta / row["r_D"].as_f64().unwrap() as f32 - 1.0).abs() < 0.15, "{report:?}");
        }
    }
}

/// The CPU form of the hardware component test: base FP8 layer output, steered by the derived
/// tables through the kernel's host reference (BF16 in and out), against the published
/// variant's own output on the same activations. Error sits at the format floor; base rows
/// stay bit-identical.
#[test]
#[ignore = "needs the GLM-5.3 FP8 dense tensors and the Keys pack (pg)"]
fn glm53_steered_rows_match_the_published_variant() {
    use cuteafd_core::variants::{bf16_rne, bf16_to_f32, site_tables, steer_rows_ref, VariantPlan};
    let Some(base_dir) = fixture_dir("CUTEAFD_GLM53_FP8_DENSE", "/home/turq/.cache/weight-variants/glm53-fp8-dense") else { return };
    let Some(pack) = fixture_dir("CUTEAFD_GLM53_KEYS", "/home/turq/models/drowzeys/keys-GLM-5.3-EXL3-Abliterated") else { return };
    let base = TensorSource::from_files(safetensors_files(&base_dir).unwrap()).unwrap();
    // Bars: the BF16 rounding of the row itself (measured per row below) plus the tensor's leftover
    // (FP16 rounding for down_proj; the BF16-master-versus-FP8-base floor for o_proj).
    for (name, kind, leftover) in [("model.layers.3.mlp.shared_experts.down_proj.weight", SiteKind::Ffn, 0.001f64),
        ("model.layers.3.self_attn.o_proj.weight", SiteKind::Attn, 0.035)] {
        let wanted = |n: &str| n == name;
        let variant = TensorSource::from_checkpoint(&crate::plan::checkpoint::Checkpoint::open_for_role(&pack, "variant", wanted).unwrap());
        let (w, v) = (base.matrix(name).unwrap(), variant.matrix(name).unwrap());
        let (op, report) = derive_site(&w, &v, &crate::families::glm5::variants::variant_opts(name)).unwrap();
        assert!(report.accept, "{report:?}");
        let key = SiteKey { layer: 3, kind };
        let plan = VariantPlan::new(1, "abliterated", "glm-abliterated", w.rows as u32, vec![(key, op)]).unwrap();
        let (a, b, k, k_max) = site_tables(&[plan], key, w.rows);
        // Eight activation rows (row 0 and 5 are base rows), y = W x and y_published = W' x.
        let mut rng = Rng(31);
        let rows = 8;
        let xs: Vec<Vec<f32>> = (0..rows).map(|_| (0..w.cols).map(|_| rng.next()).collect()).collect();
        let gemv = |m: &Matrix, x: &[f32]| -> Vec<f32> { (0..m.rows).map(|r| m.row(r).iter().zip(x).map(|(a, b)| a * b).sum()).collect() };
        let mut y: Vec<u16> = xs.iter().flat_map(|x| gemv(&w, x)).map(bf16_rne).collect();
        let before = y.clone();
        let vidx: Vec<i32> = (0..rows as i32).map(|r| if r == 0 || r == 5 { 0 } else { 1 }).collect();
        steer_rows_ref(&mut y, w.rows, rows, w.rows, &vidx, &a, &b, &k, k_max);
        for r in 0..rows {
            let row = &y[r * w.rows..][..w.rows];
            if vidx[r] == 0 { assert_eq!(row, &before[r * w.rows..][..w.rows], "{name}: base row {r} changed"); continue; }
            let published = gemv(&v, &xs[r]);
            let (mut err, mut norm, mut floor) = (0f64, 0f64, 0f64);
            for (s, p) in row.iter().zip(&published) {
                err += (f64::from(bf16_to_f32(*s)) - f64::from(*p)).powi(2);
                floor += (f64::from(bf16_to_f32(bf16_rne(*p))) - f64::from(*p)).powi(2);
                norm += f64::from(*p).powi(2);
            }
            let (rel, floor) = ((err / norm).sqrt(), (floor / norm).sqrt());
            eprintln!("{name} row {r}: relative error {rel:.5} (BF16 floor {floor:.5})");
            assert!(rel <= 1.5 * floor + leftover, "{name} row {r}: {rel} against floor {floor}");
        }
    }
}

/// The exhaustive nearest search the binary search replaced (ties to the even code).
fn e4m3_exhaustive(value: f32) -> u8 {
    if value.is_nan() { return 0x7f; }
    let sign = if value.is_sign_negative() { 0x80u8 } else { 0 };
    let x = value.abs().min(448.0);
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0u8..0x7f {
        let err = (e4m3_to_f32(code) - x).abs();
        if err < best_err || (err == best_err && code & 1 == 0 && best & 1 == 1) { best = code; best_err = err; }
    }
    sign | best
}

#[test]
fn fast_encoder_matches_the_exhaustive_search() {
    for code in 0u8..=255 {
        let v = e4m3_to_f32(code);
        if v.is_nan() { continue; }
        // Each code, the midpoints to both neighbours, and just past them.
        for x in [v, v * 1.03125, v * 0.96875, v * 1.0625, v + 2f32.powi(-10), v - 2f32.powi(-10), v * 1.5, -v * 1.07] {
            assert_eq!(f32_to_e4m3(x), e4m3_exhaustive(x), "{x}");
        }
    }
    for x in [1e9f32, -1e9, 448.0, 464.0, 0.0, -0.0, 2f32.powi(-10), 1.0625, 1.1875, f32::NAN] {
        assert_eq!(f32_to_e4m3(x), e4m3_exhaustive(x), "{x}");
    }
    let mut x = 1e-6f32;
    while x < 600.0 { assert_eq!(f32_to_e4m3(x), e4m3_exhaustive(x), "{x}"); x *= 1.000_7; }
}
