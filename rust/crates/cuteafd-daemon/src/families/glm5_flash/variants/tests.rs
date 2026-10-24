use super::*;
use cuteafd_core::variants::{add_rows_ref, bf16_rne, bf16_to_f32, steer_rows_ref, VariantSpec};

fn glm53_flash() -> GlmNextConfig {
    GlmNextConfig { vocab_size: 154880, hidden: 4096, layers: 45,
        attention: (0..45).map(|i| if i % 4 == 3 { GlmNextAttention::Mla } else { GlmNextAttention::Kda }).collect(),
        dense: (0..45).map(|i| i < 3).collect(), dense_intermediate: 12288, experts: 288, topk: 8,
        moe_intermediate: 2048, routed_scale: 2.5, swiglu_limit: 10.0, rms_norm_eps: 1e-5, hc_mult: 4,
        kda_heads: 64, kda_head_dim: 128, heads: 64, q_lora_rank: 1536, kv_lora_rank: 512, qk_nope_dim: 256,
        v_head_dim: 256, index_topk: 2048, index_kpool: 4, eos: vec![] }
}

fn key(layer: u16, kind: SiteKind) -> SiteKey { SiteKey { layer, kind } }
fn unit(h: usize, seed: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..h).map(|i| (((i * 7919 + seed * 104729) % 1000) as f32 / 500.0) - 1.0).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / n).collect()
}
fn delta(rows: usize, cols: usize, seed: usize) -> SiteOp {
    SiteOp::Delta { rows: rows as u32, cols: cols as u32,
        weight: (0..rows * cols).map(|i| bf16_rne(((i * 31 + seed) % 17) as f32 / 1700.0 - 0.005)).collect() }
}

/// dealignai's shape: 56 rank-1 sites (L12–43 o_proj, L20–43 shared down) and the two L44 deltas.
fn dealign_plan(h: usize) -> VariantPlan {
    let mut sites: Vec<(SiteKey, SiteOp)> = (12..=43).map(|l| (key(l, SiteKind::Attn), SiteOp::rank1(1.0, &unit(h, l as usize))))
        .chain((20..=43).map(|l| (key(l, SiteKind::SharedExpert), SiteOp::rank1(0.5, &unit(h, 100 + l as usize))))).collect();
    sites.push((key(44, SiteKind::Attn), delta(h, 8192, 1)));
    sites.push((key(44, SiteKind::SharedExpert), delta(h, 2048, 2)));
    VariantPlan::new(1, "uncensored", "glmf-uncensored", h as u32, sites).unwrap()
}

#[test]
fn admission_charges_tables_deltas_workspace_and_row_ids() {
    let h = 4096;
    let plan = dealign_plan(h);
    let bytes = device_bytes(&[plan.clone()], h, 64, &[64, 4096, 4096], |shapes| {
        assert_eq!(shapes, [(8192, h), (2048, h)], "one GEMV shape per delta site, (K, N)");
        Ok(1 << 20)
    }).unwrap();
    // Tables: [V+1 = 2][k = 1][h] FP32 for a and b, plus a 256-byte k table, at 56 sites.
    assert_eq!(bytes.tables, 56 * (2 * 2 * h as u64 * 4 + 256));
    // Deltas: E4M3 [h, K], per row x 128-K FP32 scales, the per-variant switch.
    assert_eq!(bytes.deltas, (h * 8192 + h * 64 * 4 + 256 + h * 2048 + h * 16 * 4 + 256) as u64);
    assert_eq!(bytes.workspace, (64 * h * 4 + (1 << 20) + PACK_ROWS * 8192 * 2) as u64);
    assert_eq!(bytes.rows, (256 + 2 * 4096 * 4) as u64);
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    assert!((3.4..3.6).contains(&mib(bytes.tables)) && (41.0..41.5).contains(&mib(bytes.deltas)), "{bytes:?}");
    // No variants: nothing charged.
    assert_eq!(device_bytes(&[], h, 64, &[64, 4096], |_| unreachable!()).unwrap().total(), 0);
    // A steering-only variant needs no delta workspace.
    let steer_only = VariantPlan::new(2, "s", "s", h as u32, vec![(key(12, SiteKind::Attn), SiteOp::rank1(1.0, &unit(h, 3)))]).unwrap();
    let bytes = device_bytes(&[steer_only], h, 64, &[64], |_| unreachable!()).unwrap();
    assert_eq!((bytes.deltas, bytes.workspace), (0, 0));
}

#[test]
fn sites_the_engine_cannot_serve_make_the_variant_unavailable() {
    let cfg = glm53_flash();
    let h = cfg.hidden;
    assert_eq!(unsupported(&dealign_plan(h), &cfg), None, "dealignai serves: L44 is a KDA layer");
    let one = |site: (SiteKey, SiteOp)| unsupported(&VariantPlan::new(1, "x", "x", h as u32, vec![site]).unwrap(), &cfg);
    assert!(one((key(43, SiteKind::Attn), delta(h, 8192, 3))).unwrap().contains("MLA"), "L43 is MLA");
    assert!(one((key(44, SiteKind::Attn), delta(h, 4096, 3))).unwrap().contains("width"));
    assert!(one((key(44, SiteKind::SharedExpert), delta(h, 12288, 3))).unwrap().contains("width"));
    assert_eq!(one((key(1, SiteKind::DenseFfn), delta(h, 12288, 3))), None, "a dense MLP delta");
    assert!(one((key(1, SiteKind::SharedExpert), SiteOp::rank1(1.0, &unit(h, 1)))).unwrap().contains("dense"));
    assert!(one((key(5, SiteKind::DenseFfn), SiteOp::rank1(1.0, &unit(h, 1)))).unwrap().contains("MoE"));
    assert!(one((key(5, SiteKind::Ffn), SiteOp::rank1(1.0, &unit(h, 1)))).unwrap().contains("combined"));
    assert!(one((key(45, SiteKind::Attn), SiteOp::rank1(1.0, &unit(h, 1)))).unwrap().contains("not run"), "MTP");
}

#[test]
fn scratch_offsets_follow_the_program_layouts() {
    // GLM Flash KDA: D = 64 x 128, in-projection rows 3D + 2·128 + 64.
    let (d, p) = (8192, 3 * 8192 + 256 + 64);
    // One row: 49 792 → 50 176 (1 KiB aligned), f|gate 32 768, q|k|v 49 152, recurrent 16 384.
    assert_eq!(kda_norm_offset(1, p, d), 50176 + 32768 + 49152 + 16384);
    // Every region of a 64-row step is already aligned.
    assert_eq!(kda_norm_offset(64, p, d), 64 * (p * 2 + 4 * d + 6 * d + 2 * d));
    assert_eq!(ffn_hidden_offset(1, 2048), 8192);
    assert_eq!(ffn_hidden_offset(3, 12288), 3 * 4 * 12288);
}

#[test]
fn only_decode_or_a_varied_prefill_launches() {
    assert!(launches(true, &[0, 0, 0]), "decode graphs keep one topology");
    assert!(!launches(false, &[0; 16]), "a base prefill chunk launches nothing");
    assert!(launches(false, &[0, 0, 1]));
    assert!(!launches(false, &[]));
}

/// The host composition of one site exactly as `variant_site` runs it: the delta GEMV over every row
/// (FP32 out) and the masked add, then steering.
fn host_site(plans: &[VariantPlan], key: SiteKey, y: &mut [u16], input: &[u16], rows: usize, vidx: &[i32]) {
    let h = plans[0].hidden as usize;
    if let Some((cols, weight)) = plans.iter().find_map(|p| match p.site(key) {
        Some(SiteOp::Delta { cols, weight, .. }) => Some((*cols as usize, weight)),
        _ => None,
    }) {
        let mut d = vec![0f32; rows * h];
        for r in 0..rows {
            for o in 0..h {
                d[r * h + o] = (0..cols).map(|c| bf16_to_f32(weight[o * cols + c]) * bf16_to_f32(input[r * cols + c])).sum();
            }
        }
        add_rows_ref(y, h, &d, h, rows, h, vidx, &delta_switch(plans, key));
    }
    let (a, b, k, k_max) = site_tables(plans, key, h);
    steer_rows_ref(y, h, rows, h, vidx, &a, &b, &k, k_max);
}

#[test]
fn mixed_batches_keep_row_order_and_base_rows_are_never_written() {
    let h = 128;
    let attn = key(44, SiteKind::Attn);
    let a = VariantPlan::new(1, "a", "m-a", h as u32, vec![(attn, delta(h, 256, 5))]).unwrap();
    let b = VariantPlan::new(2, "b", "m-b", h as u32, vec![(attn, SiteOp::rank1(1.3, &unit(h, 9)))]).unwrap();
    let plans = [a, b];
    let rows = 6;
    let y0: Vec<u16> = (0..rows * h).map(|i| if i == 0 { 0x8000 } else { bf16_rne(((i * 13) % 29) as f32 / 7.0 - 2.0) }).collect();
    let x: Vec<u16> = (0..rows * 256).map(|i| bf16_rne(((i * 5) % 23) as f32 / 11.0 - 1.0)).collect();
    let vidx = [0, 1, 2, 0, 2, 1];
    let mut y = y0.clone();
    host_site(&plans, attn, &mut y, &x, rows, &vidx);
    for r in 0..rows {
        // The same row alone, at the front of a one-row batch.
        let mut alone = y0[r * h..(r + 1) * h].to_vec();
        host_site(&plans, attn, &mut alone, &x[r * 256..(r + 1) * 256], 1, &vidx[r..r + 1]);
        assert_eq!(&y[r * h..(r + 1) * h], &alone[..], "row {r} (variant {}) depends on its batch", vidx[r]);
        if vidx[r] == 0 {
            assert_eq!(&y[r * h..(r + 1) * h], &y0[r * h..(r + 1) * h], "base row {r} written");
        } else {
            assert_ne!(&y[r * h..(r + 1) * h], &y0[r * h..(r + 1) * h], "variant row {r} unchanged");
        }
    }
    assert_eq!(y[0], 0x8000, "a base row's -0.0 survives");
    // An all-base batch is byte-identical, whatever is configured.
    let mut base = y0.clone();
    host_site(&plans, attn, &mut base, &x, rows, &[0; 6]);
    assert_eq!(base, y0);
}

// ---- component check on the real tensors (pg; run with --release -- --ignored --nocapture) --------

/// FP8 E4M3 of a row × 128-K block as the W8A16 pack's `Best` rule picks it: amax / 448 or the
/// power-of-two scale, whichever leaves the smaller squared error.
fn fp8_best_dequant(block: &[f32]) -> Vec<f32> {
    use super::super::fp8::{e4m3, e4m3_value, pow2_scale};
    let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 { return vec![0.0; block.len()]; }
    let deq = |s: f32| block.iter().map(|&v| e4m3_value(e4m3(v / s)) * s).collect::<Vec<_>>();
    let err = |q: &[f32]| q.iter().zip(block).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
    let (a, p) = (deq(amax / 448.0), deq(pow2_scale(amax)));
    if err(&p) < err(&a) { p } else { a }
}

/// `W + Q(ΔW)` against `W'` relative to ‖W‖, next to two FP8 floors (all with `Q` the device's
/// per row × 128-K `Best` quantization): the base's (`Q(W)` against `W`) and the variant's
/// (`Q(W')` against `W'`, what serving `W'` itself through the FP8 path would cost).
fn delta_rebuild(w: &[f32], wv: &[f32]) -> (f64, f64, f64) {
    let dw: Vec<f32> = wv.iter().zip(w).map(|(v, w)| bf16_to_f32(bf16_rne(v - w))).collect();
    let q: Vec<f32> = dw.chunks(128).flat_map(fp8_best_dequant).collect();
    let floor = |m: &[f32]| m.chunks(128).flat_map(fp8_best_dequant).zip(m).map(|(q, v)| f64::from(q - v).powi(2)).sum::<f64>();
    let norm = w.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    let rebuild = w.iter().zip(&q).zip(wv).map(|((w, q), v)| f64::from(w + q - v).powi(2)).sum::<f64>();
    ((rebuild / norm).sqrt(), (floor(w) / norm).sqrt(), (floor(wv) / norm).sqrt())
}

#[test]
fn fp8_packed_delta_rebuilds_the_variant_under_the_base_fp8_floor() {
    // A full-rank edit at 5% of ‖W‖ (dealignai L44's size), K a multiple of 128.
    let (rows, cols) = (64, 512);
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 2654435761usize) % 4001) as f32 / 2000.0 - 1.0).collect();
    let wv: Vec<f32> = w.iter().enumerate().map(|(i, v)| v + 0.05 * (((i * 40503) % 2001) as f32 / 1000.0 - 1.0)).collect();
    let (rebuild, floor, variant_floor) = delta_rebuild(&w, &wv);
    assert!(rebuild <= floor, "W + Q(ΔW) misses W' by {rebuild}, above the base's own FP8 floor {floor}");
    assert!(rebuild <= 0.2 * floor.min(variant_floor), "{rebuild} vs {floor}/{variant_floor}: ΔW's FP8 error scales with ‖ΔW‖");
}

/// Steered and delta sites against the variant's own weights on the same rows: `y` from the base
/// tensor through the site's operator (in the device formats: BF16 output, FP8 ΔW) versus `W' x`.
#[test]
#[ignore = "needs CUTEAFD_GLMF_BASE and CUTEAFD_GLMF_DEALIGN (zai-org/GLM-5.3-Flash, dealignai UNCENSORED-FP8); --release"]
fn dealign_sites_reproduce_the_variant_weights_on_the_host() {
    let (Some(base), Some(variant)) = (std::env::var_os("CUTEAFD_GLMF_BASE"), std::env::var_os("CUTEAFD_GLMF_DEALIGN")) else {
        eprintln!("skipping: set CUTEAFD_GLMF_BASE and CUTEAFD_GLMF_DEALIGN to local copies");
        return;
    };
    let (base, variant) = (std::path::Path::new(&base), std::path::Path::new(&variant));
    let spec = VariantSpec { name: "uncensored".into(), source: variant.display().to_string(), base: None, id: None, mode: None };
    let outcome = cuteafd_loader::variants::derive_variant(base, "zai-org/GLM-5.3-Flash", 1, &spec, 4).unwrap();
    let plan = outcome.plan.unwrap();
    let wants = cuteafd_loader::families::glm5_flash::variants::wants_tensor;
    let (b, v) = (cuteafd_loader::variants::open_source(base, "base", wants).unwrap(),
        cuteafd_loader::variants::open_source(variant, "variant", wants).unwrap());
    let rows = 8;
    for (layer, kind, name) in [(12u16, SiteKind::Attn, "self_attn.o_proj"), (23, SiteKind::Attn, "self_attn.o_proj"),
        (30, SiteKind::SharedExpert, "mlp.shared_experts.down_proj"), (44, SiteKind::Attn, "self_attn.o_proj"),
        (44, SiteKind::SharedExpert, "mlp.shared_experts.down_proj")] {
        let tensor = format!("{}{layer}.{name}.weight", cuteafd_loader::families::glm5_flash::variants::PREFIX);
        let (w, wv) = (b.matrix(&tensor).unwrap(), v.matrix(&tensor).unwrap());
        let (h, k) = (w.rows, w.cols);
        let x: Vec<u16> = (0..rows * k).map(|i| bf16_rne(((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)).collect();
        let gemv = |m: &[f32], r: usize| -> Vec<f32> { (0..h).map(|o| (0..k).map(|c| m[o * k + c] * bf16_to_f32(x[r * k + c])).sum()).collect() };
        let site = SiteKey { layer, kind };
        let op = plan.site(site).unwrap();
        // The device's ΔW: the BF16 delta quantized per row × 128-K block.
        let fp8_delta: Option<Vec<f32>> = match op {
            SiteOp::Delta { weight, .. } => Some(weight.chunks(128).flat_map(|blk|
                fp8_best_dequant(&blk.iter().map(|&q| bf16_to_f32(q)).collect::<Vec<_>>())).collect()),
            _ => None,
        };
        let (mut num, mut den, mut floor) = (0f64, 0f64, 0f64);
        for r in 0..rows {
            let reference = gemv(&wv.data, r);
            let mut y: Vec<u16> = gemv(&w.data, r).into_iter().map(bf16_rne).collect();
            match &fp8_delta {
                Some(dq) => {
                    let d = gemv(dq, r);
                    add_rows_ref(&mut y, h, &d, h, 1, h, &[1], &[0, 1]);
                }
                None => {
                    let (a, bt, kk, k_max) = site_tables(std::slice::from_ref(&plan), site, h);
                    steer_rows_ref(&mut y, h, 1, h, &[1], &a, &bt, &kk, k_max);
                }
            }
            for (o, want) in reference.iter().enumerate() {
                num += f64::from(bf16_to_f32(y[o]) - want).powi(2);
                den += f64::from(*want).powi(2);
                floor += f64::from(bf16_to_f32(bf16_rne(*want)) - want).powi(2);
            }
        }
        if fp8_delta.is_some() {
            // The official BF16 KDA o_proj already sits on an FP8 grid (its own floor is ~0.0004), so
            // the bar is what serving W' through the FP8 path costs; ΔW's FP8 error is ~2% of ‖ΔW‖.
            let (rebuild, base_floor, variant_floor) = delta_rebuild(&w.data, &wv.data);
            eprintln!("{site}: ‖W + Q(ΔW) − W'‖/‖W‖ {rebuild:.5} (FP8 floor: base {base_floor:.5}, variant {variant_floor:.5})");
            assert!(rebuild <= variant_floor, "{site}: {rebuild} above the variant's FP8 floor {variant_floor}");
        }
        let (rel, floor) = ((num / den).sqrt(), (floor / den).sqrt());
        eprintln!("{site} ({}): |y − W'x|/|W'x| {rel:.5} (one BF16 rounding {floor:.5})", if fp8_delta.is_some() { "delta" } else { "steer" });
        // The steered BF16 tensors rebuild W' to 0.2% of ‖W‖, the FP8 ones to their requantization
        // floor (≤ 1.9%); an FP8 ΔW adds ~3% of ‖ΔW‖ (≈ 5% of ‖W‖).
        let bar = match (&fp8_delta, b.dtype(&tensor)) {
            (Some(_), _) => 0.01,
            (None, Some(cuteafd_core::DType::F8E4M3)) => 0.03,
            _ => 0.01,
        };
        assert!(rel <= bar, "{site}: {rel} > {bar}");
    }
}
