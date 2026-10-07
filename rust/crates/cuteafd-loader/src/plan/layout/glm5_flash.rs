//! GLM 5.3 Flash's resident coordinator weights on one GPU, as the engine's loader
//! (`glm5_flash::weights::GlmfLoader`) uploads them: per layer the mHC operands widened to FP32,
//! the norms, the KDA projections in BF16 (or E4M3 with per-row scales, `--kda-fp8`) with their
//! short convolutions in FP32, the MLA projections as E4M3 128x128 blocks beside the absorbed BF16
//! `kv_b_proj`, the indexer in BF16, the dense or shared-expert MLP as E4M3 blocks and the router;
//! then the final norm, the head (BF16, or E4M3 per row and 128-wide K block with `--fp8-head`)
//! and the token embedding (which `--embedding-placement host` maps from pinned RAM instead).
//! The E4M3 copies come from the FP8 release or are quantized at load: the same bytes either
//! way. Every allocation is floored at 256 bytes, as the engine allocates it.
use crate::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
use crate::plan::Checkpoint;
use cuteafd_core::DType;

const PREFIX: &str = "model.language_model.";
const FLOOR: u64 = 256;

/// How the KDA in/out projections stay resident (`--kda-fp8`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GlmfKdaFp8 {
    /// The checkpoint's BF16 (the engine's default).
    #[default]
    Off,
    /// E4M3 with one scale per output row and 128-wide K block.
    Row128,
    /// E4M3 with one scale per output row (stored per 128-wide K block like row128).
    Channel,
}

/// `(group, format, bytes)` of every resident operand group on the one GPU, or None when the
/// checkpoint lacks a tensor this loader reads (or stores one in a format it does not model, such
/// as a ModelOpt NVFP4 dense MLP): the generic per-component estimate then stands.
pub(super) fn resident_weights(checkpoint: &Checkpoint, kda_fp8: GlmfKdaFp8, fp8_head: bool)
    -> Option<Vec<(String, String, u64)>> {
    let cfg = GlmNextConfig::from_hf(&checkpoint.config).ok()?;
    let meta = |name: &str| checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name)).ok()
        .map(|at| &checkpoint.tensors[at].meta);
    let numel = |name: &str| meta(name).map(|m| m.shape.iter().product::<usize>() as u64);
    let mut groups: std::collections::BTreeMap<(&'static str, &'static str), u64> = Default::default();
    let mut add = |group: &'static str, format: &'static str, bytes: u64| {
        *groups.entry((group, format)).or_default() += bytes.max(FLOOR);
    };
    // One tensor as stored (a 2-D E4M3 tensor dequantized to BF16 rows).
    let one = |name: &str| -> Option<u64> {
        let m = meta(name)?;
        Some(if m.dtype == DType::F8E4M3 && m.shape.len() == 2 { numel(name)? * 2 } else { m.byte_length })
    };
    // Row-concatenated BF16 rows; FP32 widenings.
    let rows = |names: &[String]| names.iter().map(|n| numel(n).map(|v| v * 2)).sum::<Option<u64>>();
    let f32 = |names: &[String]| names.iter().map(|n| numel(n).map(|v| v * 4)).sum::<Option<u64>>();
    let matrix = |name: &str| meta(name).filter(|m| m.shape.len() == 2 && matches!(m.dtype, DType::Bf16 | DType::F8E4M3))
        .map(|m| (m.shape[0] as u64, m.shape[1] as u64));
    // E4M3 values and FP32 scales of row-concatenated matrices: 128x128 blocks, or per row and
    // 128-wide K block.
    let fp8 = |names: &[String], blocks: bool| -> Option<(u64, u64)> {
        names.iter().try_fold((0, 0), |(values, scales), name| {
            let (n, k) = matrix(name)?;
            let grid = if blocks { n.div_ceil(128) } else { n } * k.div_ceil(128);
            Some((values + n * k, scales + grid * 4))
        })
    };
    for layer in 0..cfg.layers {
        let p = format!("{PREFIX}layers.{layer}");
        for site in ["attn", "ffn"] {
            for part in ["fn", "scale", "base"] {
                add("hyper_connection", "f32", f32(&[format!("{p}.hc_{site}_{part}")])?);
            }
        }
        add("norm", "bf16", one(&format!("{p}.input_layernorm.weight"))?);
        add("norm", "bf16", one(&format!("{p}.post_attention_layernorm.weight"))?);
        let a = |name: &str| format!("{p}.self_attn.{name}");
        match cfg.attention[layer] {
            GlmNextAttention::Kda => {
                let w_in = ["q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj"].map(|n| a(&format!("{n}.weight")));
                let w_o = [a("o_proj.weight")];
                if kda_fp8 == GlmfKdaFp8::Off {
                    add("attention", "bf16", rows(&w_in)?);
                    add("attention", "bf16", one(&w_o[0])?);
                } else {
                    for names in [&w_in[..], &w_o[..]] {
                        let (values, scales) = fp8(names, false)?;
                        add("attention", "fp8", values);
                        add("attention", "fp8-row128-scale", scales);
                    }
                }
                add("attention", "bf16", rows(&[a("f_b_proj.weight"), a("g_b_proj.weight")])?);
                add("attention", "f32", f32(&["q", "k", "v"].map(|n| a(&format!("{n}_conv1d.weight"))))?);
                add("attention", "f32", f32(&[a("A_log")])?);
                add("attention", "f32", f32(&[a("dt_bias")])?);
                add("norm", "bf16", one(&a("o_norm.weight"))?);
            }
            GlmNextAttention::Mla => {
                add("norm", "bf16", one(&a("q_a_layernorm.weight"))?);
                add("norm", "bf16", one(&a("kv_a_layernorm.weight"))?);
                // kv_b_proj absorbed into w_uk [N, latent, nope] and w_uv [N, v, latent], BF16.
                let (n, nope, v, latent) = (cfg.heads as u64, cfg.qk_nope_dim as u64, cfg.v_head_dim as u64,
                    cfg.kv_lora_rank as u64);
                (matrix(&a("kv_b_proj.weight"))? == (n * (nope + v), latent)).then_some(())?;
                add("attention", "bf16", n * latent * nope * 2);
                add("attention", "bf16", n * v * latent * 2);
                for names in [vec![a("q_a_proj.weight"), a("kv_a_proj_with_mqa.weight")], vec![a("q_b_proj.weight")],
                    vec![a("o_proj.weight")]] {
                    let (values, scales) = fp8(&names, true)?;
                    add("attention", "fp8", values);
                    add("attention", "fp8-block128-scale", scales);
                }
                let i = |name: &str| a(&format!("indexer.{name}"));
                add("indexer", "bf16", one(&i("wq_b.weight"))?);
                add("indexer", "bf16", rows(&[i("wk.weight"), i("weights_proj.weight"), i("index_kpool_compress_gate")])?);
                for name in ["k_norm.weight", "k_norm.bias", "index_kpool_compress_ape"] {
                    add("indexer", "bf16", one(&i(name))?);
                }
            }
        }
        let (group, mlp) = if cfg.dense[layer] { ("dense_ffn", format!("{p}.mlp")) }
            else { ("shared_expert", format!("{p}.mlp.shared_experts")) };
        for names in [vec![format!("{mlp}.gate_proj.weight"), format!("{mlp}.up_proj.weight")],
            vec![format!("{mlp}.down_proj.weight")]] {
            let (values, scales) = fp8(&names, true)?;
            add(group, "fp8", values);
            add(group, "fp8-block128-scale", scales);
        }
        if !cfg.dense[layer] {
            add("router", "bf16", one(&format!("{p}.mlp.gate.weight"))?);
            add("router", "f32", f32(&[format!("{p}.mlp.gate.e_score_correction_bias")])?);
        }
    }
    add("norm", "bf16", one(&format!("{PREFIX}norm.weight"))?);
    if fp8_head {
        let (values, scales) = fp8(&["lm_head.weight".to_string()], false)?;
        add("lm_head", "fp8", values);
        add("lm_head", "fp8-row128-scale", scales);
    } else {
        add("lm_head", "bf16", one("lm_head.weight")?);
    }
    add("embedding", "bf16", rows(&[format!("{PREFIX}embed_tokens.weight")])?);
    Some(groups.into_iter().map(|((group, format), bytes)| (group.to_string(), format.to_string(), bytes)).collect())
}
