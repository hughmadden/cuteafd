//! Header-only storage contract for the coordinator's Qwen loader.
use super::{resident_bytes, Qwen4Representation};
use crate::families::qwen4::Qwen4Config;
use crate::plan::{Checkpoint, Component, Family};
use crate::plan::families::qwen::QWEN4;
use anyhow::{ensure, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Qwen4CheckpointResident {
    /// Target operands, including the PLE projections and stream mixer, excluding the head.
    pub target_bytes: u64,
    /// Native MTP coordinator operands, excluding routed experts and the shared head.
    pub mtp_bytes: u64,
    pub head_bytes: u64,
    pub embedding_bytes: u64,
    /// Full mapped table backing; device placement reserves this on the GPU instead.
    pub ple_table_bytes: u64,
    /// Bytes in one gathered PLE row (160 BF16 or E4M3 values).
    pub ple_row_bytes: u64,
}

/// Mirrors `Qwen4Loader`: packed shared experts include zero rows, GDN/PLE
/// conv and decay operands widen to FP32, selectable projections/head hold
/// exactly one representation. No tensor payloads are read.
pub fn checkpoint_resident_bytes(checkpoint: &Checkpoint, cfg: &Qwen4Config,
    layers: usize, mtp: bool, representation: Qwen4Representation) -> Result<Qwen4CheckpointResident> {
    let model = QWEN4.open(checkpoint).map_err(|e| anyhow::anyhow!("{}", e.0))?;
    let layers = layers.min(cfg.layers);
    ensure!(!mtp || cfg.mtp_layers == 1 && layers == cfg.layers, "Qwen MTP requires the complete target and one MTP layer");
    let mut out = Qwen4CheckpointResident::default();
    for tensor in &checkpoint.tensors {
        let name = tensor.meta.name.as_str();
        let Some(role) = QWEN4.classify(model.spec(), name) else { continue };
        if role.layer.is_some_and(|l| l >= layers) { continue; }
        let mut bytes = tensor.meta.byte_length;
        match role.component {
            Component::RoutedExpert | Component::SpeculatorExpert | Component::Vision | Component::Other => continue,
            Component::MappedTable => {
                if name.contains(".ngram_embedding.shard_") && name.ends_with(".weight") {
                    out.ple_table_bytes += bytes;
                    ensure!(tensor.meta.shape.len() == 2, "PLE table shard is not a matrix: {name}");
                    let row = bytes / tensor.meta.shape[0] as u64;
                    ensure!(out.ple_row_bytes == 0 || out.ple_row_bytes == row, "PLE shard row widths disagree");
                    out.ple_row_bytes = row;
                }
                continue;
            }
            Component::Embedding => { out.embedding_bytes += bytes; continue; }
            Component::LmHead => continue,
            Component::Speculator if !mtp => continue,
            _ => {}
        }
        // The target uses hyper_connection_mixer, not the unused HF final norm.
        if name == "model.language_model.norm.weight" { continue; }
        let widened = name.ends_with(".linear_attn.conv1d.weight")
            || name.ends_with(".linear_attn.A_log") || name.ends_with(".linear_attn.dt_bias")
            || name.ends_with(".ple.conv1d.weight");
        if widened {
            bytes = tensor.meta.shape.iter().map(|&d| d as u64).product::<u64>() * 4;
        }
        if role.component == Component::Speculator { out.mtp_bytes += bytes; }
        else { out.target_bytes += bytes; }
    }
    // Shared gate/up/gate row concatenation: 1281 source rows, 1296 resident rows.
    let padding = (1296usize.saturating_sub(2 * cfg.shared_intermediate + 1) * cfg.hidden * 2) as u64;
    out.target_bytes += padding * layers as u64;
    if mtp { out.mtp_bytes += padding; }
    let target = resident_bytes(cfg, layers, false, representation);
    out.target_bytes = out.target_bytes.checked_sub(target.projections_bf16 as u64)
        .ok_or_else(|| anyhow::anyhow!("Qwen projection headers do not cover the selected target"))?
        + target.projections as u64;
    if mtp {
        let all = resident_bytes(cfg, layers, true, representation);
        out.mtp_bytes = out.mtp_bytes.checked_sub((all.projections_bf16 - target.projections_bf16) as u64)
            .ok_or_else(|| anyhow::anyhow!("Qwen projection headers do not cover native MTP"))?
            + (all.projections - target.projections) as u64;
    }
    out.head_bytes = target.head as u64;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing::{qwen4_config, t, write_snapshot};

    #[test]
    fn headers_reserve_packing_and_widening_without_tables_or_experts() {
        let dir = std::env::temp_dir().join(format!("qwen-resident-{}", std::process::id()));
        let config = qwen4_config(1);
        let cfg = Qwen4Config::from_hf(&config).unwrap();
        let p = "model.language_model.layers.0";
        let tensors = vec![
            t(format!("{p}.linear_attn.in_proj_qkv.weight"), "BF16", &[10240, 2560]),
            t(format!("{p}.linear_attn.in_proj_z.weight"), "BF16", &[6144, 2560]),
            t(format!("{p}.linear_attn.in_proj_b.weight"), "BF16", &[48, 2560]),
            t(format!("{p}.linear_attn.in_proj_a.weight"), "BF16", &[48, 2560]),
            t(format!("{p}.linear_attn.out_proj.weight"), "BF16", &[2560, 6144]),
            t(format!("{p}.linear_attn.conv1d.weight"), "BF16", &[10240, 1, 4]),
            t(format!("{p}.linear_attn.A_log"), "BF16", &[48]),
            t(format!("{p}.linear_attn.dt_bias"), "BF16", &[48]),
            t(format!("{p}.mlp.shared_expert.gate_proj.weight"), "BF16", &[640, 2560]),
            t(format!("{p}.mlp.shared_expert.up_proj.weight"), "BF16", &[640, 2560]),
            t(format!("{p}.mlp.shared_expert_gate.weight"), "BF16", &[1, 2560]),
            t(format!("{p}.mlp.shared_expert.down_proj.weight"), "BF16", &[2560, 640]),
            t(format!("{p}.ple.ple_embedding.ngram_embedding.shard_0.weight"), "BF16", &[1000, 160]),
            t(format!("{p}.ple.ple_embedding.layer_multipliers"), "I64", &[16]),
            t(format!("{p}.mlp.experts.0.gate_proj.weight"), "BF16", &[640, 2560]),
            t("model.language_model.embed_tokens.weight", "BF16", &[64, 2560]),
            t("model.language_model.norm.weight", "BF16", &[2560]),
            t("lm_head.weight", "BF16", &[64, 2560]),
        ];
        write_snapshot(&dir, &config, &tensors, None);
        let checkpoint = Checkpoint::open(&dir).unwrap();
        let bf16 = checkpoint_resident_bytes(&checkpoint, &cfg, 1, false, Qwen4Representation::default()).unwrap();
        assert_eq!(bf16.target_bytes, (16480 * 2560 * 2 + 2560 * 6144 * 2
            + (10240 * 4 + 48 * 2) * 4 + 1296 * 2560 * 2 + 2560 * 640 * 2) as u64);
        assert_eq!(bf16.embedding_bytes, 64 * 2560 * 2);
        assert_eq!(bf16.head_bytes, 64 * 2560 * 2);
        assert_eq!((bf16.ple_table_bytes, bf16.ple_row_bytes), (1000 * 160 * 2, 160 * 2));
        assert_eq!(bf16.mtp_bytes, 0);
        let fp8 = checkpoint_resident_bytes(&checkpoint, &cfg, 1, false,
            Qwen4Representation { fp8_projections: true, fp8_head: true }).unwrap();
        assert_eq!(fp8.head_bytes, 64 * 2560 + 64 * 20 * 4);
        let selection = resident_bytes(&cfg, 1, false,
            Qwen4Representation { fp8_projections: true, fp8_head: true });
        assert_eq!(bf16.target_bytes - fp8.target_bytes,
            (selection.projections_bf16 - selection.projections_fp8) as u64);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
