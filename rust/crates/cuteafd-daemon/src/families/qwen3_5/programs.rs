//! Fail-closed checkpoint/AOT agreement, before weight or workspace admission.
use super::weights::DenseFormat;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::programs::{ProgramSpec, Programs};
use cuteafd_loader::families::qwen3_5::Qwen35Config;

pub(crate) struct DensePrograms {
    pub(crate) kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache,
    pub(crate) prefix: &'static str,
    pub(crate) format: DenseFormat,
    pub(crate) norm_dtype: DType,
    pub(crate) scratch_bytes: usize,
}

fn native(spec: &ProgramSpec) -> cuteafd_loader::placement::families::qwen3_5::NativeProgram {
    cuteafd_loader::placement::families::qwen3_5::NativeProgram { name: spec.name.clone(),
        geometry: spec.geometry.clone(), capacity_rows: spec.capacity_rows as u64,
        pointers: spec.pointers.clone(), scalar_kinds: spec.scalar_kinds.clone(), scratch: spec.scratch.clone() }
}

fn norm_dtype(spec: &ProgramSpec) -> Result<DType> {
    cuteafd_loader::placement::families::qwen3_5::norm_dtype(&native(spec))
}

fn validate(cfg: &Qwen35Config, op: &str, rows: usize, format: cuteafd_loader::families::qwen4::Qwen4KvCache, spec: &ProgramSpec) -> Result<()> {
    cuteafd_loader::placement::families::qwen3_5::validate(cfg, op, rows, format, &native(spec))
}

fn validate_abi(op: &str, rows: usize, spec: &ProgramSpec) -> Result<()> {
    cuteafd_loader::placement::families::qwen3_5::validate_abi(op, rows, &native(spec))
}

impl DensePrograms {
    pub(crate) fn new(cfg: &Qwen35Config, programs: &Programs<'_>, kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache, format: DenseFormat) -> Result<Self> {
        // These are actual exported geometries, not aliases for Flash Next.
        let prefix = match (cfg.hidden, cfg.intermediate, cfg.heads, cfg.gdn.value_heads) {
            (4096, 12288, 16, 32) => "qwen35",
            (5120, 17408, 24, 48) => "qwen38",
            _ => anyhow::bail!("dense Qwen H{}/I{}/heads{}/GDN{} has no native BF16 export",
                cfg.hidden, cfg.intermediate, cfg.heads, cfg.gdn.value_heads),
        };
        ensure!(cfg.gdn.key_dim == 128 && cfg.gdn.value_dim == 128 && cfg.gdn.conv_kernel == 4,
            "dense Qwen exports require GDN128 and conv4");
        let mut scratch_bytes = 0;
        let mut norm = None;
        for (op, rows) in [("norm", 0), ("add", 0), ("gdn_commit", 0)].into_iter().chain(
            [64, 4096].into_iter().flat_map(|rows|
                ["gdn", "attn_producer", "full_gqa", "attn_o", "ffn"].map(|op| (op, rows)))) {
            let name = cuteafd_loader::placement::families::qwen3_5::native_program_name(prefix, op, rows, kv_format, format.fp8());
            let spec = programs.spec(&name)?;
            if matches!(op, "gdn" | "attn_producer" | "attn_o" | "ffn") {
                ensure!((spec.geometry["fp8_weights"] == "only") == format.fp8(), "{name}: checkpoint projection format mismatch");
                if format.fp8() { ensure!(spec.geometry["weight_block"] == serde_json::json!([128, 128]), "{name}: invalid FP8 block geometry"); }
            }
            validate(cfg, op, rows, kv_format, spec)?;
            validate_abi(op, rows, spec)?;
            if op == "gdn" {
                let dtype = norm_dtype(spec)?;
                ensure!(norm.as_ref().is_none_or(|previous| *previous == dtype), "decode/prefill GDN norm dtype differs");
                norm = Some(dtype);
            }
            for &bytes in spec.scratch.values() {
                scratch_bytes = scratch_bytes.max(usize::try_from(bytes).context("dense Qwen scratch exceeds usize")?);
            }
        }
        Ok(Self { prefix, kv_format, format, norm_dtype: norm.context("dense Qwen GDN norm ABI missing")?, scratch_bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::families::qwen::{GdnGeometry, GdnOutputGate, HybridAttention};
    use serde_json::json;

    fn cfg() -> Qwen35Config {
        Qwen35Config { hidden: 4096, intermediate: 12288, vocab: 248320, attention: vec![HybridAttention::Gdn],
            heads: 16, kv_heads: 4, head_dim: 256, gdn: GdnGeometry { key_heads: 16, value_heads: 32,
                key_dim: 128, value_dim: 128, conv_kernel: 4, output_gate: GdnOutputGate::Silu },
            rope_dim: 64, rope_theta: 1e7, rms_norm_eps: 1e-6, max_context: 262144, mtp_layers: 0, tied_embeddings: false }
    }

    #[test]
    fn native_pointer_scalar_and_scratch_contracts_fail_before_allocation() -> Result<()> {
        let mut spec = ProgramSpec { name: "full_gqa".into(),
            pointers: ["query", "kv_cache", "positions", "page_table", "out", "scratch"]
                .map(str::to_owned).to_vec(), scalar_kinds: vec![b'i'; 3],
            scratch: [("scratch".into(), 1024)].into(), capacity_rows: 64, geometry: json!({}) };
        validate_abi("full_gqa", 64, &spec)?;
        spec.pointers.swap(0, 1);
        assert!(validate_abi("full_gqa", 64, &spec).is_err());
        spec.pointers.swap(0, 1);
        spec.scalar_kinds[1] = b'l';
        assert!(validate_abi("full_gqa", 64, &spec).is_err());
        spec.scalar_kinds[1] = b'i';
        spec.scratch.insert("scratch".into(), 1023);
        assert!(validate_abi("full_gqa", 64, &spec).is_err());
        spec.scratch.clear();
        assert!(validate_abi("full_gqa", 64, &spec).is_err());
        Ok(())
    }

    #[test]
    fn gdn_replay_layout_and_checkpoint_gate_are_an_explicit_contract() -> Result<()> {
        let cfg = cfg();
        let mut spec = ProgramSpec { name: "gdn".into(), pointers: vec![], scalar_kinds: vec![],
            scratch: Default::default(), capacity_rows: 64, geometry: json!({"hidden":4096,"max_rows":64,
                "key_heads":16,"value_heads":32,"head_dim":128,"in_width":12352,"eps":1e-6,
                "output_gate":"silu","fp8_weights":false,"norm_dtype":"torch.float32"}) };
        validate(&cfg, "gdn", 64, cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, &spec)?;
        assert_eq!(norm_dtype(&spec)?, DType::F32);
        for (key, wrong) in [("value_heads", json!(48)), ("output_gate", json!("sigmoid")),
            ("norm_dtype", json!("torch.float16")), ("fp8_weights", json!(true)), ("in_width", json!(16480))] {
            let saved = spec.geometry[key].clone();
            spec.geometry[key] = wrong;
            assert!(validate(&cfg, "gdn", 64, cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, &spec).is_err(), "{key}");
            spec.geometry[key] = saved;
        }
        spec.geometry = json!({"key_heads":16,"value_heads":32,"channels":8192,"replay_rows":64,
            "replay_bytes":crate::shared::qwen::GdnLayout::new(cfg.gdn)?.replay_layer_bytes});
        validate(&cfg, "gdn_commit", 0, cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, &spec)?;
        spec.geometry["replay_bytes"] = json!(0);
        assert!(validate(&cfg, "gdn_commit", 0, cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, &spec).is_err());
        Ok(())
    }

    #[test]
    fn native_fp8_projection_abis_preserve_scales_bf16_gates_and_prefill_threshold() -> Result<()> {
        for rows in [64, 4096] {
            for (op, pointers) in [
                ("gdn", vec!["x", "w_qkvz_fp8", "w_qkvz_scale", "w_ba", "conv_w", "a_log", "dt_bias", "norm_w",
                    "w_out_fp8", "w_out_scale", "conv_state", "state", "slots", "seq_first", "out"]),
                ("attn_producer", vec!["x", "w_in_fp8", "w_in_scale", "q_norm", "k_norm", "rope_positions", "kv_slots",
                    "kv_cache", "query", "gate", "scratch"]),
                ("attn_o", vec!["attn", "gate", "w_o_fp8", "w_o_scale", "out", "scratch"]),
                ("ffn", vec!["x", "w_gate_up_fp8", "w_gate_up_scale", "w_down_fp8", "w_down_scale", "out", "scratch"]),
            ] {
                let mut pointers: Vec<String> = pointers.into_iter().map(str::to_owned).collect();
                if op == "gdn" {
                    if rows == 64 { pointers.push("replay".into()); }
                    pointers.push("scratch".into());
                }
                let scalars = if (op == "gdn" && rows == 64) || (op == "attn_o" && rows == 4096) { 2 } else { 1 };
                let mut spec = ProgramSpec { name: format!("{op}_fp8_m{rows}"), pointers,
                    scalar_kinds: vec![b'i'; scalars], scratch: [("scratch".into(), 1024)].into(),
                    capacity_rows: rows as u32, geometry: json!({"fp8_weights":"only"}) };
                validate_abi(op, rows, &spec)?;
                spec.pointers.swap(1, 2);
                assert!(validate_abi(op, rows, &spec).is_err(), "{op} rows={rows}: reordered native operands");
                spec.pointers.swap(1, 2);
                spec.scalar_kinds.push(b'i');
                assert!(validate_abi(op, rows, &spec).is_err(), "{op} rows={rows}: wrong threshold scalar");
            }
        }
        Ok(())
    }

    #[test]
    fn stale_attention_or_norm_geometry_is_rejected_before_loading() -> Result<()> {
        let cfg = cfg();
        let mut spec = ProgramSpec { name: "full_gqa".into(), pointers: vec![], scalar_kinds: vec![],
            scratch: Default::default(), capacity_rows: 64, geometry: json!({"max_rows":64,"heads":16,"kv_heads":4,
                "head_dim":256,"record_bytes":4096,"kv_format":"bf16","indexed":false,"page_rows":64,"max_splits":32,
                "causal":true,"empty_rows_zero":true}) };
        validate(&cfg, "full_gqa", 64, cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, &spec)?;
        for (key, wrong) in [("heads", json!(24)), ("record_bytes", json!(2048)), ("indexed", json!(true)),
            ("empty_rows_zero", json!(false)), ("causal", json!(false)), ("max_rows", json!(32))] {
            let saved = spec.geometry[key].clone();
            spec.geometry[key] = wrong;
            assert!(validate(&cfg, "full_gqa", 64, cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, &spec).is_err(), "{key}");
            spec.geometry[key] = saved;
        }
        let fp8 = cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8;
        assert!(validate(&cfg, "full_gqa", 64, fp8, &spec).is_err());
        spec.geometry["record_bytes"] = json!(2080);
        spec.geometry["kv_format"] = json!("fp8");
        validate(&cfg, "full_gqa", 64, fp8, &spec)?;
        assert!(validate(&cfg, "full_gqa", 64, cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, &spec).is_err());
        spec.geometry = json!({"hidden":4096,"eps":1e-6,"zero_centered":true});
        validate(&cfg, "norm", 0, cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, &spec)?;
        spec.geometry["zero_centered"] = json!(false);
        assert!(validate(&cfg, "norm", 0, cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, &spec).is_err());
        assert!(norm_dtype(&spec).is_err());
        Ok(())
    }
}
