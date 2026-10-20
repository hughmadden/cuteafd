//! Target-family geometry adapter for the shared native BF16 MLA packer.
use super::config::{KimiAttention, KimiK3Config};
use crate::{formats::mla::{self, MlaGeometry, MlaOperand, MlaTensor, MlaWeightError}, tensors::LoadedTensor};

pub fn pack(cfg: &KimiK3Config, layer: usize, operand: MlaOperand, ranks: usize,
    read: impl FnMut(&str) -> anyhow::Result<LoadedTensor>) -> Result<Vec<MlaTensor>, MlaWeightError> {
    if cfg.attention.get(layer) != Some(&KimiAttention::Mla) {
        return Err(MlaWeightError::Geometry("target layer is not MLA"));
    }
    let geometry = MlaGeometry { hidden: cfg.hidden, heads: cfg.heads, q_rank: cfg.q_lora_rank,
        kv_rank: cfg.kv_lora_rank, nope: cfg.qk_nope_dim, rope: cfg.qk_rope_dim, value: cfg.v_head_dim };
    mla::pack(&geometry, &format!("language_model.model.layers.{layer}.self_attn"), true, operand, ranks, read)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::{TensorInfo, TensorRole};

    fn cfg() -> KimiK3Config {
        let mut cfg = KimiK3Config::from_hf(&crate::plan::testing::kimi_k3_config()).unwrap();
        cfg.hidden = 2; cfg.heads = 4; cfg.q_lora_rank = 3;
        cfg.kv_lora_rank = 2; cfg.qk_nope_dim = 2; cfg.qk_rope_dim = 1; cfg.v_head_dim = 1;
        cfg
    }
    fn tensor(name: &str) -> anyhow::Result<LoadedTensor> {
        assert!(name.starts_with("language_model.model.layers.3.self_attn."));
        let leaf = name.rsplit_once("self_attn.").unwrap().1;
        let shape = match leaf {
            "q_a_proj.weight" => vec![3, 2], "kv_a_proj_with_mqa.weight" => vec![3, 2],
            "q_a_layernorm.weight" => vec![3], "kv_a_layernorm.weight" => vec![2],
            "q_b_proj.weight" => vec![12, 3], "kv_b_proj.weight" => vec![12, 2],
            "g_proj.weight" => vec![4, 2], "o_proj.weight" => vec![2, 4],
            _ => anyhow::bail!("unexpected {name}"),
        };
        let count = shape.iter().product::<usize>();
        let bytes: Vec<u8> = (0..count as u16).flat_map(u16::to_le_bytes).collect();
        Ok(LoadedTensor { info: TensorInfo { name: name.into(), file: "test".into(),
            dtype: DType::Bf16, shape, byte_offset: 0, byte_length: bytes.len() as u64,
            role: TensorRole::Other, layer_id: Some(3), expert_id: None, is_quantization_metadata: false },
            source_path: "test".into(), bytes, elapsed_micros: 0, sha256: String::new() })
    }
    fn words(t: &MlaTensor) -> Vec<u16> {
        t.bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()
    }
    #[test]
    fn tp_head_slices_and_uk_transpose_preserve_native_bits() {
        let cfg = cfg();
        let uk = pack(&cfg, 3, MlaOperand::Uk, 2, tensor).unwrap();
        assert_eq!(uk[0].shape, [2, 2, 2]);
        assert_eq!(words(&uk[0]), [0, 2, 1, 3, 6, 8, 7, 9]);
        assert_eq!(words(&uk[1]), [12, 14, 13, 15, 18, 20, 19, 21]);
        let uv = pack(&cfg, 3, MlaOperand::Uv, 2, tensor).unwrap();
        assert_eq!(uv[0].shape, [2, 1, 2]);
        assert_eq!(words(&uv[0]), [4, 5, 10, 11]);
        assert_eq!(words(&uv[1]), [16, 17, 22, 23]);
        let o = pack(&cfg, 3, MlaOperand::Output, 2, tensor).unwrap();
        assert_eq!(words(&o[0]), [0, 1, 4, 5]);
        assert_eq!(words(&o[1]), [2, 3, 6, 7]);
        let q = pack(&cfg, 3, MlaOperand::QB, 2, tensor).unwrap();
        assert_eq!(q[1].shape, [6, 3]);
        assert_eq!(words(&q[1]), (18..36).collect::<Vec<_>>());
        let gate = pack(&cfg, 3, MlaOperand::Gate, 2, tensor).unwrap();
        assert_eq!(words(&gate[1]), [4, 5, 6, 7]);
    }
    #[test]
    fn low_rank_inputs_replicate_and_bad_contracts_refuse() {
        let cfg = cfg();
        let a = pack(&cfg, 3, MlaOperand::QkvA, 2, tensor).unwrap();
        assert_eq!(a[0], a[1]);
        assert_eq!(a[0].shape, [6, 2]);
        assert_eq!(words(&a[0]), (0..6).chain(0..6).collect::<Vec<_>>());
        for op in [MlaOperand::QNorm, MlaOperand::KvNorm] {
            let norms = pack(&cfg, 3, op, 2, tensor).unwrap();
            assert_eq!(norms[0], norms[1]);
        }
        assert!(pack(&cfg, 0, MlaOperand::QB, 2, |_| unreachable!()).is_err());
        assert!(pack(&cfg, 3, MlaOperand::QB, 3, |_| unreachable!()).is_err());
        assert!(pack(&cfg, 3, MlaOperand::QB, 2, |name| {
            let mut t = tensor(name)?; t.info.dtype = DType::F32; Ok(t)
        }).is_err());
        assert!(pack(&cfg, 3, MlaOperand::Uk, 2, |name| {
            let mut t = tensor(name)?; t.bytes.pop(); Ok(t)
        }).is_err());
        let all = pack(&cfg, 3, MlaOperand::Uk, 1, tensor).unwrap();
        assert_eq!(all[0].shape, [4, 2, 2]);
    }
    #[test]
    fn draft_and_target_share_packing_without_reading_a_gate_or_borrowed_weights() {
        let cfg = cfg();
        let geometry = MlaGeometry { hidden: cfg.hidden, heads: cfg.heads, q_rank: cfg.q_lora_rank,
            kv_rank: cfg.kv_lora_rank, nope: cfg.qk_nope_dim, rope: cfg.qk_rope_dim, value: cfg.v_head_dim };
        for operand in [MlaOperand::QkvA, MlaOperand::QNorm, MlaOperand::KvNorm, MlaOperand::QB,
            MlaOperand::Uk, MlaOperand::Uv, MlaOperand::Output] {
            let expected = pack(&cfg, 3, operand, 1, tensor).unwrap();
            let mut reads = Vec::new();
            let draft = mla::pack(&geometry, "layers.2.self_attn", false, operand, 1, |name| {
                reads.push(name.to_owned());
                tensor(&name.replace("layers.2.self_attn", "language_model.model.layers.3.self_attn"))
            }).unwrap();
            assert_eq!(draft, expected);
            assert!(reads.iter().all(|name| name.starts_with("layers.2.self_attn.")
                && !name.contains("g_proj") && !name.contains("embed_tokens") && !name.contains("lm_head")));
        }
        assert!(mla::pack(&geometry, "layers.2.self_attn", false, MlaOperand::Gate, 1, |_| unreachable!()).is_err());
        assert!(mla::pack(&geometry, "", false, MlaOperand::QB, 1, |_| unreachable!()).is_err());
    }

    #[test]
    fn overflow_refuses_before_checkpoint_reads() {
        for field in 0..7 {
            let mut cfg = cfg();
            match field {
                0 => cfg.hidden = usize::MAX,
                1 => cfg.q_lora_rank = usize::MAX,
                2 => cfg.kv_lora_rank = usize::MAX,
                3 => cfg.qk_nope_dim = usize::MAX,
                4 => cfg.qk_rope_dim = usize::MAX,
                5 => cfg.v_head_dim = usize::MAX,
                _ => cfg.heads = usize::MAX - 1,
            }
            for operand in [MlaOperand::QkvA, MlaOperand::QNorm, MlaOperand::KvNorm,
                MlaOperand::QB, MlaOperand::Uk, MlaOperand::Uv, MlaOperand::Gate, MlaOperand::Output] {
                assert!(matches!(pack(&cfg, 3, operand, 2, |_| unreachable!()),
                    Err(MlaWeightError::Geometry(_))));
            }
        }
    }
}
