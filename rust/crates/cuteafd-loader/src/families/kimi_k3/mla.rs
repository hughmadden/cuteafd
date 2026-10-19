//! Native BF16 NoPE MLA operands, packed one at a time in checkpoint head order.
use super::config::{KimiAttention, KimiK3Config};
use crate::tensors::LoadedTensor;
use cuteafd_core::DType;
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlaOperand { QkvA, QNorm, KvNorm, QB, Uk, Uv, Gate, Output }

#[derive(Debug, PartialEq, Eq)]
pub struct MlaTensor {
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum MlaWeightError {
    #[error("Kimi MLA invalid {0}")]
    Geometry(&'static str),
    #[error("Kimi MLA tensor {name}: expected BF16 {shape:?}")]
    Tensor { name: String, shape: Vec<usize> },
    #[error("Kimi MLA checkpoint read {name}: {source}")]
    Read { name: String, source: anyhow::Error },
}
type Result<T> = std::result::Result<T, MlaWeightError>;

/// Replicate low-rank A inputs/norms; shard whole Q/K/V heads and o_proj
/// columns. UK is transposed per head; UV retains its checkpoint row order.
/// Only the requested operand is materialized, never an extra entire layer.
pub fn pack(cfg: &KimiK3Config, layer: usize, operand: MlaOperand, ranks: usize,
    mut read: impl FnMut(&str) -> anyhow::Result<LoadedTensor>) -> Result<Vec<MlaTensor>> {
    if cfg.attention.get(layer) != Some(&KimiAttention::Mla) || !matches!(ranks, 1 | 2)
        || cfg.heads == 0 || cfg.heads % ranks != 0 || cfg.hidden == 0
        || cfg.q_lora_rank == 0 || cfg.kv_lora_rank == 0 || cfg.qk_nope_dim == 0
        || cfg.qk_rope_dim == 0 || cfg.v_head_dim == 0 {
        return Err(MlaWeightError::Geometry("layer/head/rank geometry"));
    }
    let base = format!("language_model.model.layers.{layer}.self_attn");
    let extent = |shape: &[usize]| shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d))
        .ok_or(MlaWeightError::Geometry("operand extent overflow"));
    let mut load = |leaf: &str, shape: &[usize]| -> Result<Vec<u8>> {
        let name = format!("{base}.{leaf}.weight");
        let t = read(&name).map_err(|source| MlaWeightError::Read { name: name.clone(), source })?;
        if t.info.dtype != DType::Bf16 || t.info.shape != shape || t.bytes.len() != extent(shape)? {
            return Err(MlaWeightError::Tensor { name, shape: shape.to_vec() });
        }
        Ok(t.bytes)
    };
    let (heads, local, hidden, kv) = (cfg.heads, cfg.heads / ranks, cfg.hidden, cfg.kv_lora_rank);
    let qdim = cfg.qk_nope_dim.checked_add(cfg.qk_rope_dim)
        .ok_or(MlaWeightError::Geometry("query width overflow"))?;
    let vdim = cfg.qk_nope_dim.checked_add(cfg.v_head_dim)
        .ok_or(MlaWeightError::Geometry("KV projection width overflow"))?;
    let overflow = || MlaWeightError::Geometry("operand extent overflow");
    let kv_a_rows = kv.checked_add(cfg.qk_rope_dim).ok_or_else(overflow)?;
    let qkv_a_rows = cfg.q_lora_rank.checked_add(kv_a_rows).ok_or_else(overflow)?;
    let q_rows = heads.checked_mul(qdim).ok_or_else(overflow)?;
    let kv_rows = heads.checked_mul(vdim).ok_or_else(overflow)?;
    let v_rows = heads.checked_mul(cfg.v_head_dim).ok_or_else(overflow)?;
    for shape in [&[qkv_a_rows, hidden][..], &[q_rows, cfg.q_lora_rank],
        &[kv_rows, kv], &[v_rows, hidden], &[hidden, v_rows]] {
        extent(shape)?;
    }
    let mut replicate = |leaf: &str, shape: Vec<usize>| -> Result<Vec<MlaTensor>> {
        let bytes = load(leaf, &shape)?;
        Ok((0..ranks).map(|_| MlaTensor { shape: shape.clone(), bytes: bytes.clone() }).collect())
    };
    match operand {
        MlaOperand::QNorm => replicate("q_a_layernorm", vec![cfg.q_lora_rank]),
        MlaOperand::KvNorm => replicate("kv_a_layernorm", vec![kv]),
        MlaOperand::QkvA => {
            let mut bytes = load("q_a_proj", &[cfg.q_lora_rank, hidden])?;
            bytes.extend(load("kv_a_proj_with_mqa", &[kv_a_rows, hidden])?);
            let shape = vec![qkv_a_rows, hidden];
            Ok((0..ranks).map(|_| MlaTensor { shape: shape.clone(), bytes: bytes.clone() }).collect())
        }
        MlaOperand::QB | MlaOperand::Gate => {
            let (leaf, width, input) = if operand == MlaOperand::QB {
                ("q_b_proj", qdim, cfg.q_lora_rank)
            } else { ("g_proj", cfg.v_head_dim, hidden) };
            let bytes = load(leaf, &[heads * width, input])?;
            let share = bytes.len() / ranks;
            Ok((0..ranks).map(|rank| MlaTensor { shape: vec![local * width, input],
                bytes: bytes[rank * share..(rank + 1) * share].to_vec() }).collect())
        }
        MlaOperand::Output => {
            let bytes = load("o_proj", &[hidden, heads * cfg.v_head_dim])?;
            let row_bytes = heads * cfg.v_head_dim * 2;
            let share = row_bytes / ranks;
            Ok((0..ranks).map(|rank| MlaTensor { shape: vec![hidden, local * cfg.v_head_dim],
                bytes: bytes.chunks_exact(row_bytes)
                    .flat_map(|row| row[rank * share..(rank + 1) * share].iter().copied()).collect() }).collect())
        }
        MlaOperand::Uk | MlaOperand::Uv => {
            let bytes = load("kv_b_proj", &[heads * vdim, kv])?;
            let shape = if operand == MlaOperand::Uk { vec![local, kv, cfg.qk_nope_dim] }
                else { vec![local, cfg.v_head_dim, kv] };
            let mut result = Vec::with_capacity(ranks);
            for rank in 0..ranks {
                let mut packed = Vec::with_capacity(extent(&shape)?);
                for head in rank * local..(rank + 1) * local {
                    if operand == MlaOperand::Uk {
                        for k in 0..kv {
                            for d in 0..cfg.qk_nope_dim {
                                let at = ((head * vdim + d) * kv + k) * 2;
                                packed.extend_from_slice(&bytes[at..at + 2]);
                            }
                        }
                    } else {
                        let start = (head * vdim + cfg.qk_nope_dim) * kv * 2;
                        packed.extend_from_slice(&bytes[start..start + cfg.v_head_dim * kv * 2]);
                    }
                }
                result.push(MlaTensor { shape: shape.clone(), bytes: packed });
            }
            Ok(result)
        }
    }
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
