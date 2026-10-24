//! Shared native BF16 MLA packing in checkpoint head order.
use crate::tensors::LoadedTensor;
use cuteafd_core::DType;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlaGeometry {
    pub hidden: usize,
    pub heads: usize,
    pub q_rank: usize,
    pub kv_rank: usize,
    pub nope: usize,
    pub rope: usize,
    pub value: usize,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlaOperand { QkvA, QNorm, KvNorm, QB, Uk, Uv, Gate, Output }

#[derive(Debug, PartialEq, Eq)]
pub struct MlaTensor {
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum MlaWeightError {
    #[error("MLA invalid {0}")]
    Geometry(&'static str),
    #[error("MLA tensor {name}: expected BF16 {shape:?}")]
    Tensor { name: String, shape: Vec<usize> },
    #[error("MLA checkpoint read {name}: {source}")]
    Read { name: String, source: anyhow::Error },
}
type Result<T> = std::result::Result<T, MlaWeightError>;

/// Replicate low-rank A inputs/norms; shard whole Q/K/V heads and o_proj
/// columns. UK is transposed per head; UV retains its checkpoint row order.
/// Only the requested operand is materialized, never an extra entire layer.
pub fn pack(cfg: &MlaGeometry, base: &str, gated: bool, operand: MlaOperand, ranks: usize,
    mut read: impl FnMut(&str) -> anyhow::Result<LoadedTensor>) -> Result<Vec<MlaTensor>> {
    if !matches!(ranks, 1 | 2) || cfg.heads == 0 || cfg.heads % ranks != 0 || cfg.hidden == 0
        || cfg.q_rank == 0 || cfg.kv_rank == 0 || cfg.nope == 0 || cfg.rope == 0 || cfg.value == 0
        || base.is_empty() || (!gated && operand == MlaOperand::Gate) {
        return Err(MlaWeightError::Geometry("head/rank/gate geometry or tensor prefix"));
    }
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
    let (heads, local, hidden, kv) = (cfg.heads, cfg.heads / ranks, cfg.hidden, cfg.kv_rank);
    let qdim = cfg.nope.checked_add(cfg.rope)
        .ok_or(MlaWeightError::Geometry("query width overflow"))?;
    let vdim = cfg.nope.checked_add(cfg.value)
        .ok_or(MlaWeightError::Geometry("KV projection width overflow"))?;
    let overflow = || MlaWeightError::Geometry("operand extent overflow");
    let kv_a_rows = kv.checked_add(cfg.rope).ok_or_else(overflow)?;
    let qkv_a_rows = cfg.q_rank.checked_add(kv_a_rows).ok_or_else(overflow)?;
    let q_rows = heads.checked_mul(qdim).ok_or_else(overflow)?;
    let kv_rows = heads.checked_mul(vdim).ok_or_else(overflow)?;
    let v_rows = heads.checked_mul(cfg.value).ok_or_else(overflow)?;
    for shape in [&[qkv_a_rows, hidden][..], &[q_rows, cfg.q_rank],
        &[kv_rows, kv], &[v_rows, hidden], &[hidden, v_rows]] {
        extent(shape)?;
    }
    let mut replicate = |leaf: &str, shape: Vec<usize>| -> Result<Vec<MlaTensor>> {
        let bytes = load(leaf, &shape)?;
        Ok((0..ranks).map(|_| MlaTensor { shape: shape.clone(), bytes: bytes.clone() }).collect())
    };
    match operand {
        MlaOperand::QNorm => replicate("q_a_layernorm", vec![cfg.q_rank]),
        MlaOperand::KvNorm => replicate("kv_a_layernorm", vec![kv]),
        MlaOperand::QkvA => {
            let mut bytes = load("q_a_proj", &[cfg.q_rank, hidden])?;
            bytes.extend(load("kv_a_proj_with_mqa", &[kv_a_rows, hidden])?);
            let shape = vec![qkv_a_rows, hidden];
            Ok((0..ranks).map(|_| MlaTensor { shape: shape.clone(), bytes: bytes.clone() }).collect())
        }
        MlaOperand::QB | MlaOperand::Gate => {
            let (leaf, width, input) = if operand == MlaOperand::QB {
                ("q_b_proj", qdim, cfg.q_rank)
            } else { ("g_proj", cfg.value, hidden) };
            let bytes = load(leaf, &[heads * width, input])?;
            let share = bytes.len() / ranks;
            Ok((0..ranks).map(|rank| MlaTensor { shape: vec![local * width, input],
                bytes: bytes[rank * share..(rank + 1) * share].to_vec() }).collect())
        }
        MlaOperand::Output => {
            let bytes = load("o_proj", &[hidden, heads * cfg.value])?;
            let row_bytes = heads * cfg.value * 2;
            let share = row_bytes / ranks;
            Ok((0..ranks).map(|rank| MlaTensor { shape: vec![hidden, local * cfg.value],
                bytes: bytes.chunks_exact(row_bytes)
                    .flat_map(|row| row[rank * share..(rank + 1) * share].iter().copied()).collect() }).collect())
        }
        MlaOperand::Uk | MlaOperand::Uv => {
            let bytes = load("kv_b_proj", &[heads * vdim, kv])?;
            let shape = if operand == MlaOperand::Uk { vec![local, kv, cfg.nope] }
                else { vec![local, cfg.value, kv] };
            let mut result = Vec::with_capacity(ranks);
            for rank in 0..ranks {
                let mut packed = Vec::with_capacity(extent(&shape)?);
                for head in rank * local..(rank + 1) * local {
                    if operand == MlaOperand::Uk {
                        for k in 0..kv {
                            for d in 0..cfg.nope {
                                let at = ((head * vdim + d) * kv + k) * 2;
                                packed.extend_from_slice(&bytes[at..at + 2]);
                            }
                        }
                    } else {
                        let start = (head * vdim + cfg.nope) * kv * 2;
                        packed.extend_from_slice(&bytes[start..start + cfg.value * kv * 2]);
                    }
                }
                result.push(MlaTensor { shape: shape.clone(), bytes: packed });
            }
            Ok(result)
        }
    }
}

