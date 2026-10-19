//! Checkpoint-native BF16 dense/shared FFN slices and F32 residual score weights.
use super::config::KimiK3Config;
use crate::tensors::LoadedTensor;
use cuteafd_core::DType;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum W8WeightError {
    #[error("Kimi W8 invalid {0}")]
    Invalid(&'static str),
    #[error("Kimi W8 tensor {name}: expected BF16 {shape:?}")]
    Tensor { name: String, shape: Vec<usize> },
    #[error("Kimi W8 checkpoint read {name}: {source}")]
    Read { name: String, source: anyhow::Error },
}
type Result<T> = std::result::Result<T, W8WeightError>;

/// One intermediate slice. Both projections and the Situ output materialize
/// BF16. The engine performs a TP partial-sum meet after the down projection.
#[derive(Debug, PartialEq, Eq)]
pub struct FfnSlice {
    pub hidden: usize,
    pub intermediate: usize,
    pub gate_up: Vec<u8>,
    pub down: Vec<u8>,
}

pub fn pack_ffn(cfg: &KimiK3Config, layer: usize, ranks: usize,
    mut read: impl FnMut(&str) -> anyhow::Result<LoadedTensor>) -> Result<Vec<FfnSlice>> {
    let dense = *cfg.dense.get(layer).ok_or(W8WeightError::Invalid("FFN layer"))?;
    if !matches!(ranks, 1 | 2) { return Err(W8WeightError::Invalid("FFN coordinator ranks")); }
    let intermediate = if dense { cfg.dense_intermediate } else { cfg.moe_intermediate * cfg.shared_experts };
    let prefix = if dense { format!("language_model.model.layers.{layer}.mlp") }
        else { format!("language_model.model.layers.{layer}.block_sparse_moe.shared_experts") };
    let mut load = |leaf: &str, shape: &[usize]| -> Result<LoadedTensor> {
        let name = format!("{prefix}.{leaf}_proj.weight");
        let tensor = read(&name).map_err(|source| W8WeightError::Read { name: name.clone(), source })?;
        check_bf16(&tensor, shape)?;
        Ok(tensor)
    };
    let gate = load("gate", &[intermediate, cfg.hidden])?;
    let up = load("up", &[intermediate, cfg.hidden])?;
    let down = load("down", &[cfg.hidden, intermediate])?;
    split_ffn(cfg.hidden, intermediate, ranks, &gate.bytes, &up.bytes, &down.bytes)
}

/// Original BF16 norm and projection are promoted before their F32 product,
/// matching `_apply_attn_res`; never multiply them in BF16 or normalize here.
pub fn residual_score_weight(norm: &LoadedTensor, projection: &LoadedTensor, hidden: usize) -> Result<Vec<u8>> {
    check_bf16(norm, &[hidden])?;
    check_bf16(projection, &[1, hidden])?;
    Ok(norm.bytes.chunks_exact(2).zip(projection.bytes.chunks_exact(2))
        .flat_map(|(n, p)| {
            let n = f32::from_bits((u16::from_le_bytes([n[0], n[1]]) as u32) << 16);
            let p = f32::from_bits((u16::from_le_bytes([p[0], p[1]]) as u32) << 16);
            (n * p).to_le_bytes()
        }).collect())
}

fn check_bf16(tensor: &LoadedTensor, shape: &[usize]) -> Result<()> {
    let bytes = shape.iter().try_fold(2usize, |n, &v| n.checked_mul(v));
    if tensor.info.dtype != DType::Bf16 || tensor.info.shape != shape || bytes != Some(tensor.bytes.len()) {
        return Err(W8WeightError::Tensor { name: tensor.info.name.clone(), shape: shape.to_vec() });
    }
    Ok(())
}

fn split_ffn(hidden: usize, intermediate: usize, ranks: usize, gate: &[u8], up: &[u8], down: &[u8])
    -> Result<Vec<FfnSlice>> {
    if hidden == 0 || intermediate == 0 || ranks == 0 || intermediate % ranks != 0 {
        return Err(W8WeightError::Invalid("FFN split geometry"));
    }
    let expected = hidden.checked_mul(intermediate).and_then(|n| n.checked_mul(2))
        .ok_or(W8WeightError::Invalid("FFN extent overflow"))?;
    if [gate.len(), up.len(), down.len()].iter().any(|&n| n != expected) {
        return Err(W8WeightError::Invalid("FFN tensor byte extent"));
    }
    let width = intermediate / ranks;
    let bytes = expected / ranks;
    Ok((0..ranks).map(|rank| {
        let mut gate_up = Vec::with_capacity(2 * bytes);
        gate_up.extend_from_slice(&gate[rank * bytes..(rank + 1) * bytes]);
        gate_up.extend_from_slice(&up[rank * bytes..(rank + 1) * bytes]);
        let mut sliced_down = Vec::with_capacity(bytes);
        for row in down.chunks_exact(intermediate * 2) {
            sliced_down.extend_from_slice(&row[rank * width * 2..(rank + 1) * width * 2]);
        }
        FfnSlice { hidden, intermediate: width, gate_up, down: sliced_down }
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(v: impl IntoIterator<Item = u16>) -> Vec<u8> {
        v.into_iter().flat_map(u16::to_le_bytes).collect()
    }

    fn tensor(name: &str, shape: Vec<usize>, bytes: Vec<u8>) -> LoadedTensor {
        LoadedTensor { info: cuteafd_core::TensorInfo { name: name.into(), file: "test".into(),
            dtype: DType::Bf16, shape, byte_offset: 0, byte_length: bytes.len() as u64,
            role: cuteafd_core::TensorRole::Other, layer_id: None, expert_id: None,
            is_quantization_metadata: false }, source_path: "test".into(), bytes,
            elapsed_micros: 0, sha256: String::new() }
    }

    #[test]
    fn score_product_is_f32_not_bf16_and_rejects_wrong_tensor_contracts() {
        let norm = tensor("norm", vec![2], values([0x3f81, 0xbf81]));
        let projection = tensor("projection", vec![1, 2], values([0x3f81, 0x3f81]));
        let packed = residual_score_weight(&norm, &projection, 2).unwrap();
        let exact = f32::from_bits(0x3f81_0000).powi(2);
        assert_eq!(packed, [exact.to_le_bytes(), (-exact).to_le_bytes()].concat());
        assert_ne!(exact, super::super::reference::bf16(exact));
        let mut bad = norm.clone(); bad.info.dtype = DType::F32;
        assert!(residual_score_weight(&bad, &projection, 2).is_err());
        let mut bad = projection.clone(); bad.info.shape = vec![2];
        assert!(residual_score_weight(&norm, &bad, 2).is_err());
        let mut bad = norm.clone(); bad.bytes.pop();
        assert!(residual_score_weight(&bad, &projection, 2).is_err());
    }

    #[test]
    fn checkpoint_reader_uses_dense_or_shared_names_before_tp_slicing() {
        let mut cfg = KimiK3Config::from_hf(&crate::plan::testing::kimi_k3_config()).unwrap();
        cfg.hidden = 4; cfg.dense_intermediate = 8; cfg.moe_intermediate = 4; cfg.shared_experts = 2;
        for (layer, prefix) in [(0, "mlp"), (1, "block_sparse_moe.shared_experts")] {
            let slices = pack_ffn(&cfg, layer, 2, |name| {
                assert!(name.starts_with(&format!("language_model.model.layers.{layer}.{prefix}.")));
                let shape = if name.ends_with("down_proj.weight") { vec![4, 8] } else { vec![8, 4] };
                Ok(tensor(name, shape, vec![0; 64]))
            }).unwrap();
            assert_eq!(slices.len(), 2);
            assert_eq!(slices[0].intermediate, 4);
        }
        assert!(pack_ffn(&cfg, 0, 3, |_| unreachable!()).is_err());
        assert!(pack_ffn(&cfg, cfg.layers, 2, |_| unreachable!()).is_err());
    }

    #[test]
    fn tp_intermediate_slices_keep_gate_first_and_slice_down_columns() {
        let gate = values(0..32);
        let up = values(100..132);
        let down = values(200..232);
        let slices = split_ffn(4, 8, 2, &gate, &up, &down).unwrap();
        assert_eq!(slices[0].intermediate, 4);
        assert_eq!(slices[1].gate_up, values((16..32).chain(116..132)));
        assert_eq!(slices[0].gate_up, values((0..16).chain(100..116)));
        assert_eq!(slices[1].down, values((204..208).chain(212..216).chain(220..224).chain(228..232)));
        let whole = split_ffn(4, 8, 1, &gate, &up, &down).unwrap();
        assert_eq!(whole[0].gate_up, [gate.as_slice(), up.as_slice()].concat());
        assert_eq!(whole[0].down, down);
        assert!(split_ffn(4, 8, 3, &gate, &up, &down).is_err());
        assert!(split_ffn(4, 8, 2, &gate[..60], &up, &down).is_err());
    }
}
