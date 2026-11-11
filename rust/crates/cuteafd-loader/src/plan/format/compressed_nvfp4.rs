//! Compressed-tensors NVFP4 header semantics, independent of model family.
use super::*;

pub(super) fn compressed_nvfp4(members: &BTreeMap<String, &CheckpointTensor>) -> Result<QuantOperand, Malformed> {
    let anchor = members.values().next().expect("nonempty compressed-tensors group");
    let fail = |tensor: &str, reason: &str| Malformed { tensor: tensor.into(), reason: reason.into() };
    let required = ["weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"];
    for suffix in required {
        if !members.contains_key(suffix) {
            return Err(fail(&anchor.meta.name, &format!("compressed NVFP4 missing {suffix}")));
        }
    }
    if let Some((_, tensor)) = members.iter().find(|(suffix, _)| !required.contains(&suffix.as_str())) {
        return Err(fail(&tensor.meta.name, "unexpected compressed NVFP4 member"));
    }
    let weight = &members["weight_packed"].meta;
    let [n, packed] = weight.shape.as_slice() else {
        return Err(fail(&weight.name, "compressed NVFP4 requires a U8 [N, K/2] matrix"));
    };
    let k = packed.checked_mul(2).ok_or_else(|| fail(&weight.name, "compressed NVFP4 width overflow"))?;
    if weight.dtype != DType::U8 || *n == 0 || k == 0 || k % 16 != 0 {
        return Err(fail(&weight.name, "compressed NVFP4 requires nonempty U8 [N, K/2], K divisible by 16"));
    }
    let scale = &members["weight_scale"].meta;
    if scale.dtype != DType::F8E4M3 || scale.shape != [*n, k / 16] {
        return Err(fail(&scale.name, "compressed NVFP4 scales must be E4M3 [N, K/16]"));
    }
    for suffix in ["weight_global_scale", "input_global_scale"] {
        let global = &members[suffix].meta;
        if global.dtype != DType::F32 || !matches!(global.shape.as_slice(), [] | [1]) {
            return Err(fail(&global.name, "compressed NVFP4 global scale must be one FP32 value"));
        }
    }
    let divisor = TensorScale { encoding: ScaleEncoding::F32, direction: ScaleDirection::Divide };
    let mut operand = QuantOperand::plain(Encoding::E2m1, weight);
    operand.logical = vec![*n, k];
    operand.scale = Some(BlockScale { tensor: "weight_scale".into(), encoding: ScaleEncoding::E4m3,
        direction: ScaleDirection::Multiply, rows: RowTiling::Uniform { rows: 1 }, cols: 16, grid: scale.shape.clone() });
    operand.global_scale = Some(divisor);
    operand.input_scale = Some(divisor);
    Ok(operand)
}
