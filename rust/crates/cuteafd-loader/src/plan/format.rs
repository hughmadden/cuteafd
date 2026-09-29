//! Storage-format detection for one logical weight (a group of tensors that
//! share a stem, such as `x.weight` + `x.weight_scale_inv`, or the EXL3
//! `x.trellis`/`x.suh`/`x.svh`/`x.mcg` set).
use cuteafd_core::DType;
use serde::Serialize;
use std::collections::BTreeMap;

use super::checkpoint::CheckpointTensor;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case", tag = "format")]
pub enum WeightFormat {
    Bf16,
    F16,
    F32,
    /// E4M3 weights with a block scale; `block` is (rows, cols) per scale.
    Fp8Block { block: (usize, usize) },
    /// E4M3 weights with 128x128 blocks whose row blocks restart at segment
    /// boundaries (per KV head, or per checkpoint TP shard of a fused
    /// projection), so the grid has more rows than `ceil(rows / 128)`.
    Fp8SegmentedBlock { block: (usize, usize) },
    /// E4M3 weights with one scale per output row or tensor.
    Fp8PerChannel,
    /// Packed E2M1 with E8M0 group scales (OCP MX, DeepSeek FP4 experts).
    Mxfp4 { group: usize },
    /// ModelOpt NVFP4: packed E2M1, E4M3 per-16 scales, FP32 global scale.
    Nvfp4,
    /// EXL3 trellis with `bits` per weight.
    Exl3 { bits: usize },
    /// Integer lookup data (hash-route tables, offsets).
    Int,
    /// Anything else, described by its dtype set.
    Unknown { dtypes: String },
}

impl WeightFormat {
    pub fn label(&self) -> String {
        match self {
            WeightFormat::Bf16 => "bf16".into(),
            WeightFormat::F16 => "f16".into(),
            WeightFormat::F32 => "f32".into(),
            WeightFormat::Fp8Block { block } => format!("fp8-block{}x{}", block.0, block.1),
            WeightFormat::Fp8SegmentedBlock { block } => format!("fp8-block{}x{}-segmented", block.0, block.1),
            WeightFormat::Fp8PerChannel => "fp8-channel".into(),
            WeightFormat::Mxfp4 { group } => format!("mxfp4-g{group}"),
            WeightFormat::Nvfp4 => "nvfp4".into(),
            WeightFormat::Exl3 { bits } => format!("exl3-k{bits}"),
            WeightFormat::Int => "int".into(),
            WeightFormat::Unknown { dtypes } => format!("unknown({dtypes})"),
        }
    }
}

/// Suffixes that belong to the logical weight named by the stem before them.
const SUFFIXES: &[&str] = &[
    "weight_scale_inv",
    "weight_scale_2",
    "weight_scale",
    "input_scale",
    "weight",
    "scale",
    "bias",
    "trellis",
    "suh",
    "svh",
    "mcg",
    "mul1",
    "su",
    "sv",
];

/// Splits `a.b.weight_scale_inv` into (`a.b`, `weight_scale_inv`). Names with no
/// known suffix are their own stem.
pub fn split_stem(name: &str) -> (&str, &str) {
    if let Some((stem, suffix)) = name.rsplit_once('.') {
        if SUFFIXES.contains(&suffix) {
            return (stem, suffix);
        }
    }
    (name, "")
}

/// Groups tensors by stem, keeping each group's members keyed by suffix.
pub fn group_by_stem(tensors: &[CheckpointTensor]) -> BTreeMap<String, BTreeMap<String, &CheckpointTensor>> {
    let mut groups: BTreeMap<String, BTreeMap<String, &CheckpointTensor>> = BTreeMap::new();
    for tensor in tensors {
        let (stem, suffix) = split_stem(&tensor.meta.name);
        groups
            .entry(stem.to_owned())
            .or_default()
            .insert(suffix.to_owned(), tensor);
    }
    groups
}

fn dtype_label(dtype: &DType) -> String {
    match dtype {
        DType::Unknown(value) => value.clone(),
        other => format!("{other:?}").to_ascii_lowercase(),
    }
}

/// The block edge that tiles `dim` into `scales` blocks, allowing a partial
/// last block (576 rows in 128-row blocks need 5 scales). Falls back to the
/// rounded-up quotient for layouts that are not power-of-two blocked.
fn block_edge(dim: usize, scales: usize) -> usize {
    if scales <= 1 {
        return dim;
    }
    [1usize, 16, 32, 64, 128, 256, 512]
        .into_iter()
        .find(|edge| dim.div_ceil(*edge) == scales)
        .unwrap_or_else(|| dim.div_ceil(scales))
}

/// Detects the storage format of one logical weight.
pub fn detect(members: &BTreeMap<String, &CheckpointTensor>) -> WeightFormat {
    let get = |suffix: &str| members.get(suffix).map(|tensor| &tensor.meta);
    if let Some(trellis) = get("trellis") {
        // EXL3 trellis tensors are [K/16, N/16, 16 * bits] of int16.
        let bits = trellis.shape.last().copied().unwrap_or(0) / 16;
        return WeightFormat::Exl3 { bits };
    }
    let weight = get("weight").or_else(|| members.values().next().map(|tensor| &tensor.meta));
    let Some(weight) = weight else {
        return WeightFormat::Unknown { dtypes: String::new() };
    };
    match &weight.dtype {
        DType::F8E4M3 | DType::F8E5M2 => {
            let scale = get("weight_scale_inv").or_else(|| get("scale")).or_else(|| get("weight_scale"));
            match scale {
                Some(scale) if scale.shape.len() == 2 && weight.shape.len() == 2 => {
                    let rows = block_edge(weight.shape[0], scale.shape[0]);
                    let cols = block_edge(weight.shape[1], scale.shape[1]);
                    let uniform = weight.shape[0].div_ceil(128);
                    if cols == weight.shape[1] {
                        WeightFormat::Fp8PerChannel
                    } else if cols == 128 && scale.shape[0] > uniform && scale.shape[0] < weight.shape[0].div_ceil(64) {
                        // More 128-row blocks than rows / 128: blocks restart per segment.
                        WeightFormat::Fp8SegmentedBlock { block: (128, 128) }
                    } else {
                        WeightFormat::Fp8Block { block: (rows, cols) }
                    }
                }
                _ => WeightFormat::Fp8PerChannel,
            }
        }
        DType::U8 | DType::I8 | DType::F4 => {
            if get("weight_scale_2").is_some() {
                return WeightFormat::Nvfp4;
            }
            let scale = get("scale").or_else(|| get("weight_scale")).or_else(|| get("weight_scale_inv"));
            match scale {
                Some(scale) if matches!(scale.dtype, DType::F8E8M0 | DType::U8) && !weight.shape.is_empty() => {
                    // Packed two values per byte along the last axis.
                    let logical_k = weight.shape.last().copied().unwrap_or(0) * 2;
                    let groups = scale.shape.last().copied().unwrap_or(1).max(1);
                    WeightFormat::Mxfp4 { group: logical_k / groups }
                }
                Some(scale) if matches!(scale.dtype, DType::F8E4M3) => WeightFormat::Nvfp4,
                _ => WeightFormat::Unknown {
                    dtypes: members
                        .values()
                        .map(|tensor| dtype_label(&tensor.meta.dtype))
                        .collect::<Vec<_>>()
                        .join("+"),
                },
            }
        }
        DType::Bf16 => WeightFormat::Bf16,
        DType::F16 => WeightFormat::F16,
        DType::F32 => WeightFormat::F32,
        DType::I16 | DType::I32 | DType::I64 => WeightFormat::Int,
        _ => WeightFormat::Unknown {
            dtypes: members
                .values()
                .map(|tensor| dtype_label(&tensor.meta.dtype))
                .collect::<Vec<_>>()
                .join("+"),
        },
    }
}
