//! Shared header validation for checkpoint-native dense coordinator weights,
//! and the per-rank operand layout families build from it.
use crate::plan::checkpoint::{Checkpoint, CheckpointTensor};
use crate::serving_capacity::CacheGeometryError;
use anyhow::{ensure, Context};
use cuteafd_core::DType;

/// Bytes per element of a resident coordinator operand: BF16, FP32, E4M3, or
/// U8 (UE8M0 block-scale codes beside E4M3 payloads, MXFP8).
pub fn element_bytes(dtype: &DType) -> Option<u64> {
    match dtype {
        DType::Bf16 => Some(2),
        DType::F32 => Some(4),
        DType::F8E4M3 | DType::U8 => Some(1),
        _ => None,
    }
}

pub(crate) fn tensor<'a>(
    checkpoint: &'a Checkpoint,
    name: &str,
    shape: &[usize],
    dtypes: &[DType],
) -> Result<&'a CheckpointTensor, CacheGeometryError> {
    let at = checkpoint
        .tensors
        .binary_search_by(|t| t.meta.name.as_str().cmp(name))
        .map_err(|_| CacheGeometryError::ResidentTensor {
            name: name.into(),
            what: "missing from checkpoint headers".into(),
        })?;
    let t = &checkpoint.tensors[at];
    if t.meta.shape != shape || !dtypes.contains(&t.meta.dtype) {
        return Err(CacheGeometryError::ResidentTensor {
            name: name.into(),
            what: format!(
                "expected {dtypes:?} {shape:?}, found {:?} {:?}",
                t.meta.dtype, t.meta.shape
            ),
        });
    }
    let element_bytes = element_bytes(&t.meta.dtype).ok_or_else(|| CacheGeometryError::ResidentTensor {
        name: name.into(),
        what: "resident size requires BF16, FP32, E4M3 or U8 scale codes".into(),
    })?;
    let expected = shape
        .iter()
        .try_fold(element_bytes, |bytes: u64, &dim| {
            bytes.checked_mul(dim as u64)
        })
        .ok_or(CacheGeometryError::Overflow("native source tensor extent"))?;
    if t.meta.byte_length != expected {
        return Err(CacheGeometryError::ResidentTensor {
            name: name.into(),
            what: format!("expected {expected} bytes, found {}", t.meta.byte_length),
        });
    }
    Ok(t)
}

/// Exact source views and conservative replicated resident storage. TP slicing,
/// kernel packing and runtime reservations are separate contracts; this does
/// not authorize serving or infer missing workspace/graph costs as zero.
#[derive(Debug, Clone)]
pub struct NativeDenseLayout {
    pub tensors: Vec<CheckpointTensor>,
    pub resident_bytes: u64,
}

impl NativeDenseLayout {
    pub fn from_headers(
        checkpoint: &Checkpoint,
        declarations: impl IntoIterator<Item = (String, Vec<usize>, DType)>,
    ) -> Result<Self, CacheGeometryError> {
        let mut tensors = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        let mut resident_bytes = 0u64;
        for (name, shape, dtype) in declarations {
            if !names.insert(name.clone()) || shape.is_empty() || shape.contains(&0) {
                return Err(CacheGeometryError::ResidentTensor {
                    name,
                    what: "duplicate or empty native declaration".into(),
                });
            }
            let source = tensor(checkpoint, &name, &shape, &[dtype])?;
            source
                .meta
                .byte_offset
                .checked_add(source.meta.byte_length)
                .ok_or(CacheGeometryError::Overflow("native source offset"))?;
            resident_bytes = resident_bytes
                .checked_add(source.meta.byte_length.max(256))
                .ok_or(CacheGeometryError::Overflow("native resident weights"))?;
            tensors.push(source.clone());
        }
        if tensors.is_empty() {
            return Err(CacheGeometryError::ResidentTensor {
                name: "native dense layout".into(),
                what: "empty declaration".into(),
            });
        }
        Ok(Self {
            tensors,
            resident_bytes,
        })
    }

    /// Weight-only fit with explicit reservations supplied by the caller.
    /// Serial host source staging is not a second device weight copy.
    pub fn check_weight_budget(
        &self,
        ceiling: u64,
        reservations: &[u64],
    ) -> Result<u64, CacheGeometryError> {
        let required = reservations
            .iter()
            .try_fold(self.resident_bytes, |sum, bytes| sum.checked_add(*bytes))
            .ok_or(CacheGeometryError::Overflow(
                "native weights and reservations",
            ))?;
        if required > ceiling {
            return Err(CacheGeometryError::ResidentTensor {
                name: "native dense layout".into(),
                what: format!(
                    "weights plus supplied reservations need {required} bytes, ceiling {ceiling}"
                ),
            });
        }
        Ok(required)
    }
}

/// How a rank of a head split holds an operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightSlice {
    Replicated,
    /// Output rows (the rank's heads).
    Rows,
    /// Input columns (an output projection over the rank's heads).
    Cols,
}

/// One resident operand: its checkpoint sources in order, its resident
/// precision and its shape on the rank.
#[derive(Debug, Clone)]
pub struct WeightOperand {
    /// A fused operand (`gate_up_proj`, `kv_proj`) holds the first source's
    /// rows, then the second's; scale tensors fuse the same way.
    pub name: String,
    pub sources: Vec<CheckpointTensor>,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub slice: WeightSlice,
    pub allocation_bytes: u64,
}

/// One coordinator GPU's resident dense operands, derived before CUDA
/// initialization. Routed experts and separately owned stages are not in it.
#[derive(Debug, Clone)]
pub struct RankWeightLayout {
    pub rank: usize,
    pub ranks: usize,
    pub operands: Vec<WeightOperand>,
    pub resident_bytes: u64,
}

/// How a rank holds one native tensor: its slice over `ranks` GPUs (1: whole)
/// in `dtype` (the source's, or BF16 the loader narrows an FP32 source to).
#[derive(Debug, Clone, PartialEq)]
pub struct Hold {
    pub slice: WeightSlice,
    pub ranks: usize,
    pub dtype: DType,
}

impl Hold {
    /// The whole tensor in its checkpoint precision.
    pub fn whole(source: &CheckpointTensor) -> Self {
        Self { slice: WeightSlice::Replicated, ranks: 1, dtype: source.meta.dtype.clone() }
    }
}

/// A fused pair of name infixes `(first, second, fused)`: gate rows then up
/// rows, the SwiGLU operand order.
pub const GATE_UP: (&str, &str, &str) = (".gate_proj.", ".up_proj.", ".gate_up_proj.");

impl RankWeightLayout {
    /// `weights` on `rank` of `ranks` GPUs. `hold` places each tensor (`None`:
    /// this rank holds none of it). Each `fuse` pair lands the second
    /// tensor's rows after the first's in one operand named by the fused
    /// infix. Allocations are the resident extents, at least 256 bytes.
    pub fn from_native(weights: &NativeDenseLayout, rank: usize, ranks: usize,
        fuse: &[(&str, &str, &str)],
        mut hold: impl FnMut(&CheckpointTensor) -> anyhow::Result<Option<Hold>>) -> anyhow::Result<Self> {
        ensure!(matches!(ranks, 1 | 2) && rank < ranks, "coordinator rank {rank} of {ranks}");
        let sources: std::collections::BTreeMap<_, _> = weights.tensors.iter()
            .map(|t| (t.meta.name.as_str(), t)).collect();
        let mut operands = Vec::new();
        let mut resident_bytes = 0u64;
        for source in &weights.tensors {
            let name = &source.meta.name;
            if fuse.iter().any(|(_, second, _)| name.contains(second)) { continue; }
            let Some(Hold { slice, ranks: split, dtype }) = hold(source)? else { continue };
            let mut selected = vec![source.clone()];
            let mut shape = source.meta.shape.clone();
            let mut output = name.clone();
            if let Some((first, second, fused)) = fuse.iter().find(|(first, ..)| name.contains(first)) {
                let (base, suffix) = name.split_once(first).context("fused infix")?;
                let partner = sources.get(format!("{base}{second}{suffix}").as_str())
                    .with_context(|| format!("{name}: no {second} partner"))?;
                ensure!(partner.meta.shape == shape && partner.meta.dtype == source.meta.dtype,
                    "{name}: fused operands disagree");
                shape[0] = shape[0].checked_mul(2).context("fused rows overflow")?;
                selected.push((*partner).clone());
                output = format!("{base}{fused}{suffix}");
            }
            if slice != WeightSlice::Replicated {
                ensure!(selected.len() == 1 && shape.len() == 2, "{name}: only a whole matrix splits");
                let axis = if slice == WeightSlice::Rows { 0 } else { 1 };
                ensure!(split > 0 && shape[axis] % split == 0, "{name}: uneven split over {split} GPUs");
                shape[axis] /= split;
            }
            let element = element_bytes(&dtype)
                .with_context(|| format!("{name}: unsupported coordinator operand {dtype:?}"))?;
            let allocation_bytes = shape.iter().try_fold(element, |n, d| n.checked_mul(*d as u64))
                .context("operand size overflow")?.max(256);
            resident_bytes = resident_bytes.checked_add(allocation_bytes).context("rank weight size overflow")?;
            operands.push(WeightOperand { name: output, sources: selected, dtype, shape, slice, allocation_bytes });
        }
        Ok(Self { rank, ranks, operands, resident_bytes })
    }

    /// Only the operands `keep` accepts, the resident total recharged.
    pub fn retain(mut self, keep: impl FnMut(&WeightOperand) -> bool) -> anyhow::Result<Self> {
        self.operands.retain(keep);
        self.resident_bytes = self.operands.iter().try_fold(0u64, |n, o| n.checked_add(o.allocation_bytes))
            .context("rank weight size overflow")?;
        Ok(self)
    }

    /// The caller supplies every non-weight cost; nothing missing is inferred
    /// as zero.
    pub fn check_budget(&self, available: u64, reservations: &[u64]) -> anyhow::Result<u64> {
        let total = reservations.iter().try_fold(self.resident_bytes, |n, v| n.checked_add(*v))
            .context("coordinator reservation overflow")?;
        ensure!(total <= available, "coordinator weights and reservations need {total} bytes, available {available}");
        Ok(total)
    }
}
