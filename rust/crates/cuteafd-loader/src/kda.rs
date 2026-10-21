//! Checkpoint-name-independent KDA operand packing, shared by hybrid families.
use cuteafd_core::DType;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KdaOutputGate { LowRank, FullRank }

#[derive(Clone, Debug)]
pub struct KdaGeometry {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    pub output_gate: KdaOutputGate,
    pub o_norm_dtype: DType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KdaRole { Q, K, V, Fa, Fb, Ga, Gb, G, Beta, ConvQ, ConvK, ConvV, ALog, DtBias, ONorm, O }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KdaOperand { WIn, WFg, ConvW, ALog, DtBias, ONorm, WO }

impl KdaOperand {
    pub fn name(self) -> &'static str {
        match self {
            Self::WIn => "w_in", Self::WFg => "w_fg", Self::ConvW => "conv_w", Self::ALog => "a_log",
            Self::DtBias => "dt_bias", Self::ONorm => "o_norm", Self::WO => "w_o",
        }
    }
}

#[derive(Clone, Debug)]
pub struct KdaTensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum KdaError {
    #[error("invalid KDA geometry: {0}")]
    Geometry(String),
    #[error("KDA {role:?}: {detail}")]
    Tensor { role: KdaRole, detail: String },
    #[error("KDA tensor source: {0}")]
    Source(String),
}

fn tensor_error(role: KdaRole, detail: impl Into<String>) -> KdaError {
    KdaError::Tensor { role, detail: detail.into() }
}

/// Retain the active heads only after checking every padded F32 word is +0.
/// In particular, -0, a NaN and a finite nonzero tail are all rejected.
pub fn narrow_a_log(raw: &[u8], heads: usize) -> Result<Vec<f32>, KdaError> {
    if heads == 0 || raw.len() % 4 != 0 || raw.len() / 4 < heads {
        return Err(tensor_error(KdaRole::ALog, "expected F32 bytes with at least one entry per active head"));
    }
    let words: Vec<u32> = raw.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    if let Some(index) = words[heads..].iter().position(|word| *word != 0) {
        return Err(tensor_error(KdaRole::ALog, format!("padding entry {} is {:#010x}, not +0.0", heads + index, words[heads + index])));
    }
    let live: Vec<f32> = words[..heads].iter().map(|word| f32::from_bits(*word)).collect();
    if live.iter().any(|value| !value.is_finite()) {
        return Err(tensor_error(KdaRole::ALog, "live heads must be finite"));
    }
    Ok(live)
}

impl KdaGeometry {
    pub fn validate(&self, ranks: usize) -> Result<(), KdaError> {
        if self.hidden == 0 || self.hidden % 128 != 0 || self.head_dim != 128 || self.conv_kernel != 4
            || ranks == 0 || self.heads == 0 || self.heads % ranks != 0 || (self.heads / ranks) % 8 != 0
            || !matches!(self.o_norm_dtype, DType::Bf16 | DType::F32)
            || self.heads.checked_mul(self.head_dim).and_then(|d| d.checked_mul(self.hidden)).is_none()
        {
            return Err(KdaError::Geometry(format!("{self:?}, ranks={ranks}")));
        }
        Ok(())
    }

    pub fn roles(&self, operand: KdaOperand) -> Vec<KdaRole> {
        use KdaOperand as Op;
        use KdaRole as R;
        let full = self.output_gate == KdaOutputGate::FullRank;
        match operand {
            Op::WIn if full => vec![R::Q, R::K, R::V, R::Fa, R::G, R::Beta],
            Op::WIn => vec![R::Q, R::K, R::V, R::Fa, R::Ga, R::Beta],
            Op::WFg if full => vec![R::Fb],
            Op::WFg => vec![R::Fb, R::Gb],
            Op::ConvW => vec![R::ConvQ, R::ConvK, R::ConvV],
            Op::ALog => vec![R::ALog], Op::DtBias => vec![R::DtBias], Op::ONorm => vec![R::ONorm],
            Op::WO => vec![R::O],
        }
    }

    /// Where each rank's operand elements come from, from checkpoint headers
    /// alone (`meta` gives a role's dtype and shape): per rank, in operand order,
    /// one window per role. Input projections split rows, o_proj splits columns,
    /// low-rank inputs and the per-head norm replicate. Conv and dt_bias become
    /// F32 (from BF16 or F32 sources); o_norm retains the geometry's native
    /// dtype. A_log keeps its active heads, after `narrow_a_log` checks the
    /// padding of the whole raw vector. [`Self::pack`] executes this plan.
    pub fn plan(&self, operand: KdaOperand, ranks: usize,
        mut meta: impl FnMut(KdaRole) -> Result<(DType, Vec<usize>), KdaError>) -> Result<KdaPlan, KdaError>
    {
        self.validate(ranks)?;
        use KdaOperand as Op;
        use KdaRole as R;
        let full = self.output_gate == KdaOutputGate::FullRank;
        let d = self.heads * self.head_dim;
        let local_d = d / ranks;
        let dtype = match operand {
            Op::ConvW | Op::ALog | Op::DtBias => DType::F32,
            Op::ONorm => self.o_norm_dtype.clone(),
            _ => DType::Bf16,
        };
        let mut pieces = vec![Vec::new(); ranks];
        let mut at = vec![0usize; ranks];
        for role in self.roles(operand) {
            let (source, mut shape) = meta(role)?;
            if !matches!(source, DType::Bf16 | DType::F32) {
                return Err(tensor_error(role, format!("unsupported dtype {source:?}")));
            }
            let expected = match role {
                R::Q | R::K | R::V | R::G => vec![d, self.hidden],
                R::Fa | R::Ga => vec![self.head_dim, self.hidden],
                R::Fb | R::Gb => vec![d, self.head_dim],
                R::Beta => vec![self.heads, self.hidden],
                R::ConvQ | R::ConvK | R::ConvV => {
                    if shape == [d, 1, self.conv_kernel] { shape = vec![d, self.conv_kernel]; }
                    vec![d, self.conv_kernel]
                },
                R::ALog => shape.clone(),
                R::DtBias => vec![d], R::ONorm => vec![self.head_dim], R::O => vec![self.hidden, d],
            };
            if shape != expected {
                return Err(tensor_error(role, format!("expected {expected:?}, found {shape:?}")));
            }
            let mut rows = shape[0];
            let cols = shape.get(1).copied().unwrap_or(1);
            if role == R::ALog {
                if source != DType::F32 || shape.len() != 1 {
                    return Err(tensor_error(role, "A_log must be a raw F32 vector"));
                }
                if rows < self.heads {
                    return Err(tensor_error(role, "expected F32 bytes with at least one entry per active head"));
                }
                rows = self.heads;
            }
            if source != dtype && !(dtype == DType::F32 && source == DType::Bf16 && operand != Op::ONorm) {
                return Err(tensor_error(role, format!("expected {dtype:?}, found {source:?}")));
            }
            for rank in 0..ranks {
                let (window_rows, window) = match role {
                    R::Fa | R::Ga | R::ONorm => (0..rows, 0..cols),
                    R::O => (0..rows, rank * cols / ranks..(rank + 1) * cols / ranks),
                    _ => (rank * rows / ranks..(rank + 1) * rows / ranks, 0..cols),
                };
                let piece = KdaPiece { role, source: source.clone(), rows: window_rows, cols, window, at: at[rank] };
                at[rank] += piece.elements();
                pieces[rank].push(piece);
            }
        }
        let shape = match operand {
            Op::WIn => vec![3 * local_d + self.head_dim + if full { local_d } else { self.head_dim } + self.heads / ranks, self.hidden],
            Op::WFg => vec![if full { 1 } else { 2 }, local_d, self.head_dim],
            Op::ConvW => vec![3 * local_d, self.conv_kernel],
            Op::ALog => vec![self.heads / ranks], Op::DtBias => vec![local_d],
            Op::ONorm => vec![self.head_dim], Op::WO => vec![self.hidden, local_d],
        };
        Ok(KdaPlan { dtype, shape, ranks: pieces, a_log: (operand == Op::ALog).then_some(self.heads) })
    }

    /// Read each tensor role once and execute [`Self::plan`], preserving
    /// BF16/F32 bytes and head order. Only the operand being packed is
    /// materialized, not a second whole layer.
    pub fn pack(&self, operand: KdaOperand, ranks: usize,
        mut read: impl FnMut(KdaRole) -> Result<KdaTensor, KdaError>) -> Result<Vec<KdaTensor>, KdaError>
    {
        let mut tensors: Vec<(KdaRole, KdaTensor)> = Vec::new();
        let plan = self.plan(operand, ranks, |role| {
            let tensor = read(role)?;
            let element: usize = match tensor.dtype { DType::Bf16 => 2, DType::F32 => 4,
                _ => return Err(tensor_error(role, format!("unsupported dtype {:?}", tensor.dtype))) };
            let size = tensor.shape.iter().try_fold(element, |n, dim| n.checked_mul(*dim));
            if size != Some(tensor.bytes.len()) {
                return Err(tensor_error(role, format!("{:?} does not hold {} bytes", tensor.shape, tensor.bytes.len())));
            }
            let meta = (tensor.dtype.clone(), tensor.shape.clone());
            tensors.push((role, tensor));
            Ok(meta)
        })?;
        if let Some(heads) = plan.a_log {
            let (_, tensor) = tensors.first().ok_or_else(|| tensor_error(KdaRole::ALog, "A_log was not read"))?;
            narrow_a_log(&tensor.bytes, heads)?;
        }
        let element = if plan.dtype == DType::F32 { 4 } else { 2 };
        let mut out = Vec::with_capacity(ranks);
        for pieces in &plan.ranks {
            let elements = pieces.iter().map(KdaPiece::elements).sum::<usize>();
            let mut bytes = vec![0u8; elements * element];
            for piece in pieces {
                let (_, tensor) = tensors.iter().find(|(role, _)| *role == piece.role)
                    .ok_or_else(|| tensor_error(piece.role, "role was not read"))?;
                piece.copy(&tensor.bytes, &mut bytes, element);
            }
            out.push(KdaTensor { dtype: plan.dtype.clone(), shape: plan.shape.clone(), bytes });
        }
        Ok(out)
    }
}

/// One role's window of a rank's KDA operand: rows `rows` and element columns
/// `window` of the source viewed as `[rows, cols]` (1-D tensors are `[n, 1]`,
/// conv weights `[d, kernel]`), packed row-major at element `at`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KdaPiece {
    pub role: KdaRole,
    pub source: DType,
    pub rows: std::ops::Range<usize>,
    pub cols: usize,
    pub window: std::ops::Range<usize>,
    pub at: usize,
}

impl KdaPiece {
    pub fn elements(&self) -> usize { self.rows.len() * self.window.len() }

    /// Copies this window of `source` into `out` (operand elements of
    /// `element` bytes), widening BF16 to F32 exactly when they differ.
    fn copy(&self, source: &[u8], out: &mut [u8], element: usize) {
        let from = if self.source == DType::F32 { 4 } else { 2 };
        let mut at = self.at;
        for row in self.rows.clone() {
            for col in self.window.clone() {
                let i = (row * self.cols + col) * from;
                let dst = &mut out[at * element..(at + 1) * element];
                if from == element {
                    dst.copy_from_slice(&source[i..i + from]);
                } else {
                    dst.copy_from_slice(&(u32::from(u16::from_le_bytes([source[i], source[i + 1]])) << 16).to_le_bytes());
                }
                at += 1;
            }
        }
    }
}

/// A KDA operand from checkpoint headers: its element type and shape, each
/// rank's windows, and (A_log) the active heads whose padding must be checked.
#[derive(Clone, Debug)]
pub struct KdaPlan {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub ranks: Vec<Vec<KdaPiece>>,
    pub a_log: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(gate: KdaOutputGate) -> KdaGeometry {
        KdaGeometry { hidden: 128, heads: 16, head_dim: 128, conv_kernel: 4, output_gate: gate, o_norm_dtype: DType::F32 }
    }

    #[test]
    fn padding_checked_before_head_split() {
        let mut values: Vec<f32> = (0..96).map(|n| n as f32 / 10.).chain([0.; 32]).collect();
        let raw = |v: &[f32]| v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let g = KdaGeometry { hidden: 7168, heads: 96, ..geometry(KdaOutputGate::FullRank) };
        let packed = g.pack(KdaOperand::ALog, 2, |_| Ok(KdaTensor { dtype: DType::F32, shape: vec![128], bytes: raw(&values) })).unwrap();
        assert_eq!(packed[1].bytes, raw(&values[48..96]));
        for invalid in [-0., 1., f32::NAN, f32::INFINITY] {
            values[127] = invalid;
            assert!(narrow_a_log(&raw(&values), 96).is_err());
        }
        assert!(narrow_a_log(&raw(&values[..95]), 96).is_err());
    }

    #[test]
    fn low_rank_layout_matches_legacy_concatenate_then_slice() {
        let g = geometry(KdaOutputGate::LowRank);
        let d = g.heads * g.head_dim;
        let mut legacy = vec![Vec::new(); 2];
        let packed = g.pack(KdaOperand::WIn, 2, |role| {
            let rows = match role { KdaRole::Fa | KdaRole::Ga => 128, KdaRole::Beta => 16, _ => d };
            let bytes: Vec<u8> = (0..rows * 128).flat_map(|n| ((n + role as usize * 17) as u16).to_le_bytes()).collect();
            for (rank, dest) in legacy.iter_mut().enumerate() {
                if matches!(role, KdaRole::Fa | KdaRole::Ga) { dest.extend_from_slice(&bytes); }
                else { dest.extend_from_slice(&bytes[rank * bytes.len() / 2..(rank + 1) * bytes.len() / 2]); }
            }
            Ok(KdaTensor { dtype: DType::Bf16, shape: vec![rows, 128], bytes })
        }).unwrap();
        assert_eq!(packed[0].bytes, legacy[0]);
        assert_eq!(packed[1].bytes, legacy[1]);
    }

    #[test]
    fn all_glmf_operands_preserve_legacy_bytes() {
        use KdaOperand as Op;
        use KdaRole as R;
        let g = KdaGeometry { o_norm_dtype: DType::Bf16, ..geometry(KdaOutputGate::LowRank) };
        let d = g.heads * g.head_dim;
        for ranks in [1, 2] {
            for native_conv in [DType::Bf16, DType::F32] {
                for operand in [Op::WIn, Op::WFg, Op::ConvW, Op::ALog, Op::DtBias, Op::ONorm, Op::WO] {
                    let mut legacy = vec![Vec::new(); ranks];
                    let packed = g.pack(operand, ranks, |role| {
                        let shape = match role {
                            R::Q | R::K | R::V => vec![d, g.hidden],
                            R::Fa | R::Ga => vec![128, g.hidden],
                            R::Fb | R::Gb => vec![d, 128],
                            R::Beta => vec![g.heads, g.hidden],
                            R::ConvQ | R::ConvK | R::ConvV => vec![d, 1, 4],
                            R::ALog => vec![g.heads], R::DtBias => vec![d],
                            R::ONorm => vec![128], R::O => vec![g.hidden, d],
                            R::G => unreachable!(),
                        };
                        let dtype = match role {
                            R::ALog | R::DtBias => DType::F32,
                            R::ConvQ | R::ConvK | R::ConvV => native_conv.clone(),
                            _ => DType::Bf16,
                        };
                        let count = shape.iter().product::<usize>();
                        let values: Vec<u16> = (0..count).map(|i| 0x3f00 + ((i + role as usize * 17) % 128) as u16).collect();
                        let widened: Vec<u8> = values.iter().flat_map(|v| f32::from_bits(u32::from(*v) << 16).to_le_bytes()).collect();
                        let bytes = if dtype == DType::F32 { widened.clone() }
                            else { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
                        let expected = if matches!(operand, Op::ConvW | Op::ALog | Op::DtBias) { &widened } else { &bytes };
                        for (rank, dest) in legacy.iter_mut().enumerate() {
                            match role {
                                R::Fa | R::Ga | R::ONorm => dest.extend_from_slice(expected),
                                R::O => {
                                    for row in expected.chunks_exact(d * 2) {
                                        dest.extend_from_slice(&row[rank * d * 2 / ranks..(rank + 1) * d * 2 / ranks]);
                                    }
                                }
                                _ => dest.extend_from_slice(&expected[rank * expected.len() / ranks..(rank + 1) * expected.len() / ranks]),
                            }
                        }
                        Ok(KdaTensor { dtype, shape, bytes })
                    }).unwrap();
                    for (rank, tensor) in packed.iter().enumerate() {
                        assert_eq!(tensor.bytes, legacy[rank], "{operand:?}, ranks={ranks}, native_conv={native_conv:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn norm_preserves_f32_bits_and_replicates() {
        let g = geometry(KdaOutputGate::FullRank);
        let bytes: Vec<u8> = (0..128).flat_map(|i| (1.0 + i as f32 / 12345.).to_le_bytes()).collect();
        let packed = g.pack(KdaOperand::ONorm, 2, |_| Ok(KdaTensor { dtype: DType::F32, shape: vec![128], bytes: bytes.clone() })).unwrap();
        assert!(packed.iter().all(|t| t.dtype == DType::F32 && t.bytes == bytes));
        assert!(g.pack(KdaOperand::ONorm, 2, |_| Ok(KdaTensor { dtype: DType::Bf16, shape: vec![128], bytes: vec![0; 256] })).is_err());
    }
}
