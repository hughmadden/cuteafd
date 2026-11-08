//! Qwen's bundled vision tower as a Recipe choice: each candidate placement is
//! one option of a `Variable::Encoder`, so the shared solve admits the tower
//! beside the pool it chooses instead of a pre-solve guess at the KV target.
use crate::plan::encoder::{EncoderKind, EncoderPlacement};
use crate::placement::planner::{
    Activation, Allocation, ByteCurve, Component, Lifetime, Location, Refusal, ResidentWeight, Variable,
};
use cuteafd_core::memory_layout::Category;

/// Candidates in preference order; the solver keeps the earliest that fits at
/// its pool (pool first: a tower never shrinks the pool). Empty means no
/// encoder variable at all.
#[derive(Debug, Clone, Default)]
pub struct QwenEncoders {
    pub candidates: Vec<EncoderPlacement>,
    /// Host embedding cache, live only while a tower is active.
    pub media_cache: u64,
}

impl QwenEncoders {
    /// Append the encoder variable and its candidates' owners. Spark towers
    /// load on their Spark worker; only RTX towers own coordinator bytes.
    pub fn extend(&self, variables: &mut Vec<Variable>, components: &mut Vec<Component>, gpus: usize,
        head: crate::placement::planner::serving::BlasHead, maximum_pages: u64) -> Result<(), Refusal> {
        use crate::placement::planner::{Phase, PhaseOrder, serving::{blas_components, BlasOwners}};
        let mut blas = Vec::new();
        if head.bf16() {
            blas.push(BlasOwners { name: "Qwen owner0 target and MTP projections".into(),
                process: "coordinator".into(), thread: "qwen4/blas/serve".into(),
                location: Location::coordinator(0), gemm: head.gemm(), lt: head.lt(),
                lifetime: Lifetime { first: Phase::Warmup, last: Phase::Growth }, activation: Activation::always() });
        }
        let variable = variables.len();
        if !self.candidates.is_empty() {
            variables.push(Variable::Encoder((0..self.candidates.len()).collect()));
        }
        for (option, encoder) in self.candidates.iter().enumerate() {
            let mut owners = Vec::new();
            match encoder.kind {
                EncoderKind::Off => {}
                EncoderKind::Rtx { gpu } if gpu < gpus => {
                    let location = Location::coordinator(gpu as u8);
                    blas.push(BlasOwners { name: format!("Qwen vision candidate {option}"),
                        process: "coordinator".into(), thread: "qwen4/blas/media".into(), location,
                        gemm: true, lt: false, lifetime: Lifetime::RESIDENT,
                        activation: Activation::choice(variable, option) });
                    for (part, bytes, category) in [("weights", encoder.weights, Category::Weights),
                        ("scratch", encoder.scratch, Category::Workspace)] {
                        if bytes == 0 { continue; }
                        let owner = Allocation::buffer(format!("qwen.encoder.{option}.{part}"), location, bytes,
                            Lifetime::RESIDENT, category, format!("vision {}", if part == "weights" { "tower" } else { "scratch" }));
                        if part == "weights" {
                            owners.push(Component::resident_weight(ResidentWeight::from_allocation(&owner,
                                crate::placement::ResidentWeights(bytes)), ByteCurve::fixed(bytes, maximum_pages)));
                        }
                        owners.push(Component::allocation(owner, ByteCurve::fixed(bytes, maximum_pages), false, maximum_pages));
                    }
                }
                EncoderKind::Rtx { gpu } => return Err(Refusal::InvalidFact(format!(
                    "Qwen encoder candidate names coordinator {gpu} outside {gpus} owners"))),
                EncoderKind::Spark { .. } => {
                    for rank in encoder.spark_ranks() {
                        blas.push(BlasOwners { name: format!("Qwen vision {option}/worker{rank}"),
                            process: format!("qwen-media-worker-{rank}"), thread: "qwen4/blas/media".into(),
                            location: Location { device: crate::placement::planner::Device::Spark(rank as u8),
                                domain: crate::placement::planner::Domain::Managed },
                            gemm: true, lt: false, lifetime: Lifetime::RESIDENT,
                            activation: Activation::choice(variable, option) });
                    }
                }
                EncoderKind::SparkIdle { .. } => {}
            }
            if encoder.kind != EncoderKind::Off && self.media_cache > 0 {
                owners.push(Component::allocation(Allocation::buffer(format!("qwen.encoder.{option}.media"),
                    Location::HOST, self.media_cache, Lifetime::TARGET, Category::Embedding, "media/embedding-cache"),
                    ByteCurve::fixed(self.media_cache, maximum_pages), false, maximum_pages));
            }
            for mut owner in owners {
                owner.activation = Activation::choice(variable, option);
                components.push(owner);
            }
        }
        for arch in ["sm_120", "sm_121"] {
            let owners = blas.iter().filter(|owner|
                matches!(owner.location.device, crate::placement::planner::Device::Coordinator(_))
                    == (arch == "sm_120")).cloned().collect::<Vec<_>>();
            components.extend(blas_components(arch, &owners, maximum_pages, PhaseOrder::TARGET_THEN_DRAFT)?);
        }
        Ok(())
    }

    /// The admitted candidate, with the solver's reason when Auto chose Off.
    pub fn admitted(&self, solved: &crate::placement::planner::ResolvedPlan) -> Option<EncoderPlacement> {
        let index = solved.decisions.encoder?;
        let mut encoder = self.candidates.get(index)?.clone();
        let declined = self.candidates.iter().any(|candidate| candidate.kind != EncoderKind::Off);
        if encoder.kind == EncoderKind::Off && declined {
            let reasons = solved.encoder_diagnostics.iter().map(|(reason, _)| reason.as_str())
                .collect::<Vec<_>>().join("; ");
            encoder.reason = format!("Auto disabled image input at the pool-first solution: {reasons}; free memory or choose a smaller layout");
            encoder.shortfall = solved.encoder_diagnostics.iter().map(|(_, bytes)| *bytes)
                .filter(|bytes| *bytes > 0).min().unwrap_or(0);
        }
        Some(encoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::planner::{ComponentKind, Phase, PhaseOrder, serving::BlasHead};

    #[test]
    fn blas_target_is_root_only_and_media_is_a_distinct_thread() {
        for head in [BlasHead::Native, BlasHead::TensorLt, BlasHead::Gemm] {
            for encoder in [EncoderKind::Off, EncoderKind::Rtx { gpu: 0 },
                EncoderKind::Rtx { gpu: 1 }, EncoderKind::Spark { rank: 2 }] {
                let encoders = QwenEncoders { candidates: vec![EncoderPlacement { kind: encoder.clone(),
                    weights: 1024, scratch: 4096, replicas: vec![], reason: "test".into(), shortfall: 0 }],
                    media_cache: 0 };
                let mut components = Vec::new();
                encoders.extend(&mut vec![], &mut components, 2, head, 16).unwrap();
                let handles = |location| components.iter().filter_map(|component| match &component.kind {
                    ComponentKind::Allowance(owner) if owner.location == location
                        && owner.source.starts_with("cuBLAS handles:")
                        && owner.lifetime.contains(Phase::Ready, PhaseOrder::TARGET_THEN_DRAFT) => Some(owner.bytes),
                    _ => None,
                }).sum::<u64>();
                assert_eq!(handles(Location::coordinator(0)),
                    u64::from(head.gemm()) * (64 << 20)
                        + u64::from(encoder == EncoderKind::Rtx { gpu: 0 }) * (64 << 20));
                assert_eq!(handles(Location::coordinator(1)),
                    u64::from(encoder == EncoderKind::Rtx { gpu: 1 }) * (64 << 20));
                assert_eq!(handles(Location { device: crate::placement::planner::Device::Spark(2),
                    domain: crate::placement::planner::Domain::Managed }),
                    u64::from(encoder == EncoderKind::Spark { rank: 2 }) * 67_508_224);
            }
        }
    }
}
