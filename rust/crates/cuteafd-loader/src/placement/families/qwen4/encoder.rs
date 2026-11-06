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
        maximum_pages: u64) -> Result<(), Refusal> {
        if self.candidates.is_empty() { return Ok(()); }
        let variable = variables.len();
        variables.push(Variable::Encoder((0..self.candidates.len()).collect()));
        for (option, encoder) in self.candidates.iter().enumerate() {
            let mut owners = Vec::new();
            match encoder.kind {
                EncoderKind::Off => {}
                EncoderKind::Rtx { gpu } if gpu < gpus => {
                    let location = Location::coordinator(gpu as u8);
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
                EncoderKind::Spark { .. } | EncoderKind::SparkIdle { .. } => {}
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
