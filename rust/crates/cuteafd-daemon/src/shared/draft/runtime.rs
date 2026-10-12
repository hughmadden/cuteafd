//! Resource-priced selection and agreed-copy priors for family adapters.
use super::{copy::COPY_EXTENSION, resource};
use crate::shared::draft_policy::DraftHistory;
use anyhow::Result;
use cuteafd_core::{DraftPolicy, PolicySeed};

pub(crate) struct SharedPolicy {
    pub policy: DraftPolicy,
    prior: resource::Prior,
}

impl SharedPolicy {
    pub fn new(homes: &[resource::LayerHome], experts: usize, topk: usize, rows: usize,
        width: usize, prior: resource::Prior, seed: PolicySeed) -> Result<Self> {
        let geometry = resource::geometry(homes, experts, topk, rows, width, COPY_EXTENSION)?;
        let mut policy = DraftPolicy::seeded(geometry, seed)?;
        super::apply_length_rule(&mut policy);
        Ok(Self { policy, prior })
    }

    pub fn prior(&self, history: &DraftHistory, evidence: &resource::DraftEvidence<'_>, extension: usize) -> Vec<f64> {
        let mut prior = resource::prior(&self.prior, history, evidence);
        let width = match evidence {
            resource::DraftEvidence::Selector(features) => features.len(),
            resource::DraftEvidence::Head(head) => head.len(),
        };
        if width > 0 && prior.len() == width {
            prior.extend((0..extension.min(COPY_EXTENSION)).map(resource::tail_prior));
        }
        prior
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priors_price_the_entire_agreed_tail_for_every_neural_window() {
        let history = DraftHistory::default();
        for width in [1usize, 2, 3, 7, 8, 16] {
            let shared = SharedPolicy::new(&[resource::LayerHome::Remote { slice_bytes: 4e6 }],
                288, 8, 64, width, resource::Prior::Head, resource::seed()).unwrap();
            let head = vec![0.95; width];
            for extension in [0, 1, COPY_EXTENSION, COPY_EXTENSION + 1] {
                let prior = shared.prior(&history, &resource::DraftEvidence::Head(&head), extension);
                assert_eq!(prior.len(), width + extension.min(COPY_EXTENSION));
                assert!(prior[width..].iter().enumerate().all(|(i, &p)| p == resource::tail_prior(i)));
                assert_eq!(shared.policy.geometry().max_positions, width + COPY_EXTENSION);
            }
            assert!(shared.prior(&history, &resource::DraftEvidence::Head(&[]), 32).is_empty());
        }
    }
}
