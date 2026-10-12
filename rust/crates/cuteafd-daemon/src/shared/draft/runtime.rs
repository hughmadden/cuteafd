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
        Ok(Self { policy: DraftPolicy::seeded(geometry, seed)?, prior })
    }

    pub fn prior(&self, history: &DraftHistory, evidence: &resource::DraftEvidence<'_>, extension: usize) -> Vec<f64> {
        let mut prior = resource::prior(&self.prior, history, evidence);
        let width = match evidence {
            resource::DraftEvidence::Selector(features) => features.len(),
            resource::DraftEvidence::Head(head) => head.len(),
        };
        if width > 0 && prior.len() == width {
            prior.extend(std::iter::repeat_n(resource::TAIL_PRIOR, extension.min(COPY_EXTENSION)));
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
                assert!(prior[width..].iter().all(|&p| p == resource::TAIL_PRIOR));
                assert_eq!(shared.policy.geometry().max_positions, width + COPY_EXTENSION);
            }
            assert!(shared.prior(&history, &resource::DraftEvidence::Head(&[]), 32).is_empty());
        }
    }
}
