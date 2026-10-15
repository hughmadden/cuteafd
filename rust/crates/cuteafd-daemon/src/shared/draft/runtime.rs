//! Resource-priced selection and agreed-copy priors for family adapters.
use super::{copy::COPY_EXTENSION, resource};
use crate::shared::draft_policy::DraftHistory;
use anyhow::Result;
use cuteafd_core::{DraftPolicy, PolicyGeometry, PolicySeed};

pub(crate) struct SharedPolicy {
    pub policy: DraftPolicy,
    prior: resource::Prior,
}

impl SharedPolicy {
    #[cfg(test)]
    pub fn new(homes: &[resource::LayerHome], experts: usize, topk: usize, rows: usize,
        width: usize, prior: resource::Prior, seed: PolicySeed) -> Result<Self> {
        let geometry = resource::geometry(homes, experts, topk, rows, width, COPY_EXTENSION)?;
        Self::with_geometry(geometry, prior, seed)
    }

    /// Bind the family's installed resource classes, TP slices and draft widths.
    pub fn with_geometry(geometry: PolicyGeometry, prior: resource::Prior, seed: PolicySeed) -> Result<Self> {
        let mut policy = DraftPolicy::seeded(geometry, seed)?;
        super::apply_length_rule(&mut policy);
        Ok(Self { policy, prior })
    }

    pub fn prior(&self, history: &DraftHistory, evidence: &resource::DraftEvidence<'_>, extension: usize) -> Vec<f64> {
        let mut prior = resource::prior(&self.prior, history, evidence);
        let width = match evidence {
            resource::DraftEvidence::Selector(features) => features.len(),
            resource::DraftEvidence::Head(head) => head.len(),
            resource::DraftEvidence::History(rates) => rates.len(),
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
    use crate::shared::draft::{binding::{self, DraftPass, RoundRecord, RoundRequest},
        clock::RoundTimes, evidence::{Censor, DraftSource, Evidence}, routes::RoundRoutes};
    use cuteafd_core::{DraftCandidate, LayerResource, ResourceClass};

    // The GLM constructor before it delegated to with_geometry.
    fn legacy_glm(homes: &[resource::LayerHome], width: usize) -> SharedPolicy {
        let geometry = resource::geometry(homes, 288, 8, 64, width, COPY_EXTENSION).unwrap();
        let mut policy = DraftPolicy::seeded(geometry, resource::seed()).unwrap();
        super::super::apply_length_rule(&mut policy);
        SharedPolicy { policy, prior: resource::Prior::Head }
    }

    fn assert_state(left: &DraftPolicy, right: &DraftPolicy) {
        assert_eq!(left.geometry(), right.geometry());
        assert_fits(left, right);
    }

    fn assert_fits(left: &DraftPolicy, right: &DraftPolicy) {
        assert_eq!(left.stats(), right.stats());
        assert_eq!(left.selected_cell(), right.selected_cell());
        for requests in [1, 4, 8, 16] {
            assert_eq!(left.calibration(requests), right.calibration(requests));
            assert_eq!(left.cost_snapshot(false, requests), right.cost_snapshot(false, requests));
            assert_eq!(left.prediction_health(false, requests), right.prediction_health(false, requests));
            assert_eq!(left.engaged(false, requests), right.engaged(false, requests));
        }
    }

    #[test]
    fn explicit_geometry_preserves_installed_resources_and_widths() {
        let geometry = PolicyGeometry {
            layers: vec![
                LayerResource { class: None, slice_bytes: 0., group_rows: 16, timed: false },
                LayerResource { class: Some(0), slice_bytes: 1_234_567., group_rows: 16, timed: true },
                LayerResource { class: Some(1), slice_bytes: 765_432., group_rows: 32, timed: true },
            ],
            experts: 256, topk: 8, max_requests: 64, max_positions: 5 + COPY_EXTENSION,
            widths: vec![3, 5], classes: vec![ResourceClass::rtx(), ResourceClass::spark()], group_rows_by_rows: Vec::new(), regimes: 2,
        };
        let shared = SharedPolicy::with_geometry(geometry.clone(), resource::Prior::History,
            PolicySeed::weak(5)).unwrap();
        assert_eq!(shared.policy.geometry(), &geometry);
        assert_eq!(shared.policy.cold_drafts(), 5);
    }

    #[test]
    fn pooled_history_prior_is_valued_without_a_head_blend() {
        let homes = [resource::LayerHome::Remote { slice_bytes: 4e6 }];
        let shared = SharedPolicy::new(&homes, 288, 8, 64, 5, resource::Prior::History,
            PolicySeed::weak(5)).unwrap();
        let mut own = DraftHistory::default();
        own.observe(5, 0);
        let mut other = DraftHistory::default();
        other.observe(5, 5);
        let pooled = own.pooled(5, [&other].into_iter());
        assert_ne!(pooled, own.conditional(5));
        let prior = shared.prior(&own, &resource::DraftEvidence::History(&pooled), COPY_EXTENSION + 1);
        assert_eq!(&prior[..5], &pooled);
        assert_eq!(prior.len(), 5 + COPY_EXTENSION);
        assert!(prior[5..].iter().enumerate().all(|(i, &p)| p == resource::tail_prior(i)));
        assert!(shared.prior(&own, &resource::DraftEvidence::History(&[]), COPY_EXTENSION).is_empty());
    }

    #[test]
    fn glm_constructor_paths_preserve_priors_decisions_and_fits() {
        let homes = [resource::LayerHome::Dense, resource::LayerHome::Local { slice_bytes: 4e6 },
            resource::LayerHome::Remote { slice_bytes: 2e6 }];
        let mut legacy = legacy_glm(&homes, 7);
        let mut shared = SharedPolicy::new(&homes, 288, 8, 64, 7, resource::Prior::Head,
            resource::seed()).unwrap();
        let geometry = resource::geometry(&homes, 288, 8, 64, 7, COPY_EXTENSION).unwrap();
        let mut explicit = SharedPolicy::with_geometry(geometry, resource::Prior::Head,
            resource::seed()).unwrap();
        let mut history = DraftHistory::default();
        let mut priced = 0;
        for round in 0..192 {
            let n = [1, 4, 16][round / 64];
            let head: Vec<_> = (0..7).map(|p| 0.95 - p as f32 * 0.06).collect();
            let prior = legacy.prior(&history, &resource::DraftEvidence::Head(&head), COPY_EXTENSION);
            assert_eq!(prior, shared.prior(&history, &resource::DraftEvidence::Head(&head), COPY_EXTENSION));
            assert_eq!(prior, explicit.prior(&history, &resource::DraftEvidence::Head(&head), COPY_EXTENSION));
            let candidates: Vec<_> = (0..n).map(|i| resource::Candidate { id: i as u64 + 1,
                prior: &prior, limit: 7, selector: round % 2 == 0, cold: round % 64 < 4 }).collect();
            let before = resource::select_drafted(&mut legacy.policy, &candidates, 7, 64, n);
            for policy in [&mut shared.policy, &mut explicit.policy] {
                let after = resource::select_drafted(policy, &candidates, 7, 64, n);
                assert_eq!(before.lengths, after.lengths);
                assert_eq!(before.predicted, after.predicted);
            }
            priced += usize::from(before.predicted.is_some());
            let rows = (1 + round % 4).min(64 / n);
            let accepted = if round % 3 == 0 { rows } else { 1 };
            let requests: Vec<_> = (0..n).map(|i| RoundRequest { id: i as u64 + 1, rows, accepted,
                source: DraftSource::Neural, evidence: Some(Evidence::Head(&prior)),
                censor: (round % 3 == 0).then_some(Censor::OutputLimit) }).collect();
            let mut routes = RoundRoutes::new(homes.len(), 8);
            routes.begin(n * rows);
            routes.dense(0).unwrap();
            for layer in 1..homes.len() {
                let ids: Vec<_> = (0..n * rows * 8).map(|i| ((i + layer * 8 + round) % 288) as u16).collect();
                routes.push_narrow(layer, &ids).unwrap();
            }
            let record = RoundRecord { shared: false, requests: &requests, routes: &routes,
                layer_us: &[None, Some(800.), Some(500.)],
                times: RoundTimes { total_us: 5000 + n as u64 * 500, draft_us: Some(1300 + n as u64 * 300) },
                draft: Some(DraftPass { width: 7, requests: Some(n) }), predicted: before.predicted };
            for policy in [&mut legacy.policy, &mut shared.policy, &mut explicit.policy] {
                binding::observe(policy, &record).unwrap();
            }
            assert_state(&legacy.policy, &shared.policy);
            assert_state(&legacy.policy, &explicit.policy);
            history.observe(rows - 1, accepted - 1);
        }
        assert!(priced > 0, "transcript must exercise warm priced decisions as well as cold allocation");
    }

    /// No recorded benchmark data is checked in. Run with
    /// CUTEAFD_GLMF_CONSTRUCTOR_TRACE=/path/to/speculation-trace.jsonl.
    #[test]
    #[ignore]
    fn replay_recorded_glmf_constructor_parity() {
        use serde_json::Value;
        use std::collections::BTreeMap;
        use std::io::BufRead;
        let path = std::env::var("CUTEAFD_GLMF_CONSTRUCTOR_TRACE")
            .expect("set CUTEAFD_GLMF_CONSTRUCTOR_TRACE to a completed GLM Flash trace");
        let records: Vec<Value> = std::io::BufReader::new(std::fs::File::open(&path).unwrap()).lines()
            .map(|line| serde_json::from_str::<Value>(&line.unwrap()).unwrap())
            .filter(|record| record["type"] == "draft_round").collect();
        assert!(!records.is_empty(), "trace contains no draft_round observations");
        let width = records.iter().filter_map(|r| r["draft_width"].as_u64()).min().unwrap() as usize;
        let homes: Vec<_> = records[0]["geometry"].as_array().unwrap().iter().map(|layer| {
            let slice_bytes = layer["slice_bytes"].as_f64().unwrap();
            match layer["class"].as_u64() {
                None => resource::LayerHome::Dense,
                Some(0) => resource::LayerHome::Local { slice_bytes },
                Some(1) => resource::LayerHome::Remote { slice_bytes },
                class => panic!("unexpected GLM resource class {class:?}"),
            }
        }).collect();
        let mut legacy = legacy_glm(&homes, width);
        let mut shared = SharedPolicy::new(&homes, 288, 8, 64, width, resource::Prior::Head,
            resource::seed()).unwrap();
        let mut installed_geometry = shared.policy.geometry().clone();
        installed_geometry.group_rows_by_rows = vec![vec![16; homes.len()]; 65];
        let mut installed = SharedPolicy::with_geometry(installed_geometry, resource::Prior::Head,
            resource::seed()).unwrap();
        let mut histories = BTreeMap::<u64, DraftHistory>::new();
        let mut selected = 0;
        let mut priced = 0;
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record["route_layout"][0].as_u64().unwrap() as usize, homes.len());
            assert_eq!(record["route_layout"][2], 8);
            for (layer, expected) in record["geometry"].as_array().unwrap().iter().zip(&legacy.policy.geometry().layers) {
                assert_eq!(layer["class"].as_u64(), expected.class.map(u64::from));
                assert_eq!(layer["slice_bytes"].as_f64().unwrap(), expected.slice_bytes);
                assert_eq!(layer["group_rows"].as_u64().unwrap(), u64::from(expected.group_rows));
                assert_eq!(layer["timed"].as_bool().unwrap(), expected.timed);
            }
            let requests = record["requests"].as_array().unwrap();
            let priors: Vec<Option<Vec<f64>>> = requests.iter().map(|r| r["raw_confidence"].as_array()
                .map(|p| p.iter().map(|p| p.as_f64().unwrap()).collect())).collect();
            let pass = record["draft_width"].as_u64().map(|w| DraftPass { width: w as usize,
                requests: record["drafted_requests"].as_u64().map(|n| n as usize) });
            let mut predicted = None;
            if let Some(pass) = pass {
                let candidates: Vec<_> = requests.iter().zip(&priors).map(|(r, prior)| {
                    let id = r["id"].as_u64().unwrap();
                    resource::Candidate { id, prior: prior.as_deref().unwrap_or(&[]),
                        limit: prior.as_ref().map_or(0, Vec::len), selector: width == 7,
                        cold: histories.get(&id).is_none_or(DraftHistory::cold) }
                }).collect();
                let n = pass.requests.unwrap_or(candidates.len());
                let before = resource::select_drafted(&mut legacy.policy, &candidates, pass.width, 64, n);
                let after = resource::select_drafted(&mut shared.policy, &candidates, pass.width, 64, n);
                assert_eq!(before.lengths, after.lengths, "round {index}");
                assert_eq!(before.predicted, after.predicted, "round {index}");
                let after = resource::select_drafted(&mut installed.policy, &candidates, pass.width, 64, n);
                assert_eq!(before.lengths, after.lengths, "installed round {index}");
                assert_eq!(before.predicted, after.predicted, "installed round {index}");
                predicted = before.predicted;
                priced += usize::from(predicted.is_some());
                selected += 1;
            }
            let verified: Vec<_> = requests.iter().zip(&priors).map(|(r, prior)| RoundRequest {
                id: r["id"].as_u64().unwrap(), rows: r["verified_rows"].as_u64().unwrap() as usize,
                accepted: r["accepted_rows"].as_u64().unwrap() as usize,
                source: match r["source"].as_str().unwrap() {
                    "Neural" => DraftSource::Neural, "Copy" => DraftSource::Copy,
                    "Undrafted" => DraftSource::Undrafted, source => panic!("unknown source {source}"),
                },
                evidence: prior.as_deref().map(Evidence::Head),
                censor: match r["censor"].as_str() {
                    None => None, Some("Eos") => Some(Censor::Eos),
                    Some("OutputLimit") => Some(Censor::OutputLimit), Some("Cancelled") => Some(Censor::Cancelled),
                    censor => panic!("unknown censor {censor:?}"),
                },
            }).collect();
            let rows = record["route_layout"][1].as_u64().unwrap() as usize;
            let ids: Vec<u16> = record["routes"].as_array().unwrap().iter()
                .map(|id| u16::try_from(id.as_u64().unwrap()).unwrap()).collect();
            assert_eq!(ids.len(), homes.len() * rows * 8);
            let mut routes = RoundRoutes::new(homes.len(), 8);
            routes.begin(rows);
            for (layer, ids) in ids.chunks_exact(rows * 8).enumerate() { routes.push_narrow(layer, ids).unwrap(); }
            let layer_us: Vec<_> = record["layer_us"].as_array().unwrap().iter().map(Value::as_f64).collect();
            let observation = RoundRecord { shared: record["shared"].as_bool().unwrap(),
                requests: &verified, routes: &routes, layer_us: &layer_us,
                times: RoundTimes { total_us: record["total_us"].as_u64().unwrap(),
                    draft_us: record["draft_us"].as_u64() }, draft: pass, predicted };
            for policy in [&mut legacy.policy, &mut shared.policy, &mut installed.policy] {
                binding::observe(policy, &observation).unwrap();
            }
            assert_state(&legacy.policy, &shared.policy);
            assert_fits(&legacy.policy, &installed.policy);
            let ids: Vec<_> = verified.iter().map(|r| r.id).collect();
            let lengths: Vec<_> = verified.iter().map(|r| r.rows - 1).collect();
            let forecast = legacy.policy.forecast_megabytes(&ids, &lengths);
            assert_eq!(forecast, shared.policy.forecast_megabytes(&ids, &lengths));
            assert_eq!(forecast, installed.policy.forecast_megabytes(&ids, &lengths));
            for request in &verified {
                if request.source == DraftSource::Neural {
                    let proposed = if request.censor.is_some() { request.accepted - 1 } else { request.rows - 1 };
                    histories.entry(request.id).or_default().observe(proposed, request.accepted - 1);
                }
                let confidence = priors[verified.iter().position(|r| r.id == request.id).unwrap()].as_deref().unwrap_or(&[]);
                let probabilities = legacy.policy.probabilities(verified.len(), DraftCandidate { id: request.id, confidence });
                assert_eq!(probabilities, shared.policy.probabilities(verified.len(), DraftCandidate { id: request.id, confidence }));
                assert_eq!(probabilities, installed.policy.probabilities(verified.len(), DraftCandidate { id: request.id, confidence }));
            }
        }
        assert!(selected > 0);
        assert!(priced > 0, "trace must reach warm priced decisions");
        eprintln!("{path}: {} observed rounds, {selected} selections, {priced} priced; constructor parity exact", records.len());
    }

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
