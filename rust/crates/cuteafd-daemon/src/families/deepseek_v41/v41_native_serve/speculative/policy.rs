//! Binding for the online verification-length policy: the installed expert
//! placement, round observations from captured routes/timings, and the
//! `/v1/stats` export. The policy itself lives in `cuteafd_core` and performs
//! no device work.
use anyhow::{ensure, Result};
use crate::families::deepseek_v41::v41_experts::coordinator::NativeTp4Wave;
use cuteafd_core::{DsparkLayerClass, DsparkObservedRequest, DsparkPlacement, DsparkPolicy,
    DsparkRoundObservation, DSPARK_LAYERS};
use std::collections::BTreeMap;
use std::sync::Mutex;

const HIDDEN: f64 = 5120.;
const INTERMEDIATE: f64 = 2304.;

/// Bytes one device reads for one routed expert: gate, up and down slices of
/// `hidden x intermediate / tp` 4-bit values plus their UE8M0 (MXFP4, per 32)
/// or E4M3 (NVFP4, per 16) block scales. This is the standardized base for the
/// official checkpoint; other formats' differences land in the fitted bandwidth.
pub(super) fn expert_slice_bytes(tp: usize, nvfp4: bool) -> f64 {
    let scale_block = if nvfp4 { 16. } else { 32. };
    3. * HIDDEN * (INTERMEDIATE / tp as f64) * (0.5 + 1. / scale_block)
}

/// Resource class and per-device expert bytes of every installed layer.
pub(super) fn placement(transport: &NativeTp4Wave<'_>, nvfp4: bool) -> Result<DsparkPlacement> {
    let remote_tp = transport.native_topology()
        .map_or(transport.spark_world(), |topology| topology.tp() as usize);
    ensure!(remote_tp > 0, "Spark tensor-parallel width is zero");
    let class = std::array::from_fn(|layer| if transport.has_local_layer(layer) {
        DsparkLayerClass::Local } else { DsparkLayerClass::Remote });
    let bytes = std::array::from_fn(|layer| {
        let tp = if transport.has_tp2_layer(layer) { 2 }
            else if transport.has_local_layer(layer) { 1 } else { remote_tp };
        expert_slice_bytes(tp, nvfp4)
    });
    DsparkPlacement::new(class, bytes).map_err(anyhow::Error::msg)
}

pub(super) fn sigmoid(x: f32) -> f64 {
    let x = f64::from(x);
    if x >= 0. { 1. / (1. + (-x).exp()) } else { x.exp() / (1. + x.exp()) }
}

/// Feed one completed lane round to the policy. `requests` are (identity,
/// verifier rows, accepted inputs, copied) in lane order, and `width` is the
/// round's dSpark draft width (zero when no draft pass ran).
///
/// A copied request's drafts came from its own history, not the drafter, so it
/// is observed without confidence: its rows still teach the cost fits and its
/// route history, never the drafter's calibration or acceptance evidence. A
/// round with a copy is not a shape the policy selected, so it keeps no
/// prediction. A draft pass beside a copy covered only the other requests, so
/// it gives no draft-cost sample, and its time is taken out of the round total
/// so the round fit's residual stays exact.
#[allow(clippy::too_many_arguments)]
pub(super) fn observe(policy: &mut DsparkPolicy, confidence_trace: &BTreeMap<u64, Vec<f32>>, shared: bool,
    routes: &[Vec<[u32; 6]>], layer_us: &[Option<f64>], requests: &[(u64, usize, u32, bool)],
    total_us: u64, draft_us: u64, width: usize, predicted: Option<f64>) -> Result<(), &'static str> {
    // Every drafted position's confidence, verified or not: the policy
    // bounds reliability evidence to reached positions itself.
    let confidence: Vec<Option<Vec<f64>>> = requests.iter().map(|&(id, _, _, copied)| if copied { None }
        else { confidence_trace.get(&id).map(|logits| logits.iter().map(|&x| sigmoid(x)).collect()) })
        .collect();
    let observed: Vec<_> = requests.iter().zip(&confidence).map(|(&(id, rows, accepted, _), confidence)|
        DsparkObservedRequest { id, rows, accepted: accepted as usize,
            confidence: confidence.as_deref() }).collect();
    let layer_us: [Option<f64>; DSPARK_LAYERS] =
        std::array::from_fn(|layer| layer_us.get(layer).copied().flatten());
    let copied = requests.iter().any(|request| request.3);
    let (total_us, draft_us) = match (width > 0, copied) {
        (true, false) => (total_us as f64, draft_us as f64),
        (true, true) => (total_us.saturating_sub(draft_us) as f64, f64::NAN),
        (false, _) => (total_us as f64, f64::NAN),
    };
    policy.observe(DsparkRoundObservation { shared, requests: &observed,
        routes, layer_us: &layer_us, total_us, predicted_us: if copied { None } else { predicted },
        draft_us, wide: width > policy.widths().0 })
}

static SNAPSHOT: Mutex<Option<serde_json::Value>> = Mutex::new(None);

/// The latest published policy state for the serving `stats` payload.
pub(crate) fn snapshot() -> serde_json::Value {
    SNAPSHOT.lock().ok().and_then(|slot| slot.clone()).unwrap_or(serde_json::Value::Null)
}

pub(super) fn publish(policy: &DsparkPolicy, draft_limit: usize) {
    let stats = policy.stats();
    let placement = policy.placement();
    let fit = |shared: bool| {
        let snapshot = policy.cost_snapshot(shared);
        let classes: Vec<_> = [DsparkLayerClass::Local, DsparkLayerClass::Remote].iter()
            .zip(snapshot.layers).map(|(class, c)| serde_json::json!({
                "class": class.label(),
                "intercept_us": c[0], "us_per_row": c[1], "us_per_mb": c[2],
                // µs per MB → GB/s: 1 MB / (c µs) = 1000 / c GB/s.
                "effective_gb_per_s": if c[2] > 0. { 1000. / c[2] } else { f64::INFINITY },
                "samples": c[3], "residual_scale_us": c[4],
            })).collect();
        serde_json::json!({
            "warm": policy.warm(shared),
            "layers": classes,
            "round": { "intercept_us": snapshot.round[0], "us_per_row": snapshot.round[1],
                "us_per_request": snapshot.round[2], "samples": snapshot.round[3],
                "residual_scale_us": snapshot.round[4] },
            "draft": { "intercept_us": snapshot.draft[0], "us_per_request": snapshot.draft[1],
                "wide_extra_us": snapshot.draft[2], "wide_extra_us_per_request": snapshot.draft[3],
                "samples": snapshot.draft[4], "residual_scale_us": snapshot.draft[5] },
        })
    };
    let reliability: Vec<_> = (0..7).map(|p| serde_json::json!({
        "position": p + 1,
        "reached": stats.position_reached[p],
        "mean_confidence": if stats.position_reached[p] > 0 {
            stats.position_confidence[p] / stats.position_reached[p] as f64 } else { 0. },
        "mean_raw_confidence": if stats.position_reached[p] > 0 {
            stats.position_raw_confidence[p] / stats.position_reached[p] as f64 } else { 0. },
        "logit_slope": policy.calibration()[p].0,
        "logit_offset": policy.calibration()[p].1,
        "accept_rate": if stats.position_reached[p] > 0 {
            stats.position_accepted[p] as f64 / stats.position_reached[p] as f64 } else { 0. },
    })).collect();
    let local_layers = (0..DSPARK_LAYERS).filter(|&l| placement.class(l) == DsparkLayerClass::Local).count();
    let value = serde_json::json!({
        "mode": if policy.fixed() { "fixed" } else { "bandwidth" },
        "draft_limit": draft_limit,
        "local_layers": local_layers,
        "remote_expert_bytes": placement.expert_bytes(DSPARK_LAYERS - 1),
        "rounds": stats.rounds,
        "selected_rounds": stats.selected_rounds,
        "verified_rows": stats.verified_rows,
        "verified_drafts": stats.verified_drafts,
        "accepted_drafts": stats.accepted_drafts,
        "request_rounds": stats.emitted_requests,
        "draft_rows_histogram": stats.draft_rows,
        "draft_widths": { "narrow": policy.widths().0, "wide": policy.widths().1 },
        "width_rounds": { "narrow": stats.width_rounds[0], "wide": stats.width_rounds[1] },
        "confidence_reliability": reliability,
        "time_bias_us": { "solo": policy.time_bias()[0], "shared": policy.time_bias()[1] },
        "prediction": {
            "rounds": stats.predicted_rounds,
            "mean_error_us": if stats.predicted_rounds > 0 {
                stats.prediction_error_us / stats.predicted_rounds as f64 } else { 0. },
            "mean_abs_relative_error": if stats.observed_us > 0. {
                stats.prediction_abs_error_us / stats.observed_us } else { 0. },
        },
        "solo": fit(false),
        "shared": fit(true),
    });
    if let Ok(mut slot) = SNAPSHOT.lock() { *slot = Some(value); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placement() -> DsparkPlacement {
        DsparkPlacement::new([DsparkLayerClass::Remote; DSPARK_LAYERS], [4_700_160.; DSPARK_LAYERS]).unwrap()
    }
    fn routes(rows: usize) -> Vec<Vec<[u32; 6]>> {
        (0..DSPARK_LAYERS).map(|layer| (0..rows).map(|row|
            std::array::from_fn(|slot| ((layer * 7 + row * 6 + slot) % 384) as u32)).collect()).collect()
    }
    /// Request 1 drafted this round; request 2's trace is left from an earlier
    /// dSpark round.
    fn trace() -> BTreeMap<u64, Vec<f32>> {
        [(1, vec![2.0; 5]), (2, vec![-4.0; 5])].into()
    }

    /// Without copies the observation is exactly the one built before copy
    /// windows existed: the policy ends in the same state.
    #[test]
    fn rounds_without_copies_observe_as_before() {
        let layer_us = vec![Some(150.0); DSPARK_LAYERS];
        let requests = [(1u64, 5usize, 3u32, false), (2, 8, 1, false)];
        for (width, predicted) in [(5, Some(18_000.)), (7, None), (0, Some(9_000.))] {
            let confidence: Vec<Option<Vec<f64>>> = requests.iter().map(|&(id, ..)|
                trace().get(&id).map(|logits| logits.iter().map(|&x| sigmoid(x)).collect())).collect();
            let observed: Vec<_> = requests.iter().zip(&confidence).map(|(&(id, rows, accepted, _), confidence)|
                DsparkObservedRequest { id, rows, accepted: accepted as usize, confidence: confidence.as_deref() })
                .collect();
            let mut before = DsparkPolicy::new(placement(), false);
            before.observe(DsparkRoundObservation { shared: false, requests: &observed, routes: &routes(13),
                layer_us: &std::array::from_fn(|layer| layer_us[layer]), total_us: 20_000., predicted_us: predicted,
                draft_us: if width > 0 { 1_500. } else { f64::NAN }, wide: width > 5 }).unwrap();
            let mut after = DsparkPolicy::new(placement(), false);
            observe(&mut after, &trace(), false, &routes(13), &layer_us, &requests, 20_000, 1_500, width, predicted)
                .unwrap();
            assert_eq!(format!("{before:?}"), format!("{after:?}"), "width {width}");
        }
    }

    /// A copied request is verified like any other but drafted by nobody: its
    /// rows teach the cost fits, never the drafter's calibration or acceptance
    /// evidence, the draft-cost fit or the time bias, even with a stale trace.
    #[test]
    fn copy_rounds_teach_costs_but_never_the_drafters_evidence() {
        let layer_us = vec![Some(150.0); DSPARK_LAYERS];
        // Request 1 verified four drafts and accepted two; request 2 verified
        // seven and accepted none.
        let round = |copied| [(1u64, 5usize, 3u32, false), (2, 8, 1, copied)];
        let observe_round = |requests: &[(u64, usize, u32, bool)], width, predicted| {
            let mut policy = DsparkPolicy::new(placement(), false);
            let rows = requests.iter().map(|r| r.1).sum();
            observe(&mut policy, &trace(), false, &routes(rows), &layer_us, requests, 20_000, 1_500, width, predicted)
                .unwrap();
            policy
        };
        let control = observe_round(&round(false), 5, Some(18_000.));
        let copy = observe_round(&round(true), 5, Some(18_000.));
        let solo = observe_round(&round(false)[..1], 5, Some(18_000.));
        // Sensitivity: the drafted request 2 adds evidence at position one.
        assert_eq!(control.stats().position_reached[..3], [2, 1, 1]);
        // The copied request adds none: exactly request 1's evidence alone.
        assert_eq!(copy.stats().position_reached, solo.stats().position_reached);
        assert_eq!(copy.stats().position_accepted, solo.stats().position_accepted);
        assert_eq!(copy.calibration(), solo.calibration());
        assert_ne!(control.calibration(), solo.calibration());
        // No prediction and no time bias from a shape the policy did not select.
        assert_eq!((control.stats().predicted_rounds, copy.stats().predicted_rounds), (1, 0));
        assert_ne!(control.time_bias(), [0.; 2]);
        assert_eq!(copy.time_bias(), [0.; 2]);
        // The draft pass beside a copy covered one request of two: no draft sample.
        assert_eq!((control.cost_snapshot(false).draft[4], copy.cost_snapshot(false).draft[4]), (1., 0.));
        assert_eq!(copy.stats().width_rounds, [0, 0]);
        // The copy's rows still teach the layer and round fits, and taking the
        // draft time out of the total leaves the round residual exact.
        assert_eq!(copy.cost_snapshot(false).layers, control.cost_snapshot(false).layers);
        assert_eq!(copy.cost_snapshot(false).round, control.cost_snapshot(false).round);
        assert_eq!((copy.stats().rounds, copy.stats().verified_rows, copy.stats().verified_drafts), (1, 13, 11));

        // A round in which every request copied ran no draft pass at all.
        let copies = observe_round(&[(2, 8, 4, true)], 0, None);
        let undrafted = observe_round(&[(3, 8, 4, false)], 0, None);
        assert_eq!(copies.stats().position_reached, [0; 7]);
        assert_eq!(copies.cost_snapshot(false).draft[4], 0.);
        assert_eq!(copies.cost_snapshot(false).round, undrafted.cost_snapshot(false).round);
    }

    #[test]
    fn official_spark_tp4_slice_matches_the_packed_geometry() {
        // 3 x (1,474,560 FP4 bytes + 92,160 UE8M0 scale bytes).
        assert_eq!(super::expert_slice_bytes(4, false), 4_700_160.);
        assert_eq!(super::expert_slice_bytes(1, false), 18_800_640.);
        assert_eq!(super::expert_slice_bytes(2, false), 9_400_320.);
        assert_eq!(super::expert_slice_bytes(4, true), 4_976_640.);
    }
}
