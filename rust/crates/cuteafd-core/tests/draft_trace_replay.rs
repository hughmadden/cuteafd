//! Offline replay of a recorded draft-round trace (`CUTEAFD_SPECULATION_TRACE`
//! / `SPECULATION_TRACE` `draft_round` records) through the shared policy.
//!
//! Every recorded round is observed in order, so fits, calibration and route
//! history evolve as they did live. Before each observation the policy chooses
//! lengths for the round's requests from their recorded raw confidences, with
//! the priced rule and with the survival rule, and prices the recorded choice.
//! Output is one JSON line per selectable round, for offline scoring against
//! a cost truth; the recorded rounds' own lengths are what executed.
//!
//! Run: `CUTEAFD_POLICY_TRACE_REPLAY=trace.jsonl CUTEAFD_POLICY_TRACE_OUT=out.jsonl
//! cargo test -p cuteafd-core --release --test draft_trace_replay -- --ignored --nocapture`
use cuteafd_core::{DraftCandidate, DraftPolicy, DraftRoundObservation, LayerResource, ObservedDraftRequest,
    PolicyGeometry, PolicySeed, ResourceClass};
use serde_json::Value;
use std::io::{BufRead, Write};

fn geometry(record: &Value, max_positions: usize, widths: Vec<usize>) -> PolicyGeometry {
    let layers: Vec<LayerResource> = record["geometry"].as_array().unwrap().iter().map(|l| LayerResource {
        class: l["class"].as_u64().map(|c| c as u8),
        slice_bytes: l["slice_bytes"].as_f64().unwrap(),
        group_rows: l["group_rows"].as_u64().unwrap() as u8,
        timed: l["timed"].as_bool().unwrap(),
    }).collect();
    let topk = record["route_layout"][2].as_u64().unwrap() as u8;
    let experts = 1 + record["routes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).max().unwrap() as u16;
    PolicyGeometry { layers, experts: experts.max(512), topk, max_requests: 64, max_positions, widths,
        classes: vec![ResourceClass::rtx(), ResourceClass::spark()], group_rows_by_rows: record["group_rows_by_rows"].as_array().map_or_else(Vec::new, |rows|
            rows.iter().map(|row| row.as_array().unwrap().iter()
                .map(|group| group.as_u64().unwrap() as u8).collect()).collect()), regimes: 2 }
}

#[test]
#[ignore]
fn replay_recorded_trace() {
    let Ok(path) = std::env::var("CUTEAFD_POLICY_TRACE_REPLAY") else { return };
    let out = std::env::var("CUTEAFD_POLICY_TRACE_OUT").expect("CUTEAFD_POLICY_TRACE_OUT");
    let width: usize = std::env::var("CUTEAFD_POLICY_TRACE_WIDTH").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let records: Vec<Value> = std::io::BufReader::new(std::fs::File::open(&path).unwrap()).lines()
        .map_while(Result::ok).filter_map(|l| serde_json::from_str::<Value>(&l).ok())
        .filter(|r| r["type"] == "draft_round").collect();
    let positions = records.iter().flat_map(|r| r["requests"].as_array().unwrap().iter())
        .filter_map(|q| q["raw_confidence"].as_array().map(Vec::len)).max().unwrap_or(5).max(1);
    let native = if width > 0 { width } else {
        records.iter().filter_map(|r| r["draft_width"].as_u64()).min().unwrap_or(positions as u64) as usize
    }.min(positions);
    let widths = vec![native];
    let geometry = geometry(&records[0], positions, widths);
    let seed = PolicySeed::weak(native);
    let mut priced = DraftPolicy::new(geometry.clone(), seed.clone(), false).unwrap();
    let mut survival = DraftPolicy::new(geometry, seed, false).unwrap();
    survival.set_survival_lengths(true);
    let mut sink = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    let (mut selected, mut observed) = (0, 0);
    for (index, record) in records.iter().enumerate() {
        let requests = record["requests"].as_array().unwrap();
        let raw: Vec<Option<Vec<f64>>> = requests.iter().map(|q| (q["source"] == "Neural").then(||
            q["raw_confidence"].as_array().map(|c| c.iter().map(|v| v.as_f64().unwrap()).collect())).flatten()).collect();
        let ids: Vec<u64> = requests.iter().map(|q| q["id"].as_u64().unwrap()).collect();
        let drafted = record["drafted_requests"].as_u64().map(|n| n as usize);
        let recorded: Vec<usize> = requests.iter().map(|q| q["verified_rows"].as_u64().unwrap() as usize - 1).collect();
        // Choose only where every request had a neural proposal (the recorded
        // policy's own choices; copy and undrafted rounds are skipped).
        if raw.iter().all(Option::is_some) && !ids.is_empty() && record["draft_width"].is_u64() {
            let confidence: Vec<Vec<f64>> = raw.iter().map(|c| c.clone().unwrap()).collect();
            let candidates: Vec<_> = ids.iter().zip(&confidence)
                .map(|(&id, c)| DraftCandidate { id, confidence: c }).collect();
            let width = record["draft_width"].as_u64().unwrap() as usize;
            let n = drafted.unwrap_or(ids.len()).min(ids.len());
            let shared = record["shared"].as_bool().unwrap_or(false);
            let mine = priced.clone().select_drafted_within(shared, &candidates, width, usize::MAX, n).unwrap();
            let rule = survival.clone().select_drafted_within(shared, &candidates, width, usize::MAX, n).unwrap();
            if let (Some(mine), Some(rule)) = (mine, rule) {
                let calibrated: Vec<Vec<f64>> = candidates.iter().map(|c| priced.probabilities(candidates.len(), *c)).collect();
                let line = serde_json::json!({"index": index, "requests": ids.len(), "shared": shared,
                    "priced": mine.lengths, "survival": rule.lengths, "recorded": recorded,
                    "recorded_chosen": requests.iter().map(|q| q["chosen_rows"].as_u64().map(|v| v as usize - 1)).collect::<Vec<_>>(),
                    "calibrated": calibrated, "predicted_us": mine.predicted_us,
                    "recorded_predicted_us": record["predicted_us"], "total_us": record["total_us"],
                    "draft_us": record["draft_us"], "rows": record["rows"]});
                writeln!(sink, "{line}").unwrap();
                selected += 1;
            }
        }
        // Forecast vs actual expert traffic of the recorded shape, before the
        // round teaches the history.
        if record["total_us"].is_u64() || record["total_us"].is_f64() {
            let forecast = priced.forecast_megabytes(&ids, &recorded);
            let anchors = priced.forecast_megabytes(&ids, &vec![0; ids.len()]);
            let geometry = priced.geometry();
            let actual: Vec<f64> = (0..geometry.classes.len()).map(|c| record["megabytes"].as_array().unwrap().iter()
                .zip(&geometry.layers).filter(|(_, l)| l.timed && l.class == Some(c as u8))
                .map(|(m, _)| m.as_f64().unwrap()).sum()).collect();
            // B's price of one more draft row on the recorded shape: the
            // longest request's next row.
            let mut longer = recorded.clone();
            if let Some(i) = (0..longer.len()).max_by_key(|&i| raw[i].as_ref().map_or(0, Vec::len).saturating_sub(longer[i])) {
                longer[i] += 1;
            }
            let shared = record["shared"].as_bool().unwrap_or(false);
            let n = drafted.unwrap_or(ids.len()).min(ids.len());
            let width = record["draft_width"].as_u64().unwrap_or(native as u64) as usize;
            let base = priced.clone().predict_drafted(shared, &ids, &recorded, width, n);
            let plus = priced.clone().predict_drafted(shared, &ids, &longer, width, n);
            let marginal = base.zip(plus).map(|(a, b)| b - a);
            let recorded_us = base;
            let line = serde_json::json!({"forecast": index, "requests": ids.len(), "rows": record["rows"],
                "recorded_predicted_us": recorded_us,
                "marginal_us": marginal, "fit": format!("{:?}", priced.cost_snapshot(shared, ids.len())),
                "forecast_mb": forecast, "anchor_mb": anchors, "actual_mb": actual});
            writeln!(sink, "{line}").unwrap();
        }
        // Observe the round as it executed.
        let confidences: Vec<Option<Vec<f64>>> = requests.iter().zip(&raw).map(|(q, c)| {
            let accepted = q["accepted_rows"].as_u64().unwrap() as usize;
            c.as_ref().map(|c| match q["censor"].is_null() { true => c.clone(), false => c[..c.len().min(accepted - 1)].to_vec() })
        }).collect();
        let observed_requests: Vec<_> = requests.iter().zip(&confidences).map(|(q, c)| ObservedDraftRequest {
            id: q["id"].as_u64().unwrap(), rows: q["verified_rows"].as_u64().unwrap() as usize,
            accepted: q["accepted_rows"].as_u64().unwrap() as usize, confidence: c.as_deref() }).collect();
        let routes: Vec<u16> = record["routes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u16).collect();
        let layer_us: Vec<Option<f64>> = record["layer_us"].as_array().unwrap().iter().map(Value::as_f64).collect();
        let total = record["total_us"].as_f64().unwrap_or(f64::NAN);
        let draft = record["draft_us"].as_f64().unwrap_or(f64::NAN);
        let copied = requests.iter().any(|q| q["source"] == "Copy");
        for policy in [&mut priced, &mut survival] {
            let observation = DraftRoundObservation { shared: record["shared"].as_bool().unwrap_or(false),
                requests: &observed_requests, routes: &routes, layer_us: &layer_us, total_us: total,
                draft_us: if record["draft_width"].is_u64() && !(copied && drafted.is_none()) { draft } else { f64::NAN },
                drafted_requests: drafted, wide: record["draft_width"].as_u64().is_some_and(|w| w as usize > native),
                predicted_us: None };
            if policy.observe(observation).is_ok() { observed += 1; }
        }
    }
    sink.flush().unwrap();
    eprintln!("{path}: {} rounds, {selected} selectable, {} observed", records.len(), observed / 2);
}
