//! Host cost of the shared draft policy's per-round calls (diagnosis and gate).
//!
//! V4.1 Flash geometry (40 layers, 384 experts, top-6, 32 Spark layers),
//! warm fits, 16 requests with seven draft positions: the cost of one
//! `select_drafted_within`, one `choose_width` and one `observe`, as a C16 lane
//! pays them every round. Run in release mode:
//!
//! ```text
//! cargo test --release -p cuteafd-core --test draft_policy_bench -- --ignored --nocapture
//! ```
//!
//! With `CUTEAFD_POLICY_BENCH_MAX_US=N` it fails when select's median exceeds
//! N microseconds (non-regression bound: 1200 at this geometry, 2026-10-13).
use cuteafd_core::{DraftCandidate, DraftPolicy, DraftRoundObservation, LayerResource, ObservedDraftRequest,
    PolicyGeometry, PolicySeed, ResourceClass};
use std::time::Instant;

const LAYERS: usize = 40;
const TOPK: usize = 6;
const REQUESTS: usize = 16;
const POSITIONS: usize = 7;

fn geometry() -> PolicyGeometry {
    PolicyGeometry {
        layers: (0..LAYERS).map(|l| LayerResource { class: Some(u8::from(l >= 8)),
            slice_bytes: if l < 8 { 18_800_640. } else { 4_700_160. }, group_rows: 16, timed: l > 0 }).collect(),
        experts: 384, topk: TOPK as u8, max_requests: REQUESTS, max_positions: POSITIONS, widths: vec![5, 7],
        classes: vec![ResourceClass::rtx(), ResourceClass::spark()], regimes: 2,
    }
}

struct Mix(u64);
impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore]
fn draft_policy_host_cost_per_round() {
    let mut policy = DraftPolicy::new(geometry(), PolicySeed::weak(5), false).unwrap();
    let mut mix = Mix(7);
    let ids: Vec<u64> = (1..=REQUESTS as u64).collect();
    let (mut select_us, mut width_us, mut observe_us) = (Vec::new(), Vec::new(), Vec::new());
    for round in 0..600 {
        let confidence: Vec<Vec<f64>> = ids.iter().map(|_| (0..POSITIONS).map(|_| 0.5 + 0.5 * mix.unit()).collect())
            .collect();
        let candidates: Vec<_> = ids.iter().zip(&confidence).map(|(&id, c)| DraftCandidate { id, confidence: c })
            .collect();
        let requests: Vec<(u64, usize)> = ids.iter().map(|&id| (id, POSITIONS)).collect();
        let started = Instant::now();
        let width = policy.choose_width(true, &requests);
        let w = started.elapsed().as_secs_f64() * 1e6;
        let started = Instant::now();
        let selection = policy.select_drafted_within(true, &candidates, width, usize::MAX, REQUESTS).unwrap();
        let s = started.elapsed().as_secs_f64() * 1e6;
        let lengths: Vec<usize> = selection.as_ref().map_or(vec![width.min(POSITIONS); REQUESTS], |s| s.lengths.clone());
        let predicted = match &selection { Some(s) => Some(s.predicted_us),
            None => policy.predict_drafted(true, &ids, &lengths, width, REQUESTS) };
        let rows: usize = lengths.iter().map(|l| l + 1).sum();
        let routes: Vec<u16> = (0..LAYERS * rows * TOPK).map(|_| (mix.next() % 384) as u16).collect();
        let layer_us: Vec<Option<f64>> = (0..LAYERS).map(|l| (l > 0).then(|| 800. + 30. * rows as f64)).collect();
        let observed: Vec<_> = ids.iter().zip(&lengths).zip(&confidence).map(|((&id, &n), c)| ObservedDraftRequest {
            id, rows: n + 1, accepted: 1 + n / 2, confidence: Some(&c[..n.min(c.len())]) }).collect();
        let started = Instant::now();
        policy.observe(DraftRoundObservation { shared: true, requests: &observed, routes: &routes, layer_us: &layer_us,
            total_us: 40_000. + 2_000. * rows as f64, draft_us: 3_000., drafted_requests: Some(REQUESTS),
            wide: width > 5, predicted_us: predicted }).unwrap();
        let o = started.elapsed().as_secs_f64() * 1e6;
        if round >= 100 && selection.is_some() {
            select_us.push(s); width_us.push(w); observe_us.push(o);
        }
    }
    assert!(select_us.len() > 100, "the policy engaged");
    let (s, w, o) = (median(select_us), median(width_us), median(observe_us));
    eprintln!("C16 V4.1 geometry, medians: select {s:.1} us, choose_width {w:.1} us, observe {o:.1} us, total {:.1} us",
        s + w + o);
    if let Some(limit) = std::env::var("CUTEAFD_POLICY_BENCH_MAX_US").ok().and_then(|v| v.parse::<f64>().ok()) {
        assert!(s <= limit, "select median {s:.1} us exceeds {limit} us");
    }
}
