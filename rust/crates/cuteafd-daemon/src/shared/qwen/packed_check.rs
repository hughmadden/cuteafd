//! Native packed/singleton qualification shared by Qwen-scheduler families.
use super::scheduler::{DraftState, Placement, PrefillSeq};
use crate::shared::{prefill_packing::{self, Limits, MemberOutcome}, token_io::DeviceLogits};
use anyhow::{ensure, Result};
use cuteafd_ffi::NativeLibrary;

pub(crate) fn prompts(tokens: &[u32], members: usize) -> Result<Vec<Vec<u32>>> {
    const LENGTHS: [usize; 8] = [7, 9, 29, 40, 64, 65, 255, 1];
    ensure!(!tokens.is_empty() && (2..=LENGTHS.len()).contains(&members), "--packed-check needs 2..=8 members and golden tokens");
    Ok(LENGTHS[..members].iter().enumerate().map(|(i, &rows)| {
        let prefix = if i + 1 == members { 73 } else { 0 };
        (0..prefix + rows).map(|j| tokens[(i * 97 + j) % tokens.len()]).collect()
    }).collect())
}

pub(crate) fn reference_windows(windows: &[cuteafd_bench::reference::Window], first: &str,
    members: usize, genuine: bool) -> Result<Vec<cuteafd_bench::reference::Window>> {
    ensure!((2..=8).contains(&members) && windows.len() >= members, "packed reference needs distinct windows");
    let start = windows.iter().position(|w| w.id == first)
        .ok_or_else(|| anyhow::anyhow!("packed reference window {first} missing"))?;
    if !genuine {
        let selected = (0..members).map(|i| windows[(start + i) % windows.len()].clone()).collect::<Vec<_>>();
        ensure!(selected.iter().map(|w| &w.id).collect::<std::collections::BTreeSet<_>>().len() == members,
            "packed reference repeats a window id");
        return Ok(selected);
    }
    // Prefer short original windows, but never let synthetic trailing context
    // dilute the generated-row gate or leave it dominated by a single near tie.
    let mut selected = windows.iter().filter(|w| genuine_positions(w).len() >= 256)
        .cloned().collect::<Vec<_>>();
    selected.sort_by_key(|w| w.tokens.len());
    ensure!(selected.len() >= members, "not enough windows with 256 genuine rows");
    selected.truncate(members);
    ensure!(selected.iter().map(|w| &w.id).collect::<std::collections::BTreeSet<_>>().len() == members,
        "packed reference repeats a window id");
    Ok(selected)
}

fn genuine_positions(window: &cuteafd_bench::reference::Window) -> Vec<cuteafd_bench::reference::CompactPosition> {
    window.positions.iter().filter(|p| window.roles.get(p.pos).is_some_and(|role| role == "gen"))
        .cloned().collect()
}

fn singleton<P: Placement, D>(s: &mut PrefillSeq<'_, P, D>, chunk_rows: usize,
    single: &mut impl FnMut(&mut PrefillSeq<'_, P, D>) -> Result<Vec<f32>>) -> Result<Vec<f32>> {
    let mut logits = Vec::new();
    for (part, tokens) in s.tokens.chunks(chunk_rows).enumerate() {
        let first = part * chunk_rows;
        logits = single(&mut PrefillSeq { placement: s.placement, tokens, start: s.start + first,
            next: s.tokens.get(first + tokens.len()).copied().or(s.next), draft: s.draft, neural: s.neural })?;
    }
    Ok(logits)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn check<P: Placement, D: DraftState>(library: &NativeLibrary, prompts: &[Vec<u32>],
    mut packed: Vec<P>, mut alone: Vec<P>, limits: Limits, neural: bool, exact: bool,
    mut single: impl FnMut(&mut PrefillSeq<'_, P, D>) -> Result<Vec<f32>>,
    mut pass: impl FnMut(&mut [PrefillSeq<'_, P, D>], &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome)
        -> Result<Vec<Result<()>>>,
    mut snapshot: impl FnMut(&P) -> Result<Vec<u8>>) -> Result<()> {
    let n = prompts.len();
    ensure!(limits.chunk_rows > 0 && n >= 2 && prompts.last().is_some_and(|p| p.len() > 73),
        "invalid packed-check shape");
    ensure!(packed.len() == n && alone.len() == n, "packed-check placement count differs");
    let mut pd: Vec<D> = (0..n).map(|_| D::default()).collect();
    let mut ad: Vec<D> = (0..n).map(|_| D::default()).collect();
    let mut chunks = vec![(0, 0); n];
    for i in 0..n {
        let start = if i + 1 == n { 73 } else { 0 };
        chunks[i] = (start, prompts[i].len() - start);
        for (p, draft) in [(&mut packed[i], &mut pd[i]), (&mut alone[i], &mut ad[i])] {
            for (part, tokens) in prompts[i][..start].chunks(limits.chunk_rows).enumerate() {
                let at = part * limits.chunk_rows;
                single(&mut PrefillSeq { placement: p, tokens, start: at,
                    next: prompts[i].get(at + tokens.len()).copied(), draft, neural })?;
            }
        }
    }
    let groups = prefill_packing::groups(n, |r| prefill_packing::plan(&chunks[r], &limits).is_ok());
    let packed_coverage = groups.iter().any(|r| r.len() > 1);
    ensure!(exact || packed_coverage, "unrestricted packed check exercised no packed group");
    if !packed_coverage { println!("exact: singleton fallback, NOT a packing gate (cap {})", limits.rows); }
    let mut logits = vec![Vec::new(); n];
    for range in groups {
        let first = range.start;
        let mut sequences = packed[range.clone()].iter_mut().zip(&mut pd[range]).enumerate().map(|(j, (placement, draft))| {
            let i = first + j;
            PrefillSeq { placement, tokens: &prompts[i][chunks[i].0..], start: chunks[i].0,
                next: None, draft, neural }
        }).collect::<Vec<_>>();
        if sequences.len() == 1 {
            logits[first] = singleton(&mut sequences[0], limits.chunk_rows, &mut single)?;
        } else {
            for outcome in pass(&mut sequences, &mut |j, head| {
                logits[first + j] = head.row_host(library, 0)?;
                Ok(Ok(()))
            })? { outcome?; }
        }
    }
    let mut singleton_logits = Vec::with_capacity(n);
    for i in 0..n {
        singleton_logits.push(singleton(&mut PrefillSeq { placement: &mut alone[i], tokens: &prompts[i][chunks[i].0..],
            start: chunks[i].0, next: None, draft: &mut ad[i], neural }, limits.chunk_rows, &mut single)?);
    }
    for i in 0..n {
        let head = &singleton_logits[i];
        let equal = logits[i].len() == head.len() && logits[i].iter().zip(head).all(|(a, b)| a.to_bits() == b.to_bits());
        let state = snapshot(&packed[i])? == snapshot(&alone[i])?;
        let worst = logits[i].iter().zip(head).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let (kl, _) = crate::shared::golden_scores::scored_kl(head, &logits[i], head.len())?;
        let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
        let top1 = top(head) == top(&logits[i]);
        println!("packed check: member {i}, {} rows at {}: logits {equal}, state {state}, max |diff| {worst:.3e}, KL {kl:.6}, top1 {top1}", chunks[i].1, chunks[i].0);
        if exact { ensure!(equal && state, "packed member {i} differs from singleton"); }
    }
    println!("packed check: {}", if exact { "every member byte-identical" } else { "singleton deltas informational; official-reference gate required" });
    Ok(())
}

fn complete_rows(candidate: &cuteafd_bench::reference::Fidelity,
    control: &cuteafd_bench::reference::Fidelity, expected: usize) -> bool {
    expected > 0 && candidate.positions == expected && control.positions == expected
        && candidate.missing == 0 && control.missing == 0
        && candidate.non_finite == 0 && control.non_finite == 0
        && candidate.kl.is_finite() && control.kl.is_finite()
        && candidate.top1.is_finite() && control.top1.is_finite()
}

fn non_inferior(delta_kl: f64, delta_top1: f64) -> bool {
    delta_kl.is_finite() && delta_top1.is_finite() && delta_kl <= 0.005 + 1e-12 && delta_top1 >= -0.01 - 1e-12
}

/// Replay only original prefixes: synthetic partner-isolation prompts have no official oracle.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reference<P: Placement, D: DraftState>(library: &NativeLibrary,
    windows: &[cuteafd_bench::reference::Window], mut placements: Vec<P>, mut controls: Vec<P>, limits: Limits,
    neural: bool, precision: &cuteafd_bench::fidelity_precision::Precision,
    mut single: impl FnMut(&mut PrefillSeq<'_, P, D>) -> Result<Vec<f32>>,
    mut pass: impl FnMut(&mut [PrefillSeq<'_, P, D>], &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome)
        -> Result<Vec<Result<()>>>) -> Result<()> {
    use cuteafd_api::openai::probe::{Probe, ProbeSpec};
    const TAILS: [usize; 8] = [7, 9, 29, 40, 64, 65, 255, 1];
    let n = windows.len();
    ensure!((2..=8).contains(&n) && placements.len() == n && controls.len() == n && limits.chunk_rows > 0,
        "invalid official packed-check shape");
    precision.validate()?;
    let mut drafts: Vec<D> = (0..n).map(|_| D::default()).collect();
    let mut control_drafts: Vec<D> = (0..n).map(|_| D::default()).collect();
    // Initial tails cross aggregate routes; subsequent genuine rows measure
    // resulting state drift without scoring appended repository padding.
    let windows = windows.iter().map(|w| {
        let mut w = w.clone();
        w.positions = genuine_positions(&w);
        w
    }).collect::<Vec<_>>();
    let probes = windows.iter().map(|w| Probe::new(ProbeSpec {
        want: w.want(), top_k: 1, ..Default::default()
    })).collect::<Vec<_>>();
    let control_probes = windows.iter().map(|w| Probe::new(ProbeSpec {
        want: w.want(), top_k: 1, ..Default::default()
    })).collect::<Vec<_>>();
    let mut packed_groups = 0;
    for step in 0..windows.iter().map(|w| w.positions.len()).max().unwrap_or(0) {
        let mut active = Vec::new();
        for i in 0..n {
            let Some(position) = windows[i].positions.get(step) else { continue; };
            let end = position.pos;
            let start = placements[i].len();
            ensure!(start < end && end < windows[i].tokens.len(), "invalid official score frontier");
            let tail = TAILS[i].min(limits.chunk_rows).min(end - start);
            let before = end - tail;
            ensure!(controls[i].len() == start, "singleton control frontier differs");
            if before > start {
                for (placement, draft) in [(&mut placements[i], &mut drafts[i]),
                    (&mut controls[i], &mut control_drafts[i])] {
                    singleton(&mut PrefillSeq { placement, tokens: &windows[i].tokens[start..before],
                        start, next: windows[i].tokens.get(before).copied(), draft, neural },
                        limits.chunk_rows, &mut single)?;
                }
            }
            let logits = singleton(&mut PrefillSeq { placement: &mut controls[i],
                tokens: &windows[i].tokens[before..end], start: before,
                next: windows[i].tokens.get(end).copied(), draft: &mut control_drafts[i], neural },
                limits.chunk_rows, &mut single)?;
            control_probes[i].row(end, &logits);
            active.push(i);
        }
        // Members have independent frontiers and KV pages, even when they share a reference window.
        let chunks = active.iter().map(|&i| (placements[i].len(),
            windows[i].positions[step].pos - placements[i].len())).collect::<Vec<_>>();
        for range in prefill_packing::groups(active.len(), |r| prefill_packing::plan(&chunks[r], &limits).is_ok()) {
            let selected = &active[range];
            let mut sequences = placements.iter_mut().zip(&mut drafts).enumerate()
                .filter(|(i, _)| selected.contains(i)).map(|(i, (placement, draft))| {
                    let start = placement.len();
                    let end = windows[i].positions[step].pos;
                    PrefillSeq { placement, tokens: &windows[i].tokens[start..end], start,
                        next: windows[i].tokens.get(end).copied(), draft, neural }
                }).collect::<Vec<_>>();
            if selected.len() == 1 {
                let i = selected[0];
                probes[i].row(windows[i].positions[step].pos,
                    &singleton(&mut sequences[0], limits.chunk_rows, &mut single)?);
            } else {
                packed_groups += 1;
                for outcome in pass(&mut sequences, &mut |j, head| {
                    let i = selected[j];
                    probes[i].row(windows[i].positions[step].pos, &head.row_host(library, 0)?);
                    Ok(Ok(()))
                })? { outcome?; }
            }
        }
    }
    ensure!(packed_groups > 0, "official reference check exercised no packed group");
    let mut candidate_records = Vec::new();
    let mut control_records = Vec::new();
    for (i, ((window, probe), control)) in windows.iter().zip(&probes).zip(&control_probes).enumerate() {
        let record = probe.record();
        let reference = control.record();
        let score = window.score(&record.rows);
        let baseline = window.score(&reference.rows);
        #[cfg(test)]
        if let Some(out) = std::env::var_os("CUTEAFD_FIXTURE_ROWS") {
            let out = std::path::PathBuf::from(out);
            std::fs::create_dir_all(&out)?;
            std::fs::write(out.join(format!("member-{i}.json")), serde_json::to_vec_pretty(&serde_json::json!({
                "member": i, "window": window.id, "positions": window.positions.iter().map(|p| p.pos).collect::<Vec<_>>(),
                "packed": { "score": score, "probe": record },
                "singleton": { "score": baseline, "probe": reference },
            }))?)?;
        }
        let valid = record.error.is_none() && reference.error.is_none()
            && complete_rows(&score, &baseline, window.positions.len());
        let pass = valid && non_inferior(score.kl - baseline.kl, score.top1 - baseline.top1);
        println!("official packed member {i} ({}, informational): {} rows, KL {:.6}/{:.6}, top1 {:.6}/{:.6}, delta KL {:.6}, delta top1 {:.6}, pass {pass}",
            window.id, score.positions, score.kl, baseline.kl, score.top1, baseline.top1,
            score.kl - baseline.kl, score.top1 - baseline.top1);
        ensure!(valid, "official packed member {i} has incomplete or non-finite rows");
        let sparse = |records: &[cuteafd_bench::reference::Position]| {
            cuteafd_bench::reference::Fidelity::from_records(records.iter().enumerate()
                .filter(|(j, _)| j % 16 == 0).map(|(_, p)| p.clone()).collect())
        };
        let sparse_score = sparse(&score.records);
        let sparse_control = sparse(&baseline.records);
        println!("official packed sparse member {i} (informational): delta KL {:.6}, delta top1 {:.6}",
            sparse_score.kl - sparse_control.kl, sparse_score.top1 - sparse_control.top1);
        candidate_records.extend(score.records);
        control_records.extend(baseline.records);
    }
    let pooled = cuteafd_bench::reference::Fidelity::from_records(candidate_records);
    let control = cuteafd_bench::reference::Fidelity::from_records(control_records);
    let pooled_pass = pooled.positions > 0 && pooled.positions == control.positions
        && non_inferior(pooled.kl - control.kl, pooled.top1 - control.top1);
    println!("official packed pooled: {} rows, KL {:.6}/{:.6}, top1 {:.6}/{:.6}, delta KL {:.6}, delta top1 {:.6}, pass {pooled_pass}; card full panel requires tier {}",
        pooled.positions, pooled.kl, control.kl, pooled.top1, control.top1,
        pooled.kl - control.kl, pooled.top1 - control.top1, precision.tier.name());
    ensure!(pooled_pass, "official packed pooled non-inferiority failed");
    println!("official packed check: {packed_groups} packed groups, genuine-row pool non-inferior");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_dense_shapes_pack_below_route_boundary_and_leave_large_members_alone() {
        let p = prompts(&[1, 2, 3], 8).unwrap();
        let chunks = p.iter().enumerate().map(|(i, p)| {
            let start = if i == 7 { 73 } else { 0 };
            (start, p.len() - start)
        }).collect::<Vec<_>>();
        let limits = Limits { rows: 64, chunk_rows: 4096, max_context: 4096, taps: None };
        let groups = |rows| prefill_packing::groups(chunks.len(), |r|
            prefill_packing::plan(&chunks[r], &Limits { rows, ..limits }).is_ok());
        assert_eq!(groups(64), [0..3, 3..4, 4..5, 5..6, 6..7, 7..8]);
        assert_eq!(groups(0), (0..8).map(|i| i..i + 1).collect::<Vec<_>>());
        assert_eq!(groups(4096), [0..8]);
    }

    #[test]
    fn fixture_non_inferiority_has_no_absolute_tier_and_rejects_bad_deltas() {
        assert!(non_inferior(0.005, -0.01));
        assert!(non_inferior(0.0, 0.0)); // A weak singleton oracle is not a packing regression.
        assert!(!non_inferior(0.005001, 0.0));
        assert!(!non_inferior(0.0, -0.010001));
        assert!(!non_inferior(f64::NAN, 0.0));
        assert!(!non_inferior(0.0, f64::INFINITY));
        let members = [(0.006, 0.0), (-0.006, 0.0)];
        assert!(non_inferior(0.0, 0.0)); // Per-member route drift is informational.
        assert!(!members.iter().all(|&(kl, top1)| non_inferior(kl, top1)));
    }

    #[test]
    fn genuine_reference_selection_is_shortest_distinct_and_excludes_padding() {
        let windows = ["a00", "c04", "e12"].into_iter().enumerate().map(|(i, id)|
            cuteafd_bench::reference::Window { id: id.into(), block: "code".into(), bucket: "short".into(),
                tokens: vec![i as u32; 513 + i], roles: vec!["gen".into(); 513 + i], score_from: 1, top_k: 1,
                positions: (1..513).map(|pos| cuteafd_bench::reference::CompactPosition {
                    pos, next: 0, next_lp: 0.0, top: vec![], tail_lp: f64::NEG_INFINITY }).collect(), media: vec![] }).collect::<Vec<_>>();
        let selected = reference_windows(&windows, "e12", 3, true).unwrap();
        assert_eq!(selected.iter().map(|w| w.id.as_str()).collect::<Vec<_>>(), ["a00", "c04", "e12"]);
        assert_eq!(selected.iter().map(|w| w.tokens[0]).collect::<Vec<_>>(), [0, 1, 2]);
        assert!(reference_windows(&windows, "legacy", 2, true).is_err());
        assert!(reference_windows(&windows, "a00", 4, true).is_err());
        assert!(reference_windows(&[windows[0].clone(), windows[0].clone()], "a00", 2, true).is_err());
        let mut padded = windows.clone();
        padded[0].roles[200..].fill("ctx".into());
        assert!(reference_windows(&padded, "a00", 3, true).is_err());
        let genuine = genuine_positions(&padded[0]);
        assert_eq!(genuine.len(), 199);
        assert!(genuine.iter().all(|p| p.pos < 200));
        for window in &mut padded { window.roles.fill("ctx".into()); }
        // Exact uses only token seeds and must not acquire genuine-row restrictions.
        let exact = reference_windows(&padded, "e12", 3, false).unwrap();
        assert_eq!(exact.iter().map(|w| w.id.as_str()).collect::<Vec<_>>(), ["e12", "a00", "c04"]);
    }

    #[test]
    fn pooled_policy_allows_member_route_drift_but_not_incomplete_or_nonfinite_rows() {
        use cuteafd_bench::reference::Fidelity;
        let score = Fidelity { positions: 256, kl: 0.01, top1: 0.98, ..Default::default() };
        assert!(complete_rows(&score, &score, 256));
        for bad in [Fidelity { missing: 1, ..score.clone() },
            Fidelity { positions: 255, ..score.clone() },
            Fidelity { non_finite: 1, ..score.clone() },
            Fidelity { kl: f64::NAN, ..score.clone() }] {
            assert!(!complete_rows(&bad, &score, 256));
            assert!(!complete_rows(&score, &bad, 256));
        }
        assert!(!complete_rows(&Fidelity::default(), &Fidelity::default(), 0));
        // Pool by scored rows, not by member means; one member may exceed a delta bar.
        let members = [(256.0, 0.006, -0.02), (512.0, -0.003, 0.01)];
        let rows: f64 = members.iter().map(|m| m.0).sum();
        let kl = members.iter().map(|m| m.0 * m.1).sum::<f64>() / rows;
        let top1 = members.iter().map(|m| m.0 * m.2).sum::<f64>() / rows;
        assert!(non_inferior(kl, top1));
        assert!(!non_inferior(members[0].1, members[0].2));
    }

    #[derive(Clone, Default)]
    struct Placed(usize);
    impl Placement for Placed {
        fn len(&self) -> usize { self.0 }
        fn set_len(&mut self, n: usize) { self.0 = n; }
        fn units(&self) -> &[u32] { &[] }
        fn slot(&self) -> i32 { 0 }
        fn spans(&self) -> &[cuteafd_core::MediaSpan] { &[] }
    }

    #[test]
    fn singleton_reference_chunks_preserve_member_frontier_and_next_token() {
        let tokens = (0..130).collect::<Vec<_>>();
        let mut placement = Placed(73);
        let mut draft = ();
        let mut calls = Vec::new();
        let logits = singleton(&mut PrefillSeq { placement: &mut placement, tokens: &tokens,
            start: 73, next: Some(999), draft: &mut draft, neural: true }, 64, &mut |s| {
                assert_eq!(s.placement.len(), s.start);
                assert!(s.neural);
                calls.push((s.start, s.tokens.len(), s.next));
                s.placement.set_len(s.start + s.tokens.len());
                Ok(vec![s.start as f32])
            }).unwrap();
        assert_eq!(calls, [(73, 64, Some(64)), (137, 64, Some(128)), (201, 2, Some(999))]);
        assert_eq!(placement.len(), 203);
        assert_eq!(logits, [201.0]);
    }

    #[test]
    fn qwen_coordinator_inventory_requires_every_explicit_row_role() {
        use prefill_packing::{validate_program_inventory, ProgramRows::*};
        let inventory = [("qwen4_router_scores", PackedRows), ("qwen4_shared", PackedRows),
            ("qwen4_gdn_w8_m4096", SequenceLocal)];
        validate_program_inventory(["qwen4_router_scores", "qwen4_shared", "qwen4_gdn_w8_m4096"], &inventory).unwrap();
        assert!(validate_program_inventory(["qwen4_router_scores", "qwen4_shared", "qwen4_gdn_w8_m4096", "new"], &inventory).is_err());
    }

    #[test]
    fn packed_check_has_private_wrapping_prompts_and_resumed_tail() {
        let p = prompts(&[1, 2, 3], 8).unwrap();
        assert_eq!(p.iter().map(Vec::len).collect::<Vec<_>>(), [7, 9, 29, 40, 64, 65, 255, 74]);
        assert_ne!(&p[0][..3], &p[1][..3]);
        assert!(prompts(&[], 2).is_err());
        assert!(prompts(&[1], 1).is_err());
    }
}
