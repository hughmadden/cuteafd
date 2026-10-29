//! Reference-aligned diagnostic through the real segmented serving executor.
use super::{engine::Engine, pool::PoolAllocator, prefix, SparkLink};
use crate::shared::prefill_packing::{self, PackingMode};
use anyhow::{ensure, Result};
use cuteafd_api::openai::probe::{Probe, ProbeSpec};
use cuteafd_bench::reference::{Fidelity, Window};

struct Observed { logits: Vec<f32>, state: prefix::PackedState, frontier: usize }

fn widths(case: usize, member: usize) -> usize {
    match case { 0 => [29, 31, 23][member % 3], _ => [200, 127, 64][member % 3] }
}

fn arm(engine: &Engine<'_>, links: &mut [SparkLink<'_>], runtime: &tokio::runtime::Runtime,
    prompts: &[Vec<u32>], starts: &[usize], case: usize, packed: bool) -> Result<(Vec<Observed>, usize)> {
    let mut allocator = PoolAllocator::for_engine(engine)?;
    let mut placements = prompts.iter().map(|p| allocator.admit(p.len() + 1)).collect::<Result<Vec<_>>>()?;
    for placement in &placements { prefix::packed_state(engine, placement, true)?; }
    ensure!(starts.len() == prompts.len(), "diagnostic start count differs");
    for ((p, tokens), &start) in placements.iter_mut().zip(prompts).zip(starts) {
        ensure!(start < tokens.len(), "diagnostic start past prompt");
        if start > 0 { engine.prefill(p, &tokens[..start], links, runtime, 0, None)?; }
    }
    let mut logits = vec![Vec::new(); prompts.len()];
    let mut actual_packs = 0;
    while placements.iter().zip(prompts).any(|(p, t)| p.len < t.len()) {
        let active = placements.iter().zip(prompts).enumerate()
            .filter_map(|(i, (p, t))| (p.len < t.len()).then_some(i)).collect::<Vec<_>>();
        let chunks = active.iter().map(|&i| (placements[i].len,
            widths(case, i).min(prompts[i].len() - placements[i].len))).collect::<Vec<_>>();
        let groups = if packed { prefill_packing::groups(active.len(), |r| engine.packed_plan(&chunks[r]).is_ok()) }
            else { (0..active.len()).map(|i| i..i + 1).collect() };
        for range in groups {
            if range.len() == 1 {
                let i = active[range.start];
                let (start, rows) = chunks[range.start];
                logits[i].extend(engine.prefill(&mut placements[i], &prompts[i][start..start + rows],
                    links, runtime, rows, None)?);
            } else {
                actual_packs += 1;
                let ids = &active[range.clone()];
                // Active indices are ordered; collect disjoint mutable placements without unsafe.
                let mut members = placements.iter_mut().enumerate().filter_map(|(i, p)| {
                    ids.iter().position(|&id| id == i).map(|j| {
                        let (start, rows) = chunks[range.start + j];
                        (p, &prompts[i][start..start + rows])
                    })
                }).collect::<Vec<_>>();
                let ends = ids.iter().enumerate().map(|(j, &i)| {
                    let (start, rows) = chunks[range.start + j]; start + rows == prompts[i].len()
                }).collect::<Vec<_>>();
                for result in engine.prefill_packed_check(&mut members, links, &ends, &mut |j, _, rows| {
                    logits[ids[j]].extend(rows.to_host(engine.library)?);
                    Ok(())
                })? { result?; }
            }
        }
    }
    let observed = placements.iter().zip(logits).map(|(p, logits)| Ok(Observed {
        logits, state: prefix::packed_state(engine, p, false)?, frontier: p.len,
    })).collect::<Result<Vec<_>>>()?;
    Ok((observed, actual_packs))
}

fn equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

fn require_identical(a: &Observed, b: &Observed, vocab: usize, label: &str) -> Result<()> {
    let logit = a.logits.iter().zip(&b.logits).position(|(x, y)| x.to_bits() != y.to_bits());
    if let Some(i) = logit {
        println!("packed byte divergence {label}: logits row {} token {} singleton_bits {:08x} packed_bits {:08x}",
            i / vocab, i % vocab, a.logits[i].to_bits(), b.logits[i].to_bits());
    }
    let state = a.state.bytes.iter().zip(&b.state.bytes).position(|(x, y)| x != y);
    if let Some(i) = state {
        let (name, range) = b.state.tensors.iter().find(|(_, range)| range.contains(&i))
            .ok_or_else(|| anyhow::anyhow!("packed state offset {i} has no tensor"))?;
        println!("packed byte divergence {label}: state tensor {name} byte {} singleton {:02x} packed {:02x}",
            i - range.start, a.state.bytes[i], b.state.bytes[i]);
    }
    ensure!(a.logits.len() == b.logits.len() && a.state.bytes.len() == b.state.bytes.len()
        && a.state.tensors == b.state.tensors && logit.is_none() && state.is_none(),
        "{label} differs from singleton bytes (logits/state)");
    Ok(())
}

fn pair_metrics(a: &[f32], b: &[f32], vocab: usize) -> (f64, f64) {
    let normalized = |row: &[f32]| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = max + row.iter().map(|&x| (f64::from(x) - max).exp()).sum::<f64>().ln();
        row.iter().map(|&x| f64::from(x) - lse).collect::<Vec<_>>()
    };
    let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let mut kl = 0.0;
    let mut same = 0;
    for (a, b) in a.chunks_exact(vocab).zip(b.chunks_exact(vocab)) {
        same += usize::from(argmax(a) == argmax(b));
        let (a, b) = (normalized(a), normalized(b));
        kl += a.iter().zip(b).map(|(a, b)| a.exp() * (a - b)).sum::<f64>();
    }
    let rows = (a.len() / vocab) as f64;
    (kl.max(0.0) / rows, same as f64 / rows)
}

fn reference_score(window: &Window, observed: &Observed, vocab: usize) -> Fidelity {
    let probe = Probe::new(ProbeSpec { score_from: Some(window.score_from),
        score_positions: Some(window.positions.iter().map(|p| p.pos).collect()),
        want: window.want(), top_k: 32, ..Default::default() });
    for (row, logits) in observed.logits.chunks_exact(vocab).enumerate() {
        probe.row(window.score_from + row, logits);
    }
    window.score(&probe.record().rows)
}

fn noninferior(control: &Fidelity, packed: &Fidelity, label: &str) -> Result<()> {
    let delta_kl = packed.kl - control.kl;
    let delta_top1 = packed.top1 - control.top1;
    println!("packed reference {label}: singleton KL {:.8}/top1 {:.6}, packed KL {:.8}/top1 {:.6}, delta KL {delta_kl:.8}/top1 {delta_top1:.6}, positions {}",
        control.kl, control.top1, packed.kl, packed.top1, packed.positions);
    ensure!(packed.positions > 0 && packed.positions == control.positions && packed.missing == 0
        && control.missing == 0 && packed.non_finite == 0 && control.non_finite == 0
        && delta_kl <= 0.005 + 1e-12 && delta_top1 >= -0.01 - 1e-12,
        "{label} failed singleton non-inferiority");
    Ok(())
}

pub(super) fn run(engine: &Engine<'_>, links: &mut [SparkLink<'_>], runtime: &tokio::runtime::Runtime,
    windows: &[Window], members: usize, mode: PackingMode) -> Result<()> {
    ensure!((2..=16).contains(&members) && engine.shape.sequences >= members,
        "packed check needs 2 to 16 admitted sequence slots");
    ensure!(engine.full_prefill_logits && engine.prefill_rows >= 200,
        "packed check requires --full-prefill-logits and >=200 prefill rows");
    let selected = windows.iter().filter(|w| w.positions.len() == 512).take(members + 1).collect::<Vec<_>>();
    ensure!(selected.len() == members + 1, "packed check needs distinct sealed windows and a replacement partner");
    for (i, window) in selected.iter().enumerate() {
        ensure!(window.score_from > 0 && window.tokens.len() > window.score_from
            && window.positions.iter().enumerate().all(|(j, p)| p.pos == window.score_from + j)
            && window.tokens.len() - window.score_from == 512,
            "packed check window {} needs 512 contiguous scored positions", window.id);
        ensure!(selected[..i].iter().all(|other| other.id != window.id && other.tokens != window.tokens),
            "packed check windows must have distinct identities and token sequences");
    }
    let vocab = engine.cfg.vocab_size;
    for case in 0..2 {
        for replacement in [false, true] {
            let mut references = selected[..members].to_vec();
            if replacement { references[1] = selected[members]; }
            let starts = references.iter().map(|w| w.score_from - 1).collect::<Vec<_>>();
            let prompts = references.iter().map(|w| w.tokens[..w.tokens.len() - 1].to_vec()).collect::<Vec<_>>();
            let (alone, _) = arm(engine, links, runtime, &prompts, &starts, case, false)?;
            let (packed, packs) = arm(engine, links, runtime, &prompts, &starts, case, true)?;
            if mode == PackingMode::Unrestricted { ensure!(packs > 0, "packed check exercised no multi-member unit"); }
            let mut control_records = Vec::new();
            let mut packed_records = Vec::new();
            for (i, ((a, b), window)) in alone.iter().zip(&packed).zip(&references).enumerate() {
                let rows = prompts[i].len() - starts[i];
                ensure!(a.frontier == prompts[i].len() && b.frontier == a.frontier
                    && b.logits.len() == rows * vocab && a.logits.len() == b.logits.len()
                    && !b.state.bytes.is_empty() && a.logits.iter().chain(&b.logits).all(|v| v.is_finite()), "invalid packed observation");
                let (kl, top1) = pair_metrics(&a.logits, &b.logits, vocab);
                println!("packed case {case} replacement {replacement} member {i} window {}: singleton KL {kl:.8}, top1 {top1:.6}, logits byte-identical {}, state byte-identical {}",
                    window.id, equal(&a.logits, &b.logits), a.state.bytes == b.state.bytes);
                let control = reference_score(window, a, vocab);
                let score = reference_score(window, b, vocab);
                ensure!(control.positions == 512 && score.positions == 512, "incomplete member score");
                let fidelity = noninferior(&control, &score, &format!("case {case} replacement {replacement} member {i}"));
                require_identical(a, b, vocab, &format!("case {case} replacement {replacement} member {i} mode {mode:?}"))?;
                fidelity?;
                control_records.extend(control.records);
                packed_records.extend(score.records);
            }
            noninferior(&Fidelity::from_records(control_records), &Fidelity::from_records(packed_records),
                &format!("case {case} replacement {replacement} pooled"))?;
            println!("packed check case {case} replacement {replacement}: {members} members, {packs} actual multi-member units, mode {mode:?}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_identity_rejects_logit_and_named_state_differences() {
        let observed = || Observed { logits: vec![1.0, 0.0], frontier: 1,
            state: prefix::PackedState { bytes: vec![0, 1, 2, 3],
                tensors: vec![("rank0/layer2/unit0/compressed".into(), 0..2),
                    ("rank0/layer2/compressor_state0".into(), 2..4)] } };
        let a = observed();
        assert!(require_identical(&a, &observed(), 2, "equal").is_ok());
        let mut b = observed();
        b.logits[1] = -0.0;
        assert!(require_identical(&a, &b, 2, "logits").is_err());
        let mut b = observed();
        b.state.bytes[2] = 9;
        assert!(require_identical(&a, &b, 2, "state").is_err());
        let mut b = observed();
        b.state.bytes.pop();
        assert!(require_identical(&a, &b, 2, "extent").is_err());
    }

    #[test]
    fn noninferiority_checks_boundaries_missing_and_pooled_regressions() {
        let score = |kl, top1| Fidelity { positions: 512, kl, top1, ..Default::default() };
        let control = score(0.2, 0.8);
        assert!(noninferior(&control, &score(0.205, 0.79), "boundary").is_ok());
        assert!(noninferior(&control, &score(0.205001, 0.8), "kl").is_err());
        assert!(noninferior(&control, &score(0.2, 0.789999), "top1").is_err());
        for bad in [Fidelity { missing: 1, ..score(0.2, 0.8) },
            Fidelity { non_finite: 1, ..score(0.2, 0.8) },
            Fidelity { positions: 0, ..score(0.2, 0.8) }, score(f64::NAN, 0.8)] {
            assert!(noninferior(&control, &bad, "invalid").is_err());
        }
    }

    #[test]
    fn diagnostic_widths_cover_router_boundary_and_pair_metrics_are_informational() {
        assert_eq!((widths(0, 0), widths(0, 1), widths(0, 2)), (29, 31, 23));
        assert_eq!((widths(1, 0), widths(1, 1), widths(1, 2)), (200, 127, 64));
        let (kl, top1) = pair_metrics(&[1.0, 0.0], &[0.0, 1.0], 2);
        assert!(kl > 0.0);
        assert_eq!(top1, 0.0);
        assert!(!equal(&[0.0], &[-0.0]));
    }
}
