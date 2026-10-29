//! Reference-aligned diagnostic through the real segmented serving executor.
use super::{engine::Engine, pool::PoolAllocator, prefix, SparkLink};
use crate::shared::prefill_packing::{self, PackingMode};
use anyhow::{ensure, Result};
use cuteafd_api::openai::probe::{Probe, ProbeSpec};
use cuteafd_bench::reference::{Fidelity, Window};

struct Observed { logits: Vec<f32>, state: Vec<u8>, frontier: usize }

fn widths(case: usize, member: usize) -> usize {
    match case { 0 => [29, 31, 23][member % 3], 2 => [29, 31, 23][member % 3],
        _ => [200, 127, 64][member % 3] }
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
        let groups = if packed { prefill_packing::groups(active.len(), |r| {
            (case != 2 || chunks[r.clone()].iter().map(|(_, rows)| rows).sum::<usize>() <= 64)
                && engine.packed_plan(&chunks[r]).is_ok()
        }) }
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
                    && !b.state.is_empty() && a.logits.iter().chain(&b.logits).all(|v| v.is_finite()), "invalid packed observation");
                let (kl, top1) = pair_metrics(&a.logits, &b.logits, vocab);
                println!("packed case {case} replacement {replacement} member {i} window {}: singleton KL {kl:.8}, top1 {top1:.6}, logits byte-identical {}, state byte-identical {}",
                    window.id, equal(&a.logits, &b.logits), a.state == b.state);
                if mode == PackingMode::Exact {
                    ensure!(equal(&a.logits, &b.logits) && a.state == b.state,
                        "exact packed member {i} differs from singleton bytes");
                }
                let control = reference_score(window, a, vocab);
                let score = reference_score(window, b, vocab);
                ensure!(control.positions == 512 && score.positions == 512, "incomplete member score");
                noninferior(&control, &score, &format!("case {case} replacement {replacement} member {i}"))?;
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
    #[ignore = "requires the sealed V4 Flash coordinator and Spark expert packages"]
    fn live_skinny_router_packed_discriminator() -> Result<()> {
        use clap::Parser;
        #[derive(Parser)]
        struct Cli { #[command(flatten)] args: super::super::GoldenArgs }
        let snapshot = std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?;
        let peers = std::env::var("CUTEAFD_GOLDEN_PEERS")?;
        let fixtures = std::env::var("CUTEAFD_GOLDEN_FIXTURES")?;
        let manifest = std::env::var("CUTEAFD_PROGRAM_MANIFEST")?;
        let args = Cli::try_parse_from(["v4", "--snapshot", &snapshot, "--peers", &peers,
            "--manifest", &manifest, "--golden", &fixtures, "--full-prefill-logits",
            "--pool-tokens", "262144", "--rtx-expert-layers", "2", "--dspark",
            "--prefill-rows", "2752", "--prefill-waves", "3", "--max-sequences", "8"])?.args;
        let loaded = super::super::load(&args.engine)?;
        let (_, _, hash) = super::super::fidelity_publication(loaded.family)?;
        ensure!(cuteafd_bench::fidelity_rows::sha256(&args.golden.join("manifest.json"))? == hash,
            "skinny discriminator needs the immutable publication");
        let manifest: serde_json::Value = cuteafd_loader::read_json_file(&args.golden.join("manifest.json"))?;
        let reference = cuteafd_bench::fidelity_dataset::load(&args.golden, &manifest)?;
        let window = reference.windows.iter().find(|w| w.id == "legacy").unwrap();
        let prompts = vec![window.tokens[..512].to_vec(); 3];
        super::super::with_engine(&loaded, &args.engine, None, |_| Ok(0), |engine, links, runtime, _| {
            engine.configure_packing(PackingMode::Unrestricted, links)?;
            let (alone, _) = arm(engine, links, runtime, &prompts, &[0; 3], 2, false)?;
            let (packed, packs) = arm(engine, links, runtime, &prompts, &[0; 3], 2, true)?;
            ensure!(packs > 0, "skinny discriminator exercised no packs");
            for (i, (a, b)) in alone.iter().zip(&packed).enumerate() {
                ensure!(a.frontier == 512 && b.frontier == 512
                    && a.logits.len() == 512 * engine.cfg.vocab_size && a.logits.len() == b.logits.len()
                    && a.logits.iter().chain(&b.logits).all(|v| v.is_finite()), "invalid observation");
                let (kl, top1) = pair_metrics(&a.logits, &b.logits, engine.cfg.vocab_size);
                println!("skinny-only member {i}: packed-vs-singleton KL {kl:.8}, top-1 {top1:.6}, logits byte-identical {}, state byte-identical {}, packs {packs}, max packed rows 64",
                    equal(&a.logits, &b.logits), a.state == b.state);
            }
            Ok(())
        })
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
