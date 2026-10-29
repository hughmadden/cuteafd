//! Reference-aligned diagnostic through the real segmented serving executor.
use super::{engine::Engine, pool::PoolAllocator, prefix, SparkLink};
use crate::shared::prefill_packing::{self, PackingMode};
use anyhow::{ensure, Result};
use cuteafd_api::openai::probe::{Probe, ProbeSpec};
use cuteafd_bench::{fidelity_precision::Precision, reference::Window};

struct Observed { logits: Vec<f32>, state: Vec<u8>, frontier: usize }

fn widths(case: usize, member: usize) -> usize {
    match case { 0 => [29, 40, 59][member % 3], 2 => [29, 31, 23][member % 3],
        _ => [200, 127, 64][member % 3] }
}

fn arm(engine: &Engine<'_>, links: &mut [SparkLink<'_>], runtime: &tokio::runtime::Runtime,
    prompts: &[Vec<u32>], case: usize, packed: bool) -> Result<(Vec<Observed>, usize)> {
    let mut allocator = PoolAllocator::for_engine(engine)?;
    let mut placements = prompts.iter().map(|p| allocator.admit(p.len() + 1)).collect::<Result<Vec<_>>>()?;
    for placement in &placements { prefix::packed_state(engine, placement, true)?; }
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

pub(super) fn run(engine: &Engine<'_>, links: &mut [SparkLink<'_>], runtime: &tokio::runtime::Runtime,
    window: &Window, precision: &Precision, members: usize, mode: PackingMode) -> Result<()> {
    ensure!((2..=16).contains(&members) && engine.shape.sequences >= members,
        "packed check needs 2 to 16 admitted sequence slots");
    ensure!(engine.full_prefill_logits && engine.prefill_rows >= 200,
        "packed check requires --full-prefill-logits and >=200 prefill rows");
    ensure!(window.id == "legacy" && window.tokens.len() == 513 && window.positions.len() == 512,
        "packed check requires the sealed original V4 Flash legacy window");
    let vocab = engine.cfg.vocab_size;
    let prompts = vec![window.tokens[..512].to_vec(); members];
    for case in 0..2 {
        let (alone, _) = arm(engine, links, runtime, &prompts, case, false)?;
        let (packed, packs) = arm(engine, links, runtime, &prompts, case, true)?;
        if mode == PackingMode::Unrestricted { ensure!(packs > 0, "packed check exercised no multi-member unit"); }
        for (i, (a, b)) in alone.iter().zip(&packed).enumerate() {
            ensure!(a.frontier == 512 && b.frontier == 512 && b.logits.len() == 512 * vocab
                && a.logits.len() == b.logits.len() && !b.state.is_empty()
                && a.logits.iter().chain(&b.logits).all(|v| v.is_finite()), "invalid packed observation");
            let logits_equal = equal(&a.logits, &b.logits);
            let (kl, top1) = pair_metrics(&a.logits, &b.logits, vocab);
            println!("packed case {case} member {i} width {} (<=64 {}): logits byte-identical {logits_equal}, state byte-identical {}, singleton KL {kl:.8}, top-1 {top1:.6} (informational)",
                widths(case, i), widths(case, i) <= 64, a.state == b.state);
            let probe = Probe::new(ProbeSpec { score_from: Some(window.score_from),
                score_positions: Some(window.positions.iter().map(|p| p.pos).collect()),
                want: window.want(), top_k: 32, ..Default::default() });
            for (row, logits) in b.logits.chunks_exact(vocab).enumerate() { probe.row(row + 1, logits); }
            let record = probe.record();
            let score = window.score(&record.rows);
            let (top1_min, kl_max) = precision.tier.bars();
            println!("packed reference case {case} member {i}: rows {}, mean KL(top1024+tail) {:.8}, top-1 {:.6}, tier {} bars {kl_max}/{top1_min}",
                score.positions, score.kl, score.top1, precision.tier.name());
            ensure!(score.positions == 512 && score.missing == 0 && score.non_finite == 0
                && score.kl <= kl_max && score.top1 + 1e-12 >= top1_min,
                "packed member {i} failed stored-reference fidelity tier");
            // #886: same-build singleton/singleton is not byte-stable on this
            // layout. Both modes gate stored fidelity; cross-pass identity is diagnostic.
        }
        // Report partner sensitivity; #886 prevents byte gating across separate passes.
        for target in [0, members - 1] {
            let partner_index = if target == 0 { 1 } else { 0 };
            let mut changed = prompts.clone();
            for token in &mut changed[partner_index] { *token = (*token + 1) % vocab as u32; }
            ensure!(changed[partner_index] != prompts[partner_index], "partner mutation was vacuous");
            let (partner, _) = arm(engine, links, runtime, &changed, case, true)?;
            let (a, b) = (&packed[target], &partner[target]);
            ensure!(a.frontier == b.frontier && a.logits.len() == b.logits.len()
                && b.logits.iter().all(|v| v.is_finite()), "invalid partner observation");
            let (kl, top1) = pair_metrics(&a.logits, &b.logits, vocab);
            println!("packed partner case {case} target {target}: logits byte-identical {}, state byte-identical {}, KL {kl:.8}, top-1 {top1:.6} (informational; #886)",
                equal(&a.logits, &b.logits), a.state == b.state);
        }
        println!("packed check case {case}: {members} members, {packs} actual multi-member units, mode {mode:?}");
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
            let (alone, _) = arm(engine, links, runtime, &prompts, 2, false)?;
            let (packed, packs) = arm(engine, links, runtime, &prompts, 2, true)?;
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
    fn diagnostic_widths_cover_router_boundary_and_pair_metrics_are_informational() {
        assert_eq!((widths(0, 0), widths(0, 1), widths(0, 2)), (29, 40, 59));
        assert_eq!((widths(1, 0), widths(1, 1), widths(1, 2)), (200, 127, 64));
        let (kl, top1) = pair_metrics(&[1.0, 0.0], &[0.0, 1.0], 2);
        assert!(kl > 0.0);
        assert_eq!(top1, 0.0);
        assert!(!equal(&[0.0], &[-0.0]));
    }
}
