//! Golden diagnostic through the real segmented serving executor.
use super::{engine::Engine, pool::PoolAllocator, prefix, SparkLink};
use crate::shared::prefill_packing::{self, PackingMode};
use anyhow::{ensure, Context, Result};

struct Observed { logits: Vec<f32>, state: Vec<u8>, frontier: usize }

fn arm(engine: &Engine<'_>, links: &mut [SparkLink<'_>], runtime: &tokio::runtime::Runtime,
    prompts: &[Vec<u32>], chunks: &[(usize, usize)], packed: bool) -> Result<(Vec<Observed>, usize)> {
    let mut allocator = PoolAllocator::for_engine(engine)?;
    let mut placements = prompts.iter().map(|p| allocator.admit(p.len() + 1)).collect::<Result<Vec<_>>>()?;
    for ((placement, prompt), &(start, _)) in placements.iter_mut().zip(prompts).zip(chunks) {
        prefix::packed_state(engine, placement, true)?;
        for part in prompt[..start].chunks(engine.prefill_rows) {
            engine.prefill_device(placement, part, links, runtime, 0)?;
        }
    }
    let groups = if packed { prefill_packing::groups(chunks.len(), |r| engine.packed_plan(&chunks[r]).is_ok()) }
        else { (0..chunks.len()).map(|i| i..i + 1).collect() };
    let mut actual_packs = 0;
    let mut logits = vec![Vec::new(); chunks.len()];
    for range in groups {
        if range.len() == 1 {
            let i = range.start;
            logits[i] = engine.prefill_device(&mut placements[i], &prompts[i][chunks[i].0..], links, runtime, 1)?
                .context("packed check requires final logits")?.row_host(engine.library, 0)?;
        } else {
            actual_packs += 1;
            let first = range.start;
            let mut members = placements[range.clone()].iter_mut().zip(&prompts[range.clone()]).zip(&chunks[range])
                .map(|((p, tokens), &(start, _))| (p, &tokens[start..])).collect::<Vec<_>>();
            let prompt_ends = vec![true; members.len()];
            for result in engine.prefill_packed(&mut members, links, &prompt_ends, &mut |i, row| {
                logits[first + i] = row.row_host(engine.library, 0)?;
                Ok(Ok(()))
            })? { result?; }
        }
    }
    let observed = placements.iter().zip(logits).map(|(p, logits)| Ok(Observed {
        logits, state: prefix::packed_state(engine, p, false)?, frontier: p.len,
    })).collect::<Result<Vec<_>>>()?;
    Ok((observed, actual_packs))
}

pub(super) fn run(engine: &Engine<'_>, links: &mut [SparkLink<'_>], runtime: &tokio::runtime::Runtime,
    tokens: &[u32], members: usize, mode: PackingMode) -> Result<()> {
    ensure!((2..=16).contains(&members), "--packed-check takes 2 to 16 members");
    ensure!(!tokens.is_empty() && engine.shape.sequences >= members,
        "packed check needs tokens and at least {members} sequence slots");
    let mut chunks = vec![(0, 200); members];
    chunks[members - 1] = (301, 127);
    for &(_, rows) in &chunks { ensure!(rows <= engine.prefill_rows, "packed check needs >=200 prefill rows"); }
    let prompts = chunks.iter().enumerate().map(|(i, &(start, rows))|
        (0..start + rows).map(|r| tokens[(i * 97 + r) % tokens.len()]).collect::<Vec<_>>()).collect::<Vec<_>>();
    let (alone, _) = arm(engine, links, runtime, &prompts, &chunks, false)?;
    let (packed, packs) = arm(engine, links, runtime, &prompts, &chunks, true)?;
    if mode == PackingMode::Unrestricted { ensure!(packs > 0, "packed check did not exercise a multi-member unit"); }
    for (i, (a, b)) in alone.iter().zip(&packed).enumerate() {
        ensure!(a.frontier == b.frontier, "packed member {i} frontier differs");
        let logits_equal = a.logits.len() == b.logits.len()
            && a.logits.iter().zip(&b.logits).all(|(a, b)| a.to_bits() == b.to_bits());
        println!("packed member {i}: logits byte-identical {logits_equal}, state byte-identical {}", a.state == b.state);
        ensure!(b.logits.len() == engine.cfg.vocab_size && !b.state.is_empty()
            && b.logits.iter().all(|v| v.is_finite()), "empty or nonfinite packed observation");
        ensure!(logits_equal && a.state == b.state,
            "packed member {i} differs from singleton; characterize arithmetic route changes before accepting differences");
    }
    for target in [0, members - 1] {
        let partner_index = if target == 0 { 1 } else { 0 };
        let mut changed = prompts.clone();
        for token in &mut changed[partner_index] { *token = (*token + 1) % engine.cfg.vocab_size as u32; }
        ensure!(changed[partner_index] != prompts[partner_index], "partner mutation was vacuous");
        let (partner, _) = arm(engine, links, runtime, &changed, &chunks, true)?;
        let (a, b) = (&packed[target], &partner[target]);
        ensure!(a.frontier == b.frontier && a.state == b.state && a.logits.len() == b.logits.len()
            && a.logits.iter().zip(&b.logits).all(|(a, b)| a.to_bits() == b.to_bits()),
            "packed partner mutation changed member {target}");
    }
    println!("packed check: {members} members, {packs} actual multi-member units, partner isolation byte-identical, mode {mode:?}");
    Ok(())
}
