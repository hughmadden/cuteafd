//! Same-cut executor N=1/N=2 golden and shared token/cache diagnostics.
use super::engine::{Allocator, GlmfEngine, GlmfPlacement};
use anyhow::{ensure, Context, Result};

/// Token-only diagnostics may consume the shared, checksum-validated quick fidelity windows.
pub(crate) fn input_tokens(root: &std::path::Path, vocab: usize) -> Result<Vec<u32>> {
    let path = root.join("tokens.bin");
    let tokens: Vec<u32> = if path.is_file() {
        let bytes = std::fs::read(path)?;
        ensure!(bytes.len() % 4 == 0, "golden token bytes must be u32-aligned");
        bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
    } else {
        use cuteafd_bench::fidelity_dataset;
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(manifest["config"] == fidelity_dataset::GLMF_CONFIG && manifest["vocab"].as_u64() == Some(vocab as u64),
            "Flash golden fidelity config or vocabulary differs");
        let reference = fidelity_dataset::load(root, &manifest)?;
        let windows = fidelity_dataset::quick_windows(&reference)?;
        ensure!(windows.iter().all(|w| w.media.is_empty()), "Flash unit golden requires text-only windows");
        println!("unit golden input: sealed quick windows {:?}; set {} (concatenated only for same-cut execution identity)",
            windows.iter().map(|w| &w.id).collect::<Vec<_>>(), reference.set_sha256);
        windows.into_iter().flat_map(|w| w.tokens).collect()
    };
    ensure!(!tokens.is_empty() && tokens.iter().all(|&t| (t as usize) < vocab), "invalid golden tokens");
    Ok(tokens)
}

/// Every paged byte of `placement`'s units, unit by unit, buffer by buffer (each paged buffer
/// holds one equal slice per allocation unit).
pub(crate) fn unit_bytes(engine: &GlmfEngine<'_>, placement: &GlmfPlacement) -> Result<Vec<u8>> {
    engine.synchronize()?;
    let mut out = Vec::new();
    for rank in 0..engine.ranks() {
        engine.on(rank, || {
            for layer in engine.paged_buffers_on(rank) {
                for buffer in std::iter::once(layer.records).chain(layer.keys).chain([layer.pools]) {
                    let units = engine.local_units();
                    ensure!(units > 0 && buffer.bytes % units == 0, "a paged buffer that is not whole units");
                    let per_unit = buffer.bytes / units;
                    for &global in &placement.units {
                        let Some(unit) = local_unit(global, rank, engine.context_kv()) else { continue };
                        let unit = crate::shared::prefix::view(buffer, unit as usize * per_unit, per_unit)?;
                        let mut bytes = vec![0u8; per_unit];
                        engine.library.copy_d2h(&mut bytes, unit)?;
                        out.extend(bytes);
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// Compare two shared-executor schedules with identical cuts, weights and fresh pages.
pub(crate) fn unit_check(engine: &GlmfEngine<'_>, tokens: &[u32]) -> Result<()> {
    ensure!(engine.weights.layers.len() == engine.cfg.layers, "--unit-check needs every layer");
    ensure!(engine.full_prefill_logits, "--unit-check needs every prefill row's logits");
    ensure!(engine.prefill_waves >= 2, "--unit-check needs at least two admitted slots");
    let n = tokens.len().min(engine.prefill_capacity());
    let cuts = engine.prefill_cuts(n)?;
    ensure!(cuts.len() > 2, "--unit-check needs more than two units to exercise slot reuse");
    let mut allocator = Allocator::for_engine(engine);
    let (mut one, mut two) = (allocator.admit(n)?, allocator.admit(n)?);
    let started = std::time::Instant::now();
    let one_logits = engine.prefill_unit_width(&mut one, &tokens[..n], 1)?.context("unit golden needs the head")?;
    let one_seconds = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let two_logits = engine.prefill_unit_width(&mut two, &tokens[..n], 2)?.context("unit golden needs the head")?;
    let two_seconds = started.elapsed().as_secs_f64();
    let logits = one_logits.len() == two_logits.len()
        && one_logits.iter().zip(&two_logits).all(|(a, b)| a.to_bits() == b.to_bits());
    let state = engine.slot_state(one.slot)? == engine.slot_state(two.slot)?;
    let paged = unit_bytes(engine, &one)? == unit_bytes(engine, &two)?;
    let verdict = |same: bool| if same { "identical" } else { "DIFFER" };
    println!("unit check: {n} tokens (same cuts {cuts:?}): logits {}, KDA state {}, paged units {} \
        (N1 {:.0} ms, N2 {:.0} ms)", verdict(logits), verdict(state), verdict(paged),
        1e3 * one_seconds, 1e3 * two_seconds);
    allocator.release(two);
    allocator.release(one);
    ensure!(logits && state && paged, "executor N1 and N2 differ over the same cuts");
    Ok(())
}

/// Context units live on one rank; whole/head-owned layers use the global unit ID.
fn local_unit(global: u32, rank: usize, context: bool) -> Option<u32> {
    if !context { Some(global) } else { ((global & 1) as usize == rank).then_some(global / 2) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn token_only_diagnostics_reject_truncated_or_out_of_vocabulary_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        std::fs::write(&path, [1, 0, 0]).unwrap();
        assert!(input_tokens(dir.path(), 10).is_err());
        std::fs::write(&path, 10u32.to_le_bytes()).unwrap();
        assert!(input_tokens(dir.path(), 10).is_err());
        std::fs::write(&path, 9u32.to_le_bytes()).unwrap();
        assert_eq!(input_tokens(dir.path(), 10).unwrap(), vec![9]);
    }

    #[test]
    fn diagnostics_visit_each_owner_local_unit_once() {
        for global in 0..8 {
            assert_eq!(local_unit(global, 0, false), Some(global));
            assert_eq!(local_unit(global, 1, false), Some(global));
            let owners: Vec<_> = (0..2).filter_map(|rank| local_unit(global, rank, true)
                .map(|local| (rank, local))).collect();
            assert_eq!(owners, [((global & 1) as usize, global / 2)]);
        }
    }
}
