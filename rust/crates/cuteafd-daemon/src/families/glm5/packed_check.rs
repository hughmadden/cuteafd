//! Stored-token packed/singleton exact or fidelity gate, including a resumed prompt.
use super::engine::{GlmEngine, GlmPlacement, PageAllocator, PAGE_ROWS, RECORD_BYTES, INDEX_PAGE_BYTES};
use crate::shared::prefill_packing::{self, Limits};
use crate::shared::spark_intake::SparkLink;
use anyhow::{ensure, Context, Result};

const LENGTHS: [usize; 10] = [29, 40, 7, 64, 65, 127, 1, 255, 384, 100];

fn sequence(tokens: &[u32], member: usize, rows: usize) -> Vec<u32> {
    (0..rows).map(|row| tokens[(member * 97 + row) % tokens.len()]).collect()
}

fn state(engine: &GlmEngine<'_>, placement: &GlmPlacement) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for rank in 0..engine.ranks() {
        engine.on(rank, || {
            // SAFETY: the diagnostic owns this engine; all reads follow its queued work.
            unsafe { engine.library.cuda_stream_synchronize(engine.stream_of(rank))?; }
            for (records, index) in engine.paged_buffers_on(rank) {
                for (logical, &page) in placement.pages.iter().enumerate().take(placement.len.div_ceil(PAGE_ROWS)) {
                    if engine.context_kv() && page as usize % 2 != rank { continue; }
                    let local = if engine.context_kv() { page as usize / 2 } else { page as usize };
                    let rows = (placement.len - logical * PAGE_ROWS).min(PAGE_ROWS);
                    let mut read = |buffer, offset, bytes| -> Result<()> {
                        let view = crate::shared::prefix::view(buffer, offset, bytes)?;
                        let first = out.len();
                        out.resize(first + bytes, 0);
                        engine.library.copy_d2h(&mut out[first..], view)?;
                        Ok(())
                    };
                    read(records, local * PAGE_ROWS * RECORD_BYTES, rows * RECORD_BYTES)?;
                    if let Some(index) = index {
                        read(index, local * INDEX_PAGE_BYTES, rows * 128)?;
                        read(index, local * INDEX_PAGE_BYTES + PAGE_ROWS * 128, rows * 4)?;
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(out)
}

pub(crate) fn check(engine: &GlmEngine<'_>, tokens: &[u32], golden: &std::path::Path, members: usize,
    mode: prefill_packing::PackingMode, mut transport: Option<&mut SparkLink<'_>>,
    runtime: &tokio::runtime::Runtime) -> Result<()> {
    ensure!(!tokens.is_empty() && (2..=LENGTHS.len()).contains(&members), "--packed-check needs 2..10 members and stored tokens");
    let resumed = members - 1;
    let prefix = |i| if i == resumed { 2100 } else { 0 };
    let lengths = LENGTHS[..members - 1].iter().copied().chain([100]).collect::<Vec<_>>();
    let chunks = lengths.iter().enumerate().map(|(i, &rows)| (prefix(i), rows)).collect::<Vec<_>>();
    let limits = engine.packed_limits();
    for chunk in &chunks {
        prefill_packing::plan(&[*chunk], &Limits { rows: limits.chunk_rows, ..limits })?;
    }
    let groups = prefill_packing::groups(members, |range| prefill_packing::plan(&chunks[range], &limits).is_ok());
    let exact = mode == prefill_packing::PackingMode::Exact;
    ensure!(exact || groups.iter().any(|group| group.len() > 1),
        "unrestricted packed-check must exercise a pack");
    let prompts = lengths.iter().enumerate().map(|(i, &rows)| sequence(tokens, i, prefix(i) + rows)).collect::<Vec<_>>();
    let mut allocator = PageAllocator::new(engine);
    let mut packed = prompts.iter().map(|p| allocator.admit(p.len() + 1)).collect::<Result<Vec<_>>>()?;
    let mut alone = prompts.iter().map(|p| allocator.admit(p.len() + 1)).collect::<Result<Vec<_>>>()?;
    for placement in [&mut packed[resumed], &mut alone[resumed]] {
        for chunk in prompts[resumed][..2100].chunks(engine.prefill_capacity()) {
            engine.prefill_device(placement, chunk, transport.as_deref_mut().map(|t| (t, runtime)), 0)?;
        }
    }
    let chunk = |i: usize| &prompts[i][prefix(i)..];
    let mut packed_logits = vec![Vec::new(); members];
    let mut packed_groups = 0;
    for range in &groups {
        if range.len() == 1 {
            let i = range.start;
            packed_logits[i] = engine.prefill_device(&mut packed[i], chunk(i),
                transport.as_deref_mut().map(|t| (t, runtime)), 1)?.context("packed-check head")?
                .row_host(engine.library, 0)?;
        } else {
            packed_groups += 1;
            let first = range.start;
            let mut sequences = packed[range.clone()].iter_mut().enumerate().map(|(i, p)| (p, chunk(first + i))).collect::<Vec<_>>();
            let (_, outcomes) = engine.prefill_packed(&mut sequences, &mut |i, logits| {
                packed_logits[first + i] = logits.row_host(engine.library, 0)?;
                Ok(Ok(()))
            })?;
            for outcome in outcomes { outcome?; }
        }
    }
    let (mut equal, mut top1, mut worst_kl) = (true, 0usize, 0.0f64);
    for (i, placement) in alone.iter_mut().enumerate() {
        let logits = engine.prefill_device(placement, chunk(i), transport.as_deref_mut().map(|t| (t, runtime)), 1)?
            .context("packed-check singleton head")?.row_host(engine.library, 0)?;
        let same_logits = packed_logits[i].iter().map(|f| f.to_bits()).eq(logits.iter().map(|f| f.to_bits()));
        let same_state = state(engine, &packed[i])? == state(engine, placement)?;
        let kl = super::mean_kl(&packed_logits[i], &logits, logits.len());
        let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(id, _)| id);
        let same_top1 = argmax(&packed_logits[i]) == argmax(&logits);
        top1 += usize::from(same_top1);
        worst_kl = worst_kl.max(kl);
        ensure!(kl.is_finite(), "packed sequence {i} produced nonfinite KL");
        println!("packed check: sequence {i}: {} rows at {}: logits {}, live cache {}, KL {kl:.8}, top-1 {same_top1}",
            lengths[i], prefix(i), if same_logits { "identical" } else { "DIFFER" },
            if same_state { "identical" } else { "DIFFER" });
        equal &= same_logits && same_state;
    }
    for placement in packed.into_iter().chain(alone) { allocator.release(placement); }
    println!("packed check: {members} sequences, {packed_groups} packed groups, {} singleton groups; {}",
        groups.len() - packed_groups, if equal { "every sequence identical" } else { "DIFFER" });
    if exact {
        ensure!(equal, "exact prefill differs from its singleton reference");
    } else {
        let agreement = top1 as f64 / members as f64;
        println!("packed versus singleton (informational): worst member KL {worst_kl:.8}, top-1 {agreement:.4}");
        golden_fidelity(engine, tokens, golden, members, transport, runtime)?;
    }
    Ok(())
}

#[derive(Default)]
struct Fidelity {
    rows: usize,
    kl_sum: f64,
    top1: usize,
}

impl Fidelity {
    fn add(&mut self, logits: &[f32], golden: &[f32], vocab: usize) -> Result<()> {
        ensure!(vocab > 0 && logits.len() == golden.len() && logits.len().is_multiple_of(vocab),
            "packed golden logits shape differs");
        ensure!(logits.iter().chain(golden).all(|v| v.is_finite()), "nonfinite packed golden logits");
        let rows = logits.len() / vocab;
        self.kl_sum += super::mean_kl(logits, golden, vocab) * rows as f64;
        self.top1 += logits.chunks_exact(vocab).zip(golden.chunks_exact(vocab))
            .filter(|(a, b)| super::argmax(a) == super::argmax(b)).count();
        self.rows += rows;
        Ok(())
    }

    fn gate(&self) -> Result<()> {
        ensure!(self.rows > 0 && self.kl_sum.is_finite()
            && self.kl_sum / self.rows as f64 <= 0.05
            && self.top1 as f64 / self.rows as f64 >= 0.93,
            "unrestricted packing exceeds stored-golden GLM K4 fidelity bars");
        Ok(())
    }
}

/// Stored reference rows are valid only for the original, unshifted token prefix.
/// Keep the independent shifted/wrapped prompts above for isolation/exact checks.
fn golden_fidelity(engine: &GlmEngine<'_>, tokens: &[u32], golden: &std::path::Path,
    members: usize, mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let resumed = members - 1;
    let start = |i| if i == resumed { 1024 } else { 0 };
    let lengths = LENGTHS[..resumed].iter().copied().chain([100]).collect::<Vec<_>>();
    ensure!(lengths.iter().enumerate().all(|(i, &rows)| start(i) + rows <= tokens.len()),
        "stored golden needs every unshifted prompt prefix");
    let vocab = engine.cfg.vocab_size;
    let mut reference = std::fs::File::open(golden.join("logits.bin"))?;
    ensure!(reference.metadata()?.len() == (tokens.len() * vocab * 4) as u64, "stored golden logits shape differs");
    let mut allocator = PageAllocator::new(engine);
    let mut placements = lengths.iter().enumerate().map(|(i, &rows)| allocator.admit(start(i) + rows + 1))
        .collect::<Result<Vec<_>>>()?;
    for part in tokens[..start(resumed)].chunks(engine.prefill_capacity()) {
        engine.prefill_device(&mut placements[resumed], part,
            transport.as_deref_mut().map(|t| (t, runtime)), 0)?;
    }
    let limits = engine.packed_limits();
    let chunks = lengths.iter().enumerate().map(|(i, &rows)| (start(i), rows)).collect::<Vec<_>>();
    let groups = prefill_packing::groups(members, |range| prefill_packing::plan(&chunks[range], &limits).is_ok());
    ensure!(groups.iter().any(|g| g.len() > 1), "stored-golden fidelity must exercise a real pack");
    let mut scores = (0..members).map(|_| Fidelity::default()).collect::<Vec<_>>();
    for range in groups {
        let first = range.start;
        let mut sequences = placements[range.clone()].iter_mut().enumerate().map(|(j, p)| {
            let i = first + j;
            (p, &tokens[start(i)..start(i) + lengths[i]])
        }).collect::<Vec<_>>();
        let (segments, outcomes) = engine.prefill_packed(&mut sequences, &mut |_, _| Ok(Ok(())))?;
        for outcome in outcomes { outcome?; }
        for (j, segment) in segments.into_iter().enumerate() {
            let i = first + j;
            engine.packed_check_logits(segment, &mut |offset, logits| {
                let logits = logits.to_host(engine.library)?;
                let mut bytes = vec![0u8; logits.len() * 4];
                reference.seek(SeekFrom::Start(((start(i) + offset) * vocab * 4) as u64))?;
                reference.read_exact(&mut bytes)?;
                let golden = bytes.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
                scores[i].add(&logits, &golden, vocab)
            })?;
        }
    }
    for p in placements { allocator.release(p); }
    for (i, score) in scores.iter().enumerate() {
        println!("packed stored golden: sequence {i}: {} rows at {}: mean KL {:.8}, top-1 {:.4}; bars 0.05/0.93",
            score.rows, start(i), score.kl_sum / score.rows as f64, score.top1 as f64 / score.rows as f64);
    }
    for score in scores { score.gate()?; }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stored_golden_gate_uses_all_member_rows_and_rejects_bad_shapes() -> Result<()> {
        let mut score = Fidelity::default();
        score.add(&[4.0, 0.0, 0.0, 4.0], &[4.0, 0.0, 0.0, 4.0], 2)?;
        assert_eq!((score.rows, score.top1), (2, 2));
        score.gate()?;
        assert!(score.add(&[0.0], &[0.0, 1.0], 2).is_err());
        assert!(score.add(&[f32::NAN, 0.0], &[0.0, 1.0], 2).is_err());
        let mut changed = Fidelity::default();
        changed.add(&[0.0, 4.0], &[4.0, 0.0], 2)?;
        assert!(changed.gate().is_err());
        assert!(Fidelity::default().gate().is_err());
        Ok(())
    }

    #[test]
    fn stored_sequences_wrap_independently() {
        assert_eq!(sequence(&[1, 2, 3], 0, 4), [1, 2, 3, 1]);
        assert_eq!(sequence(&[1, 2, 3], 1, 4), [2, 3, 1, 2]);
    }
}
