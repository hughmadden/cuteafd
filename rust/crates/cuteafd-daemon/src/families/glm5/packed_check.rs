//! Stored-token packed/singleton byte-exact gate, including a resumed prompt.
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

pub(crate) fn check(engine: &GlmEngine<'_>, tokens: &[u32], members: usize,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime) -> Result<()> {
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
    ensure!(groups.iter().any(|group| group.len() > 1),
        "packed-check cannot exercise a pack with installed arithmetic row cap {}", limits.rows);
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
    let mut equal = true;
    for (i, placement) in alone.iter_mut().enumerate() {
        let logits = engine.prefill_device(placement, chunk(i), transport.as_deref_mut().map(|t| (t, runtime)), 1)?
            .context("packed-check singleton head")?.row_host(engine.library, 0)?;
        let same_logits = packed_logits[i].iter().map(|f| f.to_bits()).eq(logits.iter().map(|f| f.to_bits()));
        let same_state = state(engine, &packed[i])? == state(engine, placement)?;
        println!("packed check: sequence {i}: {} rows at {}: logits {}, live cache {}", lengths[i], prefix(i),
            if same_logits { "identical" } else { "DIFFER" }, if same_state { "identical" } else { "DIFFER" });
        equal &= same_logits && same_state;
    }
    for placement in packed.into_iter().chain(alone) { allocator.release(placement); }
    println!("packed check: {members} sequences, {packed_groups} packed groups, {} singleton groups; {}",
        groups.len() - packed_groups, if equal { "every sequence identical" } else { "DIFFER" });
    ensure!(equal, "packed prefill differs from its singleton reference");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stored_sequences_wrap_independently() {
        assert_eq!(sequence(&[1, 2, 3], 0, 4), [1, 2, 3, 1]);
        assert_eq!(sequence(&[1, 2, 3], 1, 4), [2, 3, 1, 2]);
    }
}
