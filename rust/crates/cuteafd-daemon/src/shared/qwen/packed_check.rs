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

#[allow(clippy::too_many_arguments)]
pub(crate) fn check<P: Placement, D: DraftState>(library: &NativeLibrary, prompts: &[Vec<u32>],
    mut packed: Vec<P>, mut alone: Vec<P>, limits: Limits, neural: bool, exact: bool,
    mut single: impl FnMut(&mut PrefillSeq<'_, P, D>) -> Result<Vec<f32>>,
    mut pass: impl FnMut(&mut [PrefillSeq<'_, P, D>], &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome)
        -> Result<Vec<Result<()>>>,
    mut snapshot: impl FnMut(&P) -> Result<Vec<u8>>) -> Result<()> {
    let n = prompts.len();
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
    ensure!(groups.iter().any(|r| r.len() > 1), "--packed-check exercised no packed group (exact route unknown or too narrow)");
    let mut logits = vec![Vec::new(); n];
    for range in groups {
        let first = range.start;
        let mut sequences = packed[range.clone()].iter_mut().zip(&mut pd[range]).enumerate().map(|(j, (placement, draft))| {
            let i = first + j;
            PrefillSeq { placement, tokens: &prompts[i][chunks[i].0..], start: chunks[i].0,
                next: None, draft, neural }
        }).collect::<Vec<_>>();
        if sequences.len() == 1 {
            logits[first] = single(&mut sequences[0])?;
        } else {
            for outcome in pass(&mut sequences, &mut |j, head| {
                logits[first + j] = head.row_host(library, 0)?;
                Ok(Ok(()))
            })? { outcome?; }
        }
    }
    for i in 0..n {
        let head = single(&mut PrefillSeq { placement: &mut alone[i], tokens: &prompts[i][chunks[i].0..],
            start: chunks[i].0, next: None, draft: &mut ad[i], neural })?;
        let equal = logits[i].len() == head.len() && logits[i].iter().zip(&head).all(|(a, b)| a.to_bits() == b.to_bits());
        let state = snapshot(&packed[i])? == snapshot(&alone[i])?;
        let worst = logits[i].iter().zip(&head).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let (kl, _) = crate::shared::golden_scores::scored_kl(&head, &logits[i], head.len())?;
        let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
        let top1 = top(&head) == top(&logits[i]);
        println!("packed check: member {i}, {} rows at {}: logits {equal}, state {state}, max |diff| {worst:.3e}, KL {kl:.6}, top1 {top1}", chunks[i].1, chunks[i].0);
        if exact { ensure!(equal && state, "packed member {i} differs from singleton"); }
        else { ensure!(kl <= 0.03 && top1, "unrestricted packed member {i} exceeds KL/top1 bar"); }
    }
    println!("packed check: every member passed ({})", if exact { "byte-identical" } else { "KL/top1" });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_check_has_private_wrapping_prompts_and_resumed_tail() {
        let p = prompts(&[1, 2, 3], 8).unwrap();
        assert_eq!(p.iter().map(Vec::len).collect::<Vec<_>>(), [7, 9, 29, 40, 64, 65, 255, 74]);
        assert_ne!(&p[0][..3], &p[1][..3]);
        assert!(prompts(&[], 2).is_err());
        assert!(prompts(&[1], 1).is_err());
    }
}
