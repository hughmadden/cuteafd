//! Native MTP speculation for Qwen 3.8 Flash Next: the MTP's canonical
//! history, its draft chain, and verify-by-replay of the target.
//!
//! The MTP layer (vLLM `Qwen4ExpMultiTokenPredictor`) keeps its own K/V
//! records: position `p` holds the pair (target pre-mixer streams at `p`,
//! token `p + 1`), which predicts token `p + 2`. Target rows whose next token
//! is known wait in the engine's stash (`pending`, per state slot) until the
//! next draft cycle; a cycle's first MTP step runs every pending row (the
//! canonical history) and drafts from each sequence's last one; later steps
//! chain on the MTP's own output streams and the previous draft (positions
//! past the committed ones are rewritten by the next cycle's canonical rows).
//!
//! The target verifies `[next, d_0, .., d_{n-1}]` in one speculative step
//! (GDN and PLE record replay inputs instead of advancing); the accepted rows
//! are committed (`qwen4_gdn_commit` / `qwen4_ple_commit`), the placement and
//! n-gram history rewound, and the kept rows' streams stashed for the MTP.
use super::engine::{MtpGroup, MtpRow, MtpSource, Qwen4Engine, Qwen4Placement, DECODE_ROWS, MTP_PENDING_ROWS};
use anyhow::{ensure, Result};
use cuteafd_loader::families::qwen4::NgramHistory;

/// A sequence's MTP bookkeeping: stash rows (position, token at position + 1)
/// whose pairs are not in the MTP's history yet.
#[derive(Debug, Clone, Default)]
pub(crate) struct MtpSeq {
    pub pending: Vec<(usize, u32)>,
    /// The last stash row's token is not known yet (the prompt's last row
    /// waits for the first sampled token).
    pub open: bool,
}

impl MtpSeq {
    /// The first sampled token completes the prompt's last pair.
    pub fn close(&mut self, token: u32) {
        if self.open {
            if let Some(last) = self.pending.last_mut() {
                last.1 = token;
            }
            self.open = false;
        }
    }
}

/// After the prefill of `chunk` (tokens from position `start`): the MTP's
/// canonical rows for every pair whose next token is known (`next`: the token
/// after the chunk, if known), and the chunk's last row stashed otherwise.
pub(crate) fn prefill_chunk(engine: &Qwen4Engine<'_>, embed: &dyn Fn(&[u32]) -> Result<Vec<u8>>,
    placement: &Qwen4Placement, start: usize, chunk: &[u32], next: Option<u32>, seq: &mut MtpSeq) -> Result<()> {
    let n = chunk.len();
    ensure!(n > 0 && seq.pending.is_empty(), "MTP prefill of an empty chunk or behind pending rows");
    let known = if next.is_some() { n } else { n - 1 };
    if next.is_none() {
        // Stash first: the MTP step reuses the target streams' buffer.
        engine.mtp_stash(false, &[(placement.slot, n - 1, 1, 0)])?;
        seq.pending.push((start + n - 1, 0));
        seq.open = true;
    }
    if known == 0 {
        return Ok(());
    }
    let tokens: Vec<u32> = (0..known).map(|i| chunk.get(i + 1).copied().or(next).unwrap_or_default()).collect();
    let rows = (0..known).map(|i| MtpRow { position: start + i, token: tokens[i], source: i as i32 }).collect();
    engine.mtp_step(false, &[MtpGroup { placement, rows }], MtpSource::Target, &[], &embed(&tokens)?, false)?;
    Ok(())
}

/// One sequence's part of a draft cycle.
pub(crate) struct DraftSeq<'p> {
    pub placement: &'p Qwen4Placement,
    pub seq: &'p mut MtpSeq,
    /// Drafts wanted (0: canonical rows only).
    pub depth: usize,
}

/// Seconds spent in the cycle's MTP steps and their count.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct DraftTiming {
    pub steps: usize,
    pub seconds: f64,
}

/// Runs the pending rows of every sequence (canonical history) and drafts up
/// to each sequence's depth; returns each sequence's drafts.
pub(crate) fn draft(engine: &Qwen4Engine<'_>, embed: &dyn Fn(&[u32]) -> Result<Vec<u8>>, seqs: &mut [DraftSeq<'_>],
    timing: &mut DraftTiming) -> Result<Vec<Vec<u32>>> {
    let mut drafts = vec![Vec::new(); seqs.len()];
    let started = std::time::Instant::now();
    // Step 0: every pending row; heads on the last row of the drafting sequences.
    let mut groups = Vec::new();
    let mut heads = Vec::new();
    let mut head_of = vec![None; seqs.len()];
    let mut tokens = Vec::new();
    let mut last_position = vec![0usize; seqs.len()];
    let mut row = 0usize;
    for (i, s) in seqs.iter().enumerate() {
        if s.seq.pending.is_empty() {
            ensure!(s.depth == 0, "a drafting sequence has no pending MTP rows");
            continue;
        }
        ensure!(!s.seq.open, "a sequence drafts before its prompt's last pair is closed");
        let rows: Vec<MtpRow> = s.seq.pending.iter().enumerate()
            .map(|(j, &(position, token))| MtpRow { position, token, source: j as i32 }).collect();
        tokens.extend(rows.iter().map(|r| r.token));
        row += rows.len();
        last_position[i] = rows.last().map_or(0, |r| r.position);
        if s.depth > 0 {
            head_of[i] = Some(heads.len());
            heads.push(row - 1);
        }
        groups.push(MtpGroup { placement: s.placement, rows });
    }
    ensure!(row <= DECODE_ROWS, "{row} pending MTP rows exceed a decode step");
    if groups.is_empty() {
        return Ok(drafts);
    }
    let (best, _) = engine.mtp_step(true, &groups, MtpSource::Pending, &heads, &embed(&tokens)?, false)?;
    timing.steps += 1;
    drop(groups);
    for s in seqs.iter_mut() {
        s.seq.pending.clear();
    }
    for (i, h) in head_of.iter().enumerate() {
        if let Some(h) = h {
            drafts[i].push(best[*h].0);
        }
    }
    // Chain steps: one row per sequence still drafting, reading its previous head row.
    let depth = seqs.iter().map(|s| s.depth).max().unwrap_or(0);
    let mut previous_head = head_of.clone().into_iter().map(|h| h.map(|h| heads[h])).collect::<Vec<_>>();
    for j in 1..depth {
        let mut groups = Vec::new();
        let mut tokens = Vec::new();
        let mut members = Vec::new();
        for (i, s) in seqs.iter().enumerate() {
            let Some(source) = previous_head[i] else { continue };
            if s.depth <= j {
                continue;
            }
            let token = *drafts[i].last().expect("a draft per step");
            tokens.push(token);
            members.push(i);
            groups.push(MtpGroup { placement: s.placement, rows: vec![MtpRow { position: last_position[i] + j, token,
                source: source as i32 }] });
        }
        if groups.is_empty() {
            break;
        }
        let heads: Vec<usize> = (0..groups.len()).collect();
        let (best, _) = engine.mtp_step(true, &groups, MtpSource::Chain, &heads, &embed(&tokens)?, false)?;
        timing.steps += 1;
        previous_head = vec![None; seqs.len()];
        for (r, &i) in members.iter().enumerate() {
            drafts[i].push(best[r].0);
            previous_head[i] = Some(r);
        }
    }
    timing.seconds += started.elapsed().as_secs_f64();
    Ok(drafts)
}

/// A sequence's verified rows: the tokens it verified from `start` (next
/// token, then drafts), its n-gram history before the step, and its first
/// row in the step.
pub(crate) struct Verified<'p> {
    pub placement: &'p mut Qwen4Placement,
    pub seq: &'p mut MtpSeq,
    pub start: usize,
    pub history: NgramHistory,
    pub rows: &'p [u32],
    pub first_row: usize,
    /// Rows kept (the next token plus the accepted drafts) and the token
    /// after them (the correction or bonus); None drops the sequence (finished).
    pub kept: Option<(usize, u32)>,
}

/// After a speculative verify: commits each sequence's kept rows to the GDN
/// and PLE state, rewinds its placement, and stashes the kept rows' streams
/// with their next tokens for the MTP.
pub(crate) fn accept(engine: &Qwen4Engine<'_>, verified: &mut [Verified<'_>], spec: bool, mtp: bool) -> Result<()> {
    let mut commits = Vec::new();
    let mut stash = Vec::new();
    for v in verified.iter_mut() {
        // Finished sequences are released: nothing to commit or stash.
        let Some((kept, after)) = v.kept else { continue };
        ensure!(kept >= 1 && kept <= v.rows.len(), "kept {kept} of {} rows", v.rows.len());
        if spec {
            commits.push((v.placement.slot, v.first_row, kept));
            engine.rewind(v.placement, v.start, v.history.clone(), &v.rows[..kept])?;
        } else {
            ensure!(kept == v.rows.len(), "a plain step keeps every row");
        }
        if !mtp {
            continue;
        }
        let at = v.seq.pending.len();
        ensure!(at + kept <= MTP_PENDING_ROWS, "the MTP stash of a sequence overflows");
        stash.push((v.placement.slot, v.first_row, kept, at));
        for j in 0..kept {
            let next = if j + 1 < kept { v.rows[j + 1] } else { after };
            v.seq.pending.push((v.start + j, next));
        }
    }
    engine.commit(&commits)?;
    if mtp {
        engine.mtp_stash(true, &stash)?;
    }
    Ok(())
}
