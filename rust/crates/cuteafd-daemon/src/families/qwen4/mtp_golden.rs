//! Speculation checks for Qwen 3.8 Flash Next (qwen4-golden): the MTP layer
//! against the torch reference (python/reference/families/qwen4/mtp.py), greedy
//! MTP speculation against plain greedy decoding, and step costs by rows.
use super::engine::{Allocator, MtpGroup, MtpOut, MtpRow, MtpSource, MtpTokens, Qwen4Engine, Qwen4Placement, DECODE_ROWS};
use super::speculate::{self, DraftSeq, DraftTiming, MtpSeq, Verified};
use super::{bf16s, similarity, GoldenArgs, Opened};
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use std::time::Instant;

fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

fn words(path: &std::path::Path) -> Result<Vec<u32>> {
    Ok(std::fs::read(path).with_context(|| format!("reading {}", path.display()))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect())
}

fn nll(logits: &[f32], target: u32) -> f64 {
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&l| (l as f64 - top).exp()).sum();
    top + sum.ln() - logits[target as usize] as f64
}

/// The MTP layer on the golden target streams (teacher forced, decode-shaped
/// steps of 64 rows) against the torch reference in `dir`.
pub(super) fn mtp_oracle(args: &GoldenArgs, opened: &Opened, engine: &Qwen4Engine<'_>, dir: &std::path::Path)
    -> Result<()> {
    let cfg = &opened.cfg;
    let (h, vocab) = (cfg.hidden, cfg.vocab_size);
    let tokens = words(&args.golden.join("tokens.bin"))?;
    let t = tokens.len();
    let n = t - 1;
    let last = cfg.layers - 1;
    let streams = std::fs::read(args.golden.join(format!("layer{last:02}.bin")))?;
    let row = 4 * h * 2;
    ensure!(streams.len() == t * row, "golden layer {last} streams do not match the tokens");
    let source = DeviceAllocation::new(&opened.library, streams.len())?;
    opened.library.copy_h2d(source.buffer, &streams)?;
    let ref_streams = std::fs::read(dir.join("mtp_streams.bin"))?;
    let ref_argmax = words(&dir.join("mtp_argmax.bin"))?;
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let logit_rows: Vec<usize> = meta["logit_rows"].as_array().context("logit_rows")?.iter()
        .map(|r| r.as_u64().map(|r| r as usize).context("row")).collect::<Result<_>>()?;
    let ref_logits: Vec<f32> = std::fs::read(dir.join("mtp_logits.bin"))?.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    ensure!(ref_argmax.len() == n && ref_streams.len() == n * row, "reference rows do not match the golden");
    let mut allocator = Allocator::new(engine.pages, engine.slots, cfg);
    let placement = allocator.admit(t)?;
    engine.start(&placement)?;
    let (mut agree, mut ours_ok, mut ref_ok, mut scored, mut ours_nll, mut row_cos) = (0, 0, 0, 0, 0f64, Vec::new());
    let mut logit_cos = Vec::new();
    let started = Instant::now();
    for first in (0..n).step_by(DECODE_ROWS) {
        let end = (first + DECODE_ROWS).min(n);
        let rows: Vec<MtpRow> = (first..end)
            .map(|p| MtpRow { position: p, token: tokens[p + 1], source: p as i32 }).collect();
        let next: Vec<u32> = rows.iter().map(|r| r.token).collect();
        let heads: Vec<usize> = (0..rows.len()).collect();
        let (best, logits) = engine.mtp_step(true, &[MtpGroup { placement: &placement, rows }],
            MtpSource::Buffer(source.buffer.ptr), &heads, MtpTokens::Host(&next), MtpOut::Download { logits: true })?;
        let logits = logits.context("logits")?;
        let ours = bf16s(&engine.mtp_output(end - first)?);
        let theirs = bf16s(&ref_streams[first * row..end * row]);
        for (a, b) in ours.chunks_exact(4 * h).zip(theirs.chunks_exact(4 * h)) {
            row_cos.push(similarity(a, b).0);
        }
        for (i, p) in (first..end).enumerate() {
            let (token, _) = best[i];
            let l = &logits[i * vocab..(i + 1) * vocab];
            ensure!(token == argmax(l), "GPU and host argmax differ at row {p}");
            agree += usize::from(token == ref_argmax[p]);
            if p + 2 < t {
                scored += 1;
                ours_ok += usize::from(token == tokens[p + 2]);
                ref_ok += usize::from(ref_argmax[p] == tokens[p + 2]);
                ours_nll += nll(l, tokens[p + 2]);
            }
            if let Some(k) = logit_rows.iter().position(|&r| r == p) {
                logit_cos.push(similarity(l, &ref_logits[k * vocab..(k + 1) * vocab]).0);
            }
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    row_cos.sort_by(f64::total_cmp);
    logit_cos.sort_by(f64::total_cmp);
    println!("MTP oracle: {n} rows in {} steps ({seconds:.2} s) | output streams cosine median {:.6} p1 {:.6} worst \
        {:.6} | logits cosine ({} rows) median {:.6} worst {:.6} | greedy draft agreement with the reference {:.2}% \
        | accuracy on x(t+2): engine {:.2}% reference {:.2}% | mean NLL engine {:.4} reference {:.4}",
        n.div_ceil(DECODE_ROWS), row_cos[row_cos.len() / 2], row_cos[row_cos.len() / 100], row_cos[0],
        logit_cos.len(), logit_cos[logit_cos.len() / 2], logit_cos[0], 100.0 * agree as f64 / n as f64,
        100.0 * ours_ok as f64 / scored as f64, 100.0 * ref_ok as f64 / scored as f64, ours_nll / scored as f64,
        meta["mean_nll"].as_f64().unwrap_or(f64::NAN));
    allocator.release(placement);
    Ok(())
}

/// Prefills `prompt` (with the MTP's canonical rows when `mtp`); returns the
/// last row's logits.
fn prefill(engine: &Qwen4Engine<'_>, placement: &mut Qwen4Placement, prompt: &[u32], mtp: Option<&mut MtpSeq>)
    -> Result<Vec<f32>> {
    let mut logits = None;
    let mut mtp = mtp;
    let mut done = 0;
    for chunk in prompt.chunks(engine.prefill_rows) {
        let start = placement.len;
        logits = engine.prefill(placement, chunk)?;
        done += chunk.len();
        if let Some(seq) = mtp.as_deref_mut() {
            speculate::prefill_chunk(engine, placement, start, chunk, prompt.get(done).copied(), seq)?;
        }
    }
    logits.context("prefill needs every layer")
}

/// Greedy decoding of `count` tokens after the golden prompt, plain and with
/// MTP speculation at depth `depth` (`spec` verify, commit, MTP canonical
/// rows); the token streams must match. Then verify-step and MTP-step
/// costs by rows.
pub(super) fn spec_decode(args: &GoldenArgs, opened: &Opened, engine: &Qwen4Engine<'_>, count: usize, depth: usize)
    -> Result<()> {
    let cfg = &opened.cfg;
    let vocab = cfg.vocab_size;
    let mut prompt = words(&args.golden.join("tokens.bin"))?;
    if let Some(p) = args.prefill {
        prompt.truncate(p);
    }
    let mut allocator = Allocator::new(engine.pages, engine.slots, cfg);
    // Plain greedy.
    let mut placement = allocator.admit(prompt.len() + count + DECODE_ROWS)?;
    let mut next = argmax(&prefill(engine, &mut placement, &prompt, None)?);
    let mut plain = vec![next];
    // Top-2 logit margin of every plain step (near-ties may flip under other row counts).
    let mut margins = vec![f32::INFINITY];
    let started = Instant::now();
    while plain.len() < count {
        let logits = engine.verify(&mut [(&mut placement, &[next][..])], None)?
            .context("decode needs every layer")?;
        next = argmax(&logits);
        let second = logits.iter().enumerate().filter(|&(i, _)| i != next as usize).map(|(_, &l)| l)
            .fold(f32::NEG_INFINITY, f32::max);
        margins.push(logits[next as usize] - second);
        plain.push(next);
    }
    let plain_s = started.elapsed().as_secs_f64();
    allocator.release(placement);
    // MTP speculation.
    let mut placement = allocator.admit(prompt.len() + count + DECODE_ROWS)?;
    let mut seq = MtpSeq::default();
    next = argmax(&prefill(engine, &mut placement, &prompt, Some(&mut seq))?);
    seq.close(next);
    let mut out = vec![next];
    let (mut cycles, mut proposed, mut accepted) = (0usize, 0usize, 0usize);
    let mut by_position = vec![(0usize, 0usize); depth.max(1)];
    let mut timing = DraftTiming::default();
    let (mut verify_s, mut commit_s) = (0f64, 0f64);
    let started = Instant::now();
    while out.len() < count {
        let room = (count - out.len()).min(DECODE_ROWS - 1);
        let want = depth.min(room.saturating_sub(1));
        let drafts = speculate::draft(engine,
            &mut [DraftSeq { placement: &placement, seq: &mut seq, depth: want }], &mut timing)?.remove(0);
        let rows: Vec<u32> = std::iter::once(next).chain(drafts.iter().copied()).collect();
        let (start, history) = (placement.len, placement.history.clone());
        let timer = Instant::now();
        let spec = rows.len() > 1;
        let logits = if spec {
            engine.verify_spec(&mut [(&mut placement, &rows[..])], None)?
        } else {
            engine.verify(&mut [(&mut placement, &rows[..])], None)?
        }.context("decode needs every layer")?;
        verify_s += timer.elapsed().as_secs_f64();
        let mut kept = 0;
        let mut after = 0;
        for j in 0..rows.len() {
            let token = argmax(&logits[j * vocab..(j + 1) * vocab]);
            kept = j + 1;
            after = token;
            out.push(token);
            if j < drafts.len() {
                by_position[j].0 += 1;
                by_position[j].1 += usize::from(token == drafts[j]);
            }
            if drafts.get(j) != Some(&token) || out.len() >= count {
                break;
            }
        }
        cycles += 1;
        proposed += drafts.len();
        accepted += kept - 1;
        let timer = Instant::now();
        speculate::accept(engine, &mut [Verified { placement: &mut placement, seq: &mut seq, start, history,
            rows: &rows, first_row: 0, kept: Some((kept, after)) }], spec, true)?;
        // SAFETY: the engine owns this stream.
        unsafe { opened.library.cuda_stream_synchronize(engine.stream)? };
        commit_s += timer.elapsed().as_secs_f64();
        next = after;
    }
    let mtp_s = started.elapsed().as_secs_f64();
    out.truncate(count);
    let same = out.iter().zip(&plain).take_while(|(a, b)| a == b).count();
    let mut sorted: Vec<f32> = margins.iter().copied().filter(|m| m.is_finite()).collect();
    sorted.sort_by(f32::total_cmp);
    println!("spec decode: {count} greedy tokens, MTP depth {depth}: {}", if same == count {
        "identical to plain greedy".to_string()
    } else {
        format!("diverges from plain greedy at token {same} (plain top-2 logit margin there {:.4}; median margin \
            {:.3}, {} of {} plain steps below 0.05)", margins[same], sorted[sorted.len() / 2],
            sorted.iter().filter(|&&m| m < 0.05).count(), sorted.len())
    });
    println!("  plain {:.1} tok/s ({:.2} ms/token) | MTP {:.1} tok/s over {cycles} cycles: {:.2} tokens/cycle, \
        accepted {accepted} of {proposed} drafts ({:.1}%) | per position: {}", count as f64 / plain_s,
        1e3 * plain_s / count as f64, count as f64 / mtp_s, count as f64 / cycles as f64,
        100.0 * accepted as f64 / proposed.max(1) as f64,
        by_position.iter().enumerate().filter(|(_, (n, _))| *n > 0)
            .map(|(j, (n, a))| format!("d{}: {:.1}%", j + 1, 100.0 * *a as f64 / *n as f64))
            .collect::<Vec<_>>().join(" "));
    println!("  per cycle: MTP steps {:.2} ms ({:.2} steps), verify {:.2} ms, commit+stash {:.2} ms",
        1e3 * timing.seconds / cycles as f64, timing.steps as f64 / cycles as f64, 1e3 * verify_s / cycles as f64,
        1e3 * commit_s / cycles as f64);
    // Step costs by rows (speculative verifies leave the state as it was).
    let pos = placement.len;
    let sample: Vec<u32> = out.iter().copied().cycle().take(DECODE_ROWS).collect();
    let mut line = String::from("  verify step (spec) ms by rows:");
    for rows in (1..=33).chain([40, 48, 56, 64]) {
        let tokens = &sample[..rows];
        let mut times = Vec::new();
        for _ in 0..7 {
            let mut p = placement.clone();
            let timer = Instant::now();
            engine.verify_spec(&mut [(&mut p, tokens)], None)?;
            times.push(timer.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        line += &format!(" {rows}:{:.2}", 1e3 * times[times.len() / 2]);
    }
    println!("{line}");
    let mut line = String::from("  MTP step ms by rows (with head):");
    for rows in [1usize, 2, 4, 8] {
        let group = MtpGroup { placement: &placement, rows: (0..rows)
            .map(|i| MtpRow { position: pos + i, token: sample[i], source: 0 }).collect() };
        let heads: Vec<usize> = (0..rows).collect();
        let mut times = Vec::new();
        for _ in 0..7 {
            let timer = Instant::now();
            engine.mtp_step(true, std::slice::from_ref(&group), MtpSource::Pending, &heads,
                MtpTokens::Host(&sample[..rows]), MtpOut::Download { logits: false })?;
            times.push(timer.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        line += &format!(" {rows}:{:.2}", 1e3 * times[times.len() / 2]);
    }
    println!("{line}");
    let mut times = Vec::new();
    for _ in 0..7 {
        let timer = Instant::now();
        engine.commit(&[(placement.slot, 0, 4)])?;
        // SAFETY: the engine owns this stream.
        unsafe { opened.library.cuda_stream_synchronize(engine.stream)? };
        times.push(timer.elapsed().as_secs_f64());
    }
    times.sort_by(f64::total_cmp);
    println!("  commit (36 GDN layers + PLE, 4 rows, one sequence) {:.3} ms", 1e3 * times[3]);
    allocator.release(placement);
    Ok(())
}
