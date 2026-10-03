//! Speculation checks for GLM 5.3 Flash (glmf-golden): the DFlash2 drafter
//! against the torch reference, drafts against the target, KDA
//! verify-by-replay against serial steps, and the verify-step cost by rows.
use super::engine::{Allocator, GlmfEngine, GlmfPlacement};
use super::{bf16s, similarity, GoldenArgs, Opened};
use crate::families::glm5::dflash::{ContextRow, DraftSeq, TAP_ROWS};
use anyhow::{ensure, Context, Result};
use std::time::Instant;

fn tokens(args: &GoldenArgs) -> Result<Vec<u32>> {
    Ok(std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect())
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

/// BF16 mean of the four mHC streams of `rows` rows of a golden layer
/// ([T, 4, hidden]): FP32 sum in stream order, then the division, as the
/// engine's tap kernel.
fn stream_mean(layer: &[u8], first: usize, rows: usize, hidden: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * hidden * 2);
    let value = |i: usize| f32::from_bits(u32::from(u16::from_le_bytes([layer[2 * i], layer[2 * i + 1]])) << 16);
    for r in first..first + rows {
        for c in 0..hidden {
            let mut sum = 0f32;
            for k in 0..4 {
                sum += value((r * 4 + k) * hidden + c);
            }
            // Round to nearest even (the tap kernel's __float2bfloat16; no NaNs here).
            let bits = (sum / 4.0).to_bits();
            out.extend_from_slice(&(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16).to_le_bytes());
        }
    }
    out
}

/// Prefills `tokens` in prefill-row chunks, feeding each chunk's tapped tail to
/// the drafter's ring `slot`; returns the last row's logits.
fn prefill_with_taps(engine: &GlmfEngine<'_>, placement: &mut GlmfPlacement, tokens: &[u32],
    slot: usize) -> Result<Vec<f32>> {
    let mut logits = None;
    for chunk in tokens.chunks(engine.prefill_capacity()) {
        let start = placement.len;
        logits = engine.prefill(placement, chunk, None)?;
        if let Some(drafter) = &engine.drafter {
            let n = chunk.len().min(TAP_ROWS);
            let first = start + chunk.len() - n;
            drafter.update(&(0..n).map(|r| ContextRow { tap_row: r, slot, position: first + r }).collect::<Vec<_>>())?;
        }
    }
    logits.context("the prefill needs every layer")
}

/// Runs the drafter alone on the golden taps at reference.py's anchor
/// positions and compares tokens, selector features and final-norm rows.
pub(super) fn draft_oracle(args: &GoldenArgs, opened: &Opened, engine: &GlmfEngine<'_>, dir: &std::path::Path)
    -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft-oracle needs --draft")?;
    let (hidden, block, drafts_per) = (opened.cfg.hidden, drafter.block(), drafter.drafts());
    let dspark = matches!(drafter, super::dspark::Drafter::Dspark(_));
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let positions: Vec<usize> = meta["positions"].as_array().context("positions")?.iter()
        .map(|p| p.as_u64().map(|p| p as usize).context("position")).collect::<Result<_>>()?;
    let words = |name: &str| -> Result<Vec<u32>> {
        Ok(std::fs::read(dir.join(name))?.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect())
    };
    let ref_tokens = words("drafts.bin")?;
    // DFlash2: selector features [N, 7, 4]; dSpark: confidence [N, 8].
    let ref_features = words(if dspark { "confidence.bin" } else { "features.bin" })?;
    let ref_hidden = bf16s(&std::fs::read(dir.join("hidden.bin"))?);
    let tokens = tokens(args)?;
    let layers: Vec<Vec<u8>> = drafter.taps().iter()
        .map(|l| std::fs::read(args.golden.join(format!("layer{l:02}.bin")))).collect::<std::io::Result<_>>()?;
    let row = hidden * 2;
    let width = layers.len() * row;
    let (mut done, mut exact, mut first, mut matched, mut worst, mut feature_error) = (0usize, 0, 0, 0, 1f64, 0f64);
    let mut draft_seconds = 0f64;
    for (index, &position) in positions.iter().enumerate() {
        while done < position {
            let n = (position - done).min(TAP_ROWS);
            let means: Vec<Vec<u8>> = layers.iter().map(|layer| stream_mean(layer, done, n, hidden)).collect();
            let mut taps = vec![0u8; n * width];
            for r in 0..n {
                for (i, mean) in means.iter().enumerate() {
                    taps[r * width + i * row..][..row].copy_from_slice(&mean[r * row..][..row]);
                }
            }
            drafter.put_taps(&taps)?;
            drafter.update(&(0..n).map(|r| ContextRow { tap_row: r, slot: 0, position: done + r }).collect::<Vec<_>>())?;
            done += n;
        }
        let anchor = tokens[position];
        let timer = Instant::now();
        let draft = drafter.draft_device(&[DraftSeq { slot: 0, anchor, position, valid_from: 0 }],
            &engine.embedding, engine.weights.head.buffer.ptr)?.remove(0);
        draft_seconds += timer.elapsed().as_secs_f64();
        let reference = &ref_tokens[index * drafts_per..][..drafts_per];
        exact += usize::from(draft.tokens == reference);
        first += usize::from(draft.tokens[0] == reference[0]);
        matched += draft.tokens.iter().zip(reference).take_while(|(a, b)| a == b).count();
        let (cosine, _) = similarity(&bf16s(&drafter.last_hidden(1)?), &ref_hidden[index * block * hidden..][..block * hidden]);
        worst = worst.min(cosine);
        if dspark {
            // Confidence of the rows whose previous token agrees (the Markov embedding is the same).
            for k in 0..drafts_per {
                if k > 0 && draft.tokens[k - 1] != reference[k - 1] {
                    break;
                }
                let theirs = f32::from_bits(ref_features[index * drafts_per + k]);
                feature_error = feature_error.max(f64::from((draft.confidence[k] - theirs).abs()));
            }
        } else if draft.tokens[0] == reference[0] {
            let theirs = f32::from_bits(ref_features[index * drafts_per * 4]);
            feature_error = feature_error.max(f64::from((draft.features[0][0] - theirs).abs()));
        }
        if draft.tokens != reference {
            println!("position {position}: engine {:?} reference {reference:?}", draft.tokens);
        }
    }
    let n = positions.len();
    println!("draft oracle ({}): {n} anchors, identical drafts {exact}/{n}, first draft {first}/{n}, matching prefix \
        {:.2} of {drafts_per}, worst final-norm cosine {worst:.6}, {} max error {feature_error:.4}, {:.2} ms/draft",
        drafter.name(), matched as f64 / n as f64, if dspark { "confidence" } else { "first-margin" },
        draft_seconds * 1e3 / n as f64);
    Ok(())
}

/// [`crate::families::glm5::dflash::replay`] on the golden taps (mHC stream means).
pub(super) fn draft_replay(args: &GoldenArgs, opened: &Opened, engine: &GlmfEngine<'_>, start: usize) -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft-replay needs --draft")?;
    let (tokens, greedy) = crate::families::glm5::dflash::golden_sequence(&args.golden, opened.cfg.vocab_size)?;
    let hidden = opened.cfg.hidden;
    let layers: Vec<Vec<u8>> = drafter.taps().iter()
        .map(|l| std::fs::read(args.golden.join(format!("layer{l:02}.bin")))).collect::<std::io::Result<_>>()?;
    let row = hidden * 2;
    let taps = |first: usize, n: usize| -> Result<Vec<u8>> {
        let means: Vec<Vec<u8>> = layers.iter().map(|layer| stream_mean(layer, first, n, hidden)).collect();
        let mut taps = vec![0u8; n * layers.len() * row];
        for r in 0..n {
            for (i, mean) in means.iter().enumerate() {
                taps[(r * layers.len() + i) * row..][..row].copy_from_slice(&mean[r * row..][..row]);
            }
        }
        Ok(taps)
    };
    crate::families::glm5::dflash::replay(drafter.replay(), &tokens, &greedy, &taps, &|t| engine.embedding.host_rows(t),
        engine.weights.head.buffer.ptr, start)
}

/// Prefills the golden prompt's first --prefill tokens, then decodes one row
/// per step (teacher-forced on tokens.bin, or greedy with --generate),
/// drafting with DFlash2 before every step; reports the accepted prefix per
/// step against the sequence.
pub(super) fn draft_run(args: &GoldenArgs, opened: &Opened, engine: &GlmfEngine<'_>) -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft")?;
    let mut sequence = tokens(args)?;
    let prefill = args.prefill.unwrap_or(sequence.len() / 2).min(sequence.len() - 1);
    let end = match args.generate {
        Some(n) if n > 0 => {
            sequence.truncate(prefill);
            prefill + n
        }
        _ => sequence.len(),
    };
    let greedy = end > sequence.len();
    let mut placement = Allocator::new(engine.pages, engine.slots).admit(end + 1)?;
    let started = Instant::now();
    let logits = prefill_with_taps(engine, &mut placement, &sequence[..prefill], 0)?;
    if greedy {
        sequence.push(argmax(&logits));
    }
    println!("prefill: {prefill} tokens in {:.2} s", started.elapsed().as_secs_f64());
    let (mut drafts, mut draft_seconds) = (Vec::new(), 0f64);
    let started = Instant::now();
    for position in prefill..end {
        let anchor = sequence[position];
        let timer = Instant::now();
        let draft = drafter.draft_device(&[DraftSeq { slot: 0, anchor, position, valid_from: 0 }], &engine.embedding,
            engine.weights.head.buffer.ptr)?;
        draft_seconds += timer.elapsed().as_secs_f64();
        drafts.push((position, draft.into_iter().next().context("draft")?));
        let logits = engine.verify(&mut [(&mut placement, 1)], &[anchor], None)?.context("decode needs every layer")?;
        drafter.update(&[ContextRow { tap_row: 0, slot: 0, position }])?;
        if greedy {
            sequence.push(argmax(&logits));
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let block = drafter.drafts();
    let mut histogram = vec![0usize; block + 1];
    let mut first_rank = [0usize; 16];
    for (position, draft) in &drafts {
        let truth = &sequence[position + 1..];
        if truth.len() < block {
            continue;
        }
        let accepted = draft.tokens.iter().zip(truth).take_while(|(d, t)| d == t).count();
        histogram[accepted] += 1;
        first_rank[draft.features[0][3] as usize] += 1;
    }
    let steps: usize = histogram.iter().sum();
    let accepted: usize = histogram.iter().enumerate().map(|(i, c)| i * c).sum();
    println!("drafts: {steps} steps, {accepted} drafted tokens accepted as a prefix ({:.2} per step, {:.1}% of {}), \
        histogram {histogram:?}, first-draft selector rank {first_rank:?}, {:.2} ms/draft, {:.1} ms/step",
        accepted as f64 / steps.max(1) as f64, 100.0 * accepted as f64 / (steps * block).max(1) as f64, steps * block,
        draft_seconds * 1e3 / drafts.len().max(1) as f64, seconds * 1e3 / drafts.len().max(1) as f64);
    if greedy {
        let text = cuteafd_loader::LoadedTokenizer::from_snapshot(&opened.checkpoint.snapshot)?
            .decode_ids(&sequence[prefill..], false).map(|d| d.text).unwrap_or_default();
        println!("generated: {text:?}");
    }
    Ok(())
}

/// Bytes that differ and the largest FP32 difference over the recurrent part
/// (`fp32_bytes`) of two slot-state copies.
fn state_delta(a: &[u8], b: &[u8], fp32_bytes: usize) -> (usize, f32) {
    let differ = a.iter().zip(b).filter(|(x, y)| x != y).count();
    let word = |s: &[u8], i: usize| f32::from_le_bytes(s[i * 4..i * 4 + 4].try_into().unwrap());
    let worst = (0..fp32_bytes / 4).map(|i| (word(a, i) - word(b, i)).abs()).fold(0f32, f32::max);
    (differ, worst)
}

/// See `GoldenArgs::replay_check`.
pub(super) fn replay_check(args: &GoldenArgs, engine: &GlmfEngine<'_>, rows: usize) -> Result<()> {
    let sequence = tokens(args)?;
    let prefill = args.prefill.unwrap_or(64).min(sequence.len() - rows);
    ensure!(rows >= 1 && prefill + rows <= sequence.len(), "--replay-check rows past the golden tokens");
    let embed = &sequence[prefill..prefill + rows];
    let kda_layers = engine.weights.layers.iter()
        .filter(|l| l.attention == cuteafd_loader::families::glm5_flash::GlmNextAttention::Kda).count();
    let fp32_bytes = kda_layers * engine.cfg.kda_heads * 128 * 128 * 4;
    let allocator = std::cell::RefCell::new(Allocator::new(engine.pages, engine.slots));
    let fresh = || -> Result<GlmfPlacement> {
        let mut placement = allocator.borrow_mut().admit(prefill + rows + 1)?;
        engine.prefill(&mut placement, &sequence[..prefill], None)?;
        Ok(placement)
    };
    let max_logit = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    for keep in 1..=rows {
        let mut serial = fresh()?;
        let mut serial_logits = Vec::new();
        for j in 0..keep {
            if let Some(l) = engine.verify(&mut [(&mut serial, 1)], &embed[j..j + 1], None)? {
                serial_logits.push(l);
            }
        }
        let mut spec = fresh()?;
        let start = spec.len;
        let logits = engine.verify_spec(&mut [(&mut spec, rows)], embed)?;
        engine.commit(&[(spec.slot, 0, keep)])?;
        spec.len = start + keep;
        let (differ, worst) = state_delta(&engine.slot_state(serial.slot)?, &engine.slot_state(spec.slot)?, fp32_bytes);
        let logit = logits.map(|l| (0..serial_logits.len()).map(|j|
            max_logit(&l[j * l.len() / rows..][..l.len() / rows], &serial_logits[j])).fold(0f32, f32::max));
        println!("keep {keep}/{rows}: speculative verify + commit vs {keep} serial steps: {differ} state bytes differ \
            (max FP32 |delta| {worst:.3e}); kept-row logits max |delta| {logit:?}");
        if keep == rows {
            let mut plain = fresh()?;
            engine.verify(&mut [(&mut plain, rows)], embed, None)?;
            let (differ, worst) = state_delta(&engine.slot_state(plain.slot)?, &engine.slot_state(spec.slot)?, fp32_bytes);
            println!("keep {keep}/{rows}: speculative verify + commit vs a plain {rows}-row verify: {differ} state bytes \
                differ (max FP32 |delta| {worst:.3e})");
            allocator.borrow_mut().release(plain);
        }
        allocator.borrow_mut().release(serial);
        allocator.borrow_mut().release(spec);
    }
    // Cost: plain N-row verify vs speculative verify + commit, same position.
    let mut placement = fresh()?;
    let start = placement.len;
    let mut time = |spec: bool| -> Result<f64> {
        let mut times = Vec::new();
        for _ in 0..7 {
            placement.len = start;
            let timer = Instant::now();
            if spec {
                engine.verify_spec(&mut [(&mut placement, rows)], embed)?;
                engine.commit(&[(placement.slot, 0, rows)])?;
                // SAFETY: the engine owns this stream.
                unsafe { engine.library.cuda_stream_synchronize(engine.stream)? };
            } else {
                engine.verify(&mut [(&mut placement, rows)], embed, None)?;
            }
            times.push(timer.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        Ok(times[times.len() / 2] * 1e3)
    };
    let (plain, spec) = (time(false)?, time(true)?);
    let (plain2, spec2) = (time(false)?, time(true)?);
    println!("{rows}-row step: plain {plain:.2} / {plain2:.2} ms, speculative + commit {spec:.2} / {spec2:.2} ms");
    Ok(())
}

/// See `GoldenArgs::bench_verify`.
pub(super) fn bench_verify(args: &GoldenArgs, engine: &GlmfEngine<'_>, max_rows: usize) -> Result<()> {
    let sequence = tokens(args)?;
    let count = args.bench_sequences.max(1);
    let prefill = args.prefill.unwrap_or(256).min(sequence.len() - max_rows - count);
    let mut allocator = Allocator::new(engine.pages, engine.slots);
    // Distinct sequences: sequence i starts i tokens later in the golden prompt.
    let mut placements = (0..count).map(|i| -> Result<GlmfPlacement> {
        let mut placement = allocator.admit(prefill + max_rows + 1)?;
        engine.prefill(&mut placement, &sequence[i..i + prefill], None)?;
        Ok(placement)
    }).collect::<Result<Vec<_>>>()?;
    let starts: Vec<usize> = placements.iter().map(|p| p.len).collect();
    println!("verify cost, {count} distinct sequence(s) after {prefill} tokens (speculative steps, median of 7):");
    for rows in 1..=max_rows {
        if count * rows > super::engine::DECODE_ROWS {
            break;
        }
        let tokens: Vec<u32> = (0..count).flat_map(|i| sequence[i + prefill..i + prefill + rows].iter().copied()).collect();
        let mut times = Vec::new();
        *engine.profile.borrow_mut() = [0.0; 3];
        for round in 0..9 {
            for (placement, &start) in placements.iter_mut().zip(&starts) {
                placement.len = start;
            }
            let mut step: Vec<(&mut GlmfPlacement, usize)> = placements.iter_mut().map(|p| (p, rows)).collect();
            let timer = Instant::now();
            engine.verify_spec(&mut step, &tokens)?;
            if round >= 2 {
                times.push(timer.elapsed().as_secs_f64());
            } else {
                *engine.profile.borrow_mut() = [0.0; 3];
            }
        }
        times.sort_by(f64::total_cmp);
        let phases = std::mem::take(&mut *engine.profile.borrow_mut());
        let n = times.len() as f64;
        println!("  {rows} rows/sequence ({} rows): {:.2} ms (min {:.2}); GPU until exchanges {:.2} ms, Spark exchanges \
            {:.2} ms, head {:.2} ms", count * rows, 1e3 * times[times.len() / 2], 1e3 * times[0], 1e3 * phases[0] / n,
            1e3 * phases[1] / n, 1e3 * phases[2] / n);
    }
    Ok(())
}
