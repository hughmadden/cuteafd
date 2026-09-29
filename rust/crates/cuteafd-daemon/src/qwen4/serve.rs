//! OpenAI-compatible API over the Qwen 3.8 Flash Next engine: continuous
//! batching with one prefill per admitted request and one decode-shaped step
//! for every active sequence, verifying copy-window drafts.
//!
//! GDN layers and the PLE conv advance their state in place (and the n-gram
//! history with them), so a verify whose drafts are rejected cannot just
//! shorten the sequence as the K/V records can: a sequence that drafts backs
//! its state slot up to a spare slot first, and after a partial acceptance
//! restores it, rewinds its n-gram history and replays the accepted rows
//! (their logits were already taken from the verify).
use super::engine::{Allocator, Qwen4Engine, Qwen4Placement, DECODE_ROWS};
use super::{embed_rows, open, Opened};
use anyhow::{Context, Result};
use cuteafd_api::native_v41::qwen::QwenEncoding;
use cuteafd_api::native_v41::{
    ConsoleHub, InferenceChunk, InferenceFinishReason, ModelEncoding, ModelProfile, NativeFailure, NativeLimits, NativeRequest,
    PromptUsage,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Most copy-window draft tokens verified per sequence and step.
const COPY_DRAFT: usize = 7;

#[derive(Debug, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub engine: super::EngineArgs,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    #[arg(long, default_value_t = 4096)]
    pub max_output: u32,
    /// Sequences decoding at once (each holds two state slots).
    #[arg(long, default_value_t = 4)]
    pub max_sequences: usize,
    /// Public model id; defaults to the snapshot's Hugging Face id.
    #[arg(long)]
    pub model_id: Option<String>,
    /// Decode one token per step (no copy-window drafts).
    #[arg(long)]
    pub no_copy_drafts: bool,
}

fn model_id(snapshot: &std::path::Path) -> Option<String> {
    snapshot.ancestors().find_map(|dir| {
        let name = dir.file_name()?.to_str()?.strip_prefix("models--")?;
        let (org, model) = name.split_once("--")?;
        Some(format!("{org}/{model}"))
    })
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    let snapshot: PathBuf = args.engine.snapshot.clone();
    let limits = NativeLimits::new(args.engine.max_context as u32, args.max_output)?;
    let encoding = Arc::new(QwenEncoding::from_snapshot(&snapshot)?);
    let eos = encoding.tokens().eos.clone();
    let profile = ModelProfile::new(
        args.model_id.clone().or_else(|| model_id(&snapshot)).context("model id")?,
        ModelEncoding::Qwen(encoding),
    );
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let mut engine_args = args.engine.clone();
    // Two state slots per sequence: its state and a verify backup.
    engine_args.slots = engine_args.slots.max(2 * args.max_sequences);
    let (worker_stats, max_sequences) = (stats.clone(), args.max_sequences);
    let draft = if args.no_copy_drafts { 0 } else { COPY_DRAFT };
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, draft, eos));
    ready_rx.await.context("engine failed before it was ready")??;
    let router = cuteafd_api::native_v41::router_for_model(queue, limits, stats, Duration::from_secs(25),
        ConsoleHub::disabled(), profile.clone());
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    tracing::info!(listen = %args.listen, model = %profile.id, "Qwen 3.8 Flash Next API is ready");
    tokio::select! {
        served = axum::serve(listener, router) => served?,
        finished = worker => finished??,
    }
    Ok(())
}

fn serve_loop(args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<()>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize,
    draft: usize, eos: Vec<u32>) -> Result<()> {
    let opened = match open(&args) {
        Ok(opened) => opened,
        Err(error) => {
            let _ = ready.send(Err(anyhow::anyhow!("{error:#}")));
            return Ok(());
        }
    };
    let mut ready = Some(ready);
    let result = opened.with_engine(&args, |engine| {
        anyhow::ensure!(engine.weights.layers.len() == engine.cfg.layers, "serve-qwen4 needs every layer");
        anyhow::ensure!(engine.experts().is_some_and(|e| !matches!(e, super::engine::Experts::SharedOnly)),
            "serve-qwen4 needs --peers (or --local-experts) for the routed experts");
        if let Some(ready) = ready.take() {
            let _ = ready.send(Ok(()));
        }
        schedule(engine, &opened, &args.snapshot, &mut receive, &stats, max_sequences.min(DECODE_ROWS), draft, eos)
    });
    if let Some(ready) = ready.take() {
        let _ = ready.send(result.as_ref().map(|_| ()).map_err(|e| anyhow::anyhow!("{e:#}")));
    }
    result
}

struct Active<'a> {
    job: NativeRequest,
    /// Prompt and generated tokens, for copy-window drafts.
    history: Vec<u32>,
    /// Current copy-draft length (halved after a fully rejected draft,
    /// doubled after a fully accepted one) and steps left before drafting
    /// resumes once it reached zero.
    draft_limit: usize,
    draft_pause: usize,
    constraint: Option<crate::v41_native_serve::constraints::State<'a>>,
    placement: Qwen4Placement,
    /// Spare state slot backing the GDN/PLE state up across a verify.
    backup: i32,
    capacity: usize,
    next: u32,
    decoder: cuteafd_loader::StreamingTokenDecoder,
    generated: usize,
    buffered: usize,
    started: Instant,
}

impl Active<'_> {
    fn select(&mut self, logits: &[f32]) -> Result<u32> {
        let position = self.placement.len as u64;
        let mask = match self.constraint.as_mut() {
            Some(state) => state.mask()?,
            None => None,
        };
        let token = self.job.sampling.select_token(logits, mask, position)
            .map_err(|e| anyhow::anyhow!("sampling: {e:?}"))? as u32;
        if let Some(state) = self.constraint.as_mut() {
            state.accept(token)?;
        }
        Ok(token)
    }

    fn send(&self, chunk: InferenceChunk) -> Result<()> {
        self.job.events.blocking_send(Ok(chunk)).map_err(|_| anyhow::anyhow!("client went away"))
    }

    /// Streams `token` (special tokens stay text for the Qwen parser); returns
    /// true when the request is finished.
    fn emit(&mut self, token: u32) -> Result<bool> {
        self.history.push(token);
        self.generated += 1;
        self.buffered += 1;
        let stop = self.job.stop_token_ids.contains(&token);
        if !stop {
            if let Some(content) = self.decoder.step(token)? {
                self.send(InferenceChunk::Text { content, content_tokens: self.buffered })?;
                self.buffered = 0;
            }
        }
        let finish = if stop {
            Some(InferenceFinishReason::Stop)
        } else if self.generated >= self.job.max_tokens || self.placement.len + 1 >= self.capacity {
            Some(InferenceFinishReason::Length)
        } else {
            None
        };
        let Some(finish) = finish else {
            self.next = token;
            return Ok(false);
        };
        let content = self.decoder.finish()?.unwrap_or_default();
        if !content.is_empty() || self.buffered > 0 {
            self.send(InferenceChunk::Text { content, content_tokens: self.buffered })?;
        }
        self.send(InferenceChunk::Finish { finish_reason: finish })?;
        Ok(true)
    }
}

/// Longest n-gram (from 8 down to 4 tokens) that ends the history and occurred
/// earlier; proposes up to `limit` tokens that followed its latest earlier
/// occurrence (a copy window). Exact: the verify step accepts only tokens the
/// model itself produces.
fn copy_drafts(history: &[u32], limit: usize) -> Vec<u32> {
    let len = history.len();
    for n in (4..=8).rev() {
        if len <= n {
            continue;
        }
        let tail = &history[len - n..];
        if let Some(start) = (0..len - n).rev().find(|&i| &history[i..i + n] == tail) {
            let from = start + n;
            return history[from..(from + limit).min(len)].to_vec();
        }
    }
    Vec::new()
}

#[allow(clippy::too_many_arguments)]
fn schedule(engine: &Qwen4Engine<'_>, opened: &Opened, snapshot: &std::path::Path,
    receive: &mut mpsc::Receiver<NativeRequest>, stats: &Mutex<serde_json::Value>, max_sequences: usize,
    draft: usize, eos: Vec<u32>) -> Result<()> {
    let mut allocator = Allocator::new(engine.pages, engine.slots, &engine.cfg);
    let mut grammars = crate::v41_native_serve::constraints::Compiler::with_vocab(
        &opened.library, snapshot.join("tokenizer.json"), engine.cfg.vocab_size, eos);
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(snapshot)?;
    let mut active: Vec<Active<'_>> = Vec::new();
    let (mut requests, mut generated_total, mut replays) = (0u64, 0u64, 0u64);
    // Verify steps (and replay steps) since the last completed request, and host seconds in verifies.
    let (mut steps, mut replay_steps, mut verify_s) = (0u64, 0u64, 0f64);
    let (hidden, vocab) = (engine.cfg.hidden, engine.cfg.vocab_size);
    loop {
        while active.len() < max_sequences {
            let job = if active.is_empty() {
                match receive.blocking_recv() {
                    Some(job) => job,
                    None => return Ok(()),
                }
            } else {
                match receive.try_recv() {
                    Ok(job) => job,
                    Err(_) => break,
                }
            };
            let reject = |job: &NativeRequest, message: String| {
                let _ = job.events.blocking_send(Err(NativeFailure::BadRequest(message)));
            };
            let constraint = match job.constraint.as_ref().map(|spec| grammars.matcher(spec)).transpose() {
                Ok(constraint) => constraint,
                Err(error) => {
                    reject(&job, format!("{error:#}"));
                    continue;
                }
            };
            let tokens = tokenizer.encode_text(&job.prompt, false)?.token_ids;
            if tokens.is_empty() || tokens.len() >= engine.max_context {
                reject(&job, format!("prompt of {} tokens is outside 1..{}", tokens.len(), engine.max_context));
                continue;
            }
            let capacity = (tokens.len() + job.max_tokens).min(engine.max_context);
            let (mut placement, backup) = match allocator.admit(capacity)
                .and_then(|p| Ok((p, allocator.spare_slot()?))) {
                Ok(admitted) => admitted,
                Err(error) => {
                    reject(&job, format!("{error:#}"));
                    continue;
                }
            };
            let admitted = (|| -> Result<Active<'_>> {
                let _ = job.events.blocking_send(Ok(InferenceChunk::Ready {
                    system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: 0 },
                }));
                let started = Instant::now();
                let mut logits = None;
                for chunk in tokens.chunks(engine.prefill_rows) {
                    let embed = embed_rows(&opened.checkpoint, chunk, hidden)?;
                    logits = engine.prefill(&mut placement, chunk, &embed)?;
                }
                let elapsed = started.elapsed().as_secs_f64();
                let phases = std::mem::take(&mut *engine.profile.borrow_mut());
                tracing::info!(tokens = tokens.len(), elapsed_ms = (1e3 * elapsed) as u64,
                    tok_s = tokens.len() as f64 / elapsed, gpu_wait_ms = (1e3 * phases[0]) as u64,
                    experts_ms = (1e3 * phases[1]) as u64, "prefill");
                let mut request = Active {
                    history: tokens.clone(),
                    draft_limit: draft,
                    draft_pause: 0,
                    decoder: cuteafd_loader::streaming_token_decoder(snapshot, false)?,
                    job, constraint, placement: placement.clone(), backup, capacity, next: 0, generated: 0,
                    buffered: 0, started: Instant::now(),
                };
                request.next = request.select(&logits.context("prefill produced no logits")?)?;
                Ok(request)
            })();
            match admitted {
                Ok(mut request) => {
                    let token = request.next;
                    match request.emit(token) {
                        Ok(false) => active.push(request),
                        Ok(true) | Err(_) => {
                            allocator.release_slot(request.backup);
                            allocator.release(request.placement);
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!("prefill failed: {error:#}");
                    allocator.release_slot(backup);
                    allocator.release(placement);
                }
            }
        }
        if active.is_empty() {
            continue;
        }
        // Each sequence verifies its next token plus a copy-window draft
        // (none when nothing repeats), within the decode programs' rows.
        let room = (DECODE_ROWS / active.len()).max(1) - 1;
        let sequences: Vec<Vec<u32>> = active.iter_mut().map(|a| {
            if a.draft_pause > 0 {
                a.draft_pause -= 1;
                if a.draft_pause == 0 {
                    a.draft_limit = 1;
                }
            }
            let limit = room.min(a.draft_limit).min(a.job.max_tokens - a.generated - 1)
                .min(a.capacity - a.placement.len - 1);
            // `emit` already appended `next` to the history.
            std::iter::once(a.next).chain(copy_drafts(&a.history, limit)).collect()
        }).collect();
        // Back up the GDN/PLE state of every sequence that drafts.
        for (request, rows) in active.iter().zip(&sequences) {
            if rows.len() > 1 {
                engine.copy_slot(request.placement.slot, request.backup)?;
            }
        }
        let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
        let histories: Vec<_> = active.iter().map(|a| a.placement.history.clone()).collect();
        let tokens: Vec<u32> = sequences.iter().flatten().copied().collect();
        let embed = embed_rows(&opened.checkpoint, &tokens, hidden)?;
        let mut rows: Vec<(&mut Qwen4Placement, &[u32])> = active.iter_mut().zip(&sequences)
            .map(|(a, s)| (&mut a.placement, s.as_slice())).collect();
        steps += 1;
        let timer = Instant::now();
        let step = engine.verify(&mut rows, &embed, None).and_then(|logits| logits.context("decode needs every layer"));
        verify_s += timer.elapsed().as_secs_f64();
        let logits = match step {
            Ok(logits) => logits,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                for request in active.drain(..) {
                    let _ = request.job.events.blocking_send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    allocator.release_slot(request.backup);
                    allocator.release(request.placement);
                }
                continue;
            }
        };
        let mut offset = 0;
        let finished: Vec<bool> = active.iter_mut().zip(&sequences).zip(&starts).map(|((request, rows), &start)| {
            let mut finished = false;
            for j in 0..rows.len() {
                // Rows 0..=j are committed; the token row j produces is next.
                request.placement.len = start + j + 1;
                match request.select(&logits[(offset + j) * vocab..][..vocab]).and_then(|t| Ok((t, request.emit(t)?))) {
                    Ok((token, done)) => {
                        finished = done;
                        if done || rows.get(j + 1) != Some(&token) {
                            break;
                        }
                    }
                    Err(_) => {
                        finished = true;
                        break;
                    }
                }
            }
            offset += rows.len();
            // Adapt the draft length to how much of it the model reproduced.
            let drafted = rows.len() - 1;
            let accepted = (request.placement.len - start).saturating_sub(1);
            if drafted > 0 && accepted == 0 {
                request.draft_limit /= 2;
                if request.draft_limit == 0 {
                    request.draft_pause = 8;
                }
            } else if drafted > 0 && accepted == drafted {
                request.draft_limit = (request.draft_limit * 2).clamp(1, draft);
            }
            finished
        }).collect();
        // Sequences that kept fewer rows than they verified: restore the
        // state and n-gram history and replay the kept rows (one step for all).
        let mut replay: Vec<(usize, usize)> = Vec::new();
        for (index, ((request, rows), &start)) in active.iter().zip(&sequences).zip(&starts).enumerate() {
            let kept = request.placement.len - start;
            if !finished[index] && kept < rows.len() {
                engine.copy_slot(request.backup, request.placement.slot)?;
                replay.push((index, kept));
            }
        }
        if !replay.is_empty() {
            replays += 1;
            replay_steps += 1;
            let tokens: Vec<u32> = replay.iter().flat_map(|&(i, kept)| sequences[i][..kept].iter().copied()).collect();
            let embed = embed_rows(&opened.checkpoint, &tokens, hidden)?;
            let mut placements: Vec<(Qwen4Placement, &[u32])> = replay.iter().map(|&(i, kept)| {
                let mut placement = active[i].placement.clone();
                placement.len = starts[i];
                placement.history = histories[i].clone();
                (placement, &sequences[i][..kept])
            }).collect();
            let mut rows: Vec<(&mut Qwen4Placement, &[u32])> =
                placements.iter_mut().map(|(p, t)| (p, *t)).collect();
            engine.verify(&mut rows, &embed, None)?;
            for ((i, _), (placement, _)) in replay.iter().zip(&placements) {
                active[*i].placement.history = placement.history.clone();
            }
        }
        for index in (0..active.len()).rev() {
            if !finished[index] {
                continue;
            }
            let request = active.remove(index);
            requests += 1;
            generated_total += request.generated as u64;
            let seconds = request.started.elapsed().as_secs_f64();
            let phases = std::mem::take(&mut *engine.profile.borrow_mut());
            tracing::info!(tokens = request.generated, seconds, tok_s = request.generated as f64 / seconds,
                active = active.len(), steps, replay_steps, verify_s, gpu_wait_s = phases[0], experts_s = phases[1],
                replays, "request complete");
            (steps, replay_steps, verify_s) = (0, 0, 0.0);
            allocator.release_slot(request.backup);
            allocator.release(request.placement);
        }
        if let Ok(mut stats) = stats.lock() {
            *stats = serde_json::json!({"requests": requests, "generated_tokens": generated_total,
                "active": active.len(), "replays": replays});
        }
    }
}
