//! OpenAI-compatible API over the GLM 5.3 Flash engine: continuous batching
//! with one prefill per admitted request and one decode-shaped step for
//! every active sequence. With a DFlash2 drafter (--draft), every step first
//! drafts after each sequence's next token on the GPU and verifies as many
//! drafts as the adaptive policy (glm/dflash_policy.rs, priced by the
//! measured Spark TP2 step cost) finds worthwhile; copy-window drafts extend
//! a DFlash2 draft they agree with (or stand alone without a drafter).
//!
//! KDA layers advance recurrent state that a rejected draft cannot simply
//! drop, so a step that verifies drafts runs speculatively
//! (`GlmfEngine::verify_spec`: the KDA state stays, each row's replay inputs
//! are recorded) and then commits every sequence's kept rows
//! (`GlmfEngine::commit`), which leaves the state serial steps would have.
//! MLA records past the kept length are rewritten by later steps.
use super::engine::{Allocator, GlmfEngine, GlmfPlacement, DECODE_ROWS};
use crate::glm::dflash::{ContextRow, Draft, DraftSeq, TAP_ROWS};
use crate::glm::dflash_policy::{self, DraftHistory, Shape};
use super::{embed_rows, open, Opened};
use anyhow::{Context, Result};
use cuteafd_api::native_v41::glm::GlmEncoding;
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
    /// Sequences decoding at once (each holds two KDA state slots).
    #[arg(long, default_value_t = 4)]
    pub max_sequences: usize,
    /// Public model id; defaults to the snapshot's Hugging Face id.
    #[arg(long)]
    pub model_id: Option<String>,
    /// Decode one token per step (no copy-window drafts).
    #[arg(long)]
    pub no_copy_drafts: bool,
    /// With --draft: verify exactly this many DFlash2 drafts per step
    /// (within the rows) instead of the adaptive policy.
    #[arg(long)]
    pub draft_fixed: Option<usize>,
}

/// Speculation settings: copy-window draft cap (0 disables) and a fixed
/// DFlash2 draft count replacing the adaptive policy.
#[derive(Debug, Clone, Copy)]
struct Policy {
    copy: usize,
    fixed: Option<usize>,
}

pub(crate) fn model_id(snapshot: &std::path::Path) -> Option<String> {
    snapshot.ancestors().find_map(|dir| {
        let name = dir.file_name()?.to_str()?.strip_prefix("models--")?;
        let (org, model) = name.split_once("--")?;
        Some(format!("{org}/{model}"))
    })
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    let snapshot: PathBuf = args.engine.snapshot.clone();
    let limits = NativeLimits::new(args.engine.max_context as u32, args.max_output)?;
    let encoding = GlmEncoding::from_snapshot(&snapshot)?;
    let profile = ModelProfile::new(
        args.model_id.clone().or_else(|| model_id(&snapshot)).context("model id")?,
        ModelEncoding::Glm(Arc::new(encoding)),
    );
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let mut engine_args = args.engine.clone();
    engine_args.slots = engine_args.slots.max(args.max_sequences);
    engine_args.draft_sequences = engine_args.draft_sequences.max(args.max_sequences);
    let (worker_stats, max_sequences) = (stats.clone(), args.max_sequences);
    let policy = Policy { copy: if args.no_copy_drafts { 0 } else { COPY_DRAFT }, fixed: args.draft_fixed };
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, policy));
    ready_rx.await.context("engine failed before it was ready")??;
    let router = cuteafd_api::native_v41::router_for_model(queue, limits, stats, Duration::from_secs(25),
        ConsoleHub::disabled(), profile.clone());
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    tracing::info!(listen = %args.listen, model = %profile.id, "GLM 5.3 Flash API is ready");
    tokio::select! {
        served = axum::serve(listener, router) => served?,
        finished = worker => finished??,
    }
    Ok(())
}

fn serve_loop(args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<()>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize,
    policy: Policy) -> Result<()> {
    let opened = match open(&args) {
        Ok(opened) => opened,
        Err(error) => {
            let _ = ready.send(Err(anyhow::anyhow!("{error:#}")));
            return Ok(());
        }
    };
    let mut ready = Some(ready);
    let result = opened.with_engine(&args, |engine| {
        anyhow::ensure!(engine.weights.layers.len() == engine.cfg.layers, "serve-glmf needs every layer");
        anyhow::ensure!(engine.experts().is_some(), "serve-glmf needs --peers (or --local-experts) for the routed experts");
        if let Some(ready) = ready.take() {
            let _ = ready.send(Ok(()));
        }
        schedule(engine, &opened, &args.snapshot, &mut receive, &stats, max_sequences.min(DECODE_ROWS), policy)
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
    /// DFlash2 ring slot and draft outcomes (None without a drafter slot).
    slot: Option<usize>,
    drafts: DraftHistory,
    /// Hash of `history` (identical sequences share it).
    digest: u64,
    /// Steps, DFlash2 drafts verified and accepted, copy drafts verified and accepted.
    counts: [usize; 5],
    constraint: Option<crate::v41_native_serve::constraints::State<'a>>,
    placement: GlmfPlacement,
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

    /// Streams `token` (special tokens stay text for the GLM parser); returns
    /// true when the request is finished.
    fn emit(&mut self, token: u32) -> Result<bool> {
        self.history.push(token);
        self.digest = digest(self.digest, token);
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

const DIGEST_SEED: u64 = 0xcbf2_9ce4_8422_2325;

fn digest(state: u64, token: u32) -> u64 {
    (state ^ u64::from(token)).wrapping_mul(0x0100_0000_01b3)
}

/// Longest n-gram (from 8 down to 4 tokens) that ends the history and occurred
/// earlier; proposes up to `limit` tokens that followed its latest earlier
/// occurrence (a copy window). Exact: the verify step accepts only tokens the
/// model itself produces.
pub(crate) fn copy_drafts(history: &[u32], limit: usize) -> Vec<u32> {
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

fn schedule(engine: &GlmfEngine<'_>, opened: &Opened, snapshot: &std::path::Path,
    receive: &mut mpsc::Receiver<NativeRequest>, stats: &Mutex<serde_json::Value>, max_sequences: usize,
    policy: Policy) -> Result<()> {
    let mut allocator = Allocator::new(engine.pages, engine.slots);
    let mut grammars = crate::v41_native_serve::constraints::Compiler::with_vocab(
        &opened.library, snapshot.join("tokenizer.json"), engine.cfg.vocab_size, engine.cfg.eos.clone());
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(snapshot)?;
    let drafter = engine.drafter.as_ref();
    let mut free_slots: Vec<usize> = drafter.map_or(Vec::new(), |d| (0..d.slots).rev().collect());
    let mut cost = dflash_policy::step_cost(&GLMF_TP2_STEP_MS, DECODE_ROWS);
    let mut skip = dflash_policy::DraftSkip::default();
    let mut active: Vec<Active<'_>> = Vec::new();
    let (mut requests, mut generated_total) = (0u64, 0u64);
    // Per request window: verify steps, and host seconds drafting, verifying
    // (engine step + commit) and selecting/streaming tokens.
    let (mut steps, mut draft_s, mut verify_s, mut emit_s) = (0u64, 0f64, 0f64, 0f64);
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
            let mut placement = match allocator.admit(capacity) {
                Ok(placement) => placement,
                Err(error) => {
                    reject(&job, format!("{error:#}"));
                    continue;
                }
            };
            let slot = free_slots.pop();
            let admitted = (|| -> Result<Active<'_>> {
                let _ = job.events.blocking_send(Ok(InferenceChunk::Ready {
                    system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: 0 },
                }));
                let started = Instant::now();
                let mut logits = None;
                for chunk in tokens.chunks(engine.prefill_capacity()) {
                    let embed = embed_rows(&opened.checkpoint, chunk, hidden)?;
                    let start = placement.len;
                    logits = engine.prefill(&mut placement, &embed, None)?;
                    // The chunk's tapped tail becomes drafter context before the next step.
                    if let (Some(drafter), Some(slot)) = (drafter, slot) {
                        let n = chunk.len().min(TAP_ROWS);
                        let first = start + chunk.len() - n;
                        drafter.update(&(0..n).map(|r| ContextRow { tap_row: r, slot, position: first + r })
                            .collect::<Vec<_>>())?;
                    }
                }
                let phases = std::mem::take(&mut *engine.profile.borrow_mut());
                tracing::info!(tokens = tokens.len(), elapsed_ms = started.elapsed().as_millis() as u64,
                    gpu_wait_ms = (1e3 * phases[0]) as u64, experts_ms = (1e3 * phases[1]) as u64,
                    head_ms = (1e3 * phases[2]) as u64, "prefill");
                let mut request = Active {
                    history: tokens.clone(),
                    digest: tokens.iter().fold(DIGEST_SEED, |d, &t| digest(d, t)),
                    draft_limit: policy.copy,
                    draft_pause: 0,
                    slot,
                    drafts: DraftHistory::default(),
                    counts: [0; 5],
                    decoder: cuteafd_loader::streaming_token_decoder(snapshot, false)?,
                    job, constraint, placement: placement.clone(), capacity, next: 0, generated: 0,
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
                            free_slots.extend(request.slot);
                            allocator.release(request.placement);
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!("prefill failed: {error:#}");
                    free_slots.extend(slot);
                    allocator.release(placement);
                }
            }
        }
        if active.is_empty() {
            continue;
        }
        // Rows each sequence may add after its next token.
        let room = (DECODE_ROWS / active.len()).max(1) - 1;
        let limits: Vec<usize> = active.iter().map(|a| room.min(a.job.max_tokens - a.generated - 1)
            .min(a.capacity - a.placement.len - 1)).collect();
        // DFlash2 drafts after every next token, then the policy's counts.
        let timer = Instant::now();
        let drafted: Vec<Option<Draft>> = match drafter {
            Some(drafter) if skip.drafts() && active.iter().any(|a| a.slot.is_some()) => {
                let seqs: Vec<(usize, DraftSeq)> = active.iter().enumerate()
                    .filter_map(|(i, a)| a.slot.map(|slot| (i, DraftSeq { slot, anchor: a.next, position: a.placement.len })))
                    .collect();
                let anchors: Vec<u32> = seqs.iter().map(|(_, s)| s.anchor).collect();
                let drafts = embed_rows(&opened.checkpoint, &anchors, hidden).and_then(|rows| drafter.draft(
                    &seqs.iter().map(|(_, s)| *s).collect::<Vec<_>>(), &rows, engine.weights.head.buffer.ptr));
                cost.observe_draft(timer.elapsed().as_secs_f64() * 1e3);
                let mut out = vec![None; active.len()];
                match drafts {
                    Ok(drafts) => {
                        for ((i, _), draft) in seqs.into_iter().zip(drafts) {
                            out[i] = Some(draft);
                        }
                    }
                    // Drafts only speed decoding up; the step verifies the next tokens alone.
                    Err(error) => tracing::warn!("DFlash2 draft failed: {error:#}"),
                }
                out
            }
            _ => vec![None; active.len()],
        };
        draft_s += timer.elapsed().as_secs_f64();
        // Identical sequences (same tokens at the same position) route alike
        // and draft alike: the policy prices and plans them as one group.
        let key = |a: &Active<'_>| (a.placement.len, a.digest);
        let inputs: Vec<dflash_policy::PlanInput<'_>> = active.iter().enumerate().map(|(i, a)| dflash_policy::PlanInput {
            key: key(a), history: &a.drafts, features: drafted[i].as_ref().map(|d| d.features.as_slice()),
            limit: limits[i],
        }).collect();
        let planned = dflash_policy::plan_counts(&inputs, policy.fixed, &cost);
        drop(inputs);
        skip.after(drafted.iter().any(Option::is_some) && policy.fixed.is_none(), planned.iter().all(|&n| n == 0));
        // Each sequence verifies its next token, then its DFlash2 drafts, or
        // a copy-window draft when it agrees with them and runs longer.
        let mut used_copy = vec![false; active.len()];
        let sequences: Vec<Vec<u32>> = active.iter_mut().enumerate().map(|(i, a)| {
            if a.draft_pause > 0 {
                a.draft_pause -= 1;
                if a.draft_pause == 0 {
                    a.draft_limit = 1;
                }
            }
            let dflash: &[u32] = drafted[i].as_ref().map_or(&[], |d| &d.tokens[..planned[i]]);
            let full: &[u32] = drafted[i].as_ref().map_or(&[], |d| &d.tokens);
            // `emit` already appended `next` to the history.
            let copy = copy_drafts(&a.history, limits[i].min(a.draft_limit));
            let agrees = copy.iter().zip(full).take_while(|(c, d)| c == d).count() >= dflash.len();
            let draft = if copy.len() > dflash.len() && agrees {
                used_copy[i] = true;
                copy
            } else {
                dflash.to_vec()
            };
            std::iter::once(a.next).chain(draft).collect()
        }).collect();
        let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
        let distinct_rows: usize = active.iter().zip(&sequences).map(|(a, rows)| (key(a), rows))
            .collect::<std::collections::HashSet<_>>().iter().map(|(_, rows)| rows.len()).sum();
        let tokens: Vec<u32> = sequences.iter().flatten().copied().collect();
        // A step with drafts runs speculatively and commits what it keeps.
        let spec = sequences.iter().any(|rows| rows.len() > 1);
        let embed = embed_rows(&opened.checkpoint, &tokens, hidden)?;
        let mut rows: Vec<(&mut GlmfPlacement, usize)> = active.iter_mut().zip(&sequences)
            .map(|(a, s)| (&mut a.placement, s.len())).collect();
        steps += 1;
        let timer = Instant::now();
        let step = if spec { engine.verify_spec(&mut rows, &embed) } else { engine.verify(&mut rows, &embed, None) }
            .and_then(|logits| logits.context("decode needs every layer"));
        let step_ms = timer.elapsed().as_secs_f64() * 1e3;
        verify_s += step_ms / 1e3;
        let logits = match step {
            Ok(logits) => logits,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                for request in active.drain(..) {
                    let _ = request.job.events.blocking_send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    free_slots.extend(request.slot);
                    allocator.release(request.placement);
                }
                continue;
            }
        };
        cost.observe_verify(Shape { rows: tokens.len(), distinct: distinct_rows, sequences: sequences.len() }, step_ms);
        let timer = Instant::now();
        let mut offset = 0;
        let mut context = Vec::new();
        let mut commits = Vec::new();
        let finished: Vec<bool> = active.iter_mut().zip(&sequences).zip(starts).enumerate()
            .map(|(i, ((request, rows), start))| {
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
            let committed = request.placement.len - start;
            if !finished {
                commits.push((request.placement.slot, offset, committed));
                if let Some(slot) = request.slot {
                    context.extend((0..committed).map(|r| ContextRow { tap_row: offset + r, slot, position: start + r }));
                }
            }
            offset += rows.len();
            let (drafted, accepted) = (rows.len() - 1, committed - 1);
            request.counts[0] += 1;
            if used_copy[i] {
                request.counts[3] += drafted;
                request.counts[4] += accepted;
            } else {
                request.counts[1] += drafted;
                request.counts[2] += accepted;
            }
            if planned[i] > 0 {
                request.drafts.observe(planned[i], accepted);
            }
            // Adapt the copy-draft length to how much of it the model reproduced.
            if used_copy[i] || drafter.is_none() {
                if drafted > 0 && accepted == 0 {
                    request.draft_limit /= 2;
                    if request.draft_limit == 0 {
                        request.draft_pause = 8;
                    }
                } else if drafted > 0 && accepted == drafted {
                    request.draft_limit = (request.draft_limit * 2).clamp(1, policy.copy.max(1));
                }
            }
            finished
        }).collect();
        emit_s += timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        if spec {
            engine.commit(&commits)?;
        }
        if let Some(drafter) = drafter {
            drafter.update(&context)?;
        }
        verify_s += timer.elapsed().as_secs_f64();
        for index in (0..active.len()).rev() {
            if !finished[index] {
                continue;
            }
            let request = active.remove(index);
            requests += 1;
            generated_total += request.generated as u64;
            let seconds = request.started.elapsed().as_secs_f64();
            let phases = std::mem::take(&mut *engine.profile.borrow_mut());
            let [steps_seen, dflash, dflash_ok, copy, copy_ok] = request.counts;
            tracing::info!(tokens = request.generated, seconds, tok_s = request.generated as f64 / seconds,
                active = active.len(), steps = steps_seen, all_steps = steps, dflash, dflash_ok, copy, copy_ok,
                draft_s, verify_s, emit_s, gpu_wait_s = phases[0], experts_s = phases[1], head_s = phases[2],
                "request complete");
            (steps, draft_s, verify_s, emit_s) = (0, 0.0, 0.0, 0.0);
            free_slots.extend(request.slot);
            allocator.release(request.placement);
        }
        if let Ok(mut stats) = stats.lock() {
            *stats = serde_json::json!({"requests": requests, "generated_tokens": generated_total,
                "active": active.len()});
        }
    }
}

/// GLM 5.3 Flash, 1 RTX PRO 6000 (325 W) + Spark TP2 (rhea, moa), recommended
/// FP8 decode config: speculative verify step ms by rows of one sequence
/// (glmf-golden --bench-verify 16 after 512 tokens, median of 7); past 16
/// rows extrapolated at the 12-16 slope (serving refits intercept and slope).
const GLMF_TP2_STEP_MS: [(usize, f64); 15] = [(1, 19.1), (2, 26.0), (3, 30.2), (4, 35.2), (5, 40.1), (6, 43.5),
    (7, 48.7), (8, 53.8), (10, 60.7), (12, 67.4), (16, 80.2), (24, 106.0), (32, 132.0), (48, 183.0), (64, 234.0)];
