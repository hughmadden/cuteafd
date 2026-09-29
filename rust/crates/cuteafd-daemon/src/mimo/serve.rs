//! OpenAI-compatible API over the MiMo V2 engine: continuous batching with
//! one prefill per admitted request and one decode-shaped step for every
//! active sequence, verifying copy-window drafts.
//!
//! MiMo keeps no recurrent state: full layers write paged records and SWA
//! layers a 256-slot ring that outlives the 128-token window by more than a
//! step's rows, so a verify whose drafts are rejected just sets the
//! sequence length back.
use super::dflash::{ContextRow, DraftSeq};
use super::engine::{Allocator, MimoEngine, MimoPlacement, DECODE_ROWS};
use crate::glm::dflash_policy::{self, DraftHistory, Group, StepCost};
use super::{open, Opened};
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
    /// Sequences decoding at once (each holds an SWA ring).
    #[arg(long, default_value_t = 4)]
    pub max_sequences: usize,
    /// Public model id; defaults to the snapshot's Hugging Face id.
    #[arg(long)]
    pub model_id: Option<String>,
    /// Decode one token per step (no copy-window drafts).
    #[arg(long)]
    pub no_copy_drafts: bool,
    /// With --draft: verify this many DFlash drafts per sequence every step
    /// instead of the adaptive plan.
    #[arg(long)]
    pub draft_fixed: Option<usize>,
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    let snapshot: PathBuf = args.engine.snapshot.clone();
    let limits = NativeLimits::new(args.engine.max_context as u32, args.max_output)?;
    // MiMo's template and tool calls follow Qwen3-Coder's XML (`<tool_call>
    // <function=NAME><parameter=KEY>VALUE</parameter>`), its reasoning `<think>`.
    let encoding = QwenEncoding::from_snapshot(&snapshot)?;
    let profile = ModelProfile::new(
        args.model_id.clone().or_else(|| crate::glmf::serve::model_id(&snapshot)).context("model id")?,
        ModelEncoding::Qwen(Arc::new(encoding)),
    );
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let mut engine_args = args.engine.clone();
    engine_args.rings = engine_args.rings.max(args.max_sequences);
    let (worker_stats, max_sequences) = (stats.clone(), args.max_sequences);
    engine_args.draft_sequences = engine_args.draft_sequences.max(args.max_sequences);
    let draft = Policy { copy: if args.no_copy_drafts { 0 } else { COPY_DRAFT }, fixed: args.draft_fixed };
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, draft));
    ready_rx.await.context("engine failed before it was ready")??;
    let router = cuteafd_api::native_v41::router_for_model(queue, limits, stats, Duration::from_secs(25),
        ConsoleHub::disabled(), profile.clone());
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    tracing::info!(listen = %args.listen, model = %profile.id, "MiMo V2 API is ready");
    tokio::select! {
        served = axum::serve(listener, router) => served?,
        finished = worker => finished??,
    }
    Ok(())
}

/// Draft settings: copy-window draft length, fixed DFlash draft count.
#[derive(Debug, Clone, Copy)]
struct Policy {
    copy: usize,
    fixed: Option<usize>,
}

/// Verify-step ms by rows for MiMo V2.6 Pro, RTX PRO 6000 + six Sparks (TP6
/// MXFP4): an estimate from the coordinator's measured decode (21.4 ms at one
/// row with FP8 qkv, +0.6 ms per row) and GB10 weight bandwidth for the
/// distinct experts a step reads; serving rescales it by what it observes.
const PRO_TP6_STEP_MS: [(usize, f64); 8] = [(1, 50.0), (2, 60.0), (4, 75.0), (8, 100.0), (16, 140.0), (24, 170.0),
    (32, 200.0), (64, 300.0)];

fn serve_loop(args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<()>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize,
    draft: Policy) -> Result<()> {
    let opened = match open(&args) {
        Ok(opened) => opened,
        Err(error) => {
            let _ = ready.send(Err(anyhow::anyhow!("{error:#}")));
            return Ok(());
        }
    };
    let mut ready = Some(ready);
    let result = opened.with_engine(&args, |engine| {
        anyhow::ensure!(engine.weights.layers.len() == engine.cfg.layers, "serve-mimo needs every layer");
        anyhow::ensure!(engine.has_experts(), "serve-mimo needs --peers (or --local-experts) for the routed experts");
        if let Some(ready) = ready.take() {
            let _ = ready.send(Ok(()));
        }
        schedule(engine, &opened, &args.snapshot, &mut receive, &stats, max_sequences.min(DECODE_ROWS), draft)
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
    /// The sequence's DFlash ring slot (None: copy-window drafts only).
    slot: Option<usize>,
    /// Recent DFlash (proposed, accepted) outcomes for the adaptive plan.
    drafts: DraftHistory,
    /// Steps, DFlash drafts verified / accepted, copy drafts verified / accepted.
    counts: [usize; 5],
    /// Current copy-draft length (halved after a fully rejected draft,
    /// doubled after a fully accepted one) and steps left before drafting
    /// resumes once it reached zero.
    draft_limit: usize,
    draft_pause: usize,
    constraint: Option<crate::v41_native_serve::constraints::State<'a>>,
    placement: MimoPlacement,
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

    /// Streams `token` (special tokens stay text for the output parser);
    /// returns true when the request is finished.
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

fn schedule(engine: &MimoEngine<'_>, opened: &Opened, snapshot: &std::path::Path,
    receive: &mut mpsc::Receiver<NativeRequest>, stats: &Mutex<serde_json::Value>, max_sequences: usize,
    policy: Policy) -> Result<()> {
    let draft = policy.copy;
    let drafter = engine.drafter.as_ref();
    let mut free_slots: Vec<usize> = drafter.map_or(Vec::new(), |d| (0..d.slots).rev().collect());
    let mut cost = StepCost::new(&PRO_TP6_STEP_MS, DECODE_ROWS);
    let mut skip = crate::glm::dflash_policy::DraftSkip::default();
    let mut allocator = Allocator::new(engine.pages, engine.rings);
    let mut grammars = crate::v41_native_serve::constraints::Compiler::with_vocab(
        &opened.library, snapshot.join("tokenizer.json"), engine.cfg.vocab_size, QwenEncoding::from_snapshot(snapshot)?.tokens().eos.clone());
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(snapshot)?;
    let embeddings = super::Embeddings::open(&opened.checkpoint, engine.cfg.hidden)?;
    let mut active: Vec<Active<'_>> = Vec::new();
    let (mut requests, mut generated_total) = (0u64, 0u64);
    // Verify steps since the last completed request, and host seconds in
    // them (engine) and in token selection + streaming.
    let mut steps = 0u64;
    let (mut verify_s, mut draft_s, mut emit_s, mut embed_s) = (0f64, 0f64, 0f64, 0f64);
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
            let slot = free_slots.pop();
            let mut placement = match allocator.admit(capacity) {
                Ok(placement) => placement,
                Err(error) => {
                    free_slots.extend(slot);
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
                    let embed = embeddings.rows(chunk)?;
                    let start = placement.len;
                    logits = engine.prefill(&mut placement, &embed, None)?;
                    // The chunk's tapped tail becomes the drafter's context.
                    if let (Some(drafter), Some(slot)) = (drafter, slot) {
                        let n = chunk.len().min(super::dflash::TAP_ROWS);
                        drafter.update(&(0..n).map(|r| ContextRow { tap_row: r, slot,
                            position: start + chunk.len() - n + r }).collect::<Vec<_>>())?;
                    }
                }
                let elapsed = started.elapsed().as_secs_f64();
                let phases = std::mem::take(&mut *engine.profile.borrow_mut());
                tracing::info!(tokens = tokens.len(), elapsed_ms = (1e3 * elapsed) as u64,
                    tok_s = tokens.len() as f64 / elapsed, gpu_wait_ms = (1e3 * phases[0]) as u64,
                    experts_ms = (1e3 * phases[1]) as u64, "prefill");
                let mut request = Active {
                    slot,
                    drafts: DraftHistory::default(),
                    counts: [0; 5],
                    history: tokens.clone(),
                    draft_limit: draft,
                    draft_pause: 0,
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
                            allocator.release(request.placement)
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
        // DFlash drafts after every next token (sequences with a ring slot),
        // then the adaptive plan's counts.
        let drafted: Vec<Option<super::dflash::Draft>> = match drafter {
            Some(drafter) if skip.drafts() && active.iter().any(|a| a.slot.is_some()) => {
                let seqs: Vec<(usize, DraftSeq)> = active.iter().enumerate()
                    .filter_map(|(i, a)| a.slot.map(|slot| (i, DraftSeq { slot, anchor: a.next, position: a.placement.len })))
                    .collect();
                let anchors: Vec<u32> = seqs.iter().map(|(_, s)| s.anchor).collect();
                let timer = Instant::now();
                let drafts = embeddings.rows(&anchors).and_then(|rows| drafter.draft(
                    &seqs.iter().map(|(_, s)| *s).collect::<Vec<_>>(), &rows, engine.weights.head.buffer.ptr));
                let ms = timer.elapsed().as_secs_f64() * 1e3;
                cost.observe_draft(ms);
                draft_s += ms / 1e3;
                let mut out = vec![None; active.len()];
                match drafts {
                    Ok(drafts) => {
                        for ((i, _), draft) in seqs.into_iter().zip(drafts) {
                            out[i] = Some(draft);
                        }
                    }
                    // Drafts only speed decoding up; the step verifies the next tokens alone.
                    Err(error) => tracing::warn!("DFlash draft failed: {error:#}"),
                }
                out
            }
            _ => vec![None; active.len()],
        };
        let planned = plan_drafts(&active, &drafted, &limits, policy.fixed, &cost);
        skip.after(drafted.iter().any(Option::is_some) && policy.fixed.is_none(), planned.iter().all(|&n| n == 0));
        // Each sequence verifies its next token, then its DFlash drafts, or a
        // copy-window draft when it agrees with them and runs longer.
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
            let copy = crate::glmf::serve::copy_drafts(&a.history, limits[i].min(a.draft_limit));
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
        let tokens: Vec<u32> = sequences.iter().flatten().copied().collect();
        let timer = Instant::now();
        let embed = embeddings.rows(&tokens)?;
        embed_s += timer.elapsed().as_secs_f64();
        let mut rows: Vec<(&mut MimoPlacement, usize)> = active.iter_mut().zip(&sequences)
            .map(|(a, s)| (&mut a.placement, s.len())).collect();
        steps += 1;
        let timer = Instant::now();
        let step = engine.verify(&mut rows, &embed, None).and_then(|logits| logits.context("decode needs every layer"));
        let step_s = timer.elapsed().as_secs_f64();
        verify_s += step_s;
        cost.observe(tokens.len(), tokens.len(), step_s * 1e3);
        let timer = Instant::now();
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
        let mut offset = 0;
        let mut context = Vec::new();
        let finished: Vec<bool> = active.iter_mut().zip(&sequences).zip(&starts).enumerate()
            .map(|(i, ((request, rows), &start))| {
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
            if let Some(slot) = request.slot.filter(|_| !finished) {
                context.extend((0..committed).map(|r| ContextRow { tap_row: offset + r, slot, position: start + r }));
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
            if used_copy[i] || request.slot.is_none() {
                if drafted > 0 && accepted == 0 {
                    request.draft_limit /= 2;
                    if request.draft_limit == 0 {
                        request.draft_pause = 8;
                    }
                } else if drafted > 0 && accepted == drafted {
                    request.draft_limit = (request.draft_limit * 2).clamp(1, draft.max(1));
                }
            }
            finished
        }).collect();
        if let Some(drafter) = drafter {
            drafter.update(&context)?;
        }
        emit_s += timer.elapsed().as_secs_f64();
        for index in (0..active.len()).rev() {
            if !finished[index] {
                continue;
            }
            let request = active.remove(index);
            requests += 1;
            generated_total += request.generated as u64;
            let seconds = request.started.elapsed().as_secs_f64();
            let phases = std::mem::take(&mut *engine.profile.borrow_mut());
            let [cycles, dflash, dflash_ok, copy, copy_ok] = request.counts;
            tracing::info!(tokens = request.generated, seconds, tok_s = request.generated as f64 / seconds,
                active = active.len(), steps, verify_s, draft_s, emit_s, embed_s, gpu_wait_s = phases[0],
                experts_s = phases[1], cycles, dflash, dflash_ok, copy, copy_ok, "request complete");
            (steps, verify_s, draft_s, emit_s, embed_s) = (0, 0.0, 0.0, 0.0, 0.0);
            free_slots.extend(request.slot);
            allocator.release(request.placement);
        }
        if let Ok(mut stats) = stats.lock() {
            *stats = serde_json::json!({"requests": requests, "generated_tokens": generated_total,
                "active": active.len()});
        }
    }
}

/// DFlash draft counts: `fixed` (within each limit), or the adaptive plan
/// (glmrt v9's schedule, `dflash_policy::plan`) over the drafting sequences
/// with each one's history of conditional acceptance (the GLM selector
/// calibration does not apply to this drafter), priced with the sequences
/// that do not draft.
fn plan_drafts(active: &[Active<'_>], drafted: &[Option<super::dflash::Draft>], limits: &[usize],
    fixed: Option<usize>, cost: &StepCost) -> Vec<usize> {
    let indices: Vec<usize> = (0..active.len()).filter(|&i| drafted[i].is_some()).collect();
    let mut counts = vec![0; active.len()];
    let width = |i: usize| drafted[i].as_ref().map_or(0, |d| d.tokens.len());
    if let Some(fixed) = fixed {
        for &i in &indices {
            counts[i] = fixed.min(limits[i]).min(width(i));
        }
        return counts;
    }
    if indices.is_empty() {
        return counts;
    }
    let groups: Vec<Group<'_>> = indices.iter().map(|&i| Group {
        history: &active[i].drafts,
        confidence: active[i].drafts.conditional(width(i)),
        room: limits[i],
        members: 1,
    }).collect();
    let others = active.len() - indices.len();
    for (&i, n) in indices.iter().zip(dflash_policy::plan(&groups, (others, others), cost)) {
        counts[i] = n;
    }
    counts
}
