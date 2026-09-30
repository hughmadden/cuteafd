//! OpenAI-compatible API over the Qwen 3.8 Flash Next engine: continuous
//! batching with one prefill per admitted request and one decode-shaped step
//! for every active sequence, verifying drafts: the native MTP layer's
//! (`--mtp N`, drafts per sequence planned by `mtp_policy`) or copy-window
//! drafts from the sequence's own history.
//!
//! Verify-by-replay: a step that verifies drafts runs speculatively (GDN and
//! PLE record each row's replay inputs and keep their state); the accepted
//! rows are then committed in one launch, the placement and n-gram history
//! rewound (K/V records past them are rewritten when those positions come
//! again). With MTP the kept rows' pre-mixer streams wait in the MTP stash
//! for the next cycle's draft step (see `speculate`).
use super::engine::{Allocator, Qwen4Engine, Qwen4Placement, DECODE_ROWS};
use super::mtp_policy;
use super::speculate::{self, DraftSeq, DraftTiming, MtpSeq, Verified};
use super::{embed_rows, open, Opened};
use crate::shared::draft_policy::{Calibration, DraftHistory, Shape};
use crate::shared::prefill_share::{add_phases, isolated_phases, Chunk, DecodeShareArgs};
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
    /// Decode one token per step (no copy-window drafts; ignored with --mtp).
    #[arg(long)]
    pub no_copy_drafts: bool,
    /// With --mtp: verify exactly --mtp drafts per sequence where room allows
    /// instead of the adaptive plan.
    #[arg(long)]
    pub mtp_fixed: bool,
    #[command(flatten)]
    pub decode_share: DecodeShareArgs,
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
    engine_args.slots = engine_args.slots.max(args.max_sequences);
    let (worker_stats, max_sequences, decode_share) = (stats.clone(), args.max_sequences, args.decode_share);
    let drafts = if engine_args.mtp > 0 {
        Drafts::Mtp { depth: engine_args.mtp.min(DECODE_ROWS - 1), fixed: args.mtp_fixed }
    } else if args.no_copy_drafts {
        Drafts::None
    } else {
        Drafts::Copy
    };
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, drafts, eos, decode_share));
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

/// Where a step's drafts come from.
#[derive(Debug, Clone, Copy)]
enum Drafts {
    None,
    Copy,
    /// Up to `depth` MTP drafts per sequence (all of them with `fixed`).
    Mtp { depth: usize, fixed: bool },
}

fn serve_loop(args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<()>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize,
    draft: Drafts, eos: Vec<u32>, decode_share: DecodeShareArgs) -> Result<()> {
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
        schedule(engine, &opened, &args.snapshot, &mut receive, &stats, max_sequences.min(DECODE_ROWS), draft, eos,
            decode_share)
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
    constraint: Option<crate::shared::constraints::State<'a>>,
    placement: Qwen4Placement,
    /// MTP stash rows and recent draft outcomes; cycles in a row planned
    /// without drafts (a probe draft follows eight).
    mtp: MtpSeq,
    outcomes: DraftHistory,
    idle: usize,
    /// Drafts proposed and accepted, verify cycles.
    proposed: usize,
    accepted: usize,
    cycles: usize,
    capacity: usize,
    next: u32,
    decoder: cuteafd_loader::StreamingTokenDecoder,
    generated: usize,
    buffered: usize,
    started: Instant,
    /// Admission order, for traces.
    id: u64,
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

/// An admitted prompt waiting for its remaining prefill chunks.
struct Prefill<'a> {
    job: NativeRequest,
    constraint: Option<crate::shared::constraints::State<'a>>,
    tokens: Vec<u32>,
    /// Prompt tokens prefilled so far.
    done: usize,
    placement: Qwen4Placement,
    capacity: usize,
    seq: MtpSeq,
    /// The last chunk's logits.
    logits: Option<Vec<f32>>,
    started: Instant,
    /// Seconds in this prompt's chunks, and their engine phases.
    busy: f64,
    phases: [f64; 2],
    id: u64,
}

/// Per-cycle step costs for policy work (`CUTEAFD_QWEN4_TRACE=path`, JSON lines).
struct Trace(std::io::BufWriter<std::fs::File>);

impl Trace {
    fn open() -> Result<Option<Self>> {
        let Ok(path) = std::env::var("CUTEAFD_QWEN4_TRACE") else { return Ok(None) };
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path)
            .with_context(|| format!("CUTEAFD_QWEN4_TRACE {path}"))?;
        Ok(Some(Self(std::io::BufWriter::new(file))))
    }

    fn cycle(&mut self, line: serde_json::Value) {
        use std::io::Write;
        if let Err(error) = writeln!(self.0, "{line}").and_then(|_| self.0.flush()) {
            tracing::warn!("trace: {error:#}");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn schedule(engine: &Qwen4Engine<'_>, opened: &Opened, snapshot: &std::path::Path,
    receive: &mut mpsc::Receiver<NativeRequest>, stats: &Mutex<serde_json::Value>, max_sequences: usize,
    drafts: Drafts, eos: Vec<u32>, decode_share: DecodeShareArgs) -> Result<()> {
    let mut allocator = Allocator::new(engine.pages, engine.slots, &engine.cfg);
    let mut grammars = crate::shared::constraints::Compiler::with_vocab(
        &opened.library, snapshot.join("tokenizer.json"), engine.cfg.vocab_size, eos);
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(snapshot)?;
    let mut active: Vec<Active<'_>> = Vec::new();
    let (mut requests, mut generated_total, mut admissions) = (0u64, 0u64, 0u64);
    // Verify steps since the last completed request, host seconds in verifies and in MTP draft steps.
    let (mut steps, mut verify_s) = (0u64, 0f64);
    let mut timing = DraftTiming::default();
    let (hidden, vocab) = (engine.cfg.hidden, engine.cfg.vocab_size);
    let embed = |tokens: &[u32]| embed_rows(&opened.checkpoint, tokens, hidden);
    let mtp = matches!(drafts, Drafts::Mtp { .. });
    let mut cost = mtp_policy::cycle_cost(DECODE_ROWS);
    let mut calibration = Calibration::default();
    let mut trace = Trace::open()?;
    let mut prefills = decode_share.queue::<Prefill<'_>>()?;
    loop {
        while active.len() + prefills.len() < max_sequences {
            let job = if active.is_empty() && prefills.is_empty() {
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
            // Room for the rows a verify may write past the last kept one.
            let placement = match allocator.admit((capacity + DECODE_ROWS).min(engine.max_context)) {
                Ok(placement) => placement,
                Err(error) => {
                    reject(&job, format!("{error:#}"));
                    continue;
                }
            };
            admissions += 1;
            let _ = job.events.blocking_send(Ok(InferenceChunk::Ready {
                system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: 0 },
            }));
            prefills.push(Prefill { job, constraint, tokens, done: 0, placement, capacity, seq: MtpSeq::default(),
                logits: None, started: Instant::now(), busy: 0.0, phases: [0.0; 2], id: admissions });
        }
        if prefills.due(!active.is_empty()) {
            // One chunk of each waiting prompt (whole prompts with --decode-share 0).
            let finished = prefills.round(|p| {
                anyhow::ensure!(!p.job.events.is_closed(), "client went away");
                let timer = Instant::now();
                let chunk = &p.tokens[p.done..(p.done + engine.prefill_rows).min(p.tokens.len())];
                let (result, phases) = isolated_phases(&engine.profile, || -> Result<()> {
                    let start = p.placement.len;
                    p.logits = engine.prefill(&mut p.placement, chunk, &embed(chunk)?)?;
                    if mtp {
                        speculate::prefill_chunk(engine, &embed, &p.placement, start, chunk,
                            p.tokens.get(p.done + chunk.len()).copied(), &mut p.seq)?;
                    }
                    Ok(())
                });
                p.done += chunk.len();
                add_phases(&mut p.phases, phases);
                p.busy += timer.elapsed().as_secs_f64();
                result?;
                Ok(if p.done == p.tokens.len() { Chunk::Done } else { Chunk::More })
            });
            for (p, prefilled) in finished {
                let elapsed = p.started.elapsed().as_secs_f64();
                let placement = p.placement.clone();
                let admitted = prefilled.and_then(|()| {
                    tracing::info!(tokens = p.tokens.len(), elapsed_ms = (1e3 * elapsed) as u64,
                        busy_ms = (1e3 * p.busy) as u64, tok_s = p.tokens.len() as f64 / p.busy,
                        gpu_wait_ms = (1e3 * p.phases[0]) as u64, experts_ms = (1e3 * p.phases[1]) as u64, "prefill");
                    let mut request = Active {
                        history: p.tokens,
                        draft_limit: COPY_DRAFT,
                        draft_pause: 0,
                        decoder: cuteafd_loader::streaming_token_decoder(snapshot, false)?,
                        job: p.job, constraint: p.constraint, placement: p.placement, mtp: p.seq,
                        outcomes: DraftHistory::default(), idle: 0, proposed: 0, accepted: 0, cycles: 0,
                        capacity: p.capacity, next: 0, generated: 0, buffered: 0, started: Instant::now(), id: p.id,
                    };
                    request.next = request.select(&p.logits.context("prefill produced no logits")?)?;
                    request.mtp.close(request.next);
                    Ok(request)
                });
                match admitted {
                    Ok(mut request) => {
                        let token = request.next;
                        match request.emit(token) {
                            Ok(false) => active.push(request),
                            Ok(true) | Err(_) => allocator.release(request.placement),
                        }
                    }
                    Err(error) => {
                        tracing::warn!("prefill failed: {error:#}");
                        allocator.release(placement);
                    }
                }
            }
            prefills.settle(!active.is_empty());
        }
        if active.is_empty() {
            continue;
        }
        let cycle = Instant::now();
        let mut rates: Vec<Vec<f64>> = Vec::new();
        let (steps_before, draft_s_before) = (timing.steps, timing.seconds);
        // Each sequence verifies its next token plus its drafts within the decode programs' rows.
        let room = (DECODE_ROWS / active.len()).max(1) - 1;
        let limits: Vec<usize> = active.iter().map(|a| room.min(a.job.max_tokens - a.generated - 1)
            .min(a.capacity - a.placement.len - 1)).collect();
        let proposals: Vec<Vec<u32>> = match drafts {
            Drafts::None => vec![Vec::new(); active.len()],
            Drafts::Copy => active.iter_mut().zip(&limits).map(|(a, &limit)| {
                if a.draft_pause > 0 {
                    a.draft_pause -= 1;
                    if a.draft_pause == 0 {
                        a.draft_limit = 1;
                    }
                }
                // `emit` already appended `next` to the history.
                copy_drafts(&a.history, limit.min(a.draft_limit))
            }).collect(),
            Drafts::Mtp { depth, fixed } => {
                let limits: Vec<usize> = limits.iter().map(|&l| l.min(depth)).collect();
                let histories: Vec<&DraftHistory> = active.iter().map(|a| &a.outcomes).collect();
                let confidence;
                (rates, confidence) = mtp_policy::acceptance(&histories, &limits, &calibration);
                let mut depths = mtp_policy::plan(&confidence, fixed.then_some(depth), &cost);
                for ((a, d), &limit) in active.iter_mut().zip(depths.iter_mut()).zip(&limits) {
                    // A sequence planned without drafts for a while probes one.
                    a.idle = if *d == 0 { a.idle + 1 } else { 0 };
                    if a.idle > 8 && limit > 0 {
                        *d = 1;
                        a.idle = 0;
                    }
                }
                let pending: usize = active.iter().map(|a| a.mtp.pending.len()).sum();
                let rows: usize = depths.iter().map(|d| d + 1).sum();
                if depths.iter().any(|&d| d > 0) || pending + rows > DECODE_ROWS / 2 {
                    let steps = timing.steps;
                    let timer = Instant::now();
                    let mut seqs: Vec<DraftSeq<'_>> = active.iter_mut().zip(&depths).map(|(a, &depth)| DraftSeq {
                        placement: &a.placement, seq: &mut a.mtp, depth }).collect();
                    let proposals = speculate::draft(engine, &embed, &mut seqs, &mut timing)?;
                    if depths.iter().any(|&d| d > 0) {
                        cost.observe_chain(timing.steps - steps, 1e3 * timer.elapsed().as_secs_f64());
                    }
                    proposals
                } else {
                    vec![Vec::new(); active.len()]
                }
            }
        };
        let sequences: Vec<Vec<u32>> = active.iter().zip(&proposals)
            .map(|(a, drafted)| std::iter::once(a.next).chain(drafted.iter().copied()).collect()).collect();
        let spec = sequences.iter().any(|rows| rows.len() > 1);
        let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
        let histories: Vec<_> = active.iter().map(|a| a.placement.history.clone()).collect();
        let tokens: Vec<u32> = sequences.iter().flatten().copied().collect();
        let mut rows: Vec<(&mut Qwen4Placement, &[u32])> = active.iter_mut().zip(&sequences)
            .map(|(a, s)| (&mut a.placement, s.as_slice())).collect();
        steps += 1;
        let timer = Instant::now();
        let step = embed(&tokens).and_then(|embedded| if spec {
            engine.verify_spec(&mut rows, &embedded, None)
        } else {
            engine.verify(&mut rows, &embedded, None)
        }).and_then(|logits| logits.context("decode needs every layer"));
        let elapsed = timer.elapsed().as_secs_f64();
        verify_s += elapsed;
        let shape = Shape::plain(tokens.len(), sequences.len());
        let predicted_ms = cost.verify_ms(shape);
        cost.observe_verify(shape, 1e3 * elapsed);
        let logits = match step {
            Ok(logits) => logits,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                for request in active.drain(..) {
                    let _ = request.job.events.blocking_send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    allocator.release(request.placement);
                }
                continue;
            }
        };
        let mut offset = 0;
        let mut kept = Vec::with_capacity(active.len());
        let finished: Vec<bool> = active.iter_mut().zip(&sequences).zip(&starts).enumerate()
            .map(|(index, ((request, rows), &start))| {
            let mut finished = false;
            let mut last = None;
            for j in 0..rows.len() {
                // Rows 0..=j are committed; the token row j produces is next.
                request.placement.len = start + j + 1;
                match request.select(&logits[(offset + j) * vocab..][..vocab]).and_then(|t| Ok((t, request.emit(t)?))) {
                    Ok((token, done)) => {
                        last = Some((j + 1, token));
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
            request.cycles += 1;
            request.proposed += drafted;
            request.accepted += accepted;
            if matches!(drafts, Drafts::Mtp { .. }) {
                request.outcomes.observe(drafted, accepted);
                if let Some(rates) = rates.get(index) {
                    calibration.observe_outcome(rates, drafted, accepted);
                }
            } else if drafted > 0 && accepted == 0 {
                request.draft_limit /= 2;
                if request.draft_limit == 0 {
                    request.draft_pause = 8;
                }
            } else if drafted > 0 && accepted == drafted {
                request.draft_limit = (request.draft_limit * 2).clamp(1, COPY_DRAFT);
            }
            kept.push(if finished { None } else { last });
            finished
        }).collect();
        // Commit the kept rows (speculative steps), rewind, and stash them for the MTP.
        let mut first_row = 0;
        let mut verified: Vec<Verified<'_>> = Vec::with_capacity(active.len());
        for (((request, rows), (&start, history)), kept) in active.iter_mut().zip(&sequences)
            .zip(starts.iter().zip(histories)).zip(&kept) {
            verified.push(Verified { placement: &mut request.placement, seq: &mut request.mtp, start, history,
                rows, first_row, kept: *kept });
            first_row += rows.len();
        }
        speculate::accept(engine, &mut verified, spec, mtp)?;
        drop(verified);
        if let Some(trace) = trace.as_mut() {
            trace.cycle(serde_json::json!({"rows": tokens.len(), "seqs": sequences.len(),
                "ids": active.iter().map(|a| a.id).collect::<Vec<_>>(),
                "depths": sequences.iter().map(|s| s.len() - 1).collect::<Vec<_>>(),
                "kept": kept.iter().map(|k| k.map_or(0, |(n, _)| n)).collect::<Vec<_>>(),
                "verify_ms": 1e3 * elapsed, "draft_steps": timing.steps - steps_before,
                "draft_ms": 1e3 * (timing.seconds - draft_s_before), "cycle_ms": 1e3 * cycle.elapsed().as_secs_f64(),
                "predicted_ms": predicted_ms, "fit": cost.fitted(), "rates": rates,
                "calibration": calibration.fitted()}));
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
                active = active.len(), steps, cycles = request.cycles, proposed = request.proposed,
                accepted = request.accepted, tokens_per_cycle = request.generated as f64 / request.cycles.max(1) as f64,
                verify_s, draft_s = timing.seconds, draft_steps = timing.steps, gpu_wait_s = phases[0],
                experts_s = phases[1], "request complete");
            (steps, verify_s, timing) = (0, 0.0, DraftTiming::default());
            allocator.release(request.placement);
        }
        if let Ok(mut stats) = stats.lock() {
            *stats = serde_json::json!({"requests": requests, "generated_tokens": generated_total,
                "active": active.len(), "prefilling": prefills.len()});
        }
        prefills.stepped(cycle.elapsed().as_secs_f64());
    }
}
