//! OpenAI-compatible API over the MiMo V2 engine: continuous batching with
//! one prefill per admitted request and one decode-shaped step for every
//! active sequence, verifying copy-window drafts.
//!
//! MiMo keeps no recurrent state: full layers write paged records and SWA
//! layers a 256-slot ring that outlives the 128-token window by more than a
//! step's rows, so a verify whose drafts are rejected just sets the
//! sequence length back.
//!
//! Prefix cache (`cuteafd_engine::prefix` over `super::prefix::MimoPrefix`):
//! admission looks the prompt up, restores the longest retained exact
//! frontier (shared pages, the copied tail page, the SWA/MTP mark) and
//! prefills only the rest; a whole-prompt hit takes its first token from the
//! retained logits. The prompt is retained at prompt end (a `Prompt`
//! snapshot, unless it was a whole hit), the conversation at a normal finish
//! (`Turn`, EOS or max_tokens with the client still there); a prefill whose
//! client left is parked at its last chunk boundary; a decode whose client
//! left is not retained. `prompt_cache_hit_tokens` reports the restored rows.
use super::dflash::{ContextRow, DraftSeq};
use super::mtp::MtpSeq;
use super::engine::{MimoEngine, MimoPlacement, DECODE_ROWS};
use super::prefix::MimoPrefix;
use crate::v41_native_serve::prefix::CudaCopyEngine;
use cuteafd_engine::prefix::{After, MarkArena, PrefixCache, PrefixConfig, SnapshotKind};
use crate::glm::dflash_policy::{self, CycleCost, DraftHistory, Group, Shape};
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
    #[command(flatten)]
    pub prefix: PrefixArgs,
}

/// The prefix cache's knobs.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct PrefixArgs {
    /// Retained snapshots per bank (prompts, completed turns); 0 turns the prefix cache off.
    #[arg(long, env = "CUTEAFD_PREFIX_CACHE_ENTRIES", default_value_t = 20)]
    pub prefix_cache_entries: usize,
    /// Device memory for retained positional marks (SWA rows, MTP hidden rows), MiB; the arena
    /// holds two marks per entry pair while they fit, and never fewer than four.
    #[arg(long, env = "CUTEAFD_PREFIX_CACHE_MARK_MIB", default_value_t = 2048)]
    pub prefix_cache_mark_mib: usize,
    /// Shortest prompt or turn worth a snapshot.
    #[arg(long, default_value_t = 64)]
    pub prefix_cache_min_tokens: usize,
    /// Pinned host memory for snapshots the device evicts (e.g. 64GiB; 0 = off).
    #[arg(long, env = "CUTEAFD_HOST_CACHE_BYTES", default_value = "0", value_parser = parse_bytes)]
    pub host_cache_bytes: u64,
    /// Shortest snapshot the host tier keeps.
    #[arg(long, default_value_t = 512)]
    pub host_cache_min_tokens: u32,
}

/// `123`, `512MiB`, `64GiB`, `1.5GB`.
pub(crate) fn parse_bytes(text: &str) -> std::result::Result<u64, String> {
    let text = text.trim();
    let split = text.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let scale: f64 = match unit {
        "" | "B" => 1.0,
        "KB" => 1e3,
        "MB" => 1e6,
        "GB" => 1e9,
        "KiB" => 1024.0,
        "MiB" => 1024.0 * 1024.0,
        "GiB" => 1024.0 * 1024.0 * 1024.0,
        other => return Err(format!("unknown byte unit {other:?}")),
    };
    let value: f64 = number.parse().map_err(|e| format!("{text:?}: {e}"))?;
    if !(value >= 0.0) {
        return Err(format!("{text:?} is negative"));
    }
    Ok((value * scale) as u64)
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
    let prefix = args.prefix.clone();
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, draft, prefix));
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

/// Verify-step ms by rows for MiMo V2.6 Pro, RTX PRO 6000 (GPU0, 325 W) + six
/// Sparks (TP6 MXFP4, FP8 wire rows): teacher-forced decode steps of one
/// sequence (`mimo-golden --timing --prefill 1000 --step-rows N`, 571 rows).
/// The coordinator alone (`--skip-experts`) takes 18.2 / 19.2 / 23.3 / 33.8 /
/// 37.2 ms at 1 / 4 / 16 / 24 / 32 rows (E4M3 decode weights up to 32 rows)
/// and 56.7 ms at 48; the rest is the Spark exchange. Serving refits its
/// intercept and slope as it observes (sequences that share experts make rows
/// cheaper than one sequence's).
const PRO_TP6_STEP_MS: [(usize, f64); 9] = [(1, 31.6), (2, 39.5), (4, 54.1), (8, 77.1), (16, 114.4), (24, 166.7),
    (32, 197.8), (48, 258.3), (64, 304.4)];

fn serve_loop(args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<()>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize,
    draft: Policy, prefix: PrefixArgs) -> Result<()> {
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
        schedule(engine, &opened, &args.snapshot, &mut receive, &stats, max_sequences.min(DECODE_ROWS), draft, &prefix)
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
    /// First position the DFlash drafter's context holds for this sequence
    /// (the prefix-cache restore point; the drafter starts cold there).
    draft_from: usize,
    /// The logit row that produced the last token, once the request finished
    /// normally (EOS or max_tokens with the client still there): what follows
    /// its `Turn` snapshot.
    turn: Option<Vec<f32>>,
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

/// The prefix cache over `engine` (always present: with zero entries it is the page allocator).
fn prefix_cache<'e, 'a>(engine: &'e MimoEngine<'a>, args: &PrefixArgs)
    -> Result<(MimoPrefix<'e, 'a>, PrefixCache<CudaCopyEngine<'a>>)> {
    let entries = args.prefix_cache_entries;
    let budget = args.prefix_cache_mark_mib << 20;
    let family = MimoPrefix::new(engine, |mark| if entries == 0 { 0 } else { MarkArena::slots_for(1, entries, mark, budget) })?;
    let host = if entries > 0 && args.host_cache_bytes > 0 {
        let config = cuteafd_hostcache::config::Config {
            bytes: args.host_cache_bytes,
            chunk_bytes: (256u64 << 20).max(family.mark_bytes() as u64).min(args.host_cache_bytes),
            min_tokens: args.host_cache_min_tokens,
            ..Default::default()
        };
        Some((config, CudaCopyEngine::new(engine.library, engine.kv_layer(0).1)?))
    } else {
        None
    };
    let layout = cuteafd_engine::prefix::PrefixFamily::layout(&family);
    let config = PrefixConfig { entries, mark_slots: family.slots(), keep_logits: true,
        min_tokens: args.prefix_cache_min_tokens };
    let cache = PrefixCache::new(layout, config, host)?;
    tracing::info!(entries, mark_slots = family.slots(), mark_bytes = family.mark_bytes(), page_bytes = layout.page_bytes,
        pages = layout.pages, host_bytes = args.host_cache_bytes, "MiMo prefix cache");
    Ok((family, cache))
}

/// Gives a finished or failed sequence's pages, ring and drafter slot back.
fn release(family: &MimoPrefix<'_, '_>, cache: &mut PrefixCache<CudaCopyEngine<'_>>, rings: &mut Vec<i32>,
    slots: &mut Vec<usize>, placement: &MimoPlacement, slot: Option<usize>) {
    if let Err(error) = cache.release(family, &placement.pages) {
        tracing::error!(%error, "releasing a MiMo sequence's pages");
    }
    rings.push(placement.ring);
    slots.extend(slot);
}

/// Prefills `tokens[placement.len..]` chunk by chunk (feeding the drafter each chunk's tapped
/// tail) and returns the last row's logits, or `None` when the client left at a chunk boundary
/// (the rows prefilled so far stay in the placement).
fn prefill_suffix(engine: &MimoEngine<'_>, embeddings: &super::Embeddings, slot: Option<usize>,
    placement: &mut MimoPlacement, tokens: &[u32], job: &NativeRequest) -> Result<Option<Vec<f32>>> {
    let started = Instant::now();
    let first = placement.len;
    let mut logits = None;
    for chunk in tokens[first..].chunks(engine.prefill_rows) {
        if job.events.is_closed() {
            return Ok(None);
        }
        let embed = embeddings.rows(chunk)?;
        let start = placement.len;
        logits = engine.prefill(placement, &embed, None)?;
        // The chunk's tapped tail becomes the drafter's context.
        if let (Some(drafter), Some(slot)) = (engine.drafter.as_ref(), slot) {
            let n = chunk.len().min(super::dflash::TAP_ROWS);
            drafter.update(&(0..n).map(|r| ContextRow { tap_row: r, slot,
                position: start + chunk.len() - n + r }).collect::<Vec<_>>())?;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let phases = std::mem::take(&mut *engine.profile.borrow_mut());
    tracing::info!(tokens = tokens.len(), cached = first, elapsed_ms = (1e3 * elapsed) as u64,
        tok_s = (tokens.len() - first) as f64 / elapsed, gpu_wait_ms = (1e3 * phases[0]) as u64,
        experts_ms = (1e3 * phases[1]) as u64, "prefill");
    logits.context("prefill produced no logits").map(Some)
}

#[allow(clippy::too_many_arguments)]
fn schedule(engine: &MimoEngine<'_>, opened: &Opened, snapshot: &std::path::Path,
    receive: &mut mpsc::Receiver<NativeRequest>, stats: &Mutex<serde_json::Value>, max_sequences: usize,
    policy: Policy, prefix: &PrefixArgs) -> Result<()> {
    let draft = policy.copy;
    let drafter = engine.drafter.as_ref();
    let mut free_slots: Vec<usize> = drafter.map_or(Vec::new(), |d| (0..d.slots).rev().collect());
    let mut cost = dflash_policy::step_cost(&PRO_TP6_STEP_MS, DECODE_ROWS);
    let mut skip = crate::glm::dflash_policy::DraftSkip::default();
    let (family, mut cache) = prefix_cache(engine, prefix)?;
    let mut free_rings: Vec<i32> = (0..engine.rings as i32).rev().collect();
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
    let vocab = engine.cfg.vocab_size;
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
            let Some(ring) = free_rings.pop() else {
                reject(&job, "SWA rings exhausted".into());
                continue;
            };
            let slot = free_slots.pop();
            cache.tick();
            let admitted = match cache.admit(&family, &tokens, capacity, true,
                |pages| MimoPlacement { pages, ring, len: 0 }) {
                Ok(admitted) => admitted,
                Err(error) => {
                    free_rings.push(ring);
                    free_slots.extend(slot);
                    reject(&job, format!("{error:#}"));
                    continue;
                }
            };
            let resume = admitted.resume;
            let mut placement = admitted.placement;
            let _ = job.events.blocking_send(Ok(InferenceChunk::Ready {
                system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: resume },
            }));
            if let Some(source) = admitted.source {
                tracing::info!(tokens = tokens.len(), resume, kind = ?source.kind, frontier = source.frontier,
                    host = source.host, "prefix cache hit");
            }
            let logits = match admitted.after.and_then(|after| after.logits) {
                // The whole prompt was retained: its logits give the first token.
                Some(logits) => Ok(Some(logits.to_vec())),
                None => prefill_suffix(engine, &embeddings, slot, &mut placement, &tokens, &job),
            };
            let logits = match logits {
                Ok(Some(logits)) => logits,
                Ok(None) => {
                    // The client left during the prefill: keep what it computed for a retry.
                    if placement.len > resume {
                        if let Err(error) = cache.park(&family, &tokens[..placement.len], &placement) {
                            tracing::warn!("parking a cancelled prefill: {error:#}");
                        }
                    }
                    release(&family, &mut cache, &mut free_rings, &mut free_slots, &placement, slot);
                    continue;
                }
                Err(error) => {
                    tracing::warn!("prefill failed: {error:#}");
                    let _ = job.events.blocking_send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    release(&family, &mut cache, &mut free_rings, &mut free_slots, &placement, slot);
                    continue;
                }
            };
            // The prompt snapshot, taken once the first token is out (it only enqueues copies).
            let mut retain_prompt = |cache: &mut PrefixCache<CudaCopyEngine<'_>>, placement: &MimoPlacement| {
                if resume < tokens.len() {
                    if let Err(error) = cache.capture(&family, SnapshotKind::Prompt, &tokens, placement,
                        After::from_logits(&logits, true)) {
                        tracing::warn!("prompt snapshot not retained: {error:#}");
                    }
                }
            };
            engine.mtp_reset(placement.ring as usize, placement.len);
            let admitted = (|| -> Result<Active<'_>> {
                let mut request = Active {
                    slot,
                    drafts: DraftHistory::default(),
                    counts: [0; 5],
                    history: tokens.clone(),
                    draft_limit: draft,
                    draft_pause: 0,
                    decoder: cuteafd_loader::streaming_token_decoder(snapshot, false)?,
                    job, constraint, placement: placement.clone(), draft_from: resume, turn: None, capacity, next: 0,
                    generated: 0, buffered: 0, started: Instant::now(),
                };
                request.next = request.select(&logits)?;
                Ok(request)
            })();
            match admitted {
                Ok(mut request) => {
                    let token = request.next;
                    let emitted = request.emit(token);
                    retain_prompt(&mut cache, &request.placement);
                    match emitted {
                        Ok(false) => active.push(request),
                        // Finished at its first token: its turn is its prompt snapshot.
                        Ok(true) | Err(_) => {
                            release(&family, &mut cache, &mut free_rings, &mut free_slots, &request.placement, request.slot)
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!("admission failed: {error:#}");
                    retain_prompt(&mut cache, &placement);
                    release(&family, &mut cache, &mut free_rings, &mut free_slots, &placement, slot);
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
                    .filter_map(|(i, a)| a.slot.map(|slot| (i, DraftSeq { slot, anchor: a.next, position: a.placement.len,
                        valid_from: a.draft_from })))
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
            // Native MTP drafts (MiMo V2 Flash; V2.6 Pro prefers DFlash).
            None if skip.drafts() && engine.mtp.is_some() => {
                let seqs: Vec<MtpSeq<'_>> = active.iter()
                    .map(|a| MtpSeq { ring: a.placement.ring as usize, len: a.placement.len, tokens: &a.history })
                    .collect();
                let stages = engine.mtp.as_ref().map_or(0, |m| m.stages.len());
                let timer = Instant::now();
                let drafts = engine.mtp_draft(&seqs, stages, &|ids: &[u32]| embeddings.rows(ids));
                let ms = timer.elapsed().as_secs_f64() * 1e3;
                cost.observe_draft(ms);
                draft_s += ms / 1e3;
                match drafts {
                    Ok(drafts) => drafts.into_iter().map(|tokens| {
                        let features = vec![[0.0, 1.0, 0.0, 0.0]; tokens.len()];
                        Some(super::dflash::Draft { tokens, features })
                    }).collect(),
                    Err(error) => {
                        tracing::warn!("MTP draft failed: {error:#}");
                        vec![None; active.len()]
                    }
                }
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
                    a.draft_limit = draft.min(1);
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
        cost.observe_verify(Shape::plain(tokens.len(), sequences.len()), step_s * 1e3);
        let timer = Instant::now();
        let logits = match step {
            Ok(logits) => logits,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                for request in active.drain(..) {
                    let _ = request.job.events.blocking_send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    release(&family, &mut cache, &mut free_rings, &mut free_slots, &request.placement, request.slot);
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
                let row = &logits[(offset + j) * vocab..][..vocab];
                match request.select(row).and_then(|t| Ok((t, request.emit(t)?))) {
                    Ok((token, done)) => {
                        finished = done;
                        if done {
                            // A normal finish (the client took the last chunk): the row that
                            // produced the last token follows the turn snapshot.
                            request.turn = Some(row.to_vec());
                        }
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
            if used_copy[i] || (drafted > 0 && drafter.is_none() && engine.mtp.is_none()) {
                if drafted > 0 && accepted == 0 {
                    request.draft_limit /= 2;
                    if request.draft_limit == 0 {
                        request.draft_pause = 8;
                    }
                } else if drafted > 0 && accepted == drafted {
                    request.draft_limit = (request.draft_limit * 2).clamp(draft.min(1), draft);
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
            if let Some(row) = &request.turn {
                // The conversation so far: every committed row (the last token is not in it).
                let rows = &request.history[..request.placement.len];
                if let Err(error) = cache.capture(&family, SnapshotKind::Turn, rows, &request.placement,
                    After::from_logits(row, true)) {
                    tracing::warn!("turn snapshot not retained: {error:#}");
                }
            }
            release(&family, &mut cache, &mut free_rings, &mut free_slots, &request.placement, request.slot);
        }
        cache.tick();
        if let Ok(mut stats) = stats.lock() {
            *stats = serde_json::json!({"requests": requests, "generated_tokens": generated_total,
                "active": active.len(), "prefix_cache": cache.stats()});
        }
    }
}

/// DFlash draft counts: `fixed` (within each limit), or the adaptive plan
/// (glmrt v9's schedule, `dflash_policy::plan`) over the drafting sequences
/// with each one's history of conditional acceptance (the GLM selector
/// calibration does not apply to this drafter), priced with the sequences
/// that do not draft.
fn plan_drafts(active: &[Active<'_>], drafted: &[Option<super::dflash::Draft>], limits: &[usize],
    fixed: Option<usize>, cost: &CycleCost) -> Vec<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        prefix: PrefixArgs,
    }

    #[test]
    fn prefix_knobs_default_on_with_the_host_tier_off() {
        let cli = Cli::parse_from(["serve"]);
        assert_eq!((cli.prefix.prefix_cache_entries, cli.prefix.host_cache_bytes), (20, 0));
        let cli = Cli::parse_from(["serve", "--prefix-cache-entries", "0", "--host-cache-bytes", "64GiB"]);
        assert_eq!((cli.prefix.prefix_cache_entries, cli.prefix.host_cache_bytes), (0, 64 << 30));
        assert_eq!(parse_bytes("512MiB"), Ok(512 << 20));
        assert_eq!(parse_bytes("1.5GB"), Ok(1_500_000_000));
        assert_eq!(parse_bytes("123"), Ok(123));
        assert!(parse_bytes("12 parsecs").is_err() && parse_bytes("-1").is_err());
    }
}
