//! OpenAI-compatible serving for DeepSeek V4: one request at a time through
//! the prefill/decode engine (the correctness baseline batching builds on).
use super::pool::{Placement, PoolAllocator};
use super::{with_engine, EngineArgs};
use crate::shared::token_io::{RowResult, SelectBatch, TokenSelector};
use crate::shared::prefill_share::{Chunk, DecodeShareArgs};
use anyhow::{Context, Result};
use cuteafd_api::openai::{
    ConsoleHub, InferenceChunk, InferenceFinishReason, ModelEncoding, ModelProfile, NativeFailure, NativeLimits,
    NativeRequest, PromptUsage,
};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[derive(Debug, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    /// Longest prompt + output accepted (at most the programs' max context).
    #[arg(long, default_value_t = 8192)]
    pub max_context: u32,
    #[arg(long, default_value_t = 4096)]
    pub max_output: u32,
    /// Public model id; defaults to the snapshot's Hugging Face id.
    #[arg(long)]
    pub model_id: Option<String>,
    /// With --dspark, speculate while at most this many sequences decode
    /// (and their verify rows fit the decode programs).
    #[arg(long, default_value_t = 10)]
    pub speculate_max_sequences: usize,
    #[command(flatten)]
    pub decode_share: DecodeShareArgs,
}

/// "…/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/<rev>" -> "deepseek-ai/DeepSeek-V4-Flash-0731".
fn model_id(snapshot: &Path) -> Option<String> {
    snapshot.ancestors().find_map(|dir| {
        let name = dir.file_name()?.to_str()?.strip_prefix("models--")?;
        let (org, model) = name.split_once("--")?;
        Some(format!("{org}/{model}"))
    })
}

fn eos_token(snapshot: &Path) -> Result<u32> {
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(snapshot.join("generation_config.json"))?)?;
    Ok(config["eos_token_id"].as_u64().context("generation_config.json eos_token_id")? as u32)
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    let limits = NativeLimits::new(args.max_context, args.max_output)?;
    let profile = ModelProfile::new(
        args.model_id.clone().or_else(|| model_id(&args.engine.snapshot)).context("model id")?,
        ModelEncoding::DeepseekV4,
    );
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let engine_args = args.engine.clone();
    let worker_stats = stats.clone();
    let (max_context, speculate_max, decode_share) =
        (args.max_context as usize, args.speculate_max_sequences, args.decode_share);
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_context, speculate_max, decode_share));
    ready_rx.await.context("engine failed before it was ready")??;
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, Duration::from_secs(25),
        ConsoleHub::disabled(), profile.clone());
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    tracing::info!(listen = %args.listen, model = %profile.id, "DeepSeek V4 API is ready");
    tokio::select! {
        served = axum::serve(listener, router) => served?,
        finished = worker => finished??,
    }
    Ok(())
}

fn serve_loop(
    args: EngineArgs,
    mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<()>>,
    stats: Arc<Mutex<serde_json::Value>>,
    max_context: usize,
    speculate_max: usize,
    decode_share: DecodeShareArgs,
) -> Result<()> {
    let loaded = match super::load(&args) {
        Ok(loaded) => loaded,
        Err(error) => {
            let _ = ready.send(Err(anyhow::anyhow!("{error:#}")));
            return Ok(());
        }
    };
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(&args.snapshot)?;
    let eos = eos_token(&args.snapshot)?;
    let mut ready = Some(ready);
    let result = with_engine(&loaded, &args, |engine, transports, runtime| {
        if max_context > engine.max_context {
            tracing::warn!(requested = max_context, supported = engine.max_context,
                "--max-context exceeds the exported programs; requests are limited to the programs' context");
        }
        if let Some(ready) = ready.take() {
            let _ = ready.send(Ok(()));
        }
        let mut selector = TokenSelector::new(&loaded.library, args.token_io.token_select, engine.cfg.vocab_size,
            engine.decode_rows)?;
        schedule(engine, &loaded, &tokenizer, eos, &mut receive, transports, runtime, &stats, speculate_max,
            decode_share, &mut selector)
    });
    if let Some(ready) = ready.take() {
        let _ = ready.send(result.as_ref().map(|_| ()).map_err(|e| anyhow::anyhow!("{e:#}")));
    }
    result
}

/// An admitted prompt waiting for its remaining prefill chunks (equal
/// chunks of `limit` tokens; the last one returns the logits).
struct Prefill<'a> {
    job: NativeRequest,
    constraint: Option<crate::shared::constraints::State<'a>>,
    tokens: Vec<u32>,
    limit: usize,
    /// Chunks prefilled so far, of `chunks`.
    index: usize,
    chunks: usize,
    placement: Placement,
    capacity: usize,
    /// The first generated token, selected after the last chunk.
    first: Option<u32>,
    started: Instant,
    /// Seconds in this prompt's chunks.
    busy: f64,
}

/// One admitted request: its placement, stream state and next input token.
struct Active<'a> {
    job: NativeRequest,
    /// Grammar for structured output and tool calls.
    constraint: Option<crate::shared::constraints::State<'a>>,
    placement: Placement,
    capacity: usize,
    next: u32,
    decoder: cuteafd_loader::StreamingTokenDecoder,
    generated: usize,
    buffered: usize,
    started: Instant,
}

impl Active<'_> {
    fn send(&self, chunk: InferenceChunk) -> Result<()> {
        self.job.events.send(Ok(chunk)).map_err(|_| anyhow::anyhow!("client went away"))
    }

    /// Streams `token`; returns true when the request is finished.
    fn emit(&mut self, token: u32, eos: u32) -> Result<bool> {
        self.generated += 1;
        self.buffered += 1;
        if token != eos {
            if let Some(content) = self.decoder.step(token)? {
                self.send(InferenceChunk::Text { content, content_tokens: self.buffered })?;
                self.buffered = 0;
            }
        }
        let finish = if token == eos {
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

/// Commits a selected token to the request's grammar.
fn take(constraint: Option<&mut crate::shared::constraints::State<'_>>, selected: &RowResult) -> Result<u32> {
    let token = selected.as_ref().map_err(|e| anyhow::anyhow!("sampling: {e:?}"))?.token;
    if let Some(state) = constraint {
        state.accept(token)?;
    }
    Ok(token)
}

/// Each row draws at the position after it, masked along its sequence's drafts.
fn select_rows(selector: &mut TokenSelector<'_>, logits: &crate::shared::token_io::DeviceLogits, active: &[Active<'_>],
    sequences: &[Vec<u32>], starts: &[usize]) -> Result<Vec<RowResult>> {
    let mut batch = SelectBatch::default();
    for ((a, rows), &start) in active.iter().zip(sequences).zip(starts) {
        batch.push_sequence(a.job.sampling, a.constraint.as_ref(), rows, start as u64 + 1)?;
    }
    selector.select(logits, &batch)
}

/// Drafts after every active sequence's next token, verifies `[next, drafts]`
/// in one step and accepts each sequence's longest matching prefix plus the
/// verifier's own token after it. Returns which requests finished.
#[allow(clippy::too_many_arguments)]
fn speculative_step(
    engine: &super::engine::Engine<'_>,
    active: &mut [Active<'_>],
    block: usize,
    eos: u32,
    transports: &mut [crate::shared::spark_intake::SparkLink<'_>],
    runtime: &tokio::runtime::Runtime,
    selector: &mut TokenSelector<'_>,
) -> Result<Vec<bool>> {
    let noise = engine.cfg.dspark_noise_token_id as u32;
    let inputs: Vec<u32> = active.iter()
        .flat_map(|a| std::iter::once(a.next).chain(std::iter::repeat_n(noise, block - 1))).collect();
    let requests: Vec<super::engine::DraftRequest<'_>> = active.iter()
        .map(|a| super::engine::DraftRequest { placement: &a.placement, token: a.next }).collect();
    let drafts = engine.draft(&requests, &inputs)?;
    // Verify no more rows than the request may still produce or hold, and no
    // draft the grammar rejects (it could never be kept).
    let sequences: Vec<Vec<u32>> = active.iter().zip(&drafts).map(|(a, draft)| {
        let room = (a.job.max_tokens - a.generated).min(a.capacity - a.placement.len - 1);
        let mut rows: Vec<u32> =
            std::iter::once(a.next).chain(draft.iter().copied().take(room.saturating_sub(1).min(block))).collect();
        if let Some(state) = a.constraint.as_ref() {
            state.truncate_proposal(&mut rows)?;
        }
        Ok(rows)
    }).collect::<Result<_>>()?;
    let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
    let mut rows: Vec<(&mut Placement, &[u32])> = active.iter_mut().zip(&sequences)
        .map(|(a, tokens)| (&mut a.placement, tokens.as_slice())).collect();
    let logits = engine.verify_device(&mut rows, transports, runtime)?;
    let selected = select_rows(selector, &logits, active, &sequences, &starts)?;
    let mut offset = 0;
    Ok(active.iter_mut().zip(&sequences).zip(starts).map(|((request, rows), start)| {
        let mut finished = false;
        for (j, _) in rows.iter().enumerate() {
            // Rows 0..=j are committed; the token row j produces is next.
            request.placement.len = start + j + 1;
            let result = take(request.constraint.as_mut(), &selected[offset + j])
                .and_then(|token| Ok((token, request.emit(token, eos)?)));
            match result {
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
        finished
    }).collect())
}

/// Continuous batching: prefill each new request (one sequence per prefill),
/// then advance every active sequence by one token in a single decode step.
#[allow(clippy::too_many_arguments)]
fn schedule(
    engine: &super::engine::Engine<'_>,
    loaded: &super::Loaded,
    tokenizer: &cuteafd_loader::LoadedTokenizer,
    eos: u32,
    receive: &mut mpsc::Receiver<NativeRequest>,
    transports: &mut [crate::shared::spark_intake::SparkLink<'_>],
    runtime: &tokio::runtime::Runtime,
    stats: &Mutex<serde_json::Value>,
    speculate_max: usize,
    decode_share: DecodeShareArgs,
    selector: &mut TokenSelector<'_>,
) -> Result<()> {
    let mut allocator = PoolAllocator::new(engine.shape);
    let mut grammars = crate::shared::constraints::Compiler::new(
        &loaded.library, loaded.snapshot.join("tokenizer.json"));
    let mut active: Vec<Active<'_>> = Vec::new();
    let (mut requests, mut generated_total) = (0u64, 0u64);
    let chunk_limit = engine.prefill_capacity().min(engine.max_context);
    let mut prefills = decode_share.queue::<Prefill<'_>>()?;
    loop {
        // Admit while sequence slots and decode rows remain.
        while allocator.free_states() > 0 && active.len() + prefills.len() < engine.decode_rows {
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
                let _ = job.events.send(Err(NativeFailure::BadRequest(message)));
            };
            if !job.images.is_empty() {
                reject(&job, "this checkpoint takes no images".into());
                continue;
            }
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
            let placement = match allocator.admit(capacity) {
                Ok(placement) => placement,
                Err(error) => {
                    reject(&job, format!("{error:#}"));
                    continue;
                }
            };
            let _ = job.events.send(Ok(InferenceChunk::Ready {
                system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: 0 },
            }));
            // Equal chunks, so no lane runs a tiny tail.
            let chunks = tokens.len().div_ceil(chunk_limit);
            let limit = tokens.len().div_ceil(chunks);
            prefills.push(Prefill { job, constraint, tokens, limit, index: 0, chunks, placement, capacity,
                first: None, started: Instant::now(), busy: 0.0 });
        }
        if prefills.due(!active.is_empty()) {
            // One chunk of each waiting prompt (whole prompts with --decode-share 0).
            let finished = prefills.round(|p| {
                anyhow::ensure!(!p.job.events.is_closed(), "client went away");
                let timer = Instant::now();
                let from = p.index * p.limit;
                let chunk = &p.tokens[from..(from + p.limit).min(p.tokens.len())];
                let last = p.index + 1 == p.chunks;
                let logits = engine.prefill_device(&mut p.placement, chunk, transports, runtime, usize::from(last))?;
                if last {
                    // The first token, while this prompt's logits are the workspace's.
                    let logits = logits.context("prefill produced no logits")?;
                    let mut batch = SelectBatch::default();
                    batch.push_next(p.job.sampling, p.constraint.as_mut(), p.placement.len as u64)?;
                    let selected = selector.select(&logits, &batch)?;
                    p.first = Some(take(p.constraint.as_mut(), &selected[0])?);
                }
                p.index += 1;
                p.busy += timer.elapsed().as_secs_f64();
                Ok(if p.index == p.chunks { Chunk::Done } else { Chunk::More })
            });
            for (p, prefilled) in finished {
                let placement = p.placement.clone();
                let admitted = prefilled.and_then(|()| {
                    tracing::debug!(tokens = p.tokens.len(), elapsed_ms = p.started.elapsed().as_millis() as u64,
                        busy_ms = (1e3 * p.busy) as u64, "prefill");
                    let mut request = Active {
                        decoder: cuteafd_loader::streaming_token_decoder(&loaded.snapshot, false)?,
                        job: p.job, constraint: p.constraint, placement: p.placement, capacity: p.capacity, next: 0,
                        generated: 0, buffered: 0, started: Instant::now(),
                    };
                    request.next = p.first.context("prefill produced no first token")?;
                    Ok(request)
                });
                match admitted {
                    Ok(mut request) => {
                        let token = request.next;
                        match request.emit(token, eos) {
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
        // One decode step over every active sequence; with the drafter and a
        // small batch, each sequence verifies its next token plus a draft.
        let block = engine.draft_block();
        let speculate = block > 0 && active.len() <= speculate_max
            && active.len() * (block + 1) <= engine.decode_rows;
        let step = if speculate {
            speculative_step(engine, &mut active, block, eos, transports, runtime, selector)
        } else {
            let sequences: Vec<Vec<u32>> = active.iter().map(|a| vec![a.next]).collect();
            let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
            let mut rows: Vec<(&mut Placement, u32)> = active.iter_mut().map(|a| (&mut a.placement, a.next)).collect();
            engine.decode_device(&mut rows, transports.first_mut(), runtime)
                .and_then(|logits| select_rows(selector, &logits, &active, &sequences, &starts))
                .map(|selected| active.iter_mut().zip(&selected).map(|(request, selected)| {
                    take(request.constraint.as_mut(), selected).and_then(|token| request.emit(token, eos)).unwrap_or(true)
                }).collect::<Vec<bool>>())
        };
        let finished = match step {
            Ok(finished) => finished,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                for request in active.drain(..) {
                    let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    allocator.release(request.placement);
                }
                continue;
            }
        };
        for index in (0..active.len()).rev() {
            if !finished[index] {
                continue;
            }
            let request = active.remove(index);
            requests += 1;
            generated_total += request.generated as u64;
            let seconds = request.started.elapsed().as_secs_f64();
            tracing::info!(tokens = request.generated, seconds, tok_s = request.generated as f64 / seconds,
                active = active.len(), "request complete");
            allocator.release(request.placement);
        }
        if let Ok(mut stats) = stats.lock() {
            *stats = serde_json::json!({"requests": requests, "generated_tokens": generated_total, "active": active.len(),
                "prefilling": prefills.len()});
        }
        prefills.stepped(cycle.elapsed().as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn model_id_from_snapshot_path() {
        let path = std::path::Path::new("/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/abc");
        assert_eq!(super::model_id(path).as_deref(), Some("deepseek-ai/DeepSeek-V4-Flash-0731"));
    }
}
