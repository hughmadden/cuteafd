//! OpenAI-compatible serving for DeepSeek V4: one request at a time through
//! the prefill/decode engine (the correctness baseline batching builds on).
use super::pool::{Placement, PoolAllocator};
use super::{embed_rows, with_engine, EngineArgs};
use anyhow::{Context, Result};
use cuteafd_api::native_v41::{
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
    let profile = ModelProfile {
        id: args.model_id.clone().or_else(|| model_id(&args.engine.snapshot)).context("model id")?,
        encoding: ModelEncoding::DeepseekV4,
    };
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let engine_args = args.engine.clone();
    let worker_stats = stats.clone();
    let (max_context, speculate_max) = (args.max_context as usize, args.speculate_max_sequences);
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_context, speculate_max));
    ready_rx.await.context("engine failed before it was ready")??;
    let router = cuteafd_api::native_v41::router_for_model(queue, limits, stats, Duration::from_secs(25),
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
        schedule(engine, &loaded, &tokenizer, eos, &mut receive, transports, runtime, &stats, speculate_max)
    });
    if let Some(ready) = ready.take() {
        let _ = ready.send(result.as_ref().map(|_| ()).map_err(|e| anyhow::anyhow!("{e:#}")));
    }
    result
}

/// One admitted request: its placement, stream state and next input token.
struct Active<'a> {
    job: NativeRequest,
    /// Grammar for structured output and tool calls.
    constraint: Option<crate::v41_native_serve::constraints::State<'a>>,
    placement: Placement,
    capacity: usize,
    next: u32,
    decoder: cuteafd_loader::StreamingTokenDecoder,
    generated: usize,
    buffered: usize,
    started: Instant,
}

impl Active<'_> {
    /// Masked selection for the row that produces this sequence's next token.
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

/// Drafts after every active sequence's next token, verifies `[next, drafts]`
/// in one step and accepts each sequence's longest matching prefix plus the
/// verifier's own token after it. Returns which requests finished.
#[allow(clippy::too_many_arguments)]
fn speculative_step(
    engine: &super::engine::Engine<'_>,
    loaded: &super::Loaded,
    active: &mut [Active<'_>],
    block: usize,
    hidden: usize,
    vocab: usize,
    eos: u32,
    transports: &mut [cuteafd_transport::v41_expert::V41Tp4Roce],
    runtime: &tokio::runtime::Runtime,
) -> Result<Vec<bool>> {
    let noise = engine.cfg.dspark_noise_token_id as u32;
    let inputs: Vec<u32> = active.iter()
        .flat_map(|a| std::iter::once(a.next).chain(std::iter::repeat_n(noise, block - 1))).collect();
    let requests: Vec<super::engine::DraftRequest<'_>> = active.iter()
        .map(|a| super::engine::DraftRequest { placement: &a.placement, token: a.next }).collect();
    let drafts = engine.draft(&requests, &embed_rows(&loaded.catalog, &inputs, hidden)?)?;
    // Verify no more rows than the request may still produce or hold.
    let sequences: Vec<Vec<u32>> = active.iter().zip(&drafts).map(|(a, draft)| {
        let room = (a.job.max_tokens - a.generated).min(a.capacity - a.placement.len - 1);
        std::iter::once(a.next).chain(draft.iter().copied().take(room.saturating_sub(1).min(block))).collect()
    }).collect();
    let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
    let tokens: Vec<u32> = sequences.iter().flatten().copied().collect();
    let embed = embed_rows(&loaded.catalog, &tokens, hidden)?;
    let mut rows: Vec<(&mut Placement, &[u32])> = active.iter_mut().zip(&sequences)
        .map(|(a, tokens)| (&mut a.placement, tokens.as_slice())).collect();
    let logits = engine.verify(&mut rows, &embed, transports, runtime)?;
    let mut offset = 0;
    Ok(active.iter_mut().zip(&sequences).zip(starts).map(|((request, rows), start)| {
        let mut finished = false;
        for (j, _) in rows.iter().enumerate() {
            // Rows 0..=j are committed; the token row j produces is next.
            request.placement.len = start + j + 1;
            let result = request.select(&logits[(offset + j) * vocab..][..vocab])
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
    transports: &mut [cuteafd_transport::v41_expert::V41Tp4Roce],
    runtime: &tokio::runtime::Runtime,
    stats: &Mutex<serde_json::Value>,
    speculate_max: usize,
) -> Result<()> {
    let mut allocator = PoolAllocator::new(engine.shape);
    let mut grammars = crate::v41_native_serve::constraints::Compiler::new(
        &loaded.library, loaded.snapshot.join("tokenizer.json"));
    let mut active: Vec<Active<'_>> = Vec::new();
    let (mut requests, mut generated_total) = (0u64, 0u64);
    let hidden = engine.cfg.dim;
    let vocab = engine.cfg.vocab_size;
    let chunk_limit = engine.prefill_capacity().min(engine.max_context);
    loop {
        // Admit while sequence slots and decode rows remain.
        while allocator.free_states() > 0 && active.len() < engine.decode_rows {
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
            let mut placement = match allocator.admit(capacity) {
                Ok(placement) => placement,
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
                let embed = embed_rows(&loaded.catalog, &tokens, hidden)?;
                let mut logits = Vec::new();
                let started = Instant::now();
                *engine.profile.borrow_mut() = super::engine::Profile::default();
                // Equal chunks, so no lane runs a tiny tail.
                let chunks = tokens.len().div_ceil(chunk_limit);
                let limit = tokens.len().div_ceil(chunks);
                for (index, (chunk, rows)) in tokens.chunks(limit).zip(embed.chunks(limit * hidden * 2)).enumerate() {
                    let logit_rows = usize::from(index + 1 == chunks);
                    logits = engine.prefill(&mut placement, chunk, rows, transports, runtime, logit_rows, None)?;
                }
                tracing::debug!(tokens = tokens.len(), elapsed_ms = started.elapsed().as_millis() as u64,
                    phases = %engine.profile.borrow().report(), "prefill");
                let last = logits.len() / vocab - 1;
                let mut request = Active {
                    decoder: cuteafd_loader::streaming_token_decoder(&loaded.snapshot, false)?,
                    job, constraint, placement: placement.clone(), capacity, next: 0, generated: 0, buffered: 0,
                    started: Instant::now(),
                };
                request.next = request.select(&logits[last * vocab..])?;
                Ok(request)
            })();
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
        if active.is_empty() {
            continue;
        }
        // One decode step over every active sequence; with the drafter and a
        // small batch, each sequence verifies its next token plus a draft.
        let block = engine.draft_block();
        let speculate = block > 0 && active.len() <= speculate_max
            && active.len() * (block + 1) <= engine.decode_rows;
        let step = if speculate {
            speculative_step(engine, loaded, &mut active, block, hidden, vocab, eos, transports, runtime)
        } else {
            let tokens: Vec<u32> = active.iter().map(|a| a.next).collect();
            let embed = embed_rows(&loaded.catalog, &tokens, hidden)?;
            let mut rows: Vec<(&mut Placement, u32)> = active.iter_mut().map(|a| (&mut a.placement, a.next)).collect();
            engine.decode(&mut rows, &embed, &mut transports[0], runtime).map(|logits| {
                active.iter_mut().enumerate().map(|(row, request)| {
                    let logits_row = &logits[row * vocab..][..vocab];
                    request.select(logits_row).and_then(|token| request.emit(token, eos)).unwrap_or(true)
                }).collect::<Vec<bool>>()
            })
        };
        let finished = match step {
            Ok(finished) => finished,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                for request in active.drain(..) {
                    let _ = request.job.events.blocking_send(Err(NativeFailure::Worker(format!("{error:#}"))));
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
            *stats = serde_json::json!({"requests": requests, "generated_tokens": generated_total, "active": active.len()});
        }
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
