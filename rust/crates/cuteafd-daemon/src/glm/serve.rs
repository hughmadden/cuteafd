//! OpenAI-compatible API over the GLM engine: continuous batching with one
//! prefill per admitted request and one decode-shaped step for every active
//! sequence.
use super::engine::{GlmEngine, GlmPlacement, PageAllocator, DECODE_ROWS};
use super::{embed_rows, open, Opened};
use anyhow::{Context, Result};
use cuteafd_api::native_v41::glm::GlmEncoding;
use cuteafd_api::native_v41::{
    ConsoleHub, InferenceChunk, InferenceFinishReason, ModelEncoding, ModelProfile, NativeFailure, NativeLimits, NativeRequest,
    PromptUsage,
};
use cuteafd_transport::v41_expert::V41Tp4Roce;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[derive(Debug, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub engine: super::EngineArgs,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    #[arg(long, default_value_t = 4096)]
    pub max_output: u32,
    /// Sequences decoding at once (at most the decode programs' 64 rows).
    #[arg(long, default_value_t = 16)]
    pub max_sequences: usize,
    /// Public model id; defaults to the snapshot's Hugging Face id.
    #[arg(long)]
    pub model_id: Option<String>,
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
    let encoding = GlmEncoding::from_snapshot(&snapshot)?;
    let profile = ModelProfile::new(
        args.model_id.clone().or_else(|| model_id(&snapshot)).context("model id")?,
        ModelEncoding::Glm(Arc::new(encoding)),
    );
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let (engine_args, worker_stats, max_sequences) = (args.engine.clone(), stats.clone(), args.max_sequences);
    let worker = tokio::task::spawn_blocking(move || serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences));
    ready_rx.await.context("engine failed before it was ready")??;
    let router = cuteafd_api::native_v41::router_for_model(queue, limits, stats, Duration::from_secs(25),
        ConsoleHub::disabled(), profile.clone());
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    tracing::info!(listen = %args.listen, model = %profile.id, "GLM API is ready");
    tokio::select! {
        served = axum::serve(listener, router) => served?,
        finished = worker => finished??,
    }
    Ok(())
}

fn serve_loop(args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<()>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize)
    -> Result<()> {
    let opened = match open(&args) {
        Ok(opened) => opened,
        Err(error) => {
            let _ = ready.send(Err(anyhow::anyhow!("{error:#}")));
            return Ok(());
        }
    };
    let result = opened.with_engine(&args, |engine, transport, runtime| {
        let transport = transport.context("serve-glm needs --peers for the routed experts")?;
        let _ = ready.send(Ok(()));
        schedule(engine, &opened, &mut receive, transport, runtime, &stats, max_sequences.min(DECODE_ROWS))
    });
    result
}

struct Active<'a> {
    job: NativeRequest,
    constraint: Option<crate::v41_native_serve::constraints::State<'a>>,
    placement: GlmPlacement,
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

fn schedule(engine: &GlmEngine<'_>, opened: &Opened, receive: &mut mpsc::Receiver<NativeRequest>,
    transport: &mut V41Tp4Roce, runtime: &tokio::runtime::Runtime, stats: &Mutex<serde_json::Value>,
    max_sequences: usize) -> Result<()> {
    let mut allocator = PageAllocator::new(engine.pages);
    let mut grammars = crate::v41_native_serve::constraints::Compiler::with_vocab(
        &opened.library, opened.snapshot.join("tokenizer.json"), engine.cfg.vocab_size, engine.cfg.eos_tokens.clone());
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(&opened.snapshot)?;
    let mut active: Vec<Active<'_>> = Vec::new();
    let (mut requests, mut generated_total) = (0u64, 0u64);
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
            let admitted = (|| -> Result<Active<'_>> {
                let _ = job.events.blocking_send(Ok(InferenceChunk::Ready {
                    system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: 0 },
                }));
                let started = Instant::now();
                let mut logits = None;
                for chunk in tokens.chunks(engine.prefill_rows) {
                    let embed = embed_rows(&opened.catalog, chunk, hidden)?;
                    logits = engine.prefill(&mut placement, &embed, Some((&mut *transport, runtime)), None)?;
                }
                tracing::debug!(tokens = tokens.len(), elapsed_ms = started.elapsed().as_millis() as u64, "prefill");
                let mut request = Active {
                    decoder: cuteafd_loader::streaming_token_decoder(&opened.snapshot, false)?,
                    job, constraint, placement: placement.clone(), capacity, next: 0, generated: 0, buffered: 0,
                    started: Instant::now(),
                };
                request.next = request.select(&logits.context("prefill produced no logits")?)?;
                Ok(request)
            })();
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
        if active.is_empty() {
            continue;
        }
        let tokens: Vec<u32> = active.iter().map(|a| a.next).collect();
        let embed = embed_rows(&opened.catalog, &tokens, hidden)?;
        let mut rows: Vec<(&mut GlmPlacement, usize)> = active.iter_mut().map(|a| (&mut a.placement, 1)).collect();
        let step = engine.verify(&mut rows, &embed, Some((&mut *transport, runtime)), None)
            .and_then(|logits| logits.context("decode needs every layer"));
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
        let finished: Vec<bool> = active.iter_mut().enumerate().map(|(row, request)| {
            request.select(&logits[row * vocab..][..vocab]).and_then(|token| request.emit(token)).unwrap_or(true)
        }).collect();
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
                active = active.len(), gpu_wait_s = phases[0], experts_s = phases[1], head_s = phases[2],
                "request complete");
            allocator.release(request.placement);
        }
        if let Ok(mut stats) = stats.lock() {
            *stats = serde_json::json!({"requests": requests, "generated_tokens": generated_total, "active": active.len()});
        }
    }
}
