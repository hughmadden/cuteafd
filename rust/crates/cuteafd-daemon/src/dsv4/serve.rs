//! OpenAI-compatible serving for DeepSeek V4: one request at a time through
//! the prefill/decode engine (the correctness baseline batching builds on).
use super::{embed_rows, with_engine, EngineArgs};
use anyhow::{ensure, Context, Result};
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
    let max_context = args.max_context as usize;
    let worker = tokio::task::spawn_blocking(move || serve_loop(engine_args, receive, ready_tx, worker_stats, max_context));
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
    let result = with_engine(&loaded, &args, |engine, transport, runtime| {
        ensure!(max_context <= engine.max_context, "--max-context {max_context} exceeds the programs' {}", engine.max_context);
        if let Some(ready) = ready.take() {
            let _ = ready.send(Ok(()));
        }
        let (mut requests, mut generated) = (0u64, 0u64);
        while let Some(job) = receive.blocking_recv() {
            let started = Instant::now();
            let events = job.events.clone();
            let outcome = serve_one(engine, &loaded, &tokenizer, eos, job, transport, runtime);
            match outcome {
                Ok(tokens) => {
                    requests += 1;
                    generated += tokens as u64;
                    let seconds = started.elapsed().as_secs_f64();
                    tracing::info!(tokens, seconds, tok_s = tokens as f64 / seconds, "request complete");
                    if let Ok(mut stats) = stats.lock() {
                        *stats = serde_json::json!({"requests": requests, "generated_tokens": generated});
                    }
                }
                Err(error) => {
                    tracing::warn!("request failed: {error:#}");
                    let _ = events.blocking_send(Err(NativeFailure::Worker(format!("{error:#}"))));
                }
            }
        }
        Ok(())
    });
    if let Some(ready) = ready.take() {
        let _ = ready.send(result.as_ref().map(|_| ()).map_err(|e| anyhow::anyhow!("{e:#}")));
    }
    result
}

fn serve_one(
    engine: &super::engine::Engine<'_>,
    loaded: &super::Loaded,
    tokenizer: &cuteafd_loader::LoadedTokenizer,
    eos: u32,
    job: NativeRequest,
    transport: &mut cuteafd_transport::v41_expert::V41Tp4Roce,
    runtime: &tokio::runtime::Runtime,
) -> Result<usize> {
    let send = |chunk: InferenceChunk| job.events.blocking_send(Ok(chunk)).map_err(|_| anyhow::anyhow!("client went away"));
    if job.constraint.is_some() || !job.images.is_empty() {
        let _ = job.events.blocking_send(Err(NativeFailure::BadRequest(
            "constrained decoding and images are not available for DeepSeek V4 yet".into())));
        return Ok(0);
    }
    let tokens = tokenizer.encode_text(&job.prompt, false)?.token_ids;
    let limit = engine.prefill_rows.min(engine.max_context);
    if tokens.is_empty() || tokens.len() > limit {
        let _ = job.events.blocking_send(Err(NativeFailure::BadRequest(
            format!("prompt of {} tokens is outside 1..={limit} for single-chunk prefill", tokens.len()))));
        return Ok(0);
    }
    send(InferenceChunk::Ready {
        system_fingerprint: None,
        prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: 0 },
    })?;
    let hidden = engine.cfg.dim;
    let vocab = engine.cfg.vocab_size;
    let capacity = (tokens.len() + job.max_tokens).min(engine.max_context);
    let mut sequence = engine.sequence(capacity)?;
    let embed = embed_rows(&loaded.catalog, &tokens, hidden)?;
    let logits = engine.prefill(&mut sequence, &tokens, &embed, transport, runtime, |_, _| Ok(()))?;
    let mut row = logits[(tokens.len() - 1) * vocab..].to_vec();
    let mut decoder = cuteafd_loader::streaming_token_decoder(&loaded.snapshot, false)?;
    let (mut generated, mut buffered) = (0usize, 0usize);
    let finish = loop {
        let position = sequence.tokens.len() as u64;
        let token = job.sampling.select_token(&row, None, position)
            .map_err(|e| anyhow::anyhow!("sampling: {e:?}"))? as u32;
        generated += 1;
        buffered += 1;
        if token != eos {
            if let Some(content) = decoder.step(token)? {
                send(InferenceChunk::Text { content, content_tokens: buffered })?;
                buffered = 0;
            }
        }
        let finish = if token == eos {
            Some(InferenceFinishReason::Stop)
        } else if generated >= job.max_tokens || sequence.tokens.len() + 1 >= capacity {
            Some(InferenceFinishReason::Length)
        } else {
            None
        };
        if let Some(finish) = finish {
            let content = decoder.finish()?.unwrap_or_default();
            if !content.is_empty() || buffered > 0 {
                send(InferenceChunk::Text { content, content_tokens: buffered })?;
            }
            break finish;
        }
        let embed = embed_rows(&loaded.catalog, &[token], hidden)?;
        row = engine.decode(&mut sequence, token, &embed, transport, runtime)?;
    };
    send(InferenceChunk::Finish { finish_reason: finish })?;
    Ok(generated)
}

#[cfg(test)]
mod tests {
    #[test]
    fn model_id_from_snapshot_path() {
        let path = std::path::Path::new("/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/abc");
        assert_eq!(super::model_id(path).as_deref(), Some("deepseek-ai/DeepSeek-V4-Flash-0731"));
    }
}
