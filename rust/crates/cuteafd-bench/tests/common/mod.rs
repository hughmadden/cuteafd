//! Shared fixtures for the bench integration tests: a fake engine behind the
//! real OpenAI router that streams 2 ms tokens and answers benchmark probes
//! with fake rows.
use cuteafd_api::openai::{InferenceChunk, InferenceFinishReason, NativeRequest, PromptUsage};
use std::time::Duration;
use tokio::sync::mpsc;

/// A worker that streams 2 ms tokens and honours probes with fake rows.
pub fn fake_engine(mut receive: mpsc::Receiver<NativeRequest>) {
    std::thread::spawn(move || {
        while let Some(job) = receive.blocking_recv() {
            std::thread::spawn(move || {
                let ids: Vec<u32> = job.probe.as_ref().and_then(|p| p.spec.prompt_ids.clone())
                    .unwrap_or_else(|| job.prompt.bytes().map(u32::from).collect());
                if let Some(probe) = &job.probe {
                    probe.admitted("fake", &ids, 0);
                }
                let _ = job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: ids.len(), prompt_cache_hit_tokens: 0 } }));
                if let Some(probe) = &job.probe {
                    let logits: Vec<f32> = (0..64).map(|v| ((v * 31 + ids.len()) % 17) as f32).collect();
                    if let Some(from) = probe.scoring() {
                        for p in from..ids.len() {
                            probe.row(p, &logits);
                        }
                        let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length }));
                        return;
                    }
                    if probe.spec.record_first {
                        probe.row(ids.len(), &logits);
                    }
                }
                for i in 0..job.max_tokens.min(48) {
                    std::thread::sleep(Duration::from_millis(2));
                    if let Some(probe) = &job.probe {
                        probe.token(7 + i as u32);
                    }
                    if job.events.send(Ok(InferenceChunk::Text { content: "a ".into(), content_tokens: 1 })).is_err() {
                        return;
                    }
                }
                let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length }));
            });
        }
    });
}
