//! Shared fixtures for the bench integration tests: a fake engine behind the
//! real OpenAI router that streams 2 ms tokens and answers benchmark probes
//! with fake rows.
use cuteafd_api::openai::{InferenceChunk, InferenceFinishReason, NativeRequest, PromptUsage};
use std::time::Duration;
use tokio::sync::mpsc;

/// Count the same character-tokenized, thinking-off single-user template as the fake
/// worker. Use the checked-in renderer fixture rather than guessing its overhead.
pub fn with_fake_token_count(router: axum::Router) -> axum::Router {
    async fn count(axum::Json(body): axum::Json<serde_json::Value>) -> axum::Json<serde_json::Value> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../cuteafd-api/tests/fixtures/v41-encoding/deepseek-v4.1-flash.json")).unwrap();
        let case = fixture["cases"].as_array().unwrap().iter().find(|case| {
            case["request"]["messages"].as_array().is_some_and(|messages| messages.len() == 1
                && messages[0]["role"] == "user" && messages[0]["content"] == "hello")
                && case["request"]["thinking"]["type"] == "disabled"
        }).expect("single-user thinking-off template fixture");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["thinking"]["type"], "disabled");
        let overhead = case["prompt"].as_str().unwrap().chars().count() - "hello".chars().count();
        axum::Json(serde_json::json!({"input_tokens": overhead + body["messages"][0]["content"].as_str().unwrap().chars().count()}))
    }
    axum::Router::new().route("/v1/messages/count_tokens", axum::routing::post(count)).fallback_service(router)
}

/// A worker that streams 2 ms tokens and honours probes with fake rows.
pub fn fake_engine(mut receive: mpsc::Receiver<NativeRequest>) {
    cuteafd_bench::context::set_resolved("prefill-units", "2752");
    std::thread::spawn(move || {
        while let Some(job) = receive.blocking_recv() {
            std::thread::spawn(move || {
                let ids: Vec<u32> = job.probe.as_ref().and_then(|p| p.spec.prompt_ids.clone())
                    .unwrap_or_else(|| job.prompt.chars().map(u32::from).collect());
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
