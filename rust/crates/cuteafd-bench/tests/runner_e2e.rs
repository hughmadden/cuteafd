//! The runner end to end against a fake engine behind the real OpenAI router:
//! a share run over loopback completes, exports render, and other clients get
//! 503 + Retry-After while it runs.
use cuteafd_api::openai::{router_with_console, ConsoleHub, InferenceChunk, InferenceFinishReason, NativeLimits,
    NativeRequest, PromptUsage};
use cuteafd_bench::report::RunStatus;
use cuteafd_bench::store::Store;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

/// A worker that streams 2 ms tokens and honours probes with fake rows.
fn fake_engine(mut receive: mpsc::Receiver<NativeRequest>) {
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn share_run_completes_and_locks_out_other_clients() {
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    fake_engine(receive);
    let router = router_with_console(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        Duration::from_secs(5), ConsoleHub::disabled());
    let dir = tempfile::tempdir().unwrap();
    let bench = cuteafd_bench::Bench::new(Store::open(dir.path()).unwrap());
    let app = cuteafd_bench::http::mount(router, bench.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    cuteafd_bench::ready(&listener);
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()).await.unwrap();
    });
    let base2 = base.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        let agent = ureq::agent();
        let started: Value = agent.post(&format!("{base2}/v1/bench/runs")).send_json(json!({"profile": "share"}))
            .unwrap().into_json().unwrap();
        let id = started["id"].as_str().unwrap().to_string();
        // A second run is refused while one is active.
        let busy = agent.post(&format!("{base2}/v1/bench/runs")).send_json(json!({"profile": "share"}));
        assert!(matches!(busy, Err(ureq::Error::Status(409, _))));
        // Other clients are locked out with Retry-After.
        let chat = agent.post(&format!("{base2}/v1/chat/completions")).send_json(json!({
            "model": "deepseek-ai/DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": "hi"}]}));
        match chat {
            Err(ureq::Error::Status(503, response)) => assert!(response.header("retry-after").is_some()),
            other => panic!("expected 503, got {:?}", other.map(|r| r.status())),
        }
        // Status says what runs.
        let status: Value = agent.get(&format!("{base2}/v1/bench/status")).call().unwrap().into_json().unwrap();
        assert_eq!(status["active"]["id"], id.as_str());
        for _ in 0..600 {
            let report: Value = agent.get(&format!("{base2}/v1/bench/runs/{id}")).call().unwrap().into_json().unwrap();
            if !matches!(report["status"].as_str(), Some("running")) {
                return (id, report);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("the run did not finish");
    }).await.unwrap();
    let (id, report) = outcome;
    assert_eq!(report["status"], "done", "{report:#}");
    let report: cuteafd_bench::report::Report = serde_json::from_value(report).unwrap();
    assert_eq!(report.status, RunStatus::Done);
    let baseline = report.baseline.as_ref().expect("baseline");
    assert_eq!(baseline.card.decode.len(), 3);
    assert!(baseline.card.decode.iter().all(|d| d.tok_s > 0.0), "{:?}", baseline.card.decode);
    assert!(baseline.card.prefill.as_ref().is_some_and(|p| p.prompt_tokens > 1000));
    assert_eq!(baseline.quality.checks.len(), 5, "{:?}", baseline.quality.checks);
    // Exports of the stored run, and the lock lifted.
    let base3 = base.clone();
    tokio::task::spawn_blocking(move || {
        let agent = ureq::agent();
        for file in ["report.svg", "card.svg", "panel-baseline.svg", "report.json", "card.png"] {
            let response = agent.get(&format!("{base3}/v1/bench/runs/{id}/{file}")).call().unwrap();
            assert_eq!(response.status(), 200, "{file}");
        }
        let status: Value = agent.get(&format!("{base3}/v1/bench/status")).call().unwrap().into_json().unwrap();
        assert!(status["active"].is_null());
        let runs: Value = agent.get(&format!("{base3}/v1/bench/runs")).call().unwrap().into_json().unwrap();
        assert_eq!(runs["runs"].as_array().unwrap().len(), 1);
    }).await.unwrap();
}
