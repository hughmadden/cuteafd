//! The runner end to end against a fake engine behind the real OpenAI router:
//! a share run over loopback completes, exports render, and other clients get
//! 503 + Retry-After while it runs.
mod common;
use common::fake_engine;
use cuteafd_api::openai::{router_with_console, ConsoleHub, NativeLimits, NativeRequest};
use cuteafd_bench::report::RunStatus;
use cuteafd_bench::store::Store;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

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
        // The lock lifts once the progress ticker has joined, shortly after the stored status turns done.
        let lifted = (0..50).any(|_| {
            let status: Value = agent.get(&format!("{base3}/v1/bench/status")).call().unwrap().into_json().unwrap();
            status["active"].is_null() || {
                std::thread::sleep(Duration::from_millis(100));
                false
            }
        });
        assert!(lifted, "the run stayed active");
        let runs: Value = agent.get(&format!("{base3}/v1/bench/runs")).call().unwrap().into_json().unwrap();
        assert_eq!(runs["runs"].as_array().unwrap().len(), 1);
    }).await.unwrap();
}
